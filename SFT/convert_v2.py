#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert agent rollout JSONL (new format) to tokenized SFT data for Emu3.5.

Layout produced (per sample):
    <BOS> {system + first-user prompt verbatim}
    <BSS> <BoG> think <EoG>
          <tool_call> ... </tool_call>
    <ESS>
    <tool_response> ... [external img → 【图片N】<BoI>...<EoI>] ... </tool_response>
    <BSS> <BoG> think <EoG>
          <tool_call> ... </tool_call>
    <ESS>
    ... (more search rounds)
    <BSS> <BoC> draw-prompt with 【图片N】refs <EoC>
          <BoI>...generated image tokens...<EoI>
    <ESS>
    <EOS>

Conventions:
  * `[external image N]url[/external image N]` inside <tool_response> is the
    only image-search image carrier; each is VQ-tokenized and labeled 【图片N】.
  * `draw` is NOT a tool: its `prompt` becomes the <BoC>...<EoC> caption of
    the final assistant turn, and its saved_path's basename is looked up in
    `tool_resp_dir` and tokenized as the generated image. The draw turn's own
    pre-think is discarded (per user request).
  * Conversation truncates at the first successful draw image.
  * Any image (external download OR generated) that fails to load → drop the
    whole sample.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Emu3.5 uses imports such as ``from src.utils...``.  The directory that
# contains that top-level ``src`` package is Emu3.5, not PROJECT_ROOT.
# Add it here instead of relying on the caller's working directory or
# PYTHONPATH so convert_v2.py also works when invoked directly.
EMU35_ROOT = os.path.join(PROJECT_ROOT, "Emu3.5")
if EMU35_ROOT not in sys.path:
    sys.path.insert(0, EMU35_ROOT)


DEFAULT_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "Emu3.5", "src", "tokenizer_emu3_ibq")
DEFAULT_TOOL_RESP_DIR = os.path.join(PROJECT_ROOT, "Data", "SFT", "image")

IGNORE_INDEX = -100


# ── Special tokens ─────────────────────────────────────────────────────────
SPECIAL = dict(
    bos="<|extra_203|>",
    eos="<|extra_204|>",
    pad="<|endoftext|>",
    eol="<|extra_200|>",
    eof="<|extra_201|>",
    tms="<|extra_202|>",
    img="<|image token|>",
    boi="<|image start|>",
    eoi="<|image end|>",
    bss="<|extra_100|>",
    ess="<|extra_101|>",
    bog="<|extra_60|>",   # before tool call
    eog="<|extra_61|>",
    boc="<|extra_50|>",   # before image generation
    eoc="<|extra_51|>",
)


@dataclass
class ImageConfig:
    image_area: int = 1048576


@dataclass
class Stats:
    total: int = 0
    written: int = 0
    skipped: int = 0
    skipped_no_draw: int = 0
    skipped_image_fail: int = 0
    # Failed-draw retries: trajectories where the assistant called `draw` /
    # `text2image` before the terminal draw turn. Earlier drafts of this
    # script kept them as plain tool_calls so the model "learns the retry
    # trajectory", but for the native-generation agent these teach the wrong
    # behavior (the model should never call `draw` — it generates the image
    # itself via <BoI>...<EoI>). We now drop the whole sample.
    skipped_failed_draw: int = 0
    images_encoded: int = 0
    external_images: int = 0
    generated_images: int = 0


# ── JSONL I/O ──────────────────────────────────────────────────────────────
def count_jsonl_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


# ── Tokenizer ──────────────────────────────────────────────────────────────
def load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=os.path.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    for key, val in SPECIAL.items():
        setattr(tokenizer, f"{key}_token", val)
    return tokenizer


def encode_text(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False, allowed_special="all")


# ── Image handling ─────────────────────────────────────────────────────────
class ImageFetchError(Exception):
    """Raised when an image (external URL or generated file) can't be loaded.

    The caller catches this at the sample level and drops the whole sample.
    """


class ImageEncoder:
    def __init__(
        self,
        tokenizer,
        vq_path: str,
        vq_type: str,
        vq_device: str,
        image_area: int,
        tool_resp_dir: str,
        download_retries: int = 3,
        download_timeout: int = 45,
        download_workers: int = 4,
    ):
        self.tokenizer = tokenizer
        self.cfg = ImageConfig(image_area=image_area)
        self.tool_resp_dir = tool_resp_dir
        self.download_retries = download_retries
        self.download_timeout = download_timeout
        self.download_workers = max(1, download_workers)

        from src.utils.input_utils import build_image
        from src.vision_tokenizer import build_vision_tokenizer

        self.vq_model = build_vision_tokenizer(vq_type, vq_path, device=vq_device)
        self.vq_model.eval()
        self.build_image_fn = build_image

        # Per-sample cache: url/saved_path -> PIL.Image (or Exception).
        # Cleared by the caller via reset_cache() at the start of each sample.
        self._pil_cache: Dict[str, Any] = {}

    # ── opening logic ─────────────────────────────────────────────────────
    def _download(self, url: str) -> Image.Image:
        """Try `download_retries` attempts via urllib (with headers), then
        one final attempt via `requests`. Raise ImageFetchError on failure."""
        import time
        import urllib.request
        from urllib.parse import urlparse

        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": origin,
            "Connection": "close",
        }
        last_exc: Optional[Exception] = None
        for attempt in range(self.download_retries):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.download_timeout) as resp:
                    data = resp.read()
                return Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as exc:
                last_exc = exc
                time.sleep(0.5 * (attempt + 1))
        try:
            import requests

            resp = requests.get(
                url, headers=headers, timeout=self.download_timeout, allow_redirects=True
            )
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as exc:
            last_exc = exc
        raise ImageFetchError(f"download failed after {self.download_retries}+1 attempts: {url[:120]} :: {last_exc}")

    def _open_external(self, ref: str) -> Image.Image:
        if not ref:
            raise ImageFetchError("empty image ref")
        if ref.startswith("data:image"):
            _, payload = ref.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        if ref.startswith(("http://", "https://")):
            return self._download(ref)
        if os.path.isfile(ref):
            return Image.open(ref).convert("RGB")
        raise ImageFetchError(f"unsupported image ref: {ref[:120]}")

    def _open_generated(self, saved_path: str) -> Image.Image:
        """Generated images live at `tool_resp_dir/<basename(saved_path)>`."""
        if not saved_path:
            raise ImageFetchError("empty saved_path")
        path = os.path.join(self.tool_resp_dir, os.path.basename(saved_path))
        if not os.path.isfile(path):
            raise ImageFetchError(f"generated image not found: {path}")
        return Image.open(path).convert("RGB")

    # ── VQ encoding ───────────────────────────────────────────────────────
    def _encode_pil(self, img: Image.Image) -> str:
        return self.build_image_fn(img, self.cfg, self.tokenizer, self.vq_model)

    # ── Per-sample cache lifecycle ───────────────────────────────────────
    def reset_cache(self) -> None:
        """Call at the start of every sample. Guarantees no cross-sample
        leakage of downloaded images even though the encoder is reused.
        """
        self._pil_cache = {}

    def prefetch_externals(self, refs: List[str]) -> None:
        """Concurrently fetch a batch of external refs (URLs / data: / local
        paths) into the per-sample cache. Failures are stored in the cache
        as the Exception, so the later .encode_external() call will raise
        deterministically and the sample is dropped — same behavior as the
        single-threaded path, just faster.

        Threads only touch self._pil_cache via its setitem (atomic in
        CPython per-key), and each thread writes to its own key, so no lock
        is needed. PIL.Image.open() inside threads is fine; the GPU model
        is never touched from a worker thread.
        """
        # de-dup and skip what's already cached
        todo = [r for r in dict.fromkeys(refs) if r and r not in self._pil_cache]
        if not todo:
            return
        workers = min(self.download_workers, len(todo))
        if workers <= 1:
            for r in todo:
                try:
                    self._pil_cache[r] = self._open_external(r)
                except Exception as exc:
                    self._pil_cache[r] = exc
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._open_external, r): r for r in todo}
            for fut in as_completed(futures):
                r = futures[fut]
                try:
                    self._pil_cache[r] = fut.result()
                except Exception as exc:
                    self._pil_cache[r] = exc

    def _cached_external(self, ref: str) -> Image.Image:
        if ref in self._pil_cache:
            obj = self._pil_cache[ref]
            if isinstance(obj, Exception):
                raise ImageFetchError(str(obj))
            return obj
        # Fallback: not prefetched (shouldn't happen on the new path).
        return self._open_external(ref)

    def encode_external(self, ref: str) -> str:
        return self._encode_pil(self._cached_external(ref))

    def encode_generated(self, saved_path: str) -> str:
        return self._encode_pil(self._open_generated(saved_path))


# ── Parsing helpers ────────────────────────────────────────────────────────
_TOOL_RESP_RE = re.compile(r"<tool_response>\s*([\s\S]*?)\s*</tool_response>")
_THINK_RE = re.compile(r"<think>\s*([\s\S]*?)\s*</think>")
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>")
_BOX_RE = re.compile(r"<\|box_start\|>([\s\S]*?)<\|box_end\|>")
_EXTERNAL_IMG_RE = re.compile(
    r"\[external image\s*(\d+)\]([\s\S]*?)\[/external image\s*\1\]",
    re.IGNORECASE,
)
_INTERNAL_IMG_RE = re.compile(
    r"\[internal image\s*\d+\]([\s\S]*?)\[/internal image\s*\d+\]"
)
_SAVED_PATH_FALLBACK_RE = re.compile(r"已保存到\s*([^\s<]+)")
_IMAGE_REF_RE = re.compile(r"\[IMAGE\s*(\d+)\]", re.IGNORECASE)


def _get_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _parse_assistant(text: str) -> Dict[str, Any]:
    """Pull <think>, <tool_call> JSON, and <|box_start|>...<|box_end|>."""
    out: Dict[str, Any] = {"think": None, "tool_call": None, "tool_call_raw": None, "box_answer": None}
    mt = _THINK_RE.search(text)
    if mt:
        out["think"] = mt.group(1).strip()
    mc = _TOOL_CALL_RE.search(text)
    if mc:
        inner = mc.group(1).strip()
        try:
            out["tool_call"] = json.loads(inner)
        except json.JSONDecodeError:
            out["tool_call_raw"] = inner
    mb = _BOX_RE.search(text)
    if mb:
        out["box_answer"] = mb.group(1).strip()
    return out


def _extract_saved_path(tool_resp_text: str) -> str:
    m = _INTERNAL_IMG_RE.search(tool_resp_text)
    if m:
        return m.group(1).strip()
    m = _SAVED_PATH_FALLBACK_RE.search(tool_resp_text)
    return m.group(1).strip() if m else ""


def _reformat_image_search_response(
    raw_text: str,
    image_encoder: ImageEncoder,
    url_to_label: Dict[str, int],
    stats: Stats,
) -> str:
    """Replace `[external image N]url[/external image N]` with
    `[IMAGE N] <VQ tokens>`. Records url → N in url_to_label so subsequent
    draw `[IMAGE k]` refs can be remapped. Raises ImageFetchError on failure.
    """

    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        url = match.group(2).strip()
        url_to_label[url] = n
        visual = image_encoder.encode_external(url)
        stats.images_encoded += 1
        stats.external_images += 1
        return f"[IMAGE {n}] {visual}"

    return _EXTERNAL_IMG_RE.sub(_replace, raw_text)


def _remap_draw_prompt(
    prompt: str, images: List[str], url_to_label: Dict[str, int]
) -> str:
    """In a draw call, `images` is the ordered list of reference URLs the
    teacher actually chose to use, and `[IMAGE k]` inside `prompt` is the
    1-indexed pointer into that list. We rewrite each `[IMAGE k]` to its
    global `[IMAGE N]` so it lines up with the labels we emitted in earlier
    `<tool_response>` blocks. References to images the teacher did NOT
    select are simply absent from `images`, so they never appear in the
    final BoC caption.
    """
    local_to_global: Dict[int, str] = {}
    for i, url in enumerate(images, start=1):
        n = url_to_label.get(url)
        if n is None:
            base = os.path.basename(url)
            for ref_url, ref_n in url_to_label.items():
                if os.path.basename(ref_url) == base:
                    n = ref_n
                    break
        local_to_global[i] = f"[IMAGE {n}]" if n is not None else f"[IMAGE ?{i}]"

    return _IMAGE_REF_RE.sub(
        lambda m: local_to_global.get(int(m.group(1)), f"[IMAGE ?{m.group(1)}]"),
        prompt,
    )


# ── Build segments ─────────────────────────────────────────────────────────
def _collect_external_refs(example: Dict[str, Any]) -> List[str]:
    """Walk an example once and return every external image URL/ref the
    sample will need. Used to prefetch concurrently before the (serial,
    GPU-bound) build_segments() pass. Does NOT include generated images
    (those live on local disk and open instantly).

    Truncates the scan at the LAST successful draw — same truncation point
    that build_segments() will use — so we don't waste downloads on
    tool_responses that are about to be dropped.
    """
    messages = example.get("messages", [])
    cut = _find_last_successful_draw(messages)
    if cut < 0:
        return []
    refs: List[str] = []
    for i, msg in enumerate(messages):
        if i > cut:
            break
        if msg.get("role") != "user":
            continue
        raw = _get_text(msg.get("content", ""))
        tr = _TOOL_RESP_RE.search(raw)
        if not tr:
            continue
        for m in _EXTERNAL_IMG_RE.finditer(tr.group(1)):
            refs.append(m.group(2).strip())
    return refs


# ── Build segments ─────────────────────────────────────────────────────────
def _is_successful_draw_response(tool_resp_text: str) -> bool:
    """A draw call is 'successful' iff the following <tool_response> carries
    a saved image path. The rollouts use two shapes:
      * `[internal image N]<path>[/internal image N]`  (canonical)
      * `已保存到 <path>` fallback
    Anything else — `[error] [Draw] ...`, HTTP 4xx/5xx, empty body, the
    truncation sentinel "number of llm calls exceeds the limit." — is a
    failed draw that the model should be allowed to retry. The old code
    locked onto the FIRST draw and broke on the next user msg, so a failed
    first draw poisoned the whole sample. We now scan ahead and only stop
    at the LAST successful draw.
    """
    if not tool_resp_text:
        return False
    return bool(
        _INTERNAL_IMG_RE.search(tool_resp_text)
        or _SAVED_PATH_FALLBACK_RE.search(tool_resp_text)
    )


def _find_last_successful_draw(messages: List[Dict[str, Any]]) -> int:
    """Return the index of the assistant message containing the LAST draw
    whose follow-up tool_response carries a saved image path. Returns -1 if
    no successful draw exists (caller drops the sample).

    Walks the conversation linearly. A draw is "the assistant message i
    whose tool_call.name ∈ {draw, text2image}", "success" is judged on
    messages[i+1] (must be a user with a tool_response containing an
    internal-image marker or 已保存到 path).
    """
    last_ok = -1
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        text = _get_text(m.get("content", ""))
        parsed = _parse_assistant(text)
        tc = parsed["tool_call"]
        if not (tc and tc.get("name") in ("draw", "text2image")):
            continue
        if i + 1 >= len(messages) or messages[i + 1].get("role") != "user":
            continue
        follow = _get_text(messages[i + 1].get("content", ""))
        tr = _TOOL_RESP_RE.search(follow)
        if tr and _is_successful_draw_response(tr.group(1)):
            last_ok = i
    return last_ok


# Each segment is (text, trainable).
def build_segments(
    example: Dict[str, Any],
    image_encoder: ImageEncoder,
    stats: Stats,
) -> Optional[List[Tuple[str, bool]]]:
    messages = example.get("messages", [])

    # ── Prefix: BOS + (system content) + "\n\n" + (first user content) ──
    system_text = ""
    first_user_text = ""
    for msg in messages:
        role = msg.get("role")
        if role == "system" and not system_text:
            system_text = _get_text(msg.get("content", "")).strip()
        elif role == "user" and not first_user_text:
            first_user_text = _get_text(msg.get("content", "")).strip()
            break

    prefix_text = SPECIAL["bos"] + system_text
    if system_text and first_user_text:
        prefix_text += "\n\n"
    prefix_text += first_user_text

    segments: List[Tuple[str, bool]] = [(prefix_text, False)]

    # ── Find the truncation point: the LAST successful draw. ──────────────
    # Earlier drafts truncated at the FIRST draw, which broke whenever the
    # model's initial draw attempt errored ("Missing required field", 403,
    # ReadTimeout, ...). Those samples were ~30% of the corpus. Now we
    # treat failed drafts as normal tool_call/tool_response pairs and only
    # stop at the draw whose response carries a saved path.
    target_draw_idx = _find_last_successful_draw(messages)
    if target_draw_idx < 0:
        stats.skipped_no_draw += 1
        return None

    # ── Walk the rest of the conversation ──
    url_to_label: Dict[str, int] = {}
    started = False                                  # skipped first user yet?
    saw_draw = False
    pending_draw_call: Optional[Dict[str, Any]] = None
    in_assistant = False                             # is BSS open?

    def _open_assistant():
        nonlocal in_assistant
        if not in_assistant:
            segments.append((SPECIAL["bss"], True))
            in_assistant = True

    def _close_assistant():
        nonlocal in_assistant
        if in_assistant:
            segments.append((SPECIAL["ess"], True))
            in_assistant = False

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "")
        if role == "system":
            continue
        if not started:
            if role == "user":
                started = True
                continue
            return None  # assistant before user

        # ── assistant ──
        if role == "assistant":
            text = _get_text(msg.get("content", ""))
            parsed = _parse_assistant(text)
            tc = parsed["tool_call"]
            # Only the FINAL successful draw is the terminal draw. Earlier
            # drafts kept failed-draw retries as ordinary tool_calls so the
            # model could "learn the retry trajectory", but the native-gen
            # agent must NEVER call a `draw` / `text2image` tool — it has to
            # emit <BoI>...<EoI> itself. Any non-terminal draw tool_call ⇒
            # this whole sample is poisoning the policy ⇒ drop it.
            is_terminal_draw = (msg_idx == target_draw_idx)

            if is_terminal_draw:
                # Defer; pre-draw <think> is intentionally discarded.
                pending_draw_call = tc
                continue

            # Failed / pre-terminal draw — drop the entire sample.
            if tc and tc.get("name") in ("draw", "text2image"):
                stats.skipped_failed_draw += 1
                return None

            # No-draw, no-tool-call assistant turn that has a box answer?
            # Sample never generates an image → drop.
            if parsed["box_answer"] and not saw_draw and tc is None and not parsed["tool_call_raw"]:
                return None

            _open_assistant()
            if parsed["think"]:
                segments.append(
                    (f"{SPECIAL['bog']} {parsed['think']} {SPECIAL['eog']}\n", True)
                )
            if tc is not None:
                segments.append(
                    (f"<tool_call>\n{json.dumps(tc, ensure_ascii=False)}\n</tool_call>", True)
                )
            elif parsed["tool_call_raw"]:
                segments.append(
                    (f"<tool_call>\n{parsed['tool_call_raw']}\n</tool_call>", True)
                )
            _close_assistant()

        # ── user (= tool_response) ──
        elif role == "user":
            raw_text = _get_text(msg.get("content", ""))
            tr_match = _TOOL_RESP_RE.search(raw_text)
            tr_text = tr_match.group(1) if tr_match else None

            if pending_draw_call is not None:
                # Last turn: BSS, BoC draw-prompt EoC, generated image, ESS, then stop.
                args = pending_draw_call.get("arguments", {}) or {}
                prompt = args.get("prompt", "") or ""
                images = args.get("images", []) or []
                if isinstance(images, str):
                    images = [images]
                mapped = _remap_draw_prompt(prompt, images, url_to_label)
                saved_path = _extract_saved_path(tr_text) if tr_text is not None else ""
                visual = image_encoder.encode_generated(saved_path)  # raises on failure
                stats.images_encoded += 1
                stats.generated_images += 1

                _open_assistant()
                segments.append((f"{SPECIAL['boc']} {mapped} {SPECIAL['eoc']}\n", True))
                segments.append((visual, True))
                _close_assistant()

                saw_draw = True
                pending_draw_call = None
                break

            if tr_text is None:
                # Plain user followup (rare): masked.
                segments.append((raw_text, False))
                continue

            reformatted = _reformat_image_search_response(
                tr_text, image_encoder, url_to_label, stats
            )
            segments.append(
                (f"<tool_response>\n{reformatted}\n</tool_response>", False)
            )

        else:
            segments.append((_get_text(msg.get("content", "")), False))

    if not saw_draw:
        stats.skipped_no_draw += 1
        return None

    _close_assistant()
    segments.append((SPECIAL["eos"], True))
    return segments


# ── Tokenize ───────────────────────────────────────────────────────────────
def tokenize_segments(
    segments: List[Tuple[str, bool]],
    tokenizer,
    max_length: int,
    truncate: str,
) -> Optional[Dict[str, Any]]:
    input_ids: List[int] = []
    labels: List[int] = []
    full_text = ""
    for text, trainable in segments:
        full_text += text
        ids = encode_text(tokenizer, text)
        input_ids.extend(ids)
        labels.extend(ids if trainable else [IGNORE_INDEX] * len(ids))

    if len(input_ids) > max_length:
        if truncate == "skip":
            return None
        if truncate == "right":
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        else:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]

    if all(l == IGNORE_INDEX for l in labels):
        return None

    return {
        "text": full_text,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert agent rollout JSONL (new format) to tokenized SFT data"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--vq-path", required=True)
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--vq-device", default="cuda:0")
    parser.add_argument("--image-area", type=int, default=1048576)
    parser.add_argument(
        "--tool-resp-dir",
        default=DEFAULT_TOOL_RESP_DIR,
    )
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--truncate", choices=["skip", "left", "right"], default="skip")
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--download-timeout", type=int, default=45)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Concurrent image downloads within a single sample. "
             "Cross-sample parallelism is via --num-shards (8 GPUs).",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    # ── Sharding (one process per GPU) ──
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parallel worker processes (typically = #GPUs).",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="0-based index of THIS worker. line_no % num_shards == shard_id are processed here.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Disable tqdm bar entirely. Use when stdout is a log file. "
             "Implied when stdout is not a TTY.",
    )
    parser.add_argument(
        "--progress-file",
        default="",
        help="Optional path. If set, periodically write {total, written, "
             "skipped, images_encoded, line_no, done} JSON for an external "
             "watcher to aggregate across shards.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=1.0,
        help="Seconds between --progress-file writes.",
    )
    args = parser.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        parser.error(f"shard-id {args.shard_id} out of range for num-shards {args.num_shards}")

    tokenizer = load_tokenizer(args.tokenizer_path)
    image_encoder = ImageEncoder(
        tokenizer=tokenizer,
        vq_path=args.vq_path,
        vq_type=args.vq_type,
        vq_device=args.vq_device,
        image_area=args.image_area,
        tool_resp_dir=args.tool_resp_dir,
        download_retries=args.download_retries,
        download_timeout=args.download_timeout,
        download_workers=args.download_workers,
    )

    stats = Stats()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    total_input = count_jsonl_lines(args.input)
    # Number of lines that fall to THIS shard (after the line_no % N == id filter
    # but before --limit). Used only to size tqdm.
    own_lines = sum(
        1 for n in range(1, total_input + 1) if (n - 1) % args.num_shards == args.shard_id
    )
    if args.limit:
        own_lines = min(own_lines, args.limit)

    desc = f"convert[{args.shard_id}/{args.num_shards}]"
    # Disable tqdm entirely when redirected to a log file. The previous
    # behavior dumped one '\r' rewrite per sample plus ANSI cursor-move
    # bytes (from position=) into the log, drowning out real errors.
    disable_bar = args.quiet_progress or (not sys.stdout.isatty())
    iterator = tqdm(
        iter_jsonl(args.input),
        total=own_lines,
        desc=desc,
        unit="sample",
        dynamic_ncols=True,
        position=args.shard_id,
        disable=disable_bar,
        mininterval=0.5,
    )

    progress_path = args.progress_file
    progress_tmp = progress_path + ".tmp" if progress_path else ""
    last_progress_ts = 0.0
    last_line_no = 0

    def _emit_progress(current_line_no: int, done: bool = False) -> None:
        """Atomically dump current shard counters for the watcher. Uses
        write+rename so the watcher never reads a torn JSON line. Throttled
        by args.progress_interval; `done=True` forces a flush.
        """
        nonlocal last_progress_ts
        if not progress_path:
            return
        now = time.time()
        if not done and (now - last_progress_ts) < args.progress_interval:
            return
        last_progress_ts = now
        payload = {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "own_lines": own_lines,
            "line_no": current_line_no,
            "total": stats.total,
            "written": stats.written,
            "skipped": stats.skipped,
            "skipped_no_draw": stats.skipped_no_draw,
            "skipped_image_fail": stats.skipped_image_fail,
            "skipped_failed_draw": stats.skipped_failed_draw,
            "images_encoded": stats.images_encoded,
            "external_images": stats.external_images,
            "generated_images": stats.generated_images,
            "done": done,
            "ts": now,
        }
        try:
            with open(progress_tmp, "w", encoding="utf-8") as pf:
                json.dump(payload, pf, ensure_ascii=False)
            os.replace(progress_tmp, progress_path)
        except OSError:
            # Watcher state is best-effort; never crash the worker on it.
            pass

    with open(args.output, "w", encoding="utf-8") as out:
        for line_no, example in iterator:
            last_line_no = line_no
            # Shard filter — strict, deterministic, no overlap.
            if (line_no - 1) % args.num_shards != args.shard_id:
                continue
            if args.limit and stats.total >= args.limit:
                break
            stats.total += 1

            # Per-sample cache reset BEFORE any download/encode work.
            # Guarantees zero cross-sample image leakage even though the
            # encoder object is reused.
            image_encoder.reset_cache()

            try:
                # Concurrently fetch every external image this sample needs,
                # THEN do the (single-threaded, GPU-bound) build_segments pass.
                refs = _collect_external_refs(example)
                if refs:
                    image_encoder.prefetch_externals(refs)

                segments = build_segments(example, image_encoder, stats)
                record = (
                    tokenize_segments(segments, tokenizer, args.max_length, args.truncate)
                    if segments is not None
                    else None
                )
            except ImageFetchError as exc:
                print(f"[skip] line {line_no}: {exc}", file=sys.stderr)
                stats.skipped += 1
                stats.skipped_image_fail += 1
                _emit_progress(line_no)
                continue
            except Exception as exc:
                print(f"[ERROR] line {line_no}: {exc}", file=sys.stderr)
                stats.skipped += 1
                _emit_progress(line_no)
                continue

            if record is None:
                stats.skipped += 1
                _emit_progress(line_no)
                continue

            record["meta"] = {
                "line_no": line_no,
                "question": example.get("question", ""),
                "rollout_id": example.get("rollout_id"),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.written += 1
            if not disable_bar:
                iterator.set_postfix(
                    w=stats.written, s=stats.skipped, img=stats.images_encoded
                )
            _emit_progress(line_no)

    # Final flush so the watcher sees the terminal counters.
    _emit_progress(last_line_no, done=True)

    print("=" * 60)
    print(f"  Shard:                  {args.shard_id}/{args.num_shards}")
    print(f"  Total:                  {stats.total}")
    print(f"  Written:                {stats.written}")
    print(f"  Skipped:                {stats.skipped}")
    print(f"     no-draw:             {stats.skipped_no_draw}")
    print(f"     failed-draw retry:   {stats.skipped_failed_draw}")
    print(f"     image-fetch-fail:    {stats.skipped_image_fail}")
    print(f"  Images encoded:         {stats.images_encoded}")
    print(f"     external (search):   {stats.external_images}")
    print(f"     generated (draw):    {stats.generated_images}")
    print("=" * 60)


if __name__ == "__main__":
    main()
