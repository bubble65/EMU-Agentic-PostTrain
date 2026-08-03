"""Emu3.5 agent inference driver.

Mirrors the layout produced by the SFT convert script at inference
time:

    <BOS>{system}\n\nUser: {question}
    <BSS><BoG>think<EoG>
    <tool_call>...</tool_call>
    <ESS>
    <tool_response>...[IMAGE N] <BoI>...<EoI> ...</tool_response>
    <BSS>... (more rounds) ...<ESS>
    <BSS><BoC>draw caption with [IMAGE k] refs<EoC>
          <BoI>H*W<IMG_TOKEN>v..v<EoI>
    <ESS><EOS>

Architecture:
  * The vLLM server (server.py) is a pure generator: prompt-in, tokens-out.
    It owns the text→image sampling-mode transition (auto via BoI stop).
  * This driver owns EVERY message-level concern:
        - System / user prompt construction
        - Tool dispatch (text_search / image_search via qwen_agent registry)
        - Building <tool_response> blocks with [IMAGE N] + vq-encoded
          reference images via the server's /encode_images endpoint
        - The global [IMAGE N] counter that the model's [IMAGE k] caption
          refs point back at
        - The "force draw on round N" safety net
  * Internal messages list uses 4 roles that map cleanly into the SFT layout:
        - "system"    : system text (emitted once after <BOS>)
        - "user"      : the FIRST user question (emitted directly after system)
        - "assistant" : a complete assistant span body (between <BSS> and <ESS>)
        - "tool"      : the tool_response body for the previous tool_call
    No other role is allowed; this matches the actual training distribution
    exactly. (No "USER:"/"ASSISTANT:" framing.)

The "force draw" mode satisfies the request "the third round must generate
the image". The training data only contains one final draw turn at the END
of the conversation, so we simulate that turn directly:

    user-question + N-1 search rounds + <BSS><BoC>{hint}<EoC><BoI>

and then ask the server to skip pass-1 (`force_image_first=True`) so it goes
straight into image sampling. After it stops on EoI we let it close out with
<ESS>.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from prompt import SYSTEM_PROMPT, USER_PREFIX  # noqa: E402
from template import SPECIAL, assemble_prompt  # noqa: E402
from tool_response_format import (  # noqa: E402
    reformat_tool_response,
    rewrite_think_to_bog_eog,
)

TOOL_DIR = Path(os.environ.get("EMU_AGENT_TOOL_DIR", str(THIS_DIR.parent / "DataRoller")))
if TOOL_DIR.exists() and str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

# Side-effect imports: register the two tools with qwen-agent's registry.
import tool_imagesearch  # noqa: F401,E402
import tool_textsearch  # noqa: F401,E402

from qwen_agent.tools import TOOL_REGISTRY  # noqa: E402


# ── Constants / env overrides ────────────────────────────────────────────
MAX_LLM_CALL_PER_RUN = int(os.environ.get("EMU_MAX_LLM_CALL_PER_RUN", "8"))
MAX_PROMPT_TOKENS = int(os.environ.get("EMU_MAX_PROMPT_TOKENS", "32768"))
EMU_SERVER_URL = os.environ.get("EMU_SERVER_URL", "http://127.0.0.1:23333")
EMU_ROOT = Path(os.environ.get("EMU_ROOT", str(THIS_DIR.parent / "Emu3.5")))
EMU_TOKENIZER_PATH = os.environ.get(
    "EMU_TOKENIZER_PATH",
    str(EMU_ROOT / "src" / "tokenizer_emu3_ibq"),
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>")
_BOC_RE = re.compile(r"<\|extra_50\|>\s*([\s\S]*?)\s*<\|extra_51\|>")
_BOI_BLOCK_RE = re.compile(r"<\|image start\|>[\s\S]*?<\|image end\|>")
_GENERATED_IMAGE_RE = re.compile(r"\[generated_image:\s*([^\]]+)\]")

_TOKENIZER_LOCK = threading.Lock()
_TOKENIZER = None


def _get_emu_tokenizer():
    """Lazy-load the text tokenizer (CPU) for prompt-length counting."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    with _TOKENIZER_LOCK:
        if _TOKENIZER is None:
            from transformers import AutoTokenizer

            _TOKENIZER = AutoTokenizer.from_pretrained(
                EMU_TOKENIZER_PATH,
                special_tokens_file=os.path.join(
                    EMU_TOKENIZER_PATH, "emu3_vision_tokens.txt"
                ),
                trust_remote_code=True,
            )
    return _TOKENIZER


def count_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    """Return the exact number of token ids assemble_prompt(messages) would
    feed to the model. Visual-token segments and special markers each count
    as 1 token, matching what the server actually receives."""
    prompt = assemble_prompt(messages)
    return len(_get_emu_tokenizer().encode(prompt, add_special_tokens=False))


# ── Tool dispatch ────────────────────────────────────────────────────────
class _ToolDispatcher:
    """Lazy-instantiate qwen-agent BaseTool classes once."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def call(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self._cache:
            cls = TOOL_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"unknown tool: {name}")
            self._cache[name] = cls()
        return self._cache[name].call(args)


# ── Server client ────────────────────────────────────────────────────────
class EmuServerClient:
    """Thin HTTP client for server.py."""

    def __init__(self, base_url: str, timeout: int = 1800):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def encode_images(self, refs: List[str]) -> Dict[str, Any]:
        if not refs:
            return {"tokens": [], "ok": [], "errors": []}
        r = requests.post(
            f"{self.base}/encode_images",
            json={"refs": refs},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def generate(self, prompt: str, **overrides) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"prompt": prompt}
        # Pass only non-None overrides so the server uses its own defaults.
        payload.update({k: v for k, v in overrides.items() if v is not None})
        r = requests.post(
            f"{self.base}/generate",
            json=payload,
            timeout=self.timeout,
        )
        if not r.ok:
            print(f"[emu-client] HTTP {r.status_code}: {r.text[:400]}")
        r.raise_for_status()
        return r.json()


# ── Per-sample state ─────────────────────────────────────────────────────
class _ImageLabelBook:
    """Tracks the global [IMAGE N] counter and remembers url ↔ label so
    later tool_calls can swap labels back to URLs."""

    def __init__(self):
        self.counter = 0
        self.label_to_url: Dict[str, str] = {}     # "[IMAGE 1]" → url
        self.url_to_label: Dict[str, str] = {}     # url → "[IMAGE 1]"

    def assign(self, url: str) -> str:
        # Same URL gets the same label (model sometimes returns duplicates).
        if url in self.url_to_label:
            return self.url_to_label[url]
        self.counter += 1
        label = f"[IMAGE {self.counter}]"
        self.label_to_url[label] = url
        self.url_to_label[url] = label
        return label


def _extract_image_urls(tool_name: str, tool_result_str: str) -> List[str]:
    """image_search → ordered list of result URLs. Other tools → []."""
    if tool_name != "image_search":
        return []
    try:
        parsed = json.loads(tool_result_str)
    except (json.JSONDecodeError, TypeError):
        return []
    items = parsed if isinstance(parsed, list) else [parsed]
    urls: List[str] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("ok"):
            continue
        for r in item.get("results", []):
            url = r.get("url", "")
            if url:
                urls.append(url)
    return urls


def _build_tool_response_text(
    tool_name: str,
    tool_result_str: str,
    server: EmuServerClient,
    label_book: _ImageLabelBook,
) -> str:
    """Build the body of a <tool_response>...</tool_response> block.

    Delegates to `tool_response_format.reformat_tool_response`, which
    rewrites the qwen-agent-style JSON return into the须桌 (须桌-style)
    `query : <q>\\n查询结果 : <body>` layout the model actually saw at
    training time (see the SFT convert script + sft.jsonl spot
    checks). For image_search, that reformatter also handles the
    `/encode_images` round-trip and inlines `[IMAGE N] <BoI>...<EoI>`
    after the corresponding 查询结果 line.
    """
    return reformat_tool_response(
        tool_name=tool_name,
        tool_result_str=tool_result_str,
        server=server,
        label_book=label_book,
    )


def _resolve_image_labels(args: Dict[str, Any], label_book: _ImageLabelBook) -> Dict[str, Any]:
    """In a tool_call's `arguments`, swap any "[IMAGE N]" or "IMAGE N" tokens
    in image-list keys back to the real URL before dispatching the tool."""
    if not isinstance(args, dict):
        return args
    out = dict(args)
    for key in ("image_urls", "images"):
        arr = out.get(key)
        if not isinstance(arr, list):
            continue
        new_arr = []
        for item in arr:
            if isinstance(item, str):
                norm = item.strip()
                # Accept "[IMAGE 1]" or "IMAGE 1" or "1"
                if norm in label_book.label_to_url:
                    new_arr.append(label_book.label_to_url[norm])
                    continue
                m = re.match(r"\[?IMAGE\s*(\d+)\]?$", norm, re.IGNORECASE)
                if m:
                    cand = f"[IMAGE {int(m.group(1))}]"
                    new_arr.append(label_book.label_to_url.get(cand, item))
                    continue
            new_arr.append(item)
        out[key] = new_arr
    return out


# ── Agent loop ───────────────────────────────────────────────────────────
class EmuAgent:
    """Single-conversation agent. One instance per dataset item is recommended;
    label_book / messages are per-run state so we don't reuse the same object
    across questions."""

    def __init__(
        self,
        server_url: str = EMU_SERVER_URL,
        system_message: str = SYSTEM_PROMPT,
        image_save_dir: Optional[str] = None,
        generate_cfg: Optional[Dict[str, Any]] = None,
        max_rounds: int = MAX_LLM_CALL_PER_RUN,
        max_prompt_tokens: int = MAX_PROMPT_TOKENS,
        force_draw_round: int = 0,
        force_draw_hint: str = (
            "Based on all retrieved information and reference images above, "
            "render the final image now."
        ),
    ):
        self.server = EmuServerClient(server_url)
        self.system_message = system_message
        self.image_save_dir = image_save_dir
        self.generate_cfg = generate_cfg or {}
        self.max_rounds = max_rounds
        self.max_prompt_tokens = max_prompt_tokens
        self.force_draw_round = int(force_draw_round or 0)
        self.force_draw_hint = force_draw_hint
        self.tools = _ToolDispatcher()

    # ───── primitives ─────
    def _generate_overrides(self) -> Dict[str, Any]:
        cfg = self.generate_cfg
        return {
            "max_new_tokens": cfg.get("max_new_tokens"),
            "text_temperature": cfg.get("text_temperature"),
            "text_top_p": cfg.get("text_top_p"),
            "text_top_k": cfg.get("text_top_k"),
            "image_temperature": cfg.get("image_temperature"),
            "image_top_p": cfg.get("image_top_p"),
            "image_top_k": cfg.get("image_top_k"),
            "image_save_dir": self.image_save_dir,
        }

    def _llm(
        self,
        prompt: str,
        *,
        force_image_first: bool = False,
        skip_post_image: bool = False,
        allow_native_image: bool = True,
        max_tries: int = 3,
    ) -> Dict[str, Any]:
        overrides = self._generate_overrides()
        overrides.update({
            "force_image_first": force_image_first or None,
            "skip_post_image": skip_post_image or None,
            "allow_native_image": False if not allow_native_image else None,
        })
        last_exc: Optional[Exception] = None
        for attempt in range(max_tries):
            try:
                return self.server.generate(prompt, **overrides)
            except Exception as exc:
                last_exc = exc
                print(f"[emu-agent] /generate attempt {attempt+1}/{max_tries} failed: {exc}")
        raise RuntimeError(f"/generate failed after {max_tries} tries: {last_exc}")

    # ───── one rollout ─────
    def run(self, question: str, *, rollout_id: int = 0) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": USER_PREFIX + question.strip()},
        ]
        label_book = _ImageLabelBook()

        all_saved_images: List[str] = []
        termination = "no termination"
        prediction = ""

        round_idx = 0
        while round_idx < self.max_rounds:
            round_idx += 1
            print(f"\n[emu-agent] === round {round_idx} ===")

            # ── Force draw branch: pre-empt with a hand-built draw primer. ──
            if self.force_draw_round and round_idx >= self.force_draw_round:
                draw_result = self._force_draw_step(messages, label_book)
                all_saved_images.extend(draw_result.get("saved_images", []))
                termination = draw_result["termination"]
                prediction = draw_result["prediction"]
                break

            # ── Normal step: ask the model, parse, maybe call a tool. ──
            prompt = assemble_prompt(messages)
            data = self._llm(prompt)
            body = (data.get("text") or "").strip()
            content = (data.get("content") or "").strip()
            saved = data.get("saved_images") or []
            all_saved_images.extend(saved)

            # Align with training: convert_v2.py rewrites <think>…</think>
            # into `<|extra_60|> … <|extra_61|>\n` before tokenizing. The
            # decoded model output is still using the literal `<think>` tags
            # (vLLM doesn't tokenize them as specials), so we run the same
            # rewrite here BEFORE anything else looks at the body. Idempotent
            # if the body already has BoG/EoG.
            body = rewrite_think_to_bog_eog(body)

            # If the model drew natively this round, accept and exit.
            # (Check BEFORE recording the assistant turn so we don't double-
            # append in the intercept-draw branch below.)
            if _BOI_BLOCK_RE.search(body):
                messages.append({"role": "assistant", "content": body})
                prediction = (saved[0] if saved else content) or "[image generated]"
                termination = f"native_draw round={round_idx}"
                break

            # Parse first tool_call out of the content (which still has
            # <tool_call>...</tool_call> verbatim — it doesn't get rewritten
            # to anything else by decode_generated).
            tc_match = _TOOL_CALL_RE.search(content) or _TOOL_CALL_RE.search(body)
            if not tc_match:
                # No tool_call, no image — model decided it's done. Take its
                # text as the prediction. For image-benchmark rollouts, a
                # text-only stop is not usable by downstream scorers, so force
                # one draw attempt before giving up.
                messages.append({"role": "assistant", "content": body})
                draw_result = self._force_draw_step(messages, label_book)
                all_saved_images.extend(draw_result.get("saved_images", []))
                if draw_result.get("saved_images"):
                    prediction = draw_result["prediction"]
                    termination = f"no_tool_force_draw round={round_idx} {draw_result['termination']}"
                else:
                    prediction = content or body
                    termination = f"no_tool_no_image round={round_idx}"
                break

            tool_call_text = tc_match.group(1).strip()
            tool_name = self._safe_tool_name(tool_call_text)

            # ── Intercept "draw": rewrite assistant body in-place to BoC/BoI. ──
            # The SFT data taught the model to issue a `draw` tool_call to ask
            # for an image, but we never registered a draw tool. Instead of
            # calling it, we splice <BoC>{prompt}<EoC><BoI> into the assistant
            # body where the <tool_call> used to live, then let the server
            # paint the visual tokens. Downstream sees a native draw turn,
            # zero `draw` traces in the transcript.
            if tool_name == "draw":
                # We need the span of the <tool_call>...</tool_call> WITHIN
                # `body` (not `content` — content has [generated_image: ...]
                # substitutions and visual blocks collapsed). Re-search.
                tc_in_body = _TOOL_CALL_RE.search(body)
                if tc_in_body is None:
                    # Defensive: if the call was only in `content`, fall back
                    # to appending body verbatim and letting the budget loop
                    # decide. Shouldn't happen in practice — tool_call is
                    # emitted as raw tokens, not stripped during decoding.
                    print("[emu-agent] WARN: draw tool_call in content but not body")
                    messages.append({"role": "assistant", "content": body})
                    prediction = content or body
                    termination = f"intercept_draw_span_missing round={round_idx}"
                    break
                draw_result = self._intercept_draw(
                    body=body,
                    tc_text=tool_call_text,
                    tc_body_span=tc_in_body.span(0),
                    messages=messages,
                )
                all_saved_images.extend(draw_result.get("saved_images", []))
                prediction = draw_result["prediction"]
                termination = f"{draw_result['termination']} round={round_idx}"
                break

            # ── Normal tool dispatch (image_search / text_search / ...) ──
            messages.append({"role": "assistant", "content": body})
            tool_result_str = self._dispatch_tool(tool_call_text, label_book)

            tool_resp_body = _build_tool_response_text(
                tool_name=tool_name,
                tool_result_str=tool_result_str,
                server=self.server,
                label_book=label_book,
            )
            messages.append({"role": "tool", "content": tool_resp_body})

            # Budget guard.
            try:
                tc = count_prompt_tokens(messages)
            except Exception as exc:
                print(f"[emu-agent] prompt-token count failed: {exc}")
                tc = -1
            print(f"[emu-agent] round={round_idx} tool={tool_name} prompt_tokens={tc}")
            if tc >= 0 and tc > self.max_prompt_tokens:
                print(
                    "[emu-agent] prompt over token cap; forcing compact draw "
                    f"round={round_idx} tokens={tc}"
                )
                draw_result = self._force_draw_step(messages, label_book)
                all_saved_images.extend(draw_result.get("saved_images", []))
                termination = (
                    f"prompt_token_cap_force_draw round={round_idx} "
                    f"{draw_result['termination']}"
                )
                prediction = draw_result["prediction"]
                break

        # If the loop exhausted without a draw, force one final attempt.
        if not all_saved_images and termination == "no termination":
            print("[emu-agent] budget exhausted without image — final force draw")
            draw_result = self._force_draw_step(messages, label_book)
            all_saved_images.extend(draw_result.get("saved_images", []))
            termination = "final_force_draw " + draw_result["termination"]
            prediction = draw_result["prediction"]

        return {
            "question": question,
            "rollout_id": rollout_id,
            "messages": self._dump_messages(messages, label_book),
            "prediction": prediction,
            "termination": termination,
            "saved_images": all_saved_images,
        }

    # ───── helpers ─────
    def _safe_tool_name(self, tc_text: str) -> str:
        try:
            return (json.loads(tc_text).get("name") or "").strip()
        except json.JSONDecodeError:
            return ""

    def _dispatch_tool(self, tc_text: str, label_book: _ImageLabelBook) -> str:
        try:
            tc = json.loads(tc_text)
        except json.JSONDecodeError as exc:
            return json.dumps({
                "ok": False,
                "error": f"tool_call is not valid JSON: {exc}",
            }, ensure_ascii=False)
        name = (tc.get("name") or "").strip()
        if not name:
            return json.dumps({"ok": False, "error": "tool_call missing 'name'"}, ensure_ascii=False)
        args = tc.get("arguments") or {}
        if not isinstance(args, dict):
            return json.dumps({"ok": False, "error": "arguments must be an object"}, ensure_ascii=False)
        args = _resolve_image_labels(args, label_book)
        try:
            return self.tools.call(name, args)
        except Exception as exc:
            return json.dumps({
                "ok": False,
                "error": f"tool execution failed: {exc}",
            }, ensure_ascii=False)

    def _force_draw_step(
        self, messages: List[Dict[str, str]], label_book: _ImageLabelBook
    ) -> Dict[str, Any]:
        """Hand-build the final assistant prefix and call /generate with
        force_image_first=True so the server skips the text pass."""
        primer = (
            f"{SPECIAL['bss']}"
            f"{SPECIAL['boc']} {self.force_draw_hint} {SPECIAL['eoc']}\n"
            f"{SPECIAL['boi']}"
        )

        def build_prompt(history: List[Dict[str, str]]) -> str:
            # Build prompt UP TO the open <BSS>, then append our own BoC..EoC
            # + BoI primer. We need open_assistant=False because we take over.
            return assemble_prompt(history, open_assistant=False) + primer

        def compact_history() -> List[Dict[str, str]]:
            # Keep the original task but drop accumulated tool transcripts and
            # reference-image tokens when the draw prompt is over model length.
            return [m for m in messages[:2] if m.get("role") in {"system", "user"}]

        prompt = build_prompt(messages)
        compacted = False
        try:
            prompt_tokens = len(_get_emu_tokenizer().encode(prompt, add_special_tokens=False))
        except Exception as exc:
            print(f"[emu-agent] FORCE DRAW token count failed: {exc}")
            prompt_tokens = -1
        if prompt_tokens > 0 and prompt_tokens > self.max_prompt_tokens - 1024:
            prompt = build_prompt(compact_history())
            compacted = True
        print(
            f"[emu-agent] FORCE DRAW prompt_chars={len(prompt)} "
            f"prompt_tokens={prompt_tokens} compacted={compacted}"
        )
        try:
            data = self._llm(prompt, force_image_first=True, skip_post_image=False)
        except Exception as exc:
            if not compacted and len(messages) > 2:
                compact_prompt = build_prompt(compact_history())
                print(
                    "[emu-agent] FORCE DRAW retry with compact context "
                    f"after error: {exc}"
                )
                try:
                    data = self._llm(
                        compact_prompt,
                        force_image_first=True,
                        skip_post_image=False,
                    )
                    compacted = True
                except Exception as retry_exc:
                    return {
                        "saved_images": [],
                        "prediction": f"[force-draw failed: {retry_exc}]",
                        "termination": f"force_draw_error: {retry_exc}",
                    }
            else:
                return {
                    "saved_images": [],
                    "prediction": f"[force-draw failed: {exc}]",
                    "termination": f"force_draw_error: {exc}",
                }
        saved = data.get("saved_images") or []
        content = (data.get("content") or "").strip()
        body = (data.get("text") or "").strip()
        # We "manually" emitted the assistant body up to <BoI>; the produced
        # `body` from server already includes the BOI tail (server re-prepends
        # the prompt's BOI before the EOI block) plus the post-image text.
        # We attach what came back as the final assistant turn.
        assistant_body = primer.replace(SPECIAL["bss"], "", 1) + body
        messages.append({"role": "assistant", "content": assistant_body})
        return {
            "saved_images": saved,
            "prediction": saved[0] if saved else (content or "[no image produced]"),
            "termination": "force_draw_ok" if saved else "force_draw_no_image",
        }

    # ── draw-tool interception ───────────────────────────────────────────

    def _intercept_draw(
        self,
        body: str,
        tc_text: str,
        tc_body_span: tuple,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """The model emitted `<tool_call>{"name":"draw", ...}</tool_call>` but
        we never registered a draw tool — that's the SFT format, not an
        actual capability. Rewrite the assistant turn in-place so it looks
        like the model drew natively (BoC/BoI), then ask the server to
        produce the visual tokens with force_image_first=True.

        Args:
          body:         the assistant body returned by /generate this round
                        (raw, still contains `<tool_call>...</tool_call>`)
          tc_text:      JSON blob inside <tool_call>...</tool_call>
          tc_body_span: (start, end) of the WHOLE `<tool_call>...</tool_call>`
                        substring within `body` — used to splice the BoC
                        primer in exactly where the call lived.
          messages:     conversation history. NOTE: the caller has NOT yet
                        appended this round's assistant turn — this function
                        appends it (rewritten) and returns the result.

        Returns dict with `saved_images`, `prediction`, `termination`. Shape
        is intentionally identical to `_force_draw_step` so the caller can
        treat it the same way.
        """
        # 1. Parse the draw arguments.
        try:
            tc = json.loads(tc_text)
        except json.JSONDecodeError as exc:
            return {
                "saved_images": [],
                "prediction": f"[draw intercept failed: bad JSON: {exc}]",
                "termination": f"intercept_draw_bad_json: {exc}",
            }
        args = tc.get("arguments") or {}
        if not isinstance(args, dict):
            return {
                "saved_images": [],
                "prediction": "[draw intercept failed: arguments not a dict]",
                "termination": "intercept_draw_bad_args",
            }
        caption = (args.get("prompt") or "").strip()
        if not caption:
            return {
                "saved_images": [],
                "prediction": "[draw intercept failed: empty prompt]",
                "termination": "intercept_draw_empty_caption",
            }

        # 2. Compose the rewritten assistant prefix:
        #
        #      <pre-tool_call text>
        #      <BoC> caption <EoC>
        #      <BoI>
        #
        # Everything AFTER </tool_call> in the original body is dropped — the
        # model only emits chatter there occasionally and we're about to
        # replace the whole turn with a real draw anyway.
        tc_start, _tc_end = tc_body_span
        # Defensive: caller (main loop) already rewrote <think>→BoG/EoG, but
        # if anyone calls _intercept_draw directly we make sure the head we
        # splice in matches training distribution.
        head = rewrite_think_to_bog_eog(body[:tc_start]).rstrip()
        primer_assistant = (
            (head + "\n" if head else "")
            + f"{SPECIAL['boc']} {caption} {SPECIAL['eoc']}\n"
            + f"{SPECIAL['boi']}"
        )

        # 3. Build the full prompt the server should continue from:
        #    history (open_assistant=False) + <BSS> + rewritten_assistant_prefix
        base = assemble_prompt(messages, open_assistant=False)
        prompt = base + SPECIAL["bss"] + primer_assistant

        # 4. Pass-2 (image) only — server already knows force_image_first
        #    means "skip text sampling, start emitting visual tokens".
        try:
            data = self._llm(prompt, force_image_first=True, skip_post_image=True)
        except Exception as exc:
            return {
                "saved_images": [],
                "prediction": f"[draw intercept generate failed: {exc}]",
                "termination": f"intercept_draw_generate_error: {exc}",
            }
        saved = data.get("saved_images") or []
        # `data["text"]` is what the server produced AFTER our primer — the
        # BoI block we opened plus its EoI close. Stitch it back on so the
        # assistant turn now ends with a complete `<BoI>...<EoI>`.
        tail = (data.get("text") or "").strip()
        assistant_body = primer_assistant + tail
        messages.append({"role": "assistant", "content": assistant_body})
        return {
            "saved_images": saved,
            "prediction": saved[0] if saved else "[no image produced]",
            "termination": "intercept_draw_ok" if saved else "intercept_draw_no_image",
        }

    def _dump_messages(self, messages: List[Dict[str, str]], label_book: _ImageLabelBook):
        """Produce a JSON-friendly view of the conversation: visual-token
        segments are collapsed into `[reference_image]`, BoC/EoC into
        <think_caption>...</think_caption>, BoG/EoG into <think>...</think>.
        Tool-call labels are left alone so the dump matches what the model
        actually emitted.
        """
        def clean(text: str) -> str:
            t = re.sub(
                rf"{re.escape(SPECIAL['boi'])}.*?{re.escape(SPECIAL['eoi'])}",
                "[reference_image]",
                text,
                flags=re.DOTALL,
            )
            t = re.sub(
                rf"{re.escape(SPECIAL['boc'])}\s*([\s\S]*?)\s*{re.escape(SPECIAL['eoc'])}",
                lambda m: f"<draw_caption>{m.group(1).strip()}</draw_caption>",
                t,
            )
            t = re.sub(
                rf"{re.escape(SPECIAL['bog'])}\s*([\s\S]*?)\s*{re.escape(SPECIAL['eog'])}",
                lambda m: f"<think>{m.group(1).strip()}</think>",
                t,
            )
            return t

        dumped = []
        for m in messages:
            c = m.get("content", "")
            if not isinstance(c, str):
                dumped.append(m)
                continue
            dumped.append({"role": m["role"], "content": clean(c)})
        return dumped


# ── Dataset driver ───────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="emu3p5-sft", help="display label used in output path")
    p.add_argument("--output", required=True, help="output directory root")
    p.add_argument("--dataset", default="draw")
    p.add_argument(
        "--data_dir",
        default=str(THIS_DIR.parent / "Data" / "RL"),
        help="directory holding {dataset}.jsonl",
    )
    p.add_argument("--server_url", default=EMU_SERVER_URL)
    p.add_argument(
        "--image_save_dir",
        default=None,
        help="server-side directory for decoded PNGs (None = server default)",
    )
    p.add_argument("--max_workers", type=int, default=1)
    p.add_argument("--roll_out_count", type=int, default=1)
    p.add_argument("--max_rounds", type=int, default=MAX_LLM_CALL_PER_RUN)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--text_temperature", type=float, default=None)
    p.add_argument("--text_top_p", type=float, default=None)
    p.add_argument("--text_top_k", type=int, default=None)
    p.add_argument("--image_temperature", type=float, default=None)
    p.add_argument("--image_top_p", type=float, default=None)
    p.add_argument("--image_top_k", type=int, default=None)
    p.add_argument(
        "--force_draw_round",
        type=int,
        default=0,
        help=(
            "If > 0, at round N the agent stops trying tool calls and forces "
            "the model to emit an image by appending <BSS><BoC>{hint}<EoC><BoI> "
            "and asking the server for force_image_first. 0 = disabled."
        ),
    )
    p.add_argument(
        "--force_draw_hint",
        default="Based on all retrieved information and reference images above, render the final image now.",
        help="Text that lives inside <BoC>...<EoC> when --force_draw_round triggers.",
    )
    return p.parse_args()


def load_dataset(path: str):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError("Input JSON must be a list of objects.")
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError("Unsupported file extension; use .json or .jsonl")
    return items


def extract_question(item: Dict[str, Any]) -> str:
    q = (item.get("question") or "").strip()
    if q:
        return q
    msgs = item.get("messages") or []
    for m in msgs:
        if m.get("role") == "user":
            c = m.get("content") or ""
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            if isinstance(c, str) and c.strip():
                if "User:" in c:
                    return c.split("User:", 1)[1].strip()
                return c.strip()
    return ""


def main():
    args = parse_args()
    model_name = os.path.basename(args.model.rstrip("/")) or "emu"
    out_dir = Path(args.output) / model_name / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] model={model_name} dataset={args.dataset}")
    print(f"[run] output_dir={out_dir}")
    print(f"[run] server={args.server_url} workers={args.max_workers}")
    print(f"[run] force_draw_round={args.force_draw_round}")
    print(f"[run] tool_dir={TOOL_DIR}")
    print(f"[run] tool_imagesearch={getattr(tool_imagesearch, '__file__', '')}")
    print(f"[run] tool_textsearch={getattr(tool_textsearch, '__file__', '')}")

    data_path = Path(args.data_dir) / f"{args.dataset}.jsonl"
    items = load_dataset(str(data_path))

    generate_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "text_temperature": args.text_temperature,
        "text_top_p": args.text_top_p,
        "text_top_k": args.text_top_k,
        "image_temperature": args.image_temperature,
        "image_top_p": args.image_top_p,
        "image_top_k": args.image_top_k,
    }

    for rollout_idx in range(1, args.roll_out_count + 1):
        out_file = out_dir / f"iter{rollout_idx}.jsonl"
        print(f"\n[run] rollout {rollout_idx}/{args.roll_out_count} → {out_file}")

        processed: set = set()
        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if "question" in d and "error" not in d:
                            processed.add(d["question"].strip())
                    except json.JSONDecodeError:
                        continue

        tasks: List[Tuple[Dict[str, Any], int]] = []
        for item in items:
            q = extract_question(item)
            if not q or q in processed:
                continue
            tasks.append((dict(item, question=q), rollout_idx))

        print(f"[run] tasks to run: {len(tasks)} (already done: {len(processed)})")
        if not tasks:
            continue

        write_lock = threading.Lock()

        def _runner(item: Dict[str, Any], rollout_id: int) -> Dict[str, Any]:
            agent = EmuAgent(
                server_url=args.server_url,
                image_save_dir=args.image_save_dir,
                generate_cfg=generate_cfg,
                max_rounds=args.max_rounds,
                force_draw_round=args.force_draw_round,
                force_draw_hint=args.force_draw_hint,
            )
            try:
                return agent.run(item["question"], rollout_id=rollout_id)
            except Exception as exc:
                return {
                    "question": item["question"],
                    "rollout_id": rollout_id,
                    "error": f"agent.run crashed: {exc}",
                    "messages": [],
                    "prediction": "[failed]",
                }

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(_runner, item, rid): (item, rid)
                for (item, rid) in tasks
            }
            for future in tqdm(
                as_completed(future_to_task),
                total=len(tasks),
                desc=f"rollout {rollout_idx}",
            ):
                item, _rid = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "question": item["question"],
                        "rollout_id": _rid,
                        "error": f"future failed: {exc}",
                        "messages": [],
                        "prediction": "[failed]",
                    }
                with write_lock, open(out_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"[run] rollout {rollout_idx} done")

    print(f"\n[run] all {args.roll_out_count} rollouts done")


if __name__ == "__main__":
    main()
