"""Turn a stray `draw` tool_call into a BoC/BoI image-generation step.

Background
----------
The SFT'd Emu3.5 was trained to issue `<tool_call>{"name":"draw", ...}</tool_call>`
to ask for image generation, but the inference server doesn't actually expose
a draw tool — instead the training data also showed the model how to emit
images natively via `<BSS><BoC>caption<EoC><BoI>visual_tokens<EoI><ESS>`
(see template.py:20). So when the model emits a draw call, we don't need to
fail; we can rewrite it into the BoC/BoI primer the model itself was trained
to continue, and let the server pass-2 actually paint the pixels.

What this script does
---------------------
Given:
  1. The assistant text that contains the draw call (the model's raw output —
     `<think>...</think><tool_call>{"name":"draw","arguments":{"prompt": ...,
     "images":[url, ...]}}</tool_call>`)
  2. The conversation history that came BEFORE this assistant turn
     (messages JSON, as built by EmuAgent — with image_search rounds and
     their `<BoI>...<EoI>` blocks already expanded)

it:
  1. parses out the draw tool_call → caption (`arguments.prompt`) and
     reference urls (`arguments.images`)
  2. (optionally, when --reencode-images) encodes those urls via
     /encode_images and appends a fresh tool_response block to the history so
     the model can "see" them next to the caption. By default, we ASSUME the
     history already contains `[IMAGE k]` references — the typical case where
     image_search has already populated them — and only the caption goes into
     BoC.
  3. assembles the prompt with `assemble_prompt(messages, open_assistant=False)`
     and appends `<BSS><BoC> {caption} <EoC>\n<BoI>` as the primer
  4. POSTs to /generate with `force_image_first=True` so the server skips
     pass-1 (text) and goes straight to image sampling — exactly the same
     handoff `_force_draw_step` in run.py uses.

CLI
---
  python -m tool_draw_to_boc \
      --assistant-text-file path/to/raw_assistant.txt \
      --history-file path/to/messages.json \
      [--server-url http://127.0.0.1:23333] \
      [--image-save-dir /tmp/out] \
      [--reencode-images]                 # encode arguments.images via /encode_images
      [--out path/to/result.json]

Or just dry-run the prompt to stdout without hitting the server:
  python -m tool_draw_to_boc --assistant-text-file ... --history-file ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Make sure we can import the in-repo modules whether this is run as a script
# or imported. emu_infer/ is already on sys.path when launched via run_*.sh
# (PYTHONPATH includes THIS_DIR), but be defensive when invoked standalone.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import requests  # noqa: E402

from template import SPECIAL, assemble_prompt  # noqa: E402


DEFAULT_SERVER_URL = os.environ.get("EMU_SERVER_URL", "http://127.0.0.1:23333")


# ── tool_call parsing ────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_draw_call(assistant_text: str) -> Optional[Dict[str, Any]]:
    """Find the FIRST `<tool_call>{"name":"draw", ...}</tool_call>` in
    `assistant_text` and return its parsed JSON. Returns None if no draw
    call is present.

    Robustness:
      - Tolerates surrounding `<think>`, whitespace, and any number of
        non-draw tool calls before the draw one.
      - If the JSON is malformed, raises ValueError with the offending blob
        so the caller can decide whether to bail or retry.
    """
    for m in _TOOL_CALL_RE.finditer(assistant_text):
        blob = m.group(1)
        try:
            call = json.loads(blob)
        except json.JSONDecodeError as exc:
            # Don't silently swallow — surfacing it once is more useful than
            # picking the wrong call.
            raise ValueError(
                f"tool_call body is not valid JSON: {exc}; blob={blob[:200]!r}"
            ) from exc
        if isinstance(call, dict) and call.get("name") == "draw":
            return call
    return None


def caption_and_refs_from_draw(call: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Pull the caption and reference urls out of a parsed draw tool_call.

    Returns (caption, refs). Caption is stripped; refs is the original list
    (may contain `[IMAGE k]` placeholders OR real urls — left to the caller
    to decide what to do with them).
    """
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        raise ValueError(f"draw.arguments is not a dict: {args!r}")
    caption = (args.get("prompt") or "").strip()
    if not caption:
        raise ValueError("draw.arguments.prompt is empty — nothing to paint")
    refs = args.get("images") or args.get("image_urls") or []
    if not isinstance(refs, list):
        raise ValueError(f"draw.arguments.images is not a list: {refs!r}")
    refs = [r for r in refs if isinstance(r, str) and r.strip()]
    return caption, refs


# ── server helpers (tiny client, no run.py dependency) ───────────────────


class _Server:
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
        payload.update({k: v for k, v in overrides.items() if v is not None})
        r = requests.post(
            f"{self.base}/generate",
            json=payload,
            timeout=self.timeout,
        )
        if not r.ok:
            print(f"[draw_to_boc] HTTP {r.status_code}: {r.text[:400]}", file=sys.stderr)
        r.raise_for_status()
        return r.json()


# ── core: rewrite draw → BoC/BoI primer ──────────────────────────────────


def build_boc_boi_prompt(
    history: List[Dict[str, str]],
    caption: str,
    *,
    extra_tool_response: Optional[str] = None,
) -> str:
    """Build the exact prompt to feed to /generate with force_image_first=True.

    Args:
      history:              the conversation messages PRIOR to the draw turn,
                            using the same schema as EmuAgent (roles
                            "system", "user", "tool", "assistant"). Any
                            `<BoI>...<EoI>` blocks inside content must
                            already be expanded by the caller (e.g. by an
                            earlier /encode_images round).
      caption:              what goes between `<BoC>` and `<EoC>`.
      extra_tool_response:  optional pre-built `<tool_response>` body to
                            append as one more `role="tool"` message right
                            before the primer — use this when the caller
                            wants to inject `[IMAGE k] <BoI>...<EoI>` blocks
                            re-encoded from the draw call's `images` list so
                            the model can attend to them.

    Returns the assembled prompt string. The final layout is::

        <history...>
        [optional <tool_response>...</tool_response>]
        <BSS><BoC> {caption} <EoC>
        <BoI>

    The trailing `<BoI>` is what tells the server (with
    `force_image_first=True`) to skip pass-1 and start sampling visual
    tokens.
    """
    msgs = list(history)
    if extra_tool_response:
        msgs.append({"role": "tool", "content": extra_tool_response})

    # assemble_prompt(open_assistant=False) gives us the prompt UP TO (but
    # NOT including) the next `<BSS>`. We hand-build the assistant prefix.
    base = assemble_prompt(msgs, open_assistant=False)
    primer = (
        f"{SPECIAL['bss']}"
        f"{SPECIAL['boc']} {caption.strip()} {SPECIAL['eoc']}\n"
        f"{SPECIAL['boi']}"
    )
    return base + primer


def run_draw_to_boc(
    assistant_text: str,
    history: List[Dict[str, str]],
    server: _Server,
    *,
    reencode_images: bool = False,
    image_save_dir: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """End-to-end: parse draw call, optionally re-encode its refs, build the
    BoC/BoI primer, hit /generate (unless dry_run), return a structured
    result the caller can persist or inspect.
    """
    call = extract_draw_call(assistant_text)
    if call is None:
        return {
            "ok": False,
            "error": "no draw tool_call found in assistant_text",
        }

    try:
        caption, refs = caption_and_refs_from_draw(call)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    extra_tool_response: Optional[str] = None
    encoded: List[Dict[str, Any]] = []
    if reencode_images and refs:
        # Skip refs that look like [IMAGE k] placeholders — they're already
        # in the history; only re-encode actual urls/data-uris/local paths.
        real_refs = [
            r for r in refs
            if not re.match(r"\[?IMAGE\s*\d+\]?$", r.strip(), re.IGNORECASE)
        ]
        if real_refs:
            enc = server.encode_images(real_refs)
            tokens = enc.get("tokens", [])
            oks = enc.get("ok", [])
            lines: List[str] = []
            # Continue numbering AFTER any [IMAGE k] already in history.
            existing = _max_image_label_in_history(history)
            for i, (url, tok, ok) in enumerate(zip(real_refs, tokens, oks), 1):
                label = f"[IMAGE {existing + i}]"
                if ok and tok:
                    lines.append(f"{label} {tok}")
                else:
                    lines.append(f"{label} [image load failed]")
                encoded.append({"url": url, "label": label, "ok": bool(ok)})
            extra_tool_response = "\n".join(lines)

    prompt = build_boc_boi_prompt(
        history, caption, extra_tool_response=extra_tool_response
    )

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "caption": caption,
            "refs": refs,
            "encoded": encoded,
            "prompt_chars": len(prompt),
            "prompt_tail": prompt[-1500:],
        }

    overrides: Dict[str, Any] = {
        "force_image_first": True,
        "skip_post_image": False,
    }
    if image_save_dir:
        overrides["image_save_dir"] = image_save_dir

    data = server.generate(prompt, **overrides)
    saved = data.get("saved_images") or []
    return {
        "ok": True,
        "dry_run": False,
        "caption": caption,
        "refs": refs,
        "encoded": encoded,
        "saved_images": saved,
        "stopped_on_eoi": data.get("stopped_on_eoi", False),
        "content": data.get("content", ""),
        "text": data.get("text", ""),
    }


# ── small utilities ──────────────────────────────────────────────────────


_IMAGE_LABEL_RE = re.compile(r"\[IMAGE\s+(\d+)\]")


def _max_image_label_in_history(history: List[Dict[str, str]]) -> int:
    """Find the highest `[IMAGE k]` index already present in history so we
    can continue numbering without colliding."""
    hi = 0
    for m in history:
        c = m.get("content", "")
        if not isinstance(c, str):
            continue
        for match in _IMAGE_LABEL_RE.finditer(c):
            try:
                k = int(match.group(1))
                if k > hi:
                    hi = k
            except ValueError:
                pass
    return hi


# ── CLI ──────────────────────────────────────────────────────────────────


def _read_text_arg(literal: Optional[str], path: Optional[str], what: str) -> str:
    if literal and path:
        raise SystemExit(f"Pass exactly one of --{what} / --{what}-file")
    if literal:
        return literal
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    raise SystemExit(f"Need --{what} or --{what}-file")


def _read_history_arg(path: Optional[str], literal: Optional[str]) -> List[Dict[str, str]]:
    if literal and path:
        raise SystemExit("Pass exactly one of --history / --history-file")
    if literal:
        data = json.loads(literal)
    elif path:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        return []
    if not isinstance(data, list):
        raise SystemExit("history must be a JSON list of {role, content} messages")
    for i, m in enumerate(data):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise SystemExit(f"history[{i}] missing role/content: {m!r}")
    return data


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="tool_draw_to_boc",
        description=(
            "Rewrite a stray <tool_call name=draw> into a BoC/BoI image-gen "
            "step and ask the emu server to actually paint it."
        ),
    )
    p.add_argument(
        "--assistant-text", default=None,
        help="The raw assistant body that contains the draw tool_call.",
    )
    p.add_argument(
        "--assistant-text-file", default=None,
        help="Read assistant text from a file instead of --assistant-text.",
    )
    p.add_argument(
        "--history", default=None,
        help="JSON-encoded list of prior messages [{role, content}, ...].",
    )
    p.add_argument(
        "--history-file", default=None,
        help="Read history JSON from a file instead of --history.",
    )
    p.add_argument(
        "--server-url", default=DEFAULT_SERVER_URL,
        help=f"Emu server base url (default {DEFAULT_SERVER_URL}).",
    )
    p.add_argument(
        "--image-save-dir", default=None,
        help="Forwarded to /generate so saved PNGs land there.",
    )
    p.add_argument(
        "--reencode-images", action="store_true",
        help=(
            "Encode the draw call's `images` urls via /encode_images and "
            "splice them in as an extra <tool_response>. Default OFF — we "
            "assume the history already contains the relevant [IMAGE k] "
            "blocks (typical when image_search was the previous round)."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Build the prompt but don't hit /generate. Prints the tail.",
    )
    p.add_argument(
        "--out", default=None,
        help="Write the result JSON to this path. Otherwise prints to stdout.",
    )
    args = p.parse_args(argv)

    assistant_text = _read_text_arg(
        args.assistant_text, args.assistant_text_file, "assistant-text"
    )
    history = _read_history_arg(args.history_file, args.history)

    server = _Server(args.server_url)
    result = run_draw_to_boc(
        assistant_text=assistant_text,
        history=history,
        server=server,
        reencode_images=args.reencode_images,
        image_save_dir=args.image_save_dir,
        dry_run=args.dry_run,
    )

    blob = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(blob)
        print(f"[draw_to_boc] wrote {args.out}")
    else:
        print(blob)

    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
