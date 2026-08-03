"""Emu3.5 SFT-format prompt assembler.

This MATCHES the exact layout produced by the SFT convert script — i.e.
the format the SFT'd model actually saw at training time. No "USER:" /
"ASSISTANT:" framing, no extra whitespace between turns; everything is glued
directly with the special tokens that delimit each segment.

The training-time text (per sample) looks like::

    <BOS>{system_text}\n\n{first_user_question}
    <BSS><BoG> think <EoG>
    <tool_call>{json}</tool_call>
    <ESS>
    <tool_response>
    [IMAGE N] <BoI>H*W<IMG>v..v<EOL>...<EoI>   # only if image_search
    ...
    </tool_response>
    <BSS> ... (more think → tool_call rounds) ... <ESS>
    <tool_response>...</tool_response>
    <BSS><BoC> draw caption with [IMAGE N] refs <EoC>
          <BoI>H*W<IMG>v..v<EOL>...<EoI>
    <ESS><EOS>

We expose a single `assemble_prompt(messages)` that takes an internal list of
dicts (with roles "system", "user", "tool", "assistant") and produces the
exact prompt string the vLLM server should feed to the model.

Each `content` is a plain string already in emu format — visual-token blocks
must already be expanded into `<BoI>H*W<IMG_TOKEN>...<EoI>` substrings by the
caller (via /encode_images).
"""
from __future__ import annotations

from typing import Dict, List, Optional


# ── Special tokens (must match the SFT convert script SPECIAL) ───────────
SPECIAL: Dict[str, str] = dict(
    bos="<|extra_203|>",
    eos="<|extra_204|>",
    pad="<|endoftext|>",
    eol="<|extra_200|>",
    eof="<|extra_201|>",
    tms="<|extra_202|>",
    img="<|image token|>",
    boi="<|image start|>",
    eoi="<|image end|>",
    bss="<|extra_100|>",     # begin assistant span
    ess="<|extra_101|>",     # end assistant span
    bog="<|extra_60|>",      # begin think
    eog="<|extra_61|>",      # end think
    boc="<|extra_50|>",      # begin draw caption
    eoc="<|extra_51|>",      # end draw caption
)


def assemble_prompt(messages: List[Dict[str, str]], *, open_assistant: bool = True) -> str:
    """Build the exact SFT-format prompt string.

    Roles understood:
      * "system"     — the system block. Emitted once at the very start as
                       `<BOS>{content}`. Multiple system messages are joined.
      * "user"       — for the FIRST user turn this is the user question and
                       is appended with `\n\n` after the system block (no
                       framing tokens; matches convert_v2.py prefix).
                       Subsequent "user" turns are forbidden — image-search /
                       text-search responses use role="tool" instead.
      * "tool"       — a tool_response block. Emitted as
                       `<tool_response>\n{content}\n</tool_response>` and
                       glued directly after the preceding `<ESS>` (no
                       whitespace, no role label).
      * "assistant"  — content is the assistant span body, i.e. what falls
                       BETWEEN `<BSS>` and `<ESS>` in training. We wrap it
                       with `<BSS>{content}<ESS>`. The body itself may
                       contain `<BoG>...<EoG>`, `<tool_call>...</tool_call>`,
                       `<BoC>...<EoC>` and `<BoI>...<EoI>` segments.

    If `open_assistant=True` (the default) and the last message is NOT an
    assistant turn, we append a trailing `<BSS>` so the model continues with
    the next assistant span. Set `open_assistant=False` when you want to
    inject a hand-built assistant prefix yourself after the call.
    """
    pieces: List[str] = []
    system_text: List[str] = []
    first_user_used = False
    last_role: Optional[str] = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if not isinstance(content, str):
            raise TypeError(
                f"assemble_prompt expects content to be a string, got "
                f"{type(content).__name__} for role={role!r}. Image blocks "
                "must already be expanded into the string by the caller."
            )

        if role == "system":
            system_text.append(content)
            continue

        if role == "user":
            if first_user_used:
                raise ValueError(
                    "assemble_prompt only supports ONE user turn (the initial "
                    "question). Tool results must use role='tool', not 'user'."
                )
            # Emit `<BOS>{system}\n\n{user_question}` exactly like the SFT prefix.
            sys_blob = ("\n\n".join(s for s in system_text if s)).strip()
            prefix = SPECIAL["bos"] + sys_blob
            if sys_blob and content:
                prefix += "\n\n"
            prefix += content
            pieces.append(prefix)
            first_user_used = True

        elif role == "assistant":
            pieces.append(f"{SPECIAL['bss']}{content}{SPECIAL['ess']}")

        elif role == "tool":
            # Glue directly after the previous ESS — no separator, matches
            # `</tool_call><|extra_101|><tool_response>` from training.
            pieces.append(f"<tool_response>\n{content}\n</tool_response>")

        else:
            raise ValueError(f"assemble_prompt: unsupported role {role!r}")

        last_role = role

    if not first_user_used:
        # Allow a system-only "prompt" if someone wants it, but the model has
        # no question to answer; fall back to just <BOS>{system}.
        sys_blob = ("\n\n".join(s for s in system_text if s)).strip()
        if sys_blob:
            pieces.append(SPECIAL["bos"] + sys_blob)

    if open_assistant and last_role != "assistant":
        pieces.append(SPECIAL["bss"])

    return "".join(pieces)


__all__ = ["SPECIAL", "assemble_prompt"]
