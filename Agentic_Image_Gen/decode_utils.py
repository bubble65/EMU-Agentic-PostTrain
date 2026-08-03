"""Decode vLLM output token ids → readable text plus on-disk PNGs.

Two outputs:

  * `text` — special tokens preserved, ESS/EOS stripped at the boundary so the
             assistant body can be re-fed into the next prompt without
             doubling up BSS/ESS.
  * `content` — display-friendly: <BoI>...<EoI> blocks are VQ-decoded, saved
                as PNGs, and replaced inline with `[generated_image: /path]`.
                Tool-call markers stay verbatim so the agent can parse them.
"""
from __future__ import annotations

import os
import re
import time
import uuid
from typing import Dict, List

import numpy as np
import torch
from PIL import Image


_BSS = "<|extra_100|>"
_ESS = "<|extra_101|>"
_EOS = "<|extra_204|>"
_BOI = "<|image start|>"
_EOI = "<|image end|>"
_EOL = "<|extra_200|>"

_VISUAL_TOKEN_RE = re.compile(r"<\|visual token (\d+)\|>")
_IMAGE_BLOCK_RE = re.compile(
    rf"{re.escape(_BOI)}.*?{re.escape(_EOI)}",
    re.DOTALL,
)


def _strip_outer_wrappers(text: str) -> str:
    """Drop boundary BSS/ESS/EOS so we can re-frame the body cleanly.

    The server's text-mode generation may stop right at ESS (with the token
    included) and the image-mode pass may end with ESS+EOS. We strip those so
    the assistant body can be re-wrapped by assemble_prompt without doubling
    up special tokens.
    """
    for tok in (_EOS, _ESS, _BSS):
        text = text.replace(tok, "")
    return text.strip()


def _decode_one_image(block_str: str, vq_model, save_dir: str):
    """Take a <BOI>...<EOI> substring, vq-decode, save PNG, return path or None."""
    rows: List[List[int]] = []
    for row_str in re.split(re.escape(_EOL), block_str):
        ids = _VISUAL_TOKEN_RE.findall(row_str)
        if ids:
            rows.append([int(x) for x in ids])
    if not rows:
        return None

    try:
        device = next(iter(vq_model.parameters())).device
        widths = {len(r) for r in rows}
        if len(widths) > 1:
            # Truncate to the shortest row width so torch.tensor doesn't ragged-error.
            w = min(widths)
            rows = [r[:w] for r in rows]
        tensor = torch.tensor(rows, dtype=torch.long, device=device)
        h, w = tensor.shape
        recon = vq_model.decode_code(tensor[None], shape=(1, h, w, 256)).float()
        recon = recon[0].permute(1, 2, 0)
        arr = ((recon + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
        img = Image.fromarray(arr)
    except Exception as exc:
        print(f"[decode] vq decode failed: {exc}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"emu_gen_{ts}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(save_dir, fname)
    try:
        img.save(path)
    except Exception as exc:
        print(f"[decode] save image failed {path}: {exc}")
        return None
    return path


def replace_visual_blocks(text: str, vq_model, save_dir: str):
    """Replace every BOI…EOI block with `[generated_image: /path]`.

    Returns (new_text, [saved_paths]).
    """
    saved: List[str] = []

    def repl(m: re.Match) -> str:
        path = _decode_one_image(m.group(0), vq_model, save_dir)
        if path is None:
            return "[image decode failed]"
        saved.append(path)
        return f"[generated_image: {path}]"

    new_text = _IMAGE_BLOCK_RE.sub(repl, text)
    return new_text, saved


def decode_generated(token_ids, tokenizer, vq_model, image_save_dir: str) -> Dict[str, object]:
    """Post-process freshly generated tokens.

    Returns:
      {
        "raw_decoded":  full tokenizer.decode (special tokens kept),
        "text":         BSS/ESS/EOS stripped — re-feedable assistant body,
        "content":      `text` with BOI…EOI → `[generated_image: /path]`,
        "saved_images": [str, ...],
      }
    """
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    body = _strip_outer_wrappers(raw)
    content, saved = replace_visual_blocks(body, vq_model, image_save_dir)
    return {
        "raw_decoded": raw,
        "text": body,
        "content": content.strip(),
        "saved_images": saved,
    }


__all__ = ["decode_generated", "replace_visual_blocks"]
