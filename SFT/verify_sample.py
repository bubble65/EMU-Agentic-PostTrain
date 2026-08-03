#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify tokenized SFT samples by decoding the VQ image tokens back to PNGs
and placing them next to the original references (URLs / draw saved_path)
so a human can eyeball alignment.

For each sample we produce a dir <out>/sample_<lineno>/:
    README.md            -- human-readable walkthrough with every text segment
                            and the resolved [IMAGE N] -> file mappings
    text.txt             -- the full detokenized text (no image tokens),
                            same as record["text"] but with VQ runs collapsed
                            to "<IMAGE_BLOCK k: HxW>" for readability
    image_<k>_decoded.png        -- decoded from the tokens emitted in the sft text
    image_<k>_original.png       -- the source: downloaded URL or copied draw png
    image_<k>_meta.txt           -- url / saved_path / size / caption snippet

Run:
    python verify_sample.py \
        --sft ../Data/SFT/converted_v2/sft.jsonl \
        --raw ../Data/SFT/UPE_raw_gensearcher_sft_trace_relpath.jsonl \
        --tool-resp-dir ../Data/SFT/image \
        --vq-path ../Emu3.5-VisionTokenizer \
        --tokenizer-path ../Emu3.5/src/tokenizer_emu3_ibq \
        --out ./SFT/verify_out \
        --line-nos 1 5 11
"""
import argparse
import io
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Special tokens (mirror convert_v2.py) ──────────────────────────────────
BOS = "<|extra_203|>"
EOS = "<|extra_204|>"
BSS = "<|extra_100|>"
ESS = "<|extra_101|>"
BOG = "<|extra_60|>"
EOG = "<|extra_61|>"
BOC = "<|extra_50|>"
EOC = "<|extra_51|>"
BOI = "<|image start|>"
EOI = "<|image end|>"
IMG = "<|image token|>"
EOL = "<|extra_200|>"

# ── Regexes ────────────────────────────────────────────────────────────────
RE_IMAGE_BLOCK = re.compile(
    re.escape(BOI) + r"(\d+)\*(\d+)" + re.escape(IMG) + r"(.*?)" + re.escape(EOI),
    re.DOTALL,
)
RE_VISUAL_TOKEN = re.compile(r"<\|visual token (\d+)\|>")
RE_IMAGE_TAG = re.compile(r"\[IMAGE\s*(\d+)\]")
RE_EXTERNAL_IMG_RAW = re.compile(
    r"\[external image\s*(\d+)\]([\s\S]*?)\[/external image\s*\1\]",
    re.IGNORECASE,
)
RE_TOOL_RESP = re.compile(r"<tool_response>\s*([\s\S]*?)\s*</tool_response>")
RE_INTERNAL_IMG = re.compile(r"\[internal image\s*\d+\]([\s\S]*?)\[/internal image\s*\d+\]")
RE_SAVED_FALLBACK = re.compile(r"已保存到\s*([^\s<]+)")
RE_THINK = re.compile(r"<think>\s*([\s\S]*?)\s*</think>")
RE_TOOL_CALL = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>")


def _get_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            it.get("text", "")
            for it in content
            if isinstance(it, dict) and it.get("type") == "text"
        )
    return ""


def _collect_originals(raw_record: Dict[str, Any]) -> Dict[int, Dict[str, str]]:
    """Return {global_idx -> {"url": ..., "kind": "external"/"draw", "saved_path": ...}}
    using the SAME global indexing convert_v2 uses (the integer inside
    `[external image N]...[/external image N]`).

    The draw turn's saved_path gets the next free index = max(external)+1.
    """
    out: Dict[int, Dict[str, str]] = {}
    messages = raw_record.get("messages", [])
    max_ext = 0
    last_saved_path: Optional[str] = None
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = _get_text(msg.get("content", ""))
        m = RE_TOOL_RESP.search(text)
        if not m:
            continue
        body = m.group(1)
        for em in RE_EXTERNAL_IMG_RAW.finditer(body):
            n = int(em.group(1))
            url = em.group(2).strip()
            out[n] = {"kind": "external", "url": url}
            if n > max_ext:
                max_ext = n
        im = RE_INTERNAL_IMG.search(body)
        if im:
            last_saved_path = im.group(1).strip()
        else:
            fm = RE_SAVED_FALLBACK.search(body)
            if fm:
                last_saved_path = fm.group(1).strip()
    if last_saved_path is not None:
        out[max_ext + 1] = {"kind": "draw", "saved_path": last_saved_path}
    return out


def _download(url: str, timeout: int = 60) -> Optional[Image.Image]:
    """Best-effort download; returns None on failure (we still keep the meta)."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}/"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Referer": origin,
            "Connection": "close",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        print(f"  [warn] download failed: {url[:120]} :: {exc}")
        return None


def _decode_image_block(
    height: int, width: int, body: str, vq_model
) -> Optional[Image.Image]:
    """Pull <|visual token NNNNNN|> out of body, reshape (H,W), call decode_code."""
    ids = [int(x) for x in RE_VISUAL_TOKEN.findall(body)]
    expected = height * width
    if len(ids) != expected:
        print(
            f"  [warn] image block size mismatch: declared {height}x{width}={expected},"
            f" got {len(ids)} tokens"
        )
        return None
    device = next(vq_model.parameters()).device
    grid = torch.tensor(ids, dtype=torch.long, device=device).view(1, height, width)
    with torch.no_grad():
        rec = vq_model.decode_code(grid, shape=(1, height, width, 256)).float()
    rec = rec[0].permute(1, 2, 0).clamp(-1, 1).detach().cpu().numpy()
    arr = ((rec + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _summarize_text(sft_text: str) -> str:
    """Replace each image block with a placeholder so the human can see structure."""
    counter = {"n": 0}

    def repl(m: re.Match) -> str:
        counter["n"] += 1
        h, w = m.group(1), m.group(2)
        return f"\n[IMAGE_BLOCK {counter['n']}: {h}x{w} tokens]\n"

    return RE_IMAGE_BLOCK.sub(repl, sft_text)


def _emit_walkthrough(
    sft_text: str,
    image_blocks: List[Tuple[int, int]],  # (h, w) per block in order
    label_for_block: List[Optional[int]],  # global [IMAGE N] tied to each block (None for the final draw)
    originals: Dict[int, Dict[str, str]],
    out_dir: Path,
) -> str:
    """Build a markdown walkthrough of the sample."""
    lines: List[str] = []
    lines.append(f"# Sample walkthrough\n")
    lines.append(f"Total image blocks: {len(image_blocks)}\n")
    lines.append("## Block ↔ original mapping\n")
    lines.append("| Block | Tokens HxW | Global tag | Kind | Source |")
    lines.append("|------|-----------|-----------|------|--------|")
    for k, ((h, w), tag) in enumerate(zip(image_blocks, label_for_block), start=1):
        kind = "?"
        src = "?"
        if tag is not None and tag in originals:
            meta = originals[tag]
            kind = meta["kind"]
            src = meta.get("url") or meta.get("saved_path", "")
        elif tag is None:
            kind = "draw"
            # tag is None means the LAST block (the generated image at the end)
            # we just guess the largest key.
            if originals:
                last_key = max(originals.keys())
                if originals[last_key]["kind"] == "draw":
                    src = originals[last_key].get("saved_path", "")
        lines.append(f"| {k} | {h}x{w} | {tag if tag is not None else '(final draw)'} | {kind} | `{src}` |")
    lines.append("")
    lines.append("## Decoded vs original (side-by-side)\n")
    for k in range(1, len(image_blocks) + 1):
        lines.append(f"### Block {k}\n")
        lines.append(f"- decoded: `image_{k}_decoded.png`")
        lines.append(f"- original: `image_{k}_original.png`")
        lines.append(f"- meta: `image_{k}_meta.txt`\n")
    lines.append("## Text segments\n")
    lines.append("```")
    lines.append(_summarize_text(sft_text))
    lines.append("```")
    return "\n".join(lines)


def _parse_boc_captions(sft_text: str) -> List[str]:
    """Extract every <BoC>...<EoC> caption in order (these label the draw block)."""
    return [
        m.group(1).strip()
        for m in re.finditer(
            re.escape(BOC) + r"\s*(.*?)\s*" + re.escape(EOC), sft_text, re.DOTALL
        )
    ]


def _assign_global_tags(sft_text: str) -> List[Optional[int]]:
    """Walk the text in order. For each image block, the label is the
    `[IMAGE N]` token emitted immediately before it INSIDE a <tool_response>.
    The final draw block lives inside <BoC>...<EoC><BoI>...<EoI>, so its
    look-back window contains EoC; in that case we return None (= "final draw").

    Caveat: the system prompt itself sometimes mentions the literal tokens
    `<|extra_50|>` / `<|extra_51|>` in its instructions ("close the caption
    with <|extra_51|>"). Those live in the prefix (before the very first BSS)
    and must NOT count as structural BoC/EoC. We anchor look-back at the
    first BSS, so anything in the prefix is invisible to it.
    """
    first_bss = sft_text.find(BSS)
    anchor = 0 if first_bss == -1 else first_bss

    blocks = list(RE_IMAGE_BLOCK.finditer(sft_text))
    tags: List[Optional[int]] = []
    prev_end = anchor
    for blk in blocks:
        window = sft_text[prev_end:blk.start()]
        prev_end = blk.end()
        last_eoc = window.rfind(EOC)
        last_boi = window.rfind(BOI)
        if last_eoc != -1 and last_eoc > last_boi:
            tags.append(None)
            continue
        matches = list(RE_IMAGE_TAG.finditer(window))
        if matches:
            tags.append(int(matches[-1].group(1)))
        else:
            tags.append(None)
    return tags


def verify_one(
    raw_record: Dict[str, Any],
    sft_record: Dict[str, Any],
    line_no: int,
    tool_resp_dir: Path,
    out_root: Path,
    vq_model,
    skip_downloads: bool,
) -> None:
    sample_dir = out_root / f"sample_{line_no}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== sample line_no={line_no} → {sample_dir} ===")

    text = sft_record["text"]
    # Snapshot the question/meta for context.
    (sample_dir / "question.txt").write_text(
        sft_record.get("meta", {}).get("question", "") + "\n", encoding="utf-8"
    )

    originals = _collect_originals(raw_record)
    print(f"  originals collected: {list(originals.keys())}")
    for k, v in originals.items():
        print(f"    [{k}] kind={v['kind']}  src={(v.get('url') or v.get('saved_path',''))[:120]}")

    # Walk every image block, decode it, write decoded.png next to a copy of
    # the original (downloaded URL or local draw PNG).
    blocks = list(RE_IMAGE_BLOCK.finditer(text))
    tags = _assign_global_tags(text)
    block_sizes = [(int(m.group(1)), int(m.group(2))) for m in blocks]
    print(f"  found {len(blocks)} image blocks; tags={tags}")

    boc_captions = _parse_boc_captions(text)
    if boc_captions:
        (sample_dir / "boc_captions.txt").write_text(
            "\n\n---\n\n".join(boc_captions), encoding="utf-8"
        )

    for k, (m, (h, w), tag) in enumerate(zip(blocks, block_sizes, tags), start=1):
        # Decode token block.
        dec = _decode_image_block(h, w, m.group(3), vq_model)
        if dec is not None:
            dec.save(sample_dir / f"image_{k}_decoded.png")
            print(f"  [block {k}] decoded {h}x{w} tokens → image_{k}_decoded.png  ({dec.size[0]}x{dec.size[1]} px)")
        else:
            print(f"  [block {k}] decode FAILED for {h}x{w} block")

        # Resolve and save the original (URL or local file).
        meta_lines = [f"block_index={k}", f"token_grid={h}x{w}"]
        if tag is not None and tag in originals:
            info = originals[tag]
            meta_lines.append(f"global_tag=[IMAGE {tag}]")
            meta_lines.append(f"kind={info['kind']}")
            if info["kind"] == "external":
                url = info["url"]
                meta_lines.append(f"url={url}")
                if skip_downloads:
                    meta_lines.append("download_skipped=1")
                else:
                    img = _download(url)
                    if img is not None:
                        img.save(sample_dir / f"image_{k}_original.png")
                        meta_lines.append(f"original_size={img.size[0]}x{img.size[1]}")
                        print(f"  [block {k}] saved original (downloaded): image_{k}_original.png")
                    else:
                        meta_lines.append("original_download_failed=1")
            elif info["kind"] == "draw":
                sp = info.get("saved_path", "")
                local = tool_resp_dir / os.path.basename(sp)
                meta_lines.append(f"saved_path={sp}")
                meta_lines.append(f"resolved_local={local}")
                if local.exists():
                    shutil.copy(local, sample_dir / f"image_{k}_original.png")
                    img = Image.open(local)
                    meta_lines.append(f"original_size={img.size[0]}x{img.size[1]}")
                    print(f"  [block {k}] copied draw original: {local}")
                else:
                    meta_lines.append("original_missing=1")
        else:
            # Probably the final draw block (no [IMAGE N] before it).
            meta_lines.append("global_tag=(final draw — no [IMAGE N] tag)")
            # Pick the last saved_path we know.
            if originals:
                last_key = max(originals.keys())
                if originals[last_key]["kind"] == "draw":
                    sp = originals[last_key]["saved_path"]
                    local = tool_resp_dir / os.path.basename(sp)
                    meta_lines.append(f"saved_path={sp}")
                    meta_lines.append(f"resolved_local={local}")
                    if local.exists():
                        shutil.copy(local, sample_dir / f"image_{k}_original.png")
                        img = Image.open(local)
                        meta_lines.append(f"original_size={img.size[0]}x{img.size[1]}")
                        print(f"  [block {k}] copied draw original (final block)")
                    else:
                        meta_lines.append("original_missing=1")
        (sample_dir / f"image_{k}_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    # Walkthrough markdown.
    walk = _emit_walkthrough(text, block_sizes, tags, originals, sample_dir)
    (sample_dir / "README.md").write_text(walk, encoding="utf-8")
    (sample_dir / "text.txt").write_text(_summarize_text(text), encoding="utf-8")
    # Also dump a structural snapshot so the user can sanity-check
    # the BSS/BoG/tool_call layout without scrolling 400k chars.
    struct: List[str] = []
    for ev_re, name in [
        (re.escape(BOS), "BOS"),
        (re.escape(BSS), "BSS"),
        (re.escape(ESS), "ESS"),
        (re.escape(BOG), "BoG"),
        (re.escape(EOG), "EoG"),
        (re.escape(BOC), "BoC"),
        (re.escape(EOC), "EoC"),
        (re.escape(BOI), "BoI"),
        (re.escape(EOI), "EoI"),
        (re.escape(EOS), "EOS"),
        (r"<tool_call>", "tool_call_open"),
        (r"</tool_call>", "tool_call_close"),
        (r"<tool_response>", "tool_response_open"),
        (r"</tool_response>", "tool_response_close"),
    ]:
        for m in re.finditer(ev_re, text):
            struct.append((m.start(), name))
    struct.sort()
    (sample_dir / "structure.txt").write_text(
        "\n".join(f"{pos:>8d}  {name}" for pos, name in struct) + "\n",
        encoding="utf-8",
    )

    print(f"  wrote walkthrough → {sample_dir / 'README.md'}")


# ── jsonl helpers ──────────────────────────────────────────────────────────
def load_lineno_index(path: str, wanted: List[int]) -> Dict[int, Dict[str, Any]]:
    """Load only the requested 1-based line numbers from a JSONL file."""
    wanted_set = set(wanted)
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i in wanted_set:
                line = line.strip()
                if line:
                    out[i] = json.loads(line)
                if len(out) == len(wanted_set):
                    break
    return out


def load_sft_by_lineno(path: str, wanted: List[int]) -> Dict[int, Dict[str, Any]]:
    """sft.jsonl after merge is sorted by meta.line_no — but we don't rely on
    that. Linear scan, filter by meta.line_no."""
    wanted_set = set(wanted)
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ln = int(rec.get("meta", {}).get("line_no", 0))
            if ln in wanted_set:
                out[ln] = rec
                if len(out) == len(wanted_set):
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="merged sft.jsonl")
    ap.add_argument("--raw", required=True, help="UPE_raw.jsonl")
    ap.add_argument("--tool-resp-dir", required=True)
    ap.add_argument("--vq-path", required=True)
    ap.add_argument("--vq-type", default="ibq")
    ap.add_argument("--vq-device", default="cuda:0")
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--out", required=True, help="output dir for verification artifacts")
    ap.add_argument("--line-nos", type=int, nargs="+", required=True,
                    help="1-based meta.line_no values to verify")
    ap.add_argument("--skip-downloads", action="store_true",
                    help="Don't fetch URL originals (still copies draw originals).")
    args = ap.parse_args()

    from src.vision_tokenizer import build_vision_tokenizer
    vq_model = build_vision_tokenizer(args.vq_type, args.vq_path, device=args.vq_device)
    vq_model.eval()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    raw_recs = load_lineno_index(args.raw, args.line_nos)
    sft_recs = load_sft_by_lineno(args.sft, args.line_nos)
    print(f"raw records found:  {sorted(raw_recs.keys())}")
    print(f"sft records found:  {sorted(sft_recs.keys())}")

    tool_resp_dir = Path(args.tool_resp_dir)
    for ln in args.line_nos:
        if ln not in raw_recs:
            print(f"[skip] line {ln}: missing in raw"); continue
        if ln not in sft_recs:
            print(f"[skip] line {ln}: missing in sft (maybe filtered out — e.g. no successful draw)"); continue
        verify_one(
            raw_recs[ln], sft_recs[ln], ln, tool_resp_dir, out_root, vq_model,
            skip_downloads=args.skip_downloads,
        )

    print("\nDone. Artifacts under:", out_root)


if __name__ == "__main__":
    main()
