"""Open image references (URL / data: / local path) and VQ-encode them into
Emu3.5 visual-token strings (BOI ... EOI) using the project's vq tokenizer.

The encoded string is what gets spliced into the next `<tool_response>` block
so the model sees the actual image in its input context.
"""
from __future__ import annotations

import base64
import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image


_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "close",
}


def open_image(ref: str, timeout: int = 30, retries: int = 3) -> Optional[Image.Image]:
    """Return a PIL.Image.RGB or None on any failure.

    Supports `data:image/...;base64,...`, `http(s)://...`, and local
    filesystem paths.
    """
    if not ref:
        return None
    if ref.startswith("data:image"):
        try:
            _, payload = ref.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        except Exception as exc:
            print(f"[image_utils] data-uri decode failed: {exc}")
            return None
    if ref.startswith(("http://", "https://")):
        return _download_image(ref, timeout=timeout, retries=retries)
    try:
        return Image.open(ref).convert("RGB")
    except Exception as exc:
        print(f"[image_utils] local open failed {ref}: {exc}")
        return None


def _download_image(url: str, timeout: int, retries: int) -> Optional[Image.Image]:
    parsed = urlparse(url)
    headers = dict(_DOWNLOAD_HEADERS)
    headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"

    last_exc: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            last_exc = exc
            time.sleep(0.3 * (attempt + 1))
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        last_exc = exc
    print(f"[image_utils] download failed {url}: {last_exc}")
    return None


def encode_to_emu_tokens(image: Image.Image, cfg, tokenizer, vq_model) -> str:
    """Wrap `src.utils.input_utils.build_image` — returns
    `<BoI>H*W<IMG_TOKEN>visual_tokens<EoI>` as a single string."""
    from src.utils.input_utils import build_image  # type: ignore

    return build_image(image, cfg, tokenizer, vq_model)


def fetch_many(refs: List[str], workers: int = 4, timeout: int = 30) -> Dict[str, Optional[Image.Image]]:
    """Concurrently fetch a list of refs. Returns ref → Image (or None on failure).

    Encoding (VQ) MUST stay serial on the GPU, but downloads can parallelize.
    """
    out: Dict[str, Optional[Image.Image]] = {}
    todo = [r for r in dict.fromkeys(refs) if r]
    if not todo:
        return out
    workers = min(max(1, workers), len(todo))
    if workers == 1:
        for r in todo:
            out[r] = open_image(r, timeout=timeout)
        return out
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(open_image, r, timeout): r for r in todo}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                out[r] = fut.result()
            except Exception:
                out[r] = None
    return out


__all__ = ["open_image", "encode_to_emu_tokens", "fetch_many"]
