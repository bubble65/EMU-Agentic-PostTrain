"""Bare-bones vLLM Emu3.5 inference server.

Two endpoints + /health:

  POST /encode_images
    in : {"refs": ["url-or-path", ...]}
    out: {"tokens": [str, ...], "ok": [bool, ...], "errors": [str, ...]}
    Each successful entry is the full "<BoI>H*W<IMG>v..v<EoI>" string the
    agent splices into the next <tool_response>.

  POST /generate
    in : {
        "prompt": str,                   # raw emu-formatted string from run.py
        "max_new_tokens": int|null,
        "text_temperature": float|null,
        "text_top_p":      float|null,
        "text_top_k":      int|null,
        "image_temperature": float|null,
        "image_top_p":       float|null,
        "image_top_k":       int|null,
        "image_save_dir": str|null,      # where to save decoded PNG(s)
        "allow_native_image": bool,      # if False, only do text pass (stop @ ESS/EOS)
        "force_image_first": bool,       # skip pass-1 text and go straight to image
        "skip_post_image": bool,         # after EOI, do NOT do a post-image text pass
        "return_raw": bool,
    }
    out: {
        "text":          str,    # BSS/ESS/EOS stripped, special tokens kept
        "content":       str,    # `text` with BOI..EOI → "[generated_image: /path]"
        "saved_images":  [str, ...],
        "prompt_tokens": int,
        "output_tokens": int,
        "stopped_on_boi": bool,  # pass-1 stopped because the model produced BoI
        "stopped_on_eoi": bool,  # pass-2 (image) closed cleanly with EoI
    }

  GET /health  →  {"status": "ok"|"loading"}

Multi-pass generation (text → image → optional post-image text) lives here
because it's a vLLM sampling concern: pass 1 stops at BoI/ESS/EOS, pass 2
swaps in the IMAGE-mode sampling params (image_top_k / image_temperature)
and stops at EoI, optional pass 3 wraps up with the post-image trailing text
(`<ESS>` etc.). The agent (run.py) hands us the assembled prompt and never
worries about sampling-mode transitions.

Env overrides (see start_server.sh):
  EMU_MODEL_PATH, EMU_TOKENIZER_PATH, EMU_VQ_PATH, EMU_TP_SIZE,
  EMU_GPU_MEM_UTIL, EMU_VQ_DEVICE, EMU_HOST, EMU_PORT,
  EMU_IMAGE_SAVE_DIR, EMU_MAX_NEW_TOKENS, EMU_IMAGE_AREA,
  EMU_TEXT_CFG, EMU_IMAGE_CFG, EMU_MAX_IMAGE_TOKENS, EMU_SEED.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

# ── Blackwell sm_103a / triton 3.4 + CUDA 13 workaround ────────────────────
# Triton 3.4 bundled with vllm 0.11.0 ships ptxas 12.8, and its
# ptx_get_version() raises on CUDA 13.x. Patch it BEFORE vllm/triton compile.
try:
    import triton  # noqa: E402
    import triton.backends.nvidia.compiler as _tc  # noqa: E402
    import functools as _functools  # noqa: E402

    _orig_ptx = getattr(_tc.ptx_get_version, "__wrapped__", _tc.ptx_get_version)

    @_functools.lru_cache()
    def _patched_ptx_get_version(cuda_version: str) -> int:
        try:
            major, minor = map(int, cuda_version.split("."))
        except Exception:
            return 87
        if major == 13:
            return 90 + max(0, min(2, minor))
        if callable(_orig_ptx):
            return _orig_ptx(cuda_version)
        return 87

    _tc.ptx_get_version = _patched_ptx_get_version
    if hasattr(_tc, "get_ptxas_version"):
        _tc.get_ptxas_version.cache_clear()
except Exception as _shim_exc:
    print(f"[emu_server] triton shim skipped: {_shim_exc}", flush=True)
# ── end shim ──────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402


def _install_emu3_tokenizer_vllm_compat() -> None:
    """Expose the tokenizer property required by vLLM's cache wrapper.

    The bundled Emu3 slow tokenizer stores its full special-token table in
    `special_tokens`, but the current vLLM version also reads the newer
    Transformers property `all_special_tokens_extended` during startup.
    """
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            return

        @property
        def all_special_tokens_extended(self):
            special_tokens = getattr(self, "special_tokens", None)
            if isinstance(special_tokens, dict) and special_tokens:
                return list(special_tokens.keys())
            return list(getattr(self, "all_special_tokens", []))

        PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended
    except Exception as exc:
        print(f"[emu_server] tokenizer compat shim skipped: {exc}", flush=True)


_install_emu3_tokenizer_vllm_compat()


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# Resolve Emu3.5 source root (so we can import src.utils.*).
EMU_ROOT_DEFAULT = PROJECT_ROOT / "Emu3.5"
EMU_ROOT = Path(os.environ.get("EMU_ROOT", str(EMU_ROOT_DEFAULT)))
if str(EMU_ROOT) not in sys.path:
    sys.path.insert(0, str(EMU_ROOT))

if str(THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parent))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from decode_utils import decode_generated  # noqa: E402
from image_utils import encode_to_emu_tokens, open_image  # noqa: E402
from template import SPECIAL  # noqa: E402


# ── Config from env ──────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "EMU_MODEL_PATH",
    str(PROJECT_ROOT / "checkpoints" / "emu3p5-sft"),
)
TOKENIZER_PATH = os.environ.get(
    "EMU_TOKENIZER_PATH",
    str(EMU_ROOT / "src" / "tokenizer_emu3_ibq"),
)
VQ_PATH = os.environ.get(
    "EMU_VQ_PATH",
    str(PROJECT_ROOT / "checkpoints" / "Emu3.5-VisionTokenizer"),
)
VQ_TYPE = os.environ.get("EMU_VQ_TYPE", "ibq")
VQ_DEVICE = os.environ.get("EMU_VQ_DEVICE", "cuda:0")
TP_SIZE = int(os.environ.get("EMU_TP_SIZE", "2"))
GPU_MEM_UTIL = float(os.environ.get("EMU_GPU_MEM_UTIL", "0.7"))
SEED = int(os.environ.get("EMU_SEED", "6666"))
IMAGE_AREA = int(os.environ.get("EMU_IMAGE_AREA", "1048576"))
DEFAULT_IMAGE_SAVE_DIR = os.environ.get(
    "EMU_IMAGE_SAVE_DIR",
    str(THIS_DIR / "workspace" / "images"),
)
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("EMU_MAX_NEW_TOKENS", "8192"))
TEXT_CFG = float(os.environ.get("EMU_TEXT_CFG", "1.0"))
IMAGE_CFG = float(os.environ.get("EMU_IMAGE_CFG", "3.0"))  # Emu3.5 interleaved recommendation
MAX_IMAGE_TOKENS = int(os.environ.get("EMU_MAX_IMAGE_TOKENS", "8192"))


STATE: Dict[str, Any] = {}
# vLLM is internally batched, but our 2-pass (text→image) workflow needs the
# image pass to see EXACTLY the prompt+pass1 tokens — serialize requests.
_GEN_LOCK = threading.Lock()


def _build_cfg(tokenizer) -> SimpleNamespace:
    """Build the cfg object expected by src/utils/input_utils.py::build_image."""
    cfg = SimpleNamespace(
        image_area=IMAGE_AREA,
        target_height=None,
        target_width=None,
        classifier_free_guidance=IMAGE_CFG,
        # task_type only matters for the legacy generate(); we run our own
        # passes so we don't actually use it.
        task_type="howto",
        sampling_params={
            "use_cache": True,
            "text_top_k": 1024,
            "text_top_p": 0.9,
            "text_temperature": 1.0,
            "image_top_k": 5120,
            "image_top_p": 1.0,
            "image_temperature": 1.0,
            "top_k": 131072,
            "top_p": 1.0,
            "temperature": 1.0,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        },
    )
    cfg.special_token_ids = {
        name.upper(): tokenizer.encode(tok)[0] for name, tok in SPECIAL.items()
    }
    return cfg


# ── FastAPI lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.utils.model_utils import build_emu3p5_vllm  # type: ignore

    print(
        f"[emu_server] loading\n"
        f"  model_path={MODEL_PATH}\n"
        f"  tokenizer_path={TOKENIZER_PATH}\n"
        f"  vq_path={VQ_PATH}\n"
        f"  tp_size={TP_SIZE}, gpu_mem_util={GPU_MEM_UTIL}\n"
        f"  vq_device={VQ_DEVICE}\n",
        flush=True,
    )
    model, tokenizer, vq_model = build_emu3p5_vllm(
        MODEL_PATH,
        TOKENIZER_PATH,
        VQ_PATH,
        vq_type=VQ_TYPE,
        vq_device=VQ_DEVICE,
        tensor_parallel_size=TP_SIZE,
        gpu_memory_utilization=GPU_MEM_UTIL,
        seed=SEED,
    )
    cfg = _build_cfg(tokenizer)
    STATE["model"] = model
    STATE["tokenizer"] = tokenizer
    STATE["vq_model"] = vq_model
    STATE["cfg"] = cfg
    STATE["bos_id"] = cfg.special_token_ids["BOS"]
    STATE["bss_id"] = cfg.special_token_ids["BSS"]
    STATE["ess_id"] = cfg.special_token_ids["ESS"]
    STATE["eos_id"] = cfg.special_token_ids["EOS"]
    STATE["boi_id"] = cfg.special_token_ids["BOI"]
    STATE["eoi_id"] = cfg.special_token_ids["EOI"]
    # Pass-1 (text mode): stop on any of BOI (model wants to draw), ESS
    # (assistant span ended → tool_call or just done thinking), EOS.
    STATE["text_stop"] = [STATE["boi_id"], STATE["ess_id"], STATE["eos_id"]]
    # Pass-2 (image mode): stop on EOI (image done).
    STATE["image_stop"] = [STATE["eoi_id"]]
    # Pass-3 (post-image text): stop on ESS or EOS so we don't run away.
    STATE["after_image_stop"] = [STATE["ess_id"], STATE["eos_id"]]
    os.makedirs(DEFAULT_IMAGE_SAVE_DIR, exist_ok=True)
    print("[emu_server] ready", flush=True)
    yield
    STATE.clear()


app = FastAPI(lifespan=lifespan)


# ── /health ──────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok" if STATE.get("model") is not None else "loading"}


# ── /encode_images ───────────────────────────────────────────────────────
class EncodeImagesRequest(BaseModel):
    refs: List[str]


@app.post("/encode_images")
def encode_images(req: EncodeImagesRequest) -> Dict[str, Any]:
    if STATE.get("vq_model") is None:
        raise HTTPException(503, "model not ready")

    tokenizer = STATE["tokenizer"]
    vq_model = STATE["vq_model"]
    cfg = STATE["cfg"]

    tokens: List[str] = []
    ok_flags: List[bool] = []
    errors: List[str] = []
    for ref in req.refs:
        img = open_image(ref)
        if img is None:
            tokens.append("")
            ok_flags.append(False)
            errors.append("open failed")
            continue
        try:
            t = encode_to_emu_tokens(img, cfg, tokenizer, vq_model)
            tokens.append(t)
            ok_flags.append(True)
            errors.append("")
        except Exception as exc:
            tokens.append("")
            ok_flags.append(False)
            errors.append(f"vq encode failed: {exc}")
    return {"tokens": tokens, "ok": ok_flags, "errors": errors}


# ── /generate ────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = None
    # Text-mode sampling overrides (only apply during pass-1/pass-3)
    text_temperature: Optional[float] = None
    text_top_p: Optional[float] = None
    text_top_k: Optional[int] = None
    # Image-mode sampling overrides (only apply during pass-2)
    image_temperature: Optional[float] = None
    image_top_p: Optional[float] = None
    image_top_k: Optional[int] = None
    image_save_dir: Optional[str] = None
    allow_native_image: bool = True
    return_raw: bool = False
    force_image_first: bool = False
    skip_post_image: bool = False


def _resolve_int(req_val, default_val):
    return int(req_val) if req_val is not None else int(default_val)


def _resolve_float(req_val, default_val):
    return float(req_val) if req_val is not None else float(default_val)


def _make_text_sp(req: GenerateRequest, cfg, stop_ids: List[int], max_tokens: int):
    """Sampling params for a TEXT-mode pass.

    The Emu3.5 patched vllm understands `extra_args` and applies the right
    differential-top-k branch based on the previously generated token, so we
    just feed both text and visual params; the sampler picks based on context.
    Stop-on-BoI in pass-1 is the official handoff signal to pass-2.
    """
    from vllm import SamplingParams

    sp = cfg.sampling_params
    extra_args = {
        "guidance_scale": TEXT_CFG,
        "text_top_k": _resolve_int(req.text_top_k, sp["text_top_k"]),
        "text_top_p": _resolve_float(req.text_top_p, sp["text_top_p"]),
        "text_temperature": _resolve_float(req.text_temperature, sp["text_temperature"]),
        "visual_top_k": _resolve_int(req.image_top_k, sp["image_top_k"]),
        "visual_top_p": _resolve_float(req.image_top_p, sp["image_top_p"]),
        "visual_temperature": _resolve_float(req.image_temperature, sp["image_temperature"]),
        "width": None,
        "height": None,
        "area": None,
    }
    return SamplingParams(
        top_k=sp["top_k"],
        top_p=sp["top_p"],
        temperature=sp["temperature"],
        max_tokens=max_tokens,
        detokenize=False,
        extra_args=extra_args,
        stop_token_ids=stop_ids,
    )


def _make_image_sp(req: GenerateRequest, cfg):
    """Sampling params for the IMAGE-mode pass (stops at EOI).

    Crucially, `area=IMAGE_AREA` is passed so the resolution logits processor
    inside the patched sampler emits "H*W" + IMG_TOKEN and the row token grid
    of the correct size. We also tighten the budget to MAX_IMAGE_TOKENS — a
    64×64 grid + delimiters is ~5–6 k tokens.
    """
    from vllm import SamplingParams

    sp = cfg.sampling_params
    extra_args = {
        "guidance_scale": IMAGE_CFG,
        "text_top_k": _resolve_int(req.text_top_k, sp["text_top_k"]),
        "text_top_p": _resolve_float(req.text_top_p, sp["text_top_p"]),
        "text_temperature": _resolve_float(req.text_temperature, sp["text_temperature"]),
        "visual_top_k": _resolve_int(req.image_top_k, sp["image_top_k"]),
        "visual_top_p": _resolve_float(req.image_top_p, sp["image_top_p"]),
        "visual_temperature": _resolve_float(req.image_temperature, sp["image_temperature"]),
        "width": None,
        "height": None,
        "area": IMAGE_AREA,
    }
    return SamplingParams(
        top_k=sp["top_k"],
        top_p=sp["top_p"],
        temperature=sp["temperature"],
        max_tokens=MAX_IMAGE_TOKENS,
        detokenize=False,
        extra_args=extra_args,
        stop_token_ids=[STATE["eoi_id"]],
    )


def _vllm_generate(model, prompt_ids: List[int], sampling_params) -> List[int]:
    """One pass through vllm. We pass the same ids as unconditional prompt:
    we are using guidance_scale == 1.0 (text) and CFG-style guidance on the
    image pass via the patched sampler's own unconditional logic; the patched
    sampler expects an `uncond_prompt_token_ids` key."""
    results = model.generate(
        {"prompt_token_ids": prompt_ids, "uncond_prompt_token_ids": prompt_ids},
        sampling_params=sampling_params,
    )
    return list(results[0].outputs[0].token_ids)


@app.post("/generate")
def generate(req: GenerateRequest) -> Dict[str, Any]:
    if STATE.get("model") is None:
        raise HTTPException(503, "model not ready")

    tokenizer = STATE["tokenizer"]
    vq_model = STATE["vq_model"]
    cfg = STATE["cfg"]
    model = STATE["model"]
    bos_id = STATE["bos_id"]
    boi_id = STATE["boi_id"]
    eoi_id = STATE["eoi_id"]

    input_ids = tokenizer.encode(req.prompt, add_special_tokens=False)
    if not input_ids or input_ids[0] != bos_id:
        input_ids = [bos_id] + input_ids

    budget = req.max_new_tokens or cfg.sampling_params["max_new_tokens"]

    produced: List[int] = []
    stopped_on_boi = False
    stopped_on_eoi = False

    with _GEN_LOCK:
        if req.force_image_first:
            # The caller has constructed `prompt` so it already ends right
            # before the visual-token stream — typically `...<BoC>caption<EoC>`
            # and an *open* `<BoI>` token. We skip pass-1 entirely and start
            # in image mode.
            #
            # For the agent to be able to vq-decode the result later, the
            # produced stream itself does not contain BOI (it's in the
            # prompt), so we re-prepend the BOI tail from the prompt to the
            # produced ids before passing to decode_generated. This guarantees
            # the full <BoI>...<EoI> block exists in `produced` for parsing.
            sp_img = _make_image_sp(req, cfg)
            out_img = _vllm_generate(model, input_ids, sp_img)
            try:
                last_boi = len(input_ids) - 1 - input_ids[::-1].index(boi_id)
                produced = input_ids[last_boi:] + out_img
            except ValueError:
                # No BOI in prompt — caller broke the contract. We still
                # decode what we got; the image block will simply be missing.
                produced = list(out_img)
            stopped_on_boi = True
            stopped_on_eoi = bool(out_img) and out_img[-1] == eoi_id

            remain = max(1, budget - len(out_img))
            if stopped_on_eoi and not req.skip_post_image:
                prompt3 = input_ids + out_img
                sp3 = _make_text_sp(req, cfg, STATE["after_image_stop"], max_tokens=remain)
                out_post = _vllm_generate(model, prompt3, sp3)
                produced.extend(out_post)

        else:
            # ── Pass-1: text generation, stop at BOI / ESS / EOS ──────────
            if req.allow_native_image:
                stop_pass1 = STATE["text_stop"]
            else:
                stop_pass1 = STATE["after_image_stop"]
            sp1 = _make_text_sp(req, cfg, stop_pass1, max_tokens=budget)
            out1 = _vllm_generate(model, input_ids, sp1)
            produced = list(out1)
            remain = max(1, budget - len(out1))
            stopped_on_boi = bool(out1) and out1[-1] == boi_id

            # ── Pass-2: image, stop at EOI ───────────────────────────────
            if stopped_on_boi and req.allow_native_image:
                prompt2 = input_ids + out1
                sp_img = _make_image_sp(req, cfg)
                out_img = _vllm_generate(model, prompt2, sp_img)
                produced.extend(out_img)
                remain = max(1, remain - len(out_img))
                stopped_on_eoi = bool(out_img) and out_img[-1] == eoi_id

                # ── Pass-3: post-image text, stop at ESS / EOS ───────────
                if stopped_on_eoi and not req.skip_post_image:
                    prompt3 = prompt2 + out_img
                    sp3 = _make_text_sp(req, cfg, STATE["after_image_stop"], max_tokens=remain)
                    out_post = _vllm_generate(model, prompt3, sp3)
                    produced.extend(out_post)

    decoded = decode_generated(
        produced,
        tokenizer=tokenizer,
        vq_model=vq_model,
        image_save_dir=req.image_save_dir or DEFAULT_IMAGE_SAVE_DIR,
    )

    resp: Dict[str, Any] = {
        "text": decoded["text"],
        "content": decoded["content"],
        "saved_images": decoded["saved_images"],
        "prompt_tokens": len(input_ids),
        "output_tokens": len(produced),
        "stopped_on_boi": stopped_on_boi,
        "stopped_on_eoi": stopped_on_eoi,
    }
    if req.return_raw:
        resp["raw_decoded"] = decoded["raw_decoded"]
    return resp


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("EMU_HOST", "0.0.0.0")
    port = int(os.environ.get("EMU_PORT", "23333"))
    uvicorn.run(app, host=host, port=port, log_level="info")
