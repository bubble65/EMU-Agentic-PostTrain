# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""In-process Emu3.5 server used by the RL rollout worker.

The numeric constants mirror ``Agentic_Image_Gen/server.py`` so online RL
rollout and standalone rollout service use the same sampling behavior.

Surface:
  * ``encode_images(refs)``   →  same shape as server.py /encode_images
  * ``generate(prompt, **kw)`` →  same shape as server.py /generate
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from . import path_setup  # noqa: F401  — side-effect: sys.path → emu_infer_fast / EMU_ROOT

from image_utils import encode_to_emu_tokens, open_image  # type: ignore  # noqa: E402
from template import SPECIAL  # type: ignore  # noqa: E402
from decode_utils import decode_generated  # type: ignore  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════
#  Sampling constants mirrored from Agentic_Image_Gen/server.py.
# ═════════════════════════════════════════════════════════════════════════
_PROJECT_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_IMAGE_SAVE_DIR = os.environ.get(
    "EMU_IMAGE_SAVE_DIR",
    str(_PROJECT_ROOT / "RL" / "experiments" / "images"),
)
DEFAULT_MAX_NEW_TOKENS = 8192            # server.py:137
TEXT_CFG = 1.0                           # server.py:138 — text pass no CFG
IMAGE_CFG = 3.0                          # server.py:139 — Emu3.5 interleaved CFG=3
MAX_IMAGE_TOKENS = 8192                  # server.py:140
IMAGE_AREA = 1048576                     # server.py:132 — 1024×1024


def _resolve_int(req_val, default_val):
    return int(req_val) if req_val is not None else int(default_val)


def _resolve_float(req_val, default_val):
    return float(req_val) if req_val is not None else float(default_val)


def _build_cfg(tokenizer) -> SimpleNamespace:
    """Verbatim copy of server.py::_build_cfg (lines 149-176)."""
    cfg = SimpleNamespace(
        image_area=IMAGE_AREA,
        target_height=None,
        target_width=None,
        classifier_free_guidance=IMAGE_CFG,
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


class InProcessEmuServer:
    """Verbatim port of emu_infer_fast/server.py's two endpoints.

    Every method body is copied line-for-line from the scaffold; only the
    differences are:
      * ``STATE[...]`` lookups → ``self.<attr>``
      * the FastAPI request object → a plain ``overrides`` dict (we pre-build
        a ``SimpleNamespace`` with the right field names so the original
        ``req.text_top_k`` access patterns keep working).

    Nothing about the model, sampling, CFG, or pass ordering changes.
    """

    def __init__(self, model, tokenizer, vq_model):
        self.model = model
        self.tokenizer = tokenizer
        self.vq_model = vq_model
        self.cfg = _build_cfg(tokenizer)
        self.bos_id = self.cfg.special_token_ids["BOS"]
        self.bss_id = self.cfg.special_token_ids["BSS"]
        self.ess_id = self.cfg.special_token_ids["ESS"]
        self.eos_id = self.cfg.special_token_ids["EOS"]
        self.boi_id = self.cfg.special_token_ids["BOI"]
        self.eoi_id = self.cfg.special_token_ids["EOI"]
        # Stop-id sets — verbatim from server.py lifespan() (lines 216-220)
        self.text_stop = [self.boi_id, self.ess_id, self.eos_id]
        self.image_stop = [self.eoi_id]
        self.after_image_stop = [self.ess_id, self.eos_id]
        self._gen_lock = threading.Lock()
        os.makedirs(DEFAULT_IMAGE_SAVE_DIR, exist_ok=True)
        print(
            "[InProcessEmuServer] ready (hard-coded scaffold constants: "
            f"IMAGE_CFG={IMAGE_CFG}, TEXT_CFG={TEXT_CFG}, "
            f"IMAGE_AREA={IMAGE_AREA}, MAX_IMAGE_TOKENS={MAX_IMAGE_TOKENS}, "
            f"DEFAULT_MAX_NEW_TOKENS={DEFAULT_MAX_NEW_TOKENS}, "
            f"visual_top_k=5120, visual_top_p=1.0, visual_temperature=1.0)",
            flush=True,
        )

    # ── /encode_images — verbatim copy of server.py lines 242-269 ────────
    def encode_images(self, refs: List[str]) -> Dict[str, Any]:
        tokens: List[str] = []
        ok_flags: List[bool] = []
        errors: List[str] = []
        for ref in refs:
            img = open_image(ref)
            if img is None:
                tokens.append("")
                ok_flags.append(False)
                errors.append("open failed")
                continue
            try:
                t = encode_to_emu_tokens(img, self.cfg, self.tokenizer, self.vq_model)
                tokens.append(t)
                ok_flags.append(True)
                errors.append("")
            except Exception as exc:
                tokens.append("")
                ok_flags.append(False)
                errors.append(f"vq encode failed: {exc}")
        return {"tokens": tokens, "ok": ok_flags, "errors": errors}

    # ── sampling-param builders — verbatim copies of server.py lines 299-364
    def _make_text_sp(self, req, stop_ids: List[int], max_tokens: int):
        from vllm import SamplingParams
        sp = self.cfg.sampling_params
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

    def _make_image_sp(self, req):
        from vllm import SamplingParams
        sp = self.cfg.sampling_params
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
            stop_token_ids=[self.eoi_id],
        )

    def _vllm_generate(self, prompt_ids: List[int], sampling_params) -> List[int]:
        """Verbatim copy of server.py::_vllm_generate (lines 367-376).

        The patched sampler reads ``uncond_prompt_token_ids`` for its CFG
        implementation — without this key the visual pass falls back to
        guidance_scale=1 and you get the garbled "noisy-paint" failure mode.
        """
        results = self.model.generate(
            {"prompt_token_ids": prompt_ids, "uncond_prompt_token_ids": prompt_ids},
            sampling_params=sampling_params,
        )
        return list(results[0].outputs[0].token_ids)

    # ── /generate — verbatim copy of server.py lines 379-479 ─────────────
    def generate(self, prompt: str, **overrides) -> Dict[str, Any]:
        # Build a server.py-shaped request object so the inner methods can
        # use the same ``req.text_top_k`` style attribute access.
        req = SimpleNamespace(
            max_new_tokens=overrides.get("max_new_tokens"),
            text_temperature=overrides.get("text_temperature"),
            text_top_p=overrides.get("text_top_p"),
            text_top_k=overrides.get("text_top_k"),
            image_temperature=overrides.get("image_temperature"),
            image_top_p=overrides.get("image_top_p"),
            image_top_k=overrides.get("image_top_k"),
            image_save_dir=overrides.get("image_save_dir"),
            allow_native_image=overrides.get("allow_native_image", True),
            return_raw=overrides.get("return_raw", False),
            force_image_first=overrides.get("force_image_first", False),
            skip_post_image=overrides.get("skip_post_image", False),
        )

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not input_ids or input_ids[0] != self.bos_id:
            input_ids = [self.bos_id] + input_ids

        budget = req.max_new_tokens or self.cfg.sampling_params["max_new_tokens"]

        produced: List[int] = []
        stopped_on_boi = False
        stopped_on_eoi = False

        with self._gen_lock:
            if req.force_image_first:
                sp_img = self._make_image_sp(req)
                out_img = self._vllm_generate(input_ids, sp_img)
                try:
                    last_boi = len(input_ids) - 1 - input_ids[::-1].index(self.boi_id)
                    produced = input_ids[last_boi:] + out_img
                except ValueError:
                    produced = list(out_img)
                stopped_on_boi = True
                stopped_on_eoi = bool(out_img) and out_img[-1] == self.eoi_id

                remain = max(1, budget - len(out_img))
                if stopped_on_eoi and not req.skip_post_image:
                    prompt3 = input_ids + out_img
                    sp3 = self._make_text_sp(req, self.after_image_stop, max_tokens=remain)
                    out_post = self._vllm_generate(prompt3, sp3)
                    produced.extend(out_post)

            else:
                # ── Pass-1: text, stop @ BOI / ESS / EOS ──────────────
                if req.allow_native_image:
                    stop_pass1 = self.text_stop
                else:
                    stop_pass1 = self.after_image_stop
                sp1 = self._make_text_sp(req, stop_pass1, max_tokens=budget)
                out1 = self._vllm_generate(input_ids, sp1)
                produced = list(out1)
                remain = max(1, budget - len(out1))
                stopped_on_boi = bool(out1) and out1[-1] == self.boi_id

                # ── Pass-2: image, stop @ EOI ─────────────────────────
                if stopped_on_boi and req.allow_native_image:
                    prompt2 = input_ids + out1
                    sp_img = self._make_image_sp(req)
                    out_img = self._vllm_generate(prompt2, sp_img)
                    produced.extend(out_img)
                    remain = max(1, remain - len(out_img))
                    stopped_on_eoi = bool(out_img) and out_img[-1] == self.eoi_id

                    # ── Pass-3: post-image text, stop @ ESS / EOS ─────
                    if stopped_on_eoi and not req.skip_post_image:
                        prompt3 = prompt2 + out_img
                        sp3 = self._make_text_sp(req, self.after_image_stop, max_tokens=remain)
                        out_post = self._vllm_generate(prompt3, sp3)
                        produced.extend(out_post)

        decoded = decode_generated(
            produced,
            tokenizer=self.tokenizer,
            vq_model=self.vq_model,
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


__all__ = ["InProcessEmuServer", "DEFAULT_IMAGE_SAVE_DIR"]
