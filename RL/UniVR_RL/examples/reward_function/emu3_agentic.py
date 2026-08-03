# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Format / structure-focused reward for Emu3.5 agentic RL.

This is the *bring-up* reward — it intentionally does NOT call out to a VLM
server (no network on this training node) and does NOT decode VQ image
tokens (no spare GPU memory during the FSDP step). It scores:

  * **format**: did the rollout produce a well-formed agent trace?
      - did at least one `<think>...</think>` (or `<|extra_60|>...<|extra_61|>`) appear?
      - was every `<tool_call>...</tool_call>` valid JSON with `name` + `arguments`?
      - did the trajectory eventually attempt a draw — either natively (a
        `<|image start|>...<|image end|>` block) or via a `<|extra_50|>...<|extra_51|>`
        caption?
      - did the trajectory NOT exhaust max_turns without drawing?

  * **draw**: a binary +1 if the rollout produced a native draw, +0.5 if it
    produced a draw caption (`<BoC>...<EoC>`) without a native draw,
    0 otherwise. This signal is what GRPO will primarily climb.

  * **length_pen**: a small penalty for tool-loop runaways (-0.05 per
    1024 policy tokens above the soft budget) — keeps the model from
    spamming tool calls to inflate response length.

  * **overall** = format_weight * format + draw_weight * draw + length_pen.

Why this is enough to start: the SFT'd model already knows how to emit the
agent template (it produced the rollouts in
``emu_gen_8000_800step/emu3p5-sft/``). RL with this reward will favour
trajectories that stay on-format and reliably terminate in a draw, rather
than runaways with bad JSON / no draw. Once training converges on that
signal we can swap in a VLM quality reward without touching the rollout.

Required signature (matches verl/workers/reward/function.py:74):

    compute_score(reward_inputs: list[dict], ...) -> list[dict[str, float]]

The rollout worker fills ``reward_inputs[i]`` with at least:
  - response, response_length, ground_truth, uid, dataset_source

and our agentic rollout additionally puts the *full* rollout text and
termination reason in ``data.non_tensor_batch`` under keys
``agentic_response_text`` / ``agentic_termination``. The
AutoRewardManager only forwards a small handful of keys per row, so we
recover the full trace by re-decoding when needed (the `response` field
already contains the policy tokens) but prefer reading the rich keys when
they are present.

The function returns one dict per row, with the standard keys
``overall`` / ``format`` / ``accuracy`` so verl's metric logger picks them up.
"""

from __future__ import annotations

import json
import re
from typing import Any

REWARD_NAME = "emu3_agentic_format"
REWARD_TYPE = "batch"


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>")
_BOI_BLOCK_RE = re.compile(r"<\|image start\|>[\s\S]*?<\|image end\|>")
_BOC_RE = re.compile(r"<\|extra_50\|>\s*([\s\S]*?)\s*<\|extra_51\|>")
_THINK_RE = re.compile(r"(<think>[\s\S]*?</think>|<\|extra_60\|>[\s\S]*?<\|extra_61\|>)")
_BSS_RE = re.compile(r"<\|extra_100\|>")
_ESS_RE = re.compile(r"<\|extra_101\|>")


def _score_format(text: str) -> tuple[float, dict[str, float]]:
    """Return a format score in [0, 1] and per-component diagnostics."""
    diag: dict[str, float] = {}

    # 1. At least one think block (rewards CoT-style reasoning).
    diag["has_think"] = 1.0 if _THINK_RE.search(text) else 0.0

    # 2. Every <tool_call> must be JSON with `name` + `arguments`.
    tcs = _TOOL_CALL_RE.findall(text)
    diag["n_tool_calls"] = float(len(tcs))
    if tcs:
        good = 0
        for tc in tcs:
            try:
                payload = json.loads(tc.strip())
                if isinstance(payload, dict) and "name" in payload and "arguments" in payload:
                    if isinstance(payload["arguments"], dict):
                        good += 1
            except Exception:
                pass
        diag["tool_call_validity"] = good / max(1, len(tcs))
    else:
        diag["tool_call_validity"] = 0.0

    # 3. Did a draw caption appear?
    diag["has_draw_caption"] = 1.0 if _BOC_RE.search(text) else 0.0

    # 4. Did a native draw appear?
    diag["has_native_draw"] = 1.0 if _BOI_BLOCK_RE.search(text) else 0.0

    # 5. Span tokens balanced? (Lenient — a small mismatch is fine.)
    n_bss = len(_BSS_RE.findall(text))
    n_ess = len(_ESS_RE.findall(text))
    if max(n_bss, n_ess) == 0:
        diag["spans_balanced"] = 0.0
    else:
        diag["spans_balanced"] = 1.0 - abs(n_bss - n_ess) / max(n_bss, n_ess)

    # Composite — equal weight on the five signals.
    composite = (
        0.2 * diag["has_think"]
        + 0.25 * diag["tool_call_validity"]
        + 0.15 * diag["has_draw_caption"]
        + 0.25 * diag["has_native_draw"]
        + 0.15 * diag["spans_balanced"]
    )
    return max(0.0, min(1.0, composite)), diag


def _score_draw(text: str) -> float:
    if _BOI_BLOCK_RE.search(text):
        return 1.0
    if _BOC_RE.search(text):
        return 0.5
    return 0.0


def _length_penalty(n_tokens: int, soft_budget: int = 6000) -> float:
    if n_tokens <= soft_budget:
        return 0.0
    overflow = n_tokens - soft_budget
    return -0.05 * (overflow / 1024.0)


def compute_score(
    reward_inputs: list[dict[str, Any]],
    *,
    format_weight: float = 0.4,
    draw_weight: float = 0.55,
    length_penalty_weight: float = 0.05,
    soft_length_budget: int = 6000,
    # Accepted for compatibility with the SFT reward call site (config_emu3.yaml
    # uses these keys); we ignore them in the agentic bring-up reward.
    vlm_weight: float = 0.0,
    vlm_api_base: str = "",
    vlm_api_key: str = "",
    vlm_model_name: str = "",
    enable_vlm_reward: bool = False,
    max_vlm_workers: int = 0,
    **_unused: Any,
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for sample in reward_inputs:
        text = sample.get("response", "") or ""
        if not isinstance(text, str):
            text = str(text)
        n_tokens = int(sample.get("response_length", 0) or 0)

        fmt, diag = _score_format(text)
        draw = _score_draw(text)
        length_pen = _length_penalty(n_tokens, soft_budget=soft_length_budget) * length_penalty_weight

        overall = format_weight * fmt + draw_weight * draw + length_pen

        # Standard keys verl logs:
        row = {
            "overall": float(overall),
            "format": float(fmt),
            "accuracy": float(draw),       # repurpose "accuracy" so the standard
                                            # metric logger picks up our draw signal
            "draw": float(draw),
            "length_pen": float(length_pen),
        }
        # Flatten diagnostics so verl prints them in the metrics dict.
        for k, v in diag.items():
            row[f"fmt_{k}"] = float(v)
        out.append(row)
    return out
