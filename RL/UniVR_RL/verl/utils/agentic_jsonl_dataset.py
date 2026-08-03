# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Lightweight jsonl dataset for the ``emu_agentic`` rollout backend.

The agentic rollout (``verl/workers/rollout/emu_agentic``) drives the whole
multi-round agent itself: it reads the user's question from ``raw_prompt_ids``
and ignores any chat-template formatting that verl would otherwise apply. So
the dataset only needs to produce the BARE MINIMUM verl protocol expects:

  tensors:  input_ids, attention_mask, position_ids   (left-padded prompt)
  non-tensors: raw_prompt_ids, ground_truth, dataset_source

Each input row is a single jsonl object with a ``question`` field (this matches
``/jfs.../tmp/data/draw.jsonl`` etc. — the exact format the emu_infer_fast
scaffold consumes verbatim).

Why a new class rather than reusing ``Emu3RLHFDataset`` or ``RLHFDataset``:
  * ``Emu3RLHFDataset`` requires parquet rows with ``problem_images`` /
    ``answer_images`` VQ tokens — it is built for VR-style video prediction,
    not for agentic text→image generation.
  * ``RLHFDataset`` runs ``tokenizer.apply_chat_template`` which doesn't exist
    on the Emu3 IBQ tokenizer (custom tokenizer, no chat template).

So this class sits next to them and is selected by ``data_loader.py`` when
``data.train_files`` points at a ``.jsonl`` (and we're talking to the
``emu_agentic`` rollout backend).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from . import torch_functional as VF


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a single jsonl file or a directory of them into a flat list."""
    if os.path.isdir(path):
        rows: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(path)):
            if name.endswith(".jsonl"):
                rows.extend(_read_jsonl(os.path.join(path, name)))
        return rows

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _coerce_paths(data_path: Any) -> List[str]:
    """Accept str | list[str] | dict (values are paths) and flatten to list."""
    if isinstance(data_path, str):
        return [data_path]
    if isinstance(data_path, list):
        return [str(p) for p in data_path]
    if isinstance(data_path, dict):
        return [str(p) for p in data_path.values()]
    raise TypeError(f"Unsupported train_files type: {type(data_path)}")


class EmuAgenticJsonlDataset(Dataset):
    """Reads ``{"question": "..."}`` jsonl rows for ``EmuAgenticRollout``.

    The agentic rollout takes the raw user question from ``raw_prompt_ids``
    (via ``_decode_question_from_raw_ids``), so the prompt we hand verl is
    just ``"User: " + question.strip()``. The rest of the SFT framing
    (system message, BoS, BSS, …) is rebuilt inside ``EmuAgent`` so we MUST
    NOT pre-apply it here.

    Args:
        data_path: str | list[str] | dict — jsonl file(s) or directory.
        tokenizer: the Emu3 IBQ tokenizer (loaded by verl).
        prompt_key: jsonl field name that holds the question (default
            ``question``). For compatibility with ``--prompt_key problem``
            launchers, ``problem`` is tried as a fallback.
        max_prompt_length: hard cap.
    """

    def __init__(
        self,
        data_path: Any,
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "question",
        max_prompt_length: int = 4096,
        truncation: str = "right",
        dataset_source: str = "emu_agentic",
        max_samples: Optional[int] = None,
        **_ignored: Any,
    ):
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.dataset_source = dataset_source

        # The Emu3 IBQ tokenizer ships without pad_token/eos_token set. The
        # rollout's ``_load_emu_tokenizer`` patches the same fields; we do the
        # same here so verl's ``postprocess_data`` has a real pad id and the
        # GRPO loss masks line up.  The canonical SPECIAL ids come from
        # ``emu_infer_fast/template.py`` (re-exported by the rollout package).
        emu_specials = {
            "pad": "<|extra_0|>",
            "eos": "<|extra_204|>",
            "bos": "<|extra_203|>",
            "bss": "<|extra_100|>",
            "ess": "<|extra_101|>",
        }
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            try:
                self.tokenizer.pad_token = emu_specials["pad"]
            except Exception:
                pass
        if getattr(self.tokenizer, "eos_token_id", None) is None:
            try:
                self.tokenizer.eos_token = emu_specials["eos"]
            except Exception:
                pass
        if getattr(self.tokenizer, "bos_token_id", None) is None:
            try:
                self.tokenizer.bos_token = emu_specials["bos"]
            except Exception:
                pass
        self.pad_token_id = self.tokenizer.pad_token_id

        rows: List[Dict[str, Any]] = []
        for one_path in _coerce_paths(data_path):
            rows.extend(_read_jsonl(one_path))
        if not rows:
            raise FileNotFoundError(f"EmuAgenticJsonlDataset: no rows loaded from {data_path}")

        cleaned: List[Dict[str, Any]] = []
        for row in rows:
            q = row.get(prompt_key) or row.get("question") or row.get("problem")
            if not isinstance(q, str) or not q.strip():
                continue
            cleaned.append({"question": q.strip(), **row})
        if max_samples is not None and max_samples > 0:
            cleaned = cleaned[: max_samples]
        self._rows = cleaned
        print(
            f"[EmuAgenticJsonlDataset] loaded {len(self._rows)} rows from {data_path}"
            f" (prompt_key={prompt_key}, max_prompt_length={max_prompt_length})",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self._rows[index]
        question = row["question"]

        # Format the prompt EXACTLY the way the agentic rollout's
        # ``_decode_question_from_raw_ids`` expects it: keep the literal
        # ``User:`` marker so it can split cleanly. The rollout DOES NOT use
        # this string for actual generation — it rebuilds the SFT prompt from
        # scratch via ``assemble_prompt`` once it has the question.
        prompt_str = f"User: {question}"
        raw_prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            else:
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]

        input_ids = torch.tensor(raw_prompt_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.size(0), dtype=torch.long)

        pad_id = self.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=pad_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # ``ground_truth`` is consumed by the reward function. We don't have
        # a reference image / answer here — the local reward in
        # ``examples/reward_function/emu_agentic_local.py`` only judges the
        # generated trajectory shape + caption + image-presence — so we just
        # echo the question for traceability.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": np.asarray(raw_prompt_ids, dtype=np.int64),
            "ground_truth": question,
            "dataset_source": self.dataset_source,
        }


__all__ = ["EmuAgenticJsonlDataset"]
