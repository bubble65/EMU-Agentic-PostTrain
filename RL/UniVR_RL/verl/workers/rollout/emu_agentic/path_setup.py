# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Side-effect imports that make rollout tools and ``Emu3.5`` reachable.

Importing this module once is the only thing that links the unmodified
rollout implementation into RL's import graph.  All other modules in
``emu_agentic/`` rely on it.

We resolve paths via env vars (``EMU_INFER_FAST_DIR`` and ``EMU_ROOT``) and
fall back to repo-relative public defaults.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_on_path(p: Path) -> None:
    s = str(p)
    if p.is_dir() and s not in sys.path:
        sys.path.insert(0, s)


_PROJECT_ROOT = Path(__file__).resolve().parents[6]

_DEFAULT_EMU_INFER_FAST = _PROJECT_ROOT / "Agentic_Image_Gen"
_DEFAULT_EMU_ROOT = _PROJECT_ROOT / "Emu3.5"

EMU_INFER_FAST_DIR = Path(os.environ.get("EMU_INFER_FAST_DIR", str(_DEFAULT_EMU_INFER_FAST)))
EMU_ROOT = Path(os.environ.get("EMU_ROOT", str(_DEFAULT_EMU_ROOT)))

# Order matters: emu_infer_fast wins for symbols both define (it doesn't, in
# practice — but its template.py is the canonical SPECIAL dict).
_ensure_on_path(EMU_INFER_FAST_DIR)
_ensure_on_path(EMU_ROOT)


__all__ = ["EMU_INFER_FAST_DIR", "EMU_ROOT"]
