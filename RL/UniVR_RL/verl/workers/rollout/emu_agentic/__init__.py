# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""emu_agentic: serial-per-sample rollout that mirrors emu_infer_fast/run.py.

Importing this package as a side-effect:
  1. Adds ``emu_infer_fast`` and ``Emu3.5`` to ``sys.path`` (path_setup).
  2. Exposes ``EmuAgenticRollout`` for ``verl.workers.rollout.__init__``.

The user's hard rule is that neither ``emu_infer_fast`` nor ``Emu3.5`` is
allowed to change.  Everything in this sub-package only *imports* from those
two repos; the multi-pass generator, the prompt assembler, the tool registry
and the VQ decode all run their original code unmodified.
"""
from . import path_setup  # noqa: F401  — side-effect import (sys.path setup)
from .vllm_rollout_emu_agentic import EmuAgenticRollout

__all__ = ["EmuAgenticRollout"]
