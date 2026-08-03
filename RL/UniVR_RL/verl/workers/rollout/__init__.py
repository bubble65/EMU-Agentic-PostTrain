# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .config import RolloutConfig
from .vllm_rollout_spmd import vLLMRollout


def _lazy_emu_agentic():
    """``EmuAgenticRollout`` pulls in ``emu_infer_fast`` + the SFT package at
    import time (vLLM, qwen-agent, the IBQ tokenizer, ...) — that's heavy and
    only relevant when the trainer is actually configured to use it.  Keep
    the regular ``vLLMRollout`` import path light and let callers resolve the
    agentic class lazily."""
    from .emu_agentic import EmuAgenticRollout  # local import → no top-level cost
    return EmuAgenticRollout


__all__ = ["RolloutConfig", "vLLMRollout", "_lazy_emu_agentic"]
