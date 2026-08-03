# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""``EmuAgenticRollout`` — serial-per-sample RL rollout that re-uses
``emu_infer_fast/run.py``'s ``EmuAgent`` *verbatim*.

Design:
  * One vLLM engine is loaded once via ``build_emu3p5_vllm`` (the exact same
    call the inference server makes, with the same args).  No LoRA, single
    process, no FSDP sharding manager wraps it.
  * One ``InProcessEmuServer`` wraps the engine and exposes the same JSON
    surface (``encode_images`` / ``generate``) as ``emu_infer_fast/server.py``.
  * For each prompt in the batch we instantiate one ``EmuAgent`` and call
    ``agent.run(question)`` — i.e. the *exact* multi-round think → tool_call →
    tool_response → ... → BoC/BoI loop used at inference time.  Tools come
    from ``qwen_agent``'s registry (``image_search`` / ``text_search``); the
    side-effect imports in ``run.py`` register them.
  * After the agent returns, we re-assemble the full ``assemble_prompt(...)``
    string, tokenize it once, and split into (prompt, response) along the
    boundary the prompt assembler emitted at the start of the user turn.
    This means the response token ids include EVERY assistant span, EVERY
    tool_response, and the final BoC/BoI image — exactly the trajectory the
    actor needs for GRPO loss computation.

The configuration knobs from emu_infer_fast (image_area, EMU_TEXT_CFG /
EMU_IMAGE_CFG, max_new_tokens, image_top_k...) are read at engine-build time
from the same env vars; ``RolloutConfig`` overrides land in ``EmuAgent``'s
``generate_cfg`` so the per-rollout sampling temps still flow through.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from transformers import AutoTokenizer, PreTrainedTokenizer

from ....protocol import DataProto
from ....utils import torch_functional as VF
from ..base import BaseRollout
from ..config import RolloutConfig
from . import path_setup  # noqa: F401  — side-effect: sys.path setup
from .in_process_server import DEFAULT_IMAGE_SAVE_DIR, InProcessEmuServer

# Import emu_infer_fast unchanged.  `run.EmuAgent` and `template.assemble_prompt`
# are the canonical multi-round driver + prompt assembler used at inference.
from prompt import SYSTEM_PROMPT, USER_PREFIX  # type: ignore  # noqa: E402
from run import EMU_SERVER_URL, EmuAgent  # type: ignore  # noqa: E402
from template import SPECIAL, assemble_prompt  # type: ignore  # noqa: E402

# Side-effect: register tools (image_search / text_search) into qwen_agent
import tool_imagesearch  # type: ignore  # noqa: F401,E402
import tool_textsearch  # type: ignore  # noqa: F401,E402


_PROJECT_ROOT = Path(__file__).resolve().parents[6]

# ── default paths that mirror Agentic_Image_Gen/start.sh ───────────────────
_DEFAULT_MODEL_PATH = os.environ.get(
    "EMU_MODEL_PATH",
    str(_PROJECT_ROOT / "checkpoints" / "emu3p5-sft"),
)
_DEFAULT_VQ_PATH = os.environ.get(
    "EMU_VQ_PATH",
    str(_PROJECT_ROOT / "checkpoints" / "Emu3.5-VisionTokenizer"),
)
_DEFAULT_VQ_TYPE = os.environ.get("EMU_VQ_TYPE", "ibq")
_DEFAULT_VQ_DEVICE = os.environ.get("EMU_VQ_DEVICE", "cuda:0")
_DEFAULT_TP_SIZE = int(os.environ.get("EMU_TP_SIZE", "2"))
_DEFAULT_GPU_MEM_UTIL = float(os.environ.get("EMU_GPU_MEM_UTIL", "0.7"))
_DEFAULT_SEED = int(os.environ.get("EMU_SEED", "6666"))


def _decode_question_from_raw_ids(
    tokenizer, raw_prompt_ids
) -> str:
    """Recover the user's natural-language question from the raw prompt token
    ids the dataset handed us.

    The RL_roll training pipeline tokenises a dataset row as
    ``{format_prompt}{user_question}`` (see examples/format_prompt/emu3.jinja),
    so the simplest, format-agnostic way to feed the question back to the
    agent is to detokenise the raw ids and pull whatever comes after the last
    ``User:`` marker — or just the full body if no such marker exists.
    """
    text = tokenizer.decode(list(raw_prompt_ids), skip_special_tokens=False)
    if "User:" in text:
        text = text.rsplit("User:", 1)[1]
    elif "USER:" in text:
        text = text.rsplit("USER:", 1)[1]
    else:
        # Last-resort: strip leading system prompt by chopping at the system-end
        # newline pattern emu3.jinja uses.  We still return *something* the agent
        # can search on; bad formatting is better than crashing.
        text = text.strip()

    for marker in ("ASSISTANT:", SPECIAL["bss"], "<|assistant|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def _load_emu_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=os.path.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    tokenizer.bos_token = SPECIAL["bos"]
    tokenizer.eos_token = SPECIAL["eos"]
    tokenizer.pad_token = SPECIAL["pad"]
    tokenizer.eol_token = SPECIAL["eol"]
    tokenizer.eof_token = SPECIAL["eof"]
    tokenizer.tms_token = SPECIAL["tms"]
    tokenizer.img_token = SPECIAL["img"]
    tokenizer.boi_token = SPECIAL["boi"]
    tokenizer.eoi_token = SPECIAL["eoi"]
    tokenizer.bss_token = SPECIAL["bss"]
    tokenizer.ess_token = SPECIAL["ess"]
    tokenizer.bog_token = SPECIAL["bog"]
    tokenizer.eog_token = SPECIAL["eog"]
    tokenizer.boc_token = SPECIAL["boc"]
    tokenizer.eoc_token = SPECIAL["eoc"]
    return tokenizer


def _build_actor_synced_emu_service(
    model_path: str,
    tokenizer_path: str,
    vq_path: str,
    *,
    vq_type: str,
    vq_device: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    seed: int,
    lora_kwargs: Dict[str, Any],
):
    """HARD-CODED copy of ``Emu3.5/src/utils/model_utils.py::build_emu3p5_vllm``
    (lines 116-153), with only two additions required for verl's Ray-per-rank
    FSDP layout:

      * ``distributed_executor_backend="external_launcher"`` — re-use the
        FSDP torch.distributed group instead of forking child workers.
      * ``disable_custom_all_reduce=True`` — Ray-per-rank GPU isolation
        breaks vLLM's CustomAllreduce ``_can_p2p`` peer check.

    Every OTHER kwarg below is a verbatim copy of the scaffold's call. If the
    scaffold's parameters ever change, edit this function by hand.
    """
    # B300 sm_103a guards — same env knobs as emu_infer_fast/start.sh.
    # These are os.environ.setdefault so RAY runtime_env can still override.
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    os.environ.setdefault("EMU_ENFORCE_EAGER", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

    from src.vision_tokenizer import build_vision_tokenizer  # type: ignore
    from vllm import LLM

    tokenizer = _load_emu_tokenizer(tokenizer_path)
    vq_model = build_vision_tokenizer(vq_type, vq_path, device=vq_device)

    resolution_map = {}
    for digit_str in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*"]:
        resolution_map[tokenizer.encode(digit_str)[0]] = digit_str

    engine_kwargs = {}
    if lora_kwargs:
        engine_kwargs.update(lora_kwargs)

    # ── HARD-CODED LLM kwargs (copy of build_emu3p5_vllm, lines 116-153) ──
    model = LLM(
        model_path,
        tokenizer=tokenizer_path,
        trust_remote_code=True,
        dtype="auto",
        # NEW for verl: reuse FSDP torch.dist group, no child workers.
        distributed_executor_backend="external_launcher",
        # NEW for verl: avoid CustomAllreduce peer-check crash under Ray-per-rank
        # GPU visibility.
        disable_custom_all_reduce=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=False,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_num_batched_tokens=32768 * 10,
        max_num_seqs=2,
        seed=seed,
        generation_config='vllm',
        scheduler_cls="vllm.v1.core.sched.batch_scheduler.Scheduler",
        enforce_eager=os.environ.get("EMU_ENFORCE_EAGER", "0") == "1",
        compilation_config=(
            None
            if os.environ.get("EMU_ENFORCE_EAGER", "0") == "1"
            else {
                "full_cuda_graph": True,
                "backend": "cudagraph",
                "cudagraph_capture_sizes": [1, 2],
            }
        ),
        additional_config={
            "boi_token_id": tokenizer.encode("<|image start|>")[0],
            "soi_token_id": tokenizer.encode("<|image token|>")[0],
            "eol_token_id": tokenizer.encode("<|extra_200|>")[0],
            "eoi_token_id": tokenizer.encode("<|image end|>")[0],
            "resolution_map": resolution_map,
        },
        enable_sleep_mode=True,
        **engine_kwargs,
    )
    model.set_tokenizer(tokenizer)
    print(f"{model.llm_engine.vllm_config=}")
    return model, tokenizer, vq_model


class EmuAgenticRollout(BaseRollout):
    """Serial rollout that uses verl's actor-synced vLLM as an Emu service.

    The agent loop, prompt assembly, tool dispatch, image-label bookkeeping
    and return JSON structure all come from ``emu_infer_fast/run.py``.  The
    model service itself is in-process so verl can sync current actor weights
    before rollout through the normal ``FSDPVLLMShardingManager`` path.
    """

    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[Any] = None,
        tokenizer_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.lora_kwargs = kwargs.get("lora_kwargs", {})

        tp_size = int(os.environ.get("EMU_TP_SIZE", str(getattr(config, "tensor_parallel_size", _DEFAULT_TP_SIZE))))

        eff_model_path = model_path or _DEFAULT_MODEL_PATH
        eff_tokenizer_path = tokenizer_path or os.environ.get(
            "EMU_TOKENIZER_PATH",
            str(path_setup.EMU_ROOT / "src" / "tokenizer_emu3_ibq"),
        )
        eff_vq_path = os.environ.get("EMU_VQ_PATH", _DEFAULT_VQ_PATH)
        print(
            f"[EmuAgenticRollout] building actor-synced in-process Emu service:\n"
            f"  model_path     = {eff_model_path}\n"
            f"  tokenizer_path = {eff_tokenizer_path}\n"
            f"  vq_path        = {eff_vq_path}\n"
            f"  tp_size        = {tp_size}\n"
            f"  gpu_mem_util   = {_DEFAULT_GPU_MEM_UTIL}\n"
            f"  vq_device      = {_DEFAULT_VQ_DEVICE}",
            flush=True,
        )
        self._model, self._tokenizer, self._vq_model = _build_actor_synced_emu_service(
            eff_model_path,
            eff_tokenizer_path,
            eff_vq_path,
            vq_type=_DEFAULT_VQ_TYPE,
            vq_device=_DEFAULT_VQ_DEVICE,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=_DEFAULT_GPU_MEM_UTIL,
            seed=_DEFAULT_SEED,
            lora_kwargs=self.lora_kwargs,
        )
        self.tokenizer = self._tokenizer
        if self.pad_token_id is None:
            self.pad_token_id = self._tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        self.server = InProcessEmuServer(self._model, self._tokenizer, self._vq_model)
        self.inference_engine = self._model
        self.inference_engine.sleep(level=1)

        # ── Per-call agent settings ──────────────────────────────────────
        # These mirror run.sh's defaults.  RolloutConfig fields override only
        # when explicitly set.
        self.max_rounds = int(os.environ.get("EMU_MAX_LLM_CALL_PER_RUN", "8"))
        self.max_new_tokens = int(
            os.environ.get("EMU_MAX_NEW_TOKENS", "8192")
        )
        self.image_save_dir = os.environ.get(
            "EMU_IMAGE_SAVE_DIR", DEFAULT_IMAGE_SAVE_DIR
        )
        self.force_draw_round = int(os.environ.get("EMU_FORCE_DRAW_ROUND", "6"))
        self.force_draw_hint = os.environ.get(
            "EMU_FORCE_DRAW_HINT",
            "Based on all retrieved information and reference images above, "
            "render the final image now.",
        )
        os.makedirs(self.image_save_dir, exist_ok=True)
        print(f"[EmuAgenticRollout] image_save_dir = {self.image_save_dir}", flush=True)
        print(
            f"[EmuAgenticRollout] max_rounds={self.max_rounds} "
            f"max_new_tokens={self.max_new_tokens} "
            f"force_draw_round={self.force_draw_round}",
            flush=True,
        )

        # Per-step artefact directory. Defaults to ``<image_save_dir>/../traces``;
        # the launcher can pin it via ``EMU_TRACE_DIR``. The actual step number
        # comes through ``prompts.meta_info['global_step']``.
        self.trace_dir = os.environ.get(
            "EMU_TRACE_DIR",
            os.path.join(os.path.dirname(self.image_save_dir.rstrip("/")), "traces"),
        )
        os.makedirs(self.trace_dir, exist_ok=True)
        print(f"[EmuAgenticRollout] trace_dir = {self.trace_dir}", flush=True)

    # ── helpers ─────────────────────────────────────────────────────────

    def _build_agent(self, image_save_dir: Optional[str] = None) -> EmuAgent:
        """Construct a fresh ``EmuAgent`` whose ``self.server`` is OUR
        in-process server.  The agent's HTTP code path never fires because we
        substitute the client attribute right after construction; this lets
        us keep ``EmuAgent`` *unmodified*.

        ``image_save_dir`` lets us point each step's rollout at a different
        sub-directory so the saved PNGs are grouped by training step.
        """
        agent = EmuAgent(
            server_url="http://in-process" if self.server is not None else self.server_url,
            system_message=SYSTEM_PROMPT,
            image_save_dir=image_save_dir or self.image_save_dir,
            generate_cfg={"max_new_tokens": self.max_new_tokens},
            max_rounds=self.max_rounds,
            force_draw_round=self.force_draw_round,
            force_draw_hint=self.force_draw_hint,
        )
        if self.server is not None:
            agent.server = self.server  # swap HTTP client → in-process server
        return agent

    def _run_one(self, question: str, rollout_id: int, image_save_dir: Optional[str] = None) -> Dict[str, Any]:
        """One agentic rollout end-to-end.  Returns the dict ``EmuAgent.run``
        produces (``question``, ``rollout_id``, ``messages``, ``prediction``,
        ``termination``, ``saved_images``), with an EXTRA ``raw_messages``
        entry: the un-collapsed message list (visual ``<BoI>...<EoI>`` blocks
        intact) — required for retokenising into RL response ids.

        ``EmuAgent.run`` runs the loop on an internal ``messages: List[dict]``
        and only dumps a cleaned (visual-block-collapsed) copy in its return
        value.  We don't want to fork ``run.py``, so we wrap the agent's
        ``_dump_messages`` method to capture its input — that input *is* the
        raw list.
        """
        agent = self._build_agent(image_save_dir=image_save_dir)
        captured: Dict[str, Any] = {"raw_messages": None}
        original_dump = agent._dump_messages

        def _spy_dump(messages, label_book):
            # Defensive deep-copy: ``messages`` is mutated by EmuAgent across
            # rounds and ``_dump_messages`` is the last thing it touches; we
            # want the post-run snapshot, not a live reference.
            captured["raw_messages"] = [dict(m) for m in messages]
            return original_dump(messages, label_book)

        agent._dump_messages = _spy_dump  # type: ignore[assignment]

        result = agent.run(question, rollout_id=rollout_id)
        result["raw_messages"] = captured["raw_messages"] or result["messages"]
        return result

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Re-assemble the full SFT-format prompt string the model 'saw' in
        toto over the whole multi-round rollout.

        We pass the messages straight through ``assemble_prompt`` with
        ``open_assistant=False`` because the LAST assistant turn is already
        closed (either via native draw → ``<BoI>...<EoI>`` or via the
        force-draw branch in ``EmuAgent``).
        """
        return assemble_prompt(messages, open_assistant=False) + SPECIAL["eos"]

    def _tokenize_with_policy_mask(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[int], List[int], List[int]]:
        """Tokenize ``assemble_prompt(messages) + EOS`` AND return a per-token
        ``policy_mask`` that is 1 only on tokens the policy actually sampled.

        Policy tokens are:
          * Every ``assistant`` message body (including the ``<BSS>`` /
            ``<ESS>`` delimiters that wrap it — those ARE sampled in normal
            rounds; in force_draw the ``<BSS>`` comes from a hand-built
            primer but the rest of the body still came from the policy, so
            we keep the whole assistant span as policy).
          * The trailing ``<EOS>`` (1 token) — harmless, models EOS choice.

        NON-policy tokens (mask = 0) are:
          * ``system`` text + ``<BOS>``
          * the first ``user`` question
          * every ``tool_response`` block (this is the bug we're fixing —
            tool outputs are environment, not policy, and must not enter the
            actor's log_prob / KL / loss budget).

        The function mirrors ``template.assemble_prompt`` exactly so the
        resulting tokens are guaranteed to match
        ``self.tokenizer.encode(_messages_to_prompt(messages))`` byte-for-byte.
        That equivalence is asserted at the end as a defensive check.

        Returns:
          (full_token_ids, prompt_split_idx, policy_mask) — caller still slices
          ``response_ids = full_token_ids[first_bss:]`` so we also return the
          policy_mask aligned to the SAME full sequence; the trainer-facing
          response_mask is taken as ``policy_mask[first_bss:]``.
        """
        tok = self.tokenizer
        full_ids: List[int] = []
        policy_mask: List[int] = []

        def _append(piece: str, *, is_policy: bool) -> None:
            ids = tok.encode(piece, add_special_tokens=False)
            full_ids.extend(ids)
            policy_mask.extend([1 if is_policy else 0] * len(ids))

        system_text: List[str] = []
        first_user_used = False
        # Phase 1: collect system messages so we can emit them only once with
        # the first user (mirrors assemble_prompt).
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            if not isinstance(content, str):
                raise TypeError(
                    f"policy-mask tokenizer expects string content, got "
                    f"{type(content).__name__} for role={role!r}"
                )

            if role == "system":
                system_text.append(content)
                continue

            if role == "user":
                if first_user_used:
                    raise ValueError("only one user turn supported")
                sys_blob = ("\n\n".join(s for s in system_text if s)).strip()
                # <BOS> + system + "\n\n" + user_question  → all not policy
                prefix = SPECIAL["bos"] + sys_blob
                if sys_blob and content:
                    prefix += "\n\n"
                prefix += content
                _append(prefix, is_policy=False)
                first_user_used = True

            elif role == "assistant":
                # Wrap with <BSS>{body}<ESS> — entire span counts as policy.
                _append(SPECIAL["bss"] + content + SPECIAL["ess"], is_policy=True)

            elif role == "tool":
                # tool_response is environment input; NOT policy.
                _append(f"<tool_response>\n{content}\n</tool_response>",
                        is_policy=False)

            else:
                raise ValueError(f"unsupported role {role!r}")

        if not first_user_used:
            sys_blob = ("\n\n".join(s for s in system_text if s)).strip()
            if sys_blob:
                _append(SPECIAL["bos"] + sys_blob, is_policy=False)

        # Trailing EOS — appended in _messages_to_prompt; we mirror it here.
        # Count EOS as policy (it terminates the trajectory and the model
        # learned to emit it; harmless to include).
        _append(SPECIAL["eos"], is_policy=True)

        # Defensive: byte-for-byte equivalence with assemble_prompt() + EOS.
        # If a future change to template.py breaks this, fail loud.
        expected = self._messages_to_prompt(messages)
        expected_ids = tok.encode(expected, add_special_tokens=False)
        if expected_ids != full_ids:
            # Shouldn't happen unless template.assemble_prompt changes; fall
            # back to the unmasked tokens so training still works, but warn.
            print(
                f"[EmuAgenticRollout] WARN: policy_mask tokenizer drifted from "
                f"assemble_prompt; got {len(full_ids)} vs expected "
                f"{len(expected_ids)} tokens. Falling back to all-policy mask.",
                flush=True,
            )
            return expected_ids, len(expected_ids), [1] * len(expected_ids)

        return full_ids, len(full_ids), policy_mask

    def _load_saved_images_for_reward(self, saved_images_per_sample: List[List[str]]) -> List[List[Any]]:
        """Load already-decoded PNGs from emu_infer_fast for reward workers.

        The regular Emu3 rollout VQ-decodes image tokens inside the rollout
        worker because reward workers may not have a GPU.  Agentic rollout
        already receives decoded PNG paths from emu_infer_fast/server.py, so
        we only need to open them as PIL images.
        """
        if not getattr(self.config, "enable_image_decode_for_reward", False):
            return []
        from PIL import Image

        decoded_images: List[List[Any]] = []
        for sample_paths in saved_images_per_sample:
            sample_images = []
            for path in sample_paths:
                try:
                    sample_images.append(Image.open(path).convert("RGB"))
                except Exception as exc:
                    print(f"[EmuAgenticRollout] failed to load saved image {path}: {exc}", flush=True)
            decoded_images.append(sample_images)
        return decoded_images

    # ── main entry point ─────────────────────────────────────────────────
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = dict(prompts.non_tensor_batch)
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        # Drop fields the agentic rollout doesn't use; mirror vLLMRollout's
        # vocabulary so trainers that pass them don't crash.
        for k in (
            "multi_modal_data",
            "uncond_prompt_ids",
            "gt_frames_info",
            "decoded_gt_images_bytes",
            "decoded_ref_images_bytes",
        ):
            non_tensor_batch.pop(k, None)

        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("EmuAgenticRollout: batch_size mismatch with raw_prompt_ids")

        # ── 1) extract questions from raw prompt ids (one per sample) ──
        base_questions: List[str] = []
        for raw in batch_raw_prompt_ids:
            base_questions.append(_decode_question_from_raw_ids(self.tokenizer, raw))

        rollout_n = int(prompts.meta_info.get("n", getattr(self.config, "n", 1)) or 1)
        if rollout_n < 1:
            raise ValueError(f"EmuAgenticRollout requires rollout n >= 1, got {rollout_n}")
        questions = [q for q in base_questions for _ in range(rollout_n)]

        global_step = int(prompts.meta_info.get("global_step", 0) or 0)
        step_image_dir = os.path.join(self.image_save_dir, f"step{global_step:03d}")
        os.makedirs(step_image_dir, exist_ok=True)
        print(
            f"[EmuAgenticRollout] step={global_step} writing images to {step_image_dir}",
            flush=True,
        )

        # ── 2) serial rollout (one-at-a-time as the user requested) ────
        results: List[Dict[str, Any]] = []
        t0 = time.time()
        for i, question in enumerate(questions):
            print(
                f"[EmuAgenticRollout] sample {i+1}/{len(questions)} q='{question[:80]}'",
                flush=True,
            )
            try:
                results.append(self._run_one(question, rollout_id=i, image_save_dir=step_image_dir))
            except Exception as exc:
                # One bad sample shouldn't sink the batch.  Record the failure
                # and synthesise an empty trajectory so the tensor shapes
                # line up downstream.
                print(f"[EmuAgenticRollout] sample {i} crashed: {exc}", flush=True)
                results.append({
                    "question": question,
                    "rollout_id": i,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PREFIX + question},
                    ],
                    "raw_messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PREFIX + question},
                    ],
                    "prediction": f"[rollout error: {exc}]",
                    "termination": f"error: {exc}",
                    "saved_images": [],
                })
        wall = time.time() - t0
        print(
            f"[EmuAgenticRollout] {len(questions)} rollouts done in {wall:.1f}s "
            f"(avg {wall / max(1, len(questions)):.1f}s/sample)",
            flush=True,
        )

        # ── 3) retokenise each trajectory and split (prompt, response) ──
        prompt_length = int(self.config.prompt_length)
        response_length = int(self.config.response_length)

        out_response_ids: List[List[int]] = []
        out_prompt_ids: List[List[int]] = []
        out_response_policy_masks: List[List[int]] = []
        out_full_text: List[str] = []
        saved_images_per_sample: List[List[str]] = []
        terminations: List[str] = []
        question_texts: List[str] = []
        predictions: List[str] = []
        dumped_messages: List[Any] = []
        raw_messages: List[Any] = []

        for res in results:
            # ``raw_messages`` holds the full <BoI>...<EoI> blocks the model
            # actually emitted; ``messages`` is the cleaned dump for humans.
            full_prompt_str = self._messages_to_prompt(res["raw_messages"])
            # Tokenize WITH a per-token policy_mask so we can mask out
            # tool_response tokens from the actor's loss + log_prob + KL.
            # Without this, the policy is wrongly trained on tool outputs that
            # it cannot influence (env-only tokens), which biases the gradient.
            full_token_ids, _, full_policy_mask = self._tokenize_with_policy_mask(
                res["raw_messages"]
            )
            # Find where the prompt ends and the response begins.  The
            # boundary is the first ``<BSS>`` (begin-assistant-span) — every
            # subsequent assistant turn / tool_response is part of the
            # agent's "response" for RL purposes.
            bss_id = self.tokenizer.encode(SPECIAL["bss"])[0]
            try:
                first_bss = full_token_ids.index(bss_id)
            except ValueError:
                first_bss = len(full_token_ids)
            prompt_ids = full_token_ids[:first_bss]
            response_ids = full_token_ids[first_bss:]
            response_policy_mask = full_policy_mask[first_bss:]

            # Pad/truncate to fixed widths the trainer expects.  We left-pad
            # the prompt and right-pad the response (matches vLLMRollout).
            if len(prompt_ids) > prompt_length:
                prompt_ids = prompt_ids[-prompt_length:]
            if len(response_ids) > response_length:
                response_ids = response_ids[:response_length]
                response_policy_mask = response_policy_mask[:response_length]

            out_prompt_ids.append(prompt_ids)
            out_response_ids.append(response_ids)
            out_response_policy_masks.append(response_policy_mask)
            out_full_text.append(full_prompt_str)
            saved_images_per_sample.append(list(res.get("saved_images") or []))
            terminations.append(str(res.get("termination") or ""))
            question_texts.append(str(res.get("question") or ""))
            predictions.append(str(res.get("prediction") or ""))
            dumped_messages.append(res.get("messages") or [])
            raw_messages.append(res.get("raw_messages") or [])

        # ── 4) build tensors in verl's expected layout ──────────────────
        device = input_ids.device
        pad = self.pad_token_id
        out_batch_size = len(out_response_ids)

        padded_prompts = torch.full(
            (out_batch_size, prompt_length), pad, dtype=input_ids.dtype, device=device
        )
        padded_responses = torch.full(
            (out_batch_size, response_length), pad, dtype=input_ids.dtype, device=device
        )
        for i, (pids, rids) in enumerate(zip(out_prompt_ids, out_response_ids)):
            if pids:
                padded_prompts[i, -len(pids):] = torch.tensor(pids, dtype=input_ids.dtype, device=device)
            if rids:
                padded_responses[i, : len(rids)] = torch.tensor(rids, dtype=input_ids.dtype, device=device)

        sequence_ids = torch.cat([padded_prompts, padded_responses], dim=-1)

        # attention / position rebuild.  Prompt: left-pad mask;  response:
        # right-pad mask with EOS-stop (matches vLLMRollout).
        prompt_attention = (padded_prompts != pad).to(attention_mask.dtype)
        eos_response_mask = VF.get_response_mask(
            response_ids=padded_responses, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )

        # Build the per-token policy mask in the response window: 1 for tokens
        # the actor actually sampled, 0 for tool_response tokens spliced in
        # by the environment.
        # IMPORTANT — TWO separate masks:
        #   * ``full_attention_mask`` ← which tokens are NOT padding. Tool
        #     response tokens ARE real context the model must SEE during
        #     forward, otherwise next assistant turn's next-token predictions
        #     are broken. So attention_mask keeps tool_response = 1.
        #   * ``response_mask`` ← which tokens count toward loss / KL /
        #     log_prob normalization. Here tool_response = 0 because those
        #     tokens came from the env, not the policy; including them would
        #     train the actor on tokens it can't control.
        policy_mask_tensor = torch.zeros_like(padded_responses,
                                              dtype=attention_mask.dtype)
        for i, pol_mask in enumerate(out_response_policy_masks):
            if pol_mask:
                policy_mask_tensor[i, : len(pol_mask)] = torch.tensor(
                    pol_mask, dtype=attention_mask.dtype, device=device
                )
        # response_mask: loss mask = (not padding / post-EOS) AND (policy token)
        response_mask = eos_response_mask * policy_mask_tensor

        # full_attention_mask: forward-visibility mask = (not padding / post-EOS)
        # only — tool_response stays VISIBLE so next-token predictions work.
        full_attention_mask = torch.cat([prompt_attention, eos_response_mask], dim=-1)

        # Diag — how many tokens did the tool_response mask remove from the
        # loss budget? (forward visibility is NOT affected.)
        try:
            kept = int(response_mask.sum().item())
            eos_only = int(eos_response_mask.sum().item())
            if eos_only > 0:
                print(
                    f"[EmuAgenticRollout] response_mask (loss): kept {kept}/{eos_only} "
                    f"tokens after tool_response stripping "
                    f"({100.0 * (eos_only - kept) / eos_only:.1f}% removed); "
                    f"attention_mask (forward) unchanged.",
                    flush=True,
                )
        except Exception:
            pass

        # Position ids: simple cumulative — the trainer only consumes deltas.
        prompt_position_ids = (prompt_attention.cumsum(dim=-1) - 1).clamp(min=0)
        delta = torch.arange(1, response_length + 1, device=device).view(1, -1).expand(out_batch_size, -1)
        response_position_ids = prompt_position_ids[:, -1:] + delta
        full_position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

        batch = TensorDict(
            {
                "prompts": padded_prompts,
                "responses": padded_responses,
                "input_ids": sequence_ids,
                "attention_mask": full_attention_mask,
                "response_mask": response_mask,
                # full-trajectory mask used by the reward worker to recover the
                # complete decoded response text (tool_response included) when
                # computing the dual judge — without it, reward sees only the
                # first few policy tokens and the final caption + image block
                # gets sliced off, which makes the judge see a truncated string.
                "response_mask_full": eos_response_mask,
                "position_ids": full_position_ids,
            },
            batch_size=out_batch_size,
        )

        # Surface per-sample debug / reward-input data to downstream.
        if rollout_n > 1:
            repeated_non_tensor_batch = {}
            for key, value in non_tensor_batch.items():
                try:
                    repeated_non_tensor_batch[key] = np.repeat(value, rollout_n, axis=0)
                except Exception:
                    repeated_non_tensor_batch[key] = np.array(
                        [item for item in value for _ in range(rollout_n)], dtype=object
                    )
            non_tensor_batch = repeated_non_tensor_batch

        non_tensor_batch["saved_images"] = np.array(saved_images_per_sample, dtype=object)
        non_tensor_batch["agentic_prediction"] = np.array(predictions, dtype=object)
        non_tensor_batch["agentic_termination"] = np.array(terminations, dtype=object)
        non_tensor_batch["agentic_question"] = np.array(question_texts, dtype=object)
        non_tensor_batch["agentic_full_text"] = np.array(out_full_text, dtype=object)
        non_tensor_batch["agentic_messages"] = np.array(dumped_messages, dtype=object)
        non_tensor_batch["agentic_raw_messages"] = np.array(raw_messages, dtype=object)
        decoded_images = self._load_saved_images_for_reward(saved_images_per_sample)
        if decoded_images:
            non_tensor_batch["decoded_images"] = np.array(decoded_images, dtype=object)

        # ── 5) write per-step trace jsonl ──────────────────────────────
        # Each step gets its own ``step{N:03d}.jsonl`` with one row per rollout
        # sample so we can audit caption / image / termination quality
        # independently of wandb. We DON'T include the raw_messages (they hold
        # the verbatim ``<|visual token …|>`` blob — multi-MB per sample); the
        # cleaned ``messages`` keep visual blocks as ``[reference_image]``.
        try:
            self._dump_step_trace(
                global_step=global_step,
                question_texts=question_texts,
                predictions=predictions,
                terminations=terminations,
                dumped_messages=dumped_messages,
                saved_images_per_sample=saved_images_per_sample,
                wall_seconds=wall,
            )
        except Exception as exc:
            # Trace dumping must never sink a training step.
            print(f"[EmuAgenticRollout] trace dump skipped: {exc}", flush=True)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    # ── trace dumping ─────────────────────────────────────────────────
    def _dump_step_trace(
        self,
        *,
        global_step: int,
        question_texts: List[str],
        predictions: List[str],
        terminations: List[str],
        dumped_messages: List[Any],
        saved_images_per_sample: List[List[str]],
        wall_seconds: float,
    ) -> None:
        """Write one trace jsonl per DP rank: ``step{N:03d}_rank{R}.jsonl``.

        Each FSDP/vLLM rank only sees its own chunk of the global batch (after
        the sharding manager's preprocess gather+chunk), so we can't centralize
        writes on a single ``rank==0`` writer without losing 3/4 of the rows
        under DP=4. Instead every TP-rank-0 writer dumps its slice with the
        rank stamped on each row; the trace dir is on jfs so all four files
        sit next to each other.
        """
        try:
            # Only TP-rank-0 writes (avoid 2x duplicate per TP pair).
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            tp_rank = int(get_tensor_model_parallel_rank())
            tp_size = int(get_tensor_model_parallel_world_size())
        except Exception:
            tp_rank, tp_size = 0, 1
        if tp_rank != 0:
            return
        # Global rank → DP rank (with TP-rank-0-only writes, this is just
        # rank // tp_size). Use the global rank in the filename so files line
        # up with the order DP_COMPUTE_PROTO chunks the batch.
        dp_rank = self.rank // max(1, tp_size)
        trace_path = os.path.join(
            self.trace_dir, f"step{global_step:03d}_dp{dp_rank}.jsonl"
        )
        with open(trace_path, "a", encoding="utf-8") as f:
            for i, (q, pred, term, msgs, imgs) in enumerate(
                zip(question_texts, predictions, terminations, dumped_messages, saved_images_per_sample)
            ):
                row = {
                    "step": global_step,
                    "dp_rank": dp_rank,
                    "global_rank": self.rank,
                    "sample": i,
                    "question": q,
                    "prediction": pred,
                    "termination": term,
                    "saved_images": list(imgs),
                    "n_images": len(imgs),
                    "messages": msgs,
                    "wall_seconds": round(float(wall_seconds), 2),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[EmuAgenticRollout] step={global_step} dp={dp_rank} trace → {trace_path}"
            f" ({len(question_texts)} rows, {sum(len(x) for x in saved_images_per_sample)} images)",
            flush=True,
        )


__all__ = ["EmuAgenticRollout"]
