#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback, set_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.emu3p5 import Emu3Config, Emu3ForCausalLM  # noqa: E402

IGNORE_INDEX = -100

SPECIAL = dict(
    bos="<|extra_203|>",
    eos="<|extra_204|>",
    pad="<|endoftext|>",
    eol="<|extra_200|>",
    eof="<|extra_201|>",
    tms="<|extra_202|>",
    img="<|image token|>",
    boi="<|image start|>",
    eoi="<|image end|>",
    bss="<|extra_100|>",
    ess="<|extra_101|>",
    bog="<|extra_60|>",
    eog="<|extra_61|>",
    boc="<|extra_50|>",
    eoc="<|extra_51|>",
)


def load_emu_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=os.path.join(tokenizer_path, "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    for key, val in SPECIAL.items():
        setattr(tokenizer, f"{key}_token", val)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def load_emu_model(args):
    config = Emu3Config.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = False

    model = Emu3ForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model


@dataclass
class TokenizedSFTCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def _pad_len(self, max_len: int) -> int:
        if not self.pad_to_multiple_of:
            return max_len
        multiple = self.pad_to_multiple_of
        return ((max_len + multiple - 1) // multiple) * multiple

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = self._pad_len(max(len(x["input_ids"]) for x in features))
        input_ids, labels, attention_mask = [], [], []
        pad_id = self.tokenizer.pad_token_id

        for item in features:
            ids = list(item["input_ids"])
            labs = list(item["labels"])
            mask = list(item.get("attention_mask", [1] * len(ids)))
            pad_len = max_len - len(ids)

            input_ids.append(ids + [pad_id] * pad_len)
            labels.append(labs + [IGNORE_INDEX] * pad_len)
            attention_mask.append(mask + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class TrainableParameterCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None or not args.should_log:
            return
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = 100 * trainable / total if total else 0.0
        print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.4f}%)")


def maybe_apply_lora(model, args):
    if not args.lora:
        return model, None

    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError("`--lora` requires `peft`. Install it with `pip install peft`.") from exc

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return model, peft_config


def build_sft_config(args, tokenizer):
    from trl import SFTConfig

    kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.data_seed,
        deepspeed=args.deepspeed,
        optim=args.optim,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_text_field="text",
        completion_only_loss=False,
        assistant_only_loss=False,
        eos_token=tokenizer.eos_token,
    )

    signature = inspect.signature(SFTConfig)
    supported = set(signature.parameters)
    if "max_length" in supported:
        kwargs["max_length"] = None
    else:
        kwargs["max_seq_length"] = None

    if args.eval_data:
        kwargs.update(eval_strategy=args.eval_strategy, eval_steps=args.eval_steps)
    else:
        kwargs.update(eval_strategy="no")

    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    kwargs = {key: val for key, val in kwargs.items() if key in supported}
    return SFTConfig(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser("SFT Emu3.5 on pre-tokenized agent data with TRL SFTTrainer")

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default="./src/tokenizer_emu3_ibq")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")

    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--attn-implementation", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_torch")

    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--eval-strategy", default="steps")
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--resume-from-checkpoint", default=None)

    args = parser.parse_args()
    if args.bf16 and args.fp16:
        parser.error("Choose only one of --bf16 or --fp16.")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = load_emu_tokenizer(args.tokenizer_path)
    model = load_emu_model(args)
    model, peft_config = maybe_apply_lora(model, args)

    def maybe_truncate(example):
        if args.max_seq_length is None:
            return example

        max_len = args.max_seq_length
        example["input_ids"] = example["input_ids"][:max_len]
        example["labels"] = example["labels"][:max_len]
        example["attention_mask"] = example["attention_mask"][:max_len]
        return example


    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    keep_cols = {"input_ids", "labels", "attention_mask"}
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_cols])

    eval_dataset = None
    if args.eval_data:
        eval_dataset = load_dataset("json", data_files=args.eval_data, split="train")
        eval_dataset = eval_dataset.remove_columns([c for c in eval_dataset.column_names if c not in keep_cols])

    if args.max_seq_length is not None:
        train_dataset = train_dataset.map(
            maybe_truncate,
            num_proc=args.dataloader_num_workers,
        )

        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                maybe_truncate,
                num_proc=args.dataloader_num_workers,
            )

    sft_config = build_sft_config(args, tokenizer)
    collator = TokenizedSFTCollator(tokenizer, pad_to_multiple_of=args.pad_to_multiple_of)

    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        peft_config=peft_config,
        callbacks=[TrainableParameterCallback()],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if trainer.is_world_process_zero():
        with open(os.path.join(args.output_dir, "sft_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
