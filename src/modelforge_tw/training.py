from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from modelforge_tw.dataset import load_splits, sha256_file
from modelforge_tw.prompts import messages_for_version, training_messages


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = "Qwen/Qwen3-1.7B"
    model_revision: str = "main"
    output_dir: Path = Path("artifacts/qwen3-1.7b-triage-lora")
    data_dir: Path = Path("data/generated")
    seed: int = 42
    max_length: int = Field(default=768, ge=128, le=4096)
    use_4bit: bool = True
    lora_rank: int = Field(default=8, ge=1, le=128)
    lora_alpha: int = Field(default=16, ge=1, le=256)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    learning_rate: float = Field(default=0.0002, gt=0)
    epochs: float = Field(default=1.0, gt=0, le=10)
    max_steps: int = Field(default=-1, ge=-1)
    train_batch_size: int = Field(default=1, ge=1, le=32)
    eval_batch_size: int = Field(default=1, ge=1, le=32)
    gradient_accumulation_steps: int = Field(default=16, ge=1, le=256)
    warmup_ratio: float = Field(default=0.05, ge=0, lt=1)
    logging_steps: int = Field(default=5, ge=1)
    eval_steps: int = Field(default=25, ge=1)
    save_steps: int = Field(default=25, ge=1)
    save_total_limit: int = Field(default=2, ge=1, le=10)
    resume_from_checkpoint: bool = True
    prompt_version: Literal[
        "optimized-v1",
        "optimized-v2",
        "optimized-v3",
        "optimized-v4",
        "optimized-v5",
    ] = "optimized-v1"


def load_training_config(path: Path) -> TrainingConfig:
    return TrainingConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _render_chat_template(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> str:
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


@dataclass
class CompletionOnlyCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            pad = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def run_training(config: TrainingConfig) -> dict[str, Any]:
    import numpy as np
    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved real training run")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    transformers.set_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_records = load_splits(config.data_dir, ["train"])
    validation_records = load_splits(config.data_dir, ["validation"])
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(record: Any) -> dict[str, list[int]]:
        full_text = _render_chat_template(
            tokenizer,
            training_messages(
                record.input,
                record.target,
                prompt_version=config.prompt_version,
            ),
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = _render_chat_template(
            tokenizer,
            messages_for_version(record.input, config.prompt_version),
            tokenize=False,
            add_generation_prompt=True,
        )
        full = tokenizer(full_text, truncation=False)
        prompt = tokenizer(prompt_text, truncation=False)
        if len(full["input_ids"]) > config.max_length:
            raise ValueError(
                f"training sample exceeds max_length for {record.record_id}: "
                f"{len(full['input_ids'])} > {config.max_length}"
            )
        labels = list(full["input_ids"])
        prompt_length = min(len(prompt["input_ids"]), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        if all(label == -100 for label in labels):
            raise ValueError(f"assistant target was fully truncated for {record.record_id}")
        return {
            "input_ids": list(full["input_ids"]),
            "attention_mask": list(full["attention_mask"]),
            "labels": labels,
        }

    train_dataset = Dataset.from_list([encode(record) for record in train_records])
    validation_dataset = Dataset.from_list([encode(record) for record in validation_records])
    load_options: dict[str, Any] = {
        "revision": config.model_revision,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    if config.use_4bit:
        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_options["torch_dtype"] = torch.bfloat16
    model: Any = AutoModelForCausalLM.from_pretrained(config.model_id, **load_options)
    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = (
        get_last_checkpoint(str(output_dir)) if config.resume_from_checkpoint else None
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=config.seed,
        data_seed=config.seed,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        # Transformers 5 accepts a fractional warmup through warmup_steps.
        warmup_steps=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if config.use_4bit else "adamw_torch_fused",
        report_to=[],
        disable_tqdm=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    duration_seconds = time.perf_counter() - started
    evaluation = trainer.evaluate()
    final_dir = output_dir / "final-adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    config_payload = config.model_dump(mode="json")
    config_hash = _stable_hash(config_payload)
    manifest_path = config.data_dir / "manifest.json"
    dataset_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    )
    report = {
        "schema_version": "1.0",
        "evidence_kind": "measured",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "resolved_model_revision": resolved_model_revision,
        "training_method": "qlora-nf4" if config.use_4bit else "lora-bf16",
        "config": config_payload,
        "training_config_sha256": config_hash,
        "resumed_from_checkpoint": last_checkpoint,
        "dataset_manifest_sha256": (
            dataset_manifest["dataset_manifest_sha256"] if dataset_manifest else None
        ),
        "dataset_generator_version": (
            dataset_manifest["generator_version"] if dataset_manifest else None
        ),
        "dataset_file_sha256": (
            {name: details["sha256"] for name, details in dataset_manifest["files"].items()}
            if dataset_manifest
            else None
        ),
        "duration_seconds": round(duration_seconds, 3),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": round(trainable / total * 100, 6),
        "train_metrics": train_result.metrics,
        "eval_metrics": evaluation,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "final_adapter": str(final_dir.relative_to(Path.cwd())),
        "final_adapter_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(final_dir.iterdir())
            if path.is_file()
        },
    }
    report_path = output_dir / "training-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def merge_adapter(
    *, model_id: str, adapter_path: Path, output_dir: Path, model_revision: str = "main"
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if use_cuda else "cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    resolved_model_revision = getattr(base.config, "_commit_hash", None)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision, use_fast=True)
    tokenizer.save_pretrained(output_dir)
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    report = {
        "schema_version": "1.0",
        "evidence_kind": "measured",
        "model_id": model_id,
        "model_revision": model_revision,
        "resolved_model_revision": resolved_model_revision,
        "merge_device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated() if use_cuda else None,
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved() if use_cuda else None,
        "adapter_path": str(adapter_path),
        "output_dir": str(output_dir),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "files": files,
    }
    (output_dir / "merge-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
