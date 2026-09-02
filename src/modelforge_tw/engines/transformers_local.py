from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Literal

from modelforge_tw.model_identity import (
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_MAX_NEW_TOKENS,
    validate_immutable_model_revision,
    validate_model_version,
    validate_resolved_model_revision,
    validate_tokenizer_snapshot_revision,
)
from modelforge_tw.prompts import DEFAULT_PROMPT_VERSION, PromptVersion, messages_for_version
from modelforge_tw.safety import (
    SAFETY_POLICY_VERSION,
    SafetyPolicyVersion,
    apply_safety_policy,
    conservative_fallback,
    parse_output,
)
from modelforge_tw.schemas import InferenceMetadata, InferenceResult, TriageRequest


def _validate_context_budget(
    *, prompt_tokens: int, max_new_tokens: int | None, context_window_tokens: int
) -> None:
    if (
        prompt_tokens < 0
        or max_new_tokens is None
        or max_new_tokens < 1
        or context_window_tokens < 1
    ):
        raise ValueError("token budgets must be positive")
    if prompt_tokens + max_new_tokens > context_window_tokens:
        raise ValueError("prompt and generation budget exceed the configured context window")


class TransformersInferenceEngine:
    """Local measured inference; heavy dependencies are imported only when selected."""

    def __init__(
        self,
        *,
        model_id: str,
        adapter_path: str | None,
        adapter_training_report: str | None = None,
        prompt_version: PromptVersion,
        model_revision: str = DEFAULT_BASE_MODEL_REVISION,
        load_in_4bit: bool = False,
        model_version: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        context_window_tokens: int = 2048,
        safety_policy_version: SafetyPolicyVersion = SAFETY_POLICY_VERSION,
    ) -> None:
        validate_immutable_model_revision(model_revision)
        validate_model_version(model_version)
        _validate_context_budget(
            prompt_tokens=1,
            max_new_tokens=max_new_tokens,
            context_window_tokens=context_window_tokens,
        )
        if bool(adapter_path) != bool(adapter_training_report):
            raise ValueError("adapter and adapter training report must be provided together")
        if adapter_path and adapter_training_report:
            from modelforge_tw.training import verify_adapter_training_lineage

            verify_adapter_training_lineage(
                model_id=model_id,
                model_revision=model_revision,
                adapter_path=Path(adapter_path),
                training_report_path=Path(adapter_training_report),
            )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.name = "transformers-local"
        self.model_id = model_id
        self.model_version = model_version
        self.model_revision = model_revision
        self.quantization = "bnb-nf4" if load_in_4bit else "bf16"
        self.prompt_version = prompt_version
        self.evidence_kind: Literal["fixture", "measured"] = "measured"
        self.max_new_tokens: int | None = max_new_tokens
        self.context_window_tokens = context_window_tokens
        self.structured_output_contract: str = "prompt-schema+bounded-repair-v1"
        self.safety_policy_version: SafetyPolicyVersion = safety_policy_version
        self._lock = threading.Lock()
        self._torch = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            model_id,
            revision=model_revision,
            use_fast=True,
        )
        validate_tokenizer_snapshot_revision(model_revision, self.tokenizer.init_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_options: dict[str, Any] = {
            "revision": model_revision,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        if load_in_4bit:
            load_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            load_options["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

        model: Any = AutoModelForCausalLM.from_pretrained(model_id, **load_options)
        validate_resolved_model_revision(
            model_revision,
            getattr(model.config, "_commit_hash", None),
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model: Any = model
        self.model.eval()

    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _render(self, request: TriageRequest) -> str:
        messages = messages_for_version(request, self.prompt_version)
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if not isinstance(rendered, str):
            raise TypeError("chat template did not return text")
        return rendered

    def generate(self, request: TriageRequest) -> InferenceResult:
        torch = self._torch
        prompt = self._render(request)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        _validate_context_budget(
            prompt_tokens=prompt_tokens,
            max_new_tokens=self.max_new_tokens,
            context_window_tokens=self.context_window_tokens,
        )
        target_device = next(self.model.parameters()).device
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        output_ids = generated[0, prompt_tokens:]
        decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        if not isinstance(decoded, str):
            raise TypeError("tokenizer decode did not return text")
        raw = decoded.strip()
        try:
            parsed = parse_output(raw, allow_bounded_repair=True)
            output, warnings = apply_safety_policy(
                request,
                parsed.output,
                policy_version=self.safety_policy_version,
            )
            repair_used = parsed.repair_used
        except (ValueError, TypeError):
            output = conservative_fallback("MODEL_OUTPUT_INVALID")
            warnings = ["MODEL_OUTPUT_INVALID", "SAFE_FALLBACK_APPLIED"]
            output, policy_warnings = apply_safety_policy(
                request,
                output,
                policy_version=self.safety_policy_version,
            )
            warnings.extend(policy_warnings)
            repair_used = False
        output_tokens = int(output_ids.shape[-1])
        return InferenceResult(
            raw_text=raw,
            output=output,
            metadata=InferenceMetadata(
                engine=self.name,
                model_id=self.model_id,
                model_version=self.model_version,
                model_revision=self.model_revision,
                quantization=self.quantization,
                prompt_version=self.prompt_version,
                elapsed_ms=elapsed_ms,
                prompt_tokens=int(prompt_tokens),
                output_tokens=output_tokens,
                generation_tokens_per_second=(
                    output_tokens / (elapsed_ms / 1000) if elapsed_ms else None
                ),
                max_new_tokens=self.max_new_tokens,
                context_window_tokens=self.context_window_tokens,
                structured_output_contract=self.structured_output_contract,
                safety_policy_version=self.safety_policy_version,
                evidence_kind="measured",
            ),
            repair_used=repair_used,
            validation_warnings=warnings,
        )


def model_load_probe(
    model_id: str,
    *,
    model_revision: str = DEFAULT_BASE_MODEL_REVISION,
    load_in_4bit: bool,
) -> dict[str, Any]:
    validate_immutable_model_revision(model_revision)
    model_version = f"base-{model_revision[:12]}"
    import torch

    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    started = time.perf_counter()
    engine = TransformersInferenceEngine(
        model_id=model_id,
        model_revision=model_revision,
        adapter_path=None,
        prompt_version=DEFAULT_PROMPT_VERSION,
        model_version=model_version,
        load_in_4bit=load_in_4bit,
    )
    load_seconds = time.perf_counter() - started
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    return {
        "model_id": model_id,
        "model_revision": model_revision,
        "model_version": model_version,
        "quantization": engine.quantization,
        "load_seconds": round(load_seconds, 4),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_allocated_before_bytes": before,
        "vram_allocated_after_bytes": after,
        "tokenizer_vocab_size": len(engine.tokenizer),
        "status": "loaded",
        "evidence_kind": "measured",
    }
