from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Literal

from modelforge_tw.prompts import messages_for_version
from modelforge_tw.safety import (
    SAFETY_POLICY_VERSION,
    apply_safety_policy,
    conservative_fallback,
    parse_output,
)
from modelforge_tw.schemas import InferenceMetadata, InferenceResult, TriageRequest


class TransformersInferenceEngine:
    """Local measured inference; heavy dependencies are imported only when selected."""

    def __init__(
        self,
        *,
        model_id: str,
        adapter_path: str | None,
        prompt_version: str,
        model_revision: str = "main",
        load_in_4bit: bool = False,
        model_version: str = "unversioned",
        max_new_tokens: int = 256,
    ) -> None:
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
        self.structured_output_contract: str = "prompt-schema+bounded-repair-v1"
        self.safety_policy_version: str = SAFETY_POLICY_VERSION
        self._lock = threading.Lock()
        self._torch = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            model_id,
            revision=model_revision,
            use_fast=True,
        )
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
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            if self.model_version == "unversioned":
                adapter = Path(adapter_path)
                self.model_version = adapter.parent.name if adapter.parent.name else adapter.name
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
        prompt_tokens = inputs["input_ids"].shape[-1]
        output_ids = generated[0, prompt_tokens:]
        decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        if not isinstance(decoded, str):
            raise TypeError("tokenizer decode did not return text")
        raw = decoded.strip()
        try:
            parsed = parse_output(raw, allow_bounded_repair=True)
            output, warnings = apply_safety_policy(request, parsed.output)
            repair_used = parsed.repair_used
        except (ValueError, TypeError):
            output = conservative_fallback("MODEL_OUTPUT_INVALID")
            warnings = ["MODEL_OUTPUT_INVALID", "SAFE_FALLBACK_APPLIED"]
            output, policy_warnings = apply_safety_policy(request, output)
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
                structured_output_contract=self.structured_output_contract,
                safety_policy_version=self.safety_policy_version,
                evidence_kind="measured",
            ),
            repair_used=repair_used,
            validation_warnings=warnings,
        )


def model_load_probe(model_id: str, *, load_in_4bit: bool) -> dict[str, Any]:
    import torch

    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    started = time.perf_counter()
    engine = TransformersInferenceEngine(
        model_id=model_id,
        adapter_path=None,
        prompt_version="optimized-v1",
        load_in_4bit=load_in_4bit,
    )
    load_seconds = time.perf_counter() - started
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    return {
        "model_id": model_id,
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
