from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from modelforge_tw.model_identity import DEFAULT_MAX_NEW_TOKENS
from modelforge_tw.prompts import PromptVersion, messages_for_version
from modelforge_tw.safety import (
    SAFETY_POLICY_VERSION,
    SafetyPolicyVersion,
    apply_safety_policy,
    conservative_fallback,
    parse_output,
)
from modelforge_tw.schemas import (
    ALLOWED_REASON_CODES,
    InferenceMetadata,
    InferenceResult,
    TriageOutput,
    TriageRequest,
)

OPENAI_HTTP_CONTRACT_VERSION = "json-schema-v2"
DEFAULT_MAX_MODEL_RESPONSE_BYTES = 256 * 1024
MAX_MODEL_RESPONSE_BYTES = 4 * 1024 * 1024


def _validate_declared_response_size(response: httpx.Response, max_bytes: int) -> None:
    values = response.headers.get_list("content-length")
    if not values:
        return
    if len(values) != 1 or not values[0].isdigit():
        raise ValueError("model response has an invalid Content-Length")
    if int(values[0]) > max_bytes:
        raise ValueError("model response exceeds byte limit")


def _read_bounded_response(response: httpx.Response, max_bytes: int) -> bytes:
    _validate_declared_response_size(response, max_bytes)
    body = bytearray()
    for chunk in response.iter_bytes():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ValueError("model response exceeds byte limit")
    return bytes(body)


def _iter_bounded_response_lines(response: httpx.Response, max_bytes: int) -> Iterator[bytes]:
    _validate_declared_response_size(response, max_bytes)
    total = 0
    buffer = bytearray()
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("model response exceeds byte limit")
        buffer.extend(chunk)
        while (newline := buffer.find(b"\n")) >= 0:
            yield bytes(buffer[:newline]).removesuffix(b"\r")
            del buffer[: newline + 1]
    if buffer:
        yield bytes(buffer).removesuffix(b"\r")


def _structured_response_format() -> dict[str, Any]:
    schema = TriageOutput.model_json_schema()
    schema["properties"]["reason_codes"]["items"] = {
        "type": "string",
        "enum": list(ALLOWED_REASON_CODES),
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "triage_output",
            "strict": True,
            "schema": schema,
        },
    }


def _validated_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as error:
        raise ValueError("model server base URL is invalid") from error
    if (
        value != value.strip()
        or any(ord(character) < 32 for character in value)
        or parsed.scheme.lower() not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "model server base URL must be HTTP(S) without credentials, query, or fragment"
        )
    return value.rstrip("/")


class OpenAIHTTPInferenceEngine:
    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        model_version: str,
        quantization: str,
        prompt_version: PromptVersion,
        timeout_seconds: float = 60,
        max_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        context_window_tokens: int = 2048,
        max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES,
        safety_policy_version: SafetyPolicyVersion = SAFETY_POLICY_VERSION,
    ) -> None:
        self.name = "openai-http"
        self.base_url = _validated_base_url(base_url)
        self.model_id = model_id
        self.model_version = model_version
        self.quantization = quantization
        self.prompt_version = prompt_version
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.max_new_tokens: int | None = max_tokens
        self.context_window_tokens = context_window_tokens
        if not 4096 <= max_response_bytes <= MAX_MODEL_RESPONSE_BYTES:
            raise ValueError("model response byte limit must be between 4096 and 4194304")
        self.max_response_bytes = max_response_bytes
        self.structured_output_contract: str = OPENAI_HTTP_CONTRACT_VERSION
        self.safety_policy_version: SafetyPolicyVersion = safety_policy_version
        self.evidence_kind: Literal["fixture", "measured"] = "measured"

    def ready(self) -> bool:
        try:
            with httpx.Client(timeout=min(self.timeout_seconds, 3.0), trust_env=False) as client:
                with client.stream("GET", f"{self.base_url}/health") as response:
                    return response.status_code == 200
        except httpx.HTTPError:
            return False

    def generate(self, request: TriageRequest) -> InferenceResult:
        payload = {
            "model": self.model_id,
            "messages": messages_for_version(request, self.prompt_version),
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
            "response_format": _structured_response_format(),
        }
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
            with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                body = json.loads(_read_bounded_response(response, self.max_response_bytes))
        elapsed_ms = (time.perf_counter() - started) * 1000
        raw = body["choices"][0]["message"]["content"]
        return self._build_result(
            request=request,
            raw=raw,
            elapsed_ms=elapsed_ms,
            usage=body.get("usage", {}),
        )

    def generate_streaming(self, request: TriageRequest) -> InferenceResult:
        payload = {
            "model": self.model_id,
            "messages": messages_for_version(request, self.prompt_version),
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
            "response_format": _structured_response_format(),
        }
        started = time.perf_counter()
        first_content_ms: float | None = None
        parts: list[str] = []
        usage: dict[str, Any] = {}
        with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
            with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                for raw_line in _iter_bounded_response_lines(response, self.max_response_bytes):
                    try:
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise ValueError("model response is not valid UTF-8") from error
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if not data or data == "[DONE]":
                        continue
                    event = json.loads(data)
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices", [])
                    if not choices:
                        continue
                    content = choices[0].get("delta", {}).get("content")
                    if content:
                        if first_content_ms is None:
                            first_content_ms = (time.perf_counter() - started) * 1000
                        parts.append(content)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return self._build_result(
            request=request,
            raw="".join(parts),
            elapsed_ms=elapsed_ms,
            usage=usage,
            ttft_ms=first_content_ms,
        )

    def _build_result(
        self,
        *,
        request: TriageRequest,
        raw: str,
        elapsed_ms: float,
        usage: dict[str, Any],
        ttft_ms: float | None = None,
    ) -> InferenceResult:
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
        output_tokens = usage.get("completion_tokens")
        return InferenceResult(
            raw_text=raw,
            output=output,
            metadata=InferenceMetadata(
                engine=self.name,
                model_id=self.model_id,
                model_version=self.model_version,
                quantization=self.quantization,
                prompt_version=self.prompt_version,
                elapsed_ms=elapsed_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                output_tokens=output_tokens,
                generation_tokens_per_second=(
                    output_tokens / (elapsed_ms / 1000) if output_tokens and elapsed_ms else None
                ),
                ttft_ms=ttft_ms,
                max_new_tokens=self.max_new_tokens,
                context_window_tokens=self.context_window_tokens,
                structured_output_contract=self.structured_output_contract,
                safety_policy_version=self.safety_policy_version,
                evidence_kind="measured",
            ),
            repair_used=repair_used,
            validation_warnings=warnings,
        )
