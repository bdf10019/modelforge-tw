from typing import Literal, Protocol

from modelforge_tw.prompts import PromptVersion
from modelforge_tw.safety import SafetyPolicyVersion
from modelforge_tw.schemas import InferenceMetadata, InferenceResult, TriageRequest

INFERENCE_IDENTITY_BINDINGS = (
    ("engine", "name"),
    ("model_id", "model_id"),
    ("model_version", "model_version"),
    ("model_revision", "model_revision"),
    ("quantization", "quantization"),
    ("prompt_version", "prompt_version"),
    ("max_new_tokens", "max_new_tokens"),
    ("context_window_tokens", "context_window_tokens"),
    ("structured_output_contract", "structured_output_contract"),
    ("safety_policy_version", "safety_policy_version"),
    ("evidence_kind", "evidence_kind"),
)


class InferenceIdentityMismatchError(ValueError):
    pass


class InferenceEngine(Protocol):
    name: str
    model_id: str
    model_version: str
    quantization: str
    prompt_version: PromptVersion
    max_new_tokens: int | None
    structured_output_contract: str
    safety_policy_version: SafetyPolicyVersion
    evidence_kind: Literal["fixture", "measured"]

    def generate(self, request: TriageRequest) -> InferenceResult: ...

    def ready(self) -> bool: ...


def validate_inference_metadata_identity(
    engine: InferenceEngine, metadata: InferenceMetadata
) -> None:
    mismatches = [
        metadata_field
        for metadata_field, engine_field in INFERENCE_IDENTITY_BINDINGS
        if getattr(metadata, metadata_field) != getattr(engine, engine_field, None)
    ]
    if mismatches:
        raise InferenceIdentityMismatchError(
            "inference metadata does not match configured engine: " + ", ".join(mismatches)
        )


def validate_inference_result(engine: InferenceEngine, result: InferenceResult) -> InferenceResult:
    validated = InferenceResult.model_validate(result.model_dump(mode="python"))
    validate_inference_metadata_identity(engine, validated.metadata)
    return validated
