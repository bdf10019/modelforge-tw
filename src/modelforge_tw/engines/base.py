from typing import Literal, Protocol

from modelforge_tw.schemas import InferenceResult, TriageRequest


class InferenceEngine(Protocol):
    name: str
    model_id: str
    model_version: str
    quantization: str
    prompt_version: str
    max_new_tokens: int | None
    structured_output_contract: str
    evidence_kind: Literal["fixture", "measured"]

    def generate(self, request: TriageRequest) -> InferenceResult: ...

    def ready(self) -> bool: ...
