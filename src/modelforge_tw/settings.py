from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from modelforge_tw.model_identity import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_MAX_NEW_TOKENS,
    validate_model_version,
)
from modelforge_tw.prompts import DEFAULT_PROMPT_VERSION, PromptVersion
from modelforge_tw.safety import SAFETY_POLICY_VERSION, SafetyPolicyVersion


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODELFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    engine: str = "fixture"
    model_id: str = DEFAULT_BASE_MODEL_ID
    model_revision: str = Field(default=DEFAULT_BASE_MODEL_REVISION, pattern=r"^[a-f0-9]{40,64}$")
    model_version: str = "qwen3-1.7b-qlora-v5"
    quantization: str = "bnb-nf4"
    prompt_version: PromptVersion = DEFAULT_PROMPT_VERSION
    adapter_path: str | None = None
    load_in_4bit: bool = True
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=32, le=512)
    context_window_tokens: int = Field(default=2048, ge=1536, le=131072)
    safety_policy_version: SafetyPolicyVersion = SAFETY_POLICY_VERSION
    llama_base_url: str = "http://127.0.0.1:8081"
    registry_path: Path = Path("outputs/model-registry.json")
    registry_deployment_role: Literal["released", "previous-good"] | None = None
    registry_artifact_root: Path | None = None
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    readiness_timeout_seconds: float = Field(default=4.0, ge=0.1, le=10)
    queue_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    max_queue_depth: int = Field(default=8, ge=0, le=1024)
    shutdown_grace_seconds: float = Field(default=35.0, ge=0, le=300)
    max_concurrency: int = Field(default=2, ge=1, le=64)
    max_request_body_bytes: int = Field(default=32768, ge=1024, le=1048576)
    max_model_response_bytes: int = Field(default=262144, ge=4096, le=4194304)
    api_key: SecretStr | None = None
    log_inputs: Literal[False] = False

    @field_validator("api_key", mode="before")
    @classmethod
    def empty_api_key_disables_authentication(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("registry_deployment_role", mode="before")
    @classmethod
    def empty_registry_role_disables_binding(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("registry_artifact_root", mode="before")
    @classmethod
    def empty_registry_artifact_root_disables_file_verification(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("model_version")
    @classmethod
    def model_version_is_explicit(cls, value: str) -> str:
        return validate_model_version(value)

    @field_validator("api_key")
    @classmethod
    def require_strong_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("API key must contain at least 32 characters")
        return value
