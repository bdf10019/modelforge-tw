from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from modelforge_tw.safety import SAFETY_POLICY_VERSION, SafetyPolicyVersion


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODELFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    engine: str = "fixture"
    model_id: str = "Qwen/Qwen3-1.7B"
    model_revision: str = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    model_version: str = "qwen3-1.7b-qlora-v5"
    quantization: str = "bnb-nf4"
    prompt_version: str = "optimized-v2"
    adapter_path: str | None = None
    load_in_4bit: bool = True
    max_new_tokens: int = Field(default=192, ge=32, le=512)
    context_window_tokens: int = Field(default=2048, ge=1536, le=131072)
    safety_policy_version: SafetyPolicyVersion = SAFETY_POLICY_VERSION
    llama_base_url: str = "http://127.0.0.1:8081"
    registry_path: Path = Path("outputs/model-registry.json")
    registry_deployment_role: Literal["released", "previous-good"] | None = None
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    queue_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    max_queue_depth: int = Field(default=8, ge=0, le=1024)
    shutdown_grace_seconds: float = Field(default=35.0, ge=0, le=300)
    max_concurrency: int = Field(default=2, ge=1, le=64)
    max_request_body_bytes: int = Field(default=32768, ge=1024, le=1048576)
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

    @field_validator("api_key")
    @classmethod
    def require_strong_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("API key must contain at least 32 characters")
        return value
