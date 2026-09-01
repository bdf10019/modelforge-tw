from pathlib import Path

from pydantic import Field
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
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_concurrency: int = Field(default=2, ge=1, le=64)
    log_inputs: bool = False
