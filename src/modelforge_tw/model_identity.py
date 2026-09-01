from __future__ import annotations

import re

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_BASE_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
IMMUTABLE_MODEL_REVISION = re.compile(r"^[a-f0-9]{40,64}$")


def validate_immutable_model_revision(value: str) -> str:
    if not IMMUTABLE_MODEL_REVISION.fullmatch(value):
        raise ValueError("model revision must be a 40-64 character lowercase commit hash")
    return value
