from __future__ import annotations

import re

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_BASE_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_MAX_NEW_TOKENS = 192
IMMUTABLE_MODEL_REVISION = re.compile(r"^[a-f0-9]{40,64}$")
MODEL_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
QUANTIZATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def validate_immutable_model_revision(value: str) -> str:
    if not IMMUTABLE_MODEL_REVISION.fullmatch(value):
        raise ValueError("model revision must be a 40-64 character lowercase commit hash")
    return value


def validate_model_version(value: str) -> str:
    if value == "unversioned" or not MODEL_VERSION.fullmatch(value):
        raise ValueError("model version must be an explicit safe deployment identifier")
    return value


def validate_quantization(value: str) -> str:
    if value == "unknown" or not QUANTIZATION_ID.fullmatch(value):
        raise ValueError("quantization must be an explicit safe identifier")
    return value
