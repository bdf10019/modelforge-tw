from __future__ import annotations

import re
from collections.abc import Mapping

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_BASE_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_MAX_NEW_TOKENS = 192
IMMUTABLE_MODEL_REVISION = re.compile(r"^[a-f0-9]{40,64}$")
MODEL_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
QUANTIZATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
HUGGINGFACE_SNAPSHOT_REVISION = re.compile(r"(?:^|[/\\])snapshots[/\\]([a-f0-9]{40,64})(?:[/\\])")


def validate_immutable_model_revision(value: str) -> str:
    if not IMMUTABLE_MODEL_REVISION.fullmatch(value):
        raise ValueError("model revision must be a 40-64 character lowercase commit hash")
    return value


def validate_resolved_model_revision(requested: str, resolved: object) -> str:
    """Require the downloaded model to resolve to the exact requested commit."""
    validate_immutable_model_revision(requested)
    if not isinstance(resolved, str):
        raise ValueError("model loader did not report a resolved immutable revision")
    validate_immutable_model_revision(resolved)
    if resolved != requested:
        raise ValueError("resolved model revision does not match requested revision")
    return resolved


def validate_tokenizer_snapshot_revision(
    requested: str,
    init_kwargs: Mapping[str, object],
) -> str:
    """Verify tokenizer asset paths resolve under one exact Hub snapshot."""
    validate_immutable_model_revision(requested)
    revisions = {
        match.group(1)
        for key, value in init_kwargs.items()
        if key.endswith("_file") and isinstance(value, str)
        if (match := HUGGINGFACE_SNAPSHOT_REVISION.search(value)) is not None
    }
    if not revisions:
        raise ValueError("tokenizer loader did not report a resolved snapshot revision")
    if len(revisions) != 1:
        raise ValueError("tokenizer assets resolved from multiple snapshot revisions")
    return validate_resolved_model_revision(requested, revisions.pop())


def validate_model_version(value: str) -> str:
    if value == "unversioned" or not MODEL_VERSION.fullmatch(value):
        raise ValueError("model version must be an explicit safe deployment identifier")
    return value


def validate_quantization(value: str) -> str:
    if value == "unknown" or not QUANTIZATION_ID.fullmatch(value):
        raise ValueError("quantization must be an explicit safe identifier")
    return value
