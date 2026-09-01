from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from modelforge_tw.io_utils import atomic_write_text
from modelforge_tw.schemas import EvaluationReport

SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


def _validate_portable_relative_path(value: str) -> str:
    parts = value.split("/")
    windows_path = PureWindowsPath(value)
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or any(ord(character) < 32 for character in value)
        or PurePosixPath(value).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("path must be a canonical portable relative path")
    return value


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["adapter", "merged", "gguf", "tokenizer"]
    relative_path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_safe(cls, value: str) -> str:
        return _validate_portable_relative_path(value)


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_id: str
    base_model: str
    base_revision: str
    license: str
    prompt_version: str
    dataset_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    release_gate_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    engine: str
    quantization: str
    max_new_tokens: int | None = Field(default=None, ge=1)
    structured_output_contract: str | None = None
    safety_policy_version: str | None = None
    evaluation_contract_version: str | None = None
    evaluation_dataset_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    evaluation_engine: str | None = None
    evaluation_model_id: str | None = None
    evaluation_model_version: str | None = None
    evaluation_evidence_kind: Literal["fixture", "measured"] | None = None
    status: Literal["candidate", "released", "retired", "blocked"] = "candidate"
    evaluation_report: str
    release_passed: bool
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @field_validator("evaluation_report")
    @classmethod
    def evaluation_report_is_safe(cls, value: str) -> str:
        return _validate_portable_relative_path(value)


class RegistryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.1"
    released: str | None = None
    previous_good: str | None = None
    manifests: dict[str, ModelManifest] = Field(default_factory=dict)
    events: list[dict[str, str]] = Field(default_factory=list)


class FileRegistry:
    def __init__(self, path: Path, *, artifact_root: Path | None = None) -> None:
        self.path = path
        self.artifact_root = artifact_root.resolve() if artifact_root else None

    def load(self) -> RegistryState:
        if not self.path.exists():
            return RegistryState()
        return RegistryState.model_validate_json(self.path.read_text(encoding="utf-8"))

    def _write(self, state: RegistryState) -> None:
        atomic_write_text(
            self.path,
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )

    @staticmethod
    def _validate_id(manifest_id: str) -> None:
        if not SAFE_ID.fullmatch(manifest_id):
            raise ValueError("manifest id contains unsupported characters")

    @staticmethod
    def _require_evaluation_attestation(manifest: ModelManifest) -> None:
        attestation = {
            "release_gate_sha256": manifest.release_gate_sha256,
            "max_new_tokens": manifest.max_new_tokens,
            "structured_output_contract": manifest.structured_output_contract,
            "safety_policy_version": manifest.safety_policy_version,
            "evaluation_contract_version": manifest.evaluation_contract_version,
            "evaluation_dataset_sha256": manifest.evaluation_dataset_sha256,
            "evaluation_engine": manifest.evaluation_engine,
            "evaluation_model_id": manifest.evaluation_model_id,
            "evaluation_model_version": manifest.evaluation_model_version,
            "evaluation_evidence_kind": manifest.evaluation_evidence_kind,
        }
        missing = [name for name, value in attestation.items() if value is None]
        if missing:
            raise ValueError("manifest lacks evaluation attestation: " + ", ".join(sorted(missing)))

    @classmethod
    def _require_releasable_evidence(cls, manifest: ModelManifest) -> None:
        cls._require_evaluation_attestation(manifest)
        if manifest.evaluation_evidence_kind != "measured":
            raise ValueError("only measured evaluation evidence can be released")

    @classmethod
    def _require_rollback_evidence(cls, state: RegistryState, manifest: ModelManifest) -> None:
        legacy_identity = (
            manifest.evaluation_dataset_sha256,
            manifest.evaluation_engine,
            manifest.evaluation_model_id,
            manifest.evaluation_model_version,
            manifest.evaluation_evidence_kind,
        )
        if state.schema_version == "1.0" and all(value is None for value in legacy_identity):
            return
        cls._require_releasable_evidence(manifest)

    def _verify_artifacts(self, manifest: ModelManifest) -> None:
        if self.artifact_root is None:
            return
        for artifact in manifest.artifacts:
            path = (self.artifact_root / artifact.relative_path).resolve()
            try:
                path.relative_to(self.artifact_root)
            except ValueError as error:
                raise ValueError("registered artifact escapes the artifact root") from error
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != artifact.bytes:
                raise ValueError(f"registered artifact size changed: {artifact.relative_path}")
            digest = sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != artifact.sha256:
                raise ValueError(f"registered artifact hash changed: {artifact.relative_path}")

    def validate_evaluation(self, manifest_id: str, evaluation: EvaluationReport) -> None:
        self._validate_id(manifest_id)
        state = self.load()
        if manifest_id not in state.manifests:
            raise KeyError(f"manifest not found: {manifest_id}")
        manifest = state.manifests[manifest_id]
        self._require_evaluation_attestation(manifest)
        checks = {
            "evaluation_dataset_sha256": (
                manifest.evaluation_dataset_sha256 == evaluation.dataset_manifest_sha256
            ),
            "evaluation_engine": manifest.evaluation_engine == evaluation.engine,
            "evaluation_model_id": manifest.evaluation_model_id == evaluation.model_id,
            "evaluation_model_version": (
                manifest.evaluation_model_version == evaluation.model_version
            ),
            "evaluation_evidence_kind": (
                manifest.evaluation_evidence_kind == evaluation.evidence_kind
            ),
            "prompt_version": manifest.prompt_version == evaluation.prompt_version,
            "release_gate_sha256": (manifest.release_gate_sha256 == evaluation.release_gate_sha256),
            "quantization": manifest.quantization == evaluation.quantization,
            "max_new_tokens": (
                manifest.max_new_tokens is None
                or manifest.max_new_tokens == evaluation.max_new_tokens
            ),
            "structured_output_contract": (
                manifest.structured_output_contract is None
                or manifest.structured_output_contract == evaluation.structured_output_contract
            ),
            "safety_policy_version": (
                manifest.safety_policy_version is None
                or manifest.safety_policy_version == evaluation.safety_policy_version
            ),
            "evaluation_contract_version": (
                manifest.evaluation_contract_version is None
                or manifest.evaluation_contract_version == evaluation.evaluation_contract_version
            ),
        }
        mismatches = [name for name, matched in checks.items() if not matched]
        if mismatches:
            raise ValueError("evaluation does not match manifest: " + ", ".join(sorted(mismatches)))

    def attest_evaluation(
        self,
        manifest_id: str,
        *,
        evaluation_report: str,
        evaluation: EvaluationReport,
    ) -> RegistryState:
        evaluation_report = _validate_portable_relative_path(evaluation_report)
        self.validate_evaluation(manifest_id, evaluation)
        required_contract = {
            "max_new_tokens": evaluation.max_new_tokens,
            "structured_output_contract": evaluation.structured_output_contract,
            "safety_policy_version": evaluation.safety_policy_version,
            "evaluation_contract_version": evaluation.evaluation_contract_version,
        }
        missing = [name for name, value in required_contract.items() if value is None]
        if missing:
            raise ValueError("evaluation cannot be attested without: " + ", ".join(missing))
        state = self.load()
        manifest = state.manifests[manifest_id]
        if manifest.status == "released" and not evaluation.release_passed:
            raise ValueError("cannot attest a failed evaluation to the released manifest")
        state.manifests[manifest_id] = manifest.model_copy(
            update={
                **required_contract,
                "evaluation_report": evaluation_report,
                "release_passed": evaluation.release_passed,
            }
        )
        state.events.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "action": "evaluation_attested",
                "manifest_id": manifest_id,
                "evaluation_report": evaluation_report,
            }
        )
        self._write(state)
        return state

    def add(self, manifest: ModelManifest) -> RegistryState:
        self._validate_id(manifest.manifest_id)
        self._require_evaluation_attestation(manifest)
        state = self.load()
        if manifest.manifest_id in state.manifests:
            raise ValueError(f"manifest already exists: {manifest.manifest_id}")
        state.manifests[manifest.manifest_id] = manifest
        state.events.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "action": "candidate_added",
                "manifest_id": manifest.manifest_id,
            }
        )
        self._write(state)
        return state

    def promote(self, manifest_id: str) -> RegistryState:
        self._validate_id(manifest_id)
        state = self.load()
        if manifest_id not in state.manifests:
            raise KeyError(f"manifest not found: {manifest_id}")
        candidate = state.manifests[manifest_id]
        self._require_releasable_evidence(candidate)
        if not candidate.release_passed or candidate.status == "blocked":
            raise ValueError("candidate did not pass its release gates")
        self._verify_artifacts(candidate)
        if state.released and state.released != manifest_id:
            current = state.manifests[state.released]
            state.manifests[state.released] = current.model_copy(update={"status": "retired"})
            state.previous_good = state.released
        state.manifests[manifest_id] = candidate.model_copy(update={"status": "released"})
        state.released = manifest_id
        state.events.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "action": "promoted",
                "manifest_id": manifest_id,
            }
        )
        self._write(state)
        return state

    def rollback(self) -> RegistryState:
        state = self.load()
        if not state.previous_good:
            raise ValueError("no previous-good manifest is available")
        target = state.previous_good
        current = state.released
        target_manifest = state.manifests[target]
        self._require_rollback_evidence(state, target_manifest)
        if not target_manifest.release_passed:
            raise ValueError("previous-good manifest no longer satisfies release metadata")
        self._verify_artifacts(target_manifest)
        if current:
            state.manifests[current] = state.manifests[current].model_copy(
                update={"status": "retired"}
            )
        state.manifests[target] = target_manifest.model_copy(update={"status": "released"})
        state.released = target
        state.previous_good = current
        state.events.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "action": "rolled_back",
                "manifest_id": target,
            }
        )
        self._write(state)
        return state

    def block(
        self,
        manifest_id: str,
        *,
        evaluation_report: str | None = None,
        release_passed: bool | None = None,
    ) -> RegistryState:
        self._validate_id(manifest_id)
        if evaluation_report is not None:
            evaluation_report = _validate_portable_relative_path(evaluation_report)
        state = self.load()
        if manifest_id not in state.manifests:
            raise KeyError(f"manifest not found: {manifest_id}")
        manifest = state.manifests[manifest_id]
        updates: dict[str, object] = {"status": "blocked"}
        if evaluation_report is not None:
            updates["evaluation_report"] = evaluation_report
        if release_passed is not None:
            updates["release_passed"] = release_passed
        state.manifests[manifest_id] = manifest.model_copy(update=updates)
        if state.released == manifest_id:
            state.released = None
        if state.previous_good == manifest_id:
            state.previous_good = None
        state.events.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "action": "blocked",
                "manifest_id": manifest_id,
                **({"evaluation_report": evaluation_report} if evaluation_report else {}),
            }
        )
        self._write(state)
        return state
