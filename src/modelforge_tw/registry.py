from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modelforge_tw.contracts import PromptVersion, SafetyPolicyVersion
from modelforge_tw.io_utils import (
    atomic_write_text,
    exclusive_file_lock,
    read_utf8_text_limited,
)
from modelforge_tw.model_identity import validate_model_version, validate_quantization
from modelforge_tw.schemas import EvaluationReport
from modelforge_tw.training import TrainingReportAttestation

SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
DeploymentRole = Literal["released", "previous-good"]


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

    manifest_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    base_model: str = Field(min_length=1, max_length=256)
    base_revision: str = Field(pattern=r"^[a-f0-9]{40,64}$")
    license: str = Field(min_length=1, max_length=128)
    prompt_version: PromptVersion
    dataset_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    release_gate_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    engine: str
    quantization: str
    max_new_tokens: int | None = Field(default=None, ge=1)
    structured_output_contract: str | None = None
    safety_policy_version: SafetyPolicyVersion | None = None
    evaluation_contract_version: str | None = None
    evaluation_dataset_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    evaluation_engine: str | None = None
    evaluation_model_id: str | None = None
    evaluation_model_version: str | None = None
    evaluation_evidence_kind: Literal["fixture", "measured"] | None = None
    status: Literal["candidate", "released", "retired", "blocked"] = "candidate"
    evaluation_report: str
    evaluation_report_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    training_report: str | None = None
    training_report_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    release_passed: bool
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @field_validator("evaluation_report", "training_report")
    @classmethod
    def report_path_is_safe(cls, value: str | None) -> str | None:
        return _validate_portable_relative_path(value) if value is not None else None

    @field_validator("evaluation_model_version")
    @classmethod
    def evaluation_model_version_is_explicit(cls, value: str | None) -> str | None:
        return validate_model_version(value) if value is not None else None

    @field_validator("quantization")
    @classmethod
    def quantization_is_explicit(cls, value: str) -> str:
        return validate_quantization(value)


class RegistryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    at: datetime
    action: Literal[
        "candidate_added",
        "evaluation_attested",
        "promoted",
        "rolled_back",
        "blocked",
    ]
    manifest_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    evaluation_report: str | None = None
    evaluation_report_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    previous_event_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    event_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("at")
    @classmethod
    def timestamp_has_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("registry event timestamp must include a timezone")
        return value

    @field_validator("evaluation_report")
    @classmethod
    def event_report_path_is_safe(cls, value: str | None) -> str | None:
        return _validate_portable_relative_path(value) if value is not None else None

    @model_validator(mode="after")
    def report_hash_requires_report(self) -> RegistryEvent:
        if self.evaluation_report_sha256 is not None and self.evaluation_report is None:
            raise ValueError("event evaluation report SHA-256 requires a report path")
        if self.action == "evaluation_attested" and self.evaluation_report is None:
            raise ValueError("evaluation attestation event requires a report path")
        if self.action not in {"evaluation_attested", "blocked"} and (
            self.evaluation_report is not None or self.evaluation_report_sha256 is not None
        ):
            raise ValueError("event action cannot include an evaluation report")
        return self


def _registry_event_sha256(event: RegistryEvent) -> str:
    payload = event.model_dump(mode="json", exclude={"event_sha256"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


class RegistryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "1.1", "1.2", "1.3", "1.4"] = "1.4"
    released: str | None = None
    previous_good: str | None = None
    manifests: dict[str, ModelManifest] = Field(default_factory=dict)
    events: list[RegistryEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state_consistency(self) -> RegistryState:
        mismatched_ids = sorted(
            key for key, manifest in self.manifests.items() if key != manifest.manifest_id
        )
        if mismatched_ids:
            raise ValueError("registry manifest keys do not match manifest IDs")

        released_ids = sorted(
            key for key, manifest in self.manifests.items() if manifest.status == "released"
        )
        expected_released = [self.released] if self.released is not None else []
        if released_ids != expected_released:
            raise ValueError("registry released pointer does not match manifest statuses")
        if self.released is not None and not self.manifests[self.released].release_passed:
            raise ValueError("released manifest does not have a passing release result")

        if self.previous_good is not None:
            if self.previous_good == self.released:
                raise ValueError("previous-good manifest cannot equal the released manifest")
            previous = self.manifests.get(self.previous_good)
            if previous is None or previous.status != "retired":
                raise ValueError("previous-good pointer must reference a retired manifest")

        missing_event_manifests = sorted(
            {event.manifest_id for event in self.events} - self.manifests.keys()
        )
        if missing_event_manifests:
            raise ValueError("registry events reference missing manifests")

        for previous_event, current_event in zip(self.events, self.events[1:], strict=False):
            if current_event.at < previous_event.at:
                raise ValueError("registry event timestamps are not chronological")

        if self.schema_version == "1.4":
            previous_digest: str | None = None
            for event in self.events:
                if event.previous_event_sha256 != previous_digest:
                    raise ValueError("registry event chain has a broken previous hash")
                if event.event_sha256 is None or event.event_sha256 != _registry_event_sha256(
                    event
                ):
                    raise ValueError("registry event chain hash does not match event contents")
                previous_digest = event.event_sha256
        return self


class RegistryAnchor(BaseModel):
    """Portable digest record for externally anchoring one registry snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0", "1.1", "1.2", "1.3", "1.4"]
    registry_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_count: int = Field(ge=0)
    event_chain_verified: bool
    event_chain_head_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class FileRegistry:
    def __init__(self, path: Path, *, artifact_root: Path | None = None) -> None:
        self.path = path
        self.lock_path = path.with_name(f".{path.name}.lock")
        self.artifact_root = artifact_root.resolve() if artifact_root else None

    def load(self) -> RegistryState:
        if not self.path.exists():
            return RegistryState()
        return RegistryState.model_validate_json(read_utf8_text_limited(self.path))

    def anchor(self) -> RegistryAnchor:
        """Validate and hash the exact snapshot read from the registry file."""
        serialized = read_utf8_text_limited(self.path)
        state = RegistryState.model_validate_json(serialized)
        chain_verified = state.schema_version == "1.4"
        chain_head = state.events[-1].event_sha256 if chain_verified and state.events else None
        return RegistryAnchor(
            schema_version=state.schema_version,
            registry_sha256=sha256(serialized.encode("utf-8")).hexdigest(),
            event_count=len(state.events),
            event_chain_verified=chain_verified,
            event_chain_head_sha256=chain_head,
        )

    def _write(self, state: RegistryState) -> None:
        state = RegistryState.model_validate(state.model_dump(mode="json"))
        atomic_write_text(
            self.path,
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )

    @staticmethod
    def _append_event(
        state: RegistryState,
        *,
        action: Literal[
            "candidate_added",
            "evaluation_attested",
            "promoted",
            "rolled_back",
            "blocked",
        ],
        manifest_id: str,
        evaluation_report: str | None = None,
        evaluation_report_sha256: str | None = None,
    ) -> None:
        previous_digest: str | None = None
        chained_events: list[RegistryEvent] = []
        if state.schema_version != "1.4":
            for existing in state.events:
                event = existing.model_copy(
                    update={
                        "previous_event_sha256": previous_digest,
                        "event_sha256": None,
                    }
                )
                event = event.model_copy(update={"event_sha256": _registry_event_sha256(event)})
                chained_events.append(event)
                previous_digest = event.event_sha256
            state.events = chained_events
            state.schema_version = "1.4"
        elif state.events:
            previous_digest = state.events[-1].event_sha256

        event = RegistryEvent(
            at=datetime.now(UTC),
            action=action,
            manifest_id=manifest_id,
            evaluation_report=evaluation_report,
            evaluation_report_sha256=evaluation_report_sha256,
            previous_event_sha256=previous_digest,
        )
        event = event.model_copy(update={"event_sha256": _registry_event_sha256(event)})
        state.events.append(event)

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
            "evaluation_report_sha256": manifest.evaluation_report_sha256,
        }
        missing = [name for name, value in attestation.items() if value is None]
        if missing:
            raise ValueError("manifest lacks evaluation attestation: " + ", ".join(sorted(missing)))

    @classmethod
    def _require_releasable_evidence(cls, manifest: ModelManifest) -> None:
        cls._require_evaluation_attestation(manifest)
        cls._require_training_attestation(manifest)
        if manifest.evaluation_evidence_kind != "measured":
            raise ValueError("only measured evaluation evidence can be released")
        if not manifest.artifacts:
            raise ValueError("at least one immutable model artifact is required for release")

    @classmethod
    def _require_rollback_evidence(cls, state: RegistryState, manifest: ModelManifest) -> None:
        legacy_identity = (
            manifest.evaluation_dataset_sha256,
            manifest.evaluation_engine,
            manifest.evaluation_model_id,
            manifest.evaluation_model_version,
            manifest.evaluation_evidence_kind,
            manifest.evaluation_report_sha256,
            manifest.training_report,
            manifest.training_report_sha256,
        )
        if state.schema_version == "1.0" and all(value is None for value in legacy_identity):
            return
        cls._require_releasable_evidence(manifest)

    @staticmethod
    def _require_training_attestation(manifest: ModelManifest) -> None:
        missing = [
            name
            for name, value in {
                "training_report": manifest.training_report,
                "training_report_sha256": manifest.training_report_sha256,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError("manifest lacks training attestation: " + ", ".join(missing))

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

    def _verify_evaluation_report(self, manifest: ModelManifest) -> None:
        if self.artifact_root is None or manifest.evaluation_report_sha256 is None:
            return
        path = (self.artifact_root / manifest.evaluation_report).resolve()
        try:
            path.relative_to(self.artifact_root)
        except ValueError as error:
            raise ValueError("evaluation report escapes the project root") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest.evaluation_report_sha256:
            raise ValueError("registered evaluation report hash changed")
        evaluation = EvaluationReport.model_validate_json(read_utf8_text_limited(path))
        self._validate_evaluation_against_manifest(manifest, evaluation)
        if manifest.release_passed != evaluation.release_passed:
            raise ValueError("evaluation release result does not match manifest")

    def _verify_training_report(self, manifest: ModelManifest) -> None:
        if (
            self.artifact_root is None
            or manifest.training_report is None
            or manifest.training_report_sha256 is None
        ):
            return
        path = (self.artifact_root / manifest.training_report).resolve()
        try:
            path.relative_to(self.artifact_root)
        except ValueError as error:
            raise ValueError("training report escapes the project root") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest.training_report_sha256:
            raise ValueError("registered training report hash changed")
        training = TrainingReportAttestation.model_validate_json(read_utf8_text_limited(path))
        checks = {
            "base_model": manifest.base_model == training.model_id,
            "base_revision": manifest.base_revision == training.resolved_model_revision,
            "dataset_manifest_sha256": (
                manifest.dataset_manifest_sha256 == training.dataset_manifest_sha256
            ),
            "training_config_sha256": (
                manifest.training_config_sha256 == training.training_config_sha256
            ),
        }
        mismatches = [name for name, matched in checks.items() if not matched]
        if mismatches:
            raise ValueError("training report does not match manifest: " + ", ".join(mismatches))

    @classmethod
    def _validate_evaluation_against_manifest(
        cls, manifest: ModelManifest, evaluation: EvaluationReport
    ) -> None:
        cls._require_evaluation_attestation(manifest)
        checks = {
            "evaluation_dataset_sha256": (
                manifest.evaluation_dataset_sha256 == evaluation.dataset_manifest_sha256
            ),
            "evaluation_engine": manifest.evaluation_engine == evaluation.engine,
            "evaluation_model_id": manifest.evaluation_model_id == evaluation.model_id,
            "evaluation_model_version": (
                manifest.evaluation_model_version == evaluation.model_version
            ),
            "evaluation_model_revision": (
                evaluation.model_revision is None
                or manifest.base_revision == evaluation.model_revision
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

    def validate_evaluation(self, manifest_id: str, evaluation: EvaluationReport) -> None:
        self._validate_id(manifest_id)
        state = self.load()
        if manifest_id not in state.manifests:
            raise KeyError(f"manifest not found: {manifest_id}")
        manifest = state.manifests[manifest_id]
        self._validate_evaluation_against_manifest(manifest, evaluation)

    def resolve_deployment(self, role: DeploymentRole) -> ModelManifest:
        if role not in {"released", "previous-good"}:
            raise ValueError("deployment role must be released or previous-good")
        state = self.load()
        manifest_id = state.released if role == "released" else state.previous_good
        if manifest_id is None:
            raise ValueError(f"registry has no {role} deployment target")
        manifest = state.manifests[manifest_id]
        self._require_releasable_evidence(manifest)
        if not manifest.release_passed:
            raise ValueError("deployment target did not pass its release gates")
        self._verify_evaluation_report(manifest)
        self._verify_training_report(manifest)
        self._verify_artifacts(manifest)
        return manifest

    def attest_evaluation(
        self,
        manifest_id: str,
        *,
        evaluation_report: str,
        evaluation_report_sha256: str,
        evaluation: EvaluationReport,
    ) -> RegistryState:
        with exclusive_file_lock(self.lock_path):
            return self._attest_evaluation_unlocked(
                manifest_id,
                evaluation_report=evaluation_report,
                evaluation_report_sha256=evaluation_report_sha256,
                evaluation=evaluation,
            )

    def _attest_evaluation_unlocked(
        self,
        manifest_id: str,
        *,
        evaluation_report: str,
        evaluation_report_sha256: str,
        evaluation: EvaluationReport,
    ) -> RegistryState:
        evaluation_report = _validate_portable_relative_path(evaluation_report)
        if not re.fullmatch(r"[a-f0-9]{64}", evaluation_report_sha256):
            raise ValueError("evaluation report SHA-256 must be lowercase hexadecimal")
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
        updated_manifest = manifest.model_copy(
            update={
                **required_contract,
                "evaluation_report": evaluation_report,
                "evaluation_report_sha256": evaluation_report_sha256,
                "release_passed": evaluation.release_passed,
            }
        )
        self._verify_evaluation_report(updated_manifest)
        state.manifests[manifest_id] = updated_manifest
        self._append_event(
            state,
            action="evaluation_attested",
            manifest_id=manifest_id,
            evaluation_report=evaluation_report,
            evaluation_report_sha256=evaluation_report_sha256,
        )
        self._write(state)
        return state

    def add(self, manifest: ModelManifest) -> RegistryState:
        with exclusive_file_lock(self.lock_path):
            return self._add_unlocked(manifest)

    def _add_unlocked(self, manifest: ModelManifest) -> RegistryState:
        self._validate_id(manifest.manifest_id)
        self._require_evaluation_attestation(manifest)
        self._require_training_attestation(manifest)
        state = self.load()
        if manifest.manifest_id in state.manifests:
            raise ValueError(f"manifest already exists: {manifest.manifest_id}")
        self._verify_evaluation_report(manifest)
        self._verify_training_report(manifest)
        self._verify_artifacts(manifest)
        state.manifests[manifest.manifest_id] = manifest
        self._append_event(
            state,
            action="candidate_added",
            manifest_id=manifest.manifest_id,
        )
        self._write(state)
        return state

    def promote(self, manifest_id: str) -> RegistryState:
        with exclusive_file_lock(self.lock_path):
            return self._promote_unlocked(manifest_id)

    def _promote_unlocked(self, manifest_id: str) -> RegistryState:
        self._validate_id(manifest_id)
        state = self.load()
        if manifest_id not in state.manifests:
            raise KeyError(f"manifest not found: {manifest_id}")
        candidate = state.manifests[manifest_id]
        self._require_releasable_evidence(candidate)
        if not candidate.release_passed or candidate.status == "blocked":
            raise ValueError("candidate did not pass its release gates")
        self._verify_evaluation_report(candidate)
        self._verify_training_report(candidate)
        self._verify_artifacts(candidate)
        if state.released and state.released != manifest_id:
            current = state.manifests[state.released]
            state.manifests[state.released] = current.model_copy(update={"status": "retired"})
            state.previous_good = state.released
        state.manifests[manifest_id] = candidate.model_copy(update={"status": "released"})
        state.released = manifest_id
        self._append_event(
            state,
            action="promoted",
            manifest_id=manifest_id,
        )
        self._write(state)
        return state

    def rollback(self) -> RegistryState:
        with exclusive_file_lock(self.lock_path):
            return self._rollback_unlocked()

    def _rollback_unlocked(self) -> RegistryState:
        state = self.load()
        if not state.previous_good:
            raise ValueError("no previous-good manifest is available")
        target = state.previous_good
        current = state.released
        target_manifest = state.manifests[target]
        self._require_rollback_evidence(state, target_manifest)
        if not target_manifest.release_passed:
            raise ValueError("previous-good manifest no longer satisfies release metadata")
        self._verify_evaluation_report(target_manifest)
        self._verify_training_report(target_manifest)
        self._verify_artifacts(target_manifest)
        if current:
            state.manifests[current] = state.manifests[current].model_copy(
                update={"status": "retired"}
            )
        state.manifests[target] = target_manifest.model_copy(update={"status": "released"})
        state.released = target
        state.previous_good = current
        self._append_event(
            state,
            action="rolled_back",
            manifest_id=target,
        )
        self._write(state)
        return state

    def block(
        self,
        manifest_id: str,
        *,
        evaluation_report: str | None = None,
        evaluation_report_sha256: str | None = None,
        release_passed: bool | None = None,
    ) -> RegistryState:
        with exclusive_file_lock(self.lock_path):
            return self._block_unlocked(
                manifest_id,
                evaluation_report=evaluation_report,
                evaluation_report_sha256=evaluation_report_sha256,
                release_passed=release_passed,
            )

    def _block_unlocked(
        self,
        manifest_id: str,
        *,
        evaluation_report: str | None = None,
        evaluation_report_sha256: str | None = None,
        release_passed: bool | None = None,
    ) -> RegistryState:
        self._validate_id(manifest_id)
        if evaluation_report is not None:
            evaluation_report = _validate_portable_relative_path(evaluation_report)
            if evaluation_report_sha256 is None or not re.fullmatch(
                r"[a-f0-9]{64}", evaluation_report_sha256
            ):
                raise ValueError("blocking with an evaluation report requires its SHA-256")
        elif evaluation_report_sha256 is not None:
            raise ValueError("evaluation report SHA-256 requires an evaluation report")
        state = self.load()
        if manifest_id not in state.manifests:
            raise KeyError(f"manifest not found: {manifest_id}")
        manifest = state.manifests[manifest_id]
        updates: dict[str, object] = {"status": "blocked"}
        if evaluation_report is not None:
            updates["evaluation_report"] = evaluation_report
            updates["evaluation_report_sha256"] = evaluation_report_sha256
        if release_passed is not None:
            updates["release_passed"] = release_passed
        updated_manifest = manifest.model_copy(update=updates)
        self._verify_evaluation_report(updated_manifest)
        state.manifests[manifest_id] = updated_manifest
        if state.released == manifest_id:
            state.released = None
        if state.previous_good == manifest_id:
            state.previous_good = None
        self._append_event(
            state,
            action="blocked",
            manifest_id=manifest_id,
            evaluation_report=evaluation_report,
            evaluation_report_sha256=evaluation_report_sha256,
        )
        self._write(state)
        return state
