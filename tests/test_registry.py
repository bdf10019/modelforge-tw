import hashlib

import pytest

from modelforge_tw.registry import ArtifactRef, FileRegistry, ModelManifest
from modelforge_tw.schemas import EvaluationReport

HASH = hashlib.sha256(b"fixture").hexdigest()


def manifest(manifest_id: str, passed: bool = True) -> ModelManifest:
    return ModelManifest(
        manifest_id=manifest_id,
        base_model="fixture",
        base_revision="fixture",
        license="MIT",
        prompt_version="optimized-v1",
        dataset_manifest_sha256=HASH,
        training_config_sha256=HASH,
        engine="fixture",
        quantization="none",
        evaluation_report="outputs/evaluation.json",
        release_passed=passed,
    )


def test_registry_promote_and_rollback(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    registry.add(manifest("v1"))
    state = registry.promote("v1")
    assert state.released == "v1"
    registry.add(manifest("v2"))
    state = registry.promote("v2")
    assert state.released == "v2"
    assert state.previous_good == "v1"
    state = registry.rollback()
    assert state.released == "v1"
    assert state.previous_good == "v2"


def test_registry_rejects_failed_or_unsafe_candidate(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    registry.add(manifest("failed", passed=False))
    with pytest.raises(ValueError):
        registry.promote("failed")
    with pytest.raises(ValueError):
        registry.add(manifest("../escape"))


def test_registry_can_block_a_released_candidate(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    registry.add(manifest("v1"))
    registry.promote("v1")
    state = registry.block(
        "v1",
        evaluation_report="outputs/full.json",
        release_passed=False,
    )
    assert state.released is None
    assert state.manifests["v1"].status == "blocked"
    assert state.manifests["v1"].release_passed is False
    assert state.manifests["v1"].evaluation_report == "outputs/full.json"
    with pytest.raises(ValueError):
        registry.promote("v1")


def test_registry_rejects_mismatched_evaluation_metadata(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    registry.add(manifest("v1"))
    report = EvaluationReport(
        run_id="test-run",
        engine="fixture",
        model_id="fixture",
        model_version="1",
        quantization="q4_k_m",
        prompt_version="optimized-v1",
        evidence_kind="fixture",
        dataset_manifest_sha256=HASH,
        release_gate_sha256=HASH,
        case_count=0,
        metrics={},
        gates={},
        release_passed=False,
        latency_ms={},
        cases=[],
    )
    with pytest.raises(ValueError, match="quantization"):
        registry.validate_evaluation("v1", report)


def test_registry_rechecks_artifact_before_promotion(tmp_path):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    candidate = manifest("v1").model_copy(
        update={
            "artifacts": [
                ArtifactRef(
                    kind="gguf",
                    relative_path="model.gguf",
                    sha256=digest,
                    bytes=8,
                )
            ]
        }
    )
    registry = FileRegistry(tmp_path / "registry.json", artifact_root=tmp_path)
    registry.add(candidate)
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash changed"):
        registry.promote("v1")


def test_registry_attests_and_locks_evaluation_contract(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    registry.add(manifest("v1").model_copy(update={"release_gate_sha256": HASH}))
    report = EvaluationReport(
        run_id="attested-run",
        engine="fixture",
        model_id="fixture",
        model_version="1",
        quantization="none",
        prompt_version="optimized-v1",
        max_new_tokens=192,
        structured_output_contract="json-schema-v2",
        safety_policy_version="risk-precedence-v2",
        evaluation_contract_version="release-metrics-v2",
        evidence_kind="fixture",
        dataset_manifest_sha256=HASH,
        release_gate_sha256=HASH,
        case_count=0,
        metrics={},
        gates={},
        release_passed=True,
        latency_ms={},
        cases=[],
    )
    state = registry.attest_evaluation(
        "v1", evaluation_report="outputs/attested.json", evaluation=report
    )
    attested = state.manifests["v1"]
    assert attested.max_new_tokens == 192
    assert attested.safety_policy_version == "risk-precedence-v2"
    assert attested.evaluation_contract_version == "release-metrics-v2"
    assert state.events[-1]["action"] == "evaluation_attested"

    mismatch = report.model_copy(update={"safety_policy_version": "obsolete-policy"})
    with pytest.raises(ValueError, match="safety_policy_version"):
        registry.validate_evaluation("v1", mismatch)
