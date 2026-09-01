import hashlib
from concurrent.futures import ThreadPoolExecutor

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
        release_gate_sha256=HASH,
        engine="fixture",
        quantization="none",
        max_new_tokens=192,
        structured_output_contract="json-schema-v2",
        safety_policy_version="risk-precedence-v2",
        evaluation_contract_version="release-metrics-v2",
        evaluation_dataset_sha256=HASH,
        evaluation_engine="fixture",
        evaluation_model_id="fixture",
        evaluation_model_version="1",
        evaluation_evidence_kind="measured",
        evaluation_report="outputs/evaluation.json",
        evaluation_report_sha256=HASH,
        release_passed=passed,
        artifacts=[
            ArtifactRef(
                kind="gguf",
                relative_path="artifacts/model.gguf",
                sha256=HASH,
                bytes=1,
            )
        ],
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


def test_registry_preserves_parallel_candidate_additions(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    candidates = [manifest(f"parallel-{index}") for index in range(8)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(registry.add, candidates))

    assert set(registry.load().manifests) == {
        f"parallel-{index}" for index in range(len(candidates))
    }


def test_registry_rejects_failed_or_unsafe_candidate(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    registry.add(manifest("failed", passed=False))
    with pytest.raises(ValueError):
        registry.promote("failed")
    with pytest.raises(ValueError):
        registry.add(manifest("../escape"))


def test_registry_rejects_release_without_model_artifact(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    registry.add(manifest("metadata-only").model_copy(update={"artifacts": []}))

    with pytest.raises(ValueError, match="at least one immutable model artifact"):
        registry.promote("metadata-only")


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../evaluation.json",
        "outputs/../evaluation.json",
        "/tmp/evaluation.json",
        "C:/outside/evaluation.json",
        "outputs\\evaluation.json",
    ],
)
def test_registry_rejects_unsafe_report_paths(unsafe_path: str):
    manifest_data = manifest("unsafe-path").model_dump()
    manifest_data["evaluation_report"] = unsafe_path
    with pytest.raises(ValueError, match="canonical portable relative path"):
        ModelManifest.model_validate(manifest_data)


@pytest.mark.parametrize(
    "unsafe_path",
    ["../model.gguf", "/tmp/model.gguf", "C:/outside/model.gguf", "models\\model.gguf"],
)
def test_registry_rejects_unsafe_artifact_paths(unsafe_path: str):
    with pytest.raises(ValueError, match="canonical portable relative path"):
        ArtifactRef(kind="gguf", relative_path=unsafe_path, sha256=HASH, bytes=0)


def test_registry_block_rejects_unsafe_report_path_without_mutation(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    registry.add(manifest("v1"))

    with pytest.raises(ValueError, match="canonical portable relative path"):
        registry.block("v1", evaluation_report="../outside.json", release_passed=False)

    state = registry.load()
    assert state.manifests["v1"].status == "candidate"
    assert state.manifests["v1"].release_passed is True
    assert state.events[-1]["action"] == "candidate_added"


def test_registry_can_block_a_released_candidate(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    registry.add(manifest("v1"))
    registry.promote("v1")
    state = registry.block(
        "v1",
        evaluation_report="outputs/full.json",
        evaluation_report_sha256=HASH,
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
        evidence_kind="measured",
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


def test_registry_rejects_evaluation_from_a_different_dataset(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    registry.add(manifest("v1"))
    report = EvaluationReport(
        run_id="wrong-dataset",
        engine="fixture",
        model_id="fixture",
        model_version="1",
        quantization="none",
        prompt_version="optimized-v1",
        evidence_kind="measured",
        dataset_manifest_sha256=hashlib.sha256(b"different-dataset").hexdigest(),
        case_count=0,
        metrics={},
        gates={},
        release_passed=False,
        latency_ms={},
        cases=[],
    )
    with pytest.raises(ValueError, match="evaluation_dataset_sha256"):
        registry.validate_evaluation("v1", report)


def test_registry_rejects_new_manifest_without_evaluation_attestation(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    incomplete = manifest("legacy").model_copy(
        update={
            "evaluation_dataset_sha256": None,
            "evaluation_engine": None,
            "evaluation_model_id": None,
            "evaluation_model_version": None,
            "evaluation_evidence_kind": None,
        }
    )
    with pytest.raises(ValueError, match="manifest lacks evaluation attestation"):
        registry.add(incomplete)


def test_registry_never_promotes_fixture_evidence(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    fixture_candidate = manifest("fixture-only").model_copy(
        update={"evaluation_evidence_kind": "fixture"}
    )
    registry.add(fixture_candidate)

    with pytest.raises(ValueError, match="only measured evaluation evidence"):
        registry.promote("fixture-only")

    state = registry.load()
    assert state.released is None
    assert state.manifests["fixture-only"].status == "candidate"


def test_registry_rechecks_attestation_before_promotion(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    state = registry.add(manifest("v1"))
    state.manifests["v1"] = state.manifests["v1"].model_copy(
        update={"evaluation_model_version": None}
    )
    registry.path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest lacks evaluation attestation"):
        registry.promote("v1")
    assert registry.load().released is None


def test_registry_allows_only_fully_legacy_previous_good_on_schema_1_rollback(tmp_path):
    registry = FileRegistry(tmp_path / "registry.json")
    registry.add(manifest("v1"))
    registry.promote("v1")
    registry.add(manifest("v2"))
    state = registry.promote("v2")
    legacy_updates = {
        "evaluation_dataset_sha256": None,
        "evaluation_engine": None,
        "evaluation_model_id": None,
        "evaluation_model_version": None,
        "evaluation_evidence_kind": None,
        "evaluation_report_sha256": None,
    }
    state.schema_version = "1.0"
    state.manifests["v1"] = state.manifests["v1"].model_copy(update=legacy_updates)
    registry.path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    rolled_back = registry.rollback()
    assert rolled_back.released == "v1"

    partial_state = rolled_back.model_copy(update={"schema_version": "1.0"}, deep=True)
    partial_state.manifests["v2"] = partial_state.manifests["v2"].model_copy(
        update={**legacy_updates, "evaluation_engine": "fixture"}
    )
    registry.path.write_text(partial_state.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest lacks evaluation attestation"):
        registry.rollback()


def test_registry_rechecks_artifact_before_promotion(tmp_path):
    evaluation_report = tmp_path / "outputs" / "evaluation.json"
    evaluation_report.parent.mkdir()
    evaluation_report.write_bytes(b"fixture")
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


def test_registry_rechecks_evaluation_report_before_promotion(tmp_path):
    report_path = tmp_path / "evaluation.json"
    report_path.write_bytes(b"original")
    candidate = manifest("v1").model_copy(
        update={
            "evaluation_report": "evaluation.json",
            "evaluation_report_sha256": hashlib.sha256(b"original").hexdigest(),
        }
    )
    registry = FileRegistry(tmp_path / "registry.json", artifact_root=tmp_path)
    registry.add(candidate)
    report_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="evaluation report hash changed"):
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
        evidence_kind="measured",
        dataset_manifest_sha256=HASH,
        release_gate_sha256=HASH,
        case_count=0,
        metrics={},
        gates={},
        release_passed=False,
        latency_ms={},
        cases=[],
    )
    with pytest.raises(ValueError, match="canonical portable relative path"):
        registry.attest_evaluation(
            "v1",
            evaluation_report="../outside.json",
            evaluation_report_sha256=HASH,
            evaluation=report,
        )
    assert registry.load().manifests["v1"].evaluation_report == "outputs/evaluation.json"

    state = registry.attest_evaluation(
        "v1",
        evaluation_report="outputs/attested.json",
        evaluation_report_sha256=HASH,
        evaluation=report,
    )
    attested = state.manifests["v1"]
    assert attested.max_new_tokens == 192
    assert attested.safety_policy_version == "risk-precedence-v2"
    assert attested.evaluation_contract_version == "release-metrics-v2"
    assert attested.evaluation_report_sha256 == HASH
    assert state.events[-1]["action"] == "evaluation_attested"

    mismatch = report.model_copy(update={"safety_policy_version": "obsolete-policy"})
    with pytest.raises(ValueError, match="safety_policy_version"):
        registry.validate_evaluation("v1", mismatch)
