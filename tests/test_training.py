import hashlib
import json

import pytest
from pydantic import ValidationError

from modelforge_tw import cli
from modelforge_tw.cli import build_parser
from modelforge_tw.engines.transformers_local import (
    TransformersInferenceEngine,
    model_load_probe,
)
from modelforge_tw.model_identity import (
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_MAX_NEW_TOKENS,
    validate_resolved_model_revision,
)
from modelforge_tw.prompts import DEFAULT_PROMPT_VERSION
from modelforge_tw.training import (
    TRAINING_DEPENDENCIES,
    TrainingConfig,
    TrainingReportAttestation,
    installed_dependency_versions,
    merge_adapter,
    verify_adapter_training_lineage,
)


def _stable_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _write_attested_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    model_file = adapter / "adapter_model.safetensors"
    model_file.write_bytes(b"adapter-fixture")
    file_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
    config = TrainingConfig().model_dump(mode="json")
    report = tmp_path / "training-report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "evidence_kind": "measured",
                "model_id": config["model_id"],
                "model_revision": config["model_revision"],
                "resolved_model_revision": config["model_revision"],
                "config": config,
                "training_config_sha256": _stable_hash(config),
                "dataset_manifest_sha256": "a" * 64,
                "final_adapter_files": {
                    model_file.name: {"bytes": model_file.stat().st_size, "sha256": file_hash}
                },
            }
        ),
        encoding="utf-8",
    )
    return adapter, report


def test_training_dependency_snapshot_is_complete_and_versioned():
    assert set(installed_dependency_versions()) == set(TRAINING_DEPENDENCIES)
    versions = installed_dependency_versions(("pip", "modelforge-definitely-missing"))
    assert versions["pip"] and versions["pip"][0].isdigit()
    assert versions["modelforge-definitely-missing"] == "not-installed"


def test_model_workflows_default_to_one_immutable_base_revision(tmp_path):
    parser = build_parser()
    evaluate = parser.parse_args(["evaluate"])
    merge = parser.parse_args(
        [
            "merge",
            "--adapter",
            str(tmp_path / "adapter"),
            "--training-report",
            str(tmp_path / "training-report.json"),
            "--output",
            str(tmp_path / "merged"),
        ]
    )
    probe = parser.parse_args(["probe-model"])

    assert TrainingConfig().model_revision == DEFAULT_BASE_MODEL_REVISION
    assert evaluate.model_revision == DEFAULT_BASE_MODEL_REVISION
    assert evaluate.prompt_version == DEFAULT_PROMPT_VERSION
    assert evaluate.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert merge.model_revision == DEFAULT_BASE_MODEL_REVISION
    assert probe.model_revision == DEFAULT_BASE_MODEL_REVISION
    assert TrainingConfig().prompt_version == DEFAULT_PROMPT_VERSION


def test_training_config_rejects_floating_model_revision():
    with pytest.raises(ValidationError, match="model_revision"):
        TrainingConfig(model_revision="main")
    with pytest.raises(ValidationError, match="prompt_version"):
        TrainingConfig(prompt_version="minimal-v1")


def test_resolved_model_revision_must_exactly_match_request():
    assert (
        validate_resolved_model_revision(DEFAULT_BASE_MODEL_REVISION, DEFAULT_BASE_MODEL_REVISION)
        == DEFAULT_BASE_MODEL_REVISION
    )
    with pytest.raises(ValueError, match="did not report"):
        validate_resolved_model_revision(DEFAULT_BASE_MODEL_REVISION, None)
    with pytest.raises(ValueError, match="does not match"):
        validate_resolved_model_revision(DEFAULT_BASE_MODEL_REVISION, "b" * 40)


def test_training_attestation_rejects_resolved_revision_drift():
    config = TrainingConfig().model_dump(mode="json")
    with pytest.raises(ValidationError, match="resolved_model_revision"):
        TrainingReportAttestation(
            schema_version="1.1",
            evidence_kind="measured",
            model_id=config["model_id"],
            model_revision=config["model_revision"],
            resolved_model_revision="b" * 40,
            config=config,
            training_config_sha256=_stable_hash(config),
            dataset_manifest_sha256="a" * 64,
        )


def test_adapter_lineage_verification_rejects_file_or_identity_drift(tmp_path):
    adapter, report = _write_attested_adapter(tmp_path)
    attestation, report_hash = verify_adapter_training_lineage(
        model_id="Qwen/Qwen3-1.7B",
        model_revision=DEFAULT_BASE_MODEL_REVISION,
        adapter_path=adapter,
        training_report_path=report,
    )
    assert attestation.model_revision == DEFAULT_BASE_MODEL_REVISION
    assert report_hash == hashlib.sha256(report.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="model_id"):
        verify_adapter_training_lineage(
            model_id="different/model",
            model_revision=DEFAULT_BASE_MODEL_REVISION,
            adapter_path=adapter,
            training_report_path=report,
        )

    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match training report"):
        verify_adapter_training_lineage(
            model_id="Qwen/Qwen3-1.7B",
            model_revision=DEFAULT_BASE_MODEL_REVISION,
            adapter_path=adapter,
            training_report_path=report,
        )


def test_cli_rejects_floating_model_revision():
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["evaluate", "--model-revision", "main"])

    assert error.value.code == 2


def test_model_operations_reject_floating_revision_before_loading_ml_runtime(tmp_path):
    with pytest.raises(ValueError, match="commit hash"):
        TransformersInferenceEngine(
            model_id="Qwen/Qwen3-1.7B",
            model_revision="main",
            adapter_path=None,
            prompt_version="optimized-v2",
            model_version="test-v1",
        )
    with pytest.raises(ValueError, match="commit hash"):
        model_load_probe("Qwen/Qwen3-1.7B", model_revision="main", load_in_4bit=True)
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="commit hash"):
        merge_adapter(
            model_id="Qwen/Qwen3-1.7B",
            model_revision="main",
            adapter_path=tmp_path / "adapter",
            training_report_path=tmp_path / "training-report.json",
            output_dir=output,
        )
    assert not output.exists()


def test_measured_cli_requires_explicit_model_version():
    args = build_parser().parse_args(["evaluate", "--engine", "transformers"])
    with pytest.raises(ValueError, match="--model-version"):
        cli._engine_from_args(args)

    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["evaluate", "--model-version", "unversioned"])
    assert error.value.code == 2


def test_llama_cli_requires_explicit_quantization():
    args = build_parser().parse_args(
        [
            "evaluate",
            "--engine",
            "llama-http",
            "--model-version",
            "qwen3-q8-v1",
        ]
    )
    with pytest.raises(ValueError, match="--quantization"):
        cli._engine_from_args(args)

    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["evaluate", "--quantization", "unknown"])
    assert error.value.code == 2
