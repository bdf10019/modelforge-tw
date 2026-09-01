import argparse
import hashlib
import json

import pytest

from modelforge_tw import cli
from modelforge_tw.registry import FileRegistry
from modelforge_tw.schemas import EvaluationReport
from modelforge_tw.training import TrainingConfig

HASH = hashlib.sha256(b"registry-cli").hexdigest()


def write_inputs(tmp_path):
    evaluation = EvaluationReport(
        run_id="cli-registry",
        engine="fixture",
        model_id="fixture-model",
        model_version="1.0",
        quantization="none",
        prompt_version="optimized-v2",
        max_new_tokens=192,
        context_window_tokens=2048,
        structured_output_contract="json-schema-v2",
        safety_policy_version="risk-precedence-v2",
        evaluation_contract_version="release-metrics-v2",
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
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
    training_config = TrainingConfig(
        model_id="fixture-base",
        model_revision=HASH,
        data_dir="data/generated",
        prompt_version="optimized-v2",
    ).model_dump(mode="json")
    training_config_sha256 = hashlib.sha256(
        json.dumps(training_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    training_path = tmp_path / "training.json"
    training_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "evidence_kind": "measured",
                "model_id": "fixture-base",
                "model_revision": HASH,
                "resolved_model_revision": HASH,
                "config": training_config,
                "dataset_manifest_sha256": HASH,
                "training_config_sha256": training_config_sha256,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "data" / "generated" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"dataset_manifest_sha256": HASH}), encoding="utf-8")
    return evaluation_path, training_path


def registry_args(tmp_path, evaluation_path, training_path, *, quantization="none"):
    return argparse.Namespace(
        registry_action="add",
        manifest_id="candidate-v1",
        registry=tmp_path / "registry.json",
        evaluation=evaluation_path,
        training_report=training_path,
        artifact=[],
        license="MIT",
        registry_engine="fixture",
        registry_quantization=quantization,
    )


def test_registry_cli_add_locks_complete_evaluation_identity(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    evaluation_path, training_path = write_inputs(tmp_path)
    args = registry_args(tmp_path, evaluation_path, training_path)

    assert cli.command_registry(args) == 0
    capsys.readouterr()
    saved = FileRegistry(args.registry).load().manifests["candidate-v1"]
    assert saved.evaluation_dataset_sha256 == HASH
    assert saved.evaluation_engine == "fixture"
    assert saved.evaluation_model_id == "fixture-model"
    assert saved.evaluation_model_version == "1.0"
    assert saved.evaluation_evidence_kind == "fixture"
    assert (
        saved.evaluation_report_sha256 == hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    )
    assert saved.release_gate_sha256 == HASH
    assert saved.max_new_tokens == 192
    assert saved.safety_policy_version == "risk-precedence-v2"


def test_registry_cli_rejects_quantization_mismatch_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    evaluation_path, training_path = write_inputs(tmp_path)
    args = registry_args(tmp_path, evaluation_path, training_path, quantization="q4_k_m")

    with pytest.raises(ValueError, match="registry quantization does not match evaluation"):
        cli.command_registry(args)
    assert not args.registry.exists()


def test_registry_cli_rejects_tampered_training_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    evaluation_path, training_path = write_inputs(tmp_path)
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["config"]["seed"] = 99
    training_path.write_text(json.dumps(training), encoding="utf-8")

    with pytest.raises(ValueError, match="training_config_sha256"):
        cli.command_registry(registry_args(tmp_path, evaluation_path, training_path))


def test_registry_cli_rejects_dataset_manifest_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    evaluation_path, training_path = write_inputs(tmp_path)
    manifest_path = tmp_path / "data" / "generated" / "manifest.json"
    manifest_path.write_text(
        json.dumps({"dataset_manifest_sha256": hashlib.sha256(b"other").hexdigest()}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="local dataset manifest"):
        cli.command_registry(registry_args(tmp_path, evaluation_path, training_path))


def test_registry_cli_rejects_training_prompt_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    evaluation_path, training_path = write_inputs(tmp_path)
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["config"]["prompt_version"] = "optimized-v1"
    training["training_config_sha256"] = hashlib.sha256(
        json.dumps(training["config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    training_path.write_text(json.dumps(training), encoding="utf-8")

    with pytest.raises(ValueError, match="training prompt version"):
        cli.command_registry(registry_args(tmp_path, evaluation_path, training_path))
