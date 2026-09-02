import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import modelforge_tw.training as training_module
from modelforge_tw import cli
from modelforge_tw.cli import build_parser
from modelforge_tw.dataset import write_dataset
from modelforge_tw.engines.transformers_local import (
    TransformersInferenceEngine,
    model_load_probe,
)
from modelforge_tw.model_identity import (
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_MAX_NEW_TOKENS,
    validate_resolved_model_revision,
    validate_tokenizer_snapshot_revision,
)
from modelforge_tw.prompts import DEFAULT_PROMPT_VERSION
from modelforge_tw.training import (
    TRAINING_DEPENDENCIES,
    TrainingConfig,
    TrainingReportAttestation,
    _is_link_or_reparse_point,
    _report_relative_path,
    installed_dependency_versions,
    merge_adapter,
    prepare_training_run,
    run_training,
    verify_adapter_training_lineage,
    verify_training_dataset_manifest,
    verify_training_run_contract,
)


def test_report_paths_are_portable_and_relative_to_the_report(tmp_path, monkeypatch):
    report_directory = tmp_path / "artifacts" / "merged"
    source = tmp_path / "artifacts" / "adapter" / "training-report.json"

    assert _report_relative_path(source, report_directory) == "../adapter/training-report.json"

    def cross_drive(*_args, **_kwargs):
        raise ValueError

    monkeypatch.setattr("modelforge_tw.training.os.path.relpath", cross_drive)
    assert _report_relative_path(source, report_directory) == "training-report.json"


def test_windows_reparse_attribute_is_treated_as_link(monkeypatch, tmp_path):
    monkeypatch.setattr(
        type(tmp_path),
        "lstat",
        lambda _path: SimpleNamespace(st_file_attributes=1024),
    )
    monkeypatch.setattr(type(tmp_path), "is_symlink", lambda _path: False)

    assert _is_link_or_reparse_point(tmp_path) is True


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


def test_training_dataset_manifest_is_verified_before_ml_runtime(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset manifest"):
        run_training(TrainingConfig(data_dir=tmp_path, output_dir=tmp_path / "output"))
    assert not (tmp_path / "output").exists()

    write_dataset(
        tmp_path,
        train_per_type=1,
        validation_per_type=1,
        test_per_type=1,
    )
    manifest = verify_training_dataset_manifest(tmp_path)
    assert manifest.audit["passed"] is True
    assert {"train.jsonl", "validation.jsonl"} <= manifest.files.keys()

    with (tmp_path / "train.jsonl").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="does not match manifest"):
        verify_training_dataset_manifest(tmp_path)


def test_training_dataset_rejects_linked_path_component(tmp_path, monkeypatch):
    monkeypatch.setattr(
        training_module,
        "_first_linked_path_component",
        lambda path: path,
    )

    with pytest.raises(ValueError, match="training data path"):
        verify_training_dataset_manifest(tmp_path)


def test_checkpoint_resume_requires_matching_run_contract(tmp_path):
    data_dir = tmp_path / "data"
    write_dataset(
        data_dir,
        train_per_type=1,
        validation_per_type=1,
        test_per_type=1,
    )
    manifest = verify_training_dataset_manifest(data_dir)
    output_dir = tmp_path / "training-output"
    config = TrainingConfig(data_dir=data_dir, output_dir=output_dir)

    prepared, checkpoint, contract_hash = prepare_training_run(config, manifest)
    assert prepared == output_dir.resolve()
    assert checkpoint is None
    assert len(contract_hash) == 64
    (output_dir / "checkpoint-10").mkdir()

    _, checkpoint, resumed_contract_hash = prepare_training_run(config, manifest)
    assert checkpoint == str(output_dir.resolve() / "checkpoint-10")
    assert resumed_contract_hash == contract_hash

    with pytest.raises(ValueError, match="does not match"):
        prepare_training_run(config.model_copy(update={"learning_rate": 0.0001}), manifest)
    with pytest.raises(ValueError, match="resume is disabled"):
        prepare_training_run(config.model_copy(update={"resume_from_checkpoint": False}), manifest)

    config_payload = config.model_dump(mode="json")
    report = TrainingReportAttestation(
        schema_version="1.2",
        evidence_kind="measured",
        model_id=config.model_id,
        model_revision=config.model_revision,
        resolved_model_revision=config.model_revision,
        resolved_tokenizer_revision=config.model_revision,
        config=config_payload,
        training_config_sha256=_stable_hash(config_payload),
        dataset_manifest_sha256=manifest.dataset_manifest_sha256,
        training_run_contract_sha256=contract_hash,
    )
    report_path = output_dir / "training-report.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    assert verify_training_run_contract(report, report_path) is not None

    (output_dir / "run-contract.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        verify_training_run_contract(report, report_path)


def test_legacy_checkpoint_is_not_silently_attested(tmp_path):
    data_dir = tmp_path / "data"
    write_dataset(data_dir, train_per_type=1, validation_per_type=1, test_per_type=1)
    manifest = verify_training_dataset_manifest(data_dir)
    output_dir = tmp_path / "legacy-output"
    (output_dir / "checkpoint-5").mkdir(parents=True)

    with pytest.raises(ValueError, match="legacy checkpoints lack"):
        prepare_training_run(
            TrainingConfig(data_dir=data_dir, output_dir=output_dir),
            manifest,
        )


def test_training_output_rejects_linked_directory_components(tmp_path):
    data_dir = tmp_path / "data"
    write_dataset(data_dir, train_per_type=1, validation_per_type=1, test_per_type=1)
    manifest = verify_training_dataset_manifest(data_dir)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    try:
        linked_output.symlink_to(real_output, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="links or reparse points"):
        prepare_training_run(
            TrainingConfig(data_dir=data_dir, output_dir=linked_output),
            manifest,
        )


def test_checkpoint_resume_rejects_linked_checkpoint(tmp_path):
    data_dir = tmp_path / "data"
    write_dataset(data_dir, train_per_type=1, validation_per_type=1, test_per_type=1)
    manifest = verify_training_dataset_manifest(data_dir)
    output_dir = tmp_path / "training-output"
    config = TrainingConfig(data_dir=data_dir, output_dir=output_dir)
    prepare_training_run(config, manifest)
    real_checkpoint = tmp_path / "real-checkpoint"
    real_checkpoint.mkdir()
    linked_checkpoint = output_dir / "checkpoint-10"
    try:
        linked_checkpoint.symlink_to(real_checkpoint, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="malformed checkpoints"):
        prepare_training_run(config, manifest)


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


def test_tokenizer_assets_must_resolve_from_requested_snapshot():
    snapshot = f"C:/cache/snapshots/{DEFAULT_BASE_MODEL_REVISION}"
    assert (
        validate_tokenizer_snapshot_revision(
            DEFAULT_BASE_MODEL_REVISION,
            {
                "vocab_file": f"{snapshot}/vocab.json",
                "merges_file": f"{snapshot}/merges.txt",
            },
        )
        == DEFAULT_BASE_MODEL_REVISION
    )
    with pytest.raises(ValueError, match="did not report"):
        validate_tokenizer_snapshot_revision(DEFAULT_BASE_MODEL_REVISION, {})
    with pytest.raises(ValueError, match="multiple snapshot"):
        validate_tokenizer_snapshot_revision(
            DEFAULT_BASE_MODEL_REVISION,
            {
                "vocab_file": f"{snapshot}/vocab.json",
                "merges_file": f"C:/cache/snapshots/{'b' * 40}/merges.txt",
            },
        )
    with pytest.raises(ValueError, match="does not match"):
        validate_tokenizer_snapshot_revision(
            DEFAULT_BASE_MODEL_REVISION,
            {"vocab_file": f"C:/cache/snapshots/{'b' * 40}/vocab.json"},
        )


def test_training_attestation_v12_requires_tokenizer_revision():
    config = TrainingConfig().model_dump(mode="json")
    with pytest.raises(ValidationError, match="resolved_tokenizer_revision"):
        TrainingReportAttestation(
            schema_version="1.2",
            evidence_kind="measured",
            model_id=config["model_id"],
            model_revision=config["model_revision"],
            resolved_model_revision=config["model_revision"],
            config=config,
            training_config_sha256=_stable_hash(config),
            dataset_manifest_sha256="a" * 64,
        )


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


@pytest.mark.parametrize(
    ("linked_target", "expected_message"),
    [("adapter", "adapter path"), ("report", "training report path")],
)
def test_adapter_lineage_rejects_linked_source_component(
    tmp_path, monkeypatch, linked_target, expected_message
):
    adapter, report = _write_attested_adapter(tmp_path)
    selected = adapter if linked_target == "adapter" else report
    monkeypatch.setattr(
        training_module,
        "_first_linked_path_component",
        lambda path: path if path == selected else None,
    )

    with pytest.raises(ValueError, match=expected_message):
        verify_adapter_training_lineage(
            model_id="Qwen/Qwen3-1.7B",
            model_revision=DEFAULT_BASE_MODEL_REVISION,
            adapter_path=adapter,
            training_report_path=report,
        )


def test_merge_rejects_nonempty_output_before_ml_runtime(tmp_path):
    adapter, report = _write_attested_adapter(tmp_path)
    output = tmp_path / "merged"
    output.mkdir()
    (output / "stale-model.safetensors").write_bytes(b"stale")

    with pytest.raises(ValueError, match="must be empty"):
        merge_adapter(
            model_id="Qwen/Qwen3-1.7B",
            model_revision=DEFAULT_BASE_MODEL_REVISION,
            adapter_path=adapter,
            training_report_path=report,
            output_dir=output,
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


def test_transformers_cli_requires_adapter_training_report_pair():
    adapter_only = build_parser().parse_args(
        [
            "evaluate",
            "--engine",
            "transformers",
            "--model-version",
            "test-v1",
            "--adapter",
            "adapter",
        ]
    )
    with pytest.raises(ValueError, match="--adapter-training-report"):
        cli._engine_from_args(adapter_only)

    report_only = build_parser().parse_args(
        [
            "evaluate",
            "--engine",
            "transformers",
            "--model-version",
            "test-v1",
            "--adapter-training-report",
            "training-report.json",
        ]
    )
    with pytest.raises(ValueError, match="--adapter and"):
        cli._engine_from_args(report_only)


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
