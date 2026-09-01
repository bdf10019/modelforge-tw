import pytest
from pydantic import ValidationError

from modelforge_tw.cli import build_parser
from modelforge_tw.engines.transformers_local import (
    TransformersInferenceEngine,
    model_load_probe,
)
from modelforge_tw.model_identity import DEFAULT_BASE_MODEL_REVISION
from modelforge_tw.training import (
    TRAINING_DEPENDENCIES,
    TrainingConfig,
    installed_dependency_versions,
    merge_adapter,
)


def test_training_dependency_snapshot_is_complete_and_versioned():
    assert set(installed_dependency_versions()) == set(TRAINING_DEPENDENCIES)
    versions = installed_dependency_versions(("pip", "modelforge-definitely-missing"))
    assert versions["pip"] and versions["pip"][0].isdigit()
    assert versions["modelforge-definitely-missing"] == "not-installed"


def test_model_workflows_default_to_one_immutable_base_revision(tmp_path):
    parser = build_parser()
    evaluate = parser.parse_args(["evaluate"])
    merge = parser.parse_args(
        ["merge", "--adapter", str(tmp_path / "adapter"), "--output", str(tmp_path / "merged")]
    )
    probe = parser.parse_args(["probe-model"])

    assert TrainingConfig().model_revision == DEFAULT_BASE_MODEL_REVISION
    assert evaluate.model_revision == DEFAULT_BASE_MODEL_REVISION
    assert merge.model_revision == DEFAULT_BASE_MODEL_REVISION
    assert probe.model_revision == DEFAULT_BASE_MODEL_REVISION


def test_training_config_rejects_floating_model_revision():
    with pytest.raises(ValidationError, match="model_revision"):
        TrainingConfig(model_revision="main")


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
        )
    with pytest.raises(ValueError, match="commit hash"):
        model_load_probe("Qwen/Qwen3-1.7B", model_revision="main", load_in_4bit=True)
    output = tmp_path / "merged"
    with pytest.raises(ValueError, match="commit hash"):
        merge_adapter(
            model_id="Qwen/Qwen3-1.7B",
            model_revision="main",
            adapter_path=tmp_path / "adapter",
            output_dir=output,
        )
    assert not output.exists()
