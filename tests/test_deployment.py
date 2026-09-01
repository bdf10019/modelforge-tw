from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_kubernetes_manifests_parse_and_default_to_q8_candidate():
    documents = list(
        yaml.safe_load_all((PROJECT_ROOT / "infra" / "k8s" / "app.yaml").read_text("utf-8"))
    )
    config = next(item for item in documents if item["kind"] == "ConfigMap")
    assert config["data"]["MODELFORGE_MODEL_ID"] == "modelforge-tw-q8_0"
    assert config["data"]["MODELFORGE_QUANTIZATION"] == "q8_0"
    assert config["data"]["MODELFORGE_CONTEXT_WINDOW_TOKENS"] == "2048"
    assert config["data"]["MODELFORGE_SAFETY_POLICY_VERSION"] == "risk-precedence-v2"

    model_deployment = next(
        item
        for item in documents
        if item["kind"] == "Deployment" and item["metadata"]["name"] == "llama-server"
    )
    args = model_deployment["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "/models/modelforge-tw-q8_0.gguf" in args
    context_index = args.index("--ctx-size")
    parallel_index = args.index("--parallel")
    assert int(args[context_index + 1]) // int(args[parallel_index + 1]) == 2048
    assert (
        model_deployment["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"][
            "nvidia.com/gpu"
        ]
        == "1"
    )


def test_all_deployment_yaml_is_syntactically_valid():
    paths = [
        PROJECT_ROOT / "compose.yaml",
        *sorted((PROJECT_ROOT / "infra" / "k8s").glob("*.yaml")),
    ]
    for path in paths:
        documents = list(yaml.safe_load_all(path.read_text("utf-8")))
        assert documents and all(document is not None for document in documents)


def test_local_llama_launcher_has_measured_context_capacity_guard():
    script = (PROJECT_ROOT / "scripts" / "serve-gguf.ps1").read_text(encoding="utf-8")
    assert "[int]$ContextSize = 8192" in script
    assert "$minimumSlotContext = 1536" in script
    assert "$ContextSize / $Parallel" in script
    assert "低於最低" in script
