import json
import shlex
import shutil
from pathlib import Path, PurePosixPath

import yaml

from modelforge_tw import cli
from modelforge_tw.dataset import load_splits
from modelforge_tw.training import verify_training_dataset_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_kubernetes_manifests_parse_and_default_to_q8_candidate():
    documents = list(
        yaml.safe_load_all((PROJECT_ROOT / "infra" / "k8s" / "app.yaml").read_text("utf-8"))
    )
    config = next(item for item in documents if item["kind"] == "ConfigMap")
    assert config["data"]["MODELFORGE_MODEL_ID"] == "modelforge-tw-q8_0"
    assert config["data"]["MODELFORGE_MODEL_VERSION"] == "qwen3-1.7b-merged-v5-q8_0-policy-v2"
    assert config["data"]["MODELFORGE_QUANTIZATION"] == "q8_0"
    assert config["data"]["MODELFORGE_CONTEXT_WINDOW_TOKENS"] == "2048"
    assert config["data"]["MODELFORGE_SAFETY_POLICY_VERSION"] == "risk-precedence-v2"
    assert config["data"]["MODELFORGE_MAX_REQUEST_BODY_BYTES"] == "32768"
    assert config["data"]["MODELFORGE_MAX_MODEL_RESPONSE_BYTES"] == "262144"
    assert config["data"]["MODELFORGE_QUEUE_TIMEOUT_SECONDS"] == "5"
    assert config["data"]["MODELFORGE_MAX_QUEUE_DEPTH"] == "16"
    assert config["data"]["MODELFORGE_SHUTDOWN_GRACE_SECONDS"] == "65"
    assert config["data"]["MODELFORGE_READINESS_TIMEOUT_SECONDS"] == "4"

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


def test_compose_baseline_is_non_privileged_and_does_not_log_inputs():
    compose = yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text("utf-8"))
    api = compose["services"]["api"]
    assert api["environment"]["MODELFORGE_LOG_INPUTS"] == "false"
    assert api["environment"]["MODELFORGE_MAX_REQUEST_BODY_BYTES"] == "32768"
    assert api["environment"]["MODELFORGE_MAX_MODEL_RESPONSE_BYTES"] == "262144"
    assert api["environment"]["MODELFORGE_QUEUE_TIMEOUT_SECONDS"] == "5"
    assert api["environment"]["MODELFORGE_MAX_QUEUE_DEPTH"] == "8"
    assert api["environment"]["MODELFORGE_SHUTDOWN_GRACE_SECONDS"] == "35"
    assert api["environment"]["MODELFORGE_API_KEY"] is None
    assert api["read_only"] is True
    assert "/tmp" in api["tmpfs"]
    assert api["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in api["security_opt"]
    assert api["ports"] == ["127.0.0.1:8000:8000"]


def test_kubernetes_runtime_security_and_probe_contracts():
    documents = list(
        yaml.safe_load_all((PROJECT_ROOT / "infra" / "k8s" / "app.yaml").read_text("utf-8"))
    )
    deployments = {
        item["metadata"]["name"]: item for item in documents if item["kind"] == "Deployment"
    }
    model = deployments["llama-server"]["spec"]["template"]["spec"]["containers"][0]
    model_pod = deployments["llama-server"]["spec"]["template"]["spec"]
    api_pod = deployments["modelforge-api"]["spec"]["template"]["spec"]
    api = api_pod["containers"][0]

    assert model["readinessProbe"]["httpGet"]["path"] == "/health"
    assert model["livenessProbe"]["httpGet"]["path"] == "/health"
    assert model["startupProbe"]["httpGet"]["path"] == "/health"
    assert model["startupProbe"]["failureThreshold"] * model["startupProbe"]["periodSeconds"] == 300
    assert model_pod["automountServiceAccountToken"] is False
    assert model_pod["securityContext"]["runAsNonRoot"] is True
    assert model_pod["securityContext"]["runAsUser"] == 10001
    assert model_pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert model["securityContext"]["allowPrivilegeEscalation"] is False
    assert model["securityContext"]["readOnlyRootFilesystem"] is True
    assert model["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert api_pod["securityContext"]["runAsNonRoot"] is True
    assert api_pod["securityContext"]["runAsUser"] == 10001
    assert api_pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert api_pod["automountServiceAccountToken"] is False
    assert api_pod["terminationGracePeriodSeconds"] == 75
    assert api["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert api["readinessProbe"]["timeoutSeconds"] == 5
    assert api["livenessProbe"]["httpGet"]["path"] == "/healthz"
    assert api["securityContext"]["allowPrivilegeEscalation"] is False
    assert api["securityContext"]["readOnlyRootFilesystem"] is True
    assert api["securityContext"]["capabilities"]["drop"] == ["ALL"]
    api_key = next(item for item in api["env"] if item["name"] == "MODELFORGE_API_KEY")
    assert api_key["valueFrom"]["secretKeyRef"] == {
        "name": "modelforge-api-auth",
        "key": "api-key",
        "optional": False,
    }


def test_training_job_is_fail_closed_and_requests_one_gpu():
    job = yaml.safe_load((PROJECT_ROOT / "infra" / "k8s" / "training-job.yaml").read_text("utf-8"))
    pod = job["spec"]["template"]["spec"]
    trainer = pod["containers"][0]
    assert job["spec"]["backoffLimit"] == 0
    assert pod["restartPolicy"] == "Never"
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["runAsUser"] == 10001
    assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert trainer["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert trainer["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert trainer["securityContext"]["allowPrivilegeEscalation"] is False
    assert trainer["securityContext"]["readOnlyRootFilesystem"] is True
    assert trainer["securityContext"]["capabilities"]["drop"] == ["ALL"]
    mounts = {item["mountPath"] for item in trainer["volumeMounts"]}
    assert mounts == {"/app/artifacts", "/app/data/generated", "/cache", "/tmp"}
    environment = {item["name"]: item["value"] for item in trainer["env"]}
    assert environment == {
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HOME": "/cache/huggingface",
        "XDG_CACHE_HOME": "/cache",
        "TMPDIR": "/tmp",
    }
    assert "configs/training.json" in trainer["args"][0]


def test_container_files_run_as_non_root_and_disable_input_logging():
    cpu = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    gpu = (PROJECT_ROOT / "Dockerfile.gpu").read_text(encoding="utf-8")
    for dockerfile in (cpu, gpu):
        assert "USER appuser" in dockerfile
        assert "MODELFORGE_LOG_INPUTS=false" in dockerfile
        assert "MODELFORGE_MAX_REQUEST_BODY_BYTES=32768" in dockerfile
        assert "MODELFORGE_MAX_MODEL_RESPONSE_BYTES=262144" in dockerfile
        assert "MODELFORGE_QUEUE_TIMEOUT_SECONDS=5" in dockerfile
        assert "MODELFORGE_MAX_QUEUE_DEPTH=8" in dockerfile
        assert "MODELFORGE_SHUTDOWN_GRACE_SECONDS=35" in dockerfile
        assert '"modelforge", "serve"' in dockerfile


def test_training_job_generates_mounted_data_when_cli_is_installed(tmp_path, monkeypatch):
    job = yaml.safe_load((PROJECT_ROOT / "infra" / "k8s" / "training-job.yaml").read_text("utf-8"))
    trainer = job["spec"]["template"]["spec"]["containers"][0]
    command = shlex.split(trainer["args"][0])
    separator = command.index("&&")
    assert command[0] == command[separator + 1] == "modelforge"

    installed_root = tmp_path / "installed" / "lib" / "python3.12"
    monkeypatch.setattr(cli, "PROJECT_ROOT", installed_root)
    parser = cli.build_parser()
    generate = parser.parse_args(command[1:separator])
    train = parser.parse_args(command[separator + 2 :])
    app_root = PurePosixPath("/app")
    generated_path = PurePosixPath(generate.output.as_posix())
    challenge_path = PurePosixPath(generate.challenge.as_posix())
    config_path = PurePosixPath(train.config.as_posix())
    writable_mounts = {
        PurePosixPath(item["mountPath"])
        for item in trainer["volumeMounts"]
        if not item.get("readOnly", False)
    }
    assert generated_path in writable_mounts

    config = json.loads(
        (PROJECT_ROOT / config_path.relative_to(app_root)).read_text(encoding="utf-8")
    )
    assert generated_path == app_root / config["data_dir"]

    # Map only the declared container paths into a temporary /app workspace.
    workspace = tmp_path / "app"
    generate.output = workspace / generated_path.relative_to(app_root)
    generate.challenge = workspace / challenge_path.relative_to(app_root)
    generate.challenge.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / challenge_path.relative_to(app_root), generate.challenge)
    generate.train_per_type = generate.validation_per_type = generate.test_per_type = 1
    assert generate.func(generate) == 0

    manifest = verify_training_dataset_manifest(generate.output)
    assert manifest.files["challenge.jsonl"].records == 14
    assert len(load_splits(generate.output, ["train", "validation", "test", "challenge"])) == 35
    assert not installed_root.exists()


def test_local_llama_launcher_has_measured_context_capacity_guard():
    script = (PROJECT_ROOT / "scripts" / "serve-gguf.ps1").read_text(encoding="utf-8")
    assert "[int]$ContextSize = 8192" in script
    assert "$minimumSlotContext = 1536" in script
    assert "$ContextSize / $Parallel" in script
    assert "低於最低" in script
    assert '$serverPath = Join-Path $projectRoot "tools\\llama.cpp-bin\\llama-server.exe"' in script
    assert "$serverHash -ne $ExpectedServerSha256.ToLowerInvariant()" in script
    assert "quantization-report.json" in script
    assert "GGUF 與 quantization report 不符" in script
    assert "Assert-NoReparsePointInPath -LiteralPath $model" in script
    assert "$workspaceRoot = Split-Path -Parent $projectRoot" in script
    assert "Split-Path -Parent (Split-Path -Parent $projectRoot)" not in script


def test_llama_preparation_rejects_stale_or_wrong_runtime_files():
    script = (PROJECT_ROOT / "scripts" / "prepare-llama.ps1").read_text(encoding="utf-8")
    assert "[IO.Compression.ZipFile]::OpenRead" in script
    assert "Compare-Object -ReferenceObject $expectedFiles -DifferenceObject $actualFiles" in script
    assert "binary directory 與鎖定壓縮檔清單不符" in script
    assert '$quantizer = Join-Path $binaryRoot "llama-quantize.exe"' in script
    assert '$server = Join-Path $binaryRoot "llama-server.exe"' in script
    assert "$quantizerHash -ne $ExpectedQuantizerSha256.ToLowerInvariant()" in script
    assert "$serverHash -ne $ExpectedServerSha256.ToLowerInvariant()" in script


def test_ci_uses_the_same_cross_platform_full_verification_entrypoint():
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github" / "workflows" / "verify.yml").read_text(encoding="utf-8")
    )
    runs = [step.get("run") for step in workflow["jobs"]["test"]["steps"] if step.get("run")]
    assert runs[-1] == "pwsh -NoProfile -File scripts/verify.ps1"

    script = (PROJECT_ROOT / "scripts" / "verify.ps1").read_text(encoding="utf-8")
    assert ".venv\\Scripts\\python.exe" in script
    assert ".venv/bin/python" in script
    assert "Get-Command python" in script
    assert "pip install --no-deps --target" in script
    assert "verify-local-evidence.ps1" in script


def test_quantization_report_uses_same_directory_atomic_replacement():
    script = (PROJECT_ROOT / "scripts" / "quantize.ps1").read_text(encoding="utf-8")
    assert 'Join-Path $outputRoot (".quantization-report."' in script
    assert "Move-Item -Force -LiteralPath $reportTemporary -Destination $reportPath" in script
    assert "Remove-Item -Force -LiteralPath $reportTemporary" in script


def test_quantization_revalidates_source_and_records_tool_lineage():
    script = (PROJECT_ROOT / "scripts" / "quantize.ps1").read_text(encoding="utf-8")
    assert "$mergeReport.files.PSObject.Properties" in script
    assert "merge report 來源檔驗證失敗" in script
    assert "source_merge_report_sha256 = $mergeReportHash" in script
    assert "source_files = $sourceFiles" in script
    assert "llama_cpp_commit = $sourceCommit" in script
    assert "quantizer_sha256 = $quantizerHash" in script
    assert "量化 output directory 必須為空" in script
    assert "[IO.FileAttributes]::ReparsePoint" in script
    assert '$sourceRoot = Join-Path $projectRoot "tools\\llama.cpp-src"' in script
    assert '$quantizerPath = Join-Path $binaryRoot "llama-quantize.exe"' in script
    assert "Assert-NoReparsePointInPath -LiteralPath $mergedPath" in script
    assert "Assert-NoReparsePointInPath -LiteralPath $outputRoot" in script
    assert "git -C $sourceRoot rev-parse HEAD" in script
    assert "git -C $sourceRoot status --porcelain --untracked-files=normal" in script
    assert "llama.cpp source tree 必須保持乾淨" in script
    assert "$quantizerHash -ne $ExpectedQuantizerSha256.ToLowerInvariant()" in script
    assert "llama-quantize.exe SHA-256 不符" in script
    assert "merge report 來源檔不能是 reparse point" in script
    assert "[StringComparison]::OrdinalIgnoreCase" in script
    assert "量化路徑必須位於專案目錄內" in script
    assert "$workspaceRoot = Split-Path -Parent $projectRoot" in script
    assert "Split-Path -Parent (Split-Path -Parent $projectRoot)" not in script
    assert "Move-Item -LiteralPath $bridgeQ8 -Destination $q8" in script
    assert "Move-Item -Force -LiteralPath $bridgeQ8" not in script
