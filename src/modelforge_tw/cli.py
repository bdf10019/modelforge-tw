from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path, PureWindowsPath

from modelforge_tw.benchmark import run_benchmark, write_benchmark
from modelforge_tw.dataset import audit_records, load_splits, sha256_file, write_dataset
from modelforge_tw.engines import FixtureEngine, OpenAIHTTPInferenceEngine
from modelforge_tw.engines.openai_http import OPENAI_HTTP_CONTRACT_VERSION
from modelforge_tw.evaluation import (
    EVALUATION_CONTRACT_VERSION,
    assess_q4_release,
    compare_reports,
    evaluate,
    write_report,
)
from modelforge_tw.external_calibration import build_external_calibration
from modelforge_tw.io_utils import (
    MAX_JSONL_LINE_BYTES,
    atomic_write_text,
    iter_utf8_lines_limited,
    read_utf8_text_limited,
)
from modelforge_tw.model_identity import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_BASE_MODEL_REVISION,
    DEFAULT_MAX_NEW_TOKENS,
    validate_immutable_model_revision,
    validate_model_version,
    validate_quantization,
)
from modelforge_tw.prompts import (
    DEFAULT_PROMPT_VERSION,
    MINIMAL_PROMPT_VERSION,
    OPTIMIZED_PROMPT_V2_VERSION,
    OPTIMIZED_PROMPT_V3_VERSION,
    OPTIMIZED_PROMPT_V4_VERSION,
    OPTIMIZED_PROMPT_V5_VERSION,
    OPTIMIZED_PROMPT_V6_VERSION,
    OPTIMIZED_PROMPT_VERSION,
)
from modelforge_tw.registry import ArtifactRef, FileRegistry, ModelManifest
from modelforge_tw.safety import SAFETY_POLICY_VERSION, SUPPORTED_SAFETY_POLICY_VERSIONS
from modelforge_tw.schemas import DatasetRecord, EvaluationReport, InferenceResult, TriageRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _engine_from_args(args: argparse.Namespace):
    if args.engine.startswith("fixture-"):
        return FixtureEngine(
            args.engine.removeprefix("fixture-"),
            safety_policy_version=args.safety_policy_version,
        )
    if args.engine == "llama-http":
        if args.model_version is None:
            raise ValueError("measured engines require --model-version")
        if args.quantization is None:
            raise ValueError("llama-http requires --quantization")
        return OpenAIHTTPInferenceEngine(
            base_url=args.base_url,
            model_id=args.model_id,
            model_revision=args.model_revision,
            model_version=args.model_version,
            quantization=args.quantization,
            prompt_version=args.prompt_version,
            timeout_seconds=args.timeout,
            max_tokens=args.max_new_tokens,
            context_window_tokens=args.context_window_tokens,
            max_response_bytes=args.max_model_response_bytes,
            safety_policy_version=args.safety_policy_version,
        )
    if args.engine == "transformers":
        if args.model_version is None:
            raise ValueError("measured engines require --model-version")
        if bool(args.adapter) != bool(args.adapter_training_report):
            raise ValueError("--adapter and --adapter-training-report must be provided together")
        from modelforge_tw.engines.transformers_local import TransformersInferenceEngine

        return TransformersInferenceEngine(
            model_id=args.model_id,
            model_revision=args.model_revision,
            adapter_path=str(args.adapter) if args.adapter else None,
            adapter_training_report=(
                str(args.adapter_training_report) if args.adapter_training_report else None
            ),
            prompt_version=args.prompt_version,
            load_in_4bit=args.load_in_4bit,
            model_version=args.model_version,
            max_new_tokens=args.max_new_tokens,
            context_window_tokens=args.context_window_tokens,
            safety_policy_version=args.safety_policy_version,
        )
    raise ValueError(f"unsupported CLI engine: {args.engine}")


def command_doctor(_: argparse.Namespace) -> int:
    checks: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "project_root_exists": PROJECT_ROOT.exists(),
        "frontier_research_directory_exists": (PROJECT_ROOT.parent / "前沿研究版").exists(),
    }
    try:
        import torch

        checks.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        checks["torch"] = "not_installed"
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0


def command_generate_data(args: argparse.Namespace) -> int:
    manifest = write_dataset(
        args.output,
        challenge_source=args.challenge,
        train_per_type=args.train_per_type,
        validation_per_type=args.validation_per_type,
        test_per_type=args.test_per_type,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def command_audit_data(args: argparse.Namespace) -> int:
    records = load_splits(args.data, args.splits)
    audit = audit_records(records)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if audit["passed"] else 1


def command_prepare_external_calibration(args: argparse.Namespace) -> int:
    config = json.loads(read_utf8_text_limited(args.config))
    manifest = build_external_calibration(args.source, args.output, config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _record_sha256(record: DatasetRecord) -> str:
    payload = record.model_dump_json()
    return hashlib.sha256(payload.encode()).hexdigest()


def _evaluation_fingerprint(args: argparse.Namespace) -> str:
    adapter_files: dict[str, str] = {}
    if args.adapter:
        adapter = args.adapter.resolve()
        paths = sorted(adapter.rglob("*")) if adapter.is_dir() else [adapter]
        adapter_files = {
            path.relative_to(adapter).as_posix() if adapter.is_dir() else path.name: sha256_file(
                path
            )
            for path in paths
            if path.is_file()
        }
    payload = {
        "cache_fingerprint_version": "9",
        "openai_http_contract_version": OPENAI_HTTP_CONTRACT_VERSION,
        "safety_policy_version": args.safety_policy_version,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "engine": args.engine,
        "base_url": args.base_url,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_version": args.model_version,
        "quantization": args.quantization,
        "prompt_version": args.prompt_version,
        "load_in_4bit": args.load_in_4bit,
        "max_new_tokens": args.max_new_tokens,
        "context_window_tokens": args.context_window_tokens,
        "max_model_response_bytes": args.max_model_response_bytes,
        "adapter_files": adapter_files,
        "adapter_training_report_sha256": (
            sha256_file(args.adapter_training_report) if args.adapter_training_report else None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_evaluation_cache(
    cache_path: Path,
    *,
    fingerprint: str,
    record_hashes: dict[str, str],
) -> tuple[dict[str, InferenceResult], int]:
    cached_results: dict[str, InferenceResult] = {}
    rejected_lines = 0
    for bounded_line in iter_utf8_lines_limited(
        cache_path,
        max_line_bytes=MAX_JSONL_LINE_BYTES,
    ):
        line = bounded_line.text
        if bounded_line.error is not None:
            rejected_lines += 1
            continue
        if line is None or not line.strip():
            continue
        try:
            cached = json.loads(line)
            record_id = cached.get("record_id")
            if (
                cached.get("fingerprint") == fingerprint
                and record_id in record_hashes
                and cached.get("record_sha256") == record_hashes[record_id]
            ):
                cached_results[record_id] = InferenceResult.model_validate(cached["result"])
        except (AttributeError, KeyError, TypeError, ValueError):
            rejected_lines += 1
    return cached_results, rejected_lines


def _cache_needs_line_boundary(cache_path: Path) -> bool:
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return False
    with cache_path.open("rb") as cache_reader:
        cache_reader.seek(-1, 2)
        return cache_reader.read(1) not in {b"\n", b"\r"}


def command_evaluate(args: argparse.Namespace) -> int:
    records = load_splits(args.data, args.splits)
    if args.limit is not None:
        records = records[: args.limit]
    engine = _engine_from_args(args)

    fingerprint = _evaluation_fingerprint(args)
    cache_path = args.resume_cache_path or args.output.with_suffix(
        args.output.suffix + ".partial.jsonl"
    )
    record_hashes = {record.record_id: _record_sha256(record) for record in records}
    cached_results: dict[str, InferenceResult] = {}
    rejected_cache_lines = 0
    if not args.no_resume_cache and cache_path.exists():
        cached_results, rejected_cache_lines = _load_evaluation_cache(
            cache_path,
            fingerprint=fingerprint,
            record_hashes=record_hashes,
        )
    if cached_results:
        print(
            f"evaluation resume cache: {len(cached_results)}/{len(records)}",
            file=sys.stderr,
            flush=True,
        )
    if rejected_cache_lines:
        print(
            f"evaluation resume cache: ignored {rejected_cache_lines} invalid line(s); "
            "affected records will be regenerated",
            file=sys.stderr,
            flush=True,
        )

    def show_progress(completed: int, total: int) -> None:
        if completed == total or completed % 5 == 0:
            print(f"evaluation progress: {completed}/{total}", file=sys.stderr, flush=True)

    serialized_gates = read_utf8_text_limited(args.gates)
    gate_config = json.loads(serialized_gates)
    release_gate_sha256 = hashlib.sha256(serialized_gates.encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_writer = (
        None if args.no_resume_cache else cache_path.open("a", encoding="utf-8", newline="\n")
    )
    if cache_writer is not None and _cache_needs_line_boundary(cache_path):
        cache_writer.write("\n")
        cache_writer.flush()

    def cache_result(record: DatasetRecord, result: InferenceResult) -> None:
        if cache_writer is None:
            return
        record_id = record.record_id
        cache_writer.write(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "record_id": record_id,
                    "record_sha256": record_hashes[record_id],
                    "result": result.model_dump(mode="json"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        cache_writer.flush()

    try:
        report = evaluate(
            engine,
            records,
            gates=gate_config["absolute"],
            release_gate_sha256=release_gate_sha256,
            progress=show_progress,
            cached_results=cached_results,
            result_callback=cache_result,
        )
    finally:
        if cache_writer is not None:
            cache_writer.close()
    write_report(report, args.output)
    summary = report.model_dump(exclude={"cases"}, mode="json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report.release_passed or report.evidence_kind == "fixture" else 2


def command_compare(args: argparse.Namespace) -> int:
    baseline = EvaluationReport.model_validate_json(read_utf8_text_limited(args.baseline))
    candidate = EvaluationReport.model_validate_json(read_utf8_text_limited(args.candidate))
    comparison = compare_reports(baseline, candidate)
    if args.profile == "q4":
        gate_config = json.loads(read_utf8_text_limited(args.gates))
        comparison["quantization_gate"] = assess_q4_release(
            baseline,
            candidate,
            gate_config["q4_relative_to_reference"],
        )
    atomic_write_text(
        args.output,
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    passed = comparison["compatible"]
    if args.profile == "q4":
        passed = passed and comparison["quantization_gate"]["passed"]
    return 0 if passed else 2


def command_benchmark(args: argparse.Namespace) -> int:
    records = load_splits(args.data, args.splits)
    if args.limit is not None:
        records = records[: args.limit]
    engine = _engine_from_args(args)
    report = run_benchmark(
        engine,
        records,
        concurrency_levels=args.concurrency,
        requests_per_level=args.requests,
        warmup_requests=args.warmup,
    )
    write_benchmark(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


def command_demo(_: argparse.Namespace) -> int:
    engine = FixtureEngine("adapted")
    request = TriageRequest(
        incident_text="三號泵浦軸承溫度持續升高，並且傳出金屬摩擦聲。",
        asset_type="泵浦",
    )
    result = engine.generate(request)
    print(result.model_dump_json(indent=2))
    return 0


def command_registry(args: argparse.Namespace) -> int:
    from modelforge_tw.training import TrainingConfig, TrainingReportAttestation

    registry = FileRegistry(args.registry, artifact_root=PROJECT_ROOT)
    if args.registry_action == "show":
        state = registry.load()
    elif args.registry_action == "anchor":
        print(registry.anchor().model_dump_json(indent=2))
        return 0
    elif args.registry_action == "add":
        if not args.manifest_id or not args.evaluation or not args.training_report:
            raise ValueError("registry add requires manifest id, evaluation, and training report")
        evaluation = EvaluationReport.model_validate_json(read_utf8_text_limited(args.evaluation))
        training = TrainingReportAttestation.model_validate_json(
            read_utf8_text_limited(args.training_report)
        )
        training_config = TrainingConfig.model_validate(training.config)
        if training_config.prompt_version != evaluation.prompt_version:
            raise ValueError(
                "training prompt version does not match evaluation: "
                f"{training_config.prompt_version} != {evaluation.prompt_version}"
            )
        configured_data_dir = training_config.data_dir
        raw_data_dir = str(training.config.get("data_dir", training_config.data_dir))
        windows_data_dir = PureWindowsPath(raw_data_dir)
        if "\\" in raw_data_dir and not windows_data_dir.is_absolute():
            configured_data_dir = Path(*windows_data_dir.parts)
        data_dir = (
            (PROJECT_ROOT / configured_data_dir).resolve()
            if not configured_data_dir.is_absolute()
            else configured_data_dir.resolve()
        )
        try:
            data_dir.relative_to(PROJECT_ROOT)
        except ValueError as error:
            raise ValueError(
                "training data directory must remain inside the project root"
            ) from error
        dataset_manifest_path = data_dir / "manifest.json"
        dataset_manifest = json.loads(read_utf8_text_limited(dataset_manifest_path))
        if dataset_manifest.get("dataset_manifest_sha256") != training.dataset_manifest_sha256:
            raise ValueError("training report does not match the local dataset manifest")
        if args.registry_quantization != evaluation.quantization:
            raise ValueError(
                "registry quantization does not match evaluation: "
                f"{args.registry_quantization} != {evaluation.quantization}"
            )
        artifact_refs: list[ArtifactRef] = []
        for specification in args.artifact:
            kind, separator, raw_path = specification.partition("=")
            if not separator or kind not in {"adapter", "merged", "gguf", "tokenizer"}:
                raise ValueError("artifact must use kind=path with a supported kind")
            candidate = Path(raw_path)
            resolved = (
                (PROJECT_ROOT / candidate).resolve()
                if not candidate.is_absolute()
                else candidate.resolve()
            )
            try:
                resolved.relative_to(PROJECT_ROOT)
            except ValueError as error:
                raise ValueError("artifact must remain inside the project root") from error
            paths = (
                sorted(path for path in resolved.rglob("*") if path.is_file())
                if resolved.is_dir()
                else [resolved]
            )
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
                artifact_refs.append(
                    ArtifactRef.model_validate(
                        {
                            "kind": kind,
                            "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size,
                        }
                    )
                )
        manifest = ModelManifest(
            manifest_id=args.manifest_id,
            base_model=training.model_id,
            base_revision=training.resolved_model_revision,
            license=args.license,
            prompt_version=evaluation.prompt_version,
            dataset_manifest_sha256=training.dataset_manifest_sha256,
            training_config_sha256=training.training_config_sha256,
            release_gate_sha256=evaluation.release_gate_sha256,
            engine=args.registry_engine,
            quantization=args.registry_quantization,
            max_new_tokens=evaluation.max_new_tokens,
            structured_output_contract=evaluation.structured_output_contract,
            safety_policy_version=evaluation.safety_policy_version,
            evaluation_contract_version=evaluation.evaluation_contract_version,
            evaluation_dataset_sha256=evaluation.dataset_manifest_sha256,
            evaluation_engine=evaluation.engine,
            evaluation_model_id=evaluation.model_id,
            evaluation_model_version=evaluation.model_version,
            evaluation_model_revision=evaluation.model_revision,
            context_window_tokens=evaluation.context_window_tokens,
            runtime_identity_contract_version="1.0",
            evaluation_evidence_kind=evaluation.evidence_kind,
            evaluation_report=args.evaluation.resolve().relative_to(PROJECT_ROOT).as_posix(),
            evaluation_report_sha256=sha256_file(args.evaluation),
            training_report=args.training_report.resolve().relative_to(PROJECT_ROOT).as_posix(),
            training_report_sha256=sha256_file(args.training_report),
            release_passed=evaluation.release_passed,
            artifacts=artifact_refs,
        )
        state = registry.add(manifest)
    elif args.registry_action == "attest":
        if not args.manifest_id or not args.evaluation:
            raise ValueError("registry attest requires manifest id and evaluation")
        evaluation = EvaluationReport.model_validate_json(read_utf8_text_limited(args.evaluation))
        attested_report_path = args.evaluation.resolve().relative_to(PROJECT_ROOT).as_posix()
        state = registry.attest_evaluation(
            args.manifest_id,
            evaluation_report=attested_report_path,
            evaluation_report_sha256=sha256_file(args.evaluation),
            evaluation=evaluation,
        )
    elif args.registry_action == "promote":
        if not args.manifest_id:
            raise ValueError("registry promote requires a manifest id")
        state = registry.promote(args.manifest_id)
    elif args.registry_action == "rollback":
        state = registry.rollback()
    elif args.registry_action == "block":
        if not args.manifest_id:
            raise ValueError("registry block requires a manifest id")
        evaluation_report: str | None = None
        evaluation_report_sha256: str | None = None
        release_passed: bool | None = None
        if args.evaluation:
            evaluation = EvaluationReport.model_validate_json(
                read_utf8_text_limited(args.evaluation)
            )
            registry.validate_evaluation(args.manifest_id, evaluation)
            evaluation_report = args.evaluation.resolve().relative_to(PROJECT_ROOT).as_posix()
            evaluation_report_sha256 = sha256_file(args.evaluation)
            release_passed = evaluation.release_passed
        state = registry.block(
            args.manifest_id,
            evaluation_report=evaluation_report,
            evaluation_report_sha256=evaluation_report_sha256,
            release_passed=release_passed,
        )
    else:
        raise ValueError(f"unsupported registry action: {args.registry_action}")
    print(state.model_dump_json(indent=2))
    return 0


def command_train(args: argparse.Namespace) -> int:
    from modelforge_tw.training import load_training_config, run_training

    report = run_training(load_training_config(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


def command_merge(args: argparse.Namespace) -> int:
    from modelforge_tw.training import merge_adapter

    report = merge_adapter(
        model_id=args.model_id,
        model_revision=args.model_revision,
        adapter_path=args.adapter,
        training_report_path=args.training_report,
        output_dir=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_probe_model(args: argparse.Namespace) -> int:
    from modelforge_tw.engines.transformers_local import model_load_probe

    report = model_load_probe(
        args.model_id,
        model_revision=args.model_revision,
        load_in_4bit=args.load_in_4bit,
    )
    atomic_write_text(
        args.output,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelforge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.set_defaults(func=command_doctor)

    generate = subparsers.add_parser("generate-data")
    generate.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "generated")
    generate.add_argument(
        "--challenge", type=Path, default=PROJECT_ROOT / "data" / "source" / "challenge.jsonl"
    )
    generate.add_argument("--train-per-type", type=int, default=120)
    generate.add_argument("--validation-per-type", type=int, default=20)
    generate.add_argument("--test-per-type", type=int, default=30)
    generate.set_defaults(func=command_generate_data)

    audit = subparsers.add_parser("audit-data")
    audit.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "generated")
    audit.add_argument("--splits", nargs="+", default=["train", "validation", "test", "challenge"])
    audit.set_defaults(func=command_audit_data)

    external = subparsers.add_parser("prepare-external-calibration")
    external.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "external-calibration.json",
    )
    external.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "raw" / "ai4i2020.csv",
    )
    external.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "external" / "generated",
    )
    external.set_defaults(func=command_prepare_external_calibration)

    def add_engine_args(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--engine",
            choices=[
                "fixture-base",
                "fixture-prompt",
                "fixture-adapted",
                "llama-http",
                "transformers",
            ],
            default="fixture-adapted",
        )
        command.add_argument("--base-url", default="http://127.0.0.1:8081")
        command.add_argument("--model-id", default=DEFAULT_BASE_MODEL_ID)
        command.add_argument(
            "--model-revision",
            type=validate_immutable_model_revision,
            default=DEFAULT_BASE_MODEL_REVISION,
        )
        command.add_argument("--model-version", type=validate_model_version)
        command.add_argument("--quantization", type=validate_quantization)
        command.add_argument("--timeout", type=float, default=60)
        command.add_argument(
            "--prompt-version",
            choices=[
                MINIMAL_PROMPT_VERSION,
                OPTIMIZED_PROMPT_VERSION,
                OPTIMIZED_PROMPT_V2_VERSION,
                OPTIMIZED_PROMPT_V3_VERSION,
                OPTIMIZED_PROMPT_V4_VERSION,
                OPTIMIZED_PROMPT_V5_VERSION,
                OPTIMIZED_PROMPT_V6_VERSION,
            ],
            default=DEFAULT_PROMPT_VERSION,
        )
        command.add_argument("--adapter", type=Path)
        command.add_argument("--adapter-training-report", type=Path)
        command.add_argument("--load-in-4bit", action="store_true")
        command.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
        command.add_argument("--context-window-tokens", type=int, default=2048)
        command.add_argument("--max-model-response-bytes", type=int, default=262144)
        command.add_argument(
            "--safety-policy-version",
            choices=SUPPORTED_SAFETY_POLICY_VERSIONS,
            default=SAFETY_POLICY_VERSION,
        )

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "generated")
    evaluation.add_argument("--splits", nargs="+", default=["test", "challenge"])
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation-report.json"
    )
    evaluation.add_argument(
        "--gates", type=Path, default=PROJECT_ROOT / "configs" / "release-gates.json"
    )
    evaluation.add_argument(
        "--no-resume-cache",
        action="store_true",
        help="停用綁定資料與推論設定的逐筆續跑快取",
    )
    evaluation.add_argument(
        "--resume-cache-path",
        type=Path,
        help="指定另一份相同推論指紋的逐筆續跑快取",
    )
    add_engine_args(evaluation)
    evaluation.set_defaults(func=command_evaluate)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "comparison-report.json"
    )
    comparison.add_argument("--profile", choices=["metrics", "q4"], default="metrics")
    comparison.add_argument(
        "--gates", type=Path, default=PROJECT_ROOT / "configs" / "release-gates.json"
    )
    comparison.set_defaults(func=command_compare)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "generated")
    benchmark.add_argument("--splits", nargs="+", default=["test"])
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    benchmark.add_argument("--requests", type=int, default=20)
    benchmark.add_argument("--warmup", type=int, default=2)
    benchmark.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "benchmark-report.json"
    )
    add_engine_args(benchmark)
    benchmark.set_defaults(func=command_benchmark)

    demo = subparsers.add_parser("run-demo")
    demo.set_defaults(func=command_demo)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=None)

    registry = subparsers.add_parser("registry")
    registry.add_argument(
        "registry_action",
        choices=["show", "anchor", "add", "attest", "promote", "rollback", "block"],
    )
    registry.add_argument("manifest_id", nargs="?")
    registry.add_argument(
        "--registry", type=Path, default=PROJECT_ROOT / "outputs" / "model-registry.json"
    )
    registry.add_argument("--evaluation", type=Path)
    registry.add_argument("--training-report", type=Path)
    registry.add_argument("--artifact", action="append", default=[])
    registry.add_argument("--license", default="Apache-2.0")
    registry.add_argument("--registry-engine", default="transformers-local")
    registry.add_argument("--registry-quantization", default="bnb-nf4")
    registry.set_defaults(func=command_registry)

    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "training.json")
    train.set_defaults(func=command_train)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--model-id", default=DEFAULT_BASE_MODEL_ID)
    merge.add_argument(
        "--model-revision",
        type=validate_immutable_model_revision,
        default=DEFAULT_BASE_MODEL_REVISION,
    )
    merge.add_argument("--adapter", type=Path, required=True)
    merge.add_argument("--training-report", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(func=command_merge)

    probe = subparsers.add_parser("probe-model")
    probe.add_argument("--model-id", default=DEFAULT_BASE_MODEL_ID)
    probe.add_argument(
        "--model-revision",
        type=validate_immutable_model_revision,
        default=DEFAULT_BASE_MODEL_REVISION,
    )
    probe.add_argument("--load-in-4bit", action="store_true")
    probe.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "model-load-probe.json"
    )
    probe.set_defaults(func=command_probe_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn

        uvicorn.run("modelforge_tw.service:app", host=args.host, port=args.port, reload=False)
        return 0
    try:
        return int(args.func(args))
    except (FileNotFoundError, TimeoutError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
