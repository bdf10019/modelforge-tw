import argparse
import json

import pytest

from modelforge_tw import cli
from modelforge_tw.benchmark import run_benchmark
from modelforge_tw.dataset import generate_records
from modelforge_tw.engines import FixtureEngine


class StreamingFixtureEngine(FixtureEngine):
    def generate_streaming(self, request):
        result = self.generate(request)
        result.metadata = result.metadata.model_copy(update={"ttft_ms": 2.5})
        return result


def test_benchmark_keeps_unavailable_ttft_explicit():
    records = generate_records(1, 1, 1)[:4]
    report = run_benchmark(
        FixtureEngine("adapted"),
        records,
        concurrency_levels=[1, 2],
        requests_per_level=4,
        warmup_requests=1,
    )
    assert report["passed"] is True
    assert report["evidence_kind"] == "fixture"
    assert report["levels"][0]["ttft_ms"] is None
    assert report["levels"][0]["ttft_status"] == "not_available_for_non_streaming_adapter"
    assert report["levels"][0]["errors"] == 0
    assert report["levels"][0]["output_token_sample_count"] == 0
    assert report["levels"][0]["ttft_sample_count"] == 0
    assert report["schema_version"] == "1.2"
    assert report["safety_policy_version"] == "risk-precedence-v2"
    assert (
        report["environment"]["peak_process_rss_bytes"]
        >= report["environment"]["process_rss_before_bytes"]
    )
    assert report["environment"]["gpu_memory_scope"] == "host-wide nvidia-smi reading"


def test_benchmark_records_streaming_ttft_when_engine_supports_it():
    records = generate_records(1, 1, 1)[:2]
    report = run_benchmark(
        StreamingFixtureEngine("adapted"),
        records,
        concurrency_levels=[1],
        requests_per_level=2,
        warmup_requests=0,
    )
    assert report["levels"][0]["ttft_status"] == "measured_from_streaming_response"
    assert report["levels"][0]["ttft_ms"]["p50"] == 2.5
    assert report["levels"][0]["ttft_sample_count"] == 2


def test_benchmark_fails_closed_when_measured_requests_error():
    class FailingEngine(FixtureEngine):
        def generate(self, request):
            raise RuntimeError("simulated inference failure")

    records = generate_records(1, 1, 1)[:2]
    report = run_benchmark(
        FailingEngine("adapted"),
        records,
        concurrency_levels=[1],
        requests_per_level=2,
        warmup_requests=0,
    )

    assert report["passed"] is False
    assert report["failure_reason"] == "one_or_more_benchmark_requests_failed"
    assert report["levels"][0]["completed"] == 0
    assert report["levels"][0]["errors"] == 2


@pytest.mark.parametrize(
    ("concurrency_levels", "warmup_requests", "message"),
    [([], 0, "at least one concurrency"), ([1], -1, "cannot be negative")],
)
def test_benchmark_rejects_invalid_run_shapes(concurrency_levels, warmup_requests, message):
    with pytest.raises(ValueError, match=message):
        run_benchmark(
            FixtureEngine("adapted"),
            generate_records(1, 1, 1)[:1],
            concurrency_levels=concurrency_levels,
            requests_per_level=1,
            warmup_requests=warmup_requests,
        )


def test_benchmark_rejects_duplicate_concurrency_levels():
    with pytest.raises(ValueError, match="must be unique"):
        run_benchmark(
            FixtureEngine("adapted"),
            generate_records(1, 1, 1)[:1],
            concurrency_levels=[1, 1],
            requests_per_level=1,
        )


def test_benchmark_fails_closed_when_inference_identity_drifts():
    class DriftingEngine(FixtureEngine):
        calls = 0

        def generate(self, request):
            result = super().generate(request)
            self.calls += 1
            if self.calls == 2:
                result.metadata = result.metadata.model_copy(
                    update={"model_version": "different-v1"}
                )
            return result

    report = run_benchmark(
        DriftingEngine("adapted"),
        generate_records(1, 1, 1)[:2],
        concurrency_levels=[1],
        requests_per_level=2,
        warmup_requests=0,
    )

    assert report["passed"] is False
    assert report["levels"][0]["completed"] == 1
    assert report["levels"][0]["errors"] == 1
    assert "metadata changed" in report["levels"][0]["error_samples"][0]


def test_benchmark_cli_returns_nonzero_and_writes_failed_report(tmp_path, monkeypatch):
    failed_report = {"passed": False, "levels": []}
    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(cli, "load_splits", lambda _data, _splits: [object()])
    monkeypatch.setattr(cli, "_engine_from_args", lambda _args: object())
    monkeypatch.setattr(cli, "run_benchmark", lambda *_args, **_kwargs: failed_report)
    args = argparse.Namespace(
        data=tmp_path,
        splits=["test"],
        limit=None,
        concurrency=[1],
        requests=1,
        warmup=0,
        output=output,
    )

    assert cli.command_benchmark(args) == 2
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False
