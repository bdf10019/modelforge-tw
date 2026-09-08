import argparse
import hashlib
import json
from uuid import UUID

import pytest

from modelforge_tw import cli
from modelforge_tw.benchmark import run_benchmark
from modelforge_tw.dataset import generate_records
from modelforge_tw.engines import FixtureEngine


class StreamingFixtureEngine(FixtureEngine):
    def generate_streaming(self, request):
        result = self.generate(request)
        result.metadata = result.metadata.model_copy(update={"elapsed_ms": 5.0, "ttft_ms": 2.5})
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
    assert report["schema_version"] == "1.4"
    assert UUID(report["run_id"]).version == 4
    assert report["levels"][0]["prompt_tokens"] is None
    assert report["levels"][0]["prompt_token_sample_count"] == 0
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


class RecordingFixtureEngine(FixtureEngine):
    def __init__(self):
        super().__init__("adapted")
        self.sent_inputs = []

    def generate(self, request):
        self.sent_inputs.append(request.model_dump(mode="json"))
        return super().generate(request)


def hash_sent_inputs(inputs):
    return hashlib.sha256(
        json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


@pytest.mark.parametrize(("requests", "warmup"), [(1, 0), (3, 2), (5, 4)])
def test_benchmark_workload_hashes_match_sent_truncated_and_cyclic_inputs(
    requests, warmup, monkeypatch
):
    monkeypatch.setattr("modelforge_tw.benchmark._gpu_snapshot", lambda: None)
    records = generate_records(1, 1, 1)[:3]
    pool = [record.input.model_dump(mode="json") for record in records]
    engine = RecordingFixtureEngine()

    report = run_benchmark(
        engine,
        records,
        concurrency_levels=[1],
        requests_per_level=requests,
        warmup_requests=warmup,
    )

    sent_warmup = engine.sent_inputs[:warmup]
    sent_measured = engine.sent_inputs[warmup:]
    assert sent_warmup == [pool[index % len(pool)] for index in range(warmup)]
    assert sent_measured == [pool[index % len(pool)] for index in range(requests)]
    assert report["workload"] == {
        "hash_contract": "sha256-canonical-json-input-sequence-v1",
        "input_pool_sha256": hash_sent_inputs(pool),
        "input_pool_records": 3,
        "requests_per_level": requests,
        "warmup_input_sequence_sha256": hash_sent_inputs(sent_warmup),
    }
    assert report["levels"][0]["input_sequence_sha256"] == hash_sent_inputs(sent_measured)
    serialized = json.dumps(report, ensure_ascii=False)
    assert all(record.input.incident_text not in serialized for record in records)
    assert all(record.record_id not in serialized for record in records)


def test_benchmark_workload_hashes_detect_input_order_selection_and_count(monkeypatch):
    monkeypatch.setattr("modelforge_tw.benchmark._gpu_snapshot", lambda: None)
    records = generate_records(1, 1, 1)[:3]

    def benchmark(pool, requests=2, warmup=1, mode="adapted"):
        return run_benchmark(
            FixtureEngine(mode),
            pool,
            concurrency_levels=[1, 2],
            requests_per_level=requests,
            warmup_requests=warmup,
        )

    baseline = benchmark(records)
    same_inputs_other_engine = benchmark(records, mode="base")
    assert baseline["run_id"] != same_inputs_other_engine["run_id"]
    assert baseline["workload"] == same_inputs_other_engine["workload"]
    sequence_hash = baseline["levels"][0]["input_sequence_sha256"]
    assert all(level["input_sequence_sha256"] == sequence_hash for level in baseline["levels"])
    assert same_inputs_other_engine["levels"][0]["input_sequence_sha256"] == sequence_hash

    reordered = benchmark([records[1], records[0], records[2]])
    assert reordered["workload"]["input_pool_sha256"] != baseline["workload"]["input_pool_sha256"]
    assert reordered["levels"][0]["input_sequence_sha256"] != sequence_hash

    changed_unselected = [record.model_copy(deep=True) for record in records]
    changed_unselected[2].input.incident_text += "補充現場紀錄。"
    unselected_report = benchmark(changed_unselected)
    assert (
        unselected_report["workload"]["input_pool_sha256"]
        != baseline["workload"]["input_pool_sha256"]
    )
    assert unselected_report["levels"][0]["input_sequence_sha256"] == sequence_hash

    changed_selected = [record.model_copy(deep=True) for record in records]
    changed_selected[0].input.asset_type = "另一台設備"
    selected_report = benchmark(changed_selected)
    assert selected_report["levels"][0]["input_sequence_sha256"] != sequence_hash
    assert (
        selected_report["workload"]["warmup_input_sequence_sha256"]
        != baseline["workload"]["warmup_input_sequence_sha256"]
    )

    more_requests = benchmark(records, requests=5)
    assert (
        more_requests["workload"]["input_pool_sha256"] == baseline["workload"]["input_pool_sha256"]
    )
    assert more_requests["levels"][0]["input_sequence_sha256"] != sequence_hash
    more_warmup = benchmark(records, warmup=4)
    assert more_warmup["levels"][0]["input_sequence_sha256"] == sequence_hash
    assert (
        more_warmup["workload"]["warmup_input_sequence_sha256"]
        != baseline["workload"]["warmup_input_sequence_sha256"]
    )

    relabeled = [record.model_copy(deep=True) for record in records]
    relabeled[0].record_id = "changed-record-id"
    relabeled_report = benchmark(relabeled)
    assert relabeled_report["workload"] == baseline["workload"]
    assert relabeled_report["levels"][0]["input_sequence_sha256"] == sequence_hash


def test_benchmark_reports_partial_prompt_token_coverage_without_inventing_samples(monkeypatch):
    monkeypatch.setattr("modelforge_tw.benchmark._gpu_snapshot", lambda: None)

    class PartialTokenEngine(FixtureEngine):
        calls = 0

        def generate(self, request):
            result = super().generate(request)
            tokens = [7, None, 0][self.calls]
            self.calls += 1
            result.metadata = result.metadata.model_copy(update={"prompt_tokens": tokens})
            return result

    report = run_benchmark(
        PartialTokenEngine("adapted"),
        generate_records(1, 1, 1)[:3],
        concurrency_levels=[1],
        requests_per_level=3,
        warmup_requests=0,
    )

    assert report["passed"] is True
    assert report["levels"][0]["completed"] == 3
    assert report["levels"][0]["prompt_tokens"] == 7
    assert report["levels"][0]["prompt_token_sample_count"] == 2


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
    assert report["levels"][0]["error_samples"][0] == {
        "code": "inference_identity_mismatch",
        "exception_type": "InferenceIdentityMismatchError",
    }


def test_benchmark_report_does_not_retain_private_exception_text():
    private_detail = "https://private.example.invalid?api_key=do-not-record"

    class LeakyFailingEngine(FixtureEngine):
        def generate(self, request):
            raise RuntimeError(private_detail)

    report = run_benchmark(
        LeakyFailingEngine("adapted"),
        generate_records(1, 1, 1)[:1],
        concurrency_levels=[1],
        requests_per_level=1,
        warmup_requests=0,
    )

    serialized = json.dumps(report)
    assert report["levels"][0]["error_samples"] == [
        {"code": "inference_request_failed", "exception_type": "RuntimeError"}
    ]
    assert private_detail not in serialized
    assert "do-not-record" not in serialized


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
