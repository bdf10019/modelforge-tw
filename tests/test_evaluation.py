import hashlib
from pathlib import Path

from modelforge_tw.dataset import generate_records, read_jsonl
from modelforge_tw.engines import FixtureEngine
from modelforge_tw.evaluation import assess_q4_release, compare_reports, evaluate
from modelforge_tw.safety import conservative_fallback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE_HASH = hashlib.sha256(b"release-gates").hexdigest()


def test_adapted_fixture_improves_event_f1_and_keeps_fixture_label():
    records = [record for record in generate_records(6, 2, 4) if record.split == "test"]
    base = evaluate(FixtureEngine("base"), records)
    adapted = evaluate(FixtureEngine("adapted"), records)
    assert adapted.evidence_kind == "fixture"
    assert adapted.metrics["event_type_macro_f1"] > base.metrics["event_type_macro_f1"]
    comparison = compare_reports(base, adapted)
    assert comparison["dataset_match"] is True
    assert comparison["release_gate_match"] is True


def test_injection_challenge_requires_review():
    records = read_jsonl(PROJECT_ROOT / "data" / "source" / "challenge.jsonl")
    injection = [record for record in records if "injection" in record.record_id]
    report = evaluate(FixtureEngine("adapted"), injection)
    assert report.metrics["injection_resistance"] == 1.0


def test_q4_gate_rejects_macro_f1_regression():
    records = [record for record in generate_records(6, 2, 4) if record.split == "test"]
    baseline = evaluate(FixtureEngine("adapted"), records).model_copy(
        update={"max_new_tokens": 128, "release_gate_sha256": GATE_HASH}
    )
    candidate = evaluate(FixtureEngine("base"), records).model_copy(
        update={
            "max_new_tokens": 128,
            "prompt_version": baseline.prompt_version,
            "release_gate_sha256": GATE_HASH,
        }
    )
    result = assess_q4_release(
        baseline,
        candidate,
        {"macro_f1_max_regression": 0.015, "raw_schema_max_regression": 0.005},
    )
    assert result["passed"] is False
    assert result["checks"]["event_type_macro_f1"]["passed"] is False


def test_q4_gate_rejects_inference_contract_mismatch():
    records = [record for record in generate_records(2, 1, 2) if record.split == "test"]
    baseline = evaluate(FixtureEngine("adapted"), records).model_copy(
        update={"max_new_tokens": 128, "release_gate_sha256": GATE_HASH}
    )
    candidate = evaluate(FixtureEngine("adapted"), records).model_copy(
        update={"max_new_tokens": 256, "release_gate_sha256": GATE_HASH}
    )
    result = assess_q4_release(
        baseline,
        candidate,
        {"macro_f1_max_regression": 0.015, "raw_schema_max_regression": 0.005},
    )
    assert result["passed"] is False
    assert result["reason"] == "comparison_contract_mismatch"
    assert result["compatibility_checks"]["max_new_tokens"] is False


def test_q4_gate_rejects_missing_release_gate_hash():
    records = [record for record in generate_records(2, 1, 2) if record.split == "test"]
    baseline = evaluate(FixtureEngine("adapted"), records).model_copy(
        update={"max_new_tokens": 128}
    )
    candidate = evaluate(FixtureEngine("adapted"), records).model_copy(
        update={"max_new_tokens": 128}
    )
    result = assess_q4_release(
        baseline,
        candidate,
        {"macro_f1_max_regression": 0.015, "raw_schema_max_regression": 0.005},
    )
    assert result["passed"] is False
    assert result["reason"] == "comparison_contract_mismatch"
    assert result["compatibility_checks"]["release_gate_hash"] is False


def test_q4_gate_rejects_context_window_mismatch():
    records = [record for record in generate_records(2, 1, 2) if record.split == "test"]
    baseline = evaluate(FixtureEngine("adapted"), records).model_copy(
        update={
            "max_new_tokens": 128,
            "context_window_tokens": 1024,
            "release_gate_sha256": GATE_HASH,
        }
    )
    candidate = baseline.model_copy(update={"context_window_tokens": 2048})
    result = assess_q4_release(
        baseline,
        candidate,
        {"macro_f1_max_regression": 0.015, "raw_schema_max_regression": 0.005},
    )
    assert result["passed"] is False
    assert result["reason"] == "comparison_contract_mismatch"
    assert result["compatibility_checks"]["context_window_tokens"] is False


def test_evaluation_can_resume_from_cached_inference_results():
    records = [record for record in generate_records(1, 1, 1) if record.split == "test"]
    cached = {}
    first = evaluate(
        FixtureEngine("adapted"),
        records,
        result_callback=lambda record, result: cached.__setitem__(record.record_id, result),
    )

    class CacheOnlyEngine(FixtureEngine):
        def generate(self, request):
            raise AssertionError("all inference results should come from cache")

    resumed = evaluate(CacheOnlyEngine("adapted"), records, cached_results=cached)
    assert resumed.metrics == first.metrics
    assert resumed.latency_ms == first.latency_ms


def test_invalid_raw_output_keeps_required_fallback_and_injection_policy():
    records = read_jsonl(PROJECT_ROOT / "data" / "source" / "challenge.jsonl")
    injection = next(record for record in records if "injection" in record.record_id)
    result = FixtureEngine("adapted").generate(injection.input)
    result.raw_text = '{"event_type":'
    result.output = conservative_fallback("MODEL_OUTPUT_INVALID")
    report = evaluate(
        FixtureEngine("adapted"),
        [injection],
        cached_results={injection.record_id: result},
    )
    assert report.metrics["raw_json_schema_rate"] == 0.0
    assert report.metrics["required_field_rate"] == 1.0
    assert report.metrics["injection_resistance"] == 1.0
