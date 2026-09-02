import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelforge_tw.dataset import generate_records, read_jsonl
from modelforge_tw.engines import FixtureEngine
from modelforge_tw.evaluation import assess_q4_release, compare_reports, evaluate
from modelforge_tw.safety import conservative_fallback
from modelforge_tw.schemas import EvaluationReport

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
    assert comparison["compatible"] is False
    assert comparison["compatibility_checks"]["prompt"] is False
    assert compare_reports(adapted, adapted)["compatible"] is True


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("dataset_manifest_sha256", "different-dataset", "dataset"),
        ("release_gate_sha256", GATE_HASH, "release_gate"),
        ("prompt_version", "minimal-v1", "prompt"),
        ("max_new_tokens", 256, "max_new_tokens"),
        ("context_window_tokens", 4096, "context_window_tokens"),
        ("structured_output_contract", "json-schema-v1", "structured_output_contract"),
        ("safety_policy_version", "risk-precedence-v1", "safety_policy_version"),
        ("evaluation_contract_version", "release-metrics-v1", "evaluation_contract_version"),
    ],
)
def test_report_comparison_rejects_every_contract_mismatch(field, value, check):
    records = [record for record in generate_records(2, 1, 2) if record.split == "test"]
    baseline = evaluate(FixtureEngine("adapted"), records)
    candidate = baseline.model_copy(update={field: value})

    comparison = compare_reports(baseline, candidate)

    assert comparison["compatible"] is False
    assert comparison["compatibility_checks"][check] is False


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


def test_evaluation_reports_postprocessing_interventions():
    records = [record for record in generate_records(2, 1, 2) if record.split == "test"]
    report = evaluate(FixtureEngine("adapted"), records)
    assert "postprocessing_intervention_rate" in report.metrics
    assert "forced_unknown_rate" in report.metrics
    assert "raw_evidence_span_precision" in report.metrics
    assert isinstance(report.validation_warning_counts, dict)
    assert all(isinstance(case.validation_warnings, list) for case in report.cases)


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


def test_evaluation_rejects_duplicate_record_ids():
    record = next(record for record in generate_records(1, 1, 1) if record.split == "test")
    with pytest.raises(ValueError, match="record IDs must be unique"):
        evaluate(FixtureEngine("adapted"), [record, record])


def test_evaluation_report_rejects_inconsistent_case_count_and_hash():
    records = [record for record in generate_records(1, 1, 1) if record.split == "test"]
    payload = evaluate(FixtureEngine("adapted"), records).model_dump(mode="json")
    payload["case_count"] += 1
    with pytest.raises(ValidationError, match="case_count"):
        EvaluationReport.model_validate(payload)

    payload["case_count"] = len(payload["cases"])
    payload["dataset_manifest_sha256"] = "not-a-sha256"
    with pytest.raises(ValidationError, match="dataset_manifest_sha256"):
        EvaluationReport.model_validate(payload)


def test_evaluation_report_rejects_false_release_and_gate_claims():
    records = [record for record in generate_records(1, 1, 1) if record.split == "test"]
    payload = evaluate(FixtureEngine("adapted"), records).model_dump(mode="json")
    payload["release_passed"] = True
    payload["gates"] = {}
    with pytest.raises(ValidationError, match="lacks required release gates"):
        EvaluationReport.model_validate(payload)

    payload = evaluate(FixtureEngine("adapted"), records).model_dump(mode="json")
    gate_name = next(iter(payload["gates"]))
    payload["gates"][gate_name]["passed"] = not payload["gates"][gate_name]["passed"]
    with pytest.raises(ValidationError, match="pass result is inconsistent"):
        EvaluationReport.model_validate(payload)


def test_evaluation_report_rejects_non_finite_metrics():
    records = [record for record in generate_records(1, 1, 1) if record.split == "test"]
    payload = evaluate(FixtureEngine("adapted"), records).model_dump(mode="json")
    payload["metrics"]["event_type_accuracy"] = float("nan")
    with pytest.raises(ValidationError, match="event_type_accuracy"):
        EvaluationReport.model_validate(payload)


def test_evaluation_report_rejects_tampered_case_derived_evidence():
    record = next(record for record in generate_records(1, 1, 1) if record.split == "test")
    original = evaluate(FixtureEngine("adapted"), [record]).model_dump(mode="json")

    payload = {**original, "cases": [dict(original["cases"][0])]}
    payload["cases"][0]["passed"] = not payload["cases"][0]["passed"]
    with pytest.raises(ValidationError, match="passed result"):
        EvaluationReport.model_validate(payload)

    payload = {**original, "metrics": dict(original["metrics"])}
    payload["metrics"]["event_type_accuracy"] = 1.0 - payload["metrics"]["event_type_accuracy"]
    with pytest.raises(ValidationError, match="metric does not match cases"):
        EvaluationReport.model_validate(payload)

    payload = {**original, "latency_ms": dict(original["latency_ms"])}
    payload["latency_ms"]["mean"] += 1
    with pytest.raises(ValidationError, match="latency_ms"):
        EvaluationReport.model_validate(payload)


def test_evaluation_rejects_mixed_inference_identity_before_caching():
    records = [record for record in generate_records(1, 1, 1) if record.split == "test"][:2]

    class DriftingEngine(FixtureEngine):
        calls = 0

        def generate(self, request):
            result = super().generate(request)
            self.calls += 1
            if self.calls == 2:
                result.metadata = result.metadata.model_copy(
                    update={"model_version": "different-model-version"}
                )
            return result

    newly_cached: list[str] = []

    with pytest.raises(ValueError, match="does not match configured engine.*model_version"):
        evaluate(
            DriftingEngine("adapted"),
            records,
            result_callback=lambda record, _result: newly_cached.append(record.record_id),
        )
    assert newly_cached == [records[0].record_id]


def test_evaluation_rejects_consistently_misreported_engine_identity():
    records = [record for record in generate_records(1, 1, 1) if record.split == "test"][:1]

    class MisreportingEngine(FixtureEngine):
        def generate(self, request):
            result = super().generate(request)
            result.metadata = result.metadata.model_copy(update={"model_id": "wrong-model"})
            return result

    with pytest.raises(ValueError, match="does not match configured engine.*model_id"):
        evaluate(MisreportingEngine("adapted"), records)


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
