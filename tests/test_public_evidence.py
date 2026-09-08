import json
from pathlib import Path
from uuid import UUID

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_evidence_snapshot_is_self_consistent():
    evidence = json.loads(
        (PROJECT_ROOT / "docs" / "measured-evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["evidence_kind"] == "measured"
    assert "cases" not in evidence
    assert evidence["dataset"]["records"] == sum(evidence["dataset"]["splits"].values())
    assert all(value == 0 for value in evidence["dataset"]["audit"].values())

    nf4 = evidence["evaluations"]["qlora_nf4"]
    q8 = evidence["evaluations"]["gguf_q8_policy_v2"]
    q4 = evidence["evaluations"]["gguf_q4_policy_v2"]
    assert nf4["release_passed"] is True
    assert nf4["max_new_tokens"] == 192
    assert nf4["structured_output_contract"] == "prompt-schema+bounded-repair-v1"
    for field in (
        "safety_policy_version",
        "evaluation_contract_version",
        "release_gate_sha256",
    ):
        assert nf4[field] == q8[field]
    for field in (
        "prompt_version",
        "max_new_tokens",
        "structured_output_contract",
        "safety_policy_version",
        "evaluation_contract_version",
        "release_gate_sha256",
    ):
        assert q8[field] == q4[field]
    assert q8["release_passed"] is True
    assert q4["release_passed"] is False
    assert evidence["registry_expected_state"] == {
        "released": "qwen3-1.7b-qlora-v5-policy-v2",
        "previous_good": "qwen3-1.7b-gguf-q8-v5-policy-v2",
        "blocked": "qwen3-1.7b-gguf-q4-v5-policy-v2",
    }
    assert evidence["registry_source_report"] == "outputs/model-registry.json"
    assert evidence["training"]["source_report"].endswith("training-report.json")
    assert all(artifact["source_report"] for artifact in evidence["artifacts"].values())
    assert all(
        benchmark["source_report"] and benchmark["requests"] > 0 and benchmark["errors"] == 0
        for benchmark in evidence["concurrency_benchmarks"].values()
    )

    comparison = evidence["q8_to_q4_comparison"]
    assert all(comparison["compatibility_checks"].values())
    assert comparison["quantization_gate_passed"] is False
    event_delta = q4["metrics"]["event_type_macro_f1"] - q8["metrics"]["event_type_macro_f1"]
    assert comparison["metric_deltas_candidate_minus_baseline"][
        "event_type_macro_f1"
    ] == pytest.approx(event_delta, abs=1e-6)
    assert -event_delta > comparison["event_macro_f1_max_regression"]


def test_external_evidence_snapshot_preserves_holdout_and_rejection_decision():
    evidence = json.loads(
        (PROJECT_ROOT / "docs" / "measured-evidence.json").read_text(encoding="utf-8")
    )
    external = evidence["external_calibration"]
    assert external["source"]["kind"] == "third-party-synthetic"
    assert external["source"]["license"] == "CC-BY-4.0"
    assert external["calibration"]["records"] == 98
    assert external["holdout"]["records"] == 98
    assert (
        external["calibration"]["source_rows_sha256"] != external["holdout"]["source_rows_sha256"]
    )
    assert external["holdout"]["used_for_prompt_tuning"] is False
    assert external["audit"]["source_row_overlap"] == 0
    assert all(value == 0 for key, value in external["audit"].items() if key not in {"records"})

    runs = external["evaluations"]
    contract_fields = {
        "model_id": "Qwen3-1.7B",
        "model_version": "qwen3-1.7b-qlora-v5",
        "quantization": "q8_0",
        "max_new_tokens": 192,
        "context_window_tokens": 2048,
        "structured_output_contract": "json-schema-v2",
        "safety_policy_version": "risk-precedence-v3",
        "evaluation_contract_version": "release-metrics-v4",
        "release_gate_sha256": "2a5b42d1926d457cc54cb48c62fd8514b75f7dd39e0673b686eac6370ea5fc7a",
        "evidence_kind": "measured",
    }
    assert all(
        run[field] == value for run in runs.values() for field, value in contract_fields.items()
    )
    internal_v2 = runs["internal_q8_prompt_v2_policy_v3"]
    external_v2 = runs["external_calibration_q8_prompt_v2"]
    calibration_v5 = runs["external_calibration_q8_prompt_v5"]
    holdout_v5 = runs["external_holdout_q8_prompt_v5"]
    regression_v5 = runs["internal_regression_q8_prompt_v5"]
    external_v6 = runs["external_calibration_q8_prompt_v6"]
    assert internal_v2["release_passed"] is True
    assert external_v2["release_passed"] is False
    assert calibration_v5["release_passed"] is True
    assert holdout_v5["release_passed"] is True
    assert regression_v5["release_passed"] is False
    assert external_v6["release_passed"] is False
    assert holdout_v5["metrics"]["raw_evidence_span_precision"] < 1.0
    warning_total = sum(holdout_v5["validation_warning_counts"].values())
    assert holdout_v5["metrics"]["postprocessing_intervention_rate"] == pytest.approx(
        warning_total / holdout_v5["cases"], abs=1e-6
    )
    assert external["decision"] == ("holdout_passed_but_prompt_v5_rejected_for_internal_regression")


def test_interview_walkthrough_preserves_claim_boundaries():
    walkthrough = (PROJECT_ROOT / "docs" / "INTERVIEW_WALKTHROUGH.md").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "INTERVIEW_WALKTHROUGH.md" in readme
    assert "evidence_kind=fixture" in walkthrough
    assert "16.33%" in walkthrough
    assert "內部 224 筆時 event F1 降到 69.72%" in walkthrough
    assert "不宣稱叢集部署實績" in walkthrough
    assert "request／model-response bytes" in walkthrough
    assert "現有 1.0 實測 registry" in walkthrough
    assert "外部不可變稽核" in walkthrough


def test_current_context_benchmarks_share_workload_and_complete_token_coverage():
    evidence = json.loads(
        (PROJECT_ROOT / "docs" / "current-context-benchmarks.json").read_text(encoding="utf-8")
    )
    server = evidence["server_config"]
    assert server["total_context_tokens"] // server["parallel_slots"] == 2048
    assert server["parallel_slots"] == 4
    assert sum(evidence["selected_case_counts"].values()) == 100
    reports = [entry["report"] for entry in evidence["measurements"].values()]
    baseline = reports[0]
    assert len({report["run_id"] for report in reports}) == len(reports) == 2
    assert {report["quantization"] for report in reports} == {"q8_0", "q4_k_m"}
    for report in reports:
        assert UUID(report["run_id"]).version == 4
        for field in (
            "model_revision",
            "prompt_version",
            "max_new_tokens",
            "context_window_tokens",
            "structured_output_contract",
            "safety_policy_version",
            "warmup_requests",
            "workload",
        ):
            assert report[field] == baseline[field]
        assert report["schema_version"] == "1.4"
        assert report["passed"] is True
        assert report["evidence_kind"] == "measured"
        assert report["context_window_tokens"] == 2048
        assert [level["concurrency"] for level in report["levels"]] == [1, 2, 4, 8]
        for level, reference in zip(report["levels"], baseline["levels"], strict=True):
            assert level["input_sequence_sha256"] == reference["input_sequence_sha256"]
            assert level["requests"] == level["completed"] == 100
            assert level["errors"] == 0
            assert level["prompt_token_sample_count"] == 100
            assert level["output_token_sample_count"] == 100
            assert level["ttft_sample_count"] == 100
            assert level["prompt_tokens"] == reference["prompt_tokens"]
            assert level["requests_per_second"] == pytest.approx(
                level["completed"] / level["wall_seconds"], abs=0.000051
            )
            assert level["output_tokens_per_second"] == pytest.approx(
                level["output_tokens"] / level["wall_seconds"], abs=0.00006
            )
