import json
from pathlib import Path

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

    comparison = evidence["q8_to_q4_comparison"]
    assert all(comparison["compatibility_checks"].values())
    assert comparison["quantization_gate_passed"] is False
    event_delta = q4["metrics"]["event_type_macro_f1"] - q8["metrics"]["event_type_macro_f1"]
    assert comparison["metric_deltas_candidate_minus_baseline"][
        "event_type_macro_f1"
    ] == pytest.approx(event_delta, abs=1e-6)
    assert -event_delta > comparison["event_macro_f1_max_regression"]
