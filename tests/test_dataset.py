from collections import Counter
from pathlib import Path

import pytest

import modelforge_tw.dataset as dataset_module
from modelforge_tw.dataset import (
    audit_records,
    generate_records,
    load_splits,
    read_jsonl,
    write_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_reader_rejects_oversized_physical_line(tmp_path, monkeypatch):
    path = tmp_path / "oversized.jsonl"
    path.write_bytes((b"x" * 33) + b"\n")
    monkeypatch.setattr(dataset_module, "MAX_JSONL_LINE_BYTES", 32)

    with pytest.raises(ValueError, match="too_large"):
        read_jsonl(path)


def test_curated_challenge_is_valid_and_evidence_is_grounded():
    records = read_jsonl(PROJECT_ROOT / "data" / "source" / "challenge.jsonl")
    assert len(records) == 14
    audit = audit_records(records)
    assert audit["passed"] is True
    assert audit["invalid_evidence_spans"] == 0


def test_generated_dataset_has_grouped_splits_and_manifest(tmp_path):
    manifest = write_dataset(
        tmp_path,
        challenge_source=PROJECT_ROOT / "data" / "source" / "challenge.jsonl",
        train_per_type=8,
        validation_per_type=3,
        test_per_type=4,
    )
    assert manifest["audit"]["passed"] is True
    assert manifest["audit"]["cross_split_template_ids"] == 0
    assert manifest["audit"]["cross_split_scenario_families"] == 0
    assert manifest["audit"]["cross_split_normalized_texts"] == 0
    assert manifest["audit"]["policy_severity_mismatches"] == 0
    assert manifest["generator_version"] == "1.3.0"
    assert manifest["files"]["challenge.jsonl"]["records"] == 14
    records = load_splits(tmp_path, ["train", "validation", "test", "challenge"])
    assert len(records) == 7 * (8 + 3 + 4) + 14


def test_default_training_severity_distribution_is_balanced():
    train = [record for record in generate_records() if record.split == "train"]
    severity_counts = Counter(record.target.severity.value for record in train)
    assert severity_counts == {
        "low": 220,
        "medium": 220,
        "high": 220,
        "critical": 60,
        "unknown": 120,
    }
