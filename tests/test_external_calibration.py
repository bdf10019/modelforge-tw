import csv
import hashlib
from pathlib import Path

import pytest

import modelforge_tw.external_calibration as external_calibration_module
from modelforge_tw.dataset import audit_records, read_jsonl
from modelforge_tw.external_calibration import build_external_calibration

HEADERS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def test_external_source_snapshot_is_bounded_and_utf8(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_bytes(b"x" * 33)
    monkeypatch.setattr(external_calibration_module, "MAX_EXTERNAL_CSV_BYTES", 32)
    with pytest.raises(ValueError, match="16 MiB limit"):
        external_calibration_module._read_source_snapshot(source)

    source.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="valid UTF-8"):
        external_calibration_module._read_source_snapshot(source)


def _row(
    udi: int,
    *,
    air: float = 300.0,
    process: float = 310.0,
    speed: int = 1600,
    torque: float = 40.0,
    wear: int = 100,
    failure: int = 0,
    twf: int = 0,
    hdf: int = 0,
    pwf: int = 0,
    osf: int = 0,
) -> dict[str, object]:
    return {
        "UDI": udi,
        "Product ID": f"L{udi:05d}",
        "Type": "L",
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Machine failure": failure,
        "TWF": twf,
        "HDF": hdf,
        "PWF": pwf,
        "OSF": osf,
        "RNF": 0,
    }


def _write_fixture(path: Path) -> None:
    base_rows = [
        _row(1, process=308.0, speed=1300, failure=1, hdf=1),
        _row(2, speed=1000, torque=20.0, failure=1, pwf=1),
        _row(3, wear=220, failure=1, twf=1),
        _row(4, wear=220, torque=60.0, failure=1, osf=1),
        _row(5, process=309.0, speed=1400),
        _row(6, speed=1000, torque=35.3),
        _row(7, wear=205),
        _row(8, wear=190, torque=55.0),
        _row(9, process=309.5, speed=1500),
        _row(10, speed=1000, torque=40.0),
        _row(11, wear=185),
        _row(12, wear=170, torque=55.0),
        _row(13),
    ]
    rows = [
        {
            **row,
            "UDI": int(row["UDI"]) + offset,
            "Product ID": f"L{int(row['UDI']) + offset:05d}",
        }
        for offset in (0, 100, 200, 300)
        for row in base_rows
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def test_external_calibration_is_frozen_audited_and_separate(tmp_path, monkeypatch):
    source = tmp_path / "ai4i.csv"
    _write_fixture(source)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {
        "transform_version": "ai4i-calibration-v2",
        "source": {
            "dataset": "AI4I fixture",
            "landing_page": "https://example.invalid/ai4i",
            "doi": "fixture",
            "license": "CC-BY-4.0",
            "kind": "third-party-synthetic",
            "expected_rows": 52,
            "csv_sha256": source_sha256,
        },
        "selection": {
            "seed": "fixture-seed",
            "holdout_seed": "fixture-holdout-seed",
            "items_per_event_severity": 1,
            "unknown_items": 1,
        },
    }
    output = tmp_path / "generated"
    source_path = source.resolve()
    source_reads = 0
    original_open = Path.open

    def observed_open(path, *args, **kwargs):
        nonlocal source_reads
        if path.resolve() == source_path:
            source_reads += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    manifest = build_external_calibration(source, output, config)
    records = read_jsonl(output / "external.jsonl")
    holdout_records = read_jsonl(output / "external_holdout.jsonl")

    assert len(records) == 13
    assert len(holdout_records) == 13
    assert all(record.split == "external" for record in records)
    assert all(record.split == "external_holdout" for record in holdout_records)
    assert manifest["output"]["records"] == 13
    assert manifest["holdout_output"]["records"] == 13
    calibration_rows = {
        udi for values in manifest["selection"]["selected_source_rows"].values() for udi in values
    }
    holdout_rows = {
        udi
        for values in manifest["holdout_selection"]["selected_source_rows"].values()
        for udi in values
    }
    assert calibration_rows.isdisjoint(holdout_rows)
    assert manifest["source_row_overlap"] == 0
    assert manifest["externality"]["not_used_for_training"] is True
    assert manifest["externality"]["holdout_not_used_for_prompt_tuning"] is True
    assert manifest["externality"]["real_factory_measurements"] is False
    assert audit_records(records + holdout_records)["passed"] is True
    assert source_reads == 1
