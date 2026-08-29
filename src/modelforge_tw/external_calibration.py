from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from modelforge_tw.dataset import audit_records, sha256_file
from modelforge_tw.schemas import (
    DatasetRecord,
    EventType,
    SafeAction,
    Severity,
    Signal,
    TriageOutput,
    TriageRequest,
)

TRANSFORM_VERSION = "ai4i-calibration-v2"
CalibrationSplit = Literal["external", "external_holdout"]
EXPECTED_COLUMNS = {
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
}


@dataclass(frozen=True)
class AI4IRow:
    udi: int
    product_type: str
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: int
    torque_nm: float
    tool_wear_min: int
    machine_failure: int
    twf: int
    hdf: int
    pwf: int
    osf: int
    rnf: int

    @classmethod
    def from_csv(cls, item: dict[str, str]) -> AI4IRow:
        return cls(
            udi=int(item["UDI"]),
            product_type=item["Type"],
            air_temperature_k=float(item["Air temperature [K]"]),
            process_temperature_k=float(item["Process temperature [K]"]),
            rotational_speed_rpm=int(item["Rotational speed [rpm]"]),
            torque_nm=float(item["Torque [Nm]"]),
            tool_wear_min=int(item["Tool wear [min]"]),
            machine_failure=int(item["Machine failure"]),
            twf=int(item["TWF"]),
            hdf=int(item["HDF"]),
            pwf=int(item["PWF"]),
            osf=int(item["OSF"]),
            rnf=int(item["RNF"]),
        )

    @property
    def temperature_delta_k(self) -> float:
        return self.process_temperature_k - self.air_temperature_k

    @property
    def power_w(self) -> float:
        return self.torque_nm * self.rotational_speed_rpm * 2 * math.pi / 60

    @property
    def overstrain_ratio(self) -> float:
        thresholds = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
        return self.tool_wear_min * self.torque_nm / thresholds[self.product_type]

    @property
    def failure_mode_count(self) -> int:
        return self.twf + self.hdf + self.pwf + self.osf + self.rnf


@dataclass(frozen=True)
class Cohort:
    name: str
    event_type: EventType
    severity: Severity


COHORTS = tuple(
    Cohort(f"{event.value}-{severity.value}", event, severity)
    for severity in (Severity.HIGH, Severity.MEDIUM, Severity.LOW)
    for event in (
        EventType.THERMAL,
        EventType.ELECTRICAL,
        EventType.MECHANICAL,
        EventType.PROCESS,
    )
)


def _is_normal(row: AI4IRow) -> bool:
    return row.machine_failure == 0 and row.failure_mode_count == 0


def _matches(row: AI4IRow, cohort: Cohort) -> bool:
    if cohort.severity == Severity.HIGH:
        flags = {
            EventType.THERMAL: row.hdf,
            EventType.ELECTRICAL: row.pwf,
            EventType.MECHANICAL: row.twf,
            EventType.PROCESS: row.osf,
        }
        return row.failure_mode_count == 1 and flags[cohort.event_type] == 1
    if not _is_normal(row):
        return False
    if cohort.event_type == EventType.THERMAL:
        if cohort.severity == Severity.MEDIUM:
            return 8.6 <= row.temperature_delta_k < 9.2 and row.rotational_speed_rpm < 1450
        return 9.2 <= row.temperature_delta_k < 9.8 and row.rotational_speed_rpm < 1550
    if cohort.event_type == EventType.ELECTRICAL:
        if cohort.severity == Severity.MEDIUM:
            return 3500 <= row.power_w < 4000 or 8200 < row.power_w <= 9000
        return 4000 <= row.power_w < 4500 or 7500 < row.power_w <= 8200
    if cohort.event_type == EventType.MECHANICAL:
        if cohort.severity == Severity.MEDIUM:
            return 200 <= row.tool_wear_min <= 240
        return 180 <= row.tool_wear_min < 200
    if cohort.event_type == EventType.PROCESS:
        if cohort.severity == Severity.MEDIUM:
            return 0.90 <= row.overstrain_ratio <= 1.0
        return 0.80 <= row.overstrain_ratio < 0.90
    return False


def _rank(row: AI4IRow, cohort_name: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{cohort_name}|{row.udi}".encode()).hexdigest()


def _signal_status(severity: Severity) -> Literal["normal", "warning", "unknown"]:
    if severity in (Severity.HIGH, Severity.MEDIUM):
        return "warning"
    if severity == Severity.LOW:
        return "normal"
    return "unknown"


def _signals(row: AI4IRow, severity: Severity, event_type: EventType) -> list[Signal]:
    relevant_status = _signal_status(severity)
    default_status: Literal["normal", "unknown"] = (
        "unknown" if severity == Severity.UNKNOWN else "normal"
    )
    signals = [
        Signal(
            name="air_temperature",
            value=row.air_temperature_k,
            unit="K",
            status=default_status,
        ),
        Signal(
            name="process_temperature",
            value=row.process_temperature_k,
            unit="K",
            status=relevant_status if event_type == EventType.THERMAL else default_status,
        ),
        Signal(
            name="rotational_speed",
            value=row.rotational_speed_rpm,
            unit="rpm",
            status=default_status,
        ),
        Signal(name="torque", value=row.torque_nm, unit="Nm", status=default_status),
        Signal(
            name="tool_wear",
            value=row.tool_wear_min,
            unit="min",
            status=(relevant_status if event_type == EventType.MECHANICAL else default_status),
        ),
    ]
    if event_type == EventType.THERMAL:
        signals.append(
            Signal(
                name="temperature_delta",
                value=round(row.temperature_delta_k, 1),
                unit="K",
                status=relevant_status,
            )
        )
    elif event_type == EventType.ELECTRICAL:
        signals.append(
            Signal(
                name="calculated_power",
                value=round(row.power_w),
                unit="W",
                status=relevant_status,
            )
        )
    elif event_type == EventType.PROCESS:
        signals.append(
            Signal(
                name="overstrain_ratio",
                value=round(row.overstrain_ratio, 3),
                unit="ratio",
                status=relevant_status,
            )
        )
    return signals


def _evidence(row: AI4IRow, cohort: Cohort) -> str:
    if cohort.event_type == EventType.THERMAL:
        detail = (
            f"製程與環境溫差 {row.temperature_delta_k:.1f} K，轉速 {row.rotational_speed_rpm} rpm"
        )
        label = "散熱"
    elif cohort.event_type == EventType.ELECTRICAL:
        detail = f"計算功率 {row.power_w:.0f} W"
        label = "功率"
    elif cohort.event_type == EventType.MECHANICAL:
        detail = f"刀具累積使用 {row.tool_wear_min} 分鐘"
        label = "磨耗"
    else:
        detail = (
            f"過應力指標為 {row.overstrain_ratio * 100:.1f}%"
            f"（{row.tool_wear_min} min × {row.torque_nm:.1f} Nm）"
        )
        label = "過應力"
    if cohort.severity == Severity.HIGH:
        return f"{label}失效旗標已觸發，{detail}，讀值已越過預警值"
    if cohort.severity == Severity.MEDIUM:
        return f"{detail}，處於{label}預警附近但失效旗標為 0"
    return f"本次單點{detail}，仍在{label}正常範圍且失效旗標為 0"


def _to_record(
    row: AI4IRow,
    cohort: Cohort,
    split: CalibrationSplit,
) -> DatasetRecord:
    evidence = _evidence(row, cohort)
    if cohort.severity == Severity.HIGH:
        suffix = "，需要隔離風險並由人員確認。"
        safe_actions = [SafeAction.ISOLATE_AND_ESCALATE]
        review_required = True
        reason_codes = ["HIGH_SEVERITY"]
    elif cohort.severity == Severity.MEDIUM:
        suffix = "，目前可維持運轉但需要安排追蹤。"
        safe_actions = [SafeAction.OBSERVE]
        review_required = False
        reason_codes = ["MEDIUM_SEVERITY"]
    else:
        suffix = "，屬單次輕微偏差，先保留紀錄。"
        safe_actions = [SafeAction.OBSERVE]
        review_required = False
        reason_codes = ["LOW_SEVERITY"]
    text = f"外部校準資料列 {row.udi}：{evidence}{suffix}"
    return DatasetRecord(
        record_id=f"{split}-ai4i-{row.udi:05d}-{cohort.name}",
        split=split,
        scenario_family_id=f"external-ai4i-row-{row.udi:05d}",
        template_id=f"{split}-ai4i-{cohort.name}-{TRANSFORM_VERSION}",
        asset_family="AI4I synthetic milling machine",
        generator_version=TRANSFORM_VERSION,
        input=TriageRequest(
            incident_text=text,
            asset_type="數控加工設備",
            signals=_signals(row, cohort.severity, cohort.event_type),
        ),
        target=TriageOutput(
            event_type=cohort.event_type,
            severity=cohort.severity,
            summary_zh_tw=f"數控加工設備出現{cohort.event_type.value}類校準事件。",
            evidence_spans=[evidence],
            missing_information=[],
            safe_actions=safe_actions,
            review_required=review_required,
            reason_codes=reason_codes,
        ),
    )


def _unknown_record(row: AI4IRow, split: CalibrationSplit) -> DatasetRecord:
    text = (
        f"外部校準資料列 {row.udi} 只有單點讀值：環境溫度 {row.air_temperature_k:.1f} K、"
        f"製程溫度 {row.process_temperature_k:.1f} K、轉速 {row.rotational_speed_rpm} rpm、"
        f"扭矩 {row.torque_nm:.1f} Nm、刀具使用 {row.tool_wear_min} 分鐘；"
        "未提供正常基準、警戒門檻、趨勢或異常現象，無法判定事件類型。"
    )
    return DatasetRecord(
        record_id=f"{split}-ai4i-{row.udi:05d}-unknown",
        split=split,
        scenario_family_id=f"external-ai4i-row-{row.udi:05d}",
        template_id=f"{split}-ai4i-unknown-{TRANSFORM_VERSION}",
        asset_family="AI4I synthetic milling machine",
        generator_version=TRANSFORM_VERSION,
        input=TriageRequest(
            incident_text=text,
            asset_type="數控加工設備",
            signals=_signals(row, Severity.UNKNOWN, EventType.UNKNOWN),
        ),
        target=TriageOutput(
            event_type=EventType.UNKNOWN,
            severity=Severity.UNKNOWN,
            summary_zh_tw="只有單點設備讀值，缺少基準、門檻與異常現象。",
            evidence_spans=[],
            missing_information=["正常基準", "警戒門檻", "趨勢或異常現象"],
            safe_actions=[SafeAction.COLLECT_EVIDENCE],
            review_required=True,
            reason_codes=["INSUFFICIENT_EVIDENCE"],
        ),
    )


def _read_rows(source_csv: Path) -> tuple[list[AI4IRow], set[str]]:
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        rows = [AI4IRow.from_csv(item) for item in reader]
    return rows, columns


def _select_records(
    rows: list[AI4IRow],
    *,
    seed: str,
    cohort_size: int,
    unknown_size: int,
    split: CalibrationSplit,
    used: set[int],
) -> tuple[list[DatasetRecord], dict[str, list[int]], Counter[str]]:
    records: list[DatasetRecord] = []
    cohort_counts: Counter[str] = Counter()
    selected_rows: dict[str, list[int]] = {}
    for cohort in COHORTS:
        candidates = [row for row in rows if row.udi not in used and _matches(row, cohort)]
        chosen = sorted(candidates, key=lambda row: _rank(row, cohort.name, seed))[:cohort_size]
        if len(chosen) != cohort_size:
            raise ValueError(
                f"external cohort {cohort.name} has {len(chosen)} rows; needs {cohort_size}"
            )
        selected_rows[cohort.name] = [row.udi for row in chosen]
        for row in chosen:
            used.add(row.udi)
            records.append(_to_record(row, cohort, split))
            cohort_counts[cohort.name] += 1

    unknown_candidates = [row for row in rows if row.udi not in used and _is_normal(row)]
    unknown = sorted(
        unknown_candidates,
        key=lambda row: _rank(row, "unknown", seed),
    )[:unknown_size]
    if len(unknown) != unknown_size:
        raise ValueError(f"external unknown cohort has {len(unknown)} rows; needs {unknown_size}")
    selected_rows["unknown"] = [row.udi for row in unknown]
    for row in unknown:
        used.add(row.udi)
        records.append(_unknown_record(row, split))
        cohort_counts["unknown"] += 1
    return records, selected_rows, cohort_counts


def _selection_manifest(
    seed: str,
    selected_rows: dict[str, list[int]],
    cohort_counts: Counter[str],
) -> dict[str, Any]:
    payload = json.dumps(selected_rows, sort_keys=True, separators=(",", ":"))
    return {
        "seed": seed,
        "selected_source_rows_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "selected_source_rows": selected_rows,
        "cohort_counts": dict(sorted(cohort_counts.items())),
    }


def _write_records(output_path: Path, records: list[DatasetRecord]) -> dict[str, Any]:
    output_path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "file": output_path.name,
        "records": len(records),
        "sha256": sha256_file(output_path),
    }


def build_external_calibration(
    source_csv: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    source = config["source"]
    selection = config["selection"]
    if config.get("transform_version") != TRANSFORM_VERSION:
        raise ValueError("external calibration transform version mismatch")
    actual_sha256 = sha256_file(source_csv)
    if actual_sha256 != source["csv_sha256"]:
        raise ValueError("external source CSV SHA-256 mismatch")
    rows, columns = _read_rows(source_csv)
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"external source columns changed: {sorted(columns)}")
    if len(rows) != int(source["expected_rows"]):
        raise ValueError("external source row count mismatch")

    seed = str(selection["seed"])
    holdout_seed = str(selection["holdout_seed"])
    if seed == holdout_seed:
        raise ValueError("external calibration and holdout seeds must differ")
    cohort_size = int(selection["items_per_event_severity"])
    unknown_size = int(selection["unknown_items"])
    used: set[int] = set()
    records, selected_rows, cohort_counts = _select_records(
        rows,
        seed=seed,
        cohort_size=cohort_size,
        unknown_size=unknown_size,
        split="external",
        used=used,
    )
    holdout_records, holdout_rows, holdout_counts = _select_records(
        rows,
        seed=holdout_seed,
        cohort_size=cohort_size,
        unknown_size=unknown_size,
        split="external_holdout",
        used=used,
    )

    audit = audit_records(records + holdout_records)
    if not audit["passed"]:
        raise ValueError(f"external calibration/holdout audit failed: {audit['errors']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "external.jsonl"
    holdout_path = output_dir / "external_holdout.jsonl"
    manifest = {
        "schema_version": "1.1",
        "transform_version": TRANSFORM_VERSION,
        "evidence_kind": "derived-from-third-party-synthetic",
        "source": {
            "dataset": source["dataset"],
            "landing_page": source["landing_page"],
            "doi": source["doi"],
            "license": source["license"],
            "kind": source["kind"],
            "csv_sha256": actual_sha256,
            "rows": len(rows),
        },
        "selection": _selection_manifest(seed, selected_rows, cohort_counts),
        "holdout_selection": _selection_manifest(
            holdout_seed,
            holdout_rows,
            holdout_counts,
        ),
        "output": _write_records(output_path, records),
        "holdout_output": _write_records(holdout_path, holdout_records),
        "source_row_overlap": 0,
        "audit": audit,
        "externality": {
            "source_independent_of_project": True,
            "not_used_for_training": True,
            "source_data_kind": "third-party-synthetic",
            "human_authored": False,
            "real_factory_measurements": False,
            "holdout_not_used_for_prompt_tuning": True,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
