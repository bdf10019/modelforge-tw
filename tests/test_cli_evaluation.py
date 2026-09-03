import hashlib
import json
from argparse import Namespace
from pathlib import Path

from modelforge_tw import cli
from modelforge_tw.dataset import generate_records, write_dataset
from modelforge_tw.engines import FixtureEngine
from modelforge_tw.evaluation import evaluate


def test_evaluate_hashes_and_applies_release_gate_from_one_read(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    write_dataset(data_dir, train_per_type=1, validation_per_type=1, test_per_type=1)
    gate_path = cli.PROJECT_ROOT / "configs" / "release-gates.json"
    expected_hash = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    output_path = tmp_path / "evaluation.json"
    args = cli.build_parser().parse_args(
        [
            "evaluate",
            "--data",
            str(data_dir),
            "--splits",
            "test",
            "--limit",
            "1",
            "--engine",
            "fixture-adapted",
            "--gates",
            str(gate_path),
            "--output",
            str(output_path),
            "--no-resume-cache",
        ]
    )
    gate_reads = 0
    original_open = Path.open

    def observed_open(path, *open_args, **open_kwargs):
        nonlocal gate_reads
        if path.resolve() == gate_path.resolve():
            gate_reads += 1
        return original_open(path, *open_args, **open_kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    assert cli.command_evaluate(args) == 0

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["release_gate_sha256"] == expected_hash
    assert gate_reads == 1


def test_evaluation_cache_skips_truncated_tail_and_keeps_valid_result(tmp_path):
    record = next(item for item in generate_records(1, 1, 1) if item.split == "test")
    result = FixtureEngine("adapted").generate(record.input)
    fingerprint = "fixture-fingerprint"
    record_hash = cli._record_sha256(record)
    valid_line = json.dumps(
        {
            "fingerprint": fingerprint,
            "record_id": record.record_id,
            "record_sha256": record_hash,
            "result": result.model_dump(mode="json"),
        }
    )
    cache_path = tmp_path / "evaluation.partial.jsonl"
    cache_path.write_text(valid_line + "\n" + '{"fingerprint":', encoding="utf-8")

    cached, rejected = cli._load_evaluation_cache(
        cache_path,
        fingerprint=fingerprint,
        record_hashes={record.record_id: record_hash},
    )

    assert set(cached) == {record.record_id}
    assert rejected == 1
    assert cli._cache_needs_line_boundary(cache_path) is True


def test_evaluation_cache_skips_oversized_line_and_recovers(tmp_path, monkeypatch):
    cache_path = tmp_path / "evaluation.partial.jsonl"
    cache_path.write_text(("x" * 33) + "\n{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "MAX_JSONL_LINE_BYTES", 32)

    cached, rejected = cli._load_evaluation_cache(
        cache_path,
        fingerprint="expected",
        record_hashes={},
    )

    assert cached == {}
    assert rejected == 1


def test_evaluation_cache_line_boundary_separates_future_entries(tmp_path):
    cache_path = tmp_path / "evaluation.partial.jsonl"
    cache_path.write_text('{"truncated":', encoding="utf-8")

    with cache_path.open("a", encoding="utf-8", newline="\n") as cache_writer:
        if cli._cache_needs_line_boundary(cache_path):
            cache_writer.write("\n")
        cache_writer.write('{"next":"valid"}\n')

    assert cache_path.read_text(encoding="utf-8").splitlines() == [
        '{"truncated":',
        '{"next":"valid"}',
    ]


def test_compare_command_fails_when_inference_contracts_differ(tmp_path):
    records = [item for item in generate_records(2, 1, 2) if item.split == "test"]
    baseline = evaluate(FixtureEngine("adapted"), records)
    candidate = baseline.model_copy(update={"prompt_version": "minimal-v1"})
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "comparison.json"
    baseline_path.write_text(baseline.model_dump_json(), encoding="utf-8")
    candidate_path.write_text(candidate.model_dump_json(), encoding="utf-8")

    exit_code = cli.command_compare(
        Namespace(
            baseline=baseline_path,
            candidate=candidate_path,
            output=output_path,
            profile="metrics",
        )
    )

    comparison = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert comparison["compatible"] is False
    assert comparison["compatibility_checks"]["prompt"] is False
