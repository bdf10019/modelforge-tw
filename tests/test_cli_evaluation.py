import json

from modelforge_tw import cli
from modelforge_tw.dataset import generate_records
from modelforge_tw.engines import FixtureEngine


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
