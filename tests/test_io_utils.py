import os

import pytest

from modelforge_tw.io_utils import atomic_write_text


def test_atomic_write_replaces_complete_content_without_temp_files(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("old", encoding="utf-8")

    atomic_write_text(target, "new\n")

    assert target.read_text(encoding="utf-8") == "new\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_preserves_previous_file_when_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    target.write_text("previous-good", encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_text(target, "incomplete replacement")

    assert target.read_text(encoding="utf-8") == "previous-good"
    assert list(tmp_path.glob("*.tmp")) == []
