import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from modelforge_tw.io_utils import (
    atomic_write_text,
    exclusive_file_lock,
    read_utf8_text_limited,
)


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


def test_exclusive_file_lock_serializes_parallel_writers(tmp_path):
    lock_path = tmp_path / ".registry.lock"
    state_guard = threading.Lock()
    active = 0
    maximum_active = 0

    def enter_critical_section() -> None:
        nonlocal active, maximum_active
        with exclusive_file_lock(lock_path):
            with state_guard:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.01)
            with state_guard:
                active -= 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: enter_critical_section(), range(8)))

    assert maximum_active == 1


def test_exclusive_file_lock_rejects_negative_timeout(tmp_path):
    with pytest.raises(ValueError, match="cannot be negative"):
        with exclusive_file_lock(tmp_path / ".registry.lock", timeout_seconds=-1):
            pass


def test_limited_text_reader_counts_bytes_and_accepts_exact_boundary(tmp_path):
    target = tmp_path / "control.json"
    target.write_bytes("台".encode())

    assert read_utf8_text_limited(target, max_bytes=3) == "台"
    with pytest.raises(ValueError, match="2-byte limit"):
        read_utf8_text_limited(target, max_bytes=2)


def test_limited_text_reader_rejects_invalid_utf8_and_invalid_limit(tmp_path):
    target = tmp_path / "control.json"
    target.write_bytes(b"\xff")

    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_utf8_text_limited(target)
    with pytest.raises(ValueError, match="must be positive"):
        read_utf8_text_limited(target, max_bytes=0)
