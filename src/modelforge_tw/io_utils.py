from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from time import monotonic, sleep
from typing import BinaryIO, Literal, NamedTuple, Protocol, cast

MAX_CONTROL_FILE_BYTES = 16 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 1024 * 1024


class LimitedTextLine(NamedTuple):
    number: int
    text: str | None
    error: Literal["too_large", "invalid_utf8"] | None


class _FcntlModule(Protocol):
    LOCK_EX: int
    LOCK_NB: int
    LOCK_UN: int

    def flock(self, file_descriptor: int, operation: int) -> None: ...


class _MsvcrtModule(Protocol):
    LK_NBLCK: int
    LK_UNLCK: int

    def locking(self, file_descriptor: int, mode: int, nbytes: int) -> None: ...


def _try_lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        msvcrt = cast(_MsvcrtModule, import_module("msvcrt"))
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    fcntl = cast(_FcntlModule, import_module("fcntl"))
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        msvcrt = cast(_MsvcrtModule, import_module("msvcrt"))
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl = cast(_FcntlModule, import_module("fcntl"))
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_file_lock(path: Path, *, timeout_seconds: float = 10.0) -> Iterator[None]:
    """Hold a cross-process advisory lock backed by a stable lock file."""
    if timeout_seconds < 0:
        raise ValueError("lock timeout cannot be negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        deadline = monotonic() + timeout_seconds
        while True:
            try:
                _try_lock(handle)
                break
            except OSError as error:
                if monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for file lock: {path.name}") from error
                sleep(0.05)
        try:
            yield
        finally:
            _unlock(handle)


def atomic_write_text(
    path: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> None:
    """Replace a text artifact only after its complete content reaches disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding=encoding, newline=newline) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_parent_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_utf8_text_limited(
    path: Path,
    *,
    max_bytes: int = MAX_CONTROL_FILE_BYTES,
) -> str:
    """Read a control-plane text file without allowing unbounded allocation."""
    if max_bytes <= 0:
        raise ValueError("file size limit must be positive")
    with path.open("rb") as handle:
        content = handle.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"file exceeds {max_bytes}-byte limit: {path.name}")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"file is not valid UTF-8: {path.name}") from error


def iter_utf8_lines_limited(
    path: Path,
    *,
    max_line_bytes: int = MAX_JSONL_LINE_BYTES,
) -> Iterator[LimitedTextLine]:
    """Yield bounded UTF-8 lines and recover after an invalid physical line."""
    if max_line_bytes <= 0:
        raise ValueError("line size limit must be positive")
    with path.open("rb") as handle:
        line_number = 0
        while raw := handle.readline(max_line_bytes + 1):
            line_number += 1
            if len(raw) > max_line_bytes:
                while raw and not raw.endswith(b"\n"):
                    raw = handle.readline(max_line_bytes + 1)
                yield LimitedTextLine(line_number, None, "too_large")
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                yield LimitedTextLine(line_number, None, "invalid_utf8")
                continue
            yield LimitedTextLine(line_number, text, None)
