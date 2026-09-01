from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from time import monotonic, sleep
from typing import BinaryIO, Protocol, cast


class _FcntlModule(Protocol):
    LOCK_EX: int
    LOCK_NB: int
    LOCK_UN: int

    def flock(self, file_descriptor: int, operation: int) -> None: ...


def _try_lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    fcntl = cast(_FcntlModule, import_module("fcntl"))
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl = cast(_FcntlModule, import_module("fcntl"))
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_file_lock(path: Path, *, timeout_seconds: float = 10.0) -> Iterator[None]:
    """Hold a cross-process advisory lock backed by a stable one-byte file."""
    if timeout_seconds < 0:
        raise ValueError("lock timeout cannot be negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
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
    finally:
        temporary.unlink(missing_ok=True)
