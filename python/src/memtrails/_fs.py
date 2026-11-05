"""Writes that land whole or raise: ``os.write`` may write fewer bytes than it was given (a full disk, a signal), and a
short write left unnoticed tears a line, or leaves the offsets an index holds pointing past the file's end."""

from __future__ import annotations

import errno
import os


def write_all(fd: int, data: bytes) -> None:
    """Every byte of ``data`` to ``fd``, or OSError."""
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError(errno.EIO, "nothing was written")
        view = view[n:]


def append_all(path: str, data: bytes) -> None:
    """``data`` appended to ``path`` (made private when it is new), whole — or OSError, and the file cut back to where
    it ended before, so what is appended next starts on a line of its own."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        start = os.fstat(fd).st_size
        try:
            write_all(fd, data)
        except OSError:
            try:
                os.ftruncate(fd, start)
            except OSError:
                pass  # what was cut short stays: a load ends or drops a torn last line
            raise
    finally:
        os.close(fd)


def write_file(path: str, data: bytes) -> None:
    """``path`` replaced by ``data`` (private), whole, or OSError."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        write_all(fd, data)
    finally:
        os.close(fd)
