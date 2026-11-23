"""A file whose bytes change right after its first read, for the tests that a command parses
the bytes its hash check read, or refuses.

`swapped_after_first_read(path, new)`: while inside, the first open of `path` for reading through
`open` or `io.open`, with or without an `opener` (which `Path.open`, `Path.read_bytes`,
`Path.read_text` and the run recorder's `sha256_file` call) reads the file's bytes as they are,
in full, and hands them to the caller; the file then holds `new`, written in place (the same
file on disk), so every later read gets `new`. A hash check that reads the file first therefore
passes, and a parse that reads the file again gets other bytes. On leaving, the file holds its
first bytes again, so a fixture shared by several tests is left as it was. The returned list
holds one entry per swap made (one, or none when the file was never read). `grown(path)` is the
file's bytes with one more line ending, bytes every reader of these files parses as before, so
a parse of them is no failure by itself.

The refusals expected: `table_refusal(path)`, a truth table changed after its check
(`days_from_truth.ReadFiles`); `file_refusal(path)`, a recorded run's file changed after its
check (`run_verify.still_verified`), as a command that reads a truth summary prints it.
"""

import builtins
import io
import os
from contextlib import contextmanager
from pathlib import Path

CHANGED = (
    "the file changed after it was read and checked (its SHA-256 is no longer that of the bytes "
    "parsed); nothing parsed from it is used"
)


def grown(path) -> bytes:
    return Path(path).read_bytes() + b"\n"


def table_refusal(path) -> str:
    return (
        f"COULD NOT RUN: {path} changed after its SHA-256 was checked against its build (its "
        "size, times or identity on disk are not those of the file read); nothing parsed from "
        "it is used\n"
    )


def file_refusal(path) -> str:
    return f"COULD NOT RUN: {path}: {CHANGED}\n"


@contextmanager
def swapped_after_first_read(path, new: bytes):
    target = os.path.realpath(path)
    first = Path(target).read_bytes()
    assert first != new, "the swapped bytes must differ from the file's"
    real_open = builtins.open
    swaps: list[str] = []

    def hooked(
        file,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ):
        reading = "r" in mode and "+" not in mode
        named = isinstance(file, (str, bytes, os.PathLike))
        if not swaps and reading and named and os.path.realpath(file) == target:
            with real_open(target, "rb") as handle:
                found = handle.read()
            with real_open(target, "wb") as handle:
                handle.write(new)
            swaps.append(target)
            if "b" in mode:
                return io.BytesIO(found)
            return io.TextIOWrapper(
                io.BytesIO(found), encoding=encoding, errors=errors, newline=newline
            )
        return real_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    saved = builtins.open, io.open
    builtins.open = io.open = hooked
    try:
        yield swaps
    finally:
        builtins.open, io.open = saved
        Path(target).write_bytes(first)
