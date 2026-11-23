"""Download the public data files listed in `tools/data_sources.csv`, unparsed.

Run as `python tools/fetch_data.py [--root <folder>]` from the project's top folder.

With `--context` it downloads instead the public context files of `tools/context_sources.csv`
(archived weather forecasts and the OxCGRT index, inputs of the departure model of plan section
7, allowed by the owner on 15 November 2026), and nothing of `tools/data_sources.csv`. That table
has the columns `set,name,role,bytes,sha256,url`; every row has the role "context", which goes to
`data/raw/<set>/<name>` like every role other than "held-out", and is checked against its byte
size and its SHA-256, committed at its first download (these files have no published checksum:
a weather file is the response to a fixed query, so its first bytes are what is committed). An
empty `bytes` or `sha256` means none is committed yet, as for the other table.

Each listed file is streamed to a temporary file under `data/.incoming/` and moved to its
destination only after its byte size and its published MD5 (where the table gives them) match.
Destinations: a file whose role is "held-out" goes to the sealed folder named by
`fleetreserve.heldout_seal` (`<sealed folder>/<set>/<name>`); every other file goes to
`data/raw/<set>/<name>`. A file already at its destination is never overwritten: it is left as
it is if it matches the table, and reported as failed if it does not.

What it never does: decode, parse, preview or print the content of any file; request any URL
other than those in the table; send anything but a generic User-Agent header. It prints only file
names, destinations, byte sizes, SHA-256 and MD5 hashes and whether they matched.

Exit status: 0 every file is in place and matches; 1 a size or checksum did not match, or an
existing file differs from the table; 2 a file could not be fetched (network or HTTP error) or the
table could not be read.
"""

import argparse
import csv
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from fleetreserve import heldout_seal

TOP = Path(__file__).resolve().parents[1]
TABLE = Path(__file__).resolve().parent / "data_sources.csv"
ROLES = ("train", "development", "held-out", "document")
CONTEXT_TABLE = Path(__file__).resolve().parent / "context_sources.csv"
CONTEXT_ROLE = "context"
INCOMING = "data/.incoming"
USER_AGENT = "data-fetch (python urllib)"
TIMEOUT_SECONDS = 120.0
ATTEMPTS = 3
BLOCK = 1 << 20

PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2

Opener = Callable[[str], BinaryIO]


@dataclass(frozen=True)
class Source:
    set_name: str
    name: str
    role: str
    size: int | None
    md5: str | None
    url: str
    sha256: str | None = None


class TableError(Exception):
    pass


class FetchError(Exception):
    """A file could not be fetched; the message holds no content and no request headers."""


def read_table(path: Path) -> list[Source]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise TableError(f"cannot read {path.name}: {error.strerror}") from None
    sources = []
    for number, row in enumerate(rows, start=2):
        if row.get("role") not in ROLES:
            raise TableError(f"{path.name} line {number}: role must be one of {ROLES}")
        if not row.get("url", "").startswith("https://"):
            raise TableError(f"{path.name} line {number}: the URL must start with https://")
        size = int(row["bytes"]) if row.get("bytes") else None
        md5 = row["md5"].lower() if row.get("md5") else None
        sources.append(Source(row["set"], row["name"], row["role"], size, md5, row["url"]))
    if not sources:
        raise TableError(f"{path.name} lists no files")
    keys = [(s.set_name, s.name) for s in sources]
    if len(set(keys)) != len(keys):
        raise TableError(f"{path.name} lists a file twice")
    return sources


def read_context_table(path: Path) -> list[Source]:
    """The rows of a context table (module text): every role "context", every URL https, a
    SHA-256 of 64 hexadecimal characters or none, no file listed twice."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise TableError(f"cannot read {path.name}: {error.strerror}") from None
    sources = []
    for number, row in enumerate(rows, start=2):
        if row.get("role") != CONTEXT_ROLE:
            raise TableError(f"{path.name} line {number}: the role must be {CONTEXT_ROLE!r}")
        if not row.get("url", "").startswith("https://"):
            raise TableError(f"{path.name} line {number}: the URL must start with https://")
        sha256 = row["sha256"].lower() if row.get("sha256") else None
        if sha256 is not None and (
            len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256)
        ):
            raise TableError(f"{path.name} line {number}: a SHA-256 has 64 hexadecimal characters")
        size = int(row["bytes"]) if row.get("bytes") else None
        sources.append(Source(row["set"], row["name"], row["role"], size, None, row["url"], sha256))
    if not sources:
        raise TableError(f"{path.name} lists no files")
    keys = [(s.set_name, s.name) for s in sources]
    if len(set(keys)) != len(keys):
        raise TableError(f"{path.name} lists a file twice")
    return sources


def destination(root: Path, source: Source) -> Path:
    if source.role == "held-out":
        return heldout_seal.sealed_path(root, source.set_name, source.name)
    return Path(root) / "data" / "raw" / source.set_name / source.name


def http_opener(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as error:
        raise FetchError(f"HTTP {error.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise FetchError(f"network error: {type(error).__name__}") from None


def stream_to(opener: Opener, url: str, temp: Path) -> tuple[int, str, str]:
    """Copy the response bytes to `temp`; return byte size, SHA-256 and MD5. Never decoded."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    try:
        with opener(url) as response, temp.open("wb") as out:
            for block in iter(lambda: response.read(BLOCK), b""):
                out.write(block)
                sha256.update(block)
                md5.update(block)
                size += len(block)
    except FetchError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise FetchError(f"network error: {type(error).__name__}") from None
    return size, sha256.hexdigest(), md5.hexdigest()


def matches(source: Source, size: int, md5: str, sha256: str | None = None) -> list[str]:
    """What does not match the table, as short phrases; empty if all matches."""
    problems = []
    if source.size is not None and size != source.size:
        problems.append(f"{size} bytes, the table says {source.size}")
    if source.md5 is not None and md5 != source.md5:
        problems.append("MD5 differs from the published one")
    if source.sha256 is not None and sha256 != source.sha256:
        problems.append("SHA-256 differs from the committed one")
    return problems


def checked(source: Source) -> str:
    parts = []
    parts.append("size as published" if source.size is not None else "no published size")
    parts.append("MD5 as published" if source.md5 is not None else "no published checksum")
    if source.sha256 is not None:
        parts.append("SHA-256 as committed")
    return ", ".join(parts)


def fetch_one(
    root: Path, source: Source, opener: Opener, say: Callable[[str], None], pause: float
) -> int:
    target = destination(root, source)
    shown = target.relative_to(root).as_posix()
    if target.exists():
        digest = heldout_seal.file_digest(target)
        problems = matches(source, digest.size, digest.md5, digest.sha256)
        if problems:
            say(f"FAILED: {shown} is already there and differs: {'; '.join(problems)}; kept")
            return FAILED
        say(f"present: {shown} {digest.size} bytes sha256 {digest.sha256} ({checked(source)})")
        return PASSED
    incoming = Path(root) / INCOMING
    incoming.mkdir(parents=True, exist_ok=True)
    temp = incoming / f"{source.set_name}--{source.name}.part"
    last_error = ""
    for attempt in range(ATTEMPTS):
        try:
            size, sha256, md5 = stream_to(opener, source.url, temp)
            break
        except FetchError as error:
            temp.unlink(missing_ok=True)
            last_error = str(error)
            if attempt < ATTEMPTS - 1:
                time.sleep(pause * 2**attempt)
    else:
        say(f"COULD NOT RUN: {shown}: {last_error}")
        return COULD_NOT_RUN
    problems = matches(source, size, md5, sha256)
    if problems:
        temp.unlink(missing_ok=True)
        say(f"FAILED: {shown}: {'; '.join(problems)}; not kept")
        return FAILED
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, target)
    say(f"fetched: {shown} {size} bytes sha256 {sha256} md5 {md5} ({checked(source)})")
    return PASSED


def fetch_all(
    root: Path,
    sources: list[Source],
    opener: Opener = http_opener,
    say: Callable[[str], None] = print,
    pause: float = 5.0,
) -> int:
    results = [fetch_one(root, source, opener, say, pause) for source in sources]
    incoming = Path(root) / INCOMING
    if incoming.is_dir() and not any(incoming.iterdir()):
        incoming.rmdir()
    n_failed = results.count(FAILED)
    n_cannot = results.count(COULD_NOT_RUN)
    say(f"{results.count(PASSED)} in place, {n_failed} failed, {n_cannot} could not be fetched")
    if n_failed:
        return FAILED
    return COULD_NOT_RUN if n_cannot else PASSED


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Download the listed public data files.")
    parser.add_argument("--root", type=Path, default=TOP, help="project top folder")
    parser.add_argument(
        "--context", action="store_true", help="download the context files of context_sources.csv"
    )
    args = parser.parse_args(argv)
    try:
        sources = read_context_table(CONTEXT_TABLE) if args.context else read_table(TABLE)
    except TableError as error:
        print(f"COULD NOT RUN: {error}")
        return COULD_NOT_RUN
    return fetch_all(args.root, sources)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
