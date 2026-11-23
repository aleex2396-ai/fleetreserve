"""The committed hash list of every data file: check it, or write it.

Run from the project's top folder as
    python tools/data_hashes.py --check [--root <folder>]
    python tools/data_hashes.py --write [--root <folder>]

The list is `data-hashes.json` in the top folder: one entry per file under `data/raw/` and the
sealed held-out folder, each with its path relative to the top folder, its byte size, the
SHA-256 of its bytes and its role ("train", "development", "held-out", "document" or
"context"). It holds nothing else about the content. Roles follow from the path (ROLES below):
every file in the sealed folder is "held-out"; a file under `data/raw/` with no matching pattern
has no role. "document" is a description that holds no data rows (a data paper, a README, the
code of a data set's authors).
"context" is a public file that holds no charging data (archived weather forecasts, the OxCGRT
index; `tools/fetch_data.py --context`).

--check exits 0 when every listed file is present with its listed size, hash and role and no
unlisted file lies under `data/raw/` or the sealed folder; 1 when a size, hash or role differs
or an unlisted file is found; 2 (could not run) when the list is missing, empty or ill-formed,
or a listed file is missing. A difference outranks a missing file. Files named in
IGNORED_NAMES (folder metadata written by the operating system) are not data and are skipped.

--write adds every unlisted file and updates the size and hash of changed files that are not
held-out. It refuses, writing nothing, when a listed held-out file's size or hash would change
(a held-out hash is committed once, at download: exit 1), when a file has no role (exit 1), or
when a listed file is missing (exit 2).

What it never does: decode, parse or print the content of a data file. Files are read only as
bytes, to hash them (`fleetreserve.heldout_seal.file_digest`).
"""

import argparse
import fnmatch
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from fleetreserve import heldout_seal

TOP = Path(__file__).resolve().parents[1]
LIST_NAME = "data-hashes.json"
RAW = "data/raw"
ROLE_NAMES = ("train", "development", "held-out", "document", "context")
# Path patterns under data/raw and their roles, first match wins (fnmatch: "*" also matches "/").
ROLES = (
    ("data/raw/acn/*", "train"),
    ("data/raw/nrel/*", "development"),
    ("data/raw/norway/*-train.csv", "train"),
    ("data/raw/norway/*-development.csv", "development"),
    ("data/raw/norway/*.pdf", "document"),
    ("data/raw/asr/README.txt", "document"),
    ("data/raw/weather/*", "context"),
    ("data/raw/oxcgrt/*", "context"),
    ("data/raw/school/*", "context"),
    # the Korean authors' plotting scripts, code only (the owner's yes of 15 November 2026)
    ("data/raw/korea/Code.zip", "document"),
)
IGNORED_NAMES = heldout_seal.IGNORED_NAMES

PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2


@dataclass(frozen=True)
class Entry:
    path: str
    bytes: int
    sha256: str
    role: str


class CouldNotRun(Exception):
    pass


def role_of(path: str) -> str | None:
    if path.startswith(heldout_seal.FOLDER + "/"):
        return "held-out"
    for pattern, role in ROLES:
        if fnmatch.fnmatchcase(path, pattern):
            return role
    return None


def data_files(root: Path) -> list[str]:
    """Paths, relative to the top folder, of every file under data/raw and the sealed folder."""
    raw = Path(root) / RAW
    found = [p for p in raw.rglob("*") if p.is_file()] if raw.is_dir() else []
    found += heldout_seal.sealed_files(root)
    names = (p.relative_to(root).as_posix() for p in found if p.name not in IGNORED_NAMES)
    return sorted(names)


def entry_for(root: Path, path: str, role: str) -> Entry:
    digest = heldout_seal.file_digest(Path(root) / path)
    return Entry(path, digest.size, digest.sha256, role)


def load_list(root: Path, allow_missing: bool = False) -> dict[str, Entry]:
    path = Path(root) / LIST_NAME
    if not path.is_file():
        if allow_missing:
            return {}
        raise CouldNotRun(f"no hash list {LIST_NAME}")
    try:
        files = json.loads(path.read_text(encoding="utf-8"))["files"]
        entries = [Entry(f["path"], f["bytes"], f["sha256"], f["role"]) for f in files]
    except (ValueError, KeyError, TypeError) as error:
        raise CouldNotRun(f"{LIST_NAME} is ill-formed ({type(error).__name__})") from None
    by_path = {entry.path: entry for entry in entries}
    if len(by_path) != len(entries):
        raise CouldNotRun(f"{LIST_NAME} lists a path twice")
    for entry in entries:
        well_formed = (
            isinstance(entry.bytes, int)
            and isinstance(entry.sha256, str)
            and len(entry.sha256) == 64
            and entry.role in ROLE_NAMES
        )
        if not well_formed:
            raise CouldNotRun(f"{LIST_NAME} has an ill-formed entry for {entry.path}")
    if not by_path and not allow_missing:
        raise CouldNotRun(f"{LIST_NAME} lists no files")
    return by_path


def save_list(root: Path, entries: list[Entry]) -> None:
    lines = [
        json.dumps({"path": e.path, "bytes": e.bytes, "sha256": e.sha256, "role": e.role})
        for e in sorted(entries, key=lambda e: e.path)
    ]
    text = '{"files": [\n' + ",\n".join(lines) + "\n]}\n"
    path = Path(root) / LIST_NAME
    temp = path.with_name(path.name + ".part")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def check(root: Path, say=print) -> int:
    try:
        listed = load_list(root)
    except CouldNotRun as error:
        say(f"COULD NOT RUN: {error}")
        return COULD_NOT_RUN
    failed = missing = False
    for path, entry in sorted(listed.items()):
        if not (Path(root) / path).is_file():
            say(f"COULD NOT RUN: {path}: listed but missing")
            missing = True
            continue
        found = entry_for(root, path, entry.role)
        if found.bytes != entry.bytes:
            say(f"FAILED: {path}: {found.bytes} bytes, the list says {entry.bytes}")
            failed = True
        elif found.sha256 != entry.sha256:
            say(f"FAILED: {path}: SHA-256 differs from the list")
            failed = True
        if role_of(path) != entry.role:
            say(f"FAILED: {path}: listed as {entry.role}, its folder makes it {role_of(path)}")
            failed = True
    for path in data_files(root):
        if path not in listed:
            say(f"FAILED: {path}: not in the hash list")
            failed = True
    if failed:
        return FAILED
    if missing:
        return COULD_NOT_RUN
    held_out = sum(1 for e in listed.values() if e.role == "held-out")
    say(
        f"{len(listed)} listed files match ({held_out} held-out); no unlisted file under "
        f"{RAW} or {heldout_seal.FOLDER}"
    )
    return PASSED


def write(root: Path, say=print) -> int:
    try:
        listed = load_list(root, allow_missing=True)
    except CouldNotRun as error:
        say(f"COULD NOT RUN: {error}")
        return COULD_NOT_RUN
    on_disk = data_files(root)
    refused = missing = False
    for path in sorted(set(listed) - set(on_disk)):
        say(f"COULD NOT RUN: {path}: listed but missing; the list is not written")
        missing = True
    result: dict[str, Entry] = {}
    added = changed = 0
    for path in on_disk:
        role = role_of(path)
        if role is None:
            say(f"FAILED: {path}: no role for this path; the list is not written")
            refused = True
            continue
        entry = entry_for(root, path, role)
        old = listed.get(path)
        if old is None:
            added += 1
            say(f"added: {path} {entry.bytes} bytes sha256 {entry.sha256} ({role})")
        elif (old.bytes, old.sha256, old.role) != (entry.bytes, entry.sha256, entry.role):
            if old.role == "held-out" or role == "held-out":
                say(f"FAILED: {path}: a listed held-out hash may not change; not written")
                refused = True
                continue
            changed += 1
            say(f"changed: {path} {entry.bytes} bytes sha256 {entry.sha256} ({role})")
        result[path] = entry
    if refused:
        return FAILED
    if missing:
        return COULD_NOT_RUN
    if not result:
        say(f"COULD NOT RUN: no data files under {RAW} or {heldout_seal.FOLDER}")
        return COULD_NOT_RUN
    save_list(root, list(result.values()))
    say(f"wrote {LIST_NAME}: {len(result)} files, {added} added, {changed} changed")
    return PASSED


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check or write the data hash list.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="compare the data with the list")
    mode.add_argument("--write", action="store_true", help="add new files to the list")
    parser.add_argument("--root", type=Path, default=TOP, help="project top folder")
    args = parser.parse_args(argv)  # a wrong call exits with status 2
    return check(args.root) if args.check else write(args.root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
