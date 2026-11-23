"""Compare stored ACN month files with their site's fingerprints table.

Run as `python tools/acn_fingerprints.py [--root <folder>] <site> [<site> ...]`, for example
`caltech jpl office001`.

For each site, `data/raw/acn/<site>-fingerprints.csv` lists `month,sessions,sha256`; the
fingerprint is the SHA-256 of the month's canonical JSON, which is the decompressed content of
`data/raw/acn/<site>/<month>.json.gz`. Each listed month must be stored with that fingerprint,
and no month may be stored without being listed. The session counts in the table are not
re-counted: that would mean parsing the sessions, and a matching fingerprint already fixes them.

What it never does: parse or print the content of a month file (it decompresses and hashes the
bytes only). It prints months, match results and the session counts the table itself holds.

Exit status: 0 every site matches; 1 a fingerprint differs or a stored month is not listed;
2 a table or a listed month file is missing.
"""

import argparse
import csv
import gzip
import hashlib
import sys
from pathlib import Path

TOP = Path(__file__).resolve().parents[1]
PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2


def decompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_site(root: Path, site: str, say=print) -> int:
    folder = Path(root) / "data" / "raw" / "acn"
    table = folder / f"{site}-fingerprints.csv"
    if not table.is_file():
        say(f"COULD NOT RUN: {site}: no fingerprints table {table.name}")
        return COULD_NOT_RUN
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        say(f"COULD NOT RUN: {site}: the fingerprints table is empty")
        return COULD_NOT_RUN
    failed = missing = False
    listed = set()
    for row in rows:
        month = row["month"]
        listed.add(month)
        path = folder / site / f"{month}.json.gz"
        if not path.is_file():
            say(f"COULD NOT RUN: {site} {month}: month file missing")
            missing = True
            continue
        if decompressed_sha256(path) != row["sha256"]:
            say(f"FAILED: {site} {month}: fingerprint differs from the table")
            failed = True
    stored = {p.name.removesuffix(".json.gz") for p in (folder / site).glob("*.json.gz")}
    for month in sorted(stored - listed):
        say(f"FAILED: {site} {month}: stored but not in the table")
        failed = True
    sessions = sum(int(row["sessions"]) for row in rows)
    if failed:
        return FAILED
    if missing:
        return COULD_NOT_RUN
    say(
        f"{site}: {len(rows)} months match their fingerprints "
        f"({rows[0]['month']} to {rows[-1]['month']}, {sessions} sessions in the table)"
    )
    return PASSED


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compare ACN month files with fingerprints.")
    parser.add_argument("--root", type=Path, default=TOP, help="project top folder")
    parser.add_argument("sites", nargs="+")
    args = parser.parse_args(argv)
    results = [check_site(args.root, site) for site in args.sites]
    if FAILED in results:
        return FAILED
    return COULD_NOT_RUN if COULD_NOT_RUN in results else PASSED


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
