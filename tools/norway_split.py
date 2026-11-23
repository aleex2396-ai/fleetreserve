"""Split the Norway session file by site role: held-out sites' rows go straight to the seal.

Run once as `python tools/norway_split.py [--root <folder>]` from the project's top folder.

Input: the sealed original `data/heldout/norway-original/Dataset1_charging_reports.csv`
(Zenodo 13896176, "Dataset 1: Residential charging reports"; columns, from the data paper's
Table 1: location, session_id, user_id, plugin_time, plugout_time, connection_time,
energy_session). The site is the `location` column. Roles (plan section 3) and outputs:
  train: OSL_S, ASK, TRO, BAR, BOD -> `data/raw/norway/Dataset1_charging_reports-train.csv`
  development: OSL_1, BER, KRO, OSL_2, BAR_2
    -> `data/raw/norway/Dataset1_charging_reports-development.csv`
  held-out: OSL_T, TRO_R -> `data/heldout/norway/Dataset1_charging_reports-held-out.csv`
The field separator is the first of DELIMITERS whose split of the header line names the site
column (the published file uses ";", and its header lists user_id before session_id; the
columns are found by name). Each output starts with the input's header line and holds the
input's rows of its sites, byte for byte and in their input order; a last row without a line
ending gets the header's. So the same input always gives the same output bytes.

Datasets 2 to 4 (user, session and hourly predictions) also hold data of the held-out sites'
users and sessions. They are not split here: they stay sealed whole in
`data/heldout/norway-original/` until a later work package needs them.

What it prints: file names, byte sizes, row counts per output file and SHA-256 hashes, nothing
else. It never prints a data row or any value from one, except that an error may name column
names, or unknown values of the site column that look like site codes (at most MAX_SHOWN,
letters, digits and "_" only, at most 12 characters; other unknown values are only counted).

It fails closed, writing nothing, when: the `location` column is not in the header; a site code
is not one of the twelve; a record spans more than one line or a line is blank; the output row
counts do not add up to the input's; an output file already exists with different bytes.

Exit status: 0 the outputs are written (or already present with the same bytes); 1 it failed
closed; 2 the input file is missing.
"""

import argparse
import csv
import hashlib
import io
import os
import re
import sys
from pathlib import Path

TOP = Path(__file__).resolve().parents[1]
INPUT = "data/heldout/norway-original/Dataset1_charging_reports.csv"
SITE_COLUMN = "location"
DELIMITERS = (",", ";", "\t")  # tried in this order on the header line
SITES = {
    "train": ("OSL_S", "ASK", "TRO", "BAR", "BOD"),
    "development": ("OSL_1", "BER", "KRO", "OSL_2", "BAR_2"),
    "held-out": ("OSL_T", "TRO_R"),
}
OUTPUTS = {
    "train": "data/raw/norway/Dataset1_charging_reports-train.csv",
    "development": "data/raw/norway/Dataset1_charging_reports-development.csv",
    "held-out": "data/heldout/norway/Dataset1_charging_reports-held-out.csv",
}
ROLE_OF_SITE = {site: role for role, sites in SITES.items() for site in sites}
MAX_SHOWN = 12
_CODE_LIKE = re.compile(r"[A-Za-z0-9_]{1,12}")
BOM = "\xef\xbb\xbf"  # a UTF-8 byte order mark, as read byte by byte

PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2


class FailClosed(Exception):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def line_ending(line: bytes) -> bytes:
    for ending in (b"\r\n", b"\n", b"\r"):
        if line.endswith(ending):
            return ending
    return b"\n"


def split_bytes(data: bytes) -> dict[str, tuple[bytes, int]]:
    """Each role's output bytes and row count; FailClosed (content-free message) otherwise."""
    lines = data.splitlines(keepends=True)
    if not lines:
        raise FailClosed("the input file is empty")
    # Latin-1 maps every byte to one character, so decoding never fails; only the ASCII site
    # codes and column names are compared.
    text = data.decode("latin-1").removeprefix(BOM)  # the mark stays in the output bytes
    first_line = text.splitlines()[0] if text else ""
    delimiter = None
    for candidate in DELIMITERS:  # the first separator whose header names the site column
        names = [name.strip() for name in next(csv.reader([first_line], delimiter=candidate))]
        if SITE_COLUMN in names:
            delimiter, header = candidate, names
            break
    if delimiter is None:
        names = [name.strip() for name in next(csv.reader([first_line]))]
        raise FailClosed(f"no column {SITE_COLUMN!r}; the header's columns are {names}")
    records = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    if len(records) != len(lines):
        raise FailClosed(
            f"{len(records)} records on {len(lines)} lines: a record spans more than one line"
        )
    column = header.index(SITE_COLUMN)
    ending = line_ending(lines[0])
    parts: dict[str, list[bytes]] = {role: [lines[0]] for role in SITES}
    unknown: set[str] = set()
    for number, (record, line) in enumerate(zip(records[1:], lines[1:], strict=True), start=2):
        if not record or column >= len(record):
            raise FailClosed(f"line {number} is blank or has too few fields")
        role = ROLE_OF_SITE.get(record[column])
        if role is None:
            unknown.add(record[column])
            continue
        parts[role].append(line if line.endswith((b"\n", b"\r")) else line + ending)
    if unknown:
        shown = sorted(code for code in unknown if _CODE_LIKE.fullmatch(code))[:MAX_SHOWN]
        others = len(unknown) - len(shown)
        raise FailClosed(
            f"{len(unknown)} site codes not among the twelve: {shown}"
            + (f" and {others} other values not shown" if others else "")
        )
    counts = {role: len(chunks) - 1 for role, chunks in parts.items()}
    if sum(counts.values()) != len(lines) - 1:
        raise FailClosed("the output row counts do not add up to the input's")
    return {role: (b"".join(chunks), counts[role]) for role, chunks in parts.items()}


def split_file(root: Path, say=print) -> int:
    source = Path(root) / INPUT
    if not source.is_file():
        say(f"COULD NOT RUN: {INPUT} is missing")
        return COULD_NOT_RUN
    data = source.read_bytes()
    try:
        outputs = split_bytes(data)
        for role, (content, _) in outputs.items():
            target = Path(root) / OUTPUTS[role]
            if target.exists() and target.read_bytes() != content:
                raise FailClosed(f"{OUTPUTS[role]} already exists with different bytes")
    except FailClosed as error:
        say(f"FAILED: {error}; nothing written")
        return FAILED
    rows_in = len(data.splitlines()) - 1
    say(f"input: {INPUT} {len(data)} bytes, {rows_in} rows, sha256 {sha256(data)}")
    for role, (content, count) in outputs.items():
        target = Path(root) / OUTPUTS[role]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temp = target.with_name(target.name + ".part")
            temp.write_bytes(content)
            os.replace(temp, target)
        say(f"{role}: {OUTPUTS[role]} {len(content)} bytes, {count} rows, sha256 {sha256(content)}")
    counts = [count for _, count in outputs.values()]
    say(f"rows add up: {' + '.join(map(str, counts))} = {sum(counts)} of {rows_in}")
    return PASSED


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Split the Norway session file by site role.")
    parser.add_argument("--root", type=Path, default=TOP, help="project top folder")
    args = parser.parse_args(argv)
    return split_file(args.root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
