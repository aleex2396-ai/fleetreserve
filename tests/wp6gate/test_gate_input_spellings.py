"""The file check of `gate_run.check_inputs` does not depend on how a path is spelled.

A file system that ignores letter case and Unicode normalisation (as this machine's does) finds
the folder `data` through the names `DATA` and `Data`, and a folder named `café` through the same
name with its accent decomposed (`cafe` and a combining acute accent). Made-up working copies
(`git init` in a temporary folder) hold a made-up sealed folder `data/heldout/norway` with copies
of a made-up build's tables; nothing of a real data folder is named. Every case is refused by
`check_inputs` with `CouldNotRun` before any file is opened, and no file or folder of the made-up
sealed folder is opened, listed or scanned: counted by identity on disk (`sealed_watch`, device
and inode), not by the path.

1. Spelled `DATA` or `Data`, in another working copy: its sealed folder as the tables folder; a
   plain folder whose table files are links to the sealed files written so; a tables folder that
   is a link to the sealed folder written so; a truth run inside the sealed folder written so.
2. The package's own top folder `TOP` (here a made-up folder `café` with no `.git` beside its
   `data/`), written with the accent decomposed: the same four routes in. The name `data` has no
   accent, so this spelling reaches the sealed folder only where the top folder is known by
   identity on disk and not by a `.git` entry.
3. The files the check allows are unchanged: the derived tables of a working copy whose `data`
   is written `DATA` are read, and so are those of the top folder written with the accent
   decomposed.
"""

import os
import shutil
import subprocess
import unicodedata
from pathlib import Path

import pytest
from gate_fake import COMMIT, gate_cars, write_build
from sealed_watch import watch

from fleetreserve import gate_run, run_verify
from fleetreserve.truth_verify import CouldNotRun

UNIT = "OSL_T"
CAFE = "caf\u00e9"  # composed: one character for the accented letter
CAFE_DECOMPOSED = unicodedata.normalize("NFD", CAFE)  # the letter and a combining accent
REFUSED = "a file is refused before it is opened: "
ROUTES = ("sealed-folder", "file-links", "folder-link", "truth-run-inside")


@pytest.fixture
def made(tmp_path, monkeypatch):
    """A made-up build, verified at a made-up commit, outside any data folder."""
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    base = tmp_path / "base"
    base.mkdir()
    run, tables = write_build(base, {UNIT: gate_cars(5)})
    return {"tmp": tmp_path, "run": run, "tables": tables}


def _seal(top: Path, made: dict) -> Path:
    """`top`'s made-up sealed folder, holding copies of the build's tables and truth run."""
    sealed = top / "data" / "heldout" / "norway"
    sealed.mkdir(parents=True)
    for kind in ("cars", "days"):
        shutil.copyfile(made["tables"] / f"{UNIT}-{kind}.csv", sealed / f"{UNIT}-{kind}.csv")
    shutil.copytree(made["run"], top / "data" / "heldout" / made["run"].name)
    return top / "data" / "heldout"


def _route_in(route: str, written: Path, made: dict) -> tuple[Path, Path]:
    """(truth run, tables folder) of one route into the sealed folder `written` (as spelled)."""
    tmp, run, tables = made["tmp"], made["run"], made["tables"]
    if route == "sealed-folder":
        return run, written / "norway"
    if route == "file-links":
        plain = tmp / "plain"
        plain.mkdir()
        for kind in ("cars", "days"):
            (plain / f"{UNIT}-{kind}.csv").symlink_to(written / "norway" / f"{UNIT}-{kind}.csv")
        return run, plain  # no setting file: the main setting, as the build's own folder
    if route == "folder-link":
        (tmp / "folder-link").symlink_to(written / "norway", target_is_directory=True)
        return run, tmp / "folder-link"
    return written / run.name, tables


def _refused_unopened(run: Path, tables: Path, sealed: Path) -> str:
    message = "accepted"
    with watch(sealed) as seen:
        try:
            gate_run.check_inputs(run, tables, (UNIT,))
        except CouldNotRun as problem:
            message = str(problem)
    assert (seen, message[: len(REFUSED)]) == ([], REFUSED)
    return message


# --- 1. DATA and Data, another working copy ------------------------------------------------------


@pytest.mark.parametrize("spelling", ["DATA", "Data"])
@pytest.mark.parametrize("route", ROUTES)
def test_another_working_copys_sealed_folder_spelled_otherwise_is_refused(made, spelling, route):
    other = made["tmp"] / "other"
    other.mkdir()
    subprocess.run(["git", "init", "--quiet", str(other)], check=True)
    sealed = _seal(other, made)
    written = other / spelling / "heldout"
    assert (written / "norway" / f"{UNIT}-cars.csv").is_file()  # one folder on this machine
    run, tables = _route_in(route, written, made)
    message = _refused_unopened(run, tables, sealed)
    assert "outside its derived tables" in message


# --- 2. the top folder with its accent decomposed ------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
def test_the_top_folder_spelled_with_its_accent_decomposed_is_refused(made, monkeypatch, route):
    top = made["tmp"] / CAFE
    top.mkdir()
    sealed = _seal(top, made)
    assert not (top / ".git").exists()  # known only as the package's top folder
    monkeypatch.setattr(gate_run, "TOP", top)
    written = made["tmp"] / CAFE_DECOMPOSED / "data" / "heldout"
    assert str(written) != str(sealed) and os.path.samefile(written, sealed)
    run, tables = _route_in(route, written, made)
    message = _refused_unopened(run, tables, sealed)
    assert "outside its derived tables" in message


# --- 3. the derived tables are read as before -----------------------------------------------------


def test_derived_tables_are_read_whatever_the_spelling(made, monkeypatch):
    other = made["tmp"] / "other"
    other.mkdir()
    subprocess.run(["git", "init", "--quiet", str(other)], check=True)
    shutil.copytree(made["tables"], other / "data" / "derived" / "truth")
    found = gate_run.check_inputs(made["run"], other / "DATA" / "derived" / "truth", (UNIT,))
    assert set(found["table_sha256"]) == {f"{UNIT}-cars.csv", f"{UNIT}-days.csv"}
    top = made["tmp"] / CAFE
    shutil.copytree(made["tables"], top / "data" / "derived" / "truth")
    monkeypatch.setattr(gate_run, "TOP", top)
    written = made["tmp"] / CAFE_DECOMPOSED / "data" / "derived" / "truth"
    found = gate_run.check_inputs(made["run"], written, (UNIT,))
    assert set(found["table_sha256"]) == {f"{UNIT}-cars.csv", f"{UNIT}-days.csv"}
