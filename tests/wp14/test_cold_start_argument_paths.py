"""The path arguments and the matrix of the cold start (`cold_start.main`), which takes the
route's checks: each path argument through `gate_run.argument_path`, the matrix through
`gate_run.check_files` before it is read, and read once (`gate_config.read`).

A made-up working copy (`git init` in a temporary folder) is the top folder the command is given
and the current folder; its made-up sealed folder `data/heldout/norway` holds copies of a made-up
verified build's tables (five days of a development unit and of the training unit the two
stand-in models are fitted on: too few for a scored day, so a run that read them stops at "no
unit has a scored day" after it has read them). Nothing of a real data folder is named. Opens,
listings and scans are counted by identity on disk (`sealed_watch`).

1. Hand case, `file:`: `--tables file:data/heldout/norway`, which pandas would read as the sealed
   folder and the file system as a folder of that literal name holding copies of the tables, ends
   with exit 2 and the one line of the URL refusal; nothing of either folder is opened.
2. Hand case, `~`: with HOME the working copy's `data` folder, `--tables ~/heldout/norway` (a
   folder of that literal name holding copies too) is refused by the folder check as the sealed
   folder it names once expanded, with exit 2; nothing of either folder is opened.
3. A matrix file in the sealed folder is refused before it is opened, with exit 2 and one line.
4. On a build with scored days (the unit with 58 days), the matrix file is opened once, and the
   SHA-256 the results record is that of its bytes.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from cold_start_fake import FIT_UNIT, cold_cars, register, write_build
from gate_fake import COMMIT, matrix_document
from sealed_watch import watch

from fleetreserve import cold_start, cold_start_gate, gate_run, run_verify

UNIT = "nrel-from-may-2019"
NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.02-empirical",
    "main-rounded-hour-k0.3-eps0.02-oracle",
)
SEALED = Path("data") / "heldout" / "norway"


def _no_gate(root=None):
    raise AssertionError("the gate is called for development units")


@pytest.fixture
def made(tmp_path, monkeypatch):
    """The made-up build, the working copy as the current folder, the matrix (module text)."""
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setattr(cold_start_gate, "require_gate", _no_gate)
    register(monkeypatch, "path_one")
    register(monkeypatch, "path_two", shift=0.5)
    base = tmp_path / "base"
    base.mkdir()
    run, tables = write_build(base, {UNIT: cold_cars(5), FIT_UNIT: cold_cars(5, "T")})
    copy = tmp_path / "copy"
    copy.mkdir()
    subprocess.run(["git", "init", "--quiet", str(copy)], check=True)
    shutil.copytree(tables, copy / SEALED)
    matrix = tmp_path / "matrix" / "matrix.json"
    matrix.parent.mkdir()
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    monkeypatch.chdir(copy)
    return {"run": run, "tables": tables, "copy": copy, "matrix": matrix, "tmp": tmp_path}


def _main(made: dict, **given) -> int:
    values = {"matrix": made["matrix"], "tables": made["tables"], "truth_run": made["run"]}
    values.update(given)
    arguments = [
        *("--model", "path_one", "--reference", "path_two", "--dial", "0.02"),
        *("--units", UNIT),
    ]
    for name, value in values.items():
        arguments += [f"--{name.replace('_', '-')}", str(value)]
    return cold_start.main(arguments, root=made["copy"])


def _literal(made: dict, text: str) -> Path:
    folder = made["copy"] / text
    shutil.copytree(made["tables"], folder)
    return folder


def test_tables_written_as_a_file_url_are_refused_before_any_open(made, capsys):
    text = "file:" + str(SEALED)
    literal = _literal(made, text)
    with watch(made["copy"] / SEALED, literal) as seen:
        status = _main(made, tables=text)
    assert (status, seen) == (2, [])
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: the path {text} is written as a URL (file:); a run reads only files on "
        "this machine named by a path, never a URL; nothing read\n"
    )


def test_tables_under_the_home_folder_are_checked_where_they_are_read(made, monkeypatch, capsys):
    data = made["copy"] / "data"
    monkeypatch.setenv("HOME", str(data))
    literal = _literal(made, "~/heldout/norway")
    with watch(data / "heldout", literal) as seen:
        status = _main(made, tables="~/heldout/norway")
    assert (status, seen) == (2, [])
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: a folder is refused: the tables folder {data / 'heldout' / 'norway'} "
        f"lies in the data folder {data} outside its derived tables {data / 'derived'}; nothing "
        "read\n"
    )


def test_a_matrix_in_the_sealed_folder_is_refused_before_it_is_opened(made, capsys):
    sealed = made["copy"] / SEALED
    shutil.copyfile(made["matrix"], sealed / "matrix.json")
    with watch(sealed) as seen:
        status = _main(made, matrix=sealed / "matrix.json")
    assert (status, seen) == (2, [])
    data = made["copy"] / "data"
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: a file is refused before it is opened: {sealed / 'matrix.json'} lies in "
        f"the data folder {data} of a working copy, outside its derived tables {data / 'derived'}\n"
    )


def test_the_matrix_is_read_once(made, monkeypatch, capsys):
    base = made["tmp"] / "scored"
    base.mkdir()
    run, tables = write_build(base, {UNIT: cold_cars(58), FIT_UNIT: cold_cars(5, "T")})
    out = made["tmp"] / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    with watch(made["matrix"].parent) as seen:
        status = _main(made, tables=tables, truth_run=run)
    assert status == 0, capsys.readouterr().out
    assert [event for event, _ in seen] == ["open"]
    document = json.loads((out / cold_start.RESULTS_NAME).read_text())
    assert document["matrix"]["sha256"] == hashlib.sha256(made["matrix"].read_bytes()).hexdigest()
