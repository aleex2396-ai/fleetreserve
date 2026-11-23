"""The two holes the review of the cold start's folder check found, closed by the file check of
`gate_run.check_inputs` (`gate_run.refused_file`), each through `cold_start.main`.

Made-up working copies (`git init` in a temporary folder): the top folder the command is given
and, for the first hole, a second working copy. The tables are `cold_start_fake`'s, five days of
a unit (too few for a scored day, so a run that read them would stop at "no unit has a scored
day"), with a verified made-up truth build listing their SHA-256, so the build check alone would
not stop a run that opened them.

1. The tables folder is the sealed folder of a working copy other than the top folder given: the
   folder check passes it (it lies outside the top folder's `data/`), and the run is refused, with
   exit 2, before any file of that sealed folder is opened.
2. The held-out tables folder is a plain folder whose two table files are links into the top
   folder's sealed folder: the folder check passes it, and the run is refused, with exit 2,
   before any file of the sealed folder is opened.
"""

import builtins
import io
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
from cold_start_fake import FIT_UNIT, cold_cars, register, write_build
from gate_fake import COMMIT, matrix_document

from fleetreserve import cold_start, cold_start_gate, gate_run, run_verify

UNIT = "nrel-from-may-2019"
NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.02-empirical",
    "main-rounded-hour-k0.3-eps0.02-oracle",
)
REFUSED = "COULD NOT RUN: a file is refused before it is opened:"


def _working_copy(folder: Path) -> Path:
    folder.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(folder)], check=True)
    return folder


@contextmanager
def _opens():
    """Every path opened through `open`, `io.open` or `os.open` while inside."""
    seen = []
    real_open, real_os_open = builtins.open, os.open

    def watched_open(file, *args, **kwargs):
        seen.append(str(file))
        return real_open(file, *args, **kwargs)

    def watched_os_open(path, *args, **kwargs):
        seen.append(str(path))
        return real_os_open(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(builtins, "open", watched_open)
        patched.setattr(io, "open", watched_open)
        patched.setattr(os, "open", watched_os_open)
        yield seen


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    register(monkeypatch, "hole_one")
    register(monkeypatch, "hole_two", shift=0.5)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return matrix


def _arguments(matrix, tables, run, *units) -> list[str]:
    return [
        *("--model", "hole_one", "--reference", "hole_two", "--dial", "0.02"),
        *("--units", *units),
        *("--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)),
    ]


def test_the_sealed_folder_of_another_working_copy_is_refused_unopened(tmp_path, setup, capsys):
    root = _working_copy(tmp_path / "top")
    other = _working_copy(tmp_path / "other")
    run, tables = write_build(tmp_path, {UNIT: cold_cars(5), FIT_UNIT: cold_cars(5, "T")})
    sealed = other / "data" / "heldout" / "x"
    sealed.parent.mkdir(parents=True)
    tables.replace(sealed)
    cold_start_gate.check_folders(root, sealed, (run,))  # the folder check lets it through
    with _opens() as seen:
        status = cold_start.main(_arguments(setup, sealed, run, UNIT), root=root)
    shown = capsys.readouterr().out
    assert status == 2
    assert REFUSED in shown and f"{sealed}/" in shown
    assert [path for path in seen if str(other / "data") in path] == []


def test_table_files_linked_into_the_sealed_folder_are_refused_unopened(
    tmp_path, setup, monkeypatch, capsys
):
    monkeypatch.setattr(cold_start_gate, "require_gate", lambda root=None: "f" * 40)
    root = _working_copy(tmp_path / "top")
    (tmp_path / "development").mkdir()
    (tmp_path / "held").mkdir()
    run, tables = write_build(tmp_path / "development", {FIT_UNIT: cold_cars(5, "T")})
    held_run, held_tables = write_build(tmp_path / "held", {"OSL_T": cold_cars(5)})
    sealed = root / "data" / "heldout" / "norway"
    sealed.mkdir(parents=True)
    plain = tmp_path / "plain"
    plain.mkdir()
    for kind in ("cars", "days"):
        target = sealed / f"source-{kind}.csv"
        (held_tables / f"OSL_T-{kind}.csv").replace(target)
        (plain / f"OSL_T-{kind}.csv").symlink_to(target)
    arguments = [
        *_arguments(setup, tables, run, "OSL_T"),
        *("--held-out-tables", str(plain), "--held-out-truth-run", str(held_run)),
    ]
    cold_start_gate.check_folders(root, tables, (run, plain, held_run))  # let through
    with _opens() as seen:
        status = cold_start.main(arguments, root=root)
    shown = capsys.readouterr().out
    assert status == 2
    assert REFUSED in shown and "leads through a link to" in shown and str(sealed) in shown
    assert [path for path in seen if str(root / "data") in path] == []
