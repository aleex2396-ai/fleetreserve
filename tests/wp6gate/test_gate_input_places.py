"""The files `gate_run.check_inputs` may open (`gate_run.refused_file`, `gate_run.path_forms`).

Made-up working copies (`git init` in a temporary folder) with made-up files; nothing of a real
data folder is named.

1. The forms of a path, worked by hand: in a folder with `a/real.csv`, a link `b -> a`, a link
   `c/link.csv -> ../b/real.csv` and a link `d -> <top>/c`, the path `d/link.csv` is first written
   as `<top>/d/link.csv`, then `<top>/c/link.csv` (d replaced), then `<top>/b/real.csv` (link.csv
   replaced by `../b/real.csv` in `c`), then `<top>/a/real.csv` (b replaced), and that last form is
   where `Path.resolve` says it leads.
2. Refused: a file in a working copy's `data/` outside `data/derived/` (the sealed folder, the raw
   files, `data/` itself, `data/derived/..`), also in a linked working copy (a `.git` file), and in
   the `data/` of the package's own top folder whatever it holds. Not refused: the derived tables,
   a folder named `data` with no `.git` beside it, a `data` folder deeper than a working copy's
   top.
3. A link from a plain folder into a working copy's sealed folder is refused, and so is a file of
   a sealed folder that is itself a link to a plain place, by its written form; a link loop is
   refused.
4. `check_inputs` refuses before it opens any file: a table that is a link to an unreadable file
   in a sealed folder is refused (no permission error), and no file at all is opened, not even
   the truth build's summary; a truth run inside a sealed folder is refused before its summary is
   read, and so is a truth run whose summary or manifest is a link into a sealed folder.
5. A table or a truth summary that is a hard link to a file of a sealed folder (one file on disk
   with two names, which no test of the path can see) is refused before any file is opened, with
   a line saying why; no file of the sealed folder is opened (counted by identity on disk,
   `sealed_watch`).
6. Every file argument of a scoring command is checked before its first open: `gate_run` and
   `learn_run` given a matrix inside a sealed folder exit 2 with one line and do not open it. A
   truth run without its own manifest is refused, and the manifest of a folder above it (here
   one in a working copy's `data/`, outside the derived tables) is not opened. A truth run folder
   that does not exist, or one with no manifest in it or in any folder above it, is refused in
   the same words as before that check.
"""

import builtins
import io
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest
from gate_fake import COMMIT, gate_cars, write_build
from sealed_watch import watch

from fleetreserve import gate_run, learn_run, run_verify
from fleetreserve.truth_verify import CouldNotRun


def _working_copy(folder: Path) -> Path:
    folder.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(folder)], check=True)
    return folder


def _file(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


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


# --- 1. the forms of a path ----------------------------------------------------------------------


def test_the_forms_of_a_path_worked_by_hand(tmp_path):
    top = tmp_path.resolve()
    _file(top / "a" / "real.csv")
    (top / "b").symlink_to("a", target_is_directory=True)
    (top / "c").mkdir()
    (top / "c" / "link.csv").symlink_to(Path("..") / "b" / "real.csv")
    (top / "d").symlink_to(top / "c", target_is_directory=True)
    forms = list(gate_run.path_forms(top / "d" / "link.csv"))
    assert forms == [
        top / "d" / "link.csv",
        top / "c" / "link.csv",
        top / "b" / "real.csv",
        top / "a" / "real.csv",
        top / "a" / "real.csv",
    ]
    assert forms[-1] == (top / "d" / "link.csv").resolve()


def test_a_relative_path_is_written_from_the_current_folder(tmp_path, monkeypatch):
    top = tmp_path.resolve()
    _file(top / "x" / "f.csv")
    monkeypatch.chdir(top / "x")
    forms = list(gate_run.path_forms(Path("..") / "x" / "f.csv"))
    assert forms[0] == top / "x" / "f.csv"
    assert forms[-1] == top / "x" / "f.csv"


# --- 2. which places are refused -----------------------------------------------------------------


def test_a_working_copys_data_folder_is_refused_outside_its_derived_tables(tmp_path):
    top = _working_copy(tmp_path / "copy")
    data = top / "data"
    for refused in (
        data / "heldout" / "norway" / "a.csv",
        data / "raw" / "a.csv",
        data / "a.csv",
        data / "derived" / ".." / "heldout" / "a.csv",
    ):
        found = gate_run.refused_file(refused)
        assert found is not None and "a working copy, outside its derived tables" in found
    for allowed in (
        data / "derived" / "truth" / "a-cars.csv",
        data / "derived" / "truth-heldout" / "a-cars.csv",
        top / "runs.noindex" / "run" / "truth-summary.json",
        top / "sub" / "data" / "heldout" / "a.csv",  # no .git beside this data folder
        tmp_path / "plain" / "data" / "heldout" / "a.csv",  # no working copy at all
    ):
        assert gate_run.refused_file(allowed) is None


def test_a_linked_working_copy_is_a_working_copy(tmp_path):
    top = tmp_path / "linked"
    _file(top / ".git", "gitdir: /elsewhere\n")  # what git writes in a linked working copy
    assert gate_run.refused_file(top / "data" / "heldout" / "a.csv") is not None
    assert gate_run.refused_file(top / "data" / "derived" / "a.csv") is None


def test_the_data_folder_of_the_packages_own_top_folder_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(gate_run, "TOP", tmp_path)  # a top folder without .git
    assert gate_run.refused_file(tmp_path / "data" / "heldout" / "a.csv") is not None
    assert gate_run.refused_file(tmp_path / "data" / "derived" / "a.csv") is None
    assert gate_run.refused_file(tmp_path / "other" / "data" / "a.csv") is None


# --- 3. links ------------------------------------------------------------------------------------


def test_a_link_into_a_sealed_folder_is_refused(tmp_path):
    top = _working_copy(tmp_path / "copy")
    sealed = _file(top / "data" / "heldout" / "norway" / "sessions.csv")
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "OSL_T-cars.csv").symlink_to(sealed)
    found = gate_run.refused_file(plain / "OSL_T-cars.csv")
    assert found is not None and "leads through a link to" in found
    assert str(sealed) in found


def test_a_sealed_folder_that_is_a_link_is_refused_by_its_written_form(tmp_path):
    top = _working_copy(tmp_path / "copy")
    elsewhere = _file(tmp_path / "disk" / "norway" / "sessions.csv").parent.parent
    (top / "data").mkdir()
    (top / "data" / "heldout").symlink_to(elsewhere, target_is_directory=True)
    path = top / "data" / "heldout" / "norway" / "sessions.csv"
    assert path.resolve() == elsewhere.resolve() / "norway" / "sessions.csv"  # leads out of data/
    found = gate_run.refused_file(path)
    assert found is not None and found.startswith(f"{path} lies in")


def test_a_link_loop_is_refused(tmp_path):
    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    found = gate_run.refused_file(tmp_path / "a")
    assert found is not None and "cannot be followed" in found


# --- 4. check_inputs refuses before it opens anything ---------------------------------------------


@pytest.fixture
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


def test_a_table_linked_to_an_unreadable_sealed_file_is_refused_unopened(tmp_path, verified):
    run, tables = write_build(tmp_path, {"OSL_T": gate_cars(5)})
    top = _working_copy(tmp_path / "copy")
    sealed = top / "data" / "heldout" / "norway"
    sealed.mkdir(parents=True)
    target = sealed / "OSL_T-cars.csv"
    (tables / "OSL_T-cars.csv").replace(target)
    (tables / "OSL_T-cars.csv").symlink_to(target)
    target.chmod(0)
    try:
        with _opens() as seen, pytest.raises(CouldNotRun, match="refused before it is opened"):
            gate_run.check_inputs(run, tables, ("OSL_T",))
    finally:
        target.chmod(0o600)
    assert seen == []  # not the summary, not the manifest, not a table


def test_a_truth_run_in_a_sealed_folder_is_refused_before_its_summary_is_read(tmp_path, verified):
    run, tables = write_build(tmp_path, {"OSL_T": gate_cars(5)})
    top = _working_copy(tmp_path / "copy")
    sealed_run = top / "data" / "heldout" / run.name
    sealed_run.parent.mkdir(parents=True)
    run.replace(sealed_run)
    with _opens() as seen, pytest.raises(CouldNotRun, match="refused before it is opened"):
        gate_run.check_inputs(sealed_run, tables, ("OSL_T",))
    assert seen == []


@pytest.mark.parametrize("name", ["truth-summary.json", "manifest.json"])
def test_a_truth_runs_file_linked_into_a_sealed_folder_is_refused_unopened(
    tmp_path, verified, name
):
    run, tables = write_build(tmp_path, {"OSL_T": gate_cars(5)})
    top = _working_copy(tmp_path / "copy")
    target = top / "data" / "heldout" / name
    target.parent.mkdir(parents=True)
    (run / name).replace(target)
    (run / name).symlink_to(target)
    with _opens() as seen, pytest.raises(CouldNotRun, match="leads through a link to"):
        gate_run.check_inputs(run, tables, ("OSL_T",))
    assert seen == []


def test_a_build_outside_any_data_folder_is_checked_as_before(tmp_path, verified):
    run, tables = write_build(tmp_path, {"OSL_T": gate_cars(5)})
    found = gate_run.check_inputs(run, tables, ("OSL_T",))
    assert found["truth_run"] == run.name
    assert set(found["table_sha256"]) == {"OSL_T-cars.csv", "OSL_T-days.csv"}


# --- 5. hard links -------------------------------------------------------------------------------


@pytest.mark.parametrize("linked", ["tables", "summary"])
def test_a_hard_linked_file_is_refused_before_it_is_opened(tmp_path, verified, linked):
    run, tables = write_build(tmp_path, {"OSL_T": gate_cars(5)})
    top = _working_copy(tmp_path / "copy")
    sealed = top / "data" / "heldout" / "norway"
    sealed.mkdir(parents=True)
    names = ["OSL_T-cars.csv", "OSL_T-days.csv"] if linked == "tables" else ["truth-summary.json"]
    folder = tables if linked == "tables" else run
    for name in names:  # the sealed copy and the run's file: one file on disk with two names
        (folder / name).replace(sealed / name)
        os.link(sealed / name, folder / name)
    with watch(sealed) as seen:
        message = "accepted"
        try:
            gate_run.check_inputs(run, tables, ("OSL_T",))
        except CouldNotRun as problem:
            message = str(problem)
    assert seen == []
    assert message == (
        f"a file is refused before it is opened: {folder / names[0]} has 2 names on disk (hard "
        "links); a file with more than one name may be another name of a sealed file, which no "
        "test of its path can see, so any file checked here with more than one name is refused, "
        "sealed or not"
    )


# --- 6. every file argument ----------------------------------------------------------------------


@pytest.mark.parametrize("command", ["gate_run", "learn_run"])
def test_a_matrix_in_a_sealed_folder_is_refused_before_it_is_opened(tmp_path, command, capsys):
    top = _working_copy(tmp_path / "copy")
    sealed = _file(top / "data" / "heldout" / "norway" / "matrix.json", "{}").parent
    main = {"gate_run": gate_run.main, "learn_run": learn_run.main}[command]
    model = [] if command == "gate_run" else ["--model", "empirical"]
    with watch(sealed) as seen:
        status = main([*model, "--matrix", str(sealed / "matrix.json")], root=top)
    assert (status, seen) == (2, [])
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: a file is refused before it is opened: {sealed / 'matrix.json'} lies in "
        f"the data folder {top / 'data'} of a working copy, outside its derived tables "
        f"{top / 'data' / 'derived'}\n"
    )


def test_a_truth_run_is_read_only_with_its_own_manifest(tmp_path, verified):
    top = _working_copy(tmp_path / "copy")
    runs = top / "data" / "derived" / "runs"
    runs.mkdir(parents=True)
    run, tables = write_build(runs, {"OSL_T": gate_cars(5)})
    manifest = json.loads((run / "manifest.json").read_text())
    above = top / "data" / "manifest.json"  # in the data folder, outside the derived tables
    summary = f"derived/runs/{run.name}/truth-summary.json"
    above.write_text(
        json.dumps({**manifest, "outputs": {summary: manifest["outputs"].popitem()[1]}})
    )
    (run / "manifest.json").unlink()
    with watch(above) as seen:
        message = "accepted"
        try:
            gate_run.check_inputs(run, tables, ("OSL_T",))
        except CouldNotRun as problem:
            message = str(problem)
    assert seen == []
    assert message == (
        f"{run / 'manifest.json'} is missing; a truth build is read only with the manifest of its "
        "own run folder"
    )


@pytest.mark.parametrize("missing", ["the run folder", "every manifest"])
def test_a_missing_truth_run_or_one_with_no_manifest_anywhere_is_refused_in_the_same_words(
    tmp_path, verified, missing
):
    run, tables = write_build(tmp_path, {"OSL_T": gate_cars(5)})
    if missing == "the run folder":
        run = tmp_path / "20260102T000000Z-wp2-truth"  # no such folder
        problem = "the file is missing"
    else:
        (run / "manifest.json").unlink()
        problem = "no manifest.json in any parent folder of the file"
    expected = f"{run / 'truth-summary.json'}: {problem}"
    assert not [p for p in (run / "x").parents if (p / "manifest.json").exists()]
    with pytest.raises(CouldNotRun) as raised:
        gate_run.check_inputs(run, tables, ("OSL_T",))
    assert str(raised.value) == expected
