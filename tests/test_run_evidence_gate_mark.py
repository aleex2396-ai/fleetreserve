"""The sealed-data gate marks the recorded run it passes in (`gate_mark`), so `run_evidence` never
reads any results file of that run from an earlier commit (point 4, reading E3), whether or not
the file holds the route's record `sealed_route`.

On a made-up sealed repository (`fake_heldout.sealed_repo`, its runs folder ignored as the
project's is, tagged `frozen-plan`), made-up commands are recorded by the real recorder
(`run_record.record_run`) at the tag, in a new Python process whose package is the repository's
copy; a later commit changes only `docs/` (a change `read_evidence` accepts for a run that did not
pass the gate).
1. A command that passes the gate (`sealed_route.require_gate`, or the opener `open_sealed`) and
   writes a results file without the key: read at the tag, stale from the later commit.
2. A command that writes the route's record into its main results and a second output file
   without it: both stale from the later commit.
3. A development run (no gate) beside them: read from the later commit as before, its run folder
   holding the log and its results only.
4. The mark is listed in the run's manifest with its SHA-256: deleting it, or changing it, leaves
   the run stale.
5. With no recorded run (the variable unset, a folder outside the repository's runs folder, a
   finished run's folder) nothing is written and the gate passes as before; a mark that cannot be
   written refuses the gate; a second pass leaves the first mark.
6. A run folder named through a symbolic link on its path is marked; a gate refused at its
   clean-tree or code check (`require_gate` and the opener) leaves no mark; the mark names the
   fix commit at a logged parsing fix.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# `fake_heldout` (the made-up repository) lives in a folder the conftest does not add to the path
sys.path.insert(0, str(Path(__file__).resolve().parent / "wp2"))

from fake_heldout import ASR_FILE, asr_csv, asr_row, commit, git, sealed_repo  # noqa: E402

from fleetreserve import gate_mark, heldout_loader, run_evidence, run_record, sealed_route  # noqa: E402
from fleetreserve.gate_summary import ROUTE_KEY  # noqa: E402

RESULTS = "results.json"
SECOND = "second.json"
ASR_BYTES = asr_csv([asr_row("EV1", "2023-02-27T07:30:00", "2023-02-27T17:30:00", 20.0)])

COMMAND = '''"""A made-up command: passes the gate in one of several ways, then writes results."""

import json
import os
import sys
from pathlib import Path

from fleetreserve import heldout_loader, sealed_route

root, how = Path(sys.argv[1]), sys.argv[2]
out = Path(os.environ["FLEETRESERVE_RUN_DIR"]) if "FLEETRESERVE_RUN_DIR" in os.environ else None
document = {"value": 7}
if how in ("gate", "twice", "record"):
    sealed_route.require_gate(root)
if how == "twice":
    sealed_route.require_gate(root)
if how == "opener":
    heldout_loader.open_sealed(root, "asr", "202410DatasetEVOfficeParking_v0.csv")
if how == "record":
    document[sealed_route.gate_summary.ROUTE_KEY] = {"kind": "held-out"}
if out is not None:
    (out / "results.json").write_text(json.dumps(document))
    if how == "record":
        (out / "second.json").write_text(json.dumps({"value": 8}))
print("done")
'''


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.delenv(run_record.RUN_DIR_VARIABLE, raising=False)


@pytest.fixture
def repo(tmp_path):
    """A made-up sealed repository whose runs folder is ignored, tagged at its last commit."""
    root = sealed_repo(tmp_path, {ASR_FILE: ASR_BYTES}, tag=False)
    with (root / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write(f"{run_record.RUNS_FOLDER}/\n")
    commit(root, "the runs folder ignored")
    git(root, "tag", heldout_loader.TAG)
    return root


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "command" / "made_up_command.py"
    path.parent.mkdir()
    path.write_text(COMMAND, encoding="utf-8")
    return path


def record(repo: Path, script: Path, how: str, monkeypatch, label: str | None = None) -> Path:
    """The real recorder runs the made-up command in a new Python process whose package is the
    repository's copy; the run folder."""
    label = label or f"made-up-{how}"
    monkeypatch.setenv("PYTHONPATH", str(repo / "src"))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    command = [sys.executable, "-B", str(script), str(repo), how]
    assert run_record.record_run(label, command, start_dir=repo) == 0
    return run_evidence.newest_run(repo, label)


def later(repo: Path) -> str:
    """A commit that changes `docs/` only (on the keep-list of `run_evidence`)."""
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "notes.md").write_text("changed\n", encoding="utf-8")
    git(repo, "add", "--", "docs/notes.md")
    git(repo, "commit", "-q", "--no-verify", "-m", "a change under docs")
    return git(repo, "rev-parse", "HEAD")


def stale(run: Path, name: str, head: str) -> str:
    with pytest.raises(run_evidence.CannotCheck) as raised:
        run_evidence.read_evidence(run, name, head)
    return str(raised.value)


MARKED = (
    f"the sealed-data gate marked its run ({gate_mark.MARK_NAME} in its folder or its manifest)"
)


# --- 1. a command through the gate whose results do not hold the key ---------------------------


@pytest.mark.parametrize("how", ["gate", "opener"])
def test_a_run_through_the_gate_without_the_key_is_stale_from_an_earlier_commit(
    repo, script, how, monkeypatch
):
    run = record(repo, script, how, monkeypatch)
    assert ROUTE_KEY not in json.loads((run / RESULTS).read_text())
    at = git(repo, "rev-parse", "HEAD")
    assert run_evidence.read_evidence(run, RESULTS, at).origin == run_evidence.AT_CHECKED_OUT
    text = stale(run, RESULTS, later(repo))
    assert MARKED in text
    assert "(reading E3)" in text


# --- 2. a second output file of a command that writes the record -------------------------------


def test_a_second_output_file_without_the_key_is_stale_from_an_earlier_commit(
    repo, script, monkeypatch
):
    run = record(repo, script, "record", monkeypatch)
    assert ROUTE_KEY in json.loads((run / RESULTS).read_text())
    assert ROUTE_KEY not in json.loads((run / SECOND).read_text())
    head = later(repo)
    assert f"its results hold the route's record {ROUTE_KEY}" in stale(run, RESULTS, head)
    assert MARKED in stale(run, SECOND, head)


# --- 3. a development run --------------------------------------------------------------------


def test_a_development_run_is_read_from_an_earlier_commit_and_unmarked(repo, script, monkeypatch):
    run = record(repo, script, "none", monkeypatch)
    manifest = json.loads((run / run_record.MANIFEST_NAME).read_text())
    assert sorted(manifest["outputs"]) == [run_record.LOG_NAME, RESULTS]
    assert sorted(path.name for path in run.iterdir()) == sorted(
        [run_record.LOG_NAME, RESULTS, run_record.MANIFEST_NAME]
    )
    evidence = run_evidence.read_evidence(run, RESULTS, later(repo))
    assert evidence.origin == run_evidence.FROM_EARLIER


# --- 4. the mark in the run's hashes ---------------------------------------------------------


def test_the_mark_is_listed_in_the_manifest_with_its_hash(repo, script, monkeypatch):
    run = record(repo, script, "twice", monkeypatch)
    mark = run / gate_mark.MARK_NAME
    manifest = json.loads((run / run_record.MANIFEST_NAME).read_text())
    assert manifest["outputs"][gate_mark.MARK_NAME] == hashlib.sha256(mark.read_bytes()).hexdigest()
    written = json.loads(mark.read_text())
    assert written["gate"] == "passed"
    assert written["tag_commit"] == git(repo, "rev-list", "-n", "1", heldout_loader.TAG)
    assert "fix_commit" not in written


@pytest.mark.parametrize("how", ["deleted", "changed"])
def test_a_mark_deleted_or_changed_leaves_the_run_stale(repo, script, how, monkeypatch):
    run = record(repo, script, "gate", monkeypatch)
    mark = run / gate_mark.MARK_NAME
    if how == "deleted":
        mark.unlink()
    else:
        mark.write_text("{}\n", encoding="utf-8")
    assert MARKED in stale(run, RESULTS, later(repo))


def test_a_mark_in_the_folder_but_not_in_the_manifest_leaves_the_run_stale(
    repo, script, monkeypatch
):
    run = record(repo, script, "none", monkeypatch)
    (run / gate_mark.MARK_NAME).write_text("{}\n", encoding="utf-8")
    assert MARKED in stale(run, RESULTS, later(repo))


# --- 5. where the gate writes, and what it does when it cannot ---------------------------------


def test_a_command_outside_the_recorder_passes_the_gate_and_writes_nothing(
    repo, script, monkeypatch
):
    monkeypatch.setenv("PYTHONPATH", str(repo / "src"))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    done = subprocess.run(
        [sys.executable, "-B", str(script), str(repo), "gate"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert not (repo / run_record.RUNS_FOLDER).exists()


def going_run(repo: Path, name: str = "20260101T000000Z-made-up") -> Path:
    folder = repo / run_record.RUNS_FOLDER / name
    folder.mkdir(parents=True)
    return folder


def test_the_gate_marks_the_going_run_of_its_repository(repo, monkeypatch):
    folder = going_run(repo)
    monkeypatch.setenv(run_record.RUN_DIR_VARIABLE, str(folder))
    found = sealed_route.require_gate(repo)
    written = json.loads((folder / gate_mark.MARK_NAME).read_text())
    assert written["tag_commit"] == str(found)


def test_the_opener_marks_the_going_run_before_it_reads_the_hash_list(repo, monkeypatch):
    folder = going_run(repo)
    monkeypatch.setenv(run_record.RUN_DIR_VARIABLE, str(folder))
    with pytest.raises(heldout_loader.SealRefused):
        heldout_loader.open_sealed(repo, "asr", "not-listed.csv")
    assert (folder / gate_mark.MARK_NAME).is_file()


@pytest.mark.parametrize("where", ["outside", "finished", "missing"])
def test_no_mark_where_no_run_of_the_repository_is_going(repo, tmp_path, where, monkeypatch):
    if where == "outside":
        folder = tmp_path / "elsewhere" / run_record.RUNS_FOLDER / "20260101T000000Z-made-up"
        folder.mkdir(parents=True)
    elif where == "finished":
        folder = going_run(repo)
        (folder / run_record.MANIFEST_NAME).write_text("{}\n", encoding="utf-8")
    else:
        folder = repo / run_record.RUNS_FOLDER / "20260101T000000Z-missing"
    monkeypatch.setenv(run_record.RUN_DIR_VARIABLE, str(folder))
    sealed_route.require_gate(repo)
    assert not (folder / gate_mark.MARK_NAME).exists()


def test_a_second_pass_leaves_the_first_mark(repo, monkeypatch):
    folder = going_run(repo)
    monkeypatch.setenv(run_record.RUN_DIR_VARIABLE, str(folder))
    (folder / gate_mark.MARK_NAME).write_text("first\n", encoding="utf-8")
    sealed_route.require_gate(repo)
    assert (folder / gate_mark.MARK_NAME).read_text() == "first\n"


def test_a_mark_that_cannot_be_written_refuses_the_gate(repo, monkeypatch):
    folder = going_run(repo)
    monkeypatch.setenv(run_record.RUN_DIR_VARIABLE, str(folder))
    folder.chmod(0o500)
    try:
        with pytest.raises(heldout_loader.SealRefused) as raised:
            sealed_route.require_gate(repo)
        assert isinstance(raised.value.__cause__, OSError)
        with pytest.raises(heldout_loader.SealRefused):
            heldout_loader.open_sealed(repo, "asr", ASR_FILE[1])
    finally:
        folder.chmod(0o700)
    assert not (folder / gate_mark.MARK_NAME).exists()
    assert heldout_loader.logged_openings(repo) == []


# --- 6. links on the path, refused gates, the fix commit ---------------------------------------


def test_a_run_folder_reached_through_a_symbolic_link_is_marked(repo, tmp_path, monkeypatch):
    folder = going_run(repo)
    link = tmp_path / "link-to-runs"
    link.symlink_to(repo / run_record.RUNS_FOLDER)
    monkeypatch.setenv(run_record.RUN_DIR_VARIABLE, str(link / folder.name))
    sealed_route.require_gate(repo)
    assert (folder / gate_mark.MARK_NAME).is_file()


def refused_by(repo: Path, how: str) -> None:
    """Makes the gate of `repo` refuse at its clean-tree check or at its code check."""
    if how == "unclean":
        (repo / "untracked.txt").write_text("not committed\n", encoding="utf-8")
        return
    with (repo / "src" / "fleetreserve" / "run_record.py").open("a", encoding="utf-8") as handle:
        handle.write("# a line the running code does not have\n")
    commit(repo, "the committed code differs from the running code")
    git(repo, "tag", "-f", heldout_loader.TAG)


@pytest.mark.parametrize("how", ["unclean", "code"])
@pytest.mark.parametrize("gate", ["require_gate", "opener"])
def test_a_refused_gate_leaves_no_mark(repo, how, gate, monkeypatch):
    folder = going_run(repo)
    monkeypatch.setenv(run_record.RUN_DIR_VARIABLE, str(folder))
    refused_by(repo, how)
    with pytest.raises(heldout_loader.SealRefused) as raised:
        if gate == "require_gate":
            sealed_route.require_gate(repo)
        else:
            heldout_loader.open_sealed(repo, "asr", ASR_FILE[1])
    expected = "uncommitted changes" if how == "unclean" else "running code"
    assert expected in str(raised.value)
    assert not (folder / gate_mark.MARK_NAME).exists()


def test_the_mark_names_the_fix_commit_at_a_logged_parsing_fix(repo, monkeypatch):
    folder = going_run(repo)
    monkeypatch.setenv(run_record.RUN_DIR_VARIABLE, str(folder))
    tag = git(repo, "rev-list", "-n", "1", heldout_loader.TAG)
    fix = "f" * 40
    assert gate_mark.mark_run(repo, tag, fix) == folder / gate_mark.MARK_NAME
    written = json.loads((folder / gate_mark.MARK_NAME).read_text())
    assert (written["tag_commit"], written["fix_commit"]) == (tag, fix)
