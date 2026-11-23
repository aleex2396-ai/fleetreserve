"""`tools/gate_t5_check.py`, the WP6 pass-mark item, on made-up recorded runs.

Runs are made with the real run recorder in a throwaway git repository inside pytest's temporary
folder, so their manifests are real; the command of each run only writes a made-up `gate.json` and
a made-up `gate-t5-breaks.csv`. The throwaway repository commits a made-up `gate/matrix.json`.

1. No gate run, or a run folder given that does not exist: could not run (2).
2. A gate run at the checked-out commit, on the committed matrix, with day-paths checked and no
   break: passed (0); any break of (a) or (b), counted or listed in the breaks file: failed (1).
3. A trial, a partial run, no day-path checked, a count missing: could not run (2).
4. A results or breaks file changed after the run, a breaks file missing, a run that exited 1, a
   run under another label: never verified or never read (2).
5. The newest gate run is the one read.
6. A run on another matrix than the committed `gate/matrix.json`, a run with no matrix hash, no
   committed matrix, or a run at another commit than the one checked out: could not run (2).
"""

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from git_run import run_git

from fleetreserve.gate_run import BREAK_COLUMNS
from fleetreserve.run_record import record_run

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
check = importlib.import_module("gate_t5_check")

WRITE = (
    "import os, sys; "
    "out = os.environ['FLEETRESERVE_RUN_DIR']; "
    "open(os.path.join(out, 'gate.json'), 'w').write(sys.argv[1]); "
    "sys.argv[2] == '-' or open(os.path.join(out, 'gate-t5-breaks.csv'), 'w').write(sys.argv[2]); "
    "sys.exit(int(sys.argv[3]))"
)
MATRIX_TEXT = '{"made-up": "matrix"}\n'
MATRIX_SHA = hashlib.sha256(MATRIX_TEXT.encode()).hexdigest()
NO_BREAKS = ",".join(BREAK_COLUMNS) + "\n"
ONE_BREAK = NO_BREAKS + ",".join("a" if c == "kind" else "1" for c in BREAK_COLUMNS) + "\n"
GOOD = {
    "trial": False,
    "partial": False,
    "matrix": {"file": "matrix.json", "sha256": MATRIX_SHA, "limit_days": None, "only": None},
    "t5": {"day_paths_checked": 960, "breaks_a": 0, "breaks_b": 0},
}


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(repo: Path, *args: str) -> None:
    run_git(["git", "-C", str(repo), *args], capture_output=True, check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    (root / ".gitignore").write_text("runs.noindex/\n")
    (root / "tracked.txt").write_text("one\n")
    (root / "gate").mkdir()
    (root / "gate" / "matrix.json").write_text(MATRIX_TEXT)
    git(root, "add", ".gitignore", "tracked.txt", "gate/matrix.json")
    git(root, "commit", "-q", "-m", "first")
    return root


def make_run(
    repo: Path,
    results=None,
    *,
    label: str = "wp6-gate",
    exit_code: int = 0,
    breaks: str = NO_BREAKS,
) -> Path:
    text = json.dumps(GOOD if results is None else results)
    runs = repo / "runs.noindex"
    before = set(runs.glob("*")) if runs.exists() else set()
    command = [sys.executable, "-c", WRITE, text, breaks, str(exit_code)]
    status = record_run(label, command, start_dir=repo)
    assert status == exit_code
    (folder,) = set(runs.glob("*")) - before
    return folder


def changed(**fields) -> dict:
    results = json.loads(json.dumps(GOOD))
    t5 = fields.pop("t5", {})
    results.update(fields)
    results["t5"].update(t5)
    return results


def run_check(repo: Path, *extra: str) -> int:
    return check.main(["--top", str(repo), *extra])


def test_no_gate_run_could_not_run(repo, capsys):
    assert run_check(repo) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
    assert run_check(repo, "--run", str(repo / "runs.noindex" / "none")) == 2


def test_a_gate_run_without_breaks_passes(repo, capsys):
    make_run(repo)
    assert run_check(repo) == 0
    assert "960 day-paths checked, 0 breaks" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["breaks_a", "breaks_b"])
def test_a_break_fails(repo, capsys, kind):
    make_run(repo, changed(t5={kind: 1}))
    assert run_check(repo) == 1
    assert "FAILED: " in capsys.readouterr().out


@pytest.mark.parametrize(
    "results",
    [
        changed(trial=True),
        changed(partial=True),
        changed(t5={"day_paths_checked": 0}),
        changed(t5={"breaks_b": None}),
        changed(t5={"breaks_a": True}),
        {"trial": False, "partial": False},
        {"partial": False, "t5": GOOD["t5"]},
        {"trial": False, "t5": GOOD["t5"]},
    ],
)
def test_a_run_that_cannot_decide_could_not_run(repo, results):
    make_run(repo, results)
    assert run_check(repo) == 2


def test_a_changed_results_file_is_not_verified(repo, capsys):
    folder = make_run(repo, changed(t5={"breaks_a": 1}))
    (folder / "gate.json").write_text(json.dumps(GOOD))
    assert run_check(repo) == 2
    assert "not verified" in capsys.readouterr().out


def test_a_run_that_exited_1_is_not_verified(repo):
    make_run(repo, exit_code=1)
    assert run_check(repo) == 2


def test_other_labels_do_not_count(repo):
    make_run(repo, label="wp6-gate-trial")
    assert run_check(repo) == 2


def test_the_newest_gate_run_is_the_one_read(repo):
    made = make_run(repo)
    old = made.parent / "20000101T000000Z-wp6-gate"  # two runs within one second share a name
    made.rename(old)
    new = make_run(repo, changed(t5={"breaks_b": 2}))
    assert old.name < new.name
    assert run_check(repo) == 1
    assert run_check(repo, "--run", str(old)) == 0


def head(repo: Path) -> str:
    done = run_git(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def test_a_good_run_at_head_on_the_committed_matrix_passes_and_says_so(repo, capsys):
    make_run(repo)
    assert run_check(repo) == 0
    out = capsys.readouterr().out
    assert f"committed gate/matrix.json (sha256 {MATRIX_SHA[:12]})" in out
    assert f"commit {head(repo)[:12]}" in out


@pytest.mark.parametrize(
    "matrix",
    [
        dict(GOOD["matrix"], sha256=hashlib.sha256(b"a smaller matrix").hexdigest()),
        {key: value for key, value in GOOD["matrix"].items() if key != "sha256"},
        None,
    ],
)
def test_a_run_on_another_matrix_could_not_run(repo, capsys, matrix):
    results = changed()
    if matrix is None:
        del results["matrix"]
    else:
        results["matrix"] = matrix
    make_run(repo, results)
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert "COULD NOT RUN" in out and "gate/matrix.json" in out
    if matrix is not None and "sha256" in matrix:
        assert matrix["sha256"][:12] in out and MATRIX_SHA[:12] in out


def test_no_committed_matrix_could_not_run(repo, capsys):
    git(repo, "rm", "-q", "gate/matrix.json")
    git(repo, "commit", "-q", "-m", "no matrix")
    make_run(repo)
    assert run_check(repo) == 2
    assert "gate/matrix.json" in capsys.readouterr().out


def test_a_run_at_another_commit_could_not_run(repo, capsys):
    make_run(repo)
    (repo / "tracked.txt").write_text("two\n")
    git(repo, "commit", "-q", "-am", "next")
    assert run_check(repo) == 2
    assert "stale" in capsys.readouterr().out


def test_a_run_from_an_earlier_commit_with_only_docs_changed_passes_and_says_so(repo, capsys):
    run = make_run(repo)
    earlier = head(repo)
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("later\n")
    git(repo, "add", "docs/notes.md")
    git(repo, "commit", "-q", "-m", "docs")
    assert run_check(repo) == 0
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith("T5 held on every day-path: ")
    assert f"({run.name} (accepted from the earlier commit {earlier[:12]}: " in last
    assert last.endswith(f"checked at commit {head(repo)[:12]})")


@pytest.mark.parametrize(
    "results, exit_code, words",
    [
        (None, 1, "ended with exit code 1, so it is not verified"),
        (changed(matrix={"sha256": "0" * 64}), 0, "ran on another matrix"),
        (changed(trial=True), 0, "is a trial run"),
    ],
    ids=["exit 1", "another matrix", "a trial"],
)
def test_a_run_from_an_earlier_commit_that_cannot_decide_is_named_with_the_note(
    repo, capsys, results, exit_code, words
):
    run = make_run(repo, results, exit_code=exit_code)
    earlier = head(repo)
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("later\n")
    git(repo, "add", "docs/notes.md")
    git(repo, "commit", "-q", "-m", "docs")
    capsys.readouterr()  # what the run recorder printed
    assert run_check(repo) == 2
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(
        f"COULD NOT RUN: {run.name} (accepted from the earlier commit {earlier[:12]}: "
    )
    assert words in lines[0]


def test_a_break_in_the_breaks_file_fails_whatever_the_counts_say(repo, capsys):
    make_run(repo, changed(verdict="passed"), breaks=ONE_BREAK)
    assert run_check(repo) == 1
    out = capsys.readouterr().out
    assert "FAILED: " in out and "1 listed in gate-t5-breaks.csv" in out


@pytest.mark.parametrize("damage", ["missing", "changed", "no header"])
def test_a_missing_changed_or_unreadable_breaks_file_could_not_run(repo, damage):
    folder = make_run(repo, breaks={"missing": "-", "no header": "", "changed": NO_BREAKS}[damage])
    if damage == "changed":
        (folder / "gate-t5-breaks.csv").write_text(ONE_BREAK)
    assert run_check(repo) == 2
