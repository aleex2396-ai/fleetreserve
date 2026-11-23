"""`tools/corner_closed_form_check.py`, the closed-form half of the WP11 pass-mark item, on
made-up recorded corner searches.

Runs are made with the real run recorder in a throwaway git repository inside pytest's temporary
folder, so their manifests are real. The throwaway repository commits the project's
`gate/matrix.json` and a stand-in package `fleetreserve` whose `corner_run` only writes a made-up
`corner.json` and whose `truth_build` only writes a made-up `truth-summary.json` (read from
environment variables), each exiting with a given status, so the recorded commands have the form
of real runs: `python -m fleetreserve.corner_run --model lgbm_tail --dial 0.02 --matrix
gate/matrix.json` after `python -m fleetreserve.truth_build`.

The made-up search (`results`), by hand: unit u1 checks its days 0 to 11 under both variants and
unit u2 its days 3 to 10 under both; every pair (day, scale) agrees, one pair of u1's forecast
variant within tolerance. A day checked under both variants is one day (reading C1), so 12 + 8 =
20 days are reproduced, the 20 the pass mark needs; counted per variant they would be 40. With
u2's day 10 left out of both variants, 19.

0. Every result begins with the line that says the check covers the closed-form half of WP11
   only and that "run matrix frozen" is not checked.
1. 20 reproduced days and no disagreement: passed (0), one line per unit and variant.
2. A disagreement: failed (1), from a run that ended with exit 1 (read as intact) and also from
   one with 0; a disagreement wins over fewer than 20 days.
3. Could not run (2): no run (with a "NO RUN" line); a stale run; a run on another matrix, of
   another command, on the held truth setting, with another tolerance or other check scales; a
   truth build that is not verified at the checked-out commit; 19 days; a run that ended with
   exit 1 and records no disagreement (a failed variant); counts that do not add up; an unexpected
   error in the check or when the tool's imports fail (never Python's own exit 1).
4. The newest run under the label is the one read.
"""

import copy
import hashlib
import importlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from git_run import run_git

from fleetreserve.dial_splits import GRID
from fleetreserve.run_record import record_run

TOP = Path(__file__).resolve().parents[2]
TOOL = TOP / "tools" / "corner_closed_form_check.py"
sys.path.insert(0, str(TOP / "tools"))
check = importlib.import_module("corner_closed_form_check")

MATRIX_BYTES = (TOP / "gate" / "matrix.json").read_bytes()
MATRIX_SHA = hashlib.sha256(MATRIX_BYTES).hexdigest()
TRUTH_LABEL = "wp2-truth"
HALF = "closed-form half only"

FAKE_RUN = """import os
import sys

out = os.environ["FLEETRESERVE_RUN_DIR"]
name, key, code = {0!r}, {1!r}, {2!r}
if key in os.environ:
    with open(os.path.join(out, name), "w") as handle:
        handle.write(os.environ[key])
sys.exit(int(os.environ.get(code, "0")))
"""
CORNER_RUN = FAKE_RUN.format("corner.json", "FAKE_CORNER", "FAKE_EXIT")
TRUTH_BUILD = FAKE_RUN.format("truth-summary.json", "FAKE_TRUTH", "FAKE_TRUTH_EXIT")
COMMAND = ["--model", "lgbm_tail", "--dial", "0.02", "--matrix", "gate/matrix.json"]


def checked(days, *, within=0, disagreements=()) -> dict:
    """A variant entry whose pass-mark check covers `days` at every scale of the grid."""
    pairs = len(days) * len(GRID)
    wrong = [
        {
            "day": day,
            "date": f"2020-01-{day + 1:02d}",
            "scale": scale,
            "closed_form_first_failure": 1,
            "corner_first_failure": 2,
            "closed_form_slot_loss": 1.0,
            "corner_worst_slot_loss": 0.5,
            "reason": reason,
        }
        for day, scale, reason in disagreements
    ]
    reasons: dict = {}
    for item in wrong:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    if within:
        reasons["within tolerance"] = within
    return {
        "configuration": "made up",
        "searched_days": [*days, 40, 41],
        "pass_mark": {
            "days_checked": len(days),
            "days": list(days),
            "scales": list(GRID),
            "pairs": pairs,
            "pairs_agreed_strict": pairs - len(wrong) - within,
            "pairs_agreed": pairs - len(wrong),
            "differences_by_reason": dict(sorted(reasons.items())),
            "disagreements": wrong,
            "agreed": not wrong,
        },
    }


def results(truth_run: str, *, u2=range(3, 11)) -> dict:
    """The made-up `corner.json` of the module text."""
    u1 = list(range(12))
    return {
        "model": "lgbm_tail",
        "dial": 0.02,
        "inputs": {"setting": "main", "truth_run": truth_run, "table_sha256": {}},
        "matrix": {"file": "matrix.json", "sha256": MATRIX_SHA},
        "run": {"check_days": 20, "check_scales": list(GRID), "slot_tolerance": 1e-9},
        "units": {
            "u0": {"status": "not calibratable: 10 days", "variants": {}},
            "u1": {
                "status": "calibratable",
                "variants": {
                    "forecast": checked(u1, within=1),
                    "oracle-departure": checked(u1),
                },
            },
            "u2": {
                "status": "calibratable",
                "variants": {"forecast": checked(list(u2)), "oracle-departure": checked(list(u2))},
            },
        },
        "failed": [],
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
    (root / "gate").mkdir(parents=True)
    (root / "fleetreserve").mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    (root / ".gitignore").write_text("runs.noindex/\n__pycache__/\n")
    (root / "tracked.txt").write_text("one\n")
    (root / "gate" / "matrix.json").write_bytes(MATRIX_BYTES)
    (root / "fleetreserve" / "__init__.py").write_text("")
    (root / "fleetreserve" / "corner_run.py").write_text(CORNER_RUN)
    (root / "fleetreserve" / "truth_build.py").write_text(TRUTH_BUILD)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first")
    return root


MADE = itertools.count(1)


def _record(repo: Path, label: str, command: list[str], expected: int) -> Path:
    """A recorded run, its folder renamed to a start later than every earlier one of the tests
    (two runs in a second would share a name)."""
    runs = repo / "runs.noindex"
    before = set(runs.glob("*")) if runs.exists() else set()
    assert record_run(label, command, start_dir=repo) == expected
    (folder,) = set(runs.glob("*")) - before
    n = next(MADE)
    return folder.rename(runs / f"20000101T{n // 3600:02d}{n // 60 % 60:02d}{n % 60:02d}Z-{label}")


def truth(repo: Path, monkeypatch) -> str:
    """A recorded truth build at the repository's commit; its folder's name."""
    monkeypatch.setenv("FAKE_TRUTH", json.dumps({"setting": "main", "units": {}}))
    monkeypatch.setenv("FAKE_TRUTH_EXIT", "0")
    return _record(repo, TRUTH_LABEL, [sys.executable, "-m", "fleetreserve.truth_build"], 0).name


def corner(
    repo: Path, monkeypatch, document: dict, *, exit_code: int = 0, module="fleetreserve.corner_run"
) -> Path:
    monkeypatch.setenv("FAKE_CORNER", json.dumps(document))
    monkeypatch.setenv("FAKE_EXIT", str(exit_code))
    return _record(repo, check.LABEL, [sys.executable, "-m", module, *COMMAND], exit_code)


def commit_change(repo: Path) -> None:
    (repo / "tracked.txt").write_text("two\n")
    git(repo, "commit", "-q", "-am", "second")


def run_check(repo: Path, capsys) -> tuple[int, list[str]]:
    capsys.readouterr()  # what the run recorder printed
    code = check.main(["--top", str(repo)])
    return code, capsys.readouterr().out.splitlines()


def edited(document: dict, change) -> dict:
    found = copy.deepcopy(document)
    change(found)
    return found


# --- 0, 1. the line on the closed-form half, and a pass -------------------------------------------


def test_twenty_reproduced_days_pass_with_a_line_per_unit_and_variant(repo, monkeypatch, capsys):
    run = corner(repo, monkeypatch, results(truth(repo, monkeypatch)))
    code, lines = run_check(repo, capsys)
    assert code == 0
    assert HALF in lines[0] and "'run matrix frozen' is not checked" in lines[0]
    pairs = len(GRID)
    assert lines[2:6] == [
        f"  u1 | forecast: 12 days, {12 * pairs} pairs: {12 * pairs - 1} agree strictly, 1 within "
        "tolerance, 0 disagree",
        f"  u1 | oracle-departure: 12 days, {12 * pairs} pairs: {12 * pairs} agree strictly, 0 "
        "within tolerance, 0 disagree",
        f"  u2 | forecast: 8 days, {8 * pairs} pairs: {8 * pairs} agree strictly, 0 within "
        "tolerance, 0 disagree",
        f"  u2 | oracle-departure: 8 days, {8 * pairs} pairs: {8 * pairs} agree strictly, 0 "
        "within tolerance, 0 disagree",
    ]
    assert lines[-1] == (
        f"passed: the closed-form worst case is reproduced on 20 days of 2 units (at least 20 "
        f"needed), {40 * pairs} pairs, none disagreeing; {run.name} (model lgbm_tail, dial 0.02)"
    )


def test_every_result_begins_with_the_closed_form_half_line(repo, monkeypatch, capsys):
    code, lines = run_check(repo, capsys)  # no run
    assert code == 2 and HALF in lines[0]
    corner(repo, monkeypatch, results(truth(repo, monkeypatch), u2=range(3, 10)))
    code, lines = run_check(repo, capsys)  # 19 days
    assert code == 2 and HALF in lines[0]


# --- 2. a disagreement ---------------------------------------------------------------------------


@pytest.mark.parametrize("exit_code", (1, 0))
def test_a_disagreement_fails(repo, monkeypatch, capsys, exit_code):
    def disagree(document):
        document["units"]["u1"]["variants"]["forecast"] = checked(
            list(range(12)), disagreements=[(4, 0.8, "decision-hour cap"), (4, 1.0, "unexplained")]
        )

    document = edited(results(truth(repo, monkeypatch)), disagree)
    run = corner(repo, monkeypatch, document, exit_code=exit_code)
    code, lines = run_check(repo, capsys)
    assert code == 1
    assert lines[2] == (
        f"  u1 | forecast: 12 days, {12 * len(GRID)} pairs: {12 * len(GRID) - 2} agree strictly, "
        "0 within tolerance, 2 disagree (decision-hour cap 1, unexplained 1); day 4 (2020-01-05) "
        "scale 0.8: closed-form hour 1, corners hour 2, decision-hour cap; day 4 (2020-01-05) "
        "scale 1: closed-form hour 1, corners hour 2, unexplained"
    )
    assert lines[-1] == (
        f"DISAGREED: 2 pairs on 1 day disagree (decision-hour cap 1, unexplained 1); {run.name}"
    )


def test_a_disagreement_wins_over_fewer_than_twenty_days(repo, monkeypatch, capsys):
    def few(document):
        del document["units"]["u2"]
        document["units"]["u1"]["variants"]["forecast"] = checked(
            [0, 1], disagreements=[(1, 0.05, "decision-hour cap")]
        )
        document["units"]["u1"]["variants"]["oracle-departure"] = checked([0, 1])

    corner(repo, monkeypatch, edited(results(truth(repo, monkeypatch)), few), exit_code=1)
    code, lines = run_check(repo, capsys)
    assert code == 1 and lines[-1].startswith("DISAGREED: 1 pair on 1 day disagree")


# --- 3. could not run ----------------------------------------------------------------------------


def test_no_run_could_not_run(repo, capsys):
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1] == "NO RUN: no run named <UTC start>-wp11-corner-search in runs.noindex/"


def test_a_stale_run_could_not_run(repo, monkeypatch, capsys):
    corner(repo, monkeypatch, results(truth(repo, monkeypatch)))
    commit_change(repo)
    code, lines = run_check(repo, capsys)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and "is stale" in lines[-1]


def docs_commit(repo: Path) -> str:
    """Commit a change under `docs/` only (on the keep-list of `run_evidence`); the commit
    before it."""
    earlier = run_git(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "notes.md").write_text(f"after {earlier}\n")
    git(repo, "add", "docs/notes.md")
    git(repo, "commit", "-q", "-m", "docs")
    return earlier


def test_a_search_and_truth_build_from_an_earlier_commit_decide_and_say_so(
    repo, monkeypatch, capsys
):
    run = corner(repo, monkeypatch, results(truth(repo, monkeypatch)))
    earlier = docs_commit(repo)
    code, lines = run_check(repo, capsys)
    assert code == 0
    assert (
        lines[1].startswith("corner search checked at commit ")
        and "(a run accepted from an earlier commit says so on its line)" in lines[1]
    )
    assert lines[-1].startswith("passed: ")
    assert f"; {run.name} (accepted from the earlier commit {earlier[:12]}: " in lines[-1]

    def disagree(document):
        document["units"]["u1"]["variants"]["forecast"] = checked(
            list(range(12)), disagreements=[(4, 1.0, "unexplained")]
        )

    run = corner(repo, monkeypatch, edited(results(truth(repo, monkeypatch)), disagree))
    earlier = docs_commit(repo)
    code, lines = run_check(repo, capsys)
    assert code == 1 and lines[-1].startswith("DISAGREED: ")
    assert lines[-1].endswith(
        f"; {run.name} (accepted from the earlier commit {earlier[:12]}: "
        "every committed path that differs between it and the checked-out commit is on the "
        "keep-list of run_evidence; uncommitted files were not compared)"
    )


@pytest.mark.parametrize(
    ("change", "says"),
    (
        (lambda d: d["matrix"].update(sha256="0" * 64), "ran on another matrix"),
        (lambda d: d["run"].update(slot_tolerance=1e-6), "slot tolerance"),
        (
            lambda d: d["units"]["u2"]["variants"]["forecast"]["pass_mark"].update(pairs_agreed=1),
            "u2 | forecast: the pass-mark counts do not add up",
        ),
        (lambda d: d["units"]["u1"].pop("variants"), "u1: corner.json is malformed"),
    ),
    ids=["another matrix", "another tolerance", "counts", "malformed"],
)
def test_a_search_from_an_earlier_commit_that_cannot_decide_is_named_with_the_note(
    repo, monkeypatch, capsys, change, says
):
    run = corner(repo, monkeypatch, edited(results(truth(repo, monkeypatch)), change))
    earlier = docs_commit(repo)
    code, lines = run_check(repo, capsys)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and says in lines[-1]
    assert f"{run.name} (accepted from the earlier commit {earlier[:12]}: " in lines[-1]


def test_a_truth_build_from_an_earlier_commit_not_verified_is_named_with_the_note(
    repo, monkeypatch, capsys
):
    monkeypatch.setenv("FAKE_TRUTH", json.dumps({"setting": "main", "units": {}}))
    monkeypatch.setenv("FAKE_TRUTH_EXIT", "1")
    command = [sys.executable, "-m", "fleetreserve.truth_build"]
    name = _record(repo, TRUTH_LABEL, command, 1).name
    corner(repo, monkeypatch, results(name))
    earlier = docs_commit(repo)
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: the truth build {name} (accepted from the earlier commit "
        f"{earlier[:12]}: every committed path that differs between it and the checked-out "
        "commit is on the keep-list of run_evidence; uncommitted files were not compared) is not "
        "verified"
    )


def test_a_truth_build_not_verified_at_the_commit_could_not_run(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    commit_change(repo)  # the truth build is now of an older commit than the search
    corner(repo, monkeypatch, results(name))
    code, lines = run_check(repo, capsys)
    assert code == 2 and "is stale" in lines[-1] and name in lines[-1]
    for missing in ("20200101T000000Z-wp2-truth", "not-a-run", None):
        corner(repo, monkeypatch, results(missing))
        code, lines = run_check(repo, capsys)
        assert code == 2 and lines[-1].startswith("COULD NOT RUN: ")


@pytest.mark.parametrize(
    ("change", "says"),
    (
        (lambda d: d["matrix"].update(sha256="0" * 64), "ran on another matrix"),
        (lambda d: d.pop("matrix"), "records no SHA-256 of its matrix"),
        (lambda d: d["inputs"].update(setting="held"), "read the truth setting 'held'"),
        (lambda d: d["run"].update(slot_tolerance=1e-6), "slot tolerance"),
        (lambda d: d["run"].update(check_scales=[0.5, 1.0]), "check scales"),
        (
            lambda d: d["units"]["u2"]["variants"]["forecast"]["pass_mark"].update(pairs_agreed=1),
            "u2 | forecast: the pass-mark counts do not add up",
        ),
        (
            lambda d: d["units"]["u2"]["variants"]["forecast"]["pass_mark"].update(days=[3, 4]),
            "u2 | forecast: the pass-mark counts do not add up",
        ),
        (lambda d: d["units"]["u1"].pop("variants"), "u1: corner.json is malformed"),
    ),
)
def test_a_run_that_cannot_decide_could_not_run(repo, monkeypatch, capsys, change, says):
    corner(repo, monkeypatch, edited(results(truth(repo, monkeypatch)), change))
    code, lines = run_check(repo, capsys)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and says in lines[-1]


def test_a_run_of_another_command_could_not_run(repo, monkeypatch, capsys):
    (repo / "fleetreserve" / "other.py").write_text(CORNER_RUN)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "other")
    corner(repo, monkeypatch, results(truth(repo, monkeypatch)), module="fleetreserve.other")
    code, lines = run_check(repo, capsys)
    assert code == 2 and "does not run fleetreserve.corner_run" in lines[-1]


def test_nineteen_days_could_not_run(repo, monkeypatch, capsys):
    corner(repo, monkeypatch, results(truth(repo, monkeypatch), u2=range(3, 10)))
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1] == (
        "COULD NOT RUN: the closed-form worst case is reproduced on 19 days, fewer than the 20 "
        "the pass mark needs, and no pair disagrees"
    )


def test_a_run_ended_with_exit_1_and_no_disagreement_could_not_run(repo, monkeypatch, capsys):
    def failed(document):
        document["units"]["u2"]["variants"]["oracle-departure"] = {
            "configuration": "made up",
            "failed": "ValueError: made up",
        }
        document["failed"] = ["u2 | oracle-departure: ValueError: made up"]

    run = corner(repo, monkeypatch, edited(results(truth(repo, monkeypatch)), failed), exit_code=1)
    code, lines = run_check(repo, capsys)
    assert code == 2  # 20 days still agree (u2's forecast variant checks 8), but the run failed
    assert lines[-1] == (
        f"COULD NOT RUN: {run.name} ended with exit code 1 and records no disagreement (failed: "
        "u2 | oracle-departure: ValueError: made up), so it cannot decide the item"
    )


def test_an_unexpected_error_could_not_run(repo, monkeypatch, capsys):
    corner(repo, monkeypatch, results(truth(repo, monkeypatch)))

    def broken(*args, **kwargs):
        raise RuntimeError("made up")

    monkeypatch.setattr(check, "reproduced_days", broken)
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1] == "COULD NOT RUN: the check stopped on RuntimeError: made up"


def test_the_tool_exits_2_when_its_imports_fail(tmp_path):
    shadow = tmp_path / "shadow" / "fleetreserve"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("")  # a package without the modules the tool imports
    environment = {**os.environ, "PYTHONPATH": str(shadow.parent)}
    done = subprocess.run(
        [sys.executable, str(TOOL), "--top", str(tmp_path)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert done.returncode == 2
    assert done.stdout.startswith("COULD NOT RUN: the check cannot start: ModuleNotFoundError")


# --- 4. the newest run ---------------------------------------------------------------------------


def test_the_newest_run_is_the_one_read(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    corner(repo, monkeypatch, results(name))
    newer = corner(repo, monkeypatch, results(name, u2=range(3, 10)))
    code, lines = run_check(repo, capsys)
    assert code == 2 and "19 days" in lines[-1]
    corner(repo, monkeypatch, results(name))
    code, lines = run_check(repo, capsys)
    assert code == 0 and newer.name not in lines[-1]
