"""`tools/rehearsal_check.py`, the WP12 pass-mark item, on rehearsal drives of made-up run lists.

Every log here is written by the real driver (`run_driver.drive`) running made-up steps in a
throwaway git repository (`drive_fake`), so its runs are recorded by the real recorder's code and
its log has the driver's own form; a few tests then change a log, a run or the commit.

1. A rehearsal that ran to its end, every step passing (the second fed the run folder of the
   first through `{run:<label>}`): passed (0), one line per step and the total duration on the
   last line, which says there is no difference from the run from the tag.
2. A logged step that failed (exit 1, or its time limit hit): failed (1), naming the step.
3. Could not run (2): no run list committed at its fixed path ("NO RUN LIST"); no rehearsal log
   of it ("NO REHEARSAL LOG"); a stale log; a log whose hash does not recompute; a log without its
   end line; a log whose lines do not follow the list (another command, another run folder put in
   for `{run:<label>}`, another list hash, other steps in its start), with its hash made to
   recompute; a run output changed after the drive; a run's manifest rewritten; a drive that
   stopped before a step; a step the recorder refused; a sealed-run log named as a rehearsal's;
   an unexpected error, and failing imports of the tool (never Python's own 1).
4. The newest log of the committed list is the one read: an older log that passed does not hide a
   newer one, a drive of another list is never read in its place, and a log that is a symbolic
   link is refused.
5. The purposes: the differences from the run from the tag (an entry of the sealed run only, a
   group's units, arguments of the sealed run only) are printed one line each and counted on the
   last line of a passing rehearsal, whose own steps alone are logged; a sealed-run drive that
   passed does not stand in for a rehearsal that failed.
6. O2's verdict (reading R7), from the two stand-in O2 steps (`drive_fake.O2_STEP`) that run list
   rules 10 and 12 require (`drive_fake.o2_entries`), the main one's read: one line per O2
   step with its verdict over the units, the units it failed on and those it could not decide,
   the exit status as without it; an O2 step that exited 0 whose results file holds no readable
   verdict (none, no file, or edited after the run) fails (1), named; an O2 step that exited 1
   fails as any step; the results file is opened only inside `run_verify`, and a symbolic link
   there into the sealed folder is never followed. A unit with no day to check is named on the
   line and counts as neither passed nor failed (the owner's decision of 19 November 2026), also
   end to end from the real O2 command's results file on three made-up units. Beside a verdict
   line, one line gives the units with a non-zero count, each named by what it holds (orders
   broken, both products on every day, with the block orders on M days among them; days not
   built, with the M days among them; only those are in O2's pass; decision 88 (3d)), or says the
   counts are missing or not readable; a report with no fault prints none; the verdict and the
   exit status are as without it.
"""

import fcntl
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import ceiling_fakes
import pytest
from drive_fake import (
    LIST_PATH,
    O2_MODULE,
    O2_UNITS,
    SEALED_MATRIX,
    SEALED_TRUTH,
    commit_list,
    document,
    drive_logs,
    entry,
    git,
    make_repo,
    no_start_check,
    o2_entries,
    read_log,
    step,
)
from opened_from import opened
from sealed_watch import watch

from fleetreserve import ceiling_report, cold_start_gate, run_driver, run_verify
from fleetreserve.ceiling_values import VALUE_KEYS, DayResult

TOP = Path(__file__).resolve().parents[2]
TOOL = TOP / "tools" / "rehearsal_check.py"
sys.path.insert(0, str(TOP / "tools"))
check = importlib.import_module("rehearsal_check")


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    """Drop every GIT_* variable and git's global and system settings for the test."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    no_start_check(monkeypatch)  # the stand-ins' starts (drive_fake, module text)
    return make_repo(tmp_path)


def two_passing() -> list[dict]:
    return [
        entry("wp2-truth", step("truth_build", "--seed", "{seed:split}")),
        entry(
            "rehearsal-learn",
            step("learn_run", "--truth-run", "{run:wp2-truth}"),
            after=["wp2-truth"],
        ),
    ]


def drive(repo: Path, entries: list[dict], *, purpose: str = "rehearsal", units=None) -> int:
    seeds = {"split": 7} if any("{seed:split}" in e["command"] for e in entries) else {}
    if any("{seed:intervals}" in e["command"] for e in entries):
        seeds["intervals"] = 7  # the interval seed of a sealed run (`with_a_sealed_run`)
    commit_list(repo, document(entries, seeds=seeds, units=units))
    return run_driver.drive(Path(LIST_PATH), purpose=purpose, start_dir=repo)


def run_check(repo: Path, capsys) -> tuple[int, list[str]]:
    capsys.readouterr()  # what the driver printed
    code = check.main(["--top", str(repo)])
    return code, capsys.readouterr().out.splitlines()


def the_log(repo: Path) -> Path:
    (path,) = drive_logs(repo)
    return path


def rehash(lines: list[bytes]) -> bytes:
    """The log's lines with its end's hash made to recompute over the lines above it and the
    end's own fields (keys sorted, without the hash)."""
    end = json.loads(lines[-1])
    end.pop("log_sha256")
    own = (json.dumps(end, sort_keys=True) + "\n").encode()
    end["log_sha256"] = hashlib.sha256(b"".join(lines[:-1]) + own).hexdigest()
    return b"".join(lines[:-1]) + (json.dumps(end, sort_keys=True) + "\n").encode()


def passing_gate(monkeypatch) -> None:
    """Make the sealed run's gate pass, as at the tag."""
    monkeypatch.setattr(
        cold_start_gate, "require_gate", lambda root: git(root, "rev-parse", "HEAD").strip()
    )


# --- 1. passed -----------------------------------------------------------------------------------


NO_DIFFERENCE = "  no difference from the run from the tag"
NO_DIFFERENCE_END = "; 0 differences from the run from the tag, none (reading R6)"


def test_a_rehearsal_that_ran_to_its_end_passes(repo, capsys):
    assert drive(repo, two_passing()) == 0
    code, lines = run_check(repo, capsys)
    assert code == 0, lines
    assert lines[0] == check.HEADER
    assert "run-lists/run-list.json" in check.HEADER
    assert lines[1] == NO_DIFFERENCE
    assert lines[2].startswith(f"log {the_log(repo).name}, list {LIST_PATH}, commit ")
    assert lines[3].startswith("  wp2-truth: run runs.noindex/")
    assert lines[4].startswith("  rehearsal-learn: run runs.noindex/")
    assert all(", exit status 0, timed out False, " in line for line in lines[3:5])
    records = read_log(the_log(repo))
    assert records[3]["command"] == step("learn_run", "--truth-run", records[2]["run"])
    seconds = records[-1]["seconds"]
    assert lines[5] == (
        f"passed: the rehearsal of the run list {LIST_PATH} ran to its end at commit "
        f"{git(repo, 'rev-parse', 'HEAD').strip()[:12]}: 2 entries, each exited 0 with no timeout "
        f"and verifies; total duration {seconds:.0f} s ({seconds / 3600:.2f} h); "
        f"{the_log(repo).name}{NO_DIFFERENCE_END}"
    )
    assert len(lines) == 6


def test_a_rehearsal_of_one_entry_says_one_entry(repo, capsys):
    assert drive(repo, [entry("wp2-truth", step("truth_build"))]) == 0
    code, lines = run_check(repo, capsys)
    assert code == 0, lines
    assert ": 1 entry, which exited 0 with no timeout and verifies; total duration " in lines[-1]


# --- 2. failed -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("failing", "said"),
    [
        (entry("rehearsal-bad", step("learn_run", "--exit", "1")), "ended with exit status 1"),
        (
            entry("rehearsal-bad", step("learn_run", "--sleep", "30"), max_seconds=1),
            "hit its time limit (exit status 143)",
        ),
    ],
    ids=["exit-1", "timed-out"],
)
def test_a_logged_entry_that_failed_fails_the_item(repo, capsys, failing, said):
    assert drive(repo, [entry("rehearsal-ok", step("truth_build")), failing]) == 1
    code, lines = run_check(repo, capsys)
    assert code == 1, lines
    assert lines[-1] == f"FAILED: rehearsal-bad {said}; {the_log(repo).name}"


# --- 3. could not run ----------------------------------------------------------------------------


def test_no_run_list_committed_could_not_run_with_a_no_run_list_line(repo, capsys):
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines == [
        check.HEADER,
        "NO RUN LIST: no run list is committed at run-lists/run-list.json at the checked-out "
        "commit",
    ]


def test_no_log_could_not_run_with_a_no_rehearsal_log_line(repo, capsys):
    commit_list(repo, document(two_passing(), seeds={"split": 7}))
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines == [
        check.HEADER,
        NO_DIFFERENCE,
        "NO REHEARSAL LOG: no rehearsal driver log of the committed run list "
        "run-lists/run-list.json, runs.noindex/drives/<UTC start>-rehearsal.jsonl",
    ]


def test_a_stale_log_could_not_run(repo, capsys):
    assert drive(repo, two_passing()) == 0
    (repo / "tracked.txt").write_text("two\n")
    git(repo, "commit", "-q", "-am", "a later commit")
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1].startswith(f"COULD NOT RUN: {the_log(repo).name} is stale: written at commit")


def _with(lines: list[bytes], k: int, change) -> list[bytes]:
    record = json.loads(lines[k])
    change(record)
    lines[k] = (json.dumps(record, sort_keys=True) + "\n").encode()
    return lines


def changed_seconds(lines: list[bytes]) -> bytes:
    return b"".join(_with(lines, 2, lambda r: r.update(seconds=0.5)))


def edited_total(lines: list[bytes]) -> bytes:
    return b"".join(_with(lines, -1, lambda r: r.update(seconds=999999)))


def no_end(lines: list[bytes]) -> bytes:
    return b"".join(lines[:-1])


def another_command_rehashed(lines: list[bytes]) -> bytes:
    return rehash(_with(lines, 1, lambda r: r["command"].append("--more")))


def another_run_folder_rehashed(lines: list[bytes]) -> bytes:
    other = "runs.noindex/20000101T000000Z-wp2-truth"
    return rehash(_with(lines, 3, lambda r: r["command"].__setitem__(-1, other)))


def another_list_hash_rehashed(lines: list[bytes]) -> bytes:
    return rehash(_with(lines, 0, lambda r: r.update(list_sha256="0" * 64)))


def other_steps_rehashed(lines: list[bytes]) -> bytes:
    return rehash(_with(lines, 0, lambda r: r["steps"][1]["command"].append("--more")))


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (changed_seconds, "the hash of the log {log} does not recompute"),
        (edited_total, "the hash of the log {log} does not recompute"),
        (no_end, "the log {log} has no end line: the drive broke off or is still going"),
        (another_command_rehashed, "the log does not follow the run list at step 1 (wp2-truth)"),
        (
            another_run_folder_rehashed,
            "the log does not follow the run list at step 2 (rehearsal-learn)",
        ),
        (
            another_list_hash_rehashed,
            "{log} drove other bytes than those of run-lists/run-list.json committed at the "
            "checked-out commit",
        ),
        (
            other_steps_rehashed,
            "the start of {log} does not list the rehearsal's steps of run-lists/run-list.json as "
            "the list instantiates them",
        ),
    ],
    ids=[
        "hash",
        "end-total",
        "no-end",
        "not-the-list",
        "another-run-folder",
        "other-list-bytes",
        "other-steps",
    ],
)
def test_a_changed_log_could_not_run(repo, capsys, change, said):
    assert drive(repo, two_passing()) == 0
    path = the_log(repo)
    path.write_bytes(change(path.read_bytes().splitlines(keepends=True)))
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1] == f"COULD NOT RUN: {said.format(log=path.name)}"


def test_a_run_output_changed_after_the_drive_could_not_run(repo, capsys):
    assert drive(repo, two_passing()) == 0
    run = read_log(the_log(repo))[2]["run"]
    with (repo / run / "step.txt").open("a") as handle:
        handle.write("changed\n")
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1].startswith(f"COULD NOT RUN: wp2-truth: {run}/step.txt does not verify: ")


def test_a_rewritten_manifest_could_not_run(repo, capsys):
    assert drive(repo, two_passing()) == 0
    run = read_log(the_log(repo))[4]["run"]
    manifest = repo / run / "manifest.json"
    manifest.write_text(json.dumps(json.loads(manifest.read_text())))  # same fields, other bytes
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: rehearsal-learn: the manifest of {run} is not the one the log records"
    )


def test_a_drive_that_stopped_before_an_entry_could_not_run(repo, capsys):
    entries = [
        entry("rehearsal-change", step("truth_build", "--dirty")),
        entry("rehearsal-next", step("learn_run")),
    ]
    assert drive(repo, entries) == 2
    git(repo, "checkout", "--", "tracked.txt")
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1] == (
        "COULD NOT RUN: the drive stopped before rehearsal-next: the git tree has uncommitted "
        "changes since the drive started"
    )


def test_a_drive_whose_last_entry_changed_the_tree_could_not_run(repo, capsys):
    entries = [
        entry("wp2-truth", step("truth_build")),
        entry("rehearsal-last", step("learn_run", "--dirty")),
    ]
    drive(repo, entries)  # every entry exits 0; the last leaves a tracked file changed
    git(repo, "checkout", "--", "tracked.txt")  # the tree is clean again when the check runs
    code, lines = run_check(repo, capsys)
    assert code == 2, lines
    assert lines[-1] == (
        f"COULD NOT RUN: the tree changed after the last step of {the_log(repo).name}: the git "
        "tree has uncommitted changes since the drive started"
    )


def hold_the_lock(repo: Path):
    (repo / "runs.noindex").mkdir(exist_ok=True)
    held = (repo / "runs.noindex" / "heavy.lock").open("a")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return held


def test_an_entry_the_recorder_refused_could_not_run(repo, capsys):
    with hold_the_lock(repo):
        assert drive(repo, two_passing()) == 2
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1] == (
        "COULD NOT RUN: the recorder did not run wp2-truth (status 2), so the rehearsal did not "
        "run it"
    )


def test_a_sealed_run_log_named_as_a_rehearsal_could_not_run(repo, capsys, monkeypatch):
    passing_gate(monkeypatch)
    assert drive(repo, two_passing(), purpose="sealed-run") == 0
    (path,) = drive_logs(repo)
    assert path.name.endswith("-sealed-run.jsonl")
    renamed = path.with_name(path.name.replace("-sealed-run.", "-rehearsal."))
    path.rename(renamed)
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: the start of {renamed.name} names the purpose 'sealed-run', not "
        "'rehearsal'"
    )


def test_an_unexpected_error_could_not_run(repo, monkeypatch, capsys):
    assert drive(repo, two_passing()) == 0

    def broken(*args, **kwargs):
        raise RuntimeError("made up")

    monkeypatch.setattr(check, "verify_run", broken)
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
    assert done.stdout.startswith("COULD NOT RUN: the check cannot start: ImportError")


# --- 4. the newest log ---------------------------------------------------------------------------


def test_an_older_log_that_passed_does_not_hide_a_newer_one(repo, capsys):
    assert drive(repo, two_passing()) == 0
    older = the_log(repo)
    older.rename(older.with_name("20000101T000000Z-rehearsal.jsonl"))  # an earlier start
    with hold_the_lock(repo):
        assert run_driver.drive(Path(LIST_PATH), purpose="rehearsal", start_dir=repo) == 2
    code, lines = run_check(repo, capsys)
    assert code == 2
    assert lines[-1].startswith("COULD NOT RUN: the recorder did not run wp2-truth")
    assert lines[2].startswith("log ") and "20000101T000000Z" not in lines[2]


QUICK_PATH = "run-lists/quick.json"  # another list, at another path


def earlier(path: Path) -> None:
    """Give a log an earlier start in its name, so a later drive's log is the newer one (two
    drives may start in one second)."""
    path.rename(path.with_name("20000101T000000Z-rehearsal.jsonl"))


def test_a_drive_of_another_list_is_not_read_in_place_of_the_committed_one(repo, capsys):
    failing = [
        entry("rehearsal-ok", step("truth_build")),
        entry("rehearsal-bad", step("learn_run", "--exit", "1")),
    ]
    commit_list(repo, document(failing))
    commit_list(repo, document([entry("rehearsal-quick", step("truth_build"))]), QUICK_PATH)
    assert run_driver.drive(Path(LIST_PATH), purpose="rehearsal", start_dir=repo) == 1
    earlier(the_log(repo))
    assert run_driver.drive(Path(QUICK_PATH), purpose="rehearsal", start_dir=repo) == 0
    code, lines = run_check(repo, capsys)
    assert code == 1, lines
    assert lines[-1] == (
        "FAILED: rehearsal-bad ended with exit status 1; 20000101T000000Z-rehearsal.jsonl"
    )


def test_a_drive_of_another_list_alone_is_no_log_of_the_committed_one(repo, capsys):
    commit_list(repo, document(two_passing(), seeds={"split": 7}))
    commit_list(repo, document([entry("rehearsal-quick", step("truth_build"))]), QUICK_PATH)
    assert run_driver.drive(Path(QUICK_PATH), purpose="rehearsal", start_dir=repo) == 0
    code, lines = run_check(repo, capsys)
    assert code == 2, lines
    assert lines[-1].startswith("NO REHEARSAL LOG: no rehearsal driver log of the committed run")


def test_a_log_that_is_a_symbolic_link_could_not_run(repo, capsys):
    assert drive(repo, two_passing()) == 0
    passing = the_log(repo)
    kept = repo / "runs.noindex" / "kept.jsonl"  # in the ignored runs folder, not a log's name
    passing.rename(kept)
    link = passing.with_name("29991231T235959Z-rehearsal.jsonl")
    link.symlink_to(kept)  # the newest name, linked to a log that passed
    code, lines = run_check(repo, capsys)
    assert code == 2, lines
    assert lines[-1] == (
        f"COULD NOT RUN: the log {link.name} is a symbolic link, so the drive it records cannot be "
        "told from a copy of another (reading R5)"
    )


# --- 5. the purposes -----------------------------------------------------------------------------


def with_a_sealed_run(*, seed: bool = True) -> tuple[list[dict], dict]:
    """The entries and groups of a list whose sealed run has entries, units and arguments of its
    own: the stand-in of the sealed truth build, the sealed route's argument and the interval
    seed, and the intervals of the scoring entry's run. With `seed` false, the list of the earlier
    form, whose sealed scoring carries no interval seed and has no intervals entry."""
    sealed = ["--held-out-truth-run", "{run:sealed-truth}"]
    if seed:
        sealed += ["--matrix", SEALED_MATRIX]
        sealed += ["--interval-seed", "{seed:intervals}", "--interval-resamples", "2000"]
    entries = [
        entry("wp2-truth", step("truth_build")),
        entry("sealed-truth", step(SEALED_TRUTH), only="sealed-run"),
        entry(
            "rehearsal-learn",
            step("learn_run", "--units", "{units}"),
            units="scored",
            extra={"sealed-run": sealed},
            after=["wp2-truth"],
        ),
    ]
    if seed:
        entries.append(
            entry(
                "sealed-intervals",
                step(
                    "day_intervals", "--run", "{run:rehearsal-learn}", "--seed", "{seed:intervals}"
                ),
                only="sealed-run",
                after=["rehearsal-learn"],
            )
        )
    units = {"rehearsal": {"scored": ["nrel-before-may-2019"]}, "sealed-run": {"scored": ["OSL_T"]}}
    return entries, units


def test_the_list_of_the_earlier_form_without_the_interval_seed_is_refused(repo, capsys):
    entries, units = with_a_sealed_run(seed=False)
    assert drive(repo, entries, units=units) == 2
    out = capsys.readouterr().out
    assert (
        "FAULT: entry 3 (rehearsal-learn): on the sealed run it scores without --interval-seed"
        in out
    )
    assert drive_logs(repo) == []


def test_the_differences_from_the_run_from_the_tag_are_listed_and_the_rehearsal_passes(
    repo, capsys
):
    entries, units = with_a_sealed_run()
    assert drive(repo, entries, units=units) == 0
    code, lines = run_check(repo, capsys)
    assert code == 0, lines
    assert lines[1:5] == [
        "  difference from the run from the tag: entry 2 (sealed-truth): runs in the sealed-run "
        "only",
        "  difference from the run from the tag: entry 3 (rehearsal-learn): its units, the group "
        "'scored': rehearsal nrel-before-may-2019; sealed-run OSL_T",
        "  difference from the run from the tag: entry 3 (rehearsal-learn): arguments for the "
        "sealed-run only: --held-out-truth-run {run:sealed-truth} --matrix "
        f"{SEALED_MATRIX} --interval-seed {{seed:intervals}} --interval-resamples 2000",
        "  difference from the run from the tag: entry 4 (sealed-intervals): runs in the "
        "sealed-run only",
    ]
    assert [line.split(":")[0] for line in lines[6:8]] == ["  wp2-truth", "  rehearsal-learn"]
    assert ": 2 entries, each exited 0 " in lines[-1]
    assert lines[-1].endswith(
        "; 4 differences from the run from the tag, listed above (reading R6)"
    )
    assert len(lines) == 9


def test_a_sealed_run_that_passed_does_not_stand_in_for_a_rehearsal_that_failed(
    repo, capsys, monkeypatch
):
    entries, units = with_a_sealed_run()
    entries[2]["extra"]["rehearsal"] = ["--exit", "1"]  # fails in the rehearsal only
    assert drive(repo, entries, units=units) == 1
    rehearsal = the_log(repo)
    passing_gate(monkeypatch)
    time.sleep(1.1)  # the two drives' runs of wp2-truth must start in different seconds
    assert run_driver.drive(Path(LIST_PATH), purpose="sealed-run", start_dir=repo) == 0
    assert len(drive_logs(repo)) == 2
    code, lines = run_check(repo, capsys)
    assert code == 1, lines
    assert lines[-1] == f"FAILED: rehearsal-learn ended with exit status 1; {rehearsal.name}"


# --- 6. O2's verdict (reading R7) ----------------------------------------------------------------


def with_o2(*arguments: str) -> list[dict]:
    """Outcome O2's two stand-in steps (`drive_fake.O2_STEP`) after the truth builds they read
    (`drive_fake.o2_entries`, run list rules 10 and 12): `rehearsal-o2` of the main setting with
    `arguments`, then `rehearsal-o2-held`, passing on its one unit."""
    return o2_entries("rehearsal-o2", *arguments)


def drive_o2(repo: Path, entries: list[dict]) -> int:
    """`drive` of a list holding the two O2 steps, on their groups of units."""
    return drive(repo, entries, units=O2_UNITS)


# The check's lines of a drive of `with_o2`: the first line, the list's 6 differences from the run
# from the tag (2 entries of the sealed run only; each O2 step's units and its arguments of the
# sealed run only), the log's line, one line per step logged, then one per O2 step logged.
O2_LINE = 12  # `rehearsal-o2`'s O2 line when its 4 steps are logged (11 with 3)
O2_LINES = 15  # the lines of a drive of `with_o2` in which every step passed


def o2_entry(repo: Path) -> dict:
    """The `entry` line of the O2 step `rehearsal-o2`, the third step of the log."""
    return read_log(the_log(repo))[6]


def o2_said(run: str, over: str, failed: str = "none", undecided: str = "none") -> str:
    return (
        f"  O2 rehearsal-o2: over the units {over}; failed on {failed}; could not decide on "
        f"{undecided} (from {run}/ceiling-report.json; not in this check's exit status, reading "
        "R7)"
    )


@pytest.mark.parametrize(
    ("verdicts", "over", "failed", "undecided"),
    [
        ("unit-a=failed,unit-b=passed", "failed", "unit-a", "none"),
        ("unit-a=passed,unit-b=could not run", "could not run", "none", "unit-b"),
        ("unit-a=passed,unit-b=passed", "passed", "none", "none"),
    ],
    ids=["failed-on-one-unit", "undecided-on-one-unit", "passed"],
)
def test_an_o2_step_prints_its_verdict_and_the_exit_status_stays(
    repo, capsys, verdicts, over, failed, undecided
):
    assert drive_o2(repo, with_o2("--verdicts", verdicts)) == 0
    code, lines = run_check(repo, capsys)
    assert code == 0, lines
    assert lines[O2_LINE] == o2_said(o2_entry(repo)["run"], over, failed, undecided)
    assert lines[-1].startswith("passed: the rehearsal of the run list ")
    assert len(lines) == O2_LINES


FAULTS_HEAD = "own faults (decision 85 (1)): "


def own(unit, orders, days, block=0, in_m=0):
    """The text of one unit's counts: each by what it holds (`ceiling_summary.o2_with_faults`)."""
    return (
        f"{unit}: {orders} orders broken (both products, every day; {block} of them block orders "
        f"on M days, in O2's pass), {days} days not built ({in_m} of them M days, in O2's pass)"
    )


@pytest.mark.parametrize(
    ("arguments", "faulted"),
    [
        (
            ("--verdicts", "unit-a=passed,unit-b=passed", "--faults", "unit-a=3:1"),
            own("unit-a", 3, 1),
        ),
        (
            (
                "--verdicts",
                "unit-a=failed,unit-b=passed",
                "--faults",
                "unit-a=4:1:2:1,unit-b=5:0",
            ),
            f"{own('unit-a', 4, 1, 2, 1)}; {own('unit-b', 5, 0)}",
        ),
    ],
    ids=["one-unit", "block-order-broken-on-an-m-day-verdict-failed"],
)
def test_an_o2_step_prints_its_own_faults_beside_its_verdict_and_the_exit_status_stays(
    repo, capsys, arguments, faulted
):
    """Decision 88 (3d): the faults of each unit (decision 85 (1)) are shown beside the verdict;
    no verdict and no exit status changes."""
    assert drive_o2(repo, with_o2(*arguments)) == 0
    code, lines = run_check(repo, capsys)
    assert code == 0, lines
    assert lines[O2_LINE + 1] == f"  O2 rehearsal-o2: {FAULTS_HEAD}{faulted}"
    assert lines[-1].startswith("passed: the rehearsal of the run list ")
    assert len(lines) == O2_LINES + 1


def test_an_o2_step_with_no_faults_prints_no_faults_line(repo, capsys):
    assert drive_o2(repo, with_o2("--verdicts", "unit-a=passed,unit-b=failed")) == 0
    code, lines = run_check(repo, capsys)
    assert code == 0, lines
    assert not [line for line in lines if "own faults" in line], lines
    assert len(lines) == O2_LINES


@pytest.mark.parametrize(
    ("arguments", "unreadable"),
    [
        (("--no-faults",), "unit-a"),
        (("--faults", "unit-a=bad"), "unit-a"),
        (("--faults", "unit-a=noblock"), "unit-a"),
        (("--faults", "unit-a=nomdays"), "unit-a"),
    ],
    ids=["no-record", "not-counts", "no-block-count", "no-days-in-m-count"],
)
def test_an_o2_step_whose_faults_record_is_not_readable_says_so_and_the_exit_status_stays(
    repo, capsys, arguments, unreadable
):
    assert drive_o2(repo, with_o2(*arguments)) == 0
    code, lines = run_check(repo, capsys)
    assert code == 0, lines
    assert lines[O2_LINE + 1] == (
        f"  O2 rehearsal-o2: {FAULTS_HEAD}the faults record is missing or not readable for "
        f"{unreadable}"
    )
    assert lines[-1].startswith("passed: the rehearsal of the run list ")
    assert len(lines) == O2_LINES + 1


@pytest.mark.parametrize(
    ("verdicts", "over", "undecided", "no_day"),
    [
        ("unit-a=no day to check,unit-b=passed,unit-c=passed", "passed", "none", "unit-a"),
        (
            "unit-a=no day to check,unit-b=no day to check",
            "no day to check",
            "none",
            "unit-a, unit-b",
        ),
        ("unit-a=no day to check,unit-b=could not run", "could not run", "unit-b", "unit-a"),
    ],
    ids=["beside-two-passed", "every-unit", "beside-one-undecided"],
)
def test_an_o2_step_names_its_units_with_no_day_to_check(
    repo, capsys, verdicts, over, undecided, no_day
):
    """The owner's decision of 19 November 2026: a unit with no day to check is named, and counts
    as neither passed nor failed; with no unit judged the verdict over the units is no day to
    check (decision 79 (3))."""
    assert drive_o2(repo, with_o2("--verdicts", verdicts)) == 0
    code, lines = run_check(repo, capsys)
    assert code == 0, lines
    said = o2_said(o2_entry(repo)["run"], over, "none", undecided)
    assert lines[O2_LINE] == said.replace(" (from ", f"; no day to check on {no_day} (from ")
    assert lines[-1].startswith("passed: the rehearsal of the run list ")


def _without_hypotheses_days(real):
    """`ceiling_fakes.unit_cars` with the unit nrel-before-may-2019 made of nine days of the kinds
    1 to 4 alone (`ceiling_fakes`: none meets T4's hypotheses), as on the development run of 19
    November 2026 (0 of 9 days)."""

    def unit_cars(unit, scale=1.0, layout=None):
        if unit != "nrel-before-may-2019":
            return real(unit, scale, layout)
        days = [k for k in range(12) if k % 5][:9]
        return [car for k in days for car in ceiling_fakes.day_cars("A", k)]

    return unit_cars


def test_o2_end_to_end_three_units_one_with_no_day_to_check(tmp_path, monkeypatch):
    """The real O2 command on three made-up units, one with no M day meeting T4's hypotheses and
    two whose days pass: exit 0; in the results file the verdict over the units is passed, the
    unit named under `no_day_to_check`; the report says so; and the O2 line this check prints
    from that file, through `o2_line`, names it."""
    monkeypatch.setattr(
        ceiling_fakes, "unit_cars", _without_hypotheses_days(ceiling_fakes.unit_cars)
    )
    run, tables = ceiling_fakes.setup_build(tmp_path, monkeypatch)
    top = tmp_path / "top"
    label = "rehearsal-o2"
    folder = f"runs.noindex/20261119T182300Z-{label}"
    (top / folder).mkdir(parents=True)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(top / folder))
    arguments = ["--run", str(run), "--tables", str(tables), "--workers", "1", "--every-day"]
    assert ceiling_report.main(arguments) == 0
    document = json.loads((top / folder / ceiling_report.RESULTS_NAME).read_text())
    every = document["every_day"]
    units = every["units"]
    none, *others = ceiling_report.UNITS
    assert none == "nrel-before-may-2019"
    assert units[none]["days_in_m"] == 9
    assert units[none]["t4_check"]["days_meeting_hypotheses"] == 0
    assert units[none]["o2_pass"] == "no day to check"
    for unit in others:
        assert units[unit]["t4_check"]["days_checked"] > 0
        assert units[unit]["o2_pass"] == "passed"
    assert every["pass_over_units"] == "passed"
    assert every["no_day_to_check"] == [none]
    assert document["failures"] == []
    text = (top / folder / ceiling_report.TEXT_NAME).read_text().splitlines()
    assert (
        f"  O2's pass per unit: {none} no day to check, {others[0]} passed, {others[1]} passed; "
        f"over the units: passed (no day to check on {none})"
    ) in text
    assert text[-1] == f"all checks held (no day to check on {none})"
    outputs = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((top / folder).iterdir())
    }
    manifest = {
        "commit": ceiling_fakes.COMMIT,
        "command": ["python", "-m", "fleetreserve.ceiling_report", *arguments],
        "start": "s",
        "end": "e",
        "exit_code": 0,
        "outputs": outputs,
    }
    (top / folder / "manifest.json").write_text(json.dumps(manifest))
    entry_line = {
        "exit_status": 0,
        "timed_out": False,
        "manifest_sha256": hashlib.sha256(
            (top / folder / "manifest.json").read_bytes()
        ).hexdigest(),
        "run": folder,
    }
    said, fault = check.o2_line(top, label, entry_line)
    assert fault is None, said
    assert said == (
        f"O2 {label}: over the units passed; failed on none; could not decide on none; no day to "
        f"check on {none} (from {folder}/ceiling-report.json; not in this check's exit status, "
        "reading R7)"
    )


def _o2_day(unit, date, in_m):
    """A made-up counting day of `unit` meeting T4's hypotheses, in M with V_hind = V_caus = 2 or
    outside M with no block value (not checked)."""
    values, outcomes = dict.fromkeys(VALUE_KEYS), dict.fromkeys(VALUE_KEYS, "not solved")
    if in_m:
        for key in ("block_hind", "block_caus", "block_aff"):
            values[key], outcomes[key] = 2.0, "optimal"
    return DayResult(
        unit, "A", date, 3, 6 if in_m else 9, 2, False, True, in_m, None, values, outcomes
    )


@pytest.mark.parametrize(
    ("days_of_a", "counting"),
    [([], 0), ([_o2_day("unit-a", "d1", False), _o2_day("unit-a", "d2", False)], 2)],
    ids=["no-counting-day", "counting-days-none-in-m"],
)
def test_o2_line_names_a_unit_with_no_m_day_beside_a_passed_unit(
    tmp_path, monkeypatch, days_of_a, counting
):
    """The owner's decision of 19 November 2026 (decision 79 (3)): a unit with no M day (no
    counting day, or every counting day outside M) beside a passed unit, in a results file
    written by the real `ceiling_report.every_day_summary`: through `o2_line` the verdict over
    the units is passed and the unit is named as having no day to check, never could not decide.
    The run's commit is made up, its check replaced (as `ceiling_fakes.setup_build` does)."""
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    results = [*days_of_a, _o2_day("unit-b", "d1", True)]
    every = ceiling_report.every_day_summary(results, ["unit-a", "unit-b"], {})
    assert every["units"]["unit-a"]["counting_days"] == counting
    assert every["units"]["unit-a"]["days_in_m"] == 0
    top, label = tmp_path / "top", "rehearsal-o2"
    folder = f"runs.noindex/20261120T090000Z-{label}"
    (top / folder).mkdir(parents=True)
    report = top / folder / ceiling_report.RESULTS_NAME
    report.write_text(json.dumps({"every_day": every, "failures": []}, sort_keys=True))
    manifest = {
        "commit": ceiling_fakes.COMMIT,
        "command": ["python", "-m", "fleetreserve.ceiling_report", "--every-day"],
        "start": "s",
        "end": "e",
        "exit_code": 0,
        "outputs": {report.name: hashlib.sha256(report.read_bytes()).hexdigest()},
    }
    (top / folder / "manifest.json").write_text(json.dumps(manifest))
    entry_line = {
        "exit_status": 0,
        "timed_out": False,
        "manifest_sha256": hashlib.sha256(
            (top / folder / "manifest.json").read_bytes()
        ).hexdigest(),
        "run": folder,
    }
    said, fault = check.o2_line(top, label, entry_line)
    assert fault is None, said
    assert said == (
        f"O2 {label}: over the units passed; failed on none; could not decide on none; no day to "
        f"check on unit-a (from {folder}/ceiling-report.json; not in this check's exit status, "
        "reading R7)"
    )


@pytest.mark.parametrize(
    ("arguments", "why"),
    [
        (("--no-verdict",), "it holds no every_day with its units' O2 passes"),
        (("--no-file",), "the file is missing"),
    ],
    ids=["no-verdict", "no-file"],
)
def test_an_o2_step_that_exited_0_with_no_readable_verdict_fails(repo, capsys, arguments, why):
    assert drive_o2(repo, with_o2(*arguments)) == 0
    code, lines = run_check(repo, capsys)
    assert code == 1, lines
    said = (
        f"O2 rehearsal-o2: exited 0, but {o2_entry(repo)['run']}/ceiling-report.json holds no "
        f"readable verdict: {why}"
    )
    assert lines[O2_LINE] == f"  {said}"
    assert lines[-1] == f"FAILED: {said}; {the_log(repo).name}"


@pytest.mark.parametrize(
    ("verdicts", "over"),
    [
        ("unit-a=failed,unit-b=passed", "passed"),
        ("unit-a=passed", "failed"),
        ("unit-a=passed,unit-b=could not run", "passed"),
        ("unit-a=failed", "could not run"),
    ],
    ids=["passed-over-a-failed-unit", "failed-over-passes", "passed-over-undecided", "undecided"],
)
def test_an_o2_pass_over_units_other_than_the_verdict_over_its_units_fails(
    repo, capsys, verdicts, over
):
    """The results file's `pass_over_units` is not `ceiling_summary.pass_over` of its units'
    passes (the file verified by its manifest): no readable verdict, the check fails, naming it."""
    assert drive_o2(repo, with_o2("--verdicts", verdicts, "--over", over)) == 0
    code, lines = run_check(repo, capsys)
    assert code == 1, lines
    said = (
        f"O2 rehearsal-o2: exited 0, but {o2_entry(repo)['run']}/ceiling-report.json holds no "
        f"readable verdict: its pass_over_units {over!r} is not the verdict over its units' passes"
    )
    assert lines[O2_LINE] == f"  {said}"
    assert lines[-1] == f"FAILED: {said}; {the_log(repo).name}"


def test_an_o2_verdict_edited_after_the_run_is_no_readable_verdict(repo, capsys):
    assert drive_o2(repo, with_o2("--verdicts", "unit-a=failed")) == 0
    run = o2_entry(repo)["run"]
    report = repo / run / "ceiling-report.json"
    report.write_text(report.read_text().replace('"failed"', '"passed"'))
    code, lines = run_check(repo, capsys)
    assert code == 1, lines
    assert lines[-1].startswith(
        f"FAILED: O2 rehearsal-o2: exited 0, but {run}/ceiling-report.json holds no readable "
        "verdict: the file has changed since the run (its hash differs); "
    )


def test_an_o2_fault_is_named_beside_a_later_step_that_failed(repo, capsys):
    entries = [*with_o2("--no-file"), entry("rehearsal-bad", step("learn_run", "--exit", "1"))]
    assert drive_o2(repo, entries) == 1
    code, lines = run_check(repo, capsys)
    assert code == 1, lines
    assert lines[-1] == (
        f"FAILED: O2 rehearsal-o2: exited 0, but {o2_entry(repo)['run']}/ceiling-report.json "
        "holds no readable verdict: the file is missing; rehearsal-bad ended with exit status 1; "
        f"{the_log(repo).name}"
    )


def test_an_o2_step_that_exited_1_has_no_verdict_read_and_fails_as_any_step(repo, capsys):
    assert drive_o2(repo, with_o2("--verdicts", "unit-a=passed", "--exit", "1")) == 1
    code, lines = run_check(repo, capsys)
    assert code == 1, lines
    assert lines[O2_LINE - 1] == (  # the held step after it never ran
        "  O2 rehearsal-o2: no verdict read, the step did not exit 0 with no timeout (exit "
        "status 1)"
    )
    assert lines[-1] == f"FAILED: rehearsal-o2 ended with exit status 1; {the_log(repo).name}"


def test_an_o2_step_is_the_every_day_command_of_ceiling_report():
    assert check.is_o2(step(O2_MODULE, "--every-day"))
    assert check.is_o2(step(O2_MODULE, "--rho", "0", "--every"))  # argparse reads it as the option
    assert not check.is_o2(step(O2_MODULE))  # the block sample of WP2 holds no O2 verdict
    assert not check.is_o2(step(O2_MODULE, "--", "--every-day"))
    assert not check.is_o2(step("learn_run", "--every-day"))


def test_the_o2_reader_opens_its_results_file_only_inside_run_verify(repo, capsys):
    assert drive_o2(repo, with_o2("--verdicts", "unit-a=failed")) == 0
    report = repo / o2_entry(repo)["run"] / "ceiling-report.json"
    with opened([report], run_verify) as seen:
        code, lines = run_check(repo, capsys)
    assert code == 0, lines
    assert lines[O2_LINE] == o2_said(o2_entry(repo)["run"], "failed", "unit-a")
    assert seen, "the results file was never opened"
    assert all(inside for _, inside in seen), seen


def test_the_o2_reader_never_follows_a_link_into_the_sealed_folder(repo):
    """The O2 step's results file made a symbolic link to a file of the same bytes in the sealed
    folder: the reader alone (`o2_line`) never opens it, and the step fails, naming the link."""
    assert drive_o2(repo, with_o2("--verdicts", "unit-a=failed")) == 0
    line = o2_entry(repo)
    report = repo / line["run"] / "ceiling-report.json"
    sealed = repo / "data" / "heldout" / "made-up" / "ceiling-report.json"
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(report.read_bytes())
    report.unlink()
    report.symlink_to(sealed)
    with watch(repo / "data" / "heldout") as seen:
        text, fault = check.o2_line(repo, "rehearsal-o2", line)
    assert seen == []
    assert fault == text
    assert text == (
        f"O2 rehearsal-o2: exited 0, but {line['run']}/ceiling-report.json holds no readable "
        f"verdict: the file cannot be read: {line['run']}/ceiling-report.json is a symbolic link"
    )


# --- 7. the one reader (`fleetreserve.plain_read`) -----------------------------------------------


def test_a_manifest_linked_to_a_sealed_file_is_never_opened_and_could_not_run(repo, capsys):
    """A passing rehearsal whose first run's manifest is then a symbolic link to a made-up sealed
    file of the same bytes: the check never opens it (a read would pass), and could not run,
    naming the link."""
    assert drive(repo, two_passing()) == 0
    run = read_log(the_log(repo))[2]["run"]
    manifest = repo / run / "manifest.json"
    target = repo / "data" / "heldout" / "asr" / "made-up.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(target)
    code, lines = run_check(repo, capsys)
    assert code == 2, lines
    assert lines[-1].startswith("COULD NOT RUN: wp2-truth: the manifest of "), lines
    assert f"{run}/manifest.json is a symbolic link" in lines[-1], lines


def test_a_failed_steps_run_through_dotdot_into_the_sealed_folder_could_not_run(repo, capsys):
    """A failed step whose run folder is written by hand as `runs.noindex/../data/heldout/asr`
    (the log's hash recomputed), with its manifest copied there: the check reads nothing there
    (a read would confirm the failure, exit 1) and could not run, naming the path."""
    failing = entry("rehearsal-bad", step("learn_run", "--exit", "1"))
    assert drive(repo, [entry("rehearsal-ok", step("truth_build")), failing]) == 1
    path = the_log(repo)
    lines = path.read_bytes().splitlines(keepends=True)
    k = max(i for i, line in enumerate(lines) if json.loads(line).get("record") == "entry")
    record = json.loads(lines[k])
    sealed = repo / "data" / "heldout" / "asr"
    sealed.mkdir(parents=True)
    (sealed / "manifest.json").write_bytes((repo / record["run"] / "manifest.json").read_bytes())
    record["run"] = "runs.noindex/../data/heldout/asr"
    lines[k] = (json.dumps(record, sort_keys=True) + "\n").encode()
    path.write_bytes(rehash(lines))
    code, out = run_check(repo, capsys)
    assert code == 2, out
    assert out[-1].startswith(
        "COULD NOT RUN: rehearsal-bad: the manifest of runs.noindex/../data/heldout/asr cannot be "
        "read: "
    ), out
    assert "not a plain path" in out[-1], out


def test_a_drive_logs_folder_that_is_a_link_is_never_listed_and_could_not_run(repo, capsys):
    """A passing rehearsal whose drive logs' folder is moved into the sealed folder and linked
    back: the check never lists it (a listing would pass), and could not run, naming the link."""
    assert drive(repo, two_passing()) == 0
    folder = repo / "runs.noindex" / "drives"
    moved = repo / "data" / "heldout" / "drives"
    moved.parent.mkdir(parents=True)
    folder.rename(moved)
    folder.symlink_to(moved)
    code, lines = run_check(repo, capsys)
    assert code == 2, lines
    assert lines[-1] == (
        "COULD NOT RUN: the folder of the drive logs is not listed: runs.noindex/drives is a "
        "symbolic link"
    ), lines
