"""The driver (`fleetreserve.run_driver`) after an entry that failed, on made-up run lists of
made-up steps in a throwaway git repository (`drive_fake`).

1. The sealed run: an entry that exits with a status other than 0 is logged as failed with its
   reason and the drive goes on; a later entry that needs its run (a `{run:<label>}` or an `after`
   label, also through an entry that was not run) is not started and is logged as not run; every
   other later entry runs. The drive's exit status is 1, its end line and its last printed line
   name every failed and every not-run entry, and the end line says it did not complete.
2. A timed-out entry is a failure like any other.
3. The rehearsal, with the same failure, stops after the entry as before (no `failed` or
   `not_run` line, no such field in its end line).
4. Stops that stay stops in the sealed run: a commit or tree changed between steps, an entry
   interrupted (the way an operator stops a drive), an entry the recorder refused; the steps not
   reached are named as not run in the end line.
5. A failing entry that runs the sealed truth build (by its command, not its place) stops the sealed
   drive at once, with the stop line of the rehearsal's stop; a failing later entry goes on.
6. A sealed-run drive in which every entry passes exits 0, with empty lists and no problems line.
7. Nothing sealed is open before the first entry that runs the sealed truth build. On the real run
   list's order (the development builds, then the sealed builds with no `after`, then the steps
   that read them) a failed or timed-out entry before the first sealed build stops the sealed
   drive at once, before the sealed build opens anything; after the opening a failing entry still
   goes on; the rehearsal is unchanged.
8. The second sealed truth build (held setting) parses the bytes the first parsed. After the first
   passed, a failure of the second is logged as failed and the drive goes on: exactly the steps
   that need its run are not run, every other step runs, the drive exits 1 (on the committed
   list's order and dependencies, and on a drive of the stand-ins). A failure of the first, or an
   interruption of the second, still stops the drive.
"""

import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path

import pytest
from drive_fake import (
    LIST_PATH,
    O2_UNITS,
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
    run_folders,
    step,
)

from fleetreserve import cold_start_gate, run_driver, run_list

REHEARSE = ["--purpose", "rehearsal"]
SEAL = ["--purpose", "sealed-run"]


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    """Drop every GIT_* variable and git's global and system settings for the test."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    no_start_check(monkeypatch)
    monkeypatch.chdir(root)

    def gate(top):  # the sealed run's gate made to pass, as at the tag
        return git(top, "rev-parse", "HEAD").strip()

    monkeypatch.setattr(cold_start_gate, "require_gate", gate)
    return root


def the_log(repo) -> tuple[list[dict], bytes]:
    (path,) = drive_logs(repo)
    return read_log(path), path.read_bytes()


def assert_hash_recomputes(data: bytes) -> None:
    lines = data.splitlines(keepends=True)
    end = json.loads(lines[-1])
    claimed = end.pop("log_sha256")
    own = (json.dumps(end, sort_keys=True) + "\n").encode()
    assert claimed == hashlib.sha256(b"".join(lines[:-1]) + own).hexdigest()


def kinds(records: list[dict]) -> list[str]:
    return [r["record"] for r in records]


def last_line(capsys) -> str:
    return [line for line in capsys.readouterr().out.splitlines() if line.strip()][-1]


def with_a_failure(failing: list[str]) -> dict:
    """A list in which `failing` is the command of the second entry, `bad`; `indep` needs no run,
    `dep` needs the run of `bad`, `dep-after` follows `bad` alone, `dep-of-dep` needs the run of
    `dep` (an entry that is not run)."""
    return document(
        [
            entry("ok", step("truth_build")),
            entry("bad", failing),
            entry("indep", step("compare_run", "--truth-run", "{run:ok}"), after=["ok"]),
            entry("dep", step("learn_run", "--truth-run", "{run:bad}"), after=["bad"]),
            entry("dep-after", step("compare_run"), after=["bad"]),
            entry("dep-of-dep", step("learn_run", "--truth-run", "{run:dep}"), after=["dep"]),
        ]
    )


# --- 1. a failed entry does not stop the sealed run ----------------------------------------------


def test_a_failed_entry_is_logged_and_the_sealed_drive_goes_on_with_what_does_not_need_it(
    repo, capsys
):
    commit_list(repo, with_a_failure(step("learn_run", "--exit", "1")))
    assert run_driver.main([LIST_PATH, *SEAL]) == 1
    records, data = the_log(repo)
    assert kinds(records) == [
        "start",
        "begin",
        "entry",  # ok
        "begin",
        "entry",  # bad
        "failed",
        "begin",
        "entry",  # indep
        "not_run",  # dep
        "not_run",  # dep-after
        "not_run",  # dep-of-dep
        "end",
    ]
    by_label = {r["label"]: r for r in records if r["record"] == "entry"}
    assert [by_label[n]["exit_status"] for n in ("ok", "bad", "indep")] == [0, 1, 0]
    failed = records[5]
    assert failed == {
        "record": "failed",
        "index": 1,
        "label": "bad",
        "reason": "the run ended with exit status 1",
    }
    not_run = [r for r in records if r["record"] == "not_run"]
    assert [(r["index"], r["label"], r["reason"]) for r in not_run] == [
        (3, "dep", "not run: needs bad, which failed"),
        (4, "dep-after", "not run: needs bad, which failed"),
        (5, "dep-of-dep", "not run: needs dep, which was not run"),
    ]
    end = records[-1]
    assert (end["completed"], end["entries_run"], end["exit_status"]) == (False, 3, 1)
    assert [f["label"] for f in end["failed"]] == ["bad"]
    assert [(n["label"], n["reason"]) for n in end["not_run"]] == [
        (r["label"], r["reason"]) for r in not_run
    ]
    assert sorted(p.name.split("-", 1)[1] for p in run_folders(repo)) == ["bad", "indep", "ok"]
    assert_hash_recomputes(data)
    printed = capsys.readouterr().out
    assert "FAILED bad: the run ended with exit status 1; the drive goes on" in printed
    assert "dep: not run: needs bad, which failed" in printed


def test_the_last_printed_line_of_a_drive_with_a_failure_names_every_problem(repo, capsys):
    commit_list(repo, with_a_failure(step("learn_run", "--exit", "3")))
    assert run_driver.main([LIST_PATH, *SEAL]) == 1
    line = last_line(capsys)
    assert line == (
        "drive with FAILED: bad and NOT RUN: dep, dep-after, dep-of-dep: this drive did not pass"
    )
    records, _ = the_log(repo)
    assert records[-1]["record"] == "end"
    assert records[-1]["exit_status"] != 0
    assert records[-1]["completed"] is False


def test_two_failures_are_both_named_and_a_later_entry_needing_both_is_not_run(repo, capsys):
    listed = document(
        [
            entry("bad-one", step("truth_build", "--exit", "1")),
            entry("bad-two", step("learn_run", "--exit", "2")),
            entry(
                "both",
                step("compare_run", "--a", "{run:bad-one}", "--b", "{run:bad-two}"),
                after=["bad-one", "bad-two"],
            ),
        ]
    )
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *SEAL]) == 1
    records, _ = the_log(repo)
    assert [f["label"] for f in records[-1]["failed"]] == ["bad-one", "bad-two"]
    (both,) = records[-1]["not_run"]
    assert both["reason"] == ("not run: needs bad-one, which failed; needs bad-two, which failed")
    assert last_line(capsys).startswith("drive with FAILED: bad-one, bad-two and NOT RUN: both")


# --- 2. a timed-out entry is a failure -----------------------------------------------------------


def test_a_timed_out_entry_is_a_failure_and_the_sealed_drive_goes_on(repo):
    commit_list(repo, with_a_failure(step("learn_run", "--sleep", "30")))
    listed = json.loads((repo / LIST_PATH).read_text())
    listed["entries"][1]["max_seconds"] = 1
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *SEAL]) == 1
    records, data = the_log(repo)
    assert kinds(records).count("entry") == 3
    bad = next(r for r in records if r["record"] == "entry" and r["label"] == "bad")
    assert bad["timed_out"] is True
    assert bad["exit_status"] == 128 + signal.SIGTERM
    (failed,) = [r for r in records if r["record"] == "failed"]
    assert failed["label"] == "bad"
    assert "time limit" in failed["reason"]
    assert [n["label"] for n in records[-1]["not_run"]] == ["dep", "dep-after", "dep-of-dep"]
    assert records[-1]["exit_status"] == 1
    assert "stop" not in kinds(records)
    assert_hash_recomputes(data)


# --- 3. the rehearsal stops as before ------------------------------------------------------------


def test_the_rehearsal_with_the_same_failure_stops_after_it(repo, capsys):
    commit_list(repo, with_a_failure(step("learn_run", "--exit", "1")))
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 1
    (path,) = drive_logs(repo)
    records = read_log(path)
    assert kinds(records) == ["start", "begin", "entry", "begin", "entry", "stop", "end"]
    assert records[5]["label"] == "bad"
    end = records[-1]
    assert (end["completed"], end["entries_run"], end["exit_status"]) == (False, 2, 1)
    assert "failed" not in end
    assert "not_run" not in end
    assert last_line(capsys).startswith("drive ended with exit status 1 after ")
    assert sorted(p.name.split("-", 1)[1] for p in run_folders(repo)) == ["bad", "ok"]


def test_the_rehearsal_with_a_timeout_stops_after_it(repo):
    commit_list(repo, with_a_failure(step("learn_run", "--sleep", "30")))
    listed = json.loads((repo / LIST_PATH).read_text())
    listed["entries"][1]["max_seconds"] = 1
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 1
    records, _ = the_log(repo)
    assert kinds(records) == ["start", "begin", "entry", "begin", "entry", "stop", "end"]
    assert "time limit" in records[5]["reason"]


# --- 4. stops that stay stops --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "reason"),
    [
        ("--commit", "the commit changed since the drive started"),
        ("--dirty", "the git tree has uncommitted changes since the drive started"),
    ],
)
def test_a_changed_commit_or_tree_still_stops_the_sealed_drive(repo, flag, reason):
    listed = document(
        [
            entry("first", step("truth_build")),
            entry("bad", step("learn_run", "--exit", "1")),
            entry("change", step("compare_run", flag)),
            entry("next", step("compare_run")),
            entry("last", step("learn_run")),
        ]
    )
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *SEAL]) == 2
    records, data = the_log(repo)
    assert kinds(records) == [
        "start",
        "begin",
        "entry",  # first
        "begin",
        "entry",  # bad
        "failed",
        "begin",
        "entry",  # change
        "stop",
        "end",
    ]
    stop = records[8]
    assert (stop["index"], stop["label"]) == (3, "next")
    assert stop["reason"].startswith(reason)
    end = records[-1]
    assert (end["completed"], end["entries_run"], end["exit_status"]) == (False, 3, 2)
    assert [f["label"] for f in end["failed"]] == ["bad"]
    assert [n["label"] for n in end["not_run"]] == ["next", "last"]
    assert all(n["reason"].startswith("not run: ") for n in end["not_run"])
    assert len(run_folders(repo)) == 3
    assert_hash_recomputes(data)


def test_an_interrupted_entry_still_stops_the_sealed_drive(repo, capsys):
    listed = document(
        [
            entry("ok", step("truth_build")),
            entry("cut", step("learn_run", "--term-parent"), after=["ok"]),
            entry("next", step("compare_run")),
        ]
    )
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *SEAL]) != 0
    records, data = the_log(repo)
    assert kinds(records) == ["start", "begin", "entry", "begin", "entry", "stop", "end"]
    cut = records[4]
    assert cut["interrupted"] is not None
    assert "interrupted" in records[5]["reason"]
    end = records[-1]
    assert end["completed"] is False
    assert end["exit_status"] != 0
    assert [n["label"] for n in end["not_run"]] == ["next"]
    assert "failed" not in kinds(records)
    assert_hash_recomputes(data)
    assert "NOT RUN: next" in last_line(capsys)


def test_a_step_the_recorder_refuses_still_stops_the_sealed_drive(repo):
    listed = document(
        [entry("locked", step("truth_build")), entry("never", step("learn_run"))],
    )
    commit_list(repo, listed)
    (repo / "runs.noindex").mkdir()
    with (repo / "runs.noindex" / "heavy.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another heavy run is going
        assert run_driver.main([LIST_PATH, *SEAL]) == 2
    records, data = the_log(repo)
    assert kinds(records) == ["start", "begin", "entry", "stop", "end"]
    assert records[-1]["exit_status"] == 2
    assert [n["label"] for n in records[-1]["not_run"]] == ["never"]
    assert run_folders(repo) == []
    assert_hash_recomputes(data)


# --- 5. a sealed drive that passes ---------------------------------------------------------------


def test_a_sealed_drive_in_which_every_entry_passes_exits_0_with_empty_lists(repo, capsys):
    commit_list(
        repo,
        document([entry("one", step("truth_build")), entry("two", step("learn_run"))]),
    )
    assert run_driver.main([LIST_PATH, *SEAL]) == 0
    records, data = the_log(repo)
    assert kinds(records) == ["start", "begin", "entry", "begin", "entry", "end"]
    end = records[-1]
    assert (end["completed"], end["entries_run"], end["exit_status"]) == (True, 2, 0)
    assert (end["failed"], end["not_run"]) == ([], [])
    assert last_line(capsys).startswith("drive ended with exit status 0 after ")
    assert_hash_recomputes(data)


# --- 5. the sealed truth build stops the drive ---------------------------------------------------


@pytest.mark.parametrize("place", ["first", "after-another-entry"])
def test_a_failing_sealed_truth_build_stops_the_sealed_drive_at_once(repo, capsys, place):
    entries = [
        entry("sealed-truth", step(SEALED_TRUTH, "--exit", "1")),
        entry("later", step("compare_run")),
        entry("later-needs", step("learn_run", "--truth-run", "{run:sealed-truth}")),
    ]
    if place != "first":  # the build is found by its command: the place does not matter
        entries.insert(0, entry("dev-truth", step("truth_build")))
    commit_list(repo, document(entries))
    assert run_driver.main([LIST_PATH, *SEAL]) == 1
    records, data = the_log(repo)
    built = 1 if place == "first" else 2
    assert kinds(records) == ["start", *["begin", "entry"] * built, "stop", "end"]
    assert records[-2] == {
        "record": "stop",
        "index": built - 1,
        "label": "sealed-truth",
        "reason": "the run ended with exit status 1",
    }
    end = records[-1]
    assert (end["completed"], end["entries_run"], end["exit_status"]) == (False, built, 1)
    assert [f["label"] for f in end["failed"]] == ["sealed-truth"]
    assert [n["label"] for n in end["not_run"]] == ["later", "later-needs"]
    assert "failed" not in kinds(records)
    assert len(run_folders(repo)) == built
    assert_hash_recomputes(data)
    assert "STOP after sealed-truth: the run ended with exit status 1" in capsys.readouterr().out


def test_a_failing_later_entry_after_a_passing_sealed_truth_build_still_goes_on(repo):
    listed = document(
        [
            entry("sealed-truth", step(SEALED_TRUTH)),
            entry("bad", step("learn_run", "--exit", "1")),
            entry("indep", step("compare_run")),
        ]
    )
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *SEAL]) == 1
    records, _ = the_log(repo)
    assert kinds(records) == ["start", *["begin", "entry"] * 2, "failed", "begin", "entry", "end"]
    assert [r["label"] for r in records if r["record"] == "entry"] == [
        "sealed-truth",
        "bad",
        "indep",
    ]
    assert records[-1]["not_run"] == []


# --- 7. nothing sealed is open before the first sealed truth build -------------------------------

REAL_LIST = Path(__file__).resolve().parents[2] / "run-lists" / "run-list.json"
REAL_ORDER = ["dev-truth", "dev-truth-held", "sealed-truth", "sealed-truth-held"]


def real_order(failing: str, *arguments: str, max_seconds: int | None = None) -> dict:
    """The order of the committed list's first entries (`o2_entries`: the development builds, the
    sealed builds with no `after`, then the two O2 steps), the entry `failing` given `arguments`."""
    entries = o2_entries("o2")
    for each in entries:
        if each["label"] == failing:
            each["command"] = [*each["command"], *arguments]
            each["max_seconds"] = max_seconds or each["max_seconds"]
    return document(entries, units=O2_UNITS)


def test_the_committed_list_puts_the_sealed_builds_after_the_development_ones_no_after():
    steps = run_list.instantiate(run_list.load(REAL_LIST), "sealed-run")
    labels = [step.label for step in steps]
    assert labels[: len(REAL_ORDER)] == REAL_ORDER
    opening = run_driver._first_opening(steps)
    assert opening == labels.index("sealed-truth")
    assert [steps[k].after for k in range(len(REAL_ORDER))] == [()] * len(REAL_ORDER)


@pytest.mark.parametrize("failing", ["dev-truth", "dev-truth-held"])
@pytest.mark.parametrize("how", ["exit status", "time limit"])
def test_a_failure_before_the_first_sealed_build_stops_the_sealed_drive_before_it_opens_anything(
    repo, capsys, failing, how
):
    if how == "exit status":
        listed = real_order(failing, "--exit", "1")
    else:
        listed = real_order(failing, "--sleep", "30", max_seconds=1)
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *SEAL]) == 1
    records, data = the_log(repo)
    built = ["dev-truth"] if failing == "dev-truth" else ["dev-truth", "dev-truth-held"]
    assert kinds(records) == ["start", *["begin", "entry"] * len(built), "stop", "end"]
    assert [r["label"] for r in records if r["record"] == "entry"] == built
    assert records[-2]["label"] == failing
    assert "failed" not in kinds(records)
    ran = sorted(p.name.split("-", 1)[1] for p in run_folders(repo))
    assert ran == sorted(built)
    assert not any("sealed" in name for name in ran)  # the sealed build opened nothing
    end = records[-1]
    assert (end["completed"], end["entries_run"], end["exit_status"]) == (False, len(built), 1)
    assert [f["label"] for f in end["failed"]] == [failing]
    assert [n["label"] for n in end["not_run"]] == [
        e["label"] for e in listed["entries"] if e["label"] not in built
    ]
    assert_hash_recomputes(data)
    assert f"STOP after {failing}: " in capsys.readouterr().out


def test_an_interrupted_entry_before_the_first_sealed_build_stops_the_sealed_drive(repo):
    commit_list(repo, real_order("dev-truth", "--term-parent"))
    assert run_driver.main([LIST_PATH, *SEAL]) != 0
    records, _ = the_log(repo)
    assert kinds(records) == ["start", "begin", "entry", "stop", "end"]
    assert not any("sealed" in p.name for p in run_folders(repo))


def test_a_failure_after_the_first_sealed_build_goes_on_on_the_committed_lists_order(repo):
    commit_list(repo, real_order("o2", "--exit", "1"))
    assert run_driver.main([LIST_PATH, *SEAL]) == 1
    records, _ = the_log(repo)
    assert "stop" not in kinds(records)
    assert [r["label"] for r in records if r["record"] == "entry"] == [
        "dev-truth",
        "dev-truth-held",
        "sealed-truth",
        "sealed-truth-held",
        "o2",
        "o2-held",
    ]
    (failed,) = [r for r in records if r["record"] == "failed"]
    assert failed["label"] == "o2"
    assert records[-1]["not_run"] == []


def test_the_rehearsal_stops_after_a_failed_development_build_as_before(repo):
    commit_list(repo, real_order("dev-truth", "--exit", "1"))
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 1
    records, _ = the_log(repo)
    assert kinds(records) == ["start", "begin", "entry", "stop", "end"]
    assert "failed" not in records[-1] and "not_run" not in records[-1]


def test_a_failure_in_a_list_with_no_sealed_truth_build_goes_on_as_before(repo):
    """No entry opens a sealed file, so there is no opening to stop before (the list of test 1)."""
    commit_list(repo, with_a_failure(step("learn_run", "--exit", "1")))
    assert run_driver.main([LIST_PATH, *SEAL]) == 1
    records, _ = the_log(repo)
    assert "stop" not in kinds(records) and "failed" in kinds(records)


# --- 8. the second sealed truth build -------------------------------------------------------------

HELD_BUILD = "sealed-truth-held"
NEEDING_THE_HELD_BUILD = ["learn-lgbm_tail-held", "intervals-lgbm_tail-held", "o2-held-rho0"]


class ListLog:
    """The drive's log as a list of the records written (the driver writes only `write`)."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, record: dict) -> None:
        self.records.append(record)


def fake_record(
    label: str, *, exit_status=0, timed_out=False, interrupted=None, manifest_sha256="0" * 64
) -> dict:
    return {
        "record": "entry",
        "label": label,
        "run": f"runs.noindex/20261122T000000Z-{label}",
        "exit_status": exit_status,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "manifest_sha256": manifest_sha256,
        "recorder_status": exit_status,
    }


def drive_committed_order(monkeypatch, tmp_path, outcomes: dict) -> tuple:
    """`_run_entries` on the committed list's sealed-run steps (their order, `after` and
    `{run:<label>}`), each step made to end as `outcomes` says (a label -> keyword arguments of
    `fake_record`; every other step passes); (the result, the labels run, the log's records)."""
    steps = run_list.instantiate(run_list.load(REAL_LIST), "sealed-run")
    ran: list[str] = []

    def run_entry(top, k, entry, command):
        ran.append(entry.label)
        return fake_record(entry.label, **outcomes.get(entry.label, {}))

    monkeypatch.setattr(run_driver, "run_entry", run_entry)
    monkeypatch.setattr(run_driver, "_changed", lambda top, commit: None)
    log = ListLog()
    result = run_driver._run_entries(tmp_path, "0" * 40, steps, log, "sealed-run")
    return result, ran, log.records, [step.label for step in steps]


@pytest.mark.parametrize("how", ["exit status", "time limit"])
def test_a_failed_second_sealed_build_goes_on_and_loses_only_the_steps_needing_it_committed_list(
    monkeypatch, tmp_path, how
):
    failure = (
        {"exit_status": 1} if how == "exit status" else {"exit_status": 124, "timed_out": True}
    )
    result, ran, records, labels = drive_committed_order(
        monkeypatch, tmp_path, {HELD_BUILD: failure}
    )
    status, count, failed, not_run = result
    assert status == 1
    assert [f["label"] for f in failed] == [HELD_BUILD]
    assert sorted(n["label"] for n in not_run) == sorted(NEEDING_THE_HELD_BUILD)
    assert "stop" not in [r["record"] for r in records]
    assert sorted(ran) == sorted(set(labels) - set(NEEDING_THE_HELD_BUILD))  # all the others ran
    assert count == len(ran) == len(labels) - len(NEEDING_THE_HELD_BUILD)


def test_a_failed_first_sealed_build_still_stops_the_drive_on_the_committed_list(
    monkeypatch, tmp_path
):
    result, ran, records, labels = drive_committed_order(
        monkeypatch, tmp_path, {"sealed-truth": {"exit_status": 1}}
    )
    status, count, failed, not_run = result
    assert status == 1
    assert ran == labels[: labels.index("sealed-truth") + 1]
    assert [f["label"] for f in failed] == ["sealed-truth"]
    assert [n["label"] for n in not_run] == labels[labels.index("sealed-truth") + 1 :]
    assert records[-1]["record"] == "stop"


def test_an_interrupted_second_sealed_build_still_stops_the_drive_on_the_committed_list(
    monkeypatch, tmp_path
):
    result, ran, records, labels = drive_committed_order(
        monkeypatch, tmp_path, {HELD_BUILD: {"exit_status": 143, "interrupted": "SIGTERM"}}
    )
    assert result[0] != 0
    assert ran == labels[: labels.index(HELD_BUILD) + 1]
    assert records[-1]["record"] == "stop"
    assert [r for r in records if r["record"] == "failed"] == []


def test_a_second_sealed_build_the_recorder_cannot_confirm_still_stops_the_drive(
    monkeypatch, tmp_path
):
    unconfirmed = {"exit_status": 1, "manifest_sha256": None}
    result, ran, records, labels = drive_committed_order(
        monkeypatch, tmp_path, {HELD_BUILD: unconfirmed}
    )
    assert result[0] == run_driver.COULD_NOT_RUN
    assert ran == labels[: labels.index(HELD_BUILD) + 1]
    assert records[-1]["record"] == "stop"


def test_a_failed_second_sealed_build_goes_on_in_a_drive_of_the_stand_ins(repo, capsys):
    commit_list(repo, real_order(HELD_BUILD, "--exit", "1"))
    assert run_driver.main([LIST_PATH, *SEAL]) == 1
    records, data = the_log(repo)
    assert "stop" not in kinds(records)
    assert [r["label"] for r in records if r["record"] == "entry"] == [
        "dev-truth",
        "dev-truth-held",
        "sealed-truth",
        HELD_BUILD,
        "o2",
    ]
    (failed,) = [r for r in records if r["record"] == "failed"]
    assert failed["label"] == HELD_BUILD
    assert [n["label"] for n in records[-1]["not_run"]] == ["o2-held"]
    assert (records[-1]["completed"], records[-1]["exit_status"]) == (False, 1)
    assert_hash_recomputes(data)
    assert f"FAILED {HELD_BUILD}: " in capsys.readouterr().out


def test_a_failed_first_sealed_build_stops_a_drive_of_the_stand_ins_on_the_committed_order(repo):
    commit_list(repo, real_order("sealed-truth", "--exit", "1"))
    assert run_driver.main([LIST_PATH, *SEAL]) == 1
    records, _ = the_log(repo)
    assert records[-2]["record"] == "stop" and records[-2]["label"] == "sealed-truth"
    assert [r["label"] for r in records if r["record"] == "entry"][-1] == "sealed-truth"
    assert not any(HELD_BUILD in p.name for p in run_folders(repo))


def test_an_interrupted_second_sealed_build_stops_a_drive_of_the_stand_ins(repo):
    commit_list(repo, real_order(HELD_BUILD, "--term-parent"))
    assert run_driver.main([LIST_PATH, *SEAL]) != 0
    records, _ = the_log(repo)
    assert records[-2]["record"] == "stop" and records[-2]["label"] == HELD_BUILD
    assert "failed" not in kinds(records)
