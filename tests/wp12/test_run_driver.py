"""The driver (`fleetreserve.run_driver`) on made-up run lists of made-up steps, in a throwaway git
repository (`drive_fake`), so the real recorder's lock and runs are those of the throwaway
repository, never of the project.

1. Two steps that pass: the drive completes (exit 0) although each step's recorder takes the
   heavy-run lock (the driver holds none); every entry is logged with its run folder, exit status,
   timeout flag, times and manifest hash; the fixed seed reaches the step in place of its
   reference; the end line holds the total duration, says the commit and the tree were unchanged
   after the last entry, and holds the hash of the lines above it and of its own fields.
2. A step that exits 1 after one that exits 0: the drive stops after it (exit 1) and the next
   entry does not run; the log shows the stop.
3. A step that outlasts its time limit: logged as a timeout, and the drive stops (exit 1).
4. A step that commits, or leaves the tree changed: the drive stops before the next entry (exit 2).
5. Refused before anything runs (exit 2, no log, no run): a tree not clean, a list not committed,
   a list that is faulty, a list that is missing.
6. A step the recorder refuses (the heavy-run lock is already held): logged with no run folder,
   and the drive stops (exit 2).
7. Labels: the checker of the run list accepts a label exactly when the real recorder, called as
   the driver calls it, records a run under it (labels that begin with '-', and labels at and one
   past the file system's limit on the run folder's name); a list with a label that begins with
   '-' drives to its end.
8. After the last entry: a last step that leaves the tree changed ends the drive with 2, and the
   end line says so.
9. The run folder is the one the recorder reports, not one found by listing the runs folder (a
   step that makes a second folder with its label does not stop the drive); a recorder whose exit
   status its manifest does not confirm stops the drive (exit 2).
10. Placeholders: a step gets the units of its group in the purpose driven in place of `{units}`
    and the run folder an earlier entry recorded in this drive in place of `{run:<label>}`.
11. Purposes: one list drives the rehearsal without the entries and arguments of the sealed run
    only, and the sealed run with them (its gate made to pass); each log is named by its purpose
    and its start lists the purpose's entries as instantiated; the purpose must be given.
12. The sealed run's gate (`cold_start_gate.require_gate`, the checks the sealed files' loader
    makes): in a made-up repository a sealed-run drive is refused before anything runs (exit 2,
    no log, no run) with no tag `frozen-plan`, with the tag on an earlier commit, with the tag on
    the commit but the tree not clean, and with the tag on the commit and a clean tree, since the
    code running is not the made-up repository's; a rehearsal never calls the gate.
13. The sealed run at a logged parsing fix (its gate made to pass as there, returning the tag's
    commit with the fix commit): the drive runs and its start line records the checked-out
    commit, the tag's commit and the fix commit; with a fix commit that is not the checked-out
    commit it is refused before anything runs (exit 2, no log, no run); at the tag its start
    line holds the same fields as before and no fix commit. With the REAL gate, in the
    throwaway repository made one at a logged parsing fix (`fake_heldout`: a made-up plan frozen
    at the tag, a refusal logged there, a correction and its entry, a copy of the running
    package under `src/fleetreserve/`): the drive runs every entry and its start line records
    the tag's commit and the fix commit; with an untracked file there it is refused with the
    gate's message (exit 2, no log, no run).
"""

import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from drive_fake import (
    LIST_PATH,
    SEALED_MATRIX,
    SEALED_TRUTH,
    commit_list,
    document,
    drive_logs,
    entry,
    git,
    make_repo,
    no_start_check,
    read_log,
    run_folders,
    step,
)

from fleetreserve import cold_start_gate, run_driver, run_list, sealed_route

_WP2 = Path(__file__).resolve().parents[1] / "wp2"  # the made-up held-out repositories
if str(_WP2) not in sys.path:
    sys.path.insert(0, str(_WP2))
import fake_heldout  # noqa: E402

LOG_NAME = re.compile(r"\d{8}T\d{6}Z-rehearsal\.jsonl")
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
    no_start_check(monkeypatch)  # the stand-ins' starts (drive_fake, module text)
    monkeypatch.chdir(root)
    return root


def the_log(repo) -> tuple[list[dict], bytes]:
    (path,) = drive_logs(repo)
    assert LOG_NAME.fullmatch(path.name)
    return read_log(path), path.read_bytes()


def assert_hash_recomputes(data: bytes) -> None:
    """The end's hash covers every line above it and the end's own fields, written as the driver
    writes a line (keys sorted), without the hash itself."""
    lines = data.splitlines(keepends=True)
    end = json.loads(lines[-1])
    claimed = end.pop("log_sha256")
    own = (json.dumps(end, sort_keys=True) + "\n").encode()
    assert claimed == hashlib.sha256(b"".join(lines[:-1]) + own).hexdigest()


def manifest(repo, record: dict) -> dict:
    return json.loads((repo / record["run"] / "manifest.json").read_text())


def test_two_steps_that_pass_complete_and_log_every_entry(repo, capsys):
    listed = document(
        [
            entry("wp2-truth", step("truth_build", "--seed", "{seed:split}")),
            entry("rehearsal-learn", step("learn_run"), max_seconds=90.5, after=["wp2-truth"]),
        ],
        seeds={"split": 7},
    )
    commit_list(repo, listed)
    head = git(repo, "rev-parse", "HEAD").strip()
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 0
    records, data = the_log(repo)
    assert [r["record"] for r in records] == ["start", "begin", "entry", "begin", "entry", "end"]
    start = records[0]
    assert start["commit"] == head
    assert start["list"] == LIST_PATH
    assert start["list_sha256"] == hashlib.sha256((repo / LIST_PATH).read_bytes()).hexdigest()
    assert (start["purpose"], start["entries"], start["format"]) == ("rehearsal", 2, "drive-log-3")
    assert start["tag_commit"] is None
    assert start["steps"] == [
        {"label": "wp2-truth", "command": step("truth_build", "--seed", "7"), "max_seconds": 60},
        {"label": "rehearsal-learn", "command": step("learn_run"), "max_seconds": 90.5},
    ]
    first, second = records[2], records[4]
    assert records[1]["command"] == step("truth_build", "--seed", "7")
    for record, label, limit in ((first, "wp2-truth", 60), (second, "rehearsal-learn", 90.5)):
        assert record["label"] == label
        assert re.fullmatch(rf"runs\.noindex/\d{{8}}T\d{{6}}Z-{label}", record["run"])
        assert (
            record["manifest_sha256"]
            == hashlib.sha256((repo / record["run"] / "manifest.json").read_bytes()).hexdigest()
        )
        assert (record["exit_status"], record["timed_out"], record["interrupted"]) == (
            0,
            False,
            None,
        )
        assert record["seconds"] > 0
        recorded = manifest(repo, record)
        assert (recorded["long"], recorded["max_seconds"], recorded["commit"]) == (
            True,
            limit,
            head,
        )
    assert manifest(repo, first)["command"] == step("truth_build", "--seed", "7")
    assert (repo / first["run"] / "step.txt").read_text() == "--seed 7\n"
    end = records[-1]
    assert (end["completed"], end["entries_run"], end["exit_status"]) == (True, 2, 0)
    assert (end["unchanged_after"], end["change_after"]) == (True, None)
    assert end["seconds"] >= first["seconds"] + second["seconds"]
    assert_hash_recomputes(data)
    with (repo / "runs.noindex" / "heavy.lock").open("a") as lock:  # the lock was let go
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert "drive ended with exit status 0" in capsys.readouterr().out


def test_a_failing_step_stops_the_drive_and_is_logged(repo):
    listed = document(
        [
            entry("rehearsal-ok", step("truth_build")),
            entry("rehearsal-bad", step("learn_run", "--exit", "1"), after=["rehearsal-ok"]),
            entry("rehearsal-never", step("compare_run")),
        ]
    )
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 1
    records, data = the_log(repo)
    kinds = [r["record"] for r in records]
    assert kinds == ["start", "begin", "entry", "begin", "entry", "stop", "end"]
    assert records[2]["exit_status"] == 0
    bad = records[4]
    assert (bad["label"], bad["exit_status"], bad["timed_out"]) == ("rehearsal-bad", 1, False)
    assert manifest(repo, bad)["exit_code"] == 1
    assert records[5] == {
        "record": "stop",
        "index": 1,
        "label": "rehearsal-bad",
        "reason": "the run ended with exit status 1",
    }
    end = records[-1]
    assert (end["completed"], end["entries_run"], end["exit_status"]) == (False, 2, 1)
    # Two runs may start in one second, so their folders' order by name is not their order.
    assert sorted(p.name.split("-", 1)[1] for p in run_folders(repo)) == [
        "rehearsal-bad",
        "rehearsal-ok",
    ]
    assert_hash_recomputes(data)


def test_a_timed_out_step_is_logged_as_a_timeout_and_stops_the_drive(repo):
    listed = document(
        [
            entry("rehearsal-slow", step("learn_run", "--sleep", "30"), max_seconds=1),
            entry("rehearsal-never", step("compare_run")),
        ]
    )
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 1
    records, data = the_log(repo)
    assert [r["record"] for r in records] == ["start", "begin", "entry", "stop", "end"]
    slow = records[2]
    assert slow["timed_out"] is True
    assert slow["exit_status"] == 128 + signal.SIGTERM
    assert slow["seconds"] < 20
    assert manifest(repo, slow)["timed_out"] is True
    assert "time limit" in records[3]["reason"]
    assert records[-1]["exit_status"] == 1
    assert len(run_folders(repo)) == 1
    assert_hash_recomputes(data)


@pytest.mark.parametrize(
    ("flag", "reason"),
    [
        ("--commit", "the commit changed since the drive started"),
        ("--dirty", "the git tree has uncommitted changes since the drive started"),
    ],
)
def test_a_changed_commit_or_tree_stops_the_drive_before_the_next_entry(repo, flag, reason):
    listed = document(
        [
            entry("rehearsal-change", step("truth_build", flag)),
            entry("rehearsal-next", step("learn_run")),
        ]
    )
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 2
    records, data = the_log(repo)
    assert [r["record"] for r in records] == ["start", "begin", "entry", "stop", "end"]
    assert records[2]["exit_status"] == 0
    stop = records[3]
    assert (stop["index"], stop["label"]) == (1, "rehearsal-next")
    assert stop["reason"].startswith(reason)
    end = records[-1]
    assert (end["completed"], end["entries_run"], end["exit_status"]) == (False, 1, 2)
    assert len(run_folders(repo)) == 1
    assert_hash_recomputes(data)


def dirty_tree(repo):
    commit_list(repo, document([entry("rehearsal-a", step("truth_build"))]))
    (repo / "untracked.txt").write_text("x\n")
    return LIST_PATH, "the git tree has uncommitted changes"


def uncommitted_list(repo):
    path = repo / "runs.noindex" / "list.json"  # an ignored place, so the tree stays clean
    path.parent.mkdir()
    path.write_text(json.dumps(document([entry("rehearsal-a", step("truth_build"))])))
    return "runs.noindex/list.json", "is not committed at the checked-out commit"


def faulty_list(repo):
    listed = document([entry("rehearsal-a", step("truth_build"))])
    del listed["entries"][0]["max_seconds"]
    commit_list(repo, listed)
    return LIST_PATH, "has no time limit (max_seconds)"


def missing_list(repo):
    return "lists/no-such-list.json", "cannot be read"


@pytest.mark.parametrize("case", [dirty_tree, uncommitted_list, faulty_list, missing_list])
def test_a_drive_that_cannot_start_is_refused_and_runs_nothing(repo, capsys, case):
    path, said = case(repo)
    assert run_driver.main([path, *REHEARSE]) == 2
    out = capsys.readouterr().out
    assert out.startswith("refused: ")
    assert said in out
    assert drive_logs(repo) == []
    assert run_folders(repo) == []


def test_a_step_the_recorder_refuses_is_logged_without_a_run_and_stops_the_drive(repo):
    listed = document(
        [
            entry("rehearsal-locked", step("truth_build")),
            entry("rehearsal-never", step("learn_run")),
        ]
    )
    commit_list(repo, listed)
    (repo / "runs.noindex").mkdir()
    with (repo / "runs.noindex" / "heavy.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another heavy run is going
        assert run_driver.main([LIST_PATH, *REHEARSE]) == 2
    records, data = the_log(repo)
    assert [r["record"] for r in records] == ["start", "begin", "entry", "stop", "end"]
    locked = records[2]
    assert (locked["run"], locked["manifest_sha256"], locked["recorder_status"]) == (None, None, 2)
    assert "a heavy run is already going" in locked["recorder_output"]
    assert "no run folder" in records[3]["reason"]
    assert records[-1]["exit_status"] == 2
    assert run_folders(repo) == []
    assert_hash_recomputes(data)


# --- 7. labels -----------------------------------------------------------------------------------

STAMP_AND_DASH = len("20000101T000000Z-")  # the recorder's run folder name: <UTC start>-<label>


def recorded(repo, label: str) -> bool:
    """Whether the real recorder, called exactly as the driver calls it, records a run under
    `label`: exit 0 and one run folder with a manifest."""
    listed = run_list.Step(
        position=0,
        label=label,
        command=tuple(step("truth_build")),
        runs=(),
        setting="main",
        max_seconds=60,
        after=(),
    )
    done = subprocess.run(
        run_driver.recorder_command(listed, step("truth_build")),
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    folders = run_folders(repo)
    made = done.returncode == 0 and len(folders) == 1 and (folders[0] / "manifest.json").is_file()
    return made and folders[0].name.endswith(f"Z-{label}")


def accepted(tmp_path, label: str, capsys) -> bool:
    """Whether the checker of the run list accepts a one-entry list with `label` (exit 0, or
    exit 1 with a fault about the label alone)."""
    path = tmp_path / "one.json"  # beside the repository, on its file system
    path.write_text(json.dumps(document([entry(label, step("truth_build"))])))
    code = run_list.main([str(path)])
    lines = capsys.readouterr().out.splitlines()
    assert code in (0, 1), lines
    if code == 1:
        assert [line for line in lines if line.startswith("FAULT: ")] == [lines[0]], lines
        assert "label" in lines[0] or "run folder name" in lines[0], lines
    return code == 0


@pytest.mark.parametrize(
    "label", ["first", "--long", "-x", "--", "at-the-limit", "one-past-the-limit"]
)
def test_the_checker_accepts_a_label_exactly_when_the_recorder_records_it(
    repo, tmp_path, capsys, label
):
    limit = os.pathconf(repo, "PC_NAME_MAX")
    if label == "at-the-limit":
        label = "a" * (limit - STAMP_AND_DASH)
    elif label == "one-past-the-limit":
        label = "a" * (limit - STAMP_AND_DASH + 1)
    verdict = accepted(tmp_path, label, capsys)
    assert verdict == recorded(repo, label)
    if label.startswith("a"):  # the two anchors: the recorder itself draws the line here
        assert verdict == (len(label) == limit - STAMP_AND_DASH)


def test_a_list_with_a_label_that_begins_with_a_dash_drives_to_its_end(repo):
    listed = document(
        [entry("first", step("truth_build")), entry("--long", step("learn_run"), after=["first"])]
    )
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 0
    records, _ = the_log(repo)
    assert [r["record"] for r in records] == ["start", "begin", "entry", "begin", "entry", "end"]
    assert re.fullmatch(r"runs\.noindex/\d{8}T\d{6}Z---long", records[4]["run"])
    assert manifest(repo, records[4])["exit_code"] == 0


# --- 8. after the last entry ---------------------------------------------------------------------


def test_a_last_entry_that_leaves_the_tree_changed_ends_the_drive_with_2(repo):
    listed = document(
        [
            entry("rehearsal-ok", step("truth_build")),
            entry("rehearsal-last", step("learn_run", "--dirty")),
        ]
    )
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 2
    records, data = the_log(repo)
    assert [r["record"] for r in records] == ["start", "begin", "entry", "begin", "entry", "end"]
    assert records[4]["exit_status"] == 0
    end = records[-1]
    assert (end["completed"], end["entries_run"], end["exit_status"]) == (False, 2, 2)
    assert end["unchanged_after"] is False
    assert end["change_after"] == "the git tree has uncommitted changes since the drive started"
    assert_hash_recomputes(data)


# --- 9. the run folder the recorder reports ------------------------------------------------------


def test_the_run_folder_is_the_one_the_recorder_reports(repo):
    listed = document([entry("rehearsal-twin", step("truth_build", "--sibling"))])
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 0
    records, _ = the_log(repo)
    run = records[2]["run"]
    assert re.fullmatch(r"runs\.noindex/\d{8}T\d{6}Z-rehearsal-twin", run)
    assert not run.startswith("runs.noindex/29991231T235959Z")  # the step's twin folder
    assert (repo / "runs.noindex" / "29991231T235959Z-rehearsal-twin").is_dir()
    assert (repo / run / "step.txt").read_text() == "--sibling\n"


def test_a_recorder_status_its_manifest_does_not_confirm_stops_the_drive(repo, monkeypatch):
    listed = document(
        [entry("rehearsal-said", step("truth_build")), entry("rehearsal-never", step("learn_run"))]
    )
    commit_list(repo, listed)
    real = subprocess.run

    def other_status(argv, *args, **kwargs):
        done = real(argv, *args, **kwargs)
        if run_driver.RECORDER in argv:
            done.returncode = 3  # the recorder's status, while its manifest records 0
        return done

    monkeypatch.setattr(subprocess, "run", other_status)
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 2
    records, data = the_log(repo)
    assert [r["record"] for r in records] == ["start", "begin", "entry", "stop", "end"]
    said = records[2]
    assert (said["recorder_status"], said["exit_status"]) == (3, 0)
    assert manifest(repo, said)["exit_code"] == 0
    assert records[3]["reason"] == (
        "the recorder ended with 3 but the manifest of its run records the exit code 0"
    )
    assert_hash_recomputes(data)


# --- 10. placeholders ----------------------------------------------------------------------------


def test_a_step_gets_its_units_and_the_run_folder_of_an_earlier_entry(repo):
    learn = step("learn_run", "--units", "{units}", "--truth-run", "{run:wp2-truth}")
    listed = document(
        [
            entry("wp2-truth", step("truth_build")),
            entry("rehearsal-learn", learn, units="scored", after=["wp2-truth"]),
        ],
        units={
            "rehearsal": {"scored": ["nrel-before-may-2019", "BER"]},
            "sealed-run": {"scored": ["OSL_T", "TRO_R"]},
        },
    )
    commit_list(repo, listed)
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 0
    records, _ = the_log(repo)
    truth, learned = records[2], records[4]
    filled = step(
        "learn_run", "--units", "nrel-before-may-2019", "BER", "--truth-run", truth["run"]
    )
    assert records[3]["command"] == filled
    assert manifest(repo, learned)["command"] == filled
    assert (repo / learned["run"] / "step.txt").read_text() == " ".join(filled[3:]) + "\n"
    as_listed = step("learn_run", "--units", "nrel-before-may-2019", "BER", "--truth-run")
    assert records[0]["steps"][1]["command"] == [*as_listed, "{run:wp2-truth}"]


# --- 11. purposes --------------------------------------------------------------------------------


def two_purposes(*, seed: bool = True) -> dict:
    """A list whose sealed run has entries of its own and arguments of its own: the stand-in of
    the sealed truth build, the sealed route's argument and the interval seed, and the intervals
    of the scoring entry's run. With `seed` false, the list of the earlier form, whose sealed
    scoring carries no interval seed and has no intervals entry (refused by rule 2)."""
    sealed_route = {"sealed-run": ["--held-out-truth-run", "{run:sealed-truth}"]}
    if seed:
        sealed_route["sealed-run"] += ["--matrix", SEALED_MATRIX]
        sealed_route["sealed-run"] += ["--interval-seed", "{seed:intervals}"]
        sealed_route["sealed-run"] += ["--interval-resamples", "2000"]
    entries = [
        entry("wp2-truth", step("truth_build")),
        entry("sealed-truth", step(SEALED_TRUTH, "--sealed"), only="sealed-run"),
        entry(
            "rehearsal-learn",
            step("learn_run", "--units", "{units}"),
            units="scored",
            extra=sealed_route,
            after=["wp2-truth"],
        ),
    ]
    if seed:
        intervals = step("day_intervals", "--run", "{run:rehearsal-learn}", "--seed")
        entries.append(
            entry(
                "sealed-intervals",
                [*intervals, "{seed:intervals}"],
                only="sealed-run",
                after=["rehearsal-learn"],
            )
        )
    return document(
        entries,
        seeds={"intervals": 7} if seed else None,
        units={
            "rehearsal": {"scored": ["nrel-before-may-2019"]},
            "sealed-run": {"scored": ["OSL_T"]},
        },
    )


def test_the_list_of_the_earlier_form_without_the_interval_seed_is_refused(
    repo, monkeypatch, capsys
):
    commit_list(repo, two_purposes(seed=False))
    passing_gate(monkeypatch)
    assert run_driver.main([LIST_PATH, *SEAL]) == 2
    out = capsys.readouterr().out
    assert (
        "FAULT: entry 3 (rehearsal-learn): on the sealed run it scores without --interval-seed"
        in out
    )
    assert drive_logs(repo) == []
    assert run_folders(repo) == []


def passing_gate(monkeypatch) -> list:
    """Make the sealed run's gate pass, as at the tag; the folders it was called with."""
    calls = []

    def gate(root):
        calls.append(root)
        return git(root, "rev-parse", "HEAD").strip()

    monkeypatch.setattr(cold_start_gate, "require_gate", gate)
    return calls


def logged(records: list[dict], kind: str) -> list[dict]:
    return [r for r in records if r["record"] == kind]


def test_one_list_drives_each_purpose_with_its_own_entries_and_arguments(repo, monkeypatch):
    commit_list(repo, two_purposes())
    head = git(repo, "rev-parse", "HEAD").strip()
    calls = passing_gate(monkeypatch)
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 0
    assert calls == []  # a rehearsal never calls the gate
    time.sleep(1.1)  # the two drives' runs of wp2-truth must start in different seconds
    assert run_driver.main([LIST_PATH, *SEAL]) == 0
    assert [path.resolve() for path in calls] == [repo.resolve()]
    logs = {path.name.split("-", 1)[1]: read_log(path) for path in drive_logs(repo)}
    assert sorted(logs) == ["rehearsal.jsonl", "sealed-run.jsonl"]
    rehearsal, sealed = logs["rehearsal.jsonl"], logs["sealed-run.jsonl"]
    assert [r["label"] for r in logged(rehearsal, "entry")] == ["wp2-truth", "rehearsal-learn"]
    assert [r["label"] for r in logged(sealed, "entry")] == [
        "wp2-truth",
        "sealed-truth",
        "rehearsal-learn",
        "sealed-intervals",
    ]
    assert (rehearsal[0]["purpose"], rehearsal[0]["tag_commit"]) == ("rehearsal", None)
    assert (sealed[0]["purpose"], sealed[0]["tag_commit"]) == ("sealed-run", head)
    assert rehearsal[0]["list_sha256"] == sealed[0]["list_sha256"]
    assert logged(rehearsal, "begin")[-1]["command"] == step(
        "learn_run", "--units", "nrel-before-may-2019"
    )
    sealed_truth = logged(sealed, "entry")[1]["run"]
    seeded = ["--matrix", SEALED_MATRIX, "--interval-seed", "7", "--interval-resamples", "2000"]
    assert logged(sealed, "begin")[2]["command"] == step(
        "learn_run", "--units", "OSL_T", "--held-out-truth-run", sealed_truth, *seeded
    )
    assert sealed[0]["steps"][2]["command"] == step(
        "learn_run", "--units", "OSL_T", "--held-out-truth-run", "{run:sealed-truth}", *seeded
    )


@pytest.mark.parametrize("purpose", [[], ["--purpose", "batch"]], ids=["none", "unknown"])
def test_a_drive_needs_its_purpose(repo, purpose):
    commit_list(repo, document([entry("wp2-truth", step("truth_build"))]))
    with pytest.raises(SystemExit) as stopped:
        run_driver.main([LIST_PATH, *purpose])
    assert stopped.value.code == 2
    assert drive_logs(repo) == []
    assert run_folders(repo) == []


# --- 12. the sealed run's gate -------------------------------------------------------------------

GATE = (
    "refused: a sealed-run drive starts only at the tag frozen-plan, or at a logged parsing fix "
    "of it, on a clean tree: "
)


def no_tag(repo):
    return (
        "no tag frozen-plan: sealed files open only at the tag frozen-plan, or at a logged parsing "
        "fix of it"
    )


def tag_on_an_earlier_commit(repo):
    git(repo, "tag", "frozen-plan", "HEAD~1")
    return "is a commit after the tag frozen-plan"


def tag_but_a_tree_not_clean(repo):
    git(repo, "tag", "frozen-plan")
    (repo / "untracked.txt").write_text("x\n")
    return "the git tree has uncommitted changes; sealed files open only clean"


def tag_and_a_clean_tree_but_other_code(repo):
    git(repo, "tag", "frozen-plan")  # the code running is this project's, not the tag's files
    return "the running code is not the tag's"


GATE_CASES = [
    no_tag,
    tag_on_an_earlier_commit,
    tag_but_a_tree_not_clean,
    tag_and_a_clean_tree_but_other_code,
]


@pytest.mark.parametrize("case", GATE_CASES)
def test_a_sealed_run_drive_away_from_the_tag_is_refused_and_runs_nothing(repo, capsys, case):
    commit_list(repo, two_purposes())
    said = case(repo)
    assert run_driver.main([LIST_PATH, *SEAL]) == 2
    out = capsys.readouterr().out
    assert out.startswith(GATE), out
    assert said in out
    assert drive_logs(repo) == []
    assert run_folders(repo) == []


# --- 13. the sealed run at a logged parsing fix --------------------------------------------------

START_KEYS = ["commit", "entries", "format", "list", "list_sha256", "pid", "purpose", "record"]
START_KEYS += ["start", "steps", "tag_commit"]  # a start line's keys at the tag, as before


def gate_at_a_fix(monkeypatch, tag_commit: str, fix_commit: str | None = None) -> None:
    """Make the sealed run's gate pass as at a logged parsing fix after the tag `tag_commit`: it
    returns the tag's commit with the fix commit (default: HEAD), as `sealed_route.require_gate`
    does there."""

    def gate(root):
        head = git(root, "rev-parse", "HEAD").strip()
        return sealed_route.GateCommit(tag_commit, fix_commit or head)

    monkeypatch.setattr(cold_start_gate, "require_gate", gate)


def test_a_sealed_run_at_a_parsing_fix_records_the_tags_commit_and_the_fix(repo, monkeypatch):
    commit_list(repo, two_purposes())
    tag_commit = git(repo, "rev-parse", "HEAD~1").strip()
    head = git(repo, "rev-parse", "HEAD").strip()
    gate_at_a_fix(monkeypatch, tag_commit)
    assert run_driver.main([LIST_PATH, *SEAL]) == 0
    (path,) = drive_logs(repo)
    start = read_log(path)[0]
    assert (start["commit"], start["tag_commit"], start["fix_commit"]) == (head, tag_commit, head)
    assert sorted(start) == sorted([*START_KEYS, "fix_commit"])


def test_a_sealed_run_away_from_the_fix_the_gate_passed_at_is_refused(repo, monkeypatch, capsys):
    commit_list(repo, two_purposes())
    tag_commit = git(repo, "rev-parse", "HEAD~1").strip()
    head = git(repo, "rev-parse", "HEAD").strip()
    gate_at_a_fix(monkeypatch, tag_commit, "0" * 40)
    assert run_driver.main([LIST_PATH, *SEAL]) == 2
    assert capsys.readouterr().out == (
        f"refused: the checked-out commit {head[:12]} is not the parsing fix {'0' * 12} the gate "
        "passed at\n"
    )
    assert drive_logs(repo) == []
    assert run_folders(repo) == []


def test_a_sealed_run_at_the_tag_writes_no_fix_commit(repo, monkeypatch):
    commit_list(repo, two_purposes())
    passing_gate(monkeypatch)
    assert run_driver.main([LIST_PATH, *SEAL]) == 0
    (path,) = drive_logs(repo)
    assert sorted(read_log(path)[0]) == sorted(START_KEYS)


def at_a_real_fix(repo) -> tuple[str, str]:
    """The drive's made-up repository made one at a logged parsing fix for the real gate: `data/`
    ignored, a copy of the running package under `src/fleetreserve/` (so the gate's check of the
    running code passes, while the steps still run the stand-in at the top), a made-up plan
    frozen at the tag, a refusal of the a.s.r. layout logged there by hand, then a correction of a
    test file of its loader and its entry in section 14; (the tag's commit, the fix commit)."""
    commit_list(repo, two_purposes())
    with (repo / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("data/\n")
    fake_heldout.copy_package(repo)
    tagged = fake_heldout.freeze(repo)
    fake_heldout.log_refusal(repo, fake_heldout.ASR_FILE, b"made-up sealed bytes\n", tagged)
    return tagged, fake_heldout.entry(repo, fake_heldout.correction(repo, "asr"))


def test_a_sealed_run_at_a_logged_parsing_fix_passes_the_real_gate(repo):
    tagged, fixed = at_a_real_fix(repo)
    assert run_driver.main([LIST_PATH, *SEAL]) == 0
    (path,) = drive_logs(repo)
    records = read_log(path)
    start = records[0]
    assert (start["commit"], start["tag_commit"], start["fix_commit"]) == (fixed, tagged, fixed)
    assert sorted(start) == sorted([*START_KEYS, "fix_commit"])
    assert [r["label"] for r in logged(records, "entry")] == [
        "wp2-truth",
        "sealed-truth",
        "rehearsal-learn",
        "sealed-intervals",
    ]


def test_a_sealed_run_at_a_logged_parsing_fix_on_a_tree_not_clean_is_refused(repo, capsys):
    at_a_real_fix(repo)
    (repo / "untracked.txt").write_text("x\n")
    assert run_driver.main([LIST_PATH, *SEAL]) == 2
    assert capsys.readouterr().out == (
        f"{GATE}the git tree has uncommitted changes; sealed files open only clean\n"
    )
    assert drive_logs(repo) == []
    assert run_folders(repo) == []
