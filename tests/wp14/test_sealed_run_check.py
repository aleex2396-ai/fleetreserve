"""`tools/sealed_run_check.py`, the one command of WP14's pass mark, on sealed-run drives written by
the real driver (`run_driver`) through the REAL gate of the run from the tag, in throwaway git
repositories (`tests/wp12/drive_fake.py`) made ones of a made-up freeze
(`tests/wp2/fake_heldout.py`: `data/` ignored, a copy of the running package under
`src/fleetreserve/`, a made-up plan frozen at the tag `frozen-plan`). The step standing in for the
held-out truth build (`TRUTH_STEP`) writes one line to the openings log as the opener writes one,
and with `--refuse-at-tag`, at the tag's commit, a line to the refusal log as a loader's layout
refusal writes one and exits 1. A correction of the a.s.r. loader's test file and its entry in
plan section 14 make the parsing fix, which the real gate accepts once the refusal is logged.
Each test runs the check with `FLEETRESERVE_RUN_DIR` naming a made-up run folder directly in the
repository's runs folder with no manifest (the gate mark's rule, `gate_mark.run_folder`), made
just before the check runs, and reads the results file it writes. The driver's start check,
which starts the real commands of the route, is replaced for the stand-in steps
(`drive_fake.no_start_check`).

The numbers are those of the tests of the simple script (section 4 of the proposal it was built
from): 1-2 end to end; 3-6 one logged run; 7-9 logged parsing fixes; 10-11 and 13 could not run;
12 logs hashed; 14 nothing sealed opened; 15 exit statuses; 16 under the run recorder; and a log
with no end line could not run. 17: each O2 step of the complete drive (the stand-in
`drive_fake.O2_STEP`) gives one line of mark 1 with its verdict, no mark changed by it, and fails
mark 1 when it exited 0 with no readable verdict; its results file is opened only inside
`run_verify`, and nothing sealed is opened; beside the verdict line, one line gives the units with a
non-zero count, each named by what it holds (block orders on M days and unbuilt M days are the
parts in O2's pass), or says the counts are missing or not readable (decision 88 (3d)), with no
mark changed by it. Hand cases on made-up drives pin how an opening or a
refusal is matched to a step.
"""

import fcntl
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import sealed_watch
from sealed_watch import watch

from fleetreserve import gate_mark, heldout_loader, run_driver, run_verify, sealed_route

TOP = Path(__file__).resolve().parents[2]
TOOL = TOP / "tools" / "sealed_run_check.py"
for _helpers in (TOP / "tools", TOP / "tests" / "wp2", TOP / "tests" / "wp12"):
    if str(_helpers) not in sys.path:
        sys.path.insert(0, str(_helpers))
fake = importlib.import_module("fake_heldout")
drive_fake = importlib.import_module("drive_fake")
opened_from = importlib.import_module("opened_from")
check = importlib.import_module("sealed_run_check")

SEAL = ["--purpose", "sealed-run"]
LIST = drive_fake.LIST_PATH
MARKS = ["one logged run", "logged parsing fixes", "logs hashed"]
# The made-up held-out truth build: one opening line, as the opener writes one, then at the tag
# with --refuse-at-tag a layout refusal, as a loader's refusal writes one, and exit 1. With the
# ignored file data/slow-before (data/slow-after) it sleeps 600 s before (after) its opening, far
# past its made-up time limit (`SLOW_LIMIT`), so the recorder stops it there. With the ignored file
# data/refuse-now it writes the layout refusal and exits 1 at any commit.
TRUTH_STEP = """import datetime
import json
import os
import subprocess
import sys
import time


def git(*args):
    quiet = ["-c", "maintenance.auto=false", "-c", "gc.auto=0"]
    done = subprocess.run(["git", *quiet, *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


head = git("rev-parse", "HEAD")
tag = git("rev-parse", "refs/tags/frozen-plan^{commit}")
now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
names = ("asr", "202410DatasetEVOfficeParking_v0.csv")
os.makedirs("data", exist_ok=True)
if os.path.exists(os.path.join("data", "slow-before")):
    time.sleep(600)
line = {"time_utc": now, "set": names[0], "file": names[1], "sha256": "0" * 64, "commit": head,
        "tag_commit": tag, "command": sys.argv}
if head != tag:
    line["fix_of_tag"] = tag
with open(os.path.join("data", "heldout-openings.log"), "a") as handle:
    handle.write(json.dumps(line) + "\\n")
if os.path.exists(os.path.join("data", "slow-after")):
    time.sleep(600)
if "--fail-at-tag" in sys.argv and head == tag:
    sys.exit(1)
refuse_now = os.path.exists(os.path.join("data", "refuse-now"))
if "--refuse-at-tag" in sys.argv and head == tag or refuse_now:
    refusal = {"time_utc": now, "set": names[0], "file": names[1], "sha256": "0" * 64,
               "commit": head, "tag_commit": tag, "layout": "a made-up layout refusal"}
    with open(os.path.join("data", "heldout-refusals.log"), "a") as handle:
        handle.write(json.dumps(refusal) + "\\n")
    sys.exit(1)
# the summary the real build always writes (its `abstained` record, empty here), which the WP14
# check reads from a held-out truth build step that passed
with open(os.path.join(os.environ["FLEETRESERVE_RUN_DIR"], "truth-summary.json"), "w") as handle:
    json.dump({"units": {}, "abstained": {}}, handle)
"""


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    """Drop every GIT_* variable and git's global and system settings for the test."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture(autouse=True)
def recorder_folder(tmp_path, monkeypatch) -> Path:
    """A made-up run folder named by `FLEETRESERVE_RUN_DIR`, directly in the runs folder of the
    made-up repository (`drive_fake.make_repo`'s `tmp_path / "repo"`). It is made only when the
    check runs (`run`, `script`), so no drive before writes the gate's mark into it."""
    folder = tmp_path / "repo" / "runs.noindex" / "recorder-run"
    monkeypatch.setenv(check.RUN_DIR_VARIABLE, str(folder))
    return folder


def truth(*arguments: str, label: str = "sealed-truth", max_seconds=60) -> dict:
    command = drive_fake.step(drive_fake.SEALED_TRUTH, *arguments)
    return drive_fake.entry(label, command, only="sealed-run", max_seconds=max_seconds)


def learn() -> dict:
    return drive_fake.entry("sealed-learn", drive_fake.step("learn_run"))


def made(
    tmp_path, monkeypatch, entries=None, *, listed=True, opener=drive_fake.SEALED_TRUTH, units=None,
    modules=None,
) -> tuple[Path, str]:  # fmt: skip
    """The throwaway repository frozen at the tag, with the run list of `entries` (default: the
    made-up truth build refusing at the tag, then one more step) on the groups of units `units`
    committed unless `listed` is false, the module `opener` being the made-up held-out truth
    build (`TRUTH_STEP`), and the stand-in modules named in `modules` ({name: text}) replaced;
    (its top folder, the tag's commit)."""
    root = drive_fake.make_repo(tmp_path)
    (root / "fleetreserve" / f"{opener}.py").write_text(TRUTH_STEP)
    for name, text in (modules or {}).items():
        (root / "fleetreserve" / f"{name}.py").write_text(text)
    with (root / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("data/\n")
    fake.copy_package(root)
    if listed:
        chosen = [truth("--refuse-at-tag"), learn()] if entries is None else entries
        drive_fake.commit_list(root, drive_fake.document(chosen, units=units))
    tagged = fake.freeze(root)
    drive_fake.no_start_check(monkeypatch)  # the stand-ins' starts (drive_fake, module text)
    monkeypatch.chdir(root)
    return root, tagged


def sealed_drive(expected: int) -> None:
    """One sealed-run drive by the real driver through the real gate, its exit status checked;
    the next drive starts in another second, so the logs' names differ."""
    assert run_driver.main([LIST, *SEAL]) == expected
    time.sleep(1.1)


def fixed(root: Path) -> str:
    """A logged parsing fix of the a.s.r. set: a correction of its loader's test file and its
    entry in plan section 14; the fix commit."""
    return fake.entry(root, fake.correction(root, "asr"))


def at_a_fix(tmp_path, monkeypatch) -> tuple[Path, str, str]:
    """A drive at the tag stopped at a layout refusal, a logged fix, a complete drive at the fix;
    (top folder, tag's commit, fix commit)."""
    root, tagged = made(tmp_path, monkeypatch)
    sealed_drive(1)
    fix = fixed(root)
    sealed_drive(0)
    return root, tagged, fix


def complete_at_the_tag(tmp_path, monkeypatch) -> Path:
    root, _ = made(tmp_path, monkeypatch, [truth(), learn()])
    sealed_drive(0)
    return root


def run(root: Path, capsys) -> tuple[int, dict, dict | None]:
    """(the exit status, {mark: (its status, its lines)}, the results file or None); the results
    file is removed after reading, so the next run writes its own."""
    Path(os.environ[check.RUN_DIR_VARIABLE]).mkdir(parents=True, exist_ok=True)
    status = check.main(["--root", str(root)])
    capsys.readouterr()
    results_file = Path(os.environ[check.RUN_DIR_VARIABLE]) / check.RESULTS_NAME
    if not results_file.exists():
        return status, {}, None
    results = json.loads(results_file.read_text(encoding="utf-8"))
    results_file.unlink()
    marks = {mark["mark"]: (mark["status"], mark["lines"]) for mark in results["marks"]}
    assert list(marks) == MARKS
    return status, marks, results


def statuses(marks: dict) -> list[str]:
    return [marks[name][0] for name in MARKS]


def logs(root: Path) -> list[Path]:
    return drive_fake.drive_logs(root)


def openings_file(root: Path) -> Path:
    return heldout_loader.log_path(root)


def refusals_file(root: Path) -> Path:
    return heldout_loader.refusals_path(root)


def by_hand(root: Path, when: str, command: list[str], commit: str, *, first: bool = False):
    """An opening of the a.s.r. file written to the openings log by hand (first, or last)."""
    line = {
        "time_utc": when,
        "set": "asr",
        "file": fake.ASR_FILE[1],
        "sha256": "0" * 64,
        "commit": commit,
        "tag_commit": commit,
        "command": command,
    }
    lines = openings_file(root).read_text(encoding="utf-8").splitlines()
    lines = [json.dumps(line), *lines] if first else [*lines, json.dumps(line)]
    openings_file(root).write_text("\n".join(lines) + "\n", encoding="utf-8")


def rewritten(path: Path, change) -> None:
    """The drive log at `path` with `change(records)` applied and its end's hash recomputed, as
    the driver would write it with those lines."""
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    change(records)
    above = b"".join(run_driver.encode(record) for record in records[:-1])
    end = records[-1]
    end["log_sha256"] = run_driver.end_digest(above, end)
    path.write_bytes(above + run_driver.encode(end))


def moved(stamp: str, seconds: int) -> str:
    when = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ") + timedelta(seconds=seconds)
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


# --- 1-2. end to end -----------------------------------------------------------------------------


def test_1_a_refusal_at_the_tag_a_logged_fix_and_a_complete_drive_pass_every_mark(
    tmp_path, monkeypatch, capsys
):
    root, tagged, fix = at_a_fix(tmp_path, monkeypatch)
    first, second = logs(root)
    status, marks, results = run(root, capsys)
    assert status == 0, marks
    assert statuses(marks) == ["passed"] * 3, marks
    assert (results["commit"], results["tag_commit"], results["fix_commit"]) == (fix, tagged, fix)
    assert results["status"] == "passed" and results["not_seen"] == check.NOT_SEEN
    assert marks["one logged run"][1][-1].startswith(
        f"one logged run: {second.name}, the one complete sealed-run drive of {LIST}, at the "
        f"logged parsing fix {fix} of the tag frozen-plan ({tagged})"
    )
    assert f"{first.name}: stopped at step 1 (sealed-truth) at a layout refusal" in "\n".join(
        marks["logged parsing fixes"][1]
    )
    runs = [r["run"] for log in (first, second) for r in drive_fake.read_log(log) if r.get("run")]
    assert len(runs) == 3
    expected = {
        LIST,
        f"runs.noindex/drives/{first.name}",
        f"runs.noindex/drives/{second.name}",
        "data/heldout-openings.log",
        "data/heldout-refusals.log",
        *(f"{run_folder}/{name}" for run_folder in runs for name in ("manifest.json", "log.txt")),
    }
    assert set(results["inputs"]) == expected
    for name, digest in results["inputs"].items():
        assert digest == hashlib.sha256((root / name).read_bytes()).hexdigest(), name


def test_1_a_refusal_whose_last_opening_is_of_another_command_does_not_count(
    tmp_path, monkeypatch, capsys
):
    """The same history with an opening of the same bytes by another command logged by hand just
    before the earlier drive's refusal (the gate still accepts the fix): the refusal follows that
    opening, which lies in no step, so it is no layout refusal of the step and mark 2 fails."""
    root, tagged, _ = at_a_fix(tmp_path, monkeypatch)
    first, _ = logs(root)
    opened = [json.loads(line) for line in openings_file(root).read_text().splitlines()]
    other = {**opened[0], "command": [str(root / "fleetreserve" / "learn_run.py")]}
    lines = [opened[0], other, *opened[1:]]
    openings_file(root).write_text("".join(json.dumps(line) + "\n" for line in lines))
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert marks["logged parsing fixes"][0] == "failed", marks
    assert any(
        line.startswith(f"FAILED: {first.name} stopped after step 1 (sealed-truth)")
        and line.endswith(
            "and no layout refusal following an opening of that step is logged inside it"
        )
        for line in marks["logged parsing fixes"][1]
    ), marks


def test_1_the_refusal_missing_from_its_log_fails_the_fixes(tmp_path, monkeypatch, capsys):
    root, _, _ = at_a_fix(tmp_path, monkeypatch)
    refusals_file(root).write_text("")
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    lines = marks["logged parsing fixes"][1]
    assert any(line.startswith("COULD NOT RUN: the gate of the run from the tag") for line in lines)
    assert any("no layout refusal following an opening of that step" in line for line in lines)


def test_1_an_approved_fix_passes_every_mark_and_fails_without_its_refusal(
    tmp_path, monkeypatch, capsys
):
    """The history of test 1 with the owner's approval of the fix commit written by hand before
    the drive at the fix (the opener's reading H7): every mark passes; the refusal then emptied
    from its log, the fixes fail as they do without the approval."""
    root, tagged = made(tmp_path, monkeypatch)
    sealed_drive(1)
    fix = fixed(root)
    fake.approve(root, fix)
    sealed_drive(0)
    status, marks, results = run(root, capsys)
    assert status == 0, marks
    assert statuses(marks) == ["passed"] * 3, marks
    assert (results["commit"], results["tag_commit"], results["fix_commit"]) == (fix, tagged, fix)
    refusals_file(root).write_text("")
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert marks["logged parsing fixes"][0] == "failed", marks


def test_2_one_complete_drive_at_the_tag_passes_with_fixes_none(tmp_path, monkeypatch, capsys):
    root, tagged = made(tmp_path, monkeypatch, [truth(), learn()])
    sealed_drive(0)
    status, marks, results = run(root, capsys)
    assert status == 0, marks
    assert (results["tag_commit"], results["fix_commit"]) == (tagged, None)
    assert marks["logged parsing fixes"][1][-1].startswith(
        f"logged parsing fixes: none, the checked-out commit is the tag's {tagged}"
    )


# --- 3-6. one logged run -------------------------------------------------------------------------


def test_3_two_complete_drives_fail(tmp_path, monkeypatch, capsys):
    root, _ = made(tmp_path, monkeypatch, [truth(), learn()])
    sealed_drive(0)
    sealed_drive(0)
    first, second = logs(root)
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert (
        f"FAILED: 2 complete sealed-run drives, {first.name}, {second.name}: the run is made once"
    ) in marks["one logged run"][1]


def test_4_a_drive_after_the_complete_one_that_ran_a_step_fails(tmp_path, monkeypatch, capsys):
    root, _ = made(tmp_path, monkeypatch, [truth(), learn()])
    sealed_drive(0)
    (root / "data" / "refuse-now").write_text("")
    sealed_drive(1)
    first, second = logs(root)
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert marks["one logged run"][1] == [
        f"FAILED: {second.name}, a sealed-run drive after the complete drive {first.name}, ran "
        "1 step"
    ], marks


def test_4_a_drive_after_the_complete_one_that_ran_no_step_passes(tmp_path, monkeypatch, capsys):
    root, _ = made(tmp_path, monkeypatch, [truth(), learn()])
    sealed_drive(0)
    with (root / "runs.noindex" / "heavy.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the recorder refuses the step
        sealed_drive(2)
    _, second = logs(root)
    assert drive_fake.read_log(second)[2]["run"] is None
    status, marks, _ = run(root, capsys)
    assert status == 0, marks
    assert (
        f"{second.name}: ran no step (no entry names a run folder), not a run"
        in (marks["one logged run"][1])
    )


def test_5_an_opening_outside_every_step_or_by_another_command_inside_one_fails(
    tmp_path, monkeypatch, capsys
):
    """One opening by hand long before the drive, one in the last second of the complete drive's
    last step by another command: each lies inside no step."""
    root, tagged = made(tmp_path, monkeypatch, [truth(), learn()])
    sealed_drive(0)
    second = drive_fake.read_log(logs(root)[0])[4]["end"].removesuffix("Z")
    by_hand(root, "2026-01-01T00:00:00.000000Z", ["by hand"], tagged)
    by_hand(root, f"{second}.000000Z", [str(root / "fleetreserve" / "compare_run.py")], tagged)
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    for when in ("2026-01-01T00:00:00", second):
        assert (
            f"FAILED: the opening at {when}.000000Z of the asr file at commit {tagged} lies inside "
            "no step of a logged sealed-run drive"
        ) in marks["one logged run"][1], marks


def test_6_a_complete_drive_with_no_opening_fails(tmp_path, monkeypatch, capsys):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    openings_file(root).write_text("")
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert marks["one logged run"][1] == [
        f"FAILED: the complete drive {only.name} holds no opening of a sealed file"
    ], marks


# --- 7-9. logged parsing fixes -------------------------------------------------------------------


def test_7_an_earlier_drive_stopped_with_no_layout_refusal_in_its_step_fails(
    tmp_path, monkeypatch, capsys
):
    root, tagged = made(tmp_path, monkeypatch, [truth("--fail-at-tag"), learn()])
    sealed_drive(1)
    # A refusal of the step's opening logged by hand an hour after it, outside the stopped step,
    # so the gate accepts the fix.
    (opened,) = [json.loads(line) for line in openings_file(root).read_text().splitlines()]
    later = datetime.strptime(opened["time_utc"], check.STAMPS[1]) + timedelta(hours=1)
    refusal = {key: opened[key] for key in ("set", "file", "sha256", "commit", "tag_commit")}
    refusal = {"time_utc": later.strftime(check.STAMPS[1]), **refusal, "layout": "made up"}
    refusals_file(root).write_text(json.dumps(refusal) + "\n", encoding="utf-8")
    fixed(root)
    sealed_drive(0)
    first, _ = logs(root)
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert statuses(marks) == ["passed", "failed", "passed"], marks
    assert marks["logged parsing fixes"][1] == [
        f"FAILED: {first.name} stopped after step 1 (sealed-truth): the run ended with exit "
        "status 1, with 1 opening of sealed data in its steps, and no layout refusal following an "
        "opening of that step is logged inside it"
    ], marks


def test_7_a_drive_that_went_on_after_a_failed_second_build_is_no_stop_at_a_layout_refusal(
    tmp_path, monkeypatch, capsys
):
    """The second sealed truth build fails after the first passed: the driver logs it as failed
    and goes on (it parses the bytes the first parsed), so the drive is not one stopped at a layout
    refusal and mark 2 still fails. (Before this, the same list gave a drive stopped at step 2 at a
    layout refusal with an opening in step 1; the run list's rule 6 puts every truth build first, so
    a stop at a layout refusal after an earlier opening that passed can no longer be driven.)"""
    entries = [truth(label="sealed-truth-main"), truth("--refuse-at-tag"), learn()]
    root, _ = made(tmp_path, monkeypatch, entries)
    sealed_drive(1)
    fixed(root)
    sealed_drive(0)
    first, _ = logs(root)
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    (line,) = marks["logged parsing fixes"][1]
    assert line.startswith(f"FAILED: {first.name} ended with exit status 1; failed step 2"), line


def slow(root: Path, when: str | None) -> None:
    """Make the made-up truth build sleep before or after its opening, or neither (None)."""
    for name in ("before", "after"):
        (root / "data" / f"slow-{name}").unlink(missing_ok=True)
    if when is not None:
        (root / "data").mkdir(exist_ok=True)
        (root / "data" / f"slow-{when}").write_text("")


# The made-up truth build's time limit where a test makes it time out: long enough that a loaded
# machine still starts it and writes its opening (two git calls) well inside it, and the complete
# drive's truth build ends well inside it too, so the history is the one the test names.
SLOW_LIMIT = 20


def timed_out_then_complete(tmp_path, monkeypatch, when: str) -> Path:
    """A drive at the tag whose truth build times out (`SLOW_LIMIT`) before or after its opening,
    then a complete drive. The openings log is read before anything is judged: no opening after
    a time-out before it, one after a time-out after it."""
    root, _ = made(tmp_path, monkeypatch, [truth(max_seconds=SLOW_LIMIT), learn()])
    slow(root, when)
    assert run_driver.main([LIST, *SEAL]) != 0
    log = openings_file(root)
    opened = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    assert len(opened) == (1 if when == "after" else 0), (when, opened)
    time.sleep(1.1)
    slow(root, None)
    sealed_drive(0)
    first, _ = logs(root)
    assert drive_fake.read_log(first)[2]["timed_out"] is True
    return root


def test_8_an_earlier_drive_timed_out_before_any_opening_is_an_operational_stop(
    tmp_path, monkeypatch, capsys
):
    root = timed_out_then_complete(tmp_path, monkeypatch, "before")
    first, _ = logs(root)
    status, marks, _ = run(root, capsys)
    assert status == 0, marks
    assert any(
        line.startswith(f"{first.name}: ")
        and line.endswith("with no opening of sealed data in any of its steps: an operational stop")
        for line in marks["logged parsing fixes"][1]
    ), marks


def test_8_an_earlier_drive_timed_out_after_its_opening_fails(tmp_path, monkeypatch, capsys):
    root = timed_out_then_complete(tmp_path, monkeypatch, "after")
    first, _ = logs(root)
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert statuses(marks) == ["passed", "failed", "passed"], marks
    (line,) = marks["logged parsing fixes"][1]
    assert line.startswith(f"FAILED: {first.name} stopped after step 1 (sealed-truth): "), line
    assert line.endswith(
        "with 1 opening of sealed data in its steps; an earlier drive that opened sealed data "
        "counts only as stopped by a layout refusal"
    ), line


def test_9_an_earlier_drive_at_a_commit_off_the_line_from_the_tag_fails(
    tmp_path, monkeypatch, capsys
):
    root, tagged, _ = at_a_fix(tmp_path, monkeypatch)
    first, _ = logs(root)
    before = drive_fake.git(root, "rev-parse", f"{tagged}~1").strip()
    rewritten(first, lambda records: records[0].update(commit=before))
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert (
        f"FAILED: {first.name} was driven at commit {before}, neither the tag's nor one on the "
        "line from the tag to the checked-out commit"
    ) in marks["logged parsing fixes"][1], marks


def test_9_an_earlier_drive_with_another_tag_commit_fails(tmp_path, monkeypatch, capsys):
    root, tagged, _ = at_a_fix(tmp_path, monkeypatch)
    first, _ = logs(root)
    rewritten(first, lambda records: records[0].update(tag_commit="0" * 40))
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert (
        f"FAILED: {first.name} was driven with the tag's commit {'0' * 40}, not {tagged}: the tag "
        "never moves"
    ) in marks["logged parsing fixes"][1], marks


# --- 10-11. could not run: the complete drive, the times and the order ---------------------------


def _start_changed(key, value):
    return lambda records: records[0].update({key: value})


def _tree_changed(records):
    records[-1]["unchanged_after"] = False


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (_start_changed("list", "run-lists/other.json"), "{name} is not a drive of the bytes of"),
        (_start_changed("list_sha256", "0" * 64), "{name} is not a drive of the bytes of"),
        (_start_changed("steps", []), "{name} is not a drive of the bytes of"),
        (_tree_changed, "the tree changed after the last step of {name}"),
        (None, "sealed-learn: {run}/step.txt does not"),
    ],
    ids=["another list", "another list hash", "other steps", "the tree changed", "output changed"],
)
def test_10_a_complete_drive_of_another_list_a_changed_tree_or_output_could_not_run(
    tmp_path, monkeypatch, capsys, change, said
):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    learned = drive_fake.read_log(only)[4]["run"]
    if change is None:
        (root / learned / "step.txt").write_text("changed\n")
    else:
        rewritten(only, change)
    status, marks, _ = run(root, capsys)
    assert status == 2, marks
    expected = "COULD NOT RUN: " + said.format(name=only.name, run=learned)
    assert any(line.startswith(expected) for line in marks["one logged run"][1]), marks


C, T, F = "c" * 40, "t" * 40, "f" * 40


class MadeUpGate:
    """A gate's `GateCommit` as `_verify_complete` reads it: the tag's commit and the fix's."""

    fix_commit = "f" * 40

    def __str__(self) -> str:
        return "t" * 40


@pytest.mark.parametrize(
    ("start", "gate", "said"),
    [
        ({"commit": C}, None, None),
        ({"commit": "d" * 40}, None, f"is stale: written at commit {'d' * 40}"),
        ({"commit": C, "tag_commit": T, check.FIX_KEY: F}, MadeUpGate(), None),
        ({"commit": C, "tag_commit": "u" * 40, check.FIX_KEY: F}, MadeUpGate(), "records"),
        ({"commit": C, "tag_commit": T, check.FIX_KEY: "g" * 40}, MadeUpGate(), "records"),
        ({"commit": C, "tag_commit": T}, MadeUpGate(), "records other commits"),
    ],
    ids=["at HEAD", "stale", "the gate's tag and fix", "another tag", "another fix", "no fix"],
)  # fmt: skip
def test_10_a_complete_drive_is_verified_only_at_the_checked_out_commit_and_the_gates_commits(
    start, gate, said
):
    """`_verify_complete` by hand, up to the committed list (none here, so nothing more is
    verified): a drive written at another commit is stale, and one whose tag or fix commit is
    not the gate's could not run; with no gate, the tag and fix are not compared."""
    lines = []
    found = check.Findings(lines.append)
    surveyed = check.Survey("c" * 40, gate, [], 0, None, None, [])
    drive = check.Drive("d.jsonl", [{"record": "start", **start}, {"record": "end"}], [], None)
    assert check._verify_complete(Path("."), surveyed, drive, found) is None
    if said is None:
        assert lines == [] and not found.cannot, lines
    else:
        assert len(lines) == 1 and said in lines[0] and found.cannot, lines


def a_step_ending_before_it_begins(records):
    for key in ("start", "end"):
        records[2][key] = moved(records[1]["start"], -60)


def a_step_beginning_before_the_last_ended(records):
    records[3]["start"] = moved(records[2]["end"], -60)


def an_end_line_before_the_last_step(records):
    records[-1]["end"] = moved(records[-2]["end"], -60)


@pytest.mark.parametrize(
    ("change", "where"),
    [
        (a_step_ending_before_it_begins, "step 1 (sealed-truth)"),
        (a_step_beginning_before_the_last_ended, "step 2 (sealed-learn)"),
        (an_end_line_before_the_last_step, "its end line"),
    ],
    ids=["entry before its begin", "begin before the last entry", "end before the last entry"],
)
def test_11_a_drive_log_whose_times_go_back_could_not_run(
    tmp_path, monkeypatch, capsys, change, where
):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (path,) = logs(root)
    rewritten(path, change)
    status, marks, _ = run(root, capsys)
    assert status == 2, marks
    for name in MARKS:
        assert marks[name][0] == "could not run", marks
        assert any(
            line.startswith(f"COULD NOT RUN: the times in the log {path.name} go back at {where}")
            for line in marks[name][1]
        ), marks


def test_11_a_drive_log_renamed_could_not_run(tmp_path, monkeypatch, capsys):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (path,) = logs(root)
    start = json.loads(path.read_bytes().splitlines()[0])["start"]
    later = datetime.strptime(moved(start, 2), "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%dT%H%M%SZ")
    renamed = path.with_name(f"{later}-sealed-run.jsonl")
    path.rename(renamed)
    status, marks, _ = run(root, capsys)
    assert status == 2, marks
    assert statuses(marks) == ["could not run"] * 3, marks
    assert any(
        line.startswith(f"COULD NOT RUN: the name of the log {renamed.name} is neither the second")
        for line in marks["one logged run"][1]
    ), marks


def clock_set_back_before(root: Path, path: Path, seconds: int) -> Path:
    """What an honest driver writes when its clock was set back by `seconds` just before the
    drive of the log at `path` started: every time in its log moved back, the log named by its
    new start, and every opening and refusal logged during the drive moved back; the new path."""
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    low = datetime.strptime(records[0]["start"], "%Y-%m-%dT%H:%M:%SZ")
    high = datetime.strptime(records[-1]["end"], "%Y-%m-%dT%H:%M:%SZ") + timedelta(seconds=1)

    def back(records):
        for record in records:
            for key in ("start", "end"):
                if key in record:
                    record[key] = moved(record[key], -seconds)

    rewritten(path, back)
    for log in (openings_file(root), refusals_file(root)):
        lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        for line in lines:
            at = datetime.strptime(line["time_utc"], "%Y-%m-%dT%H:%M:%S.%fZ")
            if low <= at < high:
                line["time_utc"] = (at - timedelta(seconds=seconds)).strftime(check.STAMPS[1])
        log.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    stamp = (low - timedelta(seconds=seconds)).strftime("%Y%m%dT%H%M%SZ")
    new = path.with_name(f"{stamp}-{path.name.split('-', 1)[1]}")
    path.rename(new)
    return new


def test_11_openings_logged_out_of_the_order_of_the_drives_could_not_run(
    tmp_path, monkeypatch, capsys
):
    """A complete drive at the tag, then a drive that opens sealed data and stops at a logged
    layout refusal (mark 1 fails). With the clock set back an hour between the two, the later
    drive's start line comes first, a layout stop before the complete drive by the start lines
    alone; its opening is logged after the complete drive's, so marks 1 and 2 could not run."""
    root, _ = made(tmp_path, monkeypatch, [truth(), learn()])
    sealed_drive(0)
    (root / "data" / "refuse-now").write_text("")
    sealed_drive(1)
    first, later = logs(root)
    earlier = clock_set_back_before(root, later, 3600)
    assert logs(root) == [earlier, first]
    opened = [json.loads(line) for line in openings_file(root).read_text().splitlines()]
    said = (
        f"COULD NOT RUN: the opening at {opened[1]['time_utc']} lies in {earlier.name} but is "
        f"logged after the opening at {opened[0]['time_utc']}, which lies in {first.name}, a "
        "drive with a later start line: the drives did not run in the order of their start "
        "lines, a clock set back between them"
    )
    status, marks, _ = run(root, capsys)
    assert status == 2, marks
    assert statuses(marks) == ["could not run", "could not run", "passed"], marks
    for name in MARKS[:2]:
        assert said in marks[name][1], marks


# --- 12. logs hashed -----------------------------------------------------------------------------


def test_12_a_manifest_changed_after_the_drive_fails(tmp_path, monkeypatch, capsys):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    record = drive_fake.read_log(only)[2]
    manifest = root / record["run"] / "manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert marks["logs hashed"][1] == [
        f"FAILED: {only.name}, step 1 (sealed-truth): the SHA-256 of {record['run']}/"
        "manifest.json is not the one the log records"
    ], marks


def test_12_a_run_log_changed_after_the_drive_fails(tmp_path, monkeypatch, capsys):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    record = drive_fake.read_log(only)[4]
    with (root / record["run"] / "log.txt").open("a") as handle:
        handle.write("changed\n")
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert marks["logs hashed"][1] == [
        f"FAILED: {only.name}, step 2 (sealed-learn): {record['run']}/log.txt is not listed in "
        "its manifest with its current SHA-256"
    ], marks


# --- 13. could not run, every mark ---------------------------------------------------------------


def no_list(tmp_path, monkeypatch):
    root, _ = made(tmp_path, monkeypatch, listed=False)
    return root, "COULD NOT RUN: no run list is committed at run-lists/run-list.json"


def no_drive_log(tmp_path, monkeypatch):
    root, _ = made(tmp_path, monkeypatch, [truth(), learn()])
    return root, "COULD NOT RUN: no sealed-run drive log, runs.noindex/drives/"


def last_line_cut_off(tmp_path, monkeypatch):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (path,) = logs(root)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]) + lines[-1][:20])
    return root, f"COULD NOT RUN: the log {path.name} is empty or its last line is cut off"


def hash_not_recomputing(tmp_path, monkeypatch):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (path,) = logs(root)
    lines = path.read_bytes().splitlines(keepends=True)
    end = json.loads(lines[-1])
    end["seconds"] += 1.0
    path.write_bytes(b"".join(lines[:-1]) + run_driver.encode(end))
    return root, f"COULD NOT RUN: the hash of the log {path.name} does not recompute"


def head_elsewhere(tmp_path, monkeypatch):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (root / "tracked.txt").write_text("a later commit\n")
    fake.commit(root, "a commit after the tag that is no logged parsing fix")
    return root, "COULD NOT RUN: the gate of the run from the tag refuses the checked-out commit"


def untracked_file(tmp_path, monkeypatch):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (root / "untracked.txt").write_text("x\n")
    return root, "the git tree has uncommitted changes"


CANNOT = [
    no_list,
    no_drive_log,
    last_line_cut_off,
    hash_not_recomputing,
    head_elsewhere,
    untracked_file,
]


@pytest.mark.parametrize("case", CANNOT)
def test_13_every_mark_could_not_run(tmp_path, monkeypatch, capsys, case):
    root, said = case(tmp_path, monkeypatch)
    status, marks, results = run(root, capsys)
    assert status == 2, marks
    assert results["status"] == "could not run"
    for name in MARKS:
        assert marks[name][0] == "could not run", (name, marks)
        assert any(said in line for line in marks[name][1]), (name, marks)
        assert not any(line.startswith("FAILED") for line in marks[name][1]), (name, marks)


def test_13_not_under_the_run_recorder_could_not_run_and_writes_nothing(
    tmp_path, monkeypatch, capsys, recorder_folder
):
    """The variable unset, empty or naming no folder; and, by the gate mark's rule
    (`gate_mark.run_folder`), a folder outside the root's runs folder, one below a folder in it,
    and a run folder in it that holds a manifest already (a finished run): each could not run,
    and no results file is written anywhere."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    capsys.readouterr()
    outside, below, finished = (
        tmp_path / "outside-run",
        root / "runs.noindex" / "a-run" / "below",
        root / "runs.noindex" / "finished-run",
    )
    for folder in (outside, below, finished):
        folder.mkdir(parents=True)
    (finished / "manifest.json").write_text("{}\n", encoding="utf-8")
    for value in (
        None,
        "",
        str(tmp_path / "no-such-folder"),
        *map(str, (outside, below, finished)),
    ):
        if value is None:
            monkeypatch.delenv(check.RUN_DIR_VARIABLE)
        else:
            monkeypatch.setenv(check.RUN_DIR_VARIABLE, value)
        assert check.main(["--root", str(root)]) == 2, value
        assert capsys.readouterr().out.startswith("COULD NOT RUN: not under the run recorder")
    assert not recorder_folder.exists()
    assert list(outside.iterdir()) == [] and list(below.iterdir()) == []
    assert [path.name for path in finished.iterdir()] == ["manifest.json"]


def test_a_drive_log_with_no_end_line_could_not_run(tmp_path, monkeypatch, capsys):
    """A drive killed with no end line is never forgiven as broken off: here a drive at the tag
    timed out before any opening, its log then cut after its step's `begin`, and a complete drive
    after it; every mark could not run, naming the log, which blocks the mark until it is
    settled by hand."""
    root = timed_out_then_complete(tmp_path, monkeypatch, "before")
    first, _ = logs(root)
    lines = first.read_bytes().splitlines(keepends=True)
    first.write_bytes(b"".join(lines[:2]))  # the start and the step's begin, every line whole
    status, marks, _ = run(root, capsys)
    assert status == 2, marks
    said = (
        f"COULD NOT RUN: the log {first.name} has no end line: the drive broke off or is still "
        "going"
    )
    for name in MARKS:
        assert marks[name][0] == "could not run", marks
        assert said in marks[name][1], marks


# --- 14. nothing sealed opened -------------------------------------------------------------------


def test_14_the_check_opens_lists_or_scans_nothing_sealed(tmp_path, monkeypatch, capsys):
    root, _, _ = at_a_fix(tmp_path, monkeypatch)
    sealed = root / "data" / "heldout" / fake.ASR_FILE[0] / fake.ASR_FILE[1]
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(b"made-up sealed bytes\n")
    with watch(root / "data" / "heldout") as seen:
        status, marks, _ = run(root, capsys)
    assert status == 0, marks
    assert seen == []


def test_14_the_routes_give_the_loaders_own_log_paths_and_refusals(tmp_path):
    root = fake.sealed_repo(tmp_path, {fake.ASR_FILE: b"x\n"}, tag=False, package=False)
    assert sealed_route.sealed_logs(root) == (
        heldout_loader.log_path(root),
        heldout_loader.refusals_path(root),
    )
    with watch(root / "data" / "heldout") as seen:
        assert sealed_route.refusals_log(root) == (heldout_loader.refusals_path(root), [])
    assert seen == []
    fake.log_refusal(root, fake.ASR_FILE, b"x\n", "0" * 40)
    path, refusals = sealed_route.refusals_log(root)
    assert refusals == heldout_loader.logged_refusals(root)
    assert [(line["set"], line["commit"]) for line in refusals] == [("asr", "0" * 40)]
    path.write_text('{"time_utc": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 does not have the log's fields"):
        sealed_route.refusals_log(root)


# --- 15. exit statuses ---------------------------------------------------------------------------


def script(*args, extra=None) -> subprocess.CompletedProcess:
    environment = fake.tool_environment()
    environment.update(extra or {})
    Path(environment[check.RUN_DIR_VARIABLE]).mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=300,
    )


def test_15_the_script_gives_each_exit_status(tmp_path, monkeypatch):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    assert script("--root", str(root)).returncode == 0
    first = openings_file(root).read_text().splitlines()[0]
    with openings_file(root).open("a", encoding="utf-8") as handle:
        handle.write(first.replace('"time_utc": "20', '"time_utc": "19', 1) + "\n")
    assert script("--root", str(root)).returncode == 1
    (root / "untracked.txt").write_text("x\n")
    assert script("--root", str(root)).returncode == 1  # a failure outranks could not run
    openings_file(root).write_text(first + "\n")
    assert script("--root", str(root)).returncode == 2
    assert script("everything").returncode == 2  # a wrong call


def test_15_a_failed_import_could_not_run(tmp_path):
    broken = tmp_path / "fleetreserve"
    broken.mkdir()
    (broken / "__init__.py").write_text("raise ImportError('a made-up broken package')\n")
    done = script(extra={"PYTHONPATH": str(tmp_path)})
    assert done.returncode == 2
    assert done.stdout.startswith("COULD NOT RUN: the check cannot start: ")


def test_15_an_error_in_a_mark_could_not_run_and_an_earlier_failure_still_decides(
    tmp_path, monkeypatch, capsys
):
    """An error no reading foresees while mark 2 judges a drive: that mark could not run, the
    other marks are judged; with a failure in mark 1 (no complete drive) the exit is 1, without
    one 2."""
    root, _ = made(tmp_path, monkeypatch)
    sealed_drive(1)

    def broken(*args):
        raise RuntimeError("made up")

    monkeypatch.setattr(check, "_earlier", broken)
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert statuses(marks) == ["failed", "could not run", "passed"], marks
    assert marks["logged parsing fixes"][1] == ["COULD NOT RUN: RuntimeError: made up"], marks
    monkeypatch.setattr(check, "check_run", lambda *args: None)
    status, marks, _ = run(root, capsys)
    assert status == 2, marks
    assert statuses(marks) == ["passed", "could not run", "passed"], marks


# --- 16. under the run recorder ------------------------------------------------------------------


def test_16_under_the_run_recorder_the_manifest_lists_the_results_file(tmp_path, monkeypatch):
    root, _ = made(tmp_path, monkeypatch, [truth(), learn()])
    sealed_drive(0)
    environment = {
        key: value
        for key, value in fake.tool_environment().items()
        if key != check.RUN_DIR_VARIABLE
    }
    done = subprocess.run(
        [
            sys.executable, "-m", "fleetreserve.run_record", "--name", "wp14-passmark", "--",
            sys.executable, str(TOOL), "--root", str(root),
        ],
        cwd=root, capture_output=True, text=True, check=False, env=environment, timeout=300,
    )  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    (folder,) = [p for p in drive_fake.run_folders(root) if p.name.endswith("-wp14-passmark")]
    manifest = json.loads((folder / "manifest.json").read_text())
    results = folder / check.RESULTS_NAME
    assert manifest["exit_code"] == 0, manifest
    assert (
        manifest["outputs"][check.RESULTS_NAME] == hashlib.sha256(results.read_bytes()).hexdigest()
    )
    assert json.loads(results.read_text())["status"] == "passed"


# --- hand cases: how a line is matched to a step -------------------------------------------------


MODULE, MODULE_FILE = ["python", "-m", "fleetreserve.made_up"], "/r/fleetreserve/made_up.py"


@pytest.mark.parametrize(
    ("command", "argv", "same"),
    [
        ([*MODULE, "--x", "1"], [MODULE_FILE, "--x", "1"], True),
        ([*MODULE, "--x", "1"], [MODULE_FILE, "--x", "2"], False),
        (["python", "-m", "fleetreserve.made_up"], ["/r/fleetreserve/other.py"], False),
        (["python", "-m", "other.made_up"], ["/r/other/made_up.py"], False),
        (["python", "fleetreserve.made_up"], ["/r/fleetreserve/made_up.py"], False),
        ("python -m fleetreserve.made_up", ["/r/fleetreserve/made_up.py"], False),
        (["python", "-m", "fleetreserve.made_up"], None, False),
        (["python", "-m", "fleetreserve.made_up"], [], False),
    ],
    ids=["same", "other arguments", "another module", "outside the package", "no -m",
         "step command text", "no opener command", "empty opener command"],
)  # fmt: skip
def test_an_openings_command_is_the_steps_only_as_its_module_file_and_arguments(
    command, argv, same
):
    assert check._step_command({"command": argv}, {"command": command}) is same


IN_START = {"record": "start", "commit": "c" * 40, "tag_commit": "t" * 40}
IN_BEGIN = {
    "record": "begin",
    "index": 0,
    "label": "a",
    "start": "2026-01-01T10:00:01Z",
    "command": ["python", "-m", "fleetreserve.made_up"],
}
IN_ENTRY = {"record": "entry", "index": 0, "label": "a", "end": "2026-01-01T10:00:05Z", "run": "r"}


def in_line(time, **change):
    line = {
        "commit": "c" * 40,
        "tag_commit": "t" * 40,
        "time_utc": f"2026-01-01T{time}.000000Z",
        "command": ["/x/fleetreserve/made_up.py"],
    }
    return {**line, **change}


def test_a_line_lies_inside_a_step_by_commit_command_and_time_to_the_second():
    """`inside`: the line's commit and tag commit are the drive's, its command (when it has one)
    the step's, its time from the step's begin to its end, both seconds included; `opened_in`
    only in a step that ran (a run folder)."""
    drive = check.Drive("d", [IN_START], [(IN_BEGIN, IN_ENTRY)], None)
    for second, held in (("10:00:03", True), ("10:00:01", True), ("10:00:05", True)):
        assert check.inside(in_line(second), drive, 0) is held
    for line in (
        in_line("10:00:00"),
        in_line("10:00:06"),
        in_line("10:00:03", commit="d" * 40),
        in_line("10:00:03", tag_commit="d" * 40),
        in_line("10:00:03", command=["/x/other.py"]),
    ):
        assert not check.inside(line, drive, 0), line
    no_command = {k: v for k, v in in_line("10:00:03").items() if k != "command"}
    assert check.inside(no_command, drive, 0)
    assert check.opened_in(in_line("10:00:03"), drive, 0)
    no_run = check.Drive("d", [IN_START], [(IN_BEGIN, {**IN_ENTRY, "run": None})], None)
    assert not check.opened_in(in_line("10:00:03"), no_run, 0)


def two_steps(second_begin: str, second_end: str) -> "check.Drive":
    """A made-up drive at commit c and tag commit t: step 1 (a) passes from 10:00:01 to 10:00:05,
    step 2 (b), of the same command, runs from `second_begin` to `second_end` and stops with exit
    status 1."""
    command = [sys.executable, "-m", "fleetreserve.truth_build"]
    ran = {
        "exit_status": 0,
        "timed_out": False,
        "interrupted": None,
        "manifest_sha256": "0" * 64,
        "run": "runs.noindex/made-up",
    }
    steps = [
        (
            {"record": "begin", "index": 0, "label": "a", "command": command,
             "start": "2026-01-01T10:00:01Z"},
            {"record": "entry", "index": 0, "label": "a", "end": "2026-01-01T10:00:05Z", **ran},
        ),
        (
            {"record": "begin", "index": 1, "label": "b", "command": command,
             "start": f"2026-01-01T{second_begin}Z"},
            {"record": "entry", "index": 1, "label": "b", "end": f"2026-01-01T{second_end}Z",
             **ran, "exit_status": 1},
        ),
    ]  # fmt: skip
    stop = {"record": "stop", "index": 1, "label": "b", "reason": "the run ended with exit 1"}
    return check.Drive("made-up.jsonl", [IN_START, stop, {"record": "end"}], steps, stop)


SAME_FILE = {"set": "asr", "file": "f.csv", "sha256": "0" * 64, "commit": "c" * 40}
SAME_FILE["tag_commit"] = "t" * 40
ARGV = ["/made/up/src/fleetreserve/truth_build.py"]


def opening(when: str, **change) -> dict:
    return {**SAME_FILE, "time_utc": f"2026-01-01T{when}Z", "command": ARGV, **change}


def refusal(when: str, **change) -> dict:
    return {**SAME_FILE, "time_utc": f"2026-01-01T{when}Z", **change}


def test_a_refusal_counts_for_a_step_through_the_last_opening_of_its_bytes_made_in_it():
    """Step 2 opens at 10:00:07 and is refused at 10:00:08: a layout stop. The refusal follows
    the last opening before it (or in its microsecond) of its set, file, SHA-256, commit and tag
    commit; an opening of other bytes between, or of the same after, changes nothing; the same
    bytes opened between by another command, or the refusal logged after the step ended, or no
    opening, is no layout refusal of the step."""
    drive = two_steps("10:00:06", "10:00:09")
    own, refused = opening("10:00:07.000000"), refusal("10:00:08.000000")
    assert check._earlier(drive, [refused], [own])[0] == "layout"
    assert check._earlier(drive, [refusal("10:00:07.000000")], [own])[0] == "layout"
    for key, value in (("set", "x"), ("file", "x"), ("sha256", "x"), ("commit", "d" * 40),
                       ("tag_commit", "d" * 40)):  # fmt: skip
        other = opening("10:00:07.500000", **{key: value}, command=["/made/up/other.py"])
        assert check._earlier(drive, [refused], [own, other])[0] == "layout", key
    after = opening("11:00:00.000000")
    assert check._earlier(drive, [refused], [own, after])[0] == "layout"
    by_another = opening("10:00:07.500000", command=["/made/up/src/fleetreserve/learn_run.py"])
    for refusals, openings in (
        ([refused], [own, by_another]),
        ([refusal("10:05:00.000000")], [own]),
    ):
        kind, text = check._earlier(drive, refusals, openings)
        assert kind == "fail" and "no layout refusal following an opening" in text, text
    assert not check.refused_in(refused, [], drive, 1)


def test_an_opening_in_a_second_two_steps_of_one_command_share():
    """Steps 1 and 2 of one command share 10:00:05. An opening then, followed by step 2's
    refusal, is step 2's: a layout stop. One then followed by no refusal (the refusal follows a
    later opening of step 2 alone) cannot be told between them: could not run. One inside step 1
    alone, which passed, fails."""
    drive = two_steps("10:00:05", "10:00:06")
    shared = opening("10:00:05.300000")
    assert [check.inside(shared, drive, k) for k in (0, 1)] == [True, True]
    assert check._earlier(drive, [refusal("10:00:05.400000")], [shared])[0] == "layout"
    own = opening("10:00:06.100000")
    kind, text = check._earlier(drive, [refusal("10:00:06.200000")], [shared, own])
    assert kind == "cannot" and "which step opened it cannot be told" in text, text
    early = opening("10:00:02.000000")
    kind, text = check._earlier(drive, [refusal("10:00:06.200000")], [early, own])
    assert kind == "fail" and "inside step 1, which passed before it" in text, text


@pytest.mark.parametrize(
    "change",
    [
        {"stop": None},
        {"entry": {"manifest_sha256": None}},
        {"entry": {"timed_out": True}},
        {"entry": {"timed_out": None}},
        {"entry": {"interrupted": "SIGINT"}},
        {"entry": {"exit_status": 0}},
    ],
    ids=["no stop line", "no manifest", "timed out", "no timed-out field", "interrupted", "passed"],
)
def test_a_drive_that_opened_sealed_data_counts_only_when_its_last_step_plainly_failed(change):
    drive = two_steps("10:00:06", "10:00:09")
    begin, entry = drive.steps[1]
    drive.steps[1] = (begin, {**entry, **change.get("entry", {})})
    drive.stop = change.get("stop", drive.stop)
    kind, text = check._earlier(drive, [refusal("10:00:08.000000")], [opening("10:00:07.000000")])
    assert kind == "fail" and "counts only as stopped by a layout refusal" in text, text


@pytest.mark.parametrize(
    ("change", "complete"),
    [
        ({}, True),
        ({"end": {"completed": "yes"}}, False),
        ({"end": {"exit_status": 1}}, False),
        ({"stop": {"record": "stop", "index": 0, "label": "a", "reason": "made up"}}, False),
        ({"end": {"entries_run": 2}}, False),
        ({"start": {"entries": 2}}, False),
        ({"entry": {"timed_out": True}}, False),
    ],
    ids=["every condition", "completed not true", "exit 1", "a stop line", "entries run",
         "the list's entries", "a step that did not pass"],
)  # fmt: skip
def test_a_drive_is_complete_only_when_every_condition_holds(change, complete):
    begin = {"record": "begin", "index": 0, "label": "a", "start": "2026-01-01T10:00:01Z"}
    entry = {
        "record": "entry",
        "index": 0,
        "label": "a",
        "exit_status": 0,
        "timed_out": False,
        "interrupted": None,
        "manifest_sha256": "0" * 64,
        **change.get("entry", {}),
    }
    start = {"record": "start", "entries": 1, **change.get("start", {})}
    end = {"record": "end", "completed": True, "exit_status": 0, "entries_run": 1}
    end.update(change.get("end", {}))
    drive = check.Drive("d", [start, end], [(begin, entry)], change.get("stop"))
    assert drive.complete() is complete


def hand_log(path: Path, records: list[dict]) -> Path:
    """A drive log written by hand: `records`, then an end line whose hash recomputes."""
    above = b"".join(run_driver.encode(record) for record in records)
    last = {"record": "end", "end": "2026-01-01T10:00:09Z", "completed": False}
    last["log_sha256"] = run_driver.end_digest(above, last)
    path.write_bytes(above + run_driver.encode(last))
    return path


HAND_START = {"record": "start", "format": run_driver.LOG_FORMAT, "purpose": "sealed-run"}
HAND_START["start"] = "2026-01-01T10:00:00Z"
HAND_BEGIN = {"record": "begin", "index": 0, "label": "a", "start": "2026-01-01T10:00:01Z"}
HAND_ENTRY = {"record": "entry", "index": 0, "label": "a", "start": "2026-01-01T10:00:01Z"}
HAND_ENTRY["end"] = "2026-01-01T10:00:02Z"
HAND_STOP = {"record": "stop", "index": 0, "label": "a", "reason": "made up"}


@pytest.mark.parametrize(
    ("lines", "said"),
    [
        ([HAND_START, {**HAND_BEGIN, "record": "begun"}, HAND_ENTRY], "at step 1"),
        ([HAND_START, HAND_BEGIN, {**HAND_ENTRY, "record": "end of step"}], "at step 1"),
        ([HAND_START, HAND_BEGIN, {**HAND_ENTRY, "index": 1}], "at step 1"),
        ([HAND_START, HAND_BEGIN, {**HAND_ENTRY, "label": "b"}], "at step 1"),
        ([HAND_START, HAND_STOP, HAND_BEGIN, HAND_ENTRY], "at step 1"),
        ([HAND_START, HAND_BEGIN], "at step 1"),
        ([{**HAND_START, "purpose": "rehearsal"}, HAND_BEGIN, HAND_ENTRY], "purpose"),
        ([HAND_START, {**HAND_BEGIN, "start": "2026-01-01T10:00:03Z"}, HAND_ENTRY], "go back"),
        ([{**HAND_START, "start": "2026-01-01T10:00:02Z"}, HAND_BEGIN, HAND_ENTRY], "name"),
    ],
    ids=["a begin of another record", "an entry of another record", "an entry of another index",
         "an entry of another label", "a stop before a step", "a begin with no entry",
         "another purpose", "a time going back", "a start two seconds after the name"],
)  # fmt: skip
def test_a_drive_log_not_of_the_drivers_lines_times_or_name_could_not_run(tmp_path, lines, said):
    path = hand_log(tmp_path / "20260101T100000Z-sealed-run.jsonl", lines)
    with pytest.raises(check.CannotCheck) as refused:
        check.read_drive(path)
    assert said in str(refused.value), refused.value
    drive = check.read_drive(hand_log(path, [HAND_START, HAND_BEGIN, HAND_ENTRY, HAND_STOP]))
    assert (drive.steps, drive.stop) == ([(HAND_BEGIN, HAND_ENTRY)], HAND_STOP)
    link = tmp_path / "20260101T100001Z-sealed-run.jsonl"
    link.symlink_to(path.name)
    with pytest.raises(check.CannotCheck, match="is a symbolic link"):
        check.read_drive(link)


def in_order(openings: list[dict]) -> tuple[int, list[str]]:
    """`_in_drive_order` over two made-up drives, d0 from 10:00:01 to 10:00:05 and d1 from
    10:00:05 to 10:00:09 (they share 10:00:05), and an earlier one with no step."""
    said: list[str] = []
    found = check.Findings(said.append)
    drives = [check.Drive("none", [IN_START], [], None)]
    for name, begin, end in (("d0", "10:00:01", "10:00:05"), ("d1", "10:00:05", "10:00:09")):
        step = (
            {**IN_BEGIN, "start": f"2026-01-01T{begin}Z"},
            {**IN_ENTRY, "end": f"2026-01-01T{end}Z"},
        )
        drives.append(check.Drive(name, [IN_START], [step], None))
    check._in_drive_order(drives, openings, found)
    return found.status(), said


def test_openings_must_be_logged_in_the_order_of_the_drives_they_lie_in():
    for openings in (
        [in_line("10:00:02"), in_line("10:00:07")],
        [in_line("10:00:07"), in_line("10:00:30"), in_line("10:00:08")],
        [in_line("10:00:30"), in_line("10:00:02")],
        [in_line("10:00:02"), in_line("10:00:05"), in_line("10:00:03")],
        [in_line("10:00:07"), in_line("10:00:05"), in_line("10:00:08")],
        [],
    ):
        assert in_order(openings) == (0, []), openings
    for openings in (
        [in_line("10:00:07"), in_line("10:00:02")],
        [in_line("10:00:07"), in_line("10:00:05"), in_line("10:00:02")],
    ):
        status, said = in_order(openings)
        assert status == 2 and len(said) == 1, said
        assert said[0].startswith(
            "COULD NOT RUN: the opening at 2026-01-01T10:00:02.000000Z lies in d0 but is logged "
            "after the opening at 2026-01-01T10:00:07.000000Z, which lies in d1"
        ), said


# --- guards pinned by a break each: unreadable and linked inputs, the fix in the gate's mark ----


def broken_line(path: Path) -> None:
    """One line that is no JSON at the end of the log at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")


def test_an_unreadable_openings_log_makes_marks_1_and_2_could_not_run_naming_it(
    tmp_path, monkeypatch, capsys
):
    """A complete drive at the tag, then one line that is no JSON in the openings log: marks 1
    and 2 could not run, each naming the log first; mark 3 hashes the log and passes."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    broken_line(openings_file(root))
    status, marks, _ = run(root, capsys)
    assert status == 2, marks
    assert statuses(marks) == ["could not run", "could not run", "passed"], marks
    for name in MARKS[:2]:
        assert marks[name][1][0].startswith("COULD NOT RUN: the openings log cannot be read: "), (
            marks
        )


def test_an_unreadable_refusal_log_never_lets_a_failing_history_pass(tmp_path, monkeypatch, capsys):
    """At the tag, a drive whose truth build opened and was refused (data/refuse-now), the
    refusal then taken out of its log, then a complete drive: mark 2 fails the first drive, which
    opened sealed data and holds no layout refusal (exit 1). One line that is no JSON in the
    refusal log then makes marks 1 and 2 could not run, each naming the log first: the exit is 2,
    never 0 (mark 2 judges no earlier drive against logs it cannot read). No step times out, so
    the history does not hang on the machine's load."""
    root, _ = made(tmp_path, monkeypatch, [truth(), learn()])
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "refuse-now").write_text("")
    sealed_drive(1)
    (root / "data" / "refuse-now").unlink()
    refusals_file(root).write_text("", encoding="utf-8")
    sealed_drive(0)
    status, marks, _ = run(root, capsys)
    assert status == 1 and statuses(marks) == ["passed", "failed", "passed"], marks
    broken_line(refusals_file(root))
    status, marks, _ = run(root, capsys)
    assert status == 2, marks
    assert statuses(marks) == ["could not run", "could not run", "passed"], marks
    for name in MARKS[:2]:
        assert marks[name][1][0].startswith("COULD NOT RUN: the refusal log cannot be read: "), (
            marks
        )


def linked_away(path: Path, target: Path) -> None:
    """The file at `path` replaced by a symbolic link to `target`, which gets its bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)


def test_an_openings_log_that_is_a_symbolic_link_makes_mark_3_could_not_run(
    tmp_path, monkeypatch, capsys
):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    linked_away(openings_file(root), tmp_path / "openings-copy.log")
    status, marks, results = run(root, capsys)
    assert status == 2, marks
    assert marks["logs hashed"][0] == "could not run", marks
    (line,) = marks["logs hashed"][1]
    assert line.startswith(
        "COULD NOT RUN: the input data/heldout-openings.log cannot be hashed: "
    ), line
    assert results["inputs"]["data/heldout-openings.log"] == "cannot be hashed"


def test_a_manifest_that_is_a_symbolic_link_makes_mark_3_could_not_run(
    tmp_path, monkeypatch, capsys
):
    """A step's manifest a link to a copy of itself: mark 3 could not run, naming it twice, as an
    input that cannot be hashed and as a link not read (never a failure: its bytes are unseen)."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    run_folder = drive_fake.read_log(only)[2]["run"]
    linked_away(root / run_folder / "manifest.json", tmp_path / "manifest-copy.json")
    status, marks, _ = run(root, capsys)
    assert status == 2, marks
    assert marks["logs hashed"][0] == "could not run", marks
    hashed, read = marks["logs hashed"][1]
    assert hashed.startswith(
        f"COULD NOT RUN: the input {run_folder}/manifest.json cannot be hashed: "
    ), hashed
    assert read == (
        f"COULD NOT RUN: {only.name}, step 1 (sealed-truth): {run_folder}/manifest.json is a "
        "symbolic link, not read"
    )


@pytest.mark.parametrize("name", ["manifest.json", "log.txt"])
def test_mark_3_never_opens_a_manifest_or_run_log_that_links_into_the_sealed_folder(
    tmp_path, monkeypatch, name
):
    """Mark 3 alone (`check_hashes`, on the survey the check reads first): a step's manifest or
    its log.txt a symbolic link to a made-up file of the same bytes in the sealed folder is never
    opened (watched), and mark 3 could not run, naming it."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    run_folder = drive_fake.read_log(only)[2]["run"]
    sealed = root / "data" / "heldout" / fake.ASR_FILE[0]
    linked_away(root / run_folder / name, sealed / name)
    s = check.survey(root, check.Findings(lambda line: None))
    found = check._mark([], False)
    with watch(root / "data" / "heldout") as seen:
        check.check_hashes(root, s, [], found)
    assert seen == []
    assert found.status() == check.COULD_NOT_RUN, found.lines
    assert found.lines == [
        f"COULD NOT RUN: {only.name}, step 1 (sealed-truth): {run_folder}/{name} is a symbolic "
        "link, not read"
    ]


# --- no file read through a link, at the file or at a folder above it ----------------------------


@contextmanager
def opens_of(monkeypatch, *folders):
    """The (event, name) of every open, listing or scan of a file or folder at or under `folders`
    while active: by path through the audit hook (`watch`), and by descriptor through `os.open`,
    `os.listdir` and `os.scandir`, whose identity is read from the descriptor (`os.fstat`): a
    name opened from a folder's descriptor (`dir_fd`) has an audit event that names no folder."""
    watched = sealed_watch.identities(*folders)
    real_open, real_listdir, real_scandir = os.open, os.listdir, os.scandir

    def seen_at(descriptor: int, event: str, name) -> None:
        mark = os.fstat(descriptor)
        if (mark.st_dev, mark.st_ino) in watched:
            by_descriptor.append((event, os.fsdecode(name)))

    def recording_open(name, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(name, flags, mode, dir_fd=dir_fd)
        seen_at(descriptor, "os.open", name)
        return descriptor

    def recording_listdir(path="."):
        if isinstance(path, int):
            seen_at(path, "os.listdir", str(path))
        return real_listdir(path)

    def recording_scandir(path="."):
        if isinstance(path, int):
            seen_at(path, "os.scandir", str(path))
        return real_scandir(path)

    by_descriptor: list = []
    with monkeypatch.context() as patch, watch(*folders) as by_path:
        patch.setattr(os, "open", recording_open)
        patch.setattr(os, "listdir", recording_listdir)
        patch.setattr(os, "scandir", recording_scandir)
        found: list = []
        yield found
        found += by_path + by_descriptor


def made_up_sealed(root: Path) -> Path:
    """A made-up sealed file (the a.s.r. set's name) in the repository's sealed folder; its
    folder."""
    sealed = root / "data" / "heldout" / fake.ASR_FILE[0]
    sealed.mkdir(parents=True, exist_ok=True)
    (sealed / fake.ASR_FILE[1]).write_bytes(b"made-up sealed bytes\n")
    return sealed


def moved_and_linked(path: Path, target: Path) -> None:
    """The file or folder at `path` moved to `target` and replaced by a symbolic link to it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)
    path.symlink_to(target)


def test_links_1_a_manifest_linked_into_the_sealed_folder_is_never_opened_by_any_mark(
    tmp_path, monkeypatch, capsys
):
    """A complete drive at the tag, the first step's manifest then a symbolic link to a file of
    its bytes in the sealed folder: the whole check never opens it (watched by path and by
    descriptor); mark 1 (which verifies the run) and mark 3 could not run, each naming it; exit
    2."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    run_folder = drive_fake.read_log(only)[2]["run"]
    sealed = made_up_sealed(root)
    linked_away(root / run_folder / "manifest.json", sealed / "manifest.json")
    with opens_of(monkeypatch, root / "data" / "heldout") as seen:
        status, marks, results = run(root, capsys)
    assert seen == []
    assert status == 2, marks
    assert statuses(marks) == ["could not run", "passed", "could not run"], marks
    named = f"{run_folder}/manifest.json is a symbolic link"
    assert any(line.startswith("COULD NOT RUN: ") and named in line for line in marks[MARKS[0]][1])
    assert (
        f"COULD NOT RUN: {only.name}, step 1 (sealed-truth): {named}, not read"
        in (marks[MARKS[2]][1])
    ), marks
    assert results["inputs"][f"{run_folder}/manifest.json"] == "cannot be hashed"


@pytest.mark.parametrize("which", ["openings", "refusal"])
def test_links_2_3_an_openings_or_refusal_log_linked_into_the_sealed_folder_is_never_opened(
    tmp_path, monkeypatch, capsys, which
):
    """A complete drive at the tag (and one made-up layout refusal logged with no opening), the
    openings log (test 2) or the refusal log (test 3) then a symbolic link to a file of its
    bytes in the sealed folder: never opened; marks 1 and 2 could not run, each naming the log
    as a link first; mark 3 could not hash it; exit 2."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    fake.log_refusal(root, fake.ASR_FILE, b"made-up sealed bytes\n", "0" * 40, opened=False)
    sealed = made_up_sealed(root)
    path = openings_file(root) if which == "openings" else refusals_file(root)
    linked_away(path, sealed / path.name)
    with opens_of(monkeypatch, root / "data" / "heldout") as seen:
        status, marks, results = run(root, capsys)
    assert seen == []
    assert status == 2, marks
    assert statuses(marks) == ["could not run"] * 3, marks
    named = f"data/{path.name} is a symbolic link"
    for name in MARKS[:2]:
        first = marks[name][1][0]
        assert first.startswith(f"COULD NOT RUN: the {which} log cannot be read: "), marks
        assert named in first, first
    assert results["inputs"][f"data/{path.name}"] == "cannot be hashed"


def test_links_4_a_run_folder_linked_into_the_sealed_folder_is_never_entered(
    tmp_path, monkeypatch, capsys
):
    """A complete drive at the tag, the first step's run folder then moved into the sealed
    folder and replaced by a symbolic link to it (its manifest and log.txt no links): nothing
    there is opened; mark 1 and mark 3 could not run, naming the folder as the link the
    manifest lies under; exit 2."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    run_folder = drive_fake.read_log(only)[2]["run"]
    sealed = made_up_sealed(root)
    moved_and_linked(root / run_folder, sealed / Path(run_folder).name)
    with opens_of(monkeypatch, root / "data" / "heldout") as seen:
        status, marks, results = run(root, capsys)
    assert seen == []
    assert status == 2, marks
    assert statuses(marks) == ["could not run", "passed", "could not run"], marks
    named = f"{run_folder}/manifest.json lies under the symbolic link {run_folder}"
    assert any(line.startswith("COULD NOT RUN: ") and named in line for line in marks[MARKS[0]][1])
    assert (
        f"COULD NOT RUN: {only.name}, step 1 (sealed-truth): {named}, not read"
        in (marks[MARKS[2]][1])
    ), marks
    for name in ("manifest.json", "log.txt"):
        assert results["inputs"][f"{run_folder}/{name}"] == "cannot be hashed"


def test_links_an_output_the_manifest_lists_linked_into_the_sealed_folder_is_never_opened(
    tmp_path, monkeypatch, capsys
):
    """A complete drive at the tag, the first step's `log.txt` (an output its manifest lists,
    which mark 1 verifies through `require_verified`) then a symbolic link to a file of its bytes
    in the sealed folder: no mark opens it; marks 1 and 3 could not run, each naming it; exit
    2."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    run_folder = drive_fake.read_log(only)[2]["run"]
    sealed = made_up_sealed(root)
    linked_away(root / run_folder / "log.txt", sealed / "log.txt")
    with opens_of(monkeypatch, root / "data" / "heldout") as seen:
        status, marks, _ = run(root, capsys)
    assert seen == []
    assert status == 2, marks
    assert statuses(marks) == ["could not run", "passed", "could not run"], marks
    named = f"{run_folder}/log.txt is a symbolic link"
    assert any(line.startswith("COULD NOT RUN: ") and named in line for line in marks[MARKS[0]][1])
    assert (
        f"COULD NOT RUN: {only.name}, step 1 (sealed-truth): {named}, not read"
        in (marks[MARKS[2]][1])
    ), marks


@pytest.mark.parametrize("folder", ["data", "runs.noindex/drives"])
def test_links_5_a_log_folder_that_is_a_link_is_never_entered(
    tmp_path, monkeypatch, capsys, folder
):
    """A complete drive at the tag, then the folder of the openings and refusal logs (`data`,
    the sealed folder's parent) or the folder of the drive logs (`runs.noindex/drives`) moved
    away and replaced by a symbolic link to it: nothing in the moved folder is opened or listed;
    every mark could not run, naming the folder; exit 2."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    fake.log_refusal(root, fake.ASR_FILE, b"made-up sealed bytes\n", "0" * 40, opened=False)
    made_up_sealed(root)
    moved = tmp_path / "moved" / Path(folder).name
    moved_and_linked(root / folder, moved)
    with opens_of(monkeypatch, moved) as seen:
        status, marks, _ = run(root, capsys)
    assert seen == []
    assert status == 2, marks
    assert statuses(marks) == ["could not run"] * 3, marks
    for name in MARKS:
        lines = marks[name][1]
        assert any(folder in line and "symbolic link" in line for line in lines), (name, lines)


def test_links_6_at_a_logged_fix_a_linked_data_folder_is_refused_by_the_gate_reading_no_log(
    tmp_path, monkeypatch, capsys
):
    """At a logged parsing fix with a complete drive (every mark passes), then `data` moved away
    and replaced by a symbolic link to it: the gate, whose check 5 reads both logs at a fix,
    refuses first, naming the link; nothing in the moved folder is opened or listed (before the
    look, both logs were opened through the link); every mark could not run; exit 2."""
    root, _, _ = at_a_fix(tmp_path, monkeypatch)
    status, marks, _ = run(root, capsys)
    assert status == 0, marks
    moved = tmp_path / "moved" / "data"
    moved_and_linked(root / "data", moved)
    with opens_of(monkeypatch, moved) as seen:
        status, marks, _ = run(root, capsys)
    assert seen == []
    assert status == 2, marks
    assert statuses(marks) == ["could not run"] * 3, marks
    gate = (
        "COULD NOT RUN: the gate of the run from the tag refuses the checked-out commit: "
        f"{root / 'data'} is a symbolic link: the gate never follows a link at the data folder, "
        "the sealed folder or a log of it; nothing opened"
    )
    for name in MARKS:
        assert marks[name][1][0] == gate, (name, marks[name][1])


def test_links_6_a_plain_output_name_under_a_linked_subfolder_is_never_opened(
    tmp_path, monkeypatch, capsys
):
    """A complete drive at the tag; then a plain output name `sub/<file>` added to the first
    step's manifest (its hash and the log's record of it made to match), where `sub` in the run
    folder is a symbolic link to the sealed folder, which holds that file and a copy of the run's
    `manifest.json` and `log.txt`: nothing there is opened (the run folder of `sub/<file>`, as
    `run_verify` finds it, is read through the check's reader too); exit 2, no pass."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    run_folder = drive_fake.read_log(only)[2]["run"]
    sealed = made_up_sealed(root)
    for name in ("manifest.json", "log.txt"):
        (sealed / name).write_bytes((root / run_folder / name).read_bytes())
    sealed_file = sealed / fake.ASR_FILE[1]
    (root / run_folder / "sub").symlink_to(sealed)
    manifest_file = root / run_folder / "manifest.json"
    manifest = json.loads(manifest_file.read_bytes())
    manifest["outputs"][f"sub/{fake.ASR_FILE[1]}"] = hashlib.sha256(
        sealed_file.read_bytes()
    ).hexdigest()
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

    def change(records):
        [e for e in records if e.get("record") == "entry"][0]["manifest_sha256"] = digest

    rewritten(only, change)
    with opens_of(monkeypatch, root / "data" / "heldout") as seen:
        status, marks, _ = run(root, capsys)
    assert seen == [], seen
    assert status == 2, marks
    assert statuses(marks)[0] == "could not run", marks


def test_16_at_a_logged_fix_the_gates_mark_in_the_pass_mark_run_names_the_fix(
    tmp_path, monkeypatch
):
    """Under the real recorder at a logged parsing fix: the check passes and the gate's mark in
    the pass-mark run's folder (`sealed_route.require_gate` calling `gate_mark.mark_run`) names
    the tag's commit and the fix commit; the manifest lists the mark."""
    root, tagged, fix = at_a_fix(tmp_path, monkeypatch)
    environment = {
        key: value
        for key, value in fake.tool_environment().items()
        if key != check.RUN_DIR_VARIABLE
    }
    done = subprocess.run(
        [
            sys.executable, "-m", "fleetreserve.run_record", "--name", "wp14-passmark", "--",
            sys.executable, str(TOOL), "--root", str(root),
        ],
        cwd=root, capture_output=True, text=True, check=False, env=environment, timeout=300,
    )  # fmt: skip
    assert done.returncode == 0, (done.stdout, done.stderr)
    (folder,) = [p for p in drive_fake.run_folders(root) if p.name.endswith("-wp14-passmark")]
    mark = json.loads((folder / gate_mark.MARK_NAME).read_text())
    assert (mark["gate"], mark["tag_commit"], mark.get("fix_commit")) == ("passed", tagged, fix)
    assert gate_mark.MARK_NAME in json.loads((folder / "manifest.json").read_text())["outputs"]
    results = json.loads((folder / check.RESULTS_NAME).read_text())
    assert (results["status"], results["tag_commit"], results["fix_commit"]) == (
        "passed",
        tagged,
        fix,
    )


# --- tests of the earlier check restored, on the one command ------------------------------------


def test_hashes_fails_a_step_whose_manifest_is_gone(tmp_path, monkeypatch, capsys):
    """A complete drive at the tag, the manifests of both its steps then deleted: mark 3 fails
    each step as a manifest whose SHA-256 is not the one the log records, never could not run,
    and says no pass line."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    entries = [line for line in drive_fake.read_log(only) if line.get("record") == "entry"]
    for entry in entries:
        (root / entry["run"] / "manifest.json").unlink()
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert marks["logs hashed"] == (
        "failed",
        [
            f"FAILED: {only.name}, step {k} ({entry['label']}): the SHA-256 of "
            f"{entry['run']}/manifest.json is not the one the log records"
            for k, entry in enumerate(entries, 1)
        ],
    ), marks


@pytest.mark.parametrize(
    "read",
    [(None, None), ([], None), (None, [])],
    ids=["neither log", "no refusals", "no openings"],
)
def test_fixes_with_the_gate_refusing_judges_no_drive_against_a_tag(tmp_path, read):
    """Mark 2 when the gate refuses the checked-out commit (a made-up survey with no gate) and
    the openings log, the refusal log or both could not be read: an earlier drive is judged
    neither against a tag there is none of nor against the logs; the mark says only the
    survey's line and could not run."""
    said = "COULD NOT RUN: the gate of the run from the tag refuses the checked-out commit: made up"
    s = check.Survey("c" * 40, None, [two_steps("10:00:06", "10:00:09")], 0, None, None, [])
    found = check._mark([said], True)
    check.check_fixes(tmp_path, s, *read, found)
    assert (found.status(), found.lines) == (check.COULD_NOT_RUN, [said])


def test_fixes_with_no_complete_drive_judges_every_drive_as_earlier(tmp_path, monkeypatch, capsys):
    """One drive stopped at a layout refusal at the tag and no complete drive: mark 2 judges it
    as an earlier drive (a layout stop) and passes; mark 1 fails for the missing complete
    drive."""
    root, _ = made(tmp_path, monkeypatch)
    sealed_drive(1)
    (only,) = logs(root)
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert statuses(marks) == ["failed", "passed", "passed"], marks
    assert any(
        line.startswith(f"{only.name}: stopped at step 1 (sealed-truth) at a layout refusal")
        for line in marks["logged parsing fixes"][1]
    ), marks


def test_an_earlier_drive_with_no_opening_in_a_step_is_an_operational_stop_whatever_else():
    """Openings outside every step (another commit, after the drive), and none at all, leave an
    operational stop; with no stop line, one or two openings in steps fail, counted in words."""
    drive = two_steps("10:00:06", "10:00:09")
    outside = [opening("10:00:07.000000", commit="d" * 40), opening("11:00:00.000000")]
    for openings in ([], outside):
        kind, text = check._earlier(drive, [], openings)
        assert kind == "operational" and text.endswith(": an operational stop"), text
    drive.stop = None
    kind, text = check._earlier(drive, [], [opening("10:00:07.000000"), *outside])
    assert kind == "fail" and "with 1 opening of sealed data in its steps; an earlier" in text
    kind, text = check._earlier(drive, [], [opening("10:00:07.000000"), opening("10:00:08.0")])
    assert kind == "fail" and "with 2 openings of sealed data in its steps; an earlier" in text


def test_a_folder_that_is_no_git_repository_could_not_run(tmp_path, capsys):
    root = tmp_path / "repo"  # the recorder's folder is made in it; no repository is
    status, marks, results = run(root, capsys)
    assert status == 2, marks
    assert results["commit"] is None
    for name in MARKS:
        assert marks[name][0] == "could not run", marks
        assert marks[name][1][0].startswith("COULD NOT RUN: git rev-parse HEAD failed: "), marks


def test_the_inputs_name_an_absent_refusal_log_and_hash_it_once_written(
    tmp_path, monkeypatch, capsys
):
    root = complete_at_the_tag(tmp_path, monkeypatch)
    status, marks, results = run(root, capsys)
    assert status == 0, marks
    opened = hashlib.sha256(openings_file(root).read_bytes()).hexdigest()
    assert results["inputs"]["data/heldout-openings.log"] == opened
    assert results["inputs"]["data/heldout-refusals.log"] == "absent"
    fake.log_refusal(root, fake.ASR_FILE, b"made-up sealed bytes\n", "0" * 40)
    status, marks, results = run(root, capsys)
    assert marks["logs hashed"][0] == "passed", marks
    refused = hashlib.sha256(refusals_file(root).read_bytes()).hexdigest()
    assert results["inputs"]["data/heldout-refusals.log"] == refused


# --- 17. O2's verdict (the WP12 check's reading R7) ----------------------------------------------


def o2(*arguments: str) -> list[dict]:
    """Outcome O2's two stand-in steps (`drive_fake.O2_STEP`) of both purposes after the truth
    builds they read (`drive_fake.o2_entries`, run list rules 10 and 12): `sealed-o2` of the main
    setting with `arguments`, then `sealed-o2-held`, passing on its one unit."""
    return drive_fake.o2_entries("sealed-o2", *arguments)


def o2_complete(tmp_path, monkeypatch, *arguments: str) -> tuple[Path, str]:
    """A complete drive at the tag of the truth builds and the two O2 steps, the sealed truth
    build of each setting the made-up held-out truth build (`TRUTH_STEP`, one opening each);
    (top folder, the run folder of `sealed-o2`, the fifth step)."""
    opener, units = drive_fake.SEALED_TRUTH, drive_fake.O2_UNITS
    root, _ = made(tmp_path, monkeypatch, o2(*arguments), opener=opener, units=units)
    sealed_drive(0)
    (only,) = logs(root)
    return root, drive_fake.read_log(only)[10]["run"]


@pytest.mark.parametrize(
    ("verdicts", "over", "failed"),
    [("unit-a=failed,unit-b=passed", "failed", "unit-a"), ("unit-a=passed", "passed", "none")],
    ids=["failed-on-one-unit", "passed"],
)
def test_17_each_o2_verdict_of_the_complete_drive_is_a_line_and_no_mark_changes(
    tmp_path, monkeypatch, capsys, verdicts, over, failed
):
    root, run_folder = o2_complete(tmp_path, monkeypatch, "--verdicts", verdicts)
    status, marks, _ = run(root, capsys)
    assert status == 0, marks
    assert statuses(marks) == ["passed"] * 3, marks
    lines = marks["one logged run"][1]
    assert (
        f"O2 sealed-o2: over the units {over}; failed on {failed}; could not decide on none "
        f"(from {run_folder}/ceiling-report.json; not in this check's exit status, reading R7)"
    ) in lines
    assert lines[-1].startswith("one logged run: "), lines


@pytest.mark.parametrize(
    ("verdicts", "over", "no_day"),
    [
        ("unit-a=no day to check,unit-b=passed,unit-c=passed", "passed", "unit-a"),
        ("unit-a=no day to check,unit-b=no day to check", "no day to check", "unit-a, unit-b"),
    ],
    ids=["beside-two-passed", "every-unit"],
)
def test_17_an_o2_line_names_the_units_with_no_day_to_check_and_no_mark_changes(
    tmp_path, monkeypatch, capsys, verdicts, over, no_day
):
    """The owner's decision of 19 November 2026: a unit with no day to check is named on mark 1's
    O2 line and counts as neither passed nor failed; with no unit judged the verdict over the
    units is no day to check (decision 79 (3)); no mark changes."""
    root, run_folder = o2_complete(tmp_path, monkeypatch, "--verdicts", verdicts)
    status, marks, _ = run(root, capsys)
    assert status == 0, marks
    assert statuses(marks) == ["passed"] * 3, marks
    assert (
        f"O2 sealed-o2: over the units {over}; failed on none; could not decide on none; no day "
        f"to check on {no_day} (from {run_folder}/ceiling-report.json; not in this check's exit "
        "status, reading R7)"
    ) in marks["one logged run"][1]


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
def test_17_an_o2_step_shows_its_own_faults_beside_its_verdict_and_no_mark_changes(
    tmp_path, monkeypatch, capsys, arguments, faulted
):
    """Decision 88 (3d): the faults of each unit (decision 85 (1)) are shown beside the verdict
    on mark 1's lines; no mark and no exit status changes."""
    root, _ = o2_complete(tmp_path, monkeypatch, *arguments)
    status, marks, _ = run(root, capsys)
    assert status == 0, marks
    assert statuses(marks) == ["passed"] * 3, marks
    lines = marks["one logged run"][1]
    (at,) = [k for k, line in enumerate(lines) if line.startswith("O2 sealed-o2: over the units")]
    assert lines[at + 1] == f"O2 sealed-o2: {FAULTS_HEAD}{faulted}", lines
    assert lines[-1].startswith("one logged run: "), lines


def test_17_an_o2_step_with_no_faults_shows_no_faults_line(tmp_path, monkeypatch, capsys):
    root, _ = o2_complete(tmp_path, monkeypatch, "--verdicts", "unit-a=failed,unit-b=passed")
    status, marks, _ = run(root, capsys)
    assert status == 0, marks
    assert statuses(marks) == ["passed"] * 3, marks
    assert not [line for line in marks["one logged run"][1] if "own faults" in line], marks


@pytest.mark.parametrize(
    "arguments",
    [
        ("--no-faults",),
        ("--faults", "unit-a=bad"),
        ("--faults", "unit-a=noblock"),
        ("--faults", "unit-a=nomdays"),
    ],
    ids=["no-record", "not-counts", "no-block-count", "no-days-in-m-count"],
)
def test_17_an_o2_step_whose_faults_record_is_not_readable_says_so_and_no_mark_changes(
    tmp_path, monkeypatch, capsys, arguments
):
    root, _ = o2_complete(tmp_path, monkeypatch, *arguments)
    status, marks, _ = run(root, capsys)
    assert status == 0, marks
    assert statuses(marks) == ["passed"] * 3, marks
    assert (
        f"O2 sealed-o2: {FAULTS_HEAD}the faults record is missing or not readable for unit-a"
    ) in marks["one logged run"][1]


def test_17_an_o2_step_with_no_readable_verdict_fails_mark_1_named(tmp_path, monkeypatch, capsys):
    root, run_folder = o2_complete(tmp_path, monkeypatch, "--no-verdict")
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert statuses(marks) == ["failed", "passed", "passed"], marks
    assert (
        f"FAILED: O2 sealed-o2: exited 0, but {run_folder}/ceiling-report.json holds no readable "
        "verdict: it holds no every_day with its units' O2 passes"
    ) in marks["one logged run"][1]


def test_17_the_o2_reader_opens_nothing_sealed_and_its_file_only_inside_run_verify(
    tmp_path, monkeypatch, capsys
):
    root, run_folder = o2_complete(tmp_path, monkeypatch, "--verdicts", "unit-a=failed")
    sealed = root / "data" / "heldout" / fake.ASR_FILE[0] / fake.ASR_FILE[1]
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(b"made-up sealed bytes\n")
    report = root / run_folder / "ceiling-report.json"
    with (
        watch(root / "data" / "heldout") as sealed_seen,
        opened_from.opened([report], run_verify) as seen,
    ):
        status, marks, _ = run(root, capsys)
    assert status == 0, marks
    assert any(
        line.startswith("O2 sealed-o2: over the units failed; failed on unit-a;")
        for line in marks["one logged run"][1]
    ), marks
    assert sealed_seen == []
    assert seen, "the results file was never opened"
    assert all(inside for _, inside in seen), seen


# A path the check reads is taken as written: a "..", a "." or an absolute start in it is never
# normalised away (`os.path.abspath` would turn `runs.noindex/../data/heldout/asr` into
# `data/heldout/asr`, under --root), and the mark that needs it could not run, naming it.


def planted(root: Path, run_folder: str, target: Path) -> None:
    """Copies of a step's manifest and log.txt planted in the folder `target`."""
    target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "log.txt"):
        (target / name).write_bytes((root / run_folder / name).read_bytes())


def run_renamed(path: Path, k: int, new: str) -> None:
    """The drive log at `path` with its `k`-th entry's run folder written as `new` by hand."""

    def change(records):
        [entry for entry in records if entry.get("record") == "entry"][k]["run"] = new

    rewritten(path, change)


SEALED_BY_DOTS = "runs.noindex/../data/heldout/asr"


@pytest.mark.parametrize("plant", [True, False], ids=["copies planted", "the sealed layout only"])
def test_dots_an_earlier_drives_run_through_dotdot_into_the_sealed_folder_could_not_run(
    tmp_path, monkeypatch, capsys, plant
):
    """An earlier drive (an operational stop) whose step's run folder is written by hand as
    `runs.noindex/../data/heldout/asr`, its log's hash recomputed; with copies of that step's
    manifest and log.txt planted in the sealed folder (the review's r3) or not (r3b): nothing
    under `data/heldout` is opened or listed, and mark 3 could not run, naming the path (exit
    2)."""
    root = timed_out_then_complete(tmp_path, monkeypatch, "before")
    first, _ = logs(root)
    sealed = made_up_sealed(root)
    if plant:
        planted(root, drive_fake.read_log(first)[2]["run"], sealed)
    run_renamed(first, 0, SEALED_BY_DOTS)
    with opens_of(monkeypatch, root / "data" / "heldout") as seen:
        status, marks, results = run(root, capsys)
    assert seen == []
    assert status == 2, marks
    assert statuses(marks) == ["passed", "passed", "could not run"], marks
    assert any(
        line.startswith("COULD NOT RUN: ") and repr(SEALED_BY_DOTS) in line
        for line in marks[MARKS[2]][1]
    ), marks
    assert all(not name.startswith(SEALED_BY_DOTS + "/") for name in results["inputs"])
    assert results["inputs"][SEALED_BY_DOTS] == "cannot be hashed"


@pytest.mark.parametrize(
    "written",
    ["runs.noindex/./{name}", "runs.noindex/drives/../{name}", "runs.noindex//{name}"],
    ids=["a '.'", "a '..' back to the folder", "an empty part"],
)
def test_dots_an_earlier_drives_run_named_with_dots_of_its_own_folder_could_not_run(
    tmp_path, monkeypatch, capsys, written
):
    """An earlier drive's run folder written by hand with a ".", a ".." or an empty part that
    still leads to its own folder: the manifest there is the logged one, yet the name is not
    `runs.noindex/<UTC start>-<label>`, so mark 3 could not run, naming it (exit 2)."""
    root = timed_out_then_complete(tmp_path, monkeypatch, "before")
    first, _ = logs(root)
    by_hand = written.format(name=Path(drive_fake.read_log(first)[2]["run"]).name)
    run_renamed(first, 0, by_hand)
    status, marks, results = run(root, capsys)
    assert status == 2, marks
    assert statuses(marks) == ["passed", "passed", "could not run"], marks
    assert any(repr(by_hand) in line for line in marks[MARKS[2]][1]), marks
    assert results["inputs"][by_hand] == "cannot be hashed"


def test_dots_an_earlier_drives_run_copied_to_a_folder_not_named_by_its_label_could_not_run(
    tmp_path, monkeypatch, capsys
):
    """An earlier drive's run folder copied to `runs.noindex/copy` and the log pointed there:
    every byte the log records is there, yet the folder is not `runs.noindex/<UTC
    start>-<label>`, so mark 3 could not run, naming it (exit 2)."""
    root = timed_out_then_complete(tmp_path, monkeypatch, "before")
    first, _ = logs(root)
    planted(root, drive_fake.read_log(first)[2]["run"], root / "runs.noindex" / "copy")
    run_renamed(first, 0, "runs.noindex/copy")
    status, marks, _ = run(root, capsys)
    assert status == 2, marks
    assert statuses(marks) == ["passed", "passed", "could not run"], marks
    assert any("'runs.noindex/copy'" in line for line in marks[MARKS[2]][1]), marks


@pytest.mark.parametrize("kind", ["..", ".", "an absolute start"])
def test_dots_the_one_reader_refuses_a_path_not_plain_and_opens_nothing(
    tmp_path, monkeypatch, kind
):
    """`_read`, `_listing` and `_digest` take the path's parts as written after `--root` and "/":
    a "..", a "." or an absolute start raises `NotPlain`, naming the path, and nothing is
    opened, though each path leads to a readable file under `--root`."""
    root = tmp_path / "repo"
    sealed = made_up_sealed(root)
    (root / "runs.noindex").mkdir()
    made = root / "runs.noindex" / "made.txt"
    made.write_bytes(b"made up\n")
    text = {
        "..": f"{root}/runs.noindex/../data/heldout/{sealed.name}/{fake.ASR_FILE[1]}",
        ".": f"{root}/runs.noindex/./made.txt",
        "an absolute start": f"{root}/{made}",
    }[kind]
    with opens_of(monkeypatch, root / "data", root / "runs.noindex") as seen:
        for reader, path in (
            (check._read, text),
            (check._digest, text),
            (check._listing, os.path.dirname(text)),
        ):
            with pytest.raises(OSError, match="not a plain path") as refused:
                reader(root, path)
            assert type(refused.value).__name__ == "NotPlain" and path in str(refused.value)
    assert seen == []


@pytest.mark.parametrize(
    "listed",
    ["../{name}/log.txt", "./log.txt", "{root}/{run}/log.txt"],
    ids=["a '..'", "a '.'", "an absolute start"],
)
def test_dots_an_output_name_not_plain_in_a_manifest_could_not_run(
    tmp_path, monkeypatch, capsys, listed
):
    """A complete drive whose first step's manifest is edited by hand to list, beside `log.txt`,
    a name with a "..", a "." or an absolute start that leads back to that same file, with its
    SHA-256 (and the log's record of the manifest's hash recomputed): mark 1 could not run,
    naming the output (`run_verify.plain_name`), instead of verifying the file under a name the
    recorder never writes (exit 2)."""
    root = complete_at_the_tag(tmp_path, monkeypatch)
    (only,) = logs(root)
    run_folder = drive_fake.read_log(only)[2]["run"]
    manifest_file = root / run_folder / "manifest.json"
    manifest = json.loads(manifest_file.read_bytes())
    name = listed.format(name=Path(run_folder).name, root=root, run=run_folder)
    manifest["outputs"][name] = manifest["outputs"]["log.txt"]
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

    def change(records):
        first = [entry for entry in records if entry.get("record") == "entry"][0]
        first["manifest_sha256"] = digest

    rewritten(only, change)
    status, marks, _ = run(root, capsys)
    assert status == 2, marks
    assert statuses(marks) == ["could not run", "passed", "passed"], marks
    assert any(
        line.startswith("COULD NOT RUN: ") and f"lists the output {name!r}" in line
        for line in marks[MARKS[0]][1]
    ), marks


# --- 18. a sealed drive that goes on after a failing entry (the driver's `failed` and `not_run`) --

# A stand-in step that fails while the ignored file data/fail-now exists, so one run list gives a
# drive whose step fails and, after the file is removed, a drive whose step passes.
FLAKY_STEP = drive_fake.STEP.replace(
    "sys.exit(int(value",
    "if os.path.exists(os.path.join('data', 'fail-now')):\n    sys.exit(1)\nsys.exit(int(value",
)


def rehearsal_entry() -> dict:
    """One step that runs in the rehearsal only, as a run list needs one."""
    return drive_fake.entry("rehearsal-learn", drive_fake.step("compare_run"), only="rehearsal")


def earlier_drive_shape(monkeypatch) -> None:
    """Make the driver write a drive as it did before a failure ahead of the first sealed truth
    build stopped it: with no such place, a failed step goes on at any place. The check still
    reads such a log (one written by an earlier driver); the lists of these tests put a failing
    step before the sealed truth build, which the driver now stops at."""
    monkeypatch.setattr(run_driver, "_first_opening", lambda steps: None)


def failing_then_dependent() -> list[dict]:
    """The sealed truth build, a step that exits 1, one that needs its run, one that does not."""
    needing = drive_fake.entry(
        "sealed-needs",
        drive_fake.step("compare_run", "--truth-run", "{run:sealed-bad}"),
        after=["sealed-bad"],
        only="sealed-run",
    )
    bad = drive_fake.entry(
        "sealed-bad", drive_fake.step("learn_run", "--exit", "1"), only="sealed-run"
    )
    free = drive_fake.entry("sealed-free", drive_fake.step("learn_run"), only="sealed-run")
    return [truth(), bad, needing, free, rehearsal_entry()]


def test_18_a_drive_that_ran_to_its_end_with_a_failed_and_a_not_run_entry_fails_mark_1_named(
    tmp_path, monkeypatch, capsys
):
    root, _ = made(tmp_path, monkeypatch, failing_then_dependent())
    sealed_drive(1)
    (only,) = logs(root)
    kinds = [record["record"] for record in drive_fake.read_log(only)]
    assert kinds == [
        "start", "begin", "entry",  # sealed-truth
        "begin", "entry", "failed",  # sealed-bad
        "not_run",  # sealed-needs
        "begin", "entry",  # sealed-free
        "end",
    ], kinds  # fmt: skip
    status, marks, results = run(root, capsys)
    assert status == 1, marks
    assert marks["one logged run"][0] == "failed", marks
    assert "could not run" not in statuses(marks) and results["status"] == "failed", marks
    expected = (
        f"FAILED: no complete sealed-run drive; the newest, {only.name}, ended with exit status 1; "
        "failed step 2 (sealed-bad): the run ended with exit status 1; not run step 3 "
        "(sealed-needs): not run: needs sealed-bad, which failed"
    )
    assert expected in marks["one logged run"][1], marks["one logged run"][1]


def test_18_the_end_lines_lists_name_a_failure_the_lines_do_not_show(tmp_path, monkeypatch, capsys):
    """A drive whose `failed` line is missing but whose end line lists the failure still fails
    mark 1 naming it: the end line's lists count with the log's lines."""
    root, _ = made(tmp_path, monkeypatch, failing_then_dependent())
    sealed_drive(1)
    (only,) = logs(root)
    rewritten(only, lambda records: records.pop(5))  # the `failed` line of sealed-bad
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    line = next(line for line in marks["one logged run"][1] if line.startswith("FAILED: no"))
    assert "failed step 2 (sealed-bad)" in line and "not run step 3 (sealed-needs)" in line, line


def test_18_a_failed_drive_never_passes_even_when_its_end_says_nothing_failed(
    tmp_path, monkeypatch, capsys
):
    """The lines say an entry failed and the end line, rewritten, says the drive completed with
    empty lists: a `failed` or `not_run` line alone keeps the drive from being complete."""
    root, _ = made(tmp_path, monkeypatch, failing_then_dependent())
    sealed_drive(1)
    (only,) = logs(root)

    def said_complete(records):
        records[-1] |= {"completed": True, "exit_status": 0, "failed": [], "not_run": []}

    rewritten(only, said_complete)
    status, marks, _ = run(root, capsys)
    assert status == 1 and marks["one logged run"][0] == "failed", marks


def test_18_an_earlier_drive_that_failed_a_step_and_not_run_one_with_no_opening_is_operational(
    tmp_path, monkeypatch, capsys
):
    """Drive 1: the development truth build fails, the sealed truth build that follows it is not
    run (it names the failed step in its `after`), no sealed file is opened; drive 2 passes. Mark 2
    counts drive 1 as an operational stop."""
    flaky = drive_fake.entry("dev-truth", drive_fake.step("truth_build"))
    after_it = {**truth(), "after": ["dev-truth"]}
    entries = [flaky, after_it, learn()]
    root, _ = made(tmp_path, monkeypatch, entries, modules={"truth_build": FLAKY_STEP})
    earlier_drive_shape(monkeypatch)
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "fail-now").write_text("")
    sealed_drive(1)
    (first,) = logs(root)
    kinds = [record["record"] for record in drive_fake.read_log(first)]
    assert kinds == ["start", "begin", "entry", "failed", "not_run", "begin", "entry", "end"], kinds
    (root / "data" / "fail-now").unlink()
    sealed_drive(0)
    status, marks, _ = run(root, capsys)
    assert status == 0, marks
    assert statuses(marks) == ["passed"] * 3, marks
    assert any(
        line.startswith(f"{first.name}: ended with exit status 1; failed step 1 (dev-truth)")
        and line.endswith("with no opening of sealed data in any of its steps: an operational stop")
        for line in marks["logged parsing fixes"][1]
    ), marks["logged parsing fixes"]


def test_18_an_earlier_drive_that_opened_sealed_files_and_failed_a_later_step_fails_mark_2(
    tmp_path, monkeypatch, capsys
):
    """Plan line 149: an earlier drive that opened the sealed files counts only as stopped by a
    layout refusal. Here the sealed truth build passed and a later step failed (the drive went
    on), so mark 2 fails with a complete drive after it."""
    flaky = drive_fake.entry("sealed-flaky", drive_fake.step("compare_run"), only="sealed-run")
    root, _ = made(
        tmp_path,
        monkeypatch,
        [truth(), flaky, rehearsal_entry()],
        modules={"compare_run": FLAKY_STEP},
    )
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "fail-now").write_text("")
    sealed_drive(1)
    (root / "data" / "fail-now").unlink()
    sealed_drive(0)
    first, _ = logs(root)
    status, marks, _ = run(root, capsys)
    assert status == 1, marks
    assert statuses(marks) == ["passed", "failed", "passed"], marks
    (line,) = [line for line in marks["logged parsing fixes"][1] if line.startswith("FAILED: ")]
    assert line.startswith(f"FAILED: {first.name} ended with exit status 1; failed step 2")
    assert "with 1 opening of sealed data in its steps" in line, line
    assert "counts only as stopped by a layout refusal" in line, line


def test_18_an_earlier_drive_stopped_at_a_layout_refusal_has_the_new_end_and_is_judged_as_before(
    tmp_path, monkeypatch, capsys
):
    """The shape the driver now writes for a refused held-out truth build: a stop line, the failed
    build and the steps not reached in the end line only."""
    root, tagged, fix = at_a_fix(tmp_path, monkeypatch)
    first, second = logs(root)
    records = drive_fake.read_log(first)
    assert [record["record"] for record in records] == ["start", "begin", "entry", "stop", "end"]
    end = records[-1]
    assert [(f["index"], f["label"]) for f in end["failed"]] == [(0, "sealed-truth")]
    assert [(n["index"], n["label"]) for n in end["not_run"]] == [(1, "sealed-learn")]
    status, marks, _ = run(root, capsys)
    assert status == 0 and statuses(marks) == ["passed"] * 3, marks
    assert any(
        line.startswith(
            f"{first.name}: stopped at step 1 (sealed-truth) at a layout refusal of the"
        )
        for line in marks["logged parsing fixes"][1]
    ), marks["logged parsing fixes"]


def test_18_a_stop_after_a_failed_and_a_not_run_step_is_read_and_judged_by_the_stop(
    tmp_path, monkeypatch, capsys
):
    """Drive 1: a step fails (the drive goes on), a step needing it is not run, then the sealed
    truth build is refused at the tag: stop line, with `failed` and `not_run` lines before it. The
    fix and a complete drive follow: judged as a layout stop, the refusal inside the build's step
    (the build's step is number 3, its index 2)."""
    bad = drive_fake.entry("dev-truth", drive_fake.step("truth_build"))
    needing = drive_fake.entry(
        "dev-needs", drive_fake.step("truth_build"), after=["dev-truth"], only="sealed-run"
    )
    entries = [bad, needing, truth("--refuse-at-tag"), rehearsal_entry()]
    root, _ = made(tmp_path, monkeypatch, entries, modules={"truth_build": FLAKY_STEP})
    earlier_drive_shape(monkeypatch)
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "fail-now").write_text("")
    sealed_drive(1)
    (first,) = logs(root)
    kinds = [record["record"] for record in drive_fake.read_log(first)]
    assert kinds == [
        "start", "begin", "entry", "failed", "not_run", "begin", "entry", "stop", "end"
    ], kinds  # fmt: skip
    read = check.read_drive(first)
    assert [len(read.steps), len(read.failed), len(read.not_run)] == [2, 1, 1]
    assert read.stop["index"] == 2
    (root / "data" / "fail-now").unlink()
    fixed(root)
    sealed_drive(0)
    status, marks, _ = run(root, capsys)
    assert status == 0 and statuses(marks) == ["passed"] * 3, marks
    assert any(
        line.startswith(f"{first.name}: stopped at step 3 (sealed-truth) at a layout refusal")
        for line in marks["logged parsing fixes"][1]
    ), marks["logged parsing fixes"]


# --- 19. `read_drive` reads the `failed` and `not_run` lines of the driver and no other -------

HAND_FAILED = {"record": "failed", "index": 0, "label": "a", "reason": "the run ended with exit 1"}
HAND_NOT_RUN = {"record": "not_run", "index": 0, "label": "a", "reason": "not run: needs x"}
NEXT_BEGIN = {**HAND_BEGIN, "index": 1, "label": "b"}
NEXT_ENTRY = {**HAND_ENTRY, "index": 1, "label": "b"}


def read_hand(tmp_path: Path, lines: list[dict]):
    path = hand_log(tmp_path / "20260101T100000Z-sealed-run.jsonl", [HAND_START, *lines])
    return check.read_drive(path)


def test_19_read_drive_accepts_a_failed_line_after_its_entry_and_a_not_run_line_for_a_step(
    tmp_path,
):
    not_run_b = {**HAND_NOT_RUN, "index": 1, "label": "b"}
    drive = read_hand(
        tmp_path,
        [HAND_BEGIN, HAND_ENTRY, HAND_FAILED, not_run_b, {**HAND_STOP, "index": 2, "label": "c"}],
    )
    assert drive.steps == [(HAND_BEGIN, HAND_ENTRY)]
    assert (drive.failed, drive.not_run) == ([HAND_FAILED], [not_run_b])
    assert drive.stop["index"] == 2
    # a not-run step comes first: the next step's index follows it
    drive = read_hand(
        tmp_path, [HAND_NOT_RUN, {**HAND_BEGIN, "index": 1}, {**HAND_ENTRY, "index": 1}]
    )
    assert (drive.not_run, drive.failed) == ([HAND_NOT_RUN], [])
    assert [begin["index"] for begin, _ in drive.steps] == [1]
    # entries of both kinds, in order, name every problem
    drive = read_hand(
        tmp_path,
        [HAND_BEGIN, HAND_ENTRY, HAND_FAILED, {**HAND_NOT_RUN, "index": 1, "label": "b"}],
    )
    assert [(kind, line["label"]) for kind, line in drive.problems()] == [
        ("failed", "a"),
        ("not run", "b"),
    ]


@pytest.mark.parametrize(
    "lines",
    [
        [HAND_FAILED],
        [HAND_BEGIN, HAND_FAILED],
        [HAND_BEGIN, HAND_ENTRY, {**HAND_FAILED, "index": 1}],
        [HAND_BEGIN, HAND_ENTRY, {**HAND_FAILED, "label": "b"}],
        [HAND_BEGIN, HAND_ENTRY, {**HAND_FAILED, "reason": None}],
        [HAND_BEGIN, HAND_ENTRY, {**HAND_FAILED, "extra": 1}],
        [HAND_BEGIN, HAND_ENTRY, {k: v for k, v in HAND_FAILED.items() if k != "reason"}],
        [HAND_BEGIN, HAND_ENTRY, HAND_FAILED, HAND_FAILED],
        [HAND_BEGIN, HAND_ENTRY, HAND_STOP, HAND_FAILED],
        [HAND_BEGIN, HAND_ENTRY, HAND_NOT_RUN],
        [HAND_BEGIN, HAND_NOT_RUN, HAND_ENTRY],
        [{**HAND_NOT_RUN, "index": 1}],
        [{**HAND_NOT_RUN, "reason": 5}],
        [{**HAND_NOT_RUN, "label": None}],
        [{**HAND_NOT_RUN, "extra": 1}],
        [{**HAND_NOT_RUN, "index": True}] + [HAND_BEGIN, HAND_ENTRY],
        [HAND_NOT_RUN, HAND_BEGIN, HAND_ENTRY],
        [HAND_NOT_RUN, HAND_NOT_RUN],
        [HAND_BEGIN, HAND_ENTRY, HAND_FAILED, {**HAND_NOT_RUN, "index": 2}],
        [HAND_NOT_RUN, HAND_FAILED],
        [{**HAND_FAILED, "record": "failed "}, HAND_BEGIN, HAND_ENTRY],
    ],
    ids=[
        "a failed line first", "a failed line after a begin", "a failed line of another index",
        "a failed line of another label", "a failed line with no reason text",
        "a failed line with another field", "a failed line without its reason",
        "two failed lines", "a failed line after the stop", "a not-run line after an entry",
        "a not-run line between begin and entry", "a not-run line of the wrong index",
        "a not-run reason that is no text", "a not-run label that is no text",
        "a not-run line with another field", "a not-run index that is true",
        "a step reusing the index of a not-run step", "two not-run lines of one index",
        "a not-run line of an index that skips", "a failed line after a not-run line",
        "a failed line of another record name",
    ],
)  # fmt: skip
def test_19_read_drive_refuses_a_failed_or_not_run_line_of_any_other_shape_or_place(
    tmp_path, lines
):
    with pytest.raises(check.CannotCheck, match="are not the driver's"):
        read_hand(tmp_path, lines)


def test_19_a_failed_or_not_run_line_of_a_rehearsal_log_is_refused(tmp_path):
    lines = [{**HAND_START, "purpose": "rehearsal"}, HAND_NOT_RUN]
    path = hand_log(tmp_path / "20260101T100000Z-sealed-run.jsonl", lines)
    with pytest.raises(check.CannotCheck, match="purpose"):
        check.read_drive(path)
