"""`tools/wp15_check.py` on a sealed-run drive that ran to its end with a failed or a not-run step
(decision 99 (5) of 22 November 2026): WP15's pass mark is not passed, the exit status is 1, never
0, the first line after the header says that the held-out run is incomplete and names every failed
and every not-run step, and each part is a line begun "information:", read only from the steps
that passed under the conditions of the module text (AN INCOMPLETE DRIVE).

The drives are those of `test_wp15_check` (the real driver through the real gate, in a throwaway
repository frozen at a made-up tag). Each condition has its own test: a stopped drive, a tree
changed after the last step, another commit, another list, another tag or fix pair, an earlier
drive, the rehearsal, a step that does not verify, a `{run:}` chain and an `after` chain that
reach a step not read.
"""

import subprocess

import pytest
import test_wp15_check as base
from test_wp15_check import isolated_git  # noqa: F401  (autouse fixture)

check = base.check
LEARN_01, LEARN_02 = check.learn_label(0.1), check.learn_label(0.2)
INTERVALS_01, INTERVALS_02 = check.intervals_label(0.1), check.intervals_label(0.2)
MAIN_LEARN = check.learn_label()
INFO = "information: "
READABLE = "readable from the steps that passed, not a pass"


def information(lines: list[str], part: str) -> str:
    """The one line of the part `part` begun "information:" that says if it is read or lost."""
    (found,) = [
        x
        for x in lines
        if x.startswith(f"{INFO}{part}: ") and (": lost, " in x or READABLE in x or "not met" in x)
    ]
    return found


def readable(lines: list[str]) -> list[str]:
    return [x for x in lines if READABLE in x]


def restore_tracked(root) -> None:
    """The file a step changed, back as committed, as the gate wants a clean tree."""
    subprocess.run(["git", "-C", str(root), "checkout", "--", "tracked.txt"], check=True)


def failed_rho_02(tmp_path, monkeypatch, found=None):
    """The made-up repository with the rho 0.2 learn run failing, after its drive."""
    root = base.made(tmp_path, monkeypatch, found)
    base.put(root, LEARN_02, "exit", "1\n")
    base.drive(1)
    return root


# --- the contract: the incomplete line, information lines, exit 1 ---------------------------------


def test_an_incomplete_drive_prints_the_incomplete_line_then_information_and_exits_1(
    tmp_path, monkeypatch, capsys
):
    root = failed_rho_02(tmp_path, monkeypatch)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert lines[0] == check.HEADER
    (log,) = base.logs(root)
    assert lines[1].startswith(f"the held-out run is incomplete: {log.name} ended with exit status")
    assert f"failed step 5 ({LEARN_02})" in lines[1], lines[1]
    assert f"not run step 6 ({INTERVALS_02})" in lines[1], lines[1]
    middle = lines[2:-1]
    assert middle and all(x.startswith(INFO) for x in middle), middle
    lost = information(lines, check.PARTS[0])
    assert LEARN_02 in lost and INTERVALS_02 in lost, lost
    assert LEARN_01 not in lost, lost  # the steps that passed are not named lost
    assert READABLE in information(lines, check.PARTS[1])
    assert READABLE in information(lines, check.PARTS[2])
    assert not any(x.endswith(": reported") or ": reported," in x for x in lines), lines
    assert lines[-1].startswith("FAILED: WP15, the held-out run is incomplete"), lines
    assert "2 of 3 parts readable as information" in lines[-1], lines[-1]


def test_the_information_lines_hold_the_figures_of_the_steps_that_passed(
    tmp_path, monkeypatch, capsys
):
    root = failed_rho_02(tmp_path, monkeypatch)
    _, lines = base.run(root, capsys)
    assert any(
        x.startswith(f"{INFO}{MAIN_LEARN} (") and "scores asr-before-march-2023" in x for x in lines
    ), lines
    assert any(x.startswith(f"{INFO}{check.DRIFT_LABEL} (") and "guarded days" in x for x in lines)


def test_an_incomplete_drive_never_exits_0_when_every_part_is_readable(
    tmp_path, monkeypatch, capsys
):
    """Only a step no part reads fails (a learn entry added to the list). Every part is readable,
    and the exit status is still 1."""
    found = [*base.entries(), base.learn_entry(label="corner-other")]
    root = base.made(tmp_path, monkeypatch, found)
    base.put(root, "corner-other", "exit", "1\n")
    base.drive(1)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert len(readable(lines)) == 3, lines
    assert "corner-other" in lines[1], lines[1]
    assert lines[-1].startswith("FAILED: WP15"), lines


# --- each condition of the module text has its own test -------------------------------------------


def test_condition_a_stopped_drive_is_never_read(tmp_path, monkeypatch, capsys):
    """A step leaves the tree changed, the driver stops before the next step: the drive has a
    `stop` line, no part is read as information (never from its steps that passed)."""
    root = base.made(tmp_path, monkeypatch)
    base.put(root, LEARN_01, "dirty", "1\n")
    base.drive(2)
    restore_tracked(root)
    status, lines = base.run(root, capsys)
    assert status != 0, lines
    assert lines[1].startswith("the held-out run is incomplete: ") and "stopped before" in lines[1]
    assert any("a stopped drive is never read" in x for x in lines), lines
    assert readable(lines) == [], lines
    assert not any(x.endswith(": reported") for x in lines), lines


def test_condition_a_tree_changed_after_the_last_step_is_never_read(tmp_path, monkeypatch, capsys):
    root = base.made(tmp_path, monkeypatch)
    base.put(root, LEARN_02, "exit", "1\n")
    base.put(root, check.DRIFT_LABEL, "dirty", "1\n")  # the last step changes the tree
    status_of_drive = base.run_driver.main([base.LIST, *base.SEAL])
    assert status_of_drive != 0
    restore_tracked(root)
    status, lines = base.run(root, capsys)
    assert status != 0, lines
    assert any("its tree changed after the last step" in x for x in lines), lines
    assert readable(lines) == [], lines


@pytest.mark.parametrize(
    "change, said",
    [
        (lambda start: start.update(commit="d" * 40), f"is stale: written at commit {'d' * 40}"),
        (
            lambda start: start.update(list_sha256="e" * 64),
            "is not a drive of the bytes of",
        ),
        (
            lambda start: start.update(tag_commit="d" * 40),
            "records other commits of the tag or the fix than those the gate passes at now",
        ),
    ],
    ids=["another-commit", "another-list", "another-tag-or-fix"],
)
def test_conditions_another_commit_list_or_tag_fix_pair_is_never_read(
    tmp_path, monkeypatch, capsys, change, said
):
    root = failed_rho_02(tmp_path, monkeypatch)
    (log,) = base.logs(root)
    base.rewritten(log, lambda records: change(records[0]))
    status, lines = base.run(root, capsys)
    assert status == 2, lines
    assert any(said in x for x in lines), lines
    assert readable(lines) == [], lines
    assert not any(x.startswith(INFO) and "lost" not in x for x in lines[2:-1]), lines


def test_condition_only_the_newest_drive_that_ran_a_step_is_read(tmp_path, monkeypatch, capsys):
    """An earlier drive failed the rho 0.1 learn run; the newest failed the rho 0.2 one: part (a)
    names the newest drive's steps and none of the earlier drive's."""
    root = base.made(tmp_path, monkeypatch)
    base.put(root, LEARN_01, "exit", "1\n")
    base.drive(1)
    (root / "data" / "made-up" / LEARN_01 / "exit").unlink()
    base.put(root, LEARN_02, "exit", "1\n")
    base.drive(1)
    first, second = base.drive_fake.drive_logs(root)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert second.name in lines[1] and first.name not in lines[1], lines[1]
    lost = information(lines, check.PARTS[0])
    assert LEARN_02 in lost and INTERVALS_02 in lost and LEARN_01 not in lost, lost
    assert len(readable(lines)) == 2, lines


def test_condition_an_earlier_drive_is_not_read_when_the_newest_stopped(
    tmp_path, monkeypatch, capsys
):
    """The earlier drive ran to its end with the steps of parts (b) and (c) passed; the newest
    stopped. Nothing is read from either."""
    root = base.made(tmp_path, monkeypatch)
    base.put(root, LEARN_02, "exit", "1\n")
    base.drive(1)
    (root / "data" / "made-up" / LEARN_02 / "exit").unlink()
    base.put(root, MAIN_LEARN, "dirty", "1\n")
    base.drive(2)
    restore_tracked(root)
    status, lines = base.run(root, capsys)
    assert status != 0, lines
    assert readable(lines) == [], lines


def test_condition_the_rehearsal_is_never_read(tmp_path, monkeypatch, capsys):
    """The sealed run failed the rho 0.1 learn run; a newer complete rehearsal drive of the same
    list holds every part's steps, and part (a) is still lost."""
    root = base.made(tmp_path, monkeypatch)
    base.put(root, LEARN_01, "exit", "1\n")
    base.drive(1)
    (root / "data" / "made-up" / LEARN_01 / "exit").unlink()
    base.drive(0, base.REHEARSE)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    lost = information(lines, check.PARTS[0])
    assert LEARN_01 in lost and "lost" in lost, lost
    assert len(readable(lines)) == 2, lines


def test_condition_a_step_that_passed_but_does_not_verify_is_not_read(
    tmp_path, monkeypatch, capsys
):
    """The main model's learn run passed, then its results file changed: part (c) is lost, naming
    the step and that it does not verify; the part whose steps verify is readable."""
    root = failed_rho_02(tmp_path, monkeypatch)
    line = base.entry_line(root, MAIN_LEARN)
    file = root / line["run"] / "learn.json"
    file.write_text(file.read_text().replace("1.5", "2.5"))
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    lost = information(lines, check.PARTS[2])
    assert f"{MAIN_LEARN} (passed but does not verify" in lost, lost
    assert READABLE in information(lines, check.PARTS[1]), lines


def test_condition_a_run_chain_reaching_a_step_not_read_is_not_read(tmp_path, monkeypatch, capsys):
    """The rho 0.1 learn run passed but does not verify. The interval step that reads its run
    passed and verifies by itself; it is lost all the same, as its `{run:}` chain reaches a step
    not read."""
    root = failed_rho_02(tmp_path, monkeypatch)
    line = base.entry_line(root, LEARN_01)
    file = root / line["run"] / "learn.json"
    file.write_text(file.read_text().replace("1.5", "2.5"))
    _, lines = base.run(root, capsys)
    lost = information(lines, check.PARTS[0])
    assert f"{LEARN_01} (passed but does not verify" in lost, lost
    assert f"{INTERVALS_01} (needs {LEARN_01}, which is not read (passed but does not verify" in (
        lost
    ), lost


def fail_in_log(log, label: str) -> None:
    """The drive log with the entry of `label` read as failed, as the driver would have written
    that step failing and the drive going on, the steps after it left as they ran."""

    def change(records):
        for k, record in enumerate(records):
            if record.get("record") == "entry" and record.get("label") == label:
                record["exit_status"] = 1
                records.insert(
                    k + 1,
                    {
                        "record": "failed",
                        "index": record["index"],
                        "label": label,
                        "reason": "made to fail",
                    },
                )
                return
        raise AssertionError(f"no entry line of {label}")

    base.rewritten(log, change)


def test_condition_a_run_chain_reaching_a_failed_step_is_not_read(tmp_path, monkeypatch, capsys):
    """A drive log in which the rho 0.1 learn run failed but the interval step that takes its run
    folder ran on (the real driver logs that step not run): the interval step is lost, naming the
    failed step, though it passed and verifies."""
    root = base.made(tmp_path, monkeypatch)
    base.drive(0)
    (log,) = base.logs(root)
    fail_in_log(log, LEARN_01)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    lost = information(lines, check.PARTS[0])
    assert f"{LEARN_01} (failed: " in lost, lost
    assert f"{INTERVALS_01} (needs {LEARN_01}, which is not read (failed: " in lost, lost
    assert f"{LEARN_02} " not in lost, lost
    assert READABLE in information(lines, check.PARTS[1]), lines


def test_condition_an_after_chain_reaching_a_failed_step_is_not_read(tmp_path, monkeypatch, capsys):
    """The drift entry names the rho 0.1 learn entry in its `after`; that step failed and the
    drift step ran on in the log: part (b) is lost, naming the `after` entry. (The real driver
    logs such a step not run, and part (b) is lost the same way: the next test.)"""
    found = base.entries()
    for entry in found:
        if entry["label"] == check.DRIFT_LABEL:
            entry["after"] = [LEARN_01]
    root = base.made(tmp_path, monkeypatch, found)
    base.drive(0)
    (log,) = base.logs(root)
    fail_in_log(log, LEARN_01)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    lost = information(lines, check.PARTS[1])
    assert f"{check.DRIFT_LABEL} (needs {LEARN_01}, which is not read (failed: " in lost, lost


def test_an_after_entry_that_failed_leaves_the_driver_to_log_the_step_not_run(
    tmp_path, monkeypatch, capsys
):
    found = base.entries()
    for entry in found:
        if entry["label"] == check.DRIFT_LABEL:
            entry["after"] = [LEARN_01]
    root = base.made(tmp_path, monkeypatch, found)
    base.put(root, LEARN_01, "exit", "1\n")
    base.drive(1)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert f"not run step 7 ({check.DRIFT_LABEL})" in lines[1], lines[1]
    lost = information(lines, check.PARTS[1])
    assert f"{check.DRIFT_LABEL} (not run: " in lost, lost


# --- a configuration recorded as failed, read as information -------------------------------------


def test_a_failed_configuration_is_named_in_the_information_line_of_its_part(
    tmp_path, monkeypatch, capsys
):
    root = base.made(tmp_path, monkeypatch)
    document = base.learn_json(base.nine(), (root / base.MAIN).read_bytes())
    unit = check.ASR_UNITS[0]
    document["units"][unit]["configurations"][base.CONFIGURATIONS[0]]["rungs"] = {
        "reason": "failed: RuntimeError: made to fail"
    }
    base.put(root, MAIN_LEARN, "learn.json", document)
    base.put(root, LEARN_02, "exit", "1\n")
    base.drive(1)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    shown = information(lines, check.PARTS[2])
    assert "not met in the steps that passed" in shown, shown
    assert f"configuration {base.CONFIGURATIONS[0]} on unit {unit}: RuntimeError: made to fail" in (
        shown
    ), shown
