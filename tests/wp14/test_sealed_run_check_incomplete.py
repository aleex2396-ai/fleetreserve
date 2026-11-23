"""`tools/sealed_run_check.py` on sealed-run drives that did not complete, written by the real
driver through the real gate in a throwaway repository frozen at a made-up tag (the drives of
`test_sealed_run_check`).

1. A failure before the first sealed truth build, on the order of the committed run list (the
   development builds, the sealed builds with no `after`, then the steps that read them), stops
   the drive before any sealed file is opened: mark 2 counts it as an operational stop and a later
   complete drive passes every mark.
2. With no complete drive, the newest drive that ran to its end is the run made once: mark 1 alone
   fails it, naming each failed and each not-run step; mark 2 passes and does not call it an
   earlier drive. An earlier drive that opened sealed data and ran to its end still fails mark 2.
3. After the failure, mark 1 prints as information only (lines begun "information:") O2's
   verdicts and the abstained held-out units read from steps that passed and verify; the exit
   status does not change by them; a step that failed, or whose run does not verify, is not read.
4. A failure line never prints "None" for what changed or for a missing reason.
"""

import json
from pathlib import Path

import test_sealed_run_check as base
from test_sealed_run_check import isolated_git, recorder_folder  # noqa: F401  (autouse fixtures)

from fleetreserve import sealed_route

drive_fake = base.drive_fake
check = base.check
REASON = "no calibration day at the start draw"
SUMMARY_STEP = (
    base.TRUTH_STEP
    + f"""
with open(os.path.join(os.environ["FLEETRESERVE_RUN_DIR"], "truth-summary.json"), "w") as handle:
    json.dump({{"units": {{}}, "abstained": {{"korea-visits": {{"reason": "{REASON}"}}}}}}, handle)
"""
)


def committed_order(held_fails: bool = False) -> list[dict]:
    """The committed list's order (`drive_fake.o2_entries`): `dev-truth`, `dev-truth-held`,
    `sealed-truth`, `sealed-truth-held` (no `after`), then the two O2 steps, the second made to
    exit 1 when `held_fails`."""
    entries = drive_fake.o2_entries("sealed-o2")
    if held_fails:
        entries[-1]["command"] = [*entries[-1]["command"], "--exit", "1"]
    return entries


def at_the_tag(tmp_path, monkeypatch, entries, **options):
    return base.made(
        tmp_path,
        monkeypatch,
        entries,
        opener=sealed_route.TRUTH_BUILD_MODULE,
        units=drive_fake.O2_UNITS,
        **options,
    )


def say(marks: dict, name: str) -> str:
    return "\n".join(marks[name][1])


# --- 1. a failure before the first sealed truth build --------------------------------------------


def test_1_a_failed_development_build_stops_before_any_opening_and_a_later_drive_passes(
    tmp_path, monkeypatch, capsys
):
    root, _ = at_the_tag(
        tmp_path, monkeypatch, committed_order(), modules={"truth_build": base.FLAKY_STEP}
    )
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "fail-now").write_text("")
    base.sealed_drive(1)
    (first,) = base.logs(root)
    kinds = [record["record"] for record in drive_fake.read_log(first)]
    assert kinds == ["start", "begin", "entry", "stop", "end"], kinds
    assert not base.openings_file(root).exists() or not base.openings_file(root).read_text()
    (root / "data" / "fail-now").unlink()
    base.sealed_drive(0)
    status, marks, _ = base.run(root, capsys)
    assert status == 0, marks
    assert base.statuses(marks) == ["passed"] * 3, marks
    assert any(
        line.startswith(f"{first.name}: stopped after step 1 (dev-truth): ")
        and line.endswith("with no opening of sealed data in any of its steps: an operational stop")
        for line in marks["logged parsing fixes"][1]
    ), marks["logged parsing fixes"]


# --- 2. one incomplete drive fails mark 1 alone --------------------------------------------------


def test_2_one_incomplete_drive_fails_mark_1_alone_naming_its_failed_step(
    tmp_path, monkeypatch, capsys
):
    root, _ = at_the_tag(tmp_path, monkeypatch, committed_order(held_fails=True))
    base.sealed_drive(1)
    (only,) = base.logs(root)
    status, marks, results = base.run(root, capsys)
    assert status == 1, marks
    assert base.statuses(marks) == ["failed", "passed", "passed"], marks
    assert results["status"] == "failed"
    assert (
        f"FAILED: no complete sealed-run drive; the newest, {only.name}, ended with exit status "
        "1; failed step 6 (sealed-o2-held): the run ended with exit status 1"
    ) in marks["one logged run"][1], marks["one logged run"]
    fixes = marks["logged parsing fixes"][1]
    assert not [line for line in fixes if line.startswith(("FAILED", "COULD NOT RUN"))], fixes
    assert not [line for line in fixes if "counts only as stopped" in line], fixes
    assert f"{only.name}: the newest drive ran to its end with no complete drive" in fixes[-2]
    for name in base.MARKS:
        assert "None" not in say(marks, name), marks[name]


def test_2_an_earlier_incomplete_drive_that_opened_sealed_data_still_fails_mark_2(
    tmp_path, monkeypatch, capsys
):
    root, _ = at_the_tag(tmp_path, monkeypatch, committed_order(held_fails=True))
    base.sealed_drive(1)
    base.sealed_drive(1)
    first, second = base.logs(root)
    status, marks, _ = base.run(root, capsys)
    assert status == 1, marks
    assert base.statuses(marks) == ["failed", "failed", "passed"], marks
    (line,) = [line for line in marks["logged parsing fixes"][1] if line.startswith("FAILED: ")]
    assert first.name in line and second.name not in line, line
    assert f"{second.name}: the newest drive ran to its end" in say(marks, "logged parsing fixes")


# --- 3. information after the failure ------------------------------------------------------------


def test_3_o2_verdicts_and_abstained_units_of_passed_steps_follow_the_failure(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(base, "TRUTH_STEP", SUMMARY_STEP)
    root, _ = at_the_tag(tmp_path, monkeypatch, committed_order(held_fails=True))
    base.sealed_drive(1)
    (only,) = base.logs(root)
    runs = {r["label"]: r["run"] for r in drive_fake.read_log(only) if r["record"] == "entry"}
    status, marks, _ = base.run(root, capsys)
    assert status == 1, marks
    lines = marks["one logged run"][1]
    failure = next(k for k, line in enumerate(lines) if line.startswith("FAILED: no complete"))
    after = lines[failure + 1 :]
    assert (
        f"information: O2 sealed-o2: over the units passed; failed on none; could not decide on "
        f"none (from {runs['sealed-o2']}/ceiling-report.json; not in this check's exit status, "
        "reading R7)"
    ) in after, after
    for build in ("sealed-truth", "sealed-truth-held"):
        assert (
            "information: korea-visits: abstained in the held-out truth build "
            f"{runs[build].rsplit('/', 1)[1]}: {REASON}; not scored, counted as not passing "
            f"(step {build})"
        ) in after, after
    assert not [line for line in after if "O2 sealed-o2-held" in line], after  # the failed step
    assert base.statuses(marks) == ["failed", "passed", "passed"], marks


def test_3_a_passed_step_that_does_not_verify_is_not_read_and_the_exit_does_not_change(
    tmp_path, monkeypatch, capsys
):
    root, _ = at_the_tag(tmp_path, monkeypatch, committed_order(held_fails=True))
    base.sealed_drive(1)
    (only,) = base.logs(root)
    run = next(
        r["run"]
        for r in drive_fake.read_log(only)
        if r["record"] == "entry" and r["label"] == "sealed-o2"
    )
    report = Path(root) / run / "ceiling-report.json"
    report.write_text(report.read_text() + " ")  # no longer the bytes its manifest records
    status, marks, _ = base.run(root, capsys)
    assert status == 1, marks
    lines = marks["one logged run"][1]
    assert not [line for line in lines if line.startswith("information: O2 sealed-o2:")], lines
    assert any(
        line.startswith("information: step sealed-o2 passed but does not verify, not read: ")
        for line in lines
    ), lines
    assert base.statuses(marks)[0] == "failed", marks


def test_3_nothing_is_read_from_an_incomplete_drive_of_another_commit(
    tmp_path, monkeypatch, capsys
):
    root, _ = at_the_tag(tmp_path, monkeypatch, committed_order(held_fails=True))
    base.sealed_drive(1)
    (only,) = base.logs(root)

    def another_commit(records):
        records[0]["commit"] = "d" * 40

    base.rewritten(only, another_commit)
    status, marks, _ = base.run(root, capsys)
    assert status == 1, marks
    lines = marks["one logged run"][1]
    assert not [line for line in lines if line.startswith("information: O2 ")], lines
    assert any(
        line.startswith("information: no O2 verdict or abstained unit is read from ")
        and "is stale" in line
        for line in lines
    ), lines


def test_3_a_complete_drive_has_no_information_lines(tmp_path, monkeypatch, capsys):
    root = base.complete_at_the_tag(tmp_path, monkeypatch)
    status, marks, _ = base.run(root, capsys)
    assert status == 0, marks
    assert not [line for line in marks["one logged run"][1] if line.startswith("information:")]


# --- 4. no "None" --------------------------------------------------------------------------------


def hand_drive(end: dict, **lines) -> "check.Drive":
    start = {"record": "start"}
    return check.Drive("d.jsonl", [start, {"record": "end", **end}], [], None, **lines)


def test_4_the_failure_line_names_no_change_when_nothing_changed():
    text = check._stopped(hand_drive({"exit_status": 1, "change_after": None}))
    assert text == "ended with exit status 1"
    changed = check._stopped(hand_drive({"exit_status": 2, "change_after": "the commit changed"}))
    assert changed == "ended with exit status 2: the commit changed"


def test_4_a_failed_or_not_run_entry_without_a_reason_says_so():
    failed = [{"index": 0, "label": "a", "reason": None}]
    not_run = [{"index": 1, "label": "b"}]
    text = check._stopped(hand_drive({"exit_status": 1}, failed=failed, not_run=not_run))
    assert text == (
        "ended with exit status 1; failed step 1 (a): no reason logged; "
        "not run step 2 (b): no reason logged"
    )
    stop = {"record": "stop", "index": 0, "label": "a"}
    drive = check.Drive("d.jsonl", [{"record": "start"}, {"record": "end"}], [], stop)
    assert check._stopped(drive) == "stopped before step 1 (a): no reason logged"
    assert "None" not in json.dumps(text)
