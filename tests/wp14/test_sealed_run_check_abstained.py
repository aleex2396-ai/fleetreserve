"""`tools/sealed_run_check.py` on a complete sealed-run drive whose held-out truth build abstained a
unit: mark 1 gives one line for it, "abstained", with the build's run, the reason and the step,
"counted as not passing"; no mark and no exit status changes by it. A held-out truth build step
that passed with no summary among its outputs could not run (exit 2), never skipped.

The drives are those of `test_sealed_run_check` (the real driver through the real gate, in a
throwaway repository frozen at a made-up tag). The step standing in for the held-out truth build
(the module `heldout_truth_build`, as the run list names it) also writes a `truth-summary.json`
that records `korea-visits` as abstained with a reason, as the real build's summary does.
"""

import test_sealed_run_check as base
from test_sealed_run_check import isolated_git, recorder_folder  # noqa: F401  (autouse fixtures)

from fleetreserve import sealed_route

REASON = "no calibration day at the start draw"
SUMMARY_STEP = (
    base.TRUTH_STEP
    + f"""
with open(os.path.join(os.environ["FLEETRESERVE_RUN_DIR"], "truth-summary.json"), "w") as handle:
    json.dump({{"units": {{}}, "abstained": {{"korea-visits": {{"reason": "{REASON}"}}}}}}, handle)
"""
)


def _truth_entry() -> dict:
    command = base.drive_fake.step(sealed_route.TRUTH_BUILD_MODULE)
    return base.drive_fake.entry("sealed-truth", command, only="sealed-run", max_seconds=60)


def test_an_abstained_unit_is_shown_as_abstained(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(base, "TRUTH_STEP", SUMMARY_STEP)
    root, _ = base.made(
        tmp_path,
        monkeypatch,
        [_truth_entry(), base.learn()],
        opener=sealed_route.TRUTH_BUILD_MODULE,
    )
    base.sealed_drive(0)
    (log,) = base.logs(root)
    truth_run = next(
        r["run"]
        for r in base.drive_fake.read_log(log)
        if r.get("label") == "sealed-truth" and r.get("run")
    )
    status, marks, _ = base.run(root, capsys)
    assert status == 0, marks
    assert base.statuses(marks) == ["passed"] * 3, marks
    lines = marks["one logged run"][1]
    expected = (
        f"korea-visits: abstained in the held-out truth build {truth_run.rsplit('/', 1)[1]}: "
        f"{REASON}; not scored, counted as not passing (step sealed-truth)"
    )
    assert expected in lines
    assert lines[-1].startswith("one logged run: ")  # the mark's verdict line is still the last


def test_a_drive_whose_build_abstained_nothing_has_no_such_line(tmp_path, monkeypatch, capsys):
    root = base.complete_at_the_tag(tmp_path, monkeypatch)
    status, marks, _ = base.run(root, capsys)
    assert status == 0, marks
    assert not [line for line in marks["one logged run"][1] if "abstained" in line]


def test_a_passed_build_step_with_no_summary_could_not_run(tmp_path, monkeypatch, capsys):
    """The stand-in build of `test_sealed_run_check`, with the lines that write its
    `truth-summary.json` taken out: the check cannot know the abstained units, so it could not
    run; it is not skipped."""
    monkeypatch.setattr(base, "TRUTH_STEP", base.TRUTH_STEP.split("# the summary the real")[0])
    root, _ = base.made(
        tmp_path,
        monkeypatch,
        [_truth_entry(), base.learn()],
        opener=sealed_route.TRUTH_BUILD_MODULE,
    )
    base.sealed_drive(0)
    status, marks, _ = base.run(root, capsys)
    assert status == 2, marks
    assert marks["one logged run"][0] == "could not run"
    assert any(
        line.startswith("COULD NOT RUN: sealed-truth: the held-out truth build step passed but")
        and "lists no truth-summary.json" in line
        for line in marks["one logged run"][1]
    ), marks
