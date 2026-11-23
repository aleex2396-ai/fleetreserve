"""`tools/sealed_run_check.py` on a sealed-run drive with a learn step whose `learn.json` lists a
failed configuration: on the held-out route a configuration that raised an error on one unit
leaves the learn step's exit status at 0 (`learn_run`), so mark 1 is where the failure is found.

The drives are those of `test_sealed_run_check` (the real driver through the real gate, in a
throwaway repository frozen at a made-up tag). The stand-in of the learn step
(`drive_fake.LEARN_STEP`) writes the `learn.json` of the real step, with the failed configurations
given by `--failed`.

1. A learn step that exited 0 with one failed configuration fails mark 1 (exit 1), the line naming
   the step, the unit, the configuration and the error; every other mark passes.
2. Each failed configuration of each learn step is named; a drive whose learn steps list none
   passes, as before.
3. A learn step that exited 0 whose `learn.json` is missing could not run (exit 2), never
   skipped.
4. A drive that ran to its end with a failed step still fails and prints, as information, the
   failed configurations of the learn steps that passed.
5. `failed_configurations` lists one configuration of each group that failed together, while every
   configuration of the group holds `failed: <error>` in its rungs: two configurations failing on
   one unit and one on another are all three named, each once.
"""

import test_sealed_run_check as base
from test_sealed_run_check import isolated_git, recorder_folder  # noqa: F401  (autouse fixtures)

from fleetreserve import sealed_route

drive_fake = base.drive_fake
FAILED = "lgbm-k0.1-empirical | OSL_T: RuntimeError: made to fail"
OTHER = "lgbm-k0.3-empirical | korea-home: ValueError: made up as well"


def learn_with(*failed: str, label: str = "sealed-learn", extra=(), rungs=()) -> dict:
    arguments = [part for text in failed for part in ("--failed", text)]
    arguments += [part for text in rungs for part in ("--rung", text)]
    command = drive_fake.step("learn_run", *arguments, *extra)
    return drive_fake.entry(label, command)


def test_a_failed_configuration_of_a_learn_step_that_exited_0_fails_mark_1(
    tmp_path, monkeypatch, capsys
):
    root, _ = base.made(tmp_path, monkeypatch, [base.truth(), learn_with(FAILED)])
    base.sealed_drive(0)  # the step exits 0: the drive is complete
    status, marks, _ = base.run(root, capsys)
    assert status == 1, marks
    assert base.statuses(marks) == ["failed", "passed", "passed"], marks
    lines = marks["one logged run"][1]
    (named,) = [line for line in lines if "made to fail" in line]
    assert named.startswith(
        "FAILED: learn step sealed-learn: configuration lgbm-k0.1-empirical failed on unit OSL_T: "
        "RuntimeError: made to fail"
    )
    assert not [line for line in lines if line.startswith("one logged run: ")]


def test_each_failed_configuration_of_each_learn_step_is_named(tmp_path, monkeypatch, capsys):
    entries = [base.truth(), learn_with(FAILED, OTHER), learn_with(label="sealed-learn-two")]
    root, _ = base.made(tmp_path, monkeypatch, entries)
    base.sealed_drive(0)
    status, marks, _ = base.run(root, capsys)
    assert status == 1, marks
    shown = "\n".join(marks["one logged run"][1])
    assert (
        "learn step sealed-learn: configuration lgbm-k0.1-empirical failed on unit OSL_T" in shown
    )
    assert (
        "learn step sealed-learn: configuration lgbm-k0.3-empirical failed on unit korea-home"
        in shown
    )
    assert "sealed-learn-two" not in shown  # the step that lists none says nothing


def test_a_drive_whose_learn_steps_list_no_failure_passes(tmp_path, monkeypatch, capsys):
    root, _ = base.made(tmp_path, monkeypatch, [base.truth(), learn_with()])
    base.sealed_drive(0)
    status, marks, _ = base.run(root, capsys)
    assert status == 0, marks
    assert base.statuses(marks) == ["passed"] * 3, marks
    assert not [line for line in marks["one logged run"][1] if "learn step" in line]


def test_a_passed_learn_step_with_no_learn_json_could_not_run(tmp_path, monkeypatch, capsys):
    """The stand-in of the learn step without its lines that write `learn.json`: the check cannot
    know the failed configurations, so it could not run; it is not skipped."""
    root, _ = base.made(
        tmp_path,
        monkeypatch,
        [base.truth(), base.learn()],
        modules={"learn_run": drive_fake.STEP},
    )
    base.sealed_drive(0)
    status, marks, _ = base.run(root, capsys)
    assert status == 2, marks
    assert marks["one logged run"][0] == "could not run"
    assert "the failed configurations cannot be read" in "\n".join(marks["one logged run"][1])


def test_an_incomplete_drive_still_fails_and_shows_the_failed_configurations(
    tmp_path, monkeypatch, capsys
):
    failing = drive_fake.entry("sealed-bad", drive_fake.step("learn_run", "--exit", "1"))
    entries = [base.truth(), learn_with(FAILED), failing]
    root, _ = base.made(tmp_path, monkeypatch, entries, opener=sealed_route.TRUTH_BUILD_MODULE)
    base.sealed_drive(1)  # a step failed: no complete drive
    status, marks, _ = base.run(root, capsys)
    assert status == 1, marks
    assert marks["one logged run"][0] == "failed"
    shown = marks["one logged run"][1]
    assert [
        line
        for line in shown
        if line.startswith("information: ") and "OSL_T: RuntimeError: made to fail" in line
    ], shown


def test_every_configuration_of_a_failed_group_is_named_not_only_the_listed_one(
    tmp_path, monkeypatch, capsys
):
    """Two configurations fail on one unit (the list names the first of them) and one on another:
    the rungs of `learn.json` hold all three."""
    second = "lgbm-k0.2-empirical | OSL_T: RuntimeError: made to fail"
    entries = [base.truth(), learn_with(FAILED, OTHER, rungs=(FAILED, second, OTHER))]
    root, _ = base.made(tmp_path, monkeypatch, entries)
    base.sealed_drive(0)
    status, marks, _ = base.run(root, capsys)
    assert status == 1, marks
    shown = marks["one logged run"][1]
    for name, unit in (
        ("lgbm-k0.1-empirical", "OSL_T"),
        ("lgbm-k0.2-empirical", "OSL_T"),
        ("lgbm-k0.3-empirical", "korea-home"),
    ):
        named = [
            line
            for line in shown
            if f"learn step sealed-learn: configuration {name} failed on unit {unit}: " in line
        ]
        assert len(named) == 1, (name, unit, shown)
    assert len([line for line in shown if "learn step" in line]) == 3, shown
