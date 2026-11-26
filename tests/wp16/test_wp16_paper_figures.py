"""The figures the text needs from the run's own point values: `o4-points`, `o5-drift`, and the
manifest's `code_commit` and `code_dirty` (made-up files in the writers' forms; nothing is run on
real data).

Red when: a loss is read from the wrong tier, rung, key or model; a limit is not beside the loss; a
unit at scale 0 is shown with a loss or a missing figure becomes 0; a model whose learn step is
missing is given a number; the drift table reads the driver's bound for the slot's, sums the wrong
losses, or leaves a WP15 part unmarked after a WP14 failure; an absent drift run is a number or
changes the exit status; the drift table drops the path's mean offer or the driver loss's bound
or reads the slot's for it; the manifest names the commit of `--root` as the code's commit, leaves
out `code_dirty`, reads another checkout's status, or lets a missing git raise.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from wp16_made_up import (
    COMMIT,
    NAME_MAIN,
    UNITS,
    MadeUp,
    learn_document,
    matrix_same,
    run_list_document,
    sha,
)

from fleetreserve import wp16_figures, wp16_o4, wp16_o5, wp16_run
from fleetreserve.wp16_cells import Missing, csv_text, markdown_text
from fleetreserve.wp16_drive import NoDrive, Source, find_drive
from fleetreserve.wp16_learn import Runs

REAL_CODE_DIRTY = wp16_run.code_dirty  # the autouse fixture below replaces it in each test
SAME = {u: NAME_MAIN for u in UNITS}
MATRIX = "gate/sealed-matrix/lgbm_tail.json"
CODE = "c" * 40
OFFERS = {"u1": (1.0, 2.0), "u2": (1.0, 3.0), "u3": (1.0, 4.0)}


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch):
    def render(spec, folder):
        name = f"{spec['kind']}.png"
        (folder / name).write_bytes(b"placeholder")
        return name

    monkeypatch.setattr(wp16_figures, "render", render)
    monkeypatch.setattr(wp16_run, "code_dirty", lambda: False)  # the real one asks git
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)


def one_way(document: dict, unit: str) -> dict:
    return document["units"][unit]["configurations"][NAME_MAIN]["rungs"]["one-way"]


def top(tmp_path, *, offers=OFFERS, hazard=None, steps=(), tweak=None):
    """A made-up drive with `learn-lgbm_tail` (offers by unit) and, with `hazard`,
    `learn-hazard_tuned_inputs`, then `steps`; `tweak(label, learn document)` edits a learn file
    before it is written. Returns (source, runs)."""
    made = MadeUp(tmp_path)
    data = made.commit_file(MATRIX, matrix_same())
    made.commit_file("run-lists/run-list.json", run_list_document())
    drive_steps = []
    for label, found in (("learn-lgbm_tail", offers), ("learn-hazard_tuned_inputs", hazard)):
        if found is None:
            continue
        made.commands[label] = ["made-up", "--matrix", MATRIX]
        document = learn_document(sha(data), found, SAME)
        if tweak is not None:
            tweak(label, document)
        drive_steps.append((label, {"learn.json": document}))
    made.drive(drive_steps + list(steps))
    source = Source(tmp_path, find_drive(tmp_path))
    return source, Runs(source)


def points_rows(runs, absent=None) -> dict:
    (table,) = wp16_o4.build_points(runs, UNITS, absent or {})
    return {(row[0], row[1]): row for row in table.rows}


# --- o4-points ------------------------------------------------------------------------------------


def test_the_points_are_the_runs_tier_a_one_way_values_each_loss_beside_its_limit(tmp_path):
    def tweak(label, document):
        test = one_way(document, "u1")["test"]
        shift = 0.01 if label == "learn-hazard_tuned_inputs" else 0.0
        test.update(
            mean_slot_loss_offering_days=0.1457 - shift,
            mean_driver_loss_offering_days=0.0424 - shift,
            mean_offer_kwh_per_day=2.5,
            offering_days=7,
        )
        one_way(document, "u1")["beta_hat"] = 0.75
        # tier B and the fixed rung hold other numbers: a read from them is a wrong read
        document["units"]["u1"]["configurations"][NAME_MAIN]["rungs"]["fixed"]["test"].update(
            mean_slot_loss_offering_days=0.9, mean_offer_kwh_per_day=9.0
        )

    _, runs = top(tmp_path, hazard=OFFERS, tweak=tweak)
    rows = points_rows(runs)
    main = rows[("lgbm_tail", "u1")]
    assert main[2] == NAME_MAIN
    assert main[3:10] == (2.5, 0.1457, 0.10, 0.0424, 0.05, 0.75, 7)
    assert main[10] == ""
    other = rows[("hazard_tuned_inputs", "u1")]
    assert other[4] == pytest.approx(0.1357) and other[6] == pytest.approx(0.0324)
    assert len(rows) == 6  # three units for each of the two models
    # red: reading tier B (`unit_test(..., "B", ...)`) or the fixed rung changes a number above;
    # reading the driver key for the slot cell (or the reverse) swaps 0.1457 and 0.0424.


def test_the_points_table_has_no_interval_and_no_verdict(tmp_path):
    _, runs = top(tmp_path)
    (table,) = wp16_o4.build_points(runs, UNITS, {})
    assert table.name == "o4-points"
    assert not any("pass" in column or "interval" in column for column in table.columns)
    text = csv_text(table)
    assert " low" not in text.splitlines()[-len(table.rows) - 1]  # no interval columns
    assert "Interval" not in repr(table.rows)
    assert "no verdict" in table.title
    # red: adding an Interval cell or a "passes" column to `POINTS_COLUMNS` turns it red.


def test_a_unit_at_scale_0_is_not_judged_and_has_no_loss_shown(tmp_path):
    def tweak(label, document):
        zero = one_way(document, "u3")
        zero["beta_hat"] = 0
        zero["test"].update(
            mean_offer_kwh_per_day=0.0,
            mean_slot_loss_offering_days=0.0,
            mean_driver_loss_offering_days=0.0,
            offering_days=0,
        )

    _, runs = top(tmp_path, tweak=tweak)
    row = points_rows(runs)[("lgbm_tail", "u3")]
    assert row[3] == 0.0  # the run's own offer
    assert row[4] == row[6] == Missing("not judged: scale 0 (no offer, so no loss)")
    assert (row[5], row[7]) == (0.10, 0.05)  # the limits stay beside the cells
    assert row[8] == 0 and row[9] == 0
    assert row[10] == "not judged: scale 0"
    other = points_rows(runs)[("lgbm_tail", "u1")]
    assert other[4] == 0.02 and other[6] == 0.01  # the unit not at zero keeps its numbers
    # red: removing the `if scale == 0:` branch of `build_points` shows 0.0 for both losses.


def test_a_model_whose_learn_step_is_missing_says_so_and_never_shows_a_zero(tmp_path):
    _, runs = top(tmp_path, steps=[("learn-hazard_tuned_inputs", "fail")])
    rows = points_rows(runs)
    for unit in UNITS:
        row = rows[("hazard_tuned_inputs", unit)]
        assert isinstance(row[3], Missing)
        assert row[3].text.startswith("not produced: failed: learn-hazard_tuned_inputs")
        assert row[4:10] == ("",) * 6  # no number, no zero
        assert row[10].startswith("failed: learn-hazard_tuned_inputs")
        assert rows[("lgbm_tail", unit)][3] == OFFERS[unit][1]  # the other model is read as before
    # red: filling the missing model's cells from the other model, or with 0, fails the asserts.


def test_a_model_with_no_step_in_the_drive_says_so_too(tmp_path):
    _, runs = top(tmp_path)
    row = points_rows(runs)[("hazard_tuned_inputs", "u1")]
    text = "not produced: no entry writes it: the drive has no step learn-hazard_tuned_inputs"
    assert row[3] == Missing(text)


def test_a_failed_unit_and_a_unit_without_a_window_are_named_not_counted_as_zero(tmp_path):
    offers = {"u1": (1.0, 2.0), "u2": "failed: ValueError: boom"}
    _, runs = top(tmp_path, offers=offers)
    rows = points_rows(runs, {"u3": "a window of zero hours"})
    failed = rows[("lgbm_tail", "u2")]
    assert failed[3].text.startswith("not produced: failed: ValueError: boom")
    assert failed[4:10] == ("",) * 6
    window = rows[("lgbm_tail", "u3")]
    assert "no window (a window of zero hours)" in window[3].text
    assert window[10].startswith("no figure: no window")
    # red: a `0` or an empty cell in place of the `Missing` in the `test is None` branch.


def test_a_test_block_that_holds_no_loss_gives_no_number(tmp_path):
    def tweak(label, document):
        del one_way(document, "u1")["test"]["mean_driver_loss_offering_days"]
        one_way(document, "u1")["test"]["mean_slot_loss_offering_days"] = None

    _, runs = top(tmp_path, tweak=tweak)
    row = points_rows(runs)[("lgbm_tail", "u1")]
    assert isinstance(row[4], Missing) and "mean_slot_loss_offering_days" in row[4].text
    assert isinstance(row[6], Missing) and "mean_driver_loss_offering_days" in row[6].text
    assert row[3] == 2.0
    # red: `test.get(key, 0)` in `_point` would show 0 for both.


def test_the_points_cells_are_written_to_csv_and_markdown(tmp_path):
    _, runs = top(tmp_path)
    (table,) = wp16_o4.build_points(runs, UNITS, {})
    assert csv_text(table).count("lgbm_tail") >= 3
    assert "| model | unit | configuration |" in markdown_text(table)


# --- o5-drift -------------------------------------------------------------------------------------


def guarded(alpha, mean, bound, over):
    return {
        "guarded": True,
        "alpha": alpha,
        "mean": mean,
        "failing_days": 5,
        "beta_1": 0.0,
        "final_state": 0.1,
        "final_bound": bound,
        "within_alpha": mean <= alpha,
        "days_running_mean_over_alpha": 3,
        "days_over_bound": over,
        "largest_excess_over_bound": 0.001,
    }


def reported(mean):
    return {"guarded": False, "alpha": None, "mean": mean, "failing_days": 4}


def path(tier, eta, steps, slot, driver, offer=2.5):
    return {
        "tier": tier,
        "start": "calibrated",
        "eta": float(eta),
        "guarded_counting_days": 120,
        "offering_days": steps,
        "mean_offer_kwh_per_counting_day": offer,
        "mean_offer_kwh_per_offering_day": offer * 3,  # another key: a read from it is wrong
        "losses": {"slot": slot, "driver": driver},
    }


def drift_document(paths=None, status="run") -> dict:
    """`drift-run.json` in the writer's form (`drift_run.unit_entry`, `drift_path.Path.summary`)."""
    if paths is None:
        paths = {
            "A": {
                "calibrated": {
                    "0.5": path(
                        "A",
                        0.5,
                        106,
                        guarded(0.10, 0.103, 0.108, 3),
                        guarded(0.05, 0.046, 0.0511, 1),
                    )
                }
            },
            "B": {
                "calibrated": {
                    "2": path(
                        "B", 2, 99, guarded(0.10, 0.097, 0.112, 2), reported(0.12), offer=1.75
                    )
                }
            },
        }
    entry = {
        "role": "held-out",
        "status": status,
        "counting_days": 156,
        "guarded_days": 106,
        "paths": paths,
    }
    return {"label": "made up", "units": {"u1": entry}, "failed_units": []}


def drift_rows(source) -> list:
    (table,) = wp16_o5.build_drift(source)
    assert table.name == "o5-drift"
    return table.rows


def test_the_drift_end_values_are_the_files_own_per_tier(tmp_path):
    source, _ = top(tmp_path, steps=[("drift-lgbm_tail", {"drift-run.json": drift_document()})])
    first, second = drift_rows(source)
    assert first == (
        *("u1", "A", "calibrated", "0.5", 106, 0.103, 0.108, 0.10, 0.046, 0.05, 4),
        *(2.5, 0.0511, "no"),
    )
    assert second[:6] == ("u1", "B", "calibrated", "2", 99, 0.097)
    assert second[6:10] == (0.112, 0.10, 0.12, "not held down (reported)")
    assert second[10] == 2  # tier B: the slot guard's steps only; the driver loss is not held down
    assert second[11:] == (1.75, "not held down (reported)", "no")
    # red: reading the driver's `final_bound` (0.0511) for the slot's bound, summing only the
    # slot's `days_over_bound` (3, not 4) or summing an unguarded loss, turns a cell red; so does
    # `_number(found, "mean_offer_kwh_per_counting_day")` read as `..._per_offering_day` (7.5), the
    # slot's `final_bound` (0.108) in the driver's bound cell, or the bound shown for tier B.


def test_the_drift_table_is_marked_information_only_when_wp14_fails(tmp_path):
    drift = ("drift-lgbm_tail", {"drift-run.json": drift_document()})
    source, _ = top(tmp_path / "ok", steps=[drift])
    assert source.drive.complete
    assert {row[-1] for row in drift_rows(source)} == {"no"}
    (table,) = wp16_o5.build_drift(source)
    assert not any("Information only" in note for note in table.notes)

    source, _ = top(tmp_path / "step", steps=[drift, ("o3", "fail")])
    assert not source.drive.complete
    assert {row[-1] for row in drift_rows(source)} == {"yes"}
    (table,) = wp16_o5.build_drift(source)
    assert any(note.startswith("Information only") for note in table.notes)

    failed_unit = {"u1": (1.0, 2.0), "u2": "failed: ValueError: boom", "u3": (1.0, 4.0)}
    source, _ = top(tmp_path / "unit", offers=failed_unit, steps=[drift])
    assert source.drive.complete  # every step exited 0, a configuration failed on a unit
    assert {row[-1] for row in drift_rows(source)} == {"yes"}
    # red: dropping `not failed` from `information_only` (`wp16_failed`) turns the third case red;
    # dropping `source.drive.complete` there turns the second red.


def test_a_drift_run_that_is_absent_is_not_produced_and_no_number(tmp_path):
    source, _ = top(tmp_path, steps=[("drift-lgbm_tail", "fail")])
    (row,) = drift_rows(source)
    assert row[0] == "all units"
    assert isinstance(row[1], Missing)
    assert row[1].text.startswith("not produced: failed: drift-lgbm_tail")
    assert row[2:13] == ("",) * 11 and row[13] == "yes"
    assert not wp16_run.any_unreadable(wp16_o5.build_drift(source))  # never moves the exit status
    # red: a number or an empty cell in the `except Missed` branch of `build_drift`.


def test_a_drive_without_a_drift_step_is_not_produced_either(tmp_path):
    source, _ = top(tmp_path)
    (row,) = drift_rows(source)
    text = "not produced: no entry writes it: the drive has no step drift-lgbm_tail"
    assert row[1] == Missing(text)


def test_a_tier_with_no_path_is_named_with_the_runs_status(tmp_path):
    only_a = drift_document()
    del only_a["units"]["u1"]["paths"]["B"]
    nothing = drift_document({}, status="too few counting days: 3")
    nothing["units"]["u2"] = nothing["units"].pop("u1")
    only_a["units"].update(nothing["units"])
    source, _ = top(tmp_path, steps=[("drift-lgbm_tail", {"drift-run.json": only_a})])
    rows = drift_rows(source)
    assert [(r[0], r[1]) for r in rows] == [("u1", "A"), ("u1", "B"), ("u2", "A"), ("u2", "B")]
    assert rows[0][4] == 106
    for row in rows[1:]:
        assert isinstance(row[2], Missing) and row[2].text.startswith("not produced: no path of")
        assert row[3:13] == ("",) * 10
    assert "(status: run)" in rows[1][2].text
    assert "(status: too few counting days: 3)" in rows[2][2].text


def test_a_count_the_file_does_not_hold_is_not_made_a_zero(tmp_path):
    document = drift_document()
    del document["units"]["u1"]["paths"]["A"]["calibrated"]["0.5"]["losses"]["slot"][
        "days_over_bound"
    ]
    del document["units"]["u1"]["paths"]["A"]["calibrated"]["0.5"]["offering_days"]
    source, _ = top(tmp_path, steps=[("drift-lgbm_tail", {"drift-run.json": document})])
    row = drift_rows(source)[0]
    assert isinstance(row[10], Missing) and "days_over_bound" in row[10].text
    assert isinstance(row[4], Missing) and "offering_days" in row[4].text
    assert row[5] == 0.103  # the other cells are still the file's
    # red: `loss.get("days_over_bound", 0)` in `_drift_row` would show 1 here (driver's only).


def test_the_two_added_figures_are_not_produced_when_the_file_lacks_them_never_zero(tmp_path):
    document = drift_document()
    tier_a = document["units"]["u1"]["paths"]["A"]["calibrated"]["0.5"]
    del tier_a["mean_offer_kwh_per_counting_day"]
    del tier_a["losses"]["driver"]["final_bound"]
    tier_b = document["units"]["u1"]["paths"]["B"]["calibrated"]["2"]
    tier_b["mean_offer_kwh_per_counting_day"] = None  # a path with no counting day writes None
    source, _ = top(tmp_path, steps=[("drift-lgbm_tail", {"drift-run.json": document})])
    first, second = drift_rows(source)
    assert isinstance(first[11], Missing)
    assert first[11].text == "not produced: the file holds no mean_offer_kwh_per_counting_day"
    assert isinstance(first[12], Missing)
    assert first[12].text == "not produced: the file holds no final_bound"
    assert isinstance(second[11], Missing) and second[11].text.startswith("not produced: ")
    assert second[12] == "not held down (reported)"  # a driver loss that is not held down has none
    assert first[5:11] == (0.103, 0.108, 0.10, 0.046, 0.05, 4)  # the other cells are the file's
    assert not wp16_run.any_unreadable(wp16_o5.build_drift(source))  # the exit status is unchanged
    # red: `value.get("final_bound", 0)` or `or 0` in the new cells gives 0 for either figure;
    # a Missing text that starts with "failed:", "not run:" or "unreadable:" turns the last line
    # red.


def test_the_drift_table_names_the_two_new_columns_and_notes_them(tmp_path):
    source, _ = top(tmp_path, steps=[("drift-lgbm_tail", {"drift-run.json": drift_document()})])
    (table,) = wp16_o5.build_drift(source)
    assert table.columns[11:] == (
        "mean offer (kWh per counting day)",
        "driver loss, T10 bound at the end",
        "information only",
    )
    assert "mean offer is the path's mean over its guarded counting days" in table.notes[-1]
    assert "| mean offer (kWh per counting day) |" in markdown_text(table)
    # red: a column left out of `DRIFT_COLUMNS` makes `table.add` raise "cells for columns".


# --- the run: files, manifest, exit status --------------------------------------------------------


def run_top(tmp_path):
    """A complete made-up drive of the truth build, one learn step and no drift step."""
    made = MadeUp(tmp_path)
    data = made.commit_file(MATRIX, matrix_same())
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.commands["learn-lgbm_tail"] = ["made-up", "--matrix", MATRIX]
    truth = {"abstained": {"u3": {"reason": "no window"}}, "units": {"u1": {}, "u2": {}}}
    gate = {
        "configurations": {
            NAME_MAIN: {
                "units": {
                    u: {"fixed_every_request": {"test_mean_offer_kwh_per_day": 0.25}}
                    for u in ("u1", "u2")
                }
            }
        }
    }
    learn = learn_document(sha(data), {"u1": (1.0, 2.0), "u2": (1.0, 3.0)}, SAME)
    made.drive(
        [
            ("sealed-truth", {"truth-summary.json": truth}),
            ("learn-lgbm_tail", {"learn.json": learn, "learn-gate.json": gate}),
        ]
    )


def test_the_two_tables_are_written_named_in_the_manifest_and_the_exit_status_is_unchanged(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)
    run_top(tmp_path)
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 0  # as before the tables
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    for name in ("o4-points.csv", "o4-points.md", "o5-drift.csv", "o5-drift.md"):
        assert manifest["outputs"][name] == sha((out / name).read_bytes())
    drift = (out / "o5-drift.csv").read_text()
    assert drift.splitlines()[0].startswith("# Drive ")  # the same banner as every table
    assert "not produced: no entry writes it: the drive has no step drift-lgbm_tail" in drift
    points = (out / "o4-points.csv").read_text()
    assert "lgbm_tail,u1," in points
    assert "O4: the run's point values" in (out / "outcomes.md").read_text()
    # red: a new table whose cells start with "failed:", "not run:" or "unreadable:" would make
    # `any_unreadable` true and this exit status 1.


# --- the manifest's code_commit -------------------------------------------------------------------


def test_the_manifest_names_the_commit_of_the_code_beside_the_commit_of_the_root(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        wp16_run, "head_commit", lambda root: CODE if Path(root) == wp16_run.TOP else COMMIT
    )
    run_top(tmp_path)
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 0  # nothing is refused
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    assert manifest["commit"] == COMMIT and manifest["code_commit"] == CODE
    shown = capsys.readouterr().out
    assert CODE in shown and COMMIT in shown and "the code that ran is at commit" in shown
    # red: `code_commit = head_commit(args.root)` gives COMMIT here; no key gives a KeyError.


def test_one_commit_is_recorded_twice_and_no_difference_is_announced(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)
    run_top(tmp_path)
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    assert manifest["commit"] == manifest["code_commit"] == COMMIT
    assert "the code that ran is at commit" not in capsys.readouterr().out


def test_a_checkout_whose_head_cannot_be_read_is_named_not_refused(tmp_path, monkeypatch):
    def head(root):
        if Path(root) == wp16_run.TOP:
            raise NoDrive("git rev-parse HEAD failed: not a git repository")
        return COMMIT

    monkeypatch.setattr(wp16_run, "head_commit", head)
    run_top(tmp_path)
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    assert manifest["code_commit"].startswith("unknown: git rev-parse HEAD failed")
    assert manifest["commit"] == COMMIT


# --- the manifest's code_dirty --------------------------------------------------------------------


def fake_git(monkeypatch, *, output=("", "", 0), error=None):
    """Replace `subprocess` inside `wp16_run` by a stand-in whose `run` records the command and
    returns (stdout, stderr, status), or raises `error`."""
    seen: list = []

    def run(command, **kwargs):
        seen.append(command)
        if error is not None:
            raise error
        stdout, stderr, status = output
        return subprocess.CompletedProcess(command, status, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(wp16_run, "subprocess", SimpleNamespace(run=run))
    return seen


def test_code_dirty_is_true_when_git_lists_a_change_and_false_when_it_lists_none(monkeypatch):
    seen = fake_git(monkeypatch, output=(" M src/fleetreserve/wp16_o5.py\n", "", 0))
    assert REAL_CODE_DIRTY() is True
    assert seen == [["git", "-C", str(wp16_run.TOP), "status", "--porcelain"]]
    fake_git(monkeypatch, output=("?? new_file.py\n", "", 0))
    assert REAL_CODE_DIRTY() is True  # an untracked file is a change too
    fake_git(monkeypatch, output=("", "", 0))
    assert REAL_CODE_DIRTY() is False
    # red: reading `args.root` instead of `TOP` changes the command asserted above; `return True`
    # or `return False` alone turns one of the three cases red.


def test_code_dirty_names_a_git_that_cannot_run_or_fails_and_never_raises(monkeypatch):
    fake_git(monkeypatch, error=FileNotFoundError(2, "No such file or directory: 'git'"))
    missing = REAL_CODE_DIRTY()
    assert isinstance(missing, str)
    assert missing.startswith("unknown: git status --porcelain could not run: ")
    fake_git(monkeypatch, output=("", "fatal: not a git repository\n", 128))
    failed = REAL_CODE_DIRTY()
    assert failed == "unknown: git status --porcelain failed: fatal: not a git repository"
    # red: no `except OSError` raises the FileNotFoundError here; no check of `returncode` reads
    # the empty output of the failed git as "clean" (False).


def test_a_missing_git_is_named_in_head_commit_and_code_head_not_raised(monkeypatch):
    fake_git(monkeypatch, error=FileNotFoundError(2, "No such file or directory: 'git'"))
    with pytest.raises(NoDrive, match="git rev-parse HEAD could not run"):
        wp16_run.head_commit(wp16_run.TOP)
    assert wp16_run.code_head().startswith("unknown: git rev-parse HEAD could not run")
    # red: no `except OSError` in `head_commit` lets the FileNotFoundError out of both calls.


def test_the_manifest_records_code_dirty_and_a_dirty_checkout_is_announced_not_refused(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)
    monkeypatch.setattr(wp16_run, "code_dirty", lambda: True)
    run_top(tmp_path)
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 0  # nothing is refused
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    assert manifest["code_dirty"] is True
    assert "has uncommitted changes" in capsys.readouterr().out
    # red: no "code_dirty" key gives a KeyError; no print gives the empty assertion above red.


@pytest.mark.parametrize("found", [False, "unknown: git status --porcelain failed: made up"])
def test_a_clean_or_unknown_checkout_is_recorded_without_a_line(
    tmp_path, monkeypatch, capsys, found
):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)
    monkeypatch.setattr(wp16_run, "code_dirty", lambda: found)
    run_top(tmp_path)
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    assert json.loads((out / "wp16-manifest.json").read_text())["code_dirty"] == found
    assert "uncommitted" not in capsys.readouterr().out
    # red: a print for every value (`if dirty:` is true for the unknown text) turns the second red.


def test_the_o2_day_counts_are_written_named_in_the_manifest_and_move_no_exit_status(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)
    run_top(tmp_path)  # no O2 step in this drive
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    for name in ("o2-day-counts.csv", "o2-day-counts.md"):
        assert manifest["outputs"][name] == sha((out / name).read_bytes())
    csv = (out / "o2-day-counts.csv").read_text()
    assert "main setting,all units,no entry writes it: the drive has no step o2-rho0" in csv
    assert "O2: the days meeting T4's hypotheses" in (out / "outcomes.md").read_text()
    # red: no `build_day_counts` call in `collect` leaves the manifest without the file (KeyError).
