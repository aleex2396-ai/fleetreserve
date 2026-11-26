"""The "information only" mark of the WP15 parts (plan section 13, WP15: after a held-out run that
WP14 fails, the parts readable from steps that passed are printed as information only), decided
once by `wp16_failed.information_only` and applied to every WP15 part: the rho 0.1 and 0.2 rows of
`o1-summary`, `o1-units` and `o1-rho`, and `o5-drift` (made-up files in the writers' forms; nothing
is run on real data).

Three ways WP14 fails, each alone: a step exited 1 (`o3`), a configuration failed on a unit while
every step exited 0, and a learn step that passed whose `learn.json` cannot be read.

Red when: a WP15 row is left unmarked after any of the three; a row that is not a WP15 part (the
rho 0 headline row) is marked; a number cell, a column or a note other than the one note changes
when the mark is on; anything at all changes when the run passed WP14; or `o5-drift` and the o1
tables disagree about the same run.
"""

import pytest
from wp16_made_up import (
    NAME_MAIN,
    UNITS,
    MadeUp,
    intervals_document,
    learn_document,
    matrix_same,
    run_list_document,
    sha,
)

from fleetreserve import wp16_o1, wp16_o5
from fleetreserve.wp16_drive import Source, find_drive
from fleetreserve.wp16_failed import SUFFIX, information_only
from fleetreserve.wp16_learn import Runs

SAME = {u: NAME_MAIN for u in UNITS}
MATRIX = "gate/sealed-matrix/lgbm_tail.json"
HEADLINE = "lgbm_tail (main model), one-way re-solve"
RHO_ONE = "lgbm_tail, rho 0.1 (WP15 sensitivity)"
RHO_TWO = "lgbm_tail, rho 0.2 (WP15 sensitivity)"
RUNS = (
    ("learn-lgbm_tail", "intervals-lgbm_tail", 2.0),
    ("learn-lgbm_tail-rho0_1", "intervals-lgbm_tail-rho0_1", 3.0),
    ("learn-lgbm_tail-rho0_2", "intervals-lgbm_tail-rho0_2", 4.0),
)
SCENARIOS = {
    "a step failed": {"extra": [("o3", "fail")]},
    "a configuration failed on a unit": {"failed_unit": True},
    "a learn result cannot be read": {"extra": [("learn-earlier_gbm", {"other.json": {}})]},
}


def drift_document() -> dict:
    """`drift-run.json`: one unit, tier A only (tier B has no path)."""
    slot = {"guarded": True, "alpha": 0.1, "mean": 0.09, "final_bound": 0.12}
    driver = {"guarded": True, "alpha": 0.05, "mean": 0.04, "final_bound": 0.06}
    slot["days_over_bound"] = driver["days_over_bound"] = 0
    found = {
        "offering_days": 50,
        "mean_offer_kwh_per_counting_day": 1.5,
        "losses": {"slot": slot, "driver": driver},
    }
    entry = {"status": "run", "paths": {"A": {"calibrated": {"0.5": found}}}}
    return {"units": {"u1": entry}}


def drive_of(top, *, extra=(), failed_unit=False):
    """A made-up drive: the main model's learn and interval steps at rho 0, 0.1 and 0.2, the drift
    step and `extra`; with `failed_unit` the rho 0.2 run holds a configuration that failed on u2
    (its step still exits 0). Returns (source, runs)."""
    made = MadeUp(top)
    data = made.commit_file(MATRIX, matrix_same())
    made.commit_file("run-lists/run-list.json", run_list_document())
    steps = []
    for label, intervals_label, one_way in RUNS:
        made.commands[label] = ["made-up", "--matrix", MATRIX]
        offers = {u: (1.0, one_way) for u in UNITS}
        if failed_unit and label.endswith("rho0_2"):
            offers["u2"] = "failed: ValueError: boom"
        steps.append((label, {"learn.json": learn_document(sha(data), offers, SAME)}))
        steps.append((intervals_label, {"day-intervals.json": intervals_document(NAME_MAIN, {})}))
    steps.append(("drift-lgbm_tail", {"drift-run.json": drift_document()}))
    made.drive(steps + list(extra))
    source = Source(top, find_drive(top))
    return source, Runs(source)


def marked(rows) -> list:
    return [row[0] for row in rows if isinstance(row[0], str) and row[0].endswith(SUFFIX)]


def test_a_run_that_passed_wp14_marks_no_wp15_part_and_adds_no_note(tmp_path):
    source, runs = drive_of(tmp_path)
    assert source.drive.complete and not information_only(source)
    summary, per_unit, rho = wp16_o1.build(source, runs, UNITS, {})
    (drift,) = wp16_o5.build_drift(source)
    for table in (summary, per_unit, rho):
        assert marked(table.rows) == []
    for table in (summary, per_unit, rho, drift):
        assert not any(note.startswith("Information only") for note in table.notes)
    assert {row[-1] for row in drift.rows} == {"no"}
    # red: a mark that is on whatever the run did (`return True` in `information_only`).


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_wp15_part_is_marked_by_the_one_decision_when_wp14_fails(tmp_path, scenario):
    source, runs = drive_of(tmp_path, **SCENARIOS[scenario])
    assert information_only(source)
    summary, per_unit, rho = wp16_o1.build(source, runs, UNITS, {})
    (drift,) = wp16_o5.build_drift(source)
    assert marked(summary.rows) == [RHO_ONE + SUFFIX, RHO_TWO + SUFFIX]
    assert marked(per_unit.rows) == [RHO_ONE + SUFFIX] * 3 + [RHO_TWO + SUFFIX] * 3
    assert [row[0] for row in rho.rows] == [HEADLINE, RHO_ONE + SUFFIX, RHO_TWO + SUFFIX]
    assert HEADLINE in [row[0] for row in summary.rows]  # the headline is not a WP15 part
    assert {row[-1] for row in drift.rows} == {"yes"}  # o5-drift reads the same decision
    for table in (summary, per_unit, rho, drift):
        notes = [note for note in table.notes if note.startswith("Information only")]
        assert len(notes) == 1 and "WP15's pass mark is not passed" in notes[0]
    # red: a part that decides for itself (`drive.complete` alone leaves the unit case unmarked,
    # `failed_configurations` alone the step case, a missing `unreadable` the third case); a
    # `_marked` call dropped from `build` leaves its table unmarked; a `wp15=True` on the rho 0 spec
    # or none on a rho spec changes the marked rows asserted above.


def test_the_mark_changes_no_number_cell_no_column_and_adds_one_note(tmp_path):
    ok_source, ok_runs = drive_of(tmp_path / "ok")
    info_source, info_runs = drive_of(tmp_path / "info", extra=[("o3", "fail")])
    assert not information_only(ok_source) and information_only(info_source)
    before = wp16_o1.build(ok_source, ok_runs, UNITS, {})
    after = wp16_o1.build(info_source, info_runs, UNITS, {})
    for old, new in zip(before, after, strict=True):
        assert (old.name, old.title, old.columns) == (new.name, new.title, new.columns)
        assert len(old.rows) == len(new.rows) > 0
        for was, now in zip(old.rows, new.rows, strict=True):
            assert now[1:] == was[1:]  # every cell but the first is the same
            assert now[0] == (was[0] + SUFFIX if was[0] in (RHO_ONE, RHO_TWO) else was[0])
        assert new.notes[:-1] == old.notes  # the same notes, then one more
        assert new.notes[-1].startswith("Information only")
    # red: a column added to the table, a number cell edited, or a note changed or a second one
    # added turns this red; so does a first cell changed on a row that is not a WP15 row.
