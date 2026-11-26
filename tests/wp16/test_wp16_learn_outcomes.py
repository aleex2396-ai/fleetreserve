"""O1 (X, Y, Z, D), O4 and O10 from made-up `learn.json`, `day-intervals.json`, `learn-gate.json`.

Red when a unit with no figure is added to X, when an interval is recomputed instead of read, when a
mixed-configuration run is given a pooled interval, when a pass mark is read at the wrong end of an
interval, or when a missing figure becomes a number.
"""

import pytest
from wp16_made_up import (
    NAME_MAIN,
    NAME_OTHER,
    UNITS,
    MadeUp,
    intervals_document,
    learn_document,
    matrix_document,
    matrix_same,
    pooled_block,
    run_list_document,
    sha,
)

from fleetreserve import wp16_o4, wp16_o10
from fleetreserve.wp16_cells import Interval, Missing
from fleetreserve.wp16_drive import Source, find_drive
from fleetreserve.wp16_learn import Runs
from fleetreserve.wp16_o1 import SPECS, Spec, ratio_of_sums, summary_row, unit_rows

SAME = {"u1": NAME_MAIN, "u2": NAME_MAIN, "u3": NAME_MAIN}
FIGURES = {
    "offer": (2.0, 1.0, 3.0),
    "availability": (0.93, 0.90, 0.95),
    "shortfall": (0.03, 0.01, 0.04),
}


def drive_with(tmp_path, matrix: dict, offers: dict, headline, intervals, extra=()):
    """A complete made-up drive with `learn-lgbm_tail` (offers by unit), its intervals and a
    learn-gate file; returns (source, runs)."""
    made = MadeUp(tmp_path)
    data = made.commit_file("gate/sealed-matrix/lgbm_tail.json", matrix)
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.commands["learn-lgbm_tail"] = ["made-up", "--matrix", "gate/sealed-matrix/lgbm_tail.json"]
    gate = {
        "configurations": {
            name: {
                "units": {
                    u: {"fixed_every_request": {"test_mean_offer_kwh_per_day": 0.25 * k}}
                    for k, u in enumerate(UNITS, start=1)
                }
            }
            for name in (NAME_MAIN, NAME_OTHER)
        }
    }
    steps = [
        (
            "learn-lgbm_tail",
            {"learn.json": learn_document(sha(data), offers, headline), "learn-gate.json": gate},
        ),
        ("intervals-lgbm_tail", {"day-intervals.json": intervals}),
        *extra,
    ]
    made.drive(steps)
    source = Source(tmp_path, find_drive(tmp_path))
    return source, Runs(source)


def test_a_pooled_interval_is_read_and_a_failed_unit_is_left_out(tmp_path):
    offers = {"u1": (1.0, 2.0), "u2": (1.0, 3.0), "u3": "failed: ValueError: boom"}
    intervals = intervals_document(
        NAME_MAIN, {"u1": FIGURES, "u2": FIGURES}, pooled_block(["u1", "u2"])
    )
    source, runs = drive_with(tmp_path, matrix_same(), offers, SAME, intervals)
    spec = SPECS[0]
    row, note = summary_row(source, runs, spec, UNITS, {})
    assert row[3] == "2 of 3"
    assert row[4] == Interval(7.0, 5.0, 9.0)  # the pooled block's, not the sum of 2.0 and 3.0
    assert row[5] == Interval(2.3, 1.7, 3.0)
    assert row[6] == Interval(1.5, 1.0, 2.0)
    assert row[7] == Interval(0.41, 0.3, 0.5)
    assert row[8] == Interval(0.51, 0.4, 0.6)
    assert row[9] == 0.25 + 0.5  # every-request offer D over the two units with a figure
    assert "u3: failed: ValueError: boom" in row[10]
    assert "pooled block" in note


def test_a_run_with_two_headline_configurations_has_no_interval_of_the_sum(tmp_path):
    offers = {"u1": (1.0, 2.0), "u2": (1.0, 3.0), "u3": (1.0, 4.0)}
    intervals = intervals_document(NAME_MAIN, {"u1": FIGURES}, pooled_block(["u1", "u2", "u3"]))
    source, runs = drive_with(tmp_path, matrix_document(), offers, None, intervals)
    row, note = summary_row(source, runs, SPECS[0], UNITS, {})
    assert row[1] == f"{NAME_MAIN}, {NAME_OTHER}"
    assert row[4] == Interval(9.0)  # the sum of the units' own points, no interval
    assert row[5].point == pytest.approx(3.0)  # (2 + 3 + 4) * 5 days over 15 days
    assert row[6].point == pytest.approx(0.5)  # per car, by reading I14 (a)
    assert row[7].point == pytest.approx(0.4)
    assert "differs by site type" in note


def test_a_unit_the_truth_build_gave_no_window_adds_nothing_and_says_why(tmp_path):
    offers = {"u1": (1.0, 2.0), "u2": (1.0, 3.0)}
    source, runs = drive_with(tmp_path, matrix_same(), offers, SAME, {"configurations": {}})
    absent = {"u3": "a window of zero hours"}
    row, _ = summary_row(source, runs, SPECS[0], UNITS, absent)
    assert row[3] == "2 of 3"
    assert row[4] == Interval(5.0)
    assert "u3: no figure: no window (a window of zero hours); counts as not passing" in row[10]
    rows = unit_rows(runs, SPECS[0], UNITS, absent)
    assert isinstance(rows[2][3], Missing)


def test_a_run_that_is_not_in_the_drive_is_a_missing_figure_not_a_number(tmp_path):
    source, runs = drive_with(tmp_path, matrix_same(), {"u1": (1.0, 2.0)}, SAME, {})
    spec = Spec("Cox, fixed plan", "learn-fixed-plan-cox", None, "fixed", "main")
    row, _ = summary_row(source, runs, spec, UNITS, {})
    assert isinstance(row[4], Missing)
    assert "no entry writes it" in row[4].text


def test_the_ratio_of_sums_is_the_sum_of_offers_over_the_sum_of_cars_per_day():
    tests = {
        "a": {"mean_offer_kwh_per_day": 2.0, "offer_kwh_per_car_per_day": 0.5},
        "b": {"mean_offer_kwh_per_day": 1.0, "offer_kwh_per_car_per_day": 0.25},
    }

    def get(test):
        return test["offer_kwh_per_car_per_day"]

    assert ratio_of_sums(tests, get, "per-car") == pytest.approx(3.0 / (4.0 + 4.0))
    tests["b"]["mean_offer_kwh_per_day"] = 0.0
    assert isinstance(ratio_of_sums(tests, get, "per-car"), Missing)


@pytest.mark.parametrize(
    ("availability", "shortfall", "verdict"),
    [
        (Interval(0.93, 0.88, 0.90), Interval(0.03, 0.05, 0.06), "yes"),  # both ends on the mark
        (Interval(0.93, 0.88, 0.8999), Interval(0.03, 0.01, 0.06), "no"),
        (Interval(0.93, 0.88, 0.95), Interval(0.03, 0.0501, 0.06), "no"),
        (Interval(0.93), Interval(0.03, 0.01, 0.06), "not judged"),
        (Missing("no figure: the run's scale is 0"), Missing("x"), "not judged"),
    ],
)
def test_the_pass_marks_are_read_at_the_ends_the_plan_names(availability, shortfall, verdict):
    assert wp16_o4.judge(availability, shortfall)[2] == verdict


def test_o4_counts_a_failed_unit_as_not_passing(tmp_path):
    offers = {"u1": (1.0, 2.0), "u2": (1.0, 3.0), "u3": "failed: ValueError: boom"}
    weak = {**FIGURES, "availability": (0.8, 0.7, 0.85)}
    intervals = intervals_document(NAME_MAIN, {"u1": FIGURES, "u2": weak})
    source, runs = drive_with(tmp_path, matrix_same(), offers, SAME, intervals)
    (table, points, both), figures = wp16_o4.build(runs, UNITS, {})
    rows = {row[1]: row for row in table.rows if row[0] == "lgbm_tail"}
    assert [rows[u][7] for u in UNITS] == ["yes", "no", "no"]
    row = next(r for r in both.rows if r[0] == "lgbm_tail")
    assert row[1:4] == (3, 1, 3)  # units, passing, judged (a unit with no figure counts as judged)
    assert row[6] == 2  # not passing
    assert rows["u3"][10].startswith("failed: ValueError: boom")
    assert figures[0]["model"] == "lgbm_tail"


def test_o10_change_is_a_percent_of_the_baseline_and_never_a_guess():
    assert wp16_o10.change(3.0, 2.0) == pytest.approx(50.0)
    assert isinstance(wp16_o10.change(3.0, 0.0), Missing)
    assert isinstance(wp16_o10.change(Missing("failed: x (y)"), 1.0), Missing)


def test_o10_reads_the_models_and_the_baselines_from_their_own_runs(tmp_path):
    made = MadeUp(tmp_path)
    data = made.commit_file("gate/sealed-matrix/lgbm_tail.json", matrix_same())
    made.commit_file("run-lists/run-list.json", run_list_document())
    command = ["made-up", "--matrix", "gate/sealed-matrix/lgbm_tail.json"]
    for label in ("learn-lgbm_tail", "learn-fixed-plan-cox"):
        made.commands[label] = command
    mine = learn_document(sha(data), {u: (3.0, 3.0) for u in UNITS}, SAME)
    theirs = learn_document(sha(data), {u: (1.0, 2.0) for u in UNITS}, SAME)
    made.drive(
        [
            ("learn-lgbm_tail", {"learn.json": mine}),
            ("learn-fixed-plan-cox", {"learn.json": theirs}),
        ]
    )
    source = Source(tmp_path, find_drive(tmp_path))
    (table,), figures = wp16_o10.build(source, Runs(source), UNITS, {})
    row = next(r for r in table.rows if r[0] == "lgbm_tail" and r[1] == "Cox")
    assert (row[3], row[4]) == (9.0, 3.0)  # fixed-plan offers: three units of 3.0 and of 1.0
    assert row[5] == 200.0
