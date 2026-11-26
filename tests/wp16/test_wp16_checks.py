"""The refusals and readings added after the first review of WP16 piece 1 (made-up files).

Red when: the beside reading is read as the main one; a matrix that is not the run's, or a
`learn.json` with no matrix hash, is accepted; a run list that is not the one the drive ran is
read; an interval file of another seed or number of resamples is read; a unit is called built
when the truth summary cannot be read; tier B is not tabled; a configuration that failed on a
unit leaves the exit status 0 and the banner saying every step passed.
"""

import json
from types import SimpleNamespace

import pytest
from wp16_made_up import (
    COMMIT,
    NAME_BESIDE,
    NAME_MAIN,
    NAME_OTHER,
    UNITS,
    MadeUp,
    intervals_document,
    learn_document,
    matrix_document,
    matrix_same,
    run_list_document,
    sha,
)

from fleetreserve import wp16_failed, wp16_figures, wp16_mixed, wp16_o1, wp16_o4, wp16_run
from fleetreserve.wp16_cells import Interval, Missing
from fleetreserve.wp16_drive import Missed, Source, find_drive
from fleetreserve.wp16_learn import Runs, names_for
from fleetreserve.wp16_o1 import SPECS, summary_row
from fleetreserve.wp16_units import abstained, beside_name

SAME = {u: NAME_MAIN for u in UNITS}
FIGURES = {
    "offer": (2.0, 1.0, 3.0),
    "availability": (0.93, 0.90, 0.95),
    "shortfall": (0.03, 0.01, 0.04),
}


def build_top(tmp_path, offers, *, beside=None, matrix_sha=None, truth=None, intervals=None):
    """A complete made-up drive with the run list, the matrix, the truth build and `learn-lgbm_tail`
    (and `intervals-lgbm_tail` when `intervals` is given); returns (source, runs)."""
    made = MadeUp(tmp_path)
    data = made.commit_file("gate/sealed-matrix/lgbm_tail.json", matrix_same())
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.commands["learn-lgbm_tail"] = ["made-up", "--matrix", "gate/sealed-matrix/lgbm_tail.json"]
    learn = learn_document(sha(data), offers, SAME, beside)
    if matrix_sha == "none":
        del learn["matrix"]
    elif matrix_sha is not None:
        learn["matrix"]["sha256"] = matrix_sha
    steps = []
    if truth is not None:
        steps.append(("sealed-truth", {"truth-summary.json": truth}))
    steps.append(("learn-lgbm_tail", {"learn.json": learn}))
    if intervals is not None:
        steps.append(("intervals-lgbm_tail", {"day-intervals.json": intervals}))
    made.drive(steps)
    source = Source(tmp_path, find_drive(tmp_path))
    return source, Runs(source)


def test_the_beside_reading_is_read_from_the_beside_configuration(tmp_path):
    offers = {u: (1.0, 2.0) for u in UNITS}
    source, runs = build_top(tmp_path, offers, beside=5.0)
    run = runs.get("learn-lgbm_tail")
    assert beside_name(matrix_same(), NAME_MAIN) == NAME_BESIDE
    assert set(names_for(run, UNITS, "beside").values()) == {NAME_BESIDE}
    assert set(names_for(run, UNITS, "main").values()) == {NAME_MAIN}
    beside = next(s for s in SPECS if s.reading == "beside")
    row, _ = summary_row(source, runs, beside, UNITS, {})
    assert row[1] == NAME_BESIDE
    assert row[4].point == 15.0  # three units of 5.0, not the main reading's 3 * 2.0
    main, _ = summary_row(source, runs, SPECS[0], UNITS, {})
    assert main[4].point == 6.0


def test_a_matrix_that_is_not_the_runs_is_refused(tmp_path):
    _, runs = build_top(tmp_path, {u: (1.0, 2.0) for u in UNITS}, matrix_sha="0" * 64)
    with pytest.raises(Missed, match="is not the matrix the run"):
        runs.get("learn-lgbm_tail")


def test_a_learn_file_with_no_matrix_hash_is_refused_not_skipped(tmp_path):
    _, runs = build_top(tmp_path, {u: (1.0, 2.0) for u in UNITS}, matrix_sha="none")
    with pytest.raises(Missed, match="records no SHA-256 of the matrix"):
        runs.get("learn-lgbm_tail")


def test_a_run_list_that_is_not_the_one_the_drive_ran_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)
    source, _ = build_top(tmp_path, {u: (1.0, 2.0) for u in UNITS})
    (tmp_path / "run-lists" / "run-list.json").write_text(json.dumps(run_list_document(["x"])))
    with pytest.raises(Missed, match="not the run list the drive ran"):
        wp16_run.nine(source)
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")]) == 2


def test_an_interval_file_of_another_seed_or_resamples_is_not_read(tmp_path):
    for field, value in (("seed", 1), ("resamples", 500)):
        top = tmp_path / field
        top.mkdir()
        intervals = intervals_document(NAME_MAIN, {"u1": FIGURES})
        intervals[field] = value
        _, runs = build_top(top, {u: (1.0, 2.0) for u in UNITS}, intervals=intervals)
        run = runs.get("learn-lgbm_tail", "intervals-lgbm_tail")
        assert run.intervals is None
        assert run.intervals_why.startswith("unreadable:")
        assert "not the run list's 20261122 and 2000" in run.intervals_why


def test_abstained_units_are_read_and_an_unreadable_summary_is_never_empty(tmp_path):
    truth = {"abstained": {"u3": {"reason": "no window"}}, "units": {}}
    top = tmp_path / "with"
    top.mkdir()
    source, _ = build_top(top, {"u1": (1.0, 2.0)}, truth=truth)
    assert abstained(source) == {"u3": "no window"}
    other = tmp_path / "without"
    other.mkdir()
    source, _ = build_top(other, {"u1": (1.0, 2.0)})  # no truth step in the drive
    with pytest.raises(Missed, match="no entry writes it"):
        abstained(source)


def test_no_unit_is_called_built_when_the_truth_summary_cannot_be_read(tmp_path):
    table = wp16_run.units_table(UNITS, {}, "unreadable: sealed-truth/truth-summary.json: x")
    assert [row[1] for row in table.rows] == ["unknown"] * 3
    assert all(isinstance(row[2], Missing) for row in table.rows)
    built = wp16_run.units_table(UNITS, {"u3": "no window"})
    assert [row[1] for row in built.rows] == ["built", "built", "none"]


def test_o4_tables_tier_b_and_its_two_points(tmp_path):
    offers = {"u1": (1.0, 2.0), "u2": (1.0, 3.0), "u3": (1.0, 4.0)}
    intervals = intervals_document(NAME_MAIN, {u: FIGURES for u in UNITS})
    _, runs = build_top(tmp_path, offers, intervals=intervals)
    (table, points, _both), _ = wp16_o4.build(runs, UNITS, {})
    row = next(r for r in table.rows if r[0] == "lgbm_tail" and r[1] == "u1")
    assert row[8] == Interval(0.93, 0.90, 0.95)  # tier B availability, from the `B` cell
    assert row[9] == Interval(0.03, 0.01, 0.04)
    mine = [r for r in points.rows if r[0] == "lgbm_tail" and r[1] == "u1"]
    by_dial = {r[3]: r for r in mine}
    assert set(by_dial) == {"none", 0.02}  # two points, no curve
    assert by_dial[0.02][4] == pytest.approx(2.2)  # the one-way offer 2.0, tier B 1.1 times it
    assert by_dial["none"][4] == pytest.approx(4.4)
    assert by_dial[0.02][5] == 0.01
    assert "Two points, not a curve" in points.notes[0]


def test_a_configuration_that_failed_on_a_unit_is_named_and_the_exit_is_1(tmp_path, monkeypatch):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)
    monkeypatch.setattr(wp16_figures, "render", lambda spec, folder: _placeholder(spec, folder))
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    offers = {"u1": (1.0, 2.0), "u2": (1.0, 3.0), "u3": "failed: ValueError: boom"}
    truth = {"abstained": {}, "units": {}}
    build_top(tmp_path, offers, truth=truth)
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 1
    line = f"learn-lgbm_tail: {NAME_MAIN} | u3: ValueError: boom"
    page = (out / "outcomes.md").read_text()
    assert "every step exited 0, but 1 configuration(s) failed" in page and line in page
    assert "complete; every step ran and passed" not in page
    first = (out / "o1-summary.csv").read_text().splitlines()[0]
    assert first.startswith("# Drive ") and line in first  # the CSV carries the same banner
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    assert manifest["drive"]["complete"] is True
    assert manifest["drive"]["failed_configurations"] == [line]
    assert "u3,built," in (out / "units.csv").read_text().splitlines()


def _placeholder(spec, folder):
    name = f"{spec['kind']}.png"
    (folder / name).write_bytes(b"placeholder")
    return name


def test_o3_and_o10_do_not_draw_a_missing_value_as_a_bar(tmp_path):
    plt = wp16_figures._plt()
    points = [
        ("u1", 40.0, 50.0, 44.0, None),
        ("u2", 30.0, 35.0, 31.0, 60.0),
        ("u3", 0.0, None, None, None),  # a real 0 is drawn
    ]
    figure, _ = wp16_figures._o3(plt, {"kind": "o3", "points": points})
    assert len(figure.axes[0].patches) == 8  # 8 values; the 4 missing ones have no bar
    plt.close(figure)
    rows = [("lgbm_tail", "Cox", 12.5), ("lgbm_tail", "earlier_gbm", None)]
    figure, _ = wp16_figures._o10(plt, {"kind": "o10", "rows": rows})
    assert len(figure.axes[0].patches) == 1
    plt.close(figure)


def test_o4_prints_the_pass_mark_both_ways_for_a_unit_at_scale_0(tmp_path):
    offers = {"u1": (1.0, 2.0), "u2": (1.0, 3.0), "u3": (1.0, 4.0)}
    intervals = intervals_document(NAME_MAIN, {"u1": FIGURES, "u2": FIGURES})  # no entry for u3
    made = MadeUp(tmp_path)
    data = made.commit_file("gate/sealed-matrix/lgbm_tail.json", matrix_same())
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.commands["learn-lgbm_tail"] = ["made-up", "--matrix", "gate/sealed-matrix/lgbm_tail.json"]
    learn = learn_document(sha(data), offers, SAME)
    learn["units"]["u2"]["configurations"][NAME_MAIN]["rungs"]["one-way"]["beta_hat"] = 0
    cell = intervals["configurations"][NAME_MAIN]["u2"]["one-way"]["A"]
    nothing = {
        "point": None,
        "interval": None,
        "reason": "the run's scale is 0: nothing is offered",
    }
    cell["availability"], cell["shortfall"] = nothing, nothing
    made.drive(
        [
            ("learn-lgbm_tail", {"learn.json": learn}),
            ("intervals-lgbm_tail", {"day-intervals.json": intervals}),
        ]
    )
    source = Source(tmp_path, find_drive(tmp_path))
    (table, _points, both), _ = wp16_o4.build(Runs(source), UNITS, {})
    verdicts = {r[1]: r[7] for r in table.rows if r[0] == "lgbm_tail"}
    assert verdicts == {
        "u1": "yes",
        "u2": "not judged: scale 0",
        "u3": "not judged: no interval",
    }
    row = next(r for r in both.rows if r[0] == "lgbm_tail")
    # units, A passing, A judged, at scale 0, no interval, judged not passing, B passing, B judged
    assert row[1:] == (3, 1, 1, 1, 1, 0, 2, 2)
    other = next(r for r in both.rows if r[0] == "hazard_tuned_inputs")  # no such run: none passes
    assert other[1:] == (3, 0, 3, 0, 0, 3, 0, 3)
    assert "Reading A" in both.notes[0] and "Reading B" in both.notes[0]
    why = {r[1]: r[10] for r in table.rows if r[0] == "lgbm_tail"}
    assert why["u2"] == "scale 0"  # the reasons are asserted, not only the verdicts
    assert why["u3"] == "no interval was written for this unit"


def test_a_scale_0_unit_with_interval_ends_is_still_counted_as_scale_0(tmp_path):
    offers = {"u1": (1.0, 2.0), "u2": (1.0, 3.0), "u3": (1.0, 4.0)}
    intervals = intervals_document(NAME_MAIN, {u: FIGURES for u in UNITS})  # u2 has ends
    made = MadeUp(tmp_path)
    data = made.commit_file("gate/sealed-matrix/lgbm_tail.json", matrix_same())
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.commands["learn-lgbm_tail"] = ["made-up", "--matrix", "gate/sealed-matrix/lgbm_tail.json"]
    learn = learn_document(sha(data), offers, SAME)
    learn["units"]["u2"]["configurations"][NAME_MAIN]["rungs"]["one-way"]["beta_hat"] = 0
    made.drive(
        [
            ("learn-lgbm_tail", {"learn.json": learn}),
            ("intervals-lgbm_tail", {"day-intervals.json": intervals}),
        ]
    )
    source = Source(tmp_path, find_drive(tmp_path))
    (table, _points, both), _ = wp16_o4.build(Runs(source), UNITS, {})
    verdicts = {r[1]: r[7] for r in table.rows if r[0] == "lgbm_tail"}
    assert verdicts == {"u1": "yes", "u2": "not judged: scale 0", "u3": "yes"}
    row = next(r for r in both.rows if r[0] == "lgbm_tail")
    assert row[1:] == (3, 2, 2, 1, 0, 0, 3, 3)  # A: 2 of 2 judged; B: scale 0 judged and passing


def test_o1_tables_the_rho_runs_from_their_own_learn_and_interval_entries(tmp_path):
    made = MadeUp(tmp_path)
    data = made.commit_file("gate/sealed-matrix/lgbm_tail.json", matrix_same())
    made.commit_file("run-lists/run-list.json", run_list_document())
    steps = []
    for label, intervals_label, one_way in (
        ("learn-lgbm_tail", "intervals-lgbm_tail", 2.0),
        ("learn-lgbm_tail-rho0_1", "intervals-lgbm_tail-rho0_1", 3.0),
        ("learn-lgbm_tail-rho0_2", "intervals-lgbm_tail-rho0_2", 4.0),
    ):
        made.commands[label] = ["made-up", "--matrix", "gate/sealed-matrix/lgbm_tail.json"]
        learn = learn_document(sha(data), {u: (1.0, one_way) for u in UNITS}, SAME)
        steps.append((label, {"learn.json": learn}))
        steps.append((intervals_label, {"day-intervals.json": intervals_document(NAME_MAIN, {})}))
    made.drive(steps)
    source = Source(tmp_path, find_drive(tmp_path))
    summary, per_unit, rho = wp16_o1.build(source, Runs(source), UNITS, {})
    assert rho.name == "o1-rho"
    assert [r[0] for r in rho.rows] == [
        "lgbm_tail (main model), one-way re-solve",
        "lgbm_tail, rho 0.1 (WP15 sensitivity)",
        "lgbm_tail, rho 0.2 (WP15 sensitivity)",
    ]
    assert [r[4].point for r in rho.rows] == [6.0, 9.0, 12.0]  # X, the sum over three units
    assert all(r[3] == "3 of 3" for r in rho.rows)
    titles = {r[0] for r in summary.rows}
    assert "lgbm_tail, rho 0.1 (WP15 sensitivity)" in titles
    assert "lgbm_tail, rho 0.2 (WP15 sensitivity)" in {r[0] for r in per_unit.rows}


def test_no_output_is_a_figure_pooled_over_units(tmp_path, monkeypatch):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)
    monkeypatch.setattr(wp16_figures, "render", lambda spec, folder: _placeholder(spec, folder))
    build_top(tmp_path, {u: (1.0, 2.0) for u in UNITS}, truth={"abstained": {}, "units": {}})
    out = tmp_path / "out"
    wp16_run.main(["--root", str(tmp_path), "--out", str(out)])
    names = json.loads((out / "wp16-manifest.json").read_text())["outputs"]
    assert not any("total" in name or "pooled" in name for name in names)
    assert "o7-corner-units.csv" in names and "o7-corner-total.csv" not in names


HAZARD = "learn-hazard_tuned_inputs"


class FakeCouldNotRun(Exception):
    pass


def hazard_top(tmp_path, offers, skip=(), gate=None, extra_jobs=()):
    """A drive with the neural model's learn step (headlines differ by site type: the matrix's
    default) and made-up per-day jobs of both configurations on every unit."""
    made = MadeUp(tmp_path)
    data = made.commit_file("gate/sealed-matrix/hazard_tuned_inputs.json", matrix_document())
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.commands[HAZARD] = ["made-up", "--matrix", "gate/sealed-matrix/hazard_tuned_inputs.json"]
    losses = {
        "rungs": ["fixed", "one-way"],
        "jobs": [
            {"unit": u, "configurations": [name]}
            for u in UNITS
            for name in (NAME_MAIN, NAME_OTHER)
            if (u, name) not in skip
        ]
        + list(extra_jobs),
    }
    files = {
        "learn.json": learn_document(sha(data), offers),
        "learn-gate.json": gate or {"configurations": {}},
        "learn-day-losses.json": losses,
    }
    made.drive([(HAZARD, files)])
    source = Source(tmp_path, find_drive(tmp_path))
    return source, Runs(source)


def fake_day_intervals(calls, raises=None, not_evaluated=()):
    def from_record(record, rungs):
        return SimpleNamespace(
            unit=record["unit"],
            configurations=tuple(record["configurations"]),
            calibration_days=record.get("cal", ()),
            test_days=record.get("test", ()),
        )

    def check_resampled(job, rungs, seed, resamples):
        if raises:
            raise FakeCouldNotRun(raises)
        calls.setdefault("resampled", []).append((job.unit, job.configurations, seed, resamples))

    def pooled(chosen, rungs, left):
        calls["pooled"] = {u: found[0].configurations for u, found in chosen.items()}
        calls["left"] = [entry["unit"] for entry in left]
        one = {"offer_kwh_per_counting_test_day": {"point": 9.0, "interval": [7.0, 11.0]}}
        return {"rungs": {"one-way": {"A": {"sum_of_unit_offers": one}}}}

    bootstrap = SimpleNamespace(Job=SimpleNamespace(from_record=from_record))
    intervals = SimpleNamespace(
        CouldNotRun=FakeCouldNotRun,
        check_resampled=check_resampled,
        _entry=lambda document, name, unit: (name, unit),
        job_points=lambda job, entry, rungs, name: ({"p": name}, []),
        job_intervals=lambda job, points, rungs, seed, resamples: ({}, {"d": job.unit}),
        not_evaluated=lambda job, found: list(not_evaluated),
        pooled=pooled,
    )
    return bootstrap, intervals


def test_mixed_headlines_pool_each_units_own_configuration_with_the_run_lists_seed(
    tmp_path, monkeypatch
):
    """The numerics are day_intervals' own (not run here); this pins which job, seed and
    resamples are handed to them, and what is left out."""
    offers = {"u1": (1.0, 2.0), "u2": (1.0, 3.0), "u3": "failed: ValueError: boom"}
    source, runs = hazard_top(tmp_path, offers)
    calls: dict = {}
    bootstrap, intervals = fake_day_intervals(calls)
    monkeypatch.setattr(wp16_mixed, "day_bootstrap", bootstrap)
    monkeypatch.setattr(wp16_mixed, "day_intervals", intervals)
    spec = next(s for s in SPECS if s.learn == HAZARD)
    row, note = summary_row(source, runs, spec, UNITS, {})
    assert row[4] == Interval(9.0, 7.0, 11.0)  # the pooled block's point and ends, read as given
    assert calls["pooled"] == {"u1": (NAME_MAIN,), "u2": (NAME_OTHER,)}  # each unit's own headline
    assert calls["left"] == ["u3"]  # the failed unit is left out and named
    assert {c[2:] for c in calls["resampled"]} == {(20261122, 2000)}
    assert "its own headline configuration" in note and "seed 20261122, 2000 resamples" in note
    assert row[4].why == ""  # an interval, so no reason is needed


def test_mixed_headlines_keep_no_interval_with_the_reason_when_the_per_day_file_fails(
    tmp_path, monkeypatch
):
    source, runs = hazard_top(tmp_path, {"u1": (1.0, 2.0), "u2": (1.0, 3.0), "u3": (1.0, 4.0)})
    bootstrap, intervals = fake_day_intervals({}, raises="the run evaluated no pairs")
    monkeypatch.setattr(wp16_mixed, "day_bootstrap", bootstrap)
    monkeypatch.setattr(wp16_mixed, "day_intervals", intervals)
    spec = next(s for s in SPECS if s.learn == HAZARD)
    row, note = summary_row(source, runs, spec, UNITS, {})
    assert row[4] == Interval(9.0)  # the sum of the points, no interval
    assert "computing it from the per-day file failed" in note
    assert "the run evaluated no pairs" in note
    assert "the run evaluated no pairs" in row[4].why  # the cell says why, in the CSV too


def test_an_unreadable_learn_file_is_named_in_the_banner_and_the_exit_is_1(tmp_path, monkeypatch):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)
    monkeypatch.setattr(wp16_figures, "render", lambda spec, folder: _placeholder(spec, folder))
    build_top(tmp_path, {u: (1.0, 2.0) for u in UNITS}, truth={"abstained": {}, "units": {}})
    (folder,) = (tmp_path / "runs.noindex").glob("*-learn-lgbm_tail")
    (folder / "learn.json").write_text("{}")  # no longer the file the run wrote
    source = Source(tmp_path, find_drive(tmp_path))
    failed, unreadable = wp16_failed.failed_configurations(source)
    assert failed == [] and len(unreadable) == 1
    assert unreadable[0].startswith("learn-lgbm_tail: unreadable:")
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 1
    page = (out / "outcomes.md").read_text()
    assert "1 learn result(s) cannot be read" in page and "learn-lgbm_tail: unreadable:" in page
    assert "complete; every step ran and passed" not in page
    assert (out / "o1-summary.csv").read_text().splitlines()[0].startswith("# Drive ")
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    assert len(manifest["drive"]["unreadable_learn_results"]) == 1


def mixed_row(tmp_path, monkeypatch, *, calls=None, raises=None, not_evaluated=(), **top):
    offers = {"u1": (1.0, 2.0), "u2": (1.0, 3.0), "u3": (1.0, 4.0)}
    source, runs = hazard_top(tmp_path, offers, **top)
    bootstrap, intervals = fake_day_intervals({} if calls is None else calls, raises, not_evaluated)
    monkeypatch.setattr(wp16_mixed, "day_bootstrap", bootstrap)
    monkeypatch.setattr(wp16_mixed, "day_intervals", intervals)
    spec = next(s for s in SPECS if s.learn == HAZARD)
    return summary_row(source, runs, spec, UNITS, {})


def test_a_mixed_sum_over_fewer_units_than_have_a_figure_gets_no_interval(tmp_path, monkeypatch):
    calls: dict = {}
    row, note = mixed_row(tmp_path, monkeypatch, calls=calls, skip={("u2", NAME_OTHER)})
    assert set(calls["pooled"]) == {"u1", "u3"}  # u2 has a figure but no per-day job
    assert row[3] == "3 of 3"
    assert row[4] == Interval(9.0)  # the sum of the three points, not the block's point
    assert "missing from the pooled block: u2" in row[4].why
    assert "missing from the pooled block: u2" in note


def test_a_unit_that_compute_would_refuse_gives_no_interval_here_either(tmp_path, monkeypatch):
    row, _ = mixed_row(tmp_path, monkeypatch, not_evaluated=["u1: 3 resamples met a pair"])
    assert row[4] == Interval(9.0) and "met a pair" in row[4].why
    split = [{"unit": "u1", "configurations": ["x"], "cal": [1], "test": [2]}]
    split.append({"unit": "u1", "configurations": ["y"], "cal": [3], "test": [4]})
    row, _ = mixed_row(tmp_path / "s", monkeypatch, extra_jobs=split)
    assert row[4] == Interval(9.0) and "two different splits" in row[4].why
    gate = {"configurations": {"other": {"units": {"u1": {"rungs": {}}}}}}
    row, _ = mixed_row(tmp_path / "g", monkeypatch, gate=gate)
    assert row[4] == Interval(9.0) and "no per-day record for other on u1" in row[4].why
