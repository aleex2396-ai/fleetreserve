"""The parts of the corner-search command (`corner_run`) on made-up inputs, without a whole run of
the command (fast; the whole command is run by `test_corner_command`).

1. The preparation jobs give `compare_run._prepare_task` each day in order with no car drawn: the
   task, fed these jobs, calls `gate_day.prepare_day` with `no_pause=()` (the search is of the
   whole fleet).
2. The day order (reading K5): a seeded permutation, pinned on two inputs.
3. The path-hours by hand: the calibration counts the audited rows of the offering calibration
   days at every scale for every method; the search's upper bound every corner of every searched
   day for every method; the pass mark every corner of the checked days at every scale.
4. `search`: every method of the registry with a scale above 0 is searched on every searched day,
   with the result of `corner_search.search_day`, in the registry's order; a method at scale 0 is
   "no offer" on every day, with no path run.
5. The timeouts by hand: per variant (the oracle variant's keys begin with `oracle_`), the
   certificates, the calibration's and the search's re-solves and the controllers' solves; a
   calibration without timeouts adds none, and a method without solver counts in the calibration
   is passed over.
6. The calibration (`calibrate`) gives `compare_run.measure_variant` the calibration days alone,
   in order, with no best-causal ceiling, and the pool's tasks of `compare_run`, and scores each
   method by `dial_splits.score_split` on all of them as calibration days, tier A. One unit and
   variant (`measure_variant`, at given scales) on made-up days: each method searched at its
   scale, or "no offer" at 0; the entry holds the method figures of
   `corner_summary.method_entry`, the pass-mark check of the first `CHECK_DAYS` searched days,
   the searched days, the offering calibration days and the calibration's counts; the search's
   path-hours (methods with a scale above 0) are counted, and `counted` called, before the
   search.
"""

import dataclasses
from collections import Counter
from types import SimpleNamespace

from corner_fake import HIDDEN, SINGLE, offer_days

from fleetreserve import (
    compare_methods,
    compare_run,
    corner_run,
    corner_search,
    corner_summary,
    dial_splits,
    gate_day,
    gate_run,
)


class ListPool:
    """The pool of `compare_run.Pool` with one worker, without its state."""

    def map(self, task, jobs):
        return [task(job) for job in jobs]


# --- 1. the preparation jobs ---------------------------------------------------------------------


def test_the_preparation_jobs_draw_no_car(monkeypatch):
    truths = ["t0", "t1", "t2"]
    monkeypatch.setattr(
        compare_run, "_STATE", {"units": {"u": SimpleNamespace(days=truths)}, "matrix": "m"}
    )
    monkeypatch.setattr(compare_run, "_forecaster", lambda unit: f"forecaster of {unit}")
    monkeypatch.setattr(gate_run, "settings", lambda matrix: f"settings of {matrix}")

    def prepare(truth, forecaster, configuration, settings, no_pause="not given"):
        return (truth, forecaster, configuration, settings, no_pause)

    monkeypatch.setattr(gate_day, "prepare_day", prepare)
    jobs = corner_run.prepare_jobs("u", 3, "configuration")
    assert [compare_run._prepare_task(job) for job in jobs] == [
        (t, "forecaster of u", "configuration", "settings of m", ()) for t in truths
    ]
    assert corner_run.prepare_jobs("u", 0, "configuration") == []


# --- 2. the day order ----------------------------------------------------------------------------


def test_the_day_order_is_a_seeded_permutation():
    days = list(range(3, 40, 3))
    first = corner_run.day_order("unit", 1, days)
    assert first == [12, 30, 33, 21, 9, 6, 27, 39, 3, 18, 15, 36, 24]
    assert corner_run.day_order("nrel", 7, range(10)) == [9, 8, 4, 6, 7, 5, 0, 1, 3, 2]
    assert corner_run.day_order("unit", 2, days) != first
    assert corner_run.day_order("other", 1, days) != first


# --- 3. the path-hours ---------------------------------------------------------------------------


def _prepared(n, abstains=False):
    return SimpleNamespace(abstains=abstains, day=SimpleNamespace(n_service=n))


def test_the_path_hours_by_hand():
    prepared = [_prepared(2), _prepared(5, True), _prepared(1), _prepared(2), _prepared(3)]
    hours = corner_run.count_path_hours(prepared, [0, 1, 2], [3, 4], 7, 1)
    grid = len(dial_splits.GRID)
    assert corner_run.AUDITED_ROWS == 8  # 3 fixed rows and 5 seeded (audit_paths)
    assert hours == {
        "calibration": (2 + 1) * 8 * grid * 7,  # day 1 abstains
        "search_upper_bound": (4 * 2 + 8 * 3) * 7,
        "pass_mark": 4 * 2 * grid,  # the first searched day only
    }
    assert corner_run.count_path_hours(prepared, [], [], 7, 20)["pass_mark"] == 0


# --- 4. the search -------------------------------------------------------------------------------

METHODS = compare_methods.select(None)


def test_every_method_with_a_scale_is_searched_on_every_day():
    prepared = {4: SINGLE, 9: HIDDEN}
    scales = {m.name: (1.0 if k % 3 else 0.0, "") for k, m in enumerate(METHODS)}
    found = corner_run.search(prepared, [9, 4], METHODS, scales, ListPool())
    assert list(found) == [m.name for m in METHODS]
    for method in METHODS:
        days = found[method.name]
        if scales[method.name][0] == 0:
            assert days == [corner_search.no_offer(method.name, 0.0)] * 2
            continue
        assert days == [corner_search.search_day(prepared[d], method, 1.0) for d in (9, 4)]
        assert all(d.status != "no offer" and d.n_paths == 4 for d in days)


def test_no_method_of_the_registry_is_left_out_of_the_search():
    scales = {m.name: (0.5, "") for m in METHODS}
    found = corner_run.search({0: SINGLE}, [0], METHODS, scales, ListPool())
    assert set(found) == {m.name for m in compare_methods.METHODS}
    assert {name: days[0].n_paths for name, days in found.items()} == dict.fromkeys(found, 4)


# --- 5. the timeouts -----------------------------------------------------------------------------


def _entry(calibration_timeouts, solves):
    rung = {"optimal": 9}
    if calibration_timeouts is not None:
        rung["timeout"] = calibration_timeouts
    return {
        "calibration_evaluations": {
            "re_solves_of_the_one_way_rung": rung,
            "controller_solver_outcomes": {
                "a": {"solves": {"timeout": 2, "optimal": 5}},
                "b": {"solves": {"optimal": 1}},
                "d": None,  # a method without solver counts in the calibration
            },
        },
        "methods": {
            "re-solve": {"resolves": {"timeout": 3, "optimal": 1}, "solver_counts": None},
            "a": {"resolves": {}, "solver_counts": {"solves": {"timeout": solves}}},
            "b": {"resolves": {}, "solver_counts": {"solves": {"optimal": 4}}},
            "c": {"status": "no offer"},
        },
    }


def test_the_timeouts_by_hand():
    prepared = [SimpleNamespace(certificate_outcome=o) for o in ("timeout", "optimal", "timeout")]
    timeouts = Counter()
    corner_run._count_timeouts(timeouts, "forecast", prepared, _entry(4, 6))
    corner_run._count_timeouts(timeouts, "oracle-departure", prepared[:1], _entry(1, 0))
    # a calibration whose one-way re-solves have no timeout adds none
    corner_run._count_timeouts(timeouts, "forecast", [], _entry(None, 0))
    assert dict(timeouts) == {
        "certificates": 2,
        "re_solves_calibration": 4,
        "re_solves_search": 6,
        "controller_solves_calibration": 4,
        "controller_solves_search": 6,
        "oracle_certificates": 1,
        "oracle_re_solves_calibration": 1,
        "oracle_re_solves_search": 3,
        "oracle_controller_solves_calibration": 2,
        "oracle_controller_solves_search": 0,
    }


# --- 6. one unit and variant ---------------------------------------------------------------------

DAYS = offer_days(6, seed=11)


def test_the_calibration_scores_each_method_on_the_calibration_days_alone(monkeypatch):
    methods = compare_methods.select(["fixed-plan", "llf"])
    seen = {}

    def measure(days, best_causal, given, evaluate_batch, compare_batch):
        seen["days"], seen["best_causal"], seen["methods"] = days, best_causal, given
        seen["evaluate"] = evaluate_batch(["e1", "e2"])
        seen["compare"] = compare_batch(["c1"])
        return {m.name: f"losses of {m.name}" for m in given}, {"made": "up"}

    def score(losses, seed, calibration, test, tier):
        seen.setdefault("scores", []).append((losses, seed, list(calibration), test, tier))
        return SimpleNamespace(scale=0.3 if losses.endswith("llf") else 0.0, reason="why")

    monkeypatch.setattr(compare_run, "measure_variant", measure)
    monkeypatch.setattr(dial_splits, "score_split", score)

    class RecordingPool:
        def map(self, task, jobs):
            return [(task, job) for job in jobs]

    scales, counts = corner_run.calibrate(DAYS, [4, 0, 2], methods, RecordingPool())
    assert seen["days"] == [DAYS[4], DAYS[0], DAYS[2]] and seen["best_causal"] == [None] * 3
    assert seen["methods"] == methods
    task = compare_run._evaluate_task
    assert seen["evaluate"] == [(task, "e1"), (task, "e2")]
    assert seen["compare"] == [(compare_run._compare_task, "c1")]
    assert seen["scores"] == [
        ("losses of fixed-plan", 0, [0, 1, 2], [], "A"),
        ("losses of llf", 0, [0, 1, 2], [], "A"),
    ]
    assert scales == {"fixed-plan": (0.0, "why"), "llf": (0.3, "why")}
    assert counts == {"made": "up"}


def test_one_variant_searches_at_the_calibrated_scales(monkeypatch):
    methods = compare_methods.select(["fixed-plan", "llf", "diagonal-split"])
    calibration, searched = [0, 2, 4], [5, 1, 3]
    chosen = {"fixed-plan": (0.6, ""), "llf": (0.0, "too few"), "diagonal-split": (1.0, "")}
    counts = {"controller_day_scale_pairs": 30}
    monkeypatch.setattr(corner_run, "calibrate", lambda *args: (chosen, counts))
    monkeypatch.setattr(corner_run, "CHECK_DAYS", 2)
    order = []
    original = corner_run.search

    def search(*args):
        order.append("search")
        return original(*args)

    monkeypatch.setattr(corner_run, "search", search)
    hours = {}
    entry, by_method, scales = corner_run.measure_variant(
        DAYS, calibration, searched, methods, ListPool(), hours, lambda: order.append("counted")
    )
    assert order == ["counted", "search"] and scales == chosen
    for m in methods:
        scale = chosen[m.name][0]
        expected = [
            corner_search.search_day(DAYS[d], m, scale)
            if scale > 0
            else corner_search.no_offer(m.name, scale)
            for d in searched
        ]
        assert by_method[m.name] == expected
        assert entry["methods"][m.name] == corner_summary.method_entry(*chosen[m.name], expected)
    corners = sum(2 ** DAYS[d].day.n_service * DAYS[d].day.n_service for d in searched)
    assert hours == {"search": corners * 2}  # two methods with a scale above 0
    assert entry["searched_days"] == searched
    assert entry["offering_calibration_days"] == 3
    assert entry["calibration_evaluations"] is counts
    checks = [corner_search.closed_form_check(DAYS[d], dial_splits.GRID) for d in searched[:2]]
    assert entry["pass_mark"] == corner_summary.pass_mark_entry(
        searched[:2], [DAYS[d].date for d in searched[:2]], checks
    )
    abstaining = [dataclasses.replace(DAYS[0], abstain_reason="made up"), *DAYS[1:]]
    entry, _, _ = corner_run.measure_variant(
        abstaining, calibration, searched, methods, ListPool(), {}, lambda: None
    )
    assert entry["offering_calibration_days"] == 2  # day 0 abstains
