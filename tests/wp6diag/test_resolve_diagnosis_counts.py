"""What a row and the summary say about early leavers, timeouts and solver errors
(`resolve_diagnosis_day`, `resolve_diagnosis_summary`), on the hand day of
`test_resolve_diagnosis_day` (two cars, one of which truly leaves in hour 1).

1. The early-leaver count needs both `d <= hour` and `d < d_hat`, with `d_hat` the forecast
   departure of the day's forecasts: a car that leaves at or after its forecast departure is not
   early, however gone it is; the bound table the re-solve used does not enter.
2. A re-solve that times out is a row with the outcome "timeout" and is not diagnosed.
3. The programmes that end in a solver error, and the first attempts that did, are counted in the
   row, and the totals reach the summary.
4. The text summary carries the definitions of the box scale (including "none"), of the
   early-leaver count (forecast departure at the decision time, every steered car, cumulative
   along a path) and of the retry; so does the results document's `definitions`.
"""

import dataclasses

import numpy as np
import pytest
from test_resolve_diagnosis_day import GONE_EARLY, _hand_prepared, _row

from fleetreserve import resolve_diagnosis
from fleetreserve import resolve_diagnosis_day as diagnosis
from fleetreserve import resolve_diagnosis_lp as relaxed
from fleetreserve import resolve_diagnosis_summary as summary
from fleetreserve.day import ForecastCar, TrueCar
from fleetreserve.lp import LpResult
from fleetreserve.resolve import ResolveResult
from fleetreserve.sparse_programme import Sizes


def _with_cars(prepared, cars, forecasts=None):
    day = dataclasses.replace(GONE_EARLY, cars=cars, forecasts=forecasts or GONE_EARLY.forecasts)
    return dataclasses.replace(prepared, day=day)


# --- 1. the early-leaver count ------------------------------------------------------------------


def test_a_car_that_leaves_at_or_after_its_forecast_departure_is_not_early():
    prepared = _hand_prepared()
    (found,) = diagnosis.collect(prepared, 1.0)[:1]
    stays = TrueCar(5, 1.0, 3.0)
    for departure, forecast in ((2, 2.0), (2, 1.0), (1, 1.0)):
        cars = (TrueCar(departure, 1.0, 1.0), stays)
        forecasts = (ForecastCar(forecast, 1.0, 1.0, 1.0), GONE_EARLY.forecasts[1])
        other = _with_cars(prepared, cars, forecasts)
        assert diagnosis.hindsight(other, found.runs, 2) == (0, 0.0), (departure, forecast)
    early = _with_cars(prepared, (TrueCar(2, 1.0, 1.0), stays))  # forecast 3: early
    assert diagnosis.hindsight(early, found.runs, 2)[0] == 1


def test_the_count_reads_the_forecast_departure_not_the_bound_the_re_solve_used():
    prepared = _hand_prepared()
    (found, _) = diagnosis.collect(prepared, 1.0)
    assert diagnosis.hindsight(prepared, found.runs, 1) == (1, pytest.approx(1.0))
    for value in (0.0, 9.0):
        table = np.full_like(prepared.departure_bounds, value)
        other = dataclasses.replace(prepared, departure_bounds=table)
        assert diagnosis.hindsight(other, found.runs, 1) == (1, pytest.approx(1.0))


# --- 2. a re-solve that times out ----------------------------------------------------------------


def test_a_re_solve_that_times_out_is_a_row_and_is_not_diagnosed(monkeypatch):
    real = diagnosis.resolve

    def timing_out(state, guard, *, two_way, budget, time_limit):
        if state.hour == 1:
            return ResolveResult("timeout", None, time_limit, 0.0, Sizes(0, 0, 0), "limit")
        return real(state, guard, two_way=two_way, budget=budget, time_limit=time_limit)

    monkeypatch.setattr(diagnosis, "resolve", timing_out)
    diagnosed = []

    def diagnose(state, guard, **kwargs):
        diagnosed.append(state.hour)
        return relaxed.diagnose(state, guard, **kwargs)

    rows = diagnosis.rows(_hand_prepared(), 1.0, diagnose=diagnose)
    timed = [r for r in rows if r["hour"] == 1]
    assert len(timed) == 2 and {r["outcome"] for r in timed} == {"timeout"}
    for row in timed:
        assert all(row[c] == "" for c in diagnosis.INFEASIBLE_ONLY)
    infeasible = [r for r in rows if r["outcome"] == "infeasible"]
    assert infeasible and diagnosed == [r["hour"] for r in infeasible]
    assert 1 not in diagnosed


# --- 3. solver errors ----------------------------------------------------------------------------


def _failing_diagnose(first_errors, second_errors):
    """A `diagnose` whose solves end in an error at the first attempt (all, or none) and whose
    retry ends in an error or is the real simplex solve."""

    def diagnose(state, guard, *, two_way, budget, time_limit):
        def first(programme, columns, values, limit):
            if first_errors:
                return LpResult("error", None, None, 0.0, limit, 4, "made up")
            return relaxed.maximise(programme, columns, values, limit, "ipm")

        def second(programme, columns, values, limit):
            if second_errors:
                return LpResult("error", None, None, 0.0, limit, 4, "made up")
            return relaxed.maximise(programme, columns, values, limit, "simplex")

        return relaxed.diagnose(
            state,
            guard,
            two_way=two_way,
            budget=budget,
            time_limit=time_limit,
            solve=relaxed.retrying(first, second),
        )

    return diagnose


def test_the_programmes_that_stay_in_error_and_the_first_errors_are_counted_in_the_row():
    rows = diagnosis.rows(_hand_prepared(), 1.0, diagnose=_failing_diagnose(True, True))
    row = _row(rows, 0, 1)
    assert row["outcome"] == "infeasible"  # the rung's own re-solve is not touched
    assert row["relaxed_errors"] == 12 and row["relaxed_timeouts"] == 0
    assert (row["solver_first_errors"], row["solver_retried_answered"]) == (12, 0)
    assert row["cause"] == "not determined"


def test_the_retries_that_answer_are_counted_and_leave_the_answers_as_they_are():
    plain = diagnosis.rows(_hand_prepared(), 1.0)
    rows = diagnosis.rows(_hand_prepared(), 1.0, diagnose=_failing_diagnose(True, False))
    row = _row(rows, 0, 1)
    assert row["relaxed_errors"] == 0
    assert (row["solver_first_errors"], row["solver_retried_answered"]) == (12, 12)
    assert row["cause"] == "tracking|power"
    assert float(row["box_scale"]) == pytest.approx(float(_row(plain, 0, 1)["box_scale"]), abs=1e-7)
    assert all(r["solver_first_errors"] == 0 for r in plain if r["outcome"] == "infeasible")
    assert all(
        r["solver_first_errors"] == r["solver_retried_answered"] == ""
        for r in plain
        if r["outcome"] != "infeasible"
    )


def test_the_summary_adds_up_the_solver_errors():
    rows = diagnosis.rows(_hand_prepared(), 1.0, diagnose=_failing_diagnose(True, False))
    head = {"unit": "u", "configuration": "c", "site": "s", "date": "d", "scale": "1.0"}
    found = summary.summarise([{**head, **r} for r in rows])
    bad = sum(r["outcome"] == "infeasible" for r in rows)
    assert found["infeasible"]["solver_first_errors"] == 12 * bad
    assert found["infeasible"]["solver_retried_answered"] == 12 * bad
    assert found["infeasible"]["relaxed_errors"] == 0
    text = "\n".join(summary.summary_lines(_document(found)))
    assert (
        f"first attempt in a solver error: {12 * bad}, of which the simplex retry answered" in text
    )


# --- 4. the definitions in the output -----------------------------------------------------------


def _document(found):
    return {
        "label": summary.LABEL,
        "run": {"configuration": "c", "rung": "one-way", "scales": ["1.0"]},
        "units": {
            "u": {
                "counting_days": 1,
                "offering_days": 1,
                "paths_per_day": 2,
                "scales": {"1.0": found},
            }
        },
        "timeouts": {"certificates": 0, "re_solves": 0, "relaxed_programmes": 0},
    }


def test_the_summary_and_the_results_carry_the_definitions():
    rows = diagnosis.rows(_hand_prepared(), 1.0)
    head = {"unit": "u", "configuration": "c", "site": "s", "date": "d", "scale": "1.0"}
    found = summary.summarise([{**head, **r} for r in rows])
    text = "\n".join(summary.summary_lines(_document(found)))
    for key, definition in summary.DEFINITIONS.items():
        assert definition in text, key
    assert (
        "none" in summary.DEFINITIONS["box_scale"] and "s = 0" in summary.DEFINITIONS["box_scale"]
    )
    early = summary.DEFINITIONS["early_leaver"]
    for phrase in (
        "decision time",
        "every steered car",
        "already seen gone",
        "cumulative along a path",
        "neither the bound the re-solve used nor the declared time",
    ):
        assert phrase in early, phrase
    assert "never retried and never read as infeasible" in summary.DEFINITIONS["solver_errors"]
    document = resolve_diagnosis.definitions()
    for key, definition in summary.DEFINITIONS.items():
        assert document[key] == definition
