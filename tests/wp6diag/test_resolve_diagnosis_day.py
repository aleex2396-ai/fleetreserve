"""The re-solves of a day as the diagnosis collects and measures them (`resolve_diagnosis_day`).

1. The measurement watches the real rung. On seeded made-up days (`made_up_offer_days`, both
   readings of when a departure is seen, scales 0.5 and 1), the re-solves collected per path
   equal, hour by hour and outcome by outcome, the one-way rung's re-solves in the gate's own
   replay of the day (`gate_day.replay`), and their counts by outcome equal what the gate's
   evaluation reports (`gate_day.evaluate`). What each collected re-solve was given (the state,
   the guard, one-way, the budget) equals, entry for entry, what the gate's replay gave its own
   re-solver for the one-way rung of that path (read by putting a recorder in the place of the
   rungs' default re-solver while the gate replays; per path the one-way rung's calls come before
   the two-way rung's). The sample holds `N_RESOLVES` re-solves, optimal and infeasible ones, and
   re-solves whose departure bound at that hour differs from the bound at the decision time. An
   abstaining day and a scale of 0 give nothing.
2. A hand day where a car has truly gone early (`GONE_EARLY`). Two service hours, three recovery
   hours, rho 0, box centre (1, 0.8), half-widths (0.5, 0.5), scale 1, departures seen from the
   rounded-down hour.
   - Car A: truly leaves in hour 1 (it draws in hour 0 only), forecast to leave at 3; rate 1.
     The fixed plan gives it (0, 0.5, 0.5).
   - Car B: leaves in hour 5 as forecast; rate 1, room 3, needs 1. The fixed plan gives it
     (0, 0.5 + 0.5 zeta_1, 0.3 + 0.5 zeta_2).
   At the start of hour 1 A is seen gone with 0.5 kW planned, so the fixed plan certainly fails
   and the rung re-solves for B alone.
   - Path (-1, -1): hour 1 asks 0.5, which B can do; hour 2 will ask 0.8 +- 0.5, up to 1.3
     against B's rate 1: infeasible. Feasible without the tracking of hour 2 or without the power
     limit, and by no other single removal; box scale 0.4 (`0.8 + 0.5 s <= 1`). The fixed plan
     stays, B draws its planned 0 in hour 1 and the hour fails. Hour 2 asks 0.3: the re-solve
     there is optimal.
   - Path (+1, +1): hour 1 asks 1.5 against B's rate 1, and hour 2 up to 1.3: infeasible, and
     only removing the power limit helps (the request of hour 1 and the tracking of hour 2 are
     each infeasible alone). Hour 2 asks 1.3: infeasible again; feasible without the request or
     without the power limit, request miss 0.3 kWh.
   Hindsight at every infeasible re-solve: one car (A) truly left before its forecast departure;
   the fixed plan still planned 1.0 kWh for it from hour 1 on (0.5 + 0.5) and 0.5 from hour 2 on.
3. A row of a re-solve that is not infeasible has no diagnosis; the state columns are those of
   the rung's run.
4. A relaxed programme that times out (injected) is counted in the row and is not read as
   infeasible.
"""

import dataclasses
import datetime
import functools
import random
from collections import Counter

import numpy as np
import pytest
from diag_fake import BUDGET, T5_TOLERANCE, prepared_of
from made_up_offer_days import offer_day

from fleetreserve import gate_day, products, rungs
from fleetreserve import resolve_diagnosis_day as diagnosis
from fleetreserve import resolve_diagnosis_lp as relaxed
from fleetreserve.certificate import Certificate
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.lp import OTHER_SECONDS, LpResult
from fleetreserve.policy import AffinePolicy
from fleetreserve.resolve import resolve
from fleetreserve.sparse_programme import Sizes

SEED = 31
N_DAYS = 10
BETAS = (0.5, 1.0)
READINGS = ("hour-start", "rounded-hour")
N_RESOLVES = 239
RUNG = "one-way"
TOLERANCE = 1e-7

GONE_EARLY = Day(
    "hand",
    datetime.date(2020, 1, 1),
    2,
    3,
    0.0,
    (TrueCar(1, 1.0, 1.0), TrueCar(5, 1.0, 3.0)),
    (ForecastCar(3.0, 1.0, 1.0, 1.0), ForecastCar(5.0, 1.0, 3.0, 1.0)),
    0.1,
)
GONE_EARLY_POLICY = AffinePolicy(
    [1.0, 0.8],
    [0.5, 0.5],
    [[0, 0.5, 0.5], [0, 0.5, 0.3]],
    [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0.5, 0], [0, 0.5]]],
)
PATHS = [(-1.0, -1.0), (1.0, 1.0)]


def _hand_prepared():
    certificate = Certificate(
        "optimal",
        products.hourly(),
        "causal",
        "made by hand",
        2.0,
        None,
        GONE_EARLY_POLICY,
        2.0,
        0.0,
        Sizes(0, 0, 0),
    )
    return prepared_of(GONE_EARLY, certificate, "rounded-hour", PATHS)


@functools.cache
def _sample():
    rng = random.Random(SEED)
    found = []
    for k in range(N_DAYS):
        day, certificate, _ = offer_day(rng, k)
        for seen in READINGS:
            found.append(prepared_of(day, certificate, seen))
    return found


# --- 1. the real rung ----------------------------------------------------------------------------


def _gate_replay(monkeypatch, prepared, beta):
    """(the gate's replay of the day, what it gave the rungs' re-solver, call by call)."""
    given = []

    def recorder(state, guard, *, two_way, budget, time_limit):
        given.append((state, guard, two_way, budget, time_limit))
        return resolve(state, guard, two_way=two_way, budget=budget, time_limit=time_limit)

    monkeypatch.setitem(rungs.run_rungs.__kwdefaults__, "resolver", recorder)
    found = gate_day.replay(prepared, beta)
    monkeypatch.undo()
    return found, given


def _same_inputs(attempt, given) -> bool:
    state, guard, two_way, budget, time_limit = given
    mine, mine_guard = attempt.state, attempt.guard
    arrays = [
        (mine.known, state.known),
        (mine.m, state.m),
        (mine.w, state.w),
        (mine.present, state.present),
        (mine.stored, state.stored),
        (mine.charge_bound, state.charge_bound),
        (mine_guard.plan.constant, guard.plan.constant),
        (mine_guard.plan.gains, guard.plan.gains),
        (mine_guard.stored, guard.stored),
    ]
    return (
        all(np.array_equal(a, b) for a, b in arrays)
        and np.array_equal(mine.departure, state.departure, equal_nan=True)
        and (mine.hour, mine.beta, mine.n_recovery) == (state.hour, state.beta, state.n_recovery)
        and mine.forecasts == state.forecasts
        and (mine_guard.hour, mine_guard.clip) == (guard.hour, guard.clip)
        and mine_guard.plan.two_way == guard.plan.two_way
        and (attempt.two_way, attempt.budget) == (two_way, budget)
        and time_limit == OTHER_SECONDS
    )


def test_the_re_solves_collected_are_those_of_the_gate_s_own_replay(monkeypatch):
    total, outcomes, later_bounds = 0, Counter(), 0
    for prepared in _sample():
        for beta in BETAS:
            collected = diagnosis.collect(prepared, beta)
            if prepared.abstains:
                assert collected == []
                continue
            replayed, given = _gate_replay(monkeypatch, prepared, beta)
            gate = replayed.by_rung[RUNG]
            assert len(collected) == len(gate) == prepared.paths.shape[0]
            start = 0
            for k, (mine, theirs) in enumerate(zip(collected, gate, strict=True)):
                assert [(a.state.hour, a.outcome) for a in mine.attempts] == [
                    (r.hour, r.outcome) for r in theirs.resolves
                ]
                np.testing.assert_array_equal(mine.runs[RUNG].run.final, theirs.final)
                one_way = given[start : start + len(mine.attempts)]
                assert len(one_way) == len(mine.attempts)
                assert all(_same_inputs(a, g) for a, g in zip(mine.attempts, one_way, strict=True))
                start += len(mine.attempts) + len(replayed.by_rung["two-way"][k].resolves)
                for a in mine.attempts:
                    here = prepared.departure_bounds[:, a.state.hour][a.state.present]
                    first = prepared.departure_bounds[:, 0][a.state.present]
                    later_bounds += int(np.any(here != first))
            assert start == len(given)
            counted = Counter(a.outcome for found in collected for a in found.attempts)
            assert (
                dict(sorted(counted.items()))
                == gate_day.evaluate(prepared, beta).rungs[RUNG].resolves
            )
            rows = diagnosis.rows(prepared, beta)
            assert Counter(r["outcome"] for r in rows) == counted
            total += len(rows)
            outcomes.update(counted)
    print(f"{total} re-solves of the one-way rung; outcomes {dict(outcomes)}")
    assert total == N_RESOLVES
    assert outcomes["optimal"] > 50 and outcomes["infeasible"] > 50
    assert later_bounds > 20


def test_an_abstaining_day_and_a_scale_of_zero_collect_nothing():
    prepared = _hand_prepared()
    assert diagnosis.collect(prepared, 0.0) == [] and diagnosis.rows(prepared, 0.0) == []
    abstaining = dataclasses.replace(prepared, abstain_reason="made to abstain")
    assert diagnosis.collect(abstaining, 1.0) == [] and diagnosis.rows(abstaining, 1.0) == []
    assert diagnosis.rows(prepared, 1.0) != []
    assert (prepared.budget, prepared.t5_tolerance_kwh) == (BUDGET, T5_TOLERANCE)


# --- 2. the hand day -----------------------------------------------------------------------------


def _row(rows, path_index, hour):
    (found,) = [r for r in rows if r["path_index"] == path_index and r["hour"] == hour]
    return found


def test_the_hand_day_s_re_solves_and_their_causes():
    rows = diagnosis.rows(_hand_prepared(), 1.0)
    assert [(r["path_index"], r["hour"], r["outcome"]) for r in rows] == [
        (0, 1, "infeasible"),
        (0, 2, "optimal"),
        (1, 1, "infeasible"),
        (1, 2, "infeasible"),
    ]
    assert [r["path"] for r in rows] == ["-1 -1", "-1 -1", "+1 +1", "+1 +1"]

    low = _row(rows, 0, 1)
    assert set(low["single"].split("|")) == {"tracking", "power"}
    assert low["cause"] == "tracking|power" and low["pairs"] == ""
    assert (low["recheck"], low["departure_relaxed"]) == ("infeasible", "infeasible")
    assert float(low["box_scale"]) == pytest.approx(0.4, abs=TOLERANCE)
    assert low["box_scale_outcome"] == "optimal"
    assert (low["need_slack_kwh"], low["need_slack_outcome"]) == ("", "infeasible")
    assert (low["request_miss_kwh"], low["request_miss_outcome"]) == ("", "infeasible")

    high = _row(rows, 1, 1)
    assert high["single"] == "power" and high["cause"] == "power"
    assert (high["box_scale"], high["box_scale_outcome"]) == ("", "infeasible")

    last = _row(rows, 1, 2)
    assert set(last["single"].split("|")) == {"request", "power"}
    assert float(last["request_miss_kwh"]) == pytest.approx(0.3, abs=TOLERANCE)


def test_the_hand_day_s_state_and_hindsight_columns():
    rows = diagnosis.rows(_hand_prepared(), 1.0)
    first, second = _row(rows, 0, 1), _row(rows, 0, 2)
    assert (first["cars"], first["present"], first["seen_gone"]) == (2, 1, 1)
    assert (first["fixed_failed_before"], first["one_way_failed_before"]) == (0, 0)
    assert (first["re_solves_before"], first["optimal_before"], first["guard_hour"]) == (0, 0, 1)
    assert (second["fixed_failed_before"], second["one_way_failed_before"]) == (1, 1)
    assert (second["re_solves_before"], second["optimal_before"], second["guard_hour"]) == (1, 0, 2)
    assert first["early_gone_cars"] == 1
    assert float(first["early_gone_expected_kwh"]) == pytest.approx(1.0, abs=TOLERANCE)
    late = _row(rows, 1, 2)
    assert late["early_gone_cars"] == 1
    assert float(late["early_gone_expected_kwh"]) == pytest.approx(0.5, abs=TOLERANCE)


def test_hindsight_counts_only_cars_gone_before_their_forecast_departure():
    prepared = _hand_prepared()
    (found, _) = diagnosis.collect(prepared, 1.0)
    assert diagnosis.hindsight(prepared, found.runs, 1) == (1, pytest.approx(1.0))
    # A leaving at its forecast departure (hour 3) is not early, and is not gone by hour 2.
    stays = dataclasses.replace(
        prepared,
        day=dataclasses.replace(GONE_EARLY, cars=(TrueCar(3, 1.0, 1.0), TrueCar(5, 1.0, 3.0))),
    )
    assert diagnosis.hindsight(stays, found.runs, 2) == (0, 0.0)
    # A leaving in hour 2, before its forecast 3: gone by hour 2 (0.5 kWh still planned), not by 1.
    early = dataclasses.replace(
        prepared,
        day=dataclasses.replace(GONE_EARLY, cars=(TrueCar(2, 1.0, 1.0), TrueCar(5, 1.0, 3.0))),
    )
    assert diagnosis.hindsight(early, found.runs, 1) == (0, 0.0)
    assert diagnosis.hindsight(early, found.runs, 2) == (1, pytest.approx(0.5))


# --- 3. rows that are not infeasible -------------------------------------------------------------


def test_only_an_infeasible_re_solve_is_diagnosed():
    seen = 0
    for prepared in _sample()[:6]:
        for row in diagnosis.rows(prepared, 1.0):
            assert tuple(row) == diagnosis.COLUMNS
            blank = all(row[c] == "" for c in diagnosis.INFEASIBLE_ONLY)
            assert blank == (row["outcome"] != "infeasible")
            if row["outcome"] == "infeasible":
                assert row["recheck"] == "infeasible"
                assert row["cause"] not in ("", "not determined")
                assert row["relaxed_timeouts"] == row["relaxed_errors"] == 0
                seen += 1
    assert seen > 10


# --- 4. a timed-out relaxed programme ------------------------------------------------------------


def test_a_timed_out_relaxed_programme_is_counted_in_the_row():
    limits = []

    def diagnose(state, guard, *, two_way, budget, time_limit):
        calls = []

        def solve(programme, columns, values, limit):
            calls.append(limit)
            if len(calls) == 2 + list(relaxed.GROUPS).index("power"):  # call 1: recheck
                return relaxed.solve_programme(programme, columns, values, limit)
            return LpResult("timeout", None, None, limit, limit, 1, "time limit reached")

        limits.append(time_limit)
        return relaxed.diagnose(
            state, guard, two_way=two_way, budget=budget, time_limit=time_limit, solve=solve
        )

    rows = diagnosis.rows(_hand_prepared(), 1.0, relaxed_time_limit=11.0, diagnose=diagnose)
    assert limits == [11.0, 11.0, 11.0]
    row = _row(rows, 0, 1)
    assert row["outcome"] == "infeasible"  # the rung's own re-solve is not touched
    assert row["single"] == "power" and row["cause"] == "power"
    assert row["relaxed_timeouts"] == 11 and row["relaxed_errors"] == 0
    assert (row["recheck"], row["departure_relaxed"]) == ("timeout", "timeout")
    assert (row["box_scale"], row["box_scale_outcome"]) == ("", "timeout")
