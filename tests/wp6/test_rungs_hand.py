"""Plan section 5 ("Rungs"): hand cases for when a rung re-solves and what it keeps.

Early-departure day (`EARLY`): two service hours, two recovery hours, rho = 0, box
`m = (1, 1)`, `w = (0.5, 0.5)`, scale 1.

- Car A: leaves in hour 2 (present in hours 0 and 1), rate 1 kW, room 1 kWh; forecast: leaves at
  3, rate 1, cap 1, need 1.
- Car B: leaves in hour 5, rate 2 kW, room 3 kWh; forecast: leaves at 5, rate 2, cap 3, need 3.
- Fixed plan (valid on the forecast sets, checked below): A draws (0, 0.5, 0.5); B draws
  (0, 0.5 + 0.5 zeta_1, 0.5 + 0.5 zeta_2).

Fixed plan on any path: hour 1 is met; in hour 2 A has left, so the fleet delivers
`0.5 + 0.5 zeta_2` against `1 + 0.5 zeta_2`: first failure in hour 2, slot loss 1/2. A ends at 0.5
(short 0.5 of 1); B, after two recovery hours at 2 kW, is full at 3. Driver loss 0.5 / 4 = 1/8.

One-way rung on `EARLY`, by the reading of when a departure is seen (`rungs.py`):

- "hour-start": A leaves within hour 2, so at the start of hour 2 it is still plugged in
  (`t <= d`) and the rung cannot know it is going. The fixed plan is not certain to fail, the rung
  copies it and fails in hour 2 too: slot loss 1/2, driver loss 1/8, no re-solve (hour 2 is the
  last service hour). The same holds for the two-way rung.
- "rounded-hour": A is absent from hour 2 (`t < d` fails), so at the start of hour 2 the rung sees
  it gone with 0.5 kW planned: the fixed plan certainly fails, and the rung re-solves hour 2 for B
  alone, so no hour fails: slot loss 0, driver loss 1/8, B full at 3, A at 0.5. The two-way rung
  copies it and never re-solves.

Gone day (`GONE`): as `EARLY`, but A leaves within hour 1 (`d = 1`) and plans 0.5 kW in hour 0 and
in hour 2 and nothing in hour 1 (`GONE_POLICY`: B covers hour 1 with `1 + 0.5 zeta_1`). The fixed
plan meets hour 1 and fails in hour 2 (A has left). At the start of hour 2 A is seen gone with
0.5 kW planned, so the fixed plan certainly fails; the one-way rung re-solves hour 2 for B alone
(`y_2` in [0.5, 1.5], within B's forecast rate 2 and its room), so no hour fails: slot loss 0,
driver loss 1/8 (A's 0.5 is lost either way), B full at 3, A at 0.5. The two-way rung copies it.
Under both readings: A is gone by the start of hour 2 either way.

Decisions from observables only: `EARLY` and the same day with A leaving within hour 3 instead
(`EARLY_STAYS`) look the same under "hour-start" at the start of hours 0 to 2 (the same draws in
hours 0 and 1, A plugged in at the start of hour 2), so every rung takes the same decisions in
hours 0 to 2. Under "rounded-hour" the two days differ at the start of hour 2 (A absent on
`EARLY`), and the one-way rung's decision in hour 2 differs: the two readings are not the same.

The cases from `ZERO` on do not depend on the reading (no car leaves within an hour in which a
decision turns on it) and run under both.

The rung engine's own discharge rule (`ZERO`): the plan's T5 unit case placed in the rung code.
Three service hours, no recovery, rho = 0, box `m = (1, 1, -0.5)`, `w = 0`. Car Z: leaves in hour
4, rate 1, room 1 (forecast room 2). Car X: leaves in hour 4, true room 0 (forecast room 1). Fixed
plan: X (0.5, 0.5, 0, 0), Z (0, 0.5, 1, -0.5). In hour 0 X takes nothing of its 0.5 kW, so no
later charge of X can exceed 0; at the start of hour 1 the fixed plan is therefore certain to fail
(X has 0.5 kW planned), before any failure. The fixed plan fails in hour 1; Z ends at 1.0. A
made-up re-solver answers the first two-way re-solve (hour 1) with the plan Z (0, 1, 1, -0.5) and
every other re-solve with "infeasible". Two-way rung: hour 1 met (Z takes 1, full); hour 2 Z can
take nothing (first failure); hour 3 the re-solve is infeasible, so the predecessor's plan stays
(plan section 5): the rung follows the one-way rung's plan, here the fixed plan, run one-way, and
Z's planned -0.5 draws nothing: Z ends at 1.0, at least the one-way rung's. The zeroing rule of
plan section 5 is not reached by the rung engine: switched off, Z still ends at 1.0 (the rule is
exercised on a two-way plan run alone in `test_replay_hand.py`).
"""

import dataclasses
import datetime

import numpy as np
import pytest

from fleetreserve import replay
from fleetreserve.carsets import day_forecast_rows
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.plans import Plan
from fleetreserve.policy import AffinePolicy
from fleetreserve.resolve import ResolveResult
from fleetreserve.rungs import DEPARTURE_SEEN, certain_failure, run_rungs, update_charge_bound
from fleetreserve.sparse_programme import Sizes

DATE = datetime.date(2020, 1, 1)
TOLERANCE = 1e-7
PATHS = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]

EARLY = Day(
    "hand",
    DATE,
    2,
    2,
    0.0,
    (TrueCar(2, 1.0, 1.0), TrueCar(5, 2.0, 3.0)),
    (ForecastCar(3.0, 1.0, 1.0, 1.0), ForecastCar(5.0, 2.0, 3.0, 3.0)),
    0.1,
)
EARLY_POLICY = AffinePolicy(
    [1.0, 1.0],
    [0.5, 0.5],
    [[0, 0.5, 0.5], [0, 0.5, 0.5]],
    [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0.5, 0], [0, 0.5]]],
)


EARLY_STAYS = dataclasses.replace(EARLY, cars=(TrueCar(3, 1.0, 1.0), TrueCar(5, 2.0, 3.0)))
GONE = dataclasses.replace(EARLY, cars=(TrueCar(1, 1.0, 1.0), TrueCar(5, 2.0, 3.0)))
GONE_POLICY = AffinePolicy(
    [1.0, 1.0],
    [0.5, 0.5],
    [[0.5, 0, 0.5], [0, 1.0, 0.5]],
    [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0.5, 0], [0, 0.5]]],
)


def _stay(day, departure_seen):
    floor = 0.0 if departure_seen == "hour-start" else 1.0
    return lambda car, hour: max(day.forecasts[car].departure, hour + floor)


def _run(day, policy, path, seen, **kwargs):
    bound = _stay(day, seen)
    return run_rungs(day, policy, 1.0, path, departure_bound=bound, departure_seen=seen, **kwargs)


def same_decisions(first, second, upto):
    """Two runs of one rung take the same decisions in hours 0 to `upto`: the same plan in force,
    broken or not, discharging or not, and the same re-solves with the same outcomes."""
    for h in range(upto + 1):
        a, b = first.plans[h], second.plans[h]
        if not (
            np.array_equal(a.constant, b.constant)
            and np.array_equal(a.gains, b.gains)
            and a.two_way == b.two_way
            and first.broken[h] == second.broken[h]
            and first.discharging[h] == second.discharging[h]
        ):
            return False
    early = [
        [(r.hour, r.outcome) for r in run.resolves if r.hour <= upto] for run in (first, second)
    ]
    return early[0] == early[1]


def test_certain_failure_by_hand():
    present = np.array([True, True, False])
    stored = np.zeros(3)
    bound = np.array([np.inf, 0.4, np.inf])
    kwargs = dict(discharge=False, budget=None)
    # A car that has left draws nothing: 1 + 0.4 at most against 1.6.
    assert certain_failure([1.0, 0.5, 0.2], 1.6, present, stored, bound, **kwargs)
    assert not certain_failure([1.0, 0.5, 0.0], 1.4, present, stored, bound, **kwargs)
    # A car may draw less than planned (unobservable), so too little planned is not certain.
    assert not certain_failure([1.0, 0.4, 0.0], 0.1, present, stored, bound, **kwargs)
    # Two-way: a discharge of at most B + x = 1 + 0.5 can be carried out, so at least -1.5.
    two = dict(discharge=True, budget=1.0)
    s = np.array([0.5, 0.0, 0.0])
    assert certain_failure([-2.0, 0.0, 0.0], -1.6, present, s, bound, **two)
    assert not certain_failure([-2.0, 0.0, 0.0], -1.4, present, s, bound, **two)


def test_charge_bound_by_hand():
    bound = np.array([np.inf, np.inf, 2.0, 1.0])
    planned = np.array([1.0, 1.0, 0.0, -0.5])
    draws = np.array([0.6, 1.0, 0.0, -0.5])
    new = update_charge_bound(bound, planned, draws, np.ones(4, bool))
    # Capped at 0.6; not capped; idle; a discharge of 0.5 frees 0.5 of room.
    np.testing.assert_array_equal(new, [0.6, np.inf, 2.0, 1.5])


@pytest.mark.parametrize(("day", "policy"), [(EARLY, EARLY_POLICY), (GONE, GONE_POLICY)])
def test_the_hand_policies_are_valid_on_the_forecast_sets(day, policy):
    rows = day_forecast_rows(day)
    assert np.all(policy.worst_case(rows) <= rows.bounds + TOLERANCE)


def _check_fixed(fixed):
    assert fixed.first_failure == 2
    assert fixed.slot_loss == pytest.approx(0.5)
    np.testing.assert_allclose(fixed.final, [0.5, 3.0], atol=TOLERANCE)
    assert fixed.driver_loss == pytest.approx(1 / 8)


@pytest.mark.parametrize("path", PATHS)
def test_hour_start_a_car_leaving_within_the_hour_fails_the_fixed_plan_and_the_rungs_alike(path):
    runs = _run(EARLY, EARLY_POLICY, path, "hour-start")
    _check_fixed(runs["fixed"].run)
    for name in ("one-way", "two-way"):
        assert runs[name].resolves == ()
        assert runs[name].own_from is None
        assert runs[name].run.first_failure == 2
        np.testing.assert_array_equal(runs[name].run.draws, runs["fixed"].run.draws)


def _check_re_solved_in_time(runs, path):
    fixed, one, two = runs["fixed"].run, runs["one-way"].run, runs["two-way"].run
    _check_fixed(fixed)
    assert [(r.hour, r.outcome) for r in runs["one-way"].resolves] == [(2, "optimal")]
    assert runs["one-way"].own_from == 2
    assert one.first_failure is None
    assert one.slot_loss == 0.0
    np.testing.assert_allclose(one.final, [0.5, 3.0], atol=TOLERANCE)
    assert one.driver_loss == pytest.approx(1 / 8)
    np.testing.assert_allclose(one.draws[1, 2], 1 + 0.5 * path[1], atol=TOLERANCE)
    assert runs["two-way"].resolves == ()
    np.testing.assert_array_equal(two.draws, one.draws)


@pytest.mark.parametrize("path", PATHS)
def test_rounded_hour_the_rung_sees_the_car_gone_at_the_start_of_its_departure_hour(path):
    _check_re_solved_in_time(_run(EARLY, EARLY_POLICY, path, "rounded-hour"), path)


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
@pytest.mark.parametrize("path", PATHS)
def test_a_car_seen_gone_lets_the_one_way_rung_re_solve_in_time(path, seen):
    _check_re_solved_in_time(_run(GONE, GONE_POLICY, path, seen), path)


@pytest.mark.parametrize("path", PATHS)
def test_hour_start_a_departure_within_the_current_hour_does_not_change_the_decisions(path):
    leaves = _run(EARLY, EARLY_POLICY, path, "hour-start")
    stays = _run(EARLY_STAYS, EARLY_POLICY, path, "hour-start")
    for name in ("one-way", "two-way"):
        assert same_decisions(leaves[name], stays[name], 2)


@pytest.mark.parametrize("path", PATHS)
def test_rounded_hour_a_departure_within_the_current_hour_changes_the_decision(path):
    leaves = _run(EARLY, EARLY_POLICY, path, "rounded-hour")
    stays = _run(EARLY_STAYS, EARLY_POLICY, path, "rounded-hour")
    assert same_decisions(leaves["one-way"], stays["one-way"], 1)
    assert not same_decisions(leaves["one-way"], stays["one-way"], 2)


def test_the_reading_of_when_a_departure_is_seen_must_be_given():
    with pytest.raises(TypeError):
        run_rungs(EARLY, EARLY_POLICY, 1.0, PATHS[0], departure_bound=_stay(EARLY, "hour-start"))
    with pytest.raises(ValueError, match="departure_seen"):
        _run(EARLY, EARLY_POLICY, PATHS[0], "hour-end")


def _made_up_resolver(outcome, calls):
    def resolver(state, guard, *, two_way, budget, time_limit):
        calls.append((state.hour, guard.hour, two_way))
        return ResolveResult(outcome, None, 0.0, 0.0, Sizes(0, 0, 0), f"solver says {outcome}")

    return resolver


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
@pytest.mark.parametrize("outcome", ["infeasible", "timeout", "error"])
def test_a_re_solve_that_is_not_optimal_leaves_the_predecessors_plan(outcome, seen):
    calls = []
    runs = _run(GONE, GONE_POLICY, (1.0, 1.0), seen, resolver=_made_up_resolver(outcome, calls))
    fixed = runs["fixed"].run
    for name in ("one-way", "two-way"):
        run = runs[name]
        np.testing.assert_array_equal(run.run.draws, fixed.draws)
        assert run.run.first_failure == fixed.first_failure == 2
        assert [(r.hour, r.outcome) for r in run.resolves] == [(2, outcome)]
        assert run.resolves[0].message == f"solver says {outcome}"  # kept for the report
        assert run.own_from is None
        assert run.plans[2] is runs["fixed"].plans[2]
    assert calls == [(2, 2, False), (2, 2, True)]


ZERO = Day(
    "zero",
    DATE,
    3,
    0,
    0.0,
    (TrueCar(4, 1.0, 1.0), TrueCar(4, 1.0, 0.0)),
    (ForecastCar(4.0, 1.0, 2.0, 1.0), ForecastCar(4.0, 1.0, 1.0, 0.0)),
    0.1,
)
ZERO_POLICY = AffinePolicy(
    [1.0, 1.0, -0.5], [0.0, 0.0, 0.0], [[0, 0.5, 1, -0.5], [0.5, 0.5, 0, 0]], np.zeros((2, 4, 3))
)
ZERO_TWO_WAY = Plan(np.array([[0, 1, 1, -0.5], [0, 0, 0, 0]]), np.zeros((2, 4, 3)), True)


def _zero_runs(seen):
    answered = []

    def resolver(state, guard, *, two_way, budget, time_limit):
        if two_way and not answered:
            answered.append(state.hour)
            return ResolveResult("optimal", ZERO_TWO_WAY, 0.0, 0.0, Sizes(0, 0, 0))
        return ResolveResult("infeasible", None, 0.0, 0.0, Sizes(0, 0, 0))

    runs = _run(ZERO, ZERO_POLICY, (0.0, 0.0, 0.0), seen, resolver=resolver)
    assert answered == [1]
    return runs


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_after_a_failed_re_solve_the_two_way_rung_follows_its_predecessor(seen):
    runs = _zero_runs(seen)
    fixed, one, two = runs["fixed"], runs["one-way"], runs["two-way"]
    assert fixed.run.first_failure == 1
    assert fixed.run.final == pytest.approx([1.0, 0.0])
    assert two.run.first_failure == 2
    np.testing.assert_allclose(two.run.draws[0], [0, 1, 0, 0], atol=TOLERANCE)
    assert [(r.hour, r.outcome) for r in two.resolves] == [(1, "optimal"), (3, "infeasible")]
    assert two.plans[2] is ZERO_TWO_WAY
    assert two.plans[3] is one.plans[3] is fixed.plans[3]  # "the predecessor's plan stays"
    assert two.discharging[3] is False
    assert two.run.final == pytest.approx([1.0, 0.0])
    assert np.all(two.run.final >= one.run.final - TOLERANCE)  # T5(b) holds


UNIT = Day(
    "unit",
    DATE,
    3,
    0,
    0.0,
    (TrueCar(4, 1.0, 1.0), TrueCar(1, 1.0, 0.5)),
    (ForecastCar(4.0, 1.0, 2.0, 0.5), ForecastCar(4.0, 1.0, 1.0, 0.0)),
    0.1,
)
UNIT_POLICY = AffinePolicy(
    [1.0, 1.0, -0.5], [0.0, 0.0, 0.0], [[0, 1, 0.5, -0.5], [0, 0, 0.5, 0]], np.zeros((2, 4, 3))
)


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
@pytest.mark.parametrize("zeroing", [True, False])
def test_the_t5_unit_case_through_the_real_re_solve(monkeypatch, zeroing, seen):
    """The plan's T5 unit case with the real re-solve (`UNIT`). Car Z: rate 1, true room 1,
    forecast room 2. Car X leaves within hour 1 (forecast to stay) and plans 0.5 kW in hour 2 only.
    Box `(1, 1, -0.5)`, `w = 0`. Hour 1: Z takes 1 and is full. At the start of hour 2 X is seen
    gone, so the plan is certain to fail before any failure: the two-way rung's real re-solve plans
    Z (1, -0.5) in hours 2 and 3 (planned stored energy 2, then 1.5, within the forecast room).
    The one-way rung's re-solve is infeasible (hour 3 asks for -0.5). Z is full and takes nothing
    in hour 2 (first failure). In hour 3 the one-way re-solve is infeasible, so the two-way rung
    follows the one-way rung's plan (the fixed plan, run one-way): Z draws nothing and ends at 1.0,
    as under the fixed plan, with the zeroing rule on or off. Under the alternative reading (the
    rung keeps its own plan) with the rule off, Z would give 0.5 back and end at 0.5."""
    if not zeroing:
        monkeypatch.setattr(replay, "discharges_carried", lambda plan, failed_before: plan.two_way)
    runs = _run(UNIT, UNIT_POLICY, (0.0, 0.0, 0.0), seen)
    fixed, one, two = runs["fixed"], runs["one-way"], runs["two-way"]
    assert fixed.run.first_failure == 2
    assert fixed.run.final[0] == pytest.approx(1.0)
    assert [(r.hour, r.outcome) for r in one.resolves] == [(2, "infeasible"), (3, "infeasible")]
    assert [(r.hour, r.outcome) for r in two.resolves] == [(2, "optimal"), (3, "infeasible")]
    plan = two.plans[2]
    assert plan.two_way
    np.testing.assert_allclose(plan.constant[0, 2:], [1.0, -0.5], atol=TOLERANCE)
    np.testing.assert_allclose(two.run.planned[0], [0, 1, 1, -0.5], atol=TOLERANCE)
    np.testing.assert_allclose(two.run.draws[0], [0, 1, 0, 0], atol=TOLERANCE)
    assert two.run.first_failure == 2
    assert two.plans[3] is fixed.plans[3]
    assert two.run.final[0] == pytest.approx(1.0)
    assert np.all(two.run.final >= one.run.final - TOLERANCE)  # T5(b) holds


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_the_rung_engine_never_reaches_the_zeroing_rule(monkeypatch, seen):
    """After its first failure in hour `h` a two-way rung's plan is broken at `h + 1`, so it
    re-solves one-way or follows its one-way predecessor: no planned discharge is left to zero."""
    monkeypatch.setattr(replay, "discharges_carried", lambda plan, failed_before: plan.two_way)
    runs = _zero_runs(seen)
    two = runs["two-way"].run
    np.testing.assert_allclose(two.draws[0], [0, 1, 0, 0], atol=TOLERANCE)
    assert two.final[0] == pytest.approx(1.0)
    assert two.final[0] >= runs["one-way"].run.final[0] - TOLERANCE
