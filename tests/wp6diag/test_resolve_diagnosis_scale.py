"""The largest feasible scale of the remaining box (`resolve_diagnosis_lp.box_scale`, step 5 there)
against a direct construction that does not use the diagnosis programme.

The definition: the largest `s` in [0, 1] such that the re-solve is feasible in the world whose
remaining box is its centre plus `s` times its half-widths, with the predecessor plan's gains on
the future requests `s` times as large and the request of the current hour as it is.

The direct construction (`direct`) builds that world as inputs of `resolve.resolve` itself: the
half-widths `w` of the hours after the current one are multiplied by `s`, and so are the
predecessor plan's gains on the requests of those hours. `resolve.draft` then writes its own rows
for them, including its own enumeration of the sign subsets of a one-way guard at that `s`. The
sample is every re-solve of the one-way rung on seeded made-up days (both readings of when a
departure is seen, three scales; some follow a car that truly left before its forecast departure),
and seeded random hand states with guards that have gains, one-way and two-way, whose cars are
absent, limited in rate or in room, or need energy.

1. Feasibility is monotone in `s` on a grid of 0, 0.1, .., 1, by the direct construction.
2. The reported largest scale agrees with the direct grid to the grid's resolution (feasible at
   every grid point up to it, infeasible at every grid point above it), and, by bisection of the
   direct construction, to 1e-4 on a subset.
3. At `s = 1` the programme is the re-solve's own: with `s` put in as 1, its rows are the rows of
   the re-solve (same matrices and right sides), and the direct construction at 1 is the re-solve;
   so an infeasible re-solve has a reported scale below 1 or none.
4. "None" (outcome infeasible) is reported exactly when `s = 0` is infeasible by the direct
   construction.
5. A hand case in which the predecessor plan has a gain on a future request: the old reading
   (the gain kept at its full size while the new plan's width shrinks) has no feasible `s`; the
   definition gives 0.4. The sign of the width rows matters in it.
6. The variable `s` has the upper bound 1 (a spy reads the solver's own value of `s`, where the
   reported scale is clipped).
"""

import dataclasses
import functools
import random

import numpy as np
import pytest
from diag_fake import BUDGET, HandCar, hand, prepared_of
from made_up_offer_days import offer_day

from fleetreserve import resolve_diagnosis_day as diagnosis
from fleetreserve import resolve_diagnosis_lp as relaxed
from fleetreserve.plans import Plan
from fleetreserve.resolve import Guard, Remaining, draft, resolve

SEED = 53
N_DAYS = 14
BETAS = (0.5, 1.0, 1.5)
READINGS = ("hour-start", "rounded-hour")
N_HAND = 120
N_MILD = 260
GRID = [k / 10 for k in range(11)]
TOLERANCE = 1e-6


def direct(state: Remaining, guard: Guard, s: float, **kwargs) -> str:
    """The outcome of `resolve.resolve` on the world of scale `s` (module text), built directly."""
    t = state.hour
    width = np.array(state.w, dtype=float)
    width[t:] *= s
    gains = np.array(guard.plan.gains, dtype=float)
    gains[:, :, t:] *= s
    plan = Plan(guard.plan.constant, gains, two_way=guard.plan.two_way)
    scaled = Guard(plan, guard.hour, guard.stored, guard.clip)
    return resolve(dataclasses.replace(state, w=width), scaled, **kwargs).outcome


def _hand_state(rng: random.Random):
    """A seeded random hand re-solve: (state, guard, kwargs of the re-solve)."""
    n = rng.randint(2, 4)
    t = rng.randint(1, n - 1)
    cars = []
    for _ in range(rng.randint(2, 4)):
        rate = rng.choice([0.5, 1.0, 2.0, 3.0])
        cars.append(
            HandCar(
                planned=tuple([0.0, *(rng.uniform(-0.5, 1.5) for _ in range(n))]),
                rate=rate,
                cap=rng.choice([1.0, 3.0, 10.0]),
                need=rng.choice([0.0, 0.0, 0.5, 2.0]),
                stored=rng.choice([0.0, 0.0, 0.4]),
                present=rng.random() < 0.8,
                departure=rng.choice([float(t) + 0.5, 9.0, 9.0]),
                charge_bound=rng.choice([np.inf, np.inf, 0.8]),
            )
        )
    gains = {}
    for car in range(len(cars)):
        value = np.zeros((n + 1, n))
        for hour in range(1, n + 1):
            value[hour, :hour] = [rng.choice([0.0, rng.uniform(-0.6, 0.6)]) for _ in range(hour)]
        gains[car] = value
    m = [rng.uniform(0.2, 2.5) for _ in range(n)]
    w = [rng.uniform(0.0, 1.2) for _ in range(n)]
    known = [rng.choice([-1.0, 0.0, 1.0]) for _ in range(t)]
    two_way = rng.random() < 0.3
    state, guard = hand(
        cars,
        hour=t,
        m=m,
        w=w,
        known=known,
        beta=rng.choice([0.5, 1.0]),
        n_recovery=rng.randint(0, 2),
        gains=gains,
        clip=not two_way,
    )
    guard = dataclasses.replace(guard, plan=Plan(guard.plan.constant, guard.plan.gains, two_way))
    return state, guard, dict(two_way=two_way, budget=BUDGET)


def _mild_state(rng: random.Random):
    """A seeded random hand re-solve that is usually feasible at the centre of the box: the cars
    share the requests, the box is set against the cars' total rate, and the guarded plan has
    each car carry its share of the request and of the half-width. Some of these have hours in
    which a planned power can take both signs (one-way guards then write several subsets)."""
    n = rng.randint(2, 4)
    t = rng.randint(1, n - 1)
    n_cars = rng.randint(2, 4)
    rates = [float(rng.choice([1, 2, 3])) for _ in range(n_cars)]
    present = [True, *(rng.random() < 0.85 for _ in range(n_cars - 1))]
    total = sum(r for r, here in zip(rates, present, strict=True) if here)
    centre = [rng.uniform(0.2, 0.6) * total for _ in range(n)]
    half = [rng.uniform(0.1, 0.6) * total for _ in range(n)]
    centre[t - 1], half[t - 1] = rng.uniform(0.1, 0.5) * total, rng.uniform(0.0, 0.3) * total
    raw = [rng.uniform(0.2, 1.0) if here else 0.0 for here in present]
    share = [value / sum(raw) for value in raw]
    known = [rng.choice([-1.0, 0.0, 1.0]) for _ in range(t)]
    beta = rng.choice([0.5, 1.0])
    cars, gains = [], {}
    for i in range(n_cars):
        value = np.zeros((n + 1, n))
        for hour in range(1, n + 1):
            value[hour, hour - 1] = beta * half[hour - 1] * share[i]
        gains[i] = value
        cars.append(
            HandCar(
                planned=tuple([0.0, *(centre[s] * share[i] for s in range(n))]),
                rate=rates[i],
                cap=rng.choice([6.0, 10.0, 30.0]),
                need=rng.choice([0.0, 0.0, 0.5]),
                stored=rng.choice([0.0, 0.0, 0.3]),
                present=present[i],
                departure=rng.choice([float(t) + 0.5, 9.0, 9.0, 9.0]),
            )
        )
    two_way = rng.random() < 0.25
    state, guard = hand(
        cars, hour=t, m=centre, w=half, known=known, beta=beta, gains=gains, clip=not two_way
    )
    guard = dataclasses.replace(guard, plan=Plan(guard.plan.constant, guard.plan.gains, two_way))
    return state, guard, dict(two_way=two_way, budget=BUDGET)


@functools.cache
def _sample():
    """The infeasible re-solves of the sample, each as (state, guard, kwargs, early leaver)."""
    found = []
    rng = random.Random(SEED)
    for k in range(N_DAYS):
        day, certificate, _ = offer_day(rng, k)
        for seen in READINGS:
            prepared = prepared_of(day, certificate, seen)
            for beta in BETAS:
                for path in diagnosis.collect(prepared, beta):
                    for a in path.attempts:
                        if a.outcome != "infeasible":
                            continue
                        early, _ = diagnosis.hindsight(prepared, path.runs, a.state.hour)
                        kwargs = dict(two_way=a.two_way, budget=a.budget)
                        found.append((a.state, a.guard, kwargs, early > 0, "day"))
    rng = random.Random(SEED + 1)
    for make, count in ((_hand_state, N_HAND), (_mild_state, N_MILD)):
        for _ in range(count):
            state, guard, kwargs = make(rng)
            if resolve(state, guard, **kwargs).outcome == "infeasible":
                found.append((state, guard, kwargs, False, "hand"))
    return found


@functools.cache
def _grid(index: int) -> tuple:
    state, guard, kwargs, _, _ = _sample()[index]
    return tuple(direct(state, guard, s, **kwargs) for s in GRID)


@functools.cache
def _reported(index: int) -> tuple:
    state, guard, kwargs, _, _ = _sample()[index]
    return relaxed.box_scale(state, guard, **kwargs)


def _indices():
    return range(len(_sample()))


def test_the_sample_is_a_few_hundred_infeasible_re_solves_with_early_leavers():
    sample = _sample()
    kinds = [row[4] for row in sample]
    print(
        f"{len(sample)} infeasible re-solves: {kinds.count('day')} on made-up days, "
        f"{kinds.count('hand')} hand states; {sum(row[3] for row in sample)} after an early leaver"
    )
    assert len(sample) >= 200
    assert kinds.count("day") >= 100 and kinds.count("hand") >= 40
    assert sum(row[3] for row in sample) >= 30
    assert any(row[2]["two_way"] for row in sample)
    assert {outcome for i in _indices() for outcome in _grid(i)} == {"optimal", "infeasible"}


def test_the_direct_construction_at_scale_one_is_the_re_solve_itself():
    for state, guard, kwargs, _, _ in _sample():
        assert direct(state, guard, 1.0, **kwargs) == "infeasible"
        assert resolve(state, guard, **kwargs).outcome == "infeasible"


def test_feasibility_is_monotone_in_the_scale_by_the_direct_construction():
    for i in _indices():
        feasible = [outcome == "optimal" for outcome in _grid(i)]
        assert not feasible[-1]
        assert feasible == sorted(feasible, reverse=True), (i, _grid(i))


def test_the_reported_scale_agrees_with_the_direct_grid():
    values = []
    for i in _indices():
        scale, outcome = _reported(i)
        grid = _grid(i)
        if outcome == "infeasible":
            assert scale is None and set(grid) == {"infeasible"}, (i, grid)
            continue
        assert outcome == "optimal" and 0.0 <= scale < 1.0, (i, scale)
        values.append(scale)
        for s, answer in zip(GRID, grid, strict=True):
            if s <= scale - TOLERANCE:
                assert answer == "optimal", (i, scale, s)
            if s >= scale + TOLERANCE:
                assert answer == "infeasible", (i, scale, s)
    print(f"{len(values)} of {len(_sample())} infeasible re-solves have a feasible scale")
    assert sum(0 < v < 1 for v in values) >= 40  # not only the two ends
    assert len(values) < len(_sample())  # and some have none


def test_the_reported_scale_agrees_with_a_bisection_of_the_direct_construction():
    checked = 0
    for i in _indices():
        scale, outcome = _reported(i)
        if outcome != "optimal" or not 0.0 < scale < 1.0 or checked >= 40:
            continue
        state, guard, kwargs, _, _ = _sample()[i]
        low, high = scale - 1e-4, scale + 1e-4
        assert direct(state, guard, max(low, 0.0), **kwargs) == "optimal", (i, scale)
        assert direct(state, guard, min(high, 1.0), **kwargs) == "infeasible", (i, scale)
        checked += 1
    assert checked >= 30


def test_none_is_reported_exactly_when_the_centre_is_infeasible():
    none = 0
    for i in _indices():
        reported_none = _reported(i)[1] == "infeasible"
        assert reported_none == (_grid(i)[0] == "infeasible"), i
        none += reported_none
    assert 10 <= none < len(_sample())


def test_at_scale_one_the_programme_is_the_re_solves_own():
    """With `s` put in as 1 the rows are those of `resolve.draft`: same matrices, same sides."""
    checked = 0
    for state, guard, kwargs, _, _ in _sample()[::7]:
        programme, s, *_ = relaxed.assemble(state, guard, **kwargs, scale=True)
        plain, *_ = relaxed.assemble(state, guard, **kwargs)
        assert s == programme.n_variables - 1  # `s` is the last variable
        for sense in ("le", "eq"):
            matrix, rhs = programme._matrix(sense)
            expected, expected_rhs = plain._matrix(sense)
            if matrix is None:  # no row of this sense (a re-solve with no present car)
                assert expected is None
                continue
            column = matrix[:, [s]].toarray().reshape(-1)
            folded = matrix[:, :s]
            np.testing.assert_allclose(rhs - column, expected_rhs, atol=1e-12)
            assert abs(folded - expected).max() <= 1e-12
        checked += 1
    assert checked >= 20


# --- a hand case with a gain in the predecessor plan --------------------------------------------


def _gain_case(planned_second_hour=0.7):
    """One car, rate 1; hour 1 asks 0.5; hour 2 asks 0.8 + 0.5 zeta_2, up to 1.3 against the rate;
    the predecessor plan gives (0.5, planned + 0.5 zeta_2), so its energy by hour 2 is at most
    0.5 + planned + 0.5 against the new plan's 1.3 at the top of the box."""
    cars = [HandCar(planned=(0, 0.5, planned_second_hour), rate=1.0)]
    gains = {0: [[0, 0], [0, 0], [0, 0.5]]}
    return hand(cars, hour=1, m=[0.5, 0.8], w=[0.0, 0.5], known=[0.0], gains=gains)


def test_a_gain_of_the_predecessor_plan_shrinks_with_the_box():
    state, guard = _gain_case()
    one_way = dict(two_way=False, budget=BUDGET)
    assert resolve(state, guard, **one_way).outcome == "infeasible"
    scale, outcome = relaxed.box_scale(state, guard, **one_way)
    assert outcome == "optimal" and scale == pytest.approx(0.4, abs=1e-7)  # 0.8 + 0.5 s <= 1
    assert direct(state, guard, 0.4 - 1e-4, **one_way) == "optimal"
    assert direct(state, guard, 0.4 + 1e-4, **one_way) == "infeasible"
    # The old reading shrinks the new plan's width and keeps the predecessor's gain 0.5 at full
    # size: the guard then needs the new plan's energy to be 0.5 (1 - s) above the old one's, and
    # the surplus is 0.8 - 0.7 = 0.1, so s >= 0.8, which the rate forbids (s <= 0.4). Nothing is
    # feasible there, the centre included; the definition says 0.4.
    for s in (0.0, 0.4):
        old_reading = dataclasses.replace(state, w=np.array([0.0, 0.5 * s]))
        assert resolve(old_reading, guard, **one_way).outcome == "infeasible"


def test_the_guard_spread_rows_carry_the_scale():
    state, guard = _gain_case()
    one_way = dict(two_way=False, budget=BUDGET)
    built = draft(state, guard, **one_way)
    gain_rows = [
        (kind, rhs) for kind, rhs in zip(built.le.kind, built.le.rhs, strict=True) if rhs != 0
    ]
    assert {kind for kind, _ in gain_rows} >= {"spread"}
    programme, s, *_ = relaxed.assemble(state, guard, **one_way, scale=True)
    matrix, rhs = programme._matrix("le")
    on_s = np.flatnonzero(matrix[:, [s]].toarray().reshape(-1))
    assert sorted(abs(matrix[on_s, s].toarray().reshape(-1))) == sorted(
        abs(rhs_value) for kind, rhs_value in gain_rows if kind == "spread"
    )
    assert np.all(rhs[on_s] == 0)


def test_the_scale_variable_stops_at_one():
    """The rows alone allow `s` up to 1.4 here (the guard: 1.3 - 0.5 s >= 0.5 + 0.1), so only the
    variable's own upper bound keeps it at 1; the reported scale is clipped, so the spy reads the
    solver's value."""
    seen = []

    def spy(programme, columns, values, time_limit):
        result = relaxed.solve_programme(programme, columns, values, time_limit)
        seen.append(float(result.x[columns[0]]))
        return result

    cars = [HandCar(planned=(0, 0.5, 0.1), rate=2.0)]
    state, guard = hand(cars, hour=1, m=[0.5, 0.8], w=[0.0, 0.5], known=[0.0])
    assert resolve(state, guard, two_way=False, budget=BUDGET).outcome == "optimal"
    assert relaxed.box_scale(state, guard, two_way=False, budget=BUDGET, solve=spy) == (
        pytest.approx(1.0, abs=1e-7),
        "optimal",
    )
    assert seen == [pytest.approx(1.0, abs=1e-7)]  # not 1.4, which the other rows allow
