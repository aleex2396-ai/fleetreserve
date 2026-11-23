"""Properties of the fifth goal of `asa-es` (`asa_equal_share`) on 1,000 seeded made-up hour views.

Views: 1 to 6 cars, 1 to 5 service hours, any hour, beliefs, energy drawn, plugged-in flags,
centres from below 0 up, recovery hours, targets below 0, inside and above the believed total;
on about a third of the views every car has the same belief and has drawn the same, so that
`asa`'s ties between cars are frequent. Each view is planned by `asa` (`asa_programme.plan_hour`)
and by `asa-es` (`asa_equal_share.plan_hour`). Every one of the 1,000 is optimal in all stages.

1. Goals 1 to 4 of `asa-es` are `asa`'s: the same outcomes and optima, bit for bit.
2. Goal 5 never worsens goals 1 to 4 beyond the stage tolerance: the plan returned, measured with
   this file's own four objectives, is within the sum of `STAGE_TOLERANCE` and HiGHS's primal
   feasibility tolerance (1e-7), times the number of cars and the number of service hours, of
   each of the four optima.
3. Every planned power lies in `[0, upper]` (A3 of `asa_programme`), and the hour's total equals
   `asa`'s to 1e-6 kWh.
4. The fifth optimum is the sum of the approximation `f` (`approximate_square`) of the plan's
   powers, to 1e-6; it is at most the sum of `f` of `asa`'s plan (to 1e-6 per car-hour, as
   `asa`'s plan keeps goals 1 to 4 to the solver's tolerances only).
5. The bound of E1: the horizon's sum of squares of `asa-es`'s plan is at most that of `asa`'s plan
   plus `G`, the sum of `largest_gap` over the horizon's car-hours (plus 1e-6 per car-hour).
   The same for the first hour alone does not hold on every view (case 5 of
   `test_asa_es_hand`), and this file asserts nothing about it.
6. On the first 200 views the fifth optimum equals that of a programme written apart in this
   file: dense rows, the epigraph form of E1 (one variable and `K` chord rows per car-hour),
   goals 1 to 4 kept as rows at `asa`'s optima, `scipy.optimize.linprog` with HiGHS.
7. The first hour's plans of `asa` and `asa-es` differ (by more than 1e-6 kW) on at least 50 of
   the 1,000 views: the tie is broken differently often enough to be seen.
"""

import numpy as np
import pytest
from scipy.optimize import linprog

from fleetreserve import asa_equal_share
from fleetreserve.asa_programme import STAGE_TOLERANCE, upper_bounds
from fleetreserve.asa_programme import plan_hour as asa_plan_hour
from fleetreserve.controller import Belief, HourView
from fleetreserve.priority_rules import most

N_VIEWS = 1000
N_DENSE = 200
SLACK = STAGE_TOLERANCE + 1e-7  # stage tolerance plus HiGHS's primal feasibility tolerance
AGREE = 1e-6


def _random(seed):
    rng = np.random.default_rng(20_000 + seed)
    cars = int(rng.integers(1, 7))
    n = int(rng.integers(1, 6))
    hour = int(rng.integers(0, n + 1))
    same = rng.uniform() < 0.33

    def draw(low, high):
        value = rng.uniform(low, high, cars)
        return np.full(cars, value[0]) if same else value

    cap = draw(0, 20)
    belief = Belief(
        departure=draw(0, n + 4),
        rate=np.full(cars, 7.2) if same else rng.choice([3.7, 7.2, 11.0], cars),
        cap=cap,
        need=cap * draw(0, 1.3),
    )
    drawn = cap * draw(0, 0.8)
    plugged = np.ones(cars, bool) if same else rng.uniform(size=cars) < 0.85
    view = HourView(hour, n, 0.0, np.zeros(hour), np.zeros(hour), drawn, plugged, belief)
    total = float(most(view).sum())
    target = (-1.0, rng.uniform(0, total), total + rng.uniform(0, 3))[int(rng.integers(0, 3))]
    view = HourView(hour, n, target, np.zeros(hour), np.zeros(hour), drawn, plugged, belief)
    centre = rng.uniform(-2, 25, n)
    return view, centre, int(rng.integers(0, 3))


@pytest.fixture(scope="module")
def planned():
    out = []
    for seed in range(N_VIEWS):
        view, centre, n_recovery = _random(seed)
        asa = asa_plan_hour(view, centre, n_recovery, time_limit=300.0)
        es = asa_equal_share.plan_hour(view, centre, n_recovery, time_limit=300.0)
        out.append((view, centre, n_recovery, asa, es))
    return out


def _gap(view, n_recovery):
    b = view.belief
    recovery = np.arange(view.n_service + 1, view.n_service + 1 + n_recovery)
    supply = np.array(
        [b.rate[i] * np.clip(b.departure[i] - recovery, 0.0, 1.0).sum() for i in range(b.n_cars)]
    )
    return b.need - view.drawn - supply


def _measure(view, centre, n_recovery, powers):
    """The four objectives of `asa_programme` (A4) of a plan (cars by hours `t..n`)."""
    hours = np.arange(view.hour, view.n_service + 1)
    gap = _gap(view, n_recovery)
    track = abs(powers[:, 0].sum() - max(view.target, 0.0))
    energy = powers.sum(axis=1)
    need = sum(max(0.0, gap[i] - energy[i]) for i in np.flatnonzero(view.plugged))
    later = sum(
        abs(powers[:, j].sum() - max(centre[s - 1], 0.0)) for j, s in enumerate(hours) if j > 0
    )
    quick = -sum((view.n_service - s + 1) * powers[:, j].sum() for j, s in enumerate(hours))
    return np.array([track, need, later, quick])


def _f(powers, upper):
    return sum(
        float(asa_equal_share.approximate_square(p, u)) if u > 0 else 0.0
        for p, u in zip(powers.reshape(-1), upper.reshape(-1), strict=True)
    )


def test_every_view_is_optimal_and_goals_1_to_4_are_asas(planned):
    for _, _, _, asa, es in planned:
        assert asa.outcomes == ("optimal",) * 4
        assert es.outcomes == ("optimal",) * 5
        assert es.optima[:4] == asa.optima


def test_goal_5_keeps_goals_1_to_4(planned):
    for view, centre, n_recovery, _, es in planned:
        measured = _measure(view, centre, n_recovery, es.powers)
        slack = SLACK * view.n_cars * view.n_service
        assert np.all(measured <= np.array(es.optima[:4]) + slack)


def test_bounds_and_the_hours_total(planned):
    for view, _, _, asa, es in planned:
        upper = upper_bounds(view)
        assert np.all(es.powers >= 0) and np.all(es.powers <= upper)
        assert abs(es.powers[:, 0].sum() - asa.powers[:, 0].sum()) <= AGREE


def test_the_fifth_optimum_is_the_approximation_of_the_plan(planned):
    for view, _, _, asa, es in planned:
        upper = upper_bounds(view)
        assert abs(es.optima[4] - _f(es.powers, upper)) <= AGREE
        assert es.optima[4] <= _f(asa.powers, upper) + AGREE * upper.size


def test_the_horizons_sum_of_squares_is_within_the_gap_of_asas(planned):
    for view, _, _, asa, es in planned:
        upper = upper_bounds(view)
        bound = float(asa_equal_share.largest_gap(upper).sum())
        assert (es.powers**2).sum() <= (asa.powers**2).sum() + bound + AGREE * upper.size


def _dense_fifth(view, centre, n_recovery, optima, pieces=asa_equal_share.PIECES):
    """Stage 5 as one dense programme: powers, the four goals' own variables, and one epigraph
    variable per car-hour with `pieces` chord rows; goals 1 to 4 kept as rows at `optima`."""
    upper = upper_bounds(view)
    cars, length = upper.shape
    hours = np.arange(view.hour, view.n_service + 1)
    live = np.flatnonzero(view.plugged)
    gap = _gap(view, n_recovery)
    room = np.maximum(view.belief.cap - view.drawn, 0.0)
    n_p = cars * length
    e = n_p
    u = n_p + 1 + np.arange(live.size)
    z = n_p + 1 + live.size + np.arange(length - 1)
    y = n_p + 1 + live.size + (length - 1) + np.arange(n_p)
    n_var = n_p + 1 + live.size + (length - 1) + n_p
    rows, rhs = [], []

    def row(entries, value):
        r = np.zeros(n_var)
        for k, v in entries:
            r[k] += v
        rows.append(r)
        rhs.append(value)

    first = [(i * length, 1.0) for i in range(cars)]
    target = max(view.target, 0.0)
    row([*first, (e, -1.0)], target)
    row([*((k, -v) for k, v in first), (e, -1.0)], -target)
    for i in range(cars):
        row([(i * length + j, 1.0) for j in range(length)], room[i])
    for k, i in enumerate(live):
        row([*((i * length + j, -1.0) for j in range(length)), (u[k], -1.0)], -gap[i])
    for j in range(1, length):
        m = max(centre[hours[j] - 1], 0.0)
        hour = [(i * length + j, 1.0) for i in range(cars)]
        row([*hour, (z[j - 1], -1.0)], m)
        row([*((k, -v) for k, v in hour), (z[j - 1], -1.0)], -m)
    costs = [np.zeros(n_var) for _ in range(5)]
    costs[0][e] = 1.0
    costs[1][u] = 1.0
    costs[2][z] = 1.0
    for i in range(cars):
        for j, s in enumerate(hours):
            costs[3][i * length + j] = -(view.n_service - s + 1)
    costs[4][y] = 1.0
    for k, optimum in enumerate(optima):
        rows.append(costs[k].copy())
        rhs.append(optimum + STAGE_TOLERANCE)
    for k in range(n_p):  # y_k >= chord line of piece q at power k
        top = upper.reshape(-1)[k]
        for q in range(pieces):
            a, b = q * top / pieces, (q + 1) * top / pieces
            row([(k, a + b), (y[k], -1.0)], a * b)
    bounds = [(0.0, upper.reshape(-1)[k]) for k in range(n_p)] + [(0.0, None)] * (n_var - n_p)
    found = linprog(costs[4], A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds)
    assert found.status == 0, found.message
    return found.fun


def test_the_fifth_optimum_equals_a_programme_written_apart(planned):
    for view, centre, n_recovery, asa, es in planned[:N_DENSE]:
        assert abs(es.optima[4] - _dense_fifth(view, centre, n_recovery, asa.optima)) <= AGREE


def test_the_first_hours_plans_differ_often_enough_to_be_seen(planned):
    differ = sum(np.abs(asa.powers[:, 0] - es.powers[:, 0]).max() > 1e-6 for *_, asa, es in planned)
    assert differ >= 50
