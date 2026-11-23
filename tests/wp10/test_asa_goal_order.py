"""The order of the goals of the linear-programme form of ASA (`asa_programme`), on seeded views.

On seeded random hour views (1 to 5 cars, 1 to 4 service hours, any hour, beliefs, energy drawn,
plugged-in flags, centres, recovery hours, targets from below 0 to above the believed total):

- stage 1's optimum equals the closed form |T0 - min(T0, believed total most)|, with T0 =
  max(target, 0) the target as the programme reads it (A7 of `asa_programme`): no plan tracks the
  hour's target better;
- stage 2's optimum equals a closed form written apart from the programme: with the hour's total
  fixed at stage 1's value T, car i's unmet need is at least base_i = max(0, gap_i - min(U_i,
  room_i)) (gap_i: need - drawn - recovery supply; U_i: its most in the later hours; room_i:
  max(cap - drawn, 0)), each unit of energy in the hour lowers it by one up to
  g_i = max(0, min(gap_i, room_i) - min(U_i, room_i)) and at most the car's most, so the least
  total is sum(base) - min(T, sum(min(most_i, g_i)));
- stage 3's optimum and stage 4's optimum equal those of one programme per stage written apart
  in this file (dense rows, one row per absolute value, `scipy.optimize.linprog` with HiGHS),
  with the earlier stages' optima kept as rows within the same tolerance;
- the final plan, measured by this file's own functions, is within `STAGE_TOLERANCE` plus HiGHS's
  primal feasibility tolerance (1e-7) of each earlier stage's optimum: a later stage never
  worsens an earlier one beyond that.
"""

import numpy as np
from scipy.optimize import linprog

from fleetreserve.asa_programme import STAGE_TOLERANCE, plan_hour
from fleetreserve.controller import Belief, HourView
from fleetreserve.priority_rules import most

N_VIEWS = 300
SLACK = STAGE_TOLERANCE + 1e-7  # stage tolerance plus HiGHS's primal feasibility tolerance
AGREE = 1e-6  # kWh: two solvers' optima of the same stage


def _random(seed):
    rng = np.random.default_rng(seed)
    cars = int(rng.integers(1, 6))
    n = int(rng.integers(1, 5))
    hour = int(rng.integers(0, n + 1))
    cap = rng.uniform(0, 20, cars)
    belief = Belief(
        departure=rng.uniform(0, n + 4, cars),
        rate=rng.uniform(0.5, 8, cars),
        cap=cap,
        need=cap * rng.uniform(0, 1.3, cars),
    )
    drawn = cap * rng.uniform(0, 0.8, cars)
    plugged = rng.uniform(size=cars) < 0.85
    view = HourView(hour, n, 0.0, np.zeros(hour), np.zeros(hour), drawn, plugged, belief)
    total = float(most(view).sum())
    target = (-1.0, rng.uniform(0, total), total + rng.uniform(0, 3))[int(rng.integers(0, 3))]
    view = HourView(hour, n, target, np.zeros(hour), np.zeros(hour), drawn, plugged, belief)
    centre = rng.uniform(0, 10, n)
    return view, centre, int(rng.integers(0, 3))


VIEWS = [_random(seed) for seed in range(N_VIEWS)]


def _parts(view, n_recovery):
    """This file's own reading of the bounds, room and gap (module text of asa_programme)."""
    b = view.belief
    hours = np.arange(view.hour, view.n_service + 1)
    upper = np.zeros((view.n_cars, hours.size))
    for i in range(view.n_cars):
        if not view.plugged[i]:
            continue
        for j, s in enumerate(hours):
            share = min(max(b.departure[i] - s, 0.0), 1.0)
            upper[i, j] = b.rate[i] * share
        upper[i, 0] = max(min(upper[i, 0], b.cap[i] - view.drawn[i]), 0.0)
    room = np.maximum(b.cap - view.drawn, 0.0)
    recovery = np.arange(view.n_service + 1, view.n_service + 1 + n_recovery)
    supply = np.array(
        [b.rate[i] * np.clip(b.departure[i] - recovery, 0.0, 1.0).sum() for i in range(b.n_cars)]
    )
    return hours, upper, room, b.need - view.drawn - supply


def _measure(view, centre, n_recovery, powers):
    """The four stage objectives of a plan (cars by hours), as minimised."""
    hours, _, _, gap = _parts(view, n_recovery)
    track = abs(powers[:, 0].sum() - max(view.target, 0.0))
    energy = powers.sum(axis=1)
    need = sum(max(0.0, gap[i] - energy[i]) for i in np.flatnonzero(view.plugged))
    later = sum(abs(powers[:, j].sum() - centre[s - 1]) for j, s in enumerate(hours) if j > 0)
    quick = -sum((view.n_service - s + 1) * powers[:, j].sum() for j, s in enumerate(hours))
    return np.array([track, need, later, quick])


def _closed_stage_1_and_2(view, n_recovery):
    hours, upper, room, gap = _parts(view, n_recovery)
    first = upper[:, 0]
    total = min(max(view.target, 0.0), first.sum())
    stage_1 = abs(max(view.target, 0.0) - total)
    future = np.minimum(upper[:, 1:].sum(axis=1), room)
    base = np.maximum(gap - future, 0.0)
    gain = np.maximum(np.minimum(gap, room) - future, 0.0)
    live = view.plugged
    stage_2 = base[live].sum() - min(total, np.minimum(first, gain)[live].sum())
    return stage_1, stage_2


def _dense_stage(view, centre, n_recovery, stage, kept):
    """Stage `stage` (2 = centre, 3 = quick, counted from 0) as one dense programme over the
    powers and one epigraph variable per absolute value or unmet need."""
    hours, upper, room, gap = _parts(view, n_recovery)
    cars, length = upper.shape
    live = np.flatnonzero(view.plugged)
    n_p = cars * length
    # variables: powers, then e (|hour total - target|), then u_i (unmet), then z_j (|later - m|)
    n_var = n_p + 1 + live.size + (length - 1)
    e = n_p
    u = n_p + 1 + np.arange(live.size)
    z = n_p + 1 + live.size + np.arange(length - 1)
    rows, rhs = [], []

    def row(entries, value):
        r = np.zeros(n_var)
        for k, v in entries:
            r[k] += v
        rows.append(r)
        rhs.append(value)

    first = [(i * length, 1.0) for i in range(cars)]
    target = max(view.target, 0.0)  # A7 of asa_programme
    row([*first, (e, -1.0)], target)  # total - target <= e
    row([*((k, -v) for k, v in first), (e, -1.0)], -target)  # target - total <= e
    for i in range(cars):
        row([(i * length + j, 1.0) for j in range(length)], room[i])
    for k, i in enumerate(live):
        row([*((i * length + j, -1.0) for j in range(length)), (u[k], -1.0)], -gap[i])
    for j in range(1, length):
        m = centre[hours[j] - 1]
        hour = [(i * length + j, 1.0) for i in range(cars)]
        row([*hour, (z[j - 1], -1.0)], m)
        row([*((k, -v) for k, v in hour), (z[j - 1], -1.0)], -m)
    costs = [np.zeros(n_var) for _ in range(4)]
    costs[0][e] = 1.0
    costs[1][u] = 1.0
    costs[2][z] = 1.0
    for i in range(cars):
        for j, s in enumerate(hours):
            costs[3][i * length + j] = -(view.n_service - s + 1)
    for k, optimum in enumerate(kept):
        rows.append(costs[k].copy())
        rhs.append(optimum + STAGE_TOLERANCE)
    bounds = [(0.0, upper.reshape(-1)[k]) for k in range(n_p)] + [(0.0, None)] * (n_var - n_p)
    found = linprog(costs[stage], A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds)
    assert found.status == 0, found.message
    return found.fun


def test_stages_1_and_2_reach_their_closed_forms():
    inside = 0
    for view, centre, n_recovery in VIEWS:
        found = plan_hour(view, centre, n_recovery, time_limit=300.0)
        assert found.optimal
        stage_1, stage_2 = _closed_stage_1_and_2(view, n_recovery)
        assert abs(found.optima[0] - stage_1) <= AGREE
        assert abs(found.optima[1] - stage_2) <= AGREE
        inside += stage_2 > 0.1
    assert inside >= 30, "the sample must hold hours with unmet need"


def test_stages_3_and_4_reach_the_optima_of_programmes_written_apart():
    for view, centre, n_recovery in VIEWS:
        found = plan_hour(view, centre, n_recovery, time_limit=300.0)
        kept = list(found.optima[:2])
        third = _dense_stage(view, centre, n_recovery, 2, kept)
        assert abs(found.optima[2] - third) <= AGREE
        fourth = _dense_stage(view, centre, n_recovery, 3, [*kept, found.optima[2]])
        assert abs(found.optima[3] - fourth) <= AGREE * view.n_service


def test_the_final_plan_keeps_every_earlier_optimum():
    for view, centre, n_recovery in VIEWS:
        found = plan_hour(view, centre, n_recovery, time_limit=300.0)
        measured = _measure(view, centre, n_recovery, found.powers)
        optima = np.array(found.optima)
        assert np.all(measured[:3] <= optima[:3] + SLACK * view.n_cars * view.n_service)
        assert abs(measured[3] - optima[3]) <= AGREE * view.n_service
