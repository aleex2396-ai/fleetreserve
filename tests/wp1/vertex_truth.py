"""Ground truth and random fleets for the T1 tests: is a box or band inside the fleet set.

A box is a polytope, so it lies inside a convex set exactly when every vertex does. A vertex
`x` (one value per hour) is inside the fleet set when the cars can split it: each car `i` draws
`u[i][t]` in `[low, high]` on its own hours and 0 elsewhere, its total lies in
`[e_low, e_high]`, and the cars sum to `x[t]` in every hour. That is one feasibility
programme per vertex, solved through `solve_lp`. Any outcome but optimal or infeasible stops
the test, so a timeout or solver error is never read as an answer.

A box may sit on a subset of the hours (`box_hours`, the service hours). The cars then draw
freely in every other hour (the recovery hours), within their own limits: only the box hours get
a sum row.

For the band (T1(c)) the same idea gives the largest scale. A band is a polytope, so it lies
inside the fleet set exactly when every vertex does, and its vertices scale linearly with the
scale `lam`: the vertices are `centre + lam * v` for fixed directions `v`. The fleet set is
convex and holds the centre, so the scales at which `centre + lam * v` is inside form an
interval `[0, lam_v]`; one programme per direction (maximise `lam` with the cars splitting
`centre + lam * v`) gives `lam_v`, and the band's largest scale is the smallest `lam_v`.
"""

import itertools

import numpy as np

from fleetreserve.gpoly import Car
from fleetreserve.lp import BOX_SECONDS, solve_lp

N_HOURS = 7  # hours in the random fleets below; the service hours are the first few of them


def random_car(rng, n_hours=N_HOURS):
    """A car with a run of hours inside `0..n_hours-1`, one- or two-way, with a positive floor."""
    start = rng.randint(0, n_hours - 1)
    end = rng.randint(start + 1, n_hours)
    n = end - start
    rate = rng.choice([1, 2, 3])
    low = 0 if rng.random() < 0.5 else -rate
    e_high = rng.uniform(0.3, 1.0) * rate * n
    e_low = rng.uniform(0.1, 0.9) * e_high  # a positive floor (the protected energy)
    return Car(frozenset(range(start, end)), low, rate, e_low, e_high)


def random_split(rng, car):
    """Powers of one car, in its hours, that meet all its limits (a point of its own set)."""
    n = len(car.hours)
    e_mid = (max(car.e_low, car.low * n) + min(car.e_high, car.high * n)) / 2
    steady = {t: e_mid / n for t in car.hours}
    random_power = {t: rng.uniform(car.low, car.high) for t in car.hours}
    total = sum(random_power.values())
    clamped = min(max(total, car.e_low), car.e_high)
    # Mix towards the steady split far enough to bring the total inside, then 10% of the rest
    # of the way, so that the split is strictly inside the car's energy limits.
    needed = 0.0 if total == clamped else (total - clamped) / (total - e_mid)
    share = needed + 0.1 * (1 - needed)
    return {t: (1 - share) * random_power[t] + share * steady[t] for t in car.hours}


def centre_and_spread(rng, cars, box_hours):
    """A point of the fleet set seen on `box_hours`, and per hour the cars' total power range."""
    centre = {t: 0.0 for t in box_hours}
    spread = {t: 0.0 for t in box_hours}
    for car in cars:
        powers = random_split(rng, car)
        for t in box_hours:
            if t in car.hours:
                centre[t] += powers[t]
                spread[t] += car.high - car.low
    return centre, spread


def _car_rows(cars, n_hours, extra_columns=0):
    """Energy rows and power bounds of the cars; variables `u[i][t]`, then `extra_columns` more."""
    n = len(cars)
    width = n * n_hours + extra_columns
    a_ub = np.zeros((2 * n, width))
    b_ub = np.zeros(2 * n)
    bounds = []
    for i, car in enumerate(cars):
        for t in range(n_hours):
            bounds.append((car.low, car.high) if t in car.hours else (0.0, 0.0))
        a_ub[2 * i, i * n_hours : (i + 1) * n_hours] = 1.0
        b_ub[2 * i] = car.e_high
        a_ub[2 * i + 1, i * n_hours : (i + 1) * n_hours] = -1.0
        b_ub[2 * i + 1] = -car.e_low
    return a_ub, b_ub, bounds


def _solve(a_ub, b_ub, a_eq, b_eq, cost, bounds):
    result = solve_lp(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=bounds,
        time_limit=BOX_SECONDS,
        method="simplex",
    )
    if result.outcome not in ("optimal", "infeasible"):
        raise AssertionError(f"programme ended {result.outcome}: {result.message}")
    return result


def vertex_in_fleet(x, cars, n_hours, box_hours=None):
    """True when the cars can split the totals `x` in `box_hours` (default: all hours).

    `x` has one value per box hour, in the order of `box_hours`; the cars have the `Car` fields.
    """
    hours = list(range(n_hours)) if box_hours is None else list(box_hours)
    n = len(cars)
    a_ub, b_ub, bounds = _car_rows(cars, n_hours)
    a_eq = np.zeros((len(hours), n * n_hours))
    for i in range(n):
        for k, t in enumerate(hours):
            a_eq[k, i * n_hours + t] = 1.0
    return _solve(a_ub, b_ub, a_eq, x, np.zeros(n * n_hours), bounds).outcome == "optimal"


def largest_scale_along(centre, direction, cars, n_hours, box_hours):
    """Largest `lam >= 0` with the cars able to split `centre + lam * direction` in `box_hours`.

    `centre` and `direction` have one value per box hour, in the order of `box_hours`. The
    direction must be non-zero (otherwise the scale is unbounded and the programme is refused
    by the solver's outcome check).
    """
    hours = list(box_hours)
    n = len(cars)
    a_ub, b_ub, bounds = _car_rows(cars, n_hours, extra_columns=1)
    a_eq = np.zeros((len(hours), n * n_hours + 1))
    for i in range(n):
        for k, t in enumerate(hours):
            a_eq[k, i * n_hours + t] = 1.0
    a_eq[:, -1] = -np.asarray(direction, dtype=float)
    cost = np.zeros(n * n_hours + 1)
    cost[-1] = -1.0
    result = _solve(a_ub, b_ub, a_eq, centre, cost, [*bounds, (0.0, None)])
    if result.outcome != "optimal":
        raise AssertionError("the centre is not inside the fleet set")
    return float(result.x[-1])


def box_inside_by_vertices(lo, hi, cars, n_hours, box_hours=None):
    """True when every vertex of the box [lo, hi] can be split among the cars.

    `lo` and `hi` have one value per box hour, in the order of `box_hours` (default: all hours).
    """
    corners = itertools.product(*zip(lo, hi, strict=True))
    return all(vertex_in_fleet(corner, cars, n_hours, box_hours) for corner in corners)
