"""Independent linear programmes over a fleet's full set, every hour written out.

Ground truth for the tests of `exact`: each car has one power variable per hour it is present, in
hour 0, the service hours and the recovery hours alike, with its own limits as rows. A
`gpoly.Car` (one-way) has its power bounds and its total energy limits; a `day.StoredEnergyCar`
(two-way) has its power bounds, a row for its stored energy after every hour (between its lower
and upper limit) and a row for its stored energy at the end (at least its need). Nothing here
uses the set functions `p`, `b`. Every programme is solved through `toy_values._Model`, which calls
`solve_lp` with a time limit and stops the test on any outcome but optimal.

The service hours are 1 to `n_service`; values per service hour are lists in that order.
"""

import random
from itertools import product

import numpy as np
from toy_values import _car_energy_rows, _Model

from fleetreserve.day import StoredEnergyCar
from fleetreserve.gpoly import Car
from fleetreserve.lp import OTHER_SECONDS, solve_lp


def _add_car(m, car):
    """One car's power variables (by hour) and rows."""
    if isinstance(car, Car):
        powers = {t: m.var(car.low, car.high) for t in sorted(car.hours)}
        _car_energy_rows(m, car, list(powers.values()))
        return powers
    if isinstance(car, StoredEnergyCar):
        powers = {t: m.var(car.low, car.high) for t in range(car.n_hours)}
        for t in range(car.n_hours):
            running = [(powers[s], 1.0) for s in range(t + 1)]
            m.le(running, car.stored_high)
            m.ge(running, car.stored_low)
        m.ge([(powers[t], 1.0) for t in range(car.n_hours)], car.need)
        return powers
    raise TypeError(f"not a car: {car!r}")


def _maximiser(m, terms):
    """The solution of `m.maximise(terms)` (that method returns only the objective)."""
    n = len(m.bounds)
    cost = np.zeros(n)
    for j, c in terms:
        cost[j] -= c
    a_ub = np.zeros((len(m.ub), n))
    for r, (row, _) in enumerate(m.ub):
        for j, c in row:
            a_ub[r, j] += c
    a_eq = np.zeros((len(m.eq), n))
    for r, (row, _) in enumerate(m.eq):
        for j, c in row:
            a_eq[r, j] += c
    result = solve_lp(
        cost,
        A_ub=a_ub if m.ub else None,
        b_ub=np.array([rhs for _, rhs in m.ub]) if m.ub else None,
        A_eq=a_eq if m.eq else None,
        b_eq=np.array([rhs for _, rhs in m.eq]) if m.eq else None,
        bounds=m.bounds,
        time_limit=OTHER_SECONDS,
        method="simplex",
    ).require_optimal()
    return result.x


def _fleet(m, cars):
    return [_add_car(m, car) for car in cars]


def _hour_terms(plans, t):
    return [(plan[t], 1.0) for plan in plans if t in plan]


def extreme(cars, subset, sign):
    """`max sign * u(A)` over the full fleet set, `A = subset` (service hours)."""
    m = _Model()
    plans = _fleet(m, cars)
    return m.maximise([(j, sign * c) for t in subset for j, c in _hour_terms(plans, t)])


def lower_and_upper(cars, subset):
    """`(min u(A), max u(A))` over the full fleet set."""
    return -extreme(cars, subset, -1.0), extreme(cars, subset, 1.0)


def largest_step(cars, n_service, start, direction, cap=None):
    """Largest `s` in `[0, cap]` with the cars able to draw `start + s * direction` in S."""
    m = _Model()
    step = m.var(0.0, cap)
    plans = _fleet(m, cars)
    for k, t in enumerate(range(1, n_service + 1)):
        m.equal(_hour_terms(plans, t) + [(step, -direction[k])], start[k])
    return m.maximise([(step, 1.0)])


def random_point(rng: random.Random, cars, n_service, n_mix=3):
    """A point of the fleet set seen on S: a random mix of `n_mix` vertices, each the optimum of
    a random objective over every hour of every car."""
    weights = [rng.random() + 0.05 for _ in range(n_mix)]
    total = sum(weights)
    point = [0.0] * n_service
    for weight in weights:
        m = _Model()
        plans = _fleet(m, cars)
        objective = [(j, rng.uniform(-1, 1)) for plan in plans for j in plan.values()]
        values = _maximiser(m, objective)
        for k, t in enumerate(range(1, n_service + 1)):
            point[k] += weight / total * sum(values[j] for j, _ in _hour_terms(plans, t))
    return point


def attainable(cars, n_service, inner, point, tolerance=1e-7):
    """True when the cars can draw `point` in S; `inner` is a point they can draw."""
    direction = [p - q for p, q in zip(point, inner, strict=True)]
    if all(abs(d) < 1e-12 for d in direction):
        return True
    return largest_step(cars, n_service, inner, direction, cap=1.0) >= 1.0 - tolerance


def free_box_by_paths(cars, n_service):
    """Largest `sum(hi - lo)` with every corner path met by a plan that knows the whole path."""
    m = _Model()
    hours = range(1, n_service + 1)
    lo = {t: m.var(None, None) for t in hours}
    hi = {t: m.var(None, None) for t in hours}
    for t in hours:
        m.le([(lo[t], 1.0), (hi[t], -1.0)], 0.0)
    for path in product((-1, 1), repeat=n_service):
        plans = _fleet(m, cars)
        for t, z in zip(hours, path, strict=True):
            m.equal(_hour_terms(plans, t) + [(lo[t] if z < 0 else hi[t], -1.0)], 0.0)
    return m.maximise([(hi[t], 1.0) for t in hours] + [(lo[t], -1.0) for t in hours])


def block_by_paths(cars, n_service):
    """Largest block half-width `lam`, centre free, every corner path met by a path-known plan."""
    m = _Model()
    hours = range(1, n_service + 1)
    lam = m.var(0.0, None)
    centre = {t: m.var(None, None) for t in hours}
    for path in product((-1, 1), repeat=n_service):
        plans = _fleet(m, cars)
        for t, z in zip(hours, path, strict=True):
            m.equal(_hour_terms(plans, t) + [(centre[t], -1.0), (lam, -float(z))], 0.0)
    return m.maximise([(lam, 1.0)])
