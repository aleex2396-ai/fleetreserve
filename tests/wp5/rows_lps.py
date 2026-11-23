"""Small dense linear programmes over a fleet given as rows (`carsets.FleetRows`), for the tests.

Each car has one free power variable per hour 0 to `n_service` and its rows `a . u <= b` written
out; nothing here uses the sparse programmes under test. Solved through `toy_values._Model`, which
calls `solve_lp` with a time limit and stops the test on any outcome but optimal.

- `rows_extreme(rows, subset, sign)`: `max sign * u(A)` over the fleet's rows, `A = subset` of the
  service hours; hour 0 is free (the projection that `exact`'s `p(A)`, `b(A)` describe).
- `committed_hindsight_value(rows)`: the hourly product's largest value `sum(hi - lo)` met on every
  corner path by plans that know the whole path, except that every car's hour-0 power is one
  number shared by every path (decided at the decision time).
"""

from itertools import product

from toy_values import _Model


def _car_powers(m, rows):
    n = rows.n_service
    return {(i, t): m.var(None, None) for i in range(rows.n_cars) for t in range(n + 1)}


def _write_rows(m, rows, power):
    for k in range(rows.n_rows):
        i = int(rows.car[k])
        terms = [
            (power(i, t), float(rows.coefficients[k, t]))
            for t in range(rows.n_service + 1)
            if rows.coefficients[k, t] != 0
        ]
        m.le(terms, float(rows.bounds[k]))


def rows_extreme(rows, subset, sign):
    m = _Model()
    u = _car_powers(m, rows)
    _write_rows(m, rows, lambda i, t: u[i, t])
    return m.maximise([(u[i, t], sign) for i in range(rows.n_cars) for t in subset])


def committed_hindsight_value(rows):
    n = rows.n_service
    m = _Model()
    hours = range(1, n + 1)
    lo = {t: m.var(None, None) for t in hours}
    hi = {t: m.var(None, None) for t in hours}
    for t in hours:
        m.le([(lo[t], 1.0), (hi[t], -1.0)], 0.0)
    first = {i: m.var(None, None) for i in range(rows.n_cars)}
    for path in product((-1, 1), repeat=n):
        u = {(i, t): m.var(None, None) for i in range(rows.n_cars) for t in hours}
        _write_rows(m, rows, lambda i, t, u=u: first[i] if t == 0 else u[i, t])
        for t, z in zip(hours, path, strict=True):
            m.equal(
                [(u[i, t], 1.0) for i in range(rows.n_cars)] + [(lo[t] if z < 0 else hi[t], -1.0)],
                0.0,
            )
    return m.maximise([(hi[t], 1.0) for t in hours] + [(lo[t], -1.0) for t in hours])
