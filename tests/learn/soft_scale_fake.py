"""Seeded made-up forecast days and independent linear programmes for the tests of `soft_scale`.

`forecast_day(rng)`: a `day.Day` of 1 to 4 cars whose forecasts (`day.ForecastCar`) have
fractional departures, so a car's last hour is a share of an hour. The day has 1 to 4 service hours
and 0 to 2 recovery hours; each forecast departure is drawn from 0.3 hours to the day's last hour,
its rate from 1 to 3 kW, its cap from 0 to 1.2 times what the rate allows by the departure, and its
need from 0 to the most the set allows (`need <= cap + recovery supply` and `need <= rate times
departure`), with a share of "tight" needs (exactly that most) and "empty" cars (cap and need 0).
Every forecast set is non-empty (`forecast_coverage.forecast_set_empty` is False). The true cars
the `Day` needs beside its forecasts are placeholders (room 0) and are not used.

`generic=True` draws "drawn" cars only (no tight need and no empty car), so no limit sits on a
kink of the set functions by construction.

`fleet_day(rng, cars, n_service, whole_hours)`: a day of `cars` drawn cars as above, with
fractional departures or, with `whole_hours`, departures on whole hours from 1 to the day's last
hour (no fractional last hour; every share is then at 0 or 1, on a kink).

`point_of(day, rng)`: a point of the fleet's forecast set on the service hours, the sum of one
feasible plan per car (each car draws the same share of its power limit in hours 0 to n, the
share drawn so that its window energy lies between its least and its most).

`shape(rng, centre, n)`: widths up and down from [0, 1] kW per hour, each zero with probability
0.2, never all zero.

The programmes use the code base's own rows of each car (`carsets.forecast_rows`), never `p` or
`b`, and are solved through `lp.solve_lp` with a time limit; any outcome but optimal stops the
test.

- `car_extreme(rows, subset, sign)`: the most (`sign` +1) or least (-1) energy one car can draw
  in the service hours `subset`, hour 0 free.
- `largest_step(cars_rows, centre, direction)`: the largest `lam` with `centre + lam direction`
  drawable by the fleet in the service hours (one programme over every car's powers).
- `vertex_scale(cars_rows, centre, w_up, w_down)`: the smallest largest step over the box's
  vertex directions (`+w_up` or `-w_down` in each hour), skipping the zero direction. The set is
  convex and holds the centre, so this is the largest scale of the box.
"""

import datetime
import itertools
import math

import numpy as np

from fleetreserve.carsets import forecast_rows
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.forecast_coverage import forecast_set_empty
from fleetreserve.lp import OTHER_SECONDS, solve_lp

DATE = datetime.date(2020, 1, 1)


def forecast_car(
    rng, n_service: int, n_recovery: int, generic: bool = False, whole_hours: bool = False
) -> ForecastCar:
    n_hours = 1 + n_service + n_recovery
    departure = float(rng.randint(1, n_hours)) if whole_hours else rng.uniform(0.3, n_hours)
    rate = rng.uniform(1.0, 3.0)
    weights = [0, 0, 1] if generic else [2, 1, 7]
    kind = rng.choices(["tight", "empty", "drawn"], weights=weights)[0]
    if kind == "empty":
        return ForecastCar(departure, rate, 0.0, 0.0)
    cap = rng.uniform(0.0, 1.2) * rate * departure
    probe = ForecastCar(departure, rate, cap, 0.0)
    supply = probe.recovery_supply(n_service, n_recovery)
    most = min(cap + supply, rate * departure)
    need = most if kind == "tight" else rng.uniform(0.0, 1.0) * most
    while need > 0 and forecast_set_empty(
        ForecastCar(departure, rate, cap, need), n_service, n_recovery
    ):
        need = math.nextafter(need, 0.0)  # step below rounding error
    return ForecastCar(departure, rate, cap, need)


def forecast_day(rng, n_service: int | None = None, generic: bool = False) -> Day:
    n = rng.randint(1, 4) if n_service is None else n_service
    n_recovery = rng.randint(0, 2)
    forecasts = [forecast_car(rng, n, n_recovery, generic) for _ in range(rng.randint(1, 4))]
    for f in forecasts:
        assert not forecast_set_empty(f, n, n_recovery)
    cars = [TrueCar(max(1, math.ceil(f.departure)), f.rate, 0.0) for f in forecasts]
    return Day("made-up", DATE, n, n_recovery, 0.0, tuple(cars), tuple(forecasts), 0.1)


def fleet_day(rng, cars: int, n_service: int, whole_hours: bool) -> Day:
    """A day of `cars` drawn cars (`generic`: no tight need, no empty car), with fractional
    departures or, with `whole_hours`, departures on whole hours from 1 to the day's last hour."""
    n_recovery = rng.randint(0, 2)
    forecasts = [
        forecast_car(rng, n_service, n_recovery, generic=True, whole_hours=whole_hours)
        for _ in range(cars)
    ]
    trues = [TrueCar(max(1, math.ceil(f.departure)), f.rate, 0.0) for f in forecasts]
    return Day("made-up", DATE, n_service, n_recovery, 0.0, tuple(trues), tuple(forecasts), 0.1)


def car_rows(day: Day):
    return [forecast_rows(f, day.n_service, day.n_recovery) for f in day.forecasts]


def point_of(day: Day, rng) -> list[float]:
    n = day.n_service
    total = np.zeros(n)
    for f in day.forecasts:
        high = np.array([f.power_cap(t) for t in range(n + 1)])
        reach = high.sum()
        least = max(f.window_need(n, day.n_recovery), 0.0)
        most = min(f.cap, reach)
        if reach > 0:
            total += high[1:] * rng.uniform(least, most) / reach
    return list(total)


def shape(rng, n: int) -> tuple[list[float], list[float]]:
    while True:
        up = [0.0 if rng.random() < 0.2 else rng.random() for _ in range(n)]
        down = [0.0 if rng.random() < 0.2 else rng.random() for _ in range(n)]
        if any(up) or any(down):
            return up, down


def _solve(cost, a_ub, b_ub, a_eq=None, b_eq=None):
    result = solve_lp(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(None, None)] * len(cost),
        time_limit=OTHER_SECONDS,
        method="simplex",
    ).require_optimal()
    return result


def car_extreme(rows, subset, sign: float) -> float:
    """The most (`sign` 1) or least (`sign` -1) of `sum_{t in subset} u_t` on the car's rows."""
    cost = np.zeros(rows.n_service + 1)
    for t in subset:
        cost[t] = -sign
    result = _solve(cost, rows.coefficients, rows.bounds)
    return -sign * float(result.objective)


def largest_step(cars_rows, centre, direction) -> float:
    n = cars_rows[0].n_service
    width = n + 1
    k = len(cars_rows)
    size = k * width + 1  # every car's powers, then lam
    blocks = []
    bounds = []
    for i, rows in enumerate(cars_rows):
        a = np.zeros((rows.bounds.size, size))
        a[:, i * width : (i + 1) * width] = rows.coefficients
        blocks.append(a)
        bounds.append(rows.bounds)
    a_eq = np.zeros((n, size))
    for t in range(1, n + 1):
        for i in range(k):
            a_eq[t - 1, i * width + t] = 1.0
        a_eq[t - 1, -1] = -float(direction[t - 1])
    cost = np.zeros(size)
    cost[-1] = -1.0
    result = _solve(cost, np.vstack(blocks), np.concatenate(bounds), a_eq, np.array(centre))
    return float(result.x[-1])


def vertex_scale(cars_rows, centre, w_up, w_down) -> float:
    n = len(centre)
    best = math.inf
    for signs in itertools.product((1, -1), repeat=n):
        direction = [w_up[t] if s > 0 else -w_down[t] for t, s in enumerate(signs)]
        if not any(direction):
            continue
        best = min(best, largest_step(cars_rows, centre, direction))
    return best
