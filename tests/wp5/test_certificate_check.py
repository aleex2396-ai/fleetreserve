"""Plan section 5, T2's check on made-up days: a policy certified on forecast sets, on corner paths.

Sample: `N_DAYS` seeded made-up days of 1 to 4 cars and 2 to 4 service hours, rho cycling through
0, 0.1 and 0.2. Each car leaves in hour 2 to 3 hours after the window, draws 1, 2 or 3 kW and has
a room of `U(0.1, 0.6)` times its rate times its hours, so that a conservative forecast still has
a non-empty set. Each day is given forecasts of two kinds and certified (causal structure) for the
hourly and the block product:

- conservative (plan T2(c1)): `d_hat <= d`, `r_hat <= r`, `e_cap <= e`, `e_need >= (1 - rho) e`,
  drawn as `d_hat = d - U(0, 0.5)`, `r_hat = r U(0.85, 1)`,
  `e_cap = need + U(0.3, 1) (e - need)`, `e_need = need + U(0, 0.1) (e_cap - need)`;
- not conservative: `d_hat = min(d + U(0.5, 2), last hour)`, `r_hat = r U(1.2, 1.6)`,
  `e_cap = e U(1, 1.3)`, `e_need = need U(0.6, 1)`.

Where a forecast set is empty the certificate is infeasible; such days are counted and left out.
Checks, on every one of the `2^n` corner paths of each certified policy:

1. The closed-form worst case `a g + |a G|_1` of every forecast row and every true row equals the
   largest `a . u` over the corner paths.
2. Conservative: no forecast row and no true row is violated, so every car receives its protected
   energy, and the fleet's powers meet every request exactly (T2(c1), (c2)).
3. Not conservative: on at least `MIN_VIOLATING` certified days some true row is violated on some
   corner path (the check can fail).
"""

import dataclasses
import functools
import random

import numpy as np
from made_up_days import DATE, RHOS

from fleetreserve import products
from fleetreserve.carsets import day_forecast_rows, true_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.day import Day, ForecastCar, TrueCar, covering_recovery_hours
from fleetreserve.products import corner_paths

SEED = 71
N_DAYS = 60
MIN_CERTIFIED = 100  # certified (day, product) cases of each kind, of 2 * N_DAYS
MIN_POSITIVE = 50  # conservative certified cases with a positive value
MIN_VIOLATING = 20
TOLERANCE = 1e-7


def _day(rng, k):
    n = rng.randint(2, 4)
    cars = []
    for _ in range(rng.randint(1, 4)):
        rate = float(rng.choice([1, 2, 3]))
        departure = rng.randint(2, n + 3)
        cars.append(TrueCar(departure, rate, rng.uniform(0.1, 0.6) * rate * departure))
    return Day("made-up", DATE, n, covering_recovery_hours(cars, n), RHOS[k % 3], tuple(cars))


def _conservative(rng, car, rho):
    need = car.need(rho)
    cap = need + rng.uniform(0.3, 1) * (car.room - need)
    return ForecastCar(
        car.departure - rng.uniform(0, 0.5),
        car.rate * rng.uniform(0.85, 1.0),
        cap,
        need + rng.uniform(0, 0.1) * (cap - need),
    )


def _optimistic(rng, car, rho, n_hours):
    return ForecastCar(
        min(car.departure + rng.uniform(0.5, 2.0), float(n_hours)),
        car.rate * rng.uniform(1.2, 1.6),
        car.room * rng.uniform(1.0, 1.3),
        car.need(rho) * rng.uniform(0.6, 1.0),
    )


@functools.cache
def _cases():
    rng = random.Random(SEED)
    days = [_day(rng, k) for k in range(N_DAYS)]
    cases = {"conservative": [], "optimistic": []}
    infeasible = {"conservative": 0, "optimistic": 0}
    for day in days:
        for kind in cases:
            if kind == "conservative":
                forecasts = tuple(_conservative(rng, c, day.rho) for c in day.cars)
            else:
                forecasts = tuple(_optimistic(rng, c, day.rho, day.n_hours) for c in day.cars)
            forecast_day = dataclasses.replace(day, forecasts=forecasts, kappa=0.1)
            rows = day_forecast_rows(forecast_day)
            for product in (products.hourly(), products.block()):
                result = certify(rows, product, "causal", shape_rule=ANY_OPTIMAL)
                if result.outcome != "optimal":
                    assert result.outcome == "infeasible", result
                    infeasible[kind] += 1
                    continue
                cases[kind].append((forecast_day, rows, true_rows(day), result))
    return cases, infeasible


def _corner_values(policy, rows):
    """`(paths, rows)`: `a . u` of every row on every corner path."""
    powers = policy.powers(corner_paths(policy.n_service))  # (paths, cars, hours)
    return np.einsum("kt,pkt->pk", rows.coefficients, powers[:, rows.car, :])


def test_the_sample_has_certified_days_of_both_kinds():
    cases, infeasible = _cases()
    print(
        {kind: len(c) for kind, c in cases.items()},
        "certified; infeasible (an empty forecast set):",
        infeasible,
    )
    for kind in cases:
        assert len(cases[kind]) + infeasible[kind] == 2 * N_DAYS
        assert len(cases[kind]) >= MIN_CERTIFIED
    assert sum(r.value > 1e-6 for *_, r in cases["conservative"]) >= MIN_POSITIVE


def test_closed_form_worst_case_equals_the_largest_corner_value():
    cases, _ = _cases()
    checked = 0
    for kind in cases:
        for _, forecast, truth, result in cases[kind]:
            for rows in (forecast, truth):
                largest = _corner_values(result.policy, rows).max(axis=0)
                closed = result.policy.worst_case(rows)
                np.testing.assert_allclose(closed, largest, rtol=0, atol=TOLERANCE)
                checked += rows.n_rows
    print(f"closed-form worst case equals the corner maximum on {checked} rows")
    assert checked > 0


def test_conservative_forecasts_violate_no_true_row_and_meet_every_request():
    cases, _ = _cases()
    for day, forecast, truth, result in cases["conservative"]:
        policy = result.policy
        for rows in (forecast, truth):
            slack = rows.bounds - _corner_values(policy, rows)
            assert slack.min() >= -TOLERANCE, (day, rows.kinds[slack.min(axis=0) < -TOLERANCE])
        paths = corner_paths(policy.n_service)
        delivered = policy.powers(paths).sum(axis=1)[:, 1:]
        np.testing.assert_allclose(delivered, policy.request(paths), rtol=0, atol=TOLERANCE)


def test_forecasts_that_are_not_conservative_can_violate_true_rows():
    cases, _ = _cases()
    violating = 0
    for _, forecast, truth, result in cases["optimistic"]:
        assert (forecast.bounds - _corner_values(result.policy, forecast)).min() >= -TOLERANCE
        violating += (truth.bounds - _corner_values(result.policy, truth)).min() < -TOLERANCE
    print(f"true rows violated on {violating} of {len(cases['optimistic'])} optimistic days")
    assert violating >= MIN_VIOLATING
