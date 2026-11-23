"""Plan sections 1, 2 and 9: each car's set as rows over hours 0 to n, recovery eliminated.

1. Hand-worked rows of one true car (one-way, two-way) and one forecast car.
2. Projection: on `N_DAYS` seeded made-up days (`made_up_days.py`), each taken one-way and two-way
   (budget cycling through `BUDGETS`), the fleet set the rows describe, seen on the service hours
   with hour 0 free, has the `p(A)`, `b(A)` of `exact` for every non-empty subset `A` of S: the
   least and the most energy in `A` by one small programme each (`rows_lps.rows_extreme`).
   `N_SUBSETS` subsets in each direction.
3. The rows of a `gpoly.Car` (toy fleets, no hour 0) have the projection `gpoly.subset_functions`
   gives on the service hours, on `N_TOY` toy fleets, one-way and two-way.
"""

import random

import numpy as np
import pytest
from made_up_days import made_up_sample
from rows_lps import rows_extreme
from test_t4_conjecture import _fleet

from fleetreserve import gpoly
from fleetreserve.carsets import (
    forecast_rows,
    gpoly_fleet_rows,
    stack,
    true_one_way_rows,
    true_rows,
    true_two_way_rows,
)
from fleetreserve.day import ForecastCar, TrueCar
from fleetreserve.exact import _bits, day_functions

SEED = 21
N_DAYS = 24
BUDGETS = (0.0, 0.5, 2.0)
N_SUBSETS = 504
N_TOY = 30
TOLERANCE = 1e-7

CAR = TrueCar(3, 2.0, 5.0)  # hours 0, 1, 2; protected energy 4 kWh at rho = 0.2


def _by_kind(rows):
    out = {}
    for a, b, kind in zip(rows.coefficients, rows.bounds, rows.kinds, strict=True):
        out.setdefault(kind, []).append((tuple(a), b))
    return out


def test_hand_worked_one_way_rows():
    rows = _by_kind(true_one_way_rows(CAR, 2, 0.2))
    assert [b for _, b in rows["power-high"]] == [2.0, 2.0, 2.0]
    assert [b for _, b in rows["power-low"]] == [0.0, 0.0, 0.0]
    assert rows["stored-high"] == [((1, 0, 0), 5.0), ((1, 1, 0), 5.0), ((1, 1, 1), 5.0)]
    assert rows["need"] == [((-1, -1, -1), pytest.approx(-4.0))]
    assert "stored-low" not in rows
    # One service hour: hour 2 is a recovery hour, supplying 2 kWh of the 4 needed.
    short = _by_kind(true_one_way_rows(CAR, 1, 0.2))
    assert short["need"] == [((-1, -1), pytest.approx(-2.0))]
    # Leaving in hour 1: power only in hour 0, no recovery.
    gone = _by_kind(true_one_way_rows(TrueCar(1, 2.0, 1.0), 2, 0.0))
    assert [b for _, b in gone["power-high"]] == [2.0, 0.0, 0.0]
    assert gone["need"] == [((-1, -1, -1), -1.0)]


def test_hand_worked_two_way_rows():
    rows = _by_kind(true_two_way_rows(CAR, 2, 0.2, 1.0))
    assert [b for _, b in rows["power-high"]] == [2.0, 2.0, 2.0]
    assert [b for _, b in rows["power-low"]] == [2.0, 2.0, 2.0]  # -u_t <= 2
    assert [b for _, b in rows["stored-high"]] == [5.0, 5.0, 5.0]
    assert rows["stored-low"] == [((-1, 0, 0), 1.0), ((-1, -1, 0), 1.0), ((-1, -1, -1), 1.0)]
    assert rows["need"] == [((-1, -1, -1), pytest.approx(-4.0))]


def test_hand_worked_forecast_rows():
    # Forecast departure 2.5: shares 1, 1, 0.5 in hours 0 to 2, 0 in recovery hour 3.
    rows = _by_kind(forecast_rows(ForecastCar(2.5, 2.0, 4.0, 3.0), 2, 1))
    assert [b for _, b in rows["power-high"]] == [2.0, 2.0, 1.0]
    assert [b for _, b in rows["stored-high"]] == [4.0, 4.0, 4.0]
    assert rows["need"] == [((-1, -1, -1), -3.0)]
    # Forecast departure 3.5: recovery hour 3 has share 0.5 and supplies 1 kWh.
    later = _by_kind(forecast_rows(ForecastCar(3.5, 2.0, 4.0, 3.0), 2, 1))
    assert later["need"] == [((-1, -1, -1), -2.0)]


def _compare_projection(rows, lower, upper, n):
    bits = _bits(n)
    checked = 0
    for mask in range(1, 2**n):
        subset = [t + 1 for t in range(n) if bits[mask, t]]
        low = -rows_extreme(rows, subset, -1.0)
        high = rows_extreme(rows, subset, 1.0)
        assert low == pytest.approx(lower[mask], abs=TOLERANCE), (subset, rows)
        assert high == pytest.approx(upper[mask], abs=TOLERANCE), (subset, rows)
        checked += 1
    return checked


def test_rows_projection_equals_exact_set_functions():
    days, _ = made_up_sample(random.Random(SEED), N_DAYS)
    checked = 0
    for k, day in enumerate(days):
        for budget in (None, BUDGETS[k % len(BUDGETS)]):
            functions = day_functions(day, budget)
            checked += _compare_projection(
                true_rows(day, budget), functions.lower, functions.upper, day.n_service
            )
    print(f"projection of the rows equals exact on {checked} subsets in each direction")
    assert checked == N_SUBSETS


def test_gpoly_rows_projection_equals_gpoly_set_functions():
    rng = random.Random(SEED)
    fleets = 0
    for k in range(N_TOY):
        n = 2 + k % 3
        cars = _fleet(rng, n, two_way=k % 2 == 1, in_window=k % 4 < 2)
        lower, upper = gpoly.subset_functions(cars, range(1, n + 1))
        _compare_projection(gpoly_fleet_rows(cars, n), lower, upper, n)
        fleets += 1
    assert fleets == N_TOY


def test_rows_hold_at_a_feasible_plan_and_fail_beyond_it():
    rows = true_one_way_rows(CAR, 2, 0.2)
    fleet = stack([rows])
    assert np.all(fleet.slack([[2.0, 2.0, 0.0]]) >= 0)  # 4 kWh: the protected energy exactly
    assert np.any(fleet.slack([[2.0, 1.0, 0.0]]) < 0)  # 3 kWh: short of it
    assert np.any(fleet.slack([[2.0, 2.0, 2.0]]) < 0)  # 6 kWh: above the room
    assert rows.coefficients.shape == (10, 3)
