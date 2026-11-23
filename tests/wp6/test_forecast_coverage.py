"""Coverage and empty forecast sets on hand-built cars; the row test against the closed-form
conditions and against a feasibility programme."""

import random

import numpy as np
import pytest

from fleetreserve import forecast_coverage as fc
from fleetreserve.carsets import forecast_rows
from fleetreserve.day import ForecastCar
from fleetreserve.lp import BOX_SECONDS, solve_lp


def test_empty_by_the_stored_energy_cap():
    # Hours 0..4, S = {1, 2}: shares 1, 1, 0.5, 0, 0. Need 4 > cap 3 + recovery supply 0.
    car = ForecastCar(2.5, 2.0, 3.0, 4.0)
    assert fc.forecast_set_empty(car, 2, 2)
    assert fc.empty_conditions(car, 2, 2) == (True, False)
    assert fc.largest_window_energy(forecast_rows(car, 2, 2)) == pytest.approx(3.0)


def test_empty_by_the_energy_before_departure():
    # Shares 1, 1, 1, 1, 0.5 at rate 1: 4.5 kWh before d_hat, 1.5 of it in recovery.
    assert not fc.forecast_set_empty(ForecastCar(4.5, 1.0, 10.0, 4.5), 2, 2)
    car = ForecastCar(4.5, 1.0, 10.0, 5.0)
    assert fc.forecast_set_empty(car, 2, 2)
    assert fc.empty_conditions(car, 2, 2) == (False, True)


def test_check_car_by_hand():
    car = ForecastCar(4.5, 1.0, 10.0, 4.5)
    check = fc.check_car(car, 4, 4.6, 1.2, 4.5, 0.0, 2, 2)
    assert (check.departure, check.departure_exact, check.departure_floor) == (False, True, True)
    assert (check.rate, check.cap, check.need) == (True, False, True)
    assert not check.all_hold and not check.empty and not check.inside_window
    held = fc.check_car(ForecastCar(2.0, 1.0, 1.5, 1.5), 3, 3.2, 1.0, 1.5, 0.0, 2, 2)
    assert held.all_hold and held.inside_window and not held.empty
    tight = fc.check_car(ForecastCar(2.0, 1.0, 1.5, 1.5), 3, 3.2, 1.0, 1.6, 0.2, 2, 2)
    assert tight.all_hold  # 1.5 >= 0.8 * 1.6 and 1.5 <= 1.6
    short = fc.check_car(ForecastCar(2.0, 1.0, 1.0, 1.5), 3, 3.2, 1.0, 1.5, 0.0, 2, 2)
    assert short.empty and (short.empty_a, short.empty_b) == (True, False)


def _feasible_by_programme(car, n_service, n_recovery):
    rows = forecast_rows(car, n_service, n_recovery)
    result = solve_lp(
        np.zeros(n_service + 1),
        A_ub=rows.coefficients,
        b_ub=rows.bounds,
        bounds=(None, None),
        time_limit=BOX_SECONDS,
        method="simplex",
    )
    assert result.outcome in ("optimal", "infeasible")
    return result.outcome == "optimal"


def test_row_test_matches_the_closed_form_and_a_feasibility_programme():
    rng = random.Random(11)
    empty_seen = full_seen = 0
    for _ in range(300):
        n_service = rng.randint(1, 4)
        n_recovery = rng.randint(0, 3)
        car = ForecastCar(
            rng.uniform(0.0, 1 + n_service + n_recovery),
            rng.uniform(0.5, 3.0),
            rng.uniform(0.0, 8.0),
            rng.uniform(0.0, 10.0),
        )
        empty = fc.forecast_set_empty(car, n_service, n_recovery)
        a, b = fc.empty_conditions(car, n_service, n_recovery)
        margin = abs(car.window_need(n_service, n_recovery)) < 1e-6
        if not margin:
            assert empty == (a or b)
            assert empty == (not _feasible_by_programme(car, n_service, n_recovery))
        empty_seen += empty
        full_seen += not empty
    assert empty_seen > 30 and full_seen > 30
