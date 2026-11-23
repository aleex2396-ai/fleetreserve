"""The check that cars staying to their bound have no shortfall (`stay_check`; plan section 6,
tier B; T2 (c1)), on made-up days.

1. Which cars are in the check: those on the events E'_i of T2 (c1): the forecast departure at or
   before the true departure, the forecast rate at most the true rate, the forecast cap at most the
   true room, the forecast need at least the true protected energy. With known energy bounds only
   the departure can fail. A car leaving before its bound is out of the check whatever it ends
   with.
2. A car in the check that ends below its protected energy by more than the tolerance is short,
   and the largest shortfall is the largest of those. A car short by no more than the tolerance
   is not short. The counts are per car and per car-path.
3. `gate_day.evaluate` carries the check for every rung (made-up day, real replay: no car in the
   check is short), and a deliberately broken replay (a staying car's final energy lowered on one
   path) is counted, with the car-path, the car and the shortfall.
"""

import dataclasses
import datetime

import numpy as np
import pytest
from gate_fake import gate_cars, matrix_document, write_tables

from fleetreserve import gate_day, stay_check
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_empirical import EmpiricalForecaster
from fleetreserve.gate_config import parse
from fleetreserve.gate_run import settings
from fleetreserve.rungs import RUNGS

DATE = datetime.date(2020, 1, 1)
TOLERANCE = 1e-6


def _day(cars, forecasts, rho=0.0):
    return Day("u", DATE, 2, 3, rho, tuple(cars), tuple(forecasts), 0.1)


# Car 0 leaves at hour 2 and its bound is 3.0 (leaves before the bound); car 1 leaves at hour 4
# and its bound is 3.0 (stays); car 2 leaves at hour 4 and its bound is 4.0 (stays to exactly its
# bound).
CARS = (TrueCar(2, 2.0, 2.0), TrueCar(4, 2.0, 3.0), TrueCar(4, 2.0, 2.0))
FORECASTS = (
    ForecastCar(3.0, 2.0, 2.0, 2.0),
    ForecastCar(3.0, 2.0, 3.0, 3.0),
    ForecastCar(4.0, 2.0, 2.0, 2.0),
)


def test_only_cars_on_the_events_of_their_bounds_are_in_the_check():
    day = _day(CARS, FORECASTS)
    assert stay_check.cars_in_check(day, TOLERANCE) == (1, 2)


def test_the_other_bounds_must_hold_too():
    stays = TrueCar(4, 2.0, 3.0)
    for forecast in (
        ForecastCar(3.0, 2.5, 3.0, 3.0),  # forecast rate above the true rate
        ForecastCar(3.0, 2.0, 3.5, 3.0),  # forecast cap above the true room
        ForecastCar(3.0, 2.0, 3.0, 2.5),  # forecast need below the true protected energy
    ):
        assert stay_check.cars_in_check(_day([stays], [forecast]), TOLERANCE) == ()
    holds = ForecastCar(3.0, 2.0, 3.0, 3.0)
    assert stay_check.cars_in_check(_day([stays], [holds]), TOLERANCE) == (0,)
    # a bound off by no more than the tolerance still holds
    close = ForecastCar(3.0, 2.0 + TOLERANCE / 2, 3.0, 3.0 - TOLERANCE / 2)
    assert stay_check.cars_in_check(_day([stays], [close]), TOLERANCE) == (0,)
    # the protected energy is (1 - rho) times the room
    protected = ForecastCar(3.0, 2.0, 3.0, 2.7)
    assert stay_check.cars_in_check(_day([stays], [protected], 0.1), TOLERANCE) == (0,)
    assert stay_check.cars_in_check(_day([stays], [protected], 0.0), TOLERANCE) == ()


def test_a_car_leaving_before_its_bound_is_not_counted_whatever_it_ends_with():
    day = _day(CARS, FORECASTS)
    finals = np.array([[0.0, 3.0, 2.0]])  # car 0 ends with nothing, cars 1 and 2 are whole
    found = stay_check.check(day, finals, TOLERANCE)
    assert (found.cars, found.car_paths) == (2, 2)
    assert (found.short_cars, found.short_car_paths, found.worst_kwh) == (0, 0, 0.0)


def test_a_staying_car_short_by_more_than_the_tolerance_is_counted():
    day = _day(CARS, FORECASTS)
    finals = np.array(
        [
            [0.0, 3.0, 2.0],  # whole
            [0.0, 2.25, 2.0],  # car 1 short by 0.75
            [0.0, 2.25, 1.5],  # car 1 short by 0.75, car 2 by 0.5
            [0.0, 3.0 - TOLERANCE / 2, 2.0],  # within the tolerance
        ]
    )
    found = stay_check.check(day, finals, TOLERANCE)
    assert (found.cars, found.car_paths) == (2, 8)
    assert (found.short_cars, found.short_car_paths) == (2, 3)
    assert found.worst_kwh == pytest.approx(0.75)


def test_the_check_adds_up_over_days():
    one = stay_check.StayCheck(2, 16, 1, 2, 0.5)
    two = stay_check.StayCheck(3, 24, 0, 0, 0.0)
    total = stay_check.add(one, two)
    assert total == stay_check.StayCheck(5, 40, 1, 2, 0.5)
    assert stay_check.add(stay_check.NONE, one) == one


# --- inside the per-day work ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def prepared_day(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tables")
    write_tables(folder, "u", gate_cars(60))
    days = load_unit_days(folder, "u", rho=0.0, no_fit="cap", small_fleet="keep")
    matrix = parse(matrix_document())
    configuration = matrix.configuration("main-rounded-hour-k0.1-empirical")
    forecaster = EmpiricalForecaster(days, reference="other-days", min_cell=20)
    found = [
        gate_day.prepare_day(t, forecaster, configuration, settings(matrix)) for t in days.days
    ]
    return next(p for p in found if not p.abstains)


def test_evaluate_checks_every_rung_and_finds_no_shortfall_on_a_plan_that_holds(prepared_day):
    found = gate_day.evaluate(prepared_day, 0.5)
    in_check = stay_check.cars_in_check(prepared_day.day, prepared_day.t5_tolerance_kwh)
    assert len(in_check) > 0
    for rung in RUNGS:
        stay = found.rungs[rung].stay
        assert stay.cars == len(in_check)
        assert stay.car_paths == len(in_check) * prepared_day.paths.shape[0]
        assert (stay.short_cars, stay.short_car_paths, stay.worst_kwh) == (0, 0, 0.0)
    assert found.compact().rungs["fixed"].stay == found.rungs["fixed"].stay


def test_a_day_that_does_not_offer_checks_no_car(prepared_day):
    found = gate_day.evaluate(prepared_day, 0.0)
    assert all(found.rungs[r].stay == stay_check.NONE for r in RUNGS)


def test_a_broken_plan_where_a_staying_car_is_short_is_counted(prepared_day, monkeypatch):
    real = gate_day.replay_day
    in_check = stay_check.cars_in_check(prepared_day.day, prepared_day.t5_tolerance_kwh)
    car = in_check[0]
    need = prepared_day.day.cars[car].need(prepared_day.day.rho)
    assert need > 0.3

    def lowered(*args, **kwargs):
        result = real(*args, **kwargs)
        runs = list(result.by_rung["one-way"])
        final = runs[2].final.copy()
        final[car] = need - 0.3
        runs[2] = dataclasses.replace(runs[2], final=final)
        return dataclasses.replace(result, by_rung={**result.by_rung, "one-way": tuple(runs)})

    monkeypatch.setattr(gate_day, "replay_day", lowered)
    found = gate_day.evaluate(prepared_day, 0.5)
    stay = found.rungs["one-way"].stay
    assert (stay.short_cars, stay.short_car_paths) == (1, 1)
    assert stay.worst_kwh == pytest.approx(0.3)
    assert found.rungs["fixed"].stay.short_car_paths == 0
