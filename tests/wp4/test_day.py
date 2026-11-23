"""The day structures of plan sections 1, 2 and 9: validation, in-window departure, the forecast
set's numbers and the conversions of a true car.

Hand-worked car (used below): departure 3, rate 2 kW, room 5 kWh, rho = 0.2, on a day with 2
service hours (hours 1 and 2). The car may draw in hours 0, 1 and 2 (t < 3). Its protected energy
is (1 - 0.2) * 5 = 4 kWh. One-way, its true set is the `gpoly.Car` on hours {0, 1, 2} with power in
[0, 2] and total energy in [4, 5]. Two-way at a budget of 1 kWh, power is in [-2, 2], the stored
energy after every hour in [-1, 5], and at least 4 at the end. It leaves in hour 3, after the last
service hour (2), so it does not leave inside the window; with departure 2 it would.
"""

import datetime

import pytest

from fleetreserve.day import (
    MAX_SERVICE_HOURS,
    Day,
    ForecastCar,
    StoredEnergyCar,
    TrueCar,
    covering_recovery_hours,
)
from fleetreserve.gpoly import Car

DATE = datetime.date(2019, 6, 3)
CAR = TrueCar(3, 2.0, 5.0)


def _day(**changes):
    fields = dict(
        unit="unit", date=DATE, n_service=2, n_recovery=0, rho=0.2, cars=(CAR,), forecasts=None
    )
    fields.update(changes)
    return Day(**fields)


def test_hand_worked_car_one_way_conversion():
    assert CAR.need(0.2) == pytest.approx(4.0)
    assert CAR.one_way(0.2) == Car(frozenset({0, 1, 2}), 0.0, 2.0, (1 - 0.2) * 5.0, 5.0)


def test_hand_worked_car_two_way_conversion():
    assert CAR.two_way(0.2, 1.0) == StoredEnergyCar(3, -2.0, 2.0, -1.0, 5.0, (1 - 0.2) * 5.0)
    assert CAR.two_way(0.0, 0.0) == StoredEnergyCar(3, -2.0, 2.0, 0.0, 5.0, 5.0)


def test_hand_worked_day():
    day = _day()
    assert day.n_hours == 3
    assert list(day.service_hours) == [1, 2]
    assert list(day.recovery_hours) == []
    assert covering_recovery_hours([CAR], 2) == 0
    assert covering_recovery_hours([CAR, TrueCar(7, 1.0, 1.0)], 2) == 4
    assert not day.departs_in_window(0)
    assert not day.has_in_window_departure
    assert day.one_way_cars() == (CAR.one_way(0.2),)
    assert day.two_way_cars(1.0) == (CAR.two_way(0.2, 1.0),)


def test_in_window_departure_is_departure_at_most_the_last_service_hour():
    assert TrueCar(2, 1.0, 1.0).departs_in_window(2)  # misses hour 2
    assert TrueCar(1, 1.0, 1.0).departs_in_window(2)
    assert not TrueCar(3, 1.0, 1.0).departs_in_window(2)  # draws in hours 0, 1, 2
    day = _day(cars=(CAR, TrueCar(2, 1.0, 1.0)), rho=0.0)
    assert [day.departs_in_window(i) for i in range(2)] == [False, True]
    assert day.has_in_window_departure


def test_lists_are_stored_as_tuples_and_numbers_as_floats():
    day = _day(cars=[CAR], rho=0, n_service=2)
    assert day.cars == (CAR,)
    assert isinstance(day.rho, float)
    assert TrueCar(3, 2, 5).rate == 2.0


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (dict(departure=0, rate=1.0, room=0.0), "departure must be at least 1"),
        (dict(departure=2.5, rate=1.0, room=0.0), "departure must be a whole number"),
        (dict(departure=True, rate=1.0, room=0.0), "departure must be a whole number"),
        (dict(departure=2, rate=0.0, room=0.0), "rate must be positive"),
        (dict(departure=2, rate=-1.0, room=0.0), "rate must be positive"),
        (dict(departure=2, rate=1.0, room=-0.1), "room must be non-negative"),
        (dict(departure=2, rate=1.0, room=float("nan")), "room must be finite"),
        (dict(departure=2, rate=float("inf"), room=1.0), "rate must be finite"),
    ],
)
def test_true_car_validation(fields, message):
    with pytest.raises(ValueError, match=message):
        TrueCar(**fields)


def test_conversions_refuse_bad_shares_budgets_and_empty_sets():
    with pytest.raises(ValueError, match="rho must be below 1"):
        CAR.need(1.0)
    with pytest.raises(ValueError, match="rho must be non-negative"):
        CAR.one_way(-0.1)
    with pytest.raises(ValueError, match="budget must be non-negative"):
        CAR.two_way(0.0, -1.0)
    # 6 kWh in 3 hours at 1 kW cannot be met: the one-way and two-way sets are both empty.
    short = TrueCar(3, 1.0, 6.0)
    with pytest.raises(ValueError, match="cannot be met"):
        short.one_way(0.0)
    with pytest.raises(ValueError, match="cannot be stored"):
        short.two_way(0.0, 2.0)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(unit=""), "unit must be a non-empty name"),
        (dict(date="2019-06-03"), "date must be a calendar date"),
        (dict(date=datetime.datetime(2019, 6, 3, 10)), "date must be a calendar date"),
        (dict(n_service=0), "n_service must be at least 1"),
        (dict(n_service=MAX_SERVICE_HOURS + 1, n_recovery=0), "above 8"),
        (dict(n_service=2.0), "n_service must be a whole number"),
        (dict(n_recovery=-1), "n_recovery must be at least 0"),
        (dict(rho=1.0), "rho must be below 1"),
        (dict(rho=-0.1), "rho must be non-negative"),
        (dict(cars=(Car(frozenset({0}), 0, 1, 0, 1),)), "every car must be a TrueCar"),
        (dict(cars=(TrueCar(4, 1.0, 1.0),)), "after the day's 3 hours"),
        (dict(cars=(TrueCar(3, 1.0, 3.5),), rho=0.1), "true set is empty"),
    ],
)
def test_day_validation(changes, message):
    with pytest.raises(ValueError, match=message):
        _day(**changes)


def test_a_car_that_needs_exactly_its_rate_times_hours_is_accepted():
    day = _day(cars=(TrueCar(3, 1.0, 3.0),), rho=0.0)
    assert day.cars[0].one_way(0.0) == Car(frozenset({0, 1, 2}), 0.0, 1.0, 3.0, 3.0)


def test_forecast_set_numbers():
    # Forecast departure 3.25 on a day with 2 service hours and 2 recovery hours (hours 3, 4):
    # shares f_t = 1, 1, 1, 0.25, 0 for hours 0..4; recovery supply 2 * (0.25 + 0) = 0.5 kWh.
    forecast = ForecastCar(3.25, 2.0, 4.0, 3.0)
    assert [forecast.share(t) for t in range(5)] == [1.0, 1.0, 1.0, 0.25, 0.0]
    assert [forecast.power_cap(t) for t in range(5)] == [2.0, 2.0, 2.0, 0.5, 0.0]
    assert forecast.recovery_supply(2, 2) == pytest.approx(0.5)
    assert forecast.window_need(2, 2) == pytest.approx(2.5)
    day = _day(n_recovery=2, forecasts=(forecast,), kappa=0.1)
    assert day.forecasts == (forecast,)
    assert day.kappa == 0.1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(forecasts=(ForecastCar(2.0, 1.0, 1.0, 1.0),)), "given together"),
        (dict(kappa=0.1), "given together"),
        (dict(forecasts=(), kappa=0.1), "0 forecasts for 1 cars"),
        (dict(forecasts=(CAR,), kappa=0.1), "every forecast must be a ForecastCar"),
        (dict(forecasts=(ForecastCar(3.5, 1.0, 1.0, 1.0),), kappa=0.1), "leaves after"),
        (dict(forecasts=(ForecastCar(2.0, 1.0, 1.0, 1.0),), kappa=1.0), "kappa must be below 1"),
        (dict(forecasts=(ForecastCar(2.0, 1.0, 1.0, 1.0),), kappa=0.0), "kappa must be positive"),
    ],
)
def test_forecast_validation(changes, message):
    with pytest.raises(ValueError, match=message):
        _day(**changes)


def test_forecast_car_validation():
    with pytest.raises(ValueError, match="forecast departure must be non-negative"):
        ForecastCar(-0.5, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="forecast need must be finite"):
        ForecastCar(1.0, 1.0, 1.0, float("inf"))


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (dict(n_hours=0, low=0, high=1, stored_low=0, stored_high=1, need=0), "at least 1"),
        (dict(n_hours=2, low=0.5, high=1, stored_low=0, stored_high=1, need=0), "must include 0"),
        (dict(n_hours=2, low=-1, high=1, stored_low=0.5, stored_high=1, need=0), "must include 0"),
        (dict(n_hours=2, low=-1, high=1, stored_low=-1, stored_high=1, need=1.5), "cannot be"),
        (dict(n_hours=2, low=-1, high=0.5, stored_low=-1, stored_high=3, need=1.5), "cannot be"),
    ],
)
def test_stored_energy_car_validation(fields, message):
    with pytest.raises(ValueError, match=message):
        StoredEnergyCar(**fields)
