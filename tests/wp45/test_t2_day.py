"""`t2_day.check_day` on hand-made days: the uncertifiable-car rules, E'_i, the closed form against
replay, and true rows on every corner path."""

import pytest

from fleetreserve import t2_day
from fleetreserve.day import ForecastCar, TrueCar

CARS = (TrueCar(4, 3.0, 6.0), TrueCar(5, 2.0, 4.0), TrueCar(3, 4.0, 2.0))


def task(forecasts, rule=t2_day.LEAVE_UNSTEERED, cars=CARS, exact=(4.5, 5.2, 3.9)):
    return t2_day.T2Task(
        unit="u",
        site="A",
        date="2020-01-01",
        n_service=3,
        n_recovery=2,
        rho=0.0,
        cars=cars,
        forecasts=tuple(forecasts),
        exact_departures=exact,
        uncertifiable=rule,
    )


def known(car: TrueCar, departure=None) -> ForecastCar:
    d = car.departure if departure is None else departure
    return ForecastCar(d, car.rate, car.room, car.room)


def test_known_forecasts_are_certified_in_e_prime_with_no_problem():
    found = t2_day.check_day(task([known(c) for c in CARS]))
    assert found.status == t2_day.CERTIFIED
    assert found.steered == 3 and found.steered_in_e_prime == 3 and found.day_in_e_prime
    assert found.value > 0
    assert found.rows_checked > 0 and found.replay_mismatches == 0
    assert found.largest_replay_gap < 1e-9
    assert found.cars_checked_in_e_prime == 3 and found.true_row_violations_in_e_prime == 0
    assert found.paths_with_request_missed == 0 and found.problems == ()


def test_a_forecast_departure_after_the_true_one_is_outside_e_prime_and_not_judged():
    late = known(CARS[2], departure=6.0)  # true departure 3: the forecast lets it stay longer
    found = t2_day.check_day(task([known(CARS[0]), known(CARS[1]), late]))
    assert found.status == t2_day.CERTIFIED
    assert found.steered_in_e_prime == 2 and not found.day_in_e_prime
    assert found.cars_checked_in_e_prime == 2
    assert found.problems == ()


def test_a_car_outside_e_prime_breaking_its_true_set_is_counted_not_judged():
    greedy = ForecastCar(3.0, 4.0, 5.0, 5.0)  # must store 5 kWh; its true room is 2
    found = t2_day.check_day(task([known(CARS[0]), known(CARS[1]), greedy]))
    assert found.status == t2_day.CERTIFIED and not found.day_in_e_prime
    assert found.cars_outside_e_prime_breaking_a_true_row == 1
    assert found.true_row_violations_in_e_prime == 0
    assert found.problems == ()


def test_an_earlier_departure_forecast_keeps_the_car_in_e_prime():
    early = known(CARS[1], departure=3.5)
    found = t2_day.check_day(task([known(CARS[0]), early, known(CARS[2])]))
    assert found.steered_in_e_prime == 3 and found.day_in_e_prime
    assert found.problems == ()


@pytest.mark.parametrize("missing", ["none", "empty"])
def test_an_uncertifiable_car_is_left_unsteered_or_the_day_abstains(missing):
    bad = None if missing == "none" else ForecastCar(1.0, 1.0, 1.0, 50.0)  # need cannot fit
    forecasts = [known(CARS[0]), bad, known(CARS[2])]
    left = t2_day.check_day(task(forecasts))
    assert left.status == t2_day.CERTIFIED and left.steered == 2
    assert left.cars_without_forecast == (missing == "none")
    assert left.cars_with_empty_set == (missing == "empty")
    abstained = t2_day.check_day(task(forecasts, rule=t2_day.DAY_ABSTAINS))
    assert abstained.status == t2_day.ABSTAINED and abstained.value is None
    assert abstained.rows_checked == 0


def test_a_day_with_no_steered_car_has_no_certificate():
    found = t2_day.check_day(task([None, None, None]))
    assert found.status == t2_day.NO_STEERED_CAR
    assert found.cars_without_forecast == 3 and not found.day_in_e_prime


def test_the_forecast_hours_cover_every_forecast_departure():
    t = task([known(CARS[0], departure=9.5), known(CARS[1]), known(CARS[2])])
    assert t2_day.forecast_hours(t) == 6  # ceil(9.5) - 1 - 3
    assert t2_day.forecast_hours(task([None, None, None])) == 2
