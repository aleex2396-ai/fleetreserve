"""The empirical forecast bounds: quantile convention, leave-one-day-out, survival tightening, the
small-cell fallback, the three modes and the oracle departure, on made-up truth tables."""

import random

import numpy as np
import pytest
from truth_tables_fake import FakeCar, date_of, regular_days, tables

from fleetreserve import days_from_truth as dft
from fleetreserve import forecast_empirical as fe


def _unit(cars, rho=0.0):
    car_table, day_table = tables(cars)
    return dft.unit_days(
        car_table, day_table, unit="unit", rho=rho, no_fit=dft.NO_FIT_CAP, small_fleet="keep"
    )


# Quantile convention


def test_order_statistic_by_hand():
    values = list(range(1, 21))  # n = 20
    assert fe.lower_quantile(values, 0.1) == 3  # k = floor(2) + 1 = 3: two values below
    assert fe.upper_quantile(values, 0.1) == 18  # two values above
    assert fe.lower_quantile(values, 0.05) == 2
    assert fe.lower_quantile(list(range(1, 11)), 0.3) == 4  # 0.3 * 10 is exactly 3
    assert fe.lower_quantile([7.5], 0.3) == 7.5
    assert fe.lower_quantile([5, 1, 3], 0.2) == 1  # unsorted input


def test_bound_holds_on_at_least_one_minus_kappa_of_the_sample_and_is_the_largest_such():
    rng = random.Random(3)
    for _ in range(400):
        n = rng.randint(1, 60)
        kappa = rng.choice(fe.KAPPAS)
        x = np.array([rng.choice([rng.random(), 0.5]) for _ in range(n)])  # ties included
        low, high = fe.lower_quantile(x, kappa), fe.upper_quantile(x, kappa)
        assert np.mean(x >= low) >= 1 - kappa - 1e-12
        assert np.mean(x <= high) >= 1 - kappa - 1e-12
        above = np.sort(x)[np.sort(x) > low]
        if above.size:  # the next larger value would leave more than kappa * n below it
            assert np.mean(x >= above[0]) < 1 - kappa


def test_quantile_refuses_bad_input():
    with pytest.raises(ValueError):
        fe.lower_quantile([], 0.1)
    with pytest.raises(ValueError):
        fe.lower_quantile([1.0], 1.0)


# Leave one day out


def _with_day_changed(cars, day, how):
    out = []
    for car in cars:
        if car.date == day:
            car = FakeCar(**{**car.__dict__, **how})
        out.append(car)
    return out


def _forecasts(unit, kappa=0.2, mode="empirical", reference="other-days", min_cell=fe.MIN_CELL):
    forecaster = fe.EmpiricalForecaster(unit, reference=reference, min_cell=min_cell)
    return {t.record.date: forecaster.forecast_day(t, kappa, mode) for t in unit.days}, forecaster


def test_a_days_own_outcomes_never_enter_its_forecast():
    cars = regular_days(6)  # each day's 8:00 cell holds the other days' 20 cars
    target = date_of(3)
    before, _ = _forecasts(_unit(cars))
    changed = _with_day_changed(cars, target, {"plug_out": 11.5, "room": 0.0, "rate": 0.5})
    after, _ = _forecasts(_unit(changed))
    assert after[target].day.forecasts == before[target].day.forecasts
    # The same change seen from another day moves that day's forecast.
    other = date_of(4)
    assert after[other].day.forecasts != before[other].day.forecasts


def test_a_session_of_the_day_is_left_out_of_its_reference_on_every_day():
    # Session "long" is plugged in at τ on days 0 and 1. Its day-0 row must not be a reference
    # of day 1, though it lies on another day.
    cars = regular_days(3, per_day=3)
    cars += [
        FakeCar("A", date_of(0), 9.0, 24 + 40.0, room=6.0, session="long"),
        FakeCar("A", date_of(1), 9.0 - 24, 40.0, room=0.0, session="long"),
    ]
    unit = _unit(cars)
    forecaster = fe.EmpiricalForecaster(unit, min_cell=1000)  # the whole site
    day1 = unit.days[1]
    found = forecaster.reference_for(day1, 0)
    assert found.fallback is True
    expected = [c.plug_out - 10.0 for c in cars if c.date != date_of(1) and c.session != "long"]
    assert np.allclose(np.sort(found.departures), np.sort(expected))


def test_earlier_days_only_uses_no_later_day_and_skips_the_first():
    cars = regular_days(10)
    unit = _unit(cars)
    forecasts, forecaster = _forecasts(unit, reference="earlier-days", min_cell=1)
    assert forecasts[date_of(0)] is None
    day = unit.days[3]
    found = forecaster.reference_for(day, 0)
    assert found.departures.size == 3 * 4
    changed = _with_day_changed(cars, date_of(5), {"plug_out": 23.0})
    later, _ = _forecasts(_unit(changed), reference="earlier-days", min_cell=1)
    assert later[date_of(3)].day.forecasts == forecasts[date_of(3)].day.forecasts


# Survival tightening


def test_survival_bound_is_after_the_hour_and_equals_the_bound_at_tau_while_no_one_has_left():
    unit = _unit(regular_days(40))
    forecaster = fe.EmpiricalForecaster(unit)
    truth = unit.days[5]
    for kappa in fe.KAPPAS:
        at_tau = forecaster.forecast_car(truth, 0, kappa, "known").departure
        # Every reference departure is at least 2.25 hours after τ.
        assert forecaster.survival_bound(truth, 0, 0, kappa) == at_tau
        assert forecaster.survival_bound(truth, 0, 2, kappa) == at_tau
        for hour in range(0, 12):
            bound = forecaster.survival_bound(truth, 0, hour, kappa)
            if bound is not None:
                assert bound > hour
                assert bound >= at_tau
        assert forecaster.survival_bound(truth, 0, 20, kappa) is None  # all have left


def test_survival_bound_rises_once_reference_cars_leave():
    unit = _unit(regular_days(40))
    forecaster = fe.EmpiricalForecaster(unit)
    truth = unit.days[5]
    assert forecaster.survival_bound(truth, 0, 3, 0.1) > forecaster.survival_bound(truth, 0, 2, 0.1)


def test_survival_bound_refuses_a_bad_hour():
    unit = _unit(regular_days(5))
    forecaster = fe.EmpiricalForecaster(unit)
    with pytest.raises(ValueError):
        forecaster.survival_bound(unit.days[0], 0, 1.5, 0.1)


# Small-cell fallback


def _two_cells(n_days):
    cars = regular_days(n_days)
    # One car a day plugged in at 6:00, leaving late: its own cell.
    cars += [
        FakeCar("A", date_of(k), 6.0, 20.0, rate=9.0, room=20.0, session=f"early-{k}")
        for k in range(n_days)
    ]
    return cars


def test_a_small_cell_falls_back_to_the_site():
    unit = _unit(_two_cells(10))  # the 6:00 cell has 9 reference cars, fewer than 20
    forecaster = fe.EmpiricalForecaster(unit)
    truth = unit.days[0]
    i = [c.session_id for c in truth.fleet].index("early-0")
    found = forecaster.reference_for(truth, i)
    assert found.fallback and found.departures.size == 9 * 5
    small = fe.EmpiricalForecaster(unit, min_cell=9)
    found = small.reference_for(truth, i)
    assert not found.fallback and np.all(found.departures == 10.0)
    day = forecaster.forecast_day(truth, 0.1, "known")
    assert day.fallback[i] and sum(day.fallback) == 1  # the 8:00 cell has 36 cars


def test_cars_from_an_earlier_day_form_their_own_cell():
    cars = regular_days(25, per_day=3)
    cars += [
        FakeCar("A", date_of(k), 8.0 - 24, 18.0, room=1.0, session=f"night-{k}") for k in range(25)
    ]
    unit = _unit(cars)
    forecaster = fe.EmpiricalForecaster(unit)
    truth = unit.days[0]
    i = [c.session_id for c in truth.fleet].index("night-0")
    found = forecaster.reference_for(truth, i)
    assert not found.fallback and np.all(found.departures == 8.0)


# Modes and the oracle departure


def test_known_mode_copies_the_truth():
    unit = _unit(regular_days(30), rho=0.2)
    forecaster = fe.EmpiricalForecaster(unit)
    truth = unit.days[4]
    for i, car in enumerate(truth.day.cars):
        forecast = forecaster.forecast_car(truth, i, 0.1, "known")
        assert (forecast.rate, forecast.cap) == (car.rate, car.room)
        assert forecast.need == pytest.approx(0.8 * car.room)


def test_empirical_and_user_rate_modes_use_the_reference_quantiles():
    unit = _unit(regular_days(30), rho=0.2)
    forecaster = fe.EmpiricalForecaster(unit, min_cell=1000)  # the whole site
    truth = unit.days[4]
    found = forecaster.reference_for(truth, 1)
    for kappa in fe.KAPPAS:
        empirical = forecaster.forecast_car(truth, 1, kappa, "empirical")
        assert empirical.departure == fe.lower_quantile(found.departures, kappa)
        assert empirical.rate == fe.lower_quantile(found.rates, kappa)
        assert empirical.cap == fe.lower_quantile(found.rooms, kappa)
        assert empirical.need == fe.upper_quantile(0.8 * found.rooms, kappa)
        user = forecaster.forecast_car(truth, 1, kappa, "user-rate")
        assert user.rate == truth.day.cars[1].rate
        assert (user.cap, user.need, user.departure) == (
            empirical.cap,
            empirical.need,
            empirical.departure,
        )
    with pytest.raises(ValueError):
        forecaster.forecast_car(truth, 1, 0.1, "other")


def test_oracle_departure_is_the_whole_departure_hour():
    unit = _unit(regular_days(30))
    forecaster = fe.EmpiricalForecaster(unit)
    truth = unit.days[2]
    for i, car in enumerate(truth.day.cars):
        forecast = forecaster.forecast_car(truth, i, 0.3, "empirical", "oracle")
        assert forecast.departure == car.departure
        assert truth.fleet[i].departure_exact > car.departure  # the exact time is not used


def test_forecast_day_extends_the_recovery_hours_to_hold_every_forecast():
    cars = [FakeCar("A", date_of(0), 8.0, 12.5, session=f"early-{j}") for j in range(3)]
    cars += [
        FakeCar("A", date_of(k), 8.0, 25.5, session=f"late-{k}-{j}")
        for k in range(1, 30)
        for j in range(3)
    ]
    unit = _unit(cars)
    forecaster = fe.EmpiricalForecaster(unit)
    truth = unit.days[0]
    assert truth.day.n_recovery == 0
    day = forecaster.forecast_day(truth, 0.3, "known").day
    assert [f.departure for f in day.forecasts] == [15.5] * 3
    assert day.n_recovery == 16 - 1 - 6
    assert day.cars == truth.day.cars and day.kappa == 0.3
