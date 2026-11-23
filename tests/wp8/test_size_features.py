"""The context row of a day for sizing (`size_features`), on made-up prepared days (`size_fake`).

1. Each column on a hand-made day of three cars (one left unsteered for want of a forecast) is the
   value computed by hand from the date, the counts, the certificate's widths and the forecasts.
2. Leaks: replacing the true cars of the prepared day by others (every departure, rate and room
   changed) leaves every row unchanged; so does changing the true cars before the day is prepared,
   when the energy forecasts are kept (the forecaster here never reads the truth). A changed
   forecast departure does change the row, so the check can see a change.
3. A day that does not offer has no row.
"""

import dataclasses
import datetime

import numpy as np
import pytest
from size_fake import START, prepared_days
from test_gate_day import SETTINGS, Stub, configuration, truth_day

from fleetreserve import gate_day, size_features
from fleetreserve.day import Day, ForecastCar, TrueCar


@pytest.fixture(scope="module")
def three_cars():
    cars = (TrueCar(4, 2.0, 2.0), TrueCar(5, 3.0, 3.0), TrueCar(3, 2.0, 1.0))
    forecasts = [ForecastCar(3.5, 2.0, 2.0, 2.0), ForecastCar(5.0, 3.0, 3.0, 3.0), None]
    day = Day("u", START + datetime.timedelta(days=40), 2, 2, 0.0, cars)
    prepared = gate_day.prepare_day(truth_day(day), Stub(forecasts), configuration(), SETTINGS)
    assert not prepared.abstains and prepared.steered == (0, 1)
    return prepared


def test_every_column_on_a_hand_made_day(three_cars):
    row = size_features.context_row(three_cars)
    assert list(row) == list(size_features.COLUMNS)
    w = np.asarray(three_cars.certificate.policy.w)
    expected = {
        "weekday": 0.0,  # 10 February 2020 was a Monday
        "month": 2.0,
        "day_of_year": 41.0,
        "fleet_at_tau": 3.0,
        "steered_cars": 2.0,
        "unsteered_cars": 1.0,
        "service_hours": 2.0,
        "box_value_kwh": three_cars.value,
        "width_min_kw": w.min(),
        "width_mean_kw": w.mean(),
        "width_max_kw": w.max(),
        "departure_min_h": 3.5,
        "departure_mean_h": 4.25,
        "departure_max_h": 5.0,
        "departure_in_window_share": 0.0,
        "lowest_level_departure_min_h": 3.5,
        "lowest_level_departure_mean_h": 4.25,
    }
    assert row == pytest.approx(expected, abs=1e-12)
    assert three_cars.value == pytest.approx(2 * w.sum(), abs=1e-6)


def test_the_lowest_level_columns_read_the_first_scenario():
    prepared = next(p for p in prepared_days(3, 6, early=0.0) if not p.abstains)
    first = prepared.day.forecasts[0]
    changed = dataclasses.replace(first, scenarios=((first.departure / 2, 0.1),))
    day = dataclasses.replace(prepared.day, forecasts=(changed, *prepared.day.forecasts[1:]))
    row = size_features.context_row(dataclasses.replace(prepared, day=day))
    departures = [f.departure for f in day.forecasts]
    lowest = [first.departure / 2, *departures[1:]]
    assert row["lowest_level_departure_min_h"] == pytest.approx(min(lowest))
    assert row["lowest_level_departure_mean_h"] == pytest.approx(np.mean(lowest))
    assert row["departure_min_h"] == pytest.approx(min(departures))


def other_cars(day: Day) -> tuple:
    """Every true car changed: a later departure, double rate and room."""
    return tuple(TrueCar(c.departure + 1, 2 * c.rate, 2 * c.room) for c in day.cars)


def test_the_true_cars_never_reach_the_row():
    changed, offering = 0, 0
    for prepared in prepared_days(4, 30):
        if prepared.abstains:
            continue
        offering += 1
        row = size_features.context_row(prepared)
        base = dataclasses.replace(prepared.day, n_recovery=prepared.day.n_recovery + 2)
        day = dataclasses.replace(base, cars=other_cars(prepared.day))
        assert size_features.context_row(dataclasses.replace(prepared, day=day)) == row
        first = prepared.day.forecasts[0]
        moved = dataclasses.replace(first, departure=first.departure + 0.5)
        day = dataclasses.replace(base, forecasts=(moved, *prepared.day.forecasts[1:]))
        changed += size_features.context_row(dataclasses.replace(prepared, day=day)) != row
    assert changed == offering >= 10


def test_true_departures_changed_before_preparing_leave_the_row_unchanged():
    rng = np.random.default_rng(8)
    for k in range(15):
        cars = tuple(TrueCar(int(rng.integers(6, 8)), 3.0, 1.0) for _ in range(3))
        forecasts = [ForecastCar(6.0, 3.0, 1.0, 1.0) for _ in cars]
        rows = []
        for shift in (0, 2):
            moved = tuple(TrueCar(c.departure + shift, c.rate, c.room) for c in cars)
            day = Day("u", START + datetime.timedelta(days=k), 2, 6, 0.0, moved)
            prepared = gate_day.prepare_day(
                truth_day(day), Stub(forecasts), configuration(), SETTINGS
            )
            rows.append(size_features.context_row(prepared))
        assert rows[0] == rows[1]


def test_a_day_that_does_not_offer_has_no_row():
    day = Day("u", START, 1, 0, 0.0, (TrueCar(1, 2.0, 0.5),))
    prepared = gate_day.prepare_day(truth_day(day), Stub([None]), configuration(), SETTINGS)
    with pytest.raises(ValueError, match="no context row"):
        size_features.context_row(prepared)
