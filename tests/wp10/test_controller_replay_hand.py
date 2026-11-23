"""A priority rule replayed on true cars, by hand: an early leaver and the two departure readings.

One day: three service hours, one recovery hour, rho = 0 (each car needs its room).

- Car X: truly leaves in hour 2 (draws in hours 0 and 1), rate 2 kW, room 4 kWh. Forecast:
  departure 4 (it overstates the departure), rate 2, cap 4, need 4.
- Car Y: truly leaves in hour 5 (recovery hour 4), rate 1 kW, room 5 kWh. Forecast: departure 5,
  rate 1, cap 5, need 5.

EDF with the forecast belief, hour-zero target 1. X has the earlier believed departure (4 < 5).

Case A, requests (2, 1, 1):

- Hour 0: target 1, X first (most 2): X 1. Hour 1: target 2, X most min(2, 4 - 1) = 2: X 2.
- Hour 2, "rounded-hour": X is not counted as plugged in (2 < 2 is false); Y most 1: Y 1;
  delivered 1, no failure. Hour 3: Y 1, delivered 1. No failure, slot loss 0.
  Y: 2 drawn, recovery hour +1: 3. X: 3. Shortfalls 1 + 2 of 4 + 5: driver loss 3/9.
- Hour 2, "hour-start": X is counted as plugged in (2 <= 2) with most min(2, 4 - 3) = 1, so EDF
  plans X 1, Y 0; X has gone and draws 0: delivered 0 against 1, first failure in hour 2, slot
  loss (3 - 2 + 1)/3 = 2/3. Hour 3: X no longer counted, Y 1. Y: 1 + recovery 1 = 2. Shortfalls
  1 + 3: driver loss 4/9.

So the two readings differ in hour 2 (test 5 of the brief).

Case B, the early leaver's failure (test 3), "rounded-hour", requests (2, 2, 1): hours 0 and 1 as in
case A; hour 2: Y alone, most 1 against target 2: delivered 1, first failure in hour 2, slot loss
2/3; hour 3: Y 1. Finals X 3, Y 3: driver loss 3/9.

Case C, the oracle belief (test 3), requests (2, 1, 1), both readings: X's believed departure is
its true 2. Hour 2 under "hour-start": X is counted as plugged in but its believed most is
2 x min(max(2 - 2, 0), 1) = 0, so Y gets 1. The draws under both readings are those of case A under
"rounded-hour": no failure.
"""

import datetime

import numpy as np
import pytest

from fleetreserve.controller import Belief, forecast_belief, oracle_belief
from fleetreserve.controller_replay import run_controller
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.priority_rules import earliest_deadline_first
from fleetreserve.rungs import DEPARTURE_SEEN

DATE = datetime.date(2020, 1, 1)
TOLERANCE = 1e-12


def _day():
    cars = (TrueCar(2, 2.0, 4.0), TrueCar(5, 1.0, 5.0))
    forecasts = (ForecastCar(4.0, 2.0, 4.0, 4.0), ForecastCar(5.0, 1.0, 5.0, 5.0))
    return Day("hand", DATE, 3, 1, 0.0, cars, forecasts, 0.1)


def _run(belief: Belief, requests, seen):
    day = _day()
    return run_controller(
        day,
        earliest_deadline_first,
        belief,
        np.zeros(3),
        np.array(requests, dtype=float),
        hour_zero_target=1.0,
        departure_seen=seen,
    )


def test_case_a_rounded_hour():
    run = _run(forecast_belief(_day()), [2, 1, 1], "rounded-hour")
    np.testing.assert_allclose(run.planned, [[1, 2, 0, 0], [0, 0, 1, 1]], atol=TOLERANCE)
    np.testing.assert_allclose(run.draws, [[1, 2, 0, 0], [0, 0, 1, 1]], atol=TOLERANCE)
    assert run.first_failure is None
    assert run.slot_loss == 0.0
    np.testing.assert_allclose(run.final, [3.0, 3.0], atol=TOLERANCE)
    assert run.driver_loss == pytest.approx(3 / 9, abs=TOLERANCE)


def test_case_a_hour_start_differs_in_hour_2():
    run = _run(forecast_belief(_day()), [2, 1, 1], "hour-start")
    np.testing.assert_allclose(run.planned, [[1, 2, 1, 0], [0, 0, 0, 1]], atol=TOLERANCE)
    np.testing.assert_allclose(run.draws, [[1, 2, 0, 0], [0, 0, 0, 1]], atol=TOLERANCE)
    assert list(run.failed) == [False, True, False]
    assert run.first_failure == 2
    assert run.slot_loss == pytest.approx(2 / 3, abs=TOLERANCE)
    np.testing.assert_allclose(run.final, [3.0, 2.0], atol=TOLERANCE)
    assert run.driver_loss == pytest.approx(4 / 9, abs=TOLERANCE)


def test_case_b_the_early_leaver_fails_the_hour_the_other_car_cannot_cover():
    run = _run(forecast_belief(_day()), [2, 2, 1], "rounded-hour")
    np.testing.assert_allclose(run.draws, [[1, 2, 0, 0], [0, 0, 1, 1]], atol=TOLERANCE)
    np.testing.assert_allclose(run.delivered, [2, 1, 1], atol=TOLERANCE)
    assert list(run.failed) == [False, True, False]
    assert run.first_failure == 2
    assert run.slot_loss == pytest.approx(2 / 3, abs=TOLERANCE)
    assert run.driver_loss == pytest.approx(3 / 9, abs=TOLERANCE)


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_case_c_the_oracle_belief_meets_the_path_under_both_readings(seen):
    run = _run(oracle_belief(_day()), [2, 1, 1], seen)
    np.testing.assert_allclose(run.draws, [[1, 2, 0, 0], [0, 0, 1, 1]], atol=TOLERANCE)
    assert run.first_failure is None
    assert run.slot_loss == 0.0
