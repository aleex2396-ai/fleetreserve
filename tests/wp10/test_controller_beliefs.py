"""The two beliefs of `controller.py` (R2), the hour-zero target (R3) and the read-only view.

Day: two service hours, one recovery hour, rho = 0.2. Car 1: departure 3, rate 2 kW, room 5 kWh
(protected 4); forecast departure 2.5, rate 1.5, cap 4.5, need 4.2. Car 2: departure 4, rate 1 kW,
room 2.5 kWh (protected 2); forecast departure 3.5, rate 0.9, cap 2.4, need 2.1.

- Forecast belief: the forecasts as given, field by field.
- Oracle belief: departures (3, 4), rates (2, 1), caps the rooms (5, 2.5), needs the protected
  energies (4, 2).
- Hour-zero target of a plan with hour-0 constants 0.7 and 0.4: 1.1.
- Laxity of a car whose need is already met (6 kWh drawn against a need of 4.2) in hour 1:
  2.5 - 1 = 1.5 (the remaining need counts as 0).
"""

import datetime

import numpy as np
import pytest

from fleetreserve.controller import (
    Belief,
    HourView,
    forecast_belief,
    hour_zero_target,
    oracle_belief,
)
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.plans import Plan
from fleetreserve.priority_rules import laxity

DATE = datetime.date(2020, 1, 1)


def _day():
    cars = (TrueCar(3, 2.0, 5.0), TrueCar(4, 1.0, 2.5))
    forecasts = (ForecastCar(2.5, 1.5, 4.5, 4.2), ForecastCar(3.5, 0.9, 2.4, 2.1))
    return Day("beliefs", DATE, 2, 1, 0.2, cars, forecasts, 0.1)


def test_the_forecast_belief_is_the_forecasts_field_by_field():
    belief = forecast_belief(_day())
    np.testing.assert_array_equal(belief.departure, [2.5, 3.5])
    np.testing.assert_array_equal(belief.rate, [1.5, 0.9])
    np.testing.assert_array_equal(belief.cap, [4.5, 2.4])
    np.testing.assert_array_equal(belief.need, [4.2, 2.1])


def test_the_oracle_belief_is_the_true_cars_with_their_protected_energy():
    belief = oracle_belief(_day())
    np.testing.assert_array_equal(belief.departure, [3.0, 4.0])
    np.testing.assert_array_equal(belief.rate, [2.0, 1.0])
    np.testing.assert_array_equal(belief.cap, [5.0, 2.5])
    np.testing.assert_allclose(belief.need, [4.0, 2.0], atol=1e-12)


def test_the_hour_zero_target_is_the_plans_hour_0_total():
    constant = np.array([[0.7, 1.0, 0.0], [0.4, 0.0, 2.0]])
    plan = Plan(constant, np.zeros((2, 3, 2)), two_way=False)
    assert hour_zero_target(plan) == pytest.approx(1.1, abs=1e-12)


def _view(drawn=(6.0, 0.0)):
    return HourView(1, 2, 1.0, [0.0], [1.0], list(drawn), [True, True], forecast_belief(_day()))


def test_the_laxity_of_a_car_whose_need_is_met_counts_no_remaining_need():
    assert laxity(_view())[0] == pytest.approx(1.5, abs=1e-12)


def test_a_controller_cannot_change_the_view_it_is_given():
    view = _view()
    for array in (view.drawn, view.plugged, view.zeta_known, view.requests_known):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        view.belief.rate[0] = 5.0


def test_a_view_with_plugged_flags_for_another_fleet_is_refused():
    belief = Belief([3.0, 4.0], [2.0, 1.0], [3.0, 2.0], [3.0, 2.0])
    with pytest.raises(ValueError, match="plugged"):
        HourView(1, 2, 1.0, [0.0], [1.0], [0.0, 0.0], [True], belief)
