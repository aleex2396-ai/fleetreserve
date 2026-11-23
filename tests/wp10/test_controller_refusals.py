"""Refusals of the controller interface and replay: each an error with a message, never a default.

- `departure_seen` missing, unknown or None;
- a belief with a rate at or below 0, or with arrays of different lengths, or a negative cap;
- a belief, zeta or requests whose length does not match the day;
- an hour view whose arrays do not match its belief or its hour;
- a controller that returns the wrong number of powers, a negative power or a power that is not
  finite;
- a plan controller for a two-way plan or for another fleet;
- a forecast belief for a day without forecasts.
"""

import datetime

import numpy as np
import pytest

from fleetreserve.controller import Belief, HourView, forecast_belief
from fleetreserve.controller_replay import run_controller
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.plan_controller import PlanController
from fleetreserve.plans import Plan
from fleetreserve.priority_rules import earliest_deadline_first

DATE = datetime.date(2020, 1, 1)


def _day(forecasts=True):
    cars = (TrueCar(3, 2.0, 3.0), TrueCar(4, 1.0, 2.0))
    if not forecasts:
        return Day("refusals", DATE, 2, 1, 0.0, cars)
    found = (ForecastCar(3.0, 2.0, 3.0, 3.0), ForecastCar(4.0, 1.0, 2.0, 2.0))
    return Day("refusals", DATE, 2, 1, 0.0, cars, found, 0.1)


def _run(controller=earliest_deadline_first, **changes):
    day = _day()
    arguments = dict(
        day=day,
        controller=controller,
        belief=forecast_belief(day),
        zeta=np.zeros(2),
        requests=np.ones(2),
        hour_zero_target=1.0,
        departure_seen="rounded-hour",
    )
    arguments.update(changes)
    return run_controller(**arguments)


def test_the_plain_run_works():
    assert _run().first_failure is None


def test_a_missing_departure_reading_is_refused():
    day = _day()
    with pytest.raises(TypeError, match="departure_seen"):
        run_controller(
            day,
            earliest_deadline_first,
            forecast_belief(day),
            np.zeros(2),
            np.ones(2),
            hour_zero_target=1.0,
        )


@pytest.mark.parametrize("seen", ["sometimes", None, ""])
def test_an_unknown_departure_reading_is_refused(seen):
    with pytest.raises(ValueError, match="departure_seen"):
        _run(departure_seen=seen)


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_a_belief_with_a_rate_at_or_below_zero_is_refused(rate):
    with pytest.raises(ValueError, match="rate must be positive"):
        Belief([3.0, 4.0], [2.0, rate], [3.0, 2.0], [3.0, 2.0])


def test_a_belief_with_arrays_of_different_lengths_is_refused():
    with pytest.raises(ValueError, match="one entry per car"):
        Belief([3.0, 4.0], [2.0, 1.0], [3.0], [3.0, 2.0])


@pytest.mark.parametrize("field", ["departure", "cap", "need"])
def test_a_belief_with_a_negative_value_is_refused(field):
    values = {"departure": [3.0], "rate": [1.0], "cap": [2.0], "need": [1.0]}
    values[field] = [-0.5]
    with pytest.raises(ValueError, match=f"believed {field}"):
        Belief(**values)


def test_a_belief_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="finite"):
        Belief([np.nan], [1.0], [2.0], [1.0])


def test_a_belief_for_another_fleet_is_refused():
    with pytest.raises(ValueError, match="belief"):
        _run(belief=Belief([3.0], [2.0], [3.0], [3.0]))


@pytest.mark.parametrize("key", ["zeta", "requests"])
def test_zeta_or_requests_of_the_wrong_length_are_refused(key):
    with pytest.raises(ValueError, match=key):
        _run(**{key: np.zeros(3)})


def test_a_hour_zero_target_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="hour-zero target"):
        _run(hour_zero_target=float("nan"))


def test_an_hour_view_that_does_not_match_its_belief_or_hour_is_refused():
    belief = Belief([3.0, 4.0], [2.0, 1.0], [3.0, 2.0], [3.0, 2.0])
    with pytest.raises(ValueError, match="belief"):
        HourView(1, 2, 1.0, [0.0], [1.0], [0.0], [True], belief)
    with pytest.raises(ValueError, match="plugged"):
        HourView(1, 2, 1.0, [0.0], [1.0], [0.0, 0.0], [1, 1], belief)
    with pytest.raises(ValueError, match="zeta_known"):
        HourView(1, 2, 1.0, [], [1.0], [0.0, 0.0], [True, True], belief)
    with pytest.raises(ValueError, match="hour"):
        HourView(3, 2, 1.0, [0.0] * 3, [1.0] * 3, [0.0, 0.0], [True, True], belief)


@pytest.mark.parametrize(
    "returned, message",
    [
        (np.zeros(3), "one planned power per car"),
        (np.array([-0.5, 0.0]), "at least 0"),
        (np.array([np.inf, 0.0]), "finite"),
    ],
)
def test_a_controller_returning_a_bad_plan_is_refused(returned, message):
    with pytest.raises(ValueError, match=message):
        _run(controller=lambda view: returned)


def test_the_plan_controller_refuses_a_two_way_plan_and_a_plan_for_another_fleet():
    with pytest.raises(ValueError, match="two-way"):
        PlanController(Plan(np.zeros((2, 3)), np.zeros((2, 3, 2)), two_way=True))
    other = PlanController(Plan(np.zeros((3, 3)), np.zeros((3, 3, 2)), two_way=False))
    with pytest.raises(ValueError, match="another fleet"):
        _run(controller=other)


def test_a_forecast_belief_needs_forecasts():
    with pytest.raises(ValueError, match="forecasts"):
        forecast_belief(_day(forecasts=False))
