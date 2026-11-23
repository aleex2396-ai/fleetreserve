"""Refusals of the linear-programme form of ASA (`asa_controller`, `asa_programme`).

- `AsaController.for_box` refuses a two-way plan (a box whose lower corner is below 0 is not
  refused: `test_asa_properties` and `test_asa_command` run one).
- The controller refuses a view whose belief (and so whose number of cars) is not the size of its
  day, and a view with another number of service hours than its box.
- A time limit that is missing (a `TypeError` from the call), 0, negative, infinite, not a number
  or a boolean is refused by the controller and by `plan_hour`; a centre of the wrong size or not
  finite, and a number of recovery hours that is negative or not a whole number, are refused by
  `plan_hour`.
"""

import math
import random

import pytest
from made_up_offer_days import offer_day

from fleetreserve.asa_controller import AsaController
from fleetreserve.asa_programme import plan_hour
from fleetreserve.controller import Belief, HourView
from fleetreserve.plans import Plan

BELIEF = Belief([3, 3], [1, 1], [10, 10], [1, 1])
VIEW = HourView(1, 2, 1.0, [0.0], [1.0], [0.0, 0.0], [True, True], BELIEF)


@pytest.fixture(scope="module")
def day_and_policy():
    rng = random.Random(1114)
    k = 0
    while True:
        day, certificate, _ = offer_day(rng, k)
        if certificate.outcome == "optimal":
            return day, certificate.policy
        k += 1


def _build(day, policy, plan, beta=1.0, **extra):
    arguments = {"n_cars": len(day.cars), "n_recovery": day.n_recovery, "time_limit": 300.0}
    arguments.update(extra)
    return AsaController.for_box(policy, beta, plan, **arguments)


def test_a_two_way_plan_is_refused(day_and_policy):
    day, policy = day_and_policy
    plan = Plan.from_policy(policy, 1.0)
    two_way = Plan(plan.constant, plan.gains, two_way=True)
    with pytest.raises(ValueError, match="one-way; a two-way plan is refused"):
        _build(day, policy, two_way)
    _build(day, policy, plan)


def test_a_belief_of_the_wrong_size_is_refused():
    controller = AsaController([0.0, 0.0], n_cars=3, n_recovery=0, time_limit=300.0)
    with pytest.raises(ValueError, match="the view has 2 cars, the controller's day 3"):
        controller(VIEW)
    short = AsaController([0.0], n_cars=2, n_recovery=0, time_limit=300.0)
    with pytest.raises(ValueError, match="the view has 2 service hours, the box 1"):
        short(VIEW)
    AsaController([0.0, 0.0], n_cars=2, n_recovery=0, time_limit=300.0)(VIEW)


BAD_LIMITS = [0, 0.0, -1.0, math.inf, math.nan, "300", None, True]


@pytest.mark.parametrize("limit", BAD_LIMITS)
def test_a_bad_time_limit_is_refused(limit):
    with pytest.raises(ValueError, match="time limit must be a positive finite number"):
        AsaController([0.0, 0.0], n_cars=2, n_recovery=0, time_limit=limit)
    with pytest.raises(ValueError, match="time limit must be a positive finite number"):
        plan_hour(VIEW, [0.0, 0.0], 0, time_limit=limit)


def test_a_missing_time_limit_is_refused():
    with pytest.raises(TypeError, match="time_limit"):
        AsaController([0.0, 0.0], n_cars=2, n_recovery=0)
    with pytest.raises(TypeError, match="time_limit"):
        plan_hour(VIEW, [0.0, 0.0], 0)


@pytest.mark.parametrize(
    ("centre", "recovery", "message"),
    [
        ([0.0], 0, "the centre must hold 2 values"),
        ([0.0, math.nan], 0, "the centre must be finite"),
        ([0.0, 0.0], -1, "the recovery hours must be a whole number"),
        ([0.0, 0.0], 1.5, "the recovery hours must be a whole number"),
    ],
)
def test_plan_hour_refuses_a_bad_centre_or_recovery(centre, recovery, message):
    with pytest.raises(ValueError, match=message):
        plan_hour(VIEW, centre, recovery, time_limit=300.0)
