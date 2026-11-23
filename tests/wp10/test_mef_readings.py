"""The hand cases behind three sentences of `mef_controller`'s readings and believed future, with
every number worked out here.

1. B-R1's alternative: one car (departure 2, rate 2, room 2, rho 0.75, so need 0.5), one
   service hour, box `[0, 2]`, hour-zero target 2, request 1 in hour 1. This module plans 2 in
   hour 0 (the future after it is met by the request 0) and then cannot draw the request 1:
   hour 1 fails. The operator of Li et al. with the weight 2 minimises
   `(u - 2)^2 - 2 log p(u)`, with the feedback `p(u)` proportional to the length of the
   requests `y` of hour 1 still feasible after `u` (`0 <= y <= 2`, `u + y <= 2`,
   `u + y >= 0.5`); on a grid of 2,000 totals its minimiser is 1, and the run that plans 1 in
   hour 0 meets the request 1: no hour fails.
2. B-R8's alternative: in hour 1 of three service hours, car 0 (believed departure 3, rate 2,
   cap 5, need 3, drawn 2) is seen to have left with 1 of its believed need undrawn, beside
   car 1 (departure 4, rate 2, cap 8, need 2), box `[0, 4]`. Kept with its believed limits it
   can still draw 2 in hour 2, and the believed future after the hour's split (car 1 takes the
   target 2) is met. With its limits taken as 0, the future is met at no total from 0 to the
   fleet's most, in hour 1 and in hour 2.
3. The believed future's recovery term: `r_hat max(d_hat - n - 1, 0)` equals
   `day.ForecastCar.recovery_supply` for believed departures before the recovery hours, inside
   them and at the day's last, `n + R + 1`; `day.Day` refuses a forecast departure after it.
"""

import dataclasses
import datetime

import numpy as np
import pytest

from fleetreserve.controller import Belief, HourView, oracle_belief
from fleetreserve.controller_replay import run_controller
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.mef_controller import MefController
from fleetreserve.mef_future import Future, enough, meets
from fleetreserve.priority_rules import least_laxity_first, most
from fleetreserve.rungs import DEPARTURE_SEEN

DATE = datetime.date(2026, 1, 5)


def _view(hour, n, target, drawn, plugged, belief):
    zeros = np.zeros(hour)
    drawn = np.array(drawn, float)
    return HourView(hour, n, target, zeros, zeros, drawn, np.array(plugged), belief)


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_a_finite_weight_can_meet_a_request_this_reading_fails(seen):
    day = Day("hand", DATE, 1, 0, 0.75, (TrueCar(2, 2.0, 2.0),))
    belief = oracle_belief(day)
    assert np.array_equal(belief.need, [0.5]) and np.array_equal(belief.cap, [2.0])

    def run(controller):
        return run_controller(
            day,
            controller,
            belief,
            np.zeros(1),
            np.array([1.0]),
            hour_zero_target=2.0,
            departure_seen=seen,
        )

    controller = MefController([0.0], [2.0])
    found = run(controller)
    assert np.array_equal(found.planned, [[2.0, 0.0]]) and found.first_failure == 1
    assert controller.deviation_counts()["deviating_hours"] == 0
    totals = np.linspace(0.0, 2.0, 2001)[:-1]  # p(2) = 0: the cost is infinite there
    length = np.minimum(2.0, 2.0 - totals) - np.maximum(0.0, 0.5 - totals)
    cost = (totals - 2.0) ** 2 - 2.0 * np.log(length)
    chosen = float(totals[np.argmin(cost)])
    assert chosen == 1.0

    def weighted(view):
        return np.array([chosen]) if view.hour == 0 else least_laxity_first(view)

    other = run(weighted)
    assert np.array_equal(other.planned, [[1.0, 1.0]])
    assert other.first_failure is None and other.slot_loss == 0.0


BELIEF = Belief([3.0, 4.0], [2.0, 2.0], [5.0, 8.0], [3.0, 2.0])
DRAWN = [2.0, 0.0]


def _futures(controller, hour, total):
    """Least-laxity-first's split of `total` with car 0 seen to have left, and the believed
    futures after it with car 0 kept with its believed limits and with its limits taken as 0."""
    left = _view(hour, 3, total, DRAWN, [False, True], BELIEF)
    split = least_laxity_first(left)
    kept = controller.future(dataclasses.replace(left, plugged=np.ones(2, bool)), split)
    none = Future(
        np.vstack([np.zeros((1, kept.n_hours)), kept.high[1:]]),
        kept.least,
        kept.room,
        kept.lo,
        kept.hi,
    )
    return left, split, kept, none


def test_a_departed_car_kept_with_its_believed_limits_can_leave_the_future_met():
    controller = MefController([0.0] * 3, [4.0] * 3)
    _, split, kept, _ = _futures(controller, 1, 2.0)
    assert np.array_equal(split, [0.0, 2.0])
    assert np.array_equal(kept.high[0], [2.0, 0.0]) and kept.least[0] == 1.0
    assert meets(kept)


@pytest.mark.parametrize("hour", [1, 2])
def test_a_departed_car_with_its_limits_taken_as_0_leaves_no_total_feasible(hour):
    controller = MefController([0.0] * 3, [4.0] * 3)
    left, *_ = _futures(controller, hour, 0.0)
    top = float(most(left).sum())
    assert top == 2.0
    for total in np.linspace(0.0, top, 41):
        _, split, _, none = _futures(controller, hour, float(total))
        assert split[0] == 0.0 and none.least[0] == 1.0
        assert not enough(none) and not meets(none)


@pytest.mark.parametrize("departure", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
def test_the_recovery_term_is_the_recovery_supply_up_to_the_days_last_hour(departure):
    n, recovery = 1, 2  # hours 0 to 3: n_hours = n + R + 1 = 4
    forecast = ForecastCar(departure, 2.0, 10.0, 5.0)
    day = Day("hand", DATE, n, recovery, 0.5, (TrueCar(4, 2.0, 10.0),), (forecast,), 0.1)
    assert day.n_hours == 4
    term = 2.0 * max(departure - n - 1, 0.0)
    assert term == forecast.recovery_supply(n, recovery)
    view = _view(1, n, 0.0, [1.0], [True], Belief([departure], [2.0], [10.0], [5.0]))
    future = MefController([0.0], [3.0]).future(view, [0.5])
    assert future.n_hours == 0 and future.least[0] == 5.0 - 1.0 - 0.5 - term


def test_a_forecast_departure_after_the_days_last_hour_is_refused():
    late = ForecastCar(4.5, 2.0, 10.0, 5.0)
    with pytest.raises(ValueError, match="leaves after the day's 4 hours"):
        Day("hand", DATE, 1, 2, 0.5, (TrueCar(4, 2.0, 10.0),), (late,), 0.1)
