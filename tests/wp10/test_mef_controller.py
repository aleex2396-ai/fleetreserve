"""The controller `mef_controller.MefController` on days built by hand, its believed future and
its registration, with every number worked out here.

1. Day A, oracle belief, three service hours, no recovery hour, rho 0.5, three cars in the order
   X, Y, Z: X leaves in hour 3 (rate 2, room 2, need 1), Y and Z in hour 4 (Y: rate 3, room 14,
   need 7; Z: rate 1, room 1, need 0.5). Box: lowest requests (1, 2, 0), highest (3, 5, 4);
   hour-zero target 1, requests (1, 2, 2.5).
   - Hours 0 and 1: least-laxity-first gives the whole target to Y (laxities 2.5, 1.67, 3.5, then
     1.5, 1, 2.5); a feasible future remains (Y 3 and X 1 in hour 2, Y 2 and Z 0.5 in hour 3), so
     the powers are least-laxity-first's.
   - Hour 2: laxities X 0.5, Y 2 - 5/3 = 0.33, Z 1.5; the request 2 goes to Y and X leaves with 1
     undrawn: no future. A total `u` gives X `max(u - 3, 0)` (Y's most is 3), so the smallest
     total that meets X's need is 4; at 4 hour 3 can still be followed (Y needs 2 of its 3, Z 0.5
     of its 1, the box allows 0 to 4), and every total up to the most, 6, leaves room. The
     controller plans 4: Y 3, X 1, Z 0.
   - Hour 3: laxities X 0 (its most is 0), Y 1 - 2/3 = 0.33, Z 0.5; the request 2.5 goes to Y and
     Z would leave 0.5 short; the smallest total that reaches Z is Y's most 3 plus 0.5 = 3.5,
     planned as Y 3, Z 0.5.
   - So two deviating hours, both service hours and neither the first; no hour without a feasible
     total. The replay fails in hour 2 (4 delivered against 2): slot loss 2/3, driver loss 0.
     Least-laxity-first plans totals 1, 1, 2, 2.5, fails no hour, and leaves X 1, Y 0.5 and Z 0.5
     short of 8.5: driver loss 2/8.5. The same under both readings of `departure_seen`.
2. Day B, forecast belief, three service hours and one recovery hour, rho 0.5, cars P and Q.
   Forecasts: P departure 4.5 (half of the recovery hour, so a supply of 2 x 0.5 = 1 kWh), rate 2,
   cap 3.5, need 3.5; Q departure 3, rate 2, cap 4, need 0. Box: lowest (0, 0, 1), highest
   (1, 1, 3); hour-zero target 0, requests (0, 0.2, 2). Hour 2: laxities P 2.5 - 3.5/2 = 0.75,
   Q 1; 0.2 to P leaves P needing 3.5 - 0.2 - 1 = 2.3 in hour 3, where it can take 2: no future.
   The smallest total that leaves P at most 2 is 0.5 (without the recovery supply it would be
   1.5); the controller plans 0.5 to P. Hour 3: P takes the request 2 and meets its need exactly.
   One deviating hour (hour 2). With lowest requests (0, 0, 1.8), highest (1, 2, 3) and requests
   (0, 2, 1.8) instead, hour 2's request 2 to P leaves it 1.5 of its cap for hour 3, which asks
   at least 1.8: the largest total that keeps 1.8 is 1.7 (and the smallest for P's need 0.5),
   so the controller plans 1.7, below the request; hour 3 then takes the request 1.8 with P's
   last room.
3. The believed future of a view, field by field: hour 1 of three service hours, four cars, one
   not counted as plugged in (left out), a departure inside a remaining hour (its share), a
   departure in the recovery hours (its supply) and a need already met (a negative `least`);
   the remaining box is the box's hours 2 and 3. After hour 3 the future has no hour.
4. Four hours built by hand in which no total leaves the believed future met, one for each way
   that can happen (a need above the cap; a need above what the hour allows; a box above what the
   rooms allow; a smallest total for the needs above the largest total for the rooms): the
   controller plans least-laxity-first's powers and counts an hour with no feasible total.
5. Two hours built by hand whose needs are met at every total, so that only the room the box's
   next hour asks for limits the total: the controller plans the largest feasible total, 1
   (below the request 2), or 0 when only the total 0 is feasible.
6. Registration: `max-entropy-feedback` right after `diagonal-split`, a controller built from the
   box `m -+ beta w` without rows; refusals of a box whose two sides differ in length and of a
   view with another number of service hours.
"""

import datetime

import numpy as np
import pytest

from fleetreserve import compare_methods, mef_controller, mef_future
from fleetreserve.compare_methods import MethodInput
from fleetreserve.controller import Belief, HourView, forecast_belief, oracle_belief
from fleetreserve.controller_replay import run_controller
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.mef_controller import MefController
from fleetreserve.plans import Plan
from fleetreserve.policy import AffinePolicy
from fleetreserve.priority_rules import least_laxity_first
from fleetreserve.rungs import DEPARTURE_SEEN

DATE = datetime.date(2026, 1, 5)
EDGE = 2e-9  # the bisections stop within `BISECTION_KWH` of the exact edge


def day_a() -> Day:
    cars = (TrueCar(3, 2.0, 2.0), TrueCar(4, 3.0, 14.0), TrueCar(4, 1.0, 1.0))
    return Day("hand", DATE, 3, 0, 0.5, cars)


def day_b() -> Day:
    cars = (TrueCar(5, 2.0, 3.5), TrueCar(3, 2.0, 4.0))
    forecasts = (ForecastCar(4.5, 2.0, 3.5, 3.5), ForecastCar(3.0, 2.0, 4.0, 0.0))
    return Day("hand", DATE, 3, 1, 0.5, cars, forecasts, 0.1)


def _run(day, controller, belief, target, requests, seen):
    zeta = np.zeros(day.n_service)
    return run_controller(
        day, controller, belief, zeta, requests, hour_zero_target=target, departure_seen=seen
    )


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_day_a_deviates_in_hours_2_and_3_with_the_totals_worked_out_by_hand(seen):
    day = day_a()
    requests = np.array([1.0, 2.0, 2.5])
    controller = MefController([1.0, 2.0, 0.0], [3.0, 5.0, 4.0])
    run = _run(day, controller, oracle_belief(day), 1.0, requests, seen)
    expected = np.array(
        [[0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 0.0, 0.5]]
    )  # cars X, Y, Z by hour 0 to 3
    assert np.allclose(run.planned, expected, rtol=0, atol=EDGE)
    assert np.allclose(run.planned.sum(axis=0), [1.0, 1.0, 4.0, 3.5], rtol=0, atol=EDGE)
    assert controller.deviation_counts() == {
        "hours": 4,
        "deviating_hours": 2,
        "deviating_service_hours": 2,
        "no_feasible_hours": 0,
    }
    assert run.first_failure == 2 and run.slot_loss == pytest.approx(2 / 3)
    assert run.driver_loss == pytest.approx(0.0, abs=1e-8)
    plain = _run(day, least_laxity_first, oracle_belief(day), 1.0, requests, seen)
    assert np.allclose(plain.planned.sum(axis=0), [1.0, 1.0, 2.0, 2.5])
    assert plain.first_failure is None and plain.driver_loss == pytest.approx(2 / 8.5)
    assert np.array_equal(run.planned[:, :2], plain.planned[:, :2])


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_day_b_counts_the_recovery_supply_and_deviates_in_hour_2(seen):
    day = day_b()
    requests = np.array([0.0, 0.2, 2.0])
    controller = MefController([0.0, 0.0, 1.0], [1.0, 1.0, 3.0])
    run = _run(day, controller, forecast_belief(day), 0.0, requests, seen)
    expected = np.array([[0.0, 0.0, 0.5, 2.0], [0.0, 0.0, 0.0, 0.0]])
    assert np.allclose(run.planned, expected, rtol=0, atol=EDGE)
    assert controller.deviation_counts() == {
        "hours": 4,
        "deviating_hours": 1,
        "deviating_service_hours": 1,
        "no_feasible_hours": 0,
    }
    assert run.first_failure == 2


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_day_b_with_a_higher_lowest_request_in_hour_3_deviates_below_the_request(seen):
    day = day_b()
    requests = np.array([0.0, 2.0, 1.8])
    controller = MefController([0.0, 0.0, 1.8], [1.0, 2.0, 3.0])
    run = _run(day, controller, forecast_belief(day), 0.0, requests, seen)
    expected = np.array([[0.0, 0.0, 1.7, 1.8], [0.0, 0.0, 0.0, 0.0]])
    assert np.allclose(run.planned, expected, rtol=0, atol=EDGE)
    assert controller.deviation_counts() == {
        "hours": 4,
        "deviating_hours": 1,
        "deviating_service_hours": 1,
        "no_feasible_hours": 0,
    }
    assert run.first_failure == 2


def _view(hour, drawn, plugged, belief, n=3, target=1.0):
    return HourView(hour, n, target, np.zeros(hour), np.zeros(hour), drawn, plugged, belief)


def test_the_believed_future_field_by_field():
    belief = Belief(
        departure=[2.5, 5.5, 4.0, 3.25],
        rate=[2.0, 1.0, 3.0, 4.0],
        cap=[6.0, 10.0, 2.0, 3.0],
        need=[4.0, 7.0, 2.0, 1.0],
    )
    view = _view(1, [1.0, 0.5, 0.0, 2.0], np.array([True, True, False, True]), belief)
    controller = MefController([0.5, 1.5, 2.5], [1.0, 2.0, 3.0])
    future = controller.future(view, [1.5, 1.0, 0.0, 0.0])
    assert np.array_equal(future.high, [[1.0, 0.0], [1.0, 1.0], [4.0, 1.0]])
    assert np.array_equal(future.least, [1.5, 4.0, -1.0])  # car 1: supply 1 x 1.5
    assert np.array_equal(future.room, [3.5, 8.5, 1.0])
    assert np.array_equal(future.lo, [1.5, 2.5]) and np.array_equal(future.hi, [2.0, 3.0])
    first = controller.future(_view(0, np.zeros(4), np.ones(4, bool), belief), np.zeros(4))
    assert np.array_equal(first.lo, [0.5, 1.5, 2.5]) and first.high.shape == (4, 3)
    assert np.array_equal(first.high[1], [1.0, 1.0, 1.0])
    last = controller.future(_view(3, np.zeros(4), np.ones(4, bool), belief), np.ones(4))
    assert last.n_hours == 0 and last.high.shape == (4, 0)
    assert np.array_equal(last.least, [3.0, 4.5, 1.0, 0.0])  # car 1: 7 - 1 - 1.5


NONE = {
    # name: (hour, departure, rate, cap, need, lo, hi, target), two service hours, two cars
    "a need above the cap": (1, [3, 3], [2, 1], [1, 5], [3, 0], [0, 0], [5, 5], 0.5),
    "a need above the most": (2, [3, 3], [2, 1], [10, 5], [3, 0], [0, 0], [5, 5], 1.0),
    "a box above the room": (1, [3, 3], [2, 1], [1, 0.5], [0, 0], [0, 3], [5, 5], 0.5),
    "two edges apart": (1, [3, 2], [2, 4], [3.7, 10], [3.6, 3], [0, 1.9], [10, 10], 2.0),
}


@pytest.mark.parametrize("name", NONE)
def test_an_hour_with_no_feasible_total_plans_least_laxity_first(name):
    """The four ways step 4 is reached, worked out by hand. A need above the cap: car 0 needs 3
    with a cap of 1 (`within_cars` fails at every total). A need above the most: in the last
    hour car 0 still needs 3 and can take 2. A box above the room: hour 2 asks at least 3 of cars
    whose rooms add up to 1.5. Two edges apart: car 0 (laxity 0.2) is filled before car 1 (0.25);
    car 1 leaves after the hour and needs 3, so the total must be at least car 0's most 2 plus
    3 = 5, while car 0 must keep 1.9 of its cap 3.7 for hour 2, so the total must be at most
    1.8."""
    hour, departure, rate, cap, need, lo, hi, target = NONE[name]
    belief = Belief(departure, rate, cap, need)
    view = _view(hour, [0.0, 0.0], np.array([True, True]), belief, n=2, target=target)
    controller = MefController(lo, hi)
    assert controller.nearest_total(view) is None
    planned = controller(view)
    assert np.array_equal(planned, least_laxity_first(view))
    assert controller.deviation_counts() == {
        "hours": 1,
        "deviating_hours": 0,
        "deviating_service_hours": 0,
        "no_feasible_hours": 1,
    }


ROOM = {
    # name: (lowest request of hour 2, the total planned), one car, two service hours, hour 1
    "room for 1 of the request 2": (2.0, 1.0),
    "room for no energy": (3.0 + mef_future.TOLERANCE, 0.0),
}


@pytest.mark.parametrize("name", ROOM)
def test_an_hour_whose_needs_are_met_at_0_deviates_down_to_the_room_it_leaves(name):
    """One car (departure 3, rate 4, cap 3, need 0) in hour 1 of two service hours, target 2: its
    need is met at every total, so the smallest feasible total is 0. Hour 2 asks at least `lo`,
    so the car must keep `lo` of its cap: the largest feasible total is `3 - lo`, 1 when `lo` is 2.
    When `lo` is 3 plus `TOLERANCE`, only the total 0 keeps it (the future is met with equality),
    an interval of one point, still a feasible total."""
    lo, total = ROOM[name]
    belief = Belief([3.0], [4.0], [3.0], [0.0])
    view = _view(1, [0.0], np.array([True]), belief, n=2, target=2.0)
    controller = MefController([0.0, lo], [5.0, 5.0])
    assert controller.nearest_total(view) == pytest.approx(total, abs=EDGE)
    assert controller(view) == pytest.approx([total], abs=EDGE)
    assert controller.deviation_counts() == {
        "hours": 1,
        "deviating_hours": 1,
        "deviating_service_hours": 1,
        "no_feasible_hours": 0,
    }


def _policy(n=2, cars=2):
    g = np.zeros((cars, n + 1))
    return AffinePolicy(np.array([2.0, 3.0]), np.array([0.5, 1.0]), g, np.zeros((cars, n + 1, n)))


def test_the_registry_builds_the_controller_from_the_box_without_rows():
    names = [m.name for m in compare_methods.METHODS]
    assert names[-4:-2] == ["diagonal-split", "max-entropy-feedback"]
    (method,) = compare_methods.select(["max-entropy-feedback"])
    assert method.kind == "controller" and method.build is compare_methods.max_entropy_feedback
    day = day_b()
    policy = _policy()
    inputs = MethodInput(day, policy, 0.4, Plan.from_policy(policy, 0.4))
    assert inputs.rows is None
    built = method.build(inputs)
    assert isinstance(built, MefController)
    assert np.allclose(built.lo, [1.8, 2.6]) and np.allclose(built.hi, [2.2, 3.4])
    assert built.deviation_counts() == dict.fromkeys(mef_controller.COUNTS, 0)


def test_the_bisection_stops_after_its_most_steps_with_an_error():
    """Near 1e10 kWh neighbouring floats lie about 2e-6 apart, so no interval there narrows to
    `BISECTION_KWH`: the bisection raises after `MAX_STEPS` steps instead of running on. An
    interval of width exactly `BISECTION_KWH` is returned without a step; an ordinary edge is
    found well within the steps, to `BISECTION_KWH`, on its holding side."""
    assert mef_controller.MAX_STEPS == 200
    calls = []

    def holds(u):
        calls.append(u)
        return u >= 1e10 + 5e-4

    with pytest.raises(mef_controller.BisectionError, match="did not narrow"):
        mef_controller._bisect(holds, 1e10, 1e10 + 1e-3, rising=True)
    assert len(calls) == mef_controller.MAX_STEPS
    calls.clear()
    width = mef_controller.BISECTION_KWH
    assert mef_controller._bisect(holds, 0.0, width, rising=True) == width  # narrow already
    assert mef_controller._bisect(holds, 0.0, width, rising=False) == 0.0 and calls == []

    def below(u):
        calls.append(u)
        return u <= 0.3

    edge = mef_controller._bisect(below, 0.0, 6.0, rising=False)
    assert len(calls) < 50
    assert 0.3 - mef_controller.BISECTION_KWH <= edge <= 0.3 and below(edge)


def test_refusals():
    with pytest.raises(ValueError, match="one entry per service hour"):
        MefController([0.0, 1.0], [1.0])
    belief = Belief([3.0], [1.0], [1.0], [0.0])
    view = _view(0, [0.0], np.array([True]), belief, n=2)
    with pytest.raises(ValueError, match="3 service hours, the view 2"):
        MefController([0.0] * 3, [1.0] * 3)(view)
