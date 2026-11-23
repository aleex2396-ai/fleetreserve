"""The diagnosis of one day-path (`asa_diagnosis_day`, `asa_diagnosis_lp`) on days built by hand,
one per cause, each worked out below. Departures seen from the rounded-down hour ("rounded-hour",
the main reading), hour-zero target 0, no recovery hours.

1. The deliverable range (`deliverable`), by hand: one car present with its protected energy
   reachable, one with it out of reach, one gone; one car whose lower end needs three later
   hours at full rate.
2. "state": two service hours, requests (1, 1), centre (1, 2). Car X truly leaves in hour 2 (rate
   1, room 1), believed to stay to 3 with need 1; car Y truly leaves in hour 3 (rate 1, room 1),
   believed to leave at 2.5 with rate 2, cap 2, need 0. In hour 1 the method (`asa`, `asa-es`)
   keeps X for hour 2, where the centre asks 2 and Y is believed to take 1 at most, so it charges
   Y 1 (goal 3); least laxity first (the reference) charges X (laxity 1 against 1.5). In hour 2 X
   has gone: the method's Y is full (range [0, 0]), the reference's Y is empty (range [1, 1],
   its protected energy 1 needs its last hour). Shortfall 1; X left earlier than believed; the
   method had charged the present car Y 1 kWh ahead.
3. "belief": one car, rate 1, room 2, rho 0.5, truly to hour 3, believed cap 1; requests (1, 1).
   The method charges 1 in hour 1 and then believes the car full: it plans 0 in hour 2, where the
   car could take 1 (range [0, 1]). A reference that plans 1 in each hour meets both. Shortfall
   1, the deficit 1 labelled "cap".
4. "choice", goal "need": one service hour, request 1, rho 0.5. Car A: rate 1, room 1, believed
   departure 1.5 at rate 2 (most 1). Car B: rate 1, true room 0.5, believed cap 1 and need 1
   (most 1). Goal 2 (the believed unmet need) puts the hour's 1 on B, which draws 0.5; earliest
   deadline first (A first) meets it. Range [0.75, 1.5]; believed_on_true 1.5; shortfall 0.5;
   0.5 planned above what B takes. The tracking tests: with goal 1 kept a tracking plan exists
   ("optimal"), with goal 2 kept none ("infeasible"). The planned total is the request, so the
   shortfall is the power planned above B's true take. Also: the same day with the believed
   limits on the true cars just at the request ("choice", not "belief"); goal "share" of
   `asa-es` on two cars believed alike; the deficits by label, each label its own amount.
5. "fallback": the day of item 4 with a solver that times out on every solve: the method plans
   by least laxity first (B's laxity 0 below A's 0.5), B again, and the hour is "fallback".
6. "other": one car, rate 2, room 4, rho 0.75, request 1, believed cap 0.5 (so the believed
   limits on the true car, 0.5, are below the request); a made-up method plans 2 and delivers 2
   (range [1, 2]), a made-up reference plans 1. The delivery above the request makes it "other",
   not "belief".
7. "choice" with a method that solves no programme: goal "rule".
8. The goal test's answers "tie" (every tracking test optimal) and "not determined" (a timeout
   or an "error"), with solvers made up for the tracking tests only.
9. The time limit given to `diagnose_path` reaches every tracking test, and only valid limits are
   taken.
10. The four groups, one hour each, with made-up controllers; `left_earlier` on cars that left
    in the hour with a believed departure above, at and below the true hour, and on a car still
    present.
11. More cases: the "state" day of item 2 with a third car Z gone from hour 1 and believed to
    stay to 3 (two cars left earlier in hour 2); the goal "share" day of item 4 with a third car C
    believed to leave at 1 and to need 5, so that goal 2's optimum is 5, not 0; a "tie": two
    cars believed alike, request 0.2 (within the first piece of the fifth goal, so splits tie
    under it), one of them truly taking 0.05, a made-up reference planning 0.2 on the other. The
    form that plans the hour on the weak car (`asa-es` with it first, `asa` with it second)
    misses, and every tracking test ends "optimal".
12. The causes' strict tests at their boundaries, on days built so that the compared numbers are
    equal in floating point: a request exactly `TOLERANCE` above the method's upper end is not
    "state"; believed limits on the true cars exactly `TOLERANCE` below the request are not
    "belief".
13. "choice" in hour 2, hour 1 met by both, so that the goal test must read hour 2's own plan
    (`hour_plans[2]`), whose stage optima differ from hour 1's; the optima the goal test is
    given are recorded and checked against the values worked out below.
    - Goal "share" (`asa-es`), two service hours: cars A (rate 2, room 2) and B (rate 2, room
      0.75), both truly to hour 3, believed alike (departure 3, rate 2, cap 4, need 0); requests
      (1, 1), centre (1, 1), rho 0. Hour 1: the fifth goal splits each hour of the horizon 0.5
      and 0.5 (upper limit 2, pieces of 0.25, 0.5 a breakpoint); optima track 0, need 0, centre
      0, quick -3 (weights 2 and 1 on totals of 1), share 4 f(0.5) = 1. Hour 2: B has 0.25 of
      room left, the plan is again 0.5 and 0.5, B draws 0.25: delivered 0.75, shortfall 0.25,
      all of it planned above B's take. Its optima: 0, 0, 0, quick -1, share 2 f(0.5) = 0.5. The
      split (0.75, 0.25) keeps goals 1 to 4 and tracks; none keeps goal 5 too: goal "share".
      Range [1.75, 1.75] (rho 0: the last hour must bring each car to its room); believed most 2
      per car, at least the true take, so no deficit. Least laxity first (a tie, car order) gives
      A both hours and meets them. Hour 1's optima would name "quick" (quick at most -3 cannot be
      kept in one hour with a total of 1).
    - Goal "need" (`asa` and `asa-es`), three service hours, so that hour 2 is not the last and
      its programme has two hours: car C truly to hour 2 (rate 1, room 1), believed to leave at 2
      with rate 1, cap 5, need 5; car A truly to hour 4 (rate 2, room 2), believed departure 2.5
      at rate 2, cap 1.5, need 0; car B truly to hour 4 (rate 1, room 0.5), believed departure 3,
      rate 1, cap 1, need 1. Requests (0.5, 1, 0), centre (0.5, 1, 0), rho 0. No car is believed
      to draw in hour 3. Hour 1: C's need can be met only in hour 1 and B's also in hour 2, so
      goal 2 puts hour 1's 0.5 on C (unmet need 4.5) and plans B 1 in hour 2; optima track 0,
      need 4.5, centre 0, quick -(3 x 0.5 + 2 x 1) = -3.5 (and for `asa-es` share
      f(0.5) + f(1) = 0.25 + 1 = 1.25, upper limits 1, pieces of 0.125). Hour 2: C has gone;
      goal 2 puts the 1 on B, which draws 0.5: shortfall 0.5. Its optima 0, 0, 0, -(2 x 1) = -2
      (and share 1). With goal 1 kept a tracking plan exists (A takes up to 2), with goal 2 kept
      none: goal "need". Range [0, 2.5] (each car has hour 3 left, which can bring it to its
      room); believed_on_true min(1, 2) + min(1, 0.5) = 1.5; A's deficit 2 - 1 = 1, labelled
      "departure" (believed rate 2 times share 0.5, 1, is below its cap 1.5). Earliest deadline
      first charges C, then A, and meets every hour; the reference delivers 0.5 in hour 1 and 1
      in hour 2. Hour 1's optima would name "quick"; the tracking rows put on hour 3's column
      in place of hour 2's would name "track" (no car is believed to draw in hour 3).
14. `left_earlier` read in the row's own hour, and the tolerance.
    - Three service hours, rho 0, made-up controllers, requests (1, 2, 1). Car 0: truly to hour
      2 (rate 1, room 1), believed to stay to 3. Car 1: truly to hour 3 (rate 1, room 1),
      believed to stay to 3.5, so in hour 2 it is present in its last hour (d = t + 1) with a
      believed departure above d. Car 2: truly to hour 4 (rate 1, room 2), believed to leave at
      4. Hour 1: the method fills car 1, the reference charges car 0. Hour 2: the method's upper
      end is car 2's rate, 1, below the request 2 ("state"); the reference charges cars 1 and 2
      and meets it (upper end 2). Hour 3: both charge car 2 and meet it. In hour 2 only car 0
      left earlier than believed (1 car); car 1 is still present and has not.
    - The tolerance of both modules is `replay.FAILURE_KWH`, 1e-4 kWh.
    - A miss by a fixed 2e-4 kWh (twice the threshold, so the hour fails) in hour 2 of two, hour
      1 met by both (made-up controllers, rho 0): "choice" (one car, rate 2, room 2, believed cap
      2; the method plans 0.5 then 1 - 2e-4 of requests (0.5, 1): believed limits on the true car
      1.5), goal "rule"; "belief" (the same with believed cap 1.5 - 2e-4: believed limits on the
      true car 1 - 2e-4); "state" (the day of item 12 with the request of hour 2 2e-4 above the
      method's upper end 1; the method plans 1). A tolerance of 1e-2 would make all three
      "other".
"""

import datetime

import numpy as np
import pytest

from fleetreserve import asa_diagnosis_day, asa_diagnosis_lp, lp, replay
from fleetreserve.asa_controller import AsaController, AsaEsController
from fleetreserve.asa_diagnosis_day import (
    TOLERANCE,
    Recorder,
    deficits,
    deliverable,
    diagnose_path,
    left_earlier,
)
from fleetreserve.asa_diagnosis_lp import NOT_DETERMINED, TIE, forbidding_goal
from fleetreserve.controller import Belief, HourView, forecast_belief
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.lp import LpResult
from fleetreserve.priority_rules import earliest_deadline_first, least_laxity_first

DATE = datetime.date(2020, 1, 1)
CLOSE = 1e-6
LIMIT = 60.0
SEEN = "rounded-hour"
FORMS = (AsaController, AsaEsController)


def _path(day, method, reference, requests, *, time_limit=LIMIT, solve=None):
    n = day.n_service
    return diagnose_path(
        day,
        method,
        reference,
        forecast_belief(day),
        np.zeros(n),
        np.array(requests, dtype=float),
        hour_zero_target=0.0,
        departure_seen=SEEN,
        time_limit=time_limit,
        solve=solve,
    )


def _only(found):
    groups, rows, tests, _, _ = found
    assert groups["method_only"] == 1 and len(rows) == 1
    return rows[0], groups, tests


# --- 1. the deliverable range --------------------------------------------------------------------


def test_the_deliverable_range_by_hand():
    # n 2, recovery 2; rho 0.5. Car 0: d 5, rate 2, room 6 (protected 3), drawn 0 before hour 1:
    # high min(2, 6) = 2, low max(0, 3 - 0 - 2 * (5 - 1 - 1)) = 0. Car 1: d 2, rate 1, room 4
    # (protected 2), drawn 0: high 1, protected 2 > 0 + 1 + 1 * 0 is out of reach, low = high = 1.
    # Car 2: d 1, gone in hour 1: 0 and 0.
    cars = (TrueCar(5, 2.0, 6.0), TrueCar(2, 1.0, 4.0), TrueCar(1, 1.0, 0.5))
    day = Day("hand", DATE, 2, 2, 0.5, cars)
    low, high = deliverable(day, [0.0, 0.0, 0.0], 1)
    np.testing.assert_allclose(low, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(high, [2.0, 1.0, 0.0])
    # Car 0 with 5.5 drawn: high min(2, 0.5) = 0.5; low 0. Car 1 with 1.5 drawn: high 1, its
    # protected 2 needs 0.5 more in hour 1, its last: low 0.5.
    low, high = deliverable(day, [5.5, 1.5, 0.0], 1)
    np.testing.assert_allclose(low, [0.0, 0.5, 0.0])
    np.testing.assert_allclose(high, [0.5, 1.0, 0.0])
    # rho 0.3: car 0 (d 5, rate 2, room 10, protected 7) in hour 1 with 3 later hours (2, 3, 4)
    # that can take 6: low 7 - 0 - 2 * 3 = 1, high 2; with 0.5 drawn, low 0.5.
    day = Day("hand", DATE, 2, 2, 0.3, (TrueCar(5, 2.0, 10.0),))
    low, high = deliverable(day, [0.0], 1)
    np.testing.assert_allclose((low[0], high[0]), (1.0, 2.0))
    low, high = deliverable(day, [0.5], 1)
    np.testing.assert_allclose((low[0], high[0]), (0.5, 2.0))


# --- 2. state ------------------------------------------------------------------------------------


def _state_day():
    cars = (TrueCar(2, 1.0, 1.0), TrueCar(3, 1.0, 1.0))
    forecasts = (ForecastCar(3.0, 1.0, 1.0, 1.0), ForecastCar(2.5, 2.0, 2.0, 0.0))
    return Day("hand", DATE, 2, 0, 0.0, cars, forecasts, 0.1)


@pytest.mark.parametrize("form", FORMS)
def test_state_a_car_leaves_before_its_forecast_after_the_method_postponed_it(form):
    day = _state_day()
    method = form(np.array([1.0, 2.0]), n_cars=2, n_recovery=0, time_limit=LIMIT)
    found = _path(day, method, least_laxity_first, [1.0, 1.0])
    row, groups, tests = _only(found)
    _, _, _, method_run, reference_run = found
    np.testing.assert_allclose(method_run.planned[:, 1], [0.0, 1.0], atol=CLOSE)
    np.testing.assert_allclose(reference_run.planned[:, 1], [1.0, 0.0], atol=CLOSE)
    assert groups["neither"] == 1
    assert (row.hour, row.cause, row.goal) == (2, "state", "")
    assert row.shortfall == pytest.approx(1.0, abs=CLOSE)
    assert (row.method_delivered, row.reference_delivered) == pytest.approx((0.0, 1.0), abs=CLOSE)
    assert (row.method_low, row.method_high) == pytest.approx((0.0, 0.0), abs=CLOSE)
    assert (row.reference_low, row.reference_high) == pytest.approx((1.0, 1.0), abs=CLOSE)
    assert row.method_first_failure and row.cars_left_earlier == 1
    assert row.charged_ahead == pytest.approx(1.0, abs=CLOSE)
    x, y = row.state_cars
    assert (x.car, x.true_departure, x.believed_departure, x.present, x.left_earlier) == (
        0,
        2,
        3.0,
        False,
        True,
    )
    assert (y.car, y.true_departure, y.believed_departure, y.present, y.left_earlier) == (
        1,
        3,
        2.5,
        True,
        False,
    )
    assert (y.method_drawn, y.reference_drawn) == pytest.approx((1.0, 0.0), abs=CLOSE)
    assert (y.method_high, y.reference_high) == pytest.approx((0.0, 1.0), abs=CLOSE)
    assert tests == {}


# --- 3. belief -----------------------------------------------------------------------------------


def _one_car_plan(view):
    return np.array([[0.0, 1.0, 1.0][view.hour]])


@pytest.mark.parametrize("form", FORMS)
def test_belief_the_believed_energy_is_below_the_true_room(form):
    day = Day(
        "hand", DATE, 2, 0, 0.5, (TrueCar(3, 1.0, 2.0),), (ForecastCar(3.0, 1.0, 1.0, 0.0),), 0.1
    )
    method = form(np.array([1.0, 1.0]), n_cars=1, n_recovery=0, time_limit=LIMIT)
    row, _, _ = _only(_path(day, method, _one_car_plan, [1.0, 1.0]))
    assert (row.hour, row.cause) == (2, "belief")
    assert row.shortfall == pytest.approx(1.0, abs=CLOSE)
    assert (row.method_low, row.method_high) == pytest.approx((0.0, 1.0), abs=CLOSE)
    assert row.believed_on_true == pytest.approx(0.0, abs=CLOSE)
    assert row.deficit_cap == pytest.approx(1.0, abs=CLOSE)
    assert row.deficit_departure == row.deficit_rate == row.deficit_plugged == 0.0
    assert row.state_cars == ()


# --- 4. choice -----------------------------------------------------------------------------------


def _choice_day():
    cars = (TrueCar(2, 1.0, 1.0), TrueCar(2, 1.0, 0.5))
    forecasts = (ForecastCar(1.5, 2.0, 1.0, 0.0), ForecastCar(2.0, 1.0, 1.0, 1.0))
    return Day("hand", DATE, 1, 0, 0.5, cars, forecasts, 0.1)


@pytest.mark.parametrize("form", FORMS)
def test_choice_goal_two_puts_the_hour_on_a_car_that_cannot_take_it(form):
    day = _choice_day()
    method = form(np.array([1.0]), n_cars=2, n_recovery=0, time_limit=LIMIT)
    found = _path(day, method, earliest_deadline_first, [1.0])
    row, _, tests = _only(found)
    assert (row.hour, row.cause, row.goal) == (1, "choice", "need")
    assert row.shortfall == pytest.approx(0.5, abs=CLOSE)
    assert (row.method_delivered, row.reference_delivered) == pytest.approx((0.5, 1.0), abs=CLOSE)
    assert (row.method_low, row.method_high) == pytest.approx((0.75, 1.5), abs=CLOSE)
    assert row.believed_on_true == pytest.approx(1.5, abs=CLOSE)
    assert row.believed_most == pytest.approx(2.0, abs=CLOSE)
    assert row.overplanned == pytest.approx(0.5, abs=CLOSE)
    assert tests == {"optimal": 1, "infeasible": 1}
    # Goal 1 planned the request as the fleet total, all of it on B, whose believed most (1) is
    # above what it truly takes (0.5): the shortfall is the power planned above that.
    planned = found[3].planned[:, 1]
    np.testing.assert_allclose(planned, [0.0, 1.0], atol=CLOSE)
    assert planned.sum() == pytest.approx(row.request, abs=CLOSE)
    assert row.shortfall == pytest.approx(row.request - planned.sum() + row.overplanned, abs=1e-12)


@pytest.mark.parametrize("form", FORMS)
def test_choice_when_the_believed_limits_on_the_true_cars_just_reach_the_request(form):
    # Item 4 with A's believed cap 0.5: believed_on_true = min(0.5, 1) + min(1, 0.5) = 1, the
    # request (to the 1e-9 kWh the programme may draw in hour 0), so the hour is "choice", not
    # "belief": the test is believed_on_true below the request by more than the tolerance.
    cars = (TrueCar(2, 1.0, 1.0), TrueCar(2, 1.0, 0.5))
    forecasts = (ForecastCar(1.5, 2.0, 0.5, 0.0), ForecastCar(2.0, 1.0, 1.0, 1.0))
    day = Day("hand", DATE, 1, 0, 0.5, cars, forecasts, 0.1)
    method = form(np.array([1.0]), n_cars=2, n_recovery=0, time_limit=LIMIT)
    row, _, _ = _only(_path(day, method, earliest_deadline_first, [1.0]))
    assert row.believed_on_true == pytest.approx(1.0, abs=CLOSE)
    assert (row.cause, row.goal) == ("choice", "need")


def test_choice_goal_five_shares_the_hour_onto_a_car_that_cannot_take_it():
    # One service hour, request 1, two cars believed alike (departure 2, rate 2, cap 2, need 0),
    # so goals 1 to 4 tie on every split; car B truly has room 0.25. The fifth goal of asa-es
    # (upper limit 2, 8 pieces of 0.25) splits the hour 0.5 and 0.5; B draws 0.25. A split
    # (0.75, 0.25) keeps goals 1 to 4 and tracks, none keeps goal 5 too: goal "share". Least
    # laxity first (a tie, car order) gives A the whole hour and meets it.
    cars = (TrueCar(2, 2.0, 2.0), TrueCar(2, 2.0, 0.25))
    forecasts = (ForecastCar(2.0, 2.0, 2.0, 0.0), ForecastCar(2.0, 2.0, 2.0, 0.0))
    day = Day("hand", DATE, 1, 0, 0.0, cars, forecasts, 0.1)
    method = AsaEsController(np.array([1.0]), n_cars=2, n_recovery=0, time_limit=LIMIT)
    found = _path(day, method, least_laxity_first, [1.0])
    row, _, tests = _only(found)
    np.testing.assert_allclose(found[3].planned[:, 1], [0.5, 0.5], atol=CLOSE)
    assert (row.cause, row.goal) == ("choice", "share")
    assert row.shortfall == pytest.approx(0.25, abs=CLOSE)
    assert row.overplanned == pytest.approx(0.25, abs=CLOSE)
    assert tests == {"optimal": 4, "infeasible": 1}


def test_the_deficits_by_label_by_hand():
    # Hour 1, nothing drawn, every car plugged in, true takes 3 each. Car 0: believed departure
    # 1.5 (share 0.5), rate 3, cap 2: most min(1.5, 2) = 1.5, "departure" (its rate alone, 3,
    # is above the cap), deficit 1.5. Car 1: departure 4, rate 1, cap 5: most 1, "rate", deficit
    # 2. Car 2: departure 4, rate 4, cap 0.5: most 0.5, "cap", deficit 2.5. Car 3 not counted as
    # plugged in: most 0, "plugged", deficit 3. Car 4: departure 4, rate 0.25, cap 0.25, a tie
    # of rate and cap: "rate" (the test is rate times share at most the cap), deficit 2.75.
    belief = Belief(
        [1.5, 4.0, 4.0, 4.0, 4.0],
        [3.0, 1.0, 4.0, 1.0, 0.25],
        [2.0, 5.0, 0.5, 5.0, 0.25],
        [0.0] * 5,
    )
    plugged = np.array([True, True, True, False, True])
    view = HourView(1, 2, 1.0, np.zeros(1), np.ones(1), np.zeros(5), plugged, belief)
    found = deficits(view, np.full(5, 3.0))
    assert found == {"plugged": 3.0, "departure": 1.5, "rate": 4.75, "cap": 2.5}
    # A car whose true take is at or below its believed most has no deficit.
    assert deficits(view, np.array([1.5, 1.0, 0.25, 0.0, 0.25])) == dict.fromkeys(found, 0.0)


# --- 10. the groups and left_earlier (listed last above) -----------------------------------------


def test_the_four_groups_by_hand():
    # Four service hours, requests 1, one car (rate 2, room 10, rho 0). Made-up controllers: the
    # method plans 1, 0, 0, 1 and the reference 0, 1, 0, 1: hour 1 is missed by the reference
    # only, hour 2 by the method only, hour 3 by both, hour 4 by neither.
    day = Day("hand", DATE, 4, 0, 0.0, (TrueCar(5, 2.0, 10.0),), (ForecastCar(5.0, 2, 10, 0),), 0.1)

    def plans(powers):
        return lambda view: np.array([0.0 if view.hour == 0 else powers[view.hour - 1]])

    found = _path(day, plans([1.0, 0.0, 0.0, 1.0]), plans([0.0, 1.0, 0.0, 1.0]), [1.0] * 4)
    groups, rows, _, _, _ = found
    assert dict(groups) == {"reference_only": 1, "method_only": 1, "both": 1, "neither": 1}
    assert [row.hour for row in rows] == [2]


def test_left_earlier_needs_a_believed_departure_above_the_true_hour():
    # Hour 3. Car 0 left in hour 3, believed 3.2: counts. Car 1 left in hour 3, believed 3 (the
    # oracle belief): does not. Car 2 left in hour 2, believed 1.5: does not. Car 3 still present
    # (d 4), believed 5: does not.
    found = left_earlier([3, 3, 2, 4], [3.2, 3.0, 1.5, 5.0], 3)
    assert found.tolist() == [True, False, False, False]


# --- 5. fallback ---------------------------------------------------------------------------------


class _Timeout:
    def __call__(self, c, *, time_limit, **_):
        return LpResult("timeout", None, None, time_limit, time_limit, 1, "made-up timeout")


def test_fallback_a_timed_out_hour_planned_by_least_laxity_first():
    day = _choice_day()
    method = AsaController(
        np.array([1.0]), n_cars=2, n_recovery=0, time_limit=LIMIT, solve=_Timeout()
    )
    found = _path(day, method, earliest_deadline_first, [1.0])
    row, _, tests = _only(found)
    np.testing.assert_allclose(found[3].planned[:, 1], [0.0, 1.0])
    assert (row.cause, row.goal) == ("fallback", "")
    assert row.shortfall == pytest.approx(0.5)
    assert method.solver_counts()["fallback_hours"] == 2 and tests == {}


# --- 6. other ------------------------------------------------------------------------------------


def test_other_a_delivery_above_the_request():
    # rho 0.75: protected 1 in the car's last hour, so low 1 and high min(2, 4) = 2.
    forecast = ForecastCar(2.0, 2, 0.5, 0)
    day = Day("hand", DATE, 1, 0, 0.75, (TrueCar(2, 2.0, 4.0),), (forecast,), 0.1)

    def two(view):
        return np.array([0.0 if view.hour == 0 else 2.0])

    def one(view):
        return np.array([0.0 if view.hour == 0 else 1.0])

    row, _, _ = _only(_path(day, two, one, [1.0]))
    assert (row.cause, row.goal) == ("other", "")
    assert row.shortfall == pytest.approx(-1.0)
    assert (row.method_low, row.method_high) == pytest.approx((1.0, 2.0))
    assert (row.believed_most, row.believed_on_true) == pytest.approx((0.5, 0.5))


# --- 7. choice by a rule -------------------------------------------------------------------------


def test_choice_by_a_method_without_a_programme_names_the_rule():
    day = _choice_day()

    def on_b(view):
        return np.array([0.0, 0.0 if view.hour == 0 else 1.0])

    row, _, tests = _only(_path(day, on_b, earliest_deadline_first, [1.0]))
    assert (row.cause, row.goal) == ("choice", "rule") and tests == {}


# --- 8. tie and not determined -------------------------------------------------------------------


class _Script:
    """A made-up solver for the tracking tests: the given outcomes in turn."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.limits = []

    def __call__(self, c, *, time_limit, **_):
        self.limits.append(time_limit)
        outcome = self.outcomes.pop(0)
        x = np.zeros(len(c)) if outcome == "optimal" else None
        return LpResult(outcome, x, 0.0 if x is not None else None, 0.0, time_limit, 0, "")


@pytest.mark.parametrize("form", FORMS)
def test_the_goal_test_answers_tie_and_not_determined(form):
    day = _choice_day()
    method = form(np.array([1.0]), n_cars=2, n_recovery=0, time_limit=LIMIT)
    recorder = Recorder(method)
    _path(day, recorder, earliest_deadline_first, [1.0])
    view, plan = recorder.views[1], method.hour_plans[1]
    goals = len(plan.optima)
    assert goals == (5 if form is AsaEsController else 4)
    take = np.array([1.0, 0.5])
    script = _Script(["optimal"] * goals)
    assert forbidding_goal(method, view, plan, take, 1.0, time_limit=5.0, solve=script) == (
        TIE,
        ("optimal",) * goals,
    )
    script = _Script(["optimal", "timeout"])
    assert forbidding_goal(method, view, plan, take, 1.0, time_limit=5.0, solve=script) == (
        NOT_DETERMINED,
        ("optimal", "timeout"),
    )
    assert script.limits == [5.0, 5.0]
    script = _Script(["error"])
    assert forbidding_goal(method, view, plan, take, 1.0, time_limit=5.0, solve=script) == (
        NOT_DETERMINED,
        ("error",),
    )


# --- 9. the time limit ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_the_time_limit_reaches_every_tracking_test(form):
    day = _choice_day()
    method = form(np.array([1.0]), n_cars=2, n_recovery=0, time_limit=LIMIT)
    limits = []

    def solve(c, *, time_limit, **kwargs):
        limits.append(time_limit)
        return lp.solve_lp(c, time_limit=time_limit, **kwargs)

    found = _path(day, method, earliest_deadline_first, [1.0], time_limit=7.5, solve=solve)
    row, _, tests = _only(found)
    assert row.goal == "need" and limits == [7.5, 7.5] and sum(tests.values()) == 2


@pytest.mark.parametrize("limit", [0.0, -1.0, float("inf"), "300"])
def test_a_bad_time_limit_is_refused_before_any_hour(limit):
    day = Day("hand", DATE, 1, 0, 0.0, (TrueCar(2, 2.0, 4.0),), (ForecastCar(2.0, 2, 4, 0),), 0.1)
    with pytest.raises(ValueError, match="time limit"):
        _path(day, least_laxity_first, least_laxity_first, [1.0], time_limit=limit)


# --- 11. more cases ------------------------------------------------------------------------------


@pytest.mark.parametrize("form", FORMS)
def test_state_counts_every_car_that_left_earlier(form):
    cars = (TrueCar(2, 1.0, 1.0), TrueCar(3, 1.0, 1.0), TrueCar(1, 1.0, 1.0))
    forecasts = (
        ForecastCar(3.0, 1.0, 1.0, 1.0),
        ForecastCar(2.5, 2.0, 2.0, 0.0),
        ForecastCar(3.0, 1.0, 1.0, 0.0),
    )
    day = Day("hand", DATE, 2, 0, 0.0, cars, forecasts, 0.1)
    method = form(np.array([1.0, 2.0]), n_cars=3, n_recovery=0, time_limit=LIMIT)
    row, _, _ = _only(_path(day, method, least_laxity_first, [1.0, 1.0]))
    assert (row.hour, row.cause, row.cars_left_earlier) == (2, "state", 2)
    assert [car.left_earlier for car in row.state_cars] == [True, False, True]


def test_choice_goal_five_when_goal_two_has_a_positive_optimum():
    cars = (TrueCar(2, 2.0, 2.0), TrueCar(2, 2.0, 0.25), TrueCar(2, 1.0, 1.0))
    forecasts = (
        ForecastCar(2.0, 2.0, 2.0, 0.0),
        ForecastCar(2.0, 2.0, 2.0, 0.0),
        ForecastCar(1.0, 1.0, 5.0, 5.0),
    )
    day = Day("hand", DATE, 1, 0, 0.0, cars, forecasts, 0.1)
    method = AsaEsController(np.array([1.0]), n_cars=3, n_recovery=0, time_limit=LIMIT)
    row, _, tests = _only(_path(day, method, least_laxity_first, [1.0]))
    assert method.hour_plans[1].optima[1] == pytest.approx(5.0)
    assert (row.cause, row.goal) == ("choice", "share")
    assert tests == {"optimal": 4, "infeasible": 1}


@pytest.mark.parametrize(
    "form, takes", [(AsaEsController, (0.05, 2.0)), (AsaController, (2.0, 0.05))]
)
def test_a_tie_every_tracking_test_optimal(form, takes):
    cars = (TrueCar(2, 2.0, takes[0]), TrueCar(2, 2.0, takes[1]))
    forecasts = (ForecastCar(2.0, 2.0, 2.0, 0.0), ForecastCar(2.0, 2.0, 2.0, 0.0))
    day = Day("hand", DATE, 1, 0, 0.0, cars, forecasts, 0.1)
    strong = int(np.argmax(takes))

    def on_strong(view):
        powers = np.zeros(2)
        powers[strong] = 0.0 if view.hour == 0 else 0.2
        return powers

    method = form(np.array([0.2]), n_cars=2, n_recovery=0, time_limit=LIMIT)
    row, _, tests = _only(_path(day, method, on_strong, [0.2]))
    assert (row.cause, row.goal) == ("choice", TIE)
    assert row.shortfall == pytest.approx(0.15, abs=CLOSE)
    assert tests == {"optimal": len(method.hour_plans[1].optima)}


# --- 12. the boundaries --------------------------------------------------------------------------


def _made_up(*plans):
    return lambda view: np.array(plans[view.hour], dtype=float)


def test_a_request_just_tolerance_above_the_upper_end_is_not_state():
    # Two service hours, cars A (rate 1, room 2) and B (rate 1, room 1). In hour 1 the method
    # fills B and the reference charges A; in hour 2 the method's upper end is 1 + 0 = 1 and the
    # request is 1 + TOLERANCE. The method plans 0, the reference delivers the request.
    cars = (TrueCar(3, 1.0, 2.0), TrueCar(3, 1.0, 1.0))
    forecasts = (ForecastCar(3.0, 1.0, 2.0, 0.0), ForecastCar(3.0, 1.0, 1.0, 0.0))
    day = Day("hand", DATE, 2, 0, 0.0, cars, forecasts, 0.1)
    y = 1.0 + TOLERANCE
    method = _made_up([0, 0], [0, 1], [0, 0])
    reference = _made_up([0, 0], [1, 0], [0.5, y - 0.5])
    row, _, _ = _only(_path(day, method, reference, [1.0, y]))
    assert row.hour == 2 and row.method_high == 1.0
    assert row.request == row.method_high + TOLERANCE
    assert row.cause != "state"


def test_believed_limits_just_tolerance_below_the_request_are_not_belief():
    # One car, rate 2, room 2, believed cap 1 - TOLERANCE: believed_on_true is exactly the request
    # minus the tolerance. A made-up method plans 0.5, a made-up reference 1.
    cap = 1.0 - TOLERANCE
    day = Day(
        "hand", DATE, 1, 0, 0.0, (TrueCar(2, 2.0, 2.0),), (ForecastCar(2.0, 2.0, cap, 0),), 0.1
    )
    row, _, _ = _only(_path(day, _made_up([0], [0.5]), _made_up([0], [1.0]), [1.0]))
    assert row.believed_on_true == row.request - TOLERANCE
    assert (row.cause, row.goal) == ("choice", "rule")


# --- 13. choice in a later hour ------------------------------------------------------------------


def _read_optima(monkeypatch):
    """Records (hour, stage optima) of every plan the goal test is given."""
    seen = []
    original = asa_diagnosis_day.forbidding_goal

    def spy(controller, view, plan, take, request, **options):
        seen.append((view.hour, plan.optima))
        return original(controller, view, plan, take, request, **options)

    monkeypatch.setattr(asa_diagnosis_day, "forbidding_goal", spy)
    return seen


def test_choice_goal_five_in_the_second_hour(monkeypatch):
    seen = _read_optima(monkeypatch)
    cars = (TrueCar(3, 2.0, 2.0), TrueCar(3, 2.0, 0.75))
    forecasts = (ForecastCar(3.0, 2.0, 4.0, 0.0), ForecastCar(3.0, 2.0, 4.0, 0.0))
    day = Day("hand", DATE, 2, 0, 0.0, cars, forecasts, 0.1)
    method = AsaEsController(np.array([1.0, 1.0]), n_cars=2, n_recovery=0, time_limit=LIMIT)
    found = _path(day, method, least_laxity_first, [1.0, 1.0])
    row, groups, tests = _only(found)
    assert groups["neither"] == 1
    np.testing.assert_allclose(found[3].planned[:, 1:], [[0.5, 0.5], [0.5, 0.5]], atol=CLOSE)
    np.testing.assert_allclose(found[4].planned[:, 1:], [[1.0, 1.0], [0.0, 0.0]], atol=CLOSE)
    np.testing.assert_allclose(method.hour_plans[1].optima, [0, 0, 0, -3, 1], atol=CLOSE)
    ((hour, optima),) = seen
    assert hour == 2
    np.testing.assert_allclose(optima, [0, 0, 0, -1, 0.5], atol=CLOSE)
    assert (row.hour, row.cause, row.goal) == (2, "choice", "share")
    assert tests == {"optimal": 4, "infeasible": 1}
    assert (row.method_delivered, row.reference_delivered) == pytest.approx((0.75, 1.0), abs=CLOSE)
    assert row.shortfall == pytest.approx(0.25, abs=CLOSE)
    assert row.overplanned == pytest.approx(0.25, abs=CLOSE)
    assert (row.method_low, row.method_high) == pytest.approx((1.75, 1.75), abs=CLOSE)
    assert (row.believed_most, row.believed_on_true) == pytest.approx((4.0, 1.75), abs=CLOSE)
    deficits = (row.deficit_plugged, row.deficit_departure, row.deficit_rate, row.deficit_cap)
    assert deficits == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("form", FORMS)
def test_choice_goal_two_in_the_second_hour(form, monkeypatch):
    seen = _read_optima(monkeypatch)
    cars = (TrueCar(2, 1.0, 1.0), TrueCar(4, 2.0, 2.0), TrueCar(4, 1.0, 0.5))
    forecasts = (
        ForecastCar(2.0, 1.0, 5.0, 5.0),
        ForecastCar(2.5, 2.0, 1.5, 0.0),
        ForecastCar(3.0, 1.0, 1.0, 1.0),
    )
    day = Day("hand", DATE, 3, 0, 0.0, cars, forecasts, 0.1)
    method = form(np.array([0.5, 1.0, 0.0]), n_cars=3, n_recovery=0, time_limit=LIMIT)
    found = _path(day, method, earliest_deadline_first, [0.5, 1.0, 0.0])
    row, groups, tests = _only(found)
    assert groups["neither"] == 2
    planned = [[0.5, 0, 0], [0, 0, 0], [0, 1, 0]]
    np.testing.assert_allclose(found[3].planned[:, 1:], planned, atol=CLOSE)
    np.testing.assert_allclose(
        found[4].planned[:, 1:], [[0.5, 0, 0], [0, 1, 0], [0] * 3], atol=CLOSE
    )
    share = (1.25,) if form is AsaEsController else ()
    np.testing.assert_allclose(method.hour_plans[1].optima, [0, 4.5, 0, -3.5, *share], atol=CLOSE)
    ((hour, optima),) = seen
    assert hour == 2
    share = (1.0,) if form is AsaEsController else ()
    np.testing.assert_allclose(optima, [0, 0, 0, -2, *share], atol=CLOSE)
    assert (row.hour, row.cause, row.goal) == (2, "choice", "need")
    assert tests == {"optimal": 1, "infeasible": 1}
    assert (row.method_delivered, row.reference_delivered) == pytest.approx((0.5, 1.0), abs=CLOSE)
    assert found[4].delivered[0] == pytest.approx(0.5, abs=CLOSE)
    assert row.shortfall == pytest.approx(0.5, abs=CLOSE)
    assert row.overplanned == pytest.approx(0.5, abs=CLOSE)
    assert (row.method_low, row.method_high) == pytest.approx((0.0, 2.5), abs=CLOSE)
    assert (row.believed_most, row.believed_on_true) == pytest.approx((2.0, 1.5), abs=CLOSE)
    assert row.deficit_departure == pytest.approx(1.0, abs=CLOSE)
    assert row.deficit_plugged == row.deficit_rate == row.deficit_cap == 0.0


# --- 14. left_earlier in the row's own hour; the tolerance ---------------------------------------


def test_a_car_present_in_its_last_hour_has_not_left_earlier():
    cars = (TrueCar(2, 1.0, 1.0), TrueCar(3, 1.0, 1.0), TrueCar(4, 1.0, 2.0))
    forecasts = (
        ForecastCar(3.0, 1.0, 1.0, 0.0),
        ForecastCar(3.5, 1.0, 1.0, 0.0),
        ForecastCar(4.0, 1.0, 2.0, 0.0),
    )
    day = Day("hand", DATE, 3, 0, 0.0, cars, forecasts, 0.1)
    method = _made_up([0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1])
    reference = _made_up([0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1])
    row, groups, _ = _only(_path(day, method, reference, [1.0, 2.0, 1.0]))
    assert groups["neither"] == 2
    assert (row.hour, row.cause, row.method_high, row.reference_high) == (2, "state", 1.0, 2.0)
    assert row.cars_left_earlier == 1
    found = [(car.true_departure, car.present, car.left_earlier) for car in row.state_cars]
    assert found == [(2, False, True), (3, True, False), (4, True, False)]
    assert [car.believed_departure for car in row.state_cars] == [3.0, 3.5, 4.0]


def test_the_tolerance_is_the_failure_threshold_of_the_replay():
    assert asa_diagnosis_day.TOLERANCE == replay.FAILURE_KWH == 1e-4
    assert asa_diagnosis_lp.TOLERANCE == replay.FAILURE_KWH


MARGIN = 2e-4  # kWh: twice the failure threshold, so the hour fails (plan section 1)


@pytest.mark.parametrize(
    "case, cause, goal",
    [("choice", "choice", "rule"), ("belief", "belief", ""), ("state", "state", "")],
)
def test_a_miss_by_twice_the_threshold_gets_its_cause(case, cause, goal):
    if case == "state":
        cars = (TrueCar(3, 1.0, 2.0), TrueCar(3, 1.0, 1.0))
        forecasts = (ForecastCar(3.0, 1.0, 2.0, 0.0), ForecastCar(3.0, 1.0, 1.0, 0.0))
        day = Day("hand", DATE, 2, 0, 0.0, cars, forecasts, 0.1)
        y = 1.0 + MARGIN
        method = _made_up([0, 0], [0, 1], [1, 0])
        reference = _made_up([0, 0], [1, 0], [0.5, y - 0.5])
        requests = [1.0, y]
    else:
        cap = 2.0 if case == "choice" else 1.5 - MARGIN
        forecast = ForecastCar(3.0, 2.0, cap, 0)
        day = Day("hand", DATE, 2, 0, 0.0, (TrueCar(3, 2.0, 2.0),), (forecast,), 0.1)
        method = _made_up([0], [0.5], [1.0 - MARGIN])
        reference = _made_up([0], [0.5], [1.0])
        requests = [0.5, 1.0]
    row, groups, _ = _only(_path(day, method, reference, requests))
    assert groups["neither"] == 1
    assert row.hour == 2 and row.shortfall == pytest.approx(MARGIN, abs=1e-12)
    believed = {"choice": 1.5, "belief": 1.0 - MARGIN, "state": 1.0}[case]
    assert row.believed_on_true == pytest.approx(believed, abs=1e-12)
    assert (row.cause, row.goal) == (cause, goal)
