"""The ends of `mef_controller`'s interval of feasible totals, its bisections and its counts,
on views and a day built by hand, with every number worked out here.

1. A total met only at the fleet's most is kept (module text, "The ends of the interval"):
   - one car (believed departure 3, rate 2, cap 10, need 4) in hour 1 of two service hours,
     target 1, box `[0, 5]` in both hours: the car can take 2 in hour 2, so it must take its
     most 2 in hour 1; the controller plans 2 and counts a deviating hour;
   - three cars in hour 2 of four service hours, drawn 3, 2 and 1, target 1.5, box highest
     requests 1.25 in hours 3 and 4: X (departure 3, rate 2, cap 10, need 5) can take nothing
     after hour 2 and still needs 2, its most; Y (departure 6, rate 3, cap 20, need 9; recovery
     supply 3 x 1 = 3) still needs 4 with a most of 3; Z (departure 5.5, rate 1, cap 8, need 4;
     supply 1 x 0.5) still needs 2.5 with a most of 1. Least-laxity-first fills X (laxity 0),
     then Z (0.5), then Y (1.67), so below the fleet's most 6 Y takes less than 3 and the needs
     left for hours 3 and 4 (0, 1 and 1.5 at the most) exceed the box's 2.5: the controller
     plans every car's most;
   - the first car with a need half `TOLERANCE` above 4: met at the most only through the
     tolerance, so the smallest total that meets it is the most itself (`low` is the fleet's
     most, not a bisection's point below it).
   In all three, `high` placed `BISECTION_KWH` below the most leaves no feasible total.
2. A day built by hand on which that end decides the losses: one car (departure 3, rate 2,
   room 8, rho 0.5, so need 4), two service hours, requests 1 and 1, hour-zero target 0. The
   controller plans 0, then 2 (the request 1 would leave 3 for hour 2, which allows 2), then 2:
   the car gets its need (driver loss below 1e-9) and hour 1 fails (slot loss 1).
   Least-laxity-first plans 0, 1, 1: no hour fails, and the car is 2 short of 4 (driver loss
   0.5); so does the controller when step 3 finds no total (a subclass whose `nearest_total`
   is None), counting hours 1 and 2 as hours with no feasible total.
3. The bisections are plain halving of `[0, sum of most]` down to `BISECTION_KWH` (1e-9), kept
   on the holding side: on a view whose smallest feasible total is found by bisection, and on
   one whose largest is, `nearest_total` is exactly the point an independent halving finds.
4. In exact arithmetic no feasible total is lost to the bisections: on 400 seeded made-up views
   whose numbers are multiples of a quarter (so every sum is exact), every total on the grid of
   quarters up to the fleet's most whose believed future is met with the tolerance set to 0 is
   returned by `nearest_total` as itself when it is the target. Totals met only at 0 and only at
   the fleet's most occur.
5. `within_cars` is read at the total 0 (module text): one car whose need is above its cap by
   about `TOLERANCE` (need 1.33, cap 1.329999999), for which `within_cars` holds at the total 0
   and fails, by rounding, at the totals 1.1 and 2: least-laxity-first's split of the target
   1.1 is found unmet, step 3 returns the target itself, and the hour is counted as deviating at
   least-laxity-first's own total.
6. `deviation_counts()` returns a copy: a later hour does not change counts already returned.
7. The bisection for `low` starts at 0: a day built by hand on which `low` reaches the driver
   loss. One car (true departure 1, rate 4, room 6, rho 0.5, so need 3), one service hour,
   request 0, hour-zero target 0, box `[0, 1.3]`; the belief says the car stays to 2 with need
   3, so hour 1 can take at most 1.3 and `enough` needs at least 1.7 in hour 0, found by plain
   halving of `[0, 4]`. With the departure seen at the start of an hour, the controller, not
   yet seeing the car gone, plans the rest of the believed need (1.3 to within the bisection)
   in hour 1, a second deviating hour; the car has left after hour 0, so its energy is `low`
   and the driver loss is `(3 - low) / 3`, each pinned to the last bit (a start of 1e-9 or
   -1e-9 moves both).
8. Four comparisons at the tolerance's edge, each on one car in hour 0 of one service hour,
   departure 2 unless stated, with the plan and the counts pinned exactly:
   - `enough` read at 0 (rate 10, cap 10, need 2 + 1.5e-9, box `[0, 2]`, target 0): it fails at
     0 by 0.5e-9 beyond the tolerance and holds at 1e-9, so `low` is a bisection's point, not 0;
   - `roomy` by the cars' limits (rate 3, cap 50, need 0, box `[3 + 1.5e-9, 10]`, target 1): the
     box asks 0.5e-9 beyond the tolerance above the car's limit in hour 1, so no total is
     feasible;
   - crossed ends (departure 2.5, rate 2, cap 3, need 4.000000001, so a recovery supply of 1;
     box `[2.0000000001, 5]`, target 1.5): the bisected `low` lies above `high` by less than
     1e-9, so no total is feasible;
   - `within_cars` read at 0 (departure 2.6, rate 4, so a recovery supply of 2.4; cap 0.25,
     need 2.6500000010000004, box `[0, 10]`, target 0): the car's remaining need is above its
     remaining room by the tolerance to within rounding, `within_cars` fails at 0 and holds at
     1e-9, so no total is feasible.
"""

import dataclasses
import datetime
import random

import numpy as np
import pytest

from fleetreserve import mef_controller, mef_future
from fleetreserve.controller import Belief, HourView, oracle_belief
from fleetreserve.controller_replay import run_controller
from fleetreserve.day import Day, TrueCar
from fleetreserve.mef_controller import MefController
from fleetreserve.mef_future import enough, meets, roomy, within_cars
from fleetreserve.priority_rules import least_laxity_first, most
from fleetreserve.rungs import DEPARTURE_SEEN

EDGE = 2e-9


def _view(hour, n, target, drawn, belief, plugged=None):
    plugged = np.ones(len(drawn), bool) if plugged is None else np.array(plugged)
    zeros = np.zeros(hour)
    return HourView(hour, n, target, zeros, zeros, np.array(drawn, float), plugged, belief)


def _one_car(need=4.0):
    return _view(1, 2, 1.0, [0.0], Belief([3.0], [2.0], [10.0], [need])), ([0, 0], [5, 5])


def _three_cars():
    belief = Belief([3.0, 6.0, 5.5], [2.0, 3.0, 1.0], [10.0, 20.0, 8.0], [5.0, 9.0, 4.0])
    return _view(2, 4, 1.5, [3.0, 2.0, 1.0], belief), ([0, 0, 0, 0], [5, 5, 1.25, 1.25])


AT_THE_MOST = {
    "one car": _one_car,
    "three cars": _three_cars,
    "one car met through the tolerance": lambda: _one_car(4.0 + 0.5 * mef_future.TOLERANCE),
}


def _after(controller, view, total):
    return controller.future(view, least_laxity_first(dataclasses.replace(view, target=total)))


@pytest.mark.parametrize("name", AT_THE_MOST)
def test_a_total_met_only_at_the_fleets_most_is_kept(name):
    view, (lo, hi) = AT_THE_MOST[name]()
    controller = MefController(lo, hi)
    top = float(most(view).sum())
    assert not meets(controller.future(view, least_laxity_first(view)))
    assert roomy(_after(controller, view, top)) and enough(_after(controller, view, top))
    assert not enough(_after(controller, view, top - 2 * mef_controller.BISECTION_KWH))
    total = controller.nearest_total(view)
    assert total is not None and top - mef_controller.BISECTION_KWH <= total <= top
    if name == "one car met through the tolerance":
        assert total == top  # the smallest feasible total is the most itself
    assert np.allclose(controller(view), most(view), rtol=0, atol=EDGE)
    assert controller.deviation_counts() == {
        "hours": 1,
        "deviating_hours": 1,
        "deviating_service_hours": 1,
        "no_feasible_hours": 0,
    }


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_a_day_on_which_the_total_at_the_most_decides_the_losses(seen):
    day = Day("hand", datetime.date(2026, 1, 5), 2, 0, 0.5, (TrueCar(3, 2.0, 8.0),))
    requests = np.array([1.0, 1.0])

    def run(controller):
        belief = oracle_belief(day)
        return run_controller(
            day,
            controller,
            belief,
            np.zeros(2),
            requests,
            hour_zero_target=0.0,
            departure_seen=seen,
        )

    controller = MefController([0.0, 0.0], [5.0, 5.0])
    found = run(controller)
    assert np.allclose(found.planned, [[0.0, 2.0, 2.0]], rtol=0, atol=EDGE)
    assert found.first_failure == 1 and found.slot_loss == pytest.approx(1.0)
    assert 0.0 <= found.driver_loss < 1e-9
    assert controller.deviation_counts() == {
        "hours": 3,
        "deviating_hours": 2,
        "deviating_service_hours": 2,
        "no_feasible_hours": 0,
    }
    plain = run(least_laxity_first)
    assert np.array_equal(plain.planned, [[0.0, 1.0, 1.0]])
    assert plain.first_failure is None and plain.driver_loss == pytest.approx(0.5)

    class Lost(MefController):
        def nearest_total(self, view):
            return None

    lost = Lost([0.0, 0.0], [5.0, 5.0])
    again = run(lost)
    for field in dataclasses.fields(plain):
        assert np.array_equal(getattr(again, field.name), getattr(plain, field.name)), field.name
    assert lost.deviation_counts() == {
        "hours": 3,
        "deviating_hours": 0,
        "deviating_service_hours": 0,
        "no_feasible_hours": 2,
    }


def _halving(holds, start, end, rising):
    """Plain halving to 1e-9 kWh, written out here."""
    while end - start > 1e-9:
        middle = 0.5 * (start + end)
        if holds(middle) == rising:
            end = middle
        else:
            start = middle
    return end if rising else start


def test_the_bisections_halve_from_0_to_the_fleets_most_down_to_1e_9():
    assert mef_controller.BISECTION_KWH == 1e-9
    view, (lo, hi) = _one_car()  # the smallest feasible total, about 2, by bisection
    controller = MefController(lo, hi)
    top = float(most(view).sum())
    assert top == 2.0 and not enough(_after(controller, view, 0.0))
    low = _halving(lambda u: enough(_after(controller, view, u)), 0.0, top, rising=True)
    assert controller.nearest_total(view) == low and 2.0 - 1e-9 <= low <= 2.0
    # one car (departure 3, rate 4, cap 3, need 0) in hour 1, target 2; hour 2 asks at least 2,
    # so the largest feasible total, about 1, is found by bisection over [0, 3]
    view = _view(1, 2, 2.0, [0.0], Belief([3.0], [4.0], [3.0], [0.0]))
    controller = MefController([0.0, 2.0], [5.0, 5.0])
    top = float(most(view).sum())
    assert top == 3.0 and not roomy(_after(controller, view, top))
    high = _halving(lambda u: roomy(_after(controller, view, u)), 0.0, top, rising=False)
    assert controller.nearest_total(view) == high and 1.0 <= high <= 1.0 + 1e-9


def _quarters(rng, low, high):
    return rng.randint(round(4 * low), round(4 * high)) / 4


def _made_up_view(rng):
    """A view and a controller; one in five is met only at the fleet's most (every car must take
    its most now and the box asks nothing later) and one in five only at 0 (the box asks in every
    later hour all that the cars can take, and every car's cap leaves room for that alone)."""
    n, recovery = rng.randint(1, 4), rng.randint(0, 2)
    hour, cars = rng.randint(0, n), rng.randint(1, 4)
    departure = [rng.randint(2 * hour + 1, 2 * (n + recovery + 1)) / 2 for _ in range(cars)]
    rate = [rng.choice([0.5, 1.0, 1.5, 2.0, 3.0]) for _ in range(cars)]
    cap = [_quarters(rng, 0, 6) for _ in range(cars)]
    need = [_quarters(rng, 0, c + 1) for c in cap]
    drawn = [_quarters(rng, 0, min(c, 2)) for c in cap]
    plugged = [rng.random() < 0.85 for _ in range(cars)]
    lo = [_quarters(rng, 0, 3) for _ in range(n)]
    hi = [x + _quarters(rng, 0, 3) for x in lo]
    mode = rng.random() if hour < n else 1.0
    if mode < 0.4:
        plugged = [True] * cars
        view = _view(hour, n, 0.0, drawn, Belief(departure, rate, cap, need), plugged)
        future = MefController(lo, hi).future(view, np.zeros(cars))
        supply = [r * max(d - n - 1, 0.0) for r, d in zip(rate, departure, strict=True)]
        if mode < 0.2:
            limit = most(view)
            need = [x + m + s for x, m, s in zip(drawn, limit, supply, strict=True)]
            lo[hour:] = hi[hour:] = [0.0] * (n - hour)
        else:
            cap = [x + h for x, h in zip(drawn, future.high.sum(axis=1), strict=True)]
            need = list(drawn)
            lo[hour:] = hi[hour:] = list(future.high.sum(axis=0))
    view = _view(hour, n, 0.0, drawn, Belief(departure, rate, cap, need), plugged)
    return view, MefController(lo, hi)


def test_in_exact_arithmetic_no_feasible_total_is_lost(monkeypatch):
    rng = random.Random(20261115)
    met, ends = 0, {"only 0": 0, "only the most": 0}
    for _ in range(400):
        view, controller = _made_up_view(rng)
        top = float(most(view).sum())
        grid = sorted({k / 4 for k in range(int(4 * top) + 1)} | {top})
        with monkeypatch.context() as patch:
            patch.setattr(mef_future, "TOLERANCE", 0.0)
            feasible = [u for u in grid if meets(_after(controller, view, u))]
        for u in feasible:
            assert controller.nearest_total(dataclasses.replace(view, target=u)) == u, (view, u)
        met += len(feasible)
        if top > 0 and feasible == [0.0]:
            ends["only 0"] += 1
        if top > 0 and feasible == [top]:
            ends["only the most"] += 1
    assert met >= 400 and min(ends.values()) >= 5, (met, ends)


def test_within_cars_is_read_at_the_total_0():
    view = _view(1, 2, 1.1, [0.0], Belief([3.0], [2.0], [1.329999999], [1.33]))
    controller = MefController([0.0, 0.0], [5.0, 5.0])
    assert within_cars(_after(controller, view, 0.0))
    assert not within_cars(_after(controller, view, 1.1))
    assert not within_cars(_after(controller, view, 2.0))
    planned = least_laxity_first(view)
    assert not meets(controller.future(view, planned))
    assert controller.nearest_total(view) == 1.1
    assert np.array_equal(controller(view), planned)
    assert controller.deviation_counts() == {
        "hours": 1,
        "deviating_hours": 1,
        "deviating_service_hours": 1,
        "no_feasible_hours": 0,
    }


def test_the_counts_returned_are_a_copy():
    view, (lo, hi) = _one_car()
    controller = MefController(lo, hi)
    controller(view)
    first = controller.deviation_counts()
    controller(view)
    assert first == {
        "hours": 1,
        "deviating_hours": 1,
        "deviating_service_hours": 1,
        "no_feasible_hours": 0,
    }
    assert controller.deviation_counts()["hours"] == 2


def test_the_bisection_for_low_starts_at_0_on_a_day_with_a_driver_loss():
    seen = DEPARTURE_SEEN[0]
    assert seen == "hour-start"
    day = Day("hand", datetime.date(2026, 1, 5), 1, 0, 0.5, (TrueCar(1, 4.0, 6.0),))
    belief = Belief([2.0], [4.0], [6.0], [3.0])
    controller = MefController([0.0], [1.3])
    view = _view(0, 1, 0.0, [0.0], belief)
    assert float(most(view).sum()) == 4.0 and not enough(_after(controller, view, 0.0))
    low = _halving(lambda u: enough(_after(controller, view, u)), 0.0, 4.0, rising=True)
    assert low == 1.699999999254942
    found = run_controller(
        day,
        controller,
        belief,
        np.zeros(1),
        np.array([0.0]),
        hour_zero_target=0.0,
        departure_seen=seen,
    )
    assert found.planned.tolist() == [[low, 1.2999999998137355]]
    assert found.slot_loss == 0.0
    assert found.driver_loss == 0.433333333581686
    assert controller.deviation_counts() == {
        "hours": 2,
        "deviating_hours": 2,
        "deviating_service_hours": 1,
        "no_feasible_hours": 0,
    }


def _hour_zero(target, rate, cap, need, departure=2.0):
    return _view(0, 1, target, [0.0], Belief([departure], [rate], [cap], [need]))


def _planned(view, lo, hi):
    controller = MefController(lo, hi)
    return controller, controller(view).tolist(), controller.deviation_counts()


def test_enough_is_read_at_0_at_the_tolerances_edge():
    view = _hour_zero(0.0, 10.0, 10.0, 2.0 + 1.5e-9)
    controller, planned, counts = _planned(view, [0.0], [2.0])
    assert not enough(_after(controller, view, 0.0))
    assert enough(_after(controller, view, 1e-9))
    assert planned == [5.820766091346741e-10]
    assert counts == {
        "hours": 1,
        "deviating_hours": 1,
        "deviating_service_hours": 0,
        "no_feasible_hours": 0,
    }


NO_FEASIBLE_IN_HOUR_0 = {
    "hours": 1,
    "deviating_hours": 0,
    "deviating_service_hours": 0,
    "no_feasible_hours": 1,
}


def test_roomy_counts_a_request_just_above_a_cars_limit():
    view = _hour_zero(1.0, 3.0, 50.0, 0.0)
    controller, planned, counts = _planned(view, [3.0 + 1.5e-9], [10.0])
    assert within_cars(_after(controller, view, 0.0))
    assert not roomy(_after(controller, view, 0.0))
    assert planned == [1.0] and counts == NO_FEASIBLE_IN_HOUR_0


def test_ends_crossed_by_less_than_the_bisections_width_leave_no_feasible_total():
    view = _hour_zero(1.5, 2.0, 3.0, 4.000000001, departure=2.5)
    controller, planned, counts = _planned(view, [2.0000000001], [5.0])
    top = float(most(view).sum())
    low = _halving(lambda u: enough(_after(controller, view, u)), 0.0, top, rising=True)
    high = _halving(lambda u: roomy(_after(controller, view, u)), 0.0, top, rising=False)
    assert (low, high) == (1.0000000009313226, 1.0) and low - high <= 1e-9
    assert controller.nearest_total(view) is None
    assert planned == [1.5] and counts == NO_FEASIBLE_IN_HOUR_0


def test_within_cars_is_read_at_0_with_a_recovery_supply():
    view = _hour_zero(0.0, 4.0, 0.25, 2.6500000010000004, departure=2.6)
    controller, planned, counts = _planned(view, [0.0], [10.0])
    assert not within_cars(_after(controller, view, 0.0))
    assert within_cars(_after(controller, view, 1e-9))
    assert planned == [0.0] and counts == NO_FEASIBLE_IN_HOUR_0
