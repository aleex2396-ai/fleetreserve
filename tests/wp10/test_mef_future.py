"""The closed-form check of `mef_future` against a linear programme of the same sets.

1. On 1,200 seeded made-up futures (1 to 5 cars, 0 to 5 remaining hours, power limits, needs and
   caps on a half-kWh grid so that many inequalities hold with equality, some cars whose need is
   above their cap, some hours of zero width), `meets` equals the feasibility of the linear
   programme `lp_meets` (variables: each car's power in each remaining hour within its limit;
   rows: each car's energy in `[least, room]`, the fleet's energy in each hour in `[lo, hi]`;
   `lp.solve_lp` with the 30-second limit of `lp.BOX_SECONDS`). Every solve ends "optimal" or
   "infeasible"; both answers occur many times, and every one of the three parts (`within_cars`,
   `enough`, `roomy`) is the only failing part of some infeasible future.
2. Hand cases of one car and one hour that each part alone catches: a need above the cap
   (`within_cars`); a need above what the hours allow, or a box whose highest request is below
   what the need forces (`enough`); a box whose lowest request is above what the cap allows
   (`roomy`); and no remaining hour (needs met or not, a car already above its cap).
3. `enough` can only turn true and `roomy` only turn false as each car's `least` and `room` fall
   by the same draw (the monotone property `mef_controller` uses), on 300 seeded futures.
4. Eight remaining hours (the plan's longest window) are taken and checked; refusals: more than
   `MAX_HOURS` hours, `lo` above `hi`, a negative power limit (-1 and -1e-7 kWh), a value that is
   not finite in any of the five fields, shapes that do not fit. A future holds read-only copies
   of the arrays it is given.
5. The tolerance: in each of the three parts a shortfall of half `TOLERANCE` counts as met and one
   of twice `TOLERANCE` does not; one of exactly `TOLERANCE` counts as met.
"""

import random

import numpy as np
import pytest

from fleetreserve import mef_future
from fleetreserve.lp import BOX_SECONDS, solve_lp
from fleetreserve.mef_future import Future, enough, meets, roomy, within_cars


def lp_meets(future: Future) -> bool:
    cars, k = future.high.shape
    n = cars * k + 1  # one fixed variable, so a future of no hour is still a programme
    bounds = [(0.0, float(future.high[i, j])) for i in range(cars) for j in range(k)]
    bounds.append((0.0, 0.0))
    rows, rhs = [], []
    for i in range(cars):
        row = np.zeros(n)
        row[i * k : (i + 1) * k] = 1.0
        rows += [row, -row]
        rhs += [future.room[i], -future.least[i]]
    for j in range(k):
        row = np.zeros(n)
        row[[i * k + j for i in range(cars)]] = 1.0
        rows += [row, -row]
        rhs += [future.hi[j], -future.lo[j]]
    found = solve_lp(
        np.zeros(n), A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds, time_limit=BOX_SECONDS
    )
    assert found.outcome in ("optimal", "infeasible"), found.outcome
    return found.outcome == "optimal"


def _grid(rng: random.Random, low: float, high: float) -> float:
    return round(rng.uniform(low, high) * 2) / 2


def made_up_future(rng: random.Random) -> Future:
    """Limits drawn about one made-up point of the sets, so that about half the futures meet."""
    cars, k = rng.randint(1, 5), rng.randint(0, 5)
    high = np.array(
        [[0.0 if rng.random() < 0.25 else _grid(rng, 0, 3) for _ in range(k)] for _ in range(cars)]
    ).reshape(cars, k)
    point = np.array([[_grid(rng, 0, h) for h in row] for row in high]).reshape(cars, k)
    energy, fleet = point.sum(axis=1), point.sum(axis=0)
    least = [x - _grid(rng, -0.5, 2) for x in energy]
    room = [x + _grid(rng, -0.5, 2) for x in energy]
    lo = [max(x - _grid(rng, -0.5, 2), 0.0) for x in fleet]
    hi = [
        max(low, x + (0.0 if rng.random() < 0.2 else _grid(rng, -0.5, 2)))
        for low, x in zip(lo, fleet, strict=True)
    ]
    return Future(high, least, room, lo, hi)


def test_the_closed_form_equals_the_programme_on_1200_made_up_futures():
    rng = random.Random(20261114)
    answers = {True: 0, False: 0}
    only = {"within_cars": 0, "enough": 0, "roomy": 0}
    for _ in range(1200):
        future = made_up_future(rng)
        closed = meets(future)
        assert closed == lp_meets(future), future
        answers[closed] += 1
        parts = {
            "within_cars": within_cars(future),
            "enough": enough(future),
            "roomy": roomy(future),
        }
        failing = [name for name, held in parts.items() if not held]
        if len(failing) == 1:
            only[failing[0]] += 1
    assert min(answers.values()) >= 300, answers
    assert min(only.values()) >= 20, only


def _one(high, least, room, lo, hi):
    return Future(np.array([[high]]), [least], [room], [lo], [hi])


def test_hand_cases_of_one_car_and_one_hour():
    fine = _one(3.0, 2.0, 4.0, 1.0, 5.0)
    assert meets(fine) and lp_meets(fine)
    over_cap = _one(3.0, 2.0, 1.5, 0.0, 5.0)  # needs 2, may draw only 1.5
    assert not within_cars(over_cap) and enough(over_cap) and roomy(over_cap)
    short = _one(3.0, 3.5, 5.0, 0.0, 5.0)  # needs 3.5 in one hour of 3
    assert within_cars(short) and not enough(short) and roomy(short)
    forced = _one(3.0, 2.5, 5.0, 0.0, 2.0)  # the need forces 2.5, the box allows at most 2
    assert within_cars(forced) and not enough(forced) and roomy(forced)
    starved = _one(3.0, 0.0, 1.0, 2.0, 4.0)  # the box asks 2 at least, the cap allows 1
    assert within_cars(starved) and enough(starved) and not roomy(starved)
    for future in (over_cap, short, forced, starved):
        assert not lp_meets(future)
    met = Future(np.zeros((2, 0)), [0.0, -1.0], [0.0, 2.0], [], [])
    assert meets(met) and lp_meets(met)
    unmet = Future(np.zeros((2, 0)), [0.0, 0.5], [0.0, 2.0], [], [])
    assert not enough(unmet) and not meets(unmet) and not lp_meets(unmet)
    over = Future(np.zeros((1, 0)), [-1.0], [-0.5], [], [])  # drawn above its cap already
    assert not roomy(over) and not lp_meets(over)


def test_enough_only_turns_true_and_roomy_only_false_as_cars_draw_more():
    rng = random.Random(7)
    for _ in range(300):
        future = made_up_future(rng)
        draws = np.array([_grid(rng, 0, 1) for _ in range(future.least.size)])
        before = (enough(future), roomy(future))
        later = Future(future.high, future.least - draws, future.room - draws, future.lo, future.hi)
        assert enough(later) or not before[0]
        assert before[1] or not roomy(later)


def test_refusals():
    eight = Future(np.ones((2, 8)), [8.0, 0.0], [8.0, 9.0], np.zeros(8), np.full(8, 2.0))
    assert mef_future.MAX_HOURS == 8 and meets(eight) and lp_meets(eight)
    nine = mef_future.MAX_HOURS + 1
    with pytest.raises(ValueError, match="above the limit"):
        Future(np.zeros((1, nine)), [0.0], [1.0], np.zeros(nine), np.ones(nine))
    with pytest.raises(ValueError, match="lo above hi"):
        _one(1.0, 0.0, 1.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="at least 0"):
        _one(-1.0, 0.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="at least 0"):
        _one(-1e-7, 0.0, 1.0, 0.0, 1.0)  # no tolerance on the sign of a limit
    for k in range(5):
        values = [1.0, 0.0, 1.0, 0.0, 1.0]
        values[k] = float("nan")
        with pytest.raises(ValueError, match="must be finite"):
            _one(*values)
        values[k] = float("inf") if k != 0 else -float("inf")
        with pytest.raises(ValueError, match="must be finite"):
            _one(*values)
    with pytest.raises(ValueError, match="one entry per car"):
        Future(np.zeros((1, 1)), [0.0], [1.0, 2.0], [0.0], [1.0])
    with pytest.raises(ValueError, match="one entry per remaining hour"):
        Future(np.zeros((1, 1)), [0.0], [1.0], [0.0], [1.0, 2.0])


@pytest.mark.parametrize("times", [0.5, 2.0])
def test_the_tolerance_of_each_part(times):
    gap = times * mef_future.TOLERANCE
    held = times < 1
    assert within_cars(_one(3.0, 1.0 + gap, 1.0, 0.0, 5.0)) == held
    assert enough(_one(0.3, 0.3 + gap, 5.0, 0.0, 5.0)) == held  # the need above the hour's most
    assert roomy(_one(3.0, 0.0, 1.0, 1.0 + gap, 5.0)) == held  # the box above the room


def test_a_futures_arrays_are_read_only_copies():
    high, least = np.array([[3.0]]), np.array([2.0])
    future = Future(high, least, [4.0], [1.0], [5.0])
    high[0, 0], least[0] = 0.0, 9.0  # the caller's arrays are not the future's
    assert future.high[0, 0] == 3.0 and future.least[0] == 2.0
    for key in ("high", "least", "room", "lo", "hi"):
        with pytest.raises(ValueError, match="read-only"):
            getattr(future, key)[0] = 0.0


def test_a_shortfall_of_exactly_the_tolerance_counts_as_met():
    """Each part compares with `<=`: built so that both sides are the same float, a shortfall of
    exactly `TOLERANCE` still holds."""
    tol = mef_future.TOLERANCE
    assert within_cars(_one(0.0, tol, 0.0, 0.0, 0.0))  # the need tol, the cap 0
    assert enough(_one(0.0, tol, tol, 0.0, 0.0))  # the need tol, an hour of most 0 and request 0
    assert roomy(_one(0.0, 0.0, 0.0, tol, tol))  # the box asks tol of a room of 0
