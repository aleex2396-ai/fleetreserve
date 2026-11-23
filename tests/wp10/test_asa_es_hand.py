"""The fifth goal of `asa-es` (`asa_equal_share`) by hand: five programmes worked out below.

Beliefs are (departure, rate, cap, need); nothing is drawn yet; every car is counted as plugged in;
no recovery hours; the default number of pieces, K = 8. A car's chord pieces in an hour whose
upper limit is U have breakpoints k U / 8, k = 0 .. 8, and slope b_k + b_(k+1) on the piece
[b_k, b_(k+1)] (the chord of the square between its two ends).

Case 1: two identical cars and a target one car could meet alone. Hour 1 of 1, target 1, A and B
(3, 2, 10, 0): each can take 2 in the hour. Goals 1 to 4 fix only the hour's total, 1 (goal 4
weighs the hour, not the cars), so every split ties under them. Breakpoints of 2 / 8 = 0.25: 0.5
is one. With a = A's power, the approximation of a^2 + (1 - a)^2 has slope 1.25 - 0.75 = 0.5 just
above a = 0.5 and -0.5 just below it: its only minimiser is (0.5, 0.5), the square's minimiser.
`asa` on the same view returns (0, 1): one car takes the whole hour (the solver's choice among
tied plans, which this test pins as observed; to 1e-7, as goal 4 may use the 1e-9 of
`STAGE_TOLERANCE` above goal 1's optimum).

Case 2: inside one piece the approximation still ties. Case 1 with target 0.8: the square's
minimiser is (0.4, 0.4); 0.4 lies inside the piece [0.25, 0.5] of both cars, where the
approximation is linear with the same slope 0.75 for both, so every split with both powers in
[0.25, 0.5], that is a in [0.3, 0.5], has the same approximate value 2 x 0.75 x 0.4 - 2 x 0.125
= 0.35 (each piece's line is 0.75 r - 0.125), the least. `asa-es` returns one of them: both
powers in [0.25, 0.5], differing by at most 0.2, and the fifth optimum 0.35.

Case 3: goals 1 to 4 fix everything and goal 5 changes nothing. Hour 1 of 2, target 3, centre
(-, 9): A (3, 1, 10, 0) and B (3, 2, 10, 0). The target is the hour's whole believed most, 1 + 2,
and the centre of hour 2 is above what both can take there, 3, so goals 1 and 3 put every car at
its upper limit in both hours: `asa-es` and `asa` plan (1, 1) for A and (2, 2) for B.

Case 4: different upper limits. Hour 1 of 1, target 1.2, A (3, 2, 10, 0) with upper limit 2 and
B (3, 1, 10, 0) with upper limit 1. The square's minimiser is (0.6, 0.6). The approximation:
A's breakpoints are multiples of 0.25, B's of 0.125. For a in (0.5, 0.575) B's power is in
(0.625, 0.7): slopes 1.25 (A) and 1.375 (B), so the approximation falls as a rises (1.25 -
1.375 < 0); for a in (0.575, 0.7) B's power is in (0.5, 0.625): slopes 1.25 and 1.125, so it
rises. Its only minimiser is a = 0.575, b = 0.625: not the square's. Sum of squares there
0.72125 against 0.72 at the square's minimiser: 0.00125 above, within the stated gap
2^2 / 256 + 1^2 / 256 = 0.01953125.

Case 5: the fifth goal sums over the horizon, so it can make the first hour less equal. Hour 1 of
2, target 2, centre (-, 2); A (3, 2, 1, 0), B (3, 2, 3, 0): A can take 1 in hour 1 (its cap),
B 2; both 2 in hour 2. Goal 1 fixes hour 1's total at 2, goal 3 hour 2's at 2, and the rooms
(1 and 3) then fix each car's energy: A 1, B 3. Goal 4 weighs hours, so every split with A
(a, 1 - a), B (2 - a, 1 + a), a in [0, 1], ties under goals 1 to 4. The square's horizon sum
a^2 + (1 - a)^2 + (2 - a)^2 + (1 + a)^2 is smallest at a = 0.5; so is the approximation's (at
a = 0.5 every power is a breakpoint; the slopes sum to 1.125 - 0.75 - 2.75 + 3.25 = 0.875 above
it and 0.875 - 1.25 - 3.25 + 2.75 = -0.875 below it). `asa-es` plans hour 1 as (0.5, 1.5), whose
sum of squares, 2.5, is above that of the tied plan with a = 1, hour 1 (1, 1), 2: the first hour
alone is not made more equal; the horizon is.

Also: the approximation on [0, 2] (`approximate_square`) is at least the square, at most
2^2 / 256 above it with that largest gap reached, equal to it at the 9 breakpoints, and convex;
a number of pieces that is not a whole number at least 1, and a time limit that is not a positive
finite number, are refused with `ValueError`.
"""

import numpy as np
import pytest

from fleetreserve import asa_equal_share
from fleetreserve.asa_equal_share import STAGES, plan_hour
from fleetreserve.asa_programme import plan_hour as asa_plan_hour
from fleetreserve.controller import Belief, HourView

TOLERANCE = 1e-7  # kWh: HiGHS's own primal feasibility tolerance is 1e-7
LIMIT = 300.0


def _view(hour, n, target, belief):
    cars = belief.n_cars
    return HourView(
        hour=hour,
        n_service=n,
        target=target,
        zeta_known=np.zeros(hour),
        requests_known=np.zeros(hour),
        drawn=np.zeros(cars),
        plugged=np.ones(cars, dtype=bool),
        belief=belief,
    )


def _plan(view, centre):
    found = plan_hour(view, centre, 0, time_limit=LIMIT)
    assert found.outcomes == ("optimal",) * len(STAGES) and len(STAGES) == 5
    return found


TWINS = Belief([3, 3], [2, 2], [10, 10], [0, 0])


def test_case_1_two_identical_cars_share_equally():
    view = _view(1, 1, 1.0, TWINS)
    np.testing.assert_allclose(_plan(view, [0.0]).powers[:, 0], [0.5, 0.5], atol=TOLERANCE)
    asa = asa_plan_hour(view, [0.0], 0, time_limit=LIMIT)
    np.testing.assert_allclose(asa.powers[:, 0], [0.0, 1.0], atol=TOLERANCE)


def test_case_2_inside_one_piece_the_approximation_ties():
    found = _plan(_view(1, 1, 0.8, TWINS), [0.0])
    a, b = found.powers[:, 0]
    assert abs(a + b - 0.8) <= TOLERANCE
    assert min(a, b) >= 0.25 - TOLERANCE and max(a, b) <= 0.5 + TOLERANCE
    assert abs(a - b) <= 0.2 + 2 * TOLERANCE
    assert abs(found.optima[4] - 0.35) <= 1e-6


def test_case_3_goals_1_to_4_fix_everything():
    view = _view(1, 2, 3.0, Belief([3, 3], [1, 2], [10, 10], [0, 0]))
    found = _plan(view, [0.0, 9.0])
    asa = asa_plan_hour(view, [0.0, 9.0], 0, time_limit=LIMIT)
    np.testing.assert_allclose(found.powers, [[1.0, 1.0], [2.0, 2.0]], atol=TOLERANCE)
    np.testing.assert_allclose(found.powers, asa.powers, atol=TOLERANCE)
    assert found.optima[:4] == asa.optima


def test_case_4_different_upper_limits():
    view = _view(1, 1, 1.2, Belief([3, 3], [2, 1], [10, 10], [0, 0]))
    found = _plan(view, [0.0])
    np.testing.assert_allclose(found.powers[:, 0], [0.575, 0.625], atol=TOLERANCE)
    squares = float((found.powers**2).sum())
    assert abs(squares - 0.72125) <= 1e-6
    gap = asa_equal_share.largest_gap(np.array([2.0, 1.0])).sum()
    assert gap == pytest.approx(0.01953125, abs=1e-15)
    assert squares - 0.72 <= gap


def test_case_5_the_horizon_is_made_more_equal_not_always_the_first_hour():
    view = _view(1, 2, 2.0, Belief([3, 3], [2, 2], [1, 3], [0, 0]))
    found = _plan(view, [0.0, 2.0])
    np.testing.assert_allclose(found.powers, [[0.5, 0.5], [1.5, 1.5]], atol=TOLERANCE)
    assert abs(float((found.powers[:, 0] ** 2).sum()) - 2.5) <= 1e-6
    tied = np.array([[1.0, 0.0], [1.0, 2.0]])  # a = 1: the same goals 1 to 4
    assert float((tied[:, 0] ** 2).sum()) == 2.0
    assert float((tied**2).sum()) > float((found.powers**2).sum())


def test_the_approximation_is_the_chord_interpolant_of_the_square():
    upper = 2.0
    r = np.linspace(0.0, upper, 4001)
    value = asa_equal_share.approximate_square(r, upper)
    assert np.all(value >= r**2 - 1e-12)
    gap = value - r**2
    assert gap.max() == pytest.approx(upper**2 / (4 * 8**2), rel=1e-6)
    knots = np.arange(9) * upper / 8
    np.testing.assert_allclose(asa_equal_share.approximate_square(knots, upper), knots**2)
    assert np.all(np.diff(value, 2) >= -1e-12)  # convex


@pytest.mark.parametrize("pieces", [0, -1, 1.5, True, "8"])
def test_a_number_of_pieces_that_is_not_a_whole_number_at_least_1_is_refused(pieces):
    with pytest.raises(ValueError, match="pieces"):
        plan_hour(_view(1, 1, 1.0, TWINS), [0.0], 0, time_limit=LIMIT, pieces=pieces)


@pytest.mark.parametrize("limit", [0.0, -1.0, float("inf"), float("nan"), True])
def test_a_bad_time_limit_is_refused(limit):
    with pytest.raises(ValueError, match="time limit"):
        plan_hour(_view(1, 1, 1.0, TWINS), [0.0], 0, time_limit=limit)
