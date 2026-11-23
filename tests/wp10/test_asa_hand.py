"""The linear-programme form of ASA (`asa_programme`) by hand: four programmes worked out below.

Beliefs are (departure, rate, cap, need); nothing is drawn yet; every car is counted as plugged in;
no recovery hours. `centre[s - 1]` is the box centre of service hour `s`; only the centres of the
hours after the view's hour are read.

Case 1: goal 2 decides, and the answer differs from EDF's, LLF's and the smoothed rule's. Hour 1
of 2, target 1.1, centre (0, 0).

- A (1.5, 2, 10, 0.6): leaves half-way through hour 1. Most in hour 1: 2 x 0.5 = 1; nothing in
  hour 2. Laxity (1.5 - 1) - 0.6 / 2 = 0.2.
- B (3, 1, 10, 1.5): most 1 in hour 1 and 1 in hour 2. Laxity (3 - 1) - 1.5 / 1 = 0.5.
- C (3, 1, 10, 0): most 1, laxity 2.

Goal 1 plans 1.1 in hour 1. Goal 2: A's unmet need is max(0, 0.6 - a1), so A needs 0.6 now; B's
is max(0, 1.5 - b1 - b2) with b2 <= 1, so B needs at least 0.5 now. 0.6 + 0.5 = 1.1 is the whole
target, so hour 1 is A 0.6, B 0.5, C 0, B plans 1 in hour 2, and no need is left unmet. Goal 3
(centre 0 in hour 2) then gives C nothing in hour 2: its optimum is |1 - 0| = 1.
EDF (departures 1.5, 3, 3) and LLF (laxities 0.2, 0.5, 2) both fill A to its most 1 and B with the
0.1 left: (1, 0.1, 0); B's unmet need is then 1.5 - 0.1 - 1 = 0.4. The smoothed rule plans
clip(2 (L + 0.8), 0, 1) for A, clip(L + 0.5, 0, 1) for B and clip(L - 1, 0, 1) for C; with A and B
inside their limits 3 L + 2.1 = 1.1 gives L = -1/3: (14/15, 1/6, 0).

Case 2: goal 1 decides over goal 2. Hour 2 of 2, target 1: A and B (3, 2, 10, 2). Each car needs
2 and can take 2, but the hour plans 1 in total (unmet need 3, goal 2's optimum). A target of -1
is read as 0 (A7 of `asa_programme`): it plans 0, and goal 1's optimum is 0. A target of 9 plans
each car's most, 2, and goal 1's optimum is 9 - 4 = 5.

Case 3: goal 3 decides with goals 1 and 2 tied. Hour 1 of 2, target 1: A and B (3, 1, 10, 0).
No car needs anything, so goal 2 is 0 for every plan; hour 2 can take 0 to 2. With centre 0.5 in
hour 2 the plan's hour-2 total is 0.5, where goal 4 alone would plan 2; with centre 3 it is 2.
With centre -1, read as 0 (A7), it is 0 and goal 3's optimum is 0 (read as -1 it would be 1).

Case 4: goal 4 decides with goals 1 to 3 tied. Hour 1 of 3, target 0.5, centre (-, 1, 1):

- A (4, 1, 1.5, 0): up to 1 in each of hours 1, 2, 3 and at most 1.5 in total.
- B (2, 1, 10, 0): up to 1 in hour 1, nothing later.

Goal 1: a1 + b1 = 0.5. Goal 3: |a2 - 1| + |a3 - 1| with a2 + a3 <= 1.5 - a1 is smallest, 0.5, at
a1 = 0 and a2 + a3 = 1.5, so b1 = 0.5. Goal 4 weighs hour 2 by 2 and hour 3 by 1 (n - s + 1), so
a2 = 1 and a3 = 0.5; (a2, a3) = (0.5, 1) ties goals 1 to 3.
"""

import numpy as np
import pytest

from fleetreserve.asa_programme import STAGES, plan_hour
from fleetreserve.controller import Belief, HourView
from fleetreserve.priority_rules import (
    earliest_deadline_first,
    least_laxity_first,
    smoothed_least_laxity,
)

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
    assert found.outcomes == ("optimal",) * len(STAGES)
    return found


CASE_1 = _view(1, 2, 1.1, Belief([1.5, 3, 3], [2, 1, 1], [10, 10, 10], [0.6, 1.5, 0]))


def test_case_1_goal_2_serves_the_leaving_car_now():
    found = _plan(CASE_1, [0.0, 0.0])
    np.testing.assert_allclose(found.powers, [[0.6, 0.0], [0.5, 1.0], [0.0, 0.0]], atol=TOLERANCE)
    np.testing.assert_allclose(found.optima[:3], [0.0, 0.0, 1.0], atol=TOLERANCE)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (earliest_deadline_first, [1.0, 0.1, 0.0]),
        (least_laxity_first, [1.0, 0.1, 0.0]),
        (smoothed_least_laxity, [14 / 15, 1 / 6, 0.0]),
    ],
)
def test_case_1_the_priority_rules_plan_otherwise(rule, expected):
    np.testing.assert_allclose(rule(CASE_1), expected, atol=1e-12)
    asa = _plan(CASE_1, [0.0, 0.0]).powers[:, 0]
    assert np.abs(asa - np.array(expected)).max() > 0.3


CASE_2 = Belief([3, 3], [2, 2], [10, 10], [2, 2])


def test_case_2_goal_1_keeps_the_hours_total_at_the_target():
    found = _plan(_view(2, 2, 1.0, CASE_2), [0.0, 0.0])
    assert found.powers.shape == (2, 1)
    assert abs(found.powers.sum() - 1.0) <= TOLERANCE
    assert np.all(found.powers >= 0) and np.all(found.powers <= 2)
    np.testing.assert_allclose(found.optima[:3], [0.0, 3.0, 0.0], atol=TOLERANCE)


@pytest.mark.parametrize(("target", "expected"), [(-1.0, [0.0, 0.0]), (9.0, [2.0, 2.0])])
def test_case_2_a_target_below_0_or_above_the_most(target, expected):
    found = _plan(_view(2, 2, target, CASE_2), [0.0, 0.0])
    np.testing.assert_allclose(found.powers[:, 0], expected, atol=TOLERANCE)
    assert abs(found.optima[0] - abs(max(target, 0.0) - sum(expected))) <= TOLERANCE


CASE_3 = Belief([3, 3], [1, 1], [10, 10], [0, 0])


@pytest.mark.parametrize(("centre", "later"), [(0.5, 0.5), (3.0, 2.0), (-1.0, 0.0)])
def test_case_3_goal_3_sets_the_later_hours_total(centre, later):
    found = _plan(_view(1, 2, 1.0, CASE_3), [0.0, centre])
    assert abs(found.powers[:, 0].sum() - 1.0) <= TOLERANCE
    assert abs(found.powers[:, 1].sum() - later) <= TOLERANCE
    assert abs(found.optima[2] - abs(max(centre, 0.0) - later)) <= TOLERANCE


CASE_4 = _view(1, 3, 0.5, Belief([4, 2], [1, 1], [1.5, 10], [0, 0]))


def test_case_4_goal_4_plans_earlier():
    found = _plan(CASE_4, [0.0, 1.0, 1.0])
    np.testing.assert_allclose(found.powers, [[0.0, 1.0, 0.5], [0.5, 0.0, 0.0]], atol=TOLERANCE)
    np.testing.assert_allclose(
        found.optima, [0.0, 0.0, 0.5, -(3 * 0.5 + 2 * 1 + 1 * 0.5)], atol=TOLERANCE
    )


def test_the_same_programme_gives_the_same_plan():
    for view, centre in ((CASE_1, [0.0, 0.0]), (CASE_4, [0.0, 1.0, 1.0])):
        first = _plan(view, centre)
        second = _plan(view, centre)
        assert np.array_equal(first.powers, second.powers)
        assert first.optima == second.optima
