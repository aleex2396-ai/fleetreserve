"""The three priority rules by hand (plan section 10: "LLF, EDF, smoothed LLF").

Case 1: one hour, three cars, three different allocations. Hour 1 of 3; nothing drawn yet; every
car plugged in; target 3 kWh. Beliefs (departure, rate, cap, need):

- A: (3, 2, 10, 2). Most this hour: min(2 x 1, 10 - 0) = 2. Laxity: (3 - 1) - 2/2 = 1.
- B: (4, 1, 10, 3). Most: 1. Laxity: (4 - 1) - 3/1 = 0.
- C: (5, 2, 10, 7). Most: 2. Laxity: (5 - 1) - 7/2 = 0.5.

The believed total most is 5, so each rule plans 3 in total.

- EDF (earliest believed departure first: A, B, C): A 2, B 1, C 0.
- LLF (least laxity first: B, C, A): B 1, C 2, A 0.
- Smoothed LLF: power_i = clip(r_i (L - laxity_i + 1), 0, most_i): A clip(2 L), B clip(L + 1),
  C clip(2 L + 1). For L in [0, 0.5]: B is at its most 1, A = 2 L, C = 2 L + 1, total 4 L + 2 = 3,
  so L = 0.25: A 0.5, B 1, C 1.5. A and C end the hour with laxity
  laxity - 1 + power / rate: A 1 - 1 + 0.5/2 = 0.25, C 0.5 - 1 + 1.5/2 = 0.25, both L.

Case 2: the same cars with target 6, above the believed total 5: every rule gives each car its
most, (2, 1, 2).

Case 3: the same cars with C not counted as plugged in, target 3: C gets 0 under every rule; the
believed total is A 2 + B 1 = 3, so every rule gives (2, 1, 0).

Case 4: EDF ties by car order: two cars with the same believed departure 2, rate 1, most 1 each,
target 1: the first car gets 1, the second 0. LLF ties the same way (equal laxity).

Case 5: the energy cap binds. One car, belief (4, 3, 5, 5), 4 kWh drawn: most min(3, 5 - 4) = 1;
target 2 gives 1. A believed departure inside the hour counts by its share: belief (1.5, 2, 9, 0)
in hour 1 has most 2 x 0.5 = 1.
"""

import numpy as np
import pytest

from fleetreserve.controller import Belief, HourView
from fleetreserve.priority_rules import (
    earliest_deadline_first,
    laxity,
    least_laxity_first,
    most,
    smoothed_least_laxity,
)

RULES = (earliest_deadline_first, least_laxity_first, smoothed_least_laxity)
TOLERANCE = 1e-12


def _view(target, *, plugged=(True, True, True), hour=1, drawn=(0.0, 0.0, 0.0), belief=None):
    belief = belief or Belief([3.0, 4.0, 5.0], [2.0, 1.0, 2.0], [10.0] * 3, [2.0, 3.0, 7.0])
    return HourView(
        hour=hour,
        n_service=3,
        target=target,
        zeta_known=np.zeros(hour),
        requests_known=np.full(hour, target),
        drawn=np.array(drawn, dtype=float),
        plugged=np.array(plugged),
        belief=belief,
    )


def test_case_1_most_and_laxity_by_hand():
    view = _view(3.0)
    np.testing.assert_allclose(most(view), [2.0, 1.0, 2.0], atol=TOLERANCE)
    np.testing.assert_allclose(laxity(view), [1.0, 0.0, 0.5], atol=TOLERANCE)


@pytest.mark.parametrize(
    "rule, expected",
    [
        (earliest_deadline_first, [2.0, 1.0, 0.0]),
        (least_laxity_first, [0.0, 1.0, 2.0]),
        (smoothed_least_laxity, [0.5, 1.0, 1.5]),
    ],
)
def test_case_1_three_rules_three_allocations(rule, expected):
    np.testing.assert_allclose(rule(_view(3.0)), expected, atol=TOLERANCE)


def test_case_1_smoothed_cars_inside_their_limits_end_with_equal_laxity():
    view = _view(3.0)
    power = smoothed_least_laxity(view)
    after = laxity(view) - 1 + power / view.belief.rate
    np.testing.assert_allclose(after[[0, 2]], [0.25, 0.25], atol=TOLERANCE)


@pytest.mark.parametrize("rule", RULES)
def test_case_2_a_target_above_the_believed_total_gives_every_car_its_most(rule):
    np.testing.assert_allclose(rule(_view(6.0)), [2.0, 1.0, 2.0], atol=TOLERANCE)


@pytest.mark.parametrize("rule", RULES)
def test_case_3_a_car_not_counted_as_plugged_in_gets_nothing(rule):
    got = rule(_view(3.0, plugged=(True, True, False)))
    assert got[2] == 0.0
    np.testing.assert_allclose(got, [2.0, 1.0, 0.0], atol=TOLERANCE)


@pytest.mark.parametrize("rule", (earliest_deadline_first, least_laxity_first))
def test_case_4_ties_go_by_car_order(rule):
    belief = Belief([2.0, 2.0], [1.0, 1.0], [5.0, 5.0], [1.0, 1.0])
    view = _view(1.0, plugged=(True, True), drawn=(0.0, 0.0), belief=belief)
    np.testing.assert_array_equal(rule(view), [1.0, 0.0])


def test_case_5_the_cap_and_a_fractional_departure_by_hand():
    capped = _view(2.0, plugged=(True,), drawn=(4.0,), belief=Belief([4.0], [3.0], [5.0], [5.0]))
    np.testing.assert_allclose(most(capped), [1.0], atol=TOLERANCE)
    for rule in RULES:
        np.testing.assert_allclose(rule(capped), [1.0], atol=TOLERANCE)
    part = _view(2.0, plugged=(True,), drawn=(0.0,), belief=Belief([1.5], [2.0], [9.0], [0.0]))
    np.testing.assert_allclose(most(part), [1.0], atol=TOLERANCE)
