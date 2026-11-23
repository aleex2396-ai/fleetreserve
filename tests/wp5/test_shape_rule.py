"""The certificate's shape rule: no default, and the second stage over the optimal face.

1. `certify` called without `shape_rule` is refused (no default); an unknown rule too.
2. Hand case: one car in hours 1 and 2, rate 1, total energy in [0, 1]. The hourly value is 1 kWh
   for many shapes; `LargestSmallestWidth` picks the widths (0.25, 0.25).
3. On `N_DAYS` seeded made-up days (hourly product, causal structure): the second stage keeps the
   value within its tolerance of the first, its smallest width is never below that of the solver's
   own vertex, and it is strictly larger on at least `MIN_CHANGED` days (the rule does something).
"""

import random

import pytest
from made_up_days import made_up_sample

from fleetreserve import products
from fleetreserve.carsets import gpoly_fleet_rows, true_rows
from fleetreserve.certificate import ANY_OPTIMAL, LargestSmallestWidth, certify
from fleetreserve.gpoly import Car

SEED = 81
N_DAYS = 40
MIN_CHANGED = 5
RULE_TOLERANCE = 1e-6
TOLERANCE = 1e-7

HAND_ROWS = gpoly_fleet_rows([Car(frozenset({1, 2}), 0.0, 1.0, 0.0, 1.0)], 2)


def test_the_shape_rule_has_no_default():
    with pytest.raises(TypeError):
        certify(HAND_ROWS, products.hourly(), "causal")
    with pytest.raises(ValueError):
        certify(HAND_ROWS, products.hourly(), "causal", shape_rule=None)


def test_hand_case_largest_smallest_width():
    for structure in ("diagonal", "causal", "full"):
        result = certify(
            HAND_ROWS, products.hourly(), structure, shape_rule=LargestSmallestWidth(RULE_TOLERANCE)
        )
        assert result.outcome == "optimal"
        assert result.first_value == pytest.approx(1.0, abs=TOLERANCE)
        assert result.value == pytest.approx(1.0, abs=1e-5)
        assert result.policy.w == pytest.approx([0.25, 0.25], abs=1e-5)
        assert result.shape_rule == "largest smallest width"


def test_second_stage_on_made_up_days():
    days, _ = made_up_sample(random.Random(SEED), N_DAYS)
    changed = 0
    for day in days:
        rows = true_rows(day)
        plain = certify(rows, products.hourly(), "causal", shape_rule=ANY_OPTIMAL)
        ruled = certify(
            rows, products.hourly(), "causal", shape_rule=LargestSmallestWidth(RULE_TOLERANCE)
        )
        assert plain.outcome == ruled.outcome == "optimal"
        assert ruled.first_value == pytest.approx(plain.value, abs=TOLERANCE * max(1, plain.value))
        assert ruled.value >= plain.value - RULE_TOLERANCE * max(1.0, plain.value) - TOLERANCE
        assert ruled.value <= plain.value + TOLERANCE * max(1.0, plain.value)
        assert ruled.policy.w.min() >= plain.policy.w.min() - TOLERANCE
        changed += ruled.policy.w.min() > plain.policy.w.min() + 1e-6
    print(f"the second stage raised the smallest width on {changed} of {N_DAYS} days")
    assert changed >= MIN_CHANGED
