"""Plan section 4: the unit tests of T3 and T4 through the sparse certificate and corner tree.

Both examples have no hour 0 and are written as `gpoly.Car` rows (`carsets.gpoly_rows`).

- T3: one car in hours 1 and 2, rate 1, energy in [1, 1.5]; box (0.6, 0.6) +- lam (1, 1). Plan:
  hindsight lam = 0.10. The tree and every affine structure reach it (one car, so every causal
  rule is the car's own plan).
- T4 gap without in-window departure: car D in hours 1-2 (rate 2, needs exactly 2), car C in hours
  1-3 (rate 1, room 10), S = {1, 2}, centre (1.1, 1.9), widths (0.1, 0.9). Plan: hindsight
  lam = 1.0, tree = affine = 0.556. The diagonal rule gives 0.556 too, and the full structure (the
  general split, which knows future requests) reaches hindsight, 1.0.
"""

import pytest

from fleetreserve import products
from fleetreserve.carsets import gpoly_fleet_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.gpoly import Car
from fleetreserve.tree import corner_tree

T3_ROWS = gpoly_fleet_rows([Car(frozenset({1, 2}), 0.0, 1.0, 1.0, 1.5)], 2)
T3_BOX = products.fixed_shape((0.6, 0.6), (1.0, 1.0))
GAP_ROWS = gpoly_fleet_rows(
    [Car(frozenset({1, 2}), 0.0, 2.0, 2.0, 2.0), Car(frozenset({1, 2, 3}), 0.0, 1.0, 0.0, 10.0)], 2
)
GAP_BOX = products.fixed_shape((1.1, 1.9), (0.1, 0.9))


def _scale(rows, box, structure):
    if structure == "tree":
        result = corner_tree(rows, box)
    else:
        result = certify(rows, box, structure, shape_rule=ANY_OPTIMAL)
    print(f"{structure}: {result.outcome} lam = {result.scale!r}")
    assert result.outcome == "optimal"
    return result.scale


@pytest.mark.parametrize("structure", ["tree", "diagonal", "causal", "full"])
def test_t3_example_is_0_10(structure):
    assert _scale(T3_ROWS, T3_BOX, structure) == pytest.approx(0.10, abs=1e-7)


@pytest.mark.parametrize(
    ("structure", "expected"),
    [("tree", 5 / 9), ("causal", 5 / 9), ("diagonal", 5 / 9), ("full", 1.0)],
)
def test_t4_gap_example(structure, expected):
    scale = _scale(GAP_ROWS, GAP_BOX, structure)
    assert scale == pytest.approx(expected, abs=1e-7)
    assert round(scale, 3) == (0.556 if expected < 1 else 1.0)
