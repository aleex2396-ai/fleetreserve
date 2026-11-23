"""The sparse certificate and corner tree equal the dense toy programmes of `tests/wp1/toy_values`.

Same set in both formulations. The toy fleets (`test_t4_conjecture._fleet`) are `gpoly.Car` present
from hour 1; every hour after the service hours is a recovery hour, free within the car's power
limits. `carsets.gpoly_rows` writes the same car over hours 0 to n: power 0 in hour 0 (the car is
not present there), its power limits in the service hours, and its total energy limits with the
recovery hours eliminated as `[e_low - high R, e_high - low R]` on hours 0 to n, which is the
toy's own elimination (`toy_values` header); the toy tree decides recovery at the leaves within
the same interval. So both describe one set, and the values must agree.

Sample: `N_FLEETS` seeded toy fleets, cycling through one-way and two-way cars, a departure inside
the window or not, and 2 to 4 service hours. Per fleet, against the toy reference:

- fixed shape (centre a random point of the fleet set, widths in [0.1, 1.0]): diagonal against
  `diagonal_scale`, causal against `affine_scale`, tree against `tree_scale`;
- block: diagonal against `block_diagonal_scale`, causal against `block_affine_scale`, tree
  against `tree_scale` without a shape;
- hourly: diagonal against `diagonal_value`, causal against `affine_value`, tree against
  `hindsight_value` (plan T4 on toy fleets: the hourly tree equals hindsight).
"""

import random

import pytest
from test_free_box_value import _point_in_fleet_set
from test_t4_conjecture import _fleet
from toy_values import (
    affine_scale,
    affine_value,
    block_affine_scale,
    block_diagonal_scale,
    diagonal_scale,
    diagonal_value,
    hindsight_value,
    tree_scale,
)

from fleetreserve import products
from fleetreserve.carsets import gpoly_fleet_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.gpoly import Car
from fleetreserve.tree import corner_tree

SEED = 31
N_FLEETS = 160
N_COMPARISONS = 9 * N_FLEETS
TOLERANCE = 1e-6  # relative to max(1, value)


def _close(a, b):
    return abs(a - b) <= TOLERANCE * max(1.0, abs(b))


def _ours(rows, product, structure):
    if structure == "tree":
        result = corner_tree(rows, product)
    else:
        result = certify(rows, product, structure, shape_rule=ANY_OPTIMAL)
    assert result.outcome == "optimal"
    return result.value if product.kind == "hourly" else result.scale


def _cases():
    rng = random.Random(SEED)
    for k in range(N_FLEETS):
        n = 2 + k % 3
        cars = _fleet(rng, n, two_way=k % 2 == 1, in_window=(k // 2) % 2 == 0)
        service = list(range(1, n + 1))
        centre = _point_in_fleet_set(rng, cars, service)
        widths = {t: rng.uniform(0.1, 1.0) for t in service}
        yield cars, service, centre, widths


def test_sparse_programmes_equal_the_toy_references():
    compared = 0
    mismatches = []
    for cars, service, centre, widths in _cases():
        rows = gpoly_fleet_rows(cars, len(service))
        fixed = products.fixed_shape([centre[t] for t in service], [widths[t] for t in service])
        shape = (cars, service, centre, widths)
        pairs = [
            (fixed, "diagonal", diagonal_scale(*shape)),
            (fixed, "causal", affine_scale(*shape)),
            (fixed, "tree", tree_scale(*shape)),
            (products.block(), "diagonal", block_diagonal_scale(cars, service)),
            (products.block(), "causal", block_affine_scale(cars, service)),
            (products.block(), "tree", tree_scale(cars, service)),
            (products.hourly(), "diagonal", diagonal_value(cars, service)),
            (products.hourly(), "causal", affine_value(cars, service)),
            (products.hourly(), "tree", hindsight_value(cars, service)),
        ]
        for product, structure, reference in pairs:
            ours = _ours(rows, product, structure)
            compared += 1
            if not _close(ours, reference):
                mismatches.append((product.kind, structure, ours, reference, cars))
    print(f"sparse equals toy on {compared - len(mismatches)} of {compared} comparisons")
    assert not mismatches, mismatches
    assert compared == N_COMPARISONS


def test_the_sample_has_positive_values_and_both_directions():
    positive = 0
    two_way = 0
    for cars, service, _, _ in _cases():
        two_way += any(car.low < 0 for car in cars)
        positive += tree_scale(cars, service) > 1e-6
    assert two_way == N_FLEETS // 2
    assert positive >= N_FLEETS // 2


@pytest.mark.parametrize("structure", ["diagonal", "causal", "full"])
def test_a_fleet_that_can_offer_nothing_has_value_zero(structure):
    # One car that must draw exactly its rate in both service hours: no width is possible.
    rows = gpoly_fleet_rows([Car(frozenset({1, 2}), 0.0, 1.0, 2.0, 2.0)], 2)
    result = certify(rows, products.hourly(), structure, shape_rule=ANY_OPTIMAL)
    assert result.outcome == "optimal"
    assert result.value == pytest.approx(0.0, abs=1e-9)
    assert corner_tree(rows, products.hourly()).value == pytest.approx(0.0, abs=1e-9)
