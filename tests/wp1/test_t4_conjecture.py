"""Plan section 4, T4 conjecture, on a stated seeded sample: V_aff = V_caus for boxes of a fixed
shape on lossless fleets.

V_aff is the largest scale of the box met on every corner path by a causal (lower-triangular)
affine rule; V_caus is the largest scale met by any causal rule, which is the corner-tree value
(plan T3). Both come from `toy_values.py`, one linear programme each. Affine is never above the
tree; the conjecture says it is never below it.

The sample is built from one seeded generator (`SEED`). It has `N_PER_GROUP` fleets in each of 32
groups, so `N_FLEETS` in all; every group is one combination of:

- shape: the block product (all widths equal, centre free, declared before the first request) or a
  fixed shape (a centre drawn from the fleet set and widths drawn from [0.1, 1.0], both fixed);
- direction: one-way cars (power in [0, rate]) or two-way cars (power in [-1 or -2, rate]);
- service hours: 2, 3, 4 or 5;
- departures: at least one car leaves inside the window (it misses the last service hour), or no
  car does.

Each fleet has 1 to 4 cars, all present from hour 1; the hours after the service hours are recovery
hours. Every fleet breaks at least one hypothesis of the T4 proposition except one-way fleets with
no departure inside the window whose recovery can supply every car's energy; those are kept too.
A drawn fleet is used only if its tree scale is positive (a fleet that can offer nothing says
nothing about the conjecture); `MAX_ATTEMPTS` bounds the draws for one fleet.

A gap is `tree - affine > TOLERANCE * max(1, tree)`. On a gap the test fails and prints the fleet
in full, so the case can be reproduced. The hindsight scale is computed too: the order affine <=
tree <= hindsight must hold, and the sample must hold at least `MIN_CAUSAL_GAPS` fleets where the
tree is strictly below hindsight (where real-time decisions cost something, and an affine rule
could fall short of the best causal rule).
"""

import functools
import random
from itertools import product

from test_free_box_value import _breaks_hypotheses, _point_in_fleet_set
from toy_values import (
    affine_scale,
    block_affine_scale,
    block_hindsight_scale,
    hindsight_scale,
    tree_scale,
)

from fleetreserve.gpoly import Car

SEED = 0
SHAPES = ("block", "fixed")
DIRECTIONS = ("one-way", "two-way")
SERVICE_HOURS = (2, 3, 4, 5)
IN_WINDOW = (True, False)
GROUPS = list(product(SHAPES, DIRECTIONS, SERVICE_HOURS, IN_WINDOW))
N_PER_GROUP = 700
N_FLEETS = 22400
MAX_ATTEMPTS = 200  # draws for one fleet before the sample is declared unbuildable
POSITIVE_SCALE = 1e-6  # a tree scale at or below this is not used
TOLERANCE = 1e-6  # relative, against max(1, tree scale)
MIN_CAUSAL_GAPS = 500  # fleets with the tree strictly below hindsight


def _car(rng, two_way, departure):
    """One car present in hours 1 to `departure`."""
    rate = rng.choice([1.0, 2.0, 3.0])
    hours = frozenset(range(1, departure + 1))
    if two_way:
        low = -rng.choice([1.0, 2.0])
        first, second = sorted(rng.uniform(low * len(hours), rate * len(hours)) for _ in range(2))
        if rng.random() < 0.3:
            second = first  # needs exactly this energy
        return Car(hours, low, rate, first, second)
    e_high = rng.uniform(0.2, 1.3) * rate * len(hours)
    e_low = e_high if rng.random() < 0.5 else rng.uniform(0.0, e_high)
    return Car(hours, 0.0, rate, min(e_low, rate * len(hours)), e_high)


def _fleet(rng, n_hours, two_way, in_window):
    """1 to 4 cars; with `in_window` the first car leaves before the last service hour, otherwise
    every car stays to the last service hour or later."""
    cars = []
    for k in range(rng.randint(1, 4)):
        if in_window and k == 0:
            departure = rng.randint(1, n_hours - 1)
        elif in_window:
            departure = rng.randint(1, n_hours + 3)
        else:
            departure = rng.randint(n_hours, n_hours + 3)
        cars.append(_car(rng, two_way, departure))
    return cars


def _case(rng, shape, direction, n_hours, in_window):
    """Draw fleets until one has a positive tree scale; return it with its three scales."""
    service = list(range(1, n_hours + 1))
    for _ in range(MAX_ATTEMPTS):
        cars = _fleet(rng, n_hours, direction == "two-way", in_window)
        if shape == "block":
            centre = widths = None
            tree = tree_scale(cars, service)
        else:
            centre = _point_in_fleet_set(rng, cars, service)
            widths = {t: rng.uniform(0.1, 1.0) for t in service}
            tree = tree_scale(cars, service, centre, widths)
        if tree <= POSITIVE_SCALE:
            continue
        if shape == "block":
            affine = block_affine_scale(cars, service)
            hindsight = block_hindsight_scale(cars, service)
        else:
            affine = affine_scale(cars, service, centre, widths)
            hindsight = hindsight_scale(cars, service, centre, widths)
        return {
            "group": (shape, direction, n_hours, in_window),
            "cars": cars,
            "service": service,
            "centre": centre,
            "widths": widths,
            "affine": affine,
            "tree": tree,
            "hindsight": hindsight,
        }
    raise AssertionError(
        f"no fleet with a positive tree scale in {MAX_ATTEMPTS} draws for group "
        f"{(shape, direction, n_hours, in_window)}"
    )


@functools.cache
def _sample():
    """The sample, drawn group by group in turn from one seeded generator."""
    rng = random.Random(SEED)
    return tuple(_case(rng, *group) for _ in range(N_PER_GROUP) for group in GROUPS)


def test_the_sample_has_the_stated_count_and_composition():
    cases = _sample()
    assert len(GROUPS) == 32
    assert N_PER_GROUP * len(GROUPS) == N_FLEETS
    assert len(cases) == N_FLEETS
    for group in GROUPS:
        assert sum(case["group"] == group for case in cases) == N_PER_GROUP
    for case in cases:
        shape, direction, n_hours, in_window = case["group"]
        cars, service = case["cars"], case["service"]
        assert service == list(range(1, n_hours + 1))
        assert (case["centre"] is None) == (shape == "block")
        assert all((car.low < 0) == (direction == "two-way") for car in cars)
        assert any(max(car.hours) < n_hours for car in cars) == in_window
        if in_window:
            assert _breaks_hypotheses(cars, service)
        assert case["tree"] > POSITIVE_SCALE


def test_affine_equals_tree_on_every_sampled_fleet():
    cases = _sample()
    gaps = []
    above = []
    causal_gaps = 0
    worst = 0.0
    for case in cases:
        slack = TOLERANCE * max(1.0, case["tree"])
        gap = (case["tree"] - case["affine"]) / max(1.0, case["tree"])
        worst = max(worst, gap)
        if case["tree"] - case["affine"] > slack:
            gaps.append(case)
        if case["affine"] > case["tree"] + slack or case["tree"] > case["hindsight"] + slack:
            above.append(case)
        causal_gaps += case["hindsight"] - case["tree"] > slack
    print(
        f"T4 conjecture: affine equals tree on {len(cases) - len(gaps)} of {len(cases)} fleets; "
        f"largest relative shortfall of affine {worst:.2e}; tree strictly below hindsight on "
        f"{causal_gaps}"
    )
    assert not gaps, f"affine below tree on {len(gaps)} fleets; all of them: {gaps}"
    assert not above, f"order affine <= tree <= hindsight broken on {len(above)} fleets: {above}"
    assert len(cases) == N_FLEETS
    assert causal_gaps >= MIN_CAUSAL_GAPS
