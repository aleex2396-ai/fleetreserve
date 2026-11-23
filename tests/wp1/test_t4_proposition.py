"""Plan section 4, T4 proposition, tested on the block product: one-way, known cars, no in-window
departure, and recovery capacity r_i * (recovery hours) >= (1 - rho) * e_i for every car imply
that the diagonal rule reaches the hindsight value.

The proposition says something only for a fixed shape. With the box free in every hour (the hourly
product) the diagonal and hindsight values are both the fleet's total flexibility on any lossless
fleet (`test_free_box_value.py`). The block product fixes the shape: the box is `m_t +- lam` in
every service hour, the centre `m` is free but declared before the first request, and the value is
the largest `lam`.

300 seeded toy fleets meet the hypotheses exactly: lossless one-way cars, rho = 0 (so each car must
end with exactly its room e_i), every car present in all of the 2 to 5 service hours (no departure
inside the window), and enough recovery hours that r_i * k_i >= e_i. On each, the block `lam` of the
diagonal rule (one programme) must equal the hindsight `lam` (from the fleet's exact set functions;
also from one power plan per corner path, and the two routes must agree). The causal-affine rule and
the corner tree are computed too, and the order diagonal <= affine <= tree <= hindsight must hold.

A control uses 300 fleets that break the hypotheses (a car leaves inside the window, or a car's
recovery cannot supply its energy). There the diagonal `lam` is never above hindsight, and falls
strictly below it on at least one fleet.
"""

import math
import random

from test_free_box_value import _breaking_fleet, _breaks_hypotheses
from toy_values import (
    block_affine_scale,
    block_diagonal_scale,
    block_hindsight_scale,
    block_hindsight_scale_by_paths,
    tree_scale,
)

from fleetreserve.gpoly import Car

N_INSTANCES = 300
TOLERANCE = 1e-6  # relative, against max(1, hindsight lam)
MIN_POSITIVE_SHARE = 0.9  # fleets with a positive block lam
MIN_CAP_SHARE = 0.25  # fleets where the cars' energy rooms (not only power) limit the box


def _instance(rng):
    """A fleet meeting the proposition's hypotheses; returns the cars and the service hours."""
    n_hours = rng.randint(2, 5)
    service = list(range(1, n_hours + 1))
    cars = []
    for _ in range(rng.randint(1, 4)):
        rate = rng.choice([1.0, 2.0, 3.0])
        room = rng.uniform(0.2, 1.3) * rate * n_hours
        recovery = math.ceil(room / rate) + rng.randint(0, 1)
        assert rate * recovery >= room  # the proposition's recovery condition, with rho = 0
        hours = frozenset(range(1, n_hours + recovery + 1))  # all service hours, then recovery
        cars.append(Car(hours, 0.0, rate, room, room))  # protected energy (1 - 0) * room
    return cars, service


def _scales(cars, service):
    """Block lam of the four rules, and of hindsight by the second route."""
    return {
        "hindsight": block_hindsight_scale(cars, service),
        "paths": block_hindsight_scale_by_paths(cars, service),
        "tree": tree_scale(cars, service),
        "affine": block_affine_scale(cars, service),
        "diagonal": block_diagonal_scale(cars, service),
    }


def _order_holds(v):
    slack = TOLERANCE * max(1.0, v["hindsight"])
    return (
        v["diagonal"] <= v["affine"] + slack
        and v["affine"] <= v["tree"] + slack
        and v["tree"] <= v["hindsight"] + slack
        and abs(v["paths"] - v["hindsight"]) <= slack
    )


def test_block_diagonal_equals_hindsight_on_300_of_300():
    rng = random.Random(0)
    agree = positive = capped = 0
    worst = 0.0
    failures = []
    for _ in range(N_INSTANCES):
        cars, service = _instance(rng)
        assert not _breaks_hypotheses(cars, service)
        v = _scales(cars, service)
        gap = abs(v["diagonal"] - v["hindsight"]) / max(1.0, v["hindsight"])
        worst = max(worst, gap)
        if gap <= TOLERANCE and _order_holds(v):
            agree += 1
        else:
            failures.append((cars, service, v))
        positive += v["hindsight"] > 1e-9
        capped += v["hindsight"] < sum(car.high for car in cars) / 2 - 1e-9
    print(
        f"block product, hypotheses met: diagonal equals hindsight on {agree} of {N_INSTANCES}; "
        f"largest relative difference {worst:.2e}; positive lam {positive}; "
        f"energy rooms limit the box {capped}"
    )
    assert not failures, f"agree {agree} of {N_INSTANCES}; failures in full: {failures}"
    assert agree == N_INSTANCES
    assert positive >= MIN_POSITIVE_SHARE * N_INSTANCES
    assert capped >= MIN_CAP_SHARE * N_INSTANCES


def test_control_block_diagonal_falls_short_when_a_hypothesis_is_broken():
    rng = random.Random(0)
    gaps = tree_short = affine_short = positive = 0
    largest = 0.0
    above = []
    for _ in range(N_INSTANCES):
        cars, service = _breaking_fleet(rng)
        v = _scales(cars, service)
        slack = TOLERANCE * max(1.0, v["hindsight"])
        if not _order_holds(v):
            above.append((cars, service, v))
        gap = v["hindsight"] - v["diagonal"]
        largest = max(largest, gap)
        gaps += gap > slack
        tree_short += v["hindsight"] - v["tree"] > slack
        affine_short += v["tree"] - v["affine"] > slack
        positive += v["hindsight"] > 1e-9
    print(
        f"block product, hypotheses broken: diagonal lam strictly below hindsight on {gaps} of "
        f"{N_INSTANCES} fleets; largest gap {largest:.4f}; tree below hindsight on {tree_short}; "
        f"affine below tree on {affine_short}; positive hindsight lam {positive}"
    )
    assert not above, f"order violated on {len(above)} fleets: {above}"
    assert gaps >= 1
