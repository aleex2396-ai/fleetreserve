"""Plan section 4, T1(a) and T1(b), from a day's set functions, against vertex linear programmes.

Sample: seeded made-up days (`made_up_days.py`; 2 to 4 service hours), one-way and two-way (budget
cycling through `BUDGETS`). A box is a polytope and the fleet set is convex, so a box is inside
exactly when each of its vertices can be drawn, and its largest scale about a centre inside the set
is the smallest, over its vertices `c + lam v`, of the largest `lam` along `v`.

- T1(a): `N_BOXES` boxes per direction (one-way, two-way), each about a point of the set
  (`fleet_lps.random_point`) with half-widths of a random share of the hour's range. Truth for
  one-way: `vertex_truth.box_inside_by_vertices`; for two-way: `fleet_lps.attainable` at every
  vertex. Required: all agree, with between a quarter and three quarters inside.
- T1(b): `N_SHAPES` shapes per direction: a centre in the set and widths up and down drawn from
  [0, 1] kW, each zero with probability `P_ZERO` (never all zero). Truth for one-way:
  `vertex_truth.largest_scale_along` at every vertex direction; for two-way:
  `fleet_lps.largest_step`. Required: all agree to a relative 1e-6, with an upper face binding in
  at least a tenth and a lower face in at least a tenth of the shapes.
"""

import itertools
import random

import numpy as np
from fleet_lps import attainable, largest_step, random_point
from made_up_days import made_up_sample
from vertex_truth import box_inside_by_vertices, largest_scale_along

from fleetreserve.exact import box_inside, box_scale, day_functions

SEED = 21
BUDGETS = (0.0, 0.5, 2.0)
N_BOXES = 120
N_SHAPES = 80
MAX_SHARE = 0.5
P_ZERO = 0.2
TOLERANCE = 1e-6


def _cases(seed, count):
    days, _ = made_up_sample(random.Random(seed), count)
    for k, day in enumerate(days):
        yield "one-way", day, None, day.one_way_cars()
        budget = BUDGETS[k % len(BUDGETS)]
        yield "two-way", day, budget, day.two_way_cars(budget)


def _vertices(lo, hi):
    return itertools.product(*zip(lo, hi, strict=True))


def test_t1a_box_inside_agrees_with_the_vertex_programmes():
    rng = random.Random(SEED)
    counts = {"one-way": [0, 0], "two-way": [0, 0]}  # agree, inside
    wrong = []
    for label, day, budget, cars in _cases(SEED, N_BOXES):
        n = day.n_service
        functions = day_functions(day, budget)
        centre = random_point(rng, cars, n)
        spread = [functions.upper[1 << k] - functions.lower[1 << k] for k in range(n)]
        share = rng.uniform(0, MAX_SHARE)
        lo = [centre[k] - share * rng.random() * spread[k] for k in range(n)]
        hi = [centre[k] + share * rng.random() * spread[k] for k in range(n)]
        if label == "one-way":
            truth = box_inside_by_vertices(lo, hi, list(cars), day.n_hours, day.service_hours)
        else:
            truth = all(attainable(cars, n, centre, list(v)) for v in _vertices(lo, hi))
        fast = box_inside(functions, lo, hi)
        counts[label][0] += truth == fast
        counts[label][1] += truth
        if truth != fast:
            wrong.append((label, day, budget, lo, hi, truth, fast))
    for label, (agree, inside) in counts.items():
        print(f"T1(a) {label}: agree {agree} of {N_BOXES}; inside {inside}")
    assert not wrong, f"{len(wrong)} boxes disagree; first: {wrong[:3]}"
    for agree, inside in counts.values():
        assert agree == N_BOXES
        assert 0.25 * N_BOXES <= inside <= 0.75 * N_BOXES


def _widths(rng, n):
    while True:
        up = [0.0 if rng.random() < P_ZERO else rng.random() for _ in range(n)]
        down = [0.0 if rng.random() < P_ZERO else rng.random() for _ in range(n)]
        if any(up) or any(down):
            return up, down


def _face_binding(functions, centre, up, down):
    """Which kind of face gives the scale: 'upper' or 'lower' (from the closed form, one side)."""
    zero = [0.0] * len(centre)
    up_only = box_scale(functions, centre, up, zero) if any(up) else np.inf
    down_only = box_scale(functions, centre, zero, down) if any(down) else np.inf
    return "upper" if up_only <= down_only else "lower"


def test_t1b_exact_scale_agrees_with_the_vertex_programmes():
    rng = random.Random(SEED + 1)
    worst = 0.0
    binding = {"upper": 0, "lower": 0}
    compared = 0
    wrong = []
    for label, day, budget, cars in _cases(SEED + 1, N_SHAPES):
        n = day.n_service
        functions = day_functions(day, budget)
        centre = random_point(rng, cars, n)
        up, down = _widths(rng, n)
        scales = []
        for vertex in _vertices([-d for d in down], up):
            if not any(vertex):
                continue
            if label == "one-way":
                scale = largest_scale_along(
                    centre, vertex, list(cars), day.n_hours, day.service_hours
                )
            else:
                scale = largest_step(cars, n, centre, list(vertex))
            scales.append(scale)
        truth = min(scales)
        fast = box_scale(functions, centre, up, down)
        gap = abs(fast - truth) / max(1.0, truth)
        worst = max(worst, gap)
        compared += 1
        binding[_face_binding(functions, centre, up, down)] += 1
        if gap > TOLERANCE:
            wrong.append((label, day, budget, centre, up, down, truth, fast))
    print(
        f"T1(b): agree {compared - len(wrong)} of {compared}; worst relative {worst:.1e}; "
        f"binding faces {binding}"
    )
    assert not wrong, f"{len(wrong)} shapes disagree; first: {wrong[:3]}"
    assert compared == 2 * N_SHAPES
    assert min(binding.values()) >= 0.1 * compared
