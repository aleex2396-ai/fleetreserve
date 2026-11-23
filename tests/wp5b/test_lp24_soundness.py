"""Soundness of LP (24) and LP (27) (`lp24`): the per-car maps keep every car in its set.

On `N_DAYS` seeded made-up days (`lp24_cases`, 2 to 4 service hours, 1 to 4 cars), each on its
true sets one-way and two-way, with the day-average base set: `N_TRIVIAL` cases have a base set
holding only the zero profile (every car without room) and are left out; both programmes solve
on the other `N_SOLVED`. For each solved programme, `N_POINTS` random profiles `u0` of the base
set and every vertex of the base set are mapped through the per-car maps: every car's power
`gamma_i + Gamma_i u0` satisfies every one of its rows within `TOLERANCE`, and the cars' powers
sum to `p + alpha u0` (LP (24)) or `p + P u0` (LP (27)). At least `MIN_POSITIVE` cases have
`alpha > 0`, so the check is not met by empty maps alone.

The same with the template base set: per number of service hours, the template pooled over every
case with that many service hours (`N_TEMPLATE` cases in all, both programmes), random points
only.
"""

import functools
import random

import numpy as np
from lp24_cases import base_points, base_vertices, cases

from fleetreserve.base_sets import template
from fleetreserve.lp24 import inner_set

SEED = 7
N_DAYS = 40
N_CASES = 2 * N_DAYS
N_TRIVIAL = 4
N_SOLVED = N_CASES - N_TRIVIAL
N_POINTS = 25
MIN_POSITIVE = 40
TOLERANCE = 1e-6
N_TEMPLATE = N_CASES


@functools.cache
def _solved():
    found = cases(SEED, N_DAYS)
    assert len(found) == N_CASES
    solved = []
    for _day, _budget, rows, base in found:
        if base is None:
            continue
        results = {p: inner_set(rows, base, p, keep_maps=True) for p in ("24", "27")}
        assert all(r.outcome == "optimal" for r in results.values()), results
        solved.append((rows, base, results))
    return solved


def _check(rows, result, points):
    broken = 0
    for u0 in points:
        powers = result.car_powers(u0)
        assert powers.shape == (rows.n_cars, rows.n_service + 1)
        broken += int(np.sum(rows.slack(powers) < -TOLERANCE))
        np.testing.assert_allclose(powers.sum(axis=0), result.profile(u0), atol=TOLERANCE)
        if result.programme == "24":
            np.testing.assert_allclose(
                powers.sum(axis=0), result.p + result.alpha * u0, atol=TOLERANCE
            )
    return broken


def test_counts():
    solved = _solved()
    assert len(solved) == N_SOLVED
    positive = sum(r["24"].alpha > TOLERANCE for *_, r in solved)
    print(f"{len(solved)} cases solved, alpha > 0 on {positive}")
    assert positive >= MIN_POSITIVE


def test_random_points_are_shared_out_within_every_car_set():
    rng = random.Random(SEED)
    broken = 0
    for rows, base, results in _solved():
        points = base_points(rng, base, N_POINTS)
        assert np.all(base.contains(points))
        for result in results.values():
            broken += _check(rows, result, points)
    assert broken == 0


def test_vertices_are_shared_out_within_every_car_set():
    broken = 0
    n_vertices = 0
    for rows, base, results in _solved():
        vertices = base_vertices(base)
        assert np.all(base.contains(vertices))
        n_vertices += len(vertices)
        for result in results.values():
            broken += _check(rows, result, vertices)
    print(f"{n_vertices} base-set vertices checked")
    assert broken == 0


def test_template_base_set_is_shared_out_within_every_car_set():
    found = cases(SEED, N_DAYS)
    by_hours = {}
    for _day, _budget, rows, _base in found:
        by_hours.setdefault(rows.n_service, []).append(rows)
    templates = {n: template(fleets) for n, fleets in by_hours.items()}
    rng = random.Random(SEED + 1)
    broken = 0
    solved = 0
    for _day, _budget, rows, _base in found:
        base = templates[rows.n_service]
        points = base_points(rng, base, N_POINTS)
        for programme in ("24", "27"):
            result = inner_set(rows, base, programme, keep_maps=True)
            assert result.outcome == "optimal"
            broken += _check(rows, result, points)
        solved += 1
    assert solved == N_TEMPLATE
    assert broken == 0
