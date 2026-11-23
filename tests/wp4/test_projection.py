"""Plan section 4, T1 and WP4: the projection check (Frank 2011). The fleet set seen on the service
hours S, with hour 0 and the recovery hours free, has only the faces p(A) <= u(A) <= b(A).

Sample: `N_DAYS` seeded made-up days (`made_up_days.py`; 2 to 4 service hours, 1 to 4 cars, rho
cycling through 0, 0.1 and 0.2, departures inside and after the window, cars with tight, empty
and drawn rooms). Each day is taken one-way and two-way, at a budget cycling through `BUDGETS`.
The ground truth is `fleet_lps.py`: one linear programme over every car's power in every hour,
with the recovery hours as explicit variables and no use of p or b.

1. Every set function: for every non-empty subset A of S, `exact`'s p(A) and b(A) equal the least
   and the most energy the fleet can draw in A (`N_SUBSETS` subsets in each direction).
2. Points: `N_POINTS` points per day, drawn in a box a little wider than the hourly limits; a
   point satisfies every face exactly when a programme finds powers for it. Required: both
   answers present, and at least `MIN_ONE_FACE` points violating exactly one face.
3. Directions: from a point of the set, the step to the boundary along `N_DIRECTIONS` random
   directions per day from the faces equals the step a programme finds.
4. Without discharge, the stored-energy pass of `exact` equals the one-way closed form of `gpoly`
   on `N_CLOSED_FORM` days of 1 to 8 service hours; and `gpoly.subset_functions` equals the
   scalar `fleet_p`, `fleet_b` there.
"""

import itertools
import random

import numpy as np
from fleet_lps import attainable, largest_step, lower_and_upper, random_point
from made_up_days import made_up_sample

from fleetreserve.day import StoredEnergyCar
from fleetreserve.exact import _bits, day_functions, subset_functions, subset_mask
from fleetreserve.gpoly import fleet_b, fleet_p
from fleetreserve.gpoly import subset_functions as gpoly_subset_functions

SEED = 4
N_DAYS = 60
BUDGETS = (0.0, 0.5, 2.0)  # kWh, toy sizes
N_SUBSETS = 1040  # non-empty subsets over the one-way and two-way days of the sample
N_POINTS = 8
N_DIRECTIONS = 3
MIN_ONE_FACE = 100
MARGIN = 0.3  # kWh beyond the hourly limits when drawing points
N_CLOSED_FORM = 300
TOLERANCE = 1e-7


def _sample():
    days, kinds = made_up_sample(random.Random(SEED), N_DAYS)
    cases = []
    for k, day in enumerate(days):
        cases.append(("one-way", day, day.one_way_cars(), day_functions(day)))
        budget = BUDGETS[k % len(BUDGETS)]
        cases.append(("two-way", day, day.two_way_cars(budget), day_functions(day, budget)))
    return days, kinds, cases


def test_the_sample_covers_the_cases():
    days, kinds, _ = _sample()
    assert len(days) == N_DAYS
    assert {day.rho for day in days} == {0.0, 0.1, 0.2}
    assert sum(day.has_in_window_departure for day in days) >= 20
    assert sum(not day.has_in_window_departure for day in days) >= 10
    every_kind = [kind for day_kinds in kinds for kind in day_kinds]
    assert every_kind.count("tight") >= 15 and every_kind.count("empty") >= 5
    assert sum(1 for day in days if day.rho == 0.0) == N_DAYS // 3  # cars need exactly their room


def test_every_set_function_equals_the_fleet_programme():
    _, _, cases = _sample()
    compared = 0
    worst = 0.0
    wrong = []
    for label, day, cars, functions in cases:
        n = day.n_service
        for size in range(1, n + 1):
            for subset in itertools.combinations(range(1, n + 1), size):
                least, most = lower_and_upper(cars, subset)
                mask = subset_mask(subset, n)
                gap = max(abs(least - functions.lower[mask]), abs(most - functions.upper[mask]))
                worst = max(worst, gap)
                compared += 1
                if gap > TOLERANCE:
                    wrong.append((label, day, subset, least, most, functions.lower[mask]))
    print(f"set functions: {compared - len(wrong)} of {compared} subsets equal; worst {worst:.1e}")
    assert not wrong, f"{len(wrong)} subsets differ; first: {wrong[:3]}"
    assert compared == N_SUBSETS


def test_a_point_meets_every_face_exactly_when_the_fleet_can_draw_it():
    _, _, cases = _sample()
    rng = random.Random(SEED + 1)
    inside = outside = one_face = 0
    wrong = []
    for label, day, cars, functions in cases:
        n = day.n_service
        bits = _bits(n)
        singles = [1 << k for k in range(n)]
        inner = random_point(rng, cars, n)
        for _ in range(N_POINTS):
            point = [
                rng.uniform(functions.lower[s] - MARGIN, functions.upper[s] + MARGIN)
                for s in singles
            ]
            sums = bits @ np.array(point)
            broken = int(
                np.sum(sums < functions.lower - 1e-9) + np.sum(sums > functions.upper + 1e-9)
            )
            drawn = attainable(cars, n, inner, point)
            if drawn != (broken == 0):
                wrong.append((label, day, point, broken, drawn))
            inside += broken == 0
            outside += broken > 0
            one_face += broken == 1
    total = len(cases) * N_POINTS
    print(
        f"points: {total - len(wrong)} of {total} agree; inside {inside}, outside {outside}, "
        f"exactly one face broken {one_face}"
    )
    assert not wrong, f"{len(wrong)} points disagree; first: {wrong[:3]}"
    assert inside + outside == total == 2 * N_DAYS * N_POINTS
    assert inside >= 0.1 * total and outside >= 0.1 * total
    assert one_face >= MIN_ONE_FACE


def test_the_step_to_the_boundary_from_the_faces_equals_the_programme():
    _, _, cases = _sample()
    rng = random.Random(SEED + 2)
    worst = 0.0
    steps = 0
    for _, day, cars, functions in cases:
        n = day.n_service
        bits = _bits(n)
        inner = random_point(rng, cars, n)
        for _ in range(N_DIRECTIONS):
            direction = np.array([rng.uniform(-1, 1) for _ in range(n)])
            slope = bits @ direction
            start = bits @ np.array(inner)
            limits = [np.inf]
            limits += list((functions.upper - start)[slope > 0] / slope[slope > 0])
            limits += list((functions.lower - start)[slope < 0] / slope[slope < 0])
            from_faces = max(min(limits), 0.0)
            from_programme = largest_step(cars, n, inner, list(direction))
            worst = max(worst, abs(from_faces - from_programme) / max(1.0, from_programme))
            steps += 1
    print(f"directions: {steps} steps, worst relative difference {worst:.1e}")
    assert steps == 2 * N_DAYS * N_DIRECTIONS
    assert worst <= TOLERANCE


def test_stored_energy_pass_without_discharge_equals_the_one_way_closed_form():
    days, _ = made_up_sample(random.Random(SEED + 3), N_CLOSED_FORM, service_range=(1, 8))
    worst = 0.0
    for day in days:
        one_way = day.one_way_cars()
        stored = [
            StoredEnergyCar(c.departure, 0.0, c.rate, 0.0, c.room, c.need(day.rho))
            for c in day.cars
        ]
        closed = subset_functions(one_way, day.n_service)
        passed = subset_functions(stored, day.n_service)
        worst = max(
            worst,
            float(np.max(np.abs(closed.lower - passed.lower))),
            float(np.max(np.abs(closed.upper - passed.upper))),
        )
        hours = list(day.service_hours)
        lower, upper = gpoly_subset_functions(one_way, hours)
        for mask in range(2 ** len(hours)):
            subset = [t for k, t in enumerate(hours) if mask >> k & 1]
            assert abs(lower[mask] - fleet_p(one_way, subset)) <= 1e-9
            assert abs(upper[mask] - fleet_b(one_way, subset)) <= 1e-9
    print(f"closed form against the stored-energy pass: {N_CLOSED_FORM} days, worst {worst:.1e}")
    assert len(days) == N_CLOSED_FORM
    assert {day.n_service for day in days} == set(range(1, 9))
    assert worst <= 1e-9
