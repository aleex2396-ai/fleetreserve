"""Seeded made-up fleets for the tests of `base_sets`, `lp24` and `inner_box`.

`cases(seed, count, service_range)`: `count` made-up days (`made_up_days.made_up_sample`), each
on its true sets one-way and two-way (budget cycling through `BUDGETS`), so `2 * count` cases.
Each case is `(day, budget, rows, base)` with `base` the day-average base set, or None when that
base set holds only the zero profile (every car without room), which `lp24` refuses.

`base_points(rng, base, count)`: random profiles of the base set (each hour uniform in
`[0, power_t]`, scaled down onto the energy limit when above it; every third point is scaled up
onto it when below, so the energy row is reached). `base_vertices(base)`: every vertex of the
base set `{0 <= u <= power, sum u <= energy}`: a set `A` of hours at full power with
`power(A) <= energy`, and a set `A` with one more hour `j` filled with the energy left when
`power(A) < energy < power(A) + power_j`.
"""

import random
from itertools import combinations

import numpy as np
from made_up_days import made_up_sample

from fleetreserve.base_sets import day_average
from fleetreserve.carsets import true_rows

BUDGETS = (0.0, 0.5, 2.0, 5.0)


def cases(seed, count, service_range=(2, 4)):
    rng = random.Random(seed)
    days, _ = made_up_sample(rng, count, service_range)
    found = []
    for k, day in enumerate(days):
        for budget in (None, BUDGETS[k % len(BUDGETS)]):
            rows = true_rows(day, budget)
            base = day_average(rows)
            found.append((day, budget, rows, None if base.is_trivial() else base))
    return found


def base_points(rng, base, count):
    points = []
    for k in range(count):
        u = np.array([rng.uniform(0.0, p) for p in base.power])
        total = u.sum()
        if total > base.energy or (k % 3 == 0 and total > 0):
            u *= base.energy / total
            u = np.minimum(u, base.power)
        points.append(u)
    return np.array(points)


def base_vertices(base):
    hours = range(base.power.size)
    vertices = []
    for size in range(len(hours) + 1):
        for subset in combinations(hours, size):
            u = np.zeros(base.power.size)
            u[list(subset)] = base.power[list(subset)]
            used = u.sum()
            if used <= base.energy + 1e-12:
                vertices.append(u)
            for j in hours:
                if j in subset or base.power[j] == 0:
                    continue
                if used < base.energy < used + base.power[j]:
                    v = u.copy()
                    v[j] = base.energy - used
                    vertices.append(v)
    return np.unique(np.round(np.array(vertices), 12), axis=0)
