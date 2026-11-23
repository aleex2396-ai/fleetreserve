"""Remark 5 of Al Taha et al. (2025): LP (27) splits into one programme per car.

On `N_DAYS` seeded made-up days (`lp24_cases`, 2 to 4 service hours), each one-way and two-way,
`N_SOLVED` cases with a non-trivial day-average base set: LP (27) on the whole fleet has the same
value as the sum of LP (27) on each car alone with the same base set, within `TOLERANCE`
(relative to `max(1, value)`). At least `MIN_SEVERAL` cases have more than one car.
"""

import numpy as np
from lp24_cases import cases

from fleetreserve.carsets import FleetRows
from fleetreserve.lp24 import inner_set

SEED = 23
N_DAYS = 30
N_SOLVED = 60
MIN_SEVERAL = 30
TOLERANCE = 1e-6


def _car(rows: FleetRows, i: int) -> FleetRows:
    mine = rows.car == i
    return FleetRows(
        rows.n_service,
        1,
        np.zeros(int(mine.sum()), int),
        rows.coefficients[mine],
        rows.bounds[mine],
        rows.kinds[mine],
    )


def test_lp27_is_the_sum_of_per_car_programmes():
    found = [(rows, base) for *_, rows, base in cases(SEED, N_DAYS) if base is not None]
    assert len(found) == N_SOLVED
    assert sum(rows.n_cars > 1 for rows, _ in found) >= MIN_SEVERAL
    differ = []
    for rows, base in found:
        whole = inner_set(rows, base, "27")
        parts = [inner_set(_car(rows, i), base, "27") for i in range(rows.n_cars)]
        assert whole.outcome == "optimal" and all(p.outcome == "optimal" for p in parts)
        total = sum(p.value for p in parts)
        if abs(whole.value - total) > TOLERANCE * max(1.0, abs(total)):
            differ.append((whole.value, total))
    assert not differ, differ
