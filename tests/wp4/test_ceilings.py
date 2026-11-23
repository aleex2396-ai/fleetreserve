"""Plan section 2: the hourly and block ceilings of `exact` against linear programmes on toy days.

Sample: `N_DAYS` seeded made-up days (`made_up_days.py`; 2 to 4 service hours), each taken one-way
and two-way (budget cycling through `BUDGETS`).

- Hourly ceiling b(S) - p(S) equals the largest free box met on every corner path by a power plan
  that knows the whole path: `toy_values.hindsight_value_by_paths` for one-way days (it takes
  `gpoly.Car`), `fleet_lps.free_box_by_paths` for both (it writes the stored-energy rows out). The
  two programmes must agree with each other on one-way days.
- Block ceiling (one linear programme over the half-width and the centre) equals
  `toy_values.block_hindsight_scale` (the same faces, from `gpoly`'s scalar functions) and the
  path programme `toy_values.block_hindsight_scale_by_paths` on one-way days, and
  `fleet_lps.block_by_paths` on two-way days. Its centre +- half-width passes `box_inside`, a
  half-width larger by `STEP` does not, and the value is 2 lam |S|.

The sample must hold at least `MIN_BLOCK_BELOW` days where the block value is below the hourly
ceiling, so the block programme is not just the hourly ceiling divided evenly.
"""

import random

import numpy as np
import pytest
from fleet_lps import block_by_paths, free_box_by_paths
from made_up_days import made_up_sample
from toy_values import (
    block_hindsight_scale,
    block_hindsight_scale_by_paths,
    hindsight_value_by_paths,
)

from fleetreserve.exact import block_ceiling, box_inside, day_functions, hourly_ceiling

SEED = 11
N_DAYS = 50
BUDGETS = (0.0, 0.5, 2.0)
TOLERANCE = 1e-6  # relative to max(1, value)
STEP = 1e-4  # kW added to the half-width to leave the set
MIN_BLOCK_BELOW = 20


def _close(a, b):
    return abs(a - b) <= TOLERANCE * max(1.0, abs(b))


@pytest.fixture(scope="module")
def days():
    sample, _ = made_up_sample(random.Random(SEED), N_DAYS)
    assert len(sample) == N_DAYS
    assert sum(day.has_in_window_departure for day in sample) >= 15
    return sample


def _two_way(days):
    for k, day in enumerate(days):
        budget = BUDGETS[k % len(BUDGETS)]
        yield day, budget, day.two_way_cars(budget)


def test_hourly_ceiling_equals_the_path_programmes(days):
    wrong = []
    for day in days:
        cars, hours = day.one_way_cars(), list(day.service_hours)
        exact_value = hourly_ceiling(day_functions(day))
        toy = hindsight_value_by_paths(list(cars), hours)
        written_out = free_box_by_paths(cars, day.n_service)
        if not (_close(exact_value, toy) and _close(exact_value, written_out)):
            wrong.append(("one-way", day, exact_value, toy, written_out))
    for day, budget, cars in _two_way(days):
        exact_value = hourly_ceiling(day_functions(day, budget))
        written_out = free_box_by_paths(cars, day.n_service)
        if not _close(exact_value, written_out):
            wrong.append(("two-way", day, budget, exact_value, written_out))
    print(f"hourly ceiling: {2 * N_DAYS - len(wrong)} of {2 * N_DAYS} days equal")
    assert not wrong, f"{len(wrong)} days differ: {wrong}"


def _check_block(functions, block, n):
    assert block.outcome == "optimal"
    assert block.value == pytest.approx(2 * block.half_width * n)
    centre = np.array(block.centre)
    assert box_inside(functions, centre - block.half_width, centre + block.half_width)
    wider = block.half_width + STEP
    assert not box_inside(functions, centre - wider, centre + wider)


def test_block_ceiling_equals_the_toy_programmes(days):
    wrong = []
    below = 0
    for day in days:
        functions = day_functions(day)
        block = block_ceiling(functions)
        _check_block(functions, block, day.n_service)
        cars, hours = list(day.one_way_cars()), list(day.service_hours)
        faces = block_hindsight_scale(cars, hours)
        paths = block_hindsight_scale_by_paths(cars, hours)
        if not (_close(block.half_width, faces) and _close(block.half_width, paths)):
            wrong.append(("one-way", day, block.half_width, faces, paths))
        below += block.value < hourly_ceiling(functions) - TOLERANCE
    for day, budget, cars in _two_way(days):
        functions = day_functions(day, budget)
        block = block_ceiling(functions)
        _check_block(functions, block, day.n_service)
        paths = block_by_paths(cars, day.n_service)
        if not _close(block.half_width, paths):
            wrong.append(("two-way", day, budget, block.half_width, paths))
        below += block.value < hourly_ceiling(functions) - TOLERANCE
    print(
        f"block ceiling: {2 * N_DAYS - len(wrong)} of {2 * N_DAYS} days equal; "
        f"block value below the hourly ceiling on {below}"
    )
    assert not wrong, f"{len(wrong)} days differ: {wrong}"
    assert below >= MIN_BLOCK_BELOW
