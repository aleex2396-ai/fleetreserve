"""Plan section 4, T4 (free shape, "tested, not proved here"): on the hourly product the diagonal,
causal and tree values equal hindsight. Checked here on the day model of `day` (hour 0 before S).

Recorded finding. On true sets with hour 0, the three values are equal to each other, but they are
below `exact`'s hourly ceiling b(S) - p(S) on many days, one-way and two-way. The cause is hour 0:
`exact` treats hour 0 as free in the projection, so its hindsight plan may choose the hour-0 powers
knowing the whole path, while every causal rule (and the tree, whose root holds hour 0) fixes them
before the first request. Against a hindsight whose hour-0 powers are one plan for every path
(`rows_lps.committed_hindsight_value`) the three values are equal on every sampled day. The
smallest failing day is kept as a test case: one car present in hours 0 and 1 (departure 2), rate
1 kW, room 1 kWh, rho 0, one service hour; exact ceiling 1 kWh, every causal value 0. Which
ceiling the project uses is not settled here.

Samples:

1. `N_DAYS` seeded made-up days (`made_up_days.py`), each on its true sets one-way and two-way
   (reading A of `day`, budget cycling through `BUDGETS`). Every case: diagonal = causal = tree =
   committed hindsight. The cases below the exact ceiling are counted: `BELOW_ONE_WAY` and
   `BELOW_TWO_WAY` (fixed by the seed).
2. `N_TOY` toy fleets (`test_t4_conjecture._fleet`, no hour 0, two-way cars as `gpoly.Car` with
   only total energy limits): diagonal = tree = `toy_values.hindsight_value`, the setting in which
   the plan's 400/400 was found.
"""

import datetime
import functools
import random

import pytest
from made_up_days import made_up_sample
from rows_lps import committed_hindsight_value
from test_t4_conjecture import _fleet
from toy_values import hindsight_value

from fleetreserve import exact, products
from fleetreserve.carsets import gpoly_fleet_rows, true_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.day import Day, TrueCar
from fleetreserve.tree import corner_tree

SEED = 61
N_DAYS = 80
BUDGETS = (0.0, 0.5, 2.0, 5.0)
BELOW_ONE_WAY = 49
BELOW_TWO_WAY = 61
N_TOY = 60
TOLERANCE = 1e-6  # relative to max(1, value)


def _close(a, b):
    return abs(a - b) <= TOLERANCE * max(1.0, abs(b))


def _hourly_values(rows):
    values = {
        s: certify(rows, products.hourly(), s, shape_rule=ANY_OPTIMAL)
        for s in ("diagonal", "causal")
    }
    values["tree"] = corner_tree(rows, products.hourly())
    assert all(v.outcome == "optimal" for v in values.values())
    return {s: v.value for s, v in values.items()}


@functools.cache
def _cases():
    days, _ = made_up_sample(random.Random(SEED), N_DAYS)
    cases = []
    for k, day in enumerate(days):
        for budget in (None, BUDGETS[k % len(BUDGETS)]):
            rows = true_rows(day, budget)
            values = _hourly_values(rows)
            values["committed"] = committed_hindsight_value(rows)
            values["ceiling"] = exact.hourly_ceiling(exact.day_functions(day, budget))
            cases.append((day, budget, values))
    return cases


def test_diagonal_causal_and_tree_equal_hindsight_with_hour_0_committed():
    cases = _cases()
    unequal = [
        (budget, v, day)
        for day, budget, v in cases
        if not all(_close(v[s], v["committed"]) for s in ("diagonal", "causal", "tree"))
    ]
    assert len(cases) == 2 * N_DAYS
    assert not unequal, unequal


def test_recorded_finding_the_values_fall_below_the_exact_ceiling_on_many_days():
    cases = _cases()
    below = {"one-way": 0, "two-way": 0}
    for _, budget, v in cases:
        assert v["tree"] <= v["ceiling"] + TOLERANCE * max(1.0, v["ceiling"])
        if v["tree"] < v["ceiling"] - TOLERANCE * max(1.0, v["ceiling"]):
            below["one-way" if budget is None else "two-way"] += 1
    print(f"hourly tree below the exact ceiling b(S) - p(S): {below} of {N_DAYS} days each")
    assert below == {"one-way": BELOW_ONE_WAY, "two-way": BELOW_TWO_WAY}


@pytest.mark.parametrize("budget", [None, 0.0])
def test_recorded_finding_smallest_day(budget):
    car = TrueCar(2, 1.0, 1.0)
    day = Day("made-up", datetime.date(2020, 1, 1), 1, 0, 0.0, (car,))
    rows = true_rows(day, budget)
    values = _hourly_values(rows)
    ceiling = exact.hourly_ceiling(exact.day_functions(day, budget))
    print(f"budget {budget}: exact ceiling {ceiling}, causal values {values}")
    assert ceiling == pytest.approx(1.0)
    assert committed_hindsight_value(rows) == pytest.approx(0.0, abs=1e-9)
    for value in values.values():
        assert value == pytest.approx(0.0, abs=1e-9)


def test_toy_fleets_without_hour_0_diagonal_and_tree_equal_hindsight():
    rng = random.Random(SEED)
    unequal = []
    for k in range(N_TOY):
        n = 2 + k % 3
        cars = _fleet(rng, n, two_way=k % 2 == 1, in_window=(k // 2) % 2 == 0)
        values = _hourly_values(gpoly_fleet_rows(cars, n))
        reference = hindsight_value(cars, list(range(1, n + 1)))
        if not all(_close(values[s], reference) for s in ("diagonal", "tree")):
            unequal.append((values, reference, cars))
    assert not unequal, unequal
