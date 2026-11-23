"""Hand-worked cases for `exact`: set functions, ceilings, T1(a), T1(b) and their edge cases.

Car X: departure 4 (hours 0 to 3), rate 1 kW, room 1 kWh, rho = 0, on a day with service hours
{1, 2} and one recovery hour (3). It needs exactly its room, 1 kWh.

One-way (power in [0, 1], total energy exactly 1), with the formulas of `gpoly`:
  A = {1} or {2}: p = max(1 - 3 hours outside * 1, 0) = 0,  b = min(1 * 1, 1) = 1;
  A = {1, 2}:     p = max(1 - 2, 0) = 0,                    b = min(2, 1) = 1.
  Hourly ceiling b(S) - p(S) = 1. Block: 2 lam <= b - p = 1 on each hour, and 4 lam <= 1 on S, so
  lam = 0.25 and the value is 2 * 0.25 * 2 = 1.0.

Two-way at budget 1 kWh (power in [-1, 1], stored energy in [-1, 1] after every hour, 1 at the
end):
  b({1, 2}) = 2: discharge 1 in hour 0 (stored -1), charge 1 in hours 1 and 2 (stored 0, then 1),
    stay in hour 3.
  p({1, 2}) = -1: charge 1 in hour 0 (stored 1), discharge 1 in hour 1 (0), stay in hour 2, charge 1
    in hour 3 (1).
  b({1}) = b({2}) = 1 and p({1}) = p({2}) = -1, by the same plans.
  Hourly ceiling 2 - (-1) = 3. Block: 2 lam <= 2 on each hour and 4 lam <= 3 on S, so lam = 0.75,
  with centre (0.25, 0.25), and the value is 3.0.

Readings of plan section 9 that differ (car Y): departure 4, rate 1, room 0.5, rho = 0, budget 0,
service hours {1, 2, 3}, A = {1, 3}. With the stored energy limited after every hour (reading A,
used): charge 0.5 in hour 1 (stored 0.5, the room), discharge 0.5 in hour 2 (stored 0, the budget
allows no lower), charge 0.5 in hour 3, so b(A) = 1.0. With only the total limited (reading B, the
`gpoly.Car` with power in [-1, 1] and total 0.5): b(A) = -2 + min(2 * 2, 0.5 + 4) = 2.0, which
needs the car to sit 1 kWh below its starting level after hour 0.
"""

import datetime

import numpy as np
import pytest

from fleetreserve import exact
from fleetreserve.day import Day, TrueCar
from fleetreserve.exact import (
    SetFunctions,
    block_ceiling,
    box_inside,
    box_scale,
    day_functions,
    hourly_ceiling,
    subset_functions,
    subset_mask,
)
from fleetreserve.gpoly import Car
from fleetreserve.lp import LpResult

DAY_X = Day("unit", datetime.date(2019, 6, 3), 2, 1, 0.0, (TrueCar(4, 1.0, 1.0),))


def _table(functions):
    return {
        hours: (functions.lower[subset_mask(hours, 2)], functions.upper[subset_mask(hours, 2)])
        for hours in [(1,), (2,), (1, 2)]
    }


def test_car_x_one_way():
    functions = day_functions(DAY_X)
    assert _table(functions) == {(1,): (0, 1), (2,): (0, 1), (1, 2): (0, 1)}
    assert functions.lower[0] == functions.upper[0] == 0
    assert hourly_ceiling(functions) == 1.0
    block = block_ceiling(functions)
    assert block.outcome == "optimal"
    assert block.half_width == pytest.approx(0.25, abs=1e-9)
    assert block.value == pytest.approx(1.0, abs=1e-9)


def test_car_x_two_way_at_budget_1():
    functions = day_functions(DAY_X, budget=1.0)
    assert _table(functions) == {(1,): (-1, 1), (2,): (-1, 1), (1, 2): (-1, 2)}
    assert hourly_ceiling(functions) == 3.0
    block = block_ceiling(functions)
    assert block.half_width == pytest.approx(0.75, abs=1e-9)
    assert block.value == pytest.approx(3.0, abs=1e-9)
    assert sum(block.centre) == pytest.approx(0.5, abs=1e-9)
    assert box_inside(functions, np.array(block.centre) - 0.75, np.array(block.centre) + 0.75)


def test_readings_of_the_two_way_set_differ_on_car_y():
    car_y = TrueCar(4, 1.0, 0.5)
    stored = subset_functions([car_y.two_way(0.0, 0.0)], 3)
    total_only = subset_functions([Car(frozenset(range(4)), -1.0, 1.0, 0.5, 0.5)], 3)
    a = subset_mask((1, 3), 3)
    assert stored.upper[a] == pytest.approx(1.0)
    assert total_only.upper[a] == pytest.approx(2.0)


def test_box_inside_on_car_x():
    functions = day_functions(DAY_X)
    assert box_inside(functions, [0.0, 0.0], [0.5, 0.5])
    assert box_inside(functions, [0.0, 0.0], [1.0, 0.0])
    assert not box_inside(functions, [0.0, 0.0], [0.6, 0.5])  # hi(S) = 1.1 > b(S) = 1
    assert not box_inside(functions, [0.0, -0.1], [0.5, 0.5])  # lo({2}) = -0.1 < p({2}) = 0
    with pytest.raises(ValueError, match="lo above hi"):
        box_inside(functions, [0.6, 0.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="2 finite values"):
        box_inside(functions, [0.0], [0.5])


def test_box_scale_on_car_x_and_its_edge_cases():
    functions = day_functions(DAY_X)
    # Centre (0.25, 0.25), widths 1 up and down: S binds, (1 - 0.5) / 2 = 0.25 up, 0.5 / 2 down.
    assert box_scale(functions, [0.25, 0.25], [1, 1], [1, 1]) == pytest.approx(0.25)
    # Width only up in hour 1: {1} gives (1 - 0.25) / 1 and S gives (1 - 0.5) / 1 = 0.5.
    assert box_scale(functions, [0.25, 0.25], [1, 0], [0, 0]) == pytest.approx(0.5)
    # Every width zero: nothing limits the scale.
    assert box_scale(functions, [0.25, 0.25], [0, 0], [0, 0]) == float("inf")
    # A centre on the boundary (c(S) = b(S) = 1) gives 0 with a width up.
    assert box_scale(functions, [0.5, 0.5], [1, 1], [0, 0]) == 0.0
    # A centre outside the set is refused, even where its zero widths would hide it.
    with pytest.raises(ValueError, match="centre is outside"):
        box_scale(functions, [0.6, 0.5], [0, 0], [1, 1])
    with pytest.raises(ValueError, match="centre is outside"):
        box_scale(functions, [0.25, -0.01], [1, 1], [1, 1])
    with pytest.raises(ValueError, match="width is negative"):
        box_scale(functions, [0.25, 0.25], [1, -1], [1, 1])


def test_set_functions_validation():
    with pytest.raises(ValueError, match="n_service must be 1 to 8"):
        subset_functions([], 9)
    with pytest.raises(ValueError, match="gpoly.Car or a StoredEnergyCar"):
        subset_functions([TrueCar(2, 1.0, 1.0)], 2)
    with pytest.raises(ValueError, match="lower must hold 4 finite values"):
        SetFunctions(2, np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="not a service hour"):
        subset_mask((0,), 2)
    empty = subset_functions([], 3)
    assert hourly_ceiling(empty) == 0.0
    assert empty.lower.flags.writeable is False


@pytest.mark.parametrize("outcome", ["timeout", "error", "infeasible"])
def test_block_ceiling_that_is_not_optimal_is_missing_not_zero(monkeypatch, outcome):
    def not_optimal(*args, **kwargs):
        return LpResult(outcome, None, None, 31.0, kwargs["time_limit"], 1, "made up")

    monkeypatch.setattr(exact, "solve_lp", not_optimal)
    block = block_ceiling(day_functions(DAY_X))
    assert block.outcome == outcome
    assert block.half_width is None and block.centre is None and block.value is None
