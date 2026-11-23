"""The base sets of `base_sets` on fleets small enough to work out by hand.

Two service hours, so hours 0, 1, 2.

- True car A: departure 2, rate 3, room 4: power limits (3, 3, 0), energy 4, upper (3, 3, 0).
- True car B: departure 5, rate 2, room 1: power limits (2, 2, 2), energy 1, upper (1, 1, 1)
  (no hour can take more than the whole energy).
- Forecast car F: departure 1.5, rate 4, cap 3, need 1: power limits (4, 2, 0), energy 3,
  upper (3, 2, 0).
- Toy car G (`gpoly.Car`): hours 1, 2, 3, power in [0.5, 2], energy in [1, 4]: power limits
  (0, 2, 2), energy 4 - 0.5 * 1 = 3.5 (hour 3 is a recovery hour at least at 0.5 kW), upper
  (0, 2, 2).

Day average of A and B: power (2, 2, 0.5), energy 2.5. Template of the fleets {A, B} and {F}:
power ((3 + 1 + 3) / 3, (3 + 1 + 2) / 3, 1 / 3), energy 8 / 3, which is not the average of the
two day averages. Two-way rows of A and B (budget 2) give the same base set as their one-way
rows: lower limits do not enter.
"""

import datetime

import numpy as np
import pytest

from fleetreserve import base_sets, lp24
from fleetreserve.carsets import (
    forecast_rows,
    gpoly_rows,
    stack,
    true_one_way_rows,
    true_rows,
    true_two_way_rows,
)
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.gpoly import Car

N = 2
A = TrueCar(2, 3.0, 4.0)
B = TrueCar(5, 2.0, 1.0)
F = ForecastCar(1.5, 4.0, 3.0, 1.0)
G = Car(frozenset({1, 2, 3}), 0.5, 2.0, 1.0, 4.0)


def _fleet(*cars):
    return stack(list(cars))


def test_car_limits_by_hand():
    rows = _fleet(
        true_one_way_rows(A, N, 0.1),
        true_one_way_rows(B, N, 0.1),
        forecast_rows(F, N, 0),
        gpoly_rows(G, N),
    )
    upper, energy = base_sets.car_limits(rows)
    np.testing.assert_allclose(upper, [[3, 3, 0], [1, 1, 1], [3, 2, 0], [0, 2, 2]])
    np.testing.assert_allclose(energy, [4, 1, 3, 3.5])


def test_day_average_by_hand():
    base = base_sets.day_average(_fleet(true_one_way_rows(A, N, 0), true_one_way_rows(B, N, 0)))
    np.testing.assert_allclose(base.power, [2, 2, 0.5])
    assert base.energy == pytest.approx(2.5)
    np.testing.assert_array_equal(base.columns, [0, 1, 2])
    two_way = base_sets.day_average(
        _fleet(true_two_way_rows(A, N, 0, 2.0), true_two_way_rows(B, N, 0, 2.0))
    )
    np.testing.assert_allclose(two_way.power, base.power)
    assert two_way.energy == pytest.approx(base.energy)


def test_template_pools_cars_not_days():
    first = _fleet(true_one_way_rows(A, N, 0), true_one_way_rows(B, N, 0))
    second = _fleet(forecast_rows(F, N, 0))
    base = base_sets.template([first, second])
    np.testing.assert_allclose(base.power, [7 / 3, 2, 1 / 3])
    assert base.energy == pytest.approx(8 / 3)
    by_days = (base_sets.day_average(first).energy + base_sets.day_average(second).energy) / 2
    assert base.energy != pytest.approx(by_days)
    with pytest.raises(ValueError):
        base_sets.template([first, _fleet(true_one_way_rows(A, 3, 0))])


def test_a_fleet_without_room_has_a_trivial_base_set():
    day = Day("made-up", datetime.date(2020, 1, 1), N, 0, 0.0, (TrueCar(2, 3.0, 0.0),) * 2)
    rows = true_rows(day)
    base = base_sets.day_average(rows)
    assert base.is_trivial()
    with pytest.raises(ValueError, match="zero profile"):
        lp24.inner_set(rows, base, "24")


def test_base_set_contains():
    base = base_sets.BaseSet([2.0, 2.0, 0.5], 2.5)
    assert base.contains([1.0, 1.0, 0.5])
    assert not base.contains([2.0, 1.0, 0.0])  # energy 3 > 2.5
    assert not base.contains([0.0, 0.0, 0.6])  # above the power limit
    assert not base.contains([-0.1, 0.0, 0.0])
