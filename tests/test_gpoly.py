"""Hand-computed cases for the set functions p, b and the box test of `fleetreserve.gpoly`."""

import pytest

from fleetreserve.gpoly import (
    MAX_HOURS,
    Car,
    band_scale,
    box_inside,
    car_b,
    car_p,
    fleet_b,
    fleet_p,
)

# One-way car: hours {0, 1}, power 0..2, energy 1..3.
ONE_WAY = Car(frozenset({0, 1}), 0, 2, 1, 3)
# Two-way car: hours {0, 1}, power -1..1, net energy -0.5..1.
TWO_WAY = Car(frozenset({0, 1}), -1, 1, -0.5, 1)


def test_one_way_set_functions():
    # A = {0}: p = max(1 - 1 hour outside * 2, 0) = 0;  b = min(1 hour * 2, 3) = 2.
    assert car_p(ONE_WAY, frozenset({0})) == 0
    assert car_b(ONE_WAY, frozenset({0})) == 2
    # A = {0, 1}: p = max(1 - 0, 0) = 1;  b = min(2 hours * 2, 3) = 3.
    assert car_p(ONE_WAY, frozenset({0, 1})) == 1
    assert car_b(ONE_WAY, frozenset({0, 1})) == 3
    # A = {}: p = max(1 - 2 * 2, 0) = 0;  b = min(0, 3) = 0.
    assert car_p(ONE_WAY, frozenset()) == 0
    assert car_b(ONE_WAY, frozenset()) == 0


def test_two_way_set_functions_and_fleet_sums():
    # Shift by -1: v = u + 1 in 0..2, energy limits -0.5 + 2 = 1.5 and 1 + 2 = 3.
    # A = {0}: p' = max(1.5 - 1 * 2, 0) = 0, b' = min(1 * 2, 3) = 2; shift back (-1 * 1):
    #   p = -1, b = 1.
    assert car_p(TWO_WAY, frozenset({0})) == -1
    assert car_b(TWO_WAY, frozenset({0})) == 1
    # A = {0, 1}: p' = max(1.5, 0) = 1.5, b' = min(4, 3) = 3; shift back (-1 * 2): p = -0.5, b = 1,
    # which are the car's own energy limits.
    assert car_p(TWO_WAY, frozenset({0, 1})) == -0.5
    assert car_b(TWO_WAY, frozenset({0, 1})) == 1
    # Fleet sums over both cars at A = {0}: p = 0 + -1, b = 2 + 1.
    assert fleet_p([ONE_WAY, TWO_WAY], {0}) == -1
    assert fleet_b([ONE_WAY, TWO_WAY], {0}) == 3


def test_box_inside_single_one_way_car():
    # The one-way car above is the whole fleet; hours {0, 1}. Faces: u0 <= 2, u1 <= 2,
    # u0 + u1 in [1, 3], and 0 <= u0, 0 <= u1.
    hours = [0, 1]
    assert box_inside([0.5, 0.5], [1.5, 1.5], [ONE_WAY], hours)  # corners (0.5,0.5): sum 1 ok
    assert not box_inside([0.4, 0.4], [1.5, 1.5], [ONE_WAY], hours)  # (0.4, 0.4): sum 0.8 < 1
    assert not box_inside([0.5, 0.5], [1.6, 1.6], [ONE_WAY], hours)  # (1.6, 1.6): sum 3.2 > 3
    assert not box_inside([0.5, 0.5], [2.1, 1.0], [ONE_WAY], hours)  # 2.1 > rate 2


def test_free_hours_outside_the_box_help_the_lower_side():
    # Car in hours {0, 1}, power 0..1, energy exactly 1.5; the box sees hour 0 only.
    # p({0}) = max(1.5 - 1 * 1, 0) = 0.5: hour 1 can give at most 1, so hour 0 must give 0.5.
    car = Car(frozenset({0, 1}), 0, 1, 1.5, 1.5)
    assert box_inside([0.5], [1.0], [car], [0])
    assert not box_inside([0.4], [1.0], [car], [0])


def test_refusals():
    with pytest.raises(ValueError):
        box_inside([0] * 17, [0] * 17, [ONE_WAY], range(MAX_HOURS + 1))
    with pytest.raises(ValueError):
        box_inside([1.0], [0.0], [ONE_WAY], [0])  # lo above hi
    with pytest.raises(ValueError):
        Car(frozenset({0}), 0, 1, 2, 3)  # needs 2 kWh in one hour at 1 kW


# Band cases. Cars X (hour 0, power 0..2, energy 1.5..2, so u0 in [1.5, 2]) and Y (hour 1, power
# 0..2, energy 0..2, so u1 in [0, 2]); service hours {0, 1}; widths 1 up and 1 down in both hours.
# Band: (c0 + s, c1 - s) with |s| <= lam.
BAND_X = Car(frozenset({0}), 0, 2, 1.5, 2)
BAND_Y = Car(frozenset({1}), 0, 2, 0, 2)


def test_band_scale_when_the_upper_term_binds():
    # Centre (1.8, 0.5), c(S) = 2.3. u0 = 1.8 + s <= 2 gives s <= 0.2; the other faces are slacker,
    # so lam* = 0.2. By the formula, A = {0}: b(A) = 2 + 0 = 2 and c(S) - p({1}) = 2.3 - 0 = 2.3;
    # min = 2, minus c(A) = 1.8 gives 0.2, over min(w_up(A), w_down({1})) = 1. A = {1}: b = 2 and
    # c(S) - p({0}) = 2.3 - 1.5 = 0.8; min = 0.8, minus c(A) = 0.5 gives 0.3, over 1.
    assert band_scale([1.8, 0.5], [1, 1], [1, 1], [BAND_X, BAND_Y], [0, 1]) == pytest.approx(0.2)


def test_band_scale_when_the_lower_term_binds():
    # Centre (1.6, 0.5), c(S) = 2.1. u0 = 1.6 + s >= 1.5 gives s >= -0.1, so lam* = 0.1. By the
    # formula, A = {1}: b(A) = 2 and c(S) - p({0}) = 2.1 - 1.5 = 0.6; min = 0.6, minus c(A) = 0.5
    # gives 0.1, over 1. A = {0}: min(2, 2.1) = 2, minus 1.6 gives 0.4, over 1.
    assert band_scale([1.6, 0.5], [1, 1], [1, 1], [BAND_X, BAND_Y], [0, 1]) == pytest.approx(0.1)


def test_band_scale_counts_free_hours_outside_the_service_hours():
    # One car, hours {0, 1, 2}, power 0..1, energy exactly 2; service hours {0, 1}, hour 2 free.
    # Seen on {0, 1}: u0, u1 in [0, 1] and u0 + u1 in [1, 2]. Centre (0.8, 0.8), c(S) = 1.6,
    # widths 1: the band (0.8 + s, 0.8 - s) needs |s| <= 0.2, so lam* = 0.2. By the formula,
    # A = {0}: b(A) = min(1, 2) = 1 and c(S) - p({1}) = 1.6 - max(2 - 2, 0) = 1.6; min = 1, minus
    # 0.8 gives 0.2, over 1. A = {1} is the same.
    car = Car(frozenset({0, 1, 2}), 0, 1, 2, 2)
    assert band_scale([0.8, 0.8], [1, 1], [1, 1], [car], [0, 1]) == pytest.approx(0.2)


def test_band_scale_omits_zero_widths_and_refuses_bad_input():
    # Widths up (1, 0), down (0, 1) in the first case: only the move (+s, -s) is allowed, and
    # w_up({0}) = 1, w_down({1}) = 1 gives A = {0} denominator 1; A = {1} has w_up({1}) = 0, so it
    # is omitted. With the numbers of the first band case, lam* = 0.2.
    cars = [BAND_X, BAND_Y]
    assert band_scale([1.8, 0.5], [1, 0], [0, 1], cars, [0, 1]) == pytest.approx(0.2)
    assert band_scale([1.8, 0.5], [0, 0], [0, 0], cars, [0, 1]) == float("inf")
    with pytest.raises(ValueError):
        band_scale([2.5, 0.5], [1, 1], [1, 1], cars, [0, 1])  # u0 = 2.5 is above the 2 kW limit
    with pytest.raises(ValueError):
        band_scale([1.8, 0.5], [-1, 1], [1, 1], cars, [0, 1])  # negative width
    with pytest.raises(ValueError):
        band_scale([0] * 17, [1] * 17, [1] * 17, [ONE_WAY], range(MAX_HOURS + 1))
