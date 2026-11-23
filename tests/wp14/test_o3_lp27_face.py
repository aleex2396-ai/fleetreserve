"""LP (27)'s search uses only solutions on LP (27)'s optimal face (`o3_lp27`, module text: the
face's tolerance).

1. A made-up day: rho 0, three service hours, cars (departure, rate kW, room kWh) (2, 2.0, 2.263),
   (6, 6.6, 35.43), (2, 2.0, 0.289); the day-average base set has power 2.963 kW on hours 0 and 1
   and 2.2 kW on hours 2 and 3, energy 9.6507 kWh. On LP (27)'s exact optimal face (the trace at
   least its optimum, no tolerance) the map entries P[2,0], P[2,1], P[3,0] and P[3,1] each have
   largest and smallest value 0, and P[1,1] - P[1,0] has largest and smallest value 0.6838 (each
   one linear programme over the face, with U0 given by its vertices). So in every optimal
   solution's set, the profiles with v_2 = v_3 = 0 and v_0, v_1 anywhere in [0, 2.963] (in U0,
   since 2 x 2.963 <= 9.6507) keep hours 2 and 3 fixed while hour 1, p_1 + P[1,0] v_0 +
   P[1,1] v_1, runs over an interval of length 2.963 (|P[1,0]| + |P[1,1]|) >= 2.963 x 0.6838 =
   2.026 (2.026000 to six places): that interval, with hours 2 and 3 fixed and hour 0 free, is an
   hourly box of value 2.026. The smallest hourly box over LP (27)'s optimal solutions is
   therefore at least 2.026 kWh, and so is the smallest found over any of them. With solutions
   up to 1e-7 (relative) below the optimal trace the search found 1.620: maps a little below the
   optimum can tilt hour 1 into hours 2 and 3, and the box loses its width.
2. A solution off the face, or a candidate programme with no solution, is passed over and counted
   exactly as a programme that ends with a solver error, never used. On the one-car day of
   `test_o3_lp27` (rho 0.2, two service hours, departure 3, rate 2, room 3.519): LP (27) with the
   cars reversed (step 2b) or the first seeded draw (step 2c) replaced by a map of trace 0 (its
   set is one point, whose box is 0) gives the same result as the same programme ending with a
   solver error, so the smallest found stays above 0; the first draw ending "infeasible" gives
   that result too, values present; every ascent step's map replaced by a map of trace 0 gives
   the result of every ascent programme ending "infeasible".
"""

import dataclasses
import datetime

import numpy as np
import pytest

from fleetreserve import o3_lp27
from fleetreserve.base_sets import day_average
from fleetreserve.carsets import true_rows
from fleetreserve.day import Day, TrueCar, covering_recovery_hours
from fleetreserve.o3_day import PRODUCTS, car_order
from fleetreserve.products import Product

OTHERS, BOX = 300.0, 30.0
TILTED_DAY = ((2, 2.0, 2.263), (6, 6.6, 35.43), (2, 2.0, 0.289))
SMALLEST_HOURLY = 2.026  # kWh, the bound of item 1
ONE_CAR_DAY = ((3, 2.0, 3.519),)
TOLERANCE = 1e-6


def _rows(cars, n_service, rho):
    cars = tuple(TrueCar(*car) for car in cars)
    recovery = covering_recovery_hours(cars, n_service)
    day = Day("made-up", datetime.date(2020, 1, 1), n_service, recovery, rho, cars)
    return true_rows(car_order(day))


def _search(rows):
    return o3_lp27.search(rows, day_average(rows), PRODUCTS, OTHERS, BOX)


def test_the_smallest_box_found_is_taken_over_optimal_solutions_only():
    rows = _rows(TILTED_DAY, 3, 0.0)
    base = day_average(rows)
    assert base.power == pytest.approx([2.963, 2.963, 2.2, 2.2], abs=1e-12)
    assert base.energy == pytest.approx(9.6507, abs=1e-4)
    found = _search(rows)
    assert found.outcomes == dict.fromkeys(PRODUCTS, "optimal")
    assert found.low["hourly"] >= SMALLEST_HOURLY - TOLERANCE, found.low


def _flat(maps):
    p, P = maps
    return np.asarray(p, dtype=float), np.zeros_like(np.asarray(P, dtype=float))


def _same_as_passed_over(found, passed_over):
    assert found.outcomes == dict.fromkeys(PRODUCTS, "optimal")
    assert all(found.low[name] > TOLERANCE for name in PRODUCTS), found.low
    assert found == passed_over


def _point_box_is_zero(rows):
    base = day_average(rows)
    hours = rows.n_service + 1
    for name in PRODUCTS:
        box, _ = o3_lp27.largest_box(
            np.zeros(hours), np.zeros((hours, hours)), base, Product(name), BOX
        )
        assert box.value == pytest.approx(0.0, abs=1e-9)


def _reversed_replaced(monkeypatch, change):
    real = o3_lp27.inner_set
    calls = []

    def inner(*args, **kwargs):
        found = real(*args, **kwargs)
        calls.append(None)
        return change(found) if len(calls) == 2 else found  # the second: the cars reversed

    monkeypatch.setattr(o3_lp27, "inner_set", inner)


def test_lp27_with_the_cars_reversed_is_used_only_on_the_face(monkeypatch):
    rows = _rows(ONE_CAR_DAY, 2, 0.2)
    _point_box_is_zero(rows)
    with monkeypatch.context() as patch:
        _reversed_replaced(patch, lambda r: dataclasses.replace(r, outcome="error", p=None, P=None))
        passed_over = _search(rows)
    with monkeypatch.context() as patch:
        _reversed_replaced(patch, lambda r: dataclasses.replace(r, P=_flat((r.p, r.P))[1]))
        found = _search(rows)
    _same_as_passed_over(found, passed_over)


def _first_draw_replaced(monkeypatch, change):
    real = o3_lp27._drawn

    def drawn(rows, base, trace, k, time_limit):
        outcome, maps, seconds = real(rows, base, trace, k, time_limit)
        return change(outcome, maps, seconds) if k == 1 else (outcome, maps, seconds)

    monkeypatch.setattr(o3_lp27, "_drawn", drawn)


def test_a_drawn_solution_is_used_only_on_the_face(monkeypatch):
    rows = _rows(ONE_CAR_DAY, 2, 0.2)
    with monkeypatch.context() as patch:
        _first_draw_replaced(patch, lambda outcome, maps, seconds: ("error", None, seconds))
        passed_over = _search(rows)
    with monkeypatch.context() as patch:
        _first_draw_replaced(patch, lambda outcome, maps, seconds: (outcome, _flat(maps), seconds))
        found = _search(rows)
    _same_as_passed_over(found, passed_over)


def test_a_candidate_programme_ending_infeasible_is_passed_over(monkeypatch):
    rows = _rows(ONE_CAR_DAY, 2, 0.2)
    with monkeypatch.context() as patch:
        _first_draw_replaced(patch, lambda outcome, maps, seconds: ("error", None, seconds))
        passed_over = _search(rows)
    with monkeypatch.context() as patch:
        _first_draw_replaced(patch, lambda outcome, maps, seconds: ("infeasible", None, seconds))
        found = _search(rows)
    _same_as_passed_over(found, passed_over)


def _ascent_replaced(monkeypatch, change):
    real = o3_lp27._ascent_step

    def step(rows, base, trace, product, profiles, time_limit):
        return change(*real(rows, base, trace, product, profiles, time_limit))

    monkeypatch.setattr(o3_lp27, "_ascent_step", step)


def test_an_ascent_map_off_the_face_ends_the_ascent(monkeypatch):
    rows = _rows(ONE_CAR_DAY, 2, 0.2)
    with monkeypatch.context() as patch:
        _ascent_replaced(patch, lambda outcome, maps, seconds: ("infeasible", None, seconds))
        passed_over = _search(rows)
    with monkeypatch.context() as patch:
        _ascent_replaced(
            patch,
            lambda outcome, maps, seconds: (outcome, maps and _flat(maps), seconds),
        )
        found = _search(rows)
    assert passed_over.skipped > 0
    _same_as_passed_over(found, passed_over)
