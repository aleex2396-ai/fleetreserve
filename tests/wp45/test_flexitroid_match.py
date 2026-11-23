"""`flexitroid_match.compare_day` on made-up one-way fleets: agreement, and every way it must not
agree (a wrong ruler, a wrong optimisation, flexitroid failing, a programme timing out)."""

import datetime
import random

import numpy as np
import pytest
from made_up_days import made_up_sample

from fleetreserve import exact
from fleetreserve import flexitroid_match as fm
from fleetreserve.day import Day, TrueCar, covering_recovery_hours
from fleetreserve.lp import LpResult

DATE = datetime.date(2020, 1, 1)
SAMPLE_DAYS = 60


def made_day(cars, n_service=4, rho=0.0) -> Day:
    return Day("made-up", DATE, n_service, covering_recovery_hours(cars, n_service), rho, cars)


SINGLE_CAR = made_day([TrueCar(6, 7.2, 20.0)])
ZERO_ROOM = made_day([TrueCar(5, 3.0, 0.0), TrueCar(7, 6.6, 15.0), TrueCar(3, 2.0, 4.0)])
LEAVES_IN_WINDOW = made_day(
    [TrueCar(2, 3.0, 4.5), TrueCar(4, 6.6, 10.0), TrueCar(9, 7.2, 30.0)], n_service=6, rho=0.1
)
EIGHT_HOURS = made_day(
    [
        TrueCar(d, r, e)
        for d, r, e in ((3, 3.3, 5.0), (8, 6.6, 30.0), (12, 7.2, 40.0), (1, 2.0, 1.0))
    ],
    n_service=8,
    rho=0.2,
)
NAMED = {
    "single car": SINGLE_CAR,
    "zero room": ZERO_ROOM,
    "leaves in window": LEAVES_IN_WINDOW,
    "eight service hours": EIGHT_HOURS,
}


def seeded_days():
    days, kinds = made_up_sample(random.Random(20261111), SAMPLE_DAYS, service_range=(1, 8))
    return days, kinds


def test_the_seeded_sample_holds_the_cases_that_matter():
    days, kinds = seeded_days()
    assert any(day.has_in_window_departure for day in days)
    assert any("empty" in day_kinds for day_kinds in kinds)  # a car with zero room
    assert any(len(day.cars) == 1 for day in days)
    assert any(day.n_service == 8 for day in days)
    assert {day.rho for day in days} == {0.0, 0.1, 0.2}


def test_flexitroid_matches_on_every_seeded_made_up_day():
    days, _ = seeded_days()
    results = [fm.compare_day(day, f"made-up {i}") for i, day in enumerate(days)]
    assert [r.verdict for r in results] == [fm.MATCH] * SAMPLE_DAYS, [
        (r.label, r.problems, r.could_not_run) for r in results if r.verdict != fm.MATCH
    ]
    assert all(r.functions_only_verdict == fm.MATCH for r in results)
    assert all(r.max_set_difference <= fm.SET_TOLERANCE for r in results)
    finite = [s for r in results for s in r.shapes if np.isfinite(s.exact_scale)]
    assert len(finite) > SAMPLE_DAYS  # most shapes have a limit, so the scales were compared
    assert any(s.exact_scale > 0.01 for s in finite)  # and not only at zero
    assert all(s.rounds > 0 for s in finite)  # the optimisation ran on each of them


@pytest.mark.parametrize("name", sorted(NAMED))
def test_flexitroid_matches_on_named_fleets(name):
    result = fm.compare_day(NAMED[name], name)
    assert result.verdict == fm.MATCH, (result.problems, result.could_not_run)
    assert len(result.shapes) == fm.SHAPES
    for shape in result.shapes:
        assert shape.lower <= shape.exact_scale * (1 + fm.SCALE_TOLERANCE) + fm.SCALE_TOLERANCE
        assert shape.upper >= shape.exact_scale * (1 - fm.SCALE_TOLERANCE) - fm.SCALE_TOLERANCE


def test_the_first_shape_on_one_car_is_the_scale_by_hand():
    # One car, 7.2 kW, 20 kWh, leaving after hour 5; service hours 1 to 4, hour 0 and hour 5
    # free. Every service hour ranges over [0, 7.2]; three hours hold at most 20 kWh (not 21.6);
    # the four hours hold at most 20 kWh and at least 20 - 2 * 7.2 = 5.6 kWh.
    result = fm.compare_day(SINGLE_CAR, "single car")
    shape = result.shapes[0]
    assert shape.w_up == shape.w_down == pytest.approx((7.2,) * 4)
    centre = np.array(shape.centre)
    # By hand, with widths 7.2 in every hour: the faces that can bind are single hours (both
    # sides), any three hours (upper side) and all four hours (both sides). Pairs, and the lower
    # side of three hours (0 kWh), give averages of the single-hour terms.
    by_hand = min(
        *((7.2 - centre) / 7.2),
        *(centre / 7.2),
        *((20.0 - centre.sum() + centre[k]) / 21.6 for k in range(4)),
        (20.0 - centre.sum()) / 28.8,
        (centre.sum() - 5.6) / 28.8,
    )
    assert shape.exact_scale == pytest.approx(by_hand, rel=1e-12)
    assert shape.lower == pytest.approx(by_hand, rel=1e-6)


# --- what must not match --------------------------------------------------------------------------


def scaled_upper(day: Day) -> exact.SetFunctions:
    found = exact.day_functions(day)
    return exact.SetFunctions(found.n_service, found.lower, found.upper * 1.01)


def one_car_dropped(day: Day) -> exact.SetFunctions:
    """The car with the most room left out (a car with zero room, or one gone before hour 1,
    would change nothing on the service hours)."""
    largest = max(range(len(day.cars)), key=lambda i: day.cars[i].room)
    cars = [car for i, car in enumerate(day.one_way_cars()) if i != largest]
    return exact.subset_functions(cars, day.n_service)


@pytest.mark.parametrize("ruler", [scaled_upper, one_car_dropped], ids=["b x 1.01", "car dropped"])
@pytest.mark.parametrize("name", ["zero room", "leaves in window", "eight service hours"])
def test_a_wrong_ruler_differs(name, ruler):
    result = fm.compare_day(NAMED[name], name, ruler=ruler)
    assert result.verdict == fm.DIFFERS
    assert result.functions_only_verdict == fm.DIFFERS
    assert any("differs from the project's" in p for p in result.problems)


def test_a_wrong_optimisation_differs_where_the_functions_alone_agree(monkeypatch):
    """The strict reading catches what the functions-only reading cannot: here flexitroid's
    greedy vertices are stretched by 2 % only when the cutting planes ask for them."""
    real = fm.greedy_service_vertex
    calls = {"n": 0}

    def stretched(aggregate, n_hours, n_service, cost):
        vertex = real(aggregate, n_hours, n_service, cost)
        calls["n"] += 1
        return vertex if calls["n"] <= 2 * n_service else vertex * 1.02

    monkeypatch.setattr(fm, "greedy_service_vertex", stretched)
    result = fm.compare_day(EIGHT_HOURS, "eight service hours")
    assert result.functions_only_verdict == fm.MATCH
    assert result.verdict == fm.DIFFERS
    assert any("optimisation bounds the scale" in p for p in result.problems)


def test_flexitroid_raising_could_not_run(monkeypatch):
    def broken(day):
        raise AssertionError("Invalid arrival/departure times")

    monkeypatch.setattr(fm, "flexitroid_aggregates", broken)
    result = fm.compare_day(ZERO_ROOM, "zero room")
    assert result.verdict == fm.COULD_NOT_RUN
    assert "flexitroid raised AssertionError" in result.could_not_run[0]


def test_a_programme_timing_out_could_not_run(monkeypatch):
    def timeout(cost, **kwargs):
        return LpResult("timeout", None, None, 301.0, kwargs["time_limit"], 1, "made up timeout")

    monkeypatch.setattr(fm, "solve_lp", timeout)
    result = fm.compare_day(LEAVES_IN_WINDOW, "leaves in window")
    assert result.verdict == fm.COULD_NOT_RUN
    assert result.functions_only_verdict == fm.MATCH
    assert all("ended 'timeout'" in reason for reason in result.could_not_run)
    assert len(result.could_not_run) == fm.SHAPES


def test_every_programme_has_the_time_limit(monkeypatch):
    limits = []
    real = fm.solve_lp

    def spy(cost, **kwargs):
        limits.append(kwargs["time_limit"])
        return real(cost, **kwargs)

    monkeypatch.setattr(fm, "solve_lp", spy)
    assert fm.compare_day(ZERO_ROOM, "zero room").verdict == fm.MATCH
    assert limits and set(limits) == {fm.OTHER_SECONDS}


def test_a_corner_that_does_not_close_could_not_run(monkeypatch):
    monkeypatch.setattr(fm, "MAX_ROUNDS", 1)
    result = fm.compare_day(EIGHT_HOURS, "eight service hours")
    assert result.verdict == fm.COULD_NOT_RUN
    assert any("did not close within 1 rounds" in r for r in result.could_not_run)


def test_an_empty_fleet_could_not_run():
    result = fm.compare_day(made_day([]), "empty")
    assert result.verdict == fm.COULD_NOT_RUN
    assert result.functions_only_verdict == fm.COULD_NOT_RUN


def test_the_shapes_depend_on_the_label_only():
    first = fm.compare_day(ZERO_ROOM, "one label")
    again = fm.compare_day(ZERO_ROOM, "one label")
    other = fm.compare_day(ZERO_ROOM, "another label")
    assert [s.w_up for s in first.shapes] == [s.w_up for s in again.shapes]
    assert [s.w_up for s in first.shapes][1:] != [s.w_up for s in other.shapes][1:]


def test_a_fixed_hour_gets_width_zero():
    # Two cars that must charge at full power in every hour they are present: every hour is
    # fixed, so every width is 0 and neither side limits the scale.
    fixed = made_day([TrueCar(6, 3.0, 18.0), TrueCar(6, 2.0, 12.0)])
    result = fm.compare_day(fixed, "fixed")
    assert result.verdict == fm.MATCH
    for shape in result.shapes:
        assert shape.w_up == shape.w_down == (0.0,) * 4
        assert shape.exact_scale == shape.lower == shape.upper == float("inf")
