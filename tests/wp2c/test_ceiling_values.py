"""The values of one day, worked by hand in the test text, and how a missing or empty day is kept.

Hand-worked days (rho = 0 throughout, so a car's protected energy is its room):

H. The hour-0 day. One car present in hours 0 and 1 (departure 2), rate 1 kW, room 1 kWh; one
   service hour (hour 1). It must end with exactly 1 kWh, so hour 0 and hour 1 together draw 1 kWh.
   Hindsight chooses hour 0 after seeing the request: hour 1 can be anything in [0, 1], so the
   hourly ceiling is b(S) - p(S) = 1 - 0 = 1 kWh, and the block ceiling (half-width 0.5 about 0.5)
   is 2 * 0.5 * 1 = 1 kWh. A causal rule fixes hour 0 first; then hour 1 must draw exactly 1 minus
   that, whatever the request, so no width is left: every causal value is 0.
E. A day meeting T4's hypotheses, one car. Departure 3 (hours 0 to 2), rate 1, room 1; one service
   hour, one recovery hour (hour 2): the recovery supply 1 * (3 - 1 - 1) = 1 equals the room, so
   the car can always finish in the recovery hour and hour 1 is free in [0, 1]. Every value is
   1 kWh: the hindsight and causal boxes are the same, 0.5 +- 0.5.
F. The same with two identical cars of departure 4 (hours 0 to 3), two service hours, one
   recovery hour (hour 3), room 1 and rate 1 each. The fleet draws between 0 and 2 kWh in each
   service hour and between 0 and 2 kWh over both (a car stores at most 1 kWh). The box m +- lam
   in both hours needs m_t - lam >= 0, m_t + lam <= 2, m_1 + m_2 + 2 lam <= 2 and
   m_1 + m_2 - 2 lam >= 0: lam = 0.5 with m = (0.5, 0.5), so the value is 2 * 0.5 * 2 = 2 kWh in
   every product, hindsight and causal alike.
"""

import datetime

import pytest

from fleetreserve import ceiling_values
from fleetreserve.ceiling_values import DayTask, compute_day, meets_t4_hypotheses
from fleetreserve.day import Day, TrueCar

DATE = datetime.date(2020, 1, 1)


def _task(n_service, n_recovery, cars, sampled=True):
    day = Day("unit", DATE, n_service, n_recovery, 0.0, tuple(cars))
    return DayTask("unit", "S", DATE.isoformat(), day, sampled)


def _numbers(result):
    return {k: v for k, v in result.values.items() if v is not None}


def test_the_hour_0_day_hindsight_is_1_kwh_and_every_causal_value_is_0():
    result = compute_day(_task(1, 0, [TrueCar(2, 1.0, 1.0)]))
    assert result.error is None
    assert result.values["hourly_hind"] == pytest.approx(1.0)
    assert result.values["hourly_causal"] == pytest.approx(0.0, abs=1e-9)
    assert result.values["hourly_tree"] == pytest.approx(0.0, abs=1e-9)
    assert result.values["block_hind"] == pytest.approx(1.0)
    for key in ("block_caus", "block_aff", "block_diag"):
        assert result.values[key] == pytest.approx(0.0, abs=1e-9), key
    assert not result.t4_hypotheses  # recovery supply 0 < room 1
    assert not result.in_window  # departure 2 is after the last service hour (1)


def test_a_day_meeting_the_hypotheses_has_all_four_block_values_equal_1_kwh():
    task = _task(1, 1, [TrueCar(3, 1.0, 1.0)])
    result = compute_day(task)
    assert meets_t4_hypotheses(task.day)
    assert result.t4_hypotheses
    for key in ("block_hind", "block_caus", "block_aff", "block_diag", "hourly_hind"):
        assert result.values[key] == pytest.approx(1.0), key
    assert result.values["hourly_causal"] == pytest.approx(1.0)


def test_two_cars_over_two_service_hours_have_all_values_2_kwh():
    task = _task(2, 1, [TrueCar(4, 1.0, 1.0), TrueCar(4, 1.0, 1.0)])
    result = compute_day(task)
    assert result.t4_hypotheses
    assert result.n_service == 2
    for key in (
        "block_hind",
        "block_caus",
        "block_aff",
        "block_diag",
        "hourly_hind",
        "hourly_causal",
    ):
        assert result.values[key] == pytest.approx(2.0), key


def test_the_hypotheses_need_no_in_window_departure_and_enough_recovery_for_every_car():
    base = _task(2, 1, [TrueCar(4, 1.0, 1.0), TrueCar(4, 1.0, 1.0)]).day
    assert meets_t4_hypotheses(base)
    # a car with no room needs no recovery, yet leaving by the last service hour still rules the
    # day out
    in_window = _task(2, 1, [TrueCar(4, 1.0, 1.0), TrueCar(2, 1.0, 0.0)]).day
    assert not meets_t4_hypotheses(in_window)
    short = _task(2, 1, [TrueCar(4, 1.0, 1.0), TrueCar(4, 1.0, 1.5)]).day
    assert not meets_t4_hypotheses(short)  # 1 * (4 - 1 - 2) = 1 < 1.5
    assert not meets_t4_hypotheses(Day("unit", DATE, 2, 0, 0.0, ()))  # an empty fleet


def test_a_day_not_in_the_sample_has_only_the_hourly_values():
    result = compute_day(_task(1, 1, [TrueCar(3, 1.0, 1.0)], sampled=False))
    assert set(_numbers(result)) == {"hourly_hind", "hourly_causal"}
    assert result.outcomes["block_caus"] == "not solved"
    assert result.outcomes["hourly_tree"] == "not solved"


def test_an_empty_fleet_has_every_value_0_without_a_solve():
    result = compute_day(_task(2, 0, [], sampled=True))
    assert result.cars == 0
    assert all(v == 0.0 for v in result.values.values())
    assert set(result.outcomes.values()) == {"empty fleet"}
    assert not result.t4_hypotheses
    assert result.seconds == {}


def test_a_solve_that_times_out_is_missing_not_zero(monkeypatch):
    real = ceiling_values.block_values

    def timed_out(functions, rows):
        found = real(functions, rows)
        found["block_caus"] = (None, "timeout", 30.0)
        found["block_aff"] = (None, "error", 1.0)
        return found

    monkeypatch.setattr(ceiling_values, "block_values", timed_out)
    result = compute_day(_task(1, 1, [TrueCar(3, 1.0, 1.0)]))
    assert result.values["block_caus"] is None
    assert result.outcomes["block_caus"] == "timeout"
    assert result.values["block_aff"] is None
    assert result.outcomes["block_aff"] == "error"
    assert result.values["block_hind"] == pytest.approx(1.0)


def test_a_day_whose_sets_cannot_be_built_gives_an_error_not_a_crash(monkeypatch):
    def refuse(day):
        raise ValueError("a stored-energy car has an empty set")

    monkeypatch.setattr(ceiling_values.exact, "day_functions", refuse)
    result = compute_day(_task(1, 1, [TrueCar(3, 1.0, 1.0)]))
    assert result.error == "a stored-energy car has an empty set"
    assert all(v is None for v in result.values.values())


def test_the_structures_asked_for_are_the_causal_and_the_diagonal_gains(monkeypatch):
    asked = []
    real = ceiling_values.certify

    def spy(rows, product, structure, **keywords):
        asked.append((product.kind, structure, keywords["shape_rule"].name))
        return real(rows, product, structure, **keywords)

    monkeypatch.setattr(ceiling_values, "certify", spy)
    compute_day(_task(1, 1, [TrueCar(3, 1.0, 1.0)]))
    assert sorted(asked) == [
        ("block", "causal", "any optimal vertex"),
        ("block", "diagonal", "any optimal vertex"),
        ("hourly", "causal", "any optimal vertex"),
    ]


def test_a_solver_result_without_a_value_is_missing_in_the_record():
    class Solved:
        value = None
        outcome = "timeout"
        seconds = 30.0

    assert ceiling_values._read(Solved()) == (None, "timeout", 30.0)
