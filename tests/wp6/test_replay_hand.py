"""Plan sections 1, 5 and 9: running a plan on true cars, its losses and recovery, by hand.

Case 1 (one-way). Three service hours, two recovery hours, rho = 0 (every car needs its room).

- Car A: leaves in hour 2 (present in hours 0 and 1), rate 2 kW, room 3 kWh.
- Car B: leaves in hour 6 (recovery hours 4 and 5), rate 1 kW, room 4 kWh.
- Car C: leaves in hour 4 (no recovery hour), rate 3 kW, room 1 kWh.

Plan (kW, hours 0, 1, 2, 3; no gains): A (1, 1.5, 1, 0); B (0, 0.5, 1.5, -0.5); C (0.5, 0, 0.8, 0).
Requests: the plan's sums in hours 1 to 3, (2.0, 3.3, -0.5).

- Hour 0: A 1, B 0, C 0.5 (no request).
- Hour 1: A 1.5 (room left 2), B 0.5, C 0; delivered 2.0, no failure.
- Hour 2: A has left (0 of 1 planned), B is capped at its rate (1 of 1.5), C at its room (0.5 of
  0.8); delivered 1.5 against 3.3: first failure.
- Hour 3: B's planned -0.5 draws nothing one-way; delivered 0 against -0.5: a failure too.
- Slot loss (3 - 2 + 1) / 3 = 2/3. After hour 3: A 2.5, B 1.5, C 1.0.
- Recovery: A none (left in hour 2); B two hours at 1 kW, room left 2.5, so +2 to 3.5; C none.
- Driver loss: shortfalls A 0.5, B 0.5, C 0 of protected energy 3 + 4 + 1 = 8: 1/8.

Case 2 (two-way, budget 1 kWh). One car, three service hours, no recovery, rate 1 kW, room
2 kWh, rho = 0. Plan (0, -0.5, -1, 1), requests (-0.5, -1, 1).

- Hour 1: discharge 0.5 (stored -0.5).
- Hour 2: planned -1, the budget allows -(1 - 0.5) = -0.5 (stored -1); delivered -0.5 against -1:
  first failure, slot loss 2/3.
- Hour 3: a charge of 1, rate 1, room left 3: 1 (stored 0), delivered 1, no failure.
- Ends at 0; driver loss 2 / 2 = 1.

Case 3, the plan's T5 unit case (section 5): one car, three service hours, no recovery, rate 1 kW,
forecast room 2.0 kWh, true room 1.0 kWh. Each plan's requests are its own powers (one car).
Fixed plan (0, 1, 0, 0): no failure, ends at 1.0. Two-way plan (0, 1, 1, -0.5): hour 1 takes 1.0
(full); hour 2 can take nothing (first failure); hour 3: with discharges zeroed after the failure
(the plan's rule) nothing is drawn and it ends at 1.0, at least the fixed plan's 1.0; with the
rule switched off (`discharges_carried` replaced) it gives 0.5 and ends at 0.5, so T5(b) breaks.
"""

import datetime

import numpy as np
import pytest

from fleetreserve import replay, rungs
from fleetreserve.day import Day, TrueCar
from fleetreserve.plans import Plan

DATE = datetime.date(2020, 1, 1)
TOLERANCE = 1e-12


def _plan(rows, two_way=False):
    constant = np.array(rows, dtype=float)
    return Plan(constant, np.zeros((*constant.shape, constant.shape[1] - 1)), two_way)


def _run(day, plan, requests, budget=None):
    return replay.run_plan(day, plan, np.zeros(day.n_service), requests, budget=budget)


def test_case_1_one_way_runner_losses_and_recovery():
    cars = (TrueCar(2, 2.0, 3.0), TrueCar(6, 1.0, 4.0), TrueCar(4, 3.0, 1.0))
    day = Day("hand", DATE, 3, 2, 0.0, cars)
    plan = _plan([[1, 1.5, 1, 0], [0, 0.5, 1.5, -0.5], [0.5, 0, 0.8, 0]])
    run = _run(day, plan, [2.0, 3.3, -0.5])
    np.testing.assert_allclose(
        run.draws, [[1, 1.5, 0, 0], [0, 0.5, 1, 0], [0.5, 0, 0.5, 0]], atol=TOLERANCE
    )
    np.testing.assert_allclose(run.delivered, [2.0, 1.5, 0.0], atol=TOLERANCE)
    assert list(run.failed) == [False, True, True]
    assert run.first_failure == 2
    assert run.slot_loss == pytest.approx(2 / 3)
    np.testing.assert_allclose(run.stored[:, -1], [2.5, 1.5, 1.0], atol=TOLERANCE)
    np.testing.assert_allclose(run.final, [2.5, 3.5, 1.0], atol=TOLERANCE)
    assert run.driver_loss == pytest.approx(1 / 8)


def test_case_2_two_way_budget_caps_a_discharge():
    day = Day("hand", DATE, 3, 0, 0.0, (TrueCar(4, 1.0, 2.0),))
    run = _run(day, _plan([[0, -0.5, -1, 1]], two_way=True), [-0.5, -1, 1], budget=1.0)
    np.testing.assert_allclose(run.draws, [[0, -0.5, -0.5, 1]], atol=TOLERANCE)
    np.testing.assert_allclose(run.stored, [[0, -0.5, -1, 0]], atol=TOLERANCE)
    assert run.first_failure == 2
    assert run.slot_loss == pytest.approx(2 / 3)
    assert run.final == pytest.approx([0.0])
    assert run.driver_loss == pytest.approx(1.0)


def test_no_failure_and_no_protected_energy_give_zero_losses():
    day = Day("hand", DATE, 2, 0, 0.0, (TrueCar(3, 1.0, 0.0),))
    run = _run(day, _plan([[0, 0, 0]]), [0.0, 0.0])
    assert run.first_failure is None
    assert run.slot_loss == 0.0
    assert run.driver_loss == 0.0


def test_a_miss_of_the_failure_tolerance_is_not_a_failure():
    day = Day("hand", DATE, 1, 0, 0.0, (TrueCar(2, 1.0, 1.0),))
    assert _run(day, _plan([[0, 0.5]]), [0.5 + 0.9e-4]).first_failure is None
    assert _run(day, _plan([[0, 0.5]]), [0.5 + 1.1e-4]).first_failure == 1


UNIT_DAY = Day("unit", DATE, 3, 0, 0.0, (TrueCar(4, 1.0, 1.0),))
FIXED = _plan([[0, 1, 0, 0]])
TWO_WAY = _plan([[0, 1, 1, -0.5]], two_way=True)


def test_t5_unit_case_two_way_plan_is_valid_on_the_forecast_room():
    forecast_day = Day("unit", DATE, 3, 0, 0.0, (TrueCar(4, 1.0, 2.0),))
    run = _run(forecast_day, TWO_WAY, [1, 1, -0.5], budget=5.0)
    assert run.first_failure is None
    np.testing.assert_allclose(run.stored, [[0, 1, 2, 1.5]], atol=TOLERANCE)


def test_t5_unit_case_discharges_zeroed_keep_t5b():
    fixed = _run(UNIT_DAY, FIXED, [1, 0, 0])
    two_way = _run(UNIT_DAY, TWO_WAY, [1, 1, -0.5], budget=5.0)
    assert fixed.first_failure is None
    assert fixed.final == pytest.approx([1.0])
    assert two_way.first_failure == 2
    np.testing.assert_allclose(two_way.draws, [[0, 1, 0, 0]], atol=TOLERANCE)
    assert two_way.final == pytest.approx([1.0])
    assert two_way.final[0] >= fixed.final[0] - TOLERANCE


def test_t5_unit_case_discharges_not_zeroed_break_t5b(monkeypatch):
    def keep_discharging(plan, failed_before):
        return plan.two_way

    monkeypatch.setattr(replay, "discharges_carried", keep_discharging)
    fixed = _run(UNIT_DAY, FIXED, [1, 0, 0])
    two_way = _run(UNIT_DAY, TWO_WAY, [1, 1, -0.5], budget=5.0)
    assert two_way.first_failure == 2
    np.testing.assert_allclose(two_way.draws, [[0, 1, 0, -0.5]], atol=TOLERANCE)
    assert two_way.final == pytest.approx([0.5])
    assert two_way.final[0] < fixed.final[0]  # T5(b) breaks
    assert rungs.replay.discharges_carried is keep_discharging  # the rungs use the same rule
