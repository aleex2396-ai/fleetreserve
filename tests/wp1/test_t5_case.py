"""Plan section 5, T5 unit test: discharging after a failure breaks T5(b).

T5(b) says each car ends with at least its fixed-plan energy under one-way re-solve, and at least
its one-way energy under each two-way rung. The Rungs paragraph gives the rule that keeps it: a
failed policy runs on, with each car capped at what it can take, and a two-way rung zeroes its
planned discharges after its own first failure. This test shows the case where dropping the last
part breaks (b).

One car, three hours (1, 2, 3), one hour each; it leaves at the end of hour 3, so there are no
recovery hours. Its rate is 1 kW both ways. Energies are the energy the car has taken since the
decision time, in kWh; the car can take its room and no more. Its forecast room is 2.0 kWh; its
true room is 1.0 kWh. With a single car, the request in each hour is the car's planned power, so
the car's delivery fails in an hour when it takes less or more than planned by more than 1e-4 kWh
(plan section 1).

- Fixed plan: charge (1.0, 0.0, 0.0). Taken: 1.0 in hour 1 (the true room allows it). No failure.
  Ends at 1.0.
- Two-way plan: charge 1.0, charge 1.0, discharge 0.5, that is (1.0, 1.0, -0.5). On the forecast
  room its energy is 1.0, 2.0, 1.5 after hours 1, 2, 3: never above the forecast room 2.0 and never
  below the fixed plan's 1.0, 1.0, 1.0, so it is a valid two-way plan on the forecast.
  On the true room: hour 1 takes 1.0 (energy 1.0, the room is full); hour 2 plans 1.0 but can take
  0.0, which is the first failure (energy 1.0).
  - Discharge kept after the failure: hour 3 gives 0.5 (energy 1.0 - 0.5 = 0.5). Ends at 0.5,
    below the fixed plan's 1.0: T5(b) breaks.
  - Discharge zeroed after the failure (the plan's rule): hour 3 draws 0.0. Ends at 1.0, at least
    the fixed plan's 1.0: T5(b) holds.

Two parts of the rule are not reached by this case and are not modelled: the car is present in
every hour of the plan (no departed car to set to zero), and its energy never falls below 0, so
the two-way limit B (energy at most B below its level at the decision time) never caps a discharge.
"""

import pytest

FAILURE_KWH = 1e-4  # plan section 1: a delivery missing its request by more than this fails
FORECAST_ROOM = 2.0
TRUE_ROOM = 1.0
FIXED_PLAN = (1.0, 0.0, 0.0)  # kW in hours 1, 2, 3
TWO_WAY_PLAN = (1.0, 1.0, -0.5)


def _run_on(plan, room, zero_discharge_after_failure):
    """Replay one car's hourly plan on its room; return (energy after each hour, first failure).

    Charging is capped at what the car can take (its room minus its energy). A discharge is drawn
    as planned, unless `zero_discharge_after_failure` is set and an earlier hour failed. The first
    failure is the first hour (from 1) whose drawn energy differs from the plan by more than
    `FAILURE_KWH`, or None.
    """
    energy = 0.0
    energies = []
    first_failure = None
    for hour, planned in enumerate(plan, start=1):
        if planned >= 0:
            drawn = min(planned, room - energy)
        elif zero_discharge_after_failure and first_failure is not None:
            drawn = 0.0
        else:
            drawn = planned
        if first_failure is None and abs(drawn - planned) > FAILURE_KWH:
            first_failure = hour
        energy += drawn
        energies.append(energy)
    return energies, first_failure


def test_two_way_plan_is_valid_on_the_forecast():
    fixed, fixed_failure = _run_on(FIXED_PLAN, FORECAST_ROOM, True)
    two_way, two_way_failure = _run_on(TWO_WAY_PLAN, FORECAST_ROOM, True)
    assert fixed == pytest.approx([1.0, 1.0, 1.0])
    assert two_way == pytest.approx([1.0, 2.0, 1.5])
    assert fixed_failure is None and two_way_failure is None
    assert all(e <= FORECAST_ROOM for e in two_way)
    assert all(t >= f for t, f in zip(two_way, fixed, strict=True))


def test_fixed_plan_ends_at_1_0():
    energies, first_failure = _run_on(FIXED_PLAN, TRUE_ROOM, True)
    assert first_failure is None
    assert energies[-1] == pytest.approx(1.0, abs=1e-12)


def test_two_way_plan_discharging_after_failure_ends_at_0_5():
    fixed, _ = _run_on(FIXED_PLAN, TRUE_ROOM, True)
    energies, first_failure = _run_on(TWO_WAY_PLAN, TRUE_ROOM, False)
    assert first_failure == 2
    assert energies == pytest.approx([1.0, 1.0, 0.5], abs=1e-12)
    assert energies[-1] < fixed[-1]  # T5(b) breaks


def test_two_way_plan_zeroing_discharge_after_failure_ends_at_1_0():
    fixed, _ = _run_on(FIXED_PLAN, TRUE_ROOM, True)
    energies, first_failure = _run_on(TWO_WAY_PLAN, TRUE_ROOM, True)
    assert first_failure == 2
    assert energies == pytest.approx([1.0, 1.0, 1.0], abs=1e-12)
    assert energies[-1] >= fixed[-1]  # T5(b) holds
