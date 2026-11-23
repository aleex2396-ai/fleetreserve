"""The fixed plan's closed form against replay when a car cannot take its decision-hour plan: a
hand case of two cars and two service hours, worked by hand below (the pass mark of plan 13
WP11, "closed-form worst case reproduced", and reading K3 of `corner_run`).

The day (`DAY`): two service hours, two recovery hours, rho 0. Car A truly leaves in hour 5
(present in hours 0 to 4), rate 1 kW, room 1.88 kWh; its forecast says rate 1.5 kW and cap
2.5 kWh, both above the truth, so A is outside E'_A (plan section 5, T2 (c1)). Car B truly
leaves in hour 5, rate 2 kW, room 3 kWh, forecast equal to the truth. The box (`POLICY`):
m = (1, 1), w = (0.5, 0.5). The plan: A draws 1.5 in hour 0 (its forecast rate) and
`0.25 + 0.25 beta zeta_t` in hours t = 1, 2; B draws 0 in hour 0 and `0.75 + 0.25 beta zeta_t`
in hours 1, 2, so the two meet `1 + 0.5 beta zeta_t`. On the forecast sets this plan holds at
every scale up to 1 (A's stored energy at most 1.5 + 0.5 + 0.5 = 2.5 = its forecast cap).

Replay (plan section 5: "cars capped at what they can take"; `replay.run_plan`). Hour 0 carries
no request, so it cannot fail; A draws min(1.5, rate 1, room 1.88) = 1.0 on every corner and at
every scale (hour 0 has no gain), half a kWh less than planned, and B draws 0. From then on A's
stored energy is 0.5 below the plan's: after hour 1 it is `1.25 + 0.25 beta zeta_1` <= 1.5 <
1.88, so hour 1 is met on every corner; after hour 2 it is `1.5 + 0.25 beta (zeta_1 + zeta_2)`,
above 1.88 only on the corner (+1, +1) and only for beta > 0.76, where A draws
1.88 - (1.25 + 0.25 beta) = 0.63 - 0.25 beta in hour 2 instead of 0.25 + 0.25 beta, a miss of
0.5 beta - 0.38 kWh (0.02 at beta 0.8, 0.12 at 1), above the 1e-4 kWh of plan section 1. On the
grid of scales (`dial_splits.GRID`, 0.05 to 1): no corner fails at 0.05 to 0.7; at 0.8 and 1 the
corner (+1, +1) fails in hour 2 (slot loss 1/2).

The closed form (plan section 6, T6: "t_min the first hour whose threshold s_dt (min over its
true-set rows of (b - a g) / |a G|_1) is below beta"), on the true rows with each car's
decision-hour draw (`fixed_plan_loss.decision_hour_draws`: A 1.0, B 0.0) in place of its planned
hour-0 power (the owner's decision of 16 November 2026):
- A's stored energy after hour 1 is `1.0 + 0.25 + 0.25 beta zeta_1` against its room 1.88:
  (1.88 - 1.25) / 0.25 = 2.52; after hour 2, `1.0 + 0.5 + 0.25 beta (zeta_1 + zeta_2)`:
  (1.88 - 1.5) / 0.5 = 0.76;
- A's power at least 0 gives 0.25 / 0.25 = 1 and at most its rate (1 - 0.25) / 0.25 = 3 in hours
  1 and 2; B's power gives 0.75 / 0.25 = 3 and (2 - 0.75) / 0.25 = 5, its stored energy
  (3 - 0.75) / 0.25 = 9 and (3 - 1.5) / 0.5 = 3; A's hour-0 rows hold under its draw.
So s_1 = min(1, 2.52, 3, 3, 5, 9) = 1 and s_2 = min(1, 0.76, 3, 3, 5, 3) = 0.76: no failing hour up
to 0.76 and hour 2 above (to 1), which is replay's on every scale of the grid, so
`corner_search.closed_form_check` agrees strictly on all ten. The draw is derived for one-way
rows: on the day's two-way rows (budget 1 kWh), or on one-way rows edited to let a car discharge,
the draw, the thresholds and the driver-loss bound refuse.

With the plan's own hour-0 power 1.5 for A (the closed form before the decision) the same rows
gave (1.88 - 1.75) / 0.25 = 0.52 and (1.88 - 2) / 0.5 = -0.24: s = (0.52, -0.24), so the closed
form named hour 2 at 0.05 to 0.5 and hour 1 at 0.6 to 1, an earlier failing hour than replay on
every scale (the difference then named "decision-hour cap").

On made-up days (`corner_fake.offer_days(MADE_UP_DAYS)`, the sample in which the corner search
first found that difference), every car's decision-hour draw is what replay draws in hour 0 on
every corner, `CAPPED_DAYS` days have a car capped in hour 0, and at every scale of the grid every
(day, scale) pair agrees by reading K3: strictly, or within the failure tolerance on
`WITHIN_TOLERANCE` pairs. With the planned hour-0 power the closed form disagreed on 12 of those
days (111 pairs). The corner search's written readings K3 and K4 say that the closed form and the
driver-loss bound count the draw, and K3 no longer names the removed "decision-hour cap". The
summary's definition of the "car short of its plan" count says that the bound counts the draw and
holds at every scale with no failing hour, and that the flag counts the decision hour too; the
corner search's text says why that leaves the count over the pairs above the bound unchanged.
"""

import dataclasses
import datetime

import numpy as np
import pytest
from corner_fake import hand, offer_days

from fleetreserve import corner_run, corner_search, corner_summary, dial_splits, fixed_plan_loss
from fleetreserve.carsets import true_rows
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.policy import AffinePolicy

ROOM_A = 1.88
DAY = Day(
    "hand",
    datetime.date(2020, 1, 1),
    2,
    2,
    0.0,
    (TrueCar(5, 1.0, ROOM_A), TrueCar(5, 2.0, 3.0)),
    (ForecastCar(5.0, 1.5, 2.5, ROOM_A), ForecastCar(5.0, 2.0, 3.0, 3.0)),
    0.1,
)
POLICY = AffinePolicy(
    [1.0, 1.0],
    [0.5, 0.5],
    [[1.5, 0.25, 0.25], [0.0, 0.75, 0.75]],
    [[[0, 0], [0.25, 0], [0, 0.25]], [[0, 0], [0.25, 0], [0, 0.25]]],
)
PREPARED = hand(DAY, POLICY)
GRID = dial_splits.GRID
# by hand (module text), per scale of the grid 0.05, 0.1, 0.2, ..., 0.8, 1
REPLAY_HOURS = [None if beta <= 0.76 else 2 for beta in GRID]
MADE_UP_DAYS = 240
CAPPED_DAYS = 18
WITHIN_TOLERANCE = 8


def test_the_grid_is_the_ten_scales_the_derivation_reads():
    assert GRID == (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    assert REPLAY_HOURS == [None] * 8 + [2] * 2


def test_the_decision_hour_draws_are_what_replay_draws_in_hour_0():
    draws = fixed_plan_loss.decision_hour_draws(POLICY, true_rows(DAY))
    assert draws.tolist() == [1.0, 0.0]  # A: min(1.5, rate 1, room 1.88); B: its plan, 0
    for beta in GRID:
        for run in corner_search.fixed_plan_corners(PREPARED, beta):
            assert run.planned[:, 0].tolist() == [1.5, 0.0]
            assert run.draws[:, 0].tolist() == draws.tolist()


def test_the_draw_refuses_rows_that_are_not_one_way():
    # two-way rows (a stored-low row, and power-low rows that let a car discharge) would give a
    # discharge interval that the fixed plan's one-way replay never draws
    two_way = true_rows(DAY, budget=1.0)
    assert "stored-low" in two_way.kinds
    for call in (
        lambda: fixed_plan_loss.decision_hour_draws(POLICY, two_way),
        lambda: fixed_plan_loss.hour_thresholds(POLICY, two_way),
        lambda: fixed_plan_loss.driver_loss_bound(POLICY, two_way, DAY, 0.5),
    ):
        with pytest.raises(ValueError, match="one-way rows only"):
            call()
    # the one-way rows with each power-low row read -u_t <= 0.5 (a car may discharge 0.5 kW)
    one_way = true_rows(DAY)
    bounds = np.where(one_way.kinds == "power-low", 0.5, one_way.bounds)
    with pytest.raises(ValueError, match="one-way rows only"):
        fixed_plan_loss.decision_hour_draws(POLICY, dataclasses.replace(one_way, bounds=bounds))


def test_the_closed_form_on_the_true_rows_by_hand():
    thresholds = fixed_plan_loss.hour_thresholds(POLICY, true_rows(DAY))
    assert thresholds == pytest.approx([1.0, 0.76], abs=1e-12)
    assert [fixed_plan_loss.first_failing_hour(thresholds, b) for b in GRID] == REPLAY_HOURS


def test_replay_caps_car_a_in_the_decision_hour_and_fails_later_or_never():
    for beta, hour in zip(GRID, REPLAY_HOURS, strict=True):
        runs = corner_search.fixed_plan_corners(PREPARED, beta)
        for run in runs:
            assert run.planned[:, 0] == pytest.approx([1.5, 0.0], abs=1e-12)
            assert run.draws[:, 0] == pytest.approx([1.0, 0.0], abs=1e-12)
        failures = [run.first_failure for run in runs]
        # corners in the order (-1, -1), (+1, -1), (-1, +1), (+1, +1)
        assert failures == [None, None, None, hour]
        top = runs[3]
        assert top.stored[0, 1] == pytest.approx(1.25 + 0.25 * beta, abs=1e-12)
        if hour is not None:
            assert top.draws[0, 2] == pytest.approx(ROOM_A - (1.25 + 0.25 * beta), abs=1e-12)
            miss = 1.0 + 0.5 * beta - top.delivered[1]
            assert miss == pytest.approx(0.5 * beta - 0.38, abs=1e-12)
            assert top.slot_loss == pytest.approx(0.5)
        else:
            assert max(run.slot_loss for run in runs) == 0.0


def test_the_check_agrees_strictly_on_every_scale():
    checks = corner_search.closed_form_check(PREPARED, GRID)
    assert [c.closed_hour for c in checks] == REPLAY_HOURS
    assert [c.corner_hour for c in checks] == REPLAY_HOURS
    assert [c.closed_slot for c in checks] == pytest.approx([0.0] * 8 + [0.5] * 2)
    assert [c.corner_slot for c in checks] == pytest.approx([0.0] * 8 + [0.5] * 2)
    assert all(c.agreed_strict and c.agreed and c.reason is None for c in checks)


def test_every_pair_of_the_made_up_days_agrees_with_the_draw():
    capped, within, reasons = 0, 0, []
    for k, prepared in enumerate(offer_days(MADE_UP_DAYS)):
        policy = prepared.certificate.policy
        draws = fixed_plan_loss.decision_hour_draws(policy, true_rows(prepared.day))
        for run in corner_search.fixed_plan_corners(prepared, 1.0):
            assert np.array_equal(run.draws[:, 0], draws)
        capped += bool(np.any(draws != policy.g[:, 0]))
        for c in corner_search.closed_form_check(prepared, GRID):
            if not c.agreed:
                reasons.append((k, c.beta, c.reason))
            within += c.reason == corner_search.WITHIN_TOLERANCE
    assert reasons == []
    assert (capped, within) == (CAPPED_DAYS, WITHIN_TOLERANCE)


def test_the_corner_runs_readings_name_the_draw_and_not_the_removed_reason():
    for key in ("K3", "K4"):
        assert "decision-hour draw" in corner_run.READINGS[key]
    assert "decision-hour cap" not in corner_run.READINGS["K3"]
    assert "every-scale bound" in corner_run.READINGS["K4"]
    assert "V1 to V8" in corner_run.READINGS["K4"]


def test_the_summary_says_what_the_bound_and_the_short_flag_count():
    # the bound counts the draw and holds where no hour fails; the flag counts every hour, the
    # decision hour included, which leaves its count over the pairs above the bound as it is with
    # the service hours alone (`corner_search` module text); the summary's own line is pinned in
    # `test_corner_summary.py`
    flag = corner_summary.DEFINITIONS["pass_mark.driver_above_bound_limited"]
    assert "the decision hour included" in flag
    assert "counts each car's decision-hour draw" in flag
    assert "holds at every scale with no failing hour" in flag
    assert not flag.endswith("the bound assumes the plan is delivered")
    doc = " ".join(corner_search.__doc__.split())
    assert "The flag counts every hour, the decision hour included, and is kept so" in doc
