"""Plan sections 5 (T2's check) and 6 (T6): the fixed plan's closed-form loss against replay.

Hand case, the early-departure day of `test_rungs_hand.py` on its true one-way rows. Hour 1:
A's power rows do not depend on the requests and hold (`+inf`); B's `u_1 <= 2` gives
(2 - 0.5) / 0.5 = 3, `-u_1 <= 0` gives 0.5 / 0.5 = 1, its stored-energy row (3 - 0.5) / 0.5 = 5;
so `s_1 = 1`. Hour 2: A has left, its row `u_2 <= 0` is broken at the centre and does not depend on
the requests, so `s_2 = -inf`. Hence `t_min = 2` for `beta <= 1` (slot loss 1/2) and 1 above it
(slot loss 1): at `beta = 1.5` B's planned power `0.5 - 0.75` in hour 1 on `zeta_1 = -1` is
negative and draws nothing, an excess of 0.25 kWh. The driver-loss bound at scale 1 is 0 (each
car's planned energy meets its need row), while replay loses 1/8: A is capped in service hour 2
(it leaves), which the bound does not cover, at a scale with a failing hour.

Sample: `N_DAYS` seeded made-up days with every kind of forecast; on each certified day, at every
scale of `BETAS`, the closed form on the true rows is compared with replay of all `2^n` corner paths
(`fixed_plan_loss.compare_with_replay`). The closed form's hour is never later than replay's (it is
a lower bound), every difference has one of the module's named reasons, and they agree on
`N_EQUAL` of `N_CASES` (day, scale) cases; `REASONS` counts the others. `N_CAPPED_DAYS` of the days
have a car that cannot take its planned hour-0 power (`fixed_plan_loss.decision_hour_draws` below
it); with that draw in the closed form every case of those days agrees (before the owner's
decision of 16 November 2026 the closed form counted the planned power and those days gave the 10
cases then named "decision-hour cap", `N_EQUAL` 470). Replay's worst driver loss exceeds the
summed per-car bound on `N_DRIVER_ABOVE` cases, each at a scale with a closed-form failing hour
(reading R2 of `fixed_plan_loss`: with each car's decision-hour draw counted, the bound holds at
every scale whose slot loss is 0).

Driver sample with hour-0 caps: `LOWER_RATE_DRAWS` seeded draws whose forecasts all overstate the
rate ("lower rate"), so a car can be planned above its true rate in hour 0; on the
`LOWER_RATE_DAYS` days with an offer above 0 (`LOWER_RATE_CAPPED` of them with a car capped in hour
0), at every scale of `LOWER_RATE_BETAS`, replay's worst driver loss is above the bound on
`LOWER_RATE_ABOVE` cases, each at a scale with a closed-form failing hour. With the planned hour-0
power in the bound (before the owner's decision) 139 were above, 3 of them at a scale with no
failing hour (draw 183 at 0.25 and 0.5: bound 0 against replay 0.0307; draw 264 at 0.5).
"""

import random
from collections import Counter

import numpy as np
import pytest
from made_up_offer_days import offer_day
from test_rungs_hand import EARLY, EARLY_POLICY

from fleetreserve.carsets import true_rows
from fleetreserve.fixed_plan_loss import (
    closed_form_slot_loss,
    compare_with_replay,
    decision_hour_draws,
    driver_loss_bound,
    first_failing_hour,
    hour_thresholds,
)

SEED = 23
N_DAYS = 80
BETAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
N_CASES = 480
N_EQUAL = 480
REASONS = {}
N_CAPPED_DAYS = 2
N_DRIVER_ABOVE = 148
LOWER_RATE_SEED = 7
LOWER_RATE_DRAWS = 300
LOWER_RATE_BETAS = (0.25, 0.5, 0.75, 1.0)
LOWER_RATE_DAYS = 267
LOWER_RATE_CAPPED = 4
LOWER_RATE_ABOVE = 130


def test_hand_case_thresholds_and_losses():
    rows = true_rows(EARLY)
    thresholds = hour_thresholds(EARLY_POLICY, rows)
    assert thresholds[0] == pytest.approx(1.0)
    assert thresholds[1] == -np.inf
    assert decision_hour_draws(EARLY_POLICY, rows).tolist() == [0.0, 0.0]  # no car capped
    assert first_failing_hour(thresholds, 1.0) == 2
    assert first_failing_hour(thresholds, 1.5) == 1
    assert closed_form_slot_loss(thresholds, 1.0) == pytest.approx(0.5)
    assert closed_form_slot_loss(thresholds, 1.5) == pytest.approx(1.0)
    assert driver_loss_bound(EARLY_POLICY, rows, EARLY, 1.0) == pytest.approx(0.0)
    one, above = compare_with_replay(EARLY, EARLY_POLICY, rows, (1.0, 1.5))
    assert (one.closed_hour, one.replay_hour, one.reason) == (2, 2, None)
    assert one.replay_driver == pytest.approx(1 / 8)
    assert (above.closed_hour, above.replay_hour, above.reason) == (1, 1, None)


def _capped(policy, rows) -> bool:
    """Some car cannot take its planned hour-0 power."""
    return bool(np.any(decision_hour_draws(policy, rows) != policy.g[:, 0]))


def test_closed_form_against_all_path_replay_on_made_up_days():
    rng = random.Random(SEED)
    cases = equal = driver_above = capped_days = 0
    reasons = Counter()
    for k in range(N_DAYS):
        day, certificate, _ = offer_day(rng, k)
        if certificate.outcome != "optimal":
            continue
        rows = true_rows(day)
        capped = _capped(certificate.policy, rows)
        capped_days += capped
        for c in compare_with_replay(day, certificate.policy, rows, BETAS):
            cases += 1
            assert c.reason not in ("replay earlier", "unexplained"), (day, c)
            if c.reason is None:
                equal += 1
                assert c.replay_slot == pytest.approx(c.closed_slot)
            else:
                reasons[c.reason] += 1
                assert c.replay_slot < c.closed_slot
            if c.replay_driver > c.earlier_driver_bound + 1e-9:
                driver_above += 1
                assert c.closed_hour is not None, (k, c)
    print(
        f"{equal} of {cases} agree; {dict(reasons)}; {capped_days} days capped in hour 0; "
        f"driver above the bound {driver_above}"
    )
    assert (cases, equal, dict(reasons), capped_days, driver_above) == (
        N_CASES,
        N_EQUAL,
        REASONS,
        N_CAPPED_DAYS,
        N_DRIVER_ABOVE,
    )


def test_the_driver_bound_holds_without_a_failing_hour_on_days_capped_in_hour_0():
    rng = random.Random(LOWER_RATE_SEED)
    days = capped_days = above = 0
    for k in range(LOWER_RATE_DRAWS):
        day, certificate, _ = offer_day(rng, k, ("lower rate",))
        if certificate.outcome != "optimal" or certificate.value <= 1e-6:
            continue
        days += 1
        rows = true_rows(day)
        capped_days += _capped(certificate.policy, rows)
        for c in compare_with_replay(day, certificate.policy, rows, LOWER_RATE_BETAS):
            if c.closed_hour is None:
                assert c.closed_slot == 0.0
                assert c.replay_driver <= c.earlier_driver_bound + 1e-9, (k, c)
            above += c.replay_driver > c.earlier_driver_bound + 1e-9
    assert (days, capped_days, above) == (LOWER_RATE_DAYS, LOWER_RATE_CAPPED, LOWER_RATE_ABOVE)
