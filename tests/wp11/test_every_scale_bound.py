"""The every-scale driver-loss bound of the fixed plan (`fixed_plan_loss`, reading R3), on hand days
and seeded made-up days.

Per car on the drawn policy: `C(beta)` the service hours whose power-high or power-low row has a
threshold below `beta`; the car holds at least `L`, the larger of the by-hour bound
`d_0 + sum_{t not in C} g_t - beta |sum_{t not in C} G_t|_1 + sum_{t in C} clip(g_t - beta |G_t|_1,
0, c_t)` and the by-corner bound `d_0 + min over corners of sum_t min(p_t, c_t)`; its term is
`max(0, N - L)`, `N` its protected energy less its recovery supply. A car with `C` empty keeps the
term of today's bound (`driver_loss_bound`).

Hand days, worked by hand:

1. `CAP` (`test_corner_hand.py`): one car, rate 1, room 2, leaves after the one service hour, no
   recovery, rho = 0; plan (1.5, 0.5 + 0.5 beta zeta_1). It draws 1 in hour 0 (its rate), so
   `d_0 = 1`, `N = 2`; hour 1's power rows give (1 - 0.5) / 0.5 = 1 and 0.5 / 0.5 = 1. At scale
   0.5 and 1, `C` is empty: the bound is today's, (2 - 1.5 + 0.5 beta) / 2 = 0.375 and 0.5. At 1.2
   hour 1 is in `C`: by hour `1 + clip(0.5 - 0.6, 0, 1) = 1`, by corner `1 + min(0.5 - 0.6, 1) =
   0.9`, so `L = 1` and the bound is (2 - 1) / 2 = 0.5 (today's 0.55); at 2, by hour 1, by corner
   0.5: 0.5 (today's 0.75). Replay's worst (zeta_1 = -1, a plan of at most 0 draws nothing): 0.375,
   0.5, 0.5, 0.5.
2. `EARLY` (`test_rungs_hand.py`): car A (rate 1, room 1) leaves in hour 2 with 0.5 kW planned in
   it and no gain, so its row `u_2 <= 0` is broken at the centre (`-inf`) and hour 2 is in `C` at
   every scale: `L_A = 0 + 0.5 + clip(0.5, 0, 0) = 0.5`, `N_A = 1`, term 0.5; car B's recovery
   supply 2 x 2 = 4 exceeds its need 3, term 0. Bound 0.5 / 4 = 1/8 at every scale, today's 0 up to
   scale 2; replay's worst 1/8.
3. `SWAP` (new): one car, rate 1, room 1.6, leaves after the two service hours, no recovery,
   rho = 0; plan (0, 0.8 + 0.5 beta zeta_1, 0.8 - 0.5 beta zeta_1): what it takes more in hour 1 it
   takes less in hour 2, so its planned total 1.6 does not depend on the path and today's bound is 0
   at every scale. Its power rows give (1 - 0.8) / 0.5 = 0.4 (rate) and 0.8 / 0.5 = 1.6 (power at
   least 0) in both hours (the box is not met: the driver loss of a one-way plan does not depend
   on the requests). At 0.4 `C` is empty: 0. At 0.5, by hour 0.55 + 0.55 = 1.1, by corner
   (zeta_1 = +1) 1 + 0.55 = 1.55: (1.6 - 1.55) / 1.6 = 0.03125. At 1, by hour 0.6, by corner
   1 + 0.3: 0.3 / 1.6 = 0.1875. At 2, by hour 0 + 0 = 0, by corner (zeta_1 = +1) 1 + min(-0.2, 1) =
   0.8: 0.5. Replay's worst: 0, 0.03125, 0.1875 and, at 2, 0.375 (the rate caps hour 1 at 1 and
   the plan of -0.2 in hour 2 draws nothing, which the by-corner bound counts as -0.2 and the
   by-hour bound as 0 in both hours; the car's exact worst case is 0.375).

Made-up days (`corner_fake.offer_days(MADE_UP_DAYS)` at every scale of `dial_splits.GRID`, and the
`LOWER_RATE_DRAWS` "lower rate" draws of seed `LOWER_RATE_SEED` at `LOWER_RATE_BETAS`, as in
`tests/wp6/test_closed_form.py`): replay's driver loss on every corner path and on `INTERIOR`
seeded paths inside the box is at most the bound on every pair (`TOLERANCE`), including the
`MADE_UP_ABOVE_TODAY` and `LOWER_RATE_ABOVE_TODAY` pairs where it is above today's bound; the bound
equals today's bit for bit on every pair with no closed-form failing hour; on a fine grid of scales
and at each power-row threshold it never falls as the scale rises.
"""

import datetime
import random

import numpy as np
import pytest
from corner_fake import offer_days
from made_up_offer_days import offer_day
from test_corner_hand import CAP
from test_rungs_hand import EARLY, EARLY_POLICY

from fleetreserve import dial_splits
from fleetreserve.carsets import true_rows
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.fixed_plan_loss import (
    compare_with_replay,
    driver_loss_bound,
    every_scale_driver_bound,
    service_hours,
)
from fleetreserve.plans import Plan
from fleetreserve.policy import AffinePolicy
from fleetreserve.replay import run_plan

TOLERANCE = 1e-9
MADE_UP_DAYS = 240
MADE_UP_ABOVE_TODAY = 840
LOWER_RATE_SEED = 7
LOWER_RATE_DRAWS = 300
LOWER_RATE_BETAS = (0.25, 0.5, 0.75, 1.0)
LOWER_RATE_DAYS = 267
LOWER_RATE_ABOVE_TODAY = 130
INTERIOR = 8
FINE = np.round(np.arange(0.0, 2.55, 0.05), 2)

SWAP_DAY = Day(
    "hand",
    datetime.date(2020, 1, 1),
    2,
    0,
    0.0,
    (TrueCar(3, 1.0, 1.6),),
    (ForecastCar(3.0, 1.0, 1.6, 1.6),),
    0.1,
)
SWAP_POLICY = AffinePolicy(
    [0.8, 0.8], [0.5, 0.5], [[0.0, 0.8, 0.8]], [[[0, 0], [0.5, 0], [-0.5, 0]]]
)


def _bounds(day, policy, betas):
    rows = true_rows(day)
    today = [driver_loss_bound(policy, rows, day, b) for b in betas]
    every = [every_scale_driver_bound(policy, rows, day, b) for b in betas]
    replay = [c.replay_driver for c in compare_with_replay(day, policy, rows, betas)]
    return today, every, replay


# --- 1 to 3. hand days ---------------------------------------------------------------------------


def test_the_decision_hour_cap_by_hand():
    today, every, replay = _bounds(CAP.day, CAP.certificate.policy, (0.5, 1.0, 1.2, 2.0))
    assert today == pytest.approx([0.375, 0.5, 0.55, 0.75])
    assert every == pytest.approx([0.375, 0.5, 0.5, 0.5])
    assert every[:2] == today[:2]  # no power row broken: today's bound, bit for bit
    assert replay == pytest.approx([0.375, 0.5, 0.5, 0.5])


def test_the_early_departure_by_hand():
    today, every, replay = _bounds(EARLY, EARLY_POLICY, (0.5, 1.0, 1.5, 2.0))
    assert today == [0.0] * 4
    assert every == pytest.approx([1 / 8] * 4)
    assert replay[1] == pytest.approx(1 / 8)
    assert all(r <= e + TOLERANCE for r, e in zip(replay, every, strict=True))


def test_the_offsetting_gains_by_hand():
    rows = true_rows(SWAP_DAY)
    hours = service_hours(SWAP_POLICY, rows)
    assert hours.thresholds[0].tolist() == pytest.approx([0.4, 0.4])
    assert hours.cap.tolist() == [[1.0, 1.0]] and hours.short.tolist() == [1.6]
    today, every, replay = _bounds(SWAP_DAY, SWAP_POLICY, (0.4, 0.5, 1.0, 2.0))
    assert today == pytest.approx([0.0] * 4, abs=1e-15)
    assert every == pytest.approx([0.0, 0.03125, 0.1875, 0.5])
    assert every[0] == today[0]
    assert replay == pytest.approx([0.0, 0.03125, 0.1875, 0.375])


def test_the_bound_refuses_rows_it_is_not_derived_for():
    rows = true_rows(EARLY, budget=1.0)  # two-way rows
    with pytest.raises(ValueError, match="one-way"):
        service_hours(EARLY_POLICY, rows)


# --- the made-up days ----------------------------------------------------------------------------


def _interior_worst(day, policy, beta, rng) -> float:
    plan = Plan.from_policy(policy, beta)
    worst = 0.0
    for _ in range(INTERIOR):
        zeta = rng.uniform(-1.0, 1.0, day.n_service)
        worst = max(worst, run_plan(day, plan, zeta, policy.request(zeta, beta)).driver_loss)
    return worst


def _scan(items, betas):
    """Per pair: (closed-form failing hour, today's bound, the every-scale bound, replay's worst
    over the corner paths, replay's worst over the interior paths)."""
    rng = np.random.default_rng(20261116)
    out = []
    for day, policy in items:
        rows = true_rows(day)
        for c in compare_with_replay(day, policy, rows, betas):
            every = every_scale_driver_bound(policy, rows, day, c.beta)
            inside = _interior_worst(day, policy, c.beta, rng)
            out.append((c.closed_hour, c.earlier_driver_bound, every, c.replay_driver, inside))
    return out


@pytest.fixture(scope="module")
def made_up():
    items = [(p.day, p.certificate.policy) for p in offer_days(MADE_UP_DAYS)]
    return items, _scan(items, dial_splits.GRID)


@pytest.fixture(scope="module")
def lower_rate():
    rng = random.Random(LOWER_RATE_SEED)
    items = []
    for k in range(LOWER_RATE_DRAWS):
        day, certificate, _ = offer_day(rng, k, ("lower rate",))
        if certificate.outcome == "optimal" and certificate.value > 1e-6:
            items.append((day, certificate.policy))
    assert len(items) == LOWER_RATE_DAYS
    return items, _scan(items, LOWER_RATE_BETAS)


SAMPLES = [("made_up", MADE_UP_ABOVE_TODAY), ("lower_rate", LOWER_RATE_ABOVE_TODAY)]


@pytest.mark.parametrize("sample, above_today", SAMPLES)
def test_replay_never_exceeds_the_every_scale_bound(sample, above_today, request):
    _, pairs = request.getfixturevalue(sample)
    above = [p for p in pairs if p[3] > p[1] + TOLERANCE]
    assert len(above) == above_today  # on the corners, the counts of today's tests
    assert all(hour is not None for hour, *_ in above)  # each at a scale with a failing hour
    assert [p for p in pairs if max(p[3], p[4]) > p[2] + TOLERANCE] == []


@pytest.mark.parametrize("sample", ["made_up", "lower_rate"])
def test_equal_to_todays_bound_wherever_no_hour_fails(sample, request):
    _, pairs = request.getfixturevalue(sample)
    calm = [p for p in pairs if p[0] is None]
    assert len(calm) > 100
    assert all(p[2] == p[1] for p in calm)


@pytest.mark.parametrize("sample", ["made_up", "lower_rate"])
def test_the_bound_never_falls_as_the_scale_rises(sample, request):
    items, _ = request.getfixturevalue(sample)
    for day, policy in items[::4]:
        rows = true_rows(day)
        hours = service_hours(policy, rows)
        switches = hours.thresholds[np.isfinite(hours.thresholds) & (hours.thresholds > 0)]
        betas = np.unique(np.r_[FINE, switches, np.nextafter(switches, np.inf)])
        values = [every_scale_driver_bound(policy, rows, day, b) for b in betas]
        assert np.all(np.diff(values) >= -1e-12), day.date
        assert max(values) <= 1.0 + 1e-12
