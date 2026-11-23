"""The every-scale driver scale (`gate_calibrate`, module text; `fixed_plan_loss`, reading R3), on
made-up hand units: why it can be above today's driver scale, what the final tier A scale can and
cannot exceed, why the formula can give a very large finite number, how the validity limit
(`policy.VALIDITY_LIMIT`, 1e5) limits every scale of the every-request route, and how replay
exceeds the closed forms by more than V5 to V7 of `fixed_plan_loss`'s module text allow above that
limit. The every-scale bound is the route's driver bound; today's bound is the earlier bound,
whose figures the route reports under `gate_calibrate.EARLIER`, a comparison only.

Every hand day has two service hours and two recovery hours, rho 0 and forecasts equal to the true
limits. Unless said otherwise its car (or each of its two cars) has rate 2 kW and leaves in hour 4,
so its recovery supply is 2 kWh (one whole recovery hour); with room 2.9 kWh its need row asks
`N = 2.9 - 2 = 0.9` kWh of hours 0 to 2 and its protected energy is 2.9 kWh. It plans 0 in hour 0
(it draws 0 there) and `g_t + beta G_t zeta` in service hour `t`. Every unit has 19 offering
calibration days, the
fewest the driver level 0.05 allows: (sum of the driver losses + 1) / 20 <= 0.05 holds only while
every day's bound is 0, so a driver scale is the scale at which some day's bound first rises above
0. The slot condition (sum of the slot losses + 1) / 20 <= 0.10 lets one day lose its whole window.

1. `DIP`: plan `0.5 + 0.5 beta zeta_1` and `1.5 + 0.1 beta zeta_2`. Hour thresholds: hour 1 power
   at least 0 at 0.5 / 0.5 = 1 (rate (2 - 0.5) / 0.5 = 3, stored energy (2.9 - 0.5) / 0.5 = 4.8);
   hour 2 stored energy (2.9 - 2) / 0.6 = 1.5 (rate 0.5 / 0.1 = 5, power at least 0 at 15). Every
   day loses its whole window above 1: slot scale 1.
   - Today's bound: the need row's worst case `0.9 - 2 + (0.5 + 0.1) beta`, above 0 from
     1.1 / 0.6 = 11/6.
   - The every-scale bound: today's up to 1 (no power row broken). Above 1 hour 1 is in `C`: on
     `zeta_1 = -1` its plan `0.5 - 0.5 beta` is below 0, and a one-way car then draws 0, not a
     negative amount. The by-hour bound counts `clip(0.5 - 0.5 beta, 0, 2) = 0` for it, so
     `L = 1.5 - 0.1 beta` (hour 2's power rows break at 5, and `clip(1.5 - 0.1 beta, 0, 2)` is the
     same number up to 15), while the by-corner bound `2 - 0.6 beta` is lower: the term
     `0.9 - 1.5 + 0.1 beta` is above 0 from 6. At 3: today's 0.7 / 2.9, the every-scale bound 0;
     at 8: today's 1 (3.7 / 2.9 capped at 1), the every-scale bound 0.2 / 2.9.
   - Driver scales 11/6 and 6; tier A is the slot scale 1 under both.
   The term that raises the every-scale driver scale on this unit: the by-hour bound's floor at 0
   on an hour whose power-low row is broken; today's bound counts the plan below 0 as energy taken
   away.
2. `FLAT`: `DIP` with no gain in hour 2. Today's bound `0.9 - 2 + 0.5 beta`, above 0 from 2.2;
   above 1 the every-scale `L` is 1.5 at every scale, so its term is 0 at every scale: the
   condition holds at every scale and the driver scale is infinite ("inf" in the results), tier A
   the slot scale 1 under both.
3. `MIXED`: one `DIP` day and 18 days of `EVEN`, plan `0.5 + 0.15 beta zeta_1` and
   `1.5 + 0.15 beta zeta_2`. `EVEN`'s thresholds: hour 1 at 0.5 / 0.15 = 10/3, hour 2 stored energy
   0.9 / 0.3 = 3 (rate 0.5 / 0.15 = 10/3). `DIP` loses its window above 1 (allowed), the `EVEN`
   days half theirs above 3: slot scale 3. `EVEN`'s today's bound `0.9 - 2 + 0.3 beta` is above 0
   from 11/3; above 10/3 both its hours are in `C` and `L = clip(0.5 - 0.15 beta, 0, 2) +
   clip(1.5 - 0.15 beta, 0, 2) = 1.5 - 0.15 beta`, above `N` from 4. Driver scales: today's
   min(11/6, 11/3) = 11/6, the every-scale one min(6, 4) = 4; tier A: today's min(3, 11/6) = 11/6,
   the every-scale one min(3, 4) = 3.

What the final tier A scale under the every-scale bound can reach: `_every_request_scale` of
`min(slot scale, driver scale)` in `gate_calibrate`, which is the smaller of the two when it is
finite, else the grid's top 2, so never above the slot scale (on every unit here). It can be above
today's final scale (`MIXED`: 3 against 11/6) and above the grid's top (`MIXED`: 3 > 2): only an
infinite scale is capped, so a finite slot scale above 2 is used as it is, under either bound, up
to the validity limit 1e5. With the slot scale and the every-scale driver scale both infinite it is
the grid's top 2, below today's final scale when today's driver scale is finite and above 2 (`H4`:
2 against 3.2).

4. Gains of at most `policy.ZERO_GAIN` (1e-12). The thresholds read a row with such gains as not
   depending on the requests (`policy.row_thresholds`), so that row is broken at every scale or at
   none; its hour can still break through its other rows. The every-scale bound and its limit
   (`gate_calibrate._every_scale_limit`) count every gain as it is, however small, so the limit is
   the day's loss as the scale grows without end (1 on `NOISE_HOUR`, `NOISE_CAR` and `W`; 1.8 /
   3.8 on `NOISE_BROKEN`, where both hours end clipped at 0 and the term is `N`). Once every hour
   with a larger gain is broken and counted at 0 or more, only such gains still move the bound,
   and the formula's driver scale is an energy margin over such a gain: a very large finite
   number, the largest scale at which the condition on the formula holds (`beta_hat_every_scale`,
   by `_beta_hat`'s doubling and bisection), so a value of the formula, not an error of the
   calibration. It is above the validity limit, so the route writes the limit as the driver scale
   and the formula's value in its mark (`gate_calibrate.VALIDITY`). The day's tiny-gain limit
   (`fixed_plan_loss`, module text) is above the validity limit on these units (`NOISE_HOUR`,
   `NOISE_CAR`: 5e13 for their power rows; `NOISE_BROKEN`: 2.5e14). On each unit the final scale
   is the slot scale:
   - `NOISE_HOUR`: `DIP` with the gain 1e-14 in hour 2. Its power rows never break, so hour 2 is
     never in `C`: above 1 `L = 1.5 - 1e-14 beta`, below `N` from 0.6 / 1e-14 = 6e13. Today's
     driver scale 2.2 (`0.9 - 2 + (0.5 + 1e-14) beta`); slot scale 1.
   - `NOISE_CAR`: two cars, `FLAT`'s plan and `0.5 + 1e-14 beta zeta_1`, `1.0`: the second car has
     no gain above `ZERO_GAIN`, so none of its power rows ever breaks and its term is today's,
     `0.9 - 1.5 + s beta` with `s` its need row's gain (1e-14, less the rounding of 1.5 + 1e-14),
     above 0 from 0.6 / s, about 6.0e13. Today's driver scale 2.2 (the first car); slot scale 1.
   - `NOISE_BROKEN`: room 3.8 (`N` 1.8), plan `0.5 + 0.5 beta zeta_1` and `2.5 + 1e-14 beta
     zeta_2`: hour 2 plans 2.5 against the rate 2, so its power-high row is broken at the centre
     (threshold `-inf`): every day loses half its window at every scale above 0 and the slot scale
     is 0. Hour 2 is in `C` at every scale, where the car draws `clip(2.5 - 1e-14 beta, 0, 2) = 2`
     up to 5e13; with hour 1 floored above 1, `L = 2` up to there, below `N` from 0.7 / 1e-14 =
     7e13. Today's driver scale 2.4 (`1.8 - 3 + 0.5 beta`).
5. `W`: car A leaves in hour 3 (no recovery hour), rate 2, room 2e-12 kWh (`N` and the protected
   energy 2e-12), plan 1e-12 in hours 1 and 2 with the gain 4e-13 on its own hour's request; car B
   leaves in hour 4, rate 2, room 1e-5, plan 2e-6 with the gain 1e-6 on its own hour's request.
   Every row of car A has gains of at most 1e-12 (4e-13 per power row, 8e-13 for its stored-energy
   row of hour 2 and its need row), so none ever breaks and its term is today's,
   `2e-12 - 2e-12 + 8e-13 beta`; car B's recovery supply covers its need (term 0). The day's loss is
   `8e-13 beta / (1e-5 + 2e-12)` under both bounds at every scale up to 2, and the condition
   (19 times it, at most `crc.TOLERANCE` 1e-9) holds up to `1e-9 (1e-5 + 2e-12) / (19 x 8e-13) =
   6.579e-4`. Car B's power-low rows break at 2e-6 / 1e-6 = 2 (its stored-energy row of hour 2 at
   (1e-5 - 4e-6) / 2e-6 = 3): slot scale 2. Both driver scales and both final scales are 6.579e-4.
   A limit that read gains of at most `ZERO_GAIN` as zero while the bound counts them would be 0
   here, so the condition would hold "at every scale" ("inf") and the scale used would be 2, where
   the summed bound is 3.04e-6 and (3.04e-6 + 1) / 20 > 0.05: the condition fails at the scale
   used. The limit must count them as the bound does, and it does. The tiny-gain limit of `W`'s
   power rows is 2.5 (car A's power-low rows, 1e-12 / 4e-13), above 6.579e-4; that of its closing
   rows is 0 (car A's stored-energy row of hour 2 is at its bound).
6. `H3`, where replay exceeds the closed forms by more than V5 to V7 allow above the limit
   (`fixed_plan_loss`, module text, "The validity limit and the stated errors"). Car A (rate 2,
   leaves in hour 4, room 4.4, `N` 2.4) plans `2 + X beta zeta_1` in hour 1, at its rate, and
   `0.5 - X beta zeta_1` in hour 2; car B (rate 2, leaves in hour 4, room 2, `N` 0) plans
   `1 + X beta zeta_1` in hour 2; `X` = 5e-13. Every row's gains are at most `ZERO_GAIN` and every
   row holds at the centre, so every threshold reads infinite: no hour ever fails or joins `C`, and
   both driver bounds are today's, car A's term `max(0, 2.4 - 2.5) = 0` (its need row's gains
   cancel) and car B's `max(0, -1 + X beta)`. Car A's power-high row of hour 1 is at its bound, so
   the day's tiny-gain limit is 0: that row's excess is `X beta`, `E_1` of hour 1 and `D` and `D_C`
   of both driver bounds (the other rows have none up to 1e12); what it adds to V7's error is
   `LAMBDA` (about 1e-7 kWh) and its `delta_k`. Replay on `zeta_1 = +1`: car A draws its rate 2 in
   hour 1, missing the request by `X beta`, which fails the hour once above `FAILURE_KWH` (beta
   above 2e8, where the slot form gives no failing hour: "replay earlier"); it ends with
   `4.5 - X beta` after recovery, short of 4.4 above 2e11: driver loss `(X beta - 0.1) / 6.4`,
   0.0078125 at 3e11 and 0.0625 at 1e12, against both bounds 0. On 19 copies the slot scale is
   infinite and both formula driver scales are `1 / s` = 1.9998e12 (`s` car B's need-row gain, `X`
   to the rounding of `1 + X`), where replay loses the whole window and 0.0625 of the protected
   energy, above the level 0.05. Both are above the validity limit, so the route uses the limit 1e5
   under both bounds, marked; there exact replay misses hour 1 by exactly `X 1e5` = 5e-8 kWh, `E_1`,
   the code's replay by at most that plus `R_1`, and neither has a driver loss.
7. `H4`: one day of one car (rate 3, leaves in hour 5, room 6.9, `N` 0.9) planning 1.5 in hour 0,
   `0.5 + 0.5 beta zeta_1` in hour 1 and 0.5 in hour 2, and 18 days of one car with room 0 (no
   protected energy) and the gain `X` in hour 1. The slot losses sum to 1 at every scale above 1
   (the first day's hour 1), and (1 + 1) / 20 = 0.10: the slot scale is infinite. Today's bound
   on the first day, `0.9 - 2.5 + 0.5 beta`, sets today's driver scale 3.2; above 1 its
   every-scale bound counts hour 1 at `clip(0.5 - 0.5 beta, 0, 3) = 0` and `L = 2 >= N`, so the
   every-scale driver scale is infinite. Tier A: today's 3.2 (finite, used as it is), the
   every-scale one the grid's top 2.
8. `BIG`: one car (room 2.5, `N` 0.5) planning `1 + 1e-6 beta zeta_1` in hour 1 and 0 in hour 2:
   ordinary gains, no row with gains of at most `ZERO_GAIN`. Hour 1's power rows break at 1e6, so
   on 19 copies the slot scale is 1e6; the need row's worst case `0.5 - 1 + 1e-6 beta` is above 0
   from 5e5, the driver scale under both bounds (no power row is broken below 1e6). Every scale of
   the route is above the validity limit: tier A, tier B and today's tier A (the earlier block)
   use the limit 1e5, each marked with the formula's values. `DIP`, `FLAT`, `MIXED`, `W` and `H4`
   have no finite scale above the limit, so nothing of theirs is marked; the noise units' tier A
   and both tiers A of `H3` are.
"""

import datetime
import math

import numpy as np
import pytest
from corner_fake import hand
from validity_errors import (
    balance_gaps,
    driver_error,
    driver_limit_error,
    exact,
    exact_bounds,
    exact_replay,
    excesses,
    forms,
    hour_errors,
    replayed,
    roundings,
)

from fleetreserve import crc, gate_calibrate
from fleetreserve.carsets import true_rows
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.fixed_plan_loss import (
    CLOSING_KINDS,
    compare_with_replay,
    drawn_policy,
    driver_loss_bound,
    every_scale_driver_bound,
    hour_thresholds,
)
from fleetreserve.gate_calibrate import EARLIER
from fleetreserve.plans import Plan
from fleetreserve.policy import VALIDITY_LIMIT, ZERO_GAIN, AffinePolicy
from fleetreserve.products import corner_paths
from fleetreserve.replay import FAILURE_KWH, run_plan

DAYS = 19
TOLERANCE = 1e-8  # relative: crc.TOLERANCE, 1e-9 on the summed losses, moves a crossing less
POWER_KINDS = ("power-high", "power-low")


def _day(rooms, departures, rates) -> Day:
    limits = tuple(zip(departures, rates, rooms, strict=True))
    cars = tuple(TrueCar(departure, rate, room) for departure, rate, room in limits)
    forecasts = tuple(
        ForecastCar(float(departure), rate, room, room) for departure, rate, room in limits
    )
    return Day("hand", datetime.date(2020, 1, 1), 2, 2, 0.0, cars, forecasts, 0.1)


def _car(centre, gain_1, gain_2, first=0.0):
    """One car's planned powers (hours 0 to 2) and gains: hour 1 on zeta_1, hour 2 on zeta_2."""
    return [first, *centre], [[0.0, 0.0], [gain_1, 0.0], [0.0, gain_2]]


def _prepared(rooms, *cars, departures=None, rates=None):
    g = [c[0] for c in cars]
    G = [c[1] for c in cars]
    m = np.sum(g, axis=0)[1:]
    w = np.abs(np.sum(G, axis=0)).sum(axis=1)[1:]
    day = _day(rooms, departures or [4] * len(rooms), rates or [2.0] * len(rooms))
    return hand(day, AffinePolicy(m, w, g, G))


DIP = _prepared([2.9], _car((0.5, 1.5), 0.5, 0.1))
FLAT = _prepared([2.9], _car((0.5, 1.5), 0.5, 0.0))
EVEN = _prepared([2.9], _car((0.5, 1.5), 0.15, 0.15))
NOISE_HOUR = _prepared([2.9], _car((0.5, 1.5), 0.5, 1e-14))
NOISE_CAR = _prepared([2.9, 2.9], _car((0.5, 1.5), 0.5, 0.0), _car((0.5, 1.0), 1e-14, 0.0))
NOISE_BROKEN = _prepared([3.8], _car((0.5, 2.5), 0.5, 1e-14))
W_PROTECTED = 1e-5 + 2e-12
W = _prepared(
    [2e-12, 1e-5],
    _car((1e-12, 1e-12), 4e-13, 4e-13),
    _car((2e-6, 2e-6), 1e-6, 1e-6),
    departures=[3, 4],
)
X = 5e-13
H3 = _prepared(
    [4.4, 2.0],
    ([0.0, 2.0, 0.5], [[0.0, 0.0], [X, 0.0], [-X, 0.0]]),
    ([0.0, 0.0, 1.0], [[0.0, 0.0], [0.0, 0.0], [X, 0.0]]),
)
H4_A = _prepared([6.9], _car((0.5, 0.5), 0.5, 0.0, first=1.5), departures=[5], rates=[3.0])
H4_Z = _prepared([0.0], _car((0.0, 0.0), X, 0.0))
BIG = _prepared([2.5], _car((1.0, 0.0), 1e-6, 0.0))

UNITS = {
    "dip": [DIP] * DAYS,
    "flat": [FLAT] * DAYS,
    "mixed": [DIP] + [EVEN] * (DAYS - 1),
}


def _route(prepared) -> tuple[dict, dict]:
    """Tier A under today's bound (the earlier block, with the shared slot scale) and under the
    every-scale bound (the route's own fields)."""
    found = gate_calibrate.every_request(prepared, list(range(DAYS)), [], earlier=True)
    return {**found[EARLIER], "slot_beta": found["slot_beta"]}, found


def _tiny_gain_limit(prepared, kinds) -> float:
    """The tiny-gain limit of the day's true rows of `kinds` (`fixed_plan_loss`, module text): the
    smallest `(b - a g) / |a G|_1` on the drawn policy over the rows with `0 < |a G|_1 <= ZERO_GAIN`
    that `row_thresholds` reads as never broken; infinite when there is none."""
    rows = true_rows(prepared.day)
    drawn = drawn_policy(prepared.certificate.policy, rows)
    centre = np.einsum("kt,kt->k", rows.coefficients, drawn.g[rows.car])
    spread = np.abs(np.einsum("kt,kts->ks", rows.coefficients, drawn.G[rows.car])).sum(axis=1)
    read = drawn.row_thresholds(rows)
    tiny = np.isin(rows.kinds, kinds) & (spread > 0) & (spread <= ZERO_GAIN) & np.isposinf(read)
    return float(((rows.bounds - centre)[tiny] / spread[tiny]).min()) if tiny.any() else math.inf


def _replayed(prepared, beta: float) -> tuple[float, float]:
    """Replay's largest slot loss and largest driver loss over the corner paths at `beta`."""
    policy = prepared.certificate.policy
    plan = Plan.from_policy(policy, beta)
    runs = [
        run_plan(prepared.day, plan, z, policy.request(z, beta))
        for z in corner_paths(prepared.day.n_service)
    ]
    return max(r.slot_loss for r in runs), max(r.driver_loss for r in runs)


# --- 1 to 3. the every-scale driver scale above today's, and the final scale ---------------------


def test_a_plan_below_0_lifts_the_every_scale_driver_scale_above_todays():
    today, every = _route(UNITS["dip"])
    assert today["slot_beta"] == 1.0
    assert today["driver_beta"] == pytest.approx(11 / 6, rel=TOLERANCE)
    assert every["driver_beta"] == pytest.approx(6.0, rel=TOLERANCE)
    assert today["beta_used"] == every["beta_used"] == 1.0
    terms = gate_calibrate.every_scale_terms(DIP)
    for beta, todays, every_scale in ((0.5, 0, 0), (1, 0, 0), (3, 0.7, 0), (8, 2.9, 0.2)):
        assert gate_calibrate.ramp_loss(terms[0], beta) == pytest.approx(todays / 2.9)
        assert gate_calibrate.every_scale_loss(terms, beta) == pytest.approx(every_scale / 2.9)


def test_a_floor_that_holds_every_bound_at_0_gives_no_limit():
    today, every = _route(UNITS["flat"])
    assert today["driver_beta"] == pytest.approx(2.2, rel=TOLERANCE)
    assert every["driver_beta"] == gate_calibrate.INF
    assert every["beta"] == today["beta_used"] == every["beta_used"] == 1.0
    assert not every["capped_at_grid_top"]


def test_the_final_scale_can_pass_todays_and_the_grids_top_but_not_the_slot_scale():
    today, every = _route(UNITS["mixed"])
    assert today["slot_beta"] == 3.0
    assert today["driver_beta"] == pytest.approx(11 / 6, rel=TOLERANCE)
    assert every["driver_beta"] == pytest.approx(4.0, rel=TOLERANCE)
    assert today["beta_used"] == pytest.approx(11 / 6, rel=TOLERANCE)
    assert every["beta_used"] == 3.0 > crc.GRID[-1]
    assert not every["capped_at_grid_top"]
    for prepared in UNITS.values():
        today, every = _route(prepared)
        assert every["beta_used"] <= today["slot_beta"]


# --- 4 and 5. gains of at most ZERO_GAIN, counted as they are ------------------------------------


def _driver_condition_holds(terms, beta: float) -> bool:
    """The driver condition on `DAYS` copies of one day: (sum of the losses + 1) / 20 <= 0.05."""
    loss_sum = DAYS * gate_calibrate.every_scale_loss(terms, beta)
    return loss_sum + 1.0 <= crc.ALPHA_DRIVER * (DAYS + 1) + crc.TOLERANCE


@pytest.mark.parametrize(
    "prepared, slot, todays, margin, scale, limit, power_limit",
    [
        (NOISE_HOUR, 1.0, 2.2, 0.6, 6e13, 1.0, 5e13),
        (NOISE_CAR, 1.0, 2.2, 0.6, 6e13, 1.0, 5e13),
        (NOISE_BROKEN, 0.0, 2.4, 0.7, 7e13, 1.8 / 3.8, 2.5e14),
    ],
    ids=["noise-hour", "noise-car", "noise-broken"],
)
def test_a_tiny_counted_gain_gives_a_very_large_finite_driver_scale(
    prepared, slot, todays, margin, scale, limit, power_limit
):
    terms = gate_calibrate.every_scale_terms(prepared)
    (_, spread, _), hours = terms
    gain = spread[-1] if prepared is NOISE_CAR else hours.spread[0, 1]
    assert 0 < gain <= ZERO_GAIN
    crossing = margin / gain
    assert crossing == pytest.approx(scale, rel=1e-3)
    assert gate_calibrate.every_scale_loss(terms, crossing * (1 - 1e-6)) == 0.0
    assert gate_calibrate.every_scale_loss(terms, crossing * (1 + 1e-6)) > 0.0
    today, every = _route([prepared] * DAYS)
    assert today["slot_beta"] == slot
    assert today["driver_beta"] == pytest.approx(todays, rel=TOLERANCE)
    assert every["driver_beta"] == VALIDITY_LIMIT
    assert every[gate_calibrate.VALIDITY]["formula_values"] == {
        "driver_beta": pytest.approx(crossing, rel=1e-6)
    }
    assert gate_calibrate.VALIDITY not in today
    assert every["beta"] == every["beta_used"] == slot
    assert not every["capped_at_grid_top"]
    assert gate_calibrate._every_scale_limit(terms) == pytest.approx(limit)
    assert gate_calibrate.every_scale_loss(terms, 1e16) == pytest.approx(limit)
    assert _tiny_gain_limit(prepared, POWER_KINDS) == pytest.approx(power_limit, rel=1e-9)


def test_the_bound_counts_gains_at_most_zero_gain_so_the_scale_used_meets_the_condition():
    terms = gate_calibrate.every_scale_terms(W)
    for beta in (1e-4, 6.5e-4, 2.0):
        expected = 8e-13 * beta / W_PROTECTED
        assert gate_calibrate.ramp_loss(terms[0], beta) == pytest.approx(expected, rel=1e-6)
        assert gate_calibrate.every_scale_loss(terms, beta) == pytest.approx(expected, rel=1e-6)
    crossing = crc.TOLERANCE * W_PROTECTED / (DAYS * 8e-13)
    assert crossing == pytest.approx(6.579e-4, rel=1e-4)
    assert _driver_condition_holds(terms, 0.99 * crossing)
    assert not _driver_condition_holds(terms, 1.01 * crossing)
    assert not _driver_condition_holds(terms, crc.GRID[-1])
    today, every = _route([W] * DAYS)
    assert today["slot_beta"] == 2.0
    assert today["driver_beta"] == pytest.approx(crossing, rel=1e-6)
    assert every["driver_beta"] == pytest.approx(crossing, rel=1e-6)
    assert today["beta_used"] == pytest.approx(crossing, rel=1e-6)
    assert every["beta_used"] == pytest.approx(crossing, rel=1e-6)
    assert not every["capped_at_grid_top"]
    assert _driver_condition_holds(terms, every["beta_used"])
    assert gate_calibrate._every_scale_limit(terms) == 1.0
    assert gate_calibrate.every_scale_loss(terms, 1e16) == 1.0
    assert _tiny_gain_limit(W, POWER_KINDS) == pytest.approx(2.5, rel=1e-9)
    assert _tiny_gain_limit(W, CLOSING_KINDS) == pytest.approx(0.0, abs=1e-9)


# --- 6 to 8. the validity limit, above it, and both scales infinite ----------------------------


def test_19_copies_of_h3_use_the_validity_limit_and_replay_there_is_within_the_error():
    today, every = _route([H3] * DAYS)
    assert today["slot_beta"] == gate_calibrate.INF
    formula = pytest.approx(1.9998e12, rel=1e-4)
    for block in (today, every):
        assert block["driver_beta"] == block["beta"] == block["beta_used"] == VALIDITY_LIMIT
        assert block[gate_calibrate.VALIDITY]["limit"] == VALIDITY_LIMIT == 1e5
        assert block[gate_calibrate.VALIDITY]["formula_values"] == {
            "driver_beta": formula,
            "beta": formula,
        }
        assert not block["capped_at_grid_top"]
    rows, policy, beta = true_rows(H3.day), H3.certificate.policy, today["beta_used"]
    f = forms(policy, rows)
    ex = excesses(f, rows, beta)
    errors, gaps = hour_errors(rows, ex), balance_gaps(policy, beta)
    assert errors == [exact(X) * exact(beta), 0] and gaps == [0, 0]
    assert float(errors[0]) == pytest.approx(5e-8)
    for every_scale in (False, True):
        assert driver_error(f, rows, beta, every_scale, ex) == errors[0]
    assert driver_loss_bound(policy, rows, H3.day, beta) == 0.0
    assert every_scale_driver_bound(policy, rows, H3.day, beta) == 0.0
    assert exact_bounds(f, policy, rows, H3.day, beta) == (0, 0)
    for z in corner_paths(2):
        misses, driver = exact_replay(H3.day, policy, beta, z)
        assert misses == [errors[0] if z[0] > 0 else 0, 0] and driver == 0
    rounding = roundings(H3.day, policy, beta, errors, gaps)
    runs = replayed(H3.day, policy, beta, corner_paths(2))
    for misses, first, driver in runs:
        assert all(m <= e + r for m, e, r in zip(misses, errors, rounding, strict=True))
        assert first is None
        assert driver == 0.0
    worst = max(misses[0] for misses, _, _ in runs)
    assert worst > errors[0] and float(worst) == pytest.approx(5e-8, rel=1e-9)


def test_every_scale_above_the_validity_limit_is_the_limit_in_both_tiers_under_both_bounds():
    found = gate_calibrate.every_request(
        [BIG] * DAYS, list(range(DAYS)), [], tier_b=True, earlier=True
    )
    tier_b, today = found["tier_b"], found[EARLIER]
    marks = [block[gate_calibrate.VALIDITY] for block in (found, tier_b, today)]
    assert [mark["formula_values"] for mark in marks] == [
        {
            "slot_beta": pytest.approx(1e6),
            "driver_beta": pytest.approx(5e5),
            "beta": pytest.approx(5e5),
        },
        {"beta": pytest.approx(1e6)},
        {"driver_beta": pytest.approx(5e5), "beta": pytest.approx(5e5)},
    ]
    assert found["slot_beta"] == found["driver_beta"] == found["beta"] == VALIDITY_LIMIT
    assert tier_b["beta"] == today["driver_beta"] == today["beta"] == VALIDITY_LIMIT
    for block in (found, tier_b, today):
        assert block["beta_used"] == VALIDITY_LIMIT
        assert not block["capped_at_grid_top"]
        assert not block["abstain"]
    for prepared in [*UNITS.values(), [W] * DAYS, [H4_A] + [H4_Z] * (DAYS - 1)]:
        found = gate_calibrate.every_request(
            prepared, list(range(DAYS)), [], tier_b=True, earlier=True
        )
        for block in (found, found["tier_b"], found[EARLIER]):
            assert gate_calibrate.VALIDITY not in block


def test_above_the_limit_replay_exceeds_both_driver_bounds_on_h3_by_more_than_the_limits_error():
    """Above the validity limit V7's error is exceeded: on `H3` exact replay's driver loss is
    above both bounds at 3e11 and 1e12 (not at 2 or 1e11) by more than V7's error (what the one
    row adds to it, `LAMBDA` and its `delta_k`, over the protected energy 6.4) and within V3's
    `D_C / P = X beta / 6.4`; at the formula's scale of 19 copies the code's replay gives 0.0625,
    above the level 0.05, where the route now uses the limit instead."""
    rows, policy = true_rows(H3.day), H3.certificate.policy
    f = forms(policy, rows)
    assert hour_thresholds(policy, rows).tolist() == [math.inf, math.inf]
    assert _tiny_gain_limit(H3, POWER_KINDS) == _tiny_gain_limit(H3, CLOSING_KINDS) == 0.0
    for beta, replayed_driver in ((2.0, 0.0), (1e11, 0.0), (3e11, 0.0078125), (1e12, 0.0625)):
        ex = excesses(f, rows, beta)
        assert _replayed(H3, beta)[1] == pytest.approx(replayed_driver, rel=1e-9)
        assert every_scale_driver_bound(policy, rows, H3.day, beta) == 0.0
        assert driver_loss_bound(policy, rows, H3.day, beta) == 0.0
        assert exact_bounds(f, policy, rows, H3.day, beta) == (0, 0)
        limit_error = driver_limit_error(f, rows, beta, True, ex) / exact(6.4)
        assert float(limit_error) == pytest.approx(1e-7 / 6.4)
        driver = max(exact_replay(H3.day, policy, beta, z)[1] for z in corner_paths(2))
        assert float(driver) == pytest.approx(replayed_driver, rel=1e-9, abs=1e-30)
        assert driver <= driver_error(f, rows, beta, True, ex) / exact(6.4)
        assert (driver > limit_error) == (beta > 2e11)
    today, _ = _route([H3] * DAYS)
    formula = today[gate_calibrate.VALIDITY]["formula_values"]["beta"]
    slot, driver = _replayed(H3, formula)
    assert (slot, driver) == (1.0, pytest.approx(0.0625, rel=1e-9))
    assert driver > crc.ALPHA_DRIVER


def test_above_the_limit_replay_fails_on_h3_where_the_slot_form_gives_no_failing_hour():
    """The same for the slot loss (reading R1): replay fails hour 1 at 4e8 and not at 1e8
    (`beta X` passes `FAILURE_KWH` at 2e8), where every threshold reads infinite; `E_1`, `X beta`,
    is above `FAILURE_KWH` at 4e8 and not at 1e8."""
    crossing = FAILURE_KWH / X
    assert crossing == pytest.approx(2e8)
    below, above = compare_with_replay(
        H3.day, H3.certificate.policy, true_rows(H3.day), [0.5 * crossing, 2.0 * crossing]
    )
    assert (below.closed_hour, below.replay_hour, below.replay_slot, below.reason) == (
        None,
        None,
        0.0,
        None,
    )
    assert (above.closed_hour, above.replay_hour, above.closed_slot, above.replay_slot) == (
        None,
        1,
        0.0,
        1.0,
    )
    assert above.reason == "replay earlier"
    policy, rows = H3.certificate.policy, true_rows(H3.day)
    f = forms(policy, rows)
    errors = [hour_errors(rows, excesses(f, rows, found.beta))[0] for found in (below, above)]
    assert errors[0] <= exact(FAILURE_KWH) < errors[1]


def test_with_both_scales_infinite_the_every_scale_tier_a_scale_is_the_grids_top():
    today, every = _route([H4_A] + [H4_Z] * (DAYS - 1))
    assert today["slot_beta"] == every["driver_beta"] == gate_calibrate.INF
    assert today["driver_beta"] == pytest.approx(3.2, rel=TOLERANCE)
    assert today["beta_used"] == today["driver_beta"]
    assert not today["capped_at_grid_top"]
    assert every["beta_used"] == crc.GRID[-1] == 2.0
    assert every["capped_at_grid_top"]
    assert every["beta_used"] < today["beta_used"]
