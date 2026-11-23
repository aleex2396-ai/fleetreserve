"""The time-ordered run of one unit under the drift guard (`drift_path`; plan section 6, T10).

1. A hand-computed case. 24 counting days, calibrated on the first 20 (`n_calibration` 20), with
   the step size eta 0.5.
   Calibration days 0 to 19 all offer; on each, the slot loss is 1 above a scale of 0.5 and the
   driver loss 1 above a scale of 0.2, 0 below. On n = 20 days the bound (n R + 1)/(n + 1) <=
   alpha holds at R = 0 (1/21 = 0.048 <= 0.05) and fails at R = 1, so the on-grid scales are:
   slot alone (alpha 0.10) 0.5, driver alone (alpha 0.05) 0.2; tier A 0.2 (both), tier B 0.5.
   Guarded days, values at scale 1: day 20 10 kWh (slot loss 1 above 0.3, driver loss 0.4 above
   0.1), day 21 8 kWh (no loss), day 22 not offering (value 0), day 23 6 kWh (slot loss 0.5 and
   driver loss 0.1 at any scale above 0). Update: beta + 0.5 (alpha - loss), alpha 0.10 slot,
   0.05 driver. Bound after t steps: alpha + (beta_1 / 0.5 + 1 - alpha) / t.

   Tier A, start "calibrated" (slot guard 0.5, driver guard 0.2; the offer is the smaller):
     day 20: scale 0.2, losses (0, 0.4), offer 2.0; slot 0.5 + 0.05 = 0.55, driver
             0.2 + 0.5 (0.05 - 0.4) = 0.025; means (0, 0.4), bounds (2.0, 1.4).
     day 21: scale 0.025, losses (0, 0), offer 0.2; slot 0.6, driver 0.05; means (0, 0.2),
             bounds (1.05, 0.725).
     day 22: does not offer: no step, offer 0, states kept.
     day 23: scale 0.05, losses (0.5, 0.1), offer 0.3; slot 0.6 - 0.2 = 0.4, driver
             0.05 - 0.025 = 0.025; means (1/6, 1/6), bounds (0.1 + 1.9/3, 0.5).
     mean offer per counting day (2.0 + 0.2 + 0 + 0.3)/4 = 0.625; 3 offering days, 0 abstaining.
   Tier A, start "zero" (both guards 0):
     day 20: scale 0, abstains, losses (0, 0), offer 0; slot 0.05, driver 0.025; bounds (1, 1).
     day 21: scale 0.025, no loss, offer 0.2; slot 0.1, driver 0.05.
     day 23: scale 0.05, losses (0.5, 0.1), offer 0.3; slot -0.1, driver 0.025.
     mean offer 0.5/4 = 0.125; 1 abstaining day of 3 offering days.
   Tier B, start "calibrated" (slot guard 0.5; the driver loss is reported, not held down):
     day 20: scale 0.5, losses (1, 0.4), offer 5.0; slot 0.5 - 0.45 = 0.05.
     day 21: scale 0.05, no loss, offer 0.4; slot 0.1.
     day 23: scale 0.1, losses (0.5, 0.1), offer 0.6; slot -0.1.
     mean offer 6.0/4 = 1.5; mean slot loss 1.5/3 = 0.5, bound 0.1 + 1.9/3.
   Tier B, start "zero": day 20 abstains (slot 0.05); day 21 scale 0.05, offer 0.4 (slot 0.1);
     day 23 scale 0.1, offer 0.6 (slot -0.1); mean offer 1.0/4 = 0.25.
   Evaluations: every offering calibration day at every grid scale; a guarded day only where it
   offers and at a scale above 0, each (day, scale) once.
2. On made-up sequences (random days, random step losses, and the plan's case of a day lost
   whenever the scale is above 0), every guard's running average loss after t steps is at most
   alpha + (beta_1 / eta + 1 - alpha) / t, for both tiers, both starts and five step sizes; the
   state is never clipped (it goes below 0) and the offered scale never exceeds the cap.
"""

import math
import random

import pytest

from fleetreserve import dial_splits, drift_guard, drift_path

GRID = dial_splits.GRID
ETA = 0.5
N_CAL = 20


def _hand_unit():
    values = [10.0] * N_CAL + [10.0, 8.0, 0.0, 6.0]
    offering = [True] * N_CAL + [True, True, False, True]
    keys = [f"2020-01-{d + 1:02d} A" for d in range(len(values))]

    def losses(day, scale):
        if day < N_CAL:
            return (1.0 if scale > 0.5 else 0.0, 1.0 if scale > 0.2 else 0.0)
        if day == 20:
            return (1.0 if scale > 0.3 else 0.0, 0.4 if scale > 0.1 else 0.0)
        if day == 21:
            return (0.0, 0.0)
        if day == 23:
            return (0.5, 0.1) if scale > 0 else (0.0, 0.0)
        raise AssertionError(f"day {day} does not offer and must not be evaluated")

    return keys, values, offering, losses


class Counting:
    """An `evaluate` that answers from `losses` and counts each (day, scale) asked for."""

    def __init__(self, losses):
        self.losses = losses
        self.calls: dict = {}

    def __call__(self, pairs):
        for pair in pairs:
            self.calls[pair] = self.calls.get(pair, 0) + 1
        return {pair: self.losses(*pair) for pair in pairs}


@pytest.fixture(scope="module")
def hand():
    keys, values, offering, losses = _hand_unit()
    evaluate = Counting(losses)
    found = drift_path.run_unit(keys, values, offering, evaluate, n_calibration=N_CAL, etas=(ETA,))
    return found, evaluate


def _path(found, tier, start):
    (path,) = [p for p in found.paths if p.tier == tier and p.start == start and p.eta == ETA]
    return path


def test_the_first_days_calibrate_each_guard_on_the_grid(hand):
    found, _ = hand
    calibration = found.calibration
    assert calibration.days == N_CAL and calibration.offering_days == N_CAL
    assert calibration.tier_scale == {"A": 0.2, "B": 0.5}
    assert calibration.loss_scale == {"A": {"slot": 0.5, "driver": 0.2}, "B": {"slot": 0.5}}
    for tier in ("A", "B"):  # the smaller guard start is the tier's own on-grid scale
        assert min(calibration.loss_scale[tier].values()) == calibration.tier_scale[tier]


# (scale, slot loss, driver loss, offer, slot state after, driver state after) per guarded day;
# None for the day that does not offer.
HAND = {
    ("A", "calibrated"): [
        (0.2, 0.0, 0.4, 2.0, 0.55, 0.025),
        (0.025, 0.0, 0.0, 0.2, 0.6, 0.05),
        None,
        (0.05, 0.5, 0.1, 0.3, 0.4, 0.025),
    ],
    ("A", "zero"): [
        (0.0, 0.0, 0.0, 0.0, 0.05, 0.025),
        (0.025, 0.0, 0.0, 0.2, 0.1, 0.05),
        None,
        (0.05, 0.5, 0.1, 0.3, -0.1, 0.025),
    ],
    ("B", "calibrated"): [
        (0.5, 1.0, 0.4, 5.0, 0.05, None),
        (0.05, 0.0, 0.0, 0.4, 0.1, None),
        None,
        (0.1, 0.5, 0.1, 0.6, -0.1, None),
    ],
    ("B", "zero"): [
        (0.0, 0.0, 0.0, 0.0, 0.05, None),
        (0.05, 0.0, 0.0, 0.4, 0.1, None),
        None,
        (0.1, 0.5, 0.1, 0.6, -0.1, None),
    ],
}
SUMMARY = {  # mean offer per counting day, offering days, abstaining days, final mean losses
    ("A", "calibrated"): (0.625, 3, 0, 0.5 / 3, 0.5 / 3),
    ("A", "zero"): (0.125, 3, 1, 0.5 / 3, 0.1 / 3),
    ("B", "calibrated"): (1.5, 3, 0, 0.5, 0.5 / 3),
    ("B", "zero"): (0.25, 3, 1, 0.5 / 3, 0.1 / 3),
}
STARTS = {("A", "calibrated"): (0.5, 0.2), ("A", "zero"): (0.0, 0.0)}
STARTS.update({("B", "calibrated"): (0.5, None), ("B", "zero"): (0.0, None)})


@pytest.mark.parametrize("case", sorted(HAND))
def test_the_guarded_days_follow_the_hand_computed_scales_and_losses(hand, case):
    found, _ = hand
    path = _path(found, *case)
    rows = path.rows
    assert [r.day for r in rows] == [20, 21, 22, 23]
    slot_1, driver_1 = STARTS[case]
    state = {"slot": slot_1, "driver": driver_1}
    sums = {"slot": 0.0, "driver": 0.0}
    for row, expected, step in zip(rows, HAND[case], (1, 2, None, 3), strict=True):
        guarded_state = {k: v for k, v in state.items() if v is not None}
        assert row.state_before == pytest.approx(guarded_state, abs=1e-12)
        if expected is None:
            assert not row.offering and row.step is None and row.offer_kwh == 0.0
            assert row.scale == 0.0 and row.losses is None
            continue
        scale, slot, driver, offer, slot_after, driver_after = expected
        assert row.offering and row.step == step
        assert row.scale == pytest.approx(scale, abs=1e-12)
        assert row.losses == pytest.approx((slot, driver), abs=1e-12)
        assert row.offer_kwh == pytest.approx(offer, abs=1e-12)
        state = {"slot": slot_after, "driver": driver_after}
        sums["slot"] += slot
        sums["driver"] += driver
        assert row.mean_loss == pytest.approx({k: v / row.step for k, v in sums.items()})
        alphas = {"slot": 0.10, "driver": 0.05}
        expected_bounds = {
            loss: alphas[loss] + (STARTS[case][k] / ETA + 1 - alphas[loss]) / row.step
            for k, loss in enumerate(("slot", "driver"))
            if STARTS[case][k] is not None
        }
        assert row.bound == pytest.approx(expected_bounds)
    final = {k: v for k, v in state.items() if v is not None}
    assert path.state == pytest.approx(final, abs=1e-12)


@pytest.mark.parametrize("case", sorted(SUMMARY))
def test_the_summary_of_each_path_is_the_hand_computed_one(hand, case):
    found, _ = hand
    summary = _path(found, *case).summary()
    offer, offering, abstaining, slot, driver = SUMMARY[case]
    assert summary["guarded_counting_days"] == 4
    assert summary["mean_offer_kwh_per_counting_day"] == pytest.approx(offer, abs=1e-12)
    assert summary["offering_days"] == offering
    assert summary["guard_abstaining_days"] == abstaining
    assert summary["losses"]["slot"]["mean"] == pytest.approx(slot, abs=1e-12)
    assert summary["losses"]["driver"]["mean"] == pytest.approx(driver, abs=1e-12)
    assert summary["losses"]["slot"]["guarded"] is True
    assert summary["losses"]["driver"]["guarded"] is (case[0] == "A")
    assert summary["losses"]["slot"]["days_over_bound"] == 0


def test_each_day_is_evaluated_only_where_it_offers_and_once_per_scale(hand):
    found, evaluate = hand
    assert all(n == 1 for n in evaluate.calls.values())
    calibration = {(d, s) for d, s in evaluate.calls if d < N_CAL}
    assert calibration == {(d, g) for d in range(N_CAL) for g in GRID}
    guarded = {(d, s) for d, s in evaluate.calls if d >= N_CAL}
    offered = {(r.day, r.scale) for p in found.paths for r in p.rows if r.scale > 0}
    assert guarded == offered
    assert all(s > 0 and d != 22 for d, s in guarded)


# --- 2. the bound on made-up sequences -----------------------------------------------------------

ETAS = (0.01, 0.1, 0.5, 1.0, 3.0)


def _random_unit(seed: int):
    rng = random.Random(seed)
    n = rng.randrange(N_CAL + 30, N_CAL + 160)
    offering = [rng.random() < 0.85 for _ in range(n)]
    values = [rng.uniform(1.0, 20.0) if o else 0.0 for o in offering]
    keys = [f"day-{d:04d} A" for d in range(n)]
    thresholds = [(rng.uniform(0.0, 1.2), rng.uniform(0.0, 1.2)) for _ in range(n)]
    levels = [(rng.choice((0.25, 0.5, 1.0)), rng.choice((0.1, 0.5, 1.0))) for _ in range(n)]
    upward = [rng.random() < 0.5 for _ in range(n)]

    def losses(day, scale):
        (ts, td), (ls, ld) = thresholds[day], levels[day]
        if upward[day]:  # a step loss that rises with the scale
            return (ls if scale > ts else 0.0, ld if scale > td else 0.0)
        draw = random.Random(f"{seed}-{day}-{scale!r}")  # any loss in [0, level], fixed per pair
        return (ls * draw.random(), ld * draw.random())

    return keys, values, offering, losses


def _lose_when_offering(n: int):
    """The plan's case: any scale above 0 loses the day (both losses 1), 0 loses nothing."""
    keys = [f"day-{d:04d} A" for d in range(n)]
    return keys, [5.0] * n, [True] * n, lambda day, scale: (1.0, 1.0) if scale > 0 else (0.0, 0.0)


UNITS = [_random_unit(seed) for seed in range(12)] + [_lose_when_offering(N_CAL + 116)]


def _runs():
    for keys, values, offering, losses in UNITS:
        yield drift_path.run_unit(
            keys, values, offering, Counting(losses), n_calibration=N_CAL, etas=ETAS
        )


def test_the_running_average_loss_obeys_t10s_bound_on_made_up_sequences():
    steps = 0
    lost = 0.0
    for found in _runs():
        assert len(found.paths) == 2 * 2 * len(ETAS)
        for path in found.paths:
            alphas = drift_path.guards(path.tier)
            for row in path.rows:
                if row.step is None:
                    continue
                steps += 1
                lost += row.losses[0]
                for loss, alpha in alphas.items():
                    bound = alpha + (path.beta_1[loss] / path.eta + 1 - alpha) / row.step
                    assert row.bound[loss] == pytest.approx(bound)
                    assert row.mean_loss[loss] <= bound + 1e-9
            assert path.summary()["losses"]["slot"]["days_over_bound"] == 0
    assert steps > 10_000 and lost > 100  # the runs stepped and lost: the bound was exercised


def test_the_state_is_unclipped_and_the_offer_is_capped():
    went_below_zero = above_cap = 0
    for found in _runs():
        for path in found.paths:
            state = dict(path.beta_1)
            for row in path.rows:
                assert row.state_before == pytest.approx(state)
                if row.step is None:
                    continue
                low = min(state.values())
                expected = 0.0 if low <= 0 else min(low, drift_path.CAP)
                assert row.scale == expected
                above_cap += low > drift_path.CAP
                for k, (loss, alpha) in enumerate(drift_path.guards(path.tier).items()):
                    state[loss] = drift_guard.step(
                        state[loss], row.losses[k], alpha, path.eta, clip=False
                    )
                went_below_zero += min(state.values()) < 0
    assert went_below_zero > 0 and above_cap > 0


def test_the_plans_case_averages_0_103_at_day_116_unclipped():
    found = drift_path.run_unit(
        *_lose_when_offering(N_CAL + 116)[:3],
        Counting(_lose_when_offering(N_CAL + 116)[3]),
        n_calibration=N_CAL,
        etas=(0.5,),
    )
    path = _path_of(found, "B", "zero", 0.5)
    last = path.rows[-1]
    assert last.step == 116
    assert round(last.mean_loss["slot"], 3) == 0.103
    assert math.isclose(path.summary()["losses"]["slot"]["mean"], last.mean_loss["slot"])


def _path_of(found, tier, start, eta):
    (path,) = [p for p in found.paths if p.tier == tier and p.start == start and p.eta == eta]
    return path
