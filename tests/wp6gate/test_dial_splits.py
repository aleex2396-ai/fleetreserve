"""The arithmetic of the dial measurement (`dial_splits`) on made-up losses.

1. The on-grid scale equals an independent brute force, for both tiers: the largest grid scale
   whose bound holds when each day's losses are the running maximum over the grid scales, with no
   shortcut through `crc` and no stopping at the first failure. Cases: random losses at several
   numbers of offering calibration days, among them 15 days (tier B calibrates, tier A abstains)
   and 5 days (both abstain); a hand-built dip that only the running maximum catches; calibration
   days that do not offer, which are not in n.
2. A split's offer is the scale times the mean certified value over all counting test days, so a
   split at scale 0 offers 0; the test losses are the raw losses at the scale over the offering
   test days; the share of the ceiling is a ratio of sums.
3. The summary over splits counts splits at scale 0 as an offer of 0, and its points, means and
   shares equal a direct recomputation (a hand-worked case and a random one).
4. The dial with the largest mean offer is picked, a tie goes to the smaller dial whatever the
   order given, and a dial without a mean offer is never picked.
"""

import numpy as np
import pytest

from fleetreserve import crc, dial_splits
from fleetreserve.dial_splits import GRID, SplitScore, UnitLosses

K = len(GRID)


def brute_force(slot, driver, alpha_slot, alpha_driver):
    """The largest grid scale whose bound holds, by the formula of the plan and nothing else."""
    n = len(slot)
    if n < 1 / alpha_slot - 1 - 1e-9 or n < 1 / alpha_driver - 1 - 1e-9:
        return 0.0
    best = 0.0
    for k in range(K):
        mean_slot = np.mean([max(day[: k + 1]) for day in slot])
        mean_driver = np.mean([max(day[: k + 1]) for day in driver])
        slot_ok = (n * mean_slot + 1) / (n + 1) <= alpha_slot + 1e-9
        driver_ok = (n * mean_driver + 1) / (n + 1) <= alpha_driver + 1e-9
        if slot_ok and driver_ok:
            best = GRID[k]
    return best


def random_losses(rng, n, driver_rate):
    """Losses that rise and fall between scales (so the running maximum matters)."""
    slot = rng.random((n, K)) * (rng.random((n, K)) < 0.05)
    driver = rng.random((n, K)) * (rng.random((n, K)) < driver_rate)
    return slot, driver


@pytest.mark.parametrize("n", [5, 9, 15, 19, 30, 50])
@pytest.mark.parametrize("tier", ["A", "B"])
def test_the_on_grid_scale_equals_the_brute_force(n, tier):
    alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
    nontrivial = set()
    for seed in range(40):
        rng = np.random.default_rng(1000 * n + seed)
        slot, driver = random_losses(rng, n, driver_rate=0.004 * (seed % 5))
        scale, _ = dial_splits.on_grid_scale(slot, driver, alpha_slot, alpha_driver)
        assert scale == brute_force(slot, driver, alpha_slot, alpha_driver)
        nontrivial.add(scale)
    if n >= 30:
        assert any(0 < s < 1 for s in nontrivial)  # the check is not vacuous


def test_too_few_offering_days_abstain_by_tier():
    slot, driver = np.zeros((15, K)), np.zeros((15, K))
    assert dial_splits.on_grid_scale(slot, driver, 0.10, 0.05) == (0.0, dial_splits.TOO_FEW)
    assert dial_splits.on_grid_scale(slot, driver, 0.10, 1.0) == (1.0, "")
    slot, driver = np.zeros((5, K)), np.zeros((5, K))
    assert dial_splits.on_grid_scale(slot, driver, 0.10, 1.0) == (0.0, dial_splits.TOO_FEW)


def test_a_dip_between_scales_is_not_a_safe_scale():
    # 30 days; three of them lose everything at the fourth scale and nothing at the later ones.
    # Raw sums are 0 at the later scales, but a loss does not go away as the scale grows.
    slot = np.zeros((30, K))
    slot[:3, 3] = 1.0
    driver = np.zeros((30, K))
    assert dial_splits.on_grid_scale(slot, driver, 0.10, 1.0)[0] == GRID[2]
    assert brute_force(slot, driver, 0.10, 1.0) == GRID[2]


def make_losses(rng, days=40, offering=0.8):
    offers = rng.random(days) < offering
    value = np.where(offers, 1.0 + 9.0 * rng.random(days), 0.0)
    slot = np.where(offers[:, None], rng.random((days, K)) * (rng.random((days, K)) < 0.008), 0.0)
    driver = np.where(offers[:, None], rng.random((days, K)) * (rng.random((days, K)) < 0.001), 0.0)
    ceiling = np.where(offers, value * (1.2 + rng.random(days)), 3.0 * rng.random(days))
    return UnitLosses(value, offers, slot, driver, ceiling)


def test_days_that_do_not_offer_are_not_in_n():
    rng = np.random.default_rng(5)
    losses = make_losses(rng, days=120, offering=0.5)
    calibration = np.arange(0, 60)
    test = np.arange(60, 120)
    kept = calibration[losses.offering[calibration]]
    for tier in ("A", "B"):
        found = dial_splits.score_split(losses, 7, calibration, test, tier)
        alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
        assert found.offering_calibration_days == len(kept)
        assert found.scale == brute_force(
            list(losses.slot[kept]), list(losses.driver[kept]), alpha_slot, alpha_driver
        )


def test_a_split_is_scored_at_its_scale_on_its_test_days():
    rng = np.random.default_rng(11)
    losses = make_losses(rng, days=100, offering=0.9)
    calibration, test = np.arange(0, 50), np.arange(50, 100)
    found = dial_splits.score_split(losses, 3, calibration, test, "B")
    assert found.scale > 0
    k = GRID.index(found.scale)
    offering_test = [d for d in test if losses.offering[d]]
    assert found.offer_kwh == pytest.approx(found.scale * np.mean([losses.value[d] for d in test]))
    assert found.slot_loss == pytest.approx(np.mean([losses.slot[d, k] for d in offering_test]))
    assert found.driver_loss == pytest.approx(np.mean([losses.driver[d, k] for d in offering_test]))
    offers = sum(found.scale * losses.value[d] for d in test)
    assert found.ceiling_share == pytest.approx(offers / sum(losses.ceiling[d] for d in test))
    assert found.offering_test_days == len(offering_test)


def test_scale_zero_offers_nothing_and_has_no_losses():
    losses = make_losses(np.random.default_rng(2), days=80, offering=0.9)
    # too few calibration days: tier A needs 19 offering days
    found = dial_splits.score_split(losses, 1, np.arange(0, 10), np.arange(10, 80), "A")
    assert found.scale == 0 and found.reason == dial_splits.TOO_FEW
    assert found.offer_kwh == 0.0
    assert (found.slot_loss, found.driver_loss, found.ceiling_share) == (None, None, None)


def test_a_missing_ceiling_leaves_the_share_out():
    rng = np.random.default_rng(13)
    losses = make_losses(rng, days=100, offering=1.0)
    losses.ceiling[60] = np.nan
    found = dial_splits.score_split(losses, 1, np.arange(0, 50), np.arange(50, 100), "B")
    assert found.scale > 0 and found.ceiling_share is None
    assert found.offer_kwh > 0


def score(scale, offer, slot=None, driver=None, share=None, reason=""):
    return SplitScore(1, scale, reason, 20, 10, offer, slot, driver, share)


def test_the_summary_by_hand():
    scores = [
        score(0.0, 0.0, reason=dial_splits.TOO_FEW),
        score(0.0, 0.0, reason=dial_splits.NO_SAFE_SCALE),
        score(0.5, 2.0, slot=0.05, driver=0.02, share=0.4),
        score(0.8, 4.0, slot=0.20, driver=0.08, share=0.6),
        score(1.0, 6.0, slot=0.10, driver=0.06),
    ]
    a = dial_splits.summarise(scores, "A")
    assert a["splits"] == 5
    assert a["mean_offer_kwh_per_test_day"] == pytest.approx(12.0 / 5)  # zero splits count as 0
    assert (a["offer_p10"], a["offer_p50"], a["offer_p90"]) == pytest.approx((0.0, 2.0, 5.2))
    assert a["splits_that_offer"] == 3 and a["share_of_splits_that_offer"] == pytest.approx(0.6)
    assert a["mean_scale"] == pytest.approx((0.5 + 0.8 + 1.0) / 5)
    assert a["splits_at_scale_zero_by_reason"] == {
        dial_splits.NO_SAFE_SCALE: 1,
        dial_splits.TOO_FEW: 1,
    }
    assert a["mean_test_slot_loss"] == pytest.approx((0.05 + 0.20 + 0.10) / 3)
    assert a["mean_test_driver_loss"] == pytest.approx((0.02 + 0.08 + 0.06) / 3)
    over = a["slot_loss_over_limit"]  # only 0.20 is above 0.10; 0.10 itself is at the limit
    assert (over["limit"], over["splits"]) == (0.10, 1)
    assert over["share_of_all_splits"] == pytest.approx(1 / 5)
    assert over["share_of_splits_that_offer"] == pytest.approx(1 / 3)
    over = a["driver_loss_over_limit"]  # 0.08 and 0.06 are above 0.05
    assert (over["limit"], over["splits"]) == (0.05, 2)
    assert over["share_of_all_splits"] == pytest.approx(2 / 5)
    assert a["splits_with_ceiling_share"] == 2
    assert a["mean_share_of_best_causal_ceiling"] == pytest.approx(0.5)


def test_tier_b_has_no_driver_limit():
    scores = [score(0.5, 2.0, slot=0.05, driver=0.5)]
    b = dial_splits.summarise(scores, "B")
    assert "driver_loss_over_limit" not in b
    assert b["mean_test_driver_loss"] == pytest.approx(0.5)  # reported, not limited
    assert b["slot_loss_over_limit"]["limit"] == crc.ALPHA_SLOT


def test_the_summary_equals_a_direct_recomputation():
    rng = np.random.default_rng(21)
    losses = make_losses(rng, days=110, offering=0.85)
    scores, direct = [], []
    for seed in range(1, 31):
        order = np.random.default_rng(seed).permutation(110)
        calibration, test = np.sort(order[:50]), np.sort(order[50:])
        scores.append(dial_splits.score_split(losses, seed, calibration, test, "A"))
        used = [d for d in calibration if losses.offering[d]]
        scale = brute_force(
            [list(losses.slot[d]) for d in used], [list(losses.driver[d]) for d in used], 0.10, 0.05
        )
        offer = scale * np.mean([losses.value[d] for d in test])
        direct.append((scale, offer))
    found = dial_splits.summarise(scores, "A")
    offers = np.array([o for _, o in direct])
    assert found["mean_offer_kwh_per_test_day"] == pytest.approx(offers.mean())
    assert found["offer_p10"] == pytest.approx(np.quantile(offers, 0.1))
    assert found["offer_p50"] == pytest.approx(np.quantile(offers, 0.5))
    assert found["offer_p90"] == pytest.approx(np.quantile(offers, 0.9))
    assert found["mean_scale"] == pytest.approx(np.mean([s for s, _ in direct]))
    assert found["splits_that_offer"] == sum(s > 0 for s, _ in direct)


def test_the_dial_with_the_largest_mean_offer_is_picked():
    assert dial_splits.pick_dial({0.0: 1.0, 0.03: 3.0, 0.1: 2.0}) == 0.03


def test_a_tie_goes_to_the_smaller_dial_whatever_the_order():
    offers = {0.1: 3.0, 0.03: 3.0, 0.05: 2.0}
    assert dial_splits.pick_dial(offers) == 0.03
    assert dial_splits.pick_dial(dict(reversed(list(offers.items())))) == 0.03
    near = {0.02: 3.0, 0.05: 3.0 + dial_splits.TIE / 2}  # within the tolerance: a tie
    assert dial_splits.pick_dial(near) == 0.02
    assert dial_splits.pick_dial({0.02: 3.0, 0.05: 3.0 + 1e-6}) == 0.05


def test_a_dial_without_an_offer_is_never_picked():
    assert dial_splits.pick_dial({0.0: None, 0.03: 0.0}) == 0.03
    assert dial_splits.pick_dial({0.0: None}) is None
    assert dial_splits.pick_dial({}) is None


def test_every_scale_is_at_most_one_and_from_the_plan_grid():
    assert tuple(g for g in crc.GRID if g <= 1.0) == GRID
    assert max(GRID) == 1.0 and len(GRID) == 10
