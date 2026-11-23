"""The scale of a resample (`day_bootstrap.scale_of`, reading I3) against the run's own
calibration, `crc.calibrate_search`, with every scale known.

1. The bracket. On 300 seeded made-up units of 9 to 40 offering days whose raw slot and driver
   losses are known at every scale (a step up at a random threshold, and on some days a bump that
   comes and goes, so the raw losses are not monotone in the scale), the full search is run with
   every loss known. The stored scales are the grid and a random part of the scales the full
   search asked for (none, some or all). Then `low` <= the full search's scale <= `high`, and a
   resample marked exact gives the full search's scale itself, and one that is not is marked as
   having met a pair not held. Both tiers. Some units must be exact and some not, or the test
   shows nothing. The same with every scale the full search asked for stored but some of its
   (day, scale) pairs marked as failed (their losses unknown): the bracket holds, and a resample
   that is not exact is marked as having met a failed pair (its search leaves the full search's
   path only at such a pair; the two bounding searches may then ask scales not stored here, which
   the scoring run evaluates, `day_resample`).
2. By hand: 10 days lose nothing at the stored scales up to 0.5 and everything from 0.6 on, and
   only the grid is stored. Tier B (no day may lose on 10 days) holds up to 0.5 and fails at 0.6;
   the halving from 0.5 to 0.6 asks 0.55, not stored. With it failing the search ends at 0.5; with
   it holding (and every later halving) it climbs to 0.6 - 0.1 / 128 = 0.59921875, seven halvings
   to a width under 1e-3. With 0.55, 0.575, ... stored as losing nothing the search climbs exactly
   to 0.59921875.
3. Too few days: 8 days in tier B, 18 in tier A abstain exactly (scale 0, marked too few).
"""

import numpy as np
import pytest

from fleetreserve import crc, day_bootstrap
from fleetreserve.dial_splits import tier_alphas


def made_up_unit(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(9, 41))
    step_slot = rng.uniform(0.0, 2.5, n)
    step_driver = rng.uniform(0.0, 2.5, n)
    driver_level = rng.uniform(0.0, 1.0, n)
    bump_from = rng.uniform(0.0, 2.0, n)
    bump_to = bump_from + rng.uniform(0.0, 0.5, n)
    bumped = rng.uniform(size=n) < 0.3

    def loss_at(beta):
        bump = bumped & (bump_from < beta) & (beta < bump_to)
        slot = np.where((beta > step_slot) | bump, 1.0, 0.0)
        driver = np.where(beta > step_driver, driver_level, 0.0)
        return slot, driver

    return rng, n, loss_at


def full_search(loss_at, n, tier):
    asked = []

    def recorded(beta):
        asked.append(beta)
        return loss_at(beta)

    return crc.calibrate_search(recorded, n, *tier_alphas(tier)), asked


def stored_losses(loss_at, scales):
    columns = [loss_at(b) for b in scales]
    slot = np.stack([c[0] for c in columns], axis=1)
    driver = np.stack([c[1] for c in columns], axis=1)
    return slot, driver


def test_the_bracket_holds_the_full_search():
    exact = inexact = 0
    for seed in range(300):
        rng, n, loss_at = made_up_unit(seed)
        tier = "A" if seed % 2 else "B"
        full, asked = full_search(loss_at, n, tier)
        share = (0.0, 0.5, 1.0)[seed % 3]
        kept = [b for b in asked if rng.uniform() < share]
        scales = sorted({*crc.GRID, *kept})
        slot, driver = stored_losses(loss_at, scales)
        found = day_bootstrap.scale_of(slot, driver, scales, *tier_alphas(tier))
        assert found.low <= full.beta <= found.high, (seed, found, full.beta)
        if found.exact:
            exact += 1
            assert found.low == full.beta == found.high
        else:
            inexact += 1
            assert found.low < found.high and found.missing_pair and not found.failed_pair
    assert exact > 50 and inexact > 50


def test_the_bracket_holds_where_pairs_failed():
    exact = inexact = 0
    for seed in range(300):
        rng, n, loss_at = made_up_unit(seed)
        tier = "A" if seed % 2 else "B"
        full, asked = full_search(loss_at, n, tier)
        scales = sorted({*crc.GRID, *asked})
        slot, driver = stored_losses(loss_at, scales)
        failed = rng.uniform(size=slot.shape) < (0.0, 0.02, 0.1)[seed % 3]
        slot[failed] = np.nan
        driver[failed] = np.nan
        found = day_bootstrap.scale_of(slot, driver, scales, *tier_alphas(tier), failed)
        assert found.low <= full.beta <= found.high, (seed, found, full.beta)
        if found.exact:
            exact += 1
            assert found.low == full.beta == found.high and not found.failed_pair
        else:
            inexact += 1
            assert found.failed_pair
    assert exact > 50 and inexact > 20


def grid_only_losses():
    scales = list(crc.GRID)
    row = [0.0 if b <= 0.5 else 1.0 for b in scales]
    return np.array([row] * 10), np.zeros((10, len(scales))), scales


def test_a_bracket_by_hand():
    slot, driver, scales = grid_only_losses()
    found = day_bootstrap.scale_of(slot, driver, scales, *tier_alphas("B"))
    assert (found.low, found.exact, found.too_few) == (0.5, False, False)
    assert found.high == pytest.approx(0.59921875, abs=1e-12)
    halvings, holding = [], 0.5
    for _ in range(7):  # the search's own halving, (holding + failing) / 2, each one holding
        holding = (holding + 0.6) / 2.0
        halvings.append(holding)
    more = sorted({*scales, *halvings})
    rows = np.array([[0.0 if b < 0.6 else 1.0 for b in more]] * 10)
    stored = day_bootstrap.scale_of(rows, np.zeros_like(rows), more, *tier_alphas("B"))
    assert stored.exact and stored.low == stored.high == pytest.approx(0.59921875, abs=1e-12)


@pytest.mark.parametrize(("tier", "days"), [("B", 8), ("A", 18)])
def test_too_few_days_abstain_exactly(tier, days):
    scales = list(crc.GRID)
    zeros = np.zeros((days, len(scales)))
    found = day_bootstrap.scale_of(zeros, zeros, scales, *tier_alphas(tier))
    assert found == day_bootstrap.Scale(0.0, 0.0, True, True)
    enough = np.zeros((days + 1, len(scales)))
    assert day_bootstrap.scale_of(enough, enough, scales, *tier_alphas(tier)).low == 2.0
