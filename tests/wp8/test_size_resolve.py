"""The calibration and scoring of sized days on the re-solve's losses (`size_resolve`), on made-up
loss tables: no day is replayed here; a table maps (day, scale) to (slot loss, driver loss).

1. Relative sizes: a hand case; no offering day; a size of 0 refused.
2. The pairs a sizing needs: every grid point on every offering day, at the scale the scorer reads.
3. A hand case small enough to compute by hand: ten calibration days, half of which miss a request
   above scale 0.5 and half above 1; tier B. Size 1 calibrates to 0.5; sizes equal to the days'
   thresholds calibrate to 1, and the offer on two test days (one of each kind, value 2 kWh) is 1.5
   kWh per day against 1.0.
4. With size 1 on every offering day the scalar, the offer and the test losses equal
   `dial_splits.score_split` on the same losses, for random tables, splits and both tiers.
5. A day that does not offer under the sizing offers 0, counts as a counting test day and is in no
   loss mean; at scalar 0 there are no losses.
6. A day's losses are read at the scalar times its relative size, not at the scalar alone.
"""

import math

import numpy as np
import pytest

from fleetreserve import dial_splits, size_resolve
from fleetreserve.size_resolve import GRID, sized_scale

NAN = math.nan


def table(thresholds, driver=None) -> dict:
    """Every scale a test can ask for: slot loss 1 above the day's threshold, else 0."""
    out = {}
    for d, t in enumerate(thresholds):
        for r in (1.0, 0.5, 0.25, 0.05, *np.linspace(0.05, 1.0, 20)):
            for g in GRID:
                s = sized_scale(g, r)
                out[(d, s)] = (1.0 if s > t else 0.0, 0.0 if driver is None else driver(d, s))
    return out


# --- 1. and 2. ---------------------------------------------------------------------------------


def test_relative_sizes():
    found = size_resolve.relative([2.0, NAN, 1.0, 0.5])
    assert found[0] == 1.0 and math.isnan(found[1]) and found[2] == 0.5 and found[3] == 0.25
    assert np.isnan(size_resolve.relative([NAN, NAN])).all()
    with pytest.raises(ValueError):
        size_resolve.relative([0.0, NAN])


def test_the_pairs_are_every_grid_point_of_every_offering_day():
    found = size_resolve.wanted([1.0, NAN, 0.5])
    assert found == {(0, g) for g in GRID} | {(2, sized_scale(g, 0.5)) for g in GRID}
    assert len(found) == 2 * len(GRID)


# --- 3. the hand case --------------------------------------------------------------------------


def test_a_hand_case():
    thresholds = [0.5] * 5 + [1.0] * 5 + [0.5, 1.0]  # days 10 and 11 are the test days
    losses = table(thresholds)
    value = np.full(12, 2.0)
    calibration, test = list(range(10)), [10, 11]
    constant = np.ones(12)
    scalar = size_resolve.calibrate(losses, constant, calibration, "B")
    assert scalar.used == 0.5 and scalar.reason == "" and scalar.days == 10
    found = size_resolve.score(losses, constant, value, scalar.used, test)
    assert found.offers_kwh == (1.0, 1.0) and found.slot_losses == (0.0, 0.0)
    sized = size_resolve.relative(np.array(thresholds))
    scalar = size_resolve.calibrate(losses, sized, calibration, "B")
    assert scalar.used == 1.0
    found = size_resolve.score(losses, sized, value, scalar.used, test)
    assert found.offers_kwh == (1.0, 2.0) and found.mean_offer_kwh == 1.5
    assert found.slot_losses == (0.0, 0.0)


# --- 4. size 1 is dial_splits --------------------------------------------------------------------


def random_case(seed: int, days: int = 60):
    rng = np.random.default_rng(seed)
    slot = np.where(
        rng.random((days, len(GRID))) < 0.08, rng.choice([0.5, 1.0], (days, len(GRID))), 0
    )
    driver = np.where(rng.random((days, len(GRID))) < 0.05, rng.random((days, len(GRID))), 0.0)
    offering = rng.random(days) < 0.85
    value = np.where(offering, rng.uniform(1.0, 9.0, days), 0.0)
    slot[~offering], driver[~offering] = 0.0, 0.0
    losses = {
        (d, sized_scale(g, 1.0)): (float(slot[d, k]), float(driver[d, k]))
        for d in range(days)
        for k, g in enumerate(GRID)
    }
    unit = dial_splits.UnitLosses(value, offering, slot, driver, np.full(days, np.nan))
    return losses, unit, np.where(offering, 1.0, NAN), value


@pytest.mark.parametrize("seed", range(6))
def test_size_one_is_dial_splits(seed):
    losses, unit, q, value = random_case(seed)
    rng = np.random.default_rng(100 + seed)
    for _ in range(5):
        order = rng.permutation(len(value))
        calibration, test = sorted(order[:30]), sorted(order[30:])
        for tier in dial_splits.TIERS:
            expected = dial_splits.score_split(unit, 0, calibration, test, tier)
            scalar = size_resolve.calibrate(losses, q, calibration, tier)
            found = size_resolve.score(losses, q, value, scalar.used, test)
            assert scalar.used == expected.scale and scalar.reason == expected.reason
            assert found.mean_offer_kwh == pytest.approx(expected.offer_kwh, abs=1e-12)
            assert found.offering_days == expected.offering_test_days
            if expected.slot_loss is None:
                assert found.slot_losses == ()
            else:
                assert np.mean(found.slot_losses) == pytest.approx(expected.slot_loss, abs=1e-12)
                assert np.mean(found.driver_losses) == pytest.approx(
                    expected.driver_loss, abs=1e-12
                )


# --- 5. and 6. ---------------------------------------------------------------------------------


def test_a_day_without_a_size_offers_nothing_and_has_no_loss():
    losses = table([1.0, 1.0, 0.0])
    r = np.array([1.0, NAN, 1.0])
    found = size_resolve.score(losses, r, [2.0, 3.0, 4.0], 0.5, [0, 1, 2])
    assert found.offers_kwh == (1.0, 0.0, 2.0) and found.offering_days == 2
    assert found.slot_losses == (0.0, 1.0)  # day 2 misses a request above scale 0
    at_zero = size_resolve.score(losses, r, [2.0, 3.0, 4.0], 0.0, [0, 1, 2])
    assert at_zero.offers_kwh == (0.0, 0.0, 0.0) and at_zero.slot_losses == ()


def test_losses_are_read_at_the_scalar_times_the_size():
    losses = table([0.3])  # loses above 0.3
    half = size_resolve.score(losses, np.array([0.5]), [1.0], 0.6, [0])
    full = size_resolve.score(losses, np.array([1.0]), [1.0], 0.6, [0])
    assert half.slot_losses == (0.0,) and full.slot_losses == (1.0,)
    assert half.offers_kwh == (pytest.approx(0.3),) and full.offers_kwh == (0.6,)
