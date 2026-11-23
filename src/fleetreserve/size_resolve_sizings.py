"""The two sizings of the re-solve what-if that are read from the one-way re-solve's own losses at
size 1 (plan section 7, T7 and the day-size net; `size_resolve` scores them). Both read the
losses `losses[(day, scale)]` = (slot, driver) of `size_resolve` at the grid points of `GRID` with
size 1, the constant sizing's losses. Each returns raw sizes (NaN on a day that does not offer),
which `size_scale.sizes` turns into sizes in the arms "scale" and "abstain".

Re-solve oracle (`failure_free_sizes`; a reference, not a method: it reads the day's own
outcome). The raw size of an offering day is the largest grid point up to which the day's slot
loss is 0 at every grid point (no request missed on any audited path), 0 when it is above 0 at the
smallest grid point: the grid's reading of the strict variant's "s_d = sup{beta : day d has no
failure}" (plan section 6), on the re-solve. The same in both tiers. The driver loss is not read:
it comes mostly from the box's centre, which does not scale (plan section 6), so a size cannot
remove it, and tier A's scalar limits it. A reading, added beside the two oracles of `size_whatif`
(the day's exact box-scale label, not read from the re-solve's losses). (Alternative: zero slot
and driver loss, which in tier A sizes every day with any driver loss at the floor.)

Per-day-type multiplier (`type_sizes`), fitted per split and tier on the split's calibration days
only. A reading of "the per-day-type multiplier under one calibrated scalar" (the owner's rule of
15 Nov 2026) and of T7 ("tested per day type"):

1. Day types as `t7_test` defines them: `t7_stat.raw_type` of the day's date and public-holiday
   flag (its weekday, or "public holiday"), pooled by `t7_stat.pool_types`, here over the split's
   calibration days that offer: a raw type with fewer than `t7_stat.MIN_TYPE_DAYS` (10) of them is
   `t7_stat.OTHER`. A test day takes its raw type when that type has at least 10 such calibration
   days, else `OTHER`. (Alternative: pool over all the unit's offering days, as `t7_unit` does;
   that reads only dates, but the counts of test days enter it.)
2. Per type, its scale: the largest grid point at which the mean, over the type's offering
   calibration days, of each loss the tier limits (each day's losses a running maximum over the
   grid, as `dial_splits.on_grid_scale` makes them) is at most the tier's alpha (with
   `crc.TOLERANCE`), 0 when the means are above it at the smallest grid point. It is the plain
   mean, without the finite-sample term of `crc` (most types have far fewer than the 19 days that
   term needs at alpha 0.05); the one calibrated scalar on top (`size_resolve.calibrate`) applies
   that term over all the calibration days. A type of `OTHER` with no calibration day takes the
   scale of all the offering calibration days together. (Alternative: T7's Lagrangian, a scale per
   type that maximises value minus a common multiplier times the loss, equalising marginal loss
   per unit value instead of the loss level.)
3. The raw size of an offering day is its type's scale. No calibration day offers: every raw size
   is 0.

Both sizings read outcomes the guarantee does not allow a sizing to read (`size_resolve`'s text):
the oracle a day's own, the multiplier its calibration days', on which the one scalar is then
calibrated again.
"""

import math
from collections import Counter

import numpy as np

from fleetreserve import crc, dial_splits, t7_stat
from fleetreserve.size_resolve import GRID, day_matrix


def failure_free_sizes(losses: dict, offering, grid=GRID) -> np.ndarray:
    """The re-solve oracle's raw sizes (module text). `offering`: one bool per day."""
    offering = np.asarray(offering, dtype=bool)
    raw = np.full(offering.shape, math.nan)
    days = np.flatnonzero(offering)
    slot, _ = day_matrix(losses, np.ones(offering.shape), days, grid)
    for row, d in enumerate(days):
        clean = np.flatnonzero(np.cumsum(slot[row] > 0) == 0)
        raw[d] = float(grid[clean[-1]]) if clean.size else 0.0
    return raw


def mean_rule_scale(slot, driver, alpha_slot: float, alpha_driver: float, grid=GRID) -> float:
    """The largest grid point at which both mean losses (each day's a running maximum over the
    grid) are within their alphas; 0 when none is, or with no day (module text, step 2)."""
    slot = np.asarray(slot, dtype=float).reshape(-1, len(grid))
    driver = np.asarray(driver, dtype=float).reshape(-1, len(grid))
    if slot.shape[0] == 0:
        return 0.0
    slot_means = np.maximum.accumulate(slot, axis=1).mean(axis=0)
    driver_means = np.maximum.accumulate(driver, axis=1).mean(axis=0)
    scale = 0.0
    for k, g in enumerate(grid):
        if slot_means[k] > alpha_slot + crc.TOLERANCE:
            break
        if driver_means[k] > alpha_driver + crc.TOLERANCE:
            break
        scale = float(g)
    return scale


def day_types(raw_types, offering, calibration) -> list:
    """Step 1 of the module text: the type of each offering day (None on the others)."""
    offering = np.asarray(offering, dtype=bool)
    counts = Counter(raw_types[d] for d in calibration if offering[d])
    kept = {t for t, n in counts.items() if n >= t7_stat.MIN_TYPE_DAYS}
    return [
        (raw_types[d] if raw_types[d] in kept else t7_stat.OTHER) if offering[d] else None
        for d in range(len(offering))
    ]


def type_sizes(losses: dict, raw_types, offering, calibration, tier: str, grid=GRID):
    """The multiplier's raw sizes on one split and tier (module text), and the scale per type."""
    offering = np.asarray(offering, dtype=bool)
    types = day_types(raw_types, offering, calibration)
    alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
    ones = np.ones(offering.shape)
    used = [d for d in calibration if offering[d]]

    def scale_of(days) -> float:
        slot, driver = day_matrix(losses, ones, days, grid)
        return mean_rule_scale(slot, driver, alpha_slot, alpha_driver, grid)

    scales = {}
    for name in sorted({t for t in types if t is not None}):
        days = [d for d in used if types[d] == name]
        scales[name] = scale_of(days if days else used)
    raw = np.full(offering.shape, math.nan)
    for d in np.flatnonzero(offering):
        raw[d] = scales[types[d]]
    return raw, scales
