"""Per-day sizes and the calibrated one-way re-solve offer they give, on the audited paths (plan
sections 6 (T6), 7 (T7); the route `dial_choice` and `compare_run` read).

A sizing gives each counting day d of a unit a size `q_d` (NaN on a day that does not offer under
it, `size_scale.sizes`). The day's scale is `gamma * r_d` with `r_d = q_d / max q` (`relative`),
the maximum over the unit's days that offer under the sizing (on one split, for a sizing fitted
per split), and one scalar `gamma` calibrated over the split's calibration days. Dividing by the
largest size is `size_scale`'s cap seen from the other side: it reads the sizes alone, and it keeps
every day's scale at or below 1 (the stay-to-bound property is claimed up to 1 only, plan section
6) while `gamma` runs over the same scales as the constant sizing's. With `q_d = 1` on every
offering day (`constant`), `r_d = 1` and everything below is `dial_splits` unchanged.

Losses (reused, no new replay): the loss of a day at scale `gamma * r_d` is the loss of the unsized
day at the scale `gamma * r_d` (the box scaled by beta about its midpoint is the same box whichever
way beta is written), so it is `gate_day.evaluate(prepared, gamma * r_d)` read at the one-way rung
(`dial_choice.RUNG`): the day-level slot and driver loss, the largest over the day's audited paths.
`losses` below maps (day, scale) to that pair; `wanted` lists the pairs a sizing needs, and the
scale is always written `sized_scale(gamma, r)` so that equal pairs are equal keys.

On-grid calibration (`calibrate`), per split and tier: `gamma` runs over `GRID`
(`dial_splits.GRID`, the gate's grid up to 1), and the calibration is `dial_splits.on_grid_scale`
on the split's calibration days that offer under the sizing, with day d's losses at the grid point
`gamma` read at `sized_scale(gamma, r_d)`: each day's losses are made a running maximum over its
own grid points, the scalar is the largest grid point at which the bound of `crc` holds (tier A:
slot loss at alpha 0.10 and driver loss at alpha 0.05; tier B: the slot loss alone), and it is 0
(abstain) with too few such days or when the bound fails at the smallest grid point. A reading:
`gamma` on the grid, not bisected, because every scalar tried on every split is a new scale for
every day (alternative: `crc.calibrate_search` per split, which needs evaluations per split).

Test scoring (`score`): the offer per counting test day is the mean over the test days of
`gamma * r_d * value_d` (0 on a day that does not offer under the sizing, and at `gamma` 0); the
losses are the day-level losses at `sized_scale(gamma, r_d)`, not the running maximum (as
`dial_splits.score_split` scores them), over the test days that offer under the sizing, and none at
`gamma` 0. The result is a `size_scale.TestScore`, so the paired differences and pooling of
`size_whatif_summary` apply unchanged.

Guarantee. T6's bound covers a sizing whose sizes are fixed before the split's calibration days
are seen. A sizing fitted on the calibration days themselves (`size_resolve_sizings.type_sizes`)
or read from the day's own outcome (the oracle sizings) is outside it; its test losses are
reported beside its offer so that this shows.
"""

import math

import numpy as np

from fleetreserve import dial_splits
from fleetreserve.size_scale import Scalar, TestScore

GRID = dial_splits.GRID


def relative(q) -> np.ndarray:
    """`q` over its largest value on the days where it is not NaN; all NaN when none is."""
    q = np.asarray(q, dtype=float)
    kept = ~np.isnan(q)
    out = np.full(q.shape, math.nan)
    if kept.any():
        top = float(q[kept].max())
        if not top > 0:
            raise ValueError("a size must be above 0")
        out[kept] = q[kept] / top
    return out


def sized_scale(gamma: float, r: float) -> float:
    """The scale of a day of relative size `r` at the scalar `gamma` (one way to write it)."""
    return float(gamma) * float(r)


def wanted(r, grid=GRID) -> set[tuple[int, float]]:
    """The (day, scale) pairs a sizing of relative sizes `r` needs: every grid point on every day
    that offers under it."""
    r = np.asarray(r, dtype=float)
    return {(int(d), sized_scale(g, r[d])) for d in np.flatnonzero(~np.isnan(r)) for g in grid}


def day_matrix(losses: dict, r, days, grid=GRID) -> tuple[np.ndarray, np.ndarray]:
    """(slot, driver), each of shape (len(days), len(grid)): day d's losses at each grid point."""
    found = [[losses[(int(d), sized_scale(g, r[d]))] for g in grid] for d in days]
    pairs = np.asarray(found, dtype=float).reshape(len(days), len(grid), 2)
    return pairs[:, :, 0], pairs[:, :, 1]


def calibrate(losses: dict, r, calibration, tier: str, grid=GRID) -> Scalar:
    """The scalar of `tier` on the calibration days `calibration` (indices) that offer under `r`."""
    r = np.asarray(r, dtype=float)
    days = [d for d in calibration if not math.isnan(r[d])]
    slot, driver = day_matrix(losses, r, days, grid)
    alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
    scale, reason = dial_splits.on_grid_scale(slot, driver, alpha_slot, alpha_driver, grid)
    return Scalar(scale, scale, reason, len(days))


def score(losses: dict, r, value, gamma: float, test) -> TestScore:
    """The test days `test` (indices) at the scalar `gamma` (module text)."""
    r = np.asarray(r, dtype=float)
    value = np.asarray(value, dtype=float)
    offers = [0.0 if math.isnan(r[d]) else float(gamma * r[d] * value[d]) for d in test]
    offering = [d for d in test if not math.isnan(r[d])]
    slot, driver = [], []
    for d in offering if gamma > 0 else []:
        found = losses[(int(d), sized_scale(gamma, r[d]))]
        slot.append(float(found[0]))
        driver.append(float(found[1]))
    return TestScore(tuple(offers), len(offering), tuple(slot), tuple(driver))


def measure(losses: dict, value, r_of, splits, tiers=dial_splits.TIERS) -> dict:
    """Per tier the list of (seed, `Scalar`, `TestScore`) over `splits` (seed, calibration, test).
    `r_of(seed, tier)` gives the relative sizes of the sizing on that split and tier."""
    out = {tier: [] for tier in tiers}
    for seed, calibration, test in splits:
        for tier in tiers:
            r = r_of(seed, tier)
            scalar = calibrate(losses, r, calibration, tier)
            out[tier].append((seed, scalar, score(losses, r, value, scalar.used, test)))
    return out
