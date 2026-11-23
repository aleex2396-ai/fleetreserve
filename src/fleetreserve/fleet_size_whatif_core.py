"""Arithmetic of the fleet-size what-if (`fleet_size_whatif`): numpy only, no files, no solver.

A what-if, not a rule: the plan has no minimum number of steered cars for a day to offer. Nothing
here names a choice or a recommendation.

Input: `dial_splits.UnitLosses` of one unit under one configuration (the one-way rung's day-level
losses of every day that offers at each scale of `dial_splits.GRID`, as `dial_choice.measure`
gives them), and, per counting day, `counts`, the number of steered cars of the day (the length of
`gate_day.PreparedDay.steered`; the command's module text defines it).

`restrict(losses, counts, min_cars)`: the same unit with every day of fewer than `min_cars` steered
cars made a day that does not offer, exactly as `gate_day.evaluate` and `dial_choice.measure` leave
a day that abstains: `offering` False, certified value 0, slot and driver losses 0 at every scale.
The ceiling of the day is not touched. With `min_cars` 0 (or no day below it) the result equals the
input in every entry. The days are those of the unit, calibration and test days alike, so a day
that stops offering is also left out of the calibration it would have entered.

`by_count(losses, counts)`: one row per distinct count, in increasing order of the count (no class
edges): the counting days with that count, the offering days, the sum of the certified value at
scale 1 in kWh, and per grid scale the sums of the slot loss and of the driver loss over those days
and the numbers of days with a slot loss and with a driver loss above 0. The rows add up to the
unit's totals.

`below(losses, counts, min_cars)`: counting days and offering days with fewer than `min_cars`
steered cars.

`gate_reading(losses, counts, oracle, oracle_counts, min_cars, ceilings, splits, names)`: the
figures of the gate reading (`dial_splits.gate_split`, `dial_splits.summarise_gate`, as
`dial_choice` reads them) with the rule of `restrict` applied to the configuration by its own
counts and to its oracle partner by the partner's own counts (the partner's fleet is the cars its
own fixed plan steers; the two counts can differ). `splits`: (seed, calibration days, test days).
`differing(counts, oracle_counts)`: the number of counting days on which the two counts differ.
"""

import math

import numpy as np

from fleetreserve import dial_splits
from fleetreserve.dial_splits import UnitLosses


def _counts(losses: UnitLosses, counts) -> np.ndarray:
    counts = np.asarray(counts)
    if counts.shape != (len(losses.value),):
        raise ValueError("counts must have one entry per counting day")
    if counts.size and (counts.min() < 0 or not np.issubdtype(counts.dtype, np.integer)):
        raise ValueError("counts must be whole numbers of cars, 0 or more")
    return counts.astype(int)


def below(losses: UnitLosses, counts, min_cars: int) -> dict:
    """Counting days and offering days with fewer than `min_cars` steered cars (module text)."""
    small = _counts(losses, counts) < min_cars
    return {
        "counting_days": int(small.sum()),
        "offering_days": int((small & losses.offering).sum()),
    }


def restrict(losses: UnitLosses, counts, min_cars: int) -> UnitLosses:
    """The unit with every day of fewer than `min_cars` steered cars not offering (module text)."""
    if min_cars < 0:
        raise ValueError("min_cars must be 0 or more")
    keep = _counts(losses, counts) >= min_cars
    rows = keep[:, None]
    return UnitLosses(
        value=np.where(keep, losses.value, 0.0),
        offering=losses.offering & keep,
        slot=np.where(rows, losses.slot, 0.0),
        driver=np.where(rows, losses.driver, 0.0),
        ceiling=losses.ceiling,
    )


def _sum(values) -> float:
    return math.fsum(float(v) for v in values)


def by_count(losses: UnitLosses, counts) -> list[dict]:
    """One row per distinct count of steered cars (module text)."""
    counts = _counts(losses, counts)
    rows = []
    for count in sorted(set(counts.tolist())):
        days = np.flatnonzero(counts == count)
        scales = []
        for k, scale in enumerate(dial_splits.GRID):
            slot, driver = losses.slot[days, k], losses.driver[days, k]
            scales.append(
                {
                    "scale": float(scale),
                    "slot_loss_sum": _sum(slot),
                    "driver_loss_sum": _sum(driver),
                    "days_with_a_slot_loss": int((slot > 0).sum()),
                    "days_with_a_driver_loss": int((driver > 0).sum()),
                }
            )
        rows.append(
            {
                "steered_cars": int(count),
                "counting_days": int(len(days)),
                "offering_days": int(losses.offering[days].sum()),
                "offer_at_scale_1_kwh_sum": _sum(losses.value[days]),
                "scales": scales,
            }
        )
    return rows


def gate_reading(
    losses: UnitLosses,
    counts,
    oracle: UnitLosses,
    oracle_counts,
    min_cars: int,
    ceilings: dict,
    splits,
    names,
) -> dict:
    """The gate-reading figures of one unit and configuration under a minimum (module text).
    `ceilings`: ceiling name -> array of one entry per counting day, NaN where missing."""
    own = restrict(losses, counts, min_cars)
    partner = restrict(oracle, oracle_counts, min_cars)
    return dial_splits.summarise_gate(
        [
            dial_splits.gate_split(own, partner, ceilings, seed, cal, test)
            for seed, cal, test in splits
        ],
        names,
    )


def differing(counts, oracle_counts) -> int:
    """The number of counting days on which the two counts of steered cars differ."""
    a, b = np.asarray(counts), np.asarray(oracle_counts)
    if a.shape != b.shape:
        raise ValueError("the two counts must have one entry per counting day")
    return int((a != b).sum())
