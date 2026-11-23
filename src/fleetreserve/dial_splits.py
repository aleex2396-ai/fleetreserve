"""Calibration on a grid over many seeded splits of one unit, and what it gives (plan section 6).

The dial of the driver-risk rows is fixed, per site type, as the value of plan section 2 with the
largest mean calibrated re-solve offer over development splits. This module is the arithmetic of
that measurement; `dial_choice` is the command that feeds it. Everything here is pure (numpy
only, no files, no solver, no randomness).

Input, `UnitLosses`: one unit under one configuration. Per counting day: the certified value at
scale 1, whether the day offers, and the one-way rung's slot loss and driver loss at each scale
of `GRID` (the day-level losses of `gate_day.evaluate`: the largest over the audited paths). Days
that abstain have value 0 and no loss. A day's ceiling is NaN when it is missing.

`GRID` is `crc.GRID` up to and including 1: the stay-to-bound property (plan section 6, tier B)
is claimed up to a scale of 1 only, so no scale above 1 is used here.

On-grid calibration (`on_grid_scale`), for one split and one tier. The calibration days are the
split's calibration days that offer. Their losses are made a running maximum over the grid scales,
as `crc.calibrate_search` makes them over the scales it evaluates. The scale is the largest grid
scale at which the bound of `crc` holds (`crc.bound_holds`) for both losses together, each at its
own alpha; the monotonised mean losses cannot fall as the scale grows, so the scales where the
bound holds form a prefix of the grid and the first failure ends the search, as in
`crc.calibrate_search`. The scale is 0 (abstain) when the offering calibration days are too few
for either alpha (`crc.enough_days`) or when the bound fails already at the smallest grid scale.
Tier A (driver-protected): slot loss at `crc.ALPHA_SLOT` and driver loss at `crc.ALPHA_DRIVER`.
Tier B (deadline contract): the slot loss alone; the driver alpha is 1, where the bound always
holds, as `gate_calibrate` sets it for its slot-only scale. A `UnitLosses` with a `rule` (the
sizing methods of `compare_run`, `chance_sizing.ChanceRule`) takes the scale from that rule in
place of `on_grid_scale`: it is given the same offering calibration days' slot and driver losses,
their day indices, the tier's two alphas and the grid, and returns a grid scale or 0 with its
reason (a rule may solve programmes; this module solves none). Without a rule (None, the
default) the scale is `on_grid_scale`'s.

Scoring (`score_split`), on the split's test days at the chosen scale: the offer per counting test
day is the scale times the mean certified value (0 on abstaining days, so 0 at scale 0); the test
losses are the raw day-level losses at that scale (not the running maximum, as the gate scores its
test days) averaged over the offering test days, and not defined (None) at scale 0 or when no test
day offers; the share of the best-causal ceiling is the sum of the offers over the sum of the
ceilings of the test days (None when a ceiling is missing or the sum is not positive).

Summary over splits (`summarise`): splits at scale 0 count as an offer of 0 in the mean and the
points. Means of the test losses are over the splits with a defined loss. "Over the limit" is a
mean test loss above its alpha (with `crc.TOLERANCE`).

Gate reading (`gate_split`, `summarise_gate`), tier A, for one unit under one configuration and
its oracle partner. The plan reads the gate on the mean calibrated re-solve offer over the
development splits: per split, the calibrated offer per counting test day (`score_split`, a split
with no safe scale offers 0), the same for the oracle partner (its own losses, calibrated by the
same rule on the same split), and the mean over the split's counting test days of each ceiling
(NaN where a day's ceiling is missing, a timeout; the split's ceiling is then missing). Over all
splits: the mean offer, the mean oracle offer and their ratio (oracle over calibrated; not a number
when the mean offer is 0). Per ceiling: the mean ceiling and the share of it, over the splits whose
ceiling is complete (a split with a missing ceiling is left out of the share and counted; a missing
ceiling is never read as 0), as mean offer on those splits over their mean ceiling (not a number
when the mean ceiling is not positive). No verdict is formed here.

Dial choice (`pick_dial`): the largest mean offer; a difference of at most `TIE` kWh is a tie, and
a tie goes to the smaller dial.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fleetreserve import crc

GRID = tuple(g for g in crc.GRID if g <= 1.0)
TIERS = ("A", "B")
TIE = 1e-9  # kWh; a smaller difference between two mean offers is a tie
QUANTILES = (0.1, 0.5, 0.9)
TOO_FEW = "too few offering calibration days"
NO_SAFE_SCALE = "the bound fails at every grid scale"


def tier_alphas(tier: str) -> tuple[float, float]:
    """(alpha for the slot loss, alpha for the driver loss) of a tier (module text)."""
    if tier == "A":
        return crc.ALPHA_SLOT, crc.ALPHA_DRIVER
    if tier == "B":
        return crc.ALPHA_SLOT, 1.0
    raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")


@dataclass(frozen=True)
class UnitLosses:
    """One unit under one configuration (module text). Shapes: (days,) and (days, len(GRID))."""

    value: np.ndarray
    offering: np.ndarray
    slot: np.ndarray
    driver: np.ndarray
    ceiling: np.ndarray
    rule: Callable | None = None  # the scale rule (module text); None: `on_grid_scale`

    def __post_init__(self) -> None:
        days = len(self.value)
        shape = (days, len(GRID))
        if self.slot.shape != shape or self.driver.shape != shape:
            raise ValueError(f"slot and driver losses must have shape {shape}")
        if len(self.offering) != days or len(self.ceiling) != days:
            raise ValueError("value, offering and ceiling must have one entry per day")


def on_grid_scale(
    slot, driver, alpha_slot: float, alpha_driver: float, grid=GRID
) -> tuple[float, str]:
    """The largest grid scale at which the bound holds for both losses, and why it is 0 if it is
    (module text). `slot`, `driver`: arrays of shape (calibration days that offer, len(grid))."""
    slot = np.asarray(slot, dtype=float).reshape(-1, len(grid))
    driver = np.asarray(driver, dtype=float).reshape(-1, len(grid))
    n = slot.shape[0]
    if not (crc.enough_days(n, alpha_slot) and crc.enough_days(n, alpha_driver)):
        return 0.0, TOO_FEW
    slot_sums = np.maximum.accumulate(slot, axis=1).sum(axis=0)
    driver_sums = np.maximum.accumulate(driver, axis=1).sum(axis=0)
    scale = 0.0
    for k, g in enumerate(grid):
        holds = crc.bound_holds(float(slot_sums[k]), n, alpha_slot) and crc.bound_holds(
            float(driver_sums[k]), n, alpha_driver
        )
        if not holds:
            break
        scale = float(g)
    return scale, ("" if scale > 0 else NO_SAFE_SCALE)


@dataclass(frozen=True)
class SplitScore:
    """One split under one tier."""

    seed: int
    scale: float
    reason: str  # why the scale is 0, empty otherwise
    offering_calibration_days: int
    offering_test_days: int
    offer_kwh: float  # per counting test day
    slot_loss: float | None
    driver_loss: float | None
    ceiling_share: float | None


def score_split(
    losses: UnitLosses, seed: int, calibration, test, tier: str, grid=GRID
) -> SplitScore:
    """Calibrate on the split's calibration days and score its test days (module text).
    `calibration`, `test`: day indices."""
    calibration = np.asarray(calibration, dtype=int)
    test = np.asarray(test, dtype=int)
    alpha_slot, alpha_driver = tier_alphas(tier)
    used = calibration[losses.offering[calibration]]
    if losses.rule is None:
        scale, reason = on_grid_scale(
            losses.slot[used], losses.driver[used], alpha_slot, alpha_driver, grid
        )
    else:
        scale, reason = losses.rule(
            losses.slot[used], losses.driver[used], used, alpha_slot, alpha_driver, grid
        )
    offering_test = test[losses.offering[test]]
    offers = scale * losses.value[test]
    slot = driver = share = None
    if scale > 0:
        k = list(grid).index(scale)
        if len(offering_test):
            slot = float(losses.slot[offering_test, k].mean())
            driver = float(losses.driver[offering_test, k].mean())
        ceilings = losses.ceiling[test]
        if not np.isnan(ceilings).any() and ceilings.sum() > 0:
            share = float(offers.sum() / ceilings.sum())
    return SplitScore(
        seed,
        scale,
        reason,
        len(used),
        len(offering_test),
        float(offers.mean()) if len(test) else 0.0,
        slot,
        driver,
        share,
    )


@dataclass(frozen=True)
class GateSplit:
    """One split read as the gate reads it (module text). Offers and ceilings are kWh per counting
    test day; `ceiling_kwh[name]` is None when a test day's ceiling is missing."""

    seed: int
    offer_kwh: float
    oracle_offer_kwh: float
    ceiling_kwh: dict


def gate_split(
    losses: UnitLosses,
    oracle: UnitLosses,
    ceilings: dict,
    seed: int,
    calibration,
    test,
    tier: str = "A",
    grid=GRID,
) -> GateSplit:
    """The calibrated offer, the oracle partner's offer and the mean ceilings of one split.
    `ceilings`: ceiling name -> array of one entry per counting day, NaN where missing."""
    if len(oracle.value) != len(losses.value):
        raise ValueError("the unit and its oracle partner must have the same days")
    test = np.asarray(test, dtype=int)
    calibrated = score_split(losses, seed, calibration, test, tier, grid)
    partner = score_split(oracle, seed, calibration, test, tier, grid)
    found = {}
    for name, per_day in ceilings.items():
        values = np.asarray(per_day, dtype=float)[test]
        found[name] = None if np.isnan(values).any() or not len(values) else float(values.mean())
    return GateSplit(seed, calibrated.offer_kwh, partner.offer_kwh, found)


def summarise_gate(splits: list[GateSplit], ceiling_names) -> dict:
    """The figures of the gate reading over all the splits of one unit and configuration."""
    n = len(splits)
    offer = _mean([s.offer_kwh for s in splits])
    oracle = _mean([s.oracle_offer_kwh for s in splits])
    by_ceiling = {}
    for name in ceiling_names:
        kept = [s for s in splits if s.ceiling_kwh[name] is not None]
        mean_ceiling = _mean([s.ceiling_kwh[name] for s in kept])
        offer_there = _mean([s.offer_kwh for s in kept])
        usable = mean_ceiling is not None and mean_ceiling > 0
        by_ceiling[name] = {
            "splits_with_ceiling": len(kept),
            "splits_left_out_ceiling_missing": n - len(kept),
            "mean_ceiling_kwh_per_test_day": mean_ceiling,
            "mean_offer_kwh_per_test_day_on_those_splits": offer_there,
            "share_of_ceiling": offer_there / mean_ceiling if usable else None,
        }
    return {
        "tier": "A",
        "splits": n,
        "mean_offer_kwh_per_test_day": offer,
        "mean_oracle_offer_kwh_per_test_day": oracle,
        "oracle_to_offer_ratio": None if not offer else oracle / offer,
        "oracle_to_offer_ratio_note": (
            "not a number: the mean offer is 0" if not offer else "oracle mean over offer mean"
        ),
        "ceilings": by_ceiling,
    }


def _mean(values) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _over(losses: list, limit: float) -> int:
    return sum(1 for loss in losses if loss > limit + crc.TOLERANCE)


def summarise(scores: list[SplitScore], tier: str) -> dict:
    """The summary of one unit, configuration and tier over its splits (module text)."""
    alpha_slot, alpha_driver = tier_alphas(tier)
    n = len(scores)
    offers = np.array([s.offer_kwh for s in scores])
    p10, p50, p90 = (float(q) for q in np.quantile(offers, QUANTILES)) if n else (None,) * 3
    offering = [s for s in scores if s.scale > 0]
    slot = [s.slot_loss for s in offering if s.slot_loss is not None]
    driver = [s.driver_loss for s in offering if s.driver_loss is not None]
    shares = [s.ceiling_share for s in offering if s.ceiling_share is not None]
    reasons: dict[str, int] = {}
    for s in scores:
        if s.scale <= 0:
            reasons[s.reason] = reasons.get(s.reason, 0) + 1

    def over(losses: list, limit: float) -> dict:
        count = _over(losses, limit)
        return {
            "limit": limit,
            "splits": count,
            "share_of_all_splits": count / n if n else None,
            "share_of_splits_that_offer": count / len(offering) if offering else None,
        }

    out = {
        "splits": n,
        "mean_offer_kwh_per_test_day": _mean(offers),
        "offer_p10": p10,
        "offer_p50": p50,
        "offer_p90": p90,
        "splits_that_offer": len(offering),
        "share_of_splits_that_offer": len(offering) / n if n else None,
        "mean_scale": _mean([s.scale for s in scores]),
        "splits_at_scale_zero_by_reason": dict(sorted(reasons.items())),
        "splits_with_test_losses": len(slot),
        "mean_test_slot_loss": _mean(slot),
        "mean_test_driver_loss": _mean(driver),
        "slot_loss_over_limit": over(slot, alpha_slot),
        "splits_with_ceiling_share": len(shares),
        "mean_share_of_best_causal_ceiling": _mean(shares),
    }
    if tier == "A":
        out["driver_loss_over_limit"] = over(driver, alpha_driver)
    return out


def pick_dial(mean_offers: dict[float, float | None]) -> float | None:
    """The dial with the largest mean offer; ties (within `TIE`) go to the smaller dial; None when
    there is no dial with a mean offer."""
    best, best_offer = None, None
    for dial in sorted(mean_offers):
        offer = mean_offers[dial]
        if offer is None:
            continue
        if best is None or offer > best_offer + TIE:
            best, best_offer = dial, offer
    return best
