"""The arithmetic of the WP6 gate (plan section 13, WP6), as pure functions.

The plan: "go if the calibrated re-solve offer is >= 10% of the ceiling on 2 of 3 and the
oracle-departure offer >= 1.5 times it." A gate script feeds these functions numbers and reads the
decision. Nothing here reads a file or draws a random number.

Per unit, on its TEST days only: `values_at_scale_1` is the certified offer value per day at scale
1 in kWh (0 for a day without a certificate), `ceilings` the hindsight ceiling per day in kWh, and
beta the calibrated scale. The calibrated offer of a day is beta times its value (the offer is
scored only at the chosen beta).

Readings of the plan's wording (each is the closest reading; the alternative is in brackets):

  G1. "Z% of the ceiling" is the ratio of sums, sum of offers over sum of ceilings, the main
      definition. The mean of per-day ratios over days with a positive ceiling is returned beside
      it. [Alternative: mean of per-day ratios as the main definition; `share_basis` selects it.]
  G2. The ratio "oracle-departure offer >= 1.5 times it" is formed per unit from the two mean
      offers in kWh per day: oracle mean over calibrated mean. [Alternative: pooled over units.]
  G3. The plan does not say on which units the 1.5 rule must hold. Three readings, chosen by the
      caller with `ratio_rule_on`:
        "all_units": every gate unit passes the 1.5 rule.
        "all_share_passing_units": every unit that passes the 10% rule also passes the 1.5 rule.
        "required_units_both": at least `required_units` units pass both rules.
      In every reading the 10% rule must hold on at least `required_units` units.
  G4. A calibrated mean offer of zero leaves the ratio undefined (None, never infinity). Such a
      unit fails the 1.5 rule (the rule cannot be shown to hold) and fails the 10% rule anyway.
  G5. Comparisons "share >= 0.10" and "ratio >= 1.5" allow a float tolerance of 1e-12, so a share
      of exactly 0.10 written as 0.3 / 3 passes.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SHARE_THRESHOLD = 0.10
RATIO_THRESHOLD = 1.5
_EPS = 1e-12
RATIO_RULES = ("all_units", "all_share_passing_units", "required_units_both")
SHARE_BASES = ("ratio_of_sums", "mean_of_ratios")


@dataclass(frozen=True, eq=False)
class UnitOffer:
    """The offer of one policy on one unit's test days at one scale."""

    beta: float
    n_days: int
    n_days_positive_ceiling: int
    offers_kwh: np.ndarray  # per day: beta * value at scale 1
    mean_offer_kwh: float  # mean of offers_kwh, kWh per day
    share_of_sums: float  # sum of offers / sum of ceilings (G1, main)
    mean_of_daily_ratios: float  # mean of offer / ceiling over days with ceiling > 0 (G1)


def unit_offer(
    values_at_scale_1: Sequence[float], ceilings: Sequence[float], beta: float
) -> UnitOffer:
    """The calibrated offer of one unit: per-day offers, their mean and the share of the ceiling.

    Raises ValueError for unequal lengths, no days, a negative or non-finite value, a negative or
    non-finite beta (a beta of infinity must be capped by the caller), or a ceiling sum of 0 (the
    share is then undefined).
    """
    values = np.asarray(values_at_scale_1, dtype=float)
    ceil = np.asarray(ceilings, dtype=float)
    if values.ndim != 1 or values.shape != ceil.shape or values.size == 0:
        raise ValueError("values and ceilings must be non-empty one-dimensional and equally long")
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(ceil))):
        raise ValueError("values and ceilings must be finite")
    if values.min() < 0.0 or ceil.min() < 0.0:
        raise ValueError("values and ceilings must not be negative")
    if not (math.isfinite(beta) and beta >= 0.0):
        raise ValueError(f"beta must be finite and >= 0, got {beta}")
    total_ceiling = float(ceil.sum())
    if not total_ceiling > 0.0:
        raise ValueError("the ceilings sum to 0: the share of the ceiling is undefined")
    offers = beta * values
    positive = ceil > 0.0
    return UnitOffer(
        beta=float(beta),
        n_days=int(values.size),
        n_days_positive_ceiling=int(positive.sum()),
        offers_kwh=offers,
        mean_offer_kwh=float(offers.mean()),
        share_of_sums=float(offers.sum()) / total_ceiling,
        mean_of_daily_ratios=float((offers[positive] / ceil[positive]).mean()),
    )


@dataclass(frozen=True)
class GateUnit:
    """A gate unit's two offers on the same test days: calibrated re-solve and oracle departure."""

    name: str
    calibrated: UnitOffer
    oracle: UnitOffer


@dataclass(frozen=True)
class UnitGate:
    name: str
    calibrated_share: float
    oracle_share: float
    calibrated_mean_kwh: float
    oracle_mean_kwh: float
    ratio: float | None  # oracle mean / calibrated mean; None when the calibrated mean is 0 (G4)
    passes_share: bool
    passes_ratio: bool
    passes_both: bool


@dataclass(frozen=True)
class GateDecision:
    go: bool
    required_units: int
    ratio_rule_on: str
    n_pass_share: int
    n_pass_both: int
    units: tuple[UnitGate, ...]


def gate_decision(
    units: Sequence[GateUnit],
    *,
    required_units: int,
    ratio_rule_on: str,
    share_threshold: float = SHARE_THRESHOLD,
    ratio_threshold: float = RATIO_THRESHOLD,
    share_basis: str = "ratio_of_sums",
) -> GateDecision:
    """Apply the gate to the listed gate units.

    `required_units`: how many units must pass the 10% rule (the plan's 2 of 3).
    `ratio_rule_on`: one of RATIO_RULES (G3). `share_basis`: "ratio_of_sums" (G1, main) or
    "mean_of_ratios". Every unit's numbers are returned, whatever the decision.
    """
    if ratio_rule_on not in RATIO_RULES:
        raise ValueError(f"ratio_rule_on must be one of {RATIO_RULES}, got {ratio_rule_on!r}")
    if share_basis not in SHARE_BASES:
        raise ValueError(f"share_basis must be one of {SHARE_BASES}, got {share_basis!r}")
    if not units:
        raise ValueError("no gate units")
    if len({u.name for u in units}) != len(units):
        raise ValueError("a gate unit name appears twice")
    if not (1 <= required_units <= len(units)):
        raise ValueError(f"required_units must be between 1 and {len(units)}")

    def share(offer: UnitOffer) -> float:
        return offer.share_of_sums if share_basis == "ratio_of_sums" else offer.mean_of_daily_ratios

    rows = []
    for u in units:
        if u.calibrated.n_days != u.oracle.n_days:
            raise ValueError(f"unit {u.name!r}: the two offers are on different numbers of days")
        cal_mean, ora_mean = u.calibrated.mean_offer_kwh, u.oracle.mean_offer_kwh
        ratio = None if cal_mean == 0.0 else ora_mean / cal_mean
        passes_share = share(u.calibrated) >= share_threshold - _EPS
        passes_ratio = ratio is not None and ratio >= ratio_threshold - _EPS
        rows.append(
            UnitGate(
                u.name,
                share(u.calibrated),
                share(u.oracle),
                cal_mean,
                ora_mean,
                ratio,
                passes_share,
                passes_ratio,
                passes_share and passes_ratio,
            )
        )
    n_share = sum(r.passes_share for r in rows)
    n_both = sum(r.passes_both for r in rows)
    if ratio_rule_on == "all_units":
        ratio_ok = all(r.passes_ratio for r in rows)
    elif ratio_rule_on == "all_share_passing_units":
        ratio_ok = all(r.passes_ratio for r in rows if r.passes_share)
    else:
        ratio_ok = n_both >= required_units
    go = n_share >= required_units and ratio_ok
    return GateDecision(go, required_units, ratio_rule_on, n_share, n_both, tuple(rows))
