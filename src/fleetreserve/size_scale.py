"""Per-day sizes and the calibrated fixed-plan offer they give (plan sections 6 (T6), 7 (T7)).

A sizing gives each counting day d of a unit a size `q_d`; the day's scale is `gamma * q_d` with one
scalar `gamma` calibrated over the unit's calibration days. Today's rule is `q_d = 1` (`constant`).

Losses: the fixed plan's every-request losses of `gate_calibrate` (closed-form slot loss from the
hour thresholds, `gate_calibrate.slot_steps`; driver-loss bound, the every-scale bound
`gate_calibrate.every_scale_loss`, the driver bound wherever one is used by the owner's decision of
16 November 2026: `gate_calibrate`, module text). The bound reads the scale only through `beta`
times the gains, so at scale `gamma * q_d` the slot loss of a day is its slot loss at `gamma` with
every hour threshold divided by `q_d`, and its driver-loss bound is the bound at `gamma` with every
gain multiplied by `q_d` (`_sized`: each ramp's slope, the second array of
`gate_calibrate.driver_terms`, and the gains and their sums `|G_t|_1` of
`fixed_plan_loss.ServiceHours`, each multiplied by `q_d`, and its power rows' thresholds divided by
`q_d`, as the slot loss's are). At `q_d = 1` they are the day's own terms, bit for bit. So a sizing
changes only the inputs of the calibration; the calibration is that of
`gate_calibrate.every_request`, unchanged: `crc.calibrate_exact` for the slot loss and
`gate_calibrate.beta_hat_every_scale` for the driver loss. Tier A takes the smaller of the slot
and driver scalars, tier B the slot scalar alone (plan section 6, Tiers). What the bound bounds,
and with what error: `fixed_plan_loss`, module text, reading R3, V3 and V7. The earlier bound
(reading R2) is not used here.

Cap. The stay-to-bound property is claimed up to scale 1 only (plan section 6), so no day's scale
may exceed 1: the scalar used is the calibrated one capped at `1 / max_d q_d`, the maximum over
the unit's days that offer under the sizing, test days included (the cap is 1 when no day offers).
A calibrated scalar is infinite when its bound holds at every scalar (tier A: both bounds), which
calibration days that lose can still allow: of 20 calibration days of size 1 with no driver loss,
one day whose slot loss is 1 above scale 0.3 gives infinity in both tiers, two such days give 0.3.
An infinite scalar, like any scalar above the cap, is replaced by the cap. Every loss is
non-decreasing in the scale, so a smaller scalar keeps the calibrated bound. The cap is computed
from the sizes alone and reads no loss itself; an outcome reaches it only through sizes computed
from one: the oracle sizes of `size_whatif` come from the days' own labels, and its gbm sizes can
depend on the labelled unit's outcomes through the other units' rows (`size_whatif`, step 3). With
`q_d = 1` the calibrated scalars are those of
`gate_calibrate.every_request` as calibrated (its written `beta` fields, or, above
`policy.VALIDITY_LIMIT`, the values under its mark `gate_calibrate.VALIDITY`), and the scalar used
is its scale capped at 1 (`every_request` caps an infinite scale at `crc.GRID[-1]`, 2, and limits a
finite one to `policy.VALIDITY_LIMIT`, 1e5, so either cap at 1 gives the same scale).

Arms. `sizes(raw, offering, arm)` turns a raw size per day (NaN where the day does not offer) into
`q`: in the arm "scale", `q_d` is the raw size clipped to [`FLOOR`, 1]; in the arm "abstain", a day
whose raw size is below `FLOOR` does not offer (its `q_d` is NaN) and the others get the raw size
capped at 1. A day with `q_d` NaN offers 0, is not a calibration day of the scalar and is in
neither test-loss mean; it is still a counting test day of the offer per counting test day.

Test scoring (`score`): the offer per counting test day is the mean over the test days of
`gamma * q_d * value_d` (0 on a day that does not offer); the losses are the slot loss and the
driver-loss bound at `gamma` with the day's inputs scaled as above, averaged over the test days
that offer, and None at `gamma` 0 or with no such day.
"""

import math
from dataclasses import dataclass

import numpy as np

from fleetreserve import crc
from fleetreserve.carsets import true_rows
from fleetreserve.fixed_plan_loss import ServiceHours, closed_form_slot_loss, hour_thresholds
from fleetreserve.gate_calibrate import (
    beta_hat_every_scale,
    every_scale_loss,
    every_scale_terms,
    slot_steps,
)
from fleetreserve.gate_day import PreparedDay

FLOOR = crc.GRID[0]  # 0.05, the smallest scale of the plan's grid
ARMS = ("scale", "abstain")
TIERS = ("A", "B")
TOO_FEW = "too few calibration days offer"
NO_SAFE_SCALE = "the bound fails at every scalar above 0"


@dataclass(frozen=True)
class DayTerms:
    """What the losses of one offering day need: its certified value at scale 1, its hour
    thresholds, and the terms of its every-scale driver-loss bound
    (`gate_calibrate.every_scale_terms`): its need rows' ramps (`driver`,
    `gate_calibrate.driver_terms`) and what the bound reads of each car's service hours (`hours`,
    `fixed_plan_loss.ServiceHours`)."""

    value: float
    thresholds: np.ndarray
    driver: tuple
    hours: ServiceHours


def day_terms(prepared: PreparedDay) -> DayTerms | None:
    """The terms of a prepared day, or None when it does not offer."""
    if prepared.abstains:
        return None
    thresholds = hour_thresholds(prepared.certificate.policy, true_rows(prepared.day))
    driver, hours = every_scale_terms(prepared)
    return DayTerms(float(prepared.value), thresholds, driver, hours)


def sizes(raw, offering, arm: str) -> np.ndarray:
    """`q` from the raw sizes `raw` on the days where `offering` is true (module text)."""
    raw = np.asarray(raw, dtype=float)
    offering = np.asarray(offering, dtype=bool)
    if np.isnan(raw[offering]).any():
        raise ValueError("a day that offers has no raw size")
    q = np.full(raw.shape, math.nan)
    if arm == "scale":
        q[offering] = np.clip(raw[offering], FLOOR, 1.0)
    elif arm == "abstain":
        kept = offering & (raw >= FLOOR)
        q[kept] = np.minimum(raw[kept], 1.0)
    else:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    return q


def cap(q) -> float:
    """`1 / max q` over the days that offer under the sizing; 1 when none does."""
    q = np.asarray(q, dtype=float)
    kept = q[~np.isnan(q)]
    return 1.0 if kept.size == 0 else float(1.0 / kept.max())


def _slot(terms: DayTerms, q: float) -> np.ndarray:
    return terms.thresholds / q


def _sized(terms: DayTerms, q: float) -> tuple:
    """The every-scale terms of the day with every gain multiplied by `q` (module text)."""
    c, s, total = terms.driver
    h = terms.hours
    hours = ServiceHours(
        h.centre, h.gains * q, h.spread * q, h.cap, h.thresholds / q, h.first, h.short
    )
    return (c, s * q, total), hours


def driver_loss(terms: DayTerms, q: float, gamma: float) -> float:
    """The day's driver-loss bound at scalar `gamma > 0` and size `q` (module text)."""
    return every_scale_loss(_sized(terms, q), gamma)


@dataclass(frozen=True)
class Scalar:
    """The scalar of one tier on one set of calibration days."""

    calibrated: float  # as calibrated; infinity when its bound holds at every scalar
    used: float  # capped (module text)
    reason: str  # why `used` is 0, empty otherwise
    days: int  # calibration days that offer under the sizing


def calibrate(terms, q, calibration, tiers=TIERS) -> dict[str, Scalar]:
    """Per tier the scalar on the calibration days `calibration` (indices) that offer under `q`."""
    q = np.asarray(q, dtype=float)
    days = [d for d in calibration if not math.isnan(q[d])]
    slot = [slot_steps(_slot(terms[d], q[d])) for d in days]
    exact = crc.calibrate_exact(slot, [((), ())] * len(days))
    top = cap(q)
    out = {}
    for tier in tiers:
        if tier == "A":
            driver = beta_hat_every_scale([_sized(terms[d], q[d]) for d in days], crc.ALPHA_DRIVER)
            calibrated = min(exact.slot_beta, driver)
            enough = crc.enough_days(len(days), crc.ALPHA_SLOT) and crc.enough_days(
                len(days), crc.ALPHA_DRIVER
            )
        elif tier == "B":
            calibrated = exact.slot_beta
            enough = crc.enough_days(len(days), crc.ALPHA_SLOT)
        else:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
        used = min(calibrated, top)
        reason = "" if used > 0 else (TOO_FEW if not enough else NO_SAFE_SCALE)
        out[tier] = Scalar(float(calibrated), float(used), reason, len(days))
    return out


@dataclass(frozen=True)
class TestScore:
    """The test days of one split at one scalar (module text)."""

    offers_kwh: tuple[float, ...]  # per counting test day
    offering_days: int  # test days that offer under the sizing
    slot_losses: tuple[float, ...]  # per offering test day; empty at scalar 0
    driver_losses: tuple[float, ...]

    @property
    def mean_offer_kwh(self) -> float:
        return float(np.mean(self.offers_kwh)) if self.offers_kwh else 0.0


def score(terms, q, gamma: float, test) -> TestScore:
    """The test days `test` (indices) at scalar `gamma` (module text)."""
    q = np.asarray(q, dtype=float)
    offers = [0.0 if math.isnan(q[d]) else gamma * q[d] * terms[d].value for d in test]
    offering = [d for d in test if not math.isnan(q[d])]
    slot, driver = [], []
    for d in offering if gamma > 0 else []:
        slot.append(closed_form_slot_loss(_slot(terms[d], q[d]), gamma))
        driver.append(driver_loss(terms[d], q[d], gamma))
    return TestScore(tuple(float(o) for o in offers), len(offering), tuple(slot), tuple(driver))


def mean_or_none(values) -> float | None:
    return float(np.mean(values)) if len(values) else None
