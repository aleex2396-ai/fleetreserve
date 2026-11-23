"""The figures of one unit, model, k and tier of the cold start (plan section 11, O9: "availability
A_k, offer share Y_k"). Pure arithmetic: numpy only, no files, no solver.

Input: `dial_splits.UnitLosses` over the days the run reads at k (the k local days, then the
scored days), with the one-way rung's day-level losses filled at every grid scale on the local
days that offer and at the chosen scales on the scored days that offer (`cold_start` fills
them); `ceilings`: ceiling name -> one value per day, NaN where a ceiling is missing.

Calibration and scoring are `dial_splits.score_split` with the local days as its calibration days
and the scored days as its test days, unchanged: the on-grid scale of the tier from the local days
that offer (`dial_splits.on_grid_scale`: 0 when they are fewer than `crc.enough_days` asks for,
1/alpha - 1, or when the bound fails at the smallest grid scale), the offer per counting scored
day (the scale times the mean certified value, 0 on days that abstain and at scale 0), the mean
slot and driver losses over the scored days that offer at that scale (None at scale 0 or when no
scored day offers).

The two figures of O9 (readings C8 and C9 of `cold_start`):

- availability A_k: 1 minus that mean slot loss, so the share of service hours delivered from the
  first failure on, averaged over the offering scored days (plan section 0: "mean slot
  availability >= 90%", the complement of the slot loss at alpha 0.10 of T6). None where the mean
  slot loss is None: nothing was offered, so there is no availability to measure;
- offer share Y_k, per ceiling: the offer per counting scored day over the mean ceiling per
  counting scored day, as `dial_splits.summarise_gate` gives the share of one split (0 when the
  offer is 0; None when a scored day's ceiling is missing or the mean ceiling is not positive).
  The plan's "the ceiling" is the hindsight ceiling (section 2: "V_hind (the ceiling)"; WP6: "the
  mean hindsight ceiling per counting test day"), the first of `gate_config.CEILINGS`; the
  best-causal one is given beside it.

Also: the failing scored days (offering scored days with a slot loss above 0 at the scale), and
the numbers of local and scored days that offer.
"""

import numpy as np

from fleetreserve import dial_splits


def availability(slot_loss: float | None) -> float | None:
    """1 minus the mean slot loss over the offering scored days, None when there is none."""
    return None if slot_loss is None else 1.0 - float(slot_loss)


def share(offer_kwh: float, ceilings) -> dict:
    """The offer per counting scored day over the mean ceiling per counting scored day (module
    text), with the mean ceiling and the number of scored days whose ceiling is missing."""
    values = np.asarray(ceilings, dtype=float)
    missing = int(np.isnan(values).sum())
    mean = None if missing or not len(values) else float(values.mean())
    usable = mean is not None and mean > 0
    return {
        "mean_ceiling_kwh_per_scored_day": mean,
        "scored_days_ceiling_missing": missing,
        "share_of_ceiling": float(offer_kwh) / mean if usable else None,
    }


def score_k(losses: dial_splits.UnitLosses, k: int, local, scored, tier: str, ceilings) -> dict:
    """The figures of one tier at k (module text). `local`, `scored`: day indices into `losses`;
    `ceilings`: ceiling name -> one value per day of `losses`."""
    local = np.asarray(local, dtype=int)
    scored = np.asarray(scored, dtype=int)
    found = dial_splits.score_split(losses, k, local, scored, tier)
    failing = None
    if found.scale > 0:
        column = list(dial_splits.GRID).index(found.scale)
        offering = scored[losses.offering[scored]]
        failing = int((losses.slot[offering, column] > 0).sum())
    return {
        "tier": tier,
        "scale": found.scale,
        "scale_zero_reason": found.reason,
        "local_days": int(len(local)),
        "offering_local_days": found.offering_calibration_days,
        "scored_days": int(len(scored)),
        "offering_scored_days": found.offering_test_days,
        "failing_scored_days": failing,
        "offer_kwh_per_counting_scored_day": found.offer_kwh,
        "mean_slot_loss_offering_scored_days": found.slot_loss,
        "availability": availability(found.slot_loss),
        "mean_driver_loss_offering_scored_days": found.driver_loss,
        "offer_share": {
            name: share(found.offer_kwh, np.asarray(values, dtype=float)[scored])
            for name, values in ceilings.items()
        },
    }
