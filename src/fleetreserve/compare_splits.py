"""The per-split rows of a comparison run (`compare_run`) and the file that holds them.

One row per unit, variant, tier, method and split, from the same calibration and scoring as the
summaries of `compare.json` (`dial_splits.score_split`, which `dial_choice.score_splits` calls for
`dial_splits.summarise`). Columns (`columns`), with an empty cell where a value does not exist:

- `unit`, `variant`, `tier`, `method`, `seed` (the split's seed, `dial_choice.split_indices`);
- `calibration_days`, `test_days`: the numbers of the split's calibration and test days (all
  counting days);
- `offering_calibration_days`: the number of its calibration days that offer (the `n` of the
  calibration);
- `scale`: the calibrated scale (0: abstain); `scale_zero_reason`: why it is 0, else empty;
- `offer_kwh_per_counting_test_day`: the scale times the mean certified value over the test
  days (0 at scale 0);
- `offering_test_days`: the number of test days that offer;
- `failing_test_days`: the number of offering test days with a slot loss above 0 at the scale
  (the day-level loss, the largest over the audited paths); empty at scale 0, where nothing is
  offered;
- `mean_test_slot_loss`, `mean_test_driver_loss`: the means of the day-level losses at the scale
  (not their running maximum) over the offering test days; empty at scale 0 or with no offering
  test day;
- `share_of_best_causal_ceiling`: the sum of the test days' offers over the sum of their
  best-causal ceilings; empty at scale 0, when a test day's ceiling is missing or when the sum is
  not positive;
- `mean_<ceiling>_ceiling_kwh_per_test_day`, one per ceiling name given (`compare_run` gives the
  ceilings of the run's matrix and, under its no-pause sensitivity, those ceilings of the fleet
  without the drawn cars, named with the suffix `-no-pause-cars-removed`): the mean of that
  ceiling over the split's test days, as `dial_splits.gate_split` takes it; empty when a test
  day's ceiling is missing. It is the same on every row of a unit and split.

From these rows every summary field and gate-reading figure of `compare.json` can be recomputed:
the summary of a unit, variant, method and tier from its rows (`dial_splits.summarise`: means and
points of the offers, means of the losses over the splits that offer with a defined loss, counts
over the limits, the scale-zero reasons), and the gate reading of a method from its tier A rows of
both variants and the ceiling columns (`dial_splits.summarise_gate`).

`write(rows, path, ceiling_names)` writes the rows as comma-separated text with one header line,
in the order the rows are given (`compare_run` gives them in the order unit, variant, tier, method,
seed), each float as Python's `repr` (it reads back to the same number), and returns the number of
rows.
"""

import csv
from pathlib import Path

import numpy as np

from fleetreserve import dial_splits

BASE_COLUMNS = (
    "unit",
    "variant",
    "tier",
    "method",
    "seed",
    "calibration_days",
    "test_days",
    "offering_calibration_days",
    "scale",
    "scale_zero_reason",
    "offer_kwh_per_counting_test_day",
    "offering_test_days",
    "failing_test_days",
    "mean_test_slot_loss",
    "mean_test_driver_loss",
    "share_of_best_causal_ceiling",
)


def ceiling_column(name: str) -> str:
    return f"mean_{name}_ceiling_kwh_per_test_day"


def columns(ceiling_names) -> tuple[str, ...]:
    return (*BASE_COLUMNS, *(ceiling_column(c) for c in ceiling_names))


def _mean_ceiling(per_day, test) -> float | None:
    values = np.asarray(per_day, dtype=float)[test]
    return None if not len(values) or np.isnan(values).any() else float(values.mean())


def split_rows(
    unit: str, variant: str, losses: dict, splits, tiers, ceilings: dict, grid=dial_splits.GRID
) -> list[dict]:
    """The rows of one unit and variant (module text), in the order tier, method, seed.
    `losses`: method -> `dial_splits.UnitLosses`; `splits`: (seed, calibration, test) day indices;
    `ceilings`: ceiling name -> one entry per counting day, NaN where missing."""
    rows = []
    for tier in tiers:
        for method, found in losses.items():
            for seed, calibration, test in splits:
                test = np.asarray(test, dtype=int)
                score = dial_splits.score_split(found, seed, calibration, test, tier, grid)
                failing = None
                if score.scale > 0:
                    k = list(grid).index(score.scale)
                    offering = test[found.offering[test]]
                    failing = int((found.slot[offering, k] > 0).sum())
                row = {
                    "unit": unit,
                    "variant": variant,
                    "tier": tier,
                    "method": method,
                    "seed": int(seed),
                    "calibration_days": len(calibration),
                    "test_days": len(test),
                    "offering_calibration_days": score.offering_calibration_days,
                    "scale": float(score.scale),
                    "scale_zero_reason": score.reason,
                    "offer_kwh_per_counting_test_day": score.offer_kwh,
                    "offering_test_days": score.offering_test_days,
                    "failing_test_days": failing,
                    "mean_test_slot_loss": score.slot_loss,
                    "mean_test_driver_loss": score.driver_loss,
                    "share_of_best_causal_ceiling": score.ceiling_share,
                }
                for name, per_day in ceilings.items():
                    row[ceiling_column(name)] = _mean_ceiling(per_day, test)
                rows.append(row)
    return rows


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write(rows: list[dict], path: Path, ceiling_names) -> int:
    """Writes the rows (module text); returns their number."""
    names = columns(ceiling_names)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(names)
        for row in rows:
            writer.writerow([_cell(row[c]) for c in names])
    return len(rows)
