"""Outcome O10: the two models against the earlier forecasts and the baselines (plan section 11).

"O10: at equal guarantee, the main model `lgbm_tail` and the neural model `hazard_tuned_inputs`
(Section 7), each against earlier forecasts and Kaplan-Meier (X%), LightGBM and Cox (Y%), X and Y
given for each of the two."

Readings (alternatives in brackets):
  W1. "At equal guarantee": every run is calibrated to the same levels (plan section 6), so the
      comparison is of their offers: the sum over the nine units of each unit's offer per counting
      test day (O1's X, `wp16_o1`), a unit with no figure adding 0 on both sides.
  W2. The change is 100 times (model minus baseline) over baseline, in the same rung: the full
      method (one-way re-solve) against `earlier_gbm` and `earlier_population`, which ran through
      the full method; the fixed plan against Kaplan-Meier, pooled LightGBM and Cox, which ran on
      the fixed plan (plan section 13, WP14). [Alternative: the model's one-way offer against the
      baselines' fixed-plan offers.]
  W3. Each baseline's change is its own column; X% is read as the columns of `earlier_gbm`,
      `earlier_population` and Kaplan-Meier and Y% as those of pooled LightGBM and Cox, as the
      sentence lists them; the plan does not say whether X and Y are each one figure. No interval
      is written for a change: day_intervals gives intervals for the two models' own runs only
      (plan section 11).
"""

from fleetreserve.wp16_cells import Missing, Table
from fleetreserve.wp16_drive import Source
from fleetreserve.wp16_learn import Runs
from fleetreserve.wp16_o1 import Spec, summary_row

MODELS = (
    ("lgbm_tail", "learn-lgbm_tail", "intervals-lgbm_tail"),
    ("hazard_tuned_inputs", "learn-hazard_tuned_inputs", "intervals-hazard_tuned_inputs"),
)
BASES = (
    ("earlier_gbm", "learn-earlier_gbm", "one-way"),
    ("earlier_population", "learn-earlier_population", "one-way"),
    ("Kaplan-Meier", "learn-fixed-plan-km", "fixed"),
    ("pooled LightGBM", "learn-fixed-plan-lgbm", "fixed"),
    ("Cox", "learn-fixed-plan-cox", "fixed"),
)
COLUMNS = (
    "model",
    "against",
    "rung",
    "model's X (kWh per counting test day)",
    "baseline's X (kWh per counting test day)",
    "change of the offer (%)",
)


def _offer(source: Source, runs: Runs, spec: Spec, units: list, absent: dict):
    cell = summary_row(source, runs, spec, units, absent)[0][4]
    return cell if isinstance(cell, Missing) else cell.point


def change(model, base):
    """100 (model - base) / base, or the `Missing` that explains why there is none."""
    for found in (model, base):
        if isinstance(found, Missing):
            return found
        if found is None:
            return Missing("no figure: no offer")
    if base <= 0:
        return Missing("no figure: the baseline's offer is 0")
    return 100.0 * (model - base) / base


def build(source: Source, runs: Runs, units: list, absent: dict) -> tuple[list[Table], list[dict]]:
    table = Table("o10-model-comparison", "O10: each model against the baselines", COLUMNS)
    rows = []
    for title, learn, intervals in MODELS:
        for base_title, base_learn, rung in BASES:
            mine = _offer(source, runs, Spec(title, learn, intervals, rung, "main"), units, absent)
            theirs = _offer(
                source, runs, Spec(base_title, base_learn, None, rung, "main"), units, absent
            )
            found = change(mine, theirs)
            table.add(title, base_title, rung, mine, theirs, found)
            rows.append((title, base_title, found if isinstance(found, float) else None))
    table.notes.append(
        "The change is read as reading W2 says: same rung on both sides, a unit with no figure "
        "adding 0, no interval (reading W3)."
    )
    return [table], [{"kind": "o10", "rows": rows}]
