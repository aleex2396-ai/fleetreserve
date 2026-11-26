"""O1's slots A and B: the comparison run's offers by method (plan sections 10, 11).

"O1 ... against ASA (A), LLF (B), earlier forecasts (C) ... A and B are compared with the
comparison run's own re-solve offer (Section 10), and the results say both."

The step `compare` (`fleetreserve.compare_run`) writes `compare.json`: per unit that could be split,
`units.<unit>.gate_reading.<method>`, the tier A reading over the run's one split
(`dial_splits.summarise_gate`): `mean_offer_kwh_per_test_day` and, per ceiling,
`ceilings.<name>.share_of_ceiling`. The methods are those of `compare_methods.METHODS`.

Readings (alternatives in brackets):
  K1. A is the `asa` row and B the `llf` row, the other methods tabled with them; each is the
      sum over the units of that unit's `mean_offer_kwh_per_test_day`, as O1's X is (`wp16_o1`,
      reading Q2), beside the comparison run's own `re-solve` row, which is not the figure of
      the scoring run (`compare_run` reading H5: it calibrates on a grid of 10 scales, the
      scoring run bisects). [Alternative: one scoring-run figure per method.]
  K2. A unit with no figure for a method (it could not be split, or a variant failed) adds 0 and
      is named in the row's last column.
  K3. The comparison runs at one dial for every unit (`dial-workplaces`, the run list's fixed
      names): the dial is printed with the table. No interval is written for a comparison method
      (plan section 11: intervals for the two models only).
"""

import math

from fleetreserve.wp16_cells import Interval, Missing, Table
from fleetreserve.wp16_drive import Missed, Source

LABEL = "compare"
RESULTS = "compare.json"
METHOD_TITLES = {
    "re-solve": "re-solve (the comparison run's own)",
    "asa": "ASA (A)",
    "llf": "LLF (B)",
}
COLUMNS = (
    "method",
    "sum of the units' offers (kWh per counting test day)",
    "units with a figure",
    "units left out",
)


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def build(source: Source, units: list) -> list[Table]:
    table = Table("o1-comparison", "O1: A, B and the other comparison methods", COLUMNS)
    try:
        document = source.json(LABEL, RESULTS)
    except Missed as problem:
        table.add("all methods", Missing(problem.text), "", "")
        return [table]
    methods = document.get("methods")
    methods = methods if isinstance(methods, list) else []
    entries = document.get("units") if isinstance(document.get("units"), dict) else {}
    for method in methods:
        total, used, left = 0.0, 0, []
        for unit in units:
            entry = entries.get(unit)
            found = None
            if isinstance(entry, dict) and isinstance(entry.get("gate_reading"), dict):
                found = entry["gate_reading"].get(method)
            value = found.get("mean_offer_kwh_per_test_day") if isinstance(found, dict) else None
            if _number(value):
                total, used = total + value, used + 1
            else:
                status = entry.get("status") if isinstance(entry, dict) else "not in the run"
                left.append(f"{unit}: no figure ({status})")
        table.add(
            METHOD_TITLES.get(method, method),
            Interval(total) if used else Missing("no figure: no unit has one"),
            f"{used} of {len(units)}",
            "; ".join(left),
        )
    failed = document.get("failed")
    table.notes.append(
        f"dial {document.get('dial')}, configurations {document.get('configurations')}, one "
        "split per unit, tier A (reading K3)."
    )
    if isinstance(failed, list) and failed:
        table.notes.append("failed in the run: " + "; ".join(str(x) for x in failed))
    return [table]
