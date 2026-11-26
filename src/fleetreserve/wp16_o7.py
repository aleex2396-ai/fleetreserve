"""Outcome O7: the corner search of each method's calibrated box (plan section 11, O7).

"O7: corner search of each method's calibrated box, 50 fixed random days per unit, with and without
oracle departures: Y% of days break ASA, LLF or maximum-entropy feedback, Y0% the fixed plan."

The step `corner` (`fleetreserve.corner_run`) writes `corner.json`: per unit and variant
(`forecast`; `oracle-departure`, "with oracle departures") and method,
`units.<unit>.variants.<variant>.methods.<method>`: the scale, and for a method that offers, the
days with a result, the days breaking and their share.

Readings (alternatives in brackets):
  X1. Y is the share of searched days with a result on which a corner breaks the box (a slot or a
      driver failure; `days_breaking` over `days_with_result`), per method, variant and unit. No
      figure is pooled over units (a reading recorded on 23 November 2026, fixed before any
      result was read). [Alternative: a share of days over the units, not built.]
  X2. A method whose calibrated scale is 0 offers nothing and has no day to break: it is named
      "no offer" in its unit's row. A unit with a failed
      variant is named with the run's error. The corners give the worst case for the fixed plan
      only; for every other method the share is a lower bound (`corner_run`).
"""

from fleetreserve.wp16_cells import Missing, Table
from fleetreserve.wp16_drive import Missed, Source

LABEL = "corner"
RESULTS = "corner.json"
VARIANTS = (
    ("forecast", "without oracle departures"),
    ("oracle-departure", "with oracle departures"),
)
COLUMNS = (
    "variant",
    "method",
    "unit",
    "scale",
    "searched days",
    "days with a result",
    "days breaking",
    "share of days breaking (%)",
    "status",
)


def build(source: Source, units: list) -> list[Table]:
    table = Table("o7-corner-units", "O7: the corner search, per unit", COLUMNS)
    try:
        document = source.json(LABEL, RESULTS)
    except Missed as problem:
        table.add("all", "all", "all units", Missing(problem.text), "", "", "", "", "")
        return [table]
    methods = document.get("methods") if isinstance(document.get("methods"), list) else []
    found = document.get("units") if isinstance(document.get("units"), dict) else {}
    for variant, words in VARIANTS:
        for method in methods:
            for unit in units:
                entry = found.get(unit)
                one = entry.get("variants", {}).get(variant) if isinstance(entry, dict) else None
                result = one.get("methods", {}).get(method) if isinstance(one, dict) else None
                if not isinstance(result, dict):
                    why = (
                        one.get("failed")
                        if isinstance(one, dict) and one.get("failed")
                        else entry.get("status", "not in the run")
                        if isinstance(entry, dict)
                        else "not in the run"
                    )
                    table.add(words, method, unit, "", "", "", "", Missing(f"no figure: {why}"), "")
                    continue
                share = result.get("share_of_days_breaking")
                table.add(
                    words,
                    method,
                    unit,
                    result.get("scale"),
                    result.get("days_searched"),
                    result.get("days_with_result"),
                    result.get("days_breaking"),
                    100.0 * share if isinstance(share, int | float) else None,
                    result.get("status"),
                )
    table.notes.append(
        "Per unit only: no figure is pooled over units (reading X1). For every method but the "
        "fixed plan the share of days breaking is a lower bound."
    )
    return [table]
