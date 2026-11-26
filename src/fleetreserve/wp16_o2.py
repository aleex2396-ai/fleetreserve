"""Outcome O2: the block product's V_caus over V_hind on every day (plan section 11, O2).

"O2 Headline measurement, block product, M days with |S|<=8: V_caus/V_hind mean X%, median Q%; P%
on days with an in-window departure; V_aff=V_caus on K. Pass: V_caus=V_hind on every one-way day
meeting T4's hypotheses."

The steps `o2-rho0` (main setting) and `o2-held-rho0` (held setting) run
`fleetreserve.ceiling_report --every-day` at rho 0 over the nine units; their
`ceiling-report.json` holds `every_day.units.<unit>` (`ceiling_summary.o2_with_faults`) and
`every_day.pass_over_units`. The two settings are tabled apart and never pooled (plan section 3).

Readings (alternatives in brackets):
  U1. X and Q are the mean and median of the per-day ratio over the M days with a positive
      hindsight value (`ratio_caus_over_hind`), in per cent; P is the same ratio over the M days
      with an in-window departure, with their number and share of the M days beside it; K is the
      number of M days on which V_aff and V_caus do not differ, out of the M days where both are
      present (`ceiling_summary`, reading P). [Alternative: P as the share alone.]
  U2. The pass is the unit's `o2_pass` and the verdict over the units `pass_over_units`, as the
      file has them (`ceiling_summary` reading N): a unit with no day to check is named as such,
      never as passed or failed. A unit the truth build gave no window is "no day to check" with
      its reason.
  U3. The faults of a unit (orders broken, days not built) are printed beside, as `ceiling_report`
      counts them; no number here is recomputed.
  U4. `o2-day-counts` (`build_day_counts`) gives the day counts the text states, from the same
      `ceiling-report.json`: per unit, `counting_days`, `days_in_m`, and, from `t4_check`
      (`ceiling_summary.t4_check` of the unit's M days), the days that meet T4's hypotheses
      (`days_meeting_hypotheses`: "few days meet them: 87 of the 2,846 counting days"), those
      of them checked (`days_checked`), those not checkable for a missing value
      (`days_not_checked_missing_value`) and the days on which V_hind differs from V_caus, the
      number of `violations`. A last row per setting is the report's own pooled entry
      (`every_day.all_units`), not a sum made here. A unit the truth build gave no window has
      none of them: "no figure", with its reason. A count the file does not hold is "no figure:
      <why>", never 0. It is a separate table so that `o2-block-product` keeps its columns.
"""

from fleetreserve.wp16_cells import Missing, Table
from fleetreserve.wp16_drive import Missed, Source

RESULTS = "ceiling-report.json"
STEPS = (("main setting", "o2-rho0"), ("held setting", "o2-held-rho0"))
COLUMNS = (
    "setting",
    "unit",
    "M days",
    "X: mean of V_caus/V_hind (%)",
    "Q: median of V_caus/V_hind (%)",
    "P: V_caus/V_hind on days with an in-window departure (%)",
    "days with an in-window departure",
    "K: days with V_aff = V_caus",
    "K out of",
    "pass",
    "faults and reasons",
)


def _percent(value):
    return None if not isinstance(value, int | float) else 100.0 * value


def _faults(entry: dict) -> str:
    own = entry.get("faults") if isinstance(entry.get("faults"), dict) else {}
    parts = [f"{key.replace('_', ' ')}: {own[key]}" for key in sorted(own) if own[key]]
    if entry.get("days_in_m_not_built"):
        parts.append(f"M days not built: {entry['days_in_m_not_built']}")
    if isinstance(entry.get("reason"), str):
        parts.append(entry["reason"])
    return "; ".join(parts)


POOLED = "all units named (the report's own pooled entry)"
DAY_COLUMNS = (
    "setting",
    "unit",
    "counting days",
    "M days",
    "M days meeting T4's hypotheses",
    "of them checked (V_caus and V_hind both present)",
    "of them not checkable (a value missing)",
    "of them with V_hind different from V_caus",
)


def _count(found, key: str):
    """The file's own whole number under `key` of the dict `found`, else the cell that says so."""
    value = found.get(key) if isinstance(found, dict) else None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return Missing(f"no figure: the report holds no {key}")


def _day_counts(setting: str, unit: str, entry) -> tuple:
    """One row of `o2-day-counts` from a unit's entry (or the pooled entry) of the report."""
    if not isinstance(entry, dict):
        return (setting, unit, *([Missing("no figure: not in the report")] * 6))
    if isinstance(entry.get("reason"), str) and "t4_check" not in entry:  # no window: no days
        return (setting, unit, *([Missing(f"no figure: {entry['reason']}")] * 6))
    check = entry.get("t4_check")
    violations = check.get("violations") if isinstance(check, dict) else None
    broken = (
        len(violations)
        if isinstance(violations, list)
        else Missing("no figure: the report holds no violations list")
    )
    return (
        setting,
        unit,
        _count(entry, "counting_days"),
        _count(entry, "days_in_m"),
        _count(check, "days_meeting_hypotheses"),
        _count(check, "days_checked"),
        _count(check, "days_not_checked_missing_value"),
        broken,
    )


def build_day_counts(source: Source, units: list) -> list[Table]:
    """`o2-day-counts` (reading U4): the days where T4's property was checked or failed, by unit and
    setting, as `ceiling-report.json` holds them; nothing is recomputed."""
    table = Table(
        "o2-day-counts",
        "O2: the days meeting T4's hypotheses, checked and failing, every day",
        DAY_COLUMNS,
    )
    for setting, label in STEPS:
        try:
            document = source.json(label, RESULTS)
        except Missed as problem:
            table.add(setting, "all units", *([Missing(problem.text)] * 6))
            continue
        every = document.get("every_day") if isinstance(document.get("every_day"), dict) else {}
        found = every.get("units") if isinstance(every.get("units"), dict) else {}
        for unit in units:
            table.add(*_day_counts(setting, unit, found.get(unit)))
        table.add(*_day_counts(setting, POOLED, every.get("all_units")))
    table.notes.append(
        "Counts as `ceiling-report.json` holds them (reading U4 of the module text of "
        "fleetreserve.wp16_o2): a day meets T4's hypotheses when it is one-way, no car leaves "
        "in the window and the recovery supply suffices (an empty fleet does not count); a "
        "unit's pass rests on the days checked, and fails on any day on which V_hind differs "
        "from V_caus. The two settings are never pooled (plan section 3)."
    )
    return [table]


def build(source: Source, units: list) -> list[Table]:
    table = Table(
        "o2-block-product", "O2: V_caus over V_hind for the block product, every day", COLUMNS
    )
    for setting, label in STEPS:
        try:
            document = source.json(label, RESULTS)
        except Missed as problem:
            table.add(setting, "all units", *([Missing(problem.text)] * 8), "")
            continue
        every = document.get("every_day") if isinstance(document.get("every_day"), dict) else {}
        found = every.get("units") if isinstance(every.get("units"), dict) else {}
        for unit in units:
            entry = found.get(unit)
            if not isinstance(entry, dict):
                table.add(setting, unit, *([Missing("no figure: not in the report")] * 8), "")
                continue
            ratio = entry.get("ratio_caus_over_hind", {})
            inside = entry.get("in_window_departure", {})
            affine = entry.get("aff_equals_caus", {})
            table.add(
                setting,
                unit,
                entry.get("days_in_m"),
                _percent(ratio.get("mean")),
                _percent(ratio.get("median")),
                _percent(inside.get("ratio_caus_over_hind", {}).get("mean")),
                inside.get("days"),
                affine.get("days"),
                affine.get("of_days"),
                entry.get("o2_pass"),
                _faults(entry),
            )
        table.notes.append(
            f"{setting}: verdict over the units (as the report gives it): "
            f"{every.get('pass_over_units', 'not in the report')}; "
            f"units with no day to check: {every.get('no_day_to_check', 'not in the report')}"
        )
    table.notes.append("The two settings are never pooled (plan section 3). Readings U1 to U3.")
    return [table]
