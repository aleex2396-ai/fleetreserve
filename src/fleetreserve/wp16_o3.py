"""Outcome O3: LP (24), LP (27) and the general split against the ceiling (plan section 11, O3).

"O3 Headline measurement, true sessions, sampled days drawn from all counting days: LP (24),
LP (27), general split at X, Y, Z% of the ceiling; LP (27) as a range: the largest box found,
with a proved upper bound beside it, a missing bound printed as missing."

The step `o3` (`fleetreserve.o3_run`) writes `o3.json`: per unit and product (hourly, block) and
value, the sampled days, the days with a value, the missing values by outcome, the statistics of
the value in kWh and, for a method, the share of the ceiling (mean, median and the ratio of means).

Readings (alternatives in brackets):
  T1. The share "of the ceiling" is given three ways, as `o3_summary` reading S1 gives it and none
      chosen here: the ratio of means (the mean value over the mean ceiling, the headline column),
      and the mean and the median of the per-day shares. [Alternative: one of them only.]
  T2. LP (27) is a range: `lp27_largest_found` with `lp27_largest_bound` beside it, and
      `lp27_smallest_found`; a missing bound is printed as missing, with the count of days and
      the outcomes behind it. LP (27) runs on the unit's first 100 days only and within the unit's
      time cap (`lp27_cap`): the days not reached are counted in their own table.
  T4. Because the days differ by method (LP (27) runs on 100 days, the others on 200, and a
      missing value leaves a day out), the figure and the table `o3-same-days` read the block
      `all_present` of `o3.json`: the same days for every value of the unit. The per-unit table
      beside it says in its note that its rows are on different days.
  T3. Figures are per unit, as the file has them. The file holds no figure over all units, and none
      is made in `build`: the sampled days differ by unit, so a figure over units would be a choice
      of weights that the plan does not state (a question for the owner; T5 answers it).
  T5. `build_combined` adds two tables beside those, made after held-out results were seen (the
      owner's decision of 24 November 2026): the rule of O1, the sum over units of each unit's
      value per counting day, carried over to O3. Reading: a unit's value per counting day is its
      mean over the unit's sampled days on which every value is present (the days of
      `o3-same-days`), for a method and for the ceiling alike; the combined share of a method is
      100 x (sum over units of the unit's mean value) / (sum over units of the unit's mean
      ceiling). The means come from `o3-days.csv` (the per-day values, which `o3.json` does not
      hold: it holds ratios, not the means they are made of). A unit with no such day is left out
      of every sum and named; no unit is given a weight by its number of days. [Alternatives: the
      mean of the unit shares; the share over all days pooled; the mean of the per-day shares.]
"""

import csv
import io
import math
import statistics

from fleetreserve.o3_summary import ZERO
from fleetreserve.wp16_cells import Missing, Table
from fleetreserve.wp16_drive import Missed, Source

LABEL = "o3"
RESULTS = "o3.json"
PRODUCTS = ("hourly", "block")
METHODS = (
    "lp24",
    "general_split",
    "general_split_no_hour0_gain",
    "lp27_smallest_found",
    "lp27_largest_found",
    "lp27_largest_bound",
)
COLUMNS = (
    "unit",
    "product",
    "value",
    "days with a value",
    "of days",
    "missing, by outcome",
    "mean (kWh)",
    "share of the ceiling: ratio of means (%)",
    "mean of per-day shares (%)",
    "median of per-day shares (%)",
)
SAME_COLUMNS = (
    "unit",
    "product",
    "days with every value",
    "value",
    "share of the ceiling: ratio of means (%)",
    "mean of per-day shares (%)",
    "median of per-day shares (%)",
)
DAYS_FILE = "o3-days.csv"
COMBINED_COLUMNS = (
    "product",
    "value",
    "units summed",
    "days summed",
    "sum of unit mean values (kWh)",
    "sum of unit mean ceilings (kWh)",
    "combined share of the ceiling (%)",
    "units left out",
)
UNIT_COLUMNS = (
    "unit",
    "product",
    "days with every value",
    "value",
    "mean value (kWh)",
    "mean ceiling (kWh)",
)
REACH_COLUMNS = (
    "unit",
    "days sampled for LP (24) and the general split",
    "days asked of LP (27)",
    "days LP (27) reached",
    "days not reached within the cap",
)


def _percent(value):
    return None if not isinstance(value, int | float) else 100.0 * value


def _missing_text(missing) -> str:
    if not isinstance(missing, dict) or not missing:
        return ""
    return ", ".join(f"{outcome}: {count}" for outcome, count in sorted(missing.items()))


def build(source: Source, units: list) -> tuple[list[Table], list[dict]]:
    table = Table("o3-ceiling-shares", "O3: methods against the ceiling, per unit", COLUMNS)
    reach = Table("o3-days", "O3: sampled days and LP (27)'s reach, per unit", REACH_COLUMNS)
    same = Table(
        "o3-same-days", "O3: every method on the same days (all_present), per unit", SAME_COLUMNS
    )
    try:
        document = source.json(LABEL, RESULTS)
    except Missed as problem:
        table.add("all units", "", "", *([Missing(problem.text)] * 7))
        reach.add("all units", *([Missing(problem.text)] * 4))
        same.add("all units", "", Missing(problem.text), *([""] * 4))
        return [table, reach, same], []
    found = document.get("units") if isinstance(document.get("units"), dict) else {}
    caps = document.get("lp27_cap") if isinstance(document.get("lp27_cap"), dict) else {}
    points = []
    for unit in units:
        entry = found.get(unit)
        if not isinstance(entry, dict):
            table.add(unit, "", "", *([Missing("no figure: not in the run")] * 7))
            reach.add(unit, *([Missing("no figure: not in the run")] * 4))
            same.add(unit, "", Missing("no figure: not in the run"), *([""] * 4))
            continue
        for product in PRODUCTS:
            values = entry.get(product, {})
            for value in ("ceiling", *METHODS):
                one = values.get(value)
                if not isinstance(one, dict):
                    continue
                share = one.get("share", {}) if isinstance(one.get("share"), dict) else {}
                table.add(
                    unit,
                    product,
                    value,
                    one.get("days_with_value"),
                    one.get("days"),
                    _missing_text(one.get("missing")),
                    one.get("kwh", {}).get("mean"),
                    _percent(one.get("ratio_of_means")),
                    _percent(share.get("mean")),
                    _percent(share.get("median")),
                )
        hourly = entry.get("hourly", {})
        every = document.get("all_present", {}).get(unit, {})
        for product in PRODUCTS:
            both = every.get(product) if isinstance(every, dict) else None
            if not isinstance(both, dict):
                same.add(unit, product, Missing("no figure: no all_present block"), *([""] * 4))
                continue
            for method in METHODS:
                one = both.get(method)
                if not isinstance(one, dict):
                    continue
                share = one.get("share", {}) if isinstance(one.get("share"), dict) else {}
                same.add(
                    unit,
                    product,
                    both.get("days"),
                    method,
                    _percent(one.get("ratio_of_means")),
                    _percent(share.get("mean")),
                    _percent(share.get("median")),
                )
        together = every.get("hourly", {}) if isinstance(every, dict) else {}
        points.append(
            (
                unit,
                *(
                    _percent(together.get(value, {}).get("ratio_of_means"))
                    for value in (
                        "lp24",
                        "general_split",
                        "lp27_largest_found",
                        "lp27_largest_bound",
                    )
                ),
            )
        )
        cap = caps.get(unit, {}) if isinstance(caps.get(unit), dict) else {}
        reach.add(
            unit,
            hourly.get("ceiling", {}).get("days"),
            cap.get("days"),
            cap.get("reached"),
            cap.get("not_reached"),
        )
    table.notes.append(
        "Each row is over the days that have its value: LP (24) and the general split on up to 200 "
        "days of a unit, LP (27) on up to 100 and within its time cap, so the rows of one unit are "
        "on different sets of days and are not to be compared; the same-days table (o3-same-days) "
        "and the figure use the days on which every value is present (reading T4)."
    )
    same.notes.append(
        "Days on which the ceiling and every method have a value (`all_present`); a unit with "
        "none has no ratio. A missing bound is a missing value, never 0."
    )
    figures = [{"kind": "o3", "points": points}]
    return [table, reach, same], figures


def _read_days(source: Source) -> list[dict]:
    """The rows of `o3-days.csv`; raises `Missed` when the step has no clean file or a column the
    combined tables read is not there."""
    data = source.output(LABEL, DAYS_FILE)
    try:
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as problem:
        raise Missed(f"unreadable: {LABEL}/{DAYS_FILE}: {problem}") from problem
    wanted = ("unit", "product", *(f"{key}_kwh" for key in ("ceiling", *METHODS)))
    absent = [name for name in wanted if not rows or name not in rows[0]]
    if absent:
        raise Missed(f"unreadable: {LABEL}/{DAYS_FILE}: no column {absent[0]}")
    return rows


def _number(text) -> float | None:
    try:
        value = float(text) if text not in (None, "") else None
    except ValueError as problem:
        raise Missed(f"unreadable: {LABEL}/{DAYS_FILE}: {text!r} is not a number") from problem
    if value is not None and not math.isfinite(value):
        raise Missed(f"unreadable: {LABEL}/{DAYS_FILE}: {text!r} is not a finite number")
    return value


def _compare_day_counts(source: Source, units: list, days: dict) -> None:
    """Each unit's kept-day count must equal the `days` of `all_present` in `o3.json` for the unit
    and product; a difference raises `Missed` naming the unit. Where `o3.json` cannot be read or
    holds no such count for a unit and product, there is nothing to compare and the count stands."""
    try:
        every = source.json(LABEL, RESULTS).get("all_present")
    except Missed:
        return
    if not isinstance(every, dict):
        return
    for unit in units:
        for product in PRODUCTS:
            block = every.get(unit, {}).get(product) if isinstance(every.get(unit), dict) else None
            expected = block.get("days") if isinstance(block, dict) else None
            kept = len(days.get((unit, product), []))
            if isinstance(expected, int) and not isinstance(expected, bool) and expected != kept:
                raise Missed(
                    f"unreadable: {LABEL}/{DAYS_FILE}: unit {unit}, {product}: {kept} days with "
                    f"every value, but {RESULTS} gives {expected}"
                )


def _share(summed: float, ceiling: float):
    if ceiling <= ZERO:
        return Missing("no figure: the summed ceiling is zero")
    return 100.0 * summed / ceiling


def build_combined(source: Source, units: list) -> list[Table]:
    """The two tables of reading T5: `o3-combined` (one row per product and value) and
    `o3-combined-units` (each unit's mean value and mean ceiling, so a reader can add them)."""
    combined = Table(
        "o3-combined",
        "O3: one combined share of the ceiling per method (applied after results)",
        COMBINED_COLUMNS,
    )
    each = Table(
        "o3-combined-units",
        "O3: each unit's mean value and mean ceiling on the days with every value",
        UNIT_COLUMNS,
    )
    try:
        rows = _read_days(source)
        days: dict[tuple, list[dict]] = {}
        for row in rows:
            if row["unit"] in units and row["product"] in PRODUCTS:
                every = [_number(row[f"{key}_kwh"]) for key in ("ceiling", *METHODS)]
                if all(value is not None for value in every):
                    days.setdefault((row["unit"], row["product"]), []).append(row)
        _compare_day_counts(source, units, days)
    except Missed as problem:
        combined.add("all products", "", *([Missing(problem.text)] * 5), "")
        each.add("all units", "", Missing(problem.text), "", "", "")
        combined.notes.append(NOTE_COMBINED)
        each.notes.append(NOTE_UNITS)
        return [combined, each]
    for product in PRODUCTS:
        kept, left = [], []
        for unit in units:
            if (unit, product) in days:
                kept.append(unit)
            else:
                left.append(unit)
                each.add(
                    unit,
                    product,
                    0,
                    Missing("left out of the sums: no day on which every value is present"),
                    "",
                    "",
                )
        means: dict[tuple, float] = {}
        for unit in kept:
            for key in ("ceiling", *METHODS):
                means[(unit, key)] = statistics.fmean(
                    float(row[f"{key}_kwh"]) for row in days[(unit, product)]
                )
            for method in METHODS:
                each.add(
                    unit,
                    product,
                    len(days[(unit, product)]),
                    method,
                    means[(unit, method)],
                    means[(unit, "ceiling")],
                )
        counted = sum(len(days[(unit, product)]) for unit in kept)
        for method in METHODS:
            if not kept:
                combined.add(
                    product,
                    method,
                    0,
                    0,
                    Missing("no figure: no unit has a day on which every value is present"),
                    "",
                    Missing("no figure: no unit has a day on which every value is present"),
                    ", ".join(left),
                )
                continue
            summed = sum(means[(unit, method)] for unit in kept)
            ceiling = sum(means[(unit, "ceiling")] for unit in kept)
            combined.add(
                product,
                method,
                len(kept),
                counted,
                summed,
                ceiling,
                _share(summed, ceiling),
                ", ".join(left),
            )
    combined.notes.append(NOTE_COMBINED)
    combined.notes.extend(_left_out_notes(units, days))
    each.notes.append(NOTE_UNITS)
    return [combined, each]


def _left_out_notes(units: list, days: dict) -> list[str]:
    lines = []
    for product in PRODUCTS:
        left = [unit for unit in units if (unit, product) not in days]
        lines.append(
            f"Left out of the {product} sums, no day on which every value is present: "
            + (", ".join(left) if left else "no unit")
            + "."
        )
    return lines


NOTE_COMBINED = (
    "Applied after results: the rule of outcome O1, the sum over units of each unit's value per "
    "counting day, carried over to O3 after the held-out results were seen (the owner's decision "
    "of 24 November 2026). It fills no slot of plan section 11. The per-unit values stay in "
    "`o3-same-days` and `o3-ceiling-shares`. Reading, not stated by the plan: a unit's value per "
    "counting day is its mean over the unit's sampled days on which the ceiling and every method "
    "have a value (the days of `o3-same-days`), for a method and for the ceiling alike; the "
    "combined share is 100 x (sum over units of the unit's mean value) / (sum over units of the "
    "unit's mean ceiling), so a unit counts once, whatever its number of days. A unit with no such "
    "day is left out of every sum of its product and is named in the last column; a missing "
    "LP (27) bound on every sampled day leaves a unit out for every method. The sums then cover "
    "fewer than nine units; the column 'units summed' gives how many. The sum of unit mean "
    "ceilings is the same in every row of a product."
)
NOTE_UNITS = (
    "The inputs of `o3-combined`: per unit, product and value, the mean of the value and the mean "
    "of the ceiling over the days with every value present, from the per-day file of the O3 step. "
    "Adding the column of means over the units of a product gives that table's sums."
)
