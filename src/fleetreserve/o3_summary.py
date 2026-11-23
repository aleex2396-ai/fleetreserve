"""Per-unit statistics, checks and the printed report of outcome O3 (`o3_run`), from the days'
results (`o3_day.DayResult`).

Per unit, product (`o3_day.PRODUCTS`) and value (`o3_day.KEYS`: the ceiling and each method):

- `days`: the sampled days (for LP (27)'s three values, the unit's LP (27) days only, the days
  of `o3_day` reading D9; the others are not counted as missing); `days_with_value`; `missing`:
  the days whose value is missing, by outcome (a timeout, any other solver outcome but
  optimal, a trivial base set with a positive ceiling, a day not built). A missing value is
  never read as zero: it is left out of every mean and median and counted here.
- `kwh`: mean, median, smallest and largest value over the days with a value, and their number.
- for a method only: `share`, the per-day share of the ceiling (value / ceiling) over the days
  where both are present and the ceiling is above `o3_day.ZERO` (mean, median, smallest,
  largest, number of days); `days_zero_ceiling`, the days with both present and a ceiling at most
  `ZERO` (no share); and `ratio_of_means`, the mean value over the mean ceiling on the days where
  both are present (None when that mean ceiling is at most `ZERO`).
- for LP (27)'s bound only (`o3_day.BOUND`): `outcomes`, its days with a value by outcome
  (`o3_lp27_bound`: closed, ceiling, open), and `above_largest_found`, the statistics of the bound
  minus `lp27_largest_found` (kWh) over the days where both are present: the width of the bracket
  in which the largest box over all optimal solutions lies.

Per unit and product, `all_present` gives the same share statistics and ratio of means of every
method over one set of days, those where the ceiling and every method have a value, so a value
missing on some days cannot make the methods' figures cover different days.

LP (27) runs on some of a unit's days only, and only while the unit's time cap lasts
(`o3_run` readings R8 and R9, `o3_day` reading D9). Its three values are counted over its own
days (`DayResult.lp27`): a day that is not one of them is not a missing value. A day of its days
not begun within the cap has them missing with the outcome `CAP_REACHED`, counted among the
missing values and, per unit, by `lp27_reach` (the unit's LP (27) line of the report); the
report's fifth line says which days LP (27) was run on and the cap, and a sixth names every
departure of the run's numbers from the plan's.

The report's fourth line (`RANGE_LINE`) reads LP (27)'s values as a range, as the owner decided on
16 November 2026 (the largest box found, with a proved upper bound beside it; a missing bound
printed as missing): the search's largest found is not the largest box over all optimal
solutions, which lies between it and the bound.

Checks (`failures`), each a fault, never clipped:

- orders that are theorems, on every day where both values are present, within `TOLERANCE`
  relative to `max(1, |right side|)`: each method's value at most the ceiling (the general splits
  are path-known rules; the boxes of LP (24) and LP (27) lie in sets inside the fleet's set, hour 0
  free per corner as in the ceiling); for the ceiling and each method, the block value at most the
  hourly value (a block box is an hourly box); LP (24)'s box at most the general split (LP (24)'s
  car maps are such a split, `o3_split`); the general split without decision-hour gain at most the
  general split (fewer gains); LP (27)'s smallest box found at most its largest found, and its
  largest found at most its bound (a broken bound is a fault, never clipped);
- a programme that ended "infeasible" or "unbounded": every programme here is feasible and
  bounded on a non-empty true set (a zero box at one point of the set is feasible; the sets are
  bounded), so either outcome is a fault, not a measurement;
- a day that could not be built (`DayResult.error`).

Readings (each with its alternative in brackets):

  S1. The headline "X% of the ceiling" is given three ways and none is chosen here: the mean and
      the median of the per-day shares, and the ratio of means (as the WP6 gate compares means,
      the owner's decision of 13 November 2026). [One of them only.]
  S2. A day with a ceiling at most `ZERO` has no share; it is counted (`days_zero_ceiling`) and
      its kWh values enter the kWh statistics. [Shares of 1 or 0 on such days.]
"""

import statistics

from fleetreserve.o3_day import (
    BOUND,
    CAP_REACHED,
    KEYS,
    LABELS,
    LP27_KEYS,
    METHODS,
    PRODUCTS,
    ZERO,
    DayResult,
)

FOUND = "lp27_largest_found"
# The report's fourth line: how LP (27)'s values are read (`o3_day` reading D3).
RANGE_LINE = (
    "LP (27) is a range, not one number: lp27_largest_found is the largest box a seeded local "
    "search found over LP (27)'s optimal solutions (not the largest over all of them; how far "
    "below that it lies is not known); lp27_largest_bound is a number no optimal solution's set "
    "holds a larger box than (proved by a linear relaxation, to the solver's tolerances; missing "
    "when not proved); the largest box over all optimal solutions lies between the two. "
    "lp27_smallest_found is the smallest box over the solutions searched."
)
TOLERANCE = 1e-6  # relative slack of the order checks
FAULT_OUTCOMES = ("infeasible", "unbounded")
# (smaller, larger, why) for the orders between methods (module text).
METHOD_ORDERS = (
    ("lp24", "general_split", "LP (24)'s car maps are a general split"),
    ("general_split_no_hour0_gain", "general_split", "fewer gains"),
    ("lp27_smallest_found", "lp27_largest_found", "smallest found at most largest found"),
    (
        "lp27_largest_found",
        "lp27_largest_bound",
        "a box an optimal solution holds is at most the bound",
    ),
)


def label(result: DayResult) -> str:
    return f"{result.unit} {result.site} {result.date}"


def _le(a: float, b: float) -> bool:
    return a <= b + TOLERANCE * max(1.0, abs(b))


def order_violations(results: list[DayResult]) -> list[str]:
    """One line per broken order (module text)."""
    lines = []
    for r in results:
        for product in PRODUCTS:
            values = r.values[product]
            ceiling = values["ceiling"]
            for method in METHODS:
                value = values[method]
                if value is not None and ceiling is not None and not _le(value, ceiling):
                    lines.append(
                        f"{label(r)} {product}: {method} {value!r} above the ceiling {ceiling!r}"
                    )
            for smaller, larger, why in METHOD_ORDERS:
                a, b = values[smaller], values[larger]
                if a is not None and b is not None and not _le(a, b):
                    lines.append(
                        f"{label(r)} {product}: {smaller} {a!r} above {larger} {b!r} ({why})"
                    )
        for key in KEYS:
            block, hourly = r.values["block"][key], r.values["hourly"][key]
            if block is not None and hourly is not None and not _le(block, hourly):
                lines.append(f"{label(r)}: {key} block {block!r} above hourly {hourly!r}")
    return lines


def failures(results: list[DayResult]) -> list[str]:
    """Every fault: days not built, programmes infeasible or unbounded, broken orders."""
    lines = [f"{label(r)}: the day could not be built: {r.error}" for r in results if r.error]
    for r in results:
        for product in PRODUCTS:
            for key in KEYS:
                if r.outcomes[product][key] in FAULT_OUTCOMES:
                    lines.append(
                        f"{label(r)} {product}: {key} ended {r.outcomes[product][key]!r}, "
                        "which a non-empty true set cannot give"
                    )
    return lines + order_violations(results)


def _stats(numbers: list[float]) -> dict:
    if not numbers:
        return {"days": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "days": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _shares(both: list[DayResult], product: str, key: str) -> dict:
    """Share statistics, zero-ceiling days and ratio of means over days with both present."""
    shares = [r.share(product, key) for r in both]
    mean_ceiling = statistics.fmean(r.values[product]["ceiling"] for r in both) if both else 0
    return {
        "share": _stats([s for s in shares if s is not None]),
        "days_zero_ceiling": sum(s is None for s in shares),
        "ratio_of_means": (
            statistics.fmean(r.values[product][key] for r in both) / mean_ceiling
            if both and mean_ceiling > ZERO
            else None
        ),
    }


def value_summary(results: list[DayResult], product: str, key: str) -> dict:
    """The statistics of one value over a unit's days (module text)."""
    if key in LP27_KEYS:
        results = [r for r in results if r.lp27]
    present = [r for r in results if r.values[product][key] is not None]
    missing: dict[str, int] = {}
    for r in results:
        if r.values[product][key] is None:
            outcome = r.outcomes[product][key]
            missing[outcome] = missing.get(outcome, 0) + 1
    entry = {
        "days": len(results),
        "days_with_value": len(present),
        "missing": dict(sorted(missing.items())),
        "kwh": _stats([r.values[product][key] for r in present]),
    }
    if key == BOUND:
        counts: dict[str, int] = {}
        for r in present:
            counts[r.outcomes[product][key]] = counts.get(r.outcomes[product][key], 0) + 1
        entry["outcomes"] = dict(sorted(counts.items()))
        pairs = [r for r in present if r.values[product][FOUND] is not None]
        entry["above_largest_found"] = _stats(
            [r.values[product][key] - r.values[product][FOUND] for r in pairs]
        )
    if key in METHODS:
        both = [r for r in present if r.values[product]["ceiling"] is not None]
        entry.update(_shares(both, product, key))
    return entry


def unit_summary(results: list[DayResult]) -> dict:
    """`{product: {key: value_summary}}` for one unit's days."""
    return {
        product: {key: value_summary(results, product, key) for key in KEYS} for product in PRODUCTS
    }


def all_present(results: list[DayResult]) -> dict:
    """`{product: {"days": n, method: shares}}` over the days of one unit where the ceiling and
    every method have a value (module text)."""
    found = {}
    for product in PRODUCTS:
        days = [r for r in results if all(r.values[product][k] is not None for k in KEYS)]
        found[product] = {"days": len(days)}
        found[product].update({method: _shares(days, product, method) for method in METHODS})
    return found


def lp27_reach(results: list[DayResult]) -> dict:
    """LP (27)'s days of one unit: `days` (the days it was asked for), `not_reached` (those not
    begun within the unit's time cap: their three values are missing, never zero) and `reached`;
    a day counts once, whichever of its values is missing."""
    asked = [r for r in results if r.lp27]
    capped = [r for r in asked if r.outcomes["hourly"][BOUND] == CAP_REACHED]
    return {
        "days": len(asked),
        "reached": len(asked) - len(capped),
        "not_reached": len(capped),
        "not_reached_days": [f"{r.site} {r.date}" for r in capped],
    }


def missing_total(units: dict) -> dict:
    """The missing values over every unit, product and value, by outcome."""
    total: dict[str, int] = {}
    for products in units.values():
        for keys in products.values():
            for entry in keys.values():
                for outcome, count in entry["missing"].items():
                    total[outcome] = total.get(outcome, 0) + count
    return dict(sorted(total.items()))


def _fmt(value, spec: str = ".3f") -> str:
    return "-" if value is None else format(value, spec)


def _settings(lp27: dict) -> str:
    return ", ".join(f"{name} {value}" for name, value in lp27["settings"].items())


def _missing_text(missing: dict) -> str:
    return ", ".join(f"{n} {outcome}" for outcome, n in missing.items()) or "none"


def _bracket_text(entry: dict) -> str:
    above = entry["above_largest_found"]
    return (
        f"; above {FOUND} by mean {_fmt(above['mean'], '.4f')} median "
        f"{_fmt(above['median'], '.4f')} largest {_fmt(above['max'], '.4f')} kWh on "
        f"{above['days']} days; outcomes {_missing_text(entry['outcomes'])} (closed: met the "
        "largest found; ceiling: the relaxations' bound at or above the ceiling; open: the "
        "relaxations allowed ran out first)"
    )


def summary_lines(document: dict) -> list[str]:
    """The report printed and written by `o3_run` (every number from `document`)."""
    arguments = document["arguments"]
    missing = document["missing_values"]
    problems = document["failures"]
    lines = [
        "O3, HEADLINE MEASUREMENT ON TRUE SETS (plan section 11): LP (24), LP (27) and the general "
        "split against the ceiling V_hind (hour 0 free); no offer, forecast or calibration.",
        f"truth build {document['inputs']['truth_run']} ({document['inputs']['setting']} "
        f"setting); units {', '.join(arguments['units'])}; {arguments['days']} days per unit, "
        f"seed {arguments['seed']}; base set {arguments['base_set']}; rho {arguments['rho']}; "
        f"time limits (s): LP (24) {arguments['seconds']['lp24']}, others "
        f"{arguments['seconds']['others']}, box {arguments['seconds']['box']}; LP (27) search "
        f"effort {arguments['lp27']['effort_share']} ({_settings(arguments['lp27'])}), bound "
        f"relaxations per product {arguments['lp27']['bound_nodes']}.",
        f"MISSING VALUES (never read as zero): {sum(missing.values())} ({_missing_text(missing)})",
        RANGE_LINE,
        f"LP (27) DAYS: LP (27) is run on the first {arguments['lp27_days']} days of each unit's "
        f"draw (by hash rank; the other sampled days, {arguments['days']} per unit asked, have "
        "its three values missing and are left out of its statistics), and a unit's LP (27) is "
        f"not begun on a day after {arguments['lp27_cap_seconds']} s have passed since the unit's "
        "days were started: the days not reached are missing, never zero.",
    ]
    if arguments["departures_from_plan"]:
        lines.append(
            "DEPARTURES FROM THE PLAN'S NUMBERS (sections 11 O3 and 13): "
            + "; ".join(arguments["departures_from_plan"])
            + "."
        )
    lines += [f"FAILED: {line}" for line in problems]
    for unit, entry in document["units"].items():
        sample = document["sample"][unit]
        lines += [
            "",
            f"UNIT {unit}: {sample['sampled_days']} of {sample['counting_days']} counting days "
            "sampled",
        ]
        reach = document["lp27_cap"][unit]
        lines.append(
            f"  LP (27): {reach['days']} days asked for, {reach['reached']} reached, "
            f"{reach['not_reached']} not reached within the cap of {arguments['lp27_cap_seconds']} "
            "s (missing, never zero)"
        )
        for product, keys in entry.items():
            ceiling = keys["ceiling"]
            lines.append(
                f"  {product}: ceiling kWh/day mean {_fmt(ceiling['kwh']['mean'])} median "
                f"{_fmt(ceiling['kwh']['median'])} on {ceiling['kwh']['days']} days; missing "
                f"{_missing_text(ceiling['missing'])}"
            )
            for method in METHODS:
                m = keys[method]
                lines.append(
                    f"    {method} ({LABELS[method]}): kWh/day mean {_fmt(m['kwh']['mean'])} "
                    f"median {_fmt(m['kwh']['median'])} on {m['kwh']['days']} days; share of "
                    f"ceiling mean {_fmt(m['share']['mean'], '.4f')} median "
                    f"{_fmt(m['share']['median'], '.4f')} on {m['share']['days']} days "
                    f"({m['days_zero_ceiling']} with ceiling 0); ratio of means "
                    f"{_fmt(m['ratio_of_means'], '.4f')}; missing {_missing_text(m['missing'])}"
                    + (_bracket_text(m) if method == BOUND else "")
                )
            common = document["all_present"][unit][product]
            parts = [
                f"{method} mean {_fmt(common[method]['share']['mean'], '.4f')} on "
                f"{common[method]['share']['days']} days, ratio of means "
                f"{_fmt(common[method]['ratio_of_means'], '.4f')}"
                for method in METHODS
            ]
            lines.append(
                f"    share of ceiling on the {common['days']} days where the ceiling and every "
                f"method have a value: {'; '.join(parts)}"
            )
    lines.append("")
    lines.append("CHECKS FAILED" if problems else "all checks held")
    return lines
