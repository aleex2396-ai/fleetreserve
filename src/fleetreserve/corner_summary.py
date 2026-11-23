"""The results of a corner search (`corner_run`): per method and per pass-mark check, the figures
over the searched days, their definitions, the per-day rows and the plain-text summary.

`method_entry(scale, reason, days)`: one method of one unit and variant, from its calibrated tier
A scale (and why it is 0, if it is) and its `corner_search.CornerDay` of each searched day, in the
searched order. A method whose scale is 0 offers nothing: its status is "no offer" and it has no
break figures (reading W4 of `corner_run`), never "does not break". Otherwise, with "days with a
result" the searched days whose status is "breaks" or "holds" (a day "timed out" has none):

- `share_of_days_breaking`: days that break over days with a result (O7's Y% and Y0%); for the
  fixed plan the corners give the worst case over the box (reading P1 of `corner_search`), for
  every other method a lower bound of it (an interior path can be worse than every corner);
- `days_breaking_by_slot_loss`, `days_with_driver_shortfall`: days with a corner that fails an
  hour (slot loss above 0), or whose undelivered protected energy is above `replay.FAILURE_KWH`
  (a day can be in both);
- `days_short_only_below_tolerance`: days that hold but have a corner with a driver loss above 0
  (undelivered protected energy above 0 and at most `FAILURE_KWH`, whether rounding left it or
  the cars are short by that little), not counted as breaks (reading K1 of `corner_run`); a day
  that breaks is not among them, whatever its other corners;
- `mean_breaking_paths`: the mean number of breaking corners over the days with a result;
- `worst_slot_loss`, `worst_driver_loss`: the largest over the days with a result;
- `breaking_days_audited_misses`: breaking days whose audited reading holds, and its share of the
  breaking days; a breaking day whose audited rows timed out is counted apart;
- `days_timed_out`, `resolves`: the days without a result and the re-solves by outcome;
- `solver_counts`: for a controller that solves programmes, its solves by outcome, hours planned
  and hours planned by its fall-back, summed over every corner of every searched day
  (`compare_day.sum_counts`); None for the other methods;
- `deviation_counts`: for a controller that counts its deviations (`max-entropy-feedback`,
  `mef_controller.COUNTS`), its counts summed over every corner of every searched day, key by
  key (`compare_day.sum_deviations`); written only for such a method (and only when a day was
  searched), so the entries of every other method are unchanged.

`pass_mark_entry(days, dates, checks)`: the closed-form check (`corner_search.closed_form_check`)
of one unit and variant, over its first searched days at every scale checked: the pairs (day, scale)
checked; how many agree strictly and how many agree (strictly, or with the reason "within
tolerance": reading K3 of `corner_run`); every pair that does not agree strictly counted by
reason; the largest slot-loss difference; each disagreement with its reason; and for the driver
loss the pairs whose corner maximum is above the bound (the every-scale bound,
`corner_search.closed_form_check`), with the largest excess, how many of them have on every corner
of the largest driver loss a car that drew less than planned, and how many of them have some
corner of that loss with no failing hour (`corner_search.driver_flags`; both counted over the
pairs above the bound only); beside them, for comparison only, the pairs above the earlier bound
and the largest excess over it. `agreed` is True when at least one pair was checked and every pair
agrees (nothing checked is not agreed); neither driver-loss bound enters it (reading K4).

`unchecked_entry(reason)`: on the sealed route, the check entry of a unit and variant the check
does not run on (reading H7 of `corner_run`): `status` "not checked" (`NOT_CHECKED`) and its
`reason`, and no figure; it is never agreed.

`day_rows(...)`: one CSV row per unit, variant, method and searched day (`ROW_FIELDS`).

`summary_lines(document, held_out=False)`: per unit, one table per variant with a row per method
(then a line per method with solver counts and a line per method with deviation counts), then the
pass-mark table; a unit that cannot be split gets its status and no numbers. With `held_out` (a
run on the sealed route, readings H5 and H6 of `corner_run`) the check's entries are read under
`HELD_OUT_CHECK_KEY`, the name `corner_run.mark_route` gives them there, and its three headings
(`HEADINGS`) name it a closed-form property check on held-out days; the one sentence of such a
summary that says what it is not is the run's label. There (reading H7) its paragraph names the
one scale as `run.check_scales` words it, a unit and variant not checked has the line "<variant>:
not checked: <reason>" in its unit's table, and the line over all units says how many were not
checked, or "nothing checked on this run" when none was, and, when the held-out truth build
abstained a unit (`sealed_route.abstained_of`), says how many such units were not checked and
count as not passing. Without `held_out` every line is as before.
"""

from collections import Counter

from fleetreserve import rotation_whatif, sealed_route
from fleetreserve.compare_day import sum_counts, sum_deviations
from fleetreserve.corner_search import BREAKS, DRIVER_TOLERANCE, HOLDS, NO_OFFER, WITHIN_TOLERANCE
from fleetreserve.gate_summary import ROUTE_KEY

# The name of the closed-form check's fields: off the sealed route the check is WP11's pass mark;
# on it, on held-out days, a property check, named so (readings H5 and H6 of `corner_run`).
CHECK_KEY = "pass_mark"
HELD_OUT_CHECK_KEY = "closed_form_check"
# The check's three headings in the summary (its paragraph, each unit's table, the line over all
# units), keyed by `held_out` of `summary_lines`.
HEADINGS = {
    False: ("Pass mark", " Pass mark (plan 13 WP11)", "Pass mark over all units"),
    True: (
        "Closed-form property check on held-out days",
        " Closed-form property check on held-out days",
        "Closed-form property check over all units",
    ),
}


def check_key(held_out: bool) -> str:
    """The name of the check's fields in a run on the sealed route (`held_out`) or off it."""
    return HELD_OUT_CHECK_KEY if held_out else CHECK_KEY


# The status of a unit and variant the check does not run on, on the sealed route (reading H7 of
# `corner_run`); off the route every check entry is `pass_mark_entry`'s, without a status.
NOT_CHECKED = "not checked"
# The line over all units of a route run on which no unit and variant was checked (reading H7).
NOTHING_CHECKED = "nothing checked on this run"


def unchecked_entry(reason: str) -> dict:
    """The check entry of a unit and variant not checked on the sealed route (module text)."""
    return {"status": NOT_CHECKED, "reason": reason}


def is_checked(entry: dict) -> bool:
    """Whether a check entry holds a check (`pass_mark_entry`) and not `unchecked_entry`."""
    return entry.get("status") != NOT_CHECKED


ROW_FIELDS = (
    "unit",
    "variant",
    "method",
    "day",
    "date",
    "site",
    "n_service",
    "scale",
    "status",
    "n_paths",
    "breaking_paths",
    "slot_breaking_paths",
    "driver_short_paths",
    "short_below_tolerance_paths",
    "timed_out_paths",
    "worst_slot_loss",
    "worst_driver_loss",
    "worst_path",
    "worst_first_failure",
    "audited_status",
    "audited_breaking_rows",
    "audited_worst_slot_loss",
    "audited_worst_driver_loss",
    "resolves",
    "solver_counts",
)

ROW_DEFINITIONS = {
    "unit": "the gate unit",
    "variant": "'forecast' or 'oracle-departure' (corner_run.VARIANTS)",
    "method": "the method's name in compare_methods.METHODS",
    "day": "the day's index among the unit's counting days",
    "date": "the day's date",
    "site": "the day's site",
    "n_service": "the day's number of service hours",
    "scale": "the method's calibrated tier A scale",
    "status": "'breaks', 'holds', 'timed out' or 'no offer' (corner_search)",
    "n_paths": "corners run: 2 to the number of service hours (0 for 'no offer')",
    "breaking_paths": (
        "corners without a timeout that fail an hour or leave more than replay.FAILURE_KWH of "
        "protected energy undelivered"
    ),
    "slot_breaking_paths": "corners without a timeout whose slot loss is above 0",
    "driver_short_paths": (
        "corners without a timeout whose undelivered protected energy is above replay.FAILURE_KWH"
    ),
    "short_below_tolerance_paths": (
        "corners without a timeout that do not break but have a driver loss above 0"
    ),
    "timed_out_paths": "corners on which a re-solve timed out",
    "worst_slot_loss": "largest slot loss over the corners without a timeout",
    "worst_driver_loss": "largest driver loss over the corners without a timeout",
    "worst_path": (
        "one worst corner, + for the high end and - for the low end in each service hour: the "
        "first in corner order with the largest slot loss, then driver loss; empty if none breaks"
    ),
    "worst_first_failure": "the worst corner's first failing service hour, empty if none",
    "audited_status": "what the day's audited rows say, read from the same corner runs",
    "audited_breaking_rows": "audited rows (repeats counted) that break",
    "audited_worst_slot_loss": "largest slot loss over the audited rows",
    "audited_worst_driver_loss": "largest driver loss over the audited rows",
    "resolves": "re-solves by outcome, 'outcome:count' joined by ';'",
    "solver_counts": (
        "for a controller that solves programmes: solves by outcome as 'outcome:count', then "
        "'hours:N' and 'fallback_hours:N', joined by ';'; empty for the other methods"
    ),
}

DEFINITIONS = {
    "method.scale": (
        "the method's calibrated tier A scale on the unit's split (reading W1): "
        "dial_splits.score_split on the split's calibration days, as compare_run calibrates"
    ),
    "method.scale_reason": "why the scale is 0, empty otherwise (dial_splits)",
    "method.status": "'searched', or 'no offer' when the scale is 0 (reading W4)",
    "method.days_searched": "searched days of the unit and variant (reading W2)",
    "method.days_with_result": "searched days whose status is 'breaks' or 'holds'",
    "method.days_breaking": (
        "days with at least one corner that fails an hour or leaves more than replay.FAILURE_KWH "
        "of protected energy undelivered (reading K1)"
    ),
    "method.share_of_days_breaking": (
        "days_breaking / days_with_result (O7's Y% and Y0%): the worst case over the box for the "
        "fixed plan, a lower bound of it for every other method"
    ),
    "method.days_breaking_by_slot_loss": "days with a corner whose slot loss is above 0",
    "method.days_with_driver_shortfall": (
        "days with a corner whose undelivered protected energy is above replay.FAILURE_KWH"
    ),
    "method.days_short_only_below_tolerance": (
        "days that hold but have a corner with a driver loss above 0: undelivered protected "
        "energy above 0 and at most replay.FAILURE_KWH (rounding, or cars short by that little), "
        "not counted as breaking"
    ),
    "method.days_timed_out": "days whose status is 'timed out': a re-solve timed out, no result",
    "method.mean_breaking_paths": "mean number of breaking corners over the days with a result",
    "method.worst_slot_loss": "largest corner slot loss over the days with a result",
    "method.worst_driver_loss": "largest corner driver loss over the days with a result",
    "method.breaking_days_audited_misses": (
        "breaking days on which the audited 8 paths, read from the same corner runs, hold"
    ),
    "method.share_of_breaking_days_audited_misses": "breaking_days_audited_misses / days_breaking",
    "method.breaking_days_audited_timed_out": "breaking days whose audited rows have no result",
    "method.resolves": "re-solves by outcome over every searched corner (re-solve method only)",
    "method.solver_counts": (
        "a controller that solves programmes: solves by outcome, hours planned, hours planned by "
        "its fall-back, over every searched corner (compare_day.sum_counts); null otherwise"
    ),
    "pass_mark.days_checked": "the first searched days checked (at most corner_run.CHECK_DAYS)",
    "pass_mark.days": "the checked days' indices among the unit's counting days, in order",
    "pass_mark.scales": "the scales of each checked day (dial_splits.GRID)",
    "pass_mark.pairs": "(day, scale) pairs checked",
    "pass_mark.pairs_agreed_strict": (
        "pairs whose closed-form first failing hour equals the corner worst case and whose slot "
        "losses differ by at most corner_search.SLOT_TOLERANCE"
    ),
    "pass_mark.pairs_agreed": (
        "pairs that agree strictly, or whose difference fixed_plan_loss.compare_with_replay "
        "labels 'within tolerance' (reading K3)"
    ),
    "pass_mark.differences_by_reason": "the pairs that do not agree strictly, counted by reason",
    "pass_mark.largest_slot_difference": (
        "largest |closed-form slot loss - corner worst slot loss| over the pairs"
    ),
    "pass_mark.disagreements": (
        "every pair that does not agree (with the tolerance), with the reason of the difference"
    ),
    "pass_mark.disagreements.day": "the disagreeing pair's day: its index among the counting days",
    "pass_mark.disagreements.date": "that day's date",
    "pass_mark.disagreements.scale": "the pair's scale",
    "pass_mark.disagreements.closed_form_first_failure": (
        "the closed form's first failing service hour, null if none"
    ),
    "pass_mark.disagreements.corner_first_failure": (
        "the earliest first failing service hour over the corners, null if none"
    ),
    "pass_mark.disagreements.closed_form_slot_loss": "the closed form's slot loss",
    "pass_mark.disagreements.corner_worst_slot_loss": "the largest slot loss over the corners",
    "pass_mark.disagreements.reason": (
        "the reason of the difference (fixed_plan_loss.compare_with_replay, or 'slot losses "
        "differ')"
    ),
    "pass_mark.disagreements_by_reason": "the disagreements counted by reason",
    "pass_mark.driver_pairs_above_bound": (
        "pairs whose corner maximum driver loss is above the driver-loss bound, the every-scale "
        "bound (gate_calibrate.every_scale_loss), by more than corner_search.DRIVER_TOLERANCE"
    ),
    "pass_mark.largest_driver_excess": "largest corner maximum minus bound, 0 when never above",
    "pass_mark.driver_above_bound_limited": (
        "of the pairs above the bound, those on which every corner of the largest driver loss "
        "(within corner_search.DRIVER_TOLERANCE of it) has a car that drew less than its planned "
        "charge in some hour, the decision hour included (rate, room or presence): the bound "
        "counts each car's decision-hour draw and each service hour at no more than the car can "
        "draw in it, so it holds at every scale with no failing hour and at every other scale "
        "(up to the errors of V3 and V7 of fixed_plan_loss's module text), and on a pair above it "
        "every such corner has a car short of its plan in a service hour (the count is the same "
        "with the decision hour left out)"
    ),
    "pass_mark.driver_above_bound_no_slot_failure": (
        "of the pairs above the bound, those on which some corner of the largest driver loss "
        "(within corner_search.DRIVER_TOLERANCE of it) fails no hour"
    ),
    "pass_mark.earlier_driver_pairs_above_bound": (
        "for comparison only: pairs whose corner maximum driver loss is above the earlier bound, "
        "the summed per-car worst cases (gate_calibrate.ramp_loss), which is not a bound where an "
        "hour fails, by more than corner_search.DRIVER_TOLERANCE"
    ),
    "pass_mark.earlier_largest_driver_excess": (
        "for comparison only: largest corner maximum minus the earlier bound, 0 when never above"
    ),
    "pass_mark.agreed": (
        "at least one pair checked and every pair agrees (neither driver-loss bound enters it)"
    ),
    **{f"row.{name}": text for name, text in ROW_DEFINITIONS.items()},
}
# `method.resolves` in a run with a sizing method (`corner_run.definitions`, reading W5).
RESOLVES_WITH_SIZING = (
    "re-solves by outcome over every searched corner (re-solve and sizing methods only)"
)
# `method.deviation_counts`, written only for a method that wrote it (`corner_run.definitions`),
# followed there by the names of those methods.
DEVIATION_DEFINITION = (
    "a controller that counts its deviations (mef_controller.COUNTS), summed over every searched "
    "corner (compare_day.sum_deviations): hours, the hours planned (0 to n on each corner); "
    "deviating_hours, those in which least-laxity-first's split of the hour's target was found "
    "to leave the believed future infeasible and the total nearest the target among those found "
    "to keep it feasible was planned instead; deviating_service_hours, those of them in service "
    "hours 1 to n; no_feasible_hours, the hours in which no total was found to keep the believed "
    "future feasible, planned as least-laxity-first plans them; absent for every other method"
)


def _mean(values) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _max(values) -> float | None:
    return max(values) if values else None


def _sum_counts(days) -> dict | None:
    found = [d.solver_counts for d in days if d.solver_counts is not None]
    return sum_counts(found)


def method_entry(scale: float, reason: str, days) -> dict:
    """One method of one unit and variant (module text). `days`: its `CornerDay` of each searched
    day."""
    if scale <= 0:
        return {
            "scale": float(scale),
            "scale_reason": reason,
            "status": NO_OFFER,
            "days_searched": len(days),
        }
    result = [d for d in days if d.status in (BREAKS, HOLDS)]
    breaking = [d for d in result if d.status == BREAKS]
    misses = [d for d in breaking if d.audited_breaks is False]
    resolves: Counter = Counter()
    for d in days:
        resolves.update(d.resolves)
    deviated = sum_deviations([d.deviation_counts for d in days if d.deviation_counts is not None])
    entry = {
        "scale": float(scale),
        "scale_reason": reason,
        "status": "searched",
        "days_searched": len(days),
        "days_with_result": len(result),
        "days_breaking": len(breaking),
        "share_of_days_breaking": len(breaking) / len(result) if result else None,
        "days_breaking_by_slot_loss": sum(d.slot_breaking_paths > 0 for d in result),
        "days_with_driver_shortfall": sum(d.driver_short_paths > 0 for d in result),
        "days_short_only_below_tolerance": sum(
            d.status == HOLDS and d.short_below_tolerance_paths > 0 for d in result
        ),
        "days_timed_out": len(days) - len(result),
        "mean_breaking_paths": _mean([d.breaking_paths for d in result]),
        "worst_slot_loss": _max([d.worst_slot_loss for d in result]),
        "worst_driver_loss": _max([d.worst_driver_loss for d in result]),
        "breaking_days_audited_misses": len(misses),
        "share_of_breaking_days_audited_misses": len(misses) / len(breaking) if breaking else None,
        "breaking_days_audited_timed_out": sum(d.audited_breaks is None for d in breaking),
        "resolves": dict(sorted(resolves.items())),
        "solver_counts": _sum_counts(days),
    }
    if deviated is not None:  # a controller that counts deviations only (module text)
        entry["deviation_counts"] = deviated
    return entry


def pass_mark_entry(days, dates, checks) -> dict:
    """The closed-form check of one unit and variant (module text). `days`, `dates`: the checked
    days' indices and dates; `checks[k]`: the `ClosedFormCheck`s of day `k`."""
    pairs = [
        (d, date, c) for d, date, found in zip(days, dates, checks, strict=True) for c in found
    ]
    wrong = [(d, date, c) for d, date, c in pairs if not c.agreed]
    differ = [c for _, _, c in pairs if not c.agreed_strict]
    above = [c for _, _, c in pairs if not c.driver_within_bound]
    excess = [c.corner_driver - c.driver_bound for c in above]
    earlier = [
        c.corner_driver - c.earlier_driver_bound
        for _, _, c in pairs
        if c.corner_driver > c.earlier_driver_bound + DRIVER_TOLERANCE
    ]
    scales = sorted({c.beta for _, _, c in pairs})
    return {
        "days_checked": len(days),
        "days": list(days),
        "scales": scales,
        "pairs": len(pairs),
        "pairs_agreed_strict": len(pairs) - len(differ),
        "pairs_agreed": len(pairs) - len(wrong),
        "differences_by_reason": dict(sorted(Counter(c.reason for c in differ).items())),
        "largest_slot_difference": max((c.slot_difference for _, _, c in pairs), default=None),
        "disagreements": [
            {
                "day": d,
                "date": date,
                "scale": c.beta,
                "closed_form_first_failure": c.closed_hour,
                "corner_first_failure": c.corner_hour,
                "closed_form_slot_loss": c.closed_slot,
                "corner_worst_slot_loss": c.corner_slot,
                "reason": c.reason,
            }
            for d, date, c in wrong
        ],
        "disagreements_by_reason": dict(sorted(Counter(c.reason for _, _, c in wrong).items())),
        "driver_pairs_above_bound": len(excess),
        "largest_driver_excess": max(excess, default=0.0),
        "driver_above_bound_limited": sum(c.driver_limited for c in above),
        "driver_above_bound_no_slot_failure": sum(c.driver_no_slot_failure for c in above),
        "earlier_driver_pairs_above_bound": len(earlier),
        "earlier_largest_driver_excess": max(earlier, default=0.0),
        "agreed": bool(pairs) and not wrong,
    }


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return value


def _solver_cell(counts) -> str:
    if counts is None:
        return ""
    cells = [f"{k}:{v}" for k, v in sorted(counts["solves"].items())]
    cells += [f"hours:{counts['hours']}", f"fallback_hours:{counts['fallback_hours']}"]
    return ";".join(cells)


def day_rows(unit: str, variant: str, method: str, scale: float, days, prepared, found) -> list:
    """The CSV rows (`ROW_FIELDS`) of one method on the searched days. `days`: day indices;
    `prepared[d]`: the prepared day; `found`: the `CornerDay` of each searched day."""
    rows = []
    for d, day in zip(days, found, strict=True):
        p = prepared[d]
        worst = (
            "" if day.worst_path is None else "".join("+" if v > 0 else "-" for v in day.worst_path)
        )
        values = {
            "unit": unit,
            "variant": variant,
            "method": method,
            "day": d,
            "date": p.date,
            "site": p.site,
            "n_service": p.day.n_service,
            "scale": float(scale),
            "status": day.status,
            "n_paths": day.n_paths,
            "breaking_paths": day.breaking_paths,
            "slot_breaking_paths": day.slot_breaking_paths,
            "driver_short_paths": day.driver_short_paths,
            "short_below_tolerance_paths": day.short_below_tolerance_paths,
            "timed_out_paths": day.timed_out_paths,
            "worst_slot_loss": day.worst_slot_loss,
            "worst_driver_loss": day.worst_driver_loss,
            "worst_path": worst,
            "worst_first_failure": day.worst_first_failure,
            "audited_status": day.audited_status,
            "audited_breaking_rows": day.audited_breaking_rows if day.n_paths else None,
            "audited_worst_slot_loss": day.audited_worst_slot_loss,
            "audited_worst_driver_loss": day.audited_worst_driver_loss,
            "resolves": ";".join(f"{k}:{v}" for k, v in day.resolves.items()),
            "solver_counts": _solver_cell(day.solver_counts),
        }
        rows.append([_cell(values[k]) for k in ROW_FIELDS])
    return rows


# --- the text ------------------------------------------------------------------------------------

METHOD_WIDTH = 14
COLUMNS = (
    ("scale", 6),
    ("days", 5),
    ("break", 7),
    ("slot", 5),
    ("driver", 7),
    ("t/o", 4),
    ("<tol", 5),
    ("paths", 7),
    ("worst slot", 11),
    ("worst drv", 10),
    ("audited misses", 15),
)


def _number(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _share(value) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def _line(cells, widths) -> str:
    return "  " + " ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip()


def _method_cells(name: str, entry: dict) -> list:
    if entry["status"] == NO_OFFER:
        return [name, _number(entry["scale"]), entry["days_searched"], "no offer"] + [""] * 8
    misses = entry["breaking_days_audited_misses"]
    share = entry["share_of_breaking_days_audited_misses"]
    return [
        name,
        _number(entry["scale"]),
        entry["days_searched"],
        _share(entry["share_of_days_breaking"]),
        entry["days_breaking_by_slot_loss"],
        entry["days_with_driver_shortfall"],
        entry["days_timed_out"],
        entry["days_short_only_below_tolerance"],
        _number(entry["mean_breaking_paths"], 1),
        _number(entry["worst_slot_loss"], 3),
        _number(entry["worst_driver_loss"], 3),
        f"{misses} ({_share(share)})",
    ]


def _variant_lines(document: dict, variant: str, found: dict) -> list[str]:
    if "failed" in found:
        return [f"  {variant}: failed ({found['failed']})"]
    head = (
        f" {variant} ({found['configuration']}): {found['days_searched']} days searched of "
        f"{found['offering_test_days']} offering test days"
    )
    if found["days_searched"] < found["days_asked"]:
        head += f" (fewer than the {found['days_asked']} asked: the unit has no more)"
    widths = [METHOD_WIDTH, *(w for _, w in COLUMNS)]
    lines = [head, _line(["method", *(n for n, _ in COLUMNS)], widths)]
    for method in document["methods"]:
        lines.append(_line(_method_cells(method, found["methods"][method]), widths))
    for method in document["methods"]:
        counts = found["methods"][method].get("solver_counts")
        if counts is not None:
            solves = ", ".join(f"{k} {v}" for k, v in sorted(counts["solves"].items()))
            lines.append(
                f"  {method} solves on the corners: {solves}; {counts['hours']} hours planned, "
                f"{counts['fallback_hours']} by its fall-back"
            )
    for method in document["methods"]:
        counts = found["methods"][method].get("deviation_counts")
        if counts is not None:
            lines.append(
                f"  {method} on the corners: hours planned {counts['hours']}, with "
                "least-laxity-first's split found infeasible and the total nearest the target "
                f"found feasible planned {counts['deviating_hours']} (in service hours "
                f"{counts['deviating_service_hours']}), with no feasible total "
                f"{counts['no_feasible_hours']}"
            )
    return lines


def _verdict(found: dict) -> str:
    if not found["pairs"]:
        return "nothing checked"
    return "agreed" if found["agreed"] else "DISAGREED"


PASS_HEADS = (
    "variant",
    "days",
    "pairs",
    "strict",
    "agreed",
    "largest diff",
    "driver > bound",
    "verdict",
)
PASS_WIDTHS = [17, 5, 6, 7, 7, 13, 15, 8]


def _pass_mark_lines(entry: dict, held_out: bool = False) -> list[str]:
    key = check_key(held_out)
    lines = [
        f"{HEADINGS[held_out][1]}: fixed-plan closed form against the corner worst case",
        _line(
            PASS_HEADS,
            PASS_WIDTHS,
        ),
    ]
    for variant, found in entry["variants"].items():
        if key not in found:
            lines.append(f"  {variant}: not checked")
            continue
        p = found[key]
        if not is_checked(p):  # on the sealed route only (reading H7 of corner_run)
            lines.append(f"  {variant}: {NOT_CHECKED}: {p['reason']}")
            continue
        lines.append(
            _line(
                [
                    variant,
                    p["days_checked"],
                    p["pairs"],
                    p["pairs_agreed_strict"],
                    p["pairs_agreed"],
                    _number(p["largest_slot_difference"], 3),
                    p["driver_pairs_above_bound"],
                    _verdict(p),
                ],
                PASS_WIDTHS,
            )
        )
        for reason, count in p["differences_by_reason"].items():
            kind = "agree within tolerance" if reason == WITHIN_TOLERANCE else "disagree"
            lines.append(f"    {count} pair(s) differ, {kind}: {reason}")
        if p["driver_pairs_above_bound"]:
            lines.append(
                f"    driver > bound: {p['driver_above_bound_limited']} with a car short of its "
                f"plan, {p['driver_above_bound_no_slot_failure']} with no failing hour"
            )
    return lines


def _total_line(total: str, found: dict, held_out: bool, abstained: dict | None = None) -> str:
    """The line over all units; on the sealed route with the units and variants not checked
    (reading H7 of corner_run) and the held-out units the held-out truth build abstained
    (`abstained`: unit -> reason; reading S4 of `sealed_route`: none checked, each counted as not
    passing, named in the summary's first lines)."""
    unchecked = len(found["not_checked"]) if held_out else 0
    parts = [f"{unchecked} unit and variant(s) not checked, listed above"] if unchecked else []
    if abstained:
        parts.append(
            f"{len(abstained)} abstained held-out unit(s) not checked, counted as "
            f"{sealed_route.COUNTS_AS}, named above"
        )
    left = "; ".join(parts)
    if held_out and not found["pairs"]:
        return f"{total}: {NOTHING_CHECKED}" + (f"; {left}" if left else "") + "."
    return (
        f"{total}: {found['pairs_agreed_strict']} of {found['pairs']} "
        f"pairs agree strictly, {found['pairs_agreed']} with the tolerance: "
        f"{_verdict(found)}" + (f"; {left}" if left else "") + "."
    )


def summary_lines(document: dict, held_out: bool = False) -> list[str]:
    """The summary's lines (module text); `held_out`: a run on the sealed route."""
    run = document["run"]
    intro, _, total = HEADINGS[held_out]
    scales = run["check_scales"]  # on the sealed route a text (reading H5 of corner_run)
    if not isinstance(scales, str):
        scales = "the scales " + ", ".join(f"{g:g}" for g in scales)
    lines = [
        *rotation_whatif.summary_head(document),
        f"Corner search for model {document.get('model')}, dial {document['dial']:g}: split seed "
        f"{run['seed']}, up to {run['days']} test days per unit and variant that offer, in a "
        f"seeded order. {document['label']}.",
        "Variants: " + "; ".join(f"{v} = {c}" for v, c in document["configurations"].items()) + ".",
        "Each method runs on every corner of its own calibrated box (its tier A scale on the "
        "split's calibration days). break: share of searched days with a corner that fails an "
        "hour or leaves more than 0.0001 kWh of protected energy undelivered, over the days with "
        "a result; slot, driver: such days by kind; t/o: days without a result (a re-solve timed "
        "out); <tol: days that hold but have a corner short by more than 0 and at most 0.0001 kWh "
        "(not counted as a break, whatever the cause); paths: mean breaking corners; audited "
        "misses: breaking days on which the "
        "audited 8 paths hold.",
        "The corners give the worst case over the box for the fixed plan only; for every other "
        "method a path inside the box can be worse than every corner, so its break share is a "
        "lower bound.",
        f"{intro}: the first {run['check_days']} searched days of each unit and variant at "
        + scales
        + "; strict: same first "
        f"failing hour and slot losses within {run['slot_tolerance']:g}; agreed: strict, or a "
        "difference labelled 'within tolerance' (at the closed form's failing hour every "
        "corner's draws miss the plan by at most 0.0001 kWh in total); driver > bound counts "
        "pairs above the driver-loss bound, the every-scale bound, which counts each car's "
        "decision-hour draw and each service hour at no more than the car can draw in it and "
        "bounds replay's driver loss at every scale up to the errors of V3 and V7 of "
        "fixed_plan_loss's module text, reported, not part of the verdict.",
    ]
    for unit, entry in document["units"].items():
        lines.append("")
        if entry["status"] != "calibratable":
            lines.append(f"{unit} ({entry['status']}, {entry['counting_days']} counting days):")
            lines.append("  no numbers: the unit cannot be split")
            continue
        lines.append(
            f"{unit} ({entry['counting_days']} counting days; split seed {run['seed']}: "
            f"{entry['calibration_days']} calibration days, {entry['test_days']} test days):"
        )
        for variant, found in entry["variants"].items():
            lines += _variant_lines(document, variant, found)
        lines += _pass_mark_lines(entry, held_out)
    pass_mark = document[check_key(held_out)]
    lines.append("")
    route = document.get(ROUTE_KEY) if held_out else None
    held = route.get("held_out_inputs") if isinstance(route, dict) else None
    lines.append(_total_line(total, pass_mark, held_out, sealed_route.abstained_of(held)))
    timeouts = document["timeouts"]
    lines.append(
        "Timeouts (recorded outcomes): "
        + (", ".join(f"{k} {v}" for k, v in timeouts.items()) if timeouts else "none")
    )
    if document["failed"]:
        lines.append("FAILED: " + "; ".join(document["failed"]))
    return lines
