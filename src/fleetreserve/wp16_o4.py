"""Outcome O4: availability and shortfall per held-out unit, with the pass marks (plan section 11).

"O4 Validation, random split, per unit and regime: availability, shortfall. Pass: upper end >= 90%,
lower end <= 5%. The shortfall pass mark applies to Tier A (Section 6); Tier B reports the offer
against the driver loss as a curve over epsilon." Intervals are given for `lgbm_tail` and
`hazard_tuned_inputs` only (plan section 11), so only their two runs are tabled.

Readings (alternatives in brackets):
  V1. Availability is 1 minus the mean slot loss over the offering test days, shortfall the mean
      driver loss over them (`day_intervals`, `definitions`), of the one-way re-solve at the unit's
      headline configuration, tier A; the intervals are those of `day-intervals.json`, the ends
      as the file has them. [Alternative: the fixed-plan rung.]
  V2. A unit passes when its availability interval's upper end is at least 0.90 and its shortfall
      interval's lower end is at most 0.05, both read as the plan words them. A unit whose
      figures are missing because the truth build gave it no window, or its configuration failed
      or could not be split, counts as not passing (plan section 11). A unit whose calibrated
      scale is 0 offers nothing, so it has no loss and no interval: it is "not judged" and is
      counted apart, never as passed or as not passing (the plan does not say; a question for the
      owner). [Alternative: counted as not passing.]
  V4. The pass mark is printed both ways (`o4-pass-both-ways`) until the owner answers: A, a
      unit at scale 0 is not judged and is outside "units passing of units judged"; B, it is
      judged and passes at zero loss. A unit's own row says "not judged: scale 0".
  V3. Tier B is tabled beside tier A with its availability and shortfall intervals, and its offer
      and driver loss at the two driver-risk levels the matrices hold (`o4-tier-b-two-points`: two
      points, not a curve); the curve over epsilon is written by no entry and is named as such.
  V5. `o4-points` (`build_points`) gives, per unit and model, the run's own point values of the
      same learn result and headline configuration (tier A, one-way rung): the offer, the mean
      slot loss and the mean driver loss over the offering test days, each loss beside its limit
      (`crc.ALPHA_SLOT`, `crc.ALPHA_DRIVER`: availability 90% and shortfall 5%), the calibrated
      scale and the offering test days. Points only: no interval and no verdict (the verdict
      needs intervals and stays in `o4-availability-shortfall`). A unit at scale 0 has no loss and
      is "not judged: scale 0"; a figure the run does not hold is "not produced: <why>", never 0.
"""

from fleetreserve.crc import ALPHA_DRIVER, ALPHA_SLOT
from fleetreserve.wp16_cells import Interval, Missing, Table, interval_of
from fleetreserve.wp16_drive import Missed
from fleetreserve.wp16_learn import Runs, names_for, scale_of, unit_intervals, unit_test

AVAILABILITY_MARK = 0.90
SHORTFALL_MARK = 0.05
MODELS = (
    ("lgbm_tail", "learn-lgbm_tail", "intervals-lgbm_tail"),
    ("hazard_tuned_inputs", "learn-hazard_tuned_inputs", "intervals-hazard_tuned_inputs"),
)
COLUMNS = (
    "model",
    "unit",
    "configuration",
    "availability (tier A)",
    "shortfall (tier A)",
    "availability upper end >= 90%",
    "shortfall lower end <= 5%",
    "passes",
    "availability (tier B)",
    "shortfall (tier B)",
    "why no figure",
)


BOTH_COLUMNS = (
    "model",
    "units",
    "A: units passing",
    "A: units judged",
    "units at scale 0 (not judged in A)",
    "units with no interval (not judged)",
    "units not passing, judged",
    "B: units passing (scale 0 counted as passing)",
    "B: units judged",
)


def _yes(value) -> str:
    return "yes" if value else "no"


def judge(availability, shortfall) -> tuple[str, str, str]:
    """(availability mark, shortfall mark, verdict) of one unit's two cells (reading V2): the
    verdict is "yes", "no" or "not judged"."""
    if isinstance(availability, Missing) or isinstance(shortfall, Missing):
        return "", "", "not judged"
    first = availability.high is not None and availability.high >= AVAILABILITY_MARK
    second = shortfall.low is not None and shortfall.low <= SHORTFALL_MARK
    if availability.high is None or shortfall.low is None:
        return _yes(first), _yes(second), "not judged"
    return _yes(first), _yes(second), "yes" if first and second else "no"


POINT_COLUMNS = (
    "model",
    "unit",
    "configuration",
    "dial (driver-risk level)",
    "tier B offer (kWh per counting test day)",
    "tier B mean driver loss (offering test days)",
)


def two_points(table: Table, title: str, run, units: list) -> None:
    """Tier B's offer and driver loss at each driver-risk level the matrix holds for the main
    family (none, and the chosen dial): two points, never a curve over epsilon (reading V3)."""
    for unit in units:
        entry = run.learn.get("units", {}).get(unit)
        configurations = entry.get("configurations") if isinstance(entry, dict) else None
        for name, offer in (configurations if isinstance(configurations, dict) else {}).items():
            if not isinstance(offer, dict) or offer.get("family") != "main":
                continue
            rungs = offer.get("rungs")
            one = rungs.get("one-way") if isinstance(rungs, dict) else None
            test = one.get("tier_b", {}).get("test") if isinstance(one, dict) else None
            if not isinstance(test, dict):
                table.add(title, unit, name, "", Missing("no figure: no tier B result"), "")
                continue
            dial = offer.get("driver_risk")
            table.add(
                title,
                unit,
                name,
                "none" if dial is None else dial,
                test.get("mean_offer_kwh_per_day"),
                test.get("mean_driver_loss_offering_days"),
            )


POINTS_COLUMNS = (
    "model",
    "unit",
    "configuration",
    "offer (kWh per counting test day)",
    "mean slot loss (offering test days)",
    "slot loss limit",
    "mean driver loss (offering test days)",
    "driver loss limit",
    "scale",
    "offering test days",
    "why no figure",
)
NOT_PRODUCED = "not produced: "
NOT_JUDGED_AT_ZERO = "not judged: scale 0 (no offer, so no loss)"


def _point(test: dict, key: str) -> "float | Missing":
    """A number of the run's `test` block as it holds it, else the cell that says it holds none
    (a number is never made up: bool and None are not numbers)."""
    found = test.get(key)
    if isinstance(found, (int, float)) and not isinstance(found, bool):
        return found
    return Missing(f"{NOT_PRODUCED}the run's test block holds no {key}")


def build_points(runs: Runs, units: list, absent: dict) -> list[Table]:
    """`o4-points`: the run's point values per model and unit (reading V5). Reads the learn
    result by the route `build` uses (`runs.get`, `names_for`, `unit_test`, `scale_of`); no file
    of its own."""
    table = Table(
        "o4-points",
        "O4: the run's point values per unit (tier A, one-way rung; points only, no interval, "
        "no verdict)",
        POINTS_COLUMNS,
    )
    for title, learn, intervals in MODELS:
        try:
            run = runs.get(learn, intervals)
        except Missed as problem:
            for unit in units:
                table.add(
                    title,
                    unit,
                    "",
                    Missing(NOT_PRODUCED + problem.text),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    problem.text,
                )
            table.notes.append(f"{title}: no point values: {problem.text}")
            continue
        names = names_for(run, units, "main")
        for unit in units:
            name = names[unit]
            test, why = unit_test(run, unit, name, "one-way", "A", absent)
            if test is None:
                table.add(
                    title,
                    unit,
                    name or "",
                    Missing(NOT_PRODUCED + why),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    why,
                )
                continue
            scale = scale_of(run, unit, name, "one-way")
            days = test.get("offering_days")
            days = days if isinstance(days, int) and not isinstance(days, bool) else None
            shown_scale = (
                scale
                if isinstance(scale, (int, float)) and not isinstance(scale, bool)
                else Missing(NOT_PRODUCED + "the run holds no calibrated scale")
            )
            shown_days = days if days is not None else Missing(NOT_PRODUCED + "no offering_days")
            if scale == 0:  # a unit at scale 0 offers nothing: no loss, as `build` says
                loss = Missing(NOT_JUDGED_AT_ZERO)
                slot, driver, why = loss, loss, "not judged: scale 0"
            else:
                slot = _point(test, "mean_slot_loss_offering_days")
                driver = _point(test, "mean_driver_loss_offering_days")
                why = ""
            table.add(
                title,
                unit,
                name,
                _point(test, "mean_offer_kwh_per_day"),
                slot,
                ALPHA_SLOT,
                driver,
                ALPHA_DRIVER,
                shown_scale,
                shown_days,
                why,
            )
    table.notes.append(
        "The run's own point values from each model's learn step, at each unit's headline "
        "configuration (tier A, one-way rung): no interval and no pass verdict; the verdict "
        "needs intervals and is in o4-availability-shortfall. A limit is the level the loss is "
        "held to (slot loss 0.10: availability at least 90%; driver loss 0.05: shortfall at "
        "most 5%). A unit at scale 0 has no loss: not judged (reading V5)."
    )
    return [table]


def build(runs: Runs, units: list, absent: dict) -> tuple[list[Table], list[dict]]:
    points_table = Table(
        "o4-tier-b-two-points",
        "O4: tier B, offer against driver loss at the two driver-risk levels held (two points)",
        POINT_COLUMNS,
    )
    both = Table("o4-pass-both-ways", "O4: the pass mark read both ways", BOTH_COLUMNS)
    table = Table("o4-availability-shortfall", "O4: availability and shortfall per unit", COLUMNS)
    figures = []
    for title, learn, intervals in MODELS:
        try:
            run = runs.get(learn, intervals)
        except Missed as problem:
            for unit in units:
                table.add(title, unit, "", Missing(problem.text), "", "", "", "no", "", "", "")
            table.notes.append(f"{title}: counts every unit as not passing: {problem.text}")
            n = len(units)
            both.add(title, n, 0, n, 0, 0, n, 0, n)
            continue
        two_points(points_table, title, run, units)
        names = names_for(run, units, "main")
        counts = {"yes": 0, "no": 0, "zero": 0, "no interval": 0}
        points = []
        for unit in units:
            name = names[unit]
            test, why = unit_test(run, unit, name, "one-way", "A", absent)
            if test is None:
                table.add(title, unit, name or "", Missing(why), "", "", "", "no", "", "", why)
                counts["no"] += 1
                points.append((unit, None, None))
                continue
            found = unit_intervals(run, name, unit, "one-way", "A")
            at_zero = scale_of(run, unit, name, "one-way") == 0
            other = unit_intervals(run, name, unit, "one-way", "B") or {}
            if found is None:
                text = run.intervals_why or "no interval was written for this unit"
                missing = Missing(f"no figure: {text}")
                kind = "scale 0" if at_zero else "no interval"
                table.add(
                    title, unit, name, missing, missing, "", "", f"not judged: {kind}", "", "", text
                )
                counts["zero" if at_zero else "no interval"] += 1
                points.append((unit, None, None))
                continue
            availability = interval_of(found.get("availability"))
            shortfall = interval_of(found.get("shortfall"))
            first, second, verdict = judge(availability, shortfall)
            if verdict == "not judged" or at_zero:  # a scale-0 unit offers nothing, ends or not
                kind = "scale 0" if at_zero else "no interval"
                counts["zero" if at_zero else "no interval"] += 1
                verdict = f"not judged: {kind}"
            else:
                counts[verdict] += 1
            table.add(
                title,
                unit,
                name,
                availability,
                shortfall,
                first,
                second,
                verdict,
                interval_of(other.get("availability")) if other else "",
                interval_of(other.get("shortfall")) if other else "",
                verdict.removeprefix("not judged: ") if verdict.startswith("not judged") else "",
            )
            points.append((unit, availability, shortfall))
        judged = counts["yes"] + counts["no"]
        both.add(
            title,
            len(units),
            counts["yes"],
            judged,
            counts["zero"],
            counts["no interval"],
            counts["no"],
            counts["yes"] + counts["zero"],
            judged + counts["zero"],
        )
        figures.append({"kind": "o4", "model": title, "points": points})
    points_table.notes.append(
        "Two points, not a curve over epsilon: the matrices hold the level 'none' and the chosen "
        "dial only; no entry writes the curve (reading V3)."
    )
    table.notes.append(
        "Tier B's offer against the driver loss as a curve over epsilon: no entry writes it; the "
        "two points the matrices hold are in o4-tier-b-two-points (reading V3)."
    )
    both.notes.append(
        "Reading A: a unit calibrated to scale 0 offers nothing, has no loss and no interval, and "
        "is not judged: it is outside the count, named, with the units judged beside the passes. "
        "Reading B: it is judged and passes at zero loss (a day that does not offer has no loss: "
        "plan section 6). A unit with no figure (no window, failed configuration, not split) "
        "counts as not passing in both; a unit with no interval for another reason is not "
        "judged in both. Both are printed until the owner answers (readings V2, V4)."
    )
    return [table, points_table, both], figures


def cell_values(cell) -> tuple[float | None, float | None, float | None]:
    """(point, low, high) of a cell, None where it holds none (the figures' reader)."""
    if isinstance(cell, Interval):
        return cell.point, cell.low, cell.high
    return None, None, None
