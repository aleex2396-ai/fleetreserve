"""Outcome O1's slots X, Y, Z and D, from the scoring runs of the drive (plan section 11, O1).

"O1 Headline: audited-set re-solve offer X kWh/day (Y per car; Z% of the ceiling) on 9 held-out
units ... every-request offer D." One row per scoring run (`SPECS`), one row per unit beside it.

Readings (each with its alternative in brackets):
  Q1. "Audited-set re-solve" is the one-way rung under tier A, at each unit's headline
      configuration (`sealed_choice.headline` of the run's matrix). [Alternative: the two-way
      rung.] The fixed-plan rows are the rung `fixed`.
  Q2. X is the sum over the units of each unit's offer per counting test day (plan section 11,
      the owner's decision of 17 November 2026), the total offered kWh over the total counting
      test days beside it. For a run with a `day-intervals.json` and one headline configuration
      for all its units, X, the total over the total, Y and Z are the pooled block of that
      configuration (`day_intervals` readings I13 to I17), points and 95% intervals as the file
      has them. Otherwise they are the sum of the units' own figures, as a point with no
      interval, and the cell's row says why: a run with no interval entry (plan section 11 gives
      intervals for `lgbm_tail` and `hazard_tuned_inputs` only), or headline configurations that
      differ by site type (no entry pools units of two configurations). The point of Y and Z is
      then the ratio of sums of `day_intervals` reading I14 (a), from the units' own offer, offer
      per car and share (a unit whose offer is 0 leaves it "no figure").
      Where headline configurations differ and the interval is computed from the per-day file
      (`wp16_mixed`), that is after the run's output was seen: its row's first cell and note carry
      the words "made apart at a later commit, after the run's output was seen; fills no slot of
      plan section 11" (`wp16_mixed.MARK`).
  Q3. A unit with no figure (the truth build gave it no window; its configuration failed or could
      not be split) adds 0 to X, counts as not passing, and is named in the row's last column
      with its reason (plan section 11; section 13, WP14).
  Q4. Z is the share of the hindsight ceiling (`hindsight-free-decision-hour`, "the ceiling" of
      the gate), the share of the best-causal ceiling beside it. [Alternative: the best-causal
      ceiling as Z.]
  Q6. The rows of rho 0.1 and 0.2 (the learn runs `learn-lgbm_tail-rho0_1`, `-rho0_2` and their
      interval entries) are in `o1-summary` and `o1-units` and, with the rho 0 row, in `o1-rho`.
  Q7. The rows of rho 0.1 and 0.2 are WP15 parts (plan section 13, WP15: rho 0.1 and 0.2 for the
      main model only). After a held-out run that WP14 fails they are printed as information
      only: `wp16_failed.information_only`, the one decision every WP15 part of the tables uses,
      is true, and then the first cell of each such row, in all three tables, ends with
      " [information only]" and a note says what the mark means. Nothing else of a row changes
      (no number cell), and when the run passed WP14 no byte of the three tables changes. The
      rho 0 row beside them is the headline, not a WP15 part, and is never marked.
  Q5. D is the every-request offer of the fixed plan, tier A, summed over the same units from
      `learn-gate.json` (`fixed_every_request.test_mean_offer_kwh_per_day`), for the rows that
      name D in `SPECS`.
"""

import math
from dataclasses import dataclass

from fleetreserve import wp16_mixed
from fleetreserve.wp16_cells import Interval, Missing, Table, interval_of
from fleetreserve.wp16_drive import Missed, Source
from fleetreserve.wp16_failed import SUFFIX, information_note, information_only
from fleetreserve.wp16_learn import (
    BEST_CAUSAL,
    HINDSIGHT,
    ModelRun,
    Runs,
    counting_days,
    names_for,
    pooled_block,
    scale_of,
    unit_intervals,
    unit_test,
)
from fleetreserve.wp16_mixed import MARK
from fleetreserve.wp16_units import interval_settings


@dataclass(frozen=True)
class Spec:
    title: str
    learn: str
    intervals: str | None
    rung: str
    reading: str  # "main" or "beside"
    every_request: bool = False
    wp15: bool = False  # a WP15 part: printed as information only after a WP14 failure (Q7)


SPECS = (
    Spec(
        "lgbm_tail (main model), one-way re-solve",
        "learn-lgbm_tail",
        "intervals-lgbm_tail",
        "one-way",
        "main",
        True,
    ),
    Spec(
        "hazard_tuned_inputs, one-way re-solve",
        "learn-hazard_tuned_inputs",
        "intervals-hazard_tuned_inputs",
        "one-way",
        "main",
        True,
    ),
    Spec("earlier_gbm, one-way re-solve", "learn-earlier_gbm", None, "one-way", "main"),
    Spec(
        "earlier_population, one-way re-solve",
        "learn-earlier_population",
        None,
        "one-way",
        "main",
    ),
    Spec("lgbm_tail, fixed plan", "learn-lgbm_tail", "intervals-lgbm_tail", "fixed", "main"),
    Spec("Kaplan-Meier, fixed plan", "learn-fixed-plan-km", None, "fixed", "main"),
    Spec("pooled LightGBM, fixed plan", "learn-fixed-plan-lgbm", None, "fixed", "main"),
    Spec("Cox, fixed plan", "learn-fixed-plan-cox", None, "fixed", "main"),
    Spec(
        "lgbm_tail, departures seen at the next hour (beside reading)",
        "learn-lgbm_tail",
        "intervals-lgbm_tail",
        "one-way",
        "beside",
    ),
    Spec(
        "lgbm_tail, forecast energy at the headline's own kappa, reading and dial",
        "learn-lgbm_tail-forecast-energy",
        "intervals-lgbm_tail-forecast-energy",
        "one-way",
        "main",
    ),
    Spec(
        "lgbm_tail, held setting",
        "learn-lgbm_tail-held",
        "intervals-lgbm_tail-held",
        "one-way",
        "main",
    ),
    Spec(
        "lgbm_tail, rho 0.1 (WP15 sensitivity)",
        "learn-lgbm_tail-rho0_1",
        "intervals-lgbm_tail-rho0_1",
        "one-way",
        "main",
        wp15=True,
    ),
    Spec(
        "lgbm_tail, rho 0.2 (WP15 sensitivity)",
        "learn-lgbm_tail-rho0_2",
        "intervals-lgbm_tail-rho0_2",
        "one-way",
        "main",
        wp15=True,
    ),
)

RHO_SPECS = (SPECS[0], SPECS[-2], SPECS[-1])  # rho 0 (main), 0.1, 0.2
NO_FILE = "day-intervals.json holds no interval for this cell (no such entry, or none written)"
SUMMARY_COLUMNS = (
    "run",
    "configuration",
    "rung",
    "units with a figure",
    "X: sum of the units' offers (kWh per counting test day)",
    "total offered kWh over total counting test days",
    "Y: offer per car per day (kWh)",
    "Z: share of the hindsight ceiling",
    "share of the best-causal ceiling",
    "D: every-request offer (kWh per counting test day)",
    "units left out",
)
UNIT_COLUMNS = (
    "run",
    "unit",
    "configuration",
    "offer (kWh per counting test day)",
    "offer per car per day (kWh)",
    "share of the hindsight ceiling",
    "share of the best-causal ceiling",
    "counting test days",
    "offering test days",
    "failing test days",
    "scale",
    "why no figure",
)


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def ratio_of_sums(tests: dict, getter, what: str):
    """`day_intervals` reading I14 (a): the sum of the units' offers over the sum of their cars (or
    ceilings) per counting test day, each recovered as offer over the unit's own ratio; a
    `Missing` when a unit's offer or ratio is not a positive number."""
    top = bottom = 0.0
    for test in tests.values():
        offer, ratio = test.get("mean_offer_kwh_per_day"), getter(test)
        if not (_number(offer) and _number(ratio) and offer > 0 and ratio > 0):
            return Missing(f"no figure: a unit's offer or {what} is 0 or missing")
        top += offer
        bottom += offer / ratio
    return top / bottom if tests and bottom > 0 else Missing(f"no figure: no unit has a {what}")


def _share(name: str):
    def get(test: dict):
        found = test.get("share_of_ceiling")
        found = found.get(name) if isinstance(found, dict) else None
        return found.get("ratio_of_sums") if isinstance(found, dict) else None

    return get


def _every_request(source: Source, spec: Spec, names: dict, units) -> object:
    try:
        gate = source.json(spec.learn, "learn-gate.json")
    except Missed as problem:
        return Missing(problem.text)
    total = 0.0
    for unit in units:
        try:
            found = gate["configurations"][names[unit]]["units"][unit]
            value = found["fixed_every_request"]["test_mean_offer_kwh_per_day"]
        except (KeyError, TypeError):
            return Missing(f"no figure: learn-gate.json has none for {unit}")
        if not _number(value):
            return Missing(f"no figure: the every-request offer of {unit} is {value!r}")
        total += value
    return total


def _left_text(left: dict) -> str:
    return "; ".join(f"{unit}: {why}" for unit, why in left.items())


def summary_row(source: Source, runs: Runs, spec: Spec, units: list, absent: dict):
    """(the row of `SUMMARY_COLUMNS` for a run, the note that says where its intervals are from
    or why it has none)."""
    try:
        run = runs.get(spec.learn, spec.intervals)
    except Missed as problem:
        return (spec.title, "", spec.rung, *([Missing(problem.text)] * 7), ""), problem.text
    names = names_for(run, units, spec.reading)
    tests, left = {}, {}
    for unit in units:
        test, why = unit_test(run, unit, names[unit], spec.rung, "A", absent)
        if test is None:
            left[unit] = why
        elif not _number(test.get("mean_offer_kwh_per_day")):
            left[unit] = "no figure: the offer is not a number; counts as not passing"
        else:
            tests[unit] = test
    used = sorted({names[u] for u in tests})
    point = sum(t["mean_offer_kwh_per_day"] for t in tests.values())
    days = {u: counting_days(run, u) for u in tests}
    weighted = sum(t["mean_offer_kwh_per_day"] * days[u] for u, t in tests.items() if days[u])
    total_days = sum(d for d in days.values() if d)
    block = pooled_block(run, used[0], spec.rung, "A") if len(used) == 1 and run.intervals else None
    mixed_why, how = "", f"pooled block of day-intervals.json, units pooled: {len(days)}"
    if len(used) > 1 and spec.intervals is not None:
        # headline configurations that differ by site type: the pooled block of each unit's own
        # configuration, by day_intervals' own functions on the per-day file (wp16_mixed)
        try:
            block, pooled_units = wp16_mixed.pooled_block(
                source, spec.learn, {u: names[u] for u in tests}, units, left, spec.rung
            )
            lacking = sorted(set(tests) - set(pooled_units))
            if lacking:  # the guard of the single-configuration path: no interval of a short sum
                raise Missed(
                    f"units with a figure are missing from the pooled block: {', '.join(lacking)}"
                )
            seed, resamples = interval_settings(source)
            how = (
                f"each unit under its own headline configuration ({', '.join(used)}), pooled by "
                "day_intervals' own functions on learn-day-losses.json (wp16_mixed), seed "
                f"{seed}, {resamples} resamples, units pooled: {len(pooled_units)}"
            )
        except Missed as problem:
            block, mixed_why = None, problem.text
    mixed = len(used) > 1 and block is not None
    if block is not None and (mixed or set(days) <= set(_block_units(run, used[0]))):
        sums = block.get("sum_of_unit_offers", {})
        over = block.get("total_over_total_days", {})
        shares = sums.get("share_of_ceiling", {}) if isinstance(sums, dict) else {}
        cells = (
            interval_of(sums.get("offer_kwh_per_counting_test_day")),
            interval_of(over.get("offer_kwh_per_counting_test_day")),
            interval_of(sums.get("offer_kwh_per_car_per_day")),
            interval_of(shares.get(HINDSIGHT)),
            interval_of(shares.get(BEST_CAUSAL)),
        )
        note = f"{spec.title}: {how}"
        if mixed:  # computed after the run's output was seen: made apart (wp16_mixed)
            note += f"; {MARK}"
    else:
        if len(used) > 1:
            why = (
                f"the headline configuration differs by site type ({', '.join(used)}): no entry "
                "pools units of two configurations, so no interval of the sum"
                + (f"; computing it from the per-day file failed: {mixed_why}" if mixed_why else "")
                + (
                    ""
                    if spec.intervals is not None
                    else " (plan section 11 gives none for this run)"
                )
            )
        else:
            why = run.intervals_why or "the pooled block does not cover these units"
        cells = (
            Interval(point, why=why) if tests else Missing("no figure: no unit has one"),
            Interval(weighted / total_days, why=why)
            if total_days
            else Missing("no figure: no test days"),
            _point(
                ratio_of_sums(tests, lambda t: t.get("offer_kwh_per_car_per_day"), "per-car"), why
            ),
            _point(ratio_of_sums(tests, _share(HINDSIGHT), "share of the ceiling"), why),
            _point(
                ratio_of_sums(tests, _share(BEST_CAUSAL), "share of the best-causal ceiling"), why
            ),
        )
        note = f"{spec.title}: points only, no interval: {why}"
    d = _every_request(source, spec, names, list(tests)) if spec.every_request and tests else ""
    row = (
        f"{spec.title} [{MARK}]" if mixed else spec.title,
        ", ".join(used) or "none",
        spec.rung,
        f"{len(tests)} of {len(units)}",
        *cells,
        d,
        _left_text(left),
    )
    return row, note


def _block_units(run: ModelRun, name: str) -> list:
    found = run.intervals.get("pooled", {}).get("configurations", {}).get(name, {})
    units = found.get("units") if isinstance(found, dict) else None
    return units if isinstance(units, list) else []


def _point(value, why: str):
    return value if isinstance(value, Missing) else Interval(value, why=why)


def unit_rows(runs: Runs, spec: Spec, units: list, absent: dict) -> list:
    """The rows of `UNIT_COLUMNS` of one run: each unit's own figures, with the interval of its
    offer when `day-intervals.json` holds one."""
    try:
        run = runs.get(spec.learn, spec.intervals)
    except Missed as problem:
        return [(spec.title, unit, "", Missing(problem.text), *([""] * 8)) for unit in units]
    names = names_for(run, units, spec.reading)
    rows = []
    for unit in units:
        name = names[unit]
        test, why = unit_test(run, unit, name, spec.rung, "A", absent)
        if test is None:
            rows.append((spec.title, unit, name or "", Missing(why), *([""] * 7), why))
            continue
        found = unit_intervals(run, name, unit, spec.rung, "A") or {}
        shares = found.get("share_of_ceiling", {})
        offer = (
            interval_of(found["offer_kwh_per_counting_test_day"])
            if "offer_kwh_per_counting_test_day" in found
            else Interval(test.get("mean_offer_kwh_per_day"), why=NO_FILE)
        )
        per_car = (
            interval_of(found["offer_kwh_per_car_per_day"])
            if "offer_kwh_per_car_per_day" in found
            else Interval(test.get("offer_kwh_per_car_per_day"), why=NO_FILE)
        )
        rows.append(
            (
                spec.title,
                unit,
                name,
                offer,
                per_car,
                interval_of(shares[HINDSIGHT])
                if HINDSIGHT in shares
                else Interval(_share(HINDSIGHT)(test), why=NO_FILE),
                interval_of(shares[BEST_CAUSAL])
                if BEST_CAUSAL in shares
                else Interval(_share(BEST_CAUSAL)(test), why=NO_FILE),
                counting_days(run, unit),
                test.get("offering_days"),
                test.get("failing_days"),
                scale_of(run, unit, name, spec.rung),
                "",
            )
        )
    return rows


def _marked(row: tuple, spec: Spec, information: bool) -> tuple:
    """`row` with " [information only]" at the end of its first cell when it is a WP15 row and the
    WP15 parts are marked (reading Q7); every other cell is the row's own."""
    if information and spec.wp15 and isinstance(row[0], str):
        return (row[0] + SUFFIX, *row[1:])
    return row


def build(source: Source, runs: Runs, units: list, absent: dict) -> list[Table]:
    summary = Table(
        "o1-summary",
        "O1: the headline slots X, Y, Z and D, by scoring run (tier A, calibrated, held-out units)",
        SUMMARY_COLUMNS,
    )
    per_unit = Table(
        "o1-units",
        "O1: each unit's offer, by scoring run (tier A)",
        UNIT_COLUMNS,
    )
    information = information_only(source)
    for spec in SPECS:
        row, note = summary_row(source, runs, spec, units, absent)
        summary.add(*_marked(row, spec, information))
        summary.notes.append(note)
        for line in unit_rows(runs, spec, units, absent):
            per_unit.add(*_marked(line, spec, information))
    summary.notes.append(
        "X, Y and Z follow readings Q1 to Q5 of the module text of fleetreserve.wp16_o1; a unit "
        "with no figure adds 0 to X and counts as not passing."
    )
    rho = Table(
        "o1-rho",
        "O1: the main model at rho 0, 0.1 and 0.2 (WP15 sensitivity, reported beside the headline)",
        SUMMARY_COLUMNS,
    )
    for spec in RHO_SPECS:
        row, note = summary_row(source, runs, spec, units, absent)
        rho.add(*_marked(row, spec, information))
        rho.notes.append(note)
    rho.notes.append(
        "Each row is the main model's run at that rho (its learn and interval entries), at its own "
        "kappa and dials; never pooled with another rho (reading Q6)."
    )
    if information:
        part = "the rows marked [information only], which are WP15 parts, are"
        for table in (summary, per_unit, rho):
            table.notes.append(information_note(part))
    return [summary, per_unit, rho]
