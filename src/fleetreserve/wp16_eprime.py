"""The share of test days in the event E' of each scoring run (plan section 5, T2; section 11).

E' is the event on which a certified day's steered cars all stay inside their certified sets. The
plan words it as: "(c1) on E'_i={d̂_i<=d_i, r̂_i<=r_i, ê_cap,i<=e_i, ê_need,i>=(1-ρ)e_i} car i gets
>= (1-ρ)e_i on every path, whatever other cars do; (c2) on E'=∩E'_i every request is met", and, for
the check of T2, "the share of days in E' shown with every "certified" claim: first its share of
the days with a certified offer above zero, and beside it its share of all counting test days".
In words: a car is in E'_i when its forecast departure is not later than its true one, its forecast
rate and stored-energy cap are not above the true ones and its forecast need is not below the
true need less the tolerated share rho; a day is in E' when it was certified and every steered car
is in E'_i. Where E' holds, the certificate is a proof of the day's offer; where it does not, the
offer rests on the calibration alone.

The step `learn-<model>` (`fleetreserve.learn_run`) writes the figures into its `learn.json`, per
unit and configuration, and the definition of the denominators beside them:
`units.<unit>.e_prime.<configuration>` (`learn_run.evaluate`, the dictionary `e_prime`, lines
`"e_prime": e_prime[name]` and `"e_prime_definition": E_PRIME_DEFINITION`), whose keys are
`learn_run.COUNTS` (`test_days`, `certified_test_days`, `days_in_e_prime`,
`steered_cars_on_certified_test_days`, `steered_cars_in_e_prime_i`) and the three shares
`share_of_test_days_in_e_prime`, `share_of_certified_test_days_in_e_prime` and
`share_of_steered_cars_in_e_prime_i` (`learn_run.e_prime_figures`). A share is null when its
denominator is 0. A configuration that failed holds only `{"reason": ...}` there (`learn_run`,
the held-out branch of `evaluate`). The held-out runs write it as the development runs do, the
same function, so the table needs no other file.

What the family cannot reach: the same figures for the development-day check
(`dev_day_checks`, `dev_day_verdicts`: `days_in_e_prime`, `share_of_certified_days_in_e_prime` and
the others) are written by a development run that is no step of the drive, and WP16 reads only
the drive, so they are not in this table.

Readings (alternatives in brackets):
  E1. One row per scoring run (each distinct learn step and reading of `wp16_o1.SPECS`) and unit,
      at the unit's headline configuration (`sealed_choice.headline` of the run's matrix); the
      reading "beside" takes the configuration of the same kappa, dial and departure kind in the
      beside family, as in O1. [Alternative: every configuration of the matrix.]
  E2. Every figure is the run's own, read as written: no count is added up and no share is
      computed here, and nothing is pooled over units (the share of one unit's days says nothing
      of another's). The two shares of days follow the plan's order: first of the certified test
      days, then of all counting test days. The third is the share of steered cars in E'_i over the
      steered cars of the certified test days.
  E3. A figure the run did not produce is "no figure: <why>" (or the step's own "failed:",
      "not run:" or "unreadable:" text), never 0 and never blank: a unit the truth build gave no
      window, a unit or configuration the run does not hold, a configuration that failed, a count
      that is not a whole number. A share the file writes as null is "no figure" with the reason
      (no certified test day, or no counting test day).
  E4. Counts that contradict one another (more days in E' than certified test days, more certified
      than counting test days, more cars in E'_i than steered cars) or a share that is not its
      count over its denominator (to 1e-9), or a share given where the denominator is 0, are
      refused: each figure of the row is "unreadable: ..." and the run's exit status is 1.
  E5. The WP15 runs (rho 0.1 and 0.2) are marked " [information only]" after a held-out run that
      WP14 fails, by the one decision `wp16_failed.information_only`, as in O1 (reading Q7).
  E6. A unit whose `status` is not "calibratable" (too few days to split) was never scored, yet
      the writer gives it counts of 0 and null shares (its test set is empty): the table shows
      "no figure: <O1's words for the unit, the gate's reason>" in every cell, never 0, and the
      exit status follows O1 (no change: "no figure" is not a failure). A unit with no status is
      "no figure" too. Other states that give zero counts: a unit scored with certified days 0
      (real: scored, nothing certified), a configuration that failed (a `reason`, E3), a unit the
      truth build abstained (not in the run: E3). No other state was found (`plan_unit` has two).
"""

import math

from fleetreserve import wp16_o1
from fleetreserve.wp16_cells import Missing, Table
from fleetreserve.wp16_drive import Missed, Source
from fleetreserve.wp16_failed import SUFFIX, information_note, information_only
from fleetreserve.wp16_learn import ModelRun, Runs, names_for, unit_test

NAME = "eprime-days"
CALIBRATABLE = "calibratable"  # `gate_run.plan_unit`: the only status of a unit that has test days
RUNG = "one-way"  # any rung: it only decides which of O1's texts is read for an unscored unit
DEFINITION_KEY = "e_prime_definition"
TOLERANCE = 1e-9
COLUMNS = (
    "run",
    "unit",
    "configuration",
    "counting test days",
    "certified test days",
    "days in E'",
    "share of certified test days in E'",
    "share of counting test days in E'",
    "steered cars on certified test days",
    "steered cars in E'_i",
    "share of steered cars in E'_i",
)
COUNT_KEYS = (
    "test_days",
    "certified_test_days",
    "days_in_e_prime",
    "steered_cars_on_certified_test_days",
    "steered_cars_in_e_prime_i",
)
# each share: (its key, the count above it, the count it is taken of)
SHARES = (
    ("share_of_certified_test_days_in_e_prime", "days_in_e_prime", "certified_test_days"),
    ("share_of_test_days_in_e_prime", "days_in_e_prime", "test_days"),
    (
        "share_of_steered_cars_in_e_prime_i",
        "steered_cars_in_e_prime_i",
        "steered_cars_on_certified_test_days",
    ),
)
NOTE = (
    "Per unit at the headline configuration, as the run's learn.json holds it (module text of "
    "fleetreserve.wp16_eprime, readings E1 to E5); nothing is added up or computed here and no "
    "figure is pooled over units. A share is of the certified test days first (the days with an "
    "offer above 0 that the gate certified), then of all counting test days, as the plan asks."
)


def runs_read() -> list[tuple[wp16_o1.Spec, str]]:
    """The scoring runs of the table, each with the label its rows carry: the distinct (learn step,
    reading) pairs of O1's list, in its order (the rung does not change E')."""
    seen: dict[tuple, wp16_o1.Spec] = {}
    for spec in wp16_o1.SPECS:
        seen.setdefault((spec.learn, spec.reading), spec)
    out = []
    for (learn, reading), spec in seen.items():
        label = learn.removeprefix("learn-")
        if reading == "beside":
            label += " (departures seen at the next hour, beside reading)"
        if spec.wp15:
            label += " (WP15 sensitivity)"
        out.append((spec, label))
    return out


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _all(cell: Missing) -> tuple:
    return (cell,) * (len(COLUMNS) - 3)


def _why_none(unit: str, absent: dict) -> str:
    if unit in absent:
        return f"no figure: no window ({absent[unit]})"
    return "no figure: the unit is not in the run"


def _figures(found: dict) -> tuple:
    """The eight figure cells (`COLUMNS[3:]`, in their order) of one configuration's `e_prime`
    entry (readings E3, E4)."""
    reason = found.get("reason")
    if reason is not None:
        text = reason if str(reason).startswith("failed:") else f"no figure: {reason}"
        return _all(Missing(text))
    for key in COUNT_KEYS:
        if not _whole(found.get(key)):
            return _all(Missing(f"no figure: the run holds no whole number {key}"))
    if not (
        found["days_in_e_prime"] <= found["certified_test_days"] <= found["test_days"]
        and found["steered_cars_in_e_prime_i"] <= found["steered_cars_on_certified_test_days"]
    ):
        return _all(
            Missing(
                "unreadable: the counts contradict one another (days in E' "
                f"{found['days_in_e_prime']}, certified test days {found['certified_test_days']}, "
                f"counting test days {found['test_days']}, steered cars in E'_i "
                f"{found['steered_cars_in_e_prime_i']} of "
                f"{found['steered_cars_on_certified_test_days']})"
            )
        )
    shares = {}
    for key, top, bottom in SHARES:
        share = found.get(key)
        up, of = found[top], found[bottom]
        if of == 0:
            if share is None:
                shares[key] = Missing(f"no figure: {bottom} is 0, so the run writes no share")
            else:
                shares[key] = Missing(f"unreadable: {key} {share!r} where {bottom} is 0")
        elif share is None:
            shares[key] = Missing(f"no figure: the run writes no {key}")
        elif not _number(share) or abs(share - up / of) > TOLERANCE:
            shares[key] = Missing(
                f"unreadable: {key} {share!r} is not {top} {up} over {bottom} {of}"
            )
        else:
            shares[key] = share
    return (
        found["test_days"],
        found["certified_test_days"],
        found["days_in_e_prime"],
        shares[SHARES[0][0]],
        shares[SHARES[1][0]],
        found["steered_cars_on_certified_test_days"],
        found["steered_cars_in_e_prime_i"],
        shares[SHARES[2][0]],
    )


def _unscored(run: ModelRun, unit: str, name: str, absent: dict, entry: dict) -> Missing:
    """The cell of a unit whose status is not "calibratable" (reading E6): the writer gives every
    day of such a unit the role "not calibratable" (`gate_run.plan_unit`), so its test set is empty
    and `learn_run.e_prime_figures` writes counts of 0 and null shares for it, though no day was
    scored. The words are O1's own for the same unit and run (`wp16_learn.unit_test`, which reads
    the reason the gate recorded: `learn_eval._rungs`)."""
    status = entry.get("status")
    if not isinstance(status, str):
        return Missing("no figure: the run holds no status for the unit")
    test, why = unit_test(run, unit, name, RUNG, "A", absent)
    if test is not None:
        return Missing(f"unreadable: the unit's status is {status!r} but a result is scored")
    return Missing(why)


def unit_rows(run: ModelRun, label: str, reading: str, units: list, absent: dict) -> list[tuple]:
    """The rows of `COLUMNS` of one run: each unit's figures at its configuration of `reading`."""
    names = names_for(run, units, reading)
    rows = []
    for unit in units:
        name = names[unit]
        entry = run.learn.get("units", {}).get(unit) if isinstance(run.learn, dict) else None
        if not isinstance(entry, dict):
            rows.append((label, unit, name or "", *_all(Missing(_why_none(unit, absent)))))
        elif name is None:
            why = "no figure: the matrix names no such configuration for the unit"
            rows.append((label, unit, "", *_all(Missing(why))))
        elif entry.get("status") != CALIBRATABLE:
            rows.append((label, unit, name, *_all(_unscored(run, unit, name, absent, entry))))
        elif not isinstance(entry.get("e_prime"), dict):
            why = f"no figure: the run holds no e_prime for the unit {unit}"
            rows.append((label, unit, name, *_all(Missing(why))))
        elif not isinstance(entry["e_prime"].get(name), dict):
            why = f"no figure: the run holds no E' figures for configuration {name}"
            rows.append((label, unit, name, *_all(Missing(why))))
        else:
            rows.append((label, unit, name, *_figures(entry["e_prime"][name])))
    return rows


def build(source: Source, runs: Runs, units: list, absent: dict) -> list[Table]:
    table = Table(
        NAME,
        "T2: the days in E', the event on which a certified day's steered cars stay in their "
        "certified sets, by scoring run (held-out units)",
        COLUMNS,
    )
    information = information_only(source)
    definition = ""
    for spec, label in runs_read():
        try:
            run = runs.get(spec.learn, spec.intervals)
        except Missed as problem:
            rows = [(label, unit, "", *_all(Missing(problem.text))) for unit in units]
        else:
            rows = unit_rows(run, label, spec.reading, units, absent)
            found = run.learn.get(DEFINITION_KEY)
            if not definition and isinstance(found, str):
                definition = found
        for row in rows:
            marked = information and spec.wp15
            table.add(row[0] + SUFFIX if marked else row[0], *row[1:])
    table.notes.append(NOTE)
    if definition:
        table.notes.append(f"The run's own definition: {definition}")
    if information:
        part = "the rows marked [information only], which are WP15 parts, are"
        table.notes.append(information_note(part))
    return [table]
