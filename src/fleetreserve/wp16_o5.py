"""Outcome O5: the properties that must hold, and the drift guard's step sizes (plan section 11).

"O5 Property (a violation is a bug): T9 identity, T5 orderings, offer ratio >= 1 on 100% of
day-paths; step sizes X1-X4."

Read from three steps of the drive:
- `ledger-lgbm_tail` (`fleetreserve.ledger_run`, `ledger.json`): `checks` holds the T9 identity
  (every day and every mean), the reproduction of the run's offers and ceilings, the plan's order of
  the terms, and `offer_order` (check 7: on every test day the fixed offer is at most the one-way
  offer and that at most the two-way offer); per unit and configuration `offer_ratio.<name>.<tier>`
  holds the shares of test days on which each step holds and the days that break it;
- `learn-lgbm_tail` (`learn.json`): `t5`, the day-paths checked for T5 (a) and (b) and the breaks;
- `drift-lgbm_tail` (`fleetreserve.drift_run`, `drift-run.json`): `sealed_route.step_sizes`, each
  unit's site type and its step size per tier (or "offers nothing").

Readings (alternatives in brackets):
  Y1. A property holds when its count of breaks is 0 and its checked count is above 0; a count that
      is missing is "no figure". "Offer ratio >= 1 on 100% of day-paths" is read as the order of
      check 7 and the share of test days of `offer_order`, at each unit's headline configuration,
      tier A. [Alternative: every configuration and tier.]
  Y2. X1 to X4 are the four step sizes of plan section 6 (T10), two tiers for each of the two site
      types, read from `sealed_route.settings.eta` of `drift-run.json` (tier -> site type -> step
      size, or "offers nothing"), the record of the settings file as the tag's commit holds it;
      its git object and SHA-256 are named in the table's note. The per-unit step sizes of the
      units the drift step ran on (`sealed_route.step_sizes`) are not tabled.
  Y3. `o5-drift` (`build_drift`) gives the end values of the drift run as `drift-run.json` holds
      them (`units.<unit>.paths.<tier>.<start>.<step size>`, the summary of
      `drift_path.Path.summary`): per unit, tier, start and step size, the steps taken (the
      offering days), the mean slot loss at the end with T10's bound at the end and the level
      (alpha), the mean driver loss with its level (none where the driver loss is reported, not
      held down), and the steps whose running mean was over the bound, summed over the losses
      held down as the run's own summary text sums them, the path's mean offer per counting day
      (`mean_offer_kwh_per_counting_day`) and the driver loss's bound at the end
      (`final_bound`, only where the driver loss is held down). It is a WP15 part: after a
      held-out run that WP14 fails (a step failed or not run, or a configuration failed on a
      unit) it is marked "information only" (plan section 13, WP15), by the one decision of
      `wp16_failed.information_only` that every WP15 part of the tables uses. A figure the file
      does not hold is "not produced: <why>", never 0.
"""

from fleetreserve.wp16_cells import Missing, Table
from fleetreserve.wp16_drive import Missed, Source
from fleetreserve.wp16_failed import information_note, information_only
from fleetreserve.wp16_learn import Runs, names_for

LEDGER = "ledger-lgbm_tail"
LEARN = "learn-lgbm_tail"
DRIFT = "drift-lgbm_tail"
COLUMNS = ("property", "checked", "breaks", "largest residual or detail", "holds")
RATIO_COLUMNS = (
    "unit",
    "configuration",
    "test days",
    "fixed <= one-way: share of test days",
    "one-way <= two-way: share of test days",
    "days that break fixed <= one-way",
    "days that break one-way <= two-way",
    "holds",
)
STEP_COLUMNS = ("tier", "site type", "step size")
DRIFT_COLUMNS = (
    "unit",
    "tier",
    "start",
    "step size",
    "steps taken",
    "slot loss, mean at the end",
    "slot loss, T10 bound at the end",
    "slot loss level",
    "driver loss, mean at the end",
    "driver loss level",
    "steps over the bound",
    "mean offer (kWh per counting day)",
    "driver loss, T10 bound at the end",
    "information only",
)
NOT_PRODUCED = "not produced: "
NOT_HELD_DOWN = "not held down (reported)"


def _count(found) -> int | None:
    return len(found) if isinstance(found, list) else None


def _holds(checked, breaks):
    if breaks is None or checked is None:
        return Missing("no figure: the count is not in the file")
    return breaks == 0 and checked > 0


def _checks_rows(table: Table, source: Source) -> None:
    try:
        checks = source.json(LEDGER, "ledger.json").get("checks")
    except Missed as problem:
        table.add("T9 identity and the ledger's checks", Missing(problem.text), "", "", "")
        checks = False
    if checks is not False and not isinstance(checks, dict):
        table.add(
            "T9 identity and the ledger's checks", Missing("no figure: no checks"), "", "", ""
        )
    if isinstance(checks, dict):
        identity = checks.get("identity", {})
        breaks = _count(identity.get("breaks"))
        days, means = identity.get("days_checked"), identity.get("means_checked")
        checked = days + means if isinstance(days, int) and isinstance(means, int) else None
        table.add(
            "T9 identity: offer + R + A + U + K = ceiling, every day and every mean",
            checked,
            breaks,
            identity.get("largest_residual_kwh"),
            _holds(checked, breaks),
        )
        for key, title in (
            ("run_offer_reproduced", "the run's offers reproduced from the day rows"),
            ("run_ceiling_sums_reproduced", "the run's ceiling sums reproduced"),
            ("sampled_days_are_the_runs", "the sampled days are the run's days"),
            ("order", "R + A, R, A and the other terms not below 0"),
            ("ceiling_ratio_one_way_reproduced", "the one-way ceilings of O8 reproduced"),
            ("offer_order", "offer order: fixed <= one-way <= two-way on every test day"),
        ):
            found = checks.get(key)
            count = _count(found.get("breaks")) if isinstance(found, dict) else None
            table.add(title, "", count, "", _holds(1, count))
        errors = _count(checks.get("sample_day_errors"))
        table.add("sampled days whose sets could not be built", "", errors, "", _holds(1, errors))
        table.add(
            "the ledger's own verdict",
            "",
            "",
            "",
            Missing("no figure: the file holds no verdict")
            if "failed" not in checks
            else not checks["failed"],
        )
    try:
        t5 = source.json(LEARN, "learn.json").get("t5")
    except Missed as problem:
        table.add("T5 (a), (b) on every day-path", Missing(problem.text), "", "", "")
        return
    if isinstance(t5, dict):
        a, b = t5.get("breaks_a"), t5.get("breaks_b")
        both = a + b if isinstance(a, int) and isinstance(b, int) else None
        table.add(
            "T5 (a) and (b): no rung fails before its predecessor; each car ends with at least "
            "the predecessor's energy",
            t5.get("day_paths_checked"),
            both,
            f"breaks (a): {a}; breaks (b): {b}",
            _holds(t5.get("day_paths_checked"), both),
        )
    else:
        table.add(
            "T5 (a) and (b) on every day-path", Missing("no figure: no t5 record"), "", "", ""
        )


def _ratio_rows(table: Table, source: Source, runs: Runs, units: list) -> None:
    try:
        document = source.json(LEDGER, "ledger.json")
        names = names_for(runs.get(LEARN), units, "main")
    except Missed as problem:
        table.add("all units", "", "", Missing(problem.text), "", "", "", "")
        return
    found = document.get("units") if isinstance(document.get("units"), dict) else {}
    for unit in units:
        name = names[unit]
        entry = (
            found.get(unit, {}).get("offer_ratio", {}) if isinstance(found.get(unit), dict) else {}
        )
        one = entry.get(name, {}).get("A") if isinstance(entry.get(name), dict) else None
        if not isinstance(one, dict):
            table.add(
                unit,
                name or "",
                "",
                Missing("no figure: no offer ratio for the unit"),
                "",
                "",
                "",
                "",
            )
            continue
        first, second = one.get("fixed_one_way_broken_days"), one.get("one_way_two_way_broken_days")
        shares = (one.get("fixed_one_way_share"), one.get("one_way_two_way_share"))
        counts = (_count(first), _count(second))
        holds = (
            Missing("no figure: no days")
            if None in counts or None in shares
            else counts == (0, 0) and shares == (1.0, 1.0)
        )
        table.add(
            unit, name, one.get("test_days"), shares[0], shares[1], counts[0], counts[1], holds
        )


def _step_rows(table: Table, source: Source) -> None:
    try:
        route = source.json(DRIFT, "drift-run.json").get("sealed_route", {})
    except Missed as problem:
        table.add("all tiers", "", Missing(problem.text))
        return
    settings = route.get("settings") if isinstance(route, dict) else None
    eta = settings.get("eta") if isinstance(settings, dict) else None
    if not isinstance(eta, dict) or not eta:
        table.add("all tiers", "", Missing("no figure: the file holds no settings.eta"))
        return
    for tier in sorted(eta):
        found = eta[tier]
        for site_type in sorted(found) if isinstance(found, dict) else ():
            table.add(tier, site_type, found[site_type])
        if not isinstance(found, dict):
            table.add(tier, "all site types", found)
    table.notes.append(
        f"From {settings.get('file')} (git object {settings.get('git_object')}, SHA-256 "
        f"{settings.get('sha256')}), as the tag's commit holds it (reading Y2)."
    )


def _number(found: dict, key: str):
    """The file's own number under `key`, else the cell that says it holds none."""
    value = found.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return Missing(f"{NOT_PRODUCED}the file holds no {key}")


def _drift_row(unit: str, tier: str, start: str, eta: str, found, marked: str) -> tuple:
    losses = found.get("losses") if isinstance(found, dict) else None
    slot = losses.get("slot") if isinstance(losses, dict) else None
    driver = losses.get("driver") if isinstance(losses, dict) else None
    if not isinstance(slot, dict) or not isinstance(driver, dict):
        none = Missing(NOT_PRODUCED + "the file holds no slot and driver losses for this path")
        return (unit, tier, start, eta, none, *[""] * 8, marked)
    counts = [loss.get("days_over_bound") for loss in (slot, driver) if loss.get("guarded") is True]
    if counts and all(isinstance(n, int) and not isinstance(n, bool) for n in counts):
        over: int | Missing = sum(counts)
    else:
        over = Missing(f"{NOT_PRODUCED}the file holds no days_over_bound of a loss held down")
    return (
        unit,
        tier,
        start,
        eta,
        _number(found, "offering_days"),
        _number(slot, "mean"),
        _number(slot, "final_bound") if slot.get("guarded") is True else NOT_HELD_DOWN,
        _number(slot, "alpha") if slot.get("guarded") is True else NOT_HELD_DOWN,
        _number(driver, "mean"),
        _number(driver, "alpha") if driver.get("guarded") is True else NOT_HELD_DOWN,
        over,
        _number(found, "mean_offer_kwh_per_counting_day"),
        _number(driver, "final_bound") if driver.get("guarded") is True else NOT_HELD_DOWN,
        marked,
    )


def build_drift(source: Source) -> list[Table]:
    """`o5-drift`: the end values of the drift run (reading Y3), from the file `_step_rows` reads
    (`drift-run.json` of `drift-lgbm_tail`); nothing is recomputed."""
    table = Table(
        "o5-drift",
        "O5: the chronological run of the drift guard, values at its end (WP15; "
        "reported, not a pass mark)",
        DRIFT_COLUMNS,
    )
    information = information_only(source)
    marked = "yes" if information else "no"
    try:
        document = source.json(DRIFT, "drift-run.json")
    except Missed as problem:
        table.add("all units", Missing(NOT_PRODUCED + problem.text), *[""] * 11, marked)
        found = None
    else:
        found = document.get("units")
        if not isinstance(found, dict) or not found:
            none = Missing(NOT_PRODUCED + "the file holds no units")
            table.add("all units", none, *[""] * 11, marked)
            found = None
    for unit, entry in (found or {}).items():
        paths = entry.get("paths") if isinstance(entry, dict) else None
        paths = paths if isinstance(paths, dict) else {}
        status = entry.get("status") if isinstance(entry, dict) else None
        for tier in ("A", "B"):
            by_start = paths.get(tier)
            if not isinstance(by_start, dict) or not by_start:
                none = Missing(
                    f"{NOT_PRODUCED}no path of tier {tier} in the file (status: {status})"
                )
                table.add(unit, tier, none, *[""] * 10, marked)
                continue
            for start, by_eta in by_start.items():
                for eta, summary in (by_eta if isinstance(by_eta, dict) else {}).items():
                    table.add(*_drift_row(unit, tier, start, eta, summary, marked))
    if information:
        table.notes.append(information_note("this WP15 part is"))
    table.notes.append(
        "The end values as the drift run's own results file gives them, not recomputed: steps "
        "taken are the days the guard stepped (offering days); the bound is T10's bound after "
        "those steps; a level is the loss's alpha (a driver loss that is reported, not held "
        "down, has none); steps over the bound are summed over the losses held down, as the "
        "run's own summary text sums them; the mean offer is the path's mean over its guarded "
        "counting days (a day with no offer counts as 0), and the driver loss's bound is T10's "
        "bound at the end, where that loss is held down (reading Y3)."
    )
    return [table]


def build(source: Source, runs: Runs, units: list) -> list[Table]:
    properties = Table("o5-properties", "O5: properties that must hold (a break is a bug)", COLUMNS)
    _checks_rows(properties, source)
    ratio = Table(
        "o5-offer-ratio", "O5: offer order per unit (tier A, headline configuration)", RATIO_COLUMNS
    )
    _ratio_rows(ratio, source, runs, units)
    steps = Table("o5-step-sizes", "O5: the drift guard's step sizes (T10)", STEP_COLUMNS)
    _step_rows(steps, source)
    return [properties, ratio, steps]
