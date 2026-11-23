"""The gate decisions of the WP6 gate run and its plain-English summary (plan 13 WP6).

The plan: "go if the calibrated re-solve offer is >= 10% of the ceiling on 2 of 3 and the
oracle-departure offer >= 1.5 times it." The arithmetic is `gate_stat`'s; this module feeds it and
writes the outcomes for every reading the matrix names, never one verdict.
Decided by the project owner on 16 Nov 2026: each car's rate and energy need are taken as known
(only the departure is forecast). Decided on 18 Nov 2026: a single car with no usable forecast is
left unsteered and the offer is built from the rest. Decided on 12 Nov 2026: a re-solving rung
sees a car gone from the start of its rounded-down departure hour (the main reading, the family
"main"); seen only at the start of the following hour is reported beside it in its own block (the
family "beside"), never pooled with it. A unit that is not calibratable counts as not passing: a
reading taken by this report, with no decision on it recorded. Still open (each outcome line is
PROVISIONAL because it depends on them): which reading of the 1.5 rule, the departure floor,
which cars count for driver loss. Fixed by the plan, and still shown in full: the ceiling is the
hindsight ceiling and the gate is read at tier A (plan section 13, WP6), the dial of the
driver-risk rows is chosen by a rule fixed in advance (plan section 6), the hourly shape rule is
the one WP5 sets (plan section 2), kappa is the largest calibrated development offer (plan
section 7). The last line of the outcome block of each tier (`agreement_line`) says, from the
decisions themselves, whether all outcomes of the main family agree.

Tiers (plan section 6): the gate is decided and reported under each tier the matrix lists
(`report.tiers`), from that tier's test data (`gate_run` builds one per tier): tier A
(driver-protected) is the key `gate` of the results, tier B (deadline contract) `gate_tier_b`.
The plan reads the gate at tier A (plan section 13, WP6); tier B is reported beside it. The
summary labels every offer line and outcome with its tier and gives each tier its own agreement
line. A number is never formed across tiers.

Per pair of configurations (an empirical-departure configuration and its oracle partner), per
ceiling and per reading of the 1.5 rule (`gate_stat.RATIO_RULES`): on each gate unit the
calibrated offer is that of the matrix's gate rung on the unit's test days, the oracle offer that of
the partner on the same test days. A unit that is not calibratable (too few counting days), whose
configuration failed, or whose ceiling is missing on a test day passes neither rule: it is listed
with its reason and no number is made up for it. `gate_stat.gate_decision` decides on the units
with numbers; a missing unit then adds a failure of the 10% rule, and of the 1.5 rule where the
reading asks every unit to pass it.

Not computed: the plan's fallback "(else 2 of 2 NREL units)". A unit that is not calibratable
counts as not passing (a reading taken by this report; no decision on it is recorded): with one
such unit and 2 of 3 required, "go" needs both other units to pass, and under the `all_units`
reading of the 1.5 rule it means no go.

kappa selection (plan section 7, "kappa: largest calibrated development offer"): per reading of
`departure_seen` and per unit, the empirical configuration of the main or the beside family with
that reading whose gate-rung offer in kWh per day on the test days is the largest (the first in
matrix order on a tie). This uses the same development days the offers are measured on, which the
summary says. On the sealed route (held-out units) the selection is not computed: `gate_tables`
with `selection` false writes no `kappa_selected_per_unit`, since the selection would pick
kappa on the scored units' test days, which plan section 7 fixes before the tag; every reader
below then reads an empty selection.

Every decision is reported per reading of `departure_seen`, since a pair's configurations carry one
reading each: the main reading's in the main tables and outcomes, the other reading's in its own
block, which has no agreement line and does not enter the main one.
"""

from fleetreserve import gate_stat
from fleetreserve.gate_config import TIER_LABELS, Matrix

PROVISIONAL = "PROVISIONAL"
# The days a results document's offers are measured on (`days_word`): development days, or the
# held-out days of a run on the sealed route, whose document holds the key `ROUTE_KEY` (written
# by `sealed_route.mark`).
DEVELOPMENT_DAYS = "development"
HELD_OUT_DAYS = "held-out"
ROUTE_KEY = "sealed_route"
SELECTION_KEY = "kappa_selected_per_unit"


def _decided_text(matrix: Matrix) -> str:
    others = len(matrix.gate_units) - 1
    if matrix.required_units == others:
        needs = "both other units" if others == 2 else f"all {others} other units"
    elif matrix.required_units < others:
        needs = f"{matrix.required_units} of the {others} other units"
    else:
        needs = 'more units than are left (so "go" is not possible)'
    return (
        "Decided by the project owner on 16 Nov 2026: each car's rate and energy need are taken "
        "as known (only its departure is forecast). "
        "Decided by the project owner on 18 Nov 2026: a car with no usable forecast is left "
        "unsteered and the offer is built from the rest. "
        "A unit that is not calibratable counts as not passing (a reading taken by this report; "
        f'no decision on it is recorded), so with one such unit "go" needs {needs}. '
        "Decided by the project owner on 12 Nov 2026: a re-solving rung sees a car gone from the "
        "start of its rounded-down departure hour (the main reading); seen only at the start of "
        "the following hour is reported beside it, never pooled with it."
    )


def _has_main(matrix: Matrix) -> bool:
    return any(c.family == "main" for c in matrix.configurations)


def _main_values(matrix: Matrix, field: str) -> str:
    """The values of `field` that the main family carries, or, for a matrix with no main
    family, that all its configurations carry, in matrix order."""
    found = []
    main = _has_main(matrix)
    for c in matrix.configurations:
        value = getattr(c, field)
        if (c.family == "main" or not main) and value not in found:
            found.append(value)
    return ", ".join(found)


def _uses(matrix: Matrix, field: str) -> str:
    """The clause naming the values of `field` the matrix uses."""
    values = _main_values(matrix, field)
    if _has_main(matrix):
        return f"the main family uses {values}"
    return f"this matrix has no main family; its configurations use {values}"


CEILING_WORDS = {
    "hindsight-free-decision-hour": "the hindsight ceiling",
    "best-causal": "the best-causal ceiling",
}


def _dial_values(matrix: Matrix, family: str) -> list[str]:
    """The dials of the driver-risk rows that the configurations of `family` carry, in matrix
    order ("none": the set without rows)."""
    found = []
    for c in matrix.configurations:
        word = "none" if c.driver_risk is None else f"{c.driver_risk:g}"
        if c.family == family and word not in found:
            found.append(word)
    return found


def _open_text(matrix: Matrix, selected: bool = True) -> str:
    """What is still open and what the plan fixes; `selected` false (the sealed route): no
    selection per unit is shown."""
    shown = (
        "every kappa and the per-unit selection of the largest offer are shown"
        if selected
        else "every kappa is shown; on the sealed route no kappa is selected per unit on the "
        "scored units' test days"
    )
    if _has_main(matrix):
        others = [d for d in _dial_values(matrix, "alternate") if d != "none"]
        dials = f"the main family uses {', '.join(_dial_values(matrix, 'main'))}"
        if others:
            dials += f"; alternates at the dials {', '.join(others)} are shown"
    else:
        dials = (
            "this matrix has no main family; its alternates use the dials "
            f"{', '.join(_dial_values(matrix, 'alternate'))}"
        )
    ceilings = " and ".join(CEILING_WORDS[c] for c in matrix.ceilings)
    tiers = " and ".join(TIER_LABELS[t] for t in matrix.tiers)
    return (
        f"Still open, so every outcome line below is {PROVISIONAL}: which reading of the 1.5 rule "
        "(the plan does not say on which units it must hold; each reading the matrix lists is "
        "shown); the departure floor (no decision on it is recorded; "
        f"{_uses(matrix, 'departure_floor')}); which cars count for driver loss (no "
        "decision on it is recorded; "
        f"{_uses(matrix, 'driver_loss_cars')}). Fixed by the plan: the ceiling in the "
        f"denominator is the hindsight ceiling (plan section 13, WP6; shown: {ceilings}); the gate "
        f"is read at tier A (plan section 13, WP6; shown, never pooled: {tiers}); the dial of the "
        f"driver-risk rows is chosen by a rule fixed in advance (plan section 6; {dials}); the "
        "hourly shape rule is the one WP5 sets (plan section 2; "
        f"{_uses(matrix, 'shape_rule')}); kappa is the largest calibrated development "
        f"offer (plan section 7; {shown})."
    )


def unit_offers(data: dict, ceiling: str):
    """`gate_stat.UnitOffer` from one unit's test data under one ceiling, or the reason there is
    none."""
    if data is None:
        return None, "no results"
    if "reason" in data:
        return None, data["reason"]
    ceilings = data["ceilings"][ceiling]
    if any(c is None for c in ceilings):
        return None, f"{ceiling} ceiling missing on a test day"
    try:
        return gate_stat.unit_offer(data["values"], ceilings, data["beta"]), None
    except ValueError as problem:
        return None, str(problem)


def decide(pairs: dict, matrix: Matrix, ceiling: str, rule: str) -> dict:
    """The gate under one ceiling and one reading of the 1.5 rule.

    `pairs[unit] = (calibrated test data, oracle test data)`; test data is a dict with `values`,
    `ceilings` (by ceiling name), `beta` and, for the summary, the offering, failing and test days
    of the gate rung, or `{"reason": ...}`. Each unit with a number carries the calibrated scale and
    those day counts."""
    numbered, missing, counts = [], {}, {}
    for unit in matrix.gate_units:
        calibrated, oracle = pairs.get(unit, (None, None))
        cal, why = unit_offers(calibrated, ceiling)
        ora, why_oracle = unit_offers(oracle, ceiling)
        if cal is None or ora is None:
            missing[unit] = why or f"oracle: {why_oracle}"
            continue
        numbered.append(gate_stat.GateUnit(unit, cal, ora))
        counts[unit] = {
            "offering_days": calibrated.get("offering_days"),
            "failing_days": calibrated.get("failing_days"),
            "test_days": calibrated.get("test_days"),
            "scale": calibrated["beta"],
        }
    units = {
        unit: {"passes_share": False, "passes_ratio": False, "reason": r}
        for unit, r in missing.items()
    }
    go = False
    n_share = n_both = 0
    if numbered:
        decision = gate_stat.gate_decision(
            numbered, required_units=min(matrix.required_units, len(numbered)), ratio_rule_on=rule
        )
        for row in decision.units:
            units[row.name] = {
                "calibrated_share": row.calibrated_share,
                "oracle_share": row.oracle_share,
                "calibrated_mean_kwh": row.calibrated_mean_kwh,
                "oracle_mean_kwh": row.oracle_mean_kwh,
                "ratio": row.ratio,
                "passes_share": row.passes_share,
                "passes_ratio": row.passes_ratio,
                **counts[row.name],
            }
        n_share, n_both = decision.n_pass_share, decision.n_pass_both
        enough = len(numbered) >= matrix.required_units
        go = decision.go and enough and not (rule == "all_units" and missing)
    return {
        "ceiling": ceiling,
        "ratio_rule_on": rule,
        "required_units": matrix.required_units,
        "go": go,
        "n_pass_share": n_share,
        "n_pass_both": n_both,
        "units": {unit: units[unit] for unit in matrix.gate_units},
    }


def gate_tables(matrix: Matrix, test_data: dict, selection: bool = True) -> dict:
    """Every gate decision: per pair of configurations, and with kappa selected per unit.

    `test_data[(configuration name, unit)]`: the unit's test data under that configuration (see
    `decide`). `selection` false (the sealed route, module text): no selection is computed or
    written."""
    names = {c.name for c in matrix.configurations}
    pairs_out = {}
    for c in matrix.configurations:
        if c.departure != "empirical" or c.oracle_partner not in names:
            continue
        pairs = {
            u: (test_data.get((c.name, u)), test_data.get((c.oracle_partner, u)))
            for u in matrix.gate_units
        }
        pairs_out[c.name] = {
            "oracle_partner": c.oracle_partner,
            "family": c.family,
            "kappa": c.kappa,
            "departure_seen": c.departure_seen,
            "driver_risk": c.driver_risk,
            "decisions": [
                decide(pairs, matrix, ceiling, rule)
                for ceiling in matrix.ceilings
                for rule in matrix.ratio_rules
            ],
        }
    if not selection:
        return {"by_pair": pairs_out}
    kappa_selected = {}
    for seen in _readings_seen(matrix):
        selected = select_kappa(matrix, test_data, seen)
        pairs = {}
        for unit, name in selected.items():
            if name is None:
                continue
            partner = matrix.configuration(name).oracle_partner
            pairs[unit] = (test_data.get((name, unit)), test_data.get((partner, unit)))
        kappa_selected[seen] = {
            "configuration_by_unit": selected,
            "note": "kappa chosen on the same development days the offers are measured on",
            "decisions": [
                decide(pairs, matrix, ceiling, rule)
                for ceiling in matrix.ceilings
                for rule in matrix.ratio_rules
            ],
        }
    return {"by_pair": pairs_out, SELECTION_KEY: kappa_selected}


SELECTED_FAMILIES = ("main", "beside")


def _readings_seen(matrix: Matrix) -> list[str]:
    """The readings of `departure_seen` of the empirical configurations of the main and the beside
    family, the main family's first."""
    found = []
    for c in sorted(matrix.configurations, key=lambda c: c.family != "main"):
        if (
            c.family in SELECTED_FAMILIES
            and c.departure == "empirical"
            and c.departure_seen not in found
        ):
            found.append(c.departure_seen)
    return found


def select_kappa(matrix: Matrix, test_data: dict, seen: str) -> dict:
    """Per unit, the empirical configuration of the main or the beside family with departures seen
    as `seen` and the largest gate-rung offer."""
    out = {}
    for unit in matrix.gate_units:
        best, best_offer = None, None
        for c in matrix.configurations:
            if c.family not in SELECTED_FAMILIES or c.departure != "empirical":
                continue
            if c.departure_seen != seen:
                continue
            data = test_data.get((c.name, unit))
            if data is None or "reason" in data or not data["values"]:
                continue
            offer = data["beta"] * sum(data["values"]) / len(data["values"])
            if best_offer is None or offer > best_offer:
                best, best_offer = c.name, offer
        out[unit] = best
    return out


def _pct(value) -> str:
    return "   -  " if value is None else f"{100 * value:6.2f}%"


def _days(rung: dict) -> str:
    """The scale and the offering and failing test days of one rung's results; at scale 0 the
    reason instead of a failing count (nothing was offered, so no day can fail)."""
    if "test" not in rung:
        return "-"
    test, scale = rung["test"], rung["beta_hat"]
    offering = f"{test['offering_days']}/{test['days']} offering"
    if scale > 0:
        return f"scale {scale:.3g}, {offering}, {test['failing_days']} failing"
    return f"scale 0 ({rung['reason']}), {offering}, no offer made"


def _kwh(value) -> str:
    return "     -" if value is None else f"{value:7.3f}"


def _tier_gate(document: dict, tier: str) -> dict | None:
    """The gate tables of `tier` in a results document (tier A is the key `gate`), or None."""
    return document["gate"] if tier == "A" else document.get("gate_tier_b")


def _tier_rung(rung: dict, tier: str) -> dict | None:
    """One rung's results as scored for `tier` (tier A: the rung's own; tier B: `tier_b`), in the
    same shape (`beta_hat`, `reason`, `test`), or None when the rung was not scored for it."""
    return rung if tier == "A" else rung.get("tier_b")


def _stay_text(rung: dict | None) -> str:
    """The stay-to-bound check of a tier B rung, or nothing when it is not in the results."""
    found = None if rung is None else rung.get("stay_to_bound")
    if found is None:
        return ""
    return (
        f"; cars staying to their bound: {found['cars_checked']} checked, "
        f"{found['cars_short']} short, largest shortfall {found['largest_shortfall_kwh']:.3g} kWh"
    )


TIER_TEXT = (
    "Two tiers (plan section 6) are reported side by side and never added, averaged or pooled: "
    f"{TIER_LABELS['A']}: the scale is the smaller of the slot and driver scales, as in T6; "
    f"{TIER_LABELS['B']}: the scale comes from the slot loss alone, the driver loss is reported "
    "and the shortfall of cars that stay to their bound is checked. The plan reads the gate at "
    "tier A (plan section 13, WP6); tier B, where the matrix lists it, is reported beside it. "
    "Every offer line, gate outcome and comparison below is given under each tier and carries its "
    "label; no line is a verdict across tiers."
)


def _main_readings(gate: dict) -> set[str]:
    """The readings of `departure_seen` that the main family's pairs in `gate` carry."""
    return {e["departure_seen"] for e in gate["by_pair"].values() if e["family"] == "main"}


def _offer_rows(document: dict, matrix: Matrix, tier: str, family: str) -> list[str]:
    """The offer table of one family under `tier`: one row per configuration and unit."""
    rung = matrix.gate_rung
    label = TIER_LABELS[tier]
    configurations = document["configurations"]
    gate = _tier_gate(document, tier)
    chosen = {} if gate is None else gate.get(SELECTION_KEY, {})
    lines = []
    for c in matrix.configurations:
        if c.family != family or c.departure != "empirical" or c.name not in configurations:
            continue
        selected = chosen.get(c.departure_seen, {}).get("configuration_by_unit", {})
        for unit in matrix.gate_units:
            row = configurations[c.name]["units"].get(unit, {})
            partner = configurations.get(c.oracle_partner, {}).get("units", {}).get(unit, {})
            front = f"  {c.departure_seen:<14}  {label:<27}  {unit:<26} {c.kappa:<5}"
            if "rungs" not in row:
                lines.append(f"{front}  {row.get('reason', 'no result')}")
                continue
            own = _tier_rung(row["rungs"][rung], tier)
            if own is None:
                lines.append(f"{front}  no {label} result")
                continue
            test = own["test"]
            shares = test["share_of_ceiling"]
            oracle_rung = _tier_rung(partner.get("rungs", {}).get(rung, {}), tier) or {}
            oracle = oracle_rung.get("test", {})
            cal_mean = test["mean_offer_kwh_per_day"]
            ora_mean = oracle.get("mean_offer_kwh_per_day")
            ratio = None if not cal_mean or ora_mean is None else ora_mean / cal_mean
            mark = "  <- largest offer" if selected.get(unit) == c.name else ""
            stay = _stay_text(own) if tier == "B" else ""
            lines.append(
                f"{front} {_kwh(cal_mean)}  "
                f"{_pct((shares['hindsight-free-decision-hour'] or {}).get('ratio_of_sums'))}"
                f"          {_pct((shares['best-causal'] or {}).get('ratio_of_sums'))}"
                f"        {_kwh(ora_mean)}  {'   -' if ratio is None else f'{ratio:5.2f}'}"
                f"  {_days(own)}; {_days(oracle_rung)}{stay}{mark}"
            )
    return lines


def days_word(document: dict) -> str:
    """The days the offers of `document` are measured on: held-out days when the run scored
    held-out units through the sealed route (the key `ROUTE_KEY`), else development days."""
    return HELD_OUT_DAYS if document.get(ROUTE_KEY) else DEVELOPMENT_DAYS


def _decision_lines(gate: dict, label: str, family: str, main_readings: set[str]) -> list[str]:
    """The pair decisions of `family` and the kappa-selected decisions of its readings, under
    `label`: the main readings for the main family, the others for the beside family."""
    lines = []
    for entry in gate["by_pair"].values():
        if entry["family"] != family:
            continue
        for d in entry["decisions"]:
            lines.append(
                f"  {PROVISIONAL} departure seen {entry['departure_seen']}, kappa "
                f"{entry['kappa']}: {label}: " + _decision_text(d)
            )
    for seen, entry in gate.get(SELECTION_KEY, {}).items():
        if (seen in main_readings) != (family == "main"):
            continue
        lines.append(
            f"  departure seen {seen}: kappa selected per unit ({entry['configuration_by_unit']}); "
            f"{label}; the selection uses the same development days:"
        )
        for d in entry["decisions"]:
            lines.append(
                f"  {PROVISIONAL} departure seen {seen}, kappa selected: {label}: "
                + _decision_text(d)
            )
    return lines


def _outcome_lines(document: dict, matrix: Matrix, tier: str) -> list[str]:
    """The gate outcomes of the main family under `tier`, ending with its agreement line."""
    label = TIER_LABELS[tier]
    gate = _tier_gate(document, tier)
    if gate is None:
        return [f"Gate outcomes, {label}: no result in this document."]
    lines = [
        f"Gate outcomes, {label} ({PROVISIONAL}: each line depends on the open decisions above; "
        f"required units {matrix.required_units} of {len(matrix.gate_units)}):"
    ]
    lines += _decision_lines(gate, label, "main", _main_readings(gate))
    lines.append(agreement_line(document, tier))
    return lines


BESIDE_BLOCK = (
    "departures seen at the start of the following hour (reported beside the main reading)"
)


def _beside_lines(document: dict, matrix: Matrix, header: str) -> list[str]:
    """The block of the beside family (module text): its offer rows and its decisions under each
    tier. It has no agreement line and is not part of the main family's."""
    lines = [
        f"Block: {BESIDE_BLOCK}, never pooled with it. The same candidate as above with a "
        "rung seeing a car gone only at the start of the hour after its departure hour; the "
        "columns are those of the main table. These outcomes are not in any agreement line.",
        header,
    ]
    for tier in matrix.tiers:
        lines += _offer_rows(document, matrix, tier, "beside")
    for tier in matrix.tiers:
        label = TIER_LABELS[tier]
        gate = _tier_gate(document, tier)
        if gate is None:
            lines.append(f"  {label}: no result in this document.")
            continue
        lines.append(
            f"  Outcomes for {label} ({PROVISIONAL}), departures seen at the start of the "
            "following hour:"
        )
        lines += _decision_lines(gate, label, "beside", _main_readings(gate))
    return lines


def _alternate_lines(document: dict, matrix: Matrix, tier: str) -> list[str]:
    """The alternates' decisions under `tier`."""
    label = TIER_LABELS[tier]
    gate = _tier_gate(document, tier)
    lines = []
    for name, entry in ({} if gate is None else gate["by_pair"]).items():
        if entry["family"] != "alternate":
            continue
        other = matrix.configuration(name)
        not_chosen = (
            "; side not chosen"
            if other.energy_bounds != "known" or other.uncertifiable_car != "leave-unsteered"
            else ""
        )
        risk = (
            ""
            if entry.get("driver_risk") is None
            else f"; driver-risk rows at epsilon {entry['driver_risk']}"
        )
        for d in entry["decisions"]:
            lines.append(
                f"  {PROVISIONAL} {name} (departure seen {entry['departure_seen']}{risk}"
                f"{not_chosen}): {label}: " + _decision_text(d)
            )
    return lines


def calibration_rule_text(rule: dict) -> str:
    """The matrix values of the calibration-days rule and what a split unit may still do."""
    below = rule["calibration_half_below"]
    return (
        f"Calibration days per unit: n_calibration {rule['n_calibration']}, "
        + (
            "no half-the-days rule (calibration_half_below null)"
            if below is None
            else f"or half of the unit's days (rounded down) under {below} days "
            f"(calibration_half_below {below})"
        )
        + ". A unit that is split can still abstain: the calibration offers nothing on too few "
        "offering calibration days."
    )


def summary_lines(document: dict, matrix: Matrix) -> list[str]:
    lines = [f"Truth tables: {document.get('inputs', {}).get('setting', 'main')} setting."]
    if document["trial"]:
        lines.append("TRIAL, not a gate result (--limit-days was given).")
    if document["partial"]:
        lines.append("PARTIAL RUN, not a gate result (--only was given).")
    t5 = document["t5"]
    if t5["breaks_a"] or t5["breaks_b"]:
        lines.append(
            f"T5 BREAKS: {t5['breaks_a']} of (a) and {t5['breaks_b']} of (b) on "
            f"{t5['day_paths_checked']} day-paths checked (each one is in gate-t5-breaks.csv)."
        )
    timeouts = document["timeouts"]
    if any(timeouts.values()):
        lines.append(f"TIMEOUTS (recorded as missing, never as 0 or infeasible): {timeouts}")
    if document["failed_configurations"]:
        lines.append(f"FAILED CONFIGURATIONS: {document['failed_configurations']}")
    lines.append(
        f"T5 on every evaluated day-path: {t5['day_paths_checked']} checked, "
        f"{t5['breaks_a']} breaks of (a), {t5['breaks_b']} of (b)."
    )
    lines.append("")
    rule = document["calibration_rule"]
    lines.append(calibration_rule_text(rule))
    lines.append("Units (counting days; the plan asks for at least 100 each):")
    for unit, entry in document["units"].items():
        if entry["calibratable"]:
            state = (
                f"split into {entry['calibration_days_used']} calibration days and "
                f"{entry['test_days']} test days"
            )
        else:
            state = entry["status"]
        lines.append(
            f"  {unit}: {entry['counting_days']} days, at least 100: "
            f"{'yes' if entry['at_least_100_counting_days'] else 'no'}; {state}"
        )
    lines.append("")
    lines.append(TIER_TEXT)
    lines.append("")
    rung = matrix.gate_rung
    lines.append(
        f"Main candidate family, gate rung {rung!r}, offers on test days (kWh per day), shares of "
        "the ceiling (ratio of sums), oracle-departure offer and the ratio oracle / calibrated, "
        f"once for each tier reported ({', '.join(TIER_LABELS[t] for t in matrix.tiers)}). "
        "Main reading: a re-solving rung sees a car gone from the start of its rounded-down "
        "departure hour."
    )
    lines.append(
        "Each offer is the mean over all test days. The last column gives, for the calibrated "
        "offer and for the oracle, the scale, the offering days (test days whose box at scale 1 "
        "has value above zero, whether or not anything is offered) out of the test days, and the "
        "failing days (offering days with slot loss above zero). At scale 0 nothing is offered: "
        "an offer of 0.000 then is an abstention, not a safe offer, there is no loss to count, and "
        "the column says why the scale is 0 (too few offering calibration days, or no scale "
        "the bound allows). Under tier B the calibrated offer's column also gives the cars "
        "staying to their bound that were checked and how many of them were short."
    )
    header = (
        "  departure seen  tier                         unit                       kappa  offer  "
        "share(hindsight) share(best-causal)  oracle  ratio  scale, offering days/test days, "
        "failing days (calibrated; oracle)"
    )
    lines.append(header)
    for tier in matrix.tiers:
        lines += _offer_rows(document, matrix, tier, "main")
    lines.append("")
    lines.append(_decided_text(matrix))
    lines.append(_open_text(matrix, selected=not document.get(ROUTE_KEY)))
    for tier in matrix.tiers:
        lines += _outcome_lines(document, matrix, tier)
        lines.append("")
    if any(c.family == "beside" for c in matrix.configurations):
        lines += _beside_lines(document, matrix, header)
        lines.append("")
    lines.append(
        f"Alternates (comparisons only; each changes one field of a candidate of the same kappa: "
        "the energy-bounds, whole-day-abstains and rounded-down-departure alternates of the "
        "candidate with departures seen at the start of the following hour, the dial alternates "
        f'of the main candidate), gate rung {rung!r} ({PROVISIONAL}). A line marked "side not '
        'chosen" takes the other side of a decision made by the project owner (energy bounds '
        "empirical, or the whole day abstains when a car has no usable forecast):"
    )
    for tier in matrix.tiers:
        lines += _alternate_lines(document, matrix, tier)
    return lines


VARIED_IN_MAIN = "kappa, the ceiling, the reading of the 1.5 rule and the per-unit kappa selection"
HELD_IN_MAIN = (
    "The hourly shape rule, the departure floor and which cars count for driver loss are held at "
    "one value in the main family"
)


def agreement_line(document: dict, tier: str | None = None) -> str:
    """Whether all outcomes of the main family agree, from the decisions of the results document.

    The outcomes are every decision of the main family's pairs (each kappa, ceiling and reading of
    the 1.5 rule) and of the per-unit kappa selection of the main family's reading. Alternates and
    the beside family (the other reading of `departure_seen`) do not enter. Nothing here is fixed
    in advance: with no outcome the line says so and claims no agreement, and with different
    outcomes it gives the counts. The line names only the settings the main family varies
    (`VARIED_IN_MAIN`) and says the settings it holds at one value (`HELD_IN_MAIN`) are not
    covered. Settings that only the alternates change (energy bounds, car with no usable
    forecast, departure floor) are not covered either.

    `tier`: the tiers are never pooled, so a line covers one tier's decisions (`_tier_gate`) and,
    when `tier` is given, ends by saying which; `None` reads tier A's and adds nothing."""
    gate = _tier_gate(document, tier or "A")
    outcomes = [
        d["go"]
        for entry in gate["by_pair"].values()
        if entry["family"] == "main"
        for d in entry["decisions"]
    ]
    main_readings = _main_readings(gate)
    for seen, entry in gate.get(SELECTION_KEY, {}).items():
        if seen in main_readings:
            outcomes += [d["go"] for d in entry["decisions"]]
    scope = "" if tier is None else f" This line covers {TIER_LABELS[tier]} only."
    if not outcomes:
        return "No main-family outcome was computed; nothing can be said about agreement." + scope
    n_go = sum(1 for go in outcomes if go)
    n_no = len(outcomes) - n_go
    if n_go and n_no:
        return (
            f"The main-family outcomes differ: {n_go} go, {n_no} no go; the outcome depends on "
            f"at least one of {VARIED_IN_MAIN}." + scope
        )
    verdict = "go" if n_go else "no go"
    return (
        f"All {len(outcomes)} main-family outcomes agree: {verdict}; varying {VARIED_IN_MAIN} "
        f"does not change the outcome. {HELD_IN_MAIN}, so this says nothing about them." + scope
    )


def _counts(row: dict) -> str:
    """The calibrated scale, offering and failing test days of a unit in a decision."""
    if row.get("offering_days") is None:
        return "no day counts"
    offering = f"{row['offering_days']}/{row['test_days']} offering"
    if row["scale"] > 0:
        return f"scale {row['scale']:.3g}, {offering}, {row['failing_days']} failing"
    return f"scale 0, {offering}, no offer made"


def _decision_text(d: dict) -> str:
    units = []
    for unit, row in d["units"].items():
        if "reason" in row:
            units.append(f"{unit} counts as not passing ({row['reason']})")
        else:
            ratio = row["ratio"]
            units.append(
                f"{unit} share {_pct(row['calibrated_share']).strip()} "
                f"ratio {'-' if ratio is None else f'{ratio:.2f}'} ({_counts(row)})"
            )
    verdict = "go" if d["go"] else "no go"
    return (
        f"ceiling {d['ceiling']}, 1.5 rule on {d['ratio_rule_on']}: {verdict} "
        f"({d['n_pass_share']} pass 10%, {d['n_pass_both']} pass both; " + "; ".join(units) + ")"
    )
