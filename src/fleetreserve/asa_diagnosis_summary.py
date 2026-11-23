"""The figures per unit and variant of `asa_diagnosis_run`, from the days' diagnoses
(`asa_diagnosis_day.DayDiagnosis`), and its plain-text summary. Pure: no files, no solver.

`variant_figures(diagnosed)`: `diagnosed` is a list of (day key, `DayDiagnosis`) in the order the
days were diagnosed. Returns, with every key defined in `DEFINITIONS`:

- `groups`: the counts of the (day, path, service hour) triples in each group of
  `asa_diagnosis_day.GROUPS` (missed by the method only, by the reference only, by both, by
  neither), summed over the days;
- `causes`: the method-only hours by cause, every cause of `asa_diagnosis_day.CAUSES` (zeros
  included); `cause_shares`: each count over the method-only hours (null when there is none);
- `choice_goals`: the hours of cause "choice" by the goal named (goal name -> count, only goals
  seen);
- `belief_deficits_kwh`: over the hours of cause "belief", the deficits of
  `asa_diagnosis_day` summed by label (every label, zeros included);
- `state`: over the hours of cause "state": `state_hours`, their number; `left_earlier_hours`, the
  hours with at least one car that left earlier than the method believed, and
  `left_earlier_share`, that over `state_hours`; `charged_ahead_hours`, the hours whose
  `charged_ahead` is above `asa_diagnosis_day.TOLERANCE` (the cars present in the hour held more
  energy under the method than under the reference), and `charged_ahead_share`; both shares null
  when `state_hours` is 0;
- `top_days`: up to five days with the most method-only hours, those with at least one, the most
  first and ties in the order diagnosed; each with its `day` key, `method_only_hours` and
  `causes` (its method-only hours by cause, every cause);
- `goal_tests`: the outcomes of the goal tests of `asa_diagnosis_lp` (outcome -> count, only
  outcomes seen);
- `solver_counts`: per role ("method", "reference") whose controllers report solver counts, their
  sum over the days (`compare_day.sum_counts`); a role without them is left out.

`summary_lines(document)`: the summary text of a results document of `asa_diagnosis_run`, which
opens with its label and `SCOPE`, what the figures can and cannot support.
"""

from collections import Counter

from fleetreserve.asa_diagnosis_day import CAUSES, GROUPS, LABELS, TOLERANCE
from fleetreserve.compare_day import sum_counts

TOP_DAYS = 5
SCOPE = (
    "These figures explain why the method misses service hours that the reference meets, at the "
    "scale of the diagnosis (without --scale, the reference's most frequent calibrated scale, "
    "D3), on the audited paths of the diagnosed development days. They are not a statement about "
    "the published algorithm a method is named after, not a ranking of methods, and not an "
    "explanation of the method's offer at its own calibrated scale."
)

DEFINITIONS = {
    "scope": "what the figures can and cannot support: " + SCOPE,
    "groups": (
        "(day, path, service hour) triples of the diagnosed days by group: method_only (the "
        "method's replay fails the hour and the reference's does not), reference_only, both, "
        "neither (replay.fails: a delivery missing its request by more than FAILURE_KWH)"
    ),
    "method_only": "the hours the method misses and the reference meets",
    "reference_only": "the hours the reference misses and the method meets",
    "both": "the hours both miss",
    "neither": "the hours neither misses",
    "causes": "the method-only hours by cause (asa_diagnosis_day, CAUSES; zeros included)",
    "cause_shares": "each cause's count over the method-only hours; null when there is none",
    "state": (
        "in causes and cause_shares, cause 'state': the request is above the upper end of the "
        "deliverable range from the method's own state by more than FAILURE_KWH; as a block of "
        "a variant's figures, the counts over the hours of that cause"
    ),
    "fallback": (
        "cause 'fallback': not 'state', and the method is a programme controller of "
        "asa_controller (asa, asa-es) whose programme of the hour ended in an outcome other than "
        "optimal, so the hour was planned by its fall-back. Any other controller has no programme "
        "per hour, so none of its hours is 'fallback': an hour diagonal-split plans by its own "
        "fall-back (least laxity first, counted in its fallback_hours) gets the cause any hour of "
        "a controller without a programme gets: 'state', 'belief', 'choice' with goal 'rule', or "
        "'other'"
    ),
    "belief": (
        "cause 'belief': not the above, the delivery fell short, and believed_on_true is below "
        "the request by more than FAILURE_KWH"
    ),
    "choice": (
        "cause 'choice': not the above, the delivery fell short, and a plan within the believed "
        "limits could have tracked on the true cars; shortfall = (request - planned fleet total) "
        "+ overplanned. For a programme controller goal 1 ('track') holds the planned total "
        "within FAILURE_KWH of the request, up to the kept row's tolerance, so the shortfall is, "
        "to within that, the power planned on cars above what they truly take"
    ),
    "other": "cause 'other': none of the above (a delivery above the request)",
    "choice_goals": (
        "the 'choice' hours by the goal that ruled out tracking (asa_diagnosis_lp.forbidding_goal "
        "for a programme controller: a stage name, 'tie' or 'not determined'; 'rule' for any "
        "other controller). A goal k named: every plan keeping the optima of goals 1 to k puts "
        "more power on some car than it truly takes, or plans a total more than FAILURE_KWH from "
        "the request, while some plan keeping goals 1 to k - 1 does neither; 'tie': a plan "
        "keeping every goal's optimum tracks, and the solver returned another. A name does not "
        "say that the controller could have known from its belief which split would deliver"
    ),
    "belief_deficits_kwh": (
        "over the 'belief' hours, the deficits max(high_i - most_i, 0) summed by label (plugged, "
        "departure, rate, cap: the term that sets most_i)"
    ),
    "plugged": "label: a truly present car not counted as plugged in",
    "departure": "label: most_i set by the believed rate times a share of the hour below 1",
    "rate": "label: most_i set by the believed rate (the whole hour believed plugged in)",
    "cap": "label: most_i set by the believed energy cap minus the energy drawn",
    "state_hours": "the hours of cause 'state'",
    "left_earlier_hours": (
        "'state' hours with at least one car gone by the hour (d <= t) whose believed departure "
        "is above its true departure hour (d_hat > d; asa_diagnosis_day.left_earlier). True "
        "departures are whole hours, the hour of unplugging rounded down, so a car believed to "
        "leave at 3.2 that left at 3.4 (d = 3) counts; one believed to leave at d does not"
    ),
    "left_earlier_share": "left_earlier_hours over state_hours; null when state_hours is 0",
    "charged_ahead_hours": (
        "'state' hours whose charged_ahead (the energy the cars present in the hour drew before "
        "it under the method minus under the reference) is above FAILURE_KWH"
    ),
    "charged_ahead_share": "charged_ahead_hours over state_hours; null when state_hours is 0",
    "top_days": (
        "up to five days with the most method-only hours (at least one), the most first, ties in "
        "the order diagnosed"
    ),
    "day": "a day's key, '<date> <site>', as gate_run.plan_unit writes it",
    "method_only_hours": "the day's method-only hours over its paths",
    "goal_tests": (
        "the outcomes of the tracking tests solved by asa_diagnosis_lp.forbidding_goal "
        "(outcome -> count, only outcomes seen); a test ending in neither optimal nor infeasible "
        "makes the goal 'not determined'"
    ),
    "solver_counts": (
        "per role (method, reference) whose controllers report solver counts (the controllers "
        "of asa_controller and of diagonal_split, each by its solver_counts): their sum over the "
        "diagnosed days and paths (compare_day.sum_counts)"
    ),
    "method": "the method (--method)",
    "reference": "the reference (--reference)",
    "solves": "the number of stage solves by outcome (lp.Outcome names)",
    "hours": "in solver_counts: the hours the controllers planned",
    "fallback_hours": "in solver_counts: the hours the controllers planned by the fall-back",
    "boxes": (
        "in solver_counts, for controllers that report it (diagonal_split): the number of boxes "
        "of each kind of diagonal_split.BOXES (compare_day.sum_counts)"
    ),
}


def _share(count: int, total: int) -> float | None:
    return None if total == 0 else count / total


def variant_figures(diagnosed) -> dict:
    """The figures of one unit and variant (module text)."""
    groups: Counter = Counter()
    causes: Counter = Counter()
    goals: Counter = Counter()
    tests: Counter = Counter()
    deficits = dict.fromkeys(LABELS, 0.0)
    state = {"state_hours": 0, "left_earlier_hours": 0, "charged_ahead_hours": 0}
    per_day = []
    solved = {"method": [], "reference": []}
    for key, day in diagnosed:
        groups.update(day.groups)
        tests.update(day.goal_tests)
        here: Counter = Counter()
        for row in day.rows:
            here[row.cause] += 1
            if row.cause == "choice":
                goals[row.goal] += 1
            elif row.cause == "belief":
                for label in LABELS:
                    deficits[label] += getattr(row, f"deficit_{label}")
            elif row.cause == "state":
                state["state_hours"] += 1
                state["left_earlier_hours"] += row.cars_left_earlier > 0
                state["charged_ahead_hours"] += row.charged_ahead > TOLERANCE
        causes.update(here)
        per_day.append((key, len(day.rows), here))
        for role, counts in (
            ("method", day.method_solver_counts),
            ("reference", day.reference_solver_counts),
        ):
            if counts is not None:
                solved[role].append(counts)
    method_only = groups["method_only"]
    state["left_earlier_share"] = _share(state["left_earlier_hours"], state["state_hours"])
    state["charged_ahead_share"] = _share(state["charged_ahead_hours"], state["state_hours"])
    ranked = sorted(
        (entry for entry in enumerate(per_day) if entry[1][1] > 0), key=lambda e: (-e[1][1], e[0])
    )
    return {
        "groups": {g: groups[g] for g in GROUPS},
        "causes": {c: causes[c] for c in CAUSES},
        "cause_shares": {c: _share(causes[c], method_only) for c in CAUSES},
        "choice_goals": dict(sorted(goals.items())),
        "belief_deficits_kwh": deficits,
        "state": state,
        "top_days": [
            {"day": key, "method_only_hours": count, "causes": {c: here[c] for c in CAUSES}}
            for _, (key, count, here) in ranked[:TOP_DAYS]
        ],
        "goal_tests": dict(sorted(tests.items())),
        "solver_counts": {role: sum_counts(found) for role, found in solved.items() if found},
    }


def _number(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _variant_lines(name: str, entry: dict) -> list[str]:
    head = f"  {name} ({entry['configuration']}): "
    if "failed" in entry:
        return [head + f"FAILED: {entry['failed']}"]
    if entry["status"] != "diagnosed":
        return [head + entry["status"]]
    figures = entry["figures"]
    groups = figures["groups"]
    lines = [
        head + f"scale {entry['scale']:g} ({entry['scale_source']}), "
        f"{entry['days_diagnosed']} days of {entry['offering_days']} offering",
        "    hours missed: method only {method_only}, reference only {reference_only}, "
        "both {both}, neither {neither}".format(**groups),
    ]
    shares = figures["cause_shares"]
    lines.append(
        "    method-only hours by cause: "
        + ", ".join(f"{c} {figures['causes'][c]} ({_number(shares[c])})" for c in figures["causes"])
    )
    if figures["choice_goals"]:
        lines.append(
            "    'choice' hours by goal: "
            + ", ".join(f"{g} {n}" for g, n in figures["choice_goals"].items())
        )
    deficits = figures["belief_deficits_kwh"]
    if figures["causes"]["belief"]:
        lines.append(
            "    'belief' hours, deficits by label (kWh): "
            + ", ".join(f"{k} {_number(v)}" for k, v in deficits.items())
        )
    state = figures["state"]
    if state["state_hours"]:
        lines.append(
            f"    'state' hours: {state['state_hours']}; a car left earlier than believed in "
            f"{state['left_earlier_hours']} ({_number(state['left_earlier_share'])}); the method "
            f"had charged the present cars ahead of the reference in "
            f"{state['charged_ahead_hours']} ({_number(state['charged_ahead_share'])})"
        )
    if figures["top_days"]:
        lines.append("    days with the most method-only hours:")
        for top in figures["top_days"]:
            causes = ", ".join(f"{c} {n}" for c, n in top["causes"].items() if n)
            lines.append(f"      {top['day']}: {top['method_only_hours']} ({causes})")
    if figures["goal_tests"]:
        lines.append(
            "    goal tests by outcome: "
            + ", ".join(f"{o} {n}" for o, n in figures["goal_tests"].items())
        )
    for role, counts in figures["solver_counts"].items():
        solves = ", ".join(f"{o} {n}" for o, n in counts["solves"].items() if n)
        lines.append(
            f"    {role} solves: {solves}; hours {counts['hours']}, by the fall-back "
            f"{counts['fallback_hours']}"
        )
    return lines


def summary_lines(document: dict) -> list[str]:
    """The summary text of a results document (module text)."""
    lines = [
        document["label"],
        document["scope"],
        f"Method {document['method']} against reference {document['reference']}; model "
        f"{document.get('model', '-')}, dial {document['dial']:g}; scale: {document['scale_rule']}",
        f"Days: {document['days_rule']}",
        "",
    ]
    for name, unit in document["units"].items():
        if unit["status"] != "calibratable":
            lines.append(f"{name}: {unit['status']}, not diagnosed")
            continue
        lines.append(f"{name}:")
        for variant, entry in unit["variants"].items():
            lines += _variant_lines(variant, entry)
    if document["failed"]:
        lines += ["", "FAILED:"] + [f"  {f}" for f in document["failed"]]
    return lines
