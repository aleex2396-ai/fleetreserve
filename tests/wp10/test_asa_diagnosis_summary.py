"""The figures of one unit and variant (`asa_diagnosis_summary.variant_figures`) from days'
diagnoses made up by hand, and the summary's lines.

Seven made-up days, in this order: day 0 with 2 method-only hours (state, choice by goal "share"),
day 1 with none, days 2 to 6 with 3, 1, 3, 2 and 1 hours. The state hours: one with a car that
left earlier than believed and 0.5 kWh charged ahead, one with two cars that left earlier, one
with 1e-5 kWh charged ahead (below the tolerance, so not ahead).

1. The groups and causes are the sums over the days; the shares are over the method-only hours.
2. The choice goals count each goal; the deficits are summed over the belief hours only.
3. The state block: 3 state hours, 2 with an early leaver, 1 charged ahead (1e-5 is not).
4. The top days: five, the most first, ties in the order diagnosed (days 2 and 4 with 3, then
   days 0 and 5 with 2, then day 3 with 1; day 6 with 1 too is the sixth and left out; day 1
   with none is never listed), each with its causes.
5. The solver counts are summed per role; a role without counts is left out. The goal-test
   outcomes are summed over the days.
6. With no method-only hour every share is null and the state shares are null.
7. The summary lines open with the label and the scope, then name the counts, the shares, the
   belief deficits and the top days; with no belief or state hour those lines are left out, and
   a solve outcome counted 0 is left out of the solves line.
"""

from collections import Counter

import pytest

from fleetreserve.asa_diagnosis_day import CAUSES, DayDiagnosis, HourRow
from fleetreserve.asa_diagnosis_summary import SCOPE, summary_lines, variant_figures


def _row(cause, goal="", early=0, ahead=0.0, cap=0.0):
    return HourRow(
        path=0,
        hour=1,
        request=1.0,
        method_delivered=0.0,
        reference_delivered=1.0,
        shortfall=1.0,
        method_first_failure=True,
        method_low=0.0,
        method_high=1.0,
        reference_low=0.0,
        reference_high=1.0,
        believed_most=1.0,
        believed_on_true=1.0,
        deficit_plugged=0.0,
        deficit_departure=0.25 if cause == "choice" else 0.0,
        deficit_rate=0.0,
        deficit_cap=cap,
        overplanned=0.0,
        cars_left_earlier=early,
        charged_ahead=ahead,
        cause=cause,
        goal=goal,
    )


def _day(rows, extra=0, counts=None, tests=()):
    found = DayDiagnosis()
    found.rows = list(rows)
    found.groups = Counter(method_only=len(rows), neither=extra, both=1)
    found.method_solver_counts = counts
    found.goal_tests = Counter(tests)
    return found


COUNTS = {"solves": {"optimal": 4, "timeout": 1}, "hours": 3, "fallback_hours": 1}
OUTS = ["optimal"] * 4 + ["infeasible"]
DAYS = [
    ("d0", _day([_row("state", early=1, ahead=0.5), _row("choice", "share")], 3, COUNTS, OUTS)),
    ("d1", _day([], 5, COUNTS)),
    ("d2", _day([_row("state", early=2), _row("belief", cap=0.75), _row("choice", "need")])),
    ("d3", _day([_row("belief", cap=0.5)])),
    ("d4", _day([_row("state", ahead=1e-5), _row("fallback"), _row("other")])),
    ("d5", _day([_row("choice", "share"), _row("choice", "tie")], tests=["optimal"] * 9)),
    ("d6", _day([_row("choice", "share")])),
]


def test_the_figures_of_made_up_days():
    figures = variant_figures(DAYS)
    assert figures["groups"] == {"method_only": 12, "reference_only": 0, "both": 7, "neither": 8}
    assert figures["causes"] == {"state": 3, "fallback": 1, "belief": 2, "choice": 5, "other": 1}
    assert figures["cause_shares"] == {c: figures["causes"][c] / 12 for c in CAUSES}
    assert figures["choice_goals"] == {"need": 1, "share": 3, "tie": 1}
    assert figures["belief_deficits_kwh"] == {
        "plugged": 0.0,
        "departure": 0.0,
        "rate": 0.0,
        "cap": 1.25,
    }
    assert figures["state"] == {
        "state_hours": 3,
        "left_earlier_hours": 2,
        "left_earlier_share": pytest.approx(2 / 3),
        "charged_ahead_hours": 1,
        "charged_ahead_share": pytest.approx(1 / 3),
    }
    top = [(t["day"], t["method_only_hours"]) for t in figures["top_days"]]
    assert top == [("d2", 3), ("d4", 3), ("d0", 2), ("d5", 2), ("d3", 1)]
    assert figures["top_days"][1]["causes"] == {
        "state": 1,
        "fallback": 1,
        "belief": 0,
        "choice": 0,
        "other": 1,
    }
    assert figures["goal_tests"] == {"infeasible": 1, "optimal": 13}
    assert figures["solver_counts"] == {
        "method": {"solves": {"optimal": 8, "timeout": 2}, "hours": 6, "fallback_hours": 2}
    }


def test_no_method_only_hour_gives_null_shares():
    figures = variant_figures([("d1", _day([], 5))])
    assert set(figures["cause_shares"].values()) == {None}
    assert figures["state"]["left_earlier_share"] is None
    assert figures["state"]["charged_ahead_share"] is None
    assert figures["top_days"] == [] and figures["solver_counts"] == {}


def test_the_summary_lines():
    entry = {
        "configuration": "made-up",
        "status": "diagnosed",
        "scale": 0.5,
        "scale_source": "--scale",
        "days_diagnosed": 7,
        "offering_days": 9,
        "figures": variant_figures(DAYS),
    }
    document = {
        "label": "made-up label",
        "scope": SCOPE,
        "method": "asa-es",
        "reference": "llf",
        "model": "empirical",
        "dial": 0.03,
        "scale_rule": "made-up rule",
        "days_rule": "made-up days",
        "units": {"u": {"status": "calibratable", "variants": {"forecast": entry}}},
        "failed": [],
    }
    lines = summary_lines(document)
    assert lines[:2] == ["made-up label", SCOPE]
    assert lines[2:4] == [
        "Method asa-es against reference llf; model empirical, dial 0.03; scale: made-up rule",
        "Days: made-up days",
    ]
    text = "\n".join(lines)
    assert "scale 0.5 (--scale), 7 days of 9 offering" in text
    assert "method only 12, reference only 0, both 7, neither 8" in text
    assert "state 3 (0.250)" in text and "choice 5 (0.417)" in text
    assert "'choice' hours by goal: need 1, share 3, tie 1" in text
    assert "'state' hours: 3; a car left earlier than believed in 2 (0.667)" in text
    assert "ahead of the reference in 1 (0.333)" in text
    assert "      d2: 3 (state 1, belief 1, choice 1)" in text
    assert "method solves: optimal 8, timeout 2; hours 6, by the fall-back 2" in text
    deficits = "plugged 0.000, departure 0.000, rate 0.000, cap 1.250"
    assert f"    'belief' hours, deficits by label (kWh): {deficits}" in lines
    # With choice hours but no belief or state hour, neither of those lines is printed, and a
    # solve outcome counted 0 is left out of the solves line.
    counts = {"solves": {"optimal": 3, "timeout": 0}, "hours": 3, "fallback_hours": 0}
    entry["figures"] = variant_figures([("d7", _day([_row("choice", "share")], 2, counts))])
    text = "\n".join(summary_lines(document))
    assert "'choice' hours by goal: share 1" in text
    assert "'belief' hours" not in text and "'state' hours" not in text
    assert "method solves: optimal 3; hours 3, by the fall-back 0" in text
