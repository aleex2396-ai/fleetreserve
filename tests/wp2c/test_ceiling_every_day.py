"""The every-day option of the ceiling report (`ceiling_report --every-day`, plan section 11, O2):
the block product on every counting day of a unit with at most 8 service hours (the M days), not
a block sample.

1. The figures of one unit (`ceiling_summary.o2_summary`) on made-up days, each against a hand
   calculation: V_caus / V_hind mean and median over the M days (a day with V_hind = 0 left out
   and counted), the days with an in-window departure (their count, share of M and ratio), the
   count K of days with V_aff = V_caus, the days outside M, and O2's pass.
2. O2's pass per unit: passed, failed (V_caus differs from V_hind on a day meeting T4's
   hypotheses), could not run (such a day with a missing value, or an M day that could not be
   built, whatever the unit's other days and whatever the error's text, an empty one included),
   no day to check (no M day meeting the hypotheses, every M day built: a unit with no counting
   day, or no M day, included); the verdict over the units judged on the units with a day to
   check (the owner's decision of 19 November 2026), "no day to check" itself when no unit has
   one, never could not run. End to end on three made-up units, one with
   no day to check; the results file's `no_day_to_check` list names those units only, never a
   unit that could not run or failed; a unit with no M day beside a passed unit leaves the
   verdict over the units passed, in the results file and in the report. An M day not built is
   one of the command's own faults (exit 1), its error text empty or not.
3. Which days are M (`ceiling_report.every_day_tasks`): every counting day with at most 8
   service hours, whatever its values or departures; a day of 9 hours is out of M and counted.
4. End to end on the made-up development tables (`ceiling_fakes`): every counting day is in M,
   each unit's figures in `ceiling-report.json` equal the same figures computed by hand from the
   rows of `o2-days.csv`, O2's pass holds on every unit (exit 0).
5. A timeout is a recorded outcome: with every tree solve ending in "timeout" (the solver's
   answer replaced, the time limit kept), V_caus is missing on every day (never zero, an empty
   cell), counted as a timeout, O2's pass could not run on every unit and no unit is listed as
   having no day to check, in the results alone:
   the exit status is 0 (`ceiling_report` reading O2-4, the owner's decision of 19 November 2026).
   Every solve of the run has a time limit, the plan's 30 s or 300 s.
6. `--seed` with `--every-day` could not run: the option draws no sample.
7. The command's own faults on a unit's days (`ceiling_report` reading O2-6, decision 85 (1) of
   20 November 2026): O2's pass is changed only by faults on the unit's M days. A broken
   block-product order on an M day makes it failed (whatever else); an M day not built makes a
   pass other than failed could not run. An hourly-product order broken on any day, a day
   outside M not built: counted in the unit's `faults`, the report's last line "CHECKS FAILED",
   never in the pass (an hourly break on an M day leaves the unit passed; a day outside M not
   built leaves it passed, or "no day to check" with no M day meeting T4's hypotheses, and the
   verdict over such units "no day to check", never could not run). Every fault counted in
   `faults`, the block breaks on M days apart. On the made-up development tables a broken block
   order gives that verdict and still exit 1 (reading O2-4: the exit 0 is the held-out
   command's).
"""

import csv
import json
import statistics
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from ceiling_fakes import setup_build

from fleetreserve import ceiling_report, exact, sparse_programme
from fleetreserve.ceiling_summary import (
    FAILED,
    NO_DAY,
    NOT_RUN,
    PASSED,
    faults,
    no_day_units,
    o2_pass,
    o2_summary,
    pass_over,
    t4_check,
)
from fleetreserve.ceiling_values import BLOCK_KEYS, VALUE_KEYS, DayResult
from fleetreserve.lp import BOX_SECONDS, OTHER_SECONDS, LpResult


def _day(date, n_service, in_window, hypotheses, block, in_m=True, outcome="optimal"):
    """A made-up day: `block` gives (V_hind, V_caus, V_aff, V_diag), None for a missing value."""
    values = dict.fromkeys(VALUE_KEYS)
    outcomes = dict.fromkeys(VALUE_KEYS, "not solved")
    values["hourly_hind"], values["hourly_causal"] = 1.0, 1.0
    outcomes["hourly_hind"], outcomes["hourly_causal"] = "exact", "optimal"
    if in_m:
        values["hourly_tree"], outcomes["hourly_tree"] = 1.0, "optimal"
        for key, value in zip(BLOCK_KEYS, block, strict=True):
            values[key] = value
            outcomes[key] = "optimal" if value is not None else outcome
    return DayResult(
        "u", "A", date, 3, n_service, 2, in_window, hypotheses, in_m, None, values, outcomes
    )


# d1 to d4 are M days (at most 8 service hours); d5 has 9 and is out of M.
DAYS = [
    _day("d1", 3, False, True, (2.0, 2.0, 2.0, 1.0)),  # ratio 1.0, V_aff = V_caus
    _day("d2", 4, True, False, (4.0, 3.6, 2.0, 2.0)),  # ratio 0.9, V_aff < V_caus
    _day("d3", 8, True, False, (5.0, 2.5, 2.5, 2.0)),  # ratio 0.5, V_aff = V_caus
    _day("d4", 6, False, False, (0.0, 0.0, 0.0, 0.0)),  # V_hind = 0: no ratio; V_aff = V_caus
    _day("d5", 9, False, True, (), in_m=False),  # out of M; meets the hypotheses, not checked
]


# --- 1. the figures against a hand calculation ---------------------------------------------------


def test_the_figures_of_one_unit_equal_the_hand_calculation():
    found = o2_summary(DAYS)
    assert (found["counting_days"], found["days_in_m"], found["days_outside_m"]) == (5, 4, 1)
    assert found["hypotheses_days_outside_m"] == 1
    ratio = found["ratio_caus_over_hind"]
    # ratios 2/2, 3.6/4, 2.5/5 = 1.0, 0.9, 0.5: mean 2.4 / 3 = 0.8, median 0.9
    assert (ratio["days"], ratio["days_in_ratio"], ratio["days_with_zero_hindsight"]) == (4, 3, 1)
    assert ratio["days_with_a_missing_value"] == 0
    assert (ratio["mean"], ratio["median"], ratio["min"]) == (0.8, 0.9, 0.5)
    inside = found["in_window_departure"]
    # d2 and d3 of the four M days: share 0.5; their ratios 0.9, 0.5: mean and median 0.7
    assert (inside["days"], inside["share_of_m"]) == (2, 0.5)
    assert (inside["ratio_caus_over_hind"]["mean"], inside["ratio_caus_over_hind"]["median"]) == (
        0.7,
        0.7,
    )
    outside = found["no_in_window_departure"]
    # d1 (ratio 1.0) and d4 (no ratio)
    assert outside["days"] == 2
    assert outside["ratio_caus_over_hind"]["days_in_ratio"] == 1
    assert outside["ratio_caus_over_hind"]["mean"] == 1.0
    # K: d1 (2 = 2), d3 (2.5 = 2.5), d4 (0 = 0); not d2 (2 < 3.6)
    assert found["aff_equals_caus"] == {"days": 3, "of_days": 4, "days_with_a_missing_value": 0}
    # T4's hypotheses on M: d1 alone, V_hind = V_caus there
    check = found["t4_check"]
    assert (check["days_meeting_hypotheses"], check["days_checked"], check["violations"]) == (
        1,
        1,
        [],
    )
    assert found["o2_pass"] == PASSED


def test_a_missing_value_is_left_out_and_counted_never_read_as_zero():
    days = [*DAYS[:3], _day("d6", 5, True, False, (3.0, None, None, 1.0), outcome="timeout")]
    found = o2_summary(days)
    ratio = found["ratio_caus_over_hind"]
    assert (ratio["days_in_ratio"], ratio["days_with_a_missing_value"]) == (3, 1)
    assert ratio["mean"] == 0.8  # as before: d6 is not a ratio of 0
    assert found["in_window_departure"]["days"] == 3  # d2, d3, d6
    assert found["in_window_departure"]["ratio_caus_over_hind"]["days_with_a_missing_value"] == 1
    assert found["aff_equals_caus"] == {"days": 2, "of_days": 3, "days_with_a_missing_value": 1}
    assert found["missing"]["block_caus"] == {"asked": 4, "timeout": 1, "other": 0}


# --- 2. O2's pass ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("days", "verdict"),
    [
        ([DAYS[0]], PASSED),
        ([DAYS[0], _day("v", 3, False, True, (2.0, 1.5, 1.5, 1.0))], FAILED),
        ([DAYS[0], _day("t", 3, False, True, (2.0, None, None, 1.0), outcome="timeout")], NOT_RUN),
        ([_day("t", 3, False, True, (2.0, None, None, 1.0), outcome="timeout")], NOT_RUN),
        (DAYS[1:4], NO_DAY),  # no M day meets the hypotheses: no day to check, never a pass
        ([], NO_DAY),
        (
            [
                _day("v", 3, False, True, (2.0, 1.5, 1.5, 1.0)),
                _day("t", 3, False, True, (2.0, None, None, 1.0), outcome="timeout"),
            ],
            FAILED,
        ),
    ],
    ids=[
        "equal",
        "a violation",
        "a timeout beside an equal day",
        "only a timeout",
        "no hypotheses day",
        "no day",
        "a violation beside a timeout",
    ],
)
def test_o2s_pass_is_passed_failed_or_could_not_run(days, verdict):
    assert o2_pass(t4_check(days), 0) == verdict  # every day here an M day, built
    assert o2_summary(days)["o2_pass"] == verdict


def _not_built(date, hypotheses, error="the sets cannot be built"):
    """A made-up M day that could not be built: no value, its `error` set."""
    built = _day(date, 3, False, hypotheses, (None, None, None, None), outcome="not solved")
    return replace(built, error=error)


@pytest.mark.parametrize(
    ("days", "verdict"),
    [
        ([*DAYS[1:4], _not_built("e", False)], NOT_RUN),
        ([_not_built("e", False)], NOT_RUN),
        ([*DAYS[1:4], _not_built("e", False, error="no day to check")], NOT_RUN),
        ([_not_built("e", True)], NOT_RUN),
        ([DAYS[0], _not_built("e", False)], NOT_RUN),
        ([DAYS[0], _not_built("e", False, error="")], NOT_RUN),
        (
            [DAYS[0], _day("v", 3, False, True, (2.0, 1.5, 1.5, 1.0)), _not_built("e", False)],
            FAILED,
        ),
        ([*DAYS[1:4], DAYS[4]], NO_DAY),
        ([DAYS[4]], NO_DAY),
    ],
    ids=[
        "an M day not built beside days with no hypotheses",
        "only an M day not built",
        "an M day not built whose error text says no day to check",
        "a hypotheses day not built",
        "an M day not built beside an equal hypotheses day",
        "an M day not built with an empty error text beside an equal hypotheses day",
        "an M day not built beside a violation",
        "a hypotheses day outside M only",
        "no M day, a hypotheses day outside M",
    ],
)
def test_no_day_to_check_is_decided_from_day_counts_alone(days, verdict):
    """A unit has no day to check (`ceiling_summary` reading N) with every M day built, none
    meeting T4's hypotheses, whether it has M days or none; an M day that could not be built,
    whatever its error text and whatever the unit's other days, leaves the unit could not run
    (failed stays failed), and a hypotheses day outside M makes no unit checkable."""
    found = o2_summary(days)
    assert found["o2_pass"] == verdict
    assert found["days_in_m_not_built"] == sum(d.error is not None and d.sampled for d in days)


def test_an_m_day_not_built_with_an_empty_error_text_is_a_fault_and_the_command_exits_1():
    """`faults` names every day whose `error` is set, its text empty or not, as `o2_summary`
    counts it; with `--every-day` on development units a fault alone gives exit 1
    (`ceiling_report.exit_status`, reading O2-4)."""
    for error in ("", "the sets cannot be built"):
        days = [DAYS[0], _not_built("e", False, error=error)]
        found = faults(days)
        assert len(found) == 1 and found[0].startswith("u A e: the day could not be built:"), error
        assert o2_summary(days)["days_in_m_not_built"] == 1
        assert ceiling_report.exit_status([], {}, found, held_out=False) == 1
    assert faults(DAYS[:4]) == []
    assert ceiling_report.exit_status([], {}, faults(DAYS[:4]), held_out=False) == 0


def _named(name, days):
    return [replace(d, unit=name) for d in days]


_TIMEOUT = _day("t", 3, False, True, (2.0, None, None, 1.0), outcome="timeout")
_VIOLATION = _day("v", 3, False, True, (2.0, 1.5, 1.5, 1.0))


@pytest.mark.parametrize(
    ("units", "passes", "over", "no_day"),
    [
        (
            {"a": DAYS[1:4], "b": [DAYS[0], _TIMEOUT], "c": DAYS[:1]},
            [NO_DAY, NOT_RUN, PASSED],
            NOT_RUN,
            ["a"],
        ),
        ({"a": DAYS[1:4], "b": [DAYS[0], _VIOLATION]}, [NO_DAY, FAILED], FAILED, ["a"]),
        (
            {"a": DAYS[1:4], "b": [DAYS[0], _not_built("e", False)], "c": DAYS[:1]},
            [NO_DAY, NOT_RUN, PASSED],
            NOT_RUN,
            ["a"],
        ),
        (
            {"a": [_TIMEOUT], "b": [_not_built("e", False, error="")]},
            [NOT_RUN, NOT_RUN],
            NOT_RUN,
            [],
        ),
        ({"a": DAYS[1:4], "b": DAYS[:1], "c": DAYS[:2]}, [NO_DAY, PASSED, PASSED], PASSED, ["a"]),
    ],
    ids=[
        "no day, a timeout on a hypotheses day, passed",
        "no day, failed",
        "no day, an M day not built beside an equal day, passed",
        "a timeout, an M day not built with an empty error text",
        "no day, passed, passed",
    ],
)
def test_the_results_files_no_day_list_names_only_the_units_with_no_day_to_check(
    units, passes, over, no_day
):
    """`ceiling_report.every_day_summary`: a unit that could not run or failed is never listed
    as having no day to check, beside a unit that is."""
    results = [d for name, days in units.items() for d in _named(name, days)]
    found = ceiling_report.every_day_summary(results, list(units), {})
    assert [found["units"][name]["o2_pass"] for name in units] == passes
    assert found["pass_over_units"] == over
    assert found["no_day_to_check"] == no_day


_OUTSIDE_M = [
    _day("o1", 9, False, True, (), in_m=False),  # meets T4's hypotheses, outside M: not checked
    _day("o2", 10, True, False, (), in_m=False),
    _day("o3", 12, False, False, (), in_m=False),
]


@pytest.mark.parametrize(
    ("units", "passes", "over", "no_day", "last"),
    [
        (
            {"a": [], "b": DAYS[:1]},
            [NO_DAY, PASSED],
            PASSED,
            ["a"],
            "all checks held (no day to check on a)",
        ),
        (
            {"a": _OUTSIDE_M, "b": DAYS[:1]},
            [NO_DAY, PASSED],
            PASSED,
            ["a"],
            "all checks held (no day to check on a)",
        ),
        (
            {"a": _OUTSIDE_M[1:], "b": DAYS[:1]},
            [NO_DAY, PASSED],
            PASSED,
            ["a"],
            "all checks held (no day to check on a)",
        ),
        (
            {"a": [], "b": _OUTSIDE_M, "c": DAYS[1:4]},
            [NO_DAY, NO_DAY, NO_DAY],
            NO_DAY,
            ["a", "b", "c"],
            "O2's pass over the units: no day to check (no day to check on a, b, c)",
        ),
        (
            {"a": DAYS[1:4], "b": [*DAYS[1:3], DAYS[4]]},
            [NO_DAY, NO_DAY],
            NO_DAY,
            ["a", "b"],
            "O2's pass over the units: no day to check (no day to check on a, b)",
        ),
        (
            {"a": DAYS[1:4], "b": [DAYS[0], _TIMEOUT]},
            [NO_DAY, NOT_RUN],
            NOT_RUN,
            ["a"],
            "O2's pass could not run on b",
        ),
        ({}, [], NOT_RUN, [], "O2's pass over the units: could not run"),
        (
            {"a": [], "b": [DAYS[0], _VIOLATION]},
            [NO_DAY, FAILED],
            FAILED,
            ["a"],
            "O2's pass failed on b (reading O2-4: in the results alone)",
        ),
        (
            {"a": [_not_built("e", False)], "b": DAYS[:1]},
            [NOT_RUN, PASSED],
            NOT_RUN,
            [],
            "O2's pass could not run on a",
        ),
        (
            {"a": [*_OUTSIDE_M, _not_built("e", False)], "b": DAYS[:1]},
            [NOT_RUN, PASSED],
            NOT_RUN,
            [],
            "O2's pass could not run on a",
        ),
    ],
    ids=[
        "no counting day, passed",
        "counting days none in M, passed",
        "counting days none in M nor meeting T4, passed",
        "every unit with no day to check",
        "every unit with M days none meeting T4",
        "no day to check beside a timeout on a hypotheses day",
        "no unit given",
        "no counting day, failed",
        "only an M day not built, passed",
        "an M day not built beside days outside M, passed",
    ],
)
def test_a_unit_with_no_m_day_has_no_day_to_check_and_the_verdict_is_judged_on_the_others(
    units, passes, over, no_day, last
):
    """The owner's decision of 19 November 2026 (decision 79 (3)): a unit with no day meeting T4's
    hypotheses, among them a unit with no counting day and a unit whose counting days are all
    outside M, is listed as having no day to check beside the verdict over the units, never
    turning it into could not run, and with every unit so the verdict over the units is no day to
    check; an M day that could not be built, or a hypotheses day with a missing value, still
    leaves its unit, and the verdict over the units, could not run, as does no unit given. Through
    `ceiling_report.every_day_summary` and the report's text."""
    results = [d for name, days in units.items() for d in _named(name, days)]
    record = {"max_service_hours": 8, "reading": "made up"}
    every = ceiling_report.every_day_summary(results, list(units), record)
    assert [every["units"][name]["o2_pass"] for name in units] == passes
    assert every["pass_over_units"] == over
    assert every["no_day_to_check"] == no_day
    document = {"inputs": {"truth_run": "t"}, "rho": 0.0, "every_day": every, "hourly": {}}
    lines = ceiling_report.every_day_lines(document, [], [])
    verdicts = ", ".join(f"{name} {verdict}" for name, verdict in zip(units, passes, strict=True))
    beside = f" (no day to check on {', '.join(no_day)})" if no_day else ""
    assert f"  O2's pass per unit: {verdicts}; over the units: {over}{beside}" in lines
    assert lines[-1] == last


@pytest.mark.parametrize(
    ("verdicts", "over"),
    [
        ([NO_DAY, PASSED, PASSED], PASSED),
        ([NO_DAY, FAILED, PASSED], FAILED),
        ([NO_DAY, NOT_RUN, PASSED], NOT_RUN),
        ([NO_DAY, NO_DAY, NO_DAY], NO_DAY),
        ([NO_DAY], NO_DAY),
        ([], NOT_RUN),
        ([PASSED, PASSED], PASSED),
        ([PASSED, NOT_RUN], NOT_RUN),
    ],
    ids=[
        "one unit with no day beside two passed",
        "one with no day beside one failed",
        "one with no day beside one could not run",
        "every unit with no day",
        "one unit with no day",
        "no unit",
        "two passed",
        "one passed, one could not run",
    ],
)
def test_the_verdict_over_the_units_is_judged_on_the_units_with_a_day_to_check(verdicts, over):
    """The owner's decision of 19 November 2026 (option (b)): a unit with no day to check counts
    as neither passed nor failed; with every given unit no day to check the verdict is no day to
    check, never could not run (decision 79 (3)); with no unit given at all (an empty input, not a
    unit with no day) it could not run."""
    assert pass_over(verdicts) == over
    names = "abc"[: len(verdicts)]
    assert no_day_units(dict(zip(names, verdicts, strict=True))) == [
        name for name, v in zip(names, verdicts, strict=True) if v == NO_DAY
    ]


# --- 3. which days are M --------------------------------------------------------------------------


def _unit(hours_and_departures):
    """A made-up unit's counting days: one per (service hours, in-window departure)."""
    days = []
    for k, (hours, departs) in enumerate(hours_and_departures):
        record = SimpleNamespace(site="A", date=f"2020-01-{k + 1:02d}")
        day = SimpleNamespace(n_service=hours, has_in_window_departure=departs, cars=())
        days.append(SimpleNamespace(record=record, day=day))
    return SimpleNamespace(days=days)


def test_m_is_every_counting_day_with_at_most_8_service_hours_whatever_else():
    layout = [(1, False), (8, True), (9, False), (12, True), (5, True), (8, False)]
    tasks, record = ceiling_report.every_day_tasks({"u": _unit(layout)})
    assert [task.date for task in tasks] == [f"2020-01-{k:02d}" for k in range(1, 7)]
    assert [task.sampled for task in tasks] == [True, True, False, False, True, True]
    assert record["counting_days"] == {"u": 6}
    assert record["days_in_m"] == {"u": 4}
    assert record["max_service_hours"] == 8


# --- 4. end to end on the made-up development tables ---------------------------------------------


def _run(run, tables, out, patch, *extra):
    out.mkdir(exist_ok=True)
    patch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return ceiling_report.main(
        ["--run", str(run), "--tables", str(tables), "--workers", "1", "--every-day", *extra]
    )


def _rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def every_day(tmp_path_factory):
    root = tmp_path_factory.mktemp("every-day")
    with pytest.MonkeyPatch.context() as patch:
        run, tables = setup_build(root, patch)
        code = _run(run, tables, root / "out", patch)
    return code, root / "out"


def _by_hand(rows):
    """The figures of one unit from the rows of `o2-days.csv`, written out (module text, 4)."""
    m = [r for r in rows if int(r["service_hours"]) <= 8]
    ratios = [
        float(r["block_v_caus_kwh"]) / float(r["block_v_hind_kwh"])
        for r in m
        if r["block_v_caus_kwh"] and float(r["block_v_hind_kwh"]) > 1e-7
    ]
    inside = [r for r in m if r["in_window_departure"] == "True"]
    equal = [
        r
        for r in m
        if abs(float(r["block_v_aff_kwh"]) - float(r["block_v_caus_kwh"]))
        <= 1e-6 * max(1.0, float(r["block_v_aff_kwh"]), float(r["block_v_caus_kwh"]))
    ]
    hypotheses = [r for r in m if r["meets_t4_hypotheses"] == "True"]
    return {
        "days_in_m": len(m),
        "mean": round(statistics.fmean(ratios), 6),
        "median": round(statistics.median(ratios), 6),
        "in_window": len(inside),
        "share": round(len(inside) / len(m), 6),
        "k": len(equal),
        "hypotheses": len(hypotheses),
    }


def test_end_to_end_every_counting_day_is_in_m_and_the_figures_match_the_rows(every_day):
    code, out = every_day
    assert code == 0
    document = json.loads((out / ceiling_report.RESULTS_NAME).read_text())
    assert "block_sample" not in document and "sample" not in document
    rows = _rows(out / ceiling_report.EVERY_DAY_NAME)
    assert len(rows) == 75  # 20 + 25 + 30 counting days, six service hours each
    assert not (out / ceiling_report.SAMPLE_NAME).exists()
    days = _rows(out / ceiling_report.DAYS_NAME)
    assert len(days) == 75 and all(r["in_o2_days"] == "True" for r in days)
    units = document["every_day"]["units"]
    assert sorted(units) == sorted(ceiling_report.UNITS)  # the results file sorts its keys
    for unit, found in units.items():
        hand = _by_hand([r for r in rows if r["unit"] == unit])
        assert found["days_in_m"] == found["counting_days"] == hand["days_in_m"]
        ratio = found["ratio_caus_over_hind"]
        assert (ratio["mean"], ratio["median"]) == (hand["mean"], hand["median"]), unit
        assert found["in_window_departure"]["days"] == hand["in_window"]
        assert found["in_window_departure"]["share_of_m"] == hand["share"]
        assert found["aff_equals_caus"]["days"] == hand["k"]
        assert found["t4_check"]["days_meeting_hypotheses"] == hand["hypotheses"] > 0
        assert found["o2_pass"] == PASSED
    assert document["every_day"]["pass_over_units"] == PASSED
    text = (out / ceiling_report.TEXT_NAME).read_text()
    assert "(a) O2 ON EVERY COUNTING DAY WITH AT MOST 8 SERVICE HOURS" in text
    assert text.endswith("all checks held\n")


# --- 5. a timeout is a recorded outcome; every solve has its time limit --------------------------


def test_a_tree_timeout_is_missing_counted_and_o2s_pass_could_not_run(tmp_path, monkeypatch):
    run, tables = setup_build(tmp_path, monkeypatch)
    limits = []
    real_exact, real_sparse = exact.solve_lp, sparse_programme.solve_lp

    def watched_exact(*args, time_limit, **kwargs):
        limits.append(time_limit)
        return real_exact(*args, time_limit=time_limit, **kwargs)

    def tree_times_out(*args, time_limit, **kwargs):
        limits.append(time_limit)
        if time_limit == OTHER_SECONDS:  # the corner tree's limit; the certificates' is 30 s
            return LpResult("timeout", None, None, time_limit, time_limit, 1, "made-up timeout")
        return real_sparse(*args, time_limit=time_limit, **kwargs)

    monkeypatch.setattr(exact, "solve_lp", watched_exact)
    monkeypatch.setattr(sparse_programme, "solve_lp", tree_times_out)
    assert _run(run, tables, tmp_path / "out", monkeypatch) == 0
    assert limits and set(limits) <= {BOX_SECONDS, OTHER_SECONDS}
    assert OTHER_SECONDS in limits and all(np.isfinite(limits))
    out = tmp_path / "out"
    document = json.loads((out / ceiling_report.RESULTS_NAME).read_text())
    for unit, found in document["every_day"]["units"].items():
        m = found["days_in_m"]
        empty = found["empty_fleet_days"]
        assert found["missing"]["block_caus"] == {"asked": m, "timeout": m - empty, "other": 0}
        assert found["ratio_caus_over_hind"]["days_with_a_missing_value"] == m - empty
        assert found["t4_check"]["days_checked"] == 0 < found["t4_check"]["days_meeting_hypotheses"]
        assert found["o2_pass"] == NOT_RUN, unit
    assert document["every_day"]["pass_over_units"] == NOT_RUN
    assert document["every_day"]["no_day_to_check"] == []
    rows = [r for r in _rows(out / ceiling_report.EVERY_DAY_NAME) if r["cars"] != "0"]
    assert rows and all(r["block_v_caus_kwh"] == "" for r in rows)
    assert {r["block_caus_outcome"] for r in rows} == {"timeout"}
    text = (out / ceiling_report.TEXT_NAME).read_text()
    assert text.endswith(f"O2's pass could not run on {', '.join(ceiling_report.UNITS)}\n")


# --- 6. refused -----------------------------------------------------------------------------------


def test_a_seed_with_every_day_could_not_run(tmp_path, monkeypatch, capsys):
    run, tables = setup_build(tmp_path, monkeypatch)
    assert _run(run, tables, tmp_path / "out", monkeypatch, "--seed", "7") == 2
    shown = capsys.readouterr().out
    assert shown.startswith("COULD NOT RUN: --seed draws the block sample")
    assert shown.count("\n") == 1
    assert not list((tmp_path / "out").iterdir())


# --- 7. the command's own faults ------------------------------------------------------------------


def _hourly_break(day: DayResult) -> DayResult:
    """`day` with its best causal hourly value 2 above its hourly ceiling 1 (a broken order)."""
    return replace(day, values={**day.values, "hourly_causal": 2.0})


EQUAL = _day("e", 3, False, True, (2.0, 2.0, 2.0, 1.0))  # V_hind = V_caus on the hypotheses
VIOLATION = _day("v", 3, False, True, (2.0, 1.5, 1.5, 1.0))  # V_hind differs from V_caus


HOURLY_M = _hourly_break(_day("h", 3, False, True, (2.0, 2.0, 2.0, 1.0)))  # an M day: 2 breaks
BLOCK_M = _day("b", 3, False, False, (2.0, 2.0, 2.5, 1.0))  # V_aff 2.5 above V_caus 2 on an M day
OUT_NOT_BUILT = replace(DAYS[4], error="made-up")  # a day outside M that could not be built


@pytest.mark.parametrize(
    ("days", "verdict", "own"),
    [
        ([EQUAL], PASSED, (0, 0)),
        ([EQUAL, HOURLY_M], PASSED, (2, 0)),  # (a)
        ([EQUAL, OUT_NOT_BUILT], PASSED, (0, 1)),  # (c)
        ([EQUAL, _hourly_break(DAYS[4])], PASSED, (1, 0)),
        ([replace(EQUAL, error="made-up"), _hourly_break(DAYS[4])], NOT_RUN, (1, 1)),
    ],
    ids=[
        "no fault",
        "an hourly order broken on an M day",
        "a day out of M not built",
        "an hourly order broken out of M",
        "an M day not built beside an hourly break out of M",
    ],
)
def test_faults_outside_m_days_block_orders_are_counted_never_in_the_pass(days, verdict, own):
    """Decision 85 (1): an hourly break (on an M day or not) and a day outside M not built are
    counted in `faults`, O2's pass as without them. At c22e5c1 every case but the first was
    failed or could not run."""
    every = ceiling_report.every_day_summary(days, ["u"], {})
    found = every["units"]["u"]
    assert found["o2_pass"] == verdict
    assert every["pass_over_units"] == verdict
    assert (found["faults"]["order_breaks"], found["faults"]["days_not_built"]) == own
    assert found["faults"]["block_order_breaks_on_m_days"] == 0


@pytest.mark.parametrize(
    ("days", "verdict", "own"),
    [
        ([EQUAL, BLOCK_M], FAILED, (1, 0)),  # (d)
        ([BLOCK_M], FAILED, (1, 0)),  # no M day meets T4's hypotheses
        ([EQUAL, replace(DAYS[1], error="made-up")], NOT_RUN, (0, 1)),  # (e)
        ([VIOLATION, replace(EQUAL, error="made-up")], FAILED, (0, 1)),
        ([replace(EQUAL, error="made-up"), BLOCK_M], FAILED, (1, 1)),
        ([EQUAL, BLOCK_M, HOURLY_M, OUT_NOT_BUILT], FAILED, (3, 1)),
    ],
    ids=[
        "a block order broken on an M day",
        "a block order broken on a unit with no day to check",
        "an M day not built",
        "a T4 violation beside an M day not built",
        "an M day not built beside a block break",
        "a block break beside an hourly break and a day out of M not built",
    ],
)
def test_faults_on_m_days_enter_the_pass(days, verdict, own):
    """A block order broken on an M day: failed; an M day not built: could not run (failed
    stays failed). The same verdicts at c22e5c1."""
    every = ceiling_report.every_day_summary(days, ["u"], {})
    found = every["units"]["u"]
    assert found["o2_pass"] == verdict
    assert every["pass_over_units"] == verdict
    assert (found["faults"]["order_breaks"], found["faults"]["days_not_built"]) == own


def test_block_breaks_on_m_days_are_counted_apart():
    """`faults` keeps every break and every day not built; the block breaks on M days, the ones
    O2's pass counts, are given apart."""
    days = [EQUAL, BLOCK_M, HOURLY_M, OUT_NOT_BUILT]
    found = ceiling_report.every_day_summary(days, ["u"], {})["units"]["u"]
    assert found["faults"] == {
        "order_breaks": 3,
        "days_not_built": 1,
        "block_order_breaks_on_m_days": 1,
    }


def _report(units: dict) -> tuple[dict, list[str]]:
    """`every_day_summary` and the report's lines of made-up units' days (no hourly part)."""
    results = [d for name, days in units.items() for d in _named(name, days)]
    record = {"max_service_hours": 8, "reading": "made-up"}
    every = ceiling_report.every_day_summary(results, list(units), record)
    problems = ceiling_report.failures([r for r in results if r.sampled], results)
    document = {
        "inputs": {"truth_run": "made-up"},
        "rho": 0.0,
        "every_day": every,
        "hourly": {},
        "failures": problems,
    }
    return every, ceiling_report.every_day_lines(document, problems, faults(results))


def test_an_hourly_break_on_an_m_day_passes_with_checks_failed_shown():
    """(a): the unit passed, its two hourly breaks counted, each a FAILED line, and the last
    line "CHECKS FAILED". Failed at c22e5c1."""
    every, lines = _report({"u": [EQUAL, HOURLY_M]})
    assert every["units"]["u"]["o2_pass"] == PASSED
    assert every["pass_over_units"] == PASSED
    assert every["units"]["u"]["faults"]["order_breaks"] == 2
    assert sum(line.startswith("FAILED: u A h: hourly") for line in lines) == 2
    assert "  O2's pass per unit: u passed; over the units: passed" in lines
    assert any("own faults on the unit's days (reading O2-6): 2 broken orders" in x for x in lines)
    assert lines[-1] == "CHECKS FAILED"


def test_a_day_out_of_m_not_built_leaves_no_day_to_check_over_every_unit():
    """(b): two units, each with a day outside M not built and no M day meeting T4's
    hypotheses: each "no day to check", the verdict over them "no day to check" (never could
    not run), the days counted and "CHECKS FAILED" shown. Could not run at c22e5c1."""
    no_hypotheses = _day("n", 3, False, False, (2.0, 2.0, 2.0, 1.0))
    every, lines = _report({"a": [no_hypotheses, OUT_NOT_BUILT], "b": [OUT_NOT_BUILT]})
    assert [every["units"][name]["o2_pass"] for name in "ab"] == [NO_DAY, NO_DAY]
    assert every["pass_over_units"] == NO_DAY
    assert every["no_day_to_check"] == ["a", "b"]
    assert [every["units"][name]["faults"]["days_not_built"] for name in "ab"] == [1, 1]
    assert sum("the day could not be built: made-up" in line for line in lines) == 2
    assert lines[-1] == "CHECKS FAILED"


@pytest.mark.parametrize(
    ("units", "passes", "over", "no_day", "last"),
    [
        ({"a": [EQUAL], "b": [EQUAL]}, [PASSED, PASSED], PASSED, ["x"], "all checks held"),
        ({"a": [EQUAL], "b": [VIOLATION]}, [PASSED, FAILED], FAILED, ["x"], None),
        ({}, [], NO_DAY, ["x"], "O2's pass over the units: no day to check"),
    ],
    ids=["beside two passed units", "beside a failed unit", "alone"],
)
def test_an_abstained_held_out_unit_is_listed_with_no_day_to_check_never_could_not_run(
    units, passes, over, no_day, last
):
    """The owner's decision of 19 November 2026 (decision 79 (3)): a held-out unit the build
    abstained has no day, so O2 lists it as having no day to check, its reason named beside it,
    and judges the verdict over the units on the units that have a day; the abstention alone
    never makes the verdict could not run. Could not run at 0ae3958."""
    results = [d for name, days in units.items() for d in _named(name, days)]
    abstained = {"x": "no calibration day"}
    every = ceiling_report.every_day_summary(
        results, list(units), {"max_service_hours": 8, "reading": "made-up"}, abstained
    )
    assert [every["units"][name]["o2_pass"] for name in units] == passes
    assert every["units"]["x"]["o2_pass"] == NO_DAY
    assert every["units"]["x"]["reason"] == "abstained (no calibration day)"
    assert every["pass_over_units"] == over
    assert every["no_day_to_check"] == no_day
    document = {"inputs": {"truth_run": "made-up"}, "rho": 0.0, "every_day": every, "hourly": {}}
    lines = ceiling_report.every_day_lines(document, [], [])
    assert "UNIT x: abstained (no calibration day); no day to check" in lines
    assert not any("could not run on" in line for line in lines)
    if last is not None:
        assert lines[-1].startswith(last)


def test_a_broken_order_on_development_units_fails_the_pass_and_keeps_exit_1(tmp_path, monkeypatch):
    """V_aff set 1 kWh above V_caus on every M day of the made-up development tables, V_hind and
    V_caus as computed: each unit's pass is failed and the breaks are in `failures`; the exit
    status stays 1 off the sealed route (reading O2-4). Every unit passed at b1e8061."""
    run, tables = setup_build(tmp_path, monkeypatch)
    real = ceiling_report.compute_day

    def broken(task):
        found = real(task)
        if task.sampled and found.values["block_caus"] is not None:
            found.values["block_aff"] = found.values["block_caus"] + 1.0
        return found

    monkeypatch.setattr(ceiling_report, "compute_day", broken)
    assert _run(run, tables, tmp_path / "out", monkeypatch) == 1
    document = json.loads((tmp_path / "out" / ceiling_report.RESULTS_NAME).read_text())
    units = document["every_day"]["units"]
    assert {unit: found["o2_pass"] for unit, found in units.items()} == dict.fromkeys(
        ceiling_report.UNITS, FAILED
    )
    assert all(found["faults"]["order_breaks"] == found["days_in_m"] for found in units.values())
    assert sum("above block_caus" in line for line in document["failures"]) == 75
    assert document["every_day"]["pass_over_units"] == FAILED
