"""The figures of `corner_summary` worked out by hand on made-up `CornerDay`s and checks.

Five searched days of one method at scale 0.4: day A breaks on 3 corners (worst slot loss 0.5,
worst driver loss 0.1; the audited rows see it), day B breaks on 1 corner by driver loss alone
(0.3; the audited rows hold), day C holds with one corner short below the tolerance, day D timed
out, day E breaks on 1 corner and its audited rows timed out. So 4 days have a result, 3 break:
share 3/4; by slot loss 2, with a driver shortfall 2; held but short below the tolerance 1; mean
breaking corners (3 + 1 + 0 + 1) / 4 = 5/4; worst slot loss 0.5, worst driver loss 0.3 (each its
own maximum, not a mean, over the days with a result: day D's corners without a timeout lose more
and are not counted); one breaking day of three missed by the audited rows (day E's timed-out
audited rows are counted apart, not as a miss). The solver counts are summed over the days that
have them. At scale 0 the same days are "no offer" with no break figures. A day that breaks and
also has a corner short below the tolerance (day F) is not among the days that hold short only
below it.

Pass mark: two days at two scales, one pair differing "within tolerance" (it agrees, not
strictly), one pair differing by "decision-hour cap" (a disagreement), one driver maximum above
its bound by 0.2 with a car short of its plan and no failing hour, and two pairs within their
bounds that have those flags too (they are not counted: the flags are counted over the pairs
above the bound); nothing checked is not agreed. The largest excess and the largest slot difference
are maxima over every pair (0.0 when every pair agrees strictly), and a disagreement is written
with every field.

The CSV writes the worst corner with + for the high end and - for the low end, in hour order; a
row is written in full (a float as its repr, an absent value empty, the re-solves in the order
counted, the solves sorted by outcome, no audited count on a day with no corner run). The whole
text summary is compared line by line on a document that reaches every branch of it.
"""

import pytest

from fleetreserve import corner_summary
from fleetreserve.corner_search import ClosedFormCheck, CornerDay

COUNTS = {"solves": {"optimal": 6, "timeout": 1}, "hours": 4, "fallback_hours": 1}


def _day(status, breaking, slot, driver, below, worst_slot, worst_driver, audited, **extra):
    return CornerDay(
        "m",
        0.4,
        status,
        4,
        breaking,
        slot,
        driver,
        below,
        4 if status == "timed out" else 0,
        worst_slot,
        worst_driver,
        extra.get("worst_path"),
        None,
        audited,
        0,
        None,
        None,
        {},
        (),
        extra.get("solver_counts"),
    )


DAYS = [
    _day("breaks", 3, 2, 1, 0, 0.5, 0.1, "breaks", solver_counts=COUNTS),
    _day("breaks", 1, 0, 1, 0, 0.0, 0.3, "holds", solver_counts=COUNTS),
    _day("holds", 0, 0, 0, 1, 0.0, 4e-17, "holds"),
    # its corners without a timeout lose more than any day with a result: not among the worst
    _day("timed out", 0, 0, 0, 0, 0.6, 0.7, "timed out"),
    _day("breaks", 1, 1, 0, 0, 0.25, 0.0, "timed out"),
]


def test_method_entry_by_hand():
    found = corner_summary.method_entry(0.4, "", DAYS)
    assert found["status"] == "searched" and found["days_searched"] == 5
    assert found["days_with_result"] == 4 and found["days_timed_out"] == 1
    assert found["days_breaking"] == 3
    assert found["share_of_days_breaking"] == pytest.approx(3 / 4)
    assert found["days_breaking_by_slot_loss"] == 2
    assert found["days_with_driver_shortfall"] == 2
    assert found["days_short_only_below_tolerance"] == 1
    assert found["mean_breaking_paths"] == pytest.approx(5 / 4)
    assert found["worst_slot_loss"] == 0.5 and found["worst_driver_loss"] == 0.3
    assert found["breaking_days_audited_misses"] == 1
    assert found["breaking_days_audited_timed_out"] == 1
    assert found["share_of_breaking_days_audited_misses"] == pytest.approx(1 / 3)
    assert found["solver_counts"] == {
        "solves": {"optimal": 12, "timeout": 2},
        "hours": 8,
        "fallback_hours": 2,
    }


def test_a_breaking_day_with_a_corner_short_below_the_tolerance_is_not_counted_as_short_only():
    # day F breaks on one corner and has another corner short by at most the tolerance; day C
    # holds with such a corner: only C holds and is short below the tolerance
    breaking_and_short = _day("breaks", 1, 1, 0, 1, 0.5, 4e-17, "breaks")
    found = corner_summary.method_entry(0.4, "", [breaking_and_short, DAYS[2]])
    assert found["days_breaking"] == 1
    assert found["days_short_only_below_tolerance"] == 1


def test_a_method_without_solver_counts_has_none():
    assert corner_summary.method_entry(0.4, "", DAYS[2:])["solver_counts"] is None


def test_scale_zero_is_no_offer_without_break_figures():
    found = corner_summary.method_entry(0.0, "the bound fails at every grid scale", DAYS)
    assert found == {
        "scale": 0.0,
        "scale_reason": "the bound fails at every grid scale",
        "status": "no offer",
        "days_searched": 5,
    }


def _check(beta, reason, difference, driver, bound, limited=False, no_failure=False, earlier=None):
    strict = reason is None
    return ClosedFormCheck(
        beta,
        1,
        1 if strict else None,
        1.0,
        1.0 - difference,
        difference,
        strict,
        strict or reason == "within tolerance",
        reason,
        bound,
        driver,
        driver <= bound + 1e-9,
        limited,
        no_failure,
        bound if earlier is None else earlier,
    )


def test_pass_mark_entry_by_hand():
    checks = [
        [
            # within the bound: its flags are not counted (only pairs above it are)
            _check(0.5, None, 0.0, 0.1, 0.1, limited=True, no_failure=True),
            _check(1.0, "within tolerance", 0.5, 0.2, 0.2, no_failure=True, earlier=0.05),
        ],
        [
            _check(0.5, None, 0.0, 0.4, 0.2, limited=True, no_failure=True),
            _check(1.0, "decision-hour cap", 0.25, 0.0, 0.0),
        ],
    ]
    found = corner_summary.pass_mark_entry([7, 3], ["d7", "d3"], checks)
    assert found["days_checked"] == 2 and found["days"] == [7, 3]
    assert found["pairs"] == 4
    assert found["pairs_agreed_strict"] == 2 and found["pairs_agreed"] == 3
    assert found["scales"] == [0.5, 1.0]
    assert found["largest_slot_difference"] == 0.5
    assert found["differences_by_reason"] == {"decision-hour cap": 1, "within tolerance": 1}
    assert found["disagreements_by_reason"] == {"decision-hour cap": 1}
    (wrong,) = found["disagreements"]
    assert (wrong["day"], wrong["date"], wrong["scale"]) == (3, "d3", 1.0)
    assert found["driver_pairs_above_bound"] == 1
    assert found["largest_driver_excess"] == pytest.approx(0.2)
    assert found["driver_above_bound_limited"] == 1
    assert found["driver_above_bound_no_slot_failure"] == 1
    # the earlier bound, a comparison only: 0.2 against 0.05 on the second pair (0.15 above it)
    # and 0.4 against 0.2 on the third (0.2 above it)
    assert found["earlier_driver_pairs_above_bound"] == 2
    assert found["earlier_largest_driver_excess"] == pytest.approx(0.2)
    assert found["agreed"] is False
    tolerated = corner_summary.pass_mark_entry([7], ["d7"], checks[:1])
    assert tolerated["agreed"] is True and tolerated["pairs_agreed_strict"] == 1


def test_nothing_checked_is_not_agreed():
    found = corner_summary.pass_mark_entry([], [], [])
    assert found["pairs"] == 0 and found["agreed"] is False


def test_the_largest_figures_are_maxima_over_every_pair():
    # two pairs above their bounds by 0.25 and 0.125: the largest excess is 0.25, not the sum;
    # every pair agrees strictly: the largest slot difference is 0.0, not "none"
    checks = [[_check(0.5, None, 0.0, 0.5, 0.25), _check(1.0, None, 0.0, 0.25, 0.125)]]
    found = corner_summary.pass_mark_entry([2], ["d2"], checks)
    assert found["driver_pairs_above_bound"] == 2
    assert found["largest_driver_excess"] == 0.25
    assert found["largest_slot_difference"] == 0.0
    assert found["pairs_agreed_strict"] == 2 and found["agreed"] is True


def test_a_disagreement_is_written_in_full():
    # every field distinct, so that a field read from another is seen
    wrong = ClosedFormCheck(
        *(0.7, 3, 5, 0.75, 0.5, 0.25, False, False, "decision-hour cap"),
        *(0.0, 0.0, True, False, False, 0.0),
    )
    found = corner_summary.pass_mark_entry([11, 4], ["d11", "d4"], [[], [wrong]])
    assert found["disagreements"] == [
        {
            "day": 4,
            "date": "d4",
            "scale": 0.7,
            "closed_form_first_failure": 3,
            "corner_first_failure": 5,
            "closed_form_slot_loss": 0.75,
            "corner_worst_slot_loss": 0.5,
            "reason": "decision-hour cap",
        }
    ]


class _Prepared:
    def __init__(self, n, date="2020-01-01", site="s"):
        self.date, self.site = date, site
        self.day = type("D", (), {"n_service": n})()


def _full_day(n_paths, audited_breaking_rows):
    return CornerDay(
        "m",
        0.4,
        "breaks",
        n_paths,
        5,
        4,
        3,
        2,
        1,
        0.75,
        0.125,
        (-1, 1),
        6,
        "holds",
        audited_breaking_rows,
        0.0625,
        0.03125,
        {"timeout": 2, "optimal": 9},  # written in the order the re-solves were counted
        (),
        {"solves": {"timeout": 1, "optimal": 3}, "hours": 7, "fallback_hours": 1},
    )


def test_a_row_is_written_in_full():
    # every field distinct; day 1 of three prepared days, then day 0 with no corner run
    prepared = [_Prepared(9, "d0", "s0"), _Prepared(2, "d1", "s1"), _Prepared(8, "d2", "s2")]
    days = [_full_day(16, 7), _full_day(0, 7)]
    rows = corner_summary.day_rows("u", "oracle", "m", 0.4, [1, 0], prepared, days)
    common = [
        "breaks",
        "VARIES",
        5,
        4,
        3,
        2,
        1,
        "0.75",
        "0.125",
        "-+",
        6,
        "holds",
        "VARIES",
        "0.0625",
        "0.03125",
        "timeout:2;optimal:9",
        "hours_cell",
    ]
    solver = "optimal:3;timeout:1;hours:7;fallback_hours:1"  # solves sorted by outcome
    first = ["u", "oracle", "m", 1, "d1", "s1", 2, "0.4", *common]
    first[9], first[20], first[-1] = 16, 7, solver
    second = ["u", "oracle", "m", 0, "d0", "s0", 9, "0.4", *common]
    second[9], second[20], second[-1] = 0, "", solver  # no corner run: no audited count
    assert rows == [first, second]
    assert len(first) == len(corner_summary.ROW_FIELDS)
    with pytest.raises(ValueError):  # a day index for every searched day
        corner_summary.day_rows("u", "oracle", "m", 0.4, [1], prepared, days)


def test_a_cell_is_empty_for_none_and_a_float_its_repr():
    assert corner_summary._cell(None) == ""
    assert corner_summary._cell(0.1 + 0.2) == "0.30000000000000004"
    assert corner_summary._cell(3) == 3 and corner_summary._cell("x") == "x"
    assert corner_summary._solver_cell(None) == ""


def test_the_rows_write_the_worst_corner_high_plus_low_minus_in_hour_order():
    day = _day("breaks", 1, 1, 0, 0, 0.5, 0.0, "breaks", worst_path=(1, -1, -1))
    (row,) = corner_summary.day_rows("u", "forecast", "m", 0.4, [0], [_Prepared(3)], [day])
    found = dict(zip(corner_summary.ROW_FIELDS, row, strict=True))
    assert found["worst_path"] == "+--"
    (row,) = corner_summary.day_rows("u", "forecast", "m", 0.4, [0], [_Prepared(3)], [DAYS[0]])
    found = dict(zip(corner_summary.ROW_FIELDS, row, strict=True))
    assert found["worst_path"] == ""
    assert found["solver_counts"] == "optimal:6;timeout:1;hours:4;fallback_hours:1"


def _document(entry, check):
    return {
        "label": "made up",
        "dial": 0.02,
        "configurations": {"forecast": "f"},
        "methods": ["m"],
        "run": {
            "seed": 1,
            "days": 5,
            "check_days": 2,
            "check_scales": [0.5, 1.0],
            "slot_tolerance": 1e-9,
        },
        "units": {
            "u": {
                "status": "calibratable",
                "counting_days": 9,
                "calibration_days": 4,
                "test_days": 5,
                "variants": {
                    "forecast": {
                        "configuration": "f",
                        "days_searched": 5,
                        "days_asked": 5,
                        "offering_test_days": 5,
                        "methods": {"m": entry},
                        "pass_mark": check,
                    }
                },
            }
        },
        "pass_mark": {
            "pairs": check["pairs"],
            "pairs_agreed_strict": check["pairs_agreed_strict"],
            "pairs_agreed": check["pairs_agreed"],
            "agreed": check["agreed"],
        },
        "timeouts": {},
        "failed": [],
    }


def _whole_document():
    """Every branch of the summary: a document head, a unit that cannot be split, a variant that
    failed, one with fewer days than asked, a method with no offer and one with solver counts, a
    variant with no pass mark, both kinds of difference, a driver maximum above its bound, nothing
    checked, timeouts and a failure. The figures of one table row are distinct from each other, so
    that a cell written in another's place is seen."""
    counts = {"solves": {"timeout": 1, "optimal": 6}, "hours": 5, "fallback_hours": 1}
    days = [
        _day("breaks", 5, 3, 2, 0, 0.5, 0.125, "breaks", solver_counts=counts),
        _day("breaks", 1, 0, 1, 0, 0.0, 0.3, "breaks", solver_counts=counts),
        _day("breaks", 2, 0, 2, 0, 0.0, 0.25, "timed out"),
        *[_day("holds", 0, 0, 0, 1, 0.0, 4e-17, "holds")] * 4,
        *[_day("timed out", 0, 0, 0, 0, 0.6, 0.7, "timed out")] * 2,
    ]
    # 9 searched, 7 with a result, 3 break (43%), by slot loss 1, by driver loss 3, timed out 2,
    # short below the tolerance 4, mean breaking corners 8/7, worst 0.5 and 0.3, no audited miss
    searched = corner_summary.method_entry(0.4, "", days)
    no_offer = corner_summary.method_entry(0.0, "the bound fails", DAYS[:2])
    strict = [
        _check(0.5, None, 0.0, 0.4, 0.2, limited=True),
        _check(0.5, None, 0.0, 0.4, 0.2, limited=True),
        _check(0.5, None, 0.0, 0.4, 0.2, no_failure=True),
        _check(0.5, None, 0.0, 0.1, 0.2),
    ]
    tolerated = [_check(1.0, "within tolerance", 0.5, 0, 0)] * 2
    # 2 days, 7 pairs, 4 strict, 6 agreed, 3 above the bound (2 limited, 1 with no failing hour)
    checks = [strict, [*tolerated, _check(0.5, "decision-hour cap", 0.25, 0.0, 0.0)]]
    checked = corner_summary.pass_mark_entry([7, 2], ["d7", "d2"], checks)
    document = _document(searched, checked)
    document["model"] = "lgbm_tail"
    document["rotation_whatif"] = {"what": "WHAT LINE", "evidence": "EVIDENCE LINE"}
    document["methods"] = ["m", "zz"]
    document["configurations"] = {"forecast": "f", "oracle": "o"}
    forecast = document["units"]["u"]["variants"]["forecast"]
    forecast["methods"]["zz"] = no_offer
    forecast["days_searched"], forecast["offering_test_days"], forecast["days_asked"] = 9, 11, 12
    variants = document["units"]["u"]["variants"]
    # a failed variant (not checked) between two checked ones
    variants["third"] = {"failed": "ValueError: made up"}
    # a method with no day with a result: no share, mean or worst loss, no breaking day
    unresolved = corner_summary.method_entry(0.4, "", [DAYS[3]])
    variants["oracle"] = {
        "configuration": "o",
        "days_searched": 1,
        "days_asked": 8,
        "offering_test_days": 1,
        "methods": {"m": unresolved, "zz": no_offer},
        "pass_mark": corner_summary.pass_mark_entry([], [], []),
    }
    # a unit that cannot be split, before one that can
    document["units"] = {"w": {"status": "too few days", "counting_days": 3}, **document["units"]}
    document["timeouts"] = {"certificates": 1, "oracle_re_solves_search": 2}
    document["failed"] = ["u third: ValueError: made up", "x: made up"]
    return document


def test_the_whole_summary_line_by_line():
    found = corner_summary.summary_lines(_whole_document())
    assert found == WHOLE_SUMMARY


WHOLE_SUMMARY = [
    "WHAT LINE",
    "EVIDENCE LINE",
    (
        "Corner search for model lgbm_tail, dial 0.02: split seed 1, up to 5 test days per "
        "unit and variant that offer, in a seeded order. made up."
    ),
    "Variants: forecast = f; oracle = o.",
    (
        "Each method runs on every corner of its own calibrated box (its tier A scale on the "
        "split's calibration days). break: share of searched days with a corner that fails an "
        "hour or leaves more than 0.0001 kWh of protected energy undelivered, over the days "
        "with a result; slot, driver: such days by kind; t/o: days without a result (a "
        "re-solve timed out); <tol: days that hold but have a corner short by more than 0 and "
        "at most 0.0001 kWh (not counted as a break, whatever the cause); paths: mean breaking "
        "corners; audited misses: breaking days on which the audited 8 paths hold."
    ),
    (
        "The corners give the worst case over the box for the fixed plan only; for every other "
        "method a path inside the box can be worse than every corner, so its break share is a "
        "lower bound."
    ),
    (
        "Pass mark: the first 2 searched days of each unit and variant at the scales 0.5, 1; "
        "strict: same first failing hour and slot losses within 1e-09; agreed: strict, or a "
        "difference labelled 'within tolerance' (at the closed form's failing hour every "
        "corner's draws miss the plan by at most 0.0001 kWh in total); driver > bound counts "
        "pairs above the driver-loss bound, the every-scale bound, which counts each car's "
        "decision-hour draw and each service hour at no more than the car can draw in it and "
        "bounds replay's driver loss at every scale up to the errors of V3 and V7 of "
        "fixed_plan_loss's module text, reported, not part of the verdict."
    ),
    "",
    "w (too few days, 3 counting days):",
    "  no numbers: the unit cannot be split",
    "",
    "u (9 counting days; split seed 1: 4 calibration days, 5 test days):",
    (
        " forecast (f): 9 days searched of 11 offering test days (fewer than the 12 asked: the "
        "unit has no more)"
    ),
    (
        "  method         scale  days  break   slot  driver  t/o  <tol  paths   worst slot  "
        "worst drv  audited misses"
    ),
    (
        "  m              0.40   9     43%     1     3       2    4     1.1     0.500       "
        "0.300      0 (0%)"
    ),
    "  zz             0.00   2     no offer",
    "  m solves on the corners: optimal 12, timeout 2; 10 hours planned, 2 by its fall-back",
    "  third: failed (ValueError: made up)",
    (
        " oracle (o): 1 days searched of 1 offering test days (fewer than the 8 asked: the "
        "unit has no more)"
    ),
    (
        "  method         scale  days  break   slot  driver  t/o  <tol  paths   worst slot  "
        "worst drv  audited misses"
    ),
    (
        "  m              0.40   1     -       0     0       1    0     -       -           -  "
        "        0 (-)"
    ),
    "  zz             0.00   2     no offer",
    " Pass mark (plan 13 WP11): fixed-plan closed form against the corner worst case",
    "  variant           days  pairs  strict  agreed  largest diff  driver > bound  verdict",
    "  forecast          2     7      4       6       0.500         3               DISAGREED",
    "    1 pair(s) differ, disagree: decision-hour cap",
    "    2 pair(s) differ, agree within tolerance: within tolerance",
    "    driver > bound: 2 with a car short of its plan, 1 with no failing hour",
    "  third: not checked",
    (
        "  oracle            0     0      0       0       -             0               "
        "nothing checked"
    ),
    "",
    "Pass mark over all units: 4 of 7 pairs agree strictly, 6 with the tolerance: DISAGREED.",
    "Timeouts (recorded outcomes): certificates 1, oracle_re_solves_search 2",
    "FAILED: u third: ValueError: made up; x: made up",
]


def test_the_summary_gives_the_miss_share_both_counts_and_the_causes():
    checks = [
        [_check(0.5, None, 0.0, 0.4, 0.2, limited=True), _check(1.0, "within tolerance", 0.5, 0, 0)]
    ]
    check = corner_summary.pass_mark_entry([7], ["d7"], checks)
    text = "\n".join(
        corner_summary.summary_lines(_document(corner_summary.method_entry(0.4, "", DAYS), check))
    )
    assert "1 (33%)" in text  # audited misses and their share of the breaking days
    assert "1 of 2 pairs agree strictly, 2 with the tolerance: agreed." in text
    assert "1 pair(s) differ, agree within tolerance: within tolerance" in text
    assert "driver > bound: 1 with a car short of its plan, 0 with no failing hour" in text
    assert (
        "m solves on the corners: optimal 12, timeout 2; 8 hours planned, 2 by its fall-back"
        in text
    )
    assert "lower bound" in text
