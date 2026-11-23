"""The statistics of outcome O3 (`o3_summary`) and the day readings D5, D6 of `o3_day`.

1. On four results made by hand: ceiling 4, 2, 0 and missing (a timeout); LP (24) 2, missing (a
   timeout), 0 and 1. LP (24)'s kWh statistics are over its three values (mean 1, median 1); its
   share is over the one day with both values and a ceiling above zero (2 / 4 = 0.5); the day
   with ceiling 0 is counted apart; the ratio of means is over the two days with both values,
   (2 + 0) / 2 over (4 + 0) / 2 = 0.5; the missing value is counted as a timeout and in the total.
2. Reading D5: a day with no car has every value 0, outcome "empty fleet", without a solve.
3. Reading D6: a day whose cars have no room (the base set holds only the zero profile) has LP
   (24) and LP (27) unsolved, value 0 (the ceiling is 0); with a room of 2e-7 kWh on one of three
   cars the base set's energy is 6.7e-8 kWh (trivial) but the hourly ceiling 2e-7 kWh is above
   `ZERO`, so both values are missing.
4. A step that raises leaves the day with `error` set, every value missing ("day not built"), and
   `failures` names it.
5. Each method's line in the summary carries its own day counts and its label, and one more line
   gives every method's share over the days where all have a value. Two days made by hand,
   hourly (ceiling, LP (24), LP (27) smallest, largest found and bound, general split, without
   hour-0 gain): (4, 2, 3, 4, 4, 3.5, 3) and (2, 1, missing, missing, 2, 2, 1), LP (27)'s two
   found values a timeout, the bound's outcomes closed and ceiling; block 1 everywhere. LP (24):
   mean 1.5 kWh on 2 days, share 0.5 on 2 days; LP (27)'s largest: 4 kWh on 1 day, share 1, 1
   timeout; the bound: 3 kWh on 2 days, 0 above the largest found on 1 day; the general split's
   share (0.875 + 1) / 2 = 0.9375 on 2 days. On the 1 day where all have a value: 0.5, 0.75, 1,
   1, 0.875 and 0.75; block: 2 days, every share 1. The header names LP (27)'s settings.
6. The bound's bracket: largest found 3 and 2 with bounds 3.5 (open) and 2 (closed), a day with
   the largest found missing (bound the ceiling, outcome ceiling) and a day with the bound missing
   (a timeout): above the largest found by 0.5 and 0 on 2 days, mean 0.25; outcomes 1 of each of
   the three over the days with a bound, 1 missing. A largest found above the bound is a
   failure.
7. The report's fourth line reads LP (27) as a range: the largest found is from a local search
   and not the largest over all optimal solutions, the bound is proved (missing when not), and
   the largest over all lies between the two.
"""

import datetime

import pytest

from fleetreserve import o3_day
from fleetreserve.day import Day, TrueCar, covering_recovery_hours
from fleetreserve.o3_day import (
    BOUND,
    INNER,
    KEYS,
    PRODUCTS,
    DayResult,
    DayTask,
    TimeLimits,
    compute_day,
)
from fleetreserve.o3_summary import (
    all_present,
    failures,
    lp27_reach,
    missing_total,
    summary_lines,
    unit_summary,
    value_summary,
)


def _result(day, ceiling, lp24, outcome_ceiling="exact", outcome_lp24="optimal"):
    values = {p: dict.fromkeys(KEYS, 1.0) for p in PRODUCTS}
    outcomes = {p: dict.fromkeys(KEYS, "optimal") for p in PRODUCTS}
    values["hourly"]["ceiling"], outcomes["hourly"]["ceiling"] = ceiling, outcome_ceiling
    values["hourly"]["lp24"], outcomes["hourly"]["lp24"] = lp24, outcome_lp24
    return DayResult("u", "A", day, 3, 2, None, values, outcomes, None, None)


RESULTS = [
    _result("2020-01-01", 4.0, 2.0),
    _result("2020-01-02", 2.0, None, outcome_lp24="timeout"),
    _result("2020-01-03", 0.0, 0.0),
    _result("2020-01-04", None, 1.0, outcome_ceiling="timeout"),
]


def test_the_statistics_leave_missing_values_out_and_count_them():
    entry = value_summary(RESULTS, "hourly", "lp24")
    assert entry["days"] == 4 and entry["days_with_value"] == 3
    assert entry["missing"] == {"timeout": 1}
    assert entry["kwh"] == {"days": 3, "mean": 1.0, "median": 1.0, "min": 0.0, "max": 2.0}
    assert entry["share"] == {"days": 1, "mean": 0.5, "median": 0.5, "min": 0.5, "max": 0.5}
    assert entry["days_zero_ceiling"] == 1
    assert entry["ratio_of_means"] == pytest.approx(0.5)
    ceiling = value_summary(RESULTS, "hourly", "ceiling")
    assert "share" not in ceiling and ceiling["missing"] == {"timeout": 1}
    assert ceiling["kwh"]["mean"] == pytest.approx(2.0)
    assert missing_total({"u": unit_summary(RESULTS)}) == {"timeout": 2}


def _day(cars, n_service=2):
    cars = tuple(cars)
    n_recovery = covering_recovery_hours(cars, n_service) if cars else 0
    return Day("u", datetime.date(2020, 1, 1), n_service, n_recovery, 0.0, cars)


def _compute(day):
    return compute_day(DayTask("u", "A", "2020-01-01", day, "day-average", TimeLimits()))


def test_an_empty_fleet_has_every_value_zero_without_a_solve(monkeypatch):
    for name in ("certify", "general_split", "search", "inner_set"):
        monkeypatch.setattr(o3_day, name, None)  # any solve would raise
    result = _compute(_day([]))
    assert result.error is None
    for product in PRODUCTS:
        assert result.values[product] == dict.fromkeys(KEYS, 0.0)
        assert result.outcomes[product] == dict.fromkeys(KEYS, o3_day.EMPTY_FLEET)


def test_a_trivial_base_set_with_zero_ceiling_gives_zero():
    result = _compute(_day([TrueCar(3, 2.0, 0.0), TrueCar(5, 1.0, 0.0), TrueCar(1, 3.0, 0.0)]))
    assert result.error is None
    for product in PRODUCTS:
        assert result.values[product]["ceiling"] == 0.0
        assert result.values[product]["general_split"] == 0.0
        assert result.values[product]["general_split_no_hour0_gain"] == 0.0
        for method in INNER:
            assert result.values[product][method] == 0.0
            assert result.outcomes[product][method] == o3_day.TRIVIAL_BASE
    assert result.alpha_24 is None and result.trace_27 is None


def test_a_trivial_base_set_with_a_positive_ceiling_is_missing():
    result = _compute(_day([TrueCar(3, 2.0, 2e-7), TrueCar(5, 1.0, 0.0), TrueCar(1, 3.0, 0.0)]))
    assert result.values["hourly"]["ceiling"] == pytest.approx(2e-7)
    assert result.values["hourly"]["ceiling"] > o3_day.ZERO
    for product in PRODUCTS:
        for method in INNER:
            assert result.values[product][method] is None
            assert result.outcomes[product][method] == o3_day.TRIVIAL_BASE
    entry = value_summary([result], "hourly", "lp24")
    assert entry["missing"] == {o3_day.TRIVIAL_BASE: 1}


def test_a_step_that_raises_leaves_the_day_not_built(monkeypatch):
    def broken(day):
        raise ValueError("made-up fault")

    monkeypatch.setattr(o3_day, "true_rows", broken)
    result = _compute(_day([TrueCar(3, 2.0, 1.0)]))
    assert result.error == "ValueError: made-up fault"
    for product in PRODUCTS:
        assert result.values[product] == dict.fromkeys(KEYS)
        assert result.outcomes[product] == dict.fromkeys(KEYS, o3_day.NOT_BUILT)
    assert failures([result]) == [
        "u A 2020-01-01: the day could not be built: ValueError: made-up fault"
    ]


def _two(date, hourly, missing=(), bound="closed"):
    values = {"hourly": dict(zip(KEYS, hourly, strict=True)), "block": dict.fromkeys(KEYS, 1.0)}
    outcomes = {p: dict.fromkeys(KEYS, "optimal") for p in PRODUCTS}
    outcomes["hourly"]["ceiling"] = outcomes["block"]["ceiling"] = "exact"
    outcomes["hourly"][BOUND], outcomes["block"][BOUND] = bound, "closed"
    for key in missing:
        values["hourly"][key], outcomes["hourly"][key] = None, "timeout"
    return DayResult("u", "A", date, 3, 2, None, values, outcomes, None, None)


TWO = [
    _two("2020-01-01", (4.0, 2.0, 3.0, 4.0, 4.0, 3.5, 3.0)),
    _two(
        "2020-01-02",
        (2.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0),
        missing=("lp27_smallest_found", "lp27_largest_found"),
        bound="ceiling",
    ),
]


def _document(results):
    units = {"u": unit_summary(results)}
    return {
        "inputs": {"truth_run": "made-up", "setting": "main"},
        "arguments": {
            "units": ["u"],
            "days": 2,
            "seed": 1,
            "base_set": "day-average",
            "rho": 0.0,
            "lp27_days": 2,
            "lp27_cap_seconds": 7200.0,
            "departures_from_plan": [],
            "seconds": {"lp24": 600.0, "others": 300.0, "box": 30.0},
            "lp27": {
                "effort_share": 0.5,
                "settings": {"draws": 20, "rounds": 12},
                "bound_nodes": 5,
            },
        },
        "sample": {"u": {"sampled_days": 2, "counting_days": 2}},
        "units": units,
        "lp27_cap": {"u": lp27_reach(results)},
        "all_present": {"u": all_present(results)},
        "missing_values": missing_total(units),
        "failures": failures(results),
    }


def test_each_method_carries_its_own_day_count_and_one_line_shares_the_common_days():
    common = all_present(TWO)
    assert common["hourly"]["days"] == 1 and common["block"]["days"] == 2
    expected = {
        "lp24": 0.5,
        "lp27_smallest_found": 0.75,
        "lp27_largest_found": 1.0,
        "lp27_largest_bound": 1.0,
        "general_split": 0.875,
        "general_split_no_hour0_gain": 0.75,
    }
    for method, share in expected.items():
        assert common["hourly"][method]["share"]["days"] == 1
        assert common["hourly"][method]["share"]["mean"] == pytest.approx(share)
        assert common["hourly"][method]["ratio_of_means"] == pytest.approx(share)
        assert common["block"][method]["share"]["days"] == 2
        assert common["block"][method]["share"]["mean"] == pytest.approx(1.0)
    text = summary_lines(_document(TWO))
    assert "MISSING VALUES (never read as zero): 2 (2 timeout)" in text
    assert text[1].endswith(
        "; LP (27) search effort 0.5 (draws 20, rounds 12), bound relaxations per product 5."
    )
    hourly = text.index("  hourly: ceiling kWh/day mean 3.000 median 3.000 on 2 days; missing none")
    assert text[hourly + 1 : hourly + 8] == [
        "    lp24 (the largest box in LP (24)'s set): kWh/day mean 1.500 median 1.500 on 2 days; "
        "share of ceiling mean 0.5000 median 0.5000 on 2 days (0 with ceiling 0); ratio of means "
        "0.5000; missing none",
        "    lp27_smallest_found (LP (27): the smallest box over the optimal solutions searched; "
        "the smallest over all optimal solutions is at most this): kWh/day mean 3.000 median "
        "3.000 on 1 days; share of ceiling mean 0.7500 median 0.7500 on 1 days (0 with ceiling "
        "0); ratio of means 0.7500; missing 1 timeout",
        "    lp27_largest_found (LP (27): the largest box over the optimal solutions searched; the "
        "largest over all optimal solutions is at least this): kWh/day mean 4.000 median 4.000 "
        "on 1 days; share of ceiling mean 1.0000 median 1.0000 on 1 days (0 with ceiling 0); "
        "ratio of means 1.0000; missing 1 timeout",
        "    lp27_largest_bound (LP (27): no optimal solution's set holds a larger box (a bound "
        "proved by a linear relaxation, to the solver's tolerances)): kWh/day mean 3.000 median "
        "3.000 on 2 days; share of ceiling mean 1.0000 median 1.0000 on 2 days (0 with ceiling 0); "
        "ratio of means 1.0000; missing none; above lp27_largest_found by mean 0.0000 median "
        "0.0000 largest 0.0000 kWh on 1 days; outcomes 1 ceiling, 1 closed (closed: met the "
        "largest found; ceiling: the relaxations' bound at or above the ceiling; open: the "
        "relaxations allowed ran out first)",
        "    general_split (general split, decision-hour gains summing to zero over the cars): "
        "kWh/day mean 2.750 median 2.750 on 2 days; share of ceiling mean 0.9375 median 0.9375 on "
        "2 days (0 with ceiling 0); ratio of means 0.9167; missing none",
        "    general_split_no_hour0_gain (general split, decision hour without gain): kWh/day mean "
        "2.000 median 2.000 on 2 days; share of ceiling mean 0.6250 median 0.6250 on 2 days (0 "
        "with ceiling 0); ratio of means 0.6667; missing none",
        "    share of ceiling on the 1 days where the ceiling and every method have a value: lp24 "
        "mean 0.5000 on 1 days, ratio of means 0.5000; lp27_smallest_found mean 0.7500 on 1 days, "
        "ratio of means 0.7500; lp27_largest_found mean 1.0000 on 1 days, ratio of means 1.0000; "
        "lp27_largest_bound mean 1.0000 on 1 days, ratio of means 1.0000; general_split mean "
        "0.8750 on 1 days, ratio of means 0.8750; general_split_no_hour0_gain mean 0.7500 on 1 "
        "days, ratio of means 0.7500",
    ]


def test_the_bound_gives_the_bracket_above_the_largest_found():
    results = [
        _two("2020-01-01", (4.0, 2.0, 3.0, 3.0, 3.5, 3.5, 3.0), bound="open"),
        _two("2020-01-02", (2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0)),
        _two("2020-01-03", (2.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0), ("lp27_largest_found",), "ceiling"),
        _two("2020-01-05", (2.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0), ("lp27_largest_found", BOUND)),
    ]
    entry = value_summary(results, "hourly", BOUND)
    assert entry["outcomes"] == {"ceiling": 1, "closed": 1, "open": 1}
    assert entry["missing"] == {"timeout": 1} and entry["days_with_value"] == 3
    assert entry["above_largest_found"] == {
        "days": 2,
        "mean": 0.25,
        "median": 0.25,
        "min": 0.0,
        "max": 0.5,
    }
    assert "outcomes" not in value_summary(results, "hourly", "lp27_largest_found")
    found_above = _two("2020-01-04", (4.0, 2.0, 3.0, 3.5, 3.0, 3.5, 3.0))
    assert failures([found_above]) == [
        "u A 2020-01-04 hourly: lp27_largest_found 3.5 above lp27_largest_bound 3.0 (a box an "
        "optimal solution holds is at most the bound)"
    ]


def test_the_report_reads_lp27_as_a_range_whose_found_end_is_not_the_largest():
    text = summary_lines(_document(TWO))
    assert text[2].startswith("MISSING VALUES")
    line = text[3]
    assert line.startswith("LP (27) is a range, not one number: lp27_largest_found is the largest")
    assert "local search" in line and "(not the largest over all of them;" in line
    assert (
        "lp27_largest_bound is a number no optimal solution's set holds a larger box than" in line
    )
    assert "missing when not proved" in line
    assert "the largest box over all optimal solutions lies between the two" in line
