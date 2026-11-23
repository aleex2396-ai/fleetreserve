"""The summary arithmetic on made-up day results, worked by hand in the test text."""

import pytest

from fleetreserve.ceiling_summary import (
    block_summary,
    failures,
    hourly_summary,
    missing_counts,
    order_violations,
    t4_check,
)
from fleetreserve.ceiling_values import BLOCK_KEYS, HOURLY_KEYS, VALUE_KEYS, DayResult


def _day(
    name,
    *,
    hind=None,
    caus=None,
    aff=None,
    diag=None,
    hourly=None,
    causal=None,
    tree=None,
    sampled=True,
    in_window=False,
    t4=False,
    outcomes=None,
):
    values = dict.fromkeys(VALUE_KEYS)
    values.update(
        block_hind=hind,
        block_caus=caus,
        block_aff=caus if aff is None else aff,
        block_diag=caus if diag is None else diag,
        hourly_hind=hourly,
        hourly_causal=causal,
        hourly_tree=tree,
    )
    found = {k: ("optimal" if values[k] is not None else "timeout") for k in VALUE_KEYS}
    found.update(outcomes or {})
    if not sampled:
        for key in (*BLOCK_KEYS, "hourly_tree"):
            values[key], found[key] = None, "not solved"
    return DayResult("unit", "S", name, 3, 6, 0, in_window, t4, sampled, None, values, found)


# Six sampled days. Ratios V_caus / V_hind: day 1: 1.0, day 2: 0.5, day 3: 0.9. Day 4 has
# V_hind = 0 (no ratio), day 5 a timed-out V_caus, day 6 a V_hind that ended in "error".
SAMPLE = [
    _day("2020-01-01", hind=10.0, caus=10.0, t4=True),
    _day("2020-01-02", hind=8.0, caus=4.0, in_window=True),
    _day("2020-01-03", hind=10.0, caus=9.0),
    _day("2020-01-04", hind=0.0, caus=0.0),
    _day("2020-01-05", hind=6.0, caus=None),
    _day("2020-01-06", hind=None, caus=5.0, outcomes={"block_hind": "error"}),
]


def test_the_mean_median_and_minimum_of_the_ratio_leave_out_zero_and_missing_days():
    summary = block_summary(SAMPLE)
    ratio = summary["ratio_caus_over_hind"]
    assert ratio["days"] == 6
    assert ratio["days_in_ratio"] == 3
    assert ratio["days_with_zero_hindsight"] == 1
    assert ratio["days_with_a_missing_value"] == 2
    assert ratio["mean"] == pytest.approx((1.0 + 0.5 + 0.9) / 3)
    assert ratio["median"] == pytest.approx(0.9)
    assert ratio["min"] == pytest.approx(0.5)
    assert summary["statement"] == (
        "mean V_caus/V_hind = 0.800000; the owner's rule threshold is 0.99"
    )


def test_days_below_hindsight_are_counted_over_the_days_with_both_values():
    summary = block_summary(SAMPLE)
    below = summary["caus_below_hind"]
    assert below["of_days"] == 4  # days 1 to 4
    assert below["days"] == 2  # days 2 and 3
    assert [d["day"] for d in below["list"]] == ["unit S 2020-01-02", "unit S 2020-01-03"]


def test_a_missing_value_is_counted_by_its_outcome_and_never_read_as_zero():
    assert missing_counts(SAMPLE, "block_caus") == {"asked": 6, "timeout": 1, "other": 0}
    assert missing_counts(SAMPLE, "block_hind") == {"asked": 6, "timeout": 0, "other": 1}
    summary = block_summary(SAMPLE)
    assert summary["missing"]["block_caus"]["timeout"] == 1
    assert summary["missing"]["block_hind"]["other"] == 1


def test_groups_split_the_ratio_by_window_departure_and_the_hypotheses():
    groups = block_summary(SAMPLE)["ratio_by_group"]
    assert groups["in-window departure"]["mean"] == pytest.approx(0.5)
    assert groups["meets T4's hypotheses"]["mean"] == pytest.approx(1.0)
    assert groups["no in-window departure"]["days"] == 5


def test_an_affine_value_below_the_tree_is_listed_as_a_finding():
    days = [
        _day("2020-02-01", hind=10.0, caus=8.0, aff=6.0),
        _day("2020-02-02", hind=5.0, caus=5.0),
    ]
    summary = block_summary(days)
    assert summary["aff_below_caus"]["days"] == 1
    assert summary["aff_below_caus"]["list"][0]["day"] == "unit S 2020-02-01"
    assert summary["diag_below_aff"]["days"] == 0


def test_no_day_forming_a_ratio_gives_no_mean_and_says_so():
    summary = block_summary([_day("2020-01-01", hind=0.0, caus=0.0)])
    assert summary["ratio_caus_over_hind"]["mean"] is None
    assert "no day forms a ratio" in summary["statement"]


def test_the_t4_check_counts_checked_and_unchecked_hypotheses_days():
    days = [
        _day("2020-03-01", hind=10.0, caus=10.0, t4=True),
        _day("2020-03-02", hind=10.0, caus=None, t4=True),
        _day("2020-03-03", hind=10.0, caus=2.0),  # not a hypotheses day: no part of the check
    ]
    check = t4_check(days)
    assert check["days_meeting_hypotheses"] == 2
    assert check["days_checked"] == 1
    assert check["days_not_checked_missing_value"] == 1
    assert check["held"] is True


def test_a_hypotheses_day_with_unequal_values_fails_the_check_with_a_line():
    days = [_day("2020-03-01", hind=10.0, caus=9.5, t4=True)]
    check = t4_check(days)
    assert check["held"] is False
    assert check["violations"][0]["day"] == "unit S 2020-03-01"
    lines = failures(days, days)
    assert any("V_hind 10.0 differs from V_caus 9.5" in line for line in lines)


def test_a_tiny_difference_within_the_tolerance_is_equality():
    days = [_day("2020-03-01", hind=10.0, caus=10.0 - 1e-8, t4=True)]
    assert t4_check(days)["held"] is True
    assert failures(days, days) == []


def test_failures_carry_the_order_lines_too():
    days = [_day("2020-03-01", hind=10.0, caus=8.0, aff=9.0)]
    assert len(failures(days, days)) == 1


def test_the_order_of_the_values_is_checked():
    assert order_violations([_day("d", hind=10.0, caus=8.0, aff=6.0, diag=4.0)]) == []
    wrong = _day("d", hind=10.0, caus=8.0, aff=9.0, diag=4.0)  # affine above the tree
    assert len(order_violations([wrong])) == 1
    above = _day("d", hind=10.0, caus=12.0)  # tree above hindsight
    assert len(order_violations([above])) == 1
    tree = _day("d", hourly=5.0, causal=5.0, tree=4.0)
    assert len(order_violations([tree])) == 1
    over = _day("d", hourly=5.0, causal=6.0, sampled=False)
    assert len(order_violations([over])) == 1


def test_a_day_that_could_not_be_built_is_a_failure():
    broken = DayResult(
        "unit", "S", "2020-04-01", 2, 6, 0, False, False, False, "an empty set",
        dict.fromkeys(VALUE_KEYS), dict.fromkeys(VALUE_KEYS, "not solved"),
    )  # fmt: skip
    assert failures([], [broken]) == ["unit S 2020-04-01: the day could not be built: an empty set"]


# Hourly: four days of one unit. (hindsight, best causal): (10, 10), (8, 0), (6, 3), (0, 0);
# the third has a window departure; the fifth has a missing causal value.
HOURLY = [
    _day("2020-05-01", hourly=10.0, causal=10.0, sampled=False),
    _day("2020-05-02", hourly=8.0, causal=0.0, sampled=False, in_window=True),
    _day("2020-05-03", hourly=6.0, causal=3.0, sampled=False, in_window=True),
    _day("2020-05-04", hourly=0.0, causal=0.0, sampled=False),
    _day("2020-05-05", hourly=7.0, causal=None, sampled=False),
]


def test_hourly_sums_means_ratios_and_shares():
    groups = hourly_summary(HOURLY)
    g = groups["all days"]
    assert g["days"] == 5
    assert g["days_with_both_values"] == 4
    assert g["days_with_a_missing_causal_value"] == 1
    assert g["sum_hindsight_kwh"] == pytest.approx(24.0)
    assert g["sum_best_causal_kwh"] == pytest.approx(13.0)
    assert g["mean_hindsight_kwh_per_day"] == pytest.approx(6.0)
    assert g["mean_best_causal_kwh_per_day"] == pytest.approx(3.25)
    assert g["ratio_of_sums"] == pytest.approx(13.0 / 24.0)
    assert g["days_with_positive_hindsight"] == 3
    assert g["per_day_ratio_mean"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert g["per_day_ratio_median"] == pytest.approx(0.5)
    assert g["share_days_differing"] == pytest.approx(2 / 4)  # days 2 and 3 of the four
    assert g["share_days_zero_causal_positive_hindsight"] == pytest.approx(1 / 4)  # day 2


def test_hourly_groups_split_by_window_departure():
    groups = hourly_summary(HOURLY)
    window = groups["in-window departure"]
    assert window["days"] == 2
    assert window["sum_hindsight_kwh"] == pytest.approx(14.0)
    assert window["sum_best_causal_kwh"] == pytest.approx(3.0)
    assert window["share_days_differing"] == pytest.approx(1.0)
    other = groups["no in-window departure"]
    assert other["days"] == 3
    assert other["days_with_both_values"] == 2
    assert other["ratio_of_sums"] == pytest.approx(10.0 / 10.0)
    assert other["share_days_zero_causal_positive_hindsight"] == pytest.approx(0.0)


def test_hourly_summary_of_a_unit_with_no_value_has_no_means():
    g = hourly_summary([_day("d", hourly=3.0, causal=None, sampled=False)])["all days"]
    assert g["days_with_both_values"] == 0
    assert g["mean_hindsight_kwh_per_day"] is None
    assert g["ratio_of_sums"] is None
    assert g["share_days_differing"] is None


def test_value_keys_cover_the_hourly_and_block_values():
    assert set(VALUE_KEYS) == set(HOURLY_KEYS) | set(BLOCK_KEYS)
