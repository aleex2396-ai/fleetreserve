"""The slopes, day types and permutation test of T7's measurement (`t7_stat`), on hand-worked days.

1. Day types: the weekday of the date, a public holiday as its own type; types with fewer than 10
   days pooled into "other"; the types tested are those with at least 10 days after pooling.
2. Losses at a scale: 0 at scale 0 and below, the closed-form slot loss (strictly below: a day
   whose threshold equals the scale does not lose) and the driver term (`t7_stat`) above it.
3. Slopes and the days whose loss changes, worked by hand: one-hour days with known thresholds at
   windows 0.05, 0.1 and 0.2 around scale 0.5; a two-hour day; a driver ramp; the one-sided window
   at scale 0.05, where the lower end is 0 and the width is the scale plus the window.
4. Marginal loss per unit value: the type's summed loss change over its width times its summed
   value, equal to its slope over its mean value.
5. The permutation p-value: against a brute-force enumeration of every assignment of the labels on
   a tiny case (6 and 7 days), and the statistic against a plain loop; p is 1 / (1 + rounds) when
   no permutation reaches the observed statistic, and 1 when the observed statistic is 0.
6. Holm's adjustment, by hand.
"""

import itertools
import math

import numpy as np
import pytest
from size_fake import ramp_terms

from fleetreserve import t7_stat
from fleetreserve.size_scale import DayTerms

NO_DRIVER = (np.zeros(0), np.zeros(0), 0.0)


def one_hour(threshold: float, value: float = 1.0) -> DayTerms:
    return ramp_terms(value, np.array([threshold]), NO_DRIVER)


# --- 1. day types --------------------------------------------------------------------------------


def test_raw_types_are_weekdays_and_a_holiday_is_its_own_type():
    assert t7_stat.raw_type("2020-01-06", False) == "Monday"
    assert t7_stat.raw_type("2020-01-12", False) == "Sunday"
    assert t7_stat.raw_type("2020-01-01", True) == t7_stat.HOLIDAY
    assert t7_stat.raw_type("2020-01-06", True) == t7_stat.HOLIDAY


def test_small_types_are_pooled_into_other_and_only_types_of_ten_days_are_tested():
    raw = ["Monday"] * 12 + ["Tuesday"] * 9 + [t7_stat.HOLIDAY] * 3 + ["Friday"] * 10
    pooled = t7_stat.pool_types(raw)
    assert pooled == ["Monday"] * 12 + ["other"] * 12 + ["Friday"] * 10
    assert t7_stat.tested_types(pooled) == ("Monday", "Friday", "other")
    small = t7_stat.pool_types(["Monday"] * 12 + ["Tuesday"] * 4 + ["Sunday"] * 5)
    assert t7_stat.tested_types(small) == ("Monday",)
    assert t7_stat.type_counts(small) == {"Monday": 12, "other": 9}


# --- 2. and 3. losses and slopes -----------------------------------------------------------------


def test_losses_are_zero_at_scale_zero_and_the_closed_forms_above():
    terms = [one_hour(-0.2), one_hour(0.5), ramp_terms(1.0, np.array([0.8, 0.3]), NO_DRIVER)]
    at_zero = t7_stat.day_losses(terms, 0.0)
    assert at_zero["slot"].tolist() == [0.0, 0.0, 0.0]
    assert t7_stat.day_losses(terms, 0.5)["slot"].tolist() == [1.0, 0.0, 0.5]
    assert t7_stat.day_losses(terms, 0.9)["slot"].tolist() == [1.0, 1.0, 1.0]
    ramp = ramp_terms(1.0, np.array([9.0]), (np.array([-0.5, -1.0]), np.array([1.0, 1.0]), 2.0))
    assert t7_stat.day_losses([ramp], 0.75)["driver"].tolist() == [pytest.approx(0.125)]
    assert t7_stat.day_losses([ramp], 0.0)["driver"].tolist() == [0.0]


def test_windows_and_the_one_sided_case():
    assert t7_stat.window(0.5, 0.1) == (pytest.approx(0.4), pytest.approx(0.6))
    assert t7_stat.window(0.05, 0.1) == (0.0, pytest.approx(0.15))
    assert t7_stat.window(0.0, 0.2) == (0.0, 0.2)


@pytest.mark.parametrize(
    "h, changes",
    [
        # thresholds 0.45, 0.7, 0.3, 0.55; L(b) = 1 when b is above the threshold
        (0.05, [1, 0, 0, 0]),  # [0.45, 0.55]: 0.45 loses at 0.55, not at 0.45; 0.55 at neither
        (0.1, [1, 0, 0, 1]),  # [0.4, 0.6]
        (0.2, [1, 0, 1, 1]),  # [0.3, 0.7]: 0.7 does not lose at 0.7; 0.3 loses at 0.7, not 0.3
    ],
)
def test_slopes_by_hand_at_each_window(h, changes):
    terms = [one_hour(0.45, 10.0), one_hour(0.7, 30.0), one_hour(0.3, 5.0), one_hour(0.55, 15.0)]
    found = t7_stat.loss_differences(terms, 0.5, h)
    assert found.width == pytest.approx(2 * h)
    assert found.differences["slot"].tolist() == changes
    types = ["X", "X", "Y", "Y"]
    readings = t7_stat.type_readings(found, "slot", np.array([t.value for t in terms]), types)
    x, y = readings["X"], readings["Y"]
    assert x.days == 2 and x.mean_value == 20.0
    assert x.slope == pytest.approx((changes[0] + changes[1]) / 2 / (2 * h))
    assert x.marginal == pytest.approx(x.slope / 20.0)
    assert y.slope == pytest.approx((changes[2] + changes[3]) / 2 / (2 * h))
    assert y.marginal == pytest.approx(y.slope / 10.0)
    # the days whose loss differs between the two ends, counted by hand from `changes`
    assert (x.changing, y.changing) == (changes[0] + changes[1], changes[2] + changes[3])


def test_a_two_hour_day_and_a_driver_ramp_by_hand():
    two = ramp_terms(4.0, np.array([0.8, 0.45]), (np.array([-0.5]), np.array([1.0]), 2.0))
    found = t7_stat.loss_differences([two], 0.5, 0.1)
    # slot: at 0.4 nothing fails, at 0.6 hour 2 fails: loss 1/2
    assert found.differences["slot"].tolist() == [0.5]
    # driver: max(0, -0.5 + b) / 2 is 0.05 at 0.6 and 0 at 0.4
    assert found.differences["driver"].tolist() == [pytest.approx(0.05)]
    readings = t7_stat.type_readings(found, "driver", np.array([4.0]), ["Z"])
    assert readings["Z"].slope == pytest.approx(0.25)
    assert readings["Z"].marginal == pytest.approx(0.0625)
    assert readings["Z"].mean_loss_low == 0.0
    assert readings["Z"].mean_loss_high == pytest.approx(0.05)


def test_the_one_sided_window_at_scale_005():
    # broken at the centre (threshold -0.2): loses at every scale above 0, not at 0
    terms = [one_hour(-0.2, 2.0), one_hour(0.1, 2.0), one_hour(0.2, 2.0)]
    found = t7_stat.loss_differences(terms, 0.05, 0.1)
    assert (found.low, found.high) == (0.0, pytest.approx(0.15))
    assert found.width == pytest.approx(0.15)
    assert found.differences["slot"].tolist() == [1.0, 1.0, 0.0]
    reading = t7_stat.type_readings(found, "slot", np.full(3, 2.0), ["W"] * 3)["W"]
    assert reading.slope == pytest.approx((2 / 3) / 0.15)
    assert reading.marginal == pytest.approx((2 / 3) / 0.15 / 2.0)


def test_a_type_with_no_day_has_no_reading():
    found = t7_stat.loss_differences([one_hour(0.45)], 0.5, 0.1)
    readings = t7_stat.type_readings(found, "slot", np.array([1.0]), ["X"], names=("X", "Y"))
    assert readings["Y"] is None and readings["X"].days == 1


# --- 4. and 5. the statistic and the permutation p-value -----------------------------------------


def loop_statistic(diff, values, codes, k, width):
    ratios = []
    for g in range(k):
        num = sum(d for d, c in zip(diff, codes, strict=True) if c == g)
        den = sum(v for v, c in zip(values, codes, strict=True) if c == g)
        ratios.append(num / (width * den))
    return max(ratios) - min(ratios)


def brute_force_p(diff, values, codes, k, width):
    observed = loop_statistic(diff, values, codes, k, width)
    hits = total = 0
    for order in itertools.permutations(range(len(codes))):
        found = loop_statistic(diff, values, [codes[i] for i in order], k, width)
        hits += found >= observed - 1e-12
        total += 1
    return hits / total


@pytest.mark.parametrize(
    "diff, values, codes, k",
    [
        ([1.0, 1.0, 0.0, 0.0, 0.5, 0.0], [2.0, 3.0, 1.0, 2.0, 4.0, 1.0], [0, 0, 0, 1, 1, 1], 2),
        ([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0, 0, 1, 1, 2, 2], 3),
        ([1, 1, 1, 0.5, 0, 0, 0], [1.0, 2.0, 1.5, 1.0, 3.0, 1.0, 2.0], [0, 0, 0, 1, 1, 1, 1], 2),
    ],
)
def test_the_permutation_p_value_matches_brute_force(diff, values, codes, k):
    diff, values, codes = np.array(diff, float), np.array(values, float), np.array(codes)
    observed, p = t7_stat.permutation_p(diff, values, codes, k, 0.2, rounds=20_000, seed=7)
    assert observed == pytest.approx(loop_statistic(diff, values, codes, k, 0.2), rel=1e-12)
    exact = brute_force_p(diff, values, codes, k, 0.2)
    # 20,000 draws: the standard error of a share near 0.5 is under 0.004
    assert abs(p - exact) < 0.015, (p, exact)


def test_the_statistic_equals_a_plain_loop_on_random_days():
    rng = np.random.default_rng(5)
    for _ in range(20):
        n, k = 40, 3
        diff = rng.choice([0.0, 0.5, 1.0], n)
        values = rng.uniform(1, 9, n)
        codes = rng.permutation(np.arange(n) % k)
        found = t7_stat.spread_statistic(diff, values, codes, k, 0.1)
        assert found == pytest.approx(loop_statistic(diff, values, codes, k, 0.1), rel=1e-12)


def test_the_p_value_counts_the_observed_order_and_is_never_zero():
    # all 20 loss changes in type 0: only 2 of the 137,846,528,820 assignments reach it, so no
    # permutation of 50 does and p is 1 / 51
    diff = np.array([1.0] * 20 + [0.0] * 20)
    found = t7_stat.permutation_p(diff, np.ones(40), np.repeat([0, 1], 20), 2, 0.2, 50, 11)
    assert found == (pytest.approx(5.0), 1 / 51)


def test_the_p_value_is_one_when_nothing_changes_and_is_repeatable():
    flat = np.zeros(12)
    codes = np.arange(12) % 2
    assert t7_stat.permutation_p(flat, np.ones(12), codes, 2, 0.2, rounds=50, seed=1) == (0.0, 1.0)
    diff = np.array([1.0] * 6 + [0.0] * 6)
    first = t7_stat.permutation_p(diff, np.ones(12), np.repeat([0, 1], 6), 2, 0.2, 500, 3)
    assert first == t7_stat.permutation_p(diff, np.ones(12), np.repeat([0, 1], 6), 2, 0.2, 500, 3)
    # every permutation's statistic is at most the observed one, which only 2 of 924 reach
    assert first[1] < 0.05


def test_an_observed_statistic_of_zero_gives_p_one():
    # both types lose on one day of two at equal values: the observed statistic is exactly 0, and
    # every permuted statistic is at or above it (equal to it on the orders that keep one loss
    # change per type, above it on the others), so every round counts and p is 1
    diff = np.array([1.0, 0.0, 1.0, 0.0])
    found = t7_stat.permutation_p(diff, np.ones(4), np.array([0, 0, 1, 1]), 2, 0.2, 100, 1)
    assert found == (0.0, 1.0)


def test_values_must_be_positive():
    with pytest.raises(ValueError, match="above 0"):
        t7_stat.marginals(np.ones(2), np.array([1.0, 0.0]), np.array([0, 1]), 2, 0.2)


def test_the_seed_of_a_unit_is_fixed_and_differs_between_units():
    assert t7_stat.permutation_seed("caltech") == t7_stat.permutation_seed("caltech")
    assert t7_stat.permutation_seed("caltech") != t7_stat.permutation_seed("jpl")


# --- 6. Holm -------------------------------------------------------------------------------------


def test_holm_by_hand():
    found = t7_stat.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    # sorted 0.01, 0.03, 0.04, 0.5 times 4, 3, 2, 1, running maximum, capped at 1
    assert found == {
        "a": pytest.approx(0.04),
        "c": pytest.approx(0.09),
        "b": pytest.approx(0.09),
        "d": pytest.approx(0.5),
    }
    assert t7_stat.holm({}) == {}
    assert t7_stat.holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}
    assert all(not math.isnan(v) for v in found.values())
