"""The paired differences between comparison methods (`compare_paired`) on hand-worked tables.

1. One pair on five splits, worked by hand: the mean difference, its 5th and 95th percentiles
   (linear interpolation between order statistics), the shares above, equal to and below 0, the
   two mean offers, the share of the reference's mean offer and the verdict.
   A difference of 5e-4 kWh is above or below 0, not equal to it: "equal to 0" is exact
   equality.
2. Ties: a pair whose offers are equal on every split has every difference 0, the share equal to
   0 is 1 and the verdict is "not separated".
3. A reference whose offer is 0 on every paired split: the share of its mean offer is None (the
   summary writes "no offer"), never a division by zero or an infinity; the other figures stand.
4. A missing value (no row for a split, or a row whose offer is empty or not finite) leaves that
   split out of the pair and is counted; it is never read as 0. With no split paired, every
   figure and the verdict are None.
5. Planted cases: a method ahead of its reference on every split is "ahead"; one behind on every
   split is "behind"; one whose sign flips between splits is "not separated".
6. The pairs: the reference table pairs every other method with `re-solve`, in the run's order;
   the second table pairs every two methods other than `re-solve`, the earlier as reference; a
   run without `re-solve` has an empty reference table.
7. `paired` over rows: the differences are taken on the same split (rows given in a shuffled
   order, with other units, variants and tiers mixed in, give the hand-worked figures); a row
   listed twice is refused.
"""

import math
import random

import pytest

from fleetreserve import compare_paired

OFFER = "offer_kwh_per_counting_test_day"
SEEDS = (1, 2, 3, 4, 5)


def _stats(method, reference, seeds=SEEDS):
    return compare_paired.pair_statistics(
        dict(zip(seeds, method, strict=True)), dict(zip(seeds, reference, strict=True)), seeds
    )


# --- 1. hand-worked ------------------------------------------------------------------------------


def test_one_pair_worked_by_hand():
    # differences 0, 1, 2, 3, 4 (sorted); 5th percentile at position 0.05 * 4 = 0.2 -> 0.2,
    # 95th at position 3.8 -> 3.8.
    found = _stats([1.0, 3.0, 2.0, 5.0, 4.0], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert found["splits_paired"] == 5
    assert found["splits_left_out_value_missing"] == 0
    assert found["mean_difference_kwh_per_test_day"] == pytest.approx(2.0, abs=1e-12)
    assert found["difference_p5_kwh_per_test_day"] == pytest.approx(0.2, abs=1e-12)
    assert found["difference_p95_kwh_per_test_day"] == pytest.approx(3.8, abs=1e-12)
    assert found["share_of_paired_splits_above_0"] == pytest.approx(0.8)
    assert found["share_of_paired_splits_equal_0"] == pytest.approx(0.2)
    assert found["share_of_paired_splits_below_0"] == 0.0
    assert found["method_mean_offer_kwh_per_test_day"] == pytest.approx(3.0)
    assert found["reference_mean_offer_kwh_per_test_day"] == pytest.approx(1.0)
    assert found["mean_difference_share_of_reference_mean_offer"] == pytest.approx(2.0)
    assert found["verdict"] == "ahead"


def test_equal_to_0_is_exact_equality_not_closeness():
    # differences 1, 2, 5e-4, 0.3, 0.7, 0, 0, 0, -5e-4, -1 (offers 2 + d against 2): 5 above, 3
    # exactly equal to 0 and 2 below. The two differences of 5e-4 kWh are not 0: counted as
    # "close to 0" they would make the share equal to 0 0.5 and the three shares sum to 1.4.
    differences = [1.0, 2.0, 5e-4, 0.3, 0.7, 0.0, 0.0, 0.0, -5e-4, -1.0]
    seeds = tuple(range(1, 11))
    found = _stats([2.0 + d for d in differences], [2.0] * 10, seeds)
    assert found["share_of_paired_splits_above_0"] == 0.5
    assert found["share_of_paired_splits_equal_0"] == 0.3
    assert found["share_of_paired_splits_below_0"] == 0.2
    shares = [found[f"share_of_paired_splits_{k}_0"] for k in ("above", "equal", "below")]
    assert sum(shares) == pytest.approx(1.0, abs=1e-12)


def test_the_percentiles_are_of_the_differences_not_of_the_offers():
    # offers of the method 10, 0, 10, 0, 10 and of the reference 9, 1, 9, 1, 9: the differences
    # are 1, -1, 1, -1, 1; the offers' own percentiles would be quite different.
    found = _stats([10.0, 0.0, 10.0, 0.0, 10.0], [9.0, 1.0, 9.0, 1.0, 9.0])
    assert found["difference_p5_kwh_per_test_day"] == pytest.approx(-1.0)
    assert found["difference_p95_kwh_per_test_day"] == pytest.approx(1.0)
    assert found["mean_difference_kwh_per_test_day"] == pytest.approx(0.2)
    assert found["share_of_paired_splits_above_0"] == pytest.approx(0.6)
    assert found["share_of_paired_splits_below_0"] == pytest.approx(0.4)
    assert found["verdict"] == "not separated"


# --- 2. ties -------------------------------------------------------------------------------------


def test_equal_offers_on_every_split_are_not_separated():
    found = _stats([2.5, 0.0, 1.0, 1.0, 3.0], [2.5, 0.0, 1.0, 1.0, 3.0])
    assert found["mean_difference_kwh_per_test_day"] == 0.0
    assert found["difference_p5_kwh_per_test_day"] == 0.0
    assert found["difference_p95_kwh_per_test_day"] == 0.0
    assert found["share_of_paired_splits_equal_0"] == 1.0
    assert found["share_of_paired_splits_above_0"] == 0.0
    assert found["share_of_paired_splits_below_0"] == 0.0
    assert found["verdict"] == "not separated"


# --- 3. a reference with offer 0 -----------------------------------------------------------------


def test_a_reference_that_offers_nothing_has_no_share():
    found = _stats([0.5, 0.0, 1.0, 0.5, 0.0], [0.0] * 5)
    assert found["reference_mean_offer_kwh_per_test_day"] == 0.0
    assert found["mean_difference_share_of_reference_mean_offer"] is None
    assert found["mean_difference_kwh_per_test_day"] == pytest.approx(0.4)
    assert found["difference_p5_kwh_per_test_day"] == pytest.approx(0.0)
    assert found["verdict"] == "not separated"
    assert all(not (isinstance(v, float) and math.isinf(v)) for v in found.values()), "no infinity"


# --- 4. missing values ---------------------------------------------------------------------------


def test_a_missing_value_is_left_out_and_counted_never_read_as_zero():
    method = {1: 3.0, 2: None, 3: 5.0, 4: float("nan"), 5: 4.0}
    reference = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}  # seed 5 has no row
    found = compare_paired.pair_statistics(method, reference, SEEDS)
    # paired: seeds 1 and 3 only, differences 2 and 4
    assert found["splits_paired"] == 2
    assert found["splits_left_out_value_missing"] == 3
    assert found["mean_difference_kwh_per_test_day"] == pytest.approx(3.0)
    assert found["difference_p5_kwh_per_test_day"] == pytest.approx(2.1)
    assert found["difference_p95_kwh_per_test_day"] == pytest.approx(3.9)
    assert found["reference_mean_offer_kwh_per_test_day"] == pytest.approx(1.0)
    assert found["share_of_paired_splits_above_0"] == 1.0
    assert found["verdict"] == "ahead"


def test_no_paired_split_gives_no_figures():
    found = compare_paired.pair_statistics({1: None, 2: 1.0}, {1: 1.0}, (1, 2))
    assert found["splits_paired"] == 0 and found["splits_left_out_value_missing"] == 2
    for key, value in found.items():
        if key not in ("splits_paired", "splits_left_out_value_missing"):
            assert value is None, key


# --- 5. planted cases ----------------------------------------------------------------------------


def test_planted_ahead_behind_and_a_flipping_sign():
    rng = random.Random(7)
    reference = [rng.uniform(1, 3) for _ in range(40)]
    seeds = tuple(range(1, 41))
    ahead = [r + rng.uniform(0.01, 0.2) for r in reference]
    behind = [r - rng.uniform(0.01, 0.2) for r in reference]
    flips = [r + (0.1 if k % 2 else -0.1) for k, r in enumerate(reference)]
    assert _stats(ahead, reference, seeds)["verdict"] == "ahead"
    assert _stats(behind, reference, seeds)["verdict"] == "behind"
    found = _stats(flips, reference, seeds)
    assert found["verdict"] == "not separated"
    assert found["share_of_paired_splits_above_0"] == pytest.approx(0.5)
    assert found["share_of_paired_splits_below_0"] == pytest.approx(0.5)
    # swapping method and reference turns ahead into behind
    assert _stats(reference, ahead, seeds)["verdict"] == "behind"


def test_the_verdict_rule_at_its_edges():
    assert compare_paired.verdict(1e-12, 5.0) == "ahead"
    assert compare_paired.verdict(0.0, 5.0) == "not separated"
    assert compare_paired.verdict(-5.0, -1e-12) == "behind"
    assert compare_paired.verdict(-5.0, 0.0) == "not separated"
    assert compare_paired.VERDICTS == ("ahead", "behind", "not separated")


# --- 6. the pairs --------------------------------------------------------------------------------


def test_the_pairs_of_the_two_tables():
    against, other = compare_paired.pairs(["re-solve", "fixed-plan", "edf", "llf"])
    assert against == [("fixed-plan", "re-solve"), ("edf", "re-solve"), ("llf", "re-solve")]
    assert other == [("edf", "fixed-plan"), ("llf", "fixed-plan"), ("llf", "edf")]
    against, other = compare_paired.pairs(["llf", "re-solve", "edf"])
    assert against == [("llf", "re-solve"), ("edf", "re-solve")]
    assert other == [("edf", "llf")]
    against, other = compare_paired.pairs(["edf", "llf", "asa"])
    assert against == []
    assert other == [("llf", "edf"), ("asa", "edf"), ("asa", "llf")]


# --- 7. over rows --------------------------------------------------------------------------------


def _row(unit, variant, tier, method, seed, offer):
    return {
        "unit": unit,
        "variant": variant,
        "tier": tier,
        "method": method,
        "seed": seed,
        OFFER: offer,
    }


def test_paired_over_rows_takes_each_difference_on_the_same_split():
    method = [1.0, 3.0, 2.0, 5.0, 4.0]
    reference = [1.0, 1.0, 1.0, 1.0, 1.0]
    rows = [_row("u", "forecast", "A", "edf", s, v) for s, v in zip(SEEDS, method, strict=True)]
    rows += [
        _row("u", "forecast", "A", "re-solve", s, v) for s, v in zip(SEEDS, reference, strict=True)
    ]
    # noise: another unit, variant and tier with other numbers
    rows += [_row("v", "forecast", "A", "edf", s, 100.0 + s) for s in SEEDS]
    rows += [_row("v", "forecast", "A", "re-solve", s, 0.0) for s in SEEDS]
    rows += [_row("u", "oracle-departure", "A", "edf", s, -7.0) for s in SEEDS]
    rows += [_row("u", "oracle-departure", "A", "re-solve", s, 7.0) for s in SEEDS]
    rows += [_row("u", "forecast", "B", m, s, 9.0) for m in ("edf", "re-solve") for s in SEEDS]
    random.Random(3).shuffle(rows)
    found = compare_paired.paired(
        rows, ["u", "v"], ["forecast", "oracle-departure"], ["A", "B"], ["re-solve", "edf"], SEEDS
    )
    (row,) = found["u"]["forecast"]["A"]["against_re_solve"]
    assert row["method"] == "edf" and row["reference"] == "re-solve"
    assert row == {"method": "edf", "reference": "re-solve", **_stats(method, reference)}
    assert found["u"]["forecast"]["A"]["other_pairs"] == []
    (row,) = found["u"]["oracle-departure"]["A"]["against_re_solve"]
    assert row["verdict"] == "behind" and row["mean_difference_kwh_per_test_day"] == -14.0
    (row,) = found["u"]["forecast"]["B"]["against_re_solve"]
    assert row["share_of_paired_splits_equal_0"] == 1.0
    (row,) = found["v"]["forecast"]["A"]["against_re_solve"]
    assert row["mean_difference_share_of_reference_mean_offer"] is None
    (row,) = found["v"]["oracle-departure"]["A"]["against_re_solve"]
    assert row["splits_paired"] == 0 and row["splits_left_out_value_missing"] == 5


def test_a_row_listed_twice_is_refused():
    rows = [_row("u", "forecast", "A", "edf", 1, 1.0), _row("u", "forecast", "A", "edf", 1, 2.0)]
    with pytest.raises(ValueError, match="listed twice"):
        compare_paired.paired(rows, ["u"], ["forecast"], ["A"], ["edf"], (1,))
