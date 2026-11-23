"""The seeded sample of development days: exactly 50, proportional with at least one per unit,
deterministic and independent of the order the days are listed in."""

import random

import pytest

from fleetreserve.block_sample import (
    DEFAULT_SEED,
    SAMPLE_SIZE,
    allocate,
    day_text,
    draw,
    sample_days,
)


def _days(counts):
    return {
        unit: [day_text(site, f"2020-{1 + k // 28:02d}-{1 + k % 28:02d}") for k in range(n)]
        for unit, (n, site) in counts.items()
    }


COUNTS = {"a-unit": (200, "S"), "b-unit": (400, "S"), "c-unit": (800, "S")}


def test_the_sample_holds_exactly_50_days_in_proportion_to_each_units_days():
    shares = allocate({"a": 200, "b": 400, "c": 800})
    # quotas 7.14, 14.29, 28.57: floors 7 + 14 + 28 = 49, the largest remainder (c) takes the 50th
    assert shares == {"a": 7, "b": 14, "c": 29}
    chosen = sample_days(_days(COUNTS))
    assert SAMPLE_SIZE == 50
    assert sum(len(v) for v in chosen.values()) == 50
    assert {u: len(v) for u, v in chosen.items()} == {"a-unit": 7, "b-unit": 14, "c-unit": 29}


def test_every_unit_has_at_least_one_day_and_the_total_stays_50():
    shares = allocate({"tiny": 3, "big": 1000, "other": 1000})
    # quotas 0.08, 24.96, 24.96: the minimum gives tiny its day, floors 1 + 24 + 24 = 49, and the
    # tie for the 50th goes to the name that sorts first
    assert shares == {"tiny": 1, "big": 25, "other": 24}


def test_a_unit_smaller_than_its_share_gives_all_its_days_and_the_rest_fill_up():
    shares = allocate({"a": 2, "b": 2, "c": 46})
    assert shares == {"a": 2, "b": 2, "c": 46}
    shares = allocate({"a": 1, "b": 1, "c": 100})
    assert shares == {"a": 1, "b": 1, "c": 48}


def test_fewer_than_50_days_in_all_is_a_refusal():
    with pytest.raises(ValueError, match="fewer than the sample size 50"):
        allocate({"a": 20, "b": 20, "c": 9})
    with pytest.raises(ValueError, match="fewer than the sample size 50"):
        sample_days(_days({"a": (20, "S"), "b": (20, "S"), "c": (9, "S")}))


def test_a_unit_without_a_counting_day_is_a_refusal():
    with pytest.raises(ValueError, match="no counting day"):
        allocate({"a": 0, "b": 100})


def test_the_draw_is_deterministic_and_does_not_depend_on_the_order_of_the_days():
    days = _days(COUNTS)
    first = sample_days(days, DEFAULT_SEED)
    assert sample_days(days, DEFAULT_SEED) == first
    shuffled = {u: random.Random(3).sample(v, len(v)) for u, v in reversed(list(days.items()))}
    assert sample_days(shuffled, DEFAULT_SEED) == first
    for unit, chosen in first.items():
        assert list(chosen) == sorted(chosen)
        assert set(chosen) <= set(days[unit])
        assert len(set(chosen)) == len(chosen)


def test_another_seed_draws_other_days():
    days = _days(COUNTS)
    assert sample_days(days, 1) != sample_days(days, 2)


def test_a_units_draw_does_not_depend_on_the_other_units_days():
    days = _days(COUNTS)
    alone = sample_days({"a-unit": days["a-unit"], "b-unit": days["b-unit"]}, 5)
    # the share of a-unit differs between the two samples, so compare the draw by its ranking:
    # the days drawn with the smaller share are drawn again with the larger one
    both = sample_days(days, 5)
    small, large = sorted((alone["a-unit"], both["a-unit"]), key=len)
    assert set(small) <= set(large)


def test_a_draw_of_all_days_is_listed_in_day_order_whatever_the_input_order():
    days = ["B 2020-01-02", "A 2020-01-03", "A 2020-01-01"]
    assert draw("u", 1, days, 3) == ("A 2020-01-01", "A 2020-01-03", "B 2020-01-02")
    assert draw("u", 1, days, 9) == ("A 2020-01-01", "A 2020-01-03", "B 2020-01-02")
