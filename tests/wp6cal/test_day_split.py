"""The seeded split of a unit's days into calibration and test days (plan section 6, protocol)."""

import datetime as dt
import random

import pytest

from fleetreserve import day_split


def make_dates(n, start=dt.date(2019, 5, 1)):
    return [start + dt.timedelta(days=k) for k in range(n)]


def test_fifty_calibration_days_and_the_rest_test_with_no_overlap():
    dates = make_dates(130)
    split = day_split.split_days("nrel-after", 7, dates, n_calibration=50, min_days=51)
    assert len(split.calibration) == 50
    assert len(split.test) == 80
    assert set(split.calibration).isdisjoint(split.test)
    assert set(split.calibration) | set(split.test) == set(dates)
    assert list(split.calibration) == sorted(split.calibration)
    assert list(split.test) == sorted(split.test)


def test_deterministic_and_independent_of_the_order_of_the_dates():
    dates = make_dates(120)
    first = day_split.split_days("u", 1, dates)
    again = day_split.split_days("u", 1, dates)
    shuffled = dates[:]
    random.Random(5).shuffle(shuffled)
    assert shuffled != dates
    reordered = day_split.split_days("u", 1, shuffled)
    reversed_ = day_split.split_days("u", 1, reversed(dates))
    assert first == again == reordered == reversed_


def test_iso_strings_and_dates_give_the_same_split():
    dates = make_dates(120)
    as_text = [d.isoformat() for d in dates]
    by_date = day_split.split_days("u", 3, dates)
    by_text = day_split.split_days("u", 3, as_text)
    assert [d.isoformat() for d in by_date.calibration] == list(by_text.calibration)


def test_a_different_seed_or_unit_gives_a_different_split():
    dates = make_dates(120)
    base = day_split.split_days("u", 1, dates)
    assert day_split.split_days("u", 2, dates).calibration != base.calibration
    assert day_split.split_days("v", 1, dates).calibration != base.calibration


def test_split_is_a_fixed_value():
    # Guards against a silent change of the hash or the ranking: the first calibration dates of
    # a fixed split. The expected values were computed once with this module.
    split = day_split.split_days("fixed-unit", 2026, make_dates(100), n_calibration=50, min_days=51)
    assert [d.isoformat() for d in split.calibration[:3]] == [
        "2019-05-01",
        "2019-05-03",
        "2019-05-04",
    ]


def test_the_calibration_days_are_spread_over_the_whole_period():
    # A random subset, not the first 50 days: both halves of the period hold calibration days.
    dates = make_dates(200)
    split = day_split.split_days("u", 11, dates, n_calibration=50, min_days=51)
    first_half = [d for d in split.calibration if d < dates[100]]
    assert 10 <= len(first_half) <= 40


def test_refuses_below_the_minimum():
    with pytest.raises(ValueError, match="fewer than the minimum 51"):
        day_split.split_days("u", 1, make_dates(50), n_calibration=50, min_days=51)
    with pytest.raises(ValueError, match="fewer than the minimum 100"):
        day_split.split_days("u", 1, make_dates(99), n_calibration=50, min_days=100)
    # at the minimum it works, with one test day
    split = day_split.split_days("u", 1, make_dates(51), n_calibration=50, min_days=51)
    assert (len(split.calibration), len(split.test)) == (50, 1)
    assert (
        len(day_split.split_days("u", 1, make_dates(100), n_calibration=50, min_days=100).test)
        == 50
    )


def test_the_minimum_cannot_be_set_below_what_leaves_a_test_day():
    assert day_split.MIN_DAYS == 101
    with pytest.raises(ValueError, match="min_days"):
        day_split.split_days("u", 1, make_dates(60), n_calibration=50, min_days=50)
    with pytest.raises(ValueError, match="min_days"):
        day_split.split_days("u", 1, make_dates(60), min_days=5, n_calibration=5)


def test_repeated_date_or_bad_seed_refused():
    dates = make_dates(60)
    with pytest.raises(ValueError, match="more than once"):
        day_split.split_days("u", 1, dates + [dates[0]])
    for seed in ("1", 1.5, True):
        with pytest.raises(ValueError, match="seed"):
            day_split.split_days("u", seed, dates)
