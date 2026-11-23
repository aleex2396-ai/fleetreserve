"""The seeded split of a unit's days into calibration days and test days (plan section 6, protocol).

"Per unit (a held-out unit, and a development unit in each of its seeded splits), 100 seeded
random counting days calibrate, or half of the unit's counting days (rounded down) when it has
fewer than 200, and the rest test, one draw for all policies"

How many days calibrate is a rule of the unit's number of days (`calibration_count`): the matrix's
`n_calibration` days when the unit has at least `half_below` days, else half of them rounded down.
`half_below` null means no rule: `n_calibration` for every unit. A unit is split when that count
leaves at least one calibration day and one test day (`is_split`); whether the calibration then
offers or abstains is the calibration's own rule, not this module's.

The draw is a function of the unit name, the seed and the set of dates, and of nothing else: not
of the order the dates are given in, not of the policy. Every date gets a pseudo-random rank from
a SHA-256 hash of (unit name, seed, date text); the `n_calibration` dates with the smallest ranks
calibrate and the others test. A hash is used rather than a random-number generator so that the
split does not change with a library version (reading: "seeded random days" means a seeded
pseudo-random subset; the alternative is a generator permutation).

Dates are any values with a stable text form and an order, for example `datetime.date` or ISO
strings; the text form is `str(date)`. Both returned lists are in date order.
"""

import hashlib
import json
from collections.abc import Iterable
from typing import NamedTuple

N_CALIBRATION = 100
# The unit size under which half of the days calibrate (plan section 6).
HALF_BELOW = 200
# The smallest unit the split accepts by default: 100 calibration days and at least one test day.
MIN_DAYS = 101


class DaySplit(NamedTuple):
    calibration: tuple
    test: tuple


def calibration_count(n_days: int, n_calibration: int, half_below: int | None) -> int:
    """The number of calibration days of a unit with `n_days` days: `n_calibration` when
    `half_below` is null or `n_days` >= `half_below`, else `n_days` // 2. The one place the rule is
    written; the count is not checked against `n_days` (see `is_split`)."""
    if half_below is not None and n_days < half_below:
        return n_days // 2
    return n_calibration


def is_split(n_days: int, count: int) -> bool:
    """Whether `count` calibration days leave at least one calibration day and one test day."""
    return 1 <= count <= n_days - 1


def _rank(unit: str, seed: int, date) -> tuple[bytes, str]:
    text = str(date)
    digest = hashlib.sha256(json.dumps([unit, seed, text]).encode("utf-8")).digest()
    return digest, text


def ranked(unit: str, seed: int, dates: Iterable) -> list:
    """`dates` in the order of their hash ranks, the smallest first: the days a draw of any count
    takes are the first of this list."""
    return sorted(dates, key=lambda d: _rank(unit, seed, d))


def split_days(
    unit: str,
    seed: int,
    dates: Iterable,
    *,
    n_calibration: int = N_CALIBRATION,
    min_days: int = MIN_DAYS,
) -> DaySplit:
    """Split `dates` into `n_calibration` calibration dates and the rest.

    Raises ValueError when the unit has fewer than `min_days` dates, when `min_days` is below
    `n_calibration` + 1 (no test day would be left), when a date repeats, or when `seed` is not an
    integer.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if n_calibration < 1 or min_days < n_calibration + 1:
        raise ValueError("min_days must be at least n_calibration + 1, and n_calibration >= 1")
    ordered = sorted(dates)
    if len(set(ordered)) != len(ordered):
        raise ValueError("a date appears more than once")
    if len(ordered) < min_days:
        raise ValueError(
            f"unit {unit!r} has {len(ordered)} days, fewer than the minimum {min_days}"
        )
    chosen = set(ranked(unit, seed, ordered)[:n_calibration])
    calibration = tuple(d for d in ordered if d in chosen)
    test = tuple(d for d in ordered if d not in chosen)
    return DaySplit(calibration, test)
