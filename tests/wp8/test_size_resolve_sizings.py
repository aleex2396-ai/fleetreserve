"""The two sizings read from the re-solve's losses at size 1 (`size_resolve_sizings`), on made-up
loss tables at the grid points (no replay), each case small enough to compute by hand.

1. The re-solve oracle: a day that never misses a request gets 1; one that misses above 0.3 gets
   0.3; one that misses at 0.05 only (a loss that does not rise with the scale) gets 0; a driver
   loss is not read; a day that does not offer has no size.
2. The plain-mean rule of a type: two days, one losing 1 above 0.4: the mean is 0.5 above 0.4, so
   0.4 at alpha 0.1 and 1 at alpha 0.5; the driver mean is held to its own alpha; no day gives 0;
   a loss that falls again at a larger scale still counts there (the running maximum).
3. Day types: on 25 offering calibration days (12 Mondays, 10 Tuesdays, 3 Wednesdays) Monday and
   Tuesday are kept, Wednesday and a test Thursday are "other"; a day that does not offer has none;
   test days do not count towards the 10 days a type needs.
4. The multiplier: Mondays lose above 0.4, the others never; tier B gives Monday 0.4 and every
   other type 1; a test day of a type with no calibration day takes the scale of all calibration
   days; with no offering calibration day every size is 0; tier A holds the driver loss too.
"""

import datetime
import math

import numpy as np

from fleetreserve import t7_stat
from fleetreserve.size_resolve import GRID
from fleetreserve.size_resolve_sizings import (
    day_types,
    failure_free_sizes,
    mean_rule_scale,
    type_sizes,
)

MONDAY = datetime.date(2020, 1, 6)


def grid_table(slot_rows, driver_rows=None) -> dict:
    out = {}
    for d, row in enumerate(slot_rows):
        for k, g in enumerate(GRID):
            driver = 0.0 if driver_rows is None else driver_rows[d][k]
            out[(d, float(g))] = (float(row[k]), float(driver))
    return out


def loses_above(t: float) -> list:
    return [1.0 if g > t else 0.0 for g in GRID]


def test_the_re_solve_oracle():
    only_first = [1.0] + [0.0] * (len(GRID) - 1)
    rows = [loses_above(2.0), loses_above(0.3), only_first, loses_above(2.0)]
    driver = [[0.0] * len(GRID), [0.0] * len(GRID), [0.0] * len(GRID), [0.5] * len(GRID)]
    found = failure_free_sizes(grid_table(rows, driver), [True, True, True, True])
    assert list(found) == [1.0, 0.3, 0.0, 1.0]
    found = failure_free_sizes(grid_table(rows), [True, False, True, True])
    assert math.isnan(found[1]) and found[0] == 1.0


def test_the_plain_mean_rule():
    slot = [loses_above(0.4), loses_above(2.0)]
    assert mean_rule_scale(slot, np.zeros((2, len(GRID))), 0.1, 1.0) == 0.4
    assert mean_rule_scale(slot, np.zeros((2, len(GRID))), 0.5, 1.0) == 1.0
    driver = [[0.2] * len(GRID), [0.0] * len(GRID)]  # mean 0.1 at every scale
    assert mean_rule_scale(slot, driver, 0.5, 0.05) == 0.0
    assert mean_rule_scale(slot, driver, 0.5, 0.1) == 1.0
    assert mean_rule_scale(np.zeros((0, len(GRID))), np.zeros((0, len(GRID))), 0.1, 1.0) == 0.0
    # a loss that does not rise with the scale counts from where it first appears: one day loses
    # at 0.3 only, the other at 0.5 only; at alpha 0.5 the running maximum stops at 0.4
    at = [[1.0 if g == point else 0.0 for g in GRID] for point in (0.3, 0.5)]
    assert mean_rule_scale(at, np.zeros((2, len(GRID))), 0.5, 1.0) == 0.4


def week_days(counts) -> list:
    """Raw types: `counts` = (Mondays, Tuesdays, Wednesdays), in that order."""
    names = []
    for k, n in enumerate(counts):
        names += [t7_stat.raw_type((MONDAY + datetime.timedelta(days=k)).isoformat(), False)] * n
    return names


def test_day_types_pool_on_the_calibration_days():
    raw = [*week_days((12, 10, 3)), "Thursday", "Monday", None]
    offering = [True] * 27 + [False]
    found = day_types(raw, offering, calibration=list(range(25)))
    assert found[:12] == ["Monday"] * 12 and found[12:22] == ["Tuesday"] * 10
    assert found[22:25] == [t7_stat.OTHER] * 3
    assert found[25:] == [t7_stat.OTHER, "Monday", None]
    # nine calibration Tuesdays and a tenth among the test days: still "other"
    raw = [*week_days((12, 9, 0)), "Tuesday"]
    found = day_types(raw, [True] * 22, calibration=list(range(21)))
    assert found[12:] == [t7_stat.OTHER] * 10


def test_the_multiplier():
    raw = [*week_days((12, 10, 3)), "Monday", "Tuesday", "Thursday"]
    rows = [loses_above(0.4)] * 12 + [loses_above(2.0)] * 13 + [loses_above(2.0)] * 3
    losses = grid_table(rows)
    offering = np.ones(28, dtype=bool)
    sizes, scales = type_sizes(losses, raw, offering, list(range(25)), "B")
    assert scales == {"Monday": 0.4, "Tuesday": 1.0, t7_stat.OTHER: 1.0}
    assert list(sizes) == [0.4] * 12 + [1.0] * 13 + [0.4, 1.0, 1.0]
    # without the three Wednesdays, "other" has no calibration day: all 22 together, of which 12
    # lose above 0.4 (mean 12/22 > 0.1), so 0.4
    raw = [*week_days((12, 10, 0)), "Thursday"]
    losses = grid_table([loses_above(0.4)] * 12 + [loses_above(2.0)] * 11)
    sizes, scales = type_sizes(losses, raw, np.ones(23, dtype=bool), list(range(22)), "B")
    assert scales[t7_stat.OTHER] == 0.4 and sizes[22] == 0.4
    sizes, _ = type_sizes(losses, raw, np.ones(23, dtype=bool), [], "B")
    assert list(sizes) == [0.0] * 23


def test_tier_a_holds_the_driver_loss_of_a_type():
    raw = week_days((12, 10, 0))
    slot = [loses_above(2.0)] * 22
    driver = [[0.1] * len(GRID)] * 12 + [[0.0] * len(GRID)] * 10  # Mondays: 0.1 at every scale
    losses = grid_table(slot, driver)
    _, scales = type_sizes(losses, raw, np.ones(22, dtype=bool), list(range(22)), "A")
    assert scales == {"Monday": 0.0, "Tuesday": 1.0}
    _, scales = type_sizes(losses, raw, np.ones(22, dtype=bool), list(range(22)), "B")
    assert scales == {"Monday": 1.0, "Tuesday": 1.0}
