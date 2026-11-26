"""`forecast_average.average_tables` on made-up forecast tables. Written, not run. Each test says
which edit turns it red.

1. The same table twice gives that table bit for bit. Red if the mean is replaced by a sum (every
   number doubles) or by a weighted mean whose weights do not add to one.
2. Two tables: every number cell is the mean of the two (a scale-1 and a scale-3 table give the
   scale-2 table) and lies between the two at every cell (random numbers). Red if a column is left
   out of `AVERAGED` (that column keeps the first table's values).
3. Rows in another order are paired by car and level, and the result has the first table's row
   order. Red if the `keyed.loc[base.index]` line of `_aligned` is removed (cars are then paired by
   position).
4. A NaN in either table is a NaN of the result, and the other table's number is not used. Red if
   `np.mean` is replaced by `np.nanmean`.
5. A table lacking a car, or holding a car the first lacks, is refused. Red if the length check or
   the `isin` check of `_aligned` is removed.
6. Another `rho`, fewer than two tables, a level off the plan's grid: refused. Red if the `rho`
   check, the count check or `check_table` is removed.
7. The inputs are not changed, and there is no weight argument: the function takes the tables and
   `what_if` only. Red if a `weights` parameter is added.
8. Three tables: the plain mean of three.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from fleetreserve.forecast_average import AVERAGED, average_tables
from fleetreserve.forecast_model import (
    KEY_COLUMNS,
    NUMBER_COLUMNS,
    SURVIVAL_COLUMNS,
    TABLE_COLUMNS,
    check_table,
)

CARS = (
    ("U", "A", "2026-01-05", "s1"),
    ("U", "A", "2026-01-05", "s2"),
    ("U", "B", "2026-01-06", "s3"),
)
LEVELS = (0.1, 0.5)


def table(scale: float = 1.0, rho: float = 0.0, levels=LEVELS) -> pd.DataFrame:
    """Three cars at each level; every number is `scale` times a quarter-hour multiple, so means of
    such tables are exact."""
    rows = []
    for k, level in enumerate(levels):
        for i, key in enumerate(CARS):
            row = dict(zip(KEY_COLUMNS, key, strict=True))
            row.update(kappa=level, rho=rho)
            for j, column in enumerate(c for c in NUMBER_COLUMNS if c != "rho"):
                row[column] = scale * (1 + 0.25 * (i + 4 * k + 8 * j))
            rows.append(row)
    return check_table(pd.DataFrame(rows, columns=list(TABLE_COLUMNS)))


def numbers(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(AVERAGED)].to_numpy(dtype=float)


def test_the_same_table_twice_is_that_table_bit_for_bit():
    one = table(1.7)
    pd.testing.assert_frame_equal(average_tables([one, one.copy()]), one, check_exact=True)


def test_two_tables_give_the_mean_of_every_number_and_lie_between_them():
    pd.testing.assert_frame_equal(
        average_tables([table(1.0), table(3.0)]), table(2.0), check_exact=True
    )
    rng = np.random.default_rng(7)
    a, b = table(), table()
    for column in AVERAGED:
        a[column] = rng.uniform(0, 12, len(a))
        b[column] = rng.uniform(0, 12, len(b))
    both = average_tables([a, b])
    low, high = np.minimum(numbers(a), numbers(b)), np.maximum(numbers(a), numbers(b))
    assert (numbers(both) >= low).all() and (numbers(both) <= high).all()
    assert (numbers(both) != numbers(a)).any() and (numbers(both) != numbers(b)).any()
    for column in AVERAGED:  # every column, the survival columns and the median among them
        assert not np.array_equal(both[column], a[column]), column
    assert set(SURVIVAL_COLUMNS) <= set(AVERAGED) and "d_median" in AVERAGED


def test_rows_in_another_order_are_paired_by_car_and_level():
    a, b = table(1.0), table(3.0)
    shuffled = b.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert not shuffled["session_id"].equals(b["session_id"])
    pd.testing.assert_frame_equal(average_tables([a, shuffled]), table(2.0), check_exact=True)
    first_shuffled = average_tables([shuffled, a])
    pd.testing.assert_frame_equal(first_shuffled["session_id"].to_frame(), shuffled[["session_id"]])
    expected = table(2.0).set_index(["session_id", "kappa"])
    got = first_shuffled.set_index(["session_id", "kappa"]).loc[expected.index]
    pd.testing.assert_frame_equal(got[list(AVERAGED)], expected[list(AVERAGED)], check_exact=True)


def test_a_nan_in_either_table_is_a_nan_of_the_result():
    a, b = table(1.0), table(3.0)
    a.loc[1, "d_hat"] = np.nan
    b.loc[2, "survival_h3"] = np.nan
    b.loc[4, "r_hat"] = np.nan
    both = average_tables([a, b])
    assert np.isnan(both.loc[1, "d_hat"]) and np.isnan(both.loc[2, "survival_h3"])
    assert np.isnan(both.loc[4, "r_hat"])
    assert np.isnan(numbers(both)).sum() == 3  # nothing else became NaN, nothing was filled
    swapped = average_tables([b, a])
    assert np.isnan(swapped.loc[1, "d_hat"]) and np.isnan(swapped.loc[2, "survival_h3"])


def test_a_missing_or_foreign_car_is_refused():
    a, b = table(1.0), table(3.0)
    with pytest.raises(ValueError, match="same cars"):
        average_tables([a, b.iloc[:-1]])
    with pytest.raises(ValueError, match="same cars"):
        average_tables([a.iloc[:-1], b])
    foreign = b.copy()
    foreign.loc[0, "session_id"] = "s9"
    with pytest.raises(ValueError, match="same cars"):
        average_tables([a, foreign])
    other_level = table(3.0, levels=(0.1, 0.4))
    with pytest.raises(ValueError, match="same cars"):
        average_tables([a, other_level])


def test_another_rho_one_table_or_a_level_off_the_grid_is_refused():
    with pytest.raises(ValueError, match="rho"):
        average_tables([table(1.0), table(3.0, rho=0.1)])
    with pytest.raises(ValueError, match="at least two"):
        average_tables([table(1.0)])
    with pytest.raises(ValueError, match="kappa"):
        average_tables([table(1.0, levels=(0.1, 0.25)), table(3.0, levels=(0.1, 0.25))])


def test_the_inputs_are_not_changed_and_no_weight_can_be_given():
    a, b = table(1.0), table(3.0)
    a_before, b_before = a.copy(), b.copy()
    average_tables([a, b])
    pd.testing.assert_frame_equal(a, a_before, check_exact=True)
    pd.testing.assert_frame_equal(b, b_before, check_exact=True)
    assert list(inspect.signature(average_tables).parameters) == ["tables", "what_if"]


def test_three_tables_give_the_plain_mean_of_three():
    got = average_tables([table(1.0), table(2.0), table(6.0)])
    pd.testing.assert_frame_equal(got, table(3.0), check_exact=True)
