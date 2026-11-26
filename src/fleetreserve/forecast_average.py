"""The equal-weight average of forecast tables (a trial made after the sealed run, `forecast_mix`).

Two departure models each give a forecast table (`forecast_model`: one row per car and level, with
the quantiles of departure, rate, room and needed energy). `average_tables` gives the table whose
every number is the plain mean of the same cell in each input, so that at each level the average
of the two models' quantiles is read as that level's quantile. There is no weight to choose: the
weights are one over the number of tables, and the function takes none.

Rules, each fixed here and not a setting of a run:

- The tables must hold the same cars at the same levels (the key columns and `kappa`). A table
  with a car or a level the other lacks is refused (`ValueError`): a model states "no forecast"
  with NaN, never by leaving a car out (`forecast_model`), so a missing row is a fault and is
  never filled from the other table.
- A cell that is NaN in any input is NaN in the average. A NaN means that the model gives no
  forecast for that car (plan section 5: a car with no usable forecast is left unsteered), and the
  mean of a forecast and no forecast is not a forecast; the other table's value is never taken in
  its place. So a survival column that one model leaves NaN stays NaN, and the gate keeps the
  bound of the hour before, as it does for that model alone.
- `rho` is not averaged: it must be the same in every table and is carried over.
- With the same table twice, the average is that table bit for bit (`(a + a) / 2 == a`); at every
  cell the average lies between the inputs.

The inputs are not changed. The result has the columns, the order of columns and the order of rows
of the first table.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fleetreserve.forecast_model import KEY_COLUMNS, NUMBER_COLUMNS, TABLE_COLUMNS, check_table

CELL_KEY = (*KEY_COLUMNS, "kappa")
AVERAGED = tuple(c for c in NUMBER_COLUMNS if c != "rho")


def _aligned(table: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """`table` indexed by the cell key, its rows in the order of `base`'s; `ValueError` when the
    two do not hold the same cells."""
    keyed = table.set_index(list(CELL_KEY))
    if len(keyed) != len(base) or not keyed.index.isin(base.index).all():
        raise ValueError(
            "the forecast tables do not hold the same cars at the same levels "
            f"({len(base)} and {len(keyed)} rows); a missing car is never filled from the other"
        )
    return keyed.loc[base.index]


def average_tables(tables: Sequence[pd.DataFrame], what_if: bool = False) -> pd.DataFrame:
    """The mean, cell by cell and with equal weights, of the forecast tables `tables` (module
    text); raises `ValueError` for fewer than two tables, a table `forecast_model.check_table`
    refuses, other cars or levels than the first table's, or another `rho`."""
    if len(tables) < 2:
        raise ValueError("an average needs at least two forecast tables")
    checked = [check_table(t, what_if) for t in tables]
    first = checked[0]
    base = first.set_index(list(CELL_KEY))
    aligned = [base, *(_aligned(t, base) for t in checked[1:])]
    rho = base["rho"].to_numpy()
    if any(not np.array_equal(rho, a["rho"].to_numpy()) for a in aligned[1:]):
        raise ValueError("the forecast tables were made for different rho")
    out = base.copy()
    for column in AVERAGED:
        stacked = np.stack([a[column].to_numpy(dtype=float) for a in aligned])
        out[column] = np.mean(stacked, axis=0)  # NaN in any input gives NaN
    return check_table(out.reset_index()[list(TABLE_COLUMNS)], what_if)
