"""The reference predictor of a day's size: LightGBM quantile regression, one unit left out at a
time (plan section 7, day-size net; a stand-in for the net, which is not built).

Input: a table with one row per labelled day, a `unit` column, the label column `LABEL` (the exact
safe scale clipped to [0, 1], `size_label`) and the context columns given. For each unit u and each
level of `LEVELS`, a booster is fitted with LightGBM's quantile objective at that level on the rows
of the other units only (`fit_rows`), with the settings of `models.lgbm` (`PARAMETERS`, `ROUNDS`:
fixed seed, deterministic mode, row-wise histograms, one thread, no sampling), and read at u's
rows. The predictions are not clipped. The other units' rows can themselves depend on u's outcomes
(`size_whatif`, step 3); this module only leaves u's rows out of the fit.

Beside it, the best constant predictor of each level fitted on the same rows: the level-quantile of
their labels with numpy's "inverted_cdf" rule (the smallest label whose empirical distribution
function reaches the level), a minimiser of the mean pinball loss over constants.

`leave_one_unit_out(table, columns)` returns, per level, the predictions of the booster and of the
constant for every row, each from the fit that left the row's unit out. A unit whose other units
hold no row gets NaN.
"""

from collections.abc import Sequence

import lightgbm
import numpy as np
import pandas as pd

from fleetreserve.models.lgbm import PARAMETERS, ROUNDS

LEVELS = (0.05, 0.1, 0.2, 0.3, 0.5)
LABEL = "label_clipped"


def fit_rows(table: pd.DataFrame, unit: str) -> np.ndarray:
    """True on the rows of every unit but `unit`."""
    return (table["unit"] != unit).to_numpy(bool)


def constant(labels, level: float) -> float:
    """The best constant predictor at `level` (module text)."""
    return float(np.quantile(np.asarray(labels, dtype=float), level, method="inverted_cdf"))


def booster_predictions(x, y, x_new, columns: Sequence[str], level: float) -> np.ndarray:
    """The booster at `level` fitted on (`x`, `y`) and read at `x_new` (module text)."""
    data = lightgbm.Dataset(x, label=y, feature_name=list(columns), free_raw_data=False)
    booster = lightgbm.train({**PARAMETERS, "alpha": level}, data, num_boost_round=ROUNDS)
    return np.asarray(booster.predict(x_new, num_threads=1), dtype=float)


def leave_one_unit_out(
    table: pd.DataFrame, columns: Sequence[str], levels: Sequence[float] = LEVELS
) -> dict[float, dict[str, np.ndarray]]:
    """Per level, {"gbm": predictions, "constant": predictions}, one per row of `table`."""
    x_all = table[list(columns)].astype(float).to_numpy()
    y_all = table[LABEL].astype(float).to_numpy()
    out = {
        float(level): {"gbm": np.full(len(table), np.nan), "constant": np.full(len(table), np.nan)}
        for level in levels
    }
    for unit in dict.fromkeys(table["unit"]):
        scored = ~fit_rows(table, unit)
        fit = ~scored
        if not fit.any():
            continue
        for level in levels:
            found = out[float(level)]
            found["gbm"][scored] = booster_predictions(
                x_all[fit], y_all[fit], x_all[scored], columns, float(level)
            )
            found["constant"][scored] = constant(y_all[fit], float(level))
    return out
