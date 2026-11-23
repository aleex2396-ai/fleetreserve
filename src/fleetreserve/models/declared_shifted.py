"""Declared departure moved earlier by the error drivers made on the training units: a cautious
version of `models.declared` (plan section 7, forecast baseline "declared departure (ACN, NREL)").

Fitted on the ACN training units (`fit_units`: caltech, jpl, office001; the Norwegian training
sites have no declared departures). The sample: on every row of their cars tables whose declared
departure is known at τ and lies after τ (`declared.usable_declared`), the error
`true departure - declared departure` (hours). Fewer than the matrix's `min_cell` such rows fail
the fit.

Per car with a usable declared departure `D` and level kappa:

- `d_hat = D + q_kappa`, `q_kappa` the lower kappa-quantile of the errors
  (`forecast_empirical.lower_quantile`), so that on the training units the true departure is
  below `d_hat` for at most a share kappa of cars. This equals `D` minus the upper
  (1 - kappa)-quantile of `declared - true`. A value below 0 is raised to 0, as in `models.lgbm`;
- `d_median = D + ` the median error (the lower 0.5-quantile), raised to at least 0;
- `survival_h1` to `survival_h8`: NaN (no tightening; the gate keeps the earlier bound), because
  the errors are measured given presence at τ only;
- `r_hat`, `e_cap`, `e_need`: NaN, as in `models.declared`.

A car without a usable declared departure gets NaN everywhere: no forecast, left unsteered.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from fleetreserve.forecast_empirical import lower_quantile
from fleetreserve.forecast_model import MEDIAN, SURVIVAL_COLUMNS
from fleetreserve.learn_features import build_table
from fleetreserve.load_acn import SITES
from fleetreserve.models.declared import table_rows, usable_declared


class DeclaredShiftedModel:
    name = "declared_shifted"
    fit_units: tuple[str, ...] = SITES

    def __init__(self, *, rho: float, min_cell: int) -> None:
        self.rho = rho
        self.min_cell = min_cell
        self.errors: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        errors = np.zeros(0)
        if len(frame):
            table = build_table(frame, rho=self.rho)
            errors = table["label_departure_h"].to_numpy(float) - usable_declared(table)
        errors = np.sort(errors[np.isfinite(errors)])
        if errors.size < self.min_cell:
            raise ValueError(
                f"declared_shifted: {errors.size} training cars with a usable declared departure, "
                f"fewer than {self.min_cell}"
            )
        self.errors = errors

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if self.errors is None:
            raise ValueError("declared_shifted: predict was called before fit")
        declared = usable_declared(frame)
        median = np.maximum(declared + lower_quantile(self.errors, MEDIAN), 0.0)
        nothing = np.full(len(frame), math.nan)

        def columns(kappa: float) -> dict:
            out = {"d_hat": np.maximum(declared + lower_quantile(self.errors, kappa), 0.0)}
            out.update(d_median=median, r_hat=nothing, e_cap=nothing, e_need=nothing)
            out.update(dict.fromkeys(SURVIVAL_COLUMNS, nothing))
            return out

        return table_rows(frame, kappas, self.rho, columns)


def make_model(matrix) -> DeclaredShiftedModel:
    return DeclaredShiftedModel(rho=matrix.rho, min_cell=matrix.min_cell)
