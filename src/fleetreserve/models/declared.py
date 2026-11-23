"""Declared departure as a departure model (plan section 7, forecast baseline "declared departure
(ACN, NREL)"; `forecast_model.DepartureModel`).

The forecast of when a car leaves is the departure its driver stated when plugging in, read from
the feature `declared_departure_h` of `learn_features` (hours from τ; NaN unless the statement
was made at or before τ, the rule of `truth` and `learn_features`). Nothing is fitted
(`fit_units` is empty) and nothing else is read.

This is a point statement: it carries no caution level. At every kappa the table holds the same
value:

- `d_hat` and `d_median`: the declared departure, where it lies after τ. A car with no declared
  departure known at τ, or one whose declared time had already passed by τ (the car is still
  plugged in, so the statement no longer says anything about when it leaves; a reading, the plan
  does not say), gets NaN: no forecast, and the gate leaves the car unsteered;
- `survival_h1` to `survival_h8` (presence at the start of service hour t): the declared departure
  where it is later than t, else NaN, so the gate keeps the bound of the hour before;
- `r_hat`, `e_cap`, `e_need`: NaN. The WP8 run matrix (`learn/matrix.json`) takes rate and energy
  as known, where `forecast_model.TableForecaster` reads only `d_hat`; under an energy mode that
  reads these columns every car would get no forecast.

Units without declared departures (the Norwegian sites) get NaN for every car.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from fleetreserve.forecast_model import KEY_COLUMNS, SURVIVAL_COLUMNS, TABLE_COLUMNS, empty_table

DECLARED = "declared_departure_h"


def usable_declared(frame: pd.DataFrame) -> np.ndarray:
    """The declared departure (hours from τ) of each row where it lies after τ, else NaN."""
    declared = frame[DECLARED].astype(float).to_numpy()
    return np.where(declared > 0, declared, math.nan)


def table_rows(frame: pd.DataFrame, kappas: Sequence[float], rho: float, columns) -> pd.DataFrame:
    """The forecast table; `columns(kappa)` maps each number column but `rho` to its values."""
    if len(frame) == 0:
        return empty_table()
    parts = []
    for kappa in kappas:
        part = pd.DataFrame({c: frame[c].astype(str).to_numpy() for c in KEY_COLUMNS})
        part["kappa"] = float(kappa)
        part["rho"] = float(rho)
        for name, values in columns(float(kappa)).items():
            part[name] = np.asarray(values, dtype=float)
        parts.append(part[list(TABLE_COLUMNS)])
    return pd.concat(parts, ignore_index=True)


class DeclaredModel:
    name = "declared"
    fit_units: tuple[str, ...] = ()

    def __init__(self, *, rho: float) -> None:
        self.rho = rho

    def fit(self, frame: pd.DataFrame) -> None:
        """Nothing to fit."""

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        declared = usable_declared(frame)
        nothing = np.full(len(frame), math.nan)

        def columns(kappa: float) -> dict:
            out = {"d_hat": declared, "d_median": declared}
            out.update(r_hat=nothing, e_cap=nothing, e_need=nothing)
            for hour, name in enumerate(SURVIVAL_COLUMNS, start=1):
                out[name] = np.where(declared > hour, declared, math.nan)
            return out

        return table_rows(frame, kappas, self.rho, columns)


def make_model(matrix) -> DeclaredModel:
    return DeclaredModel(rho=matrix.rho)
