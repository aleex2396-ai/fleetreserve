"""The level grid of the earlier analysis's per-car models, its rules, and the forecast table read
off it (plan sections 7, 12; `models.earlier_gbm`, `models.earlier_population`).

The earlier analysis's package (`ev-fleet-offer`, commit 05a6cf9, module `quantiles`) forecasts
each car at the 16 levels `LEVELS` and then, per car:

- `tidy`: sorts the values across levels (a row holding a NaN is left as it is), which can only
  lower the pinball loss (Chernozhukov et al. (2010)), and clips the stay to 0 to `HORIZON_H` and
  every other quantity to at least 0;
- `at_level`: reads a level as its column when it is on the grid, between two levels by linear
  interpolation, below the grid as 0 (the trivial bound) and above it as the last column.

Its stay target (`stay_target`): the hours from the decision time to plug-out, capped at the
horizon `HORIZON_H` (40 steps of a quarter hour, 10:00 to 20:00) and rounded down to the quarter
hour `STEP_H` (with a tolerance of 1e-9 step for float error).

The forecast table (`table`, `forecast_model.TABLE_COLUMNS`) from a model's grids of the stay,
rate and room, after `tidy`, per level kappa: `d_hat`, `r_hat`, `e_cap` the stay, rate and room at
kappa, `d_median` the stay at 0.5, `e_need` = `(1 - rho)` x the room at 1 - kappa (the protected
energy's upper quantile, plan section 7; reading E4 of `models.earlier_gbm`); the survival columns
NaN (no survival-tightened bound: the earlier models give the forecast at τ only, and the gate
keeps it in every hour, as `models.lgbm`'s `NO_SURVIVAL_BOUND`). A NaN on the grid at the level
read is a NaN in the table: no forecast.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from fleetreserve.forecast_model import (
    KEY_COLUMNS,
    MEDIAN,
    SURVIVAL_COLUMNS,
    TABLE_COLUMNS,
    empty_table,
)

LEVELS = np.array(
    [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
)
STEPS = 40
STEP_H = 0.25
HORIZON_H = STEPS * STEP_H


def stay_target(departure_h) -> np.ndarray:
    """The stay target of each departure (hours from τ; module text)."""
    hours = np.minimum(np.asarray(departure_h, dtype=float), HORIZON_H)
    return np.clip(np.floor(hours / STEP_H + 1e-9), 0, STEPS) * STEP_H


def _sort_rows(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=float)
    ok = ~np.isnan(values).any(axis=1)
    values[ok] = np.sort(values[ok], axis=1)
    return values


def tidy(stay: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """The stay grid and the other grids (cars by levels), sorted and clipped (module text)."""
    out = [np.clip(_sort_rows(stay), 0.0, HORIZON_H)]
    out += [np.clip(_sort_rows(grid), 0.0, None) for grid in others]
    return tuple(out)


def at_level(grid: np.ndarray, level: float) -> np.ndarray:
    """Each row's value at `level` (module text)."""
    grid = np.asarray(grid, dtype=float)
    if level < LEVELS[0]:
        return np.zeros(len(grid))
    level = min(level, LEVELS[-1])
    k = int(np.searchsorted(LEVELS, level, side="right")) - 1
    if k >= len(LEVELS) - 1 or LEVELS[k] == level:
        return grid[:, k]
    share = (level - LEVELS[k]) / (LEVELS[k + 1] - LEVELS[k])
    return (1 - share) * grid[:, k] + share * grid[:, k + 1]


def table(
    frame: pd.DataFrame,
    kappas: Sequence[float],
    rho: float,
    stay: np.ndarray,
    rate: np.ndarray,
    room: np.ndarray,
) -> pd.DataFrame:
    """The forecast table of the rows of `frame` from their grids (cars by `LEVELS`) of the stay
    (hours from τ), rate (kW) and room (kWh) (module text)."""
    if len(frame) == 0:
        return empty_table()
    stay, rate, room = tidy(stay, rate, room)
    median = at_level(stay, MEDIAN)
    parts = []
    for kappa in kappas:
        k = float(kappa)
        part = pd.DataFrame({c: frame[c].astype(str).to_numpy() for c in KEY_COLUMNS})
        part["kappa"] = k
        part["rho"] = float(rho)
        part["d_hat"] = at_level(stay, k)
        part["r_hat"] = at_level(rate, k)
        part["e_cap"] = at_level(room, k)
        part["e_need"] = (1 - rho) * at_level(room, round(1 - k, 12))
        part["d_median"] = median
        for name in SURVIVAL_COLUMNS:
            part[name] = math.nan
        parts.append(part[list(TABLE_COLUMNS)])
    return pd.concat(parts, ignore_index=True)
