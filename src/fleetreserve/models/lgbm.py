"""Pooled LightGBM: quantile-regression gradient boosting as a departure model (plan sections 2, 7,
13 WP8; a forecast baseline of section 7).

Fitted on the training units only (`fit_units` is `truth_build.TRAIN_UNITS`), their rows pooled
into one data set; development units are only forecast. Every input comes from
`learn_features.build_table` (fit) and `learn_features.feature_frame` (predict).

Features: `learn_features.FEATURE_COLUMNS`, all numbers (true/false as 1/0, a missing history
value as a missing value, which LightGBM sends down a learned side of each split). The site and
unit names are not features. The scored sites are never training sites, so whatever a model
learned about a named training site could not be applied to them; leaving the names out keeps one
pooled model whose prediction for a site seen in training and one never seen follows the same
rule, and a site unseen in training cannot break prediction.

Labels and rows (as in `forecast_empirical`'s reading 2): departure on every row of the training
cars tables (every car plugged in at τ, on counting and non-counting days); rate, room and
protected energy on the rows the rules keep in a fleet (a room that is a number).

One model per quantity and level (`levels_of`), each fitted with LightGBM's quantile (pinball)
objective at that level:

- `d_hat`: departure (hours from τ) at each level of the run matrix (`matrix_kappas`: each
  configuration's kappa and, under a driver dial, every level up to it); `d_median`: at 0.5;
- `r_hat`: rate (kW) at kappa; `e_cap`: room (kWh) at kappa; `e_need`: protected energy
  `(1 - rho) * room` (kWh) at 1 - kappa.

A prediction below 0 is raised to 0: every quantity is at least 0, so a lower bound raised to 0
still lies below the truth, and the gate's forecast car (`day.ForecastCar`) refuses negative
values. Nothing else is changed; in particular the departure quantiles at different kappa are not
forced into order.

Settings (`PARAMETERS`, `ROUNDS`) are fixed constants, chosen before any development unit was
scored and not tuned on development units: a modest model (200 trees of at most 15 leaves) that
fits in minutes on a laptop. Deterministic: no row or column sampling, a fixed seed, LightGBM's
deterministic mode, row-wise histograms and one thread, so two fits on the same rows give the same
forecasts. `QuantileSet` takes other settings as arguments, which only the models that say so use
(`models.lgbm_tuned`, `models.lgbm_tuned_surv`); every other model leaves them at these constants.

Survival-tightened bound (`survival_h1` to `survival_h8`, plan section 1: the kappa-quantile of
departure given presence at the start of service hour t). The plan does not say how a model that
gives quantiles given presence at τ only provides it. `SURVIVAL_BOUND` names the rule of this
model, `NO_SURVIVAL_BOUND`: those columns are left NaN, so the gate keeps the bound of the hour
before (no tightening: the forecast departure at τ holds in every hour). Another rule,
`models.conditional_fit`, is built as separate models (`models.lgbm_surv`,
`models.lgbm_tail_surv`); this model, `models.cqr` and `models.lgbm_tail` do not use it.
"""

import math
from collections.abc import Sequence

import lightgbm
import numpy as np
import pandas as pd

from fleetreserve import driver_risk
from fleetreserve.forecast_model import (
    KEY_COLUMNS,
    MEDIAN,
    SURVIVAL_COLUMNS,
    TABLE_COLUMNS,
    empty_table,
)
from fleetreserve.learn_features import FEATURE_COLUMNS, build_table
from fleetreserve.truth_build import TRAIN_UNITS

SEED = 0
ROUNDS = 200
PARAMETERS = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "max_bin": 255,
    "bagging_fraction": 1.0,
    "feature_fraction": 1.0,
    "lambda_l2": 0.0,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
    "verbose": -1,
}
NO_SURVIVAL_BOUND = "none: columns left NaN, the gate keeps the earlier bound"
SURVIVAL_BOUND = NO_SURVIVAL_BOUND

# quantity -> (label column, rows: "all" or "fleet")
LABELS = {
    "departure": ("label_departure_h", "all"),
    "rate": ("label_rate_kw", "fleet"),
    "room": ("label_room_kwh", "fleet"),
    "need": ("label_need_kwh", "fleet"),
}


def levels_of(kappas: Sequence[float]) -> dict[str, tuple[float, ...]]:
    """The quantile levels fitted per quantity (module text)."""
    ks = tuple(sorted({float(k) for k in kappas}))
    if not ks or any(not 0 < k < 1 for k in ks):
        raise ValueError(f"kappas must be a non-empty list of levels in (0, 1), got {kappas}")
    return {
        "departure": tuple(sorted({*ks, MEDIAN})),
        "rate": ks,
        "room": ks,
        "need": tuple(sorted({round(1 - k, 12) for k in ks})),
    }


def features(frame: pd.DataFrame, columns: Sequence[str] = FEATURE_COLUMNS) -> np.ndarray:
    """The feature matrix of `frame` (a `build_table` or `feature_frame` table): the columns
    `columns` (default `FEATURE_COLUMNS`) as numbers, nothing else read."""
    return frame[list(columns)].astype(float).to_numpy()


def rows_for(table: pd.DataFrame, quantity: str) -> pd.Series:
    """The rows of a `build_table` table a quantity is fitted on (module text)."""
    label, rows = LABELS[quantity]
    keep = table[label].notna()
    if rows == "fleet":
        keep &= table["label_room_kwh"].notna()
    return keep


class QuantileSet:
    """One LightGBM quantile model per quantity and level, on the feature columns `columns`
    (default `FEATURE_COLUMNS`), with the settings `parameters` and `rounds` (when None, the
    default: `PARAMETERS` and `ROUNDS`)."""

    def __init__(
        self,
        levels: dict[str, tuple[float, ...]],
        columns: Sequence[str] = FEATURE_COLUMNS,
        parameters: dict | None = None,
        rounds: int | None = None,
    ) -> None:
        self.levels = levels
        self.columns = tuple(columns)
        self.parameters = parameters
        self.rounds = rounds
        self.boosters: dict[tuple[str, float], lightgbm.Booster] = {}

    def fit(self, table: pd.DataFrame) -> "QuantileSet":
        parameters = PARAMETERS if self.parameters is None else self.parameters
        rounds = ROUNDS if self.rounds is None else self.rounds
        for quantity, levels in self.levels.items():
            keep = rows_for(table, quantity)
            if not keep.any():
                raise ValueError(f"no training row has a {quantity} label")
            x = features(table[keep], self.columns)
            y = table.loc[keep, LABELS[quantity][0]].to_numpy(float)
            for level in levels:
                data = lightgbm.Dataset(
                    x, label=y, feature_name=list(self.columns), free_raw_data=False
                )
                self.boosters[(quantity, level)] = lightgbm.train(
                    {**parameters, "alpha": level}, data, num_boost_round=rounds
                )
        return self

    def predict(self, x: np.ndarray, quantity: str, level: float) -> np.ndarray:
        """The fitted level-quantile of `quantity` for the rows of `x`, raised to at least 0."""
        booster = self.boosters.get((quantity, level))
        if booster is None:
            raise ValueError(f"no {quantity} model was fitted at level {level}")
        if x.shape[0] == 0:
            return np.zeros(0)
        return np.maximum(booster.predict(x, num_threads=1), 0.0)


def forecast_rows(
    frame: pd.DataFrame, kappas: Sequence[float], rho: float, columns: dict
) -> pd.DataFrame:
    """The forecast table from per-level columns: `columns[kappa]` maps `d_hat`, `r_hat`,
    `e_cap`, `e_need`, `d_median` to one array over the rows of `frame`; the survival columns
    follow `SURVIVAL_BOUND`."""
    if SURVIVAL_BOUND != NO_SURVIVAL_BOUND:
        raise ValueError(f"survival bound rule not built: {SURVIVAL_BOUND!r}")
    if len(frame) == 0:
        return empty_table()
    parts = []
    for kappa in kappas:
        part = pd.DataFrame({c: frame[c].astype(str).to_numpy() for c in KEY_COLUMNS})
        part["kappa"] = float(kappa)
        part["rho"] = float(rho)
        for name, values in columns[float(kappa)].items():
            part[name] = np.asarray(values, dtype=float)
        for name in SURVIVAL_COLUMNS:
            part[name] = math.nan
        parts.append(part[list(TABLE_COLUMNS)])
    return pd.concat(parts, ignore_index=True)


def quantile_columns(quantiles: QuantileSet, x: np.ndarray, kappas: Sequence[float]) -> dict:
    """Per kappa, the forecast table's number columns from `quantiles` (module text)."""
    median = quantiles.predict(x, "departure", MEDIAN)
    out = {}
    for kappa in kappas:
        k = float(kappa)
        out[k] = {
            "d_hat": quantiles.predict(x, "departure", k),
            "r_hat": quantiles.predict(x, "rate", k),
            "e_cap": quantiles.predict(x, "room", k),
            "e_need": quantiles.predict(x, "need", round(1 - k, 12)),
            "d_median": median,
        }
    return out


class LgbmModel:
    name = "lgbm"
    fit_units: tuple[str, ...] = TRAIN_UNITS

    def __init__(self, *, rho: float, kappas: Sequence[float]) -> None:
        self.rho = rho
        self.levels = levels_of(kappas)
        self.quantiles: QuantileSet | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit every quantile model on the pooled cars tables of `fit_units` (module text)."""
        self.quantiles = QuantileSet(self.levels).fit(build_table(frame, rho=self.rho))

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if self.quantiles is None:
            raise ValueError("predict before fit")
        x = features(frame)
        return forecast_rows(frame, kappas, self.rho, quantile_columns(self.quantiles, x, kappas))


def matrix_kappas(matrix) -> list[float]:
    """The levels a model built from `matrix` is fitted at (`driver_risk.matrix_levels`)."""
    return list(driver_risk.matrix_levels(matrix.configurations))


def make_model(matrix) -> LgbmModel:
    return LgbmModel(rho=matrix.rho, kappas=matrix_kappas(matrix))
