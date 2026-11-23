"""Weibull accelerated failure time as a departure model (plan sections 7, 13 WP8; a forecast
baseline of section 7; `forecast_model.DepartureModel`).

Fitted on the training units only (`fit_units` is `truth_build.TRAIN_UNITS`), their rows pooled;
development units are only forecast. Training rows, durations and covariates are those of
`models.survival_regression` (`learn_features.FEATURE_COLUMNS` only; no site or unit name).

Fit: `lifelines.WeibullAFTFitter` with a ridge penalty `PENALIZER`, which the pinned lifelines
puts on the covariate coefficients of `log lambda` and on `log k` (lifelines' `rho_` intercept),
not on the intercept of `log lambda`; options `FIT_OPTIONS` (empty: every option at the default
of the pinned lifelines). The scale depends on the covariates, `log lambda = b0 + b . x`; the
shape k (lifelines' `rho_`) is one number. A fit that does not converge raises `ValueError`
(`survival_regression.fit_checked`). The settings are fixed constants; nothing is tuned.

Curve of a car: `S(t) = exp(-(t / lambda)^k)`. Closed forms, with `c = -log(1 - q)`: the
q-quantile of departure given departure above a is `lambda * ((a / lambda)^k + c)^(1 / k)`
(above a for q above 0), and the probability of departure above a is `S(a)`. The forecast table
follows `survival_regression.curve_columns` and `forecast_table`.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter

from fleetreserve.models import survival_regression as sr
from fleetreserve.truth_build import TRAIN_UNITS

PENALIZER = 0.01
FIT_OPTIONS: dict = {}
DURATION = "duration_h"


class WeibullAftModel:
    name = "weibull_aft"
    fit_units: tuple[str, ...] = TRAIN_UNITS

    def __init__(self, *, rho: float, min_cell: int) -> None:
        self.rho = rho
        self.min_cell = min_cell
        self.covariates: sr.Covariates | None = None
        self.fitter: WeibullAFTFitter | None = None
        self.km = None

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit on the pooled cars tables of `fit_units` (module text)."""
        table, durations = sr.training_rows(frame, self.rho)
        self.covariates = sr.Covariates.fit(table)
        data = self.covariates.transform(table).assign(**{DURATION: durations})
        self.fitter = sr.fit_checked(
            self.name,
            lambda: WeibullAFTFitter(penalizer=PENALIZER).fit(
                data, duration_col=DURATION, fit_options=dict(FIT_OPTIONS)
            ),
        )
        self.km = sr.fit_energy(frame, self.rho, self.min_cell)

    def scale_and_shape(self, frame: pd.DataFrame) -> tuple[np.ndarray, float]:
        """(lambda of each row of `frame`, from its `FEATURE_COLUMNS`; the shape k)."""
        x = self.covariates.transform(frame)
        scale = self.fitter.params_.loc["lambda_"]
        log_scale = scale["Intercept"] + x[list(self.covariates.names)].to_numpy(float) @ (
            scale[list(self.covariates.names)].to_numpy(float)
        )
        shape = math.exp(float(self.fitter.params_.loc[("rho_", "Intercept")]))
        return np.exp(log_scale), shape

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if self.fitter is None:
            raise ValueError("weibull_aft: predict was called before fit")
        frame = sr.model_frame(frame)
        if len(frame) == 0:
            return sr.forecast_table(frame, kappas, self.km, {})
        scale, shape = self.scale_and_shape(frame)

        def quantile(level: float, after: float) -> np.ndarray:
            return scale * ((after / scale) ** shape - math.log1p(-level)) ** (1.0 / shape)

        def presence(after: float) -> np.ndarray:
            return np.exp(-((after / scale) ** shape))

        columns = sr.curve_columns(quantile, presence, kappas)
        return sr.forecast_table(frame, kappas, self.km, columns)


def make_model(matrix) -> WeibullAftModel:
    return WeibullAftModel(rho=matrix.rho, min_cell=matrix.min_cell)
