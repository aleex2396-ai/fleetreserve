"""Cox proportional hazards as a departure model (plan sections 7, 13 WP8; a forecast baseline of
section 7; `forecast_model.DepartureModel`).

Fitted on the training units only (`fit_units` is `truth_build.TRAIN_UNITS`), their rows pooled;
development units are only forecast. Training rows, durations and covariates are those of
`models.survival_regression` (`learn_features.FEATURE_COLUMNS` only; no site or unit name).

Fit: `lifelines.CoxPHFitter` with a ridge penalty `PENALIZER` (lifelines' Efron partial
likelihood and Breslow baseline), options `FIT_OPTIONS`, every other option at the default of the
pinned lifelines. A fit that does not converge raises `ValueError`
(`survival_regression.fit_checked`). The settings are fixed constants; nothing is tuned.

Curve of a car with partial hazard r (lifelines' `predict_partial_hazard`): `S(t) = exp(-H0(t) r)`
with `H0` the fitted baseline cumulative hazard, a step function that rises at each training
duration t_j. Its departure has mass at the t_j only, so its lower q-quantile given departure above
a is the first t_j at which `1 - S(t_j) / S(a)` exceeds q, `S(a)` taken at the last t_j at or below
a (1 before the first), NaN where no t_j does. The comparison is made on the cumulative hazard,
`(H0(t_j) - H0(a)) r > -log(1 - q)`, with no tolerance; `forecast_model.survival_quantile` makes it
on the curve with a tolerance of 1e-9, and a test finds the same numbers from both on lifelines'
fitted curves of made-up cars. The forecast table follows
`survival_regression.curve_columns` and `forecast_table`.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from fleetreserve.models import survival_regression as sr
from fleetreserve.truth_build import TRAIN_UNITS

PENALIZER = 0.01
FIT_OPTIONS = {"max_steps": 500}
DURATION = "duration_h"


class CoxModel:
    name = "cox"
    fit_units: tuple[str, ...] = TRAIN_UNITS

    def __init__(self, *, rho: float, min_cell: int) -> None:
        self.rho = rho
        self.min_cell = min_cell
        self.covariates: sr.Covariates | None = None
        self.fitter: CoxPHFitter | None = None
        self.times: np.ndarray | None = None
        self.cumulative: np.ndarray | None = None
        self.km = None

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit on the pooled cars tables of `fit_units` (module text)."""
        table, durations = sr.training_rows(frame, self.rho)
        self.covariates = sr.Covariates.fit(table)
        data = self.covariates.transform(table).assign(**{DURATION: durations})
        self.fitter = sr.fit_checked(
            self.name,
            lambda: CoxPHFitter(penalizer=PENALIZER).fit(
                data, duration_col=DURATION, fit_options=dict(FIT_OPTIONS)
            ),
        )
        baseline = self.fitter.baseline_cumulative_hazard_
        self.times = baseline.index.to_numpy(float)
        self.cumulative = baseline.iloc[:, 0].to_numpy(float)
        self.km = sr.fit_energy(frame, self.rho, self.min_cell)

    def partial_hazards(self, frame: pd.DataFrame) -> np.ndarray:
        """The partial hazard of each row of `frame`, from its `FEATURE_COLUMNS`."""
        x = self.covariates.transform(frame)
        with sr.lifelines_deprecations_ignored():
            return self.fitter.predict_partial_hazard(x).to_numpy(float)

    def _base(self, after: float) -> float:
        last = int(np.searchsorted(self.times, after, side="right")) - 1
        return 0.0 if last < 0 else float(self.cumulative[last])

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if self.fitter is None:
            raise ValueError("cox: predict was called before fit")
        frame = sr.model_frame(frame)
        if len(frame) == 0:
            return sr.forecast_table(frame, kappas, self.km, {})
        hazard = self.partial_hazards(frame)

        def quantile(level: float, after: float) -> np.ndarray:
            target = self._base(after) - math.log1p(-level) / hazard
            found = np.searchsorted(self.cumulative, target, side="right")
            inside = found < self.times.size
            return np.where(inside, self.times[np.minimum(found, self.times.size - 1)], math.nan)

        def presence(after: float) -> np.ndarray:
            return np.exp(-self._base(after) * hazard)

        columns = sr.curve_columns(quantile, presence, kappas)
        return sr.forecast_table(frame, kappas, self.km, columns)


def make_model(matrix) -> CoxModel:
    return CoxModel(rho=matrix.rho, min_cell=matrix.min_cell)
