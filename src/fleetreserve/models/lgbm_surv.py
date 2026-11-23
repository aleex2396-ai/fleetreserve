"""Pooled LightGBM with the survival-tightened bound: `models.lgbm` plus the conditional-fit rule
(plan sections 1, 7, 13 WP8; a learned departure model of section 7).

Everything of `models.lgbm` holds, and every column that model fills is filled with the same
number: the fit units (the training units only), the features (`COLUMNS`, here
`learn_features.FEATURE_COLUMNS`), the settings, the quantity and level of every booster, the rows
each is fitted on and the clipping of a prediction at 0. The one change: the columns `survival_h1`
to `survival_h8`, which `models.lgbm` leaves NaN, hold the bound of
`models.conditional_fit` (`SURVIVAL_BOUND`), fitted on the same training table and the same
feature columns: the kappa-quantile of departure given presence at the start of each service
hour, at least the hour's start, NaN for an hour with too few training rows still plugged in.

`models.lgbm` is kept as it is, so its recorded results stay reproducible; this is a separate
model with its own name.
"""

from collections.abc import Sequence

import pandas as pd

from fleetreserve.learn_features import FEATURE_COLUMNS, build_table
from fleetreserve.models.conditional_fit import CONDITIONAL_FIT, ConditionalBounds, with_bounds
from fleetreserve.models.lgbm import (
    LgbmModel,
    QuantileSet,
    features,
    forecast_rows,
    matrix_kappas,
    quantile_columns,
)

SURVIVAL_BOUND = CONDITIONAL_FIT


class LgbmSurvModel(LgbmModel):
    name = "lgbm_surv"
    COLUMNS: tuple[str, ...] = FEATURE_COLUMNS

    def __init__(self, *, rho: float, kappas: Sequence[float]) -> None:
        super().__init__(rho=rho, kappas=kappas)
        self.kappas = tuple(kappas)
        self.bounds: ConditionalBounds | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit every quantile model and every hour's bound on the pooled cars tables of
        `fit_units` (module text)."""
        table = build_table(frame, rho=self.rho)
        self.quantiles = QuantileSet(self.levels, self.COLUMNS).fit(table)
        self.bounds = ConditionalBounds(self.kappas, self.COLUMNS).fit(table)

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if self.quantiles is None or self.bounds is None:
            raise ValueError("predict before fit")
        x = features(frame, self.COLUMNS)
        table = forecast_rows(frame, kappas, self.rho, quantile_columns(self.quantiles, x, kappas))
        return with_bounds(table, self.bounds, x, kappas)


def make_model(matrix) -> LgbmSurvModel:
    return LgbmSurvModel(rho=matrix.rho, kappas=matrix_kappas(matrix))
