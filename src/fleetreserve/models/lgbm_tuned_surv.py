"""Tuned pooled LightGBM with the survival-tightened bound: `models.lgbm_tuned` plus the
conditional-fit rule (plan sections 1, 7, 13 WP8; a learned departure model of section 7).

Every column `models.lgbm_tuned` fills is filled with the same number (the same boosters on the
same columns, `models.lgbm_ctx.COLUMNS`, at the settings of the tuned settings file). The one
change is that of `models.lgbm_surv`: the columns `survival_h1` to `survival_h8` hold the bound of
`models.conditional_fit` (`SURVIVAL_BOUND`), fitted on the same training table and the same
columns. The bound's boosters use the tuned settings too (the same `num_leaves`,
`min_data_in_leaf`, `learning_rate` and rounds), although the settings were chosen by the loss of
the quantiles given presence at τ alone: the bound was not tuned for itself.

`make_model` refuses as `models.lgbm_tuned.make_model` does when the settings file is absent or
records another grid.
"""

from collections.abc import Sequence

import pandas as pd

from fleetreserve.learn_tune_grid import Settings
from fleetreserve.learn_tune_settings import load_tuned
from fleetreserve.models.conditional_fit import CONDITIONAL_FIT, ConditionalBounds, with_bounds
from fleetreserve.models.lgbm import QuantileSet, forecast_rows, matrix_kappas, quantile_columns
from fleetreserve.models.lgbm_ctx import COLUMNS, lightgbm_settings
from fleetreserve.models.lgbm_tuned import LgbmTunedModel

SURVIVAL_BOUND = CONDITIONAL_FIT


class LgbmTunedSurvModel(LgbmTunedModel):
    name = "lgbm_tuned_surv"

    def __init__(self, *, rho: float, kappas: Sequence[float], settings: Settings) -> None:
        super().__init__(rho=rho, kappas=kappas, settings=settings)
        self.kappas = tuple(kappas)
        self.bounds: ConditionalBounds | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit every quantile model and every hour's bound on the pooled cars tables of
        `fit_units`, all at the tuned settings (module text)."""
        table = self.fit_table(frame)
        parameters, rounds = lightgbm_settings(self.settings)
        self.quantiles = QuantileSet(self.levels, COLUMNS, parameters, rounds).fit(table)
        self.bounds = ConditionalBounds(self.kappas, COLUMNS, parameters, rounds).fit(table)

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if self.quantiles is None or self.bounds is None:
            raise ValueError("predict before fit")
        x = self.inputs(frame)
        table = forecast_rows(frame, kappas, self.rho, quantile_columns(self.quantiles, x, kappas))
        return with_bounds(table, self.bounds, x, kappas)


def make_model(matrix) -> LgbmTunedSurvModel:
    return LgbmTunedSurvModel(rho=matrix.rho, kappas=matrix_kappas(matrix), settings=load_tuned())
