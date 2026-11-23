"""Pooled LightGBM with the lower tail of the user's history: `models.lgbm` plus ten features (plan
sections 7, 13 WP8; a learned departure model of section 7).

Everything of `models.lgbm` holds: the fit units (the training units only), the settings
(`PARAMETERS`, `ROUNDS`), the quantity and level of every booster, the rows each is fitted on, the
clipping of a prediction at 0, the survival columns left NaN and the table format. The one change
is the input: the columns `FEATURE_COLUMNS` of `learn_features` followed by
`learn_features.TAIL_HISTORY_COLUMNS`, the minimum, lower quantiles, short-stay shares, same-weekday
median and last-session values of the departure values of the user's earlier sessions, each known
at the decision time τ (definitions in `learn_features`). The mean and median of the history, which
`models.lgbm` reads, do not show a driver who often leaves within a few hours; the lower tail does.

A car with no user id or no earlier session has the count 0 and missing tail values, as in the
history columns; LightGBM sends a missing value down a learned side of each split.
"""

from fleetreserve.learn_features import FEATURE_COLUMNS, TAIL_HISTORY_COLUMNS, build_table
from fleetreserve.models.lgbm import (
    LgbmModel,
    QuantileSet,
    features,
    forecast_rows,
    matrix_kappas,
    quantile_columns,
)

COLUMNS = (*FEATURE_COLUMNS, *TAIL_HISTORY_COLUMNS)


class LgbmTailModel(LgbmModel):
    name = "lgbm_tail"

    def fit(self, frame) -> None:
        """Fit every quantile model on the pooled cars tables of `fit_units` (module text)."""
        self.quantiles = QuantileSet(self.levels, COLUMNS).fit(build_table(frame, rho=self.rho))

    def predict(self, frame, kappas):
        if self.quantiles is None:
            raise ValueError("predict before fit")
        x = features(frame, COLUMNS)
        return forecast_rows(frame, kappas, self.rho, quantile_columns(self.quantiles, x, kappas))


def make_model(matrix) -> LgbmTailModel:
    return LgbmTailModel(rho=matrix.rho, kappas=matrix_kappas(matrix))
