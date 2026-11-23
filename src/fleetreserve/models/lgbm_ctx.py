"""Pooled LightGBM with the lower tail of the user's history and two context inputs:
`models.lgbm_tail` plus a public-holiday flag and a site-type flag (plan sections 7, 13 WP8; a
learned departure model of section 7).

Everything of `models.lgbm_tail` holds: the fit units (the training units only), the settings
(`models.lgbm.PARAMETERS`, `ROUNDS`: the present settings, so this model needs no settings file
and shows what the two inputs alone change), the quantity and level of every booster, the rows
each is fitted on, the clipping of a prediction at 0, the survival columns left NaN and the table
format. The one change is the input: `COLUMNS`, the columns of `models.lgbm_tail` followed by
`learn_features.CONTEXT_COLUMNS`:

- `public_holiday`: the row's date is a public holiday at its site (the pinned `holidays`
  package's calendars as they stand; plan section 7's dating of laws by enactment is not applied);
- `site_type`: 1 for a workplace, 0 otherwise (plan section 2's split).

Both are functions of the row's `site` and `date` alone (`learn_features.context_columns`), so
they are known at τ; the model computes them itself, on its fit table and on the frame it is
given to forecast, which carries neither. The site name itself is still not an input. A site
with no entry in the committed mappings fails the model (`ValueError`): no site gets a default.

`settings` (None here) is how `models.lgbm_tuned` gives other settings to the same code.
"""

from collections.abc import Sequence

import pandas as pd

from fleetreserve.learn_features import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    TAIL_HISTORY_COLUMNS,
    build_table,
    with_context,
)
from fleetreserve.learn_tune_grid import Settings
from fleetreserve.models.lgbm import (
    PARAMETERS,
    LgbmModel,
    QuantileSet,
    features,
    forecast_rows,
    matrix_kappas,
    quantile_columns,
)

COLUMNS = (*FEATURE_COLUMNS, *TAIL_HISTORY_COLUMNS, *CONTEXT_COLUMNS)


def lightgbm_settings(settings: Settings | None) -> tuple[dict | None, int | None]:
    """(parameters, rounds) as `QuantileSet` takes them: None, None (its defaults) without
    settings, else `PARAMETERS` with the settings' three values, and the settings' rounds."""
    if settings is None:
        return None, None
    return {**PARAMETERS, **settings.overrides()}, settings.rounds


class LgbmCtxModel(LgbmModel):
    name = "lgbm_ctx"

    def __init__(
        self, *, rho: float, kappas: Sequence[float], settings: Settings | None = None
    ) -> None:
        super().__init__(rho=rho, kappas=kappas)
        self.settings = settings

    def fit_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The fit table: `build_table` of the cars tables, with the context columns."""
        return with_context(build_table(frame, rho=self.rho))

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit every quantile model on the pooled cars tables of `fit_units` (module text)."""
        parameters, rounds = lightgbm_settings(self.settings)
        self.quantiles = QuantileSet(self.levels, COLUMNS, parameters, rounds).fit(
            self.fit_table(frame)
        )

    def inputs(self, frame: pd.DataFrame):
        """The feature matrix of the frame to forecast, its context columns set here."""
        return features(with_context(frame), COLUMNS)

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if self.quantiles is None:
            raise ValueError("predict before fit")
        x = self.inputs(frame)
        return forecast_rows(frame, kappas, self.rho, quantile_columns(self.quantiles, x, kappas))


def make_model(matrix) -> LgbmCtxModel:
    return LgbmCtxModel(rho=matrix.rho, kappas=matrix_kappas(matrix))
