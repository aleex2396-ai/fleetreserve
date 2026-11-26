"""`models.lgbm_tail` fitted with a weight on each row of its fit frame: the model of the
`equal-mass` arm of the local-refit trial (`forecast_local`, a trial made after the sealed run).

The features, labels, settings, levels, rows per quantity and `predict` are those of
`models.lgbm_tail`. The one change is in `fit`: the cars tables it is given carry the column
`forecast_local.WEIGHT_COLUMN`, one positive weight per row, and `WeightedQuantileSet` gives
LightGBM that weight, on the rows of each quantity's fit, for every quantity and level. A frame
without the column, or with a weight that is missing, not finite or not above 0, is refused
(`ValueError`): a weighted fit is never made by accident without its weights. `lgbm_tail` itself is
not changed; with every weight 1 the data set of each booster is the one `lgbm_tail` builds, with
a weight of 1 on each row.
"""

import lightgbm
import numpy as np
import pandas as pd

from fleetreserve.forecast_local import WEIGHT_COLUMN
from fleetreserve.learn_features import build_table
from fleetreserve.models.lgbm import (
    LABELS,
    PARAMETERS,
    ROUNDS,
    QuantileSet,
    features,
    matrix_kappas,
    rows_for,
)
from fleetreserve.models.lgbm_tail import COLUMNS, LgbmTailModel


class WeightedQuantileSet(QuantileSet):
    """`QuantileSet` whose `fit` takes one weight per row of the table (module text)."""

    def fit(self, table: pd.DataFrame, weights=None) -> "WeightedQuantileSet":
        if weights is None:
            raise ValueError("a weighted fit needs its weights")
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(table):
            raise ValueError("one weight per row of the table is needed")
        parameters = PARAMETERS if self.parameters is None else self.parameters
        rounds = ROUNDS if self.rounds is None else self.rounds
        for quantity, levels in self.levels.items():
            keep = rows_for(table, quantity)
            if not keep.any():
                raise ValueError(f"no training row has a {quantity} label")
            keep = keep.to_numpy(bool)
            x = features(table[keep], self.columns)
            y = table.loc[keep, LABELS[quantity][0]].to_numpy(float)
            for level in levels:
                data = lightgbm.Dataset(
                    x,
                    label=y,
                    weight=weights[keep],
                    feature_name=list(self.columns),
                    free_raw_data=False,
                )
                self.boosters[(quantity, level)] = lightgbm.train(
                    {**parameters, "alpha": level}, data, num_boost_round=rounds
                )
        return self


class LgbmTailEqualMassModel(LgbmTailModel):
    name = "lgbm_tail_equal_mass"

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit every quantile model on the pooled cars tables, each row with its weight (module
        text)."""
        if WEIGHT_COLUMN not in frame.columns:
            raise ValueError(f"the fit frame has no weight column {WEIGHT_COLUMN!r}")
        weights = frame[WEIGHT_COLUMN].to_numpy(float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("every weight must be a finite number above 0")
        table = build_table(frame, rho=self.rho)
        self.quantiles = WeightedQuantileSet(self.levels, COLUMNS).fit(table, weights)


def make_model(matrix) -> LgbmTailEqualMassModel:
    return LgbmTailEqualMassModel(rho=matrix.rho, kappas=matrix_kappas(matrix))
