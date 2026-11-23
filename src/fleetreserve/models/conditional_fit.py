"""The conditional-fit rule for the survival-tightened bound of a LightGBM quantile model (plan
section 1: at service hour t, the kappa-quantile of departure given presence at t).

A quantile model fitted on every car plugged in at the decision time τ gives the quantile given
presence at τ only. This rule fits the plan's quantity directly: for each service hour h from 1
to `day.MAX_SERVICE_HOURS` (8) and each kappa, one LightGBM quantile model of departure (hours
from τ), fitted only on the training rows whose car was still plugged in at the start of that
hour.

Hour start: service hour h starts h hours after τ (hour 1 starts one hour after τ), and a car is
still plugged in then when its exact departure, `label_departure_h`, is above h. This is the
convention of `forecast_empirical.EmpiricalForecaster.survival_bound` and of `models.km` (a
departure above the hour), and the one `gate_day` reads `survival_h<h>` with.

Inputs: the feature columns the caller names, the same the model's quantiles given presence at τ
are fitted on, each known at τ. Nothing observed after τ is read except the fact of presence at
the start of hour h, which is what the choice of training rows encodes. In prediction no outcome
is read at all: every car is given the bound of every hour, and the gate reads the bound of hour
h only for a car it steers in that hour.

Settings: `models.lgbm.PARAMETERS` and `ROUNDS` with the level as `alpha`, unchanged (the same
seed and deterministic mode), so two fits on the same rows give the same bounds. A caller may
give other settings (`parameters`, `rounds`); only `models.lgbm_tuned_surv` does, so that its
bound's boosters and its quantiles given presence at τ share one set of settings.

Few rows: an hour with fewer than `MIN_ROWS` (20) conditioned training rows is not fitted, and
its bound is NaN for every car, so the gate keeps the bound of the hour before
(`forecast_model`). 20 is the minimum cell size of the gate matrix (`min_cell`) and of
`forecast_empirical.MIN_CELL`.

Order: a bound of hour h below the hour's start is raised to it (a car present at the start of
hour h has not left before it), so the bound of hour h is at least h; it may equal h, where the
empirical bound is strictly above. Nothing else is forced: the bound of hour h may lie below the
forecast departure at τ or below the bound of an earlier hour, and the bounds at different kappa
are not put in order.
"""

import math
from collections.abc import Sequence

import lightgbm
import numpy as np
import pandas as pd

from fleetreserve.forecast_empirical import MIN_CELL
from fleetreserve.forecast_model import SURVIVAL_COLUMNS
from fleetreserve.learn_features import FEATURE_COLUMNS
from fleetreserve.models.lgbm import LABELS, PARAMETERS, ROUNDS, features, rows_for

CONDITIONAL_FIT = (
    "conditional-fit: per service hour and level, a quantile model fitted on the training rows "
    "still plugged in at the hour's start"
)
MIN_ROWS = MIN_CELL
HOURS = tuple(range(1, len(SURVIVAL_COLUMNS) + 1))


def present_at(departure: np.ndarray, hour: int) -> np.ndarray:
    """The rows still plugged in at the start of service hour `hour` (module text)."""
    return departure > hour


class ConditionalBounds:
    """One LightGBM quantile model of departure per service hour and kappa (module text)."""

    def __init__(
        self,
        kappas: Sequence[float],
        columns: Sequence[str] = FEATURE_COLUMNS,
        parameters: dict | None = None,
        rounds: int | None = None,
    ) -> None:
        self.kappas = tuple(sorted({float(k) for k in kappas}))
        if not self.kappas or any(not 0 < k < 1 for k in self.kappas):
            raise ValueError(f"kappas must be a non-empty list of levels in (0, 1), got {kappas}")
        self.columns = tuple(columns)
        self.parameters = parameters
        self.rounds = rounds
        self.rows: dict[int, int] = {}
        self.boosters: dict[tuple[int, float], lightgbm.Booster] = {}

    def fit(self, table: pd.DataFrame) -> "ConditionalBounds":
        """Fit on a `learn_features.build_table` table; an hour with fewer than `MIN_ROWS`
        conditioned rows is left without a model."""
        label = LABELS["departure"][0]
        table = table[rows_for(table, "departure")]
        departure = table[label].to_numpy(float)
        self.rows, self.boosters = {}, {}
        parameters = PARAMETERS if self.parameters is None else self.parameters
        rounds = ROUNDS if self.rounds is None else self.rounds
        for hour in HOURS:
            keep = present_at(departure, hour)
            self.rows[hour] = int(keep.sum())
            if self.rows[hour] < MIN_ROWS:
                continue
            x = features(table[keep], self.columns)
            for kappa in self.kappas:
                data = lightgbm.Dataset(
                    x, label=departure[keep], feature_name=list(self.columns), free_raw_data=False
                )
                self.boosters[(hour, kappa)] = lightgbm.train(
                    {**parameters, "alpha": kappa}, data, num_boost_round=rounds
                )
        return self

    def predict(self, x: np.ndarray, hour: int, kappa: float) -> np.ndarray:
        """The bound of service hour `hour` at level `kappa` for the rows of `x`: at least the
        hour's start, NaN when the hour was not fitted."""
        if hour not in self.rows:
            raise ValueError(f"no bound was fitted for hour {hour}: fit first, hours {HOURS}")
        if float(kappa) not in self.kappas:
            raise ValueError(f"no bound was fitted at level {kappa}")
        booster = self.boosters.get((hour, float(kappa)))
        if booster is None:
            return np.full(x.shape[0], math.nan)
        if x.shape[0] == 0:
            return np.zeros(0)
        return np.maximum(booster.predict(x, num_threads=1), float(hour))


def with_bounds(
    table: pd.DataFrame, bounds: ConditionalBounds, x: np.ndarray, kappas: Sequence[float]
) -> pd.DataFrame:
    """The forecast table `table` (`models.lgbm.forecast_rows` of the rows of `x`, one block per
    level in the order of `x`) with its survival columns filled from `bounds`."""
    out = table.copy()
    for kappa in kappas:
        rows = (out["kappa"] == float(kappa)).to_numpy()
        if int(rows.sum()) != x.shape[0]:
            raise ValueError(
                f"the table holds {int(rows.sum())} rows at level {kappa}, not one per car"
            )
        for hour, name in zip(HOURS, SURVIVAL_COLUMNS, strict=True):
            out.loc[rows, name] = bounds.predict(x, hour, kappa)
    return out
