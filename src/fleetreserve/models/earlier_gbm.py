"""The earlier analysis's learned per-car model (plan section 7, "the earlier GBM and population
models"; section 12) as a departure model (`forecast_model.DepartureModel`).

The earlier model (package `ev-fleet-offer`, commit 05a6cf9, `quantiles.GBDT` with the settings
`percar.CHOSEN` and `percar.FIXED`): scikit-learn's histogram gradient-boosted trees
(`HistGradientBoostingRegressor`) with the pinball loss, one model per level of
`earlier_grid.LEVELS` and per target, learning rate 0.1, at most 7 leaves, at least 20 rows per
leaf, 200 trees, no early stopping, seed 0. Its input matrix (`quantiles.Design`, trees form): the
numeric inputs standardised by the fit rows' mean and spread (missing values kept missing; a
spread of 0 or a mean that is missing on every row leaves the input unscaled or uncentred),
then the station as one integer column, last, marked categorical, missing for a station not seen
in the fit, and left out when the fit saw no station. Predictions per car are sorted across the
levels and clipped (`earlier_grid.tidy`).

All of that is kept here (`Design`, `fit`), and the table is read off the grids by
`earlier_grid.table`. What the interface forces, each a reading with the alternative not taken:

- E1, the history: the earlier refits each month on the site's own car-days of the 90 days
  before (and none with fewer than 200 rows). Here one fit on the training units pooled
  (`fit_units` = `truth_build.TRAIN_UNITS`): a model is fitted on training units only, and the
  scored units are other sites. The earlier's own replication at a second site ran its first
  site's fitted model unchanged there. Alternative: fit on the scored unit's own earlier days as
  an outcome reader (`forecast_model.OUTCOME_READERS`), a change to the list of models allowed to
  read outcomes, not made here.
- E2, the rows: the earlier fits on the cars present at 10:00 on weekdays. Here every row of the
  training cars tables (every car plugged in at the τ the truth sets for its site, counting and
  non-counting days), as `models.lgbm` takes them: the stay on every row, the rate and room on
  the rows the rules keep in a fleet (a room that is a number). The input scaling is fitted on
  every row.
- E3, the targets: the stay is the earlier one, `earlier_grid.stay_target` of the departure
  (`label_departure_h`: capped at 10 hours, rounded down to the quarter hour), so `d_hat` is at
  most 10 hours. The earlier's energy target (energy taken before the end of the 10-hour horizon,
  spread from plug-in to a charging-done time) is replaced by the room `label_room_kwh`: the
  cars tables hold no charging-done time, and the forecast table's `e_cap` forecasts the room.
  The earlier forecasts no rate (its offer took the fixed 6.7 kW of its site's chargers); here a
  third target, `label_rate_kw`, by the same rule and settings. Alternatives: the room capped at
  the energy of 10 hours at the rate (a second rule beside the truth's room), and a fixed rate of
  6.7 kW (a constant of one site's chargers).
- E4, protected energy: the earlier sets no protected share. `e_need` is `(1 - rho)` x the room
  grid at level 1 - kappa (`earlier_grid.table`), the upper quantile plan section 7 asks for, from
  the same fitted room models. Alternative: a fourth family fitted on `label_need_kwh`, which for
  a fixed `rho` is the room times a constant, so the room models already give it.
- E5, the inputs: `earlier_inputs.INPUTS`, 19 of the earlier's 23, each from a feature column
  known at τ (`earlier_inputs.SOURCES`, `earlier_inputs.DERIVED`), in the earlier order.
- E6, the academic calendar and the paid regime: the earlier fits at Caltech and applies the same
  inputs at its second site; here they are given by date at every site, as there
  (`earlier_calendar`, `earlier_inputs.PAID_FROM`), and the calendar flags are missing outside
  the calendar's span, where the earlier refuses the date. Alternative: the flags and the regime
  only for the two sites they were written for, missing elsewhere.
- E7, the station: the cars tables carry it, the feature frame does not. The fit codes the
  stations of its rows by site and station (stations of different sites never share a code; an
  empty station is missing, as at every Norwegian site); `predict` gives every car a missing
  station, as the earlier does for a station unseen in its fit, which is what every scored site
  is here. A predict frame holding a site whose stations were coded in the fit is refused
  (`ValueError`), because the earlier would read those codes. Alternative: carry the station in
  `learn_features.build_table`, a column every other model's table would gain.
- E8, the typed plan: `typed` and `typed_remaining_h` from the declared departure stated by τ,
  which the truth takes from the first app entry, where the earlier takes the latest entry before
  10:00; its typed minutes, typed energy and the flags for typed stays of 60 and 480 minutes are
  left out (`earlier_inputs.LEFT_OUT`): the cars tables do not hold them.
- E9, the driver: `linked` is the feature table's `user_known` (a user id in the cars table, not
  the rate fallback), where the earlier asks for a link recorded before 10:00; the history
  quantiles and the days since the last plug-in are over `learn_features`' one selection of a
  user's earlier sessions (same site, listed on an earlier date, plugged out by τ, only sessions
  plugged in at some τ), where the earlier takes every session of the driver that ended before
  10:00.
- E10, the plug-in hour: `learn_features`' `plug_in_clock_h`, hours from the date's local midnight
  on the plug-in's own clock text, so a car plugged in before a clock change on an earlier day is
  an hour off the earlier's elapsed-time reading.
- E11, the survival columns: NaN (no survival-tightened bound), as `earlier_grid.table` writes
  them.
- E12, threads: scikit-learn's default, every core (the earlier's parallel workers took one
  thread each). The trees' sums run in parallel, so the last digits of a fit can depend on the
  number of threads; on one machine two fits on the same rows agree. Alternative: limit the
  threads to one, which needs a package the project does not name as its own dependency.
- E13, an input with no value: a column of the input matrix missing on every fit row of a target
  is left out of that target's models (and of their predictions). scikit-learn 1.9.1 refuses such
  a column when it bins the inputs; a column with no value offers no split, so the trees are
  those the earlier's matrix would give. It happens on made-up tables (no declared departure, no
  history); whether it happens on the training units is not checked here. Alternative: refuse
  the fit.
"""

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from fleetreserve import earlier_grid, earlier_inputs
from fleetreserve.forecast_model import empty_table
from fleetreserve.learn_features import build_table
from fleetreserve.truth_build import TRAIN_UNITS

SETTINGS = {
    "learning_rate": 0.1,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "max_iter": 200,
    "early_stopping": False,
    "random_state": 0,
}
# target -> (label column, rows: "all" or "fleet")
TARGETS = {
    "stay": ("label_departure_h", "all"),
    "rate": ("label_rate_kw", "fleet"),
    "room": ("label_room_kwh", "fleet"),
}


def _stations(cars: pd.DataFrame) -> list:
    """(site, station) of each row, None where the station is empty (reading E7)."""
    pairs = zip(cars["site"].astype(str), cars["station"].astype(str), strict=True)
    return [(site, station) if station else None for site, station in pairs]


class Design:
    """The earlier input matrix, trees form (module text), fitted on the fit rows' inputs and
    stations."""

    def __init__(self, inputs: pd.DataFrame, stations: list) -> None:
        x = inputs.to_numpy(float)
        with warnings.catch_warnings():  # an input missing on every row is allowed
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(x, axis=0)
            spread = np.nanstd(x, axis=0)
        spread = np.where(np.isnan(spread), 0.0, spread)
        self.scale = np.where(spread > 0, spread, 1.0)
        self.mean = np.where(np.isnan(mean), 0.0, mean)
        self.stations = sorted({s for s in stations if s is not None})

    def __call__(self, inputs: pd.DataFrame, stations: list) -> np.ndarray:
        x = (inputs.to_numpy(float) - self.mean) / self.scale
        if not self.stations:
            return x
        code = {s: k for k, s in enumerate(self.stations)}
        idx = np.array([code.get(s, -1) for s in stations], dtype=float)
        return np.hstack([x, np.where(idx >= 0, idx, np.nan)[:, None]])


class EarlierGbmModel:
    name = "earlier_gbm"
    fit_units: tuple[str, ...] = TRAIN_UNITS

    def __init__(self, *, rho: float) -> None:
        self.rho = rho
        self.design: Design | None = None
        # target -> (the input columns used, one fitted model per level)
        self.models: dict[str, tuple[np.ndarray, list[HistGradientBoostingRegressor]]] = {}

    def fit(self, frame: pd.DataFrame) -> None:
        """One model per target and level on the pooled cars tables of `fit_units` (module
        text). Raises `ValueError` for no cars or a target with no row."""
        if len(frame) == 0:
            raise ValueError("earlier_gbm: there are no training cars to fit on")
        cars = frame.reset_index(drop=True)
        table = build_table(cars, rho=self.rho)
        inputs = earlier_inputs.numeric(table)
        stations = _stations(cars)
        design = Design(inputs, stations)
        x = design(inputs, stations)
        categorical = None
        if design.stations:
            categorical = np.zeros(x.shape[1], dtype=bool)
            categorical[-1] = True
        fleet = table["label_room_kwh"].notna().to_numpy(bool)
        models = {}
        for target, (label, rows) in TARGETS.items():
            y = table[label].to_numpy(float)
            if target == "stay":
                y = earlier_grid.stay_target(y)
            keep = np.isfinite(y) & (fleet if rows == "fleet" else True)
            if not keep.any():
                raise ValueError(f"earlier_gbm: no training row has a {target} label")
            columns = ~np.isnan(x[keep]).all(axis=0)  # reading E13
            mask = None
            if categorical is not None and categorical[columns].any():
                mask = categorical[columns]
            models[target] = (
                columns,
                [
                    HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=float(level),
                        categorical_features=mask,
                        **SETTINGS,
                    ).fit(x[keep][:, columns], y[keep])
                    for level in earlier_grid.LEVELS
                ],
            )
        self.design, self.models = design, models

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if self.design is None:
            raise ValueError("earlier_gbm: predict was called before fit")
        if len(frame) == 0:
            return empty_table()
        coded = {site for site, _ in self.design.stations}
        refused = sorted(coded & set(frame["site"].astype(str)))
        if refused:
            raise ValueError(
                f"earlier_gbm: the stations of the sites {refused} were coded in the fit, and the "
                "feature frame does not carry the station (reading E7)"
            )
        x = self.design(earlier_inputs.numeric(frame), [None] * len(frame))
        grids = {
            target: np.column_stack([m.predict(x[:, columns]) for m in fitted])
            for target, (columns, fitted) in self.models.items()
        }
        return earlier_grid.table(
            frame, kappas, self.rho, grids["stay"], grids["rate"], grids["room"]
        )


def make_model(matrix) -> EarlierGbmModel:
    return EarlierGbmModel(rho=matrix.rho)
