"""Shared parts of the survival-regression forecast baselines of plan section 7, Cox and Weibull AFT
(`models.cox`, `models.weibull_aft`): their training rows, their covariates, a fit that fails on
non-convergence, and their forecast table.

Training rows (`training_rows`): `learn_features.build_table` of the cars tables given to `fit`,
the rows `models.lgbm.rows_for(table, "departure")` selects (a departure label is a number); the
duration is `label_departure_h`, hours from τ to plug-out. Every row is an observed plug-out, so
nothing is censored. A duration that is not a finite number above 0 raises `ValueError`.

Covariates (`Covariates`): `learn_features.FEATURE_COLUMNS` only, as numbers, and nothing else of
the frame. Fitted on the training rows only and applied unchanged to any frame:

- a column with a missing value in training gets a 0/1 column `<name>_missing` (1 where missing);
- a missing value is replaced by the column's training median;
- a column (feature or `_missing`) with no number in training, or one value only, is dropped;
- every kept column is standardised with its training mean and standard deviation (population).

An infinite feature value in training raises `ValueError`.

Departure columns (`curve_columns`), from a model's two functions of the cars: the lower
`level`-quantile of departure given departure above `after` hours from τ, and the probability of
departure above `after`. Per level kappa: `d_hat` at kappa and `d_median` at 0.5, both with
`after` 0; `survival_h<t>` at kappa with `after` t, NaN where the probability of departure above
t is below `MIN_PRESENCE_PROBABILITY`.

Rate, room and protected energy (`r_hat`, `e_cap`, `e_need`): the columns `models.km` gives,
from a `models.km.KMModel` fitted on the same cars tables (`fit_energy`) and given the frame the
model forecasts (`forecast_table`); both models give it `model_frame` of their frame, the keys and
`FEATURE_COLUMNS` only.

Warnings: during a lifelines fit (`fit_checked`) and Cox's partial-hazard prediction, a
`DeprecationWarning` attributed to lifelines' own code is not shown
(`lifelines_deprecations_ignored`).
"""

import math
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines.exceptions import ConvergenceError, ConvergenceWarning

from fleetreserve.forecast_model import KEY_COLUMNS, MEDIAN, SURVIVAL_COLUMNS, empty_table
from fleetreserve.learn_features import FEATURE_COLUMNS, build_table
from fleetreserve.models.km import KMModel
from fleetreserve.models.lgbm import rows_for

MIN_PRESENCE_PROBABILITY = 1e-3
MISSING_SUFFIX = "_missing"
DEPARTURE_COLUMNS = ("d_hat", "d_median", *SURVIVAL_COLUMNS)


def training_rows(frame: pd.DataFrame, rho: float) -> tuple[pd.DataFrame, np.ndarray]:
    """(the training rows of `build_table`, their durations in hours) (module text)."""
    table = build_table(frame, rho=rho)
    table = table[rows_for(table, "departure")].reset_index(drop=True)
    durations = table["label_departure_h"].to_numpy(float)
    if durations.size == 0:
        raise ValueError("no training row has a departure label")
    if not (np.isfinite(durations).all() and (durations > 0).all()):
        raise ValueError("a training departure is not a finite number above 0")
    return table, durations


def _raw(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(FEATURE_COLUMNS)].astype(float).reset_index(drop=True)


@dataclass(frozen=True)
class Covariates:
    """The covariate rule of the module text, as fitted on the training rows."""

    medians: dict[str, float]
    flagged: tuple[str, ...]
    names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray

    @staticmethod
    def _expand(raw: pd.DataFrame, medians: dict, flagged: Sequence[str]) -> pd.DataFrame:
        out = {}
        for column in FEATURE_COLUMNS:
            out[column] = raw[column].fillna(medians[column]).to_numpy(float)
            if column in flagged:
                out[column + MISSING_SUFFIX] = raw[column].isna().to_numpy(float)
        return pd.DataFrame(out)

    @classmethod
    def fit(cls, table: pd.DataFrame) -> "Covariates":
        raw = _raw(table)
        if np.isinf(raw.to_numpy(float)).any():
            raise ValueError("a training feature value is infinite")
        medians = {c: float(raw[c].median()) for c in FEATURE_COLUMNS}
        flagged = tuple(c for c in FEATURE_COLUMNS if raw[c].isna().any())
        expanded = cls._expand(raw, medians, flagged)
        values = expanded.to_numpy(float)
        varies = np.isfinite(values).all(axis=0) & (np.ptp(values, axis=0) > 0)
        names = tuple(expanded.columns[varies])
        kept = values[:, varies]
        return cls(medians, flagged, names, kept.mean(axis=0), kept.std(axis=0))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The standardised covariates of the rows of `frame`, read from `FEATURE_COLUMNS`."""
        expanded = self._expand(_raw(frame), self.medians, self.flagged)[list(self.names)]
        return (expanded - self.means) / self.scales


@contextmanager
def lifelines_deprecations_ignored():
    """Inside: a `DeprecationWarning` attributed to lifelines' own code is not shown."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"lifelines\.")
        yield


def fit_checked(name: str, fit: Callable[[], object]) -> object:
    """`fit()`, with a lifelines convergence warning or error raised as `ValueError`, and
    lifelines' deprecation warnings not shown."""
    with lifelines_deprecations_ignored(), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            return fit()
        except (ConvergenceWarning, ConvergenceError) as problem:
            raise ValueError(f"{name}: the fit did not converge: {problem}") from problem


def curve_columns(
    quantile: Callable[[float, float], np.ndarray],
    presence: Callable[[float], np.ndarray],
    kappas: Sequence[float],
) -> dict[float, dict[str, np.ndarray]]:
    """Per level, the departure columns of the module text."""
    median = quantile(MEDIAN, 0.0)
    out = {}
    for kappa in kappas:
        k = float(kappa)
        if not 0 < k < 1:
            raise ValueError(f"a level is in (0, 1), got {kappa}")
        row = {"d_hat": quantile(k, 0.0), "d_median": median}
        for hour, column in enumerate(SURVIVAL_COLUMNS, start=1):
            low = presence(float(hour)) < MIN_PRESENCE_PROBABILITY
            row[column] = np.where(low, math.nan, quantile(k, float(hour)))
        out[k] = row
    return out


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The keys and `FEATURE_COLUMNS` of `frame`: the only columns the models read."""
    return frame[[*KEY_COLUMNS, *FEATURE_COLUMNS]].reset_index(drop=True)


def fit_energy(frame: pd.DataFrame, rho: float, min_cell: int) -> KMModel:
    """The `KMModel` whose rate, room and protected-energy columns the models give."""
    km = KMModel(rho=rho, min_cell=min_cell)
    km.fit(frame)
    return km


def forecast_table(
    frame: pd.DataFrame, kappas: Sequence[float], km: KMModel, columns: dict
) -> pd.DataFrame:
    """The forecast table: `km`'s table for `frame`, its departure columns replaced by `columns`
    (`curve_columns` of the rows of `frame`)."""
    if len(frame) == 0:
        return empty_table()
    table = km.predict(frame, kappas)
    for name in DEPARTURE_COLUMNS:
        table[name] = np.concatenate([columns[float(k)][name] for k in kappas])
    return table
