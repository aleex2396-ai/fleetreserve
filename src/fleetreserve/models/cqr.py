"""CQR-GBM: the pooled LightGBM quantile models (`models.lgbm`) with the departure quantile
conformalised (conformalised quantile regression, Romano et al. (2019)); a forecast baseline of
plan section 7.

Split, training units only (`fit_units` is `truth_build.TRAIN_UNITS`; development units are only
forecast): the days of the pooled training cars tables (a day is a unit, site and date) are put in
a seeded random order (`SPLIT_SEED`) and the first `CALIBRATION_SHARE` of them (rounded, at least
one day, at least one day left) form the calibration split; the rest form the fit split. A session
plugged in over several days appears on several days with the same plug-out; a calibration row
whose session (unit and session id) also has a row in the fit split is dropped, so no departure
seen in fitting is scored in calibration. The two splits share no day.

Fit: every quantile model of `models.lgbm` (same features, rows, settings and levels) on the fit
split only.

Calibration, per kappa, on the calibration rows with a departure label (every car plugged in at τ,
as the departure model is fitted): the score of row i is `s_i = q_kappa(x_i) - y_i`, the fitted
kappa-quantile (as `models.lgbm` gives it, raised to 0) minus the exact departure (hours from
τ). With `n` scores and `k = ceil((1 - kappa) * (n + 1))` (kappa taken as the exact decimal it is
written as), the adjustment `Q_kappa` is the k-th smallest score. The forecast departure is the
one-sided lower bound `d_hat = q_kappa(x) - Q_kappa`, raised to 0 when below it. For a new car
exchangeable with the calibration rows, `P(departure < q_kappa(x) - Q_kappa) <= kappa` (Romano
et al. (2019), Theorem 1, applied to one side); raising to 0 keeps it, as every departure is at
least 0. Cars of one day are not independent and the scored sites are not training sites, so for
the development units this is a construction, not a guarantee; the guarantee of the offer is the
calibration of plan section 6. When `k > n` the calibration split is too small for that kappa and
`fit` raises `ValueError`.

Not conformalised: `d_median` (the median model, read only by the concordance index) and the rate
and energy columns (`r_hat`, `e_cap`, `e_need`, from the fit-split models as in `models.lgbm`).
The survival columns follow `models.lgbm.SURVIVAL_BOUND` (left NaN).
"""

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pandas as pd

from fleetreserve.learn_features import build_table
from fleetreserve.models.lgbm import (
    LABELS,
    QuantileSet,
    features,
    forecast_rows,
    levels_of,
    matrix_kappas,
    quantile_columns,
    rows_for,
)
from fleetreserve.truth_build import TRAIN_UNITS

SPLIT_SEED = 0
CALIBRATION_SHARE = 0.25
DAY_KEY = ("unit", "site", "date")


def calibration_days(table: pd.DataFrame) -> set[tuple[str, str, str]]:
    """The days of the calibration split (module text)."""
    days = sorted(set(zip(*(table[c].astype(str) for c in DAY_KEY), strict=True)))
    if len(days) < 2:
        raise ValueError(f"a calibration split needs at least two training days, got {len(days)}")
    order = np.random.default_rng(SPLIT_SEED).permutation(len(days))
    n_calibration = min(max(round(CALIBRATION_SHARE * len(days)), 1), len(days) - 1)
    return {days[j] for j in order[:n_calibration]}


def split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fit split, calibration split) of a `build_table` table (module text)."""
    chosen = calibration_days(table)
    day = pd.Series(list(zip(*(table[c].astype(str) for c in DAY_KEY), strict=True)))
    in_calibration = day.isin(chosen).to_numpy()
    fit = table[~in_calibration].reset_index(drop=True)
    calibration = table[in_calibration]
    fitted_sessions = set(zip(fit["unit"], fit["session_id"], strict=True))
    seen = [
        key in fitted_sessions
        for key in zip(calibration["unit"], calibration["session_id"], strict=True)
    ]
    return fit, calibration[~np.array(seen, dtype=bool)].reset_index(drop=True)


def conformal_rank(kappa: float, n: int) -> int:
    """`k = ceil((1 - kappa) * (n + 1))`, the 1-based rank of the adjustment (module text)."""
    if not 0 < kappa < 1:
        raise ValueError(f"kappa must be in (0, 1), got {kappa}")
    return math.ceil((1 - Fraction(str(kappa))) * (n + 1))


def adjustment(scores: np.ndarray, kappa: float) -> float:
    """`Q_kappa`, the k-th smallest score; raises `ValueError` when `k > n`."""
    n = scores.size
    k = conformal_rank(kappa, n)
    if k > n:
        raise ValueError(
            f"the calibration split has {n} departure rows; kappa {kappa} needs at least "
            f"{math.ceil(1 / Fraction(str(kappa))) - 1}"
        )
    return float(np.sort(scores)[k - 1])


class CqrModel:
    name = "cqr"
    fit_units: tuple[str, ...] = TRAIN_UNITS

    def __init__(self, *, rho: float, kappas: Sequence[float]) -> None:
        self.rho = rho
        self.kappas = tuple(sorted({float(k) for k in kappas}))
        self.levels = levels_of(self.kappas)
        self.quantiles: QuantileSet | None = None
        self.adjustments: dict[float, float] = {}
        self.split_days: tuple[set, set] | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit on the fit split, calibrate the departure quantiles on the calibration split."""
        table = build_table(frame, rho=self.rho)
        fit, calibration = split(table)
        self.split_days = tuple(
            set(zip(*(part[c] for c in DAY_KEY), strict=True)) for part in (fit, calibration)
        )
        self.quantiles = QuantileSet(self.levels).fit(fit)
        rows = calibration[rows_for(calibration, "departure")]
        x = features(rows)
        y = rows[LABELS["departure"][0]].to_numpy(float)
        self.adjustments = {}
        for kappa in self.kappas:
            fitted = self.quantiles.predict(x, "departure", kappa)
            self.adjustments[kappa] = adjustment(fitted - y, kappa)

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if self.quantiles is None:
            raise ValueError("predict before fit")
        x = features(frame)
        columns = quantile_columns(self.quantiles, x, kappas)
        for kappa in kappas:
            k = float(kappa)
            fitted = columns[k]["d_hat"]
            columns[k]["d_hat"] = np.maximum(fitted - self.adjustments[k], 0.0)
        return forecast_rows(frame, kappas, self.rho, columns)


def make_model(matrix) -> CqrModel:
    return CqrModel(rho=matrix.rho, kappas=matrix_kappas(matrix))
