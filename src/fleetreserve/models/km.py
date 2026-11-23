"""The survival-curve baseline of WP8 (plan section 7, "Kaplan-Meier (site, arrival hour)") as a
departure model (`forecast_model.DepartureModel`), the third model of WP8.

Fitted on the training units only (`fit_units` is `truth_build.TRAIN_UNITS`, all eight), from their
cars tables. For each cell, the survival curve of departure measured from the decision time τ:
`S(t)` is the share of the cell's cars still plugged in `t` hours after τ. Every plug-out of the
training units is observed, so nothing is censored and the Kaplan-Meier estimate is the empirical
survival curve; its quantiles are the quantiles `forecast_empirical.lower_quantile` takes of the
same sample (a test checks this). The curve is read through `forecast_model.survival_quantile` and
`departure_columns`.

Sites: the scored units are different sites from the training units, so a site cell cannot be
fitted for them. The cells are pooled across all training sites and keyed by arrival hour only: the
clock hour of plug-in, with cars plugged in on an earlier calendar day as their own cell
(`forecast_empirical.EARLIER_DAY_CELL`), as the empirical forecaster has it. A cell with fewer than
the matrix's `min_cell` cars falls back to all training cars (a cell never seen in training has no
cars and so falls back too).

Per car and level kappa, with the car's cell chosen by its arrival hour and `predict` reading the
feature frame's `arrival_hour` and `arrived_earlier_day` only:

- `d_hat`, `d_median`: the lower kappa-quantile and the median of the cell's curve;
- `survival_h1` to `survival_h8`: the lower kappa-quantile of the cell's curve given presence at the
  start of hour t, the curve of the cell if at least `min_cell` of its cars were still plugged in
  then, else that of all training cars (the rule `forecast_empirical` reading 5 applies to its
  survival bound); NaN where no training car was still plugged in;
- `r_hat`, `e_cap`: the kappa-quantiles of the rate and the room, `e_need` the (1 - kappa)-quantile
  of the protected energy `(1 - rho) * room` (plan section 7), over the cars the rules keep in a
  fleet (`learn_features` labels) of the cell, or of all training cars when the cell fell back.
  Should the chosen group hold no fleet car, all training fleet cars are used.

Nothing else is read: no date, user history or event column, so the forecast of a car depends on its
arrival hour alone.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fleetreserve.forecast_empirical import EARLIER_DAY_CELL, lower_quantile, upper_quantile
from fleetreserve.forecast_model import (
    KEY_COLUMNS,
    SURVIVAL_COLUMNS,
    TABLE_COLUMNS,
    departure_columns,
    empty_table,
)
from fleetreserve.learn_features import build_table
from fleetreserve.truth_build import TRAIN_UNITS

DEPARTURE_COLUMNS = ("d_hat", "d_median", *SURVIVAL_COLUMNS)


@dataclass(frozen=True)
class _Sample:
    """The training cars of one cell: departures (hours from τ, sorted) of every car plugged in at
    τ, rates and rooms of the cars the rules keep in a fleet."""

    departures: np.ndarray
    rates: np.ndarray
    rooms: np.ndarray

    def curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Distinct departure times and `S_j = P(D > t_j)` (no censoring)."""
        n = self.departures.size
        times = np.unique(self.departures)
        return times, 1.0 - np.searchsorted(self.departures, times, side="right") / n


def _cell(arrival_hour, arrived_earlier_day) -> np.ndarray:
    return np.where(np.asarray(arrived_earlier_day, bool), EARLIER_DAY_CELL, arrival_hour).astype(
        int
    )


class KMModel:
    name = "km"
    fit_units: tuple[str, ...] = TRAIN_UNITS

    def __init__(self, *, rho: float, min_cell: int) -> None:
        self.rho = rho
        self.min_cell = min_cell
        self._pooled: _Sample | None = None
        self._cells: dict[int, _Sample] = {}

    def fit(self, frame: pd.DataFrame) -> None:
        """Group the training cars by cell (module text). Raises `ValueError` for no cars or a
        departure that is not a finite number."""
        if len(frame) == 0:
            raise ValueError("km: there are no training cars to fit on")
        table = build_table(frame, rho=self.rho)
        departures = table["label_departure_h"].to_numpy(float)
        if not np.isfinite(departures).all():
            raise ValueError("km: a training departure is not a finite number")
        rates = table["label_rate_kw"].to_numpy(float)
        rooms = table["label_room_kwh"].to_numpy(float)
        kept = np.isfinite(rooms)  # NaN: the rules leave the car out of its fleet
        cell = _cell(table["arrival_hour"].to_numpy(int), table["arrived_earlier_day"])

        def sample(rows: np.ndarray) -> _Sample:
            fleet = rows & kept
            return _Sample(np.sort(departures[rows]), rates[fleet], rooms[fleet])

        self._pooled = sample(np.ones(len(table), dtype=bool))
        self._cells = {int(c): sample(cell == c) for c in np.unique(cell)}

    def _numbers(self, cell: int, kappas: Sequence[float]) -> dict[float, dict]:
        """Per level, the table's numbers of one cell (module text)."""
        own = self._cells.get(cell)
        pooled = self._pooled
        if own is None or own.departures.size < self.min_cell:
            chosen = pooled
            columns = departure_columns(*pooled.curve(), kappas)
        else:
            chosen = own
            columns = departure_columns(*own.curve(), kappas)
            fallback = departure_columns(*pooled.curve(), kappas)
            for hour, name in enumerate(SURVIVAL_COLUMNS, start=1):
                if int((own.departures > hour).sum()) < self.min_cell:
                    for kappa in columns:
                        columns[kappa][name] = fallback[kappa][name]
        fleet = chosen if chosen.rooms.size else pooled
        out = {}
        for kappa in kappas:
            row = dict(columns[float(kappa)])
            if fleet.rooms.size:
                row.update(
                    r_hat=lower_quantile(fleet.rates, kappa),
                    e_cap=lower_quantile(fleet.rooms, kappa),
                    e_need=upper_quantile((1 - self.rho) * fleet.rooms, kappa),
                )
            else:
                row.update(r_hat=math.nan, e_cap=math.nan, e_need=math.nan)
            out[float(kappa)] = row
        return out

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if self._pooled is None:
            raise ValueError("km: predict was called before fit")
        if len(frame) == 0:
            return empty_table()
        cells = _cell(frame["arrival_hour"].to_numpy(int), frame["arrived_earlier_day"])
        numbers = {int(c): self._numbers(int(c), kappas) for c in np.unique(cells)}
        parts = []
        for kappa in kappas:
            part = frame[list(KEY_COLUMNS)].reset_index(drop=True)
            part["kappa"] = float(kappa)
            part["rho"] = float(self.rho)
            for column in (*DEPARTURE_COLUMNS, "r_hat", "e_cap", "e_need"):
                part[column] = [numbers[int(c)][float(kappa)][column] for c in cells]
            parts.append(part[list(TABLE_COLUMNS)])
        return pd.concat(parts, ignore_index=True)


def make_model(matrix) -> KMModel:
    return KMModel(rho=matrix.rho, min_cell=matrix.min_cell)
