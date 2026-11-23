"""The earlier analysis's population model (plan section 7, "the earlier GBM and population
models"; section 12) as a departure model (`forecast_model.DepartureModel`).

The earlier rule (package `ev-fleet-offer`, commit 05a6cf9, `baselines.Population`, arms 2 and 5 of
that analysis): for a car plugged in `e` hours at the decision time, take the sessions that ended
in the history window and lasted at least `e` hours; forecast its remaining stay by the
Kaplan-Meier lower quantile of their stays (`models.lower_quantile` there: the largest `x` with an
estimated chance of at most the level that the stay ends before `e + x`, given it lasted `e`;
with every session ended it is the empirical one), and its energy by the empirical lower quantile
(`numpy.quantile`, `method="inverted_cdf"`: the smallest value whose share reaches the level) of
the energy each of those sessions took in the 10-hour window starting `e` hours after its plug-in;
both at the 16 levels of `earlier_grid.LEVELS`, then sorted and clipped (`earlier_grid.tidy`).

Here, `fit` takes the pooled cars tables of the training units and `predict` reads the feature
frame's `hours_since_plug_in` (that `e`, known at τ) and the keys, nothing else; the table is read
off the grids by `earlier_grid.table`. What is kept as it was: the condition `stay >= e`, the
survival curve (`survival_curve`, the earlier `models.km_curve` with every stay ended), the lower
quantile with its relative tolerance `REL_TOL` (`remaining_stay`), the empirical lower quantile,
the 16 levels, the sort and the 10-hour clip of the stay. What the interface forces, each a
reading with the alternative not taken:

- E1, the history: the earlier refits each month on the scored site's own sessions of the 90 days
  before (the plan's per-site history window). Here one fit on the training units pooled
  (`fit_units` = `truth_build.TRAIN_UNITS`): a model is fitted on training units only, and the
  scored units are other sites. Alternative: read the scored unit's own earlier days as an outcome
  reader (`forecast_model.OUTCOME_READERS`), a change to the list of models allowed to read
  outcomes, not made here.
- E2, the sessions: the earlier takes every session that ended in the window, whatever its clock
  times. The cars tables hold only the sessions plugged in at a decision time τ, so here the
  population is those, each once (by unit and session id; a session listed on several days is
  one stay), its stay `plug_out - plug_in` in hours. A long session is more likely to span a τ,
  so this sample holds more long stays than the earlier one. Alternative: every session the
  loaders read, a data path beside the truth build that the model interface does not give.
- E3, the departure: the remaining stay `stay - e` is the hours from τ to plug-out, the
  departure of the forecast table, so `d_hat` and `d_median` are the stay grid at kappa and 0.5,
  clipped to the earlier horizon of 10 hours as the earlier does (a forecast below the truth
  stays below it). NaN where no session lasted `e` hours.
- E4, rate and room: the earlier forecasts no rate (its offer used the fixed 6.7 kW of its site)
  and an energy measured from a charging-done time that the cars tables do not hold. Here the
  same empirical lower quantile at the 16 levels of the training cars' rate and room (the labels
  `label_rate_kw`, `label_room_kwh` of `learn_features`), over the car-days the rules keep in a
  fleet (a room that is a number) whose session lasted at least `e` hours. Alternatives: re-apply
  the truth's room rule at a decision time `e` hours after each session's plug-in (needs the
  energy's timing within the session, which the tables do not hold); for the rate, the fixed
  6.7 kW (a constant of one site's chargers).
- E5, protected energy and survival columns: as `earlier_grid.table` reads them, `e_need` the
  room grid at 1 - kappa times `(1 - rho)` (the earlier sets no protected share), the survival
  columns NaN.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fleetreserve import earlier_grid
from fleetreserve.forecast_model import empty_table
from fleetreserve.truth_build import TRAIN_UNITS

REL_TOL = 1e-12


def survival_curve(stays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct stays in increasing order and the Kaplan-Meier estimate of staying longer
    than each, every stay ended (the earlier `km_curve`)."""
    stays = np.asarray(stays, dtype=float)
    times, leaving = np.unique(stays, return_counts=True)
    at_risk = stays.size - np.searchsorted(np.sort(stays), times, side="left")
    return times, np.cumprod(1.0 - leaving / at_risk)


def remaining_stay(times: np.ndarray, survival: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
    """Cars by `earlier_grid.LEVELS`: per car plugged in `elapsed` hours, the earlier lower
    quantile of the remaining stay (module text); NaN where it is not identified. The first
    stay whose survival falls below the threshold is found by a sorted search, the same
    comparison the earlier makes one by one (the curve does not increase)."""
    elapsed = np.asarray(elapsed, dtype=float)
    out = np.full((elapsed.size, len(earlier_grid.LEVELS)), np.nan)
    if times.size == 0:
        return out
    start = np.searchsorted(times, elapsed, side="left")
    surviving = np.where(start == 0, 1.0, survival[np.maximum(start - 1, 0)])
    for k, level in enumerate(earlier_grid.LEVELS):
        threshold = (1.0 - float(level)) * surviving * (1.0 - REL_TOL)
        first = np.searchsorted(-survival, -threshold, side="right")
        ok = (surviving > 0.0) & (first < times.size)
        out[ok, k] = times[first[ok]] - elapsed[ok]
    return out


def _lasted_quantiles(stays: np.ndarray, values: np.ndarray, elapsed: np.ndarray) -> np.ndarray:
    """Cars by levels: the empirical lower quantiles of `values` over the rows whose stay (sorted
    in `stays`) is at least the car's `elapsed`; NaN where no row lasted."""
    out = np.full((elapsed.size, len(earlier_grid.LEVELS)), np.nan)
    start = np.searchsorted(stays, elapsed, side="left")
    for s in np.unique(start):
        if s < values.size:
            out[start == s] = np.quantile(values[s:], earlier_grid.LEVELS, method="inverted_cdf")
    return out


def _stays(cars: pd.DataFrame) -> np.ndarray:
    plug_in = pd.to_datetime(cars["plug_in"], utc=True, format="ISO8601")
    plug_out = pd.to_datetime(cars["plug_out"], utc=True, format="ISO8601")
    return ((plug_out - plug_in) / pd.Timedelta(hours=1)).to_numpy(float)


class EarlierPopulationModel:
    name = "earlier_population"
    fit_units: tuple[str, ...] = TRAIN_UNITS

    def __init__(self, *, rho: float) -> None:
        self.rho = rho
        self._curve: tuple[np.ndarray, np.ndarray] | None = None
        self._fleet: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        """The training sessions' survival curve and the fleet car-days' stays, rates and rooms
        (module text). Raises `ValueError` for no cars or a stay that is not a finite number."""
        if len(frame) == 0:
            raise ValueError("earlier_population: there are no training cars to fit on")
        cars = frame.reset_index(drop=True)
        stays = _stays(cars)
        if not np.isfinite(stays).all():
            raise ValueError("earlier_population: a training stay is not a finite number")
        once = ~cars.duplicated(["unit", "session_id"]).to_numpy(bool)
        self._curve = survival_curve(stays[once])
        rooms = cars["room"].astype(float).to_numpy()
        fleet = np.isfinite(rooms)
        order = np.argsort(stays[fleet], kind="stable")
        rates = cars["rate"].astype(float).to_numpy()[fleet][order]
        self._fleet = (stays[fleet][order], rates, rooms[fleet][order])

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if self._curve is None or self._fleet is None:
            raise ValueError("earlier_population: predict was called before fit")
        if len(frame) == 0:
            return empty_table()
        elapsed = frame["hours_since_plug_in"].to_numpy(float)
        stays, rates, rooms = self._fleet
        return earlier_grid.table(
            frame,
            kappas,
            self.rho,
            remaining_stay(*self._curve, elapsed),
            _lasted_quantiles(stays, rates, elapsed),
            _lasted_quantiles(stays, rooms, elapsed),
        )


def make_model(matrix) -> EarlierPopulationModel:
    return EarlierPopulationModel(rho=matrix.rho)
