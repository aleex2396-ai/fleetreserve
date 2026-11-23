"""The common interface of the departure-forecast models (plan sections 2, 7, 13 WP8).

A model reaches the gate machinery (`gate_day.prepare_day`, `gate_calibrate.run_job`) only through
a forecast table: the model writes one, and `TableForecaster` reads it and serves the gate the two
calls `forecast_empirical.EmpiricalForecaster` serves.

Model (`DepartureModel`, a protocol; each model is a module `fleetreserve.models.<name>` with a
function `make_model(matrix)` that builds it from the run matrix, `gate_config.Matrix`):

- `name`: the model's name;
- `fit_units`: the units whose truth tables the model is fitted on (empty: nothing to fit). Only
  training units (`truth_build.TRAIN_UNITS`) may be named; `learn_run` refuses a model naming any
  other unit;
- `fit(frame)`: fit on `frame`, the cars tables of `fit_units` as `days_from_truth.UnitDays.cars`
  holds them (the rules applied to every row), one below the other; a learned model builds its
  features and labels from it with `learn_features.build_table`;
- `predict(frame, kappas)`: the forecast table (below) for the cars of `frame` at every level in
  `kappas`. What `frame` holds depends on one declared capability:
  - a learned model (the default): `frame` is `learn_features.feature_frame` of the units to
    forecast, the rows of counting days only: keys and features known at τ, no outcome of any
    car and no label (`learn_features` lists the columns left out and why). It cannot tell which
    cars the rules keep in a fleet, so it gives a row for every car of `frame`;
  - a model with `reads_other_day_outcomes = True`: `frame` is the units' cars tables as
    `UnitDays.cars` holds them, outcomes included, because a reference-sample model forecasts a
    day from the outcomes of the unit's other days, as the gate's empirical baseline does. A row is
    wanted for every car of a counting day's fleet (columns `counting` and `in_fleet` both true),
    and the table must not use a day's own outcomes for that day's cars. Only the models named in
    `OUTCOME_READERS` may declare it; `learn_run` refuses any other that does.

Forecast table: one row per car (key `unit`, `site`, `date`, `session_id`) and level `kappa`
(`TABLE_COLUMNS`). Numbers, each NaN where the model gives no forecast:

- `rho`: the unprotected share the energy columns were made for (one value for the whole table);
- `d_hat`: the kappa-quantile of departure given presence at τ, in hours from τ (may be
  fractional); NaN: the car has no forecast and is left unsteered;
- `r_hat`, `e_cap`: kappa-quantiles of rate (kW) and room (kWh); `e_need`: the (1 - kappa)-quantile
  of protected energy (kWh). Read only under the energy modes that use them;
- `survival_h1` to `survival_h8` (`day.MAX_SERVICE_HOURS`): the kappa-quantile of departure given
  presence at the start of service hour t (the survival-tightened bound, plan section 1); NaN
  keeps the bound of the hour before;
- `d_median`: the median departure (hours from τ), read only by the concordance index.

`write_table` writes the table as CSV (numbers written so that reading gives the same float back,
NaN as an empty field); `read_table` reads it; both refuse a table that `check_table` refuses.
`check_table` takes the levels of plan section 2 only (`forecast_empirical.KAPPAS`); with
`what_if=True` it also takes the levels off that grid (`forecast_empirical.WHAT_IF_KAPPAS`), and
`write_table`, `read_table`, `TableForecaster` and `TableFactory` pass the flag on. Only the
what-if of `dial_choice --what-if-kappa` sets it.

`TableForecaster(table)`: `forecast_car(truth, i, kappa, mode, departure)` and
`survival_bound(truth, i, hour, kappa)` as `EmpiricalForecaster` has them, from the table row of
car `i` of `truth` at `kappa`:

- energy modes (`forecast_empirical.MODES`): "known" takes the car's true rate, room and protected
  energy and reads only `d_hat`; "empirical" reads `r_hat`, `e_cap`, `e_need`; "user-rate" takes
  the true rate and reads `e_cap`, `e_need`. A NaN in a column read gives no forecast (None);
- oracle departure: `d_hat` is the car's true departure hour, as `EmpiricalForecaster` makes it;
  with "known" energy the table is not read at all (the oracle partner then does not depend on the
  model), otherwise the energy columns are read as above;
- a car with no row in the table, or a table made for another `rho`, raises `ValueError`: a model
  must state "no forecast" with NaN, never by leaving a car out;
- `survival_bound` for hours 1 to 8 reads `survival_h<hour>` (NaN gives None, so the gate keeps
  its earlier bound); another hour raises `ValueError`, as the table holds no other.

`TableFactory(table)`: a picklable forecaster factory for `gate_run.run_gate`.

Discrete survival curves (`survival_quantile`, `departure_columns`): a model that gives a
departure distribution as `S_j = P(D > t_j)` at increasing times `t_j` (hours from τ, given
presence at τ) puts the mass `S_(j-1) - S_j` at `t_j` (`S` is 1 before the first time). The lower
`level`-quantile is the smallest `t_j` whose distribution function exceeds `level`, the rule
`forecast_empirical.lower_quantile` applies to a sample (for a sample, `x_(k)` with
`k = floor(level * n) + 1` is the smallest value with more than `level * n` values at or below it);
given presence at hour h, the same on the times above h with `S` divided by `S(h)`. The
comparison allows `PROBABILITY_TOLERANCE`, so a curve made from a sample gives that sample's
quantile despite rounding. Where the curve stops before the quantile is reached, or nothing is
left at h, the result is NaN. To keep a bound below the true departure, the caller puts each
interval's mass at the interval's start.
"""

import csv
import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from fleetreserve.day import MAX_SERVICE_HOURS, ForecastCar
from fleetreserve.forecast_empirical import DEPARTURES, KAPPAS, MODES, WHAT_IF_KAPPAS

KEY_COLUMNS = ("unit", "site", "date", "session_id")
SURVIVAL_COLUMNS = tuple(f"survival_h{t}" for t in range(1, MAX_SERVICE_HOURS + 1))
NUMBER_COLUMNS = ("rho", "d_hat", "r_hat", "e_cap", "e_need", "d_median", *SURVIVAL_COLUMNS)
TABLE_COLUMNS = (*KEY_COLUMNS, "kappa", *NUMBER_COLUMNS)
PROBABILITY_TOLERANCE = 1e-9
MEDIAN = 0.5
# Model modules (`fleetreserve.models.<name>`) allowed to read the outcomes of a unit's other days
# in `predict`. Adding a name is a change for review: such a model is trusted not to use a day's own
# outcomes, which the shared path does not enforce.
OUTCOME_READERS = ("empirical",)
# Model modules allowed to read a look-ahead column (`model_inputs.LOOK_AHEAD_COLUMNS`, values of
# times after τ), and only in `learn_run`'s development evaluation (`model_inputs`, module text).
LOOK_AHEAD_MODELS = ("hazard_tuned_inputs_lookahead",)


class DepartureModel(Protocol):
    name: str
    fit_units: tuple[str, ...]
    # Optional; absent counts as False (module text).
    reads_other_day_outcomes: bool

    def fit(self, frame: pd.DataFrame) -> None: ...

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame: ...


# --- the forecast table --------------------------------------------------------------------------


def empty_table() -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype=str if c in KEY_COLUMNS else float) for c in TABLE_COLUMNS}
    )


def check_table(table: pd.DataFrame, what_if: bool = False) -> pd.DataFrame:
    """The table with its columns in order, keys as text and numbers as floats; raises
    `ValueError` for other columns, a level outside plan section 2's (with `what_if`, outside it
    and `WHAT_IF_KAPPAS`), a repeated key, an infinite number or more than one `rho`."""
    if not isinstance(table, pd.DataFrame) or list(table.columns) != list(TABLE_COLUMNS):
        raise ValueError(f"a forecast table has the columns {list(TABLE_COLUMNS)}")
    out = table.reset_index(drop=True).copy()
    for column in KEY_COLUMNS:
        out[column] = out[column].astype(str)
    for column in ("kappa", *NUMBER_COLUMNS):
        out[column] = out[column].astype(float)
    allowed = (*KAPPAS, *WHAT_IF_KAPPAS) if what_if else KAPPAS
    bad = sorted(set(out["kappa"]) - set(allowed))
    if bad:
        raise ValueError(f"forecast table: kappa must be one of {list(allowed)}, got {bad}")
    numbers = out[list(NUMBER_COLUMNS)].to_numpy(float)
    if np.isinf(numbers).any():
        raise ValueError("forecast table: a number is infinite")
    rhos = set(out["rho"])
    if len(rhos) > 1 or any(not 0 <= r < 1 for r in rhos):
        raise ValueError(f"forecast table: one rho in [0, 1) is needed, got {sorted(rhos)}")
    repeated = out.duplicated(list(KEY_COLUMNS) + ["kappa"])
    if repeated.any():
        first = out[repeated].iloc[0]
        raise ValueError(f"forecast table: a car and level appear twice: {dict(first)}")
    return out


def _text(value: float) -> str:
    value = float(value)
    return "" if math.isnan(value) else repr(value)


def write_table(table: pd.DataFrame, path, what_if: bool = False) -> None:
    """The table as CSV (module text)."""
    checked = check_table(table, what_if)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(TABLE_COLUMNS)
        keys = [checked[c].tolist() for c in KEY_COLUMNS]
        numbers = [checked[c].tolist() for c in ("kappa", *NUMBER_COLUMNS)]
        for k in range(len(checked)):
            writer.writerow([column[k] for column in keys] + [_text(n[k]) for n in numbers])


def read_table(path, what_if: bool = False) -> pd.DataFrame:
    """The table `write_table` wrote; raises `ValueError` when `check_table` refuses it."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ("kappa", *NUMBER_COLUMNS):
        if column in raw.columns:
            raw[column] = [math.nan if text == "" else float(text) for text in raw[column]]
    return check_table(raw, what_if)


# --- the forecaster the gate reads ---------------------------------------------------------------


class TableForecaster:
    """`forecast_car` and `survival_bound` from a forecast table (module text)."""

    def __init__(self, table: pd.DataFrame, what_if: bool = False) -> None:
        checked = check_table(table, what_if)
        rhos = set(checked["rho"])
        self.rho = rhos.pop() if rhos else None
        keys = zip(*(checked[c] for c in KEY_COLUMNS), checked["kappa"], strict=True)
        values = checked[list(NUMBER_COLUMNS)].to_numpy(float)
        self._rows = {key: values[k] for k, key in enumerate(keys)}
        self._column = {name: k for k, name in enumerate(NUMBER_COLUMNS)}

    def _value(self, row: np.ndarray, name: str) -> float:
        return float(row[self._column[name]])

    def _row(self, truth, i: int, kappa: float) -> np.ndarray:
        car = truth.fleet[i]
        key = (truth.day.unit, car.site, truth.record.date, car.session_id, float(kappa))
        found = self._rows.get(key)
        if found is None:
            raise ValueError(f"the forecast table has no row for {key}")
        if self.rho != truth.day.rho:
            raise ValueError(
                f"the forecast table is for rho {self.rho}, the day's is {truth.day.rho}"
            )
        return found

    def forecast_car(
        self, truth, i: int, kappa: float, mode: str, departure: str = "empirical"
    ) -> ForecastCar | None:
        """Car `i`'s forecast at level `kappa`, or None when the table gives none."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if departure not in DEPARTURES:
            raise ValueError(f"departure must be one of {DEPARTURES}, got {departure!r}")
        true = truth.day.cars[i]
        rho = truth.day.rho
        if mode == "known" and departure == "oracle":
            return ForecastCar(float(true.departure), true.rate, true.room, true.need(rho))
        row = self._row(truth, i, kappa)
        d_hat = float(true.departure) if departure == "oracle" else self._value(row, "d_hat")
        if math.isnan(d_hat):
            return None
        if mode == "known":
            return ForecastCar(d_hat, true.rate, true.room, true.need(rho))
        rate = true.rate if mode == "user-rate" else self._value(row, "r_hat")
        cap = self._value(row, "e_cap")
        need = self._value(row, "e_need")
        if any(math.isnan(v) for v in (rate, cap, need)):
            return None
        return ForecastCar(d_hat, rate, cap, need)

    def survival_bound(self, truth, i: int, hour: int, kappa: float) -> float | None:
        """The table's survival-tightened bound of car `i` at service hour `hour`, or None."""
        if isinstance(hour, bool) or not isinstance(hour, int):
            raise ValueError(f"hour must be a whole number, got {hour!r}")
        if not 1 <= hour <= MAX_SERVICE_HOURS:
            raise ValueError(f"the forecast table holds hours 1 to {MAX_SERVICE_HOURS}, not {hour}")
        value = self._value(self._row(truth, i, kappa), f"survival_h{hour}")
        return None if math.isnan(value) else value


class TableFactory:
    """A picklable factory for `gate_run.run_gate`: each unit's `TableForecaster`."""

    def __init__(self, table: pd.DataFrame, what_if: bool = False) -> None:
        self.what_if = what_if
        self.table = check_table(table, what_if)

    def __call__(self, unit_days, matrix) -> TableForecaster:
        return TableForecaster(self.table[self.table["unit"] == unit_days.unit], self.what_if)


# --- discrete survival curves --------------------------------------------------------------------


def _curve(times, survival) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    s = np.asarray(survival, dtype=float)
    if t.ndim != 1 or t.shape != s.shape or t.size == 0:
        raise ValueError("times and survival must be two lists of the same, non-zero length")
    if not (np.isfinite(t).all() and np.isfinite(s).all()):
        raise ValueError("times and survival must be finite")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must increase")
    if s.min() < 0 or s.max() > 1 or np.any(np.diff(s) > 0):
        raise ValueError("survival must lie in [0, 1] and must not increase")
    return t, s


def survival_quantile(times, survival, level: float, after: float | None = None) -> float:
    """The lower `level`-quantile of the curve's departure, given departure above `after` when it
    is given (module text); NaN when it cannot be read from the curve."""
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    t, s = _curve(times, survival)
    if after is None:
        alive, later = 1.0, np.ones(t.size, dtype=bool)
    else:
        before = np.flatnonzero(t <= after)
        alive = 1.0 if before.size == 0 else float(s[before[-1]])
        later = t > after
    if alive <= 0:
        return math.nan
    beyond = (alive - s) > level * alive + PROBABILITY_TOLERANCE
    found = np.flatnonzero(later & beyond)
    return math.nan if found.size == 0 else float(t[found[0]])


def departure_columns(times, survival, kappas: Sequence[float]) -> dict[float, dict]:
    """Per level, the forecast table's departure columns (`d_hat`, `survival_h1` to
    `survival_h8`, `d_median`) from one car's survival curve."""
    median = survival_quantile(times, survival, MEDIAN)
    out = {}
    for kappa in kappas:
        row = {"d_hat": survival_quantile(times, survival, kappa), "d_median": median}
        for t, column in enumerate(SURVIVAL_COLUMNS, start=1):
            row[column] = survival_quantile(times, survival, kappa, after=t)
        out[float(kappa)] = row
    return out
