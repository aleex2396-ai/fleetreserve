"""Which days of a unit the cold start reads at k, and the frame a model forecasts from at k (plan
section 11, outcome O9: "Cold start, k=0, 7, 14, 28, 56 local days in time order").

Everything here reads one unit's `days_from_truth.UnitDays`; nothing is fitted, solved or written.

Time order. A unit's counting days are `UnitDays.days`, which `days_from_truth.unit_days` orders
by date, then site (reading C5 of `cold_start`); `day_keys` refuses a unit whose days are in
another order or whose (date, site) repeats. `k` counts counting days (reading C3).

At k (`LOCAL_DAYS`; any whole number from 0 to `largest` for the arithmetic here):

- the local days are the unit's first k counting days (`local_indices`), and only they
  calibrate;
- the local rows (`local_rows`) are the rows of the unit's cars table whose (date, site) is at or
  before the k-th counting day's, compared as text (ISO dates, then site names): the k local
  days' rows and the rows of the days that do not count among them, with every column as the
  table holds it, outcomes included (reading C4). At k = 0 there is none;
- the end of the local period (`local_end`) is the latest decision time τ of the k local days:
  for a unit of one site the k-th counting day's τ, for a pooled unit the latest τ of its local
  site-days (reading C2a). At k = 0 there is none;
- the rows known at the end of the local period (`known_rows`) are the local rows whose session
  had plugged out by then (`plug_out` at or before it; a plug-out that cannot be read is not
  known). A session still plugged in at that time is left out with all its rows, whatever day it
  later leaves on: so every session of the k-th counting day's fleet, which is plugged in at that
  day's τ, and every longer session of an earlier local day that had not left by then;
- the scored days (`scored_indices`) are the counting days after the first `largest` (the
  largest k, 56), the same for every k (reading C6). A unit with no counting day after them has
  no scored day.

The frame at k (`frame_at`), the frame `predict` is given for the unit at k: the rows of the k
local days and of every scored day, each as `learn_run.forecast_frame` builds it (the features of
`learn_features.build_table`, the model's extra input columns, `feature_frame`'s columns, counting
rows only):

- the local days' rows from the local rows (each day's history then holds the sessions that had
  plugged out by that day's own τ, as the whole table would give it, and that τ is at or before
  the end of the local period);
- each scored day's rows from the rows known at the end of the local period and that day's own
  rows alone, so its history holds only sessions that had ended by then (reading C2: the k local
  days as known at the end of the local period are all the model knows of the unit; not the days
  between them and the scored day, nor earlier scored days, nor the later plug-out of a session
  listed on a local day).

So nothing that happened after the end of the local period reaches a row's features at k, apart
from each scored day's own rows as known at its own τ in its own forecast: a row's own outcome is
no feature (`learn_features`); a history counts only sessions that had plugged out by the row's τ,
which for a local day is at or before the end of the local period; a scored day's table holds no
session that ended after it; and the extra input columns are added row by row from the row's own
site, date and fields (`model_inputs`). Rows are in the order of the days, then as
`forecast_frame` gives them.
"""

from dataclasses import dataclass

import pandas as pd

from fleetreserve import learn_run

LOCAL_DAYS = (0, 7, 14, 28, 56)  # plan section 11, O9
LARGEST = max(LOCAL_DAYS)


@dataclass(frozen=True)
class Local:
    """What `learn_run.forecast_frame` reads of a unit: a cars table and its rho."""

    cars: pd.DataFrame
    rho: float


def day_keys(unit_days) -> list[tuple[str, str]]:
    """(date, site) of each counting day of `unit_days`, in its order; raises `ValueError` when
    that order is not by date, then site, or a key repeats (module text)."""
    keys = [(t.record.date, t.record.site) for t in unit_days.days]
    if any(a >= b for a, b in zip(keys, keys[1:], strict=False)):
        raise ValueError(
            f"{unit_days.unit}: the counting days are not in time order (date, then site) or a "
            "day repeats"
        )
    return keys


def local_indices(n_days: int, k: int) -> list[int]:
    """The indices of the first k counting days; raises `ValueError` when k is not in 0..n_days."""
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= n_days:
        raise ValueError(f"k must be a whole number from 0 to {n_days}, got {k!r}")
    return list(range(k))


def scored_indices(n_days: int, largest: int = LARGEST) -> list[int]:
    """The indices of the counting days after the first `largest` (module text)."""
    if isinstance(largest, bool) or not isinstance(largest, int) or largest < 0:
        raise ValueError(f"largest must be a whole number of at least 0, got {largest!r}")
    return list(range(largest, n_days))


def _row_keys(cars: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(cars["date"].astype(str), cars["site"].astype(str), strict=True))


def local_rows(unit_days, k: int) -> pd.DataFrame:
    """The rows of the cars table at or before the k-th counting day (module text), in the
    table's order with its index; none at k = 0."""
    keys = day_keys(unit_days)
    local_indices(len(keys), k)
    cars = unit_days.cars
    if k == 0:
        return cars.iloc[0:0]
    last = keys[k - 1]
    return cars[[key <= last for key in _row_keys(cars)]]


def local_end(unit_days, k: int) -> pd.Timestamp | None:
    """The end of the local period at k (module text): the latest τ of the first k counting days,
    in UTC; None at k = 0."""
    keys = day_keys(unit_days)
    local_indices(len(keys), k)
    if k == 0:
        return None
    taus = [unit_days.days[d].record.tau for d in range(k)]
    return pd.to_datetime(pd.Series(taus), utc=True, format="ISO8601").max()


def known_rows(unit_days, k: int) -> pd.DataFrame:
    """The local rows whose session had plugged out by the end of the local period (module text),
    in the table's order with its index; none at k = 0."""
    rows = local_rows(unit_days, k)
    if k == 0:
        return rows
    plug_out = pd.to_datetime(rows["plug_out"], utc=True, format="ISO8601", errors="coerce")
    return rows[(plug_out <= local_end(unit_days, k)).to_numpy(bool)]


def _day_rows(cars: pd.DataFrame, key: tuple[str, str]) -> pd.DataFrame:
    return cars[[found == key for found in _row_keys(cars)]]


def _of_day(frame: pd.DataFrame, key: tuple[str, str]) -> pd.DataFrame:
    date, site = key
    keep = (frame["date"].astype(str) == date) & (frame["site"].astype(str) == site)
    return frame[keep.to_numpy(bool)]


def frame_at(unit_days, k: int, scored, extra=None) -> pd.DataFrame:
    """The frame `predict` is given for the unit at k (module text). `scored`: indices of the
    scored days, each after the first k; `extra`: the model's `model_inputs.ExtraInputs` or None."""
    keys = day_keys(unit_days)
    local = local_indices(len(keys), k)
    scored = list(scored)
    if any(not k <= d < len(keys) for d in scored) or len(set(scored)) != len(scored):
        raise ValueError(f"the scored days must be distinct days after the first {k}")
    unit = unit_days.unit
    parts = []
    if local:
        rows = local_rows(unit_days, k)
        frame = learn_run.forecast_frame(False, {unit: Local(rows, unit_days.rho)}, [unit], extra)
        parts += [_of_day(frame, keys[d]) for d in local]
    known = known_rows(unit_days, k)
    for d in scored:
        own = _day_rows(unit_days.cars, keys[d])
        table = pd.concat([known, own])
        frame = learn_run.forecast_frame(False, {unit: Local(table, unit_days.rho)}, [unit], extra)
        parts.append(_of_day(frame, keys[d]))
    if not parts:
        raise ValueError(f"{unit}: at k = {k} with no scored day there is nothing to forecast")
    return pd.concat(parts, ignore_index=True)
