"""Tables to tensors for the neural hazard model (`models.hazard`; plan sections 7, 13 WP8).

Inputs, per car: the columns `learn_features.FEATURE_COLUMNS`, `TAIL_HISTORY_COLUMNS` and
`CONTEXT_COLUMNS`, nothing else (`INPUT_COLUMNS`). The encoding (`Scaler`) reads those columns of
a frame only. Besides them the module reads the key columns (`make_sets`, to build the sets)
and the label columns (`fit_data`, to build the targets); neither is an input of the encoding.
`declared_departure_h` and every other column are never read by this module. The encoding is
fitted on the fit rows only:

- the four cyclic columns (`CYCLIC_PERIODS`: weekday with period 7, month 12, day of year 366,
  arrival hour 24) enter as a sine and a cosine of their phase;
- the three count columns (`COUNT_COLUMNS`) are put through `log1p` (meant for counts with a long
  tail: an assumption about the data) and then treated like the other plain columns;
- every plain column (all input columns except the cyclic ones; true/false as 1/0) is
  standardised with the mean and standard deviation of the fit rows' finite values (a column
  whose standard deviation is at most 1e-9 keeps scale 1), then clipped to plus or minus the
  settings' `input_clip`;
- a value that can be missing (`MISSING_COLUMNS`: the history columns other than the two counts)
  becomes 0 after standardising, and a 0/1 "missing" input per such column says so. A missing
  value in any other column is an error, and so is an infinite value;
- the energy targets (`ENERGY_LABELS`) are not inputs: `fit_data` standardises each with the mean
  and standard deviation of the rows where all three are numbers (the same floor of 1e-9).

Named in the plan's inputs (section 7) and not among the inputs: energy drawn before τ, regime,
event rows, school terms, the calendars other than the public-holiday flag (laws dated by
enactment, one-off holidays by announcement, OxCGRT, the JPL 9/80 Fridays), archived weather
forecasts. Adding one here is a change to `learn_features` and to `INPUT_COLUMNS`; a model that
takes them as extra columns beside these (`model_inputs`, `models.hazard_tuned_inputs`) leaves
this module unchanged.

Restricting the inputs (`HazardSettings.inputs`, default None meaning every column): `input_mask`
gives a 0/1 vector over the `N_INPUTS` outputs of `Scaler.transform`, 1 for a named column (a
plain column and its missing-value flag, or both phases of a cyclic column). `Scaler.transform`
multiplies its result by the vector as its last step, so the network keeps its shape and an
excluded input is a constant 0. With None no vector is made and `transform` returns its float32
result as it is. A test compares the forecast tables of None and of every input column named
(a vector of ones); another compares the plain columns of `transform` with their formula from
the mean and the spread.

Sets: a set is the cars of one site-day of a unit, the rows with the same `unit`, `site` and
`date` (`make_sets`). Fitting and prediction build it with this one function, so a site-day
whose rows are all in the frame is the same set in both. Within a set the cars are ordered by
`session_id` as text, ties by their place in the frame, so the encoder sees the same sequence for
any order of the rows when the session ids of a set are different, as they are in a valid cars
table.

Batches (`batch_of`): the sets of a batch padded to the longest one, with a mask that is true for
a car. The posterior of the day factor reads the departure labels of the batch (`label_feature`:
`log1p` of the departure over `log1p` of the grid's end, 0 where there is none).
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fleetreserve.hazard_curves import bin_of
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.learn_features import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    HISTORY_COLUMNS,
    TAIL_HISTORY_COLUMNS,
)

INPUT_COLUMNS = (*FEATURE_COLUMNS, *TAIL_HISTORY_COLUMNS, *CONTEXT_COLUMNS)
CYCLIC_PERIODS = {
    "weekday": (0, 7),
    "month": (1, 12),
    "day_of_year": (1, 366),
    "arrival_hour": (0, 24),
}
COUNT_COLUMNS = ("history_sessions", "history_same_weekday_sessions", "fleet_at_tau")
PLAIN_COLUMNS = tuple(c for c in INPUT_COLUMNS if c not in CYCLIC_PERIODS)
MISSING_COLUMNS = tuple(
    c
    for c in (*HISTORY_COLUMNS, *TAIL_HISTORY_COLUMNS)
    if c not in ("history_sessions", "history_same_weekday_sessions")
)
N_INPUTS = len(PLAIN_COLUMNS) + len(MISSING_COLUMNS) + 2 * len(CYCLIC_PERIODS)
SET_KEY = ("unit", "site", "date")
ENERGY_LABELS = ("label_rate_kw", "label_room_kwh", "label_need_kwh")
UNKNOWN_SITE = 0


def _plain_values(frame: pd.DataFrame) -> np.ndarray:
    """The plain columns as floats, counts through `log1p`; missing values stay NaN."""
    values = frame[list(PLAIN_COLUMNS)].astype(float).to_numpy(copy=True)
    for column in COUNT_COLUMNS:
        place = PLAIN_COLUMNS.index(column)
        values[:, place] = np.log1p(values[:, place])
    return values


def input_mask(names) -> np.ndarray:
    """The 0/1 float32 vector over the `N_INPUTS` outputs of `Scaler.transform` that is 1 for the
    input columns `names` (module text); raises `ValueError` for a name that is not an input."""
    unknown = sorted(set(names) - set(INPUT_COLUMNS))
    if unknown:
        raise ValueError(f"not input columns: {unknown}")
    keep = (
        [c in names for c in PLAIN_COLUMNS]
        + [c in names for c in MISSING_COLUMNS]
        + [c in names for c in CYCLIC_PERIODS for _ in range(2)]
    )
    return np.array(keep, dtype=np.float32)


@dataclass(frozen=True)
class Scaler:
    """Mean and standard deviation of the plain columns over the fit rows (module text);
    `keep`, when not None, is the `input_mask` that `transform` applies."""

    mean: np.ndarray
    std: np.ndarray
    clip: float
    keep: np.ndarray | None = None

    @classmethod
    def fit(cls, frame: pd.DataFrame, settings: HazardSettings) -> "Scaler":
        values = _plain_values(frame)
        mean = np.zeros(values.shape[1])
        std = np.ones(values.shape[1])
        for place in range(values.shape[1]):
            seen = values[:, place][np.isfinite(values[:, place])]
            if seen.size:
                mean[place] = seen.mean()
                spread = seen.std()
                std[place] = spread if spread > 1e-9 else 1.0
        keep = None if settings.inputs is None else input_mask(settings.inputs)
        return cls(mean, std, settings.input_clip, keep)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """The inputs of every row of `frame` (module text), float32, shape (rows, N_INPUTS)."""
        missing_ok = [PLAIN_COLUMNS.index(c) for c in MISSING_COLUMNS]
        values = _plain_values(frame)
        if np.isinf(values).any():
            raise ValueError("an input value is infinite")
        absent = np.isnan(values)
        allowed = np.zeros(values.shape[1], dtype=bool)
        allowed[missing_ok] = True
        bad = absent.any(axis=0) & ~allowed
        if bad.any():
            names = [PLAIN_COLUMNS[k] for k in np.flatnonzero(bad)]
            raise ValueError(f"an input value is missing in the columns {names}")
        scaled = np.clip((values - self.mean) / self.std, -self.clip, self.clip)
        scaled = np.where(absent, 0.0, scaled)
        flags = absent[:, missing_ok].astype(float)
        cyclic = []
        for column, (first, period) in CYCLIC_PERIODS.items():
            phase = 2.0 * math.pi * (frame[column].astype(float).to_numpy() - first) / period
            cyclic += [np.sin(phase), np.cos(phase)]
        out = np.column_stack([scaled, flags, *cyclic])
        if not np.isfinite(out).all():
            raise ValueError("an input value is not a finite number")
        out = out.astype(np.float32)
        return out if self.keep is None else out * self.keep


@dataclass(frozen=True)
class SetIndex:
    """The sets of a frame: `rows` are positions in the frame, grouped by set and ordered within
    a set; set `k` is `rows[starts[k]:starts[k + 1]]`; `keys[k]` is its (unit, site, date)."""

    rows: np.ndarray
    starts: np.ndarray
    keys: tuple[tuple[str, str, str], ...]

    @property
    def n_sets(self) -> int:
        return len(self.keys)

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.starts)

    def members(self, k: int) -> np.ndarray:
        return self.rows[self.starts[k] : self.starts[k + 1]]


def make_sets(frame: pd.DataFrame) -> SetIndex:
    """The sets of `frame` (module text). Needs `unit`, `site`, `date` and `session_id`."""
    if len(frame) == 0:
        return SetIndex(np.zeros(0, dtype=int), np.zeros(1, dtype=int), ())
    key = pd.DataFrame({c: frame[c].astype(str).to_numpy() for c in (*SET_KEY, "session_id")})
    order = key.sort_values([*SET_KEY, "session_id"], kind="stable").index.to_numpy()
    ordered = key.iloc[order]
    group = ordered[list(SET_KEY)]
    new = (group != group.shift()).any(axis=1).to_numpy()
    starts = np.concatenate([np.flatnonzero(new), [len(order)]])
    keys = tuple(tuple(group.iloc[s]) for s in starts[:-1])
    return SetIndex(order.astype(int), starts.astype(int), keys)


def site_codes(sets: SetIndex, fit_sites: tuple[str, ...]) -> np.ndarray:
    """Per set, its site's code: 1 + the place of the site among `fit_sites`, or `UNKNOWN_SITE`
    for a site not seen in fitting, whatever its name."""
    place = {site: k + 1 for k, site in enumerate(fit_sites)}
    return np.array([place.get(key[1], UNKNOWN_SITE) for key in sets.keys], dtype=int)


def label_feature(departure: np.ndarray, settings: HazardSettings) -> np.ndarray:
    """The departure as the posterior of the day factor reads it (module text)."""
    d = np.asarray(departure, dtype=float)
    scale = math.log1p(settings.grid_end_h)
    return np.where(np.isfinite(d), np.log1p(np.clip(np.nan_to_num(d), 0.0, None)) / scale, 0.0)


@dataclass(frozen=True)
class FitData:
    """The fit rows as arrays, in frame order: inputs, departure label and bin, energy targets
    (rate, room and protected energy, standardised), the rows each is fitted on."""

    x: np.ndarray
    sets: SetIndex
    site: np.ndarray
    departure: np.ndarray
    has_departure: np.ndarray
    departure_bin: np.ndarray
    energy: np.ndarray
    has_energy: np.ndarray
    energy_mean: np.ndarray
    energy_std: np.ndarray
    label: np.ndarray


def fit_data(
    table: pd.DataFrame, scaler: Scaler, fit_sites: tuple[str, ...], settings: HazardSettings
) -> FitData:
    """`FitData` from a `learn_features.build_table` table with its context columns."""
    departure = table["label_departure_h"].to_numpy(float)
    has_departure = np.isfinite(departure)
    energy = table[list(ENERGY_LABELS)].to_numpy(float)
    has_energy = np.isfinite(energy).all(axis=1)
    if not has_departure.any():
        raise ValueError("no training row has a departure label")
    if not has_energy.any():
        raise ValueError("no training row has an energy label")
    mean = energy[has_energy].mean(axis=0)
    std = energy[has_energy].std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    sets = make_sets(table)
    return FitData(
        x=scaler.transform(table),
        sets=sets,
        site=site_codes(sets, fit_sites),
        departure=departure,
        has_departure=has_departure,
        departure_bin=bin_of(np.where(has_departure, departure, 0.0), settings),
        energy=np.where(has_energy[:, None], (energy - mean) / std, 0.0).astype(np.float32),
        has_energy=has_energy,
        energy_mean=mean,
        energy_std=std,
        label=label_feature(departure, settings).astype(np.float32),
    )


@dataclass(frozen=True)
class Batch:
    """Padded arrays of some sets: inputs (sets, cars, N_INPUTS), `mask` true for a car, the
    site code per set and, when fitting, the labels (module text)."""

    x: np.ndarray
    mask: np.ndarray
    site: np.ndarray
    rows: np.ndarray
    departure_bin: np.ndarray | None = None
    has_departure: np.ndarray | None = None
    label: np.ndarray | None = None
    energy: np.ndarray | None = None
    has_energy: np.ndarray | None = None


def batch_of(x: np.ndarray, sets: SetIndex, site: np.ndarray, chosen, data: FitData | None = None):
    """The `Batch` of the sets numbered `chosen`; `rows` holds each car's frame position (0 for
    padding). With `data` (the fit rows) the labels are included."""
    chosen = np.asarray(chosen, dtype=int)
    sizes = sets.sizes[chosen]
    longest = int(sizes.max())
    slot = np.arange(longest)[None, :]
    mask = slot < sizes[:, None]
    place = np.where(mask, sets.starts[chosen][:, None] + slot, 0)
    rows = np.where(mask, sets.rows[place], 0)
    base = {"x": x[rows] * mask[..., None], "mask": mask, "site": site[chosen], "rows": rows}
    if data is None:
        return Batch(**base)
    return Batch(
        **base,
        departure_bin=data.departure_bin[rows],
        has_departure=data.has_departure[rows] & mask,
        label=data.label[rows] * mask,
        energy=data.energy[rows],
        has_energy=data.has_energy[rows] & mask,
    )
