"""The inputs of the day-size quantile net (`size_net`; plan section 7, "Day-size quantile net:
pinball loss on fleet encoding and context"): per labelled day, the cars plugged in at its
decision time τ, each as the input vector of the departure model with the plan inputs.

`car_frame(cars, rho, sources)`: one row per row of the cars table `cars` (`UnitDays.cars`) whose
day counts, with the key columns (`forecast_model.KEY_COLUMNS`) and `COLUMNS`, built the one way
the departure model builds its frames: `learn_features.build_table`, the extra columns
`EXTRA_COLUMNS` appended by `model_inputs.ExtraInputs` from `sources` (no look-ahead column:
`ExtraInputs` refuses one unless told otherwise, and none is named), `learn_features.feature_frame`
(the labels dropped) and `learn_features.with_context`. Every column is a function of facts known
at τ (`learn_features`, `hazard_inputs`; `tests/learn/test_size_net_inputs.py` checks that cars
equal up to τ give equal rows).

`fleets(frame, rows)`: the sets of a car frame for the labelled days `rows` (a frame with `unit`,
`site` and `date`): set k holds the cars of row k's site-day, in the order of
`hazard_data.make_sets` (by session id), as positions in `frame`; a labelled day with no car in
`frame` raises `ValueError`.

`Encoder`: `model_inputs.InputsScaler` with `EXTRA_COLUMNS`, the departure model's own encoding
(cyclic date parts and arrival hour as a sine and a cosine, counts through `log1p`, every other
column standardised by the fit rows' mean and spread and clipped, a missing value 0 with a flag),
fitted on the fit rows only.

`load_context(root)`: the context files those columns read (`model_inputs.load_context`: the
weather and OxCGRT files under `data/raw/`, the committed school terms and the 9/80 parity),
raising `model_inputs.ContextError` or `nine_eighty_settings.PhaseError`, both `MatrixError`s
("could not run"); `record(context)` is what a results document keeps of them.

Readings (the plan's sentence leaves these open; each with the alternative not taken):

N1. The fleet of a day is every car plugged in at τ on the site-day: every row of the cars table
    with the day's unit, site and date. Not the day's fleet of the true sets (`Day.cars`) nor its
    steered cars, which leave out a car that departs within the decision hour (a fact after τ) and
    a car without a usable forecast. [Alternative: the steered cars of the prepared day.]
N2. The per-car inputs are those of `models.hazard_tuned_inputs`: `hazard_data.INPUT_COLUMNS` and
    its `EXTRA_INPUTS` (energy drawn under reading E3, regime, weather of the recent past,
    OxCGRT, holidays dated by enactment, JPL's 9/80 Fridays, school terms), read from that one
    declared list. Not read: the day's certificate (its value and widths) and the forecasts,
    which the boosters' context (`size_features`) reads: under the matrix's known energy bounds
    the forecast rate and room are the true ones, the user's highest average power over all
    sessions and the session energy, so they differ between cars equal up to τ; and
    `declared_departure_h`, which the departure model does not read. [Alternative: the booster's
    context columns `size_features.COLUMNS` beside these.]
N3. Site: through `site_type` and the site's calendars and weather only, no learned site vector:
    the what-if scores each training unit, a single site, with a net fitted on the other units, so
    the scored site is never seen in its own fit. [Alternative: the hazard model's site vector
    with its code for a site not seen.]
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve import learn_features, model_inputs
from fleetreserve.forecast_model import KEY_COLUMNS
from fleetreserve.hazard_data import INPUT_COLUMNS, SET_KEY, SetIndex, make_sets
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.models.hazard_tuned_inputs import EXTRA_INPUTS
from fleetreserve.sessions import TOP

EXTRA_COLUMNS = EXTRA_INPUTS
COLUMNS = (*INPUT_COLUMNS, *EXTRA_COLUMNS)
INPUT_CLIP = HazardSettings().input_clip


def car_frame(cars: pd.DataFrame, rho: float, sources) -> pd.DataFrame:
    """The car frame of `cars` (module text); `sources` is a `hazard_inputs.ContextSources`."""
    extra = model_inputs.ExtraInputs(EXTRA_COLUMNS, sources)
    table = extra.add(learn_features.build_table(cars, rho=rho), cars.reset_index(drop=True))
    frame = learn_features.feature_frame(table)
    frame = frame[frame["counting"]].reset_index(drop=True)
    frame = learn_features.with_context(frame)
    return frame[[*KEY_COLUMNS, *COLUMNS]]


def fleets(frame: pd.DataFrame, rows: pd.DataFrame) -> SetIndex:
    """The sets of `frame` for the labelled days `rows` (module text)."""
    found = make_sets(frame)
    place = {key: k for k, key in enumerate(found.keys)}
    keys = tuple(tuple(str(v) for v in key) for key in rows[list(SET_KEY)].to_numpy())
    absent = sorted({key for key in keys if key not in place})
    if absent:
        raise ValueError(f"labelled days with no car in the car frame: {absent[:5]}")
    members = [found.members(place[key]) for key in keys]
    sizes = np.array([len(m) for m in members], dtype=int)
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    order = np.concatenate(members).astype(int) if members else np.zeros(0, dtype=int)
    return SetIndex(order, starts, keys)


@dataclass(frozen=True)
class Encoder:
    """The departure model's encoding of a car's inputs, fitted on the fit rows (module text)."""

    scaler: model_inputs.InputsScaler

    @classmethod
    def fit(cls, frame: pd.DataFrame, clip: float = INPUT_CLIP) -> "Encoder":
        settings = HazardSettings(input_clip=clip)
        return cls(model_inputs.InputsScaler.fit(frame, settings, EXTRA_COLUMNS))

    @property
    def n_inputs(self) -> int:
        return self.scaler.n_inputs

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Float32 inputs, one row per row of `frame`."""
        return self.scaler.transform(frame)


def load_context(root: Path = TOP) -> model_inputs.Context:
    """The context files of the extra columns (module text)."""
    return model_inputs.load_context(root)


def record(context: model_inputs.Context) -> dict:
    """What a results document keeps of the inputs and the context files read."""
    return {
        "columns": list(COLUMNS),
        "nine_eighty_parity": context.sources.nine_eighty_parity,
        "files_sha256": dict(sorted(context.files.items())),
    }
