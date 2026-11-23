"""The day's own weather of the service hours: a LOOK-AHEAD, not known at the decision time τ
(the owner's decision of 15 Nov 2026 on the weather input, plan section 7).

The weather columns of the departure model are of the recent past (`weather_inputs`: values at
least 8 hours before τ), because the archive holds no forecast of the service hours issued
before τ for the data sets' spans. To measure what a forecast of the service hours could at
most add, a what-if gives a model the weather that was then observed in those hours, read from
the same files. That is an upper bound on a forecast's value, never a model input: a value of
an hour after τ is not known at τ. Every column here is named `lookahead_...`, none is in
`hazard_inputs.NEW_COLUMNS` (so `hazard_inputs.with_inputs` refuses them), and only a model
listed in `forecast_model.LOOK_AHEAD_MODELS` may take them, where `model_inputs` and `learn_run`
allow it (see there).

Window (`HOURS` = 8, `day.MAX_SERVICE_HOURS`, the longest service window of plan section 2): for
a row with decision time τ, with h the whole UTC hour at or before τ, the hourly values with valid
times h + 1 h to h + 8 h. Precipitation and snowfall at a valid time t are sums over the hour
ending at t, so the window covers h to h + 8 h. Reading L1: the same eight hours for every row.
[Alternative: the day's own number of service hours (`service_hours` of the truth's day table),
which the cars table does not carry.]

Columns (`COLUMNS`): `lookahead_temperature_mean_c` (the mean 2 m temperature, °C, over the
window), `lookahead_precipitation_mm` (the precipitation summed over it, mm),
`lookahead_snowfall_cm` (the snowfall summed over it, cm). NaN when any value of the window is
missing or outside the file; a site with several places takes the plain mean over its places,
NaN when any place's is (as `weather_inputs`).
"""

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from fleetreserve.context_places import SITE_PLACES
from fleetreserve.day import MAX_SERVICE_HOURS

HOURS = MAX_SERVICE_HOURS
COLUMNS = (
    "lookahead_temperature_mean_c",
    "lookahead_precipitation_mm",
    "lookahead_snowfall_cm",
)


def _window_columns(series: pd.DataFrame) -> pd.DataFrame:
    """Per valid hour e of a place: the three columns over the `HOURS` hours ending at e."""
    full = series.asfreq("h")
    rolling = full.rolling(HOURS, min_periods=HOURS)
    return pd.DataFrame(
        {
            COLUMNS[0]: rolling["temperature"].mean(),
            COLUMNS[1]: rolling["precipitation"].sum(),
            COLUMNS[2]: rolling["snowfall"].sum(),
        }
    )


def window_ends(tau: pd.Series) -> pd.DatetimeIndex:
    """The last valid hour of each row's window: the whole UTC hour at or before τ, plus `HOURS`."""
    moments = pd.to_datetime(tau.astype(str), utc=True, format="ISO8601")
    return pd.DatetimeIndex(moments.dt.floor("h") + pd.Timedelta(hours=HOURS))


def lookahead_columns(
    frame: pd.DataFrame,
    weather: Mapping[str, pd.DataFrame],
    site_places: Mapping[str, tuple[str, ...]] = SITE_PLACES,
) -> pd.DataFrame:
    """`COLUMNS` for the rows of `frame`, read from its `site` and `tau` only (module text). A
    site missing from `site_places`, or a place missing from `weather`, raises `KeyError`."""
    sites = frame["site"].astype(str).to_numpy()
    ends = window_ends(frame["tau"])
    out = np.full((len(frame), len(COLUMNS)), math.nan)
    windows: dict[str, pd.DataFrame] = {}
    for site in pd.unique(sites):
        rows = np.flatnonzero(sites == site)
        stack = []
        for place in site_places[site]:
            if place not in windows:
                windows[place] = _window_columns(weather[place])
            stack.append(windows[place].reindex(ends[rows]).to_numpy(float))
        out[rows] = np.mean(np.stack(stack), axis=0)
    return pd.DataFrame(out, columns=list(COLUMNS), index=frame.index)
