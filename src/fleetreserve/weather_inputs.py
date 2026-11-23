"""Weather inputs of the departure model, each known at the decision time τ (plan section 7).

Plan section 7 names "archived ECMWF IFS HRES forecasts (Open-Meteo, from 2017; national for
Korea)". The files are `data/raw/weather/<place>.csv`, one per place of `context_places.PLACES`,
downloaded by `tools/fetch_data.py --context` from Open-Meteo's forecast archive with the model
`ecmwf_ifs` (ECMWF IFS HRES 9 km, archived from 1 Jan 2017), times in UTC.

What that archive is (Open-Meteo's page on its historical forecast API, read 15 Nov 2026): "Each
run's first few hours are stitched into a continuous hourly timeseries", from runs "updated every
6 hours". So the value at a valid time t comes from a run started at or before t, at a lead of a
few hours. It is not a forecast of a later day: the archive keeps no longer leads for 2017 to
2023 (its single-run archive of IFS HRES starts in March 2024, its fixed-lead archive in January
2024), so no value of the service hours themselves, issued before τ, exists in it for the data
sets' spans. What is known at τ from it is the weather up to some hours before τ.

The rule (`DELAY_HOURS`): a row uses only values whose valid time is at least `DELAY_HOURS` = 8
hours before its τ. Why that is known at τ: the run behind a value at t started at or before t,
and ECMWF's dissemination schedule for HRES (its "Set I" product table, read 15 Nov 2026) delivers
the 00 UTC run's analysis at 05:40 and its forecast steps from 05:45 to 07:34 UTC, so every
product of a run is out within 7 hours 34 minutes of its start; a value at t <= τ - 8 h was
therefore published by τ - 26 min. The same holds for the 06, 12 and 18 UTC runs, whose schedule
is shifted by their start. Open-Meteo's own processing time is not counted: the bound is when
ECMWF's customers had the run. [Alternative: a longer delay, such as 12 h, for a margin over
the schedule; it moves every window 4 hours earlier.] No value of the day's service hours enters,
and no value of any time after τ - 8 h.

Columns (`COLUMNS`), for a row with decision time τ, with c the last whole UTC hour at or before
τ - `DELAY_HOURS` and W the `WINDOW_HOURS` = 24 hourly values with valid times c - 23 h to c:

- `weather_temperature_c`: the 2 m temperature (°C) at c;
- `weather_temperature_mean_24h_c`: its mean over W;
- `weather_precipitation_24h_mm`: the precipitation (mm) summed over W (each hourly value is the
  sum over the hour ending at its valid time, so W covers c - 24 h to c);
- `weather_snowfall_24h_cm`: the snowfall (cm) summed over W.

A column is NaN when any value it needs is missing or outside the file. A site with several
places (the Korean portfolios, `context_places` reading P5) takes, per column, the plain mean over
its places' values, NaN when any place's is. The same columns are built for every unit, the
held-out units included, from these public files and the row's site and τ alone.

`read_weather(data)` reads the bytes of one file and refuses (`LayoutError`) any layout but the
one fetched: a first line `latitude,longitude,elevation,utc_offset_seconds,timezone,
timezone_abbreviation`, its values with a UTC offset of 0 and time zone GMT, a blank line, the
header `HEADER`, then one row per hour, consecutive and in order, a value written as a number or
`NaN`. `load_weather(root, places)` reads the files of the named places under `root`.
"""

import io
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve.context_places import SITE_PLACES, WEATHER_SET, weather_file

DELAY_HOURS = 8
WINDOW_HOURS = 24
META_HEADER = "latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation"
HEADER = ("time", "temperature_2m (°C)", "precipitation (mm)", "snowfall (cm)")
VALUES = ("temperature", "precipitation", "snowfall")
COLUMNS = (
    "weather_temperature_c",
    "weather_temperature_mean_24h_c",
    "weather_precipitation_24h_mm",
    "weather_snowfall_24h_cm",
)


class LayoutError(ValueError):
    pass


def read_weather(data: bytes) -> pd.DataFrame:
    """One weather file's hourly values, indexed by their UTC valid time (module text)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise LayoutError("a weather file is not UTF-8 text") from None
    lines = text.split("\n")
    if len(lines) < 5 or lines[0] != META_HEADER or lines[2] != "":
        raise LayoutError("a weather file does not start with its place line and a blank line")
    meta = lines[1].split(",")
    if len(meta) != 6 or meta[3] != "0" or meta[4] != "GMT":
        raise LayoutError("a weather file's times are not in UTC")
    if tuple(lines[3].split(",")) != HEADER:
        raise LayoutError("a weather file's header is not the one fetched")
    body = "\n".join(lines[4:])
    table = pd.read_csv(
        io.StringIO(body),
        header=None,
        names=["time", *VALUES],
        dtype={"time": str},
        na_values=["NaN"],
        keep_default_na=False,
    )
    if table.empty:
        raise LayoutError("a weather file has no rows")
    try:
        times = pd.to_datetime(table["time"], format="%Y-%m-%dT%H:%M", utc=True)
        values = table[list(VALUES)].astype(float)
    except (ValueError, TypeError):
        raise LayoutError("a weather file has an unreadable time or value") from None
    steps = times.diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise LayoutError("a weather file's hours are not consecutive and in order")
    return values.set_axis(pd.DatetimeIndex(times, name="time"))


def load_weather(root: Path, places: Iterable[str]) -> dict[str, pd.DataFrame]:
    """The weather of each named place, read from `<root>/data/raw/weather/<place>.csv`."""
    folder = Path(root) / "data" / "raw" / WEATHER_SET
    return {place: read_weather((folder / weather_file(place)).read_bytes()) for place in places}


def _window_columns(series: pd.DataFrame) -> pd.DataFrame:
    """Per valid hour c of a place: the four columns over the 24 hours ending at c."""
    full = series.asfreq("h")
    rolling = full.rolling(WINDOW_HOURS, min_periods=WINDOW_HOURS)
    return pd.DataFrame(
        {
            COLUMNS[0]: full["temperature"],
            COLUMNS[1]: rolling["temperature"].mean(),
            COLUMNS[2]: rolling["precipitation"].sum(),
            COLUMNS[3]: rolling["snowfall"].sum(),
        }
    )


def cutoff_hours(tau: pd.Series, delay_hours: float = DELAY_HOURS) -> pd.DatetimeIndex:
    """c of each τ: the last whole UTC hour at or before τ - `delay_hours`."""
    moments = pd.to_datetime(tau.astype(str), utc=True, format="ISO8601")
    return pd.DatetimeIndex((moments - pd.Timedelta(hours=delay_hours)).dt.floor("h"))


def weather_columns(
    frame: pd.DataFrame,
    weather: Mapping[str, pd.DataFrame],
    site_places: Mapping[str, tuple[str, ...]] = SITE_PLACES,
    delay_hours: float = DELAY_HOURS,
) -> pd.DataFrame:
    """`COLUMNS` for the rows of `frame`, read from its `site` and `tau` only (module text).
    A site missing from `site_places`, or a place missing from `weather`, raises `KeyError`."""
    if delay_hours < DELAY_HOURS:
        raise ValueError(f"the delay must be at least {DELAY_HOURS} hours (module text)")
    sites = frame["site"].astype(str).to_numpy()
    cutoff = cutoff_hours(frame["tau"], delay_hours)
    out = np.full((len(frame), len(COLUMNS)), math.nan)
    windows: dict[str, pd.DataFrame] = {}
    for site in pd.unique(sites):
        places = site_places[site]
        rows = np.flatnonzero(sites == site)
        stack = []
        for place in places:
            if place not in windows:
                windows[place] = _window_columns(weather[place])
            stack.append(windows[place].reindex(cutoff[rows]).to_numpy(float))
        # plain mean over the places; NaN when any place's value is NaN
        out[rows] = np.mean(np.stack(stack), axis=0)
    return pd.DataFrame(out, columns=list(COLUMNS), index=frame.index)
