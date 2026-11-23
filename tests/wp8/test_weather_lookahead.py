"""The look-ahead weather columns (`weather_lookahead`) on made-up hourly values, by hand, and the
fence around them in `hazard_inputs.with_inputs` and `model_inputs.ExtraInputs`."""

import math

import numpy as np
import pandas as pd
import pytest

from fleetreserve import hazard_inputs, model_inputs, weather_lookahead


def weather(start="2019-01-01", hours=48, shift=0.0):
    times = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    k = np.arange(hours, dtype=float)
    return pd.DataFrame(
        {"temperature": k + shift, "precipitation": np.ones(hours), "snowfall": (k == 12) * 2.0},
        index=pd.DatetimeIndex(times, name="time"),
    )


def rows(*taus, site="OSL_1"):
    return pd.DataFrame({"site": [site] * len(taus), "tau": list(taus)})


def test_lookahead_by_hand():
    # τ 10:30 at +01:00 is 09:30 UTC; h = 09:00, so the window is the values at 10:00 to 17:00
    # UTC, hours 10 to 17 of the file: mean temperature 13.5, precipitation 8 x 1, snow 2 (hour 12)
    got = weather_lookahead.lookahead_columns(
        rows("2019-01-01T10:30:00+01:00"), {"oslo": weather()}
    )
    assert got.iloc[0].tolist() == [13.5, 8.0, 2.0]


def test_lookahead_reads_no_value_at_or_before_h_and_none_after_the_window():
    files = weather()
    files.loc[files.index[:10], "temperature"] = 99.0  # hours 0 to 9, at or before h
    files.loc[files.index[18:], "temperature"] = -99.0  # hours 18 on, after the window
    got = weather_lookahead.lookahead_columns(rows("2019-01-01T10:30:00+01:00"), {"oslo": files})
    assert got.iloc[0, 0] == 13.5


def test_lookahead_past_the_file_is_missing_and_places_are_averaged():
    got = weather_lookahead.lookahead_columns(
        rows("2019-01-02T20:00:00+00:00"), {"oslo": weather()}
    )
    assert got.isna().all(axis=None)
    two = weather_lookahead.lookahead_columns(
        rows("2019-01-01T09:00:00+00:00", site="two"),
        {"a": weather(), "b": weather(shift=10.0)},
        site_places={"two": ("a", "b")},
    )
    # hours 10 to 17: 13.5 and 23.5, mean 18.5
    assert two.iloc[0, 0] == 18.5


def test_lookahead_columns_are_fenced():
    assert not set(weather_lookahead.COLUMNS) & set(hazard_inputs.NEW_COLUMNS)
    assert all(c.startswith("lookahead_") for c in weather_lookahead.COLUMNS)
    cars = pd.DataFrame(
        {
            "unit": ["u"],
            "site": ["OSL_1"],
            "date": ["2019-01-01"],
            "session_id": ["s"],
            "tau": ["2019-01-01T10:30:00+01:00"],
        }
    )
    table = cars[["unit", "site", "date", "session_id"]]
    with pytest.raises(ValueError):
        hazard_inputs.with_inputs(table, cars, weather_lookahead.COLUMNS)
    sources = hazard_inputs.ContextSources(weather={"oslo": weather()})
    names = ("regime_covid", *weather_lookahead.COLUMNS)
    with pytest.raises(model_inputs.LookAheadRefused):
        model_inputs.ExtraInputs(names, sources).add(table, cars)
    got = model_inputs.ExtraInputs(names, sources, look_ahead=True).add(table, cars)
    assert list(got.columns) == [*table.columns, *names]
    assert got["lookahead_temperature_mean_c"].iloc[0] == 13.5
    assert got["regime_covid"].iloc[0] == 0 and not math.isnan(got["regime_covid"].iloc[0])
