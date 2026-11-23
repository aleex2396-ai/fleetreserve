"""Made-up inputs for the tests of the day-size net (`size_net`, `size_net_inputs`).

`context()`: made-up context sources (`model_inputs.Context`) for every training site, with hourly
weather from 1 December 2019 for 120 days, an OxCGRT index per jurisdiction from 1 January 2020,
the committed school terms and the 9/80 parity 0; a stand-in for `size_net_inputs.load_context`,
which reads the files under `data/raw/`.

`planted(rng, units, days, rule)`: a made-up car frame in the layout `size_net_inputs.car_frame`
gives (the key columns and `size_net_inputs.COLUMNS`) and one row per day with its label. Each day
has 3 to 12 cars; every day-level column is one value per day, the per-car columns vary by car,
and the columns no rule reads are noise. `rule(n_cars, school_holiday, rng)` gives the day's
clipped label: `context_rule` makes it depend on the school-holiday column and on the number of
cars, `noise_rule` on nothing.

`drawn_net(n_inputs, n_levels)`: an unfitted network (`size_net.build_net`) whose output layer's
weights are drawn under a fixed seed instead of 0, so that its output reads its inputs (an
unfitted network of `build_net` is a constant).
"""

import math

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import torch
from truth_tables_fake import date_of

from fleetreserve import (
    context_places,
    hazard_inputs,
    model_inputs,
    school_inputs,
    size_gbm,
    size_net,
    size_net_inputs,
)
from fleetreserve.forecast_model import KEY_COLUMNS
from fleetreserve.hazard_data import COUNT_COLUMNS, CYCLIC_PERIODS, MISSING_COLUMNS
from fleetreserve.learn_features import SITE_TYPES
from fleetreserve.truth_build import TRAIN_UNITS


def weather() -> pd.DataFrame:
    hours = pd.date_range("2019-12-01", periods=24 * 120, freq="h", tz="UTC")
    k = np.arange(len(hours), dtype=float)
    return pd.DataFrame(
        {"temperature": k % 17, "precipitation": k % 3, "snowfall": (k % 5 == 0) * 1.0},
        index=pd.DatetimeIndex(hours, name="time"),
    )


def context() -> model_inputs.Context:
    places = {p for site in TRAIN_UNITS for p in context_places.SITE_PLACES[site]}
    days = pd.Index(pd.date_range("2020-01-01", periods=120).date, name="date")
    jurisdictions = {context_places.SITE_OXCGRT[site] for site in TRAIN_UNITS}
    sources = hazard_inputs.ContextSources(
        weather=dict.fromkeys(places, weather()),
        oxcgrt={j: pd.Series(np.arange(120.0) % 50, index=days) for j in jurisdictions},
        nine_eighty_parity=0,
        school=school_inputs.load_terms(),
    )
    return model_inputs.Context(sources, {"made/up.csv": "0" * 64})


def context_rule(n_cars: int, school_holiday: float, rng) -> float:
    base = 0.25 if school_holiday else (1.0 if n_cars >= 8 else 0.55)
    return float(np.clip(base + rng.uniform(-0.03, 0.03), 0.0, 1.0))


def noise_rule(n_cars: int, school_holiday: float, rng) -> float:
    return float(rng.uniform(0.2, 1.0))


# Day-level columns of the made-up frame: one value per day (the other columns vary by car).
DAY_LEVEL = (
    "weekday",
    "month",
    "day_of_year",
    "fleet_at_tau",
    "public_holiday",
    "site_type",
    "regime_nrel_controller",
    "regime_asr_paid",
    "regime_asr_controller",
    "regime_covid",
    "weather_temperature_c",
    "weather_temperature_mean_24h_c",
    "weather_precipitation_24h_mm",
    "weather_snowfall_24h_cm",
    "oxcgrt_stringency",
    "public_holiday_dated",
    "nine_eighty_friday",
    "school_holiday",
    "academic_break",
)


def _value(column: str, rng):
    if column in CYCLIC_PERIODS:
        first, period = CYCLIC_PERIODS[column]
        return int(rng.integers(first, first + period))
    if column in COUNT_COLUMNS:
        return int(rng.integers(0, 30))
    if column in MISSING_COLUMNS and rng.random() < 0.2:
        return math.nan
    return float(rng.normal())


def planted(rng, units=("caltech", "jpl", "OSL_S"), days=150, rule=context_rule):
    """(car frame, labelled rows) of the module text."""
    cars, rows = [], []
    for unit in units:
        for k in range(days):
            n = int(rng.integers(3, 13))
            holiday = float(rng.random() < 0.3)
            date = date_of(k)
            day = {c: _value(c, rng) for c in DAY_LEVEL}
            day.update(
                fleet_at_tau=n,
                public_holiday=False,
                site_type=SITE_TYPES[unit],
                school_holiday=holiday,
            )
            for j in range(n):
                car = {c: _value(c, rng) for c in size_net_inputs.COLUMNS}
                car.update(day)
                car.update(
                    unit=unit,
                    site=unit,
                    date=date,
                    session_id=f"{unit}-{k}-{j}",
                    arrived_earlier_day=bool(rng.random() < 0.1),
                    user_known=bool(rng.random() < 0.8),
                )
                cars.append(car)
            rows.append(
                {"unit": unit, "site": unit, "date": date, size_gbm.LABEL: rule(n, holiday, rng)}
            )
    frame = pd.DataFrame(cars)[[*KEY_COLUMNS, *size_net_inputs.COLUMNS]]
    return frame, pd.DataFrame(rows)


def drawn_net(n_inputs: int, n_levels: int, seed: int = 5) -> size_net.SizeNet:
    """An unfitted network with its output layer's weights drawn under `seed` (module text)."""
    net = size_net.build_net(n_inputs, np.zeros(n_levels), size_net.NetSettings())
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        last = net.head[2].weight
        last.copy_(torch.randn(last.shape, generator=generator))
    return net
