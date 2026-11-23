"""Made-up cars for the tests of the neural hazard model (`models.hazard`).

`known_cars(n_days, seed, first_day)`: `PER_DAY` cars a day at the two sites `SITES` (one site a
day), the even cars in group A, the odd ones in group B, each with its own user. A car of group A
plugs in at 7:30, one of group B at 9:30 (τ is 10:00), so the arrival hour tells the group. Given
the group, a car leaves in a bin of the default time grid drawn from the known mass function
`GROUPS` (two uniform windows: an early and a late one, with a gap between), at a uniform time in
the bin. Group A: 0.42 early on [1, 4) h, 0.58 late on [5, 8) h. Group B: 0.25 early on
[0.5, 2.5) h, 0.75 late on [6, 9) h (hours from τ). No level of the tests (0.05 to 0.3) is at the
edge of a window's share, where a small error of a fitted curve would move a bound across the gap.
Rate and room are independent uniform draws (rate 3 to 5 kW, room 1 to 4 kWh).

`true_curve(group)`: the group's survival curve on the grid, `S_j = P(D > t_j)` with the mass of a
bin at its start; `true_bound(group, kappa, hour)`: its lower kappa-quantile, given presence at the
start of service hour `hour` when that is given, by `forecast_model.survival_quantile`.

`day_type_cars(n_days, seed)`: eight cars a day at the training site `caltech`, each with its own
user. The day is of type E or L by a fair draw. E: six cars plug in at 6:30, two at 9:30, and every
car leaves 3 to 4 h after τ. L: two plug in at 6:30, six at 9:30, and every car leaves 7 to 8 h
after τ. So a car that plugged in at 9:30 has the same inputs on both kinds of day, and only the
other cars of its day tell its departure.

`fresh_frame(cars, first_day)`: the feature frame of the days from `first_day`.
`FAST`: small network settings for the tests. `short_stay_model()`: a model fitted on short stays
only, at the workplace `caltech`.

`typed_support_cars()`, `typed_support_model()`: 150 days of ten cars at `caltech` (a workplace)
that all leave 0.5 to 3.5 h after τ, and 150 days of ten cars at `OSL_S` (the other site type) that
leave 0.5 to 12 h after τ; the model fitted on them, with the per-car presence floor off. So only
the other type has fit rows at the start of service hours 4 to 8.

`outside_fleet_cars()`, `outside_fleet_model()`: the 150 short-stay days at `caltech`, plus 30
later days of ten cars that leave 6 to 10 h after τ and are all outside their fleet (their room is
a missing value, as `days_from_truth.apply_car_rules` leaves it for a car outside a fleet); the
model fitted on them, with the per-car presence floor off. So only rows outside a fleet have a
departure at or above hour 4.

`use_small_tuned(monkeypatch, folder)`: a made-up grid and tuned settings file at `SMALL_TUNED`
(width 16, 4 epochs), put where `hazard_tune_settings.load_tuned` reads them by default.
"""

import dataclasses
import functools
import hashlib
import json

import numpy as np
import pandas as pd
import survival_fake as sf
from truth_tables_fake import FakeCar, date_of

from fleetreserve.forecast_model import survival_quantile
from fleetreserve.hazard_curves import PRESENCE_SHIFT, bin_starts
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.hazard_tune_grid import Setting
from fleetreserve.learn_features import build_table, feature_frame

SITES = ("caltech", "OSL_S")
PER_DAY = 10
KAPPAS = (0.05, 0.1, 0.2, 0.3)
FAST = HazardSettings(width=16, heads=2, blocks=1, head_hidden=32, epochs=40, batch_sets=32)
GROUPS = {
    "A": {"arrival": 7.5, "windows": ((0.42, 1.0, 4.0), (0.58, 5.0, 8.0))},
    "B": {"arrival": 9.5, "windows": ((0.25, 0.5, 2.5), (0.75, 6.0, 9.0))},
}
STARTS = bin_starts(HazardSettings())
WIDTHS = np.diff(STARTS)


def mass(group: str) -> np.ndarray:
    """The group's probability of each bin (the last entry is the open bin)."""
    starts = STARTS
    out = np.zeros(len(starts))
    for share, low, high in GROUPS[group]["windows"]:
        inside = (starts >= low) & (starts < high)
        out[inside] += share / inside.sum()
    return out


def true_curve(group: str) -> np.ndarray:
    return np.clip(1.0 - np.cumsum(mass(group)), 0.0, 1.0)


def true_bound(group: str, kappa: float, hour: int | None = None) -> float:
    after = None if hour is None else hour - PRESENCE_SHIFT
    return survival_quantile(STARTS, true_curve(group), kappa, after=after)


def known_cars(n_days: int, seed: int, first_day: int = 0):
    """(fakes, users) of the known-hazard cars (module text)."""
    rng = np.random.default_rng(seed)
    fakes, users = [], []
    for k in range(first_day, first_day + n_days):
        site = SITES[k % 2]
        for j in range(PER_DAY):
            group = "AB"[j % 2]
            bin_ = rng.choice(len(STARTS), p=mass(group))
            stay = STARTS[bin_] + WIDTHS[min(bin_, len(WIDTHS) - 1)] * rng.random()
            fakes.append(
                FakeCar(
                    site,
                    date_of(k),
                    GROUPS[group]["arrival"],
                    10.0 + stay,
                    rate=3.0 + 2.0 * rng.random(),
                    room=1.0 + 3.0 * rng.random(),
                    session=f"{site}-{k}-{j}",
                    energy=5.0,
                )
            )
            users.append(f"user-{k}-{j}")
    return fakes, users


def day_type_cars(n_days: int, seed: int):
    rng = np.random.default_rng(seed)
    fakes, users = [], []
    for k in range(n_days):
        late_day = rng.random() < 0.5
        for j in range(8):
            early_arrival = j < (2 if late_day else 6)
            low = 7.0 if late_day else 3.0
            fakes.append(
                FakeCar(
                    "caltech",
                    date_of(k),
                    6.5 if early_arrival else 9.5,
                    10.0 + low + rng.random(),
                    rate=3.5,
                    room=2.0,
                    session=f"d-{k}-{j}",
                    energy=4.0,
                )
            )
            users.append(f"user-{k}-{j}")
    return fakes, users


def fresh_frame(cars: pd.DataFrame, first_day: int) -> pd.DataFrame:
    table = build_table(cars, rho=0.0)
    return feature_frame(table[table["date"] >= date_of(first_day)].reset_index(drop=True))


def train_cars(cars: pd.DataFrame, first_fresh_day: int) -> pd.DataFrame:
    return cars[cars["date"] < date_of(first_fresh_day)].reset_index(drop=True)


@functools.cache
def known_data(days: int = 300, fresh: int = 40):
    """(training cars table, feature frame of the later days, their build_table table)."""
    fakes, users = known_cars(days + fresh, seed=5)
    cars = sf.cars_table(fakes, users)
    table = build_table(cars, rho=0.0)
    later = table[table["date"] >= date_of(days)].reset_index(drop=True)
    return train_cars(cars, days), feature_frame(later), later


@functools.cache
def fitted_known():
    """(model fitted on the known-hazard training days with `FAST`, feature frame of the later
    days, their `build_table` table, the model's forecast table at `KAPPAS`). The tests that use
    it only read."""
    from fleetreserve.models.hazard import HazardModel

    train, fresh, later = known_data()
    model = HazardModel(rho=0.0, kappas=KAPPAS, settings=FAST)
    model.fit(train)
    return model, fresh, later, model.predict(fresh, KAPPAS)


@functools.cache
def short_stay_model():
    """A model fitted on 150 days of ten cars that all leave 0.5 to 3.5 h after τ (so no fit car
    is plugged in at the start of service hour 4 or later), with the per-car presence floor off
    so that only the rule on the fit rows acts. The tests that use it only read."""
    from fleetreserve.models.hazard import HazardModel

    fakes, users = sf.stay_cars(150, 0, ((1.0, 0.5, 3.5),), site="caltech")
    model = HazardModel(
        rho=0.0,
        kappas=KAPPAS,
        settings=dataclasses.replace(FAST, min_presence_probability=0.0),
    )
    model.fit(sf.cars_table(fakes, users))
    return model


@functools.cache
def typed_support_cars() -> pd.DataFrame:
    short, short_users = sf.stay_cars(150, 0, ((1.0, 0.5, 3.5),), site="caltech")
    long, long_users = sf.stay_cars(150, 0, ((1.0, 0.5, 12.0),), seed=12, site="OSL_S")
    return sf.cars_table(short + long, short_users + long_users)


@functools.cache
def typed_support_model():
    from fleetreserve.models.hazard import HazardModel

    model = HazardModel(
        rho=0.0,
        kappas=KAPPAS,
        settings=dataclasses.replace(FAST, min_presence_probability=0.0),
    )
    model.fit(typed_support_cars())
    return model


@functools.cache
def outside_fleet_cars() -> pd.DataFrame:
    short, short_users = sf.stay_cars(150, 0, ((1.0, 0.5, 3.5),), site="caltech")
    long, long_users = sf.stay_cars(30, 200, ((1.0, 6.0, 10.0),), site="caltech")
    cars = sf.cars_table(short + long, short_users + long_users)
    outside = cars["date"] >= date_of(200)
    cars.loc[outside, "room"] = np.nan
    cars.loc[outside, "in_fleet"] = False
    return cars


@functools.cache
def outside_fleet_model():
    from fleetreserve.models.hazard import HazardModel

    model = HazardModel(
        rho=0.0,
        kappas=KAPPAS,
        settings=dataclasses.replace(FAST, min_presence_probability=0.0),
    )
    model.fit(outside_fleet_cars())
    return model


# --- a made-up tuned settings file (for the models built from `learn/hazard-tuned.json`) ---------

SMALL_TUNED = Setting(16, 0.003, 1.0, 4)
SMALL_GRID = {
    "about": "made up",
    "seeds": [0],
    "epochs": [4, 30],
    "width": [16, 32],
    "learning_rate": [0.003],
    "kl_weight": [1.0],
}


def use_small_tuned(monkeypatch, folder, setting: Setting = SMALL_TUNED) -> None:
    """Point `hazard_tune_grid.GRID` at a made-up grid in `folder` that holds `setting` and the
    present settings, and `hazard_tune_settings.TUNED` at a settings file in `folder` that names
    `setting` with that grid's hash."""
    from fleetreserve import hazard_tune_grid, hazard_tune_settings

    grid = folder / "grid.json"
    grid.write_text(json.dumps(SMALL_GRID))
    monkeypatch.setattr(hazard_tune_grid, "GRID", grid)
    digest = hashlib.sha256(grid.read_bytes()).hexdigest()
    tuned = folder / "hazard-tuned.json"
    document = {
        "about": "made up",
        "settings": setting.as_dict(),
        "score": 1.0,
        "grid": {"file": grid.name, "sha256": digest},
        "results": {"file": "x", "sha256": "1" * 64, "run_folder": "r", "commit": "c" * 40},
    }
    tuned.write_text(json.dumps(document))
    monkeypatch.setattr(hazard_tune_settings, "TUNED", tuned)
