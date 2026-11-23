"""Made-up units, a made-up verified truth build and stand-in models for the cold-start tests.

`cold_cars(n_days, site, ...)`: `gate_fake.gate_cars` (three cars a day, τ at 10:00, one service
hour; the second car leaves 1.3 hours after τ on every twelfth day), with `skip` the day offsets
left without cars and `thin` the offsets with two cars only (a day that does not count, since a
day counts from three cars).

`write_tables(folder, unit, cars)`: the tables of `truth_tables_fake.tables`, but every car of a
site keeps one user id across days (`<site>-user<j>`, j its place in the day), so the user
histories of `learn_features` are not empty. `write_build(folder, units)`: those tables for each
unit and a recorded truth build listing their SHA-256, as `gate_fake.write_build` writes it.

`HistoryModel`: a stand-in departure model reading one user-history input. Without history it
forecasts every car to leave 0.5 hours after τ (inside the service hour, so the day offers little);
with history, the smallest earlier departure (`history_min_departure_h`) less 0.3 hours. So what
reaches the history moves the forecasts, and a planted early departure in a day the model must not
see would show. It records every frame given to `fit` and `predict`. `register(monkeypatch, name,
...)` makes it the module `fleetreserve.models.<name>`.
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from gate_fake import COMMIT, SERVICE_HOURS, gate_cars
from truth_tables_fake import tables

from fleetreserve.forecast_model import KEY_COLUMNS, SURVIVAL_COLUMNS, TABLE_COLUMNS

FIT_UNIT = "caltech"


def cold_cars(n_days: int, site: str = "A", skip=(), thin=(), early_every: int = 12) -> list:
    cars = []
    for car in gate_cars(n_days, site, early_every=early_every):
        _, k, j = car.session.rsplit("-", 2)
        if int(k) in skip or (int(k) in thin and int(j) == 2):
            continue
        cars.append(car)
    return cars


def write_tables(folder: Path, unit: str, cars: list) -> None:
    car_table, day_table = tables(cars, unit)
    car_table["user_id"] = [
        f"{site}-user{session.rsplit('-', 1)[1]}"
        for site, session in zip(car_table["site"], car_table["session_id"], strict=True)
    ]
    day_table["service_hours"] = str(SERVICE_HOURS)
    car_table.to_csv(folder / f"{unit}-cars.csv", index=False, lineterminator="\n")
    day_table.to_csv(folder / f"{unit}-days.csv", index=False, lineterminator="\n")


def write_build(folder: Path, units: dict) -> tuple[Path, Path]:
    """Tables for `units` (name -> cars) in `folder/tables`, and a recorded truth build."""
    tables_folder = folder / "tables"
    tables_folder.mkdir()
    entries = {}
    for unit, cars in units.items():
        write_tables(tables_folder, unit, cars)
        entries[unit] = {
            "sha256": {
                kind: hashlib.sha256(
                    (tables_folder / f"{unit}-{kind}.csv").read_bytes()
                ).hexdigest()
                for kind in ("cars", "days")
            }
        }
    run = folder / "20260101T000000Z-wp2-truth"
    run.mkdir()
    summary = run / "truth-summary.json"
    summary.write_text(json.dumps({"units": entries}))
    manifest = {
        "commit": COMMIT,
        "command": ["build"],
        "start": "s",
        "end": "e",
        "exit_code": 0,
        "outputs": {"truth-summary.json": hashlib.sha256(summary.read_bytes()).hexdigest()},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run, tables_folder


class HistoryModel:
    """The stand-in model of the module text."""

    def __init__(self, name: str, rho: float, fit_units=(FIT_UNIT,), shift: float = 0.0) -> None:
        self.name = name
        self.rho = rho
        self.fit_units = tuple(fit_units)
        self.shift = shift
        self.fitted: list[pd.DataFrame] = []
        self.predicted: list[pd.DataFrame] = []

    def fit(self, frame: pd.DataFrame) -> None:
        self.fitted.append(frame.copy())

    def predict(self, frame: pd.DataFrame, kappas) -> pd.DataFrame:
        self.predicted.append(frame.copy())
        least = frame["history_min_departure_h"].to_numpy(float)
        d_hat = np.where(np.isfinite(least), least - 0.3 + self.shift, 0.5)
        parts = []
        for kappa in kappas:
            part = frame[list(KEY_COLUMNS)].reset_index(drop=True)
            part["kappa"] = float(kappa)
            part["rho"] = float(self.rho)
            part["d_hat"] = d_hat
            part["r_hat"] = 2.0
            part["e_cap"] = 1.0
            part["e_need"] = 1.0
            part["d_median"] = d_hat
            for column in SURVIVAL_COLUMNS:
                part[column] = np.nan
            parts.append(part[list(TABLE_COLUMNS)])
        return pd.concat(parts, ignore_index=True)


def register(monkeypatch, name: str, **attributes) -> dict:
    """The module `fleetreserve.models.<name>` whose `make_model` builds a `HistoryModel`; the
    models it built are kept in the returned dict under "built"."""
    made = {"built": []}

    def make_model(matrix):
        model = HistoryModel(name, matrix.rho, **attributes)
        made["built"].append(model)
        return model

    module = types.ModuleType(f"fleetreserve.models.{name}")
    module.make_model = make_model
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return made
