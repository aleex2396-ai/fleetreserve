"""Made-up gate units and a made-up verified truth build, for the tests of the gate run.

`gate_cars(n_days, site)`: `n_days` days of three cars at one site, plugged in at 8:00, τ at 10:00,
one service hour (hour 1, the shortest window, which keeps the tests quick). The cars leave about
4.4, 5.6 and 6.7 hours after τ (true departure hours 4, 5 and 6, so they can make up energy in the
recovery hours), with rates 2, 2.5 and 3 kW and rooms that vary with the day. On every twelfth day
the second car leaves early, 1.3 hours after τ (departure hour 1, within the service hour), so the
kappa-quantile of the departures is above the true departure of some cars: the fixed plan still
plans power for that car in hour 1 and fails there on some days and paths, and the calibration has
something to find. On such a day the two readings of `departure_seen` differ: under
"rounded-hour" a re-solving rung sees the car gone at the start of hour 1 and re-solves in time;
under "hour-start" it sees the car plugged in then and fails in hour 1 with the fixed plan.

`write_build(folder, units, service_hours)`: the tables of each unit (`truth_tables_fake.write`,
with the service hours set to `service_hours`, by default `SERVICE_HOURS`) and a recorded truth
build that lists their SHA-256, as `truth_build` writes it, with a manifest whose outputs match
(the commit check is replaced by the caller).

`matrix_document(names)`: the committed matrix with only the named configurations; by default the
main family at kappa 0.1 under both readings of `departure_seen`, each with its oracle partner.
The made-up units hold 60 days or fewer, so the document is given its own calibration-days rule:
50 calibration days and no half-the-days rule (`calibration_half_below` null), not the plan's.
"""

import copy
import hashlib
import json
from pathlib import Path

from truth_tables_fake import FakeCar, date_of, tables

SERVICE_HOURS = 1
COMMIT = "a" * 40
TOP = Path(__file__).resolve().parents[2]
COMMITTED = json.loads((TOP / "gate" / "matrix.json").read_text())


def gate_cars(
    n_days: int, site: str = "A", early_every: int = 12, early_out: float = 11.3
) -> list[FakeCar]:
    cars = []
    for k in range(n_days):
        leaves = [14.4, 15.6, 16.7]
        if early_every and k % early_every == 0:
            leaves[1] = early_out
        for j, out in enumerate(leaves):
            cars.append(
                FakeCar(
                    site,
                    date_of(k),
                    8.0,
                    out + 0.01 * (k % 7),
                    rate=2.0 + 0.5 * j,
                    room=1.0 + 0.5 * j + 0.1 * (k % 4),
                    session=f"{site}-{k}-{j}",
                )
            )
    return cars


def write_tables(
    folder: Path, unit: str, cars: list[FakeCar], service_hours: int = SERVICE_HOURS
) -> None:
    car_table, day_table = tables(cars, unit)
    day_table["service_hours"] = str(service_hours)
    car_table.to_csv(folder / f"{unit}-cars.csv", index=False, lineterminator="\n")
    day_table.to_csv(folder / f"{unit}-days.csv", index=False, lineterminator="\n")


def write_build(folder: Path, units: dict, service_hours: int = SERVICE_HOURS) -> tuple[Path, Path]:
    """Tables for `units` (name -> cars) in `folder/tables`, and a recorded truth build."""
    tables_folder = folder / "tables"
    tables_folder.mkdir()
    entries = {}
    for unit, cars in units.items():
        write_tables(tables_folder, unit, cars, service_hours)
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


ROUNDED = ("main-rounded-hour-k0.1-empirical", "main-rounded-hour-k0.1-oracle")
HOUR_START = ("main-k0.1-empirical", "main-k0.1-oracle")


def matrix_document(names=ROUNDED + HOUR_START) -> dict:
    document = copy.deepcopy(COMMITTED)
    document["configurations"] = [c for c in document["configurations"] if c["name"] in names]
    document["n_calibration"] = 50
    document["calibration_half_below"] = None
    return document


PLAN_LEVELS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)  # plan section 2's grid of kappa
PLAIN_NAMES = tuple(
    f"main-rounded-hour-k{k:g}-{d}" for k in PLAN_LEVELS for d in ("empirical", "oracle")
)


def matrix_document_six(names=PLAIN_NAMES) -> dict:
    """`matrix_document(names)` for names that may be at 0.4 and 0.5: the committed matrix lists
    the main family at all six levels of plan section 2, so every name is read from it."""
    return matrix_document(names)


GATE_UNITS = ("nrel-before-may-2019", "nrel-from-may-2019", "norway-development-pooled")


def gate_units(n_days: int = 60) -> dict:
    """The three gate units: the first too small to calibrate (10 days), the pooled one from two
    sites."""
    half = n_days // 2
    return {
        GATE_UNITS[0]: gate_cars(10),
        GATE_UNITS[1]: gate_cars(n_days),
        GATE_UNITS[2]: gate_cars(half, "B") + gate_cars(n_days - half, "C", early_every=9),
    }
