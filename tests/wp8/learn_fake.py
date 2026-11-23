"""Made-up units for the WP8 tests.

`small_gate_units()`: the three gate units of `gate_fake`, shortened so a gate run takes seconds:
NREL before May 2019 with 5 days (not calibratable), NREL from May 2019 with 24 days, the pooled
Norwegian unit with 24 site-days at two sites. `small_matrix(names)`: the committed learn matrix
with `n_calibration` 20 (the fewest days the driver loss at 0.05 calibrates on is 19), no
half-the-days rule (`calibration_half_below` null) and only the named configurations.

`odd_unit(folder)`: one unit of 30 days of four cars plugged in at 8:00 with six service hours:
the first car leaves 2.25 to 2.65 hours after τ (within the window), the others 7.25 to 9.65
hours after τ (after it, so the days have flexibility to offer once the forecast departure is
late enough, as it is at kappa 0.3). Added: cars that reach every branch of the empirical
forecaster: a car arriving at another hour (its cell is too small, so the site is used), a car
plugged in on the day before, and one site with a single day (its cars have no reference
sample).
"""

import json

from gate_fake import GATE_UNITS, TOP, gate_cars, write_tables
from truth_tables_fake import FakeCar, date_of, write

from fleetreserve.days_from_truth import load_unit_days

LEARN_MATRIX = json.loads((TOP / "learn" / "matrix.json").read_text())
PAIR = ("main-k0.1-empirical", "main-k0.1-oracle")


def small_gate_units() -> dict:
    return {
        GATE_UNITS[0]: gate_cars(5),
        GATE_UNITS[1]: gate_cars(24, early_every=4),
        GATE_UNITS[2]: gate_cars(12, "B", early_every=3) + gate_cars(12, "C", early_every=5),
    }


def small_matrix(names=PAIR) -> dict:
    document = json.loads(json.dumps(LEARN_MATRIX))
    document["n_calibration"] = 20
    document["calibration_half_below"] = None
    document["configurations"] = [c for c in document["configurations"] if c["name"] in names]
    return document


def write_small_tables(folder) -> None:
    for unit, cars in small_gate_units().items():
        write_tables(folder, unit, cars)


def odd_cars() -> list[FakeCar]:
    cars = [
        FakeCar(
            "A",
            date_of(k),
            8.0,
            (12.25 if j == 0 else 16.25 + j) + 0.1 * (k % 5),
            rate=2.0 + 0.5 * j,
            room=1.0 + 0.2 * (k % 3),
            session=f"A-{k}-{j}",
        )
        for k in range(30)
        for j in range(4)
    ]
    cars.append(FakeCar("A", date_of(3), 9.0, 15.3, rate=4.0, room=2.0, session="late-arrival"))
    cars.append(FakeCar("A", date_of(5), -3.0, 13.7, rate=3.0, room=1.5, session="overnight"))
    for j in range(3):
        cars.append(FakeCar("Z", date_of(7), 8.0, 13.5 + j, rate=3.0, room=2.0, session=f"z{j}"))
    return cars


def odd_unit(folder, rho: float = 0.0):
    write(folder, "odd", odd_cars())
    return load_unit_days(folder, "odd", rho=rho, no_fit="cap", small_fleet="keep")
