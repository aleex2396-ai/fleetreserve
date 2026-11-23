"""Made-up cars tables with a known conditional distribution, for the quantile models of WP8.

Each day has `PER_DAY` cars, one per car slot j; the car's arrival hour is `6 + j % 4` (clock
hours 6 to 9, plugged in at half past). Given the arrival hour a, independent draws:

- departure (hours from τ, 10:00): `2 + 1.5 * (a - 6) + 4 * U`;
- rate (kW): `3 + 0.5 * (a - 6) + 2 * V`;
- room (kWh): `1 + 3 * W`;

with U, V, W uniform on [0, 1]. So the level-q quantile given a is `2 + 1.5 * (a - 6) + 4q`
(departure), `3 + 0.5 * (a - 6) + 2q` (rate), `1 + 3q` (room). Every room fits (rate at least
3 kW, departure hour at least 2), so every car is in its day's fleet. Users are all different, so
no car has a history.
"""

import numpy as np
from truth_tables_fake import FakeCar, date_of, write

from fleetreserve.days_from_truth import load_unit_days

PER_DAY = 12


def cars(n_days: int, seed: int, site: str = "S", first_day: int = 0) -> list[FakeCar]:
    rng = np.random.default_rng(seed)
    out = []
    for k in range(first_day, first_day + n_days):
        for j in range(PER_DAY):
            a = 6 + j % 4
            u, v, w = rng.random(3)
            out.append(
                FakeCar(
                    site,
                    date_of(k),
                    a + 0.5,
                    10.0 + 2.0 + 1.5 * (a - 6) + 4.0 * u,
                    rate=3.0 + 0.5 * (a - 6) + 2.0 * v,
                    room=1.0 + 3.0 * w,
                    session=f"{site}-{k}-{j}",
                )
            )
    return out


def unit_cars(folder, unit: str, made_up: list[FakeCar], rho: float = 0.0):
    """The cars table (`UnitDays.cars`) of a unit made of `made_up`."""
    write(folder, unit, made_up)
    return load_unit_days(folder, unit, rho=rho, no_fit="cap", small_fleet="keep").cars
