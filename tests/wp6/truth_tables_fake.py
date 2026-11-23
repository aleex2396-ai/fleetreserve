"""Made-up truth tables in the layout `truth_build` writes, for the tests of `days_from_truth`,
`forecast_empirical` and `forecast_report`.

Times are given in hours from midnight of a car's date (below 0 or above 24 for an earlier or a
later day) at a fixed offset of -06:00, so no daylight-saving change occurs. τ is at 10:00. The
cars table is built as `truth.build` builds it: `depart_hour = floor(plug-out - τ)`, the day counts
when at least 3 cars are listed; the room is given directly.

Declared departure: `declared` (hours from midnight of the car's date) and `declared_at` (when it
was stated, same clock) are written as given, `declared_departure_h = declared - τ` and the stamp
of `declared_at`, each empty when not given. Unlike `truth.build`, the fake does not drop a
statement made after τ, so the tests of what reads the tables can give one.
"""

import datetime
import math
from dataclasses import dataclass

import pandas as pd

from fleetreserve.truth import CAR_COLUMNS, COUNTING_CARS, DAY_COLUMNS

ZONE = datetime.timezone(datetime.timedelta(hours=-6))
TAU_HOUR = 10
SERVICE_HOURS = 6


def stamp(day: str, hours: float) -> str:
    midnight = datetime.datetime.fromisoformat(day).replace(tzinfo=ZONE)
    moment = midnight + datetime.timedelta(hours=hours)
    return moment.isoformat()


@dataclass
class FakeCar:
    site: str
    date: str
    plug_in: float  # hours from midnight of `date`
    plug_out: float
    rate: float = 3.0
    room: float = 5.0
    session: str | None = None
    energy: float | None = None
    declared: float | None = None
    declared_at: float | None = None
    from_declared: bool = False


def tables(cars: list[FakeCar], unit: str = "unit", extra_days=()) -> tuple:
    """The cars and days tables (as text, as `pd.read_csv(dtype=str)` gives them)."""
    rows = []
    for k, car in enumerate(cars):
        depart = math.floor(car.plug_out - TAU_HOUR)
        energy = car.energy if car.energy is not None else car.room + 1.0
        rows.append(
            {
                "unit": unit,
                "site": car.site,
                "date": car.date,
                "tau": stamp(car.date, TAU_HOUR),
                "session_id": car.session or f"s{k}",
                "user_id": f"user{k}",
                "station": "1",
                "plug_in": stamp(car.date, car.plug_in),
                "plug_out": stamp(car.date, car.plug_out),
                "depart_hour": str(depart),
                "in_window_departure": str(depart <= SERVICE_HOURS),
                "energy_kwh": f"{energy:.6f}",
                "nominal_kw": "6.600000",
                "nominal_rule": "source cap",
                "rate_kw": f"{car.rate:.6f}",
                "rate_rule": "user's highest average, all sessions",
                "rate_average_kind": "charging time",
                "energy_before_kwh": f"{energy - car.room:.6f}",
                "energy_before_rule": "from plug-in at the nominal cap",
                "room_kwh": f"{car.room:.6f}",
                "room_fits": str(car.room <= car.rate * depart),
                "declared_at": "" if car.declared_at is None else stamp(car.date, car.declared_at),
                "declared_departure_h": ""
                if car.declared is None
                else f"{car.declared - TAU_HOUR:.6f}",
                "plug_out_from_declared": str(car.from_declared),
            }
        )
    frame = pd.DataFrame(rows)
    per_day = frame.groupby(["site", "date"]).size()
    keys = zip(frame.site, frame.date, strict=True)
    frame["cars_in_day"] = [str(per_day[key]) for key in keys]
    frame["day_counts"] = [str(int(n) >= COUNTING_CARS) for n in frame["cars_in_day"]]
    frame = frame.sort_values(["date", "session_id"], kind="stable").reset_index(drop=True)
    keys = sorted(set(per_day.index) | set(extra_days), key=lambda key: (key[1], key[0]))
    days = pd.DataFrame(
        {
            "unit": unit,
            "site": [s for s, _ in keys],
            "date": [d for _, d in keys],
            "tau": [stamp(d, TAU_HOUR) for _, d in keys],
            "service_hours": str(SERVICE_HOURS),
            "cars": [str(per_day.get((s, d), 0)) for s, d in keys],
            "counts": [str(per_day.get((s, d), 0) >= COUNTING_CARS) for s, d in keys],
        }
    )
    return frame[list(CAR_COLUMNS)], days[list(DAY_COLUMNS)]


def write(folder, unit: str, cars: list[FakeCar], extra_days=()) -> None:
    car_table, day_table = tables(cars, unit, extra_days)
    car_table.to_csv(folder / f"{unit}-cars.csv", index=False, lineterminator="\n")
    day_table.to_csv(folder / f"{unit}-days.csv", index=False, lineterminator="\n")


def date_of(k: int) -> str:
    return (datetime.date(2020, 1, 1) + datetime.timedelta(days=k)).isoformat()


def regular_days(n_days: int, site: str = "A", per_day: int = 4) -> list[FakeCar]:
    """`n_days` days of `per_day` cars, plugged in at 8:00 and leaving at different hours."""
    cars = []
    for k in range(n_days):
        for j in range(per_day):
            cars.append(
                FakeCar(
                    site,
                    date_of(k),
                    8.0,
                    12.25 + j + 0.1 * (k % 5),
                    rate=2.0 + 0.5 * j,
                    room=3.0 + j + 0.2 * (k % 3),
                    session=f"{site}-{k}-{j}",
                )
            )
    return cars
