"""Made-up tables and a made-up verified truth build for the tests of `o3_run`.

τ is at 10:00 (`truth_tables_fake`); a car's departure hour is the whole hours from τ to its
plug-out.

`hand_cars(site, date)`: the hand day of `test_o3_hand`: car A leaves 2.5 hours after τ
(departure hour 2) at 1 kW with 1 kWh of room; car B leaves 6.5 hours after τ (departure hour 6)
at 1 kW with 3 kWh; car C leaves within the decision hour, so `days_from_truth` leaves it out of
the fleet and the day still counts (three cars at τ). With two service hours the fleet is A and B.

`varied_cars(n_days, site, per_day, seed, service_hours)`: `n_days` days of `per_day` cars with
seeded departures (from within the window to four hours after it), rates and rooms (each room
fits the car's whole hours).

`build(folder, units, service_hours)`: `gate_fake.write_build` (the tables, and a recorded truth
build listing their SHA-256 with a manifest whose outputs match). `trust_commit(monkeypatch)`
replaces the two commit checks, as the gate tests do.
"""

import random

from gate_fake import COMMIT, write_build
from truth_tables_fake import TAU_HOUR, FakeCar, date_of

from fleetreserve import gate_run, run_verify


def hand_cars(site: str, date: str) -> list[FakeCar]:
    return [
        FakeCar(site, date, 8.0, TAU_HOUR + 2.5, rate=1.0, room=1.0, session=f"{site}-{date}-A"),
        FakeCar(site, date, 8.0, TAU_HOUR + 6.5, rate=1.0, room=3.0, session=f"{site}-{date}-B"),
        FakeCar(site, date, 8.0, TAU_HOUR + 0.5, rate=2.0, room=1.0, session=f"{site}-{date}-C"),
    ]


def varied_cars(
    n_days: int, site: str = "A", per_day: int = 4, seed: int = 0, service_hours: int = 3
) -> list[FakeCar]:
    rng = random.Random(seed)
    cars = []
    for k in range(n_days):
        for j in range(per_day):
            depart = rng.randint(1, service_hours + 4)
            rate = rng.choice([1.0, 2.0, 3.0, 6.6])
            cars.append(
                FakeCar(
                    site,
                    date_of(k),
                    8.0,
                    TAU_HOUR + depart + rng.uniform(0.05, 0.95),
                    rate=rate,
                    room=round(rng.uniform(0.0, 1.0) * rate * depart, 4),
                    session=f"{site}-{k}-{j}",
                )
            )
    return cars


def build(folder, units: dict, service_hours: int):
    """(truth run folder, tables folder) for `units` (name -> cars)."""
    return write_build(folder, units, service_hours)


def trust_commit(monkeypatch) -> None:
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
