"""Made-up gate units whose days have different numbers of cars, for the tests of the fleet-size
what-if.

`varied_cars(n_days, site)`: like `gate_fake.gate_cars` (plugged in at 8:00, τ at 10:00, one service
hour, the second car leaving early on every `early_every`-th day), but day k has `3 + k % 4` cars,
so the fleets are of 3 to 6 cars (a day counts from 3 cars on).

`varied_units(n_days)`: the three gate units of `gate_fake.GATE_UNITS`, the first too small to
calibrate, the other two with these fleets.
"""

from gate_fake import GATE_UNITS
from truth_tables_fake import FakeCar, date_of

LEAVES = (14.4, 15.6, 16.7, 15.2, 14.9, 16.1)


def varied_cars(n_days: int, site: str = "A", early_every: int = 12) -> list[FakeCar]:
    cars = []
    for k in range(n_days):
        for j in range(3 + k % 4):
            out = LEAVES[j]
            if j == 1 and early_every and k % early_every == 0:
                out = 11.3
            cars.append(
                FakeCar(
                    site,
                    date_of(k),
                    8.0,
                    out + 0.01 * (k % 7),
                    rate=2.0 + 0.5 * (j % 3),
                    room=1.0 + 0.5 * (j % 3) + 0.1 * (k % 4),
                    session=f"{site}-{k}-{j}",
                )
            )
    return cars


def varied_units(n_days: int = 60) -> dict:
    half = n_days // 2
    return {
        GATE_UNITS[0]: varied_cars(10),
        GATE_UNITS[1]: varied_cars(n_days),
        GATE_UNITS[2]: varied_cars(half, "B") + varied_cars(n_days - half, "C", early_every=9),
    }
