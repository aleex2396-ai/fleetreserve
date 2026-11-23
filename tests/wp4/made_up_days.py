"""Seeded made-up site-days for the tests of `day` and `exact`.

`made_up_day(rng, n_service, rho)` draws 1 to 4 cars. Each car's departure is drawn from hour 1 to
three hours after the last service hour, so some cars leave inside the window and some stay into
the recovery hours. Its rate is 1, 2 or 3 kW. Its room is one of:

- tight: the protected energy equals the rate times the car's hours (it must charge at full power
  in every hour it is present);
- empty: zero room;
- drawn: a share of what the rate allows, kept so that the protected energy fits.

`rho` is passed in, so a sample can hold every value of plan section 2 (0, 0.1, 0.2); with
`rho = 0` every car needs exactly its room. The recovery hours cover the latest departure.
"""

import datetime
import math

from fleetreserve.day import Day, TrueCar, covering_recovery_hours

RHOS = (0.0, 0.1, 0.2)  # plan section 2
DATE = datetime.date(2020, 1, 1)


def made_up_car(rng, n_service, rho):
    rate = float(rng.choice([1, 2, 3]))
    departure = rng.randint(1, n_service + 3)
    kind = rng.choices(["tight", "empty", "drawn"], weights=[2, 1, 7])[0]
    most = rate * departure / (1 - rho)  # the largest room whose protected energy fits
    while (1 - rho) * most > rate * departure:  # step below rounding error
        most = math.nextafter(most, 0.0)
    if kind == "tight":
        room = most
    elif kind == "empty":
        room = 0.0
    else:
        room = min(rng.uniform(0.1, 1.3) * rate * departure, rng.uniform(0.5, 1.0) * most)
    return TrueCar(departure, rate, room), kind


def made_up_day(rng, n_service, rho):
    """A day and the kinds of its cars' rooms ("tight", "empty", "drawn")."""
    drawn = [made_up_car(rng, n_service, rho) for _ in range(rng.randint(1, 4))]
    cars = [car for car, _ in drawn]
    day = Day("made-up", DATE, n_service, covering_recovery_hours(cars, n_service), rho, cars)
    return day, [kind for _, kind in drawn]


def made_up_sample(seed_rng, count, service_range=(2, 4)):
    """`count` days, cycling through `RHOS`, with 2 to 4 service hours by default."""
    days = []
    kinds = []
    for k in range(count):
        day, day_kinds = made_up_day(seed_rng, seed_rng.randint(*service_range), RHOS[k % 3])
        days.append(day)
        kinds.append(day_kinds)
    return days, kinds
