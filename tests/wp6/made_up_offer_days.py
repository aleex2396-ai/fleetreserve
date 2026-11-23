"""Seeded made-up days with forecasts and a certified offer, for the tests of the rungs.

`offer_day(rng, k, kinds)` draws `n` (2 to 4) service hours and 2 to 5 cars; rho cycles through
0, 0.1 and 0.2 with `k`. Each car: rate 1, 2 or 3 kW; departure from hour 1 to three hours after
the window (so some leave inside it); room `U(0.1, 0.6)` times its rate times its hours, so its
true set is not empty. Each car's forecast is of a kind drawn from `kinds`:

- "conservative" (plan T2(c1)): `d_hat = d - U(0, 0.5)`, `r_hat = r U(0.85, 1)`,
  `e_cap = need + U(0.3, 1) (e - need)`, `e_need = need + U(0, 0.1) (e_cap - need)`;
- "early departure": conservative except `d_hat = d + U(0.5, 2.5)` (at most the day's hours): the
  car leaves before its forecast;
- "less room": conservative except `e_cap = e U(1.2, 1.8)`: the car has less room than forecast;
- "lower rate": conservative except `r_hat = r U(1.2, 1.6)`: the car's true rate is below its
  forecast.

The day is certified on its forecast sets with the causal structure, the hourly product for even
`k` and the block product for odd `k` (`ANY_OPTIMAL` shape). `large_offer_day` draws a day of 30
cars and 6 service hours for timing. `departure_rule(day, departure_seen)` is the
made-up survival-tightened bound used by the tests, for either reading of when a departure is
seen (`rungs.py`, information model).
"""

import dataclasses

from made_up_days import DATE, RHOS

from fleetreserve import products
from fleetreserve.carsets import day_forecast_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.day import Day, ForecastCar, TrueCar, covering_recovery_hours
from fleetreserve.rungs import DEPARTURE_SEEN

KINDS = ("conservative", "early departure", "less room", "lower rate")


def _forecast(rng, car: TrueCar, rho: float, kind: str, n_hours: int) -> ForecastCar:
    need = car.need(rho)
    cap = need + rng.uniform(0.3, 1) * (car.room - need)
    departure = car.departure - rng.uniform(0, 0.5)
    rate = car.rate * rng.uniform(0.85, 1.0)
    if kind == "early departure":
        departure = min(car.departure + rng.uniform(0.5, 2.5), float(n_hours))
    elif kind == "less room":
        cap = car.room * rng.uniform(1.2, 1.8)
    elif kind == "lower rate":
        rate = car.rate * rng.uniform(1.2, 1.6)
    elif kind != "conservative":
        raise ValueError(kind)
    return ForecastCar(departure, rate, cap, need + rng.uniform(0, 0.1) * (cap - need))


def offer_day(rng, k: int, kinds=KINDS):
    """A day with forecasts, its certificate and the kinds of its forecasts."""
    n = rng.randint(2, 4)
    rho = RHOS[k % 3]
    cars = []
    for _ in range(rng.randint(2, 5)):
        rate = float(rng.choice([1, 2, 3]))
        departure = rng.randint(1, n + 3)
        cars.append(TrueCar(departure, rate, rng.uniform(0.1, 0.6) * rate * departure))
    day = Day("made-up", DATE, n, covering_recovery_hours(cars, n), rho, tuple(cars))
    drawn = [rng.choice(kinds) for _ in cars]
    forecasts = tuple(
        _forecast(rng, car, rho, kind, day.n_hours) for car, kind in zip(cars, drawn, strict=True)
    )
    day = dataclasses.replace(day, forecasts=forecasts, kappa=0.1)
    product = products.hourly() if k % 2 == 0 else products.block()
    certificate = certify(day_forecast_rows(day), product, "causal", shape_rule=ANY_OPTIMAL)
    return day, certificate, drawn


def large_offer_day(rng, n_cars=30, n_service=6):
    """A larger day for timing: rates 3, 7 or 11 kW, departures from hour 2 to four hours after
    the window, rho 0.1, forecasts conservative or with an early departure; hourly product."""
    cars = []
    for _ in range(n_cars):
        rate = float(rng.choice([3, 7, 11]))
        departure = rng.randint(2, n_service + 4)
        cars.append(TrueCar(departure, rate, rng.uniform(0.1, 0.6) * rate * departure))
    n_recovery = covering_recovery_hours(cars, n_service)
    day = Day("large", DATE, n_service, n_recovery, 0.1, tuple(cars))
    kinds = KINDS[:2]
    forecasts = tuple(_forecast(rng, c, 0.1, rng.choice(kinds), day.n_hours) for c in cars)
    day = dataclasses.replace(day, forecasts=forecasts, kappa=0.1)
    rows = day_forecast_rows(day)
    return day, certify(rows, products.hourly(), "causal", shape_rule=ANY_OPTIMAL)


def departure_rule(day: Day, departure_seen: str):
    """The made-up survival-tightened bound for a car present at the start of hour `t`: under
    "hour-start" it may still leave within hour `t`, `max(d_hat, t)`; under "rounded-hour" it is
    present in hour `t`, `max(d_hat, t + 1)` (`rungs.py`, information model)."""
    if departure_seen not in DEPARTURE_SEEN:
        raise ValueError(departure_seen)
    floor = 0.0 if departure_seen == "hour-start" else 1.0
    return lambda car, hour: max(day.forecasts[car].departure, hour + floor)
