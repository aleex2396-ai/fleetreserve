"""Made-up cars with Weibull departures, for the tests of the Cox and Weibull AFT models.

Each day has `PER_DAY` cars, one per car slot j; the car's arrival hour is `a = 6 + j % 4` (clock
hours 6 to 9, plugged in at half past; τ is 10:00). Its departure, in hours from τ, is an
independent Weibull draw with shape `SHAPE` and scale `true_scale(a)`, so the level-q quantile
given departure above h is `true_quantile(a, q, h)`. A later arrival gives a longer stay. The
Weibull family with one shape is also a proportional-hazards family, so the same cars serve the
Cox model.

`repeat_users`: slots below it have the same user on every day of a site (`r<j>`), so from the
second day on they have an earlier session (history columns are numbers); the other slots have a
new user each day (history columns missing). With `repeat_stretch` the scale of the repeat users
is multiplied by that factor.
"""

import math

import numpy as np
from survival_fake import cars_table
from truth_tables_fake import FakeCar, date_of

PER_DAY = 12
SHAPE = 2.0
BASE_SCALE = 4.0
ARRIVAL_EFFECT = 0.3


def true_scale(arrival_hour) -> np.ndarray:
    return BASE_SCALE * np.exp(ARRIVAL_EFFECT * (np.asarray(arrival_hour, float) - 7.5))


def true_quantile(arrival_hour, level: float, after: float = 0.0) -> np.ndarray:
    scale = true_scale(arrival_hour)
    return scale * ((after / scale) ** SHAPE - math.log1p(-level)) ** (1.0 / SHAPE)


def weibull_cars(
    n_days: int,
    seed: int,
    site: str = "S",
    first_day: int = 0,
    repeat_users: int = 0,
    repeat_stretch: float = 1.0,
):
    """The cars table (`UnitDays.cars`) of the made-up days (module text)."""
    rng = np.random.default_rng(seed)
    fakes, users = [], []
    for k in range(first_day, first_day + n_days):
        for j in range(PER_DAY):
            a = 6 + j % 4
            repeat = j < repeat_users
            scale = float(true_scale(a)) * (repeat_stretch if repeat else 1.0)
            stay = scale * rng.weibull(SHAPE)
            fakes.append(
                FakeCar(
                    site,
                    date_of(k),
                    a + 0.5,
                    10.0 + stay,
                    rate=3.0 + rng.random(),
                    room=1.0 + 2.0 * rng.random(),
                    session=f"{site}-{k}-{j}",
                    energy=4.0,
                )
            )
            users.append(f"r{j}" if repeat else f"{site}-u{k}-{j}")
    return cars_table(fakes, users)
