"""Made-up prepared days for the tests of the sizing what-if (`size_label`, `size_features`,
`size_scale`).

`random_day(rng, k)`: a day of two service hours with two to four cars, prepared by
`gate_day.prepare_day` with the stand-in forecaster of `test_gate_day` (known rate and room). Each
car's true departure hour is 1 to 5; its forecast departure is the true one, or, with probability
`early`, one to two hours later (the car leaves before its forecast, so the fixed plan can plan
power for a car that is gone); with probability `soon` instead one hour earlier, when the true
departure is after hour 1 (the forecast is on the safe side). Rates 2 to 4 kW, rooms up to half of
what the rate allows.

`prepared_days(seed, n, early, soon=0)`: `n` such days on consecutive dates from 1 January 2020.

`ramp_terms(value, thresholds, driver)`: a day's `size_scale.DayTerms` written by hand as driver
ramps `(c, s, total)`, its cars given no service hour the every-scale bound reads, so no power
row can be broken and the driver loss is the ramp `max(0, c + beta s)` summed over `total`, bit
for bit (`fixed_plan_loss`, reading R3).
"""

import datetime

import numpy as np
from test_gate_day import SETTINGS, Stub, configuration, truth_day

from fleetreserve import gate_day
from fleetreserve.day import Day, ForecastCar, TrueCar, covering_recovery_hours
from fleetreserve.fixed_plan_loss import ServiceHours
from fleetreserve.size_scale import DayTerms

START = datetime.date(2020, 1, 1)
N_SERVICE = 2


def random_day(rng, k: int, early: float = 0.3, unit: str = "u", soon: float = 0.0):
    cars, forecasts = [], []
    for _ in range(int(rng.integers(2, 5))):
        departure = int(rng.integers(1, 6))
        rate = float(rng.uniform(2.0, 4.0))
        room = float(rng.uniform(0.2, 0.5) * rate * departure)
        cars.append(TrueCar(departure, rate, room))
        shift = int(rng.integers(1, 3)) if rng.random() < early else 0
        if rng.random() < soon:
            shift = -min(1, departure - 1)
        forecasts.append(ForecastCar(float(departure + shift), rate, room, room))
    day = Day(
        unit,
        START + datetime.timedelta(days=k),
        N_SERVICE,
        covering_recovery_hours(cars, N_SERVICE),
        0.0,
        tuple(cars),
    )
    return gate_day.prepare_day(truth_day(day), Stub(forecasts), configuration(), SETTINGS)


def prepared_days(seed: int, n: int, early: float = 0.3, unit: str = "u", soon: float = 0.0):
    rng = np.random.default_rng(seed)
    return [random_day(rng, k, early, unit, soon) for k in range(n)]


def ramp_terms(value: float, thresholds: np.ndarray, driver) -> DayTerms:
    """`DayTerms` of the ramps `driver = (c, s, total)` whose cars have no service hour (module
    text)."""
    k = np.asarray(driver[0]).size
    hours = ServiceHours(
        centre=np.zeros((k, 0)),
        gains=np.zeros((k, 0, 0)),
        spread=np.zeros((k, 0)),
        cap=np.zeros((k, 0)),
        thresholds=np.zeros((k, 0)),
        first=np.zeros(k),
        short=np.zeros(k),
    )
    return DayTerms(value, thresholds, driver, hours)
