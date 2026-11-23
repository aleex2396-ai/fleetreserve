"""One site-day: its hours, its cars' true limits and their forecasts (plan sections 1, 2 and 9).

Hours are whole hours counted from the decision hour. Hour 0 is the decision hour: it starts at the
decision time, when the box is sold, and comes before the service hours. The service hours `S` are
hours 1 to `n_service` (at most `MAX_SERVICE_HOURS`), and the recovery hours `R` are the
`n_recovery` hours after them. The day's hours are 0 to `n_hours - 1`, with
`n_hours = 1 + n_service + n_recovery`.

A car's true limits (plan section 2) are its departure `d` (the hour in which it is unplugged,
rounded down), its rate `r` (kW) and its room `e` (kWh: the session's energy minus the energy drawn
before the decision time). With the protected share set by `rho`, its true set is

    U = {u : 0 <= u_t <= r [t < d];  sum_{s <= t} u_s <= e for every t;  sum_t u_t >= (1 - rho) e}.

Power is never negative here, so the running sum is largest at the last hour, and the set is the
one-way `gpoly.Car` on the hours `0..d-1` with power in `[0, r]` and total energy in
`[(1 - rho) e, e]` (`TrueCar.one_way`).

Two-way charging (plan section 9) lets the car discharge at up to `r`, with its stored energy never
more than the budget `B` below its level at the decision time, and its protected energy still met
by the end. `TrueCar.two_way` gives a `StoredEnergyCar`: power in `[-r, r]` in hours `0..d-1`; the
energy stored since the decision time, after every hour, in `[-B, e]`; and at least `(1 - rho) e`
stored at the end. This is the reading of section 9 used here (reading A below).

Readings of the plan where its text is not explicit:

1. A car may draw power in hour 0. Section 2 has policies whose gains are zero "if t precedes S",
   so hour 0 carries a planned power, and the room counts energy from the decision time on.
2. The recovery hours run until the last car has left: `n_recovery` must be at least the number
   of hours after `S` before the latest departure (`covering_recovery_hours`), so no car's hours
   are cut off. The true sets use each car's own departure, so extra recovery hours change nothing
   in them.
3. A car with departure 0 (unplugged within the decision hour) can draw nothing; it is refused
   here and is left out of the day by whoever builds it.
4. A day whose true set is empty for some car (`(1 - rho) e > r * d`, possible once departures are
   rounded down) is refused with `ValueError`: the ceiling of such a day is not defined.
5. In-window departure (section 1: `d <= last hour of S`) is `d <= n_service`: such a car misses at
   least the last service hour.
6. Two-way, reading A: the room limit and the budget hold after every hour, not only at the end;
   discharge is allowed in hour 0 and in the recovery hours as in the service hours. Reading B,
   the limits of `gpoly.Car` with power in `[-r, r]` and only the total energy in
   `[(1 - rho) e, e]`, lets the stored energy go above the room or more than `B` below its start
   during the day, and never uses `B`; it is not used for true cars.
7. Forecast set (section 2): `f_t` is the share of hour `t` before the forecast departure,
   `min(max(d_hat - t, 0), 1)`. "Window energy" is the energy drawn in hours 0 to `n_service`;
   "recovery hours before d_hat" counts each recovery hour by its share `f_t`, so a fractional last
   hour counts in part.
"""

import datetime
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from fleetreserve.gpoly import Car

MAX_SERVICE_HOURS = 8  # plan section 1: S has at most 8 hours


def _integer(value, name: str, smallest: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if value < smallest:
        raise ValueError(f"{name} must be at least {smallest}, got {value}")
    return int(value)


def _number(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if number < 0 or (positive and number == 0):
        raise ValueError(
            f"{name} must be {'positive' if positive else 'non-negative'}, got {value}"
        )
    return number


def _rho(rho) -> float:
    share = _number(rho, "rho")
    if share >= 1:
        raise ValueError(f"rho must be below 1, got {rho}")
    return share


@dataclass(frozen=True)
class StoredEnergyCar:
    """A lossless car whose stored energy is limited after every hour.

    Power `u_t` in `[low, high]` in hours `0..n_hours-1` and 0 afterwards. The energy stored since
    the decision time, `x_t = sum_{s <= t} u_s`, lies in `[stored_low, stored_high]` after every
    hour, and the last one is at least `need`. It starts at 0, so `stored_low <= 0 <= stored_high`;
    `low <= 0 <= high`, so the car may also stay idle.
    """

    n_hours: int
    low: float
    high: float
    stored_low: float
    stored_high: float
    need: float

    def __post_init__(self):
        _integer(self.n_hours, "n_hours", 1)
        values = (self.low, self.high, self.stored_low, self.stored_high, self.need)
        if not all(isinstance(v, Real) and math.isfinite(float(v)) for v in values):
            raise ValueError("the limits of a stored-energy car must be finite numbers")
        if not self.low <= 0 <= self.high:
            raise ValueError(f"power limits [{self.low}, {self.high}] must include 0")
        if not self.stored_low <= 0 <= self.stored_high:
            raise ValueError(
                f"stored-energy limits [{self.stored_low}, {self.stored_high}] must include 0"
            )
        if self.need > min(self.stored_high, self.high * self.n_hours):
            raise ValueError(
                f"the need {self.need} cannot be stored within {self.n_hours} hours at "
                f"{self.high} kW and a limit of {self.stored_high}"
            )


@dataclass(frozen=True)
class TrueCar:
    """A car's true limits: departure hour `departure` (draws in hour t only if t < departure),
    rate `rate` (kW) and room `room` (kWh)."""

    departure: int
    rate: float
    room: float

    def __post_init__(self):
        object.__setattr__(self, "departure", _integer(self.departure, "departure", 1))
        object.__setattr__(self, "rate", _number(self.rate, "rate", positive=True))
        object.__setattr__(self, "room", _number(self.room, "room"))

    def need(self, rho) -> float:
        """The protected energy `(1 - rho) * room`."""
        return (1 - _rho(rho)) * self.room

    def departs_in_window(self, n_service: int) -> bool:
        """True when the car leaves by the last service hour (`departure <= n_service`)."""
        return self.departure <= n_service

    def one_way(self, rho) -> Car:
        """The true set as a one-way `gpoly.Car` on hours `0..departure-1`.

        Raises `ValueError` when the protected energy is above what the rate allows.
        """
        return Car(frozenset(range(self.departure)), 0.0, self.rate, self.need(rho), self.room)

    def two_way(self, rho, budget) -> StoredEnergyCar:
        """The two-way true set at budget `budget` (kWh), reading A of plan section 9."""
        limit = _number(budget, "budget")
        return StoredEnergyCar(
            self.departure, -self.rate, self.rate, -limit, self.room, self.need(rho)
        )


@dataclass(frozen=True)
class ForecastCar:
    """A car's forecast limits at level kappa: departure `departure` (may be fractional), rate
    `rate`, stored-energy cap `cap` and needed energy `need` (plan section 2).

    The forecast set: `u_t <= rate * f_t` with `f_t` the share of hour `t` before `departure`; the
    stored energy at most `cap`; the energy drawn in hours 0 to `n_service` plus
    `rate * (the recovery hours' shares)` at least `need`.

    `scenarios` (empty unless the driver-risk rows are used): the level-wise departure bounds
    below `departure` with their probability masses, as `(bound, mass)` pairs, built by
    `driver_risk.departure_scenarios`. It lives on the car because the bounds come from the same
    forecaster as the car's other limits and one car carries them to the rows.
    """

    departure: float
    rate: float
    cap: float
    need: float
    scenarios: tuple[tuple[float, float], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "departure", _number(self.departure, "forecast departure"))
        object.__setattr__(self, "rate", _number(self.rate, "forecast rate"))
        object.__setattr__(self, "cap", _number(self.cap, "forecast cap"))
        object.__setattr__(self, "need", _number(self.need, "forecast need"))
        object.__setattr__(self, "scenarios", self._checked_scenarios())

    def _checked_scenarios(self) -> tuple[tuple[float, float], ...]:
        """The departure scenarios of the driver-risk rows (plan section 2) as pairs
        `(bound, mass)`: bounds that do not decrease, at most `departure`; masses positive."""
        out = []
        for entry in self.scenarios:
            bound, mass = entry
            bound = _number(bound, "scenario departure")
            mass = _number(mass, "scenario mass", positive=True)
            if (out and bound < out[-1][0]) or bound > self.departure:
                raise ValueError(
                    f"scenario departures must not decrease and must be at most the forecast "
                    f"departure {self.departure}, got {bound}"
                )
            out.append((bound, mass))
        return tuple(out)

    def share(self, hour: int) -> float:
        """`f_t`: the share of hour `hour` before the forecast departure, in [0, 1]."""
        return min(max(self.departure - hour, 0.0), 1.0)

    def power_cap(self, hour: int) -> float:
        """The forecast power limit `rate * f_t` in hour `hour`."""
        return self.rate * self.share(hour)

    def recovery_supply(self, n_service: int, n_recovery: int) -> float:
        """`rate` times the recovery hours before the forecast departure (shares counted)."""
        hours = range(n_service + 1, n_service + 1 + n_recovery)
        return self.rate * sum(self.share(t) for t in hours)

    def window_need(self, n_service: int, n_recovery: int) -> float:
        """The least energy to draw in hours 0 to `n_service`: `need` minus the recovery supply."""
        return self.need - self.recovery_supply(n_service, n_recovery)


def covering_recovery_hours(cars: Sequence[TrueCar], n_service: int) -> int:
    """The fewest recovery hours that keep every car's hours inside the day."""
    return max([0, *(car.departure - 1 - n_service for car in cars)])


@dataclass(frozen=True)
class Day:
    """One site-day of a unit: the window, the true cars and, optionally, their forecasts.

    `forecasts[i]` is car `i`'s forecast at level `kappa`; both are given or both left out.
    `rho` is the share of the room that is not protected (plan section 2).
    """

    unit: str
    date: datetime.date
    n_service: int
    n_recovery: int
    rho: float
    cars: tuple[TrueCar, ...]
    forecasts: tuple[ForecastCar, ...] | None = None
    kappa: float | None = None

    def __post_init__(self):
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError(f"unit must be a non-empty name, got {self.unit!r}")
        if isinstance(self.date, datetime.datetime) or not isinstance(self.date, datetime.date):
            raise ValueError(f"date must be a calendar date, got {self.date!r}")
        n_service = _integer(self.n_service, "n_service", 1)
        if n_service > MAX_SERVICE_HOURS:
            raise ValueError(f"{n_service} service hours is above {MAX_SERVICE_HOURS}")
        object.__setattr__(self, "n_service", n_service)
        object.__setattr__(self, "n_recovery", _integer(self.n_recovery, "n_recovery", 0))
        object.__setattr__(self, "rho", _rho(self.rho))
        cars = tuple(self.cars)
        if not all(isinstance(car, TrueCar) for car in cars):
            raise ValueError("every car must be a TrueCar")
        object.__setattr__(self, "cars", cars)
        for i, car in enumerate(cars):
            if car.departure > self.n_hours:
                raise ValueError(
                    f"car {i} leaves in hour {car.departure}, after the day's {self.n_hours} hours"
                )
            if car.need(self.rho) > car.rate * car.departure:
                raise ValueError(
                    f"car {i}: protected energy {car.need(self.rho)} is above rate {car.rate} "
                    f"times {car.departure} hours; its true set is empty"
                )
        self._check_forecasts()

    def _check_forecasts(self):
        if (self.forecasts is None) != (self.kappa is None):
            raise ValueError("forecasts and kappa are given together or not at all")
        if self.forecasts is None:
            return
        forecasts = tuple(self.forecasts)
        if len(forecasts) != len(self.cars):
            raise ValueError(f"{len(forecasts)} forecasts for {len(self.cars)} cars")
        if not all(isinstance(f, ForecastCar) for f in forecasts):
            raise ValueError("every forecast must be a ForecastCar")
        for i, forecast in enumerate(forecasts):
            if forecast.departure > self.n_hours:
                raise ValueError(f"forecast {i} leaves after the day's {self.n_hours} hours")
        object.__setattr__(self, "forecasts", forecasts)
        kappa = _number(self.kappa, "kappa", positive=True)
        if kappa >= 1:
            raise ValueError(f"kappa must be below 1, got {self.kappa}")
        object.__setattr__(self, "kappa", kappa)

    @property
    def n_hours(self) -> int:
        """Hours in the day: the decision hour, the service hours and the recovery hours."""
        return 1 + self.n_service + self.n_recovery

    @property
    def service_hours(self) -> range:
        return range(1, self.n_service + 1)

    @property
    def recovery_hours(self) -> range:
        return range(self.n_service + 1, self.n_hours)

    def departs_in_window(self, i: int) -> bool:
        """True when car `i` leaves by the last service hour."""
        return self.cars[i].departs_in_window(self.n_service)

    @property
    def has_in_window_departure(self) -> bool:
        return any(car.departs_in_window(self.n_service) for car in self.cars)

    def one_way_cars(self) -> tuple[Car, ...]:
        """The cars' one-way true sets."""
        return tuple(car.one_way(self.rho) for car in self.cars)

    def two_way_cars(self, budget) -> tuple[StoredEnergyCar, ...]:
        """The cars' two-way true sets at budget `budget` (kWh)."""
        return tuple(car.two_way(self.rho, budget) for car in self.cars)
