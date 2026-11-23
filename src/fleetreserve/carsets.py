"""Each car's set as linear rows over the decision hour and the service hours (plan 1, 2, 9).

The hours of a row are 0 (the decision hour) to `n_service`; a row is `a . u <= b`, with `u` the
car's power in those hours. The recovery hours are not steered: there "cars charge at full power
until full or gone" (plan section 1). Each car's recovery is therefore one number, the energy its
recovery hours can still supply before it leaves, and it enters the row of the car's needed energy:

- true car: `r` times the whole recovery hours before its departure, `r * max(d - 1 - n, 0)`;
- forecast car: `r_hat` times the sum of the shares `f_t` of the recovery hours
  (`day.ForecastCar.recovery_supply`).

Charging at full power until full meets the needed energy exactly when the stored energy at the
end of the service hours plus that supply reaches it, and it never leaves the stored-energy limits
(the stored energy only rises, up to the room). So the rows below are the car's set projected on
hours 0 to `n_service` with every recovery hour free, which is how `exact` treats them.

Row kinds, one row per hour `t` in 0 to `n_service` unless stated:

- `"power-high"`: `u_t <= high_t`;  `"power-low"`: `-u_t <= -low_t`.
- `"stored-high"`: `sum_{s <= t} u_s <= cap`;  `"stored-low"`: `-sum_{s <= t} u_s <= budget`
  (two-way cars only).
- `"need"`: one row, `-sum_{s <= n} u_s <= supply - need`.
- `"energy-high"`: one row, `sum_{s <= n} u_s <= e_high - low * R` (a `gpoly.Car`, which limits
  only its total energy; `R` its recovery hours).

The sets:

- true one-way car (plan section 2): power in `[0, r]` while present (`t < d`), 0 afterwards;
  stored energy at most `e` after every hour; the need `(1 - rho) e`.
- true two-way car at budget `B` (plan section 9, reading A of `day`): power in `[-r, r]` while
  present, 0 afterwards; stored energy in `[-B, e]` after every hour; the need `(1 - rho) e`.
- forecast car (plan section 2): power in `[0, r_hat f_t]`; stored energy at most `e_cap`; the need
  `e_need`.
- `gpoly.Car` (the toy fleets of `tests/wp1`): power in `[low, high]` in its hours and 0 in hour 0
  unless hour 0 is one of them; total energy in `[e_low, e_high]`, every one of its hours after
  `n_service` a recovery hour that is free within its power limits (so the total over hours 0 to
  `n_service` lies in `[e_low - high R, e_high - low R]`). It has no row for the stored energy
  after each hour, as `gpoly.Car` has none.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fleetreserve.day import MAX_SERVICE_HOURS, Day, ForecastCar, TrueCar
from fleetreserve.gpoly import Car

KINDS = ("power-high", "power-low", "stored-high", "stored-low", "need", "energy-high")


@dataclass(frozen=True)
class CarRows:
    """One car's rows `coefficients[k] . u <= bounds[k]` over hours 0 to `n_service`."""

    n_service: int
    coefficients: np.ndarray  # (rows, n_service + 1)
    bounds: np.ndarray  # (rows,)
    kinds: tuple[str, ...]

    def __post_init__(self):
        if not 1 <= self.n_service <= MAX_SERVICE_HOURS:
            raise ValueError(f"n_service must be 1 to {MAX_SERVICE_HOURS}, got {self.n_service}")
        a = np.array(self.coefficients, dtype=float)
        b = np.array(self.bounds, dtype=float)
        if a.ndim != 2 or a.shape[1] != self.n_service + 1 or b.shape != (a.shape[0],):
            raise ValueError("coefficients must be (rows, n_service + 1) and bounds (rows,)")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("rows must be finite")
        if len(self.kinds) != a.shape[0] or not set(self.kinds) <= set(KINDS):
            raise ValueError(f"one kind per row, each one of {KINDS}")
        a.flags.writeable = False
        b.flags.writeable = False
        object.__setattr__(self, "coefficients", a)
        object.__setattr__(self, "bounds", b)
        object.__setattr__(self, "kinds", tuple(self.kinds))


@dataclass(frozen=True)
class FleetRows:
    """The rows of every car of a fleet, stacked: row `k` belongs to car `car[k]`."""

    n_service: int
    n_cars: int
    car: np.ndarray  # (rows,) car index
    coefficients: np.ndarray  # (rows, n_service + 1)
    bounds: np.ndarray  # (rows,)
    kinds: np.ndarray  # (rows,) kind names

    @property
    def n_rows(self) -> int:
        return int(self.bounds.size)

    @property
    def last_hours(self) -> np.ndarray:
        """Per row, the last hour with a non-zero coefficient (the hour the row closes)."""
        nonzero = self.coefficients != 0
        hours = self.n_service - np.argmax(nonzero[:, ::-1], axis=1)
        return np.where(nonzero.any(axis=1), hours, 0)

    def slack(self, powers) -> np.ndarray:
        """`b - a . u` per row, with `powers[i]` car `i`'s powers in hours 0 to `n_service`."""
        u = np.asarray(powers, dtype=float)
        if u.shape != (self.n_cars, self.n_service + 1):
            raise ValueError(f"powers must be ({self.n_cars}, {self.n_service + 1})")
        return self.bounds - np.einsum("kt,kt->k", self.coefficients, u[self.car])


def stack(cars: Sequence[CarRows]) -> FleetRows:
    """The fleet's rows from each car's rows (all with the same service hours)."""
    if not cars:
        raise ValueError("a fleet needs at least one car")
    n = cars[0].n_service
    if any(c.n_service != n for c in cars):
        raise ValueError("every car's rows must have the same service hours")
    return FleetRows(
        n_service=n,
        n_cars=len(cars),
        car=np.concatenate([np.full(c.bounds.size, i) for i, c in enumerate(cars)]),
        coefficients=np.vstack([c.coefficients for c in cars]),
        bounds=np.concatenate([c.bounds for c in cars]),
        kinds=np.array([k for c in cars for k in c.kinds]),
    )


def _rows(n_service, low, high, stored_high, stored_low, need_bound, energy_high=None):
    """Rows from per-hour power limits `low`, `high`, stored-energy limits (None: no such rows),
    the need row's bound and, optionally, a total-energy upper row."""
    hours = n_service + 1
    eye = np.eye(hours)
    prefix = np.tril(np.ones((hours, hours)))
    blocks = [eye, -eye]
    bounds = [np.asarray(high, float), -np.asarray(low, float)]
    kinds = ["power-high"] * hours + ["power-low"] * hours
    if stored_high is not None:
        blocks.append(prefix)
        bounds.append(np.full(hours, float(stored_high)))
        kinds += ["stored-high"] * hours
    if stored_low is not None:
        blocks.append(-prefix)
        bounds.append(np.full(hours, float(stored_low)))
        kinds += ["stored-low"] * hours
    if energy_high is not None:
        blocks.append(np.ones((1, hours)))
        bounds.append(np.array([float(energy_high)]))
        kinds.append("energy-high")
    blocks.append(-np.ones((1, hours)))
    bounds.append(np.array([float(need_bound)]))
    kinds.append("need")
    return CarRows(n_service, np.vstack(blocks), np.concatenate(bounds), tuple(kinds))


def _true_present(car: TrueCar, n_service: int) -> np.ndarray:
    return (np.arange(n_service + 1) < car.departure).astype(float)


def true_recovery_supply(car: TrueCar, n_service: int) -> float:
    """`r` times the whole recovery hours before the car leaves."""
    return car.rate * max(car.departure - 1 - n_service, 0)


def true_one_way_rows(car: TrueCar, n_service: int, rho) -> CarRows:
    """The true one-way set (plan section 2) on hours 0 to `n_service`, recovery eliminated."""
    present = _true_present(car, n_service)
    need = car.need(rho)
    supply = true_recovery_supply(car, n_service)
    return _rows(n_service, 0.0 * present, car.rate * present, car.room, None, supply - need)


def true_two_way_rows(car: TrueCar, n_service: int, rho, budget) -> CarRows:
    """The true two-way set at budget `budget` kWh (plan section 9, reading A of `day`)."""
    limits = car.two_way(rho, budget)  # validates the budget
    present = _true_present(car, n_service)
    supply = true_recovery_supply(car, n_service)
    return _rows(
        n_service,
        limits.low * present,
        limits.high * present,
        limits.stored_high,
        -limits.stored_low,
        supply - limits.need,
    )


def forecast_rows(forecast: ForecastCar, n_service: int, n_recovery: int) -> CarRows:
    """The forecast set (plan section 2) on hours 0 to `n_service`, recovery eliminated."""
    high = np.array([forecast.power_cap(t) for t in range(n_service + 1)])
    return _rows(
        n_service,
        np.zeros(n_service + 1),
        high,
        forecast.cap,
        None,
        -forecast.window_need(n_service, n_recovery),
    )


def gpoly_rows(car: Car, n_service: int) -> CarRows:
    """A `gpoly.Car`'s set on hours 0 to `n_service`; its later hours are free recovery hours."""
    if min(car.hours) < 0:
        raise ValueError("a car's hours must be 0 or later")
    present = np.array([float(t in car.hours) for t in range(n_service + 1)])
    recovery = sum(1 for t in car.hours if t > n_service)
    return _rows(
        n_service,
        car.low * present,
        car.high * present,
        None,
        None,
        car.high * recovery - car.e_low,
        energy_high=car.e_high - car.low * recovery,
    )


def true_rows(day: Day, budget=None) -> FleetRows:
    """The day's true sets: one-way, or two-way at `budget` kWh."""
    if budget is None:
        cars = [true_one_way_rows(car, day.n_service, day.rho) for car in day.cars]
    else:
        cars = [true_two_way_rows(car, day.n_service, day.rho, budget) for car in day.cars]
    return stack(cars)


def day_forecast_rows(day: Day) -> FleetRows:
    """The day's forecast sets; the day must carry forecasts."""
    if day.forecasts is None:
        raise ValueError("the day has no forecasts")
    return stack([forecast_rows(f, day.n_service, day.n_recovery) for f in day.forecasts])


def gpoly_fleet_rows(cars: Sequence[Car], n_service: int) -> FleetRows:
    """A fleet of `gpoly.Car` (toy fleets) as rows."""
    return stack([gpoly_rows(car, n_service) for car in cars])
