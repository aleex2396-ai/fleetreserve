"""Per car of a day with forecasts: which bounds are conservative, and whether the forecast set is
empty (plan sections 2 and 5).

Conservative bounds, plan T2's event E'_i: `d_hat <= d` (the true departure hour, rounded down),
`r_hat <= r`, `e_cap <= room`, `e_need >= (1 - rho) room`; E' holds on a day when E'_i holds for
every car. `d_hat <= exact plug-out` and `floor(d_hat) <= d` are also given, for comparison only:
a fractional `d_hat` above the whole hour `d` fails E'_i even when it is before the exact plug-out.

Empty forecast set. The set is that of `carsets.forecast_rows`: power `0 <= u_t <= r_hat f_t` in
hours 0 to `n_service`, stored energy `sum_{s <= t} u_s <= e_cap` after each of those hours, and
`sum_{t <= n_service} u_t >= w`, with `w = e_need - r_hat * (sum of f_t over the recovery hours)`
(`ForecastCar.window_need`). With `H_t` the power rows' bounds and `C_t` the stored-energy rows'
bounds, the largest energy the rows allow in hours 0 to `n_service` is
`min(sum_t H_t, min_t (C_t + sum_{s > t} H_s))` (an upper bound from each row; reached by drawing
as much as allowed hour by hour), so the set is empty exactly when `w` is above it
(`forecast_set_empty`). For these rows it equals the two closed-form conditions
(`empty_conditions`): (a) `e_need > e_cap + recovery supply`; (b) `e_need > r_hat * (sum of f_t
over all the day's hours)`, the energy the car can take at `r_hat` before `d_hat`, recovery
included. A forecast departure is inside the service window when `d_hat < n_service + 1` (the car
is gone before the end of the last service hour; for a whole hour this is `d <= n_service`).
"""

import math
from dataclasses import dataclass

import numpy as np

from fleetreserve.carsets import CarRows, forecast_rows
from fleetreserve.day import ForecastCar
from fleetreserve.gpoly import TOLERANCE


def largest_window_energy(rows: CarRows) -> float:
    """The most energy a forecast car's rows (`carsets.forecast_rows`) allow in hours 0 to
    `n_service`."""
    kinds = np.array(rows.kinds)
    high = rows.bounds[kinds == "power-high"]
    cap = rows.bounds[kinds == "stored-high"]
    if not np.all(rows.bounds[kinds == "power-low"] == 0):
        raise ValueError("a forecast car's power is expected to start at 0")
    after = np.concatenate([np.cumsum(high[::-1])[::-1][1:], [0.0]])  # sum_{s > t} H_s
    return float(min(high.sum(), np.min(cap + after)))


def forecast_set_empty(forecast: ForecastCar, n_service: int, n_recovery: int) -> bool:
    """True when no power vector meets the forecast set's rows (see the module text)."""
    rows = forecast_rows(forecast, n_service, n_recovery)
    need_row = np.array(rows.kinds) == "need"
    least = -float(rows.bounds[need_row][0])  # the window need w
    return least > largest_window_energy(rows) + TOLERANCE


def empty_conditions(forecast: ForecastCar, n_service: int, n_recovery: int) -> tuple[bool, bool]:
    """The two closed-form conditions (a) and (b) of the module text."""
    supply = forecast.recovery_supply(n_service, n_recovery)
    whole_day = forecast.rate * sum(forecast.share(t) for t in range(1 + n_service + n_recovery))
    a = forecast.need > forecast.cap + supply + TOLERANCE
    b = forecast.need > whole_day + TOLERANCE
    return a, b


@dataclass(frozen=True)
class CarCheck:
    departure: bool  # d_hat <= d
    departure_exact: bool  # d_hat <= exact plug-out (comparison only)
    departure_floor: bool  # floor(d_hat) <= d (comparison only)
    rate: bool
    cap: bool
    need: bool
    empty: bool
    empty_a: bool
    empty_b: bool
    inside_window: bool  # d_hat < n_service + 1

    @property
    def all_hold(self) -> bool:
        """E'_i."""
        return self.departure and self.rate and self.cap and self.need


def check_car(
    forecast: ForecastCar,
    true_departure: int,
    exact_departure: float,
    rate: float,
    room: float,
    rho: float,
    n_service: int,
    n_recovery: int,
) -> CarCheck:
    """Coverage and emptiness of one car's forecast against its truth."""
    a, b = empty_conditions(forecast, n_service, n_recovery)
    return CarCheck(
        departure=forecast.departure <= true_departure,
        departure_exact=forecast.departure <= exact_departure,
        departure_floor=math.floor(forecast.departure) <= true_departure,
        rate=forecast.rate <= rate,
        cap=forecast.cap <= room,
        need=forecast.need >= (1 - rho) * room,
        empty=forecast_set_empty(forecast, n_service, n_recovery),
        empty_a=a,
        empty_b=b,
        inside_window=forecast.departure < n_service + 1,
    )
