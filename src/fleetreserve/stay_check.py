"""The check that cars staying to their bound have no shortfall (plan section 6, tier B; T2 (c1)).

T2 (c1): on the events E'_i = {d_hat_i <= d_i, r_hat_i <= r_i, e_cap_i <= e_i, e_need_i >= (1 - rho)
e_i} a car gets at least its protected energy on every path, whatever the other cars do. Tier B
(deadline contract) takes its scale from the slot loss alone, so the driver loss is not held down
by the scale, and this check is what remains of the driver's side of the contract: cars on E'_i
must end whole.

Which cars are checked: those on E'_i, as the day holds them (`cars_in_check`): the forecast
departure (the presence bound the plan was built on, as the forecast set uses it, fractional or
rounded down) at or before the true departure hour, and the other three bounds at their true
side, each within `tolerance`. With known energy bounds the forecast rate, cap and need are the
true ones, so only the departure can fail and the cars checked are the cars that stay to their
bound. With empirical energy bounds the check covers a car only when its other bounds hold too,
because the guarantee is conditional on all of E'_i; a car whose rate or energy bound is wrong
carries no guarantee and is not counted either way. The check is not computed for cars outside
E'_i.

A car in the check is short on a path when its energy at departure, after recovery (`replay`), is
below its protected energy by more than `tolerance`. As for T5, a break stops nothing: it is
counted and reported. `tolerance` is the matrix's `t5_tolerance_kwh`, the one tolerance the gate
run uses to call two energies equal.
"""

from dataclasses import dataclass

import numpy as np

from fleetreserve.day import Day


@dataclass(frozen=True)
class StayCheck:
    """The check on one day, one rung, all its audited paths.

    `cars`: cars in the check; `car_paths`: those cars on every path; `short_cars`: cars short on
    at least one path; `short_car_paths`: car-paths that are short; `worst_kwh`: the largest
    shortfall."""

    cars: int
    car_paths: int
    short_cars: int
    short_car_paths: int
    worst_kwh: float


NONE = StayCheck(0, 0, 0, 0, 0.0)


def add(first: StayCheck, second: StayCheck) -> StayCheck:
    """The check over two days: counts add, the worst shortfall is the larger."""
    return StayCheck(
        first.cars + second.cars,
        first.car_paths + second.car_paths,
        first.short_cars + second.short_cars,
        first.short_car_paths + second.short_car_paths,
        max(first.worst_kwh, second.worst_kwh),
    )


def cars_in_check(day: Day, tolerance: float) -> tuple[int, ...]:
    """The cars of `day` (with forecasts) on the events E'_i of T2 (c1), as indices."""
    if day.forecasts is None:
        raise ValueError("the day has no forecasts")
    found = []
    for i, (car, forecast) in enumerate(zip(day.cars, day.forecasts, strict=True)):
        if (
            forecast.departure <= car.departure
            and forecast.rate <= car.rate + tolerance
            and forecast.cap <= car.room + tolerance
            and forecast.need >= car.need(day.rho) - tolerance
        ):
            found.append(i)
    return tuple(found)


def check(day: Day, finals, tolerance: float) -> StayCheck:
    """The check on `day` from each car's energy at departure on each path, `finals` of shape
    (paths, cars)."""
    finals = np.asarray(finals, dtype=float)
    cars = cars_in_check(day, tolerance)
    if not cars:
        return NONE
    need = np.array([day.cars[i].need(day.rho) for i in cars])
    short = need[None, :] - finals[:, list(cars)]
    is_short = short > tolerance
    worst = float(short[is_short].max()) if is_short.any() else 0.0
    return StayCheck(
        len(cars),
        len(cars) * finals.shape[0],
        int(is_short.any(axis=0).sum()),
        int(is_short.sum()),
        worst,
    )
