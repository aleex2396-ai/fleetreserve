"""Driver-risk rows: the forecast set with a price on an early departure (plan section 2).

The forecast set of a car (`carsets.forecast_rows`) is built at the presence bound `D`, the level-
kappa departure bound `d_hat`. Driver-risk rows add to it, for a dial `epsilon >= 0` (the plan's
values are `DIALS`), a cap on the car's expected shortfall of needed energy over the lower levels
`kappa_1 < ... < kappa_K <= kappa` of plan section 2:

- `D_k` the level-`kappa_k` departure bound, made a running maximum over `k` and at most `D`; a
  level the forecaster does not give is `D` (`departure_scenarios`);
- `pi_k = kappa_k - kappa_{k-1}` (`kappa_0 = 0`) the mass of level `k` (`level_masses`);
- `E_k(u) = sum over the window hours t of u_t * min{1, f_t(D_k) / f_t(D)}` plus
  `rate * (the recovery hours' shares before D_k)`: the energy the car holds by `D_k` under the
  plan `u` (an hour with `f_t(D) = 0` has `u_t = 0`, and its weight is taken as 1). The window is
  hours 0 to `n_service`;
- `need_k = min{need, the largest E_k over the set at D}`.

The rows say `sum_k pi_k max{need_k - E_k(u), 0} <= epsilon * need`. The left side is convex and
piecewise linear in `u`, so it is written as one linear row per non-empty set `A` of levels,
`sum_{k in A} pi_k (need_k - E_k(u)) <= epsilon * need`: the sum of the positive parts is at most
the bound exactly when every such partial sum is. Before that, a level with `need_k - supply_k <= 0`
(`supply_k` the recovery term of `E_k`) has no shortfall anywhere on the set, since the window
energy is not negative, and is left out; levels with the same bound `D_k` are merged, their masses
added. So a car has at most `2^K - 1` rows, 63 for the plan's six levels. Every row has the kind
"need".

The rows are a model of the risk of an early departure, not a guarantee; plan section 6 calibrates
the realised losses. They are linear in the car's powers, so the certificate of plan section 5
applies to the set with them unchanged. With a car whose scenarios are all `D` (the oracle
departure) every `E_k` is the window energy plus the supply at `D`, which the need row at `D`
already bounds below by `need >= need_k`: the rows then hold on the whole set. A car without
scenarios is refused here; a day without the dial uses `carsets.day_forecast_rows`.

Levels off the plan's grid (`WHAT_IF_LEVELS`) exist only for the what-if of
`dial_choice --what-if-kappa`; there the rows run over the plan's levels and the what-if levels at
or below kappa (`levels_up_to`; `dial_choice` reading W1).
"""

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from fleetreserve import carsets
from fleetreserve.carsets import CarRows, FleetRows, stack
from fleetreserve.day import Day, ForecastCar
from fleetreserve.forecast_empirical import KAPPAS, WHAT_IF_KAPPAS

DIALS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1)  # plan section 2
LEVELS = KAPPAS  # plan section 2
WHAT_IF_LEVELS = WHAT_IF_KAPPAS  # off the plan's grid: `dial_choice --what-if-kappa` only
MASS_DIGITS = 12  # masses are differences of levels, rounded to this many decimals
NO_SHORTFALL = 1e-12  # a level with need_k - supply_k at most this has no shortfall


def levels_up_to(kappa: float) -> tuple[float, ...]:
    """The plan's levels at or below `kappa`, one of them; for a level of `WHAT_IF_LEVELS`, the
    plan's levels and the what-if levels at or below it (`dial_choice`, reading W1)."""
    if kappa in LEVELS:
        return tuple(level for level in LEVELS if level <= kappa)
    if kappa in WHAT_IF_LEVELS:
        return tuple(level for level in (*LEVELS, *WHAT_IF_LEVELS) if level <= kappa)
    raise ValueError(
        f"kappa must be one of {list(LEVELS)} or of the what-if levels {list(WHAT_IF_LEVELS)}, "
        f"got {kappa!r}"
    )


def matrix_levels(configurations) -> tuple[float, ...]:
    """The levels a model built from a run matrix's `configurations` is fitted and forecasts at:
    every configuration's own kappa and, for each configuration with a driver dial, every level up
    to its kappa (`levels_up_to`), which its driver-risk rows read (`gate_day`). A matrix that
    lists every level up to each dialled kappa gives its own kappas. A configuration without the
    field `driver_risk` has no dial."""
    found = set()
    for configuration in configurations:
        found.add(float(configuration.kappa))
        if getattr(configuration, "driver_risk", None) is not None:
            found.update(levels_up_to(configuration.kappa))
    return tuple(sorted(found))


def level_masses(levels: Sequence[float]) -> tuple[float, ...]:
    """`pi_k = kappa_k - kappa_{k-1}` with `kappa_0 = 0`."""
    return tuple(
        round(float(b - a), MASS_DIGITS) for a, b in zip((0.0, *levels[:-1]), levels, strict=True)
    )


def departure_scenarios(
    presence: float, level_departures: Sequence[float | None], masses: Sequence[float]
) -> tuple[tuple[float, float], ...]:
    """The pairs `(D_k, pi_k)`: the level departures (None: not given, read as `presence`) made a
    running maximum and capped at `presence`."""
    if len(level_departures) != len(masses):
        raise ValueError("one mass per level departure")
    out, top = [], -math.inf
    for departure, mass in zip(level_departures, masses, strict=True):
        top = max(top, presence if departure is None else departure)
        out.append((min(top, presence), mass))
    return tuple(out)


def _at(car: ForecastCar, departure: float) -> ForecastCar:
    """The car's limits with the departure bound `departure` (and no scenarios)."""
    return replace(car, departure=departure, scenarios=())


def _weights(car: ForecastCar, scenario: ForecastCar, n_service: int) -> np.ndarray:
    """`min{1, f_t(D_k) / f_t(D)}` for the window hours; 1 where `f_t(D) = 0`."""
    out = []
    for t in range(n_service + 1):
        share = car.share(t)
        out.append(1.0 if share <= 0 else min(1.0, scenario.share(t) / share))
    return np.array(out)


def _reach(car: ForecastCar, weights: np.ndarray, supply: float, n_service: int) -> float:
    """The largest `E_k` over the set at `D`: hours by weight, largest first, each up to its power
    limit, the hours' total at most the cap; plus the recovery supply."""
    left = car.cap
    total = 0.0
    for t in sorted(range(n_service + 1), key=lambda t: -weights[t]):
        take = min(car.power_cap(t), max(left, 0.0))
        total += weights[t] * take
        left -= take
    return total + supply


def shortfall_rows(
    car: ForecastCar, n_service: int, n_recovery: int, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    """The driver-risk rows of one car as `(coefficients, bounds)`, one row per non-empty set of
    levels (module text); empty arrays when no level has a shortfall."""
    if isinstance(epsilon, bool) or not isinstance(epsilon, int | float) or not epsilon >= 0:
        raise ValueError(f"epsilon must be a number from 0, got {epsilon!r}")
    if not car.scenarios:
        raise ValueError("the car has no departure scenarios")
    merged: dict = {}
    for departure, mass in car.scenarios:
        scenario = _at(car, departure)
        supply = scenario.recovery_supply(n_service, n_recovery)
        weights = _weights(car, scenario, n_service)
        need_k = min(car.need, _reach(car, weights, supply, n_service))
        if need_k - supply <= NO_SHORTFALL:
            continue
        if departure in merged:
            merged[departure] = (merged[departure][0] + mass, weights, need_k - supply)
        else:
            merged[departure] = (mass, weights, need_k - supply)
    levels = list(merged.values())
    coefficients, bounds = [], []
    for subset in range(1, 2 ** len(levels)):
        chosen = [levels[k] for k in range(len(levels)) if subset >> k & 1]
        coefficients.append(-sum(mass * weights for mass, weights, _ in chosen))
        bounds.append(epsilon * car.need - sum(mass * gap for mass, _, gap in chosen))
    return np.array(coefficients).reshape(-1, n_service + 1), np.array(bounds)


def car_rows(car: ForecastCar, n_service: int, n_recovery: int, epsilon: float) -> CarRows:
    """The forecast set of `car` (`carsets.forecast_rows`) with its driver-risk rows."""
    base = carsets.forecast_rows(car, n_service, n_recovery)
    coefficients, bounds = shortfall_rows(car, n_service, n_recovery, epsilon)
    if not bounds.size:
        return base
    return CarRows(
        n_service,
        np.vstack([base.coefficients, coefficients]),
        np.concatenate([base.bounds, bounds]),
        base.kinds + ("need",) * bounds.size,
    )


def day_rows(day: Day, epsilon: float) -> FleetRows:
    """`carsets.day_forecast_rows` with the driver-risk rows of every car; the day must carry
    forecasts, each with its scenarios."""
    if day.forecasts is None:
        raise ValueError("the day has no forecasts")
    return stack([car_rows(f, day.n_service, day.n_recovery, epsilon) for f in day.forecasts])
