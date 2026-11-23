"""The base set of LP (24) and LP (27) of Al Taha et al. (2025), in the form the earlier analysis
used (plan sections 1, 10, 12 and 13 (WP1: earlier base-set definitions copied, O6)).

A base set is

    U0 = {u : 0 <= u_t <= power_t for every hour t,  sum_t u_t <= energy}

over the hours of the rows (`carsets`): hour 0 (the decision hour) to `n_service`, one-hour
steps. Al Taha et al. (2025), equation (11), take `U0 = {u : H u <= h0}` with `h0` the average of
the cars' right-hand sides `h_i`. The earlier analysis wrote it, for cars whose sets are a power
limit and one energy ceiling, as the average of the cars' per-step upper power limits and the
average of their energy ceilings; this module copies that form.

A car's limits, read from its rows (`car_limits`):

- energy: the bound of the car's row on the energy of all hours 0 to `n_service`, that is the
  `"stored-high"` row of the last hour (true one-way and two-way cars: the room `e`; forecast car:
  `e_cap`) or the `"energy-high"` row (`gpoly.Car`). When the car has several, the smallest; when
  it has none, the sum of its upper power limits.
- per-hour upper limit: `min(high_t, energy)`, `high_t` the bound of the car's `"power-high"` row
  of hour `t` (true car: `r` while present, 0 after its departure; forecast car: `r_hat f_t`).
  The earlier analysis took `min(c_i, e_i / delta)` for the same reason: no hour can take more
  than the whole energy, and without it a car with no energy left that stays late would give the
  base set power in hours no car can use.

Lower limits do not enter: the base set's power is at least 0, as in the earlier analysis, also
for two-way cars (power down to `-r`) and for toy cars with a positive lower power limit; nor does
a car's needed energy (the `"need"` row) or its stored-energy limits before the last hour.

The two constructions the plan names:

- `day_average(rows)`: the average over the day's own fleet (plan section 12, "day-average base
  set"; the earlier analysis averaged the fleet it solved for).
- `template(fleets)`: the same average over every car of several fleets pooled (plan section 12,
  "template base set"; the earlier analysis pooled the true cars of every training morning). The
  pooling over training days is the caller's: the fleets are passed in.

Readings (where neither the plan nor the earlier analysis decides):

1. Hours. The earlier analysis's steps started at the decision time and ran to the end of a fixed
   horizon, with no service window and no recovery hours. Here the base set lives on hours 0 to
   `n_service`, the hours of the rows: the decision hour is a base-set hour like the others (the
   earlier analysis's first step also started at the decision time), and the recovery hours are
   not base-set hours, because they are eliminated in the rows (`carsets`).
2. The needed energy and the power lower limits are left out of the base set (above), as in the
   earlier analysis, whose sets had neither; the paper's `h0` would average them too.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fleetreserve.carsets import FleetRows

TRIVIAL_ENERGY = 1e-7  # kWh: a base set with less energy holds only the zero profile (as earlier)


@dataclass(frozen=True)
class BaseSet:
    """`U0 = {u : 0 <= u <= power, sum(u) <= energy}` on hours 0 to `len(power) - 1`."""

    power: np.ndarray  # (n_service + 1,) kW
    energy: float  # kWh

    def __post_init__(self):
        power = np.array(self.power, dtype=float)
        energy = float(self.energy)
        if power.ndim != 1 or power.size < 2:
            raise ValueError("power must hold one limit per hour, hour 0 and at least one more")
        if not (np.all(np.isfinite(power)) and np.isfinite(energy)):
            raise ValueError("the base set's limits must be finite")
        if np.any(power < 0) or energy < 0:
            raise ValueError("the base set's limits must be non-negative")
        power.flags.writeable = False
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "energy", energy)

    @property
    def n_service(self) -> int:
        return int(self.power.size - 1)

    @property
    def columns(self) -> np.ndarray:
        """The hours in which the base set can draw power (`power_t > 0`)."""
        return np.flatnonzero(self.power > 0)

    def is_trivial(self) -> bool:
        """True when the base set holds only the zero profile (as the earlier analysis: energy at
        most `TRIVIAL_ENERGY`, or no hour with power)."""
        return self.energy <= TRIVIAL_ENERGY or self.columns.size == 0

    def contains(self, u, tolerance=1e-9) -> np.ndarray:
        """Whether each profile `u` (shape `(hours,)` or `(points, hours)`) lies in the set."""
        u = np.asarray(u, dtype=float)
        inside = np.all(u >= -tolerance, axis=-1) & np.all(u <= self.power + tolerance, axis=-1)
        return inside & (u.sum(axis=-1) <= self.energy + tolerance)


def car_limits(rows: FleetRows) -> tuple[np.ndarray, np.ndarray]:
    """Each car's per-hour upper limits `(cars, n_service + 1)` and energy `(cars,)`, read from
    its rows as the module text states."""
    hours = rows.n_service + 1
    kinds = np.asarray(rows.kinds)
    high = np.full((rows.n_cars, hours), np.nan)
    power_rows = np.flatnonzero(kinds == "power-high")
    hour_of = np.argmax(rows.coefficients[power_rows] != 0, axis=1)
    scale = rows.coefficients[power_rows, hour_of]
    high[rows.car[power_rows], hour_of] = rows.bounds[power_rows] / scale
    if np.any(np.isnan(high)):
        raise ValueError("every car needs a power-high row for every hour")
    if np.any(high < 0):
        raise ValueError("a car's upper power limit is negative")

    whole = np.all(rows.coefficients == 1.0, axis=1)
    total_rows = np.flatnonzero(
        ((kinds == "stored-high") & whole) | ((kinds == "energy-high") & whole)
    )
    energy = high.sum(axis=1)
    np.minimum.at(energy, rows.car[total_rows], rows.bounds[total_rows])
    if np.any(energy < 0):
        raise ValueError("a car's energy limit is negative")
    return np.minimum(high, energy[:, None]), energy


def average(upper, energy) -> BaseSet:
    """The base set of the average of the given cars' upper limits and energies."""
    upper = np.asarray(upper, dtype=float)
    energy = np.asarray(energy, dtype=float)
    if upper.ndim != 2 or upper.shape[0] == 0 or energy.shape != (upper.shape[0],):
        raise ValueError("one row of upper limits and one energy per car, at least one car")
    return BaseSet(upper.mean(axis=0), float(energy.mean()))


def day_average(rows: FleetRows) -> BaseSet:
    """The day-average base set: the average over the fleet of `rows`."""
    return average(*car_limits(rows))


def template(fleets: Sequence[FleetRows]) -> BaseSet:
    """The template base set: the average over every car of the fleets pooled (each car counts
    once, so a larger fleet weighs more than a smaller one)."""
    if not fleets:
        raise ValueError("the template needs at least one fleet")
    n = fleets[0].n_service
    if any(f.n_service != n for f in fleets):
        raise ValueError("every pooled fleet must have the same hours")
    limits = [car_limits(f) for f in fleets]
    return average(np.vstack([u for u, _ in limits]), np.concatenate([e for _, e in limits]))
