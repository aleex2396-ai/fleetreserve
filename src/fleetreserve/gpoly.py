"""The exact set of what a fleet of lossless cars can draw, and the box test (plan section 4, T1).

A car has a set of hours `hours`, a power `u_t` in `[low, high]` in each of its hours (and 0 in
every other hour), and a total energy `u(hours)` in `[e_low, e_high]`. One-way charging has
`low = 0`; two-way charging has `low < 0` (plan section 9). Such a car's set, and the sum of any
number of them, is a g-polymatroid: it is exactly the set of hourly totals `u` with
`p(A) <= u(A) <= b(A)` for every set of hours `A`, where `u(A)` is the energy drawn in the hours
of `A` and `p`, `b` are the lower and upper set functions below.

Per car, with `n_in = |A & hours|`, `n_out = |hours - A|` and `r = high - low`:

- one-way (`low = 0`):  `p(A) = max(e_low - n_out * r, 0)`,  `b(A) = min(n_in * r, e_high)`.
- two-way: shift the power by `low` (`u = low + v`, `0 <= v <= r`), which moves the energy limits
  to `e_low - low * |hours|` and `e_high - low * |hours|` and gives the one-way formulas on `v`;
  shifting back adds `low * n_in`. Both cases are one formula:

      p(A) = low * n_in + max(e_low - low * |hours| - n_out * r, 0)
      b(A) = low * n_in + min(n_in * r, e_high - low * |hours|)

The fleet's functions are the sums over cars. Hours outside the box's hours count as free: they
appear in `n_out` and so help meet a car's lower energy, which is the projection of the fleet
set onto the box's hours (recovery hours are free).

`box_inside` tests a box `lo <= y <= hi` against the fleet set on the faces `p(A) <= lo(A)` and
`hi(A) <= b(A)`. This is exact: if both hold then `p(A) <= lo(A) <= y(A) <= hi(A) <= b(A)` for
every point `y` of the box, and a vertex of the box that fails one of them is outside the set.

The test enumerates every subset of the box's hours (2^n subsets), because that is the theorem's
own statement; it is refused above `MAX_HOURS` hours. The scalable alternative named in the plan
is one maximum-flow computation for the upper faces and one for the lower faces (one-way) or a
linear programme (two-way); it is not implemented here.

`subset_functions` gives `p(A)` and `b(A)` for every subset `A` of the box's hours at once, as
two arrays indexed by bit mask, from the same per-car formula.

`band_scale` is the same kind of closed form for the band (plan section 4, T1(c)): the box with
centre `c` and widths `w_up`, `w_down`, cut by zero net energy over the box's hours `S`. Over the
band, `y(A) = c(A) + d(A)` with `d(S) = 0`, so the largest `d(A)` at scale `lam` is
`lam * min(w_up(A), w_down(S - A))`, and `y(A) >= p(A)` is the upper face for the complement
`S - A`, whose bound is `c(S) - p(S - A)`. Hence `lam*` is the minimum, over the proper non-empty
subsets `A` of `S`, of
`(min(b(A), c(S) - p(S - A)) - c(A)) / min(w_up(A), w_down(S - A))`.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

MAX_HOURS = 16  # 2^16 subsets; the project's service windows have at most 8 hours
TOLERANCE = 1e-9  # slack on every comparison, in kW or kWh


@dataclass(frozen=True)
class Car:
    """A lossless car: `low <= u_t <= high` in `hours`, `e_low <= sum of u_t <= e_high`."""

    hours: frozenset[int]
    low: float
    high: float
    e_low: float
    e_high: float

    def __post_init__(self):
        if not self.hours:
            raise ValueError("a car needs at least one hour")
        if not self.low <= self.high:
            raise ValueError(f"low power {self.low} is above high power {self.high}")
        if self.e_low > self.e_high:
            raise ValueError(f"lower energy {self.e_low} is above upper energy {self.e_high}")
        if self.e_low > self.high * len(self.hours) or self.e_high < self.low * len(self.hours):
            raise ValueError("the car's energy limits cannot be met within its power limits")


def _lower(car: Car, n_in):
    """`p(A)` of one car from `n_in = |A & hours|` (a whole number or an array of them)."""
    n_out = len(car.hours) - n_in
    shifted_low = car.e_low - car.low * len(car.hours)
    return car.low * n_in + np.maximum(shifted_low - n_out * (car.high - car.low), 0.0)


def _upper(car: Car, n_in):
    """`b(A)` of one car from `n_in = |A & hours|` (a whole number or an array of them)."""
    shifted_high = car.e_high - car.low * len(car.hours)
    return car.low * n_in + np.minimum(n_in * (car.high - car.low), shifted_high)


def car_p(car: Car, subset: frozenset[int]) -> float:
    """Lower set function of one car at the set of hours `subset`."""
    return float(_lower(car, len(car.hours & subset)))


def car_b(car: Car, subset: frozenset[int]) -> float:
    """Upper set function of one car at the set of hours `subset`."""
    return float(_upper(car, len(car.hours & subset)))


def fleet_p(cars: Iterable[Car], subset: Iterable[int]) -> float:
    """The fleet's lower set function: the sum of the cars' `p(A)`."""
    chosen = frozenset(subset)
    return sum(car_p(car, chosen) for car in cars)


def fleet_b(cars: Iterable[Car], subset: Iterable[int]) -> float:
    """The fleet's upper set function: the sum of the cars' `b(A)`."""
    chosen = frozenset(subset)
    return sum(car_b(car, chosen) for car in cars)


def subset_functions(cars: Iterable[Car], hours) -> tuple[np.ndarray, np.ndarray]:
    """The fleet's `p(A)` and `b(A)` for every subset `A` of `hours`, as two arrays.

    Entry `mask` of each array belongs to the subset holding `sorted(hours)[k]` for every bit `k`
    set in `mask`; entry 0 is the empty set. Hours outside `hours` count as free, as in
    `fleet_p` and `fleet_b`. Refused above `MAX_HOURS` hours.
    """
    box_hours = sorted(set(hours))
    if len(box_hours) > MAX_HOURS:
        raise ValueError(
            f"{len(box_hours)} hours is above the limit of {MAX_HOURS} for subset enumeration"
        )
    masks = np.arange(2 ** len(box_hours), dtype=np.int64)
    lower = np.zeros(masks.size)
    upper = np.zeros(masks.size)
    for car in cars:
        car_bits = sum(1 << k for k, t in enumerate(box_hours) if t in car.hours)
        n_in = np.bitwise_count(masks & car_bits).astype(np.int64)
        lower += _lower(car, n_in)
        upper += _upper(car, n_in)
    return lower, upper


def box_inside(lo: Sequence[float], hi: Sequence[float], cars: Sequence[Car], hours) -> bool:
    """True when the box `lo[t] <= y_t <= hi[t]`, `t` in `hours`, lies inside the fleet set.

    `lo` and `hi` are indexed by hour (a list or a dict). Checks `p(A) <= lo(A)` and
    `hi(A) <= b(A)` for every subset `A` of `hours`, to within `TOLERANCE`.
    """
    box_hours = sorted(set(hours))
    if len(box_hours) > MAX_HOURS:
        raise ValueError(
            f"{len(box_hours)} hours is above the limit of {MAX_HOURS} for subset enumeration"
        )
    if any(lo[t] > hi[t] + TOLERANCE for t in box_hours):
        raise ValueError("the box has lo above hi in some hour")
    for size in range(len(box_hours) + 1):
        for chosen in combinations(box_hours, size):
            subset = frozenset(chosen)
            if sum(lo[t] for t in chosen) < fleet_p(cars, subset) - TOLERANCE:
                return False
            if sum(hi[t] for t in chosen) > fleet_b(cars, subset) + TOLERANCE:
                return False
    return True


def band_scale(centre, w_up, w_down, cars: Sequence[Car], hours) -> float:
    """Largest scale `lam` with the whole band inside the fleet set (plan section 4, T1(c)).

    The band is `{centre + d}` with `-lam * w_down[t] <= d[t] <= lam * w_up[t]` in every hour `t`
    of `hours` and `sum(d) = 0`. `centre`, `w_up` and `w_down` are indexed by hour. A subset with
    denominator zero (zero widths) puts no limit on `lam` and is omitted; if no subset limits it,
    the result is infinity. The centre must lie inside the fleet set, otherwise `ValueError`.
    Enumerates every subset of `hours`, and is refused above `MAX_HOURS` hours.
    """
    band_hours = sorted(set(hours))
    if len(band_hours) > MAX_HOURS:
        raise ValueError(
            f"{len(band_hours)} hours is above the limit of {MAX_HOURS} for subset enumeration"
        )
    if any(w_up[t] < 0 or w_down[t] < 0 for t in band_hours):
        raise ValueError("a band width is negative")
    total = sum(centre[t] for t in band_hours)
    everything = frozenset(band_hours)
    if not (
        fleet_p(cars, everything) - TOLERANCE <= total <= fleet_b(cars, everything) + TOLERANCE
    ):
        raise ValueError("the centre is outside the fleet set")
    scale = float("inf")
    for size in range(1, len(band_hours)):
        for chosen in combinations(band_hours, size):
            subset = frozenset(chosen)
            rest = everything - subset
            centre_in = sum(centre[t] for t in chosen)
            room = min(fleet_b(cars, subset), total - fleet_p(cars, rest)) - centre_in
            if room < -TOLERANCE:
                raise ValueError("the centre is outside the fleet set")
            width = min(sum(w_up[t] for t in chosen), sum(w_down[t] for t in rest))
            if width > 0:
                scale = min(scale, room / width)
    return scale
