"""The exact hindsight reference of one day on its true sets (plan sections 2, 4 (T1) and 9).

The fleet's set seen on the service hours `S`, with hour 0 and the recovery hours free, is
described by two set functions: `p(A)`, the least energy the fleet can draw in the hours `A`, and
`b(A)`, the most (`A` a subset of `S`). Each car's set is a g-polymatroid and so is their sum
(Frank 2011, Theorem 14.2.15); the projection onto `S` keeps only the faces
`p(A) <= u(A) <= b(A)` (plan section 4, T1; checked by the tests on made-up fleets). Everything here
is computed from the two arrays of a `SetFunctions`: entry `mask` belongs to the subset of `S`
holding service hour `k + 1` for each bit `k` set in `mask`.

- One-way cars (`gpoly.Car`): `p` and `b` are the closed forms of `gpoly`.
- Cars with stored-energy limits after every hour (`day.StoredEnergyCar`, the two-way true set):
  `b_i(A)` is found by a forward pass that, hour by hour, draws as much as possible in the hours of
  `A` and as little as possible in the other hours, without leaving the stored-energy levels from
  which the car can still end within its limits (found by a backward pass). It is exact: from a
  lower stored energy the car can follow any plan from a higher one while drawing at least as much
  every hour, and from a higher one it can follow a plan from a lower one while losing at most the
  difference; so the best value ahead never rises with the stored energy, and falls by at most
  one unit per unit. `p_i(A)` is the same pass with the roles of `A` and the other hours swapped.

The values computed:

- `hourly_ceiling`: the hourly product's V_hind = b(S) - p(S) (plan section 2).
- `box_inside`: T1(a), the box `[lo, hi]` is inside iff `p(A) <= lo(A)` and `hi(A) <= b(A)`.
- `box_scale`: T1(b), the largest `lam` with `[c - lam w_down, c + lam w_up]` inside:
  `min_A min{(b(A) - c(A)) / w_up(A), (c(A) - p(A)) / w_down(A)}`, terms with a zero width left
  out. A centre outside the set raises `ValueError`; when every width is zero the scale is infinite.
- `block_ceiling`: the block product's ceiling, the largest half-width `lam` of a box
  `m +- lam` in every service hour with the centre `m` free, from one linear programme over
  `(lam, m)` with the rows `m(A) + lam |A| <= b(A)` and `m(A) - lam |A| >= p(A)`; the value is
  `2 lam |S|`. The programme runs through `lp.solve_lp` with the box time limit; any outcome but
  optimal (a timeout included) is returned as a missing result, never as zero.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fleetreserve import gpoly
from fleetreserve.day import MAX_SERVICE_HOURS, Day, StoredEnergyCar
from fleetreserve.lp import BOX_SECONDS, Outcome, solve_lp

TOLERANCE = gpoly.TOLERANCE  # slack on every comparison, in kW or kWh


@dataclass(frozen=True)
class SetFunctions:
    """`lower[mask] = p(A)` and `upper[mask] = b(A)` for every subset `A` of the service hours."""

    n_service: int
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        if not 1 <= self.n_service <= MAX_SERVICE_HOURS:
            raise ValueError(f"n_service must be 1 to {MAX_SERVICE_HOURS}, got {self.n_service}")
        size = 2**self.n_service
        for name in ("lower", "upper"):
            values = np.array(getattr(self, name), dtype=float)
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must hold {size} finite values")
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    @property
    def full(self) -> int:
        """The mask of all service hours."""
        return 2**self.n_service - 1


def subset_mask(hours, n_service: int) -> int:
    """The mask of a set of service hours (each from 1 to `n_service`)."""
    mask = 0
    for t in hours:
        if not 1 <= t <= n_service:
            raise ValueError(f"hour {t} is not a service hour 1 to {n_service}")
        mask |= 1 << (t - 1)
    return mask


def _bits(n_service: int) -> np.ndarray:
    """`bits[mask, k]` is 1.0 when service hour `k + 1` is in the subset `mask`."""
    masks = np.arange(2**n_service)
    return ((masks[:, None] >> np.arange(n_service)) & 1).astype(float)


def _stored_energy_extremes(cars: Sequence[StoredEnergyCar], n_service: int):
    """The cars' summed `p(A)` and `b(A)` for every mask, by the forward and backward passes."""
    n = np.array([car.n_hours for car in cars])
    low = np.array([car.low for car in cars], dtype=float)
    high = np.array([car.high for car in cars], dtype=float)
    stored_low = np.array([car.stored_low for car in cars], dtype=float)
    stored_high = np.array([car.stored_high for car in cars], dtype=float)
    end_low = np.maximum(stored_low, np.array([car.need for car in cars], dtype=float))
    horizon = int(n.max())
    present = np.arange(horizon)[None, :] < n[:, None]  # [car, hour]
    power_low = np.where(present, low[:, None], 0.0)
    power_high = np.where(present, high[:, None], 0.0)
    # Levels allowed after each hour; from the car's last hour on, the end limits apply.
    last = np.arange(horizon)[None, :] >= (n - 1)[:, None]
    level_low = np.where(last, end_low[:, None], stored_low[:, None])
    level_high = np.broadcast_to(stored_high[:, None], level_low.shape)
    # Backward pass: after hour t the car must stay in [reach_low, reach_high] to end in limits.
    reach_low = level_low.copy()
    reach_high = level_high.copy()
    for t in range(horizon - 1, 0, -1):
        reach_low[:, t - 1] = np.maximum(level_low[:, t - 1], reach_low[:, t] - power_high[:, t])
        reach_high[:, t - 1] = np.minimum(level_high[:, t - 1], reach_high[:, t] - power_low[:, t])
    start_ok = (reach_low[:, 0] - power_high[:, 0] <= TOLERANCE) & (
        reach_high[:, 0] - power_low[:, 0] >= -TOLERANCE
    )
    if not np.all(start_ok) or np.any(reach_low > reach_high + TOLERANCE):
        raise ValueError("a stored-energy car has an empty set")

    masks = np.arange(2**n_service)
    totals = []
    for most_in_subset in (False, True):  # False gives p(A), True gives b(A)
        level = np.zeros((len(cars), masks.size))
        drawn = np.zeros((len(cars), masks.size))
        for t in range(horizon):
            least = np.maximum(power_low[:, t, None], reach_low[:, t, None] - level)
            most = np.minimum(power_high[:, t, None], reach_high[:, t, None] - level)
            if 1 <= t <= n_service:
                in_subset = ((masks >> (t - 1)) & 1).astype(bool)[None, :]
                power = np.where(in_subset == most_in_subset, most, least)
                drawn += np.where(in_subset, power, 0.0)
            else:
                power = least if most_in_subset else most
            level += power
        totals.append(drawn.sum(axis=0))
    return totals[0], totals[1]


def subset_functions(cars: Sequence, n_service: int) -> SetFunctions:
    """`p(A)` and `b(A)` of a fleet for every subset of the service hours 1 to `n_service`.

    `cars` holds `gpoly.Car` (one-way, or two-way with total energy limits only) and
    `day.StoredEnergyCar` in any mix; the hours outside the service hours are free.
    """
    if not 1 <= n_service <= MAX_SERVICE_HOURS:
        raise ValueError(f"n_service must be 1 to {MAX_SERVICE_HOURS}, got {n_service}")
    plain = [car for car in cars if isinstance(car, gpoly.Car)]
    stored = [car for car in cars if isinstance(car, StoredEnergyCar)]
    if len(plain) + len(stored) != len(cars):
        raise ValueError("every car must be a gpoly.Car or a StoredEnergyCar")
    lower, upper = gpoly.subset_functions(plain, range(1, n_service + 1))
    if stored:
        stored_lower, stored_upper = _stored_energy_extremes(stored, n_service)
        lower = lower + stored_lower
        upper = upper + stored_upper
    return SetFunctions(n_service, lower, upper)


def day_functions(day: Day, budget=None) -> SetFunctions:
    """The day's set functions on its true sets: one-way, or two-way at `budget` kWh."""
    cars = day.one_way_cars() if budget is None else day.two_way_cars(budget)
    return subset_functions(cars, day.n_service)


def hourly_ceiling(functions: SetFunctions) -> float:
    """The hourly product's ceiling V_hind = b(S) - p(S), in kWh."""
    return float(functions.upper[functions.full] - functions.lower[functions.full])


def _hour_vector(values, n_service: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (n_service,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must hold {n_service} finite values, one per service hour")
    return vector


def box_inside(functions: SetFunctions, lo, hi) -> bool:
    """T1(a): True when the box `lo[k] <= y <= hi[k]` in service hour `k + 1` is inside the set."""
    n = functions.n_service
    lo_vector = _hour_vector(lo, n, "lo")
    hi_vector = _hour_vector(hi, n, "hi")
    if np.any(lo_vector > hi_vector + TOLERANCE):
        raise ValueError("the box has lo above hi in some hour")
    bits = _bits(n)
    return bool(
        np.all(bits @ lo_vector >= functions.lower - TOLERANCE)
        and np.all(bits @ hi_vector <= functions.upper + TOLERANCE)
    )


def box_scale(functions: SetFunctions, centre, w_up, w_down) -> float:
    """T1(b): the largest `lam >= 0` with the box `centre - lam w_down .. centre + lam w_up` inside.

    Values are per service hour, in order. Raises `ValueError` for a negative width or a centre
    outside the set; returns infinity when no subset limits the scale (every width zero). A centre
    on the boundary gives 0.
    """
    n = functions.n_service
    c = _hour_vector(centre, n, "centre")
    up = _hour_vector(w_up, n, "w_up")
    down = _hour_vector(w_down, n, "w_down")
    if np.any(up < 0) or np.any(down < 0):
        raise ValueError("a width is negative")
    bits = _bits(n)
    c_sum, up_sum, down_sum = bits @ c, bits @ up, bits @ down
    room_up = functions.upper - c_sum
    room_down = c_sum - functions.lower
    if np.any(room_up < -TOLERANCE) or np.any(room_down < -TOLERANCE):
        raise ValueError("the centre is outside the fleet set")
    ratios = np.concatenate(
        [room_up[up_sum > 0] / up_sum[up_sum > 0]]
        + [room_down[down_sum > 0] / down_sum[down_sum > 0]]
    )
    if ratios.size == 0:
        return float("inf")
    return max(float(ratios.min()), 0.0)


@dataclass(frozen=True)
class BlockCeiling:
    """The block product's ceiling. `half_width`, `centre` and `value` are None unless `outcome`
    is "optimal"; a timeout is a missing result."""

    outcome: Outcome
    half_width: float | None
    centre: tuple[float, ...] | None
    value: float | None
    seconds: float


def block_ceiling(functions: SetFunctions, time_limit: float = BOX_SECONDS) -> BlockCeiling:
    """The largest `lam` over a free centre `m` with `m +- lam` inside, and the value `2 lam |S|`.

    Variables `(lam, m_1, ..., m_n)`; one pair of rows per non-empty subset `A` of `S`.
    """
    n = functions.n_service
    bits = _bits(n)[1:]
    size = bits.sum(axis=1, keepdims=True)
    a_ub = np.vstack([np.hstack([size, bits]), np.hstack([size, -bits])])
    b_ub = np.concatenate([functions.upper[1:], -functions.lower[1:]])
    cost = np.zeros(n + 1)
    cost[0] = -1.0
    result = solve_lp(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] + [(None, None)] * n,
        time_limit=time_limit,
        method="ipm",
    )
    if result.outcome != "optimal":
        return BlockCeiling(result.outcome, None, None, None, result.seconds)
    lam = float(result.x[0])
    centre = tuple(float(v) for v in result.x[1:])
    return BlockCeiling("optimal", lam, centre, 2.0 * lam * n, result.seconds)
