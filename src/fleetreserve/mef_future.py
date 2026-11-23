"""Whether the believed fleet can still follow some of the rest of a box after one hour's split,
in closed form (the check of `mef_controller`, plan section 10: "Maximum-entropy feedback on
forecast departures (Li et al.)").

`Future` holds, after the powers of hour `t` are planned, the cars still counted as plugged in
and the box's remaining service hours `t + 1` to `n` (`k` hours, possibly none):

- `high[i, j]`: the most car `i` is believed able to take in the `j`-th remaining hour;
- `least[i]`: the energy car `i` must still draw in the remaining hours for its believed need
  (it may be 0 or below, when the need is met);
- `room[i]`: the energy car `i` may still draw in the remaining hours under its believed cap;
- `lo[j]`, `hi[j]`: the box's lowest and highest request in the `j`-th remaining hour
  (`lo <= hi`).

The believed remaining set of car `i` is `{v : 0 <= v_j <= high[i, j], least[i] <= sum_j v_j <=
room[i]}`. The future is met when some request vector of the box `lo <= y <= hi` can be followed
exactly by one point of each car's set, i.e. when the sum of the cars' sets meets the box (has a
point in common with it; not that it holds the whole box). With no remaining hour that is the
cars' conditions alone. `meets(future)` tests it without a programme:

- `least[i] <= room[i]` for every car (`within_cars`), and
- for every set `A` of remaining hours (the empty set included; `2^k` sets),
  `p(A) <= hi(A)` (`enough`) and `lo(A) <= b(A)` (`roomy`), where `x(A)` is the sum over the
  hours of `A`, `p(A) = sum_i max(least[i] - high[i](not A), 0)` and
  `b(A) = sum_i min(high[i](A), room[i])`.

Why this is exact. When `least[i] <= room[i]`, `least[i] <= high[i](all)` and `room[i] >= 0`,
car `i`'s set is non-empty and is a g-polymatroid whose lower and upper set functions are the
`i`-th terms of `p` and `b` (the formulas of `gpoly`, with an upper power limit per hour); a sum
of g-polymatroids is the g-polymatroid of the sums of the functions; a box is the g-polymatroid
of `lo(A)` and `hi(A)`; and two non-empty g-polymatroids meet exactly when each one's lower
function is at most the other's upper function on every set (the intersection theorem for
g-polymatroids of Frank and Tardos). The other two car conditions are the empty set's
inequalities: `p(empty) = sum_i max(least[i] - high[i](all), 0) <= 0` and
`b(empty) = sum_i min(0, room[i]) >= 0`. The tests check the closed form against the
feasibility of a linear programme of the same sets (`lp.solve_lp`, 30-second limit) on 1,200
seeded made-up futures.

Every comparison allows `TOLERANCE`. `enough` can only change from false to true, and `roomy`
only from true to false, as cars draw more (every `least[i]` and `room[i]` falls by what car `i`
draws; no `p(A)` can rise as the `least[i]` fall, and no `b(A)` as the `room[i]` fall, while the
box's sums stay as they are), which `mef_controller` uses to find the totals that keep the
future met.

`MAX_HOURS` (8, the plan's longest service window) bounds `k`.
"""

from dataclasses import dataclass

import numpy as np

MAX_HOURS = 8
TOLERANCE = 1e-9  # kWh, on every comparison of `meets`, `enough`, `roomy` and `within_cars`


@dataclass(frozen=True)
class Future:
    """The believed remaining cars and the box's remaining hours (module text)."""

    high: np.ndarray  # (cars, k)
    least: np.ndarray  # (cars,)
    room: np.ndarray  # (cars,)
    lo: np.ndarray  # (k,)
    hi: np.ndarray  # (k,)

    def __post_init__(self):
        high = np.array(self.high, dtype=float)
        least = np.array(self.least, dtype=float).reshape(-1)
        room = np.array(self.room, dtype=float).reshape(-1)
        lo = np.array(self.lo, dtype=float).reshape(-1)
        hi = np.array(self.hi, dtype=float).reshape(-1)
        k = lo.size
        if k > MAX_HOURS:
            raise ValueError(f"{k} remaining hours is above the limit of {MAX_HOURS}")
        if hi.shape != (k,):
            raise ValueError("lo and hi must have one entry per remaining hour")
        high = high.reshape(least.size, k)
        if room.shape != least.shape:
            raise ValueError("least and room must have one entry per car")
        arrays = (high, least, room, lo, hi)
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("a future must be finite")
        if np.any(high < 0):
            raise ValueError("a believed power limit must be at least 0")
        if np.any(lo > hi):
            raise ValueError("the box has lo above hi in some hour")
        for key, value in zip(("high", "least", "room", "lo", "hi"), arrays, strict=True):
            value.flags.writeable = False
            object.__setattr__(self, key, value)

    @property
    def n_hours(self) -> int:
        return int(self.lo.size)


def _masks(k: int) -> np.ndarray:
    """(2^k, k): row `mask` holds 1 in column `j` when bit `j` of `mask` is set."""
    masks = np.arange(2**k)
    return ((masks[:, None] >> np.arange(k)) & 1).astype(float)


def enough(future: Future) -> bool:
    """`p(A) <= hi(A)` on every set `A` of remaining hours (module text)."""
    sets = _masks(future.n_hours)
    inside = future.high @ sets.T  # (cars, 2^k): high[i](A)
    outside = future.high.sum(axis=1)[:, None] - inside
    lower = np.maximum(future.least[:, None] - outside, 0.0).sum(axis=0)
    return bool(np.all(lower <= sets @ future.hi + TOLERANCE))


def roomy(future: Future) -> bool:
    """`lo(A) <= b(A)` on every set `A` of remaining hours (module text)."""
    sets = _masks(future.n_hours)
    inside = future.high @ sets.T
    upper = np.minimum(inside, future.room[:, None]).sum(axis=0)
    return bool(np.all(sets @ future.lo <= upper + TOLERANCE))


def within_cars(future: Future) -> bool:
    """`least[i] <= room[i]` for every car (module text)."""
    return bool(np.all(future.least <= future.room + TOLERANCE))


def meets(future: Future) -> bool:
    """True when the believed remaining cars can follow some request vector of the remaining box
    and meet every believed need (module text)."""
    return within_cars(future) and enough(future) and roomy(future)
