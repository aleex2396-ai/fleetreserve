"""An affine dispatch policy and its exact evaluation (plan sections 1, 2, 5 (T2) and 6 (T6)).

The policy `(m, w, g, G)`: the box `y = m + W zeta` on the service hours 1 to `n`; car `i` draws
`u_i = g_i + G_i zeta` in hours 0 to `n`. Arrays: `m`, `w` of shape `(n,)` (index `t - 1` for
service hour `t`); `g` of shape `(cars, n + 1)` (index `t` for hour `t`); `G` of shape
`(cars, n + 1, n)`, `G[i, t, s - 1]` the gain of car `i` in hour `t` on the request of service
hour `s` (row 0, the decision hour, is zero).

Scale `beta` about the midpoint (plan section 1): the box `m +- beta w`, run by
`u_i = g_i + beta G_i zeta`.

Worst case of a row `a . u_i <= b` over every `zeta` in `[-1, 1]^n` at scale `beta`:
`a . g_i + beta |a G_i|_1` (plan T2(b)); it is reached at a corner path. The largest `beta` a row
allows is `(b - a . g_i) / |a G_i|_1` (plan T6's per-row threshold). `row_thresholds` reads a row
whose `|a G_i|_1`, as computed, is at most `ZERO_GAIN` as not depending on `zeta`: `+inf` when it
holds at the centre within `ROW_TOLERANCE`, else `-inf`; any other row that fails at the centre
has a negative threshold.

Which hour a row's threshold belongs to, for T6's hourly thresholds `s_dt`, is left to the caller:
`FleetRows.last_hours` and `FleetRows.kinds` give each row's closing hour and kind.
"""

from dataclasses import dataclass

import numpy as np

from fleetreserve.carsets import FleetRows

ZERO_GAIN = 1e-12  # |a G|_1 at or below this counts as no dependence on the requests
ROW_TOLERANCE = 1e-7  # kW or kWh: a row that does not depend on the requests holds within this
# the validity limit, 1e5 exactly as a double; what it limits and why: `fixed_plan_loss`, module
# text, "The validity limit and the stated errors"
VALIDITY_LIMIT = ROW_TOLERANCE / ZERO_GAIN


@dataclass(frozen=True)
class AffinePolicy:
    m: np.ndarray
    w: np.ndarray
    g: np.ndarray
    G: np.ndarray

    def __post_init__(self):
        arrays = {k: np.array(getattr(self, k), dtype=float) for k in ("m", "w", "g", "G")}
        n = arrays["m"].size
        cars = arrays["g"].shape[0] if arrays["g"].ndim == 2 else -1
        if (
            arrays["m"].shape != (n,)
            or arrays["w"].shape != (n,)
            or arrays["g"].shape != (cars, n + 1)
            or arrays["G"].shape != (cars, n + 1, n)
        ):
            raise ValueError("policy arrays have inconsistent shapes")
        if np.any(arrays["G"][:, 0, :] != 0):
            raise ValueError("the decision hour cannot depend on the requests")
        for key, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, key, value)

    @property
    def n_service(self) -> int:
        return int(self.m.size)

    @property
    def n_cars(self) -> int:
        return int(self.g.shape[0])

    def _zeta(self, zeta) -> np.ndarray:
        z = np.asarray(zeta, dtype=float)
        if z.shape[-1] != self.n_service:
            raise ValueError(f"zeta must end in {self.n_service} entries")
        return z

    def request(self, zeta, beta=1.0) -> np.ndarray:
        """`m + beta w zeta`; `zeta` of shape `(n,)` or `(paths, n)`."""
        return self.m + beta * self.w * self._zeta(zeta)

    def powers(self, zeta, beta=1.0) -> np.ndarray:
        """`g_i + beta G_i zeta`: shape `(cars, n + 1)`, or `(paths, cars, n + 1)` for many."""
        z = self._zeta(zeta)
        return self.g + beta * np.einsum("its,...s->...it", self.G, z)

    def _forms(self, rows: FleetRows):
        if rows.n_cars != self.n_cars or rows.n_service != self.n_service:
            raise ValueError("the rows are for another fleet or window")
        centre = np.einsum("kt,kt->k", rows.coefficients, self.g[rows.car])
        gains = np.einsum("kt,kts->ks", rows.coefficients, self.G[rows.car])
        return centre, np.abs(gains).sum(axis=1)

    def worst_case(self, rows: FleetRows, beta=1.0) -> np.ndarray:
        """Per row, `max over zeta of a . u_i` at scale `beta`: `a . g_i + beta |a G_i|_1`."""
        centre, spread = self._forms(rows)
        return centre + beta * spread

    def row_thresholds(self, rows: FleetRows) -> np.ndarray:
        """Per row, its threshold `(b - a g) / |a G|_1`, the largest `beta` at which it holds for
        every `zeta`; `+inf` or `-inf` for a row with `|a G|_1` at most `ZERO_GAIN` (module
        text)."""
        centre, spread = self._forms(rows)
        room = rows.bounds - centre
        free = spread <= ZERO_GAIN
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = room / spread
        return np.where(free, np.where(room >= -ROW_TOLERANCE, np.inf, -np.inf), ratio)
