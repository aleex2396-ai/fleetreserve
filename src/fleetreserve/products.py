"""The products a box is sold as, and the gain structures of an affine rule (plan sections 1, 2, 5).

A box is `y = m + W zeta`, `zeta` in `[-1, 1]^n`, `W = diag(w) >= 0`, on the service hours 1 to
`n`; its value is `sum_t 2 w_t` (kWh, one-hour slots).

Products:

- `hourly()`: `m` and `w` free in every hour.
- `block()`: `w_t = lam` in every hour, the centre free; value `2 lam n`.
- `fixed_shape(centre, widths)`: `m = centre`, `w = lam * widths`; value `2 lam sum(widths)`.

Structures of the gains `G_i(t, s)` (hour `t` in 1 to `n`, request hour `s` in 1 to `n`; the
decision hour 0 never depends on `zeta`):

- `"diagonal"`: `G_i(t, s) = 0` unless `s = t`;
- `"causal"`: `G_i(t, s) = 0` if `s > t` (lower-triangular);
- `"full"`: unrestricted on the service hours, the general split, which uses future requests.

`add_box` puts a product's box into a `SparseProgramme`: the centre and width of every hour as
linear terms of the box's variables, the value as the objective, and the reading of a solution.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fleetreserve.sparse_programme import SparseProgramme

STRUCTURES = ("diagonal", "causal", "full")

Structure = Literal["diagonal", "causal", "full"]


@dataclass(frozen=True)
class Product:
    """`kind` is "hourly", "block" or "fixed"; `centre` and `widths` are set only for "fixed"."""

    kind: Literal["hourly", "block", "fixed"]
    centre: tuple[float, ...] | None = None
    widths: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.kind not in ("hourly", "block", "fixed"):
            raise ValueError(f"unknown product {self.kind!r}")
        if (self.kind == "fixed") != (self.centre is not None and self.widths is not None):
            raise ValueError("a fixed shape, and only it, gives a centre and widths")
        if self.kind == "fixed":
            centre = tuple(float(v) for v in self.centre)
            widths = tuple(float(v) for v in self.widths)
            if len(centre) != len(widths) or not np.all(np.isfinite(centre + widths)):
                raise ValueError("centre and widths must be finite, one of each per hour")
            if min(widths) < 0 or max(widths) == 0:
                raise ValueError("widths must be non-negative and not all zero")
            object.__setattr__(self, "centre", centre)
            object.__setattr__(self, "widths", widths)

    def check_hours(self, n_service: int):
        if self.kind == "fixed" and len(self.centre) != n_service:
            raise ValueError(f"the shape has {len(self.centre)} hours, the fleet {n_service}")


def hourly() -> Product:
    return Product("hourly")


def block() -> Product:
    return Product("block")


def fixed_shape(centre, widths) -> Product:
    return Product("fixed", tuple(centre), tuple(widths))


@dataclass(frozen=True)
class BoxColumns:
    """A box in a programme. Centre of hour `t` (index `t - 1`): the variable `centre[t - 1]`, or
    the constant `centre_constant[t - 1]` when `centre` is None. Width: `width_coefficient[t - 1]`
    times the variable `width[t - 1]`. `scale` is the variable `lam` (block, fixed shape) or None.
    """

    product: Product
    centre: np.ndarray | None
    centre_constant: np.ndarray
    width: np.ndarray
    width_coefficient: np.ndarray
    scale: int | None

    def value_terms(self) -> tuple[np.ndarray, np.ndarray]:
        """The value `sum_t 2 w_t` as (columns, coefficients)."""
        return self.width, 2.0 * self.width_coefficient

    def read(self, x) -> tuple[np.ndarray, np.ndarray, float | None]:
        """`(m, w, lam)` from a solution `x` (`lam` None for the hourly product)."""
        x = np.asarray(x, dtype=float)
        m = self.centre_constant.copy() if self.centre is None else x[self.centre]
        w = np.maximum(self.width_coefficient * x[self.width], 0.0)
        return m, w, None if self.scale is None else float(x[self.scale])


def add_box(programme: SparseProgramme, product: Product, n_service: int) -> BoxColumns:
    """The box variables of `product`: `m`, `w` (hourly), `m`, `lam` (block) or `lam` (fixed)."""
    product.check_hours(n_service)
    n = n_service
    if product.kind == "fixed":
        scale = int(programme.variables(1, 0.0, np.inf)[0])
        return BoxColumns(
            product,
            None,
            np.array(product.centre),
            np.full(n, scale),
            np.array(product.widths),
            scale,
        )
    centre = programme.variables(n)
    if product.kind == "hourly":
        width = programme.variables(n, 0.0, np.inf)
        return BoxColumns(product, centre, np.zeros(n), width, np.ones(n), None)
    scale = int(programme.variables(1, 0.0, np.inf)[0])
    return BoxColumns(product, centre, np.zeros(n), np.full(n, scale), np.ones(n), scale)


def gain_pairs(structure: str, n_service: int) -> tuple[np.ndarray, np.ndarray]:
    """The `(t, s)` pairs of service hours (1 to `n`) where a structure allows a gain."""
    if structure not in STRUCTURES:
        raise ValueError(f"structure must be one of {STRUCTURES}, got {structure!r}")
    t, s = np.meshgrid(np.arange(1, n_service + 1), np.arange(1, n_service + 1), indexing="ij")
    keep = {"diagonal": s == t, "causal": s <= t, "full": np.ones_like(t, bool)}[structure]
    return t[keep], s[keep]


def corner_paths(n_service: int) -> np.ndarray:
    """All `2^n` corner paths, `(2^n, n)` of -1 and +1; path `k` has `zeta_j = +1` where bit `j`
    of `k` is set."""
    k = np.arange(2**n_service)
    return np.where((k[:, None] >> np.arange(n_service)) & 1, 1.0, -1.0)
