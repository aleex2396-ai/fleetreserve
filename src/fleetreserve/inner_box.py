"""The largest box of a product on the service hours inside a certified inner set of LP (24) or
LP (27) (plan section 1, "format loss: ceiling minus the largest box in LP (24)'s set", and
section 11, O3).

A solved LP (24) or LP (27) (`lp24.InnerSet`) certifies the set `C = p + P U0` of profiles on
hours 0 to `n_service`: every profile in it can be shared out among the cars within their sets.
The box of a product (`products`: hourly, `[lo_t, hi_t]` free in every service hour; block,
`hi_t - lo_t = 2 lam` in every service hour, centre free) lies in `C` here when every request
path in the box, together with some power in hour 0, is a profile of `C`: hour 0 carries no
request and is free within `C` (it may differ from one corner to another), as in the ceiling
(`exact`), where hour 0 is free in the projection. A box lies in a convex set iff its corners do,
so the largest box is one linear programme over the box and one base-set profile per corner.
Value `sum_t (hi_t - lo_t)`, in kWh.

- `largest_box(inner, product)`: LP (24)'s set `p + alpha U0` by its closed form (below);
  LP (27)'s set by `vertex_box`.
- `vertex_box(p, P, base, product)`: the programme over corners, for any `p + P U0`; its time limit
  is the box limit (`lp.BOX_SECONDS`), and any outcome but optimal returns a box with no numbers.

Closed form for `p + alpha U0`, `U0 = {u : 0 <= u <= power, sum u <= energy}`. Hour 0 is free, so
it is best at `u_0 = 0`, and the set seen on the service hours `S` is
`p_S + alpha {v : 0 <= v <= power_S, sum v <= energy}`. That set holds every point between 0 and
any of its points, so a box `[lo, hi]` lies in it iff `lo >= p_S` and `hi` lies in it. Hence:

- hourly product: value `alpha min(sum_{t in S} power_t, energy)`, at `lo = p_S` and
  `hi = p_S + alpha v` with `v = power_S min(1, energy / sum power_S)` (one of the boxes of that
  value; the rule that picks among them is not set by the plan for this baseline);
- block product: `2 lam = alpha min(min_{t in S} power_t, energy / |S|)`, value `2 lam |S|`, at
  `lo = p_S`.

Readings:

1. "The largest box in LP (24)'s set" (plan section 1) is the largest box of the product in the
   certified set `p + alpha U0` with hour 0 free per corner (above). The alternative, hour 0 one
   power for every corner (decided at the decision time, as in the causal certificate), gives the
   same value for LP (24) (the closed form uses `u_0 = 0` for every corner) and possibly a smaller
   one for LP (27).
2. Another reading of the phrase, LP (24) solved with the box itself as the base set, is the
   general split of the causal certificate (`certificate.certify(..., structure="full")`) for a
   box of that shape, since `sum_i G_i = W` there is `sum_i Gamma_i = alpha I` for the box; it is
   not built here.
"""

from dataclasses import dataclass

import numpy as np

from fleetreserve.base_sets import BaseSet
from fleetreserve.lp import BOX_SECONDS, Outcome
from fleetreserve.lp24 import InnerSet
from fleetreserve.products import Product, corner_paths
from fleetreserve.sparse_programme import SparseProgramme


@dataclass(frozen=True)
class InnerBox:
    """A box `m +- w` on the service hours 1 to `n` (index `t - 1`). `value`, `m` and `w` are None
    unless `outcome` is "optimal". `seconds` is 0 for the closed form."""

    outcome: Outcome
    kind: str
    value: float | None
    m: np.ndarray | None
    w: np.ndarray | None
    seconds: float


def _check_product(product: Product):
    if product.kind not in ("hourly", "block"):
        raise ValueError("only the hourly and the block product are built here")


def scaled_box(p, alpha: float, base: BaseSet, product: Product) -> InnerBox:
    """The closed form for `p + alpha U0` (module text)."""
    _check_product(product)
    service = base.power[1:]
    p_s = np.asarray(p, dtype=float)[1:]
    if product.kind == "hourly":
        total = service.sum()
        shape = service * (min(1.0, base.energy / total) if total > 0 else 0.0)
        width = alpha * shape
    else:
        width = np.full(service.size, alpha * min(service.min(), base.energy / service.size))
    return InnerBox("optimal", product.kind, float(width.sum()), p_s + width / 2, width / 2, 0.0)


def vertex_box(
    p,
    P,
    base: BaseSet,
    product: Product,
    *,
    shared_hour0: bool = False,
    time_limit: float = BOX_SECONDS,
) -> InnerBox:
    """The largest box of `product` in `p + P U0` by one programme over the box's corners; with
    `shared_hour0`, hour 0 has one power for every corner (reading 1's alternative)."""
    _check_product(product)
    p = np.asarray(p, dtype=float)
    P = np.asarray(P, dtype=float)
    n = base.n_service
    columns = base.columns
    q = columns.size
    paths = corner_paths(n)  # (corners, n)
    corners = paths.shape[0]
    lp = SparseProgramme()
    m = lp.variables(n)
    if product.kind == "hourly":
        width = lp.variables(n, 0.0, np.inf)
    else:
        width = np.full(n, int(lp.variables(1, 0.0, np.inf)[0]))
    u = lp.variables((corners, q), 0.0, np.broadcast_to(base.power[columns], (corners, q)))
    # Energy of each corner's base-set profile.
    lp.rows(
        "le",
        np.repeat(np.arange(corners), q),
        u.reshape(-1),
        np.ones(corners * q),
        np.full(corners, base.energy),
    )
    # (P u^k)_t - m_t - zeta_kt w_t = -p_t for every corner k and service hour t.
    map_s = P[1:][:, columns]  # (n, q)
    t_rows = np.arange(corners * n).reshape(corners, n)
    row = np.concatenate(
        [
            np.repeat(t_rows.reshape(-1), q),
            t_rows.reshape(-1),
            t_rows.reshape(-1),
        ]
    )
    column = np.concatenate(
        [
            np.repeat(u[:, None, :], n, axis=1).reshape(-1),
            np.tile(m, corners),
            np.tile(width, corners),
        ]
    )
    value = np.concatenate(
        [np.tile(map_s.reshape(-1), corners), -np.ones(corners * n), -paths.reshape(-1)]
    )
    lp.rows("eq", row, column, value, -np.tile(p[1:], corners))
    if shared_hour0:
        # (P u^k)_0 - h = -p_0 for every corner k, with one free hour-0 power h.
        h = int(lp.variables(1)[0])
        lp.rows(
            "eq",
            np.concatenate([np.repeat(np.arange(corners), q), np.arange(corners)]),
            np.concatenate([u.reshape(-1), np.full(corners, h)]),
            np.concatenate([np.tile(P[0, columns], corners), -np.ones(corners)]),
            np.full(corners, -p[0]),
        )
    if product.kind == "hourly":
        objective = (width, np.full(n, 2.0))
    else:
        objective = (width[:1], np.array([2.0 * n]))
    result, _ = lp.maximise(*objective, time_limit=time_limit)
    if result.outcome != "optimal":
        return InnerBox(result.outcome, product.kind, None, None, None, result.seconds)
    x = result.x
    w = np.maximum(x[width], 0.0)
    return InnerBox("optimal", product.kind, float(2.0 * w.sum()), x[m], w, result.seconds)


def largest_box(
    inner: InnerSet,
    product: Product,
    *,
    shared_hour0: bool = False,
    time_limit: float = BOX_SECONDS,
) -> InnerBox:
    """The largest box of `product` in a solved inner set; a missing inner set gives a missing
    box with the same outcome. LP (24)'s box is the closed form with either hour-0 reading."""
    _check_product(product)
    if inner.outcome != "optimal":
        return InnerBox(inner.outcome, product.kind, None, None, None, 0.0)
    if inner.programme == "24":
        return scaled_box(inner.p, inner.alpha, inner.base, product)
    return vertex_box(
        inner.p,
        inner.P,
        inner.base,
        product,
        shared_hour0=shared_hour0,
        time_limit=time_limit,
    )
