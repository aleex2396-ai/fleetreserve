"""LP (24) and LP (27) of Al Taha et al. (2025) on per-car sets given as rows (plan sections 1, 10,
11 (O3) and 13 (WP4, WP5)).

Numbering and statements as in Al Taha, Vincent, Bitar, arXiv 2207.07067 version 3. Every car
`i` has the set `U_i = {u : a . u <= b for each of its rows}` (`carsets`, hours 0 to
`n_service`, recovery eliminated) and the base set is `U0` (`base_sets`). Their Lemma 1 (from
Sadraddini and Tedrake 2019): `gamma + Gamma X` lies in `{y : H_y y <= h_y}`, with
`X = {x : H_x x <= h_x}` non-empty, iff some `Lambda >= 0` has `Lambda H_x = H_y Gamma` and
`Lambda h_x <= h_y - H_y gamma`. Their Theorem 2: `p + P U0` lies in the fleet's set (the sum of
the `U_i`) if there are `gamma_i, Gamma_i, Lambda_i` with `[p, P] = sum_i [gamma_i, Gamma_i]`
and Lemma 1's conditions for each car.

- LP (24) ("structure preserving"): maximise `alpha` subject to `[p, alpha I] = sum_i
  [gamma_i, Gamma_i]` and Lemma 1 for each car; `Gamma_i` are full matrices, so car `i`'s power
  in an hour depends on the whole profile `u0`, later hours included (they look ahead).
- LP (27) ("general transformation"): maximise the trace of `P` subject to
  `[p, P] = sum_i [gamma_i, Gamma_i]` and Lemma 1 for each car, `P` any square matrix. Their
  Remark 5: it splits into one programme per car, maximise `Tr(Gamma_i)` subject to
  `gamma_i + Gamma_i U0` inside `U_i`, and `LP (27)`'s value is the sum.

How Lemma 1 is written here. Lemma 1 is necessary and sufficient for each car, whatever rows
describe `U0`, so `U0` is described by its irredundant rows `0 <= u_b <= power_b` (`b` in the
hours where the base set has power, `J`) and `sum_b u_b <= energy`. For a row `a . u <= b` of car
`i`, the condition is then `a . gamma_i + max over u0 in U0 of (a Gamma_i) u0 <= b`, and by
linear-programming duality that maximum is at most a number iff there are `w >= 0` (one per hour
of `J`) and `s >= 0` with `w_b >= (a Gamma_i)_b - s` and `power . w + energy s` at most that
number. Each car row gets its own `w, s` (the rows of `Lambda_i` on the base set's rows). The
earlier analysis wrote LP (24) in the same way. The columns of `Gamma_i` outside `J` multiply a
base-set power that is always 0; they are left out (zero in the results).

Readings (where the paper's wording does not decide for these sets):

1. `alpha >= 0`, where LP (24) asks `alpha > 0`: a fleet that can offer nothing gets `alpha = 0`
   rather than no solution (as in the earlier analysis).
2. LP (27)'s trace is over the hours of `J` only: an entry `P[t, t]` with `t` outside `J`
   multiplies a base-set power that is always 0, so it is not limited by any row and would make
   the trace unbounded.
3. A base set that holds only the zero profile is refused (`ValueError`): every `alpha` would
   then be feasible.
4. The certified set, `p + alpha U0` or `p + P U0`, is a set of profiles on hours 0 to
   `n_service`; the recovery hours are eliminated in the rows, as for the causal certificate.
5. LP (27)'s time limit is that of "others" (300 s), not LP (24)'s 600 s.

The programme is solved through `sparse_programme` and so `lp.solve_lp`, interior point with
crossover. Default time limits (plan section 13, "600 s LP (24), 300 s others"): `lp.LP24_SECONDS`
for LP (24) and `lp.OTHER_SECONDS` for LP (27) (reading 5: LP (27) is one of the "others"). Any
outcome but optimal returns an `InnerSet` with no numbers: a timeout is missing, never zero and
never infeasible.
"""

from dataclasses import dataclass

import numpy as np

from fleetreserve.base_sets import BaseSet
from fleetreserve.carsets import FleetRows
from fleetreserve.lp import LP24_SECONDS, OTHER_SECONDS, Outcome
from fleetreserve.sparse_programme import Sizes, SparseProgramme

PROGRAMMES = ("24", "27")


@dataclass(frozen=True)
class InnerSet:
    """The certified inner set `p + P U0` of a fleet's set.

    `value` (`alpha` for LP (24), the trace of `P` on the base set's hours for LP (27)), `p`
    (shape `(n_service + 1,)`) and `P` (shape `(n_service + 1, n_service + 1)`; `alpha I` on the
    base set's hours for LP (24), zero columns outside them) are None unless `outcome` is
    "optimal". `gamma` `(cars, n_service + 1)` and `Gamma` `(cars, n_service + 1, n_service + 1)`,
    the per-car maps, are kept only when asked for (`keep_maps=True`)."""

    outcome: Outcome
    programme: str
    base: BaseSet
    value: float | None
    p: np.ndarray | None
    P: np.ndarray | None
    gamma: np.ndarray | None
    Gamma: np.ndarray | None
    seconds: float
    sizes: Sizes

    @property
    def alpha(self) -> float | None:
        """LP (24)'s `alpha` (None for LP (27) or a missing result)."""
        return self.value if self.programme == "24" else None

    def profile(self, u0) -> np.ndarray:
        """`p + P u0` for a base-set profile `u0` (shape `(hours,)` or `(points, hours)`)."""
        if self.outcome != "optimal":
            raise ValueError("a missing result has no profile")
        return self.p + np.asarray(u0, dtype=float) @ self.P.T

    def car_powers(self, u0) -> np.ndarray:
        """`gamma_i + Gamma_i u0` per car, shape `(cars, hours)` (needs `keep_maps=True`)."""
        if self.gamma is None:
            raise ValueError("the per-car maps were not kept (keep_maps=False)")
        return self.gamma + np.einsum("its,s->it", self.Gamma, np.asarray(u0, dtype=float))


def _build(rows: FleetRows, base: BaseSet, programme: str):
    hours = rows.n_service + 1
    columns = base.columns
    q = columns.size
    n_cars, n_rows = rows.n_cars, rows.n_rows
    lp = SparseProgramme()
    alpha = int(lp.variables(1, 0.0, np.inf)[0]) if programme == "24" else None
    gamma = lp.variables((n_cars, hours))
    gain = lp.variables((n_cars, hours, q))
    w = lp.variables((n_rows, q), 0.0, np.inf)
    s = lp.variables(n_rows, 0.0, np.inf)

    # (a Gamma_i)_b - w_kb - s_k <= 0 for every row k (of car i) and base-set hour b.
    k_index, t_index = np.nonzero(rows.coefficients)
    a = rows.coefficients[k_index, t_index]
    b_index = np.arange(q)
    row = np.concatenate(
        [
            (k_index[:, None] * q + b_index[None, :]).reshape(-1),
            np.arange(n_rows * q),
            np.arange(n_rows * q),
        ]
    )
    column = np.concatenate(
        [
            gain[rows.car[k_index], t_index, :].reshape(-1),
            w.reshape(-1),
            np.repeat(s, q),
        ]
    )
    value = np.concatenate([np.repeat(a, q), -np.ones(2 * n_rows * q)])
    lp.rows("le", row, column, value, np.zeros(n_rows * q))

    # a . gamma_i + power_J . w_k + energy s_k <= b_k.
    row = np.concatenate([k_index, np.repeat(np.arange(n_rows), q), np.arange(n_rows)])
    column = np.concatenate([gamma[rows.car[k_index], t_index], w.reshape(-1), s])
    value = np.concatenate([a, np.tile(base.power[columns], n_rows), np.full(n_rows, base.energy)])
    lp.rows("le", row, column, value, rows.bounds)

    if programme == "24":
        # sum_i Gamma_i[t, b] = alpha [t == J_b] for every hour t and base-set hour b.
        row = np.concatenate([np.tile(np.arange(hours * q), n_cars), np.arange(q) + columns * q])
        column = np.concatenate([gain.reshape(-1), np.full(q, alpha)])
        value = np.concatenate([np.ones(n_cars * hours * q), -np.ones(q)])
        lp.rows("eq", row, column, value, np.zeros(hours * q))
        objective = (np.array([alpha]), np.array([1.0]))
    else:
        objective = (gain[:, columns, b_index].reshape(-1), np.ones(n_cars * q))
    return lp, alpha, gamma, gain, objective


def inner_set(
    rows: FleetRows,
    base: BaseSet,
    programme: str,
    *,
    keep_maps: bool = False,
    time_limit: float | None = None,
) -> InnerSet:
    """LP (24) (`programme="24"`) or LP (27) (`programme="27"`) on the sets `rows` with the base
    set `base` (whose hours must be those of the rows). `time_limit` None: the programme's default
    (module text)."""
    if programme not in PROGRAMMES:
        raise ValueError(f"programme must be one of {PROGRAMMES}, got {programme!r}")
    if time_limit is None:
        time_limit = LP24_SECONDS if programme == "24" else OTHER_SECONDS
    if base.n_service != rows.n_service:
        raise ValueError("the base set and the rows have different hours")
    if base.is_trivial():
        raise ValueError("the base set holds only the zero profile")
    lp, alpha, gamma, gain, objective = _build(rows, base, programme)
    result, sizes = lp.maximise(*objective, time_limit=time_limit)
    if result.outcome != "optimal":
        return InnerSet(
            result.outcome, programme, base, None, None, None, None, None, result.seconds, sizes
        )
    x = result.x
    hours = rows.n_service + 1
    columns = base.columns
    g = x[gamma]
    big = np.zeros((rows.n_cars, hours, hours))
    big[:, :, columns] = x[gain]
    p = g.sum(axis=0)
    transform = big.sum(axis=0)
    if programme == "24":
        value = float(x[alpha])
        transform = np.zeros((hours, hours))
        transform[columns, columns] = value
    else:
        value = float(np.trace(transform[np.ix_(columns, columns)]))
    for array in (p, transform, g, big):
        array.flags.writeable = False
    return InnerSet(
        "optimal",
        programme,
        base,
        value,
        p,
        transform,
        g if keep_maps else None,
        big if keep_maps else None,
        result.seconds,
        sizes,
    )
