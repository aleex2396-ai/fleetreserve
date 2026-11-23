"""The general split of outcome O3: T2's programme on the true sets with every gain the causal
structure forbids, the decision hour's included (plan sections 2, 5 (T2) and 11 (O3)).

Plan section 2 calls a rule causal when `G_i(t, s) = 0 if s > t or t precedes S`: a car's power
in a service hour may not depend on a later request, and its power in the decision hour (which
precedes `S`) may not depend on any request. The general (non-causal) split lifts both: every car
`i` has a gain `G_i(t, s)` for every hour `t` in 0 to `n` and request hour `s` in 1 to `n`. The
decision hour carries no request, so there the gains sum to zero over the cars
(`sum_i G_i(0, s) = 0`): the fleet's decision-hour power is one number for every request, while a
car's share of it may move with the request. The box `y = m + W zeta` of the product and the policy
`u_i = g_i + G_i zeta` meet `sum_i g_i(t) = m_t` and `sum_i G_i(t, s) = [s = t] w_t` in every
service hour, and every row `a . u <= b` of every car holds for every `zeta` in `[-1, 1]^n`, which
is `a . g_i + sum_s |(a G_i)_s| <= b` (plan T2(b)), each `|(a G_i)_s|` bounded by one variable.
The value is `sum_t 2 w_t`, in kWh.

`certificate.certify(..., "full")` is the same programme without the decision-hour gains (its
`products.gain_pairs` start at hour 1); O3 reports it beside this one as "general split, decision
hour without gain". With the decision-hour gains, LP (24)'s box is such a split: its base-set
profile is affine in the request with hour 0 at 0 (`inner_box`'s closed form), so the car maps
give each car an affine rule, and the fleet's decision-hour power is `p_0` for every request
(`P = alpha I` has no off-diagonal entry), so the decision-hour gains sum to zero; LP (24)'s box is
at most this value. LP (27)'s box is not bounded so: its general `P` moves the fleet's
decision-hour power with the request, which its box allows (hour 0 free per corner, as in the
ceiling) and this split does not.

One sparse linear programme through `lp.solve_lp` (interior point with crossover), with the box
time limit by default. Any outcome but optimal gives no value: a timeout is missing, never zero.
"""

from dataclasses import dataclass

import numpy as np

from fleetreserve.carsets import FleetRows
from fleetreserve.lp import BOX_SECONDS
from fleetreserve.products import Product, add_box
from fleetreserve.sparse_programme import SparseProgramme

OPTIMAL = "optimal"


@dataclass(frozen=True)
class Split:
    """The general split's value in kWh (None unless `outcome` is optimal) and the solve's
    seconds."""

    outcome: str
    value: float | None
    seconds: float


def general_split(rows: FleetRows, product: Product, *, time_limit: float = BOX_SECONDS) -> Split:
    """The largest box of `product` met by an affine split with every gain, the decision hour's
    included (module text), on the sets `rows`."""
    n, cars = rows.n_service, rows.n_cars
    hours = n + 1
    programme = SparseProgramme()
    box = add_box(programme, product, n)
    g = programme.variables((cars, hours))
    gain = programme.variables((cars, hours, n))  # G_i(t, s - 1), t in 0..n, s in 1..n

    # sum_i g_i(t) = m_t in every service hour t.
    t_index = np.arange(1, hours)
    row = np.repeat(np.arange(n), cars)
    column = g[:, t_index].T.reshape(-1)
    entries = [(row, column, np.ones(row.size))]
    if box.centre is not None:
        entries.append((np.arange(n), box.centre, -np.ones(n)))
    programme.rows(
        "eq",
        *(np.concatenate(parts) for parts in zip(*entries, strict=True)),
        box.centre_constant,
    )
    # sum_i G_i(t, s) = [s = t] w_t for t in 1..n, and sum_i G_i(0, s) = 0, for s in 1..n.
    pairs = hours * n
    row = np.tile(np.arange(pairs), cars)
    column = gain.reshape(-1)
    values = np.ones(row.size)
    diagonal = np.arange(1, hours) * n + np.arange(n)  # the pair (t, s = t)
    row = np.concatenate([row, diagonal])
    column = np.concatenate([column, box.width])
    values = np.concatenate([values, -box.width_coefficient])
    programme.rows("eq", row, column, values, np.zeros(pairs))

    # Every row k of car i: a . g_i + sum_s z_ks <= b, with z_ks >= +-(a G_i)_s.
    k_index, t_nonzero = np.nonzero(rows.coefficients)
    a = rows.coefficients[k_index, t_nonzero]
    car = rows.car[k_index]
    z = programme.variables((rows.n_rows, n), 0.0, np.inf)
    for sign in (1.0, -1.0):
        programme.rows(
            "le",
            np.concatenate(
                [
                    np.repeat(k_index * n, n) + np.tile(np.arange(n), k_index.size),
                    np.arange(rows.n_rows * n),
                ]
            ),
            np.concatenate([gain[car, t_nonzero, :].reshape(-1), z.reshape(-1)]),
            np.concatenate([sign * np.repeat(a, n), -np.ones(rows.n_rows * n)]),
            np.zeros(rows.n_rows * n),
        )
    programme.rows(
        "le",
        np.concatenate([k_index, np.repeat(np.arange(rows.n_rows), n)]),
        np.concatenate([g[car, t_nonzero], z.reshape(-1)]),
        np.concatenate([a, np.ones(rows.n_rows * n)]),
        rows.bounds,
    )
    result, _ = programme.maximise(*box.value_terms(), time_limit=time_limit)
    if result.outcome != OPTIMAL:
        return Split(result.outcome, None, result.seconds)
    _, w, _ = box.read(result.x)
    return Split(OPTIMAL, float(2.0 * w.sum()), result.seconds)
