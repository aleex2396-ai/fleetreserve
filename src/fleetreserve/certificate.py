"""The causal certificate (plan section 5, T2): the best box of a product met by an affine rule.

On per-car sets given as rows (`carsets`), the programme finds the box `y = m + W zeta` of a
product (`products`) and a policy `u_i = g_i + G_i zeta` with the gains of a structure
("diagonal", "causal" or "full") such that every request is shared out exactly
(`sum_i g_i(t) = m_t`, `sum_i G_i(t, .) = W(t, .)` on the service hours; hour 0 carries no request
and its powers do not depend on `zeta`) and every car's row holds for every `zeta` in `[-1, 1]^n`.
A row `a . u <= b` holds for every `zeta` iff `a . g + |a G|_1 <= b` (plan T2(b)); each
`|(a G)_s|` is bounded by one variable `z >= +-(a G)_s`. Rows of a car that are multiples of the
same form (`u_t <= r` and `-u_t <= 0`; `sum_{s<=t} u_s <= e` and `-sum_{s<=t} u_s <= B`) share
those variables.

It is one sparse linear programme solved through `lp.solve_lp` (interior point with crossover,
plan section 13), with the box time limit by default. Any outcome but optimal returns a
`Certificate` with no numbers: a timeout is missing, never zero and never infeasible.

Shape of the box. The largest value is often reached by many boxes (the hourly product's shape,
the block product's centre). The rule that picks one is the caller's to name, with no default:
`shape_rule=ANY_OPTIMAL` returns the solver's own optimal vertex; a `ShapeRule` (for example
`LargestSmallestWidth`) solves a second programme over the optimal face, with the value held at
least at the first value minus `tolerance * max(1, first value)`, and maximises the rule's own
objective. A rule implements `objective(programme, box)`, adding any variables and rows it needs
and returning the objective's (columns, coefficients).
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fleetreserve.carsets import FleetRows
from fleetreserve.lp import BOX_SECONDS, Outcome
from fleetreserve.policy import AffinePolicy
from fleetreserve.products import BoxColumns, Product, add_box, gain_pairs
from fleetreserve.sparse_programme import Sizes, SparseProgramme

FORM_DIGITS = 12  # forms are compared after rounding to this many decimals


class ShapeRule(Protocol):
    name: str
    tolerance: float

    def objective(self, programme: SparseProgramme, box: BoxColumns): ...


@dataclass(frozen=True)
class _AnyOptimal:
    name: str = "any optimal vertex"


ANY_OPTIMAL = _AnyOptimal()


@dataclass(frozen=True)
class LargestSmallestWidth:
    """Second stage: among the boxes of the first value (within `tolerance`), the largest
    smallest width `min_t w_t`."""

    tolerance: float
    name: str = "largest smallest width"

    def __post_init__(self):
        if not 0 <= self.tolerance < 1:
            raise ValueError(f"tolerance must be in [0, 1), got {self.tolerance}")

    def objective(self, programme: SparseProgramme, box: BoxColumns):
        smallest = int(programme.variables(1, 0.0, np.inf)[0])
        n = box.width.size
        rows = np.repeat(np.arange(n), 2)
        columns = np.column_stack([np.full(n, smallest), box.width]).reshape(-1)
        values = np.column_stack([np.ones(n), -box.width_coefficient]).reshape(-1)
        programme.rows("le", rows, columns, values, np.zeros(n))
        return np.array([smallest]), np.array([1.0])


@dataclass(frozen=True)
class Certificate:
    """The best box and its policy. `value`, `scale` and `policy` are None unless `outcome` is
    "optimal". `scale` is `lam` (block, fixed shape), None for the hourly product.
    `first_value` is the first stage's value; `seconds` and `sizes` cover every stage solved
    (`sizes` those of the last)."""

    outcome: Outcome
    product: Product
    structure: str
    shape_rule: str
    value: float | None
    scale: float | None
    policy: AffinePolicy | None
    first_value: float | None
    seconds: float
    sizes: Sizes


def _forms(rows: FleetRows):
    """Each row as `sign_scale * form`, forms shared within a car. Returns the form of each row,
    the row's factor, the forms' cars and coefficients."""
    a = rows.coefficients
    nonzero = a != 0
    first = np.argmax(nonzero, axis=1)
    factor = np.where(nonzero.any(axis=1), a[np.arange(a.shape[0]), first], 1.0)
    form = np.round(a / factor[:, None], FORM_DIGITS)
    key = np.column_stack([rows.car, form])
    unique, inverse = np.unique(key, axis=0, return_inverse=True)
    return inverse.reshape(-1), factor, unique[:, 0].astype(int), unique[:, 1:]


def _build(rows: FleetRows, product: Product, structure: str):
    n, cars = rows.n_service, rows.n_cars
    hours = n + 1
    programme = SparseProgramme()
    box = add_box(programme, product, n)
    g = programme.variables((cars, hours))
    pair_t, pair_s = gain_pairs(structure, n)
    gain = programme.variables((cars, pair_t.size))

    # Requests shared out: sum_i g_i(t) = m_t, sum_i G_i(t, s) = [s = t] w_t.
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
    diagonal = np.flatnonzero(pair_t == pair_s)
    row = np.concatenate([np.repeat(np.arange(pair_t.size), cars), diagonal])
    column = np.concatenate([gain.T.reshape(-1), box.width[pair_t[diagonal] - 1]])
    value = np.concatenate(
        [np.ones(pair_t.size * cars), -box.width_coefficient[pair_t[diagonal] - 1]]
    )
    programme.rows("eq", row, column, value, np.zeros(pair_t.size))

    # |(form G_i)_s| <= z[form, s] for every form and request hour s it reaches.
    row_form, factor, form_car, form_coefficients = _forms(rows)
    n_forms = form_car.size
    reach = form_coefficients[:, pair_t]  # (forms, pairs): coefficient on G(t, s)
    f_index, p_index = np.nonzero(reach)
    active = np.zeros((n_forms, n), bool)
    active[f_index, pair_s[p_index] - 1] = True
    z_of = np.full((n_forms, n), -1)
    z_of[active] = programme.variables(int(active.sum()), 0.0, np.inf)
    target = z_of[f_index, pair_s[p_index] - 1]  # the z of each entry
    gain_columns = gain[form_car[f_index], p_index]
    coefficients = reach[f_index, p_index]
    n_z = int(active.sum())
    z_rank = np.full(programme.n_variables, -1)
    z_rank[z_of[active]] = np.arange(n_z)
    z_row = z_rank[target]
    for sign in (1.0, -1.0):
        programme.rows(
            "le",
            np.concatenate([z_row, np.arange(n_z)]),
            np.concatenate([gain_columns, z_of[active]]),
            np.concatenate([sign * coefficients, -np.ones(n_z)]),
            np.zeros(n_z),
        )

    # Robust rows: a . g_i + |factor| sum_s z[form, s] <= b.
    k_index, t_nonzero = np.nonzero(rows.coefficients)
    z_rows = z_of[row_form]  # (rows, n)
    zk, zs = np.nonzero(z_rows >= 0)
    programme.rows(
        "le",
        np.concatenate([k_index, zk]),
        np.concatenate([g[rows.car[k_index], t_nonzero], z_rows[zk, zs]]),
        np.concatenate([rows.coefficients[k_index, t_nonzero], np.abs(factor[zk])]),
        rows.bounds,
    )
    return programme, box, g, gain, (pair_t, pair_s)


def _policy(x, rows: FleetRows, box: BoxColumns, g, gain, pairs) -> AffinePolicy:
    m, w, _ = box.read(x)
    pair_t, pair_s = pairs
    gains = np.zeros((rows.n_cars, rows.n_service + 1, rows.n_service))
    gains[:, pair_t, pair_s - 1] = x[gain]
    return AffinePolicy(m, w, x[g], gains)


def certify(
    rows: FleetRows,
    product: Product,
    structure: str,
    *,
    shape_rule,
    time_limit: float = BOX_SECONDS,
) -> Certificate:
    """The best box of `product` for `structure` on the sets `rows`, and its policy.

    `shape_rule` has no default: `ANY_OPTIMAL` or a `ShapeRule` (see the module text).
    `time_limit` applies to each stage's solve."""
    if not (shape_rule is ANY_OPTIMAL or hasattr(shape_rule, "objective")):
        raise ValueError("shape_rule must be ANY_OPTIMAL or a ShapeRule")
    programme, box, g, gain, pairs = _build(rows, product, structure)
    result, sizes = programme.maximise(*box.value_terms(), time_limit=time_limit)
    seconds = result.seconds

    def missing(outcome, first=None):
        return Certificate(
            outcome, product, structure, shape_rule.name, None, None, None, first, seconds, sizes
        )

    if result.outcome != "optimal":
        return missing(result.outcome)
    first_value = -result.objective
    if shape_rule is not ANY_OPTIMAL:
        columns, values = box.value_terms()
        floor = first_value - shape_rule.tolerance * max(1.0, abs(first_value))
        programme.rows("le", np.zeros(columns.size), columns, -values, [-floor])
        result, sizes = programme.maximise(
            *shape_rule.objective(programme, box), time_limit=time_limit
        )
        seconds += result.seconds
        if result.outcome != "optimal":
            return missing(result.outcome, first_value)
    x = result.x
    policy = _policy(x, rows, box, g, gain, pairs)
    _, _, scale = box.read(x)
    value = float(2.0 * policy.w.sum())
    return Certificate(
        "optimal",
        product,
        structure,
        shape_rule.name,
        value,
        scale,
        policy,
        first_value,
        seconds,
        sizes,
    )
