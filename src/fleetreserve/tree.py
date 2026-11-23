"""The corner tree (plan section 4, T3): the best causal rule of a product on the 2^n corner paths.

A node of depth `k` (0 to `n`) is the sequence of corners `zeta_1 .. zeta_k` (each -1 or +1) seen so
far; it holds every car's power in hour `k`, decided knowing only those corners. The root (depth 0)
holds hour 0, which carries no request. Node numbering: depth `k`, index `j` (0 to `2^k - 1`, bit
`l` of `j` set when `zeta_{l+1} = +1`) is node `2^k - 1 + j`; its parent is depth `k - 1`, index
`j mod 2^(k-1)`.

Variables per car and node: the power `u` and the stored energy `x` since the decision time
(`x(node) = x(parent) + u(node)`). A car's rows (`carsets`) are placed at the nodes of their last
hour: a row on one hour bounds `u` there, a row on the running sum `sum_{s<=t} u_s` bounds `x` at
depth `t` (the need row, with the car's recovery eliminated, bounds `x` at every leaf), and any
other row is written out over the node's ancestors. At every node of depth `k >= 1`:
`sum_i u_i(node) = m_k + zeta_k w_k` for the product's box (`products`).

One sparse programme through `lp.solve_lp` (interior point with crossover, plan section 13), with
the 300-second limit for other programmes by default. Any outcome but optimal is a missing result.
The shape among boxes of the same value is the solver's own vertex; no second stage is offered
here.
"""

from dataclasses import dataclass

import numpy as np

from fleetreserve.carsets import FleetRows
from fleetreserve.lp import OTHER_SECONDS, Outcome
from fleetreserve.products import Product, add_box
from fleetreserve.sparse_programme import Sizes, SparseProgramme


def node_count(n_service: int) -> int:
    return 2 ** (n_service + 1) - 1


def node_depths(n_service: int) -> np.ndarray:
    return np.concatenate([np.full(2**k, k) for k in range(n_service + 1)])


def node_corners(n_service: int) -> np.ndarray:
    """The last corner of each node, -1 or +1 (0 for the root)."""
    corners = [np.zeros(1)]
    for k in range(1, n_service + 1):
        j = np.arange(2**k)
        corners.append(np.where((j >> (k - 1)) & 1, 1.0, -1.0))
    return np.concatenate(corners)


def ancestors(n_service: int) -> np.ndarray:
    """`(nodes, n + 1)`: the node's ancestor at each depth up to its own (itself at its depth),
    -1 below it."""
    table = np.full((node_count(n_service), n_service + 1), -1)
    for k in range(n_service + 1):
        j = np.arange(2**k)
        nodes = 2**k - 1 + j
        for t in range(k + 1):
            table[nodes, t] = 2**t - 1 + (j & (2**t - 1))
    return table


def path_nodes(path) -> np.ndarray:
    """The nodes along a corner path (`n` entries of -1 or +1): root first, leaf last."""
    bits = (np.asarray(path) > 0).astype(int)
    index = np.concatenate([[0], np.cumsum(bits << np.arange(bits.size))])
    return 2 ** np.arange(bits.size + 1) - 1 + index


@dataclass(frozen=True)
class TreePolicy:
    """The tree rule: `node_powers[i, node]` is car `i`'s power at that node; `m`, `w` the box."""

    m: np.ndarray
    w: np.ndarray
    node_powers: np.ndarray

    @property
    def n_service(self) -> int:
        return int(self.m.size)

    def powers(self, path) -> np.ndarray:
        """`(cars, n + 1)`: every car's power in hours 0 to `n` along a corner path."""
        return self.node_powers[:, path_nodes(path)]

    def request(self, path) -> np.ndarray:
        return self.m + self.w * np.asarray(path, dtype=float)


@dataclass(frozen=True)
class TreeResult:
    """`value`, `scale` and `policy` are None unless `outcome` is "optimal"."""

    outcome: Outcome
    product: Product
    value: float | None
    scale: float | None
    policy: TreePolicy | None
    seconds: float
    sizes: Sizes


def _classify(rows: FleetRows):
    """Rows on one hour (`"power"`), on a running sum from hour 0 (`"stored"`), or other."""
    a = rows.coefficients
    nonzero = a != 0
    count = nonzero.sum(axis=1)
    last = rows.last_hours
    first_value = a[np.arange(a.shape[0]), last]
    hours = np.arange(a.shape[1])
    prefix = (hours[None, :] <= last[:, None]) & (count[:, None] > 0)
    is_prefix = np.all(np.where(prefix, a == first_value[:, None], a == 0), axis=1)
    single = count == 1
    return single, is_prefix & ~single, ~(single | is_prefix) & (count > 0), last, first_value


def _bounds(cars, hours, car, hour, coefficient, bound):
    """Per (car, hour) the interval `coefficient * v <= bound` cuts, intersected."""
    low = np.full((cars, hours), -np.inf)
    high = np.full((cars, hours), np.inf)
    limit = bound / coefficient
    up = coefficient > 0
    np.minimum.at(high, (car[up], hour[up]), limit[up])
    np.maximum.at(low, (car[~up], hour[~up]), limit[~up])
    return low, high


def corner_tree(rows: FleetRows, product: Product, time_limit: float = OTHER_SECONDS):
    """The best causal rule of `product` on the corner tree of the sets `rows`."""
    n, cars = rows.n_service, rows.n_cars
    hours = n + 1
    nodes = node_count(n)
    depth = node_depths(n)
    programme = SparseProgramme()
    box = add_box(programme, product, n)

    single, stored, other, last, lead = _classify(rows)
    zero_rows = np.flatnonzero((rows.coefficients != 0).sum(axis=1) == 0)
    u_low, u_high = _bounds(
        cars, hours, rows.car[single], last[single], lead[single], rows.bounds[single]
    )
    x_low, x_high = _bounds(
        cars, hours, rows.car[stored], last[stored], lead[stored], rows.bounds[stored]
    )
    u = programme.variables((cars, nodes), u_low[:, depth], u_high[:, depth])
    x = programme.variables((cars, nodes), x_low[:, depth], x_high[:, depth])

    # x(node) - x(parent) - u(node) = 0; at the root x - u = 0.
    table = ancestors(n)
    parent = np.where(depth > 0, table[np.arange(nodes), np.maximum(depth - 1, 0)], -1)
    eq = np.arange(cars * nodes).reshape(cars, nodes)
    has_parent = np.broadcast_to(parent >= 0, (cars, nodes))
    programme.rows(
        "eq",
        np.concatenate([eq.reshape(-1), eq.reshape(-1), eq[has_parent]]),
        np.concatenate([x.reshape(-1), u.reshape(-1), x[:, np.maximum(parent, 0)][has_parent]]),
        np.concatenate([np.ones(eq.size), -np.ones(eq.size), -np.ones(int(has_parent.sum()))]),
        np.zeros(eq.size),
    )

    # Requests at every node of depth >= 1: sum_i u_i - m_k - zeta_k w_k = c_k.
    request_nodes = np.arange(1, nodes)
    k = depth[request_nodes]
    corner = node_corners(n)[request_nodes]
    r = np.arange(request_nodes.size)
    row = [np.repeat(r, cars), r]
    column = [u[:, request_nodes].T.reshape(-1), box.width[k - 1]]
    value = [np.ones(r.size * cars), -corner * box.width_coefficient[k - 1]]
    if box.centre is not None:
        row.append(r)
        column.append(box.centre[k - 1])
        value.append(-np.ones(r.size))
    programme.rows(
        "eq",
        np.concatenate(row),
        np.concatenate(column),
        np.concatenate(value),
        box.centre_constant[k - 1],
    )

    # Other rows, written out over the ancestors of every node of their last hour.
    for index in np.flatnonzero(other):
        t_last = last[index]
        at = 2**t_last - 1 + np.arange(2**t_last)
        t_nonzero = np.flatnonzero(rows.coefficients[index])
        rows_here = np.repeat(np.arange(at.size), t_nonzero.size)
        columns = u[rows.car[index]][table[at][:, t_nonzero]].reshape(-1)
        values = np.tile(rows.coefficients[index, t_nonzero], at.size)
        programme.rows("le", rows_here, columns, values, np.full(at.size, rows.bounds[index]))

    if zero_rows.size:  # 0 <= b: left to the solver, which finds a negative b infeasible
        empty = np.array([], int)
        programme.rows("le", empty, empty, np.array([]), rows.bounds[zero_rows])

    result, sizes = programme.maximise(*box.value_terms(), time_limit=time_limit)
    if result.outcome != "optimal":
        return TreeResult(result.outcome, product, None, None, None, result.seconds, sizes)
    m, w, scale = box.read(result.x)
    policy = TreePolicy(m, w, result.x[u])
    return TreeResult(
        "optimal", product, float(2.0 * w.sum()), scale, policy, result.seconds, sizes
    )
