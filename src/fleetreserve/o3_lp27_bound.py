"""An upper bound on the largest box over LP (27)'s optimal solutions (plan sections 1, 10 and 11
(O3)), by a linear relaxation of the joint programme and branch and bound on the map's entries.

Notation as in `o3_lp27`: service hours `S` (`n` of them), base-set hours `J` (`q`), `U0 = {v :
0 <= v_b <= c_b, sum_b v_b <= E}`, LP (27)'s optimal trace `T` and optimal face `F` (the
solutions with trace at least `T`); a solution gives the map `[p, P] = sum_i [gamma_i, Gamma_i]`.
The largest box held by a map, `box(p, P)`, is the largest box of the product whose every corner
`k` (hour 0 free, sign vector `zeta_k`) is `(p + P v_k)_S` for some `v_k` in `U0`. The number
bounded is `Y* = max over F of box(p, P)`.

The joint programme. `Y*` is the optimum of: maximise the box's value over a solution of `F`, a
box (centre `m`, half-widths `w >= 0`) and profiles `v_k` in `U0`, subject to

    (C)  p_S + P_S v_k = m + zeta_k o w   for every corner k.

(C) couples the set and the box through the products `P[t, b] v_kb` of a variable of the set and
a variable of the box: it is bilinear, not jointly linear (`o3_lp27`'s module text).

The relaxation `R(l, u, K)`, for numbers `l <= u` (one pair per entry `P[t, b]`, `t` in `S`) and a
set `K` of corner paths: replace each product by a variable `z_ktb` and keep, for every corner of
`K`, service hour and base-set hour, the four McCormick rows

    z >= l v,   z >= u v + c (P - u),   z <= u v,   z <= l v + c (P - l)

with `l <= P[t, b] <= u`, (C) for the corners of `K` written as `p_S + sum_b z_k.b = m +
zeta_k o w`, LP (27)'s rows with its trace at least `T` (the face, all cars), and the box inside
the fleet's set seen on `S` (hour 0 and the recovery hours free) in one of two forms. The set
functions: `m(A) + w(A) <= b(A)` and `m(A) - w(A) >= p(A)` for every non-empty set `A` of
service hours, `p` and `b` those of `exact.day_functions` (plan section 4, T1: the fleet's set
seen on `S` has only the faces `p(A) <= u(A) <= b(A)`; the largest and least energy a corner of
the box draws in `A` are `m(A) + w(A)` and `m(A) - w(A)`), which hold every corner of the box
in the set. The cars' powers (`lifted`, used when every corner is kept and there are at most
`ALL_CORNERS`): per corner `k` of `K` and car `i`, powers `u_ik` (all hours) meeting car `i`'s
rows, with `sum_i u_ik` equal to the corner on `S`. Every row is linear, so `R` is a linear
programme; the set functions' form is solved by the dual simplex method, the cars' powers' by
the interior-point method.

Claim. If every map of `F` with `l <= P <= u` entrywise is a map of `R`'s face, then
`opt(R(l, u, K)) >= Y*(l, u)`, the largest box held by such a map, for every set `K`. Proof: take
such a solution, its largest box and the profiles of its corners, and set `z_ktb = P[t, b] v_kb`
for the corners of `K`. Each McCormick row is the product of two numbers that are at least 0,
expanded: `(P - l) v`, `(u - P)(c - v)`, `(u - P) v`, `(P - l)(c - v)`, using `l <= P <= u` and
`0 <= v <= c`. (C) holds with `z = P v`. For every corner `k`, `gamma_i + Gamma_i v_k` lies in
car `i`'s set `U_i` (Lemma 1 of Al Taha et al. (2025), `v_k` in `U0`), so `u_ik = gamma_i +
Gamma_i v_k` meets the cars' powers' rows, and their sum `p + P v_k` lies in the fleet's set
(Theorem 2), so its service hours meet the faces of T1 and the set functions' rows hold. So the
point is feasible for `R` with the same value. Leaving corners out of `K` removes rows, so a
smaller `K` gives a bound at least as large. []

The map ranges, one car at a time. LP (27)'s rows do not couple the cars (Remark 5 of Al Taha et
al. (2025): it splits into one programme per car, maximise `Tr(Gamma_i)` subject to
`gamma_i + Gamma_i U0` inside `U_i`, and `T` is the sum of the cars' optima `T_i`), so `F` is the
product of the cars' optimal faces `F_i` and every map of `F` is `P = sum_i Gamma_i` with
`Gamma_i` in `F_i`. Hence the least and largest `P[t, b]` over `F` are the sums over the cars of
the least and largest `Gamma_i[t, b]` over `F_i`: per car, its own LP (27) gives `T_i`, then two
small programmes per entry. An hour in which a car's power has one value (its `power-high` bound
at most its `power-low` bound: a car gone) needs none: its rows `u_t <= h` and `-u_t <= -h'` with
`h <= h'` give, through Lemma 1, `gamma_i(t) + max_U0 Gamma_i[t] v <= h` and
`-gamma_i(t) + max_U0 (-Gamma_i[t] v) <= -h'`, so the range of `Gamma_i[t] v` over `U0` is at most
`h - h' <= 0`; `U0` holds 0 and `min(c_b, E)` times the unit vector of each `b` in `J` (both
positive), so `Gamma_i[t, b] = 0` on the car's whole set of solutions. Root: `l = L`, `u = U`
those sums, so every map of `F` is covered and `opt(R(L, U, K)) >= Y*`.

The corners kept. A day with at most `ALL_CORNERS` corner paths keeps them all. Otherwise a node
starts from the corners its parent ended with (the root: the two constant paths, every hour low or
every hour high) and, for up to `rounds` relaxations, after each relaxation solves the small
programme of every corner path left out (`corner_gap`: the least sum of `|p_S + sum_b z_.b - m -
zeta o w|` over `v` in `U0` and `z` in the McCormick rows at the relaxation's own `P`, `l`, `u`)
and adds the `n` paths with the largest gaps above `CORNER_SLACK * max(1, ceiling)`. When no path
is left out by more than that, the relaxation's point meets every corner and its value is
`opt(R(l, u, all paths))`. Each relaxation solved is a bound by the claim, and the node's bound is
the last one solved; the paths added change only how tight it is, never whether it is a bound.

Branch and bound. A node is a box `[l, u]` of map entries with its corners; the maps of `F` inside
the nodes of any split of `[L, U]` are all of `F`'s maps, so `Y*` is at most the largest of the
nodes' bounds. A node is split in two at one entry (the entry with the largest product of its
width and of `max_k |z_ktb - P[t, b] v_kb|` over the corners kept, at the node's solution; at
that solution's value, kept inside the middle 80% of the width); a child whose first relaxation
is infeasible holds no map of `F` and is dropped; a child whose first relaxation ends in any
other outcome but optimal keeps its parent's bound (its maps are among the parent's) and is not
split again; a later relaxation of a node not ending optimal leaves the node's last bound. An
incumbent, the largest box held by a known optimal solution, is at most `Y*`: the caller's
largest found, and at each node split the box reached by up to `INCUMBENT_ROUNDS` rounds of
`o3_lp27`'s ascent (step 5) from the box of the map of `F` nearest the node's solution in the
sum of absolute differences of `p_S` and `P_S`; each map used only on the optimal face
(`o3_lp27.on_face`). Which optimal solution of a relaxation the solver returns does not change
the bound, only the incumbents and the splits: on `tests/wp14/test_o3_lp27_bound.py`'s
`RAISE_DAY` (block, 51 nodes) the cars' powers' form by the interior-point method met a map
holding the ceiling 8.786, and the set functions' form did not (largest held 8.6968 by the dual
simplex method, 8.7857 by the interior-point method), so a day with at most `ALL_CORNERS`
corners keeps the cars' powers' form. A node whose bound is at most the incumbent plus
`CLOSE * max(1, ceiling)` is closed. When the nodes run out or `nodes` nodes have been solved,
the bound is the largest of the incumbent and the open nodes' bounds: every map of `F` lies in a
closed node (its boxes at most the incumbent plus the closing slack) or an open one. Best first:
the open node with the largest bound is split next.

Numbers and their margins. Every number here is a solver's answer, so `F` is taken in the bound's
programmes as the solutions with trace at least `T - BOUND_FACE * max(1, T)` (a superset of `F`
however `T` is rounded) and each car's face in the map ranges as its solutions with trace at least
`T_i - BOUND_FACE * max(1, T_i)`; the summed `L` and `U` are widened by
`MAP_MARGIN * max(1, |.|)`, and `MARGIN * max(1, ceiling)` is added to the bound printed. HiGHS's
feasibility and optimality tolerances are 1e-7; each margin is ten times that or more. The bound
is valid to the solver's tolerances; it is not a proof in exact arithmetic.

A day by hand (`tests/wp14/test_o3_lp27_bound.py`, `HAND_DAY`). One service hour; cars A and B
each leave after hour 1 (departure 2), with rate 3 kW, rooms 1 and 3 kWh, rho 0, no recovery hour:
each car draws exactly its room over hours 0 and 1, so car `i`'s set is `u_0 + u_1 = e_i`,
`0 <= u_t <= e_i`, and the ceiling is `e_A + e_B = 4`. The day-average base set is
`U0 = {v >= 0, v_0, v_1 <= 2, v_0 + v_1 <= 2}` (corners (0, 0), (2, 0), (0, 2)). For car `i`,
`u_0 + u_1 = e_i` for every `v` in `U0` forces `Gamma_i = [[a, -c], [-a, c]]` and
`gamma_i(0) + gamma_i(1) = e_i`, with trace `a + c`; `u_0 = gamma_i(0) + a v_0 - c v_1` in
`[0, e_i]` at the three corners gives `2 (a + c) <= e_i`, so `T_i = e_i / 2`, and the face is
`c` in `[0, e_i / 2]`, `a = e_i / 2 - c`, `gamma_i(0) = 2 c`. Its hour-1 entries are
`Gamma_i[1, 0] = c - e_i / 2` in `[-e_i / 2, 0]` and `Gamma_i[1, 1] = c` in `[0, e_i / 2]`, so
`L = (-2, 0)` and `U = (0, 2)`. With `C = c_A + c_B` in `[0, 2]`, `P[1] = (C - 2, C)` and the hour-1
range over `U0`'s corners is `max(0, 2 C) - min(0, 2 (C - 2)) = 4`: every optimal solution holds
the box 4, the ceiling, so the bound is 4 (closed).

The value. `min(ceiling, bound)`: the ceiling is itself a bound (every certified set lies in the
fleet's set, hour 0 free per corner as in the ceiling), and a node is split only while its bound,
taken at most the ceiling, is above the incumbent plus the closing slack. Outcomes: `CLOSED` (the
value is within the closing slack of the incumbent), `CEILING` (not closed, and the relaxations'
bound is at or above the ceiling, so the ceiling is the value), `OPEN` (not closed: the node
budget ran out first; the value is the last bound proved, below the ceiling). When a root
programme (a map range, a car's own LP (27) or the root's first relaxation) ends in any outcome
but optimal, no bound is proved: the value is missing (None) with that programme's outcome,
never the ceiling put in its place.

Not bounded here: the smallest box over `F` from below (a min over `F` of a max over boxes; a
lower bound needs one box held by every optimal solution's set at once, itself a bilinear
programme). Every programme has a time limit: the map ranges, the cars' own LP (27), the
nearest-map programmes and the ascent's face programmes the limit `others`, the relaxations, the
corner gaps and the boxes the limit `box`; a timeout is an outcome as above (in the ascent, it
ends the ascent at the box reached and is counted as failed).
"""

import heapq
from dataclasses import dataclass, field

import numpy as np

from fleetreserve import o3_lp27
from fleetreserve.base_sets import BaseSet
from fleetreserve.carsets import CarRows, FleetRows, stack
from fleetreserve.exact import SetFunctions
from fleetreserve.lp24 import _build as lp27_programme
from fleetreserve.products import Product, corner_paths
from fleetreserve.sparse_programme import SparseProgramme

BOUND_FACE = 1e-6  # relative, on LP (27)'s trace and each car's: the floor of the bound's faces
MAP_MARGIN = 1e-6  # relative widening of the root's map ranges
MARGIN = 1e-6  # relative to max(1, ceiling), added to the bound
CLOSE = 1e-5  # relative to max(1, ceiling): a node this close to the incumbent is closed
MIDDLE = 0.1  # a split stays this share of the width away from either end
ALL_CORNERS = 16  # a day with at most this many corner paths keeps them all
CORNER_SLACK = 1e-7  # relative to max(1, ceiling): a corner missed by no more is not added
INCUMBENT_ROUNDS = 3  # ascent rounds (`o3_lp27`, step 5) from each nearest map's box
OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
CLOSED, OPEN, CEILING = "closed", "open", "ceiling"


@dataclass(frozen=True)
class LargestBound:
    """`value` (kWh; None when a root programme was not solved), `outcome` (`CLOSED`, `CEILING`,
    `OPEN`, or that programme's outcome, module text), `incumbent` (the largest box held by an
    optimal solution that the branch and bound knew, the caller's included; None if none),
    `nodes` (nodes whose bound was solved), `failed` (programmes of the bound ended in an outcome
    but optimal or infeasible), `proved` (the bound before the ceiling is taken, None when a root
    programme was not solved), `seconds` (summed solve time), `relaxations` (relaxations solved,
    every round of every node) and `corners` (the corner paths of the last node solved)."""

    value: float | None
    outcome: str
    incumbent: float | None
    nodes: int
    failed: int
    proved: float | None
    seconds: float = field(compare=False)
    relaxations: int = 0
    corners: int = 0


@dataclass(frozen=True)
class Point:
    """A relaxation's solution: the map `P_S` `(n, q)`, `p_S` `(n,)`, the box `m`, `w` `(n,)`
    and `violations` `(n, q)`, the largest `|z_ktb - P[t, b] v_kb|` over the corners kept."""

    maps: np.ndarray
    centre: np.ndarray
    m: np.ndarray
    w: np.ndarray
    violations: np.ndarray


@dataclass(frozen=True)
class _Node:
    low: np.ndarray
    high: np.ndarray
    bound: float
    point: Point
    corners: tuple[int, ...]


def _car(rows: FleetRows, i: int) -> FleetRows:
    """Car `i`'s rows alone."""
    mine = rows.car == i
    return stack(
        [
            CarRows(
                rows.n_service, rows.coefficients[mine], rows.bounds[mine], tuple(rows.kinds[mine])
            )
        ]
    )


def fixed_hours(car: FleetRows) -> np.ndarray:
    """`(n_service + 1,)`: whether the car's `power-high` bound in the hour is at most its
    `power-low` bound, so its power there has one value (module text)."""
    hours = car.n_service + 1
    high, low = np.full(hours, np.inf), np.full(hours, -np.inf)
    kinds = np.asarray(car.kinds)
    for kind, out, pick in (("power-high", high, np.minimum), ("power-low", low, np.maximum)):
        for k in np.flatnonzero(kinds == kind):
            t = int(np.flatnonzero(car.coefficients[k])[0])
            out[t] = pick(out[t], car.bounds[k] / car.coefficients[k, t])
    return high <= low


class _Simplex(o3_lp27._SimplexProgramme):
    """A copy of a programme, solved by the dual simplex method, whose matrices are built at the
    first solve and kept: one car's face (solved once per map entry and direction) and each
    relaxation (whose interior-point solves were seen to time out at the box limit on made-up
    days of 8 service hours where the dual simplex method took 1 to 3 s)."""

    def __init__(self, programme: SparseProgramme):
        super().__init__()
        self.n_variables = programme.n_variables
        self._low, self._high = list(programme._low), list(programme._high)
        self._blocks = {sense: list(blocks) for sense, blocks in programme._blocks.items()}
        self._n_rows = dict(programme._n_rows)
        self._built = {}

    def _matrix(self, sense):
        if sense not in self._built:
            self._built[sense] = super()._matrix(sense)
        return self._built[sense]


def map_ranges(rows: FleetRows, base: BaseSet, time_limit: float):
    """(outcome, L, U, seconds): the least and largest `P[t, b]` (`t` in `S`, `b` in `J`) over the
    bound's face, as sums over the cars of each car's own (module text), widened by
    `MAP_MARGIN`; L and U are None unless every programme ended optimal."""
    n, q = rows.n_service, base.columns.size
    low, high, seconds = np.zeros((n, q)), np.zeros((n, q)), 0.0
    for i in range(rows.n_cars):
        car = _car(rows, i)
        lp, _, _, gain, (columns, values) = lp27_programme(car, base, "27")
        result, spent = _solve(lp, (columns, values), time_limit)
        seconds += spent
        if result.outcome != OPTIMAL:
            return result.outcome, None, None, seconds
        trace = float(-result.objective)
        floor = trace - BOUND_FACE * max(1.0, abs(trace))
        lp.rows("le", np.zeros(columns.size), columns, -values, [-floor])
        lp = _Simplex(lp)
        fixed = fixed_hours(car)
        for t in range(1, n + 1):
            if fixed[t]:
                continue
            for b in range(q):
                for sign, out in ((1.0, high), (-1.0, low)):
                    column = gain[0, t, b]
                    result, spent = _solve(lp, (np.array([column]), np.array([sign])), time_limit)
                    seconds += spent
                    if result.outcome != OPTIMAL:
                        return result.outcome, None, None, seconds
                    out[t - 1, b] += float(result.x[column])
    return (
        OPTIMAL,
        low - MAP_MARGIN * np.maximum(1.0, np.abs(low)),
        high + MAP_MARGIN * np.maximum(1.0, np.abs(high)),
        seconds,
    )


def _rows_between(lp, columns, low, high):
    columns = np.asarray(columns).reshape(-1)
    k = np.arange(columns.size)
    lp.rows("le", k, columns, np.ones(columns.size), np.asarray(high, float).reshape(-1))
    lp.rows("le", k, columns, -np.ones(columns.size), -np.asarray(low, float).reshape(-1))


def _mccormick(lp, z, map_s, v, low, high, power):
    """The four McCormick rows of `z = P v` per corner, service hour and base-set hour; `map_s`
    a `(n, q)` array of variables, or of numbers (then the rows bound `z` alone)."""
    corners, n, q = z.shape
    count = corners * n * q
    zs = z.reshape(-1)
    vs = np.repeat(v[:, None, :], n, axis=1).reshape(-1)
    lo = np.broadcast_to(low, (corners, n, q)).reshape(-1)
    hi = np.broadcast_to(high, (corners, n, q)).reshape(-1)
    c = np.broadcast_to(power, (corners, n, q)).reshape(-1)
    k = np.arange(count)
    one = np.ones(count)
    two = np.concatenate([k, k])
    lp.rows("le", two, np.concatenate([vs, zs]), np.concatenate([lo, -one]), np.zeros(count))
    lp.rows("le", two, np.concatenate([zs, vs]), np.concatenate([one, -hi]), np.zeros(count))
    if np.issubdtype(np.asarray(map_s).dtype, np.integer):
        ps = np.broadcast_to(map_s, (corners, n, q)).reshape(-1)
        three = np.concatenate([k, k, k])
        lp.rows("le", three, np.concatenate([vs, ps, zs]), np.concatenate([hi, c, -one]), hi * c)
        lp.rows("le", three, np.concatenate([zs, vs, ps]), np.concatenate([one, -lo, -c]), -lo * c)
    else:
        given = np.broadcast_to(np.asarray(map_s, float), (corners, n, q)).reshape(-1)
        lp.rows("le", two, np.concatenate([vs, zs]), np.concatenate([hi, -one]), c * (hi - given))
        lp.rows("le", two, np.concatenate([zs, vs]), np.concatenate([one, -lo]), c * (given - lo))


def _fleet_rows(lp, functions: SetFunctions, m, width):
    """The box inside the fleet's set seen on `S`: `m(A) + w(A) <= b(A)` and
    `m(A) - w(A) >= p(A)` for every non-empty `A` (module text)."""
    n = functions.n_service
    masks = np.arange(1, 2**n)
    sets, hours = np.nonzero((masks[:, None] >> np.arange(n)) & 1)
    for sign, rhs in ((1.0, functions.upper[masks]), (-1.0, -functions.lower[masks])):
        lp.rows(
            "le",
            np.concatenate([sets, sets]),
            np.concatenate([np.asarray(m)[hours], np.asarray(width)[hours]]),
            np.concatenate([np.full(sets.size, sign), np.ones(sets.size)]),
            rhs,
        )


def _fleet_corners(lp, rows: FleetRows, m, width, paths):
    """Each corner kept on the service hours as `sum_i u_ik`, `u_ik` in car `i`'s set, hour 0
    and the recovery hours free (module text)."""
    corners, n = paths.shape
    cars, hours = rows.n_cars, n + 1
    u = lp.variables((corners, cars, hours))
    k_index, t_index = np.nonzero(rows.coefficients)
    a = rows.coefficients[k_index, t_index]
    for k in range(corners):
        lp.rows("le", k_index, u[k, rows.car[k_index], t_index], a, rows.bounds)
    r = np.arange(corners * n)
    service = u[:, :, 1:].transpose(0, 2, 1).reshape(-1)
    lp.rows(
        "eq",
        np.concatenate([np.repeat(r, cars), r, r]),
        np.concatenate([service, np.tile(m, corners), np.tile(width, corners)]),
        np.concatenate([np.ones(corners * n * cars), -np.ones(corners * n), -paths.reshape(-1)]),
        np.zeros(corners * n),
    )


def lifted(n: int, count: int) -> bool:
    """Whether a relaxation of a day of `n` service hours with `count` corners kept holds the
    box in the fleet's set through each car's powers per corner (every corner kept and at most
    `ALL_CORNERS` of them) rather than through the set functions (module text)."""
    return count == 2**n <= ALL_CORNERS


def relaxation(
    rows, base, trace, product: Product, functions: SetFunctions, low, high, corners, time_limit
):
    """`R(low, high, corners)` of the module text (`corners`: indices into
    `products.corner_paths`): (outcome, value, `Point`, seconds); value and the point are None
    unless the outcome is optimal."""
    n, q = rows.n_service, base.columns.size
    lp, _, _, p_s, map_s = o3_lp27._face(rows, base, trace, BOUND_FACE)
    _rows_between(lp, map_s, low, high)
    m, width, objective = o3_lp27._box_columns(lp, product, n)
    paths = corner_paths(n)[np.asarray(corners, int)]
    count = paths.shape[0]
    power = base.power[base.columns]
    v = lp.variables((count, q), 0.0, np.broadcast_to(power, (count, q)))
    lp.rows(
        "le",
        np.repeat(np.arange(count), q),
        v.reshape(-1),
        np.ones(count * q),
        np.full(count, base.energy),
    )
    z = lp.variables((count, n, q))
    _mccormick(lp, z, map_s, v, low, high, power)
    r = np.arange(count * n)
    lp.rows(
        "eq",
        np.concatenate([r, np.repeat(r, q), r, r]),
        np.concatenate(
            [np.tile(p_s, count), z.reshape(-1), np.tile(m, count), np.tile(width, count)]
        ),
        np.concatenate(
            [np.ones(count * n), np.ones(count * n * q), -np.ones(count * n), -paths.reshape(-1)]
        ),
        np.zeros(count * n),
    )
    if lifted(n, count):
        _fleet_corners(lp, rows, m, width, paths)
        result, seconds = _solve(lp, objective, time_limit)
    else:
        _fleet_rows(lp, functions, m, width)
        result, seconds = _solve(_Simplex(lp), objective, time_limit)
    if result.outcome != OPTIMAL:
        return result.outcome, None, None, seconds
    x = result.x
    maps, profiles, products = x[map_s], x[v], x[z]
    violations = np.abs(products - maps[None, :, :] * profiles[:, None, :]).max(axis=0)
    point = Point(maps, x[p_s], x[m], np.maximum(x[width], 0.0), violations)
    return OPTIMAL, float(-result.objective), point, seconds


def corner_gap(point: Point, low, high, base: BaseSet, path, time_limit):
    """(outcome, gap, seconds): the least `sum_t |p_t + sum_b z_tb - m_t - path_t w_t|` over `v`
    in `U0` and `z` in the McCormick rows at the point's own map (module text); 0 when the
    relaxation meets the corner `path` at the point."""
    n, q = point.maps.shape
    power = base.power[base.columns]
    lp = o3_lp27._SimplexProgramme()
    v = lp.variables((1, q), 0.0, power[None, :])
    lp.rows("le", np.zeros(q), v.reshape(-1), np.ones(q), [base.energy])
    z = lp.variables((1, n, q))
    _mccormick(lp, z, point.maps, v, low, high, power)
    plus = lp.variables(n, 0.0, np.inf)
    minus = lp.variables(n, 0.0, np.inf)
    t = np.arange(n)
    lp.rows(
        "eq",
        np.concatenate([np.repeat(t, q), t, t]),
        np.concatenate([z.reshape(-1), plus, minus]),
        np.concatenate([np.ones(n * q), -np.ones(n), np.ones(n)]),
        point.m + np.asarray(path, float) * point.w - point.centre,
    )
    result, seconds = _solve(lp, (np.concatenate([plus, minus]), -np.ones(2 * n)), time_limit)
    if result.outcome != OPTIMAL:
        return result.outcome, None, seconds
    return OPTIMAL, max(0.0, float(result.objective)), seconds


def _solve(lp: SparseProgramme, objective, time_limit):
    """`lp.maximise` (the interior-point method for a `SparseProgramme`, the nearest map and a
    car's own LP (27); the dual simplex method for a car's face in `map_ranges`, a relaxation and
    a corner's gap), and the dual simplex method (`o3_lp27`'s) when that ends with a solver
    error; (result, both solves' seconds)."""
    result, _ = lp.maximise(*objective, time_limit=time_limit)
    if result.outcome != o3_lp27.SOLVER_ERROR:
        return result, result.seconds
    again, _ = o3_lp27._SimplexProgramme.maximise(lp, *objective, time_limit=time_limit)
    return again, result.seconds + again.seconds


def nearest_box(rows, base, trace, product: Product, point: Point, others, box):
    """(the largest box (`o3_lp27.CornerBox`) held by the map of LP (27)'s optimal face nearest
    the point's `p_S`, `P_S` in the sum of absolute differences, or None when a programme did
    not end optimal or the map is off the face; seconds)."""
    lp, gamma, gain, p_s, map_s = o3_lp27._face(rows, base, trace)
    target = np.concatenate([np.asarray(point.centre).reshape(-1), point.maps.reshape(-1)])
    columns = np.concatenate([p_s.reshape(-1), map_s.reshape(-1)])
    gap = lp.variables(columns.size, 0.0, np.inf)
    k = np.arange(columns.size)
    for sign in (1.0, -1.0):  # gap >= |x - target|
        lp.rows(
            "le",
            np.concatenate([k, k]),
            np.concatenate([columns, gap]),
            np.concatenate([np.full(columns.size, sign), -np.ones(columns.size)]),
            sign * target,
        )
    result, seconds = _solve(lp, (gap, -np.ones(gap.size)), others)
    if result.outcome != OPTIMAL:
        return None, seconds
    p, P = o3_lp27._maps(result.x, rows, base, gamma, gain)
    if not o3_lp27.on_face(P, base, trace):
        return None, seconds
    found, spent = o3_lp27.largest_box(p, P, base, product, box)
    return (found if found.outcome == OPTIMAL else None), seconds + spent


def incumbent_box(rows, base, trace, product: Product, point: Point, others, box):
    """(the largest box reached by up to `INCUMBENT_ROUNDS` rounds of `o3_lp27`'s ascent from
    `nearest_box`'s, or None when that was not found; seconds; whether every programme ended
    optimal) (module text)."""
    near, seconds = nearest_box(rows, base, trace, product, point, others, box)
    if near is None:
        return None, seconds, False
    spent = {"lp27_ascent": 0.0}
    outcome, reached, _ = o3_lp27._ascend(
        rows, base, trace, product, near, others, box, spent, INCUMBENT_ROUNDS
    )
    return reached.value, seconds + spent["lp27_ascent"], outcome == OPTIMAL


def _split(node: _Node):
    point = node.point
    score = point.violations * (node.high - node.low)
    t, b = np.unravel_index(int(np.argmax(score)), score.shape)
    width = node.high[t, b] - node.low[t, b]
    at = min(
        max(point.maps[t, b], node.low[t, b] + MIDDLE * width), node.high[t, b] - MIDDLE * width
    )
    below, above = node.high.copy(), node.low.copy()
    below[t, b], above[t, b] = at, at
    return (node.low, below), (above, node.high)


class _Tally:
    def __init__(self):
        self.failed, self.seconds, self.relaxations = 0, 0.0, 0


def _node_bound(day, low, high, corners, rounds, slack, tally):
    """A node's bound (module text): (outcome of its first relaxation, value, point, corners);
    value and point None unless that outcome is optimal."""
    rows, base, trace, product, functions, box = day
    n = rows.n_service
    every = 2**n
    if every <= ALL_CORNERS:
        corners = tuple(range(every))
    found = None
    rounds = max(1, rounds)
    for round_ in range(rounds):
        outcome, value, point, spent = relaxation(
            rows, base, trace, product, functions, low, high, corners, box
        )
        tally.seconds += spent
        tally.relaxations += 1
        if outcome != OPTIMAL:
            if found is None:
                return outcome, None, None, corners
            tally.failed += 1
            break
        found = (value, point, corners)
        if len(corners) == every or round_ == rounds - 1:
            break
        gaps = []
        for k in sorted(set(range(every)) - set(corners)):
            ended, gap, spent = corner_gap(point, low, high, base, corner_paths(n)[k], box)
            tally.seconds += spent
            if ended != OPTIMAL:
                tally.failed += 1
            elif gap > slack:
                gaps.append((-gap, k))
        if not gaps:
            break
        corners = tuple(sorted(set(corners) | {k for _, k in sorted(gaps)[:n]}))
    value, point, corners = found
    return OPTIMAL, value, point, corners


def largest_bound(
    rows: FleetRows,
    base: BaseSet,
    trace: float,
    product: Product,
    functions: SetFunctions,
    ceiling: float | None,
    incumbent: float | None,
    nodes: int,
    others: float,
    box: float,
    ranges: tuple | None = None,
    rounds: int = 1,
) -> LargestBound:
    """The bound of the module text for one product, with at most `nodes` nodes (at least the
    root), up to `rounds` relaxations per node and the time limits `others` and `box` per
    programme. `ranges`: `map_ranges`'s answer for this day when the caller has it (it does not
    depend on the product; its seconds are then the caller's), else it is solved here."""
    scale = max(1.0, ceiling if ceiling is not None else 0.0)
    cap = np.inf if ceiling is None else ceiling
    slack = CORNER_SLACK * scale
    best = incumbent
    tally = _Tally()

    def result(value, outcome, used, proved, corners):
        return LargestBound(
            value,
            outcome,
            best,
            used,
            tally.failed,
            proved,
            tally.seconds,
            tally.relaxations,
            len(corners),
        )

    if ranges is None:
        outcome, low, high, spent = map_ranges(rows, base, others)
        tally.seconds += spent
    else:
        outcome, low, high, _ = ranges
    root_corners = (0, 2**rows.n_service - 1)
    if outcome != OPTIMAL:
        tally.failed += 1
        return result(None, outcome, 0, None, root_corners)
    day = (rows, base, trace, product, functions, box)
    outcome, value, point, corners = _node_bound(day, low, high, root_corners, rounds, slack, tally)
    if outcome != OPTIMAL:
        tally.failed += 1
        return result(None, outcome, 0, None, root_corners)
    used, last = 1, corners
    heap = [(-value, 0, _Node(low, high, value, point, corners))]
    stuck: list[float] = []
    count = 1
    while heap:
        top = min(-heap[0][0], cap)
        level = 0.0 if best is None else best
        if top <= level + CLOSE * scale:
            break
        if used + 2 > nodes:  # a split solves two nodes
            break
        _, _, node = heapq.heappop(heap)
        held, spent, ended = incumbent_box(rows, base, trace, product, node.point, others, box)
        tally.seconds += spent
        if not ended:
            tally.failed += 1
        if held is not None and (best is None or held > best):
            best = held
        for low_child, high_child in _split(node):
            outcome, value, point, corners = _node_bound(
                day, low_child, high_child, node.corners, rounds, slack, tally
            )
            used += 1
            last = corners
            if outcome == INFEASIBLE:
                continue
            if outcome != OPTIMAL:
                tally.failed += 1
                stuck.append(node.bound)
                continue
            count += 1
            heapq.heappush(
                heap, (-value, count, _Node(low_child, high_child, value, point, corners))
            )
    level = 0.0 if best is None else best
    open_bounds = [-entry[0] for entry in heap] + stuck
    proved = max([level, *open_bounds]) + MARGIN * scale
    value = proved if ceiling is None else min(proved, ceiling)
    if value <= level + CLOSE * scale:
        outcome = CLOSED
    elif value < proved:
        outcome = CEILING
    else:
        outcome = OPEN
    return result(value, outcome, used, proved, last)
