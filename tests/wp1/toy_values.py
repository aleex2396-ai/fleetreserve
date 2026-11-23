"""Hindsight, corner-tree and affine values of a box on a toy fleet (plan sections 2 and 4, T4).

These are test helpers for toy sizes (a few cars, at most 5 service hours). They are not the
product's causal programmes, which are a later work package. Each function solves one linear
programme through `solve_lp`.

A fleet is a list of `gpoly.Car`; the service hours `hours` carry the box, and every hour of a car
that is not a service hour is a recovery hour, free within the car's own limits. The box is
`y = m + W zeta`, `zeta` in `[-1, 1]^hours`: with a fixed centre `c` and widths `w` its corners
are `c +- lam * w` and the functions ending in `_scale` return the largest `lam`; the functions
ending in `_value` choose centre and widths freely and return the largest `sum(hi - lo)`.

- Block product (plan section 1): the widths are all equal and the centre is free; the functions
  named `block_*` return the largest half-width `lam` over the centre and the half-width together.
  The centre is declared before the first request and is not adapted afterwards. The diagonal block
  rule uses non-negative gains (a car's own half-width in each hour), as plan section 2 states.
- Hindsight: every corner path is met with the path known in advance. Two routes: the exact
  set functions of `gpoly` (`hindsight_scale`, `hindsight_value`), and one power plan per corner
  path with the recovery hours written out (`hindsight_scale_by_paths`, `hindsight_value_by_paths`).
- Tree: a causal rule meets every corner path; the power in hour `t` may depend on the requests
  up to `t` only (`tree_scale`).
- Affine: car `i` draws `u_i = g_i + G_i zeta` with `G_i` lower-triangular on the service hours,
  or diagonal when `diagonal=True`, with `sum g_i = m` and `sum G_i = W`, and the car's limits
  held for every `zeta` (a row `a (g + G zeta) <= b` is `a g + |a G|_1 <= b`; plan T2(b)). The
  recovery hours are eliminated: with `R` recovery hours the energy drawn in the service hours must
  lie in `[e_low - high * R, e_high - low * R]`, which is the plan's form of the set (section 2).
"""

from itertools import combinations, product

import numpy as np

from fleetreserve.gpoly import fleet_b, fleet_p
from fleetreserve.lp import OTHER_SECONDS, solve_lp


class _Model:
    """A linear programme built variable by variable; `maximise` solves it through `solve_lp`."""

    def __init__(self):
        self.bounds = []
        self.ub = []  # (terms, rhs): sum of coefficient * variable <= rhs
        self.eq = []  # (terms, rhs): sum of coefficient * variable == rhs

    def var(self, low=0.0, high=None):
        self.bounds.append((low, high))
        return len(self.bounds) - 1

    def le(self, terms, rhs):
        self.ub.append((terms, rhs))

    def ge(self, terms, rhs):
        self.ub.append(([(j, -c) for j, c in terms], -rhs))

    def equal(self, terms, rhs):
        self.eq.append((terms, rhs))

    def maximise(self, terms):
        n = len(self.bounds)
        cost = np.zeros(n)
        for j, c in terms:
            cost[j] -= c

        def matrix(rows):
            if not rows:
                return None, None
            a = np.zeros((len(rows), n))
            for r, (row_terms, _) in enumerate(rows):
                for j, c in row_terms:
                    a[r, j] += c
            return a, np.array([rhs for _, rhs in rows], dtype=float)

        a_ub, b_ub = matrix(self.ub)
        a_eq, b_eq = matrix(self.eq)
        result = solve_lp(
            cost,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=self.bounds,
            time_limit=OTHER_SECONDS,
            method="simplex",
        ).require_optimal()
        return -result.objective


def _subsets(hours):
    hours = sorted(hours)
    for size in range(1, len(hours) + 1):
        for chosen in combinations(hours, size):
            yield frozenset(chosen)


def hindsight_scale(cars, hours, centre, widths):
    """Largest `lam`, from the fleet's set functions (plan T1(b), the box faces)."""
    scale = float("inf")
    for subset in _subsets(hours):
        c = sum(centre[t] for t in subset)
        w = sum(widths[t] for t in subset)
        if w > 0:
            scale = min(scale, (fleet_b(cars, subset) - c) / w, (c - fleet_p(cars, subset)) / w)
    return scale


def hindsight_value(cars, hours):
    """Largest `sum(hi - lo)` with the box inside the fleet set (faces `p <= lo`, `hi <= b`)."""
    m = _Model()
    lo = {t: m.var(None, None) for t in hours}
    hi = {t: m.var(None, None) for t in hours}
    for t in hours:
        m.le([(lo[t], 1.0), (hi[t], -1.0)], 0.0)
    for subset in _subsets(hours):
        m.ge([(lo[t], 1.0) for t in subset], fleet_p(cars, subset))
        m.le([(hi[t], 1.0) for t in subset], fleet_b(cars, subset))
    return m.maximise([(hi[t], 1.0) for t in hours] + [(lo[t], -1.0) for t in hours])


def block_hindsight_scale(cars, hours):
    """Largest `lam` such that some centre `m` puts the box `m +- lam` in the fleet set (faces)."""
    m = _Model()
    lam = m.var(0.0, None)
    centre = {t: m.var(None, None) for t in hours}
    for subset in _subsets(hours):
        size = float(len(subset))
        m.ge([(centre[t], 1.0) for t in subset] + [(lam, -size)], fleet_p(cars, subset))
        m.le([(centre[t], 1.0) for t in subset] + [(lam, size)], fleet_b(cars, subset))
    return m.maximise([(lam, 1.0)])


def _car_energy_rows(m, car, powers):
    """The car's total energy limits; `powers` holds its variables in every one of its hours."""
    terms = [(j, 1.0) for j in powers]
    m.le(terms, car.e_high)
    m.ge(terms, car.e_low)


def _path_plan(m, cars, hours, request):
    """Power variables of one corner path with the future known; `request[t]` is the terms
    (variable, coefficient) and constant of the request in hour t."""
    plan = {}
    for i, car in enumerate(cars):
        powers = {t: m.var(car.low, car.high) for t in car.hours}
        _car_energy_rows(m, car, list(powers.values()))
        plan[i] = powers
    for t in hours:
        terms, constant = request[t]
        m.equal(
            [(plan[i][t], 1.0) for i, car in enumerate(cars) if t in car.hours]
            + [(j, -c) for j, c in terms],
            constant,
        )


def hindsight_scale_by_paths(cars, hours, centre, widths):
    """Largest `lam`, from one power plan per corner path (recovery hours written out)."""
    m = _Model()
    lam = m.var(0.0, None)
    for path in product((-1, 1), repeat=len(hours)):
        request = {t: ([(lam, z * widths[t])], centre[t]) for t, z in zip(hours, path, strict=True)}
        _path_plan(m, cars, hours, request)
    return m.maximise([(lam, 1.0)])


def hindsight_value_by_paths(cars, hours):
    """Largest `sum(hi - lo)`, from one power plan per corner path."""
    m = _Model()
    lo = {t: m.var(None, None) for t in hours}
    hi = {t: m.var(None, None) for t in hours}
    for t in hours:
        m.le([(lo[t], 1.0), (hi[t], -1.0)], 0.0)
    for path in product((-1, 1), repeat=len(hours)):
        request = {
            t: ([(lo[t] if z < 0 else hi[t], 1.0)], 0.0) for t, z in zip(hours, path, strict=True)
        }
        _path_plan(m, cars, hours, request)
    return m.maximise([(hi[t], 1.0) for t in hours] + [(lo[t], -1.0) for t in hours])


def block_hindsight_scale_by_paths(cars, hours):
    """Largest block `lam`, from one power plan per corner path (centre free, one for all paths)."""
    m = _Model()
    lam = m.var(0.0, None)
    centre = {t: m.var(None, None) for t in hours}
    for path in product((-1, 1), repeat=len(hours)):
        request = {
            t: ([(centre[t], 1.0), (lam, float(z))], 0.0) for t, z in zip(hours, path, strict=True)
        }
        _path_plan(m, cars, hours, request)
    return m.maximise([(lam, 1.0)])


def tree_scale(cars, hours, centre=None, widths=None):
    """Largest `lam` for which a causal rule meets every corner path.

    With `centre` and `widths` left out this is the block product: widths all 1, and the centre is
    one free variable per hour, the same at every node (it is declared in advance).

    One power per car at every node of the corner tree (a node is the sequence of corners chosen
    so far); the recovery hours are decided at the leaves, where the whole path is known.
    """
    m = _Model()
    lam = m.var(0.0, None)
    n = len(hours)
    block = centre is None
    if block:
        centre = {t: m.var(None, None) for t in hours}
        widths = dict.fromkeys(hours, 1.0)
    power = {}
    for depth in range(1, n + 1):
        t = hours[depth - 1]
        for node in product((-1, 1), repeat=depth):
            for i, car in enumerate(cars):
                if t in car.hours:
                    power[node, i] = m.var(car.low, car.high)
            m.equal(
                [(power[node, i], 1.0) for i, car in enumerate(cars) if t in car.hours]
                + [(lam, -node[-1] * widths[t])]
                + ([(centre[t], -1.0)] if block else []),
                0.0 if block else centre[t],
            )
    for leaf in product((-1, 1), repeat=n):
        for i, car in enumerate(cars):
            on_path = [power[leaf[:d], i] for d in range(1, n + 1) if (leaf[:d], i) in power]
            recovery = [m.var(car.low, car.high) for t in car.hours if t not in hours]
            _car_energy_rows(m, car, on_path + recovery)
    return m.maximise([(lam, 1.0)])


def _affine(cars, hours, diagonal, centre=None, widths=None, block=False):
    """The affine programme: largest `lam` for a given shape, the block `lam` (centre free and
    every width equal to `lam`), else the largest `sum(2 W)`. The diagonal block rule has
    non-negative gains."""
    m = _Model()
    centre_var = {t: m.var(None, None) for t in hours}
    width_var = {t: m.var(0.0, None) for t in hours}
    gain = {}  # (car, hour t, hour s): the coefficient G[t, s] of car i
    level = {}  # (car, hour t): g[t]
    for i, car in enumerate(cars):
        present = [t for t in hours if t in car.hours]
        recovery = len(car.hours - set(hours))
        absolute = {}
        for t in present:
            level[i, t] = m.var(None, None)
            for s in hours:
                if (s <= t) and (not diagonal or s == t):
                    gain[i, t, s] = m.var(0.0 if block and diagonal else None, None)
                    absolute[t, s] = m.var(0.0, None)
                    m.le([(gain[i, t, s], 1.0), (absolute[t, s], -1.0)], 0.0)
                    m.le([(gain[i, t, s], -1.0), (absolute[t, s], -1.0)], 0.0)
            row = [(absolute[t, s], 1.0) for s in hours if (t, s) in absolute]
            m.le([(level[i, t], 1.0)] + row, car.high)
            m.ge([(level[i, t], 1.0)] + [(j, -c) for j, c in row], car.low)
        # The energy drawn in the service hours: its slope in zeta[s] is the column sum of G.
        slopes = []
        for s in hours:
            column = [gain[i, t, s] for t in present if (i, t, s) in gain]
            if column:
                slope = m.var(0.0, None)
                m.le([(j, 1.0) for j in column] + [(slope, -1.0)], 0.0)
                m.le([(j, -1.0) for j in column] + [(slope, -1.0)], 0.0)
                slopes.append(slope)
        total = [(level[i, t], 1.0) for t in present]
        m.le(total + [(j, 1.0) for j in slopes], car.e_high - car.low * recovery)
        m.ge(total + [(j, -1.0) for j in slopes], car.e_low - car.high * recovery)
    for t in hours:
        m.equal(
            [(level[i, t], 1.0) for i, car in enumerate(cars) if (i, t) in level]
            + [(centre_var[t], -1.0)],
            0.0,
        )
        for s in hours:
            column = [(gain[i, t, s], 1.0) for i in range(len(cars)) if (i, t, s) in gain]
            if column:
                m.equal(column + ([(width_var[t], -1.0)] if s == t else []), 0.0)
            elif s == t:
                m.equal([(width_var[t], 1.0)], 0.0)  # no car present: nothing can be offered
    if block:
        lam = m.var(0.0, None)
        for t in hours:
            m.equal([(width_var[t], 1.0), (lam, -1.0)], 0.0)
        return m.maximise([(lam, 1.0)])
    if centre is None:
        return m.maximise([(width_var[t], 2.0) for t in hours])
    lam = m.var(0.0, None)
    for t in hours:
        m.equal([(centre_var[t], 1.0)], centre[t])
        m.equal([(width_var[t], 1.0), (lam, -widths[t])], 0.0)
    return m.maximise([(lam, 1.0)])


def affine_scale(cars, hours, centre, widths):
    """Largest `lam` for a causal (lower-triangular) affine rule."""
    return _affine(cars, hours, False, centre, widths)


def diagonal_scale(cars, hours, centre, widths):
    """Largest `lam` for a diagonal affine rule."""
    return _affine(cars, hours, True, centre, widths)


def diagonal_value(cars, hours):
    """Largest `sum(hi - lo)` for a diagonal affine rule."""
    return _affine(cars, hours, True)


def affine_value(cars, hours):
    """Largest `sum(hi - lo)` for a causal (lower-triangular) affine rule."""
    return _affine(cars, hours, False)


def block_diagonal_scale(cars, hours):
    """Largest block `lam` for a diagonal affine rule (centre free)."""
    return _affine(cars, hours, True, block=True)


def block_affine_scale(cars, hours):
    """Largest block `lam` for a causal (lower-triangular) affine rule (centre free)."""
    return _affine(cars, hours, False, block=True)
