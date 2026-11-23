"""Optimality of LP (24) and LP (27) (`lp24`) against two independent dense formulations.

1. The paper's form (Al Taha et al. (2025), programmes (24) and (27) with Lemma 1 written as
   printed): per car, a full matrix `Gamma_i` on every hour (also the hours where the base set
   has no power), a vector `gamma_i` and a non-negative multiplier matrix `Lambda_i` (one row per
   car row, one column per base-set row), with `Lambda_i H0 = H_i Gamma_i` and
   `Lambda_i h0 <= h_i - H_i gamma_i`. The base set's rows are `H0 = [I; -I; 1']`,
   `h0 = [power; 0; energy]`, on every hour. LP (24): `sum_i Gamma_i = alpha I` on every hour;
   LP (27): the trace of `sum_i Gamma_i` over the hours where the base set has power (the
   module's reading 2: the other diagonal entries are not limited). Built entry by entry with
   numpy arrays and solved by `solve_lp`'s dual simplex. On `N_DAYS` seeded made-up days with 2
   or 3 service hours, each one-way and two-way (`N_SOLVED` with a non-trivial base set), the
   values agree within `TOLERANCE` (relative to `max(1, value)`).
2. The earlier analysis's programme, written by hand from its text (its sets
   `{0 <= u_k <= c for k < d, u_k = 0 for k >= d, delta sum u <= e}` with no needed energy, its
   base set the average of `min(c, e / delta)` before the deadline and of `e`, and its dual form
   of Lemma 1 with one `w >= 0` per base-set step and one `s >= 0` per car row), with
   `delta = 1 h`, on the fleet `EARLIER_FLEET` of three cars over three steps. The same cars
   written as this repository's rows (power limits, the energy row of a `gpoly.Car` and a need
   row of 0) give the same `alpha` within `TOLERANCE`; both are `EARLIER_ALPHA`.
"""

import functools

import numpy as np
import pytest
from lp24_cases import cases

from fleetreserve.base_sets import day_average
from fleetreserve.carsets import gpoly_rows, stack
from fleetreserve.gpoly import Car
from fleetreserve.lp import OTHER_SECONDS, solve_lp
from fleetreserve.lp24 import inner_set

SEED = 19
N_DAYS = 20
N_SOLVED = 40
TOLERANCE = 1e-6
# (deadline in steps, energy ceiling kWh, power cap kW), three steps of one hour.
EARLIER_FLEET = ((3, 2.0, 1.5), (2, 3.5, 2.0), (1, 0.5, 3.0))
EARLIER_STEPS = 3
EARLIER_ALPHA = 2.3125


class _Dense:
    """A dense programme: variables with bounds, rows `A x <= b` and `A x = b`."""

    def __init__(self):
        self.low, self.high = [], []
        self.ub, self.eq = [], []

    def var(self, shape, low=-np.inf, high=np.inf):
        start = len(self.low)
        count = int(np.prod(shape))
        self.low += [low] * count
        self.high += [high] * count
        return np.arange(start, start + count).reshape(shape)

    def le(self, terms, rhs):
        self.ub.append((terms, rhs))

    def equal(self, terms, rhs):
        self.eq.append((terms, rhs))

    def maximise(self, terms):
        n = len(self.low)
        cost = np.zeros(n)
        for j, c in terms:
            cost[j] -= c

        def matrix(rows):
            a = np.zeros((len(rows), n))
            for r, (row_terms, _) in enumerate(rows):
                for j, c in row_terms:
                    a[r, j] += c
            return a, np.array([rhs for _, rhs in rows], dtype=float)

        a_ub, b_ub = matrix(self.ub)
        a_eq, b_eq = matrix(self.eq) if self.eq else (None, None)
        result = solve_lp(
            cost,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=list(zip(self.low, self.high, strict=True)),
            time_limit=OTHER_SECONDS,
            method="simplex",
        ).require_optimal()
        return -result.objective


def paper_programme(rows, base, programme):
    """Programme (24) or (27) with Lemma 1 as printed (module text, part 1)."""
    hours = rows.n_service + 1
    h0_matrix = np.vstack([np.eye(hours), -np.eye(hours), np.ones((1, hours))])
    h0 = np.concatenate([base.power, np.zeros(hours), [base.energy]])
    m = _Dense()
    alpha = m.var(1, 0.0)[0] if programme == "24" else None
    gains = []
    for i in range(rows.n_cars):
        mine = np.flatnonzero(rows.car == i)
        h_i, b_i = rows.coefficients[mine], rows.bounds[mine]
        gamma = m.var(hours)
        gain = m.var((hours, hours))
        lam = m.var((mine.size, h0_matrix.shape[0]), 0.0)
        gains.append(gain)
        for k in range(mine.size):
            for col in range(hours):  # (Lambda_i H0)[k, col] = (H_i Gamma_i)[k, col]
                terms = [(lam[k, j], h0_matrix[j, col]) for j in range(h0_matrix.shape[0])]
                terms += [(gain[t, col], -h_i[k, t]) for t in range(hours)]
                m.equal([(j, c) for j, c in terms if c != 0], 0.0)
            terms = [(lam[k, j], h0[j]) for j in range(h0.size) if h0[j] != 0]
            terms += [(gamma[t], h_i[k, t]) for t in range(hours) if h_i[k, t] != 0]
            m.le(terms, b_i[k])
    if programme == "24":
        for t in range(hours):
            for col in range(hours):
                terms = [(g[t, col], 1.0) for g in gains]
                m.equal(terms + ([(alpha, -1.0)] if t == col else []), 0.0)
        return m.maximise([(alpha, 1.0)])
    return m.maximise([(g[t, t], 1.0) for g in gains for t in base.columns])


@functools.cache
def _cases():
    found = [(rows, base) for _, _, rows, base in cases(SEED, N_DAYS, (2, 3)) if base is not None]
    assert len(found) == N_SOLVED
    return found


@pytest.mark.parametrize("programme", ["24", "27"])
def test_sparse_equals_the_paper_form(programme):
    differ = []
    for rows, base in _cases():
        sparse = inner_set(rows, base, programme)
        assert sparse.outcome == "optimal"
        dense = paper_programme(rows, base, programme)
        if abs(sparse.value - dense) > TOLERANCE * max(1.0, abs(dense)):
            differ.append((sparse.value, dense))
    assert not differ, differ


def earlier_programme(fleet, steps):
    """The earlier analysis's programme (24) as read from its text (module text, part 2)."""
    delta = 1.0
    deadline = [d for d, _, _ in fleet]
    energy = [e for _, e, _ in fleet]
    cap = [c for *_, c in fleet]
    upper = np.array(
        [[min(c, e / delta) if k < d else 0.0 for k in range(steps)] for d, e, c in fleet]
    )
    power, base_energy = upper.mean(axis=0), float(np.mean(energy))
    cols = [k for k in range(steps) if power[k] > 0]
    q = len(cols)
    m = _Dense()
    alpha = m.var(1, 0.0)[0]
    layout = []

    def support_at_most(g_terms, rhs_terms, rhs):
        w = m.var(q, 0.0)
        s = m.var(1, 0.0)[0]
        for b in range(q):
            m.le(g_terms[b] + [(w[b], -1.0), (s, -delta)], 0.0)
        m.le([(w[b], power[cols[b]]) for b in range(q)] + [(s, base_energy)] + rhs_terms, rhs)

    for i in range(len(fleet)):
        d = deadline[i]
        g = m.var(d)
        big = m.var((d, q))
        layout.append((big, d))
        for a in range(d):
            support_at_most([[(big[a, b], 1.0)] for b in range(q)], [(g[a], 1.0)], cap[i])
            support_at_most([[(big[a, b], -1.0)] for b in range(q)], [(g[a], -1.0)], 0.0)
        support_at_most(
            [[(big[a, b], delta) for a in range(d)] for b in range(q)],
            [(g[a], delta) for a in range(d)],
            energy[i],
        )
    for k in range(steps):
        present = [(big, d) for big, d in layout if d > k]
        for b, j in enumerate(cols):
            terms = [(big[k, b], 1.0) for big, _ in present]
            if k == j:
                terms.append((alpha, -1.0))
            if terms:
                m.equal(terms, 0.0)
    return m.maximise([(alpha, 1.0)])


def test_agrees_with_the_earlier_analysis_on_a_tiny_fleet():
    earlier = earlier_programme(EARLIER_FLEET, EARLIER_STEPS)
    n = EARLIER_STEPS - 1
    cars = [Car(frozenset(range(d)), 0.0, c, 0.0, e) for d, e, c in EARLIER_FLEET]
    rows = stack([gpoly_rows(car, n) for car in cars])
    base = day_average(rows)
    ours = inner_set(rows, base, "24")
    print(f"earlier form alpha {earlier:.9f}; this repository's rows alpha {ours.alpha:.9f}")
    assert ours.alpha == pytest.approx(earlier, abs=TOLERANCE)
    assert ours.alpha == pytest.approx(EARLIER_ALPHA, abs=TOLERANCE)
