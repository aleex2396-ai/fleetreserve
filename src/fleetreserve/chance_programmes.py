"""The two chance-constrained sizing programmes of Lunde et al. (2025), ALSO-X and CVaR, on one
set of samples (plan section 10: "ALSO-X, CVaR (Lunde et al. 2025)"; `chance_sizing` says what a
sample is here and how the result is used).

A sample `j` (one of `N`) has a threshold `theta_j` in `[0, top]`: the largest scale at which it
holds. A scale `beta` violates sample `j` when `beta > theta_j`. The chance constraint asks that at
most a share `epsilon` of the samples is violated; both programmes find the largest `beta` in
`[0, top]` they accept. `top` is the largest scale allowed (`chance_sizing`: the largest grid
scale).

ALSO-X (Lunde et al. (2025), Algorithm 1, as the design note
kept outside the repository, part A, read it from the paper's page 8;
the paper was not re-read for this module). The relaxed programme at budget `q` (`relaxed`), the
binaries of their programme (4) relaxed to `[0, 1]`:

    maximise beta  over beta in [0, top], y_j in [0, 1]
    subject to beta - M y_j <= theta_j  (every j),   sum_j y_j <= q,

with `M = top`, the largest value `beta - theta_j` can take (the paper takes `M` as the
quantity's largest value). The budget is bisected (`also_x`) between a lower end 0 and an upper
end `epsilon N`: at the midpoint the relaxed programme is solved; when its `beta` violates at
most `floor(epsilon N)` samples (counted in whole samples, a violation being
`beta > theta_j + VIOLATION_TOLERANCE`) the lower end moves up to the midpoint, otherwise the
upper end comes down; the bisection stops when the ends are no more than `DELTA` (the paper's
1e-5) apart. The result is the `beta` of the relaxed programme at the final lower end: that of
the last midpoint that moved it, or, when none did, of the programme at budget 0 (no sample may be
violated, so `beta = min(top, min_j theta_j)`), solved after the bisection.

CVaR (Lunde et al. (2025), programme (5), as the design note read it; `cvar_programme`):

    maximise beta  over beta in [0, top], b <= 0, zeta_j free
    subject to zeta_j >= beta - theta_j,  zeta_j >= b  (every j),
               (1/N) sum_j zeta_j <= (1 - epsilon) b.

Taking `zeta_j = max(b, beta - theta_j)` the last row reads
`b + (1/epsilon) mean_j max(beta - theta_j - b, 0) <= 0`: the conditional value at risk of the
excess `beta - theta_j` at level `epsilon`, the mean of its largest share `epsilon`, is at most
0. So `beta` is at most the mean of the smallest share `epsilon` of the thresholds (the last
sample of that share counted in part), and the programme's `beta` is that mean, capped at `top`.
The paper's bound `b <= 0` is kept as written; it never binds, since `zeta_j >= b` makes the
last row `b <= (1 - epsilon) b`, which already holds only for `b <= 0` (with the bound raised
to 5 the 300 made-up sets of the agreement test kept every scale).

What the two give in this one-variable form (tested on made-up samples, not proved here): ALSO-X
tends to the `(floor(epsilon N) + 1)`-th smallest threshold, the largest scale that violates at
most `floor(epsilon N)` samples; its relaxation does nothing distinctive with a single decision
variable, and its result is that sample quantile to within the bisection's gap (about `DELTA`
divided by the number of samples below it). CVaR gives the mean of the smallest share `epsilon`
of the thresholds, never above ALSO-X's.

Every solve goes through `sparse_programme.SparseProgramme.maximise` (`lp.solve_lp`, the
interior-point method with crossover) with the caller's time limit. A solve that does not end
optimal is a recorded outcome: in the bisection a midpoint whose programme is not optimal moves
the upper end down, as a midpoint that violates too many samples does (the lower end stays where
a solved programme put it); when the programme at the final lower end (the budget-0 programme,
solved after the bisection) or the CVaR programme is not optimal, the result is `None`
(`ChanceScale.value`), and `chance_sizing` abstains. Every outcome is kept in
`ChanceScale.outcomes`, in the order solved.
"""

import math
from dataclasses import dataclass

import numpy as np

from fleetreserve.sparse_programme import SparseProgramme

DELTA = 1e-5  # samples; the bisection's stopping gap (Lunde et al. 2025, Algorithm 1)
VIOLATION_TOLERANCE = 1e-9  # scale; beta above theta_j by more than this violates sample j
COUNT_TOLERANCE = 1e-9  # samples; floor(epsilon N + this), so float rounding of epsilon N cannot
# drop a whole sample (for example 0.1 * 30 = 3.0000000000000004 is 3, and so is 2.9999999999999996)


@dataclass(frozen=True)
class ChanceScale:
    """What one sizing programme gave: `value` the largest scale it accepts (None when the solve
    that gives it did not end optimal) and the outcome of every solve, in the order solved."""

    value: float | None
    outcomes: tuple[str, ...]


def _thresholds(thresholds, top: float) -> np.ndarray:
    theta = np.array(thresholds, dtype=float).reshape(-1)
    if theta.size == 0:
        raise ValueError("a sizing programme needs at least one sample")
    if not (math.isfinite(top) and top > 0):
        raise ValueError(f"top must be a positive finite scale, got {top!r}")
    if not np.all(np.isfinite(theta)) or theta.min() < 0 or theta.max() > top:
        raise ValueError(f"every threshold must be finite and in [0, {top}]")
    return theta


def _check(thresholds, epsilon: float, top: float) -> np.ndarray:
    theta = _thresholds(thresholds, top)
    if not (0 < epsilon < 1):
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon!r}")
    return theta


def allowed(n: int, epsilon: float) -> int:
    """The most samples of `n` that may be violated: `floor(epsilon n)` (module text)."""
    return math.floor(epsilon * n + COUNT_TOLERANCE)


def violations(beta: float, thresholds) -> int:
    """The number of samples `beta` violates (module text)."""
    return int(np.sum(beta > np.asarray(thresholds, dtype=float) + VIOLATION_TOLERANCE))


def relaxed(thresholds, budget: float, top: float, time_limit: float):
    """The relaxed ALSO-X programme at `budget` (module text): the solver's result and `beta`
    (None unless optimal)."""
    theta = _thresholds(thresholds, top)
    if not (math.isfinite(budget) and budget >= 0):
        raise ValueError(f"the budget must be a finite number >= 0, got {budget!r}")
    n = theta.size
    programme = SparseProgramme()
    (beta,) = programme.variables((1,), 0.0, top)
    y = programme.variables((n,), 0.0, 1.0)
    rows = np.arange(n)
    programme.rows(
        "le",
        np.concatenate([rows, rows]),
        np.concatenate([np.full(n, beta), y]),
        np.concatenate([np.ones(n), np.full(n, -top)]),
        theta,
    )
    programme.rows("le", np.zeros(n, int), y, 1.0, [budget])
    result, _ = programme.maximise([beta], [1.0], time_limit)
    return result, (None if result.outcome != "optimal" else float(result.x[beta]))


def also_x(thresholds, epsilon: float, top: float, time_limit: float) -> ChanceScale:
    """ALSO-X's scale: the bisection of the module text."""
    theta = _check(thresholds, epsilon, top)
    most = allowed(theta.size, epsilon)
    low, high = 0.0, epsilon * theta.size
    best = None
    outcomes = []
    while high - low > DELTA:
        middle = (low + high) / 2
        result, beta = relaxed(theta, middle, top, time_limit)
        outcomes.append(result.outcome)
        if beta is not None and violations(beta, theta) <= most:
            low, best = middle, beta
        else:
            high = middle
    if best is None:  # the lower end never moved: the programme at budget 0
        result, best = relaxed(theta, low, top, time_limit)
        outcomes.append(result.outcome)
    return ChanceScale(best, tuple(outcomes))


def cvar_programme(thresholds, epsilon: float, top: float, time_limit: float):
    """The CVaR programme of the module text: the solver's result and `beta` (None unless
    optimal)."""
    theta = _check(thresholds, epsilon, top)
    n = theta.size
    programme = SparseProgramme()
    (beta,) = programme.variables((1,), 0.0, top)
    (b,) = programme.variables((1,), -np.inf, 0.0)
    zeta = programme.variables((n,))
    rows = np.arange(n)
    # beta - zeta_j <= theta_j
    programme.rows(
        "le",
        np.concatenate([rows, rows]),
        np.concatenate([np.full(n, beta), zeta]),
        np.concatenate([np.ones(n), -np.ones(n)]),
        theta,
    )
    # b - zeta_j <= 0
    programme.rows(
        "le",
        np.concatenate([rows, rows]),
        np.concatenate([np.full(n, b), zeta]),
        np.concatenate([np.ones(n), -np.ones(n)]),
        np.zeros(n),
    )
    # (1/N) sum_j zeta_j - (1 - epsilon) b <= 0
    programme.rows(
        "le",
        np.zeros(n + 1, int),
        np.concatenate([zeta, [b]]),
        np.concatenate([np.full(n, 1.0 / n), [-(1.0 - epsilon)]]),
        [0.0],
    )
    result, _ = programme.maximise([beta], [1.0], time_limit)
    return result, (None if result.outcome != "optimal" else float(result.x[beta]))


def cvar(thresholds, epsilon: float, top: float, time_limit: float) -> ChanceScale:
    """CVaR's scale: one solve of `cvar_programme`."""
    result, beta = cvar_programme(thresholds, epsilon, top, time_limit)
    return ChanceScale(beta, (result.outcome,))
