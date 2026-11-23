"""The two sizing programmes of `chance_programmes` (ALSO-X and CVaR, Lunde et al. 2025) on
made-up samples.

1. Hand cases: ten samples with thresholds 0.2, 0.3, 0.5, 0.5, 0.6, 0.7, 0.8, 1, 1, 1 at
   epsilon 0.1 give ALSO-X 0.3 (one sample may be violated: the second smallest threshold, to
   within the bisection's gap) and CVaR 0.2 (the smallest tenth of the thresholds is the one
   sample 0.2); fifteen samples 0.1, 0.3 and thirteen of 0.5 give CVaR (0.1 + 0.5 * 0.3) / 1.5
   (a share epsilon of 15 samples is 1.5 samples, the second counted in half); five samples
   (no sample may be violated) give ALSO-X the smallest threshold exactly, by the programme at
   budget 0 after a bisection in which no midpoint moved the lower end.
2. The bisection: its number of solves is that of halving `epsilon N` to at most `DELTA`, plus
   one when the lower end never moved; `allowed` counts whole samples, with the tolerance that
   keeps `(1 - 0.9) * 10` at 1; a scale above a threshold by 1e-10 does not violate it, by 1e-8
   it does; the relaxed programme spends its budget on the excess above the lowest thresholds,
   is bounded by `top`, and its `M` is `top` (thresholds 0.2, 0.2, 0.6 and budget 0.1 give 0.25
   with `top` 1 and 0.3 with `top` 2).
3. Timeouts are recorded outcomes: with every solve timing out ALSO-X and CVaR give None and
   every outcome is "timeout"; with only the midpoints timing out, the upper end comes down at
   each and ALSO-X gives the budget-0 programme's scale.
4. Refusals: no sample, a threshold outside [0, top] or not finite, epsilon outside (0, 1), a
   negative budget.
5. On 300 seeded made-up sample sets (1 to 60 samples of grid thresholds, ties included), the
   relaxed ALSO-X programme at a seeded budget and the CVaR programme give the scale of an
   independent dense build of the same programme with `scipy.optimize.linprog`, CVaR also that
   of the Rockafellar-Uryasev form (b free, b + mean((excess - b)+) / epsilon <= 0) and of the
   closed form (the mean of the smallest share epsilon of the thresholds), to 1e-7; on 150 of
   them the whole ALSO-X bisection gives the `(floor(epsilon N) + 1)`-th smallest threshold to
   within `DELTA` below it, so that both snap to the same grid scale, and CVaR is never above
   ALSO-X after snapping.
"""

import math
import random

import numpy as np
import pytest
from scipy.optimize import linprog

from fleetreserve import chance_programmes
from fleetreserve.chance_programmes import DELTA, allowed, also_x, cvar, relaxed, violations
from fleetreserve.chance_sizing import snap
from fleetreserve.dial_splits import GRID
from fleetreserve.lp import LpResult
from fleetreserve.sparse_programme import SparseProgramme

LIMIT = 30.0
HAND = (0.2, 0.3, 0.5, 0.5, 0.6, 0.7, 0.8, 1.0, 1.0, 1.0)


def steps(n: int, epsilon: float) -> int:
    """The number of midpoints the bisection solves from [0, epsilon n] to a gap <= DELTA."""
    count, width = 0, epsilon * n
    while width > DELTA:
        width /= 2
        count += 1
    return count


# --- 1. hand cases -------------------------------------------------------------------------------


def test_the_hand_case_of_ten_samples():
    found = also_x(HAND, 0.1, 1.0, LIMIT)
    assert 0.3 - DELTA <= found.value <= 0.3 + 1e-9
    assert snap(found.value, GRID, 1e-4) == 0.3
    assert set(found.outcomes) == {"optimal"} and len(found.outcomes) == steps(10, 0.1)
    assert cvar(HAND, 0.1, 1.0, LIMIT).value == pytest.approx(0.2, abs=1e-9)
    assert cvar(HAND, 0.1, 1.0, LIMIT).outcomes == ("optimal",)


def test_cvar_counts_the_last_sample_of_its_share_in_part():
    theta = [0.5] * 13 + [0.3, 0.1]
    assert cvar(theta, 0.1, 1.0, LIMIT).value == pytest.approx((0.1 + 0.5 * 0.3) / 1.5, abs=1e-9)


def test_with_no_sample_allowed_also_x_is_the_smallest_threshold_by_the_budget_zero_programme():
    theta = (0.6, 0.4, 0.8, 0.6, 1.0)
    found = also_x(theta, 0.1, 1.0, LIMIT)
    assert found.value == pytest.approx(0.4, abs=1e-12)
    assert len(found.outcomes) == steps(5, 0.1) + 1


# --- 2. the bisection and its counts -------------------------------------------------------------


def test_allowed_counts_whole_samples_with_the_tolerance():
    assert [allowed(n, 0.1) for n in (5, 9, 10, 19, 20, 30, 799, 800)] == [0, 0, 1, 1, 2, 3, 79, 80]
    epsilon = 1 - 0.9  # 0.09999999999999998
    assert math.floor(epsilon * 10) == 0
    assert allowed(10, epsilon) == 1


def test_a_violation_is_a_scale_above_the_threshold_by_more_than_the_tolerance():
    assert violations(0.3 + 1e-10, [0.3, 0.5]) == 0
    assert violations(0.3 + 1e-8, [0.3, 0.5]) == 1
    assert violations(0.6, [0.3, 0.5, 0.6]) == 2


def test_the_relaxed_programme_spends_its_budget_above_the_lowest_thresholds():
    # two samples at 0.2 and one at 0.6: beta - 0.2 on each of two samples costs the budget 0.1
    result, beta = relaxed([0.2, 0.2, 0.6], 0.1, 1.0, LIMIT)
    assert result.outcome == "optimal"
    assert beta == pytest.approx(0.25, abs=1e-9)
    assert relaxed([0.2, 0.2, 0.6], 0.0, 1.0, LIMIT)[1] == pytest.approx(0.2, abs=1e-9)
    assert relaxed([1.0, 1.0], 5.0, 1.0, LIMIT)[1] == pytest.approx(1.0, abs=1e-9)  # top bounds it
    # M is top: with top 2 a unit of budget buys twice the excess on each sample
    assert relaxed([0.2, 0.2, 0.6], 0.1, 2.0, LIMIT)[1] == pytest.approx(0.3, abs=1e-9)


# --- 3. timeouts ---------------------------------------------------------------------------------


def _timeout(time_limit):
    return LpResult("timeout", None, None, 0.0, time_limit, 1, "Time limit reached")


def test_every_solve_timing_out_gives_no_scale_and_records_every_outcome(monkeypatch):
    def maximise(self, column, value, time_limit):
        return _timeout(time_limit), None

    monkeypatch.setattr(SparseProgramme, "maximise", maximise)
    found = also_x(HAND, 0.1, 1.0, LIMIT)
    assert found.value is None
    assert found.outcomes == ("timeout",) * (steps(10, 0.1) + 1)
    assert cvar(HAND, 0.1, 1.0, LIMIT) == chance_programmes.ChanceScale(None, ("timeout",))


def test_midpoints_timing_out_bring_the_upper_end_down(monkeypatch):
    real = SparseProgramme.maximise
    calls = []

    def maximise(self, column, value, time_limit):
        calls.append(1)
        if len(calls) <= steps(10, 0.1):
            return _timeout(time_limit), None
        return real(self, column, value, time_limit)

    monkeypatch.setattr(SparseProgramme, "maximise", maximise)
    found = also_x(HAND, 0.1, 1.0, LIMIT)
    assert found.value == pytest.approx(0.2, abs=1e-12)
    assert found.outcomes == ("timeout",) * steps(10, 0.1) + ("optimal",)


# --- 4. refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("thresholds", "epsilon", "message"),
    [
        ([], 0.1, "at least one sample"),
        ([0.2, 1.5], 0.1, r"in \[0, 1.0\]"),
        ([-0.1], 0.1, r"in \[0, 1.0\]"),
        ([math.nan], 0.1, "finite"),
        ([0.2], 0.0, r"epsilon must be in \(0, 1\)"),
        ([0.2], 1.0, r"epsilon must be in \(0, 1\)"),
    ],
)
def test_refusals(thresholds, epsilon, message):
    with pytest.raises(ValueError, match=message):
        also_x(thresholds, epsilon, 1.0, LIMIT)
    with pytest.raises(ValueError, match=message):
        cvar(thresholds, epsilon, 1.0, LIMIT)


def test_a_negative_budget_is_refused():
    with pytest.raises(ValueError, match="budget"):
        relaxed([0.2], -0.1, 1.0, LIMIT)


# --- 5. agreement with independent forms ---------------------------------------------------------

LEVELS = (0.0, *GRID)


def _sets(count: int, seed: int):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(1, 60)
        weights = [rng.random() ** 3 for _ in LEVELS]
        theta = rng.choices(LEVELS, weights=weights, k=n)
        epsilon = rng.choice((0.1, 0.1, 0.05, 0.2, 0.3))
        yield np.array(theta), epsilon, rng.uniform(0, epsilon * n)


def _dense_relaxed(theta, budget, top):
    n = theta.size
    cost = np.zeros(n + 1)
    cost[0] = -1.0
    rows = np.zeros((n + 1, n + 1))
    rows[:n, 0] = 1.0
    rows[np.arange(n), 1 + np.arange(n)] = -top
    rows[n, 1:] = 1.0
    rhs = np.concatenate([theta, [budget]])
    bounds = [(0.0, top)] + [(0.0, 1.0)] * n
    found = linprog(cost, A_ub=rows, b_ub=rhs, bounds=bounds, method="highs")
    assert found.status == 0
    return found.x[0]


def _dense_cvar(theta, epsilon, top):
    """Programme (5) as written: beta, b, zeta."""
    n = theta.size
    cost = np.zeros(n + 2)
    cost[0] = -1.0
    rows = np.zeros((2 * n + 1, n + 2))
    rows[:n, 0] = 1.0
    rows[np.arange(n), 2 + np.arange(n)] = -1.0
    rows[n : 2 * n, 1] = 1.0
    rows[n + np.arange(n), 2 + np.arange(n)] = -1.0
    rows[2 * n, 1] = -(1.0 - epsilon)
    rows[2 * n, 2:] = 1.0 / n
    rhs = np.concatenate([theta, np.zeros(n), [0.0]])
    bounds = [(0.0, top), (None, 0.0)] + [(None, None)] * n
    found = linprog(cost, A_ub=rows, b_ub=rhs, bounds=bounds, method="highs")
    assert found.status == 0
    return found.x[0]


def _rockafellar_uryasev(theta, epsilon, top):
    """beta, b free, s >= 0: s_j >= beta - theta_j - b, b + sum(s) / (epsilon n) <= 0."""
    n = theta.size
    cost = np.zeros(n + 2)
    cost[0] = -1.0
    rows = np.zeros((n + 1, n + 2))
    rows[:n, 0] = 1.0
    rows[:n, 1] = -1.0
    rows[np.arange(n), 2 + np.arange(n)] = -1.0
    rows[n, 1] = 1.0
    rows[n, 2:] = 1.0 / (epsilon * n)
    rhs = np.concatenate([theta, [0.0]])
    bounds = [(0.0, top), (None, None)] + [(0.0, None)] * n
    found = linprog(cost, A_ub=rows, b_ub=rhs, bounds=bounds, method="highs")
    assert found.status == 0
    return found.x[0]


def _cvar_closed(theta, epsilon):
    ordered = np.sort(theta)
    share = epsilon * ordered.size
    whole = math.floor(share)
    total = ordered[:whole].sum() + (share - whole) * (
        ordered[whole] if whole < ordered.size else 0
    )
    return total / share


def test_the_programmes_agree_with_independent_builds_on_300_made_up_sets():
    for theta, epsilon, budget in _sets(300, 20261115):
        result, beta = relaxed(theta, budget, 1.0, LIMIT)
        assert result.outcome == "optimal"
        assert beta == pytest.approx(_dense_relaxed(theta, budget, 1.0), abs=1e-7)
        value = cvar(theta, epsilon, 1.0, LIMIT).value
        assert value == pytest.approx(_dense_cvar(theta, epsilon, 1.0), abs=1e-7)
        assert value == pytest.approx(_rockafellar_uryasev(theta, epsilon, 1.0), abs=1e-7)
        assert value == pytest.approx(_cvar_closed(theta, epsilon), abs=1e-7)


def test_also_x_is_the_sample_quantile_and_cvar_is_never_above_it_on_150_made_up_sets():
    for theta, epsilon, _ in _sets(150, 20261116):
        closed = float(np.sort(theta)[allowed(theta.size, epsilon)])
        found = also_x(theta, epsilon, 1.0, LIMIT)
        assert set(found.outcomes) == {"optimal"}
        assert closed - DELTA - 1e-9 <= found.value <= closed + 1e-9
        assert snap(found.value, GRID, 1e-4) == snap(closed, GRID, 0.0)
        assert snap(cvar(theta, epsilon, 1.0, LIMIT).value, GRID, 1e-6) <= snap(
            found.value, GRID, 1e-4
        )
