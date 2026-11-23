"""Tests for the linear-programme solve door, on made-up numbers."""

import logging
import math
import time

import numpy as np
import pytest
import scipy.sparse as sp

import fleetreserve.lp as lp
from fleetreserve.lp import (
    BOX_SECONDS,
    LP24_SECONDS,
    OTHER_SECONDS,
    LpNotOptimal,
    LpResult,
    solve_lp,
)

METHODS = ["ipm", "simplex"]
LIMIT = 30.0


def _hard_lp(n: int = 1500, m: int = 800, seed: int = 0):
    """A dense random programme: minimise c @ x subject to A x <= b, 0 <= x <= 1.

    It is feasible (x = 0 satisfies the rows) and bounded (every variable is in a box). At the
    default size it is large enough to outlast a very small limit and small enough to solve
    quickly under a generous one.
    """
    rng = np.random.default_rng(seed)
    A = rng.random((m, n))
    b = A.sum(axis=1) / 4.0
    c = -rng.random(n)
    return c, A, b


# ---- answers ----------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_small_lp_matches_hand_computation(method):
    # minimise -x - 2y subject to x + y <= 4, 0 <= x <= 3, 0 <= y <= 3.
    # Hand computation: y pays twice as much as x per unit and fits up to its bound 3, so
    # y = 3. The row then leaves x <= 4 - 3 = 1, so x = 1. The value is -1 - 2 * 3 = -7.
    res = solve_lp(
        [-1.0, -2.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[4.0],
        bounds=[(0, 3), (0, 3)],
        time_limit=LIMIT,
        method=method,
    )
    assert res.outcome == "optimal"
    assert res.objective == pytest.approx(-7.0, abs=1e-7)
    assert res.x == pytest.approx([1.0, 3.0], abs=1e-7)
    assert res.status == 0


@pytest.mark.parametrize("method", METHODS)
def test_equality_constraints(method):
    # minimise 2x + 3y subject to x + y = 5, 0 <= x <= 4, 0 <= y <= 10.
    # Hand computation: x is the cheaper way to reach 5, so x = 4 (its bound) and y = 1.
    # The value is 2 * 4 + 3 * 1 = 11.
    res = solve_lp(
        [2.0, 3.0],
        A_eq=[[1.0, 1.0]],
        b_eq=[5.0],
        bounds=[(0, 4), (0, 10)],
        time_limit=LIMIT,
        method=method,
    )
    assert res.outcome == "optimal"
    assert res.objective == pytest.approx(11.0, abs=1e-7)
    assert res.x == pytest.approx([4.0, 1.0], abs=1e-7)


@pytest.mark.parametrize("method", METHODS)
def test_negative_lower_bounds_are_honoured(method):
    # minimise x + y subject to -x - y <= 10, -3 <= x <= 5, -2 <= y <= 5.
    # Hand computation: the row only requires x + y >= -10, which holds at the lower bounds,
    # so the minimum sits at x = -3, y = -2 with value -5. Variables that were silently
    # non-negative would give 0 instead.
    res = solve_lp(
        [1.0, 1.0],
        A_ub=[[-1.0, -1.0]],
        b_ub=[10.0],
        bounds=[(-3, 5), (-2, 5)],
        time_limit=LIMIT,
        method=method,
    )
    assert res.outcome == "optimal"
    assert res.objective == pytest.approx(-5.0, abs=1e-7)
    assert res.x == pytest.approx([-3.0, -2.0], abs=1e-7)


@pytest.mark.parametrize("method", METHODS)
def test_open_bounds_with_none_and_infinity(method):
    # minimise x + y subject to x >= 1 (as -x <= -1) and y >= 2 (as -y <= -2); both free above
    # and below by the bounds. Hand computation: x = 1, y = 2, value 3.
    res = solve_lp(
        [1.0, 1.0],
        A_ub=[[-1.0, 0.0], [0.0, -1.0]],
        b_ub=[-1.0, -2.0],
        bounds=[(None, None), (-math.inf, math.inf)],
        time_limit=LIMIT,
        method=method,
    )
    assert res.outcome == "optimal"
    assert res.x == pytest.approx([1.0, 2.0], abs=1e-7)


def test_single_bounds_pair_applies_to_every_variable():
    # minimise x + y + z subject to x + y + z >= -9 with each variable in [-2, 4].
    # Hand computation: all three sit at -2, value -6 (the row needs only >= -9).
    res = solve_lp(
        [1.0, 1.0, 1.0],
        A_ub=[[-1.0, -1.0, -1.0]],
        b_ub=[9.0],
        bounds=(-2, 4),
        time_limit=LIMIT,
    )
    assert res.outcome == "optimal"
    assert res.x == pytest.approx([-2.0, -2.0, -2.0], abs=1e-7)


def test_interior_point_returns_a_vertex():
    # Zero objective over a box with 5 equality rows: every feasible point is optimal, and an
    # interior-point answer without crossover would sit inside the box. With crossover the
    # answer is a vertex, so at most 5 variables lie strictly between their bounds.
    rng = np.random.default_rng(3)
    n, m = 120, 5
    A = rng.random((m, n))
    b = A @ rng.random(n)
    res = solve_lp(np.zeros(n), A_eq=A, b_eq=b, bounds=(0, 1), time_limit=LIMIT, method="ipm")
    assert res.outcome == "optimal"
    strictly_inside = np.sum((res.x > 1e-9) & (res.x < 1 - 1e-9))
    assert strictly_inside <= m
    assert np.allclose(A @ res.x, b, atol=1e-6)


# ---- other outcomes ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_infeasible(method):
    # x <= 1 and x >= 2 cannot both hold.
    res = solve_lp(
        [1.0],
        A_ub=[[1.0], [-1.0]],
        b_ub=[1.0, -2.0],
        bounds=[(0, None)],
        time_limit=LIMIT,
        method=method,
    )
    assert res.outcome == "infeasible"
    assert res.x is None
    assert res.objective is None


@pytest.mark.parametrize("method", METHODS)
def test_unbounded(method):
    # minimise -x with x >= 0 and only y constrained: x can grow without limit.
    res = solve_lp(
        [-1.0, 0.0],
        A_ub=[[0.0, 1.0]],
        b_ub=[1.0],
        bounds=[(0, None), (0, None)],
        time_limit=LIMIT,
        method=method,
    )
    assert res.outcome == "unbounded"
    assert res.x is None
    assert res.objective is None


# With the interior point, HiGHS (scipy 1.18) stops on the clock when the limit is already over at
# its first check (the tiny limit below). A limit that runs out later can be missed during set-up
# or crossover, and the solve then finishes late; no test relies on a limit inside that
# machine-dependent window. How quickly the solver returns is not controlled by `solve_lp`, so no
# test here asserts a return time from a real solve.


def _assert_timeout(res):
    assert res.outcome == "timeout"
    assert res.x is None
    assert res.objective is None


@pytest.mark.parametrize("method", METHODS)
def test_timeout_when_the_limit_is_already_over(method):
    c, A, b = _hard_lp()
    res = solve_lp(c, A_ub=A, b_ub=b, bounds=(0, 1), time_limit=1e-6, method=method)
    _assert_timeout(res)


def test_simplex_with_a_short_limit_times_out_or_overruns():
    # The dual simplex method was observed to stop close to its limit, so a short limit gives a
    # timeout. On a machine fast enough to finish first, the answer is optimal and late, and
    # the overrun is marked. Any other result is a fault.
    c, A, b = _hard_lp()
    res = solve_lp(c, A_ub=A, b_ub=b, bounds=(0, 1), time_limit=0.05, method="simplex")
    assert (res.outcome == "timeout" and not res.overran) or (
        res.outcome == "optimal" and res.overran
    )
    if res.outcome == "timeout":
        _assert_timeout(res)


@pytest.mark.parametrize("method", METHODS)
def test_same_programme_with_a_generous_limit_is_optimal(method):
    c, A, b = _hard_lp()
    res = solve_lp(c, A_ub=A, b_ub=b, bounds=(0, 1), time_limit=LIMIT, method=method)
    assert res.outcome == "optimal"
    assert res.x is not None
    assert np.all(A @ res.x <= b + 1e-6)


def test_solver_exception_is_an_error_not_infeasible(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("solver blew up")

    monkeypatch.setattr(lp, "linprog", broken)
    res = solve_lp([1.0], bounds=[(0, 1)], time_limit=LIMIT)
    assert res.outcome == "error"
    assert res.x is None
    assert "solver blew up" in res.message


class _FakeSolution:
    """A solver answer made up here. Unless given otherwise it carries a partial point."""

    def __init__(self, status, message, x=(0.5,), fun=1.0):
        self.status = status
        self.message = message
        self.x = None if x is None else np.array(x)
        self.fun = fun


@pytest.mark.parametrize(
    ("status", "message", "outcome"),
    [
        (4, "The problem is unbounded or infeasible. (HiGHS Status 9: x)", "error"),
        (4, "Numerical difficulties encountered.", "error"),
        (2, "(HiGHS Status 4: model_status is Model_error)", "error"),
        (5, "something new", "error"),
        (1, "Time limit reached. (HiGHS Status 13: Time limit reached)", "timeout"),
        (1, "Iteration limit reached. (HiGHS Status 14: x)", "timeout"),
        (2, "The problem is infeasible. (HiGHS Status 8: x)", "infeasible"),
        (3, "The problem is unbounded. (HiGHS Status 10: x)", "unbounded"),
    ],
)
def test_status_mapping_never_reads_doubt_as_infeasible(monkeypatch, status, message, outcome):
    # HiGHS' own "unbounded or infeasible" status could not be produced on real programmes
    # (see the module text), so the mapping is checked with solver answers made up here.
    monkeypatch.setattr(lp, "linprog", lambda *a, **k: _FakeSolution(status, message))
    res = solve_lp([1.0], bounds=[(0, 1)], time_limit=LIMIT)
    assert res.outcome == outcome
    assert res.x is None
    assert res.objective is None
    assert res.status == status
    assert res.message == message


def test_optimal_with_a_non_finite_answer_is_an_error(monkeypatch):
    fake = _FakeSolution(0, "ok", x=[np.nan], fun=0.0)
    monkeypatch.setattr(lp, "linprog", lambda *a, **k: fake)
    res = solve_lp([1.0], bounds=[(0, 1)], time_limit=LIMIT)
    assert res.outcome == "error"
    assert res.x is None


# ---- input checks -----------------------------------------------------------------------------


def test_time_limit_is_required():
    with pytest.raises(TypeError):
        solve_lp([1.0], bounds=[(0, 1)])  # type: ignore[call-arg]


def test_bounds_are_required():
    with pytest.raises(TypeError):
        solve_lp([1.0], time_limit=LIMIT)  # type: ignore[call-arg]


def test_bounds_none_is_refused():
    with pytest.raises(ValueError, match="bounds"):
        solve_lp([1.0], bounds=None, time_limit=LIMIT)


@pytest.mark.parametrize("bad", [0, -1.0, math.inf, -math.inf, math.nan, "5", None])
def test_bad_time_limit(bad):
    with pytest.raises(ValueError, match="time_limit"):
        solve_lp([1.0], bounds=[(0, 1)], time_limit=bad)


@pytest.mark.parametrize("bad", ["barrier", "", "IPM", None])
def test_unknown_method(bad):
    with pytest.raises(ValueError, match="method"):
        solve_lp([1.0], bounds=[(0, 1)], time_limit=LIMIT, method=bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"A_ub": [[1.0, 1.0]], "b_ub": [1.0, 2.0]},  # row count differs
        {"A_ub": [[1.0, 1.0, 1.0]], "b_ub": [1.0]},  # column count differs from c
        {"A_ub": [[1.0, 1.0]]},  # matrix without right-hand side
        {"b_ub": [1.0]},  # right-hand side without matrix
        {"A_eq": [[1.0, 1.0]], "b_eq": [1.0, 2.0]},
        {"A_eq": [[1.0, 1.0, 1.0]], "b_eq": [1.0]},
        {"A_eq": sp.csr_array([[1.0, 1.0, 1.0]]), "b_eq": [1.0]},
        {"A_ub": [[1.0, 1.0]], "b_ub": [math.nan]},
        {"A_ub": [[math.nan, 1.0]], "b_ub": [1.0]},
        {"A_eq": sp.csr_array([[math.nan, 1.0]]), "b_eq": [1.0]},
        {"A_ub": [[math.inf, 1.0]], "b_ub": [1.0]},
    ],
)
def test_malformed_matrices_raise(kwargs):
    with pytest.raises(ValueError):
        solve_lp([1.0, 1.0], bounds=[(0, 1)] * 2, time_limit=LIMIT, **kwargs)


@pytest.mark.parametrize(
    ("c", "bounds"),
    [
        ([math.nan, 1.0], [(0, 1)] * 2),
        ([math.inf, 1.0], [(0, 1)] * 2),
        ([[1.0, 1.0]], [(0, 1)] * 2),
        ([], []),
        ([1.0, 1.0], [(0, 1)] * 3),  # wrong number of bounds pairs
        ([1.0, 1.0], [(0, 1, 2)] * 2),  # not pairs
        ([1.0, 1.0], [(math.nan, 1), (0, 1)]),
        ([1.0, 1.0], [(0, math.nan), (0, 1)]),
        ([1.0, 1.0], [(math.inf, None), (0, 1)]),
        ([1.0, 1.0], [(None, -math.inf), (0, 1)]),
        ([1.0, 1.0], [("a", 1), (0, 1)]),
    ],
)
def test_malformed_objective_or_bounds_raise(c, bounds):
    with pytest.raises(ValueError):
        solve_lp(c, bounds=bounds, time_limit=LIMIT)


# ---- sparse input, logging, result type ------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("make", [sp.csr_array, sp.csc_matrix, sp.coo_array])
def test_sparse_and_dense_agree(method, make):
    c, A, b = _hard_lp(n=40, m=20, seed=1)
    A[A < 0.7] = 0.0
    Aeq = np.ones((1, 40))
    beq = [2.0]
    dense = solve_lp(
        c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=(0, 1), time_limit=LIMIT, method=method
    )
    sparse = solve_lp(
        c,
        A_ub=make(A),
        b_ub=b,
        A_eq=make(Aeq),
        b_eq=beq,
        bounds=(0, 1),
        time_limit=LIMIT,
        method=method,
    )
    assert dense.outcome == sparse.outcome == "optimal"
    assert sparse.objective == pytest.approx(dense.objective, abs=1e-7)
    assert sparse.x == pytest.approx(dense.x, abs=1e-6)


SPARSE_FORMATS = [
    getattr(sp, f"{fmt}_{kind}")
    for fmt in ("csr", "csc", "coo", "lil", "dok", "bsr", "dia")
    for kind in ("matrix", "array")
]


@pytest.mark.parametrize("make", SPARSE_FORMATS, ids=lambda make: make.__name__)
def test_every_sparse_format_gives_the_dense_answer(make):
    # minimise -x - 2y - z subject to x + y + z <= 4, x - y = 0, 0 <= each <= 3.
    # Hand computation: y costs most, but x = y, so (x, y) costs 3 per unit of x and uses 2 units
    # of the row; z costs 1 per unit and uses 1. Per unit of row, (x, y) pays 1.5 and z pays 1,
    # so x = y = 2 and z = 0 (row full), value -6.
    A_ub = np.array([[1.0, 1.0, 1.0]])
    A_eq = np.array([[1.0, -1.0, 0.0]])
    args = {"b_ub": [4.0], "b_eq": [0.0], "bounds": (0, 3), "time_limit": LIMIT}
    c = [-1.0, -2.0, -1.0]
    dense = solve_lp(c, A_ub=A_ub, A_eq=A_eq, **args)
    sparse = solve_lp(c, A_ub=make(A_ub), A_eq=make(A_eq), **args)
    assert dense.outcome == sparse.outcome == "optimal"
    assert sparse.objective == pytest.approx(-6.0, abs=1e-7)
    assert sparse.x == pytest.approx(dense.x, abs=1e-7)


@pytest.mark.parametrize("make", SPARSE_FORMATS, ids=lambda make: make.__name__)
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_values_in_every_sparse_format_raise(make, bad):
    A = np.array([[1.0, bad, 1.0]])
    with pytest.raises(ValueError, match="NaN or infinity"):
        solve_lp([1.0, 1.0, 1.0], A_ub=make(A), b_ub=[1.0], bounds=(0, 1), time_limit=LIMIT)


@pytest.mark.parametrize(("method", "scipy_name"), [("ipm", "highs-ipm"), ("simplex", "highs-ds")])
def test_each_method_calls_the_matching_highs_solver(monkeypatch, method, scipy_name):
    calls = []
    real = lp.linprog

    def spy(c, **kwargs):
        calls.append(kwargs["method"])
        return real(c, **kwargs)

    monkeypatch.setattr(lp, "linprog", spy)
    res = solve_lp([1.0], bounds=[(2, 3)], time_limit=LIMIT, method=method)
    assert res.outcome == "optimal"
    assert calls == [scipy_name]


def test_returned_x_is_read_only():
    res = solve_lp([1.0, 1.0], bounds=[(2, 3), (4, 5)], time_limit=LIMIT)
    assert res.x is not None
    assert not res.x.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        res.x[0] = 99.0


def test_late_seconds_on_a_late_timeout_and_an_on_time_solve(monkeypatch):
    message = "Time limit reached. (HiGHS Status 13: x)"
    monkeypatch.setattr(lp, "linprog", _slow_solver(0.05, status=1, message=message))
    late = solve_lp([1.0], bounds=[(0, 1)], time_limit=0.01)
    assert late.outcome == "timeout"
    assert late.overran is False
    assert late.late_seconds == pytest.approx(late.seconds - 0.01)
    assert late.late_seconds > 0.0
    monkeypatch.undo()
    on_time = solve_lp([1.0], bounds=[(2, 3)], time_limit=LIMIT)
    assert on_time.late_seconds == 0.0


def test_late_seconds_is_read_only_and_in_every_log_line(monkeypatch, caplog):
    assert isinstance(LpResult.late_seconds, property)
    assert LpResult.late_seconds.fset is None
    with caplog.at_level(logging.INFO, logger="fleetreserve.lp"):
        solve_lp([1.0], bounds=[(2, 3)], time_limit=LIMIT)
        monkeypatch.setattr(lp, "linprog", _slow_solver(0.05))
        late = solve_lp([1.0], bounds=[(0, 1)], time_limit=0.01)
    records = [r for r in caplog.records if r.name == "fleetreserve.lp"]
    assert len(records) == 2
    assert "late_seconds=0.000" in records[0].getMessage()
    assert f"late_seconds={late.late_seconds:.3f}" in records[1].getMessage()


def test_sparse_matrices_reach_the_solver_unchanged_in_kind(monkeypatch):
    seen = {}
    real = lp.linprog

    def spy(c, **kwargs):
        seen.update(kwargs)
        return real(c, **kwargs)

    monkeypatch.setattr(lp, "linprog", spy)
    res = solve_lp(
        [-1.0, -1.0],
        A_ub=sp.csr_array([[1.0, 1.0]]),
        b_ub=[1.0],
        A_eq=sp.csc_matrix([[1.0, -1.0]]),
        b_eq=[0.0],
        bounds=(0, 1),
        time_limit=LIMIT,
    )
    assert res.outcome == "optimal"
    assert sp.issparse(seen["A_ub"])
    assert sp.issparse(seen["A_eq"])


def test_one_info_line_per_solve(caplog):
    with caplog.at_level(logging.INFO, logger="fleetreserve.lp"):
        solve_lp(
            [1.0, 1.0],
            A_ub=[[-1.0, -1.0]],
            b_ub=[-1.0],
            A_eq=[[1.0, -1.0]],
            b_eq=[0.0],
            bounds=(0, 5),
            time_limit=12.5,
            method="simplex",
        )
    records = [r for r in caplog.records if r.name == "fleetreserve.lp"]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    text = records[0].getMessage()
    for part in [
        "outcome=optimal",
        "status=0",
        "seconds=",
        "time_limit=12.5",
        "method=simplex",
        "variables=2",
        "rows=2",
    ]:
        assert part in text


def test_log_line_records_a_timeout(caplog):
    c, A, b = _hard_lp()
    with caplog.at_level(logging.INFO, logger="fleetreserve.lp"):
        solve_lp(c, A_ub=A, b_ub=b, bounds=(0, 1), time_limit=1e-6)
    assert "outcome=timeout" in caplog.text


def test_result_is_frozen():
    res = solve_lp([1.0], bounds=[(0, 1)], time_limit=LIMIT)
    assert isinstance(res, LpResult)
    with pytest.raises(AttributeError):
        res.outcome = "infeasible"  # type: ignore[misc]


def test_result_fields_on_success():
    res = solve_lp([1.0], bounds=[(2, 3)], time_limit=LIMIT)
    assert res.outcome == "optimal"
    assert isinstance(res.x, np.ndarray)
    assert isinstance(res.objective, float)
    assert res.seconds >= 0.0
    assert isinstance(res.status, int)
    assert isinstance(res.message, str)
    assert res.objective == pytest.approx(2.0)


def test_require_optimal_returns_the_result_when_optimal():
    res = solve_lp([1.0], bounds=[(2, 3)], time_limit=LIMIT)
    assert res.require_optimal() is res


@pytest.mark.parametrize("outcome", ["infeasible", "unbounded", "timeout", "error"])
def test_require_optimal_raises_otherwise(outcome):
    res = LpResult(
        outcome=outcome,
        x=None,
        objective=None,
        seconds=0.1,
        time_limit=30.0,
        status=7,
        message="solver text",
    )
    with pytest.raises(LpNotOptimal, match=outcome):
        res.require_optimal()


def _slow_solver(seconds, status=0, message="slow", x=(1.0,), fun=1.0):
    """A stand-in for `linprog` that answers only after sleeping `seconds`."""

    def solver(*args, **kwargs):
        time.sleep(seconds)
        return _FakeSolution(status, message, x=x, fun=fun)

    return solver


def test_late_optimal_answer_is_kept_and_marked_as_overrun(monkeypatch, caplog):
    monkeypatch.setattr(lp, "linprog", _slow_solver(0.05))
    with caplog.at_level(logging.INFO, logger="fleetreserve.lp"):
        res = solve_lp([1.0], bounds=[(0, 1)], time_limit=0.01)
    assert res.outcome == "optimal"  # the answer is true, only late
    assert res.x is not None
    assert res.time_limit == 0.01
    assert res.seconds > res.time_limit
    assert res.overran is True
    records = [r for r in caplog.records if r.name == "fleetreserve.lp"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    text = records[0].getMessage()
    assert "overran" in text
    assert f"{res.seconds:.3f}" in text
    assert "0.01" in text


def test_late_infeasible_answer_is_marked_as_overrun(monkeypatch):
    message = "The problem is infeasible. (HiGHS Status 8: x)"
    monkeypatch.setattr(lp, "linprog", _slow_solver(0.05, status=2, message=message))
    res = solve_lp([1.0], bounds=[(0, 1)], time_limit=0.01)
    assert res.outcome == "infeasible"
    assert res.overran is True


def test_normal_solve_has_not_overrun_and_logs_at_info(caplog):
    with caplog.at_level(logging.INFO, logger="fleetreserve.lp"):
        res = solve_lp([1.0], bounds=[(2, 3)], time_limit=LIMIT)
    assert res.outcome == "optimal"
    assert res.time_limit == LIMIT
    assert res.overran is False
    records = [r for r in caplog.records if r.name == "fleetreserve.lp"]
    assert [r.levelno for r in records] == [logging.INFO]
    assert "overran" not in records[0].getMessage()


def test_timeout_has_not_overrun_even_when_late(monkeypatch, caplog):
    message = "Time limit reached. (HiGHS Status 13: x)"
    monkeypatch.setattr(lp, "linprog", _slow_solver(0.05, status=1, message=message))
    with caplog.at_level(logging.INFO, logger="fleetreserve.lp"):
        res = solve_lp([1.0], bounds=[(0, 1)], time_limit=0.01)
    assert res.outcome == "timeout"
    assert res.seconds > res.time_limit
    assert res.overran is False
    records = [r for r in caplog.records if r.name == "fleetreserve.lp"]
    assert [r.levelno for r in records] == [logging.INFO]


def test_overran_is_read_only():
    res = solve_lp([1.0], bounds=[(2, 3)], time_limit=LIMIT)
    assert isinstance(LpResult.overran, property)
    assert LpResult.overran.fset is None
    with pytest.raises(AttributeError):
        res.overran = True  # type: ignore[misc]


def test_interior_point_with_a_short_limit_times_out_or_overruns():
    # Known behaviour of the installed solver (see the module text): a limit that runs out while
    # the model is being set up is either seen (timeout) or missed (the solve finishes late).
    # The late answer must then be marked. Any other result is a fault.
    c, A, b = _hard_lp()
    res = solve_lp(c, A_ub=A, b_ub=b, bounds=(0, 1), time_limit=0.02, method="ipm")
    assert (res.outcome == "timeout" and not res.overran) or (
        res.outcome == "optimal" and res.overran
    )


def test_time_limit_constants_match_the_plan():
    assert BOX_SECONDS == 30.0
    assert LP24_SECONDS == 600.0
    assert OTHER_SECONDS == 300.0
