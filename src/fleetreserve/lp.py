"""The one door to the linear-programme solver.

Every later piece of code that solves a linear programme calls `solve_lp`. It minimises
`c @ x` with HiGHS (through scipy) and always returns an `LpResult`; a timeout is an outcome,
and a partial answer is never handed back as if it were a result.

Outcomes, and how they are read from the solver's own status (scipy 1.18):

- `"optimal"`: status 0. `x` and `objective` are set.
- `"timeout"`: status 1, the time or iteration limit was reached. A timeout is a missing
  result: it says nothing about whether the programme is feasible.
- `"infeasible"`: status 2 with the solver's message "The problem is infeasible", which HiGHS
  gives only for a proven infeasible programme. Status 2 also carries HiGHS model errors; those
  have no such message and are read as `"error"`.
- `"unbounded"`: status 3.
- `"error"`: everything else, including status 4 (HiGHS "unbounded or infeasible", numerical
  difficulties, unrecognised status), an exception raised by the solver, and an "optimal"
  answer that is not finite. Doubt is never read as infeasible. An exception has no solver
  status; `status` is then `NO_STATUS`.

On HiGHS' ambiguous "unbounded or infeasible" status: scipy maps it to status 4 with the
message "The problem is unbounded or infeasible". Programmes built to provoke it (infeasible
rows together with a free direction of descent, with and without presolve, both methods) were
all resolved by HiGHS as "infeasible", so the case is covered by tests on made-up solver answers
only.

Method `"ipm"` is the HiGHS interior-point solver. HiGHS runs crossover after the interior
point by default and scipy does not switch it off (its result has `crossover_nit`, the number
of crossover steps), so the answer is a vertex solution. Method `"simplex"` is the HiGHS dual
simplex.

The time limit is a request to the solver, not a guarantee. The solver checks its clock only at
certain points. For the interior-point method, both the set-up of the model and the step that
builds the starting basis for crossover can run past the limit by a wide margin, and the solve
then finishes late, with its true outcome or with a late "timeout". The dual simplex method was
observed to stop close to its limit. A hard wall-clock stop is not provided by this function.

A late answer stays what the solver said. It is marked: `LpResult.overran` is true when the
outcome is not `"timeout"` and `seconds` is more than `time_limit`, and the log line for that
solve is at WARNING level and says "overran". `LpResult.late_seconds` is the time beyond the
limit for every outcome, a late "timeout" included, and is in every log line.

Malformed input (shapes that do not fit, NaN or infinite data, a bad time limit or method) is a
caller bug and raises `ValueError`. Bounds and the time limit have no defaults: variables are
not silently non-negative, and no solve runs without a limit.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

# Time limits in seconds, set by docs/plan.md section 13 (Rules paragraph).
BOX_SECONDS = 30.0  # plan §13: 30 s for a box check
LP24_SECONDS = 600.0  # plan §13: 600 s for the 24-hour programme, LP (24)
OTHER_SECONDS = 300.0  # plan §13: 300 s for every other programme

NO_STATUS = -1  # `status` when the solver raised an exception and gave none

Outcome = Literal["optimal", "infeasible", "unbounded", "timeout", "error"]

_METHODS = {"ipm": "highs-ipm", "simplex": "highs-ds"}
_INFEASIBLE_TEXT = "The problem is infeasible."

_log = logging.getLogger("fleetreserve.lp")


class LpNotOptimal(RuntimeError):
    """Raised by `LpResult.require_optimal` when the solve did not end optimal."""


@dataclass(frozen=True)
class LpResult:
    """What one solve ended with.

    `x` and `objective` are set only when `outcome` is `"optimal"`. `status` and `message` are
    the solver's own (scipy's status code and text). `time_limit` is the limit given, in seconds.
    """

    outcome: Outcome
    x: np.ndarray | None
    objective: float | None
    seconds: float
    time_limit: float
    status: int
    message: str

    @property
    def overran(self) -> bool:
        """True when the solve ended in any outcome but `"timeout"` after more than `time_limit`.

        The answer is what the solver said and stays true; it came late.
        """
        return self.outcome != "timeout" and self.seconds > self.time_limit

    @property
    def late_seconds(self) -> float:
        """Seconds taken beyond `time_limit`, for every outcome (0.0 when on time)."""
        return max(0.0, self.seconds - self.time_limit)

    def require_optimal(self) -> "LpResult":
        """Return this result if it is optimal; otherwise raise `LpNotOptimal`."""
        if self.outcome != "optimal":
            raise LpNotOptimal(
                f"linear programme ended with outcome '{self.outcome}' "
                f"(solver status {self.status}: {self.message})"
            )
        return self


def solve_lp(
    c,
    *,
    A_ub=None,
    b_ub=None,
    A_eq=None,
    b_eq=None,
    bounds,
    time_limit,
    method="ipm",
) -> LpResult:
    """Minimise `c @ x` subject to `A_ub @ x <= b_ub`, `A_eq @ x == b_eq` and `bounds`.

    `bounds` is stated by the caller in the forms scipy's `linprog` accepts for a given
    programme: a sequence with one `(low, high)` pair per variable, or one pair for all; `None`
    or an infinity means no limit on that side. `bounds=None` is refused, because it would
    make the variables non-negative without saying so.

    `time_limit` is in seconds, a positive finite number. `method` is `"ipm"` or `"simplex"`.
    `A_ub` and `A_eq` may be dense or `scipy.sparse`; sparse matrices stay sparse.
    """
    limit = _check_time_limit(time_limit)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}, got {method!r}")
    cost = _check_vector(c, "c", None)
    n = cost.size
    if n == 0:
        raise ValueError("c must have at least one variable")
    ub, rhs_ub = _check_rows(A_ub, b_ub, n, "A_ub", "b_ub")
    eq, rhs_eq = _check_rows(A_eq, b_eq, n, "A_eq", "b_eq")
    box = _check_bounds(bounds, n)
    rows = (0 if ub is None else ub.shape[0]) + (0 if eq is None else eq.shape[0])

    start = time.perf_counter()
    try:
        solution = linprog(
            cost,
            A_ub=ub,
            b_ub=rhs_ub,
            A_eq=eq,
            b_eq=rhs_eq,
            bounds=box,
            method=_METHODS[method],
            options={"time_limit": limit},
        )
    except Exception as error:  # a solver failure is an outcome, not a crash
        seconds = time.perf_counter() - start
        result = LpResult(
            "error", None, None, seconds, limit, NO_STATUS, f"{type(error).__name__}: {error}"
        )
    else:
        seconds = time.perf_counter() - start
        result = _read_solution(solution, seconds, limit)

    fields = (
        result.outcome,
        result.status,
        result.seconds,
        limit,
        result.late_seconds,
        method,
        n,
        rows,
    )
    line = (
        "outcome=%s status=%s seconds=%.3f time_limit=%s late_seconds=%.3f method=%s "
        "variables=%d rows=%d"
    )
    if result.overran:
        _log.warning(
            line + "; overran the time limit: %.3f s taken against %s s allowed",
            *fields,
            result.seconds,
            limit,
        )
    else:
        _log.info(line, *fields)
    return result


def _read_solution(solution: Any, seconds: float, limit: float) -> LpResult:
    status = int(solution.status)
    message = str(solution.message)
    if status == 0:
        x = None if solution.x is None else np.array(solution.x, dtype=float)
        objective = None if solution.fun is None else float(solution.fun)
        if (
            x is None
            or objective is None
            or not (np.all(np.isfinite(x)) and math.isfinite(objective))
        ):
            return LpResult("error", None, None, seconds, limit, status, message)
        x.flags.writeable = False
        return LpResult("optimal", x, objective, seconds, limit, status, message)
    if status == 1:
        outcome: Outcome = "timeout"
    elif status == 2 and message.startswith(_INFEASIBLE_TEXT):
        outcome = "infeasible"
    elif status == 3:
        outcome = "unbounded"
    else:
        outcome = "error"
    return LpResult(outcome, None, None, seconds, limit, status, message)


def _check_time_limit(time_limit) -> float:
    try:
        limit = float(time_limit)
    except (TypeError, ValueError):
        raise ValueError(
            f"time_limit must be a positive finite number, got {time_limit!r}"
        ) from None
    if isinstance(time_limit, str) or not (math.isfinite(limit) and limit > 0):
        raise ValueError(f"time_limit must be a positive finite number, got {time_limit!r}")
    return limit


def _check_vector(values, name: str, length: int | None) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric vector: {error}") from error
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}")
    if length is not None and vector.size != length:
        raise ValueError(f"{name} has {vector.size} entries, expected {length}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must not contain NaN or infinity")
    return vector


def _check_rows(matrix, rhs, n: int, matrix_name: str, rhs_name: str):
    if matrix is None and rhs is None:
        return None, None
    if matrix is None or rhs is None:
        raise ValueError(f"{matrix_name} and {rhs_name} must be given together")
    if sp.issparse(matrix):
        # One conversion serves the check and the solve; every sparse format has `tocsr`, and
        # the result stays sparse.
        rows = matrix.tocsr().astype(np.float64, copy=False)
        finite = bool(np.all(np.isfinite(rows.data)))
    else:
        try:
            rows = np.asarray(matrix, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{matrix_name} must be numeric: {error}") from error
        finite = bool(np.all(np.isfinite(rows)))
    if rows.ndim != 2:
        raise ValueError(f"{matrix_name} must be two-dimensional, got shape {rows.shape}")
    if rows.shape[1] != n:
        raise ValueError(f"{matrix_name} has {rows.shape[1]} columns, c has {n} variables")
    if not finite:
        raise ValueError(f"{matrix_name} must not contain NaN or infinity")
    return rows, _check_vector(rhs, rhs_name, rows.shape[0])


def _check_bounds(bounds, n: int) -> np.ndarray:
    if bounds is None:
        raise ValueError(
            "bounds must be stated; None would make every variable non-negative without saying "
            "so (write (0, None) for each variable if that is meant)"
        )
    try:
        pairs = np.array(bounds, dtype=object)
        if pairs.shape == (2,):
            pairs = np.tile(pairs, (n, 1))
        if pairs.shape != (n, 2):
            raise ValueError(
                f"bounds must be one (low, high) pair or {n} pairs, got shape {pairs.shape}"
            )
        low = np.array([-math.inf if v is None else float(v) for v in pairs[:, 0]])
        high = np.array([math.inf if v is None else float(v) for v in pairs[:, 1]])
    except (TypeError, ValueError) as error:
        raise ValueError(f"bounds are malformed: {error}") from error
    if np.any(np.isnan(low)) or np.any(np.isnan(high)):
        raise ValueError("bounds must not contain NaN")
    if np.any(low == math.inf) or np.any(high == -math.inf):
        raise ValueError("a lower bound of +infinity or an upper bound of -infinity is not a bound")
    return np.column_stack([low, high])
