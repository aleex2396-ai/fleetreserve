"""A fifth goal after the four of `asa_programme`: the equal-sharing tie-break of Lee et al. (2021)
in a linear form, for the comparison method `asa-es` (plan section 10).

`asa_programme` holds this project's linear-programme form of ASA (method `asa`) and its readings
A1 to A5 and A7; reading A6 (one-way only) is in `asa_controller`. `asa_programme`'s text says
that the solution of its four goals may not be unique: goal 4 weighs hours, not cars, so a plan
that moves energy between cars within an hour without changing any goal ties with it, and the
solver picks one of the tied plans. The paper does not leave that tie open.
What was read of it (arXiv version 2012.02636v1, 4 December 2020; equation and section numbers of
that version): section V-B says that the utility functions it lists before this one are not
strictly concave in the rates, so the optimal solution is generally not unique, and adds the
regularizer `u^ES(r) = -sum over t in T and i in V of r_i(t)^2` to force a unique optimal
solution; it says the term also promotes equal sharing between the cars, being largest when all
rates are as low as possible. Section VI-C's case study uses ASA with the objective
`u^QC(r) + 1e-12 u^ES(r)` (named ASA-QC there), the regularizer included to promote equal sharing
between similar cars and to force a unique solution. `asa_programme` (A1) left the term out.

This module adds it as a fifth goal. The method `asa-es` of `compare_methods` (the controller
`asa_controller.AsaEsController`) plans each hour by `plan_hour` of this module; the method `asa`
is unchanged and both are reported. `asa-es` is this project's linear form of ASA with the
differences listed in `asa_programme` (A1) and below; it is not the ASA of the paper.

Readings (each the closest reading found; the alternative is in brackets):

  E1. The fifth goal, "share": the smallest sum, over the cars and the hours `t..n` of the
      programme's horizon, of a convex piecewise-linear approximation `f` of the square of each
      planned power. For car `i` in hour `s` with upper limit `U` (A3 of `asa_programme`;
      `upper[i, s - t]` of `asa_programme.upper_bounds`, whose columns are the hours `t..n`)
      above 0, `f` is the chord interpolant of `r^2` on `[0, U]` with
      `K` pieces (`pieces` of `plan_hour`, default `PIECES`, 8): breakpoints `b_k = k U / K`,
      `k = 0 .. K`, and on `[b_k, b_(k+1)]` the chord `(b_k + b_(k+1)) r - b_k b_(k+1)` between
      the square's values at the two ends (`approximate_square`; as `f` is convex, it is also the
      largest of the `K` chord lines on `[0, U]`). It is written in segment form: `K` columns
      `s_0 .. s_(K-1)` per such car-hour, `s_k` in `[0, b_(k+1) - b_k]`, one row
      `p = s_0 + .. + s_(K-1)`, and the cost `(b_k + b_(k+1)) s_k` summed. The slopes
      `b_k + b_(k+1)` rise strictly with `k`, so at an optimum no segment holds energy while an
      earlier one of the same car-hour is below its limit (moving it to the earlier one would
      lower the cost and keep `p`), and the cost of a car-hour is `f` of its power, to the
      solver's tolerances. A car-hour whose upper limit is 0 has power 0, no segment and no row
      (its square, 0, is exact). [Alternatives: the epigraph form, one variable `z` per car-hour
      and `K` rows `z >= (b_k + b_(k+1)) p - b_k b_(k+1)`, the same `f` and the same optimum,
      with `K` rows per car-hour in place of one; the true square by a quadratic-programme
      solve, which `lp.solve_lp` does not offer; the smallest largest power in each hour.]
        - Gap: `0 <= f(r) - r^2 <= (U / K)^2 / 4 = U^2 / (4 K^2)` on `[0, U]` (`U^2 / 256` at
          `K = 8`), the largest at the middle of each piece and 0 at each breakpoint, 0 and `U`
          included (`largest_gap`).
        - The approximation's minimiser is not always the square's. In
          `tests/wp10/test_asa_es_hand.py`, case 4 (upper limits 2 and 1, an hour's total of 1.2)
          it is (0.575, 0.625), the square's (0.6, 0.6). Where cars' powers lie inside pieces of
          equal slope the approximation is flat along moves between them, so tied plans remain
          (case 2: two identical cars sharing 0.8 tie anywhere from (0.3, 0.5) to (0.5, 0.3)); a
          unique solution, which the paper's square gives, is not guaranteed.
        - What it cannot do: it breaks no tie inside one piece. For `m` cars with the same upper
          limit `U` above 0 in one hour, `f(r) = (U / K) r` on the first piece `[0, U / K]` and
          `f(r) > (U / K) r` on `(U / K, U]`. So among the splits of one total `P`, at most
          `m U / K`, between these cars in that hour, the rest of the plan unchanged, every split
          that keeps each of their powers at most `U / K` gives their sum of `f` its least value,
          `(U / K) P`, and every split with a power above `U / K` a larger one. Hence a plan with
          one of these powers above `U / K` is not an optimum of the fifth goal (to the solver's
          tolerances) when a split of the same total with each power at most `U / K`, the rest
          unchanged, keeps the rows of goals 1 to 4 and the bounds of A3. But the fifth goal does
          not tell apart the splits that keep each power at most `U / K`, the equal split among
          them: which of those tied plans comes back is the solver's choice. Such a move between
          cars within an hour keeps goals 1, 3 and 4, which read only each hour's fleet total, but
          changes the planned energy over the horizon of the cars it moves energy between, which
          goal 2 and the energy bound of A3 read per car, so those rows can rule out every such
          split. Observed with the HiGHS of the lockfile (no test pins these): one service hour,
          nothing drawn, five cars each with upper limit 7.2 (`U / K = 0.9`): an hour's total of
          3.6 gave (0, 0.9, 0.9, 0.9, 0.9) under `asa-es` and (3.6, 0, 0, 0, 0) under `asa`; a
          total of 2.0 gave (0, 0, 0.2, 0.9, 0.9) and (2, 0, 0, 0, 0); a total of 2.0 with a
          believed need of 2.0 for the first car and no recovery hours gave (2, 0, 0, 0, 0) under
          both, goal 2 holding that car at 2.0.
        - Bound. With `x` the plan returned and `y` any plan that keeps the rows of goals 1 to 4,
          `sum x^2 <= sum f(x) <= sum f(y) <= sum y^2 + G`, with `G` the sum over the
          horizon's car-hours of `U^2 / (4 K^2)`: the plan's sum of squares over the horizon is
          at most `G` above the least sum of squares under goals 1 to 4 (to the solver's
          tolerances). It bounds the horizon's sum, not the first hour's: the plan applied, the
          first hour's, can be less equal than that of a plan tied under goals 1 to 4 (case 5).
        - `K = 8`. The largest gap of a car-hour, `U^2 / (4 K^2)`, falls as `1 / K^2`, and the
          programme has `K` segment columns and one row per car-hour whose upper limit is above
          0: a larger `K` multiplies each largest gap by `(8 / K)^2` and adds `K - 8` columns
          per such car-hour.
  E2. Strict order, as A4. Goals 1 to 4 are `asa_programme`'s, built by its `build_programme` and
      solved by its `solve_stages`: the same programmes as `asa`'s, so their outcomes and optima
      are those of `asa` on the same view. Goal 4's optimum is then kept as a row, as A4 keeps
      those of goals 1 to 3 (`keep_optimum`, `STAGE_TOLERANCE`), and goal 5 is solved with all
      four rows. The paper weighs the term by 1e-12 in one objective; a strict order is this
      module's reading of a tie-break. [Alternative: one programme with the approximation
      weighted by 1e-12 beside goal 4; a finite weight can trade a small loss in goal 4 for a
      gain in goal 5.] Everything else as `asa`: the view, belief, horizon and bounds of A3, a
      target or centre below 0 read as 0 (A7), the method of A5 with the time limit the caller
      gives on every one of the five stages, and a stage that ends in any outcome but "optimal"
      ends the hour's programme (no plan, later stages not solved; the controller then plans the
      hour by its fall-back, `asa_controller`).
  E3. In the registry (`compare_methods.METHODS`) `asa-es` comes after `asa`, so a comparison run
      without `--methods` runs both.

Differences from the paper beyond those of `asa_programme` (A1): the square is approximated by
chords; the term is a fifth goal in strict order, not a weighted part of one objective; it sums
over the hours left in the service window (A3), the paper's over its optimisation horizon.

`plan_hour(view, centre, n_recovery, time_limit=..., solve=..., pieces=PIECES)` returns an
`asa_programme.HourPlan`: the outcome of every stage solved, in the order of `STAGES`, and, when
all five are optimal, the planned powers of hours `t..n` (the solver's values clipped to each
power's bounds) and the five optima (the fifth: the least sum of `f`). Refused with
`ValueError`: what `asa_programme.plan_hour` refuses, and a number of pieces that is not a whole
number at least 1. The segment columns and rows of E1 are added by `share_goal`, which
`plan_hour` calls after goal 4's optimum is kept and which `asa_diagnosis_lp` calls to rebuild
the fifth goal of an hour.
"""

from numbers import Integral

import numpy as np

from fleetreserve import lp
from fleetreserve.asa_programme import STAGES as FOUR_STAGES
from fleetreserve.asa_programme import (
    HourPlan,
    Programme,
    build_programme,
    checked_time_limit,
    solve_stage,
    solve_stages,
)
from fleetreserve.controller import HourView

STAGES = (*FOUR_STAGES, "share")  # E2, in order
PIECES = 8  # E1: K


def checked_pieces(pieces) -> int:
    if isinstance(pieces, bool) or not isinstance(pieces, Integral) or pieces < 1:
        raise ValueError(f"the number of pieces must be a whole number >= 1, got {pieces!r}")
    return int(pieces)


def breakpoints(upper, pieces: int = PIECES) -> np.ndarray:
    """E1: `b_k = k U / K`, `k = 0 .. K`, for each upper limit (one row per limit)."""
    upper = np.asarray(upper, dtype=float).reshape(-1)
    return upper[:, None] * (np.arange(pieces + 1) / pieces)[None, :]


def approximate_square(power, upper: float, pieces: int = PIECES) -> np.ndarray:
    """E1: `f(r)`, the largest of the chord lines of `[0, upper]`, for each power in `power`."""
    b = breakpoints([upper], checked_pieces(pieces))[0]
    slope, intercept = b[:-1] + b[1:], b[:-1] * b[1:]
    power = np.asarray(power, dtype=float)
    return np.max(slope[None, :] * power.reshape(-1, 1) - intercept[None, :], axis=1).reshape(
        power.shape
    )


def largest_gap(upper, pieces: int = PIECES) -> np.ndarray:
    """E1: `U^2 / (4 K^2)`, the largest of `f(r) - r^2` on `[0, U]`, for each upper limit."""
    upper = np.asarray(upper, dtype=float)
    return upper**2 / (4 * checked_pieces(pieces) ** 2)


def share_goal(programme: Programme, pieces: int):
    """E1 in segment form on `programme`: `pieces` segment columns per car-hour whose upper limit
    is above 0, after the programme's own columns, and one row `p - (sum of its segments) = 0`
    each, added to `programme.eq`. Returns the fifth goal's cost and the bounds `low`, `high` of
    every column, the programme's own and the segments'."""
    columns = programme.p.reshape(-1)
    upper = programme.upper.reshape(-1)
    live = np.flatnonzero(upper > 0)
    segment = programme.n_variables + np.arange(live.size * pieces).reshape(live.size, pieces)
    b = breakpoints(upper[live], pieces)
    for k, j in enumerate(live):
        programme.eq.add(np.concatenate([[columns[j]], segment[k]]), [1.0] + [-1.0] * pieces, 0.0)
    total = programme.n_variables + segment.size
    low = np.concatenate([programme.low, np.zeros(segment.size)])
    high = np.concatenate([programme.high, np.diff(b, axis=1).reshape(-1)])
    cost = np.zeros(total)
    cost[segment.reshape(-1)] = (b[:, :-1] + b[:, 1:]).reshape(-1)
    return cost, low, high


def plan_hour(
    view: HourView, centre, n_recovery, *, time_limit, solve=None, pieces=PIECES
) -> HourPlan:
    """Goals 1 to 4 of `asa_programme`, then the fifth goal of E1 (module text)."""
    limit = checked_time_limit(time_limit)
    pieces = checked_pieces(pieces)
    solve = lp.solve_lp if solve is None else solve
    programme = build_programme(view, centre, n_recovery)
    outcomes, optima, x = solve_stages(programme, time_limit=limit, solve=solve, keep_last=True)
    if x is None:
        return HourPlan(tuple(outcomes), None, None)
    cost, low, high = share_goal(programme, pieces)
    result = solve_stage(programme, cost, low, high, time_limit=limit, solve=solve)
    outcomes.append(result.outcome)
    if result.outcome != "optimal":
        return HourPlan(tuple(outcomes), None, None)
    optima.append(float(result.objective))
    x = np.asarray(result.x, dtype=float)
    powers = np.clip(x[programme.p], 0.0, programme.upper)
    return HourPlan(tuple(outcomes), powers, tuple(optima))
