"""One hour's linear programme of this project's linear-programme form of ASA (plan section 10).

Plan section 10 names "ASA (ACN adaptive scheduling) as an hourly tracking MPC: acnportal 0.3.3
timeboxed to 2 hours, else a HiGHS LP checked on 5 ACN days". The adaptive scheduling algorithm
(ASA) of Lee et al. (2021) is not in acnportal 0.3.3, so the "else" branch is built: a
programme solved by HiGHS through `lp.solve_lp`. `asa_controller` runs it hour by hour.

What was read of the paper: Lee et al. (2021) in its arXiv version 2012.02636v1 (4 December
2020), pages 5 to 8: section V (its Algorithm 1, problem (4a)-(4b), the utility functions of
section V-B, the constraints (5a)-(5d) of section V-C, section V-D), section VI-A, VI-B and the
start of VI-C with the case study's objective. The rest of it and its journal version were not
read. Equation numbers are those of that version.

Readings of the plan's wording and of the paper (each is the closest reading; the alternative is
in brackets):

  A1. What is kept of ASA. Its receding horizon (Algorithm 1): a programme over the hours left,
      solved again at the start of every hour from the energy each car has drawn, of which only
      the first hour's powers are applied (section V-D: only the first charging rate is applied
      when one period passes between solves). Its per-car constraints: power between 0 and the
      car's rate limit while it is believed plugged in (5a), zero from its believed departure on
      (5b), and the energy delivered over the horizon at most a bound (5c). Its quick-charging
      utility as the last tie-break (A4, goal 4). Not kept: the three-phase infrastructure limits
      (5d), since plan section 3 has no site cap; the equal-share term `-sum r_i(t)^2` of the
      paper's ASA-QC objective, since this is a linear programme (`asa_equal_share` adds a
      linear form of it as a fifth goal, for the method `asa-es`); the rounding of rates to the
      pilot signals a charger allows and the smallest non-zero first rate (section V-D). The
      paper's (5c) bounds the energy by the car's remaining energy request; this programme bounds
      it by the believed cap minus the energy drawn (A3). [Alternative: bound it by the believed
      need minus the energy drawn.]
  A2. What the programme knows: the hour view (`controller.HourView`: the hour, its target, the
      energy drawn, the plugged-in flags and the belief), the centre `m` of the box the
      controller is built with, and the day's number of recovery hours. It never sees a later
      request, and it sees a true limit only through its belief: under the oracle belief of
      `controller` (R2) the belief is the cars' true departures, rates, rooms and protected
      energies, and in the oracle-departure variant of `compare_run` the belief's departures are
      the true departure hours (the oracle partner's forecasts, `gate_day`).
  A3. Horizon at hour `t`: hours `t` to `n` (the hours of the service window left; hour 0 is the
      decision hour). Variables: each car's planned power in each of those hours. Car `i`'s power
      in hour `s` lies in `[0, upper[i, s]]`: in hour `t` its `priority_rules.most` (0 for a car
      not counted as plugged in); in a later hour `s`, `r_hat f_s` with `f_s` the share of hour
      `s` before its believed departure, for a car counted as plugged in, and 0 otherwise. Its
      planned energy over the horizon is at most `max(cap - drawn, 0)`.
  A4. Goals in strict order, each one linear programme (a "stage"), each solved with the optima
      of the earlier stages kept as rows: an earlier stage's objective may not exceed its optimum
      by more than `STAGE_TOLERANCE`. These rows hold to the solver's tolerances, not exactly: a
      later stage's plan can exceed an earlier optimum by slightly more than `STAGE_TOLERANCE`
      (on the 300 seeded views of `tests/wp10/test_asa_goal_order.py`, measured with that file's
      own stage objectives at the plan returned: at most 2.0e-9 kWh above an earlier optimum;
      that file's test allows `STAGE_TOLERANCE` plus 1e-7 per car and service hour).
        1. "track": the smallest shortfall plus excess of the fleet's planned power in hour `t`
           against the hour's target (`HourView.target`), a target below 0 read as 0 (A7).
        2. "need": the smallest total believed unmet need over the cars counted as plugged in.
           Car `i`'s unmet need is `max(0, need - drawn - (its planned energy in hours t..n)
           - supply)`, with `supply` its believed rate times the recovery hours before its
           believed departure, counted as the forecast sets of plan section 2 count them
           (`day.ForecastCar.recovery_supply` of the belief's departure and rate: each recovery
           hour `n + 1 .. n + n_recovery` by its share). With the oracle belief the departure is
           a whole hour and the count equals `replay.after_recovery`'s `max(d - 1 - n, 0)` when
           `n_recovery` covers the car's departure.
        3. "centre": the smallest total absolute deviation of the fleet's planned power from the
           box centre `m_s` in each later service hour `s` (`t + 1 .. n`), a centre below 0
           read as 0 (A7). The later requests are unknown at hour `t`; the centre is the
           request at `zeta_s = 0`. [Alternative: plan against the worst corner of the box.]
        4. "quick": ASA's quick-charging utility, the largest sum over cars and hours of the
           planned power weighted by `n - s + 1`. The paper's weight, `(T - s' + 1) / T` for the
           `s'`-th of `T` hours of its horizon (section V-B), is this weight divided by `T` on
           the horizon `t..n`; dividing by the positive constant `T` changes no optimum.
      The solution may not be unique: goal 4 weighs hours, not cars, so a plan that moves energy
      between cars within an hour without changing any goal ties with it. Which of tied plans
      is returned is the solver's choice, and the same programme gives the same plan (the tests
      run it twice).
  A5. Every stage is solved by `lp.solve_lp` with the time limit the caller gives (no default)
      and the HiGHS dual simplex method, on sparse matrices (plan section 13 names interior
      point with crossover for trees and boxes; this programme is neither). A stage that ends
      in any outcome but "optimal" ends the hour's programme: no plan is returned and the later
      stages are not solved.
  A7. A request below 0. Every power of the programme is at least 0 (one-way), and a target or a
      centre below 0 is read as 0, as the priority rules read a target below 0
      (`priority_rules`, R4, which then plan nothing). With a target below 0, goal 1's optimum
      is an hour's total of 0, which the later goals may exceed by `STAGE_TOLERANCE`; in a later
      hour whose centre is below 0, goal 3 asks for as little as goal 2 allows. Beside it: the
      power limit `P(t)` of the smoothed least-laxity-first algorithm of Chen et al. (2022) is at
      least 0 (its eq. 2, as `priority_rules` reads it). [Alternative: refuse a box whose lower
      corner is below 0; in `compare_run` that refusal fails the unit and variant for every
      method.]

`plan_hour(view, centre, n_recovery, time_limit=..., solve=...)` returns a `HourPlan`: the outcome
of every stage solved, in order, and, when all four are optimal, the planned powers of hours
`t..n` (the solver's values clipped to each variable's bounds, which removes rounding outside
them) and the four optima. `solve` is `lp.solve_lp`, looked up when the hour is planned, unless a
caller passes another function with its signature (the tests pass stubs). Refused with
`ValueError`: a centre that is not one finite value per service hour, a number of recovery hours
that is not a whole number at least 0, a time limit that is not a positive finite number.

`plan_hour` is made of parts that `asa_equal_share` also calls: `build_programme` (the rows,
bounds and four objectives of A3 and A4, with the centre's and the recovery hours' refusals),
`solve_stages` (the four stages in order, each optimum kept as a row by `keep_optimum`; the
fourth's too when asked, which `plan_hour` does not ask) and `solve_stage` (one solve by A5).
"""

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import scipy.sparse as sp

from fleetreserve import lp
from fleetreserve.controller import HourView
from fleetreserve.day import ForecastCar
from fleetreserve.priority_rules import most

STAGES = ("track", "need", "centre", "quick")  # A4, in order
STAGE_TOLERANCE = 1e-9  # kWh: an earlier stage's optimum, kept as a row, may be exceeded by this
METHOD = "simplex"  # A5


@dataclass(frozen=True)
class HourPlan:
    """One hour's programme. `outcomes`: the outcome of each stage solved, in the order of
    `STAGES` (of `asa_equal_share.STAGES` for its programme); it ends at the first that is not
    "optimal". `powers` (cars by hours `t..n`) and `optima` (one per stage, each the minimised
    objective) are set only when every stage is optimal."""

    outcomes: tuple[str, ...]
    powers: np.ndarray | None
    optima: tuple[float, ...] | None

    @property
    def optimal(self) -> bool:
        return self.powers is not None


def checked_time_limit(time_limit) -> float:
    """The time limit as a positive finite number of seconds, or `ValueError`."""
    if isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)):
        raise ValueError(f"the time limit must be a positive finite number, got {time_limit!r}")
    if not (math.isfinite(time_limit) and time_limit > 0):
        raise ValueError(f"the time limit must be a positive finite number, got {time_limit!r}")
    return float(time_limit)


def checked_centre(centre, n_service: int) -> np.ndarray:
    """The box centre as one finite value per service hour, or `ValueError`."""
    array = np.array(centre, dtype=float)
    if array.shape != (n_service,):
        raise ValueError(f"the centre must hold {n_service} values, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("the centre must be finite")
    return array


def checked_recovery_hours(n_recovery) -> int:
    if isinstance(n_recovery, bool) or not isinstance(n_recovery, Integral) or n_recovery < 0:
        raise ValueError(f"the recovery hours must be a whole number >= 0, got {n_recovery!r}")
    return int(n_recovery)


def upper_bounds(view: HourView) -> np.ndarray:
    """A3: each car's largest power in each hour `t..n` (cars by hours)."""
    belief = view.belief
    later = np.arange(view.hour + 1, view.n_service + 1)
    share = np.clip(belief.departure[:, None] - later[None, :], 0.0, 1.0)
    future = np.where(view.plugged[:, None], belief.rate[:, None] * share, 0.0)
    return np.column_stack([most(view), future])


def supply(view: HourView, n_recovery: int) -> np.ndarray:
    """A4, goal 2: each car's believed recovery supply, as the forecast sets count it."""
    belief = view.belief
    return np.array(
        [
            ForecastCar(float(d), float(r), 0.0, 0.0).recovery_supply(view.n_service, n_recovery)
            for d, r in zip(belief.departure, belief.rate, strict=True)
        ]
    )


class _Rows:
    """Rows `sum_j value * x_j` against `rhs` (`<=` in `Programme.le`, `==` in `Programme.eq`),
    collected as coordinate entries."""

    def __init__(self):
        self.row, self.column, self.value, self.rhs = [], [], [], []

    def add(self, columns, values, rhs: float):
        columns = np.asarray(columns, int).reshape(-1)
        self.row.append(np.full(columns.size, len(self.rhs)))
        self.column.append(columns)
        self.value.append(np.broadcast_to(np.asarray(values, float), columns.shape))
        self.rhs.append(float(rhs))

    def matrix(self, n_variables: int):
        if not self.rhs:
            return None, None
        matrix = sp.csr_matrix(
            (np.concatenate(self.value), (np.concatenate(self.row), np.concatenate(self.column))),
            shape=(len(self.rhs), n_variables),
        )
        return matrix, np.array(self.rhs)


@dataclass
class Programme:
    """The hour's programme of A3 and A4, as `build_programme` returns it. `p`: the column of each
    planned power (cars by hours `t..n`); `upper`: their upper limits; `low`, `high`: the bounds
    of the programme's own columns (`n_variables` of them); `le`, `eq`: the rows `<=` and `==`;
    `objectives`: the cost of each stage of `STAGES`, in order. The rows grow after it is built:
    `le` by one row per stage optimum kept (`solve_stages`, `keep_optimum`), and
    `asa_equal_share.plan_hour` adds to `eq` one row per car-hour whose upper limit is above 0,
    over that car-hour's power column and its segment columns, which lie beyond the programme's
    own."""

    p: np.ndarray
    upper: np.ndarray
    low: np.ndarray
    high: np.ndarray
    le: _Rows
    eq: _Rows
    objectives: tuple[np.ndarray, ...]

    @property
    def n_variables(self) -> int:
        return int(self.low.size)


def build_programme(view: HourView, centre, n_recovery) -> Programme:
    """The programme of A3 and A4 for the hour of `view` (module text)."""
    centre = np.maximum(checked_centre(centre, view.n_service), 0.0)  # A7
    target = max(view.target, 0.0)  # A7
    n_recovery = checked_recovery_hours(n_recovery)
    t, n, cars = view.hour, view.n_service, view.n_cars
    length = n - t + 1
    upper = upper_bounds(view)
    room = np.maximum(view.belief.cap - view.drawn, 0.0)
    gap = view.belief.need - view.drawn - supply(view, n_recovery)
    plugged = np.flatnonzero(view.plugged)

    # Variables: p (cars by hours t..n), short and excess of hour t, the unmet need of each car
    # counted as plugged in, and above and below the centre in each later hour.
    p = np.arange(cars * length).reshape(cars, length)
    short, excess = cars * length, cars * length + 1
    unmet = excess + 1 + np.arange(plugged.size)
    above = excess + 1 + plugged.size + np.arange(length - 1)
    below = above + (length - 1)
    n_variables = excess + 1 + plugged.size + 2 * (length - 1)
    low = np.zeros(n_variables)
    high = np.full(n_variables, np.inf)
    high[p.reshape(-1)] = upper.reshape(-1)

    le, eq = _Rows(), _Rows()
    eq.add(np.concatenate([p[:, 0], [short, excess]]), [1.0] * cars + [1.0, -1.0], target)
    for i in range(cars):
        le.add(p[i], 1.0, room[i])
    for k, i in enumerate(plugged):  # unmet >= gap - planned energy
        le.add(np.concatenate([p[i], [unmet[k]]]), -1.0, -gap[i])
    for j in range(1, length):  # planned total - centre = above - below
        columns = np.concatenate([p[:, j], [above[j - 1], below[j - 1]]])
        eq.add(columns, [1.0] * cars + [-1.0, 1.0], centre[t + j - 1])

    weight = (n - np.arange(t, n + 1) + 1).astype(float)
    objectives = []
    for stage in STAGES:
        cost = np.zeros(n_variables)
        if stage == "track":
            cost[[short, excess]] = 1.0
        elif stage == "need":
            cost[unmet] = 1.0
        elif stage == "centre":
            cost[above] = 1.0
            cost[below] = 1.0
        else:
            cost[p] = -weight[None, :]
        objectives.append(cost)
    return Programme(p, upper, low, high, le, eq, tuple(objectives))


def solve_stage(programme: Programme, cost, low, high, *, time_limit: float, solve):
    """One stage: minimise `cost` over the programme's rows with the column bounds `low`, `high`
    (as many columns as `cost`; a column beyond the programme's own has no entry in `le`, and
    has entries in `eq` only where the caller added them, as `asa_equal_share.plan_hour` does for
    its segment columns), by `solve` with the method of A5. Returns `solve`'s `LpResult`."""
    columns = np.asarray(cost).size
    a_ub, b_ub = programme.le.matrix(columns)
    a_eq, b_eq = programme.eq.matrix(columns)
    return solve(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=np.column_stack([low, high]),
        time_limit=time_limit,
        method=METHOD,
    )


def keep_optimum(programme: Programme, cost, optimum: float) -> None:
    """A4: the row `cost @ x <= optimum + STAGE_TOLERANCE`, added to the programme's `le`."""
    columns = np.flatnonzero(cost)
    programme.le.add(columns, cost[columns], optimum + STAGE_TOLERANCE)


def solve_stages(programme: Programme, *, time_limit: float, solve, keep_last: bool):
    """The stages of `STAGES` in order, each optimum kept as a row for the next (`keep_optimum`);
    the last one's too when `keep_last`. Stops at the first outcome that is not "optimal".
    Returns the outcomes, the optima and the last stage's solution (None, with the optima, when
    a stage is not optimal)."""
    outcomes, optima = [], []
    x = None
    last = len(programme.objectives) - 1
    for k, cost in enumerate(programme.objectives):
        result = solve_stage(
            programme, cost, programme.low, programme.high, time_limit=time_limit, solve=solve
        )
        outcomes.append(result.outcome)
        if result.outcome != "optimal":
            return outcomes, optima, None
        optima.append(float(result.objective))
        x = np.asarray(result.x, dtype=float)
        if k < last or keep_last:
            keep_optimum(programme, cost, result.objective)
    return outcomes, optima, x


def plan_hour(view: HourView, centre, n_recovery, *, time_limit, solve=None) -> HourPlan:
    """The four stages of A4 for the hour of `view` (module text)."""
    limit = checked_time_limit(time_limit)
    solve = lp.solve_lp if solve is None else solve
    programme = build_programme(view, centre, n_recovery)
    outcomes, optima, x = solve_stages(programme, time_limit=limit, solve=solve, keep_last=False)
    if x is None:
        return HourPlan(tuple(outcomes), None, None)
    powers = np.clip(x[programme.p], 0.0, programme.upper)
    return HourPlan(tuple(outcomes), powers, tuple(optima))
