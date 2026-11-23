"""Why one re-solve is infeasible, and how far it is from feasible (a measurement; plan section 5).

Everything here reads one re-solve as the rung made it: its inputs (`resolve.Remaining`, the
`resolve.Guard`, one-way or two-way, the budget). The programme is the one `resolve.resolve`
solves, taken row for row from `resolve.draft`; nothing is rebuilt here. No result of this module
goes back into a rung.

Requirement groups. `resolve.draft` names the requirement each row writes (`resolve.ROW_KINDS`).
`GROUPS` gathers those kinds into the groups that are removed:

- "tracking": the kinds "centre" and "width", the present cars sharing every request of the hours
  after `t` exactly (the box that remains);
- "request": the kind "request", the present cars meeting the known request of hour `t`;
- "guard": each present car's planned stored energy at least the predecessor's under every future
  request;
- "need": each present car's needed (protected) energy by the end of the window;
- "power": each present car's power at most `min(r_hat f_s, c)`, where the departure bound
  (survival-tightened) acts through the shares `f_s`;
- "floor": each present car's power at least 0 (two-way: at least `-r_hat f_s`);
- "cap": each present car's stored energy at most its cap (two-way: also at least `-B`).

The rows of kind "spread" only define the variables `z >= |.|` that the other rows use; they are
kept in every programme below. The re-solve has no driver-risk rows (plan section 5: they "enter
the decision-time LP only"), so there is no such group.

Every programme below is a feasibility question or a small optimisation over the same variables
`h`, `H`, `z` as the re-solve, solved through `sparse_programme.SparseProgramme` (so through
`lp.solve_lp`) with a time limit, by default the plan's 300 seconds for other programmes. A
relaxed programme's answer is "feasible" (the solver ended optimal), "infeasible" (proved by the
solver), "timeout" or "error"; a timeout is never read as infeasible.

Solver errors. The interior-point method ended in HiGHS status 4 ("solve error") on some relaxed
programmes. For these programmes only (never for the re-solve's own solve in `resolve.resolve`),
`solve_programme` retries once with the dual simplex method when the first attempt ended in
"error", with the same finite time limit. The retry's outcome replaces the first. Only an
"error" is retried: a first attempt that ended in a timeout stays a timeout, and an infeasible or
optimal first attempt stays as it is. `diagnose` counts the programmes whose first attempt ended
in an error (`Diagnosis.first_errors`) and, of those, the retries that ended optimal or
infeasible (`Diagnosis.retried_answered`); a retry that ended in an error or a timeout is
counted by its own outcome.

1. Recheck: every row of the re-solve, objective 0. It asks again whether the rows have a common
   point, without the re-solve's objective.
2. One group removed: every row of the re-solve except the rows of that group, objective 0. One
   programme per group of `GROUPS`.
3. Two groups removed: the same with the rows of two groups left out; one programme per pair.
   Solved only when every programme of step 2 ended "infeasible" (`diagnose`).
4. Departure bound relaxed (a relaxation, not a removal): every row of the re-solve, but the right
   side of each row of kind "power" is `min(r_hat, c)` instead of `min(r_hat f_s, c)`: every
   present car is taken to stay through the hour. The needed energy and its recovery supply keep
   the values of the re-solve, so the programme only grows. Objective 0.
5. Box scale (a proven-exact measure; the argument follows). The question: what is the largest
   `s` in `[0, 1]` such that the re-solve is feasible in the world whose remaining box is the
   centre plus `s` times its half-widths? That world changes three things and nothing else:
   (a) the request variable of each hour after `t` is `y = m + s beta w zeta'` with `zeta'` in
   `[-1, 1]` (the request of hour `t` is already known and is not scaled; the known `zeta_1 ..
   zeta_t` are not changed); (b) the new plan, written in `zeta'`, must have the present cars
   share every remaining request exactly, so each row of kind "width",
   `sum_p H[p, j, j - 1] = beta w_{t + j}`, becomes `sum_p H[p, j, j - 1] = s beta w_{t + j}`;
   (c) the predecessor plan reads the normalised request of the sold box,
   `zeta = (y - m) / (beta w)`, which is `s zeta'` in the shrunken world, so the planned power it
   gives each car in a later hour is `offset + s * (gains . zeta')` and its gains on future
   requests are `s` times those the re-solve has. The guard rows ask the new plan's energy to be
   at least the predecessor's for every `zeta'`; a guard row has the form
   `-sum_j h[p, j] + sum_k z_k <= -constant - owed` with `z_k >= |sum_j H[p, j, k] - gain_k|` for
   the gain of the predecessor's planned energy on `zeta_{t + 1 + k}` (`gain_k`). In the
   shrunken world `gain_k` becomes `s gain_k`, so the two rows that define `z_k`,
   `+-(sum_j H[p, j, k]) - z_k <= +-gain_k`, become `+-(sum_j H[p, j, k]) - z_k -+ s gain_k <= 0`.
   The constants (`offset`, `owed`, the known requests) do not change. Every one of these changes
   is linear in `s`, so the programme stays a linear programme in `h`, `H`, `z` and `s`. The
   programme maximises `s`.

   Which rows are scaled: the "width" rows with a non-zero right side, and the "spread" rows with
   a non-zero right side. A "spread" row has a non-zero right side only when it defines a `z_k`
   of a guard row: `resolve.draft` calls `spread` with no constant for the power and stored-energy
   rows (right side 0) and with the guard's `-gains` for the guard rows (right side `+-gain_k`).

   Exactness. Part 1: for every `s` in `[0, 1]` the rows written are the shrunken world's rows.
   One-way, the predecessor's planned energy is `sum_s max(p_s, 0)` and the guard asks
   `new energy >= sum_{s in S} p_s` for every subset `S` of the hours, which is the same
   condition (the largest of the subset sums is `sum max(p_s, 0)`); `resolve.draft` writes only
   the subsets `A + C`, where `A` holds the hours with `offset - spread >= -SIGN_TOLERANCE` (never
   negative), `C` runs over the subsets of the hours that can take both signs, and the hours
   with `offset + spread <= SIGN_TOLERANCE` (never positive) are left out, `spread` being the sum
   of the absolute gains. These sets are those of `s = 1`. At a scale `s <= 1` the lower end
   `offset - s spread` is no lower and the upper end `offset + s spread` no higher, so every hour
   of `A` is still never negative and every hour left out is still never positive; an hour of
   the mixed set may become one or the other at `s`, and the subsets over the mixed set still
   contain the right choice for every `zeta'`. So the subsets written at `s = 1` give
   `sum_{A} p + sum_{mixed} max(p, 0)`, which is the predecessor's planned energy up to
   `SIGN_TOLERANCE` per hour, the tolerance the re-solve itself uses; each further row is
   implied. Rows for subsets that are not needed at `s` are valid and only repeat a weaker row.
   Part 2: feasibility is monotone in `s`. If `(h, H, z)` is feasible at `s` and `0 <= s' < s`,
   then `(h, (s'/s) H, (s'/s) z)` is feasible at `s'`: the width rows scale to `s'`, every
   other row bounds a sum of `z` or a power of `H` from above and these only shrink, and the
   defining rows of the guard's `z_k` are homogeneous in `(H, z, s)` once `h` is fixed. So the
   feasible values of `s` are `[0, s*]` or none; `s*` is the optimum, and the answer is "none"
   (outcome "infeasible") exactly when `s = 0` is infeasible. At `s = 1` the programme is the
   re-solve's own, so for an infeasible re-solve `s*` is below 1 or there is none. The tests
   check the argument against `resolve.resolve` itself, run on inputs scaled directly (the
   half-widths of the hours after `t` and the predecessor's gains on future requests multiplied
   by `s`, which makes `resolve.draft` enumerate its subsets again at that `s`).

6. Need slack: the variables of the re-solve and one more per present car, `sigma_p >= 0`. Every
   row of the re-solve, except that each row of kind "need",
   `-sum_j h[p, j] + sum(z) <= x_p + supply_p - need_p`, becomes
   `-sum_j h[p, j] + sum(z) - sigma_p <= x_p + supply_p - need_p`. Minimise `sum_p sigma_p`: the
   smallest total energy (kWh), summed over cars, by which the needed energy must be lowered for
   the other rows to hold, with the box as it is. "infeasible" here means no slack on the needed
   energy alone makes the re-solve feasible.
7. Request miss: the variables of the re-solve and two more, `short >= 0` and `over >= 0`. Every
   row of the re-solve, except that the row of kind "request", `sum_p h[p, 0] = y_t`, becomes
   `sum_p h[p, 0] + short - over = y_t`. Minimise `short + over`; the miss is `short - over`
   (kWh): the smallest amount by which the present cars' delivery in hour `t` must differ from
   the known request for the other rows to hold, above 0 when they deliver less. "infeasible"
   here means no miss in hour `t` alone makes the re-solve feasible.
"""

import itertools
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from fleetreserve.lp import OTHER_SECONDS, LpResult, solve_lp
from fleetreserve.resolve import Guard, Remaining, draft
from fleetreserve.sparse_programme import SparseProgramme

GROUPS = {
    "tracking": ("centre", "width"),
    "request": ("request",),
    "guard": ("guard",),
    "need": ("need",),
    "power": ("power",),
    "floor": ("floor",),
    "cap": ("cap", "budget"),
}
SCALED_KINDS = ("width", "spread")  # the rows with a non-zero right side that carry `s` (step 5)
FEASIBLE, INFEASIBLE, TIMEOUT, ERROR = "feasible", "infeasible", "timeout", "error"


def pair_name(first: str, second: str) -> str:
    return f"{first}+{second}"


PAIRS = tuple(itertools.combinations(GROUPS, 2))


@dataclass(frozen=True)
class Retried(LpResult):
    """The result of the retry after a first attempt that ended in an error (`first_outcome`)."""

    first_outcome: str = "error"


def maximise(programme: SparseProgramme, columns, values, time_limit: float, method: str):
    """The largest `sum_k values[k] * x[columns[k]]` of `programme` with the named `lp.solve_lp`
    method. It is `SparseProgramme.maximise` with the method a choice (that one fixes the
    interior-point method, and this module may not change it); the same programme, bounds and
    objective, and the same finite time limit."""
    cost = np.zeros(programme.n_variables)
    np.add.at(cost, np.asarray(columns, int).reshape(-1), -np.asarray(values, float).reshape(-1))
    a_ub, b_ub = programme._matrix("le")
    a_eq, b_eq = programme._matrix("eq")
    bounds = np.column_stack([np.concatenate(programme._low), np.concatenate(programme._high)])
    return solve_lp(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        time_limit=time_limit,
        method=method,
    )


def retrying(first: Callable, second: Callable) -> Callable:
    """A `solve` that runs `first` and, only when it ended in an "error", `second` once, with the
    same arguments. A timeout, an infeasible or an optimal first attempt is returned as it is; the
    second attempt's result (a `Retried`) replaces an error."""

    def solve(programme, columns, values, time_limit):
        result = first(programme, columns, values, time_limit)
        if result.outcome != "error":
            return result
        again = second(programme, columns, values, time_limit)
        return Retried(
            again.outcome,
            again.x,
            again.objective,
            again.seconds,
            again.time_limit,
            again.status,
            again.message,
            result.outcome,
        )

    return solve


def _interior_point(programme, columns, values, time_limit):
    return maximise(programme, columns, values, time_limit, "ipm")


def _simplex(programme, columns, values, time_limit):
    return maximise(programme, columns, values, time_limit, "simplex")


solve_programme = retrying(_interior_point, _simplex)
solve_programme.__doc__ = (
    "The largest `sum_k values[k] * x[columns[k]]` of `programme`, within `time_limit`: the "
    "interior-point method, and the dual simplex method once more after an error (module text)."
)


@dataclass(frozen=True)
class Diagnosis:
    """One re-solve measured (module text). `single[group]`, `pair["a+b"]`: the answer with those
    rows removed (`pair` is None when the pairs were not solved). `box_scale`, `need_slack` and
    `request_miss` are set only when their programme ended optimal; `*_outcome` is the solver's
    outcome word. `first_errors`: the programmes whose first attempt ended in an error;
    `retried_answered`: of those, the retries that ended optimal or infeasible (module text)."""

    recheck: str
    single: dict
    pair: dict | None
    departure_relaxed: str
    box_scale: float | None
    box_scale_outcome: str
    need_slack: float | None
    need_slack_outcome: str
    request_miss: float | None
    request_miss_outcome: str
    first_errors: int = 0
    retried_answered: int = 0

    def answers(self) -> list[str]:
        """Every feasibility answer of this diagnosis (steps 1 to 4 of the module text)."""
        pairs = [] if self.pair is None else list(self.pair.values())
        return [self.recheck, *self.single.values(), *pairs, self.departure_relaxed]

    def outcomes(self) -> list[str]:
        """The outcome word of every relaxed programme solved (steps 1 to 7)."""
        distances = [self.box_scale_outcome, self.need_slack_outcome, self.request_miss_outcome]
        return [*self.answers(), *distances]


def _answer(outcome: str) -> str:
    if outcome == "optimal":
        return FEASIBLE
    return outcome if outcome in (INFEASIBLE, TIMEOUT) else ERROR


def _staying(state: Remaining) -> Remaining:
    """The state with every present car's departure bound after the last hour (step 4)."""
    present = np.asarray(state.present, bool)
    return replace(state, departure=np.where(present, np.inf, np.nan))


def assemble(
    state: Remaining,
    guard: Guard,
    *,
    two_way: bool,
    budget: float | None,
    drop=(),
    stay: bool = False,
    scale: bool = False,
    slack: bool = False,
    miss: bool = False,
):
    """The re-solve's programme with the changes named: `drop`, groups whose rows are left out;
    `stay`, step 4's right sides of the "power" rows; `scale`, step 5's variable `s` (in the
    "width" rows and in the "spread" rows of the guard, whose right side moves to `s`); `slack`,
    step 6's variables `sigma`; `miss`, step 7's variables `short` and `over`. Returns
    `(programme, s, sigma, short_over)`: the programme with its rows added, the index of `s`
    (None without `scale`), the indices of `sigma` (empty without `slack`) and those of `short`
    and `over` (empty without `miss`)."""
    unknown = sorted(set(drop) - set(GROUPS))
    if unknown:
        raise ValueError(f"unknown requirement groups {unknown}; the groups are {list(GROUPS)}")
    dropped = {kind for group in drop for kind in GROUPS[group]}
    built = draft(state, guard, two_way=two_way, budget=budget)
    power = None
    if stay:
        other = draft(_staying(state), guard, two_way=two_way, budget=budget)
        power = [r for r, kind in zip(other.le.rhs, other.le.kind, strict=True) if kind == "power"]
    programme = built.programme
    s = int(programme.variables(1, 0.0, 1.0)[0]) if scale else None
    sigma = programme.variables(built.present.size if slack else 0, 0.0, np.inf)
    short_over = programme.variables(2 if miss else 0, 0.0, np.inf)
    for sense, rows in (("le", built.le), ("eq", built.eq)):
        row, column, value, rhs = [], [], [], []
        seen: Counter = Counter()
        for columns, values, bound, kind in zip(
            rows.column, rows.value, rows.rhs, rows.kind, strict=True
        ):
            index = seen[kind]
            seen[kind] += 1
            if kind in dropped:
                continue
            if kind == "power" and power is not None:
                bound = power[index]
            if scale and bound != 0 and kind in SCALED_KINDS:
                columns, values, bound = np.append(columns, s), np.append(values, -bound), 0.0
            if kind == "need" and slack:
                columns, values = np.append(columns, sigma[index]), np.append(values, -1.0)
            if kind == "request" and miss:
                columns, values = np.append(columns, short_over), np.append(values, [1.0, -1.0])
            row.append(np.full(columns.size, len(rhs)))
            column.append(columns)
            value.append(values)
            rhs.append(bound)
        if rhs:
            programme.rows(
                sense,
                np.concatenate(row),
                np.concatenate(column),
                np.concatenate(value),
                np.array(rhs),
            )
    return programme, s, sigma, short_over


def feasibility(
    state: Remaining,
    guard: Guard,
    *,
    two_way: bool,
    budget: float | None,
    drop=(),
    stay: bool = False,
    time_limit: float = OTHER_SECONDS,
    solve: Callable = solve_programme,
) -> str:
    """Steps 1 to 4 of the module text: whether the rows kept have a common point."""
    programme, *_ = assemble(state, guard, two_way=two_way, budget=budget, drop=drop, stay=stay)
    return _answer(solve(programme, [0], [0.0], time_limit).outcome)


def box_scale(
    state: Remaining,
    guard: Guard,
    *,
    two_way: bool,
    budget: float | None,
    time_limit: float = OTHER_SECONDS,
    solve: Callable = solve_programme,
) -> tuple[float | None, str]:
    """Step 5 of the module text: `(s, outcome)`, `s` the largest scale of the remaining box at
    which the re-solve is feasible, None unless the outcome is "optimal". The outcome
    "infeasible" is the answer "none": not even the centre (`s = 0`) is feasible."""
    programme, s, *_ = assemble(state, guard, two_way=two_way, budget=budget, scale=True)
    result = solve(programme, [s], [1.0], time_limit)
    if result.outcome != "optimal":
        return None, result.outcome
    return float(np.clip(result.x[s], 0.0, 1.0)), "optimal"


def need_slack(
    state: Remaining,
    guard: Guard,
    *,
    two_way: bool,
    budget: float | None,
    time_limit: float = OTHER_SECONDS,
    solve: Callable = solve_programme,
) -> tuple[float | None, str]:
    """Step 6 of the module text: `(total slack in kWh, outcome)`, the slack None unless the
    outcome is "optimal"."""
    programme, _, sigma, _ = assemble(state, guard, two_way=two_way, budget=budget, slack=True)
    if sigma.size == 0:
        result = solve(programme, [0], [0.0], time_limit)
        return (0.0 if result.outcome == "optimal" else None), result.outcome
    result = solve(programme, sigma, -np.ones(sigma.size), time_limit)
    if result.outcome != "optimal":
        return None, result.outcome
    return float(np.maximum(result.x[sigma], 0.0).sum()), "optimal"


def request_miss(
    state: Remaining,
    guard: Guard,
    *,
    two_way: bool,
    budget: float | None,
    time_limit: float = OTHER_SECONDS,
    solve: Callable = solve_programme,
) -> tuple[float | None, str]:
    """Step 7 of the module text: `(miss in kWh, outcome)`, the miss None unless the outcome is
    "optimal"."""
    programme, _, _, pair = assemble(state, guard, two_way=two_way, budget=budget, miss=True)
    result = solve(programme, pair, [-1.0, -1.0], time_limit)
    if result.outcome != "optimal":
        return None, result.outcome
    short, over = np.maximum(result.x[pair], 0.0)
    return float(short - over), "optimal"


def diagnose(
    state: Remaining,
    guard: Guard,
    *,
    two_way: bool,
    budget: float | None,
    time_limit: float = OTHER_SECONDS,
    solve: Callable = solve_programme,
) -> Diagnosis:
    """Every programme of the module text for one re-solve. The pairs (step 3) are solved only
    when every single removal (step 2) ended "infeasible"; after a timeout or an error among the
    single removals it is not known that no single group is enough, so no pair is solved."""
    solved: list[LpResult] = []

    def recording(programme, columns, values, limit):
        solved.append(solve(programme, columns, values, limit))
        return solved[-1]

    common = dict(two_way=two_way, budget=budget, time_limit=time_limit, solve=recording)
    recheck = feasibility(state, guard, **common)
    single = {group: feasibility(state, guard, drop=(group,), **common) for group in GROUPS}
    pair = None
    if all(answer == INFEASIBLE for answer in single.values()):
        pair = {pair_name(a, b): feasibility(state, guard, drop=(a, b), **common) for a, b in PAIRS}
    staying = feasibility(state, guard, stay=True, **common)
    scale, scale_outcome = box_scale(state, guard, **common)
    slack, slack_outcome = need_slack(state, guard, **common)
    miss, miss_outcome = request_miss(state, guard, **common)
    first = [r for r in solved if getattr(r, "first_outcome", None) == ERROR]
    return Diagnosis(
        recheck,
        single,
        pair,
        staying,
        scale,
        scale_outcome,
        slack,
        slack_outcome,
        miss,
        miss_outcome,
        len(first),
        sum(r.outcome in ("optimal", INFEASIBLE) for r in first),
    )


def cause(found: Diagnosis) -> str:
    """One label for the removals that make the re-solve feasible: the single groups (joined by
    "|") when there are any; else the pairs when they were solved and some are feasible; else
    "no single group and no pair"; "not determined" when no single group is feasible and a
    single removal ended in a timeout or an error."""
    singles = [group for group, answer in found.single.items() if answer == FEASIBLE]
    if singles:
        return "|".join(singles)
    if found.pair is None:
        return "not determined"
    pairs = [name for name, answer in found.pair.items() if answer == FEASIBLE]
    if pairs:
        return "|".join(pairs)
    if any(answer != INFEASIBLE for answer in found.pair.values()):
        return "not determined"
    return "no single group and no pair"
