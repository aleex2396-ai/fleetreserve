"""Which goal of a linear-programme controller's hour programme ruled out tracking the request: the
goal test of cause "choice" in `asa_diagnosis_day` (a measurement; plan section 10).

The controllers are those of `asa_controller`: `AsaController` (method `asa`, the four goals of
`asa_programme.STAGES`) and `AsaEsController` (method `asa-es`, the five of
`asa_equal_share.STAGES`). For one hour `t` of one replayed path the diagnosis has the view the
controller saw (`controller.HourView`), the `asa_programme.HourPlan` the controller recorded for
that hour (`AsaController.hour_plans`: its stage optima), the most each car truly takes in that
hour from the controller's own state (`take`, the upper ends of `asa_diagnosis_day.deliverable`)
and the hour's request `y`.

Tracking test with the first `k` goals kept (`tracking_test`). The hour's programme is built
again by `asa_programme.build_programme` from the view and the controller's own centre and
recovery hours, so it is the programme the controller solved. The optima of goals 1 to `k` are
kept as rows, each by `asa_programme.keep_optimum` at the optimum the controller recorded (so at
that optimum plus `asa_programme.STAGE_TOLERANCE`), in the way the controller keeps an earlier
stage's optimum; for the fifth
goal of `asa-es` its segment columns and rows are added by `asa_equal_share.share_goal` with
`asa_equal_share.PIECES` pieces, as `asa_equal_share.plan_hour` adds them, before its row is
kept. Two rows and one bound are added: the hour's planned total, the sum over cars of the power
in hour `t`, at most `y + TOLERANCE` and at least `y - TOLERANCE`, and each car's power in hour `t`
at most its `take` (its upper limit of A3 kept when lower). A plan meeting them draws its planned
power on every car (`replay.take` draws `min(p, take)`), so it delivers within `TOLERANCE` of the
request: it tracks. That bound is sufficient for tracking, not necessary: a plan that puts less
than `TOLERANCE` in all above what its cars take can also track, and such a plan alone does not
make a test "optimal". The programme has cost 0 and is solved by `asa_programme.solve_stage` (so by
`lp.solve_lp` with the dual simplex method) with the time limit the caller gives. Outcome
"optimal": such a plan exists; "infeasible": none does; any other outcome (a timeout, "error",
"unbounded") answers nothing.

The goal that ruled out tracking (`forbidding_goal`): the tests are solved for `k = 1, 2, ...` up
to the number of goals, in order, and stop at the first that does not end "optimal". The answer is
the name of goal `k` at the first "infeasible" (keeping goal `k` ruled out every tracking plan
that goals 1 to `k - 1` allowed); `TIE` when every test ends "optimal" (a plan tied with the
controller's under every goal tracks, and the solver returned another); `NOT_DETERMINED` when a
test ends in another outcome. The test with no goal kept is not solved: `asa_diagnosis_day`
answers the same question in closed form (cause "belief": the programme's lower bounds are 0 and
its hour-`t` upper limits are the believed most, so with no goal kept a tracking plan exists
exactly when `sum_i min(most_i, take_i)` reaches `y - TOLERANCE`).
The outcomes of the tests solved are returned in order, so they can be counted.

Nothing here goes back into a controller: the programmes are new objects, built from the
recorded view.
"""

import numpy as np

from fleetreserve import asa_equal_share, asa_programme, lp
from fleetreserve.asa_controller import AsaController, AsaEsController
from fleetreserve.asa_programme import (
    HourPlan,
    build_programme,
    checked_time_limit,
    keep_optimum,
    solve_stage,
)
from fleetreserve.controller import HourView
from fleetreserve.replay import FAILURE_KWH

TOLERANCE = FAILURE_KWH  # kWh: a delivery within this of the request tracks it (replay.fails)
TIE = "tie"
NOT_DETERMINED = "not determined"


def goals_of(controller) -> tuple[str, ...] | None:
    """The goal names of the controller's hour programme, in order; None for a controller that
    solves no programme of `asa_controller`."""
    if isinstance(controller, AsaEsController):
        return asa_equal_share.STAGES
    if isinstance(controller, AsaController):
        return asa_programme.STAGES
    return None


def tracking_test(
    controller: AsaController,
    view: HourView,
    optima,
    take,
    request: float,
    kept: int,
    *,
    time_limit,
    solve=None,
) -> str:
    """The outcome of the tracking test with the first `kept` goals kept (module text)."""
    limit = checked_time_limit(time_limit)
    solve = lp.solve_lp if solve is None else solve
    goals = goals_of(controller)
    if goals is None or not 1 <= kept <= len(goals) or len(optima) != len(goals):
        raise ValueError("the tracking test needs a programme controller, its optima and 1 to K")
    programme = build_programme(view, controller.centre, controller.n_recovery)
    objectives = programme.objectives
    for j in range(min(kept, len(objectives))):
        keep_optimum(programme, objectives[j], optima[j])
    low, high = programme.low.copy(), programme.high.copy()
    if kept > len(objectives):  # the fifth goal of asa-es
        cost, low, high = asa_equal_share.share_goal(programme, asa_equal_share.PIECES)
        keep_optimum(programme, cost, optima[len(objectives)])
    first = programme.p[:, 0]
    high[first] = np.minimum(high[first], np.asarray(take, dtype=float))
    programme.le.add(first, 1.0, request + TOLERANCE)
    programme.le.add(first, -1.0, -(request - TOLERANCE))
    result = solve_stage(programme, np.zeros(low.size), low, high, time_limit=limit, solve=solve)
    return result.outcome


def forbidding_goal(
    controller: AsaController,
    view: HourView,
    plan: HourPlan,
    take,
    request: float,
    *,
    time_limit,
    solve=None,
) -> tuple[str, tuple[str, ...]]:
    """(the goal that ruled out tracking, or `TIE` or `NOT_DETERMINED`; the outcomes of the tests
    solved, in order) for one hour whose plan was optimal in every stage (module text)."""
    goals = goals_of(controller)
    if goals is None or not plan.optimal:
        raise ValueError("the goal test needs a programme controller's optimal hour plan")
    outcomes = []
    for k in range(1, len(goals) + 1):
        outcome = tracking_test(
            controller,
            view,
            plan.optima,
            take,
            request,
            k,
            time_limit=time_limit,
            solve=solve,
        )
        outcomes.append(outcome)
        if outcome == "infeasible":
            return goals[k - 1], tuple(outcomes)
        if outcome != "optimal":
            return NOT_DETERMINED, tuple(outcomes)
    return TIE, tuple(outcomes)
