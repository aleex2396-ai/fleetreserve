"""Why a tracking controller misses a request that another one meets, hour by hour on one day-path
(a measurement for `asa_diagnosis_run`; plan section 10). Pure: no files, no clock in the
results, so it can run in a process pool.

Both controllers (the "method" and the "reference") are replayed on the same day, path, requests,
hour-zero target, belief and reading of `departure_seen` by `controller_replay.run_controller`,
unchanged; each is wrapped so that the view it sees in every hour is kept. The wrapper returns the
controller's powers as they are, so the plans are those `compare_day` makes (`diagnose_day` builds
the inputs by the calls `compare_day.evaluate` makes, `tests/wp10/test_asa_diagnosis_tie_in.py`
compares them). A service hour `t` is missed by a controller when its replay fails it
(`replay.fails`: the fleet's delivery misses the request `y_t` by more than `replay.FAILURE_KWH`,
`TOLERANCE` here). Every service hour of the path is counted in one of four groups: missed by the
method only, by the reference only, by both, by neither. Each hour missed by the method only gets
one row (`HourRow`) and one cause.

The state of a controller at the start of hour `t`: the energy `x_i` each car has drawn since the
decision time under that controller's replay (`replay.Run.stored` after hour `t - 1`). The
deliverable range from a state (`deliverable`), computed on the TRUE cars (`replay.true_limits`,
`replay.protected`) in closed form, with no programme:

- car `i` truly present in hour `t` (`t < d_i`): upper end `high_i = min(r_i, max(e_i - x_i, 0))`,
  what `replay.take` lets it draw at most; lower end `low_i = min(max(0, P_i - x_i - r_i h_i),
  high_i)`, with `P_i = (1 - rho) e_i` its protected energy and `h_i = max(d_i - 1 - t, 0)` its
  hours after `t` before its departure, service and recovery hours alike (in each it can draw its
  full rate `r_i`, as `replay.after_recovery` lets it in the recovery hours). When `P_i` is still
  reachable from `x_i` (`P_i <= x_i + high_i + r_i h_i`), `low_i` is the smallest power in hour `t`
  that keeps it reachable by the car's true departure; when it is not, `low_i = high_i`, the only
  power that keeps the car's least reachable shortfall at departure as small as it still can be.
  The room never binds that reach, as `P_i <= e_i`.
- a car not truly present in hour `t`: `low_i = high_i = 0`.

The fleet's range is `[sum low_i, sum high_i]`: the smallest and largest fleet power deliverable
in hour `t` without making any car's protected energy unreachable by its true departure (or, for
a car whose protected energy is already out of reach, without making its shortfall larger). The
lower end lets each car draw its full true rate in every later hour, whatever the later requests.
A delivery is at most `sum high_i` (`replay.take`), so a reference that meets `y_t` has
`y_t <= high + TOLERANCE` from its own state.

The method's belief in hour `t`: the most it believes each car can take, `most_i`
(`priority_rules.most` of the method's view, the hour-`t` upper limit of `asa_programme` A3), and
`believed_on_true = sum_i min(most_i, high_i)`, the most a plan within the believed limits can
draw on the true cars. The deficit of a truly present car is `max(high_i - most_i, 0)`; it is
labelled by the term that sets `most_i`: "plugged" for a car not counted as plugged in; else, with
`f = min(max(d_hat_i - t, 0), 1)`, "departure" when `r_hat_i f <= cap_i - x_i` and `f < 1`,
"rate" when `r_hat_i f <= cap_i - x_i` and `f = 1`, "cap" when `cap_i - x_i < r_hat_i f` (the
believed energy cap). The deficits add up to `sum high_i - believed_on_true`.

Causes (`CAUSES`), tested in this order, the first that holds (so they partition the method-only
hours):

1. "state": `y_t > high_m + TOLERANCE`, the upper end from the method's own state. No plan could
   have met the request from that state; the reference met it from its own. Which cars are
   present does not depend on the controller, so earlier hours left less room on the cars still
   present under the method than under the reference.
2. "fallback": the method is a controller of `asa_controller` and its hour-`t` programme ended in
   an outcome other than "optimal" in some stage (its recorded `HourPlan` has no powers), so the
   hour was planned by its fall-back.
3. "belief": the method's delivery fell short (`delivered < y_t - TOLERANCE`) and
   `believed_on_true < y_t - TOLERANCE`: the request was deliverable on the true cars from the
   method's state, but no plan within the believed limits of hour `t` (believed departures,
   rates and energy caps) could deliver it.
4. "choice": the delivery fell short and `believed_on_true >= y_t - TOLERANCE`: a plan within the
   believed limits could have tracked on the true cars, and the plan made did not (it planned
   less than the request, or planned power on cars that could not draw it). As the replay draws
   `min(p_i, high_i)` from a car planned `p_i >= 0`, `shortfall = (y_t - sum_i p_i) +
   overplanned` (below). For a controller of `asa_controller`, goal 1 ("track") minimises the gap
   between the planned fleet total `sum_i p_i` and `y_t` over plans within the believed limits,
   and the later goals keep that optimum as a row; the optimum is `max(y_t - sum_i most_i, 0)`,
   at most `TOLERANCE` here. So the planned total is within `TOLERANCE` of the request, up to the
   kept row's tolerance, and the shortfall is, to within that, the power planned on cars above
   what they truly take.
   The goal that ruled out tracking is named by `asa_diagnosis_lp.forbidding_goal` (a goal name
   of its stages, `asa_diagnosis_lp.TIE` or `asa_diagnosis_lp.NOT_DETERMINED`). A goal `k` named
   means: every plan that keeps the optima of goals 1 to `k` puts more power on some car than
   the car truly takes (so on a car whose believed limit is above that), or plans a total more
   than `TOLERANCE` from the request, while some plan keeping goals 1 to `k - 1` does neither.
   The name does not say that the controller could have known from its belief which split would
   deliver. For any other controller the goal is `RULE` (the method's own rule chose the split).
5. "other": anything else. The delivery did not fall short, so, the hour being missed, it lay
   above `y_t + TOLERANCE`.

Per row (`HourRow`), besides the cause and goal: the request, both deliveries, the shortfall
`y_t - delivered` of the method (negative for a delivery above the request), whether the hour is
the method's first failure on the path, both ranges, `sum most_i`, `believed_on_true`, the
deficits by label, the method's power planned above what its cars truly take
(`sum_i max(p_i - high_i, 0)`, `overplanned`), the number of cars that left earlier than the
method believed (`cars_left_earlier`, by `left_earlier`), and how far the method had charged
ahead of the reference: `sum_i (x_m,i - x_r,i)` over the cars truly present in hour `t`
(`charged_ahead`).

A car left earlier than believed in hour `t` (`left_earlier`) when it is gone by hour `t`
(`d_i <= t`) and its believed departure is above its true departure hour (`d_hat_i > d_i`): the
method believed it could draw for part of hour `d_i`, the first hour it truly could not. True
departures are whole hours, the hour of unplugging rounded down (`day`), and believed ones may be
fractional, so a car believed to leave at 3.2 that left at 3.4 (`d_i = 3`) counts; a car whose
believed departure equals its true departure hour does not.

The belief is the same in every hour of a day (`compare_day` builds one per day and
`controller_replay` passes it to every hour), so "the departure the method believed at each
earlier hour" is one value per car, `d_hat_i`. For each row of cause "state", one `StateCar` per
car of the day: its true and believed departures, whether it is truly present in hour `t`,
whether it left earlier than believed, the energy it drew before hour `t` under each controller
and its upper end `high_i` from each state.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from fleetreserve import replay
from fleetreserve.asa_diagnosis_lp import forbidding_goal, goals_of
from fleetreserve.asa_programme import checked_time_limit
from fleetreserve.compare_day import sum_counts
from fleetreserve.compare_methods import MethodInput
from fleetreserve.controller import Belief, HourView, forecast_belief, hour_zero_target
from fleetreserve.controller_replay import run_controller
from fleetreserve.day import Day
from fleetreserve.gate_day import PreparedDay
from fleetreserve.plans import Plan
from fleetreserve.priority_rules import most

TOLERANCE = replay.FAILURE_KWH
CAUSES = ("state", "fallback", "belief", "choice", "other")
GROUPS = ("method_only", "reference_only", "both", "neither")
LABELS = ("plugged", "departure", "rate", "cap")
RULE = "rule"


def deliverable(day: Day, stored, hour: int) -> tuple[np.ndarray, np.ndarray]:
    """(`low_i`, `high_i`) of every car in hour `hour` from the state `stored` (module text)."""
    departure, rate, room = replay.true_limits(day)
    need = replay.protected(day)
    x = np.asarray(stored, dtype=float)
    present = hour < departure
    high = np.where(present, np.minimum(rate, np.maximum(room - x, 0.0)), 0.0)
    later = np.maximum(departure - 1 - hour, 0)
    low = np.where(present, np.minimum(np.maximum(need - x - rate * later, 0.0), high), 0.0)
    return low, high


def left_earlier(departure, believed_departure, hour: int) -> np.ndarray:
    """Per car, whether it left earlier than believed in hour `hour` (module text)."""
    departure = np.asarray(departure)
    return (departure <= hour) & (np.asarray(believed_departure, dtype=float) > departure)


def deficits(view: HourView, high) -> dict[str, float]:
    """The deficits `max(high_i - most_i, 0)` summed by label (module text)."""
    belief = view.belief
    limit = most(view)
    share = np.clip(belief.departure - view.hour, 0.0, 1.0)
    by_rate = belief.rate * share
    by_cap = belief.cap - view.drawn
    found = dict.fromkeys(LABELS, 0.0)
    for i, gap in enumerate(np.maximum(np.asarray(high, dtype=float) - limit, 0.0)):
        if gap <= 0:
            continue
        if not view.plugged[i]:
            label = "plugged"
        elif by_rate[i] <= by_cap[i]:
            label = "departure" if share[i] < 1 else "rate"
        else:
            label = "cap"
        found[label] += float(gap)
    return found


class Recorder:
    """A controller that returns another controller's powers unchanged and keeps every view."""

    def __init__(self, controller):
        self.controller = controller
        self.views: list[HourView] = []

    def __call__(self, view: HourView):
        self.views.append(view)
        return self.controller(view)


@dataclass(frozen=True)
class StateCar:
    car: int
    true_departure: int
    believed_departure: float
    present: bool
    left_earlier: bool
    method_drawn: float
    reference_drawn: float
    method_high: float
    reference_high: float


@dataclass(frozen=True)
class HourRow:
    path: int
    hour: int
    request: float
    method_delivered: float
    reference_delivered: float
    shortfall: float
    method_first_failure: bool
    method_low: float
    method_high: float
    reference_low: float
    reference_high: float
    believed_most: float
    believed_on_true: float
    deficit_plugged: float
    deficit_departure: float
    deficit_rate: float
    deficit_cap: float
    overplanned: float
    cars_left_earlier: int
    charged_ahead: float
    cause: str
    goal: str
    state_cars: tuple[StateCar, ...] = ()


@dataclass
class DayDiagnosis:
    """One day at one scale: the counts of the four groups over its paths and service hours, the
    rows of the method-only hours, the outcomes of the goal tests, the solver counts of each
    controller that has them (`compare_day.sum_counts` over the paths; None otherwise) and both
    controllers' replays (`replay.Run`) per path, in path order."""

    groups: Counter = field(default_factory=Counter)
    rows: list[HourRow] = field(default_factory=list)
    goal_tests: Counter = field(default_factory=Counter)
    method_solver_counts: dict | None = None
    reference_solver_counts: dict | None = None
    runs: list = field(default_factory=list)


def _replay(day, controller, belief, zeta, requests, target, seen):
    recorder = Recorder(controller)
    run = run_controller(
        day, recorder, belief, zeta, requests, hour_zero_target=target, departure_seen=seen
    )
    return run, recorder.views


def _stored_before(run: replay.Run, hour: int) -> np.ndarray:
    return run.stored[:, hour - 1] if hour > 0 else np.zeros(run.stored.shape[0])


def classify(
    day: Day,
    method,
    method_run: replay.Run,
    method_views,
    reference_run: replay.Run,
    hour: int,
    *,
    time_limit,
    solve=None,
):
    """(cause, goal, goal-test outcomes, ranges and belief figures) of one method-only hour."""
    view = method_views[hour]
    y = float(view.target)  # the request of service hour `hour` (controller_replay)
    x_m = _stored_before(method_run, hour)
    x_r = _stored_before(reference_run, hour)
    low_m, high_m = deliverable(day, x_m, hour)
    low_r, high_r = deliverable(day, x_r, hour)
    limit = most(view)
    believed_on_true = float(np.minimum(limit, high_m).sum())
    delivered = float(method_run.delivered[hour - 1])
    goals = goals_of(method)
    plan = method.hour_plans[hour] if goals is not None else None
    goal, outcomes = "", ()
    if y > high_m.sum() + TOLERANCE:
        cause = "state"
    elif plan is not None and not plan.optimal:
        cause = "fallback"
    elif delivered < y - TOLERANCE and believed_on_true < y - TOLERANCE:
        cause = "belief"
    elif delivered < y - TOLERANCE:
        cause = "choice"
        if plan is None:
            goal = RULE
        else:
            goal, outcomes = forbidding_goal(
                method, view, plan, high_m, y, time_limit=time_limit, solve=solve
            )
    else:
        cause = "other"
    figures = {
        "x_m": x_m,
        "x_r": x_r,
        "low_m": low_m,
        "high_m": high_m,
        "low_r": low_r,
        "high_r": high_r,
        "most": limit,
        "believed_on_true": believed_on_true,
        "deficits": deficits(view, high_m),
    }
    return cause, goal, outcomes, figures


def diagnose_path(
    day: Day,
    method,
    reference,
    belief: Belief,
    zeta,
    requests,
    *,
    hour_zero_target: float,
    departure_seen: str,
    time_limit,
    path: int = 0,
    solve=None,
):
    """(group counts, rows, goal-test outcomes, method run, reference run) of one path (module
    text). `method`, `reference`: controllers, each built fresh for this path. A time limit that
    is not a positive finite number is refused with `ValueError` before any hour is replayed."""
    checked_time_limit(time_limit)
    method_run, method_views = _replay(
        day, method, belief, zeta, requests, hour_zero_target, departure_seen
    )
    reference_run, _ = _replay(
        day, reference, belief, zeta, requests, hour_zero_target, departure_seen
    )
    departure, _, _ = replay.true_limits(day)
    groups: Counter = Counter()
    tests: Counter = Counter()
    rows = []
    for hour in day.service_hours:
        missed_m = bool(method_run.failed[hour - 1])
        missed_r = bool(reference_run.failed[hour - 1])
        group = (
            "method_only"
            if missed_m and not missed_r
            else "reference_only"
            if missed_r and not missed_m
            else "both"
            if missed_m
            else "neither"
        )
        groups[group] += 1
        if group != "method_only":
            continue
        cause, goal, outcomes, f = classify(
            day,
            method,
            method_run,
            method_views,
            reference_run,
            hour,
            time_limit=time_limit,
            solve=solve,
        )
        tests.update(outcomes)
        present = hour < departure
        early = left_earlier(departure, belief.departure, hour)
        planned = method_run.planned[:, hour]
        state_cars = ()
        if cause == "state":
            state_cars = tuple(
                StateCar(
                    i,
                    int(departure[i]),
                    float(belief.departure[i]),
                    bool(present[i]),
                    bool(early[i]),
                    float(f["x_m"][i]),
                    float(f["x_r"][i]),
                    float(f["high_m"][i]),
                    float(f["high_r"][i]),
                )
                for i in range(len(day.cars))
            )
        y = float(requests[hour - 1])
        delivered = float(method_run.delivered[hour - 1])
        rows.append(
            HourRow(
                path=path,
                hour=hour,
                request=y,
                method_delivered=delivered,
                reference_delivered=float(reference_run.delivered[hour - 1]),
                shortfall=y - delivered,
                method_first_failure=method_run.first_failure == hour,
                method_low=float(f["low_m"].sum()),
                method_high=float(f["high_m"].sum()),
                reference_low=float(f["low_r"].sum()),
                reference_high=float(f["high_r"].sum()),
                believed_most=float(f["most"].sum()),
                believed_on_true=f["believed_on_true"],
                deficit_plugged=f["deficits"]["plugged"],
                deficit_departure=f["deficits"]["departure"],
                deficit_rate=f["deficits"]["rate"],
                deficit_cap=f["deficits"]["cap"],
                overplanned=float(np.maximum(planned - f["high_m"], 0.0).sum()),
                cars_left_earlier=int(early.sum()),
                charged_ahead=float((f["x_m"] - f["x_r"])[present].sum()),
                cause=cause,
                goal=goal,
                state_cars=state_cars,
            )
        )
    return groups, rows, tests, method_run, reference_run


def diagnose_day(prepared: PreparedDay, beta: float, method, reference, *, time_limit):
    """A `DayDiagnosis` of one prepared day at scale `beta` > 0 on each of its audited paths, the
    inputs built as `compare_day.evaluate` builds them; `method`, `reference`: methods of
    `compare_methods` of kind "controller". A day that abstains is refused with `ValueError`."""
    beta = float(beta)
    if prepared.abstains or beta <= 0:
        raise ValueError("a day that abstains, or a scale at most 0, has nothing to diagnose")
    policy = prepared.certificate.policy
    plan = Plan.from_policy(policy, beta)
    target = hour_zero_target(plan)
    belief = forecast_belief(prepared.day)
    inputs = MethodInput(prepared.day, policy, beta, plan, prepared.rows)
    found = DayDiagnosis()
    counts = {"method": [], "reference": []}
    for path, zeta in enumerate(prepared.paths):
        requests = policy.request(zeta, beta)
        controllers = {"method": method.build(inputs), "reference": reference.build(inputs)}
        groups, rows, tests, method_run, reference_run = diagnose_path(
            prepared.day,
            controllers["method"],
            controllers["reference"],
            belief,
            zeta,
            requests,
            hour_zero_target=target,
            departure_seen=prepared.departure_seen,
            time_limit=time_limit,
            path=path,
        )
        found.groups.update(groups)
        found.rows += rows
        found.goal_tests.update(tests)
        found.runs.append((method_run, reference_run))
        for role, controller in controllers.items():
            if hasattr(controller, "solver_counts"):
                counts[role].append(controller.solver_counts())
    found.method_solver_counts = sum_counts(counts["method"])
    found.reference_solver_counts = sum_counts(counts["reference"])
    return found
