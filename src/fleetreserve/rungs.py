"""The rungs of plan section 5 on one day-path: the fixed plan, one-way and two-way re-solve.

Rung "fixed" runs the fixed plan (`plans.Plan.from_policy`) one-way. Rung "one-way" copies the
fixed plan, rung "two-way" (budget `B`, plan section 9) copies rung "one-way". "Each copies its
predecessor until that fails, then solves a robust causal LP on the remaining hours"
(`resolve.resolve`); "if infeasible or timed out, the predecessor's plan stays".

Information model. A rung decides at the start of each hour, from what it observes then. A
true car's departure `d` is its plug-out hour rounded down (plan section 2), and it draws in hours
`t < d` only. When a rung learns that a car has gone is a reading the plan does not settle, so the
caller chooses it (`departure_seen`, required, no default); `DEPARTURE_SEEN` lists the two:

- "hour-start": the rung observes which cars are plugged in at the start of hour `t`. A car with
  `d = t` is still plugged in then (it unplugs within hour `t`), so it is seen gone only at the
  start of hour `t + 1`. Decision presence is `t <= d`. No foresight of the plug-out.
- "rounded-hour": the rung lives in the plan's hourly truth model, in which the car is absent from
  hour `d` on (its true set has `u_t <= r 1[t < d]`, and the survival-tightened bound is "given
  presence at `t`"); it sees the car gone at the start of hour `d`. Decision presence is `t < d`.
  Consistent with the hourly truth model, which already drops the car's last part hour, but the
  rung then knows up to an hour ahead that a car will unplug within the hour.

Under either reading, carrying out the hour and scoring use `t < d` (the car can draw in hour
`t`). Decision presence is used by `certain_failure`, by the choice of cars in a re-solve, by the
state handed to the re-solve and by `update_charge_bound`, and nowhere else.

Every decision input at the start of service hour `t`, and why it is observable then:

- the request `y_t`, so `zeta_1 .. zeta_t`: plan section 1, the request of an hour is known at
  its start;
- which cars are present by the chosen reading: "hour-start", seen at the plug (`t <= d`);
  "rounded-hour", the plug-out hour rounded down (`t < d`), assumed known at the hour's start;
- the energy each car has drawn in each earlier hour, and so whether each earlier hour failed:
  metered;
- an upper bound on each car's later charges: for a car that took less than a planned charge in
  an earlier hour (it was then at its true rate, full, or unplugging), no later charge can exceed
  what it took then, plus what it has discharged since; computed from the planned powers and the
  metered draws only (`update_charge_bound`);
- the forecast of each car, with its departure replaced by the caller's survival-tightened bound
  `departure_bound(car, hour)` for a car present at the start of the hour (the bound may use only
  that the car is present by the chosen reading: `d >= t` under "hour-start", `d >= t + 1` under
  "rounded-hour");
- the day's structure: the window, the recovery hours, the sold box `m`, `w` and the scale
  `beta`, all fixed before the day;
- the predecessor's plan in force at hour `t` and whether its discharges are carried out: the
  fixed plan is a function of the requests only; the one-way rung's plan, followed by the two-way
  rung, is the one-way rung's own decision from its own draws. A two-way rung that has left the
  one-way rung would need the one-way rung's draws to know that plan; the ledger takes it from the
  one-way rung's run.

The true room and the true rate are never read by a decision under either reading; they act only
through the draws of earlier hours. A departure within the current hour is not read under
"hour-start"; under "rounded-hour" it is, by assumption, known.

Plan in force. A rung follows its predecessor's plan in force (for the one-way rung, the fixed
plan) until a re-solve ends optimal, and its own plan from then on. A plan in force is "broken" at
the start of hour `t` when the rung has failed in an hour since it took that plan.

A rung re-solves at the start of service hour `t` when the plan in force is broken, or certain to
fail in hour `t` (`certain_failure`: on what the rung knows, the fleet's draw cannot come within
`FAILURE_KWH` of `y_t`, for example because a car seen gone still has planned power). So a rung
goes on copying a predecessor that has failed and re-solved, until the predecessor's new plan
fails in turn.

Every re-solve of a rung is guarded against its predecessor's plan as it was in force at the
rung's first optimal re-solve (`resolve.Guard`; the first re-solve of a rung that has not yet left
its predecessor is guarded against the predecessor's plan in force at that hour). Optimal: the
rung follows the new plan from hour `t`. Any other outcome (infeasible, timeout, error) is
recorded with the solver's message, and the predecessor's plan stays, as plan section 5 says: the
rung follows the plan its predecessor follows in hour `t`, from its own stored energy, and makes
the test again from the next hour.

T5(a) ("no rung fails before its predecessor") holds by construction: while a rung copies its
predecessor it draws exactly what the predecessor draws, so it fails in the same hours; it leaves
the predecessor only at an hour `t` by which the predecessor's plan has failed or in which it
certainly fails (decided only from what the rung knows, which bounds what the true cars can
draw). The rung's first failure is therefore never earlier than the predecessor's.

T5(b) (plan section 5: "Proof: induction over hours; two-way step in WP6"), a sketch, not a
checked proof. One-way rung against the fixed plan, when every forecast rate is at most the true
rate (`r_hat <= r`), by induction over hours on each car's stored energy:

- the rung's own plan has power at most `r_hat f <= r` and planned stored energy at most its cap,
  so the car draws its plan unless it is full or gone;
- that planned stored energy is at least the fixed plan's planned energy from the guard's hour
  (negative powers counted 0), which is at least what the car draws under the fixed plan, since a
  car never draws more than a planned charge;
- a full car holds its true room, at least any other run's energy; a car gone keeps its energy in
  both runs, and the induction carries the order to the hour it leaves;
- while the rung follows the fixed plan (before leaving it, or after a re-solve that was not
  optimal), both runs plan the same powers, and `x + min(p, r, e - x) = min(x + min(p, r), e)`
  does not decrease in `x`, so the order holds hour by hour;
- recovery `x + min(r h, e - x)` does not decrease in `x`, so the order holds at departure.

The two-way step (two-way rung against the one-way rung) needs, beyond `r_hat <= r`, a forecast
cap at most the true room (`e_cap <= e`). Otherwise a planned charge can be capped while the
planned discharge after it is carried out (the plan's T5 unit case). Hours in which the two-way
rung follows the one-way rung's plan are covered by the monotone step above (both runs plan the
same powers in that hour). The sketch stops being a proof in two places:

- the guard is against the one-way rung's plan in force at the two-way rung's first optimal
  re-solve, not against the one-way rung's later re-solved plans, which may plan more energy;
- every step holds only within the solver's tolerance.

The zeroing rule of plan section 5 ("a two-way rung zeroes planned discharges after its own first
failure") is applied (`replay.discharges_carried`), but the rung engine does not reach it: after
its first failure in hour `h` the two-way rung's plan is broken at `h + 1`, so it re-solves
one-way or follows its one-way predecessor.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fleetreserve import replay
from fleetreserve.day import Day
from fleetreserve.lp import OTHER_SECONDS
from fleetreserve.plans import Plan
from fleetreserve.policy import AffinePolicy
from fleetreserve.replay import FAILURE_KWH, Run, fails, finish, run_plan, take, true_limits
from fleetreserve.resolve import Guard, Remaining, resolve

RUNGS = ("fixed", "one-way", "two-way")
DEPARTURE_SEEN = ("hour-start", "rounded-hour")  # readings of when a departure is seen
TWO_WAY_BUDGET = 5.0  # kWh, the budget B of the WP6 two-way rung (plan sections 9 and 13)
CERTAIN_MARGIN = 1e-9  # kWh beyond FAILURE_KWH before a failure is called certain


@dataclass(frozen=True)
class Resolve:
    """One re-solve: the hour it planned from, its outcome, the solver's seconds and message."""

    hour: int
    outcome: str
    seconds: float
    late_seconds: float
    message: str = ""


@dataclass(frozen=True)
class RungRun:
    """One rung on one path. Per hour `t` (0 to `n`): `plans[t]` the plan in force, `broken[t]`
    whether it had failed before `t` since its rung took it, `discharging[t]` whether planned
    discharges were carried out. `own_from`: the first hour of the rung's own plan (None when it
    copied its predecessor throughout)."""

    rung: str
    run: Run
    plans: tuple[Plan, ...]
    broken: tuple[bool, ...]
    discharging: tuple[bool, ...]
    resolves: tuple[Resolve, ...]
    own_from: int | None


def certain_failure(
    planned, request, present, stored, charge_bound, *, discharge: bool, budget
) -> bool:
    """True when, on what a rung knows, the hour's draw cannot come within `FAILURE_KWH` of
    `request`.

    `present`: plugged in at the start of the hour. A car seen gone draws 0. A car still plugged
    in may unplug within the hour, so it may draw 0 too. A planned charge `p > 0` draws between 0
    and `min(p, c)`, `c` the
    known bound on its charges. A planned discharge draws 0 unless discharges are carried out;
    then between `max(p, -(B + x))` and 0 (its true rate is not known)."""
    p = np.asarray(planned, float)
    present = np.asarray(present, bool)
    charging = present & (p > 0)
    high = np.where(charging, np.minimum(p, charge_bound), 0.0)
    low = np.zeros_like(p)
    if discharge:
        out = present & (p < 0)
        low = np.where(out, np.maximum(p, -(budget + np.asarray(stored, float))), 0.0)
    margin = FAILURE_KWH + CERTAIN_MARGIN
    return bool(high.sum() < request - margin or low.sum() > request + margin)


def update_charge_bound(bound, planned, draws, present) -> np.ndarray:
    """The known bound on later charges after an hour: a car that took less than a planned charge
    can never take more than it took; a discharge adds to the room it may take."""
    planned = np.asarray(planned, float)
    draws = np.asarray(draws, float)
    capped = np.asarray(present, bool) & (planned > 0) & (draws < planned)
    bound = np.where(capped, np.minimum(bound, draws), bound)
    return np.where(draws < 0, bound - draws, bound)


def check_departure_seen(departure_seen: str) -> None:
    if departure_seen not in DEPARTURE_SEEN:
        raise ValueError(f"departure_seen must be one of {DEPARTURE_SEEN}, got {departure_seen!r}")


def present_for_decisions(hour: int, departure, departure_seen: str) -> np.ndarray:
    """Which cars a rung counts as present when it decides at the start of `hour` (module text):
    `hour <= d` under "hour-start", `hour < d` under "rounded-hour"."""
    check_departure_seen(departure_seen)
    departure = np.asarray(departure)
    return hour <= departure if departure_seen == "hour-start" else hour < departure


@dataclass(frozen=True)
class _Context:
    day: Day
    m: np.ndarray
    w: np.ndarray
    beta: float
    zeta: np.ndarray
    requests: np.ndarray
    departure_bound: Callable[[int, int], float]
    departure_seen: str
    budget: float
    time_limit: float
    resolver: Callable


def _fixed(context: _Context, policy: AffinePolicy) -> RungRun:
    plan = Plan.from_policy(policy, context.beta)
    run = run_plan(context.day, plan, context.zeta, context.requests)
    hours = context.day.n_service + 1
    first = run.first_failure
    broken = tuple(first is not None and first < t for t in range(hours))
    return RungRun("fixed", run, (plan,) * hours, broken, (False,) * hours, (), None)


def _state(c: _Context, t: int, present, stored, bound) -> Remaining:
    cars = len(c.day.cars)
    departure = [c.departure_bound(i, t) if present[i] else np.nan for i in range(cars)]
    return Remaining(
        t,
        c.zeta[:t].copy(),
        c.m,
        c.w,
        c.beta,
        present,
        stored.copy(),
        bound.copy(),
        np.array(departure, dtype=float),
        c.day.forecasts,
        c.day.n_recovery,
    )


def _in_force(pred: RungRun, own: Plan | None, t: int, failed_before: bool):
    """The plan a rung follows in hour `t` and whether its planned discharges are carried out."""
    if own is None:
        return pred.plans[t], pred.discharging[t]
    return own, replay.discharges_carried(own, failed_before)


def _successor(name: str, pred: RungRun, c: _Context) -> RungRun:
    day = c.day
    n = day.n_service
    departure, rate, room = true_limits(day)
    two_way = name == "two-way"
    cars = len(day.cars)
    planned = np.zeros((cars, n + 1))
    draws = np.zeros_like(planned)
    stored = np.zeros(cars)
    bound = np.full(cars, np.inf)
    own: Plan | None = None  # None while the rung follows its predecessor's plan
    guard: Guard | None = None
    taken: Plan | None = None  # the plan in force, and the hour the rung took it
    taken_at = 0
    failed_hours: list[int] = []
    failed_before = False
    own_from = None
    plans, broken, discharging, resolves = [], [], [], []
    for t in range(n + 1):
        plugged = present_for_decisions(t, departure, c.departure_seen)
        candidate, carried = _in_force(pred, own, t, failed_before)
        if candidate is not taken:
            taken, taken_at = candidate, t
        is_broken = any(taken_at <= h < t for h in failed_hours)
        if t >= 1 and (
            is_broken
            or certain_failure(
                candidate.powers(t, c.zeta[:t]),
                c.requests[t - 1],
                plugged,
                stored,
                bound,
                discharge=carried,
                budget=c.budget,
            )
        ):
            this_guard = guard or Guard(candidate, t, stored.copy(), clip=not carried)
            result = c.resolver(
                _state(c, t, plugged, stored, bound),
                this_guard,
                two_way=two_way and not failed_before,
                budget=c.budget,
                time_limit=c.time_limit,
            )
            resolves.append(
                Resolve(t, result.outcome, result.seconds, result.late_seconds, result.message)
            )
            if result.outcome == "optimal":
                own, guard = result.plan, this_guard
                own_from = t if own_from is None else own_from
            else:
                own = None  # plan section 5: "the predecessor's plan stays"
            candidate, carried = _in_force(pred, own, t, failed_before)
            taken, taken_at, is_broken = candidate, t, False
        plans.append(candidate)
        broken.append(is_broken)
        discharging.append(carried)
        planned[:, t] = candidate.powers(t, c.zeta[:t])
        draws[:, t] = take(
            planned[:, t], stored, t < departure, rate, room, discharge=carried, budget=c.budget
        )
        if t >= 1 and fails(draws[:, t].sum(), c.requests[t - 1]):
            failed_before = True
            failed_hours.append(t)
        bound = update_charge_bound(bound, planned[:, t], draws[:, t], plugged)
        stored = stored + draws[:, t]
    run = finish(day, planned, draws, c.requests)
    return RungRun(
        name,
        run,
        tuple(plans),
        tuple(broken),
        tuple(discharging),
        tuple(resolves),
        own_from,
    )


def run_rungs(
    day: Day,
    policy: AffinePolicy,
    beta: float,
    zeta,
    *,
    departure_bound: Callable[[int, int], float],
    departure_seen: str,
    last: str = "two-way",
    budget: float = TWO_WAY_BUDGET,
    time_limit: float = OTHER_SECONDS,
    resolver: Callable = resolve,
) -> dict[str, RungRun]:
    """Every rung from "fixed" up to `last` on the path `zeta` at scale `beta`.

    `policy`: the certified causal affine policy of the day's box. `departure_bound(car, hour)`:
    the survival-tightened forecast departure of a car present at the start of `hour`.
    `departure_seen`: "hour-start" or "rounded-hour" (module text), no default. `resolver`: the
    re-solve programme (`resolve.resolve`; replaceable for tests)."""
    if last not in RUNGS:
        raise ValueError(f"rung must be one of {RUNGS}, got {last!r}")
    check_departure_seen(departure_seen)
    n = day.n_service
    zeta = np.asarray(zeta, dtype=float)
    if zeta.shape != (n,) or np.any(np.abs(zeta) > 1):
        raise ValueError(f"zeta must have {n} entries in [-1, 1]")
    if last != "fixed" and day.forecasts is None:
        raise ValueError("a re-solve rung needs the day's forecasts")
    context = _Context(
        day,
        policy.m,
        policy.w,
        float(beta),
        zeta,
        policy.request(zeta, beta),
        departure_bound,
        departure_seen,
        float(budget),
        time_limit,
        resolver,
    )
    runs = {"fixed": _fixed(context, policy)}
    for previous, name in zip(RUNGS, RUNGS[1:], strict=False):
        if RUNGS.index(name) > RUNGS.index(last):
            break
        runs[name] = _successor(name, runs[previous], context)
    return runs
