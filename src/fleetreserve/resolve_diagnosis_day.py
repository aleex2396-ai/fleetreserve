"""The one-way rung's re-solves of one prepared day at one scale, each measured (plan section 5).

A measurement: nothing here changes what a rung does, and no result goes back into one.

`collect(prepared, beta)`: the day's one-way rung run on each audited path as the gate runs it.
`gate_day.replay` calls `day_replay.replay_day`, which calls `rungs.run_rungs` per path with the
certificate's policy, the scale, the day's departure-bound table, its reading of when a departure
is seen, the matrix's budget and the 300-second limit for other programmes. `collect` makes the
same call with the last rung "one-way" (the one-way rung does not depend on the rung after it)
and with a re-solver that calls `resolve.resolve` unchanged and keeps what it was given and what
it answered. An abstaining day, and a scale of 0 or less, run nothing (as `gate_day.evaluate`).

`rows(prepared, beta)`: one row (a dictionary of text and numbers, `COLUMNS`) per re-solve
attempted, in the order path, hour:

- `path_index`, `path`: the audited path; `hour`: the service hour the re-solve plans from;
- `outcome`: the re-solve's own outcome (optimal, infeasible, timeout, error, unbounded);
- state: `cars` (the steered cars of the day), `present` (cars the rung counts as plugged in at
  the start of the hour) and `seen_gone` (the others); `guard_hour` (the hour of the predecessor's
  plan the re-solve is guarded against); `fixed_failed_before`, `one_way_failed_before` (whether
  the fixed plan, and the one-way rung itself, failed in an earlier hour of this path);
  `re_solves_before`, `optimal_before` (earlier re-solves of the rung on this path, and how many
  of them ended optimal).

For an infeasible re-solve only (empty otherwise), from `resolve_diagnosis_lp.diagnose`:

- `recheck`: the answer of the same rows with objective 0;
- `single`: the groups whose removal alone makes the rows feasible, joined by "|"; `pairs`: when
  no single group does and the pairs were solved, the pairs ("a+b") that do; `cause`:
  `resolve_diagnosis_lp.cause`;
- `departure_relaxed`: the answer with every present car taken to stay through each hour;
- `box_scale`, `box_scale_outcome`: the largest scale `s` in [0, 1] of the remaining box
  (`resolve_diagnosis_lp`, step 5: the half-widths of the hours after the current one, and the
  predecessor's gains on those requests, multiplied by `s`), empty with the outcome "infeasible"
  when not even `s = 0` is feasible; `need_slack_kwh`, `need_slack_outcome`; `request_miss_kwh`,
  `request_miss_outcome`: the distances from feasible (steps 5 to 7 there), each number empty
  unless its programme ended optimal;
- `relaxed_timeouts`, `relaxed_errors`: how many of the relaxed programmes of this re-solve ended
  in a timeout, and in an error (after the retry);
- `solver_first_errors`, `solver_retried_answered`: how many of them had a first attempt that
  ended in a solver error, and of those how many the retry with the simplex method then answered
  (optimal or infeasible; `resolve_diagnosis_lp`, solver errors).

Hindsight, for an infeasible re-solve only. The true departures are read here for measurement
only; no rung sees them. `early_gone_cars`: the steered cars that have truly left by the start
of the hour (true departure hour `d <= hour`, so they draw nothing from that hour on) and whose
true departure is before the forecast departure (`d < d_hat`), where `d_hat` is the forecast
departure of the day's forecasts, made at the decision time tau: not the (survival-tightened)
bound the re-solve used and not the declared time. It counts every steered car of the day,
including the cars the rung has already seen gone, and it is cumulative along a path: a car that
left early is counted again at each later infeasible re-solve of that path.
`early_gone_expected_kwh`: the energy the fixed plan (the one-way rung's predecessor, whose plan
stays after an infeasible re-solve) plans for those cars in the service hours from `hour` on,
along this path's requests, a negative planned power counting 0 as the plan is run one-way.
"""

from dataclasses import dataclass

import numpy as np

from fleetreserve import resolve_diagnosis_lp as relaxed
from fleetreserve.gate_day import PreparedDay
from fleetreserve.lp import OTHER_SECONDS
from fleetreserve.replay import true_limits
from fleetreserve.resolve import Guard, Remaining, resolve
from fleetreserve.rungs import RUNGS, run_rungs

RUNG = RUNGS[1]  # the one-way re-solve
COLUMNS = (
    "path_index",
    "path",
    "hour",
    "outcome",
    "cars",
    "present",
    "seen_gone",
    "guard_hour",
    "fixed_failed_before",
    "one_way_failed_before",
    "re_solves_before",
    "optimal_before",
    "recheck",
    "single",
    "pairs",
    "cause",
    "departure_relaxed",
    "box_scale",
    "box_scale_outcome",
    "need_slack_kwh",
    "need_slack_outcome",
    "request_miss_kwh",
    "request_miss_outcome",
    "relaxed_timeouts",
    "relaxed_errors",
    "solver_first_errors",
    "solver_retried_answered",
    "early_gone_cars",
    "early_gone_expected_kwh",
)
INFEASIBLE_ONLY = COLUMNS[COLUMNS.index("recheck") :]
# Outcome words of a relaxed programme that are an answer or a timeout; any other is an error.
ANSWERED = ("optimal", relaxed.FEASIBLE, relaxed.INFEASIBLE, relaxed.TIMEOUT)


@dataclass(frozen=True)
class Attempt:
    """One re-solve as the rung made it, and its outcome."""

    state: Remaining
    guard: Guard
    two_way: bool
    budget: float
    outcome: str


@dataclass(frozen=True)
class PathAttempts:
    """The re-solves of the one-way rung on one path, in order, with the rung runs they came
    from (`rungs.run_rungs`: "fixed" and "one-way")."""

    path_index: int
    path: np.ndarray
    attempts: tuple[Attempt, ...]
    runs: dict


def collect(prepared: PreparedDay, beta: float, *, time_limit: float = OTHER_SECONDS):
    """The one-way rung's re-solves on every audited path of the day (module text): one
    `PathAttempts` per path, none for an abstaining day or a scale of 0 or less."""
    if prepared.abstains or beta <= 0:
        return []
    table = prepared.departure_bounds
    found = []
    for k, path in enumerate(prepared.paths):
        log: list[Attempt] = []

        def recording(state, guard, *, two_way, budget, time_limit, log=log):
            result = resolve(state, guard, two_way=two_way, budget=budget, time_limit=time_limit)
            log.append(Attempt(state, guard, two_way, budget, result.outcome))
            return result

        runs = run_rungs(
            prepared.day,
            prepared.certificate.policy,
            float(beta),
            path,
            departure_bound=lambda car, hour: float(table[car, hour]),
            departure_seen=prepared.departure_seen,
            last=RUNG,
            budget=prepared.budget,
            time_limit=time_limit,
            resolver=recording,
        )
        found.append(PathAttempts(k, np.asarray(path, dtype=float), tuple(log), runs))
    return found


def hindsight(prepared: PreparedDay, runs: dict, hour: int) -> tuple[int, float]:
    """(`early_gone_cars`, `early_gone_expected_kwh`) of the module text, from the true
    departures."""
    day = prepared.day
    departure, _, _ = true_limits(day)
    forecast = np.array([f.departure for f in day.forecasts], dtype=float)
    early = (departure <= hour) & (departure < forecast)
    planned = np.maximum(runs[RUNGS[0]].run.planned[:, hour:], 0.0)
    return int(early.sum()), float(planned[early].sum())


def _number(value) -> str:
    return "" if value is None else repr(float(value))


def _measured(found: relaxed.Diagnosis) -> dict:
    single = [g for g, answer in found.single.items() if answer == relaxed.FEASIBLE]
    pairs = [] if found.pair is None else found.pair.items()
    outcomes = found.outcomes()
    return {
        "recheck": found.recheck,
        "single": "|".join(single),
        "pairs": "|".join(name for name, answer in pairs if answer == relaxed.FEASIBLE),
        "cause": relaxed.cause(found),
        "departure_relaxed": found.departure_relaxed,
        "box_scale": _number(found.box_scale),
        "box_scale_outcome": found.box_scale_outcome,
        "need_slack_kwh": _number(found.need_slack),
        "need_slack_outcome": found.need_slack_outcome,
        "request_miss_kwh": _number(found.request_miss),
        "request_miss_outcome": found.request_miss_outcome,
        "relaxed_timeouts": outcomes.count(relaxed.TIMEOUT),
        "relaxed_errors": sum(o not in ANSWERED for o in outcomes),
        "solver_first_errors": found.first_errors,
        "solver_retried_answered": found.retried_answered,
    }


def rows(
    prepared: PreparedDay,
    beta: float,
    *,
    time_limit: float = OTHER_SECONDS,
    relaxed_time_limit: float = OTHER_SECONDS,
    diagnose=relaxed.diagnose,
) -> list[dict]:
    """One row per re-solve of the one-way rung on the day at scale `beta` (module text).
    `time_limit`: the re-solves' own limit; `relaxed_time_limit`: that of each relaxed programme."""
    out = []
    cars = len(prepared.day.cars) if prepared.day is not None else 0
    for found in collect(prepared, beta, time_limit=time_limit):
        fixed_first = found.runs[RUNGS[0]].run.first_failure
        failed = found.runs[RUNG].run.failed
        path = " ".join(f"{z:+.0f}" for z in found.path)
        for k, attempt in enumerate(found.attempts):
            hour = attempt.state.hour
            present = int(np.asarray(attempt.state.present, bool).sum())
            before = [a.outcome for a in found.attempts[:k]]
            row = {
                "path_index": found.path_index,
                "path": path,
                "hour": hour,
                "outcome": attempt.outcome,
                "cars": cars,
                "present": present,
                "seen_gone": cars - present,
                "guard_hour": attempt.guard.hour,
                "fixed_failed_before": int(fixed_first is not None and fixed_first < hour),
                "one_way_failed_before": int(bool(failed[: hour - 1].any())),
                "re_solves_before": len(before),
                "optimal_before": before.count("optimal"),
                **dict.fromkeys(INFEASIBLE_ONLY, ""),
            }
            if attempt.outcome == "infeasible":
                row.update(
                    _measured(
                        diagnose(
                            attempt.state,
                            attempt.guard,
                            two_way=attempt.two_way,
                            budget=attempt.budget,
                            time_limit=relaxed_time_limit,
                        )
                    )
                )
                gone, expected = hindsight(prepared, found.runs, hour)
                row["early_gone_cars"] = gone
                row["early_gone_expected_kwh"] = repr(expected)
            out.append(row)
    return out


def task(job) -> list[dict]:
    """`rows` of one `(prepared day, scale)`, for a process pool."""
    prepared, beta = job
    return rows(prepared, beta)
