"""One prepared day at one scale: the day-level losses of every controller method (plan section 10).

Pure: no files, no clock in the results, so it can run in a process pool.

`evaluate(prepared, beta, methods)` takes a `gate_day.PreparedDay`, a scale and methods of
`compare_methods` of kind "controller" (a method of another kind is refused with `ValueError`),
and returns name -> `MethodDay`. On a day that abstains (`prepared.abstains`) or at `beta <= 0`
every method has slot loss 0 and driver loss 0 and no path is run, as `gate_day.evaluate` gives
every rung there. Otherwise, on each of the day's audited
paths `zeta` (`prepared.paths`):

- the requests are `policy.request(zeta, beta)`, `m + beta w zeta` of the day's certified box
  (`prepared.certificate.policy`), as `rungs.run_rungs` makes them for the rungs;
- the fixed plan is `Plan.from_policy(policy, beta)`, and the hour-zero target is
  `controller.hour_zero_target` of it;
- the belief is `controller.forecast_belief(prepared.day)`, the forecasts the certificate of the
  prepared day was built on;
- each method's controller is built from one `MethodInput` (the same object on every path),
  which carries the rows the box was certified on (`prepared.rows`), and run on the steered
  sub-day (`prepared.day`) by `controller_replay.run_controller`, with `departure_seen` of the
  prepared day.

A method's day-level slot loss and driver loss are the largest over the paths, as
`gate_day.evaluate` takes them for the rungs; `first_failures` holds the first failing hour of
each path (None where no hour fails).

Solver counts: a controller that has a method `solver_counts` (`asa_controller.AsaController`,
`AsaEsController` and `diagonal_split.DiagonalSplitController`) is asked for it after its path is
run, and a method's `solver_counts` is the sum over the paths of those counts: per outcome the
number of solves, the number of hours planned, the number of hours planned by its fall-back and,
when the controllers count them (`DiagonalSplitController`), per kind the number of boxes. It is
None for a method whose controllers have no such method, and on a day that abstains or at
`beta <= 0`, where nothing is run.

Deviation counts: a controller that has a method `deviation_counts`
(`mef_controller.MefController`) is asked for it after its path is run, and a method's
`deviation_counts` is the sum over the paths of those counts, key by key (`sum_deviations`). It
is None as `solver_counts` is.
"""

from dataclasses import dataclass

from fleetreserve.compare_methods import MethodInput
from fleetreserve.controller import forecast_belief, hour_zero_target
from fleetreserve.controller_replay import run_controller
from fleetreserve.gate_day import PreparedDay
from fleetreserve.plans import Plan


@dataclass(frozen=True)
class MethodDay:
    slot_loss: float  # the largest over the paths
    driver_loss: float
    first_failures: tuple[int | None, ...]  # per path; empty when the day abstains
    solver_counts: dict | None = None  # summed over the paths (module text)
    deviation_counts: dict | None = None  # summed over the paths (module text)


def sum_counts(found: list[dict]) -> dict | None:
    """The sum of solver counts (module text); None for an empty list."""
    if not found:
        return None
    solves: dict[str, int] = {}
    for counts in found:
        for outcome, number in counts["solves"].items():
            solves[outcome] = solves.get(outcome, 0) + number
    summed = {
        "solves": solves,
        "hours": sum(c["hours"] for c in found),
        "fallback_hours": sum(c["fallback_hours"] for c in found),
    }
    if any("boxes" in c for c in found):
        boxes: dict[str, int] = {}
        for counts in found:
            for kind, number in counts.get("boxes", {}).items():
                boxes[kind] = boxes.get(kind, 0) + number
        summed["boxes"] = boxes
    return summed


def sum_deviations(found: list[dict]) -> dict | None:
    """The sum of deviation counts, key by key over the keys of the first (module text); None
    for an empty list."""
    if not found:
        return None
    return {key: sum(counts[key] for counts in found) for key in found[0]}


def evaluate(prepared: PreparedDay, beta: float, methods) -> dict[str, MethodDay]:
    """Every controller method's day-level losses at `beta` (module text)."""
    beta = float(beta)
    for method in methods:
        if method.kind != "controller":
            raise ValueError(f"method {method.name!r} is not a controller")
    if prepared.abstains or beta <= 0:
        return {m.name: MethodDay(0.0, 0.0, ()) for m in methods}
    policy = prepared.certificate.policy
    plan = Plan.from_policy(policy, beta)
    target = hour_zero_target(plan)
    belief = forecast_belief(prepared.day)
    inputs = MethodInput(prepared.day, policy, beta, plan, prepared.rows)
    runs = {m.name: [] for m in methods}
    counts = {m.name: [] for m in methods}
    deviations = {m.name: [] for m in methods}
    for zeta in prepared.paths:
        requests = policy.request(zeta, beta)
        for method in methods:
            controller = method.build(inputs)
            runs[method.name].append(
                run_controller(
                    prepared.day,
                    controller,
                    belief,
                    zeta,
                    requests,
                    hour_zero_target=target,
                    departure_seen=prepared.departure_seen,
                )
            )
            if hasattr(controller, "solver_counts"):
                counts[method.name].append(controller.solver_counts())
            if hasattr(controller, "deviation_counts"):
                deviations[method.name].append(controller.deviation_counts())
    return {
        name: MethodDay(
            max(r.slot_loss for r in found),
            max(r.driver_loss for r in found),
            tuple(r.first_failure for r in found),
            sum_counts(counts[name]),
            sum_deviations(deviations[name]),
        )
        for name, found in runs.items()
    }
