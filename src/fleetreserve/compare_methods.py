"""The comparison methods of plan section 10 that `compare_run` measures, by name.

A method is a `Method`: a name, a kind and, for a controller, a builder. Kinds:

- "controller": `build(inputs)` returns a controller of `controller.py` (a callable from an
  `HourView` to one planned power per car) for one day, scale and path; `inputs` is a
  `MethodInput`. `compare_day` builds it once per audited path and runs it with
  `controller_replay.run_controller`.
- "re-solve": the project's own one-way re-solve rung, read from `gate_day.evaluate` by
  `dial_choice.measure`; it has no builder.
- "sizing": the re-solve rung's losses with the scale of each split chosen by a rule of its own
  in place of the slot bound of `crc` (the rules of `chance_sizing` keep its driver bound in tier
  A, reading S5): `size(service_hours)` returns that rule for one unit under one variant
  (`chance_sizing.ChanceRule`, as `dial_splits.UnitLosses.rule` takes it), given the number of
  service hours of each of the unit's days. It has no builder. `also-x` and `cvar` are the two
  sizing rules of Lunde et al. (2025), readings S1 to S11 in `chance_sizing`.

`METHODS` lists the methods in the order the summary gives them and the results' list `methods`
names them when `compare_run` is run without `--methods`; with `--methods` both follow the order
given there (`select`). A later method is
added by appending one entry to `METHODS` (and its builder to this file or another module); the
command reads the list, so a method whose controller reports only its losses needs no other
change. A controller that reports counts of its own needs them carried through `compare_day`,
written by `compare_run` and summed in a line of `compare_summary`, as the solver counts of
`asa`, `asa-es` and `diagonal-split` and the deviation counts of `max-entropy-feedback` are. A
builder must be a function defined at the top
level of a module, so that the method can be sent to a worker process.

`select(names, registry)` checks the list (`METHODS` when `registry` is None) and returns the
methods asked for: every method of the list when `names` is None, else the named ones in the
order given. A name listed twice in the registry with a different kind, builder or rule, a kind
not in `KINDS`, a controller without a builder, a re-solve or a sizing method with one, a sizing
method without a rule, a controller or a re-solve with one, a name asked for that is not in the
list and a name asked for twice are refused with `MethodError`. The same entry listed twice is
one method.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fleetreserve.asa_controller import AsaController, AsaEsController
from fleetreserve.carsets import FleetRows
from fleetreserve.chance_sizing import also_x_rule, cvar_rule
from fleetreserve.controller import Controller
from fleetreserve.day import Day
from fleetreserve.diagonal_split import DiagonalSplitController, split_box
from fleetreserve.lp import BOX_SECONDS, OTHER_SECONDS
from fleetreserve.mef_controller import MefController
from fleetreserve.plan_controller import PlanController
from fleetreserve.plans import Plan
from fleetreserve.policy import AffinePolicy
from fleetreserve.priority_rules import (
    earliest_deadline_first,
    least_laxity_first,
    smoothed_least_laxity,
)

KINDS = ("controller", "re-solve", "sizing")
RE_SOLVE = "re-solve"


class MethodError(ValueError):
    """A registry that does not fit the module text, or a method name not in it."""


@dataclass(frozen=True)
class MethodInput:
    """What a builder gets for one day at one scale: the steered sub-day with its forecasts
    (`gate_day.PreparedDay.day`), the certified causal affine policy of the day's box, the scale,
    the fixed plan at that scale (`plans.Plan.from_policy(policy, beta)`) and the rows the box was
    certified on (`gate_day.PreparedDay.rows`; None when the caller gives none, and a method that
    needs them then refuses with `MethodError`)."""

    day: Day
    policy: AffinePolicy
    beta: float
    plan: Plan
    rows: FleetRows | None = None


@dataclass(frozen=True)
class Method:
    name: str
    kind: str
    build: Callable[[MethodInput], Controller] | None = None
    size: Callable | None = None  # a sizing method's rule for one unit and variant (module text)


def fixed_plan(inputs: MethodInput) -> Controller:
    """The fixed plan without re-solving (plan section 10), run as a controller."""
    return PlanController(inputs.plan)


def edf(inputs: MethodInput) -> Controller:
    return earliest_deadline_first


def llf(inputs: MethodInput) -> Controller:
    return least_laxity_first


def smoothed_llf(inputs: MethodInput) -> Controller:
    return smoothed_least_laxity


def asa(inputs: MethodInput) -> Controller:
    """This project's linear-programme form of ASA (`asa_controller.AsaController`), built by
    `for_box` from the day's box (of which it reads the centre `m`) and the fixed plan at the
    scale, with the 300-second limit of plan section 13 for other programmes on every solve."""
    return AsaController.for_box(
        inputs.policy,
        inputs.beta,
        inputs.plan,
        n_cars=len(inputs.day.cars),
        n_recovery=inputs.day.n_recovery,
        time_limit=OTHER_SECONDS,
    )


def asa_es(inputs: MethodInput) -> Controller:
    """`asa` with the equal-sharing term of `asa_equal_share` as a fifth goal
    (`asa_controller.AsaEsController`), built as `asa` is, with the same limit on every solve."""
    return AsaEsController.for_box(
        inputs.policy,
        inputs.beta,
        inputs.plan,
        n_cars=len(inputs.day.cars),
        n_recovery=inputs.day.n_recovery,
        time_limit=OTHER_SECONDS,
    )


_LAST_SPLIT: list = []  # [(inputs, split)] of the last `diagonal_split` call that solved


def diagonal_split(inputs: MethodInput) -> Controller:
    """The diagonal split of the day's box at the scale, run as a fixed plan
    (`diagonal_split.split_box` on `inputs.rows` with the centre `m` and the half-widths
    `beta w` of the certified policy, the 30-second limit of plan section 13 for a box; readings
    D1 to D4 there), as a `diagonal_split.DiagonalSplitController`.

    Without rows (`inputs.rows` None) it refuses with `MethodError`. A call with the very same
    `MethodInput` object as the call that last solved (as `compare_day` makes one per day and
    scale and builds one controller per audited path from it) reuses that call's split, and its
    controller counts no solve and no box, so a day and scale is solved and counted once."""
    if inputs.rows is None:
        raise MethodError(
            "method 'diagonal-split' needs the rows the box was certified on "
            "(MethodInput.rows), and none were given"
        )
    if _LAST_SPLIT and _LAST_SPLIT[0][0] is inputs:
        return DiagonalSplitController(_LAST_SPLIT[0][1], counted=False)
    policy = inputs.policy
    split = split_box(inputs.rows, policy.m, inputs.beta * policy.w, time_limit=BOX_SECONDS)
    _LAST_SPLIT[:] = [(inputs, split)]
    return DiagonalSplitController(split, counted=True)


def max_entropy_feedback(inputs: MethodInput) -> Controller:
    """Least-laxity-first guarded by the believed future (`mef_controller.MefController`, the
    reduced form of the maximum-entropy feedback of Li et al. that its module text describes,
    readings B-R1 to B-R9 there), with the day's box at the scale: lowest request
    `m - beta w` and highest `m + beta w` in each service hour, of the certified policy's centre
    `m` and half-widths `w`. It needs no rows: its believed sets come from the view's belief."""
    policy = inputs.policy
    low = policy.m - inputs.beta * policy.w
    return MefController(low, policy.m + inputs.beta * policy.w)


METHODS: tuple[Method, ...] = (
    Method(RE_SOLVE, "re-solve"),
    Method("fixed-plan", "controller", fixed_plan),
    Method("edf", "controller", edf),
    Method("llf", "controller", llf),
    Method("smoothed-llf", "controller", smoothed_llf),
    Method("asa", "controller", asa),
    Method("asa-es", "controller", asa_es),
    Method("diagonal-split", "controller", diagonal_split),
    Method("max-entropy-feedback", "controller", max_entropy_feedback),
    Method("also-x", "sizing", size=also_x_rule),
    Method("cvar", "sizing", size=cvar_rule),
)


def check(registry: Sequence[Method]) -> dict[str, Method]:
    """Name -> method of `registry`, in its order, after the checks of the module text."""
    found: dict[str, Method] = {}
    for method in registry:
        if method.kind not in KINDS:
            raise MethodError(f"method {method.name!r}: kind must be one of {KINDS}")
        if method.kind == "controller" and method.build is None:
            raise MethodError(f"method {method.name!r}: a controller needs a builder")
        if method.kind == "re-solve" and method.build is not None:
            raise MethodError(f"method {method.name!r}: the re-solve rung has no builder")
        if method.kind == "sizing" and method.build is not None:
            raise MethodError(f"method {method.name!r}: a sizing method has no builder")
        if method.kind == "sizing" and method.size is None:
            raise MethodError(f"method {method.name!r}: a sizing method needs a rule")
        if method.kind != "sizing" and method.size is not None:
            raise MethodError(f"method {method.name!r}: only a sizing method has a rule")
        earlier = found.get(method.name)
        if earlier is not None and earlier != method:
            raise MethodError(
                f"method {method.name!r} is registered twice with different controllers"
            )
        found.setdefault(method.name, method)
    return found


def select(names: Sequence[str] | None, registry: Sequence[Method] | None = None) -> tuple:
    """The methods asked for (module text); `registry` None reads `METHODS` at the call."""
    known = check(METHODS if registry is None else registry)
    if names is None:
        return tuple(known.values())
    if len(set(names)) != len(names):
        raise MethodError("a method is named twice")
    missing = [n for n in names if n not in known]
    if missing:
        raise MethodError(f"unknown method {missing[0]!r}; known: {', '.join(known)}")
    return tuple(known[n] for n in names)
