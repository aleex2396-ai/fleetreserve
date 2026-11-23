"""The corner search of one prepared day (plan section 1 "Corner tree", section 11 O7, 13 WP11).

Pure: no files, no clock in the results, so it can run in a process pool.

Plan section 1: "Corner tree: the 2^|S| paths at lo_t or hi_t each hour". The corners are
`products.corner_paths(n)`: corner `k` has `zeta_j = +1` where bit `j` of `k` is set, else -1
(`corner_index` gives `k` of a path of -1 and +1 entries).

`search_day(prepared, method, beta)`: one method of `compare_methods` at scale `beta` on every
corner of a prepared day (`gate_day.PreparedDay`) that offers. The requests are
`policy.request(zeta, beta)` of the day's certified box, as the comparison makes them
(`compare_day`, `rungs.run_rungs`). By the method's kind:

- "controller": built once per path from one `compare_methods.MethodInput` (the same object on
  every path, holding the rows the box was certified on, `prepared.rows`, as `compare_day`
  builds it, so a method that plans on those rows, `diagonal-split`, can be searched, and solves
  its one programme once per day and scale) and run by
  `controller_replay.run_controller` on the steered sub-day, with the belief
  `controller.forecast_belief(prepared.day)`, the hour-zero target
  `controller.hour_zero_target(Plan.from_policy(policy, beta))` and the day's `departure_seen`,
  as `compare_day.evaluate` runs it on the audited paths;
- "re-solve": the one-way re-solve rung, `day_replay.replay_day` with the rung "one-way" (which
  also runs the fixed plan it starts from), the day's departure-bound table, budget and
  `departure_seen`, and the time limit `lp.OTHER_SECONDS`, as `gate_day.replay` calls it (which
  runs every rung up to the two-way one; the one-way rung does not depend on the two-way one).
  A path on which a re-solve timed out is marked `timed_out`.
- "sizing" (`also-x`, `cvar`): run exactly as "re-solve" at the scale given; a sizing method
  differs from the re-solve only in the scale its rule chooses (`chance_sizing`; reading W5 of
  `corner_run`), so at an equal scale its day equals the re-solve's but for the method's name.

Per path (`CornerPath`): the first failing service hour (None when every request is met), the
slot loss and the driver loss of `replay.finish` (the losses `compare_day` and `gate_day` read),
and the protected energy undelivered at departure in kWh (`sum_i max(0, (1 - rho) e_i - final_i)`,
the numerator of the driver loss). A path breaks (reading K1 of `corner_run`) when an hour fails
by plan section 1's rule (a delivery missing its request by more than `replay.FAILURE_KWH`, which
is what makes the slot loss above 0) or when its undelivered protected energy is above the same
`FAILURE_KWH`. Plan section 1 states no tolerance for the driver loss; `FAILURE_KWH` is used so
that a car a few rounding steps short of its protected energy is not a break. The rule reads the
amount only, summed over the cars, not its cause: undelivered protected energy above 0 and at most
`FAILURE_KWH` is not a break, whether rounding left it or the cars are short by that little. A
path that does not break but has a driver loss above 0 is "short below the tolerance"; those
paths are counted, not hidden.

The day (`day_status`): "breaks" when a path that did not time out breaks; otherwise "timed out"
when a path timed out (a timeout is a missing result, never "holds"); otherwise "holds". At
`beta <= 0` the method offers nothing: "no offer", and no path is run. A prepared day that
abstains is refused (`ValueError`): the command searches only days that offer.

`CornerDay` also holds, over the paths that did not time out: the numbers of breaking paths, of
paths with a slot loss above 0, of paths whose undelivered protected energy is above
`FAILURE_KWH` and of paths short below the tolerance; the worst slot loss and the worst driver
loss (each its own maximum, over every path without a timeout); one worst path, among the
breaking paths the first in corner order with the largest slot loss and, among those, the largest
driver loss (None when no path breaks), and its first failing hour. Beside them the audited
reading: the day's audited paths (`prepared.paths`, all corners) read from the corner runs at
their corners, rows counted as rows (repeats included, as `audit_paths` keeps them), with the
same rule for breaking and for timeouts; it is what the audited set alone says for the same day
and scale. And the re-solves by outcome, over all paths, and, for a controller that solves
programmes (it has a method `solver_counts`, as `asa_controller.AsaController`, `AsaEsController`
and `diagonal_split.DiagonalSplitController` have), its solver counts summed over all paths by
`compare_day.sum_counts`, as `compare_day.evaluate` sums them over the audited paths (None for a
controller without them, for the re-solve and for a sizing method); for `diagonal-split` the one
solve and the one box of the day and scale are counted once, on the first path, and the hours on
every path. In the same way, for a controller that counts its deviations (it has a method
`deviation_counts`, as `mef_controller.MefController` has: `max-entropy-feedback`, plan section 11
O7), those counts summed over all paths, key by key, by `compare_day.sum_deviations` (None for
every other method and at `beta <= 0`).

Corner sufficiency is shown for the fixed plan only (reading P1 below). For the other methods the
corners are a search, not a proof: a priority rule, the re-solve or `asa` can do worse on a path
inside the box than on every corner. One example, on a seeded made-up day of the tests (4 service
hours, 5 cars): the least-laxity rule at scale 0.6 has a driver loss of 0.1965 on the path
(-1, -1, 0, -1) against 0.1466 at its worst corner. So their breaking figures are lower bounds of
what the box allows, not worst cases.

`fixed_plan_corners(prepared, beta)`: the certified fixed plan (`Plan.from_policy`) run by
`replay.run_plan` (one-way) on every corner.

`closed_form_check(prepared, betas)`: for one day, at each scale, the comparison that the pass
mark of plan section 13 WP11 ("closed-form worst case reproduced") reads from a run on
development units (`corner_run`, reading H5). On the sealed route `corner_run` makes the same
comparison on held-out test days at the fixed plan's calibrated scale alone, as a property check
that is not that pass mark (its readings H5 and H6). Reading P1: each row of the
fixed plan is linear in the request path, so the largest violation of a row over the box is
attained at a corner, and the day's worst case over the box is its worst case over the corners.
The closed form,
on the steered cars' true rows (`carsets.true_rows`), is the first failing hour
`fixed_plan_loss.first_failing_hour` of the hour thresholds `fixed_plan_loss.hour_thresholds`
(each car's decision-hour draw in place of its planned hour-0 power), and the slot loss of
`gate_calibrate.slot_steps` at the scale (`crc.step_loss`), the step form the calibration uses.
The corner worst case is the earliest first failure and the largest slot loss over
`fixed_plan_corners`. The two agree when the hours are equal and the slot losses differ by at most
`SLOT_TOLERANCE` (`agreed_strict`). When they do not, the reason is that of
`fixed_plan_loss.compare_with_replay` for the same day and scale ("within tolerance", "shortfall
and excess cancel", "replay earlier" or "unexplained"), or "slot losses differ" when that function
gives none (equal hours, slot losses apart). Reading K3 of `corner_run`: a pair is `agreed` when
it agrees strictly or its reason is "within tolerance" (at the closed form's failing hour every
corner's draws miss the plan by at most `FAILURE_KWH` in total, so replay meets the request there
by plan section 1's failure rule while the closed form names a failure: for example by rounding,
an hour threshold of 0.9999999999999998 at scale 1 where the certified box meets a car's room
exactly); every other reason stays a disagreement. No tolerance is widened or added for it.

The driver loss has a bound, not a worst case, in closed form: the every-scale bound,
`gate_calibrate.every_scale_loss` of `every_scale_terms` (`fixed_plan_loss`, reading R3, capped at
1), the driver bound wherever one is used by the owner's decision of 16 November 2026
(`gate_calibrate`, module text); the check gives the corner maximum beside it and whether it is at
most the bound plus `DRIVER_TOLERANCE`. That bound counts each car's decision-hour draw and each
service hour at no more than the car can draw in it, so it holds at every scale, with the errors
of V3 and V7 of `fixed_plan_loss`'s module text. Beside it, for comparison only, the earlier bound
(`earlier_driver_bound`: `gate_calibrate.ramp_loss` of `driver_terms`, the summed per-car worst
cases, reading R2), which assumes the plan is delivered in every service hour and so is not a
bound at a scale where an hour fails. For every pair the check also reads the corners of the
largest driver loss, those within `DRIVER_TOLERANCE` of it (`driver_flags`; reading K4 of
`corner_run`): `driver_limited` when on every one of them some
car drew less than its planned charge in some hour (`draws < planned` where `planned > 0`: for
lack of rate, room or presence), so no corner of that loss delivered the plan. The flag counts
every hour, the decision hour included, and is kept so: on a pair above the bound every corner of
the largest loss is above it, and a corner on which no car draws less than its plan in a service
hour leaves each car at least the least energy the bound counts for it on any path, a loss at most
the bound, so each such corner has a car short of its plan in a service hour and the counts over
those pairs are the same with the decision hour left out; on a pair at or below the bound a
decision-hour draw alone can set the flag (the hand day `CAP` of the tests);
`driver_no_slot_failure` when some one of them has no failing hour, so that loss is reached
without a failing hour. Each flag is read so that it does not depend on the
order of the corners and leans to the case that questions the bound or the slot loss; the
results count both over the pairs above the bound. Several corners often lie within that
tolerance of the largest loss without being equal to it (rounding), so a reading of the first
corner of the exact largest would pick among them by rounding.

Untested here: real days; whether a path inside the box can give the fixed plan a larger driver
loss than every corner (P1 covers the rows of the slot loss, not the driver loss).
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from fleetreserve import crc
from fleetreserve.carsets import true_rows
from fleetreserve.compare_day import sum_counts, sum_deviations
from fleetreserve.compare_methods import MethodInput
from fleetreserve.controller import forecast_belief, hour_zero_target
from fleetreserve.controller_replay import run_controller
from fleetreserve.day_replay import replay_day
from fleetreserve.fixed_plan_loss import compare_with_replay, first_failing_hour, hour_thresholds
from fleetreserve.gate_calibrate import every_scale_loss, every_scale_terms, ramp_loss, slot_steps
from fleetreserve.gate_day import PreparedDay
from fleetreserve.lp import OTHER_SECONDS
from fleetreserve.plans import Plan
from fleetreserve.products import corner_paths
from fleetreserve.replay import FAILURE_KWH, protected, run_plan
from fleetreserve.rungs import RUNGS

SLOT_TOLERANCE = 1e-9  # largest slot-loss difference that still agrees strictly
WITHIN_TOLERANCE = "within tolerance"  # the one reason of compare_with_replay that agrees (K3)
DRIVER_TOLERANCE = 1e-9  # a corner driver loss above the bound by more than this exceeds it
BREAKS, HOLDS, TIMED_OUT, NO_OFFER = "breaks", "holds", "timed out", "no offer"
STATUSES = (BREAKS, HOLDS, TIMED_OUT, NO_OFFER)
ONE_WAY = RUNGS[1]


def corner_index(zeta) -> int:
    """`k` of the corner `zeta` (entries -1 or +1): bit `j` is set where `zeta_j = +1`."""
    z = np.asarray(zeta, dtype=float).reshape(-1)
    if not np.all(np.isin(z, (-1.0, 1.0))):
        raise ValueError("a corner has entries -1 or +1 only")
    return int(sum(1 << j for j in np.flatnonzero(z > 0)))


@dataclass(frozen=True)
class CornerPath:
    first_failure: int | None
    slot_loss: float
    driver_loss: float
    timed_out: bool  # a re-solve on this path timed out: the path has no result
    shortfall_kwh: float = 0.0  # protected energy undelivered at departure

    @property
    def driver_short(self) -> bool:
        """Undelivered protected energy above `FAILURE_KWH` (module text)."""
        return self.shortfall_kwh > FAILURE_KWH

    @property
    def breaks(self) -> bool:
        return self.slot_loss > 0 or self.driver_short

    @property
    def short_below_tolerance(self) -> bool:
        """A driver loss above 0 on a path that does not break."""
        return not self.breaks and self.driver_loss > 0


def shortfall(day, final) -> float:
    """Protected energy undelivered at departure, kWh (the driver loss's numerator)."""
    return float(np.maximum(protected(day) - np.asarray(final, dtype=float), 0.0).sum())


def day_status(paths) -> str:
    """The status of a day's paths: "breaks", "timed out" or "holds" (module text)."""
    if any(p.breaks and not p.timed_out for p in paths):
        return BREAKS
    if any(p.timed_out for p in paths):
        return TIMED_OUT
    return HOLDS


@dataclass(frozen=True)
class Reading:
    """What a set of path results says (over the paths that did not time out)."""

    status: str
    breaking: int
    slot_breaking: int
    driver_short: int
    short_below_tolerance: int
    timed_out: int
    worst_slot: float | None
    worst_driver: float | None
    worst: int | None  # index into the paths read, None when nothing breaks


def read(paths) -> Reading:
    kept = [(k, p) for k, p in enumerate(paths) if not p.timed_out]
    worst = None
    for k, p in kept:
        if p.breaks and (
            worst is None
            or (p.slot_loss, p.driver_loss) > (paths[worst].slot_loss, paths[worst].driver_loss)
        ):
            worst = k
    return Reading(
        day_status(paths),
        sum(p.breaks for _, p in kept),
        sum(p.slot_loss > 0 for _, p in kept),
        sum(p.driver_short for _, p in kept),
        sum(p.short_below_tolerance for _, p in kept),
        len(paths) - len(kept),
        max((p.slot_loss for _, p in kept), default=None),
        max((p.driver_loss for _, p in kept), default=None),
        worst,
    )


@dataclass(frozen=True)
class CornerDay:
    """One method at one scale on every corner of one day (module text)."""

    method: str
    beta: float
    status: str
    n_paths: int
    breaking_paths: int
    slot_breaking_paths: int
    driver_short_paths: int
    short_below_tolerance_paths: int
    timed_out_paths: int
    worst_slot_loss: float | None
    worst_driver_loss: float | None
    worst_path: tuple[int, ...] | None
    worst_first_failure: int | None
    audited_status: str | None
    audited_breaking_rows: int
    audited_worst_slot_loss: float | None
    audited_worst_driver_loss: float | None
    resolves: dict
    paths: tuple[CornerPath, ...]
    solver_counts: dict | None = None  # a controller that solves programmes (module text)
    deviation_counts: dict | None = None  # a controller that counts deviations (module text)

    @property
    def audited_breaks(self) -> bool | None:
        """True when the audited rows break, False when they hold, None when they have no result
        (no offer, or timed out)."""
        if self.audited_status in (BREAKS, HOLDS):
            return self.audited_status == BREAKS
        return None


def _controller_paths(prepared: PreparedDay, method, beta: float, paths):
    policy = prepared.certificate.policy
    plan = Plan.from_policy(policy, beta)
    target = hour_zero_target(plan)
    belief = forecast_belief(prepared.day)
    inputs = MethodInput(prepared.day, policy, beta, plan, prepared.rows)  # one for every path
    out, counts, deviations = [], [], []
    for zeta in paths:
        controller = method.build(inputs)
        run = run_controller(
            prepared.day,
            controller,
            belief,
            zeta,
            policy.request(zeta, beta),
            hour_zero_target=target,
            departure_seen=prepared.departure_seen,
        )
        if hasattr(controller, "solver_counts"):
            counts.append(controller.solver_counts())
        if hasattr(controller, "deviation_counts"):
            deviations.append(controller.deviation_counts())
        lost = shortfall(prepared.day, run.final)
        out.append(CornerPath(run.first_failure, run.slot_loss, run.driver_loss, False, lost))
    return out, Counter(), sum_counts(counts), sum_deviations(deviations)


def _re_solve_paths(prepared: PreparedDay, beta: float, paths):
    table = prepared.departure_bounds
    found = replay_day(
        prepared.day,
        prepared.certificate,
        ONE_WAY,
        beta,
        paths,
        departure_bound=lambda car, hour: float(table[car, hour]),
        budget=prepared.budget,
        time_limit=OTHER_SECONDS,
        departure_seen=prepared.departure_seen,
    )
    out, outcomes = [], Counter()
    for p in found.by_rung[ONE_WAY]:
        counts = p.outcomes()
        outcomes.update(counts)
        lost = 0.0 if p.final is None else shortfall(prepared.day, p.final)
        timed = counts["timeout"] > 0
        out.append(CornerPath(p.first_failure, p.slot_loss, p.driver_loss, timed, lost))
    return out, outcomes, None, None


def path_results(prepared: PreparedDay, method, beta: float, paths):
    """Per path a `CornerPath` of `method` at `beta`, the re-solves by outcome, the solver counts
    (None unless the method's controllers solve programmes) and the deviation counts (None unless
    they count deviations)."""
    if method.kind == "controller":
        return _controller_paths(prepared, method, beta, paths)
    if method.kind in ("re-solve", "sizing"):
        return _re_solve_paths(prepared, beta, paths)
    raise ValueError(f"method {method.name!r}: unknown kind {method.kind!r}")


def run_paths(prepared: PreparedDay, method, beta: float, paths):
    """`path_results` without the deviation counts."""
    return path_results(prepared, method, beta, paths)[:3]


def no_offer(name: str, beta: float) -> CornerDay:
    """The `CornerDay` of a method that offers nothing (`beta <= 0`): no path is run."""
    return CornerDay(
        name,
        float(beta),
        NO_OFFER,
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        0,
        None,
        None,
        {},
        (),
    )


def search_day(prepared: PreparedDay, method, beta: float) -> CornerDay:
    """`method` at `beta` on every corner of `prepared` (module text)."""
    beta = float(beta)
    if prepared.abstains:
        raise ValueError(f"the day abstains ({prepared.abstain_reason}): it has no box to search")
    if beta <= 0:
        return no_offer(method.name, beta)
    corners = corner_paths(prepared.day.n_service)
    found, outcomes, solved, deviated = path_results(prepared, method, beta, corners)
    whole = read(found)
    audited = read([found[corner_index(z)] for z in prepared.paths])
    worst = None if whole.worst is None else found[whole.worst]
    return CornerDay(
        method.name,
        beta,
        whole.status,
        len(found),
        whole.breaking,
        whole.slot_breaking,
        whole.driver_short,
        whole.short_below_tolerance,
        whole.timed_out,
        whole.worst_slot,
        whole.worst_driver,
        None if whole.worst is None else tuple(int(v) for v in corners[whole.worst]),
        None if worst is None else worst.first_failure,
        audited.status,
        audited.breaking,
        audited.worst_slot,
        audited.worst_driver,
        dict(sorted(outcomes.items())),
        tuple(found),
        solved,
        deviated,
    )


def fixed_plan_corners(prepared: PreparedDay, beta: float) -> list:
    """`replay.run_plan` of the certified fixed plan at `beta` on every corner, in corner order."""
    policy = prepared.certificate.policy
    plan = Plan.from_policy(policy, float(beta))
    return [
        run_plan(prepared.day, plan, z, policy.request(z, float(beta)))
        for z in corner_paths(prepared.day.n_service)
    ]


@dataclass(frozen=True)
class ClosedFormCheck:
    """The closed form against the corner worst case of the fixed plan, one day and scale."""

    beta: float
    closed_hour: int | None
    corner_hour: int | None
    closed_slot: float
    corner_slot: float
    slot_difference: float
    agreed_strict: bool
    agreed: bool  # strictly, or by the reason "within tolerance" (reading K3)
    reason: str | None  # None when they agree strictly
    driver_bound: float  # the every-scale bound (module text)
    corner_driver: float
    driver_within_bound: bool
    driver_limited: bool  # every corner of the largest driver loss has a car short of its plan
    driver_no_slot_failure: bool  # some corner of that loss has no failing hour (driver_flags)
    earlier_driver_bound: float  # the earlier bound, a comparison only (module text)


def closed_form_check(prepared: PreparedDay, betas) -> tuple[ClosedFormCheck, ...]:
    """The closed-form comparison of one day at each scale of `betas`, the one WP11's pass mark
    reads (module text, P1)."""
    if prepared.abstains:
        raise ValueError(f"the day abstains ({prepared.abstain_reason}): it has no box to check")
    policy = prepared.certificate.policy
    rows = true_rows(prepared.day)
    thresholds = hour_thresholds(policy, rows)
    steps = slot_steps(thresholds)
    terms = every_scale_terms(prepared)
    out = []
    for beta in betas:
        beta = float(beta)
        runs = fixed_plan_corners(prepared, beta)
        hours = [r.first_failure for r in runs if r.first_failure is not None]
        corner_hour = min(hours) if hours else None
        corner_slot = max(r.slot_loss for r in runs)
        closed_hour = first_failing_hour(thresholds, beta)
        closed_slot = crc.step_loss(*steps, beta) if beta > 0 else 0.0
        difference = abs(closed_slot - corner_slot)
        strict = closed_hour == corner_hour and difference <= SLOT_TOLERANCE
        reason = None
        if not strict:
            (found,) = compare_with_replay(prepared.day, policy, rows, (beta,))
            reason = found.reason or "slot losses differ"
        bound = every_scale_loss(terms, beta) if beta > 0 else 0.0
        earlier = ramp_loss(terms[0], beta) if beta > 0 else 0.0
        corner_driver = max(r.driver_loss for r in runs)
        limited, no_failure = driver_flags(runs)
        out.append(
            ClosedFormCheck(
                beta,
                closed_hour,
                corner_hour,
                float(closed_slot),
                float(corner_slot),
                float(difference),
                strict,
                strict or reason == WITHIN_TOLERANCE,
                reason,
                float(bound),
                float(corner_driver),
                corner_driver <= bound + DRIVER_TOLERANCE,
                limited,
                no_failure,
                float(earlier),
            )
        )
    return tuple(out)


def driver_flags(runs) -> tuple[bool, bool]:
    """(`driver_limited`, `driver_no_slot_failure`) of the replays `runs` of every corner, over
    the corners whose driver loss is within `DRIVER_TOLERANCE` of the largest (reading K4 of
    `corner_run`): every one of them has a car that drew less than planned; some one of them has
    no failing hour."""
    largest = max(r.driver_loss for r in runs)
    tied = [r for r in runs if r.driver_loss >= largest - DRIVER_TOLERANCE]
    return (
        all(drew_less_than_planned(r) for r in tied),
        any(r.first_failure is None for r in tied),
    )


def drew_less_than_planned(run) -> bool:
    """Some car drew less than its planned charge in some hour (`draws < planned`, planned > 0)."""
    planned = np.asarray(run.planned, dtype=float)
    return bool(np.any((planned > 0) & (np.asarray(run.draws, dtype=float) < planned)))
