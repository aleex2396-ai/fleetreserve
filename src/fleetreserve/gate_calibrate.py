"""Calibration and test scoring of one unit under one configuration (plan sections 6, 11 O1, 13
WP6).

Input: the unit's days in date order, each as `gate_day.PreparedDay` with its two ceilings, the
role of each day in the unit's one seeded split (`day_split`, the same for every configuration),
and an `evaluate_batch` function that evaluates a list of (day, beta) pairs with
`gate_day.evaluate` (in a process pool or not; the results do not depend on it).

Per (day, beta) the evaluation runs every rung at once, and it is kept (`DayCache`): no day is
evaluated twice at one beta, whichever rung or search asks for it.

Audited-set route, per rung (fixed plan, one-way re-solve, two-way re-solve at the budget B):

- the day-level losses at beta are the largest slot loss and the largest driver loss over the
  day's audited paths. A day that abstains (no certificate, no steered car, zero value) offers
  nothing and has no loss: it is in neither mean and is not counted in n (plan section 6). n is
  the number of the unit's offering calibration days;
- `crc.calibrate_search` over the offering calibration days for both losses together (the scale
  used), and again for the slot loss alone and the driver loss alone (alpha of the other loss set
  to 1, where the bound always holds), all sharing the cache. With too few offering calibration
  days the scale is 0, and the reason states how many of the calibration days offer;
- the test days are evaluated only at the rung's chosen beta (plan section 6: "test days are
  scored only at the chosen beta"): the offer of a test day is beta times its certified value at
  scale 1, 0 on an abstaining day; the shares of each ceiling come from `gate_stat.unit_offer`
  (ratio of sums, and the mean of daily ratios beside it); the realised losses are the day-level
  losses at that beta, averaged over the offering test days only, with the numbers of offering
  and of failing days (an offering day with slot loss above 0). The mean offer is reported per
  counting test day and per offering test day. At a scale of 0 nothing is offered, so there is no
  loss to average: the loss means and the failing days are None, and the offering days (days with
  a box at scale 1) are still given.

Tiers (plan section 6, "Tiers"; O4). Two tiers are scored side by side and never pooled:

- tier A (driver-protected): the scale is `rungs[rung]["beta_hat"]`, the smaller of the slot and
  driver scales, scored as above under `rungs[rung]["test"]`. This is what the module did before
  the tiers, and nothing about it changes;
- tier B (deadline contract), scored only when `run_job` is asked (`tier_b`): the scale is the slot
  loss alone, `rungs[rung]["slot_only"]["beta_hat"]` of the same calibration, and the test days
  are scored at that scale with the same fields, under `rungs[rung]["tier_b"]["test"]`, beside the
  scale, its reason when it is 0, and the re-solves at that scale. The driver loss is reported, not
  held down. The check that cars staying to their bound have no shortfall (`stay_check`) is
  summed over the day-paths evaluated at the tier B scale, the offering calibration days and the
  test days, per rung, under `rungs[rung]["tier_b"]["stay_to_bound"]`. Like T5, a break stops
  nothing: it is counted.

The every-request route has the same two tiers: `fixed_every_request` is tier A, and with
`tier_b` its key "tier_b" holds the scale of the slot loss alone (`slot_beta`, capped at the top of
the grid and limited to the validity limit like the other) with the same test-day fields and the
offers at that scale. Its stay check
is not computed: the route replays no path.

Test days are scored only at each tier's chosen scale, and the cache keeps a day from being
evaluated twice at one scale, so a scale the tiers share is evaluated once. The tier A counts of
re-solves and evaluations are taken before tier B is scored, so they do not depend on whether it
is; the T5 counts cover every evaluated day-path of either tier.

Every-request route for the fixed plan (plan section 6, T6; O1's "every-request offer"):

- slot loss in closed form from the hour thresholds `fixed_plan_loss.hour_thresholds` on the
  steered cars' true rows, with each car's decision-hour draw in place of its planned hour-0 power
  (the same at every beta, `fixed_plan_loss` module text): the loss is `(n - t + 1) / n` once beta
  is above `min_{s <= t} s_s`, a step function, calibrated by `crc.calibrate_exact` (a negative
  threshold, a row broken at the centre, is a loss at every beta above 0);
- driver loss: the every-scale bound (`fixed_plan_loss.every_scale_driver_bound`, reading R3;
  here `every_scale_loss` of `every_scale_terms`, capped at 1, the largest a share can be), with
  each car's decision-hour draw in place of its planned hour-0 power (what it bounds, and with
  what error: `fixed_plan_loss`, module text, reading R3, V3 and V7). By the owner's decision of 16
  November 2026 it is the driver bound wherever the route, the sizing what-if, T7, the corner check
  and the comparison with replay use one (below). It does not fall as beta rises and can jump up
  where a power row breaks; its calibrated scale is found by doubling and bisection on the
  condition to `RAMP_TOLERANCE` (`beta_hat_every_scale`; the scale as calibrated has no cap and can
  be above 1 or infinite, written "inf"). A car none of whose power rows is broken at beta keeps
  the term of the earlier bound (below), which the route computes from `driver_terms`, whose gain
  per car is `worst_case(1) - worst_case(0)`: the need row's `|a G|_1` up to a rounding of at most
  `2^-53 (|a g + |a G|_1| + |a G|_1)` (with a relative `2^-53` on top), `a g` its centre as the
  code computes it. In exact arithmetic on the values the code holds, at scale beta such a term
  differs from `driver_loss_bound`'s by at most `2^-53 |a g - b|`, the rounding of its centre
  term, plus beta times its gain's rounding; at the validity limit that second part is at most
  1.111e-11 times the car's `|a g + |a G|_1| + |a G|_1`. The code sums the terms in floating point;
  that rounding is not bounded here;
- the scale is the smaller of the two. A scale with no limit (for both losses the condition
  `(sum of the days' losses + 1) / (n + 1) <= alpha` holds at every beta, as it can while some days
  lose) is capped at the top of the plan's grid (`crc.GRID[-1]`), the cap the search route
  has, and marked as capped; a finite scale above the validity limit is limited to it (below);
- the calibration uses the offering calibration days only, as in the audited-set route;
- test days: the offer at that scale, and the closed-form losses (slot loss and driver-loss bound)
  at that scale, averaged over the offering test days, with their number and the number of failing
  days. At scale 0 nothing is offered, so the loss means and the failing days are None.

Which driver-loss bound each figure uses. Every field above, tier B, the sizing what-if
(`size_scale`), T7 (`t7_stat`, `t7_unit`), the corner check (`corner_search.closed_form_check`)
and the comparison with replay (`fixed_plan_loss.compare_with_replay`) use the every-scale bound.
The earlier bound, the plan's "summed per-car worst cases" (`fixed_plan_loss.driver_loss_bound`,
reading R2; here `ramp_loss` of `driver_terms`, capped at 1), is not a bound at a scale where an
hour fails, so it is the source of no offer, scale or pass. `run_job` writes it beside tier A for
comparison only, under `fixed_every_request["earlier_driver_bound"]` (`EARLIER`, its first field
`about` the sentence `EARLIER_ABOUT`): tier A of the same route under it, that is its driver scale
(`beta_hat_ramps`, the same condition and bisection on `ramp_loss`, continuous and piecewise linear
in beta), the smaller of it and the same slot scale, capped at the top of the grid in the same way
when it is infinite and limited like every scale of the route (below), and at that scale the same
test-day fields, with the earlier bound as the driver loss, and the offers; and with tier B,
under `fixed_every_request["tier_b"]["earlier_driver_bound"]` (the same sentence `about` first),
the earlier bound's mean over the offering test days at tier B's scale
(`test_mean_driver_loss_bound_offering_days`). Every other field is the same with or without
them.

How the two driver scales differ (`tests/wp11/test_every_scale_readings.py`, worked by hand). The
every-scale bound equals the earlier bound at every beta where no car's power row is broken; on a
day with a broken power row it can be above or below it, so its scale can be lower or higher:

- lower, where a car's rate or departure caps an hour (its power-high row): the every-scale bound
  counts the hour at its cap, the earlier bound at its plan;
- higher, where an hour's plan falls below 0 on some path (its power-low row): a one-way car then
  draws 0, which the every-scale bound's by-hour bound counts, while the earlier bound counts the
  plan below 0 as energy taken away and keeps rising with beta;
- infinite, where that floor at 0 keeps every day's every-scale bound within the level at every
  beta while the earlier bound rises without limit.

A very large finite driver scale of the formula (for example an energy margin of 0.6 kWh over a
gain of 1e-14, 6e13) is a value of the formula, not an error of the calibration: the thresholds
read a row whose gains are at most `policy.ZERO_GAIN` as not depending on the requests, while the
every-scale bound and its limit (`_every_scale_limit`) count every gain as it is, so once every
hour with a larger gain is broken and counted at 0 or more, such a gain alone sets the scale. Above
the validity limit V5 to V7 of `fixed_plan_loss`'s module text are not claimed. On 19 copies of
day H3 of the tests both formula driver scales are 1.9998e12, where replay's driver loss is 0.0625;
the route uses the limit there.

The validity limit (`policy.VALIDITY_LIMIT`; what it limits and why: `fixed_plan_loss`, module
text, "The validity limit and the stated errors"). Every finite scale the route forms, in tier A
and tier B and under both driver-loss bounds, is used and written as at most the limit
(`_limited`): `slot_beta`, `driver_beta`, `beta` and `beta_used` of a block are the calibrated
values when at most the limit, else the limit. A block in which a calibrated finite value was above
the limit carries a mark under `VALIDITY` ("validity_limit"): the limit, those calibrated values by
field name (`formula_values`) and why (`VALIDITY_WHY`). A block with none has no mark, so an output
in which no scale passes the limit is as before, byte for byte. An infinite scale stays "inf", and
both scales infinite still give the grid's top 2. Where a calibrated finite scale is above the
limit, the calibration's condition holds at the limit: it holds at the calibrated scale, and every
loss is non-decreasing in the scale. What replay can differ from the closed forms by at the scale
used: `fixed_plan_loss`, module text, V2 to V8; tier A and tier B can score days with a failing
hour at that scale (the calibration allows some). Not chosen: an always-present field (such as a
flag `limited`) would change every output file; the formula's values in the scale fields with the
mark beside them would leave a value above the limit where a reader takes a certified scale. The
sizing what-if, T7 and the corner check use day scales of at most 1.2 (`size_scale`'s day scales at
most 1, T7's windows of half-width at most 0.2 about a scalar at most 1, the corner check's grid up
to 1). Their calibrations' bisections can read the formula above the limit, but a scale capped at 1
does not depend on what the formula gives above 1 (the argument above), so the limit does not reach
them.

Tier A's scale is the smaller of its driver scale and the slot scale, used as it is when finite up
to the validity limit, and the grid's top only when both are infinite, so it is never above the
slot scale. It is above the earlier bound's tier A scale when the earlier bound's driver scale is
below the validity limit and below both the slot scale and the every-scale driver scale, unless
both of these are infinite: it is then the grid's top 2, below the earlier bound's tier A scale
when that scale is above 2 (unit H4 of the tests: 2 against 3.2). It is above the grid's top when
it is finite and above 2: only an infinite scale is capped, under either bound.

T5 is counted on every evaluated day-path, calibration and test days alike; a break does not stop
anything, it is counted and each one is kept.

What the split separates: with the "other-days" forecast reference (`forecast_empirical`), every
day's forecast uses all other days of the unit, so a calibration day's forecast uses the outcomes
of test days and a test day's forecast those of calibration days. In the pooled unit, site-days of
the same date can fall on both sides of the split. This keeps the days exchangeable, which the
calibration needs, but it is not a train-then-test separation in time.
"""

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from fleetreserve import crc, gate_stat, stay_check
from fleetreserve.carsets import true_rows
from fleetreserve.fixed_plan_loss import (
    closed_form_slot_loss,
    drawn_policy,
    every_scale_shortfalls,
    hour_thresholds,
    service_hours,
)
from fleetreserve.gate_config import CEILINGS
from fleetreserve.gate_day import Ceilings, DayEvaluation, PreparedDay, evaluate
from fleetreserve.policy import VALIDITY_LIMIT
from fleetreserve.replay import protected
from fleetreserve.rungs import RUNGS

RAMP_TOLERANCE = 1e-12  # relative width at which the bisection on the driver-loss bound stops
INF = "inf"  # how an infinite number is written in the results
EARLIER = (
    "earlier_driver_bound"  # the key of a tier's figures under the earlier bound (module text)
)
EARLIER_ABOUT = (
    "the figures of this tier under the earlier driver-loss bound "
    "(fixed_plan_loss.driver_loss_bound), which is not a bound where an hour fails: a comparison "
    "only, never the source of an offer, a scale or a pass"
)
VALIDITY = "validity_limit"  # the key of the mark of a block with a scale above the limit
VALIDITY_WHY = (
    "a finite scale above policy.VALIDITY_LIMIT is written and used as that limit, and "
    "formula_values holds the values as calibrated; what the limit limits and why: "
    "fixed_plan_loss, module text, 'The validity limit and the stated errors'"
)


def number(value):
    """A float for the results file: infinity written as text, everything else as is."""
    if value is None:
        return None
    value = float(value)
    return INF if math.isinf(value) else value


# --- the cache -----------------------------------------------------------------------------------


class DayCache:
    """Evaluations by (day index, beta), each made once. `calls[(day, beta)]` counts them."""

    def __init__(self, prepared: Sequence[PreparedDay], evaluate_batch: Callable):
        self.prepared = list(prepared)
        self.evaluate_batch = evaluate_batch
        self.store: dict[tuple[int, float], DayEvaluation] = {}
        self.calls: Counter = Counter()

    def get(self, days: Sequence[int], beta: float) -> list[DayEvaluation]:
        beta = float(beta)
        wanted = [(d, beta) for d in days if (d, beta) not in self.store]
        inline = [k for k in wanted if self.prepared[k[0]].abstains]
        pooled = [k for k in wanted if not self.prepared[k[0]].abstains]
        for key in inline:
            self.store[key] = evaluate(self.prepared[key[0]], beta).compact()
        if pooled:
            results = self.evaluate_batch([(self.prepared[d], b) for d, b in pooled])
            for key, result in zip(pooled, results, strict=True):
                self.store[key] = result
        self.calls.update(wanted)
        return [self.store[(d, beta)] for d in days]


# --- the every-request route ---------------------------------------------------------------------


def slot_steps(thresholds) -> tuple[list[float], list[float]]:
    """The closed-form slot loss as `crc` steps: (thresholds, levels), non-decreasing."""
    s = np.asarray(thresholds, dtype=float)
    n = s.size
    running = np.minimum.accumulate(s)  # min over hours 1..t
    steps, levels = [], []
    for t in range(n, 0, -1):
        m = running[t - 1]
        if math.isinf(m) and m > 0:
            break
        steps.append(max(float(m), 0.0))
        levels.append((n - t + 1) / n)
    return steps, levels


def driver_terms(prepared: PreparedDay) -> tuple[np.ndarray, np.ndarray, float]:
    """Per steered car the need row's worst case at scale beta as `c + beta s`, minus its bound,
    on the policy with the decision hour's draws, and the day's protected energy (see
    `fixed_plan_loss.driver_loss_bound`)."""
    rows = true_rows(prepared.day)
    policy = drawn_policy(prepared.certificate.policy, rows)
    need = rows.kinds == "need"
    at_zero = policy.worst_case(rows, 0.0)[need]
    spread = policy.worst_case(rows, 1.0)[need] - at_zero
    return at_zero - rows.bounds[need], spread, float(protected(prepared.day).sum())


def ramp_loss(terms, beta: float) -> float:
    """The earlier driver-loss bound of one day at `beta > 0`, capped at 1 (module text: a
    comparison only)."""
    c, s, total = terms
    if total <= 0 or c.size == 0:
        return 0.0
    return min(1.0, float(np.maximum(c + beta * s, 0.0).sum() / total))


def _ramp_limit(terms) -> float:
    """The loss of one day as beta grows without bound."""
    c, s, total = terms
    if total <= 0 or c.size == 0:
        return 0.0
    if np.any(s > 0):
        return 1.0
    return min(1.0, float(np.maximum(c, 0.0).sum() / total))


def every_scale_terms(prepared: PreparedDay) -> tuple:
    """`driver_terms` of the day and the service hours the every-scale bound reads
    (`fixed_plan_loss.service_hours`, on the steered cars' true rows)."""
    hours = service_hours(prepared.certificate.policy, true_rows(prepared.day))
    return driver_terms(prepared), hours


def every_scale_loss(terms, beta: float) -> float:
    """The every-scale driver-loss bound of one day at `beta > 0` (module text), capped at 1;
    `ramp_loss` itself on a day where no car's power row is broken at `beta`."""
    (c, s, total), hours = terms
    if total <= 0 or c.size == 0:
        return 0.0
    today = np.maximum(c + beta * s, 0.0)
    return min(1.0, float(every_scale_shortfalls(hours, today, beta).sum() / total))


def _every_scale_limit(terms) -> float:
    """The every-scale loss of one day as beta grows without bound: a car none of whose power rows
    is ever broken keeps today's ramp; every other car tends to its shortfall against its
    decision-hour draw, the planned power of the hours that never depend on the requests and, in
    the other hours whose gains are 0, that power clipped to its rows (`fixed_plan_loss`, reading
    R3)."""
    (c, s, total), hours = terms
    if total <= 0 or c.size == 0:
        return 0.0
    never = np.isinf(hours.thresholds) & (hours.thresholds > 0)
    clipped = ~never.all(axis=1)
    fixed = (
        hours.first
        + np.where(never, hours.centre, 0.0).sum(axis=1)
        + np.where(~never & (hours.spread == 0), np.clip(hours.centre, 0.0, hours.cap), 0.0).sum(
            axis=1
        )
    )
    linear = np.abs(np.where(never[..., None], hours.gains, 0.0).sum(axis=1)).sum(axis=1)
    limit = np.where(s > 0, math.inf, np.maximum(c, 0.0))
    limit[clipped] = np.where(
        linear[clipped] > 0, math.inf, np.maximum(hours.short[clipped] - fixed[clipped], 0.0)
    )
    return min(1.0, float(limit.sum() / total))


def _beta_hat(days: Sequence, alpha: float, loss: Callable, limit: Callable) -> float:
    """sup{beta >= 0 : (sum_d L_d(beta) + 1) / (n + 1) <= alpha} for losses `loss(terms_d, beta)`,
    each non-decreasing for beta > 0 (0 at beta <= 0) with the limit `limit(terms_d)`, by bisection
    to `RAMP_TOLERANCE`. Infinity when the bound holds at every beta; 0 when it fails just above 0
    or there are too few days (`crc.enough_days`)."""
    n = len(days)
    if not crc.enough_days(n, alpha):
        return 0.0

    def holds(beta):
        total = sum(loss(terms, beta) for terms in days)
        return total + 1.0 <= alpha * (n + 1) + crc.TOLERANCE

    if sum(limit(terms) for terms in days) + 1.0 <= alpha * (n + 1) + crc.TOLERANCE:
        return math.inf
    if not holds(np.nextafter(0.0, 1.0)):
        return 0.0
    low, high = 0.0, 1.0
    while holds(high):
        low, high = high, 2.0 * high
    while high - low > RAMP_TOLERANCE * max(1.0, high):
        middle = 0.5 * (low + high)
        if holds(middle):
            low = middle
        else:
            high = middle
    return low


def beta_hat_ramps(days: Sequence, alpha: float) -> float:
    """`_beta_hat` for the losses `ramp_loss(terms_d)`, each non-decreasing and continuous for
    beta > 0."""
    return _beta_hat(days, alpha, ramp_loss, _ramp_limit)


def beta_hat_every_scale(days: Sequence, alpha: float) -> float:
    """`_beta_hat` for the losses `every_scale_loss(terms_d)`, each non-decreasing for beta > 0,
    with upward jumps where a power row breaks (`fixed_plan_loss`, reading R3)."""
    return _beta_hat(days, alpha, every_scale_loss, _every_scale_limit)


def offering_days(prepared: Sequence[PreparedDay], days: Sequence[int]) -> list[int]:
    """The days among `days` that offer: those that do not abstain."""
    return [d for d in days if not prepared[d].abstains]


def _every_scale_driver_loss(p: PreparedDay, beta: float) -> float:
    return every_scale_loss(every_scale_terms(p), beta)


def _earlier_driver_loss(p: PreparedDay, beta: float) -> float:
    return ramp_loss(driver_terms(p), beta)


def _every_request_test(
    prepared, test_offering, used: float, driver=_every_scale_driver_loss
) -> dict:
    """The test-day scores of the every-request route at scale `used` (module text): the closed
    forms averaged over the offering test days, the driver loss by `driver(day, beta)`. At scale 0
    nothing is offered, so the loss means and the failing days are None."""
    slot_losses, driver_losses = [], []
    for d in test_offering if used > 0 else []:
        p = prepared[d]
        thresholds = hour_thresholds(p.certificate.policy, true_rows(p.day))
        slot_losses.append(closed_form_slot_loss(thresholds, used))
        driver_losses.append(driver(p, used))
    return {
        "test_offering_days": len(test_offering),
        "test_failing_days": sum(loss > 0 for loss in slot_losses) if used > 0 else None,
        "test_mean_slot_loss_offering_days": _mean(slot_losses),
        "test_mean_driver_loss_bound_offering_days": _mean(driver_losses),
    }


def _limited(beta: float) -> float:
    """`beta` within the validity limit (module text): a finite scale above
    `policy.VALIDITY_LIMIT` becomes the limit; an infinite one stays infinite."""
    return beta if math.isinf(beta) else min(beta, VALIDITY_LIMIT)


def _validity_mark(**scales) -> dict:
    """`{VALIDITY: ...}` when a finite scale of `scales` (field name to the value as calibrated)
    is above the validity limit, else `{}` (module text)."""
    above = {k: float(v) for k, v in scales.items() if not math.isinf(v) and v > VALIDITY_LIMIT}
    if not above:
        return {}
    return {VALIDITY: {"limit": VALIDITY_LIMIT, "formula_values": above, "why": VALIDITY_WHY}}


def _every_request_scale(beta: float, offering, calibration, detail: str) -> dict:
    """The scale of one tier of the every-request route: `beta` as calibrated (infinite when the
    calibration's condition holds at every beta, module text), the scale used (capped at the top of
    the plan's grid when infinite, else limited to the validity limit), and why it is 0 if it
    is."""
    capped = math.isinf(beta)
    used = crc.GRID[-1] if capped else _limited(beta)
    reason = ""
    if used <= 0:
        too_few = not (
            crc.enough_days(len(offering), crc.ALPHA_SLOT)
            and crc.enough_days(len(offering), crc.ALPHA_DRIVER)
        )
        reason = scale_zero_reason(too_few, len(offering), len(calibration), detail)
    return {"beta": beta, "used": used, "capped": capped, "reason": reason}


def every_request(
    prepared, calibration, test, *, tier_b: bool = False, earlier: bool = False
) -> dict:
    """The fixed plan's every-request calibration and its test-day scores (module text), with the
    every-scale driver-loss bound as the driver bound. Tier A: the scale is the smaller of the
    slot and driver scales. With `tier_b`, tier B beside it under "tier_b": the scale of the slot
    loss alone, scored the same way. With `earlier`, each tier's figures under the earlier
    driver-loss bound under its key `EARLIER`, a comparison only (module text)."""
    offering = offering_days(prepared, calibration)
    cal_slot, cal_driver = [], []
    for d in offering:
        p = prepared[d]
        thresholds = hour_thresholds(p.certificate.policy, true_rows(p.day))
        cal_slot.append(slot_steps(thresholds))
        cal_driver.append(every_scale_terms(p))
    exact = crc.calibrate_exact(cal_slot, [((), ())] * len(cal_slot))
    driver_beta = beta_hat_every_scale(cal_driver, crc.ALPHA_DRIVER)
    detail = exact.reason or "the driver-loss bound fails at every beta above 0"
    scale = _every_request_scale(min(exact.slot_beta, driver_beta), offering, calibration, detail)
    used = scale["used"]
    test_offering = offering_days(prepared, test)
    out = {
        "slot_beta": number(_limited(exact.slot_beta)),
        "driver_beta": number(_limited(driver_beta)),
        "beta": number(_limited(scale["beta"])),
        "beta_used": number(used),
        "capped_at_grid_top": scale["capped"],
        "abstain": used <= 0,
        "reason": scale["reason"],
        "offering_calibration_days": len(offering),
        **_every_request_test(prepared, test_offering, used),
        **_validity_mark(slot_beta=exact.slot_beta, driver_beta=driver_beta, beta=scale["beta"]),
    }
    if tier_b:
        detail_b = exact.reason or "the slot loss fails at every beta above 0"
        scale_b = _every_request_scale(exact.slot_beta, offering, calibration, detail_b)
        used_b = scale_b["used"]
        out["tier_b"] = {
            "beta": number(_limited(scale_b["beta"])),
            "beta_used": number(used_b),
            "capped_at_grid_top": scale_b["capped"],
            "abstain": used_b <= 0,
            "reason": scale_b["reason"],
            **_every_request_test(prepared, test_offering, used_b),
            "test_mean_offer_kwh_per_day": _mean([used_b * prepared[d].value for d in test]),
            "test_mean_offer_kwh_per_offering_day": _mean(
                [used_b * prepared[d].value for d in test_offering]
            ),
            **_validity_mark(beta=scale_b["beta"]),
        }
        if earlier:
            found = _every_request_test(prepared, test_offering, used_b, _earlier_driver_loss)
            out["tier_b"][EARLIER] = {
                "about": EARLIER_ABOUT,
                "test_mean_driver_loss_bound_offering_days": found[
                    "test_mean_driver_loss_bound_offering_days"
                ],
            }
    if earlier:
        ramps = [terms for terms, _ in cal_driver]
        out[EARLIER] = _earlier_tier_a(prepared, ramps, offering, calibration, test, exact)
    return out


def _earlier_tier_a(prepared, ramps, offering, calibration, test, exact) -> dict:
    """Tier A of the every-request route under the earlier driver-loss bound (module text), a
    comparison only: the slot scale of `exact` (the same calibration as tier A), the driver scale
    of the earlier bound on the offering calibration days (`ramps`, their `driver_terms`), the
    smaller of the two, and the test-day scores and offers at it."""
    driver_beta = beta_hat_ramps(ramps, crc.ALPHA_DRIVER)
    detail = exact.reason or "the earlier driver-loss bound fails at every beta above 0"
    scale = _every_request_scale(min(exact.slot_beta, driver_beta), offering, calibration, detail)
    used = scale["used"]
    test_offering = offering_days(prepared, test)
    return {
        "about": EARLIER_ABOUT,
        "driver_beta": number(_limited(driver_beta)),
        "beta": number(_limited(scale["beta"])),
        "beta_used": number(used),
        "capped_at_grid_top": scale["capped"],
        "abstain": used <= 0,
        "reason": scale["reason"],
        **_every_request_test(prepared, test_offering, used, _earlier_driver_loss),
        "test_mean_offer_kwh_per_day": _mean([used * prepared[d].value for d in test]),
        "test_mean_offer_kwh_per_offering_day": _mean(
            [used * prepared[d].value for d in test_offering]
        ),
        **_validity_mark(driver_beta=driver_beta, beta=scale["beta"]),
    }


def _mean(values) -> float | None:
    return float(np.mean(values)) if len(values) else None


# --- the audited-set route -----------------------------------------------------------------------


def scale_zero_reason(too_few: bool, offering: int, calibration: int, detail: str) -> str:
    """Why a scale is 0, in words that tell the two causes apart: too few offering calibration days
    (abstains, nothing was calibrated) or no scale above 0 that the bound allows."""
    lead = "abstains" if too_few else "no scale is safe"
    return f"{lead}: {offering} of {calibration} calibration days offer; {detail}"


def _search(result: crc.SearchCalibration, calibration: int) -> dict:
    """`result` as results; a reason says how many of the `calibration` days offer."""
    reason = result.reason
    if reason:
        # `crc.calibrate_search` evaluates nothing when the days are too few
        reason = scale_zero_reason(not result.evaluations, result.n_days, calibration, reason)
    return {
        "beta_hat": number(result.beta),
        "bracket": [number(result.bracket[0]), number(result.bracket[1])],
        "grid_top_reached": result.grid_top_reached,
        "abstain": result.abstain,
        "reason": reason,
        "n_evaluations": result.n_evaluations,
        "evaluations": [
            [e.beta, e.mean_slot_loss, e.mean_driver_loss, e.holds] for e in result.evaluations
        ],
    }


def calibrate_rung(cache: DayCache, calibration: Sequence[int], rung: str) -> dict:
    """`crc.calibrate_search` for one rung on the offering calibration days: both losses, then each
    alone."""
    offering = offering_days(cache.prepared, calibration)

    def loss_at(beta):
        found = cache.get(offering, beta)
        slot = np.array([e.rungs[rung].slot_loss for e in found])
        driver = np.array([e.rungs[rung].driver_loss for e in found])
        return slot, driver

    n = len(offering)
    both = crc.calibrate_search(loss_at, n)
    slot = crc.calibrate_search(loss_at, n, alpha_driver=1.0)
    driver = crc.calibrate_search(loss_at, n, alpha_slot=1.0)
    out = _search(both, len(calibration))
    out["slot_only"] = _search(slot, len(calibration))
    out["driver_only"] = _search(driver, len(calibration))
    return out


def _share(values, ceilings, beta) -> dict | None:
    if any(c is None for c in ceilings):
        return None
    try:
        offer = gate_stat.unit_offer(values, ceilings, beta)
    except ValueError:
        return None
    return {
        "ratio_of_sums": offer.share_of_sums,
        "mean_of_daily_ratios": offer.mean_of_daily_ratios,
    }


def ceiling_values(ceilings: Ceilings, name: str) -> float | None:
    return ceilings.hindsight if name == CEILINGS[0] else ceilings.best_causal


def score_test(cache, prepared, ceilings, test, rung, beta) -> tuple[dict, list[dict]]:
    """The rung's offer on the test days at its chosen `beta`, and its realised losses over the
    offering test days (module text). With `beta` 0 nothing is offered, so there is no loss to
    average: the loss means and the number of failing days are None."""
    values = [prepared[d].value for d in test]
    found = cache.get(test, beta) if beta > 0 else [None] * len(test)
    slot = [0.0 if e is None else e.rungs[rung].slot_loss for e in found]
    driver = [0.0 if e is None else e.rungs[rung].driver_loss for e in found]
    offers = [beta * v for v in values]
    cars = sum(prepared[d].n_cars for d in test)
    shares = {
        name: _share(values, [ceiling_values(ceilings[d], name) for d in test], beta)
        for name in CEILINGS
    }
    offering = [i for i, d in enumerate(test) if not prepared[d].abstains]
    slot_offering = [slot[i] for i in offering]
    offered = beta > 0
    summary = {
        "days": len(test),
        "offering_days": len(offering),
        "failing_days": sum(loss > 0 for loss in slot_offering) if offered else None,
        "mean_offer_kwh_per_day": _mean(offers),
        "mean_offer_kwh_per_offering_day": _mean([offers[i] for i in offering]),
        "offer_kwh_per_car_per_day": None if cars == 0 else float(sum(offers) / cars),
        "share_of_ceiling": shares,
        "mean_slot_loss_offering_days": _mean(slot_offering) if offered else None,
        "mean_driver_loss_offering_days": _mean([driver[i] for i in offering]) if offered else None,
    }
    rows = [
        {"offer_kwh": o, "slot_loss": s, "driver_loss": r}
        for o, s, r in zip(offers, slot, driver, strict=True)
    ]
    return summary, rows


def _stay_to_bound(found: Sequence[DayEvaluation], rung: str, beta: float, tolerance) -> dict:
    """The stay-to-bound check of `rung` summed over the evaluations `found` (all at `beta`)."""
    total = stay_check.NONE
    days = 0
    for e in found:
        if not e.abstained:
            days += 1
            total = stay_check.add(total, e.rungs[rung].stay)
    return {
        "scale": beta,
        "tolerance_kwh": tolerance,
        "days": days,
        "cars_checked": total.cars,
        "car_paths_checked": total.car_paths,
        "cars_short": total.short_cars,
        "car_paths_short": total.short_car_paths,
        "largest_shortfall_kwh": total.worst_kwh,
    }


def _resolves_at(cache: DayCache, days: Sequence[int], beta: float, rung: str) -> dict:
    counts = Counter()
    for d in days:
        e = cache.store.get((d, beta))
        if e is not None:
            counts.update(e.rungs[rung].resolves)
    return dict(sorted(counts.items()))


def score_tier_b(cache, prepared, ceilings, calibration, test, rung, result) -> tuple[dict, list]:
    """Tier B of one rung (module text): the slot-only scale of `result` (the rung's calibration),
    the test days scored there, the re-solves and the stay-to-bound check at that scale. Returns
    the tier B results and its per-day rows (as `score_test`)."""
    search = result["slot_only"]
    beta = float(search["beta_hat"])
    summary, rows = score_test(cache, prepared, ceilings, test, rung, beta)
    offering = offering_days(prepared, calibration)
    cal_found = cache.get(offering, beta) if beta > 0 else []
    test_found = cache.get(test, beta) if beta > 0 else []
    tolerance = prepared[0].t5_tolerance_kwh if len(prepared) else None
    out = {
        "beta_hat": search["beta_hat"],
        "scale_from": "the slot loss alone (the driver loss is reported, not limited)",
        "abstain": search["abstain"],
        "reason": search["reason"],
        "test": summary,
        "resolves": {
            "calibration": _resolves_at(cache, offering, beta, rung),
            "test": _resolves_at(cache, test, beta, rung),
        },
        "stay_to_bound": _stay_to_bound([*cal_found, *test_found], rung, beta, tolerance),
    }
    return out, rows


def _resolves(cache: DayCache, days: set[int]) -> dict:
    out = {}
    for rung in RUNGS:
        counts = Counter()
        for (d, _), e in cache.store.items():
            if d in days:
                counts.update(e.rungs[rung].resolves)
        out[rung] = dict(sorted(counts.items()))
    return out


def _counts(items) -> dict:
    return dict(sorted(Counter(items).items()))


def run_job(
    prepared: Sequence[PreparedDay],
    ceilings: Sequence[Ceilings],
    roles: Sequence[str],
    evaluate_batch: Callable,
    *,
    tier_b: bool = False,
) -> tuple[dict, list[dict], list[dict], DayCache]:
    """Calibrate and score one unit under one configuration.

    `roles[d]` is "calibration" or "test". `tier_b`: also score tier B (module text); tier A is
    always scored. Returns the results (JSON-ready), one row per day, the T5 breaks (one row each)
    and the cache (for counting and timing)."""
    calibration = [d for d, role in enumerate(roles) if role == "calibration"]
    test = [d for d, role in enumerate(roles) if role == "test"]
    cache = DayCache(prepared, evaluate_batch)
    rungs, day_rows = {}, [dict() for _ in prepared]
    for rung in RUNGS:
        result = calibrate_rung(cache, calibration, rung)
        beta = float(result["beta_hat"])
        result["test"], rows = score_test(cache, prepared, ceilings, test, rung, beta)
        for d, row in zip(test, rows, strict=True):
            day_rows[d][rung] = row
        rungs[rung] = result
    cal_set, test_set = set(calibration), set(test)
    for rung in RUNGS:
        rungs[rung]["resolves"] = {
            "calibration": _resolves(cache, cal_set)[rung],
            "test": _resolves(cache, test_set)[rung],
        }
    evaluations = {  # tier A's, taken before tier B is scored
        "calibration": sum(1 for d, _ in cache.store if d in cal_set),
        "test": sum(1 for d, _ in cache.store if d in test_set),
    }
    if tier_b:
        for rung in RUNGS:
            scored, rows = score_tier_b(
                cache, prepared, ceilings, calibration, test, rung, rungs[rung]
            )
            rungs[rung]["tier_b"] = scored
            for d, row in zip(test, rows, strict=True):
                day_rows[d][rung]["tier_b"] = row
    fixed_exact = every_request(prepared, calibration, test, tier_b=tier_b, earlier=True)
    fixed_exact["test_mean_offer_kwh_per_day"] = _mean(
        [fixed_exact["beta_used"] * prepared[d].value for d in test]
    )
    fixed_exact["test_mean_offer_kwh_per_offering_day"] = _mean(
        [fixed_exact["beta_used"] * prepared[d].value for d in offering_days(prepared, test)]
    )
    values = [p.value for p in prepared]
    breaks, checked = [], 0
    for (d, beta), e in sorted(cache.store.items()):
        checked += e.paths_checked
        for b in e.t5_breaks:
            breaks.append({"day": d, "beta": beta, "break": b})
    beta_hats = [float(rungs[r]["beta_hat"]) for r in RUNGS]
    results = {
        "calibration_days": len(calibration),
        "offering_calibration_days": len(offering_days(prepared, calibration)),
        "test_days": len(test),
        "days_abstaining": {
            "calibration": _counts(
                prepared[d].abstain_reason for d in calibration if prepared[d].abstains
            ),
            "test": _counts(prepared[d].abstain_reason for d in test if prepared[d].abstains),
        },
        "certificate_outcomes": _counts(p.certificate_outcome for p in prepared),
        "cars": sum(p.n_cars for p in prepared),
        "steered_cars": sum(len(p.steered) for p in prepared),
        "cars_left_unsteered": _counts(reason for p in prepared for _, reason in p.unsteered),
        "mean_certified_value_kwh": {
            "all_days": _mean(values),
            "test_days": _mean([values[d] for d in test]),
        },
        "ceiling_sums_test_days_kwh": {
            name: _sum_or_none([ceiling_values(ceilings[d], name) for d in test])
            for name in CEILINGS
        },
        "ceilings_missing_test_days": {
            name: sum(ceiling_values(ceilings[d], name) is None for d in test) for name in CEILINGS
        },
        "rungs": rungs,
        "beta_hat_order_fixed_le_one_way_le_two_way": beta_hats == sorted(beta_hats),
        "fixed_every_request": fixed_exact,
        "t5": {
            "day_paths_checked": checked,
            "breaks_a": sum(b["break"].kind == "a" for b in breaks),
            "breaks_b": sum(b["break"].kind == "b" for b in breaks),
        },
        "evaluations": evaluations,
    }
    return results, day_rows, breaks, cache


def _sum_or_none(values) -> float | None:
    return None if any(v is None for v in values) else float(sum(values))
