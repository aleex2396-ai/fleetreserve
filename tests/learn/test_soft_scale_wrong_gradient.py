"""T8's check (plan section 7) can fail: `soft_scale_check`'s measuring rule handed gradients that
are deliberately wrong, on seeded made-up fleets.

The fleets: `soft_scale_fake.fleet_day` with 30 and 300 drawn cars, 1 to 4 service hours, with
fractional departures (a fractional last hour) and with departures on whole hours; the centre a
point of the fleet's forecast set (`point_of`), the widths of `shape` scaled so that the exact
scale is 1, as on a certified day. `FLEETS` fleets per size and kind of departure. Per fleet and
field (departure, cap, need, and rate, which the command reports only), one seeded random unit
direction `v`, and for each function (the soft minimum at `SIGMA`, the exact scale; the exact scale
is left out on a fleet whose smallest terms tie, as the command leaves it out) one measurement
(`soft_scale_check.measure`, which never reads the gradient).

The gradients handed to the rule (`soft_scale_check.decide`): with `G` the autograd gradient of
the unchanged function and `a = G . v`, "all": `(1 + x) a`, every coordinate wrong by the factor;
"one": `a + x G_j v_j`, the coordinate `j` with the largest `|G_j v_j|` wrong by it; `x` in
`ERRORS` (100%, 10%, 1%, 0.2% = twice the target, and 0.01% = a tenth of it), and `x = 0`. The
true relative error of a handed derivative is `|wrong - a| / |a|` (autograd of the unchanged
function is taken as the derivative: `test_soft_scale_gradient.py` and the measuring rule find it
right on these fleets).

A comparison is DECIDABLE when one of its measured steps is resolvable (`tau <= LIMIT |D| / 4`)
and shows no kink (a step that shows one decides nothing).
Asserted: (1) on every decidable comparison, every handed derivative with a true relative error
above the target (1e-3) FAILS, and every one with a true error of at most 0.4 times the target
(the 0.01% errors, the one-coordinate errors that small, and the unchanged gradient) PASSES; (2)
on every comparison, decidable or not, none with a true error above the target passes and none
with a true error of at most 0.4 times the target fails; (3) the test is not empty: in each size
and kind of departure, each function has decidable comparisons in cap and need, and with
fractional departures in departure too. The table of shares is printed (`pytest -s`), with the
reasons a comparison is not decidable (`_why`).
"""

import math
import random
from collections import Counter, defaultdict
from dataclasses import replace

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import torch
from soft_scale_fake import fleet_day, point_of, shape

from fleetreserve import soft_scale
from fleetreserve import soft_scale_check as check

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net

SEED = 1511
SIZES = (30, 300)
FLEETS = 24
ERRORS = (1.0, 0.1, 0.01, 0.002, 0.0001)
MODES = ("all", "one")
WITHIN = 0.4 * check.LIMIT


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def _case(rng, cars: int, whole: bool):
    n = rng.randint(1, 4)
    day = fleet_day(rng, cars, n, whole)
    bounds = soft_scale.forecast_bounds(day)
    centre = _t(point_of(day, rng))
    up, down = (_t(w) for w in shape(rng, n))
    scale = float(soft_scale.exact_scale(bounds, centre, up, down))
    if math.isfinite(scale) and scale > 0:
        up, down = up * scale, down * scale
    return bounds, (centre, up, down)


def _gradient(kind, bounds, field, shape_):
    x = getattr(bounds, field).clone().requires_grad_(True)
    value = check._function(kind)(replace(bounds, **{field: x}), *shape_)
    (gradient,) = torch.autograd.grad(value, x)
    return gradient


def handed(gradient, direction, error: float, mode: str) -> tuple[float, float]:
    """(the derivative handed to the rule, its true relative error)."""
    parts = gradient * direction
    true = float(parts.sum())
    if mode == "all":
        wrong = (1 + error) * true
    else:
        wrong = true + error * float(parts[int(parts.abs().argmax())])
    if wrong == true:
        return wrong, 0.0
    return wrong, abs(wrong - true) / abs(true) if true != 0 else math.inf


def _why(steps, gradient, direction) -> str:
    """Why a comparison is not decidable: `zero` (autograd 0 and the difference 0 at every step:
    the field does not move the function here), `kink` (the unchanged gradient's outcome is
    `kink`), else `rounding` (the allowance is above a quarter of the target with no kink shown at
    the end; for a derivative near 0 this is the usual case)."""
    auto = float((gradient * direction).sum())
    if auto == 0 and all(s.get("difference") == 0 for s in steps):
        return "zero"
    return "kink" if check.decide(auto, steps)[0] == "kink" else "rounding"


def _comparisons():
    """Every comparison: (size, departures, function, field, decidable, why not, [(mode, error,
    true relative error, outcome)])."""
    rng = random.Random(SEED)
    generator = torch.Generator().manual_seed(SEED)
    found = []
    for cars in SIZES:
        for whole in (False, True):
            for _ in range(FLEETS):
                bounds, shape_ = _case(rng, cars, whole)
                tied = check.smallest_terms(bounds, shape_) > 1
                for field in check.FIELDS:
                    direction = torch.randn(cars, generator=generator, dtype=torch.float64)
                    direction = direction / direction.norm()
                    for kind in check.KINDS:
                        if kind == "exact" and tied:
                            continue
                        steps = check.measure(kind, bounds, field, direction, shape_)
                        decidable = any(s.get("resolvable") and not s["kink_shown"] for s in steps)
                        gradient = _gradient(kind, bounds, field, shape_)
                        why = None if decidable else _why(steps, gradient, direction)
                        results = []
                        for mode in MODES:
                            for error in (0.0, *ERRORS):
                                wrong, true_error = handed(gradient, direction, error, mode)
                                outcome, _ = check.decide(wrong, steps)
                                results.append((mode, error, true_error, outcome))
                        departures = "whole hours" if whole else "fractional"
                        found.append((cars, departures, kind, field, decidable, why, results))
    return found


def _table(found) -> list[str]:
    cells = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    made, decidable, why = defaultdict(int), defaultdict(int), defaultdict(Counter)
    for cars, departures, kind, field, can, reason, results in found:
        key = (cars, departures, kind, field)
        made[key] += 1
        decidable[key] += can
        if reason is not None:
            why[key][reason] += 1
        for mode, error, true_error, outcome in results:
            if not can or error == 0.0:
                continue
            cell = cells[key][(mode, error)]
            if true_error > check.LIMIT:
                cell[0] += outcome == "failed"
                cell[1] += 1
            elif error == ERRORS[-1]:
                cell[0] += outcome == "passed"
                cell[1] += 1
    header = (
        "cars | departures | function | field | made | decidable | failed of decidable with "
        "true error above 1e-3, at 100%, 10%, 1%, 0.2% (all coordinates; one coordinate) | "
        "passed of decidable at 0.01% (all; one) | not decidable: zero, kink, rounding"
    )
    lines = [header]
    for key in made:
        cars, departures, kind, field = key
        share = {
            mode: [
                "{}/{}".format(*cells[key][(mode, error)]) if cells[key][(mode, error)][1] else "-"
                for error in ERRORS
            ]
            for mode in MODES
        }
        lines.append(
            f"{cars} | {departures} | {kind} | {field} | {made[key]} | {decidable[key]} | "
            f"{' '.join(share['all'][:4])}; {' '.join(share['one'][:4])} | "
            f"{share['all'][4]}; {share['one'][4]} | "
            f"{why[key]['zero']}, {why[key]['kink']}, {why[key]['rounding']}"
        )
    return lines


def test_every_decidable_wrong_gradient_fails_and_every_near_one_passes():
    found = _comparisons()
    print("\n" + "\n".join(_table(found)))
    missed, false_pass, false_fail = [], [], []
    for cars, departures, kind, field, can, _, results in found:
        for mode, error, true_error, outcome in results:
            where = (cars, departures, kind, field, mode, error, true_error, outcome)
            if true_error > check.LIMIT and outcome == "passed":
                false_pass.append(where)
            if true_error <= WITHIN and outcome == "failed":
                false_fail.append(where)
            if can and true_error > check.LIMIT and outcome != "failed":
                missed.append(where)
            if can and true_error <= WITHIN and outcome != "passed":
                missed.append(where)
    assert not false_pass, false_pass[:5]
    assert not false_fail, false_fail[:5]
    assert not missed, missed[:5]
    decidable = defaultdict(int)
    for cars, departures, kind, field, can, _, _ in found:
        decidable[(cars, departures, kind, field)] += can
    for cars in SIZES:
        for departures in ("fractional", "whole hours"):
            fields = ("cap", "need") + (("departure",) if departures == "fractional" else ())
            for kind in check.KINDS:
                for field in fields:
                    assert decidable[(cars, departures, kind, field)] > 0, (
                        cars,
                        departures,
                        kind,
                        field,
                    )
