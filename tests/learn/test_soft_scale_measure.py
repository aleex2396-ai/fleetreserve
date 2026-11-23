"""The parts of `soft_scale_check`'s measuring rule (its module text), on seeded made-up fleets
(`soft_scale_fake.fleet_day`) and on hand cases.

1. Rounding bounds against exact arithmetic: at 30 and 300 cars, with fractional and whole-hour
   departures, at the point and at the moved points `theta +- h v` of every field (`h` the exact
   scale's first step), every finite term computed in float64 is within `term_rounding`'s `e_k` of
   the same term computed in exact rational arithmetic (`fractions.Fraction`) at the exact real
   point; the exact scale is within `value_rounding`'s `e` of the exact minimum, and the soft
   minimum within its `e` of the soft minimum of the exact terms in 60-digit decimal arithmetic.
   The largest and median ratio of the true error to the bound are printed (how loose it is).
2. The allowance holds: on fractional made-up fleets, `|D - a| <= tau` for the autograd derivative
   `a` at every measured step for both functions in every field, at the first steps, at the
   smaller steps and, in cap, need and rate (where every term is concave along the line), at 10
   and 100 times the first step too, where kinks and the soft minimum's curvature are inside.
3. Concavity (rule part 1(a)): along lines in cap, need and rate at steps up to 0.5, `phi(0) >=
   (phi(t) + phi(-t)) / 2` for both functions, up to rounding; in departure a hand case is not
   concave (a share's kink at the start of an hour).
4. `judge` on known intervals: passed, failed, unresolved, a resolvable one not passed fails, an
   infinite allowance is unresolved (for a zero derivative too), 0 inside the interval.
5. `decide` and `measure` on hand fleets: a kink inside the first step but outside the smaller one
   is shown at the first step and decided at the smaller; a kink at the point is recorded `kink`;
   a non-finite derivative fails; a resolvable first step is never measured again. With the soft
   minimum's parts and value rounding set by hand: `shrinks` is `truncation + kink > rounding`,
   strictly, and the smaller step is measured only for a first step that is not resolvable and
   shrinks. A kink is recorded only when the smaller step shows one.
6. What `compare` writes: the relative error against the finite difference, `resolvable` False
   for a comparison failed far outside an interval that is not resolvable, an infinite allowance
   as null.
"""

import json
import math
import random
import statistics
from decimal import Decimal, getcontext
from fractions import Fraction

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import pytest
import torch
from soft_scale_fake import fleet_day, point_of, shape

from fleetreserve import soft_scale
from fleetreserve import soft_scale_check as check

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net

SEED = 4207
FIELD_NAMES = ("departure", "rate", "cap", "need")


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def _case(rng, cars, whole):
    n = rng.randint(1, 4)
    day = fleet_day(rng, cars, n, whole)
    bounds = soft_scale.forecast_bounds(day)
    centre = _t(point_of(day, rng))
    up, down = (_t(w) for w in shape(rng, n))
    scale = float(soft_scale.exact_scale(bounds, centre, up, down))
    if math.isfinite(scale) and scale > 0:
        up, down = up * scale, down * scale
    return bounds, (centre, up, down)


def _with(bounds, field, value) -> dict:
    return {k: value if k == field else getattr(bounds, k) for k in FIELD_NAMES}


def _fractions(values):
    return [Fraction(float(v)) for v in values]


def exact_terms(fields: dict, shape_) -> list:
    """Every term in rational arithmetic, `soft_scale.terms`' order; None for a left-out term."""
    centre, up, down = (_fractions(s) for s in shape_)
    n = len(centre)
    upper, lower = [], []
    for mask in range(2**n):
        hours = [k for k in range(n) if mask >> k & 1]
        b = p = Fraction(0)
        for d, r, cap, need in zip(*(fields[k] for k in FIELD_NAMES), strict=True):
            s = sum((min(max(d - (k + 1), Fraction(0)), Fraction(1)) for k in hours), Fraction(0))
            b += min(r * s, cap)
            p += max(need - r * (d - s), Fraction(0))
        c = sum((centre[k] for k in hours), Fraction(0))
        w_up = sum((up[k] for k in hours), Fraction(0))
        w_down = sum((down[k] for k in hours), Fraction(0))
        upper.append((b - c) / w_up if w_up > 0 else None)
        lower.append((c - p) / w_down if w_down > 0 else None)
    return upper + lower


def _soft_exact(found: list) -> Decimal:
    getcontext().prec = 60
    sigma = Decimal(check.SIGMA)
    total = sum(
        (
            (-(Decimal(t.numerator) / Decimal(t.denominator)) / sigma).exp()
            for t in found
            if t is not None
        ),
        Decimal(0),
    )
    return -sigma * total.ln()


def test_the_rounding_bounds_hold_against_exact_arithmetic():
    rng = random.Random(SEED)
    generator = torch.Generator().manual_seed(SEED)
    ratios = {"term": [], "exact": [], "soft": []}
    for cars, fleets in ((30, 6), (300, 2)):
        for whole in (False, True):
            for _ in range(fleets):
                bounds, shape_ = _case(rng, cars, whole)
                step = check.STEPS["exact"]
                for field in check.FIELDS:
                    direction = torch.randn(cars, generator=generator, dtype=torch.float64)
                    direction = direction / direction.norm()
                    base = getattr(bounds, field)
                    for t in (-step, 0.0, step):
                        moved = soft_scale.Bounds(**_with(bounds, field, base + t * direction))
                        fields = {k: _fractions(getattr(bounds, k)) for k in FIELD_NAMES}
                        fields[field] = [
                            x + Fraction(t) * Fraction(float(v))
                            for x, v in zip(fields[field], direction, strict=True)
                        ]
                        truth = exact_terms(fields, shape_)
                        found, bound = check.term_rounding(moved, shape_, field)
                        for k, exact in enumerate(truth):
                            if exact is None:
                                assert math.isinf(float(found[k]))
                                continue
                            error = abs(Fraction(float(found[k])) - exact)
                            assert error <= Fraction(float(bound[k])), (cars, field, k)
                            if bound[k] > 0:
                                ratios["term"].append(float(error) / float(bound[k]))
                        least = min(x for x in truth if x is not None)
                        for kind in check.KINDS:
                            value = float(check._function(kind)(moved, *shape_))
                            e = check.value_rounding(kind, found, bound, value)
                            if kind == "exact":
                                error = float(abs(Fraction(value) - least))
                            else:
                                error = float(abs(Decimal(value) - _soft_exact(truth)))
                            assert error <= e, (kind, cars, field, error, e)
                            ratios[kind].append(error / e)
    for name, found in ratios.items():
        largest, median = max(found), statistics.median(found)
        print(f"{name}: largest true error / bound {largest:.3g}, median {median:.3g}")


def _derivative(kind, bounds, field, direction, shape_):
    return check._autograd(check._function(kind), bounds, field, direction, shape_)[1]


@pytest.mark.parametrize("cars", (30, 300))
def test_the_allowance_holds_the_derivative(cars):
    rng = random.Random(SEED + cars)
    generator = torch.Generator().manual_seed(SEED + cars)
    worst = 0.0
    for _ in range(12):
        bounds, shape_ = _case(rng, cars, False)
        tied = check.smallest_terms(bounds, shape_) > 1
        for field in check.FIELDS:
            direction = torch.randn(cars, generator=generator, dtype=torch.float64)
            direction = direction / direction.norm()
            for kind in check.KINDS:
                if kind == "exact" and tied:
                    continue
                a = _derivative(kind, bounds, field, direction, shape_)
                first = check.STEPS[kind]
                steps = [first, first / check.SHRINK]
                if field != "departure":
                    steps += [10 * first, 100 * first]
                for step in steps:
                    found = check.measure_step(kind, bounds, field, direction, shape_, step)
                    if found["kink"] > 0 and step <= first:
                        continue  # a kink may lie at the point, where `a` is no derivative
                    gap = abs(found["difference"] - a)
                    assert gap <= found["allowance"], (kind, field, step, gap, found)
                    if found["allowance"] > 0:
                        worst = max(worst, gap / found["allowance"])
    print(f"{cars} cars: largest |D - a| / tau {worst:.3g}")


@pytest.mark.parametrize("field", ("cap", "need", "rate"))
def test_both_functions_are_concave_along_lines_in_cap_need_and_rate(field):
    rng = random.Random(SEED)
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(40):
        cars = rng.choice((3, 30))
        bounds, shape_ = _case(rng, cars, rng.random() < 0.5)
        direction = torch.randn(cars, generator=generator, dtype=torch.float64)
        direction = direction / direction.norm()
        base = getattr(bounds, field)
        for kind in check.KINDS:
            f = check._function(kind)
            for t in (1e-3, 0.05, 0.5):
                values = [
                    float(
                        f(soft_scale.Bounds(**_with(bounds, field, base + s * direction)), *shape_)
                    )
                    for s in (-t, 0.0, t)
                ]
                if not all(math.isfinite(v) for v in values):
                    continue
                slack = 1e-9 * max(1.0, *map(abs, values))
                assert values[1] >= (values[0] + values[2]) / 2 - slack, (kind, field, t, values)


def test_the_departure_direction_is_not_concave():
    """One car, one service hour, departure 1 (the start of the hour), rate 2, cap 5, need 0, centre
    0, width up 1 and none down: the only term, `b({1}) = min(2 f_1(d), 5)`, is 0 below the hour and
    rises in it, a convex kink."""
    shape_ = (_t([0.0]), _t([1.0]), _t([0.0]))

    def upper(d):
        bounds = soft_scale.Bounds(_t([d]), _t([2.0]), _t([5.0]), _t([0.0]))
        return float(soft_scale.exact_scale(bounds, *shape_))

    assert upper(1.0) < (upper(0.9) + upper(1.1)) / 2


STEP = {"step": 1e-4, "non_finite": False, "kink": 0.0, "kink_shown": False}


def _step(difference, allowance, resolvable=None):
    if resolvable is None:
        resolvable = allowance <= check.RESOLUTION * check.LIMIT * abs(difference)
    return dict(STEP, difference=difference, allowance=allowance, resolvable=resolvable)


def test_judge_on_known_intervals():
    judge = check.judge
    assert judge(1.0, _step(1.0, 1e-5)) == "passed"
    assert judge(1.0 + 2e-3, _step(1.0, 1e-5)) == "failed"
    assert judge(1.0 + 1.0005e-3, _step(1.0, 1e-5)) == "failed"  # resolvable, not shown within
    assert judge(1.0 + 1.0005e-3, _step(1.0, 1e-5, resolvable=False)) == "unresolved"
    assert judge(1.0, _step(1.0, 0.1)) == "unresolved"  # not resolvable, not shown either way
    assert judge(5.0, _step(1.0, 0.1)) == "failed"  # outside by far: shown wrong
    assert judge(0.0, _step(0.0, 1e-9)) == "unresolved"  # 0 inside: any small derivative possible
    assert judge(1.0, _step(0.0, 1e-9)) == "failed"
    assert judge(1.0, _step(1.0, math.inf)) == "unresolved"
    assert judge(0.0, _step(1.0, math.inf)) == "unresolved"  # not passed by infinite ends
    assert judge(1.0, _step(math.nan, 1e-9)) == "unresolved"
    # the ends decide: a passes only if within the limit of both ends
    assert judge(1.0, _step(1.0, 0.9e-3)) == "passed"
    assert judge(1.0, _step(1.0, 1.1e-3, resolvable=False)) == "unresolved"


def _one_car(departure, rate=2.0, cap=5.0, need=0.0):
    return soft_scale.Bounds(_t([departure]), _t([rate]), _t([cap]), _t([need]))


ONE_HOUR = (_t([1.5]), _t([1.0]), _t([1.0]))  # the upper term is the smaller one


def test_a_kink_inside_the_first_step_only_is_decided_at_the_smaller_step():
    """One car, one service hour, centre 1.5: the upper term `min(2 f_1(d), 5) - 1.5` is below the
    lower term 1.5 and has a kink where the share reaches 1 (`d = 2`). At `d = 2 - 3e-5` the kink is
    within the exact scale's first step (1e-4) and outside the smaller one (1e-6)."""
    bounds = _one_car(2.0 - 3e-5)
    direction = _t([1.0])
    steps = check.measure("exact", bounds, "departure", direction, ONE_HOUR)
    assert len(steps) == 2 and steps[0]["kink_shown"] and not steps[0]["resolvable"]
    assert steps[1]["resolvable"] and not steps[1]["kink_shown"]
    found = check.compare("exact", bounds, "departure", direction, ONE_HOUR)
    assert found["outcome"] == "passed" and found["decided_step"] == 1e-6
    assert check.decide(2.0 * 1.01, steps)[0] == "failed"


def test_a_kink_at_the_point_is_recorded_as_a_kink():
    bounds = _one_car(2.0)
    direction = _t([1.0])
    found = check.compare("exact", bounds, "departure", direction, ONE_HOUR)
    assert found["outcome"] == "kink" and found["decided_step"] is None
    assert not found["resolvable"] and len(found["steps"]) == 2
    soft = check.compare("soft", bounds, "departure", direction, ONE_HOUR)
    assert soft["outcome"] == "kink"


def test_a_resolvable_first_step_is_not_measured_again_and_a_nan_fails():
    bounds = _one_car(1.5)
    direction = _t([1.0])
    steps = check.measure("exact", bounds, "departure", direction, ONE_HOUR)
    assert len(steps) == 1 and steps[0]["resolvable"]
    assert check.decide(2.0, steps) == ("passed", 0)
    assert check.decide(math.nan, steps) == ("failed", None)
    assert check.decide(2.0, [dict(steps[0], non_finite=True)]) == ("failed", None)


def test_the_smaller_step_needs_an_unresolved_first_step_and_parts_above_the_rounding(
    monkeypatch,
):
    """The soft minimum on the one-car day, its truncation and kink parts and its value rounding
    `e` set by hand: `shrinks` is `truncation + kink > rounding` (strictly), and the smaller step
    is measured only when the first is not resolvable and shrinks."""
    bounds, direction = _one_car(1.5), _t([1.0])

    def steps_with(truncation, kink, e, hidden=0.0):
        monkeypatch.setattr(
            check, "_soft_parts", lambda points, step, reach: (truncation, kink, hidden)
        )
        monkeypatch.setattr(check, "value_rounding", lambda *args: e)
        return check.measure("soft", bounds, "departure", direction, ONE_HOUR)

    first = steps_with(0.0, 0.0, 1e-20)[0]
    r = first["rounding"]
    want = 1e-20 / 1e-5 + 2 * check.EPSILON * abs(first["difference"])
    assert r == pytest.approx(want, rel=1e-12, abs=0)
    assert first["resolvable"] and not first["shrinks"]
    # the part for kinks too small to show (rule, part 2) enters the rounding part
    hidden = steps_with(0.0, 0.0, 1e-20, hidden=3e-10)[0]["rounding"]
    assert hidden == pytest.approx(want + 3e-10, rel=1e-12, abs=0)
    cases = [
        # (truncation, kink, e): shrinks, steps measured
        ((2 * r, 0.0, 1e-20), True, 1),  # truncation alone above the rounding; resolvable
        ((r, 2 * r, 1e-20), True, 2),  # together 3 times the rounding; a kink shown
        ((r, 0.0, 1e-20), False, 1),  # equal to the rounding: not above it
        ((0.0, 0.5 * r, 1e-20), False, 2),  # a kink shown below the rounding: still smaller
        ((0.5, 0.5, 1e-20), True, 2),  # not resolvable and shrinks: the smaller step
        ((0.5, 0.0, 1e-20), True, 2),  # not resolvable and shrinks without a kink
        ((0.0, 0.0, 1e-7), False, 1),  # not resolvable (rounding 1e-2) and does not shrink
    ]
    for (truncation, kink, e), shrinks, measured in cases:
        steps = steps_with(truncation, kink, e)
        assert steps[0]["shrinks"] == shrinks, (truncation, kink, e)
        assert len(steps) == measured, (truncation, kink, e)
        assert steps[-1]["step"] == (1e-5 if measured == 1 else 1e-5 / check.SHRINK)


def test_a_kink_is_recorded_only_when_both_steps_show_one():
    wide = dict(STEP, difference=1.0, allowance=1.0, resolvable=False)  # 1.0: unresolved
    kinked = dict(wide, kink=0.5, kink_shown=True)
    assert check.judge(1.0, wide) == "unresolved"
    assert check.decide(1.0, [kinked, wide]) == ("unresolved", None)  # gone at the smaller step
    assert check.decide(1.0, [kinked]) == ("unresolved", None)  # a first step alone: no kink
    assert check.decide(1.0, [wide, kinked]) == ("unresolved", None)  # the first shows none
    assert check.decide(1.0, [kinked, kinked]) == ("kink", None)


def test_a_step_that_shows_a_kink_decides_nothing():
    """The owner's decision of 15 November 2026: a comparison that shows a kink at both step sizes
    is neither passed nor failed. A step whose interval would fail (or pass) `a`, resolvable or
    not, is `unresolved` when it shows a kink; a later step without one decides."""
    far = dict(STEP, difference=1.0, allowance=1e-6, resolvable=True)
    kinked = dict(far, kink=1e-7, kink_shown=True)
    assert check.judge(5.0, far) == "failed" and check.judge(1.0, far) == "passed"
    assert check.judge(5.0, kinked) == check.judge(1.0, kinked) == "unresolved"
    assert check.decide(5.0, [kinked, kinked]) == ("kink", None)
    assert check.decide(1.0, [kinked, kinked]) == ("kink", None)
    assert check.decide(5.0, [kinked, far]) == ("failed", 1)
    assert check.decide(1.0, [kinked, far]) == ("passed", 1)


def test_coinciding_kinks_at_the_point_are_recorded_as_a_kink_not_failed():
    """Finding 1 of the review of 2078c32. Two cars, one service hour, centre 0.5, width up 1 and
    none down (one term, `b({1}) - 0.5`): an empty car (rate 3, cap 0) leaving at 1, the start of
    the hour, and a car (rate 2, cap 50) leaving at 2, its end, both moved by the direction
    (0.9, 0.436) over its norm. Along it the term rises with slope `2 v_2` before 0 and is flat
    after (the empty car adds 0 on both sides), so there is no derivative at the point; autograd
    gives `3 v_1 / 4 + v_2` (the empty car's two halved kinks, the mean for the other car),
    outside the one-sided range. The kink is shown at both steps: recorded `kink`, not failed."""
    bounds = soft_scale.Bounds(_t([1.0, 2.0]), _t([3.0, 2.0]), _t([0.0, 50.0]), _t([0.0, 0.0]))
    shape_ = (_t([0.5]), _t([1.0]), _t([0.0]))
    direction = _t([0.9, 0.436])
    direction = direction / direction.norm()
    v1, v2 = direction.tolist()
    for kind in check.KINDS:
        found = check.compare(kind, bounds, "departure", direction, shape_)
        assert found["autograd"] == pytest.approx(0.75 * v1 + v2, rel=1e-12, abs=0)
        assert found["autograd"] > 2 * v2 * (1 + 1e-3)  # outside the one-sided range [0, 2 v2]
        assert len(found["steps"]) == 2 and all(s["kink_shown"] for s in found["steps"])
        for s in found["steps"]:
            assert s["left"] == pytest.approx(2 * v2, rel=1e-6) and abs(s["right"]) < 1e-6
        assert found["outcome"] == "kink" and found["decided_step"] is None, (kind, found)


def _two_kinks(a: float, second_rate: float = 2.0, third_car: bool = False):
    """Finding 2 of the review of 2078c32 (its cases C1 and C1b): one service hour, centre 0.5,
    width up 1 and none down (one term, `b({1}) - 0.5`). Along the direction a car (rate 2) that
    leaves just before hour 1 enters it at `t = a` (a convex kink) and a car that leaves just after
    hour 1 leaves it at `t = -a` (a concave kink): kinks of opposite convexity on both sides of the
    point, both inside the first step when `a < h`. With rates 2 and 2 their effects cancel in the
    spread `q`. The derivative at the point is 0, or with `third_car` (rate 3, leaving at 1.5,
    inside the hour) `3 v_3`."""
    v = [0.5, 0.5, 2**-0.5] if third_car else [2**-0.5, 2**-0.5]
    departures = [1.0 - a * v[0], 2.0 + a * v[1]] + ([1.5] if third_car else [])
    rates = [2.0, second_rate] + ([3.0] if third_car else [])
    cars = len(departures)
    bounds = soft_scale.Bounds(_t(departures), _t(rates), _t([50.0] * cars), _t([0.0] * cars))
    true = 3.0 * v[2] if third_car else 0.0
    return bounds, _t(v), (_t([0.5]), _t([1.0]), _t([0.0])), true


@pytest.mark.parametrize("kind", check.KINDS)
@pytest.mark.parametrize("share", (0.5, 0.1))
@pytest.mark.parametrize("second_rate", (2.0, 2.0005))
def test_kinks_of_opposite_convexity_on_both_sides_never_pass_a_wrong_gradient(
    kind, share, second_rate
):
    """Before the fix the first step showed no kink in `q` and was resolvable: the right gradient
    0 failed and a gradient equal to `D` (about 0.71, infinitely wrong) passed. The second
    differences over each half of the step show the kinks; the step decides nothing and the
    smaller step, which holds no kink, decides."""
    a = share * check.STEPS[kind]
    bounds, direction, shape_, true = _two_kinks(a, second_rate)
    found = check.compare(kind, bounds, "departure", direction, shape_)
    assert found["autograd"] == true == 0.0
    steps = check.measure(kind, bounds, "departure", direction, shape_)
    assert steps[0]["kink_shown"] and abs(steps[0]["difference"] - true) > 0.5
    assert len(steps) == 2 and not steps[1]["kink_shown"]
    assert found["outcome"] != "failed", found
    assert check.decide(steps[0]["difference"], steps)[0] == "failed"
    assert check.decide(1.0, steps)[0] == "failed"


@pytest.mark.parametrize("kind", check.KINDS)
def test_kinks_on_both_sides_with_a_derivative_that_is_not_zero(kind):
    a = 0.5 * check.STEPS[kind]
    bounds, direction, shape_, true = _two_kinks(a, third_car=True)
    found = check.compare(kind, bounds, "departure", direction, shape_)
    assert found["autograd"] == pytest.approx(true, rel=1e-12, abs=0)
    assert found["outcome"] == "passed" and found["decided_step"] == check.STEPS[kind] / 100
    steps = check.measure(kind, bounds, "departure", direction, shape_)
    assert steps[0]["kink_shown"]
    assert abs(steps[0]["difference"] - true) > 0.4  # D of the first step is off by 0.5
    for wrong in (0.01, 0.1, 0.2, 0.3):
        assert check.decide(true * (1 + wrong), steps)[0] == "failed", wrong


def test_a_kink_too_small_to_show_is_inside_the_soft_allowance():
    """Finding 3 of the review of 2078c32 (its case C2). One service hour, centre 0.3, width up 1
    and none down: the soft minimum's only term is `b({1}) - 0.3`, with `b({1}) = min(1.4, cap_1)
    + 0.4` for a car (rate 2) leaving at 1.7 and one (rate 1) leaving at 1.4. Moved in the first
    car's cap, the term has slope 1 at the point and a kink where the cap reaches 1.4, placed at
    `t0 = h (1 - x)` inside the soft minimum's first step `h` for `x` from 1e-12 to 1e-6; for the
    smaller `x` its second differences lie within four times their rounding bounds, so it does not
    show. Every measured step's allowance holds the derivative 1; a gradient wrong by just above
    the target never passes and the right one never fails. Before the fix `|D - 1| / tau` reached
    1.47 and gradients wrong by 1.0000001e-3 passed."""
    h = check.STEPS["soft"]
    shape_ = (_t([0.3]), _t([1.0]), _t([0.0]))
    direction = _t([1.0, 0.0])
    worst, hidden = 0.0, 0
    for j in range(600):
        x = 10 ** (-12 + 6 * j / 600)
        cap = 2.0 * 0.7 - h * (1 - x)
        bounds = soft_scale.Bounds(_t([1.7, 1.4]), _t([2.0, 1.0]), _t([cap, 5.0]), _t([0.0, 0.0]))
        steps = check.measure("soft", bounds, "cap", direction, shape_)
        for step in steps:
            gap = abs(step["difference"] - 1.0)
            assert gap <= step["allowance"], (x, step)
            worst = max(worst, gap / step["allowance"])
        hidden += not steps[0]["kink_shown"]  # the kink is inside every first step
        assert check.decide(1.0, steps)[0] != "failed", x
        for wrong in (1.0000001e-3, -1.0000001e-3):
            assert check.decide(1.0 + wrong, steps)[0] != "passed", (x, wrong)
    print(
        f"largest |D - 1| / tau {worst:.3g}; first steps with a kink that does not show: {hidden}"
    )
    assert hidden > 0


def test_compare_writes_the_error_against_the_difference_and_the_decided_steps_resolvable(
    monkeypatch,
):
    bounds, direction = _one_car(1.5), _t([1.0])
    difference = check.measure("soft", bounds, "departure", direction, ONE_HOUR)[0]["difference"]
    a = difference * (1 + 4e-4)
    monkeypatch.setattr(check, "_autograd", lambda *args: (0.0, a))
    found = check.compare("soft", bounds, "departure", direction, ONE_HOUR)
    assert found["outcome"] == "passed" and found["resolvable"]
    assert found["relative_error"] == pytest.approx(abs(a - difference) / abs(difference), 1e-12)
    # the other direction: a negative difference has its relative error too
    back = check.compare("soft", bounds, "departure", -direction, ONE_HOUR)
    assert back["outcome"] == "failed" and back["relative_error"] == pytest.approx(2 + 4e-4)
    # an allowance of 0.5 (not resolvable at D = 2) and autograd 10 D: failed far outside
    monkeypatch.setattr(check, "_soft_parts", lambda points, step, reach: (0.5, 0.0, 0.0))
    monkeypatch.setattr(check, "_autograd", lambda *args: (0.0, 10 * difference))
    found = check.compare("soft", bounds, "departure", direction, ONE_HOUR)
    assert found["outcome"] == "failed" and found["decided_step"] == check.STEPS["soft"]
    assert found["resolvable"] is False
    # an infinite allowance is written as null in every measured step
    monkeypatch.setattr(check, "_soft_parts", lambda points, step, reach: (math.inf, 0.0, 0.0))
    found = check.compare("soft", bounds, "departure", direction, ONE_HOUR)
    assert found["outcome"] == "unresolved" and len(found["steps"]) == 2
    assert all(s["allowance"] is None and s["truncation"] is None for s in found["steps"])
    json.dumps(found, allow_nan=False)


def test_resolvable_at_a_quarter_of_the_target():
    assert check.resolvable(2.0, 0.25 * 1e-3 * 2.0)
    assert not check.resolvable(2.0, 0.2501 * 1e-3 * 2.0)
    assert check.resolvable(-2.0, 5e-4) and not check.resolvable(0.0, 0.0 + 1e-300)
    # the guarantee at the edge (rule, part 3): an error of 0.499 of the target passes, one just
    # above the target fails
    for sign in (1.0, -1.0):
        d = 3.0 * sign
        tau = 0.25 * 1e-3 * abs(d)
        step = _step(d, tau)
        assert step["resolvable"]
        for g in (d - tau, d + tau):
            assert check.judge(g * (1 + 0.499e-3), step) == "passed"
            assert check.judge(g * (1 + 1.0001e-3), step) == "failed"
            assert check.judge(g * (1 - 1.0001e-3), step) == "failed"


def test_a_written_comparison_holds_no_infinity():
    assert check._written(math.inf) is None and check._written(math.nan) is None
    assert check._written(1.5) == 1.5 and check._written(True) is True
