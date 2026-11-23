"""The allowance of `soft_scale_check`'s measuring rule (its module text, parts 1 and 2) computed a
second time, in plain Python from the rule's formulas, and compared with the module's own.

The rounding bounds of the terms (`term_rounding`) and of one value (`value_rounding`) are pinned
by hand values in `test_soft_scale_check.py`; here they are inputs. With `h` the step, the five
points `-h, -h/2, 0, h/2, h`, `e` the largest `value_rounding` over the five, `D`, `L` and `R` the
central, left and right differences, and for any five values `x` the three second differences
`x(h) - 2 x(0) + x(-h)`, `x(0) - 2 x(-h/2) + x(-h)` and `x(h) - 2 x(h/2) + x(0)`:

- exact scale: rounding `5 e / h + 2 eps |D|`, truncation 0, kink `max(S - 4 e, 0) / (2 h)` with
  `S` the largest absolute second difference of the five values;
- soft minimum: with `r_k` the largest of a term's five bounds and `s_k` its largest absolute
  second difference, the kinked terms are those with `s_k > 4 r_k` and the straight terms the
  others; over the straight terms, with weights `pi_k` (the softmax of `-T_k / sigma` over them
  at the point), slopes `g_k = (T_k(+h) - T_k(-h)) / (2 h)`, deviations `d_k = |g_k - sum_j pi_j
  g_j|` and `pihat_k = pi_k exp(h d_k / sigma)`: truncation `h^2 / 6 * 8 sum_k pihat_k d_k^3 /
  sigma^2` and hidden `4 sum_k pihat_k r_k / h`; for a kinked term, `rho_k(t) = exp(-T_k(t) /
  sigma) / sum_{j straight} exp(-T_j(t) / sigma)` times `exp((r_k + max_{j straight} r_j) /
  sigma)`, and kink `sigma sum_k (rho_k(-h) + rho_k(h)) / (2 h) + sum_k rho_k(0) (Lambda_k +
  max_j Lambda_j)` over the kinked terms (`Lambda` the slope bounds, `max_j` over the finite
  terms); infinite when no term is straight; rounding `e / h + hidden + 2 eps |D|`;
- allowance the sum of the three; resolvable at `allowance <= RESOLUTION LIMIT |D|`; the kink
  shown when the kink part is above 0; the smaller step `h / SHRINK` measured when the first step
  is finite and either shows a kink or is not resolvable with its truncation and kink parts above
  its rounding part.

1. Hand-made points (three finite terms and one left out, bounds unequal across the five points,
   one term with `4 r < s < 40 r` over the whole step): the soft minimum's truncation, kink and
   hidden parts, also worked out by hand; a term kinked over one half of the step only is counted
   as kinked.
2. Seeded made-up fleets of 6 cars (fractional and whole-hour departures) and one car whose two
   terms nearly tie with different slopes in cap, both functions, every field: every quantity
   `measure_step` returns, and the steps `measure` takes, equal the second computation; a kink
   part above 0, a soft kink part above 0, a truncation part above the rounding part and a
   smaller step are each seen at least once. (On the seeded fleets alone the only truncation part
   above the rounding part came from a kinked term's slope, which the soft minimum's truncation
   no longer takes in: the near tie is there to keep the truncation part exercised.)
"""

import math
import random

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import pytest
import torch
from soft_scale_fake import fleet_day, point_of, shape

from fleetreserve import soft_scale
from fleetreserve import soft_scale_check as check

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net

SEED = 7741
DAYS = 10
CARS = 6


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def _soft_parts(points, step, reach):
    """The soft minimum's (truncation, kink, hidden) parts, in plain Python (module text)."""
    sigma = check.SIGMA
    keep = [k for k, value in enumerate(points[2][1].tolist()) if math.isfinite(value)]
    behind, half_behind, middle, half_ahead, ahead = (
        [p[1].tolist()[k] for k in keep] for p in points
    )
    largest = [max(p[2].tolist()[k] for p in points) for k in keep]
    seconds = [
        max(abs(a - 2 * m + b), abs(m - 2 * hb + b), abs(a - 2 * ha + m))
        for b, hb, m, ha, a in zip(behind, half_behind, middle, half_ahead, ahead, strict=True)
    ]
    bent = [k for k in range(len(keep)) if seconds[k] > 4 * largest[k]]
    straight = [k for k in range(len(keep)) if k not in bent]
    if not straight:
        return 0.0, math.inf, 0.0
    low = min(middle[k] for k in straight)
    raw = [math.exp(-(middle[k] - low) / sigma) for k in straight]
    weights = [w / sum(raw) for w in raw]
    slopes = [(ahead[k] - behind[k]) / (2 * step) for k in straight]
    mean = sum(w * g for w, g in zip(weights, slopes, strict=True))
    deviations = [abs(g - mean) for g in slopes]
    bounded = [w * math.exp(step * d / sigma) for w, d in zip(weights, deviations, strict=True)]
    third = 8 * sum(p * d**3 for p, d in zip(bounded, deviations, strict=True)) / sigma**2
    hidden = 4 * sum(p * largest[k] for p, k in zip(bounded, straight, strict=True)) / step
    lift = max(largest[k] for k in straight)

    def relative(values, k):
        least = min(values[j] for j in straight)
        total = sum(math.exp(-(values[j] - least) / sigma) for j in straight)
        power = -(values[k] - least) / sigma - math.log(total) + (largest[k] + lift) / sigma
        return math.exp(power) if power < 700 else math.inf

    reaches = [reach.tolist()[k] for k in keep]
    ends = sum(relative(behind, k) + relative(ahead, k) for k in bent)
    at_point = sum(relative(middle, k) * (reaches[k] + max(reaches)) for k in bent)
    shift = sigma * ends / (2 * step) + at_point
    return step**2 * third / 6, (math.inf if math.isnan(shift) else shift), hidden


def _rule(kind, bounds, field, direction, shape_, step):
    """Every quantity of one measured step, computed a second time from the rule."""
    base = getattr(bounds, field)
    points = []
    for t in (-step, -step / 2, 0.0, step / 2, step):
        moved = soft_scale.Bounds(
            **{
                k: (base + t * direction if k == field else getattr(bounds, k))
                for k in ("departure", "rate", "cap", "need")
            }
        )
        if kind == "soft":
            value = float(soft_scale.soft_scale(moved, *shape_, check.SIGMA))
        else:
            value = float(soft_scale.exact_scale(moved, *shape_))
        found, bound = check.term_rounding(moved, shape_, field)
        points.append((value, found, bound))
    behind, half_behind, middle, half_ahead, ahead = (p[0] for p in points)
    e = max(check.value_rounding(kind, found, bound, v) for v, found, bound in points)
    difference = (ahead - behind) / (2 * step)
    left, right = (middle - behind) / step, (ahead - middle) / step
    if kind == "exact":
        rounding = 5 * e / step + 2 * check.EPSILON * abs(difference)
        largest = max(
            abs(ahead - 2 * middle + behind),
            abs(middle - 2 * half_behind + behind),
            abs(ahead - 2 * half_ahead + middle),
        )
        truncation, kink = 0.0, max(largest - 4 * e, 0.0) / (2 * step)
    else:
        reach = _slope_bounds(bounds, shape_, field, direction)
        truncation, kink, hidden = _soft_parts(points, step, reach)
        rounding = e / step + hidden + 2 * check.EPSILON * abs(difference)
    allowance = rounding + truncation + kink
    return {
        "difference": difference,
        "left": left,
        "right": right,
        "rounding": rounding,
        "truncation": truncation,
        "kink": kink,
        "allowance": allowance,
        "resolvable": allowance <= check.RESOLUTION * check.LIMIT * abs(difference),
        "kink_shown": kink > 0,
        "shrinks": truncation + kink > rounding,
    }


def test_the_soft_parts_on_hand_made_points():
    h, sigma = 1e-5, check.SIGMA
    inf = math.inf
    # term 0: 0.5 + t, slope 1, straight; term 1: 0.51 + 2 t with a second difference of 1e-12
    # over the whole step and its right half, between 4 and 40 times its largest bound 1e-13
    # (kinked); term 2: 0.505 + 3 t, straight; term 3 is left out (a zero width). The five points
    # are -h, -h/2, 0, h/2, h; the slope bounds 1, 2, 3 and 100 (term 3's is not used).
    offsets = (-1.0, -0.5, 0.0, 0.5, 1.0)
    extra = (0.0, 0.0, 0.0, 0.0, 1e-12)
    terms = [
        _t([0.5 + t * h, 0.51 + 2 * t * h + x, 0.505 + 3 * t * h, inf])
        for t, x in zip(offsets, extra, strict=True)
    ]
    sizes = (1e-13, 0.7e-13, 0.5e-13, 0.3e-13, 0.2e-13)
    bounds = [_t([1e-15, size, 2e-15, 0.0]) for size in sizes]
    reach = _t([1.0, 2.0, 3.0, 100.0])
    points = [(0.0, found, bound) for found, bound in zip(terms, bounds, strict=True)]
    truncation, kink, hidden = check._soft_parts(points, h, reach)
    want_truncation, want_kink, want_hidden = _soft_parts(points, h, reach)
    assert want_kink > 0 and want_truncation > 0 and want_hidden > 0
    assert truncation == pytest.approx(want_truncation, rel=1e-9, abs=0)
    assert kink == pytest.approx(want_kink, rel=1e-9, abs=0)
    assert hidden == pytest.approx(want_hidden, rel=1e-9, abs=0)
    # by hand. Term 1 is kinked: its second differences
    second = max(
        abs((0.51 + 2 * h + 1e-12) - 2 * 0.51 + (0.51 - 2 * h)),  # the whole step
        abs((0.51 + 2 * h + 1e-12) - 2 * (0.51 + h) + 0.51),  # its right half
    )
    assert 4e-13 < second < 4e-12
    # the straight terms 0 and 2 at the point, 0.5 and 0.505 (sigma 0.01): weights 1 and
    # exp(-0.5) over 1 + exp(-0.5); slopes 1 and 3
    weight = math.exp(-0.5) / (1 + math.exp(-0.5))
    mean = (1 - weight) * 1.0 + weight * 3.0
    deviation = [abs(1.0 - mean), abs(3.0 - mean)]
    pihat = [
        (1 - weight) * math.exp(h * deviation[0] / sigma),
        weight * math.exp(h * deviation[1] / sigma),
    ]
    third = 8 * (pihat[0] * deviation[0] ** 3 + pihat[1] * deviation[1] ** 3) / sigma**2
    assert truncation == pytest.approx(h**2 * third / 6, rel=1e-6)
    assert hidden == pytest.approx(4 * (pihat[0] * 1e-15 + pihat[1] * 2e-15) / h, rel=1e-6)

    def relative(t, bump):  # term 1's weight against terms 0 and 2 at t h, before the lift
        others = math.exp(-(0.5 + t * h) / sigma) + math.exp(-(0.505 + 3 * t * h) / sigma)
        return math.exp(-(0.51 + 2 * t * h + bump) / sigma) / others

    lift = math.exp((1e-13 + 2e-15) / sigma)  # term 1's largest bound and the straight ones'
    ends = relative(-1.0, 0.0) + relative(1.0, 1e-12)
    want = lift * (sigma * ends / (2 * h) + relative(0.0, 0.0) * (2.0 + 3.0))
    assert kink == pytest.approx(want, rel=1e-6)
    # a term kinked over the left half only (its value at -h/2 raised by 0.6e-12): the second
    # difference over the whole step is 0 and over the left half 1.2e-12; it is counted as kinked
    bumped = [_t(found.tolist()) for found in terms]
    bumped[4] = _t([0.5 + h, 0.51 + 2 * h, 0.505 + 3 * h, inf])
    bumped[1] = _t([0.5 - h / 2, 0.51 - h + 0.6e-12, 0.505 - 1.5 * h, inf])
    points = [(0.0, found, bound) for found, bound in zip(bumped, bounds, strict=True)]
    _, kink, _ = check._soft_parts(points, h, reach)
    assert kink == pytest.approx(_soft_parts(points, h, reach)[1], rel=1e-9, abs=0)
    half = abs((0.51 + 0.0) - 2 * (0.51 - h + 0.6e-12) + (0.51 - 2 * h))
    assert 4e-13 < half < 4e-12
    ends = relative(-1.0, 0.0) + relative(1.0, 0.0)
    want = lift * (sigma * ends / (2 * h) + relative(0.0, 0.0) * (2.0 + 3.0))
    assert kink == pytest.approx(want, rel=1e-6)


def _slope_bounds(bounds, shape_, field, direction):
    """`Lambda_k = sum_i |v_i| L_i / w(A_k)`, in plain Python, for the upper terms of every subset
    then the lower ones (the sum itself for a term left out)."""
    n = shape_[0].numel()
    reach = {
        "departure": [abs(r) for r in bounds.rate.tolist()],
        "rate": [abs(d) + n for d in bounds.departure.tolist()],
    }.get(field, [1.0] * bounds.rate.numel())
    total = sum(abs(v) * r for v, r in zip(direction.tolist(), reach, strict=True))
    out = []
    for widths in (shape_[1].tolist(), shape_[2].tolist()):
        for mask in range(2**n):
            width = sum(w for k, w in enumerate(widths) if mask >> k & 1)
            out.append(total / width if width > 0 else total)
    return _t(out)


def _cases():
    rng = random.Random(SEED)
    for whole in (False, True):
        for _ in range(DAYS):
            n = rng.randint(1, 4)
            day = fleet_day(rng, CARS, n, whole)
            bounds = soft_scale.forecast_bounds(day)
            centre = _t(point_of(day, rng))
            up, down = (_t(w) for w in shape(rng, n))
            for field in check.FIELDS:
                raw = [rng.gauss(0.0, 1.0) for _ in range(CARS)]
                norm = math.sqrt(sum(x * x for x in raw))
                yield bounds, (centre, up, down), field, _t([x / norm for x in raw])
    # one car, one service hour: b({1}) = 1 (the cap binds) and p({1}) = 0, centre 0.501 and
    # widths 0.5, so the upper and lower terms 0.998 and 1.002 lie within sigma of each other
    # with slopes 2 and 0 in cap: the soft minimum's curvature within the step
    tight = soft_scale.Bounds(_t([2.5]), _t([2.0]), _t([1.0]), _t([0.5]))
    for field in check.FIELDS:
        yield tight, (_t([0.501]), _t([0.5]), _t([0.5])), field, _t([1.0])


def test_every_measured_step_follows_the_rule():
    seen = {"kink": 0, "soft kink": 0, "truncation above rounding": 0, "smaller step": 0}
    compared = 0
    for bounds, shape_, field, direction in _cases():
        for kind in check.KINDS:
            steps = check.measure(kind, bounds, field, direction, shape_)
            if steps[0]["non_finite"]:
                continue
            first = steps[0]
            smaller = first["kink_shown"] or (not first["resolvable"] and first["shrinks"])
            assert len(steps) == (2 if smaller else 1)
            for index, step in enumerate(steps):
                h = check.STEPS[kind] / (check.SHRINK if index else 1.0)
                assert step["step"] == h
                want = _rule(kind, bounds, field, direction, shape_, h)
                scale = want["rounding"]
                for key in ("difference", "left", "right", "rounding"):
                    assert step[key] == pytest.approx(want[key], rel=1e-12, abs=0), key
                for key in ("truncation", "kink", "allowance"):
                    assert step[key] == pytest.approx(want[key], rel=1e-9, abs=1e-9 * scale), key
                for key in ("resolvable", "kink_shown", "shrinks"):
                    assert step[key] == want[key], key
                compared += 1
                seen["kink"] += want["kink"] > 0
                seen["soft kink"] += kind == "soft" and want["kink"] > 0
                seen["truncation above rounding"] += want["truncation"] > want["rounding"]
            seen["smaller step"] += len(steps) == 2
    print(f"{compared} measured steps follow the rule; seen: {seen}")
    assert all(count > 0 for count in seen.values()), seen
