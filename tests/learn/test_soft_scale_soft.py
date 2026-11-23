"""Plan section 7, T8: the soft minimum's sandwich, the count of terms, and how each term depends on
the bounds, on seeded made-up forecast days (`soft_scale_fake.forecast_day`, fractional
departures).

1. Sandwich: `soft <= hard <= soft + sigma ln N` at sigma 0.001, 0.01, 0.1 and 1, N the number of
   finite terms; on a day whose two terms are equal the gap is exactly `sigma ln 2`.
2. Count: `term_count` is the number of non-empty subsets with a positive width up plus those with
   a positive width down, at most `2^(|S|+1)`; `terms` holds exactly that many finite entries.
3. Piecewise linearity along random lines, on days with no car on a kink by construction
   (`generic=True`: a tight need puts a car's `p` on its kink), every term at once (on most lines
   some term moves), second differences at step 1e-4
   (a second difference above 1e-11 counts as non-zero): on lines that move departures, caps and
   needs together (rates fixed) and on lines that move rates, caps and needs together (departures
   fixed), every non-zero second difference vanishes at step 1e-7, so it came from a kink inside
   the larger step.
4. Recorded finding: on a line that moves a car's rate and its departure together a term can be
   quadratic (a product of the two), so "piecewise linear in the bounds" holds in each of the two
   groups of directions and not jointly. Smallest case: one car, one service hour, departure 1.5,
   rate 2, cap and need 2; along rate `2 + t`, departure `1.5 + t` the upper term's second
   difference at step 0.1 is `2 * 0.1**2`.
5. Where the product enters (`soft_scale`'s module text), on the same line at step 0.1, one car,
   one service hour, subset {1}: with departure 1.5, rate 2, cap 3, need 2.5, `p = 0.5 > 0` but
   the departure is inside the service hour of the subset, so `p`'s second difference is 0 while
   `b`'s is `2 * 0.1**2` (`b` below the cap); with departure 2.5 (inside the recovery hour), rate
   2, cap 10, need 5, `p = 2 > 0` and its second difference is `-2 * 0.1**2`, while `b` is linear
   (the departure is outside the subset's hour).
"""

import math
import random

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import pytest
import torch
from soft_scale_fake import forecast_day, point_of, shape

from fleetreserve import soft_scale

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net

SEED = 91
N_DAYS = 200
SIGMAS = (0.001, 0.01, 0.1, 1.0)
N_LINES = 300


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def _case(rng, **kwargs):
    day = forecast_day(rng, **kwargs)
    up, down = shape(rng, day.n_service)
    return day, soft_scale.forecast_bounds(day), _t(point_of(day, rng)), _t(up), _t(down)


def test_the_sandwich_holds_at_every_sigma():
    rng = random.Random(SEED)
    widest = {s: 0.0 for s in SIGMAS}
    for _ in range(N_DAYS):
        _, bounds, centre, up, down = _case(rng)
        hard = float(soft_scale.exact_scale(bounds, centre, up, down))
        count = int(soft_scale.term_count(up, down))
        assert count >= 1
        for sigma in SIGMAS:
            soft = float(soft_scale.soft_scale(bounds, centre, up, down, sigma))
            slack = 1e-12 * max(1.0, abs(hard))
            assert soft <= hard + slack
            assert hard <= soft + sigma * math.log(count) + slack
            widest[sigma] = max(widest[sigma], (hard - soft) / (sigma * math.log(count) or 1))
    print(f"sandwich: largest gap as a share of sigma ln N {widest}")


def test_two_equal_terms_give_the_whole_gap():
    bounds = soft_scale.Bounds(_t([2.0]), _t([2.0]), _t([1.0]), _t([0.0]))
    # One service hour; the car draws 0 to 1 kWh in it (departure 2, so share 1; rate 2, cap 1,
    # need 0: p = 0 and b = min(2, 1) = 1). Centre 0.5, widths 1: both terms are 0.5.
    centre, up, down = _t([0.5]), _t([1.0]), _t([1.0])
    hard = float(soft_scale.exact_scale(bounds, centre, up, down))
    assert hard == pytest.approx(0.5, abs=1e-15)
    assert int(soft_scale.term_count(up, down)) == 2
    for sigma in SIGMAS:
        soft = float(soft_scale.soft_scale(bounds, centre, up, down, sigma))
        assert hard - soft == pytest.approx(sigma * math.log(2), rel=1e-12)


def test_the_count_of_terms():
    rng = random.Random(SEED + 1)
    for _ in range(N_DAYS):
        day, bounds, centre, up, down = _case(rng)
        n = day.n_service
        bits = soft_scale.subset_bits(n)
        expected = int(((bits @ up) > 0).sum() + ((bits @ down) > 0).sum())
        count = int(soft_scale.term_count(up, down))
        assert count == expected <= 2 ** (n + 1)
        found = soft_scale.terms(bounds, centre, up, down)
        assert found.shape[-1] == 2 ** (n + 1)
        assert int(torch.isfinite(found).sum()) == count


def _line(bounds, move):
    """Random unit direction over the fields named in `move`; others fixed."""
    parts = {}
    for name in ("departure", "rate", "cap", "need"):
        value = getattr(bounds, name)
        parts[name] = torch.randn_like(value) if name in move else torch.zeros_like(value)
    norm = math.sqrt(sum(float((v**2).sum()) for v in parts.values()))
    return {k: v / norm for k, v in parts.items()}


def _at(bounds, line, step):
    return soft_scale.Bounds(
        *(getattr(bounds, k) + step * line[k] for k in ("departure", "rate", "cap", "need"))
    )


def _second_difference(bounds, line, centre, up, down, step):
    values = [soft_scale.terms(_at(bounds, line, s), centre, up, down) for s in (-step, 0.0, step)]
    finite = torch.isfinite(values[1])
    return (values[0] - 2 * values[1] + values[2])[finite]


@pytest.mark.parametrize("move", [("departure", "cap", "need"), ("rate", "cap", "need")])
def test_each_term_is_piecewise_linear_along_lines_that_fix_rate_or_departure(move):
    rng = random.Random(SEED + 2)
    torch.manual_seed(SEED + 2)
    nonzero = moving = 0
    for _ in range(N_LINES):
        _, bounds, centre, up, down = _case(rng, generic=True)
        line = _line(bounds, move)
        ahead = soft_scale.terms(_at(bounds, line, 1e-4), centre, up, down)
        here = soft_scale.terms(bounds, centre, up, down)
        finite = torch.isfinite(here)
        moving += float((ahead - here)[finite].abs().max()) > 1e-9
        found = _second_difference(bounds, line, centre, up, down, 1e-4).abs()
        if float(found.max()) > 1e-11:
            nonzero += 1
            again = _second_difference(bounds, line, centre, up, down, 1e-7).abs()
            assert float(again.max()) <= 1e-11, (move, bounds, line)
    print(f"{move}: {nonzero} of {N_LINES} lines had a kink inside step 1e-4; {moving} moved")
    assert nonzero < N_LINES / 10
    assert moving > N_LINES / 2  # the terms do depend on the bounds along most lines


def test_rate_and_departure_together_make_a_term_quadratic():
    bounds = soft_scale.Bounds(_t([1.5]), _t([2.0]), _t([2.0]), _t([2.0]))
    centre, up, down = _t([0.5]), _t([1.0]), _t([0.0])
    line = {"departure": _t([1.0]), "rate": _t([1.0]), "cap": _t([0.0]), "need": _t([0.0])}
    step = 0.1
    second = _second_difference(bounds, line, centre, up, down, step)
    # upper term (b - c) / w with b = (2 + t)(0.5 + t) below the cap of 2: second difference 2 t^2
    assert second.numel() == 1
    assert float(second[0]) == pytest.approx(2 * step**2, rel=1e-9)


def _set_functions_along(bounds, step):
    """`(p, b)` of the subset {1} at `t = -step, 0, step` along rate `+ t`, departure `+ t`."""
    found = []
    for t in (-step, 0.0, step):
        moved = soft_scale.Bounds(bounds.departure + t, bounds.rate + t, bounds.cap, bounds.need)
        lower, upper = soft_scale.set_functions(moved, 1)
        found.append((float(lower[1]), float(upper[1])))
    return found


def _second(values):
    return values[0] - 2 * values[1] + values[2]


def test_the_product_enters_p_only_when_the_departure_is_outside_the_subset():
    step = 0.1
    inside = _set_functions_along(
        soft_scale.Bounds(_t([1.5]), _t([2.0]), _t([3.0]), _t([2.5])), step
    )
    p, b = zip(*inside, strict=True)
    assert p[1] == pytest.approx(0.5, abs=1e-15) and min(p) > 0
    assert abs(_second(p)) <= 1e-12
    assert max(b) < 3.0 and _second(b) == pytest.approx(2 * step**2, rel=1e-9)
    outside = _set_functions_along(
        soft_scale.Bounds(_t([2.5]), _t([2.0]), _t([10.0]), _t([5.0])), step
    )
    p, b = zip(*outside, strict=True)
    assert p[1] == pytest.approx(2.0, abs=1e-15) and min(p) > 0
    assert _second(p) == pytest.approx(-2 * step**2, rel=1e-9)
    assert abs(_second(b)) <= 1e-12
