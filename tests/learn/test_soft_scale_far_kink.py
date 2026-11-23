"""T8's check (plan section 7): a term of the soft minimum with a kink inside the step that is far
above the smallest term at the point but below it at both ends of the step (the review of
0d475a7, its blocking finding). `soft_scale_check`'s module text, rule parts 1(b) and 2.

1. The construction. Two service hours, no upper widths, lower widths `[3, 1e-7]`, centre `[5.5,
   0.5 + 21e-7]`; three cars departing at 2.5 with rate 2 and cap 10, needs 4.5, `4 - 0.006 h`
   and 0 (`h` the soft minimum's first step, 1e-5); the direction `(-0.3, 0.6, 0.742)` in the
   need field. By hand (every share: hour 1 is 1, hour 2 is 0.5, so `rate (d - s(A))` is 3, 4 and
   2 for `{1}`, `{2}` and `{1, 2}`): `p({1}) = 2.5 - 0.006 h` with slope 0.3, `p({2}) = 0.5` with
   slope -0.3 until the second car's part turns on at `t = 0.01 h` (slope 0.3 after), and `p({1,
   2}) = 4.5 - 0.006 h` with slope 0.3. The lower terms `(c(A) - p(A)) / w(A)` are then about
   1.00000002 (slope -0.1), 21 (slope 3e6, then -3e6 after `0.01 h`: -9 at `-h` and -8.4 at `+h`)
   and 0.50000070 (slope `-0.3 / (3 + 1e-7)`). The term of `{2}` lies 20.5 above the smallest at
   the point, so its weight there is about `exp(-2050)`, and below every other term at both ends
   of the first step. The soft minimum's derivative is the weighted mean of the slopes: the term
   of `{1, 2}` has weight `1 / (1 + exp(-50) + exp(-2050))`, the one of `{1}` about `exp(-50)`, so
   the derivative is `-0.3 / (3 + 1e-7)` to within `1e-21` relatively. Before the fix the first
   step showed no kink (the term's bounded weight from its central slope underflowed to 0), was
   resolvable and decided on `D = 30000`: the right gradient failed and a gradient equal to `D`
   passed. Asserted: the first step shows a kink; the right gradient is not failed and the
   gradient equal to the first `D` is not passed; a step that decides decides as the true
   derivative says. The step's kink part is the one of part 2 (its weight relative to the straight
   terms at `-h`, 0 and `h`), checked from the terms.
2. A hand case of the soft minimum's parts (`_soft_parts`) with the weights worked out by hand
   (in the test's text).
3. The slope bounds `Lambda_k` (`slope_bounds`) of the construction, by hand.
"""

import math

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import pytest
import torch

from fleetreserve import soft_scale
from fleetreserve import soft_scale_check as check

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net

H = check.STEPS["soft"]
DIRECTION = (-0.3, 0.6, (1 - 0.3**2 - 0.6**2) ** 0.5)
TRUE = -0.3 / (3 + 1e-7)  # the soft minimum's derivative (module text, part 1)


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def _construction():
    bounds = soft_scale.Bounds(
        departure=_t([2.5, 2.5, 2.5]),
        rate=_t([2.0, 2.0, 2.0]),
        cap=_t([10.0, 10.0, 10.0]),
        need=_t([4.5, 4.0 - 0.01 * H * 0.6, 0.0]),
    )
    shape = (_t([5.5, 0.5 + 21e-7]), _t([0.0, 0.0]), _t([3.0, 1e-7]))
    return bounds, shape, _t(DIRECTION)


def test_a_far_term_with_a_kink_inside_the_step_shows_and_decides_nothing():
    bounds, shape, direction = _construction()
    assert bool(soft_scale.centre_inside(bounds, shape[0]))
    terms = soft_scale.terms(bounds, *shape)
    finite = terms[torch.isfinite(terms)].tolist()
    assert finite == pytest.approx([1.00000002, 21.0, 0.5 + 21e-7 / 3], rel=1e-6)
    steps = check.measure("soft", bounds, "need", direction, shape)
    first = steps[0]
    assert first["step"] == H
    assert abs(first["difference"] - TRUE) > 1e4  # the far term moved D
    right = check.compare("soft", bounds, "need", direction, shape)
    assert right["autograd"] == pytest.approx(TRUE, rel=1e-12)
    planted = check.decide(first["difference"], steps)
    assert (right["outcome"], planted[0]) != ("failed", "passed"), (right, planted)
    assert right["outcome"] != "failed", right
    assert planted[0] != "passed", planted
    assert first["kink_shown"] and first["kink"] > 0
    # a step that decides, decides as the true derivative says
    for step in steps:
        if not step["kink_shown"] and step["resolvable"]:
            assert abs(step["difference"] - TRUE) <= step["allowance"]
            assert check.judge(TRUE, step) == "passed"
            assert check.judge(first["difference"], step) == "failed"
    print("outcomes:", right["outcome"], planted[0], [s["kink_shown"] for s in steps])


def test_the_soft_parts_with_weights_by_hand():
    """Three terms at the five points `-h, -h/2, 0, h/2, h` (h = 1e-5, sigma = 0.01):

    - term 0 is 0 at every point (straight, its rounding bound 1e-6 at every point);
    - term 1 is `sigma ln 100` at `-h` and `h`, `sigma ln 1e4` at 0 and `3 sigma ln 10` at `-h/2`
      and `h/2` (a peak at 0; its rounding bound 2e-6): its second difference over the whole step
      is `2 sigma ln 100 - 2 sigma ln 1e4 = -4 sigma ln 10`, about -0.092, above 4 times 2e-6, so
      it is kinked (over each half the second difference is 0);
    - term 2 is left out (infinite).

    The straight terms are term 0 alone: its weight among them is 1, its slope 0, the weighted
    mean slope 0, its deviation 0, so `pihat_0 = 1`, the truncation part `8 * 1 * 0^3 ...` is 0 and
    the hidden part `4 pihat_0 e_0 / h = 4 * 1e-6 / 1e-5 = 0.4`.
    Term 1's weight relative to the straight terms, `exp(-(T_1 - T_0) / sigma)`, is `exp(-ln 100)
    = 0.01` at `-h` and at `h` and `exp(-ln 1e4) = 1e-4` at 0, each raised by the rounding lift
    `exp((e_1 + max e of the straight terms) / sigma) = exp((2e-6 + 1e-6) / 0.01) = exp(3e-4)`.
    The slope bounds are 1 and 3 for the finite terms (100 for the left-out term, not used), so
    `Lambda_1 + max Lambda = 3 + 3 = 6`. The kink part is
    `exp(3e-4) (sigma (0.01 + 0.01) / (2 h) + 1e-4 * 6) = exp(3e-4) (10 + 6e-4)`.
    """
    h, sigma = 1e-5, check.SIGMA
    assert sigma == 0.01
    inf = math.inf
    peak = [
        sigma * math.log(100),
        3 * sigma * math.log(10),
        sigma * math.log(1e4),
        3 * sigma * math.log(10),
        sigma * math.log(100),
    ]
    points = [(0.0, _t([0.0, value, inf]), _t([1e-6, 2e-6, 0.0])) for value in peak]
    truncation, kink, hidden = check._soft_parts(points, h, _t([1.0, 3.0, 100.0]))
    assert truncation == 0.0
    assert hidden == pytest.approx(0.4, rel=1e-12, abs=0)
    assert kink == pytest.approx(math.exp(3e-4) * (10 + 6e-4), rel=1e-12, abs=0)
    # every term kinked (no straight term): the kink part is infinite
    points = [(0.0, _t([value, inf]), _t([2e-6, 0.0])) for value in peak]
    assert check._soft_parts(points, h, _t([3.0, 100.0]))[1] == inf


def test_the_slope_bounds_of_the_construction_by_hand():
    """In need every car's `L_i` is 1, so `Lambda_k = sum_i |v_i| / w(A_k)`: `sum_i |v_i| = 0.3 +
    0.6 + 0.742...`; the lower widths of `{1}`, `{2}` and `{1, 2}` are 3, 1e-7 and `3 + 1e-7`.
    Each true slope of the construction is within its bound (0.1, 3e6 and about 0.1)."""
    bounds, shape, direction = _construction()
    found = check.slope_bounds(bounds, shape, "need", direction)
    reach = sum(abs(x) for x in DIRECTION)
    lower = found[4:].tolist()  # the lower terms: subsets {}, {1}, {2}, {1, 2}
    assert lower[1:] == pytest.approx([reach / 3, reach / 1e-7, reach / (3 + 1e-7)], rel=1e-12)
    assert lower[1] >= 0.1 and lower[2] >= 3e6 and lower[3] >= 0.3 / (3 + 1e-7)
