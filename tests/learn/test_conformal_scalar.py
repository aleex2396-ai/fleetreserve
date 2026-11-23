"""The exact and smoothed scalars of T6 on labels (`conformal_scalar`).

1. By hand, at alpha 0.10 (19 days, so the bound allows a loss sum of 1):
   - 19 days of size 1 and label 0.6, ramp width 0.05: the 19 ramps start together, so the
     smoothed scalar is 0.6 + 0.05 / 19 = 0.6026316 (the exact scalar 0.6); its gradient in each
     size is minus the scalar over 19, in each weight (0.005 + 0.6 - 0.6026316) / 19 = 0.045 / 361;
   - 19 days of size 1 and labels 0.05, 0.10, ..., 0.95, width 0.05: the ramps tile the scale, so
     the smoothed scalar is the exact one, 0.10 (day 1 past its ramp, day 2 on it), with gradient
     -0.1 in day 2's size and 0 in the others';
   - a 20th day of weight 0 and label 0 changes neither scalar, and gets gradient 0.
   At alpha 0.25 (7 days, the allowance again 1), sizes 1, 0.5, 0.5, 1, 1, 1, 1, labels 0.2, 0.15,
   0.3, 0.9, 0.9, 0.9, 0.9, width 0.2: two ramps overlap at the crossing, so the scalar is
   (0.2 + 0.2 + 0.15) / 1.5 = 11 / 30, with gradient -(11 / 30) / 1.5 in each of the two sizes;
   the exact scalar is 0.3.
2. Too few days: with 8 days at alpha 0.10 both scalars are 0; with 9 both are the smallest
   threshold.
3. The gradient of PyTorch equals the central finite difference of the scalar in every size and
   weight, on random days.
4. As the width goes to 0: on random days (29 of them, so an allowance of 2, a whole number, and
   40) the smoothed scalar is at least the exact one and at most the width over the smallest size
   above it, falls as the width does, and is within 2e-5 of it at width 1e-6; snapped to
   the gate's grid it equals `size_resolve.calibrate` (tier B) on the same days, which the exact
   scalar snapped equals too; the gradient in the size of the one day that sets the exact scalar
   tends to -label / size^2 (Yeh et al. (2025b)), and in every other size to 0.
"""

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pytest
import torch

from fleetreserve import conformal_scalar as cs
from fleetreserve import crc, size_resolve

ALPHA = crc.ALPHA_SLOT


def scalar_and_gradients(q, s, a, alpha, tau):
    q = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    a = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    g = cs.smooth_scalar(q, torch.tensor(s, dtype=torch.float64), a, alpha, tau)
    g.backward()
    return float(g.detach()), q.grad.numpy(), a.grad.numpy()


# --- 1. by hand ----------------------------------------------------------------------------------


def test_nineteen_equal_days_by_hand():
    g, dq, da = scalar_and_gradients([1.0] * 19, [0.6] * 19, [1.0] * 19, ALPHA, 0.05)
    assert g == pytest.approx(0.6 + 0.05 / 19, abs=1e-8)
    assert g == pytest.approx(0.6026316, abs=1e-7)
    np.testing.assert_allclose(dq, -g / 19, rtol=1e-9)
    np.testing.assert_allclose(da, 0.045 / 361, rtol=1e-6)
    assert cs.exact_scalar([1.0] * 19, [0.6] * 19, [True] * 19, ALPHA) == pytest.approx(0.6)


def test_tiled_ramps_give_the_exact_scalar_by_hand():
    s = [0.05 * k for k in range(1, 20)]
    g, dq, _ = scalar_and_gradients([1.0] * 19, s, [1.0] * 19, ALPHA, 0.05)
    assert g == pytest.approx(0.1, abs=1e-8)
    assert cs.exact_scalar([1.0] * 19, s, [True] * 19, ALPHA) == pytest.approx(0.1)
    assert dq[1] == pytest.approx(-0.1, abs=1e-8)
    assert np.delete(dq, 1).tolist() == [0.0] * 18


def test_a_day_of_weight_zero_is_in_no_sum():
    g, dq, _ = scalar_and_gradients([1.0] * 20, [0.6] * 19 + [0.0], [1.0] * 19 + [0.0], ALPHA, 0.05)
    assert g == pytest.approx(0.6 + 0.05 / 19, abs=1e-8)
    assert dq[19] == 0.0
    exact = cs.exact_scalar([1.0] * 20, [0.6] * 19 + [0.0], [True] * 19 + [False], ALPHA)
    assert exact == pytest.approx(0.6)


def test_two_overlapping_ramps_by_hand():
    q = [1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    s = [0.2, 0.15, 0.3, 0.9, 0.9, 0.9, 0.9]
    g, dq, _ = scalar_and_gradients(q, s, [1.0] * 7, 0.25, 0.2)
    assert g == pytest.approx(11 / 30, abs=1e-8)
    np.testing.assert_allclose(dq[:2], -(11 / 30) / 1.5, rtol=1e-7)
    assert dq[2:].tolist() == [0.0] * 5
    assert cs.ramp_sum(g, q, s, [1.0] * 7, 0.2) == pytest.approx(1.0, abs=1e-8)
    assert cs.exact_scalar(q, s, [True] * 7, 0.25) == pytest.approx(0.3)


# --- 2. too few days -----------------------------------------------------------------------------


def test_too_few_days_give_zero_and_just_enough_the_smallest_threshold():
    rng = np.random.default_rng(5)
    for n, expect_zero in ((8, True), (9, False)):
        q, s = rng.uniform(0.2, 1.0, n), rng.uniform(0.1, 0.9, n)
        g = cs.smooth_value(q, s, np.ones(n), ALPHA, 1e-6)
        exact = cs.exact_scalar(q, s, np.ones(n, dtype=bool), ALPHA)
        if expect_zero:
            assert g == 0.0 and exact == 0.0
        else:
            assert exact == pytest.approx(float((s / q).min()))
            assert g == pytest.approx(exact, abs=1e-5)


# --- 3. the gradient -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_gradient_is_the_finite_difference(seed):
    rng = np.random.default_rng(seed)
    n, tau, h = 40, 0.05, 1e-7
    q, s, a = rng.uniform(0.05, 1.0, n), rng.uniform(0.0, 1.0, n), rng.uniform(0.3, 1.0, n)
    _, dq, da = scalar_and_gradients(q, s, a, ALPHA, tau)
    for values, grad, name in ((q, dq, "q"), (a, da, "a")):
        for i in range(n):
            up, down = values.copy(), values.copy()
            up[i] += h
            down[i] -= h
            args = {"q": q, "s": s, "a": a}
            plus = cs.smooth_value(**{**args, name: up}, alpha=ALPHA, tau=tau)
            minus = cs.smooth_value(**{**args, name: down}, alpha=ALPHA, tau=tau)
            assert grad[i] == pytest.approx((plus - minus) / (2 * h), abs=1e-6), (name, i)
    assert (dq != 0).any() and (da != 0).any()


# --- 4. as the width goes to 0 -------------------------------------------------------------------


def snapped(value: float) -> float:
    return max([0.0, *(g for g in size_resolve.GRID if g <= value)])


@pytest.mark.parametrize("n", [29, 40])  # allowances 2 (a whole number, reading C4a) and 3.1
@pytest.mark.parametrize("seed", range(6))
def test_the_smoothed_scalar_tends_to_the_exact_one_and_to_calibrate(seed, n):
    rng = np.random.default_rng([11, seed, n])
    q, s = rng.uniform(0.05, 1.0, n), rng.uniform(0.0, 1.0, n)
    r = q / q.max()
    exact = cs.exact_scalar(r, s, np.ones(n, dtype=bool), ALPHA)
    gaps = []
    for tau in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        g = cs.smooth_value(r, s, np.ones(n), ALPHA, tau)
        assert exact - 1e-12 <= g <= exact + tau / r.min() + 1e-9, (tau, g, exact)
        gaps.append(g - exact)
    assert gaps[-1] <= 2e-5
    assert gaps == sorted(gaps, reverse=True)  # a wider ramp is lower, so its scalar higher
    losses = {(d, x): (1.0 if x > s[d] + 1e-12 else 0.0, 0.0) for d, x in size_resolve.wanted(r)}
    used = size_resolve.calibrate(losses, r, list(range(n)), "B").used
    assert snapped(min(exact, 1.0)) == used
    assert snapped(min(cs.smooth_value(r, s, np.ones(n), ALPHA, 1e-7), 1.0)) == used


@pytest.mark.parametrize("seed", range(4))
def test_the_gradient_tends_to_that_of_the_critical_day(seed):
    rng = np.random.default_rng([12, seed])
    n = 40
    q, s = rng.uniform(0.05, 1.0, n), rng.uniform(0.0, 1.0, n)
    exact = cs.exact_scalar(q, s, np.ones(n, dtype=bool), ALPHA)
    k = int(np.argmin(np.abs(s / q - exact)))
    assert s[k] / q[k] == pytest.approx(exact)
    for tau, tolerance in ((1e-3, 0.05), (1e-6, 1e-4)):
        _, dq, _ = scalar_and_gradients(q, s, np.ones(n), ALPHA, tau)
        assert dq[k] == pytest.approx(-s[k] / q[k] ** 2, rel=tolerance), tau
        assert np.delete(dq, k).tolist() == [0.0] * (n - 1)
