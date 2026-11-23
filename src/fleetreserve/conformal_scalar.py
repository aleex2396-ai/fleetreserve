"""The calibrated scalar of T6 on a day's label, exact and smoothed, the smoothed one
differentiable in the days' sizes and offering weights (plan sections 6 (T6), 13 (WP9, conformal
training); Yeh et al. (2025b)). Used by `size_ct` to train a day-size model through its own
calibration; never used to calibrate an offer, which `size_resolve.calibrate` does unchanged.

A day i of a set of calibration days has a size `q_i > 0`, a label `s_i >= 0` (`size_label`: the
exact safe scale of its certified shape on its true cars, clipped to [0, 1]) and an offering weight
`a_i` in [0, 1] (1: the day offers; 0: it does not, and is in no sum). At the scalar `g` its scale
is `g q_i`, and its loss is the step `1[g q_i > s_i]`: it loses above its label and not at it (the
reading R1 of `crc`). The bound of T6 on these days is

    (sum_i a_i L_i(g) + 1) / (sum_i a_i + 1) <= alpha,   that is   sum_i a_i L_i(g) <= c,
    c = alpha (sum_i a_i + 1) - 1,

compared with `crc.TOLERANCE` as `crc` compares it, and the scalar is the supremum of the `g >= 0`
where it holds.

`exact_scalar(q, s, offering, alpha)`: with weights 0 or 1, the scalar of the step losses,
`crc.beta_hat_exact` on one step per offering day at the threshold `s_i / q_i`: 0 when the
offering days are too few (`crc.enough_days`), else the attained supremum.

`smooth_scalar(q, s, a, alpha, tau)`: the same with each step replaced by the ramp
`L_i(g) = min(max((g q_i - s_i) / tau, 0), 1)` of width `tau` above the label: 0 up to and at the
label, 1 once the scale is `tau` above it, linear between, and never above the step. The sum
`G(g) = sum_i a_i L_i(g)` is piecewise linear and non-decreasing in `g`, 0 at `g = 0`, with
breakpoints at `s_i / q_i` and `(s_i + tau) / q_i`. `crossing` walks them in order and finds the
segment on which `G` first exceeds `c + crc.TOLERANCE`: the days already past their ramp there
(`done`) and those on it (`active`). On that segment

    sum_done a_i + sum_active a_i (g q_i - s_i) / tau = c + crc.TOLERANCE,

so the scalar is `(tau (c + TOLERANCE - sum_done a_i) + sum_active a_i s_i) / sum_active a_i q_i`,
computed in PyTorch from the tensors `q`, `s`, `a` with the two sets held fixed. Its gradient is
the implicit-function gradient of the bound at the scalar, with the derivatives taken from the
right (a day whose ramp starts at the scalar is active). When `c + TOLERANCE < 0` (too little
weight, the `crc.enough_days` rule) the scalar is 0 and carries no gradient. Since every ramp is
at most its step, the smoothed scalar is at least the exact one; since the ramp of day i at `g` is
at least its step at `g - tau / q_i`, it is at most `tau / min q` above it (the smallest size of a
day of weight above 0), so it tends to the exact scalar as `tau` tends to 0; a wider ramp is
lower, so its scalar is higher. With one active day
`k` the gradient in `q_k` tends to `-s_k / q_k^2`, the derivative of the exact scalar `s_k / q_k`
through the one day that sets it (Yeh et al. (2025b), the case of piecewise constant losses).

Readings (each with the alternative not taken):

C4a. The smoothing is a linear ramp above the label. [Alternative: the logistic function of
     `(g q_i - s_i) / tau`. Where `c` is a whole number (alpha 0.10 and 19, 29, ... days) it does
     not tend to the exact scalar: with two days at thresholds t1 < t2, the rest far above, and
     c = 1, the logistic sum equals 1 at (t1 + t2) / 2 for every tau, where the exact scalar is
     t2. The ramp tends to t2.]
C4b. The scalar is not capped here; `size_ct` caps it.
"""

import math

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import torch

from fleetreserve import crc


def allowance(a_sum: float, alpha: float) -> float:
    """`c = alpha (sum a + 1) - 1`, the loss sum the bound allows (module text)."""
    return alpha * (a_sum + 1.0) - 1.0


def exact_scalar(q, s, offering, alpha: float) -> float:
    """The scalar of the step losses on the offering days (module text)."""
    q = np.asarray(q, dtype=float)
    s = np.asarray(s, dtype=float)
    offering = np.asarray(offering, dtype=bool)
    if (q[offering] <= 0).any() or (s[offering] < 0).any():
        raise ValueError("sizes must be above 0 and labels at least 0")
    days = [((float(s[i] / q[i]),), (1.0,)) for i in np.flatnonzero(offering)]
    return crc.beta_hat_exact(days, alpha)


def ramp_sum(g: float, q, s, a, tau: float) -> float:
    """`G(g)`, the weighted sum of the ramps at the scalar `g` (module text), in numpy."""
    q, s, a = (np.asarray(v, dtype=float) for v in (q, s, a))
    return float((a * np.clip((g * q - s) / tau, 0.0, 1.0)).sum())


def crossing(q, s, a, alpha: float, tau: float):
    """(done, active) of the segment on which the ramp sum first exceeds the allowance, as two
    boolean arrays; None when the allowance is below 0 (the scalar is 0). Raises `ValueError` if
    the sum never exceeds it, which needs alpha 1 or more."""
    if not tau > 0:
        raise ValueError(f"tau must be positive, got {tau}")
    q, s, a = (np.asarray(v, dtype=float) for v in (q, s, a))
    if (q <= 0).any() or (s < 0).any() or (a < 0).any() or (a > 1).any():
        raise ValueError("sizes must be above 0, labels at least 0 and weights in [0, 1]")
    limit = allowance(float(a.sum()), alpha) + crc.TOLERANCE
    if limit < 0:
        return None
    used = np.flatnonzero(a > 0)
    starts, ends = s[used] / q[used], (s[used] + tau) / q[used]
    # events in order of the scale: a ramp ends (kind 0) before one starts (kind 1) at one point
    events = sorted(
        [(float(e), 0, int(i)) for e, i in zip(ends, used, strict=True)]
        + [(float(b), 1, int(i)) for b, i in zip(starts, used, strict=True)]
    )
    done = np.zeros(len(q), dtype=bool)
    active = np.zeros(len(q), dtype=bool)
    level, point, k = 0.0, 0.0, 0  # level: G at point, which the events at a point leave as is
    while k < len(events):
        here = events[k][0]
        reached = level + float((a[active] * q[active]).sum()) / tau * (here - point)
        if reached > limit:
            return done, active
        level, point = reached, here
        while k < len(events) and events[k][0] == here:
            _, kind, i = events[k]
            active[i], done[i] = (False, True) if kind == 0 else (True, False)
            k += 1
    raise ValueError("the ramp sum never exceeds the allowance (alpha must be below 1)")


def smooth_scalar(q: torch.Tensor, s: torch.Tensor, a: torch.Tensor, alpha: float, tau: float):
    """The smoothed scalar as a 0-dimensional tensor of `q`'s dtype, differentiable in `q` and
    `a` (module text); `s` is a tensor of labels."""
    found = crossing(
        q.detach().cpu().numpy(), s.detach().cpu().numpy(), a.detach().cpu().numpy(), alpha, tau
    )
    if found is None:
        return torch.zeros((), dtype=q.dtype)
    done, active = (torch.as_tensor(m, device=q.device) for m in found)
    limit = alpha * (a.sum() + 1.0) - 1.0 + crc.TOLERANCE
    numerator = tau * (limit - a[done].sum()) + (a[active] * s[active]).sum()
    return numerator / (a[active] * q[active]).sum()


def smooth_value(q, s, a, alpha: float, tau: float) -> float:
    """`smooth_scalar` as a float, from arrays (no gradient)."""
    found = smooth_scalar(
        torch.as_tensor(np.asarray(q, dtype=float)),
        torch.as_tensor(np.asarray(s, dtype=float)),
        torch.as_tensor(np.asarray(a, dtype=float)),
        alpha,
        tau,
    )
    value = float(found)
    return value if math.isfinite(value) else math.inf
