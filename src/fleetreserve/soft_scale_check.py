"""T8's check on development days (plan sections 7 and 13 WP9): autograd against finite differences
for the exact scale of the certified shape, the soft minimum's sandwich, one batch timed, and the
scale beside the certified value.

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.soft_scale_check --model <name> --configuration <name>
        [--matrix <file>] [--units <unit> ...] [--days N] [--seed N]
        [--tables <folder>] [--truth-run <folder>]

Inputs, through the code base's own routes (no loader of its own): `--matrix` a run matrix of the
gate's format (default: the committed `gate/matrix.json`); `--configuration` one of its
configurations, which fixes kappa, the departure reading, the energy bounds, the shape rule and
the rest; `--model` the module `fleetreserve.models.<name>`, loaded, checked and fitted as
`dial_choice` does it (`learn_run.load_model`, `learn_run.check_model`,
`dial_choice.make_factory`: fitted on its `fit_units` only, predicting at the levels up to the
configuration's kappa); `--units` the units checked (default: the matrix's gate units, the
development units). The truth tables are checked as `gate_run.check_inputs` checks them (a verified
truth build at the commit checked out now, of the tables' setting, with the tables' hashes) and
read by `days_from_truth.load_unit_days` with the matrix's readings.

Training-site days. The check is also run on days of training units, so that WP9's item "T8's
check" has more checked days than the development units hold (`tools/wp9_check.py`, reading W3e;
its run list `passmarks/wp9-t8-runs.json`). Their matrices, `gate/t8-training-a.json` and
`gate/t8-training-b.json`, each score every second training unit of `truth_build.TRAIN_UNITS`
(from the first and from the second) and are rotation what-ifs (`gate_config`: a training unit is
a gate unit only with `rotation_whatif: true`). No rule is relaxed for them: `learn_run.check_model`
is given the matrix's gate units, so a model fitted on a checked unit (`lgbm_tail`, fitted on all
eight) is refused (exit 2), and `models.lgbm_tail_rotation`, fitted on the four training units
the matrix does not score, is the model of those runs. The check compares autograd with finite
differences on each day's forecast bounds, whatever model gave them: it scores no forecast and
reports no offer, so which model gave the bounds changes what is checked, not whether the check
is sound. Its outputs then say that the run is a rotation what-if, as every command that takes
such a matrix does (`rotation_whatif.mark`: the key `rotation_whatif` of the results, the label
and the definitions; `rotation_whatif.summary_head`: the first two lines of the summary), since
the tracking and certified values it writes are figures of training units; a run of a matrix
without the field carries no trace of it.

The days. Per unit, the counting days in a seeded order (`numpy.random.default_rng` of `--seed`
and the unit's name); each is prepared as the gate prepares it (`gate_day.prepare_day`: forecasts,
the steered cars, the certificate on their forecast rows, abstention); the first `--days` (default
100) days that offer are checked and the abstaining days met before them are counted by reason. A
unit with fewer offering days has all of them checked, and says so.

Per checked day, on the steered cars' forecast bounds (`soft_scale.forecast_bounds` of the
prepared sub-day) and the certified shape (centre `m`, widths up and down both `w`, of the
certificate's policy):

- `scale`: `soft_scale.exact_scale`, T1(b) on the forecast set with hour 0 and the recovery hours
  free; `soft_scale`: the soft minimum at `SIGMA`; `terms` the number N of finite terms;
  `centre_inside`; the sandwich's violation `max(soft - scale, scale - soft - SIGMA ln N, 0)`. The
  certified box lies in the forecast set, so `scale` is expected at 1 or above; days below
  `1 - 1e-6` are counted (`below_one`), not judged. Under a configuration with driver-risk rows
  the certificate is built with them (as the gate builds it) and the scale is computed on the
  forecast set without them.
- the derivative check, for the soft minimum and for the exact scale: along one seeded random unit
  direction `v` over the steered cars in each of the fields departure, cap and need (the plan's
  "departure and energy directions") and rate (reported, not part of the pass), the derivative `a`
  from autograd against finite differences of `phi(t) = f(theta + t v)`, in double precision on
  the CPU, by the measuring rule below. Each comparison is written with its `outcome`: `passed`,
  `failed`, `unresolved` or `kink` (the last two neither passed nor failed), or, for the exact
  scale on a day whose smallest finite term is not unique (`smallest_terms` above 1: another term
  within `TIE` times `max(1, |scale|)` of it), `tied`: the exact scale has a kink at the point or
  very near it, where it has no derivative (two terms equal at the point, unless their slopes
  along `v` agree) or one that the steps used cannot measure (two terms that differ by less than
  `TIE`), so no comparison is made and nothing but the outcome is written. The soft
  minimum is compared on every day. On `learn_fake.small_gate_units` the certified shape is
  limited by several faces at once on most checked days, so ties are common there; on development
  days it is untested. A value, derivative or difference that is NaN or infinite fails the
  comparison, which is written with nulls (the results file holds no NaN) and counted as
  `non_finite`.
- the deciding fields (`deciding_fields`): departure, cap and need; on a configuration whose
  forecast departures are whole hours (`whole_hour_departures`: the oracle departure, or the
  floored one) cap and need only, and the departure is NOT APPLICABLE: every point is a kink in
  that direction, so there is no derivative to measure; its comparisons are made, written and
  counted, and reported only, as the rate's are (the owner's decision of 15 November 2026, plan
  section 7, T8). The run records `deciding_fields` and `not_applicable` in its settings and the
  summary names them.
- exercised fields: per function and deciding field, `resolvable` counts the comparisons decided
  at a step where they are resolvable (rule, part 3). A deciding field of either function with
  none is named in `not_exercised` and fails the check: a check that has measured no derivative of
  a field to the resolution the target needs has not checked that field. A run on which every
  checked day's exact scale ties exits 1 for this reason; the summary says how many checked days
  tied, so that exit reads as a result of the days, not as a crash. On the made-up units of
  `tests/learn/test_soft_scale_check.py` with whole-hour departures the exact scale's cap, or the
  need of both functions, has no resolvable comparison either, so those runs still exit 1; the
  not-applicable rule names the departure only.
- Lambda's tracking of the certified causal value (a reading of the plan's "tracking"): the day's
  `scale` beside its certified value at scale 1 (`gate_day.PreparedDay.value`, kWh), and the value
  of the largest scaled certified box inside the forecast set, `scale * value`. Per unit and over
  all units: the Pearson and Spearman correlations of `scale` with the certified value and of
  `scale * value` with it; None with fewer than 3 days or when either column is constant up to
  rounding (`constant`: its range at most `CONSTANT` times `max(1, largest |value|)`), where a
  correlation would measure rounding only.

The measuring rule. `L` is `LIMIT`, the plan's 1e-3 relative error. The rule was written from this
derivation before it was first run; its steps and constants were not fitted to results.

1. What a difference can be wrong by. At step `h` the function is measured at the five points
   `OFFSETS` times `h` (`-h, -h/2, 0, h/2, h`): the central difference `D = (phi(h) - phi(-h)) /
   2h`, the one-sided ones `D- = (phi(0) - phi(-h)) / h` and `D+ = (phi(h) - phi(0)) / h`, their
   spread `q = D- - D+`, and three second differences (`second_differences`): `S = phi(h) -
   2 phi(0) + phi(-h) = -q h` over the whole step and `S-`, `S+` over its halves `[-h, 0]` and
   `[0, h]`. Along a line in one field every term `T_k` is piecewise linear in `t` (`soft_scale`:
   linear in the caps, needs and rates with the departures fixed, and in the departures, caps and
   needs with the rates fixed; a line in one field never moves a rate and a departure together),
   so each second difference is 0 unless a kink lies inside its interval or at its middle point.
   (a) The exact scale, `min_k T_k`. In cap, need and rate, `b(A)` is a sum of minima of functions
   linear in that field and `p(A)` a sum of maxima of such functions with 0, so every term, and
   their minimum, is concave along the line; for a concave `phi`, `D+ <= phi'(0+) <= phi'(0-) <=
   D-`, so `|D - phi'(0)| <= |q| / 2` wherever `phi'(0)` exists, whatever kinks lie in the step.
   In departure `phi` is piecewise linear; when the kinks inside `[-h, h]` lie on one side of 0,
   the one-sided difference of the other side is `phi'(0)` exactly and again `|D - phi'(0)| =
   |q| / 2`. Between kinks there is no truncation. Kinks on both sides of 0 within one step in the
   departure direction are not bounded by `|q| / 2`: with kinks at `-a` and `+b`, `x = (s_L - s_0)
   (h - a) / h` and `y = (s_R - s_0) (h - b) / h` (`s_L`, `s_R` the slopes beyond them), `|D -
   phi'(0)| = |x + y| / 2` while `|q| / 2 = |x - y| / 2`, so two kinks of opposite convexity
   escape it (the review of 2078c32, its corner C1). They are SHOWN instead: within one half of
   the step, a single kink away from the half's ends makes that half's second difference non-zero,
   so a step with exactly one kink on each side of 0 shows a kink in `S-` or `S+` even where the
   two cancel in `S`; several kinks within one half can cancel in its second difference (below). A
   step that shows a kink decides nothing (part 3). Not covered: kinks whose effects on all three
   second differences cancel or lie within their rounding. Examples: two kinks of opposite
   convexity each within about `4 e / |slope change|` of the point, where `phi` stays within its
   rounding of a line over the whole step; four kinks forming two equal bumps centred at `-h/2`
   and `h/2`, where all three second differences cancel exactly (the review of 0d475a7). The
   measured steps do not see them; a smaller step might, but it is measured only when the first
   shows a kink or is not resolvable (part 4). Each example needs exact cancellations or kinks
   within rounding of the point, which a seeded random direction meets with a chance not bounded
   here (a guess: zero for the exact cancellations).
   (b) The soft minimum. At a step, the finite terms are split into the KINKED ones, whose
   largest second difference of the three, `s_k`, is above `4 e_k`, four times the term's rounding
   bound (a term has no curvature, so only a kink moves them beyond rounding), and the STRAIGHT
   ones, the rest. With `Z_L(t)` and `Z_K(t)` the sums of `exp(-T_k(t) / sigma)` over each,
   exactly `phi = phi_L + sigma log(1 - rho)`, with `phi_L = -sigma log Z_L` the soft minimum of
   the straight terms alone and `rho = Z_K / (Z_L + Z_K)`, the kinked terms' share of the weight.
   So `D - phi'(0) = (D_L - phi_L'(0)) + sigma (log(1 - rho(h)) - log(1 - rho(-h))) / 2h - sigma
   (log(1 - rho))'(0)`, which reads `rho` at `-h`, 0 and `h` only, whatever the kinked terms do
   between them.
   The straight part. Where no straight term has a kink inside the step, `phi_L(t) = -sigma log
   sum_k exp(-(T_k + t g_k) / sigma)` over the straight terms, `g_k` their slopes, so `phi_L'''` is
   the third central moment of the slopes under the weights `pi_k(t)` (the softmax of `-T(t) /
   sigma` over the straight terms) divided by `sigma^2`, and the central difference's truncation
   is at most `h^2 M3 / 6` with `M3` a bound on `|phi_L'''|` over the step. With `gbar = sum_k
   pi_k(0) g_k`: `pi_k(t) <= pi_k(0) exp(|t| |g_k - gbar| / sigma)` (Jensen's inequality on the
   normaliser, which needs every term in it linear within the step: the reason kinked terms are
   kept out), `|x - gbar_t|^3 <= 4 (|x - gbar|^3 + |gbar - gbar_t|^3)` and `|gbar - gbar_t|^3 <=
   E_t |g - gbar|^3`, so `M3 = 8 sum_k pihat_k |g_k - gbar|^3 / sigma^2` with `pihat_k = pi_k(0)
   exp(h |g_k - gbar| / sigma)`, sums over the straight terms. The slopes `g_k` are the terms' own
   central differences (exact on a linear piece). A straight term may still hold a kink that does
   not show (`s_k` at most `4 e_k`, so its true second difference at most `8 e_k`): its central
   slope is then off by up to `4 e_k / h`, which reaches `D_L` with a weight at most `pihat_k` (the
   review of 2078c32, its corner C2): that part, `4 sum_k pihat_k e_k / h` over the straight
   terms, is in the rounding part (part 2).
   The kinked part. With `rho_k = exp(-T_k / sigma) / Z_L` for a kinked term and `R = Z_K / Z_L`
   their sum: `sigma |log(1 - rho)| = sigma log(1 + R) <= sigma R`; and `sigma (log(1 - rho))' =
   -sigma rho' / (1 - rho)` with `sigma rho' = sum_k pi_k (phi' - T_k')` over the kinked terms
   (`pi_k` the weight among all terms, `pi_k / (1 - rho) = rho_k`), so `|sigma (log(1 -
   rho))'(0)| <= sum_k rho_k(0) (|phi'(0)| + |T_k'(0)|)`, on each side of 0 when a kink lies at
   the point. A term's slope is at most `Lambda_k = sum_i |v_i| L_i / w(A)` (`slope_bounds`;
   `L_i` as in the rounding below, a bound on a car's slope of `b` and `p` in the field), and
   `|phi'| <= max_j Lambda_j` over the finite terms (a weighted mean of their slopes). So `|D -
   phi'(0)| <= |D_L - phi_L'(0)| + sigma (R(-h) + R(h)) / 2h + sum_k rho_k(0) (Lambda_k + max_j
   Lambda_j)`, the last sum over the kinked terms. The `rho_k` are computed from the computed
   terms, each raised by `exp((e_k + max_j e_j) / sigma)`, the `max` over the straight terms, which
   covers the terms' rounding. A `rho_k` that underflows to 0 is below about `1e-308` and puts
   less than `1e-308 (sigma / h + 2 max_j Lambda_j)` on `D`. The earlier form, a kinked term
   weighted by `pihat_k` from its central slope, did not hold (the review of 0d475a7): a term far
   above the smallest at the point but below it at both ends of the step has a central slope near
   0, its `pihat_k` underflowed to 0, and it moved `D` by 3e4 while the step showed no kink
   (`tests/learn/test_soft_scale_far_kink.py`). Not covered: as in (a), kinks within one straight
   term whose effects on its three second differences cancel or lie within its rounding.
   Rounding. Each computed term carries an error of at most `e_k` (`term_rounding`), to first
   order in the float64 machine epsilon `eps` and for any order of summation (`K` cars, `n` service
   hours, `w(A)` the width of the term's side, `C(A) = sum_{t in A} |centre_t|`): upper terms
   `eps ((K + n) b(A) + n C(A) + Q) / w(A) + (n + 2) eps |T_k|`, lower terms `eps ((K - 1) p(A) +
   (n + 2) P + n C(A) + Q) / w(A) + (n + 2) eps |T_k|`, with `P = sum_i (need_i + rate_i
   departure_i)` and `Q = sum_i |theta_i| L_i` for the rounding of the moved field's coordinates
   `theta + t v` (`L_i` bounds a car's slope of `b` and `p` in that field: 1 in cap and need, the
   rate in departure, the departure plus `n` in rate). The count: the shares are exact (for `t <=
   d <= t + 1` with `t >= 1`, `d - t` is exact by Sterbenz's lemma, and the `min` and `max` only
   select); a car's `rate * s(A)` adds at most `n` roundings of its size and `need - rate (d -
   s(A))` at most `n + 2` of `need + rate d`; a sum of `K` numbers of one sign at most `K - 1`
   roundings of the sum; `c(A)`, `w(A)`, the subtraction and the division the rest. A function's
   value carries at most `e`: for the exact scale, the largest `e_k` over the terms that can be the
   smallest given their bounds; for the soft minimum, `sum_k pi_k (e_k + 2 eps |T_k|) + (N + 3) eps
   sigma + 2 eps |phi|` (the log-sum-exp's own roundings). Taken as the largest over the five
   points (and `e_k` as the largest of a term's five), `e` puts at most `e / h` on `D` (plus
   `2 eps |D|` for its own subtraction and division), `4 e / h` on `q` and `4 e` on a second
   difference. `tests/learn/test_soft_scale_measure.py` checks `e_k` and `e` against
   exact rational (and 50-digit decimal) arithmetic at 30 and 300 cars.

2. The allowance `tau`, in three parts, `tau = rounding + truncation + kink`, a bound on `|D -
   phi'(0)|` at a step that shows no kink (except as part 1 leaves out). Exact scale: `rounding =
   5 e / h + 2 eps |D|` (`D`'s own and the rounding share of `|q| / 2`), `truncation = 0`, `kink =
   max(max(|S|, |S-|, |S+|) - 4 e, 0) / 2h`, the largest second difference beyond what rounding
   explains (for `S` alone this is `|q| / 2 - 2 e / h`, the spread beyond rounding). Soft minimum:
   `rounding = e / h + 4 sum_k pihat_k e_k / h + 2 eps |D|` and `truncation = h^2 M3 / 6`, both
   over the straight terms, and `kink = sigma (R(-h) + R(h)) / 2h + sum_k rho_k(0) (Lambda_k +
   max_j Lambda_j)` over the kinked terms (each `rho_k` raised for rounding), infinite when no
   term is straight. A kink is SHOWN at a step when its kink part is above 0: for the soft
   minimum, when some kinked term's weight against the straight terms at `-h`, 0 or `h` does not
   underflow. At a step that shows one, `tau` is written but is not a bound in every case (in
   departure, kinks on both sides) and decides nothing.

3. The decision, at a step that shows no kink, on the interval `[D - tau, D + tau]` that holds
   `phi'(0)`; a step that shows a kink decides nothing (`judge` returns `unresolved`), whether or
   not it is resolvable, since the point may be the kink, where `phi'(0)` need not exist and
   autograd's value is a convention (`soft_scale`). `|a - g| - L |g|` is
   piecewise linear in `g` and convex on each side of 0, so: the comparison is `passed` when
   `|a - g| <= L |g|` at both ends, and at 0 when 0 lies inside (the relative error of `a` is at
   most `L` for every derivative the allowance leaves possible); `failed` when `|a - g| > L |g|` at
   both ends, and at 0 and at `a` when they lie inside (above `L` for every one). It is RESOLVABLE
   when `tau <= RESOLUTION L |D|`, `RESOLUTION = 1/4`. Then 0 is outside the interval, a true
   relative error above `L` cannot pass, and one of at most `0.499 L` cannot fail: for `g'` in the
   interval, `|a - g'| <= |a - g| + 2 tau` and `L |g'| >= L (|g| - 2 tau)`, with `2 tau (1 + L) <=
   (1 + L) L |D| / 2` and `|D| <= |g| / (1 - L / 4)`, and `(1 + L) / (2 (1 - L / 4)) < 0.5007`. So
   a resolvable step that shows no kink always decides: a comparison not shown within `L` there
   fails. At a step that is not resolvable a comparison is passed or failed only when the interval
   shows it, and is otherwise neither (`unresolved`).

4. Steps. The first step is `STEPS[kind]`: the exact scale `1e-4` (no truncation between kinks, so
   as large as kinks allow: the chance of a kink of the active terms within `h` of the point is
   about `h` times their kink density along `v`); the soft minimum `1e-5`, `sigma / 1000` (its
   truncation is `(4/3) (h / sigma)^2` times the weighted third absolute moment of the slopes,
   which the step must keep small where nearly tied terms have very different slopes). A
   comparison is decided at the first step where it can be. The smaller step `h / SHRINK` is
   measured (`needs_smaller`) when the first step shows a kink, or when it is not resolvable and
   the parts that shrink with the step (kink and truncation) are larger than its rounding part
   (the soft minimum's curvature shown within the step) (`SHRINK` 100: a kink at a uniform place
   within the first step lies within the smaller one with chance 1/100, for 100 times the rounding
   part); it has its own allowance by the same formulas, and the comparison is decided there if it
   can be. A comparison that shows a kink at both steps is recorded `kink`, neither passed nor
   failed (a kink at or within `h / SHRINK` of the point, where no derivative need exist: the
   owner's decision of 15 November 2026, plan section 7, T8); any other comparison not decided is
   `unresolved`. The measurement never reads `a`, so which comparisons are resolvable does not
   depend on the gradient checked. `tests/learn/test_soft_scale_wrong_gradient.py` hands the rule
   gradients wrong by 100%, 10%, 1%, 0.2% and 0.01% on made-up fleets of 30 and 300 cars;
   `tests/learn/test_soft_scale_planted.py` hands it gradients wrong by 100% to 0.01% on the hard
   fleets of the review of 2078c32 (kinks placed at and around the point, on both sides, whole-hour
   departures, spread directions, large energies), judged against derivatives in exact arithmetic.

One batch timed: the checked days with the most common number of service hours (ties: the fewer
hours), cars padded with zero limits, `soft_scale.soft_scale` summed and its gradient in all four
fields computed, once to warm up and once timed, on the CPU in float64 and on MPS in float32 when
`torch.backends.mps.is_available()` (else "not available"). Seconds, batch size, device and dtype
are written; there is no pass threshold. The torch thread count is set to 1 before any torch
operation (LightGBM shares the process: `hazard_net`).

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary also on
standard output: `soft-scale-check.json` (every number above, per day and summarised, with the
settings and definitions) and `soft-scale-check-summary.txt`.

Exit status: 0 no deciding comparison made (in the deciding fields; the soft minimum on every
checked day, the exact scale on the days without tied smallest terms) fails by the measuring rule
and every deciding field of both functions is exercised (over all checked units together); 1 a
comparison fails (a non-finite one included) or a deciding field is not exercised; 2
could not run (a bad matrix, configuration, unit or argument; a model that is unknown, refused,
fitted on a checked unit or failing; a table missing, no verified truth build at this commit, a
table differing from it, a day that cannot be built; no checked unit has an offering day; or any
other error raised while the check runs, writing the results and the summary included, printed
with its type; the summary is printed only once both files are written). A refused or failing
model is 2 here, not 1 as in `learn_run`, because 1 is kept for the check itself failing.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import zlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import torch
from scipy import stats

from fleetreserve import dial_choice, gate_day, gate_run, learn_run, rotation_whatif, soft_scale
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import MatrixError, load
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

MATRIX = TOP / "gate" / "matrix.json"
RESULTS_NAME = "soft-scale-check.json"
SUMMARY_NAME = "soft-scale-check-summary.txt"
DAYS = 100
SEED = 1
SIGMA = 0.01
STEPS = {"exact": 1e-4, "soft": 1e-5}  # first steps (module text, rule part 4)
SHRINK = 100.0
LIMIT = 1e-3
RESOLUTION = 0.25
BELOW_ONE = 1e-6
TIE = 1e-9
CONSTANT = 1e-9
EPSILON = float(np.finfo(np.float64).eps)
FIELDS = ("departure", "cap", "need", "rate")
DECIDING = ("departure", "cap", "need")
NOT_APPLICABLE_WHOLE_HOURS = ("departure",)
KINDS = ("soft", "exact")
OUTCOMES = ("passed", "failed", "unresolved", "kink", "tied")


class NoCheck(Exception):
    """The check could not run (exit 2)."""


# --- the measuring rule (module text) -------------------------------------------------------------


def _function(kind: str):
    if kind == "soft":
        return lambda b, c, up, down: soft_scale.soft_scale(b, c, up, down, SIGMA)
    return soft_scale.exact_scale


def _autograd(f, bounds, field, direction, shape) -> tuple[float, float]:
    """(the function's value, its derivative along `direction` by autograd)."""
    x = getattr(bounds, field).clone().requires_grad_(True)
    value = f(replace(bounds, **{field: x}), *shape)
    (gradient,) = torch.autograd.grad(value, x)
    return float(value.detach()), float((gradient * direction).sum())


def _reach(bounds, field: str, n_service: int) -> torch.Tensor:
    """`L_i`: a bound on a car's slope of `b` and `p` in `field` (rule, part 1)."""
    if field == "departure":
        return bounds.rate.abs()
    if field == "rate":
        return bounds.departure.abs() + n_service
    return torch.ones_like(bounds.rate)


def term_rounding(bounds, shape, field: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(the terms, `e_k`: a bound on each computed term's rounding error; 0 for a term left out)."""
    centre, w_up, w_down = shape
    n, cars = centre.shape[-1], bounds.rate.numel()
    bits = soft_scale.subset_bits(n, centre.dtype)
    lower, upper = soft_scale.set_functions(bounds, n)
    found = soft_scale.terms(bounds, *shape)
    c_abs = centre.abs() @ bits.T
    carried = float((bounds.need.abs() + bounds.rate.abs() * bounds.departure.abs()).sum())
    moved = float((getattr(bounds, field).abs() * _reach(bounds, field, n)).sum())
    above = (cars + n) * upper.abs() + n * c_abs + moved
    below = (cars - 1) * lower.abs() + (n + 2) * carried + n * c_abs + moved
    width = torch.cat([w_up @ bits.T, w_down @ bits.T])
    finite = torch.isfinite(found)
    size = torch.cat([above, below]) / torch.where(width > 0, width, 1.0)
    bound = EPSILON * (size + (n + 2) * torch.where(finite, found, 0.0).abs())
    return found, torch.where(finite, bound, 0.0)


def value_rounding(kind: str, found: torch.Tensor, bound: torch.Tensor, value: float) -> float:
    """`e`: a bound on the rounding error of one computed value of the function (rule, part 1)."""
    finite = torch.isfinite(found)
    terms, rounding = found[finite], bound[finite]
    if kind == "exact":
        least = int(torch.argmin(terms))
        near = terms - rounding <= terms[least] + rounding[least]
        return float(rounding[near].max())
    weights = torch.softmax(-terms / SIGMA, dim=0)
    own = (terms.numel() + 3) * EPSILON * SIGMA + 2 * EPSILON * abs(value)
    return float((weights * (rounding + 2 * EPSILON * terms.abs())).sum()) + own


OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)  # the points measured, in steps (rule, part 1)


def second_differences(values: list):
    """`(S, S-, S+)` of five values at `OFFSETS` (rule, part 1): the second difference over the
    whole step and over each half of it; each is 0 where the values lie on one line."""
    behind, half_behind, middle, half_ahead, ahead = values
    return (
        ahead - 2 * middle + behind,
        middle - 2 * half_behind + behind,
        ahead - 2 * half_ahead + middle,
    )


def slope_bounds(bounds, shape, field: str, direction: torch.Tensor) -> torch.Tensor:
    """`Lambda_k`: a bound on the absolute slope of each term along `direction` in `field`,
    `sum_i |v_i| L_i / w(A)` (rule, part 1(b)); not meaningful for a term left out."""
    centre, w_up, w_down = shape
    n = centre.shape[-1]
    bits = soft_scale.subset_bits(n, centre.dtype)
    width = torch.cat([w_up @ bits.T, w_down @ bits.T])
    reach = (direction.abs() * _reach(bounds, field, n)).sum()
    return reach / torch.where(width > 0, width, 1.0)


def _soft_parts(
    points: list, step: float, slopes_bound: torch.Tensor
) -> tuple[float, float, float]:
    """(truncation, kink, hidden) of the soft minimum's allowance at `step` (rule, parts 1(b) and
    2); `points` are the five of `OFFSETS`, each (value, terms, term bounds), and `slopes_bound`
    the terms' `Lambda_k` (`slope_bounds`)."""
    finite = torch.isfinite(points[2][1])
    values = [p[1][finite] for p in points]
    behind, _, middle, _, ahead = values
    rounding = torch.stack([p[2][finite] for p in points]).amax(dim=0)
    second = torch.stack([s.abs() for s in second_differences(values)]).amax(dim=0)
    kinked = second > 4 * rounding
    straight = ~kinked
    if not bool(straight.any()):
        return 0.0, math.inf, 0.0
    # the straight terms alone: the smooth part phi_L (rule, part 1(b))
    slopes = ((ahead - behind) / (2 * step))[straight]
    log_weights = torch.log_softmax(-middle[straight] / SIGMA, dim=0)
    deviation = (slopes - (log_weights.exp() * slopes).sum()).abs()
    log_bounded = log_weights + step * deviation / SIGMA  # log pihat_k
    third = 8 * torch.exp(log_bounded + 3 * torch.log(deviation)).sum() / SIGMA**2
    hidden = 4 * torch.exp(log_bounded + torch.log(rounding[straight])).sum() / step
    # the kinked terms' weights relative to the straight terms, at -h, 0 and h, raised by rounding
    lift = (rounding[kinked] + rounding[straight].max()) / SIGMA

    def relative(at: torch.Tensor) -> torch.Tensor:
        normaliser = torch.logsumexp(-at[straight] / SIGMA, dim=0)
        return torch.exp(-at[kinked] / SIGMA - normaliser + lift)

    ends = relative(behind).sum() + relative(ahead).sum()
    reach = slopes_bound[finite]
    at_point = (relative(middle) * (reach[kinked] + reach.max())).sum()
    shift = float(SIGMA * ends / (2 * step) + at_point)
    if math.isnan(shift):  # an infinite weight times a zero bound: kinked terms dominate
        shift = math.inf
    return float(step**2 * third / 6), shift, float(hidden)


def resolvable(difference: float, allowance: float) -> bool:
    """`tau <= RESOLUTION LIMIT |D|` (rule, part 3)."""
    return allowance <= RESOLUTION * LIMIT * abs(difference)


def measure_step(kind, bounds, field, direction, shape, step: float) -> dict:
    """Differences at `step` and their allowance (rule, parts 1 and 2); `a` is not read."""
    f, base = _function(kind), getattr(bounds, field)
    points = []
    for offset in OFFSETS:
        t = offset * step
        moved = replace(bounds, **{field: base + t * direction}) if t else bounds
        found, bound = term_rounding(moved, shape, field)
        points.append((float(f(moved, *shape)), found, bound))
    values = [p[0] for p in points]
    if not all(math.isfinite(v) for v in values):
        return {"step": step, "non_finite": True}
    behind, _, middle, _, ahead = values
    e = max(value_rounding(kind, found, bound, v) for v, found, bound in points)
    difference = (ahead - behind) / (2 * step)
    left, right = (middle - behind) / step, (ahead - middle) / step
    if kind == "exact":
        rounding = 5 * e / step + 2 * EPSILON * abs(difference)
        largest = max(abs(s) for s in second_differences(values))
        truncation, kink = 0.0, max(largest - 4 * e, 0.0) / (2 * step)
    else:
        reach = slope_bounds(bounds, shape, field, direction)
        truncation, kink, hidden = _soft_parts(points, step, reach)
        rounding = e / step + hidden + 2 * EPSILON * abs(difference)
    allowance = rounding + truncation + kink
    return {
        "step": step,
        "non_finite": False,
        "difference": difference,
        "left": left,
        "right": right,
        "rounding": rounding,
        "truncation": truncation,
        "kink": kink,
        "allowance": allowance,
        "resolvable": resolvable(difference, allowance),
        "kink_shown": kink > 0,
        "shrinks": truncation + kink > rounding,
    }


def needs_smaller(first: dict) -> bool:
    """The smaller step is measured when the first is finite and shows a kink, or is not
    resolvable with its kink and truncation parts larger than its rounding part (rule, part 4)."""
    if first["non_finite"]:
        return False
    return first["kink_shown"] or (not first["resolvable"] and first["shrinks"])


def measure(kind, bounds, field, direction, shape) -> list[dict]:
    """The steps measured for one comparison: the first, and the smaller one when
    `needs_smaller`."""
    first = measure_step(kind, bounds, field, direction, shape, STEPS[kind])
    if not needs_smaller(first):
        return [first]
    return [first, measure_step(kind, bounds, field, direction, shape, STEPS[kind] / SHRINK)]


def judge(a: float, step: dict) -> str:
    """`passed`, `failed` or `unresolved` on the interval of one step (rule, part 3); a step that
    shows a kink decides nothing (`unresolved`)."""
    d, tau = step["difference"], step["allowance"]
    if step["kink_shown"] or not (math.isfinite(d) and math.isfinite(tau)):
        return "unresolved"
    ends = [d - tau, d + tau]
    widest = ends + ([0.0] if ends[0] <= 0 <= ends[1] else [])
    if all(abs(a - g) <= LIMIT * abs(g) for g in widest):
        return "passed"
    nearest = widest + ([a] if ends[0] <= a <= ends[1] else [])
    if all(abs(a - g) > LIMIT * abs(g) for g in nearest):
        return "failed"
    return "failed" if step["resolvable"] else "unresolved"


def decide(a: float, steps: list[dict]) -> tuple[str, int | None]:
    """(the outcome, the index of the step that decided it or None): rule, part 4."""
    if not math.isfinite(a) or any(s["non_finite"] for s in steps):
        return "failed", None
    for index, step in enumerate(steps):
        outcome = judge(a, step)
        if outcome != "unresolved":
            return outcome, index
    if len(steps) > 1 and all(s["kink_shown"] for s in steps):
        return "kink", None
    return "unresolved", None


def _written(value):
    """A number for the results file: None when it is not finite."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compare(kind, bounds, field, direction, shape) -> dict:
    """One comparison of autograd with finite differences (module text)."""
    f = _function(kind)
    _, auto = _autograd(f, bounds, field, direction, shape)
    steps = measure(kind, bounds, field, direction, shape)
    outcome, index = decide(auto, steps)
    non_finite = not math.isfinite(auto) or any(s["non_finite"] for s in steps)
    decided = steps[index] if index is not None else None
    error = None
    if decided is not None and decided["difference"] != 0:
        error = abs(auto - decided["difference"]) / abs(decided["difference"])
    return {
        "outcome": outcome,
        "non_finite": non_finite,
        "autograd": _written(auto),
        "decided_step": decided["step"] if decided else None,
        "resolvable": bool(decided and decided["resolvable"]),
        "relative_error": error,
        "steps": [{k: _written(v) for k, v in s.items()} for s in steps],
    }


TIED = {"outcome": "tied"}


def _tensor(values) -> torch.Tensor:
    return torch.tensor(np.asarray(values, dtype=float), dtype=torch.float64)


def day_shape(prepared) -> tuple:
    """(centre, w_up, w_down) of the certified box of a prepared day."""
    policy = prepared.certificate.policy
    return _tensor(policy.m), _tensor(policy.w), _tensor(policy.w)


def smallest_terms(bounds, shape) -> int:
    """How many finite terms lie within `TIE * max(1, |scale|)` of the smallest (1: unique)."""
    found = soft_scale.terms(bounds, *shape)
    finite = found[torch.isfinite(found)]
    least = float(finite.min())
    return int((finite <= least + TIE * max(1.0, abs(least))).sum())


def check_day(prepared, rng: np.random.Generator) -> dict:
    """Every number of one offering day (module text)."""
    bounds = soft_scale.forecast_bounds(prepared.day)
    shape = day_shape(prepared)
    scale = float(soft_scale.exact_scale(bounds, *shape))
    soft = float(soft_scale.soft_scale(bounds, *shape, SIGMA))
    count = int(soft_scale.term_count(shape[1], shape[2]))
    if count == 0:  # an offering day has a positive width, so this is not expected
        raise NoCheck(f"the certified shape of {prepared.site} {prepared.date} has no width")
    violation = max(soft - scale, scale - soft - SIGMA * math.log(count), 0.0)
    smallest = smallest_terms(bounds, shape)
    comparisons = {kind: {} for kind in KINDS}
    for field in FIELDS:
        direction = _tensor(rng.standard_normal(bounds.rate.numel()))
        direction = direction / direction.norm()
        for kind in KINDS:
            if kind == "exact" and smallest > 1:
                comparisons[kind][field] = dict(TIED)
            else:
                comparisons[kind][field] = compare(kind, bounds, field, direction, shape)
    return {
        "site": prepared.site,
        "date": prepared.date,
        "cars": prepared.n_cars,
        "steered_cars": len(prepared.steered),
        "n_service": prepared.day.n_service,
        "terms": count,
        "smallest_terms": smallest,
        "scale": scale,
        "soft_scale": soft,
        "centre_inside": bool(soft_scale.centre_inside(bounds, shape[0])),
        "sandwich_violation": violation,
        "certified_value_kwh": float(prepared.value),
        "scaled_value_kwh": scale * float(prepared.value),
        "comparisons": comparisons,
    }


# --- one unit ------------------------------------------------------------------------------------


def _unit_seed(seed: int, unit: str) -> list[int]:
    return [seed, zlib.crc32(unit.encode())]


def sample_unit(unit_days, forecaster, configuration, settings, days: int, seed: int):
    """(checked days with their prepared days, abstentions by reason, days examined)."""
    order = np.random.default_rng(_unit_seed(seed, unit_days.unit)).permutation(len(unit_days.days))
    rng = np.random.default_rng([*_unit_seed(seed, unit_days.unit), 1])
    checked, abstained, examined = [], Counter(), 0
    for index in order:
        if len(checked) == days:
            break
        examined += 1
        prepared = gate_day.prepare_day(
            unit_days.days[int(index)], forecaster, configuration, settings
        )
        if prepared.abstains:
            abstained[prepared.abstain_reason] += 1
            continue
        checked.append((check_day(prepared, rng), prepared))
    return checked, dict(sorted(abstained.items())), examined


def _pearson(x, y) -> float:
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def constant(column) -> bool:
    """True when the column's range is at most `CONSTANT * max(1, largest |value|)`."""
    values = np.asarray(column, float)
    return float(values.max() - values.min()) <= CONSTANT * max(1.0, float(np.abs(values).max()))


def _correlations(x, y) -> dict:
    """Pearson's and Spearman's (Pearson's on ranks, ties at their mean rank) correlations;
    None with fewer than 3 days or a column constant up to `CONSTANT` (`constant`)."""
    if len(x) < 3 or constant(x) or constant(y):
        return {"pearson": None, "spearman": None}
    return {
        "pearson": _pearson(x, y),
        "spearman": _pearson(stats.rankdata(x), stats.rankdata(y)),
    }


def tracking(rows: list[dict]) -> dict:
    """Lambda beside the certified value (module text)."""
    scale = [r["scale"] for r in rows]
    value = [r["certified_value_kwh"] for r in rows]
    scaled = [r["scaled_value_kwh"] for r in rows]
    return {
        "days": len(rows),
        "scale_with_certified_value": _correlations(scale, value),
        "scaled_value_with_certified_value": _correlations(scaled, value),
        "below_one": sum(s < 1 - BELOW_ONE for s in scale),
        "smallest_scale": min(scale) if scale else None,
        "median_scale": statistics.median(scale) if scale else None,
    }


def _allowance_share(c: dict) -> float:
    """`tau / (LIMIT |D|)` at the deciding step of a resolvable comparison (at most RESOLUTION)."""
    step = next(s for s in c["steps"] if s["step"] == c["decided_step"])
    return step["allowance"] / (LIMIT * abs(step["difference"]))


def whole_hour_departures(configuration) -> bool:
    """True when the configuration's forecast departures are whole hours: the oracle departure
    (the true departure hour) or the floored one (`gate_config`)."""
    return configuration.departure == "oracle" or configuration.departure_floor == "floored"


def deciding_fields(configuration) -> tuple[str, ...]:
    """`DECIDING`, without the departure on a configuration with whole-hour forecast departures,
    where it is not applicable (module text)."""
    if whole_hour_departures(configuration):
        return tuple(field for field in DECIDING if field not in NOT_APPLICABLE_WHOLE_HOURS)
    return DECIDING


def errors(rows: list[dict], deciding: tuple[str, ...] = DECIDING) -> dict:
    """Per function and field, the comparisons by outcome (a tied one is counted in `tied` and in
    nothing else) and over the resolvable ones the largest relative error and allowance share."""
    out = {}
    for kind in KINDS:
        out[kind] = {}
        for field in FIELDS:
            every = [r["comparisons"][kind][field] for r in rows]
            made = [c for c in every if c["outcome"] != "tied"]
            resolved = [c for c in made if c["resolvable"]]
            counts = Counter(c["outcome"] for c in every)
            out[kind][field] = {
                "deciding": field in deciding,
                "compared": len(made),
                **{outcome: counts.get(outcome, 0) for outcome in OUTCOMES},
                "resolvable": len(resolved),
                "at_smaller_step": sum(
                    c["decided_step"] is not None and c["decided_step"] < STEPS[kind] for c in made
                ),
                "non_finite": sum(c["non_finite"] for c in made),
                "largest_error": max((c["relative_error"] for c in resolved), default=None),
                "largest_allowance_share": max(map(_allowance_share, resolved), default=None),
            }
    return out


def not_exercised(found: dict) -> list[str]:
    """`<function> <field>` for each deciding field of each function with no resolvable
    comparison (`resolvable` 0 in `errors`)."""
    return [
        f"{kind} {field}"
        for kind in KINDS
        for field in FIELDS
        if found[kind][field]["deciding"] and found[kind][field]["resolvable"] == 0
    ]


def failed_comparisons(rows: list[dict], deciding: tuple[str, ...] = DECIDING) -> int:
    """Deciding comparisons that fail by the measuring rule, the non-finite ones included."""
    return sum(
        r["comparisons"][kind][field]["outcome"] == "failed"
        for r in rows
        for kind in KINDS
        for field in deciding
    )


def tied_days(rows: list[dict]) -> int:
    """Checked days whose smallest finite terms tie (`smallest_terms` above 1)."""
    return sum(r["smallest_terms"] > 1 for r in rows)


# --- one batch timed -----------------------------------------------------------------------------


def batch_of(prepared_days: list) -> list:
    """The checked days with the most common number of service hours (ties: fewer hours)."""
    if not prepared_days:
        return []
    counts = Counter(p.day.n_service for p in prepared_days)
    n = min(counts, key=lambda k: (-counts[k], k))
    return [p for p in prepared_days if p.day.n_service == n]


def _stacked(prepared_days, device, dtype):
    singles = [soft_scale.forecast_bounds(p.day) for p in prepared_days]
    width = max(b.rate.numel() for b in singles)

    def pad(values):
        return torch.cat([values, torch.zeros(width - values.numel(), dtype=values.dtype)])

    fields = {
        name: torch.stack([pad(getattr(b, name)) for b in singles]).to(device, dtype)
        for name in ("departure", "rate", "cap", "need")
    }
    shapes = [day_shape(p) for p in prepared_days]
    shape = tuple(torch.stack([s[k] for s in shapes]).to(device, dtype) for k in range(3))
    return fields, shape


def _sync(device: str):
    if device == "mps":
        torch.mps.synchronize()


def time_batch(prepared_days: list, device: str, dtype) -> dict:
    """Seconds of one forward and backward pass of the soft minimum over the batch (module
    text): a warm-up call, then the timed call."""
    fields, shape = _stacked(prepared_days, device, dtype)
    seconds = []
    for _ in range(2):
        leaves = {k: v.clone().requires_grad_(True) for k, v in fields.items()}
        bounds = soft_scale.Bounds(**leaves)
        _sync(device)
        start = time.perf_counter()
        soft_scale.soft_scale(bounds, *shape, SIGMA).sum().backward()
        _sync(device)
        seconds.append(time.perf_counter() - start)
    return {
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "batch_days": len(prepared_days),
        "padded_cars": int(fields["rate"].shape[-1]),
        "n_service": int(shape[0].shape[-1]),
        "warm_up_seconds": seconds[0],
        "seconds": seconds[1],
    }


def batch_timing(prepared_days: list) -> dict:
    batch = batch_of(prepared_days)
    if not batch:
        return {"cpu": None, "mps": None}
    timing = {"cpu": time_batch(batch, "cpu", torch.float64)}
    if torch.backends.mps.is_available():
        timing["mps"] = time_batch(batch, "mps", torch.float32)
    else:
        timing["mps"] = "not available"
    return timing


# --- the whole check -----------------------------------------------------------------------------


def settings_record(days: int, seed: int, deciding: tuple[str, ...] = DECIDING) -> dict:
    return {
        "days_per_unit": days,
        "seed": seed,
        "sigma": SIGMA,
        "steps": dict(STEPS),
        "shrink": SHRINK,
        "limit": LIMIT,
        "resolution": RESOLUTION,
        "tie": TIE,
        "constant": CONSTANT,
        "epsilon": EPSILON,
        "fields": list(FIELDS),
        "deciding_fields": list(deciding),
        "not_applicable": [field for field in DECIDING if field not in deciding],
        "functions": list(KINDS),
    }


DEFINITIONS = {
    "scale": (
        "T1(b) exact scale of the certified shape (centre m, widths w up and down) on the "
        "steered cars' forecast set, hour 0 and recovery hours free (soft_scale.exact_scale)"
    ),
    "soft_scale": "the soft minimum of the same terms at sigma (soft_scale.soft_scale)",
    "steps": (
        "per measured step h (the function at -h, -h/2, 0, h/2 and h): difference D (central), "
        "left and right (one-sided), and the allowance tau = rounding + truncation + kink, a bound "
        "on |D - derivative| at a step that shows no kink, except where kinks whose effects on "
        "the three second differences cancel or lie within rounding lie in the step (module "
        "text, the measuring rule, parts 1 and 2, 'Not covered'); kink_shown: kink above 0 (a "
        "second difference over the step or either half of it beyond rounding; for the soft "
        "minimum, of a term whose weight at -h, 0 or h does not underflow), and such a step "
        "decides nothing; shrinks: truncation + kink above rounding; the smaller step h / shrink "
        "is measured when the first step shows a kink, or is not resolvable and shrinks"
    ),
    "resolvable": (
        "at the deciding step (which shows no kink) tau <= resolution * limit * |D|: a true "
        "relative error above limit cannot pass there and one at most 0.499 limit cannot fail, "
        "except where kinks the rule does not cover lie in the step (module text, 'Not covered')"
    ),
    "outcome": (
        "decided at the first step that shows no kink and decides: passed: |autograd - g| <= "
        "limit |g| for every g in [D - tau, D + tau]; failed: above for every such g, or a "
        "resolvable comparison not passed, or a non-finite value; unresolved: neither, at every "
        "step measured; kink: a kink shown at both steps, neither passed nor failed; tied: the "
        "exact scale on a day whose smallest terms tie (smallest_terms above 1), not compared, "
        "neither passed nor failed, nothing else written"
    ),
    "deciding_fields": (
        "the fields whose comparisons decide the check: departure, cap and need, without the "
        "departure on a configuration with whole-hour forecast departures"
    ),
    "not_applicable": (
        "the departure on a configuration with whole-hour forecast departures (oracle or "
        "floored): every point is a kink in that direction; its comparisons are reported only"
    ),
    "decided_step": "the step whose interval decided a passed or failed outcome, else null",
    "relative_error": "|autograd - D| / |D| at the deciding step (null when undecided or D is 0)",
    "non_finite": "a value, derivative or difference that is NaN or infinite: written null, failed",
    "largest_allowance_share": "the largest tau / (limit |D|) over the resolvable comparisons",
    "at_smaller_step": "comparisons decided at the smaller step",
    "not_exercised": (
        "deciding fields, per function, with no resolvable comparison: the check fails"
    ),
    "tied_days": "checked days whose smallest finite terms tie (smallest_terms above 1)",
    "sandwich_violation": "max(soft - scale, scale - soft - sigma ln N, 0), N the finite terms",
    "tracking": (
        "a reading of the plan's 'tracking': correlations over checked days of the scale, and of "
        "scale times the certified value, with the certified value at scale 1 "
        "(gate_day.PreparedDay.value)"
    ),
    "below_one": "checked days whose scale is below 1 - 1e-6 (the certified box lies in the set)",
    "sample": (
        "per unit, the first days_per_unit offering days of a seeded order of its counting days; "
        "the abstaining days met before them are counted by reason"
    ),
}


def run(units: dict, factory, matrix, configuration, *, days: int, seed: int) -> dict:
    settings = gate_run.settings(matrix)
    deciding = deciding_fields(configuration)
    unit_entries, rows_all, prepared_all = {}, [], []
    for name, unit_days in units.items():
        forecaster = factory(unit_days, matrix)
        checked, abstained, examined = sample_unit(
            unit_days, forecaster, configuration, settings, days, seed
        )
        rows = [row for row, _ in checked]
        rows_all += rows
        prepared_all += [p for _, p in checked]
        unit_entries[name] = {
            "counting_days": len(unit_days.days),
            "examined_days": examined,
            "abstained": abstained,
            "checked_days": len(rows),
            "fewer_than_asked": len(rows) < days,
            "tied_days": tied_days(rows),
            "errors": errors(rows, deciding),
            "largest_sandwich_violation": max(
                (r["sandwich_violation"] for r in rows), default=None
            ),
            "failed_comparisons": failed_comparisons(rows, deciding),
            "tracking": tracking(rows),
            "days": rows,
        }
    if not rows_all:
        raise NoCheck("no checked unit has an offering day in the sample")
    failed = failed_comparisons(rows_all, deciding)
    found = errors(rows_all, deciding)
    missing = not_exercised(found)
    return {
        "label": "T8 check (plan section 7)",
        "definitions": dict(DEFINITIONS),  # a copy: `rotation_whatif.mark` adds to it
        "settings": settings_record(days, seed, deciding),
        "configuration": configuration.name,
        "units": unit_entries,
        "checked_days": len(rows_all),
        "tied_days": tied_days(rows_all),
        "errors": found,
        "largest_sandwich_violation": max(r["sandwich_violation"] for r in rows_all),
        "tracking": tracking(rows_all),
        "batch_timing": batch_timing(prepared_all),
        "failed_comparisons": failed,
        "not_exercised": missing,
        "passed": failed == 0 and not missing,
    }


def _number(value) -> str:
    return "none" if value is None else f"{value:.3g}"


def summary_lines(document: dict) -> list[str]:
    s = document["settings"]
    steps = ", ".join(f"{kind} {step}" for kind, step in s["steps"].items())
    deciding = s["deciding_fields"]
    named = ", ".join(deciding[:-1]) + " and " + deciding[-1] if len(deciding) > 1 else deciding[0]
    lines = [
        *rotation_whatif.summary_head(document),
        f"T8 check: model {document['model']}, configuration {document['configuration']}, "
        f"sigma {s['sigma']}, first steps {steps} (smaller: divided by {s['shrink']:g}), "
        f"limit {s['limit']}, resolvable at an allowance of {s['resolution']} limit |D|.",
        f"Deciding: {named}, for the soft minimum on every day and the exact scale "
        "on days whose smallest term is unique; rate is reported only.",
    ]
    for field in s["not_applicable"]:
        lines.append(
            f"NOT APPLICABLE: {field}, on this configuration's whole-hour forecast departures "
            "(every point is a kink in that direction); its comparisons are reported only."
        )
    for name, unit in document["units"].items():
        lines.append(
            f"{name}: {unit['checked_days']} days checked of {unit['examined_days']} examined "
            f"({unit['counting_days']} counting); abstained {unit['abstained'] or 'none'}"
            + ("; FEWER THAN ASKED" if unit["fewer_than_asked"] else "")
        )
    lines.append(
        f"Tied days: {document['tied_days']} of {document['checked_days']} checked days have tied "
        "smallest terms; the exact scale is not compared on them (a kink at or very near the "
        "point: no derivative there, or one the steps used cannot measure)."
    )
    lines.append(
        "Comparisons over all checked days (compared | tied | resolvable | passed | failed | "
        "unresolved | kink | decided at the smaller step | not finite | largest relative error "
        "and largest allowance share over the resolvable ones):"
    )
    for kind, fields in document["errors"].items():
        for field, e in fields.items():
            mark = "" if e["deciding"] else " (not deciding)"
            lines.append(
                f"  {kind} {field}{mark}: {e['compared']} | {e['tied']} | {e['resolvable']} | "
                f"{e['passed']} | {e['failed']} | {e['unresolved']} | {e['kink']} | "
                f"{e['at_smaller_step']} | {e['non_finite']} | "
                f"{_number(e['largest_error'])}, {_number(e['largest_allowance_share'])}"
            )
    t = document["tracking"]
    lines += [
        f"Largest sandwich violation: {_number(document['largest_sandwich_violation'])}.",
        f"Tracking (a reading): scale with certified value {t['scale_with_certified_value']}; "
        f"scale x value with certified value {t['scaled_value_with_certified_value']}; "
        f"days with scale below 1: {t['below_one']}; "
        f"smallest scale {_number(t['smallest_scale'])}.",
    ]
    for device, timing in document["batch_timing"].items():
        if isinstance(timing, dict):
            lines.append(
                f"Batch on {device} ({timing['dtype']}): {timing['batch_days']} days, "
                f"{timing['padded_cars']} cars, {timing['n_service']} hours: "
                f"{timing['seconds']:.4f} s (warm-up {timing['warm_up_seconds']:.4f} s)."
            )
        else:
            lines.append(f"Batch on {device}: {timing}.")
    for pair in document["not_exercised"]:
        lines.append(f"FIELD NOT EXERCISED: {pair} (no resolvable comparison).")
    verdict = "PASSED" if document["passed"] else "FAILED"
    lines.append(
        f"{verdict}: {document['failed_comparisons']} deciding comparisons failed; "
        f"{len(document['not_exercised'])} deciding fields not exercised."
    )
    return lines


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def prepare_inputs(args):
    """(matrix, configuration, model, units, factory, inputs); raises `NoCheck`."""
    try:
        if args.days < 1:
            raise MatrixError("--days must be at least 1")
        matrix = load(args.matrix)
        configuration = matrix.configuration(args.configuration)
        names = tuple(args.units) if args.units else tuple(matrix.gate_units)
        stray = [u for u in names if u not in matrix.gate_units]
        if stray:
            raise MatrixError(f"units {stray} are not gate units of the matrix")
        model = learn_run.load_model(args.model, matrix)
        fit_units = tuple(model.fit_units)
        learn_run.check_model(args.model, model, fit_units, matrix.gate_units)
    except (MatrixError, learn_run.Refused) as problem:
        raise NoCheck(str(problem)) from problem
    needed = tuple(dict.fromkeys((*names, *fit_units)))
    try:
        inputs = gate_run.check_inputs(args.truth_run, args.tables, needed)
        loaded = {
            name: load_unit_days(
                args.tables,
                name,
                rho=matrix.rho,
                no_fit=matrix.no_fit,
                small_fleet=matrix.small_fleet,
            )
            for name in needed
        }
    except (CouldNotRun, ValueError) as problem:
        raise NoCheck(str(problem)) from problem
    units = {name: loaded[name] for name in names}
    fit_frames = {name: loaded[name].cars for name in fit_units}
    scored = replace(matrix, gate_units=names)
    try:
        factory, _ = dial_choice.make_factory(
            args.model, model, scored, units, fit_frames, configuration.kappa
        )
    except (learn_run.Refused, ValueError) as problem:
        raise NoCheck(f"the model could not be used: {problem}") from problem
    return matrix, configuration, model, units, factory, inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--units", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)  # LightGBM shares the process (hazard_net)
    try:
        matrix, configuration, model, units, factory, inputs = prepare_inputs(args)
        document = run(units, factory, matrix, configuration, days=args.days, seed=args.seed)
        document["model"] = model.name
        document["inputs"] = inputs
        document["matrix"] = {
            "file": args.matrix.name,
            "sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        }
        rotation_whatif.mark(document, matrix, tuple(model.fit_units))
        text = "\n".join(summary_lines(document)) + "\n"
        run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
        if run_dir:
            folder = Path(run_dir)
            (folder / RESULTS_NAME).write_text(_json(document))
            (folder / SUMMARY_NAME).write_text(text)
    except NoCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except Exception as problem:  # anything else that stops the check is "could not run" too
        print(f"COULD NOT RUN: {type(problem).__name__}: {problem}")
        return 2
    print(text, end="")
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
