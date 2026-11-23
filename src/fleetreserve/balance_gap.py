"""The balance gap of a certificate's policy, measured exactly (`fixed_plan_loss`, module text, V8).

A certificate shares the box out exactly when its cars' planned powers add up to the box's centre
and their gains to the box's gain; where it does not, replay can miss the request by the balance
gap `B_t` (`fixed_plan_loss`, module text, V2, V4 and V8). This module measures by how much a
policy misses that, per service hour, in the terms of `fixed_plan_loss`'s module text ("Terms",
the term `B_t`, and V8). It decides nothing.

Definitions. The policy `(m, w, g, G)` of a certificate (`policy`, module text: `m`, `w` per
service hour, `g` and `G` per car and hour, the cars those of the certificate), service hours `t`
from 1 to `n`:

- the centre gap `c_t = |sum_i g_it - m_t|`;
- the gain gap `gamma_t = |sum_i G_it - W_t|_1 = sum_{s=1..n} |sum_i G_its - [s = t] w_t|`, with
  `W_t = w_t e_t` the request's own gain: the box is `y = m + W zeta` with `W = diag(w)` (plan
  section 2), whose hour `t` has the gain `w_t` on the request of hour `t` and none on another
  (the certificate's rows `sum_i G_i(t, s) = [s = t] w_t`, `certificate`, module text);
- `B_t(beta) = c_t + beta gamma_t`, the term `B_t` of `fixed_plan_loss`'s "Terms" at scale `beta`
  (the miss term of V2 and V4); at scale 1 it is V8's "gap in the centres plus the gap in the
  summed gains". It does not fall as `beta` rises (`gamma_t >= 0`).

Hour 0 carries no request (`policy`, module text: the box covers the service hours only), so it
has no gap. The decision hour's draw (`fixed_plan_loss.drawn_policy`) changes the powers of hour 0
only, so the drawn policy has these same gaps.

Arithmetic. Every gap is computed exactly in rational arithmetic (`fractions.Fraction`) on the
doubles the policy holds, as V8 is stated "in exact arithmetic" on the numbers the code holds
(`fixed_plan_loss`, module text); a gap is written as the double nearest to its exact value (one
rounding), and every comparison below is made exactly, with the constants' doubles. A policy whose
cars share the box out exactly in the doubles it holds has every gap exactly 0. A sum of `k` terms
in floating point can be off by up to `gamma_{k-1}` times the summed absolute values (the standard
model of `fixed_plan_loss`'s "Terms", without underflow), which can be above the levels below.

Levels (each compared exactly; V8 and the constants it names):

- a gain gap above `policy.ZERO_GAIN` (1e-12 kWh), the gain at or below which a row counts as not
  depending on the requests (`policy`, module text);
- a gain gap above `FAILURE_KWH / VALIDITY_LIMIT`, the exact quotient of the two doubles (1e-9
  plus 4.8e-26 kWh): at the validity limit such a gain gap alone makes `B_t` above `FAILURE_KWH`
  (V8: "at the limit a gap in the summed gains alone above `FAILURE_KWH / VALIDITY_LIMIT` ... can
  fail the hour"). The double nearest 1e-9 is above it, by 1.4e-26;
- a gain gap above `policy.ROW_TOLERANCE` (1e-7 kWh);
- an hour that "can fail by the gap alone" at scale `beta`: `B_t(beta) > FAILURE_KWH` (V8: "a
  `B_t` above `FAILURE_KWH` can fail the hour"). As `B_t` does not fall as `beta` rises, an hour
  that cannot at `beta` cannot at any scale up to `beta`. The other errors of V4's condition
  (`E_t`, `R_t`) are not measured here.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from fleetreserve.policy import ROW_TOLERANCE, VALIDITY_LIMIT, ZERO_GAIN, AffinePolicy
from fleetreserve.replay import FAILURE_KWH

FAILURE = Fraction(FAILURE_KWH)
LIMIT = Fraction(VALIDITY_LIMIT)
# The levels a gain gap is counted against (module text), by name, as exact numbers.
GAIN_LEVELS = {
    "zero_gain": Fraction(ZERO_GAIN),
    "failure_over_limit": FAILURE / LIMIT,
    "row_tolerance": Fraction(ROW_TOLERANCE),
}
SCALE_1 = "scale_1"
AT_LIMIT = "validity_limit"
FIXED_SCALES = {SCALE_1: 1.0, AT_LIMIT: VALIDITY_LIMIT}  # the scales every hour is read at


@dataclass(frozen=True)
class HourGap:
    """The exact gaps of one service hour (module text)."""

    hour: int
    centre: Fraction
    gain: Fraction

    def miss(self, beta: float) -> Fraction:
        """`B_t(beta) = c_t + beta gamma_t`, exactly, `beta` a finite double."""
        return self.centre + Fraction(beta) * self.gain

    def can_fail_alone(self, beta: float) -> bool:
        """`B_t(beta) > FAILURE_KWH` (module text)."""
        return self.miss(beta) > FAILURE

    @property
    def shared_exactly(self) -> bool:
        return self.centre == 0 and self.gain == 0


def _sum(values) -> Fraction:
    return sum((Fraction(float(v)) for v in values), Fraction(0))


def hour_gaps(policy: AffinePolicy) -> tuple[HourGap, ...]:
    """The centre and gain gaps of every service hour of `policy`, exactly (module text)."""
    n = policy.n_service
    out = []
    for t in range(1, n + 1):
        centre = abs(_sum(policy.g[:, t]) - Fraction(float(policy.m[t - 1])))
        gain = Fraction(0)
        for s in range(1, n + 1):
            own = Fraction(float(policy.w[t - 1])) if s == t else Fraction(0)
            gain += abs(_sum(policy.G[:, t, s - 1]) - own)
        out.append(HourGap(t, centre, gain))
    return tuple(out)


def as_float(value: Fraction | None) -> float | None:
    """The double nearest to `value` (`Fraction.__float__` divides two whole numbers, a division
    Python rounds correctly)."""
    return None if value is None else float(value)


def tally(gaps: Iterable[HourGap], scales: Mapping[str, float | None]) -> dict:
    """Counts over the hours `gaps`: the largest gaps, the hours shared out exactly, the hours
    whose gain gap is above each level of `GAIN_LEVELS`, and at scale 1, at the validity limit and
    at each named scale of `scales` (None: not given) the largest `B_t` and the hours that can fail
    by the gap alone (module text)."""
    gaps = list(gaps)
    read_at = dict(FIXED_SCALES) | {k: v for k, v in scales.items() if v is not None}

    def largest(values):
        values = list(values)
        return as_float(max(values)) if values else None

    return {
        "hours": len(gaps),
        "hours_shared_exactly": sum(g.shared_exactly for g in gaps),
        "largest_centre_gap_kwh": largest(g.centre for g in gaps),
        "largest_gain_gap_kwh": largest(g.gain for g in gaps),
        "gain_gap_above": {
            name: sum(g.gain > level for g in gaps) for name, level in GAIN_LEVELS.items()
        },
        "at_scales": {
            name: {
                "scale": beta,
                "largest_b_kwh": largest(g.miss(beta) for g in gaps),
                "hours_can_fail_alone": sum(g.can_fail_alone(beta) for g in gaps),
            }
            for name, beta in read_at.items()
        },
    }
