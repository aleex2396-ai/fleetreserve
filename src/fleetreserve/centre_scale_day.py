"""One day of the what-if in which the scale also moves the planned charging toward the front-loaded
plan (`centre_scale_whatif`): the front-loaded plan, the mixed policy and its replay.

Why. The scale beta of plan section 1 multiplies the box's widths and the gains; the planned
charging (the centre `g_i` of each car) stays as certified, so the risk of an early departure does
not fall with beta (plan section 6, protocol: "The scale beta alone cannot remove the risk of an
early departure, the centre g_i not scaling with it"). This what-if moves the centre with the
scale as well, toward a plan fixed at the decision time, and measures what the calibration then
gives. It is a measurement; it changes nothing in the method, and whether the method's family
changes is the owner's decision.

The front-loaded plan (`front_loaded_plan`; the plan the mix is made with, chosen for this
what-if; it is not a plan of the plan text, and its own driver loss need not be 0). Each steered
car, hour by hour from hour 0 to hour `n` in time order, draws its forecast power limit
`r_hat f_t` until its forecast cap `e_cap` is reached: `s_t = min(r_hat f_t, e_cap - sum_{u<t}
s_u)`. It is the forecast-limit form of the plan's own model of unmanaged charging (plan section
1, recovery hours: "cars charge at full power until full or gone"), and it needs nothing unknown at
the decision time. The recovery hours are not planned here: there the replay charges each car at
its full true rate until it is full or gone, as for every plan (`replay`).

Why the front-loaded plan meets every row the day was certified on (`PreparedDay.rows`):

- forecast rows (`carsets.forecast_rows`): `0 <= s_t <= r_hat f_t` and `sum_{u<=t} s_u <= e_cap`
  by construction; its window energy is `min(sum_t r_hat f_t, e_cap)`, the largest the power and
  stored-energy rows allow (`forecast_coverage.largest_window_energy`), so it meets the need row
  whenever the forecast set is not empty, which every steered car's is (`gate_day`, step 2);
- driver-risk rows (`driver_risk`): with weights `w_t = min{1, f_t(D_k) / f_t(D)}`, the energy held
  by `D_k` is `E_k(u) = sum_t w_t u_t + supply_k` and `need_k = min{need, the largest E_k}`, the
  largest found by filling hours in order of weight, largest first (ties in time order), each up
  to its power limit, the total at most the cap (`driver_risk._reach`). Over the hours where the
  car may draw (`f_t(D) > 0`) the weights do not increase with `t` (1 before `D_k`, the share
  `D_k - t` in the hour holding `D_k`, 0 after it); an hour with `f_t(D) = 0` has weight 1 but a
  power limit of 0. So that fill takes exactly `s`, `E_k(s)` is the largest `E_k`, every
  `need_k - E_k(s) <= 0`, and every row `sum_{k in A} pi_k (need_k - E_k(s)) <= epsilon need`
  holds, at every dial.

The two sums are formed in different orders, so a row can be met only to rounding (a few 1e-15
kWh below its bound); a day is "inside" when every row's slack is at least `-SET_TOLERANCE`
(`front_loaded_inside`). A day that is not inside (reading C2) is replayed unmixed at every scale
and reported by the command ("front-loaded plan outside the set: N days", with its name).

The mixed policy with weight `v` in `(0, 1]` on the certified plan (`mixed_policy`), from the
certified `(m, w, g, G)` and the front-loaded plan `s`: `g_v = (1 - v) s + v g`, gains `G`, `m_v =
(1 - v) sum_i s_i + v m` on the service hours, widths `w`. It is run at the scale `beta`, as every
policy is: car `i` draws `(1 - v) s_i + v g_i + beta G_i zeta` and the request is `m_v + beta w
zeta`. At `v = 1` the arrays are the certified policy's. Above 1 the mix gives `s` a negative
weight and is not defined (reading C1).

The room factor `c >= 1` (`mix_weight`): at scale `beta` the weight is `v = min(1, c beta)`. With
`c = 1` it is `beta` itself: car `i` draws `(1 - beta) s_i + beta (g_i + G_i zeta)`, the mix as
first built. With `c > 1` the centre moves toward the certified plan faster than the widths grow:
each car keeps the share `v` of its certified room while the box asks for the share `beta` of the
certified widths. A factor with `c beta >= 1` gives `v = 1`, so the arrays are the certified ones
and the replay is the unmixed policy's at that scale (the gate's, rung by rung; tested). At
`beta = 1` every factor gives `v = 1`.

What the figures carry, for `beta` in `[0, 1]` and every factor `c >= 1` (stated here, not reviewed
as a proof):

- T2 (a), (b) on the forecast sets, the narrower box about the moved centre. At scale `beta` car
  `i` draws `u = (1 - v) s_i + v g_i + beta G_i zeta` with `v = min(1, c beta) >= beta` (as `c >=
  1` and `beta <= 1`). Write it as `u = (1 - v) s_i + v (g_i + G_i zeta')` with `zeta' = (beta / v)
  zeta`. Since `0 < beta / v <= 1`, `zeta'` lies in `[-1, 1]^n` whenever `zeta` does, and the
  certificate puts `g_i + G_i zeta'` in car `i`'s forecast set for every such `zeta'`; `s_i` is in
  the same set (above), and the set is convex, so `u` is in it, on every path. Row by row: the
  worst case of `a . u <= b` over `zeta` in `[-1, 1]^n` is `(1 - v) a . s_i + v a . g_i + beta |a
  G_i|_1 <= (1 - v) a . s_i + v (a . g_i + |a G_i|_1) <= (1 - v) b + v b = b`, from `beta <= v`,
  `a . s_i <= b` and `a . g_i + |a G_i|_1 <= b` (the certificate). So a box narrower than the
  `v`-mix's own (half-widths `beta w` against `v w`) about the same centre meets every row the
  `v`-mix meets: it asks less of every car. The driver-risk rows are rows of the same linear form
  in `u`, so the argument covers them. It fails only where `beta > v`, which needs `c < 1` or
  `beta > 1`, both refused (readings C1, C4), and only to the solver's and the rows' tolerances.
  The box is shared out exactly: `sum_i g_v,i = m_v` and `sum_i G_i = W`.
- T2 (c1), (c2): on `E'_i` car `i`'s forecast limits lie inside its true limits, so it draws its
  plan, which lies in its forecast set on every path, and gets at least `(1 - rho) e_i`; on `E'`
  every car draws its plan and every request `m_v + beta w zeta` is met.
- T6: for each factor the family is fixed at the decision time (the forecasts, the certificate,
  `c` and `beta`); after the running maximum the losses rise with `beta`, and a day offers nothing
  and loses nothing at `beta <= 0` by the plan's definition (`gate_day.evaluate`). So the
  calibration's guarantee holds for each factor's family on its own, with the gate's readings of
  `crc` (the running maximum over the evaluated scales, R7). As `beta` tends to 0 the losses tend
  to the front-loaded plan's own losses, not to 0: the floor the mix can reach (the command
  reports it at the smallest grid scale, where the weight is `min(1, c x 0.05)`). The guarantee
  is per factor: choosing a factor after seeing the calibrated figures of several is a selection
  this text does not cover.
- The box is no longer "the certified box scaled by beta about its midpoint" (plan section 1):
  its centre `m_v` moves with `beta`, and only its widths `beta w` scale. Its value is still
  `beta` times the certified value.

The replay (`evaluate`): `day_replay.replay_day` of the mixed policy at `beta` with the rungs up to
the one-way re-solve (the fixed plan is `plans.Plan.from_policy` of the mixed policy, so the
one-way rung's predecessor and guard are the mixed plan), on the day's audited paths, with the
gate's failure rule, re-solve rule, time limit, departure bounds and losses, read as
`gate_day.evaluate` reads them (per rung the largest slot and driver loss over the paths, the first
failures, the energies at departure, the re-solves by outcome, the stay-to-bound check). The
two-way rung is not replayed and T5 is not checked here (`paths_checked` 0, no breaks). An
abstaining day and `beta <= 0` are `gate_day.evaluate`'s: losses 0, no path. A re-solve that times
out leaves the rung on its predecessor's plan and is counted by outcome, as in the gate.

Readings, each with the alternative not taken:

C1. The mix is defined for weights and scales in `(0, 1]` only; a replay above 1 is refused.
    [Alternative: extrapolate `(1 - beta) s + beta g` above 1, which leaves the convex hull of the
    two plans and carries none of the guarantees above.]
C2. A day whose front-loaded plan breaks a row by more than `SET_TOLERANCE` is replayed unmixed
    (its losses are the gate's at every scale) and named. [Alternative: leave the day out, which
    changes the calibration days and the split's counts.]
C3. `SET_TOLERANCE` is `policy.ROW_TOLERANCE` (1e-7 kW or kWh), the tolerance within which the
    code base already calls a row met. [Alternative: 0, which calls rounding a break.]
C4. A room factor is a finite number at least 1; below 1 the weight `c beta` falls under `beta`,
    the box is wider than the mix's own and the argument above does not hold, so it is refused.
    [Alternative: allow `c < 1` as a what-if without the guarantee.]
"""

import math
import time
from collections import Counter
from dataclasses import replace

import numpy as np

from fleetreserve import gate_day, stay_check
from fleetreserve.day import Day
from fleetreserve.day_replay import replay_day
from fleetreserve.gate_day import DayEvaluation, PreparedDay, RungEvaluation
from fleetreserve.lp import OTHER_SECONDS
from fleetreserve.policy import ROW_TOLERANCE, AffinePolicy
from fleetreserve.rungs import RUNGS

SET_TOLERANCE = ROW_TOLERANCE  # reading C3
RUNG = RUNGS[1]  # the one-way re-solve: the highest rung replayed
MIXED_RUNGS = RUNGS[: RUNGS.index(RUNG) + 1]  # the fixed plan and the one-way rung


def front_loaded(forecast, n_service: int) -> np.ndarray:
    """One car's front-loaded plan over hours 0 to `n_service` (module text)."""
    left = forecast.cap
    out = np.zeros(n_service + 1)
    for t in range(n_service + 1):
        take = min(forecast.power_cap(t), max(left, 0.0))
        out[t] = take
        left -= take
    return out


def front_loaded_plan(day: Day) -> np.ndarray:
    """`(cars, n + 1)`: every car's front-loaded plan on its forecast; the day must carry
    forecasts."""
    if day.forecasts is None:
        raise ValueError("the day has no forecasts")
    return np.array([front_loaded(f, day.n_service) for f in day.forecasts]).reshape(
        len(day.forecasts), day.n_service + 1
    )


def front_loaded_slack(prepared: PreparedDay) -> float:
    """The smallest slack of the front-loaded plan over the rows the day was certified on."""
    if prepared.rows is None:
        raise ValueError("the day has no rows (it abstains before the certificate)")
    return float(prepared.rows.slack(front_loaded_plan(prepared.day)).min())


def front_loaded_inside(prepared: PreparedDay) -> bool:
    """True when the front-loaded plan meets every row within `SET_TOLERANCE` (reading C3)."""
    return front_loaded_slack(prepared) >= -SET_TOLERANCE


def _check_scale(beta: float) -> float:
    beta = float(beta)
    if not 0.0 < beta <= 1.0:
        raise ValueError(f"the mix is not defined at {beta}: only in (0, 1] (reading C1)")
    return beta


def check_factor(factor: float) -> float:
    """A room factor as a float; refused unless finite and at least 1 (reading C4)."""
    factor = float(factor)
    if not (math.isfinite(factor) and factor >= 1.0):
        raise ValueError(f"a room factor must be finite and at least 1, got {factor} (reading C4)")
    return factor


def mix_weight(beta: float, factor: float = 1.0) -> float:
    """The certified plan's weight in the mix at scale `beta` with room factor `factor`: `min(1,
    factor x beta)` (module text); `beta` itself at factor 1."""
    return min(1.0, check_factor(factor) * _check_scale(beta))


def mixed_policy(policy: AffinePolicy, plan, weight: float) -> AffinePolicy:
    """The mixed policy with weight `weight` on the certified plan (module text): centre `(1 -
    weight) plan + weight g`, box centre `(1 - weight) sum_i plan_i + weight m`, widths and gains
    unchanged."""
    weight = _check_scale(weight)
    plan = np.asarray(plan, dtype=float)
    if plan.shape != policy.g.shape:
        raise ValueError(f"the plan must be {policy.g.shape}, got {plan.shape}")
    g = (1.0 - weight) * plan + weight * policy.g
    m = (1.0 - weight) * plan[:, 1:].sum(axis=0) + weight * policy.m
    return AffinePolicy(m, policy.w, g, policy.G)


def evaluate(prepared: PreparedDay, beta: float, factor: float = 1.0) -> DayEvaluation:
    """The day replayed at scale `beta` with the mixed policy of room factor `factor` by the fixed
    plan and the one-way rung (module text); a day outside the set replays the certified policy
    (reading C2)."""
    beta = float(beta)
    factor = check_factor(factor)
    if prepared.abstains or beta <= 0:
        return gate_day.evaluate(prepared, beta)
    _check_scale(beta)
    start = time.perf_counter()
    policy = prepared.certificate.policy
    if front_loaded_inside(prepared):
        weight = mix_weight(beta, factor)
        policy = mixed_policy(policy, front_loaded_plan(prepared.day), weight)
    table = prepared.departure_bounds
    result = replay_day(
        prepared.day,
        replace(prepared.certificate, policy=policy),
        RUNG,
        beta,
        prepared.paths,
        departure_bound=lambda car, hour: float(table[car, hour]),
        budget=prepared.budget,
        time_limit=OTHER_SECONDS,
        departure_seen=prepared.departure_seen,
    )
    rungs = {}
    for name in MIXED_RUNGS:
        runs = result.by_rung[name]
        outcomes = Counter()
        for r in runs:
            outcomes.update(r.outcomes())
        finals = np.array([r.final for r in runs])
        rungs[name] = RungEvaluation(
            max(r.slot_loss for r in runs),
            max(r.driver_loss for r in runs),
            tuple(r.first_failure for r in runs),
            finals,
            dict(sorted(outcomes.items())),
            sum(x.seconds for r in runs for x in r.resolves),
            stay_check.check(prepared.day, finals, prepared.t5_tolerance_kwh),
        )
    return DayEvaluation(beta, False, rungs, 0, (), time.perf_counter() - start)
