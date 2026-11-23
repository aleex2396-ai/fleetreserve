"""The fixed plan's losses in closed form, and their comparison with replay (plan sections 5, 6).

T6 (plan section 6): "Fixed plan: l_d(beta) = (|S| - t_min(beta) + 1) / |S|, t_min the first hour
whose threshold s_dt (min over its true-set rows of (b - a g) / |a G|_1) is below beta; driver loss
bounded by summed per-car worst cases, a bound only when every car draws what its plan gives it
(the owner's correction of 15 November 2026; before it: stated without the condition)."

The decision hour's draw (the owner's decision of 16 November 2026: the closed form uses what a
car can really draw in the decision hour). Replay caps a car in hour 0 at what it can take (plan
section 5: "cars capped at what they can take"): a true one-way car draws
`max(min(g_i0, r_i [d_i > 0], e_i), 0)` of its planned hour-0 power `g_i0`. `decision_hour_draws`
reads that interval from the car's own rows closing in hour 0 (`[0, min(r_i [d_i > 0], e_i)]` on
`carsets.true_rows(day)`, the rows every caller passes) and moves `g_i0` into it; `drawn_policy`
is the policy with each car's draw in place of `g_i0`, and every row below is evaluated on it.
It refuses rows that are not one-way (a "stored-low" or "energy-high" row, or a power-low row that
lets a car discharge): two-way rows would give a discharge interval that the fixed plan's one-way
replay never draws.
The fixed plan at scale `beta` is `g + beta G zeta` and hour 0 has no gain (`AffinePolicy` refuses
one), so the draw is the same at every scale and on every path: the thresholds stay one number per
service hour, and the slot loss a step function that does not fall as `beta` rises. A day on which
every car can take its planned hour-0 power keeps its policy, and every value is the one computed
from the planned hour-0 powers, bit for bit. Before the decision the rows counted the planned
hour-0 power, so a car capped in hour 0 made a later stored-energy row read broken where replay
holds (the difference was named "decision-hour cap").

Row-to-hour reading: a row belongs to the hour it closes in (`FleetRows.last_hours`, the last hour
with a non-zero coefficient). The power rows and stored-energy rows of service hour `t` give
`s_dt`. The need row counts towards driver loss only. Rows closing in hour 0 do not depend on the
requests (hour 0 has no gains), and under the draw they hold, so they have no threshold; hour 0
carries no request and cannot fail.

Why the closed form gives replay's first failing hour (reading R1), on three premises: exact
arithmetic, every row read as holding at `beta` holding on the whole box, and cars that share the
box out exactly. "The validity limit and the stated errors" (below) states what is claimed without
them. The fixed plan is open loop: car `i` plans `g_it + beta G_it zeta` in hour `t` whatever any
car drew. In replay a present car draws its plan in service hour `t` exactly when its rows closing
in `t` hold on the path for the energy it holds by then (its power at least 0 and at most
`r_i [t < d_i]`, its stored energy at most `e_i`). Until the first hour in which some car draws
other than its plan, that energy is its hour-0 draw plus its planned powers, which is what the rows
on `drawn_policy` count; each row is affine in the path, so it is broken on some corner exactly when
its threshold is below `beta`. Hence `t_min(beta)` is the first service hour in which some car draws
other than its plan on some corner, and before it the fleet meets every request exactly (the cars'
`g` sum to `m`, their `G` to `W`). A car capped later by its room, its rate or its departure is one
of those rows; the recovery hours carry no request. So replay's first failing hour is never earlier
than `t_min`, and it is later only for the two reasons `compare_with_replay` names:

- "within tolerance": at the closed form's hour every path's draws miss the plan by at most
  `FAILURE_KWH` in total, which plan section 1 does not count as a failure;
- "shortfall and excess cancel": on some path at that hour cars draw less and more than planned
  (a negative planned power draws nothing one-way) and the fleet's total stays within
  `FAILURE_KWH`.

Without the premises "replay earlier" and "unexplained" are not claimed impossible: V4 and V6
(below) state when replay fails no hour before `t_min`, and the code's replay fails an hour where
the closed form gives none on day H3 of the tests above the validity limit and on the 1,000-car day
of V6 at it.

Driver-loss bound: per car, the worst case over the requests of the shortfall of its true need row
on `drawn_policy` (`max(0, a g + beta |a G|_1 - b)`, the energy by which its decision-hour draw and
its planned powers, with the true recovery supply, can miss its protected energy), summed over
cars, as a share of the protected energy.

When it is a bound (reading R2). On a path, it bounds the replayed shortfall of a car that in every
service hour draws its plan, more than its plan (a negative plan draws nothing one-way, which leaves
the car more energy), or less only because it is full (its room `e` is at least its protected energy
`(1 - rho) e`); its hour-0 cap is counted by the draw. At a scale whose closed-form slot loss is 0,
in exact arithmetic every car in every service hour on every path draws its plan, more, less because
it is full, or less by at most the excess of a power-high row read as holding, so there it bounds
replay up to the errors of V3 and V7 (below). At a scale whose slot loss is above 0 a car capped by
its rate or its departure in a service hour can end below its planned energy, and replay's driver
loss can then be above the bound (the early-departure hand day of the tests: bound 0, replay 1/8 at
scale 1): the made-up comparisons of `tests/wp6/test_closed_form.py` count how often, each at a
scale with a failing hour; the every-scale bound below (reading R3) is 1/8 there. Before the owner's
decision of 16 November 2026 the bound counted the planned hour-0 power, so a car capped in hour 0
could take it above the bound at a scale with no failing hour.

The every-scale bound (reading R3), `every_scale_driver_bound`: a bound on every path of the box at
every scale, up to the errors of V3 and V7 (below). By the owner's decision of 16 November 2026 it
is the driver bound wherever one is used (`gate_calibrate`, module text); the bound above is
reported beside it for comparison only. Per car on `drawn_policy`, with `p_t = g_t + beta G_t zeta`
its plan in service hour `t`, `c_t = r [t < d]` the bound of its power-high row and `N = -b` of its
need row (the protected energy less the recovery supply):

- in replay the car draws `q_t = clip(p_t, 0, c_t)` in service hour `t` until it is full, so after
  the recovery hours it is short of its protected energy by `max(0, N - d_0 - sum_t q_t)`, `d_0`
  its decision-hour draw; the room drops out, as the protected energy is at most the room;
- `C(beta)`: its service hours whose power-high or power-low row has a threshold below `beta`; in
  an hour outside `C`, on every path `q_t` is `p_t`, or more (a plan below 0 by at most its
  power-low row's excess draws 0), or less by at most its power-high row's excess (below).
  Two lower bounds on `d_0 + sum_t q_t` on every
  path, of which `L` is the larger:
  - by hour: `d_0 + sum_{t not in C} g_t - beta |sum_{t not in C} G_t|_1 + sum_{t in C}
    clip(g_t - beta |G_t|_1, 0, c_t)`, each hour of `C` at the least it can draw;
  - by corner: `d_0` plus the least over the `2^n` corner paths of `sum_t min(p_t, c_t)`, a
    function concave in the path and so least at a corner. It drops the floor at 0: it is the
    tighter one where a rate or a departure caps an hour whose gain another hour offsets, the
    by-hour bound where a plan below 0 draws nothing;
- the car's term is `max(0, N - L)`, summed over the cars over the protected energy. For a car with
  `C` empty both lower bounds equal its need row's worst case, so its term is the one above,
  computed as above: where no hour fails the two bounds are equal bit for bit;
- it does not fall as `beta` rises. `C` grows with `beta`; between thresholds both lower bounds
  fall; when hour `t` joins `C` its plan lies in `[0, c_t]` on the whole box, so the by-hour bound
  falls by `beta (|A|_1 + |G_t|_1 - |A + G_t|_1) >= 0`, `A` the summed gains of the other hours
  outside `C`; the by-corner bound is a least value over a box that grows with `beta`. The bound
  can jump up where a power row breaks, and needs no running maximum;
- each term is at most `max(N, 0)`, the car's protected energy less its recovery supply, plus the
  excesses of its power-low rows of the hours outside `C` (the by-hour bound is at least minus
  their sum), so the day's bound is at most 1 plus their sum over the cars and the protected
  energy, above 1 only where such a row is exceeded (V1, below);
  `gate_calibrate.every_scale_loss` caps the day's loss at 1;
- how tight: the by-hour bound is the car's exact worst case when the gains of its hours in `C`
  share no request with its other hours (diagonal gains). The exact worst case in general is the
  least of a sum of clipped affine functions over the box, which is not a convex problem (`2^|S|`
  linear programmes, `S` the hours that can exceed their cap). The sum over cars of per-car worst
  cases is above the worst path's loss when the cars' worst paths differ, as in T6;
- a power row whose gains are at most `policy.ZERO_GAIN` (as the code computes them, below)
  counts, in its threshold, as not depending on the requests and as holding within
  `policy.ROW_TOLERANCE`, as for the slot loss, so
  it puts its hour in `C` at every scale or at none (the hour's other power row can still put it
  there); both lower bounds and the need row count every gain as it is, however small.

The validity limit and the stated errors (readings R1 to R3): the one statement, V1 to V8, of what
replay can differ from the closed forms by; other texts point here.

Terms. A row `a u <= b` of car `i` has, at scale `beta`, the excess
`max(0, a g_i + beta |a G_i|_1 - b)` on `drawn_policy`, in exact arithmetic: the most it is
exceeded on any path of the box, which does not fall as `beta` rises. It is read as holding at
`beta` when its threshold, as `policy.row_thresholds` computes it, is at least `beta`; that
function reads a row whose gains `|a G_i|_1` are at most `policy.ZERO_GAIN` by its centre alone:
as holding at every scale (`+inf`) when `b - a g_i >= -policy.ROW_TOLERANCE`, else as broken at
every scale. "Gains at most `ZERO_GAIN`" and "fails at the centre" (`b - a g_i < 0`) are as the
code computes them (`AffinePolicy._forms`): a row with the exact gains 1e-12 and 1e-30 has the
computed `|a G_i|_1` 1e-12, so it has gains at most `ZERO_GAIN`. Further:

- `delta_k = gamma_{2n+2} (sum_j |a_j g_ij| + beta sum_j |a_j| |G_ij|_1 + ROW_TOLERANCE)`, the
  rounding of the threshold of row `k` (of car `i`), `n` the service hours, `u = 2^-53` and
  `gamma_k = k u / (1 - k u)`: the standard model of floating-point rounding, without underflow;
- `E_t` (kWh): over the cars, the excesses of the car's power-high, power-low and stored-high rows
  closing in service hour `t` and of its power-low rows of hours 1 to `t - 1` (a plan below 0
  draws 0, which leaves the car more energy, so a later stored-energy row can bind sooner);
- `B_t = |sum_i g_it - m_t| + beta |sum_i G_it - W_t|_1`, `W_t` the request's own gain: the
  policy's balance gap, 0 when its cars share the box out exactly;
- `D` (kWh): the summed excesses of the power-high rows of service hours 1 to `n`; `D_C` the same
  over the hours outside each car's `C` (`C` as the code reads it); `P` the day's protected energy;
- `LAMBDA = VALIDITY_LIMIT ZERO_GAIN`, the doubles' exact product, `ROW_TOLERANCE` plus 2.5e-24
  kWh, with `policy.VALIDITY_LIMIT = ROW_TOLERANCE / ZERO_GAIN`, 1e5 exactly as a double;
- `K_t`: the rows counted in `E_t` with gains at most `ZERO_GAIN` and an excess, those that fail
  at the centre counted twice; `delta_t`: the `delta_k` of the rows counted in `E_t` with an
  excess, summed; `K` and `delta_D`: the same over the rows of `D` or of `D_C`.

"In exact arithmetic": every quantity computed exactly from the numbers the code holds (the policy,
the scale, the path, each car's rate, room, protected energy and recovery supply, the need row's
bound taken as that supply less that protected energy). Each statement says at which scales it is
claimed:

- V1, every scale: a row read as holding at `beta` has an excess of at most `delta_k`, plus
  `beta ZERO_GAIN` when its gains are at most `ZERO_GAIN`, plus `ROW_TOLERANCE` more when it also
  fails at the centre (day Z1 of the tests: a row with no gain failing at the centre by the double
  `1 + 5e-8` less 1 kWh is read as holding and has that excess at every scale);
- V2, every scale, exact arithmetic: in every service hour `t`, on every path, the cars' draws
  miss the request by at most `E_t + B_t`;
- V3, every scale, exact arithmetic: replay's driver loss is at most today's bound plus `D / P`,
  and at most the every-scale bound plus `D_C / P`: a car that is not full and draws less than its
  plan for its rate or its departure draws less by at most that row's excess, and the hours of `C`
  are counted at the least they can draw (on day Z1 it is either bound plus `D / P`);
- V4, every scale: the code's replay (`replay.run_plan` with the request of
  `AffinePolicy.request`) misses the request in service hour `t` on every path, as `replay.fails`
  computes it, by at most `E_t + B_t + R_t`, with
  `R_t = u (E_t + B_t) + (1 + u) [gamma_{F-1} sum_i r_i + gamma_3 (|m_t| + beta |w_t|)
  + sum_i sum_{s=1..t} (gamma_{s+2} (|g_is| + beta |G_is|_1) + gamma_2 e_i)]`, `F` the number of
  cars, `r_i` and `e_i` their true rates and rooms: the rounding of the fleet's sum, of the
  request, of the planned powers and of each car's room left and energy held, and of the
  difference. So it fails no hour before `t_min` while `E_t + B_t + R_t <= FAILURE_KWH` in every
  hour before `t_min`;
- V5, scales up to `VALIDITY_LIMIT`: a row read as holding with gains at most `ZERO_GAIN` has an
  excess of at most `LAMBDA + delta_k`, and `2 LAMBDA + delta_k` when it fails at the centre (V1,
  as `beta ZERO_GAIN <= LAMBDA` and `ROW_TOLERANCE < LAMBDA`). At the next double above the limit
  a row at its bound with gains exactly `ZERO_GAIN` has an excess above `LAMBDA`, by 1.46e-23 kWh;
- V6, scales up to `VALIDITY_LIMIT`, hours before `t_min` (where every row counted in `E_t` is read
  as holding): `E_t <= K_t LAMBDA + delta_t`, so the code's replay fails hour `t` only if
  `K_t LAMBDA + delta_t + B_t + R_t > FAILURE_KWH`.
  `FAILURE_KWH - 999 LAMBDA` is 1.0e-7 kWh plus 6.8e-21 and `FAILURE_KWH - 1000 LAMBDA` is
  6.8e-21 kWh: with `K_t` at most 999 the hour fails only if `delta_t + B_t + R_t` passes 1.0e-7
  kWh. With 1,000 it can fail: on the 1,000-car day of the tests (each car planning its rate
  3.3 kW with gains `ZERO_GAIN`, the box their sum in floating point) at 1e5 exact replay misses
  the request by 9.99999993e-5 kWh and the code's replay by 1.0000000002e-4 kWh, within
  `E_1 + B_1 + R_1`, and fails the hour;
- V7, scales up to `VALIDITY_LIMIT`, exact arithmetic: replay's driver loss is at most the
  every-scale bound plus `(K LAMBDA + delta_D) / P` over the rows of `D_C` (each read as holding:
  an hour with a power row read as broken is in `C`). Today's bound plus the same over the rows of
  `D` is claimed only at a scale where the day's slot form gives no failing hour (there each row
  of `D` is read as holding); at any other scale only V3 is claimed for today's bound, and its `D`
  can count the excess of a row read as broken: day `EARLY` of
  `tests/wp11/test_every_scale_calibrate.py` at scale 0.5 (slot loss 0.5; car A plans 0.5 kWh in
  the hour after it has left) has today's bound 0 and replay's driver loss 1/8 = `D / P` on every
  corner, and the every-scale bound 1/8 with `D_C` 0;
- V8, the balance gap: no code enforces it, and `balance_gap_run` measures it on the certificates
  of a recorded gate or learn run (`certificate` takes `m`, `w`, `g` and `G` from the solver, which
  meets the programme's equality rows only to its own precision). In exact
  arithmetic, where no row closing in hours 1 to `t` is exceeded, replay's miss in hour `t` on the
  worst corner is `B_t`: a `B_t` above `FAILURE_KWH` can fail the hour, and one above
  `FAILURE_KWH - E_t - R_t` leaves V4's condition unmet. At a scale up to 1 `B_t` is at most the
  gap in the centres plus the gap in the summed gains; at the limit a gap in the summed gains alone
  above `FAILURE_KWH / VALIDITY_LIMIT` (1e-9 plus 4.8e-26 kWh) can fail the hour. V3 and V7 do
  not contain it.

The code evaluates both driver bounds in floating point; that rounding is not bounded here, and V3
and V7 are in exact arithmetic. `tests/wp11/test_every_scale_validity.py` checks V1 to V7 on seeded
made-up days and V8 on hand days, beside the hand days named here, in exact rational arithmetic
with no margin in any check of a stated error.

Above `VALIDITY_LIMIT` V5 to V7 are not claimed; V1 to V4 are, with errors that can grow with
`beta` (V1's `beta ZERO_GAIN`): day H3 of `tests/wp11/test_every_scale_readings.py`, every
threshold infinite, fails its first hour in the code's replay at 4e8 and not at 1e8, and its
replayed driver loss is 0.0078125 at 3e11 and 0.0625 at 1e12, against both bounds 0.
`gate_calibrate` limits every finite scale of the every-request route to the limit. The tiny-gain
limit of a set of a day's rows, the smallest `(b - a g_i) / |a G_i|_1` over those read as holding
with gains above 0 and at most `ZERO_GAIN` (`+inf` when there is none, 0 for such a row at its
bound, below 0 for one that fails at the centre), can be 0 (day H3), so it sets no scale.

The alternative not taken, an exact rework with no limit: the every-scale bound's `C` from the
exact thresholds `(b - a g) / |a G|_1` of the power rows; the power-high excesses added to today's
bound; and for the slot form, each row with gains at most `ZERO_GAIN` given the threshold
`(b - a g + FAILURE_KWH / (2 K)) / |a G|_1`, `K` the day's number of such rows. It gives a finite
threshold to each row read as holding with gains above 0 and at most `ZERO_GAIN`, so an infinite
slot scale can become finite, and its slot part departs from T6's per-row minimum (quoted above), a
question for the plan.
"""

from dataclasses import dataclass

import numpy as np

from fleetreserve.carsets import FleetRows
from fleetreserve.day import Day
from fleetreserve.plans import Plan
from fleetreserve.policy import AffinePolicy
from fleetreserve.products import corner_paths
from fleetreserve.replay import FAILURE_KWH, protected, run_plan

CLOSING_KINDS = ("power-high", "power-low", "stored-high", "stored-low")
ONE_WAY_KINDS = ("power-high", "power-low", "stored-high", "need")


def decision_hour_draws(policy: AffinePolicy, rows: FleetRows) -> np.ndarray:
    """Per car, what it draws in hour 0 (module text): its planned hour-0 power moved into the
    interval its rows closing in hour 0 allow, `max(min(g_i0, upper), lower)`; one-way rows only
    (module text)."""
    if rows.n_cars != policy.n_cars or rows.n_service != policy.n_service:
        raise ValueError("the rows are for another fleet or window")
    discharge = rows.bounds[rows.kinds == "power-low"] > 0
    if not np.isin(rows.kinds, ONE_WAY_KINDS).all() or discharge.any():
        raise ValueError("the decision hour's draw is derived for one-way rows only")
    closing = np.isin(rows.kinds, CLOSING_KINDS) & (rows.last_hours == 0)
    a = rows.coefficients[:, 0]
    upper = np.full(rows.n_cars, np.inf)
    lower = np.full(rows.n_cars, -np.inf)
    for k in np.flatnonzero(closing & (a > 0)):
        upper[rows.car[k]] = min(upper[rows.car[k]], rows.bounds[k] / a[k])
    for k in np.flatnonzero(closing & (a < 0)):
        lower[rows.car[k]] = max(lower[rows.car[k]], rows.bounds[k] / a[k])
    return np.maximum(np.minimum(policy.g[:, 0], upper), lower)


def drawn_policy(policy: AffinePolicy, rows: FleetRows) -> AffinePolicy:
    """`policy` with each car's hour-0 power replaced by its draw (`decision_hour_draws`); the
    policy itself when every car draws its planned hour-0 power."""
    draws = decision_hour_draws(policy, rows)
    if np.array_equal(draws, policy.g[:, 0]):
        return policy
    g = np.array(policy.g)
    g[:, 0] = draws
    return AffinePolicy(policy.m, policy.w, g, policy.G)


def hour_thresholds(policy: AffinePolicy, rows: FleetRows) -> np.ndarray:
    """`s_dt` for service hours 1 to `n` (index `t - 1`), on the policy with the decision hour's
    draws (module text); `+inf` for an hour with no row."""
    thresholds = drawn_policy(policy, rows).row_thresholds(rows)
    closes = rows.last_hours
    kind = np.isin(rows.kinds, CLOSING_KINDS)
    out = np.full(rows.n_service, np.inf)
    for t in range(1, rows.n_service + 1):
        mask = kind & (closes == t)
        if mask.any():
            out[t - 1] = thresholds[mask].min()
    return out


def first_failing_hour(thresholds, beta: float) -> int | None:
    """`t_min(beta)`: the first service hour whose threshold is below `beta`, or None."""
    below = np.flatnonzero(np.asarray(thresholds) < beta)
    return int(below[0]) + 1 if below.size else None


def closed_form_slot_loss(thresholds, beta: float) -> float:
    n = len(thresholds)
    t = first_failing_hour(thresholds, beta)
    return 0.0 if t is None else (n - t + 1) / n


def _need_shortfalls(drawn: AffinePolicy, rows: FleetRows, beta: float) -> np.ndarray:
    """Per need row (one per car), `max(0, a g + beta |a G|_1 - b)` on the drawn policy."""
    need = rows.kinds == "need"
    return np.maximum(drawn.worst_case(rows, beta)[need] - rows.bounds[need], 0.0)


def driver_loss_bound(policy: AffinePolicy, rows: FleetRows, day: Day, beta: float) -> float:
    """Summed per-car worst-case shortfalls of the true need rows, on the policy with the decision
    hour's draws (module text), over the protected energy."""
    total = protected(day).sum()
    if total <= 0:
        return 0.0
    return float(_need_shortfalls(drawn_policy(policy, rows), rows, beta).sum() / total)


@dataclass(frozen=True)
class ServiceHours:
    """What the every-scale bound reads of each car (reading R3), one entry per need row, in the
    order of the need rows: on the drawn policy, the planned power `g_t` of service hours 1 to `n`
    (`centre`, shape `(k, n)`), its gains `G_t` (`gains`, `(k, n, n)`) and `|G_t|_1` (`spread`), the
    power-high row's bound `r [t < d]` (`cap`), the smaller threshold of the hour's power-high and
    power-low rows (`thresholds`), the decision-hour draw (`first`) and `N = -b` of the need row,
    the protected energy less the recovery supply (`short`)."""

    centre: np.ndarray
    gains: np.ndarray
    spread: np.ndarray
    cap: np.ndarray
    thresholds: np.ndarray
    first: np.ndarray
    short: np.ndarray


def service_hours(policy: AffinePolicy, rows: FleetRows) -> ServiceHours:
    """`ServiceHours` of the true one-way rows `rows` (`carsets.true_rows(day)`) on the policy with
    the decision hour's draws. Refuses rows of another kind of set: the bound is derived for the
    one-way replay (reading R3)."""
    drawn = drawn_policy(policy, rows)
    if not set(rows.kinds) <= {"power-high", "power-low", "stored-high", "need"}:
        raise ValueError("the every-scale bound is derived for the true one-way rows only")
    need = np.flatnonzero(rows.kinds == "need")
    cars = rows.car[need]
    if sorted(cars.tolist()) != list(range(rows.n_cars)):
        raise ValueError("each car needs exactly one need row")
    if np.any(rows.coefficients[need] != -1.0):
        raise ValueError("a need row must count every hour's power once")
    position = np.empty(rows.n_cars, dtype=int)
    position[cars] = np.arange(cars.size)
    n = rows.n_service
    thresholds = np.full((cars.size, n), np.inf)
    cap = np.full((cars.size, n), np.inf)
    row_thresholds = drawn.row_thresholds(rows)
    for k in np.flatnonzero(np.isin(rows.kinds, ("power-high", "power-low"))):
        t = int(rows.last_hours[k])
        if t == 0:
            continue
        i = position[rows.car[k]]
        a = rows.coefficients[k, t]
        if rows.kinds[k] == "power-low" and not (a < 0 and rows.bounds[k] == 0):
            raise ValueError("a one-way power-low row reads -u_t <= 0")
        if rows.kinds[k] == "power-high":
            cap[i, t - 1] = min(cap[i, t - 1], rows.bounds[k] / a)
        thresholds[i, t - 1] = min(thresholds[i, t - 1], row_thresholds[k])
    gains = drawn.G[cars, 1:, :]
    return ServiceHours(
        centre=drawn.g[cars, 1:],
        gains=gains,
        spread=np.abs(gains).sum(axis=2),
        cap=cap,
        thresholds=thresholds,
        first=drawn.g[cars, 0],
        short=-rows.bounds[need],
    )


def every_scale_shortfalls(hours: ServiceHours, today: np.ndarray, beta: float) -> np.ndarray:
    """Per need row, the every-scale bound's shortfall at `beta` (reading R3): `today` (the
    shortfall of today's bound, in the same order) for a car none of whose power rows is broken at
    `beta`, else `max(0, N - L)` with `L` the larger of the two lower bounds of reading R3 on the
    energy the car holds after the service hours, on every path of the box."""
    clipped = (hours.thresholds < beta).any(axis=1)
    if not clipped.any():
        return today
    broken = hours.thresholds[clipped] < beta
    centre, gains, cap = hours.centre[clipped], hours.gains[clipped], hours.cap[clipped]
    linear = np.where(broken, 0.0, centre).sum(axis=1) - beta * np.abs(
        np.where(broken[..., None], 0.0, gains).sum(axis=1)
    ).sum(axis=1)
    least = np.clip(centre - beta * hours.spread[clipped], 0.0, cap)
    by_hour = linear + np.where(broken, least, 0.0).sum(axis=1)
    corners = corner_paths(centre.shape[1])
    planned = centre[:, :, None] + beta * np.einsum("its,ks->itk", gains, corners)
    by_corner = np.minimum(planned, cap[:, :, None]).sum(axis=1).min(axis=1)
    lower = hours.first[clipped] + np.maximum(by_hour, by_corner)
    out = np.array(today, dtype=float)
    out[clipped] = np.maximum(hours.short[clipped] - lower, 0.0)
    return out


def every_scale_driver_bound(policy: AffinePolicy, rows: FleetRows, day: Day, beta: float) -> float:
    """The every-scale driver-loss bound (reading R3): per car the shortfall of today's bound
    (`driver_loss_bound`) where none of its power rows is broken at `beta`, else the shortfall
    against the least energy it can hold on any path of the box, summed over the protected energy.
    Its error against replay: V3 and V7 of the module text; bit-identical to `driver_loss_bound`
    where no car's power row is broken."""
    total = protected(day).sum()
    if total <= 0:
        return 0.0
    today = _need_shortfalls(drawn_policy(policy, rows), rows, beta)
    hours = service_hours(policy, rows)
    return float(every_scale_shortfalls(hours, today, beta).sum() / total)


@dataclass(frozen=True)
class Comparison:
    """One day at one scale: closed form against replay of all `2^n` corner paths.

    `driver_bound` is the every-scale bound (reading R3), the driver bound the owner's decision
    of 16 November 2026 sets wherever one is used; `earlier_driver_bound` is the bound of reading
    R2 (`driver_loss_bound`), reported beside it for comparison only: it is not a bound where an
    hour fails. `reason` is None when the first failing hours agree, else one of the module text's
    reasons, or "replay earlier" or "unexplained" (module text: reading R1, V4 and V6)."""

    beta: float
    closed_hour: int | None
    replay_hour: int | None
    closed_slot: float
    replay_slot: float
    driver_bound: float
    replay_driver: float
    reason: str | None
    earlier_driver_bound: float


def _reason(runs, closed_hour, replay_hour) -> str | None:
    if closed_hour == replay_hour:
        return None
    if closed_hour is None or (replay_hour is not None and replay_hour < closed_hour):
        return "replay earlier"
    gaps = np.array([run.draws[:, closed_hour] - run.planned[:, closed_hour] for run in runs])
    if np.abs(gaps).sum(axis=1).max() <= FAILURE_KWH:
        return "within tolerance"
    if np.any((np.abs(gaps).sum(axis=1) > FAILURE_KWH) & (np.abs(gaps.sum(axis=1)) <= FAILURE_KWH)):
        return "shortfall and excess cancel"
    return "unexplained"


def compare_with_replay(day: Day, policy: AffinePolicy, rows: FleetRows, betas) -> list:
    """For each `beta`: the closed form on the true rows `rows` against replay on every corner."""
    paths = corner_paths(day.n_service)
    thresholds = hour_thresholds(policy, rows)
    out = []
    for beta in betas:
        plan = Plan.from_policy(policy, beta)
        runs = [run_plan(day, plan, z, policy.request(z, beta)) for z in paths]
        hours = [r.first_failure for r in runs if r.first_failure is not None]
        replay_hour = min(hours) if hours else None
        closed_hour = first_failing_hour(thresholds, beta)
        out.append(
            Comparison(
                float(beta),
                closed_hour,
                replay_hour,
                closed_form_slot_loss(thresholds, beta),
                max(r.slot_loss for r in runs),
                every_scale_driver_bound(policy, rows, day, beta),
                max(r.driver_loss for r in runs),
                _reason(runs, closed_hour, replay_hour),
                driver_loss_bound(policy, rows, day, beta),
            )
        )
    return out
