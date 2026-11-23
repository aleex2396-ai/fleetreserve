"""Exact rational arithmetic (`fractions.Fraction`; every number of the day, the policy, the scale
and the path taken at its exact value) for the tests of V1 to V8 of `fixed_plan_loss`, module
text, "The validity limit and the stated errors", whose terms the names below follow.

The rows are the day's true one-way rows on the policy with the decision hour's draws
(`fixed_plan_loss.drawn_policy`). `forms` holds per row its exact centre `a g_i`, its exact gains
`a G_i` (one per request), its sizes `sum_j |a_j g_ij|` and `sum_j |a_j| |G_ij|_1`, and the code's
own reading of it (`policy.row_thresholds`, and whether the code's gains are at most `ZERO_GAIN`
and its centre fails its bound).

- `excess`: a row's excess;
- `rho`: a row's `delta_k`;
- `row_bound`: V1's bound on the excess of a row read as holding at `beta`; None for a row read as
  broken;
- `hour_errors`: `E_t`;
- `limit_bound`: V5's bound on a row's excess when it has one, else 0; `limit_errors`:
  `K_t LAMBDA + delta_t` per hour (V6); `tiny_count`: `K_t`; `driver_limit_error`: V7's
  `K LAMBDA + delta_D` over the rows of `D` or `D_C`; None where a row counted is read as broken;
- `balance_gaps`: `B_t`;
- `roundings`: `R_t` (V4);
- `driver_error`: `D` or `D_C`, or the same over the power-low rows;
- `exact_replay`: replay's rule (`replay` module text) in exact arithmetic, with each car's
  recovery supply and protected energy as the code holds them; `protected_energy`: `P`;
- `exact_bounds`: today's and the every-scale driver-loss bounds in exact arithmetic, the need
  row's bound taken as the recovery supply less the protected energy, `C` as the code reads it;
- `replayed`: the code's replay (`replay.run_plan`), with its misses as `replay.fails` computes
  them, taken at their exact values.
"""

import itertools
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from fleetreserve.fixed_plan_loss import drawn_policy, service_hours
from fleetreserve.plans import Plan
from fleetreserve.policy import ROW_TOLERANCE, VALIDITY_LIMIT, ZERO_GAIN
from fleetreserve.replay import run_plan

U = Fraction(1, 2**53)
ZERO = Fraction(0)
LAMBDA = Fraction(VALIDITY_LIMIT) * Fraction(ZERO_GAIN)  # the doubles' exact product
CLOSING = ("power-high", "power-low", "stored-high")
POWER = ("power-high", "power-low")


def gamma(k: int) -> Fraction:
    """`gamma_k = k u / (1 - k u)`, `u = 2^-53`."""
    return k * U / (1 - k * U)


def exact(x) -> Fraction:
    return Fraction(float(x))


@dataclass(frozen=True)
class Forms:
    centre: list
    gains: list
    sizes: list
    spreads: list
    read: np.ndarray
    tiny: np.ndarray
    fails_at_centre: np.ndarray
    n: int


def forms(policy, rows) -> Forms:
    drawn = drawn_policy(policy, rows)
    n = rows.n_service
    g = [[exact(v) for v in car] for car in drawn.g]
    G = [[[exact(v) for v in hour] for hour in car] for car in drawn.G]
    centre, gains, sizes, spreads = [], [], [], []
    for k in range(len(rows.kinds)):
        i = int(rows.car[k])
        a = [exact(v) for v in rows.coefficients[k]]
        centre.append(sum((a[j] * g[i][j] for j in range(n + 1)), ZERO))
        gains.append([sum((a[j] * G[i][j][s] for j in range(n + 1)), ZERO) for s in range(n)])
        sizes.append(sum((abs(a[j] * g[i][j]) for j in range(n + 1)), ZERO))
        spreads.append(sum((abs(a[j]) * sum(map(abs, G[i][j]), ZERO) for j in range(n + 1)), ZERO))
    centre_hat, spread_hat = drawn._forms(rows)
    return Forms(
        centre,
        gains,
        sizes,
        spreads,
        drawn.row_thresholds(rows),
        spread_hat <= ZERO_GAIN,
        rows.bounds - centre_hat < 0,
        n,
    )


def excess(f: Forms, rows, k: int, beta: float) -> Fraction:
    spread = sum(map(abs, f.gains[k]), ZERO)
    return max(ZERO, f.centre[k] + exact(beta) * spread - exact(rows.bounds[k]))


def excesses(f: Forms, rows, beta: float) -> list:
    """`excess` of every row (the need rows' with the code's bound; no statement uses them)."""
    return [excess(f, rows, k, beta) for k in range(len(rows.kinds))]


def rho(f: Forms, k: int, beta: float) -> Fraction:
    """`delta_k` of row `k` (V1)."""
    return gamma(2 * f.n + 2) * (f.sizes[k] + exact(beta) * f.spreads[k] + exact(ROW_TOLERANCE))


def row_bound(f: Forms, k: int, beta: float):
    if f.read[k] < beta:
        return None
    out = rho(f, k, beta)
    if f.tiny[k]:
        out += exact(beta) * exact(ZERO_GAIN)
        if f.fails_at_centre[k]:
            out += exact(ROW_TOLERANCE)
    return out


def hour_rows(rows, t: int) -> np.ndarray:
    """The rows counted in `E_t`."""
    same = np.isin(rows.kinds, CLOSING) & (rows.last_hours == t)
    earlier = (rows.kinds == "power-low") & (rows.last_hours >= 1) & (rows.last_hours < t)
    return np.flatnonzero(same | earlier)


def hour_errors(rows, ex) -> list:
    """`E_t` from the rows' excesses `ex`."""
    return [sum((ex[k] for k in hour_rows(rows, t)), ZERO) for t in range(1, rows.n_service + 1)]


def limit_bound(f: Forms, k: int, beta: float, ex) -> Fraction | None:
    """What row `k` adds to the bound up to the validity limit: 0 without an excess, else its
    `delta_k` plus `LAMBDA` (twice it failing at the centre) when its gains are at most
    `ZERO_GAIN`; None when it is read as broken."""
    if f.read[k] < beta:
        return None
    if ex[k] == 0:
        return ZERO
    extra = LAMBDA * (2 if f.fails_at_centre[k] else 1) if f.tiny[k] else ZERO
    return rho(f, k, beta) + extra


def limit_errors(f: Forms, rows, beta: float, ex) -> list:
    """`K_t LAMBDA + rho_t` per service hour; None for an hour with a row read as broken."""
    out = []
    for t in range(1, rows.n_service + 1):
        parts = [limit_bound(f, k, beta, ex) for k in hour_rows(rows, t)]
        out.append(None if None in parts else sum(parts, ZERO))
    return out


def tiny_count(f: Forms, rows, t: int, ex) -> int:
    """`K_t`: the rows of `E_t` with gains at most `ZERO_GAIN` and an excess, those failing at
    the centre counted twice."""
    return sum(
        (2 if f.fails_at_centre[k] else 1) for k in hour_rows(rows, t) if f.tiny[k] and ex[k] > 0
    )


def balance_gaps(policy, beta: float) -> list:
    b = exact(beta)
    out = []
    for t in range(1, policy.n_service + 1):
        centre = sum((exact(v) for v in policy.g[:, t]), ZERO) - exact(policy.m[t - 1])
        gains = ZERO
        for s in range(policy.n_service):
            own = exact(policy.w[t - 1]) if s == t - 1 else ZERO
            gains += abs(sum((exact(v) for v in policy.G[:, t, s]), ZERO) - own)
        out.append(abs(centre) + b * gains)
    return out


def roundings(day, policy, beta: float, errors, gaps) -> list:
    """`R_t` (V4) for service hours 1 to `n`."""
    b = exact(beta)
    cars = day.cars
    rates = sum((exact(c.rate) for c in cars), ZERO)
    out = []
    for t in range(1, day.n_service + 1):
        plans = ZERO
        for i, car in enumerate(cars):
            for s in range(1, t + 1):
                gains = sum(map(abs, map(exact, policy.G[i, s])), ZERO)
                plans += gamma(s + 2) * (abs(exact(policy.g[i, s])) + b * gains)
                plans += gamma(2) * exact(car.room)
        request = gamma(3) * (abs(exact(policy.m[t - 1])) + b * abs(exact(policy.w[t - 1])))
        inner = gamma(len(cars) - 1) * rates + plans + request
        out.append(U * (errors[t - 1] + gaps[t - 1]) + (1 + U) * inner)
    return out


def broken_hours(f: Forms, rows, beta: float) -> set:
    """`(car, hour)` pairs whose power-high or power-low row the code reads as broken at `beta`."""
    power = np.isin(rows.kinds, POWER) & (rows.last_hours >= 1)
    return {
        (int(rows.car[k]), int(rows.last_hours[k]))
        for k in np.flatnonzero(power)
        if f.read[k] < beta
    }


def driver_rows(f: Forms, rows, beta: float, every_scale: bool, kind="power-high") -> list:
    """The rows of `D` (with `every_scale`, of `D_C`), or the same of `kind`."""
    broken = broken_hours(f, rows, beta) if every_scale else set()
    return [
        int(k)
        for k in np.flatnonzero((rows.kinds == kind) & (rows.last_hours >= 1))
        if (int(rows.car[k]), int(rows.last_hours[k])) not in broken
    ]


def driver_error(f: Forms, rows, beta: float, every_scale: bool, ex, kind="power-high"):
    """`D` or `D_C` (kWh) from the rows' excesses `ex`."""
    return sum((ex[k] for k in driver_rows(f, rows, beta, every_scale, kind)), ZERO)


def driver_limit_error(f: Forms, rows, beta: float, every_scale: bool, ex):
    """`K LAMBDA + delta_D` over the rows of `D` or `D_C`; None when one is read as broken."""
    parts = [limit_bound(f, k, beta, ex) for k in driver_rows(f, rows, beta, every_scale)]
    return None if None in parts else sum(parts, ZERO)


def protected_energy(day) -> Fraction:
    """`P`: the cars' protected energies, as the code holds them, summed exactly."""
    return sum((exact(c.need(day.rho)) for c in day.cars), ZERO)


def _supply(car, n: int) -> Fraction:
    return exact(car.rate * max(car.departure - 1 - n, 0))


def exact_replay(day, policy, beta: float, z) -> tuple[list, Fraction]:
    """Per service hour `|delivered - request|`, and the driver loss, of the fixed plan at
    `beta` on the path `z`, in exact arithmetic."""
    n = day.n_service
    b = exact(beta)
    zs = [exact(v) for v in z]
    delivered = [ZERO] * n
    short = total = ZERO
    for i, car in enumerate(day.cars):
        rate, room = exact(car.rate), exact(car.room)
        x = ZERO
        for t in range(n + 1):
            plan = exact(policy.g[i, t]) + b * sum(
                (exact(policy.G[i, t, s]) * zs[s] for s in range(n)), ZERO
            )
            q = max(min(plan, rate, room - x), ZERO) if t < car.departure else ZERO
            x += q
            if t >= 1:
                delivered[t - 1] += q
        final = x + min(_supply(car, n), max(room - x, ZERO))
        need = exact(car.need(day.rho))
        short += max(ZERO, need - final)
        total += need
    misses = [
        abs(delivered[t] - (exact(policy.m[t]) + b * exact(policy.w[t]) * zs[t])) for t in range(n)
    ]
    return misses, (short / total if total > 0 else ZERO)


def exact_bounds(f: Forms, policy, rows, day, beta: float) -> tuple[Fraction, Fraction]:
    """Today's and the every-scale driver-loss bounds at `beta`, in exact arithmetic."""
    b = exact(beta)
    n = rows.n_service
    hours = service_hours(policy, rows)
    need_rows = np.flatnonzero(rows.kinds == "need")
    total = protected_energy(day)
    if total <= 0:
        return ZERO, ZERO
    today = every = ZERO
    for position, k in enumerate(need_rows):
        car = day.cars[int(rows.car[k])]
        short = exact(car.need(day.rho)) - _supply(car, n)
        term = max(ZERO, f.centre[k] + b * sum(map(abs, f.gains[k]), ZERO) + short)
        today += term
        clipped = [t for t in range(n) if hours.thresholds[position, t] < beta]
        if not clipped:
            every += term
            continue
        g = [exact(v) for v in hours.centre[position]]
        G = [[exact(v) for v in hour] for hour in hours.gains[position]]
        cap = [exact(v) for v in hours.cap[position]]
        kept = [t for t in range(n) if t not in clipped]
        linear = sum((g[t] for t in kept), ZERO) - b * sum(
            (abs(sum((G[t][s] for t in kept), ZERO)) for s in range(n)), ZERO
        )
        least = sum(
            (min(max(g[t] - b * sum(map(abs, G[t]), ZERO), ZERO), cap[t]) for t in clipped), ZERO
        )
        corner = min(
            sum(
                (
                    min(g[t] + b * sum((G[t][s] * c for s, c in enumerate(z)), ZERO), cap[t])
                    for t in range(n)
                ),
                ZERO,
            )
            for z in itertools.product((-1, 1), repeat=n)
        )
        lower = exact(hours.first[position]) + max(linear + least, corner)
        every += max(ZERO, short - lower)
    return today / total, every / total


def replayed(day, policy, beta: float, paths) -> list:
    """Per path of `paths`: (the code's misses per service hour, exactly; its first failing hour
    or None; its driver loss), replay of the fixed plan at `beta`."""
    plan = Plan.from_policy(policy, beta)
    out = []
    for z in paths:
        requests = policy.request(z, beta)
        run = run_plan(day, plan, z, requests)
        misses = [exact(v) for v in np.abs(run.delivered - requests)]
        out.append((misses, run.first_failure, run.driver_loss))
    return out
