"""The re-solve of a rung at the start of service hour `t` (plan section 5, "Rungs").

The programme plans the remaining hours `t..n` for the cars plugged in at the start of hour `t`
(a car that unplugs within hour `t` is not yet seen gone: `rungs.py`, information model). Every
input is one the rung observes then (`Remaining`). The box is the one
sold, at the scale `beta` being run: `y_s = m_s + beta w_s zeta_s`. `zeta_1 .. zeta_t` are known
(the request of hour `t` is known at its start); `zeta_{t+1} .. zeta_n` are free in `[-1, 1]`. The
plan is causal affine: car `i` draws `h_{i,s} + sum_{t < s' <= s} H_{i,s,s'} zeta_{s'}` in hour
`s`, and the present cars share every remaining request exactly.

Each present car's set is its forecast set (plan section 2) updated with what is known at `t`:

- departure: the survival-tightened bound given by the caller (plan section 1: the
  kappa-quantile of departure given presence at `t`), which sets the shares `f_s` of the power
  limit `r_hat f_s` and of the recovery supply;
- energy drawn so far `x`: the stored-energy limit becomes `max(e_cap, x) - x` from now on, and
  the needed energy counts `x`;
- an upper bound `c` on any later charge, known from an earlier hour in which the car took less
  than planned (it was then at its true rate or full): the power limit is at most `c`;
- the needed energy `e_need` is lowered, where needed, to the most the car can still reach on this
  set, `min(cap, x + sum_s min(r_hat f_s, c)) + supply`, so that a car that can no longer reach
  its forecast need does not make the whole re-solve infeasible (the guard below still protects
  it against the plan it replaces);
- two-way (plan section 9): power down to `-r_hat f_s` and stored energy at least `-B` after
  every hour; one-way: power at least 0.

The guard (plan section 5): "keeping each present car's planned stored energy >= the
predecessor's under every future request" (`Guard`). The predecessor's plan is the one in force
when the rung first left it, at hour `a` (its anchor), and its planned stored energy is counted
from the energy each car held then: `x_i(a) + sum_{s = a..s'} p_{i,s}`. For every present car and
every hour `s'` in `t..n`, the new plan's planned stored energy `x_i(t) + sum_{s = t..s'} q_{i,s}`
is at least that, for every future request. At the first re-solve `a = t` and this compares the
energy the two plans draw from now on; at a later re-solve it also asks the rung to make up what
its own plan has drawn less than the predecessor planned since `a`. When the predecessor's plan
is run one-way, a negative planned power draws nothing, so its planned energy counts `max(p, 0)`
per hour; the row "new energy >= the sum of `max(p_s, 0)`" is written exactly as one row per
subset of the hours where `p_s` can take both signs.

Objective (the plan gives none; a reading): the largest sum, over present cars, of each car's own
worst-case energy at the end of the window. The fleet's total is fixed by the requests, so this
prefers plans in which each car's end energy depends little on the future requests.

Every row must hold for every future request: a row `alpha + sum_k gamma_k zeta_k <= b` is
`alpha + sum_k |gamma_k| <= b` (plan T2(b)), each `|gamma_k|` bounded by a variable. One sparse
programme through `lp.solve_lp`, with the 300-second limit for other programmes by default
(plan section 13). Any outcome but optimal returns no plan; a timeout stays a timeout.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fleetreserve.day import ForecastCar
from fleetreserve.lp import OTHER_SECONDS, Outcome
from fleetreserve.plans import Plan
from fleetreserve.sparse_programme import Sizes, SparseProgramme

SIGN_TOLERANCE = 1e-12  # kW: a planned power within this of 0 does not change sign


@dataclass(frozen=True)
class Remaining:
    """What a rung knows at the start of service hour `hour` (arrays one entry per car).

    `known`: `zeta_1 .. zeta_hour`. `m`, `w`: the sold box; `beta` its scale. `present`: the cars
    still plugged in. `stored`: the energy each car has drawn since the decision time.
    `charge_bound`: an upper bound on any later charge (`inf` when none is known). `departure`: the
    survival-tightened forecast departure of each present car (not read for absent cars)."""

    hour: int
    known: np.ndarray
    m: np.ndarray
    w: np.ndarray
    beta: float
    present: np.ndarray
    stored: np.ndarray
    charge_bound: np.ndarray
    departure: np.ndarray
    forecasts: Sequence[ForecastCar]
    n_recovery: int

    @property
    def n_service(self) -> int:
        return int(np.asarray(self.m).size)

    def request(self) -> float:
        """The request of hour `hour`, known at its start."""
        t = self.hour
        return float(self.m[t - 1] + self.beta * self.w[t - 1] * self.known[t - 1])


@dataclass(frozen=True)
class Guard:
    """The predecessor's plan a rung guards against: `plan`, in force when the rung first left it
    at hour `hour`, with `stored` the energy each car held at the start of that hour. `clip`: the
    plan is run one-way, so a negative planned power counts 0."""

    plan: Plan
    hour: int
    stored: np.ndarray
    clip: bool

    def past(self, hour: int, known) -> np.ndarray:
        """Each car's planned stored energy at the start of `hour` on this plan, from `stored`."""
        powers = [self.plan.powers(s, known) for s in range(self.hour, hour)]
        if self.clip:
            powers = [np.maximum(p, 0.0) for p in powers]
        return np.asarray(self.stored, float) + np.sum(powers, axis=0) if powers else self.stored


@dataclass(frozen=True)
class ResolveResult:
    """`plan` is None unless `outcome` is "optimal". `message`: the solver's own message."""

    outcome: Outcome
    plan: Plan | None
    seconds: float
    late_seconds: float
    sizes: Sizes
    message: str = ""


class _Rows:
    """Rows of one sense collected as coordinate entries, added to the programme at once."""

    def __init__(self):
        self.row, self.column, self.value, self.rhs = [], [], [], []

    def add(self, columns, values, rhs):
        columns = np.asarray(columns, int).reshape(-1)
        self.row.append(np.full(columns.size, len(self.rhs)))
        self.column.append(columns)
        self.value.append(np.broadcast_to(np.asarray(values, float), columns.shape))
        self.rhs.append(float(rhs))

    def flush(self, programme: SparseProgramme, sense: str):
        if not self.rhs:
            return
        programme.rows(
            sense,
            np.concatenate(self.row),
            np.concatenate(self.column),
            np.concatenate(self.value),
            np.array(self.rhs),
        )


class _Builder:
    def __init__(self, n_present: int, length: int):
        self.programme = SparseProgramme()
        self.programme.variables(1, 0.0, 0.0)  # keeps the programme non-empty with no car
        self.length = length
        self.k = length - 1
        j, k = np.meshgrid(np.arange(length), np.arange(self.k), indexing="ij")
        self.allowed = k <= j - 1  # H[., j, k]: hour t + j on zeta_{t + 1 + k}
        shape = (n_present, length, self.k)
        allowed = np.broadcast_to(self.allowed, shape)
        self.h = self.programme.variables((n_present, length))
        self.H = self.programme.variables(
            shape, np.where(allowed, -np.inf, 0.0), np.where(allowed, np.inf, 0.0)
        )
        self.le = _Rows()
        self.eq = _Rows()

    def spread(self, p: int, weights, constant=None) -> np.ndarray:
        """Variables `z_k >= |sum_j weights_j H[p, j, k] + constant_k|`, one per `k` that can be
        non-zero."""
        weights = np.asarray(weights, float)
        constant = np.zeros(self.k) if constant is None else np.asarray(constant, float)
        out = []
        for k in range(self.k):
            hours = np.flatnonzero((weights != 0) & self.allowed[:, k])
            if hours.size == 0 and constant[k] == 0:
                continue
            z = int(self.programme.variables(1, 0.0, np.inf)[0])
            columns = np.concatenate([self.H[p, hours, k], [z]])
            for sign in (1.0, -1.0):
                values = np.concatenate([sign * weights[hours], [-1.0]])
                self.le.add(columns, values, -sign * constant[k])
            out.append(z)
        return np.array(out, dtype=int)

    def robust(self, p: int, weights, spread: np.ndarray, rhs: float):
        """`sum_j weights_j h[p, j] + sum(spread) <= rhs`."""
        weights = np.asarray(weights, float)
        hours = np.flatnonzero(weights != 0)
        columns = np.concatenate([self.h[p, hours], spread])
        values = np.concatenate([weights[hours], np.ones(spread.size)])
        self.le.add(columns, values, rhs)


def _shares(departure: float, hours) -> np.ndarray:
    return np.clip(departure - np.asarray(hours, float), 0.0, 1.0)


def _guard_terms(offset, future, clip: bool):
    """The replaced plan's energy in hours `t..t+j`, per subset: yields `(j, constant, gains)`.

    `offset`, `future`: one car's `Plan.remaining`. With `clip`, a negative planned power counts
    0: hours that are never negative count as planned, hours that are never positive are left out,
    and hours of both signs give one row per subset of them."""
    length = offset.size
    spread = np.abs(future).sum(axis=1)
    low, high = offset - spread, offset + spread
    for j in range(length):
        if not clip:
            yield j, offset[: j + 1].sum(), future[: j + 1].sum(axis=0)
            continue
        always = [s for s in range(j + 1) if low[s] >= -SIGN_TOLERANCE]
        mixed = [s for s in range(j + 1) if low[s] < -SIGN_TOLERANCE and high[s] > SIGN_TOLERANCE]
        for size in range(len(mixed) + 1):
            for chosen in itertools.combinations(mixed, size):
                hours = always + list(chosen)
                yield j, offset[hours].sum(), future[hours].sum(axis=0)


def resolve(
    state: Remaining,
    guard: Guard,
    *,
    two_way: bool,
    budget: float | None,
    time_limit: float = OTHER_SECONDS,
) -> ResolveResult:
    """Re-solve hours `state.hour .. n` for the present cars, guarded by `guard`.

    `two_way`: the new plan may discharge, within `budget` kWh; its plan is then run two-way."""
    t, n = state.hour, state.n_service
    if not 1 <= t <= n:
        raise ValueError(f"a re-solve starts in a service hour (1 to {n}), got {t}")
    if not 1 <= guard.hour <= t:
        raise ValueError(f"the guard's hour {guard.hour} must be a service hour up to {t}")
    if two_way and budget is None:
        raise ValueError("a two-way re-solve needs the budget")
    known = np.asarray(state.known, float)
    present = np.flatnonzero(np.asarray(state.present, bool))
    length = n - t + 1
    b = _Builder(present.size, length)
    hours = np.arange(t, n + 1)
    recovery = np.arange(n + 1, n + 1 + state.n_recovery)
    offset_ref, future_ref = guard.plan.remaining(t, known)
    owed = guard.past(t, known) - np.asarray(state.stored, float)  # planned since a, not drawn

    # The present cars share every remaining request.
    b.eq.add(b.h[:, 0], 1.0, state.request())
    for j in range(1, length):
        b.eq.add(b.h[:, j], 1.0, state.m[t + j - 1])
        for k in range(j):
            target = state.beta * state.w[t + j - 1] if k == j - 1 else 0.0
            b.eq.add(b.H[:, j, k], 1.0, target)

    objective_columns, objective_values = [], []
    for p, i in enumerate(present):
        forecast = state.forecasts[i]
        x = float(state.stored[i])
        f = _shares(float(state.departure[i]), hours)
        high = np.minimum(forecast.rate * f, state.charge_bound[i])
        low = -forecast.rate * f if two_way else np.zeros(length)
        cap = max(forecast.cap, x)
        supply = forecast.rate * _shares(float(state.departure[i]), recovery).sum()
        need = min(forecast.need, min(cap, x + high.sum()) + supply)

        for j in range(length):
            unit = np.eye(length)[j]
            power = b.spread(p, unit)
            b.robust(p, unit, power, high[j])
            b.robust(p, -unit, power, -low[j])
            prefix = (np.arange(length) <= j).astype(float)
            stored = b.spread(p, prefix)
            b.robust(p, prefix, stored, cap - x)
            if two_way:
                b.robust(p, -prefix, stored, budget + x)
        b.robust(p, -np.ones(length), stored, x + supply - need)
        objective_columns += [b.h[p], stored]
        objective_values += [np.ones(length), -np.ones(stored.size)]

        for j, constant, gains in _guard_terms(offset_ref[i], future_ref[i], guard.clip):
            prefix = (np.arange(length) <= j).astype(float)
            b.robust(p, -prefix, b.spread(p, prefix, -gains), -constant - owed[i])

    b.le.flush(b.programme, "le")
    b.eq.flush(b.programme, "eq")
    columns = np.concatenate([np.asarray(c, int).reshape(-1) for c in objective_columns] or [[0]])
    values = np.concatenate([np.asarray(v, float) for v in objective_values] or [[0.0]])
    result, sizes = b.programme.maximise(columns, values, time_limit=time_limit)
    if result.outcome != "optimal":
        return ResolveResult(
            result.outcome, None, result.seconds, result.late_seconds, sizes, result.message
        )

    x = result.x
    constant = np.zeros((guard.plan.n_cars, n + 1))
    gains = np.zeros((guard.plan.n_cars, n + 1, n))
    constant[present, t:] = x[b.h]
    gains[present[:, None, None], hours[None, :, None], np.arange(t, n)[None, None, :]] = np.where(
        b.allowed, x[b.H], 0.0
    )
    plan = Plan(constant, gains, two_way=two_way)
    return ResolveResult(
        "optimal", plan, result.seconds, result.late_seconds, sizes, result.message
    )
