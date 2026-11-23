"""LP (27)'s box over its optimal solutions, as the range a stated search finds (plan sections 1,
10 and 11 (O3)).

LP (27) (`lp24.inner_set(..., "27")`) maximises the trace of `P`. Many solutions share the best
trace, and their certified sets `p + P U0` hold boxes of very different sizes, so "the largest box
in LP (27)'s set" is not one number until a solution is chosen; the solution a solver returns
changes with the order the cars are listed in.

The largest box over every optimal solution is not a linear programme here. A box lies in
`p + P U0` when each of its corners `c` has a base-set profile `v` in `U0` with `(p + P v)_S = c`
(hour 0 free, as in `inner_box`). With the solution `[p, P]` taken from LP (27)'s optimal face and
the profiles `v` both free, `P v` is a product of two sets of variables, so the condition is
bilinear: for a fixed `P` it is a linear programme over the box and the profiles
(`inner_box.vertex_box`), for fixed profiles a linear programme over `P` and the box, but not over
both at once, and the pairs that meet it need not form a convex set. The same holds for the
smallest box over the optimal face (the least, over the face, of each set's largest box). Neither
is computed exactly; both are estimated from inside by the search below, in which every step is a
linear programme:

1. LP (27) on the rows as given (`lp24.inner_set`, the time limit `others`): its trace `T`. The
   caller gives the cars in a fixed order (`o3_day` sorts them), so the search solves the same
   programmes, and finds the same numbers, whatever order the cars were listed in.
2. Candidate solutions on LP (27)'s optimal face, each from one programme with the limit
   `others`; the face's programmes are LP (27)'s rows with the trace at least
   `T - FACE_TOLERANCE * max(1, T)`:
   a. the solution of step 1;
   b. LP (27) with the cars in the reverse of the given order;
   c. for each draw `k` in `1..DRAWS`, the face's solution that maximises a linear objective whose
      coefficients (one per map variable `gamma_i`, `Gamma_i`, in the order of the cars) are
      standard normal draws of `numpy.random.default_rng([SEED, k])`;
   d. for each direction set (the unit vectors of the service hours together; the all-ones
      vector; and, for each `k` in `1..RANGE_DRAWS`, one vector of signs drawn by
      `numpy.random.default_rng([SEED, 0, k])`), the face's solution with the smallest sum, over
      the set's vectors `a`, of the range of `a . (P v)_S` over `v` in `U0` (each range written
      by linear-programming duality); for the unit vectors that sum is at least the set's hourly
      box;
   e. `MIXTURES` convex combinations of 2 or 3 of the solutions of a to d (`mixtures`, seeded):
      the face is convex, so each is an optimal solution too, found without a solve.
   A solution of b to e is used only when its own trace (`trace_of`, from its map) is at least
   `T - FACE_CHECK * max(1, T)` (`on_face`).
3. Each candidate's largest box for every product: one programme over the box's corners, hour 0
   free per corner, as `inner_box.vertex_box`, solved by the dual simplex method (the limit
   `box`).
4. Smallest found: the least of the candidates' boxes.
5. Largest found: rounds of ascent, the hourly product first. Starts: the base set's own largest
   box (its corner profiles, hour 0 at 0, counted as a box of 0), the `ASCENT_STARTS` solutions
   of a to d and the `MIXTURE_STARTS` mixtures with the largest boxes and, for the block product,
   the `OTHER_STARTS` largest block boxes of the maps the hourly ascents reached (one programme
   per map, the limit `box`). With each corner's
   base-set profile from the last box held fixed, one programme over the face's maps and the box
   (each corner met as `p + P v` through its fixed profile; the limit `others`) gives a map whose
   set holds a box at least as large; that map's largest box is solved again (the limit `box`).
   Each step is a linear programme because one of the two factors of `P v` is held fixed in it.
   The ascent ends after `ROUNDS` rounds, after a round that gains less than
   `GAIN_TOLERANCE * max(1, value)`, when the map's own trace fails `on_face`, or when its
   programme ends "infeasible" or with a solver error (`PASSED_OVER`): the profiles of a solved
   box meet their corners only to the solver's accuracy, so the programme can be found
   infeasible though the last map meets it, and from the base set's box no map of the face may
   meet the fixed profiles at all; the box reached is kept. The largest box reached from any
   start or candidate; the hourly value is also at least the block value, since a block box is an
   hourly box of the same value.

The search is a local search from many starts, so a search with other starts can find a larger
box than its largest found or a smaller one than its smallest found; nothing here bounds by how
much (`o3_lp27_bound` bounds the largest from above).

Settings. The counts above are the full settings (`full_effort`, from `DRAWS` to `ROUNDS`); a
caller may pass fewer (`Effort`, `scaled_effort`). A smaller effort searches fewer solutions, so
the two statements below stay true; the seeded steps keep their seeds, so its objectives and
sign vectors are the first ones of the full settings.

The face's tolerance. `FACE_TOLERANCE` is 1e-11 and `FACE_CHECK` 2e-11, both relative: the
floor is `T - 1e-11 * max(1, T)`, with as much again for rounding. It is this small because the
largest box a set holds is not continuous in the map: a set flat in some hours holds a box with
width in another hour only while it is not tilted into the flat hours, so a map whose trace is a
little below `T` can hold a far smaller box than every optimal solution (a made-up day of three
cars: within 1e-7 of `T` the smallest box found was 1.620 kWh, while every optimal solution's
set holds at least 2.026, the test's case). In kWh the slack is `1e-11 * max(1, T)`: four orders
below the solver's feasibility tolerance (1e-7, HiGHS's default) only while `T` is at most 1,
about two orders at `T` near 65 (a made-up day of 20 cars and 6 hours), and less as `T` grows.
A solution searched is optimal for LP (27) only to within that slack, and a map so accepted can
hold a far smaller box than every optimal solution when read by a box programme finer than this
one's (a made-up day of 60 cars, `T` = 51.6: 5.78 kWh at a relative gap of 1.5e-11, against at
least 40.52 held by every optimal solution); this module's box programme (the dual simplex
method, tolerances 1e-7) was not seen to resolve such tilts below about 1e-8 kWh. So, to the
solver's precision and with this box programme only:

- the smallest found is at least the smallest box over all of LP (27)'s optimal solutions: an
  estimate of it from above, not a bound from below;
- the largest found is at most the largest over all of them (it can exceed the ceiling by about
  1e-10 kWh for the same reason): an estimate from below, from a local search that does not find
  the largest over all of them;
- how far either lies from the value over all optimal solutions is not known; both lie between 0
  and the ceiling. `o3_lp27_bound` proves a number at least the largest over all of them.

Passed over and counted in `skipped`, never used: a solution of step 2 or of the ascent whose own
trace fails `on_face`; a candidate programme of step 2 that ends "infeasible" or with a solver
error (`lp.solve_lp`'s "error"), since it gives no solution on the face; a candidate one of whose
box programmes ends with a solver error (HiGHS was seen to end a feasible box programme of a
nearly degenerate map so, by either method), left out for every product so that every product's
values come from the same candidates; a block start of step 5 whose box programme so ends; an
ascent ended as above. Leaving a solution out searches
fewer solutions, so both statements above stay true; when every candidate is left out the
values are missing ("error"). Any other solve that ends in any outcome but optimal (a timeout,
an unbounded candidate programme, an infeasible or unbounded box programme) makes both values of
every product it reaches missing, with that outcome; a timeout is never zero.
"""

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from fleetreserve.base_sets import BaseSet
from fleetreserve.carsets import CarRows, FleetRows, stack
from fleetreserve.lp import solve_lp
from fleetreserve.lp24 import _build as lp27_programme  # LP (27)'s rows, as lp24 builds them
from fleetreserve.lp24 import inner_set
from fleetreserve.products import Product, corner_paths
from fleetreserve.sparse_programme import SparseProgramme

DRAWS = 40  # seeded random objectives on the optimal face (step 2c)
RANGE_DRAWS = 4  # seeded random weights of the hours' ranges (step 2d)
MIXTURES = 100  # seeded convex combinations of the solutions of steps 2a to 2d (step 2e)
SEED = 27
ROUNDS = 12  # ascent rounds per start and product (step 5)
ASCENT_STARTS = 8  # solutions of steps 2a to 2d with the largest boxes that step 5 starts from
MIXTURE_STARTS = 4  # mixtures (step 2e) with the largest boxes that step 5 starts from
OTHER_STARTS = 2  # largest block boxes of the hourly ascents' maps that step 5 starts from
FACE_TOLERANCE = 1e-11  # relative, on LP (27)'s trace: the floor of the face's programmes
FACE_CHECK = 2e-11  # relative: a solution whose own trace is further below LP (27)'s is not used
GAIN_TOLERANCE = 1e-7  # relative, on the box value
OPTIMAL = "optimal"


@dataclass(frozen=True)
class Effort:
    """The search's settings (module text): seeded objectives (step 2c), sign vectors (2d),
    mixtures (2e), ascent starts from steps 2a to 2d and from the mixtures, block starts from the
    hourly ascents' maps, and rounds per ascent (step 5)."""

    draws: int
    range_draws: int
    mixtures: int
    ascent_starts: int
    mixture_starts: int
    other_starts: int
    rounds: int


def full_effort() -> Effort:
    """The module's settings (`DRAWS` and the rest), read when called."""
    return Effort(DRAWS, RANGE_DRAWS, MIXTURES, ASCENT_STARTS, MIXTURE_STARTS, OTHER_STARTS, ROUNDS)


def scaled_effort(share: float) -> Effort:
    """`full_effort()` with the objectives, sign vectors, mixtures and the starts from steps 2a
    to 2e times `share` (0 < share <= 1), rounded up, so each keeps at least one; the block
    starts from the hourly ascents' maps and the rounds are kept (an ascent ends as soon as a
    round gains nothing)."""
    if not (0 < share <= 1):
        raise ValueError(f"the effort share must be in (0, 1], got {share}")
    full = full_effort()
    counts = {
        name: max(1, math.ceil(share * getattr(full, name) - 1e-9))
        for name in ("draws", "range_draws", "mixtures", "ascent_starts", "mixture_starts")
    }
    return Effort(**counts, other_starts=full.other_starts, rounds=full.rounds)


SOLVER_ERROR = "error"  # `lp.solve_lp`'s outcome for a solver failure
PASSED_OVER = ("infeasible", SOLVER_ERROR)  # outcomes of a search programme passed over


@dataclass(frozen=True)
class Lp27Range:
    """LP (27)'s trace (None unless `trace_outcome` is optimal) and, per product name, the
    smallest and the largest box found (module text) in kWh, or None (missing) with the outcome
    in `outcomes`. `skipped` counts the solutions and programmes passed over (module text);
    `seconds` sums the solves of each step."""

    trace: float | None
    trace_outcome: str
    low: dict[str, float | None]
    high: dict[str, float | None]
    outcomes: dict[str, str]
    skipped: int = 0
    seconds: dict[str, float] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class CornerBox:
    """A box's value, each corner's base-set profile `(corners, base-set hours)` and the map
    `(p, P)` whose set holds it; all None unless `outcome` is optimal."""

    outcome: str
    value: float | None
    profiles: np.ndarray | None
    maps: tuple | None = None


def _box_columns(lp: SparseProgramme, product: Product, n: int):
    m = lp.variables(n)
    if product.kind == "hourly":
        width = lp.variables(n, 0.0, np.inf)
        objective = (width, np.full(n, 2.0))
    elif product.kind == "block":
        width = np.full(n, int(lp.variables(1, 0.0, np.inf)[0]))
        objective = (width[:1], np.array([2.0 * n]))
    else:
        raise ValueError("only the hourly and the block product are built here")
    return m, width, objective


def _corner_rows(n, q, profile_columns, profile_values, constant_columns, m, width):
    """`sum_b P[t, b] v_kb + c_t - m_t - zeta_kt w_t = rhs` for every corner `k` and service hour
    `t`: the product terms given per corner and hour, `(corners, n, q)` columns and values."""
    paths = corner_paths(n)
    corners = paths.shape[0]
    t_rows = np.repeat(np.arange(corners * n), q)
    rows = [t_rows, np.arange(corners * n), np.arange(corners * n)]
    columns = [profile_columns.reshape(-1), np.tile(m, corners), np.tile(width, corners)]
    values = [profile_values.reshape(-1), -np.ones(corners * n), -paths.reshape(-1)]
    if constant_columns is not None:
        rows.append(np.arange(corners * n))
        columns.append(np.tile(constant_columns, corners))
        values.append(np.ones(corners * n))
    return [np.concatenate(part) for part in (rows, columns, values)]


class _SimplexProgramme(SparseProgramme):
    """A `SparseProgramme` solved by the dual simplex method (`lp.solve_lp`, method "simplex")."""

    def maximise(self, column, value, time_limit):
        cost = np.zeros(self.n_variables)
        np.add.at(cost, np.asarray(column, int).reshape(-1), -np.asarray(value, float).reshape(-1))
        a_ub, b_ub = self._matrix("le")
        a_eq, b_eq = self._matrix("eq")
        bounds = np.column_stack([np.concatenate(self._low), np.concatenate(self._high)])
        result = solve_lp(
            cost,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            time_limit=time_limit,
            method="simplex",
        )
        return result, None


def largest_box(p, P, base: BaseSet, product: Product, time_limit: float):
    """The largest box of `product` in `p + P U0`, hour 0 free per corner (the programme of
    `inner_box.vertex_box`, here by the dual simplex method, whose vertex gives each corner's
    profile to the solver's accuracy), with each corner's base-set profile; and the seconds."""
    p = np.asarray(p, dtype=float)
    map_s = np.asarray(P, dtype=float)[1:][:, base.columns]  # (n, q)
    n, q = map_s.shape
    corners = 2**n
    lp = _SimplexProgramme()
    m, width, objective = _box_columns(lp, product, n)
    power = base.power[base.columns]
    u = lp.variables((corners, q), 0.0, np.broadcast_to(power, (corners, q)))
    lp.rows(
        "le",
        np.repeat(np.arange(corners), q),
        u.reshape(-1),
        np.ones(corners * q),
        np.full(corners, base.energy),
    )
    profile_columns = np.repeat(u[:, None, :], n, axis=1)  # (corners, n, q)
    profile_values = np.broadcast_to(map_s, (corners, n, q))
    lp.rows(
        "eq",
        *_corner_rows(n, q, profile_columns, profile_values, None, m, width),
        -np.tile(p[1:], corners),
    )
    result, _ = lp.maximise(*objective, time_limit=time_limit)
    if result.outcome != OPTIMAL:
        return CornerBox(result.outcome, None, None), result.seconds
    value = float(2.0 * np.maximum(result.x[width], 0.0).sum())
    return CornerBox(OPTIMAL, value, result.x[u], (p, np.asarray(P, dtype=float))), result.seconds


def _face(rows: FleetRows, base: BaseSet, trace: float, tolerance: float | None = None):
    """LP (27)'s programme with the trace held at its optimum (within `tolerance`, relative;
    None: `FACE_TOLERANCE`), and the service-hour sums `p_t = sum_i gamma_i(t)`,
    `P[t, b] = sum_i Gamma_i(t, b)` as variables."""
    lp, _, gamma, gain, (columns, values) = lp27_programme(rows, base, "27")
    tolerance = FACE_TOLERANCE if tolerance is None else tolerance
    floor = trace - tolerance * max(1.0, abs(trace))
    lp.rows("le", np.zeros(columns.size), columns, -values, [-floor])
    n, cars, q = rows.n_service, rows.n_cars, base.columns.size
    p_s = lp.variables(n)
    map_s = lp.variables((n, q))
    lp.rows(
        "eq",
        np.concatenate([np.repeat(np.arange(n), cars), np.arange(n)]),
        np.concatenate([gamma[:, 1:].T.reshape(-1), p_s]),
        np.concatenate([np.ones(n * cars), -np.ones(n)]),
        np.zeros(n),
    )
    lp.rows(
        "eq",
        np.concatenate([np.repeat(np.arange(n * q), cars), np.arange(n * q)]),
        np.concatenate([gain[:, 1:, :].transpose(1, 2, 0).reshape(-1), map_s.reshape(-1)]),
        np.concatenate([np.ones(n * q * cars), -np.ones(n * q)]),
        np.zeros(n * q),
    )
    return lp, gamma, gain, p_s, map_s


def _maps(x, rows: FleetRows, base: BaseSet, gamma, gain) -> tuple[np.ndarray, np.ndarray]:
    """`p` and `P` (zero columns outside the base-set hours) of a solution."""
    hours = rows.n_service + 1
    big = np.zeros((hours, hours))
    big[:, base.columns] = x[gain].sum(axis=0)
    return x[gamma].sum(axis=0), big


def _solve(lp, objective, rows, base, gamma, gain, time_limit):
    result, _ = lp.maximise(*objective, time_limit=time_limit)
    if result.outcome != OPTIMAL:
        return result.outcome, None, result.seconds
    return OPTIMAL, _maps(result.x, rows, base, gamma, gain), result.seconds


def reversed_cars(rows: FleetRows) -> FleetRows:
    """The same cars, listed in the reverse order."""
    cars = []
    for i in reversed(range(rows.n_cars)):
        mine = rows.car == i
        cars.append(
            CarRows(
                rows.n_service,
                rows.coefficients[mine],
                rows.bounds[mine],
                tuple(rows.kinds[mine]),
            )
        )
    return stack(cars)


def _drawn(rows, base, trace, k, time_limit):
    """Step 2c: the face's solution for the `k`-th seeded random objective."""
    lp, gamma, gain, _, _ = _face(rows, base, trace)
    columns = np.concatenate([gamma.reshape(-1), gain.reshape(-1)])
    values = np.random.default_rng([SEED, k]).standard_normal(columns.size)
    return _solve(lp, (columns, values), rows, base, gamma, gain, time_limit)


def trace_of(P, base: BaseSet) -> float:
    """The trace of a map over the base-set hours (LP (27)'s objective, as `lp24.inner_set`)."""
    columns = base.columns
    return float(np.trace(np.asarray(P, dtype=float)[np.ix_(columns, columns)]))


def on_face(P, base: BaseSet, trace: float) -> bool:
    """Whether a map's own trace is within `FACE_CHECK` (relative) of LP (27)'s trace `trace`."""
    return trace_of(P, base) >= trace - FACE_CHECK * max(1.0, abs(trace))


def _least_range(rows, base, trace, time_limit, directions):
    """Step 2d: the face's solution with the smallest sum, over the rows `a` of `directions`
    `(k, n)`, of the range of `a . (P v)_S` over `v` in `U0`."""
    lp, gamma, gain, _, map_s = _face(rows, base, trace)
    n, q = map_s.shape
    directions = np.asarray(directions, float).reshape(-1, n)
    k = directions.shape[0]
    power = base.power[base.columns]
    terms_columns, terms_values = [], []
    for sign in (1.0, -1.0):
        # max over U0 of sign * a . (P v)_S <= power . w + energy s when
        # w_b + s >= sign * sum_t a_t P[t, b] for every base-set hour b (duality).
        w = lp.variables((k, q), 0.0, np.inf)
        s = lp.variables(k, 0.0, np.inf)
        row = np.arange(k * q).reshape(k, q)
        lp.rows(
            "le",
            np.concatenate(
                [
                    np.repeat(row[:, None, :], n, axis=1).reshape(-1),
                    row.reshape(-1),
                    row.reshape(-1),
                ]
            ),
            np.concatenate(
                [
                    np.broadcast_to(map_s, (k, n, q)).reshape(-1),
                    w.reshape(-1),
                    np.repeat(s, q),
                ]
            ),
            np.concatenate(
                [
                    sign * np.repeat(directions[:, :, None], q, axis=2).reshape(-1),
                    -np.ones(2 * k * q),
                ]
            ),
            np.zeros(k * q),
        )
        terms_columns += [w.reshape(-1), s]
        terms_values += [-np.tile(power, k), np.full(k, -base.energy)]
    objective = (np.concatenate(terms_columns), np.concatenate(terms_values))
    return _solve(lp, objective, rows, base, gamma, gain, time_limit)


def _directions(n: int, range_draws: int | None = None) -> list[np.ndarray]:
    """Step 2d's direction sets: the unit vectors together, the all-ones vector, and
    `range_draws` (None: `RANGE_DRAWS`) seeded vectors of signs."""
    found = [np.eye(n), np.ones((1, n))]
    for k in range(1, (RANGE_DRAWS if range_draws is None else range_draws) + 1):
        signs = np.random.default_rng([SEED, 0, k]).choice([-1.0, 1.0], size=n)
        found.append(signs[None, :])
    return found


def _ascent_step(rows, base, trace, product, profiles, time_limit):
    """Step 5: the face's map and box with each corner met through its fixed base-set profile."""
    n = rows.n_service
    q = base.columns.size
    lp, gamma, gain, p_s, map_s = _face(rows, base, trace)
    m, width, objective = _box_columns(lp, product, n)
    corners = 2**n
    profile_columns = np.broadcast_to(map_s, (corners, n, q))
    profile_values = np.repeat(profiles[:, None, :], n, axis=1)
    lp.rows(
        "eq",
        *_corner_rows(n, q, profile_columns, profile_values, p_s, m, width),
        np.zeros(corners * n),
    )
    return _solve(lp, objective, rows, base, gamma, gain, time_limit)


def base_box_profiles(base: BaseSet, product: Product) -> np.ndarray:
    """The corner profiles `(corners, base-set hours)` of the base set's own largest box of the
    product on the service hours, hour 0 at 0 (the closed form of `inner_box.scaled_box` with
    `alpha = 1` and `p = 0`): the profiles step 5's first ascent starts from."""
    n = base.n_service
    service = base.power[1:]
    if product.kind == "hourly":
        total = service.sum()
        width = service * (min(1.0, base.energy / total) if total > 0 else 0.0)
    else:
        width = np.full(n, min(service.min(), base.energy / n))
    hours = np.zeros((2**n, n + 1))
    hours[:, 1:] = width * (1.0 + corner_paths(n)) / 2.0
    return hours[:, base.columns]


def _ascend(rows, base, trace, product, current, others, box, seconds, rounds):
    """Step 5 from one box: (outcome, the box reached, whether a solution or programme passed
    over ended it)."""
    for _ in range(rounds):
        outcome, maps, spent = _ascent_step(rows, base, trace, product, current.profiles, others)
        seconds["lp27_ascent"] += spent
        if outcome in PASSED_OVER or (outcome == OPTIMAL and not on_face(maps[1], base, trace)):
            return OPTIMAL, current, True
        if outcome != OPTIMAL:
            return outcome, current, False
        after, spent = largest_box(*maps, base, product, box)
        seconds["lp27_ascent"] += spent
        if after.outcome == SOLVER_ERROR:
            return OPTIMAL, current, True
        if after.outcome != OPTIMAL:
            return after.outcome, current, False
        gained = after.value - current.value
        if gained > 0:
            current = after
        if gained < GAIN_TOLERANCE * max(1.0, current.value):
            break
    return OPTIMAL, current, False


def _flipped(rows, base, time_limit):
    """Step 2b: LP (27) with the cars in the reverse of the given order."""
    found = inner_set(reversed_cars(rows), base, "27", time_limit=time_limit)
    maps = (found.p, found.P) if found.outcome == OPTIMAL else None
    return found.outcome, maps, found.seconds


def _candidate_steps(rows, base, trace, others, effort: Effort):
    """Step 2's programmes, each a function giving (outcome, maps or None, seconds)."""
    steps = [functools.partial(_flipped, rows, base, others)]
    for k in range(1, effort.draws + 1):
        steps.append(functools.partial(_drawn, rows, base, trace, k, others))
    for directions in _directions(rows.n_service, effort.range_draws):
        steps.append(functools.partial(_least_range, rows, base, trace, others, directions))
    return steps


def mixtures(candidates, count: int) -> list:
    """Step 2e: `count` seeded convex combinations of 2 or 3 distinct candidates, each map
    `(p, P)` the same combination of theirs (`numpy.random.default_rng([SEED, 2])`: the number of
    candidates, which ones, then Dirichlet(1, ..., 1) weights)."""
    if len(candidates) < 2:
        return []
    rng = np.random.default_rng([SEED, 2])
    found = []
    for _ in range(count):
        k = int(rng.integers(2, min(3, len(candidates)) + 1))
        pick = rng.choice(len(candidates), size=k, replace=False)
        weights = rng.dirichlet(np.ones(k))
        p = sum(
            w * np.asarray(candidates[i][0], dtype=float)
            for w, i in zip(weights, pick, strict=True)
        )
        P = sum(
            w * np.asarray(candidates[i][1], dtype=float)
            for w, i in zip(weights, pick, strict=True)
        )
        found.append((p, P))
    return found


def _boxes(candidates, base, products, box, seconds):
    """Steps 3 and 4 for every product at once: (each product's boxes, the outcomes of the
    products made missing, the candidates passed over)."""
    boxes = {name: [] for name in products}
    missing, skipped = {}, 0
    for index, (p, P) in enumerate(candidates):
        solved = {}
        for name in products:
            if name not in missing:
                solved[name], spent = largest_box(p, P, base, Product(name), box)
                seconds["lp27_boxes"] += spent
        for name, found in solved.items():
            if found.outcome not in (OPTIMAL, SOLVER_ERROR):
                missing[name] = found.outcome
        if any(found.outcome == SOLVER_ERROR for found in solved.values()):
            skipped += 1  # passed over for every product, so all share one set of candidates
            continue
        for name, found in solved.items():
            if name not in missing:
                boxes[name].append((index, found))
    return boxes, missing, skipped


def _top(boxes, count):
    """The `count` boxes with the largest values, in their given order (ties: the earlier)."""
    ranked = sorted(range(len(boxes)), key=lambda k: -boxes[k].value)[:count]
    return [boxes[k] for k in sorted(ranked)]


def _largest(rows, base, trace, product, boxes, solved, extra, limits, seconds, effort: Effort):
    """Step 5 for one product: (outcome, the largest box, the box reached from each start the
    ascent moved from, ascents ended by a solution or programme passed over). `boxes` are
    `(candidate index, box)`; the first `solved` candidates are those of steps 2a to 2d;
    `limits` is `(others, box)`."""
    others, box = limits
    starts = [CornerBox(OPTIMAL, 0.0, base_box_profiles(base, product))]
    starts += _top([b for k, b in boxes if k < solved], effort.ascent_starts)
    starts += _top([b for k, b in boxes if k >= solved], effort.mixture_starts) + list(extra)
    best = _top([b for _, b in boxes], 1)[0]
    reached_all, stops = [], 0
    for start in starts:
        outcome, reached, stopped = _ascend(
            rows, base, trace, product, start, others, box, seconds, effort.rounds
        )
        if outcome != OPTIMAL:
            return outcome, None, [], stops
        stops += stopped
        if reached is not start:
            reached_all.append(reached)
        if reached.value > best.value:
            best = reached
    return OPTIMAL, best, reached_all, stops


def _other_product_starts(reached, base, product, box, seconds, count):
    """The `count` largest boxes of `product` held by the maps the other product's ascents
    reached (step 5): (outcome, the boxes, programmes passed over)."""
    starts, skipped, seen = [], 0, []
    for found in reached:
        if found.maps is None or any(found.maps is maps for maps in seen):
            continue
        seen.append(found.maps)
        start, spent = largest_box(*found.maps, base, product, box)
        seconds["lp27_ascent"] += spent
        if start.outcome == SOLVER_ERROR:
            skipped += 1
        elif start.outcome != OPTIMAL:
            return start.outcome, [], skipped
        else:
            starts.append(start)
    return OPTIMAL, _top(starts, count), skipped


def search(
    rows: FleetRows,
    base: BaseSet,
    products,
    others: float,
    box: float,
    effort: Effort | None = None,
) -> Lp27Range:
    """The search of the module text on `rows` (the cars in the caller's fixed order) for each
    product name in `products`; `others` and `box` are time limits in seconds; `effort` the
    settings (None: `full_effort()`)."""
    effort = full_effort() if effort is None else effort
    seconds = dict.fromkeys(("lp27", "lp27_candidates", "lp27_boxes", "lp27_ascent"), 0.0)
    low = dict.fromkeys(products)
    high = dict.fromkeys(products)
    inner = inner_set(rows, base, "27", time_limit=others)
    seconds["lp27"] += inner.seconds
    if inner.outcome != OPTIMAL:
        missing = dict.fromkeys(products, inner.outcome)
        return Lp27Range(None, inner.outcome, low, high, missing, 0, seconds)
    trace = float(inner.value)
    candidates, skipped = [(inner.p, inner.P)], 0
    for step in _candidate_steps(rows, base, trace, others, effort):
        outcome, maps, spent = step()
        seconds["lp27_candidates"] += spent
        if outcome == OPTIMAL and on_face(maps[1], base, trace):
            candidates.append(maps)
        elif outcome == OPTIMAL or outcome in PASSED_OVER:
            skipped += 1
        else:
            missing = dict.fromkeys(products, outcome)
            return Lp27Range(trace, OPTIMAL, low, high, missing, skipped, seconds)
    solved = len(candidates)
    for maps in mixtures(candidates, effort.mixtures):
        if on_face(maps[1], base, trace):
            candidates.append(maps)
        else:
            skipped += 1
    boxes, outcomes, more = _boxes(candidates, base, products, box, seconds)
    skipped += more
    reached = []
    for name in sorted(products, key=lambda name: name != "hourly"):  # hourly first
        if name in outcomes:
            continue
        if not boxes[name]:
            outcomes[name] = SOLVER_ERROR
            continue
        outcome, extra, more = _other_product_starts(
            reached, base, Product(name), box, seconds, effort.other_starts
        )
        skipped += more
        if outcome == OPTIMAL:
            outcome, best, reached, more = _largest(
                rows,
                base,
                trace,
                Product(name),
                boxes[name],
                solved,
                extra,
                (others, box),
                seconds,
                effort,
            )
            skipped += more
        outcomes[name] = outcome
        if outcome == OPTIMAL:
            low[name] = min(b.value for _, b in boxes[name])
            high[name] = best.value
    if all(outcomes.get(name) == OPTIMAL for name in ("hourly", "block")):
        high["hourly"] = max(high["hourly"], high["block"])
    outcomes = {name: outcomes[name] for name in products}
    return Lp27Range(trace, OPTIMAL, low, high, outcomes, skipped, seconds)
