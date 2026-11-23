"""One day's comparison of T1(b) with the `flexitroid` package (plan section 4, T1; 13, WP4).

The plan's sentence is "`flexitroid` matches (b) on 50 development days". T1(b) is the largest
scale of a box with centre `c` and widths `w_up`, `w_down` inside the fleet set on the service
hours `S` (hour 0 and the recovery hours free):

    lam* = min over subsets A of S of min{(b(A) - c(A)) / w_up(A), (c(A) - p(A)) / w_down(A)},

zero widths left out (`exact.box_scale`). `flexitroid` of Mukhi et al. (2026) represents each car
as a g-polymatroid through its own set functions, sums them, and optimises a linear cost over the
sum by the greedy algorithm (`Flexitroid.solve_linear_program`).

What is compared, on one day's true one-way sets (`Day.one_way_cars`):

1. Set functions. Each car is `flexitroid.devices.level1.V1G(T, a=0, d=departure, u_max=rate,
   e_min=(1 - rho) * room, e_max=room)` with `T` the day's hours (hour 0 is flexitroid's time 0).
   Both of flexitroid's aggregates of these cars, the general `Aggregator` (the sum of the cars'
   own functions) and `V1GAggregator` (its closed form for such cars), must give the same `p(A)`
   and `b(A)` as `exact.day_functions` on every subset `A` of `S`, within `SET_TOLERANCE`
   (relative to the larger of 1 and the project's value, in kWh). flexitroid's functions on a
   subset of `S` count every other hour of the day as outside `A`, which is the projection onto
   `S` with hour 0 and the recovery hours free.
2. The formula on flexitroid's functions: `exact.box_scale` applied to the general aggregate's
   `p` and `b` gives the same scale as on the project's, within `SCALE_TOLERANCE`.
3. The scale by flexitroid's optimisation. The largest `lam` with the whole box inside the sum is
   computed without the formula: the box is inside a convex set exactly when each of its corners
   is, and a corner `c + lam w` (one direction `w` per corner) stays inside up to
   `lam_w = min over d with w . d >= 1 of max_{u in set} d . u - c . d`. That minimum is found by
   cutting planes: a small linear programme over `d` holds one row per vertex found so far, and
   flexitroid's greedy algorithm, given the cost `-d` on the service hours and 0 on the other
   hours, returns the vertex that maximises `d . u` over the whole set. The programme's value is
   a lower bound and the vertex's value an upper bound on `lam_w`; a corner is finished when they
   are within `GAP_TOLERANCE`, or as soon as its lower bound is not below the smallest upper bound
   found for any corner (it cannot be the smallest). The scale's bounds are the smallest lower and
   upper bounds over the corners; both must be within `SCALE_TOLERANCE` (relative to the larger
   of 1 and the project's scale) of `exact.box_scale` on the project's functions.

The shapes (the centre and widths given to both). The plan scales "a given shape (for offers,
the certified one)"; development days carry no certified shape here (it needs the gate's
forecast choices), so each day gets `SHAPES` shapes on the same centre:

- centre: the mean of the vertices flexitroid's greedy algorithm returns for the costs `+1` and
  `-1` on each service hour alone (0 elsewhere), `2 |S|` vertices. It is a point of flexitroid's
  set by construction, so every cutting-plane programme is bounded; `exact.box_scale` refuses it
  if it lies outside the project's set, which is then a difference.
- the hour's range: the largest minus the smallest power of the hour over those vertices; a range
  at or below `RANGE_TOLERANCE` is set to 0. The greedy algorithm gives each power as a difference
  of sums, so an hour whose power is fixed can show a range of a few 1e-15 kW; such a width would
  give the scale 0 by a rounding error and is not a test of anything.
- shape 0: both widths equal to the range; shapes 1 and up: each width the range times a seeded
  uniform draw from 0 to 1, set to 0 with probability `ZERO_WIDTH_SHARE`. The draw is seeded by
  the day's label (`shape_rng`), so it does not depend on the order days are run in.
  A shape whose widths are all 0 has no limit on either side (infinity); it is reported as such.

Verdict of a day: "differs" when any comparison above is outside its tolerance, when flexitroid's
centre is outside the project's set, or the bounds of item 3 do not hold the project's scale;
otherwise "could not run" when flexitroid raised an error, a programme did not end optimal (a
timeout included), or a corner did not close within `MAX_ROUNDS` rounds; otherwise "match".
A day on which flexitroid errors or times out is never a match.

Readings, marked (the plan's sentence does not decide them):

- Strict reading, built here: "matches (b)" means items 1, 2 and 3 all hold. The other reading,
  items 1 and 2 only (flexitroid's set functions, put into T1(b)'s formula), never uses
  flexitroid's optimisation; it is reported beside the strict verdict
  (`DayComparison.functions_only_verdict`) and does not decide anything.
- One-way only. The development days' products are on the true one-way sets, and T1's closed
  forms are the one-way ones. The two-way true sets (plan section 9, `day.StoredEnergyCar`) are
  not compared; flexitroid's general device recursion for them is not called.
- Shapes: seeded shapes on a centre built from flexitroid's vertices, as above, not the
  certified shape of an offer.

Every linear programme goes through `lp.solve_lp` (dual simplex) with the time limit
`lp.OTHER_SECONDS`. flexitroid's own column-generation classes (its `problems` folder) solve with
no time limit and are not used.
"""

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from fleetreserve import exact
from fleetreserve.day import Day
from fleetreserve.lp import OTHER_SECONDS, solve_lp

SET_TOLERANCE = 1e-9  # relative to max(1, |value|), kWh
SCALE_TOLERANCE = 1e-6  # relative to max(1, project's scale)
GAP_TOLERANCE = 1e-9  # relative to max(1, |lower bound|): a corner is finished
FEASIBLE_TOLERANCE = 1e-9  # slack allowed on w . d >= 1 in the cutting-plane programme
RANGE_TOLERANCE = 1e-9  # kW: an hour's range at or below this is 0 (rounding in the greedy sums)
MAX_ROUNDS = 500  # cutting-plane rounds per corner
SHAPES = 3
ZERO_WIDTH_SHARE = 0.25

MATCH = "match"
DIFFERS = "differs"
COULD_NOT_RUN = "could not run"

Ruler = Callable[[Day], exact.SetFunctions]


class CouldNotRun(Exception):
    """flexitroid or a programme gave no answer for the day."""


@dataclass(frozen=True)
class ShapeResult:
    """One shape: the project's scale against flexitroid's (formula and optimisation)."""

    centre: tuple[float, ...]
    w_up: tuple[float, ...]
    w_down: tuple[float, ...]
    exact_scale: float | None  # None when the project refused the centre
    formula_scale: float | None  # T1(b)'s formula on flexitroid's functions
    lower: float | None  # flexitroid's optimisation: bounds on the scale
    upper: float | None
    corners: int  # distinct corner directions
    rounds: int  # cutting-plane programmes solved
    problems: tuple[str, ...]  # differences found, empty when the shape matches
    could_not_run: str | None  # why flexitroid's optimisation gave no bounds


@dataclass(frozen=True)
class DayComparison:
    label: str
    cars: int
    n_service: int
    n_hours: int
    verdict: str
    functions_only_verdict: str
    max_set_difference: float | None  # largest relative difference over p, b, both aggregates
    shapes: tuple[ShapeResult, ...]
    vertices: int  # vertices of flexitroid's set found by its greedy algorithm
    problems: tuple[str, ...]  # differences
    could_not_run: tuple[str, ...]
    seconds: float = field(compare=False)


def flexitroid_aggregates(day: Day):
    """flexitroid's general and closed-form aggregates of the day's one-way true cars."""
    from flexitroid.aggregations.aggregator import Aggregator
    from flexitroid.aggregations.v1g_aggregator import V1GAggregator
    from flexitroid.devices.level1 import V1G

    devices = [
        V1G(day.n_hours, 0, car.departure, car.rate, car.need(day.rho), car.room)
        for car in day.cars
    ]
    return Aggregator(devices), V1GAggregator(devices)


def masks_hours(n_service: int) -> list[set[int]]:
    """The subset of service hours of every mask: bit `k` is hour `k + 1`."""
    return [{k + 1 for k in range(n_service) if mask >> k & 1} for mask in range(2**n_service)]


def flexitroid_functions(aggregate, n_service: int) -> exact.SetFunctions:
    hours = masks_hours(n_service)
    lower = [float(aggregate.p(subset)) for subset in hours]
    upper = [float(aggregate.b(subset)) for subset in hours]
    return exact.SetFunctions(n_service, np.array(lower), np.array(upper))


def relative_difference(found: np.ndarray, wanted: np.ndarray) -> float:
    return float(np.max(np.abs(found - wanted) / np.maximum(1.0, np.abs(wanted))))


def greedy_service_vertex(aggregate, n_hours: int, n_service: int, cost_on_service) -> np.ndarray:
    """flexitroid's greedy vertex minimising `cost . u` (cost 0 outside `S`), on `S` only."""
    cost = np.zeros(n_hours)
    cost[1 : n_service + 1] = cost_on_service
    vertex = np.asarray(aggregate.solve_linear_program(cost), dtype=float)
    if vertex.shape != (n_hours,) or not np.all(np.isfinite(vertex)):
        raise CouldNotRun(f"flexitroid's greedy algorithm returned {vertex!r}")
    return vertex[1 : n_service + 1].copy()


def shape_rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"flexitroid-shapes {label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def shape_widths(rng: np.random.Generator, ranges: np.ndarray, index: int):
    if index == 0:
        return ranges.copy(), ranges.copy()
    widths = []
    for _ in range(2):
        draw = rng.uniform(0.0, 1.0, size=ranges.size)
        draw[rng.uniform(0.0, 1.0, size=ranges.size) < ZERO_WIDTH_SHARE] = 0.0
        widths.append(ranges * draw)
    return widths[0], widths[1]


def corner_directions(w_up: np.ndarray, w_down: np.ndarray) -> np.ndarray:
    """One row per distinct non-zero corner direction: `w_up[t]` or `-w_down[t]` per hour."""
    n = w_up.size
    signs = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    rows = np.where(signs == 1, w_up[None, :], -w_down[None, :])
    rows = np.unique(rows, axis=0)
    return rows[np.any(rows != 0.0, axis=1)]


@dataclass
class _Pool:
    """The vertices of flexitroid's set found so far, on the service hours."""

    vertices: list[np.ndarray]

    def add(self, vertex: np.ndarray) -> bool:
        if any(np.array_equal(vertex, known) for known in self.vertices):
            return False
        self.vertices.append(vertex)
        return True


def optimised_scale(aggregate, day: Day, pool: _Pool, centre, directions):
    """Bounds `(lower, upper)` on the largest scale, and the programmes solved.

    Raises `CouldNotRun` when a programme does not end optimal or a corner does not close.
    """
    n = day.n_service
    smallest_upper = math.inf
    lowers = []
    rounds = 0
    for w in directions:
        for _ in range(MAX_ROUNDS):
            # Variables (d_1..d_n, a): minimise a - c . d, a >= V . d for every vertex V,
            # w . d >= 1.
            vertices = np.array(pool.vertices)
            rows = np.vstack(
                [np.hstack([vertices, -np.ones((len(vertices), 1))]), np.append(-w, 0.0)]
            )
            rhs = np.append(np.zeros(len(vertices)), -1.0)
            cost = np.append(-centre, 1.0)
            solved = solve_lp(
                cost,
                A_ub=rows,
                b_ub=rhs,
                bounds=[(None, None)] * (n + 1),
                time_limit=OTHER_SECONDS,
                method="simplex",
            )
            rounds += 1
            if solved.outcome != "optimal":
                raise CouldNotRun(
                    f"a cutting-plane programme ended '{solved.outcome}' ({solved.message})"
                )
            lower = float(solved.objective)
            d = solved.x[:n]
            if lower >= smallest_upper:
                lowers.append(lower)
                break
            if w @ d < 1.0 - FEASIBLE_TOLERANCE:
                raise CouldNotRun(f"a cutting-plane programme returned w . d = {w @ d}")
            vertex = greedy_service_vertex(aggregate, day.n_hours, n, -d)
            upper = float((vertex - centre) @ d)
            smallest_upper = min(smallest_upper, upper)
            if upper - lower <= GAP_TOLERANCE * max(1.0, abs(lower)):
                lowers.append(lower)
                break
            if not pool.add(vertex):
                raise CouldNotRun(
                    f"the cutting planes stalled: gap {upper - lower:.3e} with a known vertex"
                )
        else:
            raise CouldNotRun(f"a corner did not close within {MAX_ROUNDS} rounds")
    if not lowers:
        return math.inf, math.inf, rounds
    return min(lowers), smallest_upper, rounds


def _within(found, wanted, tolerance) -> bool:
    if math.isinf(wanted) or math.isinf(found):
        return found == wanted
    return abs(found - wanted) <= tolerance * max(1.0, abs(wanted))


def _scale(functions, centre, w_up, w_down, whose: str, problems: list[str]) -> float | None:
    try:
        return exact.box_scale(functions, centre, w_up, w_down)
    except ValueError as refused:
        problems.append(f"T1(b)'s formula on {whose} functions refuses the centre: {refused}")
        return None


def compare_shape(day, aggregate, pool, ours, theirs, centre, w_up, w_down) -> ShapeResult:
    """Items 2 and 3 of the module text for one shape."""
    problems: list[str] = []
    exact_scale = _scale(ours, centre, w_up, w_down, "the project's", problems)
    formula = _scale(theirs, centre, w_up, w_down, "flexitroid's", problems)
    if None not in (exact_scale, formula) and not _within(formula, exact_scale, SCALE_TOLERANCE):
        problems.append(f"formula on flexitroid's functions {formula!r}, project {exact_scale!r}")
    directions = corner_directions(w_up, w_down)
    lower = upper = None
    rounds = 0
    could_not = None
    try:
        lower, upper, rounds = optimised_scale(aggregate, day, pool, centre, directions)
    except CouldNotRun as reason:
        could_not = str(reason)
    if exact_scale is not None and lower is not None:
        inside = _within(lower, exact_scale, SCALE_TOLERANCE) and _within(
            upper, exact_scale, SCALE_TOLERANCE
        )
        if not inside:
            problems.append(
                f"flexitroid's optimisation bounds the scale in [{lower!r}, {upper!r}], "
                f"project {exact_scale!r}"
            )
    return ShapeResult(
        centre=tuple(map(float, centre)),
        w_up=tuple(map(float, w_up)),
        w_down=tuple(map(float, w_down)),
        exact_scale=exact_scale,
        formula_scale=formula,
        lower=lower,
        upper=upper,
        corners=len(directions),
        rounds=rounds,
        problems=tuple(problems),
        could_not_run=could_not,
    )


def _functions_only(largest: float, shapes) -> str:
    """The other reading's verdict: items 1 and 2 only."""
    if largest > SET_TOLERANCE:
        return DIFFERS
    for shape in shapes:
        if shape.exact_scale is None or shape.formula_scale is None:
            return DIFFERS
        if not _within(shape.formula_scale, shape.exact_scale, SCALE_TOLERANCE):
            return DIFFERS
    return MATCH


def compare_day(day: Day, label: str, ruler: Ruler = exact.day_functions) -> DayComparison:
    """The comparison of one day (see the module text). `ruler` gives the project's functions."""
    start = time.perf_counter()
    n = day.n_service
    problems: list[str] = []
    could_not: list[str] = []
    shapes: list[ShapeResult] = []
    largest = None
    functions_only = COULD_NOT_RUN
    pool = _Pool([])
    try:
        if not day.cars:
            raise CouldNotRun("the day's fleet is empty; flexitroid has no empty aggregate")
        ours = ruler(day)
        try:
            general, closed_form = flexitroid_aggregates(day)
            theirs = flexitroid_functions(general, n)
            theirs_closed = flexitroid_functions(closed_form, n)
            for t in range(n):
                for sign in (1.0, -1.0):
                    cost = np.zeros(n)
                    cost[t] = sign
                    pool.add(greedy_service_vertex(general, day.n_hours, n, cost))
        except (AssertionError, ArithmeticError, ValueError, TypeError, IndexError) as error:
            raise CouldNotRun(f"flexitroid raised {type(error).__name__}: {error}") from error
        largest = max(
            relative_difference(found, wanted)
            for functions in (theirs, theirs_closed)
            for found, wanted in ((functions.lower, ours.lower), (functions.upper, ours.upper))
        )
        if largest > SET_TOLERANCE:
            problems.append(
                f"flexitroid's p or b differs from the project's by {largest:.3e} (relative)"
            )
        vertices = np.array(pool.vertices)
        centre = vertices.mean(axis=0)
        ranges = vertices.max(axis=0) - vertices.min(axis=0)
        ranges[ranges <= RANGE_TOLERANCE] = 0.0
        rng = shape_rng(label)
        for index in range(SHAPES):
            w_up, w_down = shape_widths(rng, ranges, index)
            shape = compare_shape(day, general, pool, ours, theirs, centre, w_up, w_down)
            shapes.append(shape)
            problems.extend(f"shape {index}: {p}" for p in shape.problems)
            if shape.could_not_run is not None:
                could_not.append(f"shape {index}: {shape.could_not_run}")
        functions_only = _functions_only(largest, shapes)
    except CouldNotRun as reason:
        could_not.append(str(reason))
    return DayComparison(
        label=label,
        cars=len(day.cars),
        n_service=n,
        n_hours=day.n_hours,
        verdict=DIFFERS if problems else COULD_NOT_RUN if could_not else MATCH,
        functions_only_verdict=functions_only,
        max_set_difference=largest,
        shapes=tuple(shapes),
        vertices=len(pool.vertices),
        problems=tuple(problems),
        could_not_run=tuple(could_not),
        seconds=time.perf_counter() - start,
    )
