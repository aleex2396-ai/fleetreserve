"""LP (27)'s bound (`o3_lp27_bound`): no optimal solution of LP (27) holds a larger box than it.

1. One service hour, every vertex. With one service hour the largest box a map holds is the range
   of `P[1, J] . v` over U0, `support(P[1]) + support(-P[1])` (each support a greedy sum: the
   largest entries first, each up to its power, until the energy is used up), a convex function
   of `P[1, J]`; so its largest value over LP (27)'s optimal face is at a vertex of the face's
   image in `P[1, J]`. With two base-set hours the image is a polygon, found by maximising over
   the face each outward edge normal of the hull of the points found so far until no edge moves
   (the hull is then the image). On the days of `ONE_HOUR` (seeded made-up days with two
   base-set hours): the search's largest found <= the true largest <= the bound before the
   ceiling is taken and its value, at the root and with 5 relaxations; on the three whose true
   largest is below the ceiling, the root's bound is within `TIGHT` of it.
2. Two or three service hours. The largest box over the face is not at a vertex there (a convex
   combination of solutions can hold more, `test_o3_lp27_largest`), so only lower witnesses
   exist: the boxes an independent search found (`test_o3_lp27_largest.WITNESSES`) are each at
   most the root's bound.
3. Branch and bound. On `SPLIT_DAY` (block) the root's bound is above `SPLIT_ROOT`; with more
   relaxations (1, 3, 5, 11, 21) the bound never rises and with 21 it is below `SPLIT_21`, each
   bound at least the largest found.
4. The incumbent. On `RAISE_DAY` (block), given its largest found `RAISE_FOUND`, the branch and
   bound meets a map of the face whose block box is the ceiling (within the closing slack
   `o3_lp27_bound.CLOSE`) and closes.
5. Failures. A map range or the root relaxation ending "timeout": the bound missing, outcome
   "timeout". Both child relaxations ending "timeout": the bound is the root's, outcome open,
   both failures counted.
6. `o3_day`: the map ranges are solved once and given to both products' bounds; the block bound
   is at most the hourly bound; an incumbent above the largest found raises it, and the hourly
   largest found is then at least the block one; LP (27) not solved: the bound is missing with
   LP (27)'s outcome and nothing of the bound is solved.
7. `o3_lp27.scaled_effort`: share 1 is the full settings; a share scales the objectives, sign
   vectors, mixtures and the starts from steps 2a to 2e, rounded up and at least 1, and keeps
   the block starts and rounds; a share outside (0, 1] raises. On `test_o3_lp27`'s one-car day
   the search at share 0.1 solves fewer programmes at each limit than at share 1, and at least
   its candidates at the others' limit.
8. The day by hand of `o3_lp27_bound`'s module text (`HAND_DAY`): the map ranges are the hand
   ranges `L = (-2, 0)`, `U = (0, 2)` widened by at least the margin and by at most 1e-5; the
   relaxation alone proves 4, the ceiling; with the ceiling and the largest found, closed.
9. The map ranges solve each car's own programmes: one LP (27) per car and two programmes per
   base-set hour and service hour in which the car's power is not fixed, each with fewer than
   half the fleet face's variables.
10. Eight service hours (`EIGHT_HOUR_DAY`, 10 cars, 256 corner paths): both products' bounds
   exist at the run's time limits, at most the ceiling and at least the box of LP (27)'s own
   solution.
11. Corners added (`LAZY_DAY`, 5 service hours, block): the two constant corners' relaxation is
   above all 32 corners' by more than `LAZY_GAP`; with up to 8 relaxations the bound reaches the
   latter's (never below it), with fewer than 32 corners kept. A later relaxation ending
   "timeout" keeps the last bound, one failure counted.
"""

import dataclasses
import datetime

import numpy as np
import pytest
from test_o3_lp27 import _one_car_day, _spy
from test_o3_lp27_largest import N_SERVICE, WITNESSES

from fleetreserve import exact, o3_day, o3_lp27, o3_lp27_bound
from fleetreserve.base_sets import day_average
from fleetreserve.carsets import true_rows
from fleetreserve.day import Day, TrueCar, covering_recovery_hours
from fleetreserve.lp24 import inner_set
from fleetreserve.o3_day import BOUND, KEYS, PRODUCTS, DayTask, TimeLimits, car_order
from fleetreserve.o3_lp27_bound import CEILING, CLOSED, OPEN, LargestBound
from fleetreserve.products import Product

OTHERS, BOX = 300.0, 30.0
TOLERANCE = 1e-6  # relative to max(1, ceiling)
TIGHT = 1e-4  # kWh
# (cars as (departure, rate kW, room kWh), rho), one service hour
ONE_HOUR = {
    "below-ceiling-1": (((3, 6.6, 2.096), (3, 2.0, 5.817)), 0.0),
    "below-ceiling-2": (((1, 1.0, 0.345), (3, 2.0, 2.672)), 0.1),
    "below-ceiling-3": (((1, 2.0, 0.482), (3, 2.0, 2.233)), 0.0),
    "four-vertices-1": (((2, 3.3, 4.394), (3, 6.6, 14.206)), 0.1),
    "four-vertices-2": (((2, 6.6, 12.437),), 0.2),
}
SPLIT_DAY = (((4, 1.0, 2.721), (2, 2.0, 3.687), (4, 2.0, 2.901)), 0.1, 2)
SPLIT_ROOT, SPLIT_21 = 4.4, 4.2
RAISE_DAY = (((3, 1.0, 0.186), (3, 2.0, 3.316), (5, 3.3, 9.119)), 0.0, 2)
RAISE_FOUND = 8.2549
# The day worked by hand in `o3_lp27_bound`'s module text: one service hour, cars A and B leave
# after hour 1, rate 3 kW, rooms 1 and 3 kWh, rho 0.
HAND_DAY = (((2, 3.0, 1.0), (2, 3.0, 3.0)), 0.0, 1)
HAND_LOW, HAND_HIGH = np.array([[-2.0, 0.0]]), np.array([[0.0, 2.0]])
HAND_CEILING = 4.0
# Seeded made-up days: 10 cars over 8 service hours (256 corner paths), and 5 cars over 5 service
# hours where the block relaxation with the two constant corners is above the one with all 32.
EIGHT_HOUR_DAY = (
    (
        (4, 6.6, 4.988),
        (6, 1.0, 2.381),
        (6, 1.0, 3.184),
        (3, 3.3, 4.91),
        (7, 6.6, 35.304),
        (10, 2.0, 16.002),
        (1, 1.0, 0.95),
        (3, 6.6, 9.118),
        (10, 3.3, 24.218),
        (4, 6.6, 15.213),
    ),
    0.1,
    8,
)
LAZY_DAY = (
    ((4, 1.0, 2.365), (7, 6.6, 17.046), (5, 1.0, 0.988), (6, 2.0, 11.621), (4, 6.6, 6.568)),
    0.2,
    5,
)
LAZY_GAP = 0.02  # kWh: the two corners' relaxation is above every corner's by more


def _setting(cars, rho, n_service):
    cars = tuple(TrueCar(*car) for car in cars)
    recovery = covering_recovery_hours(cars, n_service)
    day = car_order(Day("made-up", datetime.date(2020, 1, 1), n_service, recovery, rho, cars))
    rows = true_rows(day)
    base = day_average(rows)
    functions = exact.day_functions(day)
    ceilings = {
        "hourly": float(exact.hourly_ceiling(functions)),
        "block": float(exact.block_ceiling(functions, time_limit=BOX).value),
    }
    return rows, base, float(inner_set(rows, base, "27").value), ceilings, functions


def _support(a, power, energy):
    """The largest `a . v` over U0 (greedy)."""
    total, left = 0.0, energy
    for b in np.argsort(-a):
        if a[b] <= 0 or left <= 0:
            break
        take = min(power[b], left)
        total, left = total + a[b] * take, left - take
    return total


def _farthest(rows, base, trace, direction):
    lp, _, _, _, map_s = o3_lp27._face(rows, base, trace)
    result, _ = lp.maximise(map_s[0], np.asarray(direction, float), time_limit=OTHERS)
    assert result.outcome == "optimal"
    return result.x[map_s[0]]


def _hull(points):
    """The hull's corners in order (monotone chain)."""
    points = sorted({tuple(np.round(p, 10)) for p in points})
    if len(points) <= 2:
        return [np.array(p) for p in points]

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    chains = []
    for sweep in (points, points[::-1]):
        chain = []
        for p in sweep:
            while len(chain) >= 2 and turn(chain[-2], chain[-1], p) <= 1e-12:
                chain.pop()
            chain.append(p)
        chains.append(chain[:-1])
    return [np.array(p) for p in chains[0] + chains[1]]


def _vertices(rows, base, trace):
    """Every vertex of the face's image in `P[1, J]`, two base-set hours (module text, item 1)."""
    starts = ([1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1])
    points = [_farthest(rows, base, trace, d) for d in starts]
    for _ in range(50):
        hull = _hull(points)
        if len(hull) < 3:
            return hull
        moved = False
        for a, b in zip(hull, hull[1:] + hull[:1], strict=True):
            normal = np.array([b[1] - a[1], a[0] - b[0]])
            normal /= np.linalg.norm(normal)
            x = _farthest(rows, base, trace, normal)
            if normal @ x > normal @ a + 1e-9:
                points.append(x)
                moved = True
        if not moved:
            return hull
    raise AssertionError("the polygon did not settle")


@pytest.mark.parametrize("name", ONE_HOUR)
def test_one_service_hour_found_at_most_every_vertex_at_most_the_bound(name):
    cars, rho = ONE_HOUR[name]
    rows, base, trace, ceilings, functions = _setting(cars, rho, 1)
    assert base.columns.size == 2
    power = base.power[base.columns]
    vertices = _vertices(rows, base, trace)
    true = max(_support(p, power, base.energy) + _support(-p, power, base.energy) for p in vertices)
    ceiling = ceilings["hourly"]
    slack = TOLERANCE * max(1.0, ceiling)
    assert true <= ceiling + slack
    found = o3_lp27.search(rows, base, PRODUCTS, OTHERS, BOX).high["hourly"]
    assert found <= true + slack
    for nodes in (1, 5):
        bound = o3_lp27_bound.largest_bound(
            rows, base, trace, Product("hourly"), functions, ceiling, found, nodes, OTHERS, BOX
        )
        assert bound.outcome == CLOSED, bound
        assert true <= bound.proved + slack and true <= bound.value + slack, (true, bound)
        if name.startswith("below-ceiling"):
            assert true < ceiling - 0.01
            assert bound.value <= true + TIGHT, (true, bound)


@pytest.mark.parametrize("name", WITNESSES)
def test_every_independent_witness_is_at_most_the_root_bound(name):
    rho, cars, product, value, ceiling = WITNESSES[name]
    rows, base, trace, ceilings, functions = _setting(cars, rho, N_SERVICE)
    assert ceilings[product] == pytest.approx(ceiling, abs=1e-3)
    bound = o3_lp27_bound.largest_bound(
        rows, base, trace, Product(product), functions, ceilings[product], None, 1, OTHERS, BOX
    )
    assert bound.nodes == 1 and bound.proved is not None
    assert value <= bound.proved and value <= bound.value, bound


def test_more_relaxations_never_raise_the_bound():
    cars, rho, n_service = SPLIT_DAY
    rows, base, trace, ceilings, functions = _setting(cars, rho, n_service)
    found = o3_lp27.search(rows, base, PRODUCTS, OTHERS, BOX).high["block"]
    values = []
    for nodes in (1, 3, 5, 11, 21):
        bound = o3_lp27_bound.largest_bound(
            rows,
            base,
            trace,
            Product("block"),
            functions,
            ceilings["block"],
            found,
            nodes,
            OTHERS,
            BOX,
        )
        assert bound.nodes <= nodes
        assert found <= bound.value
        values.append(bound.value)
    assert values[0] > SPLIT_ROOT and values[-1] < SPLIT_21, values
    assert all(b <= a + TOLERANCE for a, b in zip(values, values[1:], strict=False)), values


def test_the_incumbent_reaches_a_box_above_the_largest_found():
    cars, rho, n_service = RAISE_DAY
    rows, base, trace, ceilings, functions = _setting(cars, rho, n_service)
    ceiling = ceilings["block"]
    bound = o3_lp27_bound.largest_bound(
        rows, base, trace, Product("block"), functions, ceiling, RAISE_FOUND, 51, OTHERS, BOX
    )
    assert bound.incumbent > RAISE_FOUND + 0.5, bound
    assert bound.incumbent == pytest.approx(ceiling, abs=o3_lp27_bound.CLOSE * ceiling)
    assert bound.outcome == CLOSED and bound.value == pytest.approx(ceiling)


def _timeout_from(call):
    """A stand-in for `o3_lp27_bound.relaxation` that ends "timeout" from its `call`-th call."""
    real, calls = o3_lp27_bound.relaxation, []

    def relaxation(*args):
        calls.append(1)
        if len(calls) >= call:
            return "timeout", None, None, 0.0
        return real(*args)

    return relaxation


def test_a_root_programme_not_solved_leaves_the_bound_missing(monkeypatch):
    cars, rho, n_service = SPLIT_DAY
    rows, base, trace, ceilings, functions = _setting(cars, rho, n_service)
    block = Product("block")
    arguments = (rows, base, trace, block, functions, ceilings["block"], 1.0, 5, OTHERS, BOX)
    with monkeypatch.context() as patch:
        patch.setattr(o3_lp27_bound, "map_ranges", lambda *a: ("timeout", None, None, 0.0))
        bound = o3_lp27_bound.largest_bound(*arguments)
    assert (bound.value, bound.outcome, bound.proved, bound.failed) == (None, "timeout", None, 1)
    monkeypatch.setattr(o3_lp27_bound, "relaxation", _timeout_from(1))
    bound = o3_lp27_bound.largest_bound(*arguments)
    assert (bound.value, bound.outcome, bound.proved, bound.failed) == (None, "timeout", None, 1)


def test_a_child_relaxation_not_solved_keeps_its_parents_bound(monkeypatch):
    cars, rho, n_service = SPLIT_DAY
    rows, base, trace, ceilings, functions = _setting(cars, rho, n_service)
    ceiling = ceilings["block"]
    arguments = (rows, base, trace, Product("block"), functions, ceiling, 1.0, 1, OTHERS, BOX)
    root = o3_lp27_bound.largest_bound(*arguments)
    monkeypatch.setattr(o3_lp27_bound, "relaxation", _timeout_from(2))
    bound = o3_lp27_bound.largest_bound(*arguments[:-3], 3, OTHERS, BOX)
    assert bound.nodes == 3 and bound.failed >= 2
    assert bound.outcome == OPEN
    assert bound.value == pytest.approx(root.value, abs=1e-9)
    assert root.value < ceiling - 0.1


def _task():
    return DayTask(
        "u", "A", "2020-01-01", None, "day-average", TimeLimits(), bound_nodes=3, bound_rounds=2
    )


def _blank(found_hourly, found_block):
    values = {p: dict.fromkeys(KEYS) for p in PRODUCTS}
    outcomes = {p: dict.fromkeys(KEYS, "optimal") for p in PRODUCTS}
    values["hourly"].update(ceiling=6.0, lp27_largest_found=found_hourly)
    values["block"].update(ceiling=5.0, lp27_largest_found=found_block)
    return values, outcomes


def test_the_day_takes_the_block_bound_at_most_the_hourly_and_raises_the_found(monkeypatch):
    answers = {"hourly": (4.0, 3.5, CEILING), "block": (4.5, 3.8, OPEN)}
    calls = []

    ranges = ("optimal", "low", "high", 0.5)

    def fake(
        rows, base, trace, product, functions, ceiling, incumbent, nodes, others, box, given, rounds
    ):
        calls.append((product.kind, functions, ceiling, incumbent, nodes, given, rounds))
        value, incumbent, outcome = answers[product.kind]
        return LargestBound(value, outcome, incumbent, nodes, 0, value, 0.0)

    monkeypatch.setattr(o3_day, "largest_bound", fake)
    monkeypatch.setattr(o3_day, "map_ranges", lambda *a: ranges)
    values, outcomes = _blank(3.0, 2.0)
    seconds = {}
    o3_day._bounds(None, None, "functions", _task(), 5.0, values, outcomes, seconds)
    assert calls == [
        ("hourly", "functions", 6.0, 3.0, 3, ranges, 2),
        ("block", "functions", 5.0, 2.0, 3, ranges, 2),
    ]
    assert values["hourly"][BOUND] == 4.0 and outcomes["hourly"][BOUND] == CEILING
    assert values["block"][BOUND] == 4.0 and outcomes["block"][BOUND] == OPEN
    assert values["block"]["lp27_largest_found"] == 3.8
    assert values["hourly"]["lp27_largest_found"] == 3.8  # a block box is an hourly box
    assert seconds == {"lp27_bound_ranges": 0.5, "lp27_bound_hourly": 0.0, "lp27_bound_block": 0.0}


def test_the_day_leaves_the_bound_missing_when_lp27_is_not_solved(monkeypatch):
    monkeypatch.setattr(o3_day, "largest_bound", None)  # a call would raise
    monkeypatch.setattr(o3_day, "map_ranges", None)
    values, outcomes = _blank(None, None)
    for product in PRODUCTS:
        outcomes[product]["lp27_largest_found"] = "timeout"
    o3_day._bounds(None, None, None, _task(), None, values, outcomes, {})
    for product in PRODUCTS:
        assert values[product][BOUND] is None and outcomes[product][BOUND] == "timeout"


def test_the_effort_share_scales_the_search_settings():
    full = o3_lp27.full_effort()
    assert o3_lp27.scaled_effort(1.0) == full
    quarter = o3_lp27.scaled_effort(0.25)
    for name in ("draws", "range_draws", "mixtures", "ascent_starts", "mixture_starts"):
        assert getattr(quarter, name) == max(1, -(-getattr(full, name) // 4)), name
    assert (quarter.other_starts, quarter.rounds) == (full.other_starts, full.rounds)
    tiny = dataclasses.asdict(o3_lp27.scaled_effort(1e-6))
    assert tiny == dataclasses.asdict(full) | {
        "draws": 1,
        "range_draws": 1,
        "mixtures": 1,
        "ascent_starts": 1,
        "mixture_starts": 1,
    }
    for share in (0.0, -0.5, 1.5, float("nan")):
        with pytest.raises(ValueError):
            o3_lp27.scaled_effort(share)


def test_a_smaller_effort_solves_fewer_programmes(monkeypatch):
    rows = true_rows(_one_car_day())
    counts = {}
    for share in (0.1, 1.0):
        seen = _spy(monkeypatch)
        effort = o3_lp27.scaled_effort(share)
        found = o3_lp27.search(rows, day_average(rows), PRODUCTS, 50.0, 30.0, effort)
        assert set(found.outcomes.values()) == {"optimal"}
        candidates = 2 + effort.draws + len(o3_lp27._directions(2, effort.range_draws))
        assert seen.count(50.0) >= candidates
        counts[share] = (seen.count(50.0), seen.count(30.0))
    assert counts[0.1][0] < counts[1.0][0] and counts[0.1][1] < counts[1.0][1], counts


def test_the_hand_day_its_map_ranges_and_bound():
    """The module text's day by hand: car `i`'s face has `Gamma_i[1] = (c - e_i / 2, c)`, `c` in
    `[0, e_i / 2]`, so `L = (-2, 0)` and `U = (0, 2)`; every optimal solution holds the box 4,
    the ceiling, so the bound is 4: the relaxation alone (no ceiling, no incumbent) proves 4,
    open; with the ceiling and the largest found it is closed."""
    cars, rho, n_service = HAND_DAY
    rows, base, trace, ceilings, functions = _setting(cars, rho, n_service)
    assert np.allclose(base.power, [2.0, 2.0]) and base.energy == pytest.approx(2.0)
    assert trace == pytest.approx(2.0)  # T_A + T_B = 1 / 2 + 3 / 2
    assert ceilings == pytest.approx({"hourly": HAND_CEILING, "block": HAND_CEILING})
    outcome, low, high, _ = o3_lp27_bound.map_ranges(rows, base, OTHERS)
    assert outcome == "optimal"
    # outside the hand ranges by at least the margin, and by no more than the faces' floors
    # (BOUND_FACE on each car's trace) and the margin allow
    margin = o3_lp27_bound.MAP_MARGIN * np.maximum(1.0, np.abs(HAND_LOW))
    assert np.all(low <= HAND_LOW - margin + 1e-12) and np.all(low >= HAND_LOW - 1e-5), low
    margin = o3_lp27_bound.MAP_MARGIN * np.maximum(1.0, np.abs(HAND_HIGH))
    assert np.all(high >= HAND_HIGH + margin - 1e-12) and np.all(high <= HAND_HIGH + 1e-5), high
    found = o3_lp27.search(rows, base, PRODUCTS, OTHERS, BOX)
    for product in PRODUCTS:
        assert found.low[product] == pytest.approx(HAND_CEILING, abs=1e-6)
        alone = o3_lp27_bound.largest_bound(
            rows, base, trace, Product(product), functions, None, None, 1, OTHERS, BOX
        )
        assert alone.outcome == OPEN and alone.nodes == 1
        assert alone.proved == pytest.approx(HAND_CEILING, abs=1e-5), alone
        given = o3_lp27_bound.largest_bound(
            rows,
            base,
            trace,
            Product(product),
            functions,
            HAND_CEILING,
            found.high[product],
            1,
            OTHERS,
            BOX,
        )
        assert given.outcome == CLOSED and given.value == pytest.approx(HAND_CEILING)


def test_the_map_ranges_solve_each_cars_own_programmes(monkeypatch):
    """Per car: its own LP (27), then two programmes per base-set hour and service hour in which
    its power is not fixed (`fixed_hours`); each with one car's variables only."""
    cars, rho, n_service = SPLIT_DAY
    rows, base, trace, _, _ = _setting(cars, rho, n_service)
    sizes, solve = [], o3_lp27_bound._solve

    def spy(lp, objective, time_limit):
        sizes.append(lp.n_variables)
        return solve(lp, objective, time_limit)

    monkeypatch.setattr(o3_lp27_bound, "_solve", spy)
    outcome, _, _, _ = o3_lp27_bound.map_ranges(rows, base, OTHERS)
    assert outcome == "optimal"
    q = base.columns.size
    free = [
        int((~o3_lp27_bound.fixed_hours(o3_lp27_bound._car(rows, i))[1:]).sum())
        for i in range(rows.n_cars)
    ]
    assert free == [1, 2, 2]  # departures 2, 4, 4 over two service hours
    assert len(sizes) == rows.n_cars + 2 * q * sum(free)
    fleet = o3_lp27._face(rows, base, trace)[0].n_variables
    assert max(sizes) < fleet / 2, (max(sizes), fleet)


def test_the_bound_exists_on_a_day_of_eight_service_hours():
    """10 cars, 8 service hours, the run's time limits, the root's first relaxation (the two
    constant corners of 256): both products' bounds are proved (not missing), at most the
    ceiling and at least the box of LP (27)'s own solution."""
    cars, rho, n_service = EIGHT_HOUR_DAY
    rows, base, trace, ceilings, functions = _setting(cars, rho, n_service)
    ranges = o3_lp27_bound.map_ranges(rows, base, OTHERS)
    assert ranges[0] == "optimal"
    inner = inner_set(rows, base, "27")
    for product in PRODUCTS:
        kind, ceiling = Product(product), ceilings[product]
        held, _ = o3_lp27.largest_box(inner.p, inner.P, base, kind, BOX)
        assert held.outcome == "optimal"
        bound = o3_lp27_bound.largest_bound(
            rows, base, trace, kind, functions, ceiling, None, 1, OTHERS, BOX, ranges, 1
        )
        assert bound.value is not None and bound.proved is not None, bound
        assert bound.outcome in (CEILING, OPEN) and bound.nodes == 1, bound
        assert (bound.relaxations, bound.corners, bound.failed) == (1, 2, 0), bound
        assert held.value <= bound.value + TOLERANCE * max(1.0, ceiling), (held, bound)
        assert bound.value <= ceiling


def test_a_later_round_not_solved_keeps_the_last_bound(monkeypatch):
    """On `LAZY_DAY` (block), the second relaxation of the root ending "timeout": the bound is
    the first's, one failure counted, outcome not missing."""
    cars, rho, n_service = LAZY_DAY
    rows, base, trace, ceilings, functions = _setting(cars, rho, n_service)
    ceiling = ceilings["block"]
    arguments = (rows, base, trace, Product("block"), functions, ceiling, None, 1, OTHERS, BOX)
    first = o3_lp27_bound.largest_bound(*arguments, None, 1)
    monkeypatch.setattr(o3_lp27_bound, "relaxation", _timeout_from(2))
    bound = o3_lp27_bound.largest_bound(*arguments, None, 8)
    assert (bound.relaxations, bound.failed, bound.corners) == (2, 1, 2), bound
    assert bound.outcome == OPEN and bound.value == pytest.approx(first.value, abs=1e-9)


def test_corners_added_take_the_bound_to_every_corners_relaxation():
    """On `LAZY_DAY` (block): the root relaxation with the two constant corners is above the one
    with all 32 by more than `LAZY_GAP`; one relaxation per node gives the former; with up to 8,
    corners are added until none is missed and the bound is the latter's (never below it)."""
    cars, rho, n_service = LAZY_DAY
    rows, base, trace, ceilings, functions = _setting(cars, rho, n_service)
    block, ceiling = Product("block"), ceilings["block"]
    ranges = o3_lp27_bound.map_ranges(rows, base, OTHERS)
    every = 2**n_service
    two, all_ = (
        o3_lp27_bound.relaxation(
            rows, base, trace, block, functions, ranges[1], ranges[2], corners, OTHERS
        )
        for corners in ((0, every - 1), tuple(range(every)))
    )
    assert two[0] == all_[0] == "optimal"
    assert two[1] > all_[1] + LAZY_GAP and all_[1] < ceiling - 1.0, (two[1], all_[1], ceiling)
    margin = o3_lp27_bound.MARGIN * max(1.0, ceiling)
    rounds = {}
    for count in (1, 8):
        rounds[count] = o3_lp27_bound.largest_bound(
            rows, base, trace, block, functions, ceiling, None, 1, OTHERS, BOX, ranges, count
        )
    assert rounds[1].proved == pytest.approx(two[1] + margin, abs=1e-6), rounds[1]
    assert rounds[1].corners == 2 and rounds[1].relaxations == 1
    lazy = rounds[8]
    assert 2 < lazy.corners < every and 1 < lazy.relaxations <= 8, lazy
    assert all_[1] + margin - 1e-6 <= lazy.proved <= all_[1] + margin + 1e-5, (lazy, all_[1])
