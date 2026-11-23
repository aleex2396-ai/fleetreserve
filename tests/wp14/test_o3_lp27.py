"""LP (27)'s range over its optimal solutions (`o3_lp27`) and the order of the cars (`o3_day`).

1. A one-car day worked by hand. rho 0.2, two service hours, one car: departure hour 3, rate
   r = 2 kW, room e = 3.519 kWh, so the need is L = 0.8 e = 2.8152 kWh and there is no recovery
   hour. On hours 0 to 2 the car's set is U = {0 <= u_t <= 2, L <= u_0 + u_1 + u_2 <= e} (each
   stored-energy row is implied). The day-average base set of one car is U0 = {0 <= v_t <= 2,
   v_0 + v_1 + v_2 <= e}, every hour a base-set hour; its vertices are 0, 2 e_j and
   2 e_j + (e - 2) e_k (j != k).

   Write lam = L - 2 = 0.8152, mu = 1 - lam / 2 = 0.5924, delta = 2 - e / 2 = 0.2405.

   Ceiling (hour 0 free): b(S) - p(S) = e - lam = 2 + 0.2 e = 2.7038 kWh.

   LP (27): maximise the trace of Gamma subject to gamma + Gamma v in U for every v in U0. Hour j's
   row at v = 2 e_j gives gamma_j + 2 Gamma_jj <= 2, and the need at v = 0 gives
   gamma_0 + gamma_1 + gamma_2 >= L; adding, 2 trace <= 6 - L, so the trace is at most
   3 - L / 2 = 1.5924, and every solution below attains it (gamma_j + 2 Gamma_jj = 2 for each j
   and sum gamma = L), so it is the optimum.

   Solution H: gamma = (lam, 0, 2), rows of Gamma (mu, -lam/2, 0), (0, 1, 0), (-mu, -delta, 0).
   Hour 0 runs from lam - lam = 0 (v_1 = 2) to lam + 2 mu = 2 (v_0 = 2); hour 1 is v_1 in [0, 2];
   hour 2 is 2 - mu v_0 - delta v_1 >= 2 - 2 mu - delta (e - 2) = 0.450 > 0 and <= 2; the column
   sums are (0, 0.1 e, 0), so the total is L + 0.1 e v_1, within [L, e]. Its set holds the box
   hour 1 in [0, 2], hour 2 in [lam, e - 2], corners met at v = (2, 0, 0), (0.481 / mu, 0, 0),
   (0.7038 / mu, 2, 0), (0, 2, 0): value 2 + (e - 2 - lam) = 2 + 0.2 e, the ceiling. So the largest
   box over LP (27)'s optimal solutions is the ceiling, 2.7038 kWh.

   Solutions B(beta), mu <= beta <= beta_max = 1 - delta (e - 2) / 2 = 0.81734: gamma =
   (lam, 2, 0), rows of Gamma (mu, 0, -(1 - beta)), (-delta, 0, -beta), (0, 0, 1). Hour 0 is in
   [lam - 2 (1 - beta), lam + 2 mu] within [0, 2] when beta >= mu; hour 1, 2 - delta v_0 - beta v_2,
   is least at v = (e - 2, 0, 2), where it is 0 at beta_max; hour 2 is v_2; the column sums are
   (0.1 e, 0, 0). Its service hours are x = 2 - delta a - beta y, y = v_2, with a = v_0 in [0, 2].
   A box [x_lo, x_hi] x [y_lo, y_hi] needs x_hi <= 2 - beta y_hi and x_lo >= 2 - 2 delta - beta
   y_lo, so with h = y_hi - y_lo its value is at most (2 delta - beta h) + h, and h <= 2 delta /
   beta keeps the x-width at least 0: value at most 2 delta / beta, reached by x = 2 - 2 delta,
   y in [0, 2 delta / beta] (corners at a = 2 and a = 0). At beta_max: 0.481 / 0.81734 = 0.58849
   kWh, the box of the review's own LP (27) solution on this day. The solver's own vertex holds a
   box of 2.5345 kWh. So the smallest box over the optimal solutions is at most 0.58849.

   The search must therefore find the hourly largest 2.7038 (the ceiling: nothing is larger) and
   an hourly smallest of at most 0.58849; its block values lie between 0 and the block ceiling.
   The test also checks H and B(beta_max) against the car's rows at every vertex of U0.
2. The order of the cars. On `N_DAYS` seeded made-up days with two or more cars, every value of
   `o3_day.compute_day` is the same with the cars listed as made, reversed and shuffled (a seeded
   shuffle), to `TOLERANCE`.
3. The search's time limits: every programme of LP (27) and over its maps gets `others`, every box
   programme `box`. With every solve given `TINY` seconds answered as the solver answers one that
   reached its time limit (`solver_timeout_fake`: a real limit that small races the clock), both
   at `TINY` leave every value missing as a timeout, never 0; `box` alone at `TINY` leaves LP
   (27)'s trace solved and every box value missing as a timeout.
4. A box programme that ends with a solver error is passed over and counted, and the values of
   the other candidates stay; when every box programme so ends, the values are missing ("error").
   A candidate whose box programme ends with a solver error for the block product only is passed
   over for the hourly product too, so both products' smallest found come from the same
   candidates and the block one stays at most the hourly one: with every block box but that of
   step 1's solution failing, the hourly smallest found is step 1's hourly box.
"""

import dataclasses
import datetime
import itertools
import random

import numpy as np
import pytest
from made_up_days import made_up_sample
from solver_timeout_fake import force_timeouts

from fleetreserve import o3_lp27, sparse_programme
from fleetreserve.base_sets import day_average
from fleetreserve.carsets import true_rows
from fleetreserve.day import Day, TrueCar, covering_recovery_hours
from fleetreserve.lp import solve_lp
from fleetreserve.o3_day import KEYS, PRODUCTS, DayTask, TimeLimits, compute_day
from fleetreserve.products import Product

E, RHO = 3.519, 0.2
NEED = (1 - RHO) * E
LAM = NEED - 2.0
MU = 1.0 - LAM / 2.0
DELTA = 2.0 - E / 2.0
BETA_MAX = 1.0 - DELTA * (E - 2.0) / 2.0
CEILING = 2.0 + 0.2 * E
TRACE = 3.0 - NEED / 2.0
LOW_BOUND = 2.0 * DELTA / BETA_MAX
TOLERANCE = 1e-6
SEED = 52
N_DAYS = 12
TINY = 1e-6


def _one_car_day() -> Day:
    cars = (TrueCar(3, 2.0, E),)
    return Day("made-up", datetime.date(2020, 1, 1), 2, covering_recovery_hours(cars, 2), RHO, cars)


def _vertices():
    found = [np.zeros(3)]
    for j in range(3):
        found.append(2.0 * np.eye(3)[j])
        for k in range(3):
            if k != j:
                found.append(2.0 * np.eye(3)[j] + (E - 2.0) * np.eye(3)[k])
    return found


SOLUTION_H = (
    np.array([LAM, 0.0, 2.0]),
    np.array([[MU, -LAM / 2.0, 0.0], [0.0, 1.0, 0.0], [-MU, -DELTA, 0.0]]),
)
SOLUTION_B = (
    np.array([LAM, 2.0, 0.0]),
    np.array([[MU, 0.0, -(1.0 - BETA_MAX)], [-DELTA, 0.0, -BETA_MAX], [0.0, 0.0, 1.0]]),
)


def _check_the_hand_numbers():
    numbers = {
        "L": (NEED, 2.8152),
        "lam": (LAM, 0.8152),
        "mu": (MU, 0.5924),
        "delta": (DELTA, 0.2405),
        "beta_max": (BETA_MAX, 0.81734025),
        "ceiling": (CEILING, 2.7038),
        "trace": (TRACE, 1.5924),
        "2 delta / beta_max": (LOW_BOUND, 0.588494),
    }
    for name, (value, by_hand) in numbers.items():
        assert value == pytest.approx(by_hand, abs=1e-6), name


@pytest.mark.parametrize("name", ["H", "B"])
def test_the_hand_solutions_are_optimal_for_lp27_and_hold_the_hand_boxes(name):
    gamma, gain = {"H": SOLUTION_H, "B": SOLUTION_B}[name]
    rows = true_rows(_one_car_day())
    worst = max(
        float(np.max(rows.coefficients @ (gamma + gain @ v) - rows.bounds)) for v in _vertices()
    )
    assert worst <= 1e-12
    assert np.trace(gain) == pytest.approx(TRACE, abs=1e-12)
    box, _ = o3_lp27.largest_box(gamma, gain, day_average(rows), Product("hourly"), 30.0)
    assert box.value == pytest.approx({"H": CEILING, "B": LOW_BOUND}[name], abs=TOLERANCE)


def test_the_one_car_day_gives_the_range_worked_by_hand():
    _check_the_hand_numbers()
    result = compute_day(DayTask("made-up", "A", "1", _one_car_day(), "day-average", TimeLimits()))
    hourly, block = result.values["hourly"], result.values["block"]
    assert result.trace_27 == pytest.approx(TRACE, abs=TOLERANCE)
    assert hourly["ceiling"] == pytest.approx(CEILING, abs=1e-12)
    assert hourly["lp27_largest_found"] == pytest.approx(CEILING, abs=TOLERANCE)
    assert 0.0 <= hourly["lp27_smallest_found"] <= LOW_BOUND + TOLERANCE
    assert (
        0.0
        <= block["lp27_smallest_found"]
        <= block["lp27_largest_found"]
        <= block["ceiling"] + TOLERANCE
    )
    print(
        f"one-car day: hourly {hourly['lp27_smallest_found']} to "
        f"{hourly['lp27_largest_found']}, block {block['lp27_smallest_found']} to "
        f"{block['lp27_largest_found']}"
    )


def _with_cars(day: Day, cars) -> Day:
    return dataclasses.replace(day, cars=tuple(cars))


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= TOLERANCE * max(1.0, abs(a))


def test_every_value_is_the_same_whatever_the_order_of_the_cars():
    days, _ = made_up_sample(random.Random(SEED), 3 * N_DAYS)
    days = [d for d in days if len(set(d.cars)) >= 2][:N_DAYS]
    assert len(days) == N_DAYS
    shuffle = random.Random(SEED)
    changed = []
    for k, day in enumerate(days):
        mixed = list(day.cars)
        shuffle.shuffle(mixed)
        found = [
            compute_day(DayTask("made-up", "A", str(k), d, "day-average", TimeLimits()))
            for d in (day, _with_cars(day, day.cars[::-1]), _with_cars(day, mixed))
        ]
        for product, key in itertools.product(PRODUCTS, KEYS):
            values = [r.values[product][key] for r in found]
            if not all(_same(values[0], v) for v in values[1:]):
                changed.append((k, product, key, values))
    assert not changed, changed


def _spy(monkeypatch):
    seen = []

    def spy(c, **kwargs):
        seen.append(kwargs["time_limit"])
        return solve_lp(c, **kwargs)

    monkeypatch.setattr(sparse_programme, "solve_lp", spy)
    monkeypatch.setattr(o3_lp27, "solve_lp", spy)
    return seen


def test_the_search_gives_its_programmes_the_two_limits(monkeypatch):
    seen = _spy(monkeypatch)
    rows = true_rows(_one_car_day())
    found = o3_lp27.search(rows, day_average(rows), PRODUCTS, 50.0, 30.0)
    assert set(found.outcomes.values()) == {"optimal"}
    candidates = 2 + o3_lp27.DRAWS + len(o3_lp27._directions(rows.n_service))
    assert set(seen) == {50.0, 30.0}
    assert seen.count(50.0) >= candidates
    assert seen.count(30.0) >= len(PRODUCTS) * candidates


def test_a_timed_out_search_is_missing_never_zero(monkeypatch):
    force_timeouts(monkeypatch, TINY)
    rows = true_rows(_one_car_day())
    found = o3_lp27.search(rows, day_average(rows), PRODUCTS, TINY, TINY)
    assert found.trace is None and found.trace_outcome == "timeout"
    assert found.low == dict.fromkeys(PRODUCTS) and found.high == dict.fromkeys(PRODUCTS)
    assert found.outcomes == dict.fromkeys(PRODUCTS, "timeout")
    boxes = o3_lp27.search(rows, day_average(rows), PRODUCTS, 300.0, TINY)
    assert boxes.trace == pytest.approx(TRACE, abs=TOLERANCE)
    assert boxes.low == dict.fromkeys(PRODUCTS) and boxes.high == dict.fromkeys(PRODUCTS)
    assert boxes.outcomes == dict.fromkeys(PRODUCTS, "timeout")


def _failing_boxes(monkeypatch, fail):
    real = o3_lp27.largest_box
    calls = []

    def box(*args, **kwargs):
        calls.append(None)
        if fail(len(calls)):
            return o3_lp27.CornerBox("error", None, None), 0.0
        return real(*args, **kwargs)

    monkeypatch.setattr(o3_lp27, "largest_box", box)


def test_a_box_programme_with_a_solver_error_is_passed_over(monkeypatch):
    rows = true_rows(_one_car_day())
    whole = o3_lp27.search(rows, day_average(rows), PRODUCTS, 300.0, 30.0)
    _failing_boxes(monkeypatch, lambda call: call == 2)
    found = o3_lp27.search(rows, day_average(rows), PRODUCTS, 300.0, 30.0)
    assert found.outcomes == dict.fromkeys(PRODUCTS, "optimal")
    assert found.skipped == whole.skipped + 1
    assert found.high["hourly"] == pytest.approx(CEILING, abs=TOLERANCE)
    assert all(found.low[p] is not None and found.high[p] is not None for p in PRODUCTS)


def test_a_candidate_whose_block_box_fails_is_passed_over_for_the_hourly_product_too(monkeypatch):
    rows = true_rows(_one_car_day())
    base = day_average(rows)
    first = o3_lp27.inner_set(rows, base, "27", time_limit=300.0)
    kept, _ = o3_lp27.largest_box(first.p, first.P, base, Product("hourly"), 30.0)
    real = o3_lp27.largest_box

    def box(p, P, base, product, time_limit):
        if product.kind == "block" and not (
            np.array_equal(p, first.p) and np.array_equal(P, first.P)
        ):
            return o3_lp27.CornerBox("error", None, None), 0.0
        return real(p, P, base, product, time_limit)

    monkeypatch.setattr(o3_lp27, "largest_box", box)
    found = o3_lp27.search(rows, base, PRODUCTS, 300.0, 30.0)
    assert found.outcomes == dict.fromkeys(PRODUCTS, "optimal")
    assert found.low["hourly"] == pytest.approx(kept.value, abs=TOLERANCE)
    assert found.low["block"] <= found.low["hourly"] + TOLERANCE, found.low


def test_every_box_programme_with_a_solver_error_leaves_the_values_missing(monkeypatch):
    rows = true_rows(_one_car_day())
    _failing_boxes(monkeypatch, lambda call: True)
    found = o3_lp27.search(rows, day_average(rows), PRODUCTS, 300.0, 30.0)
    assert found.outcomes == dict.fromkeys(PRODUCTS, "error")
    assert found.low == dict.fromkeys(PRODUCTS) and found.high == dict.fromkeys(PRODUCTS)
