"""Plan sections 2, 4 (T3) and 5 (T2(d)): the order of the values.

In every case: diagonal <= causal <= tree <= hindsight, and causal <= full <= hindsight, where
diagonal, causal and full are the certificate's structures, tree is the corner tree, and hindsight
is the ceiling, for the hourly product, the block product and a fixed shape (centre a random point
of the set, widths drawn in [0.1, 1.0]).

Two samples:

1. Made-up days: `N_DAYS` seeded days (`made_up_days.py`; 2 to 4 service hours, 1 to 4 cars), each
   on its true sets one-way and two-way (budget cycling through `BUDGETS`); hindsight is the exact
   ceiling of `exact` (the hourly ceiling b(S) - p(S), the block ceiling, T1(b)'s scale of the
   fixed shape); the fixed centre comes from `fleet_lps.random_point`. At least
   `MIN_TREE_BELOW_DAYS` cases have the tree strictly below hindsight.
2. Toy fleets: `N_TOY` seeded fleets of `test_t4_conjecture._fleet` (no hour 0; see
   `test_toy_agreement.py` for why the rows describe the same set), hindsight from `toy_values`
   (`hindsight_scale`, `block_hindsight_scale`, `hindsight_value`). At least
   `MIN_CAUSAL_BELOW_FULL` cases have the causal rule strictly below the full one, so the second
   chain is not all equalities.
"""

import functools
import random

import numpy as np
from fleet_lps import random_point
from made_up_days import made_up_sample
from test_free_box_value import _point_in_fleet_set
from test_t4_conjecture import _fleet
from toy_values import block_hindsight_scale, hindsight_scale, hindsight_value

from fleetreserve import exact, products
from fleetreserve.carsets import gpoly_fleet_rows, true_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.tree import corner_tree

SEED = 41
N_DAYS = 40
BUDGETS = (0.0, 0.5, 2.0, 5.0)
N_DAY_CASES = 2 * N_DAYS * 3
N_TOY = 120
N_TOY_CASES = 3 * N_TOY
MIN_TREE_BELOW_DAYS = 40
MIN_CAUSAL_BELOW_FULL = 5
TOLERANCE = 1e-6  # relative to max(1, value)
CHAIN = [
    ("diagonal", "causal"),
    ("causal", "tree"),
    ("tree", "hindsight"),
    ("causal", "full"),
    ("full", "hindsight"),
]


def _values(rows, product, hindsight):
    results = {
        s: certify(rows, product, s, shape_rule=ANY_OPTIMAL) for s in ("diagonal", "causal", "full")
    }
    results["tree"] = corner_tree(rows, product)
    assert all(r.outcome == "optimal" for r in results.values()), results
    values = {s: r.value for s, r in results.items()}
    values["hindsight"] = hindsight
    return values


def _exact_hindsight(functions, product):
    if product.kind == "hourly":
        return exact.hourly_ceiling(functions)
    if product.kind == "block":
        result = exact.block_ceiling(functions)
        assert result.outcome == "optimal"
        return result.value
    widths = np.array(product.widths)
    return 2.0 * exact.box_scale(functions, product.centre, widths, widths) * widths.sum()


@functools.cache
def _day_cases():
    rng = random.Random(SEED)
    days, _ = made_up_sample(rng, N_DAYS)
    cases = []
    for k, day in enumerate(days):
        for budget in (None, BUDGETS[k % len(BUDGETS)]):
            cars = day.one_way_cars() if budget is None else day.two_way_cars(budget)
            functions = exact.day_functions(day, budget)
            rows = true_rows(day, budget)
            centre = random_point(rng, cars, day.n_service)
            widths = [rng.uniform(0.1, 1.0) for _ in range(day.n_service)]
            for product in (
                products.hourly(),
                products.block(),
                products.fixed_shape(centre, widths),
            ):
                values = _values(rows, product, _exact_hindsight(functions, product))
                cases.append(((day, budget), product.kind, values))
    return cases


@functools.cache
def _toy_cases():
    rng = random.Random(SEED)
    cases = []
    for k in range(N_TOY):
        n = 2 + k % 3
        cars = _fleet(rng, n, two_way=k % 2 == 1, in_window=(k // 2) % 2 == 0)
        service = list(range(1, n + 1))
        centre = _point_in_fleet_set(rng, cars, service)
        widths = {t: rng.uniform(0.1, 1.0) for t in service}
        rows = gpoly_fleet_rows(cars, n)
        fixed = products.fixed_shape([centre[t] for t in service], [widths[t] for t in service])
        for product, hindsight in (
            (fixed, 2 * hindsight_scale(cars, service, centre, widths) * sum(widths.values())),
            (products.block(), 2 * n * block_hindsight_scale(cars, service)),
            (products.hourly(), hindsight_value(cars, service)),
        ):
            cases.append((cars, product.kind, _values(rows, product, hindsight)))
    return cases


def _le(a, b):
    return a <= b + TOLERANCE * max(1.0, abs(b))


def _strict(a, b):
    return a < b - TOLERANCE * max(1.0, abs(b))


def _broken(cases):
    return [
        (low, high, kind, v, case)
        for case, kind, v in cases
        for low, high in CHAIN
        if not _le(v[low], v[high])
    ]


def test_values_are_ordered_on_made_up_days():
    cases = _day_cases()
    assert len(cases) == N_DAY_CASES
    assert sum(budget is not None for (_, budget), *_ in cases) == N_DAY_CASES // 2
    broken = _broken(cases)
    assert not broken, broken
    tree_below = sum(_strict(v["tree"], v["hindsight"]) for *_, v in cases)
    print(f"made-up days: {len(cases)} cases ordered; tree below hindsight on {tree_below}")
    assert tree_below >= MIN_TREE_BELOW_DAYS


def test_values_are_ordered_on_toy_fleets():
    cases = _toy_cases()
    assert len(cases) == N_TOY_CASES
    broken = _broken(cases)
    assert not broken, broken
    causal_below_full = sum(_strict(v["causal"], v["full"]) for *_, v in cases)
    print(f"toy fleets: {len(cases)} cases ordered; causal below full on {causal_below_full}")
    assert causal_below_full >= MIN_CAUSAL_BELOW_FULL
