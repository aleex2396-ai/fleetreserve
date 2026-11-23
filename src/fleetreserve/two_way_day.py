"""The two-way corner tests of one development day (plan sections 9 and 13 WP5: "two-way corner
tests match the LP"), on the day's true two-way sets at budget `BUDGET` kWh.

Reading of "match the LP" (marked; the plan does not spell it out): the corner tree's linear
programme (`tree.corner_tree`) gives a box and node powers. The tests match it when

1. replay: on every one of the 2^n corner paths, the tree's node powers along the path meet every
   row of every car's true two-way set (`carsets.true_rows` at the budget) within `ROW_TOLERANCE`
   and sum to the request `m + w zeta` in every service hour within `REQUEST_TOLERANCE`, so the
   tree's value is met by a causal rule; and
2. the box is the largest of its shape: the same tree programme with the box's shape fixed
   (`products.fixed_shape(m, w)`, the scale free) gives a scale of at least
   `1 - SCALE_TOLERANCE` and below `1 + ENLARGE`, so the box enlarged by the factor
   `1 + ENLARGE` about its centre is not met. A box with no width (value 0) is enlarged by
   `ENLARGE` kW in every hour instead (`fixed_shape(m, 1)`), and its scale must be below `ENLARGE`.

Both products are tested (hourly and block). Beside them the order of plan section 2 is checked
(diagonal <= causal affine <= tree <= the exact two-way ceiling, `exact.day_functions` at the
budget), each from its own programme: `certificate.certify` with diagonal and causal gains
(`ANY_OPTIMAL`), `tree.corner_tree`, and `exact.hourly_ceiling` or `exact.block_ceiling`.

Tolerances (set before any run on real tables): `ROW_TOLERANCE` 1e-6, `REQUEST_TOLERANCE` 1e-4
kWh (the plan's failure threshold, section 1), `SCALE_TOLERANCE` 1e-6, `ENLARGE` 1e-3.

A day with an empty fleet has every value 0 and nothing to replay ("empty fleet"). A solve that
does not end optimal leaves its value missing (None, outcome kept), never zero.
"""

from dataclasses import dataclass, field

import numpy as np

from fleetreserve import exact, products
from fleetreserve.carsets import FleetRows, true_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.day import Day
from fleetreserve.products import corner_paths
from fleetreserve.tree import TreePolicy, corner_tree, path_nodes

BUDGET = 5.0  # kWh, plan section 9 (B = 5 is never cut, plan section 13)
ROW_TOLERANCE = 1e-6
REQUEST_TOLERANCE = 1e-4
SCALE_TOLERANCE = 1e-6
ENLARGE = 1e-3
ZERO_WIDTH = 1e-9  # kW: a box with no width above this has no width

PRODUCTS = ("hourly", "block")
VALUE_NAMES = ("diag", "causal", "tree", "ceiling")
EMPTY_FLEET = "empty fleet"


@dataclass(frozen=True)
class ProductTest:
    values: dict[str, float | None]
    outcomes: dict[str, str]
    replay_paths: int
    replay_failures: int
    least_slack: float | None
    largest_request_miss: float | None
    enlarged_scale: float | None
    enlarged_outcome: str
    matched: bool | None  # None when the tree or the enlarged programme is missing


@dataclass(frozen=True)
class TwoWayResult:
    unit: str
    site: str
    date: str
    cars: int
    error: str | None
    tests: dict[str, ProductTest]
    seconds: dict[str, float] = field(default_factory=dict, compare=False)


def replay(rows: FleetRows, policy: TreePolicy) -> tuple[int, int, float, float]:
    """(paths, paths failing, least row slack, largest request miss) of the tree's node powers
    on every corner path."""
    paths = corner_paths(rows.n_service)
    nodes = np.stack([path_nodes(path) for path in paths])  # (paths, n + 1)
    powers = policy.node_powers[:, nodes]  # (cars, paths, n + 1)
    applied = np.einsum("kt,kpt->pk", rows.coefficients, powers[rows.car])
    slack = rows.bounds[None, :] - applied
    requests = policy.m[None, :] + policy.w[None, :] * paths
    miss = np.abs(powers[:, :, 1:].sum(axis=0) - requests).max(axis=1)
    failing = (slack.min(axis=1) < -ROW_TOLERANCE) | (miss > REQUEST_TOLERANCE)
    return int(paths.shape[0]), int(failing.sum()), float(slack.min()), float(miss.max())


def enlarged_scale(rows: FleetRows, policy: TreePolicy) -> tuple[float | None, str, bool, float]:
    """(scale, outcome, whether the box has width, seconds) of the tree with the box's shape."""
    has_width = bool(policy.w.max() > ZERO_WIDTH)
    widths = policy.w if has_width else np.ones_like(policy.w)
    found = corner_tree(rows, products.fixed_shape(policy.m, widths))
    return found.scale, found.outcome, has_width, float(found.seconds)


def _value(found) -> tuple[float | None, str, float]:
    value = None if found.value is None else float(found.value) + 0.0
    return value, found.outcome, float(found.seconds)


def product_test(rows, functions, name: str, seconds: dict) -> ProductTest:
    product = products.hourly() if name == "hourly" else products.block()
    values, outcomes = {}, {}
    if name == "hourly":
        ceiling = (exact.hourly_ceiling(functions) + 0.0, "exact", 0.0)
    else:
        ceiling = _value(exact.block_ceiling(functions))
    tree = corner_tree(rows, product)
    found = {
        "diag": _value(certify(rows, product, "diagonal", shape_rule=ANY_OPTIMAL)),
        "causal": _value(certify(rows, product, "causal", shape_rule=ANY_OPTIMAL)),
        "tree": _value(tree),
        "ceiling": ceiling,
    }
    for key, (value, outcome, took) in found.items():
        values[key], outcomes[key] = value, outcome
        seconds[f"{name}_{key}"] = took
    if tree.outcome != "optimal":
        return ProductTest(values, outcomes, 0, 0, None, None, None, "not solved", None)
    n_paths, failing, least, miss = replay(rows, tree.policy)
    scale, outcome, has_width, took = enlarged_scale(rows, tree.policy)
    seconds[f"{name}_enlarged"] = took
    if outcome != "optimal":
        return ProductTest(values, outcomes, n_paths, failing, least, miss, None, outcome, None)
    # With width: the box itself is met and the box enlarged by 1 + ENLARGE is not. Without:
    # the box widened by ENLARGE kW in every hour is not met.
    largest = (1.0 - SCALE_TOLERANCE <= scale < 1.0 + ENLARGE) if has_width else scale < ENLARGE
    return ProductTest(
        values, outcomes, n_paths, failing, least, miss, scale, outcome, failing == 0 and largest
    )


def two_way_day(unit: str, site: str, date: str, day: Day) -> TwoWayResult:
    if not day.cars:
        tests = {
            name: ProductTest(
                dict.fromkeys(VALUE_NAMES, 0.0),
                dict.fromkeys(VALUE_NAMES, EMPTY_FLEET),
                0,
                0,
                None,
                None,
                None,
                EMPTY_FLEET,
                True,
            )
            for name in PRODUCTS
        }
        return TwoWayResult(unit, site, date, 0, None, tests)
    try:
        functions = exact.day_functions(day, budget=BUDGET)
        rows = true_rows(day, budget=BUDGET)
    except ValueError as problem:
        return TwoWayResult(unit, site, date, len(day.cars), str(problem), {})
    seconds: dict[str, float] = {}
    tests = {name: product_test(rows, functions, name, seconds) for name in PRODUCTS}
    return TwoWayResult(unit, site, date, len(day.cars), None, tests, seconds)
