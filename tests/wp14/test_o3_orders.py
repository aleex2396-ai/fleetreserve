"""The orders between the values of outcome O3 on seeded made-up days, and the order check.

On `N_DAYS` seeded made-up days (`made_up_days.made_up_sample`: 1 to 4 cars, 2 to 4 service
hours, rho 0, 0.1 and 0.2 in turn), `o3_day.compute_day` with the day-average base set, beside
the diagonal and causal certificates (V_diag, V_aff) and the corner tree (V_caus) on the same
true sets, for the hourly and the block product. The orders that are theorems, within
`TOLERANCE` relative to `max(1, |right side|)`:

1. LP (24)'s box <= the general split <= the ceiling: LP (24)'s car maps, with the base-set
   profile affine in the request, are a general split whose decision-hour gains sum to zero (the
   fleet's decision-hour power is not requested). On ac84a00, whose "general split" had no
   decision-hour gain, LP (24) was above it on day 22 (0.248 against 0.200 kWh).
2. V_diag <= V_aff <= the general split without decision-hour gain <= the general split (plan T2
   (d), then fewer gains); on at least one day the last is strict on the hourly product (2 of
   the 40 when written).
3. Every method's value <= the ceiling and LP (27)'s smallest box found <= its largest found <=
   its bound (`o3_summary`'s check: `order_violations` finds nothing).
4. V_aff <= V_caus <= the ceiling.
5. For the ceiling and each method, block <= hourly.
6. LP (27)'s trace >= |J| LP (24)'s alpha (LP (24)'s point is feasible in LP (27); J the hours
   where the base set has power).

The orders the plan does not claim (LP (27) against LP (24) and the general split; LP (24), LP
(27) and the general split against V_caus) are counted and printed, and on the hourly product
those in `BOTH_WAYS` are above on at least one day and below on another, so neither is an order
on these days. (LP (27)'s largest box found is at least LP (24)'s and the general split's on
every one of these days, and above the general split on most: its box takes hour 0 free per
corner, as the ceiling does, while the general split holds the fleet's hour 0 at one number;
neither is a theorem.) So that the checks are not idle, at least `MIN_BELOW` days have each
method's hourly box strictly below the ceiling, and at least `MIN_BELOW` days have some block
value strictly below its hourly value.

`order_violations` itself: a result with a box above its ceiling, one with a block value above
the hourly value, and one breaking each order between methods (LP (27)'s largest found above its
bound included) are each reported.
"""

import functools
import random

import pytest
from made_up_days import made_up_sample

from fleetreserve import products
from fleetreserve.base_sets import day_average
from fleetreserve.carsets import true_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.o3_day import METHODS, PRODUCTS, DayResult, DayTask, TimeLimits, compute_day
from fleetreserve.o3_summary import order_violations
from fleetreserve.tree import corner_tree

SEED = 41
N_DAYS = 40
TOLERANCE = 1e-6
MIN_BELOW = 5
BOTH_WAYS = ("lp27_smallest_found vs lp24", "lp24 vs V_caus")


def _le(a, b):
    return a <= b + TOLERANCE * max(1.0, abs(b))


def _below(a, b):
    return a < b - TOLERANCE * max(1.0, abs(b))


@functools.cache
def _days():
    days, _ = made_up_sample(random.Random(SEED), N_DAYS)
    found = []
    for k, day in enumerate(days):
        result = compute_day(DayTask("made-up", "A", str(k), day, "day-average", TimeLimits()))
        assert result.error is None, result.error
        assert all(v is not None for p in PRODUCTS for v in result.values[p].values()), result
        rows = true_rows(day) if day.cars else None
        extra = {}
        for name in PRODUCTS:
            product = products.Product(name)
            if rows is None:
                extra[name] = {"diag": 0.0, "aff": 0.0, "caus": 0.0}
                continue
            found_values = {
                "diag": certify(rows, product, "diagonal", shape_rule=ANY_OPTIMAL),
                "aff": certify(rows, product, "causal", shape_rule=ANY_OPTIMAL),
                "caus": corner_tree(rows, product),
            }
            assert all(v.outcome == "optimal" for v in found_values.values())
            extra[name] = {k: v.value for k, v in found_values.items()}
        columns = day_average(rows).columns.size if rows is not None else 0
        found.append((result, extra, columns))
    return found


def test_lp24_at_most_general_split_at_most_the_ceiling():
    broken = []
    for k, (result, _, _) in enumerate(_days()):
        for product in PRODUCTS:
            v = result.values[product]
            if not (_le(v["lp24"], v["general_split"]) and _le(v["general_split"], v["ceiling"])):
                broken.append((k, product, v["lp24"], v["general_split"], v["ceiling"]))
    assert not broken, broken


def test_diagonal_affine_without_hour0_gain_general_split():
    broken, strict = [], 0
    for result, extra, _ in _days():
        for product in PRODUCTS:
            e, v = extra[product], result.values[product]
            fixed, split = v["general_split_no_hour0_gain"], v["general_split"]
            if not (_le(e["diag"], e["aff"]) and _le(e["aff"], fixed) and _le(fixed, split)):
                broken.append((product, e, fixed, split))
        strict += _below(
            result.values["hourly"]["general_split_no_hour0_gain"],
            result.values["hourly"]["general_split"],
        )
    assert not broken, broken
    print(f"hourly: the split without hour-0 gain strictly below the general split: {strict}")
    assert strict >= 1


def test_every_box_at_most_the_ceiling():
    results = [result for result, _, _ in _days()]
    assert order_violations(results) == []
    below = {method: 0 for method in METHODS}
    for r in results:
        for method in METHODS:
            value, ceiling = r.values["hourly"][method], r.values["hourly"]["ceiling"]
            assert _le(value, ceiling)
            below[method] += _below(value, ceiling)
        hourly = r.values["hourly"]
        assert _le(hourly["lp27_smallest_found"], hourly["lp27_largest_found"])
        assert _le(hourly["lp27_largest_found"], hourly["lp27_largest_bound"])
    print(f"hourly boxes strictly below the ceiling, days of {N_DAYS}: {below}")
    assert all(count >= MIN_BELOW for count in below.values()), below


def test_affine_tree_ceiling():
    broken = []
    for result, extra, _ in _days():
        for product in PRODUCTS:
            e, ceiling = extra[product], result.values[product]["ceiling"]
            if not (_le(e["aff"], e["caus"]) and _le(e["caus"], ceiling)):
                broken.append((product, e, ceiling))
    assert not broken, broken


def test_block_at_most_hourly():
    strictly = 0
    for result, extra, _ in _days():
        below = False
        for key in ("ceiling", *METHODS):
            block, hourly = result.values["block"][key], result.values["hourly"][key]
            assert _le(block, hourly), (key, result)
            below |= _below(block, hourly)
        for key in ("diag", "aff", "caus"):
            assert _le(extra["block"][key], extra["hourly"][key])
        strictly += below
    print(f"days with some block value strictly below its hourly value: {strictly}")
    assert strictly >= MIN_BELOW


def test_lp27_trace_at_least_lp24_alpha_on_the_base_set_hours():
    checked = 0
    for result, _, columns in _days():
        if result.alpha_24 is None:
            continue
        checked += 1
        assert _le(columns * result.alpha_24, result.trace_27), (columns, result)
    assert checked >= N_DAYS // 2


def test_orders_the_plan_does_not_claim_go_both_ways():
    counts = {}
    for result, extra, _ in _days():
        for product in PRODUCTS:
            v, caus = result.values[product], extra[product]["caus"]
            pairs = {
                "lp27_largest_found vs lp24": (v["lp27_largest_found"], v["lp24"]),
                "lp27_smallest_found vs lp24": (v["lp27_smallest_found"], v["lp24"]),
                "lp27_largest_found vs general_split": (
                    v["lp27_largest_found"],
                    v["general_split"],
                ),
                "lp24 vs V_caus": (v["lp24"], caus),
                "lp27_largest_found vs V_caus": (v["lp27_largest_found"], caus),
                "general_split vs V_caus": (v["general_split"], caus),
            }
            for name, (a, b) in pairs.items():
                slack = TOLERANCE * max(1.0, abs(b))
                key = "equal" if abs(a - b) <= slack else ("above" if a > b else "below")
                counts.setdefault((product, name), {"above": 0, "equal": 0, "below": 0})
                counts[(product, name)][key] += 1
    for (product, name), c in sorted(counts.items()):
        print(f"{product}: {name}: {c}")
    assert all(sum(c.values()) == N_DAYS for c in counts.values())
    for name in BOTH_WAYS:
        c = counts[("hourly", name)]
        assert c["above"] >= 1 and c["below"] >= 1, (name, c)


FINE = {
    "ceiling": 3.0,
    "lp24": 1.5,
    "lp27_smallest_found": 3.0,
    "lp27_largest_found": 3.0,
    "lp27_largest_bound": 3.0,
    "general_split": 3.0,
    "general_split_no_hour0_gain": 2.0,
}
BLOCK = {
    "ceiling": 2.0,
    "lp24": 1.0,
    "lp27_smallest_found": 2.0,
    "lp27_largest_found": 2.0,
    "lp27_largest_bound": 2.0,
    "general_split": 2.0,
    "general_split_no_hour0_gain": 2.0,
}


def _result(values_hourly, values_block):
    return DayResult(
        "u",
        "A",
        "2020-01-01",
        2,
        2,
        None,
        {"hourly": dict(values_hourly), "block": dict(values_block)},
        {p: dict.fromkeys(values_hourly, "optimal") for p in PRODUCTS},
        None,
        None,
    )


def test_the_order_check_reports_a_box_above_the_ceiling_and_a_block_above_hourly():
    assert order_violations([_result(FINE, BLOCK)]) == []
    above = order_violations([_result(FINE | {"general_split": 3.1}, BLOCK)])
    assert len(above) == 1 and "general_split 3.1 above the ceiling 3.0" in above[0]
    block_above = order_violations([_result(FINE, BLOCK | {"lp24": 1.8})])
    assert len(block_above) == 1 and "lp24 block 1.8 above hourly 1.5" in block_above[0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"lp24": 2.5, "general_split": 2.4}, "lp24 2.5 above general_split 2.4"),
        (
            {"general_split_no_hour0_gain": 2.8, "general_split": 2.7},
            "general_split_no_hour0_gain 2.8 above general_split 2.7",
        ),
        (
            {"lp27_smallest_found": 2.9, "lp27_largest_found": 2.8},
            "lp27_smallest_found 2.9 above lp27_largest_found 2.8",
        ),
        (
            {"lp27_smallest_found": 2.0, "lp27_largest_found": 2.9, "lp27_largest_bound": 2.8},
            "lp27_largest_found 2.9 above lp27_largest_bound 2.8",
        ),
    ],
)
def test_the_order_check_reports_each_order_between_methods(change, message):
    found = order_violations([_result(FINE | change, BLOCK)])
    assert len(found) == 1 and message in found[0], found
