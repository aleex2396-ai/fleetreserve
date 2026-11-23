"""`two_way_day` on hand-made days: the tree's replay, the enlarged box, the order, empty fleets."""

import datetime
from dataclasses import replace

import numpy as np

from fleetreserve import products, two_way_day
from fleetreserve.carsets import true_rows
from fleetreserve.day import Day, TrueCar
from fleetreserve.tree import corner_tree

DAY = Day(
    "u",
    datetime.date(2020, 1, 1),
    3,
    2,
    0.0,
    (TrueCar(4, 3.0, 6.0), TrueCar(6, 2.0, 4.0), TrueCar(3, 4.0, 2.0)),
)


def test_both_products_match_and_keep_the_order():
    found = two_way_day.two_way_day("u", "A", "2020-01-01", DAY)
    assert found.error is None
    for name in two_way_day.PRODUCTS:
        test = found.tests[name]
        assert test.matched is True
        assert test.replay_paths == 8 and test.replay_failures == 0
        assert abs(test.enlarged_scale - 1.0) < 1e-6
        v = test.values
        assert v["diag"] <= v["causal"] + 1e-6 <= v["tree"] + 2e-6 <= v["ceiling"] + 3e-6
        assert v["tree"] > 0


def test_replay_finds_node_powers_that_break_a_row_or_the_request():
    rows = true_rows(DAY, budget=two_way_day.BUDGET)
    tree = corner_tree(rows, products.hourly())
    paths, failing, _, _ = two_way_day.replay(rows, tree.policy)
    assert (paths, failing) == (8, 0)
    powers = np.array(tree.policy.node_powers)
    powers[0, -1] += 1.0  # one car, one leaf: the request there is missed
    _, failing, _, miss = two_way_day.replay(rows, replace(tree.policy, node_powers=powers))
    assert failing == 1 and miss > 0.5
    # The same node powers against a request moved up by 1 kW: every row holds, every path misses.
    shifted = replace(tree.policy, m=tree.policy.m + 1.0)
    _, failing, slack, miss = two_way_day.replay(rows, shifted)
    assert slack >= -two_way_day.ROW_TOLERANCE
    assert failing == 8 and abs(miss - 1.0) < 1e-6


def test_a_smaller_box_than_the_trees_can_be_enlarged():
    rows = true_rows(DAY, budget=two_way_day.BUDGET)
    tree = corner_tree(rows, products.block())
    half = replace(tree.policy, w=tree.policy.w / 2)
    scale, outcome, has_width, _ = two_way_day.enlarged_scale(rows, half)
    assert outcome == "optimal" and has_width
    assert abs(scale - 2.0) < 1e-5


def test_a_box_without_width_is_enlarged_in_every_hour():
    stuck = replace(DAY, cars=(TrueCar(2, 3.0, 6.0),))  # must take 6 kWh in hours 0 and 1
    found = two_way_day.two_way_day("u", "A", "2020-01-01", stuck)
    for name in two_way_day.PRODUCTS:
        test = found.tests[name]
        assert test.values["tree"] == 0.0
        assert test.enlarged_scale < two_way_day.ENLARGE and test.matched is True


def test_an_empty_fleet_has_zero_values_and_nothing_to_replay():
    empty = replace(DAY, cars=())
    found = two_way_day.two_way_day("u", "A", "2020-01-01", empty)
    for name in two_way_day.PRODUCTS:
        test = found.tests[name]
        assert set(test.values.values()) == {0.0}
        assert test.outcomes["tree"] == two_way_day.EMPTY_FLEET
        assert test.replay_paths == 0 and test.matched is True
