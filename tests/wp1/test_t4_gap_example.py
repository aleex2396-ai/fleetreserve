"""Plan section 4, T4 unit test: a gap without in-window departure.

Car D is present in hours 1-2 (rate 2, needs exactly 2 kWh). Car C is present in hours 1-3 (rate 1,
room 10 kWh, no minimum). The service hours are 1 and 2, hour 3 is a recovery hour. The box has
centre (1.1, 1.9) and widths (0.1, 0.9). Plan: hindsight lam = 1.0; corner tree and causal affine
rule lam = 0.556.

Hindsight is computed by two routes (the fleet's exact set functions, and one power plan per corner
path); the tree and the affine rule are each one programme (`toy_values.py`).
"""

import pytest
from toy_values import (
    affine_scale,
    hindsight_scale,
    hindsight_scale_by_paths,
    tree_scale,
)

from fleetreserve.gpoly import Car

SERVICE_HOURS = [1, 2]
CARS = [
    Car(frozenset({1, 2}), 0.0, 2.0, 2.0, 2.0),  # D: needs exactly 2 kWh
    Car(frozenset({1, 2, 3}), 0.0, 1.0, 0.0, 10.0),  # C: room 10 kWh, no minimum
]
CENTRE = {1: 1.1, 2: 1.9}
WIDTHS = {1: 0.1, 2: 0.9}


def _values():
    args = (CARS, SERVICE_HOURS, CENTRE, WIDTHS)
    return {
        "hindsight (set functions)": hindsight_scale(*args),
        "hindsight (corner paths)": hindsight_scale_by_paths(*args),
        "tree": tree_scale(*args),
        "affine": affine_scale(*args),
    }


def test_hindsight_is_1_0_and_tree_and_affine_are_0_556():
    values = _values()
    for name, value in values.items():
        print(f"{name}: {value!r}")
    assert values["hindsight (set functions)"] == pytest.approx(1.0, abs=5e-4)
    assert values["hindsight (corner paths)"] == pytest.approx(1.0, abs=5e-4)
    assert values["tree"] == pytest.approx(0.556, abs=5e-4)
    assert values["affine"] == pytest.approx(0.556, abs=5e-4)


def test_the_two_hindsight_routes_agree_and_the_order_holds():
    values = _values()
    assert values["hindsight (set functions)"] == pytest.approx(
        values["hindsight (corner paths)"], abs=1e-7
    )
    assert values["affine"] <= values["tree"] + 1e-7
    assert values["tree"] <= values["hindsight (set functions)"] + 1e-7
