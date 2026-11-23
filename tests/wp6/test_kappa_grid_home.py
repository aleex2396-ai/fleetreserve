"""The grid of kappa (plan section 2: 0.05, 0.1, 0.2, 0.3, 0.4, 0.5) has one home in the package,
`forecast_empirical.KAPPAS`, and every module that works on the plan's grid reads it from there.

1. The grid is the six levels; the what-if levels off the grid are 0.6 and 0.7 and none of them is
   a level of the grid; the tuning levels (`TUNING_KAPPAS`, read by the tuned models' scores) are
   the grid's levels up to 0.3.
2. The modules that take the grid hold the home's own object, not a copy of it.
3. No second copy of the grid is written as a literal under `src/`: a list, tuple or set of
   numbers that holds 0.05, 0.1, 0.2 and 0.3 in a row is found only in the home and in the two
   lists that are not the plan's grid of kappa, with the reason each is not (a scan of every
   module with `ast`, tried on made-up sources first).
"""

import ast
from pathlib import Path

import pytest

import fleetreserve
from fleetreserve import (
    driver_risk,
    forecast_empirical,
    forecast_model,
    forecast_report,
    gate_config,
    kappa_choice,
)

PLAN_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
FIRST_FOUR = (0.05, 0.1, 0.2, 0.3)
SRC = Path(fleetreserve.__file__).parent
# Files that hold the four levels in a row and are not the plan's grid of kappa.
ELSEWHERE = {
    "forecast_empirical.py": "the home of the grid itself",
    "crc.py": "GRID: the scales of the calibration (0.05 to 2), not levels of kappa",
    "size_gbm.py": "LEVELS: the quantile levels of the sizing model, with 0.5",
}


def copies_of_the_four_levels(source: str) -> list[int]:
    """The line numbers of every list, tuple or set of numbers in `source` that holds 0.05, 0.1,
    0.2 and 0.3 in a row."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.List | ast.Tuple | ast.Set):
            values = [
                e.value
                for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, int | float)
            ]
            if len(values) == len(node.elts) and any(
                tuple(values[i : i + 4]) == FIRST_FOUR for i in range(len(values))
            ):
                found.append(node.lineno)
    return found


# --- 1. the levels -------------------------------------------------------------------------------


def test_the_grid_has_six_levels_and_the_what_if_levels_lie_above_it():
    assert forecast_empirical.KAPPAS == PLAN_GRID
    assert forecast_empirical.WHAT_IF_KAPPAS == (0.6, 0.7)
    assert not set(forecast_empirical.WHAT_IF_KAPPAS) & set(forecast_empirical.KAPPAS)
    assert min(forecast_empirical.WHAT_IF_KAPPAS) > max(forecast_empirical.KAPPAS)


def test_the_tuning_levels_are_the_grids_levels_up_to_0_3():
    assert forecast_empirical.TUNING_KAPPAS == FIRST_FOUR
    assert forecast_empirical.KAPPAS[:4] == FIRST_FOUR


# --- 2. the modules read the home ----------------------------------------------------------------


def test_the_modules_that_take_the_grid_hold_the_homes_object():
    home = forecast_empirical.KAPPAS
    assert driver_risk.LEVELS is home
    assert kappa_choice.LEVELS is home
    assert gate_config.KAPPAS is home
    assert forecast_model.KAPPAS is home
    assert forecast_report.KAPPAS is home
    assert driver_risk.WHAT_IF_LEVELS is forecast_empirical.WHAT_IF_KAPPAS


# --- 3. no second copy ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "lines"),
    [
        ("K = (0.05, 0.1, 0.2, 0.3)\n", [1]),
        ("x = 1\nK = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]\n", [2]),
        ("K = {0.05, 0.1, 0.2, 0.3}\n", [1]),
        ("K = [0.5, 0.05, 0.1, 0.2, 0.3, 0.5]\n", [1]),
        ("K = (0.05, 0.1, 0.15, 0.2, 0.3)\n", []),
        ("K = (0.05, 0.1, 0.3, 0.2)\n", []),
        ("K = (0.05, 0.1, 0.2, n)\n", []),
        ("K = '0.05, 0.1, 0.2, 0.3'\n", []),
    ],
)
def test_the_scan_finds_a_copy_of_the_four_levels_and_nothing_else(source, lines):
    assert copies_of_the_four_levels(source) == lines


def test_the_four_levels_in_a_row_are_written_only_in_the_home_and_the_two_other_lists():
    found = {
        path.name
        for path in sorted(SRC.rglob("*.py"))
        if copies_of_the_four_levels(path.read_text())
    }
    assert found == set(ELSEWHERE), sorted(found - set(ELSEWHERE))
