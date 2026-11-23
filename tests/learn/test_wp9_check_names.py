"""The names `tools/wp9_check.py` reads from the outputs of commands it cannot import.

The tool runs under `uv run --frozen python`, without the learn packages, so it cannot import
`soft_scale_check`, `size_resolve_whatif`, `size_resolve_summary`, `size_ct`, `size_net` or
`size_scale` (they import PyTorch or LightGBM, directly or through others) and writes their
names and settings out itself. This test, in the learn lane, pins each one to its source, so a
renamed file, key, sizing or setting breaks here rather than turning the item into a silent
"could not run": the results file names, the comparison kinds and deciding fields of T8's check,
its default days per unit, the shuffled control's prefix, the run's field naming the development
units scored, the draws the item needs (the sizing comparison's default), the names of conformal
training and of the day-size net, the arms, the tiers, the net's level conformal training starts
from, the key of the 5th percentile over draws, and the keys of the gains by draw.
"""

import importlib
import sys
from pathlib import Path

from fleetreserve import (
    crc,
    size_ct,
    size_net,
    size_resolve_summary,
    size_resolve_whatif,
    size_scale,
    soft_scale_check,
)

TOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOP / "tools"))
check = importlib.import_module("wp9_check")


def test_the_names_of_t8s_check_are_its_own():
    assert check.SOFT_RESULTS == soft_scale_check.RESULTS_NAME
    assert check.KINDS == soft_scale_check.KINDS
    assert check.DECIDING == soft_scale_check.DECIDING
    assert check.DAYS_PER_UNIT == soft_scale_check.DAYS
    assert soft_scale_check.__name__ == check.SOFT_MODULE


def test_the_names_of_the_sizing_comparison_are_its_own():
    assert check.SIZING_RESULTS == size_resolve_whatif.RESULTS_NAME
    assert size_resolve_whatif.__name__ == check.SIZING_MODULE
    assert check.CONTROL == size_resolve_summary.CONTROL
    assert check.DEVELOPMENT == size_resolve_whatif.DEVELOPMENT
    assert check.DRAWS_NEEDED == size_resolve_whatif.CONTROL_DRAWS
    assert check.P5 in size_resolve_summary.SPREAD
    assert {check.GAINS, check.SMALLEST, check.AT_OR_ABOVE} <= set(size_resolve_summary.BY_DRAW)
    assert size_resolve_summary.control_figures({}, [])[check.BY_DRAW] is not None
    assert check.CT == size_ct.KIND
    assert check.NET == size_net.KIND
    assert check.ARMS == size_scale.ARMS
    assert check.TIERS == size_scale.TIERS
    assert check.START_LEVEL == size_ct.CtSettings().alpha == crc.ALPHA_SLOT
