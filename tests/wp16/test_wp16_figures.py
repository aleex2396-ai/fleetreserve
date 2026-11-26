"""`wp16_figures`: each figure kind is drawn into a PNG, and a figure with nothing to draw says so.

Red when a kind is not drawn, when a missing value is drawn as a value, or when an interval that
does not hold its point (a percentile interval need not) breaks the drawing. Needs matplotlib,
which the default environment has (the `figures` group and the `flexitroid` dependency).
"""

import pytest

from fleetreserve import wp16_figures
from fleetreserve.wp16_cells import Interval, Missing

PNG = b"\x89PNG"
SPECS = {
    "o3": {
        "kind": "o3",
        "points": [("u1", 40.0, 50.0, 44.0, None), ("u2", 30.0, 35.0, 31.0, 60.0)],
    },
    "o4": {
        "kind": "o4",
        "model": "lgbm_tail",
        "points": [
            ("u1", Interval(0.93, 0.90, 0.95), Interval(0.03, 0.01, 0.04)),
            ("u2", Interval(0.80, 0.82, 0.85), Interval(0.20, 0.25, 0.30)),  # point outside
            ("u3", Missing("failed: x"), None),
        ],
    },
    "o9": {
        "kind": "o9",
        "curves": [
            ("lgbm_tail", "u1", [(0, 0.9, 0.2), (7, 0.92, None)]),
            ("lgbm", "u1", [(0, 0.91, 0.25)]),
            ("lgbm_tail", "u2", [(0, None, None)]),
        ],
    },
    "o10": {
        "kind": "o10",
        "rows": [("lgbm_tail", "Cox", 12.5), ("lgbm_tail", "earlier_gbm", None)],
    },
}


@pytest.mark.parametrize("kind", sorted(SPECS))
def test_each_kind_is_drawn_into_a_png(kind, tmp_path):
    name = wp16_figures.render(SPECS[kind], tmp_path)
    assert (tmp_path / name).read_bytes().startswith(PNG)
    assert name.startswith("o" + kind[1:])


@pytest.mark.parametrize(
    ("spec", "name"),
    [
        ({"kind": "o3", "points": []}, "o3-ceiling-shares.png"),
        ({"kind": "o4", "model": "m", "points": []}, "o4-availability-shortfall-m.png"),
        ({"kind": "o9", "curves": []}, "o9-cold-start.png"),
        ({"kind": "o10", "rows": [("m", "b", None)]}, "o10-model-comparison.png"),
    ],
)
def test_a_figure_with_nothing_to_draw_is_still_written_and_says_so(spec, name, tmp_path):
    assert wp16_figures.render(spec, tmp_path) == name
    assert (tmp_path / name).read_bytes().startswith(PNG)
