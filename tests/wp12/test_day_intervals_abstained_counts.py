"""`day_intervals` on a run made through the sealed route that left a unit out because the
held-out truth build abstained it: every figure over the held-out units names that unit.

The run is the hand case of `test_day_intervals` (one scored unit) with the route's record added
to its results, the record naming one more unit as abstained, so the held-out units are two.

1. The pooled line reads "pooled over 1 of 2 units" and its left-out part names the abstained
   unit with the build's run, the reason and "counts as not passing"; it never reads "none".
2. The per-unit table has a row for the abstained unit under each configuration, with that text
   and no figure.
3. The results hold the unit under `abstained_units` and in the pooled block's `left_out`, and
   the figures of the scored unit are those of a run without the record.
4. A run without an abstained unit reads as before: "pooled over 1 units", left out "none".
"""

import json

import test_day_intervals as base
from interval_fake import NAME, RUNG, write_run
from test_day_intervals import HAND, hand_job, inject, results, run, verified  # noqa: F401

from fleetreserve import day_intervals

BUILD = "20261116T000000Z-heldout"
REASON = "no calibration day at the start draw"
TEXT = f"abstained in the held-out truth build {BUILD}: {REASON}; counts as not passing"
ROUTE = {
    "kind": "held-out",
    "held_out_inputs": {"truth_run": BUILD, "abstained": {"korea-visits": REASON}},
}


def _lines(tmp_path, monkeypatch, route) -> list[str]:
    inject(monkeypatch, base.CALIBRATION_DRAWS, base.TEST_DRAWS)
    folder = write_run(
        tmp_path / "20200101T000000Z-made-up", [hand_job()], extra={"sealed_route": route}
    )
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    return (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text().splitlines()


def _pooled(lines) -> list[str]:
    """The pooled lines of the summary, one per tier (there are two)."""
    found = [t for t in lines if t.startswith(f"{NAME} | pooled over ") and f"| {RUNG} |" in t]
    assert len(found) == 2
    return found


def test_the_pooled_line_counts_the_abstained_unit_among_the_units(tmp_path, monkeypatch):
    lines = _lines(tmp_path, monkeypatch, ROUTE)
    pooled = _pooled(lines)
    for line in pooled:
        assert line.startswith(f"{NAME} | pooled over 1 of 2 units | {RUNG} | ")
        assert f"units left out of the pooling: korea-visits ({TEXT})" in line
        assert "left out of the pooling: none" not in line


def test_the_unit_table_has_a_row_for_the_abstained_unit_and_no_figure(tmp_path, monkeypatch):
    lines = _lines(tmp_path, monkeypatch, ROUTE)
    assert f"{NAME} | korea-visits: {TEXT}" in lines
    row = f"{NAME} | korea-visits: {TEXT}"
    assert "offer" not in row and "availability" not in row and "shortfall" not in row


def test_the_results_hold_the_unit_and_the_scored_unit_is_unchanged(tmp_path, monkeypatch):
    _lines(tmp_path, monkeypatch, ROUTE)
    found = results(tmp_path / "out")
    assert found["abstained_units"] == {"korea-visits": TEXT}
    left = found["pooled"]["configurations"][NAME]["left_out"]
    assert left == [{"unit": "korea-visits", "reason": TEXT, "abstained": True}]
    other = tmp_path / "plain"
    other.mkdir()
    inject(monkeypatch, base.CALIBRATION_DRAWS, base.TEST_DRAWS)
    folder = write_run(other / "20200101T000000Z-made-up", [hand_job()])
    assert run(folder, other / "out", monkeypatch, *HAND) == 0
    plain = results(other / "out")
    assert found["configurations"] == plain["configurations"]
    for rung, tiers in found["pooled"]["configurations"][NAME]["rungs"].items():
        assert tiers == plain["pooled"]["configurations"][NAME]["rungs"][rung]


def test_a_run_without_an_abstained_unit_reads_as_before(tmp_path, monkeypatch):
    route = {"kind": "held-out", "held_out_inputs": {"truth_run": BUILD}}
    lines = _lines(tmp_path, monkeypatch, route)
    for line in _pooled(lines):
        assert line.startswith(f"{NAME} | pooled over 1 units | {RUNG} | ")
        assert "units left out of the pooling: none" in line
    assert "abstained_units" not in json.loads(
        (tmp_path / "out" / day_intervals.RESULTS_NAME).read_text()
    )
