"""`day_intervals` on a run made through the sealed route that left a unit out because the
held-out truth build abstained it: the summary gives one line for the unit, as the commands that
read the build do, and its results hold the route's record as the run wrote it. The run is the
hand case of `test_day_intervals`, with the route's record added to its results.
"""

import json

import test_day_intervals as base
from interval_fake import write_run
from test_day_intervals import HAND, hand_job, inject, results, run, verified  # noqa: F401

from fleetreserve import day_intervals

REASON = "no calibration day at the start draw"
ROUTE = {
    "kind": "held-out",
    "held_out_inputs": {
        "truth_run": "20261116T000000Z-heldout",
        "abstained": {"korea-visits": REASON},
    },
}


def test_the_summary_names_the_unit_the_run_left_out(tmp_path, monkeypatch):
    inject(monkeypatch, base.CALIBRATION_DRAWS, base.TEST_DRAWS)
    folder = write_run(
        tmp_path / "20200101T000000Z-made-up", [hand_job()], extra={"sealed_route": ROUTE}
    )
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    lines = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text().splitlines()
    assert (
        f"korea-visits: abstained in the held-out truth build 20261116T000000Z-heldout: {REASON}; "
        "not scored, counted as not passing" in lines
    )
    assert results(tmp_path / "out")["sealed_route"] == ROUTE


def test_a_run_without_an_abstained_unit_has_no_such_line(tmp_path, monkeypatch):
    inject(monkeypatch, base.CALIBRATION_DRAWS, base.TEST_DRAWS)
    route = {"kind": "held-out", "held_out_inputs": {"truth_run": "R"}}
    folder = write_run(
        tmp_path / "20200101T000000Z-made-up", [hand_job()], extra={"sealed_route": route}
    )
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    assert "abstained" not in (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text()
    assert (
        json.loads((tmp_path / "out" / day_intervals.RESULTS_NAME).read_text())["sealed_route"]
        == route
    )
