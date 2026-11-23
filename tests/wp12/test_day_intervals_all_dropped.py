"""`day_intervals` on a run whose held-out unit the held-out truth build abstained because every
one of its rows was dropped by cleaning: the unit is in the pooled "K of N" line and the unit
table, as the other abstained units are (`test_day_intervals_abstained_counts`).

The reason is the one the build itself records, from a made-up held-out file in which every row of
TRO_R is dropped (`heldout_truth_fake`); the run is the hand case of `test_day_intervals`.
"""

import sys
from pathlib import Path

import test_day_intervals as base
from interval_fake import NAME, RUNG, write_run
from test_day_intervals import HAND, hand_job, inject, run, verified  # noqa: F401

_WP2 = Path(__file__).resolve().parents[1] / "wp2"  # the made-up held-out files
if str(_WP2) not in sys.path:
    sys.path.insert(0, str(_WP2))
from heldout_truth_fake import (  # noqa: E402
    NORWAY_DAY,
    NORWAY_START,
    OSL_T_DAYS,
    TRO_R_DAYS,
    norway_bytes,
    norway_days,
    norway_rows,
    unchecked,
)

import fleetreserve  # noqa: E402
from fleetreserve import day_intervals, gate_config, heldout_truth_build, load_heldout_norway, truth  # noqa: E402

BUILD = "20261116T000000Z-heldout"
MATRIX_PATH = Path(fleetreserve.__file__).resolve().parents[2] / "gate" / "matrix.json"


def _reason(tmp_path) -> str:
    no_energy = [(user, start, end, 0.0) for user, start, end, _ in NORWAY_DAY]
    rows = norway_rows("OSL_T", norway_days(NORWAY_START, OSL_T_DAYS))
    rows += norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS), pattern=no_energy)
    loaded = {"norway": unchecked(load_heldout_norway, norway_bytes(tmp_path / "norway", rows))}
    sessions, dropped = loaded["norway"]
    result = heldout_truth_build.HeldoutBuild(
        truth.MAIN,
        heldout_truth_build.SplitRule.of(gate_config.load(MATRIX_PATH), "gate/matrix.json"),
        0,
    )
    heldout_truth_build.build_sites(result, "norway", ("TRO_R",), sessions, dropped)
    return result.abstained["TRO_R"]["reason"]


def test_the_unit_with_every_row_dropped_is_in_the_pooled_line_and_the_table(tmp_path, monkeypatch):
    reason = _reason(tmp_path)
    assert reason.startswith("every row dropped by cleaning: 36 rows")
    text = f"abstained in the held-out truth build {BUILD}: {reason}; counts as not passing"
    route = {
        "kind": "held-out",
        "held_out_inputs": {"truth_run": BUILD, "abstained": {"TRO_R": reason}},
    }
    inject(monkeypatch, base.CALIBRATION_DRAWS, base.TEST_DRAWS)
    folder = write_run(
        tmp_path / "20200101T000000Z-made-up", [hand_job()], extra={"sealed_route": route}
    )
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    lines = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text().splitlines()
    pooled = [t for t in lines if t.startswith(f"{NAME} | pooled over ") and f"| {RUNG} |" in t]
    assert len(pooled) == 2
    for line in pooled:
        assert line.startswith(f"{NAME} | pooled over 1 of 2 units | {RUNG} | ")
        assert f"units left out of the pooling: TRO_R ({text})" in line
    assert f"{NAME} | TRO_R: {text}" in lines
