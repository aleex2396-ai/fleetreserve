"""A job with too few offering calibration days does not stop the intervals of the run.

`gate_run` on the small made-up gate units of `learn_fake` (two service hours) with
`n_calibration` 8, recorded with `--interval-seed 3 --interval-resamples 30`: the two split units
have 8 calibration days each, fewer than the 9 offering days tier B needs and the 19 tier A needs
(`crc.enough_days`), so the run's own calibration evaluates no scale and abstains (scale 0), and
the per-day file holds no scale at all. So does every resample: a resample draws 8 calibration
days, and the calibration rule itself (`crc.calibrate_search`, as the run calls it) abstains on
them without evaluating anything; nothing in the intervals treats the case apart.

`day_intervals` reads the run (exit 0) and gives, for every configuration, split unit, rung and
tier: the run's scale 0, every one of the 30 resamples exact, at scale 0 and counted as too few
offering calibration days; the offer, per counting test day and per car, 0 with interval [0, 0];
the summary line "abstains in 30 of 30 resamples". No loss interval: the run's scale is 0.
"""

import json

import pytest
from gate_fake import COMMIT, write_build
from interval_fake import write_manifest
from learn_fake import small_gate_units, small_matrix

from fleetreserve import day_intervals, gate_run, run_verify
from fleetreserve.gate_config import TIER_LABELS
from fleetreserve.rungs import RUNGS


@pytest.fixture(scope="module")
def few(tmp_path_factory):
    folder = tmp_path_factory.mktemp("few")
    truth, tables = write_build(folder, small_gate_units(), service_hours=2)
    document = small_matrix()
    document["n_calibration"] = 8
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    out = folder / "20200101T000000Z-few"
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(truth)]
        assert (
            gate_run.main([*arguments, "--interval-seed", "3", "--interval-resamples", "30"]) == 0
        )
    write_manifest(out, "fleetreserve.gate_run", commit=COMMIT)
    return out


def test_the_run_holds_no_scale(few):
    losses = json.loads((few / gate_run.DAY_LOSSES_NAME).read_text())
    assert losses["jobs"]
    for record in losses["jobs"]:
        assert 0 < len(record["calibration"]["days"]) <= 8
        assert record["calibration"]["scales"] == []
        assert record["resampled"]["pairs"]["day"] == []


def test_every_resample_abstains(few, tmp_path, monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(day_intervals, "head_commit", lambda top: COMMIT)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    assert day_intervals.main(["--run", str(few), "--seed", "3", "--resamples", "30"]) == 0
    found = json.loads((out / day_intervals.RESULTS_NAME).read_text())
    text = (out / day_intervals.SUMMARY_NAME).read_text()
    assert found["problems"] == []
    checked = 0
    for name, units in found["configurations"].items():
        for unit, rungs in units.items():
            for rung in RUNGS:
                for tier in ("A", "B"):
                    mine = rungs[rung][tier]
                    assert mine["scale"]["point"] == 0.0
                    assert mine["scale"]["resamples_exact"] == 30
                    assert mine["scale"]["interval"] == [0.0, 0.0]
                    abstaining = mine["abstaining"]
                    assert abstaining["resamples_at_scale_zero"] == 30
                    assert abstaining["resamples_too_few_offering_calibration_days"] == 30
                    for key in ("offer_kwh_per_counting_test_day", "offer_kwh_per_car_per_day"):
                        assert mine[key]["point"] == 0.0
                        assert mine[key]["interval"] == [0.0, 0.0]
                    assert mine["availability"]["interval"] is None
                    line = f"{name} | {unit} | {rung} | {TIER_LABELS[tier]}: "
                    assert any(
                        t.startswith(line) and "abstains in 30 of 30 resamples" in t
                        for t in text.splitlines()
                    )
                    checked += 1
    assert checked == 2 * 2 * len(RUNGS) * 2  # two configurations, two split units
