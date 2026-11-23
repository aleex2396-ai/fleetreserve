"""`learn_run --model cox` and `--model weibull_aft` end to end on the made-up gate units of
`learn_fake` and eight made-up training units (one per `truth_build.TRAIN_UNITS`, 24 days of three
cars), with a made-up verified truth build and the matrix pair at kappa 0.1.

Each ends with exit 0 and writes `learn.json` with, for every gate unit at kappa 0.1, departure
pinball losses, a concordance index and the calibrated offers of the configuration and its oracle
partner; `fit_units` are the training units; the forecast table has a row for every car of every
counting day, each with a departure forecast.
"""

import json

import numpy as np
import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_cars, write_build
from learn_fake import PAIR, small_gate_units, small_matrix

from fleetreserve import gate_run, learn_run, run_verify
from fleetreserve.forecast_model import read_table
from fleetreserve.truth_build import TRAIN_UNITS


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    training = {
        unit: gate_cars(24, f"T{j}", early_every=3 + j) for j, unit in enumerate(TRAIN_UNITS)
    }
    run, tables = write_build(folder, {**small_gate_units(), **training})
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]


@pytest.mark.parametrize("name", ["cox", "weibull_aft"])
def test_the_model_runs_end_to_end(build, tmp_path, monkeypatch, capsys, name):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = learn_run.main([*build, "--model", name])
    out = capsys.readouterr().out
    assert code == 0, out
    results = json.loads((tmp_path / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == name
    assert results["fit_units"] == list(TRAIN_UNITS)
    assert not results["failed_configurations"]
    table = read_table(tmp_path / learn_run.TABLE_NAME)
    cars = small_gate_units()
    assert len(table) == sum(len(cars[u]) for u in GATE_UNITS)  # one level, every day counts
    assert sorted(set(table["unit"])) == sorted(GATE_UNITS)
    assert np.isfinite(table["d_hat"]).all()
    for unit in GATE_UNITS:
        entry = results["units"][unit]
        score = entry["kappas"]["0.1"]
        assert score["pinball_loss"]["departure_hours"]["mean"] is not None
        assert score["cars_with_departure_forecast"] == score["cars"]
        assert score["concordance_index"]["value"] is not None
        offer = entry["configurations"][PAIR[0]]
        assert set(offer) >= {"rungs", "oracle_partner"}
        assert offer["oracle_partner"]["name"] == PAIR[1]
    calibrated = results["units"][GATE_UNITS[1]]["configurations"][PAIR[0]]
    for rung in ("fixed", "one-way"):
        assert calibrated["rungs"][rung]["test"]["mean_offer_kwh_per_day"] >= 0
    timing = json.loads((tmp_path / learn_run.TIMING_NAME).read_text())
    assert timing["fit_seconds"] > 0
