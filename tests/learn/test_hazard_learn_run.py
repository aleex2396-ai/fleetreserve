"""`learn_run --model hazard` and `dial_choice --model hazard` end to end on made-up units, the way
`test_lgbm_tuned` and `test_dial_choice_lgbm` run the LightGBM models: the three gate units at
sites of the committed site mappings (`nrel`, `OSL_1`, `BER`), eight made-up training units (one
per `truth_build.TRAIN_UNITS`, 24 days of three cars, each at the site of its own name), a made-up
verified truth build, and the model at its default settings.

Each ends with exit 0 and writes its results: the model fitted on the training units only, a
forecast table with a row for every car of every counting day at each level that is the table the
model's `predict` returned, no failed configuration, and for `dial_choice` the two tiers and a
dial that is one of the dials.
"""

import json

import pandas as pd
import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_cars, matrix_document, write_build
from learn_fake import small_matrix

from fleetreserve import dial_choice, gate_run, learn_run, run_verify
from fleetreserve.forecast_model import SURVIVAL_COLUMNS, read_table
from fleetreserve.models import hazard
from fleetreserve.truth_build import TRAIN_UNITS

NAMES = (
    "main-rounded-hour-k0.05-empirical",
    "main-rounded-hour-k0.05-oracle",
    "main-rounded-hour-k0.1-empirical",
    "main-rounded-hour-k0.1-oracle",
    "main-rounded-hour-k0.2-empirical",
    "main-rounded-hour-k0.2-oracle",
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.03-empirical",
    "main-rounded-hour-k0.3-eps0.03-oracle",
    "main-rounded-hour-k0.3-eps0.1-empirical",
    "main-rounded-hour-k0.3-eps0.1-oracle",
)


def _units() -> dict:
    units = {
        GATE_UNITS[0]: gate_cars(5, "nrel"),
        GATE_UNITS[1]: gate_cars(24, "nrel", early_every=4),
        GATE_UNITS[2]: gate_cars(12, "OSL_1", early_every=3) + gate_cars(12, "BER", early_every=5),
    }
    for j, unit in enumerate(TRAIN_UNITS):
        units[unit] = gate_cars(24, unit, early_every=3 + j)
    return units


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = _units()
    run, tables = write_build(folder, units)
    rows = sum(len(units[u]) for u in GATE_UNITS)
    return folder, ["--tables", str(tables), "--truth-run", str(run)], rows


@pytest.fixture
def predictions(monkeypatch):
    """The tables `HazardModel.predict` returns during a run."""
    returned = []
    original = hazard.HazardModel.predict

    def recording(self, frame, kappas):
        table = original(self, frame, kappas)
        returned.append(table)
        return table

    monkeypatch.setattr(hazard.HazardModel, "predict", recording)
    return returned


def _patch(monkeypatch, folder) -> None:
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))


def test_learn_run_scores_the_hazard_model_end_to_end(
    build, tmp_path, monkeypatch, capsys, predictions
):
    folder, arguments, rows = build
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    _patch(monkeypatch, tmp_path)
    code = learn_run.main([*arguments, "--matrix", str(matrix), "--model", "hazard"])
    printed = capsys.readouterr().out
    assert code == 0, printed
    results = json.loads((tmp_path / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == "hazard"
    assert results["fit_units"] == list(TRAIN_UNITS)
    assert not results["failed_configurations"]
    table = read_table(tmp_path / learn_run.TABLE_NAME)
    assert len(table) == rows  # one level, every day counts
    assert sorted(set(table["unit"])) == sorted(GATE_UNITS)
    assert table["d_hat"].notna().all() and (table["survival_h1"] >= 1.0).all()
    assert table[list(SURVIVAL_COLUMNS)[:1]].notna().all().all()
    assert len(predictions) == 1
    pd.testing.assert_frame_equal(table, predictions[0], check_exact=True)
    for unit in GATE_UNITS:
        score = results["units"][unit]["kappas"]["0.1"]
        assert score["cars_with_departure_forecast"] == score["cars"]
        assert score["concordance_index"]["value"] is not None
        for loss in score["pinball_loss"].values():
            assert loss["mean"] is not None and loss["mean"] >= 0
        assert results["units"][unit]["configurations"]["main-k0.1-empirical"]["rungs"]
    assert json.loads((tmp_path / learn_run.TIMING_NAME).read_text())["fit_seconds"] > 0


def test_dial_choice_runs_the_hazard_model_end_to_end(
    build, tmp_path, monkeypatch, capsys, predictions
):
    folder, arguments, _ = build
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(document))
    out = tmp_path / "out"
    out.mkdir()
    _patch(monkeypatch, out)
    code = dial_choice.main(
        ["--model", "hazard", "--matrix", str(matrix), *arguments, "--splits", "4"]
    )
    assert code == 0, capsys.readouterr().out
    results = json.loads((out / dial_choice.RESULTS_NAME).read_text())
    assert results["model"] == "hazard" and results["fit_units"] == list(TRAIN_UNITS)
    assert results["failed_configurations"] == []
    table = read_table(out / dial_choice.TABLE_NAME)
    levels = {0.05, 0.1, 0.2, 0.3}
    assert set(table["kappa"]) == levels
    rows = sum(len(_units()[u]) for u in GATE_UNITS)
    assert len(table) == len(levels) * rows
    assert len(predictions) == 1
    pd.testing.assert_frame_equal(table, predictions[0], check_exact=True)
    for unit in GATE_UNITS[1:]:
        chosen = results["dial_choice"][unit]
        assert set(chosen) == {"A", "B"}
        assert all(chosen[tier]["dial"] in (0.03, 0.1) for tier in chosen)
        for found in results["units"][unit]["configurations"].values():
            assert set(found["tiers"]) == {"A", "B"}


def test_the_committed_learn_matrix_names_the_levels_the_model_is_built_at():
    from types import SimpleNamespace

    matrix = SimpleNamespace(rho=0.0, configurations=[SimpleNamespace(kappa=0.1)])
    model = hazard.make_model(matrix)
    assert model.levels["departure"] == (0.1, 0.5)
    assert model.fit_units == TRAIN_UNITS
    assert learn_run.load_model("hazard", matrix).name == "hazard"
