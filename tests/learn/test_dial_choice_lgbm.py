"""`dial_choice --model lgbm` end to end on the made-up gate units of `learn_fake` and eight
made-up training units (one per `truth_build.TRAIN_UNITS`), with a made-up verified truth build and
the dial family of the committed gate matrix at kappa 0.3 (two dials).

The command ends with exit 0 and writes its results: the model fitted on the training units only,
a forecast table with a row for every car of every counting day at each level up to kappa, the
two tiers for every configuration, and a dial that is one of the dials.
"""

import json

from gate_fake import COMMIT, GATE_UNITS, gate_cars, matrix_document, write_build
from learn_fake import small_gate_units

from fleetreserve import dial_choice, gate_run, run_verify
from fleetreserve.forecast_model import read_table
from fleetreserve.truth_build import TRAIN_UNITS

# The model is built from the kappas of the matrix, so the matrix lists every level (as the
# committed one does) although only the configurations at kappa 0.3 are measured.
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


def test_the_model_runs_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    training = {
        unit: gate_cars(24, f"T{j}", early_every=3 + j) for j, unit in enumerate(TRAIN_UNITS)
    }
    run, tables = write_build(tmp_path, {**small_gate_units(), **training})
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(document))
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    code = dial_choice.main(["--model", "lgbm", *arguments, "--splits", "4"])
    assert code == 0, capsys.readouterr().out
    results = json.loads((out / dial_choice.RESULTS_NAME).read_text())
    assert results["model"] == "lgbm" and results["fit_units"] == list(TRAIN_UNITS)
    assert results["failed_configurations"] == []
    table = read_table(out / dial_choice.TABLE_NAME)
    cars = small_gate_units()
    levels = {0.05, 0.1, 0.2, 0.3}
    assert set(table["kappa"]) == levels
    assert len(table) == len(levels) * sum(len(cars[u]) for u in GATE_UNITS)
    for unit in GATE_UNITS[1:]:
        chosen = results["dial_choice"][unit]
        assert set(chosen) == {"A", "B"}
        assert all(chosen[tier]["dial"] in (0.03, 0.1) for tier in chosen)
        for found in results["units"][unit]["configurations"].values():
            assert set(found["tiers"]) == {"A", "B"}
    assert "measured, not fixed" in (out / dial_choice.SUMMARY_NAME).read_text()
