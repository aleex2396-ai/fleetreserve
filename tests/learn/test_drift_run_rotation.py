"""The time-ordered drift-guard command (`drift_run`) with the model its recorded run uses on
training units, `lgbm_tail_rotation`, on made-up tables of the eight training units.

1. On the training unit jpl the model is fitted on the other seven training units, and the run
   ends with exit 0: the results name jpl's role "training", the seven fit units in the order of
   `TRAIN_UNITS` and the rotation what-if record (jpl left out of the fit), and the summary
   carries that record's evidence line.
2. The plain `lgbm_tail`, fitted on every training unit, could not run on jpl (exit 2), and
   nothing is written.
"""

import json

import pytest
from gate_fake import COMMIT, gate_cars, matrix_document, write_build

from fleetreserve import drift_run, gate_run, rotation_whatif, run_verify
from fleetreserve.truth_build import TRAIN_UNITS

# The model is built from the kappas of the matrix, so the matrix lists every level up to 0.3 (as
# the committed one does) although only the dial family at kappa 0.3 is read.
NAMES = (
    "main-rounded-hour-k0.05-empirical",
    "main-rounded-hour-k0.05-oracle",
    "main-rounded-hour-k0.1-empirical",
    "main-rounded-hour-k0.1-oracle",
    "main-rounded-hour-k0.2-empirical",
    "main-rounded-hour-k0.2-oracle",
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.02-empirical",
    "main-rounded-hour-k0.3-eps0.02-oracle",
)
SCORED = "jpl"
DAYS = 24
N_CAL = 10


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("drift-rotation")
    units = {u: gate_cars(DAYS, f"T{j}", early_every=3 + j) for j, u in enumerate(TRAIN_UNITS)}
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return run, tables, matrix


def _command(build, out, model):
    run, tables, matrix = build
    arguments = ["--model", model, "--dial", "0.02", "--matrix", str(matrix)]
    arguments += ["--tables", str(tables), "--truth-run", str(run), "--units", SCORED]
    arguments += ["--calibration-days", str(N_CAL), "--etas", "0.5"]
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return drift_run.main(arguments)


def test_a_training_unit_runs_with_the_model_fitted_on_the_other_seven(build, tmp_path):
    out = tmp_path / "out"
    assert _command(build, out, "lgbm_tail_rotation") == 0
    document = json.loads((out / drift_run.RESULTS_NAME).read_text())
    assert document["failed_units"] == []
    entry = document["units"][SCORED]
    assert entry["role"] == "training"
    assert entry["fit_units"] == [u for u in TRAIN_UNITS if u != SCORED]
    found = entry["rotation_whatif"]
    assert found["scored_training_units"] == [SCORED]
    assert found["left_out_of_the_fit"] == [SCORED]
    assert entry["guarded_days"] == DAYS - N_CAL
    summary = (out / drift_run.SUMMARY_NAME).read_text()
    assert f"{rotation_whatif.WHAT}: {found['evidence']}" in summary.splitlines()


def test_the_plain_model_fitted_on_the_unit_could_not_run(build, tmp_path):
    out = tmp_path / "out"
    assert _command(build, out, "lgbm_tail") == 2
    assert list(out.iterdir()) == []
