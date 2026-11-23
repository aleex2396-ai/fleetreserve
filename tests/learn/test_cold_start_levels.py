"""The cold start (`cold_start.main`) fits each model at every level it forecasts at.

`cold_start.forecast_at` predicts at the levels up to the configuration's kappa
(`driver_risk.levels_up_to`: at 0.5 the six levels 0.05 to 0.5). A model's `make_model` takes its
levels from the matrix's configurations (`models.lgbm.matrix_kappas`), and the rehearsal and sealed
matrices of the run list list one kappa only, the chosen one; by the rule of
`driver_risk.matrix_levels` a dialled configuration also brings every level up to its kappa.

Made-up build: `nrel-from-may-2019` with 60 counting days (three cars a day, `gate_fake.gate_cars`),
so four scored days after the first 56, and each training unit with 24 days. The models are the
real `lgbm_tail` (cold-started) and `lgbm` (the pooled reference).

1. With the committed rehearsal matrix (`gate/rehearsal-matrix/lgbm_tail.json`, kappa 0.5 only)
   and with the same matrix at kappa 0.2, the run ends with exit 0, nothing failed, every model
   and k has figures, and every forecast table holds exactly the levels up to kappa.
"""

import json
from pathlib import Path

import pytest
from gate_fake import COMMIT, gate_cars, write_build

from fleetreserve import (
    cold_start,
    cold_start_days,
    cold_start_gate,
    driver_risk,
    gate_run,
    run_verify,
)
from fleetreserve.forecast_model import read_table
from fleetreserve.truth_build import TRAIN_UNITS

ROOT = Path(__file__).resolve().parents[2]
REHEARSAL = ROOT / "gate" / "rehearsal-matrix" / "lgbm_tail.json"
UNIT = "nrel-from-may-2019"
MODELS = ("lgbm_tail", "lgbm")


def _at(document: dict, kappa: float) -> dict:
    """`document` with every configuration at `kappa`, its names and partners renamed to match."""
    text = json.dumps(document["configurations"]).replace("-k0.5-", f"-k{kappa:g}-")
    configurations = json.loads(text)
    for c in configurations:
        c["kappa"] = kappa
    return {**document, "configurations": configurations}


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    training = {
        unit: gate_cars(24, f"T{j}", early_every=3 + j) for j, unit in enumerate(TRAIN_UNITS)
    }
    run, tables = write_build(folder, {UNIT: gate_cars(60), **training})
    return folder, run, tables


@pytest.mark.parametrize("kappa", [0.5, 0.2])
def test_the_cold_start_forecasts_at_every_level_up_to_kappa(build, kappa, tmp_path, monkeypatch):
    folder, run, tables = build
    rehearsal = json.loads(REHEARSAL.read_text())
    assert sorted({c["kappa"] for c in rehearsal["configurations"]}) == [0.5]
    matrix = folder / f"matrix-{kappa:g}.json"
    matrix.write_text(json.dumps(rehearsal if kappa == 0.5 else _at(rehearsal, kappa)))
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setattr(cold_start_gate, "require_gate", lambda root=None: pytest.fail("gate"))
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    status = cold_start.main(
        [
            *("--model", MODELS[0], "--reference", MODELS[1], "--dial", "0.02"),
            *("--units", UNIT, "--matrix", str(matrix)),
            *("--tables", str(tables), "--truth-run", str(run)),
        ]
    )
    document = json.loads((out / cold_start.RESULTS_NAME).read_text())
    assert document["failed"] == []
    assert status == 0
    levels = list(driver_risk.levels_up_to(kappa))
    assert document["run"]["levels"] == levels
    for model in MODELS:
        for k in cold_start_days.LOCAL_DAYS:
            assert set(document["figures"][model][UNIT][str(k)]) == {"A", "B"}
            table = read_table(out / cold_start.table_name(model, k))
            assert sorted(set(table["kappa"])) == levels
