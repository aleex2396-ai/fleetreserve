"""Each earlier forecast gets its own kappa and dial (plan section 7; `dial_choice`, module text,
"The earlier forecasts", readings M1 and M2), on the made-up units of `test_dial_choice_lgbm` and
the dial family of the committed gate matrix (two dials), whose matrix lists the six levels.

1. For `earlier_gbm` and `earlier_population`, `dial_choice --kappa <level>` at each of the six
   levels of plan section 2 ends with exit 0, names the model and the training units it was
   fitted on, and its forecast table is that model's own: byte for byte the table of the model
   built from the matrix, fitted on the training units' cars and predicted on the gate units'
   forecast frame at the levels up to the run's kappa; and it is not the main model's table
   (`lgbm_tail`, its run at 0.3).
2. `kappa_choice` over one earlier forecast's six runs ends with exit 0 and names that model, and
   each pair's figures are those of that model's run at the pair's kappa. At kappa 0.3 the figures
   of each earlier forecast differ from the main model's in at least one cell: the pairs the
   readings choose from are its own.
3. Runs of two models are never read together: five runs of an earlier forecast with the main
   model's run at 0.3 end `kappa_choice` with exit 2.
4. Reading M1: the forecast at a level does not depend on the levels fitted: the rows up to 0.3
   of the run at 0.5 are those of the run at 0.3.
5. Reading M2: the fit options of the neural hazard family end the command with exit 2.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from gate_fake import COMMIT, PLAN_LEVELS, gate_cars, matrix_document_six, write_build
from learn_fake import small_gate_units

from fleetreserve import (
    dial_choice,
    driver_risk,
    gate_config,
    gate_run,
    kappa_choice,
    learn_run,
    model_inputs,
    run_verify,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import read_table, write_table
from fleetreserve.truth_build import TRAIN_UNITS

EARLIER = ("earlier_gbm", "earlier_population")
MAIN = "lgbm_tail"
NAMES = (
    *(f"main-rounded-hour-k{k:g}-{d}" for k in PLAN_LEVELS for d in ("empirical", "oracle")),
    *(f"main-rounded-hour-k0.3-eps{e:g}-{d}" for e in (0.03, 0.1) for d in ("empirical", "oracle")),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    training = {
        unit: gate_cars(24, f"T{j}", early_every=3 + j) for j, unit in enumerate(TRAIN_UNITS)
    }
    run, tables = write_build(folder, {**small_gate_units(), **training})
    document = matrix_document_six(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix, folder


def _argv(build, model: str, *extra) -> list[str]:
    run, tables, matrix, _ = build
    argv = ["--model", model, "--matrix", str(matrix), "--tables", str(tables)]
    return [*argv, "--truth-run", str(run), "--splits", "4", *extra]


def _recorded(folder: Path) -> Path:
    """The run folder with a made-up manifest, as `run_record` writes it; its results file."""
    outputs = {p.name: _sha(p) for p in sorted(folder.iterdir())}
    manifest = {"commit": COMMIT, "command": ["python", "-m", "fleetreserve.dial_choice"]}
    manifest |= {"start": "s", "end": "e", "exit_code": 0, "outputs": outputs}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder / dial_choice.RESULTS_NAME


@pytest.fixture(scope="module")
def runs(build):
    """(model, kappa) -> run folder: the six levels of each earlier forecast, the main model's
    run at 0.3."""
    wanted = [(m, k) for m in EARLIER for k in PLAN_LEVELS] + [(MAIN, 0.3)]
    found = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        for model, kappa in wanted:
            out = build[3] / "runs" / f"{model}-k{kappa:g}"
            out.mkdir(parents=True)
            monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
            code = dial_choice.main(_argv(build, model, "--kappa", str(kappa)))
            assert code == 0, (model, kappa)
            _recorded(out)
            found[(model, kappa)] = out
    return found


def _results(folder: Path) -> dict:
    return json.loads((folder / dial_choice.RESULTS_NAME).read_text())


def _own_table(build, model: str, kappa: float, folder: Path) -> Path:
    """The model's table, made apart from the command: built from the matrix, fitted on the
    training units' cars, predicted on the gate units' forecast frame (written as the command
    writes it)."""
    _, tables, matrix_path, _ = build
    matrix = gate_config.load(matrix_path)
    built = learn_run.load_model(model, matrix)
    loaded = {
        name: load_unit_days(
            tables, name, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        for name in (*matrix.gate_units, *TRAIN_UNITS)
    }
    built.fit(pd.concat([loaded[u].cars for u in TRAIN_UNITS], ignore_index=True))
    units = {name: loaded[name] for name in matrix.gate_units}
    frame = learn_run.forecast_frame(False, units, matrix.gate_units, model_inputs.extra_of(built))
    path = folder / f"own-{model}-{kappa:g}.csv"
    write_table(built.predict(frame, list(driver_risk.levels_up_to(kappa))), path)
    return path


@pytest.mark.parametrize("model", EARLIER)
@pytest.mark.parametrize("kappa", PLAN_LEVELS)
def test_each_run_uses_the_earlier_forecast_itself(build, runs, model, kappa, tmp_path):
    folder = runs[(model, kappa)]
    found = _results(folder)
    assert found["model"] == model
    assert found["fit_units"] == list(TRAIN_UNITS)
    assert found["failed_configurations"] == []
    assert found["run"]["kappa"] == kappa
    table = folder / dial_choice.TABLE_NAME
    assert _sha(table) == found["forecast_table"]["sha256"]
    assert _sha(table) == _sha(_own_table(build, model, kappa, tmp_path))
    if kappa == 0.3:
        assert _sha(table) != _sha(runs[(MAIN, 0.3)] / dial_choice.TABLE_NAME)


@pytest.mark.parametrize("model", EARLIER)
def test_kappa_choice_reads_the_model_s_own_six_runs(runs, model, tmp_path, monkeypatch):
    order = (0.2, 0.5, 0.3, 0.05, 0.4, 0.1)
    files = [str(runs[(model, k)] / dial_choice.RESULTS_NAME) for k in order]
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    assert kappa_choice.main(files) == 0
    joint = json.loads((tmp_path / kappa_choice.RESULTS_NAME).read_text())
    assert joint["run"]["model"] == model
    assert [r["kappa"] for r in joint["runs"]] == list(PLAN_LEVELS)
    differs = 0
    main = _results(runs[(MAIN, 0.3)])
    for kappa in PLAN_LEVELS:
        document = _results(runs[(model, kappa)])
        for unit, entry in joint["units"].items():
            configurations = document["units"][unit].get("configurations", {})
            for name, own in configurations.items():
                key = kappa_choice.key(own["driver_risk"])
                for tier in ("A", "B"):
                    cell = entry["pairs"][tier][f"{kappa:g}"][key]
                    for field in kappa_choice.CELL_FIELDS:
                        assert cell[field] == own["tiers"][tier][field], (unit, kappa, key, tier)
                    if kappa == 0.3:
                        other = main["units"][unit]["configurations"][name]["tiers"][tier]
                        differs += any(cell[f] != other[f] for f in kappa_choice.CELL_FIELDS)
    assert differs > 0
    for site_type, unit in kappa_choice.SITE_TYPE_UNITS.items():
        assert joint["readings"]["R1"][site_type]["unit"] == unit


@pytest.mark.parametrize("model", EARLIER)
def test_runs_of_two_models_are_not_read_together(runs, model, tmp_path, monkeypatch, capsys):
    files = [str(runs[(model, k)] / dial_choice.RESULTS_NAME) for k in PLAN_LEVELS if k != 0.3]
    files.append(str(runs[(MAIN, 0.3)] / dial_choice.RESULTS_NAME))
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    assert kappa_choice.main(files) == 2
    assert "the runs differ in model" in capsys.readouterr().out
    assert not (tmp_path / kappa_choice.RESULTS_NAME).exists()


@pytest.mark.parametrize("model", EARLIER)
def test_the_forecast_at_a_level_does_not_depend_on_the_levels_fitted(runs, model):
    at_03 = read_table(runs[(model, 0.3)] / dial_choice.TABLE_NAME)
    at_05 = read_table(runs[(model, 0.5)] / dial_choice.TABLE_NAME)
    assert set(at_05["kappa"]) == set(PLAN_LEVELS)
    below = at_05[at_05["kappa"] <= 0.3].reset_index(drop=True)
    assert below.equals(at_03)


@pytest.mark.parametrize("model", EARLIER)
@pytest.mark.parametrize("option", [("--what-if-fit-seed", "1"), ("--what-if-fit-seeds", "1,2")])
def test_the_neural_fit_options_could_not_run(build, model, option, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    assert dial_choice.main(_argv(build, model, *option)) == 2
    text = capsys.readouterr().out
    assert "COULD NOT RUN" in text and "neural hazard family" in text
    assert not (tmp_path / dial_choice.WHAT_IF_NAMES["results"]).exists()
