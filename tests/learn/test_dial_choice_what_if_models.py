"""`dial_choice --what-if-kappa <level>` with learned models on the made-up units of
`test_dial_choice_lgbm` (reading W3 of `dial_choice`: the model fitted at every level of the grid
and of the what-if list).

1. `lgbm` at the grid level 0.3: the what-if run's forecast table is byte-identical to the run's
   without the option, and every figure of its units is the same: one booster per quantity and
   level, so the boosters at the grid's levels do not change when more levels are fitted.
2. `lgbm` at 0.4 (a level of the plan's grid of six) and at 0.6 (off the grid): the table holds
   the levels up to 0.3 with the same rows as the run without the option, and the rows at the
   higher levels up to the level read; the run ends with exit 0 under the what-if label, "not a
   run of the plan" at 0.4 and "level not on the plan's grid" at 0.6.
3. `hazard` (default settings, the units of `test_hazard_learn_run`) at 0.5: the run ends with
   exit 0 and its table holds the levels up to 0.5; its rows at the levels up to 0.3 differ from
   the run's without the option in the energy columns, since the energy outputs of all levels
   are trained together in one network (W3), here six levels without the option and eight with.
"""

import hashlib
import json

import pytest
from gate_fake import (
    COMMIT,
    GATE_UNITS,
    PLAN_LEVELS,
    gate_cars,
    matrix_document_six,
    write_build,
)
from learn_fake import small_gate_units

from fleetreserve import dial_choice, gate_run, run_verify
from fleetreserve.forecast_model import read_table
from fleetreserve.truth_build import TRAIN_UNITS

NAMES = (
    *(f"main-rounded-hour-k{k:g}-{d}" for k in PLAN_LEVELS for d in ("empirical", "oracle")),
    *(f"main-rounded-hour-k0.3-eps{e:g}-{d}" for e in (0.03, 0.1) for d in ("empirical", "oracle")),
)
GRID = PLAN_LEVELS
BELOW = (0.05, 0.1, 0.2, 0.3)  # the levels of the run without the option (kappa 0.3)
TABLE = dial_choice.WHAT_IF_NAMES["table"]
RESULTS = dial_choice.WHAT_IF_NAMES["results"]


def _lgbm_units() -> dict:
    training = {
        unit: gate_cars(24, f"T{j}", early_every=3 + j) for j, unit in enumerate(TRAIN_UNITS)
    }
    return {**small_gate_units(), **training}


def _hazard_units() -> dict:
    """The units of `test_hazard_learn_run`: sites the committed site-type mapping knows."""
    units = {
        GATE_UNITS[0]: gate_cars(5, "nrel"),
        GATE_UNITS[1]: gate_cars(24, "nrel", early_every=4),
        GATE_UNITS[2]: gate_cars(12, "OSL_1", early_every=3) + gate_cars(12, "BER", early_every=5),
    }
    for j, unit in enumerate(TRAIN_UNITS):
        units[unit] = gate_cars(24, unit, early_every=3 + j)
    return units


def _runs(folder, model: str, units: dict, levels) -> dict:
    """name -> run folder, for the run without an option and a what-if run at each level."""
    run, tables = write_build(folder, units)
    document = matrix_document_six(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    arguments = ["--model", model, "--matrix", str(matrix), "--tables", str(tables)]
    arguments += ["--truth-run", str(run), "--splits", "4"]
    found = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        options = [("plain", [])]
        options += [(f"what-if-{k:g}", ["--what-if-kappa", str(k)]) for k in levels]
        for name, extra in options:
            out = folder / name
            out.mkdir()
            monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
            assert dial_choice.main([*arguments, *extra]) == 0, name
            found[name] = out
    return found


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return _runs(tmp_path_factory.mktemp("lgbm"), "lgbm", _lgbm_units(), (0.3, 0.4, 0.6))


@pytest.fixture(scope="module")
def hazard_runs(tmp_path_factory):
    return _runs(tmp_path_factory.mktemp("hazard"), "hazard", _hazard_units(), (0.5,))


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_at_0_3_the_what_if_fit_gives_the_same_table_and_figures(runs):
    plain, what_if = runs["plain"], runs["what-if-0.3"]
    assert _sha(plain / dial_choice.TABLE_NAME) == _sha(what_if / TABLE)
    grid = json.loads((plain / dial_choice.RESULTS_NAME).read_text())
    found = json.loads((what_if / RESULTS).read_text())
    assert found["what_if"]["levels_fitted"] == [*GRID, 0.6, 0.7]
    assert found["units"] == grid["units"]
    assert found["dial_choice"] == grid["dial_choice"]


@pytest.mark.parametrize(
    ("kappa", "label"), [(0.4, dial_choice.ON_GRID), (0.6, dial_choice.OFF_GRID)]
)
def test_the_rows_up_to_0_3_are_unchanged_and_the_higher_levels_are_added(runs, kappa, label):
    plain = read_table(runs["plain"] / dial_choice.TABLE_NAME)
    what_if = read_table(runs[f"what-if-{kappa:g}"] / TABLE, what_if=True)
    levels = {*BELOW, *(k for k in (0.4, 0.5, 0.6) if k <= kappa)}
    assert set(what_if["kappa"]) == levels
    unchanged = what_if[what_if["kappa"] <= 0.3].reset_index(drop=True)
    assert unchanged.equals(plain)
    for level in levels - set(BELOW):
        at = what_if[what_if["kappa"] == level]
        assert len(at) == len(plain[plain["kappa"] == 0.3])
        assert at["d_hat"].notna().all()
    found = json.loads((runs[f"what-if-{kappa:g}"] / RESULTS).read_text())
    assert label in found["label"]
    assert found["what_if"]["on_the_plans_grid"] is (kappa in GRID)
    assert found["failed_configurations"] == []


def test_the_hazard_model_runs_at_0_5_with_another_network(hazard_runs):
    plain = read_table(hazard_runs["plain"] / dial_choice.TABLE_NAME)
    what_if = read_table(hazard_runs["what-if-0.5"] / TABLE, what_if=True)
    assert set(what_if["kappa"]) == set(GRID)
    found = json.loads((hazard_runs["what-if-0.5"] / RESULTS).read_text())
    assert found["failed_configurations"] == []
    assert found["what_if"]["driver_risk_levels"] == list(GRID)
    assert found["what_if"]["on_the_plans_grid"] is True
    on_grid = what_if[what_if["kappa"] <= 0.3].reset_index(drop=True)
    keys = ["unit", "site", "date", "session_id", "kappa"]
    assert on_grid[keys].equals(plain[keys])
    energy = ["r_hat", "e_cap", "e_need"]
    assert not on_grid[energy].equals(plain[energy])
