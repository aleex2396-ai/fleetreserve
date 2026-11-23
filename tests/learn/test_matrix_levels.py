"""The levels a model built from a run matrix fits and forecasts at (`driver_risk.matrix_levels`),
as `learn_run` and every model's `make_model` (`models.lgbm.matrix_kappas`) take them.

The rule: every configuration's own kappa and, for a configuration with a driver dial, every level
up to its kappa (`driver_risk.levels_up_to`), which its driver-risk rows read (`gate_day`).

1. By hand: a matrix of kappa 0.5 only with a dialled configuration needs the six levels 0.05,
   0.1, 0.2, 0.3, 0.4 and 0.5; without the dial, 0.5 alone; a dial at 0.2 beside a plain 0.5 needs
   0.05, 0.1, 0.2 and 0.5. The committed gate and learn matrices need their own kappas, the six
   levels of plan section 2, so models built from them are unchanged.
2. `learn_run --model lgbm_tail` on the committed rehearsal matrix (kappa 0.5 only, configurations
   with the dial 0.02 and without) and made-up units: every configuration is scored, the run ends
   with exit 0, the forecast table holds the six levels and the scores are at the matrix's own
   kappa 0.5 only.
3. On a matrix that already lists the six levels, with a dialled pair at 0.3 (from the committed
   gate matrix), every output of `learn_run` but its timing file (results, forecast table, gate
   results, gate days, day losses) is byte-identical under the rule and under the rule before it
   (the matrix's own kappas).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from gate_fake import COMMIT, gate_cars, matrix_document, write_build
from learn_fake import small_gate_units

from fleetreserve import driver_risk, gate_config, gate_run, learn_run, run_verify
from fleetreserve.forecast_model import read_table
from fleetreserve.truth_build import TRAIN_UNITS

ROOT = Path(__file__).resolve().parents[2]
REHEARSAL = ROOT / "gate" / "rehearsal-matrix" / "lgbm_tail.json"
SIX = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def _c(kappa, dial=None):
    return SimpleNamespace(kappa=kappa, driver_risk=dial)


def test_the_levels_by_hand():
    assert driver_risk.matrix_levels([_c(0.5, 0.02), _c(0.5)]) == tuple(SIX)
    assert driver_risk.matrix_levels([_c(0.5), _c(0.5)]) == (0.5,)
    assert driver_risk.matrix_levels([_c(0.5), _c(0.2, 0.0)]) == (0.05, 0.1, 0.2, 0.5)


@pytest.mark.parametrize("path", ["gate/matrix.json", "learn/matrix.json"])
def test_the_committed_matrices_need_their_own_kappas(path):
    matrix, _ = gate_config.read(ROOT / path)
    own = sorted({c.kappa for c in matrix.configurations})
    assert own == SIX
    assert list(driver_risk.matrix_levels(matrix.configurations)) == own


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    training = {
        unit: gate_cars(24, f"T{j}", early_every=3 + j) for j, unit in enumerate(TRAIN_UNITS)
    }
    run, tables = write_build(folder, {**small_gate_units(), **training})
    return folder, run, tables


def _learn(build, matrix: Path, out: Path, monkeypatch) -> int:
    _, run, tables = build
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    return learn_run.main([*arguments, "--model", "lgbm_tail"])


def test_learn_run_scores_every_configuration_of_a_one_kappa_matrix_with_a_dial(
    build, tmp_path, monkeypatch
):
    out = tmp_path / "out"
    status = _learn(build, REHEARSAL, out, monkeypatch)
    results = json.loads((out / learn_run.RESULTS_NAME).read_text())
    assert results["failed_configurations"] == []
    assert status == 0
    assert sorted(set(read_table(out / learn_run.TABLE_NAME)["kappa"])) == SIX
    for unit, entry in results["units"].items():
        assert sorted(entry["kappas"]) == ["0.5"], unit


def _six_level_matrix(folder: Path) -> Path:
    """The committed gate matrix's main family at the six levels and its pair with the dial 0.02
    at 0.3, each with its oracle partner, with 20 calibration days."""
    names = [f"main-rounded-hour-k{k:g}-{d}" for k in SIX for d in ("empirical", "oracle")]
    dial = [f"main-rounded-hour-k0.3-eps0.02-{d}" for d in ("empirical", "oracle")]
    document = matrix_document((*names, *dial))
    document["n_calibration"] = 20
    path = folder / "six-levels.json"
    path.write_text(json.dumps(document))
    return path


def _outputs(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if "timing" not in p.name)


def test_on_a_matrix_with_the_six_levels_every_output_is_unchanged(build, tmp_path, monkeypatch):
    matrix = _six_level_matrix(tmp_path)
    parsed, _ = gate_config.read(matrix)
    assert sorted({c.kappa for c in parsed.configurations}) == SIX
    assert any(c.driver_risk is not None for c in parsed.configurations)
    assert _learn(build, matrix, tmp_path / "now", monkeypatch) == 0
    with monkeypatch.context() as before:
        before.setattr(
            driver_risk,
            "matrix_levels",
            lambda configurations: tuple(sorted({c.kappa for c in configurations})),
        )
        assert _learn(build, matrix, tmp_path / "before", monkeypatch) == 0
    names = _outputs(tmp_path / "now")
    assert {learn_run.RESULTS_NAME, learn_run.TABLE_NAME, learn_run.GATE_NAME} <= set(names)
    assert names == _outputs(tmp_path / "before")
    for name in names:
        now = (tmp_path / "now" / name).read_bytes()
        assert now == (tmp_path / "before" / name).read_bytes(), name
    assert sorted(set(read_table(tmp_path / "now" / learn_run.TABLE_NAME)["kappa"])) == SIX
