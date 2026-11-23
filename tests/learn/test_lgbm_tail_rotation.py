"""`models.lgbm_tail_rotation`: `lgbm_tail` fitted on the training units the run does not score.

Made-up cars tables of three training units (caltech, jpl, office001; one site and its own users
each) and, for the end-to-end runs, a made-up truth build with all eight training units.

1. `fit_units_for` is `TRAIN_UNITS` minus the gate units, in the order of `TRAIN_UNITS`; no unit
   left is refused; `make_model(matrix)` reads the matrix's gate units.
2. Rows of a left-out unit never reach the fit: altering them leaves every forecast of the other
   units' days unchanged, and a frame without them gives the same forecasts; the plain `lgbm_tail`
   changes with them.
3. For a matrix with no training gate unit the model is `lgbm_tail`: the same forecast table to
   the byte, and `dial_choice` writes the same forecast file for both.
4. `learn_run.check_model` refuses `lgbm_tail` (fitted on every training unit) for a matrix that
   scores a training unit and passes the rotation model.
5. `dial_choice --model lgbm_tail_rotation` runs end to end (exit 0) on a made-up training unit as
   the gate unit, fitted on the other seven; its outputs say it is a rotation what-if; with
   `lgbm_tail` the same matrix ends with exit 2; a run of the plan's kind carries no label.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from gate_fake import COMMIT, TOP, gate_cars, matrix_document, write_build
from learn_fake import small_gate_units
from test_dial_choice_lgbm import NAMES
from test_learn_features import _cars
from truth_tables_fake import FakeCar, date_of

from fleetreserve import dial_choice, gate_config, gate_run, learn_run, rotation_whatif, run_verify
from fleetreserve.forecast_model import write_table
from fleetreserve.learn_features import build_table, feature_frame
from fleetreserve.models import lgbm_tail, lgbm_tail_rotation
from fleetreserve.truth_build import TRAIN_UNITS

KAPPAS = (0.1, 0.3)
UNITS = ("caltech", "jpl", "office001")
SCORED = "jpl"
DAYS = 40


def _unit(name: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fakes, users = [], []
    for k in range(DAYS):
        for j in range(6):
            quick = j % 2 == 0 and rng.random() < 0.3
            stay = 0.5 + 0.5 * rng.random() if quick else 5.0 + 3.0 * rng.random()
            fakes.append(
                FakeCar(
                    name,
                    date_of(k),
                    7.0 + 0.2 * j,
                    10.0 + stay,
                    rate=3.0 + rng.random(),
                    room=1.0 + 2.0 * rng.random(),
                    session=f"{name}-{k}-{j}",
                    energy=4.0,
                )
            )
            users.append(f"{name}-user{j}")
    frame = _cars(fakes, users)
    frame["unit"] = name
    return frame


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.concat([_unit(name, 10 + i) for i, name in enumerate(UNITS)], ignore_index=True)


def altered(cars: pd.DataFrame, unit: str) -> pd.DataFrame:
    """`cars` with the outcomes of `unit`'s rows changed (departure, room and rate)."""
    out = cars.copy()
    rows = out["unit"] == unit
    out.loc[rows, "departure_exact"] = out.loc[rows, "departure_exact"] * 0.25 + 0.1
    out.loc[rows, "room"] = out.loc[rows, "room"] * 3.0
    out.loc[rows, "rate"] = out.loc[rows, "rate"] * 0.5
    return out


def forecast_rows(cars: pd.DataFrame, units) -> pd.DataFrame:
    """The features of the counting rows of `units`, built from the unaltered cars."""
    table = build_table(cars, rho=0.0)
    table = table[table["unit"].isin(units) & table["counting"]].reset_index(drop=True)
    return feature_frame(table)


def rotation(gate_units, frame_fitted: pd.DataFrame):
    model = lgbm_tail_rotation.LgbmTailRotationModel(
        rho=0.0, kappas=KAPPAS, fit_units=lgbm_tail_rotation.fit_units_for(gate_units)
    )
    model.fit(frame_fitted)
    return model


# --- 1. the fit units -------------------------------------------------------------------------


def test_the_fit_units_are_the_training_units_minus_the_gate_units_in_order():
    for gate in (("jpl",), ("BOD", "caltech"), ("OSL_S", "ASK", "TRO", "BAR", "BOD")):
        found = lgbm_tail_rotation.fit_units_for(gate)
        assert found == tuple(u for u in TRAIN_UNITS if u not in gate)
        assert not set(found) & set(gate) and len(found) == len(TRAIN_UNITS) - len(gate)
    assert lgbm_tail_rotation.fit_units_for(("nrel-from-may-2019", "OSL_1")) == TRAIN_UNITS
    assert lgbm_tail_rotation.fit_units_for(("jpl",)) == (
        "caltech",
        "office001",
        "OSL_S",
        "ASK",
        "TRO",
        "BAR",
        "BOD",
    )


def test_a_matrix_that_scores_every_training_unit_is_refused():
    with pytest.raises(gate_config.MatrixError, match="no training unit is left"):
        lgbm_tail_rotation.fit_units_for(TRAIN_UNITS)
    with pytest.raises(ValueError, match="distinct, non-empty training units"):
        lgbm_tail_rotation.LgbmTailRotationModel(rho=0.0, kappas=KAPPAS, fit_units=())
    with pytest.raises(ValueError, match="distinct, non-empty training units"):
        lgbm_tail_rotation.LgbmTailRotationModel(rho=0.0, kappas=KAPPAS, fit_units=("OSL_1",))


def test_make_model_reads_the_gate_units_of_the_matrix():
    document = matrix_document(NAMES)
    committed = gate_config.parse(document)
    model = lgbm_tail_rotation.make_model(committed)
    assert model.name == "lgbm_tail_rotation" and model.fit_units == TRAIN_UNITS
    document["gate_units"] = [SCORED, "BOD"]
    document["rotation_whatif"] = True
    model = lgbm_tail_rotation.make_model(gate_config.parse(document))
    assert model.fit_units == tuple(u for u in TRAIN_UNITS if u not in (SCORED, "BOD"))
    assert model.levels == lgbm_tail.make_model(committed).levels


# --- 2. the rows of a left-out unit ------------------------------------------------------------


def test_the_rows_of_a_left_out_unit_never_reach_the_fit(frame):
    others = tuple(u for u in UNITS if u != SCORED)
    fresh = forecast_rows(frame, others)
    base = rotation((SCORED,), frame).predict(fresh, KAPPAS)
    changed = rotation((SCORED,), altered(frame, SCORED)).predict(fresh, KAPPAS)
    pd.testing.assert_frame_equal(base, changed, check_exact=True)
    without = rotation((SCORED,), frame[frame["unit"] != SCORED].reset_index(drop=True))
    pd.testing.assert_frame_equal(base, without.predict(fresh, KAPPAS), check_exact=True)
    # the plain model is fitted on the rows, so the same alteration moves its forecasts
    plain, plain_changed = (lgbm_tail.LgbmTailModel(rho=0.0, kappas=KAPPAS) for _ in range(2))
    plain.fit(frame)
    plain_changed.fit(altered(frame, SCORED))
    first, second = plain.predict(fresh, KAPPAS), plain_changed.predict(fresh, KAPPAS)
    assert (first["d_hat"].to_numpy() != second["d_hat"].to_numpy()).any()
    # and a unit that is fitted on does move the rotation model's forecasts
    moved = rotation((SCORED,), altered(frame, "caltech")).predict(fresh, KAPPAS)
    assert (base["d_hat"].to_numpy() != moved["d_hat"].to_numpy()).any()


def test_a_frame_with_no_row_of_the_fit_units_is_refused(frame):
    model = lgbm_tail_rotation.LgbmTailRotationModel(rho=0.0, kappas=KAPPAS, fit_units=("BOD",))
    with pytest.raises(ValueError, match="no row of the fit units"):
        model.fit(frame)


# --- 3. the plan's matrices --------------------------------------------------------------------


def test_with_no_training_gate_unit_the_forecast_table_is_that_of_lgbm_tail(frame, tmp_path):
    committed = gate_config.load(TOP / "learn" / "matrix.json")
    assert not set(committed.gate_units) & set(TRAIN_UNITS)
    plain = lgbm_tail.make_model(committed)
    rotated = lgbm_tail_rotation.make_model(committed)
    assert rotated.fit_units == plain.fit_units == TRAIN_UNITS
    plain.fit(frame)
    rotated.fit(frame)
    fresh = forecast_rows(frame, UNITS)
    for name, model in (("plain", plain), ("rotated", rotated)):
        write_table(model.predict(fresh, committed_kappas(committed)), tmp_path / f"{name}.csv")
    assert (tmp_path / "plain.csv").read_bytes() == (tmp_path / "rotated.csv").read_bytes()


def committed_kappas(matrix):
    return sorted({c.kappa for c in matrix.configurations})


# --- 4. the guard -------------------------------------------------------------------------------


def test_the_guard_refuses_lgbm_tail_and_passes_the_rotation_model_for_a_training_gate_unit():
    document = matrix_document(NAMES)
    document["gate_units"] = [SCORED]
    document["required_units"] = 1
    document["rotation_whatif"] = True
    matrix = gate_config.parse(document)
    with pytest.raises(learn_run.FitOnScored):
        learn_run.check_model(
            "lgbm_tail", lgbm_tail.make_model(matrix), gate_units=matrix.gate_units
        )
    model = lgbm_tail_rotation.make_model(matrix)
    assert learn_run.check_model(model.name, model, gate_units=matrix.gate_units) is False


# --- 5. dial_choice end to end -------------------------------------------------------------------


@pytest.fixture
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


def build(tmp_path, document):
    training = {
        unit: gate_cars(24, f"T{j}", early_every=3 + j) for j, unit in enumerate(TRAIN_UNITS)
    }
    units = {**small_gate_units(), **training}
    run, tables = write_build(tmp_path, units)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(document))
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]


def rotation_document() -> dict:
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    document["gate_units"] = [SCORED]
    document["required_units"] = 1
    document["rotation_whatif"] = True
    return document


def run_dial_choice(arguments, model, folder, monkeypatch, capsys):
    folder.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
    code = dial_choice.main(["--model", model, *arguments, "--splits", "2"])
    return code, capsys.readouterr().out


def test_dial_choice_scores_a_training_unit_with_a_model_not_fitted_on_it(
    verified, tmp_path, monkeypatch, capsys
):
    arguments = build(tmp_path, rotation_document())
    out = tmp_path / "rotation"
    code, text = run_dial_choice(arguments, "lgbm_tail_rotation", out, monkeypatch, capsys)
    assert code == 0, text
    results = json.loads((out / dial_choice.RESULTS_NAME).read_text())
    left_in = [u for u in TRAIN_UNITS if u != SCORED]
    assert results["model"] == "lgbm_tail_rotation" and results["fit_units"] == left_in
    assert list(results["units"]) == [SCORED] and results["failed_configurations"] == []
    found = results["rotation_whatif"]
    assert found["scored_training_units"] == [SCORED] and found["left_out_of_the_fit"] == [SCORED]
    assert results["label"].startswith(rotation_whatif.WHAT)
    assert results["definitions"]["rotation_whatif"] == found["evidence"]
    assert "development evidence, not held-out evidence" in found["evidence"]
    summary = (out / dial_choice.SUMMARY_NAME).read_text()
    assert summary.splitlines()[:2] == [rotation_whatif.WHAT, found["evidence"]]
    assert text.splitlines()[:2] == summary.splitlines()[:2]
    table = pd.read_csv(out / dial_choice.TABLE_NAME)
    assert len(table) == 4 * 24 * 3  # four levels, 24 days of three cars: the scored unit only


def test_the_plain_model_fitted_on_the_scored_unit_could_not_run(
    verified, tmp_path, monkeypatch, capsys
):
    arguments = build(tmp_path, rotation_document())
    out = tmp_path / "plain"
    code, text = run_dial_choice(arguments, "lgbm_tail", out, monkeypatch, capsys)
    assert code == 2 and text.startswith("COULD NOT RUN") and "scores" in text
    assert list(out.iterdir()) == []


def test_a_run_of_development_units_is_the_plain_model_without_a_label(
    verified, tmp_path, monkeypatch, capsys
):
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    arguments = build(tmp_path, document)
    written = {}
    for model in ("lgbm_tail", "lgbm_tail_rotation"):
        out = tmp_path / model
        code, text = run_dial_choice(arguments, model, out, monkeypatch, capsys)
        assert code == 0, text
        files = {p.name: p.read_text() for p in out.iterdir()}
        for content in (text, *files.values()):
            assert "rotation_whatif" not in content and "what-if" not in content.lower()
        written[model] = hashlib.sha256((out / dial_choice.TABLE_NAME).read_bytes()).hexdigest()
        if model == "lgbm_tail_rotation":
            assert json.loads(files[dial_choice.RESULTS_NAME])["fit_units"] == list(TRAIN_UNITS)
    assert written["lgbm_tail"] == written["lgbm_tail_rotation"]
