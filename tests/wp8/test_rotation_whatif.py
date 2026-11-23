"""A rotation what-if: scoring a training unit with a model not fitted on it (`rotation_whatif`,
`gate_config`, `learn_run.check_model`), on made-up units and a made-up verified truth build.

1. A matrix naming a training unit among its gate units is refused unless it carries
   `rotation_whatif: true`; the field on a matrix with no training gate unit is refused; the field
   must be true or false; the committed matrices do not carry it and load as before.
2. `learn_run.check_model` refuses a model whose `fit_units` meet the gate units (`FitOnScored`,
   a `Refused`), before anything is fitted; a model fitted on the other training units passes;
   `learn_run.evaluate` has the same check.
3. `learn_run` on a training gate unit: a model fitted on another training unit runs (exit 0) and
   every output says that it is a rotation what-if, names the unit scored and the units left out of
   the fit, and gives the evidence sentence; a model fitted on the scored unit ends with exit 2,
   fitted on nothing; a training gate unit without the field ends with exit 2.
4. The same run on a matrix of development units carries no trace of the label in any output.
5. `gate_run` and `resolve_diagnosis` refuse a rotation what-if matrix (exit 2).
"""

import json

import pytest
from gate_fake import COMMIT, GATE_UNITS, TOP, gate_cars, write_build
from learn_fake import small_gate_units, small_matrix
from test_learn_guard import _Honest, _module

from fleetreserve import (
    dial_choice,
    gate_config,
    gate_run,
    learn_run,
    resolve_diagnosis,
    rotation_whatif,
)
from fleetreserve.gate_config import MatrixError
from fleetreserve.truth_build import TRAIN_UNITS

SCORED = "jpl"
OTHER = "caltech"


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    from fleetreserve import run_verify

    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


ABSENT = object()


def rotation_matrix(*, flag=True, units=(SCORED,)) -> dict:
    document = small_matrix()
    document["gate_units"] = list(units)
    document["required_units"] = 1
    if flag is not ABSENT:
        document["rotation_whatif"] = flag
    return document


def parse(document) -> gate_config.Matrix:
    return gate_config.parse(document)


# --- 1. the matrix field --------------------------------------------------------------------


def test_a_training_gate_unit_needs_the_field():
    with pytest.raises(MatrixError, match="training units"):
        parse(rotation_matrix(flag=ABSENT))
    with pytest.raises(MatrixError, match="training units"):
        parse(rotation_matrix(flag=False))
    matrix = parse(rotation_matrix())
    assert matrix.rotation_whatif is True and matrix.gate_units == (SCORED,)


def test_the_field_needs_a_training_gate_unit_and_a_true_or_false_value():
    with pytest.raises(MatrixError, match="no gate unit is a training unit"):
        parse(rotation_matrix(units=GATE_UNITS))
    for value in ("yes", 1, None):
        with pytest.raises(MatrixError, match="true or false"):
            parse(rotation_matrix(flag=value))


def test_a_mixed_matrix_scores_its_training_units_in_the_order_of_the_training_units():
    matrix = parse(rotation_matrix(units=(GATE_UNITS[1], "TRO", SCORED)))
    assert rotation_whatif.scored_units(matrix) == (SCORED, "TRO")


def test_the_committed_matrices_do_not_carry_the_field_and_load_as_before():
    for name in ("gate", "learn"):
        path = TOP / name / "matrix.json"
        assert "rotation_whatif" not in json.loads(path.read_text())
        matrix = gate_config.load(path)
        assert matrix.rotation_whatif is False
        assert not set(matrix.gate_units) & set(TRAIN_UNITS)


# --- 2. the guard -----------------------------------------------------------------------------


def test_a_model_fitted_on_a_gate_unit_is_refused_and_one_fitted_elsewhere_is_not():
    on_scored = _Honest(fit_units=(SCORED, OTHER))
    with pytest.raises(learn_run.FitOnScored, match="scores"):
        learn_run.check_model("honest", on_scored, gate_units=(SCORED,))
    assert issubclass(learn_run.FitOnScored, learn_run.Refused)
    elsewhere = _Honest(fit_units=(OTHER,))
    assert learn_run.check_model("honest", elsewhere, gate_units=(SCORED,)) is False
    # development gate units never overlap the fit units of any model
    assert learn_run.check_model("honest", on_scored, gate_units=GATE_UNITS) is False
    # without the gate units only the older checks run
    assert learn_run.check_model("honest", on_scored) is False


def test_evaluate_refuses_a_model_fitted_on_a_gate_unit_before_fitting(tmp_path):
    matrix = parse(rotation_matrix())
    model = _Honest(fit_units=(SCORED,))
    with pytest.raises(learn_run.FitOnScored):
        learn_run.evaluate("honest", model, matrix, {}, {}, workers=1)
    assert model.fitted == [] and model.given is None


def test_dial_choice_refuses_a_model_fitted_on_a_gate_unit_before_fitting():
    matrix = parse(rotation_matrix())
    model = _Honest(fit_units=(SCORED,))
    with pytest.raises(learn_run.FitOnScored):
        dial_choice.make_factory("honest", model, matrix, {}, {}, 0.3)
    assert model.fitted == [] and model.given is None


# --- 3. and 4. learn_run end to end -----------------------------------------------------------


def build(tmp_path, document):
    units = {
        **small_gate_units(),
        SCORED: gate_cars(24, "J", early_every=4),
        OTHER: gate_cars(8, "T"),
    }
    run, tables = write_build(tmp_path, units)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(document))
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]


def outputs(folder) -> dict:
    return {path.name: path.read_text() for path in sorted(folder.iterdir()) if path.is_file()}


def run_learn(tmp_path, monkeypatch, capsys, document, model):
    _module(monkeypatch, "honest", model)
    arguments = build(tmp_path, document)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    code = learn_run.main([*arguments, "--model", "honest"])
    return code, capsys.readouterr().out, outputs(out)


def test_a_model_fitted_on_another_training_unit_scores_a_training_unit(
    tmp_path, monkeypatch, capsys
):
    model = _Honest(fit_units=(OTHER,))
    code, text, files = run_learn(tmp_path, monkeypatch, capsys, rotation_matrix(), model)
    assert code == 0, text
    assert model.fitted == [[OTHER]]
    results = json.loads(files[learn_run.RESULTS_NAME])
    assert list(results["units"]) == [SCORED] and results["fit_units"] == [OTHER]
    left_out = [u for u in TRAIN_UNITS if u != OTHER]
    for name in (learn_run.RESULTS_NAME, learn_run.GATE_NAME):
        found = json.loads(files[name])["rotation_whatif"]
        assert found["what"] == rotation_whatif.WHAT
        assert found["scored_training_units"] == [SCORED]
        assert found["left_out_of_the_fit"] == left_out and found["fit_units"] == [OTHER]
        assert "development evidence, not held-out evidence" in found["evidence"]
        assert "plan section 2" in found["evidence"] and SCORED in found["evidence"]
    first = text.splitlines()[:2]
    assert first[0] == rotation_whatif.WHAT
    assert first[1] == results["rotation_whatif"]["evidence"]
    assert files[learn_run.TABLE_NAME]  # the forecast table is data and holds no label


def test_a_run_of_development_units_carries_no_trace_of_the_label(tmp_path, monkeypatch, capsys):
    model = _Honest(fit_units=(OTHER,))
    code, text, files = run_learn(tmp_path, monkeypatch, capsys, small_matrix(), model)
    assert code == 0, text
    for content in (text, *files.values()):
        assert "rotation" not in content.lower() and "what-if" not in content.lower()


def test_a_model_fitted_on_the_scored_unit_could_not_run_and_is_never_fitted(
    tmp_path, monkeypatch, capsys
):
    model = _Honest(fit_units=(SCORED,))
    code, text, files = run_learn(tmp_path, monkeypatch, capsys, rotation_matrix(), model)
    assert code == 2 and text.startswith("COULD NOT RUN") and "scores" in text
    assert model.fitted == [] and model.given is None and files == {}


def test_a_training_gate_unit_without_the_field_could_not_run(tmp_path, monkeypatch, capsys):
    model = _Honest(fit_units=(OTHER,))
    code, text, files = run_learn(
        tmp_path, monkeypatch, capsys, rotation_matrix(flag=ABSENT), model
    )
    assert code == 2 and text.startswith("COULD NOT RUN") and "rotation_whatif" in text
    assert model.fitted == [] and files == {}


# --- 5. the other commands ---------------------------------------------------------------------


def test_gate_run_and_resolve_diagnosis_refuse_a_rotation_matrix(tmp_path, capsys):
    arguments = build(tmp_path, rotation_matrix())
    assert gate_run.main(arguments) == 2
    assert "rotation_whatif" in capsys.readouterr().out
    configuration = rotation_matrix()["configurations"][0]["name"]
    code = resolve_diagnosis.main(
        [*arguments, "--model", "empirical", "--configuration", configuration, "--scale", "1"]
    )
    assert code == 2 and "rotation_whatif" in capsys.readouterr().out
