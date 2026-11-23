"""The leak guard of the shared evaluation path (`learn_run`), on the made-up gate units of
`learn_fake` plus a made-up training unit, with a made-up verified truth build.

0. `fit_units` is read once; `evaluate` refuses tables of units outside that reading.
1. A learned model's `predict` is given the counting days' features only: no column that is not
   known at τ (`learn_features.NOT_KNOWN_AT_TAU`) and no label; one row per car plugged in at τ
   on a counting day. An honest learned model fitted on a training unit runs to exit 0, and its
   `fit` sees that unit only.
2. A model that reads an outcome column in `predict` (a departure, the session energy, the room,
   the fleet marker, the hindsight rate) fails with exit 1, naming the column.
3. A model whose `fit_units` name a development unit, a gate unit or any unit that is not a
   training unit is refused with exit 1 before anything is fitted or read.
4. A model that declares `reads_other_day_outcomes` without being named in
   `forecast_model.OUTCOME_READERS` is refused with exit 1.
5. A model whose table misses a car fails (1), after being fitted on its declared units.
"""

import json
import math
import sys
import types

import pandas as pd
import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_cars, write_build
from learn_fake import small_gate_units, small_matrix

from fleetreserve import gate_run, learn_features, learn_run, run_verify
from fleetreserve.forecast_model import KEY_COLUMNS, TABLE_COLUMNS, empty_table

TRAINING = "caltech"


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {**small_gate_units(), TRAINING: gate_cars(8, "T")}
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]


def _module(monkeypatch, name, model):
    module = types.ModuleType(f"fleetreserve.models.{name}")
    module.make_model = lambda matrix: model
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _rows(frame: pd.DataFrame, kappas, d_hat=math.nan) -> pd.DataFrame:
    """A forecast table with one row per car of `frame` and level, departure `d_hat`."""
    rows = []
    for key in frame[list(KEY_COLUMNS)].itertuples(index=False):
        for kappa in kappas:
            row = dict.fromkeys(TABLE_COLUMNS, math.nan)
            row.update(zip(KEY_COLUMNS, key, strict=True))
            row.update(kappa=float(kappa), rho=0.0, d_hat=d_hat)
            rows.append(row)
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS)) if rows else empty_table()


class _Honest:
    """A learned model: a constant departure of one hour for every car it is given."""

    name = "honest"

    def __init__(self, fit_units=(TRAINING,)):
        self.fit_units = fit_units
        self.fitted = []
        self.given = None

    def fit(self, frame):
        self.fitted.append(sorted(set(frame["unit"])))

    def predict(self, frame, kappas):
        self.given = frame
        return _rows(frame, kappas, d_hat=1.0)


def test_a_learned_model_is_given_features_only(build, monkeypatch, capsys):
    model = _Honest()
    _module(monkeypatch, "honest", model)
    assert learn_run.main([*build, "--model", "honest"]) == 0, capsys.readouterr().out
    assert model.fitted == [[TRAINING]]
    given = model.given
    hidden = set(learn_features.NOT_KNOWN_AT_TAU) | set(learn_features.LABEL_COLUMNS)
    assert not set(given.columns) & hidden
    assert set(learn_features.FEATURE_COLUMNS) <= set(given.columns)
    assert given["counting"].all()
    cars = small_gate_units()
    plugged_in = sum(len(cars[u]) for u in GATE_UNITS)  # every made-up day counts
    assert len(given) == plugged_in
    assert sorted(set(given["unit"])) == sorted(GATE_UNITS)


class _Cheat:
    """A learned model that reads an outcome column of the cars it forecasts."""

    name = "cheat"
    fit_units = ()

    def __init__(self, column):
        self.column = column

    def fit(self, frame):
        pass

    def predict(self, frame, kappas):
        frame[self.column].to_numpy()
        return _rows(frame, kappas)


@pytest.mark.parametrize(
    "column", ["departure_exact", "plug_out", "energy_kwh", "room", "in_fleet", "rate_kw"]
)
def test_a_model_reading_an_outcome_fails(build, monkeypatch, capsys, column):
    _module(monkeypatch, "cheat", _Cheat(column))
    assert learn_run.main([*build, "--model", "cheat"]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "predict raised KeyError" in out and column in out


@pytest.mark.parametrize(
    "units", [(GATE_UNITS[1],), (TRAINING, "OSL_1"), ("OSL_T",), ("no-such-unit",)]
)
def test_a_model_fitted_outside_the_training_units_is_refused(build, monkeypatch, capsys, units):
    model = _Honest(fit_units=units)
    _module(monkeypatch, "honest", model)
    assert learn_run.main([*build, "--model", "honest"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("REFUSED") and "not training units" in out
    assert model.fitted == [] and model.given is None


def test_reading_other_days_outcomes_needs_to_be_listed(build, monkeypatch, capsys):
    model = _Honest(fit_units=())
    model.reads_other_day_outcomes = True
    _module(monkeypatch, "sneaky", model)
    assert learn_run.main([*build, "--model", "sneaky"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("REFUSED") and "reads_other_day_outcomes" in out
    assert model.given is None


class _Dropping(_Honest):
    """A learned model, fitted on a training unit, that leaves one car out of its table."""

    name = "dropping"

    def predict(self, frame, kappas):
        return super().predict(frame, kappas).iloc[len(kappas) :]


def test_a_model_missing_a_car_fails_after_fitting_on_its_units(build, monkeypatch, capsys):
    model = _Dropping()
    _module(monkeypatch, "dropping", model)
    assert learn_run.main([*build, "--model", "dropping"]) == 1
    assert model.fitted == [[TRAINING]]
    assert "no row" in capsys.readouterr().out


class _FlagInFit(_Honest):
    """A model not on the allow-list that sets the outcome-reader flag inside `fit`, after the
    check, hoping to be given the cars tables with the true departures in `predict`."""

    name = "flagfit"

    def fit(self, frame):
        super().fit(frame)
        self.reads_other_day_outcomes = True


def test_a_flag_set_inside_fit_does_not_change_what_predict_is_given(build, monkeypatch, capsys):
    model = _FlagInFit()
    _module(monkeypatch, "flagfit", model)
    assert learn_run.main([*build, "--model", "flagfit"]) == 0, capsys.readouterr().out
    given = model.given
    hidden = set(learn_features.NOT_KNOWN_AT_TAU) | set(learn_features.LABEL_COLUMNS)
    assert not set(given.columns) & hidden
    assert given["counting"].all()


def test_evaluate_itself_refuses_without_the_caller_checking(build, monkeypatch, tmp_path):
    from fleetreserve.gate_config import load

    matrix_file = tmp_path / "matrix.json"
    matrix_file.write_text(json.dumps(small_matrix()))
    matrix = load(matrix_file)
    for model in (_Honest(fit_units=("OSL_1",)), _Honest(fit_units=())):
        if not model.fit_units:
            model.reads_other_day_outcomes = True
        with pytest.raises(learn_run.Refused):
            learn_run.evaluate("unlisted", model, matrix, {}, {}, workers=1)
        assert model.fitted == [] and model.given is None


class _ChangingFitUnits(_Honest):
    """A model whose `fit_units` names a training unit on the first read and a scored gate unit on
    every later read, hoping to be fitted on the scored unit."""

    name = "changing"

    def __init__(self):
        super().__init__()
        self.reads = 0

    @property
    def fit_units(self):
        self.reads += 1
        return (TRAINING,) if self.reads == 1 else (GATE_UNITS[0],)

    @fit_units.setter
    def fit_units(self, value):
        pass


def test_fit_units_is_read_once_and_the_model_is_never_fitted_on_a_scored_unit(
    build, monkeypatch, capsys
):
    model = _ChangingFitUnits()
    _module(monkeypatch, "changing", model)
    code = learn_run.main([*build, "--model", "changing"])
    assert code == 0, capsys.readouterr().out
    assert model.fitted == [[TRAINING]]
    assert model.reads == 1


def test_evaluate_refuses_tables_of_units_that_are_not_declared_training_units(tmp_path):
    from fleetreserve.gate_config import load

    matrix_file = tmp_path / "matrix.json"
    matrix_file.write_text(json.dumps(small_matrix()))
    matrix = load(matrix_file)
    stray_cases = [
        {GATE_UNITS[0]: pd.DataFrame()},  # a scored unit, not declared and not a training unit
        {TRAINING: pd.DataFrame(), "jpl": pd.DataFrame()},  # a training unit, but not declared
    ]
    for fit_frames in stray_cases:
        model = _Honest(fit_units=(TRAINING,))
        with pytest.raises(learn_run.Refused, match="was given tables"):
            learn_run.evaluate("honest", model, matrix, {}, fit_frames, workers=1)
        assert model.fitted == [] and model.given is None
