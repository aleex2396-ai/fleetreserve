"""The declared-departure models `fleetreserve.models.declared` and `declared_shifted` (plan
section 7, forecast baseline "declared departure (ACN, NREL)"), on made-up units.

1. The feature `declared_departure_h` of `learn_features` is the table's value where the statement
   was made at or before τ, and NaN where it was made after τ, where no statement time is given or
   where no departure was declared. It is a column known at τ and in `feature_frame`; the flag
   `plug_out_from_declared` is not known at τ and never in it.
2. `declared` on a made-up unit: at every kappa `d_hat` and `d_median` are the declared departure
   (the same value at every kappa), NaN when none is usable (none declared, stated after τ, or
   already passed at τ); `survival_h<t>` is the declared departure for the hours before it, NaN
   after; rate and energy columns NaN; nothing is fitted.
3. `declared_shifted`, hand-worked: twenty training cars whose true departure minus declared
   departure is -2, -1.75, ..., 2.75 hours. Lower kappa-quantiles (k = floor(kappa * 20) + 1):
   0.05 -> x_(2) = -1.75, 0.1 -> x_(3) = -1.5, 0.2 -> x_(5) = -1.0, 0.3 -> x_(7) = -0.5, median
   x_(11) = 0.5. A declared departure of 3 h gives `d_hat` 1.25, 1.5, 2.0, 2.5; one of 1 h gives 0
   at kappa 0.1 (raised from -0.5). Fewer than `min_cell` training cars fail the fit.
4. `learn_run --model declared` and `--model declared_shifted` end to end on a made-up verified
   build: exit 0, scores and offers written, the forecast table's departures are the declared ones
   (NaN for statements after τ), and the model is given the feature frame only.
5. A model reading `plug_out_from_declared` in `predict` fails (1): the guard covers the new flag.
"""

import dataclasses
import json
import math
import sys
import types

import numpy as np
import pandas as pd
import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_cars, write_build
from learn_fake import small_gate_units, small_matrix
from truth_tables_fake import FakeCar, date_of, tables, write

from fleetreserve import gate_run, learn_features, learn_run, run_verify
from fleetreserve.days_from_truth import apply_car_rules, load_unit_days
from fleetreserve.forecast_model import SURVIVAL_COLUMNS, check_table, read_table
from fleetreserve.models import declared, declared_shifted

KAPPAS = (0.05, 0.1, 0.2, 0.3)


def _frame(fakes) -> pd.DataFrame:
    car_table, _ = tables(fakes)
    return learn_features.feature_frame(
        learn_features.build_table(apply_car_rules(car_table, 0.0, "cap"), rho=0.0)
    )


def _car(session, declared_h=None, stated=8.5, out=16.0):
    """A car plugged in at 8:00 (τ 10:00) leaving at `out`, declaring `declared_h` hours after τ."""
    return FakeCar(
        "A",
        date_of(0),
        8.0,
        out,
        session=session,
        declared=None if declared_h is None else 10.0 + declared_h,
        declared_at=None if declared_h is None else stated,
    )


def test_the_feature_never_holds_a_statement_made_after_tau():
    fakes = [
        _car("before", 4.5),
        _car("at-tau", 3.0, stated=10.0),
        _car("after", 4.0, stated=10.25),
        dataclasses.replace(_car("no-time", 4.0), declared_at=None),
        _car("none"),
    ]
    frame = _frame(fakes).set_index("session_id")
    value = frame["declared_departure_h"]
    assert value["before"] == 4.5 and value["at-tau"] == 3.0
    for name in ("after", "no-time", "none"):
        assert math.isnan(value[name]), name
    assert "declared_departure_h" in learn_features.KNOWN_AT_TAU
    assert "declared_at" in learn_features.KNOWN_AT_TAU
    assert "plug_out_from_declared" in learn_features.NOT_KNOWN_AT_TAU
    assert "plug_out_from_declared" not in frame.columns


def test_declared_gives_the_stated_time_at_every_kappa():
    fakes = [
        _car("plain", 4.5),
        _car("short", 0.5),
        _car("after", 4.0, stated=10.25),
        _car("passed", -1.0),
        _car("none"),
    ]
    model = declared.make_model(types.SimpleNamespace(rho=0.0))
    assert model.fit_units == ()
    model.fit(pd.DataFrame())
    table = check_table(model.predict(_frame(fakes), KAPPAS))
    assert len(table) == len(fakes) * len(KAPPAS)
    rows = table.set_index(["session_id", "kappa"])
    for kappa in KAPPAS:
        assert rows.loc[("plain", kappa), "d_hat"] == 4.5
        assert rows.loc[("plain", kappa), "d_median"] == 4.5
        assert rows.loc[("short", kappa), "d_hat"] == 0.5
        for name in ("after", "passed", "none"):
            assert math.isnan(rows.loc[(name, kappa), "d_hat"]), name
            assert math.isnan(rows.loc[(name, kappa), "d_median"]), name
        survival = [rows.loc[("plain", kappa), c] for c in SURVIVAL_COLUMNS]
        assert survival[:4] == [4.5] * 4 and all(math.isnan(v) for v in survival[4:])
        assert all(math.isnan(rows.loc[("short", kappa), c]) for c in SURVIVAL_COLUMNS)
    assert table[["r_hat", "e_cap", "e_need"]].isna().all().all()


def _training_unit(tmp_path, n=20):
    fakes = [
        FakeCar(
            "T",
            date_of(j // 4),
            8.0,
            15.0,
            session=f"t{j}",
            declared=15.0 - (j * 0.25 - 2.0),
            declared_at=8.5,
        )
        for j in range(n)
    ]
    write(tmp_path, "train", fakes)
    return load_unit_days(tmp_path, "train", rho=0.0, no_fit="cap", small_fleet="keep")


def test_declared_shifted_moves_the_statement_by_the_training_errors(tmp_path):
    unit = _training_unit(tmp_path)
    model = declared_shifted.make_model(types.SimpleNamespace(rho=0.0, min_cell=20))
    assert model.fit_units == ("caltech", "jpl", "office001")
    model.fit(unit.cars)
    table = check_table(
        model.predict(_frame([_car("three", 3.0), _car("one", 1.0), _car("none")]), KAPPAS)
    )
    rows = table.set_index(["session_id", "kappa"])
    expected = {0.05: 1.25, 0.1: 1.5, 0.2: 2.0, 0.3: 2.5}
    for kappa, value in expected.items():
        assert rows.loc[("three", kappa), "d_hat"] == pytest.approx(value)
        assert rows.loc[("three", kappa), "d_median"] == pytest.approx(3.5)
        assert math.isnan(rows.loc[("none", kappa), "d_hat"])
    assert rows.loc[("one", 0.1), "d_hat"] == 0.0
    assert rows.loc[("one", 0.3), "d_hat"] == pytest.approx(0.5)
    assert table[list(SURVIVAL_COLUMNS)].isna().all().all()


def test_declared_shifted_needs_enough_training_cars(tmp_path):
    unit = _training_unit(tmp_path, n=19)
    model = declared_shifted.make_model(types.SimpleNamespace(rho=0.0, min_cell=20))
    with pytest.raises(ValueError, match="19 training cars"):
        model.fit(unit.cars)
    with pytest.raises(ValueError, match="before fit"):
        model.predict(_frame([_car("x", 3.0)]), KAPPAS)


# --- end to end through learn_run ----------------------------------------------------------------


def _with_declared(cars):
    """Each car declares its true plug-out moved by -0.6 to +0.6 hours, stated at 8:30; every
    seventh car states it at 10:30, after τ, so it must get no forecast."""
    out = []
    for k, car in enumerate(cars):
        late = k % 7 == 3
        shift = 0.3 * ((k % 5) - 2)
        out.append(
            dataclasses.replace(
                car, declared=car.plug_out + shift, declared_at=10.5 if late else 8.5
            )
        )
    return out


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {u: _with_declared(c) for u, c in small_gate_units().items()}
    for site in ("caltech", "jpl", "office001"):
        units[site] = _with_declared(gate_cars(8, site))
    run, tables_folder = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    arguments = ["--matrix", str(matrix), "--tables", str(tables_folder), "--truth-run", str(run)]
    return arguments, tables_folder


@pytest.mark.parametrize("name", ["declared", "declared_shifted"])
def test_learn_run_scores_the_model(build, tmp_path, monkeypatch, capsys, name):
    arguments, tables_folder = build
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    assert learn_run.main([*arguments, "--model", name]) == 0, capsys.readouterr().out
    results = json.loads((tmp_path / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == name and results["failed_configurations"] == []
    table = read_table(tmp_path / learn_run.TABLE_NAME)
    for unit in GATE_UNITS:
        score = results["units"][unit]["kappas"]["0.1"]
        assert score["pinball_loss"]["departure_hours"]["mean"] is not None
        assert 0 < score["cars_with_departure_forecast"] < score["cars"]
        cars = pd.read_csv(tables_folder / f"{unit}-cars.csv", dtype=str, keep_default_na=False)
        stated_by_tau = cars["declared_at"].str[11:16] == "08:30"
        declared_h = cars["declared_departure_h"].astype(float).where(stated_by_tau)
        expected = dict(zip(cars["session_id"], declared_h, strict=True))
        mine = table[(table["unit"] == unit) & (table["kappa"] == 0.1)]
        assert len(mine) == len(cars)
        for session, d_hat in zip(mine["session_id"], mine["d_hat"], strict=True):
            want = expected[session]
            if math.isnan(want):
                assert math.isnan(d_hat)
            elif name == "declared":
                assert d_hat == pytest.approx(want)
            else:
                assert not math.isnan(d_hat) and d_hat < want  # moved earlier
    if name == "declared":
        assert results["fit_units"] == []
    else:
        assert results["fit_units"] == ["caltech", "jpl", "office001"]


def _register(monkeypatch, name, model):
    module = types.ModuleType(f"fleetreserve.models.{name}")
    module.make_model = lambda matrix: model
    monkeypatch.setitem(sys.modules, module.__name__, module)


class _Spy(declared.DeclaredModel):
    name = "spy"

    def predict(self, frame, kappas):
        self.given = frame
        return super().predict(frame, kappas)


class _Cheat(declared.DeclaredModel):
    name = "cheat"

    def predict(self, frame, kappas):
        frame["plug_out_from_declared"].to_numpy()
        return super().predict(frame, kappas)


def test_the_declared_model_is_given_features_only(build, monkeypatch, capsys):
    arguments, _ = build
    spy = _Spy(rho=0.0)
    _register(monkeypatch, "spy", spy)
    assert learn_run.main([*arguments, "--model", "spy"]) == 0, capsys.readouterr().out
    hidden = set(learn_features.NOT_KNOWN_AT_TAU) | set(learn_features.LABEL_COLUMNS)
    assert not set(spy.given.columns) & hidden
    assert "declared_departure_h" in spy.given.columns
    assert np.isfinite(spy.given["declared_departure_h"]).any()


def test_a_model_reading_the_plug_out_flag_fails(build, monkeypatch, capsys):
    arguments, _ = build
    _register(monkeypatch, "cheat", _Cheat(rho=0.0))
    assert learn_run.main([*arguments, "--model", "cheat"]) == 1
    out = capsys.readouterr().out
    assert "predict raised KeyError" in out and "plug_out_from_declared" in out
