"""The earlier analysis's learned model `fleetreserve.models.earlier_gbm` (plan sections 7, 12) on
made-up cars tables.

1. Hand-computed numbers, one group: 30 training cars equal in every input, each leaving 3 h 10 min
   after τ with rate 3 kW and room 2 kWh; every level of every family is the common target (the
   quantile of a constant), so at each kappa the departure and its median are 3 hours (3 h 10 min
   rounded down to the quarter hour), the rate 3, the room 2 and the protected energy
   0.5 x 2 = 1.
2. Hand-computed numbers, two groups told apart by the time plugged in: 25 cars plugged in 1 hour
   before τ leaving 3 h 10 min after it (rate 3, room 2) and 25 plugged in 5 hours before leaving
   12 hours after it (rate 7, room 6). At levels up to 0.4 the trees' starting value is the lower
   group's (the 0.4-quantile of the 50 values) and each tree's split moves only the upper group,
   so at each kappa a car plugged in 1 hour gets 3, 3, 2 and one plugged in 5 hours 10 (12 hours
   capped at the 10-hour horizon), 7, 6.
3. Cars equal up to τ that differ after it (departure, energy, station) get the same forecast, on
   sessions run through the real path (`truth.build`, the cars table as CSV text, the rules,
   `learn_features.build_table`).
4. The settings are the earlier ones; the station is the last input column, categorical, coded
   from the training sites; an input with no value on any fit row is left out (reading E13); a
   site whose stations were coded in the fit is refused at predict.
5. `fit_units` are the training units, the model reads no outcome of a scored unit, and
   `learn_run --model earlier_gbm` runs end to end on the made-up build (exit 0).
"""

import io
import json

import numpy as np
import pandas as pd
import pytest
from gate_fake import COMMIT, GATE_UNITS, write_build
from learn_fake import PAIR, small_gate_units, small_matrix
from truth_tables_fake import FakeCar, date_of, write

from fleetreserve import (
    earlier_grid,
    earlier_inputs,
    gate_run,
    learn_eval,
    learn_features,
    learn_run,
    run_verify,
    truth,
)
from fleetreserve.days_from_truth import apply_car_rules, load_unit_days
from fleetreserve.forecast_model import KEY_COLUMNS, TABLE_COLUMNS, check_table, read_table
from fleetreserve.gate_config import parse
from fleetreserve.models import earlier_gbm
from fleetreserve.truth_build import TRAIN_UNITS
from fleetreserve.windows import Window

KAPPAS = (0.05, 0.1, 0.2, 0.3)
RHO = 0.5
LEAVES = 13.0 + 10 / 60  # 13:10, 3 h 10 min after τ (10:00)
A = range(25)  # the cars of each group in the two-group case


def _unit(folder, name, cars):
    write(folder, name, cars)
    return load_unit_days(folder, name, rho=RHO, no_fit="cap", small_fleet="keep").cars


def _frame(cars):
    return learn_features.feature_frame(learn_features.build_table(cars, rho=RHO))


def _fitted(folder, cars):
    model = earlier_gbm.EarlierGbmModel(rho=RHO)
    model.fit(_unit(folder, "caltech", cars))
    return model


@pytest.fixture(scope="module")
def one_group(tmp_path_factory):
    folder = tmp_path_factory.mktemp("one_group")
    cars = [
        FakeCar("T", date_of(0), 8.0, LEAVES, rate=3.0, room=2.0, session=f"t{k}")
        for k in range(30)
    ]
    model = _fitted(folder, cars)
    scored = [FakeCar("D", date_of(3), 9.0, 15.0, session="scored")]
    return model, check_table(model.predict(_frame(_unit(folder, "scored", scored)), KAPPAS))


def test_one_group_by_hand(one_group):
    _, table = one_group
    assert len(table) == len(KAPPAS)
    for column, value in (("d_hat", 3.0), ("d_median", 3.0), ("r_hat", 3.0), ("e_cap", 2.0)):
        assert table[column].tolist() == [value] * len(KAPPAS), column
    assert table["e_need"].tolist() == [0.5 * 2.0] * len(KAPPAS)
    assert (table["rho"] == RHO).all() and table["survival_h1"].isna().all()


@pytest.fixture(scope="module")
def two_groups(tmp_path_factory):
    folder = tmp_path_factory.mktemp("two_groups")
    cars = [FakeCar("T", date_of(0), 9.0, LEAVES, rate=3.0, room=2.0, session=f"a{k}") for k in A]
    cars += [FakeCar("T", date_of(0), 5.0, 22.0, rate=7.0, room=6.0, session=f"b{k}") for k in A]
    model = _fitted(folder, cars)
    scored = [
        FakeCar("D", date_of(3), 9.0, 11.0, session="one-hour"),
        FakeCar("D", date_of(3), 5.0, 11.0, session="five-hours"),
    ]
    table = check_table(model.predict(_frame(_unit(folder, "scored", scored)), KAPPAS))
    return model, table.set_index(["session_id", "kappa"])


def test_two_groups_by_hand(two_groups):
    _, table = two_groups
    for kappa in KAPPAS:
        one = table.loc[("one-hour", kappa), ["d_hat", "r_hat", "e_cap"]].to_numpy(float)
        five = table.loc[("five-hours", kappa), ["d_hat", "r_hat", "e_cap"]].to_numpy(float)
        assert one.tolist() == pytest.approx([3.0, 3.0, 2.0], abs=1e-6)
        assert five.tolist() == pytest.approx([10.0, 7.0, 6.0], abs=1e-6)


def _twin_sessions(zone="America/Denver"):
    # τ 08:00; three cars plug in at 07:00 at three stations and leave 0.5, 3 and 6 hours after τ,
    # charging at 2 kW from plug-in to plug-out
    rows = []
    for k, after in enumerate((0.5, 3.0, 6.0)):
        plug_in = pd.Timestamp("2019-06-04 07:00").tz_localize(zone)
        plug_out = pd.Timestamp("2019-06-04 08:00").tz_localize(zone) + pd.Timedelta(hours=after)
        rows.append(
            {
                "site": "nrel",
                "session_id": f"s{k}",
                "user_id": f"u{k}",
                "station": str(k),
                "plug_in": plug_in,
                "plug_out": plug_out,
                "energy_kwh": 2.0 * (1.0 + after),
                "nominal_kw": 6.6,
                "nominal_rule": "max_charge_power floored at 3.3 kW",
                "charge_start": plug_in,
                "charge_end": plug_out,
                "declared_departure": None,
            }
        )
    frame = pd.DataFrame(rows)
    frame["declared_departure"] = pd.to_datetime(frame["declared_departure"], utc=True)
    frame["declared_departure"] = frame["declared_departure"].dt.tz_convert(zone)
    cars, _ = truth.build(
        frame,
        Window(8, 3, "fake"),
        zone,
        unit="nrel",
        days=[pd.Timestamp("2019-06-04").date()],
        limit_kw=50.0,
        bound_by_charge_fields=True,
        setting=truth.MAIN,
    )
    text = io.StringIO()
    cars.to_csv(text, index=False)
    text.seek(0)
    ruled = apply_car_rules(pd.read_csv(text, dtype=str, keep_default_na=False), 0.0, "cap")
    return learn_features.feature_frame(learn_features.build_table(ruled, rho=RHO))


def test_cars_equal_up_to_tau_get_the_same_forecast(two_groups):
    model, _ = two_groups
    frame = _twin_sessions()
    assert len(frame) == 3 and frame["hours_since_plug_in"].tolist() == [1.0] * 3
    table = model.predict(frame, KAPPAS)
    numbers = [c for c in TABLE_COLUMNS if c not in KEY_COLUMNS]
    first = table.loc[table["session_id"] == "s0", numbers].reset_index(drop=True)
    for session in ("s1", "s2"):
        other = table.loc[table["session_id"] == session, numbers].reset_index(drop=True)
        pd.testing.assert_frame_equal(other, first)
    assert first["d_hat"].notna().all()


def test_the_settings_and_the_station_column(two_groups):
    model, _ = two_groups
    assert set(model.models) == {"stay", "rate", "room"}
    # no training car has a history or a declared departure: those six inputs have no value on
    # any fit row and are left out (reading E13); the station is the last column, categorical
    empty = {"stay_q10", "stay_q50", "kwh_q10", "kwh_q50", "days_since_last", "typed_remaining_h"}
    expected = [name not in empty for name in earlier_inputs.INPUTS] + [True]
    for columns, fitted in model.models.values():
        assert columns.tolist() == expected
        assert len(fitted) == len(earlier_grid.LEVELS)
        for level, regressor in zip(earlier_grid.LEVELS, fitted, strict=True):
            settings = regressor.get_params()
            assert settings["loss"] == "quantile" and settings["quantile"] == float(level)
            assert (settings["learning_rate"], settings["max_leaf_nodes"]) == (0.1, 7)
            assert (settings["min_samples_leaf"], settings["max_iter"]) == (20, 200)
            assert settings["early_stopping"] is False and settings["random_state"] == 0
            mask = np.asarray(settings["categorical_features"])
            assert mask.tolist() == [False] * (sum(expected) - 1) + [True]
    assert model.design.stations == [("T", "1")]


def test_a_site_whose_stations_were_fitted_is_refused(two_groups, tmp_path):
    model, _ = two_groups
    seen = [FakeCar("T", date_of(3), 9.0, 11.0, session="seen")]
    with pytest.raises(ValueError, match="stations"):
        model.predict(_frame(_unit(tmp_path, "scored", seen)), KAPPAS)
    with pytest.raises(ValueError, match="before fit"):
        earlier_gbm.EarlierGbmModel(rho=RHO).predict(_twin_sessions(), KAPPAS)
    assert len(model.predict(_twin_sessions().iloc[:0], KAPPAS)) == 0


def test_the_model_declares_the_training_units_and_reads_no_scored_outcome():
    model = earlier_gbm.make_model(parse(small_matrix()))
    assert model.name == "earlier_gbm"
    assert tuple(model.fit_units) == TRAIN_UNITS
    assert learn_run.check_model("earlier_gbm", model, gate_units=tuple(GATE_UNITS)) is False
    assert model.rho == parse(small_matrix()).rho


def _training_cars(unit: str) -> list[FakeCar]:
    return [
        FakeCar(
            f"T-{unit}",
            date_of(k),
            8.0,
            10.0 + (3.0, 5.0, 8.0)[j],
            rate=4.0 + j,
            room=1.0 + j,
            session=f"{unit}-{k}-{j}",
        )
        for k in range(8)
        for j in range(3)
    ]


@pytest.fixture(scope="module")
def end_to_end(tmp_path_factory):
    folder = tmp_path_factory.mktemp("gbm_run")
    units = {**small_gate_units(), **{u: _training_cars(u) for u in TRAIN_UNITS}}
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    out = folder / "out"
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        code = learn_run.main(
            ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
            + ["--model", "earlier_gbm"]
        )
    return code, out


def test_learn_run_scores_the_model_end_to_end(end_to_end):
    code, out = end_to_end
    assert code == 0
    results = json.loads((out / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == "earlier_gbm"
    assert results["fit_units"] == list(TRAIN_UNITS)
    assert results["failed_configurations"] == []
    for unit in GATE_UNITS:
        score = results["units"][unit]["kappas"]["0.1"]
        assert score["pinball_loss"]["departure_hours"]["mean"] is not None
        assert score["cars_with_departure_forecast"] == score["cars"] > 0
    for unit in GATE_UNITS[1:]:
        rungs = results["units"][unit]["configurations"][PAIR[0]]["rungs"]
        for rung in learn_eval.OFFER_RUNGS:
            assert rungs[rung]["test"]["mean_offer_kwh_per_day"] >= 0
    table = read_table(out / learn_run.TABLE_NAME)
    assert len(table) == sum(len(c) for c in small_gate_units().values())
    assert np.isfinite(table[["d_hat", "r_hat", "e_cap", "e_need", "d_median"]]).all().all()
