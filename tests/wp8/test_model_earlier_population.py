"""The earlier analysis's population model `fleetreserve.models.earlier_population` (plan
sections 7, 12) on made-up cars tables.

1. Hand-computed numbers: ten training cars plugged in at 8:00 with τ at 10:00, leaving 1 to 10
   hours after τ (stays 3 to 12 hours), rates 2 to 11 kW, rooms 2 to 20 kWh, protected share 0.5.
   For a car plugged in 2 hours and one plugged in 5.5 hours before τ, the departure quantiles at
   the four levels and the median, and the rate, room and protected-energy quantiles are worked
   out below.
2. A car plugged in longer than every training stay has no forecast (NaN), and keeps its row.
3. A forecast beyond the earlier horizon of 10 hours is cut to 10 hours.
4. Each training session counts once in the departure sample, whatever number of days it is
   listed on; the rate and room samples are the cars kept in a fleet.
5. `predict` reads only the keys and the hours since plug-in, and needs a fitted model; two cars
   equal up to τ that differ after it get the same forecast; the survival columns are NaN.
6. `fit_units` are the training units, the model reads no outcome of a scored unit, and
   `learn_run --model earlier_population` runs end to end on the made-up build (exit 0).
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from gate_fake import COMMIT, GATE_UNITS, write_build
from learn_fake import PAIR, small_gate_units, small_matrix
from truth_tables_fake import FakeCar, date_of, write

from fleetreserve import gate_run, learn_eval, learn_features, learn_run, run_verify
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import (
    KEY_COLUMNS,
    SURVIVAL_COLUMNS,
    TABLE_COLUMNS,
    check_table,
    read_table,
)
from fleetreserve.gate_config import parse
from fleetreserve.models import earlier_population
from fleetreserve.truth_build import TRAIN_UNITS

KAPPAS = (0.05, 0.1, 0.2, 0.3)
RHO = 0.5


def _unit(folder, name, cars, rho=RHO):
    write(folder, name, cars)
    return load_unit_days(folder, name, rho=rho, no_fit="cap", small_fleet="keep").cars


def _training(n=10, first=1.0, site="T"):
    return [
        FakeCar(
            site,
            date_of(0),
            8.0,
            10.0 + first + j,
            rate=2.0 + j,
            room=2.0 * (j + 1),
            session=f"t-{j}",
        )
        for j in range(n)
    ]


def _scored():
    # plugged in 2 hours (8:00) and 5.5 hours (4:30) before τ, and 13 hours before (21:00 the day
    # before), longer than every training stay
    return [
        FakeCar("D", date_of(5), 8.0, 15.0, session="two"),
        FakeCar("D", date_of(5), 4.5, 15.0, session="five-and-a-half"),
        FakeCar("D", date_of(5), -3.0, 15.0, session="thirteen"),
    ]


def _frame(cars):
    return learn_features.feature_frame(learn_features.build_table(cars, rho=RHO))


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    folder = tmp_path_factory.mktemp("population")
    model = earlier_population.EarlierPopulationModel(rho=RHO)
    model.fit(_unit(folder, "caltech", _training()))
    table = check_table(model.predict(_frame(_unit(folder, "scored", _scored())), KAPPAS))
    return model, table.set_index(["session_id", "kappa"])


def test_the_hand_computed_departure_quantiles(hand):
    _, table = hand
    # plugged in 2 h: every training car lasted 2 h; remaining stays 1, ..., 10 h; the lower
    # kappa-quantile is the k-th with k = floor(kappa * 10) + 1
    for kappa, d in ((0.05, 1.0), (0.1, 2.0), (0.2, 3.0), (0.3, 4.0)):
        assert table.loc[("two", kappa), "d_hat"] == pytest.approx(d)
        assert table.loc[("two", kappa), "d_median"] == pytest.approx(6.0)  # k = 6 of 10
    # plugged in 5.5 h: the seven with stays 6 to 12 h lasted; remaining 0.5, 1.5, ..., 6.5 h
    for kappa, d in ((0.05, 0.5), (0.1, 0.5), (0.2, 1.5), (0.3, 2.5)):
        assert table.loc[("five-and-a-half", kappa), "d_hat"] == pytest.approx(d)
        assert table.loc[("five-and-a-half", kappa), "d_median"] == pytest.approx(3.5)


def test_the_hand_computed_rate_room_and_need_quantiles(hand):
    _, table = hand
    # plugged in 5.5 h: the fleet cars of the sessions that lasted are j = 3, ..., 9: rates 5 to
    # 11 kW, rooms 8, 10, ..., 20 kWh. The lower quantile at q is the smallest value whose share
    # reaches q (the ceil(q * 7)-th); protected energy 0.5 x the (1 - kappa) one.
    expected = {0.05: (5.0, 8.0, 10.0), 0.1: (5.0, 8.0, 10.0), 0.2: (6.0, 10.0, 9.0)}
    expected[0.3] = (7.0, 12.0, 8.0)
    for kappa, (rate, room, need) in expected.items():
        row = table.loc[("five-and-a-half", kappa)]
        assert (row["r_hat"], row["e_cap"], row["e_need"]) == pytest.approx((rate, room, need))
    # plugged in 2 h: all ten; rates 2 to 11, rooms 2 to 20; ceil(q * 10)-th
    row = table.loc[("two", 0.2)]
    assert (row["r_hat"], row["e_cap"], row["e_need"]) == pytest.approx((3.0, 4.0, 0.5 * 16.0))
    row = table.loc[("two", 0.1)]
    assert (row["r_hat"], row["e_cap"], row["e_need"]) == pytest.approx((2.0, 2.0, 0.5 * 18.0))


def test_a_car_plugged_in_longer_than_every_training_stay_has_no_forecast(hand):
    _, table = hand
    for kappa in KAPPAS:
        row = table.loc[("thirteen", kappa)]
        for column in ("d_hat", "r_hat", "e_cap", "e_need", "d_median"):
            assert math.isnan(row[column]), column


def test_the_survival_columns_are_nan_and_every_row_is_there(hand):
    _, table = hand
    assert len(table) == 3 * len(KAPPAS)
    assert table[list(SURVIVAL_COLUMNS)].isna().all().all()
    assert (table["rho"] == RHO).all()


def test_a_forecast_beyond_the_horizon_is_cut_to_ten_hours(tmp_path):
    model = earlier_population.EarlierPopulationModel(rho=RHO)
    model.fit(_unit(tmp_path, "caltech", _training(n=5, first=20.0)))
    table = model.predict(_frame(_unit(tmp_path, "scored", _scored()[:1])), KAPPAS)
    assert (table["d_hat"] == 10.0).all() and (table["d_median"] == 10.0).all()


def test_a_session_listed_on_two_days_counts_once(tmp_path):
    cars = _training(n=4)
    # one more session plugged in at 8:00 on day 0 and out at 12:00 on day 1 (stay 28 h), listed
    # on both days. For a car plugged in 2 h: remaining stays 1, 2, 3, 4, 26, so the median is the
    # 3rd (k = floor(0.5 * 5) + 1), 3 h; counted twice, it would be the 4th of 6, 4 h
    cars.append(FakeCar("T", date_of(0), 8.0, 36.0, rate=5.0, room=4.0, session="long"))
    cars.append(FakeCar("T", date_of(1), 8.0 - 24, 12.0, rate=5.0, room=4.0, session="long"))
    model = earlier_population.EarlierPopulationModel(rho=RHO)
    model.fit(_unit(tmp_path, "caltech", cars))
    table = model.predict(_frame(_unit(tmp_path, "scored", _scored()[:1])), (0.3,))
    # the 0.3-quantile: k = floor(0.3 * 5) + 1 = 2
    assert table["d_hat"].tolist() == [2.0]
    assert table["d_median"].tolist() == [3.0]


def test_cars_equal_up_to_tau_get_the_same_forecast(hand, tmp_path):
    model, _ = hand
    # the same plug-in; after τ one leaves at 12:30 with little room, the other at 21:00 with more
    twins = [
        FakeCar("D", date_of(5), 8.0, 12.5, rate=2.0, room=1.0, session="early"),
        FakeCar("D", date_of(5), 8.0, 21.0, rate=9.0, room=7.0, energy=30.0, session="late"),
    ]
    table = model.predict(_frame(_unit(tmp_path, "scored", twins)), KAPPAS)
    numbers = [c for c in TABLE_COLUMNS if c not in KEY_COLUMNS]
    early = table.loc[table["session_id"] == "early", numbers].reset_index(drop=True)
    late = table.loc[table["session_id"] == "late", numbers].reset_index(drop=True)
    pd.testing.assert_frame_equal(early, late)
    assert early["d_hat"].notna().all()


def test_predict_reads_the_hours_since_plug_in_and_the_keys_only(hand, tmp_path):
    model, table = hand
    frame = _frame(_unit(tmp_path, "scored", _scored()))
    kept = frame[[*KEY_COLUMNS, "hours_since_plug_in"]]
    again = model.predict(kept, KAPPAS).set_index(["session_id", "kappa"])
    pd.testing.assert_frame_equal(again, table, check_dtype=False)
    with pytest.raises(ValueError, match="before fit"):
        earlier_population.EarlierPopulationModel(rho=RHO).predict(kept, KAPPAS)
    with pytest.raises(ValueError, match="no training cars"):
        earlier_population.EarlierPopulationModel(rho=RHO).fit(
            _unit(tmp_path, "caltech", _training()).iloc[:0]
        )
    assert len(model.predict(kept.iloc[:0], KAPPAS)) == 0


def test_the_model_declares_the_training_units_and_reads_no_scored_outcome():
    model = earlier_population.make_model(parse(small_matrix()))
    assert model.name == "earlier_population"
    assert tuple(model.fit_units) == TRAIN_UNITS
    assert learn_run.check_model("earlier_population", model, gate_units=tuple(GATE_UNITS)) is False
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
    folder = tmp_path_factory.mktemp("population_run")
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
            + ["--model", "earlier_population"]
        )
    return code, out


def test_learn_run_scores_the_model_end_to_end(end_to_end):
    code, out = end_to_end
    assert code == 0
    results = json.loads((out / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == "earlier_population"
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
    # every made-up gate car is plugged in at 8:00 (2 h before τ); the training stays are 5, 7
    # and 10 h, 64 of each, all lasting 2 h: remaining 3, 5, 8 h; k = floor(0.1 * 192) + 1 = 20
    assert (table["d_hat"] == 3.0).all() and (table["d_median"] == 5.0).all()
    # rates 4 to 6 kW and rooms 1 to 3 kWh, 64 of each: the ceil(0.1 * 192) = 20th is the lowest
    assert (table["r_hat"] == 4.0).all() and (table["e_cap"] == 1.0).all()
    assert np.isfinite(table["e_need"]).all()
