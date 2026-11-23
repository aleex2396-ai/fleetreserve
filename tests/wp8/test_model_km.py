"""The survival-curve baseline `fleetreserve.models.km` on made-up stays (plan section 7, WP8).

1. Hand-computed numbers: ten cars plugged in at 8:00, τ at 10:00, leaving 1, 2, ..., 10 hours after
   τ with rates 2 to 11 kW and rooms 2 to 20 kWh (protected share 0.5). The departure quantile at
   the four levels, the median, the conditional quantile at each service hour 1 to 8, and the rate,
   room and protected-energy quantiles are the numbers worked out by hand below.
2. The survival curve gives the quantile `forecast_empirical.lower_quantile` takes of the sample.
3. Small cells: a cell with fewer than `min_cell` cars, and a cell never seen, take all training
   cars; a cell with enough cars but too few still plugged in at an hour takes all cars for that
   hour only; cars plugged in on an earlier day are their own cell.
4. Quantities are monotone in kappa where they must be.
5. `predict` reads the feature frame's arrival columns and keys only, and needs a fitted model.
6. `fit_units` are the training units, and `learn_run --model km` fits on them alone and scores the
   made-up development units: exit 0, `learn.json` holds losses, concordance index and offers.
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
from fleetreserve.forecast_empirical import lower_quantile
from fleetreserve.forecast_model import (
    KEY_COLUMNS,
    SURVIVAL_COLUMNS,
    TABLE_COLUMNS,
    check_table,
    read_table,
)
from fleetreserve.gate_config import parse
from fleetreserve.models import km
from fleetreserve.truth_build import TRAIN_UNITS

KAPPAS = (0.05, 0.1, 0.2, 0.3)
RHO = 0.5


def _cars(departures, *, plug_in=8.0, site="A", prefix="c", rates=None, rooms=None, day=0):
    """Cars plugged in at `plug_in`, leaving `departures` hours after τ (10:00); rate and room
    large enough to fit whatever the rules cap by."""
    return [
        FakeCar(
            site,
            date_of(day),
            plug_in,
            10.0 + d,
            rate=2.0 + j if rates is None else rates[j],
            room=2.0 * (j + 1) if rooms is None else rooms[j],
            session=f"{prefix}-{j}",
        )
        for j, d in enumerate(departures)
    ]


def _fitted(tmp_path, cars, *, min_cell, rho=RHO):
    write(tmp_path, "train", cars)
    unit = load_unit_days(tmp_path, "train", rho=rho, no_fit="cap", small_fleet="keep")
    model = km.KMModel(rho=rho, min_cell=min_cell)
    model.fit(unit.cars)
    return model, unit


def _ask(model, arrival_hours, kappas=KAPPAS, earlier=False):
    frame = pd.DataFrame(
        {
            "unit": "scored",
            "site": "S",
            "date": date_of(0),
            "session_id": [f"q{k}" for k in range(len(arrival_hours))],
            "arrival_hour": arrival_hours,
            "arrived_earlier_day": earlier,
        }
    )
    table = check_table(model.predict(frame, kappas))
    return {
        (row.session_id, row.kappa): row._asdict() for row in table.itertuples(index=False)
    }, table


def _same(found, expected):
    assert found == pytest.approx(expected, abs=1e-9), (found, expected)


# Worked out by hand for departures 1, 2, ..., 10 (n = 10) at arrival hour 8; the lower
# kappa-quantile is the k-th smallest with k = floor(kappa * n) + 1. Given presence at hour t the
# same rule on the m cars with departure above t.
D_HAT = {0.05: 1.0, 0.1: 2.0, 0.2: 3.0, 0.3: 4.0}
D_MEDIAN = 6.0  # k = 6
SURVIVAL = {  # kappa -> the quantile given presence at hours 1 to 8 (m = 9, 8, ..., 2)
    0.05: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    0.1: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    0.2: [3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 8.0, 9.0],
    0.3: [4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 8.0, 9.0],
}
RATE = {0.05: 2.0, 0.1: 3.0, 0.2: 4.0, 0.3: 5.0}  # rates 2..11: x_(k)
ROOM = {0.05: 2.0, 0.1: 4.0, 0.2: 6.0, 0.3: 8.0}  # rooms 2, 4, ..., 20: x_(k)
NEED = {0.05: 10.0, 0.1: 9.0, 0.2: 8.0, 0.3: 7.0}  # 0.5 * rooms, x_(n + 1 - k)


@pytest.fixture(scope="module")
def hand(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hand")
    model, unit = _fitted(folder, _cars(range(1, 11)), min_cell=2)
    return model, unit


def test_the_hand_computed_quantiles(hand):
    model, _ = hand
    rows, table = _ask(model, [8])
    assert len(table) == len(KAPPAS)
    assert list(table.columns) == list(TABLE_COLUMNS)
    for kappa in KAPPAS:
        row = rows[("q0", kappa)]
        _same(row["rho"], RHO)
        _same(row["d_hat"], D_HAT[kappa])
        _same(row["d_median"], D_MEDIAN)
        for hour, column in enumerate(SURVIVAL_COLUMNS):
            _same(row[column], SURVIVAL[kappa][hour])
        _same(row["r_hat"], RATE[kappa])
        _same(row["e_cap"], ROOM[kappa])
        _same(row["e_need"], NEED[kappa])


def test_the_survival_curve_gives_the_quantile_of_the_sample(hand):
    model, unit = hand
    sample = np.sort(unit.cars["departure_exact"].astype(float).to_numpy())
    times, survival = km._Sample(sample, sample, sample).curve()
    assert list(times) == list(sample) and survival[-1] == 0.0
    rows, _ = _ask(model, [8])
    for kappa in KAPPAS:
        assert rows[("q0", kappa)]["d_hat"] == lower_quantile(sample, kappa)
        for hour, column in enumerate(SURVIVAL_COLUMNS, start=1):
            present = sample[sample > hour]
            expected = lower_quantile(present, kappa) if present.size >= 2 else None
            if expected is not None:
                assert rows[("q0", kappa)][column] == expected


def test_no_censoring_is_needed_so_the_departures_are_the_labels(hand):
    _, unit = hand
    table = learn_features.build_table(unit.cars, rho=RHO)
    assert sorted(table["label_departure_h"]) == [float(d) for d in range(1, 11)]


# Ten cars at arrival hour 8 (departures 1..10) and two at hour 9 leaving 9.25 and 9.5 hours after
# τ with a rate of 100 kW and rooms of 30 and 32 kWh; min_cell 3. All twelve cars together:
# departures 1, ..., 9, 9.25, 9.5, 10 (n = 12), so at kappa 0.1 (k = 2) the quantile is 2, and the
# rates 2..11, 100, 100 give 3, the rooms 2, ..., 20, 30, 32 give 4, and 0.5 * rooms give the upper
# x_(11) = 15.


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("mixed")
    cars = _cars(range(1, 11)) + _cars(
        [9.25, 9.5], plug_in=9.0, prefix="n", rates=[100.0, 100.0], rooms=[30.0, 32.0]
    )
    return _fitted(folder, cars, min_cell=3)[0]


def test_a_small_cell_takes_all_cars(mixed):
    rows, _ = _ask(mixed, [9], kappas=(0.1,))
    row = rows[("q0", 0.1)]
    _same(row["d_hat"], 2.0)  # its own two cars would give 9.25
    _same(row["r_hat"], 3.0)  # its own cars would give 100
    _same(row["e_cap"], 4.0)
    _same(row["e_need"], 15.0)
    _same(row["survival_h8"], 9.0)  # all cars with departure above 8: 9, 9.25, 9.5, 10 -> x_(1)


def test_a_cell_never_seen_takes_all_cars(mixed):
    rows, _ = _ask(mixed, [15], kappas=(0.1,))
    _same(rows[("q0", 0.1)]["d_hat"], 2.0)
    _same(rows[("q0", 0.1)]["r_hat"], 3.0)


def test_a_cell_with_few_cars_left_at_an_hour_takes_all_cars_for_that_hour_only(mixed):
    rows, _ = _ask(mixed, [8], kappas=(0.3,))
    row = rows[("q0", 0.3)]
    _same(row["d_hat"], 4.0)  # the cell's own ten cars
    _same(row["r_hat"], 5.0)
    _same(row["survival_h7"], 8.0)  # three of the cell's cars are left: its own, x_(1) of 8, 9, 10
    # Hour 8: two of the cell's cars are left, fewer than min_cell. The cell alone would give 9
    # (x_(1) of 9, 10); all cars (9, 9.25, 9.5, 10, k = floor(1.2) + 1 = 2) give 9.25.
    _same(row["survival_h8"], 9.25)


def test_rate_room_and_need_quantiles_use_only_the_cars_kept_in_a_fleet(tmp_path):
    # Two cars leave within the hour after τ, so the rules leave them out of their fleet (no room).
    # Their rates and rooms are extreme: if they entered the fleet quantiles, r_hat, e_cap and
    # e_need would move away from the hand-computed numbers of the ten kept cars.
    outside = [
        FakeCar("A", date_of(0), 8.0, 10.5, rate=0.1, room=1000.0, session="out-0"),
        FakeCar("A", date_of(0), 8.0, 10.5, rate=500.0, room=0.1, session="out-1"),
    ]
    model, unit = _fitted(tmp_path, _cars(range(1, 11)) + outside, min_cell=2)
    assert int(unit.cars["room"].isna().sum()) == 2  # the two cars are outside the fleet
    rows, _ = _ask(model, [8])
    for kappa in KAPPAS:
        _same(rows[("q0", kappa)]["r_hat"], RATE[kappa])
        _same(rows[("q0", kappa)]["e_cap"], ROOM[kappa])
        _same(rows[("q0", kappa)]["e_need"], NEED[kappa])
    # the departure curve still uses every car, so the two early leavers do show there
    _same(rows[("q0", 0.2)]["d_hat"], 1.0)


def test_cars_plugged_in_on_an_earlier_day_are_their_own_cell(tmp_path):
    cars = _cars([1, 2], prefix="a") + _cars([6, 7], plug_in=-3.0, prefix="e")
    model, _ = _fitted(tmp_path, cars, min_cell=2)
    same_day, _ = _ask(model, [8], kappas=(0.1,))
    earlier, _ = _ask(model, [8], kappas=(0.1,), earlier=True)
    _same(same_day[("q0", 0.1)]["d_hat"], 1.0)
    _same(earlier[("q0", 0.1)]["d_hat"], 6.0)


def test_quantities_are_monotone_in_kappa(tmp_path):
    rng = np.random.default_rng(7)
    n = 90
    cars = _cars(
        rng.uniform(1.0, 11.0, n).round(3),
        rates=rng.uniform(3.0, 10.0, n).round(3),
        rooms=rng.uniform(1.0, 3.0, n).round(3),
    )
    model, _ = _fitted(tmp_path, cars, min_cell=20)
    rows, _ = _ask(model, [8])
    series = {
        name: [rows[("q0", kappa)][name] for kappa in KAPPAS]
        for name in ("d_hat", "r_hat", "e_cap", "e_need", *SURVIVAL_COLUMNS)
    }
    for name, values in series.items():
        assert all(math.isfinite(v) for v in values), name
        steps = np.diff(values)
        assert (steps <= 0).all() if name == "e_need" else (steps >= 0).all(), (name, values)
    assert series["e_need"][0] > series["e_need"][-1]  # not constant: the check can fail
    assert series["d_hat"][0] < series["d_hat"][-1]
    for kappa in KAPPAS:
        row = rows[("q0", kappa)]
        bounds = [row["d_hat"], *(row[c] for c in SURVIVAL_COLUMNS)]
        # Given presence later, the departure is later: each bound is above the start of its hour
        # and no earlier than the one of the hour before.
        assert all(b2 >= b1 for b1, b2 in zip(bounds, bounds[1:], strict=False))
        assert all(row[c] > hour for hour, c in enumerate(SURVIVAL_COLUMNS, start=1))
        assert row["d_hat"] <= row["d_median"]


def test_predict_reads_the_arrival_columns_and_keys_only(hand):
    model, unit = hand
    frame = learn_features.feature_frame(learn_features.build_table(unit.cars, rho=RHO))
    thin = frame[[*KEY_COLUMNS, "arrival_hour", "arrived_earlier_day"]]
    assert set(thin.columns) < set(frame.columns)
    full = model.predict(frame, KAPPAS)
    pd.testing.assert_frame_equal(full, model.predict(thin, KAPPAS))
    assert len(full) == len(frame) * len(KAPPAS)


def test_an_unfitted_model_and_a_fit_on_nothing_are_errors(hand):
    _, unit = hand
    with pytest.raises(ValueError, match="before fit"):
        km.KMModel(rho=RHO, min_cell=2).predict(unit.cars, KAPPAS)
    with pytest.raises(ValueError, match="no training cars"):
        km.KMModel(rho=RHO, min_cell=2).fit(unit.cars.iloc[0:0])


def test_an_empty_frame_gives_an_empty_table(hand):
    model, _ = hand
    table = model.predict(
        pd.DataFrame(columns=[*KEY_COLUMNS, "arrival_hour", "arrived_earlier_day"]), KAPPAS
    )
    assert len(table) == 0 and list(table.columns) == list(TABLE_COLUMNS)


# --- fit units and the shared evaluation path --------------------------------------------------


def test_the_fit_units_are_the_training_units():
    model = km.make_model(parse(small_matrix()))
    assert model.name == "km"
    assert tuple(model.fit_units) == TRAIN_UNITS and len(model.fit_units) == 8
    assert not set(model.fit_units) & set(GATE_UNITS)
    learn_run.check_model("km", model)  # raises Refused otherwise
    assert getattr(model, "reads_other_day_outcomes", False) is False
    assert model.rho == parse(small_matrix()).rho
    assert model.min_cell == parse(small_matrix()).min_cell


def _training_cars(unit: str) -> list[FakeCar]:
    """Eight days of three cars at 8:00 leaving 3, 5 and 8 hours after τ with rates 4, 5, 6 kW and
    rooms 1, 2, 3 kWh: unlike the made-up development units (about 4.4, 5.6, 6.7 hours; 2, 2.5,
    3 kW)."""
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
    folder = tmp_path_factory.mktemp("km_run")
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
            + ["--model", "km"]
        )
    return code, out


def test_learn_run_scores_km_end_to_end(end_to_end):
    code, out = end_to_end
    assert code == 0
    results = json.loads((out / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == "km"
    assert results["fit_units"] == list(TRAIN_UNITS)
    assert results["failed_configurations"] == []
    for unit in GATE_UNITS:
        entry = results["units"][unit]
        score = entry["kappas"]["0.1"]
        assert score["pinball_loss"]["departure_hours"]["mean"] is not None
        assert "value" in score["concordance_index"]
        assert score["cars_with_departure_forecast"] == score["cars"] > 0
        found = entry["configurations"][PAIR[0]]
        assert found["oracle_partner"]["name"] == PAIR[1]
    for unit in GATE_UNITS[1:]:  # the two calibratable units
        rungs = results["units"][unit]["configurations"][PAIR[0]]["rungs"]
        for rung in learn_eval.OFFER_RUNGS:
            assert rungs[rung]["test"]["mean_offer_kwh_per_day"] >= 0


def test_the_forecasts_come_from_the_training_units_alone(end_to_end):
    _, out = end_to_end
    table = read_table(out / learn_run.TABLE_NAME)
    assert set(table["unit"]) == set(GATE_UNITS)
    assert len(table) == sum(len(c) for c in small_gate_units().values())
    # 64 of each of 3, 5 and 8 hours at arrival hour 8: the 0.1-quantile is 3 (k = 20 of n = 192),
    # the median 5, rates 4 to 6 kW and rooms 1 to 3 kWh; the development units' cars leave
    # 4.4 hours or later and charge at 2 to 3 kW.
    assert (table["d_hat"] == 3.0).all() and (table["d_median"] == 5.0).all()
    assert (table["r_hat"] == 4.0).all() and (table["e_cap"] == 1.0).all()
    assert (table["e_need"] == 3.0 * (1 - float(table["rho"].iloc[0]))).all()
    assert (table["survival_h1"] == 3.0).all()  # everyone is still there at hour 1 and 3 > 1
