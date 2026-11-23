"""The conditional-fit survival bound (`models.conditional_fit`) and its two models,
`models.lgbm_surv` and `models.lgbm_tail_surv`, on made-up cars (`survival_fake`): 150 training
days and 40 later days of ten users, each car leaving early (0.2 to 3.8 hours after τ) or late
(5 to 7.5 hours after τ).

1. The older models are untouched: the forecast tables of `lgbm`, `cqr` and `lgbm_tail` on these
   cars have the SHA-256 they had at the commit before the rule was added (`OLDER_SHA256`).
2. The new models are `lgbm` and `lgbm_tail` in every other column, are fitted on the training
   units only, pass the learn guard, and two fits agree.
3. The survival columns hold a number where at least `MIN_ROWS` training rows are still plugged
   in at the hour's start and NaN otherwise, at the exact count.
4. The conditioning is real: the bound of hours 4 and 5 lies in the late group, above the
   forecast departure at τ, and each hour's boosters saw only the rows still plugged in.
5. No leak: the forecast of a day's cars does not change when their own outcomes are altered, nor
   when label columns are handed to `predict`.
6. The bound of hour h is at least h, also when the fitted model says less.
7. `learn_run --model lgbm_tail_surv` and `--model lgbm_surv` run end to end on made-up units, and
   the gate's re-solve bounds of a made-up day are later than under `lgbm_tail`.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import survival_fake as sf
from gate_fake import COMMIT, COMMITTED, GATE_UNITS, gate_cars, write_build
from learn_fake import PAIR, small_gate_units, small_matrix
from truth_tables_fake import date_of

from fleetreserve import gate_day, gate_run, learn_run, run_verify
from fleetreserve.forecast_empirical import MIN_CELL
from fleetreserve.forecast_model import SURVIVAL_COLUMNS, TableForecaster, check_table, read_table
from fleetreserve.gate_config import Configuration
from fleetreserve.learn_features import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    build_table,
    feature_frame,
)
from fleetreserve.models import conditional_fit, lgbm, lgbm_surv, lgbm_tail, lgbm_tail_surv
from fleetreserve.truth_build import TRAIN_UNITS

KAPPAS = (0.1, 0.3)
TRAIN_DAYS = 150
DAYS = 190
SURVIVAL = list(SURVIVAL_COLUMNS)
OTHER = ["d_hat", "r_hat", "e_cap", "e_need", "d_median"]
# `survival_fake.sha256_of(survival_fake.older_tables(train, fresh, KAPPAS))` at the commit before
# the conditional-fit rule was added.
OLDER_SHA256 = {
    "lgbm": "71c75806b66adaa084c14a9a6baa4b1eb584bd1ad186554caa41a6ec712d5b61",
    "cqr": "aeeff5f9672b996c8e0b454febc85170474f6647d74ae6ab3aaa491af4b78aa8",
    "lgbm_tail": "c9fd4118d329e0965cec8a614773f8034419cbab6b42175c4a778edfe016ce3f",
}
NEW = {"lgbm_surv": lgbm_surv.LgbmSurvModel, "lgbm_tail_surv": lgbm_tail_surv.LgbmTailSurvModel}
PLAIN = {"lgbm_surv": "lgbm", "lgbm_tail_surv": "lgbm_tail"}


def _fresh(cars: pd.DataFrame, first_day: int = TRAIN_DAYS) -> pd.DataFrame:
    table = build_table(cars, rho=0.0)
    return feature_frame(table[table["date"] >= date_of(first_day)].reset_index(drop=True))


@pytest.fixture(scope="module")
def data():
    fakes, users = sf.two_stay_cars(DAYS)
    cars = sf.cars_table(fakes, users)
    train = cars[cars["date"] < date_of(TRAIN_DAYS)].reset_index(drop=True)
    return train, _fresh(cars)


@pytest.fixture(scope="module")
def older(data):
    train, fresh = data
    return sf.older_tables(train, fresh, KAPPAS)


@pytest.fixture(scope="module")
def fitted(data):
    train, _ = data
    out = {}
    for name, make in NEW.items():
        out[name] = make(rho=0.0, kappas=KAPPAS)
        out[name].fit(train)
    return out


@pytest.fixture(scope="module")
def forecasts(data, fitted):
    _, fresh = data
    return {name: check_table(model.predict(fresh, KAPPAS)) for name, model in fitted.items()}


# --- 1. the older models ---


def test_the_older_models_forecast_tables_keep_their_bytes(older, tmp_path):
    assert sf.sha256_of(older, tmp_path) == OLDER_SHA256
    for table in older.values():
        assert table[SURVIVAL].isna().all().all()
    assert lgbm.SURVIVAL_BOUND == lgbm.NO_SURVIVAL_BOUND


# --- 2. the new models ---


@pytest.mark.parametrize("name", list(NEW))
def test_the_new_model_is_the_plain_one_in_every_other_column(older, fitted, forecasts, name):
    model, table, plain = fitted[name], forecasts[name], older[PLAIN[name]]
    assert model.name == name
    assert model.fit_units == TRAIN_UNITS
    assert learn_run.check_model(name, model) is False
    assert getattr(model, "reads_other_day_outcomes", False) is False
    not_survival = [c for c in table.columns if c not in SURVIVAL]
    pd.testing.assert_frame_equal(table[not_survival], plain[not_survival], check_exact=True)
    assert not table[OTHER].isna().any().any()


def test_the_bound_models_have_the_settings_and_columns_of_the_model(fitted):
    assert lgbm_surv.SURVIVAL_BOUND == conditional_fit.CONDITIONAL_FIT != lgbm.NO_SURVIVAL_BOUND
    assert lgbm_surv.LgbmSurvModel.COLUMNS == FEATURE_COLUMNS
    assert lgbm_tail_surv.LgbmTailSurvModel.COLUMNS == lgbm_tail.COLUMNS
    for model in fitted.values():
        bounds = model.bounds
        assert bounds.columns == model.COLUMNS == model.quantiles.columns
        assert set(bounds.boosters) == {(h, k) for h in range(1, 8) for k in KAPPAS}
        for (_, kappa), booster in bounds.boosters.items():
            assert booster.feature_name() == list(model.COLUMNS)
            assert booster.params["alpha"] == kappa
            assert booster.num_trees() == lgbm.ROUNDS
            for setting, value in lgbm.PARAMETERS.items():
                assert booster.params[setting] == value


def _matrix() -> SimpleNamespace:
    """The two fields of a run matrix `make_model` reads."""
    return SimpleNamespace(rho=0.0, configurations=[SimpleNamespace(kappa=k) for k in KAPPAS])


def test_two_fits_agree(data, forecasts):
    train, fresh = data
    again = lgbm_tail_surv.make_model(_matrix())
    assert again.name == "lgbm_tail_surv" and again.kappas == KAPPAS
    again.fit(train)
    pd.testing.assert_frame_equal(
        forecasts["lgbm_tail_surv"], again.predict(fresh, KAPPAS), check_exact=True
    )
    assert lgbm_surv.make_model(_matrix()).name == "lgbm_surv"


# --- 3. numbers where enough rows are left ---


@pytest.mark.parametrize("name", list(NEW))
def test_the_survival_columns_hold_numbers_until_no_training_car_is_left(
    data, fitted, forecasts, name
):
    train, _ = data
    departures = train["departure_exact"].astype(float).to_numpy()
    table = forecasts[name]
    for hour, column in enumerate(SURVIVAL, start=1):
        left = int((departures > hour).sum())
        assert fitted[name].bounds.rows[hour] == left
        if hour <= 7:
            assert left >= conditional_fit.MIN_ROWS
            assert table[column].notna().all()
        else:
            assert left == 0
            assert table[column].isna().all()


def _plain_table(departures) -> pd.DataFrame:
    """A training table of two feature columns and the departure label only."""
    n = len(departures)
    return pd.DataFrame(
        {"a": np.arange(n) % 3, "b": np.arange(n) % 5, "label_departure_h": departures}
    )


def test_the_minimum_is_twenty_rows_still_plugged_in_at_the_exact_count():
    assert conditional_fit.MIN_ROWS == MIN_CELL == COMMITTED["min_cell"] == 20
    # above 6: 21 rows; above 7 (7.0 itself is not): 20 rows; above 8: 19 rows
    table = _plain_table([6.5] * 30 + [7.0] * 5 + [7.5] + [8.5] * 19 + [math.nan] * 4)
    bounds = conditional_fit.ConditionalBounds((0.1,), ("a", "b")).fit(table)
    assert bounds.rows == {1: 55, 2: 55, 3: 55, 4: 55, 5: 55, 6: 55, 7: 20, 8: 19}
    x = lgbm.features(table.iloc[:6], ("a", "b"))
    assert not np.isnan(bounds.predict(x, 7, 0.1)).any()
    assert np.isnan(bounds.predict(x, 8, 0.1)).all()
    assert set(bounds.boosters) == {(h, 0.1) for h in range(1, 8)}
    with pytest.raises(ValueError, match="level"):
        bounds.predict(x, 7, 0.3)
    with pytest.raises(ValueError, match="hour"):
        bounds.predict(x, 9, 0.1)
    with pytest.raises(ValueError, match="hour"):
        conditional_fit.ConditionalBounds((0.1,), ("a", "b")).predict(x, 1, 0.1)


# --- 4. the conditioning ---


@pytest.mark.parametrize("name", list(NEW))
def test_a_car_seen_to_stay_gets_a_bound_in_the_late_group(forecasts, name):
    table = forecasts[name]
    low = table[table["kappa"] == 0.1]
    assert (low["d_hat"] < sf.EARLY[1]).all()  # given presence at τ, the early group decides
    for column in ("survival_h4", "survival_h5"):
        # every training car present then leaves after LATE[0]; a model fitted on all rows would
        # give about d_hat, raised to the hour's start (4 or 5)
        assert (table[column] > sf.LATE[0]).all(), column
        assert (table[column] < sf.LATE[1]).all()
        assert (table[column] > table["d_hat"]).all()
        assert (low[column] > low["d_hat"] + 2.0).all()


def test_each_hour_is_fitted_on_the_rows_still_plugged_in_only():
    # one feature value throughout; departures 0.5, 1.5, ..., 7.5, 40 rows each
    table = _plain_table(np.repeat(np.arange(8) + 0.5, 40))
    table["a"] = 0
    table["b"] = 0
    bounds = conditional_fit.ConditionalBounds((0.1, 0.3), ("a", "b")).fit(table)
    x = lgbm.features(table.iloc[:1], ("a", "b"))
    for hour in range(1, 8):
        assert bounds.rows[hour] == 40 * (8 - hour)
        for kappa in (0.1, 0.3):
            # the lowest departure still present is hour + 0.5; with all rows it would be below
            bound = float(bounds.predict(x, hour, kappa)[0])
            assert hour + 0.5 - 1e-6 <= bound <= 7.5, (hour, kappa, bound)
    assert math.isnan(float(bounds.predict(x, 8, 0.1)[0]))


# --- 5. no leak ---


@pytest.mark.parametrize("name", list(NEW))
def test_label_columns_handed_to_predict_change_nothing(data, fitted, forecasts, name):
    _, fresh = data
    labelled = fresh.assign(**{c: -7.0 * np.arange(len(fresh)) + 3.0 for c in LABEL_COLUMNS})
    pd.testing.assert_frame_equal(
        forecasts[name], fitted[name].predict(labelled, KAPPAS), check_exact=True
    )
    part = fresh.iloc[5:17].reset_index(drop=True)  # a car's bound does not depend on the others
    whole = forecasts[name]
    keys = set(part["session_id"])
    expected = whole[whole["session_id"].isin(keys)].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, fitted[name].predict(part, KAPPAS), check_exact=True)


@pytest.mark.parametrize("name", list(NEW))
def test_altering_the_outcomes_of_the_cars_forecast_does_not_change_their_forecast(fitted, name):
    last = DAYS - 1
    fakes, users = sf.two_stay_cars(DAYS)
    base = fitted[name].predict(_fresh(sf.cars_table(fakes, users), last), KAPPAS)
    assert len(base) == sf.PER_DAY * len(KAPPAS)
    changed = 0
    for k, fake in enumerate(fakes):
        if fake.date == date_of(last):
            fake.plug_out = 10.0 + (0.3 if fake.plug_out > 14.0 else 9.0) + 0.01 * k
            fake.energy = 17.0 + k
            fake.room = 0.5 + 0.001 * k
            fake.rate = 1.5
            changed += 1
    assert changed == sf.PER_DAY
    altered_cars = sf.cars_table(fakes, users)
    after = fitted[name].predict(_fresh(altered_cars, last), KAPPAS)
    pd.testing.assert_frame_equal(base, after, check_exact=True)
    assert after[SURVIVAL[:7]].notna().all().all()


# --- 6. at least the hour's start ---


class _Constant:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x, num_threads):
        return np.full(x.shape[0], self.value)


def test_the_bound_is_at_least_the_start_of_its_hour(forecasts):
    for table in forecasts.values():
        for hour, column in enumerate(SURVIVAL, start=1):
            values = table[column].dropna()
            assert (values >= hour).all()
    bounds = conditional_fit.ConditionalBounds((0.1,), ("a", "b"))
    bounds.rows = dict.fromkeys(conditional_fit.HOURS, 99)
    bounds.boosters = {(h, 0.1): _Constant(2.5) for h in conditional_fit.HOURS}
    x = np.zeros((3, 2))
    got = [bounds.predict(x, h, 0.1).tolist() for h in conditional_fit.HOURS]
    assert got == [[2.5] * 3, [2.5] * 3, *([float(h)] * 3 for h in range(3, 9))]
    assert bounds.predict(np.zeros((0, 2)), 3, 0.1).shape == (0,)


def test_with_bounds_fills_each_level_and_refuses_a_table_of_other_rows(data, fitted):
    _, fresh = data
    model = fitted["lgbm_surv"]
    x = lgbm.features(fresh, model.COLUMNS)
    table = lgbm.forecast_rows(
        fresh, KAPPAS, 0.0, lgbm.quantile_columns(model.quantiles, x, KAPPAS)
    )
    filled = conditional_fit.with_bounds(table, model.bounds, x, KAPPAS)
    assert table[SURVIVAL].isna().all().all()  # the table given is not changed
    for kappa in KAPPAS:
        rows = filled[filled["kappa"] == kappa]
        assert rows["session_id"].tolist() == fresh["session_id"].tolist()
        np.testing.assert_array_equal(
            rows["survival_h3"].to_numpy(), model.bounds.predict(x, 3, kappa)
        )
    low, high = (filled[filled["kappa"] == k]["survival_h2"].to_numpy() for k in KAPPAS)
    assert (low != high).any()
    with pytest.raises(ValueError, match="rows"):
        conditional_fit.with_bounds(table, model.bounds, x[:-1], KAPPAS)
    empty = model.predict(fresh.iloc[:0], KAPPAS)
    assert len(empty) == 0 and list(empty.columns) == list(table.columns)


# --- 7. the gate reads the bound ---


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    training = {
        unit: gate_cars(24, f"T{j}", early_every=3 + j) for j, unit in enumerate(TRAIN_UNITS)
    }
    run, tables = write_build(folder, {**small_gate_units(), **training})
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]


@pytest.mark.parametrize("name", list(NEW))
def test_learn_run_scores_the_model_end_to_end(build, tmp_path, monkeypatch, capsys, name):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = learn_run.main([*build, "--model", name])
    out = capsys.readouterr().out
    assert code == 0, out
    results = json.loads((tmp_path / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == name
    assert results["fit_units"] == list(TRAIN_UNITS)
    assert not results["failed_configurations"]
    table = read_table(tmp_path / learn_run.TABLE_NAME)
    cars = small_gate_units()
    assert len(table) == sum(len(cars[u]) for u in GATE_UNITS)
    # the training cars leave 1.3 to 6.8 hours after τ: all are present at hour 1, none at hour 7
    assert table["survival_h1"].notna().all() and (table["survival_h1"] >= 1.0).all()
    assert table[["survival_h7", "survival_h8"]].isna().all().all()
    for unit in GATE_UNITS:
        offer = results["units"][unit]["configurations"][PAIR[0]]
        assert offer["oracle_partner"]["name"] == PAIR[1]


def test_the_re_solve_of_a_made_up_day_reads_a_later_bound_than_under_lgbm_tail(tmp_path):
    train = sf.cars_table(*sf.three_stay_cars(TRAIN_DAYS))
    models = {}
    for make in (lgbm_tail.LgbmTailModel, lgbm_tail_surv.LgbmTailSurvModel):
        models[make.name] = make(rho=0.0, kappas=KAPPAS)
        models[make.name].fit(train)
    n = sf.THREE_SERVICE_HOURS
    unit = sf.write_unit(tmp_path, "dev", *sf.three_stay_cars(12, first_day=400), n)
    frame = feature_frame(build_table(unit.cars, rho=0.0))
    frame = frame[frame["counting"]].reset_index(drop=True)
    plain = TableForecaster(models["lgbm_tail"].predict(frame, KAPPAS))
    tightened = TableForecaster(models["lgbm_tail_surv"].predict(frame, KAPPAS))
    name = "main-rounded-hour-k0.3-empirical"  # the main candidate of the gate matrix
    (chosen,) = [c for c in COMMITTED["configurations"] if c["name"] == name]
    configuration = Configuration(**chosen)
    settings = gate_day.DaySettings(0.0, 5.0, 1e-6, 1e-6, 1e-6)
    offering = []
    for truth in unit.days:
        before = gate_day.prepare_day(truth, plain, configuration, settings)
        after = gate_day.prepare_day(truth, tightened, configuration, settings)
        # the forecast at τ is the same, so the steered cars and the certificate are
        assert before.steered == after.steered and before.value == after.value
        if before.abstains:
            continue
        offering.append(after)
        assert truth.day.n_service == n and after.departure_bounds.shape[1] == n + 1
        np.testing.assert_array_equal(before.departure_bounds[:, 0], after.departure_bounds[:, 0])
        # no tightening under lgbm_tail: the bound at τ holds in every hour
        assert (before.departure_bounds == before.departure_bounds[:, :1]).all()
        # at the start of hour 4 the first group of cars has left: the re-solve's bound is later
        assert after.departure_bounds[:, n].mean() > before.departure_bounds[:, n].mean() + 1.0
        assert (after.departure_bounds[:, n] >= n).all()
        for p, i in enumerate(after.steered):
            for t in range(1, n + 1):
                assert after.departure_bounds[p, t] == tightened.survival_bound(truth, i, t, 0.3)
    assert len(offering) >= 6
    evaluation = gate_day.evaluate(offering[0], 1.0)
    assert not evaluation.abstained and "one-way" in evaluation.rungs
