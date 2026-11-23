"""The Cox (`models.cox`) and Weibull AFT (`models.weibull_aft`) departure models on made-up cars
with Weibull departures (`survival_regression_fake`): 300 fitting days of a training unit, 200
fresh days of a site never seen in fitting, 12 cars a day, levels 0.05, 0.1, 0.2, 0.3.

1. Recovery: on the fresh cars, `d_hat`, `d_median` and every `survival_h<t>` that is a number are
   within `MEAN_ERROR` of the true quantile on average and within `MAX_ERROR` for every car
   (relative error), for both models.
2. A later arrival hour, which lengthens the true stay, gives a later mean `d_hat`.
3. Weibull AFT: `d_hat` and `d_median` equal lifelines' own percentiles of the fitted model; each
   `survival_h<t>` is where lifelines' fitted survival, divided by its value at t, reaches
   1 - kappa; it is NaN exactly where that survival at t is below 1e-3; it is above t.
4. Cox: every departure column equals `forecast_model.survival_quantile` of lifelines' fitted
   survival curve of the car, with the same 1e-3 rule; every `survival_h<t>` is above t.
5. Missing values: fitted on cars of which some have no history, the `_missing` columns are
   covariates; a car with missing history gets a forecast, and a different one from the same car
   with its history set to the training medians.
6. `predict` reads only the keys and `FEATURE_COLUMNS`: corrupting every other column, adding
   label and outcome columns, renaming the site and unit, or dropping every other column changes
   no number of the table.
7. Two fits on the same rows give the same table, bit for bit; the table passes
   `forecast_model.check_table`; `r_hat`, `e_cap`, `e_need` equal those of `models.km`.
8. A fit that does not converge raises `ValueError`.
9. Both declare the training units as `fit_units`, which `learn_run` accepts.
"""

import math

import numpy as np
import pandas as pd
import pytest
from survival_regression_fake import true_quantile, weibull_cars

from fleetreserve import learn_run
from fleetreserve.forecast_model import (
    KEY_COLUMNS,
    NUMBER_COLUMNS,
    SURVIVAL_COLUMNS,
    check_table,
    survival_quantile,
)
from fleetreserve.learn_features import (
    FEATURE_COLUMNS,
    HISTORY_COLUMNS,
    LABEL_COLUMNS,
    build_table,
    feature_frame,
)
from fleetreserve.models import cox, km, weibull_aft
from fleetreserve.truth_build import TRAIN_UNITS

KAPPAS = (0.05, 0.1, 0.2, 0.3)
MIN_CELL = 30
MEAN_ERROR = 0.06
MAX_ERROR = 0.12
MODELS = {"cox": cox.CoxModel, "weibull_aft": weibull_aft.WeibullAftModel}
CHECKED_CARS = 48
PRESENCE_FLOOR = 1e-3  # the rule of `models.hazard`, written here, not read from the code


def _fitted(name: str, train: pd.DataFrame):
    model = MODELS[name](rho=0.0, min_cell=MIN_CELL)
    model.fit(train)
    return model


@pytest.fixture(scope="module")
def data():
    train = weibull_cars(300, seed=1)
    fresh = feature_frame(build_table(weibull_cars(200, seed=2, site="F", first_day=400), rho=0.0))
    return train, fresh


@pytest.fixture(scope="module")
def fitted(data):
    train, fresh = data
    out = {}
    for name in MODELS:
        model = _fitted(name, train)
        out[name] = (model, model.predict(fresh, KAPPAS))
    return out


def _block(table: pd.DataFrame, kappa: float) -> pd.DataFrame:
    return table[table["kappa"] == kappa].reset_index(drop=True)


@pytest.mark.parametrize("name", MODELS)
def test_quantiles_recover_the_true_ones(data, fitted, name):
    _, fresh = data
    _, table = fitted[name]
    arrival = fresh["arrival_hour"].to_numpy(float)
    checked = 0
    for kappa in KAPPAS:
        block = _block(table, kappa)
        columns = {"d_hat": (kappa, 0.0), "d_median": (0.5, 0.0)}
        columns |= {c: (kappa, float(t)) for t, c in enumerate(SURVIVAL_COLUMNS, start=1)}
        for column, (level, after) in columns.items():
            found = block[column].to_numpy(float)
            number = ~np.isnan(found)
            assert number.mean() > 0.5, column
            error = np.abs(found[number] / true_quantile(arrival[number], level, after) - 1)
            assert error.mean() <= MEAN_ERROR, (column, kappa, error.mean())
            assert error.max() <= MAX_ERROR, (column, kappa, error.max())
            checked += int(number.sum())
    assert checked > 0


@pytest.mark.parametrize("name", MODELS)
def test_a_later_arrival_gives_a_later_forecast(data, fitted, name):
    _, fresh = data
    _, table = fitted[name]
    for kappa in KAPPAS:
        block = _block(table, kappa)
        means = block.groupby(fresh["arrival_hour"].to_numpy(int))["d_hat"].mean()
        assert list(means.index) == [6, 7, 8, 9]
        assert (np.diff(means.to_numpy()) > 0).all(), means


def _survival_at(fitter, x: pd.DataFrame, times) -> np.ndarray:
    """lifelines' fitted survival of each row of `x` at each time: (rows, times)."""
    return fitter.predict_survival_function(x, times=list(times)).to_numpy(float).T


def test_weibull_columns_follow_lifelines_fitted_curve(data, fitted):
    _, fresh = data
    model, table = fitted["weibull_aft"]
    cars = fresh.iloc[:CHECKED_CARS]
    x = model.covariates.transform(cars)
    hours = np.arange(1, len(SURVIVAL_COLUMNS) + 1, dtype=float)
    at_hours = _survival_at(model.fitter, x, hours)
    median = model.fitter.predict_percentile(x, p=0.5).to_numpy(float)
    numbers = nans = 0
    for kappa in KAPPAS:
        block = _block(table, kappa).iloc[:CHECKED_CARS]
        lifelines_q = model.fitter.predict_percentile(x, p=1 - kappa).to_numpy(float)
        np.testing.assert_allclose(block["d_hat"], lifelines_q, rtol=1e-9)
        np.testing.assert_allclose(block["d_median"], median, rtol=1e-9)
        for t, column in enumerate(SURVIVAL_COLUMNS):
            found = block[column].to_numpy(float)
            low = at_hours[:, t] < PRESENCE_FLOOR
            assert (np.isnan(found) == low).all(), column
            for i in np.flatnonzero(~low):
                assert found[i] > hours[t]
                ratio = _survival_at(model.fitter, x.iloc[[i]], [found[i]])[0, 0] / at_hours[i, t]
                assert ratio == pytest.approx(1 - kappa, rel=1e-9, abs=1e-12)
            numbers += int((~low).sum())
            nans += int(low.sum())
    assert numbers > 0 and nans > 0


def test_cox_columns_equal_survival_quantile_of_lifelines_fitted_curve(data, fitted):
    _, fresh = data
    model, table = fitted["cox"]
    cars = fresh.iloc[:CHECKED_CARS]
    x = model.covariates.transform(cars)
    times = model.fitter.baseline_survival_.index.to_numpy(float)
    curves = _survival_at(model.fitter, x, times)
    numbers = nans = 0
    for kappa in KAPPAS:
        block = _block(table, kappa).iloc[:CHECKED_CARS]
        for i in range(len(cars)):
            curve = curves[i]
            expected = {
                "d_hat": survival_quantile(times, curve, kappa),
                "d_median": survival_quantile(times, curve, 0.5),
            }
            for t, column in enumerate(SURVIVAL_COLUMNS, start=1):
                before = np.flatnonzero(times <= t)
                present = 1.0 if before.size == 0 else curve[before[-1]]
                if present < PRESENCE_FLOOR:
                    expected[column] = math.nan
                    nans += 1
                else:
                    expected[column] = survival_quantile(times, curve, kappa, after=t)
                    numbers += 1
                    assert block.loc[i, column] > t
            for column, value in expected.items():
                found = block.loc[i, column]
                assert (math.isnan(found) and math.isnan(value)) or found == value, (column, i)
    assert numbers > 0 and nans > 0


@pytest.fixture(scope="module")
def with_missing():
    train = weibull_cars(120, seed=3, repeat_users=6, repeat_stretch=2.0)
    fresh = feature_frame(
        build_table(
            weibull_cars(10, seed=4, site="F", first_day=400, repeat_users=6, repeat_stretch=2.0),
            rho=0.0,
        )
    )
    return train, fresh


@pytest.mark.parametrize("name", MODELS)
def test_a_missing_value_gets_a_forecast_and_its_indicator_is_used(with_missing, name):
    train, fresh = with_missing
    model = _fitted(name, train)
    assert "history_mean_stay_h_missing" in model.covariates.names
    missing = fresh[fresh["history_mean_stay_h"].isna()].reset_index(drop=True)
    assert len(missing) > 0
    filled = missing.copy()
    for column in HISTORY_COLUMNS[1:]:
        assert filled[column].isna().all()
        filled[column] = model.covariates.medians[column]
    as_missing = model.predict(missing, KAPPAS)
    as_filled = model.predict(filled, KAPPAS)
    assert np.isfinite(as_missing["d_hat"]).all()
    assert (as_missing["d_hat"].to_numpy() != as_filled["d_hat"].to_numpy()).any()


def _corrupted(fresh: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    out = fresh.copy()
    for column in out.columns:
        if column in FEATURE_COLUMNS or column in KEY_COLUMNS:
            continue
        out[column] = rng.normal(size=len(out)) * 100.0
    for column in (*LABEL_COLUMNS, "departure_exact", "plug_out", "in_fleet", "room"):
        out[column] = rng.normal(size=len(out)) * 100.0
    out["site"] = "another-site"
    out["unit"] = "another-unit"
    return out


@pytest.mark.parametrize("name", MODELS)
def test_predict_reads_only_the_keys_and_feature_columns(data, fitted, name):
    _, fresh = data
    model, table = fitted[name]
    numbers = list(NUMBER_COLUMNS)
    corrupted = model.predict(_corrupted(fresh), KAPPAS)
    pd.testing.assert_frame_equal(corrupted[numbers], table[numbers], check_exact=True)
    bare = model.predict(fresh[[*KEY_COLUMNS, *FEATURE_COLUMNS]], KAPPAS)
    pd.testing.assert_frame_equal(bare, table, check_exact=True)


@pytest.mark.parametrize("name", MODELS)
def test_two_fits_give_the_same_table_and_it_passes_check_table(data, fitted, name):
    train, fresh = data
    _, table = fitted[name]
    again = _fitted(name, train).predict(fresh, KAPPAS)
    pd.testing.assert_frame_equal(again, table, check_exact=True)
    checked = check_table(table)
    assert len(checked) == len(fresh) * len(KAPPAS)
    reference = km.KMModel(rho=0.0, min_cell=MIN_CELL)
    reference.fit(train)
    energy = ["r_hat", "e_cap", "e_need"]
    expected = reference.predict(fresh, KAPPAS)
    pd.testing.assert_frame_equal(table[energy], expected[energy], check_exact=True)


def test_a_fit_that_does_not_converge_raises(monkeypatch):
    train = weibull_cars(20, seed=5)
    monkeypatch.setattr(cox, "FIT_OPTIONS", {"max_steps": 1})
    with pytest.raises(ValueError, match="did not converge"):
        _fitted("cox", train)
    monkeypatch.setattr(weibull_aft, "FIT_OPTIONS", {"maxiter": 1})
    with pytest.raises(ValueError, match="did not converge"):
        _fitted("weibull_aft", train)


@pytest.mark.parametrize("name", MODELS)
def test_both_are_fitted_on_the_training_units(name):
    model = MODELS[name](rho=0.0, min_cell=MIN_CELL)
    assert model.name == name
    assert model.fit_units == TRAIN_UNITS
    learn_run.check_model(name, model)  # raises Refused otherwise
    with pytest.raises(ValueError, match="before fit"):
        model.predict(pd.DataFrame(), KAPPAS)
