"""The pooled LightGBM (`models.lgbm`) and CQR-GBM (`models.cqr`) departure models on made-up cars
tables with a known conditional distribution (`quantile_fake`): 300 fitting days of a training
unit, 300 fresh days of a site never seen in fitting, 12 cars a day, levels 0.05, 0.1, 0.2, 0.3.

1. Pooled LightGBM: on the fresh days the share of cars below each fitted bound (above it for
   `e_need`, a (1 - kappa)-quantile) is within `COVERAGE_TOLERANCE` of the level, and each fitted
   quantile is within `ERROR_SHARE` of the spread of its uniform draw of the known conditional
   quantile (mean absolute error), so a model that ignored the arrival hour would fail.
2. CQR-GBM: the share of fresh cars that leave before the adjusted bound is at most the level
   plus `CQR_ABOVE` and at least the level minus `CQR_BELOW`. With every fitted departure quantile
   moved 3 hours later, CQR still reaches it, and the same moved quantiles without the adjustment
   do not.
3. The calibration split takes whole days of the training rows, shares no day and no session with
   the fit split, holds a quarter of the days, and is the same in two fits; the adjustment's rank
   is `ceil((1 - kappa) * (n + 1))`, and a split too small for a level is refused.
4. Two fits give the same forecast table, value for value.
5. A site and unit never seen in fitting are forecast, with the same numbers as a training site
   with the same features.
6. `predict` reads only the keys and features: it runs on a frame with no label or outcome
   column, and changing the labels changes nothing.
7. Both declare the training units as `fit_units`, which `learn_run` accepts; the survival
   columns are NaN (`lgbm.SURVIVAL_BOUND`); a negative prediction is raised to 0.
"""

import math

import numpy as np
import pandas as pd
import pytest
from quantile_fake import cars, unit_cars

from fleetreserve import learn_run
from fleetreserve.forecast_model import KEY_COLUMNS, SURVIVAL_COLUMNS, check_table
from fleetreserve.learn_features import FEATURE_COLUMNS, LABEL_COLUMNS, build_table, feature_frame
from fleetreserve.models import cqr, lgbm
from fleetreserve.truth_build import TRAIN_UNITS

KAPPAS = (0.05, 0.1, 0.2, 0.3)
COVERAGE_TOLERANCE = 0.05
ERROR_SHARE = 0.1
CQR_ABOVE = 0.02
CQR_BELOW = 0.03
SPREAD = {"d_hat": 4.0, "r_hat": 2.0, "e_cap": 3.0, "e_need": 3.0}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    folder = tmp_path_factory.mktemp("quantile")
    train = unit_cars(folder, "caltech", cars(300, seed=1))
    fresh = unit_cars(folder, "fresh", cars(300, seed=2, site="F", first_day=400))
    return train, build_table(fresh, rho=0.0)


@pytest.fixture(scope="module")
def fitted(data):
    train, _ = data
    out = {}
    for model in (lgbm.LgbmModel(rho=0.0, kappas=KAPPAS), cqr.CqrModel(rho=0.0, kappas=KAPPAS)):
        model.fit(train)
        out[model.name] = model
    return out


def _known_quantile(column: str, arrival: np.ndarray, kappa: float) -> np.ndarray:
    a = arrival - 6
    if column == "d_hat":
        return 2.0 + 1.5 * a + 4.0 * kappa
    if column == "r_hat":
        return 3.0 + 0.5 * a + 2.0 * kappa
    if column == "e_cap":
        return np.full(a.shape, 1.0 + 3.0 * kappa)
    return np.full(a.shape, 1.0 + 3.0 * (1 - kappa))


def _truth(fresh: pd.DataFrame) -> dict:
    return {
        "d_hat": fresh["label_departure_h"].to_numpy(float),
        "r_hat": fresh["label_rate_kw"].to_numpy(float),
        "e_cap": fresh["label_room_kwh"].to_numpy(float),
        "e_need": fresh["label_need_kwh"].to_numpy(float),
    }


def _at(table: pd.DataFrame, kappa: float) -> pd.DataFrame:
    return table[table["kappa"] == kappa].reset_index(drop=True)


def test_fitted_quantiles_cover_their_levels(data, fitted):
    _, fresh = data
    table = check_table(fitted["lgbm"].predict(feature_frame(fresh), KAPPAS))
    truth = _truth(fresh)
    arrival = fresh["arrival_hour"].to_numpy(float)
    for kappa in KAPPAS:
        part = _at(table, kappa)
        for column, y in truth.items():
            q = part[column].to_numpy(float)
            miss = float((y > q).mean() if column == "e_need" else (y < q).mean())
            assert abs(miss - kappa) <= COVERAGE_TOLERANCE, (kappa, column, miss)
            error = float(np.abs(q - _known_quantile(column, arrival, kappa)).mean())
            assert error <= ERROR_SHARE * SPREAD[column], (kappa, column, error)
    below_median = float((truth["d_hat"] < _at(table, 0.1)["d_median"].to_numpy()).mean())
    assert abs(below_median - 0.5) <= COVERAGE_TOLERANCE


def _departure_misses(model, fresh) -> dict:
    table = model.predict(feature_frame(fresh), KAPPAS)
    y = fresh["label_departure_h"].to_numpy(float)
    return {k: float((y < _at(table, k)["d_hat"].to_numpy(float)).mean()) for k in KAPPAS}


def _reaches(misses: dict) -> bool:
    return all(k - CQR_BELOW <= m <= k + CQR_ABOVE for k, m in misses.items())


def test_cqr_reaches_the_target_coverage_on_fresh_days(data, fitted):
    _, fresh = data
    misses = _departure_misses(fitted["cqr"], fresh)
    assert _reaches(misses), misses


def test_cqr_corrects_a_quantile_model_that_is_3_hours_late(data, monkeypatch):
    train, fresh = data
    honest = lgbm.QuantileSet.predict

    def late(self, x, quantity, level):
        values = honest(self, x, quantity, level)
        return values + 3.0 if quantity == "departure" else values

    monkeypatch.setattr(lgbm.QuantileSet, "predict", late)
    moved = lgbm.LgbmModel(rho=0.0, kappas=KAPPAS)
    moved.fit(train)
    assert all(m > k + 0.2 for k, m in _departure_misses(moved, fresh).items())
    model = cqr.CqrModel(rho=0.0, kappas=KAPPAS)
    model.fit(train)
    misses = _departure_misses(model, fresh)
    assert _reaches(misses), misses


def test_the_calibration_split_takes_whole_days_apart_from_the_fit_split(data, fitted):
    train, _ = data
    model = fitted["cqr"]
    fit_days, calibration_days = model.split_days
    assert not fit_days & calibration_days
    assert len(calibration_days) == 75 and len(fit_days) == 225
    table = build_table(train, rho=0.0)
    fit, calibration = cqr.split(table)
    assert set(zip(fit["unit"], fit["site"], fit["date"], strict=True)) == fit_days
    in_both = set(fit["session_id"]) & set(calibration["session_id"])
    assert not in_both
    again = cqr.CqrModel(rho=0.0, kappas=KAPPAS)
    again.fit(train)
    assert again.split_days == model.split_days


def test_a_session_on_several_days_is_not_scored_in_calibration_after_fitting():
    rows = [
        {"unit": "u", "site": "s", "date": f"2020-01-0{day + 1}", "session_id": f"d{day}c{car}"}
        for day in range(8)
        for car in range(3)
    ]
    rows += [
        {"unit": "u", "site": "s", "date": f"2020-01-0{day + 1}", "session_id": "long"}
        for day in range(8)
    ]
    table = pd.DataFrame(rows)
    fit, calibration = cqr.split(table)
    assert len(cqr.calibration_days(table)) == 2
    assert "long" in set(fit["session_id"]) and "long" not in set(calibration["session_id"])
    assert len(calibration) == 2 * 3
    assert len(fit) == 6 * 4


def test_the_conformal_rank_and_a_split_too_small():
    assert cqr.conformal_rank(0.05, 19) == 19
    assert cqr.conformal_rank(0.1, 100) == 91
    assert cqr.conformal_rank(0.3, 9) == 7
    scores = np.arange(19, dtype=float)[::-1]
    assert cqr.adjustment(scores, 0.05) == 18.0
    assert cqr.adjustment(np.arange(100, dtype=float), 0.1) == 90.0
    with pytest.raises(ValueError, match="needs at least 19"):
        cqr.adjustment(np.zeros(18), 0.05)


@pytest.mark.parametrize("name", ["lgbm", "cqr"])
def test_two_fits_give_the_same_forecasts(data, fitted, name):
    train, fresh = data
    model = (lgbm.LgbmModel if name == "lgbm" else cqr.CqrModel)(rho=0.0, kappas=KAPPAS)
    model.fit(train)
    frame = feature_frame(fresh)
    pd.testing.assert_frame_equal(
        model.predict(frame, KAPPAS), fitted[name].predict(frame, KAPPAS), check_exact=True
    )


@pytest.mark.parametrize("name", ["lgbm", "cqr"])
def test_a_site_never_seen_in_fitting_is_forecast(data, fitted, name):
    train, fresh = data
    model = fitted[name]
    seen = build_table(train, rho=0.0).iloc[:48]
    unseen = seen.assign(unit="norway-development-pooled", site="never-seen")
    a = check_table(model.predict(feature_frame(seen), KAPPAS))
    b = check_table(model.predict(feature_frame(unseen), KAPPAS))
    assert set(b["site"]) == {"never-seen"}
    numbers = ["d_hat", "r_hat", "e_cap", "e_need", "d_median"]
    np.testing.assert_array_equal(a[numbers].to_numpy(), b[numbers].to_numpy())
    assert not np.isnan(b[numbers].to_numpy()).any()


@pytest.mark.parametrize("name", ["lgbm", "cqr"])
def test_predict_reads_only_keys_and_features(data, fitted, name):
    _, fresh = data
    model = fitted[name]
    bare = fresh[[*KEY_COLUMNS, "counting", *FEATURE_COLUMNS]]
    assert not set(bare.columns) & set(LABEL_COLUMNS)
    from_bare = model.predict(bare, KAPPAS)
    changed = fresh.copy()
    for column in LABEL_COLUMNS:
        changed[column] = -changed[column] * 7.0 + 100.0
    pd.testing.assert_frame_equal(from_bare, model.predict(changed, KAPPAS), check_exact=True)


@pytest.mark.parametrize("name", ["lgbm", "cqr"])
def test_fitted_on_training_units_only_with_no_survival_tightening(data, fitted, name):
    model = fitted[name]
    assert model.fit_units == TRAIN_UNITS
    learn_run.check_model(model.name, model)
    assert getattr(model, "reads_other_day_outcomes", False) is False
    assert lgbm.SURVIVAL_BOUND == lgbm.NO_SURVIVAL_BOUND
    _, fresh = data
    table = model.predict(feature_frame(fresh.iloc[:24]), KAPPAS)
    assert len(table) == 24 * len(KAPPAS)
    assert table[list(SURVIVAL_COLUMNS)].isna().all().all()
    assert set(table["rho"]) == {0.0}


def test_a_negative_prediction_is_raised_to_zero():
    class _Below:
        def predict(self, x, num_threads):
            return np.full(x.shape[0], -1.5)

    quantiles = lgbm.QuantileSet({"departure": (0.1,)})
    quantiles.boosters[("departure", 0.1)] = _Below()
    values = quantiles.predict(np.zeros((3, len(FEATURE_COLUMNS))), "departure", 0.1)
    assert values.tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="no rate model"):
        quantiles.predict(np.zeros((1, len(FEATURE_COLUMNS))), "rate", 0.1)
    assert math.isclose(lgbm.levels_of([0.05])["need"][0], 0.95)
