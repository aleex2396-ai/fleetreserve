"""The covariate rule of `models.survival_regression` on a hand-made table of `FEATURE_COLUMNS`.

1. A column with a missing value in training gets a `_missing` column; a missing value takes the
   training median; a column with one value only, or no number, is dropped (its `_missing` column
   too when that has one value); kept columns are standardised with the training mean and
   population standard deviation, worked out by hand below.
2. The fitted rule is applied unchanged to another frame: its missing values take the training
   median, its own spread is not used, and no column but `FEATURE_COLUMNS` is read.
3. An infinite training value, and a training duration that is not above 0, raise `ValueError`.
"""

import math

import numpy as np
import pandas as pd
import pytest
from survival_regression_fake import weibull_cars

from fleetreserve.learn_features import FEATURE_COLUMNS
from fleetreserve.models import survival_regression as sr


def _table() -> pd.DataFrame:
    table = pd.DataFrame({c: [5.0, 5.0, 5.0, 5.0] for c in FEATURE_COLUMNS})
    table["arrival_hour"] = [6.0, 7.0, 8.0, 11.0]  # mean 8, sd sqrt(3.5)
    table["history_mean_stay_h"] = [1.0, math.nan, 3.0, 8.0]  # median 3
    table["history_median_stay_h"] = [math.nan] * 4  # no number: dropped
    table["fleet_at_tau"] = [2.0, 2.0, math.nan, 2.0]  # one value: dropped, its flag kept
    table["label_departure_h"] = [1.0, 2.0, 3.0, 4.0]
    return table


def test_the_rule_worked_out_by_hand():
    rule = sr.Covariates.fit(_table())
    assert rule.names == (
        "arrival_hour",
        "fleet_at_tau_missing",
        "history_mean_stay_h",
        "history_mean_stay_h_missing",
    )
    assert rule.medians["history_mean_stay_h"] == 3.0
    x = rule.transform(_table())
    np.testing.assert_allclose(x["arrival_hour"], (np.array([6, 7, 8, 11]) - 8) / math.sqrt(3.5))
    stay = np.array([1.0, 3.0, 3.0, 8.0])
    np.testing.assert_allclose(x["history_mean_stay_h"], (stay - 3.75) / stay.std())
    flag = np.array([0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(x["history_mean_stay_h_missing"], (flag - 0.25) / flag.std())
    fleet_flag = np.array([0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(x["fleet_at_tau_missing"], (fleet_flag - 0.25) / fleet_flag.std())


def test_the_fitted_rule_is_applied_unchanged_and_reads_only_the_features():
    rule = sr.Covariates.fit(_table())
    other = pd.DataFrame({c: [5.0, 5.0] for c in FEATURE_COLUMNS})
    other["arrival_hour"] = [8.0, 8.0]
    other["history_mean_stay_h"] = [math.nan, 9.0]
    other["fleet_at_tau"] = [2.0, 7.0]
    other["label_departure_h"] = [100.0, -100.0]
    other["counting"] = [True, False]
    x = rule.transform(other)
    assert list(x.columns) == list(rule.names)
    np.testing.assert_allclose(x["arrival_hour"], [0.0, 0.0])
    stay = np.array([1.0, 3.0, 3.0, 8.0])
    np.testing.assert_allclose(x["history_mean_stay_h"], (np.array([3.0, 9.0]) - 3.75) / stay.std())
    flag = np.array([0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        x["history_mean_stay_h_missing"], (np.array([1.0, 0.0]) - 0.25) / flag.std()
    )
    bare = rule.transform(other[list(FEATURE_COLUMNS)])
    pd.testing.assert_frame_equal(bare, x, check_exact=True)


def test_an_infinite_value_or_a_duration_not_above_zero_raises():
    table = _table()
    table.loc[0, "arrival_hour"] = math.inf
    with pytest.raises(ValueError, match="infinite"):
        sr.Covariates.fit(table)
    cars = weibull_cars(3, seed=6)
    cars.loc[0, "departure_exact"] = 0.0
    with pytest.raises(ValueError, match="above 0"):
        sr.training_rows(cars, 0.0)
