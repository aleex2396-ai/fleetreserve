"""The weighted fit of `models.lgbm_tail_equal_mass` (the equal-mass arm of `forecast_local`),
with LightGBM's data set and training replaced by recorders, so that no booster is trained.
Written, not run; it belongs to the learn lane because the module imports LightGBM.

1. Every booster the model fits (each quantity and level) gets a data set whose row weight is
   the weight column's value for the rows that quantity is fitted on, and nothing else about the
   data set differs from `QuantileSet`'s. Red if `weight=weights[keep]` is dropped, or if the
   weights are not cut to the same rows as the features and labels.
2. A fit frame with no weight column, or a weight that is missing, 0 or negative, is refused.
   Red if either check in `fit` is removed.
3. `lgbm_tail` itself still builds its data sets without a weight (it is not changed).
4. With every weight 1, `WeightedQuantileSet` and the unweighted `QuantileSet` give the same
   predictions at every quantity and level, on the same made-up rows with the same settings and
   seed, to `TOLERANCE` (1e-9). This is not assumed: LightGBM may treat a weighted quantile fit
   differently (a weighted-percentile leaf rule) even when every weight is 1. The test reports
   the largest absolute difference in its failure message. If it fails, the equal-mass arm of the
   trial differs from the unweighted arm by the fitting route as well as by the weights, and the
   difference must be read before any result of the trial is read. Neither arm is changed
   to force equality.
"""

import numpy as np
import pandas as pd
import pytest

from fleetreserve.forecast_local import WEIGHT_COLUMN
from fleetreserve.models import lgbm, lgbm_tail, lgbm_tail_equal_mass


class Recorder:
    def __init__(self):
        self.datasets = []
        self.trained = []

    def dataset(self, x, label=None, weight=None, **rest):
        found = {"x": x, "label": label, "weight": weight, **rest}
        self.datasets.append(found)
        return found

    def train(self, parameters, data, num_boost_round):
        self.trained.append((parameters["alpha"], data))
        return object()


@pytest.fixture
def recorder(monkeypatch):
    found = Recorder()
    monkeypatch.setattr(lgbm.lightgbm, "Dataset", found.dataset)
    monkeypatch.setattr(lgbm.lightgbm, "train", found.train)
    return found


def made_up_table(rows: int = 10) -> pd.DataFrame:
    table = pd.DataFrame({c: np.arange(rows, dtype=float) for c in lgbm_tail.COLUMNS})
    table["label_departure_h"] = np.linspace(1.0, 9.0, rows)
    table["label_rate_kw"] = np.linspace(2.0, 3.0, rows)
    table["label_room_kwh"] = np.linspace(1.0, 4.0, rows)
    table["label_need_kwh"] = np.linspace(1.0, 4.0, rows)
    table.loc[[1, 2], ["label_rate_kw", "label_room_kwh", "label_need_kwh"]] = np.nan  # no fleet
    return table


def test_every_booster_gets_the_weights_of_the_rows_it_is_fitted_on(recorder):
    table = made_up_table()
    weights = np.arange(1.0, 11.0)
    levels = lgbm.levels_of([0.1, 0.5])
    lgbm_tail_equal_mass.WeightedQuantileSet(levels, lgbm_tail.COLUMNS).fit(table, weights)
    expected = sum(len(v) for v in levels.values())
    assert len(recorder.datasets) == len(recorder.trained) == expected
    for found in recorder.datasets:
        rows = len(found["label"])
        assert rows in (8, 10)  # the quantities with a fleet label lose rows 1 and 2
        keep = np.ones(10, dtype=bool)
        if rows == 8:
            keep[[1, 2]] = False
        assert np.array_equal(found["weight"], weights[keep])


def test_the_model_refuses_a_frame_without_usable_weights(recorder):
    model = lgbm_tail_equal_mass.LgbmTailEqualMassModel(rho=0.0, kappas=[0.5])
    with pytest.raises(ValueError, match="weight column"):
        model.fit(pd.DataFrame({"unit": ["a"]}))
    for bad in (np.nan, 0.0, -1.0, np.inf):
        with pytest.raises(ValueError, match="above 0"):
            model.fit(pd.DataFrame({"unit": ["a", "b"], WEIGHT_COLUMN: [1.0, bad]}))
    assert recorder.datasets == []


TOLERANCE = 1e-9


def random_table(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    table = pd.DataFrame({c: rng.normal(size=rows) for c in lgbm_tail.COLUMNS})
    table["label_departure_h"] = np.abs(rng.normal(5.0, 2.0, size=rows)) + 0.1
    table["label_rate_kw"] = np.abs(rng.normal(3.0, 1.0, size=rows)) + 0.1
    table["label_room_kwh"] = np.abs(rng.normal(8.0, 3.0, size=rows)) + 0.1
    table["label_need_kwh"] = table["label_room_kwh"]
    gone = rng.choice(rows, size=rows // 10, replace=False)
    table.loc[gone, ["label_rate_kw", "label_room_kwh", "label_need_kwh"]] = np.nan
    return table


def test_all_weights_one_give_the_predictions_of_the_unweighted_fit():
    table = random_table()
    levels = lgbm.levels_of([0.1, 0.5])
    plain = lgbm.QuantileSet(levels, lgbm_tail.COLUMNS).fit(table)
    weighted = lgbm_tail_equal_mass.WeightedQuantileSet(levels, lgbm_tail.COLUMNS).fit(
        table, np.ones(len(table))
    )
    x = lgbm.features(table, lgbm_tail.COLUMNS)
    assert set(plain.boosters) == set(weighted.boosters)
    differences = {
        key: float(np.max(np.abs(plain.predict(x, *key) - weighted.predict(x, *key))))
        for key in plain.boosters
    }
    worst = max(differences, key=differences.get)
    print(f"largest absolute difference {differences[worst]!r} at {worst}")
    assert differences[worst] <= TOLERANCE, (
        f"all-weights-one fit differs from the unweighted fit: largest absolute difference "
        f"{differences[worst]!r} at (quantity, level) {worst}; every difference {differences}"
    )


def test_lgbm_tail_still_builds_its_data_sets_without_a_weight(recorder):
    levels = lgbm.levels_of([0.5])
    lgbm.QuantileSet(levels, lgbm_tail.COLUMNS).fit(made_up_table())
    assert recorder.datasets and all(d["weight"] is None for d in recorder.datasets)
