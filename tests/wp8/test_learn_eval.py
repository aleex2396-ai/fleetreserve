"""The WP8 scores (`learn_eval`): pinball loss and concordance index against hand-computed values,
NaN left out, and the per-unit scores of the empirical model's table on a made-up unit.
"""

import math

import numpy as np
import pytest
from learn_fake import odd_unit

from fleetreserve import learn_eval
from fleetreserve.forecast_empirical import KAPPAS, EmpiricalForecaster
from fleetreserve.models.empirical import EmpiricalModel


def test_pinball_loss_by_hand():
    # level 0.25: y - q = -1 costs 0.75, y - q = 2 costs 0.5
    assert learn_eval.pinball_loss([1.0, 4.0], [2.0, 2.0], 0.25) == pytest.approx(0.625)
    # level 0.9: the same errors cost 0.1 and 1.8
    assert learn_eval.pinball_loss([1.0, 4.0], [2.0, 2.0], 0.9) == pytest.approx(0.95)
    assert learn_eval.pinball_loss([1.0, 4.0, 9.0], [2.0, 2.0, math.nan], 0.25) == pytest.approx(
        0.625
    )
    assert learn_eval.pinball_loss([1.0], [math.nan], 0.5) is None
    assert learn_eval.pinball_loss([3.0], [3.0], 0.1) == 0.0
    with pytest.raises(ValueError):
        learn_eval.pinball_loss([1.0], [1.0], 1.0)


def test_concordance_index_by_hand():
    # six pairs; only (2, 3) is in the wrong order
    assert learn_eval.concordance_index([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(5 / 6)
    # a tie in the scores counts one half
    assert learn_eval.concordance_index([1, 2, 3, 4], [1, 1, 2, 3]) == pytest.approx(5.5 / 6)
    # a tie in the times is not a comparable pair: two pairs, one in order
    assert learn_eval.concordance_index([1, 1, 2], [5, 0, 1]) == pytest.approx(0.5)
    # reversed order: 0
    assert learn_eval.concordance_index([1, 2, 3], [3, 2, 1]) == 0.0
    # a car without a score is left out
    assert learn_eval.concordance_index([1, 2, 3], [1, math.nan, 2]) == 1.0
    assert learn_eval.concordance_index([1, 1], [1, 2]) is None
    assert learn_eval.concordance_index([1, 2], [math.nan, math.nan]) is None


def test_concordance_index_in_chunks_equals_all_at_once(monkeypatch):
    rng = np.random.default_rng(3)
    times = rng.integers(0, 20, 300).astype(float)
    scores = np.round(times + rng.normal(0, 4, 300))
    whole = learn_eval.concordance_index(times, scores)
    monkeypatch.setattr(learn_eval, "CONCORDANCE_CELLS", 1000)
    assert learn_eval.concordance_index(times, scores) == whole
    pairs = [(i, j) for i in range(300) for j in range(300) if times[i] < times[j]]
    agree = sum(scores[i] < scores[j] for i, j in pairs)
    ties = sum(scores[i] == scores[j] for i, j in pairs)
    assert whole == pytest.approx((agree + 0.5 * ties) / len(pairs))


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    unit = odd_unit(tmp_path_factory.mktemp("odd"))
    table = EmpiricalModel(rho=0.0, reference="other-days", min_cell=20).predict(unit.cars, KAPPAS)
    return unit, table, learn_eval.unit_scores(unit, table, KAPPAS)


def test_unit_scores_of_the_empirical_table(scored):
    unit, _, scores = scored
    assert set(scores) == {
        "0.05",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
    }  # the six levels of plan section 2
    empirical = EmpiricalForecaster(unit)
    for kappa in KAPPAS:
        truth, d_hat, rate, need = [], [], [], []
        for day in unit.days:
            for i, car in enumerate(day.fleet):
                truth.append(car.departure_exact)
                found = empirical.forecast_car(day, i, kappa, "empirical")
                d_hat.append(math.nan if found is None else found.departure)
                rate.append(math.nan if found is None else found.rate)
                need.append(math.nan if found is None else found.need)
        entry = scores[repr(kappa)]
        assert entry["cars"] == len(truth)
        assert entry["cars_with_departure_forecast"] == len(truth) - 3  # site Z has no reference
        u = np.array(truth) - np.array(d_hat)
        keep = ~np.isnan(u)
        expected = np.mean(np.maximum(kappa * u[keep], (kappa - 1) * u[keep]))
        assert entry["pinball_loss"]["departure_hours"]["mean"] == pytest.approx(expected)
        rates = [c.rate for d in unit.days for c in d.day.cars]
        assert entry["pinball_loss"]["rate_kw"]["mean"] == pytest.approx(
            learn_eval.pinball_loss(rates, rate, kappa)
        )
        rooms = [c.room for d in unit.days for c in d.day.cars]
        assert entry["pinball_loss"]["need_kwh"] == {
            "level": 1 - kappa,
            "cars_scored": len(truth) - 3,
            "mean": learn_eval.pinball_loss(rooms, need, 1 - kappa),
        }
        assert 0.5 < entry["concordance_index"]["value"] <= 1.0


def test_a_car_missing_from_the_table_raises(scored):
    unit, table, _ = scored
    car = unit.days[0].fleet[0]
    rows = (table["session_id"] == car.session_id) & (table["kappa"] == 0.2)
    with pytest.raises(ValueError, match="no row"):
        learn_eval.unit_scores(unit, table[~rows], [0.1, 0.2])
    learn_eval.unit_scores(unit, table[~rows], [0.1])  # another level is still complete


def test_finite_turns_nan_into_none():
    assert learn_eval.finite({"a": [1.0, math.nan], "b": {"c": math.nan}}) == {
        "a": [1.0, None],
        "b": {"c": None},
    }
