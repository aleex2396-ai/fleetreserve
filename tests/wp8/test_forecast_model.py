"""The forecast-table interface (`forecast_model`) and the empirical model (`models.empirical`).

1. A table written and read back gives the same floats, NaN included; `check_table` refuses other
   columns, a level outside plan section 2, a repeated car and level, an infinite number and two
   values of rho.
2. `TableForecaster` fed the empirical model's table equals `EmpiricalForecaster` car by car on a
   made-up unit: every energy mode, both departures, every kappa, both references, and the
   survival bound at every service hour (None where the empirical forecaster gives None).
3. NaN handling: no departure forecast gives None, but the oracle with known energy needs no row;
   NaN rate or energy only matters in the modes that read them; a NaN survival bound gives None
   and `gate_day.prepare_day` keeps the hour before; a car with no row, another rho or an hour
   outside 1 to 8 raises.
4. A discrete survival curve gives hand-computed quantiles, and a curve made from a sample gives
   the sample's `lower_quantile`, also given presence at each hour.
"""

import math

import numpy as np
import pandas as pd
import pytest
from learn_fake import odd_unit

from fleetreserve import forecast_model as fm
from fleetreserve import gate_day
from fleetreserve.forecast_empirical import (
    DEPARTURES,
    KAPPAS,
    MODES,
    EmpiricalForecaster,
    lower_quantile,
)
from fleetreserve.gate_config import Configuration
from fleetreserve.models.empirical import EmpiricalModel


def _table(unit, reference="other-days", rho=0.0):
    model = EmpiricalModel(rho=rho, reference=reference, min_cell=20)
    model.fit(pd.DataFrame())
    return fm.check_table(model.predict(unit.cars, KAPPAS))


@pytest.fixture(scope="module")
def unit(tmp_path_factory):
    return odd_unit(tmp_path_factory.mktemp("odd"))


@pytest.fixture(scope="module")
def table(unit):
    return _table(unit)


def test_a_written_table_reads_back_exactly(table, tmp_path):
    changed = table.copy()
    changed.loc[0, "d_hat"] = 0.1 + 0.2  # not a short decimal
    changed.loc[1, "survival_h3"] = math.nan
    path = tmp_path / "forecasts.csv"
    fm.write_table(changed, path)
    back = fm.read_table(path)
    pd.testing.assert_frame_equal(back, fm.check_table(changed))
    assert back.loc[0, "d_hat"] == 0.1 + 0.2 and math.isnan(back.loc[1, "survival_h3"])
    assert ",," in path.read_text()  # NaN is an empty field


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda t: t.drop(columns="d_median"), "columns"),
        (lambda t: t.assign(kappa=0.15), "kappa"),
        (lambda t: pd.concat([t, t.iloc[:1]]), "twice"),
        (lambda t: t.assign(e_cap=math.inf), "infinite"),
        (lambda t: t.assign(rho=np.where(t.index == 0, 0.1, 0.0)), "rho"),
    ],
)
def test_a_bad_table_is_refused(table, change, message):
    with pytest.raises(ValueError, match=message):
        fm.check_table(change(table))


@pytest.mark.parametrize("reference", ["other-days", "earlier-days"])
def test_the_table_forecaster_equals_the_empirical_forecaster(unit, reference, tmp_path):
    path = tmp_path / "forecasts.csv"
    fm.write_table(_table(unit, reference), path)
    from_table = fm.TableForecaster(fm.read_table(path))
    empirical = EmpiricalForecaster(unit, reference=reference, min_cell=20)
    seen_none = seen_fallback = seen_survival_none = 0
    for truth in unit.days:
        for i in range(len(truth.day.cars)):
            found = empirical.reference_for(truth, i)
            seen_fallback += found is not None and found.fallback
            for kappa in KAPPAS:
                for mode in MODES:
                    for departure in DEPARTURES:
                        expected = empirical.forecast_car(truth, i, kappa, mode, departure)
                        got = from_table.forecast_car(truth, i, kappa, mode, departure)
                        assert got == expected, (truth.record.date, i, kappa, mode, departure)
                        seen_none += expected is None
                for hour in range(1, truth.day.n_service + 1):
                    expected = empirical.survival_bound(truth, i, hour, kappa)
                    assert from_table.survival_bound(truth, i, hour, kappa) == expected
                    seen_survival_none += expected is None
    assert seen_none > 0 and seen_fallback > 0 and seen_survival_none > 0


def test_the_empirical_median_is_the_median_of_the_reference(unit, table):
    empirical = EmpiricalForecaster(unit, reference="other-days", min_cell=20)
    truth = unit.days[4]
    car = truth.fleet[0]
    found = empirical.reference_for(truth, 0)
    row = table[(table["session_id"] == car.session_id) & (table["date"] == truth.record.date)]
    assert set(row["d_median"]) == {lower_quantile(found.departures, 0.5)}


def _set(table, truth, i, **values):
    car = truth.fleet[i]
    out = table.copy()
    rows = (out["session_id"] == car.session_id) & (out["date"] == truth.record.date)
    for column, value in values.items():
        out.loc[rows, column] = value
    return out


def test_nan_means_no_forecast_only_where_the_column_is_read(unit, table):
    truth = unit.days[2]
    no_departure = fm.TableForecaster(_set(table, truth, 0, d_hat=math.nan))
    assert no_departure.forecast_car(truth, 0, 0.1, "known") is None
    assert no_departure.forecast_car(truth, 0, 0.1, "empirical") is None
    oracle = no_departure.forecast_car(truth, 0, 0.1, "empirical", "oracle")
    assert oracle is not None and oracle.departure == truth.day.cars[0].departure
    no_rate = fm.TableForecaster(_set(table, truth, 0, r_hat=math.nan))
    assert no_rate.forecast_car(truth, 0, 0.1, "known") is not None
    assert no_rate.forecast_car(truth, 0, 0.1, "user-rate") is not None
    assert no_rate.forecast_car(truth, 0, 0.1, "empirical") is None
    no_cap = fm.TableForecaster(_set(table, truth, 0, e_cap=math.nan))
    assert no_cap.forecast_car(truth, 0, 0.1, "user-rate") is None
    assert no_cap.forecast_car(truth, 0, 0.1, "known") is not None
    no_bound = fm.TableForecaster(_set(table, truth, 0, survival_h2=math.nan))
    assert no_bound.survival_bound(truth, 0, 2, 0.1) is None
    assert no_bound.survival_bound(truth, 0, 1, 0.1) is not None


def test_the_oracle_with_known_energy_reads_no_row(unit):
    truth = unit.days[2]
    empty = fm.TableForecaster(fm.empty_table())
    got = empty.forecast_car(truth, 1, 0.3, "known", "oracle")
    car = truth.day.cars[1]
    assert got == fm.ForecastCar(float(car.departure), car.rate, car.room, car.need(0.0))
    with pytest.raises(ValueError, match="no row"):
        empty.forecast_car(truth, 1, 0.3, "known", "empirical")
    with pytest.raises(ValueError, match="no row"):
        empty.forecast_car(truth, 1, 0.3, "empirical", "oracle")


def test_a_missing_row_another_rho_or_hour_raises(unit, table):
    truth = unit.days[2]
    car = truth.fleet[0]
    rows = (table["session_id"] == car.session_id) & (table["date"] == truth.record.date)
    with pytest.raises(ValueError, match="no row"):
        fm.TableForecaster(table[~rows]).forecast_car(truth, 0, 0.1, "known")
    with pytest.raises(ValueError, match="rho"):
        fm.TableForecaster(table.assign(rho=0.1)).forecast_car(truth, 0, 0.1, "known")
    forecaster = fm.TableForecaster(table)
    for hour in (0, 9, 1.0, True):
        with pytest.raises(ValueError):
            forecaster.survival_bound(truth, 0, hour, 0.1)


def test_prepare_day_leaves_a_car_without_forecast_unsteered_and_keeps_bounds(unit, table):
    truth = unit.days[6]
    configuration = Configuration(
        "c", "main", 0.3, "empirical", "known", "leave-unsteered", "any-optimal", "as-planned",
        "steered-only", "hour-start", "partner", None,
    )  # fmt: skip
    settings = gate_day.DaySettings(0.0, 5.0, 1e-6, 1e-6, 1e-6)
    changed = _set(table, truth, 0, d_hat=math.nan)
    changed = _set(changed, truth, 1, survival_h3=math.nan, survival_h4=math.nan)
    prepared = gate_day.prepare_day(truth, fm.TableForecaster(changed), configuration, settings)
    assert prepared.unsteered == ((0, gate_day.NO_REFERENCE),)
    assert prepared.abstain_reason is None
    bounds = prepared.departure_bounds[prepared.steered.index(1)]
    plain = gate_day.prepare_day(truth, fm.TableForecaster(table), configuration, settings)
    assert plain.unsteered == ()
    before = plain.departure_bounds[plain.steered.index(1)]
    assert before[3] != before[2] and before[4] != before[2]  # the bound does move there
    assert bounds[3] == bounds[4] == bounds[2] == before[2]
    assert bounds[5] == before[5]


def test_hand_computed_survival_quantiles():
    times, survival = [1.0, 2.0, 3.0], [0.8, 0.5, 0.0]
    assert fm.survival_quantile(times, survival, 0.1) == 1.0
    assert fm.survival_quantile(times, survival, 0.2) == 2.0  # F(1) = 0.2 is not above 0.2
    assert fm.survival_quantile(times, survival, 0.5) == 3.0
    assert fm.survival_quantile(times, survival, 0.3, after=1) == 2.0  # 0.3 / 0.8 > 0.3
    assert fm.survival_quantile(times, survival, 0.4, after=1) == 3.0  # 0.3 / 0.8 < 0.4
    assert fm.survival_quantile(times, survival, 0.1, after=0.5) == 1.0
    assert math.isnan(fm.survival_quantile(times, survival, 0.1, after=3))
    assert math.isnan(fm.survival_quantile([1.0, 2.0], [0.9, 0.8], 0.3))
    for bad in (([2.0, 1.0], [0.5, 0.2]), ([1.0, 2.0], [0.5, 0.6]), ([1.0], [1.2])):
        with pytest.raises(ValueError):
            fm.survival_quantile(*bad, 0.1)


def test_a_curve_from_a_sample_gives_the_sample_quantiles():
    rng = np.random.default_rng(5)
    for n in (1, 7, 20, 41, 200):
        x = np.round(rng.uniform(0.2, 9.0, n), 1)
        times = np.unique(x)
        survival = np.array([(x > t).sum() / n for t in times])
        columns = fm.departure_columns(times, survival, KAPPAS)
        for kappa in KAPPAS:
            assert columns[kappa]["d_hat"] == lower_quantile(x, kappa)
            assert columns[kappa]["d_median"] == lower_quantile(x, 0.5)
            for hour in range(1, 9):
                left = x[x > hour]
                got = columns[kappa][f"survival_h{hour}"]
                if left.size:
                    assert got == lower_quantile(left, kappa)
                else:
                    assert math.isnan(got)


def test_the_factory_is_picklable_and_serves_one_unit(table):
    import pickle

    factory = pickle.loads(pickle.dumps(fm.TableFactory(table)))
    other = table.assign(unit="other")
    both = fm.TableFactory(pd.concat([table, other], ignore_index=True))

    class Unit:
        unit = "odd"

    assert len(factory(Unit(), None)._rows) == len(table)
    assert len(both(Unit(), None)._rows) == len(table)
