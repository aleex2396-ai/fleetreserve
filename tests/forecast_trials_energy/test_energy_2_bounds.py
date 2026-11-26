"""The two further energy values of the forecast trial `energy-2`
(`forecast_empirical.CAUTIOUS_NEED` and `CAUTIOUS_ALL`) and the four older values, on the
made-up unit of `learn_fake.odd_unit`. Written to be run through the test slot. Each test says
which edit turns it red.

1. The levels per column. At every level of the plan's grid `cautious-need` has the departure, the
   top charging speed and the storable energy of its `empirical` forecast at that level and the
   needed energy at level 0.05 (the 0.95 upper quantile); `cautious-all` has the departure at the
   level, and the three bounds at level 0.05. Checked from the reference sample
   (`EmpiricalForecaster`) and from a forecast table (`TableForecaster`) whose rows carry a
   different number in every column and level. Red if the need of `cautious-need` is read at the
   car's own level, if the rate or room of `cautious-all` is, if the need of `cautious-all` is, or
   if the departure is read at 0.05.
2. The two forecasters agree, bit for bit, for both new values; at level 0.05 both equal
   `empirical`; at the other levels the need is not below `empirical`'s and on this unit some is
   above, so the test is not a test of equal numbers; `cautious-all` is `cautious` for the rate and
   room and `cautious-need` for the need. Red if the quantile is taken on the wrong side.
3. THE OLD VALUES ARE UNCHANGED. `known`, `empirical`, `user-rate` and `cautious` give, for every
   car of the unit, every level of the grid and both departures, the same bits as the body of
   `EmpiricalForecaster.forecast_car` and of `TableForecaster.forecast_car` at the starting commit
   of this trial (the tip of `joint-36`, which already holds `cautious`), copied into this file
   (`start_empirical_car`, `start_table_car`). Red if the level of the rate, room or need of any old
   value follows the new levels. This is a new test; the first trial's own test of the same kind
   (`test_the_older_values_give_the_same_bits_...`) is left as it is.
4. The lists. The first trial's lists (`MODES`, `ENERGY_MODES`, `gate_config.ENERGY_BOUNDS`) are
   unchanged; the new values are in `TRIAL_ENERGY_MODES` and in `gate_config.MATRIX_ENERGY_BOUNDS`
   only; every matrix value is a mode a forecaster accepts. Red if a new value is added to a first
   list.
5. A table without the level-0.05 rows raises for each new value and not for `empirical`; a NaN in
   the level-0.05 need gives no forecast for the new values and not for `empirical` or `cautious`,
   and a NaN in the level-0.05 rate gives none for `cautious-all` and not for `cautious-need`;
   an unknown mode is refused. Red if the level-0.05 row is not read or a NaN is taken for a number.
6. The matrix reader takes both new values and refuses an unknown one; the oracle partners of a
   matrix with a new value are checked as for the older ones.
"""

import copy
import math
from dataclasses import replace

import pandas as pd
import pytest
from gate_fake import COMMITTED
from learn_fake import odd_unit

from fleetreserve import forecast_model as fm
from fleetreserve import gate_config
from fleetreserve.day import ForecastCar
from fleetreserve.forecast_empirical import (
    CAUTIOUS,
    CAUTIOUS_ALL,
    CAUTIOUS_LEVEL,
    CAUTIOUS_NEED,
    DEPARTURES,
    ENERGY_MODES,
    KAPPAS,
    MODES,
    TRIAL_ENERGY_MODES,
    EmpiricalForecaster,
    lower_quantile,
    upper_quantile,
)
from fleetreserve.models.empirical import EmpiricalModel

NEW = (CAUTIOUS_NEED, CAUTIOUS_ALL)
OLD = (*MODES, CAUTIOUS)


@pytest.fixture(scope="module")
def unit(tmp_path_factory):
    return odd_unit(tmp_path_factory.mktemp("odd"))


@pytest.fixture(scope="module")
def forecaster(unit):
    return EmpiricalForecaster(unit, reference="other-days", min_cell=20)


@pytest.fixture(scope="module")
def table(unit):
    model = EmpiricalModel(rho=0.0, reference="other-days", min_cell=20)
    model.fit(pd.DataFrame())
    return fm.check_table(model.predict(unit.cars, KAPPAS))


def bits(car: ForecastCar | None):
    """What 'byte-equal' means here: the hex form of each number (no tolerance), or None."""
    if car is None:
        return None
    return tuple(float(x).hex() for x in (car.departure, car.rate, car.cap, car.need))


def cars_of(unit):
    for truth in unit.days:
        for i in range(len(truth.day.cars)):
            yield truth, i


# --- the code of the starting commit, copied ------------------------------------------------------


def start_empirical_car(forecaster, truth, i, kappa, mode, departure):
    """`EmpiricalForecaster.forecast_car` as it was at the starting commit of this trial."""
    assert mode in ENERGY_MODES
    true = truth.day.cars[i]
    rho = truth.day.rho
    found = forecaster.reference_for(truth, i)
    if found is None and not (mode == "known" and departure == "oracle"):
        return None
    if departure == "oracle":
        d_hat = float(true.departure)
    else:
        d_hat = lower_quantile(found.departures, kappa)
    if mode == "known":
        return ForecastCar(d_hat, true.rate, true.room, true.need(rho))
    bound = CAUTIOUS_LEVEL if mode == CAUTIOUS else kappa  # the level of rate and cap
    rate = true.rate if mode == "user-rate" else lower_quantile(found.rates, bound)
    cap = lower_quantile(found.rooms, bound)
    need = upper_quantile((1 - rho) * found.rooms, kappa)
    return ForecastCar(d_hat, rate, cap, need)


def start_table_car(forecast_table, truth, i, kappa, mode, departure):
    """`TableForecaster.forecast_car` as it was at the starting commit of this trial."""
    assert mode in ENERGY_MODES
    true = truth.day.cars[i]
    rho = truth.day.rho
    if mode == "known" and departure == "oracle":
        return ForecastCar(float(true.departure), true.rate, true.room, true.need(rho))
    row = forecast_table._row(truth, i, kappa)
    d_hat = float(true.departure) if departure == "oracle" else forecast_table._value(row, "d_hat")
    if math.isnan(d_hat):
        return None
    if mode == "known":
        return ForecastCar(d_hat, true.rate, true.room, true.need(rho))
    bounds = forecast_table._row(truth, i, CAUTIOUS_LEVEL) if mode == CAUTIOUS else row
    rate = true.rate if mode == "user-rate" else forecast_table._value(bounds, "r_hat")
    cap = forecast_table._value(bounds, "e_cap")
    need = forecast_table._value(row, "e_need")
    if any(math.isnan(v) for v in (rate, cap, need)):
        return None
    return ForecastCar(d_hat, rate, cap, need)


# --- 3. the old values are unchanged --------------------------------------------------------------


def test_the_four_old_values_give_the_same_bits_as_the_code_at_the_starting_commit(
    unit, forecaster, table
):
    from_table = fm.TableForecaster(table)
    seen = {"none": 0, "forecast": 0}
    for truth, i in cars_of(unit):
        for kappa in KAPPAS:
            for mode in OLD:
                for departure in DEPARTURES:
                    key = (truth.record.date, i, kappa, mode, departure)
                    old = start_empirical_car(forecaster, truth, i, kappa, mode, departure)
                    assert bits(forecaster.forecast_car(truth, i, kappa, mode, departure)) == bits(
                        old
                    ), key
                    old_t = start_table_car(from_table, truth, i, kappa, mode, departure)
                    assert bits(from_table.forecast_car(truth, i, kappa, mode, departure)) == bits(
                        old_t
                    ), key
                    seen["none" if old is None else "forecast"] += 1
    assert seen["none"] > 0 and seen["forecast"] > 100  # cars with no sample and cars with one


def test_the_copy_of_the_starting_code_does_not_know_the_new_values(unit, forecaster, table):
    """The copies refuse the new values, so a test of them cannot pass for the new values by
    accident."""
    truth, i = next(cars_of(unit))
    for value in NEW:
        with pytest.raises(AssertionError):
            start_empirical_car(forecaster, truth, i, 0.2, value, "empirical")
        with pytest.raises(AssertionError):
            start_table_car(fm.TableForecaster(table), truth, i, 0.2, value, "empirical")


# --- 1. the levels per column ---------------------------------------------------------------------


def test_the_new_values_read_each_column_at_its_own_level_from_the_reference(unit, forecaster):
    checked = 0
    for truth, i in cars_of(unit):
        found = forecaster.reference_for(truth, i)
        if found is None:
            for value in NEW:
                assert forecaster.forecast_car(truth, i, 0.2, value) is None
            continue
        rho = truth.day.rho
        for kappa in KAPPAS:
            need_only = forecaster.forecast_car(truth, i, kappa, CAUTIOUS_NEED)
            everything = forecaster.forecast_car(truth, i, kappa, CAUTIOUS_ALL)
            empirical = forecaster.forecast_car(truth, i, kappa, "empirical")
            departure = lower_quantile(found.departures, kappa)
            cautious_need = upper_quantile((1 - rho) * found.rooms, CAUTIOUS_LEVEL)
            assert need_only.departure == everything.departure == departure
            assert need_only.rate == lower_quantile(found.rates, kappa) == empirical.rate
            assert need_only.cap == lower_quantile(found.rooms, kappa) == empirical.cap
            assert need_only.need == cautious_need
            assert everything.rate == lower_quantile(found.rates, CAUTIOUS_LEVEL)
            assert everything.cap == lower_quantile(found.rooms, CAUTIOUS_LEVEL)
            assert everything.need == cautious_need
            checked += 1
    assert checked > 100


def test_the_new_values_read_each_column_at_its_own_level_from_a_table(unit, table):
    """A table with a different number in every column and level: the level of each column shows."""
    marked = table.copy()
    for column, base in (
        ("d_hat", 1000.0),
        ("r_hat", 2000.0),
        ("e_cap", 3000.0),
        ("e_need", 4000.0),
    ):
        marked[column] = base + marked["kappa"]
    truth = unit.days[4]
    from_table = fm.TableForecaster(marked)
    low = CAUTIOUS_LEVEL
    for kappa in KAPPAS:
        assert from_table.forecast_car(truth, 0, kappa, CAUTIOUS_NEED) == ForecastCar(
            1000.0 + kappa, 2000.0 + kappa, 3000.0 + kappa, 4000.0 + low
        )
        assert from_table.forecast_car(truth, 0, kappa, CAUTIOUS_ALL) == ForecastCar(
            1000.0 + kappa, 2000.0 + low, 3000.0 + low, 4000.0 + low
        )
    for value, expected in (
        (CAUTIOUS_NEED, (2000.0 + 0.3, 3000.0 + 0.3, 4000.0 + low)),
        (CAUTIOUS_ALL, (2000.0 + low, 3000.0 + low, 4000.0 + low)),
    ):
        oracle = from_table.forecast_car(truth, 0, 0.3, value, "oracle")
        assert oracle.departure == float(truth.day.cars[0].departure)
        assert (oracle.rate, oracle.cap, oracle.need) == expected


# --- 2. the two forecasters agree, and the levels ------------------------------------------------


def test_the_table_forecaster_equals_the_reference_forecaster_for_the_new_values(
    unit, forecaster, table
):
    from_table = fm.TableForecaster(table)
    seen_none = seen_forecast = 0
    for truth, i in cars_of(unit):
        for kappa in KAPPAS:
            for departure in DEPARTURES:
                for value in NEW:
                    expected = forecaster.forecast_car(truth, i, kappa, value, departure)
                    got = from_table.forecast_car(truth, i, kappa, value, departure)
                    assert bits(got) == bits(expected), (truth.record.date, i, kappa, value)
                    seen_none += expected is None
                    seen_forecast += expected is not None
    assert seen_none > 0 and seen_forecast > 100


def test_at_level_0_05_both_new_values_equal_the_empirical_one(unit, forecaster):
    for truth, i in cars_of(unit):
        for value in NEW:
            assert bits(forecaster.forecast_car(truth, i, 0.05, value)) == bits(
                forecaster.forecast_car(truth, i, 0.05, "empirical")
            )


def test_the_new_values_are_the_first_trials_value_and_the_need_bound_put_together(
    unit, forecaster
):
    above = 0
    for truth, i in cars_of(unit):
        for kappa in KAPPAS:
            cautious = forecaster.forecast_car(truth, i, kappa, CAUTIOUS)
            need_only = forecaster.forecast_car(truth, i, kappa, CAUTIOUS_NEED)
            everything = forecaster.forecast_car(truth, i, kappa, CAUTIOUS_ALL)
            empirical = forecaster.forecast_car(truth, i, kappa, "empirical")
            if everything is None:
                assert None is cautious is need_only is empirical
                continue
            assert (everything.departure, everything.rate, everything.cap) == (
                cautious.departure,
                cautious.rate,
                cautious.cap,
            )
            assert everything.need == need_only.need
            assert need_only.need >= empirical.need and cautious.need == empirical.need
            above += need_only.need > empirical.need
    assert above > 0  # the upper quantile at 0.05 is above the one at 0.2 on this unit


# --- 4. the lists ---------------------------------------------------------------------------------


def test_the_first_trials_lists_are_unchanged_and_the_new_values_are_listed_apart():
    assert MODES == ("empirical", "known", "user-rate")
    assert (*MODES, CAUTIOUS) == ENERGY_MODES
    assert gate_config.ENERGY_BOUNDS == ("known", "empirical", CAUTIOUS)
    for value in NEW:
        assert value not in MODES and value not in ENERGY_MODES
        assert value not in gate_config.ENERGY_BOUNDS
    assert (*ENERGY_MODES, *NEW) == TRIAL_ENERGY_MODES
    assert (*gate_config.ENERGY_BOUNDS, *NEW) == gate_config.MATRIX_ENERGY_BOUNDS
    assert set(gate_config.MATRIX_ENERGY_BOUNDS) <= set(TRIAL_ENERGY_MODES)
    assert (CAUTIOUS_NEED, CAUTIOUS_ALL) == ("cautious-need", "cautious-all")


# --- 5. a missing row, a NaN, an unknown mode -----------------------------------------------------


def test_a_table_without_the_level_0_05_rows_raises_for_each_new_value_only(unit, table):
    truth = unit.days[4]
    only_02 = fm.TableForecaster(table[table["kappa"] == 0.2])
    assert only_02.forecast_car(truth, 0, 0.2, "empirical") is not None
    for value in NEW:
        with pytest.raises(ValueError, match="no row"):
            only_02.forecast_car(truth, 0, 0.2, value)


def _with_nan_at_level_0_05(unit, table, column):
    truth = unit.days[4]
    car = truth.fleet[0]
    changed = table.copy()
    rows = (
        (changed["session_id"] == car.session_id)
        & (changed["date"] == truth.record.date)
        & (changed["kappa"] == 0.05)
    )
    assert rows.any()
    changed.loc[rows, column] = math.nan
    return truth, fm.TableForecaster(changed)


def test_a_nan_in_the_level_0_05_need_gives_no_forecast_for_the_new_values_only(unit, table):
    truth, from_table = _with_nan_at_level_0_05(unit, table, "e_need")
    for value in NEW:
        assert from_table.forecast_car(truth, 0, 0.2, value) is None
    for value in ("empirical", "user-rate", CAUTIOUS):
        assert from_table.forecast_car(truth, 0, 0.2, value) is not None


def test_a_nan_in_the_level_0_05_rate_gives_no_forecast_for_cautious_all_only(unit, table):
    truth, from_table = _with_nan_at_level_0_05(unit, table, "r_hat")
    assert from_table.forecast_car(truth, 0, 0.2, CAUTIOUS_ALL) is None
    assert from_table.forecast_car(truth, 0, 0.2, CAUTIOUS) is None
    assert from_table.forecast_car(truth, 0, 0.2, CAUTIOUS_NEED) is not None
    assert from_table.forecast_car(truth, 0, 0.2, "empirical") is not None


def test_an_unknown_mode_is_refused_by_both_forecasters(unit, forecaster, table):
    truth = unit.days[4]
    for mode in ("cautious-ish", "cautious-need-all"):
        with pytest.raises(ValueError, match="mode must be one of"):
            forecaster.forecast_car(truth, 0, 0.2, mode)
        with pytest.raises(ValueError, match="mode must be one of"):
            fm.TableForecaster(table).forecast_car(truth, 0, 0.2, mode)


# --- 6. the matrix reader -------------------------------------------------------------------------


def _document(value):
    document = copy.deepcopy(COMMITTED)
    for entry in document["configurations"]:
        entry["energy_bounds"] = value
    return document


def test_the_matrix_reader_takes_both_new_values_and_refuses_an_unknown_one():
    for value in gate_config.MATRIX_ENERGY_BOUNDS:
        matrix = gate_config.parse(_document(value))
        assert {c.energy_bounds for c in matrix.configurations} == {value}
    with pytest.raises(gate_config.MatrixError, match="energy_bounds"):
        gate_config.parse(_document("cautious-ish"))


@pytest.mark.parametrize("value", NEW)
def test_a_partner_that_differs_in_a_new_value_is_refused(value):
    document = _document(value)
    first = next(e for e in document["configurations"] if e["oracle_partner"] is not None)
    partner = next(e for e in document["configurations"] if e["name"] == first["oracle_partner"])
    partner["energy_bounds"] = "known"
    with pytest.raises(gate_config.MatrixError, match="oracle partner differs"):
        gate_config.parse(document)


@pytest.mark.parametrize("value", NEW)
def test_the_new_values_keep_the_level_in_the_work_key_of_an_oracle_configuration(value):
    matrix = gate_config.parse(_document("known"))
    oracle = next(c for c in matrix.configurations if c.departure == "oracle")
    assert oracle.work_key()[0] is None  # known energy: the level changes nothing for an oracle
    other = replace(oracle, energy_bounds=value)
    assert other.work_key()[0] == oracle.kappa
