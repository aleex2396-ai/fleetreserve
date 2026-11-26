"""The `cautious` energy bounds (`forecast_empirical.CAUTIOUS`) and the two older values, on the
made-up unit of `learn_fake.odd_unit`. Written to be run through the test slot. Each test says which
edit turns it red.

1. The levels per column. At every level of the plan's grid a car's `cautious` forecast has the
   departure and the needed energy of its `empirical` forecast at that level, and the top charging
   speed and storable energy at level 0.05: from the reference sample (`EmpiricalForecaster`) and
   from a forecast table (`TableForecaster`) whose rows carry a different number in every column
   and level. Red if the rate or the cap is read at the car's own level, if the need is read at
   0.05, or if the departure is read at 0.05.
2. At level 0.05 `cautious` equals `empirical`; at every other level the rate and the cap are not
   above `empirical`'s, and on this unit some are below, so the test is not a test of equal
   numbers. Red if `CAUTIOUS_LEVEL` is changed to a level other than the lowest, or if a quantile of
   the wrong side is taken.
3. The older values are unchanged. `known`, `empirical` and `user-rate` give, for every car of the
   unit, every level of the grid and both departures, the same bits as the body of
   `EmpiricalForecaster.forecast_car` and of `TableForecaster.forecast_car` at aa4290a (the commit
   before the new value), copied into this file (`old_empirical_car`, `old_table_car`). Red if the
   level of the rate or the cap of `empirical` or `user-rate` follows `CAUTIOUS_LEVEL`, or if
   `known` reads a row it did not read.
4. `MODES` (which the forecast report and the table tests loop over) is unchanged and does not hold
   the new value; the new value is in `ENERGY_MODES` and in `gate_config.ENERGY_BOUNDS`; every
   `ENERGY_BOUNDS` value is a mode a forecaster accepts. Red if the new value is added to `MODES`.
5. A table with no row at level 0.05 raises for `cautious` and not for `empirical`; a NaN in the
   level-0.05 rate gives no forecast for `cautious` and not for `empirical`; an unknown mode is
   refused. Red if the level-0.05 row is not read, or if NaN is taken for a number.
6. The matrix reader takes the new value and refuses an unknown one; a matrix whose configurations
   have the new value has the oracle partners it needs (`gate_config._check_partners`).
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
    CAUTIOUS_LEVEL,
    DEPARTURES,
    ENERGY_MODES,
    KAPPAS,
    MODES,
    EmpiricalForecaster,
    lower_quantile,
    upper_quantile,
)
from fleetreserve.models.empirical import EmpiricalModel


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


# --- the code of aa4290a, copied ------------------------------------------------------------------


def old_empirical_car(forecaster, truth, i, kappa, mode, departure):
    """`EmpiricalForecaster.forecast_car` as it was at aa4290a (without the new value)."""
    assert mode in ("empirical", "known", "user-rate")
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
    rate = true.rate if mode == "user-rate" else lower_quantile(found.rates, kappa)
    cap = lower_quantile(found.rooms, kappa)
    need = upper_quantile((1 - rho) * found.rooms, kappa)
    return ForecastCar(d_hat, rate, cap, need)


def old_table_car(forecast_table, truth, i, kappa, mode, departure):
    """`TableForecaster.forecast_car` as it was at aa4290a (without the new value)."""
    assert mode in ("empirical", "known", "user-rate")
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
    rate = true.rate if mode == "user-rate" else forecast_table._value(row, "r_hat")
    cap = forecast_table._value(row, "e_cap")
    need = forecast_table._value(row, "e_need")
    if any(math.isnan(v) for v in (rate, cap, need)):
        return None
    return ForecastCar(d_hat, rate, cap, need)


# --- 3. the older values are unchanged ------------------------------------------------------------


def test_the_older_values_give_the_same_bits_as_the_code_before_the_new_value(
    unit, forecaster, table
):
    from_table = fm.TableForecaster(table)
    seen = {"none": 0, "forecast": 0}
    for truth, i in cars_of(unit):
        for kappa in KAPPAS:
            for mode in MODES:
                for departure in DEPARTURES:
                    key = (truth.record.date, i, kappa, mode, departure)
                    old = old_empirical_car(forecaster, truth, i, kappa, mode, departure)
                    assert bits(forecaster.forecast_car(truth, i, kappa, mode, departure)) == bits(
                        old
                    ), key
                    old_t = old_table_car(from_table, truth, i, kappa, mode, departure)
                    assert bits(from_table.forecast_car(truth, i, kappa, mode, departure)) == bits(
                        old_t
                    ), key
                    seen["none" if old is None else "forecast"] += 1
    assert seen["none"] > 0 and seen["forecast"] > 100  # cars with no sample and cars with one


def test_the_old_code_copy_is_not_a_copy_of_the_new_value(unit, forecaster):
    """The reference copy has no `cautious` mode: it refuses it, so a test of it could not be
    passing for the new value by accident."""
    truth, i = next(cars_of(unit))
    with pytest.raises(AssertionError):
        old_empirical_car(forecaster, truth, i, 0.2, CAUTIOUS, "empirical")


# --- 1. the levels per column ---------------------------------------------------------------------


def test_the_cautious_value_reads_each_column_at_its_own_level_from_the_reference(unit, forecaster):
    checked = 0
    for truth, i in cars_of(unit):
        found = forecaster.reference_for(truth, i)
        if found is None:
            assert forecaster.forecast_car(truth, i, 0.2, CAUTIOUS) is None
            continue
        rho = truth.day.rho
        for kappa in KAPPAS:
            car = forecaster.forecast_car(truth, i, kappa, CAUTIOUS)
            empirical = forecaster.forecast_car(truth, i, kappa, "empirical")
            assert car.departure == lower_quantile(found.departures, kappa)
            assert car.departure == empirical.departure
            assert car.rate == lower_quantile(found.rates, CAUTIOUS_LEVEL)
            assert car.cap == lower_quantile(found.rooms, CAUTIOUS_LEVEL)
            assert car.need == upper_quantile((1 - rho) * found.rooms, kappa) == empirical.need
            checked += 1
    assert checked > 100


def test_the_cautious_value_reads_each_column_at_its_own_level_from_a_table(unit, table):
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
    for kappa in KAPPAS:
        car = from_table.forecast_car(truth, 0, kappa, CAUTIOUS)
        assert car == ForecastCar(
            1000.0 + kappa, 2000.0 + CAUTIOUS_LEVEL, 3000.0 + CAUTIOUS_LEVEL, 4000.0 + kappa
        )
        empirical = from_table.forecast_car(truth, 0, kappa, "empirical")
        assert empirical == ForecastCar(
            1000.0 + kappa, 2000.0 + kappa, 3000.0 + kappa, 4000.0 + kappa
        )
    oracle = from_table.forecast_car(truth, 0, 0.3, CAUTIOUS, "oracle")
    assert oracle.departure == float(truth.day.cars[0].departure)
    assert (oracle.rate, oracle.cap, oracle.need) == (
        2000.0 + CAUTIOUS_LEVEL,
        3000.0 + CAUTIOUS_LEVEL,
        4000.0 + 0.3,
    )


def test_the_table_forecaster_equals_the_reference_forecaster_for_the_cautious_value(
    unit, forecaster, table
):
    from_table = fm.TableForecaster(table)
    seen_none = seen_forecast = 0
    for truth, i in cars_of(unit):
        for kappa in KAPPAS:
            for departure in DEPARTURES:
                expected = forecaster.forecast_car(truth, i, kappa, CAUTIOUS, departure)
                got = from_table.forecast_car(truth, i, kappa, CAUTIOUS, departure)
                assert bits(got) == bits(expected), (truth.record.date, i, kappa, departure)
                seen_none += expected is None
                seen_forecast += expected is not None
    assert seen_none > 0 and seen_forecast > 100


# --- 2. level 0.05 and the other levels -----------------------------------------------------------


def test_at_level_0_05_the_cautious_value_equals_the_empirical_one(unit, forecaster):
    assert CAUTIOUS_LEVEL == min(KAPPAS) == 0.05
    for truth, i in cars_of(unit):
        assert bits(forecaster.forecast_car(truth, i, 0.05, CAUTIOUS)) == bits(
            forecaster.forecast_car(truth, i, 0.05, "empirical")
        )


def test_at_every_other_level_the_cautious_rate_and_cap_are_not_above_the_empirical_ones(
    unit, forecaster
):
    strictly_below = 0
    for truth, i in cars_of(unit):
        for kappa in KAPPAS:
            cautious = forecaster.forecast_car(truth, i, kappa, CAUTIOUS)
            empirical = forecaster.forecast_car(truth, i, kappa, "empirical")
            if cautious is None:
                assert empirical is None
                continue
            assert cautious.rate <= empirical.rate and cautious.cap <= empirical.cap
            assert (cautious.departure, cautious.need) == (empirical.departure, empirical.need)
            strictly_below += cautious.rate < empirical.rate or cautious.cap < empirical.cap
    assert strictly_below > 0


# --- 4. the lists ---------------------------------------------------------------------------------


def test_the_modes_the_reports_loop_over_are_unchanged_and_the_new_value_is_listed_apart():
    assert MODES == ("empirical", "known", "user-rate")
    assert CAUTIOUS not in MODES
    assert (*MODES, CAUTIOUS) == ENERGY_MODES
    assert gate_config.ENERGY_BOUNDS == ("known", "empirical", CAUTIOUS)
    assert set(gate_config.ENERGY_BOUNDS) <= set(ENERGY_MODES)


# --- 5. a missing row, a NaN, an unknown mode -----------------------------------------------------


def test_a_table_without_the_level_0_05_rows_raises_for_the_cautious_value_only(unit, table):
    truth = unit.days[4]
    only_02 = fm.TableForecaster(table[table["kappa"] == 0.2])
    assert only_02.forecast_car(truth, 0, 0.2, "empirical") is not None
    assert only_02.forecast_car(truth, 0, 0.2, "known") is not None
    with pytest.raises(ValueError, match="no row"):
        only_02.forecast_car(truth, 0, 0.2, CAUTIOUS)


def test_a_nan_in_the_level_0_05_rate_gives_no_cautious_forecast_and_leaves_empirical(unit, table):
    truth = unit.days[4]
    car = truth.fleet[0]
    changed = table.copy()
    rows = (
        (changed["session_id"] == car.session_id)
        & (changed["date"] == truth.record.date)
        & (changed["kappa"] == 0.05)
    )
    assert rows.any()
    changed.loc[rows, "r_hat"] = math.nan
    from_table = fm.TableForecaster(changed)
    assert from_table.forecast_car(truth, 0, 0.2, CAUTIOUS) is None
    assert from_table.forecast_car(truth, 0, 0.2, "empirical") is not None
    assert from_table.forecast_car(truth, 0, 0.2, "user-rate") is not None


def test_an_unknown_mode_is_refused_by_both_forecasters(unit, forecaster, table):
    truth = unit.days[4]
    with pytest.raises(ValueError, match="mode must be one of"):
        forecaster.forecast_car(truth, 0, 0.2, "cautious-ish")
    with pytest.raises(ValueError, match="mode must be one of"):
        fm.TableForecaster(table).forecast_car(truth, 0, 0.2, "cautious-ish")


# --- 6. the matrix reader -------------------------------------------------------------------------


def _document(value):
    document = copy.deepcopy(COMMITTED)
    for entry in document["configurations"]:
        entry["energy_bounds"] = value
    return document


def test_the_matrix_reader_takes_the_new_value_and_refuses_an_unknown_one():
    for value in gate_config.ENERGY_BOUNDS:
        matrix = gate_config.parse(_document(value))
        assert {c.energy_bounds for c in matrix.configurations} == {value}
    with pytest.raises(gate_config.MatrixError, match="energy_bounds"):
        gate_config.parse(_document("cautious-ish"))


def test_a_partner_that_differs_in_the_new_value_is_refused():
    document = _document(CAUTIOUS)
    first = next(e for e in document["configurations"] if e["oracle_partner"] is not None)
    partner = next(e for e in document["configurations"] if e["name"] == first["oracle_partner"])
    partner["energy_bounds"] = "known"
    with pytest.raises(gate_config.MatrixError, match="oracle partner differs"):
        gate_config.parse(document)


def test_the_new_value_keeps_the_level_in_the_work_key_of_an_oracle_configuration():
    matrix = gate_config.parse(_document("known"))
    oracle = next(c for c in matrix.configurations if c.departure == "oracle")
    assert oracle.work_key()[0] is None  # known energy: the level changes nothing for an oracle
    other = replace(oracle, energy_bounds=CAUTIOUS)
    assert other.work_key()[0] == oracle.kappa  # the cautious bounds read the level for the need
