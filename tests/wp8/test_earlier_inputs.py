"""The inputs of the earlier analysis's per-car models (`earlier_inputs`), its academic calendar
(`earlier_calendar`) and its level grid (`earlier_grid`), by hand.

1. The calendar's flags on dates worked out from its tables: a term day, a summer day, a recess
   day, an Institute holiday inside a term, a year-end closure day; NaN outside its span.
2. Each input comes from the feature table's column named in `earlier_inputs.SOURCES`; the paid
   regime starts on 1 November 2018; a car with no user id has no past-session count; the typed
   departure is the declared departure stated by τ.
3. The grid: the stay target is capped at 10 hours and rounded down to the quarter hour; rows are
   sorted across levels and clipped; a level on the grid is read as its column, one between two
   levels by linear interpolation, one below the grid as 0. The forecast table: departure, rate
   and room at kappa, the median departure, protected energy (1 - rho) x room at 1 - kappa, the
   survival columns NaN; a car with no stay level identified has no departure forecast.
"""

import math

import numpy as np
import pandas as pd
import pytest
from truth_tables_fake import FakeCar, tables

from fleetreserve import earlier_calendar, earlier_grid, earlier_inputs
from fleetreserve import learn_features as lf
from fleetreserve.days_from_truth import apply_car_rules
from fleetreserve.forecast_model import SURVIVAL_COLUMNS, check_table


@pytest.mark.parametrize(
    "day, flags",
    [
        ("2019-05-15", (1.0, 0.0, 0.0)),  # third term of 2018-19
        ("2019-07-15", (0.0, 1.0, 0.0)),  # the summer of 2019
        ("2019-03-25", (0.0, 0.0, 0.0)),  # spring recess: neither term nor summer
        ("2019-11-28", (1.0, 0.0, 1.0)),  # Thanksgiving inside the first term of 2019-20
        ("2019-12-30", (0.0, 0.0, 1.0)),  # year-end closure, winter recess
        ("2020-01-01", (0.0, 0.0, 1.0)),  # the closure's last day
        ("2020-01-02", (0.0, 0.0, 0.0)),
        ("2018-06-15", (1.0, 0.0, 0.0)),  # a term's last day is in the term, not the summer
        ("2018-10-01", (1.0, 0.0, 0.0)),  # the fall term's first day ends the summer
    ],
)
def test_the_calendar_flags_by_hand(day, flags):
    got = earlier_calendar.day_flags(pd.Series([day]))
    assert tuple(got.loc[0, ["in_term", "summer", "holiday"]]) == flags


@pytest.mark.parametrize("day", ["2017-12-31", "2022-07-01", "2024-03-01"])
def test_a_date_outside_the_calendar_has_no_flags(day):
    got = earlier_calendar.day_flags(pd.Series(["2019-05-15", day]))
    assert got.loc[0, "in_term"] == 1.0
    assert got.loc[1].isna().all()


def _table():
    fakes = [
        FakeCar("A", "2018-10-31", 8.0, 17.0, session="a1", energy=10.0),
        FakeCar("A", "2018-10-31", 7.0, 16.0, session="a2", energy=5.0, declared=15.5),
        FakeCar("A", "2018-11-01", 7.5, 15.0, session="b1", energy=6.0),
        FakeCar("A", "2018-11-01", 8.0, 16.0, session="b2", energy=5.0, declared=12.0),
        FakeCar("A", "2018-11-01", 8.0, 16.0, session="b3", energy=5.0),
    ]
    fakes[1].declared_at = 7.5  # stated before τ (10:00)
    fakes[3].declared_at = 11.0  # stated after τ: not known at τ
    users = {"a1": "u", "a2": "v", "b1": "u", "b2": "", "b3": "w"}
    car_table, _ = tables(fakes)
    car_table["user_id"] = [users[s] for s in car_table["session_id"]]
    ruled = apply_car_rules(car_table, 0.0, "cap")
    return lf.feature_frame(lf.build_table(ruled, rho=0.0))


def test_each_input_comes_from_its_source_column():
    frame = _table()
    got = earlier_inputs.numeric(frame)
    assert list(got.columns) == list(earlier_inputs.INPUTS)
    direct, derived = set(earlier_inputs.SOURCES), set(earlier_inputs.DERIVED)
    assert not direct & derived and direct | derived == set(earlier_inputs.INPUTS)
    for name, source in earlier_inputs.SOURCES.items():
        np.testing.assert_array_equal(got[name], frame[source].astype(float), err_msg=name)


def test_the_inputs_by_hand():
    frame = _table().set_index("session_id")
    got = earlier_inputs.numeric(frame.reset_index()).set_index(frame.index)
    # the paid regime starts on 1 November 2018; paid needs a user id as well
    assert got.loc["a1", "paid_regime"] == 0.0 and got.loc["b1", "paid_regime"] == 1.0
    assert got.loc["b1", "paid"] == 1.0 and got.loc["b2", "paid"] == 0.0
    assert got.loc["a1", "paid"] == 0.0
    # user u has one earlier session ("a1") on day 1; w has none; b2 has no user id
    assert got.loc["b1", "n_past"] == 1.0 and got.loc["b3", "n_past"] == 0.0
    assert math.isnan(got.loc["b2", "n_past"])
    assert got.loc["b2", "linked"] == 0.0 and got.loc["b1", "linked"] == 1.0
    assert got.loc["b1", "stay_q10"] == pytest.approx(9.0)
    assert got.loc["b1", "kwh_q50"] == pytest.approx(10.0)
    # declared at 7:30 for 15:30: 5.5 hours after τ; b2's statement came after τ
    assert got.loc["a2", "typed"] == 1.0 and got.loc["a2", "typed_remaining_h"] == 5.5
    assert got.loc["b2", "typed"] == 0.0 and math.isnan(got.loc["b2", "typed_remaining_h"])
    # 31 October 2018: Wednesday, first term of 2018-19, no holiday
    assert got.loc["a1", "weekday"] == 2.0 and got.loc["a1", "month"] == 10.0
    assert tuple(got.loc["a1", ["in_term", "summer", "holiday"]]) == (1.0, 0.0, 0.0)
    assert got.loc["a2", "plugin_hour"] == 7.0 and got.loc["a2", "elapsed_h"] == 3.0
    assert got.loc["a2", "cars_present"] == 2.0


def test_the_inputs_not_built_are_the_typed_plan_fields_the_tables_do_not_hold():
    assert earlier_inputs.LEFT_OUT == ("typed_minutes", "typed_kwh", "typed_60", "typed_480")
    assert not set(earlier_inputs.LEFT_OUT) & set(earlier_inputs.INPUTS)
    assert len(earlier_inputs.INPUTS) + len(earlier_inputs.LEFT_OUT) == 23


def test_the_stay_target_is_capped_and_rounded_down():
    departures = np.array([0.1, 1.1, 2.6, 3.75, 3.9, 9.99, 10.0, 11.0, 12.5])
    expected = [0.0, 1.0, 2.5, 3.75, 3.75, 9.75, 10.0, 10.0, 10.0]
    assert earlier_grid.stay_target(departures).tolist() == expected


def test_rows_are_sorted_across_levels_and_clipped():
    stay = np.array([[3.0, 2.0, 12.0], [-1.0, 1.0, np.nan]])
    energy = np.array([[5.0, -2.0, 1.0], [1.0, 2.0, 3.0]])
    sorted_stay, sorted_energy = earlier_grid.tidy(stay, energy)
    assert sorted_stay[0].tolist() == [2.0, 3.0, 10.0]
    assert sorted_stay[1, :2].tolist() == [0.0, 1.0] and math.isnan(sorted_stay[1, 2])
    assert sorted_energy.tolist() == [[0.0, 1.0, 5.0], [1.0, 2.0, 3.0]]


def test_a_level_is_read_off_the_grid():
    levels = earlier_grid.LEVELS
    assert len(levels) == 16 and levels[0] == 0.002 and levels[-1] == 0.95
    for level in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        assert level in levels
    grid = np.tile(np.arange(16, dtype=float), (2, 1))
    k = int(np.flatnonzero(levels == 0.2)[0])
    assert earlier_grid.at_level(grid, 0.2).tolist() == [float(k)] * 2
    # 0.25 lies halfway between 0.2 and 0.3
    assert earlier_grid.at_level(grid, 0.25).tolist() == pytest.approx([k + 0.5] * 2)
    assert earlier_grid.at_level(grid, 0.001).tolist() == [0.0, 0.0]


def test_the_forecast_table_is_read_off_the_grids():
    keys = {"unit": ["u", "u"], "site": ["A", "A"], "date": ["2019-01-07"] * 2}
    frame = pd.DataFrame({**keys, "session_id": ["x", "y"]})
    levels = earlier_grid.LEVELS
    stay = np.tile(10 * levels, (2, 1))
    stay[1] = np.nan  # car y: no level identified, so no departure forecast
    rate = np.tile(1 + levels, (2, 1))
    room = np.vstack([20 * levels, (20 * levels)[::-1]])  # car y's row is sorted first
    table = check_table(earlier_grid.table(frame, (0.1, 0.3), 0.25, stay, rate, room))
    assert len(table) == 4 and (table["rho"] == 0.25).all()
    assert table[list(SURVIVAL_COLUMNS)].isna().all().all()
    rows = table.set_index(["session_id", "kappa"])
    # kappa 0.1: stay 1, median 5, rate 1.1, room 2, need 0.75 x room at 0.9 = 0.75 x 18
    expected = {0.1: (1.0, 5.0, 1.1, 2.0, 13.5), 0.3: (3.0, 5.0, 1.3, 6.0, 0.75 * 14.0)}
    columns = ["d_hat", "d_median", "r_hat", "e_cap", "e_need"]
    for kappa, numbers in expected.items():
        assert tuple(rows.loc[("x", kappa), columns]) == pytest.approx(numbers)
        assert tuple(rows.loc[("y", kappa), columns[2:]]) == pytest.approx(numbers[2:])
        assert rows.loc[("y", kappa), ["d_hat", "d_median"]].isna().all()
    assert len(earlier_grid.table(frame.iloc[:0], (0.1,), 0.25, stay[:0], rate[:0], room[:0])) == 0
