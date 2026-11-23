"""The feature columns of the earlier analysis's per-car models (`learn_features.EARLIER_COLUMNS`),
by hand on a tiny made-up unit, and their fill where there is no history.

The columns are also covered by the two whole-table tests of `test_learn_features`: every column of
`feature_frame` is listed, and no feature changes when an outcome of its own day or a later one is
altered (which alters the plug-out, the energy and every other column not known at τ).
"""

import math

import pandas as pd
import pytest
from truth_tables_fake import FakeCar, date_of, tables

from fleetreserve import learn_features as lf
from fleetreserve.days_from_truth import apply_car_rules
from fleetreserve.truth import RATE_FALLBACK


def _cars(fakes, users) -> pd.DataFrame:
    """A cars table with the rules applied, the user of each row given by `users`."""
    car_table, _ = tables(fakes)
    by_session = dict(zip([f.session for f in fakes], users, strict=True))
    car_table["user_id"] = [by_session[s] for s in car_table["session_id"]]
    return apply_car_rules(car_table, 0.0, "cap")


def _tiny():
    fakes = [
        FakeCar("A", date_of(0), 8.0, 17.0, session="first", energy=10.0),
        FakeCar("A", date_of(0), 8.25, 16.0, session="other0", energy=5.0),
        FakeCar("A", date_of(0), 9.0, 24 + 11.0, session="overnight", energy=12.0),
        FakeCar("A", date_of(1), 7.5, 15.0, session="second", energy=6.0),
        FakeCar("A", date_of(1), 9.0 - 24, 11.0, session="overnight", energy=12.0),
        FakeCar("A", date_of(1), 8.0, 16.0, session="nobody", energy=5.0),
        FakeCar("A", date_of(2), 8.0 + 13 / 60 + 30 / 3600, 14.0, session="third", energy=7.0),
        FakeCar("A", date_of(2), 8.0, 16.0, session="other2", energy=5.0),
        FakeCar("A", date_of(2), 8.0, 16.0, session="fallback", energy=5.0),
    ]
    users = ["u", "v", "u", "u", "u", "", "u", "v", "w"]
    cars = _cars(fakes, users)
    cars.loc[cars["session_id"] == "fallback", "rate_rule"] = RATE_FALLBACK
    return lf.build_table(cars, rho=0.0).set_index(["date", "session_id"])


def test_the_columns_are_in_the_feature_frame_before_the_declared_departure():
    table = _tiny().reset_index()
    frame = list(lf.feature_frame(table).columns)
    assert lf.EARLIER_COLUMNS == (
        "plug_in_clock_h",
        "history_q10_stay_h",
        "history_q10_energy_kwh",
        "history_median_energy_kwh",
        "history_days_since_last_plug_in",
    )
    # after the old features, before the declared departure; the tail columns stay last
    start = frame.index(lf.FEATURE_COLUMNS[-1]) + 1
    end = start + len(lf.EARLIER_COLUMNS)
    assert frame[start:end] == list(lf.EARLIER_COLUMNS)
    assert frame[end:] == [*lf.DECLARED_COLUMNS, *lf.TAIL_HISTORY_COLUMNS]


def test_the_history_columns_by_hand():
    table = _tiny()
    # day 2, user u: earlier sessions "first" (stay 9 h, 10 kWh, plugged in on day 0 at 8:00),
    # "overnight" (26 h, 12 kWh, day 0 at 9:00) and "second" (7.5 h, 6 kWh, day 1 at 7:30)
    third = table.loc[(date_of(2), "third")]
    # stays sorted 7.5, 9, 26: the 10% point by linear interpolation is 7.5 + 0.2 * 1.5
    assert third["history_q10_stay_h"] == pytest.approx(7.8)
    # energies sorted 6, 10, 12: 10% point 6 + 0.2 * 4, median 10
    assert third["history_q10_energy_kwh"] == pytest.approx(6.8)
    assert third["history_median_energy_kwh"] == pytest.approx(10.0)
    # the latest earlier plug-in is "second", day 1 at 7:30; τ is day 2 at 10:00: 26.5 h
    assert third["history_days_since_last_plug_in"] == pytest.approx(26.5 / 24)
    # day 1, user u: "first" alone (the overnight session is still plugged in at day 1's τ)
    second = table.loc[(date_of(1), "second")]
    assert second["history_q10_stay_h"] == pytest.approx(9.0)
    assert second["history_q10_energy_kwh"] == pytest.approx(10.0)
    assert second["history_median_energy_kwh"] == pytest.approx(10.0)
    assert second["history_days_since_last_plug_in"] == pytest.approx(26.0 / 24)


def test_the_plug_in_clock_by_hand():
    table = _tiny()
    # hours after the local midnight of the row's date, to the second; negative for an earlier day
    assert table.loc[(date_of(2), "third")]["plug_in_clock_h"] == pytest.approx(
        8.0 + 13 / 60 + 30 / 3600, abs=1e-12
    )
    assert table.loc[(date_of(0), "other0")]["plug_in_clock_h"] == pytest.approx(8.25)
    assert table.loc[(date_of(1), "overnight")]["plug_in_clock_h"] == pytest.approx(9.0 - 24)
    assert table.loc[(date_of(0), "overnight")]["plug_in_clock_h"] == pytest.approx(9.0)


def test_the_plug_in_clock_reads_a_space_before_the_time_as_well():
    # the truth build writes "2020-01-01T08:00:00-06:00"; a table written by pandas itself writes
    # "2020-01-01 08:00:00-06:00"
    fakes = [
        FakeCar("A", date_of(0), 8.25, 16.0, session="a"),
        FakeCar("A", date_of(1), 9.5 - 24, 11.0, session="b"),
    ]
    cars = _cars(fakes, ["u", "v"])
    spaced = cars.assign(plug_in=cars["plug_in"].str.replace("T", " ", regex=False))
    got = lf.build_table(spaced, rho=0.0)["plug_in_clock_h"].tolist()
    assert got == pytest.approx([8.25, 9.5 - 24])


@pytest.mark.parametrize(
    "date, session",
    [(0, "first"), (0, "overnight"), (1, "nobody"), (2, "fallback")],
)
def test_no_history_is_filled_with_nan(date, session):
    row = _tiny().loc[(date_of(date), session)]
    for column in lf.EARLIER_COLUMNS[1:]:
        assert math.isnan(row[column])
    assert not math.isnan(row["plug_in_clock_h"])
