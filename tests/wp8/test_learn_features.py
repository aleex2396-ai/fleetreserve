"""The feature and label table of the learned models (`learn_features`), on made-up cars tables.

1. Every column of a cars table (`days_from_truth.UnitDays.cars`) is listed exactly once, as known
   or not known at τ, and no feature or key column is one not known at τ.
2. No feature of a car changes when any outcome of its own day or a later day is altered: for each
   cut-off date, every column not known at τ of the rows of that date and later (and of every row
   of a session listed on them) is changed, and the features of the rows up to the cut-off date
   stay identical. The made-up unit has users with many sessions and sessions plugged in over two
   days, so the user history is exercised.
3. The user history computed by hand on a tiny case, with a session over two days left out of its
   own history; the fill for no history (count 0, NaN) for a first session, a car without a user
   id and the fallback rate rule.
4. The labels are the departure, rate, room and protected energy, and `feature_frame` holds none.
5. Event columns join on site and date, are absent by default, and a badly named column is refused.
"""

import datetime
import math

import numpy as np
import pandas as pd
import pytest
from truth_tables_fake import FakeCar, date_of, tables

from fleetreserve import learn_features as lf
from fleetreserve.days_from_truth import apply_car_rules
from fleetreserve.forecast_model import KEY_COLUMNS
from fleetreserve.truth import RATE_FALLBACK

FEATURES = [
    *KEY_COLUMNS,
    "counting",
    *lf.FEATURE_COLUMNS,
    *lf.EARLIER_COLUMNS,
    *lf.DECLARED_COLUMNS,
    *lf.TAIL_HISTORY_COLUMNS,
]


def _cars(fakes, users, rho=0.0) -> pd.DataFrame:
    """A cars table with the rules applied, the user of each row given by `users`."""
    car_table, _ = tables(fakes)
    by_session = dict(zip([f.session for f in fakes], users, strict=True))
    car_table["user_id"] = [by_session[s] for s in car_table["session_id"]]
    return apply_car_rules(car_table, rho, "cap")


def _history_unit():
    """24 days at site A: four cars a day from six users; every fourth day one session runs
    overnight into the next day (listed on both days); some cars have no user id."""
    fakes, users = [], []
    for k in range(24):
        for j in range(4):
            fakes.append(
                FakeCar(
                    "A",
                    date_of(k),
                    7.0 + 0.5 * j,
                    12.25 + 1.3 * j + 0.37 * (k % 5),
                    rate=2.0 + 0.5 * j,
                    room=1.0 + 0.3 * ((k + j) % 4),
                    session=f"A-{k}-{j}",
                    energy=4.0 + j + 0.1 * k,
                )
            )
            users.append("" if (k + j) % 7 == 0 else f"u{(k + 2 * j) % 6}")
        if k % 4 == 1 and k < 23:
            for day, start, end in ((k, 9.0, 24 + 11.5), (k + 1, 9.0 - 24, 11.5)):
                fakes.append(
                    FakeCar("A", date_of(day), start, end, room=2.0, session=f"night-{k}", energy=9)
                )
                users.append(f"u{k % 6}")
    return _cars(fakes, users)


def test_every_column_is_listed_once():
    cars = _history_unit()
    known, not_known = set(lf.KNOWN_AT_TAU), set(lf.NOT_KNOWN_AT_TAU)
    assert not known & not_known
    assert known | not_known == set(cars.columns)
    assert len(known) == len(lf.KNOWN_AT_TAU) and len(not_known) == len(lf.NOT_KNOWN_AT_TAU)
    frame = lf.feature_frame(lf.build_table(cars, rho=0.0))
    assert list(frame.columns) == FEATURES
    assert not set(frame.columns) & (not_known | set(lf.LABEL_COLUMNS))


def _later_time(text: str, hours: float) -> str:
    moment = datetime.datetime.fromisoformat(text) + datetime.timedelta(hours=hours)
    return moment.isoformat()


def _altered(cars: pd.DataFrame, cut: str, rng) -> pd.DataFrame:
    """`cars` with every column not known at τ changed on the rows of `cut` and later dates and on
    every row of a session listed on them."""
    out = cars.copy()
    sessions = set(cars.loc[cars["date"] >= cut, "session_id"])
    rows = cars["session_id"].isin(sessions).to_numpy()
    n = int(rows.sum())
    for column in lf.NOT_KNOWN_AT_TAU:
        values = out[column]
        if column == "plug_out":
            shift = rng.uniform(0.5, 30.0, n)
            changed = [_later_time(t, h) for t, h in zip(values[rows], shift, strict=True)]
        elif values.dtype == bool:
            changed = ~values[rows].to_numpy()
        elif pd.api.types.is_numeric_dtype(values):
            changed = values[rows].to_numpy() + rng.uniform(0.5, 9.0, n).astype(values.dtype)
        else:  # text: numbers written as text get a new number, other text new text
            try:
                changed = [f"{float(v) + rng.uniform(0.5, 9.0):.6f}" for v in values[rows]]
            except ValueError:
                changed = [f"{v} changed" for v in values[rows]]
        out[column] = out[column].astype(object)
        out.loc[rows, column] = changed
        out[column] = out[column].astype(values.dtype)
    return out


def test_no_feature_reads_its_own_day_or_a_later_one():
    cars = _history_unit()
    before = lf.feature_frame(lf.build_table(cars, rho=0.0))
    assert before["history_sessions"].max() >= 5  # the history is exercised
    rng = np.random.default_rng(11)
    dates = sorted(set(cars["date"]))
    for cut in dates[::3]:
        altered = _altered(cars, cut, rng)
        assert (altered["plug_out"] != cars["plug_out"]).sum() > 0
        after = lf.feature_frame(lf.build_table(altered, rho=0.0))
        kept = (cars["date"] <= cut).to_numpy()
        pd.testing.assert_frame_equal(before[kept], after[kept])
        # the check can see a change: a later day's history reads the altered outcomes
        later = (cars["date"] > cut).to_numpy() & (before["history_sessions"] > 0).to_numpy()
        if later.any():
            assert not before[later].equals(after[later])


def _tiny():
    fakes = [
        FakeCar("A", date_of(0), 8.0, 17.0, session="first", energy=10.0),
        FakeCar("A", date_of(0), 8.0, 16.0, session="other0", energy=5.0),
        FakeCar("A", date_of(0), 9.0, 24 + 11.0, session="overnight", energy=12.0),
        FakeCar("A", date_of(1), 7.5, 15.0, session="second", energy=6.0),
        FakeCar("A", date_of(1), 9.0 - 24, 11.0, session="overnight", energy=12.0),
        FakeCar("A", date_of(1), 8.0, 16.0, session="nobody", energy=5.0),
        FakeCar("A", date_of(2), 8.0, 14.0, session="third", energy=7.0),
        FakeCar("A", date_of(2), 8.0, 16.0, session="other2", energy=5.0),
        FakeCar("A", date_of(2), 8.0, 16.0, session="fallback", energy=5.0),
    ]
    users = ["u", "v", "u", "u", "u", "", "u", "v", "w"]
    cars = _cars(fakes, users)
    cars.loc[cars["session_id"] == "fallback", "rate_rule"] = RATE_FALLBACK
    return lf.build_table(cars, rho=0.0).set_index(["date", "session_id"])


def test_user_history_by_hand():
    table = _tiny()
    # day 2, user u: earlier sessions "first" (8:00-17:00, 10 kWh), "overnight" (9:00 on day 0 to
    # 11:00 on day 1, 26 h, 12 kWh) and "second" (7:30-15:00, 7.5 h, 6 kWh)
    third = table.loc[(date_of(2), "third")]
    assert third["history_sessions"] == 3
    assert third["history_mean_stay_h"] == pytest.approx((9 + 26 + 7.5) / 3)
    assert third["history_median_stay_h"] == pytest.approx(9)
    assert third["history_mean_departure_clock_h"] == pytest.approx((17 + 11 + 15) / 3)
    assert third["history_median_departure_clock_h"] == pytest.approx(15)
    assert third["history_mean_energy_kwh"] == pytest.approx((10 + 12 + 6) / 3)
    # day 1, user u: "first" only; "overnight" is still plugged in at day 1's τ, so it is in
    # neither its own history nor that of "second"
    for session in ("second", "overnight"):
        row = table.loc[(date_of(1), session)]
        assert row["history_sessions"] == 1
        assert row["history_mean_stay_h"] == pytest.approx(9)
        assert row["history_median_departure_clock_h"] == pytest.approx(17)
        assert row["history_mean_energy_kwh"] == pytest.approx(10)
    # user v: one earlier session on day 0
    assert table.loc[(date_of(2), "other2")]["history_mean_stay_h"] == pytest.approx(8)
    # other features of one row
    assert third["hours_since_plug_in"] == pytest.approx(2.0)
    assert table.loc[(date_of(1), "overnight")]["hours_since_plug_in"] == pytest.approx(25.0)
    assert bool(table.loc[(date_of(1), "overnight")]["arrived_earlier_day"])
    assert third["fleet_at_tau"] == 3 and third["arrival_hour"] == 8
    day = datetime.date.fromisoformat(date_of(2))
    assert third["weekday"] == day.weekday() and third["month"] == day.month
    assert third["day_of_year"] == day.timetuple().tm_yday


@pytest.mark.parametrize(
    "date, session",
    [
        (0, "first"),  # first session of a known user
        (0, "overnight"),
        (1, "nobody"),  # no user id
        (2, "fallback"),  # the rate rule says no user id
    ],
)
def test_no_history_is_filled(date, session):
    row = _tiny().loc[(date_of(date), session)]
    assert row["history_sessions"] == 0
    for column in lf.HISTORY_COLUMNS[1:]:
        assert math.isnan(row[column])
    assert bool(row["user_known"]) == (session not in ("nobody", "fallback"))


def test_labels_and_the_frame_without_them():
    cars = _history_unit()
    table = lf.build_table(cars, rho=0.2)
    assert table["label_departure_h"].tolist() == cars["departure_exact"].tolist()
    assert table["label_rate_kw"].tolist() == cars["rate"].tolist()
    np.testing.assert_array_equal(table["label_room_kwh"], cars["room"])
    np.testing.assert_allclose(table["label_need_kwh"], 0.8 * cars["room"])
    assert not set(lf.LABEL_COLUMNS) & set(lf.feature_frame(table).columns)


def test_events_join_on_site_and_date():
    cars = _history_unit()
    assert not [c for c in lf.build_table(cars, rho=0.0).columns if c.startswith("event_")]
    events = pd.DataFrame({"site": ["A"], "date": [date_of(3)], "event_match": [1.0]})
    frame = lf.feature_frame(lf.build_table(cars, rho=0.0, events=events))
    assert frame.columns[-1] == "event_match"
    on_day = (frame["date"] == date_of(3)).to_numpy()
    assert (frame.loc[on_day, "event_match"] == 1.0).all()
    assert frame.loc[~on_day, "event_match"].isna().all()
    with pytest.raises(ValueError, match="event_"):
        lf.build_table(cars, rho=0.0, events=events.rename(columns={"event_match": "match"}))
    with pytest.raises(ValueError, match="one row"):
        lf.build_table(cars, rho=0.0, events=pd.concat([events, events]))
