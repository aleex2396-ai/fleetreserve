"""The lower-tail history columns (`learn_features.TAIL_HISTORY_COLUMNS`), on made-up cars tables.

1. The ten columns by hand on a table with one user's eight sessions, a session over two days, a
   session over three days, another user and another site, and cars with no user id: the earlier
   sessions are the ones of `HISTORY_COLUMNS`, each session's departure value is its smallest exact
   departure over the days it is listed, the last session is the one with the latest plug-out.
2. Fill: count 0 and NaN for no history and for no user id; the same-weekday count 0 and a NaN
   median for a history with no session on the row's weekday.
3. No leak: a session never enters its own history (also over several days), nothing of the row's
   own day enters, a session still plugged in at the row's τ does not, a session of another user
   or site does not, and changing any outcome column of the row's own session or a later one
   leaves the tail columns unchanged.
4. The columns are in the table and the frame a model is given, and not in `FEATURE_COLUMNS`.
5. The wait to the next local 06:00 after a τ clock time.

Times are hours from midnight of a car's date at a fixed offset, τ at 10:00 (`truth_tables_fake`),
so a plug-out at hour h leaves h - 10 hours after the τ of the day it is listed on. 1 January 2020
is a Wednesday and `date_of(k)` is k days later.
"""

import math

import numpy as np
import pandas as pd
import pytest
from test_learn_features import _altered, _cars, _history_unit
from truth_tables_fake import FakeCar, date_of

from fleetreserve import learn_features as lf

TAIL = list(lf.TAIL_HISTORY_COLUMNS)


def _table() -> pd.DataFrame:
    def car(site, day, start, end, session, energy=5.0):
        return FakeCar(site, date_of(day), start, end, session=session, energy=energy)

    fakes = [
        car("A", 0, 8, 10.5, "u0"),  # departure value 0.5
        car("A", 1, 8, 10.8, "u1"),  # 0.8
        car("A", 2, 8, 32, "u2"),  # 22, plug-out at 8:00 on day 3
        car("A", 3, 8, 11, "u3"),  # 1, plug-out at 11:00 on day 3, after the one before
        car("A", 5, 8, 13.5, "u5"),  # 3.5
        car("A", 7, 8, 20, "u7"),  # 10
        car("A", 9, 8, 20, "u9"),  # 10
        car("B", 1, 8, 10.2, "uB"),  # the same user id at another site
        car("A", 2, 8, 10.1, "w2"),  # another user at the same site
        car("A", 4, 9, 35, "night"),  # a session over two days: 25 on day 4 ...
        car("A", 5, 9 - 24, 11, "night"),  # ... and 1 on day 5
        car("A", 6, 8, 15, "n6"),
        car("A", 0, 8, 60, "long"),  # a session over three days: 50, 26 and 2
        car("A", 1, 8 - 24, 36, "long"),
        car("A", 2, 8 - 48, 12, "long"),
        car("A", 3, 8, 14, "after"),
        car("A", 3, 8, 14, "nobody"),
        # user u, first listed before day 5, plugged out at 15:00 on day 5, after day 5's
        # session (13:30): 29 on day 4 and 5 on day 5
        car("A", 4, 9, 39, "uL"),
        car("A", 5, 9 - 24, 15, "uL"),
    ]
    users = ["u"] * 8 + ["w", "n", "n", "n", "l", "l", "l", "l", "", "u", "u"]
    return lf.build_table(_cars(fakes, users), rho=0.0).set_index(["date", "session_id"])


def _row(table: pd.DataFrame, day: int, session: str) -> pd.Series:
    return table.loc[(date_of(day), session)]


def test_the_columns_by_hand():
    row = _row(_table(), 7, "u7")  # a Wednesday; six earlier sessions of user u at site A
    assert row["history_sessions"] == 6
    assert row["history_min_departure_h"] == pytest.approx(0.5)
    # sorted values 0.5, 0.8, 1, 3.5, 5, 22: the 10% quantile is at position 0.5, the 25% at 1.25
    assert row["history_q10_departure_h"] == pytest.approx(0.65)
    assert row["history_q25_departure_h"] == pytest.approx(0.85)
    assert row["history_share_departure_under_3h"] == pytest.approx(0.5)
    # the wait from τ at 10:00 to 06:00 is 20 hours: only the value 22 left after it
    assert row["history_share_departure_before_6"] == pytest.approx(5 / 6)
    assert row["history_same_weekday_sessions"] == 1  # only day 0 was a Wednesday
    assert row["history_same_weekday_median_departure_h"] == pytest.approx(0.5)
    # the latest plug-out is day 5 at 15:00 (a session first listed on day 4), not the session
    # first listed last (day 5, plugged out at 13:30)
    assert row["history_last_departure_h"] == pytest.approx(5.0)
    assert row["history_hours_since_last_plug_out"] == pytest.approx(43.0)
    # the five latest plug-outs leave out the session of day 0 (0.5) and keep day 1's (0.8)
    assert row["history_last5_min_departure_h"] == pytest.approx(0.8)


def test_the_last_session_is_by_plug_out_and_the_last_five_drop_the_earliest():
    row = _row(_table(), 9, "u9")  # a Friday; seven earlier sessions, the first two drop out
    assert row["history_sessions"] == 7
    assert row["history_min_departure_h"] == pytest.approx(0.5)
    assert row["history_last_departure_h"] == pytest.approx(10.0)  # day 7, plugged out at 20:00
    assert row["history_hours_since_last_plug_out"] == pytest.approx(38.0)
    # the values 10, 5, 3.5, 1 and 22; day 1's 0.8 is out
    assert row["history_last5_min_departure_h"] == pytest.approx(1.0)
    assert row["history_same_weekday_sessions"] == 1  # day 2 was a Friday
    assert row["history_same_weekday_median_departure_h"] == pytest.approx(22.0)
    # the plug-out of day 2's session (8:00 on day 3) is before that of day 3's session (11:00)
    row = _row(_table(), 5, "u5")
    assert row["history_last_departure_h"] == pytest.approx(1.0)
    assert row["history_hours_since_last_plug_out"] == pytest.approx(47.0)


def test_a_session_over_several_days_counts_once_by_its_smallest_departure():
    table = _table()
    row = _row(table, 6, "n6")  # user n: the two-day session, 25 on day 4 and 1 on day 5
    assert row["history_sessions"] == 1
    for column in (
        "history_min_departure_h",
        "history_q10_departure_h",
        "history_q25_departure_h",
        "history_last_departure_h",
        "history_last5_min_departure_h",
    ):
        assert row[column] == pytest.approx(1.0)
    assert row["history_share_departure_under_3h"] == 1.0
    assert row["history_hours_since_last_plug_out"] == pytest.approx(23.0)
    row = _row(table, 3, "after")  # user l: the three-day session, 50, 26 and 2
    assert row["history_sessions"] == 1
    assert row["history_min_departure_h"] == pytest.approx(2.0)
    assert row["history_last_departure_h"] == pytest.approx(2.0)
    assert row["history_hours_since_last_plug_out"] == pytest.approx(22.0)


def test_no_session_is_in_its_own_history_nor_one_still_plugged_in_nor_one_of_another_day():
    table = _table()
    # the three-day session on each of its days, and the two-day one on day 4 and 5
    for day, session in ((0, "long"), (1, "long"), (2, "long"), (4, "night"), (5, "night")):
        row = _row(table, day, session)
        assert row["history_sessions"] == 0, (day, session)
        assert row["history_same_weekday_sessions"] == 0
        assert row[TAIL].drop("history_same_weekday_sessions").isna().all()
    # day 5, user u: the sessions of days 0 to 3 only; day 4's "night" is another user
    assert _row(table, 5, "u5")["history_sessions"] == 4
    # user n on day 6 holds the two-day session, user u on day 7 does not hold it
    assert _row(table, 7, "u7")["history_sessions"] == 6
    # the same user id at another site, and another user, do not enter
    assert _row(table, 2, "u2")["history_min_departure_h"] == pytest.approx(0.5)
    assert _row(table, 7, "u7")["history_min_departure_h"] == pytest.approx(0.5)


def test_no_history_and_no_user_id_are_filled_and_a_weekday_without_a_session_is_nan():
    table = _table()
    for row in (_row(table, 0, "u0"), _row(table, 3, "nobody")):
        assert row["history_same_weekday_sessions"] == 0
        assert row[TAIL].drop("history_same_weekday_sessions").isna().all()
    row = _row(table, 1, "u1")  # one earlier session (day 0), a Wednesday, the row a Thursday
    assert row["history_sessions"] == 1
    assert row["history_same_weekday_sessions"] == 0
    assert math.isnan(row["history_same_weekday_median_departure_h"])
    assert row["history_min_departure_h"] == pytest.approx(0.5)


def test_the_tail_columns_do_not_read_outcomes_of_the_row_or_later():
    cars = _history_unit()
    before = lf.build_table(cars, rho=0.0)
    assert (before["history_sessions"] >= 5).any()
    assert before["history_last5_min_departure_h"].notna().any()
    rng = np.random.default_rng(3)
    for cut in sorted(set(cars["date"]))[::3]:
        altered = _altered(cars, cut, rng)
        assert (altered["departure_exact"] != cars["departure_exact"]).any()
        after = lf.build_table(altered, rho=0.0)
        kept = (cars["date"] <= cut).to_numpy()
        pd.testing.assert_frame_equal(before.loc[kept, TAIL], after.loc[kept, TAIL])
        later = (cars["date"] > cut).to_numpy() & (before["history_sessions"] > 0).to_numpy()
        if later.any():  # the check can see a change
            assert not before.loc[later, TAIL].equals(after.loc[later, TAIL])


def test_the_columns_are_in_the_table_and_the_frame_and_not_in_the_old_features():
    cars = _history_unit()
    table = lf.build_table(cars, rho=0.0)
    assert len(TAIL) == 10 and len(set(TAIL)) == 10
    assert not set(TAIL) & set(lf.FEATURE_COLUMNS)
    assert not set(TAIL) & set(lf.DECLARED_COLUMNS)
    assert list(lf.feature_frame(table).columns[-len(TAIL) :]) == TAIL
    assert not set(TAIL) & set(lf.NOT_KNOWN_AT_TAU)
    assert table["history_same_weekday_sessions"].notna().all()


@pytest.mark.parametrize(
    "clock, wait", [(10.0, 20.0), (6.0, 24.0), (5.5, 0.5), (23.0, 7.0), (0.0, 6.0), (6.25, 23.75)]
)
def test_the_wait_to_the_next_early_morning_hour(clock, wait):
    assert lf._hours_to_next_early_clock(np.array([clock]))[0] == pytest.approx(wait)
