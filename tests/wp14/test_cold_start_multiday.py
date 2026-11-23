"""A session plugged in over several days and the end of the local period (`cold_start_days`):
nothing after the latest decision time τ of the k local days reaches a row's features at k, but
each scored day's own rows as known at its own τ (reading C2a of `cold_start`).

The made-up units are `cold_cars` units (τ at 10:00, three cars a day, car j keeping the user id
`A-user<j>`, plugged in at 08:00); `_unit` replaces the session `A-<first>-0` by one listed on the
days `first` to `last` (each listing plugged in at 08:00 on day `first`) that plugs out at `out`
hours after midnight of day `last`.

1. By hand, a 6-day unit whose session `A-1-0` is listed on 2 and 3 January and plugs out at 13:00
   on 3 January, at k = 3: the end of the local period is 3 January 10:00; the rows known then are
   the sessions of 1 January and the two other sessions of 2 January; the local days' histories
   are as the whole table gives them; each scored day's history holds only the known sessions
   (user 0: one, the others two), with their stays and departures written out. At k = 1 a scored
   day's history is empty: every car of the first day is plugged in at its τ.
2. The cases of the review of the first build, on a 70-day unit (scored days: the 14 after the
   first 56): a session listed on days 55 and 56 (the last local day and the first scored day at
   k = 56) and one listed on days 27 to 30 (from the last local day at k = 28 to a day that is
   neither local nor scored). Moving the plug-out from 13:00 to 15:00 changes nothing in the frame
   at that k; the scored day after day 56 counts 55 earlier sessions of user 0 with the mean stay
   written out by hand. A change of a session that ended before the end of the local period does
   move the frame, so the comparison can fail.
3. The end of the local period of a pooled unit is the latest τ of its local site-days.
"""

import math
from types import SimpleNamespace

import pandas as pd
import pytest
from cold_start_fake import cold_cars, write_tables
from truth_tables_fake import FakeCar, date_of

from fleetreserve import cold_start_days
from fleetreserve.days_from_truth import load_unit_days

UNIT = "nrel-from-may-2019"


def _unit(folder, n_days: int, spans=()):
    """A `cold_cars(n_days)` unit with each span (first, last, out) a session of user 0 listed on
    the days first to last (module text); written to `folder` and loaded."""
    cars = cold_cars(n_days)
    for first, last, out in spans:
        session = f"A-{first}-0"
        cars = [car for car in cars if car.session != session]
        for day in range(first, last + 1):
            back = 24.0 * (day - first)
            cars.append(
                FakeCar(
                    "A",
                    date_of(day),
                    8.0 - back,
                    24.0 * (last - day) + out,
                    rate=2.0,
                    room=1.0,
                    session=session,
                )
            )
    cars.sort(key=lambda car: (car.date, car.session))
    folder.mkdir()
    write_tables(folder, UNIT, cars)
    return load_unit_days(folder, UNIT, rho=0.0, no_fit="cap", small_fleet="keep")


def _frame(unit, k: int) -> pd.DataFrame:
    scored = cold_start_days.scored_indices(len(cold_start_days.day_keys(unit)))
    return cold_start_days.frame_at(unit, k, scored)


# --- 1. by hand ----------------------------------------------------------------------------------


@pytest.fixture
def small(tmp_path):
    return _unit(tmp_path / "small", 6, [(1, 2, 13.0)])


def test_the_end_of_the_local_period_and_the_rows_known_then_by_hand(small):
    assert cold_start_days.local_end(small, 0) is None
    assert cold_start_days.local_end(small, 3) == pd.Timestamp("2020-01-03T10:00:00-06:00")
    known = cold_start_days.known_rows(small, 3)
    assert list(known["session_id"]) == ["A-0-0", "A-0-1", "A-0-2", "A-1-1", "A-1-2"]
    assert cold_start_days.known_rows(small, 0).empty


def test_a_session_still_plugged_in_at_the_end_of_the_local_period_is_no_history_by_hand(small):
    frame = cold_start_days.frame_at(small, 3, [4, 5])
    assert list(frame["date"]) == [
        *["2020-01-01"] * 3,
        *["2020-01-02"] * 3,
        *["2020-01-03"] * 4,
        *["2020-01-05"] * 3,
        *["2020-01-06"] * 3,
    ]
    assert list(frame["session_id"][6:10]) == ["A-1-0", "A-2-0", "A-2-1", "A-2-2"]
    # local days, as the whole table gives them: on 3 January the two-day session has not left
    assert list(frame["history_sessions"][:10]) == [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    # scored days: user 0 has only A-0-0; users 1 and 2 their sessions of 1 and 2 January
    scored = frame.iloc[10:]
    assert list(scored["history_sessions"]) == [1, 2, 2, 1, 2, 2]
    stays = [6.4, (3.3 + 7.61) / 2, (8.7 + 8.71) / 2]
    assert list(scored["history_mean_stay_h"]) == pytest.approx(stays * 2)
    assert list(scored["history_min_departure_h"]) == pytest.approx([4.4, 1.3, 6.7] * 2)


def test_at_k_one_a_scored_day_has_no_history_of_a_counting_day_by_hand(small):
    # every car of 1 January is plugged in at its τ, the end of the local period at k = 1
    frame = cold_start_days.frame_at(small, 1, [4, 5])
    assert list(frame["date"][:3]) == ["2020-01-01"] * 3
    assert list(frame["history_sessions"]) == [0] * 9


def test_the_plug_out_after_the_end_of_the_local_period_moves_nothing_by_hand(small, tmp_path):
    later = _unit(tmp_path / "later", 6, [(1, 2, 15.0)])
    pd.testing.assert_frame_equal(
        cold_start_days.frame_at(small, 3, [4, 5]), cold_start_days.frame_at(later, 3, [4, 5])
    )


# --- 2. the review's cases -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "first", "last", "extra"),
    [(56, 55, 56, 1), (28, 27, 30, 0)],
    ids=["listed-on-the-last-local-and-first-scored-day", "listed-into-a-day-between"],
)
def test_a_later_plug_out_of_a_session_listed_on_a_local_day_moves_nothing(
    tmp_path, k, first, last, extra
):
    early = _frame(_unit(tmp_path / "early", 70, [(first, last, 13.0)]), k)
    late = _frame(_unit(tmp_path / "late", 70, [(first, last, 15.0)]), k)
    # three rows on each local and scored day, and the session's second row on scored day 56
    assert len(early) == 3 * k + 3 * 14 + extra
    pd.testing.assert_frame_equal(early, late)


def test_the_first_scored_day_after_day_56_counts_only_the_known_sessions(tmp_path):
    frame = _frame(_unit(tmp_path / "unit", 70, [(55, 56, 13.0)]), 56)
    probe = frame[(frame["date"] == "2020-02-27") & (frame["session_id"] == "A-57-0")]
    assert len(probe) == 1
    # user 0's sessions A-0-0 to A-54-0, each 6.4 hours plus 0.01 per day of the week
    assert probe["history_sessions"].item() == 55
    mean_stay = 6.4 + 0.01 * sum(day % 7 for day in range(55)) / 55
    assert sum(day % 7 for day in range(55)) == 162
    assert probe["history_mean_stay_h"].item() == pytest.approx(mean_stay)


def test_a_change_before_the_end_of_the_local_period_moves_the_frame(tmp_path):
    base = _frame(_unit(tmp_path / "base", 70, [(55, 56, 13.0)]), 56)
    moved = _frame(_unit(tmp_path / "moved", 70, [(55, 56, 13.0), (54, 54, 15.4)]), 56)
    probe = (base["date"] == "2020-02-27") & (base["session_id"] == "A-57-0")
    before = base.loc[probe, "history_mean_stay_h"].item()
    after = moved.loc[probe.to_numpy(), "history_mean_stay_h"].item()
    assert not math.isclose(before, after)
    assert after - before == pytest.approx((15.4 - 14.45) / 55)


# --- 3. a pooled unit ----------------------------------------------------------------------------


def test_the_end_of_the_local_period_of_a_pooled_unit_is_its_latest_tau():
    def day(date, site, tau):
        return SimpleNamespace(record=SimpleNamespace(date=date, site=site, tau=tau))

    pooled = SimpleNamespace(
        unit="pooled",
        days=(
            day("2020-01-01", "B", "2020-01-01T12:00:00+01:00"),
            day("2020-01-01", "C", "2020-01-01T10:00:00+01:00"),
            day("2020-01-02", "B", "2020-01-02T12:00:00+01:00"),
        ),
    )
    assert cold_start_days.local_end(pooled, 1) == pd.Timestamp("2020-01-01T11:00:00Z")
    assert cold_start_days.local_end(pooled, 2) == pd.Timestamp("2020-01-01T11:00:00Z")
    assert cold_start_days.local_end(pooled, 3) == pd.Timestamp("2020-01-02T11:00:00Z")
