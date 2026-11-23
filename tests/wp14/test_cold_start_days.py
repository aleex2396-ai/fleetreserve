"""The days the cold start reads at k (`cold_start_days`), on a made-up unit written out by hand.

The unit: one site, cars on 1 to 8 January 2020 (`cold_cars(8, skip=(3,), thin=(5,))`): no car on
4 January, two cars on 6 January (a day that does not count), three cars on every other day. Each
car j keeps the user id `A-user<j>`; on 1 January the second car leaves 1.3 hours after τ, the
others 4.4 and 6.7 hours after (one hundredth of an hour later per day after).

1. The counting days in time order, the k-day prefixes, the local rows (the day that does not
   count enters only once the prefix passes it) and the scored days, by hand.
2. The frame at k = 2 with the scored days 5 and 8 January: its rows, and the user histories of
   the scored days built from the sessions of the two local days that had left by the end of the
   local period, 2 January 10:00 (one earlier session each, 1 January's, with its departure), not
   from the days between.
3. A pooled unit's site-days are ordered by date, then site; a unit whose days are out of that
   order is refused.
4. The plan's values of k, and refusals of a k outside the unit and of scored days before k.
"""

import dataclasses

import pytest
from cold_start_fake import cold_cars, write_tables

from fleetreserve import cold_start_days, learn_run
from fleetreserve.days_from_truth import load_unit_days

KEYS = [
    ("2020-01-01", "A"),
    ("2020-01-02", "A"),
    ("2020-01-03", "A"),
    ("2020-01-05", "A"),
    ("2020-01-07", "A"),
    ("2020-01-08", "A"),
]


def _unit(folder, name, cars):
    write_tables(folder, name, cars)
    return load_unit_days(folder, name, rho=0.0, no_fit="cap", small_fleet="keep")


@pytest.fixture
def unit(tmp_path):
    return _unit(tmp_path, "made-up", cold_cars(8, skip=(3,), thin=(5,)))


def _sessions(rows) -> list[str]:
    return list(rows["session_id"])


# --- 1. prefixes, local rows, scored days --------------------------------------------------------


def test_the_counting_days_are_in_time_order(unit):
    assert cold_start_days.day_keys(unit) == KEYS


def test_the_k_day_prefixes_and_the_scored_days_by_hand(unit):
    assert cold_start_days.local_indices(6, 0) == []
    assert cold_start_days.local_indices(6, 2) == [0, 1]
    assert cold_start_days.local_indices(6, 4) == [0, 1, 2, 3]
    # the counting days after the first 3 (largest k 3), the same whatever k is
    assert cold_start_days.scored_indices(6, 3) == [3, 4, 5]
    assert [KEYS[d][0] for d in cold_start_days.scored_indices(6, 3)] == [
        "2020-01-05",
        "2020-01-07",
        "2020-01-08",
    ]
    assert cold_start_days.scored_indices(6, 6) == []
    assert cold_start_days.scored_indices(70) == list(range(56, 70))


def test_the_local_rows_by_hand(unit):
    assert _sessions(cold_start_days.local_rows(unit, 0)) == []
    assert _sessions(cold_start_days.local_rows(unit, 2)) == [
        *("A-0-0", "A-0-1", "A-0-2"),
        *("A-1-0", "A-1-1", "A-1-2"),
    ]
    four = [f"A-{day}-{j}" for day in (0, 1, 2, 4) for j in range(3)]
    assert _sessions(cold_start_days.local_rows(unit, 4)) == four
    # the fifth counting day is 7 January: the two rows of 6 January, which does not count, enter
    assert _sessions(cold_start_days.local_rows(unit, 5)) == [
        *four,
        *("A-5-0", "A-5-1"),
        *("A-6-0", "A-6-1", "A-6-2"),
    ]
    assert len(cold_start_days.local_rows(unit, 6)) == len(unit.cars)


# --- 2. the frame at k ---------------------------------------------------------------------------


def test_the_frame_at_k_two_by_hand(unit):
    frame = cold_start_days.frame_at(unit, 2, [3, 5])
    assert list(frame["date"]) == [
        *["2020-01-01"] * 3,
        *["2020-01-02"] * 3,
        *["2020-01-05"] * 3,
        *["2020-01-08"] * 3,
    ]
    assert list(frame["session_id"]) == [
        *("A-0-0", "A-0-1", "A-0-2", "A-1-0", "A-1-1", "A-1-2"),
        *("A-4-0", "A-4-1", "A-4-2", "A-7-0", "A-7-1", "A-7-2"),
    ]
    # the local days' rows: as the whole table gives them; the scored days': the sessions that
    # had left by 2 January 10:00, the end of the local period, so 1 January's only (every car of
    # 2 January is plugged in at that τ)
    assert list(frame["history_sessions"]) == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    least = list(frame["history_min_departure_h"])
    assert least[3:6] == pytest.approx([4.4, 1.3, 6.7])
    assert least[6:] == pytest.approx([4.4, 1.3, 6.7, 4.4, 1.3, 6.7])
    assert "label_departure_h" not in frame.columns
    # the whole table would give 8 January six earlier sessions of the first two users, five of
    # the third (6 January has no third car), so the frame above is not the whole table's
    whole = learn_run.forecast_frame(False, {"made-up": unit}, ["made-up"])
    assert list(whole[whole["date"] == "2020-01-08"]["history_sessions"]) == [6, 6, 5]


def test_the_frame_at_k_zero_has_no_history(unit):
    frame = cold_start_days.frame_at(unit, 0, [3, 4, 5])
    assert list(frame["date"]) == [*["2020-01-05"] * 3, *["2020-01-07"] * 3, *["2020-01-08"] * 3]
    assert set(frame["history_sessions"]) == {0}


# --- 3. time order of a pooled unit --------------------------------------------------------------


def test_a_pooled_unit_is_ordered_by_date_then_site(tmp_path):
    pooled = _unit(tmp_path, "pooled", cold_cars(2, "C") + cold_cars(2, "B"))
    assert cold_start_days.day_keys(pooled) == [
        ("2020-01-01", "B"),
        ("2020-01-01", "C"),
        ("2020-01-02", "B"),
        ("2020-01-02", "C"),
    ]
    assert _sessions(cold_start_days.local_rows(pooled, 1)) == ["B-0-0", "B-0-1", "B-0-2"]


def test_days_out_of_time_order_are_refused(unit):
    shuffled = dataclasses.replace(unit, days=(unit.days[1], unit.days[0], *unit.days[2:]))
    with pytest.raises(ValueError, match="time order"):
        cold_start_days.day_keys(shuffled)


# --- 4. the values of k and refusals -------------------------------------------------------------


def test_the_plan_values_of_k():
    assert cold_start_days.LOCAL_DAYS == (0, 7, 14, 28, 56)
    assert cold_start_days.LARGEST == 56


def test_a_k_outside_the_unit_or_a_scored_day_before_k_is_refused(unit):
    with pytest.raises(ValueError):
        cold_start_days.local_indices(6, 7)
    with pytest.raises(ValueError):
        cold_start_days.local_indices(6, -1)
    with pytest.raises(ValueError):
        cold_start_days.frame_at(unit, 3, [2, 4])
    with pytest.raises(ValueError):
        cold_start_days.frame_at(unit, 0, [])
