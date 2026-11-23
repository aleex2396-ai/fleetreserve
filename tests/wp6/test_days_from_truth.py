"""Readings 1 to 3 of `days_from_truth` and the multi-day session, on hand-worked made-up tables."""

import pytest
from truth_tables_fake import FakeCar, tables, write

from fleetreserve import days_from_truth as dft

DAY = "2020-01-01"


def _days(cars, **kwargs):
    car_table, day_table = tables(cars)
    options = {"unit": "unit", "rho": 0.0, "no_fit": dft.NO_FIT_CAP, "small_fleet": dft.SMALL_KEEP}
    options.update(kwargs)
    return dft.unit_days(car_table, day_table, **options)


def test_reading1_departure_in_the_decision_hour_is_left_out_and_the_day_still_counts():
    cars = [
        FakeCar("A", DAY, 8.0, 10.5, rate=3.0, room=2.5, session="zero"),  # depart_hour 0
        FakeCar("A", DAY, 8.0, 14.0, room=4.0, session="b"),
        FakeCar("A", DAY, 8.0, 15.0, room=4.0, session="c"),
        FakeCar("A", DAY, 8.0, 16.0, room=4.0, session="d"),
    ]
    unit = _days(cars)
    (truth,) = unit.days
    assert [car.departure for car in truth.day.cars] == [4, 5, 6]
    assert truth.record.cars_at_tau == 4 and truth.record.cars_in_fleet == 3
    assert truth.record.depart_zero_cars == 1
    assert truth.record.depart_zero_room_kwh == pytest.approx(2.5)
    assert not truth.record.fewer_than_three
    assert [car.session_id for car in truth.fleet] == ["b", "c", "d"]


def _no_fit_cars():
    # rate 2, leaves at 12:30: depart_hour 2, so at most 4 kWh fit; room 7.
    return [
        FakeCar("A", DAY, 8.0, 12.5, rate=2.0, room=7.0, session="a"),
        FakeCar("A", DAY, 8.0, 14.0, room=4.0, session="b"),
        FakeCar("A", DAY, 8.0, 15.0, room=4.0, session="c"),
        FakeCar("A", DAY, 8.0, 16.0, room=4.0, session="d"),
    ]


def test_reading2_cap_sets_the_room_to_rate_times_departure_hour():
    (truth,) = _days(_no_fit_cars()).days
    assert truth.day.cars[0].room == pytest.approx(4.0)
    assert truth.record.no_fit_cars == 1
    assert truth.record.no_fit_energy_removed_kwh == pytest.approx(3.0)
    assert truth.record.no_fit_left_out == 0
    assert len(truth.day.cars) == 4


def test_reading2_leave_out_removes_the_car_and_records_its_room():
    unit = _days(_no_fit_cars(), no_fit=dft.NO_FIT_LEAVE_OUT)
    (truth,) = unit.days
    assert len(truth.day.cars) == 3
    assert truth.record.no_fit_cars == 1 and truth.record.no_fit_left_out == 1
    assert truth.record.no_fit_room_left_out_kwh == pytest.approx(7.0)
    assert truth.record.no_fit_energy_removed_kwh == 0.0
    assert truth.record.fewer_than_three is False


def test_reading2_uses_the_protected_share():
    # rho 0.5: the protected energy 3.5 fits in 4 kWh, so nothing is capped.
    (truth,) = _days(_no_fit_cars(), rho=0.5).days
    assert truth.record.no_fit_cars == 0
    assert truth.day.cars[0].room == pytest.approx(7.0)


def _small_day():
    return [
        FakeCar("A", DAY, 8.0, 10.2, session="a"),  # depart_hour 0
        FakeCar("A", DAY, 8.0, 14.0, session="b"),
        FakeCar("A", DAY, 8.0, 15.0, session="c"),
    ]


def test_reading3_small_fleet_is_kept_by_default_and_dropped_on_request():
    kept = _days(_small_day())
    assert len(kept.days) == 1 and kept.days[0].record.fewer_than_three
    assert kept.counts()["reading3_days_fewer_than_three"] == 1
    dropped = _days(_small_day(), small_fleet=dft.SMALL_DROP)
    assert dropped.days == () and dropped.dropped_small_days == 1


def test_reading3_empty_fleet_day_is_kept_with_no_cars():
    cars = [FakeCar("A", DAY, 8.0, 10.1 + 0.1 * j, session=f"z{j}") for j in range(3)]
    (truth,) = _days(cars).days
    assert truth.day.cars == () and truth.record.empty_fleet
    assert truth.day.n_recovery == 0


def test_recovery_hours_cover_the_latest_departure():
    cars = [FakeCar("A", DAY, 8.0, 10.0 + h, session=f"c{h}") for h in (3, 9, 13.5)]
    (truth,) = _days(cars).days
    assert truth.day.n_service == 6
    assert truth.day.n_recovery == 13 - 1 - 6


def test_non_counting_days_are_not_days_but_their_cars_get_the_rules():
    cars = [*_no_fit_cars(), FakeCar("A", "2020-01-02", 8.0, 10.5, session="lone")]
    unit = _days(cars)
    assert [t.record.date for t in unit.days] == [DAY]
    lone = unit.cars[unit.cars["session_id"] == "lone"].iloc[0]
    assert lone["outcome"] == dft.LEFT_OUT_DEPART_ZERO and not lone["counting"]


def test_multi_day_session_appears_on_both_days_with_its_own_room():
    # Session "long" plugged in on day 1 at 9:00 and out on day 2 at 15:00.
    day2 = "2020-01-02"
    cars = [
        FakeCar("A", DAY, 9.0, 24 + 15.0, room=6.0, session="long"),
        FakeCar("A", DAY, 8.0, 14.0, session="b"),
        FakeCar("A", DAY, 8.0, 15.0, session="c"),
        FakeCar("A", day2, 9.0 - 24, 15.0, room=0.0, session="long"),
        FakeCar("A", day2, 8.0, 14.0, session="e"),
        FakeCar("A", day2, 8.0, 15.0, session="f"),
    ]
    unit = _days(cars)
    first, second = unit.days
    long_first = first.fleet[[c.session_id for c in first.fleet].index("long")]
    long_second = second.fleet[[c.session_id for c in second.fleet].index("long")]
    assert not long_first.arrived_earlier_day and long_second.arrived_earlier_day
    assert long_first.departure_exact == pytest.approx(29.0)
    assert long_second.departure_exact == pytest.approx(5.0)
    assert first.day.n_recovery == 29 - 1 - 6
    assert second.record.earlier_day_cars == 1
    assert second.record.earlier_day_cars_with_room == 0
    assert first.record.earlier_day_cars == 0


def test_tables_that_disagree_are_refused():
    car_table, day_table = tables(_no_fit_cars())
    day_table.loc[0, "cars"] = "5"
    with pytest.raises(ValueError, match="disagree"):
        dft.unit_days(car_table, day_table, unit="u", rho=0.0, no_fit="cap", small_fleet="keep")


def test_unknown_readings_are_refused():
    with pytest.raises(ValueError):
        _days(_no_fit_cars(), no_fit="other")
    with pytest.raises(ValueError):
        _days(_no_fit_cars(), small_fleet="other")


def test_load_from_folder(tmp_path):
    write(tmp_path, "unit", _no_fit_cars())
    unit = dft.load_unit_days(tmp_path, "unit", rho=0.0, no_fit="cap", small_fleet="keep")
    assert len(unit.days) == 1 and unit.days[0].day.unit == "unit"
    with pytest.raises(FileNotFoundError):
        dft.load_unit_days(tmp_path, "other", rho=0.0, no_fit="cap", small_fleet="keep")
