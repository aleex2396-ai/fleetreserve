"""NREL loader on a made-up CSV: NULL handling, Mountain time across both daylight-saving
changes, the nominal cap rule, removals counted by reason, and loud failure on a wrong layout."""

import pytest
from fake_raw import NREL_HEADER, nrel_row, raw_root, write_nrel

from fleetreserve import load_nrel
from fleetreserve.sessions import LayoutError


def _session(day: str, start="08:00:00", end="16:30:00", **fields):
    values = {
        "request_entry_time": f"{day} {start}",
        "expected_departure": f"{day} 16:00:00",
        "station": "01A",
        "start_date_time": f"{day} {start}",
        "end_date_time": f"{day} {end}",
        "energy_charged": "9.5",
    }
    return nrel_row(**{**values, **fields})


def test_fields_session_ids_and_nominal_rule(tmp_path):
    root = raw_root(tmp_path)
    rows = [
        _session(
            "2019-05-15",
            max_charge_power="6.1",
            start_charge="2019-05-15 09:00:00",
            termin_charge="2019-05-15 11:00:00",
            kwh_requested="10",
        ),
        _session("2019-05-16", max_charge_power="1.5", driverId="d2"),
        _session("2019-05-17", driverId="d3"),
    ]
    write_nrel(root, rows)
    sessions, dropped = load_nrel.load(root)
    assert dropped.empty
    assert list(sessions["session_id"]) == ["row-1", "row-2", "row-3"]
    assert list(sessions["nominal_kw"]) == [6.1, 3.3, 3.3]
    assert list(sessions["nominal_rule"]) == [
        "max_charge_power floored at 3.3 kW",
        "max_charge_power floored at 3.3 kW",
        "3.3 kW floor: max_charge_power missing",
    ]
    first = sessions.iloc[0]
    assert first["plug_in"].isoformat() == "2019-05-15T08:00:00-06:00"
    assert first["charge_end"].isoformat() == "2019-05-15T11:00:00-06:00"
    assert first["declared_energy_kwh"] == 10.0 and first["max_power_kw"] == 6.1
    assert first["site"] == "nrel" and first["role"] == "development"


def test_mountain_time_across_both_daylight_saving_changes(tmp_path):
    root = raw_root(tmp_path)
    rows = [
        _session("2019-03-09", start="23:30:00", end="23:59:00", energy_charged="1"),
        nrel_row(  # spans the spring change: 01:30 MST to 03:30 MDT is one hour
            request_entry_time="2019-03-10 01:30:00",
            start_date_time="2019-03-10 01:30:00",
            end_date_time="2019-03-10 03:30:00",
            energy_charged="1",
        ),
        nrel_row(  # inside the hour that does not exist
            request_entry_time="2019-03-10 02:30:00",
            start_date_time="2019-03-10 02:30:00",
            end_date_time="2019-03-10 05:00:00",
            energy_charged="1",
        ),
        nrel_row(  # starts in the doubled hour of the autumn change
            request_entry_time="2019-11-03 01:30:00",
            start_date_time="2019-11-03 01:30:00",
            end_date_time="2019-11-03 05:00:00",
            energy_charged="1",
        ),
        nrel_row(  # spans the autumn change: 00:30 MDT to 03:30 MST is four hours
            request_entry_time="2019-11-03 00:30:00",
            start_date_time="2019-11-03 00:30:00",
            end_date_time="2019-11-03 03:30:00",
            energy_charged="1",
        ),
    ]
    write_nrel(root, rows)
    sessions, dropped = load_nrel.load(root)
    hours = (sessions["plug_out"] - sessions["plug_in"]).dt.total_seconds() / 3600
    by_id = dict(zip(sessions["session_id"], hours, strict=True))
    assert by_id["row-2"] == 1.0
    assert by_id["row-5"] == 4.0
    assert sorted(dropped["session_id"]) == ["row-3", "row-4"]
    assert set(dropped["reason"]) == {"local time in the daylight-saving gap or doubled hour"}


def test_removed_rows_are_counted_by_reason(tmp_path):
    root = raw_root(tmp_path)
    rows = [
        _session("2019-05-15"),
        nrel_row(
            request_entry_time="2018-01-02 08:00:00",
            start_charge="2018-01-02 08:15:00",
            termin_charge="2018-01-02 10:00:00",
            energy_charged="4",
        ),
        _session("2019-05-16", end="08:00:00"),
        nrel_row(**{**_session("2019-05-17"), "energy_charged": "0"}),
        _session("2019-05-15"),  # repeats row 1 exactly
        _session("2019-05-18", end="08:30:00", energy_charged="12"),  # 24 kW for half an hour
    ]
    write_nrel(root, rows)
    sessions, dropped = load_nrel.load(root)
    assert list(sessions["session_id"]) == ["row-1"]
    assert dict(zip(dropped["session_id"], dropped["reason"], strict=True)) == {
        "row-2": "no plug-in or plug-out time recorded",
        "row-3": "plug-out not after plug-in",
        "row-4": "energy missing or not positive",
        "row-5": "duplicate row",
        "row-6": "average above the physical limit",
    }
    assert str(dropped.set_index("session_id").loc["row-2", "local_date"]) == "2018-01-02"


def test_plug_out_fallbacks_fill_missing_times_only_when_asked(tmp_path):
    root = raw_root(tmp_path)
    row = nrel_row(
        request_entry_time="2018-01-02 08:00:00",
        expected_departure="2018-01-02 17:00:00",
        start_charge="2018-01-02 08:15:00",
        termin_charge="2018-01-02 10:00:00",
        energy_charged="4",
    )
    write_nrel(root, [row])
    main, _ = load_nrel.load(root)
    assert main.empty
    by_charge, _ = load_nrel.load(root, missing_plug_out="charge_end")
    by_declared, _ = load_nrel.load(root, missing_plug_out="declared_departure")
    assert by_charge.iloc[0]["plug_out"].hour == 10
    assert by_declared.iloc[0]["plug_out"].hour == 17
    assert by_charge.iloc[0]["plug_in"].hour == 8


def test_missing_column_fails_loudly(tmp_path):
    root = raw_root(tmp_path)
    header = tuple(name for name in NREL_HEADER if name != "end_date_time")
    write_nrel(root, [_session("2019-05-15")], header=header)
    with pytest.raises(LayoutError):
        load_nrel.load(root)


def test_unreadable_time_fails_loudly(tmp_path):
    root = raw_root(tmp_path)
    write_nrel(root, [_session("2019-05-15", end="4:30 pm")])
    with pytest.raises(LayoutError):
        load_nrel.load(root)
