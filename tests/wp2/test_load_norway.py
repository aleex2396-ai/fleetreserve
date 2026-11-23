"""Norway loader on made-up split files: ";" and decimal comma, CET/CEST across both
daylight-saving changes, removals counted by reason, and loud failure on a wrong layout."""

import pytest
from fake_raw import NORWAY_HEADER, norway_row, raw_root, write_norway

from fleetreserve import load_norway
from fleetreserve.sessions import LayoutError


def test_fields_decimal_comma_and_nominal_cap(tmp_path):
    root = raw_root(tmp_path)
    rows = [norway_row("BER", "BER_1", "BER_9", "2019-11-28 19:08:00", "2019-11-29 07:08:00", 12.5)]
    write_norway(root, "development", rows)
    sessions, dropped = load_norway.load("development", root)
    assert dropped.empty
    row = sessions.iloc[0]
    assert row["site"] == "BER" and row["user_id"] == "BER_1" and row["session_id"] == "BER_9"
    assert row["energy_kwh"] == 12.5 and row["nominal_kw"] == 7.4
    assert row["plug_in"].isoformat() == "2019-11-28T19:08:00+01:00"
    assert row["role"] == "development" and row["station"] is None


def test_cet_and_cest_across_both_daylight_saving_changes(tmp_path):
    root = raw_root(tmp_path)
    rows = [
        # Spring, 31 March 2019: 02:00 CET jumps to 03:00 CEST.
        norway_row("ASK", "a", "spring", "2019-03-31 01:30:00", "2019-03-31 03:30:00", 1.0),
        norway_row("ASK", "a", "gap", "2019-03-31 02:30:00", "2019-03-31 05:00:00", 1.0),
        # Autumn, 27 October 2019: 02:00 to 03:00 happens twice.
        norway_row("ASK", "a", "autumn", "2019-10-27 01:30:00", "2019-10-27 03:30:00", 1.0),
        norway_row("ASK", "a", "doubled", "2019-10-27 02:30:00", "2019-10-27 05:00:00", 1.0),
    ]
    write_norway(root, "train", rows)
    sessions, dropped = load_norway.load("train", root)
    hours = (sessions["plug_out"] - sessions["plug_in"]).dt.total_seconds() / 3600
    assert dict(zip(sessions["session_id"], hours, strict=True)) == {"spring": 1.0, "autumn": 3.0}
    assert sessions.set_index("session_id").loc["spring", "plug_out"].isoformat() == (
        "2019-03-31T03:30:00+02:00"
    )
    assert sorted(dropped["session_id"]) == ["doubled", "gap"]


def test_removed_rows_are_counted_by_reason(tmp_path):
    root = raw_root(tmp_path)
    rows = [
        norway_row("TRO", "t", "ok", "2019-05-01 18:00:00", "2019-05-02 07:00:00", 10.0),
        norway_row("TRO", "t", "no-out", "2019-05-02 18:00:00", None, 10.0),
        norway_row("TRO", "t", "backwards", "2019-05-03 18:00:00", "2019-05-03 18:00:00", 1.0),
        norway_row("TRO", "t", "empty", "2019-05-04 18:00:00", "2019-05-05 07:00:00", 0.0),
        norway_row("TRO", "t", "ok", "2019-05-06 18:00:00", "2019-05-07 07:00:00", 10.0),
        norway_row("TRO", "t", "fast", "2019-05-08 18:00:00", "2019-05-08 19:00:00", 30.0),
    ]
    write_norway(root, "train", rows)
    sessions, dropped = load_norway.load("train", root)
    assert list(sessions["session_id"]) == ["ok"]
    assert list(dropped["reason"]) == [
        "no plug-out time recorded",
        "plug-out not after plug-in",
        "energy missing or not positive",
        "duplicate session id",
        "average above the physical limit",
    ]


def test_site_outside_the_files_role_fails_loudly(tmp_path):
    root = raw_root(tmp_path)
    rows = [norway_row("OSL_T", "x", "x1", "2019-05-01 18:00:00", "2019-05-02 07:00:00", 10.0)]
    write_norway(root, "train", rows)
    with pytest.raises(LayoutError):
        load_norway.load("train", root)


def test_missing_column_fails_loudly(tmp_path):
    root = raw_root(tmp_path)
    rows = [norway_row("ASK", "a", "s", "2019-05-01 18:00:00", "2019-05-02 07:00:00", 10.0)]
    write_norway(root, "train", rows, header=NORWAY_HEADER[:-1])
    with pytest.raises(LayoutError):
        load_norway.load("train", root)


def test_unreadable_time_fails_loudly(tmp_path):
    root = raw_root(tmp_path)
    rows = [norway_row("ASK", "a", "s", "2019-05-01 18:00:00", "2019-05-02 07:00:00", 10.0)]
    rows[0]["plugin_time"] = "01.05.2019 18:00"
    write_norway(root, "train", rows)
    with pytest.raises(LayoutError):
        load_norway.load("train", root)
