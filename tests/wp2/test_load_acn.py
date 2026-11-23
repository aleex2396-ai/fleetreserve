"""ACN loader on made-up month files: parsing, UTC to Pacific time across both daylight-saving
changes, declared values, removals counted by reason, and loud failure on a wrong layout."""

import gzip
import json

import pandas as pd
import pytest
from fake_raw import acn_record, acn_time, raw_root, write_acn

from fleetreserve import load_acn
from fleetreserve.sessions import LayoutError, OutsideRawData


def _load(tmp_path, records, site="caltech", month="2019-03"):
    root = raw_root(tmp_path)
    write_acn(root, site, month, records)
    return load_acn.load(site, root)


def test_utc_converted_to_pacific_across_both_daylight_saving_changes(tmp_path):
    records = [
        # Spring change, Sunday 10 March 2019: 02:00 PST jumps to 03:00 PDT (10:00 UTC).
        acn_record("before-spring", "2019-03-10 09:30:00", "2019-03-10 12:00:00", 5.0),
        acn_record("after-spring", "2019-03-10 10:30:00", "2019-03-10 12:00:00", 5.0),
        # Autumn change, Sunday 3 November 2019: 01:00 to 02:00 PDT happens again as PST.
        acn_record("first-0130", "2019-11-03 08:30:00", "2019-11-03 12:00:00", 5.0),
        acn_record("second-0130", "2019-11-03 09:30:00", "2019-11-03 12:00:00", 5.0),
    ]
    root = raw_root(tmp_path)
    write_acn(root, "jpl", "2019-03", records[:2])
    write_acn(root, "jpl", "2019-11", records[2:])
    sessions, dropped = load_acn.load("jpl", root)
    assert dropped.empty
    local = dict(zip(sessions["session_id"], sessions["plug_in"], strict=True))
    assert local["before-spring"].isoformat() == "2019-03-10T01:30:00-08:00"
    assert local["after-spring"].isoformat() == "2019-03-10T03:30:00-07:00"
    assert local["first-0130"].isoformat() == "2019-11-03T01:30:00-07:00"
    assert local["second-0130"].isoformat() == "2019-11-03T01:30:00-08:00"
    assert str(sessions["plug_in"].dt.tz) == "America/Los_Angeles"


def test_fields_and_declared_first_entry(tmp_path):
    inputs = [
        {
            "modifiedAt": acn_time("2019-03-05 16:10:00"),
            "requestedDeparture": acn_time("2019-03-06 01:00:00"),
            "kWhRequested": 20.0,
        },
        {
            "modifiedAt": acn_time("2019-03-05 16:01:00"),
            "requestedDeparture": acn_time("2019-03-05 23:00:00"),
            "kWhRequested": 10.0,
        },
    ]
    record = acn_record(
        "s1",
        "2019-03-05 16:00:00",
        "2019-03-06 02:00:00",
        8.5,
        user="77",
        station="2-39-1-1",
        done="2019-03-05 18:00:00",
        inputs=inputs,
    )
    sessions, _ = _load(tmp_path, [record])
    row = sessions.iloc[0]
    assert row["user_id"] == "77" and row["station"] == "2-39-1-1"
    assert row["energy_kwh"] == 8.5 and row["nominal_kw"] == 6.66
    assert row["charge_end"].isoformat() == "2019-03-05T10:00:00-08:00"
    assert row["declared_departure"].isoformat() == "2019-03-05T15:00:00-08:00"  # earliest entry
    assert row["declared_energy_kwh"] == 10.0
    assert pd.isna(row["charge_start"])


def _entry(modified: str, departure: str, energy: float) -> dict:
    return {
        "modifiedAt": acn_time(modified),
        "requestedDeparture": acn_time(departure),
        "kWhRequested": energy,
    }


@pytest.mark.parametrize("listed", ["oldest first", "newest first", "oldest in the middle"])
def test_the_earliest_entry_is_used_whatever_the_order_it_is_listed_in(tmp_path, listed):
    oldest = _entry("2019-03-05 16:01:00", "2019-03-05 23:00:00", 10.0)
    middle = _entry("2019-03-05 16:10:00", "2019-03-06 01:00:00", 20.0)
    newest = _entry("2019-03-05 16:30:00", "2019-03-06 03:00:00", 30.0)
    inputs = {
        "oldest first": [oldest, middle, newest],
        "newest first": [newest, middle, oldest],
        "oldest in the middle": [middle, oldest, newest],
    }[listed]
    record = acn_record("s1", "2019-03-05 16:00:00", "2019-03-06 02:00:00", 8.5, inputs=inputs)
    sessions, _ = _load(tmp_path, [record])
    row = sessions.iloc[0]
    assert row["declared_departure"].isoformat() == "2019-03-05T15:00:00-08:00"
    assert row["declared_energy_kwh"] == 10.0
    assert row["declared_at"].isoformat() == "2019-03-05T08:01:00-08:00"


def test_midnight_crossing_session_is_kept_whole(tmp_path):
    record = acn_record("night", "2019-03-06 04:00:00", "2019-03-06 16:00:00", 10.0)
    sessions, dropped = _load(tmp_path, [record])
    assert dropped.empty
    row = sessions.iloc[0]
    assert row["plug_in"].date().isoformat() == "2019-03-05"  # 20:00 local
    assert row["plug_out"].date().isoformat() == "2019-03-06"  # 08:00 local


def test_removed_rows_are_counted_by_reason(tmp_path):
    records = [
        acn_record("ok", "2019-03-05 16:00:00", "2019-03-05 20:00:00", 8.0),
        acn_record("backwards", "2019-03-05 16:00:00", "2019-03-05 16:00:00", 8.0),
        acn_record("empty", "2019-03-05 16:00:00", "2019-03-05 20:00:00", 0.0),
        acn_record("ok", "2019-03-06 16:00:00", "2019-03-06 20:00:00", 8.0),
        # A station whose sessions average 30 kW: a fast charger.
        acn_record("fast1", "2019-03-05 16:00:00", "2019-03-05 17:00:00", 30.0, station="F"),
        acn_record("fast2", "2019-03-06 16:00:00", "2019-03-06 17:00:00", 30.0, station="F"),
        acn_record("fast3", "2019-03-07 16:00:00", "2019-03-07 18:00:00", 5.0, station="F"),
        # 25 kWh in one hour at a Level 2 station.
        acn_record("impossible", "2019-03-05 16:00:00", "2019-03-05 17:00:00", 25.0),
    ]
    sessions, dropped = _load(tmp_path, records)
    assert list(sessions["session_id"]) == ["ok"]
    counts = dropped["reason"].value_counts().to_dict()
    assert counts == {
        "plug-out not after plug-in": 1,
        "energy missing or not positive": 1,
        "duplicate session id": 1,
        "fast-charging station": 3,
        "average above the physical limit": 1,
    }
    assert len(sessions) + len(dropped) == len(records)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.pop("kWhDelivered"),
        lambda r: r.update(connectionTime="2019-03-05 16:00"),
        lambda r: r.update(timezone="America/New_York"),
    ],
)
def test_wrong_layout_fails_loudly(tmp_path, change):
    record = acn_record("s", "2019-03-05 16:00:00", "2019-03-05 20:00:00", 8.0)
    change(record)
    with pytest.raises(LayoutError):
        _load(tmp_path, [record])


def test_file_that_is_not_a_month_fails_loudly(tmp_path):
    root = raw_root(tmp_path)
    write_acn(root, "caltech", "2019-03", [])
    with gzip.open(root / "acn" / "caltech" / "extra.json.gz", "wb") as handle:
        handle.write(json.dumps([]).encode())
    with pytest.raises(LayoutError):
        load_acn.load("caltech", root)


def test_folder_outside_raw_is_refused(tmp_path):
    root = raw_root(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "caltech").mkdir(parents=True)
    (root / "acn").symlink_to(elsewhere)
    with pytest.raises(OutsideRawData):
        load_acn.load("caltech", root)
