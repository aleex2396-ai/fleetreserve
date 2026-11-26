"""The Korean loader on made-up files in the layout the look at the real file found.

The layout, as the layout look of the Korean file reported it: a comma separator, UTF-8 without
a byte order mark, line ends `\\r\\n`, thirteen columns (`UserID`, `ChargerID`, `ChargerCompany`,
`Location`, `ChargerType`, `StartDay`, `StartTime`, `EndDay`, `EndTime`, `StartDatetime`,
`EndDatetime`, `Duration`, `Demand`), so no `ChargingsessionID`, no `ChargerCapacity` and no
`ChargerACDC`; `StartDatetime` and `EndDatetime` written `YYYY-MM-DD HH:MM` or `YYYY-MM-DD H:MM`
(no seconds, an hour of one digit); `StartTime` and `EndTime` `HH:MM:SS` or `H:MM:SS`; `Duration`
with negative values; `Demand` with a decimal point; the 14 location types in lower case. No
value of the real file is used: every row below is made up.
"""

from datetime import date

import pytest

from fleetreserve import load_korea
from fleetreserve.sessions import LayoutError

COLUMNS = [
    "UserID",
    "ChargerID",
    "ChargerCompany",
    "Location",
    "ChargerType",
    "StartDay",
    "StartTime",
    "EndDay",
    "EndTime",
    "StartDatetime",
    "EndDatetime",
    "Duration",
    "Demand",
]
HEADER = ",".join(COLUMNS)
LOCATIONS = [
    "accommodation",
    "apartment",
    "bus garage",
    "camping",
    "company",
    "golf",
    "hotel",
    "market",
    "public area",
    "public institution",
    "public parking lot",
    "resort",
    "restaurant",
    "sightseeing",
]
UNIT_OF = {
    "accommodation": "korea-lodging",
    "apartment": "korea-home",
    "bus garage": "korea-public-parking",
    "camping": "korea-lodging",
    "company": "korea-work",
    "golf": "korea-visits",
    "hotel": "korea-lodging",
    "market": "korea-visits",
    "public area": "korea-public-parking",
    "public institution": "korea-work",
    "public parking lot": "korea-public-parking",
    "resort": "korea-lodging",
    "restaurant": "korea-visits",
    "sightseeing": "korea-visits",
}
FAST = "fast charger (ChargerType 1)"
BACKWARDS = "plug-out not after plug-in"
DAY = "2022-06-01"


def _minutes(time: str) -> str:
    """The time column's text without its seconds, its hour as written ("7:05:30" -> "7:05")."""
    return time.rsplit(":", 1)[0]


def cells(
    location="apartment",
    start=(DAY, "8:00:00"),
    end=(DAY, "17:00:00"),
    *,
    user="17",
    charger="101",
    kind="0",
    duration="540",
    demand="9.5",
    **fields,
) -> dict:
    """One record of the real layout as a column -> text dict; `start` and `end` are (day, time)
    as the day and time columns write them, and the datetime columns are the same without the
    seconds. `fields` replaces a column's text."""
    row = {
        "UserID": user,
        "ChargerID": charger,
        "ChargerCompany": "3",
        "Location": location,
        "ChargerType": kind,
        "StartDay": start[0],
        "StartTime": start[1],
        "EndDay": end[0],
        "EndTime": end[1],
        "StartDatetime": f"{start[0]} {_minutes(start[1])}",
        "EndDatetime": f"{end[0]} {_minutes(end[1])}",
        "Duration": duration,
        "Demand": demand,
    }
    return row | fields


def file(records: list[dict], columns=COLUMNS, *, blank_after=None) -> bytes:
    """The records under the columns, `\\r\\n` after every line, no byte order mark."""
    lines = [",".join(columns)]
    for number, record in enumerate(records, start=1):
        lines.append(",".join(record[name] for name in columns))
        if number == blank_after:
            lines.append("")
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def every_location() -> list[dict]:
    """One kept slow-charger record at each of the 14 location types, so each portfolio keeps
    a row."""
    return [
        cells(location, user=str(10 + n), charger=str(200 + n))
        for n, location in enumerate(LOCATIONS)
    ]


def read(records: list[dict], **kwargs):
    return load_korea.from_bytes(file(records, **kwargs))


def row_of(sessions, number: int):
    return sessions.loc[sessions["session_id"] == str(number)].iloc[0]


def test_the_real_layout_is_read_to_the_written_out_table():
    records = every_location()
    # record 1 (accommodation): across midnight, an hour of one digit on the plug-out side
    records[0] = cells(
        "accommodation",
        (DAY, "23:59:59"),
        ("2022-06-02", "0:00:30"),
        user="10",
        charger="200",
        duration="1",
        demand="0.1",
    )
    # record 2 (apartment): the seconds of the time columns are not in the datetime columns
    records[1] = cells("apartment", (DAY, "7:05:30"), (DAY, "9:59:59"), user="0", charger="201")
    records += [
        cells("company", kind="1"),  # record 15: a fast charger
        cells("market", (DAY, "10:00:00"), (DAY, "9:30:00"), duration="-30"),  # record 16
    ]
    sessions, dropped = read(records)
    assert sorted(sessions["session_id"], key=int) == [str(n) for n in range(1, 15)]
    first = row_of(sessions, 1)
    assert first["site"] == "korea-lodging" and first["user_id"] == "10"
    assert first["plug_in"].isoformat() == "2022-06-01T23:59:59+09:00"
    assert first["plug_out"].isoformat() == "2022-06-02T00:00:30+09:00"
    second = row_of(sessions, 2)
    assert second["site"] == "korea-home" and second["user_id"] is None
    assert second["plug_in"].isoformat() == "2022-06-01T07:05:30+09:00"
    assert second["plug_out"].isoformat() == "2022-06-01T09:59:59+09:00"
    assert set(sessions["site"]) == set(load_korea.UNITS)
    assert set(sessions["station"]) == {str(200 + n) for n in range(14)}
    assert [tuple(row) for row in dropped.itertuples(index=False)] == [
        ("korea-work", "15", FAST, date(2022, 6, 1)),
        ("korea-visits", "16", BACKWARDS, date(2022, 6, 1)),
    ]


def test_the_14_location_labels_of_the_real_file_are_all_in_the_portfolio_table():
    assert set(LOCATIONS) == {name.casefold() for name in load_korea.PORTFOLIO_OF}
    sessions, dropped = read(every_location())
    assert len(sessions) == 14 and len(dropped) == 0
    for number, location in enumerate(LOCATIONS, start=1):
        assert row_of(sessions, number)["site"] == UNIT_OF[location]


def test_the_columns_the_real_file_lacks_are_not_required():
    for name in ("ChargingsessionID", "ChargerCapacity", "ChargerACDC"):
        assert name not in COLUMNS
    assert load_korea.REQUIRED == ("UserID", "ChargerID", "Location", "ChargerType", "Demand")
    # the same file with the two paper columns added gives the same table
    more = [row | {"ChargerCapacity": "2", "ChargerACDC": "1"} for row in every_location()]
    columns = (*COLUMNS, "ChargerCapacity", "ChargerACDC")
    assert read(more, columns=columns)[0].equals(read(every_location())[0])


def test_a_record_is_its_own_session_when_the_file_has_no_session_id():
    # two records equal in every column, after a blank line: two sessions numbered by their place
    # among the records (a blank line is not a record), none removed as a duplicate
    records = [*every_location(), cells(), cells()]
    sessions, dropped = read(records, blank_after=14)
    assert len(sessions) == 16 and len(dropped) == 0
    home = sessions.loc[sessions["site"] == "korea-home", "session_id"]
    assert sorted(home, key=int) == ["2", "15", "16"]
    assert sessions["session_id"].is_unique


def test_a_session_id_column_that_is_present_is_still_read_as_written():
    # the paper's own layout: the column is used, a repeated id is a duplicate (reading K9)
    records = [
        {"ChargingsessionID": f"A{n}", **record} for n, record in enumerate(every_location())
    ]
    records.append({"ChargingsessionID": "A0", **cells("hotel")})
    sessions, dropped = read(records, columns=("ChargingsessionID", *COLUMNS))
    assert sorted(sessions["session_id"]) == sorted(f"A{n}" for n in range(14))
    assert [tuple(row) for row in dropped.itertuples(index=False)] == [
        ("korea-lodging", "A0", "duplicate session id", date(2022, 6, 1))
    ]


def test_a_session_keeps_the_seconds_of_its_day_and_time_columns_and_an_hour_of_one_digit():
    # reading K5: the side's time is the day and time columns' (seconds kept, a one-digit hour
    # read as written); the datetime columns, with no seconds, are only a cross-check
    records = every_location()
    records[1] = cells("apartment", (DAY, "7:05:30"), (DAY, "9:07:59"), user="3", charger="201")
    # a two-digit hour, the datetime rounded up
    records[2] = cells(
        "bus garage", (DAY, "10:00:40"), (DAY, "10:30:15"), user="4", charger="202",
        StartDatetime=f"{DAY} 10:01",
    )  # fmt: skip
    sessions, _ = read(records)
    assert row_of(sessions, 2)["plug_in"].isoformat() == "2022-06-01T07:05:30+09:00"
    assert row_of(sessions, 2)["plug_out"].isoformat() == "2022-06-01T09:07:59+09:00"
    assert row_of(sessions, 3)["plug_in"].isoformat() == "2022-06-01T10:00:40+09:00"
    assert row_of(sessions, 3)["plug_out"].isoformat() == "2022-06-01T10:30:15+09:00"


# The start is 7:15:40 and the end 9:20:20 in the day and time columns; the datetime text under
# test, and whether it differs from the joined time by less than 60 seconds.
START_TIME, END_TIME = "7:15:40", "9:20:20"
CROSS_CHECK = [
    ("StartDatetime", f"{DAY} 7:15", True),  # seconds cut off
    ("StartDatetime", f"{DAY} 7:16", True),  # seconds rounded up (20 seconds)
    ("StartDatetime", f"{DAY} 07:16", True),
    ("StartDatetime", f"{DAY} 7:15:40", True),  # exact, with its seconds
    ("StartDatetime", f"{DAY} 7:16:39", True),  # 59 seconds after
    ("StartDatetime", f"{DAY} 7:14:41", True),  # 59 seconds before
    ("StartDatetime", f"{DAY} 7:16:40", False),  # 60 seconds after
    ("StartDatetime", f"{DAY} 7:14:40", False),  # 60 seconds before
    ("StartDatetime", f"{DAY} 7:14", False),  # 100 seconds before
    ("StartDatetime", f"{DAY} 7:17", False),  # 80 seconds after
    ("EndDatetime", f"{DAY} 9:20", True),  # cut off (20 seconds before)
    ("EndDatetime", f"{DAY} 9:21", True),  # rounded up (40 seconds after)
    ("EndDatetime", f"{DAY} 9:19:21", True),  # 59 seconds before
    ("EndDatetime", f"{DAY} 9:21:20", False),  # 60 seconds after
    ("EndDatetime", f"{DAY} 9:19", False),  # 80 seconds before
    ("EndDatetime", "2022-06-02 9:20", False),  # a day later
]


@pytest.mark.parametrize(("field", "text", "passes"), CROSS_CHECK)
def test_the_datetime_is_a_cross_check_that_passes_within_sixty_seconds(field, text, passes):
    # reading K13: a cut-off and a rounded seconds field both pass; 60 seconds or more is refused
    # with the count and no value; the session's time is the day and time columns' either way
    side = field.removesuffix("Datetime")
    records = every_location()
    records[0] = cells(
        start=(DAY, START_TIME), end=(DAY, END_TIME), user="10", charger="200", demand="0.5",
        **{field: text},
    )  # fmt: skip
    if passes:
        sessions, _ = read(records)
        assert row_of(sessions, 1)["plug_in"].isoformat() == "2022-06-01T07:15:40+09:00"
        assert row_of(sessions, 1)["plug_out"].isoformat() == "2022-06-01T09:20:20+09:00"
        return
    with pytest.raises(LayoutError) as refused:
        read(records)
    assert str(refused.value) == f"Korea {field} and {side}Day and {side}Time: 1 row disagrees"


def test_rows_that_differ_by_a_day_or_an_hour_are_counted():
    records = every_location()
    records[0] = cells(end=("2022-06-02", "17:00:00"), EndDatetime=f"{DAY} 17:00")
    records[1] = cells(start=(DAY, "8:00:00"), StartDatetime=f"{DAY} 9:00")
    records[2] = cells(start=(DAY, "8:00:00"), StartDatetime=f"{DAY} 8:00")
    with pytest.raises(LayoutError) as refused:
        read(records)
    assert str(refused.value) == "Korea StartDatetime and StartDay and StartTime: 1 row disagrees"


@pytest.mark.parametrize(
    "text",
    [
        "2022-6-1 07:00",
        "2022-06-01 7",
        "2022-06-01 007:00",
        "2022-06-01  7:00",
        "2022-06-01 7:0",
        "2022-06-01 7:00:5",
        "2022-06-01 07:00:60",
        "2022-06-01 07:00:00.5",
        "2022-06-01 24:00",
        "2022-06-01T07:00",
        "2022-06-01 ٧:00",
        " 2022-06-01 7:00",
    ],
)
def test_other_forms_of_a_datetime_are_still_refused(text):
    records = every_location()
    records[0] = cells(StartDatetime=text)
    with pytest.raises(LayoutError, match=r"^Korea StartDatetime: 1 times cannot be read"):
        read(records)


def test_a_file_with_the_datetime_columns_only_reads_them_at_the_minute():
    columns = [c for c in COLUMNS if c not in ("StartDay", "StartTime", "EndDay", "EndTime")]
    records = every_location()
    records[0] = cells(
        StartDatetime=f"{DAY} 8:05", EndDatetime="2022-06-02 0:30", user="9", charger="300"
    )
    sessions, dropped = read(records, columns=columns)
    assert len(sessions) == 14 and len(dropped) == 0
    assert set(sessions["plug_in"].dt.second) == {0}
    assert row_of(sessions, 1)["plug_out"].isoformat() == "2022-06-02T00:30:00+09:00"


@pytest.mark.parametrize("name", ["UserID", "ChargerID", "Location", "ChargerType", "Demand"])
def test_a_column_the_loader_reads_is_still_required(name):
    columns = [c for c in COLUMNS if c != name]
    with pytest.raises(LayoutError, match="^Korea file lacks columns: no separator gives UserID,"):
        read(every_location(), columns=columns)
