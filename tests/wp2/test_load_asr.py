"""The a.s.r. Utrecht car park loader on hand-written files in the layout its README gives.

One file of eighteen made-up sessions: its expected session table and dropped rows are written
out below. It covers every removal reason, both sides of 1 September 2022 and 1 March 2023 (one
session overnight across the second), the spring daylight-saving gap and the autumn doubled
hour, a shared charging ID, a missing EV identifier, a missing charging station and an average of
20 kW (kept: the limit is 22 kW). Further tests: the ";" separator with a decimal comma, columns
in another order, a file of only the columns read, spaces around the header names and a byte
order mark, only the four stated missing markers read as missing, every shared charging ID of
the README, an average of exactly 22 kW kept, the duplicate rule comparing columns the table
does not use, and loud failure on a missing column (with its whole message), an unreadable
number, a time not written exactly YYYY-MM-DD hh:mm:ss or YYYY-MM-DDThh:mm:ss (each field one
digit short in turn, and digits that are not ASCII; the form with a space is read, see
`test_load_asr_layout.py`), or a missing plug-in time.

Text fields are kept as written: users and stations of digits keep their leading zeros, also
beside a missing user, and users "TRUE" and "FALSE" stay words. A list of malformed files (a
missing value in each column in turn, impossible times of the right form, an empty file, blank
lines only, a header only, bytes that are not UTF-8, a quoted field holding the separator or a
line break, a NUL character, one field too many in the first or a later record, two fields too
few, a line of spaces, a quote never closed, text after a closing quote, a column named twice,
also with spaces around it, carriage returns as line ends, a field over the csv module's limit)
each give exactly `LayoutError` with its stated message, or a table with stated removals only.
Any other error while reading is a `LayoutError` with that error as its cause.
"""

from datetime import date

import pandas as pd
import pytest
from fake_heldout import ASR_FILE, ASR_HEADER, asr_csv, asr_row, sealed_repo

from fleetreserve import heldout_loader, load_asr
from fleetreserve.sessions import COLUMNS, LayoutError


@pytest.fixture(autouse=True)
def _rows_not_units(monkeypatch):
    """The tests of this file are about the row rules, on files that hold one or a few of the
    named units. The check that each named unit keeps a row is switched off here; it is tested
    on its own in the `_units` test file of the loader, which reads a file of every unit with the
    check on."""
    monkeypatch.setattr(load_asr, "require_periods", lambda sessions, dropped: None)


ROWS = [
    asr_row("EV001", "2022-08-31T08:00:00", "2022-08-31T17:00:00", 20.5, evse="E1"),
    asr_row("EV002", "2022-09-01T08:30:00", "2022-09-01T16:00:00", 15.0, evse="E2", channel="2"),
    asr_row("EV003", "2023-02-28T09:00:00", "2023-02-28T18:00:00", 12.25, evse="E3"),
    asr_row("EV004", "2023-03-01T07:45:00", "2023-03-01T16:15:00", 22.0, evse="E4", rail="M"),
    asr_row("EV107", "2021-05-10T08:00:00", "2021-05-10T12:00:00", 8.0, evse="E5"),
    # Spring, 27 March 2022: 02:00 CET jumps to 03:00 CEST; the session lasts one real hour.
    asr_row("EV005", "2022-03-27T01:30:00", "2022-03-27T03:30:00", 3.0, evse=None, rail=None),
    asr_row("EV006", "2022-03-27T02:30:00", "2022-03-27T05:00:00", 3.0, evse="E6"),
    # Autumn, 30 October 2022: 02:00 to 03:00 happens twice.
    asr_row("EV007", "2022-10-30T02:30:00", "2022-10-30T06:00:00", 3.0, evse="E7"),
    asr_row("EV008", "2022-05-02T08:00:00", None, 10.0, evse="E8"),
    asr_row("EV009", "2022-05-03T08:00:00", "2022-05-03T08:00:00", 1.5, evse="E9"),
    asr_row("EV010", "2022-05-04T08:00:00", "2022-05-04T17:00:00", 0.0, evse="E10"),
    asr_row("EV001", "2022-08-31T08:00:00", "2022-08-31T17:00:00", 20.5, evse="E1"),
    asr_row("EV011", "2022-06-01T08:00:00", "2022-06-01T09:00:00", 30.0, evse="E11"),
    asr_row(None, "2024-01-15T08:00:00", "2024-01-15T17:30:00", 9.5, evse="E12"),
    asr_row("EV013", "2022-05-05T08:00:00", "2022-05-05T17:00:00", None, evse="E13"),
    # Below the README's own limits (1 kWh, 10 minutes): kept, the loader does not filter again.
    asr_row("EV014", "2023-06-01T10:00:00", "2023-06-01T10:05:00", 0.5, evse="E14"),
    # Overnight across 1 March 2023: its unit and regime follow the day it plugged in.
    asr_row("EV015", "2023-02-28T20:00:00", "2023-03-01T07:00:00", 5.5, evse="E15"),
    # 20 kW on average: above the US limit of 19.2 kW, below the 22 kW used here, so kept.
    asr_row("EV016", "2022-06-02T08:00:00", "2022-06-02T09:00:00", 20.0, evse="E16"),
]

EXPECTED = [
    # session_id, user_id, station, plug_in, plug_out, energy_kwh
    ("row-5", None, "E5", "2021-05-10T08:00:00+02:00", "2021-05-10T12:00:00+02:00", 8.0),
    ("row-6", "EV005", None, "2022-03-27T01:30:00+01:00", "2022-03-27T03:30:00+02:00", 3.0),
    ("row-18", "EV016", "E16", "2022-06-02T08:00:00+02:00", "2022-06-02T09:00:00+02:00", 20.0),
    ("row-1", "EV001", "E1", "2022-08-31T08:00:00+02:00", "2022-08-31T17:00:00+02:00", 20.5),
    ("row-2", "EV002", "E2", "2022-09-01T08:30:00+02:00", "2022-09-01T16:00:00+02:00", 15.0),
    ("row-3", "EV003", "E3", "2023-02-28T09:00:00+01:00", "2023-02-28T18:00:00+01:00", 12.25),
    ("row-17", "EV015", "E15", "2023-02-28T20:00:00+01:00", "2023-03-01T07:00:00+01:00", 5.5),
    ("row-4", "EV004", "E4", "2023-03-01T07:45:00+01:00", "2023-03-01T16:15:00+01:00", 22.0),
    ("row-16", "EV014", "E14", "2023-06-01T10:00:00+02:00", "2023-06-01T10:05:00+02:00", 0.5),
    ("row-14", None, "E12", "2024-01-15T08:00:00+01:00", "2024-01-15T17:30:00+01:00", 9.5),
]

EXPECTED_DROPPED = [
    # session_id, reason, local_date
    ("row-9", "no plug-out time recorded", date(2022, 5, 2)),
    ("row-7", "local time in the daylight-saving gap or doubled hour", date(2022, 3, 27)),
    ("row-8", "local time in the daylight-saving gap or doubled hour", date(2022, 10, 30)),
    ("row-10", "plug-out not after plug-in", date(2022, 5, 3)),
    ("row-11", "energy missing or not positive", date(2022, 5, 4)),
    ("row-15", "energy missing or not positive", date(2022, 5, 5)),
    ("row-12", "duplicate row", date(2022, 8, 31)),
    ("row-13", "average above the physical limit", date(2022, 6, 1)),
]

PERIODS = {
    # session_id: (unit, regime)
    "row-5": ("asr-before-march-2023", "free"),
    "row-6": ("asr-before-march-2023", "free"),
    "row-18": ("asr-before-march-2023", "free"),
    "row-1": ("asr-before-march-2023", "free"),
    "row-2": ("asr-before-march-2023", "paid"),
    "row-3": ("asr-before-march-2023", "paid"),
    "row-17": ("asr-before-march-2023", "paid"),
    "row-4": ("asr-from-march-2023", "paid-controller"),
    "row-16": ("asr-from-march-2023", "paid-controller"),
    "row-14": ("asr-from-march-2023", "paid-controller"),
}


def test_the_hand_written_file_gives_the_written_out_table():
    sessions, dropped = load_asr.from_bytes(asr_csv(ROWS))
    assert list(sessions.columns) == list(COLUMNS)
    got = [
        (
            row.session_id,
            row.user_id,
            row.station,
            row.plug_in.isoformat(),
            row.plug_out.isoformat(),
            row.energy_kwh,
        )
        for row in sessions.itertuples()
    ]
    assert got == EXPECTED
    assert set(sessions["site"]) == {"asr"} and set(sessions["role"]) == {"held-out"}
    assert set(sessions["nominal_kw"]) == {11.0} and set(sessions["nominal_rule"]) == {"source cap"}
    assert sessions["max_power_kw"].isna().all() and sessions["declared_energy_kwh"].isna().all()
    for name in ("charge_start", "charge_end", "declared_departure"):
        assert sessions[name].isna().all()
        assert str(sessions[name].dt.tz) == "Europe/Amsterdam"
    assert list(dropped.columns) == ["site", "session_id", "reason", "local_date"]
    got_dropped = list(dropped[["session_id", "reason", "local_date"]].itertuples(index=False))
    assert [tuple(row) for row in got_dropped] == EXPECTED_DROPPED
    assert set(dropped["site"]) == {"asr"}


def test_removals_are_counted_by_reason_in_the_stated_order():
    _, dropped = load_asr.from_bytes(asr_csv(ROWS))
    counts = dropped["reason"].value_counts(sort=False)
    assert list(counts.items()) == [
        ("no plug-out time recorded", 1),
        ("local time in the daylight-saving gap or doubled hour", 2),
        ("plug-out not after plug-in", 1),
        ("energy missing or not positive", 2),
        ("duplicate row", 1),
        ("average above the physical limit", 1),
    ]
    assert list(dropped["reason"].drop_duplicates()) == list(load_asr.REASONS)


def test_the_hour_across_the_spring_change_is_one_real_hour():
    sessions, _ = load_asr.from_bytes(asr_csv(ROWS))
    row = sessions.set_index("session_id").loc["row-6"]
    assert (row["plug_out"] - row["plug_in"]) == pd.Timedelta(hours=1)


def test_both_sides_of_each_regime_date_land_in_the_right_unit_and_regime():
    sessions, _ = load_asr.from_bytes(asr_csv(ROWS))
    periods = load_asr.periods(sessions)
    assert list(periods.columns) == ["unit", "regime"]
    assert periods.index.equals(sessions.index)
    got = {sid: tuple(p) for sid, p in zip(sessions["session_id"], periods.to_numpy(), strict=True)}
    assert got == PERIODS
    assert load_asr.unit_of(date(2023, 2, 28)) == "asr-before-march-2023"
    assert load_asr.unit_of(date(2023, 3, 1)) == "asr-from-march-2023"
    assert load_asr.regime_of(date(2022, 8, 31)) == "free"
    assert load_asr.regime_of(date(2022, 9, 1)) == "paid"
    assert load_asr.regime_of(date(2023, 2, 28)) == "paid"
    assert load_asr.regime_of(date(2023, 3, 1)) == "paid-controller"


def test_semicolon_and_decimal_comma_give_the_same_table():
    comma, comma_dropped = load_asr.from_bytes(asr_csv(ROWS))
    semicolon, semicolon_dropped = load_asr.from_bytes(asr_csv(ROWS, sep=";"))
    assert b"12,25" in asr_csv(ROWS, sep=";")
    pd.testing.assert_frame_equal(comma, semicolon)
    pd.testing.assert_frame_equal(comma_dropped, semicolon_dropped)


def test_columns_are_found_by_name_in_any_order():
    header = tuple(reversed(ASR_HEADER))
    reordered, _ = load_asr.from_bytes(asr_csv(ROWS, header=header))
    plain, _ = load_asr.from_bytes(asr_csv(ROWS))
    pd.testing.assert_frame_equal(reordered, plain)


@pytest.mark.parametrize("name", load_asr.COLUMNS)
def test_a_missing_column_is_refused(name):
    header = tuple(column for column in ASR_HEADER if column != name)
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(asr_csv(ROWS, header=header))
    assert str(refused.value).startswith(
        "a.s.r. file lacks columns: no separator gives "
        "EV_id_x, start_datetime, end_datetime, total_energy, evse_id; header fields as found "
    )


def test_a_file_of_only_the_columns_read_gives_the_same_table():
    only, only_dropped = load_asr.from_bytes(asr_csv(ROWS, header=load_asr.COLUMNS))
    plain, plain_dropped = load_asr.from_bytes(asr_csv(ROWS))
    pd.testing.assert_frame_equal(only, plain)
    pd.testing.assert_frame_equal(only_dropped, plain_dropped)


def test_spaces_around_the_header_names_and_a_byte_order_mark_give_the_same_table():
    plain = asr_csv(ROWS)
    header, rest = plain.split(b"\n", 1)
    padded = b",".join(b" " + name + b" " for name in header.split(b",")) + b"\n" + rest
    expected = load_asr.from_bytes(plain)
    for content in (padded, b"\xef\xbb\xbf" + plain, b"\xef\xbb\xbf" + padded):
        got = load_asr.from_bytes(content)
        pd.testing.assert_frame_equal(got[0], expected[0])
        pd.testing.assert_frame_equal(got[1], expected[1])


def test_only_the_four_stated_markers_are_missing():
    markers = ["", "NA", "NaN", "nan", "N/A", "NULL", "None", "null", "#N/A"]
    rows = [
        asr_row(f"EV9{n}", f"2022-05-1{n % 9}T08:00:00", f"2022-05-1{n % 9}T17:00:00", 5.0, evse=m)
        for n, m in enumerate(markers)
    ]
    sessions, _ = load_asr.from_bytes(asr_csv(rows))
    stations = dict(zip(sessions["user_id"], sessions["station"], strict=True))
    assert [stations[f"EV9{n}"] for n in range(len(markers))] == [
        None,
        None,
        None,
        None,
        "N/A",
        "NULL",
        "None",
        "null",
        "#N/A",
    ]


README_SHARED_IDS = [
    "EV107",
    "EV191",
    "EV132",
    "EV148",
    "EV342",
    "EV242",
    "EV130",
    "EV173",
    "EV131",
    "EV384",
    "EV318",
    "EV127",
    "EV257",
    "EV192",
    "EV129",
    "EV211",
    "EV323",
    "EV343",
    "EV161",
    "EV456",
]


def test_every_shared_charging_id_of_the_readme_gives_a_missing_user():
    assert frozenset(README_SHARED_IDS) == load_asr.SHARED_IDS and len(README_SHARED_IDS) == 20
    rows = [
        asr_row(ev, f"2022-05-{n + 1:02d}T08:00:00", f"2022-05-{n + 1:02d}T17:00:00", 5.0)
        for n, ev in enumerate([*README_SHARED_IDS, "EV999"])
    ]
    sessions, _ = load_asr.from_bytes(asr_csv(rows))
    assert list(sessions["user_id"]) == [None] * 20 + ["EV999"]


def test_an_average_of_exactly_the_limit_is_kept():
    rows = [asr_row("EV040", "2022-06-03T08:00:00", "2022-06-03T09:00:00", 22.0, evse="E40")]
    sessions, dropped = load_asr.from_bytes(asr_csv(rows))
    assert list(sessions["energy_kwh"]) == [22.0] and dropped.empty


def test_an_unreadable_number_is_refused():
    rows = [*ROWS[:3], asr_row("EV020", "2022-05-06T08:00:00", "2022-05-06T17:00:00", "12 kWh")]
    with pytest.raises(LayoutError, match="total_energy: 1 values are not numbers"):
        load_asr.from_bytes(asr_csv(rows))


@pytest.mark.parametrize(
    "text",
    [
        "2022-5-6T08:00:00",
        "2022-05-06T8:00:00",
        "2022-05-06T08:00:00Z",
        "2022-05-06T08:00:00+02:00",
        "2022-05-06T08:00:00.000",
        "2022-05-06  08:00:00",
        "2022-05-06_08:00:00",
        " 2022-05-06T08:00:00",
        "2022-05-06T08:00:00 ",
        "2022-05-06T08:00",
        "06-05-2022T08:00:00",
        "202-05-06T08:00:00",
        "2022-5-06T08:00:00",
        "2022-05-6T08:00:00",
        "2022-05-06T08:0:00",
        "2022-05-06T08:00:0",
        "２０２２-05-06T08:00:00",
    ],
)
def test_an_unreadable_time_is_refused(text):
    rows = [*ROWS[:3], asr_row("EV020", "2022-05-06T07:00:00", text, 5.0)]
    with pytest.raises(LayoutError, match="end_datetime: 1 times cannot be read"):
        load_asr.from_bytes(asr_csv(rows))


def test_the_duplicate_rule_compares_every_column_of_the_file():
    first = asr_row("EV030", "2022-05-09T08:00:00", "2022-05-09T17:00:00", 9.0, evse="E30")
    rows = [
        first,
        dict(first),
        {**first, "capacity_kwh": "77"},
        {**first, "channel": "2"},
    ]
    sessions, dropped = load_asr.from_bytes(asr_csv(rows))
    assert list(sessions["session_id"]) == ["row-1", "row-3", "row-4"]
    assert list(zip(dropped["session_id"], dropped["reason"], strict=True)) == [
        ("row-2", "duplicate row")
    ]


def test_a_missing_plug_in_time_is_refused():
    rows = [*ROWS[:3], asr_row("EV020", None, "2022-05-06T17:00:00", 5.0)]
    with pytest.raises(LayoutError, match="no plug-in time"):
        load_asr.from_bytes(asr_csv(rows))


def _two_days(users, stations):
    return [
        asr_row(user, f"2022-05-0{n + 2}T08:00:00", f"2022-05-0{n + 2}T17:00:00", 5.0, evse=evse)
        for n, (user, evse) in enumerate(zip(users, stations, strict=True))
    ]


@pytest.mark.parametrize(
    "users, stations",
    [
        (["007", "1"], ["0012", "12"]),
        (["007", "1", None], ["0012", "12", "13"]),
        (["TRUE", "FALSE"], ["1", "0"]),
    ],
    ids=["digits", "digits-and-missing", "words-of-truth"],
)
def test_text_fields_that_look_like_numbers_or_truth_are_kept_as_written(users, stations):
    sessions, dropped = load_asr.from_bytes(asr_csv(_two_days(users, stations)))
    assert list(sessions["user_id"]) == users
    assert list(sessions["station"]) == stations
    assert dropped.empty


A_HEADER = "EV_id_x,start_datetime,end_datetime,total_energy,rail,evse_id,channel"
A_ROW_1 = "EV001,2022-08-31T08:00:00,2022-08-31T17:00:00,20.5,B,E1,1"
A_ROW_2 = "EV002,2022-09-01T08:30:00,2022-09-01T16:00:00,15.0,B,E2,2"
WHAT = "a.s.r. file"
LACKS = (
    "a.s.r. file lacks columns: no separator gives "
    "EV_id_x, start_datetime, end_datetime, total_energy, evse_id; header fields as found "
    "(a field is given as text only when it is printable ASCII with no digit, else as "
    "its shape; each cut at 40 characters, at most 40 shown): comma: 0 fields []; "
    "semicolon: 0 fields []; "
    "tab: 0 fields []; bar: 0 fields []; the file has no data record"
)
TIME_SHAPE = '; field shapes: "9999-99-99a99:99:99" x 1'


def _file(*lines: str, end: str = "\n") -> bytes:
    return (end.join(lines) + end).encode("utf-8")


def _with(row: str, field: int, value: str) -> str:
    cells = row.split(",")
    cells[field] = value
    return ",".join(cells)


BOTH = (["row-1", "row-2"], ["EV001", "EV002"], ["E1", "E2"], [])
MALFORMED = {
    # name: (file, expected): an error message, or (session ids, users, stations, reasons)
    "no user": (
        _file(A_HEADER, _with(A_ROW_1, 0, ""), A_ROW_2),
        (*BOTH[:1], [None, "EV002"], *BOTH[2:]),
    ),
    "no plug-in": (
        _file(A_HEADER, _with(A_ROW_1, 1, ""), A_ROW_2),
        "a.s.r.: a row has no plug-in time (start_datetime): 1 rows",
    ),
    "no plug-out": (
        _file(A_HEADER, _with(A_ROW_1, 2, ""), A_ROW_2),
        (["row-2"], ["EV002"], ["E2"], ["no plug-out time recorded"]),
    ),
    "no energy": (
        _file(A_HEADER, _with(A_ROW_1, 3, "NA"), A_ROW_2),
        (["row-2"], ["EV002"], ["E2"], ["energy missing or not positive"]),
    ),
    "no rail": (_file(A_HEADER, _with(A_ROW_1, 4, ""), A_ROW_2), BOTH),
    "no station": (
        _file(A_HEADER, _with(A_ROW_1, 5, "nan"), A_ROW_2),
        (*BOTH[:2], [None, "E2"], []),
    ),
    "no channel": (_file(A_HEADER, _with(A_ROW_1, 6, ""), A_ROW_2), BOTH),
    "30 February": (
        _file(A_HEADER, _with(A_ROW_1, 1, "2023-02-30T10:00:00"), A_ROW_2),
        "a.s.r. start_datetime: 1 times cannot be read" + TIME_SHAPE,
    ),
    "hour 24": (
        _file(A_HEADER, _with(A_ROW_1, 1, "2023-01-01T24:00:00"), A_ROW_2),
        "a.s.r. start_datetime: 1 times cannot be read" + TIME_SHAPE,
    ),
    "month 13": (
        _file(A_HEADER, A_ROW_1, _with(A_ROW_2, 2, "2023-13-01T10:00:00")),
        "a.s.r. end_datetime: 1 times cannot be read" + TIME_SHAPE,
    ),
    "minute 60": (
        _file(A_HEADER, A_ROW_1, _with(A_ROW_2, 2, "2023-01-01T10:60:00")),
        "a.s.r. end_datetime: 1 times cannot be read" + TIME_SHAPE,
    ),
    "empty": (b"", LACKS),
    "blank lines only": (b"\n\n", LACKS),
    "header only": (_file(A_HEADER), ([], [], [], [])),
    "not UTF-8": (
        _file(A_HEADER, A_ROW_1) + b"\xff\n",
        f"{WHAT} is not UTF-8 text (byte {len(_file(A_HEADER, A_ROW_1))})",
    ),
    "separator in quotes": (
        _file(A_HEADER, _with(A_ROW_1, 0, '"EV,001"'), A_ROW_2),
        (*BOTH[:1], ["EV,001", "EV002"], *BOTH[2:]),
    ),
    "line break in quotes": (
        _file(A_HEADER, _with(A_ROW_1, 0, '"EV\n001"'), A_ROW_2),
        (*BOTH[:1], ["EV\n001", "EV002"], *BOTH[2:]),
    ),
    "NUL": (_file(A_HEADER, _with(A_ROW_1, 0, "EV\0"), A_ROW_2), f"{WHAT} holds a NUL character"),
    "one field too many first": (
        _file(A_HEADER, A_ROW_1 + ",9", A_ROW_2),
        f"{WHAT}: record 1 has 8 fields, the header 7",
    ),
    "one field too many later": (
        _file(A_HEADER, A_ROW_1, A_ROW_2 + ",9"),
        f"{WHAT}: record 2 has 8 fields, the header 7",
    ),
    "two fields too few": (
        _file(A_HEADER, A_ROW_1.rsplit(",", 2)[0], A_ROW_2),
        f"{WHAT}: record 1 has 5 fields, the header 7",
    ),
    "spaces only": (
        _file(A_HEADER, A_ROW_1, "  ", A_ROW_2),
        f"{WHAT}: record 2 has 1 fields, the header 7",
    ),
    "quote never closed": (
        _file(A_HEADER, A_ROW_1, _with(A_ROW_2, 6, '"2')),
        f"{WHAT} cannot be split into records: unexpected end of data",
    ),
    "text after a closing quote": (
        _file(A_HEADER, _with(A_ROW_1, 0, '"EV"1'), A_ROW_2),
        f"{WHAT} cannot be split into records: ',' expected after '\"'",
    ),
    "a column named twice": (
        _file(A_HEADER + ",evse_id", A_ROW_1 + ",x", A_ROW_2 + ",y"),
        f"{WHAT} names columns more than once: 1 repeated name(s)",
    ),
    "a column named twice with spaces": (
        _file(A_HEADER + ", evse_id ", A_ROW_1 + ",x", A_ROW_2 + ",y"),
        f"{WHAT} names columns more than once: 1 repeated name(s)",
    ),
    "line ends of a carriage return": (_file(A_HEADER, A_ROW_1, A_ROW_2, end="\r"), BOTH),
    "a field longer than the csv limit": (
        _file(A_HEADER, _with(A_ROW_1, 4, "B" * 200_000), A_ROW_2),
        f"{WHAT} cannot be split into records: field larger than field limit (131072)",
    ),
}


@pytest.mark.parametrize("name", MALFORMED)
def test_a_malformed_file_gives_a_layout_error_or_a_stated_removal(name):
    content, expected = MALFORMED[name]
    if isinstance(expected, str):
        with pytest.raises(LayoutError) as refused:
            load_asr.from_bytes(content)
        assert type(refused.value) is LayoutError
        assert str(refused.value) == expected
        return
    sessions, dropped = load_asr.from_bytes(content)
    got = (list(sessions["session_id"]), list(sessions["user_id"]), list(sessions["station"]))
    assert got == tuple(expected[:3])
    assert list(dropped["reason"]) == expected[3]
    assert set(dropped["reason"]) <= set(load_asr.REASONS)


def test_any_other_error_while_reading_is_a_layout_error_with_its_cause(monkeypatch):
    made_up = RuntimeError("made up")

    def fails(naive, zone):
        raise made_up

    monkeypatch.setattr(load_asr, "localize", fails)
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(_file(A_HEADER, A_ROW_1, A_ROW_2))
    assert str(refused.value) == "a.s.r. file cannot be read: RuntimeError"
    assert refused.value.__cause__ is None  # the error's own text may hold a value of a row


def test_through_the_gated_opener_in_a_made_up_repository(tmp_path):
    content = asr_csv(ROWS)
    root = sealed_repo(tmp_path, {ASR_FILE: content})
    sessions, dropped = heldout_loader.asr(root)
    direct, direct_dropped = load_asr.from_bytes(content)
    pd.testing.assert_frame_equal(sessions, direct)
    pd.testing.assert_frame_equal(dropped, direct_dropped)
    assert [(o["set"], o["file"]) for o in heldout_loader.logged_openings(root)] == [ASR_FILE]
