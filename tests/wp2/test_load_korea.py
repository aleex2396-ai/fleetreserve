"""The Korean loader on hand-written files in the layout of the data paper's Table 7.

One file of twenty-two made-up sessions: its expected session table and dropped rows are written
out below. It covers all 14 location types and so all five portfolios, every removal reason
(two fast-charger rows among them, one that would also fail two later rules), a `UserID` of 0,
a missing `ChargerID`, a `Duration` that disagrees with the times, sessions across midnight and
a session id repeated in another portfolio. Further tests, by reading of the module text: the
header found by name after trimming and folding, with either spelling of the start column and
never both (K1); the ";" separator with a decimal comma, a byte order mark, line ends of "\\r\\n"
and of "\\r" alone, quoted header names, a field of two million characters, the four missing
markers only, bytes that are not UTF-8, rows with more fields than the header (K2); every
location type in other letter case and with spaces, and refusals of other values, Korean-script
labels and codes, with counts and no value in the message (K3); the charger type's forms and
refusals (K4); the time form, seconds of 60 and above refused, the stated range with its first
and last times, sentinel dates and Seoul's daylight-saving years outside it, the one Asia/Seoul
offset over the range, the day and time columns where no datetime column is (K5), compared
with it where one is (K13), an hour of one digit in a time column (K14); `Duration` not read and
not needed (K6); the energy (K7); the users of `UserID` 0, of a negative number and of the id
markers, other ids as written with their outer spaces (K8); the station as written, its markers
refused, the session id as written, ids repeated across chargers counted (K9); the 7 kW cap with
its label and the 22 kW limit, the capacity columns unused (K10); a session across midnight
counted on both local days by the common rule (K11); the portfolio table as the owner decided it
(K12); the fast-charger count first among the dropped rows (K-log); malformed files, each ending
in a `LayoutError` with its message or in a stated result; any other error as a `LayoutError`
naming its type only, interruptions unchanged; and the route through the gated opener.
"""

import subprocess
import sys
from datetime import date

import pandas as pd
import pytest
from fake_heldout import sealed_repo
from fake_korea import KOREA_FILE, KOREA_HEADER, korea_csv, korea_row

from fleetreserve import heldout_loader, load_korea, windows
from fleetreserve.heldout_loader import SealRefused
from fleetreserve.sessions import COLUMNS, LayoutError


@pytest.fixture(autouse=True)
def _rows_not_units(monkeypatch):
    """The tests of this file are about the row rules, on files that hold one or a few of the
    named units. The check that each named unit keeps a row is switched off here; it is tested
    on its own in the `_units` test file of the loader, which reads a file of every unit with the
    check on."""
    monkeypatch.setattr(load_korea, "require_portfolios", lambda sessions, dropped: None)


R = korea_row
ROWS = [
    R("S1", "17", "C1", "Apartment", "2021-10-01 19:00:00", "2021-10-02 07:30:00", 9.5),
    R("S2", "0", "C2", "Hotel", "2021-10-02 20:15:00", "2021-10-03 09:00:00", 12.0),
    R("S3", "18", "C3", "Resort", "2021-11-05 14:00:00", "2021-11-05 18:00:00", 6.25,
      duration="999"),
    R("S4", "19", "C4", "Accommodation", "2021-12-24 21:00:00", "2021-12-25 10:00:00", 8.0),
    R("S5", "20", "C5", "Camping", "2022-07-30 18:30:00", "2022-07-31 08:45:00", 10.5),
    R("S6", "21", None, "Company", "2022-03-02 09:00:00", "2022-03-02 18:00:00", 15.0),
    R("S7", "22", "C7", "Public institution", "2022-03-03 08:30:00", "2022-03-03 12:00:00", 5.5),
    R("S8", "23", "C8", "Public area", "2022-04-04 10:00:00", "2022-04-04 13:00:00", 7.0),
    R("S9", "0", "C9", "Public parking lot", "2022-04-05 11:00:00", "2022-04-05 15:30:00", 9.0),
    R("S10", "24", "C10", "Bus garage", "2022-04-06 02:00:00", "2022-04-06 05:00:00", 4.0),
    R("S11", "25", "C11", "Market", "2022-05-07 10:00:00", "2022-05-07 11:30:00", 3.5),
    R("S12", "26", "C12", "Restaurant", "2022-05-08 12:00:00", "2022-05-08 14:00:00", 4.5),
    R("S13", "27", "C13", "Sightseeing", "2022-05-09 09:00:00", "2022-05-09 16:00:00", 11.0),
    R("S14", "28", "C14", "golf", "2022-09-30 06:00:00", "2022-09-30 13:00:00", 13.0),
    # Fast chargers: 30 kW on average, and one without plug-out and without energy; both are
    # counted under the fast-charger reason, the first rule.
    R("S15", "29", "F1", "Apartment", "2022-01-10 10:00:00", "2022-01-10 10:40:00", 20.0,
      charger_type="1"),
    R("S16", "30", "F2", "Company", "2022-01-11 09:00:00", None, 0.0, charger_type="1"),
    R("S17", "31", "C17", "Hotel", "2022-01-12 22:00:00", None, 5.0),
    R("S18", "32", "C18", "Market", "2022-01-13 10:00:00", "2022-01-13 10:00:00", 2.0),
    R("S19", "33", "C19", "Public area", "2022-01-14 10:00:00", "2022-01-14 12:00:00", 0.0),
    R("S20", "34", "C20", "Restaurant", "2022-01-15 10:00:00", "2022-01-15 12:00:00", None),
    # The session id S6 again, in another portfolio: the second is dropped.
    R("S6", "35", "C21", "Market", "2022-01-16 10:00:00", "2022-01-16 12:00:00", 3.0),
    R("S22", "36", "C22", "Sightseeing", "2022-02-01 10:00:00", "2022-02-01 11:00:00", 25.0),
]  # fmt: skip

# site, session_id, user_id, station, plug_in, plug_out, energy_kwh ("-": missing)
EXPECTED_TEXT = """
korea-home S1 17 C1 2021-10-01T19:00:00+09:00 2021-10-02T07:30:00+09:00 9.5
korea-lodging S2 - C2 2021-10-02T20:15:00+09:00 2021-10-03T09:00:00+09:00 12.0
korea-lodging S3 18 C3 2021-11-05T14:00:00+09:00 2021-11-05T18:00:00+09:00 6.25
korea-lodging S4 19 C4 2021-12-24T21:00:00+09:00 2021-12-25T10:00:00+09:00 8.0
korea-lodging S5 20 C5 2022-07-30T18:30:00+09:00 2022-07-31T08:45:00+09:00 10.5
korea-public-parking S8 23 C8 2022-04-04T10:00:00+09:00 2022-04-04T13:00:00+09:00 7.0
korea-public-parking S9 - C9 2022-04-05T11:00:00+09:00 2022-04-05T15:30:00+09:00 9.0
korea-public-parking S10 24 C10 2022-04-06T02:00:00+09:00 2022-04-06T05:00:00+09:00 4.0
korea-visits S11 25 C11 2022-05-07T10:00:00+09:00 2022-05-07T11:30:00+09:00 3.5
korea-visits S12 26 C12 2022-05-08T12:00:00+09:00 2022-05-08T14:00:00+09:00 4.5
korea-visits S13 27 C13 2022-05-09T09:00:00+09:00 2022-05-09T16:00:00+09:00 11.0
korea-visits S14 28 C14 2022-09-30T06:00:00+09:00 2022-09-30T13:00:00+09:00 13.0
korea-work S6 21 - 2022-03-02T09:00:00+09:00 2022-03-02T18:00:00+09:00 15.0
korea-work S7 22 C7 2022-03-03T08:30:00+09:00 2022-03-03T12:00:00+09:00 5.5
"""
EXPECTED = [
    tuple(None if f == "-" else float(f) if n == 6 else f for n, f in enumerate(line.split()))
    for line in EXPECTED_TEXT.strip().splitlines()
]

EXPECTED_DROPPED = [
    # site, session_id, reason, local_date
    ("korea-home", "S15", "fast charger (ChargerType 1)", date(2022, 1, 10)),
    ("korea-work", "S16", "fast charger (ChargerType 1)", date(2022, 1, 11)),
    ("korea-lodging", "S17", "no plug-out time recorded", date(2022, 1, 12)),
    ("korea-visits", "S18", "plug-out not after plug-in", date(2022, 1, 13)),
    ("korea-public-parking", "S19", "energy missing or not positive", date(2022, 1, 14)),
    ("korea-visits", "S20", "energy missing or not positive", date(2022, 1, 15)),
    ("korea-visits", "S6", "duplicate session id", date(2022, 1, 16)),
    ("korea-visits", "S22", "average above the physical limit", date(2022, 2, 1)),
]

# The owner's grouping of 15 November 2026, written out here a second time on purpose.
OWNER_GROUPING = {
    "korea-home": ["Apartment"],
    "korea-lodging": ["Hotel", "Resort", "Accommodation", "Camping"],
    "korea-work": ["Company", "Public institution"],
    "korea-public-parking": ["Public area", "Public parking lot", "Bus garage"],
    "korea-visits": ["Market", "Restaurant", "Sightseeing", "Golf"],
}
CHOSEN = ("ChargingsessionID", "UserID", "ChargerID", "Location", "ChargerType")
READ = (*CHOSEN, "StartDatetime", "EndDatetime", "Duration", "Demand")


def _rows(frame: pd.DataFrame) -> list[tuple]:
    return [
        (
            row.site,
            row.session_id,
            row.user_id,
            row.station,
            row.plug_in.isoformat(),
            row.plug_out.isoformat(),
            row.energy_kwh,
        )
        for row in frame.itertuples()
    ]


def _same(got, expected) -> None:
    pd.testing.assert_frame_equal(got[0], expected[0])
    pd.testing.assert_frame_equal(got[1], expected[1])


def _one(location="Apartment", **fields) -> dict:
    """One row; a datetime given in `fields` gives its side's day and time columns too, split as
    `korea_row` splits them, unless `fields` sets either of those (reading K13)."""
    row = R("K1", "5", "C1", location, "2022-06-01 08:00:00", "2022-06-01 17:00:00", 9.0)
    for side, (day, time) in load_korea.DAY_TIME.items():
        if side in fields and day not in fields and time not in fields:
            text = fields[side]
            row[day] = None if text is None else text.split(" ")[0]
            row[time] = None if text is None else text.split(" ")[-1]
    row.update(fields)
    return row


def test_the_hand_written_file_gives_the_written_out_table():
    sessions, dropped = load_korea.from_bytes(korea_csv(ROWS))
    assert list(sessions.columns) == list(COLUMNS)
    assert _rows(sessions) == EXPECTED
    assert set(sessions["role"]) == {"held-out"}
    assert set(sessions["nominal_kw"]) == {7.0}
    assert set(sessions["nominal_rule"]) == {"7 kW assumed: no kW in the data"}
    assert sessions["max_power_kw"].isna().all() and sessions["declared_energy_kwh"].isna().all()
    for name in ("charge_start", "charge_end", "declared_departure"):
        assert sessions[name].isna().all()
        assert str(sessions[name].dt.tz) == "Asia/Seoul"
    assert str(sessions["plug_in"].dt.tz) == "Asia/Seoul"
    assert list(dropped.columns) == ["site", "session_id", "reason", "local_date"]
    assert [tuple(row) for row in dropped.itertuples(index=False)] == EXPECTED_DROPPED


def test_removals_are_counted_by_reason_in_the_stated_order_the_fast_chargers_first():
    _, dropped = load_korea.from_bytes(korea_csv(ROWS))
    assert list(dropped["reason"].value_counts(sort=False).items()) == [
        ("fast charger (ChargerType 1)", 2),
        ("no plug-out time recorded", 1),
        ("plug-out not after plug-in", 1),
        ("energy missing or not positive", 2),
        ("duplicate session id", 1),
        ("average above the physical limit", 1),
    ]
    assert list(dropped["reason"].drop_duplicates()) == list(load_korea.REASONS)
    assert load_korea.REASONS[0] == "fast charger (ChargerType 1)"


def test_k_log_every_fast_charger_row_is_counted_whatever_else_it_holds():
    fast = [
        _one(**{"ChargingsessionID": f"F{n}", "ChargerType": "1", **fields})
        for n, fields in enumerate(
            [
                {},
                {"EndDatetime": None},
                {"Demand": 0.0},
                {"EndDatetime": "2022-06-01 08:00:00"},
                {"Demand": 500.0},
                {"ChargingsessionID": "K1"},
            ]
        )
    ]
    rows = [_one(), *fast]
    sessions, dropped = load_korea.from_bytes(korea_csv(rows))
    assert list(sessions["session_id"]) == ["K1"]
    assert list(dropped["reason"]) == ["fast charger (ChargerType 1)"] * 6
    assert list(dropped["session_id"]) == ["F0", "F1", "F2", "F3", "F4", "K1"]


def test_k12_the_portfolio_table_is_the_owners_grouping():
    assert load_korea.UNITS == (
        "korea-home",
        "korea-lodging",
        "korea-work",
        "korea-public-parking",
        "korea-visits",
    )
    grouped = {unit: [] for unit in load_korea.UNITS}
    for location, unit in load_korea.PORTFOLIO_OF.items():
        grouped[unit].append(location)
    assert grouped == OWNER_GROUPING
    assert len(load_korea.PORTFOLIO_OF) == 14


def test_k12_each_location_type_lands_in_its_portfolio_as_the_site():
    rows = [
        _one(location, ChargingsessionID=f"L{n}")
        for n, location in enumerate(load_korea.PORTFOLIO_OF)
    ]
    sessions, _ = load_korea.from_bytes(korea_csv(rows))
    got = dict(zip(sessions["session_id"], sessions["site"], strict=True))
    expected = {f"L{n}": unit for n, unit in enumerate(load_korea.PORTFOLIO_OF.values())}
    assert got == expected


# The 14 values of `Location` as the authors' scripts (`Code.zip`, figshare 22495141, read on 15
# November 2026; no data row seen) write them in their comments and their label table: lower
# case English words, the golf course written "golf".
SCRIPT_LOCATIONS = {
    "apartment": "korea-home",
    "hotel": "korea-lodging",
    "resort": "korea-lodging",
    "accommodation": "korea-lodging",
    "camping": "korea-lodging",
    "company": "korea-work",
    "public institution": "korea-work",
    "public area": "korea-public-parking",
    "public parking lot": "korea-public-parking",
    "bus garage": "korea-public-parking",
    "market": "korea-visits",
    "restaurant": "korea-visits",
    "sightseeing": "korea-visits",
    "golf": "korea-visits",
}


def test_k3_the_location_labels_of_the_authors_scripts_land_in_their_portfolios():
    rows = [_one(label, ChargingsessionID=f"L{n}") for n, label in enumerate(SCRIPT_LOCATIONS)]
    sessions, _ = load_korea.from_bytes(korea_csv(rows))
    got = dict(zip(sessions["session_id"], sessions["site"], strict=True))
    assert got == {f"L{n}": unit for n, unit in enumerate(SCRIPT_LOCATIONS.values())}


DAY_TIME = ("StartDay", "StartTime", "EndDay", "EndTime")
NO_DATETIME = tuple(n for n in KOREA_HEADER if n not in ("StartDatetime", "EndDatetime"))


@pytest.mark.parametrize(
    "dropped",
    [("StartDatetime", "EndDatetime"), ("StartDatetime",), ("EndDatetime",)],
)
def test_k5_without_a_datetime_column_its_time_is_the_day_and_time_columns_joined(dropped):
    # The authors' script `1. dataset upload.R` builds StartDatetime and EndDatetime from
    # StartDay with StartTime and EndDay with EndTime.
    header = tuple(n for n in KOREA_HEADER if n not in dropped)
    _same(
        load_korea.from_bytes(korea_csv(ROWS, header=header)),
        load_korea.from_bytes(korea_csv(ROWS)),
    )
    misspelled = tuple("SrartDatetime" if n == "StartDatetime" else n for n in header)
    _same(
        load_korea.from_bytes(korea_csv(ROWS, header=misspelled)),
        load_korea.from_bytes(korea_csv(ROWS)),
    )


def test_k5_the_joined_day_and_time_are_read_by_the_datetime_rules():
    def load(**fields):
        rows = [_one(), _one(ChargingsessionID="K2", **fields)]
        return load_korea.from_bytes(korea_csv(rows, header=NO_DATETIME))

    sessions, _ = load(StartTime="23:30:00", EndDay="2022-06-02", EndTime="06:15:59")
    assert [t.isoformat() for t in sessions["plug_in"]] == [
        "2022-06-01T08:00:00+09:00",
        "2022-06-01T23:30:00+09:00",
    ]
    assert sessions["plug_out"].iloc[1].isoformat() == "2022-06-02T06:15:59+09:00"
    for fields, message in (
        ({"StartTime": "07:00"}, "Korea StartDay and StartTime: 1 times cannot be read"),
        ({"StartTime": "7:0:00"}, "Korea StartDay and StartTime: 1 times cannot be read"),
        ({"StartTime": " 7:00:00"}, "Korea StartDay and StartTime: 1 times cannot be read"),
        ({"EndTime": "717:00:00"}, "Korea EndDay and EndTime: 1 times cannot be read"),
        ({"EndDay": "2022/06/01"}, "Korea EndDay and EndTime: 1 times cannot be read"),
        ({"EndTime": "17:00:60"}, "Korea EndDay and EndTime: 1 times cannot be read"),
        (
            {"StartDay": "1970-01-01"},
            "Korea StartDay and StartTime: 1 times outside 2021-09-01 00:00:00 to "
            "2022-10-31 23:59:59",
        ),
        ({"StartTime": None}, "Korea file: a row has no plug-in time or no session id"),
        ({"StartDay": None}, "Korea file: a row has no plug-in time or no session id"),
    ):
        with pytest.raises(LayoutError) as refused:
            load(**fields)
        assert str(refused.value) == message
    for fields in ({"EndTime": None}, {"EndDay": None}):
        sessions, dropped = load(**fields)
        assert list(sessions["session_id"]) == ["K1"]
        assert list(dropped["reason"]) == ["no plug-out time recorded"]


def test_k14_a_one_digit_hour_in_a_time_column_is_read_with_a_leading_zero():
    rows = [
        _one(),
        _one(ChargingsessionID="K2", StartTime="7:05:00", EndTime="9:59:59"),
        _one(ChargingsessionID="K3", StartTime="07:05:00", EndTime="09:59:59"),
    ]
    sessions, _ = load_korea.from_bytes(korea_csv(rows, header=NO_DATETIME))
    times = {
        row.session_id: (row.plug_in.isoformat(), row.plug_out.isoformat())
        for row in sessions.itertuples()
    }
    assert times == {
        "K1": ("2022-06-01T08:00:00+09:00", "2022-06-01T17:00:00+09:00"),
        "K2": ("2022-06-01T07:05:00+09:00", "2022-06-01T09:59:59+09:00"),
        "K3": ("2022-06-01T07:05:00+09:00", "2022-06-01T09:59:59+09:00"),
    }
    # with the datetime columns present, a one-digit hour of the time column agrees with them
    agree = [
        _one(StartTime="8:00:00"),
        _one(ChargingsessionID="K2", EndDatetime="2022-06-01 09:30:00", EndTime="9:30:00"),
    ]
    _same(
        load_korea.from_bytes(korea_csv(agree)),
        load_korea.from_bytes(korea_csv(agree, header=READ)),
    )
    # the datetime column keeps the paper's form: no one-digit hour there
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(korea_csv([_one(StartDatetime="2022-06-01 8:00:00")]))
    assert str(refused.value) == "Korea StartDatetime: 1 times cannot be read"


def test_k13_both_forms_of_a_side_are_compared_and_a_disagreement_refuses_the_set():
    # agreeing forms give the table of the datetime columns alone
    _same(
        load_korea.from_bytes(korea_csv(ROWS)),
        load_korea.from_bytes(korea_csv(ROWS, header=READ)),
    )
    later = "2022-06-01 08:00:01"
    cases = [
        # (rows, message): each count is the number of rows whose two forms differ
        (
            [_one(StartTime="09:00:00"), _one(ChargingsessionID="K2", StartDay="2022-06-02")],
            "Korea StartDatetime and StartDay and StartTime: 2 rows disagree",
        ),
        (
            [_one(), _one(ChargingsessionID="K2", StartDatetime=later, StartTime="08:00:00")],
            "Korea StartDatetime and StartDay and StartTime: 1 row disagrees",
        ),
        (
            [_one(EndTime="17:00:01"), _one(ChargingsessionID="K2"), _one(EndDay="2022-06-02")],
            "Korea EndDatetime and EndDay and EndTime: 2 rows disagree",
        ),
        # one form missing and the other present is a disagreement, on either side of it
        (
            [
                _one(EndTime=None),
                _one(ChargingsessionID="K2", EndDatetime=None, EndTime="17:00:00"),
            ],
            "Korea EndDatetime and EndDay and EndTime: 2 rows disagree",
        ),
        (
            [_one(StartDay=None)],
            "Korea StartDatetime and StartDay and StartTime: 1 row disagrees",
        ),
        # the time columns are read by the rules of a joined time first
        (
            [_one(StartTime="25:61")],
            "Korea StartDay and StartTime: 1 times cannot be read",
        ),
        (
            [_one(EndDay="2023-06-01")],
            "Korea EndDay and EndTime: 1 times outside 2021-09-01 00:00:00 to 2022-10-31 23:59:59",
        ),
    ]
    for rows, message in cases:
        with pytest.raises(LayoutError) as refused:
            load_korea.from_bytes(korea_csv(rows))
        assert str(refused.value) == message
    # both forms missing agree: the row is dropped as before
    sessions, dropped = load_korea.from_bytes(
        korea_csv([_one(), _one(ChargingsessionID="K2", EndDatetime=None, EndDay=None)])
    )
    assert (list(sessions["session_id"]), list(dropped["reason"])) == (
        ["K1"],
        ["no plug-out time recorded"],
    )


@pytest.mark.parametrize(
    "dropped, noise",
    [
        (("StartTime",), {"StartDay": "x"}),
        (("StartDay",), {"StartTime": "x"}),
        (("EndDay",), {"EndTime": "x"}),
        (("EndTime",), {"EndDay": "x"}),
        (DAY_TIME, {}),
    ],
)
def test_k13_a_side_without_both_its_day_and_time_columns_is_not_compared(dropped, noise):
    header = tuple(n for n in KOREA_HEADER if n not in dropped)
    rows = [{**row, **noise} for row in ROWS]
    _same(
        load_korea.from_bytes(korea_csv(rows, header=header)),
        load_korea.from_bytes(korea_csv(ROWS)),
    )


@pytest.mark.parametrize(
    "dropped",
    [
        ("StartDatetime", "StartDay"),
        ("StartDatetime", "StartTime"),
        ("EndDatetime", "EndDay"),
        ("EndDatetime", "EndTime"),
        ("StartDatetime", "EndDatetime", *DAY_TIME),
    ],
)
def test_k5_a_side_with_neither_its_datetime_nor_both_its_day_and_time_is_refused(dropped):
    header = tuple(n for n in KOREA_HEADER if n not in dropped)
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(korea_csv(ROWS, header=header))
    assert str(refused.value) == LACKS


def test_k1_the_paper_misspelling_of_the_start_column_gives_the_same_table():
    header = tuple("SrartDatetime" if n == "StartDatetime" else n for n in KOREA_HEADER)
    assert b"SrartDatetime" in korea_csv(ROWS, header=header)
    _same(
        load_korea.from_bytes(korea_csv(ROWS, header=header)),
        load_korea.from_bytes(korea_csv(ROWS)),
    )


def test_k1_both_spellings_of_the_start_column_are_refused():
    header = (*KOREA_HEADER, "SrartDatetime")
    with pytest.raises(LayoutError, match="^Korea file has both StartDatetime and SrartDatetime$"):
        load_korea.from_bytes(korea_csv(ROWS, header=header))


SPLIT = (
    "Korea file: a row cannot be split under the header (more fields than the header, or a "
    "quote never closed)"
)
LACKS = (
    "Korea file lacks columns: no separator gives ChargingsessionID, UserID, ChargerID, "
    "Location, ChargerType, Demand, a start (StartDatetime or SrartDatetime, or StartDay and "
    "StartTime) and an end (EndDatetime, or EndDay and EndTime)"
)


@pytest.mark.parametrize(
    "name",
    [
        "ChargingsessionID",
        "UserID",
        "ChargerID",
        "Location",
        "ChargerType",
        "EndDatetime",
        "Demand",
        "StartDatetime",
    ],
)
def test_k1_a_missing_column_read_is_refused_with_the_names_wanted(name):
    assert load_korea.REQUIRED == (
        "ChargingsessionID",
        "UserID",
        "ChargerID",
        "Location",
        "ChargerType",
        "Demand",
    )
    assert (load_korea.START, load_korea.END) == ("StartDatetime", "EndDatetime")
    # without the day and time columns, which can stand in for a missing datetime column (K5)
    header = tuple(column for column in KOREA_HEADER if column not in (name, *DAY_TIME))
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(korea_csv(ROWS, header=header))
    assert str(refused.value) == LACKS


def test_k1_columns_are_found_by_name_in_any_order_in_any_letter_case_and_with_spaces():
    plain = load_korea.from_bytes(korea_csv(ROWS))
    reordered = tuple(reversed(KOREA_HEADER))
    _same(load_korea.from_bytes(korea_csv(ROWS, header=reordered)), plain)
    for header in (
        tuple(name.upper() for name in KOREA_HEADER),
        tuple(name.lower() for name in KOREA_HEADER),
        tuple(f"  {name} " for name in KOREA_HEADER),
    ):
        _same(load_korea.from_bytes(korea_csv(ROWS, header=header)), plain)


def test_k1_header_names_equal_after_folding_are_refused():
    header = (*KOREA_HEADER, "demand")
    with pytest.raises(
        LayoutError, match="^Korea file: 1 header names repeat after trimming and case folding$"
    ):
        load_korea.from_bytes(korea_csv(ROWS, header=header))
    header = (*KOREA_HEADER, "chargerid", " ChargerID")
    with pytest.raises(
        LayoutError, match="^Korea file: 2 header names repeat after trimming and case folding$"
    ):
        load_korea.from_bytes(korea_csv(ROWS, header=header))


def test_k1_k5_k10_a_file_of_only_the_columns_read_gives_the_same_table():
    # The capacity, current and company columns are not used (K10); the day and time columns are
    # not needed where the datetime columns are present (K5).
    only = load_korea.from_bytes(korea_csv(ROWS, header=READ))
    _same(only, load_korea.from_bytes(korea_csv(ROWS)))


def test_k10_the_columns_not_read_may_hold_anything():
    # the day and time columns are compared with the datetime columns (K13), so not here
    noise = {"ChargerCompany": "x", "ChargerCapacity": "fast?", "ChargerACDC": "DC"}
    rows = [{**row, **noise} for row in ROWS]
    _same(load_korea.from_bytes(korea_csv(rows)), load_korea.from_bytes(korea_csv(ROWS)))


def test_k2_semicolon_and_decimal_comma_give_the_same_table():
    content = korea_csv(ROWS, sep=";")
    assert b";6,25\n" in content and b"Public parking lot" in content
    _same(load_korea.from_bytes(content), load_korea.from_bytes(korea_csv(ROWS)))


def test_k2_a_byte_order_mark_and_windows_line_ends_give_the_same_table():
    plain = korea_csv(ROWS)
    expected = load_korea.from_bytes(plain)
    crlf = plain.replace(b"\n", b"\r\n")
    for content in (b"\xef\xbb\xbf" + plain, crlf, b"\xef\xbb\xbf" + crlf):
        _same(load_korea.from_bytes(content), expected)


def test_k2_line_ends_of_a_carriage_return_alone_give_the_same_table():
    for sep in (",", ";"):
        plain = korea_csv(ROWS, sep=sep)
        expected = load_korea.from_bytes(plain)
        cr = plain.replace(b"\n", b"\r")
        assert b"\n" not in cr
        for content in (cr, b"\xef\xbb\xbf" + cr):
            _same(load_korea.from_bytes(content), expected)


def test_k2_quoted_header_names_under_either_separator_give_the_same_table():
    for sep in (",", ";"):
        plain = korea_csv(ROWS, sep=sep)
        first, rest = plain.split(b"\n", 1)
        quoted = sep.encode().join(b'"' + name + b'"' for name in first.split(sep.encode()))
        _same(load_korea.from_bytes(quoted + b"\n" + rest), load_korea.from_bytes(plain))


def test_k1_k2_header_names_are_read_as_written_line_ends_inside_quotes_included():
    # Two further names that differ only in a quoted line end are two names, not a repeat.
    plain = korea_csv(ROWS)
    first, rest = plain.split(b"\n", 1)
    for extra in (b',"a\rb","a\nb"', b',"a\r\nb","a\nb"'):
        rows = rest.replace(b"\n", b",1,2\n")
        _same(load_korea.from_bytes(first + extra + b"\n" + rows), load_korea.from_bytes(plain))


def test_k2_a_field_of_two_million_characters_is_read():
    sessions, _ = load_korea.from_bytes(korea_csv([_one(ChargerID="X" * 2_000_000)]))
    assert list(sessions["station"]) == ["X" * 2_000_000]


def test_k2_only_the_four_stated_markers_are_missing():
    markers = ["", "NA", "NaN", "nan"]
    rows = [_one(ChargingsessionID=f"M{n}", ChargerID=m) for n, m in enumerate(markers)]
    sessions, _ = load_korea.from_bytes(korea_csv(rows))
    assert list(sessions["station"]) == [None] * 4
    # Other forms are text: in the session id (which has no marker rule) they stay as written.
    others = ["N/A", "NULL", "None", "null", "-", " ", "NA ", " nan", "#N/A", "n.a."]
    rows = [
        _one(ChargingsessionID=m, StartDatetime=f"2022-06-01 0{n}:00:00")
        for n, m in enumerate(others)
    ]
    sessions, dropped = load_korea.from_bytes(korea_csv(rows))
    assert list(sessions["session_id"]) == others and dropped.empty


@pytest.mark.parametrize("codec", ["cp949", "euc-kr", "utf-16"])
def test_k2_another_encoding_is_refused(codec):
    content = korea_csv([_one()]).decode("utf-8").replace("Apartment", "아파트").encode(codec)
    with pytest.raises(LayoutError, match="^Korea file is not UTF-8 text$"):
        load_korea.from_bytes(content)


def test_k2_rows_with_more_fields_than_the_header_are_refused():
    plain = korea_csv([_one(), _one(ChargingsessionID="K2")])
    every = plain.replace(b"\n", b",extra\n").replace(b",extra\n", b"\n", 1)
    some = plain.rstrip(b"\n") + b",extra\n"
    with pytest.raises(LayoutError, match="^Korea file: rows hold more fields than the header$"):
        load_korea.from_bytes(every)
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(some)
    assert str(refused.value) == SPLIT


def test_k2_a_row_with_fewer_fields_has_its_last_fields_missing():
    content = korea_csv([_one()])
    short = content.rstrip(b"\n").rsplit(b",", 1)[0] + b"\n"
    assert short.count(b",") == content.count(b",") - 1
    sessions, dropped = load_korea.from_bytes(short)
    assert sessions.empty
    assert list(dropped["reason"]) == ["energy missing or not positive"]


def test_a_file_without_rows_gives_empty_tables():
    sessions, dropped = load_korea.from_bytes(korea_csv([]))
    assert sessions.empty and list(sessions.columns) == list(COLUMNS)
    assert dropped.empty and list(dropped.columns) == ["site", "session_id", "reason", "local_date"]


@pytest.mark.parametrize(
    "written",
    ["APARTMENT", "apartment", " Apartment ", " GOLF ", "PUBLIC PARKING LOT", "Bus Garage"],
)
def test_k3_location_types_match_after_trimming_and_folding(written):
    sessions, _ = load_korea.from_bytes(korea_csv([_one(written)]))
    folded = written.strip().casefold()
    expected = [u for name, u in load_korea.PORTFOLIO_OF.items() if name.casefold() == folded]
    assert list(sessions["site"]) == expected and len(expected) == 1


@pytest.mark.parametrize(
    "values, rows, distinct",
    [
        (["Office"], 1, 1),
        (["아파트", "아파트", "호텔"], 3, 2),
        (["1", "2", "14"], 3, 3),
        ([None, "Golfcc"], 2, 2),
        (["Golf CC", "golf course"], 2, 2),
        (["Public  area"], 1, 1),
        (["Public_institution", "public_institution"], 2, 1),
    ],
)
def test_k3_a_value_outside_the_location_types_is_refused_by_count_alone(values, rows, distinct):
    file = [_one(), *(_one(v, ChargingsessionID=f"X{n}") for n, v in enumerate(values))]
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(korea_csv(file))
    assert str(refused.value) == (
        f"Korea Location: {rows} rows hold no location type of the paper ({distinct} distinct "
        "values)"
    )
    for value in values:
        assert value is None or value not in str(refused.value)


@pytest.mark.parametrize(
    "written, fast",
    [("0", False), ("1", True), ("0.0", False), ("1.0", True), (" 1 ", True), ("00", False)],
)
def test_k4_the_charger_type_as_text_or_number(written, fast):
    sessions, dropped = load_korea.from_bytes(korea_csv([_one(ChargerType=written)]))
    assert len(sessions) == (0 if fast else 1)
    assert list(dropped["reason"]) == (["fast charger (ChargerType 1)"] if fast else [])


def test_k4_the_charger_type_under_semicolon_with_a_decimal_comma():
    content = korea_csv(
        [_one(ChargerType=1.0), _one(ChargingsessionID="K2", ChargerType=0.0)], sep=";"
    )
    assert b";1,0;" in content
    sessions, dropped = load_korea.from_bytes(content)
    assert list(sessions["session_id"]) == ["K2"] and list(dropped["session_id"]) == ["K1"]


@pytest.mark.parametrize("written", ["2", "-1", "0.5", "fast", "slow", "", "True", '"1,0"', "inf"])
def test_k4_another_charger_type_is_refused(written):
    rows = [_one(), _one(ChargingsessionID="K2", ChargerType=written)]
    with pytest.raises(LayoutError, match="^Korea ChargerType: 1 values are neither 0 nor 1$"):
        load_korea.from_bytes(korea_csv(rows))


@pytest.mark.parametrize(
    "text",
    [
        "2022-6-01 17:00:00",
        "2022-06-1 17:00:00",
        "2022-06-01 7:00:00",
        "2022-06-01 17:0:00",
        "2022-06-01 17:00:0",
        "2022-06-01T17:00:00",
        "2022-06-01 17:00:00+09:00",
        "2022-06-01 17:00:00Z",
        "2022-06-01 17:00:00.000",
        "2022-06-01 17:00",
        " 2022-06-01 17:00:00",
        "2022-06-01 17:00:00 ",
        "2022/06/01 17:00:00",
        "01-06-2022 17:00:00",
        "2022-02-30 17:00:00",
        "２０２２-06-01 17:00:00",
    ],
)
def test_k5_an_unreadable_time_is_refused(text):
    rows = [_one(), _one(ChargingsessionID="K2", EndDatetime=text)]
    with pytest.raises(LayoutError, match="^Korea EndDatetime: 1 times cannot be read$"):
        load_korea.from_bytes(korea_csv(rows))
    rows = [_one(), _one(ChargingsessionID="K2", StartDatetime=text)]
    with pytest.raises(LayoutError, match="^Korea StartDatetime: 1 times cannot be read$"):
        load_korea.from_bytes(korea_csv(rows))


def test_k5_the_times_are_seoul_wall_clock_at_utc_plus_nine():
    sessions, _ = load_korea.from_bytes(korea_csv([_one()]))
    plug_in = sessions["plug_in"].iloc[0]
    assert plug_in.isoformat() == "2022-06-01T08:00:00+09:00"
    assert plug_in.tz_convert("UTC").isoformat() == "2022-05-31T23:00:00+00:00"


@pytest.mark.parametrize("name", ["StartDatetime", "EndDatetime"])
@pytest.mark.parametrize(
    "text",
    [
        "0000-01-01 10:00:00",
        "0001-01-01 00:00:00",
        "1900-01-01 00:00:00",
        "1970-01-01 09:00:00",
        "1987-05-10 02:30:00",
        "1988-05-08 02:30:00",
        "1988-10-09 02:30:00",
        "2021-08-31 23:59:59",
        "2022-11-01 00:00:00",
        "2099-12-31 23:59:59",
        "9999-12-31 23:59:59",
    ],
)
def test_k5_a_time_outside_the_stated_range_is_refused_with_a_count(name, text):
    # Sentinel dates, Seoul's daylight-saving changes of 1987 and 1988 and year 0 among them.
    rows = [_one(), _one(ChargingsessionID="K2", **{name: text}), _one(ChargingsessionID="K3")]
    if name == "StartDatetime":
        rows[2][name] = text
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(korea_csv(rows))
    count = 2 if name == "StartDatetime" else 1
    assert str(refused.value) == (
        f"Korea {name}: {count} times outside 2021-09-01 00:00:00 to 2022-10-31 23:59:59"
    )
    assert refused.value.__cause__ is None


def test_k5_the_first_and_last_times_of_the_range_are_read():
    assert (load_korea.FIRST_TIME, load_korea.LAST_TIME) == (
        "2021-09-01 00:00:00",
        "2022-10-31 23:59:59",
    )
    rows = [
        _one(StartDatetime="2021-09-01 00:00:00", EndDatetime="2021-09-01 10:00:00"),
        _one(ChargingsessionID="K2", StartDatetime="2022-10-31 14:00:00",
             EndDatetime="2022-10-31 23:59:59"),
    ]  # fmt: skip
    sessions, dropped = load_korea.from_bytes(korea_csv(rows))
    assert dropped.empty
    assert [t.isoformat() for t in sessions["plug_in"]] == [
        "2021-09-01T00:00:00+09:00",
        "2022-10-31T14:00:00+09:00",
    ]
    assert sessions["plug_out"].iloc[1].isoformat() == "2022-10-31T23:59:59+09:00"


def test_k5_asia_seoul_has_one_offset_over_the_whole_range():
    # Why no time of the range can fall in a daylight-saving gap or doubled hour.
    hours = pd.date_range(load_korea.FIRST_TIME, load_korea.LAST_TIME, freq="h")
    local = hours.tz_localize(load_korea.ZONE, nonexistent="NaT", ambiguous="NaT")
    assert not local.isna().any()
    assert {t.utcoffset() for t in local} == {pd.Timedelta(hours=9)}


@pytest.mark.parametrize("name", ["StartDatetime", "EndDatetime"])
@pytest.mark.parametrize("seconds", ["60", "61", "99"])
def test_k5_a_seconds_field_above_59_is_refused(name, seconds):
    text = f"2022-06-01 10:00:{seconds}"
    rows = [_one(), _one(ChargingsessionID="K2", **{name: text})]
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(korea_csv(rows))
    assert str(refused.value) == f"Korea {name}: 1 times cannot be read"


def test_k5_a_missing_plug_in_time_or_session_id_is_refused():
    for field in ("StartDatetime", "ChargingsessionID"):
        rows = [_one(), _one(**{"ChargingsessionID": "K2", field: None})]
        with pytest.raises(
            LayoutError, match="^Korea file: a row has no plug-in time or no session id$"
        ):
            load_korea.from_bytes(korea_csv(rows))
    header = tuple("SrartDatetime" if n == "StartDatetime" else n for n in KOREA_HEADER)
    with pytest.raises(LayoutError, match="no plug-in time"):
        load_korea.from_bytes(korea_csv([_one(StartDatetime=None)], header=header))


def test_k6_duration_is_not_read_whatever_it_holds():
    plain = load_korea.from_bytes(korea_csv(ROWS))
    for duration in (None, "0", "-5", "12345.5", "1e3", "01:30:00", "90 min", '"1,5"', "x"):
        rows = [{**row, "Duration": duration} for row in ROWS]
        _same(load_korea.from_bytes(korea_csv(rows)), plain)


def test_k6_a_file_without_duration_gives_the_same_table():
    header = tuple(name for name in KOREA_HEADER if name != "Duration")
    assert "Duration" not in load_korea.REQUIRED
    _same(
        load_korea.from_bytes(korea_csv(ROWS, header=header)),
        load_korea.from_bytes(korea_csv(ROWS)),
    )


@pytest.mark.parametrize("written", ["12 kWh", '"1,5"', "x"])
def test_k7_an_energy_that_is_not_a_number_is_refused(written):
    rows = [_one(), _one(ChargingsessionID="K2", Demand=written)]
    with pytest.raises(LayoutError, match="^Korea Demand: 1 values are not numbers$"):
        load_korea.from_bytes(korea_csv(rows))


def test_k7_demand_is_the_energy_in_kwh_with_either_decimal_mark():
    rows = [_one(Demand=12.75)]
    for sep in (",", ";"):
        sessions, _ = load_korea.from_bytes(korea_csv(rows, sep=sep))
        assert list(sessions["energy_kwh"]) == [12.75]


K8_USERS = [
    # zero: no user
    ("0", None),
    ("0.0", None),
    ("00", None),
    (" 0 ", None),
    ("-0", None),
    # a negative number: no user
    ("-1", None),
    ("-2.5", None),
    (" -1 ", None),
    ("-inf", None),
    # a marker of a missing id, after trimming and case folding: no user
    ("NULL", None),
    ("null", None),
    (" Null ", None),
    ("N/A", None),
    ("n/a", None),
    ("None", None),
    ("-", None),
    (" - ", None),
    ("NA ", None),
    (" nan", None),
    (" ", None),
    ("   ", None),
    # the stated missing markers of K2: no user
    (None, None),
    ("NA", None),
    # anything else: the user as written, outer spaces kept
    ("17", "17"),
    ("017", "017"),
    ("0x0", "0x0"),
    ("U0", "U0"),
    ("1e3", "1e3"),
    ("0.5", "0.5"),
    ("1e-9", "1e-9"),
    ("0.0001", "0.0001"),
    ("+1", "+1"),
    ("inf", "inf"),
    (" 17", " 17"),
    ("17 ", "17 "),
    ("--", "--"),
    ("N/A/1", "N/A/1"),
]


def test_k8_a_user_id_of_zero_a_negative_number_or_a_missing_marker_is_no_user():
    rows = [_one(ChargingsessionID=f"U{n}", UserID=u) for n, (u, _) in enumerate(K8_USERS)]
    sessions, _ = load_korea.from_bytes(korea_csv(rows))
    users = dict(zip(sessions["session_id"], sessions["user_id"], strict=True))
    assert [(u, users[f"U{n}"]) for n, (u, _) in enumerate(K8_USERS)] == K8_USERS


def test_k8_a_user_id_of_zero_with_a_decimal_comma_is_no_user():
    content = korea_csv(
        [
            _one(UserID="0,0"),
            _one(ChargingsessionID="K2", UserID="2,5"),
            _one(ChargingsessionID="K3", UserID="-1,5"),
        ],
        sep=";",
    )
    sessions, _ = load_korea.from_bytes(content)
    assert list(sessions["user_id"]) == [None, "2,5", None]


@pytest.mark.parametrize(
    "written",
    ["NULL", "null", " Null ", "N/A", "n/a", "None", "-", " - ", "NA ", " nan", " ", "-1", "-0.5"],
)
def test_k9_a_charger_id_written_as_a_missing_marker_or_a_negative_number_is_refused(written):
    rows = [_one(), _one(ChargingsessionID="K2", ChargerID=written), _one(ChargingsessionID="K3")]
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(korea_csv(rows))
    assert str(refused.value) == (
        "Korea ChargerID: 1 values are written as a missing marker or a negative number"
    )
    assert written.strip() == "" or written.strip() not in str(refused.value)


def test_k9_charger_ids_that_are_not_markers_stay_as_written():
    written = ["0", "007", " C1", "C1 ", "--", "+1", "N/A/1"]
    rows = [_one(ChargingsessionID=f"C{n}", ChargerID=c) for n, c in enumerate(written)]
    sessions, _ = load_korea.from_bytes(korea_csv(rows))
    assert list(sessions["station"]) == written


def test_k9_ids_in_korean_script_or_with_other_letters_stay_as_written():
    rows = [
        _one(ChargingsessionID="세션-1", UserID="사용자", ChargerID="충전기 7"),
        _one(ChargingsessionID="é2", UserID="ü", ChargerID="Ω",
             StartDatetime="2022-06-01 09:00:00"),
    ]  # fmt: skip
    for content in (korea_csv(rows), b"\xef\xbb\xbf" + korea_csv(rows, sep=";")):
        sessions, dropped = load_korea.from_bytes(content)
        assert dropped.empty
        assert list(sessions["session_id"]) == ["세션-1", "é2"]
        assert list(sessions["user_id"]) == ["사용자", "ü"]
        assert list(sessions["station"]) == ["충전기 7", "Ω"]


def test_k9_ids_with_outer_spaces_stay_as_written_and_differ_from_the_bare_id():
    rows = [
        _one(ChargingsessionID="A1", UserID="17", ChargerID="C1"),
        _one(ChargingsessionID=" A1", UserID=" 17", ChargerID=" C1",
             StartDatetime="2022-06-01 09:00:00"),
        _one(ChargingsessionID="A1 ", UserID="17 ", ChargerID="C1 ",
             StartDatetime="2022-06-01 10:00:00"),
    ]  # fmt: skip
    sessions, dropped = load_korea.from_bytes(korea_csv(rows))
    assert dropped.empty
    assert list(sessions["session_id"]) == ["A1", " A1", "A1 "]
    assert list(sessions["user_id"]) == ["17", " 17", "17 "]
    assert list(sessions["station"]) == ["C1", " C1", "C1 "]


def test_k9_station_and_session_id_as_written():
    rows = [
        _one(ChargingsessionID="0042", ChargerID="007"),
        _one(ChargingsessionID="42", ChargerID=None, StartDatetime="2022-06-01 09:00:00"),
    ]
    sessions, _ = load_korea.from_bytes(korea_csv(rows))
    assert list(sessions["session_id"]) == ["0042", "42"]
    assert list(sessions["station"]) == ["007", None]


def test_k9_a_repeated_session_id_is_dropped_in_any_portfolio_the_first_kept():
    rows = [
        _one("Market"),
        _one("Apartment", StartDatetime="2022-06-01 07:00:00"),
        _one("Market", ChargingsessionID="K2"),
    ]
    sessions, dropped = load_korea.from_bytes(korea_csv(rows))
    assert list(zip(sessions["site"], sessions["session_id"], strict=True)) == [
        ("korea-visits", "K1"),
        ("korea-visits", "K2"),
    ]
    assert [tuple(r) for r in dropped.itertuples(index=False)] == [
        ("korea-home", "K1", "duplicate session id", date(2022, 6, 1))
    ]


def test_k9_an_id_held_only_by_a_removed_row_is_kept():
    rows = [
        _one(ChargerType="1"),
        _one(StartDatetime="2022-06-01 09:00:00", EndDatetime=None),
        _one(StartDatetime="2022-06-01 10:00:00"),
        _one(StartDatetime="2022-06-01 11:00:00"),
    ]
    sessions, dropped = load_korea.from_bytes(korea_csv(rows))
    assert [t.isoformat() for t in sessions["plug_in"]] == ["2022-06-01T10:00:00+09:00"]
    assert list(dropped["reason"]) == [
        "fast charger (ChargerType 1)",
        "no plug-out time recorded",
        "duplicate session id",
    ]


def test_k9_ids_unique_only_per_charger_are_counted_in_the_dropped_rows():
    # Not covered (module text): if the file's ids repeat across chargers, every repeat is
    # removed, and the count under the duplicate reason in the dropped rows shows it.
    rows = [
        _one(ChargingsessionID=str(n), ChargerID=charger, StartDatetime=f"2022-06-0{n} 08:00:00",
             EndDatetime=f"2022-06-0{n} 17:00:00")
        for charger in ("C1", "C2", "C3")
        for n in (1, 2, 3, 4)
    ]  # fmt: skip
    sessions, dropped = load_korea.from_bytes(korea_csv(rows))
    assert list(sessions["session_id"]) == ["1", "2", "3", "4"]
    assert set(sessions["station"]) == {"C1"}
    assert dropped["reason"].value_counts().to_dict() == {"duplicate session id": 8}


def test_k10_an_average_of_exactly_the_limit_is_kept_and_just_above_dropped():
    assert load_korea.LIMIT_KW == 22.0 and load_korea.NOMINAL_KW == 7.0
    end = "2022-06-01 09:00:00"
    rows = [
        _one(EndDatetime=end, Demand=22.0),
        _one(ChargingsessionID="K2", EndDatetime=end, Demand=22.001),
    ]
    sessions, dropped = load_korea.from_bytes(korea_csv(rows))
    assert list(sessions["session_id"]) == ["K1"]
    assert list(zip(dropped["session_id"], dropped["reason"], strict=True)) == [
        ("K2", "average above the physical limit")
    ]


def test_k11_a_session_across_midnight_is_one_row_counted_on_both_local_days():
    rows = [
        _one(StartDatetime="2022-06-01 21:00:00", EndDatetime="2022-06-02 07:30:00"),
        _one(ChargingsessionID="K2", StartDatetime="2022-06-01 12:00:00",
             EndDatetime="2022-06-01 18:00:00"),
    ]  # fmt: skip
    sessions, dropped = load_korea.from_bytes(korea_csv(rows))
    assert dropped.empty and len(sessions) == 2
    night = sessions.set_index("session_id").loc["K1"]
    assert night["plug_in"].date() == date(2022, 6, 1)
    assert night["plug_out"].date() == date(2022, 6, 2)
    assert night["plug_out"] - night["plug_in"] == pd.Timedelta(hours=10, minutes=30)
    days = [date(2022, 6, 1), date(2022, 6, 2)]
    for hour, counts in ((23, [1, 0]), (6, [0, 1]), (8, [0, 0]), (13, [1, 0])):
        instants = windows.local_instants(days, hour, load_korea.ZONE)
        got = windows.plugged_in_counts(sessions["plug_in"], sessions["plug_out"], instants)
        assert list(got) == counts


def test_k11_a_dropped_session_across_midnight_is_dated_by_its_plug_in():
    rows = [
        _one(StartDatetime="2022-06-01 23:30:00", EndDatetime="2022-06-02 00:00:01", Demand=30.0)
    ]
    _, dropped = load_korea.from_bytes(korea_csv(rows))
    assert [tuple(r) for r in dropped.itertuples(index=False)] == [
        ("korea-home", "K1", "average above the physical limit", date(2022, 6, 1))
    ]


def test_through_the_gated_opener_in_a_made_up_repository(tmp_path):
    content = korea_csv(ROWS)
    root = sealed_repo(tmp_path, {KOREA_FILE: content})
    assert heldout_loader.KOREA == KOREA_FILE == ("korea", "ChargingRecords.csv")
    got = heldout_loader.korea(root)
    _same(got, load_korea.from_bytes(content))
    assert [(o["set"], o["file"]) for o in heldout_loader.logged_openings(root)] == [KOREA_FILE]


def test_a_refused_opening_calls_no_parser_and_logs_nothing(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {KOREA_FILE: korea_csv(ROWS)}, tag=False)
    calls = []
    monkeypatch.setattr(load_korea, "from_bytes", lambda content: calls.append(content))
    result = None
    with pytest.raises(SealRefused, match="no tag frozen-plan"):
        result = heldout_loader.korea(root)
    assert result is None and calls == []
    assert heldout_loader.logged_openings(root) == []


def test_k10_the_average_rule_divides_the_energy_by_the_hours_plugged_in():
    rows = [
        _one(ChargingsessionID="LONG", EndDatetime="2022-06-01 20:00:00", Demand=40.0),
        _one(ChargingsessionID="SHORT", EndDatetime="2022-06-01 08:15:00", Demand=10.0),
    ]
    sessions, dropped = load_korea.from_bytes(korea_csv(rows))
    assert list(sessions["session_id"]) == ["LONG"]
    assert list(zip(dropped["session_id"], dropped["reason"], strict=True)) == [
        ("SHORT", "average above the physical limit")
    ]


def test_k7_a_small_positive_energy_is_kept_and_a_negative_one_dropped():
    rows = [
        _one(ChargingsessionID="SMALL", Demand=0.5),
        _one(ChargingsessionID="TINY", Demand=0.001),
        _one(ChargingsessionID="NEGATIVE", Demand=-1.0),
    ]
    sessions, dropped = load_korea.from_bytes(korea_csv(rows))
    assert list(sessions["session_id"]) == ["SMALL", "TINY"]
    assert list(sessions["energy_kwh"]) == [0.5, 0.001]
    assert list(zip(dropped["session_id"], dropped["reason"], strict=True)) == [
        ("NEGATIVE", "energy missing or not positive")
    ]


_PLAIN = korea_csv([_one()])
_HEADER, _ROW = _PLAIN.decode("utf-8").splitlines()


def _file(*lines: str) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


def _cells(**fields) -> str:
    return korea_csv([_one(**fields)]).decode("utf-8").splitlines()[1]


CANNOT_SPLIT_HEADER = "Korea file: the header line cannot be split: "
# Malformed files: each must end in a LayoutError with exactly this message, or in the stated
# result (kept session ids, dropped reasons). Any other error fails the test.
MALFORMED = {
    "empty": (b"", LACKS),
    "only a byte order mark": (b"\xef\xbb\xbf", LACKS),
    "only a newline": (b"\n", LACKS),
    "only separators": (_file("," * 15), LACKS),
    "a blank first line": (b"\n" + _PLAIN, LACKS),
    "a header quote never closed": (b'"' + _PLAIN, CANNOT_SPLIT_HEADER + "unexpected end of data"),
    "a header name with text after its closing quote": (
        _file('"ChargingsessionID"x' + _HEADER[len("ChargingsessionID") :], _ROW),
        CANNOT_SPLIT_HEADER + "',' expected after '\"'",
    ),
    "a header field over the csv field limit": (
        _file(_HEADER + "," + "x" * 200_000, _ROW),
        CANNOT_SPLIT_HEADER + "field larger than field limit (131072)",
    ),
    "a NUL in a header name": (_file(_HEADER + "\x00", _ROW), LACKS),
    "a header with a name repeated": (
        _file(_HEADER + ",DEMAND", _ROW + ",1"),
        "Korea file: 1 header names repeat after trimming and case folding",
    ),
    # Several faults at once: the first check in the stated order names the file.
    "both start names and a name repeated": (
        _file(_HEADER + ",SrartDatetime,demand", _ROW + ",x,1"),
        "Korea file: 1 header names repeat after trimming and case folding",
    ),
    "both start names and a row that cannot be split": (
        _file(_HEADER + ",SrartDatetime", _ROW + ",x", _ROW + ",x,y,z"),
        "Korea file has both StartDatetime and SrartDatetime",
    ),
    "no plug-in time in one row and an unreadable one in another": (
        _file(
            _HEADER,
            _cells(StartDatetime=None),
            _cells(ChargingsessionID="K2", StartDatetime="2022-06-01 7:00:00"),
        ),
        "Korea file: a row has no plug-in time or no session id",
    ),
    "an unreadable plug-in time and an unreadable plug-out time": (
        _file(_HEADER, _cells(StartDatetime="x", EndDatetime="y")),
        "Korea StartDatetime: 1 times cannot be read",
    ),
    "an unreadable plug-out time and an energy that is not a number": (
        _file(_HEADER, _cells(EndDatetime="y", Demand="z")),
        "Korea EndDatetime: 1 times cannot be read",
    ),
    "an energy that is not a number and a charger type outside": (
        _file(_HEADER, _cells(Demand="z", ChargerType="2")),
        "Korea Demand: 1 values are not numbers",
    ),
    "a charger type outside and a location outside": (
        _file(_HEADER, _cells(ChargerType="2", Location="Office")),
        "Korea ChargerType: 1 values are neither 0 nor 1",
    ),
    "a location outside and a charger id marker": (
        _file(_HEADER, _cells(Location="Office", ChargerID="NULL")),
        "Korea Location: 1 rows hold no location type of the paper (1 distinct values)",
    ),
    "a time outside the range and an unreadable plug-out time": (
        _file(_HEADER, _cells(StartDatetime="1970-01-01 00:00:00", EndDatetime="y")),
        "Korea StartDatetime: 1 times outside 2021-09-01 00:00:00 to 2022-10-31 23:59:59",
    ),
    "a data quote never closed": (_file(_HEADER, _ROW, '"K2' + _ROW[2:]), SPLIT),
    "a row with two more fields": (
        _file(_HEADER, _ROW, _cells(ChargingsessionID="K2") + ",a,b"),
        SPLIT,
    ),
    "every row with two more fields": (
        _file(_HEADER, _ROW + ",a,b"),
        "Korea file: rows hold more fields than the header",
    ),
    "a row of separators only": (
        _file(_HEADER, _ROW, "," * 15),
        "Korea file: a row has no plug-in time or no session id",
    ),
    "the header again as a row": (
        _file(_HEADER, _ROW, _HEADER),
        "Korea StartDatetime: 1 times cannot be read",
    ),
    "a NUL in Demand": (
        _file(_HEADER, _cells(Demand="9\x00")),
        "Korea Demand: 1 values are not numbers",
    ),
    "Location of spaces only": (
        _file(_HEADER, _cells(Location="   ")),
        "Korea Location: 1 rows hold no location type of the paper (1 distinct values)",
    ),
    "a time of year 0": (
        _file(_HEADER, _cells(EndDatetime="0000-01-01 00:00:00")),
        "Korea EndDatetime: 1 times outside 2021-09-01 00:00:00 to 2022-10-31 23:59:59",
    ),
    "an hour of 24": (
        _file(_HEADER, _cells(StartDatetime="2022-06-01 24:00:00")),
        "Korea StartDatetime: 1 times cannot be read",
    ),
    "a byte that is not UTF-8": (_PLAIN.replace(b"C1", b"C\xff"), "Korea file is not UTF-8 text"),
    "UTF-16 with its mark": (
        _PLAIN.decode("utf-8").encode("utf-16"),
        "Korea file is not UTF-8 text",
    ),
    # Stated results, not refusals:
    "Demand inf": (
        _file(_HEADER, _cells(Demand="inf")),
        ([], ["average above the physical limit"]),
    ),
    "Demand 1e400": (
        _file(_HEADER, _cells(Demand="1e400")),
        ([], ["average above the physical limit"]),
    ),
    "Demand -inf": (
        _file(_HEADER, _cells(Demand="-inf")),
        ([], ["energy missing or not positive"]),
    ),
    "a blank line between rows": (
        _file(_HEADER, _ROW, "", _cells(ChargingsessionID="K2")),
        (["K1", "K2"], []),
    ),
    "line ends mixed": (
        (_HEADER + "\r\n" + _ROW + "\n" + _cells(ChargingsessionID="K2") + "\r").encode("utf-8"),
        (["K1", "K2"], []),
    ),
    # its missing EndDatetime disagrees with the EndDay and EndTime it still holds (K13)
    "a row with fewer fields": (
        _file(_HEADER, _ROW.rsplit(",", 3)[0]),
        "Korea EndDatetime and EndDay and EndTime: 1 row disagrees",
    ),
    "a quoted newline in an id": (
        _file(_HEADER, _ROW.replace(",C1,", ',"C\n1",')),
        (["K1"], []),
    ),
}


@pytest.mark.parametrize("case", MALFORMED)
def test_a_malformed_file_ends_in_a_layout_error_or_a_stated_result(case):
    content, expected = MALFORMED[case]
    if isinstance(expected, str):
        with pytest.raises(LayoutError) as refused:
            load_korea.from_bytes(content)
        assert str(refused.value) == expected
        return
    sessions, dropped = load_korea.from_bytes(content)
    assert (list(sessions["session_id"]), list(dropped["reason"])) == expected


@pytest.mark.parametrize("error", [ValueError("S1 17"), KeyError("S1"), OverflowError("17")])
def test_any_other_error_while_reading_is_a_layout_error_that_names_its_type_only(
    monkeypatch, error
):
    def fails(raw):
        raise error

    monkeypatch.setattr(load_korea, "to_sessions", fails)
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(_PLAIN)
    assert str(refused.value) == f"Korea file cannot be read: {type(error).__name__}"
    assert refused.value.__cause__ is None and refused.value.__suppress_context__


@pytest.mark.parametrize("error", [LayoutError("Korea x"), KeyboardInterrupt(), SystemExit(3)])
def test_a_layout_error_or_an_interruption_while_reading_passes_unchanged(monkeypatch, error):
    def fails(raw):
        raise error

    monkeypatch.setattr(load_korea, "to_sessions", fails)
    with pytest.raises(type(error)) as raised:
        load_korea.from_bytes(_PLAIN)
    assert raised.value is error


def test_the_text_columns_of_the_session_table_are_plain_objects():
    sessions, _ = load_korea.from_bytes(korea_csv(ROWS))
    for name in ("site", "role", "session_id", "user_id", "station", "nominal_rule"):
        assert sessions[name].dtype == object, name


def test_loading_the_opener_module_loads_the_korean_parser_before_any_opening():
    # The opener checks the package's loaded modules against the tag (its reading H5), so the
    # parser must be loaded when the opener's module is, not later.
    code = (
        "import sys, fleetreserve.heldout_loader; print('fleetreserve.load_korea' in sys.modules)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "True"
