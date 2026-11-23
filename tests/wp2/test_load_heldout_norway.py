"""The held-out Norwegian sites OSL_T and TRO_R on made-up files in the published split layout.

The same made-up rows, written once with the held-out site codes and once with two development
site codes, must give the same session table and the same dropped rows, up to the site codes and
the role. A training or development site in the held-out file, a missing column, an unreadable
time, an unreadable connection time, a row without a plug-in time and a row without a session id
are refused. A time with one-digit fields, and an empty user, energy or connection time, are read
as `load_norway` reads them (a missing user is None in both). Read
through the gated opener in a made-up repository, the file gives the same tables and one logged
opening.

Text fields are kept as written: a user or session id of digits keeps its leading zeros, and a
user written "null" or "None" is a user; only an empty field and NA are missing, so an energy
"NaN", a connection time "nan" and a plug-out "N/A" or "null" are refused. The count in a
refusal is the number of rows refused. A list of malformed files (a missing value in each column
in turn, impossible times, an empty file, blank lines only, a header only, bytes that are not
UTF-8, a quoted field holding the separator or a line break, a NUL character, one field too many
in the first or a later record, two fields too few, a line of spaces, a quote never closed, text
after a closing quote, a column named twice, carriage returns as line ends) each give exactly
`LayoutError` with its stated message, or a table with stated removals only. Any other error
while reading is a `LayoutError` with that error as its cause; a field over the csv module's
limit is a `LayoutError`.
"""

import pandas as pd
import pytest
from fake_heldout import NORWAY_FILE, sealed_repo
from fake_raw import NORWAY_HEADER, norway_row, raw_root, write_norway

from fleetreserve import heldout_loader, load_heldout_norway, load_norway
from fleetreserve.sessions import LayoutError


@pytest.fixture(autouse=True)
def _rows_not_units(monkeypatch):
    """The tests of this file are about the row rules, on files that hold one or a few of the
    named units. The check that each named unit keeps a row is switched off here; it is tested
    on its own in the `_units` test file of the loader, which reads a file of every unit with the
    check on."""
    monkeypatch.setattr(load_heldout_norway, "require_sites", lambda sessions, dropped: None)


RELABEL = {"OSL_T": "BER", "TRO_R": "KRO"}


def _rows(osl, tro):
    return [
        norway_row(osl, "u1", "s1", "2019-11-28 19:08:00", "2019-11-29 07:08:00", 12.5),
        norway_row(osl, "u1", "s2", "2020-03-29 01:30:00", "2020-03-29 03:30:00", 1.0),
        norway_row(osl, "u2", "gap", "2020-03-29 02:30:00", "2020-03-29 05:00:00", 1.0),
        norway_row(tro, "u3", "s3", "2020-10-25 01:30:00", "2020-10-25 03:30:00", 2.0),
        norway_row(tro, "u3", "doubled", "2020-10-25 02:30:00", "2020-10-25 05:00:00", 2.0),
        norway_row(tro, "u3", "no-out", "2020-05-02 18:00:00", None, 10.0),
        norway_row(tro, "u4", "backwards", "2020-05-03 18:00:00", "2020-05-03 18:00:00", 1.0),
        norway_row(tro, "u4", "empty", "2020-05-04 18:00:00", "2020-05-05 07:00:00", 0.0),
        norway_row(tro, "u4", "s3", "2020-05-06 18:00:00", "2020-05-07 07:00:00", 10.0),
        norway_row(osl, "u5", "fast", "2021-05-08 18:00:00", "2021-05-08 19:00:00", 30.0),
    ]


def _held_out_bytes(tmp_path, rows, header=NORWAY_HEADER):
    return write_norway(raw_root(tmp_path / "made"), "held-out", rows, header=header).read_bytes()


def test_equal_to_the_training_loader_on_the_same_rows_relabelled(tmp_path):
    content = _held_out_bytes(tmp_path, _rows("OSL_T", "TRO_R"))
    sessions, dropped = load_heldout_norway.from_bytes(content)
    root = raw_root(tmp_path / "development")
    write_norway(root, "development", _rows("BER", "KRO"))
    expected, expected_dropped = load_norway.load("development", root)
    assert list(sessions["session_id"]) == ["s1", "s2", "s3"] and len(dropped) == 7
    assert set(sessions["role"]) == {"held-out"}
    assert sessions["role"].dtype == object
    relabelled = sessions.assign(
        site=sessions["site"].map(RELABEL).astype(object),
        role=pd.Series("development", index=sessions.index, dtype=object),
    )
    pd.testing.assert_frame_equal(relabelled, expected)
    dropped_relabelled = dropped.assign(site=dropped["site"].map(RELABEL))
    pd.testing.assert_frame_equal(dropped_relabelled, expected_dropped, check_dtype=False)
    assert dropped.dtypes.equals(expected_dropped.dtypes)


@pytest.mark.parametrize("site", ["ASK", "BER", "OSL_S", "BAR_2"])
def test_a_training_or_development_site_is_refused(tmp_path, site):
    rows = _rows("OSL_T", "TRO_R")
    rows.append(norway_row(site, "x", "x1", "2019-05-01 18:00:00", "2019-05-02 07:00:00", 10.0))
    with pytest.raises(LayoutError, match="outside the held-out sites"):
        load_heldout_norway.from_bytes(_held_out_bytes(tmp_path, rows))


def test_a_missing_column_is_refused(tmp_path):
    content = _held_out_bytes(tmp_path, _rows("OSL_T", "TRO_R"), header=NORWAY_HEADER[:-1])
    with pytest.raises(LayoutError, match="lacks columns"):
        load_heldout_norway.from_bytes(content)


def test_an_unreadable_time_is_refused(tmp_path):
    rows = _rows("OSL_T", "TRO_R")
    rows[0]["plugin_time"] = "28.11.2019 19:08"
    with pytest.raises(LayoutError, match="cannot be read"):
        load_heldout_norway.from_bytes(_held_out_bytes(tmp_path, rows))


def test_one_digit_fields_are_read_as_the_training_loader_reads_them(tmp_path):
    padded = [norway_row("OSL_T", "u", "s", "2019-01-02 08:00:00", "2019-01-02 17:00:00", 5.0)]
    short = [norway_row("OSL_T", "u", "s", "2019-01-02 08:00:00", "2019-01-02 17:00:00", 5.0)]
    short[0]["plugin_time"] = "2019-1-2 8:00:00"
    from_padded, _ = load_heldout_norway.from_bytes(_held_out_bytes(tmp_path / "a", padded))
    from_short, _ = load_heldout_norway.from_bytes(_held_out_bytes(tmp_path / "b", short))
    pd.testing.assert_frame_equal(from_short, from_padded)


def test_through_the_gated_opener_in_a_made_up_repository(tmp_path):
    content = _held_out_bytes(tmp_path, _rows("OSL_T", "TRO_R"))
    root = sealed_repo(tmp_path, {NORWAY_FILE: content})
    sessions, dropped = heldout_loader.norway_sites(root)
    direct, direct_dropped = load_heldout_norway.from_bytes(content)
    pd.testing.assert_frame_equal(sessions, direct)
    pd.testing.assert_frame_equal(dropped, direct_dropped)
    openings = heldout_loader.logged_openings(root)
    assert [(o["set"], o["file"]) for o in openings] == [NORWAY_FILE]


def _both(tmp_path, rows, patch):
    """The held-out loader on `rows` with the site codes OSL_T and TRO_R, and the development
    loader on the same rows relabelled, each file changed by `patch` after it is written."""
    held_out = patch(_held_out_bytes(tmp_path, rows))
    relabelled = [{**row, "location": RELABEL[row["location"]]} for row in rows]
    root = raw_root(tmp_path / "development")
    path = write_norway(root, "development", relabelled)
    path.write_bytes(patch(path.read_bytes()))
    return load_heldout_norway.from_bytes(held_out), load_norway.load("development", root)


def test_missing_fields_are_read_as_the_training_loader_reads_them(tmp_path):
    rows = [
        norway_row("OSL_T", "no-user", "m1", "2019-11-28 19:08:00", "2019-11-29 07:08:00", 12.5),
        norway_row("TRO_R", "u2", "m2", "2019-11-28 19:08:00", "2019-11-29 07:08:00", 7.0),
        norway_row("TRO_R", "u3", "m3", "2019-11-30 19:08:00", "2019-12-01 07:08:00", 6.0),
    ]
    rows[1]["energy_session"] = 777.25
    rows[2]["connection_time"] = 555.5

    def patch(content):
        content = content.replace(b'"no-user"', b'""')
        return content.replace(b"777,25", b"").replace(b"555,5", b"")

    (sessions, dropped), (expected, expected_dropped) = _both(tmp_path, rows, patch)
    assert list(sessions["session_id"]) == ["m1", "m3"]
    assert sessions.set_index("session_id").loc["m1", "user_id"] is None
    assert expected.set_index("session_id").loc["m1", "user_id"] is None
    assert list(dropped["session_id"]) == list(expected_dropped["session_id"]) == ["m2"]
    assert list(dropped["reason"]) == list(expected_dropped["reason"])
    relabelled = sessions.assign(
        site=sessions["site"].map(RELABEL).astype(object),
        role=pd.Series("development", index=sessions.index, dtype=object),
    )
    pd.testing.assert_frame_equal(relabelled, expected)


def test_an_unreadable_connection_time_is_refused(tmp_path):
    rows = _rows("OSL_T", "TRO_R")
    rows[0]["connection_time"] = 4321.5
    content = _held_out_bytes(tmp_path, rows).replace(b"4321,5", b"twelve")
    with pytest.raises(LayoutError, match="connection_time: 1 values are not numbers"):
        load_heldout_norway.from_bytes(content)


def test_a_row_without_a_plug_in_time_is_refused(tmp_path):
    rows = _rows("OSL_T", "TRO_R")
    rows[3]["plugin_time"] = None
    with pytest.raises(LayoutError, match="a row has no plug-in time or session id"):
        load_heldout_norway.from_bytes(_held_out_bytes(tmp_path, rows))


def test_a_row_without_a_session_id_is_refused(tmp_path):
    content = _held_out_bytes(tmp_path, _rows("OSL_T", "TRO_R")).replace(b'"backwards"', b'""')
    with pytest.raises(LayoutError, match="a row has no plug-in time or session id"):
        load_heldout_norway.from_bytes(content)


HEADER = ";".join(f'"{name}"' for name in NORWAY_HEADER)
ROW_1 = '"OSL_T";"u1";"s1";2019-11-28 19:08:00;2019-11-29 07:08:00;12;12,5'
ROW_2 = '"TRO_R";"u2";"s2";2019-11-30 19:08:00;2019-12-01 07:08:00;12;7,5'


def _file(*lines: str, end: str = "\n") -> bytes:
    return (end.join(lines) + end).encode("utf-8")


def _with(row: str, field: int, value: str) -> str:
    cells = row.split(";")
    cells[field] = value
    return ";".join(cells)


def test_text_fields_that_look_like_numbers_are_kept_as_written():
    rows = [_with(_with(ROW_1, 1, '"007"'), 2, '"0123"'), _with(_with(ROW_2, 1, "8"), 2, "45")]
    sessions, dropped = load_heldout_norway.from_bytes(_file(HEADER, *rows))
    assert list(sessions["user_id"]) == ["007", "8"]
    assert list(sessions["session_id"]) == ["0123", "45"]
    assert dropped.empty


def test_a_user_written_null_is_a_user():
    rows = [_with(ROW_1, 1, '"null"'), _with(ROW_2, 1, '"None"')]
    sessions, _ = load_heldout_norway.from_bytes(_file(HEADER, *rows))
    assert list(sessions["user_id"]) == ["null", "None"]


@pytest.mark.parametrize(
    "field, text, message",
    [
        (6, "NaN", "Norway energy_session: 1 values are not numbers"),
        (5, "nan", "Norway connection_time: 1 values are not numbers"),
        (4, "N/A", "Norway plugout_time: 1 times cannot be read"),
        (4, "null", "Norway plugout_time: 1 times cannot be read"),
    ],
)
def test_only_an_empty_field_and_na_are_missing(field, text, message):
    with pytest.raises(LayoutError) as refused:
        load_heldout_norway.from_bytes(_file(HEADER, _with(ROW_1, field, text), ROW_2))
    assert str(refused.value) == message


def test_the_count_of_unreadable_times_is_the_number_of_rows(tmp_path):
    rows = [_with(ROW_1, 3, "28.11.2019 19:08"), _with(ROW_2, 3, "30.11.2019 19:08"), ROW_1]
    with pytest.raises(LayoutError) as refused:
        load_heldout_norway.from_bytes(_file(HEADER, *rows))
    assert str(refused.value) == "Norway plugin_time: 2 times cannot be read"


WHAT = "Norway held-out file"
NO_LOCATION = "Norway held-out file: a row has no location"
NO_ID = "Norway held-out file: a row has no plug-in time or session id"
MALFORMED = {
    # name: (file, expected): an error message, or (session ids, users, removal reasons)
    "no location": (_file(HEADER, _with(ROW_1, 0, ""), ROW_2), NO_LOCATION),
    "no location and another site": (
        _file(HEADER, _with(ROW_1, 0, ""), _with(ROW_2, 0, '"ASK"')),
        NO_LOCATION,
    ),
    "no user": (_file(HEADER, _with(ROW_1, 1, ""), ROW_2), (["s1", "s2"], [None, "u2"], [])),
    "no session id": (_file(HEADER, _with(ROW_1, 2, ""), ROW_2), NO_ID),
    "no plug-in": (_file(HEADER, _with(ROW_1, 3, ""), ROW_2), NO_ID),
    "no plug-out": (
        _file(HEADER, _with(ROW_1, 4, ""), ROW_2),
        (["s2"], ["u2"], ["no plug-out time recorded"]),
    ),
    "no connection time": (
        _file(HEADER, _with(ROW_1, 5, ""), ROW_2),
        (["s1", "s2"], ["u1", "u2"], []),
    ),
    "no energy": (
        _file(HEADER, _with(ROW_1, 6, "NA"), ROW_2),
        (["s2"], ["u2"], ["energy missing or not positive"]),
    ),
    "30 February": (
        _file(HEADER, _with(ROW_1, 3, "2019-02-30 10:00:00"), ROW_2),
        "Norway plugin_time: 1 times cannot be read",
    ),
    "hour 24": (
        _file(HEADER, _with(ROW_1, 3, "2019-11-28 24:00:00"), ROW_2),
        "Norway plugin_time: 1 times cannot be read",
    ),
    "month 13": (
        _file(HEADER, ROW_1, _with(ROW_2, 4, "2019-13-01 07:08:00")),
        "Norway plugout_time: 1 times cannot be read",
    ),
    "empty": (b"", f"{WHAT} has no header line"),
    "blank lines only": (b"\n\n", f"{WHAT} has no header line"),
    "header only": (_file(HEADER), ([], [], [])),
    "not UTF-8": (
        _file(HEADER, ROW_1) + b"\xff\n",
        f"{WHAT} is not UTF-8 text (byte {len(_file(HEADER, ROW_1))})",
    ),
    "separator in quotes": (
        _file(HEADER, _with(ROW_1, 1, '"u;1"'), ROW_2),
        (["s1", "s2"], ["u;1", "u2"], []),
    ),
    "line break in quotes": (
        _file(HEADER, _with(ROW_1, 1, '"u\n1"'), ROW_2),
        (["s1", "s2"], ["u\n1", "u2"], []),
    ),
    "NUL": (_file(HEADER, _with(ROW_1, 1, '"u\0"'), ROW_2), f"{WHAT} holds a NUL character"),
    "one field too many first": (
        _file(HEADER, ROW_1 + ";9", ROW_2),
        f"{WHAT}: record 1 has 8 fields, the header 7",
    ),
    "one field too many later": (
        _file(HEADER, ROW_1, ROW_2 + ";9"),
        f"{WHAT}: record 2 has 8 fields, the header 7",
    ),
    "two fields too few": (
        _file(HEADER, ROW_1.rsplit(";", 2)[0], ROW_2),
        f"{WHAT}: record 1 has 5 fields, the header 7",
    ),
    "spaces only": (
        _file(HEADER, ROW_1, "  ", ROW_2),
        f"{WHAT}: record 2 has 1 fields, the header 7",
    ),
    "quote never closed": (
        _file(HEADER, ROW_1, _with(ROW_2, 6, '"7,5')),
        f"{WHAT} cannot be split into records: unexpected end of data",
    ),
    "text after a closing quote": (
        _file(HEADER, _with(ROW_1, 1, '"u"1'), ROW_2),
        f"{WHAT} cannot be split into records: ';' expected after '\"'",
    ),
    "a column named twice": (
        _file(HEADER + ';"user_id"', ROW_1 + ";x", ROW_2 + ";y"),
        f"{WHAT} names columns more than once: ['user_id']",
    ),
    "line ends of a carriage return": (
        _file(HEADER, ROW_1, ROW_2, end="\r"),
        (["s1", "s2"], ["u1", "u2"], []),
    ),
}


@pytest.mark.parametrize("name", MALFORMED)
def test_a_malformed_file_gives_a_layout_error_or_a_stated_removal(name):
    content, expected = MALFORMED[name]
    if isinstance(expected, str):
        with pytest.raises(LayoutError) as refused:
            load_heldout_norway.from_bytes(content)
        assert type(refused.value) is LayoutError
        assert str(refused.value) == expected
        return
    sessions, dropped = load_heldout_norway.from_bytes(content)
    assert (list(sessions["session_id"]), list(sessions["user_id"])) == expected[:2]
    assert list(dropped["reason"]) == expected[2]
    assert set(dropped["reason"]) <= set(NORWAY_REASONS)


NORWAY_REASONS = (  # the removal reasons `load_norway.to_sessions` states in its module text
    "no plug-out time recorded",
    "local time in the daylight-saving gap or doubled hour",
    "plug-out not after plug-in",
    "energy missing or not positive",
    "duplicate session id",
    "average above the physical limit",
)


def test_any_other_error_while_reading_is_a_layout_error_with_its_cause(monkeypatch):
    made_up = RuntimeError("made up")

    def fails(role, raw):
        raise made_up

    monkeypatch.setattr(load_norway, "to_sessions", fails)
    with pytest.raises(LayoutError) as refused:
        load_heldout_norway.from_bytes(_file(HEADER, ROW_1, ROW_2))
    assert str(refused.value) == "Norway held-out file cannot be read: RuntimeError: made up"
    assert refused.value.__cause__ is made_up


def test_a_field_longer_than_the_csv_limit_is_a_layout_error():
    with pytest.raises(LayoutError) as refused:
        load_heldout_norway.from_bytes(_file(HEADER, _with(ROW_1, 1, "u" * 200_000), ROW_2))
    assert str(refused.value) == (
        "Norway held-out file cannot be split into records: field larger than field limit (131072)"
    )
