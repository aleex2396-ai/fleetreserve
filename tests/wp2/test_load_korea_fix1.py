"""The Korean loader's draft first parsing fix, on made-up bytes: wider forms of the column names
and refusals that state a layout fact and never a value.

Wider forms and refusals (readings K1, K2 and K16): the separators tab and "|" after "," and ";";
header names compared after removing a byte order mark, surrounding spaces and surrounding double
quotes, and case-folded; no other name is accepted. A refusal for a header that lacks a name
lists, for each separator tried, the first line's fields (as text only when ASCII and without an
ASCII digit, else as a shape; at most 40, cut at 30 characters, their count given), the words
"the first line has the shape of a data record" when it has, and, for the separator with the most
first-line fields, the shape of each field of the second record (digit 9, letter a, ASCII
punctuation and space kept, any other character ?, cut at 30 characters). A field that cannot be
read as a time, a number, a charger type, a location type or a charger id is refused with its
column, the number of rows that fail and up to five distinct shapes with counts, never a value. A
file that is not UTF-8 is refused with the encoding tried and a byte offset. Every made-up value
below that must not appear in a message is distinctive, and the tests assert its absence.
"""

import csv
import io
import json
import random
import re
import string

import pandas as pd
import pytest
from fake_korea import KOREA_HEADER, korea_csv, korea_row

from fleetreserve import load_korea
from fleetreserve.sessions import LayoutError


@pytest.fixture(autouse=True)
def _rows_not_units(monkeypatch):
    """These tests are about reading one or two rows; the check that every named unit keeps a
    row is tested on its own in the units test file."""
    monkeypatch.setattr(load_korea, "require_portfolios", lambda sessions, dropped: None)


R = korea_row
ROWS = [
    R("S1", "17", "C1", "Apartment", "2021-10-01 01:00:00", "2021-10-02 05:30:00", 9.5),
    R("S2", "0", "C2", "Hotel", "2021-10-02 02:15:00", "2021-10-03 04:00:00", 12.0),
]
# Made-up values that no message may carry: each holds a digit or is not ASCII.
LOUD = {
    "ChargingsessionID": "QZV7716X",
    "UserID": "86753099",
    "ChargerID": "ZRK5521Q",
    "Demand": "31.4159",
    "ChargerCompany": "COQ8841",
}


def _one(location="Apartment", **fields) -> dict:
    row = R("K1", "5", "C1", location, "2022-06-01 04:00:00", "2022-06-01 05:00:00", 9.0)
    row.update(fields)
    return row


def _loud(number: int, **fields) -> dict:
    """A row whose every value is distinctive: `number` makes each differ from the others."""
    values = {name: f"{value}{number}" for name, value in LOUD.items()}
    row = R(
        values["ChargingsessionID"],
        values["UserID"],
        values["ChargerID"],
        "아파트",
        f"2022-06-0{number} 0{number}:07:06",
        f"2022-06-0{number} 0{number + 1}:08:09",
        values["Demand"],
        ChargerCompany=values["ChargerCompany"],
    )
    row.update(fields)
    return row


def _tokens(rows: list[dict]) -> set[str]:
    return {str(v) for row in rows for v in row.values() if v is not None and str(v) != ""}


def _refused(content: bytes) -> str:
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(content)
    return str(refused.value)


def _same(got, expected) -> None:
    pd.testing.assert_frame_equal(got[0], expected[0])
    pd.testing.assert_frame_equal(got[1], expected[1])


def _data_lines(rows: list[dict]) -> bytes:
    """The rows as a file with no header line."""
    lines = korea_csv(rows).decode("utf-8").splitlines()[1:]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _no_value(message: str, rows: list[dict]) -> None:
    """Nothing that is not ASCII, and no value of the rows that holds a digit (of more than three
    characters: a count or a short shape is not told from a value), is in the message."""
    assert message.isascii(), message
    for token in _tokens(rows):
        if any(char.isdigit() for char in token) and len(token) > 3:
            assert token not in message, token


# --- a headerless file ---------------------------------------------------------------------


def test_a_headerless_file_is_refused_with_its_layout_and_none_of_its_values():
    rows = [_loud(1), _loud(2), _loud(3)]
    message = _refused(_data_lines(rows))
    assert message.startswith("Korea file lacks columns: no separator gives UserID")
    _no_value(message, rows)
    # the first line is given as shapes, field by field, never as text: every one of its fields
    # holds a digit or is not ASCII
    assert "comma: 16 fields" in message and "shape " in message
    assert 'shape "aaa9999a9"' in message  # QZV7716X1: letters and digits, no text
    assert 'shape "???"' in message  # the Korean location
    assert "semicolon: 1 fields" in message and "tab: 1 fields" in message
    assert "bar: 1 fields" in message
    # the second record's field shapes, under the comma, the separator of the most fields
    assert "; second record under comma (16 fields), field shapes" in message
    assert '"aaa9999a9"' in message


def test_a_first_line_of_the_shape_of_a_data_record_is_given_as_shapes_with_those_words():
    message = _refused(_data_lines([_loud(1), _loud(2), _loud(3)]))
    assert "comma: 16 fields, the first line has the shape of a data record [" in message
    # a first line that differs in shape from the second record does not say so
    rows = [_loud(1), _loud(2, UserID="Q")]
    message = _refused(_data_lines(rows))
    assert "comma: 16 fields [" in message
    assert "the first line has the shape of a data record" not in message
    _no_value(message, rows)


def test_a_one_row_headerless_file_says_it_has_no_second_record():
    row = _loud(1)
    message = _refused(_data_lines([row]))
    _no_value(message, [row])
    assert message.endswith("; the file has no second record")


def test_a_title_line_is_given_as_a_shape_when_it_holds_a_digit_or_is_not_ascii():
    message = _refused("충전 기록 2022\n".encode() + korea_csv([_loud(1)]))
    assert message.isascii()
    assert "충전" not in message
    assert 'shape "?? ?? 9999"' in message


# --- a header that lacks a column ----------------------------------------------------------


def test_a_missing_column_gives_the_first_lines_field_names_and_the_second_records_shapes():
    rows = [_loud(1, Location="Hotel")]
    header = tuple(name for name in KOREA_HEADER if name != "Demand")
    message = _refused(korea_csv(rows, header=header))
    assert message.startswith("Korea file lacks columns: no separator gives UserID")
    assert "comma: 15 fields [" in message and '"ChargingsessionID"' in message
    assert '"ChargerType"' in message and '"EndDatetime"' in message
    assert '"Demand"' not in message.split("; second record")[0].split("comma:")[1]
    _no_value(message, rows)
    assert "; second record under comma (15 fields), field shapes" in message
    assert '"aaa9999a9"' in message  # the session id as a shape, not as text


def test_a_header_name_with_a_digit_or_not_ascii_is_given_as_a_shape():
    header = (*(n for n in KOREA_HEADER if n != "Demand"), "Extra7", "충전량")
    message = _refused(korea_csv([_one()], header=header))
    assert 'shape "aaaaa9"' in message and 'shape "???"' in message
    assert "Extra7" not in message and message.isascii()


def test_a_header_name_is_cut_at_30_characters_and_at_most_40_fields_are_shown():
    header = (*(n for n in KOREA_HEADER if n != "Demand"), "x" * 50)
    message = _refused(korea_csv([_one()], header=header))
    assert '"' + "x" * 30 + '"' in message and "x" * 31 not in message
    many = ",".join(f"c{n}" for n in range(60)) + "\n" + ",".join("1" * 60) + "\n"
    message = _refused(many.encode())
    assert "comma: 60 fields [" in message
    assert message.count('shape "a99"') + message.count('shape "a9"') == 40


@pytest.mark.parametrize(
    "alias", ["ev_id", "Energy", "SessionID", "Charging session ID", "Demand_kWh", "Locations"]
)
def test_another_name_is_never_accepted(alias):
    header = tuple(alias if name == "Demand" else name for name in KOREA_HEADER)
    assert _refused(korea_csv(ROWS, header=header)).startswith("Korea file lacks columns")


def test_a_one_character_difference_in_a_name_is_not_fuzzy_matched():
    header = tuple("ChargerID2" if name == "ChargerID" else name for name in KOREA_HEADER)
    assert _refused(korea_csv(ROWS, header=header)).startswith("Korea file lacks columns")


# --- wider forms of the same names ---------------------------------------------------------


@pytest.mark.parametrize("sep", ["\t", "|"])
def test_a_tab_or_bar_separated_file_gives_the_table_of_the_comma_file(sep):
    plain = load_korea.from_bytes(korea_csv(ROWS))
    _same(load_korea.from_bytes(korea_csv(ROWS, sep=sep)), plain)


@pytest.mark.parametrize("sep", ["\t", "|"])
def test_under_a_tab_or_bar_a_decimal_comma_is_not_a_number(sep):
    message = _refused(korea_csv([_one(Demand="12,5")], sep=sep))
    assert message == 'Korea Demand: 1 values are not numbers; field shapes: "99,9" x 1'


def test_header_names_in_double_quotes_after_spaces_and_in_capitals_give_the_same_table():
    plain = load_korea.from_bytes(korea_csv(ROWS))
    lines = korea_csv(ROWS).decode("utf-8").splitlines()
    spaced = ", ".join(f' "{name.upper()}" ' for name in KOREA_HEADER)
    _same(load_korea.from_bytes(("\n".join([spaced, *lines[1:]]) + "\n").encode()), plain)
    bom = "﻿" + "\n".join(lines) + "\n"
    _same(load_korea.from_bytes(bom.encode("utf-8")), plain)
    both = "﻿" + "\n".join([spaced, *lines[1:]]) + "\n"
    _same(load_korea.from_bytes(both.encode("utf-8")), plain)


def test_a_name_twice_in_a_wider_form_is_a_column_named_twice():
    lines = korea_csv(ROWS).decode("utf-8").splitlines()
    content = "\n".join([lines[0] + ',"DEMAND"', *(line + ",1" for line in lines[1:])]) + "\n"
    assert _refused(content.encode()) == (
        "Korea file: 1 header names repeat after trimming and case folding"
    )


def test_comma_and_semicolon_come_first_and_tab_and_bar_are_tried_after_them():
    assert load_korea.SEPARATORS == (",", ";", "\t", "|")
    # a file whose header has the names under ";" is still read under ";", decimal comma included
    _same(
        load_korea.from_bytes(korea_csv(ROWS, sep=";")),
        load_korea.from_bytes(korea_csv(ROWS)),
    )


# --- a field that cannot be read: its column, a count and shapes, never a value -------------


def test_a_bad_time_gives_the_column_the_count_and_the_shape_alone():
    rows = [_one(), _one(ChargingsessionID="K2", EndDatetime="2022-06-01 4:08:9")]
    assert _refused(korea_csv(rows)) == (
        'Korea EndDatetime: 1 times cannot be read; field shapes: "9999-99-99 9:99:9" x 1'
    )
    # the day and time columns joined: the shape of the joined text
    header = tuple(n for n in KOREA_HEADER if n not in ("StartDatetime", "EndDatetime"))
    rows = [_one(), _one(ChargingsessionID="K2", StartTime="4:8")]
    assert _refused(korea_csv(rows, header=header)) == (
        'Korea StartDay and StartTime: 1 times cannot be read; field shapes: "9999-99-99 99:9" x 1'
    )


def test_a_bad_time_in_korean_script_is_shown_only_as_question_marks():
    rows = [_one(), _one(ChargingsessionID="K2", StartDatetime="오전 7시 8분")]
    assert _refused(korea_csv(rows)) == (
        'Korea StartDatetime: 1 times cannot be read; field shapes: "?? 9? 9?" x 1'
    )


def test_at_most_five_distinct_shapes_are_given_with_their_counts_and_the_rest_counted():
    bad = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "gggggggg", "hhhhhhhh"]
    rows = [_one(ChargingsessionID=f"K{n}", EndDatetime=text) for n, text in enumerate(bad)]
    assert _refused(korea_csv(rows)) == (
        "Korea EndDatetime: 8 times cannot be read; field shapes: "
        '"a" x 1, "aa" x 1, "aaa" x 1, "aaaa" x 1, "aaaaa" x 1, and 2 more shapes'
    )
    rows = [_one(ChargingsessionID=f"K{n}", EndDatetime="x") for n in range(3)]
    assert _refused(korea_csv(rows)) == (
        'Korea EndDatetime: 3 times cannot be read; field shapes: "a" x 3'
    )


def test_a_shape_is_cut_at_30_characters():
    message = _refused(korea_csv([_one(EndDatetime="9" * 80)]))
    assert message.endswith('"' + "9" * 30 + '" x 1') and "9" * 31 not in message


def test_a_bad_energy_gives_its_shape_alone():
    rows = [_one(Demand="12 kWh"), _one(ChargingsessionID="K2", Demand="x7")]
    assert _refused(korea_csv(rows)) == (
        'Korea Demand: 2 values are not numbers; field shapes: "99 aaa" x 1, "a9" x 1'
    )


def test_a_charger_type_outside_is_given_by_shape_and_a_missing_one_as_missing():
    rows = [
        _one(ChargerType="fast"),
        _one(ChargingsessionID="K2", ChargerType="2"),
        _one(ChargingsessionID="K3", ChargerType=None),
    ]
    assert _refused(korea_csv(rows)) == (
        'Korea ChargerType: 3 values are neither 0 nor 1; field shapes: "aaaa" x 1, "9" x 1, '
        "missing x 1"
    )


def test_an_unknown_location_in_korean_script_is_shown_only_as_question_mark_shapes():
    rows = [
        _one(),
        _one("아파트", ChargingsessionID="K2"),
        _one("호텔", ChargingsessionID="K3"),
        _one("아파트 7동", ChargingsessionID="K4"),
    ]
    message = _refused(korea_csv(rows))
    assert message == (
        "Korea Location: 3 rows hold no location type of the paper (3 distinct values); "
        'field shapes: "???" x 1, "??" x 1, "??? 9?" x 1'
    )
    assert message.isascii()
    for word in ("아파트", "호텔", "7동"):
        assert word not in message


def test_an_unknown_location_in_latin_script_is_given_by_shape_not_by_value():
    message = _refused(korea_csv([_one("Zanzibarville")]))
    assert "Zanzibarville" not in message
    assert message.endswith('field shapes: "aaaaaaaaaaaaa" x 1')


def test_a_charger_id_marker_is_given_by_shape_alone():
    rows = [_one(ChargerID="NULL"), _one(ChargingsessionID="K2", ChargerID="-12")]
    assert _refused(korea_csv(rows)) == (
        "Korea ChargerID: 2 values are written as a missing marker or a negative number; "
        'field shapes: "aaaa" x 1, "-99" x 1'
    )


def test_rows_with_no_plug_in_time_or_no_session_id_are_counted_apart():
    rows = [
        _one(),
        _one(ChargingsessionID="K2", StartDatetime=None),
        _one(ChargingsessionID=None),
        _one(ChargingsessionID=None, StartDatetime=None),
    ]
    assert _refused(korea_csv(rows)) == (
        "Korea file: a row has no plug-in time or no session id: 2 rows with no plug-in time, "
        "2 rows with no session id"
    )


def test_rows_with_more_fields_than_the_header_give_the_header_width():
    plain = korea_csv([_one(), _one(ChargingsessionID="K2")])
    every = plain.replace(b"\n", b",extra\n").replace(b",extra\n", b"\n", 1)
    assert _refused(every) == (
        "Korea file: rows hold more fields than the header (the header has 16 fields)"
    )
    assert _refused(plain.rstrip(b"\n") + b",extra\n") == (
        "Korea file: a row cannot be split under the header (more fields than the header, or a "
        "quote never closed; the header has 16 fields)"
    )


# --- bytes that are not UTF-8 --------------------------------------------------------------


def test_a_file_that_is_not_utf8_is_refused_with_the_encoding_tried_and_the_byte_offset():
    text = korea_csv([_one("Apartment")]).decode("utf-8")
    for codec in ("cp949", "euc-kr"):
        content = text.replace("Apartment", "아파트").encode(codec)
        first = next(n for n, byte in enumerate(content) if byte >= 0x80)
        assert _refused(content) == (
            "Korea file is not UTF-8 text (utf-8-sig tried; the first byte it cannot read is "
            f"at offset {first})"
        )
    assert _refused(text.encode("utf-16")).endswith("is at offset 0)")


def test_the_byte_offset_counts_a_byte_order_mark():
    plain = b"abc\xffdef"
    assert _refused(plain).endswith("at offset 3)")
    assert _refused(b"\xef\xbb\xbf" + plain).endswith("at offset 6)")


# --- no message carries a value of the file ------------------------------------------------


def _bad_files() -> dict[str, tuple[bytes, list[dict]]]:
    """Files with a distinctive value in every field of every row, each with one fault."""
    rows = [_loud(1), _loud(2), _loud(3)]
    header = tuple(name for name in KOREA_HEADER if name != "Demand")
    hotel = {"Location": "Hotel"}
    return {
        "a missing column": (korea_csv(rows, header=header), rows),
        "a headerless file": (_data_lines(rows), rows),
        "a headerless file whose second row differs in shape": (
            _data_lines([_loud(1), _loud(2, UserID="QZ-8", ChargerType="x9")]),
            rows,
        ),
        "a bad end time": (korea_csv([_loud(1, **hotel), _loud(2, EndDatetime="2022-6-1")]), rows),
        "a bad energy": (korea_csv([_loud(1, **hotel), _loud(2, Demand="QZ31.4x")]), rows),
        "an unknown location": (korea_csv(rows), rows),
        "a charger type": (korea_csv([_loud(1, **hotel, ChargerType="Q77")]), rows),
        "a charger id marker": (korea_csv([_loud(1, **hotel, ChargerID="-77")]), rows),
        "rows with more fields": (korea_csv(rows).replace(b"\n", b",QZ77,QV88\n"), rows),
        "a tab file with a decimal comma": (
            korea_csv([_loud(1, **hotel, Demand="31,4159")], sep="\t"),
            rows,
        ),
    }


@pytest.mark.parametrize("case", list(_bad_files()))
def test_no_message_carries_a_value_of_the_file(case):
    content, rows = _bad_files()[case]
    message = _refused(content)
    assert message.isascii(), message
    for token in _tokens(rows) | {"QZ31.4x", "Q77", "2022-6-1", "QZ77", "QV88", "31,4159"}:
        if (any(char.isdigit() for char in token) and len(token) > 3) or not token.isascii():
            assert token not in message, (case, token)
    # no run of letters and digits such as an id, and no year of a date, is in a message (a
    # shape writes every digit as 9)
    assert not re.search(r"[A-Z]{2,}[0-9]{2,}", message), message
    assert not re.search(r"20[0-9]{2}-", message), message


# --- the second correction (reading K16): a first-line field with a line break --------------
# A quote that opens a field in line 1 and is not closed there makes that field run over the
# lines that follow; its text must never be printed. Made-up digit-free, ASCII values.

LATER_ONE = "QuokkaLineTwo"
LATER_TWO = "WombatLineTwo"
LATER_THREE = "NumbatLineThree"
LATER = (LATER_ONE, LATER_TWO, LATER_THREE, "Quokka", "Wombat", "Numbat")
SEPARATORS = [",", ";", "\t", "|"]


def _absent(message: str) -> None:
    for value in LATER:
        assert value not in message


@pytest.mark.parametrize("sep", SEPARATORS)
def test_an_open_quote_in_line_one_never_prints_the_lines_that_follow(sep):
    content = f'a{sep}"b\n{LATER_ONE}{sep}{LATER_TWO}\n{LATER_THREE}{sep}yyy\n'.encode()
    message = _refused(content)
    assert "lacks columns" in message
    _absent(message)
    assert 'shape "a?aaaaaaaaaaaaa' in message  # the runaway field, as its shape


@pytest.mark.parametrize("sep", SEPARATORS)
def test_an_open_quote_after_several_fields_never_prints_the_lines_that_follow(sep):
    head = sep.join(["user", "energy", "station"])
    content = f'{head}{sep}"x\n{LATER_ONE}{sep}{LATER_TWO}\n{LATER_THREE}\n'.encode()
    _absent(_refused(content))


@pytest.mark.parametrize("sep", SEPARATORS)
def test_a_quote_that_closes_in_a_later_line_never_prints_the_lines_it_covers(sep):
    content = (
        f'a{sep}"b\n{LATER_ONE}{sep}{LATER_TWO}"{sep}{TAIL_ONE}\n{LATER_THREE}{sep}{TAIL_TWO}\n'
    ).encode()
    _absent(_refused(content))


@pytest.mark.parametrize("end", ["\r", "\r\n"])
@pytest.mark.parametrize("sep", SEPARATORS)
def test_a_lone_carriage_return_or_a_crlf_line_end_inside_the_open_field_is_not_printed(sep, end):
    content = f'a{sep}"b{end}{LATER_ONE}{sep}{LATER_TWO}{end}{LATER_THREE}{end}'.encode()
    message = _refused(content)
    _absent(message)
    assert "\\r" not in message  # a line break is never printed as text, only as a shape


@pytest.mark.parametrize("sep", SEPARATORS)
def test_a_quoted_separator_is_one_field_and_the_text_after_it_is_not_printed(sep):
    content = f'"p{sep}q"{sep}r{sep}"s\n{LATER_ONE}{sep}{LATER_TWO}\n{LATER_THREE}\n'.encode()
    _absent(_refused(content))  # the header cannot be split: a fixed text, no value


@pytest.mark.parametrize("sep", SEPARATORS)
def test_a_quoted_separator_in_a_closed_field_is_one_text_field_and_the_next_line_is_shapes(sep):
    content = f'"p{sep}q"{sep}r\n{LATER_ONE}{sep}{LATER_TWO}\n'.encode()
    message = _refused(content)
    _absent(message)
    if sep == "\t":  # a tab is not printable ASCII: the field that holds it is a shape
        assert 'shape "a?a"' in message
    else:
        assert json.dumps(f"p{sep}q") in message  # one field of line 1, with no digit: text
    assert '"r"' in message


@pytest.mark.parametrize("sep", SEPARATORS)
def test_blank_lines_before_the_first_record_show_the_next_line_as_shapes_only(sep):
    content = f"\n\r\n{LATER_ONE}{sep}{LATER_TWO}\n{LATER_THREE}{sep}d\n".encode()
    _absent(_refused(content))


def test_a_digit_free_ascii_first_line_without_a_line_break_is_still_printed_as_text():
    # the remaining, stated risk (reading K16)
    message = _refused(b"alpha,beta,gamma\nQuokkaLineTwo,x,y\n")
    assert '["alpha", "beta", "gamma"]' in message
    assert LATER_ONE not in message


def test_a_field_is_text_only_when_ascii_with_no_digit_and_no_line_break():
    assert load_korea.first_line_field("name") == '"name"'
    assert load_korea.first_line_field("a b,c") == '"a b,c"'
    assert load_korea.first_line_field("ab\ncd") == 'shape "aa?aa"'
    assert load_korea.first_line_field("ab\rcd") == 'shape "aa?aa"'
    assert load_korea.first_line_field("ab\r\ncd") == 'shape "aa??aa"'
    assert load_korea.first_line_field("ab1") == 'shape "aa9"'
    assert load_korea.first_line_field("충전") == 'shape "??"'
    assert load_korea.first_line_field("x" * 50) == json.dumps("x" * 30)


TAIL_ONE = "TailSecretOne"
TAIL_TWO = "TailSecretTwo"
TAILS = (TAIL_ONE, TAIL_TWO, "TailSecret", LATER_ONE, LATER_TWO, LATER_THREE)


def _tails_absent(message: str) -> None:
    for value in TAILS:
        assert value not in message


@pytest.mark.parametrize("sep", SEPARATORS)
@pytest.mark.parametrize("end", ["\n", "\r", "\r\n"])
def test_fields_after_a_quoted_field_that_runs_over_a_line_end_are_shapes(sep, end):
    # a further case: the quote closes on line 2, the fields after it are on line 2 and are
    # digit-free ASCII, so each would be text if the record were taken as the first line's
    content = (
        f'"hdr{end}x"{sep}{TAIL_ONE}{sep}{TAIL_TWO}{end}{LATER_ONE}{sep}{LATER_TWO}{end}'
    ).encode()
    message = _refused(content)
    _tails_absent(message)
    assert 'shape "aaa?' in message  # the field that holds the line end, as its shape
    assert f"{'comma' if sep == ',' else ''}" in message


@pytest.mark.parametrize("sep", SEPARATORS)
@pytest.mark.parametrize("end", ["\n", "\r", "\r\n"])
def test_a_quote_closing_on_line_three_is_followed_by_shapes_only(sep, end):
    content = (
        f'a{sep}"b{end}{LATER_ONE}{end}{LATER_TWO}"{sep}{TAIL_ONE}{sep}{TAIL_TWO}{end}{LATER_THREE}'
    ).encode()
    _tails_absent(_refused(content))


@pytest.mark.parametrize("sep", SEPARATORS)
def test_a_first_line_that_is_a_complete_record_by_itself_is_still_text_for_its_own_fields(sep):
    content = f"alpha{sep}beta{sep}gamma\n{LATER_ONE}{sep}{LATER_TWO}\n".encode()
    message = _refused(content)
    assert '"alpha", "beta", "gamma"' in message
    _tails_absent(message)


def test_on_first_line_is_true_only_for_a_record_that_is_the_first_physical_line_alone():
    assert load_korea.on_first_line("ab,cd\nef,gh\n", ",", ["ab", "cd"])
    assert not load_korea.on_first_line("ab,cd", ",", ["ab", "cd"])  # one record, no line end
    assert not load_korea.on_first_line("ab,cd\n", ",", ["ab", "cd"])  # one record, a line end
    assert not load_korea.on_first_line(
        "ab,cd\r\n\r\n", ",", ["ab", "cd"]
    )  # blank lines only after
    assert not load_korea.on_first_line("ab,cd\u2028ef,gh", ",", ["ab", "cd\u2028ef", "gh"])
    assert load_korea.on_first_line('"a,b",c\nx\n', ",", ["a,b", "c"])
    assert not load_korea.on_first_line('"a\nb",c\nx\n', ",", ["a\nb", "c"])
    assert not load_korea.on_first_line('"a\rb",c\rx\r', ",", ["a\rb", "c"])
    assert not load_korea.on_first_line('a,"b\nc"\n', ",", ["a", "b\nc"])
    assert not load_korea.on_first_line("\nab,cd\n", ",", ["ab", "cd"])
    assert not load_korea.on_first_line("\r\nab,cd\n", ",", ["ab", "cd"])


def _marker(rng: random.Random) -> str:
    return "Qk" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(7))


def test_no_digit_free_value_after_the_first_physical_line_reaches_a_refusal_in_random_files():
    """Fuzz: line 1 is random quotes, separators and short text; every field of every later
    line is a digit-free marker word (some followed by a quote that closes a quote of line 1,
    some quoted, some with a line end inside). None of the markers may appear in a refusal."""
    rng = random.Random(20261123)
    bits = ['"', '""', ",", ";", "\t", "|", "ab", " ", "x y", '"open', 'x"', "q"]
    ends = ["\n", "\r", "\r\n"]
    refused_files = 0
    for _case in range(600):
        sep = rng.choice(SEPARATORS)
        end = rng.choice(ends)
        first = "".join(rng.choice(bits) for _ in range(rng.randint(1, 8)))
        markers: list[str] = []
        lines = []
        for _line in range(rng.randint(1, 3)):
            fields = []
            for _field in range(rng.randint(1, 4)):
                marker = _marker(rng)
                markers.append(marker)
                shape_of_field = rng.choice(["plain", "closing", "quoted", "inner"])
                if shape_of_field == "closing":
                    fields.append(marker + '"')
                elif shape_of_field == "quoted":
                    fields.append('"' + marker + '"')
                elif shape_of_field == "inner":
                    marker_two = _marker(rng)
                    markers.append(marker_two)
                    fields.append('"' + marker + rng.choice(ends) + marker_two + '"')
                else:
                    fields.append(marker)
            lines.append(sep.join(fields))
        body = first + end + end.join(lines) + rng.choice(["", end])
        try:
            load_korea.from_bytes(body.encode())
        except LayoutError as refused:
            message = str(refused)
        except Exception:  # noqa: BLE001  (a file that is not a refusal here is not a leak case)
            continue
        else:
            continue
        refused_files += 1
        for marker in markers:
            assert marker not in message, (body, marker)
    assert refused_files > 500


# ---- a repeated header name is refused with a count, never a name ---------------------

NAMES = [
    "ChargingsessionID", "UserID", "ChargerID", "Location", "ChargerType", "Demand",
    "StartDatetime", "EndDatetime",
]  # fmt: skip
DUP_FIELDS = {
    "a date": "2022-08-31",
    "korean text": "한국",
    "a quoted name over two lines": '"MkSecretAA\nMkSecretBB"',
    "a name with a digit": "Chan7",
    "digit-free text": "Chanel",
    "two empty names": "",
}


@pytest.mark.parametrize("sep", SEPARATORS)
@pytest.mark.parametrize("what", list(DUP_FIELDS))
def test_a_repeated_extra_header_name_is_refused_with_its_count_only(sep, what):
    field = DUP_FIELDS[what]
    header = sep.join(NAMES + [field, field])
    content = (header + "\n" + sep.join(["1"] * (len(NAMES) + 2)) + "\n").encode()
    assert _refused(content) == "Korea file: 1 header names repeat after trimming and case folding"


def test_two_repeated_names_of_a_header_are_counted_not_named():
    header = ",".join(NAMES + ["Chan7", "Chan7", "한국", "한국"])
    content = (header + "\n" + ",".join(["x"] * (len(NAMES) + 4)) + "\n").encode()
    assert _refused(content) == "Korea file: 2 header names repeat after trimming and case folding"


# ---- seeded property test: what a refusal may print -----------------------------------------
# A file is its first physical line, then later lines. The first line is built from a fixed
# vocabulary that holds no marker; every field of a later line is a distinctive digit-free marker
# word, some with a unique six-digit number, non-ASCII text, quotes, a separator or a line break.
# Every message a refusal gives must be ASCII; hold no marker and no number of the file; hold, as
# text, only printable-ASCII, digit-free pieces of the first physical line (the text up to the
# first boundary of `str.splitlines`), the first record of the whole file being that line's own
# first record and a second non-blank record following; and give anything else as a shape.

BOUNDARIES = [
    "\n",
    "\r",
    "\r\n",
    "\x0b",
    "\x0c",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
]
PROPERTY_FILES = 8000
FIRST_WORDS = [
    "alpha", "beta gamma", "User Name", "category", "x", "Apartment", "hotel", "label", "plain",
    "A-B", "q_r", "EV1", "5", "2022-08-31", "EV_id", "none", "",
]  # fmt: skip
FIRST_ODD = ["\x01", "\x0b", "\\n", "\u2028", "\x85", "é", "한국", "٣٤٥", "１２３", " ", "﻿"]
EXTRA_NAMES = ["ChargerCompany", "Remark"]
LEAD = "Korea file lacks columns: no separator gives"
REPEAT = re.compile(r"Korea file: \d+ header names repeat after trimming and case folding")
SEPARATOR_OF = {"comma": ",", "semicolon": ";", "tab": "\t", "bar": "|"}


class _Gen:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.count = 0
        self.numbers: list[str] = []

    def marker(self) -> str:
        self.count += 1
        letters = "".join(chr(97 + (self.count // 26**i) % 26) for i in range(5))
        return "Mk" + letters + "Zz"

    def number(self) -> str:
        number = str(self.rng.randint(100000, 999999))
        self.numbers.append(number)
        return number

    def first_field(self, sep: str) -> str:
        rng, word = self.rng, self.rng.choice(FIRST_WORDS)
        draw = rng.random()
        for limit, field in [
            (0.35, word),
            (0.42, rng.choice(FIRST_ODD) + word),
            (0.50, '"' + word + '"'),
            (0.55, '"' + word + sep + 'zz"'),
            (0.60, '"' + word + '""qq"'),
            (0.70, '"' + word),
            (0.74, word + '"qq'),
            (0.78, '"' + word + '"tail'),
            (0.83, word * rng.randint(20, 90)),
            (0.86, '""'),
            (0.88, '"'),
        ]:
            if draw < limit:
                return field
        return word

    def later_field(self, sep: str) -> str:
        rng, mark = self.rng, self.marker()
        draw = rng.random()
        if draw < 0.20:
            return mark + self.number()
        if draw < 0.26:
            return self.number() + mark
        if draw < 0.31:
            return mark + rng.choice(["é", "٣", "한", " "])
        if draw < 0.39:
            return '"' + mark + '"'
        if draw < 0.44:
            return '"' + mark + sep + self.marker() + '"'
        if draw < 0.49:
            return '"' + mark + '""' + self.marker() + '"'
        if draw < 0.56:
            return '"' + mark
        if draw < 0.60:
            return mark + '"' + self.marker()
        if draw < 0.68:
            return '"' + mark + rng.choice(["\n", "\r\n", "\r"]) + self.marker() + '"'
        if draw < 0.71:
            return '"' + mark + rng.choice(["\n", "\r\n", "\r"])
        if draw < 0.76:
            return '"' + mark + '"' + self.marker()
        if draw < 0.78:
            return ""
        if draw < 0.80:
            return "2022-08-3" + str(rng.randint(1, 9)) + "T08" + mark
        if draw < 0.82:
            return mark * rng.randint(5, 60)
        if draw < 0.825:
            return "9" * rng.randint(131100, 131500)
        return mark

    def first_line(self, sep: str) -> str:
        rng = self.rng
        draw = rng.random()
        if draw < 0.08:
            return ""
        if draw < 0.12:
            return rng.choice([" ", "   ", "\t"])
        if draw < 0.40:  # some of the names, maybe all of them
            cols = NAMES + EXTRA_NAMES
            rng.shuffle(cols)
            cols = cols[: rng.choice([5, 6, 8, 9])]
            cols = [c if rng.random() < 0.8 else '"' + c + '"' for c in cols]
            for _ in range(rng.randint(0, 2)):
                cols.insert(rng.randrange(len(cols) + 1), self.first_field(sep))
            fields = cols
        else:
            fields = [self.first_field(sep) for _ in range(rng.choice([1, 1, 2, 3, 4, 6, 9]))]
        return rng.choice([sep, sep, sep, rng.choice(SEPARATORS)]).join(fields)

    def repeated_header_file(self, sep: str, index: int) -> tuple[str, list[str]]:
        """A header that holds all five names and two equal extra fields, in the forms of
        DUP_FIELDS and others, then a row of as many fields (so the file reaches the
        repeated-name refusal): the first line, and the later lines."""
        rng = self.rng
        forms = [
            "2022-08-31",
            "한국",
            "Chan7",
            "Chanel",
            "",
            '"' + rng.choice(FIRST_WORDS[:8]) + "\n" + self.marker() + '"',
            '"' + rng.choice(FIRST_WORDS[:8]) + "\r\n" + self.marker() + '"',
            rng.choice(NAMES).upper(),
            " " + rng.choice(NAMES) + " ",
            self.number(),
        ]
        extra = forms[index % len(forms)]
        cols = NAMES + [extra, extra] + ([rng.choice(EXTRA_NAMES)] if index % 3 == 0 else [])
        rng.shuffle(cols)
        row = [self.marker() if rng.random() < 0.7 else self.number() for _ in cols]
        return sep.join(cols), [sep.join(row)]

    def build(self, index: int) -> tuple[bytes, str]:
        rng = self.rng
        sep = rng.choice(SEPARATORS)
        self.numbers.clear()
        if index % 6 == 0:
            line, later = self.repeated_header_file(sep, index // 6)
        else:
            line = self.first_line(sep)
            later = []
            for _ in range(rng.choice([0, 0, 1, 1, 2, 2, 3, 5])):
                other = rng.choice([sep, sep, sep, rng.choice(SEPARATORS)])
                width = rng.choice([1, 2, 3, 5, 6, 8, 9, 10, 11])
                later.append(other.join(self.later_field(sep) for _ in range(width)))
            if later and rng.random() < 0.2:
                later[0] = sep.join(self.later_field(sep) for _ in line.split(sep))
        first_end = rng.choice(["\n", "\r\n", "\r"])
        one_record = index % 7 == 3  # lines end with a character the csv module does not read
        if one_record:
            first_end = rng.choice(["\u2028", "\x1e", "\x0b", "\x1c", "\x85"])
            line = line.replace("\n", " ").replace("\r", " ")
            later = [record.replace("\n", " ").replace("\r", " ") for record in later]
        text = line
        for record in later:
            end = first_end if one_record or rng.random() < 0.8 else rng.choice(["\n", "\r"])
            text += end + record
            if rng.random() < 0.08:
                text += first_end
        if rng.random() < 0.5:
            text += first_end
        if one_record:  # with and without a real line end after the one record
            text += rng.choice(["", "", "\n", "\r\n", "\r"])
        if index % 5 == 2 and index % 6 != 0 and not one_record:  # any boundary of splitlines
            boundary = rng.choice(BOUNDARIES)
            ordinary = sep.join(self.later_field(sep) for _ in range(rng.choice([1, 3, 5])))
            tail = rng.choice(
                ["\n" + ordinary, "\r\n" + ordinary, "\n   \n", "\r\n \t \r\n", "", "\n"]
            )
            text = boundary.join([line, *later]) + tail
        text = rng.choice(["", "", "", "﻿", "﻿﻿"]) + text
        return text.encode(), line


_JSON = json.JSONDecoder()
_PART = re.compile(r"(comma|semicolon|tab|bar): (\d+) fields(, the first line has the shape of)?")


def _parse_lacks(message: str):
    """The texts (separator, text) and the shapes a "lacks columns" message prints, or None when
    the message is not in its known form."""
    marker = "shown): "
    at = message.index(marker) + len(marker)
    texts, shapes = [], []
    for part in range(4):
        found = _PART.match(message, at)
        if not found:
            return None
        at = found.end()
        if message.startswith(" a data record", at):
            at += len(" a data record")
        if not message.startswith(" [", at):
            return None
        at += 2
        while message[at] != "]":
            is_shape = message.startswith("shape ", at)
            at += 6 if is_shape else 0
            value, at = _JSON.raw_decode(message, at)
            (shapes if is_shape else texts).append(value if is_shape else (found[1], value))
            if message[at] == ",":
                at += 2
            elif message[at] != "]":
                return None
        at += 1
        if part < 3:
            if not message.startswith("; ", at):
                return None
            at += 2
    if message.startswith("; the file has no second record", at):
        return texts, shapes
    tail = re.compile(
        r"; second record under (comma|semicolon|tab|bar) \(\d+ fields\), field shapes "
        r"\(each cut at 30 characters, at most 40 shown\): \["
    ).match(message, at)
    if not tail:
        return None
    at = tail.end()
    while message[at] != "]":
        value, at = _JSON.raw_decode(message, at)
        shapes.append(value)
        if message[at] == ",":
            at += 2
        elif message[at] != "]":
            return None
    return (texts, shapes) if at == len(message) - 1 else None


def _first_record(text: str, sep: str):
    try:
        for row in csv.reader(io.StringIO(text, newline=""), delimiter=sep):
            return row
    except csv.Error:
        return None
    return []


def _problems(content: bytes, numbers: list[str], message: str) -> list[str]:
    bad = []
    if not message.isascii():
        bad.append("not ASCII")
    if re.search(r"Mk[a-z]{5}Zz", message):
        bad.append("a marker word of a later line")
    bad += [f"the file number {n}" for n in numbers if n in message]
    if "2022-08-3" in message:
        bad.append("a date")
    cleaned = re.sub(r"\\u00[0-1][0-9a-f]", "", message)  # a control character's escape
    for digits in re.findall(r"\d+", cleaned):
        if len(digits) >= 4 and set(digits) != {"9"} and digits != "131072":
            bad.append(f"the digit run {digits}")
    if REPEAT.fullmatch(message) or not message.startswith(LEAD):
        return bad
    parsed = _parse_lacks(message)
    if parsed is None:
        return bad + ["a message not in its known form"]
    texts, shapes = parsed
    text = content.decode("utf-8-sig")
    first = (text.splitlines() or [""])[0]
    for name, value in texts:
        sep = SEPARATOR_OF[name]
        try:
            second = sum(
                1
                for row in csv.reader(io.StringIO(text, newline=""), delimiter=sep)
                if any(field.strip(" \t") for field in row)
            )
        except csv.Error:
            second = 0
        if second < 2:
            bad.append(f"text {value!r} printed with no second non-blank record")
        if not all(" " <= c <= "~" and c not in string.digits for c in value):
            bad.append(f"text {value!r} is not printable ASCII without a digit")
        letters = iter(first)
        if not all(c in letters for c in value):
            bad.append(f"text {value!r} not from the first physical line")
        try:
            own = next(csv.reader(io.StringIO(first, newline=""), delimiter=sep), [])
        except csv.Error:
            own = None
        if _first_record(text, sep) != own:
            bad.append(f"text {value!r} from a first record that is not the first line's own")
    for value in shapes:
        if len(value) > 30 or not all(
            c in "a9?" or c in string.punctuation or c == " " for c in value
        ):
            bad.append(f"the shape {value!r} is not a shape")
    return bad


def test_no_refusal_prints_a_value_beyond_the_first_line_or_a_digit_or_non_ascii_text():
    gen = _Gen(20261123)
    refused = repeated = lacking = texts = 0
    for index in range(PROPERTY_FILES):
        content, _line = gen.build(index)
        try:
            load_korea.from_bytes(content)
        except LayoutError as error:
            message = str(error)
        else:
            continue
        refused += 1
        repeated += bool(REPEAT.fullmatch(message))
        lacking += message.startswith(LEAD)
        parsed = _parse_lacks(message) if message.startswith(LEAD) else None
        texts += len(parsed[0]) if parsed else 0
        problems = _problems(content, gen.numbers, message)
        assert not problems, (content, message, problems)
    # the generator reaches what it is meant to reach
    assert refused > 0.9 * PROPERTY_FILES
    assert repeated >= PROPERTY_FILES // 8
    assert lacking > 0.5 * PROPERTY_FILES
    assert texts > 500


ONE_RECORD_ENDS = ["\u2028", "\x1e", "\x0b", "\x85"]
FINAL_ENDS = ["", "\n", "\r\n", "\r"]


@pytest.mark.parametrize("final", FINAL_ENDS)
@pytest.mark.parametrize("end", ONE_RECORD_ENDS)
@pytest.mark.parametrize("sep", SEPARATORS)
def test_a_file_that_is_one_record_prints_shapes_only_with_or_without_a_final_line_end(
    sep, end, final
):
    # the lines end with a character the csv module does not read as a line end, so the whole
    # file is one record, the first "line" of which holds the later "lines" as fields; a real
    # line end at its very end makes no second record
    content = (
        end.join(
            [
                sep.join(["a", "b", "c"]),
                sep.join([LATER_ONE, "GreenMarker", LATER_TWO]),
                LATER_THREE,
            ]
        )
        + final
    ).encode()
    message = _refused(content)
    _absent(message)
    assert "GreenMarker" not in message
    assert not re.search(r'(?<!shape )"[abc]"', message)  # no field of it as text
    assert "shape " in message


@pytest.mark.parametrize("final", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("sep", SEPARATORS)
def test_a_one_line_file_with_a_final_line_end_prints_shapes_only(sep, final):
    content = (sep.join(["alpha", "beta", LATER_ONE, LATER_ONE]) + final).encode()
    message = _refused(content)
    _absent(message)
    assert '"alpha"' not in message and '"beta"' not in message


@pytest.mark.parametrize("sep", SEPARATORS)
def test_a_one_line_file_without_a_line_end_prints_shapes_only(sep):
    content = sep.join(["alpha", "beta", LATER_ONE, LATER_ONE]).encode()
    message = _refused(content)
    _absent(message)
    assert '"alpha"' not in message and '"beta"' not in message


@pytest.mark.parametrize("final", FINAL_ENDS)
@pytest.mark.parametrize("end", ONE_RECORD_ENDS)
def test_a_header_with_five_names_and_a_repeated_field_in_a_one_record_file(end, final):
    header = ",".join(NAMES + [LATER_ONE, LATER_ONE])
    content = (header + end + ",".join(["y"] * (len(NAMES) + 2)) + final).encode()
    message = _refused(content)
    _absent(message)


MARKERS = ("PurpleMarker", "GreenMarker", "OrangeMarker", "BlueMarker", "Zeta", "Eta")


def _markers_absent(message: str) -> None:
    for marker in MARKERS:
        assert marker not in message


@pytest.mark.parametrize("sep", SEPARATORS)
@pytest.mark.parametrize(
    "template",
    [
        # every row ends with \x1e, then a spaces-only line
        "a{s}b{s}c\x1ePurpleMarker{s}GreenMarker{s}Zeta\x1eOrangeMarker{s}BlueMarker{s}Eta\n   \n",
        # the first line is a record by the csv module's reading, a second record follows
        "a{s}b{s}c\x1ePurpleMarker{s}GreenMarker{s}Zeta\nq{s}r{s}s{s}t{s}u\n",
        # U+2028 in the first record, then a spaces-only line
        "a{s}b{s}c\u2028PurpleMarker{s}GreenMarker{s}Zeta\n   \n",
        # the same with a tab-and-space line, and with a carriage return
        "a{s}b{s}c\x85PurpleMarker{s}GreenMarker{s}Zeta\r\n \t \r\n",
        "a{s}b{s}c\x0bPurpleMarker{s}GreenMarker{s}Zeta\rq{s}r\r",
        "a{s}b{s}c\x1cPurpleMarker{s}GreenMarker{s}Zeta\x1dOrangeMarker{s}BlueMarker{s}Eta",
        "a{s}b{s}c\x0cPurpleMarker{s}GreenMarker{s}Zeta\u2029OrangeMarker{s}BlueMarker{s}Eta\n",
    ],
)
def test_a_boundary_of_splitlines_inside_the_first_record_makes_every_field_a_shape(sep, template):
    message = _refused(template.format(s=sep).encode())
    _markers_absent(message)
    header_part = message.split("; second record")[0]
    assert not re.search(r'(?<!shape )"[abc]"', header_part)  # not even the first line's fields
    assert "shape " in message


@pytest.mark.parametrize("sep", SEPARATORS)
def test_a_spaces_only_second_line_is_not_a_second_record(sep):
    content = f"alpha{sep}beta\n   \n\t\n".encode()
    message = _refused(content)
    assert '"alpha"' not in message and '"beta"' not in message


@pytest.mark.parametrize("sep", SEPARATORS)
def test_a_header_above_an_ordinary_second_line_is_still_text_for_its_own_fields(sep):
    content = f"alpha{sep}beta\nQuokkaLineTwo{sep}x\n".encode()
    assert '"alpha", "beta"' in _refused(content)


def test_a_field_is_text_only_when_printable_ascii_with_no_digit():
    assert load_korea.first_line_field("name x~") == '"name x~"'
    assert load_korea.first_line_field("a\x0bb") == 'shape "a?a"'
    assert load_korea.first_line_field("a\x1eb") == 'shape "a?a"'
    assert load_korea.first_line_field("a\x7fb") == 'shape "a?a"'
    assert load_korea.first_line_field("a\u2028b") == 'shape "a?a"'
    assert load_korea.first_line_field("a1") == 'shape "a9"'
    assert load_korea.first_line_field("\t") == 'shape "?"'
