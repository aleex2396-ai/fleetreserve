"""The a.s.r. loader's first parsing fix, on made-up bytes: wider forms of the five column names
and refusals that state a layout fact and never a value.

Wider forms and refusals (readings A1, A13 and A14): the separators tab and "|" after "," and
";"; header names compared after removing a byte order mark, surrounding spaces and surrounding
double quotes, and case-folded; no other name is accepted. A refusal for a header that lacks a
name lists, for each separator tried, the header's field names as found (cut at 40 characters,
at most 40 of them, their count given) and, for the separator with the most header fields, the
shape of each field of the first data record (digit 9, letter a, ASCII punctuation and space
kept, any other character ?, cut at 30 characters). A field that cannot be read as a time or a
number is refused with its column, its shape and the number of records that fail, never its
value. Every made-up value below that must not appear in a message is distinctive, and the tests
assert its absence.
"""

import csv
import io
import json
import random
import re
import string

import pandas as pd
import pytest
from fake_heldout import ASR_HEADER, asr_csv, asr_row

from fleetreserve import load_asr
from fleetreserve.sessions import LayoutError

ROWS = [
    asr_row("EV001", "2022-08-31T08:00:00", "2022-08-31T17:00:00", 20.5, evse="E1"),
    asr_row("EV002", "2023-03-02T08:30:00", "2023-03-02T16:00:00", 15.0, evse="E2", channel="2"),
]


@pytest.fixture(autouse=True)
def _rows_not_units(monkeypatch):
    monkeypatch.setattr(load_asr, "require_periods", lambda sessions, dropped: None)


def _same_table(content: bytes, plain: bytes = asr_csv(ROWS)):
    got = load_asr.from_bytes(content)
    expected = load_asr.from_bytes(plain)
    pd.testing.assert_frame_equal(got[0], expected[0])
    pd.testing.assert_frame_equal(got[1], expected[1])


@pytest.mark.parametrize("sep", ["\t", "|"])
def test_a_tab_or_bar_separated_file_gives_the_same_table(sep):
    _same_table(asr_csv(ROWS, sep=sep))


def test_the_comma_is_tried_before_the_others_and_a_semicolon_file_is_still_read():
    _same_table(asr_csv(ROWS, sep=";"))


def test_a_tab_file_keeps_a_decimal_point_and_a_comma_in_a_field_is_not_a_decimal_comma():
    rows = [asr_row("EV003", "2022-05-06T08:00:00", "2022-05-06T17:00:00", "12,5")]
    with pytest.raises(LayoutError, match="total_energy: 1 values are not numbers"):
        load_asr.from_bytes(asr_csv(rows, sep="\t"))


def _header_cased(content: bytes, change) -> bytes:
    header, rest = content.split(b"\n", 1)
    names = header.decode().split(",")
    return (",".join(change(name) for name in names)).encode() + b"\n" + rest


def test_a_header_in_other_letter_case_gives_the_same_table():
    upper = _header_cased(asr_csv(ROWS), str.upper)
    assert b"EV_ID_X" in upper
    _same_table(upper)
    lower = _header_cased(asr_csv(ROWS), str.lower)
    assert b"ev_id_x" in lower
    _same_table(lower)


def test_a_quoted_header_gives_the_same_table():
    quoted = _header_cased(asr_csv(ROWS), lambda name: f'"{name}"')
    assert b'"EV_id_x"' in quoted
    _same_table(quoted)
    both = _header_cased(asr_csv(ROWS), lambda name: f'" {name.upper()} "')
    _same_table(both)


def test_a_quoted_upper_case_tab_header_with_a_byte_order_mark_gives_the_same_table():
    plain = asr_csv(ROWS, sep="\t")
    header, rest = plain.split(b"\n", 1)
    header = b"\t".join(b'"' + name.upper() + b'"' for name in header.split(b"\t"))
    _same_table(b"\xef\xbb\xbf" + header + b"\n" + rest)


def test_two_header_names_that_are_one_name_in_other_case_are_refused():
    rows = [{**row, "EVSE_ID": "X"} for row in ROWS]
    content = asr_csv(rows, header=(*ASR_HEADER, "EVSE_ID"))
    with pytest.raises(LayoutError, match="names columns more than once: 1 repeated name\\(s\\)"):
        load_asr.from_bytes(content)


@pytest.mark.parametrize(
    "name, alias",
    [
        (b"EV_id_x", b"ev_id"),
        (b"EV_id_x", b"EVID_x"),
        (b"start_datetime", b"start_date"),
        (b"end_datetime", b"end_time"),
        (b"total_energy", b"total_energy_kwh"),
        (b"evse_id", b"evse"),
        (b"evse_id", b"evse id"),
    ],
)
def test_another_name_is_never_accepted(name, alias):
    """No alias and no fuzzy match: a name that is not one of the five, in any case, is absent."""
    content = asr_csv(ROWS).replace(name, alias, 1)
    assert alias in content
    with pytest.raises(LayoutError, match="a.s.r. file lacks columns"):
        load_asr.from_bytes(content)


SECRET_USER = "EVSECRET77"
SECRET_ENERGY = "31415.926"
SECRET_STATION = "ZZ-4242-Q"


def _lacking(name):
    header = tuple(column for column in ASR_HEADER if column != name)
    rows = [
        asr_row(
            SECRET_USER,
            "2022-08-31T08:00:00",
            "2022-08-31T17:00:00",
            SECRET_ENERGY,
            evse=SECRET_STATION,
        )
    ]
    return asr_csv(rows, header=header)


def test_a_header_lacking_a_column_names_the_fields_found_and_the_record_shapes_only():
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(_lacking("end_datetime"))
    message = str(refused.value)
    assert message.startswith(
        "a.s.r. file lacks columns: no separator gives "
        "EV_id_x, start_datetime, end_datetime, total_energy, evse_id"
    )
    # the header's fields as found, with their count, for the separators tried
    assert "comma: 13 fields" in message
    assert '"EV_id_x"' in message and '"channel"' in message and '"ownership"' in message
    assert "semicolon: 1 fields" in message
    assert "tab: 1 fields" in message
    assert "bar: 1 fields" in message
    # the shapes of the first data record, under the separator with the most header fields
    assert "first data record under comma (13 fields)" in message
    assert '"aaaaaaaa99"' in message  # the made-up user, as a shape
    assert '"aa-9999-a"' in message  # the made-up station
    assert '"9999-99-99a99:99:99"' in message
    assert '"99999.999"' in message
    # no value of the file
    for value in (SECRET_USER, SECRET_ENERGY, SECRET_STATION, "2022-08-31", "17:00:00"):
        assert value not in message


def test_the_refusal_cuts_names_and_shapes_and_gives_the_counts():
    names = [f"column_{chr(97 + n // 26)}{chr(97 + n % 26)}_" + "x" * 50 for n in range(45)]
    header = ",".join(names)
    record = ",".join(["Ab1-" + "7" * 60] * 45)
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes((header + "\n" + record + "\n").encode())
    message = str(refused.value)
    assert "comma: 45 fields" in message
    assert json_name(names[0][:40]) in message and json_name(names[0]) not in message
    assert json_name(names[39][:40]) in message and "column_bo_" not in message
    assert "aa9-" + "9" * 26 in message and "9" * 31 not in message


def json_name(name: str) -> str:
    return json.dumps(name)


def test_a_header_name_with_a_digit_is_given_as_its_shape_and_the_other_names_as_text():
    header = (*ASR_HEADER[:2], "load_2022_kwh", *ASR_HEADER[3:])
    rows = [{**row, "load_2022_kwh": "x"} for row in ROWS]
    content = asr_csv(rows, header=header)  # end_datetime is no longer named
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(content)
    message = str(refused.value)
    assert 'shape "aaaa_9999_aaa"' in message and "load_2022_kwh" not in message
    assert '"EV_id_x", "start_datetime", shape "aaaa_9999_aaa", "total_energy"' in message


def test_a_real_looking_header_is_given_by_its_names():
    header = ("EV_id_x", "start_datetime", "total_energy", "rail", "evse_id", "channel")
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(asr_csv(ROWS, header=header))
    message = str(refused.value)
    assert 'comma: 6 fields ["EV_id_x", "start_datetime", "total_energy", "rail", "evse_id"' in (
        message
    )
    assert "the first line has the shape of a data record" not in message


HEADERLESS_USER = "EVSECRET77"


def test_a_file_with_no_header_never_prints_its_first_line_as_text():
    """The first line of the real file has never been seen, so it may be data. (Not red at the
    base: the old message printed nothing of the first line, so this is asserted on the new code
    only.) Every field here holds a digit, so every one is printed as its shape."""
    first = asr_row(HEADERLESS_USER, "2022-08-31T08:00:00", "2022-08-31T17:00:00", 31415.926)
    first.update(rail="R9", evse="ZZ-4242-Q", channel="1")
    content = asr_csv([first, *ROWS])
    lines = content.split(b"\n")
    content = b"\n".join(lines[1:])  # drop the generated header: the first line is a record
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(content)
    message = str(refused.value)
    assert "lacks columns" in message
    for value in (HEADERLESS_USER, "31415", "ZZ-4242-Q", "2022-08-31", "08:00:00", "17:00:00"):
        assert value not in message
    assert 'shape "aaaaaaaa99"' in message


def test_a_first_line_with_the_shape_of_the_second_record_is_said_to_be_data():
    one = asr_row("EVAAA11", "2022-08-31T08:00:00", "2022-08-31T17:00:00", 20.5, evse="E1")
    two = asr_row("EVBBB22", "2022-09-01T09:00:00", "2022-09-01T18:00:00", 11.2, evse="E2")
    content = b"\n".join(asr_csv([one, two]).split(b"\n")[1:])
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(content)
    message = str(refused.value)
    assert "comma: 14 fields, the first line has the shape of a data record [shape " in message
    for value in ("EVAAA11", "EVBBB22", "2022-08-31", "20.5"):
        assert value not in message


def test_the_shape_of_a_field_keeps_ascii_punctuation_and_spaces_and_hides_everything_else():
    assert load_asr.shape("Ab1 -_.:/é\t\n中") == "aa9 -_.:/????"
    assert load_asr.shape("") == ""
    assert load_asr.shape("x" * 31) == "a" * 30


def test_a_file_that_is_not_a_table_is_refused_with_layout_only():
    content = b"just one line of words with no table\nnext SECRETWORD line\n"
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(content)
    assert "SECRETWORD" not in str(refused.value)
    assert "lacks columns" in str(refused.value)


@pytest.mark.parametrize("column", ["start_datetime", "end_datetime"])
def test_a_bad_time_is_refused_with_its_shape_and_never_its_value(column):
    bad = "2022-05-06 08:30:00.123456"
    rows = [
        *ROWS,
        asr_row("EV020", "2022-05-06T07:00:00", "2022-05-06T17:00:00", 5.0),
    ]
    rows[2][column] = bad
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(asr_csv(rows))
    message = str(refused.value)
    assert f"a.s.r. {column}: 1 times cannot be read" in message
    assert '"9999-99-99 99:99:99.999999"' in message
    for part in ("2022-05-06", "08:30:00", "123456"):
        assert part not in message


def test_the_count_and_each_distinct_shape_of_bad_times_are_given():
    rows = [
        asr_row("EV020", "2022-05-06T07:00:00", "2022-05-06T17:00:00Z", 5.0),
        asr_row("EV021", "2022-05-07T07:00:00", "2022-05-07T17:00:00Z", 5.0),
        asr_row("EV022", "2022-05-08T07:00:00", "2022-05-08 17:00", 5.0),
    ]
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(asr_csv(rows))
    message = str(refused.value)
    assert "end_datetime: 3 times cannot be read" in message
    assert '"9999-99-99a99:99:99a" x 2' in message
    assert '"9999-99-99 99:99" x 1' in message
    assert "2022-05-0" not in message


def test_a_field_that_is_not_a_number_is_refused_with_its_shape_and_never_its_value():
    rows = [
        asr_row("EV020", "2022-05-06T08:00:00", "2022-05-06T17:00:00", "about 31 kWh"),
        asr_row("EV021", "2022-05-07T08:00:00", "2022-05-07T17:00:00", "12 kWh"),
    ]
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(asr_csv(rows))
    message = str(refused.value)
    assert "total_energy: 2 values are not numbers" in message
    assert '"aaaaa 99 aaa" x 1' in message and '"99 aaa" x 1' in message
    assert "about" not in message and "kWh" not in message


def test_a_missing_plug_in_time_is_counted_without_a_value():
    rows = [
        *ROWS,
        asr_row("EV020", None, "2022-05-06T17:00:00", 5.0),
        asr_row("EV021", None, "2022-05-07T17:00:00", 5.0),
    ]
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(asr_csv(rows))
    message = str(refused.value)
    assert "no plug-in time (start_datetime)" in message and "2 rows" in message
    assert "2022-05-06" not in message


# The second correction (reading A14): a first-line field is printed as text only when it is ASCII
# and holds no ASCII digit and no line break. A quote that opens a field in line 1 and is not
# closed there makes that field run over the lines that follow; its text must never be printed.

LATER_ONE = "QuokkaLineTwo"
LATER_TWO = "WombatLineTwo"
LATER_THREE = "NumbatLineThree"
LATER = (LATER_ONE, LATER_TWO, LATER_THREE, "Quokka", "Wombat", "Numbat")
SEPARATORS = [",", ";", "\t", "|"]


def _refused(content: bytes) -> str:
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(content)
    return str(refused.value)


def _absent(message: str) -> None:
    for value in LATER:
        assert value not in message


@pytest.mark.parametrize("sep", SEPARATORS)
def test_an_open_quote_in_line_one_never_prints_the_lines_that_follow(sep):
    content = f'a{sep}"b\n{LATER_ONE}{sep}{LATER_TWO}\n{LATER_THREE}{sep}yyy\n'.encode()
    message = _refused(content)
    assert "lacks columns" in message
    _absent(message)
    assert 'shape "a?aaaaaaaaaaaaa' in message  # the runaway field "b" + line break + ..., a shape


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
def test_a_record_with_a_quoted_separator_and_an_open_quote_is_all_shapes(sep):
    # the open quote after the quoted-separator field runs over the lines that follow, so the
    # record is not the first physical line alone: every field of it is a shape
    content = f'"p{sep}q"{sep}r{sep}"s\n{LATER_ONE}{sep}{LATER_TWO}\n{LATER_THREE}\n'.encode()
    message = _refused(content)
    _absent(message)
    assert json.dumps(f"p{sep}q") not in message
    assert '"r"' not in message


@pytest.mark.parametrize("sep", SEPARATORS)
def test_blank_lines_before_the_first_record_make_it_print_as_shapes_only(sep):
    content = f"\n\r\n{LATER_ONE}{sep}{LATER_TWO}\n{LATER_THREE}{sep}d\n".encode()
    message = _refused(content)
    _absent(message)
    assert "shape " in message


def test_a_digit_free_ascii_first_line_without_a_line_break_is_still_printed_as_text():
    # the remaining, stated risk (reading A14): names are mostly such text
    content = b"alpha,beta,gamma\nQuokkaLineTwo,x,y\n"
    message = _refused(content)
    assert '["alpha", "beta", "gamma"]' in message
    assert LATER_ONE not in message


def test_a_field_is_text_only_when_ascii_with_no_digit_and_no_line_break():
    assert load_asr.first_line_field("name") == '"name"'
    assert load_asr.first_line_field("a b,c") == '"a b,c"'
    assert load_asr.first_line_field("ab\ncd") == 'shape "aa?aa"'
    assert load_asr.first_line_field("ab\rcd") == 'shape "aa?aa"'
    assert load_asr.first_line_field("ab\r\ncd") == 'shape "aa??aa"'
    assert load_asr.first_line_field("ab1") == 'shape "aa9"'
    assert load_asr.first_line_field("café") == 'shape "aaa?"'
    assert load_asr.first_line_field("x" * 50) == json.dumps("x" * 40)


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


@pytest.mark.parametrize("sep", SEPARATORS)
def test_a_first_line_that_is_a_complete_quoted_record_is_text_and_the_next_line_is_not(sep):
    content = f'"p{sep}q"{sep}r\n{TAIL_ONE}{sep}{TAIL_TWO}\n'.encode()
    message = _refused(content)
    if sep == "\t":  # a tab is not printable ASCII: the field that holds it is a shape
        assert 'shape "a?a"' in message
    else:
        assert json.dumps(f"p{sep}q") in message
    assert '"r"' in message
    _tails_absent(message)


def test_on_first_line_is_true_only_for_a_record_that_is_the_first_physical_line_alone():
    assert load_asr.on_first_line("ab,cd\nef,gh\n", ",", ["ab", "cd"])
    assert not load_asr.on_first_line("ab,cd", ",", ["ab", "cd"])  # one record, no line end
    assert not load_asr.on_first_line("ab,cd\n", ",", ["ab", "cd"])  # one record, a line end
    assert not load_asr.on_first_line("ab,cd\r\n\r\n", ",", ["ab", "cd"])  # blank lines only after
    assert not load_asr.on_first_line("ab,cd\u2028ef,gh", ",", ["ab", "cd\u2028ef", "gh"])
    assert load_asr.on_first_line('"a,b",c\nx\n', ",", ["a,b", "c"])
    assert not load_asr.on_first_line('"a\nb",c\nx\n', ",", ["a\nb", "c"])
    assert not load_asr.on_first_line('"a\rb",c\rx\r', ",", ["a\rb", "c"])
    assert not load_asr.on_first_line('a,"b\nc"\n', ",", ["a", "b\nc"])
    assert not load_asr.on_first_line("\nab,cd\n", ",", ["ab", "cd"])
    assert not load_asr.on_first_line("\r\nab,cd\n", ",", ["ab", "cd"])


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
            load_asr.from_bytes(body.encode())
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


# ---- the refusal for a repeated header name states a count, never a name ---------------------

NAMES = ["EV_id_x", "start_datetime", "end_datetime", "total_energy", "evse_id"]
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
    content = (header + "\n" + sep.join(["1", "2", "3", "4", "5", "6", "7"]) + "\n").encode()
    assert _refused(content) == "a.s.r. file names columns more than once: 1 repeated name(s)"


def test_two_repeated_names_of_a_header_are_counted_not_named():
    header = ",".join(NAMES + ["Chan7", "Chan7", "한국", "한국", "evse_id"])
    content = (header + "\n" + ",".join("x" * 10 for _ in range(10)) + "\n").encode()
    assert _refused(content) == "a.s.r. file names columns more than once: 3 repeated name(s)"


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
EXTRA_NAMES = ["rail", "channel"]
LEAD = "a.s.r. file lacks columns: no separator gives"
REPEAT = re.compile(r"a\.s\.r\. file names columns more than once: \d+ repeated name\(s\)")
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
            cols = cols[: rng.choice([3, 4, 5, 6])]
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
        if (
            index % 5 == 2 and index % 6 != 0 and not one_record
        ):  # rows joined by any boundary of str.splitlines
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
    if message.startswith("; the file has no data record", at):
        return texts, shapes
    tail = re.compile(
        r"; first data record under (comma|semicolon|tab|bar) \(\d+ fields\), field shapes "
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
            if row:
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
            load_asr.from_bytes(content)
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
    content = (header + end + ",".join(["y"] * 7) + final).encode()
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
    header_part = message.split("; first data record")[0]
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
    assert load_asr.first_line_field("name x~") == '"name x~"'
    assert load_asr.first_line_field("a\x0bb") == 'shape "a?a"'
    assert load_asr.first_line_field("a\x1eb") == 'shape "a?a"'
    assert load_asr.first_line_field("a\x7fb") == 'shape "a?a"'
    assert load_asr.first_line_field("a\u2028b") == 'shape "a?a"'
    assert load_asr.first_line_field("a1") == 'shape "a9"'
    assert load_asr.first_line_field("\t") == 'shape "?"'
