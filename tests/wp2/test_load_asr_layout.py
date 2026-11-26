"""The a.s.r. loader on made-up files built to the layout the layout look of 23 November showed.

The layout: a byte order mark, then 14 fields separated by semicolons with `\\r\\n` line ends and
the header `EV_id_x;start_datetime;end_datetime;total_energy;rail;evse_uid;channel;capacity_kwh;
commute_km_range_min;commute_km_range_max;EV_brand_selfreported;EV_model_selfreported;
capacity_kwh_selfreported;ownership`; times written `YYYY-MM-DD hh:mm:ss` (a space, never a
`T`), never empty; `total_energy` with a decimal point; `rail` one of 15 labels; `evse_uid`
digits only; survey columns mostly empty. Every value below is made up. The tests read such
files through `from_bytes` and check the table; they are red when the loader refuses the file
and green when it reads it (readings A1, A3, A7, A13 of the module).
"""

import pandas as pd
import pytest

from fleetreserve import load_asr
from fleetreserve.sessions import LayoutError

HEADER = (
    "EV_id_x;start_datetime;end_datetime;total_energy;rail;evse_uid;channel;capacity_kwh;"
    "commute_km_range_min;commute_km_range_max;EV_brand_selfreported;EV_model_selfreported;"
    "capacity_kwh_selfreported;ownership"
)
RAILS = [
    "B",
    "C",
    "E",
    "H",
    "J",
    "K",
    "L",
    "M",
    "carport groot",
    "carport klein",
    "mennekes",
    "shared car",
    "visitors",
    "visitors facility management",
    "wheelchair parking",
]
SURVEY_EMPTY = [""] * 7  # capacity_kwh and the six survey columns


def record(ev, start, end, energy, rail="B", uid="10234", channel="1", survey=SURVEY_EMPTY):
    """One record of 14 fields: the seven columns of a session, then `survey` (seven fields)."""
    fields = [ev, start, end, energy, rail, uid, channel, *survey]
    assert len(fields) == 14
    return ";".join(fields)


def real_file(*records: str, header: str = HEADER, bom: bool = True) -> bytes:
    """The records in the real layout: byte order mark, `\\r\\n` line ends, a line end last."""
    text = "\r\n".join([header, *records]) + "\r\n"
    return (("﻿" if bom else "") + text).encode("utf-8")


BEFORE_ROW = record("EV12", "2022-11-08 07:41:12", "2022-11-08 09:05:40", "12.5", uid="000451")
FROM_ROW = record("EV345", "2023-04-04 08:00:00", "2023-04-04 09:30:30", "20", uid="778")
EARLY = ("2022-06-01 07:00:00", "2022-06-01 09:00:00")


def test_the_real_layout_is_read():
    sessions, dropped = load_asr.from_bytes(real_file(BEFORE_ROW, FROM_ROW))
    assert dropped.empty
    assert list(sessions["session_id"]) == ["row-1", "row-2"]
    assert list(sessions["user_id"]) == ["EV12", "EV345"]
    assert list(sessions["station"]) == ["000451", "778"]
    assert [str(value) for value in sessions["plug_in"]] == [
        "2022-11-08 07:41:12+01:00",
        "2023-04-04 08:00:00+02:00",
    ]
    assert [str(value) for value in sessions["plug_out"]] == [
        "2022-11-08 09:05:40+01:00",
        "2023-04-04 09:30:30+02:00",
    ]
    assert list(sessions["energy_kwh"]) == [12.5, 20.0]
    assert list(load_asr.periods(sessions)["unit"]) == list(load_asr.UNITS)


@pytest.mark.parametrize("rail", RAILS)
def test_every_rail_value_is_read_and_kept(rail):
    row = record("EV77", *EARLY, "4.25", rail, "555")
    sessions, dropped = load_asr.from_bytes(real_file(row, BEFORE_ROW, FROM_ROW))
    assert dropped.empty
    assert len(sessions) == 3
    assert "555" in list(sessions["station"])


def test_a_file_of_all_fifteen_rails_keeps_every_row():
    rows = [record(f"EV{n}", *EARLY, "4.25", rail, str(n)) for n, rail in enumerate(RAILS, 1)]
    sessions, dropped = load_asr.from_bytes(real_file(*rows, BEFORE_ROW, FROM_ROW))
    assert dropped.empty
    assert len(sessions) == len(RAILS) + 2


def test_rows_differing_only_in_rail_are_not_duplicates():
    rows = [record("EV9", *EARLY, "4.25", rail, "555") for rail in ("visitors", "shared car")]
    sessions, dropped = load_asr.from_bytes(real_file(*rows, BEFORE_ROW, FROM_ROW))
    assert dropped.empty and len(sessions) == 4


def test_survey_columns_with_commas_and_apostrophes_are_read_as_text():
    survey = [
        "75",
        "30",
        "45",
        "Mijn merk (zie ook x)",
        "Model 3, long",
        "75 kWh '+/- 5",
        "Gekocht",
    ]
    row = record(
        "EV88", "2022-06-02 08:00:00", "2022-06-02 09:00:00", "9.5", "H", "42", "2", survey
    )
    sessions, dropped = load_asr.from_bytes(real_file(row, BEFORE_ROW, FROM_ROW))
    assert dropped.empty and len(sessions) == 3


def test_the_t_form_of_a_time_is_still_read_and_equals_the_space_form():
    spaced, _ = load_asr.from_bytes(real_file(BEFORE_ROW, FROM_ROW))
    t_row = record("EV12", "2022-11-08T07:41:12", "2022-11-08T09:05:40", "12.5", uid="000451")
    t_form, _ = load_asr.from_bytes(real_file(t_row, FROM_ROW))
    pd.testing.assert_frame_equal(spaced, t_form)


def test_a_file_without_a_byte_order_mark_or_with_line_feeds_is_read_too():
    plain = load_asr.from_bytes(real_file(BEFORE_ROW, FROM_ROW))[0]
    no_bom = load_asr.from_bytes(real_file(BEFORE_ROW, FROM_ROW, bom=False))[0]
    unix = load_asr.from_bytes(real_file(BEFORE_ROW, FROM_ROW).replace(b"\r\n", b"\n"))[0]
    pd.testing.assert_frame_equal(plain, no_bom)
    pd.testing.assert_frame_equal(plain, unix)


def test_the_station_may_still_be_named_evse_id():
    old = real_file(BEFORE_ROW, FROM_ROW, header=HEADER.replace("evse_uid", "evse_id"))
    new = real_file(BEFORE_ROW, FROM_ROW)
    pd.testing.assert_frame_equal(load_asr.from_bytes(old)[0], load_asr.from_bytes(new)[0])


def test_both_station_names_in_one_header_are_a_column_named_twice():
    header = HEADER + ";evse_id"
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(real_file(BEFORE_ROW + ";x", FROM_ROW + ";y", header=header))
    assert str(refused.value) == "a.s.r. file names columns more than once: 1 repeated name(s)"


def test_the_station_name_is_read_whatever_its_case_and_spacing():
    header = HEADER.replace("evse_uid", " EVSE_UID ")
    sessions, _ = load_asr.from_bytes(real_file(BEFORE_ROW, FROM_ROW, header=header))
    assert list(sessions["station"]) == ["000451", "778"]


@pytest.mark.parametrize("alias", ["evse", "evse uid", "evseuid", "station"])
def test_no_other_station_name_is_accepted(alias):
    data = real_file(BEFORE_ROW, FROM_ROW, header=HEADER.replace("evse_uid", alias))
    with pytest.raises(LayoutError, match="a.s.r. file lacks columns"):
        load_asr.from_bytes(data)


def test_the_spring_gap_and_autumn_hour_in_the_space_form_follow_the_same_rules():
    gap = record("EV1", "2022-03-27 02:30:00", "2022-03-27 05:00:00", "3", uid="1")
    doubled = record("EV2", "2022-10-30 02:30:00", "2022-10-30 06:00:00", "3", uid="2")
    sessions, dropped = load_asr.from_bytes(real_file(gap, doubled, BEFORE_ROW, FROM_ROW))
    assert list(dropped["reason"]) == ["local time in the daylight-saving gap or doubled hour"] * 2
    assert len(sessions) == 2


@pytest.mark.parametrize(
    "bad",
    [
        "2022-05-08 08:00",
        " 2022-05-08 08:00:00",
        "2022-05-08  08:00:00",
        "2022-5-08 08:00:00",
        "2022-05-08 08:00:00Z",
        "2022-05-08T08:00:00+01:00",
        "2022-05-08 08:00:00.5",
        "2022-13-08 08:00:00",
        "2022-05-08 08:00:00 ",
    ],
)
def test_other_forms_of_a_time_are_still_refused(bad):
    row = record("EV3", "2022-05-08 07:00:00", bad, "5")
    with pytest.raises(LayoutError, match="end_datetime: 1 times cannot be read"):
        load_asr.from_bytes(real_file(row, BEFORE_ROW, FROM_ROW))
