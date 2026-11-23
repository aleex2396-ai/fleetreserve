"""The school-term columns (`school_inputs`) on made-up tables and on the committed one, each
with a case read by hand, and the rule that a source counts only from the day after it was
published."""

import csv
import math
from pathlib import Path

import pandas as pd
import pytest

from fleetreserve import context_places, hazard_inputs, school_inputs

TOP = Path(__file__).resolve().parents[2]
HEAD = "calendar,kind,first,last,source\n"
# bergen-2019-20.html is taken as published on 18 Aug 2019 (`SOURCES`)
BERGEN = "bergen-2019-20.html"


def frame(site, *days):
    return pd.DataFrame({"site": [site] * len(days), "date": list(days)})


def terms(*lines):
    return school_inputs.read_terms(HEAD + "".join(f"{line}\n" for line in lines))


def test_a_day_off_a_school_day_and_a_day_outside_by_hand():
    rows = terms(
        f"bergen,cover,2019-09-02,2019-09-06,{BERGEN}",
        f"bergen,off,2019-09-04,2019-09-05,{BERGEN}",
    )
    got = school_inputs.school_columns(
        frame("BER", "2019-09-03", "2019-09-04", "2019-09-05", "2019-09-06", "2019-09-07"), rows
    )
    values = got["school_holiday"].tolist()
    assert values[:4] == [0.0, 1.0, 1.0, 0.0]
    assert math.isnan(values[4])


def test_a_source_counts_only_from_the_day_after_its_publication():
    rows = terms(
        f"bergen,cover,2019-08-16,2019-08-20,{BERGEN}",
        f"bergen,off,2019-08-16,2019-08-20,{BERGEN}",
    )
    got = school_inputs.school_columns(frame("BER", "2019-08-18", "2019-08-19"), rows)
    assert math.isnan(got["school_holiday"].iloc[0])
    assert got["school_holiday"].iloc[1] == 1.0


def test_the_latest_version_published_before_the_day_decides():
    # the committed Baerum 2020-21: first version (10 Sep 2020) has 6 Apr 2021 and 14 May 2021
    # off, the revision adopted on 8 Dec 2020 has both as school days; autumn break 28 Sep to
    # 2 Oct 2020 in both
    rows = school_inputs.load_terms()
    got = school_inputs.school_columns(
        frame("BAR", "2020-09-10", "2020-10-01", "2021-04-06", "2021-05-14", "2021-05-13"), rows
    )
    values = got["school_holiday"].tolist()
    assert math.isnan(values[0])
    assert values[1:] == [1.0, 0.0, 0.0, 1.0]
    first_only = tuple(r for r in rows if "revised" not in r.source)
    again = school_inputs.school_columns(frame("BAR", "2021-04-06", "2021-05-14"), first_only)
    assert again["school_holiday"].tolist() == [1.0, 1.0]


def test_committed_table_by_hand_utrecht_and_jeffco():
    rows = school_inputs.load_terms()
    # region midden: herfst 16 to 24 Oct 2021, summer 18 Jul to 30 Aug 2020
    got = school_inputs.school_columns(
        frame("asr", "2021-10-18", "2021-10-25", "2020-08-01", "2020-08-31"), rows
    )
    assert got["school_holiday"].tolist() == [1.0, 0.0, 1.0, 0.0]
    # Jeffco 2020-21 is published on 3 Dec 2020 (its file's own day; printed 13 Nov 2020): its
    # Thanksgiving week and 3 Dec are not known, 4 Dec (a school day) and the winter break from
    # 23 Dec are
    days = ("2020-11-24", "2020-12-03", "2020-12-04", "2020-12-23")
    jeffco = school_inputs.school_columns(frame("nrel", *days), rows)["school_holiday"]
    assert math.isnan(jeffco.iloc[0]) and math.isnan(jeffco.iloc[1])
    assert jeffco.iloc[2:].tolist() == [0.0, 1.0]


def test_korea_has_none_and_academic_break_is_caltech_only():
    rows = school_inputs.load_terms()
    got = school_inputs.school_columns(frame("korea-work", "2022-03-02"), rows)
    assert got.isna().all(axis=None)
    caltech = school_inputs.school_columns(frame("caltech", "2018-12-20", "2019-01-08"), rows)
    # winter recess 15 Dec 2018 to 6 Jan 2019; PUSD's winter break 21 Dec to 4 Jan
    assert caltech["academic_break"].tolist() == [1.0, 0.0]
    assert caltech["school_holiday"].tolist() == [0.0, 0.0]
    jpl = school_inputs.school_columns(frame("jpl", "2019-12-23"), rows)
    assert math.isnan(jpl["academic_break"].iloc[0]) and jpl["school_holiday"].iloc[0] == 1.0
    with pytest.raises(KeyError):
        school_inputs.school_columns(frame("nowhere", "2019-12-23"), rows)


@pytest.mark.parametrize(
    "lines",
    [
        (f"bergen,holiday,2019-09-02,2019-09-06,{BERGEN}",),
        (f"bergen,cover,2019-09-06,2019-09-02,{BERGEN}",),
        ("bergen,cover,2019-09-02,2019-09-06,bergen-2018-19.html",),
        (
            f"bergen,cover,2019-09-02,2019-09-06,{BERGEN}",
            f"bergen,off,2019-09-06,2019-09-09,{BERGEN}",
        ),
        (
            f"bergen,cover,2019-09-02,2019-09-06,{BERGEN}",
            "bergen,off,2019-09-04,2019-09-04,bodo-2019-20.pdf",
        ),
        (
            f"bergen,cover,2019-09-02,2019-09-06,{BERGEN}",
            f"bergen,cover,2019-09-06,2019-09-09,{BERGEN}",
        ),
        (f"bergen,cover,2019-9-2,2019-09-06,{BERGEN}",),
    ],
)
def test_malformed_tables_are_refused(lines):
    with pytest.raises(ValueError):
        terms(*lines)


def test_a_summer_row_is_published_with_the_later_of_its_two_files():
    assert school_inputs.published("pusd-2019-20.pdf+pusd-2020-21.pdf").isoformat() == "2020-08-12"


def test_no_source_is_taken_as_published_before_a_date_recorded_for_it():
    # reading S2: the latest date a source shows, a PDF's own creation and modification days
    # included; every PDF source has its days recorded
    pdfs = {name for name in school_inputs.SOURCES if name.endswith(".pdf")}
    assert set(school_inputs.FILE_DATES) == pdfs
    early = {
        name: (source.published.isoformat(), max(school_inputs.FILE_DATES[name]).isoformat())
        for name, source in school_inputs.SOURCES.items()
        if name in pdfs and source.published < max(school_inputs.FILE_DATES[name])
    }
    assert early == {}
    assert all(school_inputs.FILE_DATES[name] for name in pdfs)


def test_committed_sources_match_the_download_table():
    with (TOP / "tools" / "context_sources.csv").open(newline="") as handle:
        listed = {r["name"] for r in csv.DictReader(handle) if r["set"] == school_inputs.SCHOOL_SET}
    assert listed == set(school_inputs.SOURCES)
    rows = school_inputs.load_terms()
    used = {part for r in rows for part in r.source.split("+")}
    assert used == set(school_inputs.SOURCES)
    calendars = {r.calendar for r in rows}
    for column in school_inputs.COLUMNS:
        mapping = school_inputs.SITE_CALENDARS[column]
        assert set(context_places.SITE_PLACES) <= set(mapping)
        assert {c for c in mapping.values() if c is not None} <= calendars


def test_school_columns_through_the_assembly_need_the_rows():
    cars = pd.DataFrame(
        {
            "unit": ["u"],
            "site": ["BER"],
            "date": ["2019-10-08"],
            "session_id": ["s"],
            "tau": ["2019-10-08T08:00:00+02:00"],
        }
    )
    table = cars[["unit", "site", "date", "session_id"]]
    with pytest.raises(ValueError):
        hazard_inputs.with_inputs(table, cars, ("school_holiday",))
    sources = hazard_inputs.ContextSources(school=school_inputs.load_terms())
    got = hazard_inputs.with_inputs(table, cars, ("school_holiday",), sources)
    # Bergen's autumn break 7 to 11 Oct 2019
    assert got["school_holiday"].tolist() == [1.0]
