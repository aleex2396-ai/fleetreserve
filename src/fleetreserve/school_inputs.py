"""School-term inputs of the departure model, each known at the decision time τ (plan section 7:
"school terms (official published dates for the Netherlands, Norway and the US sites; none for
Korea, whose source needs a key)").

Sources. Every calendar used is an official document of the body that sets the dates (a
ministry, a municipality, a school district, a university registrar), listed as a row of set
`school` in `tools/context_sources.csv` with its URL, size and SHA-256 and downloaded unparsed by
`tools/fetch_data.py --context` to `data/raw/school/`. Most of the official servers no longer
hold the old years, or refuse a plain client (HTTP 403); those rows are the Internet Archive's
unmodified copy (`web.archive.org/web/<capture>id_/<official URL>`) of the official page or file.
[Reading S1: an archived copy of the official document counts as the official document.
Alternative: leave every such year out, which leaves only the Dutch, La Canada and three Jeffco
years.] No account, key or login is used; no third-party calendar site is a source.

The dates are transcribed by hand from those files into `learn/school-terms.csv` (committed; the
raw files never enter git). Each row is a calendar, a kind, a first and a last day (inclusive)
and its source file (two files joined by `+` for a summer between two school years: the earlier
year's file states its last day, the later one its first). Kinds: `cover`, the days the source
states the school calendar of (a school year from the first day it lists, pupil-free days
included, to its last school day, or a summer between two such years); `off`, days of that
span without school for pupils (breaks, holidays, pupil-free and planning days, the summer).
A day inside a `cover` row and in none of its `off` rows is a school day. Weekend days are
treated like any other day.

Known at τ (`SOURCES`): a source is taken as published on the latest date it shows for the
version used (a printed adoption, revision or update date; a page's "updated" stamp; a PDF's
creation or modification date; for a summer row, the later of its two files), and the archive's
capture date where it shows none. A row of date d reads only sources published strictly before
d, so a calendar revised during the year (Baerum 2020-21, adopted again on 8 Dec 2020) is read
in its first version before the revision and in the revised one after it: of the `cover` rows
of a calendar that hold d and were published before d, the latest published one decides.
[Reading S2: the latest date shown, not the earliest (an adoption date before a later revision
would claim the revision's dates were known earlier). Alternative: the earliest date shown.]
Every PDF source's own creation and modification days are recorded in `FILE_DATES`, read with
`pdfinfo -isodates` from the file of the committed hash, each both in the file's own time offset
and in UTC (the later of the two where they differ); a test refuses a `published` day earlier
than any of them. Six sources show a printed date before the file's own days and are taken as
published on the file's latest day: Pasadena 2019-20 (printed 30 Apr 2019, file 3 May 2019),
2020-21 (31 Jul 2020, file 12 Aug 2020) and 2021-22 (4 Dec 2020, file 13 Jan 2021); Sunnyvale
2019-20 (10 Jan 2019, file 11 Feb 2019) and 2020-21 (11 Dec 2018, file 14 Dec 2018); Jeffco
2020-21 (13 Nov 2020, file 3 Dec 2020). Whether their content changed after the printed date is
not known. The pages (`.html`) are dated by the stamp they show or their capture date only.

Columns (`COLUMNS`), per row of a cars table (its `site` and `date`):
- `school_holiday`: 1 on a day off for pupils of the public schools where the site lies, 0 on a
  school day and on a weekend day inside a term (weekends are not days off here), NaN where no
  calendar of the site covers the day by a source published before it, and for the Korean
  portfolios (no allowed source). Calendars (`SITE_CALENDARS`): caltech
  Pasadena Unified; jpl La Canada Unified; office001 Sunnyvale School District (reading S3: the
  district of the P1 point of `context_places`; alternative: Fremont Union High School District,
  whose archived calendar pages carry no date); nrel Jeffco Public Schools (Golden); the
  Norwegian sites their municipality's skolerute (Oslo; Asker, the municipality before its
  merger of 1 Jan 2020; Baerum; Bergen, a "veiledende" (recommended) calendar; Bodo; Trondheim;
  Hole, where Krokkleiva lies); asr the Dutch region "midden" (Utrecht).
- `academic_break`: at caltech only, 1 on a day of the Caltech registrar's academic calendar
  without instruction, study or examination days (a Thanksgiving or Institute holiday, the
  winter and spring recesses, the summer from the day after Commencement to the day before the
  fall term's first day of classes), 0 on the other days of a term, NaN at every other site.
  [Reading S4: a separate column, since a university's terms and a district's differ.
  Alternative: no Caltech column.]

What is in each calendar (reading S5, the same everywhere): the dates for all pupils of the
district's or municipality's elementary schools; days off for some grades only (Jeffco 2020-21
grades PK-5; Oslo days for grades 1 to 10 only are kept, as those are its elementary pupils) are
counted only where they hold for the elementary pupils; the Dutch "adviesdata" (advised dates)
for the autumn and spring breaks count as breaks, the optional second May week does not. Remote
teaching (2020-21) and the COVID-19 closures of spring 2020 are not in the calendars and not
here (OxCGRT's index and the regime flags carry them).

Not covered, as no official dated source was found (a web search on 15 Nov 2026; the days are
NaN): Oslo 2017-18 and 2018-19 (the 2019 list of the city page is known only from a capture of
2020); Bodo and Trondheim 2018-19 (Trondheim's only source of that year is a calendar feed whose
bytes change on every download, so its hash cannot be committed); Asker after its merger;
Hole before November 2020; La Canada and Sunnyvale 2018-19; Caltech 2021-22 (its only version
is dated 3 Jun 2022). Days are also NaN before a source's publication date: Jeffco 2016-17
before 6 Sep 2016, Jeffco 2020-21 before 4 Dec 2020, Pasadena's summer of 2020 (its row read
from the 2019-20 and 2020-21 files) up to 12 Aug 2020, Caltech 2019-20 before 15 May 2020 and
2020-21 before 5 Feb 2021, Baerum 2020-21 before 10 Sep 2020, Sunnyvale 2021-22 before 27 Aug
2021. The transcription is by hand and is checked here only for its form (`read_terms`); its
dates were read from the files with `pdftotext` or as rendered pages (two scanned Pasadena
years, the colour-coded Jeffco calendars), so a misread day is possible.

`read_terms(text)` reads the table and refuses (`ValueError`) a row of another kind, a first day
after its last, a source not in `SOURCES`, an `off` row outside every `cover` row of its own
source, and two overlapping `cover` rows of one calendar published on the same day.
`school_columns(frame, terms)` gives the columns; a site missing from `SITE_CALENDARS` raises
`KeyError`.
"""

import csv
import io
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from fleetreserve.context_places import KOREA_SITES
from fleetreserve.sessions import TOP

COLUMNS = ("school_holiday", "academic_break")
TERMS_FILE = TOP / "learn" / "school-terms.csv"
HEADER = ("calendar", "kind", "first", "last", "source")
KINDS = ("cover", "off")
SCHOOL_SET = "school"


@dataclass(frozen=True)
class Source:
    published: date
    shown: str


SOURCES: dict[str, Source] = {
    "nl-stcrt-2018-11687.pdf": Source(date(2018, 3, 2), "Staatscourant 2018 nr. 11687, issue date"),
    "nl-stcrt-2021-25317.pdf": Source(
        date(2021, 5, 21), "Staatscourant 2021 nr. 25317, issue date"
    ),
    "oslo-2019-20.html": Source(
        date(2018, 12, 13), "page metadata, modified (capture 19 Jul 2019)"
    ),
    "oslo-2020-21.html": Source(date(2020, 7, 14), "page metadata, modified (capture 6 Aug 2020)"),
    "asker-2018-19.html": Source(date(2018, 11, 1), "'Sist oppdatert 01.11.2018'"),
    "asker-2019-20.html": Source(date(2019, 7, 12), "'Sist oppdatert 12.07.2019'"),
    "baerum-2018-19.pdf": Source(date(2018, 6, 20), "PDF modified (file name 'vedtatt')"),
    "baerum-2019-20.pdf": Source(date(2018, 9, 18), "PDF created and modified"),
    "baerum-2020-21.html": Source(date(2020, 9, 10), "'Oppdatert 10. sep. 2020'"),
    "baerum-2020-21-revised.html": Source(
        date(2020, 12, 8),
        "'vedtatt av Hovedutvalg for barn og unge 08.12.20' (capture 23 Jan 2021)",
    ),
    "baerum-2021-22.pdf": Source(date(2020, 12, 16), "PDF created; adopted 08.12.20 as printed"),
    "bergen-2019-20.html": Source(date(2019, 8, 18), "no date shown: the capture date"),
    "bodo-2019-20.pdf": Source(date(2019, 5, 24), "PDF created and modified"),
    "trondheim-2019-20.pdf": Source(date(2019, 1, 21), "PDF created (footer 'juni 2018')"),
    "hole-2020-21.html": Source(date(2020, 11, 5), "'Oppdatert 05. november 2020'"),
    "pusd-2017-18.pdf": Source(date(2017, 3, 28), "PDF modified (hand-written 2/28/15)"),
    "pusd-2018-19.pdf": Source(date(2018, 4, 12), "PDF modified (hand-written 6/6/16)"),
    "pusd-2019-20.pdf": Source(date(2019, 5, 3), "PDF modified (printed 'Revised 4/30/2019')"),
    "pusd-2020-21.pdf": Source(
        date(2020, 8, 12), "PDF modified (printed '7/31/2020', file name 'Revised')"
    ),
    "pusd-2021-22.pdf": Source(date(2021, 1, 13), "PDF created and modified (printed '12/4/2020')"),
    "lcusd-2019-20.pdf": Source(
        date(2019, 8, 7), "upload date in the file's address (printed 4/29/2019)"
    ),
    "lcusd-2020-21.pdf": Source(
        date(2019, 8, 6), "upload date in the file's address (printed 4/29/19)"
    ),
    "lcusd-2021-22.pdf": Source(date(2019, 8, 6), "PDF created; upload date in the file's address"),
    "ssd-2019-20.pdf": Source(
        date(2019, 2, 11), "PDF modified (printed 'Updated 01/10/19', approved 12/06/18)"
    ),
    "ssd-2020-21.pdf": Source(
        date(2018, 12, 14), "PDF created and modified (printed 'Board Approved 12/11/18')"
    ),
    "ssd-2021-22.pdf": Source(date(2021, 8, 27), "file name '8.27.21' (approved 01.07.21)"),
    "jeffco-2016-17.pdf": Source(date(2016, 9, 6), "PDF created and modified"),
    "jeffco-2017-18.pdf": Source(date(2017, 4, 28), "PDF created and modified"),
    "jeffco-2018-19.pdf": Source(date(2017, 10, 30), "the capture date (PDF created 16 Sep 2015)"),
    "jeffco-2019-20.pdf": Source(date(2019, 8, 1), "PDF created and modified"),
    "jeffco-2020-21.pdf": Source(
        date(2020, 12, 3), "PDF created and modified (printed 'Updated 11/13/20')"
    ),
    "jeffco-2021-22.pdf": Source(date(2021, 7, 2), "PDF created and modified"),
    "caltech-2017-18.pdf": Source(date(2017, 3, 1), "printed 'Modified: 2-15-17 & 3-1-17'"),
    "caltech-2018-19.pdf": Source(date(2018, 3, 5), "printed 'Revised: 3-5-18'"),
    "caltech-2019-20.pdf": Source(
        date(2020, 5, 15), "file name 'Updated_5-15-20' (printed 3-18-20)"
    ),
    "caltech-2020-21.pdf": Source(date(2021, 2, 5), "printed 'Modified: ... 02-05-21'"),
}
# Every PDF source's own creation and modification days, as `pdfinfo -isodates` reads them from
# the file of the committed hash, each both in the file's own time offset and in UTC (module
# text); a source's `published` day is never earlier than any of them.
FILE_DATES: dict[str, tuple[date, ...]] = {
    "nl-stcrt-2018-11687.pdf": (date(2018, 2, 27),),
    "nl-stcrt-2021-25317.pdf": (date(2021, 5, 18),),
    "baerum-2018-19.pdf": (date(2018, 6, 19), date(2018, 6, 20)),
    "baerum-2019-20.pdf": (date(2018, 9, 18),),
    "baerum-2021-22.pdf": (date(2020, 12, 16),),
    "bodo-2019-20.pdf": (date(2019, 5, 24),),
    "trondheim-2019-20.pdf": (date(2019, 1, 21),),
    "pusd-2017-18.pdf": (date(2015, 10, 20), date(2015, 10, 21), date(2017, 3, 28)),
    "pusd-2018-19.pdf": (date(2017, 10, 30), date(2018, 4, 12)),
    "pusd-2019-20.pdf": (date(2019, 4, 12), date(2019, 4, 13), date(2019, 5, 3)),
    "pusd-2020-21.pdf": (date(2020, 8, 11), date(2020, 8, 12)),
    "pusd-2021-22.pdf": (date(2021, 1, 13),),
    "lcusd-2019-20.pdf": (date(2019, 8, 7),),
    "lcusd-2020-21.pdf": (date(2019, 8, 6),),
    "lcusd-2021-22.pdf": (date(2019, 8, 6),),
    "ssd-2019-20.pdf": (date(2019, 2, 7), date(2019, 2, 11)),
    "ssd-2020-21.pdf": (date(2018, 12, 14),),
    "ssd-2021-22.pdf": (date(2021, 5, 12), date(2021, 8, 27)),
    "jeffco-2016-17.pdf": (date(2016, 9, 6),),
    "jeffco-2017-18.pdf": (date(2017, 4, 28),),
    "jeffco-2018-19.pdf": (date(2015, 9, 16),),
    "jeffco-2019-20.pdf": (date(2019, 8, 1),),
    "jeffco-2020-21.pdf": (date(2020, 12, 3),),
    "jeffco-2021-22.pdf": (date(2021, 7, 2),),
    "caltech-2017-18.pdf": (date(2017, 3, 1),),
    "caltech-2018-19.pdf": (date(2018, 3, 5),),
    "caltech-2019-20.pdf": (date(2020, 5, 15),),
    "caltech-2020-21.pdf": (date(2021, 2, 5),),
}

_NORWAY = {
    "ASK": "asker",
    **dict.fromkeys(("BAR", "BAR_2"), "baerum"),
    "BER": "bergen",
    "BOD": "bodo",
    "KRO": "hole",
    **dict.fromkeys(("OSL_1", "OSL_2", "OSL_S", "OSL_T"), "oslo"),
    **dict.fromkeys(("TRO", "TRO_R"), "trondheim"),
}
SITE_CALENDARS: dict[str, dict[str, str | None]] = {
    "school_holiday": {
        "caltech": "pusd",
        "jpl": "lcusd",
        "office001": "ssd",
        "nrel": "jeffco",
        **_NORWAY,
        "asr": "nl-midden",
        **dict.fromkeys(KOREA_SITES, None),
    },
    "academic_break": {
        "caltech": "caltech-academic",
        **dict.fromkeys(("jpl", "office001", "nrel", *_NORWAY, "asr", *KOREA_SITES), None),
    },
}


@dataclass(frozen=True)
class Row:
    calendar: str
    kind: str
    first: date
    last: date
    source: str
    published: date


def published(source: str) -> date:
    """When a row's source is taken as published (module text); `ValueError` for a file not in
    `SOURCES`."""
    parts = source.split("+")
    unknown = [p for p in parts if p not in SOURCES]
    if unknown or not parts:
        raise ValueError(f"school terms: no such source {unknown}")
    return max(SOURCES[p].published for p in parts)


def read_terms(text: str) -> tuple[Row, ...]:
    """The rows of the terms table (module text)."""
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if tuple(header or ()) != HEADER:
        raise ValueError(f"school terms: the header is {','.join(HEADER)}")
    rows = []
    for number, fields in enumerate(reader, start=2):
        if len(fields) != len(HEADER):
            raise ValueError(f"school terms line {number}: {len(HEADER)} fields are needed")
        calendar, kind, first, last, source = fields
        if kind not in KINDS:
            raise ValueError(f"school terms line {number}: the kind is one of {KINDS}")
        try:
            start, end = date.fromisoformat(first), date.fromisoformat(last)
        except ValueError:
            raise ValueError(f"school terms line {number}: a day is written YYYY-MM-DD") from None
        if start > end:
            raise ValueError(f"school terms line {number}: the first day is after the last")
        rows.append(Row(calendar, kind, start, end, source, published(source)))
    covers = [r for r in rows if r.kind == "cover"]
    for r in rows:
        if r.kind == "off" and not any(
            c.source == r.source
            and c.calendar == r.calendar
            and c.first <= r.first
            and r.last <= c.last
            for c in covers
        ):
            raise ValueError(f"school terms: {r} lies outside every span of its source")
    for k, a in enumerate(covers):
        for b in covers[k + 1 :]:
            if (
                a.calendar == b.calendar
                and a.published == b.published
                and a.first <= b.last
                and b.first <= a.last
            ):
                raise ValueError(f"school terms: {a} and {b} overlap with one publication day")
    return tuple(rows)


def load_terms(path=TERMS_FILE) -> tuple[Row, ...]:
    return read_terms(path.read_text(encoding="utf-8"))


def day_value(rows: tuple[Row, ...], calendar: str, day: date) -> float:
    """1, 0 or NaN for one calendar and day (module text)."""
    covering = [
        r
        for r in rows
        if r.kind == "cover"
        and r.calendar == calendar
        and r.first <= day <= r.last
        and r.published < day
    ]
    if not covering:
        return math.nan
    chosen = max(covering, key=lambda r: r.published)
    off = any(
        r.kind == "off"
        and r.calendar == calendar
        and r.source == chosen.source
        and r.first <= day <= r.last
        for r in rows
    )
    return 1.0 if off else 0.0


def school_columns(
    frame: pd.DataFrame,
    rows: tuple[Row, ...],
    site_calendars: Mapping[str, Mapping[str, str | None]] = SITE_CALENDARS,
) -> pd.DataFrame:
    """`COLUMNS` for the rows of `frame`, read from its `site` and `date` only (module text)."""
    sites = frame["site"].astype(str).to_numpy()
    days = pd.to_datetime(frame["date"].astype(str), format="%Y-%m-%d").dt.date.to_numpy()
    out = {}
    for column in COLUMNS:
        mapping = site_calendars[column]
        values = np.full(len(frame), math.nan)
        cache: dict[tuple[str, date], float] = {}
        for k, (site, day) in enumerate(zip(sites, days, strict=True)):
            calendar = mapping[site]
            if calendar is None:
                continue
            key = (calendar, day)
            if key not in cache:
                cache[key] = day_value(rows, calendar, day)
            values[k] = cache[key]
        out[column] = values
    return pd.DataFrame(out, index=frame.index)
