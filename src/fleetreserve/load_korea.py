"""The Korean charging transactions (held out) into the common session table, one unit per
portfolio, from bytes.

Source: figshare 22495141, file `ChargingRecords.csv`, the data of Baek, Lee and Kim (2024),
Scientific Data 11:262, handed over as bytes by the held-out loader's gated opener. This module
reads no file.

Layout: from the data paper, as summarised from it, and from the authors' plotting scripts
(`Code.zip` on the same figshare record: six R scripts, code and comments only, read as text and
never run on 15 November 2026, the owner's yes of that day; `docs/prior-looks.md`); no row and no
header line of the file has been seen. The paper's Table 7 names 16 columns: `ChargingsessionID`,
`UserID`, `ChargerID`, `ChargerCompany`, `Location`, `ChargerType`, `ChargerCapacity`,
`ChargerACDC`, `StartDay`, `StartTime`, `EndDay`, `EndTime`, `StartDatetime`, `EndDatetime`,
`Duration`, `Demand`. The paper says: `Location` is the charger's installed location type, one of
the 14 types of its Table 6; `ChargerType` is 1 for a fast charger and 0 for a slow one; `UserID` 0
marks another company's member or a non-member; `StartDatetime` and `EndDatetime` are the start and
end of the connection between car and charger, written YYYY-MM-DD HH:MM:SS; `Duration` is the
connection's length in minutes; `Demand` is the energy charged in kWh. It states no time zone and
gives the capacity only as a category; its text writes the start column as `SrartDatetime`. The
summary of it gives no field separator and no form of a missing value. The scripts read the file
with `data.table::fread` (which finds the separator itself, so they do not settle it) and use,
spelled so, `StartDay`, `StartTime`, `EndDay`, `EndTime`, `UserID`, `ChargerID`, `ChargerCompany`,
`Location` and `Demand`; they build `StartDatetime` and `EndDatetime` themselves from `StartDay` and
`StartTime` and from `EndDay` and `EndTime` (`ymd_hms` of the two joined by a space), subtract days
as dates, and read the times as hours, minutes and seconds. Their comments write the 14 location
types in lower case English, the golf course as "golf"; name `UserID` 0 as "non-member/members of
other companies"; and give `ChargerCompany` 1 as the authors' own company's chargers and 0 as other
companies'. They never name `ChargerType`, `ChargingsessionID`, `ChargerCapacity`, `ChargerACDC`,
`Duration`, `StartDatetime` or `EndDatetime` as read from the file, and state no encoding and no
time zone.

The session table is filled from the columns `REQUIRED`, one start time and one end time (K5);
a missing one raises `LayoutError`. Where the datetime column of a side and both its day and its
time column are present, the two forms are compared (K13). The other columns, `ChargerCompany`,
`ChargerCapacity`, `ChargerACDC` and `Duration` among them, and a day or time column whose side
lacks the other one of the pair, are read as text and not used, and need not be present.

Readings, each with its alternative in brackets:

  K1. Header names: found by name, not position, after trimming spaces and case folding, so
      `chargingsessionid` or ` Demand ` is read as the paper's name. The start column is
      `StartDatetime` or `SrartDatetime`: at most one of the two may be present (with neither,
      the start comes from the day and time columns, K5). Two header
      names that are equal after trimming and folding raise `LayoutError`. [Alternative: the
      names exactly as printed.]
  K2. Field separator: the first of "," and ";" whose split of the header holds every name read;
      the header is the file's first record as the csv module reads it in strict mode, with the
      line ends pandas reads ("\\n", "\\r\\n" or "\\r" alone); a header that neither separator
      can split raises `LayoutError`. Under ";" a decimal comma in a number is read as a
      decimal point, under "," the decimal mark is ".". The bytes must be UTF-8 text, with or
      without a byte order mark; any other encoding (UTF-16, or EUC-KR or CP949 with Korean
      characters) raises `LayoutError`. Missing value: an empty field, `NA`, `NaN` or `nan`, as
      written, without trimming (the id columns add their own forms, K8 and K9). A blank line is
      skipped, a quoted field may hold a separator or a line end, and a NUL character is text.
      A row with fewer fields than the header has its last fields missing; a row with more
      raises `LayoutError`, as does a quote never closed. [Alternatives: one fixed separator;
      another encoding or missing marker; every row of the header's width, as `load_asr`
      requires.]
  K3. `Location`: one of the 14 names of `PORTFOLIO_OF`, the types of the paper's Table 6 as
      the authors' scripts write them (the golf course "golf", which the summary of the paper
      gave as "Golf CC"), compared after trimming and case folding. A value outside them, a
      missing one included, raises `LayoutError` with the number of such rows and of distinct
      such values (after folding), never a value.
      Korean-script labels and numeric codes are NOT covered: if the file writes the location
      types that way, every row is outside the table and the set cannot be loaded until a
      mapping of those labels is added. [Alternative: none is built.]
  K4. `ChargerType`: a number equal to 0 (slow) or 1 (fast) as `pandas.to_numeric` reads it,
      spaces around it allowed: "0", "1", "0.0", " 1 " and the like. Anything else, a missing
      value included, raises `LayoutError`. [Alternative: words such as "slow" and "fast",
      which the paper does not use.]
  K5. Times: the start is `StartDatetime` (or `SrartDatetime`) where the header has it, else
      `StartDay` and `StartTime` joined by one space, as the authors' script builds it; the end
      likewise `EndDatetime`, else `EndDay` and `EndTime` (each side on its own). A side with
      neither its datetime column nor both its day and time columns raises `LayoutError`, and a
      joined time is missing where its day or its time is missing. Every time read is exactly
      YYYY-MM-DD HH:MM:SS, every field zero-padded, ASCII digits only, seconds 00 to 59 (the whole
      text must match `TIME_PATTERN`), and a real date and time; any other form, a seconds field of
      60 included, raises `LayoutError`. Every time read lies in `FIRST_TIME` to `LAST_TIME` (1
      September 2021 to 31 October 2022): the paper's span, 30 September 2021 to 30 September 2022,
      with a margin of about a month on each side for sessions that begin before the span or end
      after it; a time outside raises `LayoutError` with the count, so sentinel dates (year 0, 1,
      1900, 1970) cannot become sessions of decades. They are read as Asia/Seoul wall clock (plan
      section 3: "KST, assumed"). Asia/Seoul has one offset, UTC+09:00, over the whole range, so no
      time read falls in a daylight-saving gap or doubled hour. Where a datetime column is present,
      it gives its side's time, and its side's day and time columns, when both are present, are
      compared with it (K13). The messages of a joined time name both columns ("StartDay and
      StartTime"). The time column's hour may have one digit (K14).
      [Alternatives: times in UTC; no range, any year read; the paper's span with no margin; the day
      and time columns always, as the scripts read them.]
  K6. Plug-in is the start time of K5, plug-out its end time (the paper's "connection"). The file
      has no charging start or end, so `charge_start` and `charge_end` are missing. `Duration`
      (the connection's length in minutes) is not read: the table takes the connection from the
      two times, and a file without the column, or with any text in it, gives the same table.
      [Alternative: `Duration` required and read as a number, as `load_norway` reads its stated
      connection time, refusing the whole set for a column the table does not use.]
  K7. Energy: `Demand` in kWh.
  K8. User: `UserID` as written, outer spaces kept (" 17" and "17" are two users). A missing
      user (no history) is given by: a value that `pandas.to_numeric` reads as 0 or as a
      negative number, spaces around it allowed ("0", "0.0", "00", " 0 ", "-1", "-2.5"); or a
      value that, after trimming and case folding, is one of `ID_MARKERS` (blank, "na", "nan",
      "null", "n/a", "none", "-"). Zero is the paper's mark for another company's member or a
      non-member, as `load_asr` treats its shared IDs; a negative number and those markers are
      read as forms of a missing id, so that they never pool unrelated sessions into one
      user's history. A fraction ("0.5") or any other text is a user as written.
      [Alternatives: keep 0, the negative numbers and the markers as users as written.]
  K9. Station: `ChargerID` as written, outer spaces kept, missing where it is missing (K2). A
      value that is one of `ID_MARKERS` after trimming and case folding, or that
      `pandas.to_numeric` reads as a negative number, raises `LayoutError` with the count: such
      a form would pool unrelated chargers into one station, and the paper gives no missing
      form for it. [Alternative: read it as a missing station, as an empty field is.]
      Session id: `ChargingsessionID` as written, outer spaces kept (" A1" and "A1" are two
      sessions); a row without one raises `LayoutError`. The file has no site identifier, so
      a portfolio pools every charger of its location types, at car parks that have nothing
      else in common. [Alternative: none; no site can be formed.]
  K10. Nominal cap: 7 kW for every session (plan section 3: "assumed: the data give a capacity
      category and no kW"), labelled `NOMINAL_RULE` ("7 kW assumed: no kW in the data") in the
      session table's `nominal_rule` column and in every table that keeps that column. Physical
      limit for the drop rule below: 22 kW, the European three-phase limit
      `sessions.PHYSICAL_LIMIT_KW["norway"]`, above what an alternating-current charger at 32 A
      on a 380 V three-phase supply can give (about 21 kW). Its consequence: a slow-charger row
      whose average lies between 7 and 22 kW is kept, and under the plan's main rate (the
      user's highest observed average) it raises that user's rate above the 7 kW cap.
      `ChargerCapacity` and `ChargerACDC` are not used: their categories are not stated, and
      plan section 3 gives them no use (its rate sensitivity takes the 7 kW cap).
      [Alternative: 7.04 kW, a single-phase 32 A charger at 220 V, as the limit, which drops
      every row whose average is above it.]
  K11. A session that crosses midnight stays one row, its plug-in on the evening's date and its
      plug-out on the next; which day it counts on is the common rule (plugged in at τ).
  K12. Units: the five portfolios of `UNITS`, from the 14 location types by the one table
      `PORTFOLIO_OF` (the grouping of 15 November 2026, confirmed on 16 November 2026;
      moving a type is a one-line edit). A session's `site` is its portfolio's unit
      name, so each portfolio is a site of the table and its dropped rows are placed in it.
      Nothing is merged: a portfolio with few counting days abstains later by the plan's
      counting rule. [Alternative: one site `korea` with the portfolio in a column of its own,
      which the common table does not have.]
  K13. Both forms of a time: where the header holds a side's datetime column and both its day
      and its time column, the day and time joined (K5, K14) are read by the rules of K5 (form,
      then range, with their messages) and must give, in every row, the time read from the
      datetime column; a row where one form is missing and the other is not differs too. Any
      differing row raises `LayoutError` "Korea StartDatetime and StartDay and StartTime: <n>
      rows disagree" ("1 row disagrees" for one; the same for the end), with the count and
      never a value. The paper
      gives both forms one meaning, and the authors' scripts read the day and time columns, so
      a file in which they differ would give other times than the scripts'. A side with only one
      of its day and time columns is not compared. [Alternatives: no comparison, the datetime
      column alone, a disagreement unseen; rows compared only where both forms are present, so
      a time missing in one form is taken from the datetime column alone.]
  K14. A time column's hour of one digit ("7:05:00", the hour 0 to 9 written without its
      leading zero) is read with a leading zero ("07:05:00") before the day and time are joined,
      wherever the time column is read (K5, K13); minutes and seconds keep two digits, and the
      datetime columns keep the paper's form exactly (a one-digit hour there raises
      `LayoutError`). The paper states the datetime form only; a time column written by a
      program that drops the leading zero would otherwise refuse the set at the opening.
      [Alternative: the time column held to the datetime form's two-digit hour, refusing such a
      file.]

Rows removed, by reason, in this order (`REASONS`): the fast-charger rule first, then the order
of `load_norway`:
1. `fast charger (ChargerType 1)`: only slow-charger sessions are used (the owner's decision of
   15 November 2026; plan section 3). Every row with `ChargerType` 1 is removed here, before any
   other rule, so its count is the number of fast-charger rows in the file; it is on record in
   the dropped rows of every command that loads the set and writes their counts.
2. `no plug-out time recorded`.
3. `plug-out not after plug-in`.
4. `energy missing or not positive`.
5. `duplicate session id`: a session id held by an earlier row that the rules above kept (the
   first such row is kept; a row whose id was held only by rows already removed is kept), as
   in `load_norway` and `load_acn`, which compare within one site; here the ids are compared
   across all five portfolios, since the file is one source whose ids the paper calls charging
   session ids.
6. `average above the physical limit`: energy over plug-in to plug-out above 22 kW.
Every removed row is one row of the dropped rows, with its reason, so each count is in what
the loader returns.

  K15. Portfolios: each of the five portfolios of K12 must have a row kept, or hold only
      fast-charger rows. A portfolio with no row in the file, or with a row that is not a
      fast-charger row and none kept (all removed by the reasons 2 to 6 above), raises
      `LayoutError` whose message gives the portfolio, the number of rows read and the number
      removed by each reason, never a value of a row: a named portfolio with no usable session is
      a fault in reading the file, not a result about the portfolio, so the refusal is logged and
      the parsing-fix route of plan section 13 applies. A portfolio whose every row is a
      fast-charger row is a result about the data (the slow-charger rule, plan section 3): it is
      returned with no session and abstains later. [Alternative: every portfolio with no row
      kept abstains.]

These raise `LayoutError`, checked in this order, so that the first one found names the file:
bytes that are not UTF-8; a header that cannot be split or that lacks a column; a repeated
column; both start columns; a row that cannot be split under the header or that holds more
fields than it; a row without a plug-in time or session id; an unreadable plug-in time, then
one outside the range; the same for the plug-out time; for the start, then the end, where both
forms are present (K13), the day and time columns unreadable, then outside the range, then
disagreeing with the datetime column; an energy that is not a number; a
charger type, then a location type, then a charger id outside the readings; then, with the
table built, a portfolio with no row kept that is not all fast-charger rows (K15). The message
names a column or the file, with a count where rows or names are counted, and never a value of a
row. Any other error while reading
(`Exception`, not an interruption) raises `LayoutError` "Korea file cannot be read" with the
error's type name only and no cause attached, since an error's own text may hold a value of a
row. [Alternative: the error passes unchanged, or with its text.]

Not covered, each loud, and what would settle it before the freeze:
- Session ids unique only per charger: every repeat across chargers is removed as a duplicate.
  The count under `duplicate session id` in the dropped rows shows it: about as many rows as
  the chargers share ids. The authors' scripts never use `ChargingsessionID`, so they do not
  settle it; the count at the opening does.
- `ChargerType`: the scripts never use it, so its presence and its coding rest on the paper
  alone; a file without it refuses the set.
- The 14 location names: settled by the scripts, which write them; a value outside them still
  refuses the set, with its row count and distinct count.

Role: `sessions.finish` accepts only the roles train and development, so the table is checked by
`finish` under the role development and its role column is then set to "held-out".
"""

import csv
import io
import re

import numpy as np
import pandas as pd

from fleetreserve.heldout_limits import (
    ENERGY_ABOVE_KWH,
    FAST,
    FIRST_TIME,
    LAST_TIME,
    LIMIT_KW,
    SLOW,
)
from fleetreserve.sessions import (
    Dropper,
    LayoutError,
    finish,
    hours_between,
    localize,
)

ROLE = "held-out"
CHECKED_AS = "development"
ZONE = "Asia/Seoul"
NOMINAL_KW = 7.0
NOMINAL_RULE = "7 kW assumed: no kW in the data"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# The whole text, ASCII digits only, seconds 00 to 59.
TIME_PATTERN = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:[0-5]\d"
# A time column's text whose hour has one digit ("7:05:00"; reading K14), ASCII digits only.
ONE_DIGIT_HOUR = re.compile(r"\d:", flags=re.ASCII)
# The removal limits (`LIMIT_KW`, `ENERGY_ABOVE_KWH`, `SLOW`, `FAST`) and the time range
# (`FIRST_TIME`, `LAST_TIME`; the paper's span with a margin, reading K5) are in `heldout_limits`.
SEPARATORS = (",", ";")
MISSING = ["", "NA", "NaN", "nan"]
# Further forms of a missing id in `UserID` and `ChargerID`, after trimming and case folding.
ID_MARKERS = ("", "na", "nan", "null", "n/a", "none", "-")
START = "StartDatetime"
START_NAMES = (START, "SrartDatetime")
END = "EndDatetime"
# The day and time columns that give a time when its datetime column is absent (reading K5).
DAY_TIME = {START: ("StartDay", "StartTime"), END: ("EndDay", "EndTime")}
REQUIRED = (
    "ChargingsessionID",
    "UserID",
    "ChargerID",
    "Location",
    "ChargerType",
    "Demand",
)
UNITS = ("korea-home", "korea-lodging", "korea-work", "korea-public-parking", "korea-visits")
# Location type (the paper's Table 6) -> unit; one line per type (reading K12).
PORTFOLIO_OF = {
    "Apartment": "korea-home",
    "Hotel": "korea-lodging",
    "Resort": "korea-lodging",
    "Accommodation": "korea-lodging",
    "Camping": "korea-lodging",
    "Company": "korea-work",
    "Public institution": "korea-work",
    "Public area": "korea-public-parking",
    "Public parking lot": "korea-public-parking",
    "Bus garage": "korea-public-parking",
    "Market": "korea-visits",
    "Restaurant": "korea-visits",
    "Sightseeing": "korea-visits",
    "Golf": "korea-visits",
}
UNIT = "_unit"  # the column of read_raw's table that holds each row's unit
REASONS = (
    "fast charger (ChargerType 1)",
    "no plug-out time recorded",
    "plug-out not after plug-in",
    "energy missing or not positive",
    "duplicate session id",
    "average above the physical limit",
)


def _fold(name: str) -> str:
    return name.strip().casefold()


def _has_side(names: set[str], side: str) -> bool:
    """Whether the folded header `names` give the start or end time `side`: its datetime
    column (either spelling for the start), or both its day and its time column (K5)."""
    given = START_NAMES if side == START else (END,)
    return any(_fold(name) in names for name in given) or all(
        _fold(name) in names for name in DAY_TIME[side]
    )


def header(text: str) -> tuple[str, list[str]]:
    """The separator (reading K2) and the header names, trimmed and folded (reading K1): the
    first record of the text, read by the csv module in strict mode with every line end that
    pandas reads ("\\n", "\\r\\n", "\\r"). A header that no separator can split raises
    `LayoutError` with the csv module's error under ","."""
    wanted = {_fold(name) for name in REQUIRED}
    errors = []
    for sep in SEPARATORS:
        records = csv.reader(io.StringIO(text, newline=""), delimiter=sep, strict=True)
        try:
            names = [_fold(name) for name in next(records, [])]
        except csv.Error as error:
            errors.append(error)
            continue
        if wanted <= set(names) and _has_side(set(names), START) and _has_side(set(names), END):
            return sep, names
    if len(errors) == len(SEPARATORS):
        raise LayoutError(f"Korea file: the header line cannot be split: {errors[0]}")
    raise LayoutError(
        "Korea file lacks columns: no separator gives "
        + ", ".join(REQUIRED)
        + f", a start ({' or '.join(START_NAMES)}, or {' and '.join(DAY_TIME[START])})"
        + f" and an end ({END}, or {' and '.join(DAY_TIME[END])})"
    )


def _number(text: pd.Series, sep: str) -> pd.Series:
    """Numbers written in the file, a decimal comma read as a point under ";"; unreadable ones
    become missing."""
    written = text.astype("string")
    if sep == ";":
        written = written.str.replace(",", ".", regex=False)
    return pd.to_numeric(written, errors="coerce")


def _read_number(raw: pd.DataFrame, name: str, sep: str) -> pd.Series:
    number = _number(raw[name], sep)
    bad = raw[name].notna() & number.isna()
    if bad.any():
        raise LayoutError(f"Korea {name}: {int(bad.sum())} values are not numbers")
    return number.astype(float)


def _marked(text: pd.Series, sep: str) -> pd.Series:
    """Ids written as a missing marker (`ID_MARKERS`) or as a negative number (readings K8,
    K9); a missing id is not marked."""
    folded = text.map(lambda v: None if v is None else _fold(v))
    negative = (_number(text, sep) < 0).fillna(False).astype(bool)
    return folded.isin(ID_MARKERS) | negative


def _joined(raw: pd.DataFrame, side: str) -> tuple[pd.Series, str]:
    """The day and time columns of `side` joined by one space, missing where either is missing,
    a time of one-digit hour given its leading zero (reading K14), and the name its messages
    give (reading K5)."""
    day, time = DAY_TIME[side]
    both = raw[day].notna() & raw[time].notna()
    hour = raw.loc[both, time].map(lambda v: "0" + v if ONE_DIGIT_HOUR.match(v) else v)
    joined = pd.Series(None, index=raw.index, dtype=object)
    joined[both] = raw.loc[both, day] + " " + hour
    return joined, f"{day} and {time}"


def _time_text(raw: pd.DataFrame, side: str) -> tuple[pd.Series, str]:
    """The text of the start or end time `side` and the name its messages give: the datetime
    column when the header has it, else the day and time columns joined (reading K5)."""
    if side in raw.columns:
        return raw[side], side
    return _joined(raw, side)


def _compare(raw: pd.DataFrame, side: str, read: pd.Series) -> None:
    """Reading K13: where the header has the datetime column of `side` and both its day and its
    time column, the joined day and time, read by the rules of K5, must give the time `read`
    from the datetime column in every row, a missing time agreeing only with a missing one;
    otherwise `LayoutError` with the number of rows that differ."""
    if side not in raw.columns or not all(name in raw.columns for name in DAY_TIME[side]):
        return
    joined, name = _joined(raw, side)
    other = _read_time(joined, name)
    same = (read == other).fillna(False).astype(bool) | (read.isna() & other.isna())
    if not same.all():
        differ = int((~same).sum())
        rows = "1 row disagrees" if differ == 1 else f"{differ} rows disagree"
        raise LayoutError(f"Korea {side} and {name}: {rows}")


def _read_time(text: pd.Series, name: str) -> pd.Series:
    written = text.astype("string")
    exact = written.str.fullmatch(TIME_PATTERN, flags=re.ASCII).fillna(False).astype(bool)
    parsed = pd.to_datetime(written.where(exact), format=TIME_FORMAT, errors="coerce")
    bad = text.notna() & parsed.isna()
    if bad.any():
        raise LayoutError(f"Korea {name}: {int(bad.sum())} times cannot be read")
    outside = (parsed < pd.Timestamp(FIRST_TIME)) | (parsed > pd.Timestamp(LAST_TIME))
    if outside.any():
        raise LayoutError(
            f"Korea {name}: {int(outside.sum())} times outside {FIRST_TIME} to {LAST_TIME}"
        )
    return localize(parsed, ZONE)


def read_raw(content: bytes) -> pd.DataFrame:
    """The file as text columns under the paper's names, the missing markers read as missing;
    times read as Asia/Seoul times, `Demand` read, `Location`, `ChargerType` and `ChargerID`
    checked, the unit of each row in the column `UNIT`, and a `UserID` of 0, of a negative
    number or of a marker made missing (readings K1 to K9)."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise LayoutError("Korea file is not UTF-8 text") from None
    sep, names = header(text)
    if len(set(names)) != len(names):
        raise LayoutError(
            f"Korea file: {len(names) - len(set(names))} header names repeat after trimming "
            "and case folding"
        )
    if all(_fold(name) in names for name in START_NAMES):
        raise LayoutError("Korea file has both " + " and ".join(START_NAMES))
    try:
        raw = pd.read_csv(
            io.BytesIO(content),
            sep=sep,
            dtype=str,
            keep_default_na=False,
            na_values=MISSING,
            encoding="utf-8-sig",
        )
    except pd.errors.ParserError:
        raise LayoutError(
            "Korea file: a row cannot be split under the header (more fields than the header, "
            "or a quote never closed)"
        ) from None
    if not raw.index.equals(pd.RangeIndex(len(raw))):
        raise LayoutError("Korea file: rows hold more fields than the header")
    paper = {_fold(name): name for name in (*REQUIRED, START, END, *DAY_TIME[START])}
    paper |= {_fold(name): name for name in DAY_TIME[END]}
    paper[_fold(START_NAMES[1])] = START
    raw.columns = [paper.get(name, name) for name in names]
    raw = raw.astype(object).where(raw.notna(), None)
    start, start_name = _time_text(raw, START)
    end, end_name = _time_text(raw, END)
    if start.isna().any() or raw["ChargingsessionID"].isna().any():
        raise LayoutError("Korea file: a row has no plug-in time or no session id")
    raw[START] = _read_time(start, start_name)
    raw[END] = _read_time(end, end_name)
    _compare(raw, START, raw[START])
    _compare(raw, END, raw[END])
    raw["Demand"] = _read_number(raw, "Demand", sep)
    kind = _number(raw["ChargerType"], sep)
    bad = ~kind.isin([SLOW, FAST])
    if bad.any():
        raise LayoutError(f"Korea ChargerType: {int(bad.sum())} values are neither 0 nor 1")
    raw["ChargerType"] = kind.astype(float)
    types = {_fold(name): unit for name, unit in PORTFOLIO_OF.items()}
    unit = raw["Location"].map(lambda v: None if v is None else types.get(_fold(v)))
    outside = unit.isna()
    if outside.any():
        distinct = raw.loc[outside, "Location"].map(lambda v: v if v is None else _fold(v))
        raise LayoutError(
            f"Korea Location: {int(outside.sum())} rows hold no location type of the paper "
            f"({distinct.nunique(dropna=False)} distinct values)"
        )
    raw[UNIT] = unit
    marked = _marked(raw["ChargerID"], sep)
    if marked.any():
        raise LayoutError(
            f"Korea ChargerID: {int(marked.sum())} values are written as a missing marker or a "
            "negative number"
        )
    zero = (_number(raw["UserID"], sep) == 0).fillna(False).astype(bool)
    raw["UserID"] = raw["UserID"].where(~(zero | _marked(raw["UserID"], sep)), None)
    return raw


def to_sessions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The raw table to (session table, dropped rows)."""
    nat = pd.Series(pd.NaT, index=raw.index, dtype=f"datetime64[us, {ZONE}]")
    table = pd.DataFrame(
        {
            "site": raw[UNIT],
            "role": CHECKED_AS,
            "session_id": raw["ChargingsessionID"],
            "user_id": raw["UserID"],
            "station": raw["ChargerID"],
            "plug_in": raw[START],
            "plug_out": raw[END],
            "charge_start": nat,
            "charge_end": nat,
            "energy_kwh": raw["Demand"],
            "max_power_kw": np.nan,
            "declared_departure": nat,
            "declared_energy_kwh": np.nan,
            "nominal_kw": NOMINAL_KW,
            "nominal_rule": NOMINAL_RULE,
        },
        index=raw.index,
    )
    dropper = Dropper(table["site"], table["session_id"], raw[START].dt.date)
    fast, no_end, backwards, energy, duplicate, average_rule = REASONS
    table = dropper.drop(table, raw["ChargerType"] == FAST, fast)
    table = dropper.drop(table, table["plug_out"].isna(), no_end)
    table = dropper.drop(table, table["plug_out"] <= table["plug_in"], backwards)
    table = dropper.drop(table, ~(table["energy_kwh"] > ENERGY_ABOVE_KWH), energy)
    table = dropper.drop(table, table["session_id"].duplicated(keep="first"), duplicate)
    average = table["energy_kwh"] / hours_between(table["plug_out"], table["plug_in"])
    table = dropper.drop(table, average > LIMIT_KW, average_rule)
    sessions = finish(table)
    sessions["role"] = pd.Series(ROLE, index=sessions.index, dtype=object)
    return sessions, dropper.table()


def require_portfolios(sessions: pd.DataFrame, dropped: pd.DataFrame) -> None:
    """`LayoutError` for the first of `UNITS` with no row kept that is not a portfolio of
    fast-charger rows only (reading K15), naming the portfolio and the counts of rows read and
    removed by each reason (in the order the rules removed them)."""
    for unit in UNITS:
        if (sessions["site"] == unit).any():
            continue
        removed = dropped.loc[dropped["site"] == unit, "reason"].tolist()
        if removed and set(removed) == {REASONS[0]}:
            continue
        counts = ", ".join(
            f"{reason}: {removed.count(reason)}" for reason in dict.fromkeys(removed)
        )
        raise LayoutError(
            f"Korea file: portfolio {unit} has no row kept: {len(removed)} rows read"
            + (f", removed by reason {counts}" if removed else "")
        )


def from_bytes(content: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The session table of the five Korean portfolios and its dropped rows. Any error other
    than `LayoutError` while reading becomes `LayoutError` naming the error's type only; a
    portfolio with no row kept is a `LayoutError` unless its rows are all fast-charger rows."""
    try:
        sessions, dropped = to_sessions(read_raw(content))
        require_portfolios(sessions, dropped)
        return sessions, dropped
    except LayoutError:
        raise
    except Exception as error:
        raise LayoutError(f"Korea file cannot be read: {type(error).__name__}") from None
