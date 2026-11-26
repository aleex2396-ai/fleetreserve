"""The a.s.r. Utrecht office car park (held out) into the common session table, from bytes.

Source: 4TU 80ef3824-3f5d-4e45-8794-3b8791efbd13, file `202410DatasetEVOfficeParking_v0.csv`,
handed over as bytes by the held-out loader's gated opener. This module reads no file.

Layout read from the data set's README, corrected by the layout look of 23 November 2026 (the
owner's decisions 108 and 109: line 1 of the file, the shape of each column's values and the
labels of the columns with at most 20 distinct digit-free values were read, nothing else; the
look's report stays outside git): one row per charging session; 14 columns `EV_id_x` (EV
identifier pseudonym, from the registered charging ID), `start_datetime` and `end_datetime`
(charger connection and disconnection, local time, UTC+01:00 in winter and UTC+02:00 in summer;
written YYYY-MM-DD hh:mm:ss in the file, the README wrote YYYY-MM-DDThh:mm:ss), `total_energy`
(kWh charged in the session), `rail`, `evse_uid` (charging station identifier; the README
called it `evse_id`), `channel` (socket 1 or 2), and survey columns. The session table is
filled from the columns `COLUMNS` only; they are found by name, and a missing one raises
`LayoutError`. The other columns of the file are read too, as text, for one use: the duplicate
rule (5 below) compares every column of the file.

Readings of the README, each with its alternative in brackets:

  A1. Field separator: the first of ",", ";", a tab and "|" (in this order) whose split of the
      header line holds every name in `COLUMNS` (the README names no separator; the real file
      uses ";", as the layout look showed; the first fix added the tab and "|" after the real
      file was refused). Under ";" a decimal comma in
      `total_energy` is read as a decimal point, as in the Norway files; under ",", a tab and
      "|" the decimal mark is ".". A byte order mark at the start of the file is skipped.
      [Alternative: one fixed separator; a decimal comma also under a tab or "|".]
  A2. Missing value: an empty field, `NA`, `NaN` or `nan` (the README says values are missing
      but not how they are written). [Alternative: another marker, which would then be read as
      text and, in a time or number column, raise `LayoutError`.]
  A3. Times: exactly YYYY-MM-DD hh:mm:ss, the form of the file (a space between date and time,
      as the layout look showed), or the same with a "T" in place of the space, the form of the
      README, with every field zero-padded (the whole text must match `TIME_PATTERN`, ASCII
      digits only), Europe/Amsterdam wall clock. Any other form raises `LayoutError`:
      one-digit fields, an offset or "Z", fractions of a second, two spaces or any other
      character between date and time, a leading or trailing space, a time without seconds.
      [Alternative: only the form of the file; times in UTC or carrying an offset.]
  A4. Plug-in and plug-out are `start_datetime` and `end_datetime` ("charger connection" and
      "disconnection"). The README names no separate charging start or end, so `charge_start`
      and `charge_end` are missing.
  A5. `total_energy` is the session's energy in kWh, as plan section 3 states.
  A6. User: `EV_id_x`. The README lists twenty charging IDs that "cannot be attributed to a
      unique user" (`SHARED_IDS`); their sessions get a missing user. [Alternative: keep them as
      users.] The README says "at least" these IDs, so other shared IDs, if any, stay users.
  A7. Station: `evse_uid` (the file's name for the column, from the layout look; digits only,
      kept as text with any leading zeros) as text, missing where it is missing (the README:
      Rail and EVSE information is not provided for all data points); `channel` does not enter
      the station (it counts only in the duplicate rule). The README's name `evse_id` is
      accepted too (`ALIASES`): the table's column is the same whichever name the header
      holds, and a header holding both is a column named twice (A13). [Alternatives: one
      identifier per socket, `evse_uid` with `channel`; only the name `evse_uid`.]
  A8. Session id: the README names no session id column; a session's id is `row-<n>`, n its
      data row's number in the file (the first data row is 1), as `load_nrel` does.
  A9. The README's own filters (sessions under 1 kWh, under 10 minutes or over 24 hours are not
      included) are taken as applied by the publisher and are not applied again.
  A10. Regimes and units by the local date of plug-in (`periods`), or of any given day
      (`regime_of`, `unit_of`): free before 1 September 2022; paid from 1 September 2022 (the
      README: "After September 1st, 2022"; plan section 3: "paid from September 2022"); paid
      with the controller from 1 March 2023 (README: "As of March 2023"; the day is not stated).
      Units: before and from 1 March 2023 (plan section 3). The README limits the controller to
      the employee deck's rails other than rail M; the regime here is by date for the whole car
      park. [Alternatives: paid from 2 September 2022; the controller regime only on the
      employee deck's rails other than M.]
  A11. Text and records, checked before `pandas.read_csv` reads the file (`decoded`,
      `records`): the bytes must be UTF-8 text (a byte order mark at the start skipped) with
      no NUL character; the csv module in strict mode must split them into records (blank
      lines left out, as `read_csv` leaves them out; a quoted field may hold the separator or
      a line break, and is then read as text); there must be a header record; every record
      must have the header's number of fields; no column name may appear twice in the header
      (names compared with spaces around them removed). A file that fails one of these raises
      `LayoutError`. So a record with one field too many is never read with its first field as
      the table's index, and a short record is never read with its absent fields as missing
      values. [Alternative: `read_csv` alone, which pads a short record with missing values
      and, when the first record has one field more than the header, takes the first column
      as the index.]

  A13. Header names (first fix): a header field is one of the five names of `COLUMNS` when,
      after removing a byte order mark, spaces around it and double quotes around it, and
      case-folding it, it equals the name case-folded, or, for the station, `evse_uid`
      case-folded (`ALIASES`); the column is then read as that name. No other name is
      accepted: no other alias, no fuzzy match. Two header fields that are the same name in
      this reading are a column named twice. [Alternative: the names exactly as the README
      writes them, with spaces around them removed.]
  A14. Refusals (first fix): a refusal states a layout fact and never a value of the file. The
      refusal for a header with no separator that gives the five names lists, for each separator
      tried, the fields of the file's first record (each cut at 40 characters, at most 40 shown,
      the count given), a field printed as text only when the first record lies wholly on the
      file's first physical line and a second non-blank csv record follows it (`on_first_line`:
      the first physical line is the text up to the first boundary of Python's `str.splitlines`:
      `\\n`, `\\r`, `\\r\\n`, `\\x0b`, `\\x0c`, `\\x1c`, `\\x1d`, `\\x1e`, `\\x85`, U+2028,
      U+2029; it, parsed alone under the separator, equals the first record of the whole text
      field by field; a non-blank record holds a character other than a space or a tab; a file of
      one record, with or without a line end after it, prints shapes only) and the field is
      printable ASCII (space to `~`) without an ASCII digit, and as its shape otherwise; so every
      field is a shape when blank lines come before the first record or when a quoted field of it
      runs over a line end (the fields after its closing quote, on a later line, included); and
      all of them as shapes with the words "the first line has the shape of a data record" when,
      under that separator, every field of the first line has the shape of the same field of the
      second record (the same number of fields); and, for the separator whose first line splits
      into the most fields (the first of them on a tie), the shape of each field of the first data
      record (at most 40 shown): each ASCII digit is `9`, each ASCII letter `a`, each other ASCII
      punctuation character or space is kept, any other character is `?`, and a shape is cut at 30
      characters. A field that cannot be read as a time or a number is refused with its column,
      the number of records that fail and the shapes of those fields (the first five distinct
      shapes, each with its count), in the same form. The first line of a refused file is not
      known to be a header (the rule must hold for any file, whatever its first line holds), so a
      field with a digit, which names rarely hold and values often do, is never printed as text;
      nor is any text from a later line, nor any
      non-ASCII text. The refusal for a header that names a column more than once gives the number
      of repeated names and never a name (a repeated field may hold a value). Remaining risk: a
      field of the file's first physical line, when that line is a complete csv record by itself
      and a second non-blank record follows, that is printable ASCII with no digit and is a data
      value (a category label) is still printed as text. Any other error while reading raises
      `LayoutError` "a.s.r. file cannot be read" with the error's type name only, since an error's
      own text may hold a value of a row. [Alternative: the first line printed as shapes only,
      naming no field; every field of it as text.]

Nominal cap: 11 kW per plug (plan section 3; the README: 22 kW per double charger, 11 kW per
plug when two cars are connected). Physical limit for the drop rule below: 22 kW, the European
three-phase limit `sessions.PHYSICAL_LIMIT_KW["norway"]`, which is also the README's capacity of
a double charger.

Rows removed, by reason, in this order (`REASONS`): the order of `load_norway`, with the
duplicate rule of `load_nrel`:
1. `no plug-out time recorded`.
2. `local time in the daylight-saving gap or doubled hour` (plug-in or plug-out).
3. `plug-out not after plug-in`.
4. `energy missing or not positive`.
5. `duplicate row`: a row equal to an earlier one in every column of the file, the columns not
   in `COLUMNS` included (the first is kept), as in `load_nrel`, since the README names no
   session id. Every column is compared as text, the two times and the energy after they are
   read; two rows that differ only in a column outside `COLUMNS`, such as `channel` or
   `capacity_kwh`, are both kept.
6. `average above the physical limit`: energy over plug-in to plug-out above 22 kW.

A missing column, an unreadable time or number, a row without a plug-in time, or a file that
fails reading A11 raises `LayoutError`.

  A12. Periods: each of the two units of A10 must have a row kept. A period with no row in the
      file (by the local date of plug-in), or whose rows are all removed by the reasons above,
      raises `LayoutError` whose message gives the period, the number of rows read and the number
      removed by each reason, never a value of a row: a named period with no session is a fault
      in reading the file (the file spans both periods), not a result about the period, so the
      refusal is logged and the parsing-fix route of plan section 13 applies. [Alternative: a
      period with no row kept abstains.]

Errors, as a class: `from_bytes` raises `LayoutError` for every file it cannot read. The checks
of this module raise it with their own messages; any other `Exception` raised while the bytes
are read or the table is built (a field longer than the csv module's limit of 131,072
characters, for example) is raised as `LayoutError` with that exception as its cause, so a
caller meets one kind of error. A file that is read gives rows kept, or removed by the reasons
above, and nothing else.

Role: `sessions.finish` accepts only the roles train and development, so the table is checked by
`finish` under the role development and its role column is then set to "held-out".
"""

import csv
import io
import json
import re
import string
from datetime import date

import numpy as np
import pandas as pd

from fleetreserve.heldout_limits import ENERGY_ABOVE_KWH, LIMIT_KW
from fleetreserve.sessions import (
    Dropper,
    LayoutError,
    finish,
    hours_between,
    localize,
)

SITE = "asr"
ROLE = "held-out"
CHECKED_AS = "development"
ZONE = "Europe/Amsterdam"
NOMINAL_KW = 11.0
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
# the whole text, ASCII digits only; a space or a "T" between date and time (reading A3)
TIME_PATTERN = r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
SEPARATORS = (",", ";", "\t", "|")
SEPARATOR_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "bar"}
NAME_CUT = 40  # characters of a header name in a refusal (reading A14)
FIELDS_SHOWN = 40  # header names and record shapes in a refusal
SHAPE_CUT = 30  # characters of a field's shape
SHAPES_SHOWN = 5  # distinct shapes of the fields that fail, in a refusal
PUNCTUATION = frozenset(string.punctuation + " ")
MISSING = ["", "NA", "NaN", "nan"]
COLUMNS = ("EV_id_x", "start_datetime", "end_datetime", "total_energy", "evse_id")
ALIASES = {"evse_uid": "evse_id"}  # the file's name for the station column (reading A7)
NAMES = {name.casefold(): name for name in COLUMNS} | ALIASES
SHARED_IDS = frozenset(
    {
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
    }
)
PAID_FROM = date(2022, 9, 1)
CONTROLLER_FROM = date(2023, 3, 1)
UNITS = ("asr-before-march-2023", "asr-from-march-2023")
REGIMES = ("free", "paid", "paid-controller")
REASONS = (
    "no plug-out time recorded",
    "local time in the daylight-saving gap or doubled hour",
    "plug-out not after plug-in",
    "energy missing or not positive",
    "duplicate row",
    "average above the physical limit",
)


def decoded(content: bytes, encoding: str, what: str) -> str:
    """The bytes as text (reading A11): bytes that are not text in `encoding`, or a NUL
    character, raise `LayoutError`."""
    try:
        text = content.decode(encoding)
    except UnicodeDecodeError as error:
        raise LayoutError(f"{what} is not UTF-8 text (byte {error.start})") from error
    if "\0" in text:
        raise LayoutError(f"{what} holds a NUL character")
    return text


def records(text: str, sep: str, what: str) -> list[list[str]]:
    """The text's records as the csv module splits them in strict mode, blank lines left out
    (reading A11); raise `LayoutError` for quoting it refuses, for no header record and for a
    record whose number of fields is not the header's."""
    lines = io.StringIO(text, newline="")
    try:
        rows = [row for row in csv.reader(lines, delimiter=sep, strict=True) if row]
    except csv.Error as error:
        raise LayoutError(f"{what} cannot be split into records: {error}") from error
    if not rows:
        raise LayoutError(f"{what} has no header line")
    for number, row in enumerate(rows[1:], start=1):
        if len(row) != len(rows[0]):
            raise LayoutError(
                f"{what}: record {number} has {len(row)} fields, the header {len(rows[0])}"
            )
    return rows


def normal(name: str) -> str:
    """A header name for comparison (reading A13): a byte order mark, spaces and double quotes
    around it removed, case-folded."""
    return name.lstrip("\ufeff").strip().strip('"').strip().casefold()


def shape(value: str) -> str:
    """The shape of a field (reading A14): ASCII digits `9`, ASCII letters `a`, other ASCII
    punctuation and spaces kept, anything else `?`, cut at `SHAPE_CUT` characters."""
    return "".join(
        "9"
        if char in string.digits
        else "a"
        if char in string.ascii_letters
        else char
        if char in PUNCTUATION
        else "?"
        for char in value[:SHAPE_CUT]
    )


def shapes_text(values) -> str:
    """The distinct shapes of the given fields with their counts, the first `SHAPES_SHOWN` of
    them in order of first appearance (reading A14); never a value."""
    counts: dict[str, int] = {}
    for value in values:
        key = shape(str(value))
        counts[key] = counts.get(key, 0) + 1
    listed = list(counts.items())[:SHAPES_SHOWN]
    text = ", ".join(f"{json.dumps(key)} x {count}" for key, count in listed)
    more = len(counts) - len(listed)
    return text + (f", and {more} more shapes" if more else "")


def first_records(text: str, sep: str, number: int) -> list[list[str]]:
    """The first `number` records of the text under `sep`, split without strict mode, blank
    lines left out (the header, then the first data record); fewer when the text has fewer or
    when the csv module cannot split the next one."""
    records_found: list[list[str]] = []
    try:
        for row in csv.reader(io.StringIO(text, newline=""), delimiter=sep):
            if row:
                records_found.append(row)
            if len(records_found) == number:
                break
    except csv.Error:
        pass
    return records_found


def blank(record: list[str]) -> bool:
    """Whether a csv record holds no character other than a space or a tab (an empty line is
    the empty record)."""
    return not any(field.strip(" \t") for field in record)


def first_physical_line(text: str) -> str:
    """The text up to the first boundary `str.splitlines` recognises (`\\n`, `\\r`, `\\r\\n`,
    `\\x0b`, `\\x0c`, `\\x1c`, `\\x1d`, `\\x1e`, `\\x85`, U+2028, U+2029), empty when the text
    is empty (reading A14)."""
    lines = text.splitlines()
    return lines[0] if lines else ""


def on_first_line(text: str, sep: str, record: list[str]) -> bool:
    """Whether the first csv record of the text lies wholly on the file's first physical line
    and is followed by a second non-blank record (reading A14): the first physical line,
    parsed alone under `sep`, equals `record` field by field, `record` is not blank, and the
    text holds at least one more non-blank csv record. False for a leading blank line, for a
    quoted field that runs over a line end (so for every field of that record, those after the
    closing quote included), for a record that the csv module reads over a boundary of
    `first_physical_line` that it does not itself read as a line end (U+2028, `\\x1e` and the
    like), when the csv module cannot split the text, and for a file of one record."""
    try:
        rows = list(csv.reader(io.StringIO(first_physical_line(text), newline=""), delimiter=sep))
        if (rows[0] if rows else []) != record or blank(record):
            return False
        whole = csv.reader(io.StringIO(text, newline=""), delimiter=sep)
        return sum(1 for row in whole if not blank(row)) >= 2
    except csv.Error:
        return False


def first_line_field(field: str) -> str:
    """A field of the file's first line as a refusal prints it (reading A14): its text, cut at
    `NAME_CUT` characters, only when it holds printable ASCII (space to `~`) and no ASCII digit;
    otherwise its shape. A control character or a line break in it means a quoted field that
    runs over later lines, or text that is not the first line's."""
    if all(" " <= char <= "~" and char not in string.digits for char in field):
        return json.dumps(field[:NAME_CUT])
    return "shape " + json.dumps(shape(field))


def refusal(text: str) -> str:
    """The text of the refusal when no separator gives the five names (reading A14): the layout
    as found, never a value."""
    found = {sep: first_records(text, sep, 2) for sep in SEPARATORS}
    parts = []
    for sep, rows in found.items():
        names = rows[0] if rows else []
        like_data = (
            len(rows) == 2
            and len(rows[1]) == len(names)
            and all(shape(a) == shape(b) for a, b in zip(names, rows[1], strict=True))
        )
        note = ", the first line has the shape of a data record" if like_data else ""
        if like_data or not on_first_line(text, sep, names):
            shown = ", ".join("shape " + json.dumps(shape(n)) for n in names[:FIELDS_SHOWN])
        else:
            shown = ", ".join(first_line_field(name) for name in names[:FIELDS_SHOWN])
        parts.append(f"{SEPARATOR_NAMES[sep]}: {len(names)} fields{note} [{shown}]")
    widest = max(SEPARATORS, key=lambda sep: len(found[sep][0]) if found[sep] else 0)
    message = (
        "a.s.r. file lacks columns: no separator gives "
        + ", ".join(COLUMNS)
        + "; header fields as found (a field is given as text only when it is printable ASCII"
        + " with no digit, else as its shape; each"
        + f" cut at {NAME_CUT} characters, at most {FIELDS_SHOWN} shown): "
        + "; ".join(parts)
    )
    if len(found[widest]) < 2:
        return message + "; the file has no data record"
    record = found[widest][1]
    shown = ", ".join(json.dumps(shape(field)) for field in record[:FIELDS_SHOWN])
    return (
        message
        + f"; first data record under {SEPARATOR_NAMES[widest]} ({len(record)} fields), field"
        + f" shapes (each cut at {SHAPE_CUT} characters, at most {FIELDS_SHOWN} shown): [{shown}]"
    )


def separator(text: str) -> str:
    """The first of `SEPARATORS` whose split of the header record names every column read
    (readings A1, A13); `LayoutError` with the layout found when none does."""
    for sep in SEPARATORS:
        rows = first_records(text, sep, 1)
        if set(COLUMNS) <= {NAMES.get(normal(name)) for name in (rows[0] if rows else [])}:
            return sep
    raise LayoutError(refusal(text))


def read_raw(content: bytes) -> pd.DataFrame:
    """The file as text columns, the missing markers read as missing; times parsed to naive
    local times and the energy read as a number."""
    text = decoded(content, "utf-8-sig", "a.s.r. file")
    sep = separator(text)
    names = [name.strip() for name in records(text, sep, "a.s.r. file")[0]]
    read = [NAMES[normal(name)] for name in names if normal(name) in NAMES]
    repeated = sorted(
        {name for name in names if names.count(name) > 1}
        | {name for name in read if read.count(name) > 1}
    )
    if repeated:
        raise LayoutError(
            f"a.s.r. file names columns more than once: {len(repeated)} repeated name(s)"
        )
    raw = pd.read_csv(
        io.BytesIO(content),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        na_values=MISSING,
        encoding="utf-8-sig",
    )
    raw.columns = [NAMES.get(normal(str(name)), str(name).strip()) for name in raw.columns]
    missing = [name for name in COLUMNS if name not in raw.columns]
    if missing:
        raise LayoutError(f"a.s.r. file lacks columns {missing}")
    raw = raw.astype(object).where(raw.notna(), None)
    for name in ("start_datetime", "end_datetime"):
        text = raw[name]
        written = text.astype("string")
        exact = written.str.fullmatch(TIME_PATTERN, flags=re.ASCII).fillna(False).astype(bool)
        matched = written.where(exact).str.replace(" ", "T", regex=False)
        parsed = pd.to_datetime(matched, format=TIME_FORMAT, errors="coerce")
        bad = text.notna() & parsed.isna()
        if bad.any():
            raise LayoutError(
                f"a.s.r. {name}: {int(bad.sum())} times cannot be read; field shapes: "
                + shapes_text(text[bad])
            )
        raw[name] = parsed
    if raw["start_datetime"].isna().any():
        count = int(raw["start_datetime"].isna().sum())
        raise LayoutError(f"a.s.r.: a row has no plug-in time (start_datetime): {count} rows")
    written = raw["total_energy"]
    text = written.astype("string")
    if sep == ";":
        text = text.str.replace(",", ".", regex=False)
    number = pd.to_numeric(text, errors="coerce")
    bad = text.notna() & number.isna()
    if bad.any():
        raise LayoutError(
            f"a.s.r. total_energy: {int(bad.sum())} values are not numbers; field shapes: "
            + shapes_text(written[bad])
        )
    raw["total_energy"] = number.astype(float)
    raw.index = pd.RangeIndex(1, len(raw) + 1)
    return raw


def to_sessions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The raw table to (session table, dropped rows)."""
    nat = pd.Series(pd.NaT, index=raw.index, dtype=f"datetime64[us, {ZONE}]")
    user = raw["EV_id_x"].map(lambda value: None if value is None else str(value))
    user = user.where(~user.isin(SHARED_IDS), None)
    table = pd.DataFrame(
        {
            "site": SITE,
            "role": CHECKED_AS,
            "session_id": [f"row-{n}" for n in raw.index],
            "user_id": user,
            "station": raw["evse_id"],
            "plug_in": localize(raw["start_datetime"], ZONE),
            "plug_out": localize(raw["end_datetime"], ZONE),
            "charge_start": nat,
            "charge_end": nat,
            "energy_kwh": raw["total_energy"],
            "max_power_kw": np.nan,
            "declared_departure": nat,
            "declared_energy_kwh": np.nan,
            "nominal_kw": NOMINAL_KW,
            "nominal_rule": "source cap",
        },
        index=raw.index,
    )
    dropper = Dropper(table["site"], table["session_id"], raw["start_datetime"].dt.date)
    first, gap, backwards, energy, duplicate, average_rule = REASONS
    table = dropper.drop(table, raw["end_datetime"].isna(), first)
    table = dropper.drop(table, table["plug_in"].isna() | table["plug_out"].isna(), gap)
    table = dropper.drop(table, table["plug_out"] <= table["plug_in"], backwards)
    table = dropper.drop(table, ~(table["energy_kwh"] > ENERGY_ABOVE_KWH), energy)
    repeated = raw.loc[table.index].astype(str).duplicated(keep="first")
    table = dropper.drop(table, repeated, duplicate)
    average = table["energy_kwh"] / hours_between(table["plug_out"], table["plug_in"])
    table = dropper.drop(table, average > LIMIT_KW, average_rule)
    sessions = finish(table)
    sessions["role"] = pd.Series(ROLE, index=sessions.index, dtype=object)
    return sessions, dropper.table()


def require_periods(sessions: pd.DataFrame, dropped: pd.DataFrame) -> None:
    """`LayoutError` for the first of `UNITS` with no row kept (reading A12), naming the period and
    the counts of rows read and removed by each reason (in the order the rules removed them)."""
    kept = periods(sessions)["unit"]
    gone = dropped["local_date"].map(unit_of)
    for unit in UNITS:
        if (kept == unit).any():
            continue
        removed = dropped.loc[(gone == unit).to_numpy(), "reason"].tolist()
        counts = ", ".join(
            f"{reason}: {removed.count(reason)}" for reason in dict.fromkeys(removed)
        )
        raise LayoutError(
            f"a.s.r. file: period {unit} has no row kept: {len(removed)} rows read"
            + (f", removed by reason {counts}" if removed else "")
        )


def from_bytes(content: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The a.s.r. session table and its dropped rows; `LayoutError` for a file it cannot read or
    in which a period has no row kept."""
    try:
        sessions, dropped = to_sessions(read_raw(content))
        require_periods(sessions, dropped)
        return sessions, dropped
    except LayoutError:
        raise
    except Exception as error:
        raise LayoutError(f"a.s.r. file cannot be read: {type(error).__name__}") from None


def unit_of(day: date) -> str:
    """The unit of a local date (reading A10): before or from 1 March 2023."""
    return UNITS[0] if day < CONTROLLER_FROM else UNITS[1]


def regime_of(day: date) -> str:
    """The regime of a local date (reading A10): free, paid, or paid with the controller."""
    if day < PAID_FROM:
        return REGIMES[0]
    return REGIMES[1] if day < CONTROLLER_FROM else REGIMES[2]


def periods(sessions: pd.DataFrame) -> pd.DataFrame:
    """Unit and regime of each session by the local date of its plug-in, on its index."""
    days = sessions["plug_in"].dt.date
    return pd.DataFrame(
        {"unit": days.map(unit_of), "regime": days.map(regime_of)}, index=sessions.index
    )
