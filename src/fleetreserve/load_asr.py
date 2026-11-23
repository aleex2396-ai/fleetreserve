"""The a.s.r. Utrecht office car park (held out) into the common session table, from bytes.

Source: 4TU 80ef3824-3f5d-4e45-8794-3b8791efbd13, file `202410DatasetEVOfficeParking_v0.csv`,
handed over as bytes by the held-out loader's gated opener. This module reads no file.

Layout read from the data set's README only (no row of the file has been seen): one row per
charging session; columns `EV_id_x` (EV identifier pseudonym, from the registered charging ID),
`start_datetime` and `end_datetime` (charger connection and disconnection, written
YYYY-MM-DDThh:mm:ss, local time, UTC+01:00 in winter and UTC+02:00 in summer), `total_energy`
(kWh charged in the session), `rail`, `evse_id` (charging station identifier), `channel` (socket
1 or 2), and survey columns. The session table is filled from the columns `COLUMNS` only; they
are found by name, and a missing one raises `LayoutError`. The other columns of the file are
read too, as text, for one use: the duplicate rule (5 below) compares every column of the file.

Readings of the README, each with its alternative in brackets:

  A1. Field separator: the first of "," and ";" whose split of the header line holds every name
      in `COLUMNS` (the README names no separator). Under ";" a decimal comma in `total_energy`
      is read as a decimal point, as in the Norway files; under "," the decimal mark is ".".
      A byte order mark at the start of the file is skipped. [Alternative: one fixed
      separator.]
  A2. Missing value: an empty field, `NA`, `NaN` or `nan` (the README says values are missing
      but not how they are written). [Alternative: another marker, which would then be read as
      text and, in a time or number column, raise `LayoutError`.]
  A3. Times: exactly YYYY-MM-DDThh:mm:ss with every field zero-padded (the whole text must match
      `TIME_PATTERN`, ASCII digits only), Europe/Amsterdam wall clock. Any other form raises
      `LayoutError`: one-digit fields, an offset or "Z", fractions of a second, a space for "T",
      a leading or trailing space, a time without seconds. [Alternative: times in UTC or
      carrying an offset.]
  A4. Plug-in and plug-out are `start_datetime` and `end_datetime` ("charger connection" and
      "disconnection"). The README names no separate charging start or end, so `charge_start`
      and `charge_end` are missing.
  A5. `total_energy` is the session's energy in kWh, as plan section 3 states.
  A6. User: `EV_id_x`. The README lists twenty charging IDs that "cannot be attributed to a
      unique user" (`SHARED_IDS`); their sessions get a missing user. [Alternative: keep them as
      users.] The README says "at least" these IDs, so other shared IDs, if any, stay users.
  A7. Station: `evse_id` as text, missing where `evse_id` is missing (the README: Rail and EVSE
      information is not provided for all data points); `channel` does not enter the station
      (it counts only in the duplicate rule). [Alternative: one identifier per socket,
      `evse_id` with `channel`.]
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
import re
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
TIME_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"  # the whole text, ASCII digits only
SEPARATORS = (",", ";")
MISSING = ["", "NA", "NaN", "nan"]
COLUMNS = ("EV_id_x", "start_datetime", "end_datetime", "total_energy", "evse_id")
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


def separator(text: str) -> str:
    """The first of `SEPARATORS` whose split of the header record names every column read."""
    for sep in SEPARATORS:
        lines = io.StringIO(text, newline="")
        names = next((row for row in csv.reader(lines, delimiter=sep) if row), [])
        if set(COLUMNS) <= {name.strip() for name in names}:
            return sep
    raise LayoutError("a.s.r. file lacks columns: no separator gives " + ", ".join(COLUMNS))


def read_raw(content: bytes) -> pd.DataFrame:
    """The file as text columns, the missing markers read as missing; times parsed to naive
    local times and the energy read as a number."""
    text = decoded(content, "utf-8-sig", "a.s.r. file")
    sep = separator(text)
    names = [name.strip() for name in records(text, sep, "a.s.r. file")[0]]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise LayoutError(f"a.s.r. file names columns more than once: {repeated}")
    raw = pd.read_csv(
        io.BytesIO(content),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        na_values=MISSING,
        encoding="utf-8-sig",
    )
    raw.columns = [str(name).strip() for name in raw.columns]
    missing = [name for name in COLUMNS if name not in raw.columns]
    if missing:
        raise LayoutError(f"a.s.r. file lacks columns {missing}")
    raw = raw.astype(object).where(raw.notna(), None)
    for name in ("start_datetime", "end_datetime"):
        text = raw[name]
        written = text.astype("string")
        exact = written.str.fullmatch(TIME_PATTERN, flags=re.ASCII).fillna(False).astype(bool)
        parsed = pd.to_datetime(written.where(exact), format=TIME_FORMAT, errors="coerce")
        bad = text.notna() & parsed.isna()
        if bad.any():
            raise LayoutError(f"a.s.r. {name}: {int(bad.sum())} times cannot be read")
        raw[name] = parsed
    if raw["start_datetime"].isna().any():
        raise LayoutError("a.s.r.: a row has no plug-in time (start_datetime)")
    text = raw["total_energy"].astype("string")
    if sep == ";":
        text = text.str.replace(",", ".", regex=False)
    number = pd.to_numeric(text, errors="coerce")
    bad = text.notna() & number.isna()
    if bad.any():
        raise LayoutError(f"a.s.r. total_energy: {int(bad.sum())} values are not numbers")
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
        problem = f"{type(error).__name__}: {error}"
        raise LayoutError(f"a.s.r. file cannot be read: {problem}") from error


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
