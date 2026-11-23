"""The held-out Norwegian sites OSL_T and TRO_R into the common session table, from bytes.

Input: the bytes of the sealed split file that `tools/norway_split.py` writes for the held-out
sites (Zenodo 13896176, Dataset 1), handed over by the held-out loader's gated opener. This
module reads no file. The layout is the training and development split files' layout, which
`load_norway` reads: the same columns, ";" between fields, decimal comma, `NA` for a missing
value, times written "2019-11-28 19:08:00" in Europe/Oslo wall clock. Times are read exactly as
`load_norway` reads them, by `pandas.to_datetime` with `load_norway.TIME_FORMAT`; that reading
also accepts a month, day or hour written with one digit (as in "2019-1-2 8:00:00"), and a time
it cannot read raises `LayoutError`. The form is not made stricter here, so that the held-out
rows are read as the training and development rows of the same source file were.

Reused from `load_norway` by import: its column list, time format and number reading, and
`to_sessions`, which builds the table, removes rows by reason in its stated order and checks
the result with `sessions.finish`. `load_norway.read_raw` takes a path under `data/raw` only, so
the reading of the bytes is done here, by `read_raw` below, which follows `load_norway.read_raw`
step by step on an in-memory buffer, after the checks of text and records below and with the
check of a missing location added.

Sites: OSL_T and TRO_R only. A row with no location, and any other site code, a training or
development one included, raises `LayoutError`, as `load_norway` does for a site outside its
file's role (the row with no location is checked first, so it never meets the sorting of the
site codes, which cannot compare a missing location with a code).

Text and records: before `read_csv`, the bytes are checked as the a.s.r. loader checks its
file (`load_asr.decoded` and `load_asr.records`, its reading A11), with ";" between fields:
text that is not UTF-8, a NUL character,
quoting the csv module refuses in strict mode, no header record, a record whose number of
fields is not the header's, and a column name that appears twice in the header raise
`LayoutError`. These checks only refuse: a file that passes them is read exactly as
`load_norway` reads its files. [Alternative: `read_csv` alone, as `load_norway` does, which
pads a short record with missing values and, when the first record has one field more than the
header, takes the first column as the index.]

Units: each of the two sites must have a row kept. A site with no row in the file, or whose rows
are all removed by the reasons of `load_norway.to_sessions`, raises `LayoutError` whose message
gives the site, the number of rows read and the number removed by each reason, never a value of
a row: a site that is named and has no session is a fault in reading the file, not a result about
the site, so the refusal is logged and the parsing-fix route of plan section 13 applies.

Errors, as a class: `from_bytes` raises `LayoutError` for every file it cannot read. The checks
of this module raise it with their own messages; any other `Exception` raised while the bytes
are read or the table is built is raised as `LayoutError` with that exception as its cause. A
file that is read gives rows kept, or removed by `load_norway.to_sessions`'s stated reasons,
and nothing else.

Role: `sessions.finish` accepts only the roles train and development, so the rows go through
`load_norway.to_sessions` under the role development and the role column of the result is then
set to "held-out". Every other column is as `load_norway` gives it.
"""

import io

import pandas as pd

from fleetreserve import load_asr, load_norway
from fleetreserve.sessions import LayoutError

SITES = ("OSL_T", "TRO_R")
ROLE = "held-out"
CHECKED_AS = "development"
WHAT = "Norway held-out file"


def read_raw(content: bytes) -> pd.DataFrame:
    """The file's rows with times parsed to naive local times and numbers read."""
    header = load_asr.records(load_asr.decoded(content, "utf-8", WHAT), ";", WHAT)[0]
    repeated = sorted({name for name in header if header.count(name) > 1})
    if repeated:
        raise LayoutError(f"{WHAT} names columns more than once: {repeated}")
    raw = pd.read_csv(
        io.BytesIO(content), sep=";", dtype=str, keep_default_na=False, na_values=["", "NA"]
    )
    columns = list(load_norway.COLUMNS)
    missing = [name for name in columns if name not in raw.columns]
    if missing:
        raise LayoutError(f"Norway held-out file lacks columns {missing}")
    raw = raw[columns].astype(object).where(raw[columns].notna(), None)
    if raw["location"].isna().any():
        raise LayoutError("Norway held-out file: a row has no location")
    unknown = sorted(set(raw["location"]) - set(SITES))
    if unknown:
        raise LayoutError(f"Norway file holds sites outside the held-out sites: {unknown}")
    for name in ("plugin_time", "plugout_time"):
        text = raw[name]
        parsed = pd.to_datetime(text, format=load_norway.TIME_FORMAT, errors="coerce")
        bad = text.notna() & parsed.isna()
        if bad.any():
            raise LayoutError(f"Norway {name}: {int(bad.sum())} times cannot be read")
        raw[name] = parsed
    if raw["plugin_time"].isna().any() or raw["session_id"].isna().any():
        raise LayoutError("Norway held-out file: a row has no plug-in time or session id")
    for name in ("connection_time", "energy_session"):
        raw[name] = load_norway._number(raw[name].astype("string"), name)
    raw.index = pd.RangeIndex(1, len(raw) + 1)
    return raw


def require_sites(sessions: pd.DataFrame, dropped: pd.DataFrame) -> None:
    """`LayoutError` for the first of `SITES` with no row kept, naming the site and the counts of
    rows read and removed by each reason (in the order the rules removed them)."""
    for site in SITES:
        if (sessions["site"] == site).any():
            continue
        removed = dropped.loc[dropped["site"] == site, "reason"].tolist()
        counts = ", ".join(
            f"{reason}: {removed.count(reason)}" for reason in dict.fromkeys(removed)
        )
        raise LayoutError(
            f"{WHAT}: site {site} has no row kept: {len(removed)} rows read"
            + (f", removed by reason {counts}" if removed else "")
        )


def from_bytes(content: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The session table of the held-out Norwegian sites and its dropped rows; `LayoutError`
    for a file it cannot read or in which a site has no row kept."""
    try:
        sessions, dropped = load_norway.to_sessions(CHECKED_AS, read_raw(content))
        require_sites(sessions, dropped)
    except LayoutError:
        raise
    except Exception as error:
        problem = f"{type(error).__name__}: {error}"
        raise LayoutError(f"{WHAT} cannot be read: {problem}") from error
    sessions["role"] = pd.Series(ROLE, index=sessions.index, dtype=object)
    return sessions, dropped
