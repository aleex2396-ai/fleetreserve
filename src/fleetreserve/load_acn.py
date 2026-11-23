"""ACN-Data sessions (Caltech, JPL, office001; training sites) into the common session table.

Layout read: `data/raw/acn/<site>/<YYYY-MM>.json.gz`, each a gzip-compressed JSON list of the
ACN-Data API's session records (Lee et al. 2019), as `tools/fetch_acn.py` stores them. Fields
used: `sessionID`, `connectionTime` (plug-in), `disconnectTime` (plug-out), `doneChargingTime`
(end of charging), `kWhDelivered`, `userID`, `stationID`, `timezone`, `userInputs`. Times are
written in UTC ("Wed, 01 May 2019 01:18:34 GMT") and converted to the site's local time,
America/Los_Angeles; every record's own `timezone` field must name that zone.

Declared departure and energy: the first entry of `userInputs` by its `modifiedAt` time
(`requestedDeparture`, `kWhRequested`); later edits are not used here. The time the declared
departure was stated, `declared_at`, is that entry's `modifiedAt`; it is missing when the record
has no entry or no entry's `modifiedAt` can be read (the entry used is then the list's first, and
`truth` does not use its declared departure). Nominal cap: 6.66 kW (plan section 3).

Rows removed, by reason, in this order:
1. `plug-out not after plug-in`.
2. `energy missing or not positive`.
3. `duplicate session id` (the first is kept).
4. `fast-charging station`: every session at a station whose median session average power
   (energy over plug-in to plug-out) is above the physical limit of a Level 2 charger,
   19.2 kW. A reading: the plan's 6.66 kW nominal cap describes Level 2 chargers only.
5. `average above the physical limit`: a session whose energy over its plug-in to plug-out
   time is above 19.2 kW.

A missing field, an unreadable time, a wrong time zone or a file name that is not a month
raises `LayoutError`; nothing is guessed.
"""

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve.sessions import (
    PHYSICAL_LIMIT_KW,
    RAW_ROOT,
    Dropper,
    LayoutError,
    finish,
    hours_between,
    require_raw_path,
)

SITES = ("caltech", "jpl", "office001")
ROLE = "train"
ZONE = "America/Los_Angeles"
NOMINAL_KW = 6.66
LIMIT_KW = PHYSICAL_LIMIT_KW["acn"]
TIME_FORMAT = "%a, %d %b %Y %H:%M:%S GMT"
FIELDS = (
    "sessionID",
    "connectionTime",
    "disconnectTime",
    "doneChargingTime",
    "kWhDelivered",
    "userID",
    "stationID",
    "timezone",
    "userInputs",
)
_MONTH_FILE = re.compile(r"\d{4}-\d{2}\.json\.gz")


def month_files(site: str, raw_root: Path = RAW_ROOT) -> list[Path]:
    if site not in SITES:
        raise LayoutError(f"unknown ACN site {site!r}")
    folder = require_raw_path(Path(raw_root) / "acn" / site, raw_root)
    if not folder.is_dir():
        raise FileNotFoundError(f"no ACN folder for {site}: {folder}")
    files = sorted(folder.iterdir())
    names = [path.name for path in files]
    odd = [name for name in names if not _MONTH_FILE.fullmatch(name)]
    if odd:
        raise LayoutError(f"files in the ACN {site} folder that are not month files: {odd[:5]}")
    if not files:
        raise LayoutError(f"the ACN {site} folder holds no month files")
    return [require_raw_path(path, raw_root) for path in files]


def read_records(site: str, raw_root: Path = RAW_ROOT) -> list[dict]:
    records: list[dict] = []
    for path in month_files(site, raw_root):
        with gzip.open(path, "rb") as handle:
            items = json.loads(handle.read())
        if not isinstance(items, list):
            raise LayoutError(f"{path.name}: not a list of session records")
        for item in items:
            missing = [name for name in FIELDS if name not in item]
            if missing:
                raise LayoutError(f"{site} {path.name}: a record lacks fields {missing}")
        records.extend(items)
    return records


def parse_utc(values: pd.Series, what: str, *, allow_missing: bool) -> pd.Series:
    """UTC time strings to local times; an unreadable one raises LayoutError."""
    text = values.astype(object)
    present = text.notna()
    if not allow_missing and not present.all():
        raise LayoutError(f"{what}: {int((~present).sum())} records have no time")
    parsed = pd.to_datetime(text, format=TIME_FORMAT, utc=True, errors="coerce")
    bad = present & parsed.isna()
    if bad.any():
        raise LayoutError(f"{what}: {int(bad.sum())} times cannot be read")
    return parsed.dt.tz_convert(ZONE)


def _text(values: pd.Series) -> pd.Series:
    return pd.Series(
        [None if value is None or value != value else str(value) for value in values],
        index=values.index,
        dtype=object,
    )


def _first_entries(inputs: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """The first app entry's requested departure (text), energy and `modifiedAt` (UTC, NaT when
    unreadable), per record."""
    departure, energy, stated = [], [], []
    for entries in inputs:
        if not entries:
            departure.append(None)
            energy.append(np.nan)
            stated.append(pd.NaT)
            continue
        if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
            raise LayoutError("userInputs is not a list of entries")
        stamps = pd.to_datetime(
            pd.Series([e.get("modifiedAt") for e in entries], dtype=object),
            format=TIME_FORMAT,
            utc=True,
            errors="coerce",
        )
        # The earliest readable time; with none readable, the list's first entry.
        position = int(np.argmin(stamps.rank(method="first", na_option="bottom").to_numpy()))
        first = entries[position]
        departure.append(first.get("requestedDeparture"))
        value = first.get("kWhRequested")
        energy.append(np.nan if value is None else float(value))
        stated.append(stamps.iloc[position])
    return (
        pd.Series(departure, index=inputs.index, dtype=object),
        pd.Series(energy, index=inputs.index, dtype=float),
        pd.Series(pd.to_datetime(stated, utc=True), index=inputs.index),
    )


def to_sessions(site: str, records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Records of one site to (session table, dropped rows)."""
    raw = pd.DataFrame.from_records(records, columns=list(FIELDS))
    zones = set(raw["timezone"].astype(object))
    if zones != {ZONE}:
        raise LayoutError(f"{site}: time zone field holds {sorted(map(str, zones))}, not {ZONE}")
    plug_in = parse_utc(raw["connectionTime"], f"{site} connectionTime", allow_missing=False)
    plug_out = parse_utc(raw["disconnectTime"], f"{site} disconnectTime", allow_missing=False)
    done = parse_utc(raw["doneChargingTime"], f"{site} doneChargingTime", allow_missing=True)
    energy = pd.to_numeric(raw["kWhDelivered"], errors="coerce")
    declared_text, declared_energy, stated = _first_entries(raw["userInputs"])
    declared = parse_utc(declared_text, f"{site} requestedDeparture", allow_missing=True)
    nat = pd.Series(pd.NaT, index=raw.index, dtype=plug_in.dtype)
    table = pd.DataFrame(
        {
            "site": site,
            "role": ROLE,
            "session_id": _text(raw["sessionID"]),
            "user_id": _text(raw["userID"]),
            "station": _text(raw["stationID"]),
            "plug_in": plug_in,
            "plug_out": plug_out,
            "charge_start": nat,
            "charge_end": done,
            "energy_kwh": energy.astype(float),
            "max_power_kw": np.nan,
            "declared_departure": declared,
            "declared_energy_kwh": declared_energy,
            "nominal_kw": NOMINAL_KW,
            "nominal_rule": "source cap",
            "declared_at": stated.dt.tz_convert(ZONE),
            "plug_out_from_declared": False,
        }
    )
    if table["session_id"].isna().any():
        raise LayoutError(f"{site}: a record has no session id")
    dropper = Dropper(table["site"], table["session_id"], plug_in.dt.date)
    table = dropper.drop(table, table["plug_out"] <= table["plug_in"], "plug-out not after plug-in")
    table = dropper.drop(table, ~(table["energy_kwh"] > 0), "energy missing or not positive")
    table = dropper.drop(
        table, table["session_id"].duplicated(keep="first"), "duplicate session id"
    )
    average = table["energy_kwh"] / hours_between(table["plug_out"], table["plug_in"])
    station_median = average.groupby(table["station"].fillna("")).transform("median")
    table = dropper.drop(table, station_median > LIMIT_KW, "fast-charging station")
    average = table["energy_kwh"] / hours_between(table["plug_out"], table["plug_in"])
    table = dropper.drop(table, average > LIMIT_KW, "average above the physical limit")
    return finish(table), dropper.table()


def load(site: str, raw_root: Path = RAW_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The session table of one ACN site and its dropped rows."""
    return to_sessions(site, read_records(site, raw_root))
