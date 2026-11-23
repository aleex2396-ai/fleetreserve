"""NREL workplace sessions (two sites, pooled; development) into the common session table.

Layout read: `data/raw/nrel/dataSet_Oct2021.csv` (data.nlr.gov 182), comma-separated, quoted,
with `NULL` for a missing value. Columns used: `driverId`, `request_entry_time`,
`expected_departure`, `station`, `start_date_time`, `end_date_time`, `start_charge`,
`termin_charge`, `max_charge_power`, `kwh_requested`, `energy_charged`. Times are local wall
clock, "2016-11-03 10:01:03"; the clock is Mountain time with daylight saving
(America/Denver), assumed (plan section 3). Energy is `energy_charged` in kWh, assumed.

The file has no session id: a session's id is `row-<n>`, n its data row's number in the file
(the first data row is 1).

Plug-in and plug-out are `start_date_time` and `end_date_time`. A reading: these are the only
fields that record a plug-in and plug-out; `start_charge` and `termin_charge` record when
charging started and ended, and `request_entry_time` when the driver entered a request. The
file has `start_date_time` and `end_date_time` only from April 2019; earlier rows are removed
under `no plug-in or plug-out time recorded`. For counting the effect of other readings only,
`missing_plug_out` can fill them instead: plug-in from `request_entry_time`, plug-out from
`termin_charge` (`"charge_end"`) or from `expected_departure` (`"declared_departure"`).

Declared departure: `expected_departure`, stated at `request_entry_time` (`declared_at`; the
file has one request time per row, so a later change of the request, if any, is not visible).
`plug_out_from_declared` marks the rows whose plug-out was taken from `expected_departure`: only
under `missing_plug_out="declared_departure"`, and there exactly the rows without a recorded
`start_date_time` or `end_date_time` (both plug-in and plug-out are then filled). Under the main
reading (`missing_plug_out=None`) and under `"charge_end"` no row is marked.

Nominal cap (plan section 3): `max_charge_power` floored at 3.3 kW; a session without
`max_charge_power` gets 3.3 kW, marked in `nominal_rule`.

Rows removed, by reason, in this order:
1. `no plug-in or plug-out time recorded`.
2. `local time in the daylight-saving gap or doubled hour` (plug-in or plug-out).
3. `plug-out not after plug-in`.
4. `energy missing or not positive`.
5. `duplicate row`: a row equal in every column to an earlier one.
6. `fast-charging station` and 7. `average above the physical limit`, as for ACN (19.2 kW).

A missing column or an unreadable time raises `LayoutError`.
"""

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
    localize,
    require_raw_path,
)

FILE = Path("nrel") / "dataSet_Oct2021.csv"
SITE = "nrel"
ROLE = "development"
ZONE = "America/Denver"
FLOOR_KW = 3.3
LIMIT_KW = PHYSICAL_LIMIT_KW["nrel"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
COLUMNS = (
    "driverId",
    "vehicle_model",
    "request_entry_time",
    "expected_departure",
    "station",
    "start_date_time",
    "end_date_time",
    "start_charge",
    "termin_charge",
    "controlled_duration",
    "miles_requested",
    "max_charge_power",
    "kwh_requested",
    "energy_charged",
    "cost_for_session",
    "afterPaid",
)
TIMES = (
    "request_entry_time",
    "expected_departure",
    "start_date_time",
    "end_date_time",
    "start_charge",
    "termin_charge",
)
NUMBERS = ("max_charge_power", "kwh_requested", "energy_charged")
PLUG_OUT_FALLBACKS = (None, "charge_end", "declared_departure")


def read_raw(raw_root: Path = RAW_ROOT) -> pd.DataFrame:
    """The file as text columns, `NULL` read as missing; times parsed to naive local times."""
    path = require_raw_path(Path(raw_root) / FILE, raw_root)
    raw = pd.read_csv(path, dtype=str, na_values=["NULL"], keep_default_na=False)
    missing = [name for name in COLUMNS if name not in raw.columns]
    if missing:
        raise LayoutError(f"NREL file lacks columns {missing}")
    raw = raw.astype(object).where(raw.notna(), None)
    for name in TIMES:
        text = raw[name]
        parsed = pd.to_datetime(text, format=TIME_FORMAT, errors="coerce")
        bad = text.notna() & parsed.isna()
        if bad.any():
            raise LayoutError(f"NREL {name}: {int(bad.sum())} times cannot be read")
        raw[name] = parsed
    for name in NUMBERS:
        text = raw[name]
        number = pd.to_numeric(text, errors="coerce")
        bad = text.notna() & number.isna()
        if bad.any():
            raise LayoutError(f"NREL {name}: {int(bad.sum())} values are not numbers")
        raw[name] = number.astype(float)
    if raw["request_entry_time"].isna().any() or raw["driverId"].isna().any():
        raise LayoutError("NREL: a row has no request_entry_time or driverId")
    raw.index = pd.RangeIndex(1, len(raw) + 1)
    return raw


def to_sessions(
    raw: pd.DataFrame, missing_plug_out: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The raw table to (session table, dropped rows)."""
    if missing_plug_out not in PLUG_OUT_FALLBACKS:
        raise ValueError(f"missing_plug_out must be one of {PLUG_OUT_FALLBACKS}")
    plug_in_naive = raw["start_date_time"]
    plug_out_naive = raw["end_date_time"]
    from_declared = pd.Series(False, index=raw.index)
    if missing_plug_out is not None:
        no_times = plug_in_naive.isna() | plug_out_naive.isna()
        if missing_plug_out == "declared_departure":
            from_declared = no_times
        source = "termin_charge" if missing_plug_out == "charge_end" else "expected_departure"
        plug_in_naive = plug_in_naive.where(~no_times, raw["request_entry_time"])
        plug_out_naive = plug_out_naive.where(~no_times, raw[source])
    power = raw["max_charge_power"]
    has_power = power.notna()
    table = pd.DataFrame(
        {
            "site": SITE,
            "role": ROLE,
            "session_id": [f"row-{n}" for n in raw.index],
            "user_id": raw["driverId"].astype(str),
            "station": raw["station"],
            "plug_in": localize(plug_in_naive, ZONE),
            "plug_out": localize(plug_out_naive, ZONE),
            "charge_start": localize(raw["start_charge"], ZONE),
            "charge_end": localize(raw["termin_charge"], ZONE),
            "energy_kwh": raw["energy_charged"],
            "max_power_kw": power,
            "declared_departure": localize(raw["expected_departure"], ZONE),
            "declared_energy_kwh": raw["kwh_requested"],
            "nominal_kw": np.where(has_power, np.maximum(power.fillna(0.0), FLOOR_KW), FLOOR_KW),
            "nominal_rule": np.where(
                has_power,
                "max_charge_power floored at 3.3 kW",
                "3.3 kW floor: max_charge_power missing",
            ),
            "declared_at": localize(raw["request_entry_time"], ZONE),
            "plug_out_from_declared": from_declared,
        },
        index=raw.index,
    )
    dropper = Dropper(table["site"], table["session_id"], raw["request_entry_time"].dt.date)
    no_record = plug_in_naive.isna() | plug_out_naive.isna()
    table = dropper.drop(table, no_record, "no plug-in or plug-out time recorded")
    in_gap = table["plug_in"].isna() | table["plug_out"].isna()
    table = dropper.drop(table, in_gap, "local time in the daylight-saving gap or doubled hour")
    table = dropper.drop(table, table["plug_out"] <= table["plug_in"], "plug-out not after plug-in")
    table = dropper.drop(table, ~(table["energy_kwh"] > 0), "energy missing or not positive")
    repeated = raw.loc[table.index, list(COLUMNS)].astype(str).duplicated(keep="first")
    table = dropper.drop(table, repeated, "duplicate row")
    average = table["energy_kwh"] / hours_between(table["plug_out"], table["plug_in"])
    station_median = average.groupby(table["station"].fillna("")).transform("median")
    table = dropper.drop(table, station_median > LIMIT_KW, "fast-charging station")
    average = table["energy_kwh"] / hours_between(table["plug_out"], table["plug_in"])
    table = dropper.drop(table, average > LIMIT_KW, "average above the physical limit")
    return finish(table), dropper.table()


def load(
    raw_root: Path = RAW_ROOT, missing_plug_out: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The NREL session table and its dropped rows."""
    return to_sessions(read_raw(raw_root), missing_plug_out)
