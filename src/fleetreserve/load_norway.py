"""Norway residential sessions (training and development sites) into the common session table.

Layout read: the split outputs `data/raw/norway/Dataset1_charging_reports-train.csv` and
`...-development.csv` written by `tools/norway_split.py` from Zenodo 13896176, Dataset 1. The
data paper's Table 1 gives the columns: `location`, `session_id`, `user_id`, `plugin_time`,
`plugout_time`, `connection_time` (hours, decimal comma), `energy_session` (kWh, decimal
comma). Fields are separated by ";", times are written "2019-11-28 19:08:00" and a missing
value is written NA. The paper
states Central European Time with daylight saving; times are read as wall clock in
Europe/Oslo.

Sites per file (plan section 3): train OSL_S, ASK, TRO, BAR, BOD; development OSL_1, BER, KRO,
OSL_2, BAR_2. A site code outside the file's role raises `LayoutError`.

Nominal cap: 7.4 kW for every site. A reading of plan section 3 ("Norway mostly 7.4 kW"): the
data paper gives no per-site value, only that for most users the charge point's maximum was
7.4 kW (32 A).

Rows removed, by reason, in this order:
1. `no plug-out time recorded` (the paper sets plug-out to missing for sessions it judged
   wrong).
2. `local time in the daylight-saving gap or doubled hour` (plug-in or plug-out).
3. `plug-out not after plug-in`.
4. `energy missing or not positive`.
5. `duplicate session id` (the first is kept).
6. `average above the physical limit`: energy over plug-in to plug-out above 22 kW.

A missing column, an unreadable time or number, or an unknown site raises `LayoutError`.
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

FILES = {
    "train": Path("norway") / "Dataset1_charging_reports-train.csv",
    "development": Path("norway") / "Dataset1_charging_reports-development.csv",
}
SITES = {
    "train": ("OSL_S", "ASK", "TRO", "BAR", "BOD"),
    "development": ("OSL_1", "BER", "KRO", "OSL_2", "BAR_2"),
}
ZONE = "Europe/Oslo"
NOMINAL_KW = 7.4
LIMIT_KW = PHYSICAL_LIMIT_KW["norway"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
COLUMNS = (
    "location",
    "user_id",
    "session_id",
    "plugin_time",
    "plugout_time",
    "connection_time",
    "energy_session",
)


def _number(text: pd.Series, what: str) -> pd.Series:
    cleaned = text.str.replace(",", ".", regex=False)
    number = pd.to_numeric(cleaned, errors="coerce")
    bad = text.notna() & number.isna()
    if bad.any():
        raise LayoutError(f"Norway {what}: {int(bad.sum())} values are not numbers")
    return number.astype(float)


def read_raw(role: str, raw_root: Path = RAW_ROOT) -> pd.DataFrame:
    """One role's file with times parsed to naive local times and numbers read."""
    if role not in FILES:
        raise LayoutError(f"unknown role {role!r}")
    path = require_raw_path(Path(raw_root) / FILES[role], raw_root)
    raw = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, na_values=["", "NA"])
    missing = [name for name in COLUMNS if name not in raw.columns]
    if missing:
        raise LayoutError(f"Norway {role} file lacks columns {missing}")
    raw = raw[list(COLUMNS)].astype(object).where(raw[list(COLUMNS)].notna(), None)
    unknown = sorted(set(raw["location"]) - set(SITES[role]))
    if unknown:
        raise LayoutError(f"Norway {role} file holds sites outside its role: {unknown}")
    for name in ("plugin_time", "plugout_time"):
        text = raw[name]
        parsed = pd.to_datetime(text, format=TIME_FORMAT, errors="coerce")
        bad = text.notna() & parsed.isna()
        if bad.any():
            raise LayoutError(f"Norway {name}: {int(bad.sum())} times cannot be read")
        raw[name] = parsed
    if raw["plugin_time"].isna().any() or raw["session_id"].isna().any():
        raise LayoutError(f"Norway {role}: a row has no plug-in time or session id")
    raw["connection_time"] = _number(raw["connection_time"].astype("string"), "connection_time")
    raw["energy_session"] = _number(raw["energy_session"].astype("string"), "energy_session")
    raw.index = pd.RangeIndex(1, len(raw) + 1)
    return raw


def to_sessions(role: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nat = pd.Series(pd.NaT, index=raw.index, dtype=f"datetime64[us, {ZONE}]")
    table = pd.DataFrame(
        {
            "site": raw["location"],
            "role": role,
            "session_id": raw["session_id"],
            "user_id": raw["user_id"],
            "station": None,
            "plug_in": localize(raw["plugin_time"], ZONE),
            "plug_out": localize(raw["plugout_time"], ZONE),
            "charge_start": nat,
            "charge_end": nat,
            "energy_kwh": raw["energy_session"],
            "max_power_kw": np.nan,
            "declared_departure": nat,
            "declared_energy_kwh": np.nan,
            "nominal_kw": NOMINAL_KW,
            "nominal_rule": "source cap",
        },
        index=raw.index,
    )
    dropper = Dropper(table["site"], table["session_id"], raw["plugin_time"].dt.date)
    table = dropper.drop(table, raw["plugout_time"].isna(), "no plug-out time recorded")
    in_gap = table["plug_in"].isna() | table["plug_out"].isna()
    table = dropper.drop(table, in_gap, "local time in the daylight-saving gap or doubled hour")
    table = dropper.drop(table, table["plug_out"] <= table["plug_in"], "plug-out not after plug-in")
    table = dropper.drop(table, ~(table["energy_kwh"] > 0), "energy missing or not positive")
    repeated = table.duplicated(["site", "session_id"], keep="first")
    table = dropper.drop(table, repeated, "duplicate session id")
    average = table["energy_kwh"] / hours_between(table["plug_out"], table["plug_in"])
    table = dropper.drop(table, average > LIMIT_KW, "average above the physical limit")
    return finish(table), dropper.table()


def load(role: str, raw_root: Path = RAW_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The session table of one role's Norway sites and its dropped rows."""
    return to_sessions(role, read_raw(role, raw_root))
