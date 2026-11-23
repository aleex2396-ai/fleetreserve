"""The common session table every loader returns, and the rules all loaders share.

One row per charging session. Columns, in this order:

- `site`: the site code (`caltech`, `jpl`, `office001`; `nrel` for the two NREL sites, which
  the plan pools; the Norway location codes such as `OSL_S`).
- `role`: `train` or `development` (plan section 3).
- `session_id`: the source's session identifier, as text, unique within the site.
- `user_id`: the source's user identifier as text, or missing. In the Norway data it names a
  charge point user account (for example `ASK_Plass15`).
- `station`: the charger's identifier as text, or missing (Norway has none).
- `plug_in`, `plug_out`: plug-in and plug-out, timezone-aware, in the site's local time.
- `charge_start`, `charge_end`: when charging started and ended, where the source records them
  (ACN `doneChargingTime` as `charge_end`; NREL `start_charge`, `termin_charge`); else missing.
- `energy_kwh`: the session's energy in kWh (ACN `kWhDelivered`, NREL `energy_charged`, Norway
  `energy_session`).
- `max_power_kw`: NREL `max_charge_power` (the highest power measured in the session); else
  missing.
- `declared_departure`, `declared_energy_kwh`: what the driver declared (ACN first app entry,
  NREL `expected_departure` and `kwh_requested`); else missing.
- `nominal_kw`: the nominal rate cap of plan section 3: ACN 6.66 kW; NREL `max_charge_power`
  floored at 3.3 kW; Norway 7.4 kW; a.s.r. 11 kW; Korea 7 kW (assumed).
- `nominal_rule`: how `nominal_kw` was set (`source cap`, `max_charge_power floored at 3.3 kW`,
  `3.3 kW floor: max_charge_power missing`, or, for Korea, `7 kW assumed: no kW in the data`).

Optional columns (`OPTIONAL_COLUMNS`), kept after the others when a loader gives them (ACN and
NREL do; Norway, which has no declared values, does not):

- `declared_at`: when the declared departure was stated, timezone-aware (ACN `modifiedAt` of the
  first app entry, NREL `request_entry_time`); missing when not known. `truth` uses a declared
  departure only when it was stated at or before the day's decision time.
- `plug_out_from_declared`: true when the session's plug-out was taken from the declared
  departure (only NREL's counting reading `missing_plug_out="declared_departure"` does this).

Times of one table share one time zone (one table per source). Rows the loaders remove are never
dropped silently: each loader also returns a table of dropped rows with `site`, `session_id`
(missing when the row has none), `reason` and `local_date` (the local date of the row's earliest
recorded time, used to place the drop in a unit and period).
"""

from pathlib import Path

import numpy as np
import pandas as pd

TOP = Path(__file__).resolve().parents[2]
RAW_ROOT = TOP / "data" / "raw"

COLUMNS = (
    "site",
    "role",
    "session_id",
    "user_id",
    "station",
    "plug_in",
    "plug_out",
    "charge_start",
    "charge_end",
    "energy_kwh",
    "max_power_kw",
    "declared_departure",
    "declared_energy_kwh",
    "nominal_kw",
    "nominal_rule",
)
TIME_COLUMNS = ("plug_in", "plug_out", "charge_start", "charge_end", "declared_departure")
TEXT_COLUMNS = ("site", "role", "session_id", "user_id", "station", "nominal_rule")
NUMBER_COLUMNS = ("energy_kwh", "max_power_kw", "declared_energy_kwh", "nominal_kw")
DECLARED_AT = "declared_at"
FROM_DECLARED = "plug_out_from_declared"
OPTIONAL_COLUMNS = (DECLARED_AT, FROM_DECLARED)
DROP_COLUMNS = ("site", "session_id", "reason", "local_date")
ROLES = ("train", "development")

# Highest average power an AC charger can give, used by the drop rule `average above the
# physical limit` and to find fast-charging stations. A reading of plan section 3 ("WP2 checks
# ... units"): US Level 2 at most 80 A at 240 V = 19.2 kW (ACN, NREL); Europe three-phase 32 A at
# 230 V = 22 kW (Norway).
PHYSICAL_LIMIT_KW = {"acn": 19.2, "nrel": 19.2, "norway": 22.0}


class LayoutError(ValueError):
    """The file does not have the published layout (a column is missing, a time is unreadable)."""


class OutsideRawData(PermissionError):
    """A loader was asked to read a path outside the raw data folder."""


def require_raw_path(path: Path, raw_root: Path = RAW_ROOT) -> Path:
    """The resolved path, if it lies inside `raw_root`, a folder `data/raw`; else refuse.

    Symbolic links are resolved first, so a link inside `data/raw` that points elsewhere is
    refused too.
    """
    root = Path(raw_root).resolve()
    if root.parts[-2:] != ("data", "raw"):
        raise OutsideRawData(f"the raw data root must be a folder data/raw, not {raw_root}")
    resolved = Path(path).resolve()
    if resolved != root and root not in resolved.parents:
        raise OutsideRawData(f"loaders read only under data/raw; refused: {path}")
    return resolved


def hours_between(later: pd.Series, earlier: pd.Series) -> np.ndarray:
    """Elapsed hours (real time, across daylight-saving changes) from `earlier` to `later`."""
    return ((later - earlier) / pd.Timedelta(hours=1)).to_numpy(dtype=float, na_value=np.nan)


class Dropper:
    """Removes rows by named reason, in order, and keeps every removed row's record."""

    def __init__(self, site: pd.Series, session_id: pd.Series, local_date: pd.Series):
        self._site = site
        self._session = session_id
        self._date = local_date
        self.dropped: list[pd.DataFrame] = []

    def drop(self, frame: pd.DataFrame, mask, reason: str) -> pd.DataFrame:
        """Rows of `frame` where `mask` is false; the others are recorded under `reason`."""
        mask = pd.Series(np.asarray(mask, dtype=bool), index=frame.index)
        removed = frame.index[mask]
        if len(removed):
            self.dropped.append(
                pd.DataFrame(
                    {
                        "site": self._site.loc[removed].astype(object).to_numpy(),
                        "session_id": self._session.loc[removed].astype(object).to_numpy(),
                        "reason": reason,
                        "local_date": self._date.loc[removed].to_numpy(),
                    }
                )
            )
        return frame.loc[~mask]

    def table(self) -> pd.DataFrame:
        if not self.dropped:
            return pd.DataFrame({name: pd.Series(dtype=object) for name in DROP_COLUMNS})
        out = pd.concat(self.dropped, ignore_index=True)
        out["local_date"] = out["local_date"].astype(object)
        return out[list(DROP_COLUMNS)]


def finish(table: pd.DataFrame) -> pd.DataFrame:
    """Check a loader's output against the common table's rules; return it sorted.

    Raises ValueError on a broken rule: a missing column, a time column that is not
    timezone-aware, more than one time zone, a missing plug-in or plug-out, plug-out not after
    plug-in, a missing or non-positive energy, a non-positive nominal cap, a repeated session id
    within a site, or an unknown role. Optional columns given are kept after the others.
    """
    missing = [name for name in COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"session table is missing columns: {missing}")
    optional = [name for name in OPTIONAL_COLUMNS if name in table.columns]
    out = table[[*COLUMNS, *optional]].copy()
    zones = set()
    for name in (*TIME_COLUMNS, *(DECLARED_AT,) * (DECLARED_AT in optional)):
        if not isinstance(out[name].dtype, pd.DatetimeTZDtype):
            raise ValueError(f"{name} must be timezone-aware")
        zones.add(str(out[name].dt.tz))
    if len(zones) != 1:
        raise ValueError(f"one time zone per table, found {sorted(zones)}")
    if out["plug_in"].isna().any() or out["plug_out"].isna().any():
        raise ValueError("every session needs a plug-in and a plug-out")
    if not (out["plug_out"] > out["plug_in"]).all():
        raise ValueError("every session must plug out after it plugs in")
    if not (out["energy_kwh"] > 0).all():
        raise ValueError("every session needs a positive energy")
    if not (out["nominal_kw"] > 0).all():
        raise ValueError("every session needs a positive nominal cap")
    if not out["role"].isin(ROLES).all():
        raise ValueError(f"role must be one of {ROLES}")
    if out.duplicated(["site", "session_id"]).any():
        raise ValueError("session ids must be unique within a site")
    for name in TEXT_COLUMNS:
        out[name] = out[name].astype(object).where(out[name].notna(), None)
    for name in NUMBER_COLUMNS:
        out[name] = out[name].astype(float)
    if FROM_DECLARED in out.columns:
        out[FROM_DECLARED] = out[FROM_DECLARED].astype(bool)
    out = out.sort_values(["site", "plug_in", "session_id"], kind="stable")
    return out.reset_index(drop=True)


def localize(naive: pd.Series, zone: str) -> pd.Series:
    """Local wall-clock times to timezone-aware ones; a time in the daylight-saving gap or the
    doubled hour becomes missing (the caller drops it under its own reason)."""
    return naive.dt.tz_localize(zone, nonexistent="NaT", ambiguous="NaT")
