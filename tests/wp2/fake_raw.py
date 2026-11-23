"""Made-up raw data files in the published layouts, written under a temporary `data/raw`.

Layouts copied from the sources' published forms: ACN-Data API session records (month files of
gzip-compressed JSON, times in UTC as "Wed, 01 May 2019 15:00:00 GMT"); the NREL workplace CSV
(quoted fields, `NULL` for missing, local times "2019-05-15 08:00:00"); the Norway Dataset 1
split files (";" between fields, decimal comma, `NA` for missing, local times). No real data
is read or copied.
"""

import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path

from fleetreserve.windows import Window

# The window the made-up days in these tests were built around: τ 10:00 local, S 11:00 to 17:00
# (six service hours). It is a fixed test input, not `windows.WORKPLACE`, so the tests keep
# their meaning whatever value the workplace window holds.
TEST_WINDOW = Window(
    tau_hour=10, service_hours=6, rule="made-up window for the tests: τ 10:00, S 11:00 to 17:00"
)

ACN_TIME = "%a, %d %b %Y %H:%M:%S GMT"
NREL_HEADER = (
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
NORWAY_HEADER = (
    "location",
    "user_id",
    "session_id",
    "plugin_time",
    "plugout_time",
    "connection_time",
    "energy_session",
)
NORWAY_SITES = {
    "train": ("OSL_S", "ASK", "TRO", "BAR", "BOD"),
    "development": ("OSL_1", "BER", "KRO", "OSL_2", "BAR_2"),
}


def raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "raw"
    root.mkdir(parents=True, exist_ok=True)
    return root


def acn_time(text: str | None) -> str | None:
    """'2019-05-01 15:00:00' (UTC) to the API's text form."""
    if text is None:
        return None
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").strftime(ACN_TIME)


def acn_record(
    session_id: str,
    connect: str,
    disconnect: str,
    kwh: float,
    *,
    user: str | None = "u1",
    station: str = "1-1-1-1",
    done: str | None = None,
    inputs: list | None = None,
    zone: str = "America/Los_Angeles",
) -> dict:
    """One API record; times are UTC as 'YYYY-MM-DD HH:MM:SS'."""
    return {
        "_id": session_id,
        "clusterID": "0001",
        "connectionTime": acn_time(connect),
        "disconnectTime": acn_time(disconnect),
        "doneChargingTime": acn_time(done),
        "kWhDelivered": kwh,
        "sessionID": session_id,
        "siteID": "0001",
        "spaceID": "S1",
        "stationID": station,
        "timezone": zone,
        "userID": user,
        "userInputs": inputs,
    }


def write_acn(root: Path, site: str, month: str, records: list[dict]) -> Path:
    folder = root / "acn" / site
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{month}.json.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(json.dumps(records).encode())
    return path


def nrel_row(**fields) -> dict:
    row = dict.fromkeys(NREL_HEADER, None)
    row.update(driverId="d1", vehicle_model="Car", miles_requested="20", afterPaid="False")
    row.update(fields)
    return row


def write_nrel(root: Path, rows: list[dict], header=NREL_HEADER) -> Path:
    folder = root / "nrel"
    folder.mkdir(parents=True, exist_ok=True)
    lines = [",".join(f'"{name}"' for name in header)]
    for row in rows:
        cells = []
        for name in header:
            value = row.get(name)
            if value is None:
                cells.append("NULL")
            elif name == "afterPaid":
                cells.append(str(value))
            else:
                cells.append(f'"{value}"')
        lines.append(",".join(cells))
    path = folder / "dataSet_Oct2021.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def norway_row(site, user, session, plug_in, plug_out, energy) -> dict:
    """A row; times 'YYYY-MM-DD HH:MM:SS' local, or None for a missing plug-out."""
    if plug_out is None:
        hours = None
    else:
        start = datetime.strptime(plug_in, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(plug_out, "%Y-%m-%d %H:%M:%S")
        hours = (end - start).total_seconds() / 3600
    return {
        "location": site,
        "user_id": user,
        "session_id": session,
        "plugin_time": plug_in,
        "plugout_time": plug_out,
        "connection_time": hours,
        "energy_session": energy,
    }


def _comma(value: float | None) -> str:
    return "NA" if value is None else f"{value:g}".replace(".", ",")


def write_norway(root: Path, role: str, rows: list[dict], header=NORWAY_HEADER) -> Path:
    folder = root / "norway"
    folder.mkdir(parents=True, exist_ok=True)
    lines = [";".join(f'"{name}"' for name in header)]
    for row in rows:
        cells = []
        for name in header:
            value = row.get(name)
            if name in ("location", "user_id", "session_id"):
                cells.append(f'"{value}"')
            elif name in ("connection_time", "energy_session"):
                cells.append(_comma(value))
            else:
                cells.append("NA" if value is None else value)
        lines.append(";".join(cells))
    path = folder / f"Dataset1_charging_reports-{role}.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def evening_sessions(site: str, first_day: str, days: int = 2) -> list[dict]:
    """Four cars a day plugging in at 15:00, 16:00, 17:00 and 17:30, leaving at 07:00 next day.

    On every day the median plugged-in count peaks at 18:00 (4 cars), so the window rule gives
    τ = 17:00, and 3 cars are plugged in at τ (the 17:00 car plugs in exactly at τ).
    """
    rows = []
    start = datetime.strptime(first_day, "%Y-%m-%d")
    for k in range(days):
        day = start + timedelta(days=k)
        leave = (day + timedelta(days=1, hours=7)).strftime("%Y-%m-%d %H:%M:%S")
        for n, (hour, minute) in enumerate(((15, 0), (16, 0), (17, 0), (17, 30))):
            plug_in = (day + timedelta(hours=hour, minutes=minute)).strftime("%Y-%m-%d %H:%M:%S")
            rows.append(norway_row(site, f"{site}_u{n}", f"{site}_{k}_{n}", plug_in, leave, 10.0))
    return rows


def full_tree(tmp_path: Path) -> Path:
    """A raw folder with every training and development source, each with counting days."""
    root = raw_root(tmp_path)
    for site in ("caltech", "jpl", "office001"):
        records = [
            acn_record(
                f"{site}-{n}",
                "2019-05-01 15:00:00",  # 08:00 local (UTC-7)
                "2019-05-02 00:00:00",  # 17:00 local
                12.0 + n,
                user=f"u{n}" if n else None,
                done="2019-05-01 18:00:00",
                station=f"1-1-1-{n}",
            )
            for n in range(3)
        ]
        write_acn(root, site, "2019-05", records)
    rows = [nrel_row(request_entry_time="2018-06-01 08:00:00", energy_charged="5")]
    for day in ("2019-04-10", "2019-05-15"):
        for n in range(3):
            rows.append(
                nrel_row(
                    driverId=f"d{n}",
                    request_entry_time=f"{day} 08:00:05",
                    expected_departure=f"{day} 16:00:00",
                    station=f"LV11-0{n}",
                    start_date_time=f"{day} 08:00:00",
                    end_date_time=f"{day} 16:30:00",
                    start_charge=f"{day} 09:00:00" if n else None,
                    # Car 2's recorded charging ends before τ 10:00, car 1's after it.
                    termin_charge={0: None, 1: f"{day} 11:00:00", 2: f"{day} 09:30:00"}[n],
                    max_charge_power="6.1" if n else None,
                    kwh_requested="10",
                    energy_charged="9.5",
                )
            )
    write_nrel(root, rows)
    for role, sites in NORWAY_SITES.items():
        rows = [row for site in sites for row in evening_sessions(site, "2019-11-05")]
        write_norway(root, role, rows)
    return root
