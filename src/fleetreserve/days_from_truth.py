"""A unit's counting days as `day.Day` objects with true cars, read from its truth tables.

Input: the two tables `truth_build` writes for one unit, `<unit>-cars.csv` and `<unit>-days.csv`
(columns `truth.CAR_COLUMNS` and `truth.DAY_COLUMNS`), read as text. Nothing else is read. The
tables are parsed from their bytes (`parse_tables`), never from a path handed to pandas, which
would read a path written as a URL (`file:...`) or starting with `~` as another file: either the
bytes of the two files (`read_tables`, `load_unit_days`: each file opened by its path with the
plain `open`) or bytes a caller has already read and hashed (`parse_unit_days`, as
`gate_run.read_inputs` returns them), so the rows a run scores are the bytes whose SHA-256 it
records. When those bytes come as a `ReadFiles` (as `gate_run.read_inputs` and
`sealed_route.read_two_builds` return them), which holds for each file the folder it was read
from and what the file system said of it just before the read (`os.stat`: device, inode, size,
change and modification times), `parse_unit_days` asks the file system again after the parse
(`stat` only: no file is opened again) and raises `truth_verify.CouldNotRun` when either of the
unit's two files is no longer the file read: a table changed between its check and its parse is
refused, and a caller that parsed the file again from its path instead would make no such check.

Every counting day of the days table (at least 3 cars plugged in at τ, plan section 1) becomes one
`TruthDay`: a `Day` whose cars are the true cars `TrueCar(depart_hour, rate_kw, room)`, the
per-day record of what the rules below changed, and per car what the forecasts need (site,
arrival hour, exact plug-out time). The day's service hours are its row's `service_hours`; its
recovery hours are `day.covering_recovery_hours` of its cars; `Day.unit` is the unit's name (for
the pooled Norwegian development unit, the pooled name), and the site is kept in the record.

`Day` refuses two kinds of car that occur in the tables. The plan does not say what to do with
them, so each rule below is a reading, chosen by an argument and counted per day and per unit.

Reading 1, departure within the decision hour (`depart_hour` 0). Such a car cannot draw power in
any whole hour. It is left out of the day's fleet; it still counted towards the 3 cars plugged
in at τ, so the day stays a counting day. Its count and the room it carried are recorded. No
other reading is offered: a `Day` cannot hold such a car, and giving it a whole hour would add
power it never had.

Reading 2, protected energy that does not fit (`(1 - rho) * room > rate * depart_hour`), which
makes the car's true set empty. `NO_FIT_CAP` (the default): the room is capped at
`rate * depart_hour`, the energy the car can still take in its whole hours; the energy removed is
recorded. `NO_FIT_LEAVE_OUT`: the car is left out of the fleet; the room it carried is recorded.
The car count is the same under both.

Reading 3, a counting day whose fleet has fewer than 3 cars once reading 1 (and, under
`NO_FIT_LEAVE_OUT`, reading 2) has left cars out, or no car at all. `SMALL_KEEP` (the default):
the day is kept, because the counting rule is applied at τ. `SMALL_DROP`: such days are left out.
Both counts are recorded.

The same rules, car by car, are applied to every row of the cars table, also on days that do not
count, so that the forecasts can use the other days' cars (`UnitDays.cars`).

A session plugged in over several days appears on each of those days with its whole-session energy;
its room on each day is that energy minus what the truth rule draws before that day's τ (from
plug-in at the nominal cap, for NREL inside the recorded charging time). The record counts, per
day, the cars plugged in on an earlier calendar day and how many of them still have room.
"""

import io
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve.day import Day, TrueCar, covering_recovery_hours
from fleetreserve.truth import CAR_COLUMNS, COUNTING_CARS, DAY_COLUMNS
from fleetreserve.truth_verify import CouldNotRun

NO_FIT_CAP = "cap"
NO_FIT_LEAVE_OUT = "leave-out"
NO_FIT_READINGS = (NO_FIT_CAP, NO_FIT_LEAVE_OUT)
SMALL_KEEP = "keep"
SMALL_DROP = "drop"
SMALL_READINGS = (SMALL_KEEP, SMALL_DROP)
ROOM_TOLERANCE = 1e-6  # kWh: the tables are written with six decimals

# What happened to a row of the cars table under the rules.
KEPT = "kept"
LEFT_OUT_DEPART_ZERO = "left out: departure within the decision hour"
CAPPED = "room capped at rate times departure hour"
LEFT_OUT_NO_FIT = "left out: protected energy does not fit"


@dataclass(frozen=True)
class DayRecord:
    """What the rules changed on one counting day."""

    unit: str
    site: str
    date: str
    tau: str
    cars_at_tau: int
    cars_in_fleet: int
    depart_zero_cars: int
    depart_zero_room_kwh: float
    no_fit_cars: int
    no_fit_energy_removed_kwh: float  # under the cap reading: room minus rate * depart_hour
    no_fit_left_out: int  # under the leave-out reading: the same cars, left out
    no_fit_room_left_out_kwh: float
    fewer_than_three: bool
    empty_fleet: bool
    earlier_day_cars: int  # cars in the fleet plugged in on an earlier calendar day
    earlier_day_cars_with_room: int


@dataclass(frozen=True)
class FleetCar:
    """What the forecasts need about one car of a day's fleet (aligned with `Day.cars`)."""

    row: int  # position in `UnitDays.cars`
    session_id: str
    site: str
    arrival_hour: int  # clock hour of plug-in
    arrived_earlier_day: bool
    departure_exact: float  # hours from τ to the exact plug-out


@dataclass(frozen=True)
class TruthDay:
    day: Day
    record: DayRecord
    fleet: tuple[FleetCar, ...]


@dataclass(frozen=True)
class UnitDays:
    """A unit's counting days and its cars table with the rules applied to every row.

    `cars` columns: the table's own (as text), plus `departure_exact` (hours from τ to the exact
    plug-out), `depart`, `rate`, `room_table` (numbers), `arrival_hour`, `arrived_earlier_day`,
    `counting` (the row's day counts), `outcome` (one of the outcomes above), `in_fleet`, and
    `room` (the room after the rules; NaN for a car left out).
    """

    unit: str
    rho: float
    no_fit: str
    small_fleet: str
    days: tuple[TruthDay, ...]
    cars: pd.DataFrame
    dropped_small_days: int

    def counts(self) -> dict:
        """Per-unit totals of the records, under the readings chosen."""
        records = [d.record for d in self.days]
        return unit_counts(records, self.dropped_small_days, self.no_fit, self.small_fleet)


def unit_counts(records, dropped_small_days: int, no_fit: str, small_fleet: str) -> dict:
    def total(name):
        return sum(getattr(r, name) for r in records)

    return {
        "reading_no_fit": no_fit,
        "reading_small_fleet": small_fleet,
        "days": len(records),
        "days_left_out_small_fleet": dropped_small_days,
        "cars_at_tau": total("cars_at_tau"),
        "cars_in_fleet": total("cars_in_fleet"),
        "reading1_depart_zero_cars": total("depart_zero_cars"),
        "reading1_depart_zero_room_kwh": round(total("depart_zero_room_kwh"), 6),
        "reading2_no_fit_cars": total("no_fit_cars"),
        "reading2_cap_energy_removed_kwh": round(total("no_fit_energy_removed_kwh"), 6),
        "reading2_leave_out_cars": total("no_fit_left_out"),
        "reading2_leave_out_room_kwh": round(total("no_fit_room_left_out_kwh"), 6),
        "reading3_days_fewer_than_three": sum(r.fewer_than_three for r in records),
        "reading3_days_empty_fleet": sum(r.empty_fleet for r in records),
        "earlier_day_cars": total("earlier_day_cars"),
        "earlier_day_cars_with_room": total("earlier_day_cars_with_room"),
    }


def read_tables(folder: Path, unit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One unit's two truth tables, as text. Raises `FileNotFoundError` when one is missing."""
    cars = (Path(folder) / f"{unit}-cars.csv").read_bytes()
    days = (Path(folder) / f"{unit}-days.csv").read_bytes()
    return parse_tables(cars, days, unit)


def parse_tables(cars: bytes, days: bytes, unit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One unit's two truth tables, as text, from the bytes of its two files (module text)."""
    cars = pd.read_csv(io.BytesIO(cars), dtype=str, keep_default_na=False)
    days = pd.read_csv(io.BytesIO(days), dtype=str, keep_default_na=False)
    if list(cars.columns) != list(CAR_COLUMNS) or list(days.columns) != list(DAY_COLUMNS):
        raise ValueError(f"{unit}: the tables do not have the published columns")
    return cars, days


def _utc(text: pd.Series) -> pd.Series:
    return pd.to_datetime(text, utc=True, format="ISO8601")


def apply_car_rules(cars: pd.DataFrame, rho: float, no_fit: str) -> pd.DataFrame:
    """The cars table with readings 1 and 2 applied to every row (see the module text)."""
    if no_fit not in NO_FIT_READINGS:
        raise ValueError(f"no_fit must be one of {NO_FIT_READINGS}, got {no_fit!r}")
    if not 0 <= rho < 1:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    out = cars.reset_index(drop=True).copy()
    tau = _utc(out["tau"])
    out["departure_exact"] = (_utc(out["plug_out"]) - tau) / pd.Timedelta(hours=1)
    out["depart"] = out["depart_hour"].astype(int)
    out["rate"] = out["rate_kw"].astype(float)
    out["room_table"] = out["room_kwh"].astype(float)
    # Plug-in and τ are written in local time with their offset: the clock hour and the calendar
    # date are read from the text.
    out["arrival_hour"] = out["plug_in"].str[11:13].astype(int)
    out["arrived_earlier_day"] = out["plug_in"].str[:10] < out["date"]
    out["counting"] = out["day_counts"] == "True"
    deliverable = out["rate"] * out["depart"]
    no_fit_row = (1 - rho) * out["room_table"] > deliverable
    zero = out["depart"] == 0
    outcome = np.full(len(out), KEPT, dtype=object)
    outcome[no_fit_row.to_numpy() & ~zero.to_numpy()] = (
        CAPPED if no_fit == NO_FIT_CAP else LEFT_OUT_NO_FIT
    )
    outcome[zero.to_numpy()] = LEFT_OUT_DEPART_ZERO
    out["outcome"] = outcome
    out["in_fleet"] = ~out["outcome"].isin([LEFT_OUT_DEPART_ZERO, LEFT_OUT_NO_FIT])
    room = out["room_table"].where(out["outcome"] != CAPPED, deliverable)
    out["room"] = room.where(out["in_fleet"])
    out["no_fit"] = no_fit_row & ~zero
    return out


def _record(unit, site, day_date, tau, rows: pd.DataFrame, fleet: pd.DataFrame) -> DayRecord:
    zero = rows[rows["outcome"] == LEFT_OUT_DEPART_ZERO]
    no_fit = rows[rows["no_fit"]]
    excess = (no_fit["room_table"] - no_fit["rate"] * no_fit["depart"]).sum()
    left_out = no_fit[no_fit["outcome"] == LEFT_OUT_NO_FIT]
    earlier = fleet[fleet["arrived_earlier_day"]]
    return DayRecord(
        unit=unit,
        site=site,
        date=day_date,
        tau=tau,
        cars_at_tau=len(rows),
        cars_in_fleet=len(fleet),
        depart_zero_cars=len(zero),
        depart_zero_room_kwh=float(zero["room_table"].sum()),
        no_fit_cars=len(no_fit),
        no_fit_energy_removed_kwh=float(excess) if len(left_out) == 0 else 0.0,
        no_fit_left_out=len(left_out),
        no_fit_room_left_out_kwh=float(left_out["room_table"].sum()),
        fewer_than_three=len(fleet) < COUNTING_CARS,
        empty_fleet=len(fleet) == 0,
        earlier_day_cars=len(earlier),
        earlier_day_cars_with_room=int((earlier["room"] > ROOM_TOLERANCE).sum()),
    )


def unit_days(
    cars: pd.DataFrame,
    days: pd.DataFrame,
    *,
    unit: str,
    rho: float,
    no_fit: str,
    small_fleet: str,
) -> UnitDays:
    """The counting days of one unit from its two tables (as text; see `read_tables`)."""
    if small_fleet not in SMALL_READINGS:
        raise ValueError(f"small_fleet must be one of {SMALL_READINGS}, got {small_fleet!r}")
    table = apply_car_rules(cars, rho, no_fit)
    groups = {key: frame for key, frame in table.groupby(["site", "date"], sort=False)}
    built, dropped = [], 0
    counting_days = days[days["counts"] == "True"].sort_values(["date", "site"], kind="stable")
    for site, day_date, tau, service, listed in zip(
        counting_days["site"],
        counting_days["date"],
        counting_days["tau"],
        counting_days["service_hours"],
        counting_days["cars"],
        strict=True,
    ):
        rows = groups.get((site, day_date))
        if rows is None or len(rows) != int(listed):
            raise ValueError(f"{unit} {site} {day_date}: the two tables disagree on the cars")
        if not (rows["tau"] == tau).all():
            raise ValueError(f"{unit} {site} {day_date}: the two tables disagree on τ")
        fleet = rows[rows["in_fleet"]]
        record = _record(unit, site, day_date, tau, rows, fleet)
        if record.fewer_than_three and small_fleet == SMALL_DROP:
            dropped += 1
            continue
        true_cars = tuple(
            TrueCar(int(d), float(r), float(e))
            for d, r, e in zip(fleet["depart"], fleet["rate"], fleet["room"], strict=True)
        )
        n_service = int(service)
        day = Day(
            unit=unit,
            date=date.fromisoformat(day_date),
            n_service=n_service,
            n_recovery=covering_recovery_hours(true_cars, n_service),
            rho=rho,
            cars=true_cars,
        )
        info = tuple(
            FleetCar(
                row=int(index),
                session_id=str(item.session_id),
                site=str(item.site),
                arrival_hour=int(item.arrival_hour),
                arrived_earlier_day=bool(item.arrived_earlier_day),
                departure_exact=float(item.departure_exact),
            )
            for index, item in zip(fleet.index, fleet.itertuples(index=False), strict=True)
        )
        built.append(TruthDay(day, record, info))
    return UnitDays(unit, rho, no_fit, small_fleet, tuple(built), table, dropped)


def load_unit_days(
    folder: Path, unit: str, *, rho: float, no_fit: str, small_fleet: str
) -> UnitDays:
    """`unit_days` on the tables in `folder`."""
    cars, days = read_tables(folder, unit)
    return unit_days(cars, days, unit=unit, rho=rho, no_fit=no_fit, small_fleet=small_fleet)


def parse_unit_days(
    read: dict[str, bytes], unit: str, *, rho: float, no_fit: str, small_fleet: str
) -> UnitDays:
    """`unit_days` on the bytes of the unit's two tables in `read` (file name to bytes, as
    `gate_run.read_inputs` returns them; module text). When `read` is a `ReadFiles`, each of the
    two files is checked again after the parse; raises `CouldNotRun` when one changed."""
    names = (f"{unit}-cars.csv", f"{unit}-days.csv")
    cars, days = parse_tables(read[names[0]], read[names[1]], unit)
    if isinstance(read, ReadFiles):
        for name in names:
            read.confirm(name)
    return unit_days(cars, days, unit=unit, rho=rho, no_fit=no_fit, small_fleet=small_fleet)


def file_state(path: Path) -> tuple[int, ...]:
    """What `os.stat` says of `path`: device, inode, size, change and modification times."""
    found = os.stat(path)
    return (found.st_dev, found.st_ino, found.st_size, found.st_ctime_ns, found.st_mtime_ns)


class ReadFiles(dict):
    """File name to the bytes read once from the folder `folders[name]` and hashed by a check
    (`gate_run.read_inputs`), with the file's `file_state` taken just before that read in
    `states[name]`; `confirm(name)` asks the file system again (module text)."""

    def __init__(self, read: dict[str, bytes], folders: dict, states: dict) -> None:
        super().__init__(read)
        self.folders = dict(folders)
        self.states = dict(states)

    def confirm(self, name: str) -> None:
        """Raises `CouldNotRun` when the file `name` is no longer the file read."""
        path = Path(self.folders[name]) / name
        try:
            current = file_state(path)
        except OSError as problem:
            raise CouldNotRun(f"{path} cannot be found after its check: {problem}") from problem
        if current != self.states[name]:
            raise CouldNotRun(
                f"{path} changed after its SHA-256 was checked against its build (its size, "
                "times or identity on disk are not those of the file read); nothing parsed from "
                "it is used"
            )
