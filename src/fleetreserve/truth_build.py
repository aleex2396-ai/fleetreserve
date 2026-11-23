"""Build the training and development truth tables and report the data checks (WP2).

Command line, from the project's top folder:
    python -m fleetreserve.truth_build [--raw-root data/raw] [--setting main|held] [--out <folder>]

It loads every training and development source from `data/raw/` (never any other folder),
runs the checks of `data_checks`, sets each site's window (`windows`), builds each unit's truth
table (`truth`) and counts the effect of other readings (`truth_readings`). Units (plan 1, 13):
- training: caltech, jpl, office001 (ACN); OSL_S, ASK, TRO, BAR, BOD (Norway);
- development: nrel-before-may-2019 and nrel-from-may-2019 (NREL's two sites pooled, split by
  the day of τ at 1 May 2019, the managed-charging controller); OSL_1, BER, KRO, OSL_2, BAR_2;
  and norway-development-pooled, the gate's "pooled Norwegian development sites", read as the
  union of the five development sites' own site-days, each with its site's window.

Settings (plan 3): `--setting main` (the default) is the main truth; `--setting held` is the
second full setting, reported apart and never pooled with the main one: no charging before τ,
so every car's energy before τ is 0 and its room is the session energy (`truth.HELD`). Nothing
else differs. The two settings write to different default folders (`OUT`, `OUT_HELD`); the held
folder also holds `SETTING_FILE`, a one-line note naming the setting, so a folder of tables says
which setting it holds (a folder without the note holds the main one). A build refuses (exit 2) to
write into a folder that already holds tables of the other setting. The held summary document and
report state the setting on their first lines; the main ones are as they were before the held
setting existed (no setting line), so a summary without a `setting` key is a main one.

Outputs, in `--out` (never in git): `<unit>-cars.csv` (one row per car plugged in at τ on a day
with at least one car; columns of `truth.CAR_COLUMNS`) and `<unit>-days.csv` (every calendar
day; `truth.DAY_COLUMNS`). Times are ISO 8601 with their offset; numbers have six decimals. When
`FLEETRESERVE_RUN_DIR` is set (the run recorder sets it), it also writes there
`truth-summary.json` (per unit: sessions, rows removed by reason, days, counting days, cars per
counting day, shares, fallback shares, reading counts, and the SHA-256 of each table written)
and `truth-report.txt` (what it prints). The same inputs give the same bytes.

The report prints the findings first: checks that contradict an assumption of the plan.

Exit status: 0 the tables are written; 1 a file does not have its published layout; 2 a raw
data file is missing, the output folder cannot be written or holds the other setting's tables.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve import data_checks, load_acn, load_norway, load_nrel, truth, truth_readings
from fleetreserve.sessions import PHYSICAL_LIMIT_KW, RAW_ROOT, TOP, LayoutError
from fleetreserve.windows import (
    WORKPLACE,
    Window,
    occupancy_profile,
    site_days,
    window_from_profile,
)

OUT = TOP / "data" / "derived" / "truth"
OUT_HELD = TOP / "data" / "derived" / "truth-held"
DEFAULT_OUT = {truth.MAIN: OUT, truth.HELD: OUT_HELD}
SETTING_FILE = "truth-setting.txt"
WINDOWS_AS_MAIN = "- Windows as in the main setting (plan 3): same τ and service hours per unit."
SUMMARY_NAME = "truth-summary.json"
REPORT_NAME = "truth-report.txt"
NREL_CHANGE = date(2019, 5, 1)
POOLED = "norway-development-pooled"
NREL_UNITS = ("nrel-before-may-2019", "nrel-from-may-2019")
TRAIN_UNITS = ("caltech", "jpl", "office001", "OSL_S", "ASK", "TRO", "BAR", "BOD")
DEVELOPMENT_UNITS = (
    "nrel-before-may-2019",
    "nrel-from-may-2019",
    "OSL_1",
    "BER",
    "KRO",
    "OSL_2",
    "BAR_2",
    POOLED,
)
UNITS = {"train": TRAIN_UNITS, "development": DEVELOPMENT_UNITS}
FALLBACK_KEYS = (
    "rate_nominal_cap_no_user_id",
    "sessions_missing_user_id",
    "nominal_cap_floor_max_charge_power_missing",
    "energy_before_no_charging_time",
    "rate_average_without_charging_time",
)
# Session counts stated in plan section 3 (ACN) and in the Norway data paper's Table 1.
STATED_ROWS = {
    "caltech": 31424,
    "jpl": 33638,
    "office001": 1683,
    "OSL_S": 9757,
    "ASK": 6372,
    "TRO": 2168,
    "BAR": 1969,
    "BOD": 508,
    "OSL_1": 464,
    "BER": 308,
    "KRO": 492,
    "OSL_2": 127,
    "BAR_2": 1028,
}


def _share(part, whole) -> float | None:
    return None if whole == 0 else round(float(part) / float(whole), 6)


def summarize(
    unit: str,
    role: str,
    cars: pd.DataFrame,
    days: pd.DataFrame,
    sessions: pd.DataFrame,
    dropped: pd.DataFrame,
) -> dict:
    counting = cars[cars["day_counts"]]
    per_day = days.loc[days["counts"], "cars"].to_numpy(float)
    n = len(counting)
    spread = (
        {
            "median": round(float(np.median(per_day)), 3),
            "p10": round(float(np.percentile(per_day, 10)), 3),
            "p90": round(float(np.percentile(per_day, 90)), 3),
        }
        if len(per_day)
        else None
    )
    history = counting["rate_rule"] == truth.RATE_HISTORY
    fallback = {
        "rate_nominal_cap_no_user_id": _share(
            (counting["rate_rule"] == truth.RATE_FALLBACK).sum(), n
        ),
        "sessions_missing_user_id": _share(sessions["user_id"].isna().sum(), len(sessions)),
        "nominal_cap_floor_max_charge_power_missing": _share(
            (counting["nominal_rule"] == "3.3 kW floor: max_charge_power missing").sum(), n
        ),
        "energy_before_no_charging_time": _share(
            (counting["energy_before_rule"] == truth.BEFORE_RULE_UNBOUNDED).sum(), n
        ),
        "rate_average_without_charging_time": _share(
            (history & (counting["rate_average_kind"] == "plug-in to plug-out")).sum(),
            history.sum(),
        ),
    }
    return {
        "unit": unit,
        "role": role,
        "sites": sorted(set(days["site"])),
        "sessions_loaded": len(sessions),
        "rows_removed": {
            str(reason): int(count)
            for reason, count in dropped["reason"].value_counts().sort_index().items()
        },
        "days": len(days),
        "days_with_a_car": int((days["cars"] > 0).sum()),
        "counting_days": int(days["counts"].sum()),
        "cars_on_counting_days": n,
        "cars_per_counting_day": spread,
        "share_in_window_departure": _share(counting["in_window_departure"].sum(), n),
        "share_zero_room": _share((counting["room_kwh"] <= 1e-12).sum(), n),
        "share_room_not_fitting_rate_times_depart_hour": _share((~counting["room_fits"]).sum(), n),
        "share_depart_hour_zero": _share((counting["depart_hour"] == 0).sum(), n),
        "share_declared_departure_known_at_tau": _share(
            counting["declared_departure_h"].notna().sum(), n
        ),
        "cars_plug_out_from_declared": int(counting["plug_out_from_declared"].sum()),
        "fallback_shares": fallback,
    }


def _times_text(series: pd.Series) -> pd.Series:
    text = series.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    text = text.str[:-2] + ":" + text.str[-2:]
    return text.astype(object).where(series.notna(), "")


def csv_bytes(frame: pd.DataFrame) -> bytes:
    out = frame.copy()
    for name in out.columns:
        if isinstance(out[name].dtype, pd.DatetimeTZDtype):
            out[name] = _times_text(out[name])
        elif name == "date":
            out[name] = [str(value) for value in out[name]]
    text = out.to_csv(index=False, lineterminator="\n", float_format="%.6f")
    return text.encode()


def _drops_between(dropped: pd.DataFrame, first: date | None, last: date | None):
    dates = pd.Series(list(dropped["local_date"]), dtype=object)
    keep = np.ones(len(dropped), dtype=bool)
    if first is not None:
        keep &= np.array([d is not None and d >= first for d in dates], dtype=bool)
    if last is not None:
        keep &= np.array([d is not None and d < last for d in dates], dtype=bool)
    return dropped[keep]


class Build:
    """Everything one build computes, kept for writing and reporting."""

    def __init__(self, setting: str = truth.MAIN) -> None:
        if setting not in truth.SETTINGS:
            raise ValueError(f"setting must be one of {truth.SETTINGS}, not {setting!r}")
        self.setting = setting
        self.tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        self.summary: dict[str, dict] = {}
        self.checks: dict[str, dict] = {}
        self.windows: dict[str, dict] = {}
        self.alternatives: dict[str, dict] = {}

    def add_unit(self, unit, role, cars, days, sessions, dropped, readings) -> None:
        self.tables[unit] = (cars, days)
        entry = summarize(unit, role, cars, days, sessions, dropped)
        entry["reading_changes"] = readings
        self.summary[unit] = entry


def _window_entry(window: Window, zone: str, detail: dict | None = None) -> dict:
    entry = {
        "zone": zone,
        "tau": f"{window.tau_hour:02d}:00",
        "service_hours": window.service_hours,
        "service": f"{(window.tau_hour + 1) % 24:02d}:00 to "
        f"{(window.tau_hour + 1 + window.service_hours) % 24:02d}:00",
        "rule": window.rule,
    }
    if detail:
        entry.update(detail)
    return entry


def acn_site_tables(
    sessions: pd.DataFrame, site: str, window: Window, setting: str = truth.MAIN
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One ACN site's cars table and days table for `window` (all days of the site)."""
    return truth.build(
        sessions,
        window,
        load_acn.ZONE,
        unit=site,
        days=site_days(sessions["plug_in"]),
        limit_kw=PHYSICAL_LIMIT_KW["acn"],
        bound_by_charge_fields=False,
        setting=setting,
    )


def build_acn(result: Build, raw_root: Path, workplace: Window = WORKPLACE) -> None:
    limit = PHYSICAL_LIMIT_KW["acn"]
    checks = {
        "dst_shift": {},
        "units": {},
        "midnight": {},
        "drop_shares": {},
        "counts_against_plan": {},
    }
    for site in load_acn.SITES:
        records = load_acn.read_records(site, raw_root)
        sessions, dropped = load_acn.to_sessions(site, records)
        checks["dst_shift"][site] = data_checks.dst_shift(sessions, load_acn.ZONE)
        checks["units"][site] = data_checks.units(sessions, limit)
        checks["midnight"][site] = data_checks.midnight(sessions)
        checks["drop_shares"][site] = data_checks.drop_shares(len(records), dropped)
        checks["counts_against_plan"][site] = (len(records), STATED_ROWS[site])
        checks.setdefault("weekday_plug_in_hours", {})[site] = data_checks.weekday_plug_in_hours(
            sessions
        )
        result.windows[site] = _window_entry(workplace, load_acn.ZONE)
        cars, day_table = acn_site_tables(sessions, site, workplace, result.setting)
        readings = truth_readings.count_changes(
            cars, sessions, limit_kw=limit, bound_by_charge_fields=False, setting=result.setting
        )
        result.add_unit(site, "train", cars, day_table, sessions, dropped, readings)
    result.checks["acn"] = checks


def build_nrel(result: Build, raw_root: Path, workplace: Window = WORKPLACE) -> None:
    limit = PHYSICAL_LIMIT_KW["nrel"]
    raw = load_nrel.read_raw(raw_root)
    sessions, dropped = load_nrel.to_sessions(raw)
    requests = pd.DataFrame(
        {
            "plug_in": raw["request_entry_time"]
            .dt.tz_localize(load_nrel.ZONE, nonexistent="NaT", ambiguous="NaT")
            .dropna()
        }
    )
    result.checks["nrel"] = {
        "dst_shift": {
            "nrel plug-in": data_checks.dst_shift(sessions, load_nrel.ZONE),
            "nrel request entry": data_checks.dst_shift(requests, load_nrel.ZONE),
        },
        "gap_and_doubled": {
            name: data_checks.gap_and_doubled(raw[name], load_nrel.ZONE)
            for name in ("request_entry_time", "start_date_time", "end_date_time")
        },
        "units": {"nrel": data_checks.units(sessions, limit)},
        "midnight": {"nrel": data_checks.midnight(sessions)},
        "drop_shares": {"nrel": data_checks.drop_shares(len(raw), dropped)},
        "weekday_plug_in_hours": {"nrel": data_checks.weekday_plug_in_hours(sessions)},
        "rows_read": len(raw),
    }
    average, kind = truth.session_averages(sessions, limit)
    rates = truth.user_rates(sessions, average, kind)
    all_days = site_days(sessions["plug_in"])
    plug_in_dates = sessions["plug_in"].dt.date
    for unit, first, last in (
        (NREL_UNITS[0], None, NREL_CHANGE),
        (NREL_UNITS[1], NREL_CHANGE, None),
    ):
        days = [d for d in all_days if (first is None or d >= first) and (last is None or d < last)]
        cars, day_table = truth.build(
            sessions,
            workplace,
            load_nrel.ZONE,
            unit=unit,
            days=days,
            limit_kw=limit,
            bound_by_charge_fields=True,
            rates=rates,
            setting=result.setting,
        )
        in_regime = np.ones(len(sessions), dtype=bool)
        if first is not None:
            in_regime &= (plug_in_dates >= first).to_numpy()
        if last is not None:
            in_regime &= (plug_in_dates < last).to_numpy()
        readings = truth_readings.count_changes(
            cars, sessions, limit_kw=limit, bound_by_charge_fields=True, setting=result.setting
        )
        result.add_unit(
            unit,
            "development",
            cars,
            day_table,
            sessions[in_regime],
            _drops_between(dropped, first, last),
            readings,
        )
        result.windows[unit] = _window_entry(workplace, load_nrel.ZONE)
    alternatives = {}
    for fill in ("charge_end", "declared_departure"):
        other, _ = load_nrel.to_sessions(raw, missing_plug_out=fill)
        days = site_days(other["plug_in"])
        _, day_table = truth.build(
            other,
            workplace,
            load_nrel.ZONE,
            unit="alternative",
            days=days,
            limit_kw=limit,
            bound_by_charge_fields=True,
            setting=result.setting,
        )
        before = np.array([d < NREL_CHANGE for d in day_table["date"]])
        alternatives[f"plug-out from {fill} where not recorded"] = {
            "counting_days_before_may_2019": int(day_table["counts"][before].sum()),
            "counting_days_from_may_2019": int(day_table["counts"][~before].sum()),
        }
    result.alternatives["nrel"] = alternatives


def build_norway(result: Build, raw_root: Path) -> None:
    limit = PHYSICAL_LIMIT_KW["norway"]
    checks = {
        "dst_shift": {},
        "units": {},
        "midnight": {},
        "drop_shares": {},
        "counts_against_plan": {},
        "gap_and_doubled": {},
        "stated_duration": {},
        "weekday_plug_in_hours": {},
    }
    pooled_cars, pooled_days, pooled_sessions, pooled_drops = [], [], [], []
    for role in ("train", "development"):
        raw = load_norway.read_raw(role, raw_root)
        sessions, dropped = load_norway.to_sessions(role, raw)
        rows_read = raw["location"].value_counts()
        for name in ("plugin_time", "plugout_time"):
            checks["gap_and_doubled"][f"{role} {name}"] = data_checks.gap_and_doubled(
                raw[name], load_norway.ZONE
            )
        checks["stated_duration"][role] = data_checks.stated_duration(raw, load_norway.ZONE)
        for site in load_norway.SITES[role]:
            part = sessions[sessions["site"] == site]
            gone = dropped[dropped["site"] == site]
            checks["dst_shift"][site] = data_checks.dst_shift(part, load_norway.ZONE)
            checks["units"][site] = data_checks.units(part, limit)
            checks["midnight"][site] = data_checks.midnight(part)
            checks["drop_shares"][site] = data_checks.drop_shares(int(rows_read[site]), gone)
            checks["counts_against_plan"][site] = (int(rows_read[site]), STATED_ROWS[site])
            checks["weekday_plug_in_hours"][site] = data_checks.weekday_plug_in_hours(part)
            profile = occupancy_profile(part["plug_in"], part["plug_out"], load_norway.ZONE)
            window, detail = window_from_profile(profile)
            days = list(profile.index)
            cars, day_table = truth.build(
                part,
                window,
                load_norway.ZONE,
                unit=site,
                days=days,
                limit_kw=limit,
                bound_by_charge_fields=False,
                setting=result.setting,
            )
            alternative = Window(detail["alternative_tau_hour_earliest_peak"], 8, "alternative")
            _, other_days = truth.build(
                part,
                alternative,
                load_norway.ZONE,
                unit="alternative",
                days=days,
                limit_kw=limit,
                bound_by_charge_fields=False,
                setting=result.setting,
            )
            detail["alternative_counting_days"] = int(other_days["counts"].sum())
            result.windows[site] = _window_entry(window, load_norway.ZONE, detail)
            readings = truth_readings.count_changes(
                cars,
                part,
                limit_kw=limit,
                bound_by_charge_fields=False,
                setting=result.setting,
            )
            result.add_unit(site, role, cars, day_table, part, gone, readings)
            if role == "development":
                pooled_cars.append(cars.assign(unit=POOLED))
                pooled_days.append(day_table.assign(unit=POOLED))
                pooled_sessions.append(part)
                pooled_drops.append(gone)
    cars = pd.concat(pooled_cars, ignore_index=True)
    days = pd.concat(pooled_days, ignore_index=True)
    sessions = pd.concat(pooled_sessions, ignore_index=True)
    readings = {
        key: _pooled_count(
            [
                result.summary[site]["reading_changes"][key]
                for site in load_norway.SITES["development"]
            ]
        )
        for key in result.summary["OSL_1"]["reading_changes"]
    }
    result.add_unit(POOLED, "development", cars, days, sessions, pd.concat(pooled_drops), readings)
    result.checks["norway"] = checks


def _pooled_count(counts: list):
    """The sum of per-site counts; `None` (not applicable) when the sites have none."""
    return None if any(count is None for count in counts) else sum(counts)


def build_all(
    raw_root: Path = RAW_ROOT, workplace: Window = WORKPLACE, setting: str = truth.MAIN
) -> Build:
    """Every unit's tables in `setting`. `workplace` is the window of the ACN and NREL units; the
    Norway sites set their own."""
    result = Build(setting)
    build_acn(result, raw_root, workplace)
    build_nrel(result, raw_root, workplace)
    build_norway(result, raw_root)
    return result


def all_findings(result: Build) -> list[str]:
    out = []
    for source, checks in result.checks.items():
        out.extend(data_checks.findings(source, checks))
    for unit in ("nrel-before-may-2019", "nrel-from-may-2019", POOLED):
        counting = result.summary[unit]["counting_days"]
        if counting < 100:
            out.append(
                f"{unit}: {counting} counting days, under the 100 the WP6 gate asks of each "
                "gate unit (plan 13 WP6)"
            )
    return out


def setting_line(setting: str) -> str:
    """The first line of the held setting's report (the main report has none)."""
    return (
        f"SETTING: {setting}. No charging before τ: energy before τ is 0 for every car and room "
        "is the session energy. Reported apart from the main setting, never pooled with it."
    )


def report_lines(result: Build, written: dict[str, dict]) -> list[str]:
    lines = []
    if result.setting != truth.MAIN:
        lines += [setting_line(result.setting), ""]
    lines.append("FINDINGS (checks that contradict an assumption of the plan)")
    found = all_findings(result)
    lines += [f"- {text}" for text in found] or ["- none"]
    lines.append("")
    lines.append("WINDOWS")
    if result.setting != truth.MAIN:
        lines.append(WINDOWS_AS_MAIN)
    for unit, entry in result.windows.items():
        extra = ""
        if "peak_median" in entry:
            extra = (
                f"; median peak {entry['peak_median']} cars at {entry['peak_hour']:02d}:00 over "
                f"{entry['days']} days; earliest-peak reading τ "
                f"{entry['alternative_tau_hour_earliest_peak']:02d}:00 would give "
                f"{entry['alternative_counting_days']} counting days"
            )
        lines.append(
            f"- {unit}: τ {entry['tau']}, S {entry['service']} ({entry['service_hours']} hours)"
            f"{extra}"
        )
    lines.append("")
    lines.append("UNITS")
    for unit, entry in result.summary.items():
        spread = entry["cars_per_counting_day"] or {}
        lines.append(
            f"- {unit} ({entry['role']}): sessions {entry['sessions_loaded']}, removed "
            f"{entry['rows_removed'] or 'none'}, days {entry['days']}, counting days "
            f"{entry['counting_days']}, cars per counting day median {spread.get('median')} "
            f"(p10 {spread.get('p10')}, p90 {spread.get('p90')}), in-window departure "
            f"{entry['share_in_window_departure']}, zero room {entry['share_zero_room']}, "
            f"room not fitting {entry['share_room_not_fitting_rate_times_depart_hour']}, "
            f"depart hour 0 {entry['share_depart_hour_zero']}, declared departure known at τ "
            f"{entry['share_declared_departure_known_at_tau']}"
        )
        lines.append(f"  fallback shares {entry['fallback_shares']}")
        lines.append(f"  cars changed by other readings {entry['reading_changes']}")
    lines.append("")
    lines.append("READINGS COUNTED PER UNIT (cars on counting days whose value would change)")
    for key, text in truth_readings.READINGS.items():
        if result.setting == truth.HELD and key in truth_readings.NOT_APPLICABLE_HELD:
            text += " (not applicable in the held setting: no energy is drawn before τ)"
        lines.append(f"- {key}: {text}")
    lines.append("")
    lines.append("ALTERNATIVE READINGS (counts only)")
    for source, entries in result.alternatives.items():
        for name, numbers in entries.items():
            lines.append(f"- {source}, {name}: {numbers}")
    lines.append("")
    lines.append("CHECKS")
    for source, checks in result.checks.items():
        for name, value in checks.items():
            lines.append(f"- {source} {name}: {json.dumps(value, sort_keys=True, default=str)}")
    lines.append("")
    lines.append("TABLES WRITTEN")
    for unit, hashes in written.items():
        lines.append(f"- {unit}: {hashes}")
    return lines


def read_setting(folder: Path) -> str:
    """The setting a folder of tables holds: the text of its `SETTING_FILE`, `main` without one."""
    note = Path(folder) / SETTING_FILE
    return note.read_text().strip() if note.is_file() else truth.MAIN


def refuse_other_setting(out: Path, setting: str) -> None:
    """Raise `OSError` if `out` already holds tables of a setting other than `setting`."""
    folder = Path(out)
    holds_tables = folder.is_dir() and any(folder.glob("*-cars.csv"))
    if (holds_tables or (folder / SETTING_FILE).is_file()) and read_setting(folder) != setting:
        raise OSError(f"{folder} already holds tables of the {read_setting(folder)} setting")


def write(result: Build, out: Path) -> dict[str, dict]:
    refuse_other_setting(out, result.setting)
    out.mkdir(parents=True, exist_ok=True)
    if result.setting != truth.MAIN:
        (out / SETTING_FILE).write_text(result.setting + "\n")
    written = {}
    for unit, (cars, days) in result.tables.items():
        hashes = {}
        for kind, frame in (("cars", cars), ("days", days)):
            data = csv_bytes(frame)
            (out / f"{unit}-{kind}.csv").write_bytes(data)
            hashes[kind] = hashlib.sha256(data).hexdigest()
        written[unit] = hashes
    return written


def summary_document(result: Build, written: dict[str, dict]) -> dict:
    units = {}
    for unit, entry in result.summary.items():
        units[unit] = {**entry, "sha256": written[unit]}
    document = {
        "units": units,
        "windows": result.windows,
        "alternatives": result.alternatives,
        "findings": all_findings(result),
        "checks": result.checks,
    }
    if result.setting != truth.MAIN:
        document["setting"] = result.setting
    return document


def summary_text(document: dict) -> str:
    """The summary document as written. A document with a `setting` key (the held setting) has
    it as its first line; one without (the main setting) is the sorted document alone."""
    rest = {key: value for key, value in document.items() if key != "setting"}
    text = json.dumps(rest, indent=1, sort_keys=True, default=str) + "\n"
    if "setting" in document:
        text = '{\n "setting": ' + json.dumps(document["setting"]) + "," + text[1:]
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--setting", choices=truth.SETTINGS, default=truth.MAIN)
    parser.add_argument(
        "--out", type=Path, default=None, help="default: by setting (OUT, OUT_HELD)"
    )
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else DEFAULT_OUT[args.setting]
    try:
        refuse_other_setting(out, args.setting)
    except OSError as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        result = build_all(args.raw_root, setting=args.setting)
    except FileNotFoundError as problem:
        print(f"COULD NOT RUN: a raw data file is missing: {problem}")
        return 2
    except LayoutError as problem:
        print(f"FAILED: a file does not have its published layout: {problem}")
        return 1
    try:
        written = write(result, out)
    except OSError as problem:
        print(f"COULD NOT RUN: cannot write the tables: {problem}")
        return 2
    lines = report_lines(result, written)
    text = "\n".join(lines) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        document = summary_document(result, written)
        (folder / SUMMARY_NAME).write_text(summary_text(document))
        (folder / REPORT_NAME).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
