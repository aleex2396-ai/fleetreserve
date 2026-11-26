"""The WP2 pass-mark checks on the written truth tables (plan 13 WP2).

Command line, from the project's top folder:
    python -m fleetreserve.truth_verify train|development|fallbacks
        [--setting main|held] [--run <run folder>] [--tables <folder>]

The run is the recorded truth build (`fleetreserve.run_record --name wp2-truth`); by default the
latest folder `runs.noindex/*-wp2-truth`. Its `truth-summary.json` must pass
`run_verify.read_verified` (exit code 0, not timed out, hashes match; the bytes hashed are the
bytes parsed, and the file still holds them after the parse), and the run's commit must be the
commit checked out now. Tables are read from `data/derived/truth/` by default.

Settings (plan 3). `--setting held` checks the second full setting: the run is the latest
`runs.noindex/*-wp2-truth-held` (`fleetreserve.run_record --name wp2-truth-held`) and the tables
are read from `data/derived/truth-held/` by default. The main lookup never finds a held run. A
summary's setting (its `setting` key; main when it has none), the setting a tables folder holds
(`truth_build.SETTING_FILE`) and the setting asked for must agree, or the check could not run
(`require_matching_setting`; every command that reads the tables uses it). In the held setting
the rules are the same, except that energy before τ must be 0 for every car with the held rule
text, and the NREL rule about recorded charging ending before τ (an energy-before rule of the main
setting) is not applied.

- `train`, `development`: every unit of the role (`truth_build.UNITS`) is in the summary; its
  two tables exist and have the SHA-256 the run recorded; and the tables hold these rules:
  at least one counting day and one car; the published columns; times written with their
  offset; plug-in <= τ < plug-out; τ on its row's date at its window's clock hour;
  depart_hour = floor((plug-out - τ) / 1 hour), so depart_hour >= 0; in-window departure =
  depart_hour <= |S|; energy, nominal cap and rate > 0; 0 <= energy before τ <= energy; room =
  energy - energy before, so 0 <= room <= energy; no session twice on one day; each day's car
  count agrees between the two tables; a day counts exactly when it has at least 3 cars; a
  declared departure is written together with its statement time, and only when that time is at
  or before τ; no car's plug-out was taken from its declared departure. For
  the NREL units, whose energy before τ is bounded by the charge fields, the recorded charging
  end (`termin_charge`, read again through the NREL loader from `data/raw`) is joined to each
  car: a car whose recorded charging ended at or before τ has room 0.
- `fallbacks`: every training and development unit's summary has each fallback share of
  `truth_build.FALLBACK_KEYS` as a number from 0 to 1. This is the stricter of two judges: a
  share written as `null` (an undefined share, 0 of 0) fails here. The pass mark of WP2 (item 3)
  does not run this item; it runs `tools/wp2_fallback_check.py`, which judges the same shares
  alike but accepts a `null` share whose denominator is exactly 0.

Exit status: 0 passed; 1 a rule is broken; 2 could not run (no `data/raw`, no recorded build
at this commit, a table missing, the NREL file missing).
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve import load_nrel
from fleetreserve.run_verify import CannotCheck, NotVerified, read_verified, still_verified
from fleetreserve.sessions import RAW_ROOT, TOP, LayoutError
from fleetreserve.truth import (
    BEFORE_RULE_HELD,
    CAR_COLUMNS,
    COUNTING_CARS,
    DAY_COLUMNS,
    HELD,
    MAIN,
    SETTINGS,
)
from fleetreserve.truth_build import (
    DEFAULT_OUT,
    FALLBACK_KEYS,
    NREL_UNITS,
    SUMMARY_NAME,
    UNITS,
    read_setting,
)

RUN_LABELS = {MAIN: "wp2-truth", HELD: "wp2-truth-held"}
RUN_LABEL = RUN_LABELS[MAIN]
TOLERANCE = 2e-6  # the tables are written with six decimals
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}")


class CouldNotRun(Exception):
    pass


def _head_commit() -> str:
    done = subprocess.run(
        ["git", "-C", str(TOP), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise CouldNotRun("git rev-parse HEAD failed")
    return done.stdout.strip()


def latest_run(runs: Path, setting: str = MAIN) -> Path:
    """The latest recorded truth build of `setting`; a run of the other setting is never found."""
    label = RUN_LABELS[setting]
    found = sorted(path for path in runs.glob(f"*-{label}") if path.is_dir())
    if not found:
        raise CouldNotRun(f"no recorded truth build (runs.noindex/*-{label})")
    return found[-1]


def summary_setting(summary: dict) -> str:
    """The setting a truth summary was built in; a summary without the key is a main one."""
    setting = summary.get("setting", MAIN)
    if setting not in SETTINGS:
        raise CouldNotRun(f"the recorded truth build names an unknown setting: {setting!r}")
    return setting


def require_matching_setting(summary: dict, tables: Path, run_name: str) -> str:
    """The setting of a verified summary, if the tables folder holds that same setting."""
    setting = summary_setting(summary)
    found = read_setting(tables)
    if found != setting:
        raise CouldNotRun(
            f"the tables in {tables} are of the {found} setting but the recorded truth build "
            f"{run_name} is of the {setting} setting"
        )
    return setting


def load_summary(run: Path, head: str) -> dict:
    """The truth summary of `run`, verified at `head`, parsed from the bytes whose SHA-256 the
    check against its manifest compared (`run_verify.read_verified`, one read), and refused
    (`CouldNotRun`) when after the parse the file no longer holds those bytes
    (`run_verify.still_verified`): the table hashes a caller compares are those of the summary
    checked. A summary not verified raises `NotVerified`."""
    path = run / SUMMARY_NAME
    try:
        manifest, data = read_verified(path)
    except CannotCheck as problem:
        raise CouldNotRun(f"{path}: {problem}") from problem
    if manifest["commit"] != head:
        raise CouldNotRun(
            f"the recorded truth build {run.name} is at commit {manifest['commit'][:12]}, "
            f"not at the checked-out commit {head[:12]}"
        )
    summary = json.loads(data)
    try:
        still_verified(path, data)
    except NotVerified as problem:
        raise CouldNotRun(f"{path}: {problem}") from problem
    return summary


def _times(text: pd.Series, what: str, problems: list[str]) -> pd.Series:
    bad = ~text.fillna("").str.fullmatch(_TIME)
    if bad.any():
        problems.append(f"{what}: {int(bad.sum())} times without an offset or unreadable")
    return pd.to_datetime(text, utc=True, format="ISO8601", errors="coerce")


def nrel_charge_ends(raw_root: Path = RAW_ROOT) -> pd.Series:
    """Recorded charging end (UTC) of every NREL session, by (site, session_id)."""
    try:
        sessions, _ = load_nrel.load(raw_root)
    except (FileNotFoundError, LayoutError) as problem:
        raise CouldNotRun(f"the NREL session table could not be read: {problem}") from problem
    keys = pd.MultiIndex.from_arrays([sessions["site"], sessions["session_id"]])
    return pd.Series(sessions["charge_end"].dt.tz_convert("UTC").array, index=keys)


def check_tables(
    cars: pd.DataFrame,
    days: pd.DataFrame,
    tau_hours: dict,
    charge_ends: pd.Series | None = None,
    setting: str = MAIN,
) -> list[str]:
    """Problems found in one unit's tables (text columns as read); empty when every rule holds.

    `tau_hours` maps each site of the unit to the clock hour of its τ. `charge_ends` (from
    `nrel_charge_ends`), given for a unit bounded by its charge fields, adds the rule that a
    car whose recorded charging ended at or before τ has room 0. In the held setting energy
    before τ must be 0 with the held rule text and `charge_ends` is not used.
    """
    problems: list[str] = []
    if list(cars.columns) != list(CAR_COLUMNS) or list(days.columns) != list(DAY_COLUMNS):
        return ["the columns are not the published ones"]
    if len(cars) == 0 or len(days) == 0:
        return ["a table is empty"]
    tau = _times(cars["tau"], "cars tau", problems)
    plug_in = _times(cars["plug_in"], "plug_in", problems)
    plug_out = _times(cars["plug_out"], "plug_out", problems)
    _times(days["tau"], "days tau", problems)
    if problems:
        return problems
    present = (plug_in <= tau) & (tau < plug_out)
    if not present.all():
        problems.append(f"{int((~present).sum())} cars not plugged in at τ")
    for frame, what in ((cars, "cars"), (days, "days")):
        text = frame["tau"]
        on_date = text.str[:10] == frame["date"]
        if not on_date.all():
            problems.append(f"{what}: {int((~on_date).sum())} τ not on their row's date")
        expected = frame["site"].map(tau_hours)
        hour = text.str[11:13].astype(int)
        if expected.isna().any() or not (hour == expected).all():
            problems.append(f"{what}: τ not at the window's clock hour")
    depart = pd.to_numeric(cars["depart_hour"], errors="coerce")
    exact = np.floor((plug_out - tau) / pd.Timedelta(hours=1))
    if depart.isna().any() or not (depart == exact).all() or (depart < 0).any():
        problems.append("depart_hour is not floor((plug-out - τ) / 1 hour)")
    service = cars.merge(days, on=["site", "date"], how="left")["service_hours"]
    service = pd.to_numeric(service, errors="coerce").to_numpy()
    in_window = cars["in_window_departure"] == "True"
    if not (in_window.to_numpy() == (depart.to_numpy() <= service)).all():
        problems.append("in_window_departure is not depart_hour <= |S|")
    number = {
        name: pd.to_numeric(cars[name], errors="coerce").to_numpy(float)
        for name in ("energy_kwh", "nominal_kw", "rate_kw", "energy_before_kwh", "room_kwh")
    }
    for name in ("energy_kwh", "nominal_kw", "rate_kw"):
        if not (number[name] > 0).all():
            problems.append(f"{name} is not positive everywhere")
    energy, before, room = number["energy_kwh"], number["energy_before_kwh"], number["room_kwh"]
    if not ((before >= -TOLERANCE) & (before <= energy + TOLERANCE)).all():
        problems.append("energy before τ outside 0 to the session energy")
    if not (np.abs(room - (energy - before)) <= TOLERANCE).all():
        problems.append("room is not energy minus energy before τ")
    if not ((room >= -TOLERANCE) & (room <= energy + TOLERANCE)).all():
        problems.append("room outside 0 to the session energy")
    if cars.duplicated(["site", "date", "session_id"]).any():
        problems.append("a session is listed twice on one day")
    per_day = {key: int(n) for key, n in cars.groupby(["site", "date"]).size().items()}
    day_cars = pd.to_numeric(days["cars"], errors="coerce")
    listed = {
        (site, day): int(n)
        for site, day, n in zip(days["site"], days["date"], day_cars, strict=True)
        if n > 0
    }
    if per_day != listed:
        problems.append("car counts per day differ between the two tables")
    in_day = pd.to_numeric(cars["cars_in_day"], errors="coerce")
    expected_in_day = [per_day[key] for key in zip(cars["site"], cars["date"], strict=True)]
    if not (in_day.to_numpy() == np.array(expected_in_day)).all():
        problems.append("cars_in_day differs from the cars listed")
    counts = days["counts"] == "True"
    if not (counts == (day_cars >= COUNTING_CARS)).all():
        problems.append("a day counts without at least 3 cars, or has 3 and does not count")
    car_counts = cars["day_counts"] == "True"
    if not (car_counts == (in_day >= COUNTING_CARS)).all():
        problems.append("day_counts in the cars table disagrees with the 3-car rule")
    if not counts.any():
        problems.append("no counting day")
    stated_text, declared_text = cars["declared_at"], cars["declared_departure_h"]
    if not ((stated_text == "") == (declared_text == "")).all():
        problems.append("a declared departure without its statement time, or the reverse")
    given = stated_text != ""
    stated = _times(stated_text[given], "declared_at", problems)
    if (stated > tau[given]).any():
        problems.append(f"{int((stated > tau[given]).sum())} declared departures stated after τ")
    from_declared = cars["plug_out_from_declared"] != "False"
    if from_declared.any():
        problems.append(
            f"{int(from_declared.sum())} cars have a plug-out taken from the declared departure"
        )
    if setting == HELD:
        if (np.abs(before) > TOLERANCE).any():
            problems.append("energy before τ is not 0 in the held setting")
        if (cars["energy_before_rule"] != BEFORE_RULE_HELD).any():
            problems.append("energy_before_rule is not the held rule everywhere")
    elif charge_ends is not None:
        keys = pd.MultiIndex.from_arrays([cars["site"], cars["session_id"]])
        missing = ~keys.isin(charge_ends.index)
        if missing.any():
            problems.append(f"{int(missing.sum())} cars not in the NREL session table")
        end = charge_ends.reindex(keys)
        ended = (end.notna() & (end.array <= tau.array)).to_numpy()
        kept = ended & (room > TOLERANCE)
        if kept.any():
            problems.append(
                f"{int(kept.sum())} cars keep room although their recorded charging ended at or "
                "before τ"
            )
    return problems


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_role(
    role: str, summary: dict, tables: Path, raw_root: Path = RAW_ROOT
) -> tuple[int, list[str]]:
    lines, failed = [], False
    charge_ends = None
    setting = summary_setting(summary)
    for unit in UNITS[role]:
        entry = summary["units"].get(unit)
        if entry is None:
            lines.append(f"FAILED: {unit}: not in the recorded build's summary")
            failed = True
            continue
        paths = {kind: tables / f"{unit}-{kind}.csv" for kind in ("cars", "days")}
        for path in paths.values():
            if not path.is_file():
                raise CouldNotRun(f"{path} is missing")
        changed = [kind for kind, path in paths.items() if _sha256(path) != entry["sha256"][kind]]
        if changed:
            lines.append(f"FAILED: {unit}: {changed} differ from the recorded build's tables")
            failed = True
            continue
        cars = pd.read_csv(paths["cars"], dtype=str, keep_default_na=False)
        days = pd.read_csv(paths["days"], dtype=str, keep_default_na=False)
        sites = entry["sites"]
        tau_hours = {}
        for site in sites:
            window = summary["windows"].get(unit if len(sites) == 1 else site)
            if window is None:
                window = summary["windows"][site]
            tau_hours[site] = int(window["tau"][:2])
        if unit in NREL_UNITS and charge_ends is None and setting == MAIN:
            charge_ends = nrel_charge_ends(raw_root)
        ends = charge_ends if unit in NREL_UNITS and setting == MAIN else None
        problems = check_tables(cars, days, tau_hours, ends, setting)
        if problems:
            failed = True
            lines += [f"FAILED: {unit}: {problem}" for problem in problems]
        else:
            zero = int((pd.to_numeric(cars["depart_hour"]) == 0).sum())
            lines.append(
                f"passed: {unit}: {len(cars)} car-days, {int((days['counts'] == 'True').sum())} "
                f"counting days, {zero} car-days with depart_hour 0"
            )
    return (1 if failed else 0), lines


def verify_fallbacks(summary: dict) -> tuple[int, list[str]]:
    lines, failed = [], False
    for unit in UNITS["train"] + UNITS["development"]:
        shares = (summary["units"].get(unit) or {}).get("fallback_shares")
        if not isinstance(shares, dict):
            lines.append(f"FAILED: {unit}: no fallback shares logged")
            failed = True
            continue
        bad = [
            key
            for key in FALLBACK_KEYS
            if not isinstance(shares.get(key), int | float) or not 0.0 <= shares[key] <= 1.0
        ]
        if bad:
            lines.append(f"FAILED: {unit}: fallback shares missing or outside 0 to 1: {bad}")
            failed = True
        else:
            lines.append(f"passed: {unit}: {json.dumps(shares, sort_keys=True)}")
    return (1 if failed else 0), lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("train", "development", "fallbacks"))
    parser.add_argument("--setting", choices=SETTINGS, default=MAIN)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--tables", type=Path, default=None, help="default: by setting")
    args = parser.parse_args(argv)
    tables = args.tables if args.tables is not None else DEFAULT_OUT[args.setting]
    try:
        if not RAW_ROOT.is_dir():
            raise CouldNotRun("no data/raw folder in this checkout")
        run = args.run if args.run is not None else latest_run(TOP / "runs.noindex", args.setting)
        summary = load_summary(run, _head_commit())
        if summary_setting(summary) != args.setting:
            raise CouldNotRun(
                f"the recorded truth build {run.name} is of the {summary_setting(summary)} "
                f"setting, not the {args.setting} setting asked for"
            )
        require_matching_setting(summary, tables, run.name)
        if args.what == "fallbacks":
            status, lines = verify_fallbacks(summary)
        else:
            status, lines = verify_role(args.what, summary, tables)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except NotVerified as problem:
        print(f"FAILED: the recorded truth build is not verified: {problem}")
        return 1
    print(f"run: {run.name}")
    for line in lines:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
