"""The fallback-share item of the WP2 pass mark (plan 13 WP2: "fallback shares logged").

Run as `python tools/wp2_fallback_check.py [--setting main|held] [--run <run folder>]
[--tables <folder>]`, from the project's top folder. It reads the recorded truth build exactly as
`python -m fleetreserve.truth_verify fallbacks` does (the same lookups, from `truth_verify`: the
latest `runs.noindex/*-wp2-truth`, its summary verified at the checked-out commit, the same
"could not run" cases) and judges every training and development unit as
`truth_verify.verify_fallbacks` does, with one difference.

A share the truth build wrote as `null` has a denominator of 0 and is undefined, not missing. It
passes, and is printed as `<key> undefined: 0 of 0 (<what the denominator counts>)`, only when the
denominator, rebuilt from the same unit's entry and tables, is exactly 0:
- `sessions_missing_user_id`: `sessions_loaded` is 0;
- the three shares over counting cars: `cars_on_counting_days` is 0;
- `rate_average_without_charging_time`: the counting cars whose rate came from history, counted
  exactly from the unit's cars table (`<unit>-cars.csv` in the tables folder, as `truth_verify`
  locates it): the rows whose `day_counts` is True and whose `rate_rule` is
  `truth.RATE_HISTORY`, is 0. The table is used only after its SHA-256 equals the one the
  recorded build holds for it (`summary["units"][unit]["sha256"]["cars"]`); a different hash
  fails the unit, and a missing table could not run (exit 2), as in `truth_verify`.
Anything else that is `null`, a missing key, a value that is not a number, and a number outside
0 to 1 fails.

Exit status: 0 passed; 1 a share is not accepted, or the recorded build is not verified; 2 could
not run (the same cases as `truth_verify`).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from fleetreserve import truth
from fleetreserve import truth_verify as tv
from fleetreserve.truth_build import DEFAULT_OUT, FALLBACK_KEYS, UNITS

SESSIONS = "sessions_missing_user_id"
CARS_SHARES = (
    "rate_nominal_cap_no_user_id",
    "nominal_cap_floor_max_charge_power_missing",
    "energy_before_no_charging_time",
)
HISTORY_SHARE = "rate_average_without_charging_time"


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _share_ok(value) -> bool:
    return _number(value) and 0.0 <= value <= 1.0


def _count(entry: dict, name: str):
    """A count of the unit's entry when it is an integer (not a bool, not a float), else None."""
    value = entry.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _undefined(key: str, entry: dict) -> str | None:
    """What the denominator of `key` counts, when it is exactly 0 for this unit; else None.
    The history share is judged from the cars table, not here."""
    if key == SESSIONS:
        return "sessions loaded" if _count(entry, "sessions_loaded") == 0 else None
    cars = _count(entry, "cars_on_counting_days")
    if key in CARS_SHARES:
        return "counting cars" if cars == 0 else None
    return None


def _history_cars(unit: str, entry: dict, tables: Path) -> int | str:
    """The counting cars of `unit` whose rate came from history, counted from its cars table; or
    a text saying why the table cannot be used (a hash that differs from the recorded one)."""
    path = tables / f"{unit}-cars.csv"
    if not path.is_file():
        raise tv.CouldNotRun(f"{path} is missing")
    recorded = (entry.get("sha256") or {}).get("cars")
    if tv._sha256(path) != recorded:
        return "cars table differs from the recorded build's table"
    cars = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"day_counts", "rate_rule"} <= set(cars.columns):
        return "cars table has no day_counts or rate_rule column"
    return int(((cars["day_counts"] == "True") & (cars["rate_rule"] == truth.RATE_HISTORY)).sum())


def verify_fallbacks(summary: dict, tables: Path) -> tuple[int, list[str]]:
    lines, failed = [], False
    for unit in UNITS["train"] + UNITS["development"]:
        entry = summary["units"].get(unit) or {}
        shares = entry.get("fallback_shares")
        if not isinstance(shares, dict):
            lines.append(f"FAILED: {unit}: no fallback shares logged")
            failed = True
            continue
        bad, shown, table_problem = [], [], None
        for key in FALLBACK_KEYS:
            value = shares.get(key)
            if _share_ok(value):
                continue
            counts = None
            if key in shares and value is None:
                if key != HISTORY_SHARE:
                    counts = _undefined(key, entry)
                else:
                    found = _history_cars(unit, entry, tables)
                    if isinstance(found, str):
                        table_problem = found
                    elif found == 0:
                        counts = "counting cars whose rate came from history"
            if counts is None:
                bad.append(key)
            else:
                shown.append(f"    {key} undefined: 0 of 0 ({counts})")
        if table_problem:
            lines.append(f"FAILED: {unit}: {table_problem}")
            failed = True
        elif bad:
            lines.append(f"FAILED: {unit}: fallback shares missing or outside 0 to 1: {bad}")
            failed = True
        else:
            lines.append(f"passed: {unit}: {json.dumps(shares, sort_keys=True)}")
            lines += shown
    return (1 if failed else 0), lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--setting", choices=tv.SETTINGS, default=tv.MAIN)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--tables", type=Path, default=None, help="default: by setting")
    args = parser.parse_args(argv)
    tables = args.tables if args.tables is not None else DEFAULT_OUT[args.setting]
    try:
        if not tv.RAW_ROOT.is_dir():
            raise tv.CouldNotRun("no data/raw folder in this checkout")
        run = (
            args.run
            if args.run is not None
            else tv.latest_run(tv.TOP / "runs.noindex", args.setting)
        )
        summary = tv.load_summary(run, tv._head_commit())
        if tv.summary_setting(summary) != args.setting:
            raise tv.CouldNotRun(
                f"the recorded truth build {run.name} is of the {tv.summary_setting(summary)} "
                f"setting, not the {args.setting} setting asked for"
            )
        tv.require_matching_setting(summary, tables, run.name)
        status, lines = verify_fallbacks(summary, tables)
    except tv.CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except tv.NotVerified as problem:
        print(f"FAILED: the recorded truth build is not verified: {problem}")
        return 1
    print(f"run: {run.name}")
    for line in lines:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
