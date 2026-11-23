"""Empirical forecast bounds on the training and development truth tables: coverage, empty forecast
sets and the hourly ceiling, per unit, kappa and mode (input to plan 13 WP6).

Command line, from the project's top folder:
    python -m fleetreserve.forecast_report [--run <wp2-truth run folder>] [--tables <folder>]
        [--reference other-days|earlier-days] [--min-cell 20] [--rho 0] [--units <unit> ...]

Units: the three gate units (`GATE_UNITS`) and, for comparison, caltech and jpl (`UNITS`), or the
units `--units` names: training or development units of the truth build only
(`unit_list.KNOWN_UNITS`, from `truth_build.TRAIN_UNITS` and `truth_build.DEVELOPMENT_UNITS`),
each once and never a pooled unit beside a site it pools, or the run could not run before
anything is read; a held-out unit's name is refused (`unit_list`).
A unit of `GATE_UNITS` is marked as a gate unit in the report. Each unit's days
come from `days_from_truth` (readings: room capped, small fleets kept; the cars the other readings
would change are counted), its forecasts from `forecast_empirical` at every kappa of plan section 2,
every mode, and both the empirical and the oracle departure, and the checks from
`forecast_coverage`. The hourly ceiling of each day is `exact.hourly_ceiling` on its true sets
(one-way, hour 0 free). No offer, certificate or replay is computed.

The truth tables are used only if they are those of the recorded truth build at the checked-out
commit: the latest `runs.noindex/*-wp2-truth` run (or `--run`) must pass `run_verify` and be at
the commit checked out now (as `truth_verify` decides), and each table's SHA-256 must be the one
that build recorded. The build's setting (main, or held: plan 3) must be the one the tables folder
holds, or the run could not run; the setting is written under `inputs` in the results.

Outputs: the report on standard output; when `FLEETRESERVE_RUN_DIR` is set (the run recorder sets
it), also `forecast-report.json` (every number) and `forecast-report.txt` (the report) there.

Exit status: 0 the report is written; 1 a day could not be built from the tables; 2 could not run
(a unit list `unit_list` refuses, a unit the truth build does not hold, a table missing, no
verified truth build at this commit, a table differing from that build).
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

from fleetreserve import exact
from fleetreserve.days_from_truth import (
    NO_FIT_CAP,
    NO_FIT_LEAVE_OUT,
    ROOM_TOLERANCE,
    SMALL_KEEP,
    UnitDays,
    load_unit_days,
)
from fleetreserve.forecast_coverage import check_car
from fleetreserve.forecast_empirical import (
    DEPARTURES,
    KAPPAS,
    MIN_CELL,
    MODES,
    REFERENCES,
    EmpiricalForecaster,
)
from fleetreserve.run_verify import NotVerified
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT, read_setting
from fleetreserve.truth_verify import (
    CouldNotRun,
    _head_commit,
    latest_run,
    load_summary,
    require_matching_setting,
)
from fleetreserve.unit_list import add_units_argument, check_units

GATE_UNITS = ("nrel-before-may-2019", "nrel-from-may-2019", "norway-development-pooled")
UNITS = (*GATE_UNITS, "caltech", "jpl")
RESULTS_NAME = "forecast-report.json"
TEXT_NAME = "forecast-report.txt"
CHECKS = ("departure", "departure_exact", "departure_floor", "rate", "cap", "need")


def _share(part, whole) -> float | None:
    return None if whole == 0 else round(part / whole, 6)


def _spread(values) -> dict | None:
    if not values:
        return None
    return {
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
    }


def check_inputs(run: Path | None, tables: Path, units: Sequence[str] = UNITS) -> dict:
    """The verified truth build's summary and the hashes of the tables of `units`; raises
    `CouldNotRun` when the tables cannot be used."""
    folder = run if run is not None else latest_run(TOP / "runs.noindex", read_setting(tables))
    try:
        summary = load_summary(folder, _head_commit())
    except NotVerified as problem:
        raise CouldNotRun(f"the recorded truth build is not verified: {problem}") from problem
    setting = require_matching_setting(summary, tables, folder.name)
    hashes = {}
    for unit in units:
        entry = summary["units"].get(unit)
        if entry is None:
            raise CouldNotRun(f"{unit} is not in the recorded truth build {folder.name}")
        for kind in ("cars", "days"):
            path = tables / f"{unit}-{kind}.csv"
            if not path.is_file():
                raise CouldNotRun(f"{path} is missing")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry["sha256"][kind]:
                raise CouldNotRun(
                    f"{path.name} differs from the table the recorded build {folder.name} wrote"
                )
            hashes[f"{unit}-{kind}.csv"] = digest
    return {"setting": setting, "truth_run": folder.name, "table_sha256": hashes}


def multi_day(unit: UnitDays) -> dict:
    """Rows of counting days whose car was plugged in on an earlier calendar day."""
    table = unit.cars
    counting = table[table["counting"]]
    earlier = counting[counting["arrived_earlier_day"]]
    first_date = table.groupby(["site", "session_id"])["date"].transform("min")
    repeat = counting[counting["date"] > first_date.loc[counting.index]]
    with_room = repeat[repeat["room_table"] > ROOM_TOLERANCE]
    return {
        "counting_rows_plugged_in_on_an_earlier_day": len(earlier),
        "counting_rows_of_a_session_listed_on_an_earlier_day": len(repeat),
        "of_those_with_room": len(with_room),
        "their_room_kwh": round(float(with_room["room_table"].sum()), 6),
        "of_those_with_room_by_energy_before_rule": {
            str(k): int(v) for k, v in with_room["energy_before_rule"].value_counts().items()
        },
    }


def ceilings(unit: UnitDays) -> dict:
    per_day, per_car = [], []
    for truth in unit.days:
        value = exact.hourly_ceiling(exact.day_functions(truth.day))
        per_day.append(value)
        if truth.day.cars:
            per_car.append(value / len(truth.day.cars))
    return {"kwh_per_day": _spread(per_day), "kwh_per_car_per_day": _spread(per_car)}


def forecast_table(unit: UnitDays, forecaster: EmpiricalForecaster) -> list[dict]:
    rows = []
    for kappa in KAPPAS:
        for mode in MODES:
            for departure in DEPARTURES:
                rows.append(_one(unit, forecaster, kappa, mode, departure))
    return rows


def _one(unit, forecaster, kappa, mode, departure) -> dict:
    held = dict.fromkeys(CHECKS, 0)
    cars = all_cars = inside = empty = empty_inside = empty_a = empty_b = fallback = 0
    days = days_e = days_empty = skipped = 0
    for truth in unit.days:
        if not truth.day.cars:
            continue
        found = forecaster.forecast_day(truth, kappa, mode, departure)
        if found is None:
            skipped += 1
            continue
        day = found.day
        days += 1
        day_holds, day_empty = True, False
        for i, (forecast, true) in enumerate(zip(day.forecasts, day.cars, strict=True)):
            check = check_car(
                forecast,
                true.departure,
                truth.fleet[i].departure_exact,
                true.rate,
                true.room,
                day.rho,
                day.n_service,
                day.n_recovery,
            )
            all_cars += 1
            for name in CHECKS:
                held[name] += getattr(check, name)
            cars += check.all_hold
            day_holds &= check.all_hold
            inside += check.inside_window
            empty += check.empty
            empty_inside += check.empty and check.inside_window
            empty_a += check.empty_a
            empty_b += check.empty_b
            day_empty |= check.empty
            fallback += found.fallback[i]
        days_e += day_holds
        days_empty += day_empty
    return {
        "kappa": kappa,
        "mode": mode,
        "departure": departure,
        "days_with_forecasts": days,
        "days_skipped_no_reference": skipped,
        "cars": all_cars,
        "share_conservative": {name: _share(held[name], all_cars) for name in CHECKS},
        "share_cars_in_e_prime": _share(cars, all_cars),
        "share_days_in_e_prime": _share(days_e, days),
        "cars_forecast_inside_window": inside,
        "share_cars_empty_forecast_set": _share(empty, all_cars),
        "share_cars_empty_inside_window": _share(empty_inside, all_cars),
        "share_cars_empty_outside_window": _share(empty - empty_inside, all_cars),
        "share_cars_empty_condition_a": _share(empty_a, all_cars),
        "share_cars_empty_condition_b": _share(empty_b, all_cars),
        "share_days_with_an_empty_forecast_set": _share(days_empty, days),
        "share_cars_site_fallback": (
            _share(fallback, all_cars) if departure == "empirical" else None
        ),
    }


def unit_entry(tables: Path, unit: str, rho: float, reference: str, min_cell: int) -> dict:
    main = load_unit_days(tables, unit, rho=rho, no_fit=NO_FIT_CAP, small_fleet=SMALL_KEEP)
    other = load_unit_days(tables, unit, rho=rho, no_fit=NO_FIT_LEAVE_OUT, small_fleet=SMALL_KEEP)
    readings = main.counts()
    alternative = other.counts()
    readings["under_leave_out_days_fewer_than_three"] = alternative[
        "reading3_days_fewer_than_three"
    ]
    readings["under_leave_out_days_empty_fleet"] = alternative["reading3_days_empty_fleet"]
    sizes = [len(t.day.cars) for t in main.days]
    forecaster = EmpiricalForecaster(main, reference=reference, min_cell=min_cell)
    return {
        "counting_days": len(main.days),
        "cars_per_day": _spread(sizes),
        "readings": readings,
        "multi_day_sessions": multi_day(main),
        "hourly_ceiling": ceilings(main),
        "forecasts": forecast_table(main, forecaster),
    }


def _pct(value) -> str:
    return "  -  " if value is None else f"{100 * value:5.1f}"


def report_lines(document: dict) -> list[str]:
    lines = [
        f"truth build {document['inputs']['truth_run']}; rho {document['rho']}; reference "
        f"{document['reference']}; minimum cell {document['min_cell']}",
        "Shares in percent. E' cars/days: all four bounds conservative. Empty: no power vector "
        "meets the forecast set's rows (inside / outside: forecast departure before / after the "
        "end of the last service hour; (a) e_need > e_cap + recovery supply, (b) e_need > "
        "r_hat times the hours before d_hat). Days empty: days with at least one empty forecast "
        "set.",
        "",
    ]
    for unit, entry in document["units"].items():
        r = entry["readings"]
        ceiling = entry["hourly_ceiling"]["kwh_per_day"] or {}
        per_car = entry["hourly_ceiling"]["kwh_per_car_per_day"] or {}
        lines += [
            f"UNIT {unit}{' (gate unit)' if unit in GATE_UNITS else ''}",
            f"  counting days {entry['counting_days']}; cars per day {entry['cars_per_day']}",
            f"  reading 1 (departure in hour 0, left out): {r['reading1_depart_zero_cars']} cars, "
            f"{r['reading1_depart_zero_room_kwh']} kWh room",
            f"  reading 2 (protected energy does not fit): {r['reading2_no_fit_cars']} cars; "
            f"capped: {r['reading2_cap_energy_removed_kwh']} kWh removed; left out instead: "
            f"{r['reading2_leave_out_room_kwh']} kWh room",
            f"  reading 3 (fewer than 3 cars after the rules, kept): "
            f"{r['reading3_days_fewer_than_three']} days ({r['reading3_days_empty_fleet']} "
            f"empty); under leave-out {r['under_leave_out_days_fewer_than_three']} "
            f"({r['under_leave_out_days_empty_fleet']} empty)",
            f"  multi-day sessions: {entry['multi_day_sessions']}",
            f"  hourly ceiling kWh per day mean {ceiling.get('mean')} median "
            f"{ceiling.get('median')}; per car mean {per_car.get('mean')} median "
            f"{per_car.get('median')}",
            "  kappa mode       departure | E' cars days | d_hat  r_hat  e_cap  e_need |"
            " empty all  in   out  (a)   (b)  | days empty | fallback",
        ]
        for row in entry["forecasts"]:
            s = row["share_conservative"]
            lines.append(
                f"  {row['kappa']:<5} {row['mode']:<10} {row['departure']:<9} |"
                f" {_pct(row['share_cars_in_e_prime'])} {_pct(row['share_days_in_e_prime'])} |"
                f" {_pct(s['departure'])}  {_pct(s['rate'])}  {_pct(s['cap'])}  "
                f"{_pct(s['need'])} |"
                f" {_pct(row['share_cars_empty_forecast_set'])}"
                f" {_pct(row['share_cars_empty_inside_window'])}"
                f" {_pct(row['share_cars_empty_outside_window'])}"
                f" {_pct(row['share_cars_empty_condition_a'])}"
                f" {_pct(row['share_cars_empty_condition_b'])} |"
                f" {_pct(row['share_days_with_an_empty_forecast_set'])}      |"
                f" {_pct(row['share_cars_site_fallback'])}"
            )
        lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--reference", choices=REFERENCES, default="other-days")
    parser.add_argument("--min-cell", type=int, default=MIN_CELL)
    parser.add_argument("--rho", type=float, default=0.0)
    add_units_argument(parser, UNITS)
    args = parser.parse_args(argv)
    try:
        names = check_units(args.units)
        inputs = check_inputs(args.run, args.tables, names)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    document = {
        "inputs": inputs,
        "rho": args.rho,
        "reference": args.reference,
        "min_cell": args.min_cell,
        "units": {},
    }
    try:
        for unit in names:
            document["units"][unit] = unit_entry(
                args.tables, unit, args.rho, args.reference, args.min_cell
            )
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    text = "\n".join(report_lines(document)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / RESULTS_NAME).write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        (folder / TEXT_NAME).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
