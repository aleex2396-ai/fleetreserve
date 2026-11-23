"""The WP5 checks on the development days (plan section 13, WP5: "diag <= affine <= tree <=
hindsight every development day; T2's check; two-way corner tests match the LP; T4 log started,
split by in-window departure").

Command line, from the project's top folder (every forecast argument is required; none has a
default, so a run always names its configuration. Decided by the project owner: rate and energy
known (`--energy-bounds known`, on 16 Nov 2026) and a car with no usable forecast left unsteered
(`--uncertifiable-car leave-unsteered`, on 18 Nov 2026); the other values stay as comparisons.
The kappa for the gate is the plan's rule (section 7); the departure floor has no recorded
decision):
    python -m fleetreserve.dev_day_checks --t2-sample-size <n>
        --energy-bounds known|empirical --uncertifiable-car leave-unsteered|day-abstains
        --kappa <level> --departure-floor as-planned|floored
        [--run <wp2-truth run folder>] [--tables <folder>] [--seed 2026] [--workers 4]
        [--units <unit> ...]

Units: `nrel-before-may-2019`, `nrel-from-may-2019` and `norway-development-pooled` (`UNITS`), or
the units `--units` names: training or development units of the truth build only
(`unit_list.KNOWN_UNITS`, from `truth_build.TRAIN_UNITS` and `truth_build.DEVELOPMENT_UNITS`),
each once and never a pooled unit beside a site it pools, or the run could not run before
anything is read; a held-out unit's name is refused (`unit_list`).
Every counting day, rho = 0, true sets from `days_from_truth` (readings: room
capped, small fleets kept), as `ceiling_report` loads them. The truth tables are used only if they
are those of the recorded truth build at the checked-out commit (`ceiling_report.check_inputs`).

(i) Order (`dev_day_values`, `dev_day_verdicts.order_check`), every counting day, one-way.
(ii) T4 log (`dev_day_verdicts.t4_log`), every counting day, block product.
(iii) T2's check (`t2_day`), on a seeded sample of `--t2-sample-size` counting days (the pass mark
    uses 100; a run with another size says so and cannot satisfy it). The sample uses the
    allocation and draw of `block_sample` (proportional to each unit's counting days, at least one
    per unit; smallest SHA-256 ranks), with the unit name prefixed by `SAMPLE_LABEL` in the hash,
    so it is not the block sample's 50 days by construction. Forecasts: `forecast_empirical`
    (reference `other-days`, cells of at least `MIN_CELL` cars, empirical departure quantiles at
    `--kappa`); rate and energy `known` (the car's own) or `empirical` (`--energy-bounds`);
    departure `as-planned` (the kappa-quantile in fractional hours) or `floored` (rounded down to
    the whole hour); cars without a forecast or with an empty forecast set handled by
    `--uncertifiable-car`.
(iv) Two-way corner tests (`two_way_day`) at B = 5 kWh on the same sampled days.

Per-day work runs in `--workers` processes (1 runs in this process); the results files do not
depend on the worker count: days are sorted by unit, date and site, and no timing is in them.

Outputs in `FLEETRESERVE_RUN_DIR` when it is set (the summary is always printed):
`dev-day-checks.json` (configuration, counts, the verdict of each check with its lines),
`dev-days.csv` (the order values and T4 flags of every counting day), `t2-sample.csv` (T2 and
two-way per sampled day), `dev-day-checks.txt` (the summary, any FAILED line first) and
`dev-day-seconds.json` (solve seconds and the worker count; differs between runs).

Exit status: 1 a check failed (a FAILED line); else 2 a check could not run (a timeout or another
missing value, a check that looked at nothing) or the command could not run (a unit list
`unit_list` refuses, a unit the truth build does not hold, a table missing, no verified truth
build at this commit, a table differing from it, fewer counting days than the sample, a bad
argument); else 0, every check passed.
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from fleetreserve.block_sample import DEFAULT_SEED, allocate, day_text, draw
from fleetreserve.ceiling_report import check_inputs, write_csv
from fleetreserve.ceiling_values import DayTask
from fleetreserve.days_from_truth import NO_FIT_CAP, SMALL_KEEP, UnitDays, load_unit_days
from fleetreserve.dev_day_values import ORDER_KEYS, OrderValues, order_values
from fleetreserve.dev_day_verdicts import (
    COULD_NOT_RUN,
    FAILED,
    T4_GROUPS,
    label,
    order_check,
    t2_check,
    t4_log,
    t4_rows,
    two_way_check,
)
from fleetreserve.forecast_empirical import MIN_CELL, EmpiricalForecaster
from fleetreserve.t2_day import UNCERTIFIABLE_RULES, T2Result, T2Task, check_day
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun
from fleetreserve.two_way_day import BUDGET, PRODUCTS, VALUE_NAMES, TwoWayResult, two_way_day
from fleetreserve.unit_list import add_units_argument, check_units

UNITS = ("nrel-before-may-2019", "nrel-from-may-2019", "norway-development-pooled")
RHO = 0.0
PASS_MARK_SAMPLE_SIZE = 100  # plan section 5, T2's check: "on 100 development days"
SAMPLE_LABEL = "t2-check"
ENERGY_BOUNDS = {"known": "known", "empirical": "empirical"}  # -> forecast_empirical mode
DEPARTURE_FLOORS = ("as-planned", "floored")
REFERENCE = "other-days"
RESULTS_NAME = "dev-day-checks.json"
TEXT_NAME = "dev-day-checks.txt"
DAYS_NAME = "dev-days.csv"
SAMPLE_NAME = "t2-sample.csv"
SECONDS_NAME = "dev-day-seconds.json"
CHECKS = ("order", "t4_log", "t2", "two_way")


@dataclass(frozen=True)
class DevDayTask:
    day: DayTask
    t2: T2Task | None  # set on sampled days only
    sampled: bool

    @property
    def key(self) -> tuple[str, str, str]:
        return self.day.key


@dataclass(frozen=True)
class DevDayResult:
    order: OrderValues
    t2: T2Result | None
    two_way: TwoWayResult | None


def compute(task: DevDayTask) -> DevDayResult:
    order = order_values(task.day)
    if not task.sampled:
        return DevDayResult(order, None, None)
    t2 = check_day(task.t2) if task.t2 is not None else None
    d = task.day
    return DevDayResult(order, t2, two_way_day(d.unit, d.site, d.date, d.day))


def sample(units: dict[str, UnitDays], size: int, seed: int) -> tuple[dict, dict]:
    """The sampled day texts of each unit and the sample's description."""
    listed = {
        name: [day_text(t.record.site, t.record.date) for t in unit.days]
        for name, unit in units.items()
    }
    shares = allocate({name: len(days) for name, days in listed.items()}, size)
    chosen = {
        name: draw(f"{SAMPLE_LABEL} {name}", seed, listed[name], shares[name])
        for name in sorted(listed)
    }
    return chosen, {
        "size": size,
        "pass_mark_size": PASS_MARK_SAMPLE_SIZE,
        "seed": seed,
        "label": SAMPLE_LABEL,
        "reading": "proportional to each unit's counting days, at least one per unit "
        "(block_sample.allocate); within a unit, the smallest SHA-256 ranks of "
        f"('{SAMPLE_LABEL} <unit>', seed, site and date) (block_sample.draw)",
        "counting_days": {name: len(days) for name, days in listed.items()},
        "sampled_days": {name: len(days) for name, days in chosen.items()},
    }


def t2_task(forecaster: EmpiricalForecaster, truth, args) -> T2Task:
    day = truth.day
    forecasts = []
    for i in range(len(day.cars)):
        found = forecaster.forecast_car(
            truth, i, args.kappa, ENERGY_BOUNDS[args.energy_bounds], departure="empirical"
        )
        if found is not None and args.departure_floor == "floored":
            found = replace(found, departure=float(math.floor(found.departure)))
        forecasts.append(found)
    return T2Task(
        unit=day.unit,
        site=truth.record.site,
        date=truth.record.date,
        n_service=day.n_service,
        n_recovery=day.n_recovery,
        rho=day.rho,
        cars=day.cars,
        forecasts=tuple(forecasts),
        exact_departures=tuple(car.departure_exact for car in truth.fleet),
        uncertifiable=args.uncertifiable_car,
    )


def build_tasks(units: dict[str, UnitDays], args) -> tuple[list[DevDayTask], dict]:
    chosen, plan = sample(units, args.t2_sample_size, args.seed)
    tasks = []
    for name, unit in units.items():
        picked = set(chosen[name])
        forecaster = EmpiricalForecaster(unit, reference=REFERENCE, min_cell=MIN_CELL)
        for truth in unit.days:
            site, day_date = truth.record.site, truth.record.date
            sampled = day_text(site, day_date) in picked
            day_task = DayTask(name, site, day_date, truth.day, sampled)
            t2 = t2_task(forecaster, truth, args) if sampled else None
            tasks.append(DevDayTask(day_task, t2, sampled))
    tasks.sort(key=lambda task: task.key)
    return tasks, plan


def run_days(tasks: list[DevDayTask], workers: int) -> list[DevDayResult]:
    """The results in the order of `tasks`, whatever the worker count."""
    if workers <= 1:
        return [compute(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(compute, tasks, chunksize=1))


def _num(value) -> str:
    return "" if value is None else repr(round(float(value), 9))


DAYS_HEADER = [
    "unit",
    "site",
    "date",
    "cars",
    "service_hours",
    "in_window_departure",
    "meets_t4_hypotheses",
    "in_t2_sample",
    *(f"{k}_kwh" for k in ORDER_KEYS),
    *(f"{k}_outcome" for k in ORDER_KEYS),
    "block_gap_hind_minus_tree_kwh",
    "block_gap_tree_minus_affine_kwh",
]


def days_rows(results: list[DevDayResult]) -> list[list]:
    rows = []
    t4 = t4_rows([r.order for r in results])
    for r, log in zip(results, t4, strict=True):
        o = r.order
        rows.append(
            [
                o.unit,
                o.site,
                o.date,
                o.cars,
                o.n_service,
                o.in_window,
                o.t4_hypotheses,
                r.t2 is not None or r.two_way is not None,
                *(_num(o.values[k]) for k in ORDER_KEYS),
                *(o.outcomes[k] for k in ORDER_KEYS),
                _num(log["gap_hind_minus_tree"]),
                _num(log["gap_tree_minus_affine"]),
            ]
        )
    return rows


T2_COLUMNS = (
    "status",
    "steered",
    "steered_in_e_prime",
    "day_in_e_prime",
    "cars_without_forecast",
    "cars_with_empty_set",
    "rows_checked",
    "replay_mismatches",
    "cars_checked_in_e_prime",
    "true_row_violations_in_e_prime",
    "cars_outside_e_prime_breaking_a_true_row",
    "paths_with_request_missed",
)
TWO_WAY_COLUMNS = [f"{p}_{k}" for p in PRODUCTS for k in VALUE_NAMES]
SAMPLE_HEADER = [
    "unit",
    "site",
    "date",
    "cars",
    *(f"t2_{c}" for c in T2_COLUMNS),
    "t2_hourly_certified_kwh",
    *(f"two_way_{c}_kwh" for c in TWO_WAY_COLUMNS),
    *(f"two_way_{p}_matched" for p in PRODUCTS),
    *(f"two_way_{p}_scale_with_shape_fixed" for p in PRODUCTS),
]


def sample_rows(results: list[DevDayResult]) -> list[list]:
    rows = []
    for r in results:
        if r.two_way is None:
            continue
        o, t2, tw = r.order, r.t2, r.two_way
        t2_values = [getattr(t2, c) if t2 is not None else "" for c in T2_COLUMNS]
        tests = tw.tests
        rows.append(
            [
                o.unit,
                o.site,
                o.date,
                o.cars,
                *t2_values,
                _num(t2.value) if t2 is not None else "",
                *(
                    _num(tests[p].values[k]) if p in tests else ""
                    for p in PRODUCTS
                    for k in VALUE_NAMES
                ),
                *(tests[p].matched if p in tests else "" for p in PRODUCTS),
                *(_num(tests[p].enlarged_scale) if p in tests else "" for p in PRODUCTS),
            ]
        )
    return rows


def day_seconds(result: DevDayResult) -> dict:
    """The solve seconds of one day, by value."""
    found = {k: round(v, 4) for k, v in sorted(result.order.seconds.items())}
    if result.t2 is not None:
        found["t2_certificate"] = round(result.t2.seconds, 4)
    if result.two_way is not None:
        found |= {f"two_way_{k}": round(v, 4) for k, v in sorted(result.two_way.seconds.items())}
    return found


def _verdict_line(name: str, check: dict) -> str:
    return (
        f"{name}: {check['verdict'].upper()}; covered {check['covered_days']} of "
        f"{check['of_days']} days ({check['timeout_days']} with a timeout, "
        f"{check['other_missing_days']} with another missing value)"
    )


def report_lines(document: dict) -> list[str]:
    checks = document["checks"]
    failures = [line for c in CHECKS for line in checks[c]["failure_lines"]]
    lines = [f"FAILED: {line}" for line in failures]
    if failures:
        lines.append("")
    config = document["configuration"]
    plan = document["sample"]
    lines += [
        f"truth build {document['inputs']['truth_run']}; rho {config['rho']}; true sets.",
        f"T2 configuration: energy bounds {config['energy_bounds']}, uncertifiable car "
        f"{config['uncertifiable_car']}, kappa {config['kappa']}, departure "
        f"{config['departure_floor']}; two-way budget {config['budget_kwh']} kWh.",
        f"T2 sample: {plan['size']} days, seed {plan['seed']}, label {plan['label']}; "
        + ", ".join(f"{u} {n}" for u, n in plan["sampled_days"].items()),
    ]
    if plan["size"] != PASS_MARK_SAMPLE_SIZE:
        lines.append(
            f"NOTE: the T2 sample has {plan['size']} days, not the pass mark's "
            f"{PASS_MARK_SAMPLE_SIZE}; this run cannot satisfy the pass mark"
        )
    lines.append("")
    names = {
        "order": "(i) order diag <= affine <= tree <= hindsight, every day, both products",
        "t4_log": "(ii) T4 log, block product, every day",
        "t2": "(iii) T2's check, sampled days",
        "two_way": f"(iv) two-way corner tests at B = {config['budget_kwh']} kWh, sampled days",
    }
    for c in CHECKS:
        lines.append(_verdict_line(names[c], checks[c]))
        lines += [f"  could not run: {reason}" for reason in checks[c]["could_not_run_reasons"]]
    t4 = checks["t4_log"]
    lines += [
        "",
        f"T4 log: {t4['hypothesis_days']} days meet T4's hypotheses; V_hind = V_caus checked on "
        f"{t4['hypothesis_days_checked']} of them",
    ]
    for name in T4_GROUPS:
        g = t4["groups"][name]
        lines.append(
            f"  {name}: {g['days']} days; tree below hindsight on {g['days_tree_below_hind']} "
            f"of {g['days_with_hind_and_tree']} (mean gap {g['mean_gap_hind_minus_tree_kwh']} "
            f"kWh); affine below tree on {g['days_affine_below_tree']} of "
            f"{g['days_with_tree_and_affine']} (mean gap {g['mean_gap_tree_minus_affine_kwh']} "
            "kWh)"
        )
    if t4["findings_tree_above_affine"]:
        lines.append(
            "  FINDING (the plan's open conjecture T4 says V_aff = V_caus; recorded, not a "
            f"failure): V_caus above V_aff on {len(t4['findings_tree_above_affine'])} days:"
        )
        lines += [f"    {line}" for line in t4["findings_tree_above_affine"]]
    t2 = checks["t2"]
    lines += [
        "",
        f"T2: certified on {t2['certified_days']} of {t2['of_days']} sampled days; days in E': "
        f"{t2['days_in_e_prime']} (share of sampled days {t2['share_of_sampled_days_in_e_prime']}"
        f", of certified days {t2['share_of_certified_days_in_e_prime']}); share of steered cars "
        f"in E'_i {t2['share_of_steered_cars_in_e_prime_i']} of "
        f"{t2['steered_cars_on_certified_days']}",
        f"  days without a certificate by the rule: {t2['days_without_certificate_by_rule']}; "
        f"cars left unsteered: {t2['cars_left_unsteered_no_forecast']} without a forecast, "
        f"{t2['cars_left_unsteered_empty_forecast_set']} with an empty forecast set, of "
        f"{t2['cars_on_sampled_days']} cars",
        f"  (a) closed form against replay: {t2['replay_mismatches']} mismatches on "
        f"{t2['forecast_rows_checked']} forecast rows",
        f"  (b) cars in E'_i breaking a true row: {t2['true_row_violations_in_e_prime_i']} of "
        f"{t2['cars_checked_in_e_prime_i']}; corner paths with a request missed: "
        f"{t2['paths_with_request_missed']}; cars outside E'_i breaking a true row (counted, "
        f"not judged): {t2['cars_outside_e_prime_i_breaking_a_true_row']}",
        "",
        "two-way: both products' trees matched on "
        f"{checks['two_way']['days_both_products_matched']} of {checks['two_way']['of_days']} "
        "sampled days",
        "",
    ]
    verdicts = [checks[c]["verdict"] for c in CHECKS]
    if FAILED in verdicts:
        lines.append("CHECKS FAILED")
    elif COULD_NOT_RUN in verdicts:
        lines.append("COULD NOT RUN: a check could not run")
    else:
        lines.append("all checks passed")
    return lines


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--t2-sample-size", type=int, required=True)
    parser.add_argument("--energy-bounds", choices=tuple(ENERGY_BOUNDS), required=True)
    parser.add_argument("--uncertifiable-car", choices=UNCERTIFIABLE_RULES, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--departure-floor", choices=DEPARTURE_FLOORS, required=True)
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=4)
    add_units_argument(parser, UNITS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if not 0 < args.kappa < 1 or args.t2_sample_size < 1 or args.workers < 1:
        print("COULD NOT RUN: --kappa must be in (0, 1), --t2-sample-size and --workers >= 1")
        return 2
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir and not Path(run_dir).is_dir():
        print(f"COULD NOT RUN: the run folder {run_dir} does not exist")
        return 2
    try:
        names = check_units(args.units)
        inputs = check_inputs(args.run, args.tables, names)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        units = {
            unit: load_unit_days(
                args.tables, unit, rho=RHO, no_fit=NO_FIT_CAP, small_fleet=SMALL_KEEP
            )
            for unit in names
        }
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    try:
        tasks, plan = build_tasks(units, args)
    except ValueError as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    start = time.perf_counter()
    results = run_days(tasks, args.workers)
    wall = time.perf_counter() - start
    sampled = [r for r in results if r.two_way is not None]
    unbuilt = sum(r.t2 is None for r in sampled)
    checks = {
        "order": order_check([r.order for r in results]),
        "t4_log": t4_log([r.order for r in results]),
        "t2": t2_check([r.t2 for r in sampled if r.t2 is not None], plan["size"], unbuilt),
        "two_way": two_way_check([r.two_way for r in sampled], plan["size"], 0),
    }
    document = {
        "inputs": inputs,
        "configuration": {
            "rho": RHO,
            "budget_kwh": BUDGET,
            "energy_bounds": args.energy_bounds,
            "uncertifiable_car": args.uncertifiable_car,
            "kappa": args.kappa,
            "departure_floor": args.departure_floor,
            "forecast_departure": "empirical kappa-quantile by site and arrival hour",
            "forecast_reference": REFERENCE,
            "forecast_min_cell": MIN_CELL,
        },
        "sample": plan,
        "counting_days": len(results),
        "checks": checks,
        "t4_log_days": t4_rows([r.order for r in results]),
    }
    text = "\n".join(report_lines(document)) + "\n"
    print(text, end="")
    if run_dir:
        folder = Path(run_dir)
        (folder / RESULTS_NAME).write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        (folder / TEXT_NAME).write_text(text)
        write_csv(folder / DAYS_NAME, DAYS_HEADER, days_rows(results))
        write_csv(folder / SAMPLE_NAME, SAMPLE_HEADER, sample_rows(results))
        seconds = {
            "workers": args.workers,
            "wall_seconds": round(wall, 3),
            "days": {label(r.order): day_seconds(r) for r in results},
        }
        (folder / SECONDS_NAME).write_text(json.dumps(seconds, indent=1, sort_keys=True) + "\n")
    verdicts = [checks[c]["verdict"] for c in CHECKS]
    if FAILED in verdicts:
        return 1
    return 2 if COULD_NOT_RUN in verdicts else 0


if __name__ == "__main__":
    sys.exit(main())
