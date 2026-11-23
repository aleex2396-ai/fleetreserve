"""T3's check (plan section 4: "Check: tree matches replay on every audited path"): the one-way
corner tree's own policy replayed hour by hour on the true cars of every development day, along
every audited path of the day.

Command line, from the project's top folder:
    python -m fleetreserve.t3_replay --truth-run <wp2-truth run folder> [--tables <folder>]
        [--workers 4] [--units <unit> ...]

Per counting day and product (`PRODUCTS`, reading R2):

1. The tree. `tree.corner_tree` on the day's true one-way sets (`carsets.true_rows`, rho 0), with
   the time limit for other programmes (`TREE_SECONDS`, 300 s, plan section 13). Its value is
   V_tree; its policy (`tree.TreePolicy`) holds every car's power at every node of the tree.
2. The paths: the day's audited set (`audit_paths.audited_paths(day.unit, day.date, n)`), the
   same call the gate makes (`gate_day.prepare_day`, step 6), so the same 8 rows, repeats
   included, counted as rows (as `audit_paths` says a caller counts them).
3. The replay (`replay_path`), hour by hour, hours 0 (the decision hour, the root node) to n. At
   the start of hour `t >= 1` the corner `zeta_t` is revealed and the node one step down the tree
   is taken from the node of hour `t - 1` and `zeta_t` alone, so the power of hour `t` is read
   from what is known at that hour. Each car is commanded the node's power and draws it as far as
   its true car can (reading R3): the command is held within the car's physical rows that close in
   that hour (`PHYSICAL_KINDS`: power limits, 0 after its departure, and the stored-energy
   limits), given what it drew before.
4. Per path: (a) every request met: in every service hour the cars' summed draw is within
   `REQUEST_TOLERANCE` (1e-4 kWh, the plan's failure threshold, section 1) of the request
   `m_t + w_t zeta_t`; (b) every true row of every car holds on the draws within `ROW_TOLERANCE`
   (1e-6 kWh), the protected-energy row (`need`) included, which no cap enforces; (c) the
   replayed value, `2 sum_t zeta_t (D_t - m_t)` with `D_t` the summed draw (the box width the
   replay realised on the path: above the centre at an upper corner, below it at a lower corner),
   is within `VALUE_TOLERANCE` of V_tree (reading R4); (d) no command was cut: in every hour each
   car's draw is within `ROW_TOLERANCE` of its command, so the tree's own powers, not the true
   cars' limits, meet the requests. The path matches when (a) to (d) hold, and is a mismatch
   otherwise; a cut is named with the car (its position in the day's fleet, from 0), the hour and
   its size in kW.

A tree that does not end optimal (a timeout or any other outcome) leaves the day's V_tree missing
for that product: each of its audited rows is counted missing, never a match and never zero, with
the outcome kept. A day whose true sets cannot be written (`ValueError`) is missing the same way.
A day with no car in the fleet has nothing to replay: it is counted as an empty day, and none of
its rows is checked, matched or missing (reading R5).

Readings the plan leaves open, each with its alternative in brackets:

R1. The days: every counting day of the units named, by default the three development units the
    gate and the WP5 checks read (`UNITS`: the two NREL regimes and the pooled Norwegian
    development sites), rho 0, readings room capped and small fleets kept (`days_from_truth`,
    as `dev_day_checks` loads them). The pooled unit seeds the audited paths with its own name,
    as the gate does. Only development units of the truth build may be named
    (`truth_build.DEVELOPMENT_UNITS`): training units are refused as well as held-out units.
    [A seeded sample of the days, as T2's check's 100; or each Norwegian site as its own unit,
    which would seed other paths than the gate's.]
R2. The products: both, hourly (the offered product) and block (whose V_caus enters O2 and the
    ledger's block terms), each with its own tree and replay. [Hourly only.]
R3. The replay draws what a true car can take: a command above a car's power or stored-energy
    limit is cut to it, a command below its power floor raised to it, and the shortfall shows as a
    missed request. A command is never raised to meet the protected energy; that row is checked
    only. A cut above `ROW_TOLERANCE` is itself a mismatch (check (d)): cuts of opposite sign in
    one hour can leave the summed draw on the request, so a tree whose powers break a car's
    limits would otherwise read as a match. The draws after a cut are those the true cars make.
    [The command replayed as given, a command outside a physical row by more than
    `ROW_TOLERANCE` then a broken row, so a mismatch.] The cuts above the tolerance are counted
    per path and the largest is named.
R4. `VALUE_TOLERANCE` is `2 * REQUEST_TOLERANCE` per service hour (`2 n 1e-4` kWh on a day of
    `n` service hours), the gap the plan's failure threshold allows at each end of the box. With
    (a) met, (c) then holds by construction for the tree's own policy, whose V_tree is
    `2 sum_t w_t`; (c) is recorded as the plan asks and catches a V_tree other than the
    policy's box. [A fixed 1e-6 kWh, at the solver's own accuracy, which would count round-off
    of the summed draws as a mismatch.]
R5. A day with no car is not a checked day: the exit status does not depend on it. [Its rows
    counted as matches: no request, no car, nothing missed.]

Outputs in `FLEETRESERVE_RUN_DIR` when it is set (the summary is always printed):
`t3-replay.json` (inputs, configuration, counts per unit and product, every mismatch and missing
line), `t3-replay-paths.csv` (one row per day, product and audited row), `t3-replay.txt` (the
summary: days, paths, matches, mismatches, missing; any MISMATCH line first, then MISSING lines)
and `t3-replay-seconds.json` (solve seconds and the worker count; differs between runs). The
first three do not depend on the worker count. Recorded with `run_record`, which writes the
manifest of every file in the run folder.

Exit status: 0 every checked row matched and nothing is missing; 1 a row did not match (or a day
could not be built from the tables); 2 could not run (a missing value, nothing checked, a unit
refused, a file refused before it is opened (`gate_run.check_files`), a table missing, no verified
truth build at this commit, a table differing from it, a bad argument).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fleetreserve import gate_run, products
from fleetreserve.audit_paths import audited_paths
from fleetreserve.carsets import FleetRows, true_rows
from fleetreserve.ceiling_report import write_csv
from fleetreserve.day import Day
from fleetreserve.days_from_truth import NO_FIT_CAP, SMALL_KEEP, parse_unit_days
from fleetreserve.lp import OTHER_SECONDS
from fleetreserve.products import Product
from fleetreserve.tree import TreePolicy, corner_tree
from fleetreserve.truth_build import DEVELOPMENT_UNITS, OUT
from fleetreserve.truth_verify import CouldNotRun
from fleetreserve.unit_list import add_units_argument, check_units

UNITS = ("nrel-before-may-2019", "nrel-from-may-2019", "norway-development-pooled")
RHO = 0.0
PRODUCTS = ("hourly", "block")
TREE_SECONDS = OTHER_SECONDS
REQUEST_TOLERANCE = 1e-4  # kWh, plan section 1
ROW_TOLERANCE = 1e-6  # kWh
VALUE_TOLERANCE_PER_HOUR = 2 * REQUEST_TOLERANCE  # kWh (reading R4)
PHYSICAL_KINDS = ("power-high", "power-low", "stored-high", "stored-low")

MATCH = "match"
MISMATCH = "mismatch"
MISSING = "missing"
EMPTY = "empty fleet"

RESULTS_NAME = "t3-replay.json"
PATHS_NAME = "t3-replay-paths.csv"
TEXT_NAME = "t3-replay.txt"
SECONDS_NAME = "t3-replay-seconds.json"


def product_named(name: str) -> Product:
    return {"hourly": products.hourly, "block": products.block}[name]()


@dataclass(frozen=True)
class PathCheck:
    """One audited row replayed: `verdict` is `MATCH` or `MISMATCH`."""

    corners: tuple[int, ...]
    verdict: str
    requests_met: bool
    largest_request_miss: float
    first_missed_hour: int | None
    rows_broken: int
    least_row_slack: float
    replayed_value: float
    value_gap: float
    largest_cut: float
    cuts: int  # (car, hour) pairs cut by more than ROW_TOLERANCE
    cut_car: int | None  # the largest cut's car (position in the fleet) and hour, None if none
    cut_hour: int | None


def _step(node: int, hour: int, corner: float) -> int:
    """The node of hour `hour` below `node` (hour `hour - 1`) when `zeta_hour` is `corner`
    (`tree` module text: depth `k`, index `j`, bit `k - 1` of `j` set for +1)."""
    depth = hour - 1
    index = node - (2**depth - 1) + ((1 if corner > 0 else 0) << depth)
    return 2**hour - 1 + index


def _limits(rows: FleetRows, physical: np.ndarray, drawn: np.ndarray, hour: int):
    """Per car, the interval of powers in `hour` its physical rows closing in that hour allow,
    given what it drew before (`drawn`, hours below `hour`)."""
    low = np.full(rows.n_cars, -np.inf)
    high = np.full(rows.n_cars, np.inf)
    closing = np.flatnonzero(physical & (rows.last_hours == hour))
    for k in closing:
        car = rows.car[k]
        a = rows.coefficients[k]
        residual = rows.bounds[k] - float(a[:hour] @ drawn[car, :hour])
        if a[hour] > 0:
            high[car] = min(high[car], residual / a[hour])
        else:
            low[car] = max(low[car], residual / a[hour])
    return low, high


def replay_path(rows: FleetRows, policy: TreePolicy, v_tree: float, corners) -> PathCheck:
    """The policy replayed on the true cars `rows` along one corner path (module text, 3 and 4)."""
    n = rows.n_service
    zeta = np.asarray(corners, dtype=float)
    physical = np.isin(rows.kinds, PHYSICAL_KINDS)
    drawn = np.zeros((rows.n_cars, n + 1))
    cut = np.zeros((rows.n_cars, n + 1))
    node = 0
    for hour in range(n + 1):
        if hour >= 1:
            node = _step(node, hour, zeta[hour - 1])  # only the corners up to this hour
        command = policy.node_powers[:, node]
        low, high = _limits(rows, physical, drawn, hour)
        drawn[:, hour] = np.minimum(np.maximum(command, low), high)
        cut[:, hour] = np.abs(command - drawn[:, hour])
    total = drawn[:, 1:].sum(axis=0)
    request = policy.m + policy.w * zeta
    miss = np.abs(total - request)
    missed = np.flatnonzero(miss > REQUEST_TOLERANCE)
    slack = rows.slack(drawn)
    replayed = float(2.0 * np.sum(zeta * (total - policy.m)))
    gap = abs(replayed - v_tree)
    met = missed.size == 0
    broken = int((slack < -ROW_TOLERANCE).sum())
    value_ok = gap <= VALUE_TOLERANCE_PER_HOUR * n
    cuts = int((cut > ROW_TOLERANCE).sum())
    hour, car = divmod(int(np.argmax(cut.T)), rows.n_cars)  # the first largest, by hour then car
    return PathCheck(
        corners=tuple(int(z) for z in zeta),
        verdict=MATCH if met and broken == 0 and value_ok and cuts == 0 else MISMATCH,
        requests_met=met,
        largest_request_miss=float(miss.max()),
        first_missed_hour=int(missed[0]) + 1 if missed.size else None,
        rows_broken=broken,
        least_row_slack=float(slack.min()),
        replayed_value=replayed,
        value_gap=gap,
        largest_cut=float(cut.max()),
        cuts=cuts,
        cut_car=int(car) if cuts else None,
        cut_hour=int(hour) if cuts else None,
    )


@dataclass(frozen=True)
class ProductCheck:
    """One product on one day. `status`: "solved", `MISSING` or `EMPTY`; `outcome` the tree's
    (or why the rows could not be written); `paths` empty unless solved."""

    product: str
    status: str
    outcome: str
    v_tree: float | None
    rows: int  # the audited rows of the day
    paths: tuple[PathCheck, ...]
    seconds: float


def check_rows(rows: FleetRows, product: Product, name: str, paths) -> ProductCheck:
    """The tree of `product` on the sets `rows`, replayed along each of `paths`."""
    found = corner_tree(rows, product, time_limit=TREE_SECONDS)
    n_rows = len(paths)
    if found.outcome != "optimal" or found.policy is None or found.value is None:
        return ProductCheck(name, MISSING, found.outcome, None, n_rows, (), float(found.seconds))
    v_tree = float(found.value)
    checks = tuple(replay_path(rows, found.policy, v_tree, path) for path in paths)
    return ProductCheck(name, "solved", "optimal", v_tree, n_rows, checks, float(found.seconds))


@dataclass(frozen=True)
class DayTask:
    unit: str
    site: str
    date: str
    day: Day

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.unit, self.date, self.site)


@dataclass(frozen=True)
class DayCheck:
    unit: str
    site: str
    date: str
    cars: int
    n_service: int
    products: tuple[ProductCheck, ...]


def check_day(task: DayTask) -> DayCheck:
    day = task.day
    n = day.n_service
    paths = audited_paths(day.unit, day.date, n)
    found = []
    if not day.cars:
        found = [ProductCheck(p, EMPTY, EMPTY, None, len(paths), (), 0.0) for p in PRODUCTS]
    else:
        try:
            rows = true_rows(day)
        except ValueError as problem:
            reason = f"true sets not written: {problem}"
            found = [ProductCheck(p, MISSING, reason, None, len(paths), (), 0.0) for p in PRODUCTS]
        else:
            found = [check_rows(rows, product_named(p), p, paths) for p in PRODUCTS]
    return DayCheck(task.unit, task.site, task.date, len(day.cars), n, tuple(found))


def run_days(tasks: list[DayTask], workers: int) -> list[DayCheck]:
    """The checks in the order of `tasks`, whatever the worker count."""
    if workers <= 1:
        return [check_day(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(check_day, tasks, chunksize=1))


# --- counts, lines and files ---------------------------------------------------------------------


def _label(day: DayCheck, product: str) -> str:
    return f"{day.unit} {day.site} {day.date} {product}"


def _corners(corners) -> str:
    return "".join("+" if z > 0 else "-" for z in corners)


def counts(days: list[DayCheck]) -> dict:
    """Days and rows by verdict, over `days` (module text)."""
    found = {
        "days": len(days),
        "day_products": 0,
        "day_products_solved": 0,
        "day_products_missing": 0,
        "day_products_empty": 0,
        "paths": 0,
        "paths_checked": 0,
        "matches": 0,
        "mismatches": 0,
        "missing": 0,
        "paths_of_empty_days": 0,
    }
    for day in days:
        for p in day.products:
            found["day_products"] += 1
            found["paths"] += p.rows
            if p.status == EMPTY:
                found["day_products_empty"] += 1
                found["paths_of_empty_days"] += p.rows
            elif p.status == MISSING:
                found["day_products_missing"] += 1
                found["missing"] += p.rows
            else:
                found["day_products_solved"] += 1
                found["paths_checked"] += len(p.paths)
                found["matches"] += sum(c.verdict == MATCH for c in p.paths)
                found["mismatches"] += sum(c.verdict == MISMATCH for c in p.paths)
    return found


def mismatch_lines(days: list[DayCheck]) -> list[str]:
    lines = []
    for day in days:
        for p in day.products:
            for row, c in enumerate(p.paths, start=1):
                if c.verdict != MISMATCH:
                    continue
                lines.append(
                    f"{_label(day, p.product)} row {row} path {_corners(c.corners)}: requests "
                    f"met {c.requests_met} (largest miss {c.largest_request_miss:.6g} kWh, first "
                    f"in hour {c.first_missed_hour}); true rows broken {c.rows_broken} (least "
                    f"slack {c.least_row_slack:.6g} kWh); replayed value {c.replayed_value:.6g} "
                    f"against V_tree {p.v_tree:.6g} kWh (gap {c.value_gap:.6g}); commands cut "
                    f"{c.cuts}"
                    + (
                        f" (largest: car {c.cut_car} in hour {c.cut_hour}, {c.largest_cut:.6g} kW)"
                        if c.cuts
                        else ""
                    )
                )
    return lines


def missing_lines(days: list[DayCheck]) -> list[str]:
    return [
        f"{_label(day, p.product)}: tree {p.outcome}; its {p.rows} audited rows missing"
        for day in days
        for p in day.products
        if p.status == MISSING
    ]


def _num(value) -> str:
    return "" if value is None else repr(round(float(value), 9))


PATHS_HEADER = [
    "unit",
    "site",
    "date",
    "cars",
    "service_hours",
    "product",
    "row",
    "path",
    "verdict",
    "tree_outcome",
    "v_tree_kwh",
    "replayed_value_kwh",
    "value_gap_kwh",
    "requests_met",
    "largest_request_miss_kwh",
    "first_missed_hour",
    "true_rows_broken",
    "least_row_slack_kwh",
    "largest_cut_kw",
    "cuts_above_tolerance",
    "largest_cut_car",
    "largest_cut_hour",
]


def path_rows(days: list[DayCheck]) -> list[list]:
    rows = []
    for day in days:
        head = [day.unit, day.site, day.date, day.cars, day.n_service]
        for p in day.products:
            if p.status != "solved":
                verdict = MISSING if p.status == MISSING else EMPTY
                for row in range(1, p.rows + 1):
                    blank = [""] * (len(PATHS_HEADER) - 10)
                    rows.append([*head, p.product, row, "", verdict, p.outcome, *blank])
                continue
            for row, c in enumerate(p.paths, start=1):
                rows.append(
                    [
                        *head,
                        p.product,
                        row,
                        _corners(c.corners),
                        c.verdict,
                        p.outcome,
                        _num(p.v_tree),
                        _num(c.replayed_value),
                        _num(c.value_gap),
                        c.requests_met,
                        _num(c.largest_request_miss),
                        "" if c.first_missed_hour is None else c.first_missed_hour,
                        c.rows_broken,
                        _num(c.least_row_slack),
                        _num(c.largest_cut),
                        c.cuts,
                        "" if c.cut_car is None else c.cut_car,
                        "" if c.cut_hour is None else c.cut_hour,
                    ]
                )
    return rows


def verdict(total: dict) -> str:
    if total["mismatches"]:
        return "failed"
    if total["missing"] or not total["paths_checked"]:
        return "could not run"
    return "passed"


def report_lines(document: dict, mismatches: list[str], missing: list[str]) -> list[str]:
    total = document["counts"]["all"]
    lines = [f"MISMATCH: {line}" for line in mismatches]
    lines += [f"MISSING: {line}" for line in missing]
    if lines:
        lines.append("")
    config = document["configuration"]
    lines += [
        f"truth build {document['inputs']['truth_run']}; rho {config['rho']}; true one-way sets; "
        f"products {', '.join(config['products'])}; tree time limit {config['tree_seconds']} s.",
        f"tolerances: request {config['request_tolerance_kwh']} kWh per hour, true row "
        f"{config['row_tolerance_kwh']} kWh, value {config['value_tolerance_kwh_per_hour']} kWh "
        "per service hour.",
        "",
        f"days {total['days']} (day-products {total['day_products']}: solved "
        f"{total['day_products_solved']}, missing {total['day_products_missing']}, empty fleet "
        f"{total['day_products_empty']}); paths {total['paths']} (audited rows): checked "
        f"{total['paths_checked']}, matches {total['matches']}, mismatches "
        f"{total['mismatches']}, missing {total['missing']}, on empty days "
        f"{total['paths_of_empty_days']}",
    ]
    for unit, products_found in document["counts"]["units"].items():
        for product, c in products_found.items():
            lines.append(
                f"  {unit} {product}: days {c['days']}, paths checked {c['paths_checked']}, "
                f"matches {c['matches']}, mismatches {c['mismatches']}, missing {c['missing']}, "
                f"empty days {c['day_products_empty']}"
            )
    lines.append("")
    found = verdict(total)
    if found == "failed":
        lines.append("T3 FAILED: the tree does not match its replay on every audited path")
    elif found == "could not run":
        lines.append("T3 COULD NOT RUN: a value is missing or no path was checked")
    else:
        lines.append("T3 passed: the tree matches its replay on every audited path")
    return lines


def _only(days: list[DayCheck], unit: str, product: str) -> list[DayCheck]:
    picked = []
    for day in days:
        if day.unit == unit:
            kept = tuple(p for p in day.products if p.product == product)
            picked.append(DayCheck(day.unit, day.site, day.date, day.cars, day.n_service, kept))
    return picked


def document_of(inputs: dict, names, days: list[DayCheck]) -> dict:
    units = {
        unit: {product: counts(_only(days, unit, product)) for product in PRODUCTS}
        for unit in names
    }
    return {
        "inputs": inputs,
        "configuration": {
            "rho": RHO,
            "products": list(PRODUCTS),
            "tree_seconds": TREE_SECONDS,
            "request_tolerance_kwh": REQUEST_TOLERANCE,
            "row_tolerance_kwh": ROW_TOLERANCE,
            "value_tolerance_kwh_per_hour": VALUE_TOLERANCE_PER_HOUR,
            "physical_kinds": list(PHYSICAL_KINDS),
            "paths": "audit_paths.audited_paths(unit, date, n): 8 rows, repeats included",
        },
        "counts": {"all": counts(days), "units": units},
        "verdict": verdict(counts(days)),
        "mismatches": mismatch_lines(days),
        "missing": missing_lines(days),
    }


# --- the command ---------------------------------------------------------------------------------


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--truth-run", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    add_units_argument(parser, UNITS)
    return parser.parse_args(argv)


def development_units(units) -> tuple[str, ...]:
    """`units` when each is a development unit (reading R1); raises `CouldNotRun` otherwise,
    before anything is read."""
    names = check_units(units)
    other = [unit for unit in names if unit not in DEVELOPMENT_UNITS]
    if other:
        raise CouldNotRun(
            f"--units names {', '.join(other)}, not a development unit of the truth build "
            f"({', '.join(DEVELOPMENT_UNITS)})"
        )
    return names


def build_tasks(units) -> list[DayTask]:
    tasks = [
        DayTask(name, truth.record.site, truth.record.date, truth.day)
        for name, unit in units.items()
        for truth in unit.days
    ]
    return sorted(tasks, key=lambda task: task.key)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.workers < 1:
        print("COULD NOT RUN: --workers must be at least 1")
        return 2
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir and not Path(run_dir).is_dir():
        print(f"COULD NOT RUN: the run folder {run_dir} does not exist")
        return 2
    try:
        names = development_units(args.units)
        gate_run.argument_paths(args, ("tables", "truth_run"))
        inputs, read = gate_run.read_inputs(args.truth_run, args.tables, names)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        units = {
            name: parse_unit_days(read, name, rho=RHO, no_fit=NO_FIT_CAP, small_fleet=SMALL_KEEP)
            for name in names
        }
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    tasks = build_tasks(units)
    start = time.perf_counter()
    days = run_days(tasks, args.workers)
    wall = time.perf_counter() - start
    document = document_of(inputs, names, days)
    text = "\n".join(report_lines(document, document["mismatches"], document["missing"])) + "\n"
    print(text, end="")
    if run_dir:
        folder = Path(run_dir)
        (folder / RESULTS_NAME).write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        (folder / TEXT_NAME).write_text(text)
        write_csv(folder / PATHS_NAME, PATHS_HEADER, path_rows(days))
        seconds = {
            "workers": args.workers,
            "wall_seconds": round(wall, 3),
            "days": {
                f"{d.unit} {d.site} {d.date}": {p.product: round(p.seconds, 4) for p in d.products}
                for d in days
            },
        }
        (folder / SECONDS_NAME).write_text(json.dumps(seconds, indent=1, sort_keys=True) + "\n")
    found = document["verdict"]
    if found == "failed":
        return 1
    return 2 if found == "could not run" else 0


if __name__ == "__main__":
    sys.exit(main())
