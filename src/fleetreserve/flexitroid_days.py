"""flexitroid against T1(b) on 50 development days (plan section 4, T1; section 13, WP4).

Command line, from the project's top folder:
    python -m fleetreserve.flexitroid_days [--run <wp2-truth run folder>] [--tables <folder>]
        [--workers 4] [--units <unit> ...]

The pass mark reads a recorded run of this command:
    python -m fleetreserve.run_record --name wp4-flexitroid --max-seconds 3600 --
        uv run --frozen python -m fleetreserve.flexitroid_days --workers 4

Days: the block sample of plan 13 WP2 (`block_sample.sample_days`, seed `DEFAULT_SEED`, 50 days),
drawn from the counting days of the development units `nrel-before-may-2019`,
`nrel-from-may-2019` and `norway-development-pooled` (`ceiling_report.UNITS`), rho = 0, true
one-way sets from `days_from_truth` (readings: room capped, small fleets kept), loaded as
`ceiling_report` loads them. It is the project's existing seeded sample of 50 development days,
so the comparison runs on the same days as the block-product sample. The truth tables are used
only if they are those of the recorded truth build at the checked-out commit
(`ceiling_report.check_inputs`). Nothing under the held-out data is read.

Reading U3: `--units` names other units, as for `ceiling_report` (the plan's words for this
command name no units; the brief that added the argument to the report commands names this
command beside them). Training or development units of the truth build only
(`unit_list.KNOWN_UNITS`, from `truth_build.TRAIN_UNITS` and `truth_build.DEVELOPMENT_UNITS`),
each once and never a pooled unit beside a site it pools, or the run could not run before
anything is read; a held-out unit's name is refused (`unit_list`).
The days are then the same draw on the units named, which is not the block sample
of plan 13 WP2 unless the units are `ceiling_report.UNITS` in some order, and the sample's
reading in the results says so (`OTHER_UNITS`). The alternative, not taken, keeps this command on
`ceiling_report.UNITS` alone.

Each day is compared by `flexitroid_match.compare_day` (what is compared, the tolerances and the
readings are in that module's text).

Per-day work runs in `--workers` processes (1 runs in this process). The results files do not
depend on the worker count: days are sorted by unit, date and site, and no timing is in them.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the report is always printed):
`flexitroid-match.json` (every day's comparison, the counts, the tolerances, the flexitroid
version and commit), `flexitroid-match.txt` (the report, any FAILED line first) and
`flexitroid-seconds.json` (seconds per day and the worker count; differs between runs).

Exit status: 1 some day differs; else 2 some day could not run, or the command could not run (a
unit list `unit_list` refuses, a unit the truth build does not hold, a table missing, no verified
truth build at this commit, a table differing from it, fewer than 50 counting days, flexitroid not
installed); else 0, all 50 days match.
"""

import argparse
import importlib.metadata
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from fleetreserve import flexitroid_match as fm
from fleetreserve.block_sample import DEFAULT_SEED, SAMPLE_SIZE, day_text, sample_days
from fleetreserve.ceiling_report import UNITS, check_inputs
from fleetreserve.day import Day
from fleetreserve.days_from_truth import NO_FIT_CAP, SMALL_KEEP, UnitDays, load_unit_days
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun
from fleetreserve.unit_list import add_units_argument, check_units

RHO = 0.0
RESULTS_NAME = "flexitroid-match.json"
TEXT_NAME = "flexitroid-match.txt"
SECONDS_NAME = "flexitroid-seconds.json"
VERDICTS = (fm.MATCH, fm.DIFFERS, fm.COULD_NOT_RUN)
READING = (
    "the block sample of plan 13 WP2 (block_sample.sample_days): proportional to each unit's "
    "counting days, at least one per unit; within a unit, the smallest SHA-256 ranks of (unit, "
    "seed, site and date)"
)
OTHER_UNITS = (
    "the draw of the block sample of plan 13 WP2 on the units named by --units, not that sample: "
)


@dataclass(frozen=True)
class Task:
    unit: str
    site: str
    date: str
    day: Day

    @property
    def label(self) -> str:
        return f"{self.unit} {day_text(self.site, self.date)}"

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.unit, self.date, self.site)


def flexitroid_source() -> dict:
    """The installed flexitroid's version and the commit it was built from."""
    found = importlib.metadata.distribution("flexitroid")
    direct = json.loads(found.read_text("direct_url.json") or "{}")
    return {
        "version": found.version,
        "url": direct.get("url"),
        "commit": direct.get("vcs_info", {}).get("commit_id"),
    }


def build_tasks(units: dict[str, UnitDays], seed: int = DEFAULT_SEED) -> tuple[list[Task], dict]:
    """The sampled days, sorted by unit, date and site, and the sample's description."""
    listed = {
        name: [day_text(t.record.site, t.record.date) for t in unit.days]
        for name, unit in units.items()
    }
    chosen = sample_days(listed, seed, SAMPLE_SIZE)
    tasks = []
    for name, unit in units.items():
        picked = set(chosen[name])
        for truth in unit.days:
            site, day_date = truth.record.site, truth.record.date
            if day_text(site, day_date) in picked:
                tasks.append(Task(name, site, day_date, truth.day))
    tasks.sort(key=lambda task: task.key)
    return tasks, {
        "size": SAMPLE_SIZE,
        "seed": seed,
        "reading": READING if set(units) == set(UNITS) else OTHER_UNITS + READING,
        "counting_days": {name: len(days) for name, days in listed.items()},
        "sampled_days": {name: len(days) for name, days in chosen.items()},
    }


def compare(task: Task) -> fm.DayComparison:
    return fm.compare_day(task.day, task.label)


def run_days(tasks: list[Task], workers: int) -> list[fm.DayComparison]:
    """The comparisons in the order of `tasks`, whatever the worker count."""
    if workers <= 1:
        return [compare(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(compare, tasks, chunksize=1))


def day_entry(task: Task, result: fm.DayComparison) -> dict:
    entry = asdict(result)
    entry.pop("seconds")
    entry.update(unit=task.unit, site=task.site, date=task.date)
    return entry


def counts(entries: list[dict]) -> dict:
    found = {verdict: 0 for verdict in VERDICTS}
    other = {verdict: 0 for verdict in VERDICTS}
    for entry in entries:
        found[entry["verdict"]] += 1
        other[entry["functions_only_verdict"]] += 1
    return {"days": len(entries), "strict": found, "functions_only": other}


def document(inputs: dict, plan: dict, source: dict, entries: list[dict]) -> dict:
    return {
        "inputs": inputs,
        "rho": RHO,
        "sample": plan,
        "flexitroid": source,
        "reading": "strict: flexitroid's p and b on every subset of S, T1(b)'s formula on them, "
        "and the scale from flexitroid's greedy algorithm as the cutting-plane oracle all agree "
        "with exact.py; the functions-only reading (no optimisation) is reported, not decided on",
        "tolerances": {
            "set_relative": fm.SET_TOLERANCE,
            "scale_relative": fm.SCALE_TOLERANCE,
            "gap_relative": fm.GAP_TOLERANCE,
            "range_kw": fm.RANGE_TOLERANCE,
            "max_rounds_per_corner": fm.MAX_ROUNDS,
            "shapes_per_day": fm.SHAPES,
            "zero_width_share": fm.ZERO_WIDTH_SHARE,
        },
        "days": entries,
        "counts": counts(entries),
    }


def _scale_text(shape: dict) -> str:
    def num(value):
        return "-" if value is None else f"{value:.9g}"

    return (
        f"{num(shape['exact_scale'])} vs [{num(shape['lower'])}, {num(shape['upper'])}] "
        f"({shape['corners']} corners, {shape['rounds']} programmes)"
    )


def report_lines(doc: dict) -> list[str]:
    days = doc["days"]
    failed = [
        f"FAILED: {d['unit']} {d['site']} {d['date']}: {problem}"
        for d in days
        for problem in d["problems"]
    ]
    missing = [
        f"COULD NOT RUN: {d['unit']} {d['site']} {d['date']}: {reason}"
        for d in days
        for reason in d["could_not_run"]
    ]
    source = doc["flexitroid"]
    total = doc["counts"]
    lines = [
        *failed,
        *missing,
        f"flexitroid {source['version']} at commit {source['commit']}; truth build "
        f"{doc['inputs']['truth_run']}; rho {doc['rho']}; true one-way sets.",
        f"Reading: {doc['reading']}.",
        f"Sample: {doc['sample']['size']} days, seed {doc['sample']['seed']}; "
        f"{doc['sample']['reading']}.",
        f"Tolerances: {json.dumps(doc['tolerances'], sort_keys=True)}",
        "",
        "unit site date | cars S hours | verdict (functions only) | largest set difference | "
        "per shape: project's scale vs flexitroid's bounds",
    ]
    for d in days:
        difference = "-" if d["max_set_difference"] is None else f"{d['max_set_difference']:.2e}"
        lines.append(
            f"{d['unit']} {d['site']} {d['date']} | {d['cars']} {d['n_service']} {d['n_hours']} | "
            f"{d['verdict']} ({d['functions_only_verdict']}) | {difference} | "
            + "; ".join(_scale_text(s) for s in d["shapes"])
        )
    lines += [
        "",
        f"days {total['days']}: strict {json.dumps(total['strict'], sort_keys=True)}; functions "
        f"only {json.dumps(total['functions_only'], sort_keys=True)}",
    ]
    return lines


def exit_status(doc: dict) -> int:
    strict = doc["counts"]["strict"]
    if strict[fm.DIFFERS]:
        return 1
    if strict[fm.COULD_NOT_RUN] or strict[fm.MATCH] != SAMPLE_SIZE:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    add_units_argument(parser, UNITS)
    args = parser.parse_args(argv)
    try:
        source = flexitroid_source()
    except importlib.metadata.PackageNotFoundError:
        print("COULD NOT RUN: flexitroid is not installed")
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
        print(f"COULD NOT RUN: a day could not be built from the tables: {problem}")
        return 2
    try:
        tasks, plan = build_tasks(units)
    except ValueError as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    start = time.perf_counter()
    results = run_days(tasks, args.workers)
    wall = time.perf_counter() - start
    entries = [day_entry(task, result) for task, result in zip(tasks, results, strict=True)]
    doc = document(inputs, plan, source, entries)
    text = "\n".join(report_lines(doc)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / RESULTS_NAME).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        (folder / TEXT_NAME).write_text(text)
        seconds = {
            "workers": args.workers,
            "wall_seconds": round(wall, 3),
            "days": {
                task.label: round(result.seconds, 3)
                for task, result in zip(tasks, results, strict=True)
            },
        }
        (folder / SECONDS_NAME).write_text(json.dumps(seconds, indent=1, sort_keys=True) + "\n")
    return exit_status(doc)


if __name__ == "__main__":
    sys.exit(main())
