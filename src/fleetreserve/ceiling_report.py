"""Block-product sample and hourly ceilings on the development days (input to plan 13 WP2), and
outcome O2 on every counting day of a unit (`--every-day`, plan section 11), on development units
or, through the sealed-data route, on held-out units.

Command line, from the project's top folder:
    python -m fleetreserve.ceiling_report [--run <wp2-truth run folder>] [--tables <folder>]
        [--rho 0] [--seed 2026] [--workers 4] [--units <unit> ...]
        [--every-day] [--held-out-tables <folder>] [--held-out-truth-run <folder>]

`--truth-run` is another name of `--run` (the name every command of the sealed route uses).

Units: `nrel-before-may-2019`, `nrel-from-may-2019` and `norway-development-pooled` (`UNITS`), or
the units `--units` names: training or development units of the truth build only
(`unit_list.KNOWN_UNITS`, from `truth_build.TRAIN_UNITS` and `truth_build.DEVELOPMENT_UNITS`),
each once and never a pooled unit beside a site it pools, or the run could not run before
anything is read (`unit_list`); or every one a held-out unit (`sealed_route.HELD_OUT_UNITS`),
each once, whose tables are read only through the sealed-data route below. Held-out units named
with other units could not run before anything is read (the route's refusal).
Their counting days come from `days_from_truth` (readings: room capped, small
fleets kept), on true one-way sets.

(a) Block sample. A seeded sample of exactly 50 counting days (`block_sample`: proportional to each
unit's counting days, at least one per unit; the alternative, equal per unit, is not offered). On
each: the block product's V_hind, V_caus (corner tree), V_aff and V_diag, the day's size, whether
a car departs inside the window, and whether the day meets T4's hypotheses. The summary the plan's
owner's rule needs is the mean of V_caus / V_hind (`ceiling_summary` says who is left out of it).
The report states the number and the threshold; it decides nothing.

(a') Every day, with `--every-day` in place of (a): outcome O2 (plan section 11: "block product,
M days with |S|<=8"). The M days of a unit are every counting day with at most
`EVERY_DAY_MAX_SERVICE_HOURS` (8) service hours (`every_day_tasks`); on each the four block
values of (a) are computed, and nothing is sampled. Per unit and over all the units named
(`ceiling_summary.o2_summary`): V_caus / V_hind mean and median, the days with an in-window
departure (count, share of M, ratio), K (V_aff = V_caus), the days outside M, and O2's pass
(passed, failed, could not run, or no day to check; `ceiling_summary.o2_pass`), with one verdict
over the units (`ceiling_summary.pass_over`), judged on the units that have an M day meeting T4's
hypotheses, the units with none listed by name beside it (`no_day_to_check`; the owner's decision
of 19 November 2026, `ceiling_summary` reading N), "no day to check" itself when no unit has one,
in the results alone, never in the exit status (reading O2-4).
`--seed` with `--every-day` could not run: no sample is drawn. The refusal is `seed_refusal`, on
the parsed arguments, and `SEED_REFUSED_WITH` declares it for the seeds item of WP13's pass-mark
tool (`tools/freeze_check.py`, item `lists`, reading F12), which takes a seed option as not
required on a run-list step only when this command's parser or `seed_refusal` refuses it there.

(b) Hourly ceilings on every counting day of the units: the hindsight ceiling with hour 0
free and the best causal value, both from `ceiling_values` (which records why the second can be
below the first). On the sampled days (the M days with `--every-day`) the corner tree's hourly
value is added as a cross-check.

The command computes no offer, no forecast and no calibration.

Every solve has its time limit (`ceiling_values`: 30 s for the block ceiling and the
certificates, 300 s for the corner tree; `lp`); a solve that ends in a timeout or any other
outcome but optimal gives a missing value, never zero, kept with its outcome and counted per
value (`missing`); a value is never solved again at another limit.

The truth tables (development and training units) are used only if they are those of the
recorded truth build at the checked-out commit: the latest `runs.noindex/*-wp2-truth` run (or
`--run`) must pass `run_verify` and be at
the commit checked out now (as `truth_verify` decides), and each table's SHA-256 must be the one
that build recorded (the same refusal as `forecast_report`). The build's setting (main, or held:
plan 3) must be the one the tables folder holds, or the run could not run; the setting is written
under `inputs` in the results.

Held-out units, through the sealed-data route (`sealed_route.units_route`, as `o3_run` takes it:
steps 1 to 4 of `sealed_route.matrix_route`, with texts naming the units named; it starts only at
the tag `frozen-plan`, or at a logged parsing fix of it, with a clean tree, and its checks stop
an honest operator's ordinary mistakes and are not a defence against deliberate evasion:
`sealed_route`, module text, first paragraph). When every unit of `--units` is a held-out unit,
before any table is read: each path argument is put through `gate_run.argument_path`; every
folder named lies outside the held-out places of the top folder (`sealed_route.check_folders`);
the tag gate passes (the held-out loader's checks 2 to 8); `--held-out-truth-run` is given
(`--held-out-tables` defaults to the held-out build's folder of the main setting); no unit is
named twice; and `--every-day` is given (`ROUTE_NEEDS_EVERY_DAY`). Then `read_route_units`
(`sealed_route.read_two_builds` with no fit unit): the development build (`--tables`, `--run`)
is checked for its setting, none of its tables read; the held-out tables folder must hold that
same setting; the units' two tables are read once from the held-out build's folder, each with the
SHA-256 that build's recorded run lists, its summary verified at the checked-out commit; and the
bytes hashed are the bytes parsed (`days_from_truth.parse_unit_days`, with the readings above).
Everything after that is as off the route. With other units the two held-out options could not
run, and without them nothing above is checked: the run, its outputs and its summary are as
before the route.

What the command fits, chooses, samples or calibrates, on the route and off it: nothing is
fitted or calibrated, and no setting is chosen from a result. With `--every-day` the days are
chosen by their number of service hours alone, which the truth build's window fixes before any
value is computed; the time limits and `--rho` are given before the run. Every value is a
hindsight or best real-time value of a day's own true sets.

Readings (alternatives in brackets):
  O2-1. On a held-out unit M is drawn from all its counting days, its build's calibration days
      among them: plan section 11 (O2) names M days with |S|<=8 and no calibration or test days,
      and O2 has no scale to calibrate (as `o3_run` reading R5). [The test days only, which needs
      the held-out build's split rule, `sealed_route.check_split_rule`.]
  O2-2. On the route the block sample of (a) could not run: it is WP2's development rule, and O2
      is stated on the M days. [The route taking (a') without the option.]
  O2-3. A counting day with more than 8 service hours is outside M: counted, and the days of
      them that meet T4's hypotheses counted, not checked; O2's pass is judged on M. [Such a
      hypotheses day making the pass "could not run".]
  O2-4. Exit status with `--every-day` (below), by the owner's decisions of 19 and 20 November
      2026 ("O2's held-out command always exits 0 when it ran, its verdict only in its results
      file, so a scientific fail does not stop the sealed run"; an order break or a day not built
      written in the results file, and on an M day in its verdict (reading O2-6), a non-zero
      exit only for the command's own faults that stop it running): O2's pass (passed, failed or
      could not run) and the T4 equality it rests on in the results alone (`every_day`,
      `failures` and the report), as `o3_run` reading R3 does for a missing value. On held-out
      units the exit status is 0 whenever the command ran, a broken order or a day that could not
      be built included (each a line of `failures`, counted in its unit's `faults`, and in its
      pass when on an M day, reading O2-6), and 2 when it could not run: a day not parsed from
      the held-out tables read stops it before any day is judged. On development units a broken
      order or a day not built still gives 1: the owner's words name the held-out command, and
      a development run can be run again. So a
      failed O2, or a fault on a held-out day, does not stop the drive of the run list: on the
      sealed run a step whose exit status is not 0 is logged as failed and the drive goes on
      (the rehearsal stops at it; `run_driver`, module text). [O2's pass in the exit status, 1
      when it failed on a unit and 2 when it could not run on one, which stops the drive at that
      step; exit 0 on development units too; a held-out table not parsed written in a results
      file with exit 0.]
  O2-5. On the route `inputs` is the held-out build's check (the tables the values come from)
      and the development build's is under `sealed_route` (as `o3_run` reading R6).
  O2-6. A unit's pass is changed only by the command's own faults on its M days, on both
      routes (`ceiling_summary.o2_with_faults`; decision 85 (1) of 20 November 2026): a
      block-product order broken on an M day makes it "failed" (the plan states the orders, and
      a violation fails the check); an M day that could not be built makes a pass other than
      "failed" "could not run" (`ceiling_summary.o2_pass`). An hourly-product order broken on
      any day and a day outside M not built are counted in the unit's `faults` (every break and
      every day not built, the block breaks on M days apart) and shown as lines of `failures`
      with "CHECKS FAILED", never in its pass: O2 is the block product on the M days. The verdict
      over the units follows from the units' passes (`pass_over`). [Any broken order makes it
      "failed", else any day not built, in M or not, "could not run", so a unit with a fault is
      never "passed"; a broken order as "could not run".]
  O2-7. A held-out unit the held-out build abstained (`sealed_route` reading S4) has no counting
      day: its pass is "no day to check" with its reason named beside it, as for a unit with no
      day meeting T4's hypotheses (the owner's decision of 19 November 2026, `ceiling_summary`
      reading N), and the verdict over the units is judged on the units that have a day; the
      abstention never makes it "could not run". Every other count over the held-out units is as
      reading S4 says. [An abstained unit's O2 pass "could not run", which made the verdict over
      the units "could not run".]

Per-day work runs in `--workers` processes (1 runs in this process). The results files do not
depend on the worker count: days are sorted by unit, date and site, and no timing is in them.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the report is always printed):
`ceiling-report.json` (every number), `ceiling-report.txt` (the report), `ceiling-days.csv` (the
hourly values of every day), `block-sample.csv` (the sampled days) and `ceiling-seconds.json`
(solve seconds and the worker count; differs between runs). With `--every-day`,
`ceiling-report.json` holds `every_day` in place of `sample` and `block_sample`, `o2-days.csv`
(the M days, in the columns of `block-sample.csv`) replaces `block-sample.csv`, and the column
`in_block_sample` of `ceiling-days.csv` is `in_o2_days`. On the route `ceiling-report.json` holds
`sealed_route` (`gate_summary.ROUTE_KEY`: the kind, the label `ROUTE_LABEL`, the tag's commit, at
a parsing fix the fix commit, the development build's check, `checks`), and the report's first
two lines are that label with the tag's commit and the two builds, then the `checks` text.

Exit status: 0 everything ran and every check held (the T4 equality and the orders of
`ceiling_summary`; with `--every-day` O2's pass, whatever it is, is in the results alone, and
on held-out units 0 whenever the command ran, its broken orders and days not built in the results
and, when on an M day, in its verdict, readings O2-4 and O2-6); 1 a check failed or a day could
not be built (with `--every-day`, on development units alone, a broken order or a day not
built); 2 could not run (a unit list refused, a unit the truth build does not hold, a table
missing, no verified truth build at this commit, a table differing from that build, fewer than
50 counting days without `--every-day`, `--seed` with `--every-day`, a refusal of the sealed
route, a day not parsed from the held-out tables read, or builds of two settings).
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fleetreserve import ceiling_summary, gate_run, gate_summary, sealed_route
from fleetreserve.block_sample import DEFAULT_SEED, SAMPLE_SIZE, day_text, sample_days
from fleetreserve.ceiling_summary import (
    FAILED,
    GROUPS,
    NO_DAY,
    PASSED,
    ZERO,
    block_summary,
    failures,
    faults,
    hourly_summary,
    label,
    no_day_units,
    o2_summary,
    pass_over,
)
from fleetreserve.ceiling_values import BLOCK_KEYS, DayResult, DayTask, compute_day
from fleetreserve.days_from_truth import (
    NO_FIT_CAP,
    SMALL_KEEP,
    UnitDays,
    load_unit_days,
    parse_unit_days,
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

UNITS = ("nrel-before-may-2019", "nrel-from-may-2019", "norway-development-pooled")
RESULTS_NAME = "ceiling-report.json"
TEXT_NAME = "ceiling-report.txt"
DAYS_NAME = "ceiling-days.csv"
SAMPLE_NAME = "block-sample.csv"
SECONDS_NAME = "ceiling-seconds.json"
EVERY_DAY_NAME = "o2-days.csv"
EVERY_DAY_MAX_SERVICE_HOURS = 8  # plan section 11, O2: "M days with |S|<=8"
# The label of a run on the sealed route (module text: outputs).
ROUTE_LABEL = (
    f"HELD-OUT RESULT, through the sealed-data gate {sealed_route.AT_THE_TAG}, with a clean tree: "
    "outcome O2 on held-out units, on every counting day with at most "
    f"{EVERY_DAY_MAX_SERVICE_HOURS} service hours, every value a hindsight or best real-time "
    "value of the day's true sets; nothing is fitted, calibrated, sampled or chosen from a result"
)
ROUTE_NEEDS_EVERY_DAY = (
    "on the sealed route outcome O2 is measured on every counting day with at most "
    f"{EVERY_DAY_MAX_SERVICE_HOURS} service hours (--every-day); the block sample of 50 days is "
    "the development rule of WP2 (reading O2-2); nothing read"
)
SEED_WITH_EVERY_DAY = (
    "--seed draws the block sample, and --every-day draws none: every counting day with at most "
    f"{EVERY_DAY_MAX_SERVICE_HOURS} service hours is computed; nothing read"
)
# Each seed option and the options with which this command refuses it (module text, (a')).
SEED_REFUSED_WITH = {"--seed": ("--every-day",)}


def seed_refusal(args) -> str | None:
    """The refusal of a seed option given with an option that excludes it (`SEED_REFUSED_WITH`),
    on the parsed `args`; None when there is none."""
    if args.every_day and args.seed is not None:
        return SEED_WITH_EVERY_DAY
    return None


def check_inputs(run: Path | None, tables: Path, units: Sequence[str] = UNITS) -> dict:
    """The verified truth build's name and the hashes of the tables of `units`; raises
    `CouldNotRun` otherwise."""
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


def take_route(root: Path, args) -> sealed_route.Route:
    """The sealed route when every unit named is a held-out unit, its path arguments first put
    through `gate_run.argument_path` (module text); else the route off, after refusing the
    held-out options if given. Raises `sealed_route.RouteRefused`, or `CouldNotRun` for a path
    written as a URL."""
    if all(unit in sealed_route.HELD_OUT_UNITS for unit in args.units):
        gate_run.argument_paths(args)
    return sealed_route.units_route(
        root, args.units, args.tables, args.truth_run, args.held_out_tables, args.held_out_truth_run
    )


def route_units(args, held_out: bool) -> tuple[str, ...]:
    """The units to read, after the route's decision (`held_out`: every unit is a held-out
    unit), and the options that go with them; raises `CouldNotRun` (module text)."""
    if held_out:
        names = tuple(args.units)  # every one a held-out unit (the route's decision)
        if len(set(names)) != len(names):
            raise CouldNotRun(f"the units are refused: a unit is named twice: {list(names)}")
        if not args.every_day:
            raise CouldNotRun(ROUTE_NEEDS_EVERY_DAY)
    else:
        names = check_units(args.units)
    refused = seed_refusal(args)
    if refused is not None:
        raise CouldNotRun(refused)
    return names


def read_route_units(
    truth_run, tables: Path, route: sealed_route.Route, units, rho: float
) -> tuple[dict, dict, dict[str, UnitDays]]:
    """On the route: the held-out build's record, the development build's record (no table) and
    each unit's counting days, read once from the held-out build's folder (module text). Raises
    `CouldNotRun` or `ValueError` (a day not built: could not run, exit 2, reading O2-4)."""
    development, held_out, read = sealed_route.read_two_builds(
        truth_run, tables, (), route.truth_run, route.tables, units
    )
    scored = sealed_route.scored_units(units, held_out)  # abstaining units out
    if not scored:
        raise CouldNotRun(sealed_route.no_unit_left(held_out))
    loaded = {
        unit: parse_unit_days(read, unit, rho=rho, no_fit=NO_FIT_CAP, small_fleet=SMALL_KEEP)
        for unit in scored
    }
    return held_out, development, loaded


def mark_route(document: dict, route: sealed_route.Route, development: dict | None) -> None:
    """Records the route in `document` (module text: outputs); nothing when the route is off."""
    if not route.held_out:
        return
    document[gate_summary.ROUTE_KEY] = {
        "kind": sealed_route.HELD_OUT,
        "label": ROUTE_LABEL,
        "tag_commit": route.tag_commit,
        **sealed_route.fix_record(route),
        "development_inputs": development,
        "checks": sealed_route.CHECKS,
    }


def route_head(document: dict) -> list[str]:
    """The report's first two lines on the route (module text: outputs); none off it."""
    found = document.get(gate_summary.ROUTE_KEY)
    if not found:
        return []
    return [
        f"{found['label']} (tag commit {found['tag_commit']}{sealed_route.fix_text(found)}). The "
        f"units' tables from the held-out build {document['inputs']['truth_run']}; the "
        "development build "
        f"{found['development_inputs']['truth_run']} checked for its setting only, none of its "
        "tables read.",
        found["checks"],
        *sealed_route.abstention_lines(document["inputs"]),
    ]


def every_day_tasks(units: dict[str, UnitDays]) -> tuple[list[DayTask], dict]:
    """Every counting day as a task, the M days (at most `EVERY_DAY_MAX_SERVICE_HOURS` service
    hours) marked `sampled`, sorted by unit, date and site, and the record of the days (module
    text, (a')). Which days are M depends on each day's number of service hours alone."""
    tasks = []
    for name, unit in units.items():
        for truth in unit.days:
            in_m = truth.day.n_service <= EVERY_DAY_MAX_SERVICE_HOURS
            tasks.append(DayTask(name, truth.record.site, truth.record.date, truth.day, in_m))
    tasks.sort(key=lambda task: task.key)
    return tasks, {
        "max_service_hours": EVERY_DAY_MAX_SERVICE_HOURS,
        "reading": f"every counting day with at most {EVERY_DAY_MAX_SERVICE_HOURS} service hours, "
        "nothing sampled; a day with more is counted outside M (readings O2-1 and O2-3)",
        "counting_days": {name: len(unit.days) for name, unit in units.items()},
        "days_in_m": {name: sum(t.sampled for t in tasks if t.unit == name) for name in units},
    }


def abstained_units_o2(abstained: dict) -> dict:
    """The entry of each unit the held-out build abstained in O2's verdict over the units: it has
    no day, so its pass is "no day to check", as for a unit with no day meeting T4's hypotheses
    (the owner's decision of 19 November 2026, `ceiling_summary` reading N), and the verdict is
    judged on the units that have a day, never "could not run" because of it. The entry says why
    it has none (reading S4 of `sealed_route`; there the unit counts as not passing in the other
    counts over the units, not in O2's). It has no day, so no fault."""
    return {
        name: {
            "o2_pass": NO_DAY,
            "reason": sealed_route.abstained_text(reason),
            "days_in_m_not_built": 0,
            "faults": dict.fromkeys(
                ("order_breaks", "days_not_built", "block_order_breaks_on_m_days"), 0
            ),
        }
        for name, reason in abstained.items()
    }


def every_day_summary(
    results: list[DayResult], names, record: dict, abstained: dict | None = None
) -> dict:
    """Outcome O2 per unit, each unit's pass with its faults counted (reading O2-6), over the
    units named, and one verdict over the units."""
    units = {
        name: ceiling_summary.o2_with_faults([r for r in results if r.unit == name])
        for name in names
    }
    units.update(abstained_units_o2(abstained or {}))  # no day to check, with its reason
    pooled = o2_summary(results)
    del pooled["o2_pass"]  # the pass is per unit; `pass_over` gives one verdict over them
    passes = {name: found["o2_pass"] for name, found in units.items()}
    return {
        **record,
        "units": units,
        "all_units": pooled,
        "pass_over_units": pass_over(passes.values()),
        "no_day_to_check": no_day_units(passes),
    }


def build_tasks(units: dict[str, UnitDays], seed: int) -> tuple[list[DayTask], dict]:
    """Every counting day as a task, sampled or not, sorted by unit, date and site."""
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
            sampled = day_text(site, day_date) in picked
            tasks.append(DayTask(name, site, day_date, truth.day, sampled))
    tasks.sort(key=lambda task: task.key)
    return tasks, {
        "size": SAMPLE_SIZE,
        "seed": seed,
        "reading": "proportional to each unit's counting days, at least one per unit "
        "(alternative: equal per unit); within a unit, the smallest SHA-256 ranks of "
        "(unit, seed, site and date)",
        "counting_days": {name: len(days) for name, days in listed.items()},
        "sampled_days": {name: len(days) for name, days in chosen.items()},
    }


def run_days(tasks: list[DayTask], workers: int) -> list[DayResult]:
    """The results in the order of `tasks`, whatever the worker count."""
    if workers <= 1:
        return [compute_day(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(compute_day, tasks, chunksize=1))


def _num(value) -> str:
    return "" if value is None else repr(round(float(value), 9))


def _ratio(result: DayResult):
    hind, caus = result.values["block_hind"], result.values["block_caus"]
    return None if None in (hind, caus) or hind <= ZERO else caus / hind


def write_csv(path: Path, header: list[str], rows: list[list]):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def days_rows(results: list[DayResult]) -> list[list]:
    return [
        [
            r.unit,
            r.site,
            r.date,
            r.cars,
            r.n_service,
            r.in_window,
            r.sampled,
            _num(r.values["hourly_hind"]),
            _num(r.values["hourly_causal"]),
            r.outcomes["hourly_causal"],
        ]
        for r in results
    ]


DAYS_HEADER = [
    "unit",
    "site",
    "date",
    "cars",
    "service_hours",
    "in_window_departure",
    "in_block_sample",
    "hourly_hindsight_kwh",
    "hourly_best_causal_kwh",
    "hourly_best_causal_outcome",
]
EVERY_DAY_DAYS_HEADER = ["in_o2_days" if n == "in_block_sample" else n for n in DAYS_HEADER]


def sample_rows(sampled: list[DayResult]) -> list[list]:
    return [
        [
            r.unit,
            r.site,
            r.date,
            r.cars,
            r.n_service,
            r.n_recovery,
            r.in_window,
            r.t4_hypotheses,
            *(_num(r.values[k]) for k in BLOCK_KEYS),
            _num(_ratio(r)),
            _num(r.values["hourly_causal"]),
            _num(r.values["hourly_tree"]),
            *(r.outcomes[k] for k in BLOCK_KEYS),
        ]
        for r in sampled
    ]


SAMPLE_HEADER = [
    "unit",
    "site",
    "date",
    "cars",
    "service_hours",
    "recovery_hours",
    "in_window_departure",
    "meets_t4_hypotheses",
    "block_v_hind_kwh",
    "block_v_caus_kwh",
    "block_v_aff_kwh",
    "block_v_diag_kwh",
    "ratio_caus_over_hind",
    "hourly_best_causal_kwh",
    "hourly_tree_kwh",
    *(f"{k}_outcome" for k in BLOCK_KEYS),
]


def _fmt(value, spec: str = ".4f") -> str:
    return "-" if value is None else format(value, spec)


def _ratio_line(name: str, ratio: dict) -> str:
    return (
        f"  {name}: {ratio['days']} days; {ratio['days_in_ratio']} form a ratio "
        f"({ratio['days_with_zero_hindsight']} with V_hind = 0 and "
        f"{ratio['days_with_a_missing_value']} with a missing value left out); "
        f"mean {_fmt(ratio['mean'], '.6f')} median {_fmt(ratio['median'], '.6f')} "
        f"min {_fmt(ratio['min'], '.6f')}"
    )


def report_lines(document: dict, sampled: list[DayResult], problems: list[str]) -> list[str]:
    block = document["block_sample"]
    plan = document["sample"]
    lines = [
        f"truth build {document['inputs']['truth_run']}; rho {document['rho']}; seed "
        f"{plan['seed']}. True one-way sets. No offer, forecast or calibration is computed.",
        "",
        f"(a) BLOCK SAMPLE: {block['days']} counting days. Spread (reading): {plan['reading']}.",
    ]
    for unit in plan["counting_days"]:
        lines.append(
            f"  {unit}: {plan['counting_days'][unit]} counting days, "
            f"{plan['sampled_days'][unit]} sampled"
        )
    lines += [
        "",
        "  Sampled days (kWh; Y/N: departure inside the window, meets T4's hypotheses):",
        "  unit site date | cars S | win hyp | V_hind V_caus V_aff V_diag | ratio",
    ]
    for r in sampled:
        v = r.values
        lines.append(
            f"  {r.unit} {r.site} {r.date} | {r.cars} {r.n_service} | "
            f"{'Y' if r.in_window else 'N'} {'Y' if r.t4_hypotheses else 'N'} | "
            f"{_fmt(v['block_hind'])} {_fmt(v['block_caus'])} {_fmt(v['block_aff'])} "
            f"{_fmt(v['block_diag'])} | {_fmt(_ratio(r))}"
        )
    lines += [
        "",
        "  Ratio V_caus / V_hind:",
        _ratio_line("all sampled days", block["ratio_caus_over_hind"]),
    ]
    for name, ratio in block["ratio_by_group"].items():
        lines.append(_ratio_line(name, ratio))
    caus, aff = block["caus_below_hind"], block["aff_below_caus"]
    lines += [
        f"  days with V_caus < V_hind: {caus['days']} of {caus['of_days']} (tolerance 1e-6 "
        "relative)",
        f"  days with V_aff < V_caus: {aff['days']} of {aff['of_days']}",
        f"  days with V_diag < V_aff: {block['diag_below_aff']['days']} of "
        f"{block['diag_below_aff']['of_days']}",
    ]
    if aff["days"]:
        lines.append(
            "  FINDING: V_aff < V_caus on the days below (the plan's open conjecture T4 says "
            "they are equal):"
        )
        lines += [
            f"    {d['day']}: V_aff {d['block_aff']} V_caus {d['block_caus']}" for d in aff["list"]
        ]
    check = block["t4_check"]
    lines.append(
        f"  T4 check: {check['days_meeting_hypotheses']} sampled days meet the hypotheses; "
        f"V_hind = V_caus checked on {check['days_checked']}, not checkable "
        f"{check['days_not_checked_missing_value']}, violations {len(check['violations'])}"
    )
    missing = ", ".join(
        f"{key} {m['timeout']} timed out + {m['other']} other of {m['asked']}"
        for key, m in block["missing"].items()
    )
    lines += [
        f"  missing values (a missing value is never read as zero): {missing}",
        f"  sampled days with an empty fleet (all values 0): {block['empty_fleet_days']}",
        f"  {block['statement']}",
        "",
        *hourly_lines(document),
        *(f"FAILED: {line}" for line in problems),
        "CHECKS FAILED" if problems else "all checks held",
    ]
    return lines


def _missing_line(missing: dict) -> str:
    return ", ".join(
        f"{key} {m['timeout']} timed out + {m['other']} other of {m['asked']}"
        for key, m in missing.items()
    )


def _o2_lines(name: str, found: dict) -> list[str]:
    """One unit's, or all units', figures of outcome O2 (module text, (a'))."""
    inside, others = found["in_window_departure"], found["no_in_window_departure"]
    equal, check = found["aff_equals_caus"], found["t4_check"]
    lines = [
        f"  {name}: {found['counting_days']} counting days, {found['days_in_m']} in M, "
        f"{found['days_outside_m']} with more service hours ({found['hypotheses_days_outside_m']} "
        "of them meeting T4's hypotheses, not checked); "
        f"{found['empty_fleet_days']} M days with an empty fleet (all values 0)",
        _ratio_line("  V_caus / V_hind over the M days", found["ratio_caus_over_hind"]),
        f"    M days with an in-window departure: {inside['days']} (share of M "
        f"{_fmt(inside['share_of_m'], '.6f')})",
        _ratio_line("  V_caus / V_hind on them", inside["ratio_caus_over_hind"]),
        _ratio_line(
            f"  V_caus / V_hind on the {others['days']} M days without one",
            others["ratio_caus_over_hind"],
        ),
        f"    V_aff = V_caus (K): {equal['days']} of {equal['of_days']} M days with both values "
        f"({equal['days_with_a_missing_value']} with a missing value)",
        "    missing values (a missing value is never read as zero): "
        f"{_missing_line(found['missing'])}",
    ]
    if "o2_pass" in found:
        lines.append(
            "    O2's pass (V_hind = V_caus on every M day meeting T4's hypotheses): "
            f"{found['o2_pass'].upper()}: {check['days_meeting_hypotheses']} M days meet them, "
            f"checked {check['days_checked']}, not checkable "
            f"{check['days_not_checked_missing_value']}, violations {len(check['violations'])}"
        )
        lines += [
            f"      V_hind {v['block_hind']} differs from V_caus {v['block_caus']} on {v['day']}"
            for v in check["violations"]
        ]
        own = found.get("faults", {})
        if any(own.values()):
            lines.append(
                f"      with the command's own faults on the unit's days (reading O2-6): "
                f"{own['order_breaks']} broken orders, {own['days_not_built']} days not built; "
                f"in O2's pass: {own['block_order_breaks_on_m_days']} block orders broken and "
                f"{found['days_in_m_not_built']} days not built on M days"
            )
    return lines


def every_day_lines(document: dict, problems: list[str], faults: list[str]) -> list[str]:
    """The report with `--every-day` (module text, (a')); `faults`: the lines of `problems` that
    are the command's own (reading O2-4)."""
    every = document["every_day"]
    lines = [
        f"truth build {document['inputs']['truth_run']}; rho {document['rho']}. True one-way "
        "sets. No offer, forecast or calibration is computed.",
        "",
        f"(a) O2 ON EVERY COUNTING DAY WITH AT MOST {every['max_service_hours']} SERVICE HOURS "
        f"(the M days; plan section 11, O2), block product. Days (reading): {every['reading']}.",
    ]
    for name, found in every["units"].items():
        if "reason" in found:  # a unit the held-out build abstained (no day to report)
            lines.append(f"UNIT {name}: {found['reason']}; {found['o2_pass']}")
            continue
        lines += _o2_lines(f"UNIT {name}", found)
    lines += _o2_lines("ALL UNITS NAMED", every["all_units"])
    verdicts = ", ".join(f"{name} {found['o2_pass']}" for name, found in every["units"].items())
    no_day = every["no_day_to_check"]
    beside = f" (no day to check on {', '.join(no_day)})" if no_day else ""
    lines += [
        f"  O2's pass per unit: {verdicts}; over the units: {every['pass_over_units']}{beside}",
        "",
        *hourly_lines(document),
        *(f"FAILED: {line}" for line in problems),
    ]
    not_run = [
        name for name, found in every["units"].items() if found["o2_pass"] not in (PASSED, NO_DAY)
    ]
    failed = [name for name, found in every["units"].items() if found["o2_pass"] == FAILED]
    if faults:
        lines.append("CHECKS FAILED")
    elif failed:
        lines.append(
            f"O2's pass failed on {', '.join(failed)} (reading O2-4: in the results alone)"
        )
    elif not_run:
        lines.append(f"O2's pass could not run on {', '.join(not_run)}")
    elif every["pass_over_units"] != PASSED:  # no day to check on every unit, or no unit
        lines.append(f"O2's pass over the units: {every['pass_over_units']}{beside}")
    elif no_day:
        lines.append(f"all checks held{beside}")
    else:
        lines.append("all checks held")
    return lines


def hourly_lines(document: dict) -> list[str]:
    """Part (b) of the report and the blank line after it."""
    lines = ["(b) HOURLY CEILINGS on every counting day (hindsight with hour 0 free; best causal):"]
    for unit, groups in document["hourly"].items():
        lines.append(f"  UNIT {unit}")
        for name in GROUPS:
            g = groups[name]
            lines += [
                f"    {name}: {g['days']} days ({g['days_with_a_missing_causal_value']} missing)",
                f"      sum kWh: hindsight {_fmt(g['sum_hindsight_kwh'])} causal "
                f"{_fmt(g['sum_best_causal_kwh'])}; mean per day: hindsight "
                f"{_fmt(g['mean_hindsight_kwh_per_day'])} causal "
                f"{_fmt(g['mean_best_causal_kwh_per_day'])}; ratio of sums "
                f"{_fmt(g['ratio_of_sums'], '.6f')}",
                f"      per-day ratio over {g['days_with_positive_hindsight']} days with a "
                f"positive ceiling: mean {_fmt(g['per_day_ratio_mean'], '.6f')} median "
                f"{_fmt(g['per_day_ratio_median'], '.6f')}; share of days differing "
                f"{_fmt(g['share_days_differing'], '.6f')}; share zero causal with positive "
                f"ceiling {_fmt(g['share_days_zero_causal_positive_hindsight'], '.6f')}",
            ]
    lines.append("")
    return lines


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    found.add_argument("--run", "--truth-run", dest="truth_run", type=Path, default=None)
    found.add_argument("--tables", type=Path, default=OUT)
    found.add_argument("--rho", type=float, default=0.0)
    found.add_argument("--seed", type=int, default=None)  # DEFAULT_SEED for the block sample
    found.add_argument("--workers", type=int, default=4)
    found.add_argument("--every-day", action="store_true")
    found.add_argument("--held-out-tables", type=Path, default=None)
    found.add_argument("--held-out-truth-run", type=Path, default=None)
    add_units_argument(found, UNITS)
    return found


def exit_status(problems: list[str], every: dict | None, own: list[str], held_out: bool) -> int:
    """0 or 1 once the days were computed (module text: exit status): without `--every-day` 1 on
    any of `problems`; with it (`every`) 0 on held-out units (`held_out`), and on development
    units 1 on any of `own` (days not built, broken orders) alone, O2's pass never entering it
    (reading O2-4)."""
    if every is None:
        return 1 if problems else 0
    return 1 if own and not held_out else 0


def main(argv: list[str] | None = None, root: Path = TOP) -> int:
    args = parser().parse_args(argv)
    development = route = None
    try:
        route = take_route(root, args)
        names = route_units(args, route.held_out)
        if route.held_out:
            inputs, development, units = read_route_units(
                args.truth_run, args.tables, route, names, args.rho
            )
            names = sealed_route.scored_units(names, inputs)  # abstaining units out
        else:
            inputs = check_inputs(args.truth_run, args.tables, names)
    except (CouldNotRun, sealed_route.RouteRefused) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except ValueError as problem:  # a held-out day not built from the bytes read
        if route is not None and route.held_out:  # before any day is judged (reading O2-4)
            print(f"COULD NOT RUN: a day could not be built from the tables: {problem}")
            return 2
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    try:
        if not route.held_out:
            units = {
                unit: load_unit_days(
                    args.tables, unit, rho=args.rho, no_fit=NO_FIT_CAP, small_fleet=SMALL_KEEP
                )
                for unit in names
            }
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    try:
        if args.every_day:
            tasks, plan = every_day_tasks(units)
        else:
            tasks, plan = build_tasks(units, DEFAULT_SEED if args.seed is None else args.seed)
    except ValueError as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    start = time.perf_counter()
    results = run_days(tasks, args.workers)
    wall = time.perf_counter() - start
    sampled = [r for r in results if r.sampled]
    problems = failures(sampled, results)
    hourly = {unit: hourly_summary([r for r in results if r.unit == unit]) for unit in names}
    every = (
        every_day_summary(results, names, plan, sealed_route.abstained_of(inputs))
        if args.every_day
        else None
    )
    if every is None:
        document = {
            "inputs": inputs,
            "rho": args.rho,
            "sample": plan,
            "block_sample": block_summary(sampled),
            "hourly": hourly,
            "failures": problems,
        }
        lines = report_lines(document, sampled, problems)
    else:
        document = {
            "inputs": inputs,
            "rho": args.rho,
            "every_day": every,
            "hourly": hourly,
            "failures": problems,
        }
        lines = every_day_lines(document, problems, faults(results))
    mark_route(document, route, development)
    text = "\n".join([*route_head(document), *lines]) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / RESULTS_NAME).write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        (folder / TEXT_NAME).write_text(text)
        if every is None:
            write_csv(folder / DAYS_NAME, DAYS_HEADER, days_rows(results))
            write_csv(folder / SAMPLE_NAME, SAMPLE_HEADER, sample_rows(sampled))
        else:
            write_csv(folder / DAYS_NAME, EVERY_DAY_DAYS_HEADER, days_rows(results))
            write_csv(folder / EVERY_DAY_NAME, SAMPLE_HEADER, sample_rows(sampled))
        seconds = {
            "workers": args.workers,
            "wall_seconds": round(wall, 3),
            "days": {
                label(r): {k: round(v, 4) for k, v in sorted(r.seconds.items())} for r in results
            },
        }
        (folder / SECONDS_NAME).write_text(json.dumps(seconds, indent=1, sort_keys=True) + "\n")
    return exit_status(problems, every, faults(results), route.held_out)


if __name__ == "__main__":
    sys.exit(main())
