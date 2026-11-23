"""Outcome O3 as a command: LP (24), LP (27) and the general split against the ceiling on a seeded
sample of counting days, on true sets (plan sections 1, 10, 11 (O3) and 13).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.o3_run --units <unit> ... --days N --seed S --base-set day-average
        [--lp27-days 100] [--lp27-cap-seconds 7200]
        [--rho 0] [--tables <folder>] [--truth-run <folder>] [--workers 1]
        [--lp24-seconds 600] [--other-seconds 300] [--box-seconds 30]
        [--lp27-effort 0.1] [--lp27-bound-nodes 1]
        [--held-out-tables <folder>] [--held-out-truth-run <folder>]

`--units`, `--days`, `--seed` and `--base-set` have no default: the run list states them
(`--days` is the plan's 200 per unit for LP (24), the general splits and the ceiling).

Units: named once each, and either every one a training or a development unit
(`truth_build.TRAIN_UNITS`, `truth_build.DEVELOPMENT_UNITS`) or every one a held-out unit
(`sealed_route.HELD_OUT_UNITS`), whose tables are read only through the sealed-data route below
(reading R2). Any other name, and held-out units named with other units, is refused before a
table is opened (exit 2).

Tables of training and development units. `read_units` is the one function that reads them: the
truth build is checked as `gate_run.check_inputs` checks it (the latest recorded `wp2-truth`
build of the tables' setting, or `--truth-run`, verified by `run_verify` at the commit checked
out now, of the setting the tables folder holds, and each unit's two tables with the SHA-256 that
build recorded; otherwise exit 2), then each unit's counting days are built by
`days_from_truth.load_unit_days` with the readings `ceiling_report` uses (room capped when the
protected energy does not fit; small fleets kept) at `--rho` (default 0, the plan's main value).
Held-out units' tables are read by `read_route_units`, below.

Held-out units, through the sealed-data route (`sealed_route.units_route`: steps 1 to 4 of
`sealed_route.matrix_route`, in its order, with texts naming the units named; it starts only at
the tag `frozen-plan` with a clean tree, and its checks stop an honest operator's ordinary
mistakes and are not a defence against deliberate evasion: `sealed_route`, module text, first
paragraph).
When every unit of `--units` is a held-out unit, before any table is read: each path argument is
put through `gate_run.argument_path` (a path written as a URL could not run; a leading `~` is
expanded); every folder named lies outside the held-out places of the top folder
(`sealed_route.check_folders`); the tag gate passes (the held-out loader's checks 2 to 8: at the
tag `frozen-plan`, clean, the running code the tag's); and `--held-out-truth-run` is given
(`--held-out-tables` defaults to the held-out build's folder of the main setting,
`sealed_route.held_out_tables`). Away from the tag a held-out unit could not run (exit 2) before
any table is opened. Then `read_route_units` (`sealed_route.read_two_builds` with no fit unit):
the development build (`--tables`, `--truth-run`) is checked as every run on the route checks it
(verified at the checked-out commit, of the setting its tables folder holds), none of its tables
read; the held-out tables folder must hold that same setting; the units' two tables are read
once from the held-out build's folder, each with the SHA-256 that build's recorded run lists, its
summary verified at the checked-out commit; and the bytes hashed are the bytes parsed
(`days_from_truth.parse_unit_days`, with `read_units`'s readings). Everything after that is as
off the route. With other units the two held-out options could not run, and without them nothing
above is checked: the run, its outputs and its summary are as before.

What the command fits, chooses, samples or calibrates, and from which days (the same on the
route and off it):
  L1. The sample. Per unit, `--days` of its counting days (LP (27) runs on the first
      `--lp27-days` of them in the order of their hash ranks, reading R8): every day of its
      days table with at least 3 cars plugged in at τ (plan section 1; `days_from_truth`),
      whatever the held-out
      build's calibration split made of it (reading R5). The days drawn are those with the
      smallest SHA-256 of (unit, `--seed`, "site date") (`block_sample.draw`). What enters the
      draw is the unit's name, the seed, the count and which days count; no value computed on a
      day (no ceiling, box or energy) enters it. `--seed` and `--days` have no default: the run
      list fixes them before the tag.
  L2. The base set. `--base-set` names one rule for every day of every unit, with no default,
      fixed in the run list before the tag. Only "day-average" is built: each day's base set is
      the average over that day's own true rows (`o3_day` reading D1), a hindsight quantity of
      the day, as the ceiling is; nothing is fitted on other days.
  L3. Time limits and effort. `--lp24-seconds`, `--other-seconds` and `--box-seconds` (defaults
      the plan's section 13 values), `--lp27-effort` and `--lp27-bound-nodes` (defaults set from
      solve times on made-up days, reading R4): one value each for every day and unit, given
      before the run and recorded in `o3.json`. None is chosen by looking at a result: a solve
      that runs out of time gives a missing value, never a retry at another limit (the one
      second solve in the code, `o3_lp27_bound._solve`, is the dual simplex method at the same
      limit after a solver error, a fixed rule).
  L4. Inside a day, LP (27)'s search ranks its own candidate solutions by the boxes they hold on
      that day (`o3_lp27` step 5), with seeds fixed in its code (`o3_lp27.SEED`): that is how the
      day's value is computed, and nothing passes from one day or unit to another.
  L5. Nothing is fitted or calibrated and no setting is chosen from a result: there is no model,
      no forecast, no calibration and no option that selects a setting per unit, so the route
      has none to refuse. Every value is a hindsight quantity of a sampled day's true sets
      (`o3_day`); the label of a run on the route says so (`ROUTE_LABEL`). The only run folders
      read are the truth builds', each verified at the checked-out commit.

Sample. Per unit, `--days` of its counting days, drawn by `block_sample.draw`: the days with the
smallest SHA-256 ranks of (unit, `--seed`, "site date"), so the same seed and tables give the same
days, whatever order they are listed in. The first `--lp27-days` of the draw in the order of
those ranks (`day_split.ranked`) are LP (27)'s days; every day of the draw is started in that
order, within a unit and one unit after another. Each sampled day is computed by
`o3_day.compute_day` with the base set `--base-set` and the time limits (defaults:
`lp.LP24_SECONDS`, `lp.OTHER_SECONDS`, `lp.BOX_SECONDS`), in `--workers` processes (1 runs in
this process). The results do not depend on the worker count: days are sorted by unit, date and
site, and no timing is in them, unless a unit's cap cuts its LP (27) (reading R9).

Readings (each with its alternative in brackets; those of one day are `o3_day`'s D1 to D6, those
of the statistics `o3_summary`'s S1 and S2):

  R1. "100-200 sampled days" (plan O3) is read per unit, `--days` per unit. [In total over the
      units.] A unit with fewer counting days than `--days` gives all of them, and the summary
      says so. [Such a unit refused.] A unit with no counting day could not run (exit 2).
  R2. Held-out units are read only through the sealed-data route above, when every unit named is
      a held-out unit; a mix with other units could not run, and away from the tag a held-out
      unit could not run (exit 2) before any table is opened. [Refuse held-out units outright,
      as before the route; or accept a mix, each unit's tables from its own build.]
  R3. Exit status (below): a missing value (a timeout, the solver outcome "error", a trivial base
      set with a positive ceiling) is a recorded outcome, counted per unit, product, value and
      outcome and named on the summary's third line; it does not change the exit status. [Exit 1
      on any missing value.]
  R4. LP (27)'s cost: `--lp27-effort` (a share in (0, 1] of the search's full settings,
      `o3_lp27.scaled_effort`; default `EFFORT_SHARE`) and `--lp27-bound-nodes` (nodes per
      product of the bound, one relaxation each, `o3_lp27_bound`; default `RUN_BOUND_NODES`),
      both recorded in `o3.json` and the summary. A smaller effort finds a narrower range; the
      bound shows what is left open. The defaults: share 0.1, at which the search's ascent (most
      of its cost on days of 8 service hours) already has its fewest starts (one from each
      source; a smaller share cuts only the candidates, 21 s of the search's 342 s on the
      30-car day below), and one node, the root alone (each further split adds a nearest-map
      programme and ascent programmes over the face, each up to the 300-s limit). Measured on
      made-up days, one process at background priority on a loaded machine, per day: 10 cars,
      3 service hours, 3.5 s; 10 cars, 8 hours, 53 s and 332 s; 30 cars, 8 hours, 415 s, the
      search's ascent 310 s of it and the block product's largest found missing (a programme
      at the 300-s limit). So 100 days in about two hours on 6 workers (about 430 s per day)
      holds only if few days are as large as 30 cars and 8 hours. On each of those three
      8-hour days the root relaxation was at or above the ceiling for both products, and on the
      first 10-car one neither 21 nodes (about 32 s per product) nor added corners lowered it
      (79.38706 against the ceiling 79.38698): there the bound says no more than the ceiling.
      [The full settings and 21 nodes.]
  R5. On a held-out unit the sample is drawn from all its counting days, its build's calibration
      days among them. Plan section 11 (O3): "Headline measurement, true sessions, 100-200
      sampled days: LP (24), LP (27), general split at X, Y, Z% of the ceiling"; it names no
      calibration or test days, and O3 has no scale to calibrate. Section 6 splits a unit's
      counting days into calibration and test days for the calibrated offers ("test days are
      scored only at the chosen β"), and section 5's ledger (T9: "Also V_aff-V_diag and format
      loss") reports the format loss (the ceiling minus LP (24)'s largest box, section 1)
      without naming its days. [The test days
      only, so that O3 covers the days the offers are scored on; that would need the split rule
      of the held-out build, `sealed_route.check_split_rule`.]
  R6. On the route the development build is checked too (`--tables`, `--truth-run`: verified at
      the checked-out commit, the held-out tables of its setting), as every run on the route reads
      two builds of one setting (`sealed_route`, module text), though none of its tables is read.
      `o3.json`'s `inputs` is then the held-out build's check (the tables the values come from,
      as off the route), and the development build's is under `sealed_route`. [The held-out
      build alone; or `inputs` the development build's, as in `learn_run`.]
  R7. On the route the sample size, seed, base set, time limits and LP (27)'s cost settings are
      taken as given, each fixed in the run list before the tag; none is compared with a plan
      number to refuse the run (100 to 200 days; 600, 300 and 30 s). [Refuse on the route a
      `--days` outside 100 to 200 or time limits other than the plan's.] The summary names every
      departure from the plan's numbers (reading R10).
  R8. Plan section 11 (O3): "LP (27) runs on 100 sampled days per unit, LP (24) and the general
      split on 200 per unit; LP (27)'s run time is capped at about two hours per unit, the days
      not reached within a unit's cap recorded as missing" (the owner's decisions of 18 November
      2026). Read as: one draw of `--days` days per unit, LP (24), the general splits and the
      ceilings on all of them, LP (27) on the first `--lp27-days` of the draw by hash rank, so
      the 100 are among the 200 and a larger `--days` adds days after them, never before. With
      fewer sampled days than `--lp27-days`, LP (27) runs on all of them. [Two draws, one of
      each size; or the 100 the first by date.] `--lp27-days` defaults to the plan's 100 and
      `--days` has no default (the run list states the 200). [A default of 200 for `--days`.]
  R9. The cap (`--lp27-cap-seconds`, default 7200, the plan's two hours) is wall-clock time per
      unit from the moment the unit's first day is started, the units one after another, each on
      `--workers` processes. A day of LP (27) whose programmes begin after that time has its
      three LP (27) values missing (outcome `o3_day.CAP_REACHED`), never zero, and the summary
      counts them per unit (`o3.json` key `lp27_cap`). A day begun in time runs to its end, each
      solve under its own limit (plan section 13), so a unit can pass the cap by up to one day.
      The days are started in the order of their hash ranks, so the days not reached are the
      last ranks of LP (27)'s days, not the latest dates. A cap that cuts a run makes its values
      depend on the machine's speed: the results then differ between runs, and the summary says
      which days were not reached. [A cap on the seconds the solves add up to.]
  R10. Plan numbers: 200 days (`PLAN_DAYS`), 100 LP (27) days (`PLAN_LP27_DAYS`), a cap of 7200 s
      (`PLAN_CAP_SECONDS`) and the three time limits of section 13. `o3.json`'s `arguments` key
      `departures_from_plan` lists each of them that a run's arguments differ from, and the
      summary prints the list; an empty list is not printed. A run of fewer days than the plan's
      is a quick look. [Refuse such a run.]

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the summary is always printed):
`o3.json` (every number, the sampled days and, per unit, LP (27)'s days (`sample`), the checks,
per unit `all_present`, the methods' shares over the days where every value is present, and per
unit `lp27_cap`, LP (27)'s days asked for, reached and not reached within the cap),
`o3-summary.txt` (the summary),
`o3-days.csv` (one row per sampled day and product: each value in kWh, its outcome and its share
of the ceiling, LP (24)'s alpha, LP (27)'s trace and the count of what LP (27)'s search passed
over, `o3_lp27`) and `o3-seconds.json` (solve seconds and the worker count; differs between
runs). On the route `o3.json`'s `inputs` is the held-out build's check (reading R6) and the key
`sealed_route` (`gate_summary.ROUTE_KEY`) holds the kind, the label `ROUTE_LABEL`, the tag's
commit, the development build's check (`development_inputs`, no table) and `checks`
(`sealed_route.CHECKS`: what the route's checks stop and what they do not); the summary's first
line is that label with the tag's commit and the two builds, its second the `checks` text, and
the rest is as off the route. Off the route the outputs are as before the route.

Exit status: 0 every sampled day was built, no programme ended infeasible or unbounded, and every
order of `o3_summary` held; 1 any of these failed, or a day could not be built from the tables;
2 could not run (an argument or unit refused, a base set not built, no verified truth build at
this commit, a table missing or differing from that build, a unit with no counting day; for
held-out units, a refusal of the route above or builds of two settings).
"""

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from fleetreserve import gate_run, gate_summary, o3_day, sealed_route
from fleetreserve.block_sample import day_text, draw
from fleetreserve.day_split import ranked
from fleetreserve.days_from_truth import (
    NO_FIT_CAP,
    SMALL_KEEP,
    UnitDays,
    load_unit_days,
    parse_unit_days,
)
from fleetreserve.lp import BOX_SECONDS, LP24_SECONDS, OTHER_SECONDS
from fleetreserve.o3_day import (
    BASE_SETS,
    BOUND_NODES,
    KEYS,
    METHODS,
    PRODUCTS,
    DayResult,
    DayTask,
    TimeLimits,
    compute_day,
)
from fleetreserve.o3_lp27 import Effort, scaled_effort
from fleetreserve.o3_summary import (
    all_present,
    failures,
    label,
    lp27_reach,
    missing_total,
    summary_lines,
    unit_summary,
)
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import DEVELOPMENT_UNITS, OUT, TRAIN_UNITS
from fleetreserve.truth_verify import CouldNotRun

RESULTS_NAME = "o3.json"
SUMMARY_NAME = "o3-summary.txt"
DAYS_NAME = "o3-days.csv"
SECONDS_NAME = "o3-seconds.json"
UNITS = (*TRAIN_UNITS, *DEVELOPMENT_UNITS)
PLAN_DAYS = 200  # plan section 11, O3: LP (24) and the general split, per unit
PLAN_LP27_DAYS = 100  # LP (27), per unit
PLAN_CAP_SECONDS = 7200.0  # LP (27)'s run time per unit, "about two hours"
EFFORT_SHARE = 0.1  # --lp27-effort's default, the least cost (reading R4)
RUN_BOUND_NODES = 1  # --lp27-bound-nodes's default, the root alone (reading R4)
# The label of a run on the sealed route (module text: outputs; the leak list, L5).
ROUTE_LABEL = (
    f"HELD-OUT RESULT, through the sealed-data gate {sealed_route.AT_THE_TAG}, with a clean tree: "
    "outcome O3 on held-out units, every value a hindsight quantity of the sampled days' true "
    "sets; nothing is "
    "fitted, calibrated or chosen from a result"
)


class Refused(Exception):
    """An argument or unit this command does not run on; the command could not run."""


def check_units(units, held_out: bool = False) -> tuple[str, ...]:
    """`units` when each is a training or development unit, or (`held_out`, the route's finding
    that every unit is a held-out unit) a held-out unit, named once; raises `Refused`."""
    units = tuple(units)
    outside = [] if held_out else [u for u in units if u not in UNITS]
    if outside:
        raise Refused(
            f"{outside} are not training or development units ({', '.join(UNITS)}); held-out "
            f"units are read only through the sealed-data route {sealed_route.AT_THE_TAG} "
            "(reading R2)"
        )
    if not units or len(set(units)) != len(units):
        raise Refused(f"the units must be named once each, got {list(units)}")
    return units


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


def lp27_settings(args) -> tuple[Effort, int]:
    """LP (27)'s search settings and the bound's nodes per product; raises `Refused`."""
    share = args.lp27_effort
    if not (math.isfinite(share) and 0 < share <= 1):
        raise Refused(f"--lp27-effort must be in (0, 1], got {share}")
    if args.lp27_bound_nodes < 1:
        raise Refused("--lp27-bound-nodes must be at least 1")
    return scaled_effort(share), args.lp27_bound_nodes


def check_arguments(args) -> TimeLimits:
    """The time limits; raises `Refused` for a base set not built or a bad number."""
    if args.base_set not in BASE_SETS:
        raise Refused(
            f"base set {args.base_set!r} is not built; built: {', '.join(sorted(BASE_SETS))} "
            "(the template base set is reading D1's alternative)"
        )
    if args.days < 1:
        raise Refused("--days must be at least 1")
    if args.lp27_days < 1:
        raise Refused("--lp27-days must be at least 1")
    if not (math.isfinite(args.lp27_cap_seconds) and args.lp27_cap_seconds > 0):
        raise Refused(
            f"--lp27-cap-seconds must be a positive finite number, got {args.lp27_cap_seconds}"
        )
    if args.workers < 1:
        raise Refused("--workers must be at least 1")
    if not (math.isfinite(args.rho) and 0 <= args.rho < 1):
        raise Refused("--rho must be in [0, 1)")
    limits = TimeLimits(args.lp24_seconds, args.other_seconds, args.box_seconds)
    for name, value in vars(limits).items():
        if not (math.isfinite(value) and value > 0):
            raise Refused(f"the {name} time limit must be a positive finite number, got {value}")
    return limits


def read_units(
    truth_run: Path | None, tables: Path, units, rho: float
) -> tuple[dict, dict[str, UnitDays]]:
    """The verified truth build's record and each unit's counting days, from one tables folder
    (module text). Raises `CouldNotRun` (exit 2) or `ValueError` (a day not built, exit 1)."""
    inputs = gate_run.check_inputs(truth_run, tables, units)
    loaded = {
        unit: load_unit_days(tables, unit, rho=rho, no_fit=NO_FIT_CAP, small_fleet=SMALL_KEEP)
        for unit in units
    }
    return inputs, loaded


def read_route_units(
    truth_run: Path | None, tables: Path, route: sealed_route.Route, units, rho: float
) -> tuple[dict, dict, dict[str, UnitDays]]:
    """On the route: the held-out build's record, the development build's record (no table) and
    each unit's counting days, read once from the held-out build's folder (module text). Raises
    `CouldNotRun` (exit 2) or `ValueError` (a day not built, exit 1)."""
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
    """The summary's first two lines on the route (module text: outputs); none off it."""
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


def sample_tasks(
    loaded: dict[str, UnitDays],
    days: int,
    seed: int,
    base_set: str,
    limits: TimeLimits,
    lp27: tuple[Effort | None, int] = (None, BOUND_NODES),
    lp27_days: int | None = None,
) -> tuple[list[DayTask], dict]:
    """The sampled days of every unit as tasks, sorted by unit, date and site, and the record of
    the sample (reading R1). LP (27)'s days are the first `lp27_days` of a unit's draw in the
    order of their hash ranks (None: all of them; reading R8); each task's `rank` is its place in
    that order. Raises `Refused` for a unit with no counting day."""
    tasks, record = [], {}
    for unit, unit_days in loaded.items():
        listed = {day_text(t.record.site, t.record.date): t for t in unit_days.days}
        if not listed:
            raise Refused(f"{unit} has no counting day")
        chosen = draw(unit, seed, list(listed), days)
        order = ranked(unit, seed, chosen)
        asked = len(order) if lp27_days is None else min(lp27_days, len(order))
        with_lp27 = set(order[:asked])
        record[unit] = {
            "counting_days": len(listed),
            "sampled_days": len(chosen),
            "fewer_than_asked": len(listed) < days,
            "days": list(chosen),
            "lp27_days": [text for text in chosen if text in with_lp27],
        }
        for rank, text in enumerate(order):
            truth = listed[text]
            tasks.append(
                DayTask(
                    unit,
                    truth.record.site,
                    truth.record.date,
                    truth.day,
                    base_set,
                    limits,
                    *lp27,
                    lp27=text in with_lp27,
                    rank=rank,
                )
            )
    tasks.sort(key=lambda task: task.key)
    return tasks, record


def run_days(tasks: list[DayTask], workers: int) -> list[DayResult]:
    """The results in the order of `tasks`, whatever the worker count."""
    if workers <= 1:
        return [compute_day(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(compute_day, tasks, chunksize=1))


def run_units(tasks: list[DayTask], workers: int, cap_seconds: float) -> list[DayResult]:
    """The results in the order of `tasks`. The units are run one after another; a unit's days
    are started in the order of their hash ranks (`DayTask.rank`), each task given the deadline
    `cap_seconds` after the unit's start for LP (27) (reading R9)."""
    results: list[DayResult] = []
    for unit in sorted({task.unit for task in tasks}):
        started = [task for task in tasks if task.unit == unit]
        deadline = o3_day.now() + cap_seconds
        order = [replace(task, lp27_deadline=deadline) for task in sorted(started, key=_by_rank)]
        results += sorted(run_days(order, workers), key=lambda result: result.key)
    return results


def _by_rank(task: DayTask) -> int:
    return task.rank


def departures(args) -> list[str]:
    """What this run's numbers do differently from the plan's (reading R10)."""
    named = [
        ("days per unit", args.days, PLAN_DAYS),
        ("LP (27) days per unit", args.lp27_days, PLAN_LP27_DAYS),
        ("LP (27) cap per unit (s)", args.lp27_cap_seconds, PLAN_CAP_SECONDS),
        ("LP (24) time limit (s)", args.lp24_seconds, LP24_SECONDS),
        ("time limit of the other programmes (s)", args.other_seconds, OTHER_SECONDS),
        ("box time limit (s)", args.box_seconds, BOX_SECONDS),
    ]
    return [
        f"{name} {given:g}, the plan's {plan:g}" for name, given, plan in named if given != plan
    ]


def _num(value) -> str:
    return "" if value is None else repr(round(float(value), 9))


DAYS_HEADER = [
    "unit",
    "site",
    "date",
    "cars",
    "service_hours",
    "product",
    *(f"{key}_{part}" for key in KEYS for part in ("kwh", "outcome")),
    *(f"{method}_share" for method in METHODS),
    "lp24_alpha",
    "lp27_trace",
    "lp27_skipped",
]


def days_csv(results: list[DayResult]) -> str:
    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(DAYS_HEADER)
    for r in results:
        for product in PRODUCTS:
            writer.writerow(
                [
                    r.unit,
                    r.site,
                    r.date,
                    r.cars,
                    r.n_service,
                    product,
                    *(
                        cell
                        for key in KEYS
                        for cell in (_num(r.values[product][key]), r.outcomes[product][key])
                    ),
                    *(_num(r.share(product, method)) for method in METHODS),
                    _num(r.alpha_24),
                    _num(r.trace_27),
                    r.lp27_skipped,
                ]
            )
    return handle.getvalue()


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    found.add_argument("--units", nargs="+", required=True)
    found.add_argument("--days", type=int, required=True)
    found.add_argument("--seed", type=int, required=True)
    found.add_argument("--base-set", required=True)
    found.add_argument("--rho", type=float, default=0.0)
    found.add_argument("--tables", type=Path, default=OUT)
    found.add_argument("--truth-run", type=Path, default=None)
    found.add_argument("--lp27-days", type=int, default=PLAN_LP27_DAYS)
    found.add_argument("--lp27-cap-seconds", type=float, default=PLAN_CAP_SECONDS)
    found.add_argument("--workers", type=int, default=1)
    found.add_argument("--lp24-seconds", type=float, default=LP24_SECONDS)
    found.add_argument("--other-seconds", type=float, default=OTHER_SECONDS)
    found.add_argument("--box-seconds", type=float, default=BOX_SECONDS)
    found.add_argument("--lp27-effort", type=float, default=EFFORT_SHARE)
    found.add_argument("--lp27-bound-nodes", type=int, default=RUN_BOUND_NODES)
    found.add_argument("--held-out-tables", type=Path, default=None)
    found.add_argument("--held-out-truth-run", type=Path, default=None)
    return found


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None, root: Path = TOP) -> int:
    args = parser().parse_args(argv)
    try:
        route = take_route(root, args)
        units = check_units(args.units, route.held_out)
        limits = check_arguments(args)
        effort, bound_nodes = lp27_settings(args)
    except (Refused, sealed_route.RouteRefused, CouldNotRun) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    development = None
    try:
        if route.held_out:
            inputs, development, loaded = read_route_units(
                args.truth_run, args.tables, route, units, args.rho
            )
            units = sealed_route.scored_units(units, inputs)  # abstaining units out
        else:
            inputs, loaded = read_units(args.truth_run, args.tables, units, args.rho)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    try:
        tasks, sample = sample_tasks(
            loaded,
            args.days,
            args.seed,
            args.base_set,
            limits,
            (effort, bound_nodes),
            args.lp27_days,
        )
    except Refused as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    start = time.perf_counter()
    results = run_units(tasks, args.workers, args.lp27_cap_seconds)
    wall = time.perf_counter() - start
    problems = failures(results)
    per_unit = {unit: unit_summary([r for r in results if r.unit == unit]) for unit in units}
    table = days_csv(results)
    document = {
        "inputs": inputs,
        "arguments": {
            "units": list(units),
            "days": args.days,
            "seed": args.seed,
            "base_set": args.base_set,
            "rho": args.rho,
            "lp27_days": args.lp27_days,
            "lp27_cap_seconds": args.lp27_cap_seconds,
            "departures_from_plan": departures(args),
            "seconds": vars(limits),
            "lp27": {
                "effort_share": args.lp27_effort,
                "settings": dataclasses.asdict(effort),
                "bound_nodes": bound_nodes,
            },
        },
        "sample": sample,
        "units": per_unit,
        "all_present": {
            unit: all_present([r for r in results if r.unit == unit]) for unit in units
        },
        "lp27_cap": {unit: lp27_reach([r for r in results if r.unit == unit]) for unit in units},
        "missing_values": missing_total(per_unit),
        "failures": problems,
        "days_file": {
            "file": DAYS_NAME,
            "rows": len(results) * len(PRODUCTS),
            "sha256": hashlib.sha256(table.encode()).hexdigest(),
        },
    }
    mark_route(document, route, development)
    text = "\n".join([*route_head(document), *summary_lines(document)]) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / DAYS_NAME).write_text(table)
        seconds = {
            "workers": args.workers,
            "wall_seconds": round(wall, 3),
            "days": {
                label(r): {k: round(v, 4) for k, v in sorted(r.seconds.items())} for r in results
            },
        }
        (folder / SECONDS_NAME).write_text(_json(seconds))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
