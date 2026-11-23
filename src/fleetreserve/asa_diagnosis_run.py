"""Why one tracking controller of the comparison misses requests another meets, per day, hour and
cause (a measurement on development units; plan section 10).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.asa_diagnosis_run --model <name> --dial <value> [--matrix <file>]
        [--method <name>] [--reference <name>] [--scale <beta>] [--days N] [--splits N]
        [--workers N] [--tables <folder>] [--truth-run <folder>]

A diagnosis: it replays two controller methods of `compare_methods` (`--method`, default
`asa-es`; `--reference`, default `llf`) as `compare_run` measures them and explains the hours the
method misses and the reference meets. It changes no method, tunes nothing and writes no choice.

Inputs as `compare_run` takes them off the sealed route, by the same calls, except that here the
path arguments do not pass `gate_run.argument_paths`, the matrix file does not pass
`gate_run.check_files` and the matrix is read by `gate_config.load` (`compare_run` makes those
two checks and reads it once by `gate_config.read`, its reading H4); this command has no route for
held-out units (`sealed_route`, module text, first paragraph). The inputs: the model
(`learn_run.load_model`, `learn_run.check_model`, `dial_choice.make_factory` at the kappa of the
dial's configuration), the
truth tables (`gate_run.check_inputs`, `days_from_truth.load_unit_days`), the matrix's gate units,
`--dial` and its two variants ("forecast": the dial's configuration; "oracle-departure": its
oracle partner, `compare_run.dial_configurations`), the days prepared by `gate_day.prepare_day`
through `compare_run.Pool` and `compare_run`'s own preparation task, and the audited paths of each
prepared day (`PreparedDay.paths`). The method and the reference must be two different controller
methods (`compare_methods.select`). The diagnosis is of the whole fleet: it has no no-pause option
(plan section 9; `compare_run --no-pause-share`), and its preparation task is given no drawn car.

Readings of the brief's wording (each the closest reading; the alternative in brackets):

  D1. Units. Every gate unit of the matrix that `gate_run.plan_unit` can split, as in
      `compare_run`; any other is reported with its status and not diagnosed. [Alternative:
      diagnose an unsplit unit at a given scale.]
  D2. Days. The unit's counting days in the seeded order of the splits: by the rank that
      `day_split.split_days` gives each day key ("<date> <site>") for the seed `SEED` (1, the first
      split seed of `dial_choice.split_indices`), smallest first, so the first `k` of them are
      the calibration days of that split, `k` its number of calibration days
      (`day_split.calibration_count`: the matrix's `n_calibration`, or half the counting days
      rounded down for a unit with fewer counting days than a non-null
      `calibration_half_below`). The rank is `day_split._rank`, the one `day_split.split_days`
      sorts by, read here by its private name. Per variant, the first `--days` of them (default
      50) that offer under that variant are diagnosed, in that order. [Alternative: the corner
      search's own order, not in this code base.]
  D3. Scale. `--scale`: one scale of `dial_splits.GRID`, used for every unit and variant (any
      other value is refused). Without it, per unit and variant, from the reference's tier A
      calibration on the splits of seeds 1 to `--splits` (default 200; `dial_choice.split_indices`,
      `dial_splits.score_split` of the reference's losses at every grid scale on every offering
      day, as `compare_run.measure_controllers` measures them): among the splits on which the
      reference's calibrated tier A offer is above 0, the calibrated scale that the most of them
      have; a tie goes to the larger scale. With no such split the variant is not diagnosed
      (status "no reference offer"). The losses carry no ceiling (`best_causal` unset), which the
      scale does not read. [Alternative: the largest scale at which the reference offers on more
      than half of the splits.]
  D4. Replay and diagnosis of each diagnosed day at that scale on each of its audited paths:
      `asa_diagnosis_day.diagnose_day` (its module text defines the deliverable range, the groups
      of hours and the causes), with the goal tests of `asa_diagnosis_lp` under the limit of plan
      section 13 for other programmes (`lp.OTHER_SECONDS`, 300 s) on every test.

Timeouts are recorded outcomes: a day whose certificate times out abstains (`gate_day`) and is not
diagnosed; the certificates that timed out are counted per variant; a goal test that times out
makes its hour's goal "not determined" and is counted in `goal_tests`; the controllers' own solves
are counted by outcome in `solver_counts`.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary also on
standard output; the results, the tables and the summary do not depend on `--workers` (the
timing file, kept apart, does):
- `asa-diagnosis.json`: inputs, readings, the scope (`asa_diagnosis_summary.SCOPE`: what the
  figures can and cannot support), and per unit and variant the scale, the days and the figures
  of `asa_diagnosis_summary`; every key is defined under `definitions` (a key of a map listed in
  `definitions.name_maps` is a name: a unit, a variant, a goal, an outcome, a file, a kind of
  box);
- `asa-diagnosis-hours.csv`: one row per hour missed by the method only (`HOURS_COLUMNS`, each
  defined under `definitions.hours_columns`), ordered by unit, variant, day (in the order
  diagnosed, D2), path and hour;
- `asa-diagnosis-state-cars.csv`: per row of cause "state", one row per car of the day
  (`CARS_COLUMNS`, defined under `definitions.state_cars_columns`);
- `asa-diagnosis-summary.txt`: the text of `asa_diagnosis_summary.summary_lines`;
- `asa-diagnosis-forecasts.csv`: the model's forecast table (its hash is in the results);
- `asa-diagnosis-timing.json`: wall-clock seconds, kept apart.
Floats in the tables are written as Python's `repr`, booleans as "True" or "False".

Exit status: 0 every unit and variant ran; 1 a unit and variant failed (recorded; the others go
on), a model refused or failing, a day that cannot be built from the tables; 2 could not run: an
unknown model, a `--dial` not in the dial family, a matrix without a dial family, a method or
reference not a controller method of the registry or the same as the other, a `--scale` not on
the grid, a number of days, splits or workers below 1, a missing or unverified truth build, a
table missing or differing from it, a model fitted on a scored unit.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

from fleetreserve import (
    asa_diagnosis_day,
    asa_diagnosis_summary,
    compare_methods,
    compare_run,
    dial_choice,
    dial_splits,
    gate_run,
    learn_run,
    rotation_whatif,
)
from fleetreserve.compare_methods import MethodError
from fleetreserve.day_split import _rank
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import write_table
from fleetreserve.gate_config import Matrix, MatrixError, load
from fleetreserve.lp import OTHER_SECONDS
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

RESULTS_NAME = "asa-diagnosis.json"
HOURS_NAME = "asa-diagnosis-hours.csv"
CARS_NAME = "asa-diagnosis-state-cars.csv"
SUMMARY_NAME = "asa-diagnosis-summary.txt"
TABLE_NAME = "asa-diagnosis-forecasts.csv"
TIMING_NAME = "asa-diagnosis-timing.json"
SEED = 1
DAYS = 50
VARIANTS = compare_run.VARIANTS
LABEL = "diagnosis of a comparison method on development units: development evidence"
NAME_MAPS = (
    "units",
    "variants",
    "configurations",
    "table_sha256",
    "choice_goals",
    "goal_tests",
    "solves",
    "rotation_whatif",
    "boxes",
)
_KEYS = ("unit", "variant", "day", "day_index")
HOURS_COLUMNS = (
    *_KEYS,
    *(f.name for f in fields(asa_diagnosis_day.HourRow) if f.name != "state_cars"),
)
CARS_COLUMNS = (
    *_KEYS,
    "path",
    "hour",
    *(f.name for f in fields(asa_diagnosis_day.StateCar)),
)

READINGS = {
    "D1": (
        "units: every gate unit gate_run.plan_unit can split, as in compare_run; any other is "
        "reported with its status and not diagnosed"
    ),
    "D2": (
        "days: the counting days ranked by day_split's rank of each day key for seed 1 (the "
        "first split seed), smallest first; per variant the first --days that offer under it"
    ),
    "D3": (
        "scale: --scale (a grid scale) for every unit and variant; without it, per unit and "
        "variant, among the splits of seeds 1 to --splits on which the reference's calibrated "
        "tier A offer is above 0, the calibrated scale the most of them have, a tie to the "
        "larger; no such split: status 'no reference offer'"
    ),
    "D4": (
        "every diagnosed day at that scale on each of its audited paths by "
        "asa_diagnosis_day.diagnose_day, the goal tests of asa_diagnosis_lp under lp.OTHER_SECONDS"
    ),
}

HOURS_DEFINITIONS = {
    "unit": "the gate unit",
    "variant": "forecast or oracle-departure",
    "day": "the day's key, '<date> <site>'",
    "day_index": "the day's index in the unit's counting days (gate_run.plan_unit order)",
    "path": "the audited path's index in PreparedDay.paths",
    "hour": "the service hour t (1 to n)",
    "request": "the request y_t, kWh",
    "method_delivered": "the method's fleet delivery in the hour, kWh",
    "reference_delivered": "the reference's fleet delivery in the hour, kWh",
    "shortfall": "request minus method_delivered, kWh (negative: above the request)",
    "method_first_failure": "whether the hour is the method's first failing hour on the path",
    "method_low": "sum of low_i from the method's state (asa_diagnosis_day.deliverable), kWh",
    "method_high": "sum of high_i from the method's state, kWh",
    "reference_low": "sum of low_i from the reference's state, kWh",
    "reference_high": "sum of high_i from the reference's state, kWh",
    "believed_most": "sum of priority_rules.most of the method's view, kWh",
    "believed_on_true": "sum of min(most_i, high_i) from the method's view and state, kWh",
    "deficit_plugged": "deficits labelled 'plugged' (asa_diagnosis_day.deficits), kWh",
    "deficit_departure": "deficits labelled 'departure', kWh",
    "deficit_rate": "deficits labelled 'rate', kWh",
    "deficit_cap": "deficits labelled 'cap', kWh",
    "overplanned": "sum of max(p_i - high_i, 0) of the method's planned powers in the hour, kWh",
    "cars_left_earlier": (
        "cars gone by the hour (d <= t) with a believed departure above d (d_hat > d; "
        "asa_diagnosis_day.left_earlier: d is the hour of unplugging rounded down, so a car "
        "believed to leave at 3.2 that left at 3.4 counts)"
    ),
    "charged_ahead": (
        "over the cars present in the hour (t < d), energy drawn before it under the method "
        "minus under the reference, kWh"
    ),
    "cause": "the cause (asa_diagnosis_day.CAUSES)",
    "goal": "for cause 'choice', the goal named (asa_diagnosis_summary choice_goals); else empty",
}

CARS_DEFINITIONS = {
    **{k: HOURS_DEFINITIONS[k] for k in (*_KEYS, "path", "hour")},
    "car": "the car's index in the steered sub-day",
    "true_departure": "its true departure hour d (it draws in hours t < d)",
    "believed_departure": (
        "its believed departure d_hat, the same in every hour of the day (the belief is one per "
        "day)"
    ),
    "present": "whether it is truly present in the hour (t < d)",
    "left_earlier": (
        "whether it is gone by the hour (d <= t) with a believed departure above d (d_hat > d; "
        "asa_diagnosis_day.left_earlier: d is the hour of unplugging rounded down, so a car "
        "believed to leave at 3.2 that left at 3.4 counts; one believed to leave at d does not)"
    ),
    "method_drawn": "energy it drew before the hour under the method, kWh",
    "reference_drawn": "energy it drew before the hour under the reference, kWh",
    "method_high": "its high_i from the method's state, kWh",
    "reference_high": "its high_i from the reference's state, kWh",
}


def definitions() -> dict:
    """Every key the results document holds (module text)."""
    return {
        **asa_diagnosis_summary.DEFINITIONS,
        "label": LABEL,
        "definitions": "this block",
        "name_maps": (
            "the keys of these maps are names (units, variants, goals, outcomes, files, kinds of "
            "box), not fields: " + ", ".join(NAME_MAPS)
        ),
        "readings": "the readings D1 to D4 of asa_diagnosis_run",
        "D1": READINGS["D1"],
        "D2": READINGS["D2"],
        "D3": READINGS["D3"],
        "D4": READINGS["D4"],
        "day_range": (
            "asa_diagnosis_day.deliverable: per truly present car high_i = min(r_i, max(e_i - "
            "x_i, 0)) and low_i = min(max(0, P_i - x_i - r_i max(d_i - 1 - t, 0)), high_i), 0 for "
            "a car not present; the fleet's range is [sum low_i, sum high_i]"
        ),
        "hours_columns": dict(HOURS_DEFINITIONS),
        "state_cars_columns": dict(CARS_DEFINITIONS),
        "method": "the method (--method); in solver_counts, the method's controllers",
        "reference": "the reference (--reference); in solver_counts, the reference's controllers",
        "model": "the forecasting model (--model)",
        "fit_units": "the units the model was fitted on",
        "dial": "the dial (--dial)",
        "configurations": "variant -> configuration name",
        "scale_rule": "how the scale was chosen (D3), in words",
        "days_rule": "how the days were chosen (D2), in words",
        "run": "the run's settings",
        "days": "in run: --days; in a variant: the keys of the days diagnosed, in order",
        "splits": "--splits, the split seeds 1 to N of the scale rule (read only without --scale)",
        "seed": "the seed of the day order (D2)",
        "grid": "dial_splits.GRID",
        "time_limit_seconds": "the time limit of every goal test",
        "variants": "variant -> its entry",
        "units": "unit -> its entry",
        "status": (
            "a unit's: gate_run.plan_unit's status; a variant's: 'diagnosed' or 'no reference "
            "offer'"
        ),
        "counting_days": "the unit's counting days",
        "configuration": "the variant's configuration name",
        "failed": (
            "in a variant: why it failed; in the document: every failed unit and variant, as "
            "'<unit> | <variant>: <why>'"
        ),
        "offering_days": "the unit's counting days that offer under the variant",
        "scale": "the scale of the diagnosis",
        "scale_source": "'--scale' or 'rule' (D3)",
        "reference_scales": (
            "for the rule: [scale, number of splits] for each calibrated scale of the splits on "
            "which the reference offers, by scale"
        ),
        "offering_splits": "for the rule: the splits on which the reference's tier A offer is > 0",
        "days_diagnosed": "the number of days diagnosed",
        "figures": "the figures of asa_diagnosis_summary.variant_figures",
        "timeouts": (
            "certificates and oracle_certificates: the days whose certificate timed out under the "
            "forecast and the oracle-departure variant; goal_tests: the goal tests that timed out"
        ),
        "certificates": "forecast-variant certificates that timed out",
        "oracle_certificates": "oracle-departure-variant certificates that timed out",
        "hours_table": "the file of the hour rows: file, rows, sha256",
        "state_cars_table": "the file of the state cars: file, rows, sha256",
        "forecast_table": "the model's forecast table: file, rows, sha256",
        "file": "a file name in the run folder",
        "rows": "the number of rows of that file",
        "sha256": "its SHA-256",
        "inputs": "gate_run.check_inputs: setting, truth_run, table_sha256",
        "setting": "the truth tables' setting",
        "truth_run": "the verified truth build's folder name",
        "table_sha256": "table file -> SHA-256",
        "matrix": "the matrix file's name and SHA-256",
    }


# --- days and scale ------------------------------------------------------------------------------


def day_order(name: str, keys) -> list[int]:
    """The indices of the unit's days in the seeded order of D2."""
    return sorted(range(len(keys)), key=lambda d: _rank(name, SEED, keys[d]))


def reference_scale(losses: dial_splits.UnitLosses, splits) -> tuple[float | None, list, int]:
    """(the scale of D3 or None, [scale, splits] pairs, the number of offering splits)."""
    scores = [dial_splits.score_split(losses, s, cal, test, "A") for s, cal, test in splits]
    offered = Counter(score.scale for score in scores if score.offer_kwh > 0)
    pairs = [[scale, offered[scale]] for scale in sorted(offered)]
    if not offered:
        return None, pairs, 0
    chosen = max(offered, key=lambda scale: (offered[scale], scale))
    return chosen, pairs, sum(offered.values())


# --- the per-day work ----------------------------------------------------------------------------


def _diagnose_task(job):
    prepared, beta, method, reference, time_limit = job
    found = asa_diagnosis_day.diagnose_day(prepared, beta, method, reference, time_limit=time_limit)
    found.runs = []  # the replays are for the tests; the command does not keep them
    return found


def _variant(pool, name, plan, days_wanted, configuration, methods, options, out):
    """One unit and variant: its entry, and its rows appended to `out` (hours, cars, timeouts)."""
    method, reference = methods
    count = len(plan.keys)
    jobs = [(name, i, configuration, ()) for i in range(count)]  # no car drawn (module text)
    prepared = pool.map(compare_run._prepare_task, jobs)
    key = "certificates" if out["variant"] == "forecast" else "oracle_certificates"
    out["timeouts"][key] += sum(p.certificate_outcome == "timeout" for p in prepared)
    entry = {
        "configuration": configuration.name,
        "offering_days": int(sum(not p.abstains for p in prepared)),
    }
    if options["scale"] is not None:
        scale, entry["scale_source"] = options["scale"], "--scale"
    else:
        losses, _ = compare_run.measure_controllers(
            prepared,
            [None] * count,
            (reference,),
            lambda jobs: pool.map(compare_run._compare_task, jobs),
        )
        splits = dial_choice.split_indices(
            name, plan.days, options["matrix"], range(1, options["splits"] + 1)
        )
        scale, entry["reference_scales"], entry["offering_splits"] = reference_scale(
            losses[reference.name], splits
        )
        entry["scale_source"] = "rule"
        if scale is None:
            entry["status"] = "no reference offer"
            return entry
    chosen = [d for d in day_order(name, plan.keys) if not prepared[d].abstains][:days_wanted]
    found = pool.map(
        _diagnose_task,
        [(prepared[d], scale, method, reference, options["time_limit"]) for d in chosen],
    )
    entry.update(status="diagnosed", scale=scale, days_diagnosed=len(chosen))
    entry["days"] = [plan.keys[d] for d in chosen]
    entry["figures"] = asa_diagnosis_summary.variant_figures(
        [(plan.keys[d], day) for d, day in zip(chosen, found, strict=True)]
    )
    out["timeouts"]["goal_tests"] += entry["figures"]["goal_tests"].get("timeout", 0)
    for d, day in zip(chosen, found, strict=True):
        keys = {"unit": name, "variant": out["variant"], "day": plan.keys[d], "day_index": d}
        for row in day.rows:
            fields_of = asdict(row)
            cars = fields_of.pop("state_cars")
            out["hours"].append({**keys, **fields_of})
            for car in cars:
                out["cars"].append({**keys, "path": row.path, "hour": row.hour, **car})
    return entry


def run(matrix: Matrix, units: dict, factory, configurations: dict, methods, options: dict):
    """Everything but the input checks and the files: (document, hour rows, car rows, timing).
    `options`: scale (None for D3), days, splits, workers, time_limit."""
    started = time.perf_counter()
    pool = compare_run.Pool(options["workers"], units, matrix, factory)
    entries, failed, timing = {}, {}, {"units": {}}
    out = {"hours": [], "cars": [], "timeouts": Counter(certificates=0, oracle_certificates=0)}
    out["timeouts"]["goal_tests"] = 0
    try:
        for name in matrix.gate_units:
            plan = gate_run.plan_unit(name, units[name], matrix)
            entry = {"status": plan.status, "counting_days": len(plan.keys), "variants": {}}
            entries[name] = entry
            if plan.status != "calibratable":
                continue
            begun = time.perf_counter()
            for variant, configuration in configurations.items():
                out["variant"] = variant
                try:
                    entry["variants"][variant] = _variant(
                        pool,
                        name,
                        plan,
                        options["days"],
                        configuration,
                        methods,
                        {**options, "matrix": matrix},
                        out,
                    )
                except Exception as problem:  # recorded, the other variants go on
                    failed[(name, variant)] = f"{type(problem).__name__}: {problem}"
                    entry["variants"][variant] = {
                        "configuration": configuration.name,
                        "failed": failed[(name, variant)],
                    }
            timing["units"][name] = time.perf_counter() - begun
    finally:
        pool.close()
    method, reference = methods
    scale_rule = (
        f"--scale {options['scale']:g} for every unit and variant"
        if options["scale"] is not None
        else (
            f"per unit and variant, the reference's most frequent calibrated tier A scale over "
            f"the splits of seeds 1 to {options['splits']} on which it offers (D3)"
        )
    )
    document = {
        "label": LABEL,
        "scope": asa_diagnosis_summary.SCOPE,
        "definitions": definitions(),
        "readings": dict(READINGS),
        "method": method.name,
        "reference": reference.name,
        "dial": configurations["forecast"].driver_risk,
        "configurations": {variant: c.name for variant, c in configurations.items()},
        "scale_rule": scale_rule,
        "days_rule": (
            f"the first {options['days']} offering days per variant in the order of seed {SEED} "
            "(D2)"
        ),
        "run": {
            "days": options["days"],
            "splits": options["splits"],
            "seed": SEED,
            "grid": list(dial_splits.GRID),
            "time_limit_seconds": options["time_limit"],
        },
        "units": entries,
        "timeouts": dict(sorted(out["timeouts"].items())),
        "failed": sorted(f"{a} | {b}: {why}" for (a, b), why in failed.items()),
    }
    timing["total_seconds"] = time.perf_counter() - started
    return document, out["hours"], out["cars"], timing


# --- files ---------------------------------------------------------------------------------------


def _cell(value) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def write_rows(rows: list[dict], path: Path, columns) -> int:
    """The rows as comma-separated text with one header line; returns their number."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row[c]) for c in columns])
    return len(rows)


def _file_entry(folder: Path, name: str, rows: int) -> dict:
    return {
        "file": name,
        "rows": rows,
        "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest(),
    }


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _checked(args) -> tuple:
    """The method and reference, or `MatrixError`/`MethodError` for a refused argument."""
    for flag, value in (("--workers", args.workers), ("--days", args.days)):
        if value < 1:
            raise MatrixError(f"{flag} must be at least 1")
    if args.splits < 1:
        raise MatrixError("--splits must be at least 1")
    if args.scale is not None and args.scale not in dial_splits.GRID:
        grid = ", ".join(f"{g:g}" for g in dial_splits.GRID)
        raise MatrixError(f"--scale {args.scale:g} is not a scale of the grid ({grid})")
    methods = compare_methods.select([args.method, args.reference])
    for method in methods:
        if method.kind != "controller":
            raise MethodError(f"method {method.name!r} is not a controller")
    return methods


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dial", type=float, required=True)
    parser.add_argument("--matrix", type=Path, default=dial_choice.MATRIX)
    parser.add_argument("--method", default="asa-es")
    parser.add_argument("--reference", default="llf")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--splits", type=int, default=dial_choice.SPLITS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        methods = _checked(args)
        matrix = load(args.matrix)
        configurations = compare_run.dial_configurations(matrix, args.dial)
        model = learn_run.load_model(args.model, matrix)
    except (MatrixError, MethodError) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    fit_units = tuple(model.fit_units)
    try:
        learn_run.check_model(args.model, model, fit_units, matrix.gate_units)
    except learn_run.FitOnScored as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    needed = tuple(dict.fromkeys((*matrix.gate_units, *fit_units)))
    try:
        inputs = gate_run.check_inputs(args.truth_run, args.tables, needed)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        loaded = {
            name: load_unit_days(
                args.tables,
                name,
                rho=matrix.rho,
                no_fit=matrix.no_fit,
                small_fleet=matrix.small_fleet,
            )
            for name in needed
        }
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    units = {name: loaded[name] for name in matrix.gate_units}
    fit_frames = {name: loaded[name].cars for name in fit_units}
    try:
        factory, table = dial_choice.make_factory(
            args.model, model, matrix, units, fit_frames, configurations["forecast"].kappa
        )
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: the model could not be scored: {problem}")
        return 1
    options = {
        "scale": args.scale,
        "days": args.days,
        "splits": args.splits,
        "workers": args.workers,
        "time_limit": OTHER_SECONDS,
    }
    document, hours, cars, timing = run(matrix, units, factory, configurations, methods, options)
    document["model"] = model.name
    document["fit_units"] = list(fit_units)
    rotation_whatif.mark(document, matrix, fit_units)
    document["inputs"] = inputs
    document["matrix"] = {
        "file": args.matrix.name,
        "sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
    }
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        write_table(table, folder / TABLE_NAME)
        document["forecast_table"] = _file_entry(folder, TABLE_NAME, len(table))
        count = write_rows(hours, folder / HOURS_NAME, HOURS_COLUMNS)
        document["hours_table"] = _file_entry(folder, HOURS_NAME, count)
        count = write_rows(cars, folder / CARS_NAME, CARS_COLUMNS)
        document["state_cars_table"] = _file_entry(folder, CARS_NAME, count)
    text = "\n".join(asa_diagnosis_summary.summary_lines(document)) + "\n"
    print(text, end="")
    if run_dir:
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / TIMING_NAME).write_text(_json(timing))
    return 1 if document["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
