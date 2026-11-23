"""Measure the one-way rung's re-solves: which requirement makes them infeasible, and how far
they are from feasible (plan section 5, "Rungs").

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.resolve_diagnosis --model <name> --configuration <name>
        --scale <beta> [<beta> ...] [--unit <name> ...] [--matrix <file>] [--workers N]
        [--tables <folder>] [--truth-run <folder>]

A measurement. It reads the rung as the gate runs it and changes nothing in it; it writes no
choice and no recommendation. Plan section 5: after an infeasible re-solve "the predecessor's
plan stays". This command records, for every re-solve the one-way rung attempts, its outcome
and state, and for every infeasible one the requirement groups whose removal makes it feasible,
its distances from feasible and what was true in hindsight (`resolve_diagnosis_day`,
`resolve_diagnosis_lp`; the true departures are used for measurement only).

`--model <name>`: the module `fleetreserve.models.<name>`, checked, fitted and turned into the
forecaster exactly as `dial_choice` does (`dial_choice.make_factory`). `--matrix`: a run matrix of
the gate's format (default: the committed one). `--configuration`: one configuration of the
matrix, by name. `--unit`: a gate unit of the matrix, repeatable (default: every gate unit).
`--scale`: one or more scales above 0 of the certified box. The truth tables are checked as
`gate_run.check_inputs` checks them (a verified truth build at the commit checked out now, of
the tables' setting, with the tables' hashes).

Per unit: every counting day is prepared under the configuration (`gate_day.prepare_day`); every
day that offers is run at each scale on each of its audited paths
(`resolve_diagnosis_day.rows`). Days that abstain have no re-solve and are counted.

Timeouts are recorded outcomes. A day whose certificate times out abstains (`gate_day`). A
re-solve that times out is a row with the outcome "timeout" and is not diagnosed. Every relaxed
programme has the 300-second limit for other programmes (plan section 13); one that times out is
recorded as "timeout", never as infeasible, and is counted. A relaxed programme whose first
attempt ends in a solver error is tried once more with the simplex method (never the re-solve
itself, and never after a timeout); first errors and answered retries are counted in the rows and
the summary (`resolve_diagnosis_lp`, "Solver errors").

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary also on
standard output. They hold no wall-clock time and do not depend on `--workers`:
- `resolve-diagnosis.csv`: one row per re-solve (`COLUMNS`);
- `resolve-diagnosis.json`: per unit and scale the counts, shares and quantiles
  (`resolve_diagnosis_summary`), with `definitions`;
- `resolve-diagnosis-summary.txt`: the text summary;
- `resolve-diagnosis-forecasts.csv`: the model's forecast table (its hash is in the results).

Exit status: 0 every unit ran (also when no re-solve is infeasible, which the summary then
says); 1 the model was refused or failed, or a unit failed; 2 could not run (a bad matrix,
argument, model, configuration or unit name, a model fitted on a gate unit, a rotation what-if
matrix (`rotation_whatif` in `gate_config`: this command does not label its outputs as one), a
table missing, no verified truth build at this commit, a table differing from it).
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

from fleetreserve import dial_choice, gate_run, learn_run, resolve_diagnosis_day
from fleetreserve import resolve_diagnosis_summary as summary
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import write_table
from fleetreserve.gate_config import Configuration, Matrix, MatrixError, load
from fleetreserve.resolve_diagnosis_lp import GROUPS
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

MATRIX = dial_choice.MATRIX
ROWS_NAME = "resolve-diagnosis.csv"
RESULTS_NAME = "resolve-diagnosis.json"
SUMMARY_NAME = "resolve-diagnosis-summary.txt"
TABLE_NAME = "resolve-diagnosis-forecasts.csv"
COLUMNS = ("unit", "configuration", "site", "date", "scale", *resolve_diagnosis_day.COLUMNS)


def definitions() -> dict:
    return {
        "re_solve": (
            "one attempt of the one-way rung to plan the remaining hours of a path, at the start "
            "of a service hour; its outcome is the solver's (optimal, infeasible, timeout, error)"
        ),
        "groups": {group: list(kinds) for group, kinds in GROUPS.items()},
        "groups_text": (
            "requirement groups of the re-solve programme, each a set of row kinds of "
            "resolve.draft; a group is removed by leaving its rows out; the re-solve has no "
            "driver-risk rows"
        ),
        "cause": (
            "the single groups whose removal alone makes an infeasible re-solve feasible, joined "
            "by '|'; when there is none, the pairs ('a+b') that do; 'no single group and no pair' "
            "when no pair does; 'not determined' when a removal needed for the answer ended in a "
            "timeout or an error"
        ),
        "departure_relaxed": (
            "the re-solve with every present car's power limit taken as if it stayed through "
            "each hour (a relaxation of the departure bound, not a removal)"
        ),
        "box_scale": summary.DEFINITIONS["box_scale"],
        "need_slack_kwh": (
            "with the box as it is, the smallest total lowering of the needed energy, summed over "
            "present cars, that makes the re-solve feasible; outcome 'infeasible': none does"
        ),
        "request_miss_kwh": (
            "the smallest amount by which the present cars' delivery in the current hour must "
            "differ from its known request for the other rows to hold; above 0 when they deliver "
            "less; outcome 'infeasible': none does"
        ),
        "hindsight": (
            "early_gone_cars and early_gone_expected_kwh read the true departures, for "
            "measurement only: steered cars that had truly left by the start of the hour, before "
            "the forecast departure the box was built on, and the energy the fixed plan still "
            "planned for them in the service hours from that hour on along the path"
        ),
        "early_leaver": summary.DEFINITIONS["early_leaver"],
        "solver_errors": summary.DEFINITIONS["solver_errors"],
        "timeouts": (
            "a re-solve that times out is a row with outcome 'timeout' and is not diagnosed; a "
            "relaxed programme that times out is recorded as 'timeout', never as infeasible"
        ),
    }


def scale_text(scale: float) -> str:
    return repr(float(scale))


def measure_unit(runner, name: str, n_days: int, configuration: Configuration, scales):
    """(entry of the results document, rows) of one unit."""
    prepared = runner.map(gate_run._prepare_task, [(name, i, configuration) for i in range(n_days)])
    jobs = [(p, scale) for p in prepared for scale in scales]
    found = runner.map(resolve_diagnosis_day.task, jobs)
    rows = []
    for (p, scale), day_rows in zip(jobs, found, strict=True):
        head = {
            "unit": name,
            "configuration": configuration.name,
            "site": p.site,
            "date": p.date,
            "scale": scale_text(scale),
        }
        rows += [{**head, **row} for row in day_rows]
    entry = {
        "counting_days": n_days,
        "offering_days": sum(not p.abstains for p in prepared),
        "certificate_timeouts": sum(p.certificate_outcome == "timeout" for p in prepared),
        "paths_per_day": max([0, *(p.paths.shape[0] for p in prepared)]),
        "scales": {
            scale_text(scale): summary.summarise(
                [r for r in rows if r["scale"] == scale_text(scale)]
            )
            for scale in scales
        },
        "all_scales": summary.summarise(rows),
    }
    return entry, rows


def run(matrix: Matrix, units: dict, factory, configuration: Configuration, names, scales, workers):
    """Everything but the input checks and the writing of files: (results document, rows).
    `units`: unit name -> `UnitDays` for the matrix's gate units; `factory`: as
    `gate_run.run_gate` takes it; `names`: the units measured."""
    runner = gate_run.Runner(workers, units, matrix, factory)
    entries, rows, failed = {}, [], []
    try:
        for name in names:
            try:
                entry, found = measure_unit(
                    runner, name, len(units[name].days), configuration, scales
                )
            except Exception as problem:  # recorded, the other units go on
                failed.append(f"{name}: {type(problem).__name__}: {problem}")
                entries[name] = {"failed": failed[-1]}
                continue
            entries[name] = entry
            rows += found
    finally:
        runner.close()
    done = [e for e in entries.values() if "failed" not in e]
    document = {
        "label": summary.LABEL,
        "definitions": definitions(),
        "run": {
            "configuration": configuration.name,
            "configuration_fields": asdict(configuration),
            "rung": resolve_diagnosis_day.RUNG,
            "scales": [scale_text(s) for s in scales],
            "units": list(names),
        },
        "units": entries,
        "timeouts": {
            "certificates": sum(e["certificate_timeouts"] for e in done),
            "re_solves": sum(e["all_scales"]["by_outcome"].get("timeout", 0) for e in done),
            "relaxed_programmes": sum(
                e["all_scales"]["infeasible"]["relaxed_timeouts"] for e in done
            ),
        },
        "failed_units": failed,
    }
    return document, rows


def rows_csv(rows: list[dict]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--scale", type=float, nargs="+", required=True)
    parser.add_argument("--unit", action="append", default=None)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if not all(math.isfinite(s) and s > 0 for s in args.scale):
            raise MatrixError("every --scale must be a number above 0")
        if len(set(args.scale)) != len(args.scale):
            raise MatrixError("a --scale is given twice")
        matrix = load(args.matrix)
        if matrix.rotation_whatif:
            raise MatrixError(
                "a rotation what-if matrix (rotation_whatif) is run by learn_run and dial_choice, "
                "which label it; the outputs of this command carry no such label"
            )
        configuration = matrix.configuration(args.configuration)
        names = tuple(args.unit) if args.unit else matrix.gate_units
        outside = [name for name in names if name not in matrix.gate_units]
        if outside or len(set(names)) != len(names):
            raise MatrixError(
                f"--unit must name distinct gate units of the matrix {list(matrix.gate_units)}, "
                f"got {list(names)}"
            )
        model = learn_run.load_model(args.model, matrix)
    except MatrixError as problem:
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
            args.model, model, matrix, units, fit_frames, configuration.kappa
        )
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: the model could not be scored: {problem}")
        return 1
    document, rows = run(matrix, units, factory, configuration, names, args.scale, args.workers)
    document["model"] = model.name
    document["fit_units"] = list(fit_units)
    document["inputs"] = inputs
    document["matrix"] = {
        "file": args.matrix.name,
        "sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
    }
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        write_table(table, folder / TABLE_NAME)
        document["forecast_table"] = {
            "file": TABLE_NAME,
            "rows": len(table),
            "sha256": hashlib.sha256((folder / TABLE_NAME).read_bytes()).hexdigest(),
        }
    text = "\n".join(summary.summary_lines(document)) + "\n"
    print(text, end="")
    if run_dir:
        (folder / ROWS_NAME).write_text(rows_csv(rows))
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
    return 1 if document["failed_units"] else 0


if __name__ == "__main__":
    sys.exit(main())
