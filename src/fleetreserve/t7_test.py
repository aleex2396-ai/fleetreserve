"""T7's test as a measurement on the training units (plan section 7, T7: "a constant multiplier is
optimal if marginal expected loss per unit value is equal across contexts. Tested per day type on
training sites"; section 11, O11). A measurement with a verdict rule, not a pass mark.

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.t7_test --model <name> --dial <epsilon> [--matrix <file>]
        [--splits N] [--workers N] [--units <unit> ...] [--tables <folder>] [--truth-run <folder>]

Before any table is opened the arguments, the units, the matrix, the dial and the model are
checked by the functions `size_whatif` checks them with (`learn_tune.check_units`,
`size_whatif.find_configuration`, `size_whatif.check_models`), with the same refusals and exit
statuses (see `size_whatif`); then the truth tables are checked by `gate_run.check_inputs` and
loaded as `size_whatif` loads them.

Per unit u (`--units`, default every unit of `truth_build.TRAIN_UNITS`):

1. The model is fitted on u's fit units and u's counting days prepared by
   `size_whatif.prepare_unit` (so the days, forecasts and certificates of `size_whatif`), with
   `--workers` passed to its `gate_run.Runner`.
2. Each prepared day gets its `size_scale.DayTerms` (None for a day that does not offer, a
   certificate that timed out included, counted by reason). Each offering day gets its raw day type
   (`t7_stat.raw_type`) from its date and its `public_holiday` flag, the column `size_whatif` adds
   (`learn_features.context_columns`, the pinned `holidays` package).
3. Splits: seeds 1 to `--splits`, by `dial_choice.split_indices` (the splits of `size_whatif` and
   `dial_choice`); a unit that the matrix's rule cannot split has none.
4. `t7_unit.measure_unit`: the constant scales, the readings and the verdicts (see that module for
   every reading, and `t7_stat` for the statistic).

After every unit, Holm's adjusted p-values across the units whose verdict in that tier is
"UNEQUAL" or "not shown unequal", per tier, loss and window (`t7_summary.add_holm`); they are
reported beside the verdicts and do not change them.

A unit for which any of steps 1 to 4 raises an error other than a failed internal check is
recorded under `failed_units` and the other units go on. A failed internal check
(`t7_unit.InternalCheckFailed`: the losses read at a split's scalar differ from
`size_scale.score`'s) stops the command at once and writes no file.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set, the summary also on standard output:
- `t7.json`: every number, by `t7_summary.results`;
- `t7-summary.txt`: the summary text (`t7_summary.summary_lines`);
- `t7-timing.json`: wall-clock seconds, kept apart from the other files.

Exit status: 0 every unit ran (and the files were written when `FLEETRESERVE_RUN_DIR` is set),
whatever the verdicts; 1 a failed internal
check (printed as "FAILED: internal check: ..."), a unit that failed, a day that could not be
built from the tables, or a model refused as `size_whatif` refuses it ("REFUSED"); 2 could not run
(a bad argument, matrix or model name, a dial not in the matrix's family, a unit that is not a
training unit, a model fitted on a scored unit or one of `forecast_model.OUTCOME_READERS`, a table
missing, no verified truth build at this commit, a table differing from it).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from fleetreserve import (
    dial_choice,
    gate_run,
    learn_features,
    learn_run,
    learn_tune,
    size_scale,
    size_whatif,
    t7_stat,
    t7_summary,
    t7_unit,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import MatrixError, load
from fleetreserve.truth_build import OUT, TRAIN_UNITS
from fleetreserve.truth_verify import CouldNotRun

RESULTS_NAME = "t7.json"
SUMMARY_NAME = "t7-summary.txt"
TIMING_NAME = "t7-timing.json"


def day_inputs(prepared) -> tuple[list, list, dict, int]:
    """Step 2 of the module text: (terms, raw types, days not offering by reason, timeouts)."""
    terms = [size_scale.day_terms(p) for p in prepared]
    offering = [d for d, t in enumerate(terms) if t is not None]
    raw = [None] * len(prepared)
    if offering:
        frame = pd.DataFrame(
            {
                "site": [prepared[d].site for d in offering],
                "date": [prepared[d].date for d in offering],
            }
        )
        holidays = learn_features.context_columns(frame)["public_holiday"].to_numpy(bool)
        for d, holiday in zip(offering, holidays, strict=True):
            raw[d] = t7_stat.raw_type(prepared[d].date, bool(holiday))
    reasons = {}
    for p in prepared:
        if p.abstains:
            reasons[p.abstain_reason] = reasons.get(p.abstain_reason, 0) + 1
    timeouts = sum(p.certificate_outcome == "timeout" for p in prepared)
    return terms, raw, dict(sorted(reasons.items())), timeouts


def measure(unit, models, matrix, configuration, units, loaded, seeds, workers) -> dict:
    """Steps 1 to 4 of the module text for unit `unit`."""
    model, fit_units = models[unit]
    prepared = size_whatif.prepare_unit(
        unit, model, fit_units, matrix, configuration, units, loaded, workers
    )
    terms, raw, reasons, timeouts = day_inputs(prepared)
    plan = gate_run.plan_unit(unit, units[unit], matrix)
    splits = (
        dial_choice.split_indices(unit, units[unit], matrix, seeds)
        if plan.status == "calibratable"
        else []
    )
    kind = size_whatif.site_type(units[unit])
    return {
        "site_type": t7_summary.TYPE_NAMES.get(kind),
        "status": plan.status,
        "fit_units": list(fit_units),
        "counting_days": len(prepared),
        "days_not_offering_by_reason": reasons,
        "certificate_timeouts": timeouts,
        **t7_unit.measure_unit(unit, terms, raw, splits),
    }


def run(name, models, matrix, configuration, units, loaded, *, splits, workers):
    """Every unit: (results document, timing). Raises `t7_unit.InternalCheckFailed`."""
    started = time.perf_counter()
    timing = {"units": {}}
    seeds = list(range(1, splits + 1))
    found, failed = {}, {}
    for unit in units:
        begun = time.perf_counter()
        try:
            found[unit] = measure(
                unit, models, matrix, configuration, units, loaded, seeds, workers
            )
        except t7_unit.InternalCheckFailed:
            raise
        except Exception as problem:  # recorded, the other units go on
            failed[unit] = f"{type(problem).__name__}: {problem}"
        timing["units"][unit] = time.perf_counter() - begun
    run_fields = {
        "model": name,
        "matrix_configuration": configuration.name,
        "dial": configuration.driver_risk,
        "kappa": configuration.kappa,
        "splits": splits,
        "seeds": [seeds[0], seeds[-1]] if seeds else [],
        "n_calibration": matrix.n_calibration,
        "calibration_half_below": matrix.calibration_half_below,
        "rounds": t7_stat.ROUNDS,
        "windows": list(t7_stat.WINDOWS),
        "main_window": t7_stat.MAIN_WINDOW,
        "level": t7_stat.LEVEL,
        "min_type_days": t7_stat.MIN_TYPE_DAYS,
        "units": list(units),
    }
    document = t7_summary.results(found, failed, run_fields)
    timing["total_seconds"] = time.perf_counter() - started
    return document, timing


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dial", type=float, required=True)
    parser.add_argument("--matrix", type=Path, default=dial_choice.MATRIX)
    parser.add_argument("--splits", type=int, default=dial_choice.SPLITS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--units", nargs="+", default=list(TRAIN_UNITS))
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if args.splits < 1:
            raise MatrixError("--splits must be at least 1")
        units = learn_tune.check_units(args.units)
        matrix = load(args.matrix)
        configuration = size_whatif.find_configuration(matrix, args.dial)
        models = size_whatif.check_models(args.model, matrix, units)
    except (MatrixError, learn_tune.NotTraining, learn_run.FitOnScored) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    needed = tuple(dict.fromkeys(u for unit in units for u in (unit, *models[unit][1])))
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
    scored = {name: loaded[name] for name in units}
    try:
        document, timing = run(
            args.model,
            models,
            matrix,
            configuration,
            scored,
            loaded,
            splits=args.splits,
            workers=args.workers,
        )
    except t7_unit.InternalCheckFailed as problem:
        print(f"FAILED: internal check: {problem}")
        return 1
    document["inputs"] = inputs
    document["matrix"] = {
        "file": args.matrix.name,
        "sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
    }
    text = "\n".join(t7_summary.summary_lines(document)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / TIMING_NAME).write_text(_json(timing))
    return 1 if document["failed_units"] else 0


if __name__ == "__main__":
    sys.exit(main())
