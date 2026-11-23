"""What sizing each day on its own would give: a measurement on the training units (plan section 7,
day-size net and T7, "tested per day type on training sites"; section 6 for the calibration).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.size_whatif --model <name> --dial <epsilon> [--matrix <file>]
        [--splits N] [--workers N] [--units <unit> ...] [--tables <folder>] [--truth-run <folder>]

It is a what-if on training units, not a run of the plan; the results' `label` and the summary's
first line say so.

`--units` (default: every unit of `truth_build.TRAIN_UNITS`): the units labelled; any other unit
is refused (exit 2) before a table is opened (`learn_tune.check_units`). For each unit u the model
`fleetreserve.models.<name>` is built by its `make_model` from the matrix with u as its only gate
unit, and `learn_run.check_model` must pass with the gate units (u,). Its refusals end the
command before a table is opened; the first that applies decides: `fit_units` not all training
units, exit 1 ("REFUSED"); `fit_units` holding u, exit 2 ("COULD NOT RUN"); a declared
`reads_other_day_outcomes` by a model not in `forecast_model.OUTCOME_READERS`, exit 1
("REFUSED"); one by a model in it, exit 2 ("COULD NOT RUN": here it would read u's own outcomes).
The truth tables of the units and of the models' fit units are checked as
`gate_run.check_inputs` checks them.

Per unit u:

1. The model is fitted and the forecast table made as `dial_choice.make_factory` does, at the
   levels up to the kappa of the configuration. The configuration is the dial configuration of
   `dial_choice.find_family` whose `driver_risk` equals `--dial` (none: exit 2). Every counting day
   is prepared by `gate_day.prepare_day` (certificate with the box time limit).
2. Each day that offers gets its label (`size_label`), its context row (`size_features`, with
   `learn_features.context_columns` added) and its loss terms (`size_scale.day_terms`). A day that
   does not offer, a certificate that timed out included, gets no row and is counted by reason.
3. The reference predictor (`size_gbm.leave_one_unit_out`): u's rows are predicted by boosters
   fitted on the rows of the other labelled units. With `--units` naming fewer units, fewer rows
   are fitted on. Those rows need not be free of u's outcomes: the forecast model of another unit
   v is fitted on v's fit units, which can hold u (with `lgbm_tail_rotation`, every training unit
   but v), so v's forecasts, certificates and context rows, and with them the boosters that size
   u's days, can depend on u's outcomes.
4. Sizings (`size_scale`): `constant` (size 1); `gbm-<level>-scale` and `gbm-<level>-abstain`
   (the booster's prediction as the raw size); `oracle-scale` and `oracle-abstain` (the day's own
   unclipped label as the raw size: it uses the day's outcome, so it is a reference, not a method).
   The sizes of a unit are computed once, after every unit's rows exist and before any split of
   that unit is scored, and are the same on every split. `constant` is 1 on every offering day
   whatever the labels. Because of step 3 the gbm sizings can depend on the scored unit's
   outcomes, so the plan's argument for a function fitted on other units does not cover them, and
   a gbm sizing's measured gain is not shown to be that of a sizing fitted without those outcomes.
5. Splits: seeds 1 to `--splits`, each made by `dial_choice.split_indices` (the matrix's
   calibration-days rule), so the splits of `dial_choice`; they read the unit's name, its days'
   dates and sites, the seed and that rule, no size and no outcome. A unit that the rule cannot
   split is labelled but not scored. Per split, sizing and tier: `size_scale.calibrate` on the
   calibration days and `size_scale.score` on the test days.

The day-size net (`label_units` with `net_sources`, which `size_resolve_whatif` passes and this
command does not, so its outputs are as before): beside step 3, the net's predictions of every
labelled day (`net_predictions`: `size_net.leave_one_unit_out` on the labelled days, the rows the
boosters are fitted on, and on the cars plugged in at τ of the labelled units,
`size_net_inputs.car_frame`), in the frame's columns `NET_DAY_COLUMNS`; beside step 4 the
sizings `net-<level>-<arm>`, the net's prediction as the raw size (`predicted_sizings`, as for
the boosters). Step 3's caveat applies to them unchanged: the net reads no forecast, but the
other units' labels it is fitted on come from their certificates, so from forecast models that
can be fitted on u.

Conformal training (`label_units` with `conformal`, which `size_resolve_whatif` passes with the
net and this command does not): beside the net, the outputs of `size_ct.leave_one_unit_out` on the
same labelled days, each with its certified value (`box_value_kwh`), and the same cars, one per
arm of `size_scale.ARMS`, in the frame's columns `CT_DAY_COLUMNS`; beside step 4 the sizings
`ct-<arm>`, each arm's output through that arm (`conformal_sizings`). Step 3's caveat applies to
them as to the net's.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set, the summary also on standard output:
- `size-whatif-days.csv`: one row per labelled day (`DAY_COLUMNS`);
- `size-whatif.json`: every number, by `size_whatif_summary.results`;
- `size-whatif-summary.txt`: the summary text (`size_whatif_summary.summary_lines`);
- `size-whatif-timing.json`: wall-clock seconds, kept apart from the other files.

Exit status: 0 every unit ran; 1 the model was refused (fit units that are not all training
units, or a declared outcome reader not in `forecast_model.OUTCOME_READERS`) or failed, a day
could not be built from the tables, or a unit failed; 2 could not run (a bad argument, matrix or
model name, a dial not in the matrix's family, a unit that is not a training unit, a model fitted
on a labelled unit or one of `forecast_model.OUTCOME_READERS`, a table missing, no verified truth
build at this commit, a table differing from it).
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve import (
    dial_choice,
    gate_run,
    learn_features,
    learn_run,
    learn_tune,
    size_ct,
    size_features,
    size_gbm,
    size_label,
    size_net,
    size_net_inputs,
    size_scale,
    size_whatif_summary,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import MatrixError, load
from fleetreserve.truth_build import OUT, TRAIN_UNITS
from fleetreserve.truth_verify import CouldNotRun

MATRIX = dial_choice.MATRIX
DAYS_NAME = "size-whatif-days.csv"
RESULTS_NAME = "size-whatif.json"
SUMMARY_NAME = "size-whatif-summary.txt"
TIMING_NAME = "size-whatif-timing.json"
CONTEXT = (*size_features.COLUMNS, *learn_features.CONTEXT_COLUMNS)
LABEL_COLUMNS = ("label_scale", size_gbm.LABEL, "centre_inside", "fixed_plan_scale")
ROW_COLUMNS = ("day", "unit", "site", "date", *LABEL_COLUMNS, *size_features.COLUMNS)
DAY_COLUMNS = (
    "unit",
    "site",
    "date",
    *LABEL_COLUMNS,
    *CONTEXT,
    *(f"gbm_q{level:g}" for level in size_gbm.LEVELS),
)
NET = size_net.KIND
NET_DAY_COLUMNS = tuple(f"{NET}_q{level:g}" for level in size_gbm.LEVELS)
CT = size_ct.KIND
CT_DAY_COLUMNS = tuple(f"{CT}_{arm}" for arm in size_scale.ARMS)


class OutcomeReader(learn_run.FitOnScored):
    """A model that reads other days' outcomes; here it would read the labelled unit's own."""


@dataclass
class UnitData:
    """One unit after step 2 of the module text."""

    name: str
    site_type: int | None
    counting_days: int
    status: str
    fit_units: tuple[str, ...]
    rows: list  # one dict per offering day, in day order, with its day index under "day"
    terms: list  # `size_scale.DayTerms` or None, one per counting day
    not_offering: dict  # abstain reason -> days
    certificate_timeouts: int
    splits: list  # (seed, calibration indices, test indices)


def unit_matrix(matrix, unit: str):
    return replace(matrix, gate_units=(unit,), rotation_whatif=True)


def check_models(name: str, matrix, units) -> dict:
    """Per unit the model, built and checked (module text); raises `MatrixError`,
    `learn_run.FitOnScored` (could not run) or `learn_run.Refused`."""
    models = {}
    for unit in units:
        model = learn_run.load_model(name, unit_matrix(matrix, unit))
        fit_units = tuple(model.fit_units)
        if learn_run.check_model(name, model, fit_units, (unit,)):
            raise OutcomeReader(
                f"model {name} reads other days' outcomes, which here are those of the labelled "
                f"unit {unit}"
            )
        models[unit] = (model, fit_units)
    return models


def find_configuration(matrix, dial: float):
    _, dials = dial_choice.find_family(matrix)
    for configuration in dials:
        if configuration.driver_risk == dial:
            return configuration
    raise MatrixError(
        f"no dial configuration at {dial} in the matrix; its dials are "
        f"{[c.driver_risk for c in dials]}"
    )


def label_days(prepared) -> tuple[list, list, dict, int]:
    """Step 2 of the module text: (rows, terms, days not offering by reason, timeouts)."""
    rows, terms, reasons = [], [], {}
    for index, day in enumerate(prepared):
        terms.append(size_scale.day_terms(day))
        if day.abstains:
            reasons[day.abstain_reason] = reasons.get(day.abstain_reason, 0) + 1
            continue
        label = size_label.day_label(day)
        rows.append(
            {
                "day": index,
                "unit": day.unit,
                "site": day.site,
                "date": day.date,
                "label_scale": label.scale,
                size_gbm.LABEL: label.clipped,
                "centre_inside": label.centre_inside,
                "fixed_plan_scale": label.fixed_plan_scale,
                **size_features.context_row(day),
            }
        )
    timeouts = sum(day.certificate_outcome == "timeout" for day in prepared)
    return rows, terms, dict(sorted(reasons.items())), timeouts


def prepare_unit(name, model, fit_units, matrix, configuration, units, loaded, workers):
    """Steps 1 and 2 of the module text for unit `name`."""
    days = units[name]
    factory, _ = dial_choice.make_factory(
        name,
        model,
        unit_matrix(matrix, name),
        {name: days},
        {u: loaded[u].cars for u in fit_units},
        configuration.kappa,
    )
    runner = gate_run.Runner(workers, {name: days}, unit_matrix(matrix, name), factory)
    try:
        prepared = runner.map(
            gate_run._prepare_task, [(name, i, configuration) for i in range(len(days.days))]
        )
    finally:
        runner.close()
    return prepared


def site_type(days) -> int | None:
    types = {learn_features.SITE_TYPES[t.record.site] for t in days.days}
    if len(types) > 1:
        raise ValueError(f"the unit {days.unit} holds sites of both types")
    return types.pop() if types else None


def with_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(**{c: [] for c in learn_features.CONTEXT_COLUMNS})
    calendar = learn_features.context_columns(frame)
    return frame.assign(
        public_holiday=calendar["public_holiday"].astype(float),
        site_type=calendar["site_type"].astype(float),
    )


def predicted_sizings(data: UnitData, predictions: dict, prefix: str) -> dict:
    """The sizings `<prefix>-<level>-<arm>` of a predictor of step 4: name -> q, one per
    counting day. `predictions`: level -> one prediction per row of `data.rows`, the raw sizes."""
    n = data.counting_days
    offering = np.zeros(n, dtype=bool)
    rows = [row["day"] for row in data.rows]
    offering[rows] = True
    out = {}
    for level in size_gbm.LEVELS:
        raw = np.full(n, math.nan)
        raw[rows] = predictions[float(level)]
        if np.isnan(raw[rows]).any():
            continue  # no other unit to fit on (`size_gbm`, `size_net`): no sizing from it
        for arm in size_scale.ARMS:
            out[f"{prefix}-{level:g}-{arm}"] = size_scale.sizes(raw, offering, arm)
    return out


def conformal_sizings(data: UnitData, outputs: dict) -> dict:
    """The sizings `ct-<arm>` (module text, conformal training): name -> q, one per counting day.
    `outputs`: arm -> one raw size per row of `data.rows`, each read through its own arm."""
    n = data.counting_days
    offering = np.zeros(n, dtype=bool)
    rows = [row["day"] for row in data.rows]
    offering[rows] = True
    out = {}
    for arm in size_scale.ARMS:
        raw = np.full(n, math.nan)
        raw[rows] = outputs[arm]
        if np.isnan(raw[rows]).any():
            continue  # no other unit to fit on (`size_ct`): no sizing from it
        out[f"{CT}-{arm}"] = size_scale.sizes(raw, offering, arm)
    return out


def sizings(
    data: UnitData, predictions: dict, net: dict | None = None, ct: dict | None = None
) -> dict:
    """The sizings of step 4: name -> q, one per counting day. `predictions`: level -> one
    booster prediction per row of `data.rows`; `net`: the same of the day-size net, or None;
    `ct`: arm -> the conformal training's outputs per row, or None."""
    n = data.counting_days
    offering = np.zeros(n, dtype=bool)
    rows = [row["day"] for row in data.rows]
    offering[rows] = True
    out = {"constant": size_scale.sizes(np.where(offering, 1.0, math.nan), offering, "scale")}
    out.update(predicted_sizings(data, predictions, "gbm"))
    if net is not None:
        out.update(predicted_sizings(data, net, NET))
    if ct is not None:
        out.update(conformal_sizings(data, ct))
    raw = np.full(n, math.nan)
    raw[rows] = [row["label_scale"] for row in data.rows]
    for arm in size_scale.ARMS:
        out[f"oracle-{arm}"] = size_scale.sizes(raw, offering, arm)
    return out


def measure(data: UnitData, q_by_sizing: dict) -> dict:
    """Step 5 of the module text: sizing -> tier -> list of (seed, `size_scale.Scalar`,
    `size_scale.TestScore`)."""
    out = {}
    for sizing, q in q_by_sizing.items():
        per_tier = {tier: [] for tier in size_scale.TIERS}
        for seed, calibration, test in data.splits:
            scalars = size_scale.calibrate(data.terms, q, calibration)
            for tier, scalar in scalars.items():
                found = size_scale.score(data.terms, q, scalar.used, test)
                per_tier[tier].append((seed, scalar, found))
        out[sizing] = per_tier
    return out


@dataclass
class Labelled:
    """Every unit after steps 1 to 4 of the module text."""

    data: dict  # unit -> `UnitData`
    prepared: dict  # unit -> its prepared days (`gate_day.PreparedDay`), in day order
    frame: pd.DataFrame  # one row per labelled day (`ROW_COLUMNS`, calendar and gbm columns)
    per_unit: dict  # unit -> level -> {"gbm": ..., "constant": ...} predictions of its rows
    q_all: dict  # unit -> sizing -> q, one per counting day (step 4)
    failed: dict  # unit -> why it failed
    timing: dict
    net_fits: dict = field(default_factory=dict)  # unit -> level -> `size_net.Fitted.record`
    ct_fits: dict = field(default_factory=dict)  # unit -> arm -> `size_ct.Fitted.record`


def net_cars(data: dict, units: dict, sources) -> pd.DataFrame:
    """The cars plugged in at τ of the labelled units `data`, from their cars tables
    (`size_net_inputs.car_frame`)."""
    cars = [
        size_net_inputs.car_frame(units[u].cars, rho=units[u].rho, sources=sources) for u in data
    ]
    return pd.concat(cars, ignore_index=True) if cars else pd.DataFrame()


def net_predictions(frame: pd.DataFrame, cars: pd.DataFrame, records: dict | None = None) -> dict:
    """The day-size net's predictions of every row of `frame` (module text, the net): level ->
    {"net": predictions}, from the cars `cars` (`net_cars`); `records`, when given, gets the
    record of each network fitted (`size_net.leave_one_unit_out`)."""
    return size_net.leave_one_unit_out(
        frame[["unit", "site", "date", size_gbm.LABEL]], cars, records=records
    )


def label_units(
    models,
    matrix,
    configuration,
    units,
    loaded,
    *,
    splits,
    workers,
    net_sources=None,
    conformal=False,
) -> Labelled:
    """Steps 1 to 4 of the module text for every unit; with `net_sources` (the context sources of
    `size_net_inputs`) also the day-size net, and with `conformal` as well conformal training
    (module text)."""
    if conformal and net_sources is None:
        raise ValueError("conformal training needs the net's context sources")
    timing = {"units": {}}
    seeds = list(range(1, splits + 1))
    data, prepared_by_unit, failed = {}, {}, {}
    for unit in units:
        begun = time.perf_counter()
        model, fit_units = models[unit]
        try:
            prepared = prepare_unit(
                unit, model, fit_units, matrix, configuration, units, loaded, workers
            )
            rows, terms, reasons, timeouts = label_days(prepared)
        except Exception as problem:  # recorded, the other units go on
            failed[unit] = f"{type(problem).__name__}: {problem}"
            continue
        prepared_by_unit[unit] = prepared
        plan = gate_run.plan_unit(unit, units[unit], matrix)
        data[unit] = UnitData(
            unit,
            site_type(units[unit]),
            len(prepared),
            plan.status,
            fit_units,
            rows,
            terms,
            reasons,
            timeouts,
            dial_choice.split_indices(unit, units[unit], matrix, seeds)
            if plan.status == "calibratable"
            else [],
        )
        timing["units"][unit] = time.perf_counter() - begun
    frame = with_calendar(
        pd.DataFrame([row for d in data.values() for row in d.rows], columns=list(ROW_COLUMNS))
    )
    predictions = size_gbm.leave_one_unit_out(frame, CONTEXT)
    for level in size_gbm.LEVELS:
        frame[f"gbm_q{level:g}"] = predictions[float(level)]["gbm"]
    net_fits, ct_fits, ct = {}, {}, None
    if net_sources is not None:
        begun = time.perf_counter()
        cars = net_cars(data, units, net_sources)
        net = net_predictions(frame, cars, net_fits)
        for level in size_gbm.LEVELS:
            frame[f"{NET}_q{level:g}"] = net[float(level)][NET]
            predictions[float(level)][NET] = net[float(level)][NET]
        timing["net_seconds"] = time.perf_counter() - begun
        if conformal:
            begun = time.perf_counter()
            columns = ["unit", "site", "date", size_gbm.LABEL, size_ct.VALUE]
            ct = size_ct.leave_one_unit_out(frame[columns], cars, records=ct_fits)
            for arm in size_scale.ARMS:
                frame[f"{CT}_{arm}"] = ct[arm]
            timing["ct_seconds"] = time.perf_counter() - begun
    q_all, per_unit = {}, {}
    for unit, d in data.items():
        rows = (frame["unit"] == unit).to_numpy(bool)
        per_unit[unit] = {
            level: {kind: values[rows] for kind, values in found.items()}
            for level, found in predictions.items()
        }
        net_of = None
        if net_sources is not None:
            net_of = {level: f[NET] for level, f in per_unit[unit].items()}
        ct_of = None if ct is None else {arm: ct[arm][rows] for arm in size_scale.ARMS}
        q_all[unit] = sizings(
            d, {level: f["gbm"] for level, f in per_unit[unit].items()}, net_of, ct_of
        )
    return Labelled(
        data, prepared_by_unit, frame, per_unit, q_all, failed, timing, net_fits, ct_fits
    )


def run(name, models, matrix, configuration, units, loaded, *, splits, workers):
    """Steps 1 to 5 for every unit: (results document, day rows, timing)."""
    started = time.perf_counter()
    found = label_units(
        models, matrix, configuration, units, loaded, splits=splits, workers=workers
    )
    data, frame, timing = found.data, found.frame, found.timing
    measured = {unit: measure(d, found.q_all[unit]) for unit, d in data.items()}
    document = size_whatif_summary.results(
        data, frame, found.per_unit, found.q_all, measured, found.failed
    )
    document["run"] = run_fields(name, matrix, configuration, units, splits)
    day_rows = frame[list(DAY_COLUMNS)].to_dict("records")
    timing["total_seconds"] = time.perf_counter() - started
    return document, day_rows, timing


def run_fields(name, matrix, configuration, units, splits) -> dict:
    """The results' "run" entry: the model, configuration, splits and units of the run."""
    seeds = list(range(1, splits + 1))
    return {
        "model": name,
        "matrix_configuration": configuration.name,
        "dial": configuration.driver_risk,
        "kappa": configuration.kappa,
        "splits": splits,
        "seeds": [seeds[0], seeds[-1]] if seeds else [],
        "n_calibration": matrix.n_calibration,
        "calibration_half_below": matrix.calibration_half_below,
        "floor": size_scale.FLOOR,
        "levels": list(size_gbm.LEVELS),
        "units": list(units),
    }


def _text(value) -> str:
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return repr(float(value))


def days_csv(rows, columns=DAY_COLUMNS) -> str:
    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_text(row[c]) for c in columns])
    return handle.getvalue()


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def parser(description: str) -> argparse.ArgumentParser:
    """The command line of the module text."""
    found = argparse.ArgumentParser(description=description)
    found.add_argument("--model", required=True)
    found.add_argument("--dial", type=float, required=True)
    found.add_argument("--matrix", type=Path, default=MATRIX)
    found.add_argument("--splits", type=int, default=dial_choice.SPLITS)
    found.add_argument("--workers", type=int, default=1)
    found.add_argument("--units", nargs="+", default=list(TRAIN_UNITS))
    found.add_argument("--tables", type=Path, default=OUT)
    found.add_argument("--truth-run", type=Path, default=None)
    return found


class Stop(Exception):
    """The command ends before step 1 with `code` after printing `text`."""

    def __init__(self, code: int, text: str):
        super().__init__(text)
        self.code = code
        self.text = text


@dataclass
class Inputs:
    """What the checks before step 1 give: the checked arguments and the loaded tables."""

    units: tuple
    matrix: object
    configuration: object
    models: dict
    inputs: dict  # `gate_run.check_inputs`
    loaded: dict  # unit -> `UnitDays`, for the labelled units and every model's fit units

    @property
    def labelled(self) -> dict:
        return {name: self.loaded[name] for name in self.units}


def load_inputs(args) -> Inputs:
    """The checks and refusals of the module text, then the tables; raises `Stop`."""
    try:
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if args.splits < 1:
            raise MatrixError("--splits must be at least 1")
        units = learn_tune.check_units(args.units)
        matrix = load(args.matrix)
        configuration = find_configuration(matrix, args.dial)
        models = check_models(args.model, matrix, units)
    except (MatrixError, learn_tune.NotTraining, learn_run.FitOnScored) as problem:
        raise Stop(2, f"COULD NOT RUN: {problem}") from problem
    except learn_run.Refused as problem:
        raise Stop(1, f"REFUSED: {problem}") from problem
    needed = tuple(dict.fromkeys(u for unit in units for u in (unit, *models[unit][1])))
    try:
        inputs = gate_run.check_inputs(args.truth_run, args.tables, needed)
    except CouldNotRun as problem:
        raise Stop(2, f"COULD NOT RUN: {problem}") from problem
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
        raise Stop(1, f"FAILED: a day could not be built from the tables: {problem}") from problem
    return Inputs(units, matrix, configuration, models, inputs, loaded)


def main(argv: list[str] | None = None) -> int:
    args = parser(__doc__.splitlines()[0]).parse_args(argv)
    try:
        found = load_inputs(args)
    except Stop as stop:
        print(stop.text)
        return stop.code
    document, day_rows, timing = run(
        args.model,
        found.models,
        found.matrix,
        found.configuration,
        found.labelled,
        found.loaded,
        splits=args.splits,
        workers=args.workers,
    )
    document["inputs"] = found.inputs
    document["matrix"] = {
        "file": args.matrix.name,
        "sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
    }
    text = "\n".join(size_whatif_summary.summary_lines(document)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / DAYS_NAME).write_text(days_csv(day_rows))
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / TIMING_NAME).write_text(_json(timing))
    return 1 if document["failed_units"] else 0


if __name__ == "__main__":
    sys.exit(main())
