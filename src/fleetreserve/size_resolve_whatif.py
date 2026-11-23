"""What sizing each day on its own would give on the one-way re-solve: a measurement on the
training units (plan section 7, day-size net and T7; section 6 for the calibration). The same
what-if as `size_whatif`, scored on the method the gate and the comparisons read (the one-way
re-solve, `dial_choice.RUNG`, with the day-level losses of its audited paths) instead of the fixed
plan's every-request losses.

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.size_resolve_whatif --model <name> --dial <epsilon> [--matrix <file>]
        [--splits N] [--workers N] [--units <unit> ...] [--tables <folder>] [--truth-run <folder>]
        [--control-draws N] [--development-units <unit> ...]

It is a what-if on training units (and, with `--development-units`, on development units scored
with sizings fitted on the training units only), not a run of the plan; the results' `label` and
the summary's first line say so. It forms no verdict: it reports the figures.

Before step 1, `--control-draws` (default `CONTROL_DRAWS`) is checked to be at least 1 and every
unit of `--development-units` to be a development unit named once (`check_development`; else exit
2, "COULD NOT RUN", no file written, no table opened), then the arguments, units, matrix, dial and
model are checked and the tables loaded by
`size_whatif.load_inputs`, with the same refusals and exit statuses as `size_whatif`, then those
of the development units (`load_development`, below); then the
context files of the day-size net are read (`size_net_inputs.load_context`; missing or unreadable:
exit 2, "COULD NOT RUN", no file written). Steps 1 to 4 are those of `size_whatif`
(`size_whatif.label_units` with the net and conformal training): every counting day prepared,
labelled and given its context row; the leave-one-unit-out boosters, day-size net (`size_net`)
and conformal training (`size_ct`), on the same labelled days; the sizings `constant`,
`gbm-<level>-<arm>`, `net-<level>-<arm>`, `ct-<arm>` and `oracle-<arm>`; the splits of
`dial_choice`. Then, per unit that the matrix's rule splits:

5. Evaluations (`gate_day.evaluate` through `gate_run._evaluate_task`, in a pool of `--workers`
   processes, every re-solve with its time limit): every offering day at every scale a sizing
   needs (`size_resolve.wanted`), each (day, scale) once; the one-way rung's day-level slot and
   driver losses are kept. A re-solve that times out leaves the rung on its predecessor's plan
   (`rungs`) and the day enters with the losses that replay gives; the re-solves are counted by
   outcome under `evaluations`. A day whose certificate timed out does not offer (step 2).
6. Two sizings read from those losses at size 1 (`size_resolve_sizings`): `oracle-resolve-<arm>`,
   the largest grid scale up to which the day misses no request on any audited path (a
   reference: the day's own outcome), and `day-type-<arm>`, the per-day-type multiplier fitted
   on each split's calibration days; the scales they need are evaluated as in step 5.
7. Per sizing, split and tier, `size_resolve.calibrate` on the calibration days and
   `size_resolve.score` on the test days: the scalar on the gate's grid, the offer per counting
   test day, the test losses.
8. Shuffled controls (`measure_controls`, after the unit's sizings are measured): for every
   sizing `net-<level>-<arm>`, `ct-<arm>` and `day-type-<arm>` of the unit, its control
   `shuffled-<name>`, once per draw (`control_sizings`): the same relative sizes moved across the
   unit's labelled days (those that offer under `constant`) by the unit's permutation for that
   draw (`control_order`), the same on every split and tier (`shuffled`); the scales they need
   are evaluated as in step 5 and counted apart (`control_evaluations`), so the evaluations of
   step 5 are those without the controls; each draw is then measured as in step 7. A failure in
   this step is recorded under `failed_controls` and leaves the unit's sizings and evaluations
   as measured; the unit's sizings and its pools then carry no difference from a control.

Development units (`--development-units`, default none, each one of
`truth_build.DEVELOPMENT_UNITS`, so a held-out unit's name is refused as every other name outside
that list): scored with the finished sizings, fitted on the training units only and on nothing of
a development unit (the owner's decision of 16 November 2026: conformal training is scored on
development units, fitted on training units only). `load_development`: per development unit u the
model is built and checked as `size_whatif.check_models` does for a training unit (u its only gate
unit; the same refusals and exit statuses; with `lgbm_tail_rotation` it is fitted on every
training unit), after the training units' tables are loaded and before any development unit's
table is opened; the truth tables of the development units (and of any fit unit not loaded
before) are checked by `gate_run.check_inputs`, against the same truth build as the training
units' (else exit 2), and listed in the results' `inputs` beside theirs; a day that cannot be
built is exit 1. `label_development`, after the training units' steps 1 to 4 and before any
evaluation: steps 1 and 2 for each development unit as for a training unit (`size_whatif`:
prepared, labelled, given its context row; split by the matrix's rule); then, in place of step
3's leave-one-unit-out fits, one fit of each predictor on every labelled day of the training
units (the rows the training units' own predictors are fitted on): the boosters and the constant
(`size_gbm.booster_predictions`, `size_gbm.constant`) per level, the day-size net per level
(`size_net.fit_days`, `size_net.predict_days`) and conformal training per arm (`size_ct.fit_start`,
`size_ct.fit_days`, `size_ct.predict_days`), each on the cars of the training units' labelled
days and read at the development units' labelled days with their own cars; step 4's sizings from
those predictions. A failure in these fits fails every development unit (recorded under
`failed_units`) and leaves the training units as they are. Steps 5 to 8 then run for each
development unit as for a training unit, after the training units. The training units' figures,
fits, controls and pools are those of the same run without development units; the development
units are in no pool.

`size_resolve_summary` builds the results and the summary: per sizing the offer, its paired
difference from `constant` on the same splits with its spread, and that difference as a share of
`constant`'s offer; for a sizing with a shuffled control its difference from the control's mean
over the draws, with the spread of that difference over splits and, apart, its spread over draws
(`difference_from_shuffled`, `size_resolve_summary.control_figures`). On the gate's grid any
spread of sizes, even one unrelated to the days, can gain over `constant`
(`size_resolve_summary.definitions`, "shuffled_control"), so a learned sizing's gain is read
against its control as well, and a gain beyond chance against the control's spread over draws.

Readings of the controls, each with the alternative not taken:

SC1. The permutation is within the unit, over its labelled days, one per unit and draw, used on
    every split and tier and by every control of the unit in that draw. [Alternative: a new
    permutation per split, which gives every split its own (day, scale) pairs to evaluate, up to
    `--splits` times as many re-solves per draw.]
SC2. A day that does not offer under the learned sizing (arm abstain) carries its abstention with
    it: the control abstains on as many labelled days, moved. [Alternative: moving sizes only
    among the days that offer under the sizing, which keeps which days abstain, so any skill in
    choosing them would stay in the control.]
SC3. Controlled: the day-size net, conformal training and the per-day-type multiplier. The boosters
    are left out to hold the evaluations down, the oracles are references, and `constant` is its
    own control. [Alternative: a control for every `gbm-<level>-<arm>` too, ten more sets of
    evaluations.]
SC4. `--control-draws` draws per unit, by default `CONTROL_DRAWS` = 19, draw k's permutation from
    `CONTROL_SEED`, k and the unit's name. Chance: a sizing whose sizes carry nothing about the
    days is, given its sizes, one more random arrangement of them, so it beats every one of n
    draws with a chance of at most 1/(n+1), at most 5% from 19 draws (the fewest such n; the
    gains by draw, `size_resolve_summary.by_draw_figures`, show it), and it falls inside the p5
    to p95 band of its draws' offers (0 inside the spread over draws) with a chance of about
    0.62 for 5 draws, 0.76 for 10, 0.82 for 20 and 0.86 for 40 (a simulation of exchangeable
    values, numpy's linear interpolation), and 0 for one draw, whose band is one point. Cost: the
    controls do not only re-use (day, scale) pairs already evaluated; a day given another day's
    size needs its re-solve at that size's scales, so a draw needs at most 10 new pairs (the
    grid points up to 1) per labelled day for each set of distinct moved sizes (the scale and
    abstain arms of one sizing share most of theirs), fewer where sizes repeat (a net kept at
    its starting constant moves the same size everywhere, and the day-type sizes take few
    values, so their pairs saturate as draws grow); the cost is about linear in the draws for
    the net and conformal training, so 19 draws cost about twice the 10 of the default before;
    the pairs and seconds a run's controls took are given under `control_evaluations` and in the
    timing file under `controls`. [Alternative: 10 draws, the default before, where beating every
    draw has a chance of up to 1/11 = 9%; one draw, whose difference can be far from 0 for a
    noise sizing on a small unit and whose spread over splits says nothing of another draw; or
    more draws, at their cost.] With
    `constant`, steps 5 to 7 are `dial_choice`'s measurement and `dial_splits.score_split` on the
    same days.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set, the summary also on standard output:
- `size-resolve-whatif-days.csv`: one row per labelled day (`DAY_COLUMNS`: those of
  `size_whatif.DAY_COLUMNS`, then the net's predictions `size_whatif.NET_DAY_COLUMNS`, then the
  conformal training's outputs `size_whatif.CT_DAY_COLUMNS`), the development units' rows after
  the training units';
- `size-resolve-whatif.json`: every number, by `size_resolve_summary.results` (with
  `failed_controls`, step 8); under `run`, `control_draws`; under `control_evaluations`, per
  unit, the (day, scale) pairs and one-way re-solves the shuffled controls added (step 8); under
  `net_inputs` the net's input columns and the context files read (`size_net_inputs.record`);
  under `net_fits`, per scored unit and level, the record of the network that predicts it
  (`size_net.Fitted.record`: its days, the epoch kept, 0 being the starting constant, and its
  validation losses); under `ct_fits`, per scored unit and arm, the record of the conformal
  training that predicts it (`size_ct.Fitted.record`: its days, the epoch kept, 0 being the
  day-size net it starts from, its validation offers and steps); with development units, under
  `run` their names (`DEVELOPMENT`), under `units` their entries, and under `development_fits`
  the records of the finished fits that predict them (`net` per level, `ct` per arm);
- `size-resolve-whatif-summary.txt`: the summary text (`size_resolve_summary.summary_lines`);
- `size-resolve-whatif-timing.json`: wall-clock seconds, kept apart from the other files (under
  `evaluations` per unit steps 5 to 8, under `controls` step 8).

Exit status: 0 every unit ran; 1 the model was refused or failed, a day could not be built from
the tables, a unit failed (in its labelling, its finished sizings or its evaluations) or a unit's
controls failed (its other figures kept); 2 could not run (as `size_whatif`, `--control-draws`
below 1, a unit of `--development-units` that is not a development unit or is named twice, or a
development unit's tables of another truth build).
"""

import hashlib
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve import (
    dial_choice,
    gate_run,
    learn_run,
    size_ct,
    size_gbm,
    size_net,
    size_net_inputs,
    size_resolve,
    size_resolve_sizings,
    size_resolve_summary,
    size_scale,
    size_whatif,
    t7_stat,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import MatrixError
from fleetreserve.truth_build import DEVELOPMENT_UNITS
from fleetreserve.truth_verify import CouldNotRun

DAYS_NAME = "size-resolve-whatif-days.csv"
DAY_COLUMNS = (*size_whatif.DAY_COLUMNS, *size_whatif.NET_DAY_COLUMNS, *size_whatif.CT_DAY_COLUMNS)
RESULTS_NAME = "size-resolve-whatif.json"
SUMMARY_NAME = "size-resolve-whatif-summary.txt"
TIMING_NAME = "size-resolve-whatif-timing.json"
RUNG = dial_choice.RUNG
CHUNK = 2000  # (day, scale) pairs sent to the pool at once
TYPE_SIZING = "day-type"
ORACLE_SIZING = "oracle-resolve"
ARMS = size_scale.ARMS
CONTROL = size_resolve_summary.CONTROL
CONTROLLED = (f"{size_whatif.NET}-", f"{size_whatif.CT}-")  # fixed sizings with a control
CONTROL_SEED = 0
CONTROL_DRAWS = 19  # the default of `--control-draws`: reading SC4
DEVELOPMENT = "development_units"  # the run's field naming the development units scored
NET, CT = size_whatif.NET, size_whatif.CT


class Evaluator:
    """The one-way rung's losses of one unit's days by (day, scale), each pair evaluated once."""

    def __init__(self, runner, prepared):
        self.runner = runner
        self.prepared = prepared
        self.losses: dict = {}
        self.outcomes: Counter = Counter()

    def add(self, pairs) -> None:
        todo = sorted(set(pairs) - set(self.losses))
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start : start + CHUNK]
            jobs = [(self.prepared[d], s) for d, s in chunk]
            results = self.runner.map(gate_run._evaluate_task, jobs)
            for key, found in zip(chunk, results, strict=True):
                rung = found.rungs[RUNG]
                self.losses[key] = (float(rung.slot_loss), float(rung.driver_loss))
                self.outcomes.update(rung.resolves)

    def counts(self) -> dict:
        return {
            "day_scale_pairs": len(self.losses),
            "re_solves_of_the_one_way_rung": dict(sorted(self.outcomes.items())),
        }


def raw_types(data, frame) -> list:
    """Per counting day its raw day type (`t7_stat.raw_type`), None on a day that does not offer."""
    found = [None] * data.counting_days
    rows = frame[frame["unit"] == data.name]
    for day, date, holiday in zip(rows["day"], rows["date"], rows["public_holiday"], strict=True):
        found[int(day)] = t7_stat.raw_type(date, bool(holiday))
    return found


def fixed(r):
    return lambda seed, tier: r


def unit_sizings(data, q_all: dict, evaluator: Evaluator, types) -> tuple[dict, dict]:
    """Steps 5 and 6 for one unit: (sizing -> r_of(seed, tier), the multiplier's scale per type
    on every split and tier)."""
    offering = ~np.isnan(q_all["constant"])
    r_of = {name: fixed(size_resolve.relative(q)) for name, q in q_all.items()}
    if not data.splits:  # not scored (`size_whatif`, step 5): nothing to evaluate
        for name in (ORACLE_SIZING, TYPE_SIZING):
            for arm in ARMS:
                r_of[f"{name}-{arm}"] = None
        return r_of, {}
    evaluator.add(pair for f in r_of.values() for pair in size_resolve.wanted(f(None, None)))
    raw = size_resolve_sizings.failure_free_sizes(evaluator.losses, offering)
    oracle = {arm: size_resolve.relative(size_scale.sizes(raw, offering, arm)) for arm in ARMS}
    per_split, scales = {}, {tier: {} for tier in size_scale.TIERS}
    for seed, calibration, _ in data.splits:
        for tier in size_scale.TIERS:
            raw, found = size_resolve_sizings.type_sizes(
                evaluator.losses, types, offering, calibration, tier
            )
            for name, scale in found.items():
                scales[tier].setdefault(name, []).append(scale)
            for arm in ARMS:
                per_split[(arm, seed, tier)] = size_resolve.relative(
                    size_scale.sizes(raw, offering, arm)
                )
    evaluator.add(
        pair for r in (*oracle.values(), *per_split.values()) for pair in size_resolve.wanted(r)
    )
    for arm in ARMS:
        r_of[f"{ORACLE_SIZING}-{arm}"] = fixed(oracle[arm])
    for arm in ARMS:
        r_of[f"{TYPE_SIZING}-{arm}"] = lambda seed, tier, arm=arm: per_split[(arm, seed, tier)]
    return r_of, scales


def control_order(unit: str, n: int, draw: int) -> np.ndarray:
    """The unit's permutation of its `n` labelled days for the draw `draw` (readings SC1, SC4)."""
    return np.random.default_rng([CONTROL_SEED, draw, *unit.encode()]).permutation(n)


def shuffled(r, days, order) -> np.ndarray:
    """`r` with the values of the days `days` moved by `order`: day `days[k]` takes the value of
    day `days[order[k]]`; every other day keeps its own (NaN on a day that does not offer)."""
    r = np.asarray(r, dtype=float)
    out = r.copy()
    out[days] = r[days][order]
    return out


def control_sizings(data, q_all: dict, r_of: dict, evaluator, draws: int) -> dict:
    """Step 8 for one unit: `shuffled-<name>` -> per draw, r_of(seed, tier) (None when the
    sizing's is), the scales they need evaluated."""
    days = np.flatnonzero(~np.isnan(q_all["constant"]))
    orders = [control_order(data.name, len(days), k) for k in range(draws)]
    out, needed = {}, []
    for name, q in q_all.items():
        if name.startswith(CONTROLLED):
            r = size_resolve.relative(q)
            moved = [shuffled(r, days, order) for order in orders]
            out[f"{CONTROL}-{name}"] = [fixed(m) for m in moved]
            needed += moved
    for arm in ARMS:
        name = f"{TYPE_SIZING}-{arm}"
        f = r_of[name]
        if f is None:
            out[f"{CONTROL}-{name}"] = [None] * draws
            continue
        out[f"{CONTROL}-{name}"] = [
            lambda seed, tier, f=f, order=order: shuffled(f(seed, tier), days, order)
            for order in orders
        ]
        needed += [
            g(seed, tier)
            for g in out[f"{CONTROL}-{name}"]
            for seed, _, _ in data.splits
            for tier in size_scale.TIERS
        ]
    if data.splits:
        distinct = {r.tobytes(): r for r in needed}.values()  # day-type's sizes repeat
        evaluator.add(pair for r in distinct for pair in size_resolve.wanted(r))
    return out


def added(before: dict, after: dict) -> dict:
    """The evaluation counts (`Evaluator.counts`) from `before` to `after`."""
    outcomes = Counter(after["re_solves_of_the_one_way_rung"])
    outcomes.subtract(before["re_solves_of_the_one_way_rung"])
    return {
        "day_scale_pairs": after["day_scale_pairs"] - before["day_scale_pairs"],
        "re_solves_of_the_one_way_rung": {k: v for k, v in sorted(outcomes.items()) if v},
    }


def measure_controls(data, q_all, r_of, evaluator, value, draws) -> dict:
    """Step 8 for one unit, with each draw measured: `shuffled-<name>` -> per draw, what
    `size_resolve.measure` gives."""
    found = control_sizings(data, q_all, r_of, evaluator, draws)
    return {
        name: [size_resolve.measure(evaluator.losses, value, f, data.splits) for f in fs]
        for name, fs in found.items()
    }


@dataclass
class Scores:
    """Steps 5 to 8 of every unit scored, each by unit (module text)."""

    failed: dict
    measured: dict = field(default_factory=dict)
    evaluations: dict = field(default_factory=dict)
    type_scales: dict = field(default_factory=dict)
    controls: dict = field(default_factory=dict)
    counted: dict = field(default_factory=dict)
    failed_controls: dict = field(default_factory=dict)


def score_units(labelled, runner, scores: Scores, timing: dict, control_draws: int) -> None:
    """Steps 5 to 8 for every unit of `labelled` (a `size_whatif.Labelled`), into `scores`."""
    for unit, data in labelled.data.items():
        begun = time.perf_counter()
        prepared = labelled.prepared[unit]
        value = np.array([p.value for p in prepared], dtype=float)
        evaluator = Evaluator(runner, prepared)
        q_all = labelled.q_all[unit]
        try:
            r_of, scores.type_scales[unit] = unit_sizings(
                data, q_all, evaluator, raw_types(data, labelled.frame)
            )
            scores.measured[unit] = {
                sizing: size_resolve.measure(evaluator.losses, value, f, data.splits)
                for sizing, f in r_of.items()
            }
        except Exception as problem:  # recorded, the other units go on
            scores.failed[unit] = f"{type(problem).__name__}: {problem}"
            scores.type_scales.pop(unit, None)
            continue
        scores.evaluations[unit] = evaluator.counts()
        started_controls = time.perf_counter()
        try:
            scores.controls[unit] = measure_controls(
                data, q_all, r_of, evaluator, value, control_draws
            )
        except Exception as problem:  # recorded, the unit's sizings are kept
            scores.failed_controls[unit] = f"{type(problem).__name__}: {problem}"
        scores.counted[unit] = added(scores.evaluations[unit], evaluator.counts())
        timing.setdefault("controls", {})[unit] = time.perf_counter() - started_controls
        timing.setdefault("evaluations", {})[unit] = time.perf_counter() - begun


@dataclass
class Development:
    """The development units of `--development-units` after the checks before step 1: their
    names, their models (`size_whatif.check_models`) and their tables, with those of any fit unit
    not loaded for the training units."""

    units: tuple
    models: dict
    loaded: dict


def check_development(names) -> tuple:
    """`--development-units`, each a development unit named once (module text); raises
    `size_whatif.Stop` (exit 2) before any table is opened."""
    found = tuple(dict.fromkeys(names))
    twice = [name for name in found if list(names).count(name) > 1]
    if twice:
        raise size_whatif.Stop(
            2, f"COULD NOT RUN: --development-units names {', '.join(twice)} twice"
        )
    other = [name for name in found if name not in DEVELOPMENT_UNITS]
    if other:
        raise size_whatif.Stop(
            2,
            f"COULD NOT RUN: --development-units takes development units only "
            f"({', '.join(DEVELOPMENT_UNITS)}): {', '.join(other)}: not a development unit",
        )
    return found


def load_development(args, found, names: tuple) -> tuple:
    """(the `Development` of `names`, or None when there are none; the inputs record with their
    tables beside the training units'), by the checks of the module text; raises
    `size_whatif.Stop`."""
    if not names:
        return None, found.inputs
    try:
        models = size_whatif.check_models(args.model, found.matrix, names)
    except (MatrixError, learn_run.FitOnScored) as problem:
        raise size_whatif.Stop(2, f"COULD NOT RUN: {problem}") from problem
    except learn_run.Refused as problem:
        raise size_whatif.Stop(1, f"REFUSED: {problem}") from problem
    needed = tuple(
        dict.fromkeys(
            u for unit in names for u in (unit, *models[unit][1]) if u not in found.loaded
        )
    )
    try:
        checked = gate_run.check_inputs(args.truth_run, args.tables, needed)
    except CouldNotRun as problem:
        raise size_whatif.Stop(2, f"COULD NOT RUN: {problem}") from problem
    same = ("setting", "truth_run")
    if any(checked[key] != found.inputs[key] for key in same):
        raise size_whatif.Stop(
            2,
            f"COULD NOT RUN: the development units' tables are of the truth build "
            f"{checked['truth_run']} ({checked['setting']}), the training units' of "
            f"{found.inputs['truth_run']} ({found.inputs['setting']})",
        )
    matrix = found.matrix
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
        raise size_whatif.Stop(
            1, f"FAILED: a day could not be built from the tables: {problem}"
        ) from problem
    tables = {**found.inputs["table_sha256"], **checked["table_sha256"]}
    return Development(names, models, loaded), {**found.inputs, "table_sha256": tables}


def finished_predictions(train, frame, train_cars, cars, fits: dict) -> tuple[dict, dict]:
    """The finished sizings' predictions at the development units' labelled days `frame`
    (module text): (level -> {"gbm", "constant", `NET`: one per row}, arm -> conformal training's
    output per row), each from one fit on the training units' labelled days `train` with their
    cars `train_cars`, read with the development units' cars `cars`; adds the fits' records to
    `fits` (`net` per level, `ct` per arm). NaN everywhere when either set of days is empty."""
    n = len(frame)
    predictions = {
        float(level): {kind: np.full(n, np.nan) for kind in ("gbm", "constant", NET)}
        for level in size_gbm.LEVELS
    }
    ct = {arm: np.full(n, np.nan) for arm in ARMS}
    if not n or train.empty:
        return predictions, ct
    context = list(size_whatif.CONTEXT)
    x, y = train[context].astype(float).to_numpy(), train[size_gbm.LABEL].astype(float).to_numpy()
    x_new = frame[context].astype(float).to_numpy()
    rows = ["unit", "site", "date", size_gbm.LABEL]
    for level, found in predictions.items():
        found["gbm"] = size_gbm.booster_predictions(x, y, x_new, context, level)
        found["constant"] = np.full(n, size_gbm.constant(y, level))
        fitted = size_net.fit_days(train[rows], train_cars, (level,))
        found[NET] = size_net.predict_days(fitted, frame[rows], cars)[:, 0]
        fits["net"][f"{level:g}"] = fitted.record
    rows = [*rows, size_ct.VALUE]
    start = size_ct.fit_start(train[rows], train_cars)
    for arm in ARMS:
        fitted = size_ct.fit_days(train[rows], train_cars, arm, None, start)
        ct[arm] = size_ct.predict_days(fitted, frame[rows], cars)
        fits["ct"][arm] = fitted.record
    return predictions, ct


def label_development(
    development: Development, found, matrix, configuration, loaded, *, splits, workers, sources
) -> tuple:
    """Steps 1 to 4 for the development units with the finished sizings (module text): (a
    `size_whatif.Labelled` of them, the finished fits' records). `found`: the training units'
    `size_whatif.Labelled`; `loaded`: unit -> `UnitDays`, the training units', the development
    units' and their fit units'."""
    timing = {"units": {}}
    seeds = list(range(1, splits + 1))
    data, prepared_by_unit, failed = {}, {}, {}
    for unit in development.units:
        begun = time.perf_counter()
        model, fit_units = development.models[unit]
        try:
            prepared = size_whatif.prepare_unit(
                unit, model, fit_units, matrix, configuration, loaded, loaded, workers
            )
            rows, terms, reasons, timeouts = size_whatif.label_days(prepared)
        except Exception as problem:  # recorded, the other units go on
            failed[unit] = f"{type(problem).__name__}: {problem}"
            continue
        prepared_by_unit[unit] = prepared
        plan = gate_run.plan_unit(unit, loaded[unit], matrix)
        data[unit] = size_whatif.UnitData(
            unit,
            size_whatif.site_type(loaded[unit]),
            len(prepared),
            plan.status,
            fit_units,
            rows,
            terms,
            reasons,
            timeouts,
            dial_choice.split_indices(unit, loaded[unit], matrix, seeds)
            if plan.status == "calibratable"
            else [],
        )
        timing["units"][unit] = time.perf_counter() - begun
    frame = size_whatif.with_calendar(
        pd.DataFrame(
            [row for d in data.values() for row in d.rows], columns=list(size_whatif.ROW_COLUMNS)
        )
    )
    fits = {"net": {}, "ct": {}}
    begun = time.perf_counter()
    try:
        train_cars = size_whatif.net_cars(found.data, loaded, sources)
        cars = size_whatif.net_cars(data, loaded, sources)
        predictions, ct = finished_predictions(found.frame, frame, train_cars, cars, fits)
    except Exception as problem:  # every development unit fails, the training units go on
        why = f"the finished sizings were not fitted: {type(problem).__name__}: {problem}"
        failed.update(dict.fromkeys(data, why))
        empty = size_whatif.Labelled({}, {}, frame.iloc[0:0], {}, {}, failed, timing)
        return empty, {"net": {}, "ct": {}}
    timing["fit_seconds"] = time.perf_counter() - begun
    for level, found_level in predictions.items():
        frame[f"gbm_q{level:g}"] = found_level["gbm"]
        frame[f"{NET}_q{level:g}"] = found_level[NET]
    for arm in ARMS:
        frame[f"{CT}_{arm}"] = ct[arm]
    q_all, per_unit = {}, {}
    for unit, d in data.items():
        rows = (frame["unit"] == unit).to_numpy(bool)
        per_unit[unit] = {
            level: {kind: values[rows] for kind, values in by_kind.items()}
            for level, by_kind in predictions.items()
        }
        q_all[unit] = size_whatif.sizings(
            d,
            {level: f["gbm"] for level, f in per_unit[unit].items()},
            {level: f[NET] for level, f in per_unit[unit].items()},
            {arm: ct[arm][rows] for arm in ARMS},
        )
    labelled = size_whatif.Labelled(data, prepared_by_unit, frame, per_unit, q_all, failed, timing)
    return labelled, fits


def run(
    name,
    models,
    matrix,
    configuration,
    units,
    loaded,
    *,
    splits,
    workers,
    context,
    control_draws=CONTROL_DRAWS,
    development=None,
):
    """Steps 1 to 8 for every unit: (results document, day rows, timing). `context`: the net's
    context files (`size_net_inputs.load_context`); `control_draws`: the draws of each shuffled
    control (reading SC4); `development`: the development units to score (a `Development`), or
    None."""
    started = time.perf_counter()
    found = size_whatif.label_units(
        models,
        matrix,
        configuration,
        units,
        loaded,
        splits=splits,
        workers=workers,
        net_sources=context.sources,
        conformal=True,
    )
    timing = found.timing
    developed = fits = None
    if development is not None:
        developed, fits = label_development(
            development,
            found,
            matrix,
            configuration,
            {**loaded, **development.loaded},
            splits=splits,
            workers=workers,
            sources=context.sources,
        )
        timing["development"] = developed.timing
    scores = Scores(dict(found.failed))
    runner = gate_run.Runner(workers, {}, matrix)
    try:
        score_units(found, runner, scores, timing, control_draws)
        if developed is not None:
            scores.failed.update(developed.failed)
            score_units(developed, runner, scores, timing, control_draws)
    finally:
        runner.close()
    scored = {u: d for u, d in found.data.items() if u in scores.measured}
    predictions = found.per_unit
    developed_scored = None
    if developed is not None:
        developed_scored = {u: d for u, d in developed.data.items() if u in scores.measured}
        predictions = {**found.per_unit, **developed.per_unit}
    document = size_resolve_summary.results(
        scored,
        predictions,
        scores.measured,
        scores.evaluations,
        scores.type_scales,
        scores.failed,
        scores.controls,
        scores.failed_controls,
        development=developed_scored,
    )
    document["run"] = {
        **size_whatif.run_fields(name, matrix, configuration, units, splits),
        "rung": RUNG,
        "grid": list(size_resolve.GRID),
        "control_draws": control_draws,
    }
    document["control_evaluations"] = scores.counted
    document["net_inputs"] = size_net_inputs.record(context)
    document["net_fits"] = found.net_fits
    document["ct_fits"] = found.ct_fits
    day_rows = found.frame[list(DAY_COLUMNS)].to_dict("records")
    if development is not None:
        document["run"][DEVELOPMENT] = list(development.units)
        document["development_fits"] = fits
        if not developed.frame.empty:
            day_rows += developed.frame[list(DAY_COLUMNS)].to_dict("records")
    timing["total_seconds"] = time.perf_counter() - started
    return document, day_rows, timing


def main(argv: list[str] | None = None) -> int:
    parser = size_whatif.parser(__doc__.splitlines()[0])
    parser.add_argument("--control-draws", type=int, default=CONTROL_DRAWS)
    parser.add_argument("--development-units", nargs="+", default=[])
    args = parser.parse_args(argv)
    if args.control_draws < 1:
        print("COULD NOT RUN: --control-draws must be at least 1")
        return 2
    try:
        names = check_development(args.development_units)
        found = size_whatif.load_inputs(args)
        development, inputs = load_development(args, found, names)
    except size_whatif.Stop as stop:
        print(stop.text)
        return stop.code
    try:
        context = size_net_inputs.load_context()
    except MatrixError as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    document, day_rows, timing = run(
        args.model,
        found.models,
        found.matrix,
        found.configuration,
        found.labelled,
        found.loaded,
        splits=args.splits,
        workers=args.workers,
        context=context,
        control_draws=args.control_draws,
        development=development,
    )
    document["inputs"] = inputs
    document["matrix"] = {
        "file": args.matrix.name,
        "sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
    }
    text = "\n".join(size_resolve_summary.summary_lines(document)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / DAYS_NAME).write_text(size_whatif.days_csv(day_rows, DAY_COLUMNS))
        (folder / RESULTS_NAME).write_text(size_whatif._json(document))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / TIMING_NAME).write_text(size_whatif._json(timing))
    return 1 if document["failed_units"] or document["failed_controls"] else 0


if __name__ == "__main__":
    sys.exit(main())
