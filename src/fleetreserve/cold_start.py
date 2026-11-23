"""The cold start of outcome O9 on one list of units (plan section 11: "Cold start, k=0, 7, 14, 28,
56 local days in time order: availability A_k, offer share Y_k, against pooled LightGBM").

Command line, from the project's top folder (under `fleetreserve.run_record`):
    python -m fleetreserve.cold_start --model <name> --dial <epsilon> --units <unit> ...
        [--reference <name>] [--matrix <file>] [--workers N] [--tables <folder>]
        [--truth-run <folder>] [--held-out-tables <folder>] [--held-out-truth-run <folder>]

For each unit named, each of the two models and each k of `cold_start_days.LOCAL_DAYS`: the model
forecasts the unit's local days and scored days from what it may know at k
(`cold_start_days.frame_at`), the one-way re-solve offer is calibrated on the local days and scored
on the scored days (`cold_start_score`), and A_k and Y_k are reported per tier.

Readings (each the one closest to the plan's words; the alternative is named and not taken):

- C1, the models. The cold-started model is `--model`; for the plan's O9 it is section 7's main
  model (`lgbm_tail`, the owner's decision of 16 November 2026, so it is an argument). The
  reference is `--reference`, by default `lgbm`, section 7's "pooled LightGBM", cold-started by the
  same protocol on the same days, and both are reported side by side ("against pooled LightGBM").
  Not taken: cold-starting pooled LightGBM alone and comparing it with its own figures on every
  day; taking `lgbm_tail` (the model the WP6 gate is read with) as the reference; giving Y_k as the
  main model's offer over the reference's offer (the plan's other shares are shares of the
  ceiling; the two offers are both given, so that ratio can be read off).
- C2, what the k local days feed. They feed the calibration and the inputs that read the unit's
  own past (the user histories of `learn_features`): on a local day as the local rows give them
  at that day's own τ, on every scored day from the local rows known at the end of the local
  period alone (C2a), the same for every scored day. The model is fitted on the training units
  alone, once, the same fit at every k (plan section 7: "Trained on training sites"), and never on
  a row of a scored unit (`learn_run.check_model`). Not taken: refitting the model with the local
  days, or a site term learned from them; a history that grows with every day after k, so that k
  changes only the calibration.
- C2a, the end of the local period is the latest decision time τ of the k local days (for a unit
  of one site, the k-th counting day's τ; `cold_start_days.local_end`). A scored day's history
  holds only the sessions of the local rows that had plugged out by then
  (`cold_start_days.known_rows`), so nothing that happened after it reaches a row's features at
  k, apart from each scored day's own rows as known at its own τ. The plan says only "k local
  days in time order"; this reads them as what is known when the last of them is forecast. Its
  consequence: no session of the k-th counting day's fleet (each is plugged in at that day's τ),
  and no longer stay of an earlier local day that had not ended by then, is in a scored day's
  history; at k = 1 that history holds no session of a counting day. Not taken: the end of the
  k-th local calendar day (local midnight), which would also count the k-th day's sessions that
  left that day after its τ; and the local sessions whenever they left, under which a session
  listed on a local day and still plugged in after it carries its later plug-out, an outcome of a
  later day, into the history of every scored day after that plug-out.
- C3, k counts counting days (the days `days_from_truth` keeps), not calendar days. Not taken:
  calendar days from the unit's first day.
- C4, the local rows are every row of the unit's cars table at or before the k-th counting day,
  including days that do not count. Not taken: the rows of the k counting days only.
- C5, time order is by date, then site; a pooled unit counts its site-days in that one order. Not
  taken: k days per site of a pooled unit.
- C6, the scored days are the same for every k: the counting days after the first 56 (the largest
  k), so the five figures of a unit are read on the same days. A unit without a counting day after
  them gets no figures. Not taken: each k scoring every day after its own first k.
- C7, below the fewest offering days the bound needs (T6: abstain unless n >= 1/alpha - 1, so 9
  offering days for the slot loss at alpha 0.10 and 19 for the driver loss at 0.05), the tier
  abstains: the offer is 0 and A_k is not defined. So k = 0 and k = 7 offer 0 in both tiers and
  k = 14 offers 0 in tier A whatever the model forecasts, and those figures say only that. Not
  taken: a scale carried over from other units' days (the plan has no such rule).
- C8, the calibration is the on-grid route of `dial_choice` (`dial_splits.on_grid_scale` on the
  scales of `dial_splits.GRID`) on the one-way rung (`dial_choice.RUNG`, O1's audited-set re-solve
  offer), both tiers, never pooled; a scored day is evaluated only at each tier's chosen scale.
  Not taken: the bisection of `gate_calibrate` to a tolerance of 0.001.
- C9, A_k and Y_k are as `cold_start_score` gives them: 1 minus the mean slot loss over the
  offering scored days, and the offer per counting scored day over the mean hindsight ceiling per
  counting scored day, the best-causal ceiling beside it.
- C10, the configuration is the dial family's member at `--dial` (`compare_run.dial_configurations`
  on the matrix): one run per site type, since the dial is fixed per site type (plan section 6).
  The command does not check that the units named share a site type.
- C11, held-out units are scored only through the gate (`cold_start_gate.require_gate`, the
  held-out loader's checks 2 to 8) and only from tables of a recorded held-out truth build
  (`--held-out-truth-run`) checked against it at the commit checked out
  (`cold_start_gate.read_two_builds`, each build as `gate_run.check_inputs` checks it);
  development and held-out units are never in one run. Not taken: scoring held-out tables found
  without a recorded build. Every check of this route is the shared one of `sealed_route`, which
  the cold start calls by the names of `cold_start_gate`. A held-out run's results hold `gate`
  (the tag's commit and the fix commit) and, as every results file of a run through the gate
  does, the key `sealed_route` (`gate_summary.ROUTE_KEY`: the kind, the label and the same two
  commits), so that `run_evidence` never reads the run from an earlier commit (its point 4,
  reading E3). A development run has neither key.
- C12, a model that reads other days' outcomes (`forecast_model.OUTCOME_READERS`: the empirical
  forecaster) is refused, and so is a look-ahead model (`learn_run.check_model`): the k local days
  reach a model only through its inputs. Not taken: the empirical forecaster on the local rows.
- C13, not computed: the oracle-departure partner, tier B's stay-to-bound check, and the 95%
  bootstrap intervals of plan section 11 (the days file holds the per-day values they would be
  drawn from).
- C14, the truth of the local days is the truth build's, not rebuilt at k. The losses the
  calibration reads and the values certified on the local days come from tables set from the
  unit's days as a whole, not from its first k: the decision time and service hours of a Norway
  site come from the occupancy of all its days (`truth_build`), those of a held-out Norway unit or
  Korean portfolio from a window search over its days (the held-out truth build), and the main rate
  rule takes a user's highest average power over all sessions at the site, later ones included
  (`learn_features`, `rate_kw`). A local day's losses also need its own outcomes, so the losses of
  the last local days hold departures after the end of the local period (C2a), some on a scored
  day (at k = 56, a car of the 56th counting day that leaves after the first scored day's τ). So
  the calibration and the certified values at k use facts from after the k local days; C2a's rule
  is about features, and these are no features. Not taken: a truth build per k from the first k
  days only; leaving out of the calibration a local day whose outcome ends after the end of the
  local period.
- C15, the levels fitted. Each model forecasts at the levels up to the configuration's kappa
  (`driver_risk.levels_up_to`, the levels the driver-risk rows read). It is built from the matrix
  by its `make_model`, which takes the matrix's levels by the shared rule
  (`driver_risk.matrix_levels`: each configuration's kappa and, under a driver dial, every level up
  to it, as `learn_run` predicts at), so a matrix that lists the chosen kappa only (as the
  rehearsal and sealed matrices of the run list do) builds a model with a forecast at each of
  those levels. The fit frames are unchanged. Not taken: refusing a matrix that does not list
  every level up to kappa.

Order of the checks: each path argument through `gate_run.argument_path` (a path written as a URL
could not run; a leading `~` is expanded); the units (`cold_start_gate.check_units`); the folders
the run reads (`cold_start_gate.check_folders`, folders as given: under the top folder's `data/`
only `data/derived/` is read, so a folder named there by a slip, the sealed folder among them, is
refused; what these checks stop and do not stop is in `sealed_route`, module text, first
paragraph; `--tables` also out of the held-out build's folders; each file inside them is checked
by `gate_run.check_inputs` before any is opened, in any working copy's `data/`, as written,
through a link and where its links lead) and, for
held-out units, the gate, before anything is read; then the matrix (its file through
`gate_run.check_files` before it is read, and read once, `gate_config.read`, so the SHA-256 the
results record is that of the matrix the run parsed), the dial and the two models; then, for
held-out units, the gate again (so the model modules loaded since are the checked-out commit's
too), which must pass at the tag's commit and fix commit it passed at first
(`sealed_route.same_gate`), or the run could not run (exit 2); then the
tables of the fit units (and of the development units) against the recorded truth build
(`--tables`, `--truth-run`) and, for held-out units, theirs against the held-out build
(`--held-out-tables`, default `cold_start_gate.held_out_tables(root)` under the top folder the
command is given, `--held-out-truth-run`, required). For held-out units the tables are parsed
from the bytes the two checks hashed (`cold_start_gate.read_two_builds`, then
`days_from_truth.parse_unit_days`, which hashes each table again after the parse and could not
run, exit 2, when one changed), as in `learn_run`; for development units they are read again from
the same paths (`days_from_truth.load_unit_days`, which parses the bytes the plain `open` reads and
hands pandas no path), so each development table is read twice.

Timeouts are recorded outcomes, as in `dial_choice`: a day whose certificate times out abstains; a
re-solve that times out leaves the rung on its predecessor's plan; a missing best-causal ceiling
leaves that share None. Their numbers are in `timeouts`.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary also on
standard output:
- `cold-start.json` (`RESULTS_NAME`): every figure per model, unit, k and tier, the day keys of the
  local and scored days, the readings, the inputs and the hashes of the forecast tables;
- `cold-start-summary.txt` (`SUMMARY_NAME`, `cold_start_summary`);
- `cold-start-days.csv` (`DAYS_NAME`): one row per model, k, unit, tier and local or scored day;
- `cold-start-forecasts-<model>-k<k>.csv`: each forecast table (`forecast_model.write_table`);
- `cold-start-timing.json` (`TIMING_NAME`): wall-clock seconds, kept apart from the results.
A run on development units is labelled a rehearsal (`LABELS`): plan section 11's slots are filled
by the held-out run only.

Exit status: 0 every model, k and unit was scored; 1 a model was refused or failed, or a k or unit
failed; 2 could not run (a bad argument, matrix, dial or model name, a path written as a URL,
units refused, a folder in a held-out place, a file in a refused place, the gate refused, a model
fitted on a scored unit, no unit with a scored day, a table missing, no verified truth build at
this commit, a table differing from it).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve import (
    cold_start_days,
    cold_start_gate,
    cold_start_score,
    cold_start_summary,
    compare_run,
    dial_choice,
    dial_splits,
    driver_risk,
    gate_run,
    learn_run,
    model_inputs,
    sealed_route,
)
from fleetreserve.cold_start_gate import SealRefused
from fleetreserve.days_from_truth import load_unit_days, parse_unit_days
from fleetreserve.forecast_model import TableFactory, check_table, write_table
from fleetreserve.gate_calibrate import DayCache, ceiling_values
from fleetreserve.gate_config import MatrixError, read
from fleetreserve.gate_summary import ROUTE_KEY
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

MATRIX = dial_choice.MATRIX
REFERENCE = "lgbm"  # plan section 7's pooled LightGBM (reading C1)
RUNG = dial_choice.RUNG
RESULTS_NAME = "cold-start.json"
SUMMARY_NAME = "cold-start-summary.txt"
DAYS_NAME = "cold-start-days.csv"
TIMING_NAME = "cold-start-timing.json"
LABELS = {
    cold_start_gate.DEVELOPMENT: (
        "REHEARSAL on development units: not outcome O9, whose slots only the held-out run fills "
        "(plan section 11)"
    ),
    cold_start_gate.HELD_OUT: (
        f"held-out units, through the sealed-data gate {sealed_route.AT_THE_TAG}"
    ),
}
DAY_COLUMNS = (
    "model",
    "k",
    "unit",
    "tier",
    "role",
    "date",
    "site",
    "offering",
    "certificate_outcome",
    "abstain_reason",
    "value_kwh",
    "scale",
    "offer_kwh",
    "slot_loss_at_scale",
    "driver_loss_at_scale",
    "hindsight_kwh",
    "best_causal_kwh",
)


def table_name(model: str, k: int) -> str:
    return f"cold-start-forecasts-{model}-k{k}.csv"


# --- the models ----------------------------------------------------------------------------------


@dataclass
class Checked:
    """A model that passed the checks (reading C12), with its one reading of `fit_units`."""

    name: str
    model: object
    fit_units: tuple[str, ...]
    extra: object | None


def check_models(names, matrix, scored_units) -> list[Checked]:
    """Each model loaded and checked as `learn_run` checks it, with `scored_units` as the units
    scored; raises `MatrixError` (no such model), `learn_run.FitOnScored`, `learn_run.Refused`."""
    found = []
    for name in names:
        model = learn_run.load_model(name, matrix)
        fit_units = tuple(model.fit_units)
        reader = learn_run.check_model(name, model, fit_units, tuple(scored_units))
        if reader:
            raise learn_run.Refused(
                f"model {name} reads other days' outcomes; the cold start gives a model the k "
                "local days only through its inputs (reading C12)"
            )
        found.append(Checked(name, model, fit_units, model_inputs.extra_of(model)))
    return found


def fit(checked: Checked, fit_frames: dict) -> None:
    """`fit` on the cars tables of the model's `fit_units`, as `dial_choice.make_factory` does."""
    frames = [fit_frames[u] for u in checked.fit_units]
    learn_run._call(
        "fit", checked.model.fit, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )


def forecast_at(checked: Checked, units: dict, k: int, scored: dict, kappa: float) -> pd.DataFrame:
    """The model's checked forecast table at k for `units` (name -> `UnitDays`): the frame of each
    unit at k (`cold_start_days.frame_at`, the scored days `scored[name]`), predicted at the levels
    up to `kappa`."""
    frames = [cold_start_days.frame_at(units[u], k, scored[u], checked.extra) for u in units]
    frame = pd.concat(frames, ignore_index=True)
    levels = list(driver_risk.levels_up_to(kappa))
    return check_table(learn_run._call("predict", checked.model.predict, frame, levels))


# --- one unit at one k ---------------------------------------------------------------------------


@dataclass
class Measured:
    """The one-way rung's losses of one unit at one k (`dial_splits.UnitLosses` over all its
    counting days; only the local and scored days are filled) and what made them."""

    losses: dial_splits.UnitLosses
    prepared: dict  # day index -> `gate_day.PreparedDay`
    cache: DayCache


def measure_at(runner, unit: str, n_days: int, local, scored, configuration) -> Measured:
    """The local and scored days of `unit` prepared under `configuration`; every local day that
    offers evaluated at every scale of the grid, every scored day that offers at each tier's
    scale from the local days only (reading C8). Cells not evaluated are NaN and never read."""
    days = [*local, *scored]
    prepared = runner.map(gate_run._prepare_task, [(unit, d, configuration) for d in days])
    position = {d: p for p, d in enumerate(days)}
    cache = DayCache(prepared, lambda pairs: runner.map(gate_run._evaluate_task, pairs))
    value = np.zeros(n_days)
    offering = np.zeros(n_days, dtype=bool)
    for d, p in zip(days, prepared, strict=True):
        value[d] = p.value
        offering[d] = not p.abstains
    grid = dial_splits.GRID
    slot = np.full((n_days, len(grid)), np.nan)
    driver = np.full((n_days, len(grid)), np.nan)

    def fill(chosen, column):
        found = cache.get([position[d] for d in chosen], grid[column])
        for d, evaluation in zip(chosen, found, strict=True):
            slot[d, column] = evaluation.rungs[RUNG].slot_loss
            driver[d, column] = evaluation.rungs[RUNG].driver_loss

    local_offering = [d for d in local if offering[d]]
    for column in range(len(grid)):
        fill(local_offering, column)
    used = np.asarray(local_offering, dtype=int)
    scales = set()
    for tier in dial_splits.TIERS:
        alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
        scale, _ = dial_splits.on_grid_scale(slot[used], driver[used], alpha_slot, alpha_driver)
        if scale > 0:
            scales.add(scale)
    scored_offering = [d for d in scored if offering[d]]
    for scale in sorted(scales):
        fill(scored_offering, list(grid).index(scale))
    losses = dial_splits.UnitLosses(value, offering, slot, driver, np.full(n_days, np.nan))
    return Measured(losses, dict(zip(days, prepared, strict=True)), cache)


def ceilings_of(runner, unit: str, n_days: int, scored, names) -> tuple[dict, int]:
    """Ceiling name -> one value per counting day (NaN off the scored days and where missing),
    and the number of best-causal ceilings that timed out."""
    found = runner.map(gate_run._ceiling_task, [(unit, d) for d in scored])
    arrays = {name: np.full(n_days, np.nan) for name in names}
    for d, ceilings in zip(scored, found, strict=True):
        for name in names:
            v = ceiling_values(ceilings, name)
            arrays[name][d] = np.nan if v is None else v
    return arrays, sum(c.best_causal_outcome == "timeout" for c in found)


def _number(value) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else repr(value)


def day_rows(model, k, unit, keys, measured: Measured, figures, days, ceilings, names):
    """The rows of the days file of one model, k and unit (module text); `days`: (local day
    indices, scored day indices)."""
    rows = []
    local, scored = days
    losses = measured.losses
    for tier, found in figures.items():
        scale = found["scale"]
        column = list(dial_splits.GRID).index(scale) if scale > 0 else None
        for role, chosen in (("local", local), ("scored", scored)):
            for d in chosen:
                p = measured.prepared[d]
                offers = bool(losses.offering[d])
                at = column is not None and offers
                rows.append(
                    {
                        "model": model,
                        "k": k,
                        "unit": unit,
                        "tier": tier,
                        "role": role,
                        "date": keys[d][0],
                        "site": keys[d][1],
                        "offering": offers,
                        "certificate_outcome": p.certificate_outcome,
                        "abstain_reason": p.abstain_reason or "",
                        "value_kwh": repr(float(losses.value[d])),
                        "scale": repr(scale),
                        "offer_kwh": repr(float(scale * losses.value[d])),
                        "slot_loss_at_scale": _number(
                            float(losses.slot[d, column]) if at else None
                        ),
                        "driver_loss_at_scale": _number(
                            float(losses.driver[d, column]) if at else None
                        ),
                        "hindsight_kwh": _number(float(ceilings[names[0]][d])),
                        "best_causal_kwh": _number(float(ceilings[names[1]][d])),
                    }
                )
    return rows


# --- one run -------------------------------------------------------------------------------------


def plan_units(units: dict, local_days=cold_start_days.LOCAL_DAYS) -> dict:
    """Per unit: its counting days, the scored days (reading C6) and the local days of each k."""
    largest = max(local_days)
    plans = {}
    for name, days in units.items():
        keys = cold_start_days.day_keys(days)
        scored = cold_start_days.scored_indices(len(keys), largest)
        plans[name] = {
            "keys": keys,
            "scored": scored,
            "status": "scored"
            if scored
            else f"no scored day: {len(keys)} counting days, none after the first {largest}",
        }
    return plans


def _key(key) -> str:
    return f"{key[0]} {key[1]}"


def run(
    matrix,
    configuration,
    units: dict,
    models,
    fit_frames: dict,
    *,
    workers: int,
    local_days=cold_start_days.LOCAL_DAYS,
):
    """Everything but the input checks and the writing of files: (results document, day rows,
    forecast tables by (model, k), timing). `units`: name -> `UnitDays` of the units scored;
    `models`: `Checked`, the cold-started model first; `fit_frames`: unit -> cars table."""
    started = time.perf_counter()
    names = tuple(matrix.ceilings)
    plans = plan_units(units, local_days)
    scored_units = {n: units[n] for n, p in plans.items() if p["scored"]}
    scored = {n: plans[n]["scored"] for n in scored_units}
    timeouts, failed, timing = Counter(), [], {"seconds": {}}
    figures, rows, tables, ceilings, evaluations = {}, [], {}, {}, {}
    for checked in models:
        figures[checked.name] = {unit: {} for unit in scored_units}
        evaluations[checked.name] = {unit: {} for unit in scored_units}
        try:
            fit(checked, fit_frames)
        except learn_run.ModelFailed as problem:
            failed.append(f"{checked.name}: {problem}")
            continue
        for k in local_days:
            begun = time.perf_counter()
            try:
                table = forecast_at(checked, scored_units, k, scored, configuration.kappa)
            except (learn_run.ModelFailed, ValueError) as problem:
                failed.append(f"{checked.name} | k={k}: {type(problem).__name__}: {problem}")
                continue
            tables[(checked.name, k)] = table
            timing["seconds"][f"{checked.name} | k={k} | forecast"] = time.perf_counter() - begun
            runner = gate_run.Runner(workers, scored_units, matrix, TableFactory(table))
            try:
                for unit in scored_units:
                    one = time.perf_counter()
                    keys = plans[unit]["keys"]
                    local = cold_start_days.local_indices(len(keys), k)
                    try:
                        if unit not in ceilings:
                            ceilings[unit], missing = ceilings_of(
                                runner, unit, len(keys), scored[unit], names
                            )
                            timeouts["best_causal_ceilings"] += missing
                        measured = measure_at(
                            runner, unit, len(keys), local, scored[unit], configuration
                        )
                    except Exception as problem:  # recorded, the other units and k go on
                        failed.append(
                            f"{checked.name} | k={k} | {unit}: {type(problem).__name__}: {problem}"
                        )
                        continue
                    timeouts["certificates"] += sum(
                        p.certificate_outcome == "timeout" for p in measured.prepared.values()
                    )
                    found = dial_choice._evaluations(measured.cache)
                    timeouts["re_solves"] += found["re_solves_of_the_one_way_rung"].get(
                        "timeout", 0
                    )
                    evaluations[checked.name][unit][str(k)] = found
                    by_tier = {
                        tier: cold_start_score.score_k(
                            measured.losses, k, local, scored[unit], tier, ceilings[unit]
                        )
                        for tier in dial_splits.TIERS
                    }
                    figures[checked.name][unit][str(k)] = by_tier
                    rows += day_rows(
                        checked.name,
                        k,
                        unit,
                        keys,
                        measured,
                        by_tier,
                        (local, scored[unit]),
                        ceilings[unit],
                        names,
                    )
                    timing["seconds"][f"{checked.name} | k={k} | {unit}"] = (
                        time.perf_counter() - one
                    )
            finally:
                runner.close()
    timing["total_seconds"] = time.perf_counter() - started
    document = {
        "run": {
            "local_days": list(local_days),
            "largest_k": max(local_days),
            "configuration": configuration.name,
            "dial": configuration.driver_risk,
            "kappa": configuration.kappa,
            "levels": list(driver_risk.levels_up_to(configuration.kappa)),
            "rung": RUNG,
            "scales": list(dial_splits.GRID),
            "tiers": list(dial_splits.TIERS),
            "ceilings": list(names),
            "the_ceiling": names[0],
        },
        "models": {
            "cold_started": models[0].name,
            "reference": models[1].name if len(models) > 1 else None,
            "fit_units": {m.name: list(m.fit_units) for m in models},
            "context_inputs": {m.name: model_inputs.record(m.model) for m in models},
        },
        "units": {
            name: {
                "counting_days": len(plan["keys"]),
                "status": plan["status"],
                "scored_days": [_key(plan["keys"][d]) for d in plan["scored"]],
                "local_days": {
                    str(k): [_key(plan["keys"][d]) for d in range(k)]
                    for k in local_days
                    if plan["scored"]
                },
            }
            for name, plan in plans.items()
        },
        "figures": figures,
        "evaluations": evaluations,
        "timeouts": dict(sorted(timeouts.items())),
        "failed": failed,
    }
    return document, rows, tables, timing


def definitions() -> dict:
    """The words of the figures, as the results carry them."""
    return {
        "readings": "cold_start module text, readings C1 to C14",
        "local_days": (
            "the unit's first k counting days in time order (date, then site); on a scored day "
            "the model sees the unit's rows at or before the k-th of them whose session had "
            "plugged out by the latest decision time of the k days, the same for every scored "
            "day, and that day's own rows (C2 to C5)"
        ),
        "availability": (
            "A_k: 1 minus the mean slot loss of the one-way rung over the offering scored days at "
            "the tier's scale from the k local days; None when nothing is offered"
        ),
        "offer_share": (
            "Y_k: the offer per counting scored day over the mean ceiling per counting scored "
            "day; the plan's ceiling is the hindsight one, the best-causal one is beside it; None "
            "when a scored day's ceiling is missing or the mean ceiling is not positive"
        ),
        "scored_days": "the counting days after the first largest_k, the same for every k (C6)",
        "abstain": (
            "below 9 offering local days (slot loss at 0.10) or 19 (driver loss at 0.05, tier A) "
            "the tier abstains: offer 0, A_k None (C7)"
        ),
    }


# --- the command ---------------------------------------------------------------------------------


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _load(tables: Path, names, matrix) -> dict:
    return {
        name: load_unit_days(
            tables, name, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        for name in names
    }


def _parse(read, names, matrix) -> dict:
    return {
        name: parse_unit_days(
            read, name, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        for name in names
    }


def main(argv: list[str] | None = None, root: Path = TOP) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--dial", type=float, required=True)
    parser.add_argument("--units", nargs="+", required=True)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--held-out-tables", type=Path, default=None)
    parser.add_argument("--held-out-truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        gate_run.argument_paths(args)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        names, kind = cold_start_gate.check_units(args.units)
    except cold_start_gate.UnitsRefused as problem:
        print(f"COULD NOT RUN: the units are refused: {problem}; nothing read")
        return 2
    held_out = kind == cold_start_gate.HELD_OUT
    try:
        cold_start_gate.check_folders(
            root, args.tables, (args.truth_run, args.held_out_tables, args.held_out_truth_run)
        )
    except cold_start_gate.FoldersRefused as problem:
        print(f"COULD NOT RUN: a folder is refused: {problem}; nothing read")
        return 2
    gate = {}
    try:
        if held_out:
            gate["tag_commit"] = cold_start_gate.require_gate(root)
            gate.update(sealed_route.fix_record(gate["tag_commit"]))
    except SealRefused as problem:
        print(f"COULD NOT RUN: REFUSED BY THE GATE: {problem}; nothing read")
        return 2
    try:
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if args.reference == args.model:
            raise MatrixError("--reference must name a model other than --model")
        if held_out and args.held_out_truth_run is None:
            raise MatrixError("held-out units need --held-out-truth-run (reading C11)")
        gate_run.check_files([args.matrix])
        matrix, matrix_sha256 = read(args.matrix)
        configuration = compare_run.dial_configurations(matrix, args.dial)["forecast"]
        models = check_models((args.model, args.reference), matrix, names)
    except (MatrixError, CouldNotRun) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.FitOnScored as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    try:
        if held_out:  # at the commits it passed at above (`sealed_route.same_gate`)
            again = cold_start_gate.require_gate(root)
            sealed_route.same_gate(gate["tag_commit"], gate.get(sealed_route.FIX_KEY), again)
    except SealRefused as problem:
        print(f"COULD NOT RUN: REFUSED BY THE GATE: {problem}; nothing read")
        return 2
    fit_units = tuple(dict.fromkeys(u for m in models for u in m.fit_units))
    held_tables = args.held_out_tables
    if held_out and held_tables is None:
        held_tables = cold_start_gate.held_out_tables(root)
    try:
        if held_out:
            inputs, held_inputs, table_bytes = cold_start_gate.read_two_builds(
                args.truth_run, args.tables, fit_units, args.held_out_truth_run, held_tables, names
            )
            names = sealed_route.scored_units(names, held_inputs)  # abstaining units out
            if not names:
                raise CouldNotRun(sealed_route.no_unit_left(held_inputs))
        else:
            inputs = gate_run.check_inputs(args.truth_run, args.tables, (*fit_units, *names))
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        if held_out:  # the bytes the two checks hashed (module text)
            fit_frames = {
                u: days.cars for u, days in _parse(table_bytes, fit_units, matrix).items()
            }
            units = _parse(table_bytes, names, matrix)
        else:
            fit_frames = {u: days.cars for u, days in _load(args.tables, fit_units, matrix).items()}
            units = _load(args.tables, names, matrix)
        plans = plan_units(units)
    except CouldNotRun as problem:  # a table changed after its check
        print(f"COULD NOT RUN: {problem}")
        return 2
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    if not any(p["scored"] for p in plans.values()):
        statuses = "; ".join(f"{name}: {plan['status']}" for name, plan in plans.items())
        print(f"COULD NOT RUN: no unit has a scored day ({statuses})")
        return 2
    document, rows, tables, timing = run(
        matrix, configuration, units, models, fit_frames, workers=args.workers
    )
    document = {
        "label": LABELS[kind],
        "kind": kind,
        "definitions": definitions(),
        **document,
        "inputs": inputs,
        "matrix": {"file": args.matrix.name, "sha256": matrix_sha256},
    }
    if held_out:
        document["held_out_inputs"] = held_inputs
        document["gate"] = gate
        document[ROUTE_KEY] = {"kind": kind, "label": LABELS[kind], **gate}  # reading C11
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        document["forecast_tables"] = {}
        for (model, k), table in tables.items():
            path = folder / table_name(model, k)
            write_table(table, path)
            document["forecast_tables"][f"{model} | k={k}"] = {
                "file": path.name,
                "rows": len(table),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    text = "\n".join(cold_start_summary.summary_lines(document)) + "\n"
    print(text, end="")
    if run_dir:
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / DAYS_NAME).write_text(gate_run._csv(rows, DAY_COLUMNS, document["label"]))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / TIMING_NAME).write_text(_json(timing))
    return 1 if document["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
