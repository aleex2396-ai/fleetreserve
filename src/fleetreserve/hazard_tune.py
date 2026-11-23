"""Tune the neural hazard departure model on the training sites alone, one site held out at a
time, and measure on the same folds what each part of the model is worth (plan section 7, hazard
model and its ablations; section 13, WP8).

Command line, from the project's top folder:
    python -m fleetreserve.hazard_tune [--grid learn/hazard-grid.json] [--matrix learn/matrix.json]
        [--tables <folder>] [--truth-run <folder>] [--units <unit> ...] [--workers N]

Data and folds are those of `learn_tune`, by its own functions (`check_units`, `tune_table`,
`sites_of`, `fold_rows`; `gate_run.check_inputs` against the recorded truth build). Files read:
for each unit named in `--units`, its cars table and its days table (`{unit}-cars.csv` and
`{unit}-days.csv` in `--tables`: `check_inputs` hashes both and `load_unit_days` reads both); the
tables folder's setting note, if it has one; the summary and the manifest of the recorded truth
build (the summary is read as a whole, not only the entries of those units); the matrix; the
grid. Tables are opened for the units named in `--units` and for no other unit, and each of
those must be a training unit: a unit that is not one is refused before anything is read
(exit 2). A test lists the files opened when the tables folder also holds the tables of a unit
that is not named. A fold is one site: the network is fitted on every row of the other sites and
forecast for the held-out site's rows of counting days. The rows are those `tune_table` keeps,
the rows with a departure label (the rows without one are in neither the fit nor the forecast
frame). The forecast goes through `models.hazard.HazardModel.predict` on
`learn_features.feature_frame` of the held-out rows, a frame with no label column. The held-out
site is not among the sites the network was fitted on, so it gets the vector of a site not seen in
fitting (`hazard_data.site_codes`), as a development site does in a recorded run. A user's history
comes from `learn_features.build_table` of the pooled tables and is read within the row's own
site only (`learn_features`, module text).

Score, as in `learn_tune` and by the same functions (`learn_eval.pinball_loss`,
`learn_tune.candidate_entry`; the mean over seeds is added here): per held-out site, the mean
pinball loss of the departure bound `d_hat` (hours from τ) against `label_departure_h` over the
site's scored rows, summed over the tuning levels kappa (`TUNING_KAPPAS`, `learn_tune` reading T1:
the grid's levels up to 0.3); a candidate's score is
the plain mean of that over the held-out sites. Lower is better. Reported beside it and not used
for the choice: each site's score and losses, the miss rate per level (the share of scored rows
whose car left before the bound), the mean score within each site type, and the spread of the
score over seeds.

Seeds. The fit of a network depends on its seed. Every fitted candidate is fitted once for every
seed of the grid file, in every fold. A site's result is the mean over the seeds, a candidate's
score the plain mean over the sites of those, and the spread over seeds (minimum, maximum and
standard deviation, divisor n, of the per-seed mean over sites) is reported beside it.

Epochs without refitting. The grid's epoch values are scored from one fit per (fitted candidate,
seed, fold), run to the largest value and forecast at the end of each smaller one
(`hazard_fit.fit`, argument `checkpoints`): nothing in the training loop reads the total number
of epochs, so the fit of a smaller number of epochs is the state the longer fit passes through at
that epoch (a test compares the forecast tables and the weights of the two byte for byte).

Candidates: the grid file (`hazard_tune_grid`). The choice is `hazard_tune_grid.choose`, fixed in
the code before any run on training data: the lowest mean score; a tie within
`learn_tune_grid.TIE` goes to the smaller width, then the fewer epochs, then the larger divergence
weight, then the larger learning rate. The present settings (`hazard_tune_grid.PRESENT`, the
defaults of `HazardSettings`) are one candidate and are named in the results.

The settings `min_presence_support` and `min_presence_probability` only decide which
`survival_h<t>` columns of a forecast are NaN; they never change `d_hat`, which is what is scored,
and they are not in the grid. `TuneModel` sets the fit-row counts by site type with the function
`HazardModel.fit` uses (`models.hazard.presence_support_of`).

Parts (`PARTS`, defined in `hazard_parts`), at the present settings, on the same folds and seeds,
never part of the choice: the set encoder off; the site embedding off; the day factor off; the
set encoder and the day factor off; calendar-only inputs (`CALENDAR_INPUTS`: the date parts and
`public_holiday`; the other inputs are put to 0, `HazardSettings.inputs`). The other switches
stay on in that part, the site embedding and the set encoder among them. Each part's entry holds
its score minus the present settings' score on the same folds and seeds.

Reference, in the same results file: the pooled LightGBM at `models.lgbm`'s fixed settings on the
columns of `models.lgbm_tail` (the "features+tail" set of `learn_tune`), scored on the same folds
and rows by `learn_tune.fold_scores`. It is a reference, not a candidate.

Work: one task per (fitted candidate, seed, fold), per (part, seed, fold) and per (fold) for the
reference. `--workers N` runs the tasks in N processes, each on one PyTorch thread; the results do
not depend on N. The table of features is built once, before the folds. In the process that runs
tasks (this one with `--workers 1`, otherwise each process of the pool) the fit rows of a fold as
arrays and the frame to forecast as the model reads it are made once for each fold and shared by
the fits of that process (the arrays by those that use the same input rules). The probe fits of
the time estimate run before the tasks, on a first fold made in this process for that purpose
only, so with `--workers 1` the first fold's are made twice, once for the probe fits and once for
the tasks. A line with the number of fits and a rough time estimate (from one-epoch probe fits) is
printed before the work starts. No linear programme is solved; the command has no time limit of
its own, the recorder's `--max-seconds` bounds it.

Outputs, written into `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the
summary also on standard output:
- `hazard-tune.json`: the score and choice rules, the grid file's name and SHA-256, the inputs, the
  fit units, each site's unit, type and row counts, every candidate's settings, score, spread over
  seeds, rank, per-type means, miss rates and per-site results, the present settings' entry, the
  parts' entries, the reference entry and the chosen candidate.
  `fleetreserve.hazard_tune_settings` writes the tuned settings file from it;
- `hazard-tune-summary.txt`: the summary;
- `hazard-tune-timing.json`: wall-clock seconds, kept apart from the results.

Exit status: 0 every candidate was scored; 1 the tables could not be scored (a day could not be
built, a site without a type or a country, a site with no scored row, fewer than two sites, a
bound that is not a number); 2 could not run (a bad grid, matrix or argument, a unit that is not
a training unit, a table missing, no verified truth build at this commit, a table differing from
it).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import torch

from fleetreserve import gate_run, hazard_fit, learn_tune
from fleetreserve import hazard_tune_grid as grids
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_empirical import TUNING_KAPPAS
from fleetreserve.gate_config import MatrixError, load
from fleetreserve.hazard_data import Scaler, fit_data
from fleetreserve.hazard_parts import CALENDAR_INPUTS, PARTS  # noqa: F401  (kept as a name here)
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.hazard_tune_grid import (
    CHOICE_RULE,
    Grid,
    GridError,
    Setting,
    choose,
    load_grid,
    rank,
)
from fleetreserve.learn_eval import pinball_loss
from fleetreserve.learn_features import WORKPLACE, feature_frame, with_context
from fleetreserve.learn_run import MATRIX
from fleetreserve.learn_tune_grid import PRESENT as LGBM_PRESENT
from fleetreserve.models import hazard as hazard_model
from fleetreserve.models.hazard import QUANTITIES, HazardModel, presence_support_of
from fleetreserve.truth_build import OUT, TRAIN_UNITS
from fleetreserve.truth_verify import CouldNotRun

RESULTS_NAME = "hazard-tune.json"
SUMMARY_NAME = "hazard-tune-summary.txt"
TIMING_NAME = "hazard-tune-timing.json"
REFERENCE_FEATURE_SET = "features+tail"
SCORE_RULE = (
    "per held-out training site, the mean pinball loss of the departure bound (hours from tau) "
    "over the site's rows of counting days, summed over the levels, as the mean over the seeds; "
    "then the plain mean over the held-out sites; lower is better"
)
SHOWN = 5
TASK_KINDS = ("candidate", "part", "reference")

_WORKER: dict = {}


class BoundNotANumber(ValueError):
    """A forecast departure bound is not a number, so the fold cannot be scored."""


# --- one fold ------------------------------------------------------------------------------------


@dataclass
class Prepared:
    """What the fits of one fold with the same input rules share: the fit rows as arrays, the
    scaler fitted on them and the support counts of the fit rows (module text)."""

    fit_sites: tuple[str, ...]
    scaler: Scaler
    data: object
    presence_support: dict  # per site type, the fit rows reaching each hour (`models.hazard`)


@dataclass
class Fold:
    """The held-out site's fold: the fit rows, the frame to forecast (no label column), the labels
    of the scored rows and the shared inputs by input rules (`data_key`)."""

    site: str
    fit_table: pd.DataFrame
    frame: pd.DataFrame
    truth: np.ndarray
    fit_rows: int
    scored_rows: int
    prepared: dict = field(default_factory=dict)
    context: pd.DataFrame | None = None  # the frame as the model reads it (`TuneModel.attach`)


def prepare_fold(table: pd.DataFrame, site: str) -> Fold:
    """The fold that holds `site` out (`learn_tune.fold_rows`), from the table `tune_table` made."""
    fit, scored = learn_tune.fold_rows(table, site)
    if not scored.any():
        raise ValueError(f"the site {site} has no row of a counting day to score")
    held = table[scored]
    return Fold(
        site=site,
        fit_table=table[fit].reset_index(drop=True),
        frame=feature_frame(held).reset_index(drop=True),
        truth=held[learn_tune.LABEL].to_numpy(float),
        fit_rows=int(fit.sum()),
        scored_rows=int(scored.sum()),
    )


def data_key(settings: HazardSettings) -> tuple:
    """The settings that decide the inputs and targets of a fit (and nothing else)."""
    return (
        settings.input_clip,
        settings.inputs,
        settings.fine_step_h,
        settings.fine_until_h,
        settings.coarse_step_h,
        settings.grid_end_h,
    )


class TuneModel(HazardModel):
    """`HazardModel` that fits from a fold's shared inputs and reads checkpoints. `fit_prepared`
    does what `HazardModel.fit` does after the table is built; `predict` is `HazardModel.predict`,
    with the frame as the model reads it (`HazardModel._context`, the slow calendar columns)
    taken from the fold when the frame asked for is the fold's own (`attach`). A test compares the
    forecast table with that of `HazardModel.fit` and `predict` byte for byte."""

    name = "hazard_tune"

    def __init__(self, *, kappas: Sequence[float], settings: HazardSettings) -> None:
        super().__init__(rho=0.0, kappas=kappas, settings=settings)
        self.shared: tuple | None = None

    def fit_prepared(
        self,
        prepared: Prepared,
        checkpoints: Mapping[int, Callable] | None = None,
    ) -> None:
        self.fit_sites, self.scaler = prepared.fit_sites, prepared.scaler
        self.presence_support = prepared.presence_support
        data = prepared.data
        self.energy_mean, self.energy_std = data.energy_mean, data.energy_std
        position = {q: k for k, q in enumerate(QUANTITIES)}
        fitted = hazard_fit.fit(
            data,
            self.settings,
            len(self.fit_sites),
            np.array([level for _, level in self.outputs]),
            np.array([position[q] for q, _ in self.outputs]),
            checkpoints,
        )
        self.net, self.history = fitted.net, fitted.history

    def attach(self, fold: Fold) -> None:
        """Take `fold`'s frame as read by the model (made once per fold) for `predict`."""
        if fold.context is None:
            read = [c for c in fold.frame.columns if c in hazard_model._READ]
            fold.context = with_context(fold.frame[read])
        self.shared = (fold.frame, fold.context)

    def _context(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.shared is not None and frame is self.shared[0]:
            return self.shared[1]
        return super()._context(frame)


def prepare(fold: Fold, settings: HazardSettings) -> Prepared:
    """The shared inputs of `fold` under the input rules of `settings`, made once per rules."""
    key = data_key(settings)
    if key not in fold.prepared:
        table = fold.fit_table
        fit_sites = tuple(sorted(set(table["site"].astype(str))))
        scaler = Scaler.fit(table, settings)
        data = fit_data(table, scaler, fit_sites, settings)
        fold.prepared[key] = Prepared(fit_sites, scaler, data, presence_support_of(table))
    return fold.prepared[key]


def bounds_of(table: pd.DataFrame) -> dict[float, np.ndarray]:
    """The departure bound `d_hat` of each level, in the order of the forecast frame's rows, raised
    to at least 0 as `learn_tune` raises its bounds; raises `BoundNotANumber`."""
    out = {}
    for kappa in TUNING_KAPPAS:
        bound = table.loc[table["kappa"] == float(kappa), "d_hat"].to_numpy(float)
        if np.isnan(bound).any():
            raise BoundNotANumber(f"a departure bound at level {kappa} is not a number")
        out[float(kappa)] = np.maximum(bound, 0.0)
    return out


def forecast_bounds(
    fold: Fold, settings: HazardSettings, epochs: Sequence[int]
) -> dict[int, dict[float, np.ndarray]]:
    """Fit once under `settings` (whose `epochs` is the largest of `epochs`, or later) and return
    the held-out site's bounds at the end of each of the epochs `epochs`."""
    prepared = prepare(fold, settings)
    model = TuneModel(kappas=TUNING_KAPPAS, settings=settings)
    model.attach(fold)
    found: dict[int, dict[float, np.ndarray]] = {}

    def reader(epoch: int) -> Callable:
        def read(net) -> None:
            model.net = net
            found[epoch] = bounds_of(model.predict(fold.frame, TUNING_KAPPAS))

        return read

    model.fit_prepared(prepared, {epoch: reader(epoch) for epoch in epochs})
    return found


def score_bounds(fold: Fold, bounds: Mapping[float, np.ndarray]) -> dict:
    """The fold's result for the bounds (module text), in the format of `learn_tune.fold_scores`."""
    losses, misses = {}, {}
    for kappa in TUNING_KAPPAS:
        bound = bounds[float(kappa)]
        losses[repr(float(kappa))] = pinball_loss(fold.truth, bound, kappa)
        misses[repr(float(kappa))] = float(np.mean(fold.truth < bound))
    return {
        "score": float(sum(losses.values())),
        "pinball_loss": losses,
        "miss_rate": misses,
        "fit_rows": fold.fit_rows,
        "scored_rows": fold.scored_rows,
    }


# --- tasks ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """`kind` "candidate": the fitted candidate `fit` (width, learning rate, divergence weight) at
    `seed`, scored at each of `epochs`; "part": the part `name` at the present settings and `seed`;
    "reference": the pooled LightGBM reference. All on the fold that holds `site` out."""

    kind: str
    site: str
    seed: int = 0
    fit: tuple = ()
    name: str = ""
    epochs: tuple[int, ...] = ()


def tasks_of(grid: Grid, sites: Sequence[str]) -> list[Task]:
    """Every task (module text): the grid's fitted candidates, then the parts, then the
    reference; each over the sites, and the neural ones over the seeds."""
    tasks = [
        Task("candidate", site, seed, fit=fit, epochs=grid.epochs)
        for fit in grid.fits
        for seed in grid.seeds
        for site in sites
    ]
    tasks += [
        Task("part", site, seed, name=name, epochs=(grids.PRESENT.epochs,))
        for name in PARTS
        for seed in grid.seeds
        for site in sites
    ]
    tasks += [Task("reference", site) for site in sites]
    return tasks


def _start_worker(table: pd.DataFrame) -> None:
    _WORKER.clear()
    _WORKER.update(table=table, folds={})
    torch.set_num_threads(1)


def _fold(site: str) -> Fold:
    if site not in _WORKER["folds"]:
        _WORKER["folds"][site] = prepare_fold(_WORKER["table"], site)
    return _WORKER["folds"][site]


def _run_task(task: Task):
    """The result of a task: `{epoch: result}` for a candidate, a result for the others."""
    if task.kind == "reference":
        columns = learn_tune.FEATURE_SETS[REFERENCE_FEATURE_SET]
        return learn_tune.fold_scores(_WORKER["table"], columns, LGBM_PRESENT, task.site)
    fold = _fold(task.site)
    if task.kind == "part":
        settings = grids.PRESENT.hazard(task.seed, **PARTS[task.name])
    else:
        width, rate, weight = task.fit
        settings = Setting(width, rate, weight, max(task.epochs)).hazard(task.seed)
    found = forecast_bounds(fold, settings, task.epochs)
    results = {epoch: score_bounds(fold, bounds) for epoch, bounds in found.items()}
    return results if task.kind == "candidate" else results[grids.PRESENT.epochs]


def run_tasks(table: pd.DataFrame, tasks: list, *, workers: int, progress=None) -> list:
    """The result of every task, in the order of `tasks`, from this process (one worker) or a pool
    of processes; the results are the same either way."""
    if workers == 1:
        _start_worker(table)
        return _collect(map(_run_task, tasks), len(tasks), progress)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_start_worker, initargs=(table,)
    ) as pool:
        return _collect(pool.map(_run_task, tasks, chunksize=1), len(tasks), progress)


def _collect(found, total: int, progress) -> list:
    results = []
    for result in found:
        results.append(result)
        if progress is not None:
            progress(len(results), total)
    return results


# --- results -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Described:
    """Settings as a document, for `learn_tune.candidate_entry`."""

    document: dict

    def as_dict(self) -> dict:
        return self.document


def site_result(results: Sequence[dict], seeds: Sequence[int]) -> dict:
    """A site's result from its per-seed results: the means over the seeds, and the score by
    seed."""
    levels = list(results[0]["pinball_loss"])
    return {
        "score": float(np.mean([r["score"] for r in results])),
        "pinball_loss": {
            k: float(np.mean([r["pinball_loss"][k] for r in results])) for k in levels
        },
        "miss_rate": {k: float(np.mean([r["miss_rate"][k] for r in results])) for k in levels},
        "score_by_seed": {str(s): r["score"] for s, r in zip(seeds, results, strict=True)},
        "fit_rows": results[0]["fit_rows"],
        "scored_rows": results[0]["scored_rows"],
    }


def entry_of(
    document: dict,
    by_seed: Mapping[int, Mapping[str, dict]],
    workplace: Mapping[str, bool],
) -> dict:
    """The entry of a candidate, part or the present settings, from `by_seed[seed][site]`, the
    fold results: `learn_tune.candidate_entry` over the seed means of the sites, with the score by
    seed (the mean over sites of that seed's scores) and its spread."""
    seeds = list(by_seed)
    sites = list(by_seed[seeds[0]])
    folds = {s: site_result([by_seed[seed][s] for seed in seeds], seeds) for s in sites}
    entry = learn_tune.candidate_entry(Described(document), folds, dict(workplace))
    per_seed = {
        str(seed): float(np.mean([by_seed[seed][s]["score"] for s in sites])) for seed in seeds
    }
    values = list(per_seed.values())
    entry["score_by_seed"] = per_seed
    entry["seed_spread"] = {
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(np.std(values)),
    }
    return entry


def gather(
    tasks: Sequence[Task],
    found: Mapping,
    seeds: Sequence[int],
    kind: str,
    *,
    fit: tuple | None = None,
    name: str | None = None,
    epoch: int | None = None,
) -> dict:
    """`{seed: {site: fold result}}` of the tasks of one kind (a candidate's fit key, or a part's
    name); a candidate's tasks hold a result per epoch, of which `epoch` is read."""
    out: dict = {seed: {} for seed in seeds}
    for task in tasks:
        if task.kind != kind or (fit is not None and task.fit != fit):
            continue
        if name is not None and task.name != name:
            continue
        result = found[task]
        out[task.seed][task.site] = result if epoch is None else result[epoch]
    return out


def tune(table: pd.DataFrame, grid: Grid, *, workers: int = 1, progress=None) -> dict:
    """The results document of the tuning on `table` (`learn_tune.tune_table`), without the
    inputs."""
    sites = learn_tune.sites_of(table)
    first = table.drop_duplicates("site").set_index("site")
    workplace = {s: bool(first.loc[s, "site_type"] == WORKPLACE) for s in sites}
    tasks = tasks_of(grid, sites)
    found = dict(
        zip(tasks, run_tasks(table, tasks, workers=workers, progress=progress), strict=True)
    )

    candidates = []
    for setting in grid.candidates:
        results = gather(
            tasks, found, grid.seeds, "candidate", fit=setting.fit_key(), epoch=setting.epochs
        )
        candidates.append(entry_of(setting.as_dict(), results, workplace))
    scored = [(s, c["score"]) for s, c in zip(grid.candidates, candidates, strict=True)]
    for setting, candidate in zip(grid.candidates, candidates, strict=True):
        candidate["rank"] = rank(scored, setting)
    chosen = choose(scored)
    by_setting = dict(zip(grid.candidates, candidates, strict=True))
    present = {**by_setting[grids.PRESENT]}
    parts = {}
    for name, changes in PARTS.items():
        results = gather(tasks, found, grid.seeds, "part", name=name)
        document = {"present_settings": grids.PRESENT.as_dict(), "changes": _plain(changes)}
        entry = entry_of(document, results, workplace)
        entry["score_minus_present"] = entry["score"] - present["score"]
        parts[name] = entry
    reference_folds = {t.site: found[t] for t in tasks if t.kind == "reference"}
    reference = learn_tune.candidate_entry(LGBM_PRESENT, reference_folds, workplace)
    reference["about"] = (
        "a reference, not a candidate: the pooled LightGBM at the fixed settings of models.lgbm "
        f"on the columns of the '{REFERENCE_FEATURE_SET}' set (models.lgbm_tail), scored on the "
        "same folds and rows by learn_tune.fold_scores"
    )
    reference["feature_set"] = REFERENCE_FEATURE_SET
    reference["columns"] = list(learn_tune.FEATURE_SETS[REFERENCE_FEATURE_SET])
    counting = table["counting"].to_numpy(bool)
    return {
        "score_rule": SCORE_RULE,
        "choice_rule": CHOICE_RULE,
        "kappas": [float(k) for k in TUNING_KAPPAS],
        "seeds": list(grid.seeds),
        "grid": {
            "file": grid.file,
            "sha256": grid.sha256,
            "candidates": len(grid.candidates),
            "fitted_candidates": len(grid.fits),
            "epochs": list(grid.epochs),
        },
        "fit_units": list(dict.fromkeys(table["unit"])),
        "rows": int(len(table)),
        "sites": {
            site: {
                "unit": str(first.loc[site, "unit"]),
                "site_type": "workplace" if workplace[site] else "other",
                "rows": int((table["site"] == site).sum()),
                "scored_rows": int(((table["site"] == site).to_numpy(bool) & counting).sum()),
            }
            for site in sites
        },
        "other_settings": "the defaults of hazard_settings.HazardSettings",
        "candidates": candidates,
        "chosen": {
            "settings": chosen.as_dict(),
            "score": by_setting[chosen]["score"],
            "rank": by_setting[chosen]["rank"],
        },
        "present": present,
        "parts": parts,
        "reference": reference,
    }


def _plain(changes: Mapping) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in changes.items()}


# --- estimate, summary and command ---------------------------------------------------------------


def work_of(tasks: list[Task]) -> dict[str, int]:
    """The number of fits by kind: network fits for candidates and parts, boosting fits (one per
    level) for the reference."""
    count = {kind: sum(1 for t in tasks if t.kind == kind) for kind in TASK_KINDS}
    count["reference"] *= len(TUNING_KAPPAS)
    return count


def estimate_line(fold: Fold, grid: Grid, tasks: list[Task], workers: int) -> str:
    """The number of fits and a rough time estimate from one-epoch probe fits, one per width of
    the grid, on the first fold (the first epoch carries some set-up time, so it errs high)."""
    seconds_per_epoch, forecast_seconds = {}, 0.0
    for width in sorted({s.width for s in grid.candidates} | {grids.PRESENT.width}):
        present = grids.PRESENT
        settings = Setting(width, present.learning_rate, present.kl_weight, 1).hazard(0)
        prepared = prepare(fold, settings)
        model = TuneModel(kappas=TUNING_KAPPAS, settings=settings)
        model.attach(fold)
        started = time.perf_counter()
        model.fit_prepared(prepared)
        seconds_per_epoch[width] = time.perf_counter() - started
        started = time.perf_counter()
        model.predict(fold.frame, TUNING_KAPPAS)
        forecast_seconds = max(forecast_seconds, time.perf_counter() - started)
    total = 0.0
    for task in tasks:
        if task.kind == "candidate":
            total += max(task.epochs) * seconds_per_epoch[task.fit[0]]
            total += len(task.epochs) * forecast_seconds
        elif task.kind == "part":
            total += (
                grids.PRESENT.epochs * seconds_per_epoch[grids.PRESENT.width] + forecast_seconds
            )
    count = work_of(tasks)
    return (
        f"Estimate: {len(tasks)} tasks, {count['candidate'] + count['part']} network fits "
        f"({count['candidate']} of candidates, {count['part']} of parts) and {count['reference']} "
        f"boosting fits of the reference; about {total / workers / 60.0:.0f} minutes with "
        f"{workers} worker(s) (rough: from one-epoch probe fits on the first fold, the "
        "boosting fits not counted)."
    )


def _settings_text(document: dict) -> str:
    return Setting(**document).text()


def summary_lines(results: dict) -> list[str]:
    sites = results["sites"]
    grid = results["grid"]
    lines = [
        f"Tuning of the neural hazard departure model on {len(sites)} training sites, one held "
        f"out at a time ({results['rows']} rows, {grid['candidates']} candidates of "
        f"{grid['file']}, {grid['fitted_candidates']} fitted, seeds {results['seeds']}).",
        f"Score: {results['score_rule']}.",
        "Sites: "
        + "; ".join(
            f"{site} ({s['site_type']}, {s['rows']} rows, {s['scored_rows']} scored)"
            for site, s in sites.items()
        )
        + ".",
        "Tables read: "
        + ", ".join(results["inputs"]["table_sha256"])
        + ", besides the recorded truth build's summary and manifest; a table of a unit that is "
        "not a training unit was not opened.",
        "The parts and the reference below are not candidates: they are measured on the same "
        "folds and take no part in the choice.",
    ]
    by_settings = {json.dumps(c["settings"], sort_keys=True): c for c in results["candidates"]}

    def line(head: str, entry: dict) -> str:
        types = ", ".join(
            f"{name} {'none' if value is None else f'{value:.4f}'}"
            for name, value in entry["score_by_site_type"].items()
        )
        spread = entry["seed_spread"]
        return (
            f"{head}: {_settings_text(entry['settings'])}: score {entry['score']:.4f} (seeds "
            f"{spread['min']:.4f} to {spread['max']:.4f}, std {spread['std']:.4f}), rank "
            f"{entry['rank']} of {len(results['candidates'])} (by site type: {types})"
        )

    lines.append(
        line("Chosen", by_settings[json.dumps(results["chosen"]["settings"], sort_keys=True)])
    )
    lines.append(line("Present settings", results["present"]))
    lines.append(
        "Parts at the present settings (score, minus the present score): "
        + "; ".join(
            f"{name} {part['score']:.4f} ({part['score_minus_present']:+.4f})"
            for name, part in results["parts"].items()
        )
        + "."
    )
    reference = results["reference"]
    lines.append(
        f"Reference (not a candidate), pooled LightGBM on '{reference['feature_set']}' at the "
        f"settings of models.lgbm: score {reference['score']:.4f}."
    )
    best = sorted(results["candidates"], key=lambda c: c["rank"])[:SHOWN]
    lines.append(f"The {len(best)} candidates of lowest score:")
    lines += [
        f"  {c['rank']}. {_settings_text(c['settings'])}: {c['score']:.4f}; miss rates "
        + ", ".join(f"{k}: {v:.3f}" for k, v in c["miss_rate"].items())
        for c in best
    ]
    return lines


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _progress(done: int, total: int) -> None:
    if done == total or done % max(1, total // 20) == 0:
        print(f"{done} of {total} tasks done", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=Path, default=None)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--units", nargs="+", default=list(TRAIN_UNITS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        units = learn_tune.check_units(args.units)
        grid = load_grid(args.grid)
        matrix = load(args.matrix)
        inputs = gate_run.check_inputs(args.truth_run, args.tables, units)
    except (learn_tune.NotTraining, GridError, MatrixError, CouldNotRun) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    started = time.perf_counter()
    try:
        frames = {
            unit: load_unit_days(
                args.tables,
                unit,
                rho=matrix.rho,
                no_fit=matrix.no_fit,
                small_fleet=matrix.small_fleet,
            ).cars
            for unit in units
        }
        table = learn_tune.tune_table(frames, rho=matrix.rho)
        sites = learn_tune.sites_of(table)
        tasks = tasks_of(grid, sites)
        torch.set_num_threads(1)
        print(estimate_line(prepare_fold(table, sites[0]), grid, tasks, args.workers), flush=True)
        results = tune(table, grid, workers=args.workers, progress=_progress)
    except ValueError as problem:
        print(f"FAILED: the training tables could not be scored: {problem}")
        return 1
    results["inputs"] = inputs
    results["matrix"] = {
        "file": args.matrix.name,
        "sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "rho": matrix.rho,
        "no_fit": matrix.no_fit,
        "small_fleet": matrix.small_fleet,
    }
    text = "\n".join(summary_lines(results)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / RESULTS_NAME).write_text(_json(results))
        (folder / SUMMARY_NAME).write_text(text)
        timing = {"workers": args.workers, "total_seconds": time.perf_counter() - started}
        (folder / TIMING_NAME).write_text(_json(timing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
