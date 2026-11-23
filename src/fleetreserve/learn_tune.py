"""Tune the pooled LightGBM departure model on the training sites alone, one site held out at a
time (plan section 7, pooled LightGBM; section 13, WP8).

Command line, from the project's top folder:
    python -m fleetreserve.learn_tune [--grid learn/tune-grid.json] [--matrix learn/matrix.json]
        [--tables <folder>] [--truth-run <folder>] [--units <unit> ...] [--workers N]

Data. Only the truth tables (cars and days) of training units (`truth_build.TRAIN_UNITS`, the
default of `--units`) are read, besides the recorded truth build's summary and manifest, the way
`learn_run` gets a model's fit frame: checked against the recorded truth build at the checked-out
commit (`gate_run.check_inputs`) and read with `days_from_truth.load_unit_days` at the matrix's
`rho`, `no_fit` and `small_fleet` (nothing else of the matrix is used). A unit that is not a
training unit is refused before anything is read (exit 2), and `tune_table` refuses it again.
Development units and the units kept for the final run are never read.

Folds. Every training unit is one site (the ACN sites and the Norwegian training sites are units
of their own; `tune_table` refuses a site found in two units), so a fold is one site, taken from
the `site` column: for each site the boosters are fitted on the rows of every other site and
scored on that site (`fold_rows`). Fit rows: every row of the other sites with a departure label
(every car plugged in at τ, on counting and non-counting days, as `models.lgbm` fits departure).
Scored rows: the held-out site's rows of counting days, the rows `learn_run` asks a forecast for.
The features of a row come from `learn_features.build_table` of the pooled tables, where a user's
history is read within the row's own site only, and from `learn_features.with_context`.

Score, fixed before any run on training data. For a held-out site: the pinball loss
(`learn_eval.pinball_loss`) of the departure bound (hours from τ, raised to at least 0 as the
models raise it) against `label_departure_h`, as a mean over the site's scored rows, summed over
the tuning levels kappa (`forecast_empirical.TUNING_KAPPAS`: 0.05, 0.1, 0.2, 0.3). A candidate's
score is the plain mean of that over the held-out sites: each site weighs the same, so the large
sites do not decide alone. Lower is better. Reported beside it and not used for the choice: each
site's score and losses, the miss rate per level (the share of scored rows whose car left before
the bound; per site, and its plain mean over sites) and the mean score within each site type
(workplaces, other sites).

Reading T1 (the levels of the score). Plan section 2's grid of kappa has six levels (0.05 to 0.5),
and the tuned settings were chosen with this score over the first four. The score keeps those four
(`TUNING_KAPPAS`, the grid's levels up to 0.3), so that the settings already chosen are the ones
this score would choose again; no tuned model is re-tuned for the two added levels, and a tuned
model fitted at the grid forecasts at 0.4 and 0.5 with settings no score looked at there.
[Alternative: score over all six levels and tune again, which gives different settings and so a
new model; the owner's decision.]

Candidates: the grid file (`learn_tune_grid`), each fitted on `TUNED_SET`, the columns of
`models.lgbm_ctx` (`FEATURE_COLUMNS`, `TAIL_HISTORY_COLUMNS`, `CONTEXT_COLUMNS`). The choice is
`learn_tune_grid.choose`: the lowest score, ties to the fewer leaves, the larger
`min_data_in_leaf`, the fewer rounds. The present settings (`learn_tune_grid.PRESENT`) are also
scored on the two smaller feature sets of `FEATURE_SETS`, so the results show what each addition
is worth on the training sites. Every other LightGBM parameter is `models.lgbm.PARAMETERS`
(deterministic, one thread per fit).

Work: one task per (feature set, candidate, held-out site), four fits each. `--workers N` runs
the tasks in N processes; the results do not depend on N. A line with the number of fits and a
rough time estimate (from a short probe fit) is printed before the work starts. No linear
programme is solved; the run is bounded by the recorder's `--max-seconds`.

Outputs, written into `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the
summary also on standard output:
- `tune.json`: the score rule, the grid file's name and SHA-256, the inputs, the fit units, each
  site's unit, type and row counts, every candidate's settings, score, rank, per-type means, miss
  rates and per-site results, the present settings' entry per feature set, and the chosen
  candidate. `fleetreserve.learn_tune_settings` writes the tuned settings file from it;
- `tune-summary.txt`: the summary;
- `tune-timing.json`: wall-clock seconds, kept apart from the results.

Exit status: 0 every candidate was scored; 1 the tables could not be scored (a day could not be
built, a site without a type or a country, a site with no scored row, fewer than two sites);
2 could not run (a bad grid, matrix or argument, a unit that is not a training unit, a table
missing, no verified truth build at this commit, a table differing from it).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from fleetreserve import gate_run
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_empirical import TUNING_KAPPAS
from fleetreserve.gate_config import MatrixError, load
from fleetreserve.learn_eval import pinball_loss
from fleetreserve.learn_features import (
    FEATURE_COLUMNS,
    TAIL_HISTORY_COLUMNS,
    WORKPLACE,
    build_table,
    with_context,
)
from fleetreserve.learn_run import MATRIX
from fleetreserve.learn_tune_grid import PRESENT, Grid, GridError, Settings, choose, load_grid, rank
from fleetreserve.models.lgbm import LABELS, PARAMETERS, features
from fleetreserve.models.lgbm_ctx import COLUMNS as CONTEXT_SET_COLUMNS
from fleetreserve.truth_build import OUT, TRAIN_UNITS
from fleetreserve.truth_verify import CouldNotRun

RESULTS_NAME = "tune.json"
SUMMARY_NAME = "tune-summary.txt"
TIMING_NAME = "tune-timing.json"
LABEL = LABELS["departure"][0]
TUNED_SET = "features+tail+context"
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "features": tuple(FEATURE_COLUMNS),
    "features+tail": (*FEATURE_COLUMNS, *TAIL_HISTORY_COLUMNS),
    TUNED_SET: tuple(CONTEXT_SET_COLUMNS),
}
SITE_TYPE_NAMES = {"workplace": True, "other": False}
SCORE_RULE = (
    "per held-out training site, the mean pinball loss of the departure bound (hours from tau) "
    "over the site's rows of counting days, summed over the levels; then the plain mean over "
    "the held-out sites; lower is better"
)
CHOICE_RULE = (
    "lowest score; a tie within 1e-9 goes to the fewer leaves, then the larger "
    "min_data_in_leaf, then the fewer rounds"
)
PROBE_ROUNDS = 20
SHOWN = 5

_WORKER: dict = {}


class NotTraining(ValueError):
    """A unit that is not a training unit was named; nothing of it is read."""


def check_units(units: Sequence[str]) -> tuple[str, ...]:
    """`units` when each is a training unit, named once; raises `NotTraining` otherwise."""
    units = tuple(units)
    outside = [u for u in units if u not in TRAIN_UNITS]
    if outside:
        raise NotTraining(
            f"{outside} are not training units; the tuning reads training units only "
            f"({', '.join(TRAIN_UNITS)})"
        )
    if not units or len(set(units)) != len(units):
        raise NotTraining(f"the units must be named once each, got {list(units)}")
    return units


def tune_table(frames: dict[str, pd.DataFrame], *, rho: float) -> pd.DataFrame:
    """The table every fold is cut from: features, context columns and labels of the pooled cars
    tables `frames` (unit -> cars table), the rows with a departure label (module text)."""
    check_units(tuple(frames))
    pooled = pd.concat(list(frames.values()), ignore_index=True)
    table = with_context(build_table(pooled, rho=rho))
    table = table[table[LABEL].notna()].reset_index(drop=True)
    shared = table.groupby("site")["unit"].nunique()
    if (shared > 1).any():
        raise ValueError(f"the sites {sorted(shared[shared > 1].index)} are in two units")
    return table


def sites_of(table: pd.DataFrame) -> list[str]:
    """The sites of the table, in the order they first appear; at least two are needed."""
    sites = list(dict.fromkeys(table["site"]))
    if len(sites) < 2:
        raise ValueError(f"holding one site out needs at least two sites, got {sites}")
    return sites


def fold_rows(table: pd.DataFrame, site: str) -> tuple[np.ndarray, np.ndarray]:
    """(fit rows, scored rows) of the fold that holds `site` out: every row of the other sites;
    the site's rows of counting days (module text)."""
    held = (table["site"] == site).to_numpy(bool)
    scored = held & table["counting"].to_numpy(bool)
    return ~held, scored


def _quantile(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    columns: Sequence[str],
    parameters: dict,
    rounds: int,
) -> np.ndarray:
    """The quantile `parameters["alpha"]` of `y` given `x`, fitted and read at `x_new`, raised to
    at least 0 (as `models.lgbm.QuantileSet` fits and predicts)."""
    data = lightgbm.Dataset(x, label=y, feature_name=list(columns), free_raw_data=False)
    booster = lightgbm.train(parameters, data, num_boost_round=rounds)
    return np.maximum(booster.predict(x_new, num_threads=1), 0.0)


def fold_scores(
    table: pd.DataFrame,
    columns: Sequence[str],
    settings: Settings,
    site: str,
    kappas: Sequence[float] = TUNING_KAPPAS,
) -> dict:
    """The held-out site's result for one candidate on the columns `columns` (module text)."""
    fit, scored = fold_rows(table, site)
    if not scored.any():
        raise ValueError(f"the site {site} has no row of a counting day to score")
    x = features(table[fit], columns)
    y = table.loc[fit, LABEL].to_numpy(float)
    x_new = features(table[scored], columns)
    truth = table.loc[scored, LABEL].to_numpy(float)
    parameters = {**PARAMETERS, **settings.overrides()}
    losses, misses = {}, {}
    for kappa in kappas:
        bound = _quantile(x, y, x_new, columns, {**parameters, "alpha": kappa}, settings.rounds)
        losses[repr(float(kappa))] = pinball_loss(truth, bound, kappa)
        misses[repr(float(kappa))] = float(np.mean(truth < bound))
    return {
        "score": float(sum(losses.values())),
        "pinball_loss": losses,
        "miss_rate": misses,
        "fit_rows": int(fit.sum()),
        "scored_rows": int(scored.sum()),
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def candidate_entry(settings: Settings, folds: dict[str, dict], workplace: dict[str, bool]) -> dict:
    """A candidate's results from its per-site results `folds` (site -> `fold_scores`): the score
    (the plain mean over sites), the same mean within each site type and the mean miss rates."""
    sites = list(folds)
    levels = list(folds[sites[0]]["miss_rate"])
    return {
        "settings": settings.as_dict(),
        "score": _mean([folds[s]["score"] for s in sites]),
        "score_by_site_type": {
            name: _mean([folds[s]["score"] for s in sites if workplace[s] is flag])
            for name, flag in SITE_TYPE_NAMES.items()
        },
        "miss_rate": {k: _mean([folds[s]["miss_rate"][k] for s in sites]) for k in levels},
        "sites": folds,
    }


def tasks_of(grid: Grid, sites: Sequence[str]) -> list[tuple[str, Settings, str]]:
    """Every (feature set, settings, held-out site): the grid on `TUNED_SET`, then the present
    settings on the smaller feature sets."""
    tasks = [(TUNED_SET, settings, site) for settings in grid.candidates for site in sites]
    tasks += [(name, PRESENT, site) for name in FEATURE_SETS if name != TUNED_SET for site in sites]
    return tasks


def _start_worker(table: pd.DataFrame) -> None:
    _WORKER.clear()
    _WORKER["table"] = table


def _run_task(task: tuple[str, Settings, str]) -> dict:
    name, settings, site = task
    return fold_scores(_WORKER["table"], FEATURE_SETS[name], settings, site)


def run_tasks(table: pd.DataFrame, tasks: list, *, workers: int, progress=None) -> list[dict]:
    """The result of every task, in the order of `tasks`, from this process (one worker) or a
    pool of processes; the results are the same either way."""
    results = []
    if workers == 1:
        _start_worker(table)
        found = map(_run_task, tasks)
        return _collect(found, len(tasks), progress)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_start_worker, initargs=(table,)
    ) as pool:
        results = _collect(pool.map(_run_task, tasks, chunksize=1), len(tasks), progress)
    return results


def _collect(found, total: int, progress) -> list[dict]:
    results = []
    for result in found:
        results.append(result)
        if progress is not None:
            progress(len(results), total)
    return results


def tune(table: pd.DataFrame, grid: Grid, *, workers: int = 1, progress=None) -> dict:
    """The results document of the tuning on `table` (`tune_table`), without the inputs."""
    sites = sites_of(table)
    first = table.drop_duplicates("site").set_index("site")
    workplace = {s: bool(first.loc[s, "site_type"] == WORKPLACE) for s in sites}
    tasks = tasks_of(grid, sites)
    found = dict(
        zip(tasks, run_tasks(table, tasks, workers=workers, progress=progress), strict=True)
    )

    def entry(name: str, settings: Settings) -> dict:
        folds = {site: found[(name, settings, site)] for site in sites}
        return candidate_entry(settings, folds, workplace)

    candidates = [entry(TUNED_SET, settings) for settings in grid.candidates]
    scored = [(s, c["score"]) for s, c in zip(grid.candidates, candidates, strict=True)]
    for settings, candidate in zip(grid.candidates, candidates, strict=True):
        candidate["rank"] = rank(scored, settings)
    chosen = choose(scored)
    by_settings = dict(zip(grid.candidates, candidates, strict=True))
    counting = table["counting"].to_numpy(bool)
    return {
        "score_rule": SCORE_RULE,
        "choice_rule": CHOICE_RULE,
        "kappas": [float(k) for k in TUNING_KAPPAS],
        "grid": {"file": grid.file, "sha256": grid.sha256, "candidates": len(grid.candidates)},
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
        "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "tuned_feature_set": TUNED_SET,
        "candidates": candidates,
        "chosen": {
            "settings": chosen.as_dict(),
            "score": by_settings[chosen]["score"],
            "rank": by_settings[chosen]["rank"],
        },
        "present": {
            "settings": PRESENT.as_dict(),
            "score": by_settings[PRESENT]["score"],
            "rank": by_settings[PRESENT]["rank"],
        },
        "present_settings_by_feature_set": {
            name: by_settings[PRESENT] if name == TUNED_SET else entry(name, PRESENT)
            for name in FEATURE_SETS
        },
    }


def work_of(tasks: list) -> tuple[int, int]:
    """(fits, boosting rounds) of the tasks: one fit per task and level."""
    return len(tasks) * len(TUNING_KAPPAS), sum(s.rounds for _, s, _ in tasks) * len(TUNING_KAPPAS)


def estimate_line(table: pd.DataFrame, tasks: list, workers: int) -> str:
    """The number of fits and a rough time estimate from a probe fit of `PROBE_ROUNDS` rounds at
    the present settings on the first fold."""
    fits, rounds = work_of(tasks)
    fit, scored = fold_rows(table, tasks[0][2])
    columns = FEATURE_SETS[TUNED_SET]
    started = time.perf_counter()
    _quantile(
        features(table[fit], columns),
        table.loc[fit, LABEL].to_numpy(float),
        features(table[scored], columns),
        columns,
        {**PARAMETERS, **PRESENT.overrides(), "alpha": TUNING_KAPPAS[0]},
        PROBE_ROUNDS,
    )
    per_round = (time.perf_counter() - started) / PROBE_ROUNDS
    minutes = rounds * per_round / workers / 60.0
    return (
        f"Estimate: {len(tasks)} tasks, {fits} fits, {rounds} boosting rounds in all; about "
        f"{minutes:.0f} minutes with {workers} worker(s) (rough: from a {PROBE_ROUNDS}-round "
        "probe fit at the present settings; larger trees take longer per round)."
    )


def _settings_text(document: dict) -> str:
    return Settings(**document).text()


def summary_lines(results: dict) -> list[str]:
    sites = results["sites"]
    lines = [
        f"Tuning of the pooled LightGBM departure model on {len(sites)} training sites, one "
        f"held out at a time ({results['rows']} rows, {results['grid']['candidates']} "
        f"candidates of {results['grid']['file']}).",
        f"Score: {results['score_rule']}.",
        "Sites: "
        + "; ".join(
            f"{site} ({s['site_type']}, {s['rows']} rows, {s['scored_rows']} scored)"
            for site, s in sites.items()
        )
        + ".",
    ]
    by_settings = {json.dumps(c["settings"], sort_keys=True): c for c in results["candidates"]}

    def line(head: str, entry: dict) -> str:
        types = ", ".join(
            f"{name} {'none' if value is None else f'{value:.4f}'}"
            for name, value in entry["score_by_site_type"].items()
        )
        return (
            f"{head}: {_settings_text(entry['settings'])}: score {entry['score']:.4f}, rank "
            f"{entry['rank']} of {len(results['candidates'])} (by site type: {types})"
        )

    for head, key in (("Chosen", "chosen"), ("Present settings", "present")):
        lines.append(line(head, by_settings[json.dumps(results[key]["settings"], sort_keys=True)]))
    lines.append(
        "Present settings by feature set: "
        + "; ".join(
            f"{name} {entry['score']:.4f}"
            for name, entry in results["present_settings_by_feature_set"].items()
        )
        + "."
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
        units = check_units(args.units)
        grid = load_grid(args.grid)
        matrix = load(args.matrix)
        inputs = gate_run.check_inputs(args.truth_run, args.tables, units)
    except (NotTraining, GridError, MatrixError, CouldNotRun) as problem:
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
        table = tune_table(frames, rho=matrix.rho)
        tasks = tasks_of(grid, sites_of(table))
        print(estimate_line(table, tasks, args.workers), flush=True)
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
