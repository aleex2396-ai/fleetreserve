"""Pooled calibration across small development units: one scale for several units (a trial made
after the sealed run, `after_sealed_run`).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.pooled_calibration --model <name> [--units <unit> ...]
        [--kappa <level>] [--dial <value>] [--splits N] [--workers N] [--matrix <file>]
        [--tables <folder>] [--truth-run <folder>]

Plan section 6 calibrates one scale per unit, on that unit's own calibration days. A unit with few
days has too few offering calibration days (the slot loss needs 9, the driver loss 19; `crc`) and
offers nothing. This command measures the other reading: per seeded split, one scale calibrated on
the offering calibration days of all the units together, then each unit's test days scored at that
scale, beside the same split's scale of each unit alone. The guarantee of `crc` then holds on
average over the pooled days, not unit by unit (reading P2). It changes no plan definition and
no pass mark, and is not part of the plan.

Units. `--units` names the units pooled (default: the five Norwegian development sites,
`DEFAULT_UNITS`). A name that is not a development unit of the truth build (`DEVELOPMENT_UNITS`)
is refused before anything is read (`check_units`, exit 2): held-out, training and sealed names
alike, and the pooled Norwegian unit, which holds the days of its sites. The command reads no
held-out table and calls no held-out loader. Units whose days cannot be split (`gate_run.plan_unit`)
are listed and left out of the pool.

Configuration. The dial family of the matrix (default `gate/matrix.json`) at `--kappa` (default
0.5, the main level) as `dial_choice --kappa` makes it (`dial_choice.family_at`), the member whose
dial is `--dial` (default 0.02, the main dial), with the empirical departure. The model is built,
checked and fitted as `dial_choice` does (`dial_choice.make_factory`). Every day that offers is
evaluated once at each scale of `dial_splits.GRID` (`dial_choice.measure`), so a split costs no
solve. Splits are seeds 1 to `--splits` (default 200), made unit by unit by
`dial_choice.split_indices` (the matrix's calibration-days rule).

Calibration. Per split and tier (`dial_splits.tier_alphas`: A, the slot loss at alpha 0.10 and the
driver loss at alpha 0.05; B, the slot loss alone), the losses of the offering calibration days of
all pooled units are put in one list (`pooled_scale`) and given to `crc.calibrate_search` on the
grid of scales, without bisection (reading P1), so the scale is a grid scale. A unit's test days
are scored at that scale by `dial_splits.score_split` with a rule that returns it
(`UnitLosses.rule`). With one unit the pooled scale is that unit's own (test `one unit`).

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set, the summary also on standard output:
- `pooled-calibration.json` (`RESULTS_NAME`): per unit and tier the summary of the unit's own
  scale and of the pooled scale over the splits (`dial_splits.summarise`), and per tier the
  pool's figures: the scales chosen, the offering calibration days, and the mean test losses over
  the offering test days of all units together;
- `pooled-calibration-summary.txt`, `pooled-calibration-forecasts.csv` (the model's forecast
  table, hashed in the results) and `pooled-calibration-timing.json`;
- `after_sealed_run.FILE_NAME`, the label file.
Every output says that it is a trial made after held-out results were seen
(`after_sealed_run`): the key `made_after_sealed_run` and the label of the results, the first
lines of the summary and the label file. The label is always on: this command has no run
without it.

Exit status: 0 every unit was measured; 1 the model was refused or failed, or a measure failed; 2
could not run (a bad argument, matrix or model name, a unit that is not a development unit, no
unit that can be split, a model fitted on a scored unit, a table missing, no verified truth build
at this commit, a table differing from it).

Readings (alternatives in brackets):
  P1. The pooled scale is a grid scale, from `crc.calibrate_search` with a bisection tolerance
      above the widest gap of the grid, so no scale between two grid scales is evaluated. This is
      the scale `dial_splits.on_grid_scale` gives on the same days (test `matches the on-grid
      rule`), so the pooled and the single-unit figures are measured as `dial_choice` measures.
      [Alternative: bisection between grid scales, which needs every day evaluated at every
      bisection scale and is not done.]
  P2. A split's pool holds the offering calibration days of every unit that can be split, each
      day once. Days of different units are not weighted by unit size. [Alternative: equal weight
      per unit.]
  P3. The pool's test losses are the means over the offering test days of all units together at
      the pooled scale, so they are weighted by the units' offering test days. Each unit's own
      losses are reported beside them.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from fleetreserve import (
    after_sealed_run,
    crc,
    dial_choice,
    dial_splits,
    dial_summary,
    gate_run,
    learn_run,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import write_table
from fleetreserve.gate_config import TIER_LABELS, MatrixError, load
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import DEVELOPMENT_UNITS, OUT, POOLED
from fleetreserve.truth_verify import CouldNotRun

MATRIX = TOP / "gate" / "matrix.json"
RESULTS_NAME = "pooled-calibration.json"
SUMMARY_NAME = "pooled-calibration-summary.txt"
TABLE_NAME = "pooled-calibration-forecasts.csv"
TIMING_NAME = "pooled-calibration-timing.json"
SPLITS = 200
KAPPA = 0.5
DIAL = 0.02
GRID = dial_splits.GRID
TIERS = dial_splits.TIERS
RUNG = dial_choice.RUNG
UNITS = tuple(u for u in DEVELOPMENT_UNITS if u != POOLED)
DEFAULT_UNITS = tuple(u for u in UNITS if not u.startswith("nrel"))  # the five Norwegian sites
LABEL = (
    "a trial: one scale from the pooled offering calibration days of several units, each unit "
    "scored at it; the guarantee holds on average over the pool, not per unit"
)


# --- the units ------------------------------------------------------------------------------------


def check_units(units) -> tuple[str, ...]:
    """`units` as a tuple when each is a development unit of the truth build, other than its pooled
    Norwegian unit, named once; raises `CouldNotRun` otherwise (module text). Nothing is read."""
    units = tuple(units)
    if not units:
        raise CouldNotRun("--units names no unit")
    unknown = [u for u in units if u not in UNITS]
    if unknown:
        raise CouldNotRun(
            f"--units names {', '.join(unknown)}, not a development unit pooled here "
            f"({', '.join(UNITS)}); training, held-out and sealed names are refused, and the "
            f"unit {POOLED} holds the days of its sites, so it is not pooled beside them"
        )
    twice = sorted({u for u in units if units.count(u) > 1})
    if twice:
        raise CouldNotRun(f"--units names {', '.join(twice)} more than once")
    return units


# --- the pooled scale -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PooledScale:
    """The scale of one split and tier from the pooled days (module text)."""

    scale: float
    reason: str  # why the scale is 0, empty otherwise
    offering_calibration_days: int
    by_unit: dict  # unit -> its offering calibration days in the pool


def pooled_scale(losses: dict, calibration: dict, tier: str) -> PooledScale:
    """One scale calibrated on the offering calibration days of every unit in `losses` together
    (module text, reading P1). `losses`: unit -> `dial_splits.UnitLosses`; `calibration`: unit ->
    the split's calibration day indices of that unit."""
    alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
    slot, driver, given = [], [], {}
    for name, found in losses.items():
        days = np.asarray(calibration[name], dtype=int)
        used = days[found.offering[days]]
        slot.append(found.slot[used])
        driver.append(found.driver[used])
        given[name] = len(used)
    empty = np.zeros((0, len(GRID)))
    pooled_slot = np.concatenate(slot) if slot else empty
    pooled_driver = np.concatenate(driver) if driver else empty
    column = {g: k for k, g in enumerate(GRID)}

    def loss_at(beta):
        return pooled_slot[:, column[beta]], pooled_driver[:, column[beta]]

    found = crc.calibrate_search(
        loss_at, len(pooled_slot), alpha_slot, alpha_driver, grid=GRID, tolerance=1.0
    )
    reason = ""
    if found.beta <= 0:
        reason = dial_splits.NO_SAFE_SCALE if found.evaluations else dial_splits.TOO_FEW
    return PooledScale(found.beta, reason, len(pooled_slot), given)


def _rule(scale: float, reason: str):
    """A scale rule (`dial_splits.UnitLosses.rule`) that gives `scale` whatever days it is given."""

    def rule(slot, driver, used, alpha_slot, alpha_driver, grid):
        return scale, reason

    return rule


def score_pooled(losses: dict, splits: dict, tier: str):
    """(the pool's scale per split, each unit's score per split at its own scale, each unit's
    score per split at the pooled scale) for one tier. `splits`: unit -> [(seed, calibration
    days, test days)], the same seeds in the same order for every unit."""
    names = list(losses)
    seeds = [seed for seed, _, _ in splits[names[0]]]
    if any([s for s, _, _ in splits[name]] != seeds for name in names):
        raise ValueError("the units' splits are not of the same seeds")
    pools, own, pooled = [], {n: [] for n in names}, {n: [] for n in names}
    for i, seed in enumerate(seeds):
        found = pooled_scale(losses, {n: splits[n][i][1] for n in names}, tier)
        pools.append(found)
        for name in names:
            _, calibration, test = splits[name][i]
            own[name].append(dial_splits.score_split(losses[name], seed, calibration, test, tier))
            fixed = replace(losses[name], rule=_rule(found.scale, found.reason))
            pooled[name].append(dial_splits.score_split(fixed, seed, calibration, test, tier))
    return pools, own, pooled


def _over(values: list, limit: float) -> int:
    return sum(1 for v in values if v > limit + crc.TOLERANCE)


def _mean(values: list) -> float | None:
    return float(np.mean(values)) if values else None


def pool_summary(pools: list, pooled: dict, splits: dict, tier: str) -> dict:
    """The pool's figures of one tier over the splits (module text, reading P3)."""
    alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
    names = list(pooled)
    slot, driver, offers = [], [], []
    for i in range(len(pools)):
        scored = [(n, pooled[n][i]) for n in names]
        parts = [
            (s.offering_test_days, s.slot_loss, s.driver_loss)
            for _, s in scored
            if s.slot_loss is not None
        ]
        days = sum(d for d, _, _ in parts)
        if days:
            slot.append(sum(d * a for d, a, _ in parts) / days)
            driver.append(sum(d * b for d, _, b in parts) / days)
        test_days = sum(len(splits[n][i][2]) for n in names)
        total = sum(s.offer_kwh * len(splits[n][i][2]) for n, s in scored)
        offers.append(total / test_days if test_days else 0.0)
    scales = [p.scale for p in pools]
    out = {
        "splits": len(pools),
        "splits_with_scale_above_zero": sum(1 for s in scales if s > 0),
        "mean_scale": _mean(scales),
        "scales": dict(
            sorted(Counter(f"{s:g}" for s in scales).items(), key=lambda pair: float(pair[0]))
        ),
        "splits_at_scale_zero_by_reason": dict(
            sorted(Counter(p.reason for p in pools if p.scale <= 0).items())
        ),
        "mean_offering_calibration_days": _mean([p.offering_calibration_days for p in pools]),
        "mean_offer_kwh_per_test_day_over_all_units_test_days": _mean(offers),
        "splits_with_test_losses": len(slot),
        "mean_test_slot_loss": _mean(slot),
        "mean_test_driver_loss": _mean(driver),
        "slot_loss_over_limit": {
            "limit": alpha_slot,
            "splits": _over(slot, alpha_slot),
            "share_of_splits_with_test_losses": (
                _over(slot, alpha_slot) / len(slot) if slot else None
            ),
        },
    }
    if tier == "A":
        out["driver_loss_over_limit"] = {
            "limit": alpha_driver,
            "splits": _over(driver, alpha_driver),
            "share_of_splits_with_test_losses": (
                _over(driver, alpha_driver) / len(driver) if driver else None
            ),
        }
    return out


# --- one run --------------------------------------------------------------------------------------


def definitions(matrix, configuration, splits: int) -> dict:
    return {
        "trial": LABEL,
        "pooled_scale": (
            "per split and tier, one scale from crc.calibrate_search on the offering calibration "
            "days of every unit that can be split, put in one list; a grid scale of "
            "dial_splits.GRID (reading P1); 0 when the pool is too small or the bound fails at the "
            "smallest scale"
        ),
        "own_scale": "the same split's scale of the unit alone (dial_splits.on_grid_scale)",
        "splits": (
            f"{splits} seeded splits per unit (seeds 1 to {splits}), unit by unit, each made by "
            f"day_split with the calibration-days rule (n_calibration {matrix.n_calibration}, "
            f"calibration_half_below {matrix.calibration_half_below})"
        ),
        "unit_figures": (
            "per unit and tier, dial_splits.summarise over the splits at the unit's own scale and "
            "at the pooled scale; the test losses are the means over the unit's offering test days"
        ),
        "pool_figures": (
            "per tier over the splits: the scales chosen, the mean offering calibration days, and "
            "the mean test losses over the offering test days of all units together (reading P3)"
        ),
        "configuration": configuration.name,
        "guarantee": (
            "the bound of crc holds on average over the pooled days; it is not claimed per unit"
        ),
        "tiers": "the tiers are separate blocks and are never added or averaged",
    }


def run(matrix, units: dict, factory, configuration, *, splits: int, workers: int):
    """(results document, timing) of the pooled calibration of `units` (unit -> `UnitDays`, the
    names of `matrix.gate_units`) under `configuration`; `factory` as `gate_run.run_gate` takes it.
    Everything but the input checks and the writing of files (module text)."""
    started = time.perf_counter()
    seeds = list(range(1, splits + 1))
    runner = gate_run.Runner(workers, units, matrix, factory)
    entries, losses, found_splits = {}, {}, {}
    timeouts, failed, timing = Counter(), {}, {"units": {}}
    try:
        for name in matrix.gate_units:
            days = units[name]
            plan = gate_run.plan_unit(name, days, matrix)
            entries[name] = {
                "counting_days": len(plan.keys),
                "calibration_days_used": plan.calibration_days,
                "test_days": plan.test_days,
                "status": plan.status,
            }
            if plan.status != "calibratable":
                continue
            begun = time.perf_counter()
            try:
                ceilings = runner.map(
                    gate_run._ceiling_task, [(name, i) for i in range(len(plan.keys))]
                )
                timeouts["best_causal_ceilings"] += sum(
                    c.best_causal_outcome == "timeout" for c in ceilings
                )
                best_causal = [c.best_causal for c in ceilings]
                prepared = runner.map(
                    gate_run._prepare_task,
                    [(name, i, configuration) for i in range(len(plan.keys))],
                )
                found, cache = dial_choice.measure(
                    prepared, best_causal, lambda pairs: runner.map(gate_run._evaluate_task, pairs)
                )
            except Exception as problem:  # recorded, the other units go on
                failed[name] = f"{type(problem).__name__}: {problem}"
                entries[name]["failed"] = failed[name]
                continue
            timeouts["certificates"] += sum(p.certificate_outcome == "timeout" for p in prepared)
            timeouts["re_solves"] += dial_choice._evaluations(cache)[
                "re_solves_of_the_one_way_rung"
            ].get("timeout", 0)
            entries[name]["offering_days"] = int(found.offering.sum())
            losses[name] = found
            found_splits[name] = dial_choice.split_indices(name, units[name], matrix, seeds)
            timing["units"][name] = time.perf_counter() - begun
    finally:
        runner.close()
    pool = {}
    if losses:
        for tier in TIERS:
            pools, own, pooled = score_pooled(losses, found_splits, tier)
            for name in losses:
                entries[name].setdefault("tiers", {})[tier] = {
                    "own_scale": dial_splits.summarise(own[name], tier),
                    "pooled_scale": dial_splits.summarise(pooled[name], tier),
                }
            pool[tier] = pool_summary(pools, pooled, found_splits, tier)
    timing["total_seconds"] = time.perf_counter() - started
    document = {
        "label": f"pooled calibration: {LABEL}",
        "definitions": definitions(matrix, configuration, splits),
        "run": {
            "splits": splits,
            "seeds": [seeds[0], seeds[-1]] if seeds else [],
            "scales": list(GRID),
            "n_calibration": matrix.n_calibration,
            "calibration_half_below": matrix.calibration_half_below,
            "kappa": configuration.kappa,
            "dial": configuration.driver_risk,
            "configuration": configuration.name,
            "tiers": list(TIERS),
            "rung": RUNG,
            "units": list(matrix.gate_units),
            "units_pooled": sorted(losses),
        },
        "units": entries,
        "pool": pool,
        "timeouts": dict(sorted(timeouts.items())),
        "failed_units": sorted(f"{unit}: {why}" for unit, why in failed.items()),
    }
    return document, timing


def finish(document: dict) -> dict:
    """The results document with the label of a trial made after the sealed run (always)."""
    after_sealed_run.mark(document, True)
    return document


def summary_lines(document: dict) -> list[str]:
    run = document["run"]
    lines = [
        *after_sealed_run.summary_head(document),
        f"Pooled calibration for model {document['model']}: {run['splits']} seeded splits per "
        f"unit (seeds 1 to {run['splits']}), main reading, kappa {run['kappa']:g}, dial "
        f"{run['dial']:g}, one-way re-solve rung.",
        LABEL[0].upper() + LABEL[1:] + ".",
        "Per split one scale is calibrated on the offering calibration days of all pooled units "
        "together, and each unit's test days are scored at it; own scale: the same split's scale "
        "of the unit alone. Offers are kWh per counting test day.",
        "The two tiers are separate blocks and are never added or averaged.",
        "Units pooled: " + (", ".join(run["units_pooled"]) or "none"),
    ]
    for unit, entry in document["units"].items():
        lines.append("")
        if entry["status"] != "calibratable":
            lines.append(f"{unit} ({entry['status']}, {entry['counting_days']} counting days):")
            lines.append("  left out of the pool: the unit cannot be split")
            continue
        if "failed" in entry:
            lines.append(f"{unit}: failed ({entry['failed']})")
            continue
        lines.append(
            f"{unit} ({entry['counting_days']} counting days, {entry['offering_days']} offer; "
            f"each split: {entry['calibration_days_used']} calibration days and "
            f"{entry['test_days']} test days):"
        )
        for tier in TIERS:
            lines.append(f" {TIER_LABELS[tier]}")
            lines.append(dial_summary._header())
            for label, key in (("own scale", "own_scale"), ("pooled scale", "pooled_scale")):
                lines.append(dial_summary._row(label, entry["tiers"][tier][key], tier))
    for tier, found in document["pool"].items():
        lines.append("")
        offer = found["mean_offer_kwh_per_test_day_over_all_units_test_days"]
        over = found["slot_loss_over_limit"]["share_of_splits_with_test_losses"]
        lines.append(
            f"Pool, {TIER_LABELS[tier]}: {found['splits_with_scale_above_zero']} of "
            f"{found['splits']} splits have a scale above 0 (mean scale "
            f"{dial_summary._number(found['mean_scale'], 3)}, mean offering calibration days "
            f"{dial_summary._number(found['mean_offering_calibration_days'], 1)}); scales "
            f"{found['scales']}; mean offer over all units' test days "
            f"{dial_summary._number(offer)} kWh; mean test slot loss "
            f"{dial_summary._number(found['mean_test_slot_loss'], 3)}, over its limit in "
            f"{dial_summary._share(over)} of the splits with test losses"
        )
        if found["splits_at_scale_zero_by_reason"]:
            reasons = found["splits_at_scale_zero_by_reason"]
            lines.append(f"  splits at scale 0 by reason: {reasons}")
    lines.append("")
    timeouts = document["timeouts"]
    lines.append(
        "Timeouts (recorded outcomes): "
        + (", ".join(f"{k} {v}" for k, v in timeouts.items()) if timeouts else "none")
    )
    if document["failed_units"]:
        lines.append("FAILED: " + "; ".join(document["failed_units"]))
    return lines


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def write_outputs(folder: Path, document: dict, text: str, timing: dict) -> None:
    """The results, the summary, the timing and the label file in the run folder `folder`."""
    folder = Path(folder)
    (folder / RESULTS_NAME).write_text(_json(document))
    (folder / SUMMARY_NAME).write_text(text)
    (folder / TIMING_NAME).write_text(_json(timing))
    after_sealed_run.write_note(folder, True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--units", nargs="+", default=list(DEFAULT_UNITS), metavar="UNIT")
    parser.add_argument("--kappa", type=float, default=KAPPA)
    parser.add_argument("--dial", type=float, default=DIAL)
    parser.add_argument("--splits", type=int, default=SPLITS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        units_named = check_units(args.units)  # before any file is read
        gate_run.argument_paths(args)
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if args.splits < 1:
            raise MatrixError("--splits must be at least 1")
        gate_run.check_files([args.matrix])
        matrix = load(args.matrix)
        family = dial_choice.find_family(matrix)
        measured, family = dial_choice.family_at(matrix, family, args.kappa)
        chosen = [c for c in family[1] if c.driver_risk == args.dial]
        if len(chosen) != 1:
            raise MatrixError(
                f"--dial {args.dial:g} is not one of the family's dials "
                f"{[c.driver_risk for c in family[1]]}"
            )
        configuration = chosen[0]
        scored = replace(
            measured, gate_units=units_named, required_units=min(measured.required_units, 1)
        )
        model = learn_run.load_model(args.model, matrix)
    except (CouldNotRun, MatrixError) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    fit_units = tuple(model.fit_units)
    try:
        learn_run.check_model(args.model, model, fit_units, scored.gate_units)
    except learn_run.FitOnScored as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    needed = tuple(dict.fromkeys((*scored.gate_units, *fit_units)))
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
                rho=scored.rho,
                no_fit=scored.no_fit,
                small_fleet=scored.small_fleet,
            )
            for name in needed
        }
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    units = {name: loaded[name] for name in scored.gate_units}
    fit_frames = {name: loaded[name].cars for name in fit_units}
    try:
        factory, table = dial_choice.make_factory(
            args.model, model, scored, units, fit_frames, configuration.kappa
        )
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: the model could not be scored: {problem}")
        return 1
    document, timing = run(
        scored, units, factory, configuration, splits=args.splits, workers=args.workers
    )
    if not document["run"]["units_pooled"]:
        print("COULD NOT RUN: no unit named can be split, so nothing is pooled")
        return 2
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
        write_table(table, folder / TABLE_NAME, False)
        document["forecast_table"] = {
            "file": TABLE_NAME,
            "rows": len(table),
            "sha256": hashlib.sha256((folder / TABLE_NAME).read_bytes()).hexdigest(),
        }
    finish(document)
    text = "\n".join(summary_lines(document)) + "\n"
    print(text, end="")
    if run_dir:
        write_outputs(Path(run_dir), document, text, timing)
    return 1 if document["failed_units"] else 0


if __name__ == "__main__":
    sys.exit(main())
