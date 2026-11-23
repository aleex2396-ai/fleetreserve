"""What the calibrated offer would be if the scale also moved the planned charging toward the
front-loaded plan: a what-if on the development units (plan sections 1, 5 and 6; the mixed family
of `centre_scale_day`).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.centre_scale_whatif --model <name> --configurations <name> [<name> ...]
        --splits N [--room-factors C [C ...]] [--matrix <file>] [--workers N] [--tables <folder>]
        [--truth-run <folder>]

It is a what-if, not a run of the plan: the results' `label` and the summary's first line say so.
It reports numbers and forms no verdict; whether the method's family changes, and which room
factor if any, is the owner's decision.

Inputs, checked as `dial_choice` checks them: `--model` (`fleetreserve.models.<name>`, checked,
fitted and turned into the forecaster of every certificate and re-solve by
`dial_choice.make_factory`, at the plan's levels up to the largest kappa of the configurations);
`--matrix` (default the committed gate matrix), whose gate units are the units measured; a rotation
what-if matrix is refused; `--configurations`, names of the matrix's configurations (none twice);
`--splits` N, seeded splits per unit, seeds 1 to N, made by `dial_choice.split_indices` with the
matrix's calibration-days rule (seed `split_seed` of the matrix is the gate run's split);
`--room-factors` (default `1 2 4 8`), the room factors `c` of `centre_scale_day` (the certified
plan's weight in the mix at scale `beta` is `min(1, c beta)`; widths `beta` x the certified
widths): each finite and at least 1, none twice, 1 among them (reading W5); the truth tables,
checked by `gate_run.check_inputs` (a verified truth build at the commit checked out now).

Per gate unit that the rule splits, its ceilings (`gate_day.day_ceilings`), and per configuration:

1. Every day prepared (`gate_day.prepare_day`). For each offering day the front-loaded plan is
   checked against the rows the day was certified on (`centre_scale_day.front_loaded_inside`); a
   day outside is named with its smallest slack and replayed unmixed under every room factor
   (reading C2 of `centre_scale_day`). At scale 1 the mixed policy's arrays must equal the
   certified policy's on every offering day (the weight there is 1 under every factor).
2. Evaluation caches (`gate_calibrate.DayCache`, each (day, scale) evaluated once per cache):
   unmixed, `gate_day.evaluate` through `gate_run._evaluate_task` (the gate's own replay, every
   rung); one mixed cache per room factor, `centre_scale_day.evaluate` with that factor
   (`mixed_cache`: the mixed policy of that factor at that scale, fixed plan and one-way rung).
3. The gate's identity, on the gate run's split (seed `split_seed`, whether or not it is among the
   seeds): the unmixed figures of the fixed plan and the one-way rung, made as in step 6, must equal
   those of `gate_calibrate.run_job` (the function `gate_run` calls) on the same days and roles,
   with its evaluations served from the unmixed cache; every field but the re-solve counts over the
   whole cache (which hold the other splits' evaluations too).
4. The floor of the mix: every offering day at the smallest grid scale (`FLOOR_SCALE`, 0.05) in
   every cache; per rung the mean slot and driver loss over the offering days, each room factor
   beside unmixed. As the scale tends to 0 the mixed losses of factor 1 tend to the front-loaded
   plan's own; this is the lowest the mix reaches on the grid. At a factor `c` the weight at the
   floor scale is `min(1, 0.05 c)`.
5. The identity at scale 1: every offering day is replayed at scale 1 in the unmixed cache and in
   every mixed cache (the weight is 1 there under every factor, so the mixed policy is the
   certified one), and per factor the fixed plan's and the one-way rung's losses, first failures,
   re-solves by outcome and stay-to-bound checks must equal the unmixed replay's. A difference
   where either replay has a re-solve that timed out is counted and printed, not refused (a time
   limit can end one solve and not the other); any other difference is a failure (exit 1).
6. Per split and rung (the fixed plan and the one-way rung), per kind: unmixed,
   `gate_calibrate.calibrate_rung` (the gate's search on the gate's grid to 2), `score_test` at the
   tier A scale and `score_tier_b`; each room factor, the same with `crc.calibrate_search` on the
   grid up to 1 (`GRID`, `calibrate_mixed`; reading W1), read from that factor's cache. So the
   calibration's losses at every evaluated scale are those of the policy that factor replays at
   that scale: the policy changes with the scale, each scale is evaluated once per factor, and the
   running maximum and the bisection of `crc` act on the losses of the policy of each evaluated
   scale (R6, R7 of `crc`). Tier A: the smaller of the slot and driver scales; tier B: the slot
   loss alone. The figures of each split come from `centre_scale_summary.split_record`, the paired
   difference of each factor minus unmixed beside them.

Timeouts are counted, never read as 0 (`timeouts`): a certificate that times out makes its day
abstain (`gate_day`); a re-solve that times out leaves its rung on the predecessor's plan
(`rungs`), counted by kind (each room factor and unmixed) over the evaluations; a best-causal
ceiling that times out is missing and leaves the share of that split's test days out.

Readings, each with the alternative not taken:

W1. The mixed scale is calibrated on the grid up to 1 (`dial_splits.GRID`), the mix not being
    defined above it (reading C1 of `centre_scale_day`); a mixed scale that holds at 1 stays at 1.
    The unmixed scale keeps the gate's grid up to 2, so that it is the gate's figure; the paired
    difference therefore compares a scale capped at 1 with one capped at 2. [Alternative: cap the
    unmixed scale at 1 too, which is then not the gate's figure.]
W2. The floor is read at the smallest grid scale over all offering days of the unit (not per
    split), and is the same for both tiers (the tiers read the same losses). [Alternative: a scale
    below the grid, which no calibration evaluates.]
W3. The two-way rung is not replayed for the mixed policy (the what-if measures the fixed plan and
    the one-way rung); the unmixed replay is the gate's and runs it, unread.
W4. Whether the floor is well below the unmixed losses is given no number here: the summary prints
    the floor, mixed beside unmixed, before any calibrated figure, and decides nothing.
    [Alternative: a fixed ratio, for example the mixed driver loss at most half the unmixed one,
    printed as a yes or no line; no text sets that ratio.]
W5. Room factor 1 is always among the factors, so that the summary's parts written before room
    factors existed (the floor and the calibrated figures of the mix) keep their meaning; a list
    without 1 is refused (exit 2). With 1 alone the results and the summary are those written
    before room factors existed, apart from step 5 (no `run.room_factors`, no `room_factor`
    definition, no room table). [Alternative: measure 1 whether listed or not, which runs a factor
    the command line did not name.]
W6. A difference at scale 1 (step 5) is a failure, exit 1, with the mismatch printed and nothing
    written: the replays disagree where they must not, so the figures are not those of the policy
    they name. [Alternative: exit 2 (could not run), as before step 5 evaluated every day.]

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set, the summary also on standard output:
- `centre-scale-whatif.json`: every number (`definitions`, `run`, per unit and configuration the
  days outside the set, the identity counts, the floor, each split's figures and the figures over
  splits, each with one entry per kind; `timeouts`, `failed_configurations`, the model, the inputs,
  the matrix's hash, the forecast table's hash);
- `centre-scale-whatif-summary.txt`: the summary (`centre_scale_summary.summary_lines`);
- `centre-scale-whatif-forecasts.csv`: the model's forecast table;
- `centre-scale-whatif-timing.json`: wall-clock seconds, kept apart.

Exit status: 0 every unit and configuration ran and every identity held; 1 the model was refused or
failed, a day could not be built from the tables, a configuration failed (recorded, the others go
on), or the identity at scale 1 failed (step 5; the mismatch is printed and nothing is written); 2
could not run (a bad argument, matrix or model name, a rotation what-if matrix, a model fitted on a
scored unit, a table missing, no verified truth build at this commit, a table differing from it)
or an identity of steps 1 or 3 failed (the mismatch is printed and nothing is written).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from fleetreserve import (
    centre_scale_day,
    centre_scale_summary,
    crc,
    dial_choice,
    dial_splits,
    gate_calibrate,
    gate_run,
    learn_run,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import write_table
from fleetreserve.gate_calibrate import DayCache, offering_days
from fleetreserve.gate_config import Matrix, MatrixError, load
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

MATRIX = dial_choice.MATRIX
RESULTS_NAME = "centre-scale-whatif.json"
SUMMARY_NAME = "centre-scale-whatif-summary.txt"
TABLE_NAME = "centre-scale-whatif-forecasts.csv"
TIMING_NAME = "centre-scale-whatif-timing.json"
GRID = dial_splits.GRID  # reading W1
FLOOR_SCALE = GRID[0]  # reading W2
RUNGS_SHOWN = centre_scale_day.MIXED_RUNGS
LABEL = centre_scale_summary.LABEL
COMPARED = ("slot_loss", "driver_loss", "first_failures", "resolves", "stay")  # step 5
ROOM_FACTORS = (1.0, 2.0, 4.0, 8.0)  # the default of --room-factors


class IdentityFailed(Exception):
    """An identity of steps 1 or 3 failed (module text); the message says where."""


class ScaleOneDiffers(Exception):
    """The identity at scale 1 (step 5) failed; the message says where (reading W6)."""


# --- the per-day work ----------------------------------------------------------------------------


def _mixed_task(job):
    """`job`: (prepared day, scale) for room factor 1, or (prepared day, scale, room factor)."""
    prepared, beta, *factor = job
    return centre_scale_day.evaluate(prepared, beta, *factor).compact()


def mixed_cache(runner, prepared, factor: float) -> DayCache:
    """The evaluation cache of room factor `factor` (step 2): each (day, scale) replayed once by
    `centre_scale_day.evaluate` with that factor, through `runner`."""

    def batch(pairs):
        return runner.map(_mixed_task, [(p, beta, factor) for p, beta in pairs])

    return DayCache(prepared, batch)


def calibrate_mixed(cache: DayCache, calibration, rung: str) -> dict:
    """`gate_calibrate.calibrate_rung` on the grid up to 1 (reading W1), from `cache`."""
    offering = offering_days(cache.prepared, calibration)

    def loss_at(beta):
        found = cache.get(offering, beta)
        slot = np.array([e.rungs[rung].slot_loss for e in found])
        driver = np.array([e.rungs[rung].driver_loss for e in found])
        return slot, driver

    n = len(offering)
    out = gate_calibrate._search(crc.calibrate_search(loss_at, n, grid=GRID), len(calibration))
    out["slot_only"] = gate_calibrate._search(
        crc.calibrate_search(loss_at, n, alpha_driver=1.0, grid=GRID), len(calibration)
    )
    out["driver_only"] = gate_calibrate._search(
        crc.calibrate_search(loss_at, n, alpha_slot=1.0, grid=GRID), len(calibration)
    )
    return out


def score(cache: DayCache, ceilings, calibration, test, calibrate) -> dict:
    """Per rung shown, the gate-format results of one split (module text, step 6)."""
    prepared = cache.prepared
    found = {}
    for rung in RUNGS_SHOWN:
        result = calibrate(cache, calibration, rung)
        beta = float(result["beta_hat"])
        result["test"], _ = gate_calibrate.score_test(cache, prepared, ceilings, test, rung, beta)
        result["tier_b"], _ = gate_calibrate.score_tier_b(
            cache, prepared, ceilings, calibration, test, rung, result
        )
        found[rung] = result
    return found


def served(cache: DayCache):
    """An `evaluate_batch` for `gate_calibrate.run_job` that reads `cache` (step 3)."""
    index = {id(p): d for d, p in enumerate(cache.prepared)}

    def batch(pairs):
        if not pairs:
            return []
        (beta,) = {b for _, b in pairs}
        return cache.get([index[id(p)] for p, _ in pairs], beta)

    return batch


def gate_identity(cache: DayCache, ceilings, roles, found: dict) -> list[str]:
    """Step 3: where the unmixed figures `found` of the gate's split differ from `run_job`'s."""
    gate, *_ = gate_calibrate.run_job(cache.prepared, ceilings, roles, served(cache), tier_b=True)
    mismatches = []
    for rung in RUNGS_SHOWN:
        theirs = {k: v for k, v in gate["rungs"][rung].items() if k != "resolves"}
        if found[rung] != theirs:
            keys = sorted(k for k in theirs if theirs[k] != found[rung].get(k))
            mismatches.append(f"{rung}: {', '.join(keys) or 'the keys differ'}")
    return mismatches


def _timed_out(evaluation) -> bool:
    return any(evaluation.rungs[r].resolves.get("timeout", 0) for r in RUNGS_SHOWN)


def scale_one_identity(mixed: DayCache, unmixed: DayCache) -> tuple[int, list[int], list[int]]:
    """Step 5: (days compared, days differing with a timed-out re-solve, days differing without)."""
    compared, timed, other = 0, [], []
    for (d, beta), found in sorted(mixed.store.items()):
        gate = unmixed.store.get((d, beta))
        if beta != 1.0 or gate is None or found.abstained:
            continue
        compared += 1
        same = all(
            getattr(found.rungs[r], field) == getattr(gate.rungs[r], field)
            for r in RUNGS_SHOWN
            for field in COMPARED
        )
        if not same:
            (timed if _timed_out(found) or _timed_out(gate) else other).append(d)
    return compared, timed, other


def policies_at_one(prepared) -> tuple[int, list[str]]:
    """Step 1: (offering days checked, days whose mixed policy at scale 1 is not the certified)."""
    checked, differ = 0, []
    for p in prepared:
        if p.abstains:
            continue
        checked += 1
        certified = p.certificate.policy
        plan = centre_scale_day.front_loaded_plan(p.day)
        at_one = centre_scale_day.mixed_policy(certified, plan, 1.0)
        if not all(
            np.array_equal(getattr(at_one, k), getattr(certified, k)) for k in ("m", "w", "g", "G")
        ):
            differ.append(f"{p.date} {p.site}")
    return checked, differ


def outside_days(prepared) -> list[dict]:
    """Step 1: the offering days whose front-loaded plan is outside the set, with the slack."""
    found = []
    for p in prepared:
        if p.abstains:
            continue
        slack = centre_scale_day.front_loaded_slack(p)
        if slack < -centre_scale_day.SET_TOLERANCE:
            found.append({"day": f"{p.date} {p.site}", "smallest_slack_kwh": slack})
    return found


def floor(mixed: dict, unmixed: DayCache) -> dict:
    """Step 4: per rung shown, the floor of the mix (`centre_scale_summary.floor_figures`);
    `mixed`: room factor -> its cache, factor 1 among them."""
    offering = offering_days(unmixed.prepared, range(len(unmixed.prepared)))
    kinds = {centre_scale_summary.kind_name(c): cache for c, cache in mixed.items()}
    kinds[centre_scale_summary.UNMIXED] = unmixed
    found = {kind: cache.get(offering, FLOOR_SCALE) for kind, cache in kinds.items()}
    out = {}
    for rung in RUNGS_SHOWN:
        losses = {
            kind: (
                [e.rungs[rung].slot_loss for e in found[kind]],
                [e.rungs[rung].driver_loss for e in found[kind]],
            )
            for kind in found
        }
        mixed_one = losses.pop(centre_scale_summary.MIXED)
        unmixed_losses = losses.pop(centre_scale_summary.UNMIXED)
        out[rung] = centre_scale_summary.floor_figures(
            mixed_one, unmixed_losses, len(offering), FLOOR_SCALE, losses
        )
    return out


def scale_one(mixed: dict, unmixed: DayCache) -> tuple[int, int, list[str]]:
    """Step 5: every offering day replayed at scale 1 in `unmixed` and in each factor's cache
    (`mixed`: room factor -> cache), then compared (`scale_one_identity`). Returns (day and factor
    pairs compared, pairs differing with a timed-out re-solve, the other differences named)."""
    prepared = unmixed.prepared
    offering = offering_days(prepared, range(len(prepared)))
    unmixed.get(offering, 1.0)
    compared, timed, named = 0, 0, []
    for factor, cache in mixed.items():
        cache.get(offering, 1.0)
        count, late, other = scale_one_identity(cache, unmixed)
        compared += count
        timed += len(late)
        named += [f"room {factor:g}: {prepared[d].date} {prepared[d].site}" for d in other]
    return compared, timed, named


def _resolves(cache: DayCache) -> dict:
    out = {}
    for found in cache.store.values():
        for rung, evaluation in found.rungs.items():
            out.setdefault(rung, Counter()).update(evaluation.resolves)
    return {rung: dict(sorted(c.items())) for rung, c in sorted(out.items())}


# --- one unit under one configuration ------------------------------------------------------------


def measure(
    runner, name, configuration, ceilings, roles, splits, gate_seed, factors=(1.0,)
) -> dict:
    """Steps 1 to 6 for one unit and configuration (module text), each room factor of `factors`
    (1 among them); raises `IdentityFailed` or `ScaleOneDiffers`."""
    prepared = runner.map(
        gate_run._prepare_task, [(name, i, configuration) for i in range(len(ceilings))]
    )
    where = f"{name} | {configuration.name}"
    checked, differ = policies_at_one(prepared)
    if differ:
        raise IdentityFailed(
            f"{where}: the mixed policy at scale 1 is not the certified on {differ}"
        )
    outside = outside_days(prepared)
    unmixed = DayCache(prepared, lambda pairs: runner.map(gate_run._evaluate_task, pairs))
    mixed = {factor: mixed_cache(runner, prepared, factor) for factor in factors}
    calibration = [d for d, role in enumerate(roles) if role == "calibration"]
    test = [d for d, role in enumerate(roles) if role == "test"]
    gate_found = score(unmixed, ceilings, calibration, test, gate_calibrate.calibrate_rung)
    mismatches = gate_identity(unmixed, ceilings, roles, gate_found)
    if mismatches:
        raise IdentityFailed(f"{where}: unmixed figures differ from the gate's: {mismatches}")
    floors = floor(mixed, unmixed)
    compared, timed, other = scale_one(mixed, unmixed)
    if other:
        raise ScaleOneDiffers(
            f"{where}: at scale 1 the mixed replay differs from the gate's on {other}"
        )
    records = []
    for seed, cal, tst in splits:
        found = {
            "unmixed": gate_found
            if seed == gate_seed
            else score(unmixed, ceilings, cal, tst, gate_calibrate.calibrate_rung),
        }
        for factor, cache in mixed.items():
            kind = centre_scale_summary.kind_name(factor)
            found[kind] = score(cache, ceilings, cal, tst, calibrate_mixed)
        records.append(centre_scale_summary.split_record(seed, found))
    kinds = {centre_scale_summary.kind_name(c): cache for c, cache in mixed.items()}
    kinds[centre_scale_summary.UNMIXED] = unmixed
    evaluations = {f"{kind}_day_scale_pairs": len(cache.store) for kind, cache in kinds.items()}
    evaluations.update({f"re_solves_{kind}": _resolves(cache) for kind, cache in kinds.items()})
    return {
        "driver_risk": configuration.driver_risk,
        "offering_days": sum(not p.abstains for p in prepared),
        "front_loaded_outside": outside,
        "identity": {
            "policies_checked": checked,
            "scale_one_compared": compared,
            "scale_one_differ_with_timeout": timed,
        },
        "floor": floors,
        "evaluations": evaluations,
        "certificate_timeouts": sum(p.certificate_outcome == "timeout" for p in prepared),
        "splits": records,
    }


# --- one run -------------------------------------------------------------------------------------


def definitions(factors=(1.0,)) -> dict:
    """The words of the results; a `room_factor` entry only when a factor other than 1 is run
    (reading W5)."""
    out = {
        "front_loaded_plan": (
            "each steered car, hour by hour from hour 0 to the last service hour, at its forecast "
            "power limit r_hat f_t until its forecast cap e_cap is reached (centre_scale_day)"
        ),
        "mixed_policy": (
            "at scale beta in (0, 1]: planned charging (1 - beta) x front-loaded plan + beta x "
            "certified plan, gains beta x certified gains, box centre (1 - beta) x the "
            "front-loaded plan's fleet total + beta x certified centre, widths beta x certified "
            "widths; the "
            "certified policy at beta = 1; not defined above 1"
        ),
        "unmixed": "the gate's own figures: gate_day.evaluate, gate_calibrate on the gate's grid",
        "floor": (
            f"per rung, the mean slot and driver loss over the unit's offering days at scale "
            f"{FLOOR_SCALE:g} (the smallest grid scale), mixed and unmixed; the same for both tiers"
        ),
        "split_figures": (
            "per split, rung, kind and tier: the calibrated scale, the offer per counting test day "
            "(scale x mean certified value), the share of each ceiling (ratio of sums), the "
            "offering and failing test days and the mean slot and driver loss over the offering "
            "test days (null at scale 0); mixed_minus_unmixed is paired by split"
        ),
        "over_splits": (
            "mean, 5th and 95th percentile (linear interpolation) over the splits where the figure "
            "is defined, with that number of splits"
        ),
        "front_loaded_outside": (
            "offering days whose front-loaded plan breaks a row the day was certified on by more "
            f"than {centre_scale_day.SET_TOLERANCE:g}; each is replayed unmixed at every scale"
        ),
        "timeouts": (
            "a certificate that times out makes its day abstain; a re-solve that times out leaves "
            "its rung on the predecessor's plan and is counted; a best-causal ceiling that times "
            "out is missing and leaves the share of that split out"
        ),
    }
    if tuple(factors) != (1.0,):
        out["room_factor"] = (
            "room factor c >= 1: at scale beta in (0, 1] the planned charging is (1 - w) x "
            "front-loaded plan + w x certified plan with w = min(1, c x beta), box centre (1 - w) "
            "x the front-loaded plan's fleet total + w x certified centre, widths beta x certified "
            "widths, gains beta x certified gains; c = 1 is the mixed policy above (kind "
            "'mixed'), each other factor has the kind 'mixed_room_<c>'; w = 1, the certified "
            "policy at that scale, wherever c x beta >= 1"
        )
    return out


def run(
    matrix: Matrix,
    units: dict,
    factory,
    configurations,
    *,
    splits: int,
    workers: int,
    factors=(1.0,),
):
    """Everything but the input checks and the writing of files: (document, timing); raises
    `IdentityFailed` or `ScaleOneDiffers`. `factors`: the room factors, 1 among them."""
    started = time.perf_counter()
    kinds = (*(centre_scale_summary.kind_name(c) for c in factors), "unmixed")
    seeds = list(range(1, splits + 1))
    runner = gate_run.Runner(workers, units, matrix, factory)
    entries, failed, timeouts, timing = {}, {}, Counter(), {"units": {}}
    identity = {"seed": matrix.split_seed, "gate_checked": 0, "policies_checked": 0}
    identity.update(scale_one_compared=0, scale_one_differ_with_timeout=0)
    try:
        for name in matrix.gate_units:
            days = units[name]
            plan = gate_run.plan_unit(name, days, matrix)
            entry = {
                "counting_days": len(plan.keys),
                "calibration_days_used": plan.calibration_days,
                "test_days": plan.test_days,
                "status": plan.status,
                "configurations": {},
            }
            entries[name] = entry
            if plan.status != "calibratable":
                continue
            begun = time.perf_counter()
            ceilings = runner.map(
                gate_run._ceiling_task, [(name, i) for i in range(len(plan.keys))]
            )
            timeouts["best_causal_ceilings"] += sum(
                c.best_causal_outcome == "timeout" for c in ceilings
            )
            indices = dial_choice.split_indices(name, days, matrix, seeds)
            for configuration in configurations:
                one = time.perf_counter()
                try:
                    found = measure(
                        runner,
                        name,
                        configuration,
                        ceilings,
                        plan.roles,
                        indices,
                        matrix.split_seed,
                        factors,
                    )
                except (IdentityFailed, ScaleOneDiffers):
                    raise
                except Exception as problem:  # recorded, the other configurations go on
                    failed[(name, configuration.name)] = f"{type(problem).__name__}: {problem}"
                    entry["configurations"][configuration.name] = {
                        "driver_risk": configuration.driver_risk,
                        "failed": failed[(name, configuration.name)],
                    }
                    continue
                found["over_splits"] = centre_scale_summary.over_splits(
                    found["splits"], matrix.ceilings
                )
                entry["configurations"][configuration.name] = found
                identity["gate_checked"] += 1
                for key in (
                    "policies_checked",
                    "scale_one_compared",
                    "scale_one_differ_with_timeout",
                ):
                    identity[key] += found["identity"][key]
                timeouts["certificates"] += found["certificate_timeouts"]
                for kind in kinds:
                    counts = found["evaluations"][f"re_solves_{kind}"]
                    timeouts[f"re_solves_{kind}"] += sum(
                        counts.get(rung, {}).get("timeout", 0) for rung in RUNGS_SHOWN
                    )
                timing["units"][f"{name} | {configuration.name}"] = time.perf_counter() - one
            timing["units"][name] = time.perf_counter() - begun
    finally:
        runner.close()
    document = {
        "label": LABEL,
        "definitions": definitions(factors),
        "run": {
            "splits": splits,
            "seeds": [seeds[0], seeds[-1]],
            "split_seed": matrix.split_seed,
            "configurations": [c.name for c in configurations],
            "grid_mixed": list(GRID),
            "grid_unmixed": list(crc.GRID),
            "floor_scale": FLOOR_SCALE,
            "rungs": list(RUNGS_SHOWN),
            "tiers": list(centre_scale_summary.TIERS),
            "ceilings": list(matrix.ceilings),
            "gate_ceiling": matrix.ceilings[0],
            "n_calibration": matrix.n_calibration,
            "calibration_half_below": matrix.calibration_half_below,
        },
        "units": entries,
        "identity": identity,
        "front_loaded_outside_days": sum(
            len(c.get("front_loaded_outside", []))
            for e in entries.values()
            for c in e["configurations"].values()
        ),
        "timeouts": dict(sorted(timeouts.items())),
        "failed_configurations": sorted(f"{a} | {b}: {why}" for (a, b), why in failed.items()),
    }
    if tuple(factors) != (1.0,):  # reading W5
        document["run"]["room_factors"] = list(factors)
    timing["total_seconds"] = time.perf_counter() - started
    return document, timing


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _configurations(matrix: Matrix, names) -> tuple:
    if len(set(names)) != len(names):
        raise MatrixError("a configuration is named twice")
    return tuple(matrix.configuration(n) for n in names)


def room_factors(values) -> tuple[float, ...]:
    """The room factors of the command line, smallest first (reading W5); `MatrixError` (exit 2)
    unless each is finite and at least 1, none is named twice and 1 is among them."""
    try:
        factors = tuple(centre_scale_day.check_factor(v) for v in values)
    except ValueError as problem:
        raise MatrixError(f"--room-factors: {problem}") from None
    names = [centre_scale_summary.kind_name(c) for c in factors]
    if len(set(names)) != len(names):
        raise MatrixError("--room-factors: a room factor is named twice")
    if 1.0 not in factors:
        raise MatrixError("--room-factors: 1, the mix as first built, must be among the factors")
    return tuple(sorted(factors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--configurations", nargs="+", required=True)
    parser.add_argument("--splits", type=int, required=True)
    parser.add_argument("--room-factors", nargs="+", type=float, default=list(ROOM_FACTORS))
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if args.splits < 1:
            raise MatrixError("--splits must be at least 1")
        factors = room_factors(args.room_factors)
        matrix = load(args.matrix)
        if matrix.rotation_whatif:
            raise MatrixError("a rotation what-if matrix (rotation_whatif) is not measured here")
        configurations = _configurations(matrix, args.configurations)
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
    kappa = max(c.kappa for c in configurations)
    try:
        factory, table = dial_choice.make_factory(
            args.model, model, matrix, units, fit_frames, kappa
        )
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: the model could not be scored: {problem}")
        return 1
    try:
        document, timing = run(
            matrix,
            units,
            factory,
            configurations,
            splits=args.splits,
            workers=args.workers,
            factors=factors,
        )
    except IdentityFailed as problem:
        print(f"COULD NOT RUN: identity failed: {problem}")
        return 2
    except ScaleOneDiffers as problem:  # reading W6
        print(f"FAILED: identity failed: {problem}")
        return 1
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
    text = "\n".join(centre_scale_summary.summary_lines(document)) + "\n"
    print(text, end="")
    if run_dir:
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / TIMING_NAME).write_text(_json(timing))
    return 1 if document["failed_configurations"] else 0


if __name__ == "__main__":
    sys.exit(main())
