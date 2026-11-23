"""The 95% intervals over days of a recorded scoring run (plan section 11: "intervals: 95%
bootstrap over days"; O1: "X's interval above 0"; O4: "upper end >= 90%, lower end <= 5%").

Command line, from the project's top folder:
    python -m fleetreserve.day_intervals --run <run folder> --seed <integer> [--resamples N]

`--run`: one run recorded by `run_record` of `python -m fleetreserve.gate_run` (it read its results
from `gate.json` and `gate-day-losses.json`) or of `python -m fleetreserve.learn_run`
(`learn-gate.json` and `learn-day-losses.json`). Both files must be verified by `run_evidence` at
the commit checked out now: the run ended with exit 0, was neither stopped nor timed out, the files
are those it wrote, and it was recorded at this commit or at an earlier one. An earlier commit is
accepted for a run off the sealed route when every committed path that differs from the checked-out
commit is on the keep-list of `run_evidence` (the summary then names the earlier commit, as the
pass-mark tools do); a run through the sealed route (its results hold `sealed_route`) is accepted
only at the commit it was recorded at (`run_evidence` reading E3). The manifest's command must
name the module; anything else could not run. `--seed`: required, no default (the run list fixes
it). `--resamples`: 2,000 unless given (the project owner's decision of 16 Nov 2026). The run
must have been recorded with `--interval-seed` equal to `--seed` and `--interval-resamples` equal
to `--resamples` (reading I12); otherwise the command could not run.

Per configuration, unit, rung (fixed plan, one-way re-solve, two-way re-solve) and tier (A, and B
when the run scores it), the run's own value and the 95% interval of: the scale, the offer in kWh
per counting test day, the offer per car, the offer as a share of each ceiling, availability,
shortfall, slot loss and driver loss, with the share of resamples abstaining. The arithmetic and
its readings I1 to I8 are `day_bootstrap`'s. The command adds:

  I9.  Units are those the run reports, one by one; the run pools no units beyond its own (the
       pooled Norwegian unit is one unit of the run). The per-unit intervals are formed unit by
       unit; the figures over several units are the pooled block (readings I13 to I17).
       Configurations that share one work key share every number, so they share every interval.
  I10. One split per unit. `gate_run` and `learn_run` draw one split per unit
       (`gate_run.plan_unit`), as the plan's held-out run does. A run of many splits
       (`dial_choice`, `compare_run`) is refused: the mean over splits of each split's interval
       ends is not the interval of the mean offer over splits nor of any one split's offer, and a
       reader would take it for one. [Alternative: resample within each split and report the
       mean of the interval ends under a label saying so.]
  I11. The point values are the run's own (`rungs.<rung>.test` and `rungs.<rung>.tier_b.test` of
       its results). Before any resample, the per-day file is checked against them: the scale
       recomputed on the run's own calibration days by `crc.calibrate_search` must equal the
       run's scale exactly, and the offer, the offer per car, the shares and the losses recomputed
       from the per-day file must equal the run's within 1e-9 (relative). A difference fails the
       command (exit 1) and no interval of that job is written.
  I12. Where the replay is done. The scoring run itself evaluates, with `--interval-seed`, every
       (day, scale) pair the resamples' searches ask for (`day_resample`): it holds the prepared
       days, the evaluation and the worker pool, so nothing is prepared twice and the pairs come
       from the very days the run scored. This command reads them and replays nothing. A
       resample that meets a pair the run recorded as failed is bounded (`day_bootstrap` reading
       I3), counted and labelled; one that meets a pair neither evaluated nor failed fails the
       command (exit 1). [Alternative: this command re-prepares the days from the run's verified
       inputs and replays the pairs itself, which repeats the run's preparation.]

  I13. Pooled figures. Plan section 11 states outcome O1 as one offer "on 9 held-out units" and
       does not say whether X is per unit or one figure over the units, so the block `pooled`
       gives, per configuration, rung and tier, over all units the configuration has in the run,
       two labelled figures, each with its point value (the run's own) and its interval:
       (a) `sum_of_unit_offers`: the sum over units of each unit's offer per counting test day
       (the rule `drift_settings` reading E13 uses for its offers); (b) `total_over_total_days`:
       the total offered kWh over the counting test days of all units together, which is the mean
       of the units' offers weighted by their counting test days. With as many counting test days
       in every unit (a) is (b) times the number of units; with different counts it is not.
       [Alternative: the unweighted mean over units of the unit's offer, which equals (b) only
       when the counts are equal: not written.]
  I14. Per car and the share of each ceiling are pooled with the same numerator: (a) is the sum
       over units of the unit's offer per counting test day over the sum over units of the unit's
       cars (or ceiling) per counting test day; (b) is the total offered kWh over the total cars
       (or total ceiling) of all units' test days, a ratio of sums (`gate_stat` reading G1). A
       pooled share is not computed when a test day's ceiling is missing in any unit.
       [Alternative: the sum or the mean over units of each unit's own per car figure or share.]
  I15. Resampling. Each unit's calibration days and test days are drawn within that unit, each
       unit independently (`day_bootstrap.draws` takes the unit's name), and the scale is
       recomputed per unit per resample as the per-unit intervals do (`day_bootstrap.
       resample_offer`, one call serving both); resample r of a pooled figure combines resample r
       of every unit. A unit that abstains in a resample offers 0 there. The units themselves are
       not resampled: the units of the run are the population, not a sample of units.
       [Alternative: draw the units too, with replacement, which treats the units as a sample.]
  I16. A pooled interval is bracketed as a unit's is (`day_bootstrap` reading I4): every pooled
       figure rises with each unit's offer, so a resample's lower bound combines the units' lower
       scales and its upper bound their upper ones. The interval is the rule's own when every
       unit's scale is exact in every resample, else an outer bound; `resamples_every_scale_
       exact` counts the resamples in which every unit's scale is exact.
  I17. No pooled block per group of units (site type). No one home maps a run's unit names to a
       group: `learn_features.SITE_TYPES` maps site names, which the a.s.r. and NREL units are
       not, and the run list's groups are not recorded in a run's files. A configuration with a
       unit that has no interval (the per-day file differs from the run, reading I11) has no
       pooled block, and its `reason` names the unit. Only units with a per-day record are
       pooled (a unit the run could not split has none; the block's `units` lists those pooled,
       and the summary line says how many; the block's `left_out` and the end of that line
       name every unit of the run that is not pooled, with the run's reason: one it could not
       split, one whose configuration failed, on a held-out run with the reason `failed: <error>`
       and never as abstained; empty, and "none", when there is none). Only
       tiers that every pooled unit scores are pooled. [Alternative: a `--group name=unit,unit`
       argument, a choice for the run list to make.]
  I18. A held-out unit the held-out truth build abstained (the run's `sealed_route` record,
       `sealed_route.abstained_counted`) has no per-day record and no figure, but it is one of the
       held-out units and stays in every count over them, counted as not passing (reading S4 of
       `sealed_route`). The pooled line reads "pooled over K of N units", K the units pooled and
       N all the units of the run, and names it in its left-out part ("abstained in the held-out
       truth build <run>: <reason>; counts as not passing"); each configuration's table has a row
       for it with that text and no figure; `day-intervals.json` holds it under
       `abstained_units` and in the block's `left_out` (`abstained`: true). A run with no such
       unit reads as before ("pooled over K units"). A unit whose configuration failed (reading
       I17) is counted in N the same way, though it is not "abstained". [Alternative: leave it
       out of the counts.]

Not computed here: the every-request route of the fixed plan (O1's "every-request offer D"), whose
closed form is not in the per-day file; the comparison methods of `compare_run` (O1's A and B).

Outputs, written into `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary
also on standard output: `day-intervals.json` (every number), `day-intervals-summary.txt`,
`day-intervals-timing.json` (wall-clock seconds, kept apart). `day-intervals.json` also holds the
key `pooled` (readings I13 to I17) and, whenever the run read holds the key `sealed_route`,
whatever its type, a copy of it, so that `run_evidence` reading E3 covers this file as it covers
the run.

Exit status: 0 every interval was computed; 1 a check of I11 failed or a resample met a pair not
evaluated (I12); 2 could not run (no such run, not verified, stale, another command, a malformed
file, a bad argument, no resamples' pairs of this seed and number).
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

from fleetreserve import day_bootstrap, gate_run, learn_run, sealed_route
from fleetreserve.day_bootstrap import FIGURES
from fleetreserve.gate_config import CEILINGS, TIER_LABELS
from fleetreserve.gate_summary import ROUTE_KEY
from fleetreserve.run_evidence import VERIFIED, CannotCheck, head_commit, read_evidence
from fleetreserve.sessions import TOP

RESULTS_NAME = "day-intervals.json"
SUMMARY_NAME = "day-intervals-summary.txt"
TIMING_NAME = "day-intervals-timing.json"
KINDS = {
    "fleetreserve.gate_run": (gate_run.RESULTS_NAME, gate_run.DAY_LOSSES_NAME),
    "fleetreserve.learn_run": (learn_run.GATE_NAME, learn_run.DAY_LOSSES_NAME),
}
TIER_KEYS = {"A": None, "B": "tier_b"}
RELATIVE = 1e-9  # reading I11


class CouldNotRun(Exception):
    """The run cannot be read (exit status 2)."""


def read_run(run: Path, head: str) -> tuple[str, dict, dict, dict, str]:
    """(module, manifest, results document, per-day file, note) of a verified run (module text);
    the note is what `run_evidence` says after the name of a run read from an earlier commit, and
    empty for a run at the checked-out commit."""
    present = [module for module, (name, _) in KINDS.items() if (run / name).is_file()]
    if len(present) != 1:
        raise CouldNotRun(
            f"{run.name} holds {len(present)} of the results files "
            f"{[name for name, _ in KINDS.values()]}; a gate_run or learn_run run holds one"
        )
    module = present[0]
    results_name, losses_name = KINDS[module]
    try:
        results = read_evidence(run, results_name, head)
        losses = read_evidence(run, losses_name, head)
    except CannotCheck as problem:
        raise CouldNotRun(str(problem)) from problem
    for found in (results, losses):
        if found.status != VERIFIED:
            raise CouldNotRun(f"{run.name} is not verified: it ended with exit code 1 or 2")
    command = results.manifest.get("command")
    if not isinstance(command, list) or module not in command:
        raise CouldNotRun(f"{run.name}'s recorded command does not run python -m {module}")
    return module, results.manifest, results.data, losses.data, results.note


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=RELATIVE, abs_tol=RELATIVE)


def check_points(job: day_bootstrap.Job, rung: str, tier: str, points: dict) -> list[str]:
    """Reading I11 for one rung and tier: what differs between the per-day file and the run."""
    problems = []
    found = job.run_scale(rung, tier)
    if not (found.exact and found.low == points["scale"]):
        problems.append(f"scale recomputed {found.low!r}, the run's {points['scale']!r}")
    stored = job.tests[rung][tier]["scale"]
    if stored != points["scale"]:
        problems.append(f"the per-day file's test scale {stored!r}, the run's {points['scale']!r}")
    test = job.test_days
    beta = points["scale"]
    offers = [beta * v for v in job.value[test]]
    cars = job.cars[test].sum()
    recomputed = {
        "offer_kwh_per_counting_test_day": float(sum(offers) / len(offers)),
        "offer_kwh_per_car_per_day": None if cars == 0 else float(sum(offers) / cars),
    }
    for name, value in recomputed.items():
        if not _close(value, points[name]):
            problems.append(f"{name} recomputed {value!r}, the run's {points[name]!r}")
    for name in CEILINGS:
        ceilings = job.ceilings[name][test]
        value = None
        if not any(math.isnan(c) for c in ceilings) and ceilings.sum() > 0:
            value = float(sum(offers) / ceilings.sum())
        if not _close(value, points["share_of_ceiling"][name]):
            problems.append(f"share of {name} recomputed {value!r}, the run's")
    losses = job.tests[rung][tier]
    offering = job.offers[test]
    for key, name in (("slot", "slot_loss"), ("driver", "driver_loss")):
        value = None
        if losses[key] is not None and offering.any():
            value = float(losses[key][offering].mean())
        if not _close(value, points[name]):
            problems.append(f"{name} recomputed {value!r}, the run's {points[name]!r}")
    return problems


def _entry(document: dict, name: str, unit: str) -> dict:
    try:
        return document["configurations"][name]["units"][unit]
    except KeyError as problem:
        raise CouldNotRun(f"the results have no entry for {name} on {unit}") from problem


def tiers_of(job: day_bootstrap.Job, rungs) -> list[str]:
    return [t for t in TIER_KEYS if all(t in job.tests[r] for r in rungs)]


def job_points(job: day_bootstrap.Job, entry: dict, rungs, name: str) -> tuple[dict, list]:
    """The run's own values of one job under configuration `name` (its `entry` in the results),
    by rung and tier, and the problems of reading I11 (none: an empty list)."""
    points, problems = {}, []
    for rung in rungs:
        for tier in tiers_of(job, rungs):
            scored = entry["rungs"][rung] if tier == "A" else entry["rungs"][rung][TIER_KEYS[tier]]
            points[(rung, tier)] = day_bootstrap.point_values(scored)
            problems += [
                f"{name} on {job.unit}, {rung}, tier {tier}: {p}"
                for p in check_points(job, rung, tier, points[(rung, tier)])
            ]
    return points, problems


def job_intervals(
    job: day_bootstrap.Job, points: dict, rungs, seed: int, resamples: int
) -> tuple[dict, dict]:
    """The intervals of one job, by rung and tier, from one draw for the unit (reading I8), and
    the resamples they were made from (`day_bootstrap.resample_offer`, by rung and tier), which
    the pooled figures read (reading I15)."""
    calibration, test = day_bootstrap.draws(
        seed, job.unit, len(job.calibration_days), len(job.test_days), resamples
    )
    out, drawn = {}, {}
    for rung in rungs:
        out[rung] = {}
        for tier in tiers_of(job, rungs):
            found = points[(rung, tier)]
            drawn[(rung, tier)] = day_bootstrap.resample_offer(job, rung, tier, calibration, test)
            out[rung][tier] = {
                **day_bootstrap.offer_intervals(
                    job, rung, tier, found, calibration, test, drawn[(rung, tier)]
                ),
                **day_bootstrap.loss_intervals(job, rung, tier, found, test),
            }
    return out, drawn


def left_out(document: dict, name: str, jobs: dict) -> list[dict]:
    """The units of configuration `name` in the run that have no job in `jobs` (they hold no
    per-day record: the run could not split them, or the configuration failed on them), each with
    the run's reason (reading I17)."""
    entries = document["configurations"][name]["units"]
    left = [
        {"unit": unit, "reason": str(entry.get("reason", "no per-day record"))}
        for unit, entry in entries.items()
        if unit not in jobs
    ]
    return left + [
        {"unit": unit, "reason": text, "abstained": True}
        for unit, text in abstained_units(document).items()
    ]


def abstained_units(document: dict) -> dict[str, str]:
    """The held-out units the held-out truth build abstained, as the run's route record names
    them (`sealed_route.abstained_counted`): none off the route or when none abstained. They are
    units of the run that no per-day record holds, so they are among the units left out and
    count as not passing (reading I18)."""
    route = document.get(ROUTE_KEY)
    held = route.get("held_out_inputs") if isinstance(route, dict) else None
    return sealed_route.abstained_counted(held)


def pooled(jobs: dict, rungs, left: list[dict] | None = None) -> dict:
    """The pooled block of one configuration (readings I13 to I17). `jobs`: unit -> (job, the
    run's own values by rung and tier, the job's resamples), or None for a unit with no
    interval. `left`: the units of the run left out (`left_out`), written as `left_out`."""
    units = list(jobs)
    missing = [u for u in units if jobs[u] is None]
    if missing:
        return {
            "units": units,
            "left_out": left or [],
            "reason": f"{missing} have no interval (the per-day file differs from the run, I11)",
        }
    tiers = [t for t in TIER_KEYS if all(t in tiers_of(jobs[u][0], rungs) for u in units)]
    out = {
        "units": units,
        "left_out": left or [],
        "counting_test_days": {u: len(jobs[u][0].test_days) for u in units},
        "rungs": {},
    }
    for rung in rungs:
        out["rungs"][rung] = {
            tier: day_bootstrap.pooled_intervals(
                [jobs[u][0] for u in units],
                [jobs[u][1][(rung, tier)] for u in units],
                [jobs[u][2][(rung, tier)] for u in units],
            )
            for tier in tiers
        }
    return out


def check_resampled(job: day_bootstrap.Job, rungs, seed: int, resamples: int) -> None:
    """Raise `CouldNotRun` unless the run evaluated the pairs of this seed's resamples for every
    tier it scores (reading I12)."""
    found = job.resampled
    again = f"record the scoring run with --interval-seed {seed} --interval-resamples {resamples}"
    if found is None:
        raise CouldNotRun(
            f"{job.unit}: the per-day file holds no resamples' pairs (the run was recorded "
            f"without --interval-seed); {again}"
        )
    if (found["seed"], found["resamples"]) != (seed, resamples):
        raise CouldNotRun(
            f"{job.unit}: the run evaluated the pairs of seed {found['seed']} and "
            f"{found['resamples']} resamples, not of seed {seed} and {resamples}; {again}"
        )
    lacking = [t for t in tiers_of(job, rungs) if t not in found["tiers"]]
    if lacking:
        raise CouldNotRun(f"{job.unit}: the run evaluated no resamples' pairs for tier {lacking}")


def not_evaluated(job: day_bootstrap.Job, found: dict) -> list[str]:
    """Reading I12: a resample that met a pair neither evaluated nor failed is a fault."""
    out = []
    for rung, tiers in found.items():
        for tier, entry in tiers.items():
            count = entry["scale"]["resamples_with_pairs_not_evaluated"]
            if count:
                out.append(
                    f"{job.unit}, {rung}, tier {tier}: {count} resamples met a (day, scale) pair "
                    "the run neither evaluated nor recorded as failed"
                )
    return out


def compute(module, manifest, document, losses, run_name, seed, resamples, note=""):
    """The results document of the command and the problems of reading I11."""
    rungs = list(losses["rungs"])
    jobs = []
    try:
        for record in losses["jobs"]:
            jobs.append(day_bootstrap.Job.from_record(record, rungs))
    except (KeyError, TypeError, ValueError) as problem:
        raise CouldNotRun(f"the per-day file cannot be read: {problem}") from problem
    roles = {}
    for job in jobs:
        key = (tuple(job.calibration_days), tuple(job.test_days))
        if roles.setdefault(job.unit, key) != key:
            raise CouldNotRun(f"{job.unit} has two different splits in the per-day file")
    covered = {(name, job.unit) for job in jobs for name in job.configurations}
    for name, configuration in document["configurations"].items():
        for unit, entry in configuration["units"].items():
            if "rungs" in entry and (name, unit) not in covered:
                raise CouldNotRun(f"the per-day file has no record for {name} on {unit}")
    for job in jobs:
        check_resampled(job, rungs, seed, resamples)
    configurations, problems, kept = {}, [], {}
    for job in jobs:
        try:
            checked = [
                job_points(job, _entry(document, name, job.unit), rungs, name)
                for name in job.configurations
            ]
        except (KeyError, TypeError) as problem:
            raise CouldNotRun(f"the results cannot be read for {job.unit}: {problem}") from problem
        wrong = [p for _, found in checked for p in found]
        problems += wrong
        found = None if wrong else job_intervals(job, checked[0][0], rungs, seed, resamples)
        if found is not None:
            found, drawn = found
            problems += not_evaluated(job, found)
            kept[job.unit, job.configurations] = (job, checked[0][0], drawn)
        for name in job.configurations:
            configurations.setdefault(name, {})[job.unit] = found
    blocks = {}
    for name in configurations:
        mine = {}
        for job in jobs:
            if name in job.configurations:
                mine[job.unit] = kept.get((job.unit, job.configurations))
        blocks[name] = pooled(mine, rungs, left_out(document, name, mine))
    run = {
        "folder": run_name,
        "command": module,
        "commit": manifest["commit"],
        "start": manifest["start"],
        "end": manifest["end"],
        "outputs_sha256": manifest["outputs"],
    }
    if note:
        run["accepted_from_earlier_commit"] = note
    results = {
        "definitions": definitions(),
        "run": run,
        "label": document.get("label"),
        "trial": document.get("trial"),
        "partial": document.get("partial"),
        "setting": document.get("inputs", {}).get("setting"),
        "marks": {
            key: document[key] for key in ("rotation_whatif", "look_ahead") if key in document
        },
        "seed": seed,
        "resamples": resamples,
        "levels": list(day_bootstrap.LEVELS),
        "quantile_method": day_bootstrap.QUANTILE_METHOD,
        "configurations": configurations,
        "pooled": {"definitions": pooled_definitions(), "configurations": blocks},
        "problems": problems,
    }
    if ROUTE_KEY in document:
        results[ROUTE_KEY] = copy.deepcopy(document[ROUTE_KEY])
    if abstained_units(document):
        results["abstained_units"] = abstained_units(document)
    return results, problems


def definitions() -> dict:
    return {
        "interval": (
            "the 2.5% and 97.5% percentiles over resamples (numpy quantile, method linear), no "
            "bias correction; offers: calibration and test days resampled, each group with "
            "replacement and its own count, the scale recomputed by the run's calibration; "
            "losses: test days resampled at the run's scale"
        ),
        "scale_bracket": (
            "the run evaluated every (day, scale) pair the resamples' searches ask for, so every "
            "scale is the rule's own (resamples_exact); only where a pair's evaluation failed "
            "(resamples_with_failed_pairs) does the scale lie between the search with that "
            "pair's losses 1 and the search with them 0, and the offer interval, the lower "
            "percentile of the lower offers and the upper percentile of the upper offers, is then "
            "an outer bound; lower_end_between and upper_end_between say how far each end can move"
        ),
        "availability": "1 minus the mean slot loss over the offering test days",
        "shortfall": "the mean driver loss over the offering test days",
        "abstaining": "resamples whose scale is 0 (offer 0), certain and possible",
        "units": "per unit as the run reports; the figures over units are the block `pooled`",
        "point": "the run's own value",
    }


def pooled_definitions() -> dict:
    return {
        FIGURES[0]: (
            "the sum over the configuration's units of each unit's offer per counting test day; "
            "per car and as a share of a ceiling, over the sum over the units of the unit's cars "
            "(or ceiling) per counting test day"
        ),
        FIGURES[1]: (
            "the total offered kWh over the total counting test days of all the configuration's "
            "units; per car and as a share of a ceiling, over the total cars (or ceiling) of all "
            "the units' test days"
        ),
        "interval": (
            "each unit's calibration days and test days are drawn within the unit, units "
            "independently, the scale recomputed per unit per resample as for the unit's own "
            "interval; resample r combines resample r of every unit; a unit that abstains offers "
            "0; the units are the population, not resampled; percentiles as for a unit, an outer "
            "bound where a unit's scale is not exact"
        ),
        "point": "the run's own values of the units, combined as the figure says",
        "left_out": (
            "the units of the run that are not pooled, each with the run's reason: a unit the "
            "run could not split holds no per-day record, and nothing of it is in the figures; a "
            "held-out unit the held-out truth build abstained is among them (`abstained`: true), "
            "counted as not passing"
        ),
    }


def _number(value, digits=3) -> str:
    return "none" if value is None else f"{value:.{digits}f}"


def _range(found: dict, digits=3) -> str:
    interval = found.get("interval")
    if interval is None:
        return f"{_number(found.get('point'), digits)} (no interval: {found.get('reason', 'none')})"
    return f"{_number(found['point'], digits)} [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def summary_lines(results: dict) -> list[str]:
    lines = []
    if results["trial"] or results["partial"]:
        lines.append(f"{results['label']}: intervals of a run that is not a gate result.")
    for key, value in results["marks"].items():
        lines.append(f"The run is marked {key}: {json.dumps(value, sort_keys=True)[:200]}")
    run = results["run"]
    lines.append(
        f"95% intervals over days (percentile bootstrap, {results['resamples']} resamples, seed "
        f"{results['seed']}) of the recorded run {run['folder']}"
        f"{run.get('accepted_from_earlier_commit', '')} ({run['command']}, commit "
        f"{run['commit'][:12]}), truth setting {results['setting']}."
    )
    lines.append(
        "Offer: in every resample the unit's calibration days and its test days are drawn with "
        "replacement, each group as many days as the run has; the scale is recomputed by the "
        "run's own calibration search on the drawn calibration days that offer (a day drawn "
        "twice counts twice), from the losses the scoring run evaluated at every (day, scale) "
        "pair that search asks for. Losses (availability, shortfall): the test days alone are "
        "drawn, at the run's scale. The lines below are per unit; the figures over all the units "
        "of a configuration follow them, headed Pooled; the tiers are never pooled."
    )
    lines.append(_exactness(results))
    route = results.get(ROUTE_KEY)
    held = route.get("held_out_inputs") if isinstance(route, dict) else None
    lines += sealed_route.abstention_lines(held)  # a unit the run left out (no interval for it)
    for problem in results["problems"]:
        lines.append(f"FAILED: {problem}")
    for name, units in results["configurations"].items():
        for unit, rungs in units.items():
            if rungs is None:
                lines.append(
                    f"{name} | {unit}: no interval (the per-day file differs from the run)"
                )
                continue
            for rung, tiers in rungs.items():
                for tier, found in tiers.items():
                    lines.append(
                        f"{name} | {unit} | {rung} | {TIER_LABELS[tier]}: {_offer(found)}; "
                        f"{_abstaining(found)}; availability {_range(found['availability'])}, "
                        f"shortfall {_range(found['shortfall'])}"
                    )
        for unit, text in results.get("abstained_units", {}).items():
            lines.append(f"{name} | {unit}: {text}")  # in the table, with no figure (I18)
    return lines + pooled_lines(results)


def pooled_lines(results: dict) -> list[str]:
    """The pooled block in words (readings I13 to I17)."""
    lines = [
        "Pooled over the units of each configuration, two figures that O1 could mean, never one "
        f"without the other: (a) {FIGURES[0]}, the sum over units of each unit's offer per "
        f"counting test day; (b) {FIGURES[1]}, the total offered kWh over the counting test days "
        "of all units together. Each unit's days are drawn within the unit, units independently; "
        "resample r combines resample r of every unit; a unit that abstains offers 0 there; the "
        "units are the population and are not resampled."
    ]
    for name, block in results["pooled"]["configurations"].items():
        if "rungs" not in block:
            lines.append(
                f"{name} | pooled: no pooled figure, {block['reason']}; {_left_out(block)}"
            )
            continue
        for rung, tiers in block["rungs"].items():
            for tier, found in tiers.items():
                a, b = found[FIGURES[0]], found[FIGURES[1]]
                lines.append(
                    f"{name} | pooled over {_of_units(block)} | {rung} | "
                    f"{TIER_LABELS[tier]}: offer (a) "
                    f"{_range(a['offer_kwh_per_counting_test_day'])} kWh, (b) "
                    f"{_range(b['offer_kwh_per_counting_test_day'])} kWh per counting test day; "
                    f"per car (a) {_range(a['offer_kwh_per_car_per_day'])}, (b) "
                    f"{_range(b['offer_kwh_per_car_per_day'])} kWh per car per day; "
                    f"{_pooled_exact(found)}; {_left_out(block)}"
                )
    return lines


def _of_units(block: dict) -> str:
    """The count in the pooled line: "K units", or "K of N units" when a unit the held-out build
    abstained, or one whose configuration failed, is left out (I17, I18)."""
    found = len(block["units"])
    if not any(u.get("abstained") or _failed(u["reason"]) for u in block["left_out"]):
        return f"{found} units"
    return f"{found} of {found + len(block['left_out'])} units"


def _failed(reason: str) -> bool:
    """Whether a left-out unit's reason is that its configuration failed: `failed: <error>` on a
    held-out run, `configuration failed` otherwise (`gate_run.assemble`)."""
    return reason.startswith("failed:") or reason == "configuration failed"


def _left_out(block: dict) -> str:
    """The units of the run not pooled, in words (reading I17)."""
    left = block["left_out"]
    named = ", ".join(f"{u['unit']} ({u['reason']})" for u in left) or "none"
    return f"units left out of the pooling: {named}"


def _pooled_exact(found: dict) -> str:
    n, exact = found["resamples"], found["resamples_every_scale_exact"]
    if exact == n:
        return f"every unit's scale exact in every resample ({n} of {n})"
    return f"outer bounds: every unit's scale exact in {exact} of {n} resamples"


def _exactness(results: dict) -> str:
    """Whether every resample's scale is the rule's own, over the whole run."""
    total = exact = failed = missing = 0
    for units in results["configurations"].values():
        for rungs in units.values():
            for tiers in (rungs or {}).values():
                for found in tiers.values():
                    total += found["scale"]["resamples"]
                    exact += found["scale"]["resamples_exact"]
                    failed += found["scale"]["resamples_with_failed_pairs"]
                    missing += found["scale"]["resamples_with_pairs_not_evaluated"]
    if exact == total:
        return (
            f"Every scale is exact: in {exact} of {total} resamples (over every configuration, "
            "unit, rung and tier with an interval) the scale is the calibration rule's own on "
            "the drawn days."
        )
    return (
        f"Not every scale is exact: {total - exact} of {total} resamples (over every "
        f"configuration, unit, rung and tier with an interval) have a scale known only between "
        f"two bounds: {failed} met a pair whose evaluation failed in the run and {missing} a "
        "pair the run did not evaluate (FAILED below; one resample can meet both); each line "
        "says how many, and its offer interval is then an outer bound."
    )


def _offer(found: dict) -> str:
    """The offer interval per counting test day, and whether it is the rule's own."""
    scale = found["scale"]
    offer = found["offer_kwh_per_counting_test_day"]
    n, exact = scale["resamples"], scale["resamples_exact"]
    if exact == n:
        return f"offer {_range(offer)} kWh per counting test day, every scale exact ({n} of {n})"
    low = offer["lower_end_between"] or [None, None]
    high = offer["upper_end_between"] or [None, None]
    ends = offer["interval"] or [None, None]
    return (
        f"offer {_number(offer['point'])}, outer bound [{_number(ends[0])}, {_number(ends[1])}] "
        f"kWh per counting test day (lower end between {_number(low[0])} and "
        f"{_number(low[1])}, upper end between {_number(high[0])} and {_number(high[1])}); "
        f"{exact} of {n} scales exact, {scale['resamples_with_failed_pairs']} of {n} resamples "
        f"met a failed evaluation, {scale['resamples_with_pairs_not_evaluated']} a pair not "
        "evaluated"
    )


def _abstaining(found: dict) -> str:
    """How many resamples abstain (scale 0), and how many of them for too few offering days."""
    n = found["scale"]["resamples"]
    certain = found["abstaining"]["resamples_at_scale_zero"]
    possible = found["abstaining"]["resamples_possibly_at_scale_zero"]
    few = found["abstaining"]["resamples_too_few_offering_calibration_days"]
    count = f"{certain}" if certain == possible else f"{certain} to {possible}"
    return f"abstains in {count} of {n} resamples ({few} for too few offering calibration days)"


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resamples", type=int, default=day_bootstrap.RESAMPLES)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        if args.seed < 0:
            raise CouldNotRun("--seed must be a non-negative integer")
        if args.resamples < 1:
            raise CouldNotRun("--resamples must be at least 1")
        found = read_run(args.run, head_commit(TOP))
        module, manifest, document, losses, note = found
        results, problems = compute(
            module, manifest, document, losses, args.run.name, args.seed, args.resamples, note
        )
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    text = "\n".join(summary_lines(results)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / RESULTS_NAME).write_text(_json(results))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / TIMING_NAME).write_text(_json({"seconds": time.perf_counter() - started}))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
