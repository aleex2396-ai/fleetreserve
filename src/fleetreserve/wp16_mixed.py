"""O1's interval of X for a run whose units have different headline configurations.

Plan section 11 gives intervals for `lgbm_tail` and `hazard_tuned_inputs`; the neural model's
headline configuration differs by site type, and `day_intervals` writes its pooled block one
configuration at a time (`day_intervals` reading I13 to I17). The run's own files hold what is
needed to pool each unit under its own headline configuration: `learn-day-losses.json` has, for
every unit and every group of configurations sharing one work key (`day_losses`, a "job"), each
day's certified value, role and offering flag, the scales and losses of the calibration and the
test days at the chosen scale per rung and tier, and the (day, scale) pairs of the resamples of
the run's `--interval-seed` (written by `learn_run.py` as `DAY_LOSSES_NAME`; `learn-gate-days.csv`
beside it holds each day's offers); `learn-gate.json` holds the run's own values.

`pooled_block(source, learn_label, headline, units, left, rung)` does what
`day_intervals.compute` does for one configuration, with its own functions and no other
method (reading I13 to I17):
1. `day_bootstrap.Job.from_record` for each job of `learn-day-losses.json`;
2. for each unit, the job that holds the unit's headline configuration; a unit with no such job (it
   could not be split, its configuration failed, the truth build gave it no window) is in `left`;
3. the checks of `compute` on the whole file (one split per unit, a record for every configuration
   the run scored, `check_resampled` for every job: the pairs of the run list's seed and 2,000
   resamples), then for each unit `job_points` (the per-day file must equal the run's own values,
   reading I11), `job_intervals` (the unit's resamples, drawn with its name) and `not_evaluated`
   (a resample that met a pair the run neither evaluated nor recorded as failed is a fault, as in
   `compute`); a unit `compute` would refuse gives no interval here. `job_points` is the one
   self-check of reading I11 as `day_intervals` has it (round 3, `eb119c8`): a unit whose run share
   of a ceiling is null because a test day's ceiling is a rounding of 0 (about -1e-14 kWh) agrees
   with the per-day file and keeps its interval (`job_intervals` writes "no figure" with its reason
   for that share alone); a ceiling below the rounding allowance is a problem of its own, and the
   block is then not made. No copy of the check is made here;
4. `day_intervals.pooled` over those units: resample r of the sum combines resample r of each unit.

Made apart: this interval is computed after the run's output was seen, by code of a later commit
than the run's, so it carries `MARK` (`day_intervals.APART_SENTENCE`, the sentence `--apart-from`
stamps): the returned block has the key `made_apart`, its row in `o1-summary` and `o1-rho` names it
in its first cell and its note, and the manifest of `wp16_run` lists the rows. It fills no slot of
plan section 11 (the owner's decisions of 22 November 2026 in plan section 13, WP14).

The seed and the number of resamples are those of the run list (`wp16_units.interval_settings`).
The result has the shape of one entry of `pooled.configurations.<name>.rungs.<rung>.<tier>` of
`day-intervals.json` (what `wp16_learn.pooled_block` returns), with the key `made_apart` added.
Anything that does not hold raises `Missed` with its reason, and O1 then says "no interval" with
that reason: no other method is used.
"""

from fleetreserve import day_bootstrap, day_intervals
from fleetreserve.day_intervals import APART_SENTENCE as MARK  # the one sentence, not a copy
from fleetreserve.learn_run import DAY_LOSSES_NAME, GATE_NAME
from fleetreserve.wp16_drive import Missed, Source
from fleetreserve.wp16_units import interval_settings

RUNG = "one-way"
TIER = "A"


def pooled_block(
    source: Source, learn_label: str, headline: dict, units: list, left: dict, rung: str = RUNG
) -> tuple[dict, list]:
    """(the pooled block of the units' headline configurations, the units pooled); `headline`:
    unit -> configuration name; `left`: unit -> why it has no figure (module text); raises
    `Missed`."""
    seed, resamples = interval_settings(source)
    document = source.json(learn_label, GATE_NAME)
    losses = source.json(learn_label, DAY_LOSSES_NAME)
    try:
        rungs = list(losses["rungs"])
        jobs = [day_bootstrap.Job.from_record(record, rungs) for record in losses["jobs"]]
    except (KeyError, TypeError, ValueError) as problem:
        raise Missed(f"unreadable: {learn_label}/{DAY_LOSSES_NAME}: {problem}") from problem
    # the checks of `compute` on the whole per-day file: one split per unit, a record for every
    # configuration the run scored, the resamples' pairs of this seed for every job
    splits: dict = {}
    for job in jobs:
        key = (tuple(job.calibration_days), tuple(job.test_days))
        if splits.setdefault(job.unit, key) != key:
            raise Missed(f"unreadable: {learn_label}: {job.unit} has two different splits")
    covered = {(name, job.unit) for job in jobs for name in job.configurations}
    try:
        for name, configuration in document["configurations"].items():
            for unit, entry in configuration["units"].items():
                if "rungs" in entry and (name, unit) not in covered:
                    raise Missed(
                        f"unreadable: {learn_label}: no per-day record for {name} on {unit}"
                    )
        for job in jobs:
            day_intervals.check_resampled(job, rungs, seed, resamples)
    except (KeyError, TypeError) as problem:
        raise Missed(f"unreadable: {learn_label}: {problem}") from problem
    except day_intervals.CouldNotRun as problem:
        raise Missed(f"unreadable: {learn_label}: {problem}") from problem
    chosen, used = {}, []
    left_out = [{"unit": unit, "reason": why} for unit, why in left.items()]
    for unit in units:
        name = headline.get(unit)
        job = next((j for j in jobs if j.unit == unit and name in j.configurations), None)
        if job is None or unit in left:
            if unit not in left:
                left_out.append({"unit": unit, "reason": "no per-day record of its configuration"})
            continue
        try:
            entry = day_intervals._entry(document, name, unit)
            points, problems = day_intervals.job_points(job, entry, rungs, name)
            if problems:
                raise Missed(f"unreadable: {learn_label}: {problems[0]}")
            found, drawn = day_intervals.job_intervals(job, points, rungs, seed, resamples)
            problems = day_intervals.not_evaluated(job, found)
            if problems:
                raise Missed(f"unreadable: {learn_label}: {problems[0]}")
        except day_intervals.CouldNotRun as problem:
            raise Missed(f"unreadable: {learn_label}: {problem}") from problem
        chosen[unit] = (job, points, drawn)
        used.append(unit)
    if not chosen:
        raise Missed(f"no figure: no unit of {learn_label} has a per-day record")
    block = day_intervals.pooled(chosen, rungs, left_out)
    found = block.get("rungs", {}).get(rung, {}).get(TIER)
    if not isinstance(found, dict):
        raise Missed(f"no figure: the pooled block of {learn_label} has no {rung} tier {TIER}")
    return {**found, "made_apart": MARK}, used
