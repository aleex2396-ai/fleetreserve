"""The per-day losses a scoring run's calibration read, as one results file (plan sections 6, 11).

Plan section 11: "intervals: 95% bootstrap over days". A bootstrap that recalibrates in every
resample needs each calibration day's losses as a function of the scale; the run's other files
hold only the means over all offering calibration days at each scale the search evaluated
(`gate_calibrate`, `rungs.<rung>.evaluations`), and `learn_run` keeps no per-day file at all. This
module writes what the calibration itself read, kept in memory by `gate_calibrate.DayCache`, and
nothing it did not read: no day is evaluated for this file.

`records(groups, plans, prepared, ceilings, results)` gives one record per job that was
calibrated (a configuration group, the configurations sharing one work key, on one unit that is
split), in matrix order and then unit order. `text(records)` is the file. `gate_run` writes it as
`gate-day-losses.json`, `learn_run` as `learn-day-losses.json`; every other output of both is
unchanged.

A record holds:

- `unit`, `configurations` (the names sharing the work, in matrix order);
- `days`: every day of the unit in the run's order, with its role in the unit's one split
  ("calibration" or "test"), its cars, whether it offers (a box at scale 1; `gate_day`), its
  certified value at scale 1 in kWh (0 on a day that does not offer), and both ceilings in kWh
  (null where the best-causal ceiling is missing);
- `calibration`: `days`, the positions of the offering calibration days; `scales`, every scale at
  which the calibration evaluated them, ascending (the union over the three rungs and the three
  searches of each, which share one cache, so every offering calibration day is evaluated at every
  one of them); `slot_loss` and `driver_loss` per rung, one row per offering calibration day and
  one column per scale: the raw day-level losses (the largest over the audited paths), before the
  running maximum `crc.calibrate_search` takes over the scales it evaluates;
- `test`: `days`, the positions of the test days; per rung and tier ("A", and "B" when the run
  scores tier B) the scale the run chose (`scale`) and, when it is above 0, the raw slot and
  driver losses of every test day at it (`slot_loss`, `driver_loss`; 0 on a day that does not
  offer, which the run leaves out of its means); null lists at scale 0, where nothing is scored;
- `resampled`, only when the run was given `--interval-seed` (`day_resample`): the seed, the number
  of resamples and the tiers searched; `pairs`, every (day, scale) pair the resamples' searches
  asked for beyond the run's own, ascending by scale and then day, with the raw slot and driver
  losses per rung (`day` the day's position, `scale`, `slot_loss` and `driver_loss` per rung, one
  value per pair); `failed`, the pairs whose evaluation raised an error, with the error;
  `own_calibration_pairs`, the run's own evaluations of offering calibration days, and
  `searches`, the searches run, for comparing the work. Without the option the record has no such
  key, and the file is what it was.

Every number is written as JSON writes a Python float, which reads back to the same float.
"""

import json
from collections.abc import Sequence

from fleetreserve.gate_calibrate import ceiling_values
from fleetreserve.gate_config import CEILINGS
from fleetreserve.rungs import RUNGS

TIERS_KEY = {"A": None, "B": "tier_b"}
DEFINITION = (
    "per job (configurations sharing one work key, on one unit that is split): every day's role, "
    "cars, offering flag, certified value and ceilings; per rung the raw slot and driver losses of "
    "every offering calibration day at every scale the calibration evaluated; per rung and tier "
    "the chosen scale and the test days' raw losses at it (null lists at scale 0)"
)


def _scale(entry: dict, tier: str) -> float | None:
    """The scale the run chose for one rung and tier, None when the tier was not scored."""
    if tier == "A":
        return float(entry["beta_hat"])
    found = entry.get(TIERS_KEY[tier])
    return None if found is None else float(found["beta_hat"])


def resampled_section(found) -> dict:
    """The `resampled` entry of a record (module text) from a `day_resample.Resampled`."""
    pairs = sorted(found.evaluated, key=lambda key: (key[1], key[0]))
    failed = sorted(found.failed, key=lambda key: (key[1], key[0]))
    return {
        "seed": found.seed,
        "resamples": found.resamples,
        "tiers": list(found.tiers),
        "own_calibration_pairs": found.own_calibration_pairs,
        "searches": found.searches,
        "pairs": {
            "day": [d for d, _ in pairs],
            "scale": [b for _, b in pairs],
            "slot_loss": {
                rung: [found.evaluated[k].rungs[rung].slot_loss for k in pairs] for rung in RUNGS
            },
            "driver_loss": {
                rung: [found.evaluated[k].rungs[rung].driver_loss for k in pairs] for rung in RUNGS
            },
        },
        "failed": {
            "day": [d for d, _ in failed],
            "scale": [b for _, b in failed],
            "error": [found.failed[k] for k in failed],
        },
    }


def job_record(
    unit: str, configurations, roles, prepared, ceilings, results, cache, resampled=None
) -> dict:
    """The record of one calibrated job (module text). `results` and `cache` are the first and
    last items `gate_calibrate.run_job` returns; `resampled`, the job's `day_resample.Resampled`
    when the run evaluated its resamples' pairs."""
    calibration = [d for d, role in enumerate(roles) if role == "calibration"]
    test = [d for d, role in enumerate(roles) if role == "test"]
    offering = [d for d in calibration if not prepared[d].abstains]
    chosen = set(offering)
    scales = sorted({beta for d, beta in cache.store if d in chosen})
    missing = [(d, b) for d in offering for b in scales if (d, b) not in cache.store]
    if missing:
        raise ValueError(f"offering calibration days not evaluated at every scale: {missing[:3]}")
    calibration_losses = {"slot_loss": {}, "driver_loss": {}}
    for rung in RUNGS:
        rows = [[cache.store[(d, b)].rungs[rung] for b in scales] for d in offering]
        calibration_losses["slot_loss"][rung] = [[e.slot_loss for e in row] for row in rows]
        calibration_losses["driver_loss"][rung] = [[e.driver_loss for e in row] for row in rows]
    by_tier = {}
    for rung in RUNGS:
        by_tier[rung] = {}
        for tier in TIERS_KEY:
            scale = _scale(results["rungs"][rung], tier)
            if scale is None:
                continue
            found = [cache.store[(d, scale)].rungs[rung] for d in test] if scale > 0 else None
            by_tier[rung][tier] = {
                "scale": scale,
                "slot_loss": None if found is None else [e.slot_loss for e in found],
                "driver_loss": None if found is None else [e.driver_loss for e in found],
            }
    record = {
        "unit": unit,
        "configurations": list(configurations),
        "days": {
            "date": [p.date for p in prepared],
            "site": [p.site for p in prepared],
            "role": list(roles),
            "cars": [p.n_cars for p in prepared],
            "offers": [not p.abstains for p in prepared],
            "certified_value_kwh": [float(p.value) for p in prepared],
            "ceiling_kwh": {
                name: [_float_or_none(ceiling_values(c, name)) for c in ceilings]
                for name in CEILINGS
            },
        },
        "calibration": {"days": offering, "scales": scales, **calibration_losses},
        "test": {"days": test, "rungs": by_tier},
    }
    if resampled is not None:
        record["resampled"] = resampled_section(resampled)
    return record


def _float_or_none(value) -> float | None:
    return None if value is None else float(value)


def records(
    groups: Sequence,
    plans: dict,
    prepared: dict,
    ceilings: dict,
    results: dict,
    resampled: dict | None = None,
) -> list:
    """One record per calibrated job, in matrix order and then unit order (module text).
    `results[(head name, unit)]` is what `gate_calibrate.run_job` returned for the job;
    `resampled[(head name, unit)]`, its `day_resample.Resampled`, when the run made them."""
    out = []
    for group in groups:
        names = [c.name for c in group]
        for unit, plan in plans.items():
            job = (group[0].name, unit)
            if job not in results:
                continue
            found, _, _, cache = results[job]
            extra = None if resampled is None else resampled[job]
            out.append(
                job_record(
                    unit, names, plan.roles, prepared[job], ceilings[unit], found, cache, extra
                )
            )
    return out


def text(found: list) -> str:
    """The file: the definition and the records, compact JSON on one line."""
    document = {"definition": DEFINITION, "rungs": list(RUNGS), "jobs": found}
    return json.dumps(document, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
