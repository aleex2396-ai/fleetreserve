"""Made-up recorded scoring runs for the tests of the intervals over days (`day_intervals`).

Calibration days lose in steps: a day's slot loss is 1 at every scale above its threshold and 0
at and below it (`crc` reading R1); its driver loss is 0, or a level above a second threshold
(`calibration["driver"]`). `asked(thresholds, tier)` gives the scales `crc.calibrate_search`
itself asks for on step slot losses, so a made-up file holds the scales a run's search would have
evaluated.

`unit(...)`: one job's record in the form `day_losses` writes, for the rungs named, each rung with
the same losses, and the run's results entry it must agree with. With `resample=(seed, n)` the
record holds the resamples' pairs as the scoring run writes them: `day_resample.resample_pairs`
itself, with the made-up days' step losses as the replay (`replay`), the draws of
`day_bootstrap.draws` (or the test's own, when it replaces them) and, for `fails`, (day, scale)
pairs whose replay raises an error. `write_run(folder, jobs, module)`: a run folder of `gate_run`
(or `learn_run`) with its results file, its per-day file and a manifest that verifies once the
caller replaces the commit check.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fleetreserve import crc, day_losses, day_resample, gate_run, learn_run
from fleetreserve.dial_splits import tier_alphas
from fleetreserve.gate_config import CEILINGS
from fleetreserve.rungs import RUNGS

COMMIT = "c" * 40
RUNG = "one-way"
NAME = "made-up-configuration"
SEED = 7
RESAMPLES = 2000


def step_rows(thresholds, scales) -> list[list[float]]:
    return [[1.0 if b > t else 0.0 for b in scales] for t in thresholds]


def made_up_day(offers: bool, threshold=None, driver=None):
    """A stand-in for a prepared day: whether it offers and its step losses."""
    return SimpleNamespace(
        abstains=not offers, threshold=threshold, driver=(0.0, 0.0) if driver is None else driver
    )


def replay(day, beta: float):
    """The made-up day's evaluation at `beta`, every rung alike, as `gate_day.evaluate` gives it."""
    slot = 1.0 if beta > day.threshold else 0.0
    level, above = day.driver
    losses = SimpleNamespace(slot_loss=slot, driver_loss=level if beta > above else 0.0)
    return SimpleNamespace(rungs={rung: losses for rung in RUNGS})


def batch(fails=frozenset()):
    """An `evaluate_batch` of made-up days; a batch holding a pair of `fails` raises."""

    def evaluate(pairs):
        if any((id(day), beta) in fails for day, beta in pairs):
            raise RuntimeError("a made-up evaluation failed")
        return [replay(day, beta) for day, beta in pairs]

    return evaluate


def asked(thresholds, tier: str) -> list[float]:
    """The scales `crc.calibrate_search` asks for on step days with these thresholds."""
    found = []

    def loss_at(beta):
        found.append(beta)
        return np.array([1.0 if beta > t else 0.0 for t in thresholds]), np.zeros(len(thresholds))

    crc.calibrate_search(loss_at, len(thresholds), *tier_alphas(tier))
    return found


def _test_entry(scale, values, cars, ceilings, offers, slot, driver) -> dict:
    """The run's own test summary of one rung and tier, as `gate_calibrate.score_test` gives it."""
    offered = [beta * v for beta, v in zip([scale] * len(values), values, strict=True)]
    shares = {}
    for name in CEILINGS:
        total = sum(ceilings[name])
        shares[name] = {"ratio_of_sums": sum(offered) / total, "mean_of_daily_ratios": None}
    offering = [i for i, o in enumerate(offers) if o]
    scored = scale > 0 and offering
    return {
        "mean_offer_kwh_per_day": float(np.mean(offered)),
        "offer_kwh_per_car_per_day": sum(offered) / sum(cars),
        "share_of_ceiling": shares,
        "mean_slot_loss_offering_days": (
            float(np.mean([slot[i] for i in offering])) if scored else None
        ),
        "mean_driver_loss_offering_days": (
            float(np.mean([driver[i] for i in offering])) if scored else None
        ),
    }


def unit(
    name: str,
    calibration: dict,
    test: dict,
    scales,
    tiers: dict,
    rungs=(RUNG,),
    configurations=(NAME,),
    resample=(SEED, RESAMPLES),
    fails=(),
):
    """(record, results entry) of one made-up job.

    `calibration`: "thresholds" (one per offering calibration day, in order), "offers" (one
    flag per calibration day) and, optionally, "driver" ((level, threshold) per offering day).
    `test`: "values", "cars", "offers", "ceilings" (name -> list). `tiers`: tier -> (scale, slot
    losses, driver losses) of the test days, the run's choice. `resample`: (seed, resamples) of
    the resamples' pairs, or None for a run recorded without them. `fails`: (day position, scale)
    pairs whose replay raises."""
    n_cal = len(calibration["offers"])
    n_test = len(test["values"])
    roles = ["calibration"] * n_cal + ["test"] * n_test
    offering = [d for d in range(n_cal) if calibration["offers"][d]]
    assert len(offering) == len(calibration["thresholds"])
    drivers = calibration.get("driver") or [None] * len(offering)
    scales = sorted(set(float(b) for b in scales))
    rows = step_rows(calibration["thresholds"], scales)
    days = [made_up_day(False) for _ in range(n_cal + n_test)]
    for d, threshold, driver in zip(offering, calibration["thresholds"], drivers, strict=True):
        days[d] = made_up_day(True, threshold, driver)
    for k, offers in enumerate(test["offers"]):
        days[n_cal + k] = made_up_day(offers)
    driver_rows = [[replay(days[d], b).rungs[RUNG].driver_loss for b in scales] for d in offering]
    test_days = list(range(n_cal, n_cal + n_test))
    by_tier = {}
    for tier, (scale, slot, driver) in tiers.items():
        by_tier[tier] = {
            "scale": scale,
            "slot_loss": list(slot) if scale > 0 else None,
            "driver_loss": list(driver) if scale > 0 else None,
        }
    record = {
        "unit": name,
        "configurations": list(configurations),
        "days": {
            "date": [f"2020-01-{d + 1:02d}" for d in range(n_cal + n_test)],
            "site": ["A"] * (n_cal + n_test),
            "role": roles,
            "cars": [4] * n_cal + list(test["cars"]),
            "offers": list(calibration["offers"]) + list(test["offers"]),
            "certified_value_kwh": [10.0 if o else 0.0 for o in calibration["offers"]]
            + [float(v) for v in test["values"]],
            "ceiling_kwh": {
                c: [100.0] * n_cal + [float(v) for v in test["ceilings"][c]] for c in CEILINGS
            },
        },
        "calibration": {
            "days": offering,
            "scales": scales,
            "slot_loss": {r: rows for r in rungs},
            "driver_loss": {r: driver_rows for r in rungs},
        },
        "test": {"days": test_days, "rungs": {r: by_tier for r in rungs}},
    }
    if resample is not None:
        own = {(d, b): replay(days[d], b) for d in offering for b in scales}
        failing = frozenset((id(days[d]), float(b)) for d, b in fails)
        found = day_resample.resample_pairs(
            days, own, roles, name, *resample, tuple(tiers), batch(failing)
        )
        record["resampled"] = day_losses.resampled_section(found)
    entry = {}
    for rung in rungs:
        scored = {}
        for tier, (scale, slot, driver) in tiers.items():
            found = {
                "beta_hat": scale,
                "test": _test_entry(
                    scale,
                    test["values"],
                    test["cars"],
                    test["ceilings"],
                    test["offers"],
                    slot,
                    driver,
                ),
            }
            if tier == "A":
                scored.update(found)
            else:
                scored["tier_b"] = found
        entry[rung] = scored
    return record, {"rungs": entry}


def write_run(
    folder: Path, jobs, module: str = "fleetreserve.gate_run", rungs=(RUNG,), extra=None
) -> Path:
    """A recorded run of `module` holding `jobs` (pairs from `unit`); `extra`: more keys of its
    results document (the route's record, say)."""
    folder.mkdir(parents=True)
    configurations = {}
    for record, entry in jobs:
        for name in record["configurations"]:
            configurations.setdefault(name, {"units": {}})["units"][record["unit"]] = entry
    document = {
        "label": "gate run",
        "trial": False,
        "partial": False,
        "inputs": {"setting": "main"},
        "configurations": configurations,
        **(extra or {}),
    }
    losses = {"definition": "made up", "rungs": list(rungs), "jobs": [r for r, _ in jobs]}
    if module == "fleetreserve.gate_run":
        names = (gate_run.RESULTS_NAME, gate_run.DAY_LOSSES_NAME)
    else:
        names = (learn_run.GATE_NAME, learn_run.DAY_LOSSES_NAME)
    (folder / names[0]).write_text(json.dumps(document))
    (folder / names[1]).write_text(json.dumps(losses))
    write_manifest(folder, module)
    return folder


def write_manifest(folder: Path, module: str, exit_code: int = 0, commit: str = COMMIT) -> None:
    outputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.iterdir())
        if p.name != "manifest.json"
    }
    manifest = {
        "commit": commit,
        "command": ["uv", "run", "--frozen", "python", "-m", module],
        "start": "2020-01-01T00:00:00Z",
        "end": "2020-01-01T00:01:00Z",
        "exit_code": exit_code,
        "outputs": outputs,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
