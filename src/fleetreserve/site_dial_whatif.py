"""What the offer would be if each unit's own calibration days chose the driver-risk dial: a
what-if on the development units (plan sections 2 and 6).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.site_dial_whatif --model <name> --fixed-dials <unit>=<dial> [...]
        [--matrix <file>] [--splits N] [--workers N] [--tables <folder>] [--truth-run <folder>]

It is a what-if, not a run of the plan: the results' `label` and the summary's first line say so.
It forms no verdict and no pass-mark tool reads it. Plan section 6 fixes one dial per site type
before the tag, from development data (`dial_choice`); here, on every development split, the
unit's own calibration days choose the dial instead, by three variants, scored beside the fixed
dial on the same split.

Inputs, checked as `dial_choice` checks them: `--model` (`fleetreserve.models.<name>`, checked,
fitted and turned into the forecaster of every certificate and re-solve by
`dial_choice.make_factory`, at the plan's levels up to the family's kappa); `--matrix` (default
the committed gate matrix), whose gate units are the units measured and whose dial family
(`dial_choice.find_family`: kappa 0.3 and the dials 0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1
in the committed matrix) gives the dials; a rotation what-if matrix is refused; `--splits` N
(default 200), seeds 1 to N, made by `dial_choice.split_indices`; `--fixed-dials`, one
`<unit>=<dial>` for every gate unit, each a dial of the family (the dial the plan fixes for the
unit's site type); the truth tables, checked by `gate_run.check_inputs`.

Measurement, per gate unit that the matrix's rule splits and per dial of the family: every day
prepared (`gate_day.prepare_day`) and every offering day evaluated once at each scale of
`dial_splits.GRID` (`dial_choice.measure`: the one-way rung's day-level slot and driver losses,
the certified value at scale 1). The configuration without rows, the oracle partners and the
ceilings are not measured (reading D7). Calibration everywhere is `dial_splits`' on-grid rule on
the grid up to 1 (reading D1), per tier: tier A at slot alpha 0.10 and driver alpha 0.05, tier B
at slot alpha 0.10 alone (the driver loss reported).

Why the choice needs care. Plan section 6 (T6) bounds the expected test loss at the calibrated
scale for a dial fixed before the calibration days are seen, when the calibration and test days
are exchangeable (order-free). Choosing the dial on the calibration days and calibrating the scale
on the same days is a selection that argument does not cover: the dial taken is the one whose
calibrated scale came out high on those days, partly by chance, so its test losses can exceed the
levels. Per split and tier, four arms (`split_arms`):

- fixed: the unit's dial of `--fixed-dials`, calibrated on all the split's calibration days at
  the plan's levels and scored on its test days (`dial_splits.score_split`): the figure
  `dial_choice` gives for that dial on the same split.
- S1, split. The split's calibration days are cut in two by a seeded draw that looks at no loss
  (`pick_part`, reading D2): the choosing part (half, rounded down) and the calibrating part (the
  rest). Each dial is calibrated on the choosing part at the plan's levels and the choice rule
  below takes one; its scale is calibrated on the calibrating part alone, at the plan's levels,
  and scored on the test days. Validity: the dial is a function of the choosing part only; given
  those days, the calibrating days and the test days are still exchangeable with one another, so
  T6's argument holds at the chosen dial exactly as at a dial fixed in advance, and so on average
  over the choice. Price: the scale is calibrated on about half the days, so the abstain rule
  needs 19 offering days in the calibrating part (driver alpha 0.05) and its +1 weighs twice.
- S2, union bound. Each of the K dials of the family (K = 8 in the committed matrix) is calibrated
  on all the split's calibration days at the levels divided by K (`union_levels`: slot 0.10 / K
  in both tiers, driver 0.05 / K in tier A; tier B has no driver level to divide); the choice rule
  takes one, scored at its own scale. Validity: for each dial alone T6 bounds the expected test
  loss by level / K; losses are never negative, so the test loss of whichever dial is taken is at
  most the sum of the K dials' test losses, whose expectation is at most K x level / K = the
  level, for the slot and the driver loss each, whatever the rule of choice. Price: calibrating
  at all needs (n + 1) x level / K >= 1 (`crc.enough_days`): with K = 8, n >= 159 offering
  calibration days for the driver loss in tier A (level 0.00625) and n >= 79 for the slot loss
  (0.0125). With the plan's 100 calibration days a split has at most 100, so tier A under S2
  abstains on every split of every unit, by arithmetic.
- S3, naive. Each dial calibrated on all the split's calibration days at the plan's levels; the
  choice rule takes one, scored at its own scale. NO GUARANTEE ARGUMENT: the dial's scale is
  calibrated on the very days that chose it. An upper reference, never a candidate.

The choice rule (`choose`, reading D3): on the choosing days (S1: the choosing part; S2, S3: all
the split's calibration days), each dial's calibrated offer is its scale times the mean certified
value over those counting days (0 on a day that abstains); the largest is taken; offers within
`dial_splits.TIE` kWh of the largest are a tie, which goes to the fixed dial when it is among
them, else to the smallest dial. When no dial has an offer above 0 on the choosing days, S1 takes
the fixed dial (a rule fixed in advance, so S1 stays valid) and S2 and S3 take no dial (scale 0,
offer 0; counted as "none").

Beside each variant, its reference: the fixed dial calibrated the variant's way (S1 on the
calibrating part at the plan's levels; S2 on all the calibration days at the divided levels; S3
as the fixed arm). Variant minus reference is the effect of the choice alone; reference minus the
fixed arm is the price of the variant's device (0 for S3). When every dial has the same losses
and values the choice has no effect, and variant minus reference is exactly 0.

Reading the fixed arm: the plan fixes the dial as the one with the largest mean calibrated tier A
offer over the 200 development splits of the unit that sets it for its site type (workplaces:
NREL from May 2019; Norway and Korea: the pooled Norwegian development sites). At that unit, on
the same splits, the fixed dial is the best single dial in hindsight, its test days included;
the variants choose from calibration days only, so their paired difference there is read against
an in-sample best. A unit whose type's dial was set on another unit has no such advantage. Per
unit the results give the dial that the plan's rule picks on this run's own splits
(`site_dial_summary.by_dial`), beside the fixed dial given, without acting on it.

Timeouts are recorded outcomes, never read as 0: a day whose certificate times out under a dial
abstains under that dial (`gate_day`; value 0, no loss, in no mean), counted per dial; a re-solve
that times out leaves the rung on its predecessor's plan (`rungs`) and the day enters with the
losses that replay gives, counted once per evaluated (day, scale).

Readings, each with the alternative not taken:

D1. Calibration is `dial_choice`'s on-grid rule on the grid up to 1 (no bisection), the one-way
    rung, so the fixed arm is `dial_choice`'s figure for that dial (the measurement that fixed the
    dial). [Alternative: the gate's search, bisection on the grid up to 2; it evaluates other
    scales per arm and split.]
D2. S1's choosing part is half the split's calibration days, rounded down, drawn by
    `day_split.split_days` under the label "<unit> | dial pick" and the split's seed; the
    "earlier part" of the brief is read as the first part of a seeded random order. [Alternative:
    the earlier dates. Under exchangeable days that is valid too, but the calibrating part then
    lies in the later half of the period while the test days span all of it.]
D3. The choice rule above; a tie goes to the fixed dial, which the choosing days give no reason to
    leave. [Alternative: ties to the smaller dial always, as `dial_splits.pick_dial`.]
D4. S2's K is the number of dials in the matrix's family; the configuration without rows is not a
    dial. [Alternative: an unequal division of the levels, which the brief does not state.]
D5. The one-way rung only, the rung on which the plan fixes the dial. [Alternative: the fixed plan
    too, whose losses each evaluation also gives.]
D6. The fixed dial per unit is given on the command line. [Alternative: taken from this run's own
    splits by the plan's rule at the unit that sets it, which ties it to the same splits.]
D7. No ceilings, no oracle partners, no configuration without rows: the figures need none of them.
    [Alternative: the share of the best-causal ceiling, as `dial_choice` gives it, at one solve
    per day.]
D8. The interval of a paired difference is its 5th to 95th percentile over the splits, as the
    centre-scale what-if reports it; the splits share days, so it is a spread over splits, not a
    confidence interval of the mean.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set, the summary also on standard output:
- `site-dial-whatif.json`: every number (`definitions`, `run`, per unit the fixed dial, the
  offering days and evaluations per dial, each split's arms per tier, the figures over splits and
  per dial; `timeouts`, `failed_units`, the model, the inputs, the matrix's hash, the forecast
  table's hash);
- `site-dial-whatif-summary.txt`: the summary (`site_dial_summary.summary_lines`);
- `site-dial-whatif-forecasts.csv`: the model's forecast table;
- `site-dial-whatif-timing.json`: wall-clock seconds, kept apart.

Exit status: 0 every unit that the rule splits was measured; 1 the model was refused or failed, a
day could not be built from the tables, or a unit failed (recorded, the others go on); 2 could not
run (a bad argument, matrix or model name, a matrix without the dial family, a rotation what-if
matrix, a fixed dial missing or not a dial of the family, a model fitted on a scored unit, a table
missing, no verified truth build at this commit, a table differing from it).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from fleetreserve import crc, dial_choice, dial_splits, gate_run, learn_run, site_dial_summary
from fleetreserve.day_split import split_days
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import write_table
from fleetreserve.gate_config import Matrix, MatrixError, load
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

MATRIX = dial_choice.MATRIX
SPLITS = dial_choice.SPLITS
RUNG = dial_choice.RUNG
RESULTS_NAME = "site-dial-whatif.json"
SUMMARY_NAME = "site-dial-whatif-summary.txt"
TABLE_NAME = "site-dial-whatif-forecasts.csv"
TIMING_NAME = "site-dial-whatif-timing.json"
PICK_LABEL = "{unit} | dial pick"  # reading D2
VARIANTS = site_dial_summary.VARIANTS
ARMS = site_dial_summary.ARMS
TIERS = dial_splits.TIERS
LABEL = site_dial_summary.LABEL


# --- the arithmetic of one split -----------------------------------------------------------------


def pick_part(unit: str, seed: int, keys, calibration) -> list[int]:
    """S1's choosing part (reading D2): half of the day indices `calibration`, rounded down, drawn
    by `day_split.split_days` from the days' keys under the label `PICK_LABEL` and `seed`; in
    day order. Empty when the split has fewer than two calibration days."""
    calibration = list(calibration)
    count = len(calibration) // 2
    if count < 1:
        return []
    drawn = split_days(
        PICK_LABEL.format(unit=unit),
        seed,
        [keys[d] for d in calibration],
        n_calibration=count,
        min_days=count + 1,
    )
    chosen = set(drawn.calibration)
    return sorted(d for d in calibration if keys[d] in chosen)


def union_levels(tier: str, k: int) -> tuple[float, float]:
    """S2's levels (module text): the tier's slot alpha over `k`, and its driver alpha over `k`
    in tier A (tier B has no driver level to divide: its driver alpha stays 1)."""
    if k < 1:
        raise ValueError(f"the family must have at least one dial, got {k}")
    alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
    return alpha_slot / k, (alpha_driver / k if tier == "A" else alpha_driver)


def _at_levels(alpha_slot: float, alpha_driver: float):
    """A `dial_splits.UnitLosses.rule` that calibrates on the grid at these levels, not the
    tier's."""

    def rule(slot, driver, used, tier_slot, tier_driver, grid):
        return dial_splits.on_grid_scale(slot, driver, alpha_slot, alpha_driver, grid)

    return rule


def calibrated_offer(
    losses: dial_splits.UnitLosses, days, alpha_slot: float, alpha_driver: float
) -> float:
    """A dial's calibrated offer on the choosing days `days` (module text, the choice rule): the
    on-grid scale on those that offer, times the mean certified value over all of them."""
    days = np.asarray(days, dtype=int)
    if not len(days):
        return 0.0
    used = days[losses.offering[days]]
    scale, _ = dial_splits.on_grid_scale(
        losses.slot[used], losses.driver[used], alpha_slot, alpha_driver
    )
    return float(scale * losses.value[days].mean())


def choose(offers: dict, fixed: float) -> float | None:
    """The dial the choice rule takes from `offers` (dial -> calibrated offer on the choosing
    days; reading D3), or None when no offer is above 0."""
    best = max(offers.values())
    if best <= 0:
        return None
    tied = [dial for dial, offer in offers.items() if offer >= best - dial_splits.TIE]
    return fixed if fixed in tied else min(tied)


def _key(dial: float) -> str:
    return f"{dial:g}"


def _arm(dial, score: dial_splits.SplitScore) -> dict:
    return {
        "dial": dial,
        "scale": score.scale,
        "reason": score.reason,
        "offering_calibration_days": score.offering_calibration_days,
        "offering_test_days": score.offering_test_days,
        "offer_kwh_per_test_day": score.offer_kwh,
        "slot_loss": score.slot_loss,
        "driver_loss": score.driver_loss,
    }


def _variant(dial, offers, score, reference, fixed_offer: float) -> dict:
    out = _arm(dial, score)
    out["choosing_offers"] = {_key(d): offers[d] for d in sorted(offers)}
    out["no_dial_offered"] = max(offers.values()) <= 0
    out["reference_offer_kwh_per_test_day"] = reference.offer_kwh
    out["minus_fixed"] = score.offer_kwh - fixed_offer
    out["minus_reference"] = score.offer_kwh - reference.offer_kwh
    return out


def split_arms(losses: dict, fixed: float, seed: int, calibration, test, pick, tier: str) -> dict:
    """The four arms of one split and tier (module text), and `by_dial`: each dial's offer when
    calibrated on all the calibration days at the plan's levels. `losses`: dial ->
    `dial_splits.UnitLosses` of one unit; `fixed`: a key of `losses`; `calibration`, `test`,
    `pick` (the choosing part, within `calibration`): day indices."""
    if fixed not in losses:
        raise ValueError(f"the fixed dial {fixed} is not a dial of the family {sorted(losses)}")
    calibration = [int(d) for d in calibration]
    chosen_days = {int(d) for d in pick}
    if not chosen_days <= set(calibration):
        raise ValueError("the choosing part must lie within the calibration days")
    rest = [d for d in calibration if d not in chosen_days]
    pick = sorted(chosen_days)
    alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
    by_dial = {
        d: dial_splits.score_split(one, seed, calibration, test, tier) for d, one in losses.items()
    }
    fixed_score = by_dial[fixed]
    out = {"fixed": _arm(fixed, fixed_score)}

    offers = {d: calibrated_offer(one, pick, alpha_slot, alpha_driver) for d, one in losses.items()}
    dial = choose(offers, fixed)
    taken = fixed if dial is None else dial  # S1 falls back to the fixed dial
    score = dial_splits.score_split(losses[taken], seed, rest, test, tier)
    reference = dial_splits.score_split(losses[fixed], seed, rest, test, tier)
    out["S1"] = _variant(taken, offers, score, reference, fixed_score.offer_kwh)

    union_slot, union_driver = union_levels(tier, len(losses))
    rule = _at_levels(union_slot, union_driver)
    offers = {
        d: calibrated_offer(one, calibration, union_slot, union_driver) for d, one in losses.items()
    }
    dial = choose(offers, fixed)
    score = dial_splits.score_split(
        replace(losses[fixed if dial is None else dial], rule=rule), seed, calibration, test, tier
    )
    reference = dial_splits.score_split(
        replace(losses[fixed], rule=rule), seed, calibration, test, tier
    )
    out["S2"] = _variant(dial, offers, score, reference, fixed_score.offer_kwh)

    offers = {
        d: calibrated_offer(one, calibration, alpha_slot, alpha_driver) for d, one in losses.items()
    }
    dial = choose(offers, fixed)
    score = by_dial[fixed if dial is None else dial]
    out["S3"] = _variant(dial, offers, score, fixed_score, fixed_score.offer_kwh)
    out["by_dial"] = {_key(d): by_dial[d].offer_kwh for d in sorted(by_dial)}
    return out


def split_record(unit: str, losses: dict, fixed: float, keys, seed: int, calibration, test):
    """One split's arms under both tiers, with the choosing part drawn by `pick_part`."""
    pick = pick_part(unit, seed, keys, calibration)
    return {
        "seed": seed,
        "choosing_days": len(pick),
        "calibrating_days": len(calibration) - len(pick),
        "tiers": {
            tier: split_arms(losses, fixed, seed, calibration, test, pick, tier) for tier in TIERS
        },
    }


# --- one unit ------------------------------------------------------------------------------------


def measure_unit(runner, name: str, dials, n_days: int) -> tuple[dict, dict]:
    """Per dial configuration of `dials`, the unit's `dial_splits.UnitLosses` (module text,
    measurement), and per dial the offering days, timeouts and evaluations."""
    losses, records = {}, {}
    for configuration in dials:
        prepared = runner.map(
            gate_run._prepare_task, [(name, i, configuration) for i in range(n_days)]
        )
        found, cache = dial_choice.measure(
            prepared,
            [None] * n_days,
            lambda pairs: runner.map(gate_run._evaluate_task, pairs),
        )
        evaluations = dial_choice._evaluations(cache)
        losses[configuration.driver_risk] = found
        records[_key(configuration.driver_risk)] = {
            "configuration": configuration.name,
            "offering_days": int(found.offering.sum()),
            "certificate_timeouts": sum(p.certificate_outcome == "timeout" for p in prepared),
            "evaluations": evaluations,
        }
    return losses, records


# --- one run -------------------------------------------------------------------------------------


def definitions(k: int) -> dict:
    union = {tier: union_levels(tier, k) for tier in TIERS}
    return {
        "label": LABEL,
        "fixed": (
            "the unit's dial of --fixed-dials, calibrated on all the split's calibration days at "
            "the plan's levels and scored on its test days: dial_choice's figure for that dial"
        ),
        "S1": (
            "split: a seeded draw that looks at no loss cuts the calibration days into a choosing "
            "part (half, rounded down) and a calibrating part; the dial with the largest "
            "calibrated offer on the choosing part is taken and its scale calibrated on the "
            "calibrating part alone at the plan's levels. Valid (plan section 6, T6): the dial is "
            "a function of the choosing part only, and given it the calibrating and test days "
            "stay exchangeable"
        ),
        "S2": (
            f"union bound over the {k} dials: each calibrated on all the calibration days at the "
            f"levels divided by {k} (tier A slot {union['A'][0]:g}, driver {union['A'][1]:g}; "
            f"tier B slot {union['B'][0]:g}), the largest calibrated offer taken. Valid: the "
            "taken dial's test loss is at most the sum over the dials, whose expectation is at "
            f"most {k} x level / {k} = the level, for any rule of choice. Calibrating at all needs "
            f"at least {crc.min_days(union['A'][1])} offering calibration days in tier A and "
            f"{crc.min_days(union['B'][0])} in tier B"
        ),
        "S3": (
            "naive: each dial calibrated on all the calibration days at the plan's levels, the "
            "largest calibrated offer taken. NO GUARANTEE ARGUMENT: the scale is calibrated on "
            "the days that chose the dial; an upper reference only"
        ),
        "choice_rule": (
            "each dial's calibrated offer on the choosing days is its on-grid scale times the mean "
            "certified value over those counting days; the largest is taken; a tie (within "
            f"{dial_splits.TIE} kWh) goes to the fixed dial when it is tied, else to the smallest "
            "dial; no offer above 0: S1 takes the fixed dial, S2 and S3 take none (offer 0)"
        ),
        "reference": (
            "the fixed dial calibrated the variant's way (S1 on the calibrating part; S2 at the "
            "divided levels; S3 as the fixed arm); variant minus reference is the effect of the "
            "choice alone, reference minus fixed the price of the variant's device"
        ),
        "fixed_dial_in_hindsight": (
            "at the unit whose 200 development splits set its site type's dial (workplaces: NREL "
            "from May 2019; Norway and Korea: the pooled Norwegian development sites) the fixed "
            "dial is the best single dial in hindsight over the same splits, test days included; "
            "the variants choose from calibration days only"
        ),
        "figures": (
            "per split and tier: the offer per counting test day (scale x mean certified value; "
            "0 at scale 0), the mean slot and driver loss over the offering test days at the "
            "arm's scale (raw day-level losses of the one-way rung; null at scale 0 or with no "
            "offering test day); over splits: mean, 5th and 95th percentile (linear "
            "interpolation) over the splits where defined; paired differences by split, with "
            f"the splits higher, equal (within {dial_splits.TIE} kWh) and lower; the interval "
            "is a spread over splits, which share days, not a confidence interval"
        ),
        "timeouts": (
            "a certificate that times out makes its day abstain under that dial (counted per "
            "dial); a re-solve that times out leaves the rung on its predecessor's plan (counted "
            "once per evaluated day and scale)"
        ),
    }


def run(matrix: Matrix, units: dict, factory, family, fixed: dict, *, splits, workers):
    """Everything but the input checks and the writing of files: (document, timing). `fixed`:
    gate unit -> a dial of the family (`fixed_dials`)."""
    started = time.perf_counter()
    _, dials = family
    values = [c.driver_risk for c in dials]
    seeds = list(range(1, splits + 1))
    runner = gate_run.Runner(workers, units, matrix, factory)
    entries, failed, timeouts, timing = {}, {}, Counter(), {"units": {}}
    try:
        for name in matrix.gate_units:
            days = units[name]
            plan = gate_run.plan_unit(name, days, matrix)
            entry = {
                "counting_days": len(plan.keys),
                "calibration_days_used": plan.calibration_days,
                "test_days": plan.test_days,
                "status": plan.status,
                "fixed_dial": fixed[name],
            }
            entries[name] = entry
            if plan.status != "calibratable":
                continue
            begun = time.perf_counter()
            try:
                losses, per_dial = measure_unit(runner, name, dials, len(plan.keys))
            except Exception as problem:  # recorded, the other units go on
                failed[name] = f"{type(problem).__name__}: {problem}"
                entry["failed"] = failed[name]
                continue
            for found in per_dial.values():
                timeouts["certificates"] += found["certificate_timeouts"]
                timeouts["re_solves"] += found["evaluations"]["re_solves_of_the_one_way_rung"].get(
                    "timeout", 0
                )
            measured = time.perf_counter()
            records = [
                split_record(name, losses, fixed[name], plan.keys, seed, cal, test)
                for seed, cal, test in dial_choice.split_indices(name, days, matrix, seeds)
            ]
            entry["dials"] = per_dial
            entry["splits"] = records
            entry["over_splits"] = site_dial_summary.over_splits(records, values)
            entry["by_dial"] = site_dial_summary.by_dial(records, values)
            timing["units"][name] = {
                "measurement_seconds": measured - begun,
                "split_seconds": time.perf_counter() - measured,
            }
    finally:
        runner.close()
    document = {
        "label": LABEL,
        "definitions": definitions(len(values)),
        "run": {
            "splits": splits,
            "seeds": [seeds[0], seeds[-1]],
            "split_seed": matrix.split_seed,
            "kappa": dials[0].kappa,
            "rung": RUNG,
            "dials": values,
            "scales": list(dial_splits.GRID),
            "tiers": list(TIERS),
            "arms": list(ARMS),
            "union_levels": {tier: list(union_levels(tier, len(values))) for tier in TIERS},
            "fixed_dials": dict(fixed),
            "n_calibration": matrix.n_calibration,
            "calibration_half_below": matrix.calibration_half_below,
        },
        "units": entries,
        "timeouts": {key: timeouts.get(key, 0) for key in ("certificates", "re_solves")},
        "failed_units": sorted(f"{name}: {why}" for name, why in failed.items()),
    }
    timing["total_seconds"] = time.perf_counter() - started
    return document, timing


def fixed_dials(items, matrix: Matrix, dials) -> dict:
    """`--fixed-dials` as gate unit -> dial; `MatrixError` (exit 2) unless every gate unit has one
    dial of the family and no other unit is named."""
    values = {c.driver_risk for c in dials}
    found = {}
    for item in items:
        unit, sign, text = item.partition("=")
        if not sign:
            raise MatrixError(f"--fixed-dials takes <unit>=<dial>, got {item!r}")
        if unit not in matrix.gate_units:
            raise MatrixError(f"--fixed-dials: {unit!r} is not a gate unit of the matrix")
        if unit in found:
            raise MatrixError(f"--fixed-dials: {unit!r} is named twice")
        try:
            dial = float(text)
        except ValueError:
            raise MatrixError(f"--fixed-dials: {text!r} is not a number") from None
        if dial not in values:
            raise MatrixError(
                f"--fixed-dials: {dial:g} is not a dial of the family {sorted(values)}"
            )
        found[unit] = dial
    missing = [u for u in matrix.gate_units if u not in found]
    if missing:
        raise MatrixError(f"--fixed-dials: no fixed dial for {missing}")
    return {u: found[u] for u in matrix.gate_units}


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixed-dials", nargs="+", required=True)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--splits", type=int, default=SPLITS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if args.splits < 1:
            raise MatrixError("--splits must be at least 1")
        matrix = load(args.matrix)
        if matrix.rotation_whatif:
            raise MatrixError("a rotation what-if matrix (rotation_whatif) is not measured here")
        family = dial_choice.find_family(matrix)
        fixed = fixed_dials(args.fixed_dials, matrix, family[1])
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
    try:
        factory, table = dial_choice.make_factory(
            args.model, model, matrix, units, fit_frames, family[0].kappa
        )
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: the model could not be scored: {problem}")
        return 1
    document, timing = run(
        matrix, units, factory, family, fixed, splits=args.splits, workers=args.workers
    )
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
    text = "\n".join(site_dial_summary.summary_lines(document)) + "\n"
    print(text, end="")
    if run_dir:
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / TIMING_NAME).write_text(_json(timing))
    return 1 if document["failed_units"] else 0


if __name__ == "__main__":
    sys.exit(main())
