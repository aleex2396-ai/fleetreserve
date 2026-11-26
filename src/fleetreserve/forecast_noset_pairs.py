"""Paired comparison of the arms of the forecast trial `noset` (`forecast_noset`; a trial made
after the sealed run, `after_sealed_run`).

Input: per arm, the results documents of `site_dial_whatif.run`, one per seed (the arm `tail` has
one document, the neural arms `tuned`, `inputs` and `noset` one per seed), all made on the same
matrix, units, level, dial family and seeded splits. Each document holds, per unit that was split
and per split (a seed of the splits, `dial_choice.split_indices`) and tier, two things that
`compare` reads, each under its own line:

- `plan_rule` (the judged line): the offer per counting test day at the dial that the project's
  development rule picks for the run (plan section 6, `dial_splits.pick_dial`: the dial with the
  largest mean offer over the run's seeded splits at the unit and tier, a tie to the smaller dial;
  `site_dial_summary.by_dial`), each split's offer being the one `site_dial_whatif` records for that
  dial (`by_dial`: the scale calibrated on all of the split's calibration days). The rule is applied
  to every run alike, so each seed of a neural arm has its own dial. It reads the offers of the
  test days of the development units, as every development figure of the project does.
- `calibration_days_only` (the check line): the arm S1 of `site_dial_whatif`, the offer of the dial
  chosen from half of the split's calibration days, its scale calibrated on the other half and
  scored on the test days (module text of `site_dial_whatif`). No test day reaches its choice.

Both lines come from the one measurement of the run (the dials and scales evaluated once), so the
second line needs no further fit and no further evaluation.

`compare` reads, per unit and tier (the tiers are never pooled or averaged) and per line, for each
arm:

- `seed_figures`: the mean over the splits of each seed's offer; for `tail` one figure; with
  `seed_min` and `seed_max` (the spread of this trial's seeds);
- `per_split_mean`: per split, the mean over the arm's seeds of the offer (for `tail` its own); the
  mean over the splits of it is the arm's `mean_offer`, also the mean of the seed figures;
- `dials_picked` (line `plan_rule` only): each seed's dial under the rule;

and the paired cells (`forecast_mix_pairs.paired_cell`, its `main` being the comparison arm and
its `trial` the arm named first), per split by seed, differences in kWh per counting test day:

- `against_tuned`: `noset` against `tuned` (one difference: the set encoder; the judged cell);
- `against_inputs`: `noset` against `inputs`;
- `against_tail`: `noset` against `tail`;
- `inputs_against_tail`: `inputs` against `tail`, information only.

A cell is None when an arm is missing (it failed) or a document has no split figures for the unit
(the unit could not be split, or failed in that run). A unit that cannot be split has its status
only. `ValueError` when two documents differ in what they ran (`forecast_mix_pairs.check_same_run`)
or when seeds of the splits differ.
"""

import numpy as np

from fleetreserve import forecast_mix_pairs, site_dial_summary

ARMS = ("tail", "tuned", "inputs", "noset")
LINES = ("plan_rule", "calibration_days_only")
PLAN_RULE, CHECK = LINES
TIERS = ("A", "B")
PAIRS = {
    "against_tuned": ("noset", "tuned"),
    "against_inputs": ("noset", "inputs"),
    "against_tail": ("noset", "tail"),
    "inputs_against_tail": ("inputs", "tail"),
}


def dial_key(dial: float) -> str:
    """The key of a dial in a split's `by_dial` (the one `site_dial_whatif` writes)."""
    return f"{dial:g}"


def plan_dial(document: dict, unit: str, tier: str) -> float | None:
    """The dial that the plan's development rule picks for the run at the unit and tier: the one
    with the largest mean offer over the run's splits (`site_dial_summary.by_dial`, which calls
    `dial_splits.pick_dial`); None when the document holds no splits of the unit."""
    entry = document["units"].get(unit)
    if entry is None or "splits" not in entry:
        return None
    return site_dial_summary.by_dial(entry["splits"], document["run"]["dials"])[tier][
        "dial_by_the_plans_rule"
    ]


def split_offers(document: dict, unit: str, tier: str, line: str) -> list[dict] | None:
    """The offers of one unit and tier per split on `line`, [{seed, offer_kwh}] in the order of
    the splits; None when the document holds no splits of the unit. `plan_rule`: each split's
    offer at the run's dial under the plan's rule; `calibration_days_only`: the arm S1."""
    entry = document["units"].get(unit)
    if entry is None or "splits" not in entry:
        return None
    if line == CHECK:
        return [
            {"seed": r["seed"], "offer_kwh": r["tiers"][tier]["S1"]["offer_kwh_per_test_day"]}
            for r in entry["splits"]
        ]
    if line == PLAN_RULE:
        key = dial_key(plan_dial(document, unit, tier))
        return [
            {"seed": r["seed"], "offer_kwh": r["tiers"][tier]["by_dial"][key]}
            for r in entry["splits"]
        ]
    raise ValueError(f"a line is one of {LINES}, got {line!r}")


def mean_over_seeds(lists: list[list[dict]]) -> list[dict]:
    """Per split the mean of the lists' offers; `ValueError` unless every list has the same seeds
    of the splits in the same order."""
    seeds = [s["seed"] for s in lists[0]]
    for other in lists[1:]:
        if [s["seed"] for s in other] != seeds:
            raise ValueError("the runs' splits are not of the same seeds in the same order")
    return [
        {"seed": seed, "offer_kwh": float(np.mean([found[i]["offer_kwh"] for found in lists]))}
        for i, seed in enumerate(seeds)
    ]


def figure(offers: list[dict]) -> float | None:
    """The mean offer over the splits of `offers`; None for no split."""
    return float(np.mean([s["offer_kwh"] for s in offers])) if offers else None


def arm_block(documents: list[dict] | None, unit: str, tier: str, line: str) -> dict | None:
    """One arm's figures at a unit, tier and line (module text); None when the arm is missing or
    one of its documents holds no splits of the unit."""
    if not documents:
        return None
    lists = [split_offers(d, unit, tier, line) for d in documents]
    if any(found is None for found in lists):
        return None
    mean = mean_over_seeds(lists)
    figures = [figure(found) for found in lists]
    block = {
        "seeds": len(lists),
        "seed_figures": figures,
        "seed_min": min(figures),
        "seed_max": max(figures),
        "per_split_mean": mean,
        "mean_offer": figure(mean),
    }
    if line == PLAN_RULE:
        block["dials_picked"] = [plan_dial(d, unit, tier) for d in documents]
    return block


def line_block(documents: dict, unit: str, tier: str, line: str) -> dict:
    """The arms and the paired cells of one unit, tier and line (module text)."""
    blocks = {arm: arm_block(documents.get(arm), unit, tier, line) for arm in ARMS}
    block = {
        "arms": {
            arm: None if one is None else {k: v for k, v in one.items() if k != "per_split_mean"}
            for arm, one in blocks.items()
        },
        "seed_figures": {
            arm: None if one is None else one["seed_figures"] for arm, one in blocks.items()
        },
    }
    for name, (first, second) in PAIRS.items():
        one, other = blocks[first], blocks[second]
        block[name] = (
            None
            if one is None or other is None
            else forecast_mix_pairs.paired_cell(other["per_split_mean"], one["per_split_mean"])
        )
    return block


def compare(documents: dict) -> dict:
    """The comparison (module text). `documents`: arm -> list of results documents; an arm that
    failed is left out or has an empty list. Per unit that was split and per tier: the two lines
    under `tiers[tier][line]`."""
    every = [d for found in documents.values() for d in found]
    if not every:
        raise ValueError("no run to compare")
    for other in every[1:]:
        forecast_mix_pairs.check_same_run(every[0], other)
    found: dict = {}
    for unit, entry in every[0]["units"].items():
        out = {"status": entry["status"], "counting_days": entry["counting_days"]}
        found[unit] = out
        if entry["status"] != "calibratable":
            continue
        out["tiers"] = {
            tier: {line: line_block(documents, unit, tier, line) for line in LINES}
            for tier in TIERS
        }
    return found
