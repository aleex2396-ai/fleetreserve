"""Paired comparison of two models' calibrated offers, split by split (the forecast trial `mix`,
`forecast_mix`; a trial made after the sealed run, `after_sealed_run`).

Input: the two runs of `dial_choice.run` on the same matrix, units, level, dial family and seeded
splits, each with its per-split offers (`dial_choice.split_offers_entry`: per unit, configuration
and tier a list of {seed, offer_kwh, scale}, in the order of the seeds) and its results document
(`dial_choice.run`'s first value). A split is the same split in both runs (same seed, so the same
calibration days and the same test days), so the offers of one seed are a pair.

`paired_cell` compares one cell (unit, configuration, tier): per split the difference `trial -
main` of the offers per counting test day, a split with no safe scale counting as an offer of 0 as
`dial_splits.summarise` counts it. It gives the two means, the mean difference and its standard
error over the splits, the points p10, p50 and p90 of the difference, the number of splits where
the trial offers more, the same and less, and the ratio of the two means (None when the main mean
is 0). The standard error is the spread over the seeded splits of one unit's days; it is not the
noise of the whole experiment (the splits share their days, and the units have few days).

`compare` reads, per unit and tier (the tiers are never pooled or averaged):

- `dials`: every configuration of the dial family and the configuration without rows, each as one
  paired cell, both models at the same configuration;
- `same_dial`: both models at the dial the main model picks (`dial_splits.pick_dial`, the plan's
  rule: the largest mean offer, a tie to the smaller dial);
- `own_dial`: each model at the dial it picks itself by that rule, as plan sections 6 and 7 give
  each model its own dial. A model with no dial that has a mean offer has none, and the cell is
  not given.

The own-dial cell is the headline of the trial (fixed before any result: `forecast_mix`, module
text); the others are reported beside it. The dial is picked on the same 200 splits that score it,
for both models alike, as `dial_choice` reports it.
"""

import numpy as np

from fleetreserve import dial_splits

CELL_FIELDS = (
    "splits",
    "mean_main_offer_kwh_per_test_day",
    "mean_trial_offer_kwh_per_test_day",
    "mean_difference_kwh_per_test_day",
    "standard_error_of_the_difference",
    "difference_p10",
    "difference_p50",
    "difference_p90",
    "splits_trial_offers_more",
    "splits_equal",
    "splits_trial_offers_less",
    "ratio_of_means_trial_to_main",
)


def _mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def paired_cell(main: list[dict], trial: list[dict]) -> dict:
    """The paired figures of one cell (module text); `ValueError` when the splits' seeds differ."""
    if [s["seed"] for s in main] != [s["seed"] for s in trial]:
        raise ValueError("the two runs' splits are not of the same seeds in the same order")
    a = np.array([s["offer_kwh"] for s in main], dtype=float)
    b = np.array([s["offer_kwh"] for s in trial], dtype=float)
    difference = b - a
    n = len(difference)
    p10, p50, p90 = (
        (float(q) for q in np.quantile(difference, dial_splits.QUANTILES)) if n else (None,) * 3
    )
    mean_main = _mean(a)
    return {
        "splits": n,
        "mean_main_offer_kwh_per_test_day": mean_main,
        "mean_trial_offer_kwh_per_test_day": _mean(b),
        "mean_difference_kwh_per_test_day": _mean(difference),
        "standard_error_of_the_difference": (
            float(difference.std(ddof=1) / np.sqrt(n)) if n > 1 else None
        ),
        "difference_p10": p10,
        "difference_p50": p50,
        "difference_p90": p90,
        "splits_trial_offers_more": int((difference > 0).sum()),
        "splits_equal": int((difference == 0).sum()),
        "splits_trial_offers_less": int((difference < 0).sum()),
        "ratio_of_means_trial_to_main": None if not mean_main else float(b.mean() / mean_main),
    }


def _by_dial(configurations: dict) -> dict:
    """Configuration name -> dial, for the configurations that measured (a dial or None)."""
    return {name: found["driver_risk"] for name, found in configurations.items()}


def _tier_offers(configurations: dict, name: str, tier: str):
    """The per-split entries of one configuration and tier, or None when it failed."""
    found = configurations.get(name, {})
    return None if "tiers" not in found else found["tiers"][tier]


def _name_of(configurations: dict, dial) -> str | None:
    """The one configuration of the dial `dial`, or None (no dial, or not one such)."""
    if dial is None:
        return None
    names = [n for n, found in configurations.items() if found["driver_risk"] == dial]
    return names[0] if len(names) == 1 else None


def _twin(configurations: dict, dial) -> str | None:
    """The one configuration of the dial `dial` (None: the one without rows), or None."""
    names = [n for n, found in configurations.items() if found["driver_risk"] == dial]
    return names[0] if len(names) == 1 else None


def check_same_run(main_document: dict, trial_document: dict, ignore=()) -> None:
    """`ValueError` unless the two results are of the same level, dials, seeds, scales and units;
    the run keys named in `ignore` are not compared (the level, for the extra line)."""
    for key in ("kappa", "dials", "seeds", "splits", "scales", "n_calibration", "tiers", "rung"):
        if key in ignore:
            continue
        if main_document["run"][key] != trial_document["run"][key]:
            raise ValueError(f"the two runs differ in run.{key}")
    if list(main_document["units"]) != list(trial_document["units"]):
        raise ValueError("the two runs measured different units")


def compare(
    main_document: dict,
    main_offers: dict,
    trial_document: dict,
    trial_offers: dict,
    ignore=(),
):
    """The paired comparison of the two runs (module text): unit -> status and, per tier, the
    cells `dials`, `same_dial` and `own_dial`. A unit that cannot be split has its status only; a
    configuration that failed in either run is left out of `dials` and named under `failed`.
    Configurations are paired by their dial, not by name, so two runs at different levels (names
    hold the level) pair; `ignore` names run keys not compared (`check_same_run`)."""
    check_same_run(main_document, trial_document, ignore)
    tiers = main_document["run"]["tiers"]
    found: dict = {}
    for unit, entry in main_document["units"].items():
        out = {"status": entry["status"], "counting_days": entry["counting_days"]}
        found[unit] = out
        if entry["status"] != "calibratable":
            continue
        m, t = main_offers[unit], trial_offers[unit]
        twin = {name: _twin(t, found["driver_risk"]) for name, found in m.items()}
        failed = sorted(
            name
            for name in m
            if "tiers" not in m[name] or twin[name] is None or "tiers" not in t[twin[name]]
        )
        out["failed"] = failed
        out["tiers"] = {}
        for tier in tiers:
            cells = {}
            for name, dial in _by_dial(m).items():
                if name in failed:
                    continue
                cells[name] = {
                    "dial": dial,
                    **paired_cell(m[name]["tiers"][tier], t[twin[name]]["tiers"][tier]),
                }
            picks = {
                "main": main_document["dial_choice"][unit][tier]["dial"],
                "trial": trial_document["dial_choice"][unit][tier]["dial"],
            }
            block = {"dials": cells, "picked_dial": picks, "same_dial": None, "own_dial": None}
            main_name = _name_of(m, picks["main"])
            trial_name = _name_of(t, picks["trial"])
            if main_name is not None and main_name not in failed:
                block["same_dial"] = {
                    "dial": picks["main"],
                    **paired_cell(m[main_name]["tiers"][tier], t[twin[main_name]]["tiers"][tier]),
                }
            if None not in (main_name, trial_name):
                main_cell = _tier_offers(m, main_name, tier)
                trial_cell = _tier_offers(t, trial_name, tier)
                if main_cell is not None and trial_cell is not None:
                    block["own_dial"] = paired_cell(main_cell, trial_cell)
            out["tiers"][tier] = block
    return found
