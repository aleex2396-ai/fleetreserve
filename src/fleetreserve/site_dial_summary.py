"""The figures over splits and the summary text of the site-dial what-if (`site_dial_whatif`).

Pure functions: numpy only, no files, no solver.

Over splits (`over_splits`), per tier and arm (fixed, S1, S2, S3), from the split records of
`site_dial_whatif.split_record`: the offer per counting test day as mean, 5th and 95th percentile
(`centre_scale_summary.stats`: numpy's linear interpolation, over the splits where it is defined;
an offer is defined on every split, 0 at scale 0); the mean scale; the splits that offer; the
mean slot and driver loss over the offering test days, averaged over the splits where they are
defined, and the splits whose loss is above its limit (slot 0.10 in both tiers, driver 0.05 in
tier A; `crc.TOLERANCE`); how often each dial was taken ("none": S2 or S3 took no dial). For the
variants also: the splits where no dial had an offer above 0 on the choosing days; the paired
difference to the fixed arm and to the variant's reference (`site_dial_whatif`, module text), each
as mean, 5th and 95th percentile with the splits higher, equal (within `dial_splits.TIE`) and
lower; the reference's offer.

Per dial (`by_dial`), per tier: the mean over splits of the offer of that dial calibrated on all
the calibration days at the plan's levels (`dial_choice`'s figure), and the dial the plan's rule
picks from those means (`dial_splits.pick_dial`: the largest; a tie to the smaller dial).

The summary (`summary_lines`) gives the label, the run, one line per arm on what it is and its
validity, the in-hindsight note on the fixed arm, the timeouts and the failed units; then per tier
one table, a row per unit and arm. It forms no verdict.
"""

import numpy as np

from fleetreserve import crc, dial_splits
from fleetreserve.centre_scale_summary import stats
from fleetreserve.gate_config import TIER_LABELS

LABEL = (
    "WHAT-IF, not a run of the plan: each unit's own calibration days choose the driver-risk "
    "dial on every split (S1 split, S2 union bound, S3 naive with no guarantee argument), beside "
    "the dial fixed per site type; it reports numbers and decides nothing"
)
ARMS = ("fixed", "S1", "S2", "S3")
VARIANTS = ARMS[1:]
NONE = "none"


def _key(dial: float) -> str:
    return f"{dial:g}"


def _limit(tier: str) -> tuple[float, float | None]:
    alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
    return alpha_slot, (alpha_driver if tier == "A" else None)


def _over(values, limit: float) -> int:
    return sum(1 for v in values if v > limit + crc.TOLERANCE)


def compare(values) -> dict:
    """A paired difference over splits: `stats` with the splits higher, equal and lower."""
    out = stats(values)
    kept = [v for v in values if v is not None]
    out["higher"] = sum(1 for v in kept if v > dial_splits.TIE)
    out["lower"] = sum(1 for v in kept if v < -dial_splits.TIE)
    out["equal"] = len(kept) - out["higher"] - out["lower"]
    return out


def arm_figures(arms: list[dict], tier: str, dials) -> dict:
    """The figures over splits of one arm under `tier` (module text); `arms`: its split records."""
    slot_limit, driver_limit = _limit(tier)
    slot = [a["slot_loss"] for a in arms if a["slot_loss"] is not None]
    driver = [a["driver_loss"] for a in arms if a["driver_loss"] is not None]
    picked = {_key(d): 0 for d in dials}
    picked[NONE] = 0
    for a in arms:
        picked[NONE if a["dial"] is None else _key(a["dial"])] += 1
    out = {
        "splits": len(arms),
        "offer_kwh_per_test_day": stats([a["offer_kwh_per_test_day"] for a in arms]),
        "mean_scale": float(np.mean([a["scale"] for a in arms])) if arms else None,
        "splits_that_offer": sum(1 for a in arms if a["scale"] > 0),
        "splits_with_test_losses": len(slot),
        "mean_test_slot_loss": float(np.mean(slot)) if slot else None,
        "mean_test_driver_loss": float(np.mean(driver)) if driver else None,
        "slot_loss_over_limit": _over(slot, slot_limit),
        "dial_taken": picked,
    }
    if driver_limit is not None:
        out["driver_loss_over_limit"] = _over(driver, driver_limit)
    if "minus_fixed" in (arms[0] if arms else {}):
        out["no_dial_offered_on_choosing_days"] = sum(1 for a in arms if a["no_dial_offered"])
        out["minus_fixed"] = compare([a["minus_fixed"] for a in arms])
        out["reference_offer_kwh_per_test_day"] = stats(
            [a["reference_offer_kwh_per_test_day"] for a in arms]
        )
        out["minus_reference"] = compare([a["minus_reference"] for a in arms])
    return out


def over_splits(records: list[dict], dials) -> dict:
    """Per tier and arm the figures over the splits `records` (module text)."""
    if not records:
        return {}
    tiers = records[0]["tiers"]
    return {
        tier: {
            arm: arm_figures([r["tiers"][tier][arm] for r in records], tier, dials) for arm in ARMS
        }
        for tier in tiers
    }


def by_dial(records: list[dict], dials) -> dict:
    """Per tier, each dial's mean offer over splits and the dial the plan's rule picks (module
    text)."""
    if not records:
        return {}
    out = {}
    for tier in records[0]["tiers"]:
        means = {
            d: float(np.mean([r["tiers"][tier]["by_dial"][_key(d)] for r in records]))
            for d in dials
        }
        out[tier] = {
            "mean_offer_kwh_per_test_day": {_key(d): means[d] for d in dials},
            "dial_by_the_plans_rule": dial_splits.pick_dial(means),
        }
    return out


# --- the text ------------------------------------------------------------------------------------


def _number(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _figure(entry: dict, digits: int = 2) -> str:
    """One figure over splits: the value with one split, else the mean [p5, p95]."""
    if entry["splits"] == 0:
        return "-"
    if entry["splits"] == 1:
        return _number(entry["mean"], digits)
    return (
        f"{_number(entry['mean'], digits)} [{_number(entry['p5'], digits)}, "
        f"{_number(entry['p95'], digits)}]"
    )


def _paired(entry: dict) -> str:
    return f"{_figure(entry)} ({entry['higher']}/{entry['equal']}/{entry['lower']})"


def _taken(picked: dict) -> str:
    return " ".join(f"{dial}:{count}" for dial, count in picked.items() if count)


def _loss(mean, over) -> str:
    return "-" if mean is None else f"{mean:.3f} ({over})"


def _row(unit: str, arm: str, figures: dict) -> tuple:
    variant = arm in VARIANTS
    return (
        unit,
        arm,
        _figure(figures["offer_kwh_per_test_day"]),
        _paired(figures["minus_fixed"]) if variant else "-",
        _paired(figures["minus_reference"]) if variant else "-",
        _number(figures["mean_scale"], 3),
        f"{figures['splits_that_offer']}/{figures['splits']}",
        _loss(figures["mean_test_slot_loss"], figures["slot_loss_over_limit"]),
        _loss(figures["mean_test_driver_loss"], figures.get("driver_loss_over_limit", "-")),
        _taken(figures["dial_taken"]),
    )


HEAD = (
    "unit",
    "arm",
    "offer",
    "minus fixed (higher/equal/lower)",
    "minus reference (higher/equal/lower)",
    "scale",
    "offering splits",
    "slot loss (splits over)",
    "driver loss (splits over)",
    "dial taken: splits",
)


def _table(rows: list[tuple]) -> list[str]:
    widths = [max(len(row[k]) for row in rows) for k in range(len(rows[0]))]
    return [
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]


def _unit_lines(document: dict) -> list[str]:
    lines = []
    for unit, entry in document["units"].items():
        if entry["status"] != "calibratable":
            lines.append(f"{unit}: {entry['status']}")
        elif "failed" in entry:
            lines.append(f"{unit}: FAILED: {entry['failed']}")
        else:
            rule = entry["by_dial"].get("A", {}).get("dial_by_the_plans_rule")
            means = entry["by_dial"].get("A", {}).get("mean_offer_kwh_per_test_day", {})
            lines.append(
                f"{unit}: fixed dial {_key(entry['fixed_dial'])}; the plan's rule on this run's "
                f"splits (tier A) gives {'-' if rule is None else _key(rule)}; mean tier A offer "
                "by dial: " + ", ".join(f"{d} {_number(v)}" for d, v in means.items())
            )
    return lines


def summary_lines(document: dict) -> list[str]:
    """The summary text of a results document (module text)."""
    run = document["run"]
    definitions = document["definitions"]
    timeouts = document["timeouts"]
    failed = document["failed_units"]
    splits = run["splits"]
    shown = "the split's value" if splits == 1 else "mean [p5, p95] over the splits"
    lines = [
        LABEL,
        (
            f"Model {document['model']}; matrix {document['matrix']['file']}; {splits} split(s), "
            f"seeds {run['seeds'][0]} to {run['seeds'][-1]}; kappa {run['kappa']:g}; rung "
            f"{run['rung']}; dials {', '.join(_key(d) for d in run['dials'])}; on-grid scales up "
            f"to {run['scales'][-1]:g}, as dial_choice calibrates."
        ),
        *(f"{arm}: {definitions[arm]}." for arm in ARMS),
        f"Choice rule: {definitions['choice_rule']}.",
        f"Reference: {definitions['reference']}.",
        f"Fixed dial in hindsight: {definitions['fixed_dial_in_hindsight']}.",
        (
            f"Timeouts: certificates {timeouts['certificates']} (each day abstains under its "
            f"dial); re-solves {timeouts['re_solves']}."
        ),
        "Failed units: " + ("none" if not failed else "; ".join(failed)),
        *_unit_lines(document),
        (
            f"Per tier, unit and arm ({shown}): offer in kWh per counting test day; paired "
            "differences by split; mean calibrated scale; splits that offer; mean slot and driver "
            "loss over the offering test days with the splits over the limit; dials taken."
        ),
    ]
    for tier in run["tiers"]:
        lines.append(TIER_LABELS[tier])
        rows = [HEAD]
        for unit, entry in document["units"].items():
            figures = entry.get("over_splits", {}).get(tier, {})
            rows += [_row(unit, arm, figures[arm]) for arm in ARMS if arm in figures]
        lines += [f"  {line}" for line in _table(rows)]
    return lines
