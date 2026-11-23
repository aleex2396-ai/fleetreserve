"""The plain-text summary of a dial-choice results document (`dial_choice`).

One table per unit and tier, rows: the configuration without driver-risk rows, then each dial in
increasing order. One line per unit and tier names the dial with the largest mean offer, labelled
as measured and not fixed. The tiers are two separate blocks; nothing is added or averaged across
them, and the no-rows row is listed but is never the dial named. A run whose family was measured at
another level (`dial_choice --kappa`) prints how the family was made (`definitions.kappa_option`)
before the calibration rule. A what-if run (`dial_choice --what-if-kappa`) begins with what it is,
in capitals ("WHAT-IF, LEVEL NOT ON THE PLAN'S GRID" or "WHAT-IF, NOT A RUN OF THE PLAN"), and the
readings it rests on. A run with a fit option (`--what-if-fit-seed`, `--what-if-fit-levels`) also
prints the seed and the levels the network was fitted at, and where each came from. A run that
averages seeds (`--what-if-fit-seeds`) also prints what is averaged (`seed_average` reading A1)
and `ENERGY_MEANS`.
"""

from fleetreserve import rotation_whatif
from fleetreserve.gate_config import TIER_LABELS
from fleetreserve.gate_summary import calibration_rule_text

LABEL = "measured, not fixed: the plan fixes the dial at the main model before the tag"
ENERGY_MEANS = (
    "The energy columns (r_hat, e_cap, e_need) are means of the members' quantiles, not "
    "quantiles of the averaged distribution, and carry no certificate on their own."
)
COLUMNS = (
    ("row", 14),
    ("offer", 8),
    ("p10", 8),
    ("p50", 8),
    ("p90", 8),
    ("offering", 9),
    ("scale", 6),
    ("slot loss", 9),
    ("slot>lim", 9),
    ("driver loss", 11),
    ("driver>lim", 10),
    ("of ceiling (offering splits only)", 10),
)
GATE_COLUMNS = (
    ("row", 14),
    ("offer", 8),
    ("ceiling", 8),
    ("share", 8),
    ("oracle", 8),
    ("oracle/offer", 13),
    ("splits without ceiling", 22),
)


def _number(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _share(value) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def _row(label: str, tier: dict, tier_name: str) -> str:
    driver_over = tier["driver_loss_over_limit"] if tier_name == "A" else None
    cells = [
        label,
        _number(tier["mean_offer_kwh_per_test_day"]),
        _number(tier["offer_p10"]),
        _number(tier["offer_p50"]),
        _number(tier["offer_p90"]),
        _share(tier["share_of_splits_that_offer"]),
        _number(tier["mean_scale"], 3),
        _number(tier["mean_test_slot_loss"], 3),
        _share(tier["slot_loss_over_limit"]["share_of_splits_that_offer"]),
        _number(tier["mean_test_driver_loss"], 3),
        "-" if driver_over is None else _share(driver_over["share_of_splits_that_offer"]),
        _share(tier["mean_share_of_best_causal_ceiling"]),
    ]
    return (
        "  "
        + " ".join(
            cell.ljust(width) for cell, (_, width) in zip(cells, COLUMNS, strict=True)
        ).rstrip()
    )


def _header() -> str:
    return "  " + " ".join(name.ljust(width) for name, width in COLUMNS).rstrip()


def _gate_cells(label: str, found: dict, ceiling: str) -> list[str]:
    by_ceiling = found["ceilings"][ceiling]
    ratio = found["oracle_to_offer_ratio"]
    return [
        label,
        _number(found["mean_offer_kwh_per_test_day"]),
        _number(by_ceiling["mean_ceiling_kwh_per_test_day"]),
        _share(by_ceiling["share_of_ceiling"]),
        _number(found["mean_oracle_offer_kwh_per_test_day"]),
        _number(ratio),
        str(by_ceiling["splits_left_out_ceiling_missing"]),
    ]


def _gate_lines(document: dict, entry: dict) -> list[str]:
    run = document["run"]
    thresholds = run["thresholds"]
    lines = [
        f" Gate reading, tier A (driver-protected), mean over all {run['splits']} splits "
        "(kWh per counting test day)"
    ]
    for ceiling in run["ceilings"]:
        first = (
            " (the ceiling of the gate run's first decision)"
            if ceiling == run["gate_ceiling"]
            else ""
        )
        lines.append(f"  ceiling {ceiling}{first}")
        lines.append("    " + " ".join(name.ljust(w) for name, w in GATE_COLUMNS).rstrip())
        for found in entry["configurations"].values():
            label = "no rows" if found["driver_risk"] is None else f"eps {found['driver_risk']:g}"
            reading = found.get("gate_reading", {})
            if "ceilings" not in reading:
                why = reading.get("failed", "no result")
                lines.append(f"    {label}: not computed ({why})")
                continue
            cells = _gate_cells(label, reading, ceiling)
            row = " ".join(c.ljust(w) for c, (_, w) in zip(cells, GATE_COLUMNS, strict=True))
            lines.append("    " + row.rstrip())
    lines.append(
        "  A dash in share or oracle/offer is a figure that does not exist: no split had a "
        "complete positive ceiling, or the mean offer is 0 so there is no ratio."
    )
    lines.append(
        f"  Plan thresholds, shown beside the figures and not applied here: the offer is at least "
        f"{_share(thresholds['share_of_ceiling'])} of the ceiling; the oracle-departure offer is "
        f"at least {thresholds['oracle_to_offer_ratio']:g} times the offer. This command gives "
        "no verdict; the decision is the owner's."
    )
    return lines


def _what_if_head(document: dict) -> list[str]:
    """The first lines of a what-if run's summary (`dial_choice --what-if-kappa` or a fit option):
    what it is in capitals, the readings it rests on and, with a fit option, the seed and the
    levels the network was fitted at; nothing for a run without the key `what_if`."""
    found = document.get("what_if")
    if found is None:
        return []
    head = [f"{found['what'].upper()}.", f"{found['readings']}."]
    fit = found.get("fit")
    if fit is not None:
        levels = ", ".join(f"{k:g}" for k in fit["levels_fitted"])
        head.append(
            f"Fit of the network: {seed_words(fit)} ({fit['seed_from']}); levels fitted {levels} "
            f"({fit['levels_from']})."
        )
        if "seeds" in fit:
            head.append(f"Averaged: {fit['averaged']}.")
            head.append(ENERGY_MEANS)
    return head


def seed_words(fit: dict) -> str:
    """The seed of a fit record in words: 'seed N', or for an average over seeds
    (`dial_choice --what-if-fit-seeds`, key `seeds`) 'seeds N1, N2 averaged'."""
    if "seeds" in fit:
        return f"seeds {', '.join(str(s) for s in fit['seeds'])} averaged"
    return f"seed {fit['seed']}"


def summary_lines(document: dict) -> list[str]:
    run = document["run"]
    setting = document.get("inputs", {}).get("setting", "main")
    lines = [
        *rotation_whatif.summary_head(document),
        *_what_if_head(document),
        f"Dial choice for model {document['model']}: {run['splits']} seeded splits per unit "
        f"(seeds 1 to {run['splits']}), {setting} truth setting, main reading "
        f"(departures seen from the rounded-down departure hour), kappa {run['kappa']}.",
        "Scales used: "
        + ", ".join(f"{g:g}" for g in run["scales"])
        + " (the plan's grid up to 1).",
        "Offers are kWh per counting test day, one-way re-solve, calibrated on each split's "
        "calibration days; p10, p50 and p90 are over splits; splits with no safe scale count as "
        "an offer of 0. offering: share of splits with a scale above 0. slot loss and driver "
        "loss: mean test loss over the splits that offer; slot>lim and driver>lim: share of "
        "those splits whose mean test loss is over its limit. of ceiling: offer as a share of "
        "the best-causal ceiling of the test days, averaged over the splits that offer only; the "
        "gate-reading tables give the shares over all splits.",
        "The two tiers are separate blocks and are never added or averaged.",
    ]
    option = document.get("definitions", {}).get("kappa_option")
    if option is not None:
        lines.append(f"{option}.")
    lines.append(calibration_rule_text(run))
    for unit, entry in document["units"].items():
        lines.append("")
        if entry["status"] != "calibratable":
            lines.append(f"{unit} ({entry['status']}, {entry['counting_days']} counting days):")
            lines.append("  no numbers: the unit cannot be split")
            continue
        lines.append(
            f"{unit} ({entry['counting_days']} counting days, each split: "
            f"{entry['calibration_days_used']} calibration days and {entry['test_days']} test "
            "days):"
        )
        for tier in ("A", "B"):
            if tier not in document["run"]["tiers"]:
                continue
            lines.append(f" {TIER_LABELS[tier]}")
            lines.append(_header())
            for name, found in entry["configurations"].items():
                if "tiers" not in found:
                    lines.append(f"  {name}: failed ({found.get('failed')})")
                    continue
                dial = found["driver_risk"]
                label = "no rows" if dial is None else f"eps {dial:g}"
                lines.append(_row(label, found["tiers"][tier], tier))
            chosen = document["dial_choice"][unit][tier]
            if chosen["dial"] is None:
                lines.append(f"  {unit}, {TIER_LABELS[tier]}: no dial has a mean offer; {LABEL}")
            else:
                lines.append(
                    f"  {unit}, {TIER_LABELS[tier]}: eps {chosen['dial']:g} has the largest "
                    f"mean offer, {_number(chosen['mean_offer_kwh_per_test_day'])} kWh per "
                    f"test day; {LABEL}"
                )
        lines += _gate_lines(document, entry)
    timeouts = document["timeouts"]
    lines.append("")
    lines.append(
        "Timeouts (recorded outcomes): "
        + (", ".join(f"{k} {v}" for k, v in timeouts.items()) if timeouts else "none")
    )
    if document["failed_configurations"]:
        lines.append("FAILED: " + "; ".join(document["failed_configurations"]))
    return lines
