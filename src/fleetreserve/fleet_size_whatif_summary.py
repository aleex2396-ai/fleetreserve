"""The plain-text summary of a fleet-size what-if results document (`fleet_size_whatif`).

Per unit and configuration: the table by exact number of steered cars (at scale 1; every grid
scale is in the results file), then the number of days that stop offering for each minimum, then
one table per tier in the layout of the dial-choice summary (`dial_summary`), one row per minimum,
then the gate-reading table of tier A per ceiling, one row per minimum. The tiers are separate
blocks and are never added or averaged. The text names no choice and no recommendation and reads
no figure against a threshold; the plan's thresholds are shown beside the figures.
"""

from fleetreserve import dial_summary
from fleetreserve.gate_config import TIER_LABELS
from fleetreserve.gate_summary import calibration_rule_text

LABEL = (
    "fleet-size what-if: a measurement; the plan has no minimum number of steered cars for a day "
    "to offer, and this changes no rung, gate or calibration rule and no existing result"
)
TABLE_SCALE = 1.0
COUNT_COLUMNS = (
    ("cars", 6),
    ("days", 6),
    ("offering", 9),
    ("offer kWh", 11),
    ("slot loss", 10),
    ("driver loss", 12),
    ("slot fail", 10),
    ("driver fail", 11),
)
BELOW_COLUMNS = (("min cars", 9), ("days below", 11), ("offering days no longer offering", 32))


def _line(cells, columns) -> str:
    return (
        "  " + " ".join(str(c).ljust(w) for c, (_, w) in zip(cells, columns, strict=True)).rstrip()
    )


def _header(columns) -> str:
    return "  " + " ".join(name.ljust(width) for name, width in columns).rstrip()


def _count_row(row: dict) -> str:
    found = next(s for s in row["scales"] if s["scale"] == TABLE_SCALE)
    cells = [
        row["steered_cars"],
        row["counting_days"],
        row["offering_days"],
        f"{row['offer_at_scale_1_kwh_sum']:.2f}",
        f"{found['slot_loss_sum']:.3f}",
        f"{found['driver_loss_sum']:.3f}",
        found["days_with_a_slot_loss"],
        found["days_with_a_driver_loss"],
    ]
    return _line(cells, COUNT_COLUMNS)


def _gate_lines(run: dict, found: dict) -> list[str]:
    """The gate-reading tables of one configuration, one row per minimum (module text)."""
    lines = [
        f" Gate reading, tier A, mean over all {run['splits']} splits (kWh per counting test day)"
    ]
    shown = [item for item in found["minimums"] if "ceilings" in item["gate_reading"]]
    for item in found["minimums"]:
        reading = item["gate_reading"]
        if "ceilings" not in reading:
            why = reading.get("failed") or reading.get("not_computed", "no result")
            lines.append(f"  min {item['min_cars']}: not computed ({why})")
    if not shown:
        return lines
    differ = found.get("oracle_count_differs_on_days")
    lines.append(
        f"  oracle partner {found['oracle_partner']}: its days are restricted by its own count "
        f"of steered cars; the two counts differ on {differ} counting days."
    )
    for ceiling in run["ceilings"]:
        lines.append(f"  ceiling {ceiling}")
        lines.append("    " + " ".join(n.ljust(w) for n, w in dial_summary.GATE_COLUMNS).rstrip())
        for item in shown:
            cells = dial_summary._gate_cells(
                f"min {item['min_cars']}", item["gate_reading"], ceiling
            )
            columns = dial_summary.GATE_COLUMNS
            row = " ".join(c.ljust(w) for c, (_, w) in zip(cells, columns, strict=True))
            lines.append("    " + row.rstrip())
    thresholds = run["thresholds"]
    lines.append(
        f"  Plan thresholds, shown beside the figures and not applied here: the offer is at least "
        f"{thresholds['share_of_ceiling'] * 100:.0f}% of the ceiling; the oracle-departure offer "
        f"is at least {thresholds['oracle_to_offer_ratio']:g} times the offer."
    )
    return lines


def summary_lines(document: dict) -> list[str]:
    run = document["run"]
    setting = document.get("inputs", {}).get("setting", "main")
    lines = [
        document["label"],
        f"Model {document['model']}: {run['splits']} seeded splits per unit (seeds 1 to "
        f"{run['splits']}), {setting} truth setting, one-way re-solve, calibrated on each "
        "split's calibration days. Minimums tried: " + ", ".join(str(n) for n in run["min_cars"]),
        "Steered cars: " + document["definitions"]["steered_cars"],
        "A minimum N: " + document["definitions"]["minimum"],
        calibration_rule_text(run),
        "By-count tables show scale 1 and are over all counting days of the unit; sums of the "
        "losses are over those days; slot fail and driver fail count days with a loss above 0.",
        "Offers are kWh per counting test day; p10, p50 and p90 are over splits; splits with no "
        "safe scale count as an offer of 0. The two tiers are separate blocks and are never "
        "added or averaged.",
    ]
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
        for name, found in entry["configurations"].items():
            lines.append("")
            dial = found.get("driver_risk")
            kind = "no driver-risk rows" if dial is None else f"eps {dial:g}"
            lines.append(f" {name} ({kind})")
            if "failed" in found:
                lines.append(f"  failed: {found['failed']}")
                continue
            lines.append(f"  by number of steered cars (scale {TABLE_SCALE:g}):")
            lines.append(_header(COUNT_COLUMNS))
            lines += [_count_row(row) for row in found["by_count"]]
            lines.append("  days that stop offering:")
            lines.append(_header(BELOW_COLUMNS))
            for item in found["minimums"]:
                cells = [
                    item["min_cars"],
                    f"{item['counting_days_below']} of {entry['counting_days']}",
                    f"{item['offering_days_removed']} of {found['offering_days']}",
                ]
                lines.append(_line(cells, BELOW_COLUMNS))
            for tier in run["tiers"]:
                lines.append(f" {TIER_LABELS[tier]}")
                lines.append(dial_summary._header())
                for item in found["minimums"]:
                    lines.append(
                        dial_summary._row(f"min {item['min_cars']}", item["tiers"][tier], tier)
                    )
            lines += _gate_lines(run, found)
    timeouts = document["timeouts"]
    lines.append("")
    lines.append(
        "Timeouts (recorded outcomes): "
        + (", ".join(f"{k} {v}" for k, v in timeouts.items()) if timeouts else "none")
    )
    if document["failed_configurations"]:
        lines.append("FAILED: " + "; ".join(document["failed_configurations"]))
    return lines
