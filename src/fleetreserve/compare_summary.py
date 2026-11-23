"""The plain-text summary of a comparison results document (`compare_run`).

Per unit that can be split: one table per tier, one row per method, the forecast variant's
columns and then the oracle-departure variant's columns on the same row; then one tier A table of
the gate reading per method. A unit that cannot be split gets a line with its status and a line
saying that it has no numbers. The tiers are separate blocks and nothing is added or averaged
across them.

Columns of a tier table, per variant, from the method's `dial_splits.summarise` fields: offer
(`mean_offer_kwh_per_test_day`), of ceiling (`mean_share_of_best_causal_ceiling`, over the splits
that offer only, as `dial_summary` shows it), slot loss and driver loss (`mean_test_slot_loss`,
`mean_test_driver_loss`), offering (`share_of_splits_that_offer`) and p10 (`offer_p10`). Columns
of the gate-reading table, from `dial_splits.summarise_gate`: the mean offer, the share of each
ceiling of the run (over the splits whose ceiling is complete: a split with a missing ceiling is
left out of that share and counted in the results), the oracle-departure variant's mean offer and
its ratio to the mean offer.
A dash is a figure that does not exist, or a cell of a variant that failed (a line under the table
names the failure).

A results document of the no-pause sensitivity (`compare_run`: it has the key `no_pause`) gets a
first line (on the sealed route its third, after the route's two lines below) naming the
sensitivity, its share and seed and the ceilings the "of ceiling" column and the gate-reading
table read, and, per unit that can be split, one line under the unit's heading
with its counts of cars and of cars that cannot pause (`no_pause.counts`; the share drawn rounded
to a whole percent) and of days with every car drawn. A document without that key gets neither.

After the line of timeouts, when a controller reported solver counts (`compare_run`,
`evaluations.controller_solver_outcomes`), one line gives each such method's counts summed over
the units and variants: its solves by outcome, the hours it planned and the hours it planned by its
fall-back, and, for a method that counts boxes (`diagonal-split`), the number of boxes of each
kind. The outcomes are given in the order of `lp.Outcome` and the kinds in the order of
`diagonal_split.BOXES` (any other key after them, in the order found), whatever the order of
the keys in the results, so the line made from `compare.json` as written (keys sorted) is the
line made from the results before they were written. A run without such a controller has no
such line. Then, when a controller reported deviation counts (`compare_run`,
`evaluations.controller_deviations`), one line gives each such method's counts summed over the
units and variants: the hours it planned, those in which least-laxity-first's split was found
infeasible and the total nearest the target found feasible was planned (and how many of them
were service hours), and those with no feasible total (`mef_controller`, steps 2 to 4). A run
without such a controller has no such line. Then, when the run has a sizing method (`compare_run`,
`evaluations.sizing_solver_outcomes`), one line gives each such method's counts summed over the
units and variants: the calibration sets sized and the programmes solved by outcome (in the order
of `lp.Outcome`, any other key after them). A run without a sizing method has no such line, and
neither has it the line after the calibration rule that says what a sizing method is.

The method column of the tables is `METHOD_WIDTH` characters wide, or as wide as the longest
method name of the run when that is longer.

After all of that (and after the line of failures, when there is one), the paired differences of
`compare_paired` (results key `paired_differences`): one paragraph saying what they are, then per
unit that can be split, tier and variant, the table against `re-solve` and the table of the other
pairs, one row per pair: the mean difference (method minus reference, kWh per counting test day,
three decimals), its 5th and 95th percentiles over the paired splits, the shares of paired splits
above, exactly equal to and below 0, the mean difference as a share of the reference's mean offer
("no offer" when that mean is 0), the verdict, the number of paired splits and the number left
out. A dash is a figure that does not exist (no paired split). A table with no pair (no
`re-solve` in the run, or fewer than two other methods) gets one line saying so. A unit that
cannot be split gets one line. A results document without `paired_differences` gets none of these
lines.

A results document of the sealed route (`compare_run`: it has the key `sealed_route`) gets the
route's two lines first (`sealed_route.summary_head`: the tag's commit and the two builds, then
what the route's checks stop and what they do not); its line on the run names the one split per
unit its held-out build drew (with its seed) in place of the seeded splits, its line on what the
columns are says what each is on the one split (`ROUTE_COLUMNS_TEXT` in place of `COLUMNS_TEXT`),
the heading of each gate-reading table says "on the one split", and the paragraph on the paired
differences is followed by the results' definition of their interval on the route
(`compare_run.ROUTE_INTERVAL`). A document without that key gets none of these.
"""

from fleetreserve import rotation_whatif, sealed_route
from fleetreserve.diagonal_split import BOXES, OUTCOMES
from fleetreserve.gate_config import TIER_LABELS
from fleetreserve.gate_summary import ROUTE_KEY, calibration_rule_text
from fleetreserve.mef_controller import COUNTS as DEVIATION_COUNTS

METHOD_WIDTH = 14
COLUMNS = (
    ("offer", 8),
    ("of ceiling", 11),
    ("slot loss", 10),
    ("driver loss", 12),
    ("offering", 9),
    ("p10", 8),
)
# The summary's line on what the columns are (module text), off the route and on it.
COLUMNS_TEXT = (
    "Every method runs on the same certified box, audited paths and steered cars, and is "
    "calibrated on each split's calibration days and scored on its test days; offers are kWh "
    "per counting test day, p10 over splits, splits with no safe scale counting as 0. "
    "of ceiling: share of the best-causal ceiling over the splits that offer only; the "
    "gate-reading table gives shares over all splits whose ceiling is complete (a split with "
    "a missing ceiling is left out and counted in the results). slot loss and driver loss: "
    "mean test "
    "loss over the splits that offer. offering: share of splits with a scale above 0."
)
ROUTE_COLUMNS_TEXT = (
    "Every method runs on the same certified box, audited paths and steered cars, and is "
    "calibrated on the one split's calibration days and scored on its test days; offers are kWh "
    "per counting test day on that split, 0 where it has no safe scale, so offer and p10 are one "
    "figure. of ceiling: share of the best-causal ceiling where the split offers; the "
    "gate-reading table gives the split's share where its ceiling is complete (a missing ceiling "
    "is counted in the results). slot loss and driver loss: the split's mean test loss where it "
    "offers. offering: 100% where the split's scale is above 0, else 0%."
)


def _number(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _share(value) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def _cells(tier: dict | None) -> list[str]:
    if tier is None:
        return ["-"] * len(COLUMNS)
    return [
        _number(tier["mean_offer_kwh_per_test_day"]),
        _share(tier["mean_share_of_best_causal_ceiling"]),
        _number(tier["mean_test_slot_loss"], 3),
        _number(tier["mean_test_driver_loss"], 3),
        _share(tier["share_of_splits_that_offer"]),
        _number(tier["offer_p10"]),
    ]


def _line(cells: list[str], widths: list[int]) -> str:
    return "  " + " ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip()


def _method_width(document: dict) -> int:
    """The method column's width: `METHOD_WIDTH`, or the longest method name of the run."""
    return max([METHOD_WIDTH, *(len(m) for m in document["methods"])])


def _tier_lines(document: dict, entry: dict, tier: str) -> list[str]:
    variants = document["run"]["variants"]
    named = _method_width(document)
    widths = [named] + [w for _ in variants for _, w in COLUMNS]
    lines = [f" {TIER_LABELS[tier]}"]
    block = sum(w for _, w in COLUMNS) + len(COLUMNS) - 1
    lines.append(_line(["", *variants], [named] + [block] * len(variants)))
    lines.append(_line(["method"] + [n for _ in variants for n, _ in COLUMNS], widths))
    for method in document["methods"]:
        cells = [method]
        for variant in variants:
            found = entry["variants"].get(variant, {})
            if "methods" not in found:
                cells += _cells(None)
                continue
            cells += _cells(found["methods"][method][tier])
        lines.append(_line(cells, widths))
    for variant in variants:
        found = entry["variants"].get(variant, {})
        if "failed" in found:
            lines.append(f"  {variant}: failed ({found['failed']})")
    return lines


def _on_route(document: dict) -> bool:
    """Whether the document is of a run on the sealed route (module text)."""
    return bool(document.get(ROUTE_KEY))


def _splits_text(document: dict) -> str:
    run = document["run"]
    if _on_route(document):
        return f"the one split per unit its held-out build drew (seed {run['seeds'][0]})"
    return f"{run['splits']} seeded splits per unit (seeds 1 to {run['splits']})"


def _gate_lines(document: dict, entry: dict) -> list[str]:
    ceilings = document["run"]["ceilings"]
    names = ["method", "offer", *(f"share of {c}" for c in ceilings), "oracle", "oracle/offer"]
    widths = [_method_width(document), 8, *(max(len(f"share of {c}"), 8) for c in ceilings), 8, 12]
    over = f"mean over all {document['run']['splits']} splits"
    if _on_route(document):
        over = "on the one split"
    lines = [
        f" Gate reading, tier A (driver-protected), {over} (kWh per counting test day; oracle: "
        "the oracle-departure variant)",
        _line(names, widths),
    ]
    reading = entry.get("gate_reading")
    if reading is None:
        lines.append("  not computed: a variant did not run")
        return lines
    for method in document["methods"]:
        found = reading[method]
        cells = [method, _number(found["mean_offer_kwh_per_test_day"])]
        cells += [_share(found["ceilings"][c]["share_of_ceiling"]) for c in ceilings]
        cells += [
            _number(found["mean_oracle_offer_kwh_per_test_day"]),
            _number(found["oracle_to_offer_ratio"]),
        ]
        lines.append(_line(cells, widths))
    return lines


def _in_order(counts: dict, order) -> list:
    return [k for k in order if k in counts] + [k for k in counts if k not in order]


def _solver_line(document: dict) -> list[str]:
    totals: dict = {}
    for entry in document["units"].values():
        for variant in entry["variants"].values():
            found = variant.get("evaluations", {}).get("controller_solver_outcomes", {})
            for method, counts in found.items():
                total = totals.setdefault(method, {"solves": {}, "hours": 0, "fallback_hours": 0})
                for outcome, number in counts["solves"].items():
                    total["solves"][outcome] = total["solves"].get(outcome, 0) + number
                total["hours"] += counts["hours"]
                total["fallback_hours"] += counts["fallback_hours"]
                if "boxes" in counts:
                    boxes = total.setdefault("boxes", {})
                    for kind, number in counts["boxes"].items():
                        boxes[kind] = boxes.get(kind, 0) + number
    if not totals:
        return []
    parts = []
    for method in document["methods"]:
        if method not in totals:
            continue
        total = totals[method]
        solves = ", ".join(
            f"{o} {total['solves'][o]}" for o in _in_order(total["solves"], OUTCOMES)
        )
        part = (
            f"{method}: solves {solves}; hours planned {total['hours']}, by the fall-back "
            f"{total['fallback_hours']}"
        )
        if "boxes" in total:
            boxes = total["boxes"]
            part += "; boxes " + ", ".join(f"{k} {boxes[k]}" for k in _in_order(boxes, BOXES))
        parts.append(part)
    return ["Controller solves (all units and variants together): " + "; ".join(parts) + "."]


def _no_pause_head(document: dict) -> list[str]:
    found = document.get("no_pause")
    if found is None:
        return []
    return [
        f"NO-PAUSE SENSITIVITY (plan section 9; Brinkel et al. (2024)): a seeded share "
        f"{found['share']} of the cars (seed {found['seed']}) charge as logged, without "
        "flexibility, and are left unsteered by every method; this is not the run without the "
        "sensitivity. of ceiling: share of the whole fleet's best-causal ceiling, the cars that "
        "cannot pause included; the gate-reading table also gives the shares of the ceilings of "
        "the fleet without those cars (names ending -no-pause-cars-removed).",
    ]


def _no_pause_unit(entry: dict) -> list[str]:
    found = entry.get("no_pause")
    if found is None:
        return []
    cars, drawn = found["cars"], found["cars_drawn"]
    share = _share(drawn / cars) if cars else "-"
    every = found["days_every_car_drawn"]
    if every == 1:
        days = "1 day with every car drawn, which abstains"
    else:
        days = f"{every} days with every car drawn, which abstain"
    return [
        f" cars that cannot pause: {drawn} of {cars} cars of the counting days ({share}); {days}"
    ]


def _deviation_line(document: dict) -> list[str]:
    totals: dict = {}
    for entry in document["units"].values():
        for variant in entry["variants"].values():
            found = variant.get("evaluations", {}).get("controller_deviations", {})
            for method, counts in found.items():
                total = totals.setdefault(method, dict.fromkeys(DEVIATION_COUNTS, 0))
                for key in DEVIATION_COUNTS:
                    total[key] += counts[key]
    if not totals:
        return []
    parts = [
        f"{method}: hours planned {totals[method]['hours']}, with least-laxity-first's split "
        "found infeasible and the total nearest the target found feasible planned "
        f"{totals[method]['deviating_hours']} (in service hours "
        f"{totals[method]['deviating_service_hours']}), with no feasible total "
        f"{totals[method]['no_feasible_hours']}"
        for method in document["methods"]
        if method in totals
    ]
    return ["Controller deviations (all units and variants together): " + "; ".join(parts) + "."]


def _sizing_line(document: dict) -> list[str]:
    totals: dict = {}
    for entry in document["units"].values():
        for variant in entry["variants"].values():
            found = variant.get("evaluations", {}).get("sizing_solver_outcomes", {})
            for method, counts in found.items():
                total = totals.setdefault(method, {"calibration_sets": 0, "solves": {}})
                total["calibration_sets"] += counts["calibration_sets"]
                for outcome, number in counts["solves"].items():
                    total["solves"][outcome] = total["solves"].get(outcome, 0) + number
    if not totals:
        return []
    parts = []
    for method in document["methods"]:
        if method not in totals:
            continue
        total = totals[method]
        solves = ", ".join(
            f"{o} {total['solves'][o]}" for o in _in_order(total["solves"], OUTCOMES)
        )
        parts.append(f"{method}: calibration sets {total['calibration_sets']}; solves {solves}")
    return ["Sizing solves (all units and variants together): " + "; ".join(parts) + "."]


def _sizing_head(document: dict) -> list[str]:
    sizing = document.get("definitions", {}).get("sizing")
    if sizing is None:
        return []
    return [
        f"Sizing methods, {sizing['methods']}. Readings S1 to S11 in the results "
        "(definitions.sizing.readings)."
    ]


def summary_lines(document: dict) -> list[str]:
    run = document["run"]
    setting = document.get("inputs", {}).get("setting", "main")
    lines = [
        *sealed_route.summary_head(document),
        *_no_pause_head(document),
        *rotation_whatif.summary_head(document),
        f"Comparison methods for model {document.get('model')}, dial {document['dial']:g}: "
        f"{_splits_text(document)}, {setting} truth setting, kappa {run['kappa']}. "
        f"{document['label']}.",
        "Variants: " + "; ".join(f"{v} = {c}" for v, c in document["configurations"].items()) + ".",
        "Scales used: " + ", ".join(f"{g:g}" for g in run["scales"]) + ".",
        ROUTE_COLUMNS_TEXT if _on_route(document) else COLUMNS_TEXT,
        "The two tiers are separate blocks and are never added or averaged.",
        calibration_rule_text(run),
        *_sizing_head(document),
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
        lines += _no_pause_unit(entry)
        for tier in run["tiers"]:
            lines += _tier_lines(document, entry, tier)
        lines += _gate_lines(document, entry)
    timeouts = document["timeouts"]
    lines.append("")
    lines.append(
        "Timeouts (recorded outcomes): "
        + (", ".join(f"{k} {v}" for k, v in timeouts.items()) if timeouts else "none")
    )
    lines += _solver_line(document)
    lines += _deviation_line(document)
    lines += _sizing_line(document)
    if document["failed"]:
        lines.append("FAILED: " + "; ".join(document["failed"]))
    lines += _paired_lines(document)
    return lines


PAIRED_COLUMNS = (
    ("method", METHOD_WIDTH),
    ("reference", METHOD_WIDTH),
    ("mean", 9),
    ("p5", 9),
    ("p95", 9),
    ("above", 6),
    ("equal", 6),
    ("below", 6),
    ("of ref", 9),
    ("verdict", 14),
    ("paired", 7),
    ("left out", 8),
)
PAIRED_TEXT = (
    "Paired differences (compare_paired): method minus reference on the same split, kWh per "
    "counting test day, over the splits where both have a value. p5, p95: the 5th and 95th "
    "percentiles of those differences over this unit's splits. The splits share the same days "
    "and differ only in which days calibrate and which are scored, so p5 to p95 is the spread "
    "over those draws at this unit, not a confidence interval for new days or other sites. "
    "above, equal, below: shares of paired splits with a difference above, exactly equal to and "
    "below 0. of ref: the mean difference over the reference's mean offer on the paired splits "
    "(no offer: that mean is 0). verdict: ahead when p5 is above 0, behind when p95 is below 0, "
    "not separated otherwise; with very few paired splits p5 and p95 rest on very few "
    "differences (with one, both are that difference), and so does the verdict. left out: "
    "splits left out because one or both of the two methods has no value on them, never read "
    "as 0."
)
EMPTY_TABLE = {
    "against_re_solve": "none: re-solve is not in the run",
    "other_pairs": "none: fewer than two methods other than re-solve",
}
TABLE_NAMES = {"against_re_solve": "against re-solve", "other_pairs": "other pairs"}


def _difference(value) -> str:
    return "-" if value is None else f"{value:+.3f}"


def _of_reference(row: dict) -> str:
    if row["splits_paired"] == 0:
        return "-"
    share = row["mean_difference_share_of_reference_mean_offer"]
    return "no offer" if share is None else f"{100 * share:+.1f}%"


def _paired_row(row: dict) -> list[str]:
    return [
        row["method"],
        row["reference"],
        _difference(row["mean_difference_kwh_per_test_day"]),
        _difference(row["difference_p5_kwh_per_test_day"]),
        _difference(row["difference_p95_kwh_per_test_day"]),
        _share(row["share_of_paired_splits_above_0"]),
        _share(row["share_of_paired_splits_equal_0"]),
        _share(row["share_of_paired_splits_below_0"]),
        _of_reference(row),
        row["verdict"] or "-",
        str(row["splits_paired"]),
        str(row["splits_left_out_value_missing"]),
    ]


def _paired_lines(document: dict) -> list[str]:
    paired = document.get("paired_differences")
    if paired is None:
        return []
    named = _method_width(document)
    widths = [named, named, *(w for _, w in PAIRED_COLUMNS[2:])]
    lines = ["", PAIRED_TEXT]
    if _on_route(document):
        lines.append(paired["definitions"]["interval"])
    for unit in document["units"]:
        lines.append("")
        found = paired["units"].get(unit)
        if found is None:
            lines.append(f"{unit}: no paired differences: the unit cannot be split")
            continue
        lines.append(f"{unit}, paired differences:")
        for tier in document["run"]["tiers"]:
            for variant in document["run"]["variants"]:
                for table, rows in found[variant][tier].items():
                    lines.append(f" {TIER_LABELS[tier]}, {variant}, {TABLE_NAMES[table]}")
                    if not rows:
                        lines.append(f"  {EMPTY_TABLE[table]}")
                        continue
                    lines.append(_line([n for n, _ in PAIRED_COLUMNS], widths))
                    lines += [_line(_paired_row(row), widths) for row in rows]
    return lines
