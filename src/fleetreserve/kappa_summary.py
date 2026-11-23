"""The plain-text summary of a kappa-choice document (`kappa_choice`).

Per gate unit and tier, one table: four lines per kappa (the mean offer, the share of splits that
offer, and the mean slot and driver losses over the splits that offer) and one column per dial,
the configuration without rows ("none") first. Then per unit the timeouts per pair, as
"certificate timeouts/re-solve timeouts". A pair that is not defined is printed "not defined" on
every line, never as a number; a loss over no offering split is "-". Then what each reading of the
joint rule gives per site type, every such line labelled "measured, not fixed", and the site types
where the two readings differ. The tiers are separate blocks; nothing is added or averaged across
them. Every number comes from the document; this module computes none.

With what-if runs (`kappa_choice` reading K9) a block follows, its every heading and reading line
labelled "what-if, level not on the plan's grid": per unit and tier the table and timeouts of the
what-if runs, then R1 and R2 over all levels given. Without them the summary ends as before.
What-if runs that carry a fit (`kappa_choice` reading K10) print the seed and the levels fitted on
every line that names one; runs of different fits are blocks of their own, each headed "FIT:", and
the summary says that none of them is read with another.
"""

from fleetreserve.dial_summary import seed_words
from fleetreserve.gate_config import TIER_LABELS

KAPPA_WIDTH = 6
FIGURE_WIDTH = 12
CELL_WIDTH = 12
NOT_DEFINED = "not defined"
FIGURES = (
    ("offer", "mean_offer_kwh_per_test_day", lambda v: f"{v:.2f}"),
    ("offering", "share_of_splits_that_offer", lambda v: f"{100 * v:.0f}%"),
    ("slot loss", "mean_test_slot_loss", lambda v: f"{v:.3f}"),
    ("driver loss", "mean_test_driver_loss", lambda v: f"{v:.3f}"),
)


def _key(dial) -> str:
    return "none" if dial is None else f"{dial:g}"


def _line(first: str, cells: list[str], figure: str | None = None) -> str:
    head = f"  {first:<{KAPPA_WIDTH}} "
    if figure is not None:
        head += f"{figure:<{FIGURE_WIDTH}} "
    return (head + " ".join(c.ljust(CELL_WIDTH) for c in cells)).rstrip()


def _cell(cell: dict, field: str, show) -> str:
    if not cell["defined"]:
        return NOT_DEFINED
    value = cell[field]
    return "-" if value is None else show(value)


def _table(entry: dict, tier: str, kappas: list, columns: list) -> list[str]:
    lines = [f" {TIER_LABELS[tier]}", _line("kappa", [_key(d) for d in columns], "figure")]
    for kappa in kappas:
        row = entry["pairs"][tier][f"{kappa:g}"]
        for k, (figure, field, show) in enumerate(FIGURES):
            cells = [_cell(row[_key(d)], field, show) for d in columns]
            lines.append(_line(f"{kappa:g}" if k == 0 else "", cells, figure))
    return lines


def _timeouts(entry: dict, kappas: list, columns: list) -> list[str]:
    lines = [
        " Timeouts per pair, certificate timeouts/one-way re-solve timeouts (the same for both "
        "tiers):",
        _line("kappa", [_key(d) for d in columns]),
    ]
    for kappa in kappas:
        row = entry["timeouts"][f"{kappa:g}"]
        cells = []
        for d in columns:
            found = row[_key(d)]
            if found is None:
                cells.append(NOT_DEFINED)
            else:
                certificates = "-" if found["certificates"] is None else found["certificates"]
                cells.append(f"{certificates}/{found['re_solves']}")
        lines.append(_line(f"{kappa:g}", cells))
    return lines


def _pair_line(site_type: str, unit: str, reading: str, kappa, found, label: str) -> str:
    head = f"  {site_type} ({unit}), {reading}: "
    if found is None or kappa is None:
        return head + f"{NOT_DEFINED} (no pair to choose from: reading K7)"
    return (
        head + f"kappa {kappa:g}, eps {found['dial']:g}, {found['mean_offer']:.2f} kWh per "
        f"counting test day; {label}"
    )


def _reading_lines(document: dict) -> list[str]:
    return ["", *_readings(document["readings"], document["definitions"], document["run"])]


def _readings(readings: dict, definitions: dict, run: dict) -> list[str]:
    sites = run["site_types"]
    label = readings["label"]
    lines = [f"{definitions['R1']}:"]
    for site_type, unit in sites.items():
        found = readings["R1"][site_type]
        kappa = None if found is None else found["kappa"]
        lines.append(_pair_line(site_type, unit, "R1", kappa, found, label))
    r2 = readings["R2"]
    sums = "; ".join(
        f"{level}: {NOT_DEFINED if s is None else f'{s:.2f}'}"
        for level, s in r2["sums_by_kappa"].items()
    )
    lines += [f"{definitions['R2']}:", f"  Sums by kappa (kWh per counting test day): {sums}"]
    for site_type, unit in sites.items():
        lines.append(_pair_line(site_type, unit, "R2", r2["kappa"], r2["dials"][site_type], label))
    if readings["differ_at"]:
        lines.append(
            f"The two readings differ at {', '.join(readings['differ_at'])}. Which reading holds "
            "is the owner's decision."
        )
    else:
        lines.append(
            "The two readings give the same pair at every site type. Which reading holds is the "
            "owner's decision."
        )
    lines.append(f"Ties: {definitions['ties']}.")
    return lines


def summary_lines(document: dict) -> list[str]:
    run = document["run"]
    definitions = document["definitions"]
    kappas = run["kappas"]
    columns = [None, *run["dials"]]
    seeds = run["seeds"]
    lines = [
        f"Kappa choice for model {run['model']}: the dial family measured at each level of plan "
        f"section 2 (kappa {', '.join(f'{k:g}' for k in kappas)}), one dial-choice run per level, "
        f"each recorded at commit {run['commit'][:12]}; {run['splits']} seeded splits per unit "
        f"(seeds {seeds[0]} to {seeds[-1]}), {run['rung']} rung, "
        f"{(run.get('inputs') or {}).get('setting', 'main')} truth setting.",
        "Runs read: "
        + "; ".join(f"kappa {r['kappa']:g}: {r['run_folder']}" for r in document["runs"]),
        "Per pair, from the run at its kappa: offer, the mean calibrated offer in kWh per "
        "counting test day (a split with no safe scale counts as 0); offering, the share of "
        "splits with a scale above 0; slot loss and driver loss, the mean test losses over the "
        "splits that offer ('-': no split offers).",
        f"'{NOT_DEFINED}': {definitions['not_defined']}.",
        f"Columns: the dial eps of plan section 2; 'none': {definitions['none']}.",
        f"{definitions['kappa_0.05']}.",
        f"The two tiers are separate blocks; {definitions['tiers']}.",
    ]
    for unit, entry in document["units"].items():
        lines.append("")
        status = "" if entry["status"] == "calibratable" else f", {entry['status']}"
        lines.append(f"{unit} ({entry['counting_days']} counting days{status}):")
        for tier in run["tiers"]:
            lines += _table(entry, tier, kappas, columns)
        lines += _timeouts(entry, kappas, columns)
    lines += _reading_lines(document)
    lines.append("")
    lines.append(
        "Timeouts over each whole run (recorded outcomes; ceilings and oracle partners included): "
        + "; ".join(
            f"kappa {level}: "
            + (", ".join(f"{k} {v}" for k, v in found.items()) if found else "none")
            for level, found in document["run_timeouts"].items()
        )
    )
    if "what_if" in document:
        lines += _what_if_lines(document)
    return lines


def _fit_words(fit: dict) -> str:
    levels = ", ".join(f"{k:g}" for k in fit["levels_fitted"])
    return f"{seed_words(fit)}, levels fitted {levels}"


def _what_if_lines(document: dict) -> list[str]:
    """The what-if block (module text): one block, or one per fit under `separate_fits`."""
    found = document["what_if"]
    blocks = found.get("separate_fits")
    if blocks is None:
        return _block_lines(document, found, found, False)
    head = found["label"].upper()
    lines = ["", f"{head}.", f"{found['definitions']['what_if']}."]
    lines.append(f"{found['definitions']['fits']}.")
    lines.append(
        f"{len(blocks)} sets of what-if runs of different fits follow, each read apart: no table "
        "and no reading below takes a figure from two of them."
    )
    for block in blocks:
        lines += _block_lines(document, found, block, True)
    return lines


def _block_lines(document: dict, found: dict, block: dict, apart: bool) -> list[str]:
    """The lines of one block of what-if runs; `found` is the document's key `what_if`, `apart`
    whether the block is one of several fits."""
    run = document["run"]
    kappas = block["kappas"]
    columns = [None, *run["dials"]]
    head = found["label"].upper()
    fit = block.get("fit")
    lines = [""]
    if apart:
        words = "none given, the seed is not stated" if fit is None else _fit_words(fit)
        lines.append(f"{head}, FIT: {words}.")
    else:
        lines += [f"{head}.", f"{block['definitions']['what_if']}."]
        if fit is not None:
            lines += [f"{block['definitions']['fits']}.", f"Fit of the network: {_fit_words(fit)}."]
    lines.append(
        "What-if runs read: "
        + "; ".join(
            f"kappa {r['kappa']:g}: {r['run_folder']} ({r['what']})"
            + ("" if r.get("fit") is None else f" [{_fit_words(r['fit'])}]")
            for r in block["runs"]
        )
    )
    for unit, entry in block["units"].items():
        lines.append("")
        status = "" if entry["status"] == "calibratable" else f", {entry['status']}"
        lines.append(f"{unit} ({entry['counting_days']} counting days{status}), what-if levels:")
        for tier in run["tiers"]:
            lines += _table(entry, tier, kappas, columns)
        lines += _timeouts(entry, kappas, columns)
    over_all = block["readings_over_all_levels"]
    levels = ", ".join(f"{k:g}" for k in over_all["levels"])
    lines += ["", f"Over all levels given ({levels}), {head}:"]
    if "mixes" in over_all:
        lines.append(f"{over_all['mixes']}.")
    lines += _readings(over_all, document["definitions"], run)
    lines.append("")
    lines.append(
        "Timeouts over each whole what-if run (recorded outcomes): "
        + "; ".join(
            f"kappa {level}: " + (", ".join(f"{k} {v}" for k, v in t.items()) if t else "none")
            for level, t in block["run_timeouts"].items()
        )
    )
    return lines
