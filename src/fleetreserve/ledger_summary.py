"""The plain-text summary of a ledger document (`ledger_run`).

On held-out units, read through the sealed route (the ledger's key `sealed_route`), the first
three lines say so: a held-out result through the route at the tag (its commit and, at a logged
parsing fix, the fix commit; the route run read, which `ledger_run` reads only when it was recorded
at the commit checked out now; the held-out build its scored tables come from), what the route's
checks stop and what they do not (`sealed_route.CHECKS`), and that the ceiling and the losses are
hindsight quantities of the scored test days (U against the run's certified value, from its
forecast sets), used to choose nothing (`ledger_run.HINDSIGHT`). Without that key no such line is
printed. Then the labels of the run (a rotation what-if, a look-ahead model, a partial run) and,
when a check failed, which one, with its first breaks. Then per unit: the true-set terms (R + A on
every test day; R and A on the tree sample with the days on which A is not 0; V_aff - V_diag; F),
the block product's terms, and per tier one table over every test day with V_aff (offer, R + A, U,
K) and one over the tree sample (offer, R, A, U, K), one row per configuration and rung. Every term
is in kWh per counting test day with its share of the ceiling of the same days. The tiers are
separate tables; nothing is added or averaged across them.

Outcome O8's ceiling ratio (`ceiling_ratio`, the ledger's key `ceiling_ratio`): one line after the
head names the budget with its label ("B=5" only at 5 kWh, and the words "not outcome O8's B=5"
at another value), what V is, the discharge D as not run and what the offer ratio X is. After each
unit's lines: the sums of its one-way and two-way ceilings over its test days, V, the test days with
a one-way ceiling of 0 (counted, the first of them named, their two-way sum), and per configuration
and tier the run's two-way offer as a share of the two-way ceiling. On the sealed route that line
says it is a held-out unit through the route.

Outcome O8's offer ratio X and outcome O5's offer order (the ledger's unit key `offer_ratio`),
after those lines: one line per configuration with, per tier, X (the two-way offer's sum over the
one-way offer's sum over the unit's test days, "not defined (one-way sum 0)" at 0, "not defined (no
test days)" for a unit with none) and, apart, the test days on which fixed <= one-way holds and the
test days on which one-way <= two-way holds (outcome O8's daily pass), each with its share; under
it, per tier, a line for the days whose one-way offer is 0 ("one-way offer 0": counted, the first
named, their two-way sum) and one per step for the days that break it ("ORDER BROKEN": counted, the
first named with their three offers). A broken step is also a failed check (the seventh), named in
the "CHECK FAILED" lines that open the summary.
"""

from fleetreserve import model_inputs, rotation_whatif, sealed_route
from fleetreserve.gate_config import TIER_LABELS
from fleetreserve.gate_summary import ROUTE_KEY
from fleetreserve.ledger_terms import JOINT, SPLIT
from fleetreserve.rungs import RUNGS

SHOWN = 3  # breaks named per failed check
NAMES = {"offer": "offer", "r_plus_a": "R + A", "r": "R", "a": "A", "u": "U", "k": "K"}
CELL = 19


def _kwh(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def _pct(value) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _term(block: dict, name: str) -> str:
    return f"{_kwh(block['mean_kwh'][name])} ({_pct(block['share_of_ceiling'][name])})"


def _counts(counts: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items()) or "none"


def _failed_lines(checks: dict) -> list[str]:
    found = [
        ("the identity offer + R + A + U + K = ceiling", checks["identity"]["breaks"]),
        ("the run's own offer means", checks["run_offer_reproduced"]["breaks"]),
        ("the run's ceiling sums", checks["run_ceiling_sums_reproduced"]["breaks"]),
        (
            "the sampled days against the run's ceilings",
            checks["sampled_days_are_the_runs"]["breaks"],
        ),
        ("the plan's order (a term below 0)", checks["order"]["breaks"]),
        ("the sampled days' sets", checks["sample_day_errors"]),
        (
            "the ceiling ratio's one-way ceilings against the run's",
            checks["ceiling_ratio_one_way_reproduced"]["breaks"],
        ),
        (
            "the offer order fixed <= one-way <= two-way (plan T5, outcome O5)",
            checks["offer_order"]["breaks"],
        ),
    ]
    lines = []
    for what, breaks in found:
        if breaks:
            more = len(breaks) - SHOWN
            named = "; ".join(breaks[:SHOWN]) + (f"; and {more} more" if more > 0 else "")
            lines.append(f"CHECK FAILED: {what}, {len(breaks)} breaks: {named}")
    return lines


def _table(label: str, rows: list, names, block: str) -> list[str]:
    width = max([len("configuration"), *(len(name) for name, _ in rows)])
    head = ["configuration".ljust(width), "rung".ljust(8), "scale".ljust(7), "days".ljust(5)]
    head += [NAMES[n].ljust(CELL) for n in names]
    lines = [label, "    " + " ".join(head).rstrip()]
    for name, rungs in rows:
        for rung in RUNGS:
            found = rungs[rung]
            shown = found[block]
            cells = [name.ljust(width), rung.ljust(8), f"{found['scale']:.3g}".ljust(7)]
            cells.append(str(shown["days"]).ljust(5))
            cells += [_term(shown, n).ljust(CELL) for n in names]
            lines.append("    " + " ".join(cells).rstrip())
    return lines


def _unit_lines(unit: str, entry: dict) -> list[str]:
    if "hourly_true_sets" not in entry:
        return [f"{unit}: no ledger ({entry.get('reason', entry['status'])})"]
    true = entry["hourly_true_sets"]
    test, tree = true["test_days"], true["tree_sample"]
    affine, fmt = true["affine_minus_diagonal"], true["format_loss"]
    left = test["left_out_best_causal_missing"]
    left_text = f", {left} left out (best-causal ceiling missing)" if left else ""
    lines = [
        f"{unit} ({entry['test_days']} test days, {len(entry['tree_sample_days'])} drawn for the "
        "tree sample):",
        f"  hourly product, true sets, {test['days']} test days{left_text}: ceiling "
        f"{_kwh(test['mean_ceiling_kwh'])}, V_aff {_kwh(test['mean_kwh']['best_causal'])}, "
        f"R + A {_term(test, 'r_plus_a')}",
        f"  tree sample, {tree['days']} of {tree['drawn']} drawn days (tree outcomes "
        f"{_counts(tree['tree_outcomes'])}): ceiling {_kwh(tree['mean_ceiling_kwh'])}, R "
        f"{_term(tree, 'r')}, A {_term(tree, 'a')}; A is not 0 on {tree['a_not_zero_days']} "
        f"days (largest |A| {_kwh(tree['largest_abs_a_kwh'])} kWh)",
        f"  V_aff - V_diag {_term(affine, 'affine_minus_diagonal')} on {affine['days']} days; "
        f"F {_term(fmt, 'format_loss')} on {fmt['days']} days (largest box in LP (24)'s set "
        f"{_kwh(fmt['mean_kwh']['box'])}; {fmt['sets']} sets, {fmt['base_set']} base set; box "
        f"outcomes {_counts(fmt['box_outcomes'])}), beside the sum",
    ]
    block = entry["block_product"]
    sample, price = block["tree_sample"], block["price_of_the_block"]
    lines.append(
        f"  block product, tree sample: ceiling {_kwh(sample['mean_ceiling_kwh'])}, R "
        f"{_term(sample, 'r')}, A {_term(sample, 'a')} on {sample['days']} of {block['drawn']} "
        f"days; V_aff - V_diag {_term(block['affine_minus_diagonal'], 'affine_minus_diagonal')}; "
        f"price of the block {_term(price, 'price_of_the_block')} of the hourly ceiling on "
        f"{price['days']} days; U and K {block['u']}"
    )
    configurations = entry["configurations"]
    for tier in TIER_LABELS:
        rows = [
            (name, found["tiers"][tier])
            for name, found in configurations.items()
            if tier in found.get("tiers", {})
        ]
        if not rows:
            continue
        lines += _table(
            f"  {TIER_LABELS[tier]}, every test day with V_aff, kWh per day (share of ceiling):",
            rows,
            JOINT,
            "test_days",
        )
        lines += _table(
            f"  {TIER_LABELS[tier]}, tree sample, kWh per day (share of the ceiling):",
            rows,
            SPLIT,
            "tree_sample",
        )
    for name, found in configurations.items():
        if "reason" in found:
            lines.append(f"  {name}: no ledger ({found['reason']})")
    return lines


def _ratio_head(found: dict) -> str:
    """The line on outcome O8's ceiling ratio for the whole run (module text)."""
    return (
        f"Outcome O8 at {found['label']} kWh (the matrix's two_way_budget_kwh{found['note']}): "
        "the ceiling ratio V, the sum over a unit's test days of the exact two-way hindsight "
        "ceiling over the sum of the exact one-way one (true sets, the decision hour and the "
        "recovery hours free), per unit and never pooled; beside it the run's two-way offer as a "
        "share of the two-way ceiling of the same days, per configuration and tier. Discharge D: "
        f"{found['discharge_d']}. Offer ratio X: {found['offer_ratio_x']}."
    )


def _ratio_lines(entry: dict, found: dict, held_out: bool) -> list[str]:
    """One unit's lines on outcome O8's ceiling ratio (module text); none without a ledger."""
    block = entry.get("ceiling_ratio")
    if block is None:
        return []
    head = f"  O8 ceiling ratio V at {found['label']} kWh{found['note']}"
    head += ", held-out unit through the sealed route" if held_out else ""
    if "reason" in block:
        return [f"{head}: not given ({block['reason']}: {', '.join(block['days_not_built'])})"]
    zero = block["one_way_zero_days"]
    if zero:
        more = len(zero) - SHOWN
        named = ", ".join(zero[:SHOWN]) + (f" and {more} more" if more > 0 else "")
        zero_text = (
            f"{len(zero)} test days with a one-way ceiling of 0, in both sums (two-way "
            f"{_kwh(block['two_way_sum_on_one_way_zero_days_kwh'])} kWh on them): {named}"
        )
    else:
        zero_text = "no test day with a one-way ceiling of 0"
    v = _undefined(block["test_days"]) if block["v"] is None else f"{block['v']:.3f}"
    lines = [
        f"{head}, true sets, {block['test_days']} test days: one-way ceiling sum "
        f"{_kwh(block['one_way_sum_kwh'])} kWh, two-way {_kwh(block['two_way_sum_kwh'])} kWh, "
        f"V = {v}; {zero_text}"
    ]
    shares = block["two_way_offer_share_of_two_way_ceiling"]
    if shares:
        lines.append("  the run's two-way offer as a share of the two-way ceiling, same days:")
    for name, tiers in shares.items():
        cells = ", ".join(f"{TIER_LABELS[tier]} {_pct(share)}" for tier, share in tiers.items())
        lines.append(f"    {name}: {cells}")
    return lines


def _undefined(days: int) -> str:
    """The reason a ratio of sums is not defined: no test day, or a denominator of 0."""
    return "not defined (no test days)" if days == 0 else "not defined (one-way sum 0)"


def _x_text(block: dict) -> str:
    x = _undefined(block["test_days"]) if block["x"] is None else f"{block['x']:.3f}"
    days = block["test_days"]
    return (
        f"X = {x}, fixed <= one-way on {block['fixed_one_way_days']} of {days} test days "
        f"({_pct(block['fixed_one_way_share'])}), one-way <= two-way on "
        f"{block['one_way_two_way_days']} of {days} test days "
        f"({_pct(block['one_way_two_way_share'])})"
    )


def _named(days: list) -> str:
    more = len(days) - SHOWN
    return "; ".join(days[:SHOWN]) + (f"; and {more} more" if more > 0 else "")


def _offer_ratio_lines(entry: dict, found: dict) -> list[str]:
    """One unit's lines on outcome O8's offer ratio X (module text); none without a ledger."""
    block = entry.get("offer_ratio")
    if not block:
        return []
    lines = [
        f"  O8 offer ratio X at {found['label']} kWh{found['note']}, the run's two-way offer over "
        "its one-way offer, summed over the test days, and the test days on which fixed <= "
        "one-way and those on which one-way <= two-way, per configuration:"
    ]
    for name, tiers in block.items():
        cells = "; ".join(f"{TIER_LABELS[t]} {_x_text(b)}" for t, b in tiers.items())
        lines.append(f"    {name}: {cells}")
        for tier, b in tiers.items():
            label = TIER_LABELS[tier]
            zero = b["one_way_zero_days"]
            if zero:
                lines.append(
                    f"      one-way offer 0, {label}, in both sums (two-way "
                    f"{_kwh(b['two_way_sum_on_one_way_zero_days_kwh'])} kWh on them), "
                    f"{len(zero)} test days: {_named(zero)}"
                )
            for step, key in (
                ("fixed <= one-way", "fixed_one_way_broken_days"),
                ("one-way <= two-way", "one_way_two_way_broken_days"),
            ):
                broken = b[key]
                if broken:
                    lines.append(
                        f"      ORDER BROKEN, {step}, {label}, {len(broken)} of "
                        f"{b['test_days']} test days: {_named(broken)}"
                    )
    return lines


def route_head(ledger: dict) -> list[str]:
    """The first three lines on the sealed route (module text); none off it."""
    found = ledger.get(ROUTE_KEY)
    if not found:
        return []
    return [
        f"{found['label']} (tag commit {found['tag_commit']}{sealed_route.fix_text(found)}): a "
        "held-out result, the ledger of "
        f"the route run {found['run']}, the scored units' tables from the held-out build "
        f"{found['held_out_inputs']['truth_run']} (the development build "
        f"{found['development_inputs']['truth_run']} checked for its setting, none of its "
        "tables read).",
        found["checks"],
        found["hindsight"],
        *sealed_route.abstention_lines(found["held_out_inputs"]),
    ]


def summary_lines(ledger: dict) -> list[str]:
    run, checks, sample = ledger["run"], ledger["checks"], ledger["tree_sample"]
    lines = [
        *route_head(ledger),
        *rotation_whatif.summary_head(run),
        *model_inputs.summary_head(run),
    ]
    if run["partial"]:
        lines.append(run["label"])
    lines += _failed_lines(checks)
    identity = checks["identity"]
    lines += [
        f"Ledger of plan T9 (outcome O6), hourly product, from the recorded run {run['folder']} "
        f"({run['command']}), {run['setting']} truth setting.",
        "Terms in kWh per counting test day, each with its share of the ceiling of the same days: "
        "the ceiling V_hind (the decision hour free); the offer, the run's scale times V_cert; "
        "R = V_hind - V_caus; A = V_caus - V_aff; U = V_aff - V_cert; K = (1 - scale) V_cert; "
        "offer + R + A + U + K = ceiling. F, the format loss, is beside the sum, not in it.",
        f"R and A apart on the tree sample, {sample['tree_days_per_unit']} test days per unit, "
        f"seed {sample['seed']}; R + A on every test day.",
        f"Identity checked on {identity['days_checked']} day rows and {identity['means_checked']} "
        f"means: largest residual {identity['largest_residual_kwh']:.2e} kWh (tolerance "
        f"{identity['tolerance']:g} of max(1, ceiling)), {len(identity['breaks'])} breaks.",
        "Tiers are separate tables and are never added or averaged.",
        _ratio_head(ledger["ceiling_ratio"]),
    ]
    held_out = bool(ledger.get(ROUTE_KEY))
    for unit, entry in ledger["units"].items():
        lines.append("")
        lines += _unit_lines(unit, entry)
        lines += _ratio_lines(entry, ledger["ceiling_ratio"], held_out)
        lines += _offer_ratio_lines(entry, ledger["ceiling_ratio"])
    return lines
