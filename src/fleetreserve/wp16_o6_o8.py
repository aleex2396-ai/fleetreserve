"""Outcomes O6 and O8: the ledger's terms and the two-way ratios (plan sections 5 (T9), 9, 11).

"O6: ledger terms R, A, U, K, F% of the ceiling per unit and product (for the hourly product R is
reported as measured against the ceiling with the decision hour's power free, and A as measured on
a day sample, zero expected by T4 ...); for the block product R, A and F, plus the price of the
block, its calibrated offers being cut."
"O8, B=5: offer ratio X (pass: >= 1 daily), ceiling ratio V, sum over sum ...; the discharge D
kWh per car-day is reported as not run."

The step `ledger-lgbm_tail` (`fleetreserve.ledger_run`) writes `ledger.json`; the headline
configuration of each unit is that of the matrix of `learn-lgbm_tail` (`wp16_units`).

Readings (alternatives in brackets):
  R1. The hourly O6 row is tier A, one-way rung, at the unit's headline configuration, on the
      tree sample (the days on which the corner tree was solved), where offer + R + A + U + K is
      the ceiling; every term is the file's share of the mean ceiling of the same days
      (`ledger_terms.means`). The same terms over all test days (R and A together, U, K) are in the
      table beside. [Alternative: the two-way rung or tier B.]
  R2. F is the hourly format loss `format_loss` and V_aff - V_diag `affine_minus_diagonal`, both
      beside the sum and not in it. For the block product the file holds R, A, V_aff - V_diag and
      the price of the block, and U and K as "not computed": the block's F is not written by any
      entry of the run list, and is named as such.
  R3. O8's X is `offer_ratio.<configuration>.A.x` (the sum of the two-way offers over the sum of
      the one-way offers over the unit's test days), V is `ceiling_ratio.v` (the two-way ceilings'
      sum over the one-way ceilings' sum, B = 5 kWh), and the two-way offer as a share of the
      two-way ceiling is beside them; the daily pass of X (one-way <= two-way) is in O5's table.
      D is "not run" as the plan says.
"""

from fleetreserve.wp16_cells import Missing, Table
from fleetreserve.wp16_drive import Missed, Source
from fleetreserve.wp16_learn import Runs, names_for

LEDGER = "ledger-lgbm_tail"
LEARN = "learn-lgbm_tail"
TERMS = ("offer", "r", "a", "u", "k")
O6_COLUMNS = (
    "unit",
    "product",
    "days",
    "offer",
    "R",
    "A",
    "U",
    "K",
    "F (format loss)",
    "V_aff - V_diag",
    "price of the block",
    "R + A over all test days",
    "days with A not 0",
)
O8_COLUMNS = (
    "unit",
    "configuration",
    "test days",
    "X: two-way offers over one-way offers",
    "V: two-way ceilings over one-way ceilings (B = 5 kWh)",
    "two-way offer as a share of the two-way ceiling",
    "test days with a one-way ceiling of 0",
    "D: discharge (kWh per car-day)",
)


def _share(block, key):
    shares = block.get("share_of_ceiling") if isinstance(block, dict) else None
    return shares.get(key) if isinstance(shares, dict) else None


def _unit_rows(table: Table, unit: str, entry: dict, name) -> None:
    if not isinstance(entry, dict) or "hourly_true_sets" not in entry:
        why = entry.get("reason") or entry.get("status") if isinstance(entry, dict) else None
        table.add(
            unit, "both", "", Missing(f"no figure: {why or 'not in the ledger'}"), *([""] * 9)
        )
        return
    tiers = entry.get("configurations", {}).get(name, {}).get("tiers", {})
    rung = tiers.get("A", {}).get("one-way", {}) if isinstance(tiers, dict) else {}
    tree, tests = rung.get("tree_sample", {}), rung.get("test_days", {})
    true_sets = entry["hourly_true_sets"]
    table.add(
        unit,
        "hourly",
        tree.get("days"),
        *(_share(tree, term) for term in TERMS),
        _share(true_sets.get("format_loss"), "format_loss"),
        _share(true_sets.get("affine_minus_diagonal"), "affine_minus_diagonal"),
        "",
        _share(tests, "r_plus_a"),
        true_sets.get("tree_sample", {}).get("a_not_zero_days"),
    )
    block = entry.get("block_product", {})
    table.add(
        unit,
        "block",
        block.get("drawn"),
        Missing("no entry writes it: the block's calibrated offers are cut"),
        _share(block.get("tree_sample"), "r"),
        _share(block.get("tree_sample"), "a"),
        block.get("u"),
        block.get("k"),
        Missing("no entry writes it"),
        _share(block.get("affine_minus_diagonal"), "affine_minus_diagonal"),
        _share(block.get("price_of_the_block"), "price_of_the_block"),
        "",
        "",
    )


def build(source: Source, runs: Runs, units: list) -> list[Table]:
    o6 = Table(
        "o6-ledger",
        "O6: where the offer is lost (shares of the ceiling, tier A, one-way)",
        O6_COLUMNS,
    )
    o8 = Table("o8-two-way", "O8: two-way charging at B = 5 kWh", O8_COLUMNS)
    try:
        document = source.json(LEDGER, "ledger.json")
        names = names_for(runs.get(LEARN), units, "main")
    except Missed as problem:
        o6.add("all units", "", "", Missing(problem.text), *([""] * 9))
        o8.add("all units", "", "", Missing(problem.text), "", "", "", "")
        return [o6, o8]
    found = document.get("units") if isinstance(document.get("units"), dict) else {}
    for unit in units:
        name = names[unit]
        entry = found.get(unit)
        _unit_rows(o6, unit, entry, name)
        ratio = entry.get("ceiling_ratio", {}) if isinstance(entry, dict) else {}
        offers = (
            entry.get("offer_ratio", {}).get(name, {}).get("A") if isinstance(entry, dict) else None
        )
        offers = offers if isinstance(offers, dict) else {}
        share = ratio.get("two_way_offer_share_of_two_way_ceiling", {}).get(name, {}).get("A")
        zero = ratio.get("one_way_zero_days")
        if not offers and not ratio.get("v"):
            why = ratio.get("reason", "not in the ledger")
            o8.add(unit, name or "", "", Missing(f"no figure: {why}"), "", "", "", "")
            continue
        o8.add(
            unit,
            name,
            offers.get("test_days"),
            offers.get("x"),
            ratio.get("v"),
            share,
            len(zero) if isinstance(zero, list) else None,
            "not run (the owner's decision of 16 November 2026)",
        )
    o6.notes.append("Readings R1 and R2 of the module text of fleetreserve.wp16_o6_o8.")
    o8.notes.append("Readings R3 of the module text of fleetreserve.wp16_o6_o8.")
    return [o6, o8]
