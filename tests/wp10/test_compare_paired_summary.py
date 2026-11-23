"""The tables of paired differences in the comparison summary (`compare_summary`), on made-up
per-split offers whose shares above, equal to and below 0 differ from each other and between the
tiers and variants.

The run: unit "u" can be split, unit "w" cannot; tiers A and B; variants "forecast" and
"oracle-departure"; methods `re-solve`, `edf` and `llf`; 10 splits. `re-solve` offers 2 kWh per
counting test day on every split. `edf` offers 2 + f d_s, with the differences
d = 1, 2, 5e-4, 0.3, 0.7, 0, 0, 0, -5e-4, -1 and the factor f = 1 (tier A, forecast), -1 (tier B,
forecast), 2 (tier A, oracle-departure) and -2 (tier B, oracle-departure). `llf` offers 2 on every
split but the first, where it offers 2.5.

1. The headings: for each tier, variant and table, in that order, one heading line with the
   tier's label, the variant and the table's name, then the table's rows; each heading stands over
   the rows of its own tier and variant.
2. The cells of a row: `edf` against `re-solve` shows its mean difference, 5th and 95th
   percentiles, the shares above (5 of 10 or 2 of 10), exactly equal to (3 of 10) and below (2 of
   10 or 5 of 10) 0 each in its own column, the share of the reference's mean offer, the verdict
   and the numbers of paired and left-out splits, as worked by hand here.
3. A unit that cannot be split gets one line and no table.
"""

from fleetreserve import compare_paired, compare_summary

OFFER = "offer_kwh_per_counting_test_day"
SEEDS = tuple(range(1, 11))
DIFFERENCES = (1.0, 2.0, 5e-4, 0.3, 0.7, 0.0, 0.0, 0.0, -5e-4, -1.0)
FACTOR = {
    ("A", "forecast"): 1.0,
    ("B", "forecast"): -1.0,
    ("A", "oracle-departure"): 2.0,
    ("B", "oracle-departure"): -2.0,
}
TIERS = ("A", "B")
VARIANTS = ("forecast", "oracle-departure")
METHODS = ("re-solve", "edf", "llf")
LABELS = {"A": "tier A (driver-protected)", "B": "tier B (deadline contract)"}

# edf against re-solve, worked by hand. With f = 1 the sorted differences are -1, -5e-4, 0, 0, 0,
# 5e-4, 0.3, 0.7, 1, 2: mean 3.0 / 10 = 0.3; the 5th percentile at position 0.45 is
# -1 + 0.45 (1 - 5e-4) = -0.550225; the 95th at position 8.55 is 1 + 0.55 (2 - 1) = 1.55; 5 above
# 0, 3 equal, 2 below; of ref 0.3 / 2 = 15%. A factor f > 0 multiplies the mean and both
# percentiles by f; a factor f < 0 multiplies the mean by f, makes the 5th percentile f times the
# 95th and the 95th f times the 5th, and swaps the shares above and below.
EDF_CELLS = {
    ("A", "forecast"): ["+0.300", "-0.550", "+1.550", "50%", "30%", "20%", "+15.0%"],
    ("B", "forecast"): ["-0.300", "-1.550", "+0.550", "20%", "30%", "50%", "-15.0%"],
    ("A", "oracle-departure"): ["+0.600", "-1.100", "+3.100", "50%", "30%", "20%", "+30.0%"],
    ("B", "oracle-departure"): ["-0.600", "-3.100", "+1.100", "20%", "30%", "50%", "-30.0%"],
}


def _row(tier, variant, method, seed, offer):
    return {
        "unit": "u",
        "variant": variant,
        "tier": tier,
        "method": method,
        "seed": seed,
        OFFER: offer,
    }


def _document() -> dict:
    rows = []
    for tier in TIERS:
        for variant in VARIANTS:
            f = FACTOR[(tier, variant)]
            for s, d in zip(SEEDS, DIFFERENCES, strict=True):
                rows.append(_row(tier, variant, "re-solve", s, 2.0))
                rows.append(_row(tier, variant, "edf", s, 2.0 + f * d))
                rows.append(_row(tier, variant, "llf", s, 2.5 if s == 1 else 2.0))
    paired = compare_paired.paired(rows, ["u"], VARIANTS, TIERS, METHODS, SEEDS)
    return {
        "units": {"u": {}, "w": {}},
        "run": {"tiers": list(TIERS), "variants": list(VARIANTS)},
        "methods": list(METHODS),
        "paired_differences": {"units": paired},
    }


def _tables(lines: list[str]) -> list[tuple[str, list[list[str]]]]:
    """(heading, the cells of each row under it) for every heading line, in order."""
    found = []
    for line in lines:
        if line.startswith(" tier "):
            found.append((line, []))
        elif found and line.startswith("  ") and line.split()[0] in METHODS:
            found[-1][1].append(line.split())
    return found


def test_each_heading_stands_over_its_own_tier_and_variant():
    lines = compare_summary._paired_lines(_document())
    tables = _tables(lines)
    expected = [
        f" {LABELS[tier]}, {variant}, {name}"
        for tier in TIERS
        for variant in VARIANTS
        for name in ("against re-solve", "other pairs")
    ]
    assert [heading for heading, _ in tables] == expected
    for heading, rows in tables:
        tier = "A" if heading.startswith(" tier A ") else "B"
        variant = heading.split(", ")[1]
        if heading.endswith("against re-solve"):
            assert [r[:2] for r in rows] == [["edf", "re-solve"], ["llf", "re-solve"]]
            assert rows[0][2:9] == EDF_CELLS[(tier, variant)], heading
        else:
            assert [r[:2] for r in rows] == [["llf", "edf"]]


def test_the_cells_of_a_paired_row():
    lines = compare_summary._paired_lines(_document())
    tables = dict(_tables(lines))
    for (tier, variant), cells in EDF_CELLS.items():
        edf, llf = tables[f" {LABELS[tier]}, {variant}, against re-solve"]
        assert edf[2:9] == cells
        assert edf[9:] == ["not", "separated", "10", "0"]
        # llf against re-solve: one difference of 0.5 and nine of 0: mean 0.05, p5 0, p95 0.275
        # (position 8.55 between 0 and 0.5), 10% above, 90% equal, 0% below, of ref 2.5%.
        assert llf[2:] == [
            *["+0.050", "+0.000", "+0.275", "10%", "90%", "0%", "+2.5%"],
            *["not", "separated", "10", "0"],
        ]


def test_a_unit_that_cannot_be_split_gets_one_line():
    lines = compare_summary._paired_lines(_document())
    assert "w: no paired differences: the unit cannot be split" in lines
    assert lines.count("u, paired differences:") == 1
    assert lines[:2] == ["", compare_summary.PAIRED_TEXT]
