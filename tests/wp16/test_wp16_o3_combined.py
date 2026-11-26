"""O3's combined share per method (`wp16_o3.build_combined`) from a made-up `o3-days.csv`.

Red when the combined share is not the sum over units of each unit's mean value over the sum over
units of each unit's mean ceiling: the hand case has two units with different day counts, so the
mean of the unit shares, the share over all days pooled and the mean of the per-day shares each
give another figure. Also red when a day with a value missing is counted, when a unit with no day
on which every value is present is not named, when a missing file is read as a figure, or when
the older O3 tables change.
"""

import csv
import io

import pytest
from wp16_made_up import UNITS, MadeUp, run_list_document

from fleetreserve import o3_run, o3_summary, wp16_cells, wp16_o3, wp16_run
from fleetreserve.wp16_drive import Source, find_drive

HEADER = o3_run.DAYS_HEADER
METHODS = wp16_o3.METHODS


def day(unit, product, ceiling, lp24, general, bound=None):
    """One row of `o3-days.csv` in the writer's columns. The two LP (27) boxes found take LP (24)'s
    value, the bound and the split without the hour-0 gain take the split's; a `bound` of "" is a
    missing value."""
    kwh = {
        "ceiling": ceiling,
        "lp24": lp24,
        "lp27_smallest_found": lp24,
        "lp27_largest_found": lp24,
        "lp27_largest_bound": general if bound is None else bound,
        "general_split": general,
        "general_split_no_hour0_gain": general,
    }
    row = dict.fromkeys(HEADER, "")
    row.update(unit=unit, site="s", date="2026-01-01", cars=1, service_hours=1, product=product)
    for key, value in kwh.items():
        row[f"{key}_kwh"] = "" if value == "" else repr(float(value))
        row[f"{key}_outcome"] = "missing" if value == "" else "optimal"
    return row


def csv_of(rows):
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


# u1: three days with every value (hourly ceilings 8, 10, 12) and one day with no bound whose values
# are huge, so a rule that keeps it gives another figure. u2: one hourly day. u3: no hourly day with
# every value. Block: u1 and u3 have one day, u2 has none.
ROWS = [
    day("u1", "hourly", 8, 1, 4),
    day("u1", "hourly", 10, 2, 5),
    day("u1", "hourly", 12, 3, 6),
    day("u1", "hourly", 100, 100, 100, bound=""),
    day("u2", "hourly", 2, 2, 0.5),
    day("u3", "hourly", 6, 6, 6, bound=""),
    day("u1", "block", 4, 1, 2),
    day("u2", "block", 9, 9, 9, bound=""),
    day("u3", "block", 4, 3, 4),
]


def tables_of(tmp_path, files):
    made = MadeUp(tmp_path)
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.drive([("o3", files)])
    source = Source(tmp_path, find_drive(tmp_path))
    return source, wp16_o3.build_combined(source, UNITS)


def combined_rows(table):
    return {(row[0], row[1]): row for row in table.rows}


def test_the_combined_share_is_the_sum_of_unit_means_over_the_sum_of_unit_mean_ceilings(tmp_path):
    _, (combined, _) = tables_of(tmp_path, {"o3.json": {}, "o3-days.csv": csv_of(ROWS)})
    assert combined.columns[:7] == (
        "product",
        "value",
        "units summed",
        "days summed",
        "sum of unit mean values (kWh)",
        "sum of unit mean ceilings (kWh)",
        "combined share of the ceiling (%)",
    )
    row = combined_rows(combined)[("hourly", "lp24")]
    # u1: mean value 2, mean ceiling 10 (3 days); u2: 2 and 2 (1 day): 100 x 4 / 12
    assert row[2:6] == (2, 4, pytest.approx(4.0), pytest.approx(12.0))
    assert row[6] == pytest.approx(100 * 4 / 12)
    assert row[7] == "u3"
    # the three other rules, each of which gives another figure on this case (red under each)
    mean_of_unit_shares = 100 * (2 / 10 + 2 / 2) / 2
    pooled_days = 100 * (1 + 2 + 3 + 2) / (8 + 10 + 12 + 2)
    mean_of_day_shares = 100 * (1 / 8 + 2 / 10 + 3 / 12 + 2 / 2) / 4
    for wrong in (mean_of_unit_shares, pooled_days, mean_of_day_shares):
        assert row[6] != pytest.approx(wrong, abs=1.0)
    general = combined_rows(combined)[("hourly", "general_split")]
    assert general[6] == pytest.approx(100 * (5 + 0.5) / 12)
    assert general[6] != pytest.approx(100 * (5 / 10 + 0.5 / 2) / 2, abs=1.0)
    assert general[6] != pytest.approx(100 * (4 + 5 + 6 + 0.5) / 32, abs=1.0)
    # the sum of mean ceilings is the same for every method of a product
    assert all(r[5] == pytest.approx(12.0) for r in combined.rows if r[0] == "hourly")


def test_every_value_name_of_the_same_days_table_has_a_row_in_each_product(tmp_path):
    _, (combined, _) = tables_of(tmp_path, {"o3.json": {}, "o3-days.csv": csv_of(ROWS)})
    assert [(r[0], r[1]) for r in combined.rows] == [
        (product, method) for product in wp16_o3.PRODUCTS for method in wp16_o3.METHODS
    ]


def test_a_day_with_a_missing_value_is_left_out_of_every_method(tmp_path):
    # u1's fourth hourly day has a value for the ceiling and LP (24) but no bound: with it the
    # unit's mean ceiling would be 32.5, not 10
    _, (combined, units) = tables_of(tmp_path, {"o3.json": {}, "o3-days.csv": csv_of(ROWS)})
    first = {(r[0], r[1], r[3]): r for r in units.rows if r[3] != "" and not hasattr(r[3], "text")}
    assert first[("u1", "hourly", "lp24")][2] == 3
    assert first[("u1", "hourly", "lp24")][4:] == (pytest.approx(2.0), pytest.approx(10.0))
    assert combined_rows(combined)[("hourly", "lp24")][3] == 4


def test_a_unit_with_no_day_with_every_value_is_named_in_the_note_and_the_row(tmp_path):
    _, (combined, units) = tables_of(tmp_path, {"o3.json": {}, "o3-days.csv": csv_of(ROWS)})
    notes = " ".join(combined.notes)
    assert "Left out of the hourly sums, no day on which every value is present: u3." in notes
    assert "Left out of the block sums, no day on which every value is present: u2." in notes
    assert combined_rows(combined)[("block", "lp24")][7] == "u2"
    assert combined_rows(combined)[("block", "lp24")][2:4] == (2, 2)  # u1 and u3, one day each
    # the left-out unit has its own row in the per-unit table and no sum takes it
    left = [r for r in units.rows if r[0] == "u3" and r[1] == "hourly"]
    assert len(left) == 1 and left[0][2] == 0 and "left out" in left[0][3].text
    # the note states the rule and when it was applied
    assert "after the held-out results were seen" in notes
    assert "the sum over units of each unit's value per counting day" in notes


def test_the_per_unit_means_add_to_the_combined_sums(tmp_path):
    _, (combined, units) = tables_of(tmp_path, {"o3.json": {}, "o3-days.csv": csv_of(ROWS)})
    for (product, method), row in combined_rows(combined).items():
        mine = [r for r in units.rows if r[1] == product and r[3] == method]
        assert row[2] == len(mine)
        assert row[4] == pytest.approx(sum(r[4] for r in mine))
        assert row[5] == pytest.approx(sum(r[5] for r in mine))
        assert row[3] == sum(r[2] for r in mine)


def test_no_unit_with_a_day_gives_missing_and_never_a_zero(tmp_path):
    rows = [day("u1", "hourly", 8, 1, 4, bound="")]
    _, (combined, _) = tables_of(tmp_path, {"o3.json": {}, "o3-days.csv": csv_of(rows)})
    row = combined_rows(combined)[("hourly", "lp24")]
    assert row[2:4] == (0, 0)
    assert isinstance(row[6], wp16_cells.Missing) and "no unit has a day" in row[6].text
    assert row[7] == "u1, u2, u3"
    zero = [day("u1", "hourly", 0, 0, 0)]
    _, (combined, _) = tables_of(tmp_path / "zero", {"o3.json": {}, "o3-days.csv": csv_of(zero)})
    share = combined_rows(combined)[("hourly", "lp24")][6]
    assert isinstance(share, wp16_cells.Missing) and "ceiling is zero" in share.text


def test_a_step_without_the_per_day_file_or_a_step_that_did_not_run_gives_missing(tmp_path):
    _, (combined, units) = tables_of(tmp_path, {"o3.json": {}})
    cell = combined.rows[0][2]
    assert isinstance(cell, wp16_cells.Missing) and cell.text.startswith("unreadable: o3/")
    assert isinstance(units.rows[0][2], wp16_cells.Missing)
    made = MadeUp(tmp_path / "none")
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.drive([("sealed-truth", {"truth-summary.json": {}})])
    source = Source(tmp_path / "none", find_drive(tmp_path / "none"))
    (combined, _) = wp16_o3.build_combined(source, UNITS)
    assert "no entry writes it" in combined.rows[0][2].text
    # a column the writer does not have is unreadable, not a share
    bad = csv_of(ROWS).replace("lp24_kwh", "lp24_kwh_x", 1)
    _, (combined, _) = tables_of(tmp_path / "bad", {"o3.json": {}, "o3-days.csv": bad})
    assert "no column lp24_kwh" in combined.rows[0][2].text


def test_the_columns_read_are_the_ones_the_o3_step_writes():
    names = {f"{key}_kwh" for key in ("ceiling", *METHODS)} | {"unit", "product"}
    assert names <= set(o3_run.DAYS_HEADER)


def test_the_older_o3_tables_are_the_same_with_or_without_the_new_ones(tmp_path):
    document = {
        "units": {"u1": {"hourly": {"ceiling": {"days": 3, "kwh": {"mean": 5.0}}}, "block": {}}},
        "all_present": {"u1": {"hourly": {"days": 3}, "block": {"days": 0}}},
    }
    files = {"o3.json": document, "o3-days.csv": csv_of(ROWS)}
    source, new = tables_of(tmp_path / "with", files)
    first = [wp16_cells.csv_text(t) for t in wp16_o3.build(source, UNITS)[0]]
    source, _ = tables_of(tmp_path / "without", {"o3.json": document})
    second = [wp16_cells.csv_text(t) for t in wp16_o3.build(source, UNITS)[0]]
    assert first == second
    assert [t.name for t in wp16_o3.build(source, UNITS)[0]] == [
        "o3-ceiling-shares",
        "o3-days",
        "o3-same-days",
    ]
    assert [t.name for t in new] == ["o3-combined", "o3-combined-units"]


def test_the_figures_and_tables_command_writes_both_new_tables_after_the_older_o3_ones(tmp_path):
    source, _ = tables_of(tmp_path, {"o3.json": {}, "o3-days.csv": csv_of(ROWS)})
    names = [t.name for t in wp16_run.collect(source)[0]]
    start = names.index("o3-same-days")
    assert names[start : start + 3] == ["o3-same-days", "o3-combined", "o3-combined-units"]


def present(hourly, block):
    """An `o3.json` with the `all_present` day counts of the three units: (u1, u2, u3) per
    product."""
    return {
        "all_present": {
            unit: {"hourly": {"days": hourly[i]}, "block": {"days": block[i]}}
            for i, unit in enumerate(UNITS)
        }
    }


@pytest.mark.parametrize("cell", ["nan", "inf", "-inf"])
def test_a_cell_that_is_not_finite_is_unreadable_never_a_sum(tmp_path, cell):
    rows = [dict(row) for row in ROWS]
    rows[0]["lp24_kwh"] = cell
    _, (combined, units) = tables_of(tmp_path, {"o3.json": {}, "o3-days.csv": csv_of(rows)})
    first = combined.rows[0][2]
    assert isinstance(first, wp16_cells.Missing)
    assert repr(cell) in first.text and "not a finite number" in first.text
    assert isinstance(units.rows[0][2], wp16_cells.Missing)


def test_a_cell_that_is_not_a_number_is_unreadable_never_a_skipped_day(tmp_path):
    rows = [dict(row) for row in ROWS]
    rows[1]["general_split_kwh"] = "five"
    _, (combined, _) = tables_of(tmp_path, {"o3.json": {}, "o3-days.csv": csv_of(rows)})
    cell = combined.rows[0][2]
    assert isinstance(cell, wp16_cells.Missing) and "'five' is not a number" in cell.text
    assert len(combined.rows) == 1


def test_a_tiny_negative_ceiling_has_no_share_and_does_not_stop_the_table(tmp_path):
    rows = [day("u1", "hourly", -1e-9, 0, 0), day("u2", "hourly", 2, 2, 0.5)]
    _, (combined, _) = tables_of(tmp_path, {"o3.json": {}, "o3-days.csv": csv_of(rows)})
    only_u1 = [day("u1", "hourly", -1e-9, 0, 0)]
    _, (alone, _) = tables_of(tmp_path / "alone", {"o3.json": {}, "o3-days.csv": csv_of(only_u1)})
    share = combined_rows(alone)[("hourly", "lp24")][6]
    assert isinstance(share, wp16_cells.Missing) and "ceiling is zero" in share.text
    assert combined_rows(alone)[("hourly", "lp24")][2] == 1
    assert combined_rows(combined)[("hourly", "lp24")][6] == pytest.approx(100 * 2 / (2 - 1e-9))


def test_the_kept_day_counts_agree_with_the_all_present_block_of_the_summary(tmp_path):
    files = {"o3.json": present((3, 1, 0), (1, 0, 1)), "o3-days.csv": csv_of(ROWS)}
    _, (combined, _) = tables_of(tmp_path, files)
    assert combined_rows(combined)[("hourly", "lp24")][6] == pytest.approx(100 * 4 / 12)
    # one unit's count differs: the tables say so, naming the unit, and print no figure
    for counts, unit in (((3, 2, 0), "u2"), ((3, 1, 1), "u3")):
        files = {"o3.json": present(counts, (1, 0, 1)), "o3-days.csv": csv_of(ROWS)}
        _, (combined, units) = tables_of(tmp_path / unit, files)
        cell = combined.rows[0][2]
        assert isinstance(cell, wp16_cells.Missing) and unit in cell.text and "hourly" in cell.text
        assert len(combined.rows) == 1 and isinstance(units.rows[0][2], wp16_cells.Missing)
    files = {"o3.json": present((3, 1, 0), (1, 1, 1)), "o3-days.csv": csv_of(ROWS)}
    _, (combined, _) = tables_of(tmp_path / "block", files)
    assert "u2" in combined.rows[0][2].text and "block" in combined.rows[0][2].text


def test_the_zero_limit_is_the_one_of_the_summary_module():
    assert wp16_o3.ZERO is o3_summary.ZERO


def test_the_note_says_the_sums_may_cover_fewer_than_nine_units(tmp_path):
    _, (combined, _) = tables_of(tmp_path, {"o3.json": {}, "o3-days.csv": csv_of(ROWS)})
    assert (
        "The sums then cover fewer than nine units; the column 'units summed' gives how many."
        in " ".join(combined.notes)
    )
