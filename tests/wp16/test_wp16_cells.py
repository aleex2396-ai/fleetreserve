"""`wp16_cells`: the CSV and Markdown text of a table, and the cell of a summary's interval.

Red when an interval's ends are changed or dropped, when a missing figure turns into a number, or
when a table's columns do not line up with its rows.
"""

import csv
import io

import pytest

from fleetreserve.wp16_cells import (
    Interval,
    Missing,
    Table,
    csv_text,
    interval_of,
    markdown_text,
    text,
)


def made_up() -> Table:
    table = Table("t", "A title", ("name", "x", "y"))
    table.add("a", Interval(1.5, 1.0, 2.25), 3)
    table.add("b", Missing("failed: step (exit status 1)"), True)
    table.add("c", Interval(0.1), None)
    table.notes.append("a note")
    return table


def test_an_interval_column_is_written_as_four_columns_with_exact_numbers_and_the_notes():
    lines = csv_text(made_up()).splitlines()
    assert lines[0] == "# a note"  # the banner and notes of the Markdown file come first
    rows = list(csv.reader(io.StringIO("\n".join(lines[1:]))))
    assert rows[0] == ["name", "x", "x low", "x high", "x why no interval", "y"]
    assert rows[1] == ["a", "1.5", "1.0", "2.25", "", "3"]
    assert rows[2] == ["b", "failed: step (exit status 1)", "", "", "", "yes"]
    assert rows[3] == ["c", "0.1", "", "", "the summary writes no interval", ""]


def test_a_point_without_ends_says_why_in_both_files():
    table = Table("t", "T", ("x",))
    table.add(Interval(2.0, why="the headline differs by site type"))
    assert csv_text(table).splitlines()[1] == "2.0,,,the headline differs by site type"
    assert "2 (no interval: the headline differs by site type)" in markdown_text(table)


def test_a_missing_figure_is_never_a_number():
    table = Table("t", "T", ("x",))
    table.add(Missing("not run: step (an earlier step)"))
    assert csv_text(table).splitlines()[1] == "not run: step (an earlier step)"
    assert text(Missing("no entry writes it")) == "no entry writes it"


def test_markdown_shows_the_interval_and_the_notes():
    page = markdown_text(made_up())
    assert page.startswith("## A title\n")
    assert "| a | 1.5 [1, 2.25] | 3 |" in page
    assert "| c | 0.1 (no interval) |  |" in page
    assert "| b | failed: step (exit status 1) | yes |" in page
    assert "- a note" in page


def test_a_pipe_in_a_cell_does_not_break_the_row():
    table = Table("t", "T", ("x",))
    table.add("a | b")
    assert "| a \\| b |" in markdown_text(table)


def test_a_row_of_the_wrong_length_is_refused():
    table = Table("t", "T", ("x", "y"))
    with pytest.raises(ValueError, match="1 cells for 2 columns"):
        table.add(1)


def test_the_cell_of_a_summary_interval_keeps_its_ends():
    found = interval_of({"point": 2.0, "interval": [1.0, 5.0]})
    assert found == Interval(2.0, 1.0, 5.0)
    none = interval_of({"point": 2.0, "interval": None})
    assert none == Interval(2.0) and none.why


def test_a_summary_with_no_point_is_a_missing_figure():
    found = interval_of({"point": None, "interval": None, "reason": "the run's scale is 0"})
    assert found == Missing("no figure: the run's scale is 0")
    assert isinstance(interval_of(None), Missing)
