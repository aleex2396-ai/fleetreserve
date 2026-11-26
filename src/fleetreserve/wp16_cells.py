"""Cells and tables of the outcome tables of WP16, and their CSV and Markdown text.

A cell is a number, a word, an `Interval` (a point and the two ends of its 95% interval, read
from a summary, never computed here) or a `Missing` (the text a slot says when its figure is not
there: "failed: <step> (<reason>)", "not run: <step> (<reason>)", "no entry writes it: ..." or
"unreadable: ..."). A `Missing` is never turned into a number.

CSV: the notes come first, one line each beginning "# " (as in the Markdown file, banner
included); a column that holds an `Interval` in any row is written as four columns, `<name>`,
`<name> low`, `<name> high` and `<name> why no interval` (filled where the ends are missing); a
`Missing` in it is written in the first of the four and the other three are empty. Numbers are
written exactly (`repr` of a float). Markdown: the same table with numbers to six significant
digits, an interval as `point [low, high]` (a point alone as `point (no interval: why)`), then
the notes.
"""

import csv
import io
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Interval:
    """A point and its interval ends; `why` says why the ends are missing (it is not compared)."""

    point: float | None
    low: float | None = None
    high: float | None = None
    why: str = field(default="", compare=False)


@dataclass(frozen=True)
class Missing:
    text: str


@dataclass
class Table:
    name: str  # the file stem of its outputs
    title: str
    columns: tuple[str, ...]
    rows: list[tuple] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, *cells) -> None:
        if len(cells) != len(self.columns):
            raise ValueError(f"{self.name}: {len(cells)} cells for {len(self.columns)} columns")
        self.rows.append(cells)


def _short(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _exact(value) -> str:
    if isinstance(value, Missing):
        return value.text
    if isinstance(value, float):
        return repr(value)
    return _short(value)


def text(cell) -> str:
    """A cell as the Markdown table shows it."""
    if isinstance(cell, Missing):
        return cell.text
    if isinstance(cell, Interval):
        if cell.low is None or cell.high is None:
            note = f" (no interval: {cell.why})" if cell.why else " (no interval)"
            return _short(cell.point) + note
        return f"{_short(cell.point)} [{_short(cell.low)}, {_short(cell.high)}]"
    return _short(cell)


def wide_columns(table: Table) -> set[int]:
    return {
        k
        for k in range(len(table.columns))
        if any(isinstance(row[k], Interval) for row in table.rows)
    }


def csv_text(table: Table) -> str:
    wide = wide_columns(table)
    header: list[str] = []
    for k, name in enumerate(table.columns):
        header += (
            [name, f"{name} low", f"{name} high", f"{name} why no interval"]
            if k in wide
            else [name]
        )
    out = io.StringIO()
    for note in table.notes:  # the same banner and notes as the Markdown file
        out.write("# " + note.replace("\n", " ") + "\n")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    for row in table.rows:
        cells: list[str] = []
        for k, cell in enumerate(row):
            if isinstance(cell, Interval):
                no_ends = cell.low is None or cell.high is None
                why = (cell.why or "the summary writes no interval") if no_ends else ""
                cells += [_exact(cell.point), _exact(cell.low), _exact(cell.high), why]
            elif k in wide:
                cells += [_exact(cell), "", "", ""]
            else:
                cells.append(_exact(cell))
        writer.writerow(cells)
    return out.getvalue()


def markdown_text(table: Table) -> str:
    def clean(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = [f"## {table.title}", ""]
    lines.append("| " + " | ".join(clean(c) for c in table.columns) + " |")
    lines.append("|" + "|".join("---" for _ in table.columns) + "|")
    for row in table.rows:
        lines.append("| " + " | ".join(clean(text(c)) for c in row) + " |")
    if table.notes:
        lines.append("")
        lines += [f"- {note}" for note in table.notes]
    return "\n".join(lines) + "\n"


def interval_of(found) -> "Interval | Missing":
    """The cell of a summary's `{"point": ..., "interval": [low, high] or None, "reason": ...}`
    (`day_bootstrap.bracketed` and `plain`), the ends exactly as the summary has them."""
    if not isinstance(found, dict) or "point" not in found:
        return Missing("no figure: the summary holds none")
    point, ends = found.get("point"), found.get("interval")
    if point is None:
        why = found.get("reason")
        return Missing(f"no figure: {why}" if isinstance(why, str) else "no figure")
    if isinstance(ends, list) and len(ends) == 2:
        return Interval(point, ends[0], ends[1])
    why = found.get("reason")
    return Interval(point, why=why if isinstance(why, str) else "the summary holds no interval")
