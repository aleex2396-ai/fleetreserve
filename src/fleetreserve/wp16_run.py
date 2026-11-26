"""The outcome tables and figures of WP16, from the one drive of the run from the tag (plan sections
11 and 13, WP16: "Pass: every number traces to a logged run").

Command line, from the project's top folder (usually under `fleetreserve.run_record`, which sets
`FLEETRESERVE_RUN_DIR`; the figures need the dependency group `figures`):
    python -m fleetreserve.wp16_run [--root <folder>] [--out <folder>]
        [--dev-days-run <run folder>] [--dial-choice-run <run folder>]

`--root`: the project's top folder (default: the folder above `src/`). `--out`: where the files are
written (default: `FLEETRESERVE_RUN_DIR`; one of the two is needed). `--dev-days-run` and
`--dial-choice-run`: a finished development run of `dev_day_checks` or of `dial_choice`, as a path
under the top folder; each adds one table of development figures (`wp16_devdays`: `dev-days-checks`
and `dev-dial-choice`, with the run named in the manifest under `development_runs`). Without them
no table, file or exit status changes.

It finds the drive as `wp16_drive.find_drive` says, reads the summaries of the steps that passed,
fills the slots of plan section 11 and writes, for each table of the builders (`wp16_o1`,
`wp16_eprime` (the share of days in the event E' of T2), `wp16_compare`, `wp16_o2` (the block
product and its day counts), `wp16_o3`, `wp16_o4`, `wp16_o5`, `wp16_o6_o8`, `wp16_o7`, `wp16_o9`,
`wp16_o10`) a CSV and a Markdown file of its own name,
`outcomes.md` (every table in one page), the figures (`wp16_figures`: O3, O4, O9, O10) and
`wp16-manifest.json`. A slot whose step failed or did not run says "failed: <step> (<reason>)" or
"not run: <step> (<reason>)", never a figure from elsewhere (plan section 13, WP14). With a drive
that is not complete every table opens with the line that says so: it is not a passed held-out
run.

`wp16-manifest.json` holds the commit of `--root` (`commit`), the commit of the code that ran
(`code_commit`: HEAD of the checkout this module is imported from; when it differs from `commit`
both are in the manifest and a line says so on standard output, nothing is refused),
`code_dirty` (whether `git status --porcelain` of that same checkout lists anything: true, false
or "unknown: <why>", a line on standard output when true, nothing refused), the command as
run, the drive (its file name,
SHA-256, commit, whether it is complete and each step that did not pass), the SHA-256 of every
summary file read, by its path under the top folder (the run list and matrices among them), and the
SHA-256 of every file written, and `made_apart`: the sentence and each table row whose figures
were made after the run's output was seen, at a later commit, and fill no slot of plan section 11
(an empty list when none). Intervals are read from the summaries (`day-intervals.json`, 95%,
percentile, 2,000 resamples, only for `lgbm_tail` and `hazard_tuned_inputs`); the one exception is
the pooled interval of a model whose headline configurations differ by site type, made from the
per-day file by `day_intervals`' own functions (`wp16_mixed`) and marked made apart in its row.

Exit status: 0 every file written from a complete drive and no slot unreadable; 1 files written,
but the drive is not complete or a slot says failed, not run, could not run or unreadable (the
pass mark of WP14 does not hold, or a summary could not be read: the files say which); 2 could not
run (no drive
that can be taken, no output folder, a folder that cannot be written, matplotlib missing).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from fleetreserve import (
    wp16_cells,
    wp16_compare,
    wp16_devdays,
    wp16_eprime,
    wp16_figures,
    wp16_o1,
    wp16_o2,
    wp16_o3,
    wp16_o4,
    wp16_o5,
    wp16_o6_o8,
    wp16_o7,
    wp16_o9,
    wp16_o10,
)
from fleetreserve.wp16_cells import Missing, Table
from fleetreserve.wp16_drive import Missed, NoDrive, Source, find_drive
from fleetreserve.wp16_failed import failed_configurations
from fleetreserve.wp16_learn import Runs
from fleetreserve.wp16_mixed import MARK
from fleetreserve.wp16_units import abstained, nine

TOP = Path(__file__).resolve().parents[2]
RUN_DIR_VARIABLE = "FLEETRESERVE_RUN_DIR"
MANIFEST_NAME = "wp16-manifest.json"
OUTCOMES_NAME = "outcomes.md"
BAD_START = ("failed:", "not run:", "could not run:", "unreadable:")


def head_commit(root: Path) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as problem:  # no git executable
        raise NoDrive(f"git rev-parse HEAD could not run: {problem}") from problem
    if done.returncode != 0:
        raise NoDrive(f"git rev-parse HEAD failed: {done.stderr.strip()}")
    return done.stdout.strip()


def code_head() -> str:
    """HEAD of the checkout this module is imported from (`TOP`), or why it cannot be read: the
    manifest records it, so a checkout that has no git is named, not refused."""
    try:
        return head_commit(TOP)
    except NoDrive as problem:
        return f"unknown: {problem}"


def code_dirty() -> "bool | str":
    """Whether the checkout this module is imported from (`TOP`) has changes `git status
    --porcelain` lists (untracked files included): true, false, or "unknown: <why>" when git
    cannot run or fails; the manifest records it, nothing is refused."""
    try:
        done = subprocess.run(
            ["git", "-C", str(TOP), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as problem:  # no git executable
        return f"unknown: git status --porcelain could not run: {problem}"
    if done.returncode != 0:
        return f"unknown: git status --porcelain failed: {done.stderr.strip()}"
    return bool(done.stdout.strip())


def banner(drive, failed: list[str], unreadable: list[str] = ()) -> str:
    """The line every table opens with: whether the drive is complete, and each configuration that
    failed on a unit although its step exited 0 (module text)."""
    head = f"Drive {drive.name} (commit {drive.commit}): "
    lost = "; learn results that cannot be read: " + "; ".join(unreadable) if unreadable else ""
    if not drive.complete:
        return (
            head
            + "NOT COMPLETE, so this is not a passed held-out run (plan section 13, WP14): "
            + "; ".join(drive.problems())
            + ("; failed configurations: " + "; ".join(failed) if failed else "")
            + lost
        )
    if unreadable and not failed:
        return (
            head + f"every step exited 0, but {len(unreadable)} learn result(s) cannot be read, "
            "so this is not a passed held-out run: " + "; ".join(unreadable)
        )
    if failed:
        return (
            head + f"every step exited 0, but {len(failed)} configuration(s) failed on a unit, so "
            "WP14's pass mark fails and this is not a passed held-out run: "
            + "; ".join(failed)
            + lost
        )
    return head + "complete; every step ran and passed, and no configuration failed on a unit."


def steps_table(drive) -> Table:
    table = Table("steps", "The steps of the drive", ("step", "status", "reason", "run folder"))
    for step in drive.steps.values():
        table.add(step.label, step.status, step.reason, step.run or "")
    return table


def units_table(units: list, absent: dict, unknown: str | None = None) -> Table:
    """The nine units and whether the truth build gave each a window; with `unknown` (why the
    truth summary could not be read) no unit is called built."""
    table = Table("units", "The nine held-out units", ("unit", "window", "reason"))
    for unit in units:
        if unknown is not None:
            table.add(unit, "unknown", Missing(unknown))
        else:
            table.add(unit, "none" if unit in absent else "built", absent.get(unit, ""))
    return table


def collect(source: Source) -> tuple[list[Table], list[dict]]:
    """Every table and figure spec of the builders, in the order of the plan's outcomes."""
    units = nine(source)
    unknown = None
    try:
        absent = abstained(source)
    except Missed as problem:
        absent, unknown = {}, problem.text
    runs = Runs(source)
    tables = [steps_table(source.drive), units_table(units, absent, unknown)]
    figures: list[dict] = []
    tables += wp16_o1.build(source, runs, units, absent)
    tables += wp16_eprime.build(source, runs, units, absent)
    tables += wp16_compare.build(source, units)
    tables += wp16_o2.build(source, units)
    tables += wp16_o2.build_day_counts(source, units)
    more, drawn = wp16_o3.build(source, units)
    tables, figures = tables + more, figures + drawn
    tables += wp16_o3.build_combined(source, units)
    more, drawn = wp16_o4.build(runs, units, absent)
    tables, figures = tables + more, figures + drawn
    tables += wp16_o4.build_points(runs, units, absent)
    tables += wp16_o5.build(source, runs, units)
    tables += wp16_o5.build_drift(source)
    tables += wp16_o6_o8.build(source, runs, units)
    tables += wp16_o7.build(source, units)
    more, drawn = wp16_o9.build(source, units, absent)
    tables, figures = tables + more, figures + drawn
    more, drawn = wp16_o10.build(source, runs, units, absent)
    return tables + more, figures + drawn


def made_apart(tables: list[Table]) -> dict:
    """The rows whose figures were made after the run's output was seen, by code of a later commit
    (`wp16_mixed.MARK` in their first cell), for the manifest: the sentence and each table and
    row."""
    return {
        "sentence": MARK,
        "rows": [
            {"table": table.name, "row": row[0]}
            for table in tables
            for row in table.rows
            if isinstance(row[0], str) and MARK in row[0]
        ],
    }


def any_unreadable(tables: list[Table]) -> bool:
    return any(
        isinstance(cell, Missing) and cell.text.startswith(BAD_START)
        for table in tables
        for row in table.rows
        for cell in row
    )


def _write(folder: Path, name: str, text: str, written: dict) -> None:
    data = text.encode("utf-8")
    (folder / name).write_bytes(data)
    written[name] = hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=TOP)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dev-days-run", type=Path, default=None)
    parser.add_argument("--dial-choice-run", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or (
        Path(os.environ[RUN_DIR_VARIABLE]) if os.environ.get(RUN_DIR_VARIABLE) else None
    )
    if out is None:
        print(f"COULD NOT RUN: give --out or set {RUN_DIR_VARIABLE}")
        return 2
    try:
        commit = head_commit(args.root)
        code_commit = code_head()
        dirty = code_dirty()
        drive = find_drive(args.root)
        source = Source(args.root, drive)
        tables, figures = collect(source)
        development, development_runs = wp16_devdays.build(
            source,
            {"dev_days_run": args.dev_days_run, "dial_choice_run": args.dial_choice_run},
        )
        tables += development
    except (NoDrive, Missed) as problem:
        print(f"COULD NOT RUN: {getattr(problem, 'text', problem)}")
        return 2
    failed, unreadable = failed_configurations(source)
    first = banner(drive, failed, unreadable)
    for table in tables:
        table.notes.insert(0, first)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    try:
        for table in tables:
            _write(out, f"{table.name}.csv", wp16_cells.csv_text(table), written)
            _write(out, f"{table.name}.md", wp16_cells.markdown_text(table), written)
        _write(
            out,
            OUTCOMES_NAME,
            "# Outcome tables\n\n" + "\n".join(wp16_cells.markdown_text(t) for t in tables),
            written,
        )
        for spec in figures:
            name = wp16_figures.render(spec, out)
            written[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
        manifest = {
            "commit": commit,
            "code_commit": code_commit,
            "code_dirty": dirty,
            "command": ["python", "-m", "fleetreserve.wp16_run", *(argv or sys.argv[1:])],
            "drive": {
                "file": drive.name,
                "sha256": drive.log_sha256,
                "commit": drive.commit,
                "complete": drive.complete,
                "failed_configurations": failed,
                "unreadable_learn_results": unreadable,
                "steps_not_passed": drive.problems(),
            },
            "made_apart": made_apart(tables),
            "summaries_read": dict(sorted(source.hashes.items())),
            "outputs": dict(sorted(written.items())),
        }
        if development_runs:  # only when an option was given: the manifest is otherwise unchanged
            manifest["development_runs"] = development_runs
        (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    except ImportError as problem:
        print(f"COULD NOT RUN: the figures need the dependency group figures: {problem}")
        return 2
    except OSError as problem:
        print(f"COULD NOT RUN: a file cannot be written: {problem}")
        return 2
    except Exception as problem:  # a drawing error: the tables are written, no manifest is
        print(f"COULD NOT RUN: a figure could not be drawn: {type(problem).__name__}: {problem}")
        return 2
    print(first)
    if code_commit != commit:
        print(f"the code that ran is at commit {code_commit}, --root is at commit {commit}")
    if dirty is True:
        print("the checkout the code ran from has uncommitted changes (code_dirty: true)")
    print(f"wrote {len(written)} files and {MANIFEST_NAME} into {out.name}")
    complete = drive.complete and not failed and not unreadable
    return 0 if complete and not any_unreadable(tables) else 1


if __name__ == "__main__":
    sys.exit(main())
