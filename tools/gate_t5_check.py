"""The WP6 pass-mark item, read from the recorded gate run (plan section 13, WP6: "T5 on 100% of
day-paths").

Run as `python tools/gate_t5_check.py [--run <run folder>] [--top <folder>]`.

The run is the folder given, or else the newest folder in `runs.noindex/` named
`<UTC start>-wp6-gate` (`fleetreserve.run_evidence`; an older run is never used in its place).
`run_evidence` reads its results file `gate.json`, written by `fleetreserve.gate_run`: an
unfinished, timed-out, interrupted or changed run, or a stale one (recorded at another commit than
the one checked out now and not read from an earlier commit by the rule of `run_evidence`, module
text there), cannot decide the item. Every line this tool writes that names a run read from an
earlier commit carries `Evidence.note` after the run's name, whatever the line then says. A run
that ended with exit 1 or 2 (a configuration failed, or the run could not run) cannot decide it
either: only a run that ended with exit 0 is read. The breaks file `gate-t5-breaks.csv` must
verify too.

The run must be of the committed matrix: the SHA-256 that `gate.json` records for the matrix file
it read must equal that of `gate/matrix.json` at the checked-out commit. A run on another matrix (a
smaller one, or one with another T5 tolerance) cannot decide the item.

The item passes when the run is a gate run (neither a trial with `--limit-days` nor a partial run
with `--only`), checked at least one day-path and found no break of T5 (a) or (b). Breaks are
counted from the counts in `gate.json` and from the rows of the breaks file, never from a verdict
word: any break in either is a failure. A trial, a partial run or a run that checked no day-path
cannot decide the item.

Exit status: 0 passed; 1 T5 broke on some day-path; 2 could not run (no run folder, not verified,
stale, another matrix, a value missing, a trial or partial run, no day-path checked).
"""

import argparse
import csv
import hashlib
import io
import subprocess
import sys
from pathlib import Path

from fleetreserve.gate_run import BREAK_COLUMNS, BREAKS_NAME, RESULTS_NAME
from fleetreserve.run_evidence import CannotCheck, head_commit, newest_run, read_evidence
from fleetreserve.run_verify import NotVerified, require_verified

LABEL = "wp6-gate"
MATRIX = "gate/matrix.json"


def committed_matrix_sha256(top: Path) -> str:
    """SHA-256 of `gate/matrix.json` as committed at the checked-out commit."""
    done = subprocess.run(
        ["git", "-C", str(top), "show", f"HEAD:{MATRIX}"], capture_output=True, check=False
    )
    if done.returncode != 0:
        raise CannotCheck(f"{MATRIX} is not committed at the checked-out commit")
    return hashlib.sha256(done.stdout).hexdigest()


def check_matrix(results: dict, committed: str, name: str) -> None:
    matrix = results.get("matrix")
    used = matrix.get("sha256") if isinstance(matrix, dict) else None
    if not isinstance(used, str):
        raise CannotCheck(f"{name} does not record the SHA-256 of its matrix, so {MATRIX} differs")
    if used != committed:
        raise CannotCheck(
            f"{name} ran on another matrix: its matrix has SHA-256 {used[:12]}, the committed "
            f"{MATRIX} has {committed[:12]}"
        )


def listed_breaks(run: Path, name: str) -> int:
    """The number of rows in the run's verified breaks file; `name` is how a line names the run."""
    file = run / BREAKS_NAME
    try:
        require_verified(file)
    except NotVerified as problem:
        raise CannotCheck(f"{BREAKS_NAME} of {name} is not verified: {problem}") from problem
    lines = [line for line in file.read_text().splitlines() if not line.startswith("#")]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    if not rows or tuple(rows[0]) != BREAK_COLUMNS:
        raise CannotCheck(f"{BREAKS_NAME} of {name} has no header row")
    return len(rows) - 1


def _count(t5: dict, name: str) -> int:
    value = t5.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CannotCheck(f"{RESULTS_NAME} has no count {name!r}")
    return value


def check(results: dict, name: str, committed: str, head: str, listed: int) -> int:
    if results.get("trial") is not False:
        raise CannotCheck(f"{name} is a trial run (--limit-days), not a gate result")
    if results.get("partial") is not False:
        raise CannotCheck(f"{name} is a partial run (--only), not a gate result")
    check_matrix(results, committed, name)
    t5 = results.get("t5")
    if not isinstance(t5, dict):
        raise CannotCheck(f"{RESULTS_NAME} has no T5 counts")
    checked = _count(t5, "day_paths_checked")
    broken_a, broken_b = _count(t5, "breaks_a"), _count(t5, "breaks_b")
    if checked == 0:
        raise CannotCheck(f"{name} checked no day-path")
    text = (
        f"{checked} day-paths checked, {broken_a} breaks of T5 (a), {broken_b} of T5 (b), "
        f"{listed} listed in {BREAKS_NAME}"
    )
    where = (
        f"{name}, on the committed {MATRIX} (sha256 {committed[:12]}), checked at commit "
        f"{head[:12]}"
    )
    if broken_a or broken_b or listed:
        print(f"FAILED: {text} ({where})")
        return 1
    print(f"T5 held on every day-path: {text} ({where})")
    return 0


def check_run(run: Path, top: Path) -> int:
    head = head_commit(top)
    evidence = read_evidence(run, RESULTS_NAME, head)
    name = f"{run.name}{evidence.note}"
    if not evidence.verified:
        raise CannotCheck(
            f"{name} ended with exit code {evidence.manifest.get('exit_code')}, so it is not "
            "verified and cannot decide the item"
        )
    committed = committed_matrix_sha256(top)
    return check(evidence.data, name, committed, head, listed_breaks(run, name))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/gate_t5_check.py",
        description="Decide the WP6 pass-mark item from the recorded gate run.",
    )
    parser.add_argument("--run", type=Path, help="the run folder (default: the newest gate run)")
    parser.add_argument(
        "--top",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the repository top folder, where runs.noindex/ is (default: this repository)",
    )
    options = parser.parse_args(argv)
    try:
        run = options.run if options.run is not None else newest_run(options.top, LABEL)
        return check_run(run, options.top)
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
