"""One item of the WP3 pass mark, read from a recorded re-run (plan section 13, WP3).

Run as `python tools/earlier_rerun_check.py <item> [--run <run folder>]` with exactly one item:
  --kwh         the perfect-knowledge offer is within 0.1 kWh of the published 45.6 kWh;
  --learned     the learned model's share of it is within 0.1 percentage point of 0.4%;
  --population  the population model's share of it is within 0.1 percentage point of 3.9%.

The run is the folder given, or else the newest folder in `runs.noindex/` named
`<UTC start>-wp3-rerun`. Its manifest must verify (`fleetreserve.run_verify`) for the results file
`earlier-rerun.json` that `tools/earlier_rerun.py` writes: an unfinished, failed, interrupted or
changed run never verifies. The numbers are read from the verified results file. How long the run
took is never judged (plan section 13, WP3): a run that finished, verified, has its numbers decided
however long it took.

Tolerance. Each limit is inclusive: a difference of exactly 0.1 passes. Floating-point is handled
by comparing decimal numbers built from the shortest text that identifies each value (Python's
repr), so 45.7 against 45.6 is a difference of exactly 0.1 and passes whatever binary rounding
would say. A share is turned into percentage points by moving the decimal point (times 100,
exactly). A value beyond the limit by even one unit in the last place fails.

A run stopped by the run recorder's stop limit (`--max-seconds`, a guard against a hanging run,
not a pass mark) did not finish and produced no numbers, so every item could not run (exit 2).

Exit status: 0 the item passed; 1 the value is outside its limit; 2 it could not run (no run
folder, not verified, a file or value missing, a run stopped by the recorder's stop limit).
"""

import argparse
import json
import math
import re
import sys
from decimal import Decimal
from pathlib import Path

import earlier_rerun

from fleetreserve.run_record import MANIFEST_NAME, RUNS_FOLDER
from fleetreserve.run_verify import NotVerified, require_verified

LABEL = "wp3-rerun"
_RUN_NAME = re.compile(r"\d{8}T\d{6}Z-" + re.escape(LABEL))

# item -> (name in the results, unit shown, factor to that unit, limit in that unit)
ITEMS = {
    "kwh": ("perfect_knowledge_kwh", "kWh", 1, earlier_rerun.TOLERANCE_KWH),
    "learned": ("learned_share", "%", 100, earlier_rerun.TOLERANCE_POINTS),
    "population": ("population_share", "%", 100, earlier_rerun.TOLERANCE_POINTS),
}


class CannotCheck(Exception):
    """The item cannot be decided (exit status 2)."""


def exact(value: float) -> Decimal:
    """The decimal number whose text is the shortest one that identifies the float."""
    return Decimal(repr(float(value)))


def show(number: Decimal) -> str:
    """A decimal number written plainly, without trailing zeros."""
    return format(number.normalize(), "f")


def difference(measured: Decimal, published: Decimal) -> Decimal:
    return abs(measured - published)


def stopped_wording(run: Path, manifest: dict) -> str:
    limit = manifest.get("max_seconds")
    shown = f"{limit:g} s" if isinstance(limit, int | float) else "unknown"
    return (
        f"{run.name} was stopped by the run recorder at its stop limit of {shown} (a guard "
        "against a hanging run, not a pass mark), so it did not finish and produced no numbers"
    )


def newest_run(top: Path) -> Path:
    runs = top / RUNS_FOLDER
    folders = (
        sorted(path for path in runs.iterdir() if path.is_dir() and _RUN_NAME.fullmatch(path.name))
        if runs.is_dir()
        else []
    )
    if not folders:
        raise CannotCheck(f"no run named <UTC start>-{LABEL} in {RUNS_FOLDER}/")
    return folders[-1]


def read_manifest(run: Path) -> dict:
    try:
        manifest = json.loads((run / MANIFEST_NAME).read_text())
    except (OSError, ValueError) as problem:
        raise CannotCheck(f"{MANIFEST_NAME} of {run.name} cannot be read: {problem}") from problem
    if not isinstance(manifest, dict):
        raise CannotCheck(f"{MANIFEST_NAME} of {run.name} is not a JSON object")
    return manifest


def verified_results(run: Path) -> dict:
    results_file = run / earlier_rerun.RESULTS_NAME
    try:
        require_verified(results_file)
    except NotVerified as problem:
        raise CannotCheck(f"{run.name} is not verified: {problem}") from problem
    try:
        return json.loads(results_file.read_text())
    except (OSError, ValueError) as problem:
        raise CannotCheck(f"{earlier_rerun.RESULTS_NAME} cannot be read: {problem}") from problem


def check_number(item: str, run: Path, manifest: dict) -> int:
    if manifest.get("timed_out", False):
        raise CannotCheck(stopped_wording(run, manifest))
    name, unit, factor, tolerance = ITEMS[item]
    results = verified_results(run)
    try:
        value = results["measured"][name]
    except (KeyError, TypeError) as problem:
        raise CannotCheck(f"{earlier_rerun.RESULTS_NAME} has no measured {name}") from problem
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise CannotCheck(f"the measured {name} is not a finite number: {value!r}")
    published = exact(earlier_rerun.PUBLISHED[name]) * factor
    measured = exact(value) * factor
    gap = difference(measured, published)
    limit = exact(tolerance)
    text = (
        f"measured {show(measured)} {unit}, published {show(published)} {unit}, "
        f"difference {show(gap)} (limit {show(limit)}, inclusive); run {run.name}"
    )
    if gap <= limit:
        print(f"within the limit: {text}")
        return 0
    print(f"FAILED: outside the limit: {text}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/earlier_rerun_check.py",
        description="Decide one item of the WP3 pass mark from the recorded re-run.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    for item in ITEMS:
        group.add_argument(f"--{item}", dest="item", action="store_const", const=item)
    parser.add_argument("--run", type=Path, help="the run folder (default: the newest re-run)")
    parser.add_argument(
        "--top",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the repository top folder, where runs.noindex/ is (default: this repository)",
    )
    options = parser.parse_args(argv)
    try:
        run = options.run if options.run is not None else newest_run(options.top)
        if not run.is_dir():
            raise CannotCheck(f"the run folder {run.name} does not exist")
        return check_number(options.item, run, read_manifest(run))
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
