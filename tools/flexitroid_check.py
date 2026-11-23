"""The flexitroid item of the WP4 pass mark: "`flexitroid` matches (b) on 50 development days"
(plan section 4, T1; section 13, WP4).

Run as `python tools/flexitroid_check.py [--run <run folder>] [--top <folder>]`.

The run is the folder given, or else the newest folder `runs.noindex/<UTC start>-wp4-flexitroid`
(`fleetreserve.run_evidence`; an older run is never used in its place), made by
`python -m fleetreserve.flexitroid_days`. Its `flexitroid-match.json` must be recorded at the
commit checked out now or read from an earlier commit by the rule of `run_evidence` (module text
there; else it is stale). Every line this tool writes that names a run read from an earlier
commit carries `Evidence.note` after the run's name, whatever the line then says. What a day's
comparison is, and the strict reading decided on, are in `fleetreserve.flexitroid_match`.

The item passes only when the run is verified (it ended with exit 0) and lists exactly 50
distinct development days (the units of `ceiling_report.UNITS`, sample size 50), each with the
verdict "match", and the counts in the file agree with the days listed. A day that differs fails
the item, also when read from an intact run that ended with exit 1. A day that could not run
(flexitroid raised an error, a programme timed out or did not end optimal) is never a match.

The days must be of every one of those units. The block sample gives each unit it is drawn on at
least one day (`block_sample.allocate`), so a run of `flexitroid_days --units` on a part of them
lists days of that part only; such a run could not run for the item, whatever its days found, as
does one with a day of any other unit.

Exit status: 0 all 50 days match; 1 some day differs; 2 could not run (no run, a run not
verified, a stale run, fewer than 50 days, a day that could not run, a day of another unit, days
not of every unit, a file that does not add up).
"""

import argparse
import sys
from pathlib import Path

from fleetreserve.block_sample import SAMPLE_SIZE
from fleetreserve.ceiling_report import UNITS
from fleetreserve.flexitroid_days import RESULTS_NAME, VERDICTS, counts
from fleetreserve.flexitroid_match import COULD_NOT_RUN, DIFFERS, MATCH
from fleetreserve.run_evidence import CannotCheck, head_commit, newest_run, read_evidence

LABEL = "wp4-flexitroid"
ITEM = "flexitroid matches T1(b) on 50 development days (plan section 4)"


def _days(data: dict) -> list[dict]:
    days = data.get("days")
    if not isinstance(days, list) or not all(isinstance(d, dict) for d in days):
        raise CannotCheck(f"{RESULTS_NAME} lists no days")
    keys = set()
    for day in days:
        if day.get("verdict") not in VERDICTS or day.get("functions_only_verdict") not in VERDICTS:
            raise CannotCheck(f"a day has no known verdict: {day.get('verdict')!r}")
        if day.get("unit") not in UNITS:
            raise CannotCheck(f"{day.get('unit')!r} is not a development unit")
        key = (day.get("unit"), day.get("site"), day.get("date"))
        if key in keys:
            raise CannotCheck(f"the day {' '.join(map(str, key))} is listed twice")
        keys.add(key)
    found = {day["unit"] for day in days}
    if found != set(UNITS):
        raise CannotCheck(
            f"the run's days are of the units {', '.join(sorted(found)) or 'none'}, not of every "
            f"unit of this pass mark ({', '.join(UNITS)}: ceiling_report.UNITS)"
        )
    if data.get("counts") != counts(days):
        raise CannotCheck(f"the counts in {RESULTS_NAME} do not agree with the days listed")
    return days


def check(run: Path, top: Path) -> int:
    evidence = read_evidence(run, RESULTS_NAME, head_commit(top))
    data = evidence.data
    days = _days(data)
    print(f"item: {ITEM}; run {run.name}{evidence.note}")
    differing = [d for d in days if d["verdict"] == DIFFERS]
    if differing:
        for day in differing:
            for problem in day.get("problems", []):
                print(f"FAILED: {day['unit']} {day['site']} {day['date']}: {problem}")
        print(f"FAILED: {len(differing)} of {len(days)} days differ")
        return 1
    if data.get("sample", {}).get("size") != SAMPLE_SIZE or len(days) != SAMPLE_SIZE:
        raise CannotCheck(f"the run covers {len(days)} days, not the {SAMPLE_SIZE} of the plan")
    missing = [d for d in days if d["verdict"] == COULD_NOT_RUN]
    if missing:
        first = missing[0]
        raise CannotCheck(
            f"{len(missing)} of {len(days)} days could not run, first {first['unit']} "
            f"{first['site']} {first['date']}: {'; '.join(first.get('could_not_run', []))}"
        )
    if not evidence.verified:
        raise CannotCheck(
            f"{run.name}{evidence.note} ended with exit code {evidence.manifest.get('exit_code')}, "
            "so it is not verified and cannot pass the item"
        )
    if any(d["verdict"] != MATCH for d in days):  # unreachable while VERDICTS has three entries
        raise CannotCheck("a day has a verdict other than match")
    source = data.get("flexitroid", {})
    print(
        f"all {len(days)} development days match (flexitroid {source.get('version')} at commit "
        f"{source.get('commit')})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/flexitroid_check.py",
        description="Decide the flexitroid item of the WP4 pass mark from the recorded run.",
    )
    parser.add_argument("--run", type=Path, help="the run folder (default: the newest run)")
    parser.add_argument(
        "--top",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the repository top folder, where runs.noindex/ is (default: this repository)",
    )
    options = parser.parse_args(argv)
    try:
        run = options.run if options.run is not None else newest_run(options.top, LABEL)
        return check(run, options.top)
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
