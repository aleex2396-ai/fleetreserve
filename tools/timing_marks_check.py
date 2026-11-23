"""One timing item of the WP4 pass mark, read from a recorded timing run (plan section 13, WP4).

Run as `python tools/timing_marks_check.py <item> [--run <run folder>] [--top <folder>]` with
exactly one item, one per part of the plan's words "solve times of the box LP and the tree (100
cars, 8 hours) and of LP (24) (50 cars, 14 hours) logged with the machine they ran on":
  --box   the box LP's solve times (100 cars, 8 hours);
  --tree  the tree's solve times (100 cars, 8 hours);
  --lp24  LP (24)'s solve times (50 cars, 14 hours).

The plan adds "Run time is reported, never judged": no time is compared with any limit, and how
busy the machine was decides nothing. An item passes when the run is verified, at the plan's
sizes, and records every solve time of the item's programmes as a number together with the
machine it ran on (`machine`: platform, processor, processor count and Python version, as
`fleetreserve.timing_marks` writes it). The times, the machine and its load averages at the
start and the end are printed.

The run is the folder given, or else the newest folder `runs.noindex/<UTC start>-wp4-timing`
(`fleetreserve.run_evidence`; an older run is never used in its place), made by
`python -m fleetreserve.timing_marks`. Its `timing-marks.json` must be verified and recorded at
the commit checked out now or read from an earlier commit by the rule of `run_evidence` (module
text there; else it is stale). Every line this tool writes that names a run read from an earlier
commit carries `Evidence.note` after the run's name, whatever the line then says. A run that did
not end with exit 0 (some solve did not end optimal) or was stopped is not verified, so its items
could not run.

Readings, marked (the plan's wording does not decide them):

1. "Box LP" is read as the programme that produces the offered box: the causal certificate of the
   hourly product with the shape rule `largest-smallest-width`, both stages together
   (`hourly-certificate`). The first stage alone and the block product's certificate are printed
   beside it. The general split (plan section 1), which is a slower programme, is not the box
   programme and is not timed here.
2. "The tree" is the corner tree of both products (`hourly-tree`, `block-tree`), on the box
   LP's day (100 cars, 8 service hours).
3. "LP (24) ... 14 hours" is read as all 14 hours written out as programme hours (`lp24-steered`,
   the stricter and literal reading); the other reading of `lp24_timing` (hour 0, 8 service
   hours, 5 recovery hours eliminated) is printed beside it.
4. Sizes: the run must be at the plan's sizes (100 cars and 8 service hours; 50 cars and 14
   hours) with 3 solves per programme; any other run could not run for the item.

Exit status: 0 the times are logged with the machine; 2 it could not run (no run, not verified,
stale, wrong sizes, a time or the machine description missing or malformed). Nothing here fails
with exit 1, since no time is judged.
"""

import argparse
import math
import sys
from pathlib import Path

from fleetreserve.run_evidence import CannotCheck, head_commit, newest_run, read_evidence
from fleetreserve.timing_marks import RESULTS_NAME

LABEL = "wp4-timing"
REPEATS = 3
BOX_SIZE = {"cars": 100, "service_hours": 8}
LP24_SIZE = {"cars": 50, "hours": 14}
MACHINE_TEXT_FIELDS = ("platform", "processor", "python")

# item -> (the plan's words, programmes logged for the item, programmes printed beside)
ITEMS = {
    "box": (
        "solve times of the box LP (100 cars, 8 hours)",
        ("hourly-certificate",),
        ("hourly-certificate-first-stage", "block-certificate"),
    ),
    "tree": ("solve times of the tree (100 cars, 8 hours)", ("hourly-tree", "block-tree"), ()),
    "lp24": (
        "solve times of LP (24) (50 cars, 14 hours)",
        ("lp24-steered",),
        ("lp24-eliminated",),
    ),
}


def _number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def machine(data: dict) -> dict:
    """The recorded machine description; raise CannotCheck when it is missing or malformed."""
    found = data.get("machine")
    if not isinstance(found, dict):
        raise CannotCheck("the machine it ran on is not recorded (no machine description)")
    for field in MACHINE_TEXT_FIELDS:
        if not isinstance(found.get(field), str) or not found[field].strip():
            raise CannotCheck(f"the machine description has no {field}: {found.get(field)!r}")
    count = found.get("cpu_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise CannotCheck(f"the machine description has no processor count: {count!r}")
    return found


def load_text(data: dict) -> str:
    """The one-minute load averages at the start and the end, as recorded; reported only."""
    parts = []
    for when in ("start", "end"):
        try:
            load = data[f"machine_at_{when}"]["load_average_1_5_15_min"][0]
        except (KeyError, TypeError, IndexError):
            load = None
        parts.append(
            f"{load:.2f} at the {when}" if _number(load) else f"not recorded at the {when}"
        )
    return "one-minute load average " + ", ".join(parts)


def check_sizes(data: dict, item: str) -> None:
    if data.get("repeats") != REPEATS:
        raise CannotCheck(f"the run has {data.get('repeats')!r} solves per programme, not 3")
    key, wanted = ("lp24_fleet", LP24_SIZE) if item == "lp24" else ("box_day", BOX_SIZE)
    found = data.get(key)
    if not isinstance(found, dict) or any(found.get(k) != v for k, v in wanted.items()):
        raise CannotCheck(f"the run's {key} is {found!r}, not at the plan's sizes {wanted}")


def programme(data: dict, name: str) -> dict:
    try:
        found = data["programmes"][name]
        solves = found["solves"]
    except (KeyError, TypeError) as problem:
        raise CannotCheck(f"the run has no programme {name}") from problem
    if not isinstance(solves, list) or len(solves) != REPEATS:
        raise CannotCheck(f"{name} does not list {REPEATS} solves")
    for solve in solves:
        if not isinstance(solve, dict) or not _number(solve.get("seconds")):
            raise CannotCheck(f"{name} has a solve without a time")
    return found


def describe(name: str, found: dict) -> str:
    times = ", ".join(f"{s['seconds']:.3f} s ({s.get('outcome')})" for s in found["solves"])
    return f"{name}: {times}"


def check_item(item: str, run: Path, top: Path) -> int:
    words, logged, beside = ITEMS[item]
    evidence = read_evidence(run, RESULTS_NAME, head_commit(top))
    if not evidence.verified:
        raise CannotCheck(
            f"{run.name}{evidence.note} ended with exit code {evidence.manifest.get('exit_code')}, "
            "so it is not verified and cannot pass an item"
        )
    data = evidence.data
    check_sizes(data, item)
    described = machine(data)
    found = {name: programme(data, name) for name in (*logged, *beside)}
    print(f"item: {words}, logged with the machine they ran on (plan section 13, WP4)")
    print("run time is reported, never judged (plan section 13, WP4)")
    if item == "box":
        print(
            "reading: the box LP is the certificate that produces the offered box (hourly "
            "product, causal gains, shape rule largest-smallest-width, both stages); the general "
            "split, which is slower, is not the box programme"
        )
    print(
        f"machine: {described['processor']}, {described['cpu_count']} processors, "
        f"{described['platform']}, Python {described['python']}; {load_text(data)}"
    )
    for name in logged:
        print(f"  logged: {describe(name, found[name])}")
    for name in beside:
        print(f"  beside it: {describe(name, found[name])}")
    print(
        f"logged: every solve time of {', '.join(logged)}, with the machine, in {run.name}"
        f"{evidence.note}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/timing_marks_check.py",
        description="Check one timing item of the WP4 pass mark: solve times logged with the "
        "machine, from the recorded timing run.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    for item in ITEMS:
        group.add_argument(f"--{item}", dest="item", action="store_const", const=item)
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
        return check_item(options.item, run, options.top)
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
