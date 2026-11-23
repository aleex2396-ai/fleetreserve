"""Time the programmes of the WP4 timing items on seeded made-up fleets (plan section 13, WP4:
"solve times of the box LP and the tree (100 cars, 8 hours) and of LP (24) (50 cars, 14 hours)
logged with the machine they ran on. Run time is reported, never judged").

Command line, from the project's top folder:
    python -m fleetreserve.timing_marks [--cars 100] [--service-hours 8] [--lp24-cars 50]
        [--lp24-hours 14] [--repeats 3] [--seed 0]

Made-up fleets (no data is read). Cars come from `lp24_timing.made_up_cars`: a departure hour
drawn from 1 to the fleet's hours, a rate from `lp24_timing.RATES`, a room drawn as a share of what
the rate allows before the car leaves, at most `lp24_timing.MAX_ROOM` kWh.

- Box and tree day: `--cars` cars drawn for `1 + --service-hours + RECOVERY_HOURS` hours, on a
  day of `--service-hours` service hours and `RECOVERY_HOURS` recovery hours (with 8 service
  hours the day spans 14 hours, as LP (24)'s fleet does), true one-way sets, rho = `BOX_RHO` (the
  plan's main value), recovery eliminated (`carsets.true_rows`).
- LP (24) fleet: `--lp24-cars` cars drawn for `--lp24-hours` hours, rho = `LP24_RHO` (the default
  of `lp24_timing`, which says why rho = 0 gives alpha = 0), under both readings of "14 hours"
  that `lp24_timing` offers: `steered` (every hour written out as a programme hour) and
  `eliminated` (hour 0, 8 service hours, the rest recovery hours, eliminated).

Programmes timed, each `--repeats` times, in this order, with the time limits of `lp.py`:

- `hourly-certificate`: the causal certificate of the hourly product with the shape rule
  `largest-smallest-width` (`certificate.LargestSmallestWidth`, tolerance `SHAPE_TOLERANCE`), both
  stages together: the programme the hourly offer uses;
- `hourly-certificate-first-stage`: its first stage alone (`certificate.ANY_OPTIMAL`);
- `block-certificate`: the causal certificate of the block product (`ANY_OPTIMAL`: the block value
  does not depend on the centre chosen among those of equal value);
- `hourly-tree`, `block-tree`: the corner tree of each product (`tree.corner_tree`);
- `lp24-steered`, `lp24-eliminated`: LP (24) with the day-average base set
  (`lp24_timing.time_reading`).

Each solve's `seconds` is the wall-clock time of the whole call (building the programme and every
stage's solve); `solver_seconds` is the solver's own time, listed beside. `best_seconds` is the
smallest `seconds` of a programme when all its solves ended optimal, else None.

The machine is described once (`machine`: platform, processor, processor count, Python version),
and its load averages (1, 5 and 15 minutes, `os.getloadavg`) and processor count are recorded at
the start and at the end, so a reader can see how busy it was. The command compares nothing with
any limit; `tools/timing_marks_check.py` checks that the times and the machine are logged.

Output: printed lines and, when `FLEETRESERVE_RUN_DIR` is set, `timing-marks.json` there.

Exit status: 0 every solve ended optimal; 1 a solve ended otherwise (a timeout included); 2 the
command could not run (a bad argument, the results file not writable).
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from fleetreserve import lp24_timing, products
from fleetreserve.carsets import true_rows
from fleetreserve.certificate import ANY_OPTIMAL, LargestSmallestWidth, certify
from fleetreserve.day import MAX_SERVICE_HOURS, Day
from fleetreserve.tree import corner_tree

RESULTS_NAME = "timing-marks.json"
RECOVERY_HOURS = 5
BOX_RHO = 0.0
LP24_RHO = 0.1
SHAPE_TOLERANCE = 1e-6
SHAPE_RULE = "largest-smallest-width"
PROGRAMMES = (
    "hourly-certificate",
    "hourly-certificate-first-stage",
    "block-certificate",
    "hourly-tree",
    "block-tree",
    "lp24-steered",
    "lp24-eliminated",
)


def box_day(n_cars: int, n_service: int, seed: int) -> Day:
    """The made-up day of the box and tree marks (see the module text)."""
    hours = 1 + n_service + RECOVERY_HOURS
    cars = tuple(lp24_timing.made_up_cars(n_cars, hours, seed))
    return Day("made-up", datetime.date(2020, 1, 1), n_service, RECOVERY_HOURS, BOX_RHO, cars)


def _sizes(sizes) -> dict:
    return {"variables": sizes.variables, "rows": sizes.rows, "nonzeros": sizes.nonzeros}


def _timed(call, repeats: int) -> dict:
    """Run `call` (returning an object with `outcome`, `seconds` and `sizes`) `repeats` times."""
    solves = []
    sizes = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        took = time.perf_counter() - start
        sizes = result.sizes
        solves.append(
            {"outcome": result.outcome, "seconds": took, "solver_seconds": float(result.seconds)}
        )
    optimal = all(s["outcome"] == "optimal" for s in solves)
    return {
        "solves": solves,
        "best_seconds": min(s["seconds"] for s in solves) if optimal else None,
        "sizes": _sizes(sizes),
    }


def _lp24(rows, repeats: int) -> dict:
    found = lp24_timing.time_reading(rows, repeats)
    solves = [
        {
            "outcome": s["outcome"],
            "seconds": s["seconds_with_build"],
            "solver_seconds": s["solve_seconds"],
        }
        for s in found["solves"]
    ]
    optimal = all(s["outcome"] == "optimal" for s in solves)
    return {
        "solves": solves,
        "best_seconds": min(s["seconds"] for s in solves) if optimal else None,
        "sizes": {k: found[k] for k in ("variables", "rows", "nonzeros")},
        "programme_hours": found["programme_hours"],
    }


def machine_state() -> dict:
    one, five, fifteen = os.getloadavg()
    return {"load_average_1_5_15_min": [one, five, fifteen], "cpu_count": os.cpu_count()}


def _processor() -> str:
    """The processor's name: on macOS the brand string, else what `platform` reports."""
    if sys.platform == "darwin":
        try:
            done = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            done = None
        if done is not None and done.returncode == 0 and done.stdout.strip():
            return done.stdout.strip()
    return platform.processor() or platform.machine() or "unknown"


def machine_description() -> dict:
    """The machine the programmes ran on, as the plan asks the solve times to be logged with."""
    return {
        "platform": platform.platform(),
        "processor": _processor(),
        "cpu_count": os.cpu_count() or 0,
        "python": platform.python_version(),
    }


def time_all(args) -> dict:
    day = box_day(args.cars, args.service_hours, args.seed)
    rows = true_rows(day)
    hourly, block = products.hourly(), products.block()
    rule = LargestSmallestWidth(SHAPE_TOLERANCE)
    lp24_cars = lp24_timing.made_up_cars(args.lp24_cars, args.lp24_hours, args.seed)
    calls = {
        "hourly-certificate": lambda: _timed(
            lambda: certify(rows, hourly, "causal", shape_rule=rule), args.repeats
        ),
        "hourly-certificate-first-stage": lambda: _timed(
            lambda: certify(rows, hourly, "causal", shape_rule=ANY_OPTIMAL), args.repeats
        ),
        "block-certificate": lambda: _timed(
            lambda: certify(rows, block, "causal", shape_rule=ANY_OPTIMAL), args.repeats
        ),
        "hourly-tree": lambda: _timed(lambda: corner_tree(rows, hourly), args.repeats),
        "block-tree": lambda: _timed(lambda: corner_tree(rows, block), args.repeats),
        "lp24-steered": lambda: _lp24(
            lp24_timing.steered_rows(lp24_cars, args.lp24_hours, LP24_RHO), args.repeats
        ),
        "lp24-eliminated": lambda: _lp24(
            lp24_timing.eliminated_rows(lp24_cars, args.lp24_hours, LP24_RHO), args.repeats
        ),
    }
    return {name: calls[name]() for name in PROGRAMMES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cars", type=int, default=100)
    parser.add_argument("--service-hours", type=int, default=MAX_SERVICE_HOURS)
    parser.add_argument("--lp24-cars", type=int, default=50)
    parser.add_argument("--lp24-hours", type=int, default=14)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if (
        min(args.cars, args.lp24_cars, args.repeats) < 1
        or not 1 <= args.service_hours <= MAX_SERVICE_HOURS
        or args.lp24_hours < 1 + MAX_SERVICE_HOURS
    ):
        print(
            "COULD NOT RUN: --cars, --lp24-cars and --repeats must be at least 1, "
            f"--service-hours 1 to {MAX_SERVICE_HOURS}, --lp24-hours at least "
            f"{1 + MAX_SERVICE_HOURS}"
        )
        return 2
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir and not Path(run_dir).is_dir():
        print(f"COULD NOT RUN: the run folder {run_dir} does not exist")
        return 2
    machine = machine_description()
    start = machine_state()
    programmes = time_all(args)
    end = machine_state()
    document = {
        "machine": machine,
        "machine_at_start": start,
        "machine_at_end": end,
        "box_day": {
            "generator": "lp24_timing.made_up_cars",
            "cars": args.cars,
            "service_hours": args.service_hours,
            "recovery_hours": RECOVERY_HOURS,
            "rho": BOX_RHO,
            "seed": args.seed,
            "sets": "true one-way, recovery eliminated",
            "shape_rule": SHAPE_RULE,
            "shape_tolerance": SHAPE_TOLERANCE,
        },
        "lp24_fleet": {
            "generator": "lp24_timing.made_up_cars",
            "cars": args.lp24_cars,
            "hours": args.lp24_hours,
            "rho": LP24_RHO,
            "seed": args.seed,
            "base_set": "day average",
        },
        "repeats": args.repeats,
        "seconds_measured": "wall clock of the whole call: build and every stage's solve",
        "programmes": programmes,
    }
    print(
        f"machine: {machine['processor']}, {machine['cpu_count']} processors, "
        f"{machine['platform']}, Python {machine['python']}"
    )
    print(
        f"load average (1, 5, 15 min) at start {start['load_average_1_5_15_min']}, at end "
        f"{end['load_average_1_5_15_min']}; {start['cpu_count']} processors"
    )
    for name, found in programmes.items():
        sizes = found["sizes"]
        times = ", ".join(f"{s['seconds']:.3f} ({s['outcome']})" for s in found["solves"])
        best = "-" if found["best_seconds"] is None else f"{found['best_seconds']:.3f}"
        print(
            f"{name}: {sizes['variables']} variables, {sizes['rows']} rows, "
            f"{sizes['nonzeros']} nonzeros; seconds {times}; best {best}"
        )
    if run_dir:
        try:
            (Path(run_dir) / RESULTS_NAME).write_text(json.dumps(document, indent=1) + "\n")
        except OSError as problem:
            print(f"COULD NOT RUN: cannot write the results file: {problem}")
            return 2
    solved = all(
        s["outcome"] == "optimal" for found in programmes.values() for s in found["solves"]
    )
    print("every solve ended optimal" if solved else "FAILED: a solve did not end optimal")
    return 0 if solved else 1


if __name__ == "__main__":
    sys.exit(main())
