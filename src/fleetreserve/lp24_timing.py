"""Time LP (24) (`lp24`) on a seeded made-up fleet (plan section 13, WP4: "solve times ... of LP
(24) (50 cars, 14 hours) logged with the machine they ran on. Run time is reported, never judged").

Command line:
    python -m fleetreserve.lp24_timing [--cars 50] [--hours 14] [--repeats 3] [--seed 0]
        [--reading steered|eliminated|both] [--rho 0.1]

The fleet: `--cars` made-up cars, each with a departure hour drawn from 1 to `--hours`, a rate
drawn from `RATES` (kW) and a room drawn as a share in `[0.1, 0.9]` of what its rate allows
before it leaves, at most `MAX_ROOM` kWh; its protected energy is `(1 - rho)` times the room,
with `rho = 0.1` by default (one of the plan's values, section 2). Reading: the plan's main value
`rho = 0` makes every car's total energy fixed when no recovery hour is left to it (the steered
reading below), and then LP (24)'s `alpha` is 0 (the base set holds profiles of different
totals); `--rho 0` runs it. The base set is the day-average one (`base_sets.day_average`).

How "14 hours" maps onto the rows (a reading; the plan does not say):

- `steered` (default, the closest to the earlier analysis, whose programme steered every step of
  its horizon from the decision time on, with no service window and no recovery hours): all
  `--hours` hours are hours of the rows, hour 0 and `--hours - 1` later hours, each car with its
  power limits, a stored-energy row after every hour and its need (the earlier analysis's sets
  had no need). The rows are written here, since `carsets` allows at most 8 hours after hour 0.
- `eliminated` (the alternative): the plan's day, hour 0, 8 service hours and `--hours - 9`
  recovery hours, the same cars on their true one-way sets (`carsets.true_rows`), recovery hours
  eliminated, so the programme has 9 hours.

Each reading is solved `--repeats` times; the command prints, per reading, the programme's sizes,
`alpha`, and the seconds of each solve (solver only, and with the build), and the best of them.
It does not compare them with any limit. When `FLEETRESERVE_RUN_DIR` is set, the same numbers are
written there to `RESULTS_NAME`.

Exit status: 0 when every solve ended optimal, 1 when a solve ended otherwise (a timeout
included), 2 when the command could not run.
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from fleetreserve.base_sets import day_average
from fleetreserve.carsets import FleetRows, true_rows
from fleetreserve.day import MAX_SERVICE_HOURS, Day, TrueCar
from fleetreserve.lp24 import inner_set

RATES = (3.3, 6.6, 7.4, 11.0)  # kW
MAX_ROOM = 40.0  # kWh
RESULTS_NAME = "lp24-timing.json"
READINGS = ("steered", "eliminated")


def made_up_cars(n_cars: int, hours: int, seed: int) -> list[TrueCar]:
    rng = np.random.default_rng(seed)
    cars = []
    for _ in range(n_cars):
        departure = int(rng.integers(1, hours + 1))
        rate = float(rng.choice(RATES))
        room = min(float(rng.uniform(0.1, 0.9)) * rate * departure, MAX_ROOM)
        cars.append(TrueCar(departure, rate, room))
    return cars


def steered_rows(cars: list[TrueCar], hours: int, rho: float) -> FleetRows:
    """Every hour written out: power in `[0, r]` while present, stored energy at most the room
    after every hour, at least the protected energy `(1 - rho) e` by the end."""
    eye = np.eye(hours)
    prefix = np.tril(np.ones((hours, hours)))
    blocks, bounds, kinds, owner = [], [], [], []
    for i, car in enumerate(cars):
        present = (np.arange(hours) < car.departure).astype(float)
        blocks += [eye, -eye, prefix, -np.ones((1, hours))]
        bounds += [car.rate * present, np.zeros(hours), np.full(hours, car.room), [-car.need(rho)]]
        kinds += ["power-high"] * hours + ["power-low"] * hours + ["stored-high"] * hours
        kinds.append("need")
        owner.append(np.full(3 * hours + 1, i))
    return FleetRows(
        n_service=hours - 1,
        n_cars=len(cars),
        car=np.concatenate(owner),
        coefficients=np.vstack(blocks),
        bounds=np.concatenate([np.asarray(b, float) for b in bounds]),
        kinds=np.array(kinds),
    )


def eliminated_rows(cars: list[TrueCar], hours: int, rho: float) -> FleetRows:
    """The plan's day: hour 0, 8 service hours, the rest recovery hours, eliminated."""
    n_recovery = hours - 1 - MAX_SERVICE_HOURS
    if n_recovery < 0:
        raise ValueError(f"the eliminated reading needs at least {MAX_SERVICE_HOURS + 1} hours")
    day = Day("made-up", datetime.date(2020, 1, 1), MAX_SERVICE_HOURS, n_recovery, rho, tuple(cars))
    return true_rows(day)


def time_reading(rows: FleetRows, repeats: int) -> dict:
    base = day_average(rows)
    solves = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = inner_set(rows, base, "24")
        total = time.perf_counter() - start
        solves.append(
            {
                "outcome": result.outcome,
                "alpha": result.alpha,
                "solve_seconds": result.seconds,
                "seconds_with_build": total,
            }
        )
    return {
        "cars": rows.n_cars,
        "programme_hours": rows.n_service + 1,
        "variables": result.sizes.variables,
        "rows": result.sizes.rows,
        "nonzeros": result.sizes.nonzeros,
        "solves": solves,
        "best_solve_seconds": min(s["solve_seconds"] for s in solves),
        "best_seconds_with_build": min(s["seconds_with_build"] for s in solves),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cars", type=int, default=50)
    parser.add_argument("--hours", type=int, default=14)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reading", choices=(*READINGS, "both"), default="steered")
    parser.add_argument("--rho", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.cars < 1 or args.hours < 2 or args.repeats < 1 or not 0 <= args.rho < 1:
        print(
            "COULD NOT RUN: --cars and --repeats must be at least 1, --hours at least 2, "
            "--rho in [0, 1)"
        )
        return 2
    readings = READINGS if args.reading == "both" else (args.reading,)
    try:
        cars = made_up_cars(args.cars, args.hours, args.seed)
        build = {"steered": steered_rows, "eliminated": eliminated_rows}
        timings = {
            name: time_reading(build[name](cars, args.hours, args.rho), args.repeats)
            for name in readings
        }
    except ValueError as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    document = {
        "programme": "LP (24), day-average base set",
        "cars": args.cars,
        "hours": args.hours,
        "seed": args.seed,
        "rho": args.rho,
        "repeats": args.repeats,
        "readings": timings,
    }
    for name, t in timings.items():
        outcomes = ", ".join(s["outcome"] for s in t["solves"])
        print(
            f"{name}: {t['cars']} cars, {t['programme_hours']} programme hours; "
            f"{t['variables']} variables, {t['rows']} rows, {t['nonzeros']} nonzeros; "
            f"outcomes {outcomes}; alpha {t['solves'][-1]['alpha']}"
        )
        print(
            f"{name}: solve seconds "
            + ", ".join(f"{s['solve_seconds']:.3f}" for s in t["solves"])
            + "; with build "
            + ", ".join(f"{s['seconds_with_build']:.3f}" for s in t["solves"])
            + f"; best {t['best_solve_seconds']:.3f} ({t['best_seconds_with_build']:.3f})"
        )
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        try:
            (Path(run_dir) / RESULTS_NAME).write_text(json.dumps(document, indent=1) + "\n")
        except OSError as problem:
            print(f"COULD NOT RUN: cannot write the results file: {problem}")
            return 2
    solved = all(s["outcome"] == "optimal" for t in timings.values() for s in t["solves"])
    print("solved" if solved else "FAILED: a solve did not end optimal")
    return 0 if solved else 1


if __name__ == "__main__":
    sys.exit(main())
