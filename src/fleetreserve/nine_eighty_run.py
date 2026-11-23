"""Estimate the parity of JPL's 9/80 Fridays off from JPL's training days (plan section 7).

Command line, from the project's top folder, under `fleetreserve.run_record`:
    python -m fleetreserve.nine_eighty_run [--tables <folder>] [--truth-run <folder>]

JPL's staff work a 9/80 schedule: every second Friday is off. Which of the two alternating sets
of Fridays is off is in no source this project holds (`calendar_inputs`, module text), so it is
estimated from JPL's own training days: `calendar_inputs.nine_eighty_phase`, the fortnight parity
whose counting Fridays have the fewer cars plugged in at τ on average.

Reads: the truth build's tables of the training unit `jpl` (`<tables>/jpl-cars.csv`,
`<tables>/jpl-days.csv`), checked against the recorded truth build at the checked-out commit as
`gate_run.check_inputs` checks them; of the day table, the columns `site`, `date`, `cars` and
`counts` only; the cars table is hashed, not parsed. No other unit is read. This is a look at
training data: once run, `docs/prior-looks.md` lists the run, what it read (JPL's number of cars
at τ on each counting Friday) and what it printed (the two parities' Friday counts and mean
cars, and the parity).

Output, in `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary also on
standard output: `nine-eighty-phase.json` (`nine_eighty_settings.RESULTS_NAME`): the unit, the
estimate (`parity`, `fridays_0`, `fridays_1`, `mean_cars_0`, `mean_cars_1`), the rule and the
truth build's summary for the unit (`inputs`). `nine_eighty_settings` writes the committed
settings file from it.

Exit status: 0 a parity was estimated; 1 none was (a parity with no counting Friday, or equal
means: the results are still written); 2 could not run (no verified truth build at this commit,
a table missing or differing from it).
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from fleetreserve import gate_run
from fleetreserve.calendar_inputs import nine_eighty_phase
from fleetreserve.days_from_truth import read_tables
from fleetreserve.nine_eighty_settings import RESULTS_NAME, UNIT
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

RULE = (
    "the fortnight parity (calendar_inputs.fortnight_parity) whose counting Fridays at jpl have "
    "the fewer cars plugged in at the decision time on average; none when a parity has no "
    "counting Friday or the means are equal"
)


def estimate(tables: Path) -> dict:
    """The estimate of the module text from the day table of `UNIT` in `tables`."""
    _, days = read_tables(tables, UNIT)
    found = nine_eighty_phase(days[["site", "date", "cars", "counts"]])
    return {
        key: (None if isinstance(v, float) and math.isnan(v) else v) for key, v in found.items()
    }


def summary(found: dict) -> str:
    def mean(value):
        return "none" if value is None else f"{value:.3f}"

    parities = "; ".join(
        f"parity {p}: {found[f'fridays_{p}']} counting Fridays, mean cars at τ "
        f"{mean(found[f'mean_cars_{p}'])}"
        for p in (0, 1)
    )
    chosen = found["parity"]
    verdict = (
        "no parity estimated" if chosen is None else f"estimated parity of Fridays off {chosen}"
    )
    return f"JPL 9/80 Fridays, {UNIT} training days: {parities}; {verdict}.\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        inputs = gate_run.check_inputs(args.truth_run, args.tables, (UNIT,))
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    found = estimate(args.tables)
    text = summary(found)
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        document = {"unit": UNIT, "estimate": found, "rule": RULE, "inputs": inputs}
        (Path(run_dir) / RESULTS_NAME).write_text(
            json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n"
        )
    return 1 if found["parity"] is None else 0


if __name__ == "__main__":
    sys.exit(main())
