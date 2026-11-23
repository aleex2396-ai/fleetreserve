"""One item of the WP5 pass mark, read from the recorded development-day checks (plan section 13,
WP5: "diag <= affine <= tree <= hindsight every development day; T2's check; two-way corner tests
match the LP; T4 log started, split by in-window departure").

Run as `python tools/dev_day_check.py <item> [options] [--run <run folder>] [--top <folder>]`
with exactly one item:
  --order                 the order on every counting day, both products;
  --t2 --sample-size <n> --energy-bounds <b> --uncertifiable-car <rule> --kappa <k>
       --departure-floor <rule>
                          T2's check on <n> sampled days under that forecast configuration;
  --two-way --sample-size <n>
                          the two-way corner tests at B = 5 kWh on <n> sampled days;
  --t4-log                the T4 log exists, covers every counting day, and V_hind = V_caus held
                          on every day meeting T4's hypotheses.

The run is the folder given, or else the newest folder `runs.noindex/<UTC start>-wp5-dev-days`
(`fleetreserve.run_evidence`; an older run is never used in its place), made by
`python -m fleetreserve.dev_day_checks`, recorded at the commit checked out now or read from an
earlier commit by the rule of `run_evidence` (module text there; else it is stale). Every line
this tool writes that names a run read from an earlier commit carries `Evidence.note` after the
run's name, whatever the line then says. Its `dev-day-checks.json` holds each check's verdict
with its lines and coverage.

An item passes only when the run is verified (it ended with exit 0), it read exactly the
development units of `dev_day_checks.UNITS` (the keys of its sample's `counting_days`), its
configuration is the one asked for (rho 0, budget 5 kWh, the sample size and, for T2, every
forecast argument), and the check covered every day it should have, with no timeout and no other
missing value (a timeout is missing, never a pass). A run that ended with exit 1 or 2 can fail an
item (its check has failure lines) but never pass one. A run on other units (the command's
`--units` can name others, or a part of them) or with another configuration could not run for the
item, whatever its checks found.

Exit status: 0 the item passed; 1 it failed; 2 it could not run.
"""

import argparse
import math
import sys
from pathlib import Path

from fleetreserve.dev_day_checks import (
    DEPARTURE_FLOORS,
    ENERGY_BOUNDS,
    RESULTS_NAME,
    RHO,
    UNITS,
)
from fleetreserve.dev_day_verdicts import COULD_NOT_RUN, FAILED, PASSED
from fleetreserve.run_evidence import CannotCheck, head_commit, newest_run, read_evidence
from fleetreserve.t2_day import UNCERTIFIABLE_RULES
from fleetreserve.two_way_day import BUDGET

LABEL = "wp5-dev-days"
ITEMS = {"order": "order", "t2": "t2", "two-way": "two_way", "t4-log": "t4_log"}
WORDS = {
    "order": "diag <= affine <= tree <= hindsight every development day (plan 13 WP5)",
    "t2": "T2's check (plan 5)",
    "two_way": "two-way corner tests match the LP (plan 9, 13 WP5)",
    "t4_log": "T4 log started, split by in-window departure (plan 4, 13 WP5)",
}
MAX_PRINTED = 20


def _same(found, wanted) -> bool:
    """Equal, and a number only where a number is asked for (no tolerance)."""
    if isinstance(wanted, float):
        number = isinstance(found, int | float) and not isinstance(found, bool)
        return number and math.isfinite(found) and found == wanted
    return found == wanted


def check_unit_list(data: dict) -> None:
    """The run read exactly the units of `dev_day_checks.UNITS`, in any order."""
    try:
        found = data["sample"]["counting_days"]
    except (KeyError, TypeError) as problem:
        raise CannotCheck("the results file does not say which units the run read") from problem
    if not isinstance(found, dict):
        raise CannotCheck("the results file does not say which units the run read")
    if set(found) != set(UNITS):
        named = ", ".join(sorted(map(str, found))) or "none"
        raise CannotCheck(
            f"the run read the units {named}, not the units of this pass mark "
            f"({', '.join(UNITS)}: dev_day_checks.UNITS)"
        )


def check_configuration(data: dict, options) -> None:
    try:
        config = data["configuration"]
        plan = data["sample"]
    except (KeyError, TypeError) as problem:
        raise CannotCheck("the results file has no configuration or sample") from problem
    wanted = {"rho": float(RHO), "budget_kwh": float(BUDGET)}
    if options.item == "t2":
        wanted |= {
            "energy_bounds": options.energy_bounds,
            "uncertifiable_car": options.uncertifiable_car,
            "kappa": float(options.kappa),
            "departure_floor": options.departure_floor,
        }
    for key, value in wanted.items():
        if not _same(config.get(key), value):
            raise CannotCheck(
                f"the run's {key} is {config.get(key)!r}, not {value!r} as this item asks"
            )
    if options.item in ("t2", "two_way") and plan.get("size") != options.sample_size:
        raise CannotCheck(
            f"the run's sample has {plan.get('size')!r} days, not {options.sample_size} as this "
            "item asks"
        )


def expected_days(data: dict, item: str) -> int:
    key = "counting_days" if item in ("order", "t4_log") else None
    found = data.get(key) if key else data["sample"].get("size")
    if isinstance(found, bool) or not isinstance(found, int):
        raise CannotCheck("the results file does not say how many days it should cover")
    return found


def show_t2(check: dict) -> None:
    print(
        f"T2: certified on {check.get('certified_days')} of {check.get('of_days')} sampled days; "
        f"days in E': {check.get('days_in_e_prime')} (share of sampled days "
        f"{check.get('share_of_sampled_days_in_e_prime')}, of certified days "
        f"{check.get('share_of_certified_days_in_e_prime')}); share of steered cars in E'_i "
        f"{check.get('share_of_steered_cars_in_e_prime_i')}; cars left unsteered: "
        f"{check.get('cars_left_unsteered_no_forecast')} without a forecast, "
        f"{check.get('cars_left_unsteered_empty_forecast_set')} with an empty forecast set; "
        f"days without a certificate by the rule: {check.get('days_without_certificate_by_rule')}"
    )


def check_item(options, run: Path, top: Path) -> int:
    item = options.item
    evidence = read_evidence(run, RESULTS_NAME, head_commit(top))
    data = evidence.data
    check_unit_list(data)
    check_configuration(data, options)
    try:
        check = data["checks"][item]
        result = check["verdict"]
        lines = check["failure_lines"]
        covered, of = check["covered_days"], check["of_days"]
        timeouts, other = check["timeout_days"], check["other_missing_days"]
    except (KeyError, TypeError) as problem:
        raise CannotCheck(f"the results file has no complete {item} check") from problem
    print(f"item: {WORDS[item]}; run {run.name}{evidence.note}")
    coverage = (
        f"covered {covered} of {of} days ({timeouts} with a timeout, {other} with another "
        "missing value)"
    )
    print(coverage)
    if item == "t2":
        show_t2(check)
    if result == FAILED:
        for line in lines[:MAX_PRINTED]:
            print(f"FAILED: {line}")
        if len(lines) > MAX_PRINTED:
            print(f"FAILED: and {len(lines) - MAX_PRINTED} more lines in the run's results")
        return 1
    if result == COULD_NOT_RUN:
        reasons = "; ".join(check.get("could_not_run_reasons", [])) or "no reason recorded"
        raise CannotCheck(f"the check could not run in {run.name}{evidence.note}: {reasons}")
    if result != PASSED:
        raise CannotCheck(f"the check's verdict {result!r} is not one of the three")
    should = expected_days(data, item)
    if of != should or covered != of or timeouts != 0 or other != 0 or of == 0:
        raise CannotCheck(f"the check did not cover all {should} days it should have: {coverage}")
    if item == "t4_log":
        log = data.get("t4_log_days")
        if not isinstance(log, list) or len(log) != should:
            raise CannotCheck("the T4 log does not list every counting day")
        print(
            f"T4 log: {check.get('hypothesis_days')} days meet T4's hypotheses; V_hind = V_caus "
            f"held on all {check.get('hypothesis_days_checked')} of them; findings V_caus above "
            f"V_aff (recorded, not judged): {len(check.get('findings_tree_above_affine', []))}"
        )
        if check.get("hypothesis_days_checked") != check.get("hypothesis_days"):
            raise CannotCheck("V_hind = V_caus was not checked on every hypothesis day")
    if not evidence.verified:
        raise CannotCheck(
            f"{run.name}{evidence.note} ended with exit code {evidence.manifest.get('exit_code')}, "
            "so it is not verified and cannot pass an item"
        )
    print(f"passed: {coverage}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/dev_day_check.py",
        description="Decide one item of the WP5 pass mark from the recorded development checks.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    for flag, item in ITEMS.items():
        group.add_argument(f"--{flag}", dest="item", action="store_const", const=item)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--energy-bounds", choices=tuple(ENERGY_BOUNDS))
    parser.add_argument("--uncertifiable-car", choices=UNCERTIFIABLE_RULES)
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--departure-floor", choices=DEPARTURE_FLOORS)
    parser.add_argument("--run", type=Path, help="the run folder (default: the newest run)")
    parser.add_argument(
        "--top",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the repository top folder, where runs.noindex/ is (default: this repository)",
    )
    options = parser.parse_args(argv)
    needed = {
        "t2": ("sample_size", "energy_bounds", "uncertifiable_car", "kappa", "departure_floor"),
        "two_way": ("sample_size",),
    }.get(options.item, ())
    lacking = [name for name in needed if getattr(options, name) is None]
    if lacking:
        print(
            "COULD NOT RUN: this item needs "
            + ", ".join("--" + name.replace("_", "-") for name in lacking)
        )
        return 2
    try:
        run = options.run if options.run is not None else newest_run(options.top, LABEL)
        return check_item(options, run, options.top)
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
