"""The units a report command reads, as its `--units` argument (`ceiling_report`,
`dev_day_checks`, `flexitroid_days`, `forecast_report`).

Without `--units` a command reads its default list, the fixed tuple it read before the argument
existed, in the same order, so a run without the argument is the run it was. Each name given must
be one of `KNOWN_UNITS` and be named once, or the command could not run (exit 2); a command checks
the list before it reads any run folder or table.

`KNOWN_UNITS` are the units the truth build writes (plan section 3): the training units
`truth_build.TRAIN_UNITS` and the development units `truth_build.DEVELOPMENT_UNITS`, in that order.
Held-out units are not among them: their data are read only through the sealed-data path at the
tag (plan section 3, "Seal rule": the held-out loader and the held-out truth build), and these
commands have no route to the tables that path writes, so a held-out unit's name is refused like
any other name the truth build does not write. `ceiling_report` alone has the sealed route
(`sealed_route.units_route`): a list of held-out units only goes there, a list mixing them with
other units is refused there, and only a list of no held-out unit is checked here.

Reading U1: a list given is read in the order given, so an output that lists units in the order
read (a report's lines per unit) follows it, and an output that sorts them (a table sorted by
unit, a results file with sorted keys) does not; the days drawn by a seeded sample do not depend
on it (`block_sample`). The alternative, not taken, puts every list in one fixed order.

Reading U2: a list may not name a pooled unit beside one of the sites it pools
(`norway-development-pooled` and `OSL_1`, say; `POOLS`): the pooled unit holds every day of each
of its sites, so such a site-day would be read twice and counted twice in a figure a command
pools over all its units. Such a list could not run (exit 2), checked with the rest of the list.
The alternative, not taken, reads each as its own unit.
"""

import argparse
from collections.abc import Sequence

from fleetreserve import load_norway
from fleetreserve.sealed_route import AT_THE_TAG
from fleetreserve.truth_build import DEVELOPMENT_UNITS, POOLED, TRAIN_UNITS
from fleetreserve.truth_verify import CouldNotRun

KNOWN_UNITS = (*TRAIN_UNITS, *DEVELOPMENT_UNITS)
POOLS = {POOLED: load_norway.SITES["development"]}  # the truth build's pooled unit and its sites


def add_units_argument(parser: argparse.ArgumentParser, default: Sequence[str]) -> None:
    """`--units <unit> [<unit> ...]`, read as `args.units`, by default `default`."""
    parser.add_argument("--units", nargs="+", default=list(default), metavar="UNIT")


def check_units(units: Sequence[str]) -> tuple[str, ...]:
    """`units` as a tuple when each is a known unit named once and no pooled unit is named beside
    a site it pools; raises `CouldNotRun` otherwise."""
    units = tuple(units)
    unknown = [unit for unit in units if unit not in KNOWN_UNITS]
    if unknown:
        raise CouldNotRun(
            f"--units names {', '.join(unknown)}, not a training or development unit of the truth "
            f"build ({', '.join(KNOWN_UNITS)}); held-out units are read only through the "
            f"sealed-data path {AT_THE_TAG}, and this command has no route to them"
        )
    twice = sorted({unit for unit in units if units.count(unit) > 1})
    if twice:
        raise CouldNotRun(f"--units names {', '.join(twice)} more than once")
    if not units:
        raise CouldNotRun("--units names no unit")
    for pooled, sites in POOLS.items():
        inside = [unit for unit in units if unit in sites]
        if pooled in units and inside:
            raise CouldNotRun(
                f"--units names {pooled} beside {', '.join(inside)}, which it pools, so their "
                "site-days would be read twice and counted twice in a figure pooled over the "
                "units named"
            )
    return units
