"""The run list's checker makes the sealed route's decision as each command of the route makes it
at its start (`run_list.route_fault`, reading L15), on made-up argument sets.

1. The table of the route's commands (`run_list.ROUTE_COMMANDS`) is the modules under `src/`
   that reach the sealed-data gate as the scan of `tests/test_run_evidence_gate_callers.py` finds
   them, less the held-out truth build and the driver, each with the one deciding function of
   the route it names (`sealed_route.matrix_route`, `units_route`, or `check_units` through
   `cold_start_gate`).
2. Parity: for each command and each argument set of its table (the review's four kinds of
   list (a) to (d), the honest set, and each refusal branch of the deciding functions reached
   from a command line), the checker's verdict equals what the REAL command's start does: its
   `main` runs, in this process, in a made-up repository at the tag `frozen-plan`
   (`tests/wp2/fake_heldout.py`), with the matrices committed there, until the first thing it
   does after the route's decision, which is replaced by a stand-in that stops it ("passed").
   A refusal before that is the command's exit 2 and its printed line; the checker's line holds
   that refusal's text. The commands' own checks of other options before the route (a dial of
   the matrix's family) are replaced by stand-ins that accept, since the checker does not make
   them (module text, "Not checked").
3. The checker opens, lists and scans nothing under the made-up repository's data folder, its
   sealed folder included (`tests/wp6gate/sealed_watch.py`), on every set; every refusal branch
   of the three deciding functions is reached by the table.
"""

import os
import sys
from pathlib import Path

import pytest
from drive_fake import write_matrix

from fleetreserve import (
    ceiling_report,
    cold_start,
    compare_run,
    corner_run,
    drift_run,
    gate_run,
    learn_run,
    ledger_run,
    o3_run,
    run_list,
    sealed_route,
)

TESTS = Path(__file__).resolve().parents[1]
for folder in (TESTS / "wp2", TESTS / "wp6gate", TESTS):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import fake_heldout  # noqa: E402
import sealed_watch  # noqa: E402
from heldout_truth_fake import sealed_files  # noqa: E402

TOP = Path(__file__).resolve().parents[2]
NINE = list(sealed_route.HELD_OUT_UNITS)
DEV = ["nrel-before-may-2019", "nrel-from-may-2019"]
TRAIN = ["caltech"]
MATRICES = {
    "gate/nine.json": NINE,
    "gate/ten.json": [*NINE, DEV[0]],
    "gate/development.json": DEV,
}
PASSED = "passed the route's decision"
ONE = "main-k0.05-empirical"  # a configuration of the committed matrix, for `--only`


class Stop(Exception):
    """The stand-in for the first thing a command does after the route's decision."""


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """A made-up repository at the tag: made-up sealed files, the running package, the made-up
    matrices and copies of the two committed default matrices, in the tag's commit."""
    folder = tmp_path_factory.mktemp("parity")
    root = fake_heldout.sealed_repo(folder, sealed_files(folder / "made"), tag=False)
    for path, units in MATRICES.items():
        write_matrix(root, path, units)
    for path in ("gate/matrix.json", "learn/matrix.json"):  # the commands' defaults
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes((TOP / path).read_bytes())
    fake_heldout.commit(root, "made-up matrices")
    fake_heldout.git(root, "tag", fake_heldout.TAG)
    return root


# --- 1. the table of the route's commands -------------------------------------------------------


def test_the_route_commands_are_the_modules_that_reach_the_gate_with_their_deciding_function():
    import test_run_evidence_gate_callers as scan_test

    scan = scan_test.Scan(scan_test.sources_of(scan_test.SOURCE))
    callers = {
        key for key in scan.trees if key not in scan_test.DEFINING and scan.reaches_the_gate(key)
    }
    deciding = {run_list.MATRIX_ROUTE, run_list.UNITS_ROUTE, run_list.CHECK_UNITS}

    def named(key: str) -> set[str]:
        refs = {ref for unit in scan.units(key) for ref in unit.refs}
        return {name for module, name in refs if module in scan_test.DEFINING} & deciding

    # the held-out truth build passes the gate itself and is rule 1's build; the driver drives
    # the checker itself calls the deciding functions (`run_list`, in NO_RESULTS there)
    assert callers - set(run_list.ROUTE_COMMANDS) == {
        "heldout_truth_build",
        "run_driver",
        "run_list",
        "run_starts",  # the start check: the real starts, at a made-up tag (run_starts)
    }
    assert set(run_list.ROUTE_COMMANDS) <= callers
    assert {key: named(key) for key in run_list.ROUTE_COMMANDS} == {
        key: {function} for key, function in run_list.ROUTE_COMMANDS.items()
    }
    assert named("heldout_truth_build") == named("run_driver") == set()


# --- 2. parity with the real commands' starts ---------------------------------------------------

ROUTE = [
    "--tables",
    "data/derived/development",
    "--truth-run",
    "data/derived/runs/development",
]
TABLES = ["--held-out-tables", "data/derived/truth-sealed"]
TRUTH = ["--held-out-truth-run", "data/derived/runs/sealed-truth"]
ALL = [*ROUTE, *TABLES, *TRUTH]

# what each command needs before the route that no set changes
NEEDS = {
    "learn_run": ["--model", "made-up"],
    "gate_run": [],
    "compare_run": ["--model", "made-up", "--dial", "0.02", "--splits", "1"],
    "corner_run": ["--model", "made-up", "--dial", "0.02"],
    "ledger_run": ["--run", "data/derived/runs/scored", "--tree-days", "1", "--seed", "1"],
    "drift_run": ["--model", "made-up", "--dial", "0.02"],
    "o3_run": ["--days", "1", "--seed", "1", "--base-set", "made-up"],
    "ceiling_report": ["--every-day"],
    "cold_start": ["--model", "made-up", "--dial", "0.02"],
}
# the first thing each command does after the route's decision
AFTER = {
    "learn_run": [(learn_run, "load_model")],
    "gate_run": [(gate_run, "read_inputs")],
    "compare_run": [(learn_run, "load_model")],
    "corner_run": [(corner_run, "route_refusal")],
    "ledger_run": [(ledger_run, "check_run_files"), (ledger_run, "read_run")],
    "drift_run": [(drift_run, "check_units")],
    "o3_run": [(o3_run, "check_units")],
    "ceiling_report": [(ceiling_report, "route_units")],
    "cold_start": [(cold_start, "read")],
}
MAINS = {
    "learn_run": learn_run.main,
    "gate_run": gate_run.main,
    "compare_run": compare_run.main,
    "corner_run": corner_run.main,
    "ledger_run": ledger_run.main,
    "drift_run": drift_run.main,
    "o3_run": o3_run.main,
    "ceiling_report": ceiling_report.main,
    "cold_start": cold_start.main,
}


def matrix(path: str) -> list[str]:
    return ["--matrix", path]


def units(*names: str) -> list[str]:
    return ["--units", *names]


MATRIX_SETS = {
    "honest: the nine, the route's options": [*matrix("gate/nine.json"), *ALL],
    "(a) the nine and a development unit, the truth run": [*matrix("gate/ten.json"), *ALL],
    "(b) a development matrix, the truth run and tables": [
        *matrix("gate/development.json"),
        *ALL,
    ],
    "(b) a development matrix, the truth run alone": [
        *matrix("gate/development.json"),
        *ROUTE,
        *TRUTH,
    ],
    "(c) the nine, the tables but no truth run": [*matrix("gate/nine.json"), *ROUTE, *TABLES],
    "(c) the nine, no held-out option": [*matrix("gate/nine.json"), *ROUTE],
    "a development matrix off the route": [*matrix("gate/development.json"), *ROUTE],
}
DEFAULT_SETS = {  # the commands with a default matrix
    "(b) no --matrix (its default), the truth run": [*ALL],
    "no --matrix (its default), off the route": [*ROUTE],
}
GATE_SETS = {
    "(d) the trial option --only on the route": [*matrix("gate/nine.json"), *ALL, "--only", ONE],
    "(d) the trial option --limit-days on the route": [
        *matrix("gate/nine.json"),
        *ALL,
        "--limit-days",
        "1",
    ],
}
UNIT_SETS = {
    "honest: the nine, the route's options": [*units(*NINE), *ALL],
    "(a) the nine and a development unit, the truth run": [*units(*NINE, DEV[0]), *ALL],
    "(b) development units, the truth run and tables": [*units(*DEV), *ALL],
    "(b) development units, the truth run alone": [*units(*DEV), *ROUTE, *TRUTH],
    "(c) the nine, the tables but no truth run": [*units(*NINE), *ROUTE, *TABLES],
    "(c) the nine, no held-out option": [*units(*NINE), *ROUTE],
    "development units off the route": [*units(*DEV), *ROUTE],
    "a training unit off the route": [*units(*TRAIN), *ROUTE],
    "a unit named twice": [*units(DEV[0], DEV[0]), *ROUTE],
    "an unknown unit": [*units("no-such-unit"), *ROUTE],
}
DRIFT_SETS = {"no --units (its default), off the route": [*ROUTE]}


def cases():
    for module, function in run_list.ROUTE_COMMANDS.items():
        if function == run_list.MATRIX_ROUTE:
            sets = dict(MATRIX_SETS)
            if run_list.command_default(module, "--matrix") is not None:
                sets |= DEFAULT_SETS
            if module == "gate_run":
                sets |= GATE_SETS
        else:
            sets = dict(UNIT_SETS) | (DRIFT_SETS if module == "drift_run" else {})
        for name, arguments in sets.items():
            yield pytest.param(module, arguments, id=f"{module}: {name}")


def real_start(module: str, arguments: list[str], root: Path, monkeypatch, capsys) -> str:
    """What the real command's start does with `arguments`: PASSED, or its printed refusal."""

    def stop(*args, **kwargs):
        raise Stop

    for owner, name in AFTER[module]:
        monkeypatch.setattr(owner, name, stop)
    # a check of another option before the route, which the checker does not make
    monkeypatch.setattr(compare_run, "dial_configurations", lambda m, d: {"forecast": None})
    capsys.readouterr()
    try:
        status = MAINS[module]([*NEEDS[module], *arguments], root=root)
    except Stop:
        return PASSED
    printed = capsys.readouterr().out.strip()
    assert status == 2, printed
    assert printed.startswith("COULD NOT RUN: "), printed
    return printed.removeprefix("COULD NOT RUN: ")


def checker(module: str, arguments: list[str], root: Path) -> str | None:
    step = run_list.Step(
        position=0,
        label="made-up",
        command=(sys.executable, "-m", f"fleetreserve.{module}", *NEEDS[module], *arguments),
        runs=(),
        setting="main",
        max_seconds=60,
        after=(),
    )
    with sealed_watch.watch(root / "data") as seen:
        found = run_list.route_fault(step, root)
    assert seen == []
    return found


@pytest.mark.parametrize(("module", "arguments"), list(cases()))
def test_the_checker_decides_the_route_as_the_commands_start_does(
    repo, monkeypatch, capsys, module, arguments
):
    monkeypatch.chdir(repo)
    found = checker(module, arguments, repo)
    real = real_start(module, arguments, repo, monkeypatch, capsys)
    if real == PASSED:
        assert found is None, found
    else:
        assert found is not None, real
        # the refusal's text, without a command's own reading name (" (reading C11)")
        assert real.split(" (reading ")[0] in found, (real, found)


def test_every_refusal_branch_of_the_deciding_functions_is_reached(repo, monkeypatch, capsys):
    """The table reaches the mix, the held-out options with no held-out unit, the missing truth
    run and `gate_run`'s own refusal on the route, for each deciding function."""
    monkeypatch.chdir(repo)
    reached = {}
    for param in cases():
        module, arguments = param.values
        found = checker(module, arguments, repo) or ""
        function = run_list.ROUTE_COMMANDS[module]
        for branch in (
            "mix held-out units",
            "for held-out units, and none",
            "held-out units need --held-out-truth-run",
            "builds a scored unit's forecasts",
            "for trials on development units",
            "are training units",
            "is named twice",
            "are neither development units",
        ):
            if branch in found:
                reached.setdefault(function, set()).add(branch)
    assert reached == {
        run_list.MATRIX_ROUTE: {
            "mix held-out units",
            "for held-out units, and none",
            "held-out units need --held-out-truth-run",
            "builds a scored unit's forecasts",
            "for trials on development units",
        },
        run_list.UNITS_ROUTE: {
            "mix held-out units",
            "for held-out units, and none",
            "held-out units need --held-out-truth-run",
        },
        run_list.CHECK_UNITS: {
            "held-out units need --held-out-truth-run",
            "are training units",
            "is named twice",
            "are neither development units",
            # check_units' mix ("never both") is caught by the line below
        },
    }
    mix = UNIT_SETS["(a) the nine and a development unit, the truth run"]
    mixed = checker("cold_start", mix, repo)
    assert mixed is not None and "never both" in mixed
