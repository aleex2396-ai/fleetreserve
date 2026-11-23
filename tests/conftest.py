"""Test-session rule: a skipped, expected-to-fail, deselected or unrun test is never a pass.

A session that would end with status 0 but had any such test, or collected tests that neither
ran nor were skipped (for example with --collect-only or an early exit), ends with status 2
(could not run). A session with a failure keeps status 1. A session that collected no tests
keeps pytest's own status 5.

Second rule: a test file that reads a file kept only on this machine (never in git) is left out
of a run that does not name it, when that file is missing. A fresh checkout then still passes the
whole-suite check. A run that names the test file is not left out, and the test file itself ends
the run with status 2 (could not run) when its file is missing, so the work package's pass mark
cannot count it as passed.

Third rule: the folder `tests/learn` holds tests that need the packages of the `learn`
dependency group. A run leaves it out unless it is given `--learn-lane`, so the default
environment never imports those packages and never skips or fails for their absence. The learn
lane is run by `tools/run_tests.py --lane learn` inside `uv run --frozen --group learn`.

Import path: the toy reference programmes in `tests/wp1` (`toy_values.py`, `vertex_truth.py`),
the made-up days and full-set programmes in `tests/wp4` (`made_up_days.py`, `fleet_lps.py`), the
made-up truth tables and offer days in `tests/wp6` (`truth_tables_fake.py`,
`made_up_offer_days.py`), the made-up development build in `tests/wp2c` (`ceiling_fakes.py`),
the made-up gate units and truth build in `tests/wp6gate` (`gate_fake.py`), the count of opens of
made-up sealed files by identity on disk there (`sealed_watch.py`) and the made-up WP8
units in `tests/wp8` (`learn_fake.py`) are importable from every test folder, so later work
packages reuse them rather than copy them, also when their own folder runs alone.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _helpers in (
    ROOT / "tests" / "wp1",
    ROOT / "tests" / "wp4",
    ROOT / "tests" / "wp6",
    ROOT / "tests" / "wp2c",
    ROOT / "tests" / "wp6gate",
    ROOT / "tests" / "wp8",
):
    if _helpers.is_dir() and str(_helpers) not in sys.path:
        sys.path.insert(0, str(_helpers))
LEARN_FOLDER = "tests/learn"
# Test file (relative to the repository top) -> the local-only file it reads (same base).
NEEDS_LOCAL_FILE = {
    "tests/wp1/test_earlier_offer_matrices.py": "reference/part1/src/fleetoffer/offer.py",
}

_not_run = {"skipped": 0, "xfailed": 0, "xpassed": 0, "deselected": 0}
_collected: set[str] = set()
_seen: set[str] = set()


def pytest_addoption(parser):
    parser.addoption(
        "--learn-lane",
        action="store_true",
        default=False,
        help="also collect tests/learn, which needs the learn dependency group",
    )


def pytest_ignore_collect(collection_path, config):
    try:
        name = collection_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None
    in_learn = name == LEARN_FOLDER or name.startswith(LEARN_FOLDER + "/")
    if in_learn and not config.getoption("--learn-lane"):
        return True
    needed = NEEDS_LOCAL_FILE.get(name)
    return True if needed is not None and not (ROOT / needed).is_file() else None


def pytest_deselected(items):
    _not_run["deselected"] += len(items)


def pytest_collection_finish(session):
    _collected.update(item.nodeid for item in session.items)


def pytest_collectreport(report):
    # A skip while a test file loads (module-level skip, importorskip) has no test item.
    if report.skipped:
        _not_run["skipped"] += 1


def pytest_runtest_logreport(report):
    expected_to_fail = hasattr(report, "wasxfail")
    if report.when == "call" or report.skipped:
        _seen.add(report.nodeid)
    if report.skipped:
        _not_run["xfailed" if expected_to_fail else "skipped"] += 1
    elif report.passed and report.when == "call" and expected_to_fail:
        _not_run["xpassed"] += 1


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    unrun = len(_collected - _seen)
    problems = {kind: n for kind, n in _not_run.items() if n}
    if unrun:
        problems["collected but not run"] = unrun
    if exitstatus == pytest.ExitCode.OK and problems:
        detail = ", ".join(f"{n} {kind}" for kind, n in problems.items())
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line("")  # end the progress line (quiet mode leaves it open)
            reporter.write_line(f"could not run all tests ({detail}): exit status 2")
        session.exitstatus = 2
