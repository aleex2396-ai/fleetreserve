"""Run the whole test suite as several pytest processes at once and give one result.

Run as `python tools/run_tests.py [--lane default|learn] [--set fast|slow|all] [--jobs N]
[--sequential] [--root FOLDER]`. The result has the
meaning of one plain `pytest -c pyproject.toml` call over the whole suite, and the same three
exit statuses as `check.sh`: 0 every test passed, 1 a test failed, 2 could not run.

Steps:
  1. One collection-only pytest call lists every test id exactly as a whole-suite run would
     collect it (so the project's own rules for leaving out a file still apply). If that call
     itself ends in anything but "collected, nothing run", the result is 2, as it is for a plain
     whole-suite run.
  2. The test files are split into groups by the approximate weights in `tests/weights.txt`
     (a greedy split, heaviest file first). A file missing from the table gets a default
     weight, so every collected file is always in a group. The groups must cover the collected
     files exactly once; if not, the result is 2.
  3. Each group runs in its own `python -m pytest` process, all at once. Its whole output is
     kept and printed when all groups are done, in group order.
  4. The result is read from each group's exit status and summary line.

Result rules:
  1  if any group ended with exit status 1 (a test failed or errored).
  2  if no group failed, and any group could not run: exit status 2, 3, 4 or 5, killed by a
     signal, a summary line that cannot be read, a passed count that differs from the number of
     tests collected for that group, or any skipped, expected-to-fail or deselected test;
     also if the total passed differs from the collected total, or nothing was collected.
  0  otherwise.

Lanes: `default` (the default) is the suite without `tests/learn`. `learn` is only `tests/learn`,
which needs the packages of the `learn` dependency group; start it with
`uv run --frozen --group learn python tools/run_tests.py --lane learn`. The result rules are the
same, so a lane with no tests is 2.

Sets: `all` (the default) runs every collected file. `slow` runs only the files listed in
`tests/slow.txt` (one path per line, `<path>  # <reason>`), `fast` runs every other file. A file is
left out by not being given to a group, never by a pytest skip or deselect, so the rule above
that any skipped or deselected test is "could not run" holds for every set. The whole lane is
still collected first, so the project's own rules for leaving a file out still apply. Result 2
(could not run) for: a list line that is not `<path>  # <reason>` or names a path twice; a listed
file that does not exist; a listed file of the chosen lane with no collected test; a chosen set
with no test. Fast and slow together cover every collected file exactly once; the run stops with
2 if they would not. A listed file under `tests/learn/` belongs to the learn lane and is ignored by
the default lane, and a listed file elsewhere is ignored by the learn lane. `--set all` does not
read the list. The list is `tests/slow.txt` in the root unless `--slow-list` names another file.

`--sequential` runs the one plain pytest call (for a set: with the slow files given to
`--ignore`, or only the slow files named). The last printed line holds the totals, and for a
non-zero result, the group and the files that caused it.
"""

import argparse
import concurrent.futures
import os
import re
import resource
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_JOBS = 6
DEFAULT_WEIGHT = 1.0
WEIGHTS_FILE = Path(__file__).resolve().parents[1] / "tests" / "weights.txt"
PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2
SETS = ("fast", "slow", "all")
SLOW_LIST = Path("tests") / "slow.txt"  # in the root; one line `<path>  # <reason>` per file
LEARN_FOLDER = "tests/learn/"
# Lanes: the default lane is the whole suite without `tests/learn`. The learn lane is only
# `tests/learn`; it needs the `learn` dependency group, so its caller starts it in that environment.
# Each lane is (pytest options for every call, the paths the listing call is given).
LANES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "default": ((), ()),
    "learn": (("--learn-lane",), ("tests/learn",)),
}


class CouldNotRun(Exception):
    """Something stops the run from giving a pass or a fail; the exit status is 2.

    The message is one line. `detail` holds further lines to print before the final line.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


# --- the weights table ----


def read_weights(path: Path) -> dict[str, float]:
    """Read lines `<weight> <path>`; a path ending in `/` is a folder and covers its files."""
    try:
        text = path.read_text()
    except OSError as error:
        raise CouldNotRun(f"cannot read the weights table {path.name}: {error}") from error
    table: dict[str, float] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        try:
            weight = float(parts[0])
        except ValueError:
            weight = -1.0
        if len(parts) != 2 or not weight > 0:
            raise CouldNotRun(f"{path.name}:{number}: expected '<weight above 0> <path>'")
        table[parts[1]] = weight
    return table


def weight_of(file: str, table: dict[str, float]) -> float:
    """A file entry wins, then the longest folder entry that holds the file, then the default."""
    if file in table:
        return table[file]
    folders = [name for name in table if name.endswith("/") and file.startswith(name)]
    if folders:
        return table[max(folders, key=len)]
    return DEFAULT_WEIGHT


# --- the slow list ----


def lane_of(file: str) -> str:
    """The lane that collects a test file: `learn` for tests/learn, else `default`."""
    return "learn" if file.startswith(LEARN_FOLDER) else "default"


def read_slow_list(path: Path, root: Path) -> list[str]:
    """Read lines `<path>  # <reason>`; every listed file must exist in the root."""
    try:
        text = path.read_text()
    except OSError as error:
        raise CouldNotRun(f"cannot read the slow list {path.name}: {error}") from error
    listed: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        name, hash_sign, reason = raw.partition("#")
        file = name.strip()
        if not hash_sign or not reason.strip() or len(file.split()) != 1:
            raise CouldNotRun(f"{path.name}:{number}: expected '<path>  # <reason>'")
        if not file.endswith(".py") or file.startswith("/") or ".." in file.split("/"):
            raise CouldNotRun(f"{path.name}:{number}: {file} is not a test file path in the root")
        if file in listed:
            raise CouldNotRun(f"{path.name}:{number}: {file} is listed twice")
        if not (root / file).is_file():
            raise CouldNotRun(f"{path.name}:{number}: {file} does not exist")
        listed.append(file)
    return listed


def select_set(
    collected: dict[str, int], listed: list[str], lane: str, test_set: str
) -> dict[str, int]:
    """Return the collected files of the chosen set; fast and slow together are `collected`."""
    if test_set == "all":
        return collected
    mine = {file for file in listed if lane_of(file) == lane}
    for file in sorted(mine - set(collected)):
        raise CouldNotRun(f"{file} is in the slow list but no test of it was collected")
    slow = {file: n for file, n in collected.items() if file in mine}
    fast = {file: n for file, n in collected.items() if file not in mine}
    if {**fast, **slow} != collected or set(fast) & set(slow):
        raise CouldNotRun("the fast and the slow set do not cover the collected files once each")
    chosen = fast if test_set == "fast" else slow
    if not chosen:
        raise CouldNotRun(f"the {test_set} set of the {lane} lane holds no tests")
    return chosen


# --- collecting and grouping ----

_ID = re.compile(r"^(?P<file>[^\s:][^:]*\.py)::\S")
_COLLECT_ONLY_NOTE = re.compile(r"^could not run all tests \(\d+ collected but not run\): exit")


def pytest_command(config: str, *args: str) -> list[str]:
    return [sys.executable, "-m", "pytest", "-c", config, *args]


def collect(root: Path, config: str, lane: str = "default") -> dict[str, int]:
    """Return {test file: number of tests}, as a pytest run of the whole lane collects them."""
    options, paths = LANES[lane]
    done = subprocess.run(
        pytest_command(config, "--collect-only", "-p", "no:cacheprovider", *options, *paths),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = done.stdout.splitlines()
    # The project's rule ends a run that collected tests but ran none with status 2 and one
    # note line. That is the only collection-only ending that counts as a good listing.
    if done.returncode != 0 and not (
        done.returncode == 2 and any(_COLLECT_ONLY_NOTE.match(line) for line in lines)
    ):
        tail = "\n".join(lines[-15:] + done.stderr.splitlines()[-5:])
        raise CouldNotRun(f"collection ended with exit status {done.returncode}", tail)
    counts: dict[str, int] = {}
    for line in lines:
        found = _ID.match(line)
        if found:
            counts[found["file"]] = counts.get(found["file"], 0) + 1
    if not counts:
        raise CouldNotRun("no test ids were collected")
    return counts


def split_into_groups(files: dict[str, float], jobs: int) -> list[list[str]]:
    """Greedy split: heaviest file first, each into the lightest group so far."""
    size = max(1, min(jobs, len(files)))
    loads = [0.0] * size
    groups: list[list[str]] = [[] for _ in range(size)]
    for file in sorted(files, key=lambda name: (-files[name], name)):
        lightest = min(range(size), key=lambda index: (loads[index], index))
        groups[lightest].append(file)
        loads[lightest] += files[file]
    return [sorted(group) for group in groups if group]


def check_partition(collected: dict[str, int], groups: list[list[str]]) -> None:
    """Every collected file is in exactly one group, and no group holds anything else."""
    seen: dict[str, int] = {}
    for number, group in enumerate(groups, start=1):
        for file in group:
            if file in seen:
                raise CouldNotRun(f"{file} is in group {seen[file]} and group {number}")
            seen[file] = number
    missing = sorted(set(collected) - set(seen))
    extra = sorted(set(seen) - set(collected))
    if missing:
        raise CouldNotRun(f"collected but in no group: {', '.join(missing)}")
    if extra:
        raise CouldNotRun(f"in a group but not collected: {', '.join(extra)}")


def plan(
    root: Path,
    config: str,
    jobs: int,
    weights: Path,
    grouping: Callable[[dict[str, float], int], list[list[str]]] = split_into_groups,
    lane: str = "default",
    test_set: str = "all",
    slow_list: Path | None = None,
) -> tuple[dict[str, int], list[list[str]]]:
    """Collect, choose the set, split and check the split. Runs no test."""
    table = read_weights(weights)
    listed = [] if test_set == "all" else read_slow_list(slow_list or root / SLOW_LIST, root)
    collected = select_set(collect(root, config, lane), listed, lane, test_set)
    groups = grouping({file: weight_of(file, table) for file in collected}, jobs)
    check_partition(collected, groups)
    return collected, groups


# --- running the groups ----

_SUMMARY = re.compile(r"^\W*(?P<body>.*\S) in \d+(?:\.\d+)?s(?: \(\d+:\d{2}:\d{2}\))?\W*$")
_PART = re.compile(r"^(?P<n>\d+) (?P<kind>[a-z][a-z ]*)$")
_PROBLEM_FILE = re.compile(r"^(?:FAILED|ERROR) (?P<file>[^\s:]+\.py)")
_KINDS = {"error": "errors", "warning": "warnings"}
BAD_KINDS = ("failed", "errors", "skipped", "xfailed", "xpassed", "deselected")
KNOWN_KINDS = {"passed", "warnings", *BAD_KINDS}


def parse_summary(output: str) -> dict[str, int] | None:
    """Read the last 'N passed, M failed in 1.2s' line into counts; None if there is none."""
    for line in reversed(output.splitlines()):
        found = _SUMMARY.match(line)
        if not found:
            continue
        counts: dict[str, int] = {}
        body = found["body"]
        if body != "no tests ran":
            for part in body.split(", "):
                piece = _PART.match(part)
                if not piece:
                    return None
                kind = _KINDS.get(piece["kind"], piece["kind"])
                if kind not in KNOWN_KINDS:
                    return None
                counts[kind] = counts.get(kind, 0) + int(piece["n"])
        return counts
    return None


@dataclass
class GroupResult:
    number: int
    files: list[str]
    expected: int
    returncode: int | None = None
    output: str = ""
    seconds: float = 0.0
    counts: dict[str, int] | None = None
    status: int = COULD_NOT_RUN
    reason: str = ""
    problem_files: list[str] = field(default_factory=list)

    def passed(self) -> int:
        return (self.counts or {}).get("passed", 0)


def judge(result: GroupResult) -> None:
    """Set the result's status, reason and the files named as the cause."""
    result.counts = parse_summary(result.output)
    counts = result.counts or {}
    named = {m["file"] for m in map(_PROBLEM_FILE.match, result.output.splitlines()) if m}
    result.problem_files = sorted(named & set(result.files))
    code = result.returncode
    if code is not None and code < 0:
        result.reason = f"killed by signal {-code}"
    elif code == 1:
        result.status, result.reason = FAILED, "a test failed or errored"
    elif code != 0:
        result.reason = f"pytest exit status {code}"
    elif result.counts is None:
        result.reason = "the summary line cannot be read"
    elif any(counts.get(kind, 0) for kind in BAD_KINDS):
        result.reason = "a test was skipped, expected to fail or deselected"
    elif counts.get("passed", 0) != result.expected:
        result.reason = f"{counts.get('passed', 0)} passed, {result.expected} collected"
    else:
        result.status, result.reason = PASSED, ""
    if result.status != PASSED and not result.problem_files:
        result.problem_files = list(result.files)


def run_groups(
    root: Path,
    config: str,
    groups: list[list[str]],
    collected: dict[str, int],
    lane: str = "default",
) -> list[GroupResult]:
    """Run every group at once; return the judged results in group order."""
    results = [
        GroupResult(number, group, sum(collected[f] for f in group))
        for number, group in enumerate(groups, start=1)
    ]
    running: list[subprocess.Popen[str]] = []

    def run(result: GroupResult) -> None:
        start = time.monotonic()
        command = pytest_command(config, "-p", "no:cacheprovider", *LANES[lane][0], *result.files)
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running.append(process)
        result.output, _ = process.communicate()
        result.returncode = process.returncode
        result.seconds = time.monotonic() - start
        judge(result)
        print(
            f"group {result.number} finished: exit {result.returncode}, {result.seconds:.1f} s",
            file=sys.stderr,
            flush=True,
        )

    def stop_all(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    old = signal.signal(signal.SIGTERM, stop_all)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(results)) as pool:
            futures = [pool.submit(run, result) for result in results]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                for process in running:
                    if process.poll() is None:
                        process.kill()
                raise
    finally:
        signal.signal(signal.SIGTERM, old)
    return results


def decide(results: list[GroupResult], expected_total: int) -> tuple[int, str]:
    """Return (exit status, the first cause or an empty text)."""
    failed = [r for r in results if r.status == FAILED]
    stuck = [r for r in results if r.status == COULD_NOT_RUN]
    if failed:
        return FAILED, cause(failed[0])
    if stuck:
        return COULD_NOT_RUN, cause(stuck[0])
    total = sum(r.passed() for r in results)
    if expected_total == 0 or total != expected_total:
        return COULD_NOT_RUN, f"{total} passed in all groups, {expected_total} collected"
    return PASSED, ""


def cause(result: GroupResult) -> str:
    return f"group {result.number}: {result.reason}; files: {', '.join(result.problem_files)}"


def total_of(results: list[GroupResult], kind: str) -> int:
    return sum((r.counts or {}).get(kind, 0) for r in results)


# --- command line ----


def cpu_seconds() -> float:
    used = resource.getrusage(resource.RUSAGE_CHILDREN)
    return used.ru_utime + used.ru_stime


def run_sequential(
    root: Path,
    config: str,
    lane: str = "default",
    test_set: str = "all",
    slow_list: Path | None = None,
) -> int:
    options, paths = LANES[lane]
    extra: list[str] = []
    if test_set != "all":
        listed = read_slow_list(slow_list or root / SLOW_LIST, root)
        mine = [file for file in listed if lane_of(file) == lane]
        if not mine:
            raise CouldNotRun(f"the {test_set} set of the {lane} lane holds no tests")
        if test_set == "fast":
            extra = [f"--ignore={file}" for file in mine]
        else:
            paths = tuple(mine)
    done = subprocess.run(pytest_command(config, *options, *paths, *extra), cwd=root, check=False)
    return done.returncode if done.returncode in (PASSED, FAILED) else COULD_NOT_RUN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the test suite in several processes.")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="processes (default 6)")
    parser.add_argument("--sequential", action="store_true", help="one plain pytest call")
    parser.add_argument(
        "--lane",
        choices=sorted(LANES),
        default="default",
        help="default: the suite without tests/learn; learn: only tests/learn (needs the learn "
        "dependency group)",
    )
    parser.add_argument(
        "--set",
        dest="test_set",
        choices=SETS,
        default="all",
        help="fast: every file but those in the slow list; slow: only those; all: every file "
        "(default)",
    )
    parser.add_argument("--slow-list", type=Path, default=None, help="default: tests/slow.txt")
    parser.add_argument("--plan", action="store_true", help="print the groups and run nothing")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="pyproject.toml", help="pytest config, in the root")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_FILE)
    return parser


def finish(status: int, text: str) -> int:
    print(f"run_tests: exit {status}: {text}", flush=True)
    return status


def main(
    argv: list[str] | None = None,
    grouping: Callable[[dict[str, float], int], list[list[str]]] = split_into_groups,
) -> int:
    args = build_parser().parse_args(argv)  # a usage error exits with status 2
    root = args.root.resolve()
    if args.jobs < 1:
        return finish(COULD_NOT_RUN, "--jobs must be at least 1")
    if args.sequential:
        try:
            return run_sequential(root, args.config, args.lane, args.test_set, args.slow_list)
        except CouldNotRun as error:
            return finish(COULD_NOT_RUN, f"could not run: {error}")
    jobs = min(args.jobs, os.cpu_count() or 1)
    start = time.monotonic()
    try:
        collected, groups = plan(
            root,
            args.config,
            jobs,
            args.weights,
            grouping,
            args.lane,
            args.test_set,
            args.slow_list,
        )
    except CouldNotRun as error:
        if error.detail:
            print(error.detail)
        return finish(COULD_NOT_RUN, f"could not run: {error}")
    if args.plan:
        for number, group in enumerate(groups, start=1):
            tests = sum(collected[f] for f in group)
            print(f"group {number}: {len(group)} files, {tests} tests: {' '.join(group)}")
        return PASSED
    cpu_before = cpu_seconds()
    try:
        results = run_groups(root, args.config, groups, collected, args.lane)
    except OSError as error:
        return finish(COULD_NOT_RUN, f"could not run: {error}")
    for result in results:
        print(f"\n===== group {result.number}: {len(result.files)} files =====")
        print(result.output.rstrip("\n"))
    print()
    for result in results:
        print(
            f"group {result.number}: {len(result.files)} files, {result.expected} tests, "
            f"{result.passed()} passed, {result.seconds:.1f} s, exit {result.returncode}"
        )
    status, why = decide(results, sum(collected.values()))
    if why:
        for result in results:
            if result.status != PASSED:
                print(f"cause: {cause(result)}")
    seconds, cpu = time.monotonic() - start, cpu_seconds() - cpu_before
    totals = ", ".join(
        f"{total_of(results, kind)} {kind}" for kind in ("passed", "failed", "errors", "skipped")
    )
    tail = f"{totals}, {len(groups)} groups, {seconds:.1f} s (cpu {cpu:.1f} s)"
    if args.test_set != "all":
        tail += f", set {args.test_set}"
    return finish(status, f"{tail}; {why}" if why else tail)


if __name__ == "__main__":
    sys.exit(main())
