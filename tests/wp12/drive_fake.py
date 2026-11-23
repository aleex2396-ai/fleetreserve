"""A throwaway git repository for drives of made-up run lists, for the tests of `run_driver` and of
the WP12 pass-mark check.

The repository holds, at its top, a stand-in package `fleetreserve`: `run_record.py` is a byte copy
of the real recorder (so each entry is recorded by the real code, which takes its real heavy-run
lock in the throwaway repository's own `runs.noindex/`), and `truth_build.py`, `learn_run.py`,
`compare_run.py`, `day_intervals.py` and the sealed truth build's module (`SEALED_TRUTH`, the name
`sealed_route.TRUTH_BUILD_MODULE` holds, which the run list's checker reads) are one made-up step
each (`STEP`). The recorder runs a command from the top
folder, so `python -m fleetreserve.<name>` there finds the stand-in, not the installed package;
the checker of the run list, in the test's own process, sees the real package, where the five
module names exist. The made-up step:
- writes its arguments to `step.txt` in its run folder (so a test reads the units and the run
  folders the driver put in place of `{units}` and `{run:<label>}`);
- sleeps for `--sleep <seconds>` (default 0);
- with `--dirty`, appends a line to the tracked file `tracked.txt`; with `--commit`, also commits
  that change;
- with `--sibling`, makes a second folder beside its run folder with the same label and a later
  start, `runs.noindex/29991231T235959Z-<label>`, so that a listing of the runs folder finds two;
- with `--term-parent`, sends SIGTERM to its parent, the recorder, and waits (the run is recorded
  as interrupted);
- exits with `--exit <status>` (default 0).
`ceiling_report.py` stands in for outcome O2's command (`O2_STEP`): it writes `ceiling-report.json`
into its run folder with each unit's O2 pass from `--verdicts <unit>=<pass>,...` (default one unit
passed) and the verdict over them as `ceiling_summary.pass_over` gives it, or as
`--over <verdict>` says; each unit's `faults` record (`ceiling_summary.o2_with_faults`) holds
zero counts, or `--faults <unit>=<orders broken>:<days not built>[:<block orders broken on M
days>[:<days in M not built>]],...` for the units named (`<unit>=bad` makes that unit's record no
mapping of counts, `noblock` one with no block count, `nomdays` an entry with no
`days_in_m_not_built`), or none with `--no-faults`; with
`--no-verdict` the file holds no `every_day`, with `--no-file` none is written; it exits with
`--exit <status>` (default 0). `o2_entries` gives outcome O2's two steps as the run list's rules
10 and 12 require them (rho 0, one in the main and one in the held setting, the owner's
decision 79 (4)), with the truth builds they read, on the groups of units `O2_UNITS`.
The driver's start check (`run_starts.start_faults`, called by `run_driver.prepare`) starts the
REAL commands of the sealed route, which refuse the stand-ins' made-up arguments, so a test that
drives these stand-ins replaces it with `no_start_check`, which records each call and finds no
fault; the start check in the driver is tested on the real commands in
`tests/wp12/test_run_driver_starts.py`.
The repository also holds `SEALED_MATRIX`, the committed gate matrix with the nine held-out units as
its gate units, for the run list's rule 3, and copies of the two committed matrices the route's
commands read by default (`DEFAULT_MATRICES`), which the checker's rule 5 reads for a step that
names no `--matrix`.
"""

import json
import sys
from pathlib import Path

from git_run import run_git

from fleetreserve import run_starts, sealed_route

TOP = Path(__file__).resolve().parents[2]
RECORDER = TOP / "src" / "fleetreserve" / "run_record.py"
SEALED_TRUTH = sealed_route.TRUTH_BUILD_MODULE  # the stand-in of the sealed truth build
SEALED_MATRIX = "gate/sealed-matrix.json"  # a made-up matrix of the nine held-out units
DEFAULT_MATRICES = ("gate/matrix.json", "learn/matrix.json")  # the commands' defaults
STEP_MODULES = ("truth_build", "learn_run", "compare_run", SEALED_TRUTH, "day_intervals")
LIST_PATH = "run-lists/run-list.json"  # the one fixed path the WP12 check reads
STEP = """import os
import signal
import subprocess
import sys
import time

arguments = sys.argv[1:]


def value(flag, default):
    return arguments[arguments.index(flag) + 1] if flag in arguments else default


with open(os.path.join(os.environ["FLEETRESERVE_RUN_DIR"], "step.txt"), "w") as handle:
    handle.write(" ".join(arguments) + "\\n")
if "--term-parent" in arguments:  # the recorder is stopped, as an operator stops a drive
    os.kill(os.getppid(), signal.SIGTERM)
    time.sleep(30)
time.sleep(float(value("--sleep", "0")))
if "--dirty" in arguments or "--commit" in arguments:
    with open("tracked.txt", "a") as handle:
        handle.write("changed by a step\\n")
if "--commit" in arguments:
    quiet = ["-c", "maintenance.auto=false", "-c", "gc.auto=0"]
    subprocess.run(["git", *quiet, "commit", "-q", "-am", "a step's commit"], check=True)
if "--sibling" in arguments:
    run_dir = os.environ["FLEETRESERVE_RUN_DIR"]
    label = os.path.basename(run_dir).split("-", 1)[1]
    os.mkdir(os.path.join(os.path.dirname(run_dir), "29991231T235959Z-" + label))
sys.exit(int(value("--exit", "0")))
"""
# The stand-in of the sealed truth build also writes the summary the real build always writes (its
# `abstained` record, empty here), which the WP14 check reads; a passed build step with none could
# not be checked.
TRUTH_BUILD_STEP = STEP.replace(
    "sys.exit(int(value",
    "with open(os.path.join(os.environ['FLEETRESERVE_RUN_DIR'], 'truth-summary.json'), 'w') as h:\n"
    '    h.write(\'{"units": {}, "abstained": {}}\')\n'
    "sys.exit(int(value",
)
# The stand-in of the learn step also writes the `learn.json` the real step writes, holding the
# failed configurations (`--failed "<configuration> | <unit>: <error>"`, each given, as
# `gate_run` writes one; none by default), which the WP14 check reads from a learn step that passed.
# `--rung "<configuration> | <unit>: <error>"` writes, as `learn_eval` does, the reason `failed:
# <error>` in that configuration's rungs for that unit, without listing it: `gate_run` lists one
# configuration of each group that failed together.
LEARN_STEP = STEP.replace(
    "sys.exit(int(value",
    "import json\n"
    "failed = [arguments[k + 1] for k, a in enumerate(arguments) if a == '--failed']\n"
    "units = {}\n"
    "for k, a in enumerate(arguments):\n"
    "    if a == '--rung':\n"
    "        name, rest = arguments[k + 1].split(' | ', 1)\n"
    "        unit, error = rest.split(': ', 1)\n"
    "        units.setdefault(unit, {'configurations': {}})['configurations'][name] = {\n"
    "            'rungs': {'reason': 'failed: ' + error}}\n"
    "with open(os.path.join(os.environ['FLEETRESERVE_RUN_DIR'], 'learn.json'), 'w') as h:\n"
    "    json.dump({'failed_configurations': failed, 'units': units}, h)\n"
    "sys.exit(int(value",
)
O2_MODULE = "ceiling_report"  # the stand-in of outcome O2's command
O2_STEP = """import json
import os
import sys

arguments = sys.argv[1:]
given = arguments[arguments.index("--verdicts") + 1] if "--verdicts" in arguments else ""
verdicts = dict(pair.split("=") for pair in given.split(",")) if given else {"unit-a": "passed"}
values = [value for value in verdicts.values() if value != "no day to check"]
if "failed" in values:
    over = "failed"
elif not values:
    over = "no day to check"
elif all(value == "passed" for value in values):
    over = "passed"
else:
    over = "could not run"
if "--over" in arguments:
    over = arguments[arguments.index("--over") + 1]
given = arguments[arguments.index("--faults") + 1] if "--faults" in arguments else ""
counts = {pair.split("=")[0]: pair.split("=")[1].split(":") for pair in given.split(",") if pair}


def counts_of(name):
    given = [part for part in counts.get(name, []) if part.isdigit()]  # not the words below
    orders, days, block, in_m = (given + ["0"] * 4)[:4]
    return int(orders), int(days), int(block), int(in_m)


def faults_of(name):
    if counts.get(name) == ["bad"]:
        return "not a mapping of counts"
    orders, days, block, _ = counts_of(name)
    found = {"order_breaks": orders, "days_not_built": days, "block_order_breaks_on_m_days": block}
    if counts.get(name) == ["noblock"]:
        del found["block_order_breaks_on_m_days"]
    return found


def unit_of(name, verdict):
    found = {
        "o2_pass": verdict,
        "faults": faults_of(name),
        "days_in_m_not_built": counts_of(name)[3],
    }
    if counts.get(name) == ["nomdays"]:
        del found["days_in_m_not_built"]
    return found


document = {
    "every_day": {
        "units": {name: unit_of(name, verdict) for name, verdict in verdicts.items()},
        "pass_over_units": over,
    },
    "failures": [],
}
if "--no-faults" in arguments:
    for found in document["every_day"]["units"].values():
        del found["faults"]
if "--no-verdict" in arguments:
    del document["every_day"]
if "--no-file" not in arguments:
    path = os.path.join(os.environ["FLEETRESERVE_RUN_DIR"], "ceiling-report.json")
    with open(path, "w") as handle:
        handle.write(json.dumps(document, indent=1, sort_keys=True) + "\\n")
sys.exit(int(arguments[arguments.index("--exit") + 1]) if "--exit" in arguments else 0)
"""


def no_start_check(monkeypatch) -> list:
    """The driver's start check replaced for the stand-in steps (module text); the lists it was
    called with."""
    calls = []

    def recorded(run_list_document, top):
        calls.append(run_list_document)
        return []

    monkeypatch.setattr(run_starts, "start_faults", recorded)
    return calls


def git(repo: Path, *args: str) -> str:
    done = run_git(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout


def write_matrix(root: Path, path: str, units) -> None:
    """The committed gate matrix with its gate units replaced by `units`, at `path` under
    `root`."""
    matrix = json.loads((TOP / "gate" / "matrix.json").read_text())
    matrix["gate_units"] = list(units)
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(matrix, indent=2) + "\n")


def make_repo(tmp_path: Path) -> Path:
    """The throwaway repository of the module text, with one commit."""
    root = tmp_path / "repo"
    package = root / "fleetreserve"
    package.mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    (root / ".gitignore").write_text("runs.noindex/\n__pycache__/\n")
    (root / "tracked.txt").write_text("one\n")
    (package / "__init__.py").write_text("")
    (package / "run_record.py").write_bytes(RECORDER.read_bytes())
    for name in STEP_MODULES:
        text = {SEALED_TRUTH: TRUTH_BUILD_STEP, "learn_run": LEARN_STEP}.get(name, STEP)
        (package / f"{name}.py").write_text(text)
    (package / f"{O2_MODULE}.py").write_text(O2_STEP)
    write_matrix(root, SEALED_MATRIX, sealed_route.HELD_OUT_UNITS)  # rule 3 of the run list
    for path in DEFAULT_MATRICES:  # rule 5 of the run list reads a command's default matrix
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes((TOP / path).read_bytes())
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first")
    return root


def step(module: str, *arguments: str) -> list[str]:
    """A made-up step's command: `python -m fleetreserve.<module> <arguments>`."""
    assert module in (*STEP_MODULES, O2_MODULE)
    return [sys.executable, "-m", f"fleetreserve.{module}", *arguments]


def entry(
    label: str, command: list[str], *, max_seconds=60, after=(), only=None, units=None, extra=None
) -> dict:
    """An entry of a run list of the form `run-list-2`: of both purposes unless `only` names one,
    on no group of units unless `units` names one, with the extra arguments `extra`."""
    return {
        "label": label,
        "only": only,
        "command": command,
        "units": units,
        "extra": dict(extra or {}),
        "setting": "main",
        "max_seconds": max_seconds,
        "after": list(after),
    }


O2_GROUP = "nine"  # the group of units of the two O2 steps, as in the committed list
O2_UNITS = {  # rule 10: the nine held-out units in the sealed run; development units rehearsed
    "rehearsal": {O2_GROUP: ["nrel-before-may-2019", "nrel-from-may-2019"]},
    "sealed-run": {O2_GROUP: list(sealed_route.HELD_OUT_UNITS)},
}


def o2_entries(label: str, *arguments: str) -> list[dict]:
    """Outcome O2's two steps of the run list's rule 12 (the owner's decision 79 (4)) after the
    truth builds they read: `truth_build` of the main and the held setting, then, in the sealed
    run only, the sealed truth build (`SEALED_TRUTH`) of each setting; then the step `label` of
    the main setting with `arguments`, and `<label>-held` of the held setting. Each runs
    `--every-day` at `--rho 0` on the group `O2_GROUP` (`O2_UNITS`, rule 10), reads its setting's
    truth build by `--truth-run`, and in the sealed run its sealed build by
    `--held-out-truth-run`, the held one the held setting's sealed tables (`--held-out-tables`)."""
    held_tables = str(sealed_route.held_out_tables(Path(), "held"))

    def build(name: str, module: str, setting: str, only=None) -> dict:
        given = ("--setting", "held") if setting == "held" else ()
        made = entry(name, step(module, *given), only=only)
        return {**made, "setting": setting}

    def o2(name: str, setting: str, build_label: str, sealed: list[str], given=()) -> dict:
        command = step(O2_MODULE, "--units", "{units}", "--every-day", "--rho", "0")
        command += ["--truth-run", f"{{run:{build_label}}}", *given]
        extra = {"sealed-run": sealed}
        made = entry(name, command, units=O2_GROUP, extra=extra, after=[build_label])
        return {**made, "setting": setting}

    return [
        build("dev-truth", "truth_build", "main"),
        build("dev-truth-held", "truth_build", "held"),
        build("sealed-truth", SEALED_TRUTH, "main", only="sealed-run"),
        build("sealed-truth-held", SEALED_TRUTH, "held", only="sealed-run"),
        o2(label, "main", "dev-truth", ["--held-out-truth-run", "{run:sealed-truth}"], arguments),
        o2(
            f"{label}-held",
            "held",
            "dev-truth-held",
            ["--held-out-tables", held_tables, "--held-out-truth-run", "{run:sealed-truth-held}"],
        ),
    ]


def document(entries: list[dict], *, seeds=None, units=None) -> dict:
    """A run list of the form `run-list-2` with its entries, seeds and groups of units (purpose ->
    group -> units; none by default)."""
    return {
        "format": "run-list-2",
        "about": "a made-up run list",
        "units": {"rehearsal": {}, "sealed-run": {}, **(units or {})},
        "fixed": {"seeds": dict(seeds or {}), "counts": {}, "names": {}},
        "entries": entries,
    }


def commit_list(repo: Path, listed: dict, path: str = LIST_PATH) -> Path:
    """Write the list and commit it; the file's path."""
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(listed, indent=2) + "\n")
    git(repo, "add", path)
    git(repo, "commit", "-q", "-m", f"the run list {path}")
    return file


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def drive_logs(repo: Path) -> list[Path]:
    folder = repo / "runs.noindex" / "drives"
    return sorted(folder.iterdir()) if folder.is_dir() else []


def run_folders(repo: Path) -> list[Path]:
    runs = repo / "runs.noindex"
    if not runs.is_dir():
        return []
    return sorted(path for path in runs.iterdir() if path.is_dir() and path.name != "drives")
