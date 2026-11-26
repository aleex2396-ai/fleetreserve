"""The start check of a run list: each step of the sealed run that runs a command of the sealed
route is started, by the command's own code, at a made-up tag in a made-up repository, up to its
first read of data, so an option the command refuses at its start is caught before the one run
from the tag (plan section 13, WP14: "Held-out run, once, from the tag").

Called by the run list's checker (`python -m fleetreserve.run_list <run list>`, `start_faults`)
and by the driver before its first step (`run_driver.prepare`), after the list's own rules pass.
The probe of one step is this module's command line, which only the check itself calls:
    python -m fleetreserve.run_starts <module> <made-up top folder> <argument> ...

THE MADE-UP REPOSITORY. A temporary folder holding the tree of the checked-out commit of the
working copy the commands run from (`git archive HEAD`: the committed matrices and settings a run
reads, as the run from the tag reads them), with the running package's files in place of
`src/fleetreserve/` (the code the probes run, so the gate's comparison of the running code with
the commit passes, reading T2), committed once in a new git repository and tagged `frozen-plan`.
It holds no data: no sealed file, no truth table, no run. It is removed when the check ends.

THE PROBE. One process per step, run from the made-up top folder with the environment of the
check less every `GIT_` and `FLEETRESERVE_` variable (so no run folder is written to and no gate
mark is made): the step's command line after its module, each `{run:<label>}` filled with a run
folder of that label under `runs.noindex/` that does not exist, is given to the command's own
`main(arguments, root=<made-up top folder>)`. Before the call:
- each function a command of the route calls for its first read of data (`FIRST_READS`) is
  replaced by a stop: the probe prints `REACHED DATA: <function>` and ends with `REACHED`;
- an audit hook stops it the same way at any opening or listing of a path in a data place
  (`data/` or `runs.noindex/` of the made-up folder or of the package's own top folder), so no
  data are opened even by a read `FIRST_READS` does not name (reading T3).
So everything a command checks at its start runs as at the tag: its options, its matrix read
from the made-up commit, the sealed route's decision, the folders, the gate (which passes at the
made-up tag, with a clean tree and the commit's code), its own checks of its other options after
the route (`compare_run --splits` other than 1, `corner_run`'s seed, days and methods,
`drift_run --etas` and `--calibration-days`, its step-size file at the tag), and a model's
loading and checks (one that reads other days' outcomes is refused).

Verdict per step: the stop reached, or the command's end with exit 0 before any read, passes; any
other end before the stop (an exit 1 or 2 with its printed line, a refusal of `argparse`, an
error) is a fault, with the command's last printed line. A probe that cannot decide (a package
the model needs is not installed in the environment the check runs in, or no end within
`SECONDS`) makes the check undecidable (`run_list.CannotDecide`, exit 2).

Readings (alternatives in brackets):
  T1. Only the steps of the commands of the sealed route (`run_list.ROUTE_COMMANDS`) are started:
      their starts hold the refusals that apply on the route alone, which the rehearsal never
      meets. The sealed truth build is never started (its start opens sealed files through the
      loader), nor the other commands of the list. [Alternative: start every step, with stops
      for each command's first read.]
  T2. The package that runs is the running one, laid over the commit's tree, so the check of a
      working copy with changes not committed starts the code as it is; at the tag, with a clean
      tree, the two are the same. [Alternative: the commit's code alone, which refuses every
      step of a working copy with changes.]
  T3. "The first read of data" is the first call of a function of `FIRST_READS`, or the first
      opening or listing of a path in a data place, whichever comes first. A command whose first
      read is by another function, one that only asks whether a path exists, is stopped by
      neither: it then refuses at a missing file, which is a fault here. A test pins the stops
      to each command (`tests/wp12/test_run_starts.py`). [Alternative: made-up tables and runs
      for every option, which the gate's checks of a build at the commit would refuse.]
  T4. What comes before the program in a step's command (`uv run --frozen --group learn --group
      baselines`) is not run: the probe runs in the check's own interpreter, so the check runs in
      an environment with every group the list's commands name (`uv run --frozen --group learn
      --group baselines python -m fleetreserve.run_list ...`). [Alternative: run each prefix,
      which builds an environment in the made-up folder per step.]
  T5. The check is made by the checker's command line and by the driver before its first step,
      for either purpose (`run_driver` reading D7): a drive never relies on a check made by hand
      before it, so no step of a list the check refuses starts, and no sealed file is opened.
      [Alternative: the checker's command line only, the list being frozen at the tag.]
"""

import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import fleetreserve

PACKAGE = fleetreserve.__name__
PACKAGE_PATH = Path("src") / PACKAGE
REACHED, UNDECIDED = 97, 98  # the probe's own exit statuses
REACHED_LINE, UNDECIDED_LINE = "REACHED DATA: ", "CANNOT DECIDE: "
SECONDS = 600.0  # the longest a probe may take
# The functions a command of the route calls for its first read of data (reading T3), by module
# of the package: the common ones first, then a command's own.
COMMON_READS = (
    ("sealed_route", "check_split_rule"),
    ("sealed_route", "read_two_builds"),
    ("cold_start_gate", "read_two_builds"),
    ("gate_run", "check_inputs"),
    ("gate_run", "read_inputs"),
)
OWN_READS = {
    "ledger_run": (("ledger_run", "check_run_files"), ("ledger_run", "read_run")),
    "o3_run": (("o3_run", "read_route_units"), ("o3_run", "read_units")),
}
DATA_PLACES = ("data", "runs.noindex")
_EVENTS = ("open", "os.listdir", "os.scandir")
_COMMITTER = ("-c", "user.name=start-check", "-c", "user.email=start-check@example.invalid")


def first_reads(module: str) -> tuple[tuple[str, str], ...]:
    """The (module, function) stops of the command `module` (reading T3)."""
    return (*COMMON_READS, *OWN_READS.get(module, ()))


# --- the probe, in its own process ---------------------------------------------------------------


def _reached(what: str) -> None:
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    os.write(1, f"\n{REACHED_LINE}{what}\n".encode())
    os._exit(REACHED)


def _guard(places: Sequence[str]):
    def hook(event, args):
        if event not in _EVENTS or not args:
            return
        path = args[0]
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not isinstance(path, str | os.PathLike):
            return
        full = os.path.abspath(os.fspath(path))
        for place in places:
            if full == place or full.startswith(place + os.sep):
                _reached(f"{event} of {full}")

    return hook


def probe(module: str, root: Path, arguments: Sequence[str]) -> int:
    """Start the command `module` on `arguments` at the made-up top folder `root` up to its first
    read of data (module text, THE PROBE): `REACHED`, the command's own exit status, or
    `UNDECIDED`."""
    from fleetreserve.sessions import TOP

    tops = (Path(root), Path(TOP))
    places = sorted({os.path.abspath(top / place) for top in tops for place in DATA_PLACES})
    places += [os.path.realpath(p) for p in places if os.path.realpath(p) not in places]
    sys.addaudithook(_guard(places))
    for owner, name in first_reads(module):
        found = importlib.import_module(f"{PACKAGE}.{owner}")

        def stop(*args, _what=f"{owner}.{name}", **kwargs):
            _reached(_what)

        setattr(found, name, stop)
    try:
        command = importlib.import_module(f"{PACKAGE}.{module}")
        status = command.main(list(arguments), root=Path(root))
    except SystemExit as ended:
        status = ended.code if isinstance(ended.code, int) else 2
    except ModuleNotFoundError as problem:
        if problem.name and not problem.name.startswith(PACKAGE):
            print(f"{UNDECIDED_LINE}the package {problem.name} is not installed (reading T4)")
            return UNDECIDED
        raise
    sys.stdout.flush()
    return int(status or 0)


# --- the check -----------------------------------------------------------------------------------


def _git(folder: Path, *args: str, data: bool = False) -> subprocess.CompletedProcess:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    return subprocess.run(
        ["git", "-C", str(folder), *args],
        capture_output=True,
        text=not data,
        env=environment,
        check=False,
    )


def made_up_repository(top: Path, folder: Path) -> Path:
    """The made-up repository of the module text in `folder`, from the working copy `top`; its top
    folder. Raises `run_list.CannotDecide`."""
    from fleetreserve.run_list import CannotDecide
    from fleetreserve.sealed_route import TAG

    archive = _git(Path(top), "archive", "--format=tar", "HEAD", data=True)
    if archive.returncode != 0:
        raise CannotDecide(
            f"the checked-out commit of {top} cannot be archived: {archive.stderr.decode().strip()}"
        )
    root = Path(folder) / "made-up"
    root.mkdir()
    with tarfile.open(fileobj=BytesIO(archive.stdout)) as tree:
        tree.extractall(root, filter="data")
    package = root / PACKAGE_PATH
    shutil.rmtree(package, ignore_errors=True)
    shutil.copytree(
        Path(fleetreserve.__file__).parent, package, ignore=shutil.ignore_patterns("__pycache__")
    )
    steps = (
        ("init", "-q"),
        ("add", "-A", "-f", "--", str(PACKAGE_PATH)),
        ("add", "-A"),
        (*_COMMITTER, "commit", "-q", "--no-verify", "--no-gpg-sign", "-m", "made-up"),
        ("tag", TAG),
    )
    for step in steps:
        done = _git(root, *step)
        if done.returncode != 0:
            raise CannotDecide(
                f"the made-up repository cannot be made: git {step[0]}: {done.stderr.strip()}"
            )
    if _git(root, "status", "--porcelain").stdout.strip():
        raise CannotDecide("the made-up repository is not clean after its commit")
    return root


def _arguments(step) -> list[str]:
    """The step's arguments after its module, each `{run:<label>}` a run folder that does not
    exist."""
    from fleetreserve import run_list
    from fleetreserve.run_record import RUNS_FOLDER, run_folder_name, utc_now

    runs = {label: f"{RUNS_FOLDER}/{run_folder_name(utc_now(), label)}" for _, label in step.runs}
    command = run_list.fill_runs(step, runs)
    return command[run_list._module_at(command) + 1 :]


def _last_line(done: subprocess.CompletedProcess) -> str:
    for text in (done.stdout, done.stderr):
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return "nothing printed"


def start_fault(step, root: Path) -> str | None:
    """The fault of the step's start at the made-up repository `root`, or None (module text).
    Raises `run_list.CannotDecide`."""
    from fleetreserve.run_list import CannotDecide, _where, step_module

    module = step_module(step)
    environment = {
        k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "FLEETRESERVE_"))
    }
    try:
        done = subprocess.run(
            [sys.executable, "-m", f"{PACKAGE}.run_starts", module, str(root), *_arguments(step)],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as problem:
        raise CannotDecide(
            f"{_where(step)}: its start did not reach its first read of data in {SECONDS:g} s"
        ) from problem
    if done.returncode == UNDECIDED:
        raise CannotDecide(f"{_where(step)}: its start cannot be decided: {_last_line(done)}")
    if done.returncode in (REACHED, 0):
        return None
    return (
        f"{_where(step)}: started at a made-up tag, {module} stops before its first read of data "
        f"with exit {done.returncode}: {_last_line(done)} (the start check)"
    )


def start_faults(run_list_document, top: Path | None) -> list[str]:
    """The start check of every step of the sealed run that runs a command of the sealed route
    (reading T1), one line per fault; empty when none. `top` is the working copy the commands
    run from. Raises `run_list.CannotDecide`."""
    from fleetreserve import run_list

    steps = run_list.route_steps(run_list_document)
    if not steps:
        return []
    if top is None:
        raise run_list.CannotDecide(
            "the start check needs the git working copy the commands run from, and the list is "
            "in none"
        )
    with tempfile.TemporaryDirectory(prefix="run-starts-") as folder:
        root = made_up_repository(top, Path(folder)).resolve()
        found = [start_fault(s, root) for s in steps]
    return [line for line in found if line is not None]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2:
        print(f"{UNDECIDED_LINE}a probe takes <module> <made-up top folder> <argument> ...")
        return UNDECIDED
    module, root, *rest = arguments
    return probe(module, Path(root), rest)


if __name__ == "__main__":
    sys.exit(main())
