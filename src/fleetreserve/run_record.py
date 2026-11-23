"""Record one run of a command: a run folder with its log, its results and a manifest.

Command line:
    python -m fleetreserve.run_record --name <label> [--long] [--max-seconds <n>]
        -- <command> [arguments...]

The run starts only from a clean git tree. The run folder is
`runs.noindex/<UTC start>-<label>/` under the git top folder. The command runs from the top
folder with `FLEETRESERVE_RUN_DIR` set to the run folder, so it writes its results files there.
Its output goes to `log.txt`. When the command ends, `manifest.json` records the commit, the
command, the start and end, the exit code and the SHA-256 of every file in the run folder.

`--long` marks a heavy run: one at a time (an exclusive lock on `runs.noindex/heavy.lock`),
kept awake by `caffeinate -i`. A heavy run must also have a time limit.

`--max-seconds <n>` (a positive, finite number) is a wall-clock limit on the whole run. When it
is reached while the command is still running, the recorder sends SIGTERM to the command's
whole process group and, if any process of the group is still alive after a grace period (30
seconds; a parameter of `record_run`, not settable from the command line or environment),
SIGKILL to the group. The run is still recorded: the manifest holds the command's real exit code,
`max_seconds`, and `timed_out: true`, and the verifier never accepts a timed-out run's results.

While the command runs, SIGINT, SIGTERM and SIGHUP sent to the recorder are passed on to the
command's whole process group (a second one kills the group). The recorder waits until the
group has ended, writes the manifest with an `interrupted` field naming the signal, and only then
lets go of the heavy-run lock. This also holds while a timed-out command is being stopped.

Exit status: the command's own exit code when it ran, 2 when the run was refused.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

MANIFEST_NAME = "manifest.json"
LOG_NAME = "log.txt"
RUNS_FOLDER = "runs.noindex"
LOCK_NAME = "heavy.lock"
RUN_DIR_VARIABLE = "FLEETRESERVE_RUN_DIR"
DEFAULT_PREFIX = ("caffeinate", "-i")
REFUSED = 2
NOT_STARTED = 127
MAX_LISTED = 20
DEFAULT_GRACE_SECONDS = 30.0
POLL_SECONDS = 0.05
RUN_STAMP = "%Y%m%dT%H%M%SZ"  # the UTC start at the front of a run folder's name

_LABEL = re.compile(r"[A-Za-z0-9_-]+")


class Refusal(Exception):
    """The run must not start; the message says why."""


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_time(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(folder: Path, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(folder), *args], capture_output=True, text=True, check=False
        )
    except OSError as problem:
        raise Refusal(f"git could not be started: {problem}") from problem
    if done.returncode != 0:
        raise Refusal(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _check_clean(top: Path) -> None:
    changed = _git(top, "status", "--porcelain").splitlines()
    if changed:
        shown = "\n".join(f"  {line}" for line in changed[:MAX_LISTED])
        more = f"\n  ... and {len(changed) - MAX_LISTED} more" if len(changed) > MAX_LISTED else ""
        raise Refusal(f"the git tree has uncommitted changes:\n{shown}{more}")


@contextmanager
def _heavy_lock(runs: Path) -> Iterator[None]:
    runs.mkdir(exist_ok=True)
    with (runs / LOCK_NAME).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as problem:
            raise Refusal("a heavy run is already going") from problem
        yield


def run_folder_name(start: datetime, label: str) -> str:
    """The name of the run folder of a run that starts at `start`: `<UTC start>-<label>`."""
    return f"{start.strftime(RUN_STAMP)}-{label}"


def _make_run_folder(runs: Path, label: str) -> tuple[Path, datetime]:
    start = utc_now()
    runs.mkdir(exist_ok=True)
    folder = runs / run_folder_name(start, label)
    try:
        folder.mkdir()
    except FileExistsError as problem:
        raise Refusal(f"the run folder {folder} already exists") from problem
    return folder, start


def _hash_outputs(folder: Path) -> dict[str, str]:
    outputs = {}
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder).as_posix()
        if path.is_file() and relative != MANIFEST_NAME:
            outputs[relative] = sha256_file(path)
    return outputs


def _write_manifest(folder: Path, manifest: dict) -> None:
    temporary = folder / f".{MANIFEST_NAME}.tmp"
    with temporary.open("w") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, folder / MANIFEST_NAME)


class _Forwarder:
    """Passes SIGINT, SIGTERM and SIGHUP on to the command's process group.

    The command runs in its own session, so a signal sent to the recorder (or to its terminal's
    process group) would not reach it by itself. The first signal is passed on as it is; a
    second one kills the whole group.
    """

    SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def __init__(self) -> None:
        self.group: int | None = None
        self.received: list[int] = []
        self.finished = False
        self.timed_out = False

    def handle(self, number: int, frame: object) -> None:
        if self.finished:
            return
        self.received.append(number)
        if self.group is not None:
            self._send(number if len(self.received) == 1 else signal.SIGKILL)

    def start(self, group: int) -> None:
        """The command is running; pass on any signal that came before it started."""
        self.group = group
        for index, number in enumerate(self.received):
            self._send(signal.SIGKILL if index else number)

    def group_alive(self) -> bool:
        if self.group is None:
            return False
        try:
            os.killpg(self.group, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def wait_for_group(self) -> None:
        """After a signal: wait until no process of the command's group is left."""
        while self.group_alive():
            time.sleep(POLL_SECONDS)

    def stop_after_limit(self, process: subprocess.Popen, grace_seconds: float) -> None:
        """The time limit is reached: SIGTERM to the group, SIGKILL if it outlasts the grace.

        Returns when no process of the group is left. The leader is reaped on the way, because
        an unreaped process still counts as alive.
        """
        self.timed_out = True
        self._send(signal.SIGTERM)
        grace_ends = time.monotonic() + grace_seconds
        killed = False
        while True:
            process.poll()
            if not self.group_alive():
                return
            if not killed and time.monotonic() >= grace_ends:
                self._send(signal.SIGKILL)
                killed = True
            time.sleep(POLL_SECONDS)

    def _send(self, number: int) -> None:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(self.group, number)

    @property
    def interrupted(self) -> str | None:
        return signal.Signals(self.received[0]).name if self.received else None


def _command_exists(top: Path, name: str) -> bool:
    if os.sep in name:
        path = top / name
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(name) is not None


def _run_command(
    top: Path,
    folder: Path,
    argv: list[str],
    command: list[str],
    forwarder: _Forwarder,
    max_seconds: float | None,
    grace_seconds: float,
):
    """Run the command; return its exit code and an error text if it could not be started."""
    if argv != command and not _command_exists(top, command[0]):
        # With a prefix the failure would happen inside the prefix program, with no clear error.
        return NOT_STARTED, f"could not start {command[0]!r}: not found or not executable"
    environment = {**os.environ, RUN_DIR_VARIABLE: str(folder)}
    with (folder / LOG_NAME).open("wb") as log:
        try:
            process = subprocess.Popen(
                argv,
                cwd=top,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except (OSError, ValueError) as problem:
            return NOT_STARTED, f"could not start {argv[0]!r}: {problem}"
        forwarder.start(process.pid)
        deadline = None if max_seconds is None else time.monotonic() + max_seconds
        while True:
            slice_seconds = 0.5
            if deadline is not None:
                slice_seconds = min(slice_seconds, max(deadline - time.monotonic(), 0.0))
            try:
                code = process.wait(timeout=slice_seconds)
                break
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() >= deadline:
                    if process.poll() is not None:  # it ended just as the limit was reached
                        code = process.returncode
                        break
                    forwarder.stop_after_limit(process, grace_seconds)
                    code = process.wait()
                    break
        if forwarder.received:
            forwarder.wait_for_group()
    return (code if code >= 0 else 128 - code), None


def _execute(
    top: Path,
    folder: Path,
    start: datetime,
    commit: str,
    command: list[str],
    argv: list[str],
    *,
    long: bool,
    max_seconds: float | None,
    grace_seconds: float,
) -> int:
    forwarder = _Forwarder()
    previous = {}
    try:
        for number in forwarder.SIGNALS:
            previous[number] = signal.signal(number, forwarder.handle)
    except ValueError:  # not the main thread: signals cannot be caught here
        pass
    try:
        exit_code, error = _run_command(
            top, folder, argv, command, forwarder, max_seconds, grace_seconds
        )
        forwarder.finished = True
        end = utc_now()
        manifest = {
            "commit": commit,
            "command": command,
            "start": format_time(start),
            "end": format_time(end),
            "exit_code": exit_code,
            "outputs": _hash_outputs(folder),
            "long": long,
            "max_seconds": max_seconds,
            "timed_out": forwarder.timed_out,
        }
        if error is not None:
            manifest["error"] = error
        if forwarder.interrupted is not None:
            manifest["interrupted"] = forwarder.interrupted
        _write_manifest(folder, manifest)
        print(folder)
        print(f"exit code {exit_code}")
        if forwarder.timed_out:
            print(f"the run hit its time limit of {max_seconds:g} seconds and was stopped")
        return exit_code
    finally:
        forwarder.finished = True
        for number, handler in previous.items():
            signal.signal(number, handler)


def _valid_limit(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def record_run(
    name: str,
    command: Sequence[str],
    *,
    long: bool = False,
    max_seconds: float | None = None,
    prefix: Sequence[str] = DEFAULT_PREFIX,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    start_dir: Path | None = None,
) -> int:
    """Run `command` as described in the module text; return the tool's exit status.

    `prefix` is put in front of the command for a long run, and `grace_seconds` is the time
    between SIGTERM and SIGKILL when the time limit is reached. Both are parameters for tests
    only; the command line cannot change them.
    """
    command = list(command)
    try:
        if not _LABEL.fullmatch(name):
            raise Refusal("the label may hold only letters, digits, '-' and '_'")
        if not command:
            raise Refusal("no command was given")
        if max_seconds is not None and not _valid_limit(max_seconds):
            raise Refusal("the time limit must be a positive, finite number of seconds")
        if long and max_seconds is None:
            raise Refusal("a long run needs a time limit (--max-seconds)")
        if long and prefix and shutil.which(prefix[0]) is None:
            raise Refusal(f"{prefix[0]} was not found on the search path")
        top = Path(_git(Path(start_dir or Path.cwd()), "rev-parse", "--show-toplevel").strip())
        _check_clean(top)
        commit = _git(top, "rev-parse", "HEAD").strip()
        with ExitStack() as stack:
            if long:
                stack.enter_context(_heavy_lock(top / RUNS_FOLDER))
            folder, start = _make_run_folder(top / RUNS_FOLDER, name)
            argv = [*prefix, *command] if long else command
            return _execute(
                top,
                folder,
                start,
                commit,
                command,
                argv,
                long=long,
                max_seconds=max_seconds,
                grace_seconds=grace_seconds,
            )
    except Refusal as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return REFUSED


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    split = arguments.index("--") if "--" in arguments else len(arguments)
    parser = argparse.ArgumentParser(
        prog="python -m fleetreserve.run_record",
        usage="%(prog)s --name LABEL [--long] [--max-seconds N] -- COMMAND [ARGUMENTS...]",
        description="Run a command from a clean git tree and record the run.",
        allow_abbrev=False,
    )
    parser.add_argument("--name", required=True, help="letters, digits, '-' and '_'")
    parser.add_argument("--long", action="store_true", help="heavy run: one at a time, awake")
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="time limit for the whole run, in seconds; required with --long",
    )
    options = parser.parse_args(arguments[:split])
    command = arguments[split + 1 :]
    if not command:
        parser.error("a command is needed after '--'")
    return record_run(options.name, command, long=options.long, max_seconds=options.max_seconds)


if __name__ == "__main__":
    sys.exit(main())
