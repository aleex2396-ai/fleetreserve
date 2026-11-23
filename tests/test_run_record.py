"""The run recorder refuses unsafe runs and writes a manifest for every run it starts.

Every test works in a throwaway git repository inside pytest's temporary folder, with git's own
configuration switched off, so nothing depends on the machine's git settings.
"""

import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import pytest
from git_run import run_git

from fleetreserve import run_record
from fleetreserve.run_record import record_run
from fleetreserve.run_verify import main as verify_main

TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
TIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
MANIFEST_KEYS = {
    "commit",
    "command",
    "start",
    "end",
    "exit_code",
    "outputs",
    "long",
    "max_seconds",
    "timed_out",
}


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    """Drop every GIT_* variable and git's global and system settings for the test."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(repo: Path, *args: str) -> str:
    done = run_git(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with one commit that ignores `runs.noindex/`, as the project does."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    (root / ".gitignore").write_text("runs.noindex/\n")
    (root / "tracked.txt").write_text("one\n")
    git(root, "add", ".gitignore", "tracked.txt")
    git(root, "commit", "-q", "-m", "first")
    return root


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def run_folders(repo: Path) -> list[Path]:
    runs = repo / "runs.noindex"
    if not runs.is_dir():
        return []
    return sorted(path for path in runs.iterdir() if path.is_dir())


def read_manifest(folder: Path) -> dict:
    return json.loads((folder / "manifest.json").read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


WRITE_RESULTS = """
import os, sys
folder = os.environ["FLEETRESERVE_RUN_DIR"]
os.makedirs(os.path.join(folder, "sub"))
open(os.path.join(folder, "result.txt"), "w").write("42\\n")
open(os.path.join(folder, "sub", "deep.txt"), "w").write("deep\\n")
print("OUT-LINE-STANDARD")
print("ERR-LINE-ERROR", file=sys.stderr)
"""


def test_clean_run_records_every_field(repo, capsys):
    status = record_run("first", py(WRITE_RESULTS), start_dir=repo)
    assert status == 0
    (folder,) = run_folders(repo)
    assert re.fullmatch(r"\d{8}T\d{6}Z-first", folder.name)
    manifest = read_manifest(folder)
    assert set(manifest) == MANIFEST_KEYS
    assert manifest["commit"] == git(repo, "rev-parse", "HEAD").strip()
    assert manifest["command"] == py(WRITE_RESULTS)
    assert manifest["exit_code"] == 0
    assert manifest["long"] is False
    assert manifest["max_seconds"] is None
    assert manifest["timed_out"] is False
    assert TIME_PATTERN.fullmatch(manifest["start"])
    assert TIME_PATTERN.fullmatch(manifest["end"])
    start = datetime.strptime(manifest["start"], TIME_FORMAT).replace(tzinfo=UTC)
    end = datetime.strptime(manifest["end"], TIME_FORMAT).replace(tzinfo=UTC)
    assert start <= end
    assert folder.name.startswith(start.strftime("%Y%m%dT%H%M%SZ"))
    assert manifest["outputs"] == {
        "log.txt": sha256(folder / "log.txt"),
        "result.txt": hashlib.sha256(b"42\n").hexdigest(),
        "sub/deep.txt": hashlib.sha256(b"deep\n").hexdigest(),
    }
    log = (folder / "log.txt").read_text()
    assert "OUT-LINE-STANDARD" in log
    assert "ERR-LINE-ERROR" in log
    shown = capsys.readouterr().out
    assert str(folder) in shown
    assert "exit code 0" in shown
    assert "OUT-LINE-STANDARD" not in shown


def test_manifest_is_written_atomically_and_leaves_no_temporary_file(repo):
    record_run("first", py("pass"), start_dir=repo)
    (folder,) = run_folders(repo)
    assert sorted(path.name for path in folder.iterdir()) == ["log.txt", "manifest.json"]


def test_failing_command_still_gets_a_manifest(repo):
    code = WRITE_RESULTS + "\nsys.exit(3)\n"
    status = record_run("fails", py(code), start_dir=repo)
    assert status == 3
    (folder,) = run_folders(repo)
    manifest = read_manifest(folder)
    assert manifest["exit_code"] == 3
    assert "result.txt" in manifest["outputs"]


def test_command_killed_by_a_signal_records_the_shell_style_code(repo):
    code = "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"
    status = record_run("killed", py(code), start_dir=repo)
    assert status == 128 + signal.SIGTERM
    (folder,) = run_folders(repo)
    assert read_manifest(folder)["exit_code"] == 128 + signal.SIGTERM


def test_command_that_cannot_start_records_127_and_an_error(repo):
    status = record_run("missing", ["no_such_program_for_this_test"], start_dir=repo)
    assert status == 127
    (folder,) = run_folders(repo)
    manifest = read_manifest(folder)
    assert manifest["exit_code"] == 127
    assert "no_such_program_for_this_test" in manifest["error"]
    assert manifest["outputs"] == {"log.txt": sha256(folder / "log.txt")}


def test_two_clean_runs_in_a_row_are_allowed(repo):
    # runs.noindex is ignored by git, so the first run does not make the tree dirty.
    assert record_run("a", py("pass"), start_dir=repo) == 0
    assert record_run("b", py("pass"), start_dir=repo) == 0
    assert len(run_folders(repo)) == 2


def modify_tracked(repo: Path) -> None:
    (repo / "tracked.txt").write_text("changed\n")


def stage_change(repo: Path) -> None:
    (repo / "tracked.txt").write_text("staged\n")
    git(repo, "add", "tracked.txt")


def add_untracked(repo: Path) -> None:
    (repo / "new_file.txt").write_text("new\n")


def delete_tracked(repo: Path) -> None:
    (repo / "tracked.txt").unlink()


@pytest.mark.parametrize("dirty", [modify_tracked, stage_change, add_untracked, delete_tracked])
def test_dirty_tree_is_refused_without_running_or_creating_a_folder(repo, capsys, dirty):
    dirty(repo)
    marker = repo / "ran.txt"
    status = record_run("dirty", py(f"open({str(marker)!r}, 'w').close()"), start_dir=repo)
    assert status == 2
    assert not marker.exists()
    assert run_folders(repo) == []
    assert "uncommitted" in capsys.readouterr().err


@pytest.mark.parametrize("label", ["", "a b", "a/b", "../x", "a.b", "café", "a\n"])
def test_bad_label_is_refused(repo, label):
    marker = repo / "ran.txt"
    status = record_run(label, py(f"open({str(marker)!r}, 'w').close()"), start_dir=repo)
    assert status == 2
    assert not marker.exists()
    assert run_folders(repo) == []


@pytest.mark.parametrize("label", ["a", "Run_1-b", "7"])
def test_good_labels_are_accepted(repo, label):
    assert record_run(label, py("pass"), start_dir=repo) == 0


def test_existing_run_folder_is_never_reused(repo, monkeypatch):
    monkeypatch.setattr(run_record, "utc_now", lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
    assert record_run("same", py("pass"), start_dir=repo) == 0
    (folder,) = run_folders(repo)
    assert folder.name == "20260102T030405Z-same"
    before = (folder / "manifest.json").read_bytes()
    assert record_run("same", py("pass"), start_dir=repo) == 2
    assert run_folders(repo) == [folder]
    assert (folder / "manifest.json").read_bytes() == before


def test_empty_command_is_refused(repo):
    assert record_run("empty", [], start_dir=repo) == 2
    assert run_folders(repo) == []


def test_folder_outside_git_is_refused(tmp_path):
    outside = tmp_path / "plain"
    outside.mkdir()
    assert record_run("x", py("pass"), start_dir=outside) == 2
    assert not (outside / "runs.noindex").exists()


def test_run_dir_variable_and_working_folder_reach_the_command(repo):
    inner = repo / "inner"
    inner.mkdir()
    code = (
        "import os\n"
        "folder = os.environ['FLEETRESERVE_RUN_DIR']\n"
        "open(os.path.join(folder, 'env.txt'), 'w').write(folder)\n"
        "open(os.path.join(folder, 'cwd.txt'), 'w').write(os.getcwd())\n"
    )
    assert record_run("env", py(code), start_dir=inner) == 0
    (folder,) = run_folders(repo)
    assert Path((folder / "env.txt").read_text()) == folder
    assert Path((folder / "cwd.txt").read_text()).resolve() == repo.resolve()


# Prefix that writes a marker, then runs the rest of the command line, like `caffeinate -i`.
MARKING_PREFIX = [
    sys.executable,
    "-c",
    "import os, sys; "
    "open(os.path.join(os.environ['FLEETRESERVE_RUN_DIR'], 'prefix-ran'), 'w').close(); "
    "os.execvp(sys.argv[1], sys.argv[1:])",
]


def test_long_run_uses_the_prefix_and_records_only_the_command(repo):
    command = py(WRITE_RESULTS)
    status = record_run(
        "heavy", command, long=True, max_seconds=60, prefix=MARKING_PREFIX, start_dir=repo
    )
    assert status == 0
    (folder,) = run_folders(repo)
    manifest = read_manifest(folder)
    assert manifest["long"] is True
    assert manifest["command"] == command
    assert "prefix-ran" in manifest["outputs"]
    assert "result.txt" in manifest["outputs"]


def test_short_run_does_not_use_the_prefix(repo):
    assert record_run("light", py("pass"), prefix=MARKING_PREFIX, start_dir=repo) == 0
    (folder,) = run_folders(repo)
    assert "prefix-ran" not in read_manifest(folder)["outputs"]


def test_long_run_without_the_prefix_program_is_refused(repo):
    marker = repo / "ran.txt"
    status = record_run(
        "heavy",
        py(f"open({str(marker)!r}, 'w').close()"),
        long=True,
        max_seconds=60,
        prefix=["no_such_prefix_program_for_this_test", "-i"],
        start_dir=repo,
    )
    assert status == 2
    assert not marker.exists()
    assert run_folders(repo) == []


def test_long_run_with_the_real_caffeinate_or_refusal_when_it_is_missing(repo):
    status = record_run("heavy", py(WRITE_RESULTS), long=True, max_seconds=60, start_dir=repo)
    if shutil.which("caffeinate"):
        assert status == 0
        (folder,) = run_folders(repo)
        manifest = read_manifest(folder)
        assert manifest["long"] is True
        assert manifest["exit_code"] == 0
        assert "result.txt" in manifest["outputs"]
    else:
        assert status == 2
        assert run_folders(repo) == []


def hold_lock(repo: Path):
    runs = repo / "runs.noindex"
    runs.mkdir(exist_ok=True)
    held = (runs / "heavy.lock").open("a")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return held


def test_second_heavy_run_is_refused_while_the_lock_is_held(repo, capsys):
    marker = repo / "ran.txt"
    with hold_lock(repo):
        status = record_run(
            "heavy",
            py(f"open({str(marker)!r}, 'w').close()"),
            long=True,
            max_seconds=60,
            prefix=(),
            start_dir=repo,
        )
        assert status == 2
        assert not marker.exists()
        assert run_folders(repo) == []
        assert "heavy run is already going" in capsys.readouterr().err
        # A run that is not marked long is not held back by the lock.
        assert record_run("light", py("pass"), start_dir=repo) == 0
    # Once the lock is let go, a heavy run goes ahead.
    status = record_run("heavy", py("pass"), long=True, max_seconds=60, prefix=(), start_dir=repo)
    assert status == 0


def test_lock_is_held_for_the_whole_run_and_released_afterwards(repo):
    code = (
        "import fcntl, os\n"
        "folder = os.environ['FLEETRESERVE_RUN_DIR']\n"
        "lock = open(os.path.join(os.path.dirname(folder), 'heavy.lock'), 'a')\n"
        "try:\n"
        "    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    state = 'free'\n"
        "except BlockingIOError:\n"
        "    state = 'held'\n"
        "open(os.path.join(folder, 'lock-state.txt'), 'w').write(state)\n"
    )
    status = record_run("heavy", py(code), long=True, max_seconds=60, prefix=(), start_dir=repo)
    assert status == 0
    (folder,) = run_folders(repo)
    assert (folder / "lock-state.txt").read_text() == "held"
    with hold_lock(repo):  # taking the lock succeeds, so the run released it
        pass


def cli(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fleetreserve.run_record", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_command_line_returns_the_commands_exit_code_and_prints_little(repo):
    noisy = "print('X' * 5000); import sys; sys.exit(5)"
    done = cli(repo, "--name", "cli", "--", sys.executable, "-c", noisy)
    assert done.returncode == 5
    (folder,) = run_folders(repo)
    lines = done.stdout.splitlines()
    assert lines == [str(folder), "exit code 5"]
    assert read_manifest(folder)["exit_code"] == 5


def test_command_line_long_flag_is_the_recorders_own(repo):
    done = cli(
        repo, "--name", "long", "--long", "--max-seconds", "60", "--", sys.executable, "-c", "pass"
    )
    # With the real caffeinate present the run goes ahead; without it the run is refused.
    assert done.returncode == (0 if shutil.which("caffeinate") else 2)


def test_command_line_passes_dashes_after_the_separator_to_the_command(repo):
    done = cli(repo, "--name", "dash", "--", sys.executable, "-c", "pass", "--long", "--name")
    assert done.returncode == 0
    (folder,) = run_folders(repo)
    manifest = read_manifest(folder)
    assert manifest["command"][-2:] == ["--long", "--name"]
    assert manifest["long"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--name", "x"],
        ["--name", "x", "--"],
        ["--", "true"],
        ["--name", "x", "true"],
        ["--name", "bad label", "--", "true"],
        ["--name", "x", "--bogus", "--", "true"],
    ],
)
def test_command_line_misuse_exits_2_and_creates_nothing(repo, arguments):
    done = cli(repo, *arguments)
    assert done.returncode == 2
    assert run_folders(repo) == []


def test_command_line_has_no_switch_to_skip_the_dirty_tree_refusal(repo):
    add_untracked(repo)
    for extra in (["--force"], ["--allow-dirty"], ["--no-check"]):
        done = cli(repo, "--name", "x", *extra, "--", "true")
        assert done.returncode == 2
    done = cli(repo, "--name", "x", "--", "true")
    assert done.returncode == 2
    assert run_folders(repo) == []


# A prefix that exits straight into the command, so the command itself is the group leader.
EXEC_PREFIX = [sys.executable, "-c", "import os, sys; os.execvp(sys.argv[1], sys.argv[1:])"]


@pytest.mark.parametrize("command", ["no_such_program_for_this_test", "./no_such_program"])
def test_long_run_of_a_missing_command_records_127_and_an_error(repo, command):
    status = record_run(
        "missing", [command], long=True, max_seconds=60, prefix=EXEC_PREFIX, start_dir=repo
    )
    assert status == 127
    (folder,) = run_folders(repo)
    manifest = read_manifest(folder)
    assert manifest["exit_code"] == 127
    assert command in manifest["error"]
    assert manifest["long"] is True


def wait_for(condition, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def pid_file(repo: Path) -> Path | None:
    for folder in run_folders(repo):
        if (folder / "pid.txt").is_file() and (folder / "pid.txt").read_text():
            return folder / "pid.txt"
    return None


# Writes its process number, then sleeps; the default reaction to SIGINT is restored because a
# caller that started the test in the background may have set it to "ignore".
SLEEPER = """
import os, signal, time
signal.signal(signal.SIGINT, signal.default_int_handler)
folder = os.environ['FLEETRESERVE_RUN_DIR']
open(os.path.join(folder, 'pid.txt'), 'w').write(str(os.getpid()))
time.sleep(30)
"""


def stop_leftovers(recorder: subprocess.Popen, pid: int | None) -> None:
    if recorder.poll() is None:
        recorder.kill()
        recorder.wait()
    for stream in (recorder.stdout, recorder.stderr):
        stream.close()
    if pid is not None and alive(pid):
        os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize(
    ("sent", "code"),
    [(signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)],
    ids=["SIGTERM", "SIGINT", "SIGHUP"],
)
def test_signal_to_the_recorder_ends_the_command_and_still_leaves_a_manifest(repo, sent, code):
    recorder = subprocess.Popen(
        [sys.executable, "-m", "fleetreserve.run_record", "--name", "stop", "--", *py(SLEEPER)],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid = None
    try:
        assert wait_for(lambda: pid_file(repo) is not None), "the command did not start"
        pid = int(pid_file(repo).read_text())
        recorder.send_signal(sent)
        _, errors = recorder.communicate(timeout=20)
    finally:
        stop_leftovers(recorder, pid)
    assert recorder.returncode == code
    assert "Traceback" not in errors
    assert wait_for(lambda: not alive(pid), timeout=5), "the command is still running"
    (folder,) = run_folders(repo)
    manifest = read_manifest(folder)
    assert manifest["exit_code"] == code
    assert manifest["interrupted"] == sent.name
    assert "log.txt" in manifest["outputs"]
    assert verify_main([str(folder / "log.txt")]) == 1


# Ends slowly when asked to stop, so a test can look at the lock while the command winds down.
SLOW_TO_STOP = """
import os, signal, sys, time
folder = os.environ['FLEETRESERVE_RUN_DIR']
def stop(number, frame):
    open(os.path.join(folder, 'stopping.txt'), 'w').close()
    time.sleep(3)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
open(os.path.join(folder, 'pid.txt'), 'w').write(str(os.getpid()))
time.sleep(30)
"""

# Starts a heavy run with a wrapper that stays alive beside the command, as `caffeinate -i` does.
HEAVY_DRIVER = """
import sys
from pathlib import Path
from fleetreserve.run_record import record_run
run_it = "import subprocess, sys; sys.exit(subprocess.run(sys.argv[1:]).returncode)"
wrapper = [sys.executable, "-c", run_it]
repo = Path(sys.argv[1])
sys.exit(
    record_run(
        "first", sys.argv[2:], long=True, max_seconds=600, prefix=wrapper, start_dir=repo
    )
)
"""


def test_heavy_lock_is_kept_until_the_command_has_really_ended(repo, capsys):
    recorder = subprocess.Popen(
        [sys.executable, "-c", HEAVY_DRIVER, str(repo), *py(SLOW_TO_STOP)],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid = None
    try:
        assert wait_for(lambda: pid_file(repo) is not None), "the command did not start"
        pid = int(pid_file(repo).read_text())
        recorder.send_signal(signal.SIGTERM)
        (first,) = run_folders(repo)
        assert wait_for(lambda: (first / "stopping.txt").exists()), "the command did not stop"
        # The wrapper is gone by now, the command is still winding down: the lock must hold.
        assert alive(pid)
        capsys.readouterr()
        second = record_run(
            "second", py("pass"), long=True, max_seconds=60, prefix=(), start_dir=repo
        )
        assert second == 2
        assert "heavy run is already going" in capsys.readouterr().err
        assert run_folders(repo) == [first]
        _, errors = recorder.communicate(timeout=30)
    finally:
        stop_leftovers(recorder, pid)
    assert recorder.returncode != 0
    assert "Traceback" not in errors
    assert not alive(pid)
    manifest = read_manifest(first)
    assert manifest["exit_code"] != 0
    assert manifest["interrupted"] == "SIGTERM"
    # Now that the first run has fully ended, a heavy run is allowed again.
    third = record_run("third", py("pass"), long=True, max_seconds=60, prefix=(), start_dir=repo)
    assert third == 0


# ---- the run-level time limit ------------------------------------------------------------------
#
# The limit is a wall-clock limit on the whole run. These tests start real commands that would
# sleep for 30 seconds and give them a limit of a couple of seconds. None of them asserts how
# long anything took; a hang shows up as a bounded wait that fails.


def read_number(repo: Path, name: str) -> int | None:
    """The number written to `name` inside the (only) run folder, or None if not there yet."""
    for folder in run_folders(repo):
        path = folder / name
        if path.is_file() and path.read_text():
            return int(path.read_text())
    return None


def kill_if_alive(*pids: int | None) -> None:
    for pid in pids:
        if pid is not None and alive(pid):
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def child_numbers(repo: Path, *names: str) -> list[int | None]:
    return [read_number(repo, name) for name in names]


# Ignores SIGTERM, so only the kill after the grace period can end it.
IGNORES_SIGTERM = """
import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
folder = os.environ['FLEETRESERVE_RUN_DIR']
open(os.path.join(folder, 'pid.txt'), 'w').write(str(os.getpid()))
time.sleep(30)
"""

# Traps SIGTERM and exits cleanly with code 0, as a well-behaved program might.
TRAPS_SIGTERM_EXITS_0 = """
import os, signal, sys, time
signal.signal(signal.SIGTERM, lambda number, frame: sys.exit(0))
folder = os.environ['FLEETRESERVE_RUN_DIR']
open(os.path.join(folder, 'pid.txt'), 'w').write(str(os.getpid()))
time.sleep(30)
"""

# Notes that SIGTERM arrived and carries on; SIGINT keeps its default reaction.
NOTES_SIGTERM_AND_CARRIES_ON = """
import os, signal, time
folder = os.environ['FLEETRESERVE_RUN_DIR']
signal.signal(signal.SIGINT, signal.default_int_handler)
signal.signal(
    signal.SIGTERM, lambda number, frame: open(os.path.join(folder, 'term.txt'), 'w').close()
)
open(os.path.join(folder, 'pid.txt'), 'w').write(str(os.getpid()))
time.sleep(30)
"""

# A command whose helper process ignores SIGTERM and outlives the command itself.
_HELPER_SOURCE = "\n".join(
    [
        "import os, signal, time",
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
        "folder = os.environ['FLEETRESERVE_RUN_DIR']",
        "open(os.path.join(folder, 'helper.pid'), 'w').write(str(os.getpid()))",
        "time.sleep(30)",
    ]
)
LEAVES_A_HELPER = (
    "import os, subprocess, sys, time\n"
    f"subprocess.Popen([sys.executable, '-c', {_HELPER_SOURCE!r}])\n"
    "folder = os.environ['FLEETRESERVE_RUN_DIR']\n"
    "open(os.path.join(folder, 'pid.txt'), 'w').write(str(os.getpid()))\n"
    "time.sleep(30)\n"
)


def test_run_that_outlasts_its_limit_is_stopped_and_recorded(repo, capsys):
    pid = None
    try:
        status = record_run("slow", py(SLEEPER), max_seconds=2, grace_seconds=10, start_dir=repo)
        pid = read_number(repo, "pid.txt")
        assert pid is not None, "the command did not start"
        assert status == 128 + signal.SIGTERM
        assert wait_for(lambda: not alive(pid), timeout=5), "the command is still running"
    finally:
        kill_if_alive(pid)
    (folder,) = run_folders(repo)
    manifest = read_manifest(folder)
    assert set(manifest) == MANIFEST_KEYS
    assert manifest["exit_code"] == 128 + signal.SIGTERM
    assert manifest["timed_out"] is True
    assert manifest["max_seconds"] == 2
    assert "interrupted" not in manifest
    assert "log.txt" in manifest["outputs"]
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [str(folder), f"exit code {128 + signal.SIGTERM}"]
    assert len(lines) == 3
    assert "time limit" in lines[2]


def test_command_that_ignores_sigterm_is_killed_after_the_grace_period(repo):
    pid = None
    try:
        status = record_run(
            "stubborn", py(IGNORES_SIGTERM), max_seconds=2, grace_seconds=1, start_dir=repo
        )
        pid = read_number(repo, "pid.txt")
        assert pid is not None, "the command did not start"
        assert status == 128 + signal.SIGKILL
        assert wait_for(lambda: not alive(pid), timeout=5), "the command is still running"
    finally:
        kill_if_alive(pid)
    manifest = read_manifest(run_folders(repo)[0])
    assert manifest["exit_code"] == 128 + signal.SIGKILL
    assert manifest["timed_out"] is True
    assert "interrupted" not in manifest


def test_helper_process_left_in_the_group_is_killed_after_the_grace_period(repo):
    helper = pid = None
    try:
        status = record_run(
            "helper", py(LEAVES_A_HELPER), max_seconds=2, grace_seconds=1, start_dir=repo
        )
        pid, helper = child_numbers(repo, "pid.txt", "helper.pid")
        assert pid is not None, "the command did not start"
        assert helper is not None, "the helper did not start"
        # The command itself ended on SIGTERM; the helper ignored it and needed the kill.
        assert status == 128 + signal.SIGTERM
        assert wait_for(lambda: not alive(helper), timeout=5), "the helper is still running"
    finally:
        kill_if_alive(pid, helper)
    assert read_manifest(run_folders(repo)[0])["timed_out"] is True


def test_command_that_traps_sigterm_and_exits_0_still_counts_as_timed_out(repo):
    pid = None
    try:
        status = record_run(
            "trap", py(TRAPS_SIGTERM_EXITS_0), max_seconds=2, grace_seconds=10, start_dir=repo
        )
        pid = read_number(repo, "pid.txt")
        assert pid is not None, "the command did not start"
    finally:
        kill_if_alive(pid)
    assert status == 0
    (folder,) = run_folders(repo)
    manifest = read_manifest(folder)
    assert manifest["exit_code"] == 0
    assert manifest["timed_out"] is True
    assert "interrupted" not in manifest
    assert verify_main([str(folder / "log.txt")]) == 1


def test_run_that_finishes_in_time_is_not_marked_timed_out_and_verifies(repo):
    status = record_run("quick", py(WRITE_RESULTS), max_seconds=60, start_dir=repo)
    assert status == 0
    (folder,) = run_folders(repo)
    manifest = read_manifest(folder)
    assert manifest["timed_out"] is False
    assert manifest["max_seconds"] == 60
    assert "interrupted" not in manifest
    assert verify_main([str(folder / "result.txt")]) == 0


def test_failing_command_inside_the_limit_keeps_its_own_exit_code(repo):
    status = record_run("fails", py("import sys; sys.exit(3)"), max_seconds=60, start_dir=repo)
    assert status == 3
    manifest = read_manifest(run_folders(repo)[0])
    assert manifest["exit_code"] == 3
    assert manifest["timed_out"] is False


def test_long_run_without_a_time_limit_is_refused(repo, capsys):
    marker = repo / "ran.txt"
    command = py(f"open({str(marker)!r}, 'w').close()")
    status = record_run("heavy", command, long=True, prefix=(), start_dir=repo)
    assert status == 2
    assert not marker.exists()
    assert run_folders(repo) == []
    assert "time limit" in capsys.readouterr().err


def test_command_line_long_run_without_a_time_limit_is_refused(repo):
    marker = repo / "ran.txt"
    code = f"open({str(marker)!r}, 'w').close()"
    done = cli(repo, "--name", "heavy", "--long", "--", sys.executable, "-c", code)
    assert done.returncode == 2
    assert not marker.exists()
    assert run_folders(repo) == []
    assert "time limit" in done.stderr


def test_command_line_has_no_switch_to_allow_a_long_run_without_a_limit(repo):
    for extra in (["--no-limit"], ["--max-seconds", "none"], ["--grace-seconds", "1"]):
        done = cli(repo, "--name", "heavy", "--long", *extra, "--", "true")
        assert done.returncode == 2
    assert run_folders(repo) == []


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan", "abc", "", "-inf", "1e999"])
def test_command_line_bad_time_limit_is_refused_and_nothing_runs(repo, value):
    marker = repo / "ran.txt"
    code = f"open({str(marker)!r}, 'w').close()"
    done = cli(repo, "--name", "bad", "--max-seconds", value, "--", sys.executable, "-c", code)
    assert done.returncode == 2
    assert not marker.exists()
    assert run_folders(repo) == []


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), -0.5, "3", True, False])
def test_library_bad_time_limit_is_refused_and_nothing_runs(repo, value):
    marker = repo / "ran.txt"
    command = py(f"open({str(marker)!r}, 'w').close()")
    assert record_run("bad", command, max_seconds=value, start_dir=repo) == 2
    assert not marker.exists()
    assert run_folders(repo) == []


def test_command_line_time_limit_stops_the_command_and_prints_one_line_about_it(repo):
    done = cli(repo, "--name", "cli", "--max-seconds", "2", "--", *py(SLEEPER))
    pid = read_number(repo, "pid.txt")
    try:
        assert done.returncode == 128 + signal.SIGTERM
        assert pid is not None, "the command did not start"
        assert wait_for(lambda: not alive(pid), timeout=5), "the command is still running"
    finally:
        kill_if_alive(pid)
    (folder,) = run_folders(repo)
    lines = done.stdout.splitlines()
    assert lines[:2] == [str(folder), f"exit code {128 + signal.SIGTERM}"]
    assert len(lines) == 3
    assert "time limit" in lines[2]
    manifest = read_manifest(folder)
    assert manifest["max_seconds"] == 2
    assert manifest["timed_out"] is True


def test_command_line_time_limit_that_is_not_reached_leaves_a_normal_run(repo):
    done = cli(repo, "--name", "cli", "--max-seconds", "60", "--", sys.executable, "-c", "pass")
    assert done.returncode == 0
    (folder,) = run_folders(repo)
    assert done.stdout.splitlines() == [str(folder), "exit code 0"]
    manifest = read_manifest(folder)
    assert manifest["max_seconds"] == 60
    assert manifest["timed_out"] is False


# Runs the recorder in a separate process with a time limit and a grace period, through a
# wrapper that stays alive beside the command, as `caffeinate -i` does.
LIMIT_DRIVER = """
import sys
from pathlib import Path
from fleetreserve.run_record import record_run
run_it = "import subprocess, sys; sys.exit(subprocess.run(sys.argv[1:]).returncode)"
wrapper = [sys.executable, "-c", run_it]
repo, limit, grace = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
status = record_run(
    "first", sys.argv[4:], long=True, max_seconds=limit, grace_seconds=grace,
    prefix=wrapper, start_dir=repo,
)
sys.exit(status)
"""


def start_limited_recorder(repo: Path, limit: float, grace: float, command: list[str]):
    return subprocess.Popen(
        [sys.executable, "-c", LIMIT_DRIVER, str(repo), str(limit), str(grace), *command],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_heavy_lock_is_kept_until_a_timed_out_command_has_really_ended(repo, capsys):
    recorder = start_limited_recorder(repo, 2, 20, py(SLOW_TO_STOP))
    pid = None
    try:
        assert wait_for(lambda: pid_file(repo) is not None), "the command did not start"
        pid = int(pid_file(repo).read_text())
        (first,) = run_folders(repo)
        # The limit stops the command with SIGTERM; it then takes 3 seconds to wind down.
        assert wait_for(lambda: (first / "stopping.txt").exists()), "the limit did not fire"
        # The wrapper is gone by now, the command is still winding down: the lock must hold.
        assert alive(pid)
        capsys.readouterr()
        second = record_run(
            "second", py("pass"), long=True, max_seconds=60, prefix=(), start_dir=repo
        )
        assert second == 2
        assert "heavy run is already going" in capsys.readouterr().err
        assert run_folders(repo) == [first]
        _, errors = recorder.communicate(timeout=30)
    finally:
        stop_leftovers(recorder, pid)
    assert "Traceback" not in errors
    assert not alive(pid)
    manifest = read_manifest(first)
    assert manifest["timed_out"] is True
    assert manifest["max_seconds"] == 2
    assert "interrupted" not in manifest
    # Now that the first run has fully ended, a heavy run is allowed again.
    third = record_run("third", py("pass"), long=True, max_seconds=60, prefix=(), start_dir=repo)
    assert third == 0


def test_signal_during_the_grace_period_still_ends_the_run_and_leaves_a_manifest(repo):
    # The command ignores SIGTERM and the grace period is long, so only the signal can end it.
    recorder = start_limited_recorder(repo, 2, 60, py(NOTES_SIGTERM_AND_CARRIES_ON))
    pid = None
    try:
        assert wait_for(lambda: pid_file(repo) is not None), "the command did not start"
        pid = int(pid_file(repo).read_text())
        (folder,) = run_folders(repo)
        assert wait_for(lambda: (folder / "term.txt").exists()), "the limit did not fire"
        recorder.send_signal(signal.SIGINT)
        _, errors = recorder.communicate(timeout=20)
    finally:
        stop_leftovers(recorder, pid)
    assert "Traceback" not in errors
    assert recorder.returncode != 0
    assert wait_for(lambda: not alive(pid), timeout=5), "the command is still running"
    manifest = read_manifest(folder)
    assert manifest["timed_out"] is True
    assert manifest["interrupted"] == "SIGINT"
    assert manifest["exit_code"] != 0
    assert verify_main([str(folder / "log.txt")]) == 1


def test_signal_before_the_limit_is_recorded_as_before_and_not_as_a_timeout(repo):
    recorder = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fleetreserve.run_record",
            "--name",
            "stop",
            "--max-seconds",
            "600",
            "--",
            *py(SLEEPER),
        ],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    pid = None
    try:
        assert wait_for(lambda: pid_file(repo) is not None), "the command did not start"
        pid = int(pid_file(repo).read_text())
        recorder.send_signal(signal.SIGTERM)
        _, errors = recorder.communicate(timeout=20)
    finally:
        stop_leftovers(recorder, pid)
    assert recorder.returncode == 128 + signal.SIGTERM
    assert "Traceback" not in errors
    manifest = read_manifest(run_folders(repo)[0])
    assert manifest["interrupted"] == "SIGTERM"
    assert manifest["timed_out"] is False
    assert manifest["max_seconds"] == 600
