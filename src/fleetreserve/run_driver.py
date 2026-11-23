"""The driver: run the entries of one purpose of a run list in order, one at a time, each as a
recorded heavy run, and keep one log of the drive (plan section 13: WP12 "Dress rehearsal of
held-out scripts on development units. Pass: done, its duration logged"; WP14 "Held-out run, once,
from the tag ... Pass: one logged run plus logged parsing fixes; logs hashed"; rehearsal item W4).

Command line, from the repository's working copy (its git top folder is found from the current
folder):
    python -m fleetreserve.run_driver <run list> --purpose rehearsal|sealed-run
A long drive is started as `caffeinate -i uv run --frozen --group learn --group baselines python
-m fleetreserve.run_driver <run list> --purpose <purpose>`, detached (`nohup ... &`), in an
environment with every group the list's commands name, since its start check loads each model
(`run_starts` reading T4); without one the start check cannot decide and the drive is refused.
The purpose must be given.

BEFORE ANYTHING RUNS: for the purpose `sealed-run`, the gate of the run from the tag passes
(reading D6): the checks that the sealed files' loader makes before it opens one, called through
`cold_start_gate.require_gate` (the working copy is the repository's main one, the tag
`frozen-plan` names the checked-out commit or the checked-out commit is a logged parsing fix
after it, as the loader's reading H6 states, the tree is clean with no tracked file hidden from
`git status`, and the code running is the checked-out commit's), and the commit checked out is
the one the gate passed at: the tag's, or the fix commit. For
both purposes: the working tree is clean; the run list is committed at the checked-out commit and
the file holds exactly the committed bytes (reading D1); those bytes pass the checker (`run_list`,
its labels held against the limit on a name in the working copy's runs folder, with the sealed
run's rules of its module text, the matrices of rules 3 and 5 read from git's objects at the
checked-out commit, for either purpose, so a rehearsal checks the sealed run's list); the list
passes the start check (`run_starts.start_faults`, reading D7): each step of the sealed run that
runs a command of the sealed route is started by its own code at a made-up tag in a made-up
repository holding the checked-out commit and no data, up to its first read of data, and any
refusal before that read refuses the drive; `caffeinate` is on the search path. The commit
checked out is recorded. Nothing of a step runs, and no sealed file is opened, before all of
these pass.

THE STEPS are the list's instantiation for the purpose (`run_list.instantiate`): the entries that
run in it, in the list's order, each command with `{units}` replaced by the units of the entry's
group in that purpose, each fixed value by its value and the purpose's extra arguments added at
its end.

THEN EACH STEP, in that order:
- the commit and the clean tree are checked again; if either changed, the drive stops before the
  step;
- each `{run:<label>}` of its command is replaced by the run folder that the step `<label>`
  recorded earlier in this drive (`run_list.fill_runs`);
- the step runs as
      caffeinate -i <python> -m fleetreserve.run_record --name=<label> --long
          --max-seconds <max_seconds> -- <command>
  from the top folder, where <python> is the interpreter running the driver and <command> is the
  step's command as filled in. The label is one argument joined to `--name=`, so a label that
  begins with '-' is never read as an option. The recorder keeps the command's output in the
  run's `log.txt`, writes the run's manifest and prints the run's folder as its first line; that
  folder is the step's run when it is `runs.noindex/<UTC start>-<label>` of this working copy
  (the runs folder is never listed to find it);
- for the rehearsal, the drive stops after the first entry whose recorded run ended with an exit
  status other than 0, timed out or was interrupted, and after an entry the recorder refused,
  whose run folder or manifest cannot be found, or whose recorder ended with another exit status
  than the one its manifest records. For the sealed run, an entry whose recorded run ended with
  an exit status other than 0 or timed out is logged as failed (a `failed` line with its reason)
  and the drive goes on; a later step that needs the run of an entry that failed or was not run
  (a `{run:<label>}` of its command, or a label of its `after`) is not started and is logged as
  not run (a `not_run` line: `not run: needs <label>, which failed` or `which was not run`); every
  other later step runs. Nothing sealed is open before the first entry that runs the sealed
  truth build (found by its command, not its place), so before it the sealed run stops, as the
  rehearsal does, after a failed, timed-out or interrupted entry: the steps not reached are named
  as not run in its end line, and the stop opened no sealed file (a re-drive is allowed, WP14
  mark 2). The sealed run also stops at once after a failed entry that runs the sealed truth
  build (the held-out files' parsing, whose planned stop is a logged layout refusal and an
  approved fix, plan section 13, WP14) when no earlier entry that runs it has passed; once one
  has, a later such entry that fails (the second build parses the bytes the first parsed, so its
  failure is no layout fault) is logged as failed like any later step, and only the steps that
  need its run are not run. The sealed run stops at once, whatever the entry, after an entry that
  was interrupted, that the recorder refused, whose run folder or manifest cannot be found, or
  whose recorder ended with another exit status than the one its manifest records, and before a
  step when the commit or the tree changed (the steps not reached are then named as not run in
  its end line).

AFTER THE LAST ENTRY RUN, the commit and the clean tree are checked once more and the result goes
in the end line; a change makes the drive's exit status 2 even when every entry passed.

THE LOCK. The driver takes no lock of its own. Each entry's recorder takes the exclusive heavy-run
lock for its own run (`run_record --long`), so a driver holding that lock would wait on its own
entries. Two drives at once cannot both run an entry: the second's recorder is refused, and that
drive stops.

The driver names no unit and no module: every label, command, unit, time limit and fixed value
comes from the list. A truth build is an entry like any other.

THE LOG: `runs.noindex/drives/<UTC start>-<purpose>.jsonl` (reading D3), one JSON object per
line, each written and flushed to disk when it happens:
- `start`: the log's format `drive-log-3`, the UTC start, the commit, the list's path from the top
  folder and the SHA-256 of its bytes, the purpose driven, the tag's commit (`tag_commit`, null
  for a rehearsal), for a sealed run at a logged parsing fix only the fix commit (`fix_commit`,
  the checked-out commit; a drive at the tag writes no such field, so its start line is as
  before), the number of steps (`entries`), the steps as instantiated (`steps`: each
  one's label, command with each `{run:<label>}` still as written, and time limit), the driver's
  process number;
- `begin`, before each step: its index (from 0, in the purpose's steps), label, the command as
  run (every placeholder filled in), its time limit and the UTC start;
- `entry`, after each entry: its index and label; `run`, the run folder the recorder reported,
  from the top folder (null when it reported none of this working copy); `exit_status`, the exit
  code its manifest records (when no manifest was read, the recorder's own exit status);
  `timed_out`; `interrupted` (the signal the manifest names, or null); `start` and `end` (UTC) and
  `seconds` (its duration); `manifest_sha256` (the SHA-256 of the run's `manifest.json`, null
  when none was read); `recorder_status`, the recorder's own exit status, and `recorder_output`,
  the last characters of what it printed;
- `failed`, for the sealed run only, right after the `entry` line of an entry that failed and
  after which the drive went on: its index, label and the reason; `not_run`, for the sealed run
  only, in place of `begin` and `entry` for a step not started because it needs the run of an
  entry that failed or was not run: its index, label and the reason;
- `stop`, when the drive stops early: the index and label of the entry it stopped at or before,
  and the reason;
- `end`, always last: the UTC end, `seconds` (the total duration of the drive), `entries_run`,
  `completed` (every entry ran and passed, and nothing changed after the last); for the sealed
  run only `failed` and `not_run`, each a list of `{index, label, reason}` naming every entry
  that failed and every step not run (a drive that stopped names the steps it did not reach); the
  driver's
  `exit_status`, `unchanged_after` (the commit and the clean tree were as at the start after the
  last entry run) and `change_after` (what changed, or null), and `log_sha256`: the SHA-256 of
  every byte of the log above this line followed by this line's own fields as written without
  `log_sha256` (`end_digest`), so an edit of the total duration breaks it too.
A drive cut off by a signal or a crash leaves a log without `end`.

Exit status: 0 every entry ran and its recorded run ended with exit 0, neither timed out nor
interrupted, and the commit and the tree were unchanged after the last; 1 an entry's recorded run
did not (the rehearsal stopped after it; the sealed run went on, or stopped when it was
interrupted, came before the first sealed truth build, or was that first build), so a sealed-run
drive with a failed or
not-run entry never exits 0, and its last printed line names every failed and every not-run
entry; 2 the drive could not run: it was refused before the first entry (no purpose or an
unknown one; for the sealed run, its gate refused; not in a git working copy, a tree not clean,
a list missing, not committed, changed, faulty or undecidable, a step's start that the start
check refuses or cannot decide, no `caffeinate`, the log cannot be made), or it stopped before
an entry because the commit or the tree changed, or after an entry the recorder refused, whose
run cannot be found or whose recorder's status its manifest does not confirm, or the commit or
the tree changed after the last entry. An unexpected error is 2 too.

Readings (alternatives in brackets):

  D1. A drive is always of a committed list: the list's file must be tracked at the checked-out
      commit with the same bytes, and the bytes run are the commit's (`git show`), so the log's
      commit and list hash name the list exactly. [Alternative: drive an uncommitted list and
      copy its bytes into the log.]
  D2. The entries' dependencies (`after`) need nothing at run time for the rehearsal: the checker
      requires each to name an earlier entry, the entries run in the list's order, and the drive
      stops at the first entry that does not pass, so an entry runs only after every earlier one
      passed. In the sealed run, which is made once and opens the held-out files once, a failed
      entry does not stop the drive: the entries that do not need its run go on, and those that
      need it (by `{run:<label>}` or `after`, also through an entry not run) are logged as not
      run; before the first entry that runs the sealed truth build, which opens the held-out
      files, a failed entry stops the sealed run too, since nothing sealed is open yet, and so
      does a failure of that first build itself; a later build of that kind (the second parses
      the bytes the first parsed) that fails after the first passed goes on like any other step.
      [Alternative: stop at the first failure in both purposes, as the rehearsal does.]
  D3. The log is a file of its own in the ignored `runs.noindex/drives/`, named by its start and
      the purpose driven, so the WP12 pass-mark check finds the newest rehearsal log by its name.
      It is not inside a recorded run: a drive lasts hours, and a long recorded run takes the
      heavy-run lock that its entries need. [Alternative: run the driver under `run_record`
      without `--long` and write the log into that run's folder, so its manifest hashes it too.]
  D4. No signal is passed on. A drive is stopped cleanly by stopping the running entry's
      recorder (`run_record` passes the signal on to its command and records the run as
      interrupted), after which the driver logs the entry and stops. A signal sent to the driver
      alone ends it without an `end` line, and the running entry goes on to its own end.
  D5. An entry's run is the folder the recorder itself reports (the first line it prints), never
      one found by listing the runs folder, where a step or another drive could have made a
      folder with the same label; and the recorder's own exit status must equal the exit code its
      manifest records, or the drive stops (2), since the two then disagree on what ran.
      [Alternative: read the manifest alone and log the recorder's status beside it.]
  D6. The sealed run's gate is the one the cold start's command uses (`cold_start_gate`, which
      calls the sealed files' loader's own checks 2 to 8 in the loader's order), not a second
      check of the tag written here, and it runs before anything else, so a sealed-run drive
      away from the tag and from a logged parsing fix of it, on a tree not clean or with code
      that is not the checked-out commit's, is refused with the loader's own message, writes no
      log and runs nothing. At a fix the gate returns the fix commit with the tag's
      (`sealed_route.GateCommit`), the checked-out commit must be that fix commit, and the start
      line records both. It is imported only when the
      sealed run is driven, so a rehearsal never calls the gate; the checker loads the route's
      modules to read the sealed truth build's module, the nine held-out units and the route's
      decision at each command's start (`run_list` readings L11, L14 and L15), for either
      purpose, and calls nothing of the gate (L15: the deciding functions only on the branches
      that return before the folders and the gate). The steps
      of the sealed run still make their own checks when they run; the driver's checks of the
      commit and the clean tree before each step keep it at the commit the gate passed at.
      [Alternative: drive the sealed run anywhere and leave the tag to each step's own gate.]
  D7. The start check is made by the driver itself, for either purpose, after the list's rules
      and before the log and the first step, with no option that leaves it out: a refusal a
      command makes at its start (`compare_run --splits` other than 1, for one) would otherwise
      come only when its step runs, after the sealed truth builds have opened the sealed files
      (`run_starts` reading T5). Its probes open no data (`run_starts`, THE PROBE). A drive whose
      start check cannot decide (a model package missing, a probe that does not end) is refused.
      [Alternative: the checker's command line only, run by hand before the drive.]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from fleetreserve import run_list, run_starts
from fleetreserve.run_record import (
    DEFAULT_PREFIX,
    MANIFEST_NAME,
    RUN_STAMP,
    RUNS_FOLDER,
    format_time,
    sha256_file,
    utc_now,
)

LOG_FORMAT = "drive-log-3"
TAG = "frozen-plan"  # named in messages only; the gate itself is the loader's (reading D6)
FIX_KEY = "fix_commit"  # the start line's fix commit, at a logged parsing fix only (reading D6)
DRIVES_FOLDER = "drives"
LOG_SUFFIX = ".jsonl"
RECORDER = "fleetreserve.run_record"
OUTPUT_KEPT = 2000  # the last characters of the recorder's own output kept per entry
PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2
# The purposes whose drive goes on after an entry that failed (module text, reading D2).
CONTINUES = frozenset({run_list.SEALED_RUN})


class Refused(Exception):
    """The drive cannot start; the message says why (exit status 2)."""


def _git(folder: Path, *args: str, text: bool = True):
    try:
        done = subprocess.run(
            ["git", "-C", str(folder), *args], capture_output=True, text=text, check=False
        )
    except OSError as problem:
        raise Refused(f"git could not be started: {problem}") from problem
    if done.returncode != 0:
        message = done.stderr if text else done.stderr.decode(errors="replace")
        raise Refused(f"git {' '.join(args)} failed: {message.strip()}")
    return done.stdout


def tree_state(top: Path) -> tuple[str, list[str]]:
    """(the checked-out commit, the lines of `git status --porcelain`)."""
    commit = _git(top, "rev-parse", "HEAD").strip()
    return commit, _git(top, "status", "--porcelain").splitlines()


def committed_list(top: Path, path: Path) -> tuple[str, bytes]:
    """(the list's path from the top folder, its committed bytes) (reading D1)."""
    file = path.resolve()
    try:
        relative = file.relative_to(top.resolve()).as_posix()
    except ValueError as problem:
        raise Refused(f"the run list {path} is not inside the working copy {top}") from problem
    try:
        on_disk = file.read_bytes()
    except OSError as problem:
        raise Refused(f"the run list {path} cannot be read: {problem}") from problem
    try:
        committed = _git(top, "show", f"HEAD:{relative}", text=False)
    except Refused as problem:
        raise Refused(f"the run list {relative} is not committed at the checked-out commit") from (
            problem
        )
    if committed != on_disk:
        raise Refused(f"the run list {relative} differs from its committed bytes")
    return relative, committed


def gate(top: Path, purpose: str) -> tuple[str | None, str | None]:
    """For the sealed run, (the tag's commit, the fix commit at a logged parsing fix or None)
    once its gate passes (reading D6); (None, None) for a rehearsal. Raises `Refused`."""
    if purpose != run_list.SEALED_RUN:
        return None, None
    from fleetreserve import cold_start_gate  # only for the sealed run (reading D6)

    try:
        found = cold_start_gate.require_gate(top)
    except cold_start_gate.SealRefused as problem:
        raise Refused(
            f"a {purpose} drive starts only at the tag {TAG}, or at a logged parsing fix of it, "
            f"on a clean tree: {problem}"
        ) from problem
    return found, getattr(found, FIX_KEY, None)


def prepare(top: Path, list_path: Path) -> tuple[str, str, bytes, run_list.RunList]:
    """(commit, list path, list bytes, the parsed list) once the drive may start; raises
    `Refused` (module text)."""
    commit, changed = tree_state(top)
    if changed:
        shown = "; ".join(changed[:5]) + (" ..." if len(changed) > 5 else "")
        raise Refused(f"the git tree has uncommitted changes: {shown}")
    relative, data = committed_list(top, list_path)
    try:
        parsed = run_list.parse(run_list.decode(data), name_max=run_list.run_folder_limit(top))
        units = run_list.unit_faults(parsed, top)  # rules 3, 5 and 8, for either purpose
        if units:
            raise run_list.Faulty(units)
        started = run_starts.start_faults(parsed, top)  # the start check (reading D7)
        if started:
            raise run_list.Faulty(started)
    except run_list.Faulty as problem:
        lines = "\n".join(f"FAULT: {line}" for line in problem.faults)
        raise Refused(f"the run list {relative} is faulty ({problem}):\n{lines}") from problem
    except run_list.CannotDecide as problem:
        raise Refused(f"the run list {relative} cannot be decided: {problem}") from problem
    if shutil.which(DEFAULT_PREFIX[0]) is None:
        raise Refused(f"{DEFAULT_PREFIX[0]} was not found on the search path")
    return commit, relative, data, parsed


def encode(record: dict) -> bytes:
    """One line of the log, as written: the record's JSON with sorted keys, and a newline."""
    return (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")


def end_digest(above: bytes, end: dict) -> str:
    """The `log_sha256` of an end line (module text): the SHA-256 of the bytes of the lines above
    it followed by the end line's own fields as written without `log_sha256`."""
    fields = {key: value for key, value in end.items() if key != "log_sha256"}
    return hashlib.sha256(above + encode(fields)).hexdigest()


class Log:
    """The drive's log (module text): each line flushed to disk as it is written."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.written = bytearray()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("xb")

    def __enter__(self) -> "Log":
        return self

    def __exit__(self, *_) -> None:
        self._handle.close()

    def _line(self, record: dict) -> bytes:
        line = encode(record)
        self._handle.write(line)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return line

    def write(self, record: dict) -> None:
        self.written += self._line(record)

    def end(self, record: dict) -> None:
        """The last line: `record` with its `log_sha256` (`end_digest`)."""
        self._line({**record, "log_sha256": end_digest(bytes(self.written), record)})


def log_path(top: Path, purpose: str) -> Path:
    stamp = utc_now().strftime(RUN_STAMP)
    return top / RUNS_FOLDER / DRIVES_FOLDER / f"{stamp}-{purpose}{LOG_SUFFIX}"


def reported_folder(top: Path, label: str, output: str) -> Path | None:
    """The run folder the recorder printed as the first line of its output, when it is
    `runs.noindex/<UTC start>-<label>` of the working copy at `top` and exists; else None."""
    lines = output.splitlines()
    if not lines or not lines[0]:
        return None
    try:
        folder = Path(lines[0]).resolve()
        runs = (top / RUNS_FOLDER).resolve()
    except (OSError, RuntimeError):
        return None
    named = re.fullmatch(r"\d{8}T\d{6}Z-" + re.escape(label), folder.name)
    if folder.parent != runs or not named or not folder.is_dir():
        return None
    return folder


def recorder_command(entry: run_list.Step, command: list[str]) -> list[str]:
    """The recorder call that runs one step (module text)."""
    return [
        *DEFAULT_PREFIX,
        sys.executable,
        "-m",
        RECORDER,
        f"--name={entry.label}",
        "--long",
        "--max-seconds",
        str(entry.max_seconds),
        "--",
        *command,
    ]


def run_entry(top: Path, k: int, entry: run_list.Step, command: list[str]) -> dict:
    """Run one entry as a recorded run; its `entry` record."""
    start = utc_now()
    clock = time.monotonic()
    done = subprocess.run(
        recorder_command(entry, command),
        cwd=top,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    seconds = time.monotonic() - clock
    record = {
        "record": "entry",
        "index": k,
        "label": entry.label,
        "run": None,
        "exit_status": done.returncode,
        "timed_out": False,
        "interrupted": None,
        "start": format_time(start),
        "end": format_time(utc_now()),
        "seconds": seconds,
        "manifest_sha256": None,
        "recorder_status": done.returncode,
        "recorder_output": (done.stdout + done.stderr)[-OUTPUT_KEPT:],
    }
    folder = reported_folder(top, entry.label, done.stdout)
    if folder is None:
        return record
    record["run"] = folder.relative_to(top.resolve()).as_posix()
    manifest_file = folder / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_file.read_text())
        digest = sha256_file(manifest_file)
    except (OSError, ValueError):
        return record
    code = manifest.get("exit_code") if isinstance(manifest, dict) else None
    if not isinstance(code, int) or isinstance(code, bool):
        return record
    record["exit_status"] = code
    record["timed_out"] = manifest.get("timed_out", False) is True
    record["interrupted"] = manifest.get("interrupted")
    record["manifest_sha256"] = digest
    return record


def outcome(record: dict) -> int | None:
    """None when the entry passed; else the drive's exit status (module text)."""
    if record["manifest_sha256"] is None or record["recorder_status"] != record["exit_status"]:
        return COULD_NOT_RUN
    if record["exit_status"] != 0 or record["timed_out"] or record["interrupted"] is not None:
        return FAILED
    return None


def _reason(record: dict) -> str:
    if record["manifest_sha256"] is None:
        found = (
            "no run folder" if record["run"] is None else f"no readable manifest in {record['run']}"
        )
        return f"the recorder ended with {record['recorder_status']} and {found} was found"
    if record["recorder_status"] != record["exit_status"]:
        return (
            f"the recorder ended with {record['recorder_status']} but the manifest of its run "
            f"records the exit code {record['exit_status']}"
        )
    if record["timed_out"]:
        return f"the run hit its time limit (exit status {record['exit_status']})"
    if record["interrupted"] is not None:
        return f"the run was interrupted by {record['interrupted']}"
    return f"the run ended with exit status {record['exit_status']}"


def logged_steps(steps: Sequence[run_list.Step]) -> list[dict]:
    """The `steps` of the log's start line (module text)."""
    return [
        {"label": s.label, "command": list(s.command), "max_seconds": s.max_seconds} for s in steps
    ]


def drive(list_path: Path, *, purpose: str, start_dir: Path | None = None) -> int:
    """Drive the purpose `purpose` of the list in the file `list_path` as the module text says;
    the exit status."""
    folder = Path(start_dir or Path.cwd())
    try:
        if purpose not in run_list.PURPOSES:
            raise Refused(f"the purpose {purpose!r} is not one of {', '.join(run_list.PURPOSES)}")
        top = Path(_git(folder, "rev-parse", "--show-toplevel").strip())
        tag_commit, fix_commit = gate(top, purpose)
        commit, relative, data, parsed = prepare(top, folder / list_path)
        if fix_commit is not None and commit != fix_commit:
            raise Refused(
                f"the checked-out commit {commit[:12]} is not the parsing fix "
                f"{fix_commit[:12]} the gate passed at"
            )
        if fix_commit is None and tag_commit is not None and commit != tag_commit:
            raise Refused(
                f"the checked-out commit {commit[:12]} is not the tag's {tag_commit[:12]}"
            )
        steps = run_list.instantiate(parsed, purpose)
        log = Log(log_path(top, purpose))
    except (Refused, OSError) as problem:
        print(f"refused: {problem}")
        return COULD_NOT_RUN
    with log:
        clock = time.monotonic()
        log.write(
            {
                "record": "start",
                "format": LOG_FORMAT,
                "start": format_time(utc_now()),
                "commit": commit,
                "list": relative,
                "list_sha256": hashlib.sha256(data).hexdigest(),
                "purpose": purpose,
                "tag_commit": tag_commit,
                **({} if fix_commit is None else {FIX_KEY: fix_commit}),
                "entries": len(steps),
                "steps": logged_steps(steps),
                "pid": os.getpid(),
            }
        )
        print(f"drive log: {log.path}")
        status, ran, failed, not_run = _run_entries(top, commit, steps, log, purpose)
        after = _changed(top, commit)  # once more, after the last entry run (module text)
        if after is not None and status == PASSED:
            print(f"STOP after the last entry: {after}")
            status = COULD_NOT_RUN
        seconds = time.monotonic() - clock
        end = {
            "record": "end",
            "end": format_time(utc_now()),
            "seconds": seconds,
            "entries_run": ran,
            "completed": status == PASSED,
            "exit_status": status,
            "unchanged_after": after is None,
            "change_after": after,
        }
        if purpose in CONTINUES:
            end |= {"failed": failed, "not_run": not_run}
        log.end(end)
    print(f"drive ended with exit status {status} after {seconds:.0f} s: {log.path}")
    if failed or not_run:
        print(_problems(failed, not_run))
    return status


def _changed(top: Path, commit: str) -> str | None:
    """What changed since the drive started (the commit or the clean tree), or None (module
    text)."""
    try:
        now, changed = tree_state(top)
    except Refused as problem:
        return f"the tree cannot be checked: {problem}"
    if now != commit:
        return f"the commit changed since the drive started (now {now[:12]})"
    if changed:
        return "the git tree has uncommitted changes since the drive started"
    return None


def _needs(entry: run_list.Step, unusable: dict[str, str]) -> list[str]:
    """What the step needs of entries that failed or did not run, one phrase each (module text):
    the entries whose run it names (`{run:<label>}`) and those it follows (`after`)."""
    labels = dict.fromkeys([label for _, label in entry.runs] + list(entry.after))
    return [f"needs {label}, which {unusable[label]}" for label in labels if label in unusable]


def _opens_sealed_build(entry: run_list.Step) -> bool:
    """Whether the step runs the sealed truth build (`sealed_route.TRUTH_BUILD_MODULE`), by its
    command and not its place; a command whose module cannot be read counts as one (module
    text)."""
    from fleetreserve.sealed_route import TRUTH_BUILD_MODULE

    try:
        return run_list.step_module(entry) == TRUTH_BUILD_MODULE
    except StopIteration:
        return True


def _first_opening(steps: Sequence[run_list.Step]) -> int | None:
    """The place of the first step that runs the sealed truth build, which opens the held-out
    files: nothing sealed is open before it, so a failure before it, or of it, stops the sealed
    drive; a later step of that kind goes on after a failure once an earlier one passed (module
    text); None when no step runs it."""
    return next((k for k, step in enumerate(steps) if _opens_sealed_build(step)), None)


def _problems(failed: list[dict], not_run: list[dict]) -> str:
    """The line that names every failed and every not-run entry (module text)."""
    parts = []
    if failed:
        parts.append(f"FAILED: {', '.join(f['label'] for f in failed)}")
    if not_run:
        parts.append(f"NOT RUN: {', '.join(n['label'] for n in not_run)}")
    return f"drive with {' and '.join(parts)}: this drive did not pass"


def _run_entries(
    top: Path, commit: str, steps: Sequence[run_list.Step], log: Log, purpose: str
) -> tuple[int, int, list[dict], list[dict]]:
    """Run the steps in order (module text); (the drive's exit status, the steps run, the entries
    that failed, the entries not run), the last two as `{index, label, reason}` for a purpose in
    `CONTINUES` and empty for any other."""
    runs: dict[str, str] = {}  # label -> the run folder it recorded in this drive
    unusable: dict[str, str] = {}  # label -> "failed" or "was not run" (a purpose in CONTINUES)
    failed: list[dict] = []
    not_run: list[dict] = []
    ran = 0
    opening = _first_opening(steps)
    opened = False  # an entry that opens the held-out files has passed in this drive
    for k, entry in enumerate(steps):
        reason = _changed(top, commit)
        if reason is not None:
            log.write({"record": "stop", "index": k, "label": entry.label, "reason": reason})
            print(f"STOP before {entry.label}: {reason}")
            return COULD_NOT_RUN, ran, failed, _unreached(purpose, steps, k, not_run, reason)
        if purpose in CONTINUES:
            needs = _needs(entry, unusable)
            if needs:
                why = f"not run: {'; '.join(needs)}"
                unusable[entry.label] = "was not run"
                not_run.append({"index": k, "label": entry.label, "reason": why})
                log.write({"record": "not_run", "index": k, "label": entry.label, "reason": why})
                print(f"{entry.label}: {why}")
                continue
        command = run_list.fill_runs(entry, runs)
        log.write(
            {
                "record": "begin",
                "index": k,
                "label": entry.label,
                "command": command,
                "max_seconds": entry.max_seconds,
                "start": format_time(utc_now()),
            }
        )
        print(f"begin {entry.label}")
        record = run_entry(top, k, entry, command)
        ran += 1
        log.write(record)
        print(f"end {entry.label}: exit status {record['exit_status']}, run {record['run']}")
        stopped = outcome(record)
        opens = _opens_sealed_build(entry)
        if stopped is None:
            runs[entry.label] = record["run"]
            opened = opened or opens
            continue
        reason = _reason(record)
        # a later entry that opens the held-out files again (the second truth build parses the
        # bytes the first parsed) goes on after a failure once an earlier one passed; the first
        # such entry stops the drive, as does any step before it
        goes_on = (
            purpose in CONTINUES
            and stopped == FAILED
            and record["interrupted"] is None
            and (opened if opens else (opening is None or opening < k))
        )
        if goes_on:
            # a recorded failure (exit status not 0, or the time limit): the drive goes on
            unusable[entry.label] = "failed"
            failed.append({"index": k, "label": entry.label, "reason": reason})
            log.write({"record": "failed", "index": k, "label": entry.label, "reason": reason})
            print(f"FAILED {entry.label}: {reason}; the drive goes on")
            continue
        log.write({"record": "stop", "index": k, "label": entry.label, "reason": reason})
        print(f"STOP after {entry.label}: {reason}")
        if purpose in CONTINUES and stopped == FAILED:
            failed.append({"index": k, "label": entry.label, "reason": reason})
        stop_reason = f"the drive stopped after {entry.label}: {reason}"
        return stopped, ran, failed, _unreached(purpose, steps, k + 1, not_run, stop_reason)
    if failed or not_run:
        return FAILED, ran, failed, not_run
    return PASSED, ran, failed, not_run


def _unreached(
    purpose: str, steps: Sequence[run_list.Step], first: int, not_run: list[dict], reason: str
) -> list[dict]:
    """`not_run` with the steps from `first` on added, when a stop left them unreached, for a
    purpose in `CONTINUES`; else the empty list (the stop line names where a drive stopped)."""
    if purpose not in CONTINUES:
        return []
    more = [
        {"index": k, "label": s.label, "reason": f"not run: {reason}"}
        for k, s in enumerate(steps)
        if k >= first
    ]
    return not_run + more


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fleetreserve.run_driver",
        description="Run the entries of one purpose of a committed run list in order, one recorded "
        "run at a time.",
    )
    parser.add_argument("run_list", type=Path, help="the run list, a committed JSON file")
    parser.add_argument(
        "--purpose",
        required=True,
        choices=run_list.PURPOSES,
        help=(
            "the purpose driven: the rehearsal, or the run from the tag (only at the tag "
            f"{TAG}, or at a logged parsing fix of it)"
        ),
    )
    options = parser.parse_args(argv)
    try:
        return drive(options.run_list, purpose=options.purpose)
    except Exception as problem:  # an unexpected error: could not run, never Python's own 1
        print(f"COULD NOT RUN: the drive stopped on {type(problem).__name__}: {problem}")
        return COULD_NOT_RUN


if __name__ == "__main__":
    sys.exit(main())
