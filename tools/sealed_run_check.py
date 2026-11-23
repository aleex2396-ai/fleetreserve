"""The check of work package 14's pass mark (plan section 13, WP14: "Pass: one logged run plus
logged parsing fixes, the scoring runs made again at a fix commit belonging to that one run; logs
hashed"), one command: `python tools/sealed_run_check.py [--root <folder>]` (`--root`: the
project's top folder, by default the folder above `tools/`). `passmarks/wp14.sh` runs it under
the run recorder: `python -m fleetreserve.run_record --name wp14-passmark -- bash passmark.sh
wp14`. It writes one results file, `wp14-passmark.json`, into the run folder of the recorded run
going now, by the gate mark's rule (`gate_mark.run_folder`): the folder `FLEETRESERVE_RUN_DIR`
names, when it lies directly in the runs folder of `--root` and holds no manifest yet. The file
holds the checked-out commit, the tag's and the fix's commits, each mark's status and finding
lines, the SHA-256 of every input by its path under `--root` ("absent" when there is none), and
what the check cannot see. With no such run folder (the variable unset or empty, naming no
folder, a folder elsewhere, or one holding a manifest already: a finished run), it writes
nothing and could not run; so too when the file cannot be written. Exit status: the worst
mark's, 0 passed, 1 failed, 2 could not run (never a pass); a failure is 1 even when another mark
could not run. Any other error, a failed import among them, is could not run.

It opens, lists or reads no file of the sealed folder. It reads git; the gate of the sealed routes
(`sealed_route.require_gate`, the held-out loader's own checks: the clean main working copy, the
code of the checked-out commit, which is the tag's or a logged parsing fix of it); the run list
committed at the checked-out commit and each drive log, by the WP12 check's readers
(`tools/rehearsal_check.py`); the `manifest.json` and `log.txt` of each run a drive log names, and
the `ceiling-report.json` of each O2 step of the complete drive (mark 1); and
the openings and refusal logs beside the sealed folder (`sealed_route.openings_log`,
`refusals_log`, `sealed_logs`, their bytes parsed by `openings_in` and `refusals_in`). Every file
it reads by its own path (each drive log, manifest, `log.txt` and output a manifest lists, and the
two logs) it reads, and the drive logs' folder it lists, through one reader
(`fleetreserve.plain_read`): it takes the path's parts below `--root` as written, before any
normalising, and refuses a "..", a "." or an absolute start, opening nothing; then it opens each
name from `--root` down from the descriptor of the folder above it, never following a symbolic
link (`O_NOFOLLOW`, after a look at the name), so a link at the file or at any folder between
`--root` and it is never followed. Either way the mark that needs the file could not run, naming
the path or the link. Every run folder a drive log names, in every drive, must be
`runs.noindex/<UTC start>-<label>` of its step's label, as `verify_run` requires of the complete
drive's, and every output name a manifest lists must be plain (`run_verify.plain_name`); else the
mark that reads it could not run, and nothing in it is read. The committed run list is read
from git. A drive log is `runs.noindex/drives/<UTC start>-sealed-run.jsonl`; it could not run
when it is a link, has a line cut off, has no end line
(a drive killed or still going), has an end whose `log_sha256` does not recompute, has lines that
are not the driver's (for each step a `begin` then an `entry` of its index and label, then, when
that entry failed and the drive went on, a `failed` line of the same index and label with its
reason; in place of a step the driver did not start, a `not_run` line of its index, label and
reason; the indices follow one another through both; at most one `stop`, last), has times that go
back, or has a name that is neither the second of its start line
nor the second before it. Drives are taken in the order of their start lines; openings logged out
of that order (a clock set back between drives) could not run.

A drive is complete when its end says it completed with exit status 0, every step of the list ran
and each passed, with no `failed` or `not_run` line and none in its end line. A drive that ran to
its end with entries that failed or steps not run is no complete drive: mark 1 fails (never could
not run) and names every failed entry and every step not run with its reason, from the log's lines
and its end line. A drive that ran to its end has no `stop` line. With no complete drive, the
newest drive that ran a step and ran to its end is the sealed run made once: mark 1 alone fails it
(it is not judged by mark 2 as an earlier drive), and after the failure mark 1 prints, as
information only (lines begun "information:", never a change of a mark or an exit status), O2's
verdicts, the abstained held-out units and the failed configurations of the learn steps read from
the steps of that drive that passed and verify,
when the drive is of the committed list at the checked-out commit with the gate's tag and fix
commits and its tree is unchanged after the last step. A drive ran no step when no entry names a
run folder: it is not a run. An opening
or a refusal lies inside a step when its commit and tag commit are the drive's and its time, cut
to the second, lies from the step's `begin` to its entry's end, both included; an opening must
also be of the step's command (its module's file and the step's arguments), and is made in a step
only when that step ran.

1. One logged run: exactly one complete drive; no later drive that ran a step; every opening
   inside a step that ran; at least one inside the complete drive. The complete drive is at the
   checked-out commit, with the gate's tag and fix commits, of the committed list's bytes and its
   sealed-run steps; its tree is unchanged after; each run verifies there (`verify_run`). For
   each O2 step of the complete drive (a step running `fleetreserve.ceiling_report` with
   `--every-day`), one line of this mark gives O2's verdict over the units and the units whose
   pass failed or could not be decided, and those with no day to check (counted as neither
   passed nor failed, `ceiling_summary` reading N), read from its `ceiling-report.json` by the
   WP12 check's reader (`tools/rehearsal_check.py`, reading R7: through
   `run_verify.read_verified`, no symbolic link followed). The verdict never decides the mark
   (the owner's decision of 19 November 2026, `ceiling_report` reading O2-4), but an O2 step
   that exited 0 whose results file holds no readable verdict fails it, named.
   Beside that line, one line gives the units' own faults, each count named by what it holds
   (orders broken, with those that are block orders on M days; days not built, with those that are
   M days; only the latter of each are in O2's pass), or says a unit's counts are missing or not
   readable; no mark and no exit status changes by it (decision 88 (3d)).
   Each learn step of the complete drive (a step running `fleetreserve.learn_run`) that exited 0
   has its `failed_configurations` read from its `learn.json` (through `run_verify.read_verified`):
   on the held-out route a configuration that raised an error on one unit is written as `failed:
   <error>` and counts as not passing, and the learn step still exits 0 (`learn_run`, the owner's
   decision 99 (8) of 22 November 2026), so this reading is where it is found. The list names one
   configuration of each group that failed together, so the units' rungs are read too: each
   configuration holding `failed: <error>` there is named as well. Each failed
   configuration fails this mark, naming the step, the unit, the configuration and the error;
   a learn step that exited 0 with no readable list could not run.
2. Logged parsing fixes: every earlier drive that ran a step (not the newest, when it ran to its
   end with no complete drive: mark 1 judges it) is of the gate's tag commit and of a
   commit on the line from the tag to the checked-out commit, and either holds no opening in any
   step (an operational stop), or stopped at a step that plainly failed (exit status not 0, not
   timed out, not interrupted) in which lies a layout refusal following an opening of the same
   bytes made in that step, with no opening in a step before it. An opening that lies in two
   steps of one command, with no refusal following it, could not run.
3. Logs hashed: each step's manifest has the SHA-256 the drive log records and lists its
   `log.txt` with that file's SHA-256 now; each drive log's own hash recomputes on reading. A
   manifest or a `log.txt` with a symbolic link on its path (the file, or a folder above it) is
   never read, and could not run; so too an input that cannot be hashed, a link on its path among
   them.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

try:
    import rehearsal_check as wp12  # the WP12 check's readers of a drive log, unchanged
    from freeze_check import Findings

    from fleetreserve import gate_mark, run_list, sealed_route
    from fleetreserve.learn_run import RESULTS_NAME as LEARN_RESULTS
    from fleetreserve.plain_read import NeverRead
    from fleetreserve.plain_read import listing as _listing
    from fleetreserve.plain_read import read as _read
    from fleetreserve.run_driver import DRIVES_FOLDER, FIX_KEY, LOG_SUFFIX, logged_steps
    from fleetreserve.run_evidence import CannotCheck
    from fleetreserve.run_record import (
        LOG_NAME,
        MANIFEST_NAME,
        RUN_DIR_VARIABLE,
        RUNS_FOLDER,
    )
    from fleetreserve.run_verify import NotVerified
    from fleetreserve.truth_build import SUMMARY_NAME as TRUTH_SUMMARY_NAME
except Exception as problem:  # the check cannot start: could not run, never Python's exit 1
    print(f"COULD NOT RUN: the check cannot start: {type(problem).__name__}: {problem}")
    sys.exit(2)

TOP = Path(__file__).resolve().parents[1]
PURPOSE = run_list.SEALED_RUN
LIST = wp12.LIST
RESULTS_NAME = "wp14-passmark.json"
PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2
WORDS = {PASSED: "passed", FAILED: "failed", COULD_NOT_RUN: "could not run"}
MARKS = ("one logged run", "logged parsing fixes", "logs hashed")
LOG_PATTERN = re.compile(r"\d{8}T\d{6}Z-" + re.escape(PURPOSE) + re.escape(LOG_SUFFIX))
STAMPS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ")
NAME_STAMP = "%Y%m%dT%H%M%SZ"  # the driver's UTC start at the front of a drive log's name
LABEL = re.compile(r"[A-Za-z0-9_-]+")  # a step's label, as the run list allows it
TRUTH_BUILD_STEP = f"fleetreserve.{sealed_route.TRUTH_BUILD_MODULE}"  # a held-out build step
LEARN_STEP = "fleetreserve.learn_run"  # a learn step
FAILED_CONFIGURATION = re.compile(r"(.+?) \| (.+?): (.*)", re.DOTALL)  # gate_run's entry text
FAILED_START = "failed:"  # the start of the reason a failed configuration holds in its rungs
NO_REASON = "no reason logged"
NOT_SEEN = (
    "Not seen by this check: a drive log, the openings log or the refusal log edited by hand (a "
    "drive log with its hash recomputed), and a clock set back between two drives when no opening "
    "of the drive run later is logged after one of the other, which can hide a later drive that "
    "ran steps and opened nothing."
)


@dataclass
class Drive:
    """One readable drive log (module text)."""

    name: str
    records: list[dict]
    steps: list[tuple[dict, dict]]  # (begin, entry) of each step run, in order
    stop: dict | None
    failed: list[dict] = field(default_factory=list)  # the `failed` lines, in order
    not_run: list[dict] = field(default_factory=list)  # the `not_run` lines, in order

    @property
    def start(self) -> dict:
        return self.records[0]

    @property
    def end(self) -> dict:
        return self.records[-1]

    def ran_no_step(self) -> bool:
        return not any(isinstance(entry.get("run"), str) for _, entry in self.steps)

    def problems(self) -> list[tuple[str, dict]]:
        """Every entry that failed and every step not run, ("failed" or "not run", its
        `{index, label, reason}`), in the order of their index: those the log's lines record and
        those the end line lists (a drive that stopped lists the steps it did not reach there
        only)."""
        found: dict[tuple[str, object], dict] = {}
        for kind, lines, key in (
            ("failed", self.failed, "failed"),
            ("not run", self.not_run, "not_run"),
        ):
            listed = self.end.get(key)
            for line in [*lines, *(listed if isinstance(listed, list) else [])]:
                if isinstance(line, dict):
                    found.setdefault((kind, line.get("index")), line)
        return sorted(
            ((kind, line) for (kind, _), line in found.items()),
            key=lambda pair: (
                pair[1]["index"] if type(pair[1].get("index")) is int else len(found) + 10**6
            ),
        )

    def complete(self) -> bool:
        end, entries = self.end, self.start.get("entries")
        return (
            end.get("completed") is True
            and end.get("exit_status") == 0
            and self.stop is None
            and not self.failed
            and not self.not_run
            and not end.get("failed")
            and not end.get("not_run")
            and end.get("entries_run") == entries == len(self.steps)
            and all(wp12.passed(entry) for _, entry in self.steps)
        )


@dataclass
class Survey:
    """What every mark reads first (module text)."""

    head: str
    gate: object | None  # the gate's `GateCommit`, None when it refused
    drives: list[Drive]
    unreadable: int
    data: bytes | None  # the committed list's bytes
    parsed: object | None  # the committed list
    paths: list[Path]  # every drive log's path, readable or not


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise CannotCheck(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _plural(count: int, one: str, many: str | None = None) -> str:
    return f"{count} {one if count == 1 else (many or one + 's')}"


def read_drive(path: Path, root: Path | None = None) -> Drive:
    """A drive log, read as the module text says, through `_read` from `root` (by default the
    log's folder); raises `CannotCheck`."""
    path = Path(path)
    try:
        data = _read(path.parent if root is None else root, path)
    except NeverRead as problem:
        raise CannotCheck(f"the log {path.name} is not read: {problem}") from problem
    except OSError as problem:
        raise CannotCheck(f"the log {path.name} cannot be read: {problem}") from problem
    records = wp12.log_records(path.name, data)  # a log with no end line could not run
    if records[0].get("purpose") != PURPOSE:
        raise CannotCheck(f"the start of {path.name} names another purpose than {PURPOSE!r}")
    _named_by_its_start(path.name, records[0].get("start"))
    body, steps, stop, failed, not_run, at = records[1:-1], [], None, [], [], 0
    while at < len(body):
        line, k = body[at], len(steps) + len(not_run)  # k: the index the next step must have
        if line.get("record") == "stop" and at == len(body) - 1:
            stop = line
            break
        if line.get("record") == "not_run":  # a step not started, in place of its begin and entry
            if not _logged_as(line, "not_run", k):
                raise CannotCheck(f"the lines of {path.name} are not the driver's at step {k + 1}")
            not_run.append(line)
            at += 1
            continue
        entry = body[at + 1] if at + 1 < len(body) else {}
        follows = (
            line.get("record") == "begin"
            and entry.get("record") == "entry"
            and line.get("index") == entry.get("index") == k
            and line.get("label") == entry.get("label")
        )
        if not follows:
            raise CannotCheck(f"the lines of {path.name} are not the driver's at step {k + 1}")
        steps.append((line, entry))
        at += 2
        if at < len(body) and body[at].get("record") == "failed":  # right after its entry
            if not _logged_as(body[at], "failed", k, entry.get("label")):
                raise CannotCheck(f"the lines of {path.name} are not the driver's at step {k + 1}")
            failed.append(body[at])
            at += 1
    _in_time_order(path.name, records[0], steps, records[-1])
    return Drive(path.name, records, steps, stop, failed, not_run)


def _logged_as(line: dict, record: str, k: int, label=None) -> bool:
    """Whether `line` is a `failed` or `not_run` line as the driver writes it: exactly its record,
    index (`k`), label (that of the entry, for a `failed` line) and reason, the reason text."""
    return (
        set(line) == {"record", "index", "label", "reason"}
        and line["record"] == record
        and type(line["index"]) is int
        and line["index"] == k
        and isinstance(line["label"], str)
        and (label is None or line["label"] == label)
        and isinstance(line["reason"], str)
    )


def _named_by_its_start(name: str, start) -> None:
    """The log's name is the second of its start line or the second before it (the driver takes
    the name, then writes the start); else `CannotCheck`."""
    try:
        named = datetime.strptime(name[: len("20260101T000000Z")], NAME_STAMP)
    except ValueError:
        raise CannotCheck(f"the name of the log {name} is not a UTC time") from None
    if not named <= _second(start) <= named + timedelta(seconds=1):
        raise CannotCheck(
            f"the name of the log {name} is neither the second of its start line, {start}, nor "
            "the second before it: a log renamed, or written by another clock"
        )


def _in_time_order(name: str, start: dict, steps: list, end: dict) -> None:
    """Every time the driver wrote, cut to the second, in the order of the lines: the start, each
    step's begin and its entry's start and end, the end line; else `CannotCheck`."""
    stamps = [("its start line", start.get("start"))]
    for k, (begin, entry) in enumerate(steps):
        where = f"step {k + 1} ({begin.get('label')})"
        stamps += [(where, begin.get("start"))]
        stamps += [(where, entry[key]) for key in ("start", "end") if key in entry]
    stamps.append(("its end line", end.get("end")))
    previous = None
    for where, text in stamps:
        when = _second(text)
        if previous is not None and when < previous[1]:
            raise CannotCheck(
                f"the times in the log {name} go back at {where}: {previous[0]} then {text}"
            )
        previous = (text, when)


def survey(root: Path, found: Findings) -> Survey:
    """The gate, the committed list and every drive log (module text)."""
    root = Path(root)
    head = _git(root, "rev-parse", "HEAD").strip()
    try:
        gate = sealed_route.require_gate(root)
    except sealed_route.SealRefused as problem:
        found.cannot_run(
            f"the gate of the run from the tag refuses the checked-out commit: {problem}"
        )
        gate = None
    data = wp12.committed_bytes(root, LIST)
    parsed = None
    if data is None:
        found.cannot_run(f"no run list is committed at {LIST} at the checked-out commit")
    else:
        try:
            parsed = wp12.committed_list(root, data)
        except CannotCheck as problem:
            found.cannot_run(str(problem))
    folder = root / RUNS_FOLDER / DRIVES_FOLDER
    try:
        names = _listing(root, folder)
    except NeverRead as problem:
        raise CannotCheck(f"the folder of the drive logs is not listed: {problem}") from problem
    except (FileNotFoundError, NotADirectoryError):
        names = []
    except OSError as problem:
        raise CannotCheck(f"the folder of the drive logs cannot be listed: {problem}") from problem
    paths = sorted(folder / name for name in names if LOG_PATTERN.fullmatch(name))
    drives, unreadable = [], 0
    for path in paths:
        try:
            drives.append(read_drive(path, root))
        except CannotCheck as problem:
            found.cannot_run(str(problem))
            unreadable += 1
    if not paths:
        found.cannot_run(
            f"no sealed-run drive log, {RUNS_FOLDER}/{DRIVES_FOLDER}/<UTC start>-{PURPOSE}"
            f"{LOG_SUFFIX}"
        )
    return Survey(head, gate, drives, unreadable, data, parsed, paths)


def _second(text) -> datetime:
    for form in STAMPS:
        try:
            return datetime.strptime(text, form).replace(microsecond=0)
        except (TypeError, ValueError):
            continue
    raise CannotCheck(f"the time {text!r} is not a UTC time of the driver's or a log's form")


def _step_command(line: dict, begin: dict) -> bool:
    """Whether an opening's command (the opener's `sys.argv`) is the step's as run: the file of
    the step's module `fleetreserve.<module>` and the step's arguments after it."""
    command, argv = begin.get("command"), line.get("command")
    if not isinstance(command, list) or not isinstance(argv, list) or not argv:
        return False
    for at in range(len(command) - 1):
        module = command[at + 1]
        if command[at] == "-m" and isinstance(module, str) and module.startswith("fleetreserve."):
            file = "/" + module.replace(".", "/") + ".py"
            return str(argv[0]).endswith(file) and argv[1:] == command[at + 2 :]
    return False


def inside(line: dict, drive: Drive, k: int) -> bool:
    """Whether a log line (an opening or a refusal) lies inside step `k` of `drive`."""
    begin, entry = drive.steps[k]
    start = drive.start
    if (line.get("commit"), line.get("tag_commit")) != (
        start.get("commit"),
        start.get("tag_commit"),
    ):
        return False
    if "command" in line and not _step_command(line, begin):
        return False
    when = _second(line.get("time_utc"))
    return _second(begin.get("start")) <= when <= _second(entry.get("end"))


def opened_in(line: dict, drive: Drive, k: int) -> bool:
    """Whether an opening lies inside step `k` of `drive` and its recorder ran the step (the
    entry names a run folder): a step its recorder refused ran no command."""
    return isinstance(drive.steps[k][1].get("run"), str) and inside(line, drive, k)


def paired_opening(refusal: dict, openings: list[dict]) -> dict | None:
    """The opening a layout refusal follows: the last opening before it of the same set, file,
    SHA-256, commit and tag commit; None when there is none."""
    same = ("set", "file", "sha256", "commit", "tag_commit")
    before = [
        line
        for line in openings
        if all(line.get(key) == refusal.get(key) for key in same)
        and _moment(line.get("time_utc")) <= _moment(refusal.get("time_utc"))
    ]
    return before[-1] if before else None


def _moment(text) -> datetime:
    try:
        return datetime.strptime(text, STAMPS[1])
    except (TypeError, ValueError):
        raise CannotCheck(f"the time {text!r} is not a UTC time of a log's form") from None


def refused_in(refusal: dict, openings: list[dict], drive: Drive, k: int) -> bool:
    """Whether a layout refusal was logged inside step `k` of `drive`: the opening it follows
    lies inside that step, commands compared, and the refusal itself too."""
    opening = paired_opening(refusal, openings)
    return opening is not None and inside(opening, drive, k) and inside(refusal, drive, k)


def _in_drive_order(drives: list[Drive], openings: list[dict], found: Findings) -> None:
    """The openings log is written line after line as the openers run, so the drives its lines
    lie in must come in the order of their start lines: an opening that lies only in drives
    before the first drive an opening above it lies in could not run."""
    floor, above = -1, {}
    for line in openings:
        held = [
            at
            for at, drive in enumerate(drives)
            if any(opened_in(line, drive, k) for k in range(len(drive.steps)))
        ]
        if held and held[-1] < floor:
            found.cannot_run(
                f"the opening at {line.get('time_utc')} lies in {drives[held[-1]].name} but is "
                f"logged after the opening at {above.get('time_utc')}, which lies in "
                f"{drives[floor].name}, a drive with a later start line: the drives did not run "
                "in the order of their start lines, a clock set back between them"
            )
            return
        if held and held[0] > floor:
            floor, above = held[0], line


def _stopped(drive: Drive) -> str:
    """How an incomplete drive ended, in words; for a drive that ran to its end with entries that
    failed or steps not run, every one of them is named with its reason."""
    if drive.stop is not None:
        k, label = drive.stop.get("index"), drive.stop.get("label")
        where = "after" if any(begin.get("index") == k for begin, _ in drive.steps) else "before"
        number = k + 1 if isinstance(k, int) else k
        return f"stopped {where} step {number} ({label}): {drive.stop.get('reason') or NO_REASON}"
    text = f"ended with exit status {drive.end.get('exit_status')}"
    if drive.end.get("change_after"):  # what changed after the last step, when something did
        text += f": {drive.end['change_after']}"
    named = []
    for kind, line in drive.problems():
        index = line.get("index")
        number = f"step {index + 1}" if type(index) is int else "step"
        named.append(f"{kind} {number} ({line.get('label')}): {line.get('reason') or NO_REASON}")
    return f"{text}; {'; '.join(named)}" if named else text


def _unfit(s: Survey, drive: Drive) -> str | None:
    """Why `drive` is no drive of the committed list at the checked-out commit with the gate's
    tag and fix commits (the text a mark says), "" when the list could not be read (said
    elsewhere), None when it is one."""
    start = drive.start
    if start.get("commit") != s.head:
        return (
            f"{drive.name} is stale: written at commit {start.get('commit')}, not at the "
            f"checked-out commit {s.head}"
        )
    if s.gate is not None and (start.get("tag_commit"), start.get(FIX_KEY)) != (
        str(s.gate),
        s.gate.fix_commit,
    ):
        return (
            f"{drive.name} records other commits of the tag or the fix than those the gate passes "
            "at now"
        )
    if s.parsed is None:
        return ""
    steps = run_list.instantiate(s.parsed, PURPOSE)
    if (start.get("list"), start.get("list_sha256"), start.get("steps")) != (
        LIST,
        hashlib.sha256(s.data).hexdigest(),
        logged_steps(steps),
    ):
        return (
            f"{drive.name} is not a drive of the bytes of {LIST} committed at the checked-out "
            "commit, with its sealed-run steps as the list instantiates them"
        )
    return None


def _verify_complete(root: Path, s: Survey, drive: Drive, found: Findings) -> list | None:
    """The one complete drive against the committed list at the checked-out commit (module
    text); its steps as logged, or None when it could not run."""
    end = drive.end
    why = _unfit(s, drive)
    if why is not None:
        if why:
            found.cannot_run(why)
        return None
    steps = run_list.instantiate(s.parsed, PURPOSE)
    logged, _ = wp12.logged_entries(steps, drive.records)
    if end.get("unchanged_after") is not True:
        raise CannotCheck(f"the tree changed after the last step of {drive.name}")
    for step, command, line in logged:
        try:
            wp12.verify_run(root, s.head, step, command, line, lambda path: _read(root, path))
        except CannotCheck as problem:
            cause = problem.__cause__
            if isinstance(cause, NeverRead) and str(cause) not in str(problem):  # name the path
                raise CannotCheck(f"{problem}: {cause}") from problem
            raise
    return logged


def abstained_lines(root: Path, steps, found: Findings) -> None:
    """For each held-out truth build step of `steps` ((label, command as run, `entry` line) of the
    complete drive's steps, which exited 0), one line per unit its summary records as abstained,
    with the build's run, the reason and the step; the summary is read through
    `run_verify.read_verified` as O2's results are. An abstained unit is shown as abstained: not
    scored, counted as not passing; no mark and no exit status changes by it. A build step that
    passed with no readable summary could not run, never skipped (module text, mark 1)."""
    for label, command, line in steps:
        named = isinstance(command, list) and any(
            command[k] == "-m" and command[k + 1] == TRUTH_BUILD_STEP
            for k in range(len(command) - 1)
        )
        run = line.get("run")
        if not named or not wp12.passed(line) or not isinstance(run, str):
            continue
        where = f"{run}/{TRUTH_SUMMARY_NAME}"
        try:
            listed = wp12.manifest_of(root, line, wp12._reader(root)).get("outputs")
            if not isinstance(listed, dict) or TRUTH_SUMMARY_NAME not in listed:
                found.cannot_run(
                    f"{label}: the held-out truth build step passed but its manifest lists no "
                    f"{TRUTH_SUMMARY_NAME}, so its abstained units cannot be known"
                )
                continue
            _, data = wp12.read_verified(root / run / TRUTH_SUMMARY_NAME, wp12._reader(root))
            summary = json.loads(data)
            recorded = summary.get("abstained") or {}
            reasons = {unit: entry["reason"] for unit, entry in recorded.items()}
        except (
            CannotCheck,
            NotVerified,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ) as problem:
            found.cannot_run(f"{label}: the abstained units cannot be read from {where}: {problem}")
            continue
        record = {"truth_run": Path(run).name, "abstained": reasons}
        for text in sealed_route.abstention_lines(record):
            found.say(f"{text} (step {label})")


def is_learn(command) -> bool:
    """Whether a step's command as run is a learn step (module text, mark 1)."""
    return isinstance(command, list) and any(
        command[k] == "-m" and command[k + 1] == LEARN_STEP for k in range(len(command) - 1)
    )


def _failed_in_rungs(document: dict, listed: list[str]) -> list[str]:
    """The failed configurations that `learn.json` records only in its units' rungs, as the
    `<configuration> | <unit>: <error>` texts of `failed_configurations`: that list names one
    configuration of each group that failed together (`gate_run`), while every configuration of
    the group holds the reason `failed: <error>` in its rungs (`learn_eval`), and so does an
    oracle partner that failed. Those already in `listed` are left out."""
    known = {
        (named[1], named[2]) for text in listed if (named := FAILED_CONFIGURATION.fullmatch(text))
    }
    units = document.get("units")
    extra = []
    for unit, entry in (units if isinstance(units, dict) else {}).items():
        configurations = entry.get("configurations") if isinstance(entry, dict) else None
        for name, offer in (configurations if isinstance(configurations, dict) else {}).items():
            if not isinstance(offer, dict):
                continue
            partner = offer.get("oracle_partner")
            partner = partner if isinstance(partner, dict) else {}
            for cell, rungs in (
                (name, offer.get("rungs")),
                (partner.get("name"), partner.get("rungs")),
            ):
                reason = rungs.get("reason") if isinstance(rungs, dict) else None
                if (
                    isinstance(cell, str)
                    and isinstance(reason, str)
                    and reason.startswith(FAILED_START)
                    and (cell, unit) not in known
                ):
                    known.add((cell, unit))
                    extra.append(f"{cell} | {unit}: {reason.removeprefix(FAILED_START).strip()}")
    return extra


def failed_configurations(root: Path, steps, found: Findings) -> None:
    """For each learn step of `steps` ((label, command as run, `entry` line) of the steps that
    exited 0), the failed configurations its `learn.json` lists (`failed_configurations`, read
    through `run_verify.read_verified` as O2's results are): on the held-out route a configuration
    that raised an error on a unit leaves the step's exit status 0 (`learn_run`), so this reading
    is the only place it is found. Each fails the mark, naming the step, the unit, the
    configuration and the error. A learn step that passed with no readable list could not run,
    never skipped (module text, mark 1)."""
    for label, command, line in steps:
        run = line.get("run")
        if not is_learn(command) or not wp12.passed(line) or not isinstance(run, str):
            continue
        where = f"{run}/{LEARN_RESULTS}"
        try:
            listed = wp12.manifest_of(root, line, wp12._reader(root)).get("outputs")
            if not isinstance(listed, dict) or LEARN_RESULTS not in listed:
                raise CannotCheck(f"its manifest lists no {LEARN_RESULTS}")
            _, data = wp12.read_verified(root / run / LEARN_RESULTS, wp12._reader(root))
            document = json.loads(data)
            failed = document["failed_configurations"]
            if not isinstance(failed, list) or not all(isinstance(text, str) for text in failed):
                raise ValueError("failed_configurations is not a list of texts")
            failed = failed + _failed_in_rungs(document, failed)
        except (
            CannotCheck,
            NotVerified,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ) as problem:
            found.cannot_run(
                f"{label}: the failed configurations cannot be read from {where}: {problem}"
            )
            continue
        for text in failed:
            named = FAILED_CONFIGURATION.fullmatch(text)
            what = (
                f"configuration {named[1]} failed on unit {named[2]}: {named[3]}" if named else text
            )
            found.fail(f"learn step {label}: {what} (read from {where})")


def ran_to_its_end(s: Survey) -> Drive | None:
    """The newest drive that ran a step, when it ran to its end (no `stop` line; its end line
    lists the entries that failed and the steps not run): with no complete drive it is the sealed
    run made once, judged by mark 1 alone and no earlier drive (module text); else None."""
    ran = [drive for drive in s.drives if not drive.ran_no_step()]
    return ran[-1] if ran and ran[-1].stop is None else None


def incomplete_information(root: Path, s: Survey, drive: Drive, found: Findings) -> None:
    """Information after the failure of an incomplete drive that ran to its end (module text):
    O2's verdicts, the abstained held-out units and the failed configurations of the learn steps,
    read from the steps that passed and verify at the checked-out commit, each line begun
    "information:". The drive must be of the committed list at the checked-out commit, with the
    gate's tag and fix commits, and its tree unchanged after the last step; a step that failed,
    was not run or does not verify is never read, nor a step whose run is one of those. Nothing
    here changes a mark or an exit status: the lines
    about a result that cannot be read are lines too."""
    note = Findings(lambda text: found.say(f"information: {text}"))
    why = _unfit(s, drive)
    if why is None and drive.end.get("unchanged_after") is not True:
        why = f"the tree changed after the last step of {drive.name}"
    if why is not None:
        if why:
            found.say(
                f"information: no O2 verdict or abstained unit is read from {drive.name}: {why}"
            )
        return
    steps = run_list.instantiate(s.parsed, PURPOSE)
    runs: dict[str, str] = {}  # label -> the run folder of a step that passed and verifies
    read = []  # (label, command as run, `entry` line) of those steps
    for begin, entry in drive.steps:
        index = begin.get("index")
        if type(index) is not int or not 0 <= index < len(steps):
            continue
        step = steps[index]
        if not wp12.passed(entry) or step.label not in (begin.get("label"), entry.get("label")):
            continue
        try:
            command = run_list.fill_runs(step, runs)  # a run of a step not read is no run here
        except KeyError:
            continue
        if begin.get("command") != command or begin.get("max_seconds") != step.max_seconds:
            continue
        try:
            wp12.verify_run(root, s.head, step, command, entry, lambda path: _read(root, path))
        except CannotCheck as problem:
            found.say(
                f"information: step {step.label} passed but does not verify, not read: {problem}"
            )
            continue
        runs[step.label] = entry["run"]
        read.append((step.label, command, entry))
    for text, fault in wp12.o2_lines(root, read):
        note.say(text if fault is None else f"{text} (not read)")
    abstained_lines(root, read, note)
    failed_configurations(root, read, note)


def check_run(root: Path, s: Survey, openings: list[dict] | None, found: Findings) -> None:
    """Mark 1, one logged run (module text)."""
    complete = [d for d in s.drives if d.complete()]
    if len(complete) > 1:
        names = ", ".join(d.name for d in complete)
        found.fail(f"{len(complete)} complete sealed-run drives, {names}: the run is made once")
    if complete:
        first = complete[0]
        for drive in s.drives[s.drives.index(first) + 1 :]:
            if drive.ran_no_step():
                found.say(f"{drive.name}: ran no step (no entry names a run folder), not a run")
            else:
                found.fail(
                    f"{drive.name}, a sealed-run drive after the complete drive {first.name}, "
                    f"ran {_plural(len(drive.steps), 'step')}"
                )
    elif s.unreadable:
        found.cannot_run("no readable log holds a complete drive, and a log could not be read")
    else:
        newest = s.drives[-1]
        found.fail(f"no complete sealed-run drive; the newest, {newest.name}, {_stopped(newest)}")
    inside_complete = 0
    if openings is not None:
        _in_drive_order(s.drives, openings, found)
        for line in openings:
            holders = [d for d in s.drives for k in range(len(d.steps)) if opened_in(line, d, k)]
            if complete and complete[0] in holders:
                inside_complete += 1
            if holders:
                continue
            where = (
                f"the opening at {line['time_utc']} of the {line['set']} file at commit "
                f"{line['commit']} lies inside no step of a logged sealed-run drive"
            )
            if s.unreadable:
                found.cannot_run(f"{where} that could be read")
            else:
                found.fail(where)
        if len(complete) == 1 and inside_complete == 0:
            found.fail(f"the complete drive {complete[0].name} holds no opening of a sealed file")
    if not complete and not s.unreadable and (drive := ran_to_its_end(s)) is not None:
        incomplete_information(root, s, drive, found)
    if len(complete) != 1:
        return
    steps = [(e.get("label"), b.get("command"), e) for b, e in complete[0].steps]
    for text, fault in wp12.o2_lines(root, steps):  # O2's verdicts and own faults (mark 1)
        if fault is None:
            found.say(text)
        else:
            found.fail(text)
    failed_configurations(root, steps, found)  # read first, as O2's verdicts are
    logged = _verify_complete(root, s, complete[0], found)
    if logged is not None:
        abstained_lines(root, steps, found)
    if logged is not None and found.status() == PASSED:
        where = sealed_route.passed_at(str(s.gate), s.gate.fix_commit)
        found.say(
            f"one logged run: {complete[0].name}, the one complete sealed-run drive of {LIST}, at "
            f"{where}: {run_list.entries_text(len(logged))}, each exited 0 with no timeout and "
            f"verifies; {_plural(len(openings), 'opening')} logged, each inside a step of a logged "
            f"sealed-run drive, {inside_complete} inside the complete one"
        )


def _number(drive: Drive, j: int) -> int:
    """The number of the j-th step run in the drive's list of steps: its index plus 1 (a step the
    driver did not start, a `not_run` line, takes a number too)."""
    index = drive.steps[j][0].get("index")
    return index + 1 if type(index) is int else j + 1


def _earlier(drive: Drive, refusals: list[dict], openings: list[dict]) -> tuple[str, str]:
    """How an earlier drive that ran a step counts: ("operational", line) when no opening lies
    in any of its steps; ("layout", line) when it stopped at a step during which a layout refusal
    was logged, with no opening in a step before it; ("fail", text) or ("cannot", text) else."""
    steps = range(len(drive.steps))
    held_by = [(line, [j for j in steps if opened_in(line, drive, j)]) for line in openings]
    held_by = [(line, js) for line, js in held_by if js]
    if not held_by:
        return "operational", (
            f"{drive.name}: {_stopped(drive)}, with no opening of sealed data in any of its "
            "steps: an operational stop"
        )
    k = len(drive.steps) - 1
    entry = drive.steps[k][1]
    label = entry.get("label")
    plain_failure = (
        drive.stop is not None
        and not wp12.passed(entry)
        and isinstance(entry.get("manifest_sha256"), str)
        and entry.get("exit_status") != 0
        and entry.get("timed_out") is False
        and entry.get("interrupted") is None
    )
    opened = f"{_plural(len(held_by), 'opening')} of sealed data in its steps"
    if not plain_failure:
        return "fail", (
            f"{drive.name} {_stopped(drive)}, with {opened}; an earlier drive that opened sealed "
            "data counts only as stopped by a layout refusal"
        )
    held = [line for line in refusals if refused_in(line, openings, drive, k)]
    if not held:
        return "fail", (
            f"{drive.name} {_stopped(drive)}, with {opened}, and no layout refusal following an "
            "opening of that step is logged inside it"
        )
    paired = [paired_opening(line, openings) for line in held]
    at_k = _number(drive, k)
    for j in range(k):
        before = [line for line, js in held_by if j in js and k not in js]
        shared = [line for line, js in held_by if j in js and k in js and line not in paired]
        if before:
            return "fail", (
                f"{drive.name} stopped at step {at_k} ({label}) at a layout refusal, but "
                f"{_plural(len(before), 'opening')} lie inside step {_number(drive, j)}, which "
                "passed before it: results from the sealed files could have been made before the "
                "stop"
            )
        if shared:
            return "cannot", (
                f"{drive.name}: the opening at {shared[0].get('time_utc')} lies inside step "
                f"{_number(drive, j)} and step {at_k}, which share a second and a command, and no "
                "refusal follows it: which step opened it cannot be told"
            )
    sets = sorted({line["set"] for line in held})
    return "layout", (
        f"{drive.name}: stopped at step {at_k} ({label}) at a layout refusal of the "
        f"{', '.join(sets)} file logged at commit {drive.start.get('commit')}"
    )


def check_fixes(root: Path, s: Survey, openings, refusals, found: Findings) -> None:
    """Mark 2, logged parsing fixes (module text)."""
    if openings is not None:
        _in_drive_order(s.drives, openings, found)
    complete = [d for d in s.drives if d.complete()]
    sole = None if complete else ran_to_its_end(s)  # the sealed run made once, judged by mark 1
    earlier = s.drives[: s.drives.index(complete[0])] if complete else list(s.drives)
    if sole is not None:
        earlier.remove(sole)
        found.say(
            f"{sole.name}: the newest drive ran to its end with no complete drive; it is no "
            "earlier drive, and mark 1 judges it"
        )
    line_commits = None
    if s.gate is not None:
        tag = str(s.gate)
        line_commits = {tag, *_git(root, "rev-list", f"{tag}..{s.head}").split()}
    counted = {"layout": 0, "operational": 0}
    for drive in earlier:
        if drive.ran_no_step():
            found.say(f"{drive.name}: ran no step (no entry names a run folder), not a run")
            continue
        start = drive.start
        if line_commits is not None and start.get("tag_commit") != str(s.gate):
            found.fail(
                f"{drive.name} was driven with the tag's commit {start.get('tag_commit')}, not "
                f"{s.gate}: the tag never moves"
            )
        elif line_commits is not None and start.get("commit") not in line_commits:
            found.fail(
                f"{drive.name} was driven at commit {start.get('commit')}, neither the tag's nor "
                "one on the line from the tag to the checked-out commit"
            )
        if refusals is None or openings is None:
            continue
        kind, text = _earlier(drive, refusals, openings)
        if kind == "fail":
            found.fail(text)
        elif kind == "cannot":
            found.cannot_run(text)
        else:
            counted[kind] += 1
            found.say(text)
    if found.status() == PASSED:
        tag, fix = str(s.gate), s.gate.fix_commit
        drives = (
            f"{counted['layout']} earlier drives stopped at a logged layout refusal, "
            f"{counted['operational']} stopped with no opening of sealed data"
        )
        if fix is None:
            found.say(
                f"logged parsing fixes: none, the checked-out commit is the tag's {tag}; {drives}"
            )
        else:
            line = _git(root, "rev-list", "--reverse", f"{tag}..{fix}").split()
            found.say(
                f"logged parsing fixes: the gate accepts the checked-out commit {fix} as a logged "
                f"parsing fix of the tag's commit {tag} (its line, oldest first: "
                f"{', '.join(line)}); {drives}"
            )


def run_named(entry: dict) -> bool:
    """The entry's `run` is `runs.noindex/<UTC start>-<label>` of its own label, as `verify_run`
    requires of the complete drive's: no "..", "." or absolute start can be in it."""
    run, label = entry.get("run"), entry.get("label")
    return (
        isinstance(run, str)
        and isinstance(label, str)
        and LABEL.fullmatch(label) is not None
        and re.fullmatch(rf"{re.escape(RUNS_FOLDER)}/\d{{8}}T\d{{6}}Z-{re.escape(label)}", run)
        is not None
    )


def _in_run(root: Path, run: str, name: str) -> str:
    """The path of `name` in the run folder `run` under `root`, as written: joined as text, so
    the reader sees every part (`Path` would drop a "." and let an absolute `run` replace
    `root`)."""
    return f"{os.fspath(root)}/{run}/{name}"


def check_hashes(root: Path, s: Survey, unhashed: list[str], found: Findings) -> None:
    """Mark 3, logs hashed (module text)."""
    for problem in unhashed:
        found.cannot_run(problem)
    manifests = 0
    for drive in s.drives:
        for k, (_, entry) in enumerate(drive.steps):
            run, digest = entry.get("run"), entry.get("manifest_sha256")
            if not isinstance(run, str) or not isinstance(digest, str):
                continue
            where = f"{drive.name}, step {k + 1} ({entry.get('label')})"
            if not run_named(entry):
                found.cannot_run(
                    f"{where}: the run folder {run!r} is not {RUNS_FOLDER}/<UTC start>-<label> of "
                    "the step's label, not read"
                )
                continue
            data = _bytes_or_link(root, _in_run(root, run, MANIFEST_NAME))
            if isinstance(data, NeverRead):
                found.cannot_run(f"{where}: {data}, not read")
                continue
            if data is None or hashlib.sha256(data).hexdigest() != digest:
                found.fail(
                    f"{where}: the SHA-256 of {run}/{MANIFEST_NAME} is not the one the log records"
                )
                continue
            try:  # read only once its SHA-256 is the logged one
                manifest = json.loads(data)
            except (ValueError, RecursionError):
                manifest = None
            outputs = manifest.get("outputs") if isinstance(manifest, dict) else None
            log = _bytes_or_link(root, _in_run(root, run, LOG_NAME))
            if isinstance(log, NeverRead):
                found.cannot_run(f"{where}: {log}, not read")
                continue
            current = None if log is None else hashlib.sha256(log).hexdigest()
            if not isinstance(outputs, dict) or outputs.get(LOG_NAME) != current or not current:
                found.fail(
                    f"{where}: {run}/{LOG_NAME} is not listed in its manifest with its current "
                    "SHA-256"
                )
                continue
            manifests += 1
    if found.status() == PASSED:
        found.say(
            f"logs hashed: {_plural(len(s.drives), 'sealed-run drive log')}, each log_sha256 "
            f"recomputes; {_plural(manifests, 'manifest')} with the SHA-256 the logs record, each "
            f"listing its {LOG_NAME} with its current SHA-256"
        )


def _bytes_or_link(root: Path, path: Path):
    """A file's bytes by `_read` from `root`: the `LinkOnPath` when a link lies on its path
    (never followed), None when it cannot be read otherwise (absent among them)."""
    try:
        return _read(root, path)
    except NeverRead as problem:
        return problem
    except OSError:
        return None


def _digest(root: Path, path: Path) -> str:
    """The SHA-256 of a file's bytes by `_read` from `root`, "absent" when there is none (no
    file, or no folder, at a name of its path); a link on its path raises `LinkOnPath`."""
    try:
        return hashlib.sha256(_read(root, path)).hexdigest()
    except (FileNotFoundError, NotADirectoryError):
        return "absent"


def inputs(root: Path, s: Survey | None) -> tuple[dict[str, str], list[str]]:
    """({path under `root`: its SHA-256 or "absent"}, a line for each input that cannot be
    hashed): the committed list's bytes, every drive log, each run's manifest and log, the
    openings and refusal logs."""
    root, table, problems = Path(root), {}, []
    paths = []
    if s is not None:
        table[LIST] = "absent" if s.data is None else hashlib.sha256(s.data).hexdigest()
        paths += [(path.relative_to(root).as_posix(), path) for path in s.paths]
        for drive in s.drives:
            for _, entry in drive.steps:
                run = entry.get("run")
                if not isinstance(run, str):
                    continue
                if not run_named(entry):
                    table[run] = "cannot be hashed"
                    problems.append(
                        f"the inputs in {run!r} cannot be hashed: not {RUNS_FOLDER}/<UTC start>-"
                        "<label> of its step's label, not read"
                    )
                    continue
                paths += [(f"{run}/{n}", _in_run(root, run, n)) for n in (MANIFEST_NAME, LOG_NAME)]
    paths += [(path.relative_to(root).as_posix(), path) for path in sealed_route.sealed_logs(root)]
    for name, path in paths:
        try:
            table[name] = _digest(root, path)
        except OSError as problem:
            table[name] = "cannot be hashed"
            problems.append(f"the input {name} cannot be hashed: {problem}")
    return table, problems


def _read_logs(root: Path) -> tuple[list | None, list | None, list[str]]:
    """(the openings, the refusals, a line for each log that cannot be read); a log that cannot
    be read is None."""
    read, problems = [], []
    for path, parse, what in zip(
        sealed_route.sealed_logs(root),
        (sealed_route.openings_in, sealed_route.refusals_in),
        ("openings", "refusal"),
        strict=True,
    ):
        try:
            try:
                content = _read(root, path)
            except (FileNotFoundError, NotADirectoryError):
                content = b""  # an absent log holds no line, as the loader reads it
            read.append(parse(content))
        except (OSError, ValueError) as problem:
            problems.append(f"the {what} log cannot be read: {type(problem).__name__}: {problem}")
            read.append(None)
    return read[0], read[1], problems


def _mark(first: list[str], cannot: bool) -> Findings:
    """A mark's findings, holding its lines, which begin with the survey's."""
    lines = list(first)
    found = Findings(lines.append)
    found.lines, found.cannot = lines, cannot
    return found


def check(root: Path, say=print) -> int:
    """The three marks, their results file and the exit status (module text)."""
    root = Path(root)
    folder = gate_mark.run_folder(root)  # the gate mark's rule: a recorded run going in `root`
    if folder is None:
        say(
            f"COULD NOT RUN: not under the run recorder ({RUN_DIR_VARIABLE} names no folder "
            f"directly in {RUNS_FOLDER} of {root} that holds no {MANIFEST_NAME} yet), so no "
            "results file can be written"
        )
        return COULD_NOT_RUN
    first = []
    begun = Findings(first.append)
    try:
        s = survey(root, begun)
    except CannotCheck as problem:
        begun.cannot_run(str(problem))
        s = None
    marks = [_mark(first, begun.cannot) for _ in MARKS]
    table, unhashed = {}, []
    try:
        table, unhashed = inputs(root, s)
    except Exception as problem:  # never Python's own 1
        unhashed = [f"the inputs cannot be hashed: {type(problem).__name__}: {problem}"]
    if s is not None and s.paths:
        openings, refusals, unread = _read_logs(root)
        for found, needs, job in (
            (marks[0], unread, lambda f: check_run(root, s, openings, f)),
            (marks[1], unread, lambda f: check_fixes(root, s, openings, refusals, f)),
            (marks[2], [], lambda f: check_hashes(root, s, unhashed, f)),
        ):
            for problem in needs:
                found.cannot_run(problem)
            try:
                job(found)
            except CannotCheck as problem:
                found.cannot_run(str(problem))
            except Exception as problem:  # never Python's own 1; earlier failures still decide
                found.cannot_run(f"{type(problem).__name__}: {problem}")
    statuses = [found.status() for found in marks]
    status = FAILED if FAILED in statuses else max(statuses)
    record = {
        "check": "tools/sealed_run_check.py, the pass mark of work package 14 (plan section 13)",
        "commit": None if s is None else s.head,
        "tag_commit": None if s is None or s.gate is None else str(s.gate),
        "fix_commit": None if s is None or s.gate is None else s.gate.fix_commit,
        "status": WORDS[status],
        "marks": [
            {"mark": name, "status": WORDS[found.status()], "lines": found.lines}
            for name, found in zip(MARKS, marks, strict=True)
        ],
        "inputs": table,
        "not_seen": NOT_SEEN,
    }
    for line in first:
        say(line)
    for name, found in zip(MARKS, marks, strict=True):
        for line in found.lines[len(first) :]:
            say(line)
        say(f"{name}: {WORDS[found.status()]}")
    results = Path(folder) / RESULTS_NAME
    try:
        results.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError as problem:
        say(f"COULD NOT RUN: the results file {results} cannot be written: {problem}")
        return FAILED if status == FAILED else COULD_NOT_RUN
    say(f"results: {results}; {NOT_SEEN}")
    return status


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check WP14's pass mark.")
    parser.add_argument("--root", type=Path, default=TOP, help="project top folder")
    args = parser.parse_args(argv)  # a wrong call exits with status 2
    try:
        return check(args.root)
    except Exception as problem:
        print(f"COULD NOT RUN: {type(problem).__name__}: {problem}")
        return COULD_NOT_RUN


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
