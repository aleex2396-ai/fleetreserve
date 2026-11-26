"""The sealed-run drive that WP16 reads, and the files of its runs (plan section 13, WP16).

WP16 fills the slots of plan section 11 from one drive log of the run from the tag,
`runs.noindex/drives/<UTC start>-sealed-run.jsonl`, as `tools/sealed_run_check.py` finds it.

`find_drive(root)` reads every drive log of that purpose through `plain_read` (no link, no "..",
no "." and no absolute part of a path is opened) and takes:
- the one drive that is complete (its end line says it completed with exit status 0, it has no
  `stop` line and every step of the list ran and passed), when exactly one is; a later drive that
  ran a step is then a refusal, as in the WP14 check;
- else the newest drive that ran a step and ran to its end (no `stop` line): its slots are filled
  from the steps that passed, `Drive.complete` is False, and a step that failed or did not run
  stays named as such (plan section 13, WP14: "the results of the steps that passed are reported
  with those names beside them, never as a passed held-out run");
- else `NoDrive`: no drive, a drive that stopped or broke off, two complete drives.

A drive log is taken only when its end line's `log_sha256` recomputes (`run_driver.end_digest`).

`Source` reads a step's output file: the step must have passed, its manifest must have the
SHA-256 the drive log records, name the drive's commit, exit code 0 and no time-out or
interruption, list the file, and the file must have the SHA-256 the manifest lists. Every file
read, by its path under the top folder, is recorded with its SHA-256 (`Source.hashes`), for the
manifest of the outputs. A file that cannot be read for any reason raises `Missed` with the text
that goes into the slot ("failed: <step> (<reason>)", "not run: <step> (<reason>)" or
"unreadable: ..."); a slot is never filled from elsewhere (plan section 13, WP14, last bullets).

Nothing here opens a file of the sealed folder, and the held-out data are never read: only the
drive log, the manifests and the results files of the runs it names, and files committed in the
repository (the run list, the matrices).
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from fleetreserve import plain_read, run_list
from fleetreserve.run_driver import DRIVES_FOLDER, LOG_FORMAT, LOG_SUFFIX, end_digest
from fleetreserve.run_record import MANIFEST_NAME, RUNS_FOLDER
from fleetreserve.run_verify import plain_name

PURPOSE = run_list.SEALED_RUN
LOG_PATTERN = re.compile(r"\d{8}T\d{6}Z-" + re.escape(PURPOSE) + re.escape(LOG_SUFFIX))
LIST_FILE = "run-lists/run-list.json"


class NoDrive(Exception):
    """No drive log can be taken as the run from the tag (the message says why)."""


class Missed(Exception):
    """A file a slot needs cannot be read; `text` is what the slot says instead of a figure."""

    def __init__(self, text: str):
        super().__init__(text)
        self.text = text


@dataclass(frozen=True)
class Step:
    """One step of a drive: `status` is "passed", "failed" or "not run"; `reason` says why a
    step did not pass; `run` and `manifest_sha256` are those the log records (None for a step
    that did not run)."""

    label: str
    index: int
    command: list | None
    status: str
    reason: str
    run: str | None
    manifest_sha256: str | None

    def text(self) -> str:
        return f"{self.status}: {self.label} ({self.reason})"


@dataclass
class Drive:
    name: str
    log_sha256: str
    commit: str
    tag_commit: str | None
    complete: bool
    steps: dict[str, Step] = field(default_factory=dict)
    stopped: bool = False
    ran: bool = False  # at least one step was begun
    list_sha256: str | None = None  # the run list's SHA-256 the drive's start line records

    def problems(self) -> list[str]:
        """The failed and not-run steps, in the list's order, as the slots name them."""
        return [s.text() for s in self.steps.values() if s.status != "passed"]


def _passed(entry: dict) -> bool:
    return (
        entry.get("exit_status") == 0
        and entry.get("timed_out") is False
        and entry.get("interrupted") is None
        and isinstance(entry.get("manifest_sha256"), str)
    )


def _failure_reason(entry: dict) -> str:
    if not isinstance(entry.get("manifest_sha256"), str):
        return f"no readable manifest (the recorder ended with {entry.get('recorder_status')})"
    if entry.get("timed_out"):
        return f"it hit its time limit (exit status {entry.get('exit_status')})"
    if entry.get("interrupted") is not None:
        return f"it was interrupted by {entry.get('interrupted')}"
    return f"it ended with exit status {entry.get('exit_status')}"


def parse_log(name: str, data: bytes) -> Drive:
    """The drive a log's bytes record; raises `NoDrive` when they are not a readable drive log
    (an empty or cut-off log, a line that is not JSON, no start line of the format, no end line,
    an end line whose hash does not recompute)."""
    lines = data.splitlines(keepends=True)
    if not lines or not data.endswith(b"\n"):
        raise NoDrive(f"the log {name} is empty or its last line is cut off")
    try:
        records = [json.loads(line) for line in lines]
    except ValueError as problem:
        raise NoDrive(f"a line of the log {name} is not JSON: {problem}") from problem
    if not all(isinstance(r, dict) for r in records):
        raise NoDrive(f"a line of the log {name} is not a JSON object")
    start, end = records[0], records[-1]
    if start.get("record") != "start" or start.get("format") != LOG_FORMAT:
        raise NoDrive(f"the log {name} does not begin with a {LOG_FORMAT} start line")
    if start.get("purpose") != PURPOSE:
        raise NoDrive(f"the log {name} is not of the purpose {PURPOSE}")
    if end.get("record") != "end":
        raise NoDrive(f"the log {name} has no end line: the drive broke off or is still going")
    if end.get("log_sha256") != end_digest(b"".join(lines[:-1]), end):
        raise NoDrive(f"the hash of the log {name} does not recompute")
    listed = start.get("steps")
    if not isinstance(listed, list) or not all(isinstance(s, dict) for s in listed):
        raise NoDrive(f"the start line of {name} lists no steps")
    labels = [s.get("label") for s in listed]
    commands = {s.get("label"): s.get("command") for s in listed}
    steps: dict[str, Step] = {}
    begun: dict[int, dict] = {}
    stopped = False
    for record in records[1:-1]:
        kind, label, index = record.get("record"), record.get("label"), record.get("index")
        if kind == "stop":
            stopped = True
        elif kind == "begin":
            begun[index] = record
        elif kind == "entry":
            command = begun.get(index, {}).get("command")
            ok = _passed(record)
            run, digest = record.get("run"), record.get("manifest_sha256")
            steps[label] = Step(
                label,
                index,
                command if isinstance(command, list) else None,
                "passed" if ok else "failed",
                "" if ok else _failure_reason(record),
                run if isinstance(run, str) else None,
                digest if isinstance(digest, str) else None,
            )
        elif kind == "failed" and label in steps:  # the driver's own reason for the failure
            reason = record.get("reason")
            if isinstance(reason, str):
                steps[label] = replace(steps[label], status="failed", reason=reason)
        elif kind == "not_run":
            reason = record.get("reason")
            text = reason if isinstance(reason, str) else "no reason logged"
            steps[label] = Step(label, index, None, "not run", text, None, None)
    for k, label in enumerate(labels):  # a step the log has no line for (a drive that stopped)
        if label not in steps:
            steps[label] = Step(
                label, k, None, "not run", "the drive stopped before it", None, None
            )
        elif steps[label].command is None and isinstance(commands.get(label), list):
            steps[label] = replace(steps[label], command=commands[label])
    ordered = {label: steps[label] for label in labels}
    complete = (
        end.get("completed") is True
        and end.get("exit_status") == 0
        and not stopped
        and end.get("entries_run") == start.get("entries") == len(labels)
        and all(s.status == "passed" for s in ordered.values())
    )
    return Drive(
        name=name,
        log_sha256=hashlib.sha256(data).hexdigest(),
        commit=str(start.get("commit")),
        tag_commit=start.get("tag_commit") if isinstance(start.get("tag_commit"), str) else None,
        complete=complete,
        steps=ordered,
        stopped=stopped,
        ran=bool(begun),
        list_sha256=start.get("list_sha256") if isinstance(start.get("list_sha256"), str) else None,
    )


def find_drive(root: Path) -> Drive:
    """The drive WP16 reads (module text); raises `NoDrive`."""
    folder = Path(root) / RUNS_FOLDER / DRIVES_FOLDER
    try:
        names = sorted(n for n in plain_read.listing(root, folder) if LOG_PATTERN.fullmatch(n))
    except (FileNotFoundError, NotADirectoryError):
        names = []
    except OSError as problem:
        raise NoDrive(f"the folder of the drive logs cannot be listed: {problem}") from problem
    if not names:
        raise NoDrive(f"no sealed-run drive log in {RUNS_FOLDER}/{DRIVES_FOLDER}")
    drives, unreadable = [], []
    for name in names:
        try:
            drives.append(parse_log(name, plain_read.read(root, folder / name)))
        except (NoDrive, OSError) as problem:
            unreadable.append(f"{name}: {problem}")
    ran = [d for d in drives if d.ran]
    complete = [d for d in ran if d.complete]
    if len(complete) > 1:
        raise NoDrive(f"{len(complete)} complete sealed-run drives: which is the run is not known")
    if complete:
        if ran[-1] is not complete[0]:
            raise NoDrive(f"{ran[-1].name} ran a step after the complete drive {complete[0].name}")
        return complete[0]
    if not ran:
        more = f" ({'; '.join(unreadable)})" if unreadable else ""
        raise NoDrive(f"no sealed-run drive ran a step{more}")
    newest = ran[-1]
    if newest.stopped:
        raise NoDrive(f"the newest drive that ran a step, {newest.name}, stopped: nothing is read")
    return newest


class Source:
    """The files a drive's runs hold, and the committed files, read once each and recorded
    (module text)."""

    def __init__(self, root: Path, drive: Drive):
        self.root = Path(root)
        self.drive = drive
        self.hashes: dict[str, str] = {}

    def why_not(self, label: str) -> str | None:
        """None when the step passed, else the text a slot of it says."""
        step = self.drive.steps.get(label)
        if step is None:
            return f"no entry writes it: the drive has no step {label}"
        return None if step.status == "passed" else step.text()

    def _read(self, relative: str) -> bytes:
        data = plain_read.read(self.root, self.root / relative)
        self.hashes[relative] = hashlib.sha256(data).hexdigest()
        return data

    def committed(self, relative: str) -> bytes:
        """A file of the repository (the run list, a matrix), by its path under the top folder."""
        try:
            return self._read(relative)
        except OSError as problem:
            raise Missed(f"unreadable: {relative}: {problem}") from problem

    def output(self, label: str, name: str) -> bytes:
        """The file `name` of the run of the step `label`, once it verifies (module text)."""
        reason = self.why_not(label)
        if reason is not None:
            raise Missed(reason)
        step = self.drive.steps[label]
        where = f"{label}/{name}"
        pattern = rf"{re.escape(RUNS_FOLDER)}/\d{{8}}T\d{{6}}Z-{re.escape(label)}"
        if step.run is None or not re.fullmatch(pattern, step.run) or not plain_name(name):
            raise Missed(f"unreadable: {where}: the drive log names no run folder of that label")
        try:
            manifest_bytes = plain_read.read(self.root, self.root / step.run / MANIFEST_NAME)
            manifest = json.loads(manifest_bytes)
            if hashlib.sha256(manifest_bytes).hexdigest() != step.manifest_sha256:
                raise Missed(f"unreadable: {where}: its manifest is not the one the log records")
            outputs = manifest.get("outputs")
            ok = (
                manifest.get("commit") == self.drive.commit
                and manifest.get("exit_code") == 0
                and manifest.get("timed_out", False) is False
                and "interrupted" not in manifest
                and isinstance(outputs, dict)
                and name in outputs
            )
            if not ok:
                raise Missed(f"unreadable: {where}: the manifest does not list it as a clean run")
            data = plain_read.read(self.root, self.root / step.run / name)
        except (OSError, ValueError, AttributeError) as problem:
            raise Missed(f"unreadable: {where}: {problem}") from problem
        digest = hashlib.sha256(data).hexdigest()
        if digest != outputs[name]:
            raise Missed(f"unreadable: {where}: it is not the file its run wrote")
        self.hashes[f"{step.run}/{name}"] = digest
        return data

    def json(self, label: str, name: str, *, committed: bool = False) -> dict:
        """A JSON object: an output of a step, or (with `committed`) a committed file."""
        data = self.committed(name) if committed else self.output(label, name)
        try:
            document = json.loads(data)
        except (ValueError, RecursionError) as problem:
            raise Missed(f"unreadable: {label}/{name}: not JSON: {problem}") from problem
        if not isinstance(document, dict):
            raise Missed(f"unreadable: {label}/{name}: not a JSON object")
        return document
