"""The WP12 pass-mark item, read from the committed run list and the newest driver log of its
rehearsal (plan section 13, WP12: "Dress rehearsal of held-out scripts on development units.
Pass: done, its duration logged").

Run as `python tools/rehearsal_check.py [--top <folder>]`.

THE LIST, at one fixed path (as the WP8 check reads the fixed `gate/matrix.json`):
`run-lists/run-list.json`, the one list from which both the rehearsal and the run from the tag are
derived (`fleetreserve.run_list`). It must be committed at the checked-out commit and pass the
checker; with no file committed there the item could not run, with a `NO RUN LIST` line. What the
rehearsal runs is the list's rehearsal instantiation (`run_list.instantiate`, reading L9 of
`run_list`), and the check prints, one line each, every difference between it and the run from the
tag as the checker lists them (`run_list.differences`: units, entries of one purpose only,
arguments of one purpose only; reading R6).
Beside each O2 verdict line the check prints the units' own faults, each count named by what it
holds (orders broken, both products on every day, with those that are block orders on M days; days
not built, with those that are M days; only the latter of each are in O2's pass), or says that a
unit's counts are missing or not readable; no verdict and no exit status changes (decision 88
(3d)).

THE LOG read is the newest file `runs.noindex/drives/<UTC start>-rehearsal.jsonl` (by name, so by
start time) whose start line names the list's fixed path (reading R5); an older log of that list
is never used in its place, so a newer drive that failed or broke off is not hidden by an older one
that passed. It is written by
    caffeinate -i uv run --frozen --group learn --group baselines python -m
        fleetreserve.run_driver run-lists/run-list.json --purpose rehearsal
(`fleetreserve.run_driver`, whose module text gives the log's form; the two groups because its
start check loads every model the list's learning steps name). The logs' folder, the log,
each manifest and each output a manifest lists are read through `fleetreserve.plain_read`: a
symbolic link on the path, or a "..", "." or absolute part of it as written, is never opened, and
the item could not run; so too an output name that is not plain (`run_verify.plain_name`). The
log decides nothing (could not run) unless:
- it is a file, not a symbolic link (reading R5);
- every line is a JSON object, the first a `start` of the driver's log format and the last an
  `end`, and the end's `log_sha256` equals the SHA-256 of every byte above it followed by the
  end's own fields (`run_driver.end_digest`: the log's hash recomputes, and covers the total
  duration; a log without its end, from a drive that broke off or is still going, cannot be
  read);
- it was written at the commit checked out now, or at an earlier commit that
  `run_evidence` accepts (reading R1; else it is stale);
- it records the SHA-256 of the list's committed bytes, the purpose `rehearsal`, and the
  rehearsal's steps as the list instantiates them, so only a rehearsal drive of exactly that
  file's bytes can pass;
- its lines follow those steps: for each in order a `begin` with the step's index, label, command
  as run (each `{run:<label>}` filled in with the run folder the log records for that earlier
  step) and time limit, then an `entry` with its index and label, up to the last step or to a
  `stop`.

From such a log:
- FAILED (1): a step's recorded run ended with an exit status other than 0, timed out or was
  interrupted (reading R2), as its `entry` line says and as its manifest, with the SHA-256 the
  line records, confirms;
- passed (0): every step of the rehearsal ran, in order, and exited 0 with no timeout; the end
  says the commit and the clean tree were unchanged after the last step and the drive completed
  with exit status 0; the total duration is logged (reading R3); and each step's recorded run
  verifies at the commit the log was written at (reading R4): its folder is
  `runs.noindex/<UTC start>-<label>`, its manifest has the SHA-256 the log records, that commit,
  the step's command as run,
  `long` true and the step's time limit, and every output file it lists verifies
  (`fleetreserve.run_verify.require_verified`: exit code 0, neither timed out nor interrupted, the
  file's SHA-256 as recorded);
- could not run (2) otherwise: the drive stopped before a step (the commit or the tree changed)
  or after a step the recorder refused, the commit or the tree changed after the last step, a run
  folder or manifest missing or changed, a run that does not verify, no total duration.

Readings (alternatives in brackets):

  R1. "Done" is one rehearsal drive of the committed list `run-lists/run-list.json` to its end,
      every step passing, at the checked-out commit or at an earlier commit C that the rule of
      `run_evidence` accepts, decided by its own function (`_not_earlier`, the one
      `read_evidence` uses; that module's text, points 1 to 3 and "The premise"): C is a commit
      and an ancestor of the checked-out commit; no word of any rehearsal step's command, as the
      list instantiates it, names a place of the keep-list `UNREAD_BY_RUNS` (its reading E1); and
      every committed path that differs between C and the checked-out commit lies under the
      keep-list. Since `run-lists/` is not on it, the list's bytes at C are those committed now,
      which the log's start must record as before. Such a log's line is followed by one naming it
      with `run_evidence.earlier_note` of C, every later line naming the log carries the same
      words, and each step's run is verified at C, the commit its manifest must record. Point 4
      of that rule (a run through the sealed-data gate) is not asked: a rehearsal drives
      development units and passes no gate, and a sealed run's log is refused by its purpose.
      [Alternatives: only a drive at the checked-out commit, so that any later commit, docs only
      too, makes the log stale; or a drive at an earlier commit whose list bytes are unchanged,
      whatever else changed.]
  R2. A recorded run that ended with an exit status other than 0, timed out or was interrupted is
      a failed step (1): the rehearsal ran the step and it did not succeed, whatever the step's
      own reading of its status. A step the recorder refused (no run folder: another heavy run
      held the lock, say) is could not run (2): the step never ran. [Alternative: read a step's
      own exit status 2 as could not run, as `passmark.sh` reads an item's.]
  R3. "Its duration logged" is the total duration of the drive in the log's end (`seconds`, a
      finite number >= 0, covered by the end's hash, so an edited total does not recompute), with
      each step's duration in its `entry` line; the check prints both. The hash guards against
      an edit by accident only: whoever edits the log can recompute it. [Alternative: the time
      from the first step's start to the last step's end, recomputed from the UTC stamps.]
  R4. "Verifies" covers every output file each run's manifest lists, not one results file, since
      a step may be any command of the package. [Alternative: verify `log.txt` alone.]
  R5. The log read is the newest one whose start names `run-lists/run-list.json`: a drive of
      another list (a shorter one, say) is never read in its place, and does not hide the newest
      drive of the committed list either. A newer log whose first line cannot be read stops the
      check (could not run), since it cannot be told whose drive it was, and so does a log that
      is a symbolic link (a passing log linked in under a newer name would otherwise hide the
      newest drive). The logs live in the ignored runs folder, outside git: deleting the newest
      log makes an older one the one read, and whoever can write there can change what is read.
      [Alternative: read the newest rehearsal log of any list and answer could not run when it is
      not of the committed list, so another list's drive hides an earlier failure as 2.]
  R6. "The held-out scripts on development units" are the committed list's rehearsal
      instantiation (reading L9 of `run_list`): the run from the tag is derived from the same
      list, so the two run the same scripts by construction, and can differ only as the closed
      list of `run_list` reading L8 allows (the units of a group, entries of one purpose only,
      arguments of one purpose only). The check prints every such difference and its number, and
      judges none of them: whether each is the one meant is for the reader. The run from the tag
      itself is WP14's, not checked here. [Alternatives: fail the item when the run from the tag
      has an entry the rehearsal lacks; or compare two committed lists entry by entry, the form
      before this one, which no real pair could pass.]

  R7. Outcome O2's verdict (the owner's decision of 19 November 2026, `ceiling_report` reading
      O2-4: an O2 step exits 0 whatever its pass, the verdict in its results file alone). An O2
      step is a logged step whose command as run calls `-m fleetreserve.ceiling_report` with
      `--every-day`, or a shortening of it that `argparse` reads as it, before any `--`. For each
      O2 step logged, one line `O2 <label>:` gives, from its `ceiling-report.json`, the verdict
      over the units (`pass_over_units`), the units whose pass failed and those where it could not
      run (could not decide), and, when there are any, the units with no day to check (none of
      their M days meets T4's hypotheses; such a unit counts as neither passed nor failed, the
      owner's decision of 19 November 2026, `ceiling_summary` reading N). The file is read only
      through `run_verify.read_verified` (the run
      verifies, the bytes read are those its manifest lists, through `fleetreserve.plain_read`: no
      symbolic link, "..", "." or absolute part is opened), from
      the run folder the log records, once that folder is named `runs.noindex/<UTC start>-<label>`.
      The verdict never enters the exit status; but an O2 step that exited 0 with no timeout whose
      results file holds no readable verdict (the file missing or not verifying, not JSON, no
      `every_day` with a pass of passed, failed, could not run or no day to check for each of its
      units, or a
      `pass_over_units` other than `ceiling_summary.pass_over` of them) fails the check (1), named
      on the last line, whatever else the check finds. An O2 step that did not exit 0 gets a line
      saying no verdict is read; its exit status is judged as any step's. [Alternatives: O2's
      verdict in the exit status, which decision 74 took out of the O2 step's; or every
      `ceiling_report` step read, which would fail a block-sample step, whose results hold no O2
      verdict.]

Output: a first line naming what is read; one line per difference between the purposes (or one
saying there is none); a line naming the log; one line per step logged (label, run folder, exit
status, timeout, duration); one line per O2 step logged (reading R7); a last line `passed:`,
`FAILED:`, `NO RUN LIST:`, `NO REHEARSAL LOG:` or `COULD NOT RUN:` with the reason.

Exit status: 0 passed; 1 failed (a logged step, or an O2 step whose results file holds no readable
verdict, reading R7); 2 could not run (no run list, no log, a stale log, or anything it cannot
decide by the list above). An unexpected error is could not run too, never 1
(Python's own exit status for an uncaught error is 1, which `passmark.sh` reads as failed), also
when the tool's own imports fail.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

try:
    from fleetreserve import plain_read, run_list
    from fleetreserve.ceiling_report import RESULTS_NAME as O2_RESULTS
    from fleetreserve.ceiling_summary import FAILED as O2_FAILED
    from fleetreserve.ceiling_summary import NO_DAY as O2_NO_DAY
    from fleetreserve.ceiling_summary import NOT_RUN as O2_NOT_RUN
    from fleetreserve.ceiling_summary import PASSED as O2_PASSED
    from fleetreserve.ceiling_summary import no_day_units, pass_over
    from fleetreserve.run_driver import (
        DRIVES_FOLDER,
        LOG_FORMAT,
        LOG_SUFFIX,
        end_digest,
        logged_steps,
    )
    from fleetreserve.run_evidence import CannotCheck, _not_earlier, earlier_note, head_commit
    from fleetreserve.run_record import MANIFEST_NAME, RUNS_FOLDER
    from fleetreserve.run_verify import NotVerified, plain_name, read_verified, require_verified
except Exception as problem:  # the check cannot start: could not run, never Python's exit 1
    print(f"COULD NOT RUN: the check cannot start: {type(problem).__name__}: {problem}")
    sys.exit(2)

PURPOSE = run_list.REHEARSAL
LIST = "run-lists/run-list.json"  # the one fixed path of the run list
PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2
HEADER = f"WP12: the rehearsal drive of the committed run list {LIST}"
O2_MODULE, O2_OPTION = "fleetreserve.ceiling_report", "--every-day"  # an O2 step (reading R7)
O2_WORDS = (O2_PASSED, O2_FAILED, O2_NOT_RUN, O2_NO_DAY)


class NoList(Exception):
    """No run list is committed at its fixed path (exit status 2)."""


class NoLog(Exception):
    """No rehearsal driver log of the committed list exists (exit status 2)."""


def committed_bytes(top: Path, path: str) -> bytes | None:
    """The bytes of `path` as committed at the checked-out commit; None when it is not there."""
    done = subprocess.run(
        ["git", "-C", str(top), "show", f"HEAD:{path}"], capture_output=True, check=False
    )
    return done.stdout if done.returncode == 0 else None


def committed_list(top: Path, data: bytes) -> run_list.RunList:
    """The run list of the committed bytes `data`, once it passes the checker; raises
    `CannotCheck`."""
    try:
        return run_list.parse(run_list.decode(data), name_max=run_list.run_folder_limit(top))
    except run_list.Faulty as problem:
        raise CannotCheck(f"the run list {LIST} is faulty: {problem.faults[0]}") from problem
    except run_list.CannotDecide as problem:
        raise CannotCheck(f"the run list {LIST} cannot be decided: {problem}") from problem


def newest_log(top: Path) -> Path:
    """The newest rehearsal driver log of the committed list (reading R5); raises `NoLog` or
    `CannotCheck`."""
    folder = top / RUNS_FOLDER / DRIVES_FOLDER
    pattern = re.compile(r"\d{8}T\d{6}Z-" + re.escape(PURPOSE) + re.escape(LOG_SUFFIX))
    try:
        names = plain_read.listing(top, folder)
    except plain_read.NeverRead as problem:
        raise CannotCheck(f"the folder of the drive logs is not listed: {problem}") from problem
    except (FileNotFoundError, NotADirectoryError):
        names = []
    except OSError as problem:
        raise CannotCheck(f"the folder of the drive logs cannot be listed: {problem}") from problem
    found = sorted(
        folder / name
        for name in names
        if pattern.fullmatch(name) and ((folder / name).is_symlink() or (folder / name).is_file())
    )
    for path in reversed(found):
        if path.is_symlink():
            raise CannotCheck(
                f"the log {path.name} is a symbolic link, so the drive it records cannot be told "
                "from a copy of another (reading R5)"
            )
        try:
            start = json.loads(plain_read.read(top, path).split(b"\n", 1)[0])
        except (OSError, ValueError) as problem:
            raise CannotCheck(
                f"the first line of the log {path.name} cannot be read, so whose drive it was "
                "cannot be told"
            ) from problem
        if not isinstance(start, dict):
            raise CannotCheck(f"the first line of the log {path.name} is not a JSON object")
        if start.get("list") == LIST:
            return path
    raise NoLog(
        f"no rehearsal driver log of the committed run list {LIST}, "
        f"{RUNS_FOLDER}/{DRIVES_FOLDER}/<UTC start>-{PURPOSE}{LOG_SUFFIX}"
    )


def read_log(path: Path, top: Path | None = None) -> list[dict]:
    """The log's lines, once its hash recomputes (module text), read by `plain_read.read` from
    `top` (by default the log's folder); raises `CannotCheck`."""
    try:
        data = plain_read.read(path.parent if top is None else top, path)
    except OSError as problem:
        raise CannotCheck(f"the log {path.name} cannot be read: {problem}") from problem
    return log_records(path.name, data)


def log_records(name: str, data: bytes) -> list[dict]:
    """The lines of the log named `name` from its bytes `data`, once its hash recomputes, as
    `read_log` reads them; reads no file; raises `CannotCheck`."""
    lines = data.splitlines(keepends=True)
    if not lines or not data.endswith(b"\n"):
        raise CannotCheck(f"the log {name} is empty or its last line is cut off")
    try:
        records = [json.loads(line) for line in lines]
    except ValueError as problem:
        raise CannotCheck(f"a line of the log {name} is not JSON: {problem}") from problem
    if not all(isinstance(r, dict) for r in records):
        raise CannotCheck(f"a line of the log {name} is not a JSON object")
    first, last = records[0], records[-1]
    if first.get("record") != "start" or first.get("format") != LOG_FORMAT:
        raise CannotCheck(f"the log {name} does not begin with a {LOG_FORMAT} start line")
    if last.get("record") != "end":
        raise CannotCheck(f"the log {name} has no end line: the drive broke off or is still going")
    if last.get("log_sha256") != end_digest(b"".join(lines[:-1]), last):
        raise CannotCheck(f"the hash of the log {name} does not recompute")
    return records


def logged_entries(
    steps: tuple[run_list.Step, ...], records: list[dict]
) -> tuple[list[tuple[run_list.Step, list[str], dict]], dict | None]:
    """[(step, its command as run, its `entry` line)] in the steps' order, and the `stop` line or
    None, once the lines follow the steps (module text)."""
    body = records[1:-1]
    found, at = [], 0
    runs: dict[str, str] = {}  # label -> the run folder the log records for it
    for k, step in enumerate(steps):
        if at < len(body) and body[at].get("record") == "stop":
            break
        begin = body[at] if at < len(body) else {}
        line = body[at + 1] if at + 1 < len(body) else {}
        try:
            command = run_list.fill_runs(step, runs)
        except KeyError as problem:
            raise CannotCheck(
                f"the log records no run folder of {problem.args[0]}, whose run step {k + 1} "
                f"({step.label}) names"
            ) from problem
        follows = (
            begin.get("record") == "begin"
            and begin.get("index") == k
            and begin.get("label") == step.label
            and begin.get("command") == command
            and begin.get("max_seconds") == step.max_seconds
            and line.get("record") == "entry"
            and line.get("index") == k
            and line.get("label") == step.label
        )
        if not follows:
            raise CannotCheck(
                f"the log does not follow the run list at step {k + 1} ({step.label})"
            )
        found.append((step, command, line))
        at += 2
        if not passed(line):
            break
        if isinstance(line.get("run"), str):
            runs[step.label] = line["run"]
    rest = body[at:]
    if len(rest) > 1 or (rest and rest[0].get("record") != "stop"):
        raise CannotCheck("the log holds lines the run list does not account for")
    return found, (rest[0] if rest else None)


def passed(line: dict) -> bool:
    return (
        line.get("exit_status") == 0
        and line.get("timed_out") is False
        and line.get("interrupted") is None
        and isinstance(line.get("manifest_sha256"), str)
    )


def manifest_of(top: Path, line: dict, read: Callable[[Path], bytes] | None = None) -> dict:
    """The manifest of a step's run, once its SHA-256 is the one the log records, read once
    through `read` (a function of a path giving its bytes, raising `OSError`; by default
    `plain_read.read` from `top`, which opens no link and no "..", "." or absolute part of the
    run folder as the log writes it)."""
    run = line.get("run")
    if not isinstance(run, str):
        raise CannotCheck(f"{line.get('label')}: the log names no run folder")
    file = f"{top}/{run}/{MANIFEST_NAME}"  # as written: `Path` would drop a "."
    try:
        data = (_reader(top) if read is None else read)(file)
        digest, manifest = hashlib.sha256(data).hexdigest(), json.loads(data)
    except (OSError, ValueError) as problem:
        raise CannotCheck(
            f"{line.get('label')}: the manifest of {run} cannot be read: {problem}"
        ) from problem
    if digest != line.get("manifest_sha256") or not isinstance(manifest, dict):
        raise CannotCheck(
            f"{line.get('label')}: the manifest of {run} is not the one the log records"
        )
    return manifest


def _reader(top: Path) -> Callable[[Path], bytes]:
    """`plain_read.read` from `top`: no link, and no "..", "." or absolute part, is opened."""
    return lambda path: plain_read.read(top, path)


def failure(top: Path, line: dict) -> str:
    """The failure of a step that did not pass, once its manifest confirms it; raises
    `CannotCheck` for a step that never ran (reading R2)."""
    label = line.get("label")
    if line.get("run") is None or line.get("manifest_sha256") is None:
        raise CannotCheck(
            f"the recorder did not run {label} (status {line.get('recorder_status')}), so the "
            "rehearsal did not run it"
        )
    manifest = manifest_of(top, line)
    confirmed = (
        manifest.get("exit_code") == line.get("exit_status")
        and manifest.get("timed_out", False) == line.get("timed_out")
        and manifest.get("interrupted") == line.get("interrupted")
    )
    if not confirmed:
        raise CannotCheck(f"{label}: the manifest of {line.get('run')} does not confirm the log")
    if line.get("timed_out"):
        return f"{label} hit its time limit (exit status {line.get('exit_status')})"
    if line.get("interrupted") is not None:
        return f"{label} was interrupted by {line.get('interrupted')}"
    return f"{label} ended with exit status {line.get('exit_status')}"


def verify_run(
    top: Path,
    head: str,
    step: run_list.Step,
    command: list[str],
    line: dict,
    read: Callable[[Path], bytes] | None = None,
) -> None:
    """A step's recorded run verifies at `head` (reading R4); raises `CannotCheck`. The manifest
    and each output are read through `read` (by default `plain_read.read` from `top`;
    `manifest_of`, `require_verified`); an output name that is not plain is never read."""
    read = _reader(top) if read is None else read
    run = line.get("run")
    pattern = rf"{re.escape(RUNS_FOLDER)}/\d{{8}}T\d{{6}}Z-{re.escape(step.label)}"
    if not isinstance(run, str) or not re.fullmatch(pattern, run):
        raise CannotCheck(f"{step.label}: the run folder {run!r} is not named by its label")
    manifest = manifest_of(top, line, read)
    if manifest.get("commit") != head:
        raise CannotCheck(f"{step.label}: {run} was recorded at another commit")
    if manifest.get("command") != command:
        raise CannotCheck(f"{step.label}: {run} recorded another command than the step's")
    if manifest.get("long") is not True or manifest.get("max_seconds") != step.max_seconds:
        raise CannotCheck(f"{step.label}: {run} was not a long run with the step's time limit")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise CannotCheck(f"{step.label}: the manifest of {run} lists no output")
    for name in sorted(outputs):
        if not plain_name(name):
            raise CannotCheck(
                f"{step.label}: the manifest of {run} lists the output {name!r}, not a plain name "
                "in its folder (a '..', '.' or empty part, or an absolute start): not read"
            )
        try:
            require_verified(top / run / name, read)
        except NotVerified as problem:
            raise CannotCheck(f"{step.label}: {run}/{name} does not verify: {problem}") from problem


def is_o2(command) -> bool:
    """Whether a step's command as run is outcome O2's (reading R7)."""
    if not isinstance(command, list):
        return False
    for k in range(len(command) - 1):
        if command[k] == "-m" and command[k + 1] == O2_MODULE:
            rest = command[k + 2 :]
            rest = rest[: rest.index("--")] if "--" in rest else rest
            return any(
                isinstance(part, str) and len(part) > 2 and O2_OPTION.startswith(part)
                for part in rest
            )
    return False


def o2_verdict(data: bytes) -> tuple[str, list[str], list[str], list[str]]:
    """(the verdict over the units, the units whose pass failed, those it could not decide, those
    with no day to check) of the bytes of an O2 results file; raises `ValueError` saying what is
    missing (reading R7)."""
    try:
        document = json.loads(data)
    except (ValueError, RecursionError) as problem:
        raise ValueError(f"it is not JSON: {problem}") from problem
    every = document.get("every_day") if isinstance(document, dict) else None
    units = every.get("units") if isinstance(every, dict) else None
    if not isinstance(units, dict) or not units:
        raise ValueError("it holds no every_day with its units' O2 passes")
    verdicts = {
        name: found.get("o2_pass") if isinstance(found, dict) else None
        for name, found in units.items()
    }
    if not all(isinstance(v, str) and v in O2_WORDS for v in verdicts.values()):
        raise ValueError(f"a unit's O2 pass is not one of: {', '.join(O2_WORDS)}")
    over = every.get("pass_over_units")
    if over != pass_over(verdicts.values()):
        raise ValueError(f"its pass_over_units {over!r} is not the verdict over its units' passes")
    failed = [name for name, v in verdicts.items() if v == O2_FAILED]
    undecided = [name for name, v in verdicts.items() if v == O2_NOT_RUN]
    return over, failed, undecided, no_day_units(verdicts)


FAULT_COUNTS = ("order_breaks", "days_not_built", "block_order_breaks_on_m_days")


def _count(n) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n >= 0


def o2_faults(data: bytes) -> tuple[list[str], list[str]]:
    """(each unit's own faults, one text per unit, for the units with a non-zero count, the units
    whose counts are missing or not readable) of the bytes of an O2 results file whose verdict
    `o2_verdict` read (decision 88 (3d)). Each count is named by what it holds
    (`ceiling_summary.o2_with_faults`): `order_breaks`, the broken orders of both products on every
    day, of which `block_order_breaks_on_m_days` are in O2's pass; `days_not_built`, M days or not,
    of which the unit's `days_in_m_not_built` are in O2's pass. A unit is readable when its
    `faults` is a mapping with those three counts and its entry holds `days_in_m_not_built`, each
    a whole number, no less than 0."""
    units = json.loads(data)["every_day"]["units"]
    shown, unreadable = [], []
    for name, found in units.items():
        own = found.get("faults") if isinstance(found, dict) else None
        in_m = found.get("days_in_m_not_built") if isinstance(found, dict) else None
        readable = (
            isinstance(own, dict)
            and all(_count(own.get(key)) for key in FAULT_COUNTS)
            and _count(in_m)
        )
        if not readable:
            unreadable.append(name)
        elif any(own[key] for key in FAULT_COUNTS) or in_m:
            shown.append(
                f"{name}: {own['order_breaks']} orders broken (both products, every day; "
                f"{own['block_order_breaks_on_m_days']} of them block orders on M days, in O2's "
                f"pass), {own['days_not_built']} days not built ({in_m} of them M days, in O2's "
                "pass)"
            )
    return shown, unreadable


def o2_faults_lines(label, data: bytes) -> list[str]:
    """The lines giving the faults of an O2 results file beside its verdict line (decision 88
    (3d)): none when no unit has a fault and every record is readable; never silent on a record
    that is missing or not a mapping of counts."""
    shown, unreadable = o2_faults(data)
    head = f"O2 {label}: own faults (decision 85 (1)): "
    lines = [head + "; ".join(shown)] if shown else []
    if unreadable:
        lines.append(
            f"{head}the faults record is missing or not readable for {', '.join(unreadable)}"
        )
    return lines


def o2_read(top: Path, label, line: dict) -> tuple[str, str | None, list[str]]:
    """(the line of an O2 step with its `entry` line `line`, the fault that fails the check or
    None, the lines of its own faults beside the verdict), its verdict read through
    `run_verify.read_verified` (reading R7)."""
    if not passed(line):
        return (
            f"O2 {label}: no verdict read, the step did not exit 0 with no timeout (exit status "
            f"{line.get('exit_status')})",
            None,
            [],
        )
    run = line.get("run")
    pattern = rf"{re.escape(RUNS_FOLDER)}/\d{{8}}T\d{{6}}Z-{re.escape(str(label))}"
    if not isinstance(run, str) or not re.fullmatch(pattern, run):
        fault = f"O2 {label}: exited 0, but the log names no run folder of its label ({run!r})"
        return f"{fault}, so no verdict is read", fault, []
    where = f"{run}/{O2_RESULTS}"
    try:
        _, data = read_verified(top / run / O2_RESULTS, _reader(top))
        over, failed, undecided, no_day = o2_verdict(data)
    except (NotVerified, ValueError) as problem:
        fault = f"O2 {label}: exited 0, but {where} holds no readable verdict: {problem}"
        return fault, fault, []
    beside = f"; no day to check on {', '.join(no_day)}" if no_day else ""
    return (
        f"O2 {label}: over the units {over}; failed on {', '.join(failed) or 'none'}; could not "
        f"decide on {', '.join(undecided) or 'none'}{beside} (from {where}; not in this check's "
        "exit status, reading R7)",
        None,
        o2_faults_lines(label, data),
    )


def o2_line(top: Path, label, line: dict) -> tuple[str, str | None]:
    """(the line of an O2 step with its `entry` line `line`, the fault that fails the check or
    None), its verdict read through `run_verify.read_verified` (reading R7)."""
    text, fault, _ = o2_read(top, label, line)
    return text, fault


def o2_lines(top: Path, logged) -> list[tuple[str, str | None]]:
    """`o2_line` of each O2 step of `logged`, in order, each followed by the lines of its own
    faults (decision 88 (3d), none the fault that fails the check); `logged`: (label, command as
    run, `entry` line) of each step logged (reading R7)."""
    lines = []
    for label, command, line in logged:
        if is_o2(command):
            text, fault, beside = o2_read(top, label, line)
            lines += [(text, fault), *[(own, None) for own in beside]]
    return lines


def _duration(seconds: float) -> str:
    return f"{seconds:.0f} s ({seconds / 3600:.2f} h)"


def decide(top: Path, head: str) -> tuple[int, str]:
    """(exit status, last line) of the committed list and the newest log of its rehearsal; prints
    the lines before the last as it reads them; raises `NoList`, `NoLog` or `CannotCheck` (module
    text)."""
    data = committed_bytes(top, LIST)
    if data is None:
        raise NoList(f"no run list is committed at {LIST} at the checked-out commit")
    parsed = committed_list(top, data)
    steps = run_list.instantiate(parsed, PURPOSE)
    found_differences = run_list.differences(parsed)
    for line in found_differences:
        print(f"  difference from the run from the tag: {line}")
    if not found_differences:
        print("  no difference from the run from the tag")
    differ = (
        f"{run_list.differences_text(len(found_differences))} from the run from the tag, "
        f"{'listed above' if found_differences else 'none'} (reading R6)"
    )
    path = newest_log(top)
    records = read_log(path, top)
    start, end = records[0], records[-1]
    written = start.get("commit")
    print(f"log {path.name}, list {start.get('list')}, commit {str(written)[:12]}")
    named = path.name
    if written != head:
        words = [word for step in steps for word in step.command]
        reason = _not_earlier(top, written, words, head, ())
        if reason:
            raise CannotCheck(
                f"{path.name} is stale: written at commit {str(written)[:12]}, not at the "
                f"checked-out commit {head[:12]}, and {reason}"
            )
        named = f"{path.name}{earlier_note(written)}"  # reading R1
        print(f"  {named}")
    if start.get("list_sha256") != hashlib.sha256(data).hexdigest():
        raise CannotCheck(
            f"{named} drove other bytes than those of {LIST} committed at the checked-out commit"
        )
    if start.get("purpose") != PURPOSE:
        raise CannotCheck(
            f"the start of {named} names the purpose {start.get('purpose')!r}, not {PURPOSE!r}"
        )
    if start.get("steps") != logged_steps(steps):
        raise CannotCheck(
            f"the start of {named} does not list the rehearsal's steps of {LIST} as the list "
            "instantiates them"
        )
    found, stop = logged_entries(steps, records)
    for _, _, line in found:
        seconds = line.get("seconds")
        shown = _duration(seconds) if isinstance(seconds, int | float) else repr(seconds)
        print(
            f"  {line.get('label')}: run {line.get('run')}, exit status {line.get('exit_status')}"
            f", timed out {line.get('timed_out')}, {shown}"
        )
    o2 = o2_lines(top, [(step.label, command, line) for step, command, line in found])
    for text, _ in o2:
        print(f"  {text}")
    faults = [fault for _, fault in o2 if fault is not None]
    try:
        code, last = judge(top, written, named, end, steps, found, stop, differ)
    except CannotCheck as problem:
        if not faults:
            raise
        return FAILED, f"FAILED: {'; '.join(faults)}; and COULD NOT RUN: {problem}"
    if faults:
        rest = last.removeprefix("FAILED: ") if code == FAILED else named
        return FAILED, f"FAILED: {'; '.join(faults)}; {rest}"
    return code, last


def judge(
    top: Path, written: str, named: str, end: dict, steps, found, stop, differ: str
) -> tuple[int, str]:
    """(exit status, last line) of the steps `found` logged in the log written at the commit
    `written` and named `named` (its file name, with the note of reading R1 when it was written
    at an earlier commit), whose end line is `end`, O2's verdicts aside; raises `CannotCheck`
    (module text)."""
    if found and not passed(found[-1][2]):
        return FAILED, f"FAILED: {failure(top, found[-1][2])}; {named}"
    if len(found) < len(steps):
        why = stop.get("reason") if stop else "no reason logged"
        raise CannotCheck(f"the drive stopped before {steps[len(found)].label}: {why}")
    if end.get("unchanged_after") is not True:
        raise CannotCheck(
            f"the tree changed after the last step of {named}: {end.get('change_after')}"
        )
    if (end.get("completed"), end.get("exit_status"), end.get("entries_run")) != (
        True,
        0,
        len(steps),
    ):
        raise CannotCheck(f"the end of {named} does not say the drive completed")
    seconds = end.get("seconds")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, int | float)
        or not math.isfinite(seconds)
        or seconds < 0
    ):
        raise CannotCheck(f"{named} logs no total duration")
    for step, command, line in found:
        verify_run(top, written, step, command, line)
    return PASSED, (
        f"passed: the rehearsal of the run list {LIST} ran to its end at commit {written[:12]}: "
        f"{run_list.entries_text(len(found))}, {'which' if len(found) == 1 else 'each'} exited 0 "
        f"with no timeout and verifies; total duration {_duration(seconds)}; {named}; {differ}"
    )


def check(top: Path) -> int:
    print(HEADER)
    head = head_commit(top)
    try:
        code, last = decide(top, head)
    except NoList as problem:
        print(f"NO RUN LIST: {problem}")
        return COULD_NOT_RUN
    except NoLog as problem:
        print(f"NO REHEARSAL LOG: {problem}")
        return COULD_NOT_RUN
    print(last)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/rehearsal_check.py",
        description=f"Decide the WP12 pass-mark item from the committed run list {LIST} and the "
        "newest driver log of its rehearsal.",
    )
    parser.add_argument(
        "--top",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the repository top folder, where runs.noindex/ is (default: this repository)",
    )
    options = parser.parse_args(argv)
    try:
        return check(options.top)
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return COULD_NOT_RUN
    except Exception as problem:  # an unexpected error (module text)
        print(f"COULD NOT RUN: the check stopped on {type(problem).__name__}: {problem}")
        return COULD_NOT_RUN


if __name__ == "__main__":
    sys.exit(main())
