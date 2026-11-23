"""The check of work package 15's pass mark (plan section 13, WP15: "Sensitivities: ρ=0.1, 0.2 for
the main model only; chronological calibration (T10, at OSL_T); regimes (no split beyond the two
a.s.r. units). Pass: all reported."), one command, read-only: `python tools/wp15_check.py [--root
<folder>]` (`--root`: the project's top folder, by default the folder above `tools/`).
`passmarks/wp15.sh` runs it.

WHAT "REPORTED" IS READ AS. The plan's line is read through the readings W1 to W3 that the
committed run list states in its `about` (`run-lists/run-list.json`). They are not plan text, and
the first printed line says so:
  W1. ρ 0.1 and 0.2 for the main model only: the learn runs `learn-lgbm_tail-rho0_1` and
      `learn-lgbm_tail-rho0_2`, each on a matrix of that ρ, and the interval runs
      `intervals-lgbm_tail-rho0_1` and `intervals-lgbm_tail-rho0_2` over them.
  W2. The chronological calibration (T10, at OSL_T): the drift guard's run `drift-lgbm_tail` on
      OSL_T alone.
  W3. The regimes, no split beyond the two a.s.r. units: the two a.s.r. units
      (`load_asr.UNITS`) scored as units of their own by the main model's learn run
      `learn-lgbm_tail`.
The labels are fixed here from those readings (the main model `lgbm_tail`, the two values of ρ,
the unit OSL_T); everything else (each entry's command, its matrix, its units, the run it reads) is
read from the entries of the committed list that bear those labels. A label the list lacks, or an
entry not of the run from the tag, could not run, naming the label.

WHICH RUN. The run from the tag, read as the WP14 check reads it (`tools/sealed_run_check.py`:
its survey, the gate of the run from the tag, the drive logs `runs.noindex/drives/<UTC
start>-sealed-run.jsonl` and their matching rules, imported from it, not copied): the one complete
sealed-run drive of the committed list at the checked-out commit, which the gate accepts as the
tag's or a logged parsing fix of it; every step of it verifies there (`verify_run`: its run folder,
its manifest with the SHA-256 the log records, the checked-out commit, the step's command as run,
its time limit, and every output file as written by a run that ended with exit 0, neither stopped
nor timed out). A rehearsal drive (`-rehearsal.jsonl`) is never read: with no sealed-run drive log
the item could not run, and a line says when rehearsal logs are all there is. Each results file
read is read once, its bytes hashed against the manifest and then parsed.

AN INCOMPLETE DRIVE (decision 99 (5) of 22 November 2026; plan section 13, WP15). With no complete
drive, WP15's pass mark is not passed and the exit status is 1, never 0 (WP15 ran, its pass is not
met; 2 is could not run). The first printed line after the header says that the held-out run is
incomplete and names every failed and every not-run step of the newest drive. Then each part (a),
(b), (c) is printed as a line begun "information:", read only from the steps that passed, under
these conditions, each of which leaves the part unread when it does not hold: the drive is the
newest sealed-run drive that ran a step (never an earlier drive, never the rehearsal's); it ran to
its end (a drive with a `stop` line is never read); its tree is unchanged after the last step; it
is a drive of the checked-out commit, of the committed list's bytes and of the gate's tag and fix
commits; the step passed and verifies (`verify_run`); and every entry its command takes a
`{run:<label>}` from, and every entry named in its `after`, was itself read (a step whose chain
reaches a failed, not-run or unread step is not read). A part that needs a step not read is named
as lost, with the step and why. A part read shows its figures as information, never as "reported".

THE THREE PARTS, each decided:
  (a) ρ 0.1 and ρ 0.2. Per value r: the learn entry runs `learn_run` with `--model lgbm_tail`
      and a `--matrix` committed at the checked-out commit whose `rho` is r; its `learn.json`
      names that matrix's SHA-256; and for every gate unit of the matrix and every configuration
      the run scores, under tier A, the fixed and one-way rungs (`learn_eval.OFFER_RUNGS`) hold the
      offer per counting test day and the offer's share of each of the two hourly ceilings
      (`gate_config.CEILINGS`, ratio of sums), each a finite number (reading V2), or the
      configuration holds only the reason that its unit cannot be split (reading V6). The interval
      entry runs `day_intervals` on that learn entry's run of the same drive (`--run
      {run:<learn label>}`), and its `day-intervals.json` holds, for the same cells (those of a
      unit that cannot be split have none), the interval of the offer and of each share, two
      finite ends each, with no problem listed (reading V3).
  (b) The chronological calibration at OSL_T. The drift entry runs `drift_run` with `--model
      lgbm_tail` and `--units OSL_T` alone; its `drift-run.json` holds OSL_T as run, not failed,
      with guarded days, and names its day table `drift-run-days.csv` by the SHA-256 and row count
      the table has; the table holds OSL_T's guarded days for each tier the run holds a path of,
      each day's offer a number and each offering step's slot and driver losses numbers (reading
      V4). A tier with no path offers nothing (the drift guard's reading E29) and is named so.
      When no tier holds a path, the results' own record must give every tier "offers nothing"
      for OSL_T (reading V6).
  (c) The regimes. The main model's learn entry runs `learn_run` with `--model lgbm_tail`; its
      `learn.json` holds each a.s.r. unit as a unit of its own, with the figures of (a) (or the
      reason of reading V6), and no other unit whose name begins `asr` (a pooled a.s.r. unit)
      (reading V5).
A part with a reason of reading V6 is "reported, with the reason", printed with that reason.

EXIT STATUS. 0 only when all three parts are reported (with the reason, reading V6, or not), from a
complete drive, and WP14's pass mark passes on that drive: WP15 runs WP14's own check
(`sealed_run_check.check`, the marks run, fixes and hashes) on the same drive, its output
collected and not printed, into a throwaway run folder of its own in the runs folder (removed
after), and when that does not exit 0 it prints each line of it that failed or could not run begun
"WP14:" and exits 1 (decision 99 (5): never passed while WP14 fails). 1 for an incomplete drive
that ran to its end under the conditions above, however its parts read. 1 also when a part's run
ended and failed (a step of the newest drive, at the checked-out commit, that ended with an exit
status other than 0, not timed out and not interrupted, as its manifest confirms) in a drive that
is not read as information (one that stopped or whose tree changed), or its results are there and
lack a needed figure (no reason of reading V6 given), or a configuration is recorded as failed
(reading V6). 2 otherwise: no sealed-run drive log, only rehearsal logs, a log that cannot be read,
a stale log, a drive that broke off or stopped, two complete drives, a later drive that ran a step,
a gate that refuses the checked-out commit, a missing or misnamed entry, a run that timed out or
was stopped, a run that does not verify, a results file missing or changed, an unexpected error.
The worst part decides, and a failure outranks could not run (as in the WP14 check and
`passmark.sh`). Any other error, a failed
import among them, is could not run, never Python's 1.

It opens, lists or reads no file of the sealed folder: it reads git, the gate's checks, the
committed list and matrices, the drive logs, and the manifests and results files of the runs a
drive log names (`tests/wp15/test_wp15_check.py` pins this).

Readings (alternatives in brackets):
  V1. The run from the tag is the one complete drive (WP14's mark 1). With no complete drive the
      parts are read only as information from the steps that passed, never as a pass (the
      owner's decision 99 (5), 22 November 2026; module text, AN INCOMPLETE DRIVE). [Alternative:
      read the parts from any drive whose steps passed, and pass.]
  V2. "The offer and both hourly ceilings at that ρ": tier A's offer per counting test day and
      its share of each ceiling (ratio of sums) for every configuration and gate unit of the
      learn run. [Alternatives: the ceiling sums of `learn-gate.json`; tier B as well; the
      headline configuration of each site type alone.]
  V3. "Its intervals over days": for the same cells, the interval of the offer and of each share
      in `day-intervals.json`. [Alternative: the pooled block over units.]
  V4. "The drift guard's per-day losses and offer on OSL_T": the rows of OSL_T in the day table,
      checked against the results' own record of it. [Alternative: the per-path summaries of
      `drift-run.json` alone.]
  V5. "No split beyond the two a.s.r. units": the main model's learn run scores each a.s.r. unit
      on its own and no pooled a.s.r. unit. [Alternative: a run of its own pooling the two units
      (W3's own alternative), which this check would not find.]
  V6. Reported, with the reason (the owner's decisions of 19 and 22 November 2026), in three cases
      and no other: (1) a unit that cannot be split, read as a unit of `learn.json` whose status
      starts "not calibratable", with 0 test days, whose configuration's rungs hold only the reason
      `{"reason": <that status>}` (as `gate_run.plan_unit` and `learn_eval.offers` write it); its
      cells need no figure and no interval; (2) a drift run in which no tier offers anything
      (reading E29): no tier of OSL_T holds a path and the record `sealed_route.step_sizes` gives
      OSL_T "offers nothing" for every tier of the run; (3) a held-out unit the held-out truth
      build abstained (its `truth-summary.json`, the run the entry's `--held-out-truth-run`
      names), absent from the results: "<unit>: no figure: the held-out truth build gave it no
      window (<reason>)". A configuration that holds only another reason, or no figure and no
      reason, still lacks its figures. A configuration recorded as failed (`failed: <error>` in
      its rungs, or listed in `failed_configurations` as `<configuration> | <unit>: <error>`) is
      never reported: the part fails, naming the unit, the configuration and the error.
      [Alternative: the cases (2) and (3) fail the part.]
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:
    import rehearsal_check as wp12  # the WP12 check's readers of a drive log, unchanged
    import sealed_run_check as wp14  # the WP14 check's survey of the run from the tag, unchanged
    from freeze_check import Findings

    from fleetreserve import (
        day_intervals,
        drift_run,
        drift_settings,
        learn_eval,
        learn_run,
        load_asr,
        run_list,
    )
    from fleetreserve.gate_config import CEILINGS
    from fleetreserve.run_driver import DRIVES_FOLDER, LOG_SUFFIX
    from fleetreserve.run_evidence import CannotCheck
    from fleetreserve.run_record import RUN_DIR_VARIABLE, RUNS_FOLDER
    from fleetreserve.truth_build import SUMMARY_NAME as TRUTH_SUMMARY_NAME
except Exception as problem:  # the check cannot start: could not run, never Python's exit 1
    print(f"COULD NOT RUN: the check cannot start: {type(problem).__name__}: {problem}")
    sys.exit(2)

TOP = Path(__file__).resolve().parents[1]
LIST = wp12.LIST
PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2
WORDS = {PASSED: "passed", FAILED: "FAILED", COULD_NOT_RUN: "COULD NOT RUN"}
MAIN_MODEL = "lgbm_tail"  # the main model (reading W1)
RHOS = (0.1, 0.2)  # the values of ρ of plan WP15 (reading W1)
DRIFT_UNIT = "OSL_T"  # the unit of plan T10's chronological run (reading W2)
ASR_UNITS = tuple(load_asr.UNITS)  # the two a.s.r. units (reading W3)
TIER = "A"
PARTS = ("(a) rho 0.1 and 0.2", "(b) chronological calibration at OSL_T", "(c) regimes")
HEADER = (
    "WP15 (plan section 13: sensitivities; pass: all reported), read by the readings W1 to W3 "
    f"that the run list {LIST} states in its about, not plan text: W1 rho 0.1 and 0.2 for the "
    "main model are the learn and interval entries at each rho; W2 the chronological "
    "calibration is the drift guard's entry on OSL_T alone; W3 the "
    "regimes are the two a.s.r. units scored as units of their own by the main model's learn "
    "entry; read from the run from the tag (the one complete sealed-run drive), never from the "
    "rehearsal"
)


def _rho_text(rho: float) -> str:
    return f"{rho:g}".replace(".", "_")


def learn_label(rho: float | None = None) -> str:
    """The learn entry of the main model, at the matrix's own ρ (None) or at `rho` (W1, W3)."""
    return f"learn-{MAIN_MODEL}" if rho is None else f"learn-{MAIN_MODEL}-rho{_rho_text(rho)}"


def intervals_label(rho: float) -> str:
    return f"intervals-{MAIN_MODEL}-rho{_rho_text(rho)}"


DRIFT_LABEL = f"drift-{MAIN_MODEL}"


def needed_labels() -> tuple[str, ...]:
    """Every entry label WP15 reads, in the order of the parts."""
    rho = tuple(x for r in RHOS for x in (learn_label(r), intervals_label(r)))
    return (*rho, DRIFT_LABEL, learn_label())


class Lacks(Exception):
    """A part's results are there and lack a needed figure (exit status 1)."""


class Failed(Exception):
    """A part's run ended and failed, or a configuration is recorded as failed (exit status 1)."""


class Lost(Exception):
    """A part needs a step of an incomplete drive that was not read (information only)."""


@dataclass
class Run:
    """The run from the tag: each step's (step, command as run, entry line) by label; for an
    incomplete drive that is read as information (`incomplete`), those of the steps that passed
    and verify and whose chain does too, with `lost` giving each other step's label and why it is
    not read; for any other incomplete newest drive, the failure of each step WP15 needs that
    ended and did not pass (`failed`)."""

    name: str
    steps: dict  # label -> (step, command, line)
    failed: dict  # label -> (what happened, plain failure) of a needed step that did not pass
    incomplete: bool = False  # read as information only (module text, AN INCOMPLETE DRIVE)
    lost: dict = field(default_factory=dict)  # label -> why the step is not read


def drive_entries(steps, drive) -> list:
    """[(step, its command as run, its `entry` line)] of every step the drive ran, in the list's
    order, as `rehearsal_check.logged_entries` reads a rehearsal's, but through a sealed-run
    drive's lines: a step the driver did not start has a `not_run` line of its index and label,
    and a step that failed is followed by its `failed` line (`sealed_run_check.read_drive`
    checks their shapes); raises `CannotCheck` when the lines do not follow the list."""
    begun = {begin.get("index"): (begin, line) for begin, line in drive.steps}
    skipped = {line.get("index"): line for line in drive.not_run}
    found, runs = [], {}
    for k, step in enumerate(steps):
        if k in skipped:
            if skipped[k].get("label") != step.label:
                raise CannotCheck(f"the log does not follow the run list at step {k + 1}")
            continue
        if k not in begun:
            break  # the drive stopped before this step
        begin, line = begun[k]
        try:
            command = run_list.fill_runs(step, runs)
        except KeyError as problem:
            raise CannotCheck(
                f"the log records no run folder of {problem.args[0]}, whose run step {k + 1} "
                f"({step.label}) names"
            ) from problem
        if not (
            begin.get("label") == step.label
            and line.get("label") == step.label
            and begin.get("command") == command
            and begin.get("max_seconds") == step.max_seconds
        ):
            raise CannotCheck(
                f"the log does not follow the run list at step {k + 1} ({step.label})"
            )
        found.append((step, command, line))
        if wp12.passed(line) and isinstance(line.get("run"), str):
            runs[step.label] = line["run"]
    if len(found) != len(drive.steps):
        raise CannotCheck("the log holds lines the run list does not account for")
    return found


def the_run(root: Path, s, found: Findings) -> Run | None:
    """The run from the tag (module text, WHICH RUN; reading V1), or None when it could not run
    (the reason in `found`)."""
    complete = [d for d in s.drives if d.complete()]
    if len(complete) > 1:
        found.cannot_run(
            f"{len(complete)} complete sealed-run drives, {', '.join(d.name for d in complete)}: "
            "which is the run from the tag cannot be told (the WP14 check fails it)"
        )
        return None
    if complete:
        drive = complete[0]
        later = [d for d in s.drives[s.drives.index(drive) + 1 :] if not d.ran_no_step()]
        if later:
            found.cannot_run(
                f"{later[0].name}, a sealed-run drive after the complete drive {drive.name}, ran "
                "a step: which is the run from the tag cannot be told (the WP14 check fails it)"
            )
            return None
        logged = wp14._verify_complete(root, s, drive, found)  # raises CannotCheck
        if logged is None:
            return None
        return Run(drive.name, {step.label: (step, c, line) for step, c, line in logged}, {})
    if not s.drives:
        found.cannot_run("no sealed-run drive log can be read")
        return None
    ran = [d for d in s.drives if not d.ran_no_step()]
    if not ran:
        found.cannot_run("no sealed-run drive ran a step (no entry names a run folder)")
        return None
    drive = ran[-1]  # the newest drive that ran a step: never an earlier one
    found.say(f"the held-out run is incomplete: {drive.name} {wp14._stopped(drive)}")
    if (
        drive.stop is None
        and wp14._unfit(s, drive) is None
        and drive.end.get("unchanged_after") is True
    ):
        run = incomplete_run(root, s, drive)
        found.say(
            f"information: WP15's pass mark is not passed; each part below is read only from the "
            f"steps of {drive.name} that passed and verify at the checked-out commit, whose chain "
            "of runs and `after` entries was read as well (a drive that ran to its end, its tree "
            "unchanged)"
        )
        return run
    if drive.stop is not None:
        found.say(f"information: no part is read from {drive.name}: a stopped drive is never read")
    elif drive.end.get("unchanged_after") is not True and wp14._unfit(s, drive) is None:
        found.say(
            f"information: no part is read from {drive.name}: its tree changed after the last step"
        )
    said: list[str] = []
    probe = Findings(said.append)
    try:
        logged = wp14._verify_complete(root, s, drive, probe)
    except CannotCheck:
        # Raised only after the drive was found to be of the checked-out commit, the gate's
        # commits and the committed list's sealed-run steps (a stale drive or another list's
        # returns None first): a step that did not pass is then read below.
        logged = None
    if probe.cannot:
        for line in said:
            found.cannot_run(line.removeprefix("COULD NOT RUN: "))
        return None
    steps = run_list.instantiate(s.parsed, wp14.PURPOSE)
    failed = {}
    # The sealed-run drive goes on after a failed entry, so any step of the newest drive may have
    # failed: each step WP15 needs that ended and did not pass is read from its own entry.
    for step, _, line in drive_entries(steps, drive):
        if logged is None and step.label in needed_labels() and not wp12.passed(line):
            text = wp12.failure(root, line)  # raises CannotCheck for a step that never ran
            plain = line.get("timed_out") is False and line.get("interrupted") is None
            failed[step.label] = (text, plain)
    return Run(drive.name, {}, failed)


def _why(root: Path, line: dict) -> str:
    """What happened to a step that did not pass, as its manifest confirms it, else the log's own
    words."""
    try:
        return wp12.failure(root, line)
    except CannotCheck as problem:
        return (
            f"{line.get('label')} did not pass (exit status {line.get('exit_status')}): {problem}"
        )


def incomplete_run(root: Path, s, drive) -> Run:
    """The steps of the incomplete `drive` that are read as information (module text, AN INCOMPLETE
    DRIVE): the drive is one that ran to its end, of the checked-out commit, with its tree
    unchanged. A step is read when it passed, verifies (`verify_run`) and every entry it takes a
    `{run:<label>}` from or names in its `after` was read; each other step is lost, with the
    reason. Raises `CannotCheck` when the log does not follow the run list."""
    steps = run_list.instantiate(s.parsed, wp14.PURPOSE)
    begun = {begin.get("index"): (begin, line) for begin, line in drive.steps}
    skipped = {line.get("index"): line for line in drive.not_run}
    read: dict = {}
    lost: dict = {}
    folders: dict[str, str] = {}  # label -> the run folder of every step that ran, for `{run:}`
    for k, step in enumerate(steps):
        if k in skipped:
            if skipped[k].get("label") != step.label:
                raise CannotCheck(f"the log does not follow the run list at step {k + 1}")
            reason = str(skipped[k].get("reason") or wp14.NO_REASON)
            lost[step.label] = reason if reason.startswith("not run") else f"not run: {reason}"
            continue
        if k not in begun:
            raise CannotCheck(f"the log holds no line of step {k + 1} ({step.label})")
        begin, line = begun[k]
        try:
            command = run_list.fill_runs(step, folders)
        except KeyError as problem:
            lost[step.label] = f"the run of {problem.args[0]}, which it reads, was not recorded"
            continue
        if not (
            begin.get("label") == step.label
            and line.get("label") == step.label
            and begin.get("command") == command
            and begin.get("max_seconds") == step.max_seconds
        ):
            raise CannotCheck(
                f"the log does not follow the run list at step {k + 1} ({step.label})"
            )
        if isinstance(line.get("run"), str):
            folders[step.label] = line["run"]
        if not wp12.passed(line):
            lost[step.label] = f"failed: {_why(root, line)}"
            continue
        needs = dict.fromkeys([*(label for _, label in step.runs), *step.after])
        unread = [label for label in needs if label not in read]
        if unread:
            lost[step.label] = "; ".join(
                f"needs {label}, which is not read ({lost.get(label, 'it is not an entry run')})"
                for label in unread
            )
            continue
        try:
            wp12.verify_run(root, s.head, step, command, line, lambda path: wp14._read(root, path))
        except CannotCheck as problem:
            lost[step.label] = f"passed but does not verify: {problem}"
            continue
        read[step.label] = (step, command, line)
    if len(begun) + len(skipped) != len(drive.steps) + len(drive.not_run):
        raise CannotCheck("the log holds lines the run list does not account for")
    return Run(drive.name, read, {}, incomplete=True, lost=lost)


def entry_step(s, run: Run, label: str, reading: str):
    """(step, command as run, entry line) of `label` in the run from the tag; raises
    `CannotCheck` (missing entry, not run) or `Failed` (its run ended and failed)."""
    entries = {e.label: e for e in s.parsed.entries}
    if label not in entries:
        raise CannotCheck(
            f"the committed run list {LIST} has no entry {label}, which {reading} names"
        )
    if not entries[label].runs_in(wp14.PURPOSE):
        raise CannotCheck(f"the entry {label} of {LIST} does not run in the run from the tag")
    if label in run.lost:
        raise Lost(f"{label} is lost: {run.lost[label]}")
    if label in run.failed:
        text, plain = run.failed[label]
        if plain:
            raise Failed(f"{label}'s run in {run.name} ended and failed: {text}")
        raise CannotCheck(f"{label}'s run in {run.name} did not end by itself: {text}")
    if label not in run.steps:
        raise CannotCheck(f"{label} has no verified run in a complete drive ({run.name})")
    return run.steps[label]


def _one_value(step, option: str) -> str | None:
    given = run_list.step_options(step, (option,))[option]
    return given[0].value if len(given) == 1 else None


def _command_is(step, module: str, label: str) -> None:
    """The step runs `fleetreserve.<module>` with `--model` the main model; raises
    `CannotCheck`."""
    if run_list.step_module(step) != module:
        raise CannotCheck(f"the entry {label} runs {run_list.step_module(step)}, not {module}")
    model = _one_value(step, "--model")
    if model != MAIN_MODEL:
        raise CannotCheck(
            f"the entry {label} runs the model {model}, not the main model {MAIN_MODEL}"
        )


def results(root: Path, label: str, line: dict, name: str) -> bytes:
    """The bytes of the results file `name` of a step's run, once its SHA-256 is the one the
    run's manifest lists (the manifest as the drive log records it); raises `CannotCheck`."""
    manifest = wp12.manifest_of(root, line)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or name not in outputs:
        raise CannotCheck(f"{label}: the run {line.get('run')} lists no {name} in its manifest")
    try:
        data = (root / line["run"] / name).read_bytes()
    except OSError as problem:
        raise CannotCheck(f"{label}: {line['run']}/{name} cannot be read: {problem}") from problem
    if hashlib.sha256(data).hexdigest() != outputs[name]:
        raise CannotCheck(f"{label}: {line['run']}/{name} is not the file its run wrote")
    return data


def _json(label: str, data: bytes, name: str) -> dict:
    try:
        document = json.loads(data)
    except (ValueError, RecursionError) as problem:
        raise CannotCheck(f"{label}: {name} is not JSON: {problem}") from problem
    if not isinstance(document, dict):
        raise CannotCheck(f"{label}: {name} is not a JSON object")
    return document


def _finite(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _get(document, *keys):
    for key in keys:
        if not isinstance(document, dict) or key not in document:
            return None
        document = document[key]
    return document


NOT_SPLIT = "not calibratable"  # the start of a unit's status when it cannot be split (gate_run)


def unsplit_reason(document: dict, unit: str, offer) -> str | None:
    """The reason of a configuration that holds only the reason that its unit cannot be split
    (reading V6 (1)), else None."""
    entry = _get(document, "units", unit)
    status, test_days = _get(entry, "status"), _get(entry, "test_days")
    if not isinstance(status, str) or not status.startswith(NOT_SPLIT):
        return None
    if type(test_days) is not int or test_days != 0:
        return None
    return status if _get(offer, "rungs") == {"reason": status} else None


def _reasons_text(reasons: dict) -> str:
    return "; ".join(
        f"{unit}: {reason} ({wp14._plural(count, 'configuration')})"
        for (unit, reason), count in reasons.items()
    )


def _gone_text(unit: str, reason: str) -> str:
    """How a unit the held-out truth build abstained is reported (reading V6 (3))."""
    return f"{unit}: no figure: the held-out truth build gave it no window ({reason})"


def _pieces(reasons: dict, gone: dict) -> list[str]:
    """The reasons of reading V6 (1) and (3) of one learn run, as the part prints them."""
    return [
        *([_reasons_text(reasons)] if reasons else []),
        *(_gone_text(unit, reason) for unit, reason in gone.items()),
    ]


def abstained_units(root: Path, run: Run, step) -> dict[str, str]:
    """The units the held-out truth build that `step` reads (the `{run:<label>}` its
    `--held-out-truth-run` names) recorded as abstained, with the reason of each (reading V6 (3)):
    a unit its `truth-summary.json` lists under `abstained` and writes no tables for. None when
    the step names no build run or the build was not read; raises `CannotCheck` when its summary
    does not verify."""
    option = run_list._one(run_list.step_options(step, (run_list.TRUTH_RUN,)), run_list.TRUTH_RUN)
    label = run_list._run_of(step, option)
    if label is None or label not in run.steps:
        return {}
    line = run.steps[label][2]
    summary = _json(label, results(root, label, line, TRUTH_SUMMARY_NAME), TRUTH_SUMMARY_NAME)
    written = summary.get("units")
    recorded = summary.get("abstained")
    written = written if isinstance(written, dict) else {}
    recorded = recorded if isinstance(recorded, dict) else {}
    found = {}
    for unit, entry in recorded.items():
        reason = entry.get("reason") if isinstance(entry, dict) else None
        if unit not in written and isinstance(reason, str) and reason:
            found[unit] = reason
    return found


FAILED_CONFIGURATION = re.compile(r"(.+?) \| (.+?): (.*)", re.DOTALL)  # `gate_run`'s list entry
FAILED_START = "failed:"  # the start of the reason a failed configuration holds in its rungs


def failed_cells(document: dict, units, cells: dict) -> dict[tuple[str, str], str]:
    """(configuration, unit) -> error of every configuration of `units` that `learn.json` records
    as failed (reading V6): its rungs hold the reason `failed: <error>`, or the document's
    `failed_configurations` lists it as `<configuration> | <unit>: <error>`. `cells` is the
    document's (configuration, unit) -> offer of the units read."""
    found = {}
    for (name, unit), offer in cells.items():
        reason = _get(offer, "rungs", "reason")
        if isinstance(reason, str) and reason.startswith(FAILED_START):
            found[name, unit] = reason.removeprefix(FAILED_START).strip()
    listed = _get(document, "failed_configurations")
    for text in listed if isinstance(listed, list) else []:
        named = FAILED_CONFIGURATION.fullmatch(text) if isinstance(text, str) else None
        if named and named[2] in units and (named[1], named[2]) not in found:
            found[named[1], named[2]] = named[3].removeprefix(FAILED_START).strip()
    return found


def offer_cells(
    document: dict, units, label: str, known=lambda: {}
) -> tuple[list[tuple[str, str]], dict, dict]:
    """The (configuration, unit) cells of a learn run's `learn.json` for `units`, after checking
    each holds tier A's offer and share of each ceiling for both offer rungs (reading V2), the
    cells reported with the reason, counted by (unit, reason) (reading V6 (1)), and the units
    reported as abstained by the held-out truth build, with its reason (reading V6 (3); `known()`
    gives them, read only when a unit has no configuration); raises `Failed` naming each
    configuration recorded as failed (reading V6) and `Lacks` naming the first cells that lack a
    figure."""
    lacking, cells, reasons, gone, offers = [], [], {}, {}, {}
    for unit in units:
        configurations = _get(document, "units", unit, "configurations")
        if not isinstance(configurations, dict) or not configurations:
            if unit in known():
                gone[unit] = known()[unit]
                continue
            cause = (
                "the learn run has no entry for it"
                if _get(document, "units", unit) is None
                else "its entry in the learn run holds no configuration"
            )
            lacking.append(
                f"{unit}: no figure: {cause}, and the held-out truth build recorded no "
                "abstention for it"
            )
            continue
        for name, offer in configurations.items():
            offers[name, unit] = offer
    failed = failed_cells(document, units, offers)
    if failed:
        items = [f"configuration {c} on unit {u}: {why}" for (c, u), why in failed.items()]
        raise Failed(
            f"{label} records {wp14._plural(len(failed), 'configuration')} as failed, so no unit "
            f"of it is reported: {'; '.join(items[:3])}"
        )
    for (name, unit), offer in offers.items():
        reason = unsplit_reason(document, unit, offer)
        if reason is not None:
            reasons[unit, reason] = reasons.get((unit, reason), 0) + 1
            continue
        cells.append((name, unit))
        for rung in learn_eval.OFFER_RUNGS:
            test = _get(offer, "rungs", rung, "test")
            where = f"{name} on {unit}, {rung}"
            if not _finite(_get(test, "mean_offer_kwh_per_day")):
                lacking.append(f"{where}: no offer per counting test day")
            for ceiling in CEILINGS:
                if not _finite(_get(test, "share_of_ceiling", ceiling, "ratio_of_sums")):
                    lacking.append(f"{where}: no share of the {ceiling} ceiling")
    if lacking:
        raise Lacks(
            f"{label} lacks {wp14._plural(len(lacking), 'figure')}: {'; '.join(lacking[:3])}"
        )
    return cells, reasons, gone


def _interval(found) -> bool:
    ends = _get(found, "interval")
    return isinstance(ends, list) and len(ends) == 2 and all(_finite(e) for e in ends)


def check_intervals(document: dict, cells, label: str) -> None:
    """The intervals of every cell (reading V3); raises `Lacks`."""
    if document.get("problems"):
        raise Lacks(f"{label}: day-intervals.json lists problems: {document['problems'][0]}")
    lacking = []
    for name, unit in cells:
        for rung in learn_eval.OFFER_RUNGS:
            found = _get(document, "configurations", name, unit, rung, TIER)
            where = f"{name} on {unit}, {rung}"
            if not _interval(_get(found, "offer_kwh_per_counting_test_day")):
                lacking.append(f"{where}: no interval of the offer")
            for ceiling in CEILINGS:
                if not _interval(_get(found, "share_of_ceiling", ceiling)):
                    lacking.append(f"{where}: no interval of the share of the {ceiling} ceiling")
    if lacking:
        raise Lacks(
            f"{label} lacks {wp14._plural(len(lacking), 'interval')}: {'; '.join(lacking[:3])}"
        )


def matrix_of(root: Path, step, label: str) -> tuple[dict, str]:
    """The committed matrix a learn entry names and the SHA-256 of its bytes; raises
    `CannotCheck`."""
    path = _one_value(step, "--matrix")
    if path is None:
        raise CannotCheck(f"the entry {label} names no one --matrix")
    data = wp12.committed_bytes(root, path)
    if data is None:
        raise CannotCheck(
            f"the matrix {path} of {label} is not committed at the checked-out commit"
        )
    return _json(label, data, path), hashlib.sha256(data).hexdigest()


def learn_document(root: Path, label: str, line: dict, digest: str) -> dict:
    """The verified `learn.json` of a learn run, once it names the matrix of SHA-256 `digest`."""
    document = _json(label, results(root, label, line, learn_run.RESULTS_NAME), "learn.json")
    if _get(document, "matrix", "sha256") != digest:
        raise CannotCheck(f"{label}: its learn.json names another matrix than its entry's")
    return document


def _known(root: Path, run: Run, step):
    """`abstained_units` of `step`, read once and only when asked for."""
    cache: list[dict] = []

    def known() -> dict[str, str]:
        if not cache:
            cache.append(abstained_units(root, run, step))
        return cache[0]

    return known


def part_rho(root: Path, s, run: Run, say) -> list[str]:
    """Part (a) (module text): the reasons of reading V6; raises `CannotCheck`, `Failed` or
    `Lacks`."""
    said = []
    for rho in RHOS:
        learn, intervals = learn_label(rho), intervals_label(rho)
        step, _, line = entry_step(s, run, learn, "reading W1")
        _command_is(step, run_list.LEARN_MODULE, learn)
        matrix, digest = matrix_of(root, step, learn)
        if matrix.get("rho") != rho:
            raise CannotCheck(f"the matrix of {learn} is at rho {matrix.get('rho')}, not {rho:g}")
        document = learn_document(root, learn, line, digest)
        units = matrix.get("gate_units")
        if not isinstance(units, list) or not units:
            raise CannotCheck(f"the matrix of {learn} names no gate unit")
        cells, reasons, gone = offer_cells(document, units, learn, _known(root, run, step))
        step_i, _, line_i = entry_step(s, run, intervals, "reading W1")
        if run_list.step_module(step_i) != run_list.INTERVALS_MODULE:
            raise CannotCheck(f"the entry {intervals} does not run day_intervals")
        option = run_list._one(run_list.step_options(step_i, ("--run",)), "--run")
        if run_list._run_of(step_i, option) != learn:
            raise CannotCheck(f"the entry {intervals} does not read the run of {learn}")
        data = results(root, intervals, line_i, day_intervals.RESULTS_NAME)
        check_intervals(_json(intervals, data, day_intervals.RESULTS_NAME), cells, intervals)
        say(
            f"  rho {rho:g}: {learn} ({line['run']}) holds the offer and its share of both "
            f"ceilings for {len(cells)} configuration-unit cells of {len(units)} units, "
            f"{intervals} ({line_i['run']}) their intervals"
            + (f"; with the reason, no figure: {_reasons_text(reasons)}" if reasons else "")
            + "".join(f"; {_gone_text(unit, reason)}" for unit, reason in gone.items())
        )
        if reasons or gone:
            said.append(f"rho {rho:g}: {'; '.join(_pieces(reasons, gone))}")
    return said


def offers_nothing(document: dict, tiers) -> bool:
    """Whether the results' record gives OSL_T "offers nothing" for every tier in `tiers` (reading
    V6 (2))."""
    eta = _get(document, "sealed_route", "step_sizes", DRIFT_UNIT, "eta")
    return (
        bool(tiers)
        and isinstance(eta, dict)
        and all(eta.get(t) == drift_settings.OFFERS_NOTHING for t in tiers)
    )


def part_drift(root: Path, s, run: Run, say) -> list[str]:
    """Part (b) (module text): the reason of reading V6, if any; raises `CannotCheck`, `Failed`
    or `Lacks`."""
    step, _, line = entry_step(s, run, DRIFT_LABEL, "reading W2")
    _command_is(step, "drift_run", DRIFT_LABEL)
    units = run_list._values(step, "--units")
    if units != [DRIFT_UNIT]:
        raise CannotCheck(f"the entry {DRIFT_LABEL} runs on {units}, not on {DRIFT_UNIT} alone")
    name, days_name = drift_run.RESULTS_NAME, drift_run.DAYS_NAME
    document = _json(DRIFT_LABEL, results(root, DRIFT_LABEL, line, name), name)
    table = results(root, DRIFT_LABEL, line, days_name)
    entry = _get(document, "units", DRIFT_UNIT)
    failed_units = document.get("failed_units")
    if failed_units or _get(entry, "status") != "run":
        if entry is None and not failed_units:  # not in the results: abstained, or another cause
            abstained = abstained_units(root, run, step)
            if DRIFT_UNIT in abstained:
                reason = (
                    f"{_gone_text(DRIFT_UNIT, abstained[DRIFT_UNIT])}; the only chronological "
                    "check is lost"
                )
                say(f"  {DRIFT_LABEL} ({line['run']}): {reason}")
                return [reason]
            cause = (
                f"{DRIFT_UNIT} is not in the drift run's results, and the held-out truth build "
                "recorded no abstention for it"
            )
        elif failed_units:
            cause = f"{DRIFT_UNIT} did not run, the drift run lists it as failed: {failed_units}"
        else:
            cause = f"{DRIFT_UNIT} has the status {_get(entry, 'status')!r}, not 'run'"
        raise Lacks(f"{DRIFT_LABEL}: {cause}")
    if not isinstance(entry.get("guarded_days"), int) or entry["guarded_days"] < 1:
        raise Lacks(f"{DRIFT_LABEL}: {DRIFT_UNIT} has no guarded day")
    rows = list(csv.DictReader(io.StringIO(table.decode("utf-8"))))
    record = document.get("days_file")
    if _get(record, "sha256") != hashlib.sha256(table).hexdigest() or _get(record, "rows") != len(
        rows
    ):
        raise CannotCheck(f"{DRIFT_LABEL}: {days_name} is not the table its results record")
    tiers, said = _get(document, "run", "tiers") or [], []
    paths = entry.get("paths") if isinstance(entry.get("paths"), dict) else {}
    for tier in tiers:
        if not paths.get(tier):
            said.append(f"tier {tier} runs no path (offers nothing)")
            continue
        mine = [r for r in rows if r.get("unit") == DRIFT_UNIT and r.get("tier") == tier]
        stepped = [r for r in mine if r.get("offering") == "true" and r.get("step")]
        if not mine or not all(_number(r.get("offer_kwh")) for r in mine):
            raise Lacks(f"{DRIFT_LABEL}: tier {tier} of {DRIFT_UNIT} lacks its days' offers")
        if not all(_number(r.get("slot_loss")) and _number(r.get("driver_loss")) for r in stepped):
            raise Lacks(f"{DRIFT_LABEL}: tier {tier} of {DRIFT_UNIT} lacks a day's losses")
        said.append(f"tier {tier}: {len(mine)} guarded days, {len(stepped)} offering steps")
    if len(said) == sum("no path" in x for x in said):
        if not offers_nothing(document, tiers):
            raise Lacks(f"{DRIFT_LABEL}: {DRIFT_UNIT} has no per-day loss in any tier")
        reason = (
            f"no tier offers anything on {DRIFT_UNIT}: its run's record gives tiers "
            f'{", ".join(tiers)} "{drift_settings.OFFERS_NOTHING}" (reading E29)'
        )
        say(
            f"  {DRIFT_LABEL} ({line['run']}) on {DRIFT_UNIT}: {entry['guarded_days']} guarded "
            f"days; with the reason, no figure: {reason}"
        )
        return [reason]
    say(f"  {DRIFT_LABEL} ({line['run']}) on {DRIFT_UNIT}: {'; '.join(said)}")
    return []


def _number(text) -> bool:
    try:
        return math.isfinite(float(text))
    except (TypeError, ValueError):
        return False


def part_regimes(root: Path, s, run: Run, say) -> list[str]:
    """Part (c) (module text): the reasons of reading V6; raises `CannotCheck`, `Failed` or
    `Lacks`."""
    label = learn_label()
    step, _, line = entry_step(s, run, label, "reading W3")
    _command_is(step, run_list.LEARN_MODULE, label)
    _, digest = matrix_of(root, step, label)
    document = learn_document(root, label, line, digest)
    units = _get(document, "units")
    units = units if isinstance(units, dict) else {}
    known = _known(root, run, step)
    absent = [u for u in ASR_UNITS if u not in units]
    missing = [u for u in absent if u not in known()] if absent else []
    pooled = [u for u in units if u.startswith("asr") and u not in ASR_UNITS]
    if missing or pooled:
        raise Lacks(
            f"{label}: the a.s.r. units are not scored as units of their own: missing "
            f"{missing or 'none'}, other a.s.r. units {pooled or 'none'}"
        )
    cells, reasons, gone = offer_cells(document, ASR_UNITS, label, known)
    say(
        f"  {label} ({line['run']}) scores {', '.join(ASR_UNITS)} as units of their own, "
        f"{len(cells)} configuration-unit cells with the offer and both shares"
        + (f"; with the reason, no figure: {_reasons_text(reasons)}" if reasons else "")
        + "".join(f"; {_gone_text(unit, reason)}" for unit, reason in gone.items())
    )
    return _pieces(reasons, gone)


def _rehearsal_only(root: Path) -> int:
    folder = Path(root) / RUNS_FOLDER / DRIVES_FOLDER
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.name.endswith(f"-rehearsal{LOG_SUFFIX}"))


def check(root: Path, say=print) -> int:
    """The three parts and the exit status (module text)."""
    root = Path(root)
    say(HEADER)
    begun = Findings(say)
    try:
        s = wp14.survey(root, begun)
    except CannotCheck as problem:
        begun.cannot_run(str(problem))
        s = None
    if s is not None and not s.paths and _rehearsal_only(root):
        say(
            f"  only rehearsal drive logs are present ({_rehearsal_only(root)}): the rehearsal is "
            "never read for WP15"
        )
    run = None
    if s is not None and not begun.cannot:
        try:
            run = the_run(root, s, begun)
        except CannotCheck as problem:
            begun.cannot_run(str(problem))
    if run is not None and run.incomplete:
        return _information(root, s, run, say)
    statuses = []
    for part, job in zip(PARTS, (part_rho, part_drift, part_regimes), strict=True):
        found, reasons = Findings(say), []
        if run is None:
            found.cannot_run(f"{part}: not decided, the run from the tag could not be read")
        else:
            try:
                reasons = job(root, s, run, say)
            except (Failed, Lacks) as problem:
                found.fail(f"{part}: {problem}")
            except CannotCheck as problem:
                found.cannot_run(f"{part}: {problem}")
            except Exception as problem:  # never Python's own 1
                found.cannot_run(f"{part}: {type(problem).__name__}: {problem}")
        if found.status() == PASSED:
            with_reason = f", with the reason: {'; '.join(reasons)}" if reasons else ""
            say(f"{part}: reported{with_reason}")
        statuses.append(found.status())
    others = Findings(say)
    if run is not None:
        wp14_refusals(root, others)
    status = FAILED if FAILED in (*statuses, others.status()) else max(*statuses, others.status())
    labels = ", ".join(needed_labels())
    say(
        f"{WORDS[status]}: WP15, {sum(x == PASSED for x in statuses)} of {len(PARTS)} parts "
        f"reported, from the entries {labels} of {LIST}"
        + ("" if run is None else f", in {run.name}")
        + ("" if others.status() == PASSED else "; WP14's pass mark does not pass")
    )
    return status


def wp14_refusals(root: Path, found: Findings) -> None:
    """WP15 is never passed while WP14 fails (decision 99 (5); plan section 13, WP15): WP14's own
    check (`sealed_run_check.check`: the marks run, fixes and hashes) is run on the same drive with
    its output collected, not printed. When it does not exit 0, every line of it that says a mark
    failed or could not run is printed here begun "WP14:", and `found` fails, so the exit status is
    1, whatever WP14's own (2 is could not run, 1 failed: either way the pass mark is not met).
    The check writes a results file into a run folder going now; it is given a throwaway folder
    of its own in the runs folder, removed after, so WP15's own run folder (when there is one)
    gets no file of it and the variable is put back."""
    runs = Path(root) / RUNS_FOLDER
    said: list[str] = []
    status = COULD_NOT_RUN
    saved = os.environ.get(RUN_DIR_VARIABLE)
    folder = None
    try:
        runs.mkdir(exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="wp15-wp14-", dir=runs))
        os.environ[RUN_DIR_VARIABLE] = str(folder)
        status = wp14.check(root, said.append)
    except Exception as problem:  # never Python's own 1
        said.append(f"COULD NOT RUN: {type(problem).__name__}: {problem}")
    finally:
        if saved is None:
            os.environ.pop(RUN_DIR_VARIABLE, None)
        else:
            os.environ[RUN_DIR_VARIABLE] = saved
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
    if status == PASSED:
        return
    named = [
        line
        for line in said
        if line.startswith(("FAILED: ", "COULD NOT RUN: "))
        or (
            ": " in line
            and line.rsplit(": ", 1)[1] in (wp14.WORDS[FAILED], wp14.WORDS[COULD_NOT_RUN])
        )
    ]
    for line in named or [f"exit status {status}"]:
        found.fail(f"WP14: {line.removeprefix('FAILED: ')}")


def part_needs() -> tuple[tuple[str, ...], ...]:
    """The entry labels each part reads, in the order of the parts."""
    rho = tuple(x for r in RHOS for x in (learn_label(r), intervals_label(r)))
    return rho, (DRIFT_LABEL,), (learn_label(),)


def _information(root: Path, s, run: Run, say) -> int:
    """The three parts of an incomplete drive that ran to its end, each as a line begun
    "information:" (module text, AN INCOMPLETE DRIVE); the exit status is FAILED."""

    def note(text: str) -> None:
        say(f"information: {text.strip()}")

    readable = 0
    for part, job, needs in zip(
        PARTS, (part_rho, part_drift, part_regimes), part_needs(), strict=True
    ):
        lost = [f"{label} ({run.lost[label]})" for label in needs if label in run.lost]
        if lost:
            note(f"{part}: lost, it needs a step that was not read: {'; '.join(lost)}")
            continue
        try:
            reasons = job(root, s, run, note)
        except (Failed, Lacks, Lost) as problem:
            note(f"{part}: not met in the steps that passed: {problem}")
            continue
        except CannotCheck as problem:
            note(f"{part}: could not be read: {problem}")
            continue
        except Exception as problem:  # never Python's own 1
            note(f"{part}: could not be read: {type(problem).__name__}: {problem}")
            continue
        with_reason = f", with the reason: {'; '.join(reasons)}" if reasons else ""
        note(f"{part}: readable from the steps that passed, not a pass{with_reason}")
        readable += 1
    say(
        f"{WORDS[FAILED]}: WP15, the held-out run is incomplete (WP14 fails), so its pass mark is "
        f"not passed; {readable} of {len(PARTS)} parts readable as information, from the entries "
        f"{', '.join(needed_labels())} of {LIST}, in {run.name}"
    )
    return FAILED


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check WP15's pass mark.")
    parser.add_argument("--root", type=Path, default=TOP, help="project top folder")
    args = parser.parse_args(argv)  # a wrong call exits with status 2
    try:
        return check(args.root)
    except Exception as problem:
        print(f"COULD NOT RUN: {type(problem).__name__}: {problem}")
        return COULD_NOT_RUN


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
