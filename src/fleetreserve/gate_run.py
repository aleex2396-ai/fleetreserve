"""The WP6 gate run: development days through forecast, certificate, replay with the rungs,
calibration and the gate arithmetic, every number written to results files (plan 13 WP6).

Command line, from the project's top folder:
    python -m fleetreserve.gate_run --matrix <file> [--tables <folder>] [--truth-run <folder>]
        [--workers N] [--only <configuration name>] [--limit-days N]
        [--held-out-tables <folder>] [--held-out-truth-run <folder>]
        [--interval-seed S [--interval-resamples N]]

`--matrix`: the run matrix (`gate_config`; the committed one is `gate/matrix.json`). The truth
tables are used only if they are those of the recorded truth build at the checked-out commit,
checked exactly as `forecast_report` checks them (the latest `runs.noindex/*-wp2-truth` run, or
`--truth-run`, must pass `run_verify` and be at the commit checked out now, and each gate unit's
two tables must have the SHA-256 that build recorded). The build's setting (`main`, or `held`: no
charging before τ; plan 3) must be the setting the tables folder holds (`truth_build.SETTING_FILE`),
or the run could not run; without `--truth-run` the latest run of the tables' setting is used
(`*-wp2-truth` or `*-wp2-truth-held`). The setting is written to `gate.json` under `inputs` and
is the first line of the summary.

Per gate unit: the counting days (`days_from_truth`, with the matrix's readings), their two
ceilings on all cars (`gate_day.day_ceilings`), and the one seeded split (`day_split`, the same for
every configuration). The number of calibration days of a unit is `day_split.calibration_count`
of its counting days N, the matrix's `n_calibration` and `calibration_half_below` (the plan: 100, or
half of the days rounded down under 200). A unit whose count leaves no calibration day or no test
day is reported as "not calibratable: N days" and gets no calibrated number; any other unit is
split, and its calibration may still abstain (plan T6: too few offering calibration days). The
results state per unit N, the calibration days used and the test days. Per configuration of the
matrix (computed once per `Configuration.work_key`), per unit that is split: every day prepared
(`gate_day.prepare_day`), then calibrated and scored (`gate_calibrate.run_job`). The gate
decisions and the summary come from `gate_summary`.

Forecaster: `run_gate` takes a `factory`, called once per unit and process as
`factory(unit_days, matrix)`, that returns the forecaster `gate_day.prepare_day` reads. It must be
picklable (it is sent to the worker processes). The command line always uses the default,
`empirical_forecaster` (`forecast_empirical.EmpiricalForecaster` with the matrix's readings);
`learn_run` passes another one.

Held-out units (plan section 13, WP12 and WP14), through the one shared route of `sealed_route`
(`sealed_route.matrix_route`, the cold start's checks, in its order), which starts only at the tag
`frozen-plan`, or at a logged parsing fix of it (the held-out loader's module text), with a clean
tree; its checks stop an honest operator's ordinary mistakes and are not
a defence against deliberate evasion (`sealed_route`, module text, first paragraph, with the
commands that have the route and those that have none yet). When the matrix's gate units are
held-out units (all of them; a mix with other units could not run), before anything is read: every
folder named lies outside the held-out places of the top folder, the tag gate passes (the held-out
loader's checks 2 to 8: at the tag `frozen-plan` or a logged parsing fix of it, clean, the running
code the checked-out commit's) and
`--held-out-truth-run` is given. Then the run could not run, with no table read: `--limit-days` and
`--only` are refused (`sealed_route.trial_refusal`), and so is the run itself, since this command's
only forecaster, the empirical forecaster (either `reference`), builds a scored unit's forecasts
from that unit's own other days, test days included (`sealed_route.OUTCOME_READER`, as the cold
start refuses such a model, its reading C12). A held-out unit's offer is scored by `learn_run` with
a model fitted on training units; the other commands that score held-out units through the route
are named in `sealed_route`, module text, first paragraph. With other gate units the two held-out
options could not run, and without them nothing
above is checked: the run, its outputs and its summary are as before.

`--workers N`: the per-day work runs in a pool of N processes (N = 1: in this process). The results
files do not depend on N. `--only <name>`: only that configuration; the outputs say "PARTIAL RUN,
not a gate result". `--limit-days N`, for trials only: each unit keeps its first N counting days
(forecasts still use the whole cars table); every output says "TRIAL, not a gate result".
`--interval-seed S` (no default) and `--interval-resamples N` (default 2,000): after every job's
own calibration and scoring, the (day, scale) pairs the calibration searches of the N resamples
of seed S ask for are evaluated (`day_resample`) and written into the per-day losses file, for the
intervals over days (`day_intervals`, which needs them); every other output is the same as
without the option, the timing file apart.

Outputs, written into `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the
summary also on standard output:
- `gate.json`: every number (per unit, configuration and rung; tier A under `rungs.<rung>.test`
  and `gate`, tier B under `rungs.<rung>.tier_b` and `gate_tier_b`, plan section 6);
- `gate-days.csv`: one row per unit, configuration and day (tier B's offer and losses in the
  columns `<rung>_tier_b_*`, empty when tier B is not scored);
- `gate-t5-breaks.csv`: every T5 break, one row each (a header only when there is none);
- `gate-summary.txt`: the plain-English summary;
- `gate-timing.json`: wall-clock and solver seconds, kept apart from the compared results;
- `gate-day-losses.json`: the per-day losses the calibration read and the test days' losses at the
  chosen scales (`day_losses`), for the intervals over days (`day_intervals`).

The files a run may open (`check_inputs`, which many other commands that read truth tables call
too; which commands have the held-out route is said in `sealed_route`, module text, first
paragraph).
Before it opens anything, `check_inputs` refuses (`CouldNotRun`, so exit 2) every table, the
tables folder's setting file, and the truth run's summary and manifest, that lies in the data
folder of a working copy outside its derived tables (`refused_file`). A data folder is either
(a) a folder whose name is `data` in any spelling a file system that ignores letter case and
Unicode normalisation takes for it (the names compared after decomposition and case folding,
`folded`: `DATA` and `Data` are `data`) and whose parent folder holds a `.git` entry (a file or a
folder, as git writes in a main or a linked working copy) or is this package's own top folder
`TOP` (as written, or the same folder on disk); or (b) the folder that is `TOP`'s `data` folder on
disk (the same device and inode, `identity`), whatever it is called on the way in. A file in a
data folder is refused unless the part of its path after the data folder is named `derived`
exactly (so a path through `data/DERIVED/` is refused even where it is the derived folder). This
stops an honest operator's ordinary mistakes: a table, setting file, truth summary or manifest
named by a slip inside a working copy's data folder outside its derived tables, the sealed folder
or the raw files, in any spelling. It is not a defence against deliberate evasion (links or
copies made on purpose, a second working copy), and it proves nothing about the seal; that proof
is the plan's own (section 3 and WP13: the committed hashes, the held-out loader, which opens a
sealed file only at the tag `frozen-plan`, or at a logged parsing fix of it, with a clean tree,
and its openings log). Not refused:
a folder named `data` whose parent holds no `.git` entry and is not `TOP` (a plain folder of
that name, or a working copy's copied without its `.git`); another working
copy's data folder reached by a name that does not fold to `data` and is not a link (identity on
disk is compared with `TOP`'s data folder only, the one top folder known here; no such name is
known on a file system whose only equivalences are letter case and Unicode normalisation).
Each file is compared in every form it takes (`path_forms`): as written, made absolute from the
current folder with no call to the file system, then after each link on its path is replaced by
where it points, one link at a time, and last where its links lead. A form in a refused place
refuses the file before the next link is followed, so no part of a refused place is looked at
beyond the link that points into it; a link to a sealed file from a folder outside `data/` is
refused, and so is a sealed folder that is itself a link elsewhere, by its written form. Only
`stat` and `readlink` are asked of the file system: whether each part of the path is a link and
where it points, whether a `.git` entry exists beside a folder named `data`, and the identity on
disk of the folders on the path; no file is opened, listed, read or hashed until every file of
the run has passed. A path with more than `LINK_LIMIT` links is refused. Last, a file whose path
has passed is refused when it has more than one name on disk (`hard_linked`: a regular file whose
link count, read by one `stat` without opening it, is above 1). A hard link is a second name of
the same file, and no path shows the other names its file has, so a sealed file reached through a
hard link from a plain folder passes every test of its path. Every hard-linked file checked here
is therefore refused, sealed or not; that is the safe side, since the truth builds and the run
recorder write each file under one name, so a second name exists only where someone made one,
and a refused run loses nothing a fresh copy of the file (`cp`) does not restore.

Every file `gate_run`, `learn_run`, `compare_run` and `corner_run` are given passes the same check
before its first open (and every file the cold start is given, `cold_start`); the other commands
that call `check_inputs` check their tables, setting file and truth run this way, and of them
`ledger_run` also passes its matrix file through `check_files` (and, on the route, its run's
manifest and results files), and `drift_run` its matrix file on the route. `main` (and
`learn_run.main`, `compare_run.main` and `corner_run.main`) checks the matrix file `--matrix`
with `check_files` before reading it; every other argument of `main` and
`learn_run.main` is a folder (`--tables`, `--truth-run` and the two held-out
folders, whose files `check_inputs` checks before it opens any, and which the route's folder check
also checks), a number (`--workers`, `--limit-days`) or a name (`--only`; `--model`, a module of
the package). A truth run is read only with the manifest in its own folder: `check_inputs` refuses
a run folder that holds the truth summary but no manifest while a folder above it holds one
(`foreign_manifest`, asked by `stat` only), so `run_verify` never opens the manifest of a folder
above it; a run folder that does not exist, or one with no manifest in it or above it, is refused
by `run_verify` in its own words, as before. Not checked here: the files a model module opens
itself (the context files that `model_inputs.load_context` reads at fixed places under `TOP`'s
`data/raw/` and `learn/`, none in the sealed folder), and the output folder
`FLEETRESERVE_RUN_DIR`, which a run writes and never reads.

Path arguments (`argument_path`, `argument_paths`). Before any check or open, `main`,
`learn_run.main`, `compare_run.main`, `corner_run.main`, `ledger_run.main` (with its `--run` too)
and `cold_start.main` put each path argument (`--matrix`, `--tables`, `--truth-run`,
`--held-out-tables`, `--held-out-truth-run`) through `argument_path`, and `o3_run.main` and
`drift_run.main` do so on the route (`o3_run.take_route`, `drift_run.take_route`):
a path written as a URL, a scheme (a letter, then letters, digits, `+`, `-` or `.`, then `:`)
before its first `/` in any letter case (`file:`, `FILE:`, `http:`, `s3:`), could not run (exit
2), since pandas reads such a string as a URL and not as the folder of that name; any other path
has a leading `~` expanded (`os.path.expanduser`), as pandas would expand it. The checks then
judge the path as written and every form it takes up to where its links lead, which is its real
path (the path `os.path.realpath` gives: the last form of `path_forms`, `Path.resolve` in the
route's folder check), and the files are opened by that same path. `main` and `learn_run.main`
read each table once (`read_inputs`): its bytes are hashed against the build and the same bytes
are parsed (`days_from_truth.parse_unit_days`); the matrix file too (`gate_config.read`, as
`cold_start` reads it). So the SHA-256 a run records for its tables and matrix are those of the
rows and the matrix it ran on.

Exit status: 0 every configuration ran; 1 a configuration failed (or a day could not be built
from the tables); 2 could not run (a bad matrix or argument, a path written as a URL, a rotation
what-if matrix (`rotation_whatif` in `gate_config`; `learn_run` and `dial_choice` run those), a
file in a refused place, a table missing, no verified truth build at this commit, a table
differing from that build; for held-out units, a refusal of the route above).

Readings of the route, each with its alternative in brackets:
  R1. The kind of a run is that of the matrix's gate units, the units it scores. [Alternative: a
      command-line choice of the kind, checked against the units.]
  R2. A truth run is read only with the manifest of its own folder, on every run (above).
      [Alternative: `run_verify`'s search of the folders above the run for a manifest, as before;
      that search opens a file in a folder no check has seen.]
  R4. A path argument is checked as written (`~` expanded) and in every form it takes up to its
      real path, and is not replaced by its real path before the checks, so a path written inside
      a data folder whose link leads out of it is still refused by its written form, as before.
      [Alternative: each argument replaced by `os.path.realpath` before any check, which drops
      that refusal.]
The readings of what the route refuses are those of `sealed_route` (S1 to S3).
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from fleetreserve import (
    day_bootstrap,
    day_losses,
    day_resample,
    gate_calibrate,
    gate_day,
    gate_summary,
)
from fleetreserve.day_split import calibration_count, is_split, split_days
from fleetreserve.days_from_truth import ReadFiles, UnitDays, file_state, parse_unit_days
from fleetreserve.forecast_empirical import EmpiricalForecaster
from fleetreserve.gate_config import CEILINGS, TIER_LABELS, Matrix, MatrixError, read
from fleetreserve.run_record import MANIFEST_NAME
from fleetreserve.run_verify import NotVerified
from fleetreserve.rungs import RUNGS
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT, SETTING_FILE, read_setting
from fleetreserve.truth_build import SUMMARY_NAME as TRUTH_SUMMARY_NAME
from fleetreserve.truth_verify import (
    CouldNotRun,
    _head_commit,
    latest_run,
    load_summary,
    require_matching_setting,
)

RESULTS_NAME = "gate.json"
DAYS_NAME = "gate-days.csv"
BREAKS_NAME = "gate-t5-breaks.csv"
SUMMARY_NAME = "gate-summary.txt"
TIMING_NAME = "gate-timing.json"
DAY_LOSSES_NAME = "gate-day-losses.json"
TRIAL_TEXT = "TRIAL, not a gate result"
PARTIAL_TEXT = "PARTIAL RUN, not a gate result"
LEAST_DAYS_PLAN = 100  # plan 13 WP6: each gate unit >= 100 counting days


DATA_FOLDER = "data"
DERIVED_FOLDER = "derived"
GIT_ENTRY = ".git"
LINK_LIMIT = 40


def path_forms(path) -> Iterator[Path]:
    """Every form of `path` in order (module text): as written, made absolute with no call to the
    file system; after each link on it is replaced by where it points; last where it leads.
    Raises `OSError` past `LINK_LIMIT` links."""
    given = Path(path)
    full = given if given.is_absolute() else Path.cwd() / given
    yield Path(os.path.normpath(full))
    current, pending, links = Path(full.anchor), list(full.parts[1:]), 0
    while pending:
        name = pending.pop(0)
        if name == "..":
            current = current.parent
            continue
        step = current / name
        if not os.path.islink(step):
            current = step
            continue
        links += 1
        if links > LINK_LIMIT:
            raise OSError(f"more than {LINK_LIMIT} links on the path {path}")
        target = Path(os.readlink(step))
        if target.is_absolute():
            current = Path(target.anchor)
            pending = [*target.parts[1:], *pending]
        else:
            pending = [*target.parts, *pending]
        yield Path(os.path.normpath(current.joinpath(*pending)))
    yield current


def folded(name: str) -> str:
    """`name` as a file system that ignores letter case and Unicode normalisation compares it:
    decomposed (Unicode NFD), case folded, decomposed again (canonical caseless matching)."""
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", name).casefold())


def identity(path) -> tuple[int, int] | None:
    """The device and inode of what `path` leads to (links followed), or None when there is
    nothing there; one `stat`, nothing opened."""
    try:
        found = os.stat(path)
    except (OSError, ValueError):
        return None
    return found.st_dev, found.st_ino


def _is_data_folder(folder: Path, top_data: tuple | None) -> bool:
    """Whether `folder` is the data folder of a working copy or of `TOP` (module text):
    named `data` in any spelling, beside a `.git` entry or in `TOP`; or `TOP`'s own `data`
    folder on disk, whatever it is called on the way in."""
    if folded(folder.name) == folded(DATA_FOLDER):
        top = folder.parent
        if top == TOP or os.path.lexists(top / GIT_ENTRY):
            return True
        if identity(top) is not None and identity(top) == identity(TOP):
            return True
    return top_data is not None and identity(folder) == top_data


def _data_folder(form: Path, top_data: tuple | None) -> Path | None:
    """The data folder of a working copy (or of `TOP`) that `form` lies in outside its derived
    tables, the outermost such; None when there is none (module text). `top_data`: the identity
    of `TOP`'s data folder."""
    for folder in reversed(form.parents):
        if not _is_data_folder(folder, top_data):
            continue
        if form.parts[len(folder.parts)] != DERIVED_FOLDER:
            return folder
    return None


def refused_file(path) -> str | None:
    """Why a run may not open the file `path`, or None (module text)."""
    written = None
    top_data = identity(TOP / DATA_FOLDER)
    try:
        for form in path_forms(path):
            written = form if written is None else written
            folder = _data_folder(form, top_data)
            if folder is None:
                continue
            place = (
                f"the data folder {folder} of a working copy, outside its derived tables "
                f"{folder / DERIVED_FOLDER}"
            )
            if form == written:
                return f"{path} lies in {place}"
            return f"{path} leads through a link to {form}, which lies in {place}"
    except OSError as problem:
        return f"{path} cannot be followed: {problem}"
    return None


def hard_linked(path) -> str | None:
    """Why the file `path` is refused for having more than one name on disk, or None; one
    `stat`, nothing opened (module text). A missing file is left to the reader that needs it."""
    try:
        found = os.stat(path)
    except (OSError, ValueError):
        return None
    if not stat.S_ISREG(found.st_mode) or found.st_nlink <= 1:
        return None
    return (
        f"{path} has {found.st_nlink} names on disk (hard links); a file with more than one name "
        "may be another name of a sealed file, which no test of its path can see, so any file "
        "checked here with more than one name is refused, sealed or not"
    )


def check_files(paths) -> None:
    """Raises `CouldNotRun` for the first of `paths` a run may not open (module text)."""
    for path in paths:
        found = refused_file(path)
        if found is None:
            found = hard_linked(path)
        if found is not None:
            raise CouldNotRun(f"a file is refused before it is opened: {found}")


def foreign_manifest(folder: Path) -> bool:
    """True when `run_verify` would read the truth summary of the run `folder` with the manifest
    of a folder above it: the summary is there, the folder holds no manifest, and a folder above
    it does (by `stat` only, nothing opened; module text)."""
    if not (folder / TRUTH_SUMMARY_NAME).is_file() or (folder / MANIFEST_NAME).is_file():
        return False
    return any((above / MANIFEST_NAME).is_file() for above in Path(os.path.abspath(folder)).parents)


URL_FORM = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")  # a URL's scheme, as the readers take it
PATH_ARGUMENTS = ("matrix", "tables", "truth_run", "held_out_tables", "held_out_truth_run")


def argument_path(path) -> Path:
    """A path argument as a run takes it (module text): refused (`CouldNotRun`) when it is
    written as a URL (a scheme, letters then `:`, before its first `/`, in any letter case:
    `file:`, `FILE:`, `http:`, `s3:`), else with a leading `~` expanded (`os.path.expanduser`),
    so every check and every open take the same path."""
    text = os.fspath(path)
    found = URL_FORM.match(text.split("/", 1)[0])
    if found is not None:
        raise CouldNotRun(
            f"the path {text} is written as a URL ({found.group(0)}); a run reads only files on "
            "this machine named by a path, never a URL; nothing read"
        )
    return Path(os.path.expanduser(text))


def argument_paths(args, names=PATH_ARGUMENTS) -> None:
    """Every path argument of `args` named in `names` (absent or None: skipped) put through
    `argument_path` in place; raises `CouldNotRun` for the first refused."""
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, argument_path(value))


def check_inputs(run: Path | None, tables: Path, units) -> dict:
    """The verified truth build's summary for `units`; raises `CouldNotRun` when the tables
    cannot be used (as `forecast_report.check_inputs`, for the gate units). Every file it opens
    is first checked by `check_files` (module text)."""
    return read_inputs(run, tables, units)[0]


def read_inputs(run: Path | None, tables: Path, units) -> tuple[dict, dict[str, bytes]]:
    """`check_inputs`'s record and the bytes of each table it hashed, by file name: each table
    is read once, so a caller that parses these bytes (`days_from_truth.parse_unit_days`) scores
    the rows whose SHA-256 the record holds; they come as a `days_from_truth.ReadFiles`, so that
    parse refuses a table changed after its read (module text)."""
    tables = Path(tables)
    names = [f"{unit}-{kind}.csv" for unit in units for kind in ("cars", "days")]
    check_files([tables / SETTING_FILE, *(tables / name for name in names)])
    folder = run if run is not None else latest_run(TOP / "runs.noindex", read_setting(tables))
    manifest = Path(folder) / MANIFEST_NAME
    check_files([Path(folder) / TRUTH_SUMMARY_NAME, manifest])
    if foreign_manifest(Path(folder)):  # never the manifest of a folder above it (module text)
        raise CouldNotRun(
            f"{manifest} is missing; a truth build is read only with the manifest of its own run "
            "folder"
        )
    try:
        summary = load_summary(folder, _head_commit())
    except NotVerified as problem:
        raise CouldNotRun(f"the recorded truth build is not verified: {problem}") from problem
    setting = require_matching_setting(summary, tables, folder.name)
    hashes, read, states = {}, {}, {}
    for unit in units:
        entry = summary["units"].get(unit)
        if entry is None:
            raise CouldNotRun(f"{unit} is not in the recorded truth build {folder.name}")
        for kind in ("cars", "days"):
            path = tables / f"{unit}-{kind}.csv"
            if not path.is_file():
                raise CouldNotRun(f"{path} is missing")
            states[path.name] = file_state(path)  # just before the read (`ReadFiles`)
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry["sha256"][kind]:
                raise CouldNotRun(
                    f"{path.name} differs from the table the recorded build {folder.name} wrote"
                )
            hashes[path.name] = digest
            read[path.name] = data
    record = {"setting": setting, "truth_run": folder.name, "table_sha256": hashes}
    return record, ReadFiles(read, dict.fromkeys(read, tables), states)


# --- the per-day work, in this process or in a pool ----------------------------------------------

_WORKER: dict = {}


def empirical_forecaster(unit_days: UnitDays, matrix: Matrix) -> EmpiricalForecaster:
    """The default forecaster factory: the empirical forecaster with the matrix's readings."""
    return EmpiricalForecaster(unit_days, reference=matrix.reference, min_cell=matrix.min_cell)


def _start_worker(units: dict, matrix: Matrix, factory=empirical_forecaster) -> None:
    _WORKER.clear()
    _WORKER.update(units=units, matrix=matrix, factory=factory, forecasters={})


def _forecaster(unit: str):
    found = _WORKER["forecasters"].get(unit)
    if found is None:
        found = _WORKER["factory"](_WORKER["units"][unit], _WORKER["matrix"])
        _WORKER["forecasters"][unit] = found
    return found


def settings(matrix: Matrix) -> gate_day.DaySettings:
    return gate_day.DaySettings(
        matrix.rho,
        matrix.two_way_budget_kwh,
        matrix.shape_tolerance,
        matrix.zero_value_kwh,
        matrix.t5_tolerance_kwh,
    )


def _ceiling_task(job):
    unit, index = job
    return gate_day.day_ceilings(_WORKER["units"][unit].days[index].day)


def _prepare_task(job):
    unit, index, configuration = job
    truth = _WORKER["units"][unit].days[index]
    return gate_day.prepare_day(
        truth, _forecaster(unit), configuration, settings(_WORKER["matrix"])
    )


def _evaluate_task(job):
    prepared, beta = job
    return gate_day.evaluate(prepared, beta).compact()


class Runner:
    """Maps the per-day tasks in this process (`workers` 1) or in a process pool."""

    def __init__(self, workers: int, units: dict, matrix: Matrix, factory=empirical_forecaster):
        self.workers = workers
        _start_worker(units, matrix, factory)
        self.pool = None
        if workers > 1:
            self.pool = ProcessPoolExecutor(
                max_workers=workers, initializer=_start_worker, initargs=(units, matrix, factory)
            )

    def map(self, task, jobs) -> list:
        jobs = list(jobs)
        if self.pool is None:
            return [task(job) for job in jobs]
        return list(self.pool.map(task, jobs))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


# --- one run ---------------------------------------------------------------------------------


@dataclass
class UnitPlan:
    name: str
    days: UnitDays
    keys: list  # "date site" per day, in the order of `days.days`
    roles: list  # "calibration", "test" or "not calibratable" per day
    status: str
    split: dict
    calibration_days: int = 0  # the calibration days used; 0 when the unit is not split
    test_days: int = 0  # the test days; 0 when the unit is not split


def plan_unit(name: str, days: UnitDays, matrix: Matrix) -> UnitPlan:
    keys = [f"{t.record.date} {t.record.site}" for t in days.days]
    count = calibration_count(len(keys), matrix.n_calibration, matrix.calibration_half_below)
    if not is_split(len(keys), count):
        roles = ["not calibratable"] * len(keys)
        status = f"not calibratable: {len(keys)} days"
        return UnitPlan(name, days, keys, roles, status, {})
    split = split_days(name, matrix.split_seed, keys, n_calibration=count, min_days=count + 1)
    chosen = set(split.calibration)
    roles = ["calibration" if k in chosen else "test" for k in keys]
    record = {"seed": matrix.split_seed, "calibration": list(split.calibration)}
    return UnitPlan(
        name, days, keys, roles, "calibratable", record, len(split.calibration), len(split.test)
    )


def _limit(days: UnitDays, limit: int | None) -> UnitDays:
    if limit is None:
        return days
    return replace(days, days=days.days[:limit])


def _fields(configuration) -> dict:
    return asdict(configuration)


def _job_groups(matrix: Matrix, only: str | None) -> list[list]:
    """Configurations grouped by work key, in matrix order."""
    chosen = [c for c in matrix.configurations if only is None or c.name == only]
    groups: dict = {}
    for c in chosen:
        groups.setdefault(c.work_key(), []).append(c)
    return list(groups.values())


def run_gate(
    matrix: Matrix,
    units: dict,
    *,
    workers: int,
    only: str | None,
    trial: bool,
    factory=empirical_forecaster,
    selection: bool = True,
    loss_records: list | None = None,
    intervals: tuple[int, int] | None = None,
    counted_units: dict | None = None,
    held_out: bool = False,
):
    """Everything but the input checks and the writing of files. `factory` makes each unit's
    forecaster (module text). `selection` false: no kappa selection per unit is computed or
    written (`gate_summary.gate_tables`; the sealed route, through `learn_run`). `loss_records`: a
    list that is given the per-day loss records of the calibrated jobs (`day_losses.records`),
    when the caller passes one. `intervals`: (seed, resamples) of `--interval-seed`; the
    resamples' pairs are then evaluated after every job's own calibration and scoring
    (`day_resample`) and written into those records. `held_out` (the sealed route only): a unit
    whose configuration, or whose ceilings, raised an error is written as `failed: <error>` under
    each configuration it failed in (`assemble`); a unit whose ceilings raised fails under all of
    them and its days are not prepared. Off the route that error ends the run, as before."""
    started = time.perf_counter()
    timing = {"workers": workers, "phases": {}, "jobs": {}}
    plans = {name: plan_unit(name, units[name], matrix) for name in matrix.gate_units}
    groups = _job_groups(matrix, only)
    runner = Runner(workers, units, matrix, factory)
    try:
        t0 = time.perf_counter()
        ceilings, ceiling_failed = {}, {}
        for name, plan in plans.items():
            try:
                ceilings[name] = runner.map(
                    _ceiling_task, [(name, i) for i in range(len(plan.keys))]
                )
            except Exception as problem:
                if not held_out:
                    raise
                # sealed route only: the unit is recorded as failed under every configuration
                ceiling_failed[name] = f"{type(problem).__name__}: {problem}"
        timing["phases"]["ceilings_seconds"] = time.perf_counter() - t0
        timing["ceilings_solver_seconds"] = sum(
            c.seconds for found in ceilings.values() for c in found
        )
        t0 = time.perf_counter()
        prepared, failed = {}, {}
        for group in groups:
            head = group[0]
            for name, plan in plans.items():
                if name in ceiling_failed:
                    failed[(head.name, name)] = ceiling_failed[name]
                    continue
                try:
                    prepared[(head.name, name)] = runner.map(
                        _prepare_task, [(name, i, head) for i in range(len(plan.keys))]
                    )
                except Exception as problem:  # recorded, the other configurations go on
                    failed[(head.name, name)] = f"{type(problem).__name__}: {problem}"
        timing["phases"]["prepare_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        jobs = [
            (group[0].name, name)
            for group in groups
            for name, plan in plans.items()
            if plan.status == "calibratable" and (group[0].name, name) in prepared
        ]

        def one(job):
            begun = time.perf_counter()
            head, unit = job
            try:
                found = gate_calibrate.run_job(
                    prepared[job],
                    ceilings[unit],
                    plans[unit].roles,
                    lambda pairs: runner.map(_evaluate_task, pairs),
                    tier_b="B" in matrix.tiers,
                )
            except Exception as problem:  # recorded, the other configurations go on
                return job, None, f"{type(problem).__name__}: {problem}", begun
            return job, found, None, begun

        if workers > 1 and jobs:
            with ThreadPoolExecutor(max_workers=len(jobs)) as threads:
                finished = list(threads.map(one, jobs))
        else:
            finished = [one(job) for job in jobs]
        timing["phases"]["calibrate_seconds"] = time.perf_counter() - t0
        resampled = None
        if intervals is not None:
            t0 = time.perf_counter()
            resampled = _resample(finished, prepared, plans, intervals, matrix, runner, workers)
            timing["phases"]["interval_pairs_seconds"] = time.perf_counter() - t0
            timing["interval_pairs"] = {
                f"{job[0]} | {job[1]}": {
                    "pairs": len(found.evaluated),
                    "failed": len(found.failed),
                    "own_calibration_pairs": found.own_calibration_pairs,
                    "seconds": found.seconds,
                }
                for job, found in resampled.items()
            }
    finally:
        runner.close()
    results = {}
    for job, found, problem, _begun in finished:
        if problem is not None:
            failed[job] = problem
            continue
        results[job] = found
        cache = found[3]
        timing["jobs"][f"{job[0]} | {job[1]}"] = _job_timing(cache, prepared[job])
    timing["phases"]["total_seconds"] = time.perf_counter() - started
    if loss_records is not None:
        loss_records.extend(
            day_losses.records(groups, plans, prepared, ceilings, results, resampled)
        )
    assembled = assemble(
        matrix,
        plans,
        groups,
        ceilings,
        prepared,
        results,
        failed,
        trial,
        only,
        selection,
        counted_units,
        held_out,
    )
    return assembled, timing


def _resample(finished, prepared, plans, intervals, matrix, runner, workers) -> dict:
    """The resamples' pairs of every calibrated job (`day_resample`), by job."""
    seed, resamples = intervals
    tiers = ("A", "B") if "B" in matrix.tiers else ("A",)
    done = [(job, found) for job, found, problem, _ in finished if problem is None]

    def one(item):
        job, found = item
        unit = job[1]
        return job, day_resample.resample_pairs(
            prepared[job],
            found[3].store,
            plans[unit].roles,
            unit,
            seed,
            resamples,
            tiers,
            lambda pairs: runner.map(_evaluate_task, pairs),
        )

    if workers > 1 and done:
        with ThreadPoolExecutor(max_workers=len(done)) as threads:
            return dict(threads.map(one, done))
    return dict(one(item) for item in done)


def _job_timing(cache, prepared) -> dict:
    evaluations = list(cache.store.values())
    return {
        "evaluations": len(evaluations),
        "evaluation_seconds": sum(e.seconds for e in evaluations),
        "resolve_seconds_by_rung": {
            rung: sum(e.rungs[rung].resolve_seconds for e in evaluations) for rung in RUNGS
        },
        "certificate_seconds": sum(p.certificate_seconds for p in prepared),
    }


def assemble(
    matrix,
    plans,
    groups,
    ceilings,
    prepared,
    results,
    failed,
    trial,
    only,
    selection=True,
    counted_units=None,
    held_out=False,
):
    """The results document, the day rows, the T5 break rows and the gate test data.
    `counted_units` (the sealed route only; `sealed_route.abstained_gate_data`): units the run did
    not score that the gate decisions still count, each with the test data every configuration
    gets for it, `{"reason": ...}`: the unit then passes neither rule, as a unit that cannot be
    calibrated does (`gate_summary`), and stays in the denominator of every decision."""
    unit_entries = {}
    for name, plan in plans.items():
        outcomes = Counter(c.best_causal_outcome for c in ceilings.get(name, ()))
        unit_entries[name] = {
            "counting_days": len(plan.keys),
            "calibration_days_used": plan.calibration_days,
            "test_days": plan.test_days,
            "at_least_100_counting_days": len(plan.keys) >= LEAST_DAYS_PLAN,
            "calibratable": plan.status == "calibratable",
            "status": plan.status,
            "split": plan.split,
            "readings": plan.days.counts(),
            "best_causal_ceiling_outcomes": dict(sorted(outcomes.items())),
        }
    configurations, day_rows, break_rows, test_data = {}, [], [], {}
    test_data_b = {}
    t5 = {"day_paths_checked": 0, "breaks_a": 0, "breaks_b": 0}
    timeouts = Counter()
    for group in groups:
        head = group[0]
        names = [c.name for c in group]
        per_unit = {}
        for unit, plan in plans.items():
            job = (head.name, unit)
            days = prepared.get(job)
            if job in failed:
                if held_out:
                    text = f"failed: {failed[job]}"
                    per_unit[unit] = {"failed": text, "reason": text}
                else:
                    per_unit[unit] = {"failed": failed[job], "reason": "configuration failed"}
                entry_rows = None
            elif plan.status != "calibratable":
                per_unit[unit] = {
                    "reason": plan.status,
                    "certificate_outcomes": dict(
                        sorted(Counter(p.certificate_outcome for p in days).items())
                    ),
                }
                entry_rows = [{} for _ in days]
            else:
                found, entry_rows, breaks, _ = results[job]
                per_unit[unit] = found
                for key in ("day_paths_checked", "breaks_a", "breaks_b"):
                    t5[key] += found["t5"][key]
                for rung in RUNGS:
                    for role in ("calibration", "test"):
                        timeouts["re_solves"] += found["rungs"][rung]["resolves"][role].get(
                            "timeout", 0
                        )
                for b in breaks:
                    break_rows.append(_break_row(unit, names, plan, days, b))
            if days is not None:
                timeouts["certificates"] += sum(p.certificate_outcome == "timeout" for p in days)
            for c in group:
                test_data[(c.name, unit)] = _test_data(
                    per_unit[unit], plan, days, ceilings.get(unit), matrix.gate_rung
                )
                if "B" in matrix.tiers:
                    test_data_b[(c.name, unit)] = _test_data(
                        per_unit[unit], plan, days, ceilings.get(unit), matrix.gate_rung, "B"
                    )
            if entry_rows is not None:
                for c in group:
                    day_rows += _day_rows(unit, c.name, plan, days, ceilings[unit], entry_rows)
        for c in group:
            configurations[c.name] = {
                "configuration": _fields(c),
                "work_shared_with": [n for n in names if n != c.name],
                "units": per_unit,
            }
    for name in plans:
        timeouts["best_causal_ceilings"] += sum(
            c.best_causal_outcome == "timeout" for c in ceilings.get(name, ())
        )
    ordered = {
        c.name: configurations[c.name] for c in matrix.configurations if c.name in configurations
    }
    document = {
        "definitions": definitions(matrix),
        "calibration_rule": calibration_rule(matrix),
        "trial": trial,
        "partial": only is not None,
        "label": TRIAL_TEXT if trial else (PARTIAL_TEXT if only is not None else "gate run"),
        "units": unit_entries,
        "configurations": ordered,
        "t5": t5,
        "timeouts": dict(sorted(timeouts.items())),
        "failed_configurations": sorted(f"{a} | {b}: {why}" for (a, b), why in failed.items()),
    }
    gate_matrix = matrix
    if only is not None:
        gate_matrix = replace(
            matrix, configurations=tuple(c for c in matrix.configurations if c.name in ordered)
        )
    if counted_units:
        gate_matrix = replace(gate_matrix, gate_units=(*gate_matrix.gate_units, *counted_units))
        for c in matrix.configurations:
            for unit, entry in counted_units.items():
                test_data[(c.name, unit)] = test_data_b[(c.name, unit)] = dict(entry)
        document["counted_units_not_scored"] = {u: e["reason"] for u, e in counted_units.items()}
    document["gate"] = gate_summary.gate_tables(gate_matrix, test_data, selection)
    if "B" in matrix.tiers:
        document["gate_tier_b"] = gate_summary.gate_tables(gate_matrix, test_data_b, selection)
    return document, day_rows, break_rows


def calibration_rule(matrix) -> dict:
    """The matrix values of the calibration-days rule, as the results state them."""
    return {
        "n_calibration": matrix.n_calibration,
        "calibration_half_below": matrix.calibration_half_below,
        "rule": CALIBRATION_RULE_TEXT,
    }


CALIBRATION_RULE_TEXT = (
    "a unit with N counting days has n_calibration calibration days when N is at least "
    "calibration_half_below (null: always), else N // 2; it is split when that leaves at least "
    "one calibration day and one test day, and the calibration may still abstain on a unit that "
    "is split (too few offering calibration days)"
)


def definitions(matrix) -> dict:
    """Words the numbers of the results document need, in one place."""
    return {
        "offering_day": (
            "a counting day whose box at scale 1 has value above zero_value_kwh "
            f"({matrix.zero_value_kwh}); no steered car, no optimal certificate or a value at or "
            "below it makes the day abstain, whatever the scale"
        ),
        "failing_day": "an offering test day with slot loss above 0",
        "losses": (
            "fields ending in offering_days are means over the offering test days; at a scale of 0 "
            "nothing is offered, so they and failing_days are null"
        ),
        "evaluations": (
            "evaluations.calibration counts evaluations of offering calibration days only; "
            "evaluations.test counts the test days at the chosen scales, abstaining days included; "
            "both count tier A's scales only"
        ),
        "tiers": {
            "A": (
                f"{TIER_LABELS['A']}: the scale is the smaller of the slot and driver scales; "
                "rungs.<rung>.test and the key gate"
            ),
            "B": (
                f"{TIER_LABELS['B']}: the scale comes from the slot loss alone; the driver loss is "
                "reported and cars staying to their bound are checked for shortfall; "
                "rungs.<rung>.tier_b and the key gate_tier_b"
            ),
            "reported": list(matrix.tiers),
            "pooling": "the tiers are reported side by side and never pooled or averaged",
            "deciding_tier": "A",
        },
    }


def _test_data(entry, plan, days, ceilings, rung, tier="A") -> dict:
    """What the gate decisions read of one unit under one configuration, for `tier`."""
    if "rungs" not in entry:
        return {"reason": entry.get("reason", "no result")}
    test = [d for d, role in enumerate(plan.roles) if role == "test"]
    scored = entry["rungs"][rung] if tier == "A" else entry["rungs"][rung]["tier_b"]
    shown = scored["test"]
    return {
        "offering_days": shown["offering_days"],
        "failing_days": shown["failing_days"],
        "test_days": shown["days"],
        "values": [days[d].value for d in test],
        "ceilings": {
            name: [gate_calibrate.ceiling_values(ceilings[d], name) for d in test]
            for name in CEILINGS
        },
        "beta": float(scored["beta_hat"]),
    }


def _day_rows(unit, configuration, plan, days, ceilings, entry_rows) -> list[dict]:
    rows = []
    for d, (p, c) in enumerate(zip(days, ceilings, strict=True)):
        reasons = Counter(r for _, r in p.unsteered)
        row = {
            "unit": unit,
            "configuration": configuration,
            "site": p.site,
            "date": p.date,
            "role": plan.roles[d],
            "cars": p.n_cars,
            "steered_cars": len(p.steered),
            "unsteered_no_reference": reasons[gate_day.NO_REFERENCE],
            "unsteered_empty_set": reasons[gate_day.EMPTY_SET],
            "abstain_reason": p.abstain_reason or "",
            "certificate_outcome": p.certificate_outcome,
            "certified_value_kwh": repr(p.value),
            "ceiling_hindsight_kwh": repr(c.hindsight),
            "ceiling_best_causal_kwh": "" if c.best_causal is None else repr(c.best_causal),
        }
        for rung in RUNGS:
            scored = entry_rows[d].get(rung)
            for field in ("offer_kwh", "slot_loss", "driver_loss"):
                row[f"{rung}_{field}"] = "" if scored is None else repr(scored[field])
            scored_b = None if scored is None else scored.get("tier_b")
            for field in ("offer_kwh", "slot_loss", "driver_loss"):
                row[f"{rung}_tier_b_{field}"] = "" if scored_b is None else repr(scored_b[field])
        rows.append(row)
    return rows


DAY_COLUMNS = (
    "unit",
    "configuration",
    "site",
    "date",
    "role",
    "cars",
    "steered_cars",
    "unsteered_no_reference",
    "unsteered_empty_set",
    "abstain_reason",
    "certificate_outcome",
    "certified_value_kwh",
    "ceiling_hindsight_kwh",
    "ceiling_best_causal_kwh",
    *(f"{rung}_{field}" for rung in RUNGS for field in ("offer_kwh", "slot_loss", "driver_loss")),
    *(
        f"{rung}_tier_b_{field}"
        for rung in RUNGS
        for field in ("offer_kwh", "slot_loss", "driver_loss")
    ),
)
BREAK_COLUMNS = (
    "unit",
    "configurations",
    "site",
    "date",
    "beta",
    "path_index",
    "path",
    "kind",
    "rung",
    "predecessor",
    "car",
    "value",
    "predecessor_value",
)


def _break_row(unit, names, plan, days, item) -> dict:
    b = item["break"]
    p = days[item["day"]]
    path = " ".join(f"{z:+.0f}" for z in p.paths[b.path])
    return {
        "unit": unit,
        "configurations": ";".join(names),
        "site": p.site,
        "date": p.date,
        "beta": repr(item["beta"]),
        "path_index": b.path,
        "path": path,
        "kind": b.kind,
        "rung": b.rung,
        "predecessor": b.predecessor,
        "car": "" if b.car is None else b.car,
        "value": "" if b.value is None else repr(b.value),
        "predecessor_value": "" if b.predecessor_value is None else repr(b.predecessor_value),
    }


def _csv(rows, columns, top: str | None) -> str:
    handle = io.StringIO()
    if top is not None:
        handle.write(f"# {top}\n")
    writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def table_top(document) -> str | None:
    """The label line the tables of a trial or partial run start with; None for a full run."""
    return document["label"] if document["trial"] or document["partial"] else None


def day_table(day_rows, document) -> str:
    """The text of `gate-days.csv` (`DAY_COLUMNS`) for the day rows of a run's `document`."""
    return _csv(day_rows, DAY_COLUMNS, table_top(document))


def write_outputs(folder: Path, document, day_rows, break_rows, text, timing) -> None:
    top = table_top(document)
    (folder / RESULTS_NAME).write_text(_json(document))
    (folder / DAYS_NAME).write_text(day_table(day_rows, document))
    (folder / BREAKS_NAME).write_text(_csv(break_rows, BREAK_COLUMNS, top))
    (folder / SUMMARY_NAME).write_text(text)
    timing = dict(timing, label=document["label"])
    (folder / TIMING_NAME).write_text(_json(timing))


def interval_option(seed: int | None, resamples: int | None) -> tuple[int, int] | None:
    """(seed, resamples) of `--interval-seed` and `--interval-resamples`, None without them;
    raises `MatrixError` on a bad value (module text)."""
    if seed is None:
        if resamples is not None:
            raise MatrixError("--interval-resamples needs --interval-seed")
        return None
    if seed < 0:
        raise MatrixError("--interval-seed must be a non-negative integer")
    count = day_bootstrap.RESAMPLES if resamples is None else resamples
    if count < 1:
        raise MatrixError("--interval-resamples must be at least 1")
    return seed, count


def main(argv: list[str] | None = None, root: Path = TOP) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--held-out-tables", type=Path, default=None)
    parser.add_argument("--held-out-truth-run", type=Path, default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--interval-seed", type=int, default=None)
    parser.add_argument("--interval-resamples", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        argument_paths(args)
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if args.limit_days is not None and args.limit_days < 1:
            raise MatrixError("--limit-days must be at least 1")
        intervals = interval_option(args.interval_seed, args.interval_resamples)
        check_files([args.matrix])
        matrix, matrix_sha256 = read(args.matrix)
        if matrix.rotation_whatif:
            raise MatrixError(
                "a rotation what-if matrix (rotation_whatif) is run by learn_run and dial_choice, "
                "which label it and check the model; this command's default forecaster fits "
                "nothing and its outputs carry no such label"
            )
        if args.only is not None:
            matrix.configuration(args.only)
    except (MatrixError, CouldNotRun) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    # imported here: `sealed_route` imports this module for `check_inputs`
    from fleetreserve import sealed_route

    try:
        route = sealed_route.matrix_route(
            root,
            matrix.gate_units,
            args.tables,
            args.truth_run,
            args.held_out_tables,
            args.held_out_truth_run,
        )
    except sealed_route.RouteRefused as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    if route.held_out:
        trial_options = {"--limit-days": args.limit_days, "--only": args.only}
        forecaster = f"the empirical forecaster of gate_run (reference {matrix.reference})"
        refusal = sealed_route.trial_refusal(trial_options) or sealed_route.OUTCOME_READER.format(
            what=forecaster
        )
        print(f"COULD NOT RUN: {refusal}")
        return 2
    try:
        inputs, table_bytes = read_inputs(args.truth_run, args.tables, matrix.gate_units)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        units = {
            name: _limit(
                parse_unit_days(
                    table_bytes,
                    name,
                    rho=matrix.rho,
                    no_fit=matrix.no_fit,
                    small_fleet=matrix.small_fleet,
                ),
                args.limit_days,
            )
            for name in matrix.gate_units
        }
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    trial = args.limit_days is not None
    loss_records: list = []
    (document, day_rows, break_rows), timing = run_gate(
        matrix,
        units,
        workers=args.workers,
        only=args.only,
        trial=trial,
        loss_records=loss_records,
        intervals=intervals,
    )
    document["inputs"] = inputs
    document["matrix"] = {
        "file": args.matrix.name,
        "sha256": matrix_sha256,
        "limit_days": args.limit_days,
        "only": args.only,
    }
    text = "\n".join(gate_summary.summary_lines(document, matrix)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        write_outputs(Path(run_dir), document, day_rows, break_rows, text, timing)
        (Path(run_dir) / DAY_LOSSES_NAME).write_text(day_losses.text(loss_records))
    return 1 if document["failed_configurations"] else 0


if __name__ == "__main__":
    sys.exit(main())
