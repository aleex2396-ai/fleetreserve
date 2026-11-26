"""Development figures beside the outcome tables of WP16: the development-day checks and the dial
choice (plan section 13, WP16: "every number traces to a logged run").

WP16 reads the passed steps of the drive of the run from the tag. Two sets of figures that the
paper quotes come from development runs, which are no step of that drive: the checks on the
development days (`dev_day_checks`, whose `dev-day-checks.json` holds the figures of T2's check,
the T4 log, the order check and the two-way corner tests) and the dial choice (`dial_choice`,
whose `dial-choice.json` holds, per development unit, configuration and tier, the offer, its
spread over the splits, the losses and the gate reading). This module reads one finished run of
each, named on the command line of `wp16_run`, and writes one table per run:

    --dev-days-run <run folder>      table `dev-days-checks`
    --dial-choice-run <run folder>   table `dev-dial-choice`

Without the options nothing here runs and no table, file or exit status of `wp16_run` changes.
The run folder is a path under the top folder (`--root`), as the recorder wrote it
(`runs.noindex/<UTC start>-<label>`), or a copy of one elsewhere under the top folder. A folder
with a part named `data` on its path below the top folder is refused.

Readings:
  D1. The run is read through its own `manifest.json`, as a step of the drive is, without the
      drive log: the manifest must name an exit code of 0, no time-out, no interruption and no
      start error, a recorded command that runs `python -m <module>` of the table, and list the
      results file with the SHA-256 the file has. A run whose manifest shows a non-zero exit code
      is not read as passed: its table holds one row. A check that failed ends with 1 and gives
      "failed: <run> (it ended with exit status 1)"; one that could not run ends with 2 and gives
      "could not run: <run> (it ended with exit status 2)" (any other non-zero code is "failed").
      Both give exit status 1 of `wp16_run`. A manifest or a file that cannot be read, a hash
      that does not recompute, a run of another command, a file that is not JSON, a folder with a
      part named `data`: "unreadable: ...". Every file read is recorded with its SHA-256 beside
      the files of the drive (`Source.hashes`), and the manifest of `wp16_run` names the run, its
      commit and its command under `development_runs`.
      Each table of a run that was read has a row "commit of the run", the commit as the run's
      own `manifest.json` gives it, copied as written; a manifest with no commit (missing, null
      or empty) gives "no figure: no commit in the manifest" there (and in `development_runs`),
      not a refusal; a commit that is not text is "unreadable: ...".
  D2. Every figure is copied as the file writes it: nothing is computed, added up or pooled. A
      figure the file does not hold, or writes as null, is "no figure: <why>", never 0 or blank.
  D3. A figure the file contradicts in itself (a count above the count it is part of, a share that
      is not its count over its denominator, a verdict that does not follow from its failure
      lines and its coverage, a dial whose offer is not the offer of its configuration) is
      "unreadable: ..." in the cells concerned, and the exit status of `wp16_run` is 1 (the same
      rule as for the other tables: `wp16_run.any_unreadable`). A configuration or a gate reading
      that failed on a unit is "failed: <reason>" and also gives exit status 1.
  D4. The tables say "development" in their first column and their title, and in the note that
      follows the banner of the drive (`wp16_run` writes that banner first on every table, ahead
      of this module's own notes): they are not figures of the held-out run and fill no slot of
      plan section 11. The development-day run is
      made at the forecast setup its configuration rows state (not the headline setup of the
      learning runs); the dial-choice run is made on the development units at its own kappa.
  D5. The rows are in the long form (scope, reading, quantity, figure, key in the file), so that a
      figure is found by the name the writer gives it; the last column is that name as a dotted
      path through the JSON file (a configuration name that holds a dot is part of one name).
  D6. A run of the what-if kind (a `what_if` or `rotation_whatif` key in `dial-choice.json`) is
      not a run of the plan and is refused as unreadable.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from fleetreserve import plain_read
from fleetreserve.run_record import MANIFEST_NAME
from fleetreserve.wp16_cells import Missing, Table
from fleetreserve.wp16_drive import Missed, Source

# The writers' own names, read from their module texts and pinned by the tests against the
# writers' constants (`dev_day_checks.RESULTS_NAME`, `dev_day_verdicts`,
# `dial_choice.RESULTS_NAME`): importing the writers here would load their solver and learning
# code for every run.
DEV_DAYS_MODULE = "fleetreserve.dev_day_checks"
DEV_DAYS_FILE = "dev-day-checks.json"
DIAL_MODULE = "fleetreserve.dial_choice"
DIAL_FILE = "dial-choice.json"
PASSED, FAILED, COULD_NOT_RUN = "passed", "failed", "could not run"
VERDICTS = (PASSED, FAILED, COULD_NOT_RUN)
CHECKS = ("order", "t4_log", "t2", "two_way")
T4_GROUPS = (
    "meets T4's hypotheses",
    "other days, no in-window departure",
    "other days, in-window departure",
)

DEV_DAYS_NAME = "dev-days-checks"
DIAL_NAME = "dev-dial-choice"
DATA = "development"
COLUMNS = ("data", "scope", "reading", "quantity", "figure", "key in the file")
RUN_FOLDER = re.compile(r"\d{8}T\d{6}Z-[A-Za-z0-9_-]+")
ROUNDED = 1e-6  # the writers round some shares to six digits (`dev_day_verdicts._r`)
EXACT = 1e-9
NONE_LISTED = "none (the file lists none)"
NO_COMMIT = "no figure: no commit in the manifest"
COMMIT_KEY = "manifest.commit"  # the commit row's "key in the file": its place in manifest.json
EXIT_COULD_NOT_RUN = 2  # the exit status of a check that could not run (`run_record`'s users)
UNDER_DATA = "data"  # a path part that no run folder may have

COUNT, INTEGER, NUMBER, TEXT = "count", "integer", "number", "text"
WORDS = {
    COUNT: "a whole number of 0 or more",
    INTEGER: "a whole number",
    NUMBER: "a finite number",
    TEXT: "text",
}


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid(kind: str, value) -> bool:
    if kind == COUNT:
        return _whole(value) and value >= 0
    if kind == INTEGER:
        return _whole(value)
    if kind == NUMBER:
        number = isinstance(value, int | float) and not isinstance(value, bool)
        return number and math.isfinite(value)
    return isinstance(value, str)


def dig(document, path: tuple) -> tuple[bool, object]:
    """(whether the dictionaries of `document` hold every name of `path`, the value there)."""
    here = document
    for name in path:
        if not isinstance(here, dict) or name not in here:
            return False, None
        here = here[name]
    return True, here


def key_of(path: tuple) -> str:
    return ".".join(str(name) for name in path)


def valid_at(document, kind: str, *path):
    """The value at `path` when it is valid as `kind`, else None (a missing or null value, or one
    of another type, takes no part in a comparison)."""
    found, value = dig(document, path)
    return value if found and _valid(kind, value) else None


def flag(bad: dict, path: tuple, text: str) -> None:
    bad.setdefault(key_of(path), f"unreadable: {text}")


def not_above(bad: dict, document, small: tuple, big: tuple, why: str = "") -> None:
    """Flag `small` when both are counts and it is above `big`."""
    a, b = valid_at(document, COUNT, *small), valid_at(document, COUNT, *big)
    if a is not None and b is not None and a > b:
        flag(bad, small, f"{key_of(small)} {a} is above {key_of(big)} {b}{why}")


def is_share_of(bad: dict, document, share: tuple, top: tuple, bottom: tuple, tolerance: float):
    """Flag a share that is a number and not `top` over `bottom` (or given where `bottom` is 0)."""
    found, value = dig(document, share)
    up, of = valid_at(document, COUNT, *top), valid_at(document, COUNT, *bottom)
    if not found or value is None or up is None or of is None:
        return
    if not _valid(NUMBER, value):
        return
    if of == 0:
        flag(bad, share, f"{key_of(share)} {value!r} where {key_of(bottom)} is 0")
    elif abs(value - up / of) > tolerance:
        flag(bad, share, f"{key_of(share)} {value!r} is not {key_of(top)} {up} over {of}")


class Writer:
    """Adds the rows of one table: a figure is the file's own value at a path, or the text that
    says why there is none (readings D2, D3)."""

    def __init__(self, table: Table, document: dict, bad: dict):
        self.table, self.document, self.bad = table, document, bad

    def row(self, scope: str, reading: str, quantity: str, cell, path: tuple) -> None:
        self.table.add(DATA, scope, reading, quantity, cell, key_of(path))

    def figure(self, scope: str, reading: str, quantity: str, path: tuple, kind: str) -> None:
        key = key_of(path)
        found, value = dig(self.document, path)
        if key in self.bad:
            cell = Missing(self.bad[key])
        elif not found:
            cell = Missing(f"no figure: the file holds no {key}")
        elif value is None:
            cell = Missing(f"no figure: the file writes null for {key}")
        elif _valid(kind, value):
            cell = value
        else:
            cell = Missing(f"unreadable: {key} is {str(value)[:60]!r}, not {WORDS[kind]}")
        self.row(scope, reading, quantity, cell, path)

    def mapping(self, scope: str, reading: str, quantity: str, path: tuple, kind: str) -> None:
        """One row per entry of a dictionary of figures."""
        found, value = dig(self.document, path)
        if not found or not isinstance(value, dict):
            self.figure(scope, reading, quantity, path, kind)
        elif not value:
            self.row(scope, reading, quantity, NONE_LISTED, path)
        else:
            for name in sorted(value):
                self.figure(scope, reading, f"{quantity}: {name}", (*path, name), kind)

    def listing(self, scope: str, reading: str, quantity: str, path: tuple, kind: str) -> None:
        """One row per entry of a list."""
        found, value = dig(self.document, path)
        if not found or not isinstance(value, list):
            self.figure(scope, reading, quantity, path, kind)
        elif not value:
            self.row(scope, reading, quantity, NONE_LISTED, path)
        else:
            for k, item in enumerate(value, 1):
                cell = item
                if not _valid(kind, item):
                    cell = Missing(f"unreadable: {key_of(path)} entry {k} is not {WORDS[kind]}")
                self.row(scope, reading, f"{quantity} {k}", cell, (*path, k - 1))


@dataclass(frozen=True)
class Read:
    """A development run read through its manifest (reading D1)."""

    document: dict
    run: str  # the run folder below the top folder
    commit: str | None  # as the manifest gives it; None when it gives none
    command: list
    exit_code: int
    manifest_sha256: str
    results: str  # the file read
    results_sha256: str


def runs_module(command, module: str) -> bool:
    """Whether a recorded command runs `python -m <module>`."""
    if not isinstance(command, list):
        return False
    return any(command[k - 1 : k + 1] == ["-m", module] for k in range(1, len(command)))


def read_run(source: Source, folder: Path, module: str, name: str) -> Read:
    """The run `folder` (a path under the top folder, or below it when relative), once its manifest
    and the file `name` verify (reading D1); raises `Missed` with the text of the table's row."""
    label = Path(folder).name
    path = Path(folder) if Path(folder).is_absolute() else source.root / folder
    try:
        parts = plain_read.plain_parts(source.root, path)
    except plain_read.NotPlain:
        raise Missed(f"unreadable: {label}: not a plain path under the top folder") from None
    relative = "/".join(parts)
    if UNDER_DATA in parts:
        raise Missed(f"unreadable: {label}: a folder under {UNDER_DATA}/ is no run folder")
    if not RUN_FOLDER.fullmatch(parts[-1]):
        raise Missed(f"unreadable: {label}: not the name of a run folder (<UTC start>-<label>)")
    try:
        manifest_bytes = plain_read.read(source.root, path / MANIFEST_NAME)
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError, RecursionError) as problem:
        raise Missed(f"unreadable: {label}/{MANIFEST_NAME}: {problem}") from problem
    if not isinstance(manifest, dict):
        raise Missed(f"unreadable: {label}/{MANIFEST_NAME}: not a JSON object")
    code = manifest.get("exit_code")
    if not _whole(code):
        raise Missed(f"unreadable: {label}/{MANIFEST_NAME}: it holds no exit code")
    if code == EXIT_COULD_NOT_RUN:
        raise Missed(f"could not run: {label} (it ended with exit status {code})")
    if code != 0:
        raise Missed(f"failed: {label} (it ended with exit status {code})")
    if manifest.get("timed_out", False) is not False:
        raise Missed(f"failed: {label} (it hit its time limit)")
    if "interrupted" in manifest:
        raise Missed(f"failed: {label} (it was interrupted by {manifest['interrupted']})")
    if "error" in manifest:
        raise Missed(f"failed: {label} (it could not start)")
    commit, command = manifest.get("commit"), manifest.get("command")
    if commit is None or commit == "":
        commit = None  # no figure: the tables say so (reading D1)
    elif not isinstance(commit, str):
        raise Missed(f"unreadable: {label}/{MANIFEST_NAME}: its commit is not text")
    if not runs_module(command, module):
        raise Missed(f"unreadable: {label}: its recorded command does not run python -m {module}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get(name), str):
        raise Missed(f"unreadable: {label}/{name}: the manifest does not list it")
    try:
        data = plain_read.read(source.root, path / name)
    except OSError as problem:
        raise Missed(f"unreadable: {label}/{name}: {problem}") from problem
    digest = hashlib.sha256(data).hexdigest()
    if digest != outputs[name]:
        raise Missed(f"unreadable: {label}/{name}: it is not the file its run wrote")
    try:
        document = json.loads(data)
    except (ValueError, RecursionError) as problem:
        raise Missed(f"unreadable: {label}/{name}: not JSON: {problem}") from problem
    if not isinstance(document, dict):
        raise Missed(f"unreadable: {label}/{name}: not a JSON object")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    source.hashes[f"{relative}/{MANIFEST_NAME}"] = manifest_digest
    source.hashes[f"{relative}/{name}"] = digest
    return Read(document, relative, commit, command, code, manifest_digest, name, digest)


def commit_row(table: Table, commit: str | None) -> None:
    """The run's commit as its manifest gives it, or "no figure" when the manifest gives none."""
    cell = commit if commit else Missing(NO_COMMIT)
    table.add(DATA, "run", "", "commit of the run", cell, COMMIT_KEY)


# --- the development-day checks -------------------------------------------------------------------

DEV_DAYS_TITLE = (
    "Development figures: the checks on the development days (T2's check with the days in E', the "
    "T4 log, the order check, the two-way corner tests), as the run's dev-day-checks.json holds "
    "them"
)
DEV_DAYS_NOTE = (
    "Development figures read from one finished run of dev_day_checks, not from a step of the "
    "drive and not figures of the held-out run (module text of fleetreserve.wp16_devdays, "
    "readings D1 to D5). Each figure is the file's own, copied as written; the key is its name in "
    "the file. The configuration rows state the forecast setup of the run."
)

CONFIGURATION = (
    ("rho", "rho", NUMBER),
    ("budget_kwh", "two-way budget, kWh", NUMBER),
    ("energy_bounds", "energy bounds", TEXT),
    ("uncertifiable_car", "car with no usable forecast", TEXT),
    ("kappa", "kappa", NUMBER),
    ("departure_floor", "departure", TEXT),
    ("forecast_departure", "forecast of the departure", TEXT),
    ("forecast_reference", "forecast reference", TEXT),
    ("forecast_min_cell", "smallest cell of the forecast, cars", COUNT),
)
SAMPLE = (
    ("size", "T2 sample, days", COUNT),
    ("pass_mark_size", "T2 sample of the pass mark, days", COUNT),
    ("seed", "seed", INTEGER),
    ("label", "sample label", TEXT),
)
VERDICT_KEYS = (
    ("verdict", "verdict", TEXT),
    ("of_days", "days the check should cover", COUNT),
    ("covered_days", "days covered", COUNT),
    ("timeout_days", "days with a timeout", COUNT),
    ("other_missing_days", "days with another missing value", COUNT),
)
T4_KEYS = (
    ("days", "days", COUNT),
    ("days_with_hind_and_tree", "days with a hindsight and a tree value", COUNT),
    ("days_tree_below_hind", "days with the tree below hindsight", COUNT),
    ("mean_gap_hind_minus_tree_kwh", "mean gap hindsight minus tree, kWh", NUMBER),
    ("days_with_tree_and_affine", "days with a tree and an affine value", COUNT),
    ("days_affine_below_tree", "days with the affine value below the tree", COUNT),
    ("mean_gap_tree_minus_affine_kwh", "mean gap tree minus affine, kWh", NUMBER),
)
T2_KEYS = (
    ("certified_days", "sampled days certified", COUNT),
    ("days_in_e_prime", "days in E'", COUNT),
    ("share_of_sampled_days_in_e_prime", "share of sampled days in E'", NUMBER),
    ("share_of_certified_days_in_e_prime", "share of certified days in E'", NUMBER),
    ("steered_cars_on_certified_days", "steered cars on certified days", COUNT),
    ("share_of_steered_cars_in_e_prime_i", "share of steered cars in E'_i", NUMBER),
    ("cars_on_sampled_days", "cars on sampled days", COUNT),
    ("cars_left_unsteered_no_forecast", "cars left unsteered, no forecast", COUNT),
    ("cars_left_unsteered_empty_forecast_set", "cars left unsteered, empty forecast set", COUNT),
    ("forecast_rows_checked", "forecast rows checked", COUNT),
    ("replay_mismatches", "closed form against replay, mismatches", COUNT),
    ("cars_checked_in_e_prime_i", "cars in E'_i checked", COUNT),
    ("true_row_violations_in_e_prime_i", "cars in E'_i breaking a true row", COUNT),
    (
        "cars_outside_e_prime_i_breaking_a_true_row",
        "cars outside E'_i breaking a true row (counted, not judged)",
        COUNT,
    ),
    ("paths_with_request_missed", "corner paths with a request missed", COUNT),
)


def _verdict_problems(bad: dict, document: dict, name: str) -> None:
    """The verdict of one check must follow from its failure lines and its cover."""
    base = ("checks", name)
    path = (*base, "verdict")
    found, verdict = dig(document, path)
    lines = dig(document, (*base, "failure_lines"))[1]
    of = valid_at(document, COUNT, *base, "of_days")
    covered = valid_at(document, COUNT, *base, "covered_days")
    short = None not in (of, covered) and covered < of
    here = key_of(path)
    if found and verdict is not None and verdict not in VERDICTS:
        flag(bad, path, f"{here} is {str(verdict)[:40]!r}, not one of {VERDICTS}")
    elif verdict == PASSED and (lines or short):
        flag(bad, path, f"{here} is passed but the failure lines or the cover say not")
    elif verdict == FAILED and lines == []:
        flag(bad, path, f"{here} is failed but no failure line is listed")
    elif verdict == COULD_NOT_RUN and lines:
        flag(bad, path, f"{here} is could not run but failure lines are listed")
    not_above(bad, document, (*base, "covered_days"), (*base, "of_days"))


def _t4_problems(bad: dict, document: dict) -> None:
    t4 = ("checks", "t4_log")
    groups = (*t4, "groups")
    for name in T4_GROUPS:
        g = (*groups, name)
        not_above(bad, document, (*g, "days_tree_below_hind"), (*g, "days_with_hind_and_tree"))
        not_above(bad, document, (*g, "days_with_hind_and_tree"), (*g, "days"))
        not_above(bad, document, (*g, "days_affine_below_tree"), (*g, "days_with_tree_and_affine"))
        not_above(bad, document, (*g, "days_with_tree_and_affine"), (*g, "days"))
    sizes = [valid_at(document, COUNT, *groups, name, "days") for name in T4_GROUPS]
    of = valid_at(document, COUNT, *t4, "of_days")
    if None not in (*sizes, of) and sum(sizes) != of:
        flag(bad, (*t4, "of_days"), f"the three groups hold {sum(sizes)} days of {of}")
    not_above(bad, document, (*t4, "hypothesis_days_checked"), (*t4, "hypothesis_days"))
    first = valid_at(document, COUNT, *groups, T4_GROUPS[0], "days")
    held = valid_at(document, COUNT, *t4, "hypothesis_days")
    if None not in (first, held) and first != held:
        flag(bad, (*t4, "hypothesis_days"), f"hypothesis_days {held} is not the group's {first}")


def _t2_problems(bad: dict, document: dict) -> None:
    t2 = ("checks", "t2")
    not_above(bad, document, (*t2, "days_in_e_prime"), (*t2, "certified_days"))
    not_above(bad, document, (*t2, "certified_days"), (*t2, "of_days"))
    for share, bottom in (
        ("share_of_sampled_days_in_e_prime", "of_days"),
        ("share_of_certified_days_in_e_prime", "certified_days"),
    ):
        top = (*t2, "days_in_e_prime")
        is_share_of(bad, document, (*t2, share), top, (*t2, bottom), ROUNDED)
    path = (*t2, "share_of_steered_cars_in_e_prime_i")
    share = valid_at(document, NUMBER, *path)
    steered = valid_at(document, COUNT, *t2, "steered_cars_on_certified_days")
    if share is not None and (not 0 <= share <= 1 or steered == 0):
        flag(bad, path, f"{key_of(path)} is {share!r} with {steered} steered cars")
    not_above(bad, document, (*t2, "replay_mismatches"), (*t2, "forecast_rows_checked"))
    violations = (*t2, "true_row_violations_in_e_prime_i")
    not_above(bad, document, violations, (*t2, "cars_checked_in_e_prime_i"))


def dev_days_problems(document: dict) -> dict:
    """The cells the file contradicts (reading D3): their key -> the text of the cell."""
    bad: dict = {}
    for name in CHECKS:
        _verdict_problems(bad, document, name)
    days = valid_at(document, COUNT, "counting_days")
    size = valid_at(document, COUNT, "sample", "size")
    for name, whole in (("order", days), ("t4_log", days), ("t2", size), ("two_way", size)):
        path = ("checks", name, "of_days")
        of = valid_at(document, COUNT, *path)
        if None not in (of, whole) and of != whole:
            flag(bad, path, f"{name} covers {of} days where the file counts {whole}")
    _t4_problems(bad, document)
    _t2_problems(bad, document)
    two_way = ("checks", "two_way")
    not_above(bad, document, (*two_way, "days_both_products_matched"), (*two_way, "of_days"))
    return bad


def dev_days_table(document: dict, commit: str | None) -> Table:
    table = Table(DEV_DAYS_NAME, DEV_DAYS_TITLE, COLUMNS)
    commit_row(table, commit)
    write = Writer(table, document, dev_days_problems(document))
    for name, quantity, kind in CONFIGURATION:
        write.figure("configuration", "", quantity, ("configuration", name), kind)
    for name, quantity, kind in SAMPLE:
        write.figure("sample", "", quantity, ("sample", name), kind)
    write.mapping("sample", "", "counting days", ("sample", "counting_days"), COUNT)
    write.mapping("sample", "", "sampled days", ("sample", "sampled_days"), COUNT)
    write.figure("all days", "", "counting days", ("counting_days",), COUNT)
    scopes = {
        "order": "order check, every day",
        "t4_log": "T4 log, block product, every day",
        "t2": "T2's check, sampled days",
        "two_way": "two-way corner tests, sampled days",
    }
    for name in CHECKS:
        scope, base = scopes[name], ("checks", name)
        for key, quantity, kind in VERDICT_KEYS:
            write.figure(scope, "", quantity, (*base, key), kind)
        write.listing(scope, "", "failure line", (*base, "failure_lines"), TEXT)
        reasons = (*base, "could_not_run_reasons")
        write.listing(scope, "", "could not run because", reasons, TEXT)
        if name == "t4_log":
            hypotheses = "days meeting T4's hypotheses"
            write.figure(scope, "", hypotheses, (*base, "hypothesis_days"), COUNT)
            both = "of them with V_hind and V_caus"
            write.figure(scope, "", both, (*base, "hypothesis_days_checked"), COUNT)
            for group in T4_GROUPS:
                for key, quantity, kind in T4_KEYS:
                    write.figure(scope, group, quantity, (*base, "groups", group, key), kind)
            findings = (*base, "findings_tree_above_affine")
            write.listing(scope, "", "day with V_caus above V_aff", findings, TEXT)
        elif name == "t2":
            for key, quantity, kind in T2_KEYS:
                write.figure(scope, "", quantity, (*base, key), kind)
            by_rule = (*base, "days_without_certificate_by_rule")
            write.mapping(scope, "", "days without a certificate by the rule", by_rule, COUNT)
        elif name == "two_way":
            matched = "days on which both products' trees matched"
            write.figure(scope, "", matched, (*base, "days_both_products_matched"), COUNT)
    return table


# --- the dial choice -----------------------------------------------------------------------------

DIAL_TITLE = (
    "Development figures: the dial choice on the development units (offer, spread over the "
    "splits, losses and the gate reading, per unit, configuration and tier), as the run's "
    "dial-choice.json holds them"
)
DIAL_NOTE = (
    "Development figures read from one finished run of dial_choice, not from a step of the "
    "drive and not figures of the held-out run (module text of fleetreserve.wp16_devdays, "
    "readings D1 to D6). Each figure is the file's own, copied as written; the key is its name "
    "in the file. Offers are kWh per counting test day, one-way re-solve, calibrated on each "
    "split's calibration days; a split with no safe scale counts as an offer of 0. The dial "
    "named is measured, not fixed."
)
RUN_FIGURES = (
    ("label of the run", ("label",), TEXT),
    ("model", ("model",), TEXT),
    ("seeded splits per unit", ("run", "splits"), COUNT),
    ("kappa", ("run", "kappa"), NUMBER),
    ("rung read", ("run", "rung"), TEXT),
    ("ceiling of the gate run's first decision", ("run", "gate_ceiling"), TEXT),
    (
        "plan threshold: the offer's share of the ceiling",
        ("run", "thresholds", "share_of_ceiling"),
        NUMBER,
    ),
    (
        "plan threshold: the oracle-departure offer over the offer",
        ("run", "thresholds", "oracle_to_offer_ratio"),
        NUMBER,
    ),
)
UNIT_FIGURES = (
    ("status", "status", TEXT),
    ("counting_days", "counting days", COUNT),
    ("calibration_days_used", "calibration days in each split", COUNT),
    ("test_days", "test days in each split", COUNT),
)
TIER_FIGURES = (
    (("mean_offer_kwh_per_test_day",), "mean offer", NUMBER),
    (("offer_p10",), "offer, p10 over splits", NUMBER),
    (("offer_p50",), "offer, p50 over splits", NUMBER),
    (("offer_p90",), "offer, p90 over splits", NUMBER),
    (("splits_that_offer",), "splits that offer", COUNT),
    (("share_of_splits_that_offer",), "share of splits that offer", NUMBER),
    (("mean_scale",), "mean scale", NUMBER),
    (("mean_test_slot_loss",), "mean test slot loss (offering splits)", NUMBER),
    (
        ("slot_loss_over_limit", "share_of_splits_that_offer"),
        "share of offering splits with the slot loss over its limit",
        NUMBER,
    ),
    (("mean_test_driver_loss",), "mean test driver loss (offering splits)", NUMBER),
    (
        ("driver_loss_over_limit", "share_of_splits_that_offer"),
        "share of offering splits with the driver loss over its limit",
        NUMBER,
    ),
    (
        ("mean_share_of_best_causal_ceiling",),
        "offer as a share of the best-causal ceiling (offering splits only)",
        NUMBER,
    ),
)
GATE_FIGURES = (
    ("mean_offer_kwh_per_test_day", "mean offer over all splits", NUMBER),
    ("mean_oracle_offer_kwh_per_test_day", "oracle-departure offer over all splits", NUMBER),
    ("splits", "splits", COUNT),
)
CEILING_FIGURES = (
    ("splits_with_ceiling", "splits with the ceiling", COUNT),
    ("splits_left_out_ceiling_missing", "splits left out, ceiling missing", COUNT),
    ("mean_ceiling_kwh_per_test_day", "mean ceiling", NUMBER),
    ("mean_offer_kwh_per_test_day_on_those_splits", "mean offer on those splits", NUMBER),
    ("share_of_ceiling", "mean offer over mean ceiling", NUMBER),
)
OFFER = "mean_offer_kwh_per_test_day"


def dict_at(document: dict, *path) -> dict:
    found, value = dig(document, path)
    return value if found and isinstance(value, dict) else {}


def valid_list(document: dict, path: tuple) -> list:
    """The text entries of the list at `path` (empty when there is none)."""
    found, value = dig(document, path)
    if not found or not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def tiers_of(document: dict) -> list:
    return valid_list(document, ("run", "tiers")) or ["A", "B"]


def configurations_of(entry) -> list[tuple]:
    """(name, dial or None) of a unit's configurations: no rows first, then by dial."""
    found = entry.get("configurations") if isinstance(entry, dict) else None
    if not isinstance(found, dict):
        return []
    named = []
    for name, config in found.items():
        dial = config.get("driver_risk") if isinstance(config, dict) else None
        named.append((name, dial if _valid(NUMBER, dial) else None))
    return sorted(named, key=lambda pair: (pair[1] is not None, pair[1] or 0.0, pair[0]))


def configuration_label(dial) -> str:
    return "no rows" if dial is None else f"eps {dial:g}"


def _tier_problems(bad: dict, document: dict, base: tuple) -> None:
    not_above(bad, document, (*base, "splits_that_offer"), (*base, "splits"))
    share = (*base, "share_of_splits_that_offer")
    is_share_of(bad, document, share, (*base, "splits_that_offer"), (*base, "splits"), EXACT)
    value = valid_at(document, NUMBER, *share)
    if value is not None and not 0 <= value <= 1:
        flag(bad, share, f"{key_of(share)} {value!r} is not between 0 and 1")
    names = ("offer_p10", "offer_p50", "offer_p90")
    points = [valid_at(document, NUMBER, *base, name) for name in names]
    for k in (0, 1):
        low, high = points[k], points[k + 1]
        if None not in (low, high) and low > high + EXACT:
            flag(bad, (*base, names[k]), f"{names[k]} {low!r} is above the next point {high!r}")
    offer = valid_at(document, NUMBER, *base, OFFER)
    if offer is not None and offer < 0:
        flag(bad, (*base, OFFER), f"a negative mean offer {offer!r}")


def _gate_problems(bad: dict, document: dict, gate: tuple) -> None:
    total = valid_at(document, COUNT, *gate, "splits")
    for ceiling in valid_list(document, ("run", "ceilings")):
        base = (*gate, "ceilings", ceiling)
        kept = valid_at(document, COUNT, *base, "splits_with_ceiling")
        left = valid_at(document, COUNT, *base, "splits_left_out_ceiling_missing")
        if None not in (kept, left, total) and kept + left != total:
            path = (*base, "splits_with_ceiling")
            flag(bad, path, f"{kept} splits with the ceiling and {left} left out are not {total}")
        share = valid_at(document, NUMBER, *base, "share_of_ceiling")
        offer = valid_at(document, NUMBER, *base, "mean_offer_kwh_per_test_day_on_those_splits")
        mean = valid_at(document, NUMBER, *base, "mean_ceiling_kwh_per_test_day")
        whole = None not in (share, offer, mean)
        if whole and (mean <= 0 or abs(share - offer / mean) > EXACT * max(1.0, abs(share))):
            flag(bad, (*base, "share_of_ceiling"), f"share {share!r} is not {offer!r}/{mean!r}")


def _chosen_problems(bad: dict, document: dict) -> None:
    """The dial named for a unit and tier has the offer of its configuration."""
    units = dict_at(document, "units")
    for unit in dict_at(document, "dial_choice"):
        for tier in tiers_of(document):
            base = ("dial_choice", unit, tier)
            dial = valid_at(document, NUMBER, *base, "dial")
            offer = valid_at(document, NUMBER, *base, OFFER)
            if dial is None or offer is None:
                continue
            for name, found in configurations_of(units.get(unit)):
                config = ("units", unit, "configurations", name, "tiers", tier, OFFER)
                there = valid_at(document, NUMBER, *config)
                if found == dial and there is not None and abs(there - offer) > EXACT:
                    flag(bad, (*base, OFFER), f"offer {offer!r} is not that of {name} ({there!r})")


def dial_problems(document: dict) -> dict:
    """The cells the file contradicts (reading D3): their key -> the text of the cell."""
    bad: dict = {}
    for unit, entry in dict_at(document, "units").items():
        for name, _ in configurations_of(entry):
            config = ("units", unit, "configurations", name)
            for tier in tiers_of(document):
                _tier_problems(bad, document, (*config, "tiers", tier))
            _gate_problems(bad, document, (*config, "gate_reading"))
    _chosen_problems(bad, document)
    return bad


def _tier_rows(write: Writer, document: dict, unit: str, config: tuple, where: str) -> None:
    for tier in tiers_of(document):
        for path, quantity, kind in TIER_FIGURES:
            if tier != "A" and path[0] == "driver_loss_over_limit":
                continue  # the file writes it for tier A only
            figure = (*config, "tiers", tier, *path)
            write.figure(unit, f"tier {tier}, {where}", quantity, figure, kind)


def _gate_rows(write: Writer, document: dict, unit: str, config: tuple, where: str) -> None:
    gate = (*config, "gate_reading")
    reading = f"gate reading (tier A), {where}"
    failed = dig(document, (*gate, "failed"))
    if failed[0]:
        text = Missing(f"failed: {unit} | {config[-1]}: gate reading: {str(failed[1])[:200]}")
        write.row(unit, reading, "gate reading", text, (*gate, "failed"))
        return
    for key, quantity, kind in GATE_FIGURES:
        write.figure(unit, reading, quantity, (*gate, key), kind)
    ratio = (*gate, "oracle_to_offer_ratio")
    note = dig(document, (*gate, "oracle_to_offer_ratio_note"))[1]
    quantity = "oracle offer over mean offer"
    if dig(document, ratio) == (True, None) and isinstance(note, str):
        write.row(unit, reading, quantity, Missing(f"no figure: {note}"), ratio)
    else:
        write.figure(unit, reading, quantity, ratio, NUMBER)
    for ceiling in valid_list(document, ("run", "ceilings")):
        for key, quantity, kind in CEILING_FIGURES:
            figure = (*gate, "ceilings", ceiling, key)
            write.figure(unit, f"{reading}, ceiling {ceiling}", quantity, figure, kind)


def dial_table(document: dict, commit: str | None) -> Table:
    for refused in ("what_if", "rotation_whatif"):
        if refused in document:
            raise Missed(f"unreadable: {DIAL_FILE}: it holds {refused}, so it is not a plan run")
    table = Table(DIAL_NAME, DIAL_TITLE, COLUMNS)
    commit_row(table, commit)
    write = Writer(table, document, dial_problems(document))
    for quantity, path, kind in RUN_FIGURES:
        write.figure("run", "", quantity, path, kind)
    write.listing("run", "", "dial", ("run", "dials"), NUMBER)
    write.listing("run", "", "tier", ("run", "tiers"), TEXT)
    write.listing("run", "", "ceiling", ("run", "ceilings"), TEXT)
    write.mapping("run", "", "timeouts", ("timeouts",), COUNT)
    write.listing("run", "", "configuration that failed", ("failed_configurations",), TEXT)
    units = dict_at(document, "units")
    if not units:
        write.figure("run", "", "units", ("units",), TEXT)
    for unit in sorted(units):
        for key, quantity, kind in UNIT_FIGURES:
            write.figure(unit, "", quantity, ("units", unit, key), kind)
        entry = units[unit]
        if not isinstance(entry, dict) or entry.get("status") != "calibratable":
            continue  # the unit cannot be split: the file writes no numbers for it
        for tier in tiers_of(document):
            reading = f"tier {tier}, dial chosen"
            chosen = (("dial", "dial with the largest mean offer"), (OFFER, "its mean offer"))
            for key, quantity in chosen:
                write.figure(unit, reading, quantity, ("dial_choice", unit, tier, key), NUMBER)
        for name, dial in configurations_of(entry):
            config = ("units", unit, "configurations", name)
            where = configuration_label(dial)
            found = entry["configurations"][name]
            if isinstance(found, dict) and "failed" in found:
                text = Missing(f"failed: {unit} | {name}: {str(found['failed'])[:200]}")
                write.row(unit, where, "configuration", text, (*config, "failed"))
                continue
            _tier_rows(write, document, unit, config, where)
            _gate_rows(write, document, unit, config, where)
    return table


# --- the two options ---------------------------------------------------------------------

OPTIONS = (
    ("dev_days_run", DEV_DAYS_MODULE, DEV_DAYS_FILE, DEV_DAYS_NAME, DEV_DAYS_TITLE, DEV_DAYS_NOTE),
    ("dial_choice_run", DIAL_MODULE, DIAL_FILE, DIAL_NAME, DIAL_TITLE, DIAL_NOTE),
)
MAKERS = {"dev_days_run": dev_days_table, "dial_choice_run": dial_table}


def _failed(name: str, title: str, note: str, text: str) -> Table:
    table = Table(name, title, COLUMNS)
    table.add(DATA, "run", "", "read", Missing(text), "")
    table.notes.append(note)
    return table


def build(source: Source, folders: dict) -> tuple[list[Table], dict]:
    """The tables of the options that are given (`folders`: option -> run folder or None), and
    what the manifest of `wp16_run` records of the runs read (`development_runs`). A run that
    cannot be read is a table of one row that says why (reading D1)."""
    tables: list[Table] = []
    info: dict = {}
    for option, module, name, table_name, title, note in OPTIONS:
        folder = folders.get(option)
        if folder is None:
            continue
        try:
            read = read_run(source, Path(folder), module, name)
            table = MAKERS[option](read.document, read.commit)
        except Missed as problem:
            tables.append(_failed(table_name, title, note, problem.text))
            continue
        table.notes.append(note)
        at = f"at commit {read.commit}" if read.commit else "with no commit in its manifest"
        table.notes.append(
            f"Run {read.run.rsplit('/', 1)[-1]} {at}, exit code "
            f"{read.exit_code}; {read.results} has the SHA-256 {read.results_sha256}."
        )
        tables.append(table)
        info[option] = {
            "run": read.run,
            "commit": read.commit or NO_COMMIT,
            "command": read.command,
            "exit_code": read.exit_code,
            "manifest_sha256": read.manifest_sha256,
            "file": read.results,
            "file_sha256": read.results_sha256,
        }
    return tables, info
