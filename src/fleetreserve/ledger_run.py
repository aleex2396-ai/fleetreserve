"""The ledger of plan T9 and outcome O6 for the hourly product, from one recorded scoring run (plan
sections 1, 2, 5 (T9), 11 (O5, O6)).

Command line, from the project's top folder:
    python -m fleetreserve.ledger_run --run <run folder> --matrix <file> --tree-days N --seed S
        [--tables <folder>] [--truth-run <folder>] [--workers N]
        [--held-out-tables <folder>] [--held-out-truth-run <folder>]

The run (`--run`) is one recorded run of `python -m fleetreserve.gate_run` (it read `gate.json`
and `gate-days.csv`) or of `python -m fleetreserve.learn_run` (`learn-gate.json` and
`learn-gate-days.csv`), named by its recorded command. It is read only when both files are
verified (`run_evidence`, `run_verify`: the run ended with exit 0, was neither stopped nor timed
out, the files are those it wrote), and each of the two is parsed from the bytes its check hashed
(`run_verify.read_verified`), the run refused (could not run, exit 2) when after the parse a file
no longer holds those bytes (`run_verify.still_verified`). Off the sealed route (below) it is
read when it was recorded at the commit checked out now or, by the earlier-run rule of
`run_evidence`, at an earlier commit since which only the files that rule keeps aside changed
(else it is stale). On the sealed route
it is read only when it was recorded at the commit checked out now: the tag's commit, or the fix
commit at a logged parsing fix (reading P-h). It must not be a trial (`--limit-days`), and
`--matrix` must be the file it ran on (the SHA-256 it recorded). A partial run (`--only`) is read,
and its label leads every output. Every path argument first goes through `gate_run.argument_path`
(a path written as a URL could not run; a leading `~` is expanded), and the matrix file passes
`gate_run.check_files` before it is read, as in `learn_run` (reading L11).

What the ledger measures. The ceiling V_hind and the values V_caus, V_aff and V_diag, the largest
box in LP (24)'s set and the block product's values are computed on the true sets of the scored
test days: they are hindsight quantities of those days by definition (plan T9 prices what the
offer loses for not knowing the day), and so are the terms built from them alone (R, A, F,
V_aff - V_diag, the price of the block). U = V_aff - V_cert is V_aff, on the true sets, less the
run's certified value V_cert, which the run computed on its forecast sets (plan section 2: T2's
LP on forecast sets). The offer and K come from the run's own scale and certified value. Every
term is measured after the run scored its offers, and none is used to choose anything: no scale,
configuration, kappa, dial or day is picked from them, here or in the run.

What is fitted, chosen, tuned, sampled or calibrated here, and from which days (on the sealed
route below and off it alike):
1. Fitted, tuned or calibrated: nothing. The scale of the offer and of K and the certified value
   V_cert are the run's own, read from its files. The run calibrated the scale on each unit's
   calibration days (on the route, the held-out build's own drawn days: `learn_run` checks them,
   and the ledger checks them again below), and every day row's role must be the split's (the
   days and roles of the tables, below).
2. Sampled: the tree sample, `--tree-days` test days per split unit drawn with `--seed`
   (`block_sample.draw`): the test days with the smallest seeded SHA-256 ranks of the unit's name,
   the seed and the day's text "site date". It is drawn from the test days' names alone: no value,
   ceiling, offer, loss or outcome of any day enters it, so it selects nothing by results. The seed
   and the number of days are settings fixed before the run (the plan gives no number: T9, "is
   measured on a day sample").
3. Chosen: nothing per unit. Every configuration, tier and rung the run scored is tabled, and no
   configuration, kappa, dial or day is picked by a result; the ledger has no option that selects
   a setting per unit (on the route the run made no kappa selection per unit either: its record's
   `kappa_selection`).
4. Read from the scored units' days: the true sets of their test days (the hindsight values above)
   and the roles of their calibration days, only to rebuild the split; no term is computed on a
   calibration day.
5. The base set of F is the day-average base set of each day's own true cars (reading L5), one rule
   for every day, not chosen per unit.

Held-out units, through the sealed route (`sealed_route`, plan sections 3 and 13: WP12 rehearses
it on development units, WP14 runs it from the tag). The ledger scores nothing itself: on held-out
units it reads a `learn_run` run made through the route, and the scored units' truth tables only
from the held-out truth build's output. When the matrix's gate units are held-out units (all of
them; a mix with other units could not run), before anything is read: `sealed_route.matrix_route`
(every folder named lies outside the held-out places of the top folder; the tag gate passes, the
held-out loader's checks 2 to 8: at the tag `frozen-plan` or at a logged parsing fix of it, clean,
the running code the checked-out commit's;
`--held-out-truth-run` is given, and `--held-out-tables` defaults to the held-out build's main
folder), the run folder too lies outside those places (`sealed_route.check_folders`), and its
manifest and results files pass `gate_run.check_files`. The run is then read as above, and it must
be a run through the route at the checked-out commit and setting: its manifest's commit is the
commit checked out now, the one the gate passed at (the tag's commit, or the fix commit at a
logged parsing fix), else it could not run before anything else is read, with one line naming the
run, its commit and the commit taken (reading P-h); it is then read through
`run_evidence.read_evidence` at that commit in the ordinary way, so neither the earlier-run rule
nor that module's rule E3 is reached. Its results hold the route's record (`sealed_route.mark`,
the key `sealed_route`) naming the tag's commit and, at a logged parsing fix, the fix commit, as
the gate passes now (no fix commit at the tag); a run without that record, or whose record names
another tag commit or fix commit, a fix commit at the tag or none at a fix, could not run, with
one line naming the commits of the record and the gate's in full. Before any table
is read the matrix's split rule must be the held-out build's (`sealed_route.check_split_rule`).
The two builds are read through `sealed_route.read_two_builds`: the development build (`--tables`,
`--truth-run`) verified at this commit with none of its tables read (the ledger fits nothing;
reading L12), the scored units' tables from the held-out build (`--held-out-tables`,
`--held-out-truth-run`) checked against that build's recorded run, the two of one setting. The
development build must be the one the run read (its name under `inputs`), and the held-out build
and each scored unit's two tables the ones the run read (the build's name, its setting and the
SHA-256 under `sealed_route.held_out_inputs`), or the run could not run. The days are parsed from
the bytes hashed, and each scored unit's calibration days as the split draws them must be those the
build recorded (`sealed_route.check_split_days`). Everything else is as above. `ledger.json` then
holds `sealed_route` (the tag's commit, `fix_commit` at a logged parsing fix and no such key at the
tag, the run read, the two builds' checks, the run's fit units
and kappa selection, `checks`: what the route's checks stop and what they do not,
`sealed_route.CHECKS`, and `hindsight`, the paragraph "What the ledger measures" in short), and the
summary's first three lines say so: a held-out result through the route, the `checks` text, and
the `hindsight` text. With other gate units the two held-out options could not run, and without
them nothing of the route is checked: the run, its outputs and its summary are as before.

The true days. The ledger needs the corner tree, the diagonal value and LP (24) on true sets, so
it reads the truth tables of the run's gate units as `gate_run` does (`gate_run.check_inputs`: the
recorded truth build at this commit, each table with the SHA-256 that build recorded), and each
table must have the SHA-256 the run recorded. The days and their split are rebuilt
(`days_from_truth.load_unit_days` with the matrix's readings, `gate_run.plan_unit`), and each
configuration's day rows in the run must be those days in that order with those roles.

Per configuration of the run, gate unit that is split, tier (A, and B when the run scored it) and
rung (`rungs.RUNGS`: fixed plan, one-way re-solve, two-way at the run's budget; "rungs tabled
separately on the audited set"), on the test days (the days the run's offer is a mean over), per
day (`ledger_terms.day_terms`): the ceiling V_hind (`ceiling_hindsight_kwh`), V_aff
(`ceiling_best_causal_kwh`), V_cert (`certified_value_kwh`), the rung's scale for the tier
(`rungs.<rung>.beta_hat`, tier B `rungs.<rung>.tier_b.beta_hat`), the run's offer of the day,
and R + A, U, K. Two sets of days, each with every mean over its own days:

- every test day with V_aff (a test day whose best-causal ceiling is missing in the run is left
  out and counted): offer + (R + A) + U + K = ceiling per day and in the means;
- the tree sample: `--tree-days` test days per unit drawn with `--seed` (`block_sample.draw`, the
  seeded hash ranking, on the texts "site date"; all test days when a unit has fewer), each
  solved by `ledger_day.solve_day` in `--workers` processes; the days whose tree ended optimal
  and that have V_aff: offer + R + A + U + K = ceiling per day and in the means.

Per unit (true sets, so the same for every configuration): R + A over the test days; R and A on
the tree sample, with the days on which A is not 0 counted as a finding (T4's tested equality
predicts A = 0 on the hourly product; A is never clipped); V_aff - V_diag; the format loss F
(the ceiling minus the largest hourly box in LP (24)'s set), beside the sum and not in it; and the
block product on the tree sample: its R and A against its own ceiling, its V_aff - V_diag and the
price of the block (hourly V_hind minus block V_hind), once per unit, with U and K
`ledger_terms.NOT_COMPUTED_BLOCK` (the gate certifies the hourly product only).

Outcome O8's ceiling ratio V (`ceiling_ratio`, its readings C1 to C6), per split unit with a
ledger, at the matrix's `two_way_budget_kwh` (the budget of the run's two-way rung; no option sets
another): on every test day the exact one-way and two-way hindsight ceilings of the day's true
cars (`ceiling_ratio.day_ceilings`, the decision hour and the recovery hours free), their sums over
the test days and V, the two-way sum over the one-way sum; the test days whose one-way ceiling is
0 are counted and named; per configuration and tier, the run's two-way offer as a share of the
two-way ceiling over the same days. The label is written from the budget ("B=5" only at 5 kWh).
The discharge D is printed as not run (the owner's decision of 16 November 2026).

Outcome O8's offer ratio X and outcome O5's offer order (`ceiling_ratio.offer_ratio`, readings C7
to C9), per split unit with a ledger, configuration and tier, from the day rows' offers of the
three rungs (`<rung>_offer_kwh`, tier B `<rung>_tier_b_offer_kwh`) on the test days: the sum of the
two-way offer over the sum of the one-way offer; the test days whose one-way offer is 0, counted
and named; the share of test days on which fixed <= one-way and, apart, the share on which one-way
<= two-way (outcome O8's daily pass), the other days of each step named with their three offers. A
day that breaks either step fails check 7 (reading C8). The offers are the run's own: no file is
opened and nothing is computed for X beyond sums and comparisons of them.

Nothing is fitted, chosen or calibrated for V or X, on the sealed route or off it, and no file is
opened for them beyond those read above.

Checks (any one failing makes the exit status 1, the outputs are still written, and the summary
leads with it):

1. The identity on every day and in every mean, to `ledger_terms.IDENTITY_TOLERANCE`. The offer is
   the run's and K is (1 - beta) V_cert from the run's scale and certified value, so this checks
   that every recorded offer is the scale times the certified value.
2. The run's own offer: per configuration, unit, tier and rung, the mean of the day rows' offers
   over all test days equals the run's `test.mean_offer_kwh_per_day`, and the sums of both
   ceilings over the test days equal the run's `ceiling_sums_test_days_kwh`.
3. The sampled days are the run's days: the recomputed `hourly_hind` and `hourly_causal` equal
   the run's two ceilings of the day (within `CROSS_TOLERANCE` of max(1, |value|)).
4. The plan's order (`ledger_terms.below_order`): R + A, R, A, V_aff - V_diag, F and the block's
   R, A, V_aff - V_diag and price are not below 0.
5. Every sampled day's sets could be built.
6. The ceiling ratio's one-way ceilings: on every test day of every configuration, the one-way
   ceiling recomputed for V equals the run's `ceiling_hindsight_kwh` (within `CROSS_TOLERANCE` of
   max(1, |value|)); a difference, or a test day whose sets could not be built, names the day.
7. The offer order (plan T5, outcome O5: "a violation is a bug"): on every test day of every
   configuration and tier, the run's fixed offer is at most its one-way offer and its one-way
   offer is at most its two-way offer (`ceiling_ratio.ORDER_TOLERANCE`). A broken step is one
   break, naming the unit, configuration, tier, step, the number of test days it breaks and the
   first of them with their three offers.

A solve that did not end optimal (a timeout) is a recorded outcome, counted as missing; it is not a
failed check.

Readings (where the plan and the run do not decide; each alternative in brackets):

  L1. The ceiling of R is `exact`'s hourly ceiling with the decision hour free (the gate's
      "hindsight-free-decision-hour"), as decided on 16 November 2026; R is reported as measured,
      never written as 0. [Alternative: a hindsight with the decision hour's power one
      plan for every path, under which R is 0 by construction; not built.]
  L2. K's scale is the rung's audited-set scale of the tier, one per unit, configuration, rung and
      tier of the run's one split. The every-request route's scale (`fixed_every_request`) is not
      tabled. [Alternative: table it as a fourth row.]
  L3. Every rung, the two-way rung included, is measured against the one-way ceilings the run
      records: its offer is its scale times the one-way certificate. [Alternative: the two-way
      ceilings for the two-way rung, which would need two-way V_aff and V_caus as well.]
  L4. R and A are split on the tree sample only; R + A is measured on every test day.
      [Alternative: the tree on every test day, at up to 300 s a day.]
  L5. F: true sets with the day-average base set (`ledger_day`), a reading confirmed on
      21 November 2026. [Alternatives: forecast sets (each configuration's
      steered cars' forecast rows, so F would differ per configuration); the template base set
      (a pooling rule over training days is needed, which the plan does not set).]
  L6. F, V_aff - V_diag and the block product's values are measured on the tree sample.
      [Alternative: on every test day.]
  L7. The hourly V_diag is computed here (`ledger_day`), since neither the run nor
      `ceiling_values` gives it. [Alternative: report the hourly V_aff - V_diag as not given.]
  L8. A share of the ceiling is the mean term over the mean ceiling of the same days (the ratio
      of sums), so the shares of offer, R, A, U and K add up to 1. [Alternative: the mean of daily
      ratios, undefined on a day whose ceiling is 0.]
  L9. A test day whose best-causal ceiling is missing in the run leaves both sets of days, and is
      counted. [Alternative: the run's offer mean taken as given and the identity left open.]
  L10. The price of the block is reported once per unit, on the tree sample, as a share of the
      hourly ceiling; the block's R and A are shares of the block ceiling. [Alternative: one
      figure over every unit together.]
  L11. The path arguments and the matrix file are checked on every run, as `learn_run` checks
      them, since whether the route applies is known only once the matrix is read; on the
      existing made-up runs nothing changes. [Alternative: checked on the route only, the kind
      of a run then given by a command-line choice checked against the units.]
  L12. On the route the development build is checked (verified at this commit, of the held-out
      build's setting, the build the run read) and none of its tables is read: the ledger fits
      nothing, so the fit units' tables are not hashed again. [Alternative: read the run's fit
      units' tables and compare them with the SHA-256 under its `inputs`.]
  L13. A run through the route is known by the route's record in its results naming the tag's
      commit and the fix commit the gate passes at (none at the tag); its command is checked
      only as a scoring run's (a `gate_run` run never holds the record, since on held-out units
      it could not run). [Alternative: also require the command
      `learn_run`.]
  P-h. On the sealed route the ledger reads a scoring run only when it is a run through the route
      recorded at the commit checked out now, the one the gate passed at: the tag's commit, or,
      when the gate passes at a logged parsing fix, that fix commit. Nothing recorded at any
      other commit is read, whatever data set it scores: after a logged fix every scoring run is
      made again at the fix commit, as the truth builds are (`truth_verify` takes a build only at
      the commit checked out now), so one ledger never reads a run of a data set parsed by a
      loader other than the checked-out commit's. Confirmed on 17 November 2026; it is
      not in the `readings` list of `ledger.json`, which off the route stays as before.
      [Alternative: take truth builds and scoring runs per data set along the fix chain (a run of
      a set whose loader files are byte-identical to the fix commit's, at any commit of the chain
      from the tag to the fix), which would save making the unaffected sets' runs again after a
      fix, at the price of a longer rule.]

Outputs, written into `FLEETRESERVE_RUN_DIR` when it is set (the summary also on standard output):
- `ledger.json`: every number, the definitions, the readings' list and the checks;
- `ledger-days.csv`: one row per configuration, unit, tier, rung and test day, with every term;
- `ledger-sample.csv`: one row per sampled day, every value of `ledger_day` with its outcome and the
  run's two ceilings beside the recomputed ones;
- `ledger-ceiling-ratio.csv`: one row per test day of a split unit with a ledger: its one-way and
  two-way ceilings, the run's recorded one-way ceiling and the budget;
- `ledger-summary.txt`: the plain-English summary (`ledger_summary`);
- `ledger-timing.json`: solve seconds and the seconds of the ceiling ratio's ceilings, kept apart
  from the compared results.

Exit status: 0 every one of the seven checks held; 1 a check failed (a broken offer order, check 7,
among them; the outputs are written first), or a day could not be built from the tables;
2 could not run (a bad argument or matrix, a path written as a URL, a matrix file in a refused
place, a run that is not verified, stale, a trial or not a scoring run, a matrix other than the
run's, no verified truth build at this commit, a table other than the run's, the run's days not
those of the tables; for held-out units, a refusal of the route above, a run not made through it
or not recorded at the commit checked out now (reading P-h), a build, setting or table other than
the run's, a split rule or calibration days other than the held-out build's).
"""

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fleetreserve import (
    ceiling_ratio,
    gate_run,
    learn_run,
    ledger_summary,
    ledger_terms,
    rotation_whatif,
    sealed_route,
)
from fleetreserve.block_sample import day_text, draw
from fleetreserve.ceiling_values import DayTask
from fleetreserve.days_from_truth import load_unit_days, parse_unit_days
from fleetreserve.gate_config import CEILINGS, MatrixError, load
from fleetreserve.gate_summary import ROUTE_KEY
from fleetreserve.ledger_day import KEYS, LP24_BOX, solve_day
from fleetreserve.model_inputs import LOOK_AHEAD_KEY
from fleetreserve.run_evidence import CannotCheck, read_evidence
from fleetreserve.run_record import MANIFEST_NAME
from fleetreserve.run_verify import NotVerified, read_verified, still_verified
from fleetreserve.rungs import RUNGS
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun, _head_commit

RESULTS_NAME = "ledger.json"
DAYS_NAME = "ledger-days.csv"
SAMPLE_NAME = "ledger-sample.csv"
RATIO_NAME = "ledger-ceiling-ratio.csv"
SUMMARY_NAME = "ledger-summary.txt"
TIMING_NAME = "ledger-timing.json"
CROSS_TOLERANCE = 1e-6  # relative to max(1, |value|): a recomputed ceiling against the run's
SCORING = {
    "fleetreserve.gate_run": (gate_run.RESULTS_NAME, gate_run.DAYS_NAME),
    "fleetreserve.learn_run": (learn_run.GATE_NAME, learn_run.DAYS_NAME),
}
TIERS = ("A", "B")
HOURLY_ORDER = ("r", "a", "affine_minus_diagonal", "format_loss")
BLOCK_ORDER = ("r", "a", "affine_minus_diagonal", "price_of_the_block")
READINGS = tuple(f"L{k}" for k in range(1, 14))
# The paragraph "What the ledger measures" of the module text in short: recorded on the route and
# printed as the summary's third line there.
HINDSIGHT = (
    "The ceiling, the losses R and A, the format loss F, V_aff - V_diag and the block product's "
    "terms are hindsight quantities of the scored test days, computed on their true sets; U is "
    "V_aff, computed on their true sets, less the run's certified value V_cert, which the run "
    "computed on its forecast sets; the offer and K come from the run's own scale and certified "
    "value. Every term is measured after the run scored its offers, and none is used to choose "
    "anything (no scale, configuration, kappa, dial or day)."
)


# --- the run -------------------------------------------------------------------------------------


def scoring_module(command) -> str | None:
    """The scoring module a recorded command runs with `python -m`, or None."""
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        return None
    for k in range(1, len(command)):
        if command[k - 1] == "-m" and command[k] in SCORING:
            return command[k]
    return None


def read_day_rows(path: Path) -> list[dict]:
    """The rows of a run's day table; a leading label line (`# ...`) is skipped."""
    return day_rows(path.read_text())


def day_rows(text: str) -> list[dict]:
    """The rows of a day table's text; a leading label line (`# ...`) is skipped."""
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    return list(csv.DictReader(lines))


def not_at_head(folder: Path, commit, head: str, route: sealed_route.Route) -> str:
    """The one line that refuses, on the sealed route, a scoring run recorded at `commit`, not at
    the checked-out commit `head` (module text, reading P-h)."""
    if route.fix_commit is None:
        where = "the tag's commit"
    else:
        where = f"the logged parsing fix of the tag's commit {route.tag_commit}"
    return (
        f"{folder.name} was recorded at commit {commit}, not at the checked-out commit "
        f"{head} ({where}); on the sealed route the ledger reads a scoring run only when it "
        "was recorded at the checked-out commit, so after a logged parsing fix every scoring run "
        "is made again at the fix commit"
    )


def read_run(
    folder: Path, head: str, route: sealed_route.Route | None = None
) -> tuple[str, dict, list[dict]]:
    """(scoring module, results document, day rows) of a verified scoring run at `head`; raises
    `CannotCheck`. Off the sealed route a run recorded at an earlier commit is read as
    `run_evidence.read_evidence` allows (its earlier-run rule); on the route (`route.held_out`) a
    run recorded at any commit other than `head` is refused first (`not_at_head`, reading P-h)."""
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
    except (OSError, ValueError) as problem:
        raise CannotCheck(f"the manifest of {folder.name} cannot be read: {problem}") from problem
    module = scoring_module(manifest.get("command") if isinstance(manifest, dict) else None)
    if module is None:
        raise CannotCheck(
            f"{folder.name} is not a recorded run of {' or '.join(SCORING)} (its command)"
        )
    results_name, days_name = SCORING[module]
    if route is not None and route.held_out and manifest.get("commit") != head:
        raise CannotCheck(not_at_head(folder, manifest.get("commit"), head, route))
    evidence = read_evidence(folder, results_name, head)
    if not evidence.verified:
        raise CannotCheck(
            f"{folder.name} ended with exit code {evidence.manifest.get('exit_code')}, so it is "
            "not verified"
        )
    read = {}
    for name in (results_name, days_name):  # each parsed from the bytes it was verified on
        try:
            read[name] = read_verified(folder / name)[1]
        except NotVerified as problem:
            raise CannotCheck(f"{name} of {folder.name} is not verified: {problem}") from problem
    try:
        document = json.loads(read[results_name])
        rows = day_rows(read[days_name].decode("utf-8"))
    except ValueError as problem:
        raise CannotCheck(f"the files of {folder.name} cannot be parsed: {problem}") from problem
    if not isinstance(document, dict):
        raise CannotCheck(f"{results_name} of {folder.name} is not a JSON object")
    for name in (results_name, days_name):
        try:
            still_verified(folder / name, read[name])
        except NotVerified as problem:
            raise CannotCheck(f"{name} of {folder.name}: {problem}") from problem
    return module, document, rows


def _at(document, *keys):
    """`document[keys[0]][keys[1]]...`, or None where a key is absent or a level is no mapping."""
    for key in keys:
        if not isinstance(document, dict) or key not in document:
            return None
        document = document[key]
    return document


def check_document(document: dict, matrix_file: Path, folder: Path) -> None:
    """The run's results describe a full or partial run on `matrix_file`; raises `CannotCheck`."""
    if document.get("trial") is not False:
        raise CannotCheck(f"{folder.name} is a trial run (or does not say it is not)")
    recorded = _at(document, "matrix", "sha256")
    given = hashlib.sha256(matrix_file.read_bytes()).hexdigest()
    if recorded != given:
        raise CannotCheck(
            f"{folder.name} ran on another matrix: SHA-256 {str(recorded)[:12]}, the given "
            f"{matrix_file.name} has {given[:12]}"
        )
    if not isinstance(document.get("configurations"), dict):
        raise CannotCheck(f"the results of {folder.name} hold no configurations")


def check_tables(document: dict, inputs: dict, units, folder: Path) -> None:
    """Each table the ledger reads is the one the run read; raises `CannotCheck`."""
    recorded = _at(document, "inputs", "table_sha256") or {}
    for unit in units:
        for kind in ("cars", "days"):
            name = f"{unit}-{kind}.csv"
            if recorded.get(name) != inputs["table_sha256"][name]:
                raise CannotCheck(f"{name} is not the table {folder.name} read")
    if _at(document, "inputs", "setting") != inputs["setting"]:
        raise CannotCheck(f"{folder.name} read another truth setting than the tables hold")


# --- the sealed route (module text) --------------------------------------------------------------


def check_run_files(root: Path, tables: Path, folder: Path) -> None:
    """On the route, before the run is read: its folder lies outside the held-out places of `root`
    and its manifest and results files pass `gate_run.check_files`. Raises
    `sealed_route.RouteRefused` or `CouldNotRun`."""
    try:
        sealed_route.check_folders(root, tables, (folder,))
    except sealed_route.FoldersRefused as problem:
        raise sealed_route.RouteRefused(
            f"a folder is refused: {problem}; nothing read"
        ) from problem
    names = [MANIFEST_NAME, *(name for pair in SCORING.values() for name in pair)]
    gate_run.check_files([folder / name for name in names])


def check_route_run(document: dict, route: sealed_route.Route, folder: Path) -> dict:
    """The run's record of the route, when it names the tag's commit and the fix commit the gate
    passes at now (module text); raises `CannotCheck`."""
    found = document.get(ROUTE_KEY)
    if not isinstance(found, dict) or found.get("kind") != sealed_route.HELD_OUT:
        raise CannotCheck(
            f"{folder.name} holds no record of the sealed route; on held-out units the ledger "
            "reads only a run made through the route"
        )
    recorded = (found.get("tag_commit"), found.get(sealed_route.FIX_KEY))
    if recorded != (route.tag_commit, route.fix_commit):
        raise CannotCheck(
            f"{folder.name} went through the sealed route at {sealed_route.passed_at(*recorded)} "
            f"by its record, not at {sealed_route.passed_at(route.tag_commit, route.fix_commit)}, "
            "where the gate passes now; on the sealed route the ledger reads a run only when its "
            "record names the tag's commit and the fix commit the gate passes at"
        )
    return found


def check_route_tables(document, found, inputs, held_inputs, units, folder: Path) -> None:
    """The development build and the held-out build read here, and each scored unit's tables, are
    those the run read (module text); raises `CannotCheck`."""
    held = found.get("held_out_inputs") or {}
    for what, theirs, ours in (
        ("development", document.get("inputs") or {}, inputs),
        ("held-out", held, held_inputs),
    ):
        if theirs.get("truth_run") != ours["truth_run"]:
            raise CannotCheck(
                f"{folder.name} read the {what} build {theirs.get('truth_run')}, not the "
                f"{what} build {ours['truth_run']} given here"
            )
    check_tables({"inputs": held}, held_inputs, units, folder)


def route_record(route, found: dict, folder: Path, inputs: dict, held_inputs: dict) -> dict:
    """What `ledger.json` records under `sealed_route` on the route (module text)."""
    return {
        "kind": sealed_route.HELD_OUT,
        "label": sealed_route.LABEL,
        "tag_commit": route.tag_commit,
        **sealed_route.fix_record(route),
        "run": folder.name,
        "development_inputs": inputs,
        "held_out_inputs": held_inputs,
        "run_fit_units": found.get("fit_units"),
        "run_kappa_selection": found.get("kappa_selection"),
        "checks": sealed_route.CHECKS,
        "hindsight": HINDSIGHT,
    }


def _number(text: str, where: str) -> float:
    try:
        value = float(text)
    except ValueError as problem:
        raise CannotCheck(f"{where} is not a number: {text!r}") from problem
    if not math.isfinite(value):
        raise CannotCheck(f"{where} is not finite: {text!r}")
    return value


def _optional(text: str, where: str) -> float | None:
    return None if text == "" else _number(text, where)


def group_rows(rows: list[dict]) -> dict:
    """The day rows by (configuration, unit), each in the order of the file."""
    grouped: dict = {}
    for row in rows:
        grouped.setdefault((row["configuration"], row["unit"]), []).append(row)
    return grouped


def match_days(document: dict, grouped: dict, plans: dict, folder: Path) -> None:
    """Every configuration's day rows of every unit are the plan's days, in order, with its roles;
    raises `CannotCheck`."""
    for name, entry in document["configurations"].items():
        for unit, plan in plans.items():
            found = entry["units"].get(unit, {})
            rows = grouped.get((name, unit), [])
            if "failed" in found and not rows:
                continue
            seen = [(f"{r['date']} {r['site']}", r["role"]) for r in rows]
            if seen != list(zip(plan.keys, plan.roles, strict=True)):
                raise CannotCheck(
                    f"the day rows of {unit} under {name} in {folder.name} are not the days and "
                    "roles of the tables"
                )


# --- the sample ----------------------------------------------------------------------------------


def draw_sample(plans: dict, seed: int, tree_days: int) -> dict:
    """Per split unit, the drawn test days as texts "site date" (module text)."""
    out = {}
    for unit, plan in plans.items():
        if plan.status != "calibratable":
            continue
        texts = []
        for truth, role in zip(plan.days.days, plan.roles, strict=True):
            if role == "test":
                texts.append(day_text(truth.record.site, truth.record.date))
        out[unit] = draw(unit, seed, texts, tree_days)
    return out


def sample_tasks(plans: dict, sample: dict) -> list[DayTask]:
    tasks = []
    for unit, texts in sample.items():
        chosen = set(texts)
        for truth in plans[unit].days.days:
            site, date = truth.record.site, truth.record.date
            if day_text(site, date) in chosen:
                tasks.append(DayTask(unit, site, date, truth.day, True))
    return sorted(tasks, key=lambda t: (t.unit, day_text(t.site, t.date)))


def solve_sample(tasks: list[DayTask], workers: int) -> dict:
    """`ledger_day.solve_day` of every task, by (unit, site, date)."""
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            solved = list(pool.map(solve_day, tasks))
    else:
        solved = [solve_day(task) for task in tasks]
    return {s.key: s for s in solved}


# --- the ledger ----------------------------------------------------------------------------------


class Checks:
    """The checks of the module text, counted as the ledger is built."""

    def __init__(self):
        self.identity_days = 0
        self.identity_breaks = []
        self.identity_means = 0
        self.largest_residual = 0.0
        self.offer_breaks = []
        self.ceiling_breaks = []
        self.cross_breaks = []
        self.order_breaks = []
        self.day_errors = []
        self.ratio_breaks = []
        self.offer_order_breaks = []

    def identity(self, residual, ceiling, where: str, mean: bool = False) -> None:
        if residual is None:
            return
        if mean:
            self.identity_means += 1
        else:
            self.identity_days += 1
        self.largest_residual = max(self.largest_residual, abs(residual))
        if not ledger_terms.within(residual, ceiling, ledger_terms.IDENTITY_TOLERANCE):
            self.identity_breaks.append(f"{where}: residual {residual!r} kWh")

    def order(self, value, ceiling, where: str) -> None:
        if ledger_terms.below_order(value, ceiling):
            self.order_breaks.append(f"{where}: {value!r} kWh")

    @property
    def failed(self) -> bool:
        return bool(
            self.identity_breaks
            or self.offer_breaks
            or self.ceiling_breaks
            or self.cross_breaks
            or self.order_breaks
            or self.day_errors
            or self.ratio_breaks
            or self.offer_order_breaks
        )

    def document(self) -> dict:
        return {
            "failed": self.failed,
            "identity": {
                "tolerance": ledger_terms.IDENTITY_TOLERANCE,
                "days_checked": self.identity_days,
                "means_checked": self.identity_means,
                "largest_residual_kwh": self.largest_residual,
                "breaks": self.identity_breaks,
            },
            "run_offer_reproduced": {"breaks": self.offer_breaks},
            "run_ceiling_sums_reproduced": {"breaks": self.ceiling_breaks},
            "sampled_days_are_the_runs": {
                "tolerance": CROSS_TOLERANCE,
                "breaks": self.cross_breaks,
            },
            "order": {"tolerance": ledger_terms.ORDER_TOLERANCE, "breaks": self.order_breaks},
            "sample_day_errors": self.day_errors,
            "ceiling_ratio_one_way_reproduced": {
                "tolerance": CROSS_TOLERANCE,
                "breaks": self.ratio_breaks,
            },
            "offer_order": {
                "tolerance": ceiling_ratio.ORDER_TOLERANCE,
                "breaks": self.offer_order_breaks,
            },
        }


def _test_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["role"] == "test"]


def _ceilings(row: dict, where: str) -> tuple[float, float | None]:
    return (
        _number(row["ceiling_hindsight_kwh"], f"{where} ceiling_hindsight_kwh"),
        _optional(row["ceiling_best_causal_kwh"], f"{where} ceiling_best_causal_kwh"),
    )


def _tree(solved: dict, unit: str, row: dict) -> float | None:
    found = solved.get((unit, row["site"], row["date"]))
    return None if found is None else found.values["hourly_tree"]


def _counts(items) -> dict:
    out: dict = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items()))


def hourly_true_sets(unit: str, rows: list[dict], sample: tuple, solved: dict, checks) -> dict:
    """The configuration-independent hourly terms of one unit (module text)."""
    days = []
    for row in rows:
        ceiling, best = _ceilings(row, f"{unit} {row['date']} {row['site']}")
        days.append({"ceiling": ceiling, "best_causal": best, "r_plus_a": None, "row": row})
        if best is not None:
            days[-1]["r_plus_a"] = ceiling - best
            checks.order(ceiling - best, ceiling, f"{unit} {row['date']} {row['site']} R + A")
    complete = [d for d in days if d["best_causal"] is not None]
    test_days = ledger_terms.means(complete, ("best_causal", "r_plus_a"))
    test_days["left_out_best_causal_missing"] = len(days) - len(complete)
    chosen = set(sample)
    sampled = [d for d in days if day_text(d["row"]["site"], d["row"]["date"]) in chosen]
    tree_outcomes = []
    split, diagonal, formats = [], [], []
    for d in sampled:
        found = solved[(unit, d["row"]["site"], d["row"]["date"])]
        where = f"{unit} {d['row']['date']} {d['row']['site']}"
        tree_outcomes.append(found.outcomes["hourly_tree"])
        tree, best, ceiling = found.values["hourly_tree"], d["best_causal"], d["ceiling"]
        if tree is not None and best is not None:
            split.append({"ceiling": ceiling, "r": ceiling - tree, "a": tree - best, "tree": tree})
            checks.order(ceiling - tree, ceiling, f"{where} R")
            checks.order(tree - best, ceiling, f"{where} A")
        diag = found.values["hourly_diag"]
        if diag is not None and best is not None:
            diagonal.append({"ceiling": ceiling, "affine_minus_diagonal": best - diag})
            checks.order(best - diag, ceiling, f"{where} V_aff - V_diag")
        box = found.values[LP24_BOX]
        if box is not None:
            formats.append({"ceiling": ceiling, "box": box, "format_loss": ceiling - box})
            checks.order(ceiling - box, ceiling, f"{where} F")
    tree_block = ledger_terms.means(split, ("tree", "r", "a"))
    nonzero = [
        d["a"]
        for d in split
        if not ledger_terms.within(d["a"], d["ceiling"], ledger_terms.ORDER_TOLERANCE)
    ]
    tree_block |= {
        "drawn": len(sampled),
        "tree_outcomes": _counts(tree_outcomes),
        "a_not_zero_days": len(nonzero),
        "largest_abs_a_kwh": max((abs(d["a"]) for d in split), default=None),
    }
    format_block = ledger_terms.means(formats, ("box", "format_loss"))
    format_block |= {
        "sets": "true",
        "base_set": "day-average",
        "programme": "LP (24)",
        "box_outcomes": _counts(
            solved[(unit, d["row"]["site"], d["row"]["date"])].outcomes[LP24_BOX] for d in sampled
        ),
    }
    return {
        "test_days": test_days,
        "tree_sample": tree_block,
        "affine_minus_diagonal": ledger_terms.means(diagonal, ("affine_minus_diagonal",)),
        "format_loss": format_block,
    }


def block_product(unit: str, rows: list[dict], sample: tuple, solved: dict, checks) -> dict:
    """The block product's terms of one unit on the tree sample (module text)."""
    chosen = set(sample)
    days = []
    for row in rows:
        if day_text(row["site"], row["date"]) not in chosen:
            continue
        found = solved[(unit, row["site"], row["date"])].values
        terms = ledger_terms.block_terms(
            found["hourly_hind"],
            found["block_hind"],
            found["block_caus"],
            found["block_aff"],
            found["block_diag"],
        )
        where = f"{unit} {row['date']} {row['site']} block"
        for name in BLOCK_ORDER:
            reference = (
                terms["hourly_ceiling"] if name == "price_of_the_block" else terms["ceiling"]
            )
            if reference is not None:
                checks.order(terms[name], reference, f"{where} {name}")
        days.append(terms)

    def having(*names):
        return [d for d in days if all(d[n] is not None for n in names)]

    price = [  # the price is a share of the hourly ceiling (reading L10)
        {**d, "ceiling": d["hourly_ceiling"], "block_ceiling": d["ceiling"]}
        for d in having("hourly_ceiling", "ceiling", "price_of_the_block")
    ]
    affine = having("ceiling", "affine_minus_diagonal")
    return {
        "drawn": len(days),
        "tree_sample": ledger_terms.means(having("ceiling", "r", "a"), ("r", "a")),
        "affine_minus_diagonal": ledger_terms.means(affine, ("affine_minus_diagonal",)),
        "price_of_the_block": ledger_terms.means(price, ("block_ceiling", "price_of_the_block")),
        "u": ledger_terms.NOT_COMPUTED_BLOCK,
        "k": ledger_terms.NOT_COMPUTED_BLOCK,
    }


def _scored(rungs: dict, rung: str, tier: str) -> dict:
    return rungs[rung] if tier == "A" else rungs[rung]["tier_b"]


def _tiers(rungs: dict) -> tuple[str, ...]:
    return TIERS if all("tier_b" in rungs[r] for r in RUNGS) else TIERS[:1]


def _offer_column(rung: str, tier: str) -> str:
    return f"{rung}_offer_kwh" if tier == "A" else f"{rung}_tier_b_offer_kwh"


def rung_ledger(name, unit, tier, rung, entry, rows, solved, checks, day_out) -> dict:
    """The ledger of one configuration, unit, tier and rung (module text)."""
    scored = _scored(entry["rungs"], rung, tier)
    beta = scored["beta_hat"]
    if isinstance(beta, bool) or not isinstance(beta, int | float) or not math.isfinite(beta):
        raise CannotCheck(f"{name} {unit} {rung} tier {tier}: the scale {beta!r} is not a number")
    beta = float(beta)
    days = []
    for row in rows:
        where = f"{name} {unit} {row['date']} {row['site']} tier {tier} {rung}"
        ceiling, best = _ceilings(row, where)
        terms = ledger_terms.day_terms(
            ceiling,
            best,
            _number(row["certified_value_kwh"], f"{where} certified_value_kwh"),
            beta,
            _number(row[_offer_column(rung, tier)], f"{where} offer"),
            _tree(solved, unit, row),
        )
        joint = ledger_terms.identity_residual(terms, ledger_terms.JOINT)
        split = ledger_terms.identity_residual(terms, ledger_terms.SPLIT)
        checks.identity(joint, ceiling, where)
        checks.identity(split, ceiling, f"{where} (R and A apart)")
        days.append(terms)
        in_sample = (unit, row["site"], row["date"]) in solved
        day_out.append(_day_row(name, unit, tier, rung, row, in_sample, terms, joint, split))
    complete = [d for d in days if d["u"] is not None]
    sampled = [d for d in complete if d["r"] is not None]
    test_days = ledger_terms.means(complete, ledger_terms.JOINT)
    tree_sample = ledger_terms.means(sampled, ledger_terms.SPLIT)
    head = f"{name} {unit} tier {tier} {rung}"
    for block, names, label in (
        (test_days, ledger_terms.JOINT, "test days"),
        (tree_sample, ledger_terms.SPLIT, "tree sample"),
    ):
        block["identity_residual_kwh"] = ledger_terms.identity_of_means(block, names)
        checks.identity(
            block["identity_residual_kwh"],
            block["mean_ceiling_kwh"] or 0.0,
            f"{head} mean over the {label}",
            mean=True,
        )
    test_days["left_out_best_causal_missing"] = len(days) - len(complete)
    run_offer = scored["test"]["mean_offer_kwh_per_day"]
    all_offer = ledger_terms.means(days, ("offer",))["mean_kwh"]["offer"] if days else None
    if not _same(all_offer, run_offer):
        checks.offer_breaks.append(f"{head}: day rows {all_offer!r}, the run {run_offer!r}")
    return {
        "scale": beta,
        "run_mean_offer_kwh_per_test_day": run_offer,
        "mean_offer_kwh_per_test_day": all_offer,
        "test_days": test_days,
        "tree_sample": tree_sample,
    }


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return ledger_terms.within(a - b, b, ledger_terms.IDENTITY_TOLERANCE)


def check_ceiling_sums(name, unit, entry, rows, checks) -> None:
    """Check 2's ceiling sums for one configuration and unit."""
    recorded = entry.get("ceiling_sums_test_days_kwh", {})
    columns = {CEILINGS[0]: "ceiling_hindsight_kwh", CEILINGS[1]: "ceiling_best_causal_kwh"}
    for ceiling, column in columns.items():
        values = [r[column] for r in rows]
        found = None if "" in values else float(sum(_number(v, column) for v in values))
        if not _same(found, recorded.get(ceiling)):
            checks.ceiling_breaks.append(
                f"{name} {unit} {ceiling}: day rows {found!r}, the run {recorded.get(ceiling)!r}"
            )


def cross_check(unit, rows: list[dict], sample: tuple, solved: dict, checks) -> None:
    """Check 3 for the sampled days of one unit."""
    chosen = set(sample)
    for row in rows:
        if day_text(row["site"], row["date"]) not in chosen:
            continue
        found = solved[(unit, row["site"], row["date"])]
        where = f"{unit} {row['date']} {row['site']}"
        if found.error is not None:
            checks.day_errors.append(f"{where}: {found.error}")
            continue
        ceiling, best = _ceilings(row, where)
        pairs = (("hourly_hind", ceiling), ("hourly_causal", best))
        for key, recorded in pairs:
            again = found.values[key]
            if recorded is None or again is None:
                continue
            if not ledger_terms.within(again - recorded, recorded, CROSS_TOLERANCE):
                checks.cross_breaks.append(
                    f"{where} {key}: {again!r} against the run's {recorded!r}"
                )


# --- outcome O8's ceiling ratio (module text) ----------------------------------------------------


def ratio_day_ceilings(plans: dict, budget: float) -> dict:
    """(unit, site, date) -> `ceiling_ratio.day_ceilings` at `budget` of every test day of a split
    unit, or the text of the problem when a set could not be built."""
    out: dict = {}
    for unit, plan in plans.items():
        if plan.status != "calibratable":
            continue
        for truth, role in zip(plan.days.days, plan.roles, strict=True):
            if role != "test":
                continue
            key = (unit, truth.record.site, truth.record.date)
            try:
                out[key] = ceiling_ratio.day_ceilings(truth.day, budget)
            except ValueError as problem:
                out[key] = str(problem)
    return out


def _two_way_offers(name: str, unit: str, tier: str, rows: list[dict]) -> list[float]:
    column = _offer_column(RUNGS[2], tier)
    return [
        _number(r[column], f"{name} {unit} {r['date']} {r['site']} tier {tier} {column}")
        for r in rows
    ]


ORDER_STEPS = (
    ("fixed <= one-way", "fixed_one_way_broken_days"),
    ("one-way <= two-way", "one_way_two_way_broken_days"),
)
ORDER_DAYS_NAMED = 3  # days named in one break of check 7


def check_offer_order(unit, name, tier, found: dict, checks) -> None:
    """Check 7 of one unit, configuration and tier: each broken step is a break (module text)."""
    for step, key in ORDER_STEPS:
        broken = found[key]
        if broken:
            more = len(broken) - ORDER_DAYS_NAMED
            named = "; ".join(broken[:ORDER_DAYS_NAMED]) + (
                f"; and {more} more" if more > 0 else ""
            )
            checks.offer_order_breaks.append(
                f"{unit}, {name}, tier {tier}, {step}: {len(broken)} of {found['test_days']} test "
                f"days: {named}"
            )


def unit_offer_ratio(unit, names, document, grouped, checks) -> dict:
    """Outcome O8's offer ratio X and outcome O5's offer order of one unit, per configuration and
    tier, and check 7 of its test days (module text; `ceiling_ratio.offer_ratio`)."""
    out: dict = {}
    for name in names:
        found = document["configurations"][name]["units"].get(unit, {})
        if "rungs" not in found:
            continue
        config_rows = _test_rows(grouped.get((name, unit), []))
        out[name] = {}
        for tier in _tiers(found["rungs"]):
            days = []
            for row in config_rows:
                where = f"{name} {unit} {row['date']} {row['site']} tier {tier}"
                offers = [
                    _number(row[_offer_column(rung, tier)], f"{where} {rung} offer")
                    for rung in RUNGS
                ]
                days.append((day_text(row["site"], row["date"]), *offers))
            out[name][tier] = dataclasses.asdict(ceiling_ratio.offer_ratio(days))
            check_offer_order(unit, name, tier, out[name][tier], checks)
    return out


def unit_ceiling_ratio(unit, rows, names, document, grouped, ceilings, budget, checks, out) -> dict:
    """Outcome O8's ceiling ratio of one unit (module text; `ceiling_ratio`), check 6 of its test
    days, and their rows of `RATIO_NAME` appended to `out`."""
    for name in names:
        for row in _test_rows(grouped.get((name, unit), [])):
            found = ceilings[(unit, row["site"], row["date"])]
            if isinstance(found, str):
                continue
            where = f"{name} {unit} {row['date']} {row['site']}"
            recorded = _number(row["ceiling_hindsight_kwh"], f"{where} ceiling_hindsight_kwh")
            if not ledger_terms.within(found[0] - recorded, recorded, CROSS_TOLERANCE):
                checks.ratio_breaks.append(
                    f"{where}: the one-way ceiling {found[0]!r} recomputed, the run's {recorded!r}"
                )
    days, unbuilt = [], []
    for row in rows:
        found = ceilings[(unit, row["site"], row["date"])]
        text = day_text(row["site"], row["date"])
        if isinstance(found, str):
            unbuilt.append(text)
            checks.ratio_breaks.append(f"{unit} {text}: the sets could not be built: {found}")
            continue
        days.append((text, *found))
        out.append(
            {
                "unit": unit,
                "site": row["site"],
                "date": row["date"],
                "one_way_ceiling_kwh": repr(found[0]),
                "run_ceiling_hindsight_kwh": row["ceiling_hindsight_kwh"],
                "two_way_ceiling_kwh": repr(found[1]),
                "budget_kwh": repr(budget),
            }
        )
    if unbuilt:
        return {"reason": "a test day's sets could not be built", "days_not_built": unbuilt}
    shares = {}
    for name in names:
        found = document["configurations"][name]["units"].get(unit, {})
        if "rungs" not in found:
            continue
        config_rows = _test_rows(grouped.get((name, unit), []))
        two_way = [ceilings[(unit, r["site"], r["date"])][1] for r in config_rows]
        shares[name] = {
            tier: ceiling_ratio.offer_share(_two_way_offers(name, unit, tier, config_rows), two_way)
            for tier in _tiers(found["rungs"])
        }
    return {
        **dataclasses.asdict(ceiling_ratio.ratio(days)),
        "two_way_offer_share_of_two_way_ceiling": shares,
    }


def ratio_record(budget: float) -> dict:
    """What `ledger.json` records under `ceiling_ratio` for the whole run (module text)."""
    return {
        "outcome": "O8",
        "budget_kwh": budget,
        "label": ceiling_ratio.budget_label(budget),
        "plan_budget_kwh": ceiling_ratio.PLAN_BUDGET_KWH,
        "note": ceiling_ratio.budget_note(budget),
        "definition": ceiling_ratio.DEFINITION,
        "readings": list(ceiling_ratio.READINGS),
        "discharge_d": ceiling_ratio.DISCHARGE,
        "offer_ratio_x": ceiling_ratio.OFFER_RATIO,
    }


def _text(value) -> str:
    return "" if value is None else repr(value)


DAY_COLUMNS = (
    "configuration",
    "unit",
    "tier",
    "rung",
    "site",
    "date",
    "in_tree_sample",
    "ceiling_kwh",
    "best_causal_kwh",
    "tree_kwh",
    "certified_kwh",
    "scale",
    "offer_kwh",
    "r_plus_a_kwh",
    "r_kwh",
    "a_kwh",
    "u_kwh",
    "k_kwh",
    "identity_residual_kwh",
    "identity_residual_r_a_apart_kwh",
)


def _day_row(name, unit, tier, rung, row, in_sample, terms, joint, split) -> dict:
    return {
        "configuration": name,
        "unit": unit,
        "tier": tier,
        "rung": rung,
        "site": row["site"],
        "date": row["date"],
        "in_tree_sample": str(in_sample),
        "ceiling_kwh": _text(terms["ceiling"]),
        "best_causal_kwh": _text(terms["best_causal"]),
        "tree_kwh": _text(terms["tree"]),
        "certified_kwh": _text(terms["certified"]),
        "scale": _text(terms["beta"]),
        "offer_kwh": _text(terms["offer"]),
        "r_plus_a_kwh": _text(terms["r_plus_a"]),
        "r_kwh": _text(terms["r"]),
        "a_kwh": _text(terms["a"]),
        "u_kwh": _text(terms["u"]),
        "k_kwh": _text(terms["k"]),
        "identity_residual_kwh": _text(joint),
        "identity_residual_r_a_apart_kwh": _text(split),
    }


SAMPLE_COLUMNS = (
    "unit",
    "site",
    "date",
    "cars",
    "n_service",
    "error",
    *(f"{key}{suffix}" for key in KEYS for suffix in ("", "_outcome")),
    "format_loss_kwh",
    "run_ceiling_hindsight_kwh",
    "run_ceiling_best_causal_kwh",
)


def _sample_row(found, row: dict) -> dict:
    out = {
        "unit": found.unit,
        "site": found.site,
        "date": found.date,
        "cars": found.cars,
        "n_service": found.n_service,
        "error": found.error or "",
    }
    for key in KEYS:
        out[key] = _text(found.values[key])
        out[f"{key}_outcome"] = found.outcomes[key]
    hind, box = found.values["hourly_hind"], found.values[LP24_BOX]
    out["format_loss_kwh"] = "" if hind is None or box is None else repr(hind - box)
    out["run_ceiling_hindsight_kwh"] = row["ceiling_hindsight_kwh"]
    out["run_ceiling_best_causal_kwh"] = row["ceiling_best_causal_kwh"]
    return out


RATIO_COLUMNS = (
    "unit",
    "site",
    "date",
    "one_way_ceiling_kwh",
    "run_ceiling_hindsight_kwh",
    "two_way_ceiling_kwh",
    "budget_kwh",
)


def assemble(
    module, folder, document, matrix, plans, grouped, sample, solved, seed, tree_days, ceilings
):
    """(ledger document, day rows, sample rows, ceiling ratio rows); `ceilings` from
    `ratio_day_ceilings` at the matrix's budget."""
    budget = matrix.two_way_budget_kwh
    checks = Checks()
    units, day_out, sample_out, ratio_out = {}, [], [], []
    names = list(document["configurations"])
    for unit, plan in plans.items():
        if plan.status != "calibratable":
            units[unit] = {"status": plan.status, "test_days": 0}
            continue
        # The ceilings are the same under every configuration (`gate_run.run_gate`): the rows of
        # the first configuration with rows for the unit are read for the true-set terms.
        source = next((grouped[(n, unit)] for n in names if (n, unit) in grouped), [])
        rows = _test_rows(source)
        errors_before = len(checks.day_errors)
        cross_check(unit, rows, sample[unit], solved, checks)
        chosen = set(sample[unit])
        for row in rows:
            if day_text(row["site"], row["date"]) in chosen:
                sample_out.append(_sample_row(solved[(unit, row["site"], row["date"])], row))
        if len(checks.day_errors) > errors_before:
            units[unit] = {
                "status": plan.status,
                "test_days": len(rows),
                "reason": "a sampled day's sets could not be built",
            }
            continue
        entry = {
            "status": plan.status,
            "test_days": len(rows),
            "tree_sample_days": list(sample[unit]),
            "hourly_true_sets": hourly_true_sets(unit, rows, sample[unit], solved, checks),
            "block_product": block_product(unit, rows, sample[unit], solved, checks),
            "configurations": {},
        }
        for name in names:
            found = document["configurations"][name]["units"].get(unit, {})
            if "rungs" not in found:
                why = found.get("failed") or found.get("reason") or "no result"
                entry["configurations"][name] = {"reason": why}
                continue
            config_rows = _test_rows(grouped.get((name, unit), []))
            check_ceiling_sums(name, unit, found, config_rows, checks)
            tiers = {}
            for tier in _tiers(found["rungs"]):
                tiers[tier] = {
                    rung: rung_ledger(
                        name, unit, tier, rung, found, config_rows, solved, checks, day_out
                    )
                    for rung in RUNGS
                }
            entry["configurations"][name] = {"tiers": tiers}
        entry["ceiling_ratio"] = unit_ceiling_ratio(
            unit, rows, names, document, grouped, ceilings, budget, checks, ratio_out
        )
        entry["offer_ratio"] = unit_offer_ratio(unit, names, document, grouped, checks)
        units[unit] = entry
    ledger = {
        "ceiling_ratio": ratio_record(budget),
        "definitions": DEFINITIONS,
        "readings": list(READINGS),
        "product": "hourly (the block product: R, A, V_aff - V_diag and its price only)",
        "run": run_record(module, folder, document),
        "tree_sample": {
            "tree_days_per_unit": tree_days,
            "seed": seed,
            "rule": SAMPLE_RULE,
            "days": {unit: list(texts) for unit, texts in sample.items()},
        },
        "units": units,
        "checks": checks.document(),
    }
    return ledger, day_out, sample_out, ratio_out


SAMPLE_RULE = (
    "per split unit, the test days with the smallest ranks of the seeded hash of (unit, seed, "
    "'site date') (block_sample.draw), all of them when the unit has fewer"
)
DEFINITIONS = {
    "ceiling": (
        "V_hind, the hourly ceiling of the day's true cars with the decision hour free "
        "(exact.hourly_ceiling, the run's ceiling_hindsight_kwh)"
    ),
    "offer": "the run's offer of the day: the rung's scale for the tier times V_cert",
    "r": "R = V_hind - V_caus, the price of real-time decisions (V_caus: the corner tree)",
    "a": "A = V_caus - V_aff, the price of affine rules (V_aff: the run's best-causal ceiling)",
    "r_plus_a": "R + A = V_hind - V_aff, measured on every test day",
    "u": "U = V_aff - V_cert, the price of unknown cars (V_cert: the run's certified value)",
    "k": "K = (1 - scale) V_cert, the price of calibration",
    "identity": "offer + R + A + U + K = ceiling, per day and in every mean over one set of days",
    "format_loss": (
        "F = V_hind minus the largest hourly box in LP (24)'s set on true sets with the "
        "day-average base set; beside the sum, not in it"
    ),
    "affine_minus_diagonal": "V_aff - V_diag, the diagonal value on true sets",
    "price_of_the_block": "hourly V_hind minus block V_hind",
    "share_of_ceiling": "a mean term over the mean ceiling of the same days",
    "units": "kWh per counting test day",
}


def run_record(module: str, folder: Path, document: dict) -> dict:
    out = {
        "folder": folder.name,
        "command": module,
        "label": document.get("label"),
        "partial": document.get("partial"),
        "setting": _at(document, "inputs", "setting"),
        "truth_run": _at(document, "inputs", "truth_run"),
        "matrix_sha256": _at(document, "matrix", "sha256"),
    }
    for key in (rotation_whatif.KEY, LOOK_AHEAD_KEY):
        if key in document:
            out[key] = document[key]
    return out


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


def write_outputs(folder: Path, ledger, day_rows, sample_rows, ratio_rows, text, timing) -> None:
    top = ledger["run"]["label"] if ledger["run"]["partial"] else None
    (folder / RESULTS_NAME).write_text(_json(ledger))
    (folder / DAYS_NAME).write_text(_csv(day_rows, DAY_COLUMNS, top))
    (folder / SAMPLE_NAME).write_text(_csv(sample_rows, SAMPLE_COLUMNS, top))
    (folder / RATIO_NAME).write_text(_csv(ratio_rows, RATIO_COLUMNS, top))
    (folder / SUMMARY_NAME).write_text(text)
    (folder / TIMING_NAME).write_text(_json(timing))


def main(argv: list[str] | None = None, root: Path = TOP) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--tree-days", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--held-out-tables", type=Path, default=None)
    parser.add_argument("--held-out-truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        gate_run.argument_paths(args, (*gate_run.PATH_ARGUMENTS, "run"))
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if args.tree_days < 1:
            raise MatrixError("--tree-days must be at least 1")
        gate_run.check_files([args.matrix])
        matrix = load(args.matrix)
        route = sealed_route.matrix_route(
            root,
            matrix.gate_units,
            args.tables,
            args.truth_run,
            args.held_out_tables,
            args.held_out_truth_run,
        )
        if route.held_out:
            check_run_files(root, args.tables, args.run)
        module, document, rows = read_run(args.run, _head_commit(), route)
        check_document(document, args.matrix, args.run)
        if route.held_out:
            found = check_route_run(document, route, args.run)
            recorded = sealed_route.check_split_rule(matrix, route)
            inputs, held_inputs, table_bytes = sealed_route.read_two_builds(
                args.truth_run, args.tables, (), route.truth_run, route.tables, matrix.gate_units
            )
            matrix = sealed_route.scored_matrix(matrix, held_inputs)  # abstaining units out
            check_route_tables(document, found, inputs, held_inputs, matrix.gate_units, args.run)
        else:
            inputs = gate_run.check_inputs(args.truth_run, args.tables, matrix.gate_units)
            check_tables(document, inputs, matrix.gate_units, args.run)
    except (MatrixError, CannotCheck, CouldNotRun, sealed_route.RouteRefused) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    readings = {"rho": matrix.rho, "no_fit": matrix.no_fit, "small_fleet": matrix.small_fleet}
    try:
        days = {
            name: (
                parse_unit_days(table_bytes, name, **readings)
                if route.held_out
                else load_unit_days(args.tables, name, **readings)
            )
            for name in matrix.gate_units
        }
    except CouldNotRun as problem:  # a table changed after its check (on the route)
        print(f"COULD NOT RUN: {problem}")
        return 2
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    if route.held_out:
        try:
            sealed_route.check_split_days(matrix, days, recorded, route.truth_run.name)
        except CouldNotRun as problem:
            print(f"COULD NOT RUN: {problem}")
            return 2
    plans = {name: gate_run.plan_unit(name, days[name], matrix) for name in matrix.gate_units}
    grouped = group_rows(rows)
    try:
        match_days(document, grouped, plans, args.run)
        sample = draw_sample(plans, args.seed, args.tree_days)
        solved = solve_sample(sample_tasks(plans, sample), args.workers)
        started = time.perf_counter()
        ceilings = ratio_day_ceilings(plans, matrix.two_way_budget_kwh)
        ratio_seconds = time.perf_counter() - started
        ledger, day_rows, sample_rows, ratio_rows = assemble(
            module,
            args.run,
            document,
            matrix,
            plans,
            grouped,
            sample,
            solved,
            args.seed,
            args.tree_days,
            ceilings,
        )
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except (KeyError, TypeError) as problem:  # a field the ledger reads is not in the run's files
        print(f"COULD NOT RUN: the run's files lack what the ledger reads: {problem!r}")
        return 2
    if route.held_out:
        ledger[ROUTE_KEY] = route_record(route, found, args.run, inputs, held_inputs)
    text = "\n".join(ledger_summary.summary_lines(ledger)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        timing = {
            "workers": args.workers,
            "seconds": {" ".join(key): found.seconds for key, found in sorted(solved.items())},
            "ceiling_ratio_seconds": ratio_seconds,
            "ceiling_ratio_test_days": len(ceilings),
        }
        write_outputs(Path(run_dir), ledger, day_rows, sample_rows, ratio_rows, text, timing)
    return 1 if ledger["checks"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
