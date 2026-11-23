"""The WP10 pass-mark item, read from the recorded comparison runs (plan section 13, WP10:
"Baselines, no-pause sensitivity. Pass: shared offers, paths, output files").

Run as `python tools/compare_shared_check.py [--top <folder>]`.

What the pass mark requires, as read from the plan (the readings P1 to P9 at the end name each
choice the words leave open, with its alternative; where two readings are honest the stricter
one is taken, except in P3, P4, P5 and P8, which follow the owner's decisions of 16 November
2026):

R1 Methods. Every baseline and comparison method of plan section 10 that a method of the
   registry implements is measured, beside the calibrated one-way re-solve it shares its offers
   with: the check requires the 11 built methods (the owner's decision of 16 November 2026).
   `METHODS` lists them in the section's order with the registry name of
   `fleetreserve.compare_methods` that implements each:
   - `re-solve`: the reference; section 10, "All get the same forecasts, requests, calibration
     and loss", is read as: the same as the calibrated re-solve (section 1, "Causal offer: the
     certified box at scale beta-hat, run by the rungs").
   - `diagonal-split`: "Diagonal split (includes homothets, Zhao et al. 2017)".
   - `fixed-plan`: "fixed plan without re-solving".
   - `asa` and `asa-es`: "ASA (ACN adaptive scheduling) as an hourly tracking MPC" (reading P2).
   - `llf`, `edf`, `smoothed-llf`: "LLF, EDF, smoothed LLF (Chen et al. 2022)".
   - `max-entropy-feedback`: "Maximum-entropy feedback on forecast departures (Li et al.)".
   - `also-x`, `cvar`: "ALSO-X, CVaR (Lunde et al. 2025)".
   Not methods of this item: "The earlier analysis (Section 12), which looks ahead" (reading P3)
   and "LP (24), LP (27), general split (O3)" (reading P4). A listed method with no registry name
   (None) or with a name not in the registry is "no method": the item could not run (none is
   listed; this guards a method listed before a method of the registry implements it). Not
   required: the methods section 10 lists under "Not run".
R2 Oracle departures (section 10, "plus oracle departures"): every method is measured under both
   variants of `compare_run` ("forecast" and "oracle-departure", its reading C4).
R3 Shared offers, paths and steered cars. In each comparison run, under every gate unit and
   variant, every method is measured on the same prepared days: the run records them once, as
   `measured_on` (its days, 8 audited-path rows a day as plan section 1's audited set has, and
   the digests of the boxes, the paths and the steered cars; `compare_run` module text). The
   split table shows it method by method: on every unit, variant, tier and split, every method of
   the run has a row, and the rows agree in the calibration days, the test days, the offering
   calibration days, the offering test days, every ceiling column and the certified value (the
   offer over the scale, `dial_splits.score_split`: the scale times the mean certified value of
   the test days), compared where at least two methods have a scale above 0, to a relative
   difference of `VALUE_TOLERANCE`. A row missing or a disagreement shows R3 false.
R4 Shared output files (reading P7): all the methods are in the same files of one run as the
   re-solve: the results' list `methods`, each method's summary for both tiers under both
   variants of every gate unit and its gate reading, and its rows of the split table. The files
   are those of `compare_run.output_names` (results, summary, forecast table, split table,
   timing), each in the run folder and listed in the manifest with the SHA-256 it has now
   (`run_verify.require_verified`); the results' `split_table` and `forecast_table` name those
   files with the manifest's SHA-256; the split table has the header of `compare_splits.columns`
   and exactly one row per unit that can be split, variant, tier, method and split, as many as
   `split_table.rows` says.
R5 Calibration and inputs. The runs use the plan's 200 seeded development splits (section 6:
   "200 seeded development splits ... are the splits on which the dial is fixed and the WP6 gate
   is read"; seeds 1 to 200), both tiers ("A" and "B", section 6, never pooled), the committed
   `gate/matrix.json`, the main truth setting (section 3) and a recorded truth build, verified
   and recorded at the checked-out commit or read from an earlier commit by the rule of
   `run_evidence` (`runs.noindex/<UTC start>-wp2-truth`, the one the run names in
   `inputs.truth_run`, with every truth table the run lists having the SHA-256 that build
   recorded, and both tables of every gate unit listed). The build's setting is read from its
   summary as `truth_verify.summary_setting` reads it, as the truth build writes it: a summary
   without a `setting` key is a main one (the build writes the key only for the held setting),
   the held setting is not the main one, and a setting that is neither could not run. Every gate
   unit of the matrix must be one that can be split: a unit that cannot has no comparison, so
   the item could not run.
R6 Models and dials (readings P5 and P6). The comparisons are made with the main forecasting
   model, `lgbm_tail` (the owner's decision of 16 November 2026), which is also the model the
   WP6 gate is read with (section 13, WP6), named in this file (`GATE_MODEL`, the one entry of
   `MODELS`). Each model is compared at the dial section
   6's rule gives it per site type, read from its recorded dial-choice run (`wp6-dial-choice-
   <model>`, `python -m fleetreserve.dial_choice --model <model> --matrix gate/matrix.json
   --splits 200`, read as a comparison run is ("Runs read"), of that model, on the committed
   matrix and the main truth setting with its recorded truth build as in R5, with 200 splits; its
   seeds and tiers are not read, `dial_choice` having no option for them): the tier A dial at the
   unit `DIAL_SOURCES` names for each gate unit (section 6: "workplaces: from NREL from May 2019;
   Norway and Korea: from the pooled Norwegian development sites"). One comparison run is read
   per model and distinct dial, for the gate units of that dial; its checks cover every gate
   unit of the run.
R7 No-pause sensitivity (reading P8). For each comparison of R6 the same comparison is run at
   `--no-pause-share 1/3` with the default seed (`no_pause.DEFAULT_SEED`; plan section 9: "a
   seeded 0 or 1/3 of cars charge as logged"), with the checks of R1 to R5 on its own files,
   and against the run without the draw: the same methods, the same forecast table (its
   SHA-256; `compare_run` reading N5), the same truth tables and, under every unit and variant,
   the same days, path rows and audited paths (`measured_on`); its boxes and steered cars are
   its own. The two-way rung under the draw is not required (reading P8; `compare_run` reading
   N4: no command writes it). A requirement listed in `NOT_BUILT` is "not built": the item could
   not run (none is listed; this guards a requirement listed before a command writes it).

Runs read. Each is the newest folder in `runs.noindex/` of its label (`run_evidence.newest_run`;
an older run is never used in its place), with its results verified, and recorded at the commit
checked out now or read from an earlier commit by the rule of `run_evidence` (module text there;
else stale). Every line this tool writes that names a run read from an earlier commit (a
comparison run, a dial-choice run or a truth build) carries `Evidence.note` after the run's name,
whatever the line then says (not written by this tool, so without it: Python's own text of an
unexpected error, after "the check stopped on", which can name a file in a run's folder).
Recorded by
    python -m fleetreserve.run_record --name <label> --long --max-seconds <n> --
        uv run --frozen --group learn python -m fleetreserve.compare_run --model <model>
        --dial <dial> --matrix gate/matrix.json --splits 200 --workers <n>
        [--no-pause-share 1/3]
with the label `compare_label(model, dial, share)`: `wp10-compare-<model>-eps<dial>` (the dial
as written by Python's `g` format without its point, so 0.02 is `eps002`), and `-no-pause` added
for the sensitivity. A run decides nothing (could not run) unless its recorded command runs
`python -m fleetreserve.compare_run`, its results name the model and the dial asked for (and, for
the sensitivity, the share 1/3 and the default seed; without it, no draw), and the checks of R5
hold.

Readings (the alternative in brackets):

  P1. "Shared offers": every method runs on the same certified box at its own calibrated scale,
      calibrated on the same days and scored on the same test days (`compare_run` readings C1 and
      C3), as outcome O1 compares offers "against ASA (A), LLF (B)". [Alternative: one scale, the
      re-solve's, for every method, so the offers are equal and only the losses differ.]
  P2. ASA: both forms in the registry, `asa` (the linear-programme form) and `asa-es` (with the
      equal-sharing term), are required while the owner has not said which stands in outcome O1.
      [Alternative: only the form the owner names.]
  P3. "The earlier analysis (Section 12), which looks ahead" is not a method of this item: it is
      the WP3 re-run of section 12 and outcome O1's "earlier forecasts (C)", the re-solve run
      with the earlier GBM and population forecasts, forecast models of section 7 (the owner's
      decision of 16 November 2026). [Alternative not taken: a method, section 12's offer (full
      affine shares, which look ahead, in LP (24) with a template base set), run beside the
      others; no method implements it.]
  P4. "LP (24), LP (27), general split (O3)" are not methods of this item: they are measured only
      by outcome O3 on true sessions ("100-200 sampled days"; section 13: "LP (24) on a
      sample"), which has no forecast or calibration (the owner's decision of 16 November 2026).
      [Alternative not taken: methods of the comparison, as "All get the same forecasts,
      requests, calibration and loss" says; no method implements them.]
  P5. Models: the main forecasting model `lgbm_tail` alone, which is also the gate's model (the
      owner's decision of 16 November 2026). [Alternative not taken: also `hazard_tuned_inputs`,
      the neural variant reported beside the main model by the same decision.]
  P6. Dials: read from each model's recorded dial-choice run (R6, read as "Runs read" says), by
      section 6's rule (tier A, per site type), never a number written here. Tier B is read at
      the same dial; section 6's "the driver loss is reported at every epsilon" is read as the
      re-solve's report (outcome O4), not a comparison at every dial. [Alternative: a fixed dial
      written in this file; for tier B, the comparisons at every dial.]
  P7. "Output files": every method in the same files of one run as the re-solve. [Alternative:
      methods spread over several runs whose `measured_on` records agree under every unit and
      variant.]
  P8. No-pause: the share 1/3 at the default seed, for every comparison of R6; the one-way
      re-solve and the baselines under the draw are enough, and the two-way rung under the draw
      is not required (the owner's decision of 16 November 2026). [Alternative not taken: also
      the two-way rung under the draw, since section 9 states the sensitivity under two-way
      charging; no command writes it.]
  P9. The ASA check "on 5 ACN days" of section 10 is a condition of the ASA form, not a part of
      this item, whose words are "shared offers, paths, output files". [Alternative: required
      here too.]

Output: a first line with the checked-out commit and the matrix; per model a line for its dials
(`dials:`, or `NO RUN:` / `COULD NOT RUN:` with the reason); per comparison and per sensitivity run
a line `shown:` with the run folder, the units it is read for and the counts behind R1 to R4
(and R7), `FAILED:` with the first disagreements, or `NO RUN:` / `COULD NOT RUN:` (on each of
these lines a run read from an earlier commit has `Evidence.note` after its name, as "Runs read"
says); a line `NO METHOD:` per method with no implementation and `NOT BUILT:` per requirement no
command writes; a last line with the counts.

Exit status: 0 every requirement shown; 1 a requirement shown false by a run that decides (this
wins over 2, as in `check.sh`); 2 could not run (no run, a stale run, a run that cannot decide,
no method, not built, no committed matrix or one that cannot be read). An unexpected error is
could not run too, never 1 (Python's own exit status for an uncaught error is 1, which
`passmark.sh` reads as failed): within one requirement's check its line is `COULD NOT RUN:` with
the error and the other requirements are still checked; anywhere else the last line is `COULD NOT
RUN:`. Requirements counted in the last line: per model its dials, each comparison and each
sensitivity run; each method with no implementation; each requirement not built.
"""

import argparse
import csv
import io
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from learn_logged_check import MATRIX, committed_matrix

from fleetreserve import (
    audit_paths,
    compare_methods,
    compare_run,
    compare_splits,
    dial_choice,
    dial_splits,
    no_pause,
    truth_build,
)
from fleetreserve.run_evidence import (
    CannotCheck,
    head_commit,
    newest_run,
    read_evidence,
    with_note,
)
from fleetreserve.run_record import RUNS_FOLDER
from fleetreserve.run_verify import NotVerified, require_verified
from fleetreserve.truth import MAIN
from fleetreserve.truth_verify import RUN_LABELS, CouldNotRun, summary_setting

COMPARE_RUN = "fleetreserve.compare_run"
DIAL_CHOICE = "fleetreserve.dial_choice"
NAMED = 3  # the disagreements named per line
PATH_ROWS_A_DAY = 3 + audit_paths.N_SEEDED  # the three fixed audited paths and the seeded ones
RECORD_FIELDS = ("days", "path_rows", "boxes_sha256", "paths_sha256", "steered_sha256")
SHARED_COLUMNS = (
    "calibration_days",
    "test_days",
    "offering_calibration_days",
    "offering_test_days",
)
TABLE_KINDS = ("cars", "days")
TRUTH_RUN = re.compile(r"\d{8}T\d{6}Z-" + re.escape(RUN_LABELS[MAIN]))
CALIBRATABLE = "calibratable"
PASSED, FAILED, COULD_NOT_RUN = "shown", "failed", "could not run"

GATE_MODEL = "lgbm_tail"  # plan section 13, WP6: the forecast model the gate is read with
# The models compared: the main forecasting model alone, which is the gate's model (the owner's
# decision of 16 November 2026; reading P5).
MODELS = (GATE_MODEL,)
SPLITS = 200
NO_PAUSE_SHARE = "1/3"
NO_PAUSE_SEED = no_pause.DEFAULT_SEED
VALUE_TOLERANCE = 1e-9
# The unit whose dial-choice reading gives each gate unit its dial (plan section 6: "workplaces:
# from NREL from May 2019; Norway and Korea: from the pooled Norwegian development sites").
DIAL_SOURCES = {
    "nrel-before-may-2019": "nrel-from-may-2019",
    "nrel-from-may-2019": "nrel-from-may-2019",
    "norway-development-pooled": "norway-development-pooled",
}


@dataclass(frozen=True)
class Method:
    """A method of plan section 10: its words there and its registry name in
    `fleetreserve.compare_methods` (None: no method implements it)."""

    plan_name: str
    registry: str | None


METHODS = (
    Method("the calibrated one-way re-solve, the reference", "re-solve"),
    Method("diagonal split (includes homothets, Zhao et al. 2017)", "diagonal-split"),
    Method("fixed plan without re-solving", "fixed-plan"),
    Method("ASA as an hourly tracking MPC, linear-programme form", "asa"),
    Method("ASA as an hourly tracking MPC, with the equal-sharing term", "asa-es"),
    Method("LLF", "llf"),
    Method("EDF", "edf"),
    Method("smoothed LLF (Chen et al. 2022)", "smoothed-llf"),
    Method("maximum-entropy feedback on forecast departures (Li et al.)", "max-entropy-feedback"),
    Method("ALSO-X (Lunde et al. 2025)", "also-x"),
    Method("CVaR (Lunde et al. 2025)", "cvar"),
)

# Requirements no command writes yet: each is "not built", so the item could not run. None is
# listed: the two-way rung under the draw is not required (reading P8).
NOT_BUILT = ()


@dataclass(frozen=True)
class Setup:
    """What the item requires: the methods, the models, the number of splits and the requirements
    no command writes (the module's constants, smaller in the tests)."""

    methods: tuple = METHODS
    models: tuple = MODELS
    splits: int = SPLITS
    not_built: tuple = NOT_BUILT


def dial_text(dial: float) -> str:
    """A dial as written in a run label: Python's `g` format without its point."""
    return f"{dial:g}".replace(".", "")


def compare_label(model: str, dial: float, share: str | None = None) -> str:
    """The run label of a comparison of `model` at `dial`, with the draw when `share` is given."""
    label = f"wp10-compare-{model}-eps{dial_text(dial)}"
    return label if share is None else f"{label}-no-pause"


def dial_label(model: str) -> str:
    """The run label of a model's recorded dial-choice run."""
    return f"wp6-dial-choice-{model}"


class NoRun(CannotCheck):
    """No run of the label: the item's line says NO RUN."""


@dataclass(frozen=True)
class Context:
    """What every requirement's check reads: the folder, the checked-out commit, the SHA-256 and
    gate units of the committed matrix, the requirement and the registry names of its methods
    that the registry has."""

    top: Path
    head: str
    committed: str
    units: tuple
    setup: Setup
    required: tuple


@dataclass(frozen=True)
class Read:
    """A comparison run that can decide: its folder, results, manifest, output file names, the
    output files' problems found while reading it, and the evidence's note (`Evidence.note`, empty
    for a run at the checked-out commit)."""

    run: Path
    document: dict
    manifest: dict
    names: dict
    problems: tuple
    note: str = ""


def _at(document, *keys):
    """`document[keys[0]][keys[1]]...`, or None where a key is absent or a level is no mapping."""
    for key in keys:
        if not isinstance(document, dict) or key not in document:
            return None
        document = document[key]
    return document


def _number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _runs_module(command, module: str) -> bool:
    """Whether a recorded command runs `python -m <module>`."""
    if not isinstance(command, list):
        return False
    return any(command[k - 1 : k + 1] == ["-m", module] for k in range(1, len(command)))


def _newest(top: Path, label: str) -> Path:
    try:
        return newest_run(top, label)
    except CannotCheck as problem:
        raise NoRun(str(problem)) from problem


def _verified(run: Path, results_name: str, head: str):
    evidence = read_evidence(run, results_name, head)
    if not evidence.verified:
        raise CannotCheck(
            f"{run.name}{evidence.note} ended with exit code {evidence.manifest.get('exit_code')}, "
            "so it is not verified and cannot decide the item"
        )
    return evidence


def _named(problems) -> str:
    more = len(problems) - NAMED
    return "; ".join(problems[:NAMED]) + (f"; and {more} more" if more > 0 else "")


def _stopped(problem: Exception) -> str:
    return f"the check stopped on {type(problem).__name__}: {problem}"


def check_matrix(ctx: Context, run: Path, document: dict, results_name: str) -> None:
    """Raises `CannotCheck` unless the run used the committed matrix (R5)."""
    used = _at(document, "matrix", "sha256")
    if not isinstance(used, str):
        raise CannotCheck(
            f"{results_name} of {run.name} records no SHA-256 of its matrix, so it is not shown to "
            f"be of the committed {MATRIX}"
        )
    if used != ctx.committed:
        raise CannotCheck(
            f"{run.name} ran on another matrix: SHA-256 {used[:12]}, the committed {MATRIX} has "
            f"{ctx.committed[:12]}"
        )


def _table(run: Path, table) -> tuple[str, str]:
    for kind in TABLE_KINDS:
        suffix = f"-{kind}.csv"
        if isinstance(table, str) and table.endswith(suffix):
            return table.removesuffix(suffix), kind
    raise CannotCheck(f"{run.name} lists the truth table {table!r}, which no truth build writes")


def check_inputs(ctx: Context, run: Path, document: dict) -> None:
    """Raises `CannotCheck` unless the run read the main truth setting from a recorded truth build
    that is verified and recorded at `ctx.head` or read from an earlier commit by the rule of
    `run_evidence`, every truth table it lists as that build recorded it, and lists both tables of
    every gate unit (R5)."""
    setting = _at(document, "inputs", "setting")
    if setting != MAIN:
        raise CannotCheck(f"{run.name} read the truth setting {setting!r}, not {MAIN!r}")
    name = _at(document, "inputs", "truth_run")
    if not isinstance(name, str) or not TRUTH_RUN.fullmatch(name):
        raise CannotCheck(
            f"{run.name} names no recorded truth build of the {MAIN} setting (its "
            f"inputs.truth_run is {name!r}, not <UTC start>-{RUN_LABELS[MAIN]})"
        )
    try:
        truth = _verified(ctx.top / RUNS_FOLDER / name, truth_build.SUMMARY_NAME, ctx.head)
    except CannotCheck as problem:
        build = "the truth build" if name in str(problem) else f"the truth build {name}"
        raise CannotCheck(f"{build} that {run.name} names cannot decide: {problem}") from problem
    summary, build = truth.data, f"{name}{truth.note}"
    try:
        found = summary_setting(summary)
    except CouldNotRun as problem:
        raise CannotCheck(f"the truth build {build}: {problem}") from problem
    if found != MAIN:
        raise CannotCheck(f"the truth build {build} is of the setting {found!r}, not {MAIN!r}")
    tables = _at(document, "inputs", "table_sha256")
    if not isinstance(tables, dict):
        raise CannotCheck(f"{run.name} lists no truth tables")
    for unit in ctx.units:
        for kind in TABLE_KINDS:
            if f"{unit}-{kind}.csv" not in tables:
                raise CannotCheck(f"{run.name} lists no {unit}-{kind}.csv among its truth tables")
    for table, digest in sorted(tables.items()):
        unit, kind = _table(run, table)
        recorded = _at(summary, "units", unit, "sha256", kind)
        if recorded is None:
            raise CannotCheck(f"{table} of {run.name} is not a table of the truth build {build}")
        if digest != recorded:
            raise CannotCheck(
                f"{table} of {run.name} has the SHA-256 {str(digest)[:12]}, not the "
                f"{str(recorded)[:12]} that the truth build {build} recorded"
            )


def check_splits(ctx: Context, run: Path, document: dict) -> int:
    """Raises `CannotCheck` unless the run has the required number of splits (R5)."""
    splits = _at(document, "run", "splits")
    if not _whole(splits) or splits != ctx.setup.splits:
        raise CannotCheck(f"{run.name} has {splits!r} splits, not {ctx.setup.splits}")
    return splits


def read_dials(ctx: Context, model: str) -> tuple[Path, dict, str]:
    """The model's dial-choice run, gate unit -> its tier A dial (R6) and the evidence's note
    (`Evidence.note`, empty for a run at the checked-out commit); raises `NoRun` or
    `CannotCheck`."""
    run = _newest(ctx.top, dial_label(model))
    evidence = _verified(run, dial_choice.RESULTS_NAME, ctx.head)
    try:
        return run, _dials(ctx, model, run, evidence), evidence.note
    except CannotCheck as problem:
        raise with_note(problem, evidence) from problem


def _dials(ctx: Context, model: str, run: Path, evidence) -> dict:
    if not _runs_module(evidence.manifest.get("command"), DIAL_CHOICE):
        raise CannotCheck(
            f"the recorded command of {run.name} does not run python -m {DIAL_CHOICE}"
        )
    document = evidence.data
    if document.get("model") != model:
        raise CannotCheck(
            f"{run.name} chose the dials of the model {document.get('model')!r}, not {model!r}"
        )
    check_matrix(ctx, run, document, dial_choice.RESULTS_NAME)
    check_inputs(ctx, run, document)
    check_splits(ctx, run, document)
    dials = {}
    for unit in ctx.units:
        source = DIAL_SOURCES[unit]
        dial = _at(document, "dial_choice", source, "A", "dial")
        if not _number(dial) or dial <= 0:
            raise CannotCheck(
                f"{run.name} gives no tier A dial at {source}, so the dial of {unit} is not known"
            )
        dials[unit] = dial
    return dials


def _check_draw(run: Path, document: dict, share: str | None) -> None:
    drawn = document.get("no_pause")
    if share is None:
        if drawn is not None:
            raise CannotCheck(
                f"{run.name} draws cars that charge as logged (share {_at(drawn, 'share')}, seed "
                f"{_at(drawn, 'seed')}); the comparison without the draw draws none"
            )
        return
    if not isinstance(drawn, dict):
        raise CannotCheck(
            f"{run.name} draws no car that charges as logged; the sensitivity needs the share "
            f"{share} at seed {NO_PAUSE_SEED}"
        )
    seed = drawn.get("seed")
    if drawn.get("share") != share or not _whole(seed) or seed != NO_PAUSE_SEED:
        raise CannotCheck(
            f"{run.name} draws the share {drawn.get('share')} at seed {seed}, not the share "
            f"{share} at seed {NO_PAUSE_SEED}"
        )


def _check_calibration(ctx: Context, run: Path, document: dict) -> None:
    splits = check_splits(ctx, run, document)
    seeds = _at(document, "run", "seeds")
    if seeds != [1, splits]:
        raise CannotCheck(f"{run.name} has the seeds {seeds!r}, not 1 to {splits} ([1, {splits}])")
    tiers = _at(document, "run", "tiers")
    if tiers != list(dial_splits.TIERS):
        raise CannotCheck(f"{run.name} has the tiers {tiers!r}, not {list(dial_splits.TIERS)!r}")
    variants = _at(document, "run", "variants")
    if variants != list(compare_run.VARIANTS):
        raise CannotCheck(
            f"{run.name} has the variants {variants!r}, not {list(compare_run.VARIANTS)!r}"
        )
    ceilings = _at(document, "run", "ceilings")
    if not isinstance(ceilings, list) or not all(isinstance(c, str) for c in ceilings):
        raise CannotCheck(f"{run.name} lists no ceilings")
    units = document.get("units")
    if not isinstance(units, dict):
        raise CannotCheck(f"{run.name} has no units in its results")
    for unit in ctx.units:
        status = _at(units, unit, "status")
        if unit in units and status != CALIBRATABLE:
            raise CannotCheck(
                f"{unit} has the status {status!r} in {run.name}: it cannot be split, so the run "
                "has no comparison there"
            )


def _check_files(run: Path, manifest: dict, names: dict) -> list[str]:
    """The output files' problems (R4); raises `CannotCheck` for a file changed since the run."""
    outputs = manifest.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else {}
    problems = []
    for name in names.values():
        path = run / name
        if not path.is_file():
            problems.append(f"the output file {name} is missing")
        elif name not in outputs:
            problems.append(f"{name} is not listed in the manifest")
        else:
            try:
                require_verified(path)
            except NotVerified as problem:
                raise CannotCheck(f"{name} of {run.name} is not verified: {problem}") from problem
    return problems


def read_comparison(ctx: Context, model: str, dial: float, share: str | None) -> Read:
    """The comparison run of `model` at `dial` (with the draw when `share` is given), once it is
    shown to decide (module text, "Runs read"); raises `NoRun` or `CannotCheck`."""
    run = _newest(ctx.top, compare_label(model, dial, share))
    names = compare_run.output_names(0 if share is None else no_pause.SHARES[share])
    evidence = _verified(run, names["results"], ctx.head)
    try:
        problems = _comparison_problems(ctx, model, dial, share, run, evidence, names)
    except CannotCheck as problem:
        raise with_note(problem, evidence) from problem
    return Read(run, evidence.data, evidence.manifest, names, tuple(problems), evidence.note)


def _comparison_problems(ctx: Context, model, dial, share, run: Path, evidence, names) -> list:
    if not _runs_module(evidence.manifest.get("command"), COMPARE_RUN):
        raise CannotCheck(
            f"the recorded command of {run.name} does not run python -m {COMPARE_RUN}"
        )
    document = evidence.data
    if document.get("model") != model:
        raise CannotCheck(f"{run.name} compares the model {document.get('model')!r}, not {model!r}")
    found = document.get("dial")
    if not _number(found) or found != dial:
        written = f"{found:g}" if _number(found) else repr(found)
        raise CannotCheck(f"{run.name} compares at the dial {written}, not {dial:g}")
    _check_draw(run, document, share)
    check_matrix(ctx, run, document, names["results"])
    check_inputs(ctx, run, document)
    _check_calibration(ctx, run, document)
    return _check_files(run, evidence.manifest, names)


def _record_problems(place: str, record, counting) -> list[str]:
    if not isinstance(record, dict):
        return [f"{place}: no record measured_on"]
    lacking = [field for field in RECORD_FIELDS if field not in record]
    if lacking:
        return [f"{place}: measured_on has no {', '.join(lacking)}"]
    days, rows = record["days"], record["path_rows"]
    problems = []
    if not _whole(days) or days != counting:
        problems.append(
            f"{place}: measured_on has {days!r} days, not the unit's {counting!r} counting days"
        )
    if not _whole(days) or not _whole(rows) or rows != PATH_ROWS_A_DAY * days:
        problems.append(
            f"{place}: measured_on has {rows!r} audited-path rows on {days!r} days, not "
            f"{PATH_ROWS_A_DAY} a day"
        )
    return problems


def result_problems(ctx: Context, document: dict, methods: list, problems: list) -> int:
    """Adds the missing results and records (R2 to R4); returns the unit-variants checked."""
    checked = 0
    for unit in ctx.units:
        entry = document["units"].get(unit)
        if not isinstance(entry, dict):
            problems.append(f"{unit} is not in the results")
            continue
        for variant in compare_run.VARIANTS:
            place = f"{unit} {variant}"
            found = _at(entry, "variants", variant)
            if not isinstance(_at(found, "methods"), dict):
                problems.append(f"{place}: no results")
                continue
            checked += 1
            for method in methods:
                for tier in dial_splits.TIERS:
                    if not isinstance(_at(found, "methods", method, tier), dict):
                        problems.append(f"{place}: {method}: no summary for tier {tier}")
                        break
            problems += _record_problems(
                place, found.get("measured_on"), entry.get("counting_days")
            )
        for method in methods:
            if not isinstance(_at(entry, "gate_reading", method), dict):
                problems.append(f"{unit}: {method}: no gate reading")
    return checked


def _shown(value: str) -> str:
    return value if value != "" else "empty"


def _shared(place: str, present: list, columns: list, problems: list) -> int:
    """Adds the disagreements of one unit, variant, tier and split (R3); returns 1 when the
    certified value was compared there, else 0."""
    if not present:
        return 0
    first, reference = present[0]
    for method, row in present[1:]:
        for column in columns:
            if row[column] != reference[column]:
                problems.append(
                    f"{place}: {method} has {column} {_shown(row[column])}, {first} "
                    f"{_shown(reference[column])}"
                )
    values = []
    for method, row in present:
        try:
            scale = float(row["scale"])
            if scale > 0:
                values.append((method, float(row["offer_kwh_per_counting_test_day"]) / scale))
        except ValueError:
            problems.append(f"{place}: {method} has no number for its scale or its offer")
    if len(values) < 2:
        return 0
    (first, value), *rest = values
    for method, other in rest:
        if not math.isclose(other, value, rel_tol=VALUE_TOLERANCE, abs_tol=0.0):
            problems.append(
                f"{place}: {method} has the certified value {other!r}, {first} {value!r} (the "
                "offer over the scale)"
            )
    return 1


def split_problems(ctx: Context, read: Read, methods: list, problems: list):
    """Adds the split table's problems (R3, R4); returns (rows, groups, groups where the certified
    value was compared), or None when the table cannot be read row by row."""
    name = read.names["splits"]
    path = read.run / name
    if not path.is_file() or name not in (read.manifest.get("outputs") or {}):
        return None  # named among the output files' problems
    ceilings = read.document["run"]["ceilings"]
    header = list(compare_splits.columns(ceilings))
    reader = csv.reader(io.StringIO(path.read_text()))
    if next(reader, None) != header:
        problems.append(
            f"the split table's header is not that of compare_splits.columns for the run's "
            f"ceilings {ceilings}"
        )
        return None
    rows = []
    for number, line in enumerate(reader, start=2):
        if len(line) != len(header):
            problems.append(
                f"line {number} of the split table has {len(line)} cells, not {len(header)}"
            )
        else:
            rows.append(dict(zip(header, line, strict=True)))
    splits = ctx.setup.splits
    expected = len(ctx.units) * len(compare_run.VARIANTS) * len(dial_splits.TIERS)
    expected *= len(methods) * splits
    said = _at(read.document, "split_table", "rows")
    if len(rows) != said or len(rows) != expected:
        problems.append(
            f"the split table has {len(rows)} rows; the results say {said!r}; {expected} are "
            "expected (one per unit, variant, tier, method and split)"
        )
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["unit"], row["variant"], row["tier"], row["method"], row["seed"])].append(row)
    columns = [*SHARED_COLUMNS, *(compare_splits.ceiling_column(c) for c in ceilings)]
    wanted, groups, compared = set(), 0, 0
    for unit in ctx.units:
        for variant in compare_run.VARIANTS:
            for tier in dial_splits.TIERS:
                for seed in range(1, splits + 1):
                    groups += 1
                    place = f"{unit} {variant} tier {tier} split {seed}"
                    present = []
                    for method in methods:
                        key = (unit, variant, tier, method, str(seed))
                        wanted.add(key)
                        found = by_key.get(key, [])
                        if not found:
                            problems.append(f"{place}: no row of {method}")
                        elif len(found) > 1:
                            problems.append(f"{place}: {len(found)} rows of {method}")
                        else:
                            present.append((method, found[0]))
                    compared += _shared(place, present, columns, problems)
    for unit, variant, tier, method, seed in by_key.keys() - wanted:
        problems.append(
            f"a row of {unit} {variant} tier {tier} split {seed} for {method}, which is not a "
            "unit, variant, tier, method and split of the run"
        )
    return len(rows), groups, compared


def twin_problems(ctx: Context, read: Read, main: Read, problems: list) -> int:
    """Adds where the run with the draw differs from the run without it in what must be the same
    (R7); returns the unit-variants whose records were compared."""
    mine, theirs, other = read.document, main.document, f"{main.run.name}{main.note}"
    if mine.get("methods") != theirs.get("methods"):
        problems.append(
            f"the methods differ from those of {other}: {mine.get('methods')} against "
            f"{theirs.get('methods')}"
        )
    if _at(mine, "forecast_table", "sha256") != _at(theirs, "forecast_table", "sha256"):
        problems.append(f"the forecast table differs from that of {other}")
    if _at(mine, "inputs", "table_sha256") != _at(theirs, "inputs", "table_sha256"):
        problems.append(f"the truth tables differ from those of {other}")
    compared = 0
    for unit in ctx.units:
        for variant in compare_run.VARIANTS:
            record = _at(mine, "units", unit, "variants", variant, "measured_on")
            without = _at(theirs, "units", unit, "variants", variant, "measured_on")
            if not isinstance(record, dict) or not isinstance(without, dict):
                continue  # named among each run's own problems
            compared += 1
            for field, words in (
                ("days", "days"),
                ("path_rows", "audited-path rows"),
                ("paths_sha256", "audited paths"),
            ):
                if record.get(field) != without.get(field):
                    problems.append(f"{unit} {variant}: {words} differ from those of {other}")
    return compared


def _judge(ctx: Context, read: Read, units: list, main: Read | None, share: str | None):
    """(outcome, the line's text after its name) of a run that can decide."""
    document = read.document
    problems = list(read.problems)
    for key, kind in (("split_table", "splits"), ("forecast_table", "table")):
        name = read.names[kind]
        entry = document.get(key)
        listed = read.manifest.get("outputs") or {}
        if _at(entry, "file") != name:
            problems.append(
                f"the results name no {name} as their {key.replace('_', ' ')} (they name "
                f"{_at(entry, 'file')!r})"
            )
        elif name in listed and _at(entry, "sha256") != listed[name]:
            problems.append(
                f"the results give {name} the SHA-256 {str(_at(entry, 'sha256'))[:12]}, the "
                f"manifest {str(listed[name])[:12]}"
            )
    methods = document.get("methods")
    if not isinstance(methods, list) or not all(isinstance(m, str) for m in methods):
        problems.append("the results list no methods")
        methods = []
    missing = [m for m in ctx.required if m not in methods]
    if missing:
        problems.append(
            f"methods {len(ctx.required) - len(missing)} of {len(ctx.required)}; missing "
            f"{', '.join(missing)}"
        )
    checked = result_problems(ctx, document, methods, problems)
    table = split_problems(ctx, read, methods, problems)
    twins = None if main is None else twin_problems(ctx, read, main, problems)
    if problems:
        return FAILED, f"{read.run.name}{read.note}; {_named(problems)}"
    if share is not None and main is None:
        return COULD_NOT_RUN, (
            f"{read.run.name}{read.note}; the run without the draw could not be read, so its "
            "methods, forecast table, truth tables and paths cannot be compared"
        )
    rows, groups, compared = table
    text = (
        f"{read.run.name}{read.note}; read for {', '.join(units)}; {len(ctx.required)} of "
        f"{len(ctx.required)} methods in its files; {checked} unit-variants with every method's "
        f"summaries and gate reading and the record measured_on; split table {rows} rows, every "
        f"method on the same days and ceilings on {groups} unit, variant, tier and split groups, "
        f"the certified value compared on {compared} of them; {len(read.names)} output files "
        "listed in the manifest with their SHA-256"
    )
    if twins is not None:
        text += (
            f"; its methods, forecast table, truth tables and, on {twins} unit-variants, its "
            f"days, audited-path rows and audited paths are those of {main.run.name}"
            f"{main.note}"
        )
    return PASSED, text


def check_comparison(ctx: Context, model: str, dial: float, units: list, share, main):
    """(outcome, line, the run read or None) of one comparison run (R1 to R5), or of its
    sensitivity (`share` given; R7, against `main`, the run without the draw or None)."""
    name = f"{model} at dial {dial:g}"
    if share is not None:
        name += f", no-pause {share} seed {NO_PAUSE_SEED}"
    read = None
    try:
        read = read_comparison(ctx, model, dial, share)
        outcome, text = _judge(ctx, read, units, main, share)
    except NoRun as problem:
        return COULD_NOT_RUN, f"NO RUN: {name}: {problem}", read
    except CannotCheck as problem:
        return COULD_NOT_RUN, f"COULD NOT RUN: {name}: {problem}", read
    except Exception as problem:  # an unexpected error (module text)
        return COULD_NOT_RUN, f"COULD NOT RUN: {name}: {_stopped(problem)}", read
    start = {PASSED: "shown", FAILED: "FAILED", COULD_NOT_RUN: "COULD NOT RUN"}[outcome]
    return outcome, f"{start}: {name}: {text}", read


def check_model(ctx: Context, model: str):
    """(outcome, line) of the model's dials and of each of its comparisons (R6, R7)."""
    name = f"dials of {model}"
    dials = None
    try:
        run, dials, note = read_dials(ctx, model)
        read = ", ".join(f"{u} {dials[u]:g} (read at {DIAL_SOURCES[u]})" for u in ctx.units)
        yield PASSED, f"dials: {model}: {run.name}{note}; tier A dial by gate unit: {read}"
    except NoRun as problem:
        yield COULD_NOT_RUN, f"NO RUN: {name}: {problem}"
    except CannotCheck as problem:
        yield COULD_NOT_RUN, f"COULD NOT RUN: {name}: {problem}"
    except Exception as problem:  # an unexpected error (module text)
        dials = None
        yield COULD_NOT_RUN, f"COULD NOT RUN: {name}: {_stopped(problem)}"
    if dials is None:
        line = (
            f"COULD NOT RUN: comparisons of {model}: its dials are not known, so the runs to read"
        )
        yield COULD_NOT_RUN, f"{line} are not known"
        return
    by_dial = {}
    for unit in ctx.units:
        by_dial.setdefault(dials[unit], []).append(unit)
    for dial, units in by_dial.items():
        outcome, line, main = check_comparison(ctx, model, dial, units, None, None)
        yield outcome, line
        outcome, line, _ = check_comparison(ctx, model, dial, units, NO_PAUSE_SHARE, main)
        yield outcome, line


def no_method(method: Method, registry: set) -> str | None:
    """The line of a listed method that no method of the registry implements (R1), else None."""
    if method.registry is None:
        return (
            f"NO METHOD: {method.plan_name}: plan section 10 names it; no method of "
            "fleetreserve.compare_methods implements it"
        )
    if not isinstance(method.registry, str) or method.registry not in registry:
        return (
            f"NO METHOD: {method.registry} ({method.plan_name}): not a method of "
            "fleetreserve.compare_methods"
        )
    return None


def check_all(top: Path, setup: Setup) -> int:
    head = head_commit(top)
    committed, matrix = committed_matrix(top)
    print(
        f"WP10 comparison runs checked at commit {head[:12]} (a run accepted from an earlier "
        f"commit says so on its line), on the committed {MATRIX} ({committed[:12]})"
    )
    units = tuple(matrix.gate_units)
    outside = [u for u in units if u not in DIAL_SOURCES]
    if outside:
        raise CannotCheck(
            f"the gate units {', '.join(outside)} of the committed {MATRIX} have no unit in "
            "DIAL_SOURCES to read their dial at"
        )
    if not setup.models:
        raise CannotCheck("no model is listed")
    if not setup.methods:
        raise CannotCheck("no method is listed")
    registry = {m.name for m in compare_methods.METHODS}
    required = tuple(
        dict.fromkeys(
            m.registry
            for m in setup.methods
            if isinstance(m.registry, str) and m.registry in registry
        )
    )
    ctx = Context(top, head, committed, units, setup, required)
    outcomes = []

    def say(outcome: str, line: str) -> None:
        outcomes.append(outcome)
        print(line)

    for model in setup.models:
        try:
            for outcome, line in check_model(ctx, model):
                say(outcome, line)
        except Exception as problem:  # an unexpected error (module text)
            say(COULD_NOT_RUN, f"COULD NOT RUN: {model}: {_stopped(problem)}")
    for method in setup.methods:
        line = no_method(method, registry)
        if line is not None:
            say(COULD_NOT_RUN, line)
    for text in setup.not_built:
        say(COULD_NOT_RUN, f"NOT BUILT: {text}")
    print(
        f"{outcomes.count(PASSED)} of {len(outcomes)} requirements shown, "
        f"{outcomes.count(FAILED)} failed, {outcomes.count(COULD_NOT_RUN)} could not run"
    )
    if FAILED in outcomes:
        return 1
    return 2 if COULD_NOT_RUN in outcomes else 0


REQUIRED = Setup()


def main(argv: list[str] | None = None, setup: Setup = REQUIRED) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/compare_shared_check.py",
        description="Decide the WP10 pass-mark item from the recorded comparison runs.",
    )
    parser.add_argument(
        "--top",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the repository top folder, where runs.noindex/ is (default: this repository)",
    )
    options = parser.parse_args(argv)
    try:
        return check_all(options.top, setup)
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except Exception as problem:  # an unexpected error (module text)
        print(f"COULD NOT RUN: {_stopped(problem)}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
