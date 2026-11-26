"""Forecast trial `mix`: the tree model and the neural model averaged, against the main model, on
the development gate units (a trial made after the sealed run, `after_sealed_run`).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.forecast_mix [--workers N]

The other options exist for tests and for a smaller matrix; the trial is the command run with NO
option (`--workers` changes the number of processes only). A command that gives `--kappa`,
`--splits`, `--main-model`, `--trial-model`, `--extra-level`, `--settings` or `--matrix` with a
value other than the committed settings' ends with exit 2 (`overrides`), unless it is also given
`--not-the-trial`, which runs it as another run whose label and first summary line say so. Each
option changes a setting: `--kappa`
the level, `--splits` the number of splits, `--main-model` and `--trial-model` the two models
compared, `--settings` the settings file read, `--matrix` the run matrix (its units, dial family and
calibration rule), `--tables` and `--truth-run` the tables read. The output says what was run
(`run`: the level, the splits, the models and the members of the trial model as run) and whether
every value equals the committed settings file's (`settings_check`, always against the committed
file, whatever `--settings` names; the summary says so, and names `--tables` and `--truth-run`
when they differ from the defaults, as they change the data read).

Every setting is fixed in `trials-after-sealed-run/forecast-mix.json` (`SETTINGS_FILE`), with its
source, before any result; the command reads its level, splits and the two models' names from
there and refuses to run without it.

What it does. Two models are measured as `dial_choice --kappa <level>` measures a model, in one
process, on the same matrix, the same units, the same dial family and the same seeded splits:

- the main model, `lgbm_tail` (plan section 7);
- the trial, `models.mix_tail_inputs`: at every level the mean, with equal weights, of the
  quantiles of `lgbm_tail` and of the neural model `hazard_tuned_inputs` at its one seed
  (`forecast_average`). No weight is searched or tuned.

Each model is checked, fitted on the training units (`learn_run.check_model`, `fit_units`) and
forecast as `dial_choice.make_factory` does; every configuration of the dial family at the level is
measured by `dial_choice.run`, so each day is prepared and evaluated at every scale of the grid,
and each split's scale is calibrated on that split's calibration days alone and scored on its test
days (`dial_splits`). The two runs are then paired split by split (`forecast_mix_pairs`).

Why no test day's outcome reaches its own forecast or calibration:

- The forecast. Both members are fitted on the cars tables of `fit_units` only, which
  `learn_run.check_model` limits to the training units and refuses if one is scored
  (`FitOnScored`); a scored unit is a development unit, never a training unit. The frame a model
  predicts from is `learn_run.forecast_frame`: the rows of counting days with keys and features
  known at the decision time, no outcome and no label (`learn_features.feature_frame`). The mix
  adds no input and no outcome of its own (`models.mix_tail_inputs`).
- The calibration. A split's scale comes from the losses of its calibration days only
  (`dial_splits.score_split`, `calibration` days), and its test days are scored at that scale.
  The test days of a seed are not in its calibration days (`dial_choice.split_indices`).
- The dial. Each model's own dial is the one with the largest mean tier A offer over the 200
  splits, the plan's rule (section 6), which reads the offers of the same splits that report
  them; the main model's is picked the same way, and `dials` and `same_dial` are reported beside
  `own_dial`. This is the dial choice `dial_choice` reports, not a leak of a day's own outcome
  into its own forecast.

The headline, fixed before any result: tier A, `own_dial` (each model at its own dial), per unit,
the mean paired difference `trial - main` in kWh per counting test day and its standard error over
the splits. `same_dial` and every dial of `dials` are reported beside it; the two tiers are never
pooled or averaged. It is read by a rule fixed in the settings file before any result
(`forecast_mix_reading`): per judged unit a bar, 8% of the main arm's offer at the workplace unit
and 30% at the pooled Norwegian unit (the neural model's seed spread there at kappa 0.3, an
analogy and not this trial's measured noise), and a
word, "shows a gain", "shows a loss" or "no measurable difference", for the share of the paired
difference; the one-line reading comes from the main level only. One pre-set extra line, the mix
at the extra level (0.4) against the main model at its own level, is judged by the same rule, is
reported beside it and never enters the one-line reading. The extra level is run in the same
command: the trial model is fitted again (the fit repeats) and forecast at that level.

Units. The command refuses, before it opens any table, truth build or model file, a matrix whose
gate units are not all development units of the truth build (`check_units`, exit 2). It names no
loader of the sealed data and reads no table of it. The committed matrix `gate/matrix.json` has the
three development gate units.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set, the summary also on standard output:
- `forecast-mix.json` (`RESULTS_NAME`): the settings, the inputs, both models' headline figures and
  the paired comparison, with `definitions`;
- `forecast-mix-summary.txt`, `forecast-mix-timing.json`;
- per run (`main`, `trial`, and `extra`, the trial model at the extra level): the full results of
  `dial_choice.run`
  (`forecast-mix-<side>-dial-choice.json`), its per-split offers
  (`forecast-mix-<side>-split-offers.json`) and its forecast table
  (`forecast-mix-<side>-forecasts.csv`, its hash is in the results);
- `after_sealed_run.FILE_NAME`, the label file.
Every output says that it is a trial made after held-out results were seen (the key
`made_after_sealed_run` and the label of each results file, the first lines of the summary and the
label file). The label is always on.

Exit status: 0 every configuration of both runs measured; 1 a model was refused or failed, or a
configuration failed; 2 could not run (a bad argument, matrix, settings file or model name, a unit
that is not a development unit, a model fitted on a scored unit, a table missing, no verified truth
build at this commit, a table differing from it).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from fleetreserve import (
    after_sealed_run,
    dial_choice,
    dial_summary,
    forecast_mix_pairs,
    forecast_mix_reading,
    gate_run,
    learn_run,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import write_table
from fleetreserve.gate_config import TIER_LABELS, MatrixError, load
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import DEVELOPMENT_UNITS, OUT
from fleetreserve.truth_verify import CouldNotRun

MATRIX = TOP / "gate" / "matrix.json"
SETTINGS_FILE = TOP / "trials-after-sealed-run" / "forecast-mix.json"
REQUIRED_SETTINGS = ("level", "extra_level", "splits", "main_model", "trial_model", "reading")
RESULTS_NAME = "forecast-mix.json"
SUMMARY_NAME = "forecast-mix-summary.txt"
TIMING_NAME = "forecast-mix-timing.json"
NOT_THE_TRIAL_FILE = "not-the-trial.txt"
SIDES = ("main", "trial")  # the two models
RUNS = ("main", "trial", "extra")  # the three measured runs; `extra` is the trial model again
NOT_THE_TRIAL_FLAG = "--not-the-trial"
NOT_THE_TRIAL = "NOT THE TRIAL OF THE COMMITTED SETTINGS: this run was asked for as another run"
LABEL = (
    "a trial: the average of the tree model's and the neural model's departure quantiles against "
    "the main model, paired split by split on development units"
)
UNCERTAINTY = (
    "The seeded splits share days, so the standard error over the splits understates the "
    "uncertainty of the whole experiment."
)
ONE_TRAINING = (
    "The mix uses one training (seed 0) of the neural member, so part of any difference is that "
    "one training's chance."
)


def side_name(side: str, kind: str) -> str:
    """The file name of `kind` (`dial-choice`, `split-offers`, `forecasts`) for a run `side`."""
    return f"forecast-mix-{side}-{kind}." + ("csv" if kind == "forecasts" else "json")


def extra_label(level: float) -> str:
    return f"extra line: the mix at level {level:g}"


# --- the settings and the units -------------------------------------------------------------------


def load_settings(path: Path = SETTINGS_FILE) -> dict:
    """The settings of the trial in the file `path`; raises `CouldNotRun` when the file is missing,
    is not JSON, lacks a key of `REQUIRED_SETTINGS` or has no bar in its `reading`."""
    try:
        gate_run.check_files([path])
        found = json.loads(Path(path).read_text())
    except (OSError, ValueError) as problem:
        raise CouldNotRun(
            f"the settings file {Path(path).name} cannot be read: {problem}"
        ) from problem
    missing = [k for k in REQUIRED_SETTINGS if k not in found]
    if missing:
        raise CouldNotRun(f"the settings file {Path(path).name} lacks {', '.join(missing)}")
    if not isinstance(found["reading"], dict) or not found["reading"].get("bars"):
        raise CouldNotRun(f"the settings file {Path(path).name} has no bars in its reading")
    return found


def bars_of(settings: dict) -> dict:
    """Judged unit -> bar (a share of the main arm's offer) from the `reading` block."""
    return {unit: float(found["share"]) for unit, found in settings["reading"]["bars"].items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def members_of(model) -> list[str]:
    """The names of the models a run's trial model averages (its `tail` and `neural` members), or
    the model's own name when it has none: read from what was run, never from the settings file."""
    found = [getattr(model, side, None) for side in ("tail", "neural")]
    names = [getattr(m, "name", type(m).__name__) for m in found if m is not None]
    return names or [model.name]


def settings_check(
    committed: dict, ran: dict, committed_hashes: dict, inputs: dict | None = None
) -> dict:
    """Whether every value of the run equals the committed settings file's (module text).
    `committed`: the committed file's content, never the file a run was given by `--settings`;
    `ran`: level, extra_level, splits, main_model, trial_model, members and the SHA-256 of the
    matrix and of the settings file as run; `committed_hashes`: the SHA-256 of the committed matrix
    and settings file under the same keys. Each difference is listed with both values. `inputs`:
    the data options (`--tables`, `--truth-run`) that differ from their defaults, which change
    the data read and not the settings; they are reported beside the check."""
    expected = {
        "level": committed["level"],
        "extra_level": committed["extra_level"],
        "splits": committed["splits"],
        "main_model": committed["main_model"],
        "trial_model": committed["trial_model"],
        "members": list(committed.get("members", [])),
        **committed_hashes,
    }
    differences = {
        key: {"run": ran.get(key), "committed": value}
        for key, value in expected.items()
        if ran.get(key) != value
    }
    return {
        "every_value_equals_the_committed_settings": not differences,
        "differences": differences,
        "inputs_that_differ_from_the_defaults": dict(inputs or {}),
    }


def overrides(args, committed: dict) -> list[str]:
    """The options of `args` that change a setting: a value other than the committed settings'
    for `--kappa`, `--splits`, `--main-model`, `--trial-model` or `--extra-level`, or a settings
    file or a matrix that is not the committed file (by content). Nothing is read but those two."""
    found = []
    for option, attribute, key in (
        ("--kappa", "kappa", "level"),
        ("--extra-level", "extra_level", "extra_level"),
        ("--splits", "splits", "splits"),
        ("--main-model", "main_model", "main_model"),
        ("--trial-model", "trial_model", "trial_model"),
    ):
        value = getattr(args, attribute)
        if value is not None and value != committed[key]:
            found.append(f"{option} {value} (committed: {committed[key]})")
    if _sha256(args.settings) != _sha256(SETTINGS_FILE):
        found.append("--settings (not the committed settings file)")
    if _sha256(args.matrix) != _sha256(MATRIX):
        found.append("--matrix (not the committed run matrix)")
    return found


def data_options(args) -> dict:
    """The data options of `args` that differ from their defaults."""
    found = {}
    if Path(args.tables) != OUT:
        found["tables"] = str(args.tables)
    if args.truth_run is not None:
        found["truth_run"] = str(args.truth_run)
    return found


def check_units(units) -> tuple[str, ...]:
    """`units` as a tuple when each is a development unit of the truth build; raises `CouldNotRun`
    otherwise. Nothing is read."""
    units = tuple(units)
    if not units:
        raise CouldNotRun("the matrix names no gate unit")
    unknown = [u for u in units if u not in DEVELOPMENT_UNITS]
    if unknown:
        raise CouldNotRun(
            f"the matrix's gate units include {', '.join(unknown)}, not a development unit "
            f"({', '.join(DEVELOPMENT_UNITS)}); training, held-out and sealed names are refused: "
            "the trial runs on development units only"
        )
    return units


# --- one model ------------------------------------------------------------------------------------


@dataclass
class Measured:
    """What one run gave."""

    name: str
    model: object
    document: dict
    timing: dict
    offers: dict
    table: object


def measure(name, model, matrix, family, units, loaded, splits, workers) -> Measured:
    """The model fitted and every configuration of the family measured (module text); raises
    `learn_run.Refused` and `ValueError` as `dial_choice.make_factory` does."""
    fit_frames = {u: loaded[u].cars for u in model.fit_units}
    factory, table = dial_choice.make_factory(
        name, model, matrix, units, fit_frames, family[0].kappa
    )
    offers: dict = {}
    document, timing = dial_choice.run(
        matrix, units, factory, splits=splits, workers=workers, family=family, split_offers=offers
    )
    document["model"] = model.name
    document["fit_units"] = list(model.fit_units)
    return Measured(name, model, document, timing, offers, table)


# --- the summary ----------------------------------------------------------------------------------


def _dial(value) -> str:
    return "none" if value is None else f"{value:g}"


def _cell_line(label: str, cell: dict | None) -> str:
    if cell is None:
        return f"  {label}: no figure (a model has no dial with a mean offer, or a run failed)"
    number = dial_summary._number
    ratio = cell["ratio_of_means_trial_to_main"]
    return (
        f"  {label}: main {number(cell['mean_main_offer_kwh_per_test_day'])}, trial "
        f"{number(cell['mean_trial_offer_kwh_per_test_day'])} kWh; difference "
        f"{number(cell['mean_difference_kwh_per_test_day'])} (standard error "
        f"{number(cell['standard_error_of_the_difference'])}; p10 "
        f"{number(cell['difference_p10'])}, p50 {number(cell['difference_p50'])}, p90 "
        f"{number(cell['difference_p90'])}); trial offers more in "
        f"{cell['splits_trial_offers_more']}, the same in {cell['splits_equal']}, less in "
        f"{cell['splits_trial_offers_less']} of {cell['splits']} splits; ratio of means "
        f"{number(ratio, 3)}"
    )


def _verdict_line(read: dict | None) -> str:
    if read is None:
        return "  Reading (tier A, own dial): not judged, the unit has no bar"
    share = "none" if read["share"] is None else f"{100 * read['share']:.1f}%"
    return (
        f"  Reading (tier A, own dial): share {share} of the main arm's offer against the bar "
        f"{100 * read['bar']:.0f}%: {read['word']}"
    )


def _unit_blocks(comparison: dict, read: dict) -> list[str]:
    """The lines of a comparison, per unit and tier, with the unit's word after tier A."""
    lines = []
    for unit, entry in comparison.items():
        lines.append("")
        if entry["status"] != "calibratable":
            lines.append(f"{unit} ({entry['status']}, {entry['counting_days']} counting days):")
            lines.append("  no numbers: the unit cannot be split")
            continue
        lines.append(f"{unit} ({entry['counting_days']} counting days):")
        if entry["failed"]:
            lines.append(f"  failed configurations (left out): {', '.join(entry['failed'])}")
        for tier, block in entry["tiers"].items():
            picks = block["picked_dial"]
            note = " (information only, not read by the rule)" if tier != "A" else ""
            lines.append(f" {TIER_LABELS[tier]}{note}")
            lines.append(
                _cell_line(
                    f"own dial (main {_dial(picks['main'])}, trial {_dial(picks['trial'])})",
                    block["own_dial"],
                )
            )
            lines.append(_cell_line(f"same dial ({_dial(picks['main'])})", block["same_dial"]))
            for cell in block["dials"].values():
                label = "no rows" if cell["dial"] is None else f"eps {cell['dial']:g}"
                lines.append(_cell_line(label, cell))
            if tier == "A":
                lines.append(_verdict_line(read.get(unit)))
    return lines


def summary_lines(document: dict) -> list[str]:
    run = document["run"]
    reading = document["reading"]
    lines = []
    if "not_the_trial" in document:
        asked = ", ".join(document["not_the_trial"]["overrides"]) or "no option changing a setting"
        lines.append(f"{NOT_THE_TRIAL}; {asked}")
    lines += [
        *after_sealed_run.summary_head(document),
        f"Forecast trial mix: main model {run['main_model']} against {run['trial_model']} (the "
        f"mean, with equal weights, of the departure quantiles of {', '.join(run['members'])}), "
        f"{run['splits']} seeded splits per unit (seeds 1 to {run['splits']}), kappa "
        f"{run['kappa']:g}, one-way re-solve rung.",
        LABEL[0].upper() + LABEL[1:] + ".",
        "Offers are kWh per counting test day, calibrated on each split's calibration days; the "
        "difference is trial minus main on the same split (same seed, same days); a split with no "
        "safe scale counts as an offer of 0. The headline is tier A at each model's own dial "
        "(the dial with its largest mean offer); same dial: both at the main model's dial.",
        "The two tiers are separate blocks and are never added or averaged.",
        UNCERTAINTY,
        ONE_TRAINING,
        reading["rule"],
        reading["no_other_sites"],
    ]
    check = document["settings_check"]
    if check["every_value_equals_the_committed_settings"]:
        lines.append("Every value of this run equals the committed settings file's.")
    else:
        changed = "; ".join(
            f"{k}: run {v['run']!r}, committed {v['committed']!r}"
            for k, v in check["differences"].items()
        )
        lines.append(f"A value of this run differs from the committed settings file's ({changed}).")
    if check["inputs_that_differ_from_the_defaults"]:
        other = check["inputs_that_differ_from_the_defaults"]
        data = ", ".join(f"{k} {v}" for k, v in other.items())
        lines.append(f"Data options differing from the defaults (the data may differ): {data}.")
    lines += _unit_blocks(document["comparison"], reading["main_level"]["units"])
    lines.append("")
    one = reading["main_level"]["one_line"]
    lines.append(f"One-line reading (level {run['kappa']:g} only): {one}.")
    extra = reading["extra_line"]
    lines.append("")
    lines.append(
        f"{extra['label']} (the main model stays at level {run['kappa']:g}; reported as an "
        "extra, judged by the same rule, never part of the one-line reading):"
    )
    lines += _unit_blocks(document["extra_comparison"], extra["units"])
    lines.append("")
    for side in RUNS:
        failed = document["models"][side]["failed_configurations"]
        lines.append(
            f"Timeouts of the {side} run (recorded outcomes): "
            + (
                ", ".join(f"{k} {v}" for k, v in document["models"][side]["timeouts"].items())
                or "none"
            )
        )
        if failed:
            lines.append(f"FAILED in the {side} run: " + "; ".join(failed))
    return lines


# --- the command ----------------------------------------------------------------------------------


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def definitions(committed: dict, splits: int) -> dict:
    return {
        "trial": LABEL,
        "pairing": (
            f"the two models are measured on the same {splits} seeded splits per unit (seeds 1 to "
            f"{splits}) of the same matrix, so a seed is the same calibration days and the same "
            "test days for both; per split the difference trial minus main of the calibrated "
            "offer per counting test day (forecast_mix_pairs)"
        ),
        "mix": (
            "at every level, the mean of the two members' values with equal weights, cell by "
            "cell; a cell that is NaN in either member is NaN (forecast_average)"
        ),
        "headline": "tier A, own_dial: each model at the dial with its largest mean offer",
        "reading": {
            "rule": forecast_mix_reading.RULE,
            "no_other_sites": forecast_mix_reading.NO_OTHER_SITES,
            "bars": committed["reading"]["bars"],
        },
        "committed_settings": committed,
        "tiers": "the tiers are separate blocks and are never added or averaged",
        "uncertainty": [UNCERTAINTY, ONE_TRAINING],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--settings", type=Path, default=SETTINGS_FILE)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--extra-level", type=float, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--main-model", default=None)
    parser.add_argument("--trial-model", default=None)
    parser.add_argument(NOT_THE_TRIAL_FLAG, dest="not_the_trial", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.settings)
        committed = load_settings(SETTINGS_FILE)
        kappa = settings["level"] if args.kappa is None else args.kappa
        extra_level = settings["extra_level"] if args.extra_level is None else args.extra_level
        splits = settings["splits"] if args.splits is None else args.splits
        names = {
            "main": settings["main_model"] if args.main_model is None else args.main_model,
            "trial": settings["trial_model"] if args.trial_model is None else args.trial_model,
        }
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if splits < 1:
            raise MatrixError("--splits must be at least 1")
        gate_run.argument_paths(args)
        gate_run.check_files([args.matrix, args.settings])
        changed = overrides(args, committed)
        if changed and not args.not_the_trial:
            raise CouldNotRun(
                f"{', '.join(changed)} change a setting of the trial; the trial is this command "
                f"with no option. To run it as another run, labelled so, give {NOT_THE_TRIAL_FLAG}"
            )
        matrix = load(args.matrix)
        check_units(matrix.gate_units)  # before any table, truth build or model file is read
        family = dial_choice.find_family(matrix)
        measured_matrix, main_family = dial_choice.family_at(matrix, family, kappa)
        extra_matrix, extra_family = dial_choice.family_at(matrix, family, extra_level)
        models = {side: learn_run.load_model(names[side], matrix) for side in SIDES}
    except (CouldNotRun, MatrixError) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        for side in SIDES:
            learn_run.check_model(
                names[side], models[side], tuple(models[side].fit_units), matrix.gate_units
            )
    except learn_run.FitOnScored as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    fit_units = tuple(dict.fromkeys(u for side in SIDES for u in models[side].fit_units))
    needed = tuple(dict.fromkeys((*matrix.gate_units, *fit_units)))
    try:
        inputs = gate_run.check_inputs(args.truth_run, args.tables, needed)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        loaded = {
            name: load_unit_days(
                args.tables,
                name,
                rho=matrix.rho,
                no_fit=matrix.no_fit,
                small_fleet=matrix.small_fleet,
            )
            for name in needed
        }
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    units = {name: loaded[name] for name in matrix.gate_units}
    started = time.perf_counter()
    found: dict[str, Measured] = {}
    plans = (
        ("main", "main", measured_matrix, main_family),
        ("trial", "trial", measured_matrix, main_family),
        ("extra", "trial", extra_matrix, extra_family),
    )
    try:
        for run_name, side, run_matrix, run_family in plans:
            found[run_name] = measure(
                names[side],
                models[side],
                run_matrix,
                run_family,
                units,
                loaded,
                splits,
                args.workers,
            )
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: a model could not be scored: {problem}")
        return 1
    try:
        comparison = forecast_mix_pairs.compare(
            found["main"].document,
            found["main"].offers,
            found["trial"].document,
            found["trial"].offers,
        )
        extra_comparison = forecast_mix_pairs.compare(
            found["main"].document,
            found["main"].offers,
            found["extra"].document,
            found["extra"].offers,
            ignore=("kappa",),
        )
    except ValueError as problem:
        print(f"FAILED: the runs cannot be paired: {problem}")
        return 1
    matrix_record = {"file": args.matrix.name, "sha256": _sha256(args.matrix)}
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    folder = Path(run_dir) if run_dir else None
    failed = []
    for run_name in RUNS:
        one = found[run_name]
        document = one.document
        document["inputs"] = inputs
        document["matrix"] = matrix_record
        after_sealed_run.mark(document, True)
        if args.not_the_trial:
            document["not_the_trial"] = {"overrides": changed}
            document["label"] = f"{NOT_THE_TRIAL}. {document['label']}"
        if folder is not None:
            table_name = side_name(run_name, "forecasts")
            write_table(one.table, folder / table_name)
            document["forecast_table"] = {
                "file": table_name,
                "rows": len(one.table),
                "sha256": hashlib.sha256((folder / table_name).read_bytes()).hexdigest(),
            }
        failed += [f"{run_name}: {text}" for text in document["failed_configurations"]]
    bars = bars_of(settings)
    units_read = forecast_mix_reading.read_units(comparison, bars)
    extra_read = forecast_mix_reading.read_units(extra_comparison, bars)
    first = found["main"].document
    ran = {
        "level": kappa,
        "extra_level": extra_level,
        "splits": splits,
        "main_model": names["main"],
        "trial_model": names["trial"],
        "members": members_of(models["trial"]),
        "matrix_sha256": matrix_record["sha256"],
        "settings_sha256": _sha256(args.settings),
    }
    reference = {"matrix_sha256": _sha256(MATRIX), "settings_sha256": _sha256(SETTINGS_FILE)}
    document = {
        "label": f"forecast trial mix: {LABEL}",
        "definitions": definitions(committed, splits),
        "run": {
            "kappa": kappa,
            "extra_level": extra_level,
            "splits": splits,
            "seeds": first["run"]["seeds"],
            "main_model": names["main"],
            "trial_model": names["trial"],
            "members": members_of(models["trial"]),
            "dials": first["run"]["dials"],
            "tiers": first["run"]["tiers"],
            "rung": first["run"]["rung"],
            "units": list(matrix.gate_units),
        },
        "models": {
            run_name: {
                "name": found[run_name].document["model"],
                "fit_units": found[run_name].document["fit_units"],
                "level": found[run_name].document["run"]["kappa"],
                "forecast_table": found[run_name].document.get("forecast_table"),
                "dial_choice": found[run_name].document["dial_choice"],
                "timeouts": found[run_name].document["timeouts"],
                "failed_configurations": found[run_name].document["failed_configurations"],
            }
            for run_name in RUNS
        },
        "reading": {
            "rule": forecast_mix_reading.RULE,
            "no_other_sites": forecast_mix_reading.NO_OTHER_SITES,
            "bars": settings["reading"]["bars"],
            "main_level": {
                "level": kappa,
                "units": units_read,
                "one_line": forecast_mix_reading.one_line(units_read),
            },
            "extra_line": {
                "label": extra_label(extra_level),
                "level": extra_level,
                "main_arm_level": kappa,
                "units": extra_read,
                "in_the_one_line_reading": False,
            },
        },
        "settings_check": settings_check(committed, ran, reference, data_options(args)),
        "comparison": comparison,
        "extra_comparison": extra_comparison,
        "failed_configurations": sorted(failed),
        "inputs": inputs,
        "matrix": matrix_record,
    }
    after_sealed_run.mark(document, True)
    if args.not_the_trial:
        document["not_the_trial"] = {"overrides": changed}
        document["label"] = f"{NOT_THE_TRIAL}. {document['label']}"
    text = "\n".join(summary_lines(document)) + "\n"
    print(text, end="")
    if folder is not None:
        timing = {run_name: found[run_name].timing for run_name in RUNS}
        timing["total_seconds"] = time.perf_counter() - started
        for run_name in RUNS:
            one = found[run_name].document
            results = _json(one)
            (folder / side_name(run_name, "dial-choice")).write_text(results)
            offers = dial_choice.split_offers_document(
                one, found[run_name].offers, side_name(run_name, "dial-choice"), results
            )
            (folder / side_name(run_name, "split-offers")).write_text(_json(offers))
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / TIMING_NAME).write_text(_json(timing))
        after_sealed_run.write_note(folder, True)
        if args.not_the_trial:
            with (folder / after_sealed_run.FILE_NAME).open("a") as label_file:
                label_file.write(f"{NOT_THE_TRIAL}\n")
            note = "\n".join([NOT_THE_TRIAL, *changed])
            (folder / NOT_THE_TRIAL_FILE).write_text(note + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
