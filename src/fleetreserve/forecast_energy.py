"""Forecast trial `energy`: more cautious energy bounds against the project's energy bounds and
against known energy, on the development gate units (a trial made after the sealed run,
`after_sealed_run`). APPROVED TO RUN by the owner's decision of 24 November 2026: the committed
settings file says `approved_to_run: true`, and the command ends with exit 2 if that file does not.

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.forecast_energy [--workers N]

The other options exist for tests and for a smaller matrix; the trial is the command run with NO
option (`--workers` changes the number of processes only). A command that gives `--kappa`,
`--splits`, `--model`, `--settings` or `--matrix` with a value other than the committed settings'
ends with exit 2 (`overrides`), unless it is also given `--not-the-trial`, which runs it as another
run whose label and first summary line say so. `--tables` and `--truth-run` change the tables read:
they are reported (`settings_check`, `inputs_that_differ_from_the_defaults`), not refused.

The approval is read from the COMMITTED settings file only (`committed_approval`): a copy given by
`--settings`, with or without `--not-the-trial`, never grants it. The command ends with exit 2 when
the committed value is not true, before any table, truth build or model is read (a held-out unit
name is refused even earlier).

Every setting is fixed in `trials-after-sealed-run/forecast-energy.json` (`SETTINGS_FILE`), with its
source, before any result; the command reads its level, splits, model and bars and loss limits from
there and refuses to run without it.

What it does. One model (`lgbm_tail`) is checked, fitted once on the training units and made to
forecast at the levels of the dial rows up to the run's level (`dial_choice.make_factory`), so
every arm reads the same forecast table. Three arms are then measured as `dial_choice --kappa
<level>` measures a model (`dial_choice.run`), on the same units, the same dial family at the level
and the same seeded splits, each on a matrix whose configurations have one `energy_bounds` value:

1. `known`: the car's true rate, storable energy and needed energy (only the departure forecast);
2. `empirical`: the three bounds as the model's quantiles at the level;
3. `cautious` (`forecast_empirical.CAUTIOUS`): the rate and the storable energy at level 0.05, the
   needed energy as in arm 2.

The arms differ in the energy bounds alone. Each split's scale is calibrated on that split's
calibration days alone and scored on its test days (`dial_splits`); each arm is read at its own
dial. The arms are then paired split by split and read by the rule fixed in the settings file
(`forecast_energy_reading`: the module text holds the rule, the bars and the one-line reading).

Why no test day's outcome reaches its own forecast or calibration:

- The forecast. The model is fitted on the cars tables of its `fit_units` only, which
  `learn_run.check_model` limits to the training units and refuses if one is scored
  (`FitOnScored`). The frame it predicts from is `learn_run.forecast_frame`: rows of counting days
  with keys and features known at the decision time, no outcome and no label.
- The calibration. A split's scale comes from the losses of its calibration days only
  (`dial_splits.score_split`) and its test days are scored at that scale; the test days of a seed
  are not in its calibration days (`dial_choice.split_indices`).
- The dial. Each arm's own dial is the one with the largest mean tier A offer over the 200 splits,
  the plan's rule (section 6), which reads the offers of the same splits that report them, for
  every arm alike, as every development figure of the project does. It is not free of test-day
  information.

Units. The command refuses, before it opens any table, truth build or model file, a matrix whose
gate units are not all development units of the truth build (`forecast_mix.check_units`, exit 2).
It names no loader of the sealed data and reads no table of it.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set, the summary also on standard output:
- `forecast-energy.json` (`RESULTS_NAME`): the settings, the inputs, the three arms' headline
  figures, the paired comparison and the reading, with `definitions`;
- `forecast-energy-summary.txt`, `forecast-energy-timing.json`;
- one forecast table `forecast-energy-forecasts.csv` (its hash is in the results), and per arm
  (`known`, `empirical`, `cautious`) the full results of `dial_choice.run`
  (`forecast-energy-<arm>-dial-choice.json`) and its per-split offers
  (`forecast-energy-<arm>-split-offers.json`);
- `after_sealed_run.FILE_NAME`, the label file.
Every output says that it is a trial made after held-out results were seen (the key
`made_after_sealed_run` and the label of each results file, the first lines of the summary and the
label file). The label is always on.

Exit status: 0 every configuration of the three arms measured; 1 the model was refused or failed,
or a configuration failed; 2 could not run (a bad argument, matrix, settings file or model name,
not approved to run, a unit that is not a development unit, a model fitted on a scored unit, a
table missing, no verified truth build at this commit, a table differing from it).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from fleetreserve import (
    after_sealed_run,
    crc,
    dial_choice,
    dial_summary,
    forecast_energy_reading,
    forecast_mix,
    gate_run,
    learn_run,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_empirical import CAUTIOUS_LEVEL
from fleetreserve.forecast_model import write_table
from fleetreserve.gate_config import ENERGY_BOUNDS, TIER_LABELS, MatrixError, load
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

MATRIX = forecast_mix.MATRIX
SETTINGS_FILE = TOP / "trials-after-sealed-run" / "forecast-energy.json"
REQUIRED_SETTINGS = (
    "approved_to_run",
    "level",
    "rate_cap_level",
    "need_level",
    "splits",
    "model",
    "arms",
    "reading",
)
RESULTS_NAME = "forecast-energy.json"
SUMMARY_NAME = "forecast-energy-summary.txt"
TIMING_NAME = "forecast-energy-timing.json"
TABLE_NAME = "forecast-energy-forecasts.csv"
NOT_THE_TRIAL_FILE = forecast_mix.NOT_THE_TRIAL_FILE
NOT_THE_TRIAL_FLAG = forecast_mix.NOT_THE_TRIAL_FLAG
NOT_THE_TRIAL = forecast_mix.NOT_THE_TRIAL
ARMS = forecast_energy_reading.ARMS  # arm 1, arm 2, arm 3: the order the arms are measured in
LIMIT_KEYS = ("slot", "driver")
LABEL = (
    "a trial: more cautious energy bounds (the top charging speed and the storable energy at "
    "level 0.05, the needed energy unchanged) against the project's energy bounds and against "
    "known energy, each arm at its own driver-risk dial, paired split by split on development "
    "units"
)
UNCERTAINTY = (
    "The seeded splits share days, so the standard error over the splits understates the "
    "uncertainty of the whole experiment."
)
ONE_FIT = (
    "The three arms read one fit of one model, so part of any difference between arms is not "
    "this model's chance; a different model or fit could give a different difference."
)


def side_name(arm: str, kind: str) -> str:
    """The file name of `kind` (`dial-choice`, `split-offers`) for an arm."""
    return f"forecast-energy-{arm}-{kind}.json"


# --- the settings and the units -------------------------------------------------------------------


def load_settings(path: Path = SETTINGS_FILE) -> dict:
    """The settings of the trial in the file `path`; raises `CouldNotRun` when the file is missing,
    is not JSON, lacks a key of `REQUIRED_SETTINGS`, has no bars or loss limits in its `reading`,
    or names arms other than `ARMS` in order."""
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
    limits = found["reading"].get("loss_limits")
    if not isinstance(limits, dict) or not all(k in limits for k in LIMIT_KEYS):
        raise CouldNotRun(f"the settings file {Path(path).name} has no loss limits in its reading")
    if found["arms"] != list(ARMS):
        raise CouldNotRun(f"the settings file {Path(path).name} names arms {found['arms']!r}")
    return found


def committed_approval() -> bool:
    """Whether the COMMITTED settings file (`SETTINGS_FILE`, the file of the checkout this module
    is imported from) says `approved_to_run: true` (module text). It reads that file and no other,
    so no `--settings` copy and no other option can grant the approval; `CouldNotRun` when the file
    cannot be read. Tests approve by replacing this function."""
    return load_settings(SETTINGS_FILE)["approved_to_run"] is True


def bars_of(settings: dict) -> dict:
    """Judged unit -> bar (a share of arm 1's offer) from the `reading` block."""
    return {unit: float(found["share"]) for unit, found in settings["reading"]["bars"].items()}


def limits_of(settings: dict) -> dict:
    """The loss limits of the `reading` block: `slot` and `driver`."""
    return {k: float(settings["reading"]["loss_limits"][k]) for k in LIMIT_KEYS}


def settings_check(
    committed: dict, ran: dict, committed_hashes: dict, inputs: dict | None = None
) -> dict:
    """Whether every value of the run equals the committed settings file's (module text).
    `committed`: the committed file's content, never the file a run was given by `--settings`;
    `ran`: level, rate_cap_level, need_level, splits, model, arms and the SHA-256 of the matrix and
    of the settings file as run; `committed_hashes`: the SHA-256 of the committed matrix and
    settings file under the same keys. Each difference is listed with both values. `inputs`: the
    data options (`--tables`, `--truth-run`) that differ from their defaults, which change the
    data read and not the settings; they are reported beside the check."""
    expected = {
        "level": committed["level"],
        "rate_cap_level": committed["rate_cap_level"],
        "need_level": committed["need_level"],
        "splits": committed["splits"],
        "model": committed["model"],
        "arms": list(committed["arms"]),
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
    """The options of `args` that change a setting: a value other than the committed settings' for
    `--kappa`, `--splits` or `--model`, or a settings file or a matrix that is not the committed
    file (by content). Nothing is read but those two files. The approval is not an override: it is
    read from the committed file alone (`committed_approval`)."""
    found = []
    for option, attribute, value in (
        ("--kappa", "kappa", committed["level"]),
        ("--splits", "splits", committed["splits"]),
        ("--model", "model", committed["model"]),
    ):
        given = getattr(args, attribute)
        if given is not None and given != value:
            found.append(f"{option} {given} (committed: {value})")
    if forecast_mix._sha256(args.settings) != forecast_mix._sha256(SETTINGS_FILE):
        found.append("--settings (not the committed settings file)")
    if forecast_mix._sha256(args.matrix) != forecast_mix._sha256(MATRIX):
        found.append("--matrix (not the committed run matrix)")
    return found


def with_energy_bounds(matrix, value: str):
    """The matrix with every configuration's `energy_bounds` set to `value` (a value of
    `gate_config.ENERGY_BOUNDS`); an oracle partner changes with its configuration, so the
    partners stay equal in that field (`gate_config._check_partners`). Nothing else changes."""
    if value not in ENERGY_BOUNDS:
        raise MatrixError(f"energy_bounds must be one of {ENERGY_BOUNDS}, got {value!r}")
    return replace(
        matrix,
        configurations=tuple(replace(c, energy_bounds=value) for c in matrix.configurations),
    )


# --- one arm --------------------------------------------------------------------------------------


@dataclass
class Measured:
    """What one arm gave."""

    arm: str
    document: dict
    timing: dict
    offers: dict


def measure(arm, matrix, family, units, factory, splits, workers) -> Measured:
    """The arm measured by `dial_choice.run` on `factory` (module text); raises `ValueError` as
    `dial_choice.run` does. `matrix` and `family` are the arm's own (`with_energy_bounds`, then
    `dial_choice.family_at`)."""
    offers: dict = {}
    document, timing = dial_choice.run(
        matrix, units, factory, splits=splits, workers=workers, family=family, split_offers=offers
    )
    document["energy_bounds"] = arm
    return Measured(arm, document, timing, offers)


# --- the summary ----------------------------------------------------------------------------------


def _number(value, digits: int = 2) -> str:
    return dial_summary._number(value, digits)


def _percent(value) -> str:
    return "none" if value is None else f"{100 * value:.1f}%"


def _dial(value) -> str:
    return "none" if value is None else f"{value:g}"


def _block_lines(block: dict) -> list[str]:
    picks = block["picked_dial"]
    dials = ", ".join(f"{arm} {_dial(picks[arm])}" for arm in ARMS)
    if block["cell"] is None:
        return [f"  own dials ({dials}): no figure (an arm has no dial with a mean offer)"]
    means = block["mean_offer_kwh_per_test_day"]
    cell = block["cell"]
    shares = block["share_of_known_offer"]
    return [
        f"  own dials ({dials}); mean offer, kWh per counting test day: known "
        f"{_number(means['known'])}, empirical {_number(means['empirical'])}, cautious "
        f"{_number(means['cautious'])}",
        f"  cautious minus empirical: "
        f"{_number(cell['mean_difference_kwh_per_test_day'])} (standard "
        f"error {_number(cell['standard_error_of_the_difference'])}; p10 "
        f"{_number(cell['difference_p10'])}, p50 {_number(cell['difference_p50'])}, p90 "
        f"{_number(cell['difference_p90'])}); cautious offers more in "
        f"{cell['splits_trial_offers_more']}, the same in {cell['splits_equal']}, less in "
        f"{cell['splits_trial_offers_less']} of {cell['splits']} splits",
        f"  as a share of the known-energy offer: empirical {_percent(shares['empirical'])}, "
        f"cautious {_percent(shares['cautious'])}; the difference "
        f"{_percent(block['difference_share_of_known_offer'])}; of the gap between known and "
        f"empirical {_percent(block['recovered_share_of_the_gap'])}",
    ]


def _verdict_line(read: dict, losses: dict | None) -> str:
    if read["word"] is None:
        return f"  Reading (tier A): {read['note']}; figures printed only"
    shown = (
        "none"
        if losses is None
        else (f"slot {_number(losses['slot'], 3)}, driver {_number(losses['driver'], 3)}")
    )
    note = "" if read["note"] is None else f" ({read['note']})"
    return (
        f"  Reading (tier A): difference {_percent(read['share'])} of the known-energy offer "
        f"against the bar {100 * read['bar']:.0f}%, cautious arm's mean test losses {shown}: "
        f"{read['word']}{note}"
    )


def _unit_blocks(comparison: dict, read: dict) -> list[str]:
    lines = []
    for unit, entry in comparison.items():
        lines.append("")
        if entry["status"] != "calibratable":
            lines.append(f"{unit} ({entry['status']}, {entry['counting_days']} counting days):")
            lines.append("  no numbers: the unit cannot be split")
            continue
        lines.append(f"{unit} ({entry['counting_days']} counting days):")
        for tier, block in entry["tiers"].items():
            note = " (information only, not read by the rule)" if tier != "A" else ""
            lines.append(f" {TIER_LABELS[tier]}{note}")
            lines += _block_lines(block)
            if tier == "A":
                lines.append(_verdict_line(read[unit], block["cautious_test_losses"]))
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
        f"Forecast trial energy: model {run['model']}, three arms (known energy; the project's "
        f"empirical energy bounds; the cautious bounds: top charging speed and storable energy at "
        f"level {run['rate_cap_level']:g}, needed energy as in the empirical bounds), "
        f"{run['splits']} seeded splits per unit (seeds 1 to {run['splits']}), departure level "
        f"{run['kappa']:g}, one-way re-solve rung.",
        LABEL[0].upper() + LABEL[1:] + ".",
        "Offers are kWh per counting test day, calibrated on each split's calibration days; the "
        "difference is arm 3 (cautious) minus arm 2 (empirical) on the same split (same seed, "
        "same days); a split with no safe scale counts as an offer of 0. Each arm is read at its "
        "own dial (the dial with its largest mean offer).",
        "The two tiers are separate blocks and are never added or averaged.",
        UNCERTAINTY,
        ONE_FIT,
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
    lines += _unit_blocks(document["comparison"], reading["units"])
    lines.append("")
    lines.append(f"One-line reading: {reading['one_line']}.")
    lines.append("")
    for arm in ARMS:
        found = document["arms"][arm]
        timeouts = ", ".join(f"{k} {v}" for k, v in found["timeouts"].items()) or "none"
        lines.append(f"Timeouts of the {arm} arm (recorded outcomes): {timeouts}")
        if found["failed_configurations"]:
            lines.append(f"FAILED in the {arm} arm: " + "; ".join(found["failed_configurations"]))
    return lines


# --- the command ----------------------------------------------------------------------------------


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def definitions(committed: dict, splits: int) -> dict:
    return {
        "trial": LABEL,
        "arms": committed["arm_definitions"],
        "pairing": (
            f"the three arms are measured on the same {splits} seeded splits per unit (seeds 1 to "
            f"{splits}) of the same matrix and the same forecast table, so a seed is the same "
            "calibration days and the same test days for every arm; per split the difference "
            "arm 3 (cautious) minus arm 2 (empirical) of the calibrated offer per counting test "
            "day (forecast_energy_reading, forecast_mix_pairs.paired_cell)"
        ),
        "headline": "tier A, each arm at the dial with its largest mean offer",
        "reading": {
            "rule": forecast_energy_reading.RULE,
            "no_other_sites": forecast_energy_reading.NO_OTHER_SITES,
            "bars": committed["reading"]["bars"],
            "loss_limits": committed["reading"]["loss_limits"],
        },
        "committed_settings": committed,
        "tiers": "the tiers are separate blocks and are never added or averaged",
        "uncertainty": [UNCERTAINTY, ONE_FIT],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--settings", type=Path, default=SETTINGS_FILE)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(NOT_THE_TRIAL_FLAG, dest="not_the_trial", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.settings)
        committed = load_settings(SETTINGS_FILE)
        kappa = settings["level"] if args.kappa is None else args.kappa
        splits = settings["splits"] if args.splits is None else args.splits
        name = settings["model"] if args.model is None else args.model
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
        forecast_mix.check_units(matrix.gate_units)  # before any table, truth build or model file
        # The approval is the committed file's alone: no `--settings` copy and no option grants it.
        # Before any table, truth build or model file is read.
        if not committed_approval():
            raise CouldNotRun(
                "this trial is not approved to run: the committed settings file does not say "
                "approved_to_run true (a copy given with --settings cannot grant it); a commit "
                "that sets it follows the owner's yes"
            )
        if settings["approved_to_run"] is not True:
            raise CouldNotRun(
                "this trial is not approved to run: the settings file given says "
                "approved_to_run is not true"
            )
        family = dial_choice.find_family(matrix)
        measured_matrix, _ = dial_choice.family_at(matrix, family, kappa)
        arm_runs = {}
        for arm in ARMS:
            arm_matrix = with_energy_bounds(matrix, arm)
            arm_runs[arm] = dial_choice.family_at(
                arm_matrix, dial_choice.find_family(arm_matrix), kappa
            )
        model = learn_run.load_model(name, matrix)
    except (CouldNotRun, MatrixError) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        learn_run.check_model(name, model, tuple(model.fit_units), matrix.gate_units)
    except learn_run.FitOnScored as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    needed = tuple(dict.fromkeys((*matrix.gate_units, *model.fit_units)))
    try:
        inputs = gate_run.check_inputs(args.truth_run, args.tables, needed)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        loaded = {
            unit: load_unit_days(
                args.tables,
                unit,
                rho=matrix.rho,
                no_fit=matrix.no_fit,
                small_fleet=matrix.small_fleet,
            )
            for unit in needed
        }
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    units = {unit: loaded[unit] for unit in matrix.gate_units}
    started = time.perf_counter()
    try:
        fit_frames = {u: loaded[u].cars for u in model.fit_units}
        factory, table = dial_choice.make_factory(
            name, model, measured_matrix, units, fit_frames, kappa
        )
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: the model could not be fitted or forecast: {problem}")
        return 1
    found: dict[str, Measured] = {}
    try:
        for arm in ARMS:
            arm_matrix, arm_family = arm_runs[arm]
            found[arm] = measure(arm, arm_matrix, arm_family, units, factory, splits, args.workers)
    except ValueError as problem:
        print(f"FAILED: an arm could not be scored: {problem}")
        return 1
    try:
        comparison = forecast_energy_reading.compare(
            {arm: found[arm].document for arm in ARMS}, {arm: found[arm].offers for arm in ARMS}
        )
    except ValueError as problem:
        print(f"FAILED: the arms cannot be paired: {problem}")
        return 1
    matrix_record = {"file": args.matrix.name, "sha256": forecast_mix._sha256(args.matrix)}
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    folder = Path(run_dir) if run_dir else None
    table_record = None
    if folder is not None:
        write_table(table, folder / TABLE_NAME)
        table_record = {
            "file": TABLE_NAME,
            "rows": len(table),
            "sha256": hashlib.sha256((folder / TABLE_NAME).read_bytes()).hexdigest(),
        }
    failed = []
    for arm in ARMS:
        one = found[arm].document
        one["model"] = name
        one["fit_units"] = list(model.fit_units)
        one["inputs"] = inputs
        one["matrix"] = matrix_record
        one["label"] = f"forecast trial energy, arm {arm}: {one['label']}"
        if table_record is not None:
            one["forecast_table"] = table_record
        after_sealed_run.mark(one, True)
        if args.not_the_trial:
            one["not_the_trial"] = {"overrides": changed}
            one["label"] = f"{NOT_THE_TRIAL}. {one['label']}"
        failed += [f"{arm}: {text}" for text in one["failed_configurations"]]
    units_read = forecast_energy_reading.read_units(
        comparison, bars_of(settings), limits_of(settings)
    )
    first = found[ARMS[0]].document
    ran = {
        "level": kappa,
        "rate_cap_level": CAUTIOUS_LEVEL,
        "need_level": kappa,
        "splits": splits,
        "model": name,
        "arms": list(ARMS),
        "matrix_sha256": matrix_record["sha256"],
        "settings_sha256": forecast_mix._sha256(args.settings),
    }
    reference = {
        "matrix_sha256": forecast_mix._sha256(MATRIX),
        "settings_sha256": forecast_mix._sha256(SETTINGS_FILE),
    }
    document = {
        "label": f"forecast trial energy: {LABEL}",
        "definitions": definitions(committed, splits),
        "run": {
            "kappa": kappa,
            "rate_cap_level": CAUTIOUS_LEVEL,
            "need_level": kappa,
            "splits": splits,
            "seeds": first["run"]["seeds"],
            "model": name,
            "arms": list(ARMS),
            "dials": first["run"]["dials"],
            "tiers": first["run"]["tiers"],
            "rung": first["run"]["rung"],
            "units": list(matrix.gate_units),
            "loss_limits": {"slot": crc.ALPHA_SLOT, "driver": crc.ALPHA_DRIVER},
        },
        "arms": {
            arm: {
                "energy_bounds": arm,
                "model": name,
                "fit_units": list(model.fit_units),
                "level": found[arm].document["run"]["kappa"],
                "forecast_table": table_record,
                "dial_choice": found[arm].document["dial_choice"],
                "timeouts": found[arm].document["timeouts"],
                "failed_configurations": found[arm].document["failed_configurations"],
            }
            for arm in ARMS
        },
        "reading": {
            "rule": forecast_energy_reading.RULE,
            "no_other_sites": forecast_energy_reading.NO_OTHER_SITES,
            "bars": settings["reading"]["bars"],
            "loss_limits": settings["reading"]["loss_limits"],
            "units": units_read,
            "one_line": forecast_energy_reading.one_line(units_read),
        },
        "settings_check": settings_check(
            committed, ran, reference, forecast_mix.data_options(args)
        ),
        "comparison": comparison,
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
        timing = {arm: found[arm].timing for arm in ARMS}
        timing["total_seconds"] = time.perf_counter() - started
        for arm in ARMS:
            one = found[arm].document
            results = _json(one)
            (folder / side_name(arm, "dial-choice")).write_text(results)
            offers = dial_choice.split_offers_document(
                one, found[arm].offers, side_name(arm, "dial-choice"), results
            )
            (folder / side_name(arm, "split-offers")).write_text(_json(offers))
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
