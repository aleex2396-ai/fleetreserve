"""Forecast trial `energy-2`: all three energy bounds at level 0.05 against the project's energy
bounds (judged), and the needed-energy bound alone at level 0.05 (information only), on the
development gate units (a trial made after the sealed run, `after_sealed_run`, after the first
energy trial `forecast_energy`, whose files are not changed). APPROVED TO RUN by the owner's
decision of 25 November 2026: the committed settings file says `approved_to_run: true`, and the
command ends with exit 2 if that file does not.

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.forecast_energy_2 [--workers N] [--first-run FOLDER]

The other options exist for tests and for a smaller matrix; the trial is the command run with no
option but `--workers` and `--first-run` (`--workers` changes the number of processes only;
`--first-run` names the first energy trial's recorded run folder, see below). A command that gives
`--kappa`, `--splits`, `--model`, `--settings` or `--matrix` with a value other than the committed
settings' ends with exit 2 (`overrides`), unless it is also given `--not-the-trial`, which runs it
as another run whose label and first summary line say so. `--tables`, `--truth-run` and
`--first-run` change the data read or the comparison made: they are reported, not refused.

The approval is read from the COMMITTED settings file only (`committed_approval`): a copy given by
`--settings`, with or without `--not-the-trial`, never grants it. The command ends with exit 2 when
the committed value is not true, before any table, truth build or model is read (a held-out unit
name is refused even earlier). The guard compares the values of a run with the committed file on
disk, not with git: a run starts from a clean commit, and a test pins that the committed file is the
one the code and the plan give.

Every setting is fixed in `trials-after-sealed-run/forecast-energy-2.json` (`SETTINGS_FILE`), with
its source, before any result; the command reads its level, splits, model and bars and loss limits
from there and refuses to run without it.

What it does. One model (`lgbm_tail`) is checked, fitted once on the training units and made to
forecast at the levels of the dial rows up to the run's level (`dial_choice.make_factory`), so
every arm reads the same forecast table. Five arms are then measured as `dial_choice --kappa
<level>` measures a model (`dial_choice.run`), on the same units, the same dial family at the level
and the same seeded splits, each on a matrix whose configurations have one `energy_bounds` value:

1. `known`: the car's true rate, storable energy and needed energy (only the departure forecast);
2. `empirical`: the three bounds as the model's quantiles at the level;
3. `cautious` (`forecast_empirical.CAUTIOUS`, the first trial's value, re-run): the rate and the
   storable energy at level 0.05, the needed energy as in arm 2; read by no rule here;
4. `cautious-need` (`forecast_empirical.CAUTIOUS_NEED`): the needed energy alone at level 0.05 (the
   0.95 quantile), the other two as in arm 2; information only;
5. `cautious-all` (`forecast_empirical.CAUTIOUS_ALL`): all three at level 0.05; the judged arm.

The arms differ in the energy bounds alone. Each split's scale is calibrated on that split's
calibration days alone and scored on its test days (`dial_splits`); each arm is read at its own
dial. The arms are then paired split by split and read by the rule fixed in the settings file
(`forecast_energy_2_reading`: the module text holds the rule, the bars and the one-line reading).
Arms 1 to 3 are re-run only so that every arm is paired with arms 4 and 5 split by split; the
first trial's figures stand as its result.

`--first-run FOLDER` names the run folder of the first energy trial. The command then reads that
folder's `forecast-energy.json` (no other file) and prints, for arms 1 to 3 and each judged unit
(tier A), this run's mean offer at the arm's own dial beside the first run's, and the word FAULT
with both numbers for any difference at all (exact equality of the stored numbers, no tolerance). A
fault is printed and is not chosen between; the exit status is not changed by it. Without the
option the output says that the first run was not compared.

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
- `forecast-energy-2.json` (`RESULTS_NAME`): the settings, the inputs, the five arms' headline
  figures, the paired comparisons, the reading and the first-run check, with `definitions`;
- `forecast-energy-2-summary.txt`, `forecast-energy-2-timing.json`;
- one forecast table `forecast-energy-2-forecasts.csv` (its hash is in the results), and per arm
  the full results of `dial_choice.run` (`forecast-energy-2-<arm>-dial-choice.json`) and its
  per-split offers (`forecast-energy-2-<arm>-split-offers.json`);
- `after_sealed_run.FILE_NAME`, the label file.
Every output says that it is a trial made after held-out results were seen (the key
`made_after_sealed_run` and the label of each results file, the first lines of the summary and the
label file). The label is always on. Its sentence names the UTC date on which the run started,
read from the clock once at the start of `main` (`utc_now`) and used in every output of the run, so
a run that passes midnight carries one date; the rest of the sentence is the shared one
(`after_sealed_run.sentence`).

`reading.first_run_faults` of the results file is the list of the first-run faults (empty when the
first run was compared and equal, null when not compared), and the summary's first lines say the
count of FAULT lines, "First run compared: equal" or "First run not compared". A fault changes no
figure, no comparison, no reading word and not the exit status.

Exit status: 0 every configuration of the five arms measured; 1 the model was refused or failed,
or a configuration failed; 2 could not run (a bad argument, matrix, settings file, first-run folder
or model name, not approved to run, a unit that is not a development unit, a model fitted on a
scored unit, a table missing, no verified truth build at this commit, a table differing from it).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from fleetreserve import (
    after_sealed_run,
    crc,
    dial_choice,
    forecast_energy_2_reading,
    forecast_mix,
    gate_run,
    learn_run,
)
from fleetreserve import forecast_energy as first_trial
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_empirical import CAUTIOUS_LEVEL
from fleetreserve.forecast_model import write_table
from fleetreserve.gate_config import MATRIX_ENERGY_BOUNDS, TIER_LABELS, MatrixError, load
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

MATRIX = forecast_mix.MATRIX
SETTINGS_FILE = TOP / "trials-after-sealed-run" / "forecast-energy-2.json"
REQUIRED_SETTINGS = (
    "approved_to_run",
    "level",
    "rate_cap_level",
    "need_level",
    "cautious_need_level",
    "splits",
    "model",
    "arms",
    "reading",
)
RESULTS_NAME = "forecast-energy-2.json"
SUMMARY_NAME = "forecast-energy-2-summary.txt"
TIMING_NAME = "forecast-energy-2-timing.json"
TABLE_NAME = "forecast-energy-2-forecasts.csv"
FIRST_RESULTS_NAME = first_trial.RESULTS_NAME  # the one file read of the first trial's run folder
FIRST_RUN_ARMS = forecast_energy_2_reading.ARMS[:3]  # arms 1 to 3, the arms the first trial ran
NOT_THE_TRIAL_FILE = forecast_mix.NOT_THE_TRIAL_FILE
NOT_THE_TRIAL_FLAG = forecast_mix.NOT_THE_TRIAL_FLAG
NOT_THE_TRIAL = forecast_mix.NOT_THE_TRIAL
ARMS = forecast_energy_2_reading.ARMS  # arms 1 to 5: the order the arms are measured in
LIMIT_KEYS = first_trial.LIMIT_KEYS
LABEL = (
    "a trial: all three energy bounds at level 0.05 (the top charging speed and the storable "
    "energy at the 0.05 lower quantile, the needed energy at the 0.95 upper quantile) against the "
    "project's energy bounds and against known energy, and the needed energy alone at level 0.05 "
    "as information, each arm at its own driver-risk dial, paired split by split on development "
    "units"
)
SETTINGS_TRIED = (
    "The three settings tried: cautious (arm 3, the first energy trial's value, re-run; read by no "
    "rule here): top charging speed and storable energy at the 0.05 lower quantile, needed energy "
    "as in arm 2; cautious-need (arm 4, information only): the needed energy alone at the 0.95 "
    "upper quantile, the other two as in arm 2; cautious-all (arm 5, the judged one): all three at "
    "level 0.05."
)
UNCERTAINTY = first_trial.UNCERTAINTY
ONE_FIT = (
    "The five arms read one fit of one model, so part of any difference between arms is not "
    "this model's chance; a different model or fit could give a different difference."
)
NOT_COMPARED = (
    "The first energy trial's run was not compared (no --first-run given): arms 1 to 3 of this "
    "run are not checked against its recorded figures."
)


def side_name(arm: str, kind: str) -> str:
    """The file name of `kind` (`dial-choice`, `split-offers`) for an arm."""
    return f"forecast-energy-2-{arm}-{kind}.json"


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
    return first_trial.bars_of(settings)


def limits_of(settings: dict) -> dict:
    """The loss limits of the `reading` block: `slot` and `driver`."""
    return first_trial.limits_of(settings)


def settings_check(
    committed: dict, ran: dict, committed_hashes: dict, inputs: dict | None = None
) -> dict:
    """Whether every value of the run equals the committed settings file's (module text).
    `committed`: the committed file's content, never the file a run was given by `--settings`;
    `ran`: level, rate_cap_level, need_level, cautious_need_level, splits, model, arms and the
    SHA-256 of the matrix and of the settings file as run; `committed_hashes`: the SHA-256 of the
    committed matrix and settings file under the same keys. Each difference is listed with both
    values. `inputs`: the data options (`--tables`, `--truth-run`) that differ from their defaults,
    which change the data read and not the settings; they are reported beside the check."""
    expected = {
        "level": committed["level"],
        "rate_cap_level": committed["rate_cap_level"],
        "need_level": committed["need_level"],
        "cautious_need_level": committed["cautious_need_level"],
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
    read from the committed file alone (`committed_approval`); `--first-run` is not one either."""
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
    `gate_config.MATRIX_ENERGY_BOUNDS`); an oracle partner changes with its configuration, so the
    partners stay equal in that field (`gate_config._check_partners`). Nothing else changes."""
    if value not in MATRIX_ENERGY_BOUNDS:
        raise MatrixError(f"energy_bounds must be one of {MATRIX_ENERGY_BOUNDS}, got {value!r}")
    return replace(
        matrix,
        configurations=tuple(replace(c, energy_bounds=value) for c in matrix.configurations),
    )


# --- one arm --------------------------------------------------------------------------------------


def measure(arm, matrix, family, units, factory, splits, workers):
    """The arm measured by `dial_choice.run` (`forecast_energy.measure`, unchanged)."""
    return first_trial.measure(arm, matrix, family, units, factory, splits, workers)


# --- the first run's figures ----------------------------------------------------------------------


def read_first_run(folder: Path) -> dict:
    """The recorded output of the first energy trial in the run folder `folder` (module text): the
    file `FIRST_RESULTS_NAME` and no other is read. Returns its SHA-256 and, per unit, the mean
    offers of tier A at each arm's own dial (`comparison`, `mean_offer_kwh_per_test_day`; None for
    a unit or an arm without a figure). `CouldNotRun` when the file cannot be read or has no
    `comparison`."""
    path = Path(folder) / FIRST_RESULTS_NAME
    try:
        gate_run.check_files([path])
        raw = path.read_bytes()
        document = json.loads(raw)
        comparison = document["comparison"]
        if not isinstance(comparison, dict):
            raise ValueError("comparison is not a table of units")
    except (OSError, ValueError, KeyError, TypeError) as problem:
        raise CouldNotRun(
            f"the first trial's results {FIRST_RESULTS_NAME} cannot be read from the folder "
            f"{Path(folder).name}: {problem}"
        ) from problem
    means = {}
    for unit, entry in comparison.items():
        tiers = entry.get("tiers") if isinstance(entry, dict) else None
        block = tiers.get("A") if isinstance(tiers, dict) else None
        found = block.get("mean_offer_kwh_per_test_day") if isinstance(block, dict) else None
        means[unit] = found if isinstance(found, dict) else None
    return {"sha256": hashlib.sha256(raw).hexdigest(), "means": means}


def first_run_check(first: dict | None, comparison: dict, units) -> dict:
    """What this run's arms 1 to 3 gave beside the first trial's recorded figures (module text):
    for each of `units` (the judged units) and each arm of `FIRST_RUN_ARMS`, this run's mean tier A
    offer at the arm's own dial and the first run's, equal or not by exact equality of the stored
    numbers (no tolerance; a missing figure on one side only is a difference). `first` is
    `read_first_run`'s answer, or None when the option was not given."""
    if first is None:
        return {"compared": False, "note": NOT_COMPARED, "figures": [], "faults": []}
    figures, faults = [], []
    for unit in units:
        block = comparison.get(unit, {}).get("tiers", {}).get("A")
        this = {} if block is None else block["mean_offer_kwh_per_test_day"]
        then = first["means"].get(unit) or {}
        for arm in FIRST_RUN_ARMS:
            ours, theirs = this.get(arm), then.get(arm)
            equal = ours == theirs
            figures.append({"unit": unit, "arm": arm, "this_run": ours, "first_run": theirs})
            if not equal:
                faults.append(
                    f"FAULT: {unit}, arm {arm}: this run {ours!r} kWh, the first run {theirs!r} kWh"
                )
    return {
        "compared": True,
        "first_results_sha256": first["sha256"],
        "figures": figures,
        "faults": faults,
        "note": (
            "a difference of any size is a FAULT, printed and not chosen between; the reading of "
            "arm 5 against arm 2 is given from this run's own arms either way"
        ),
    }


# --- the summary ----------------------------------------------------------------------------------


def _number(value, digits: int = 2) -> str:
    return first_trial._number(value, digits)


def _percent(value) -> str:
    return first_trial._percent(value)


def _dial(value) -> str:
    return first_trial._dial(value)


def _losses(losses: dict | None) -> str:
    if losses is None:
        return "none"
    return f"slot {_number(losses['slot'], 3)}, driver {_number(losses['driver'], 3)}"


def _comparison_lines(block: dict, which: str, heading: str) -> list[str]:
    """The lines of one paired comparison (`judged` or `information`); `heading` says what it is."""
    found = block[which]
    trial = forecast_energy_2_reading.COMPARISONS[which]
    main = forecast_energy_2_reading.MAIN_ARM
    if found is None:
        return [f"  {heading}: no figure (an arm it needs has no dial with a mean offer)"]
    cell = found["cell"]
    return [
        f"  {heading}: {trial} minus {main}: "
        f"{_number(cell['mean_difference_kwh_per_test_day'])} (standard error "
        f"{_number(cell['standard_error_of_the_difference'])}; p10 "
        f"{_number(cell['difference_p10'])}, p50 {_number(cell['difference_p50'])}, p90 "
        f"{_number(cell['difference_p90'])}); {trial} offers more in "
        f"{cell['splits_trial_offers_more']}, the same in {cell['splits_equal']}, less in "
        f"{cell['splits_trial_offers_less']} of {cell['splits']} splits; the difference "
        f"{_percent(found['difference_share_of_known_offer'])} of the known-energy offer, "
        f"{_percent(found['recovered_share_of_the_gap'])} of the gap between known and empirical; "
        f"{trial}'s mean test losses {_losses(found['test_losses'])}"
    ]


def _block_lines(block: dict, tier: str) -> list[str]:
    picks = block["picked_dial"]
    dials = ", ".join(f"{arm} {_dial(picks[arm])}" for arm in ARMS)
    means = block["mean_offer_kwh_per_test_day"]
    shares = block["share_of_known_offer"]
    lines = [
        f"  own dials ({dials}); mean offer, kWh per counting test day: "
        + ", ".join(f"{arm} {_number(means[arm])}" for arm in ARMS),
        "  as a share of the known-energy offer: "
        + ", ".join(f"{arm} {_percent(shares[arm])}" for arm in ARMS[1:]),
    ]
    judged = (
        "JUDGED comparison" if tier == "A" else "arm 5 against arm 2 (tier B, information only)"
    )
    lines += _comparison_lines(block, "judged", judged)
    lines += _comparison_lines(
        block, "information", f"({forecast_energy_2_reading.INFORMATION_LABEL}) arm 4 against arm 2"
    )
    return lines


def _verdict_line(read: dict, losses: dict | None) -> str:
    if read["word"] is None:
        return f"  Reading (tier A): {read['note']}; figures printed only"
    note = "" if read["note"] is None else f" ({read['note']})"
    return (
        f"  Reading (tier A, arm 5 minus arm 2): difference {_percent(read['share'])} of the "
        f"known-energy offer against the bar {100 * read['bar']:.0f}%, cautious-all arm's mean "
        f"test losses {_losses(losses)}: {read['word']}{note}"
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
            lines += _block_lines(block, tier)
            if tier == "A":
                judged = block["judged"]
                lines.append(
                    _verdict_line(read[unit], None if judged is None else judged["test_losses"])
                )
    return lines


def _first_run_lines(check: dict) -> list[str]:
    if not check["compared"]:
        return [check["note"]]
    lines = [
        "Arms 1 to 3 of this run against the first energy trial's recorded figures (tier A, mean "
        "offer at each arm's own dial, exact equality):"
    ]
    for item in check["figures"]:
        lines.append(
            f"  {item['unit']}, arm {item['arm']}: this run {item['this_run']!r}, the first run "
            f"{item['first_run']!r}"
        )
    lines += check["faults"] or ["  no difference: every figure equals the first run's."]
    return lines


def first_run_faults(check: dict) -> list[str] | None:
    """The faults of `first_run_check`'s answer `check` as the reading holds them: the list of its
    FAULT lines (empty when the first run was compared and equal), or None when the first run was
    not compared. Only copies what `first_run_check` found."""
    return list(check["faults"]) if check["compared"] else None


def first_run_status_line(faults: list[str] | None) -> str:
    """The one line of the summary's first lines about the first run: the count of FAULT lines when
    above 0, "First run compared: equal", or "First run not compared"."""
    if faults is None:
        return "First run not compared"
    if not faults:
        return "First run compared: equal"
    return (
        f"First run compared: {len(faults)} FAULT line(s) (listed below and in `first_run_faults` "
        "of `reading` in the results file)"
    )


def summary_lines(document: dict) -> list[str]:
    run = document["run"]
    reading = document["reading"]
    lines = []
    if "not_the_trial" in document:
        asked = ", ".join(document["not_the_trial"]["overrides"]) or "no option changing a setting"
        lines.append(f"{NOT_THE_TRIAL}; {asked}")
    lines += [
        *after_sealed_run.summary_head(document),
        first_run_status_line(reading["first_run_faults"]),
        f"Forecast trial energy-2: model {run['model']}, five arms (known energy; the project's "
        f"empirical energy bounds; cautious; cautious-need; cautious-all), {run['splits']} seeded "
        f"splits per unit (seeds 1 to {run['splits']}), departure level {run['kappa']:g}, one-way "
        "re-solve rung.",
        LABEL[0].upper() + LABEL[1:] + ".",
        SETTINGS_TRIED,
        "Offers are kWh per counting test day, calibrated on each split's calibration days; each "
        "difference is a trial arm minus arm 2 (empirical) on the same split (same seed, same "
        "days); a split with no safe scale counts as an offer of 0. Each arm is read at its own "
        "dial (the dial with its largest mean offer).",
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
    lines += _first_run_lines(document["first_run_check"])
    lines += _unit_blocks(document["comparison"], reading["units"])
    lines.append("")
    not_judged = [
        f"{unit}: {found['note']}"
        for unit, found in reading["units"].items()
        if found["word"] is None
    ]
    lines += [f"Not judged: {text}" for text in not_judged]
    lines.append(f"One-line reading: {reading['one_line']}.")
    faults = document["first_run_check"]["faults"]
    if faults:
        lines.append(
            f"{len(faults)} FAULT line(s) against the first run's figures above; the reading is "
            "from this run's own arms."
        )
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
        "settings_tried": SETTINGS_TRIED,
        "pairing": (
            f"the five arms are measured on the same {splits} seeded splits per unit (seeds 1 to "
            f"{splits}) of the same matrix and the same forecast table, so a seed is the same "
            "calibration days and the same test days for every arm; per split the difference "
            "arm 5 (cautious-all) minus arm 2 (empirical) of the calibrated offer per counting "
            "test day is the judged comparison, and arm 4 (cautious-need) minus arm 2 the "
            "information comparison (forecast_energy_2_reading, forecast_mix_pairs.paired_cell)"
        ),
        "headline": "tier A, each arm at the dial with its largest mean offer",
        "reading": {
            "rule": forecast_energy_2_reading.RULE,
            "no_other_sites": forecast_energy_2_reading.NO_OTHER_SITES,
            "bars": committed["reading"]["bars"],
            "loss_limits": committed["reading"]["loss_limits"],
        },
        "first_run_comparison": committed["first_run_comparison"],
        "committed_settings": committed,
        "tiers": "the tiers are separate blocks and are never added or averaged",
        "uncertainty": [UNCERTAINTY, ONE_FIT],
    }


def utc_now() -> datetime:
    """The clock: the current time in UTC (tests replace it)."""
    return datetime.now(UTC)


def main(argv: list[str] | None = None) -> int:
    day = after_sealed_run.day_text(utc_now())  # once; the one date of every output of the run
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--settings", type=Path, default=SETTINGS_FILE)
    parser.add_argument("--first-run", type=Path, default=None)
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
        gate_run.argument_paths(args, (*gate_run.PATH_ARGUMENTS, "first_run"))
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
        first = None if args.first_run is None else read_first_run(args.first_run)
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
    found: dict[str, first_trial.Measured] = {}
    try:
        for arm in ARMS:
            arm_matrix, arm_family = arm_runs[arm]
            found[arm] = measure(arm, arm_matrix, arm_family, units, factory, splits, args.workers)
    except ValueError as problem:
        print(f"FAILED: an arm could not be scored: {problem}")
        return 1
    try:
        comparison = forecast_energy_2_reading.compare(
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
        one["label"] = f"forecast trial energy-2, arm {arm}: {one['label']}"
        if table_record is not None:
            one["forecast_table"] = table_record
        after_sealed_run.mark(one, True, day)
        if args.not_the_trial:
            one["not_the_trial"] = {"overrides": changed}
            one["label"] = f"{NOT_THE_TRIAL}. {one['label']}"
        failed += [f"{arm}: {text}" for text in one["failed_configurations"]]
    bars = bars_of(settings)
    units_read = forecast_energy_2_reading.read_units(comparison, bars, limits_of(settings))
    first_doc = found[ARMS[0]].document
    first_check = first_run_check(first, comparison, list(bars))
    ran = {
        "level": kappa,
        "rate_cap_level": CAUTIOUS_LEVEL,
        "need_level": kappa,
        "cautious_need_level": CAUTIOUS_LEVEL,
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
        "label": f"forecast trial energy-2: {LABEL}",
        "definitions": definitions(committed, splits),
        "run": {
            "kappa": kappa,
            "rate_cap_level": CAUTIOUS_LEVEL,
            "need_level": kappa,
            "cautious_need_level": CAUTIOUS_LEVEL,
            "splits": splits,
            "seeds": first_doc["run"]["seeds"],
            "model": name,
            "arms": list(ARMS),
            "dials": first_doc["run"]["dials"],
            "tiers": first_doc["run"]["tiers"],
            "rung": first_doc["run"]["rung"],
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
            "rule": forecast_energy_2_reading.RULE,
            "no_other_sites": forecast_energy_2_reading.NO_OTHER_SITES,
            "bars": settings["reading"]["bars"],
            "loss_limits": settings["reading"]["loss_limits"],
            "units": units_read,
            "one_line": forecast_energy_2_reading.one_line(units_read),
            "first_run_faults": first_run_faults(first_check),
        },
        "first_run_check": first_check,
        "settings_check": settings_check(
            committed, ran, reference, forecast_mix.data_options(args)
        ),
        "comparison": comparison,
        "failed_configurations": sorted(failed),
        "inputs": inputs,
        "matrix": matrix_record,
    }
    after_sealed_run.mark(document, True, day)
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
        after_sealed_run.write_note(folder, True, day)
        if args.not_the_trial:
            with (folder / after_sealed_run.FILE_NAME).open("a") as label_file:
                label_file.write(f"{NOT_THE_TRIAL}\n")
            note = "\n".join([NOT_THE_TRIAL, *changed])
            (folder / NOT_THE_TRIAL_FILE).write_text(note + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
