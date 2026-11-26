"""Forecast trial `noset`: the neural departure model without its set encoder against three
others (the same model with its set encoder, the neural model the project reports and the main
model), on the development gate units, each with its own driver-risk dial and the neural models on
several seeds (a trial made after the sealed run, `after_sealed_run`). APPROVED TO RUN by the
owner's decision of 24 November 2026: the committed settings file says `approved_to_run: true`,
and the command ends with exit 2 if that file does not say so.

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.forecast_noset [--workers N]

The other options exist for tests and for a smaller matrix; the trial is the command run with NO
option (`--workers` changes the number of processes only). A command that gives `--kappa`,
`--splits`, `--seeds`, `--model-tail`, `--model-tuned`, `--model-inputs`, `--model-noset`,
`--settings` or `--matrix` with a value other than the committed settings' ends with exit 2
(`overrides`), unless it is also given `--not-the-trial`, which runs it as another run whose label
and first summary line say so. `--tables` and `--truth-run` change the tables read: they are
reported (`settings_check`, `inputs_that_differ_from_the_defaults`), not refused.

The approval is read from the COMMITTED settings file only (`committed_approval`: the file of the
checkout this module is imported from, `SETTINGS_FILE`, as `settings_check` finds it). A copy of the
settings given by `--settings`, with or without `--not-the-trial`, never grants it: the command
ends with exit 2 when the committed value is not true, before any table, truth build or model is
read (a held-out unit name is refused even earlier). A copy that says false refuses as well.

Every setting is fixed in `trials-after-sealed-run/forecast-noset.json` (`SETTINGS_FILE`), with its
source, before any result.

What it does. Ten runs, one after the other, on the same matrix, the same units, the same dial
family at the level and the same seeded splits: `lgbm_tail`; `hazard_tuned`,
`hazard_tuned_inputs` and `hazard_tuned_no_set` each at each seed of `neural_seeds`. A run checks
its model (`learn_run.check_model`), fits it on the training units (`dial_choice.make_factory`, the
neural model with its seed set by `dial_choice.with_fit_seed`) and measures it with
`site_dial_whatif.run`: every dial of the family at every scale of the grid, then, per split, the
offer of every dial calibrated on all the calibration days (`by_dial`) and the offer of the dial
chosen from calibration days alone (variant S1 of that module). The two lines of figures come from
that one measurement, so the second needs no further fit or evaluation:

- the JUDGED line (`forecast_noset_pairs.PLAN_RULE`): each run at the dial the project's development
  rule picks for it (plan section 6, `dial_splits.pick_dial` through `site_dial_summary.by_dial`:
  the largest mean offer over the run's own 200 splits at the unit and tier). That rule reads the
  test-day offers of the development units, for every arm and seed alike, as every development
  figure of the project does; the bars of the reading apply to this line;
- the CHECK line (`forecast_noset_pairs.CHECK`): the arm S1, the dial chosen on half of each
  split's calibration days alone. No bar and no verdict: only the sign and size of each difference;
  a check difference of the opposite sign to a gain or a loss of the judged line at a judged unit
  makes the one-line reading "mixed"; a check difference of exactly 0 or an undefined one is silent
  (the judged word stands, and the silent units are counted per comparison).

The runs are then paired split by split (`forecast_noset_pairs`) and read by the rule fixed in the
settings file (`forecast_noset_reading`).

Why no test day's outcome reaches its own forecast or calibration:

- The forecast. Every model is fitted on the cars tables of its `fit_units` only, which
  `learn_run.check_model` limits to the training units and refuses if one is scored
  (`FitOnScored`). The frame a model predicts from is `learn_run.forecast_frame`: rows of
  counting days with keys and features known at the decision time, no outcome and no label.
- The calibration. On both lines a split's scale is calibrated on calibration days only
  (`dial_splits.score_split`) and the test days are scored at that scale. On the check line the
  dial is also chosen from calibration days only (`site_dial_whatif.split_arms`: `calibrated_offer`
  over `pick` only, then `choose`; the test days are not in the calibration days,
  `dial_choice.split_indices`). On the judged line the dial is chosen by the plan's rule from the
  mean offers over all the splits, which include the test-day offers: that line is the project's
  development-data rule and is not free of test-day information; the check line is.

The headline, fixed before any result: tier A, per unit, the judged line, each arm's offer per
counting test day, paired by split; the two tiers are never pooled or averaged. The reading is
`forecast_noset_reading`.

Units. The command refuses, before it opens any table, truth build or model file, a matrix whose
gate units are not all development units of the truth build (`forecast_mix.check_units`, exit 2).
It names no loader of the sealed data and reads no table of it.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set, the summary also on standard output:
- `forecast-noset.json` (`RESULTS_NAME`): the settings, the inputs, the runs, the comparison and
  the reading, with `definitions`;
- `forecast-noset-summary.txt`, `forecast-noset-timing.json`;
- per run (`tail`, `tuned-seed<N>`, `inputs-seed<N>`, `noset-seed<N>`): the full results of
  `site_dial_whatif.run` (`forecast-noset-<run>-dial-whatif.json`) and its forecast table
  (`forecast-noset-<run>-forecasts.csv`, its hash is in the results);
- `after_sealed_run.FILE_NAME`, the label file.
Every output says that it is a trial made after held-out results were seen (the key
`made_after_sealed_run` and the label of each results file, the first lines of the summary and the
label file). The label is always on.

Exit status: 0 every run measured; 1 a model was refused or failed, or a unit failed in a run (the
reading then says NOT_READ where it needs it); 2 could not run (a bad argument, matrix, settings
file or model name, not approved to run, a unit that is not a development unit, a model fitted on
a scored unit, a table missing, no verified truth build at this commit, a table differing from it).
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
    forecast_mix,
    forecast_noset_pairs,
    forecast_noset_reading,
    gate_run,
    learn_run,
    site_dial_whatif,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import write_table
from fleetreserve.gate_config import TIER_LABELS, MatrixError, load
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

MATRIX = forecast_mix.MATRIX
SETTINGS_FILE = TOP / "trials-after-sealed-run" / "forecast-noset.json"
REQUIRED_SETTINGS = (
    "approved_to_run",
    "level",
    "splits",
    "neural_seeds",
    "arms",
    "reading",
)
RESULTS_NAME = "forecast-noset.json"
SUMMARY_NAME = "forecast-noset-summary.txt"
TIMING_NAME = "forecast-noset-timing.json"
NOT_THE_TRIAL_FILE = forecast_mix.NOT_THE_TRIAL_FILE
NOT_THE_TRIAL_FLAG = forecast_mix.NOT_THE_TRIAL_FLAG
NOT_THE_TRIAL = forecast_mix.NOT_THE_TRIAL
ARMS = forecast_noset_pairs.ARMS  # tail, tuned, inputs, noset: the order the runs are made in
NEURAL_ARMS = ("tuned", "inputs", "noset")
LABEL = (
    "a trial: the neural departure model without its set encoder against the same model with it, "
    "against the neural model the project reports and against the main model, each with its own "
    "driver-risk dial, the neural models on several seeds, paired split by split on development "
    "units"
)
UNCERTAINTY = (
    "The seeded splits share days, so the standard error over the splits understates the "
    "uncertainty of the whole experiment."
)
FEW_SEEDS = (
    "Each neural model has three seeds, which measure the spread that a seed gives only roughly; "
    "the bars are the neural model's seed spread from an earlier look, an analogy and not this "
    "trial's own measure; the spread of this trial's seeds (the least and the greatest figure of "
    "each arm and unit) is printed beside each arm."
)
PLAN_RULE_NOTE = (
    "Judged line: each run at the dial that the project's development rule picks for it (the "
    "largest mean offer over its own splits), which reads the test-day offers of the development "
    "units, for every arm and seed alike, as every development figure of the project does."
)
CHECK_NOTE = (
    "Check line: each run at the dial chosen on half of each split's calibration days alone, its "
    "scale calibrated on the other half (variant S1 of the site-dial what-if); its offers may be "
    "lower than the judged line's and are not comparable to earlier figures; the bars do not "
    "apply to it and no verdict is formed on it, only the sign and size of each difference."
)


@dataclass(frozen=True)
class Run:
    """One of the runs: its name, its arm, its model and its seed (None: the model's own)."""

    name: str
    arm: str
    model: str
    seed: int | None


@dataclass
class Measured:
    """What one run gave."""

    run: Run
    document: dict
    timing: dict
    table: object


def side_name(run: str, kind: str) -> str:
    """The file name of `kind` (`dial-whatif`, `forecasts`) for the run `run`."""
    return f"forecast-noset-{run}-{kind}." + ("csv" if kind == "forecasts" else "json")


def plan_runs(models: dict, seeds) -> list[Run]:
    """The runs of the trial (module text): `tail` once, then each neural arm at each seed;
    `models`: arm -> model name."""
    runs = [Run("tail", "tail", models["tail"], None)]
    for arm in NEURAL_ARMS:
        runs += [Run(f"{arm}-seed{seed}", arm, models[arm], seed) for seed in seeds]
    return runs


def seeded_model(model, name: str, seed: int):
    """`model` with the seed of its fit settings replaced (`dial_choice.with_fit_seed`)."""
    return dial_choice.with_fit_seed(model, name, seed)


# --- the settings and the units -------------------------------------------------------------------


def parse_seeds(text: str) -> tuple[int, ...]:
    """The seeds of `--seeds` in increasing order; `MatrixError` unless `text` lists one or more
    distinct whole numbers of 0 or more, separated by commas."""
    parts = [p.strip() for p in text.split(",")]
    if not all(p.isascii() and p.isdecimal() for p in parts):
        raise MatrixError(
            f"--seeds takes whole numbers of 0 or more separated by commas, got {text!r}"
        )
    seeds = [int(p) for p in parts]
    if len(set(seeds)) != len(seeds):
        raise MatrixError(f"--seeds lists a seed twice: {text!r}")
    return tuple(sorted(seeds))


def load_settings(path: Path = SETTINGS_FILE) -> dict:
    """The settings of the trial in the file `path`; raises `CouldNotRun` when the file is missing,
    is not JSON, lacks a key of `REQUIRED_SETTINGS`, names no model of an arm, lists seeds that are
    not distinct whole numbers or has no bars in its `reading`."""
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
    if not isinstance(found["arms"], dict) or set(found["arms"]) != set(ARMS):
        raise CouldNotRun(
            f"the settings file {Path(path).name} must name a model for each of {ARMS}"
        )
    seeds = found["neural_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise CouldNotRun(f"the settings file {Path(path).name} lists seeds that are not distinct")
    if not isinstance(found["reading"], dict) or not found["reading"].get("bars"):
        raise CouldNotRun(f"the settings file {Path(path).name} has no bars in its reading")
    return found


def committed_approval() -> bool:
    """Whether the COMMITTED settings file (`SETTINGS_FILE`, the file of the checkout this module
    is imported from) says `approved_to_run: true` (module text). It reads that file and no other,
    so no `--settings` copy and no other option can grant the approval; `CouldNotRun` when the file
    cannot be read. Tests approve by replacing this function."""
    return load_settings(SETTINGS_FILE)["approved_to_run"] is True


def bars_of(settings: dict) -> dict:
    """Judged unit -> bar (a share of the comparison arm's offer) from the `reading` block."""
    return {unit: float(found["share"]) for unit, found in settings["reading"]["bars"].items()}


def settings_check(
    committed: dict, ran: dict, committed_hashes: dict, inputs: dict | None = None
) -> dict:
    """Whether every value of the run equals the committed settings file's (module text).
    `committed`: the committed file's content, never the file a run was given by `--settings`;
    `ran`: level, splits, neural_seeds, the four models' names under `models` and the SHA-256 of
    the matrix and of the settings file as run; `committed_hashes`: the SHA-256 of the committed
    matrix and settings file. Each difference is listed with both values. `inputs`: the data
    options (`--tables`, `--truth-run`) that differ from their defaults, reported beside the
    check."""
    expected = {
        "level": committed["level"],
        "splits": committed["splits"],
        "neural_seeds": list(committed["neural_seeds"]),
        "models": dict(committed["arms"]),
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
    `--kappa`, `--splits`, `--seeds` or a model option, or a settings file or a matrix that is not
    the committed file (by content). Nothing is read but those two files. The approval is not an
    override: it is read from the committed file alone (`committed_approval`)."""
    found = []
    for option, attribute, value in (
        ("--kappa", "kappa", committed["level"]),
        ("--splits", "splits", committed["splits"]),
        ("--model-tail", "model_tail", committed["arms"]["tail"]),
        ("--model-tuned", "model_tuned", committed["arms"]["tuned"]),
        ("--model-inputs", "model_inputs", committed["arms"]["inputs"]),
        ("--model-noset", "model_noset", committed["arms"]["noset"]),
    ):
        given = getattr(args, attribute)
        if given is not None and given != value:
            found.append(f"{option} {given} (committed: {value})")
    committed_seeds = tuple(sorted(committed["neural_seeds"]))
    if args.seeds is not None and parse_seeds(args.seeds) != committed_seeds:
        found.append(f"--seeds {args.seeds} (committed: {committed['neural_seeds']})")
    if forecast_mix._sha256(args.settings) != forecast_mix._sha256(SETTINGS_FILE):
        found.append("--settings (not the committed settings file)")
    if forecast_mix._sha256(args.matrix) != forecast_mix._sha256(MATRIX):
        found.append("--matrix (not the committed run matrix)")
    return found


# --- one run --------------------------------------------------------------------------------------


def measure(run: Run, model, matrix, family, fixed, units, loaded, splits, workers) -> Measured:
    """The model fitted on its fit units and measured by `site_dial_whatif.run` (module text);
    raises `learn_run.Refused` and `ValueError` as `dial_choice.make_factory` does. `fixed`: gate
    unit -> the fallback dial of S1 (`fallback_dials`)."""
    fit_frames = {u: loaded[u].cars for u in model.fit_units}
    factory, table = dial_choice.make_factory(
        run.model, model, matrix, units, fit_frames, family[0].kappa
    )
    document, timing = site_dial_whatif.run(
        matrix, units, factory, family, fixed, splits=splits, workers=workers
    )
    document["model"] = model.name
    document["fit_units"] = list(model.fit_units)
    document["fit_seed"] = run.seed
    document["run_name"] = run.name
    return Measured(run, document, timing, table)


def fallback_dials(matrix, family) -> dict:
    """Per gate unit the smallest dial of the family: the fixed dial that `site_dial_whatif`
    needs, which S1 takes only when no dial has an offer above 0 on the choosing half and which
    breaks a tie to the smaller dial in any case. It reads no outcome."""
    smallest = min(c.driver_risk for c in family[1])
    return {unit: smallest for unit in matrix.gate_units}


# --- the summary ----------------------------------------------------------------------------------


def _num(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _signed(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def _cell_line(label: str, cell: dict | None) -> str:
    if cell is None:
        return f"  {label}: no figure (an arm failed or the unit could not be split)"
    return (
        f"  {label}: comparison arm {_num(cell['mean_main_offer_kwh_per_test_day'])}, "
        f"no-encoder arm {_num(cell['mean_trial_offer_kwh_per_test_day'])} kWh; difference "
        f"{_num(cell['mean_difference_kwh_per_test_day'])} (standard error "
        f"{_num(cell['standard_error_of_the_difference'])}; p10 {_num(cell['difference_p10'])}, "
        f"p50 {_num(cell['difference_p50'])}, p90 {_num(cell['difference_p90'])}); no-encoder "
        f"offers more in {cell['splits_trial_offers_more']}, the same in {cell['splits_equal']}, "
        f"less in {cell['splits_trial_offers_less']} of {cell['splits']} splits; ratio of means "
        f"{_num(cell['ratio_of_means_trial_to_main'], 3)}"
    )


LABELS = {
    "against_tuned": (
        "no-encoder model against hazard_tuned (the same model with its set encoder; one "
        "difference; the judged comparison)"
    ),
    "against_inputs": (
        "no-encoder model against hazard_tuned_inputs (the neural model the project reports; "
        "two differences: the set encoder and extra input columns)"
    ),
    "against_tail": (
        "no-encoder model against lgbm_tail (the main model; a different kind of model)"
    ),
    "inputs_against_tail": "information: hazard_tuned_inputs against lgbm_tail (not read)",
}
LINE_HEADS = {
    forecast_noset_pairs.PLAN_RULE: (
        "Judged line (the plan's development rule: each run at the dial with its largest mean "
        "offer over the splits)"
    ),
    forecast_noset_pairs.CHECK: (
        "Check line (the dial chosen from calibration days alone; no verdict, sign and size only)"
    ),
}


def _arm_line(name: str, found: dict | None) -> str:
    if found is None:
        return f"  {name}: no figure (the arm failed or the unit could not be split)"
    seeds = ", ".join(_num(v) for v in found["seed_figures"])
    text = (
        f"  {name}: mean offer {_num(found['mean_offer'])} kWh per counting test day; "
        f"{'seed figures' if found['seeds'] > 1 else 'figure'} {seeds}"
    )
    if found["seeds"] > 1:
        text += f" (least {_num(found['seed_min'])}, greatest {_num(found['seed_max'])})"
    if "dials_picked" in found:
        text += "; dial picked " + ", ".join(
            "none" if d is None else f"{d:g}" for d in found["dials_picked"]
        )
    return text


def _verdict_lines(unit: str, reading: dict) -> list[str]:
    lines = []
    for which in forecast_noset_reading.COMPARISONS:
        read = reading["comparisons"][which]["units"].get(unit)
        if read is None:
            continue
        share = "none" if read["share"] is None else f"{100 * read['share']:.1f}%"
        lines.append(
            f"  Reading ({LABELS[which]}): share {share} against the bar "
            f"{100 * read['bar']:.0f}%: {read['word']}"
            + (
                f" (by the bar alone: {read['word_by_the_bar_alone']})"
                if read["word"] != read["word_by_the_bar_alone"]
                else ""
            )
        )
    return lines


def _check_lines(unit: str, reading: dict) -> list[str]:
    """The sign and size of each judged comparison on the check line, with the sentence of
    `forecast_noset_reading.check_note` beside it (no verdict)."""
    lines = []
    for which in forecast_noset_reading.COMPARISONS:
        check = reading["comparisons"][which]["check"].get(unit)
        read = reading["comparisons"][which]["units"].get(unit)
        if check is None or read is None:
            continue
        share = "none" if check["share"] is None else f"{100 * check['share']:+.1f}%"
        agree = forecast_noset_reading.check_note(read, check)
        lines.append(
            f"  Check ({LABELS[which]}): difference {_signed(check['difference'])} kWh, share "
            f"{share}; {agree}"
        )
    return lines


def _unit_blocks(comparison: dict, reading: dict) -> list[str]:
    lines = []
    for unit, entry in comparison.items():
        lines.append("")
        if entry["status"] != "calibratable":
            lines.append(f"{unit} ({entry['status']}, {entry['counting_days']} counting days):")
            lines.append("  no numbers: the unit cannot be split")
            continue
        lines.append(f"{unit} ({entry['counting_days']} counting days):")
        for tier, tier_block in entry["tiers"].items():
            note = " (information only, not read by the rule)" if tier != "A" else ""
            lines.append(f" {TIER_LABELS[tier]}{note}")
            for line in forecast_noset_pairs.LINES:
                block = tier_block[line]
                lines.append(f" {LINE_HEADS[line]}:")
                for arm in ARMS:
                    lines.append(_arm_line(arm, block["arms"][arm]))
                for which, label in LABELS.items():
                    lines.append(_cell_line(label, block[which]))
                if tier == "A" and line == forecast_noset_pairs.PLAN_RULE:
                    lines += _verdict_lines(unit, reading)
                if tier == "A" and line == forecast_noset_pairs.CHECK:
                    lines += _check_lines(unit, reading)
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
        f"Forecast trial noset: {run['models']['noset']} against {run['models']['tuned']}, "
        f"{run['models']['inputs']} and {run['models']['tail']}; the neural models at seeds "
        f"{', '.join(map(str, run['seeds']))}; {run['splits']} seeded splits per unit, kappa "
        f"{run['kappa']:g}, one-way re-solve rung.",
        LABEL[0].upper() + LABEL[1:] + ".",
        "Offers are kWh per counting test day; a split with no safe scale counts as an offer of "
        "0. The difference is the no-encoder arm minus the comparison arm on the same split; a "
        "neural arm enters with the mean of its seeds, split by split.",
        PLAN_RULE_NOTE,
        CHECK_NOTE,
        "The two tiers are separate blocks and are never added or averaged.",
        UNCERTAINTY,
        FEW_SEEDS,
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
    lines += _unit_blocks(document["comparison"], reading)
    lines.append("")
    for which in forecast_noset_reading.COMPARISONS:
        one = reading["comparisons"][which]["one_line"]
        judged = " (the judged line)" if which == forecast_noset_reading.JUDGED else ""
        lines.append(f"One-line reading{judged}, {LABELS[which]}: {one}.")
        silent = reading["comparisons"][which]["silent_count"]
        total = len(reading["comparisons"][which]["units"])
        lines.append(
            f"  The check line was silent at {silent} of {total} judged units"
            + (" (the reading rests on the judged line alone there)." if silent else ".")
        )
    lines.append(f"Overall reading: {reading['overall']}.")
    lines.append("")
    for name, one in document["runs"].items():
        lines.append(
            f"Timeouts of the run {name} (recorded outcomes): "
            + (", ".join(f"{k} {v}" for k, v in one["timeouts"].items() if v) or "none")
        )
    for failed in document["failed_configurations"]:
        lines.append(f"FAILED: {failed}")
    return lines


# --- the command ----------------------------------------------------------------------------------


def definitions(committed: dict) -> dict:
    return {
        "trial": LABEL,
        "question": committed["question"],
        "pairing": (
            "every run is measured on the same seeded splits per unit (seeds 1 to the number of "
            "splits) of the same matrix, so a split is the same calibration days and the same "
            "test days for every run; per split the difference of two arms' offers per counting "
            "test day on each line (forecast_noset_pairs); a neural arm enters with the mean of "
            "its seeds' offers on that split"
        ),
        "dial_rules": committed["dial_rules"],
        "fallback_dial": committed["fallback_dial"],
        "reading": {
            "rule": forecast_noset_reading.RULE,
            "no_other_sites": forecast_noset_reading.NO_OTHER_SITES,
            "bars": committed["reading"]["bars"],
        },
        "committed_settings": committed,
        "tiers": "the tiers are separate blocks and are never added or averaged",
        "uncertainty": [UNCERTAINTY, FEW_SEEDS, PLAN_RULE_NOTE, CHECK_NOTE],
    }


def _json(value) -> str:
    return forecast_mix._json(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--settings", type=Path, default=SETTINGS_FILE)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--model-tail", default=None)
    parser.add_argument("--model-tuned", default=None)
    parser.add_argument("--model-inputs", default=None)
    parser.add_argument("--model-noset", default=None)
    parser.add_argument(NOT_THE_TRIAL_FLAG, dest="not_the_trial", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.settings)
        committed = load_settings(SETTINGS_FILE)
        kappa = settings["level"] if args.kappa is None else args.kappa
        splits = settings["splits"] if args.splits is None else args.splits
        seeds = (
            tuple(sorted(settings["neural_seeds"]))
            if args.seeds is None
            else parse_seeds(args.seeds)
        )
        names = {
            "tail": settings["arms"]["tail"] if args.model_tail is None else args.model_tail,
            "tuned": settings["arms"]["tuned"] if args.model_tuned is None else args.model_tuned,
            "inputs": (
                settings["arms"]["inputs"] if args.model_inputs is None else args.model_inputs
            ),
            "noset": settings["arms"]["noset"] if args.model_noset is None else args.model_noset,
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
        run_matrix, run_family = dial_choice.family_at(matrix, family, kappa)
        runs = plan_runs(names, seeds)
        checked = {
            name: learn_run.load_model(name, matrix) for name in dict.fromkeys(names.values())
        }
    except (CouldNotRun, MatrixError) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        for name, model in checked.items():
            learn_run.check_model(name, model, tuple(model.fit_units), matrix.gate_units)
    except learn_run.FitOnScored as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    fit_units = tuple(dict.fromkeys(u for model in checked.values() for u in model.fit_units))
    needed = tuple(dict.fromkeys((*matrix.gate_units, *fit_units)))
    del checked  # each run builds its own model, so no fit of one run reaches another
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
    fixed = fallback_dials(run_matrix, run_family)
    started = time.perf_counter()
    found: dict[str, Measured] = {}
    failed_runs: dict[str, str] = {}
    for run in runs:
        try:
            model = learn_run.load_model(run.model, matrix)
            if run.seed is not None:
                model = seeded_model(model, run.model, run.seed)
            found[run.name] = measure(
                run, model, run_matrix, run_family, fixed, units, loaded, splits, args.workers
            )
        except (learn_run.Refused, ValueError, MatrixError) as problem:  # recorded, the rest go on
            failed_runs[run.name] = f"{type(problem).__name__}: {problem}"
        model = None  # the next run builds its own
    complete = {}
    for arm in ARMS:
        of_arm = [r for r in runs if r.arm == arm]
        if of_arm and all(r.name in found for r in of_arm):
            complete[arm] = [found[r.name].document for r in of_arm]
    try:
        comparison = forecast_noset_pairs.compare(complete)
    except ValueError as problem:
        print(f"FAILED: the runs cannot be paired: {problem}")
        return 1
    matrix_record = {"file": args.matrix.name, "sha256": forecast_mix._sha256(args.matrix)}
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    folder = Path(run_dir) if run_dir else None
    failed = [f"{name}: {text}" for name, text in failed_runs.items()]
    for run in runs:
        if run.name not in found:
            continue
        one = found[run.name]
        document = one.document
        document["inputs"] = inputs
        document["matrix"] = matrix_record
        after_sealed_run.mark(document, True)
        if args.not_the_trial:
            document["not_the_trial"] = {"overrides": changed}
            document["label"] = f"{NOT_THE_TRIAL}. {document['label']}"
        if folder is not None:
            table_name = side_name(run.name, "forecasts")
            write_table(one.table, folder / table_name)
            document["forecast_table"] = {
                "file": table_name,
                "rows": len(one.table),
                "sha256": hashlib.sha256((folder / table_name).read_bytes()).hexdigest(),
            }
        failed += [f"{run.name}: {text}" for text in document["failed_units"]]
    bars = bars_of(settings)
    comparisons = {}
    for which in forecast_noset_reading.COMPARISONS:
        read = forecast_noset_reading.read_units(comparison, bars, which)
        check = forecast_noset_reading.check_units(comparison, bars, which)
        silent = forecast_noset_reading.silent_units(read, check)
        comparisons[which] = {
            "units": read,
            "check": check,
            "disagreeing_units": forecast_noset_reading.disagreements(read, check),
            "silent_units": silent,
            "silent_count": len(silent),
            "one_line": forecast_noset_reading.one_line(read, check),
        }
    overall = forecast_noset_reading.overall(
        {which: one["one_line"] for which, one in comparisons.items()}
    )
    first = next(iter(found.values())).document if found else None
    ran = {
        "level": kappa,
        "splits": splits,
        "neural_seeds": list(seeds),
        "models": names,
        "matrix_sha256": matrix_record["sha256"],
        "settings_sha256": forecast_mix._sha256(args.settings),
    }
    reference = {
        "matrix_sha256": forecast_mix._sha256(MATRIX),
        "settings_sha256": forecast_mix._sha256(SETTINGS_FILE),
    }
    document = {
        "label": f"forecast trial noset: {LABEL}",
        "definitions": definitions(committed),
        "run": {
            "kappa": kappa,
            "splits": splits,
            "seeds": list(seeds),
            "split_seeds": first["run"]["seeds"],
            "models": dict(names),
            "models_as_run": {name: one.document["model"] for name, one in found.items()},
            "runs": [run.name for run in runs],
            "dials": first["run"]["dials"],
            "fallback_dials": fixed,
            "tiers": first["run"]["tiers"],
            "rung": first["run"]["rung"],
            "units": list(matrix.gate_units),
        },
        "runs": {
            name: {
                "arm": one.run.arm,
                "model": one.document["model"],
                "fit_seed": one.run.seed,
                "fit_units": one.document["fit_units"],
                "forecast_table": one.document.get("forecast_table"),
                "timeouts": one.document["timeouts"],
                "failed_units": one.document["failed_units"],
            }
            for name, one in found.items()
        },
        "failed_runs": failed_runs,
        "reading": {
            "rule": forecast_noset_reading.RULE,
            "no_other_sites": forecast_noset_reading.NO_OTHER_SITES,
            "bars": settings["reading"]["bars"],
            "comparisons": comparisons,
            "overall": overall,
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
        timing = {name: one.timing for name, one in found.items()}
        timing["total_seconds"] = time.perf_counter() - started
        for name, one in found.items():
            (folder / side_name(name, "dial-whatif")).write_text(_json(one.document))
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
