"""The tree model refitted with a unit's own first days: a trial made after the sealed run
(`after_sealed_run`), on development units only.

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.forecast_local [--units <unit> ...] [--workers N] [--matrix <file>]
        [--settings <file>] [--tables <folder>] [--truth-run <folder>]

Question. The cold start of plan section 11 (`cold_start`) fits the main model (`lgbm_tail`) on
training units only and lets a scored unit's first days reach it through the calibration and the
user histories alone; its reading C2 names "refitting the model with the local days" as not taken.
This trial takes it: the same model, refitted once per unit on the training units' rows and that
unit's own first k counting days, in two ways fixed in advance (the two refit arms below), then
compared with the model not refitted, on the same later days, at the same level, dial,
calibration rule and seeded splits. It changes no plan definition
and no pass mark and fills no slot of the plan.

Settings. Every value is in the committed file `trials-after-sealed-run/forecast-local.json`
(`SETTINGS`), written before any result. The trial is the command run with no option. No option
sets one value; but options exist that change what is read: `--settings` names another settings
file (every value in it), `--matrix` another run matrix (the configurations, the calibration
rule and the readings of the gate), `--units` fewer units than the file names, `--tables` and
`--truth-run` the tables and the truth build read, and `--workers` the number of processes. The
results record the settings file's name and SHA-256 (`settings`) and the matrix's (`matrix`), so a
run with an option shows it. `read_settings` refuses a file with another set of
keys, a `local_days` that is not one of the plan's k (`cold_start_days.LOCAL_DAYS`, above 0) and
`refits` other than exactly `REFITS`: neither refit arm can be dropped, added or reordered. Both
arms are always run and always reported, whatever they show; the better one is never given alone.

Units. A unit that is not a development unit of the truth build (`DEVELOPMENT_UNITS`) is refused
by `check_units` before any data file is read (exit 2; the settings file is read first, to
check the units it names): held-out, training and sealed names alike, in `--units` and in the
settings file. The module names no loader of held-out data. A unit with too
few counting days after its first k to be split (`gate_run.plan_unit`) is listed and left out.

The arms, all built from the matrix (`learn_run.load_model`):
- main: `lgbm_tail`, fitted once on the training units' cars tables, as
  `dial_choice.make_factory` fits it;
- `unweighted-pool`: per unit, `lgbm_tail` fitted on the same tables followed by the unit's local
  fit rows (`local_fit_rows`), each row once with weight 1; the fit is the model's own `fit`;
- `equal-mass`: per unit, the same rows, each training-pool row with weight 1 and each local row
  with the weight (number of pool rows) / (number of local fit rows) (`equal_mass_weight`), so
  the local rows weigh as much in total as the pool. The model is `models.lgbm_tail_equal_mass`
  (`EQUAL_MASS_SUFFIX`): `lgbm_tail` with the weight column `WEIGHT_COLUMN` of the fit frame given
  to every booster it fits (each quantity and level) as the row weight of the LightGBM data set;
  its features, labels, settings and predict are those of `lgbm_tail`. The weight counts every
  row of the fit frame, so the equal total mass holds over the rows, not for each quantity's
  subset of rows (a quantity is fitted on the rows that have its label).
The guard of `learn_run.check_model` that a model's `fit_units` share no unit with the scored
units is run as always (all models declare the training units); the local rows are the one thing
put in beside them, and they are rows of the days the scores never read.

Local days (reading F1). The first k counting days of the unit in `cold_start_days`' order (date,
then site, so a pooled unit counts site-days; its reading C5), k = `Settings.local_days`. Not
taken: calendar days; k per site.

Local fit rows (reading F2). The rows of the unit's cars table at or before the k-th counting day
(`cold_start_days.local_rows`, its reading C4: also the rows of days that do not count), kept
when the session had plugged out by the decision time τ of the earliest day after the k local days
(`first_forecast_time`, `local_fit_rows`). A forecast day is any counting day after the k-th, so
every outcome a fit row holds was known before the first forecast day began, and no day at or
after a forecast day enters the fit; the fit is made once and serves every later day. Rows of the
first k site-days can be left out at any unit: a session still plugged in at that time, or with no
readable plug-out, is left out; at a pooled unit two sites can also share a date, so the k-th
site-day can share its date with the first forecast site-day. `local_row_counts` gives, per unit
and refit arm, the site-days asked for, the site-days with a kept row, the rows kept and the rows
left out, in the results and the summary. Not taken:
the end of the local period of `cold_start_days.local_end` (it is not later than this time, and
this rule is the stricter guarantee); a refit before each forecast day (it would add days
between the k-th and the forecast day, which the trial leaves out on purpose).

Days read (reading F3). The days of the unit after its first k are split into calibration and
test days by `dial_choice.split_indices` on those days alone (`tail_of`; the matrix's
calibration-days rule, `day_split`), seeds 1 to `splits`, and every arm is scored on the same
splits. The first k days are in no calibration set and no test set of either arm (`score_arms`
blanks their losses; line `without_first`). A test day's own outcome reaches neither its forecast
(the fit rows are the first k days' rows, none a test day's) nor its calibration (calibration
reads the calibration days' losses, `dial_splits.score_split`). The user histories in a forecast
table are those of `learn_features` for the whole unit, as in every run of `lgbm_tail`, and are
the same in every arm. Not taken: the existing splits of all the unit's days, which would score
the first k days.

Level, dial, rung (reading F4). The main level `kappa` 0.5 and the dial 0.02: the level and the
dial that the rules of plan sections 6 and 7 gave the main model at both site types in the
development runs (`docs/prior-looks.md`, the six-level runs: workplaces 17.70, Norway 2.48 kWh per
test day), the same for every arm: no refit arm is given a level or a dial of its own.
Calibration: the on-grid rule of `dial_splits` on the one-way re-solve rung, both tiers, never
pooled, as `dial_choice` measures; the empirical-departure configuration of that family.

Paired comparison (reading F5). Per unit and tier, over the splits: each arm's summary
(`dial_splits.summarise`) and, for each refit arm, the paired difference arm minus main per split
(`compare_paired.pair_statistics`: mean, 5th and 95th percentile, shares above, equal and below 0,
and its fixed word "ahead", "behind" or "not separated"). The range is the spread over the seeded
splits of one unit's days, not an interval for new days. A reference line, `main_all_days`: the
main arm scored on all the unit's days with the splits of `dial_choice` (the figures a
`dial_choice --kappa 0.5` run gives for the same model), so the harness can be read against the
recorded main figures; it is not part of the comparison.

Reading (reading F6), fixed in the settings file's block `reading` before any result and printed
in the summary and in `definitions`. The trial gives two refit arms, two units that are scored
and two tiers, eight comparisons; only four are judged. The judged comparison is tier A, per unit
and per refit arm: the paired mean difference of the offer (arm minus main) over the seeded
splits as a share of the main arm's mean offer (`compare_paired`'s
`mean_difference_share_of_reference_mean_offer`). The noise bar of each unit is
`reading.noise_bar_share`: 8% at the workplace unit and 30% at the pooled Norwegian unit, the
shares by which the seed alone moved an offer (`docs/prior-looks.md`, the entry that begins with
the words "The neural model's seed test"). They are the neural model's range over three seeds
(with the plan's own runs) at one level, kappa 0.3; the refitted tree model has one fixed seed, so
the bars are an analogy and not this trial's measured noise, and at the pooled Norwegian unit a
difference of up to 30% therefore reads "no measurable difference". An arm "shows a
gain at a unit" (`GAIN`) only if the share is above the bar, "shows a loss" (`LOSS`) only if it
is below minus the bar, otherwise "no measurable difference" (`verdict_word`). Tier B's four
cells are printed as numbers, labelled information. The one line (`one_line_reading`): "the refit
helps" only if at least one arm shows a gain at a unit and no arm shows a loss at any unit; "the
refit hurts" if any arm shows a loss and none a gain; otherwise "no measurable difference; a mixed
result is reported as mixed"; "not read" when a judged cell is missing, among them when a unit
that has a bar has no judged cell (`missing_judged_units`; left out, failed or not run; the
reason is in the results and the summary). The 200 splits share the unit's days, so the standard
error over splits understates the uncertainty, and the trial cannot speak for the Dutch or Korean
sites, none of which is a development unit. No arm is added after a result is seen.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set, the summary also on standard output:
- `forecast-local.json` (`RESULTS_NAME`): the settings, per unit the days, the split sizes, the
  local fit rows and the weight, and per tier the three arms, the paired figures and the
  reference;
- `forecast-local-summary.txt`, one forecast table per arm (`TABLE_NAMES`, hashed in the
  results) and `forecast-local-timing.json`;
- `after_sealed_run.FILE_NAME`, the label file.
Every output says that it is a trial made after held-out results were seen: the key
`made_after_sealed_run` and the label of the results, the first lines of the summary and the label
file. The label is always on.

Exit status: 0 every unit left in was measured; 1 the model was refused or failed, or a measure
failed; 2 could not run (a bad argument, settings, matrix or model name, a unit that is not a
development unit, no unit with enough days, a model fitted on a scored unit, a table missing, no
verified truth build at this commit, a table differing from it).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve import (
    after_sealed_run,
    cold_start_days,
    compare_paired,
    dial_choice,
    dial_splits,
    dial_summary,
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
SETTINGS = TOP / "trials-after-sealed-run" / "forecast-local.json"
RESULTS_NAME = "forecast-local.json"
SUMMARY_NAME = "forecast-local-summary.txt"
REFITS = ("unweighted-pool", "equal-mass")
TABLE_NAMES = {
    "main": "forecast-local-forecasts-main.csv",
    "unweighted-pool": "forecast-local-forecasts-unweighted-pool.csv",
    "equal-mass": "forecast-local-forecasts-equal-mass.csv",
}
WEIGHT_COLUMN = "fit_weight"
EQUAL_MASS_SUFFIX = "_equal_mass"
TIMING_NAME = "forecast-local-timing.json"
SETTING_KEYS = (
    "about",
    "model",
    "kappa",
    "dial",
    "local_days",
    "splits",
    "units",
    "refits",
    "reading",
)
READING_KEYS = ("judged", "noise_bar_share", "noise_bar_source", "tier_b", "one_line", "limits")
JUDGED_TIER = "A"
GAIN = "shows a gain"
LOSS = "shows a loss"
NOTHING = "no measurable difference"
UNDEFINED = "not defined (the main arm's mean offer is 0)"
NO_BAR = "no bar is fixed for this unit"
HELPS = "the refit helps"
HURTS = "the refit hurts"
MIXED = "no measurable difference; a mixed result is reported as mixed"
NOT_READ = "not read: a judged cell is missing"
TIERS = dial_splits.TIERS
RUNG = dial_choice.RUNG
LOCAL_FRAMES = "local rows"
LABEL = (
    "a trial: the main tree model refitted with the unit's own first days, in two ways fixed in "
    "advance, against the same model not refitted, on the unit's later days"
)


# --- the settings and the units -------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """The values of `trials-after-sealed-run/forecast-local.json` (module text)."""

    model: str
    kappa: float
    dial: float
    local_days: int
    splits: int
    units: tuple[str, ...]
    refits: tuple[str, ...]
    reading: dict


def check_units(units) -> tuple[str, ...]:
    """`units` as a tuple when each is a development unit of the truth build, named once; raises
    `CouldNotRun` otherwise (module text). Nothing is read."""
    units = tuple(units)
    if not units:
        raise CouldNotRun("no unit is named")
    unknown = [u for u in units if u not in DEVELOPMENT_UNITS]
    if unknown:
        raise CouldNotRun(
            f"{', '.join(unknown)}: not a development unit ({', '.join(DEVELOPMENT_UNITS)}); "
            "training, held-out and sealed names are refused"
        )
    twice = sorted({u for u in units if units.count(u) > 1})
    if twice:
        raise CouldNotRun(f"{', '.join(twice)} named more than once")
    return units


def read_reading(block) -> dict:
    """The block `reading` of the settings (module text, reading F6); raises `CouldNotRun` when
    it is absent, has other keys, a text that is empty or a noise bar that is not a share above 0
    and below 1 of a development unit."""
    if not isinstance(block, dict) or set(block) != set(READING_KEYS):
        raise CouldNotRun(f"the settings file's reading must hold exactly {list(READING_KEYS)}")
    for key in READING_KEYS:
        if key != "noise_bar_share" and (not isinstance(block[key], str) or not block[key]):
            raise CouldNotRun(f"the settings file's reading.{key} must be a text")
    bars = block["noise_bar_share"]
    if not isinstance(bars, dict) or not bars:
        raise CouldNotRun("reading.noise_bar_share must name a bar for at least one unit")
    check_units(list(bars))
    for unit, bar in bars.items():
        if isinstance(bar, bool) or not isinstance(bar, int | float) or not 0 < bar < 1:
            raise CouldNotRun(f"the noise bar of {unit} must be a share above 0 and below 1")
    return block


def verdict_word(share, bar) -> str:
    """The word for a tier A cell (module text, reading F6): `share`, the paired mean difference
    as a share of the main arm's mean offer, against the unit's `bar`. Above the bar: `GAIN`;
    below minus the bar: `LOSS`; on or inside it: `NOTHING`; no share (the main offer is 0):
    `UNDEFINED`; no bar: `NO_BAR`."""
    if bar is None:
        return NO_BAR
    if share is None:
        return UNDEFINED
    if share > bar:
        return GAIN
    if share < -bar:
        return LOSS
    return NOTHING


def missing_judged_units(bars, entries) -> list[str]:
    """The units that have a bar in the settings (`bars`) and no judged cell: no calibratable
    entry in `entries` (module text, reading F6). A unit without a bar is not judged and is
    never listed."""
    return sorted(unit for unit in bars if entries.get(unit, {}).get("status") != "calibratable")


def one_line_reading(words, missing=()) -> str:
    """The trial's one line from the words of its judged cells (module text, reading F6);
    `missing`: the units with a bar and no judged cell (`missing_judged_units`), which make the
    line `NOT_READ`."""
    words = list(words)
    if not words or NO_BAR in words or list(missing):
        return NOT_READ
    gain, loss = GAIN in words, LOSS in words
    if gain and not loss:
        return HELPS
    if loss and not gain:
        return HURTS
    return MIXED


def read_settings(path: Path) -> Settings:
    """The settings in the file `path` (module text); raises `CouldNotRun`. The units are
    checked by `check_units`."""
    try:
        document = json.loads(Path(path).read_bytes().decode())
    except (OSError, ValueError) as problem:
        raise CouldNotRun(f"the settings file cannot be read: {problem}") from problem
    if not isinstance(document, dict) or set(document) != set(SETTING_KEYS):
        raise CouldNotRun(f"the settings file must hold exactly the keys {list(SETTING_KEYS)}")
    if not isinstance(document["units"], list) or not isinstance(document["model"], str):
        raise CouldNotRun("the settings file's model must be a name and its units a list")
    for key in ("kappa", "dial"):
        if isinstance(document[key], bool) or not isinstance(document[key], int | float):
            raise CouldNotRun(f"the settings file's {key} must be a number")
    k = document["local_days"]
    if isinstance(k, bool) or k not in cold_start_days.LOCAL_DAYS or k <= 0:
        raise CouldNotRun(
            f"local_days must be one of the plan's k above 0 {list(cold_start_days.LOCAL_DAYS)}"
        )
    if document["refits"] != list(REFITS):
        raise CouldNotRun(
            f"refits must be exactly {list(REFITS)}: both arms are run and reported, "
            "neither is dropped, added or reordered"
        )
    splits = document["splits"]
    if isinstance(splits, bool) or not isinstance(splits, int) or splits < 1:
        raise CouldNotRun("splits must be a whole number of at least 1")
    units = check_units(document["units"])
    reading = read_reading(document["reading"])
    return Settings(
        document["model"],
        float(document["kappa"]),
        float(document["dial"]),
        k,
        splits,
        units,
        tuple(document["refits"]),
        reading,
    )


# --- the days -------------------------------------------------------------------------------------


def tail_of(unit_days, k: int):
    """The unit with only its counting days after the first k (the cars table is whole)."""
    return replace(unit_days, days=unit_days.days[k:])


def first_forecast_time(unit_days, k: int) -> pd.Timestamp:
    """The earliest decision time τ, in UTC, of the counting days after the first k (module text,
    reading F2); raises `ValueError` when there is none."""
    cold_start_days.local_indices(len(unit_days.days), k)
    later = [d.record.tau for d in unit_days.days[k:]]
    if not later:
        raise ValueError(f"{unit_days.unit}: no counting day after the first {k}")
    return pd.to_datetime(pd.Series(later), utc=True, format="ISO8601").min()


def local_fit_rows(unit_days, k: int) -> pd.DataFrame:
    """The rows each refit arm adds to the training rows (module text, reading F2): the local rows
    whose session had plugged out by `first_forecast_time`; a plug-out that cannot be read is not
    known, so its row is left out."""
    rows = cold_start_days.local_rows(unit_days, k)
    cutoff = first_forecast_time(unit_days, k)
    plug_out = pd.to_datetime(rows["plug_out"], utc=True, format="ISO8601", errors="coerce")
    return rows[(plug_out <= cutoff).to_numpy(bool)]


def local_row_counts(unit_days, k: int, fit_rows: pd.DataFrame) -> dict:
    """What the cutoff of `local_fit_rows` kept, as counts only (module text, reading F2):
    `fit_rows` is that function's result for the same unit and k. `local_site_days_asked`: k;
    `local_site_days_with_a_kept_row`: how many of the first k counting days have at least one
    kept row; `rows_kept`: the rows of `fit_rows`; `rows_left_out`: the local rows the cutoff
    removed. Nothing here reaches a fit or a score."""
    first_k = set(cold_start_days.day_keys(unit_days)[:k])
    kept_days = set(cold_start_days._row_keys(fit_rows)) & first_k
    local = len(cold_start_days.local_rows(unit_days, k))
    return {
        "local_site_days_asked": k,
        "local_site_days_with_a_kept_row": len(kept_days),
        "rows_kept": len(fit_rows),
        "rows_left_out": local - len(fit_rows),
    }


def equal_mass_weight(pool_rows: int, local_rows: int) -> float:
    """The weight of each local fit row of the `equal-mass` arm: the number of training-pool rows
    over the number of local fit rows, so the local rows weigh as much in total as the pool, each
    pool row weighing 1 (module text); raises `ValueError` for no local row or no pool row."""
    if pool_rows < 1 or local_rows < 1:
        raise ValueError(f"equal mass needs rows on both sides, got {pool_rows} and {local_rows}")
    return pool_rows / local_rows


def refit_frames(arm: str, fit_frames: dict, local_rows: pd.DataFrame, unit: str) -> dict:
    """The cars tables a refit arm's model is fitted on (module text): the training units'
    tables, then the unit's local fit rows. `unweighted-pool`: the tables as they are, no weight
    column. `equal-mass`: each table with the column `WEIGHT_COLUMN` of 1 and the local rows with
    `equal_mass_weight`. Raises `ValueError` for another arm."""
    local = f"{LOCAL_FRAMES} of {unit}"
    if arm == REFITS[0]:
        return {**fit_frames, local: local_rows}
    if arm == REFITS[1]:
        weight = equal_mass_weight(sum(len(f) for f in fit_frames.values()), len(local_rows))
        pool = {name: f.assign(**{WEIGHT_COLUMN: 1.0}) for name, f in fit_frames.items()}
        return {**pool, local: local_rows.assign(**{WEIGHT_COLUMN: weight})}
    raise ValueError(f"no refit arm {arm!r}; the arms are {list(REFITS)}")


def arm_model_name(model: str, arm: str) -> str:
    """The model module of a refit arm: the main model's own, or its equal-mass variant."""
    return model if arm == REFITS[0] else f"{model}{EQUAL_MASS_SUFFIX}"


def tail_splits(name: str, unit_days, matrix, k: int, seeds):
    """(the plan of the days after the first k, [(seed, calibration indices, test indices)] of the
    whole unit's day numbering): the splits of `dial_choice.split_indices` on the later days,
    shifted by k (module text, reading F3)."""
    tail = tail_of(unit_days, k)
    plan = gate_run.plan_unit(name, tail, matrix)
    if plan.status != "calibratable":
        return plan, []
    found = dial_choice.split_indices(name, tail, matrix, seeds)
    shifted = [(seed, [d + k for d in cal], [d + k for d in test]) for seed, cal, test in found]
    return plan, shifted


def without_first(losses: dial_splits.UnitLosses, k: int) -> dial_splits.UnitLosses:
    """The losses with the first k days blanked: no value, not offering, losses and ceiling not a
    number, so that nothing scored can have read them (module text, reading F3)."""
    value, offering = losses.value.copy(), losses.offering.copy()
    slot, driver, ceiling = losses.slot.copy(), losses.driver.copy(), losses.ceiling.copy()
    value[:k], offering[:k], ceiling[:k] = 0.0, False, np.nan
    slot[:k], driver[:k] = np.nan, np.nan
    return replace(
        losses, value=value, offering=offering, slot=slot, driver=driver, ceiling=ceiling
    )


# --- scoring --------------------------------------------------------------------------------------


def offers_of(scores: list) -> dict:
    return {s.seed: s.offer_kwh for s in scores}


def score_arms(main: dial_splits.UnitLosses, refits: dict, splits, k: int) -> dict:
    """Per tier: each arm's per-split scores (`refits`: arm -> losses, beside the main arm's) and,
    per refit arm, the paired figures arm minus main on the same splits (module text, reading
    F5). The first k days' losses are blanked first."""
    seeds = [seed for seed, _, _ in splits]
    out = {}
    for tier in TIERS:
        scores = {
            arm: dial_choice.score_each(without_first(losses, k), splits, (tier,))[tier]
            for arm, losses in (("main", main), *refits.items())
        }
        out[tier] = {
            "scores": scores,
            "paired": {
                arm: compare_paired.pair_statistics(
                    offers_of(scores[arm]), offers_of(scores["main"]), seeds
                )
                for arm in refits
            },
        }
    return out


def measure_arm(runner, name: str, n_days: int, configuration):
    """(the unit's day-level losses, the certificate and re-solve timeouts) of one arm: every day
    prepared, every offering day evaluated at each scale of the grid (`dial_choice.measure`)."""
    prepared = runner.map(gate_run._prepare_task, [(name, i, configuration) for i in range(n_days)])
    losses, cache = dial_choice.measure(
        prepared, [None] * n_days, lambda pairs: runner.map(gate_run._evaluate_task, pairs)
    )
    timeouts = {
        "certificates": sum(p.certificate_outcome == "timeout" for p in prepared),
        "re_solves": dial_choice._evaluations(cache)["re_solves_of_the_one_way_rung"].get(
            "timeout", 0
        ),
    }
    return losses, timeouts


# --- one run --------------------------------------------------------------------------------------


def reading_text(reading: dict) -> list[str]:
    """The reading fixed before any result, as lines (module text, reading F6)."""
    bars = ", ".join(f"{unit} {100 * bar:g}%" for unit, bar in reading["noise_bar_share"].items())
    return [
        f"Judged: {reading['judged']}.",
        f"Noise bar per unit: {bars}. Source: {reading['noise_bar_source']}.",
        f'An arm "{GAIN}" at a unit only if its share is above the unit\'s bar, "{LOSS}" only if '
        f'it is below minus the bar, otherwise "{NOTHING}".',
        f"Tier B: {reading['tier_b']}.",
        f"One line: {reading['one_line']}.",
        f"Limits: {reading['limits']}.",
    ]


def definitions(settings: Settings, matrix, configuration) -> dict:
    return {
        "trial": LABEL,
        "local_days": (
            f"the unit's first {settings.local_days} counting days in the order of "
            "cold_start_days (date, then site; a pooled unit counts site-days)"
        ),
        "local_fit_rows": (
            "the rows of the unit's cars table at or before the last local day whose session had "
            "plugged out by the decision time of the earliest later counting day, added once to "
            "the training units' rows for each refit arm's one fit: weight 1 in unweighted-pool, "
            "(training rows) / (local rows) in equal-mass (module text, reading F2)"
        ),
        "local_rows_by_arm": (
            "per unit and refit arm: the local site-days asked for (k), the local site-days with "
            "at least one kept row, the rows kept and the rows the cutoff left out; counts only "
            "(module text, reading F2). Rows kept and left out count all local rows, those of "
            "days that do not count included; the site-day count covers the first k counting "
            "site-days only"
        ),
        "days_read": (
            "the unit's counting days after the first local days, split by the matrix's "
            f"calibration-days rule (n_calibration {matrix.n_calibration}, "
            f"calibration_half_below {matrix.calibration_half_below}) with seeds 1 to "
            f"{settings.splits}; the same splits for every arm; the local days are in neither "
            "set (reading F3)"
        ),
        "paired": (
            "per tier and refit arm, arm minus main per split (compare_paired); the range is the "
            "spread over the splits of one unit's days, not an interval for new days"
        ),
        "main_all_days": (
            "the main arm on all the unit's days with the splits of dial_choice: a reading of the "
            "harness against the recorded main figures, not part of the comparison"
        ),
        "configuration": configuration.name,
        "tiers": "the tiers are separate blocks and are never added or averaged",
        "reading": reading_text(settings.reading),
    }


def share_cells(found: dict, bar) -> dict:
    """Per refit arm of one tier's paired figures: the share of the main arm's mean offer, and
    for the judged tier the bar and the word (module text, reading F6)."""
    cells = {}
    for arm, pair in found["paired"].items():
        share = pair["mean_difference_share_of_reference_mean_offer"]
        cells[arm] = {"share_of_main_offer": share}
        if bar is not False:
            cells[arm].update({"bar": bar, "word": verdict_word(share, bar)})
    return cells


def unit_entry(
    days, plan, k, fit_rows, pool_rows, scored, reference, offering, timeouts, bar=None
) -> dict:
    """One unit's block of the results; `bar`: the unit's noise bar (reading F6)."""
    keys = cold_start_days.day_keys(days)
    return {
        "counting_days": len(keys),
        "local_days": [f"{d} {s}" for d, s in keys[:k]],
        "days_after_the_local_days": len(keys) - k,
        "calibration_days_used": plan.calibration_days,
        "test_days": plan.test_days,
        "status": plan.status,
        "local_fit_rows": len(fit_rows),
        "local_rows_by_arm": {arm: local_row_counts(days, k, fit_rows) for arm in REFITS},
        "equal_mass_weight": equal_mass_weight(pool_rows, len(fit_rows)),
        "first_forecast_time": str(first_forecast_time(days, k)),
        "offering_days_after_the_local_days": offering,
        "timeouts": timeouts,
        "tiers": {
            tier: {
                **{arm: dial_splits.summarise(s, tier) for arm, s in found["scores"].items()},
                "paired_minus_main": found["paired"],
                "main_all_days": reference[tier],
                "reading": share_cells(found, bar if tier == JUDGED_TIER else False),
            }
            for tier, found in scored.items()
        },
    }


def run(matrix, units: dict, arms: dict, configuration, settings: Settings, *, workers: int):
    """(results document, timing) of the trial on `units` (name -> `UnitDays`, the names of
    `matrix.gate_units`); `arms`: {"main": (factory, table), "refits": {arm: {unit: (factory,
    table)}}} with the arms of `REFITS`; `pool_rows`: from `arms["pool_rows"]`, the number of
    training-pool rows.
    Everything but the input checks and the writing of files (module text)."""
    started = time.perf_counter()
    k = settings.local_days
    seeds = list(range(1, settings.splits + 1))
    entries, failed, timing = {}, {}, {"units": {}}
    for name in matrix.gate_units:
        days = units[name]
        plan, splits = tail_splits(name, days, matrix, k, seeds)
        if plan.status != "calibratable" or not splits:
            entries[name] = {
                "counting_days": len(days.days),
                "days_after_the_local_days": max(len(days.days) - k, 0),
                "status": f"left out: {plan.status} after the first {k} days",
            }
            continue
        begun = time.perf_counter()
        n = len(days.days)
        measured, timeouts = {}, {}
        try:
            pairs = [("main", arms["main"])]
            pairs += [(arm, arms["refits"][arm][name]) for arm in REFITS]
            for arm, (factory, _) in pairs:
                runner = gate_run.Runner(workers, {name: days}, matrix, factory)
                try:
                    measured[arm], timeouts[arm] = measure_arm(runner, name, n, configuration)
                finally:
                    runner.close()
        except Exception as problem:  # recorded, the other units go on
            failed[name] = f"{type(problem).__name__}: {problem}"
            entries[name] = {"counting_days": n, "status": "failed", "failed": failed[name]}
            continue
        scored = score_arms(measured["main"], {arm: measured[arm] for arm in REFITS}, splits, k)
        all_days = dial_choice.split_indices(name, days, matrix, seeds)
        reference = {
            tier: dial_splits.summarise(
                dial_choice.score_each(measured["main"], all_days, (tier,))[tier], tier
            )
            for tier in TIERS
        }
        offering = {arm: int(found.offering[k:].sum()) for arm, found in measured.items()}
        entries[name] = unit_entry(
            days,
            plan,
            k,
            local_fit_rows(days, k),
            arms["pool_rows"],
            scored,
            reference,
            offering,
            timeouts,
            settings.reading["noise_bar_share"].get(name),
        )
        timing["units"][name] = time.perf_counter() - begun
    timing["total_seconds"] = time.perf_counter() - started
    document = {
        "label": f"forecast trial local: {LABEL}",
        "definitions": definitions(settings, matrix, configuration),
        "run": {
            "model": settings.model,
            "kappa": configuration.kappa,
            "dial": configuration.driver_risk,
            "configuration": configuration.name,
            "local_days": k,
            "refits": list(settings.refits),
            "splits": settings.splits,
            "seeds": [seeds[0], seeds[-1]],
            "scales": list(dial_splits.GRID),
            "rung": RUNG,
            "tiers": list(TIERS),
            "units": list(matrix.gate_units),
        },
        "units": entries,
        "failed_units": sorted(f"{unit}: {why}" for unit, why in failed.items()),
    }
    words = [
        cell["word"]
        for entry in entries.values()
        if entry.get("status") == "calibratable"
        for cell in entry["tiers"][JUDGED_TIER]["reading"].values()
    ]
    missing = missing_judged_units(settings.reading["noise_bar_share"], entries)
    document["reading"] = {
        "rule": settings.reading,
        "words": words,
        "units_with_a_bar_and_no_judged_cell": missing,
        "one_line": NOT_READ if failed else one_line_reading(words, missing),
    }
    return document, timing


def finish(document: dict) -> dict:
    """The results document with the label of a trial made after the sealed run (always)."""
    after_sealed_run.mark(document, True)
    return document


def _signed(value) -> str:
    return "-" if value is None else f"{value:+.2f}"


def summary_lines(document: dict) -> list[str]:
    run = document["run"]
    lines = [
        *after_sealed_run.summary_head(document),
        f"Forecast trial local, model {run['model']}: refitted with the unit's first "
        f"{run['local_days']} counting days, once, in two ways (unweighted-pool: each local row "
        f"weighs 1; equal-mass: the local rows weigh as much in total as the training rows) "
        f"against the same model not refitted; both ways are always reported; level "
        f"{run['kappa']:g}, dial {run['dial']:g}, one-way re-solve rung, "
        f"{run['splits']} seeded splits of the days after the first {run['local_days']}.",
        LABEL[0].upper() + LABEL[1:] + ".",
        "Offers are kWh per counting test day. The two tiers are separate blocks and are never "
        "added or averaged. The paired range is the spread over the splits of one unit's days, "
        "not an interval for new days.",
        "Reading fixed before any result:",
        *(f"  {line}" for line in document["definitions"]["reading"]),
    ]
    for unit, entry in document["units"].items():
        lines.append("")
        if entry["status"] != "calibratable":
            lines.append(f"{unit} ({entry['counting_days']} counting days): {entry['status']}")
            continue
        lines.append(
            f"{unit} ({entry['counting_days']} counting days; {entry['local_fit_rows']} local fit "
            f"rows, equal-mass weight {entry['equal_mass_weight']:.2f}; "
            f"{entry['days_after_the_local_days']} days after the local days: each split "
            f"{entry['calibration_days_used']} calibration and {entry['test_days']} test days):"
        )
        for arm, counts in entry["local_rows_by_arm"].items():
            lines.append(
                f"  {arm}: local site-days asked for {counts['local_site_days_asked']}, with at "
                f"least one kept row {counts['local_site_days_with_a_kept_row']}; rows kept "
                f"{counts['rows_kept']}, rows left out by the cutoff {counts['rows_left_out']}"
            )
        for tier, found in entry["tiers"].items():
            lines.append(f" {TIER_LABELS[tier]}")
            lines.append(dial_summary._header())
            for label, key in (
                ("main", "main"),
                ("unweighted", REFITS[0]),
                ("equal-mass", REFITS[1]),
                ("main, all days", "main_all_days"),
            ):
                lines.append(dial_summary._row(label, found[key], tier))
            for arm in REFITS:
                pair = found["paired_minus_main"][arm]
                lines.append(
                    f"  {arm} minus main over {pair['splits_paired']} splits: mean "
                    f"{_signed(pair['mean_difference_kwh_per_test_day'])} "
                    f"[{_signed(pair['difference_p5_kwh_per_test_day'])}, "
                    f"{_signed(pair['difference_p95_kwh_per_test_day'])}] kWh; above 0 in "
                    f"{dial_summary._share(pair['share_of_paired_splits_above_0'])}, equal in "
                    f"{dial_summary._share(pair['share_of_paired_splits_equal_0'])}, below in "
                    f"{dial_summary._share(pair['share_of_paired_splits_below_0'])}; "
                    f"{pair['verdict']}"
                )
            for arm, cell in found["reading"].items():
                share = cell["share_of_main_offer"]
                shown = "-" if share is None else f"{100 * share:+.1f}%"
                if "word" in cell:
                    bar = "-" if cell["bar"] is None else f"{100 * cell['bar']:g}%"
                    lines.append(
                        f"  {arm}, share of the main offer {shown} against a bar of {bar}: "
                        f"{cell['word']}"
                    )
                else:
                    lines.append(
                        f"  {arm}, share of the main offer {shown} (information only, not judged)"
                    )
    lines.append("")
    lines.append(f"Reading of the trial: {document['reading']['one_line']}")
    missing = document["reading"]["units_with_a_bar_and_no_judged_cell"]
    if missing:
        lines.append(
            "Reason: no judged cell for " + ", ".join(missing) + " (a unit with a bar that was "
            "left out, failed or not run), so the one line is not read from the other units alone."
        )
    if document["failed_units"]:
        lines.append("")
        lines.append("FAILED: " + "; ".join(document["failed_units"]))
    return lines


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def write_outputs(folder: Path, document: dict, text: str, timing: dict) -> None:
    """The results, the summary, the timing and the label file in the run folder `folder`."""
    folder = Path(folder)
    (folder / RESULTS_NAME).write_text(_json(document))
    (folder / SUMMARY_NAME).write_text(text)
    (folder / TIMING_NAME).write_text(_json(timing))
    after_sealed_run.write_note(folder, True)


def configuration_of(matrix, kappa: float, dial: float):
    """(the matrix with the family at `kappa`, its configuration with the dial `dial`), as
    `dial_choice` makes them (module text, reading F4); raises `MatrixError`."""
    family = dial_choice.find_family(matrix)
    measured, family = dial_choice.family_at(matrix, family, kappa)
    chosen = [c for c in family[1] if c.driver_risk == dial]
    if len(chosen) != 1:
        dials = [c.driver_risk for c in family[1]]
        raise MatrixError(f"the dial {dial:g} is not one of the family's dials {dials}")
    return measured, chosen[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", nargs="+", default=None, metavar="UNIT")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--settings", type=Path, default=SETTINGS)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.units is not None:
            check_units(args.units)  # before any file is read at all
        gate_run.argument_paths(args, (*gate_run.PATH_ARGUMENTS, "settings"))
        settings = read_settings(args.settings)  # its units are checked here, still before data
        named = check_units(args.units if args.units is not None else settings.units)
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        gate_run.check_files([args.matrix])
        matrix = load(args.matrix)
        measured, configuration = configuration_of(matrix, settings.kappa, settings.dial)
        scored = replace(measured, gate_units=named, required_units=min(measured.required_units, 1))
        model = learn_run.load_model(settings.model, matrix)
    except (CouldNotRun, MatrixError) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    fit_units = tuple(model.fit_units)
    try:
        outcome_reader = learn_run.check_model(settings.model, model, fit_units, named)
        if outcome_reader:
            raise learn_run.Refused(
                f"model {settings.model} reads other days' outcomes; it has no fit to refit"
            )
    except learn_run.FitOnScored as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    needed = tuple(dict.fromkeys((*named, *fit_units)))
    try:
        inputs = gate_run.check_inputs(args.truth_run, args.tables, needed)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    k = settings.local_days
    try:
        loaded = {
            name: load_unit_days(
                args.tables,
                name,
                rho=scored.rho,
                no_fit=scored.no_fit,
                small_fleet=scored.small_fleet,
            )
            for name in needed
        }
        for name in named:
            cold_start_days.day_keys(loaded[name])  # the days must be in time order
        kept = tuple(
            n
            for n in named
            if gate_run.plan_unit(n, tail_of(loaded[n], k), scored).status == "calibratable"
        )
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    if not kept:
        print(f"COULD NOT RUN: no unit named has days enough after its first {k} to be split")
        return 2
    scored = replace(scored, gate_units=kept)
    units = {name: loaded[name] for name in kept}
    fit_frames = {name: loaded[name].cars for name in fit_units}
    try:
        factory, table = dial_choice.make_factory(
            settings.model, model, scored, units, fit_frames, configuration.kappa
        )
        arms = {
            "main": (factory, table),
            "refits": {arm: {} for arm in REFITS},
            "pool_rows": sum(len(f) for f in fit_frames.values()),
        }
        for name in kept:
            rows = local_fit_rows(units[name], k)
            for arm in REFITS:
                arm_name = arm_model_name(settings.model, arm)
                arms["refits"][arm][name] = dial_choice.make_factory(
                    arm_name,
                    learn_run.load_model(arm_name, matrix),
                    replace(scored, gate_units=(name,)),
                    {name: units[name]},
                    refit_frames(arm, fit_frames, rows, name),
                    configuration.kappa,
                )
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: the model could not be scored: {problem}")
        return 1
    document, timing = run(scored, units, arms, configuration, settings, workers=args.workers)
    for name in named:
        if name not in kept:
            document["units"][name] = {
                "counting_days": len(loaded[name].days),
                "status": f"left out: too few counting days after the first {k}",
            }
    document["model"] = model.name
    document["fit_units"] = list(fit_units)
    document["inputs"] = inputs
    document["settings"] = {
        "file": args.settings.name,
        "sha256": hashlib.sha256(args.settings.read_bytes()).hexdigest(),
    }
    document["matrix"] = {
        "file": args.matrix.name,
        "sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
    }
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        tables = {
            "main": arms["main"][1],
            **{
                arm: pd.concat([arms["refits"][arm][n][1] for n in kept], ignore_index=True)
                for arm in REFITS
            },
        }
        document["forecast_tables"] = {}
        for arm, found in tables.items():
            write_table(found, folder / TABLE_NAMES[arm], False)
            document["forecast_tables"][arm] = {
                "file": TABLE_NAMES[arm],
                "rows": len(found),
                "sha256": hashlib.sha256((folder / TABLE_NAMES[arm]).read_bytes()).hexdigest(),
            }
    finish(document)
    text = "\n".join(summary_lines(document)) + "\n"
    print(text, end="")
    if run_dir:
        write_outputs(Path(run_dir), document, text, timing)
    return 1 if document["failed_units"] else 0


if __name__ == "__main__":
    sys.exit(main())
