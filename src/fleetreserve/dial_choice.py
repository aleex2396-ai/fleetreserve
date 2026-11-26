"""Measure the calibrated re-solve offer at each dial of the driver-risk rows (plan section 6).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.dial_choice --model <name> [--matrix <file>] [--splits N]
        [--workers N] [--tables <folder>] [--truth-run <folder>]
        [--kappa <level> | --what-if-kappa <level>]
        [--what-if-fit-seed <N> | --what-if-fit-seeds <N1,N2,...>]
        [--what-if-fit-levels all|grid]

Plan section 6 fixes the dial, per site type, "as the value of Section 2 with the largest mean
calibrated re-solve offer over 200 development splits at the main model and kappa". This command
measures that and reports it. It writes no chosen value into code: the plan fixes the dial at the
main model before the tag, and the summary says so.

`--model <name>`: the module `fleetreserve.models.<name>`, as `learn_run` takes it. `empirical`
is the gate run's own forecaster (`models.empirical`, which gives the forecasts of
`forecast_empirical` as a table), so the empirical forecast of the gate run needs no other code.
The model is checked and fitted as `learn_run` does (its refusals, `fit_units`, `forecast_frame`
with a model's extra input columns, `check_table`; a look-ahead model is refused, `model_inputs`),
and the table is the forecaster of every certificate and re-solve
(`forecast_model.TableFactory`). A fitted model is built from the levels of the matrix, so the
matrix must list every level up to the family's kappa (the committed one lists 0.05 to 0.3, the
plan's grid up to 0.3; a run at 0.4 or 0.5 needs a matrix that lists them). `--matrix`: a
run matrix of the gate's format (default: the committed one); its gate units are the units
measured. `--splits`: seeded splits per unit (default 200), seeds 1 to N, each made by `day_split`
with the matrix's calibration-days rule (`day_split.calibration_count`: `n_calibration` days, or
half of the unit's days under `calibration_half_below`). The truth tables are checked as
`gate_run.check_inputs` checks them (a verified truth build at the commit checked out now, of the
tables' setting, with the tables' hashes).

`--kappa <level>`: the dial family measured at that level of plan section 2 instead of its own
(`family_at`, reading K2), so that plan section 7's rule for kappa ("largest calibrated
development offer") can be measured with the dial's rule, one run per level, and read jointly by
`kappa_choice`. The configuration without rows is the matrix's own configuration at that level
that agrees with the family's in every other field, found by its fields (the committed matrix has
one at each of 0.05, 0.1, 0.2 and 0.3); each dial configuration and its oracle partner are the
family's with only `kappa` changed, named with their one `-k<level>-` part changed to the new
level (a name without that part, or a new name the matrix already has, is refused). The model is
loaded, checked and fitted as without the option and predicts at the levels up to the level
given (`make_factory`).
At the family's own level the option changes none of the four outputs below. At another level the
results record how the family was made (`definitions.kappa_option`) and the summary prints it.
With the option the run also writes `dial-choice-certificate-timeouts.json`
(`CERTIFICATE_TIMEOUTS_NAME`): per unit that can be split and configuration, the days whose
certificate timed out, which the results count over the whole run only; `kappa_choice` prints
them per pair of kappa and dial.

Readings of `--kappa`:
  K1. The driver-risk rows at the lowest level, 0.05. Plan section 2 takes them "over the levels
      kappa_1 < ... < kappa_K <= kappa", so at 0.05 there is one level, kappa_1 = 0.05, with
      pi_1 = 0.05 and D_1 the level-0.05 departure bound, which is the presence bound d_hat
      itself. Then E_1(u) is the window energy plus the recovery supply at d_hat, and the need row
      of the forecast set holds it at or above the need, which is at least need_1; so the row
      holds on the whole forecast set at every dial (the argument of `driver_risk`'s module text
      for the oracle departure). Each pair (0.05, dial) is defined and measured like every other
      pair, and its set is the set without rows. [Alternative: the rows run over the levels
      strictly below kappa, none at 0.05, so the dial sets no row there and the pairs (0.05,
      dial) are not defined; they would then be printed "not defined", never as an offer of 0.]
  K2. The family at another level is the committed family with only `kappa` changed, and the
      matrix's own configuration without rows at that level. [Alternative: dial configurations
      at every level written into the run matrix; not done, the committed matrix is unchanged.]

`--what-if-kappa <level>`: a WHAT-IF, NOT A RUN OF THE PLAN. The dial family measured at a level
of `driver_risk.WHAT_IF_LEVELS` (0.6 and 0.7, off plan section 2's grid) or at a level of
the grid, by the steps of `--kappa`, with these changes (readings W1 to W4). Plan section 7 takes
kappa as the "largest calibrated development offer" over the grid; a largest offer at the grid's
top does not show where the largest offer over higher levels lies, and the calibration of plan
section 6 holds at any level, so higher levels are measured here as development evidence only.
`--kappa` and `--what-if-kappa` are not given together; `--kappa` keeps refusing every level off
the grid.
- The results' label, the key `what_if` (`what_if_record`) and the summary's first line say
  "what-if, level not on the plan's grid" (at a level off the grid) or "what-if, not a run of the
  plan" (at a grid level), and every file has its own name (`WHAT_IF_NAMES`): no file named
  `dial-choice.json` is written, so no pass-mark tool and no reader of the plan's runs reads it.
  `kappa_choice` reads such runs apart from the grid's (its reading K9).
- The forecast table may hold the what-if levels (`forecast_model.check_table`, `what_if`).

Readings of `--what-if-kappa`:
  W1. The driver-risk rows at a what-if level run over the plan's levels and the what-if levels
      at or below it (`driver_risk.levels_up_to`): at 0.7 over 0.05, 0.1, 0.2, 0.3, 0.4,
      0.5, 0.6 and 0.7. [Alternative: over the plan's levels and kappa itself only (at 0.7: 0.05 to
      0.5 and 0.7). At 0.6 the two are the same levels. At 0.7 they differ: the alternative
      gives the top level the mass from 0.5 up, and the top level, at the presence bound, prices no
      shortfall (reading K1's argument), so it prices less of an early departure.]
  W2. The configuration without rows at a what-if level, which the matrix does not hold, is the
      family's configuration without rows with only `kappa` changed, and its oracle partner
      likewise, named with their `-k<level>-` part changed, as the dial configurations are (K2).
      At a grid level the matrix's own is taken, as with `--kappa`: also at 0.4 and 0.5, which are
      grid levels (the what-if is then the comparison at the same fit, as at 0.3) and which a
      matrix that does not list them refuses. [Alternative: configurations
      at the what-if levels written into the run matrix; not done, the matrix is unchanged.]
  W3. The model is built (`make_model`) and fitted at every level of the grid and of the what-if
      list (`FIT_LEVELS_WHAT_IF`), whatever the level measured, so every what-if run of a model
      uses one fit. A model with one booster per level (`models.lgbm`, `models.lgbm_tail`) gives
      the same forecasts at the grid's levels as when fitted at the grid alone; a network with one
      energy output per level trained together (`models.hazard` and the models built on it) is a
      different network, so its what-if run at a grid level (`--what-if-kappa 0.3`) is the
      comparison at the same fit. [Alternative: fit at the grid and the what-if levels up to the
      level measured; each level would then have its own network.]
  W4. The calibration, the scales, the splits, the tiers and the gate reading are those of the
      grid's runs; nothing else changes.

`--what-if-fit-seed <N>` and `--what-if-fit-levels all|grid`: WHAT-IF OPTIONS FOR THE FIT OF THE
NEURAL HAZARD FAMILY (models whose `settings` are `HazardSettings`: `hazard`, `hazard_tuned`,
`hazard_tuned_inputs` and the ablations built on them), asked because the energy outputs of all
levels are trained together in one network (reading W3), so the network fitted at the plan's six
levels and the network fitted at all eight differ, and nothing says whether a difference between
their offers comes from the extra outputs or from the one seed each was fitted with. Either option
(or both) may be given with `--kappa`, with `--what-if-kappa` or with neither (the family's own
level); a model outside the family ends the command with exit 2 and a plain message.
- `--what-if-fit-seed N` (a whole number, 0 or more): the network is fitted with seed N instead of
  the settings' seed (`HazardSettings.seed`: the initial weights, the order of the sets, the
  draws); every other setting is the model's.
- `--what-if-fit-seeds N1,N2,...` (two or more distinct whole numbers, 0 or more, separated by
  commas): the forecast is the average over the listed seeds' networks (`seed_average`, reading
  A1: the mean of the members' survival curves, the bounds read from that mean curve, the energy
  columns the mean of the members' values), each network built and fitted exactly as
  `--what-if-fit-seed N` builds and fits it, one after another. The seeds are put in increasing
  order, so the order of the list changes no number. Not given with `--what-if-fit-seed`.
- `--what-if-fit-levels grid|all`: the levels the network is fitted at, whichever level is read:
  `grid` the plan's six (`driver_risk.LEVELS`, built from the run matrix, which must list exactly
  those), `all` the eight (`FIT_LEVELS_WHAT_IF`). Without the option `--kappa` and neither option
  fit at the matrix's levels (the grid, once the matrix lists its six) and `--what-if-kappa` at
  all eight (reading W3, unchanged). `grid` with a level
  off the grid ends the command with exit 2: a network fitted at the grid has no output there.
A run with either option is a what-if: it is not a run of the plan and no pass-mark
tool reads it. It writes the files `WHAT_IF_NAMES` and no others (with the certificate timeouts of
a `--kappa` run), its label and key `what_if` say so, and `what_if.fit` records the seed, the
levels fitted and where each came from (with `--what-if-fit-seeds`, the key `seeds` holds the
sorted seed list in place of `seed`, and `averaged` says what is averaged); the summary's head
prints them. Everything else is that
of the run without the option at the same level: the table, the calibration and the splits. The
fit with the same seed and levels repeats byte for byte (`hazard_fit`).

Readings of the fit options:
  W5. The seed given replaces the settings' seed for the one fit of the run and nothing else; the
      network is not refitted at several seeds in one run. A list of seeds averaged in one run
      is the separate option `--what-if-fit-seeds` (reading A1 of `seed_average`), a fit of its
      own: `kappa_choice` reads its runs apart from single-seed fits and from other seed lists.
  W6. `--what-if-fit-levels grid` means the matrix's own levels, taken to be the plan's six and
      checked to be so (the committed matrix lists four, so `grid` is refused with it until the
      matrix lists the six). [Alternative: the plan's six whatever the matrix lists; not done, the
      matrix is what a model is built from (`make_model`).]

The earlier forecasts (`models.earlier_gbm`, `models.earlier_population`; plan section 7: each of
the two models and each of the two earlier forecasts run through the full method "gets its own κ
... and its own dial ε (Section 6), by the same rules"). Each gets its own kappa and dial by the
same commands as `lgbm_tail` and `hazard_tuned_inputs`: this command with `--model earlier_gbm`
(or `earlier_population`) and `--kappa` at each level of plan section 2's grid, with the same
matrix (its dial family, its six levels and every shared value), splits, seeds, scales, tiers,
rung and rule, then `kappa_choice` over that model's six runs. No line of this command or of
`kappa_choice` depends on the model's name, and none was added for them: the matrix is unchanged.
What their own form changes, each a reading:
  M1. Their levels. Both forecast at the 16 levels of `earlier_grid.LEVELS` whatever levels the
      matrix lists (`make_model` reads only rho) and read each level of the table off that grid
      (`earlier_grid.at_level`). The six levels of the grid and the what-if levels 0.6 and 0.7
      are among the 16, so no level is interpolated, and, as with one booster per level (W3),
      the forecast at a level does not depend on the levels fitted: a what-if run at a grid level
      gives the table of the run at that level. [Alternative: fit them at the matrix's levels
      only, which would change the earlier models' own rule of 16 levels.]
  M2. Their fit. `earlier_population` has no fitted parameter and no seed: its fit is the
      training sessions' survival curve and the empirical quantiles of their rates and rooms, the
      same on every run. `earlier_gbm`'s trees use the earlier analysis's fixed seed 0, and their
      sums run on every core (its reading E12), so two runs on two machines may differ in the
      last digits. Neither is of the neural hazard family, so the fit options
      (`--what-if-fit-seed`, `--what-if-fit-seeds`, `--what-if-fit-levels`) end the command with
      exit 2 for them, as for `lgbm_tail`. [Alternative: a seed option for `earlier_gbm`'s trees;
      not built, the earlier settings fix the seed.]
  M3. Their departure. No survival columns (`earlier_grid.table`): the departure bound and every
      level-kappa_k bound D_k of the driver-risk rows is the forecast at the decision time, kept
      in every hour, as for `models.lgbm`. The stay is capped at the earlier horizon of 10 hours
      (`earlier_grid.HORIZON_H`), so no bound lies more than 10 hours after the decision time.
      [Alternative: none without changing the earlier models.]
  M4. A car without a forecast. `earlier_population` forecasts nothing (NaN) for a car plugged
      in longer than every training session lasted; such a car has no forecast and is left
      unsteered, as a NaN of any model's table is read (`forecast_model`). [Alternative: refuse
      the run, which no other model's car without a forecast does.]

Configurations, found by their fields: the dial family is every configuration with the empirical
departure, departures seen from the rounded-down departure hour (the main reading) and a number
for `driver_risk`; they must agree in every other field but name, family and oracle partner, and
the dial values must differ. The configuration without rows is the one configuration with those
fields and `driver_risk` null. A matrix without a dial family, or with a family that does not fit
this description, is refused.

Per unit that can be split (the rule leaves a calibration day and a test day) and configuration:
every day is prepared (`gate_day.prepare_day`); every day that offers is evaluated once at each
scale of `dial_splits.GRID` (`crc.GRID` up to and including 1: the stay-to-bound property is
claimed up to 1 only) through `gate_day.evaluate`, with `gate_calibrate.DayCache`, so no (day,
scale) is evaluated twice. The one-way rung's day-level slot and driver losses at each scale are
kept. The evaluation runs every rung up to the two-way one, as `gate_day.evaluate` always does;
only the one-way rung is read. Per split and tier the calibration and scoring are `dial_splits`:
tier A (slot loss at alpha 0.10 and driver loss at alpha 0.05) and tier B (the slot loss alone),
never pooled.

Gate reading (`gate_reading`, `dial_splits.gate_split`; definitions.gate_reading in the results):
for each unit that can be split and each configuration, over all the splits and for tier A, the mean
calibrated offer, the mean of each ceiling of `report.ceilings` and the offer as a share of it, the
oracle partner's mean offer (its days prepared and measured like the configuration's own, its
scale calibrated by the same rule on the same splits) and its ratio to the mean offer. The plan's
thresholds are printed beside the figures; no verdict is formed.

Timeouts are recorded outcomes. A day whose certificate times out abstains (`gate_day`): it offers
0 at every scale and is in no mean. A re-solve that times out leaves the rung on its predecessor's
plan (`rungs`), and the day enters with the losses that replay gives; the numbers of such
re-solves are counted once per evaluated (day, scale). A missing best-causal ceiling (a timeout)
leaves the ceiling share of the splits whose test days include that day out.

`--made-after-sealed-run` (`after_sealed_run`): a trial on development units made after the sealed
run's results were seen. The results (key `made_after_sealed_run`, in the results file and the
per-split file, and the label), the first lines of the summary and the file
`after_sealed_run.FILE_NAME` of the run folder then say so. Without it nothing changes.

A rotation what-if (the matrix field `rotation_whatif`, `gate_config`) scores a training unit with
a model not fitted on it (`models.lgbm_tail_rotation`); the results (`label`, `definitions` and the
key `rotation_whatif`) and the first lines of the summary say that it is not a run of the plan,
name the units left out of the fit and say that the figures are development evidence
(`rotation_whatif`). A model fitted on a scored unit ends the command with exit 2.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary also on
standard output:
- `dial-choice.json`: every number (`dial_choice.RESULTS_NAME`), with `definitions`;
- `dial-choice-summary.txt`: one table per unit and tier and one line per unit and tier naming
  the dial with the largest mean offer;
- `dial-choice-forecasts.csv`: the model's forecast table (its hash is in the results);
- `dial-choice-timing.json`: wall-clock seconds, kept apart from the compared results;
- with `--kappa` only, `dial-choice-certificate-timeouts.json`: the run's kappa and the
  certificate timeouts per unit and configuration;
- `dial-choice-split-offers.json` (`SPLIT_OFFERS_NAME`): the per-split values behind the
  summaries, per unit, configuration and tier one entry per split (seed, calibrated offer per
  counting test day, grid scale), with the run's identity (model, fit units, matrix and its hash,
  truth inputs, the run's seeds, kappa and dials, the label and the what-if record when there is
  one) and the SHA-256 of the results file it belongs to; the mean of a cell's offers is that
  cell's `mean_offer_kwh_per_test_day` (`dial_splits.summarise`);
- with `--what-if-kappa` or a fit option, the same files under the names `WHAT_IF_NAMES`
  (`dial-choice-what-if.json` and so on) and no other.

Exit status: 0 every configuration ran; 1 the model was refused or failed, or a configuration
failed; 2 could not run (a bad matrix, argument or model name, a matrix without the dial family,
a model fitted on a scored unit, a table missing, no verified truth build at this commit, a table
differing from it; with `--kappa`, a level outside plan section 2's grid or a family that cannot
be made at it; with `--what-if-kappa`, a level outside the grid and the what-if list, or both
options given; with a fit option, a model outside the neural hazard family, a seed below 0, or
`--what-if-fit-levels grid` with a level off the grid or a matrix whose levels are not the
plan's six; with `--what-if-fit-seeds`, fewer than two seeds, a seed listed twice, a seed that is
not a whole number of 0 or more, or `--what-if-fit-seed` given too).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve import (
    after_sealed_run,
    crc,
    dial_splits,
    dial_summary,
    driver_risk,
    gate_run,
    learn_run,
    model_inputs,
    rotation_whatif,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import TableFactory, check_table, write_table
from fleetreserve.gate_calibrate import DayCache, ceiling_values
from fleetreserve.gate_config import (
    TIER_LABELS,
    Configuration,
    Matrix,
    MatrixError,
    _check_partners,
    load,
)
from fleetreserve.gate_stat import RATIO_THRESHOLD, SHARE_THRESHOLD
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.rungs import RUNGS
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

MATRIX = TOP / "gate" / "matrix.json"
RESULTS_NAME = "dial-choice.json"
SUMMARY_NAME = "dial-choice-summary.txt"
TABLE_NAME = "dial-choice-forecasts.csv"
TIMING_NAME = "dial-choice-timing.json"
CERTIFICATE_TIMEOUTS_NAME = "dial-choice-certificate-timeouts.json"
SPLIT_OFFERS_NAME = "dial-choice-split-offers.json"
PLAN_NAMES = {
    "results": RESULTS_NAME,
    "summary": SUMMARY_NAME,
    "table": TABLE_NAME,
    "timing": TIMING_NAME,
    "certificate_timeouts": CERTIFICATE_TIMEOUTS_NAME,
    "split_offers": SPLIT_OFFERS_NAME,
}
WHAT_IF_NAMES = {
    "results": "dial-choice-what-if.json",
    "summary": "dial-choice-what-if-summary.txt",
    "table": "dial-choice-what-if-forecasts.csv",
    "timing": "dial-choice-what-if-timing.json",
    "certificate_timeouts": "dial-choice-what-if-certificate-timeouts.json",
    "split_offers": "dial-choice-what-if-split-offers.json",
}
FIT_LEVELS_WHAT_IF = (*driver_risk.LEVELS, *driver_risk.WHAT_IF_LEVELS)  # reading W3
FIT_LEVEL_CHOICES = ("grid", "all")  # `--what-if-fit-levels`
AVERAGED = (
    "the forecast is the average over the seeds' networks, each fitted as --what-if-fit-seed "
    "fits it: the mean of the members' survival curves (a mixture with equal weights, a proper "
    "distribution), the departure bounds read from that mean curve, the energy columns the mean "
    "of the members' values (seed_average reading A1)"
)
OFF_GRID = "what-if, level not on the plan's grid"
ON_GRID = "what-if, not a run of the plan"
SPLITS = 200
RUNG = RUNGS[1]  # the one-way re-solve
MAIN_READING = "rounded-hour"  # `Configuration.departure_seen` of the main reading
LABEL = dial_summary.LABEL


# --- the configurations --------------------------------------------------------------------------


def _same_but_dial(c: Configuration) -> Configuration:
    return replace(c, name="", family="", oracle_partner=None, driver_risk=None)


def find_family(matrix: Matrix) -> tuple[Configuration, tuple[Configuration, ...]]:
    """(the configuration without rows, the dial configurations in increasing dial) of the matrix
    (module text); raises `MatrixError` when there is no such family."""
    candidates = [
        c
        for c in matrix.configurations
        if c.departure == "empirical" and c.departure_seen == MAIN_READING
    ]
    dials = [c for c in candidates if c.driver_risk is not None]
    if not dials:
        raise MatrixError(
            "the matrix has no dial family: no configuration with the empirical departure, "
            f"departures seen from the {MAIN_READING} hour and a number for driver_risk"
        )
    shapes = {_same_but_dial(c) for c in dials}
    if len(shapes) != 1:
        raise MatrixError("the dial configurations differ in a field other than the dial")
    values = [c.driver_risk for c in dials]
    if len(set(values)) != len(values):
        raise MatrixError("two dial configurations have the same dial")
    (shape,) = shapes
    plain = [c for c in candidates if c.driver_risk is None and _same_but_dial(c) == shape]
    if len(plain) != 1:
        raise MatrixError(
            f"the dial family needs exactly one configuration without driver-risk rows that "
            f"agrees with it in every other field; found {len(plain)}"
        )
    return plain[0], tuple(sorted(dials, key=lambda c: c.driver_risk))


def _name_at(name: str, old: float, new: float) -> str:
    part = f"-k{old:g}-"
    if name.count(part) != 1:
        raise MatrixError(
            f"the configuration name {name!r} does not hold its level once as {part!r}, so its "
            f"name at kappa {new:g} cannot be made"
        )
    return name.replace(part, f"-k{new:g}-")


def _made_at(matrix: Matrix, plain: Configuration, kappa: float) -> list:
    """The configuration without rows `plain` and its oracle partner with only `kappa` changed
    and their names made for it (reading W2)."""
    made = replace(plain, kappa=kappa, name=_name_at(plain.name, plain.kappa, kappa))
    if plain.oracle_partner is None:
        return [made]
    partner = matrix.configuration(plain.oracle_partner)
    partner_at = replace(partner, kappa=kappa, name=_name_at(partner.name, plain.kappa, kappa))
    return [replace(made, oracle_partner=partner_at.name), partner_at]


def family_at(matrix: Matrix, family, kappa: float, what_if: bool = False) -> tuple[Matrix, tuple]:
    """(the matrix with the configurations made for `kappa` added, the family at `kappa`) (module
    text, `--kappa`, reading K2; with `what_if`, `--what-if-kappa`, reading W2); at the family's
    own level both unchanged. Raises `MatrixError`."""
    plain, dials = family
    allowed = FIT_LEVELS_WHAT_IF if what_if else driver_risk.LEVELS
    option = "--what-if-kappa" if what_if else "--kappa"
    if isinstance(kappa, bool) or kappa not in allowed:
        raise MatrixError(f"{option} must be one of {list(allowed)}, got {kappa!r}")
    if kappa == plain.kappa:
        return matrix, family
    shape = _same_but_dial(replace(plain, kappa=kappa))
    plains = [
        c for c in matrix.configurations if c.driver_risk is None and _same_but_dial(c) == shape
    ]
    made = []
    if what_if and not plains and kappa not in driver_risk.LEVELS:
        made = _made_at(matrix, plain, kappa)
        plains = made[:1]
    if len(plains) != 1:
        raise MatrixError(
            f"the matrix needs exactly one configuration without driver-risk rows at kappa "
            f"{kappa:g} that agrees with the dial family in every other field; found {len(plains)}"
        )
    dials_at = []
    for dial in dials:
        partner = matrix.configuration(dial.oracle_partner)
        partner_at = replace(partner, kappa=kappa, name=_name_at(partner.name, plain.kappa, kappa))
        dial_at = replace(
            dial,
            kappa=kappa,
            name=_name_at(dial.name, plain.kappa, kappa),
            oracle_partner=partner_at.name,
        )
        made += [dial_at, partner_at]
        dials_at.append(dial_at)
    taken = sorted({c.name for c in matrix.configurations} & {c.name for c in made})
    if taken:
        raise MatrixError(f"the matrix already has configurations named {taken}")
    configurations = (*matrix.configurations, *made)
    _check_partners(configurations)
    return replace(matrix, configurations=configurations), (plains[0], tuple(dials_at))


def what_if_record(family_measured, fit: dict | None = None) -> dict:
    """The key `what_if` of a what-if run's results (module text): what the run is, its level, the
    plan's grid, the levels fitted and the levels of the driver-risk rows. With `fit` (a run with
    a fit option, `fit_record`) the levels fitted are the run's own, the text names the seed and
    the record holds `fit`."""
    plain, dials = family_measured
    kappa = plain.kappa
    on_grid = kappa in driver_risk.LEVELS
    grid = ", ".join(f"{k:g}" for k in driver_risk.LEVELS)
    levels_fitted = FIT_LEVELS_WHAT_IF if fit is None else tuple(fit["levels_fitted"])
    fitted = ", ".join(f"{k:g}" for k in levels_fitted)
    seed = "" if fit is None else f" and {dial_summary.seed_words(fit)}"
    reading = "W3" if fit is None else "W3, W5 and W6"
    if fit is not None and "seeds" in fit:
        reading = "W3, W5, W6 and seed_average reading A1"
    if on_grid:
        what = (
            f"{ON_GRID}: kappa {kappa:g}, a level of the plan's grid, measured with the model "
            f"fitted at the levels {fitted}{seed} (dial_choice reading {reading})"
        )
    else:
        what = (
            f"{OFF_GRID}: kappa {kappa:g} (plan section 2's grid: {grid}); not a run of the "
            f"plan, development evidence only; the model fitted at the levels {fitted}{seed}"
        )
    fitted_text = (
        "the model fitted at every level of the grid and of the what-if list"
        if fit is None
        else f"the model fitted at the levels {fitted} with {dial_summary.seed_words(fit)} "
        + ("(W5, W6)" if "seeds" not in fit else f"(W5, W6; {fit['averaged']})")
    )
    record = {
        "what": what,
        "label": f"{what}; {LABEL}",
        "kappa": kappa,
        "on_the_plans_grid": on_grid,
        "plan_grid": list(driver_risk.LEVELS),
        "what_if_levels": list(driver_risk.WHAT_IF_LEVELS),
        "levels_fitted": list(levels_fitted),
        "driver_risk_levels": list(driver_risk.levels_up_to(kappa)),
        "configuration_without_rows": plain.name,
        "dial_configurations": [c.name for c in dials],
        "readings": (
            "dial_choice readings W1 to W4: the driver-risk rows over the plan's levels and the "
            "what-if levels at or below kappa; the configuration without rows and the oracle "
            "partners off the grid made from the family with only kappa changed; "
            f"{fitted_text}; everything else as on the grid"
        ),
    }
    if fit is not None:
        record["fit"] = fit
    return record


def fit_record(
    model,
    seed_option: int | None,
    levels_option: str | None,
    what_if: bool,
    seeds_option: tuple[int, ...] | None = None,
) -> dict:
    """The key `fit` of a fit-option run's `what_if` record: the seed and the levels the network
    is fitted at, read from the model itself, and where each came from (module text). With
    `seeds_option` (`--what-if-fit-seeds`) the key `seeds` holds the members' seeds, read from
    each member, in place of `seed`."""
    if seeds_option is not None:
        return {
            "seeds": [m.settings.seed for m in model.members],
            "seed_from": "--what-if-fit-seeds",
            "averaged": AVERAGED,
            **_levels_record(model.members[0], levels_option, what_if),
        }
    return {
        "seed": model.settings.seed,
        "seed_from": "--what-if-fit-seed" if seed_option is not None else "the settings' seed",
        **_levels_record(model, levels_option, what_if),
    }


def _levels_record(model, levels_option: str | None, what_if: bool) -> dict:
    if levels_option is None:
        levels_from = (
            "default of --what-if-kappa: all eight" if what_if else "default: the matrix's levels"
        )
    else:
        levels_from = f"--what-if-fit-levels {levels_option}"
    return {
        "levels_fitted": list(model.levels["rate"]),
        "levels_from": levels_from,
        "energy_levels_per_quantity": len(model.levels["rate"]),
    }


def parse_seeds(text: str) -> tuple[int, ...]:
    """The seeds of `--what-if-fit-seeds` in increasing order; `MatrixError` unless `text` lists
    two or more distinct whole numbers of 0 or more, separated by commas (module text)."""
    parts = [p.strip() for p in text.split(",")]
    if not all(p.isascii() and p.isdecimal() for p in parts):
        raise MatrixError(
            "--what-if-fit-seeds takes whole numbers of 0 or more separated by commas, "
            f"got {text!r}"
        )
    seeds = [int(p) for p in parts]
    if len(seeds) < 2:
        raise MatrixError(
            f"--what-if-fit-seeds averages two or more seeds, got {text!r}; one seed is "
            "--what-if-fit-seed"
        )
    if len(set(seeds)) != len(seeds):
        raise MatrixError(f"--what-if-fit-seeds lists a seed twice: {text!r}")
    return tuple(sorted(seeds))


def check_fit_options(
    matrix: Matrix,
    seed: int | None,
    levels: str | None,
    kappa: float | None,
    what_if: bool,
    seeds: str | None = None,
) -> tuple[int, ...] | None:
    """Raises `MatrixError` for a fit option the run cannot take (module text): a seed below 0,
    or `grid` with a level off the grid, or with a matrix whose levels are not the plan's six,
    or a seed list `parse_seeds` refuses or given with a single seed. `kappa`: the level of
    `--what-if-kappa` (read only with `what_if`). Returns the sorted seeds of `seeds`, or None."""
    if seed is not None and seeds is not None:
        raise MatrixError("give --what-if-fit-seed or --what-if-fit-seeds, not both")
    if seed is not None and seed < 0:
        raise MatrixError(f"--what-if-fit-seed must be a whole number of 0 or more, got {seed}")
    listed = None if seeds is None else parse_seeds(seeds)
    if levels == "grid":
        if what_if and kappa not in driver_risk.LEVELS:
            raise MatrixError(
                f"--what-if-fit-levels grid cannot read kappa {kappa:g}: a network fitted at "
                f"the plan's grid {list(driver_risk.LEVELS)} has no output there"
            )
        own = sorted({c.kappa for c in matrix.configurations})
        if own != list(driver_risk.LEVELS):
            raise MatrixError(
                f"--what-if-fit-levels grid needs a matrix whose levels are the plan's grid "
                f"{list(driver_risk.LEVELS)}, got {own}"
            )
    return listed


def averaged_model(build, name: str, seeds: tuple[int, ...]):
    """The model of `--what-if-fit-seeds` (module text, `seed_average`): one member per seed,
    each made by `build()` (a fresh model, as a single-seed run loads it) and given its seed by
    `with_fit_seed`. `MatrixError` for a model outside the neural hazard family, before a second
    member is built."""
    members = [with_fit_seed(build(), name, seeds[0])]
    members += [with_fit_seed(build(), name, seed) for seed in seeds[1:]]
    from fleetreserve import seed_average  # the learn group's packages, only on this path

    return seed_average.average_of(members)


def with_fit_seed(model, name: str, seed: int | None):
    """`model` with the seed of its settings replaced by `seed` (None: unchanged); `MatrixError`
    when `model` is not of the neural hazard family (module text). The settings are a frozen
    record, so the model's own are replaced, not changed."""
    if not isinstance(getattr(model, "settings", None), HazardSettings):
        raise MatrixError(
            "--what-if-fit-seed and --what-if-fit-levels are for the neural hazard family "
            f"(models with HazardSettings); {name} is not one"
        )
    if seed is not None:
        model.settings = replace(model.settings, seed=seed)
    return model


def fit_matrix(matrix: Matrix, plain: Configuration) -> Matrix:
    """The matrix a what-if run builds its model from (reading W3): the matrix with one made-up
    configuration at each level of `FIT_LEVELS_WHAT_IF` it lacks, so that a model's `make_model`,
    which takes its levels from the matrix's configurations, fits every level. It is used for
    building the model only."""
    have = {c.kappa for c in matrix.configurations}
    added = tuple(
        replace(plain, kappa=k, name=f"what-if-fit-level-{k:g}", oracle_partner=None)
        for k in FIT_LEVELS_WHAT_IF
        if k not in have
    )
    return replace(matrix, configurations=(*matrix.configurations, *added))


def kappa_option(own: float, family_measured) -> str:
    """The text of `definitions.kappa_option`: the matrix's family, of level `own`, measured as
    `family_measured` (`family_at`) at another level."""
    plain = family_measured[0]
    return (
        f"--kappa {plain.kappa:g}: the dial family of the matrix (kappa {own:g}) measured at "
        f"kappa {plain.kappa:g}: the configuration without rows is the matrix's {plain.name}, "
        "and each dial configuration and its oracle partner are the family's with only kappa "
        f"changed (dial_choice reading K2); the model predicts at the levels up to "
        f"{plain.kappa:g}"
    )


# --- the forecaster ------------------------------------------------------------------------------


def make_factory(
    name: str,
    model,
    matrix: Matrix,
    units: dict,
    fit_frames: dict,
    kappa: float,
    what_if: bool = False,
):
    """The forecaster factory of model `name` (module text): the model is checked and fitted as
    `learn_run` does, and predicts at the levels of the dial rows up to `kappa` (with `what_if`, a
    table at what-if levels is taken). Returns (factory, table). Raises `learn_run.Refused` and
    `learn_run.ModelFailed`."""
    fit_units = tuple(model.fit_units)
    outcome_reader = learn_run.check_model(name, model, fit_units, matrix.gate_units)
    frames = list(fit_frames.values())
    learn_run._call(
        "fit", model.fit, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    extra = model_inputs.extra_of(model)
    frame = learn_run.forecast_frame(outcome_reader, units, matrix.gate_units, extra)
    levels = list(driver_risk.levels_up_to(kappa))
    table = check_table(learn_run._call("predict", model.predict, frame, levels), what_if)
    return TableFactory(table, what_if), table


# --- one unit under one configuration ------------------------------------------------------------


def measure(prepared, best_causal, evaluate_batch) -> tuple[dial_splits.UnitLosses, DayCache]:
    """The one-way rung's day-level losses of every day that offers at each scale of the grid
    (module text), with the cache that made the evaluations (`cache.calls` counts them)."""
    cache = DayCache(prepared, evaluate_batch)
    days, scales = len(prepared), len(dial_splits.GRID)
    offering = [d for d, p in enumerate(prepared) if not p.abstains]
    slot, driver = np.zeros((days, scales)), np.zeros((days, scales))
    for k, scale in enumerate(dial_splits.GRID):
        for d, found in zip(offering, cache.get(offering, scale), strict=True):
            slot[d, k] = found.rungs[RUNG].slot_loss
            driver[d, k] = found.rungs[RUNG].driver_loss
    losses = dial_splits.UnitLosses(
        value=np.array([p.value for p in prepared], dtype=float),
        offering=np.array([not p.abstains for p in prepared], dtype=bool),
        slot=slot,
        driver=driver,
        ceiling=np.array([np.nan if c is None else c for c in best_causal], dtype=float),
    )
    return losses, cache


def split_indices(name: str, days, matrix: Matrix, seeds) -> list[tuple[int, list, list]]:
    """(seed, calibration day indices, test day indices) of each seed's split of the unit, made by
    `gate_run.plan_unit` (so `day_split`) with the matrix's calibration-days rule."""
    found = []
    for seed in seeds:
        roles = gate_run.plan_unit(name, days, replace(matrix, split_seed=seed)).roles
        found.append(
            (
                seed,
                [d for d, role in enumerate(roles) if role == "calibration"],
                [d for d, role in enumerate(roles) if role == "test"],
            )
        )
    return found


def score_each(losses: dial_splits.UnitLosses, splits, tiers) -> dict:
    """Per tier the score of every split (`dial_splits.score_split`), in the order of `splits`."""
    return {
        tier: [dial_splits.score_split(losses, seed, cal, test, tier) for seed, cal, test in splits]
        for tier in tiers
    }


def summarise_scores(scores: dict) -> dict:
    """Per tier the summary of the scores of `score_each` (`dial_splits.summarise`)."""
    return {tier: dial_splits.summarise(found, tier) for tier, found in scores.items()}


def score_splits(losses: dial_splits.UnitLosses, splits, tiers) -> dict:
    """Per tier the summary over the splits (`dial_splits.summarise`)."""
    return summarise_scores(score_each(losses, splits, tiers))


def split_offers_entry(scores: dict) -> dict:
    """Per tier the per-split values of `score_each`: the split's seed,
    the calibrated offer per counting test day and the grid scale (0 when it abstains), in the
    order of the splits, so the mean of the offers is the summary's mean offer."""
    return {
        tier: [{"seed": s.seed, "offer_kwh": s.offer_kwh, "scale": s.scale} for s in found]
        for tier, found in scores.items()
    }


def gate_reading(
    runner, name, partner, losses, ceilings, best_causal, splits, matrix, timeouts: Counter
) -> dict:
    """The figures of the gate reading (module text) of one unit and configuration: the oracle
    partner's days are prepared and measured like the configuration's own, and the splits are read
    by `dial_splits.gate_split`. `ceilings`: ceiling name -> per-day array (NaN where missing). A
    failure is returned as {"failed": reason} and the other numbers stay."""
    try:
        prepared = runner.map(
            gate_run._prepare_task, [(name, i, partner) for i in range(len(best_causal))]
        )
        oracle, cache = measure(
            prepared, best_causal, lambda pairs: runner.map(gate_run._evaluate_task, pairs)
        )
    except Exception as problem:  # recorded, the other numbers stay
        return {"failed": f"{type(problem).__name__}: {problem}", "oracle_partner": partner.name}
    timeouts["oracle_certificates"] += sum(p.certificate_outcome == "timeout" for p in prepared)
    timeouts["oracle_re_solves"] += _evaluations(cache)["re_solves_of_the_one_way_rung"].get(
        "timeout", 0
    )
    found = dial_splits.summarise_gate(
        [
            dial_splits.gate_split(losses, oracle, ceilings, seed, cal, test)
            for seed, cal, test in splits
        ],
        matrix.ceilings,
    )
    found["oracle_partner"] = partner.name
    return found


def _evaluations(cache: DayCache) -> dict:
    outcomes: Counter = Counter()
    for found in cache.store.values():
        outcomes.update(found.rungs[RUNG].resolves)
    return {
        "day_scale_pairs": len(cache.calls),
        "pairs_evaluated_more_than_once": sum(1 for n in cache.calls.values() if n > 1),
        "re_solves_of_the_one_way_rung": dict(sorted(outcomes.items())),
    }


# --- one run -------------------------------------------------------------------------------------


def definitions(matrix: Matrix, family, splits: int) -> dict:
    plain, dials = family
    return {
        "scales": (
            "crc.GRID up to and including 1; scales above 1 are not used because the "
            "stay-to-bound property is claimed up to 1 only"
        ),
        "splits": (
            f"{splits} seeded splits per unit (seeds 1 to {splits}), each made by day_split with "
            f"the calibration-days rule (n_calibration {matrix.n_calibration}, "
            f"calibration_half_below {matrix.calibration_half_below}); the seed "
            f"{matrix.split_seed} is the split of the gate run"
        ),
        "on_grid_calibration": (
            "per split and tier, the largest grid scale at which the calibration bound of crc "
            "holds on the split's calibration days that offer, with each day's losses made a "
            "running maximum over the grid scales; scale 0 (abstain) when too few of them offer "
            "or when the bound fails at the smallest grid scale"
        ),
        "tiers": {
            "A": f"{TIER_LABELS['A']}: slot loss at alpha {crc.ALPHA_SLOT} and driver loss at "
            f"alpha {crc.ALPHA_DRIVER}",
            "B": f"{TIER_LABELS['B']}: the slot loss alone; the driver loss is reported",
            "pooling": "the tiers are reported side by side and never pooled or averaged",
        },
        "mean_offer_kwh_per_test_day": (
            "per split, the scale times the mean certified value over the counting test days "
            "(0 on days that abstain and at scale 0); the mean and the points p10, p50, p90 are "
            "over splits, splits at scale 0 counting as 0; points are linear-interpolated"
        ),
        "test_losses": (
            "per split, the day-level losses of the one-way rung at the chosen scale averaged "
            "over the offering test days; the means are over the splits that offer and have an "
            "offering test day; 'over the limit' is a mean above 0.10 (slot) or 0.05 (driver)"
        ),
        "share_of_best_causal_ceiling": (
            "per split, the sum of the offers over the sum of the best-causal ceilings of the "
            "test days (left out for a split with a missing ceiling); the mean is over splits "
            "that offer"
        ),
        "gate_reading": (
            "tier A, per unit and configuration, over all the splits: the mean calibrated offer "
            "per counting test day (a split with no safe scale counts as 0); per ceiling of "
            "report.ceilings, the mean over the splits of the mean ceiling per counting test day "
            "and the share of it, mean offer over mean ceiling; the oracle-departure offer, the "
            "oracle partner of the configuration (the configuration's field oracle_partner, whose "
            "departure bounds are the cars' true departure hours) read at the same rung "
            "(one-way), tier and splits and calibrated by the same on-grid rule from its own "
            "losses, and its ratio to the mean offer (null when the mean offer is 0). The gate "
            "run's first decision takes the ceiling gate_ceiling (the first of "
            "report.ceilings) as the denominator of the share and the oracle partner's offer over "
            "the offer as the ratio, on one split; here the same quantities are read as means "
            "over all splits. A split whose ceiling is missing on a test day (a timeout) is left "
            "out of that ceiling's mean and share and counted, never read as 0. The plan's "
            "thresholds are listed in run.thresholds; no verdict is formed"
        ),
        "dial_choice": (
            "per unit and tier, the dial with the largest mean offer among the dial "
            f"configurations; a difference of at most {dial_splits.TIE} kWh is a tie and goes to "
            "the smaller dial; the configuration without rows is listed but is not a dial. "
            f"{LABEL}"
        ),
        "configuration_without_rows": plain.name,
        "dial_configurations": {f"{c.driver_risk:g}": c.name for c in dials},
        "timeouts": (
            "a day whose certificate times out abstains: it offers 0 at every scale and is in "
            "no mean; a re-solve that times out leaves the rung on its predecessor's plan and "
            "the day enters with the losses of that replay; a missing ceiling leaves the ceiling "
            "share of the splits that test on that day out"
        ),
    }


def run(
    matrix: Matrix,
    units: dict,
    factory,
    *,
    splits: int,
    workers: int,
    family=None,
    certificate_timeouts: dict | None = None,
    split_offers: dict | None = None,
):
    """Everything but the input checks and the writing of files: (results document, timing).
    `units`: unit name -> `UnitDays` for the matrix's gate units; `factory`: as `gate_run.run_gate`
    takes it. `certificate_timeouts`, when given, receives per unit that can be split and
    configuration measured the number of days whose certificate timed out (module text,
    `--kappa`). `split_offers`, when given, receives per unit that can be split and
    configuration measured the per-split values (`split_offers_entry`)."""
    started = time.perf_counter()
    plain, dials = family if family is not None else find_family(matrix)
    configurations = (plain, *dials)
    seeds = list(range(1, splits + 1))
    tiers = dial_splits.TIERS
    runner = gate_run.Runner(workers, units, matrix, factory)
    unit_entries, timeouts, failed, timing = {}, Counter(), {}, {"units": {}}
    try:
        for name in matrix.gate_units:
            days = units[name]
            plan = gate_run.plan_unit(name, days, matrix)
            entry = {
                "counting_days": len(plan.keys),
                "calibration_days_used": plan.calibration_days,
                "test_days": plan.test_days,
                "status": plan.status,
                "split_of_the_gate_run": plan.split,
                "configurations": {},
            }
            unit_entries[name] = entry
            if plan.status != "calibratable":
                continue
            begun = time.perf_counter()
            ceilings = runner.map(
                gate_run._ceiling_task, [(name, i) for i in range(len(plan.keys))]
            )
            timeouts["best_causal_ceilings"] += sum(
                c.best_causal_outcome == "timeout" for c in ceilings
            )
            best_causal = [c.best_causal for c in ceilings]
            ceiling_arrays = {
                ceiling: np.array(
                    [
                        np.nan if v is None else v
                        for v in (ceiling_values(c, ceiling) for c in ceilings)
                    ],
                    dtype=float,
                )
                for ceiling in matrix.ceilings
            }
            indices = split_indices(name, days, matrix, seeds)
            for configuration in configurations:
                one = time.perf_counter()
                try:
                    prepared = runner.map(
                        gate_run._prepare_task,
                        [(name, i, configuration) for i in range(len(plan.keys))],
                    )
                    losses, cache = measure(
                        prepared,
                        best_causal,
                        lambda pairs: runner.map(gate_run._evaluate_task, pairs),
                    )
                except Exception as problem:  # recorded, the other configurations go on
                    failed[(name, configuration.name)] = f"{type(problem).__name__}: {problem}"
                    entry["configurations"][configuration.name] = {
                        "driver_risk": configuration.driver_risk,
                        "failed": failed[(name, configuration.name)],
                    }
                    if split_offers is not None:
                        split_offers.setdefault(name, {})[configuration.name] = {
                            "driver_risk": configuration.driver_risk,
                            "failed": failed[(name, configuration.name)],
                        }
                    continue
                scores = score_each(losses, indices, tiers)
                if split_offers is not None:
                    split_offers.setdefault(name, {})[configuration.name] = {
                        "driver_risk": configuration.driver_risk,
                        "tiers": split_offers_entry(scores),
                    }
                found = _evaluations(cache)
                timed_out = sum(p.certificate_outcome == "timeout" for p in prepared)
                timeouts["certificates"] += timed_out
                if certificate_timeouts is not None:
                    certificate_timeouts.setdefault(name, {})[configuration.name] = timed_out
                timeouts["re_solves"] += found["re_solves_of_the_one_way_rung"].get("timeout", 0)
                entry["configurations"][configuration.name] = {
                    "driver_risk": configuration.driver_risk,
                    "offering_days": int(losses.offering.sum()),
                    "evaluations": found,
                    "tiers": summarise_scores(scores),
                    "gate_reading": gate_reading(
                        runner,
                        name,
                        matrix.configuration(configuration.oracle_partner),
                        losses,
                        ceiling_arrays,
                        best_causal,
                        indices,
                        matrix,
                        timeouts,
                    ),
                }
                reading = entry["configurations"][configuration.name]["gate_reading"]
                if "failed" in reading:
                    failed[(name, reading["oracle_partner"])] = reading["failed"]
                timing["units"][f"{name} | {configuration.name}"] = time.perf_counter() - one
            timing["units"][name] = time.perf_counter() - begun
    finally:
        runner.close()
    document = {
        "label": f"dial choice: {LABEL}",
        "definitions": definitions(matrix, (plain, dials), splits),
        "run": {
            "splits": splits,
            "seeds": [seeds[0], seeds[-1]] if seeds else [],
            "scales": list(dial_splits.GRID),
            "n_calibration": matrix.n_calibration,
            "calibration_half_below": matrix.calibration_half_below,
            "kappa": plain.kappa,
            "tiers": list(tiers),
            "rung": RUNG,
            "dials": [c.driver_risk for c in dials],
            "ceilings": list(matrix.ceilings),
            "gate_ceiling": matrix.ceilings[0],
            "thresholds": {
                "share_of_ceiling": SHARE_THRESHOLD,
                "oracle_to_offer_ratio": RATIO_THRESHOLD,
            },
        },
        "units": unit_entries,
        "dial_choice": {
            name: {tier: _choice(entry, tier) for tier in tiers}
            for name, entry in unit_entries.items()
            if entry["status"] == "calibratable"
        },
        "timeouts": dict(sorted(timeouts.items())),
        "failed_configurations": sorted(f"{a} | {b}: {why}" for (a, b), why in failed.items()),
    }
    timing["total_seconds"] = time.perf_counter() - started
    return document, timing


def _choice(entry: dict, tier: str) -> dict:
    """The dial of the unit's entry with the largest mean offer under `tier` (module text)."""
    offers = {}
    for found in entry["configurations"].values():
        if found["driver_risk"] is not None and "tiers" in found:
            offers[found["driver_risk"]] = found["tiers"][tier]["mean_offer_kwh_per_test_day"]
    dial = dial_splits.pick_dial(offers)
    return {
        "dial": dial,
        "mean_offer_kwh_per_test_day": None if dial is None else offers[dial],
        "mean_offers_by_dial": {f"{d:g}": offers[d] for d in sorted(offers)},
        "label": LABEL,
    }


SPLIT_OFFERS_DEFINITION = (
    "per gate unit that can be split, configuration and tier, one entry per split of the run's "
    "seeds, in the order of the seeds: the seed, the calibrated one-way re-solve offer per "
    "counting test day (`offer_kwh`, 0 when the split abstains) and the grid scale it was "
    "calibrated at; the mean of a cell's offers is that cell's mean_offer_kwh_per_test_day in the "
    "results file named under `results`, whose SHA-256 is given; a configuration that failed has "
    "no entries"
)


def split_offers_document(document: dict, found: dict, results_name: str, text: str) -> dict:
    """The per-split file of a run (module text): its identity, the SHA-256 of the results text
    `text` written as `results_name`, and per unit that can be split the entries `found`."""
    identity = (
        "label",
        "what_if",
        "rotation_whatif",
        after_sealed_run.KEY,
        "model",
        "fit_units",
        "matrix",
        "inputs",
        "forecast_table",
        "run",
    )
    return {
        **{k: document[k] for k in identity if k in document},
        "definitions": {"split_offers": SPLIT_OFFERS_DEFINITION},
        "results": {"file": results_name, "sha256": hashlib.sha256(text.encode()).hexdigest()},
        "units": {
            unit: {
                "status": entry["status"],
                "counting_days": entry["counting_days"],
                "configurations": found.get(unit, {}),
            }
            for unit, entry in document["units"].items()
        },
    }


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--splits", type=int, default=SPLITS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--what-if-kappa", type=float, default=None)
    parser.add_argument("--what-if-fit-seed", type=int, default=None)
    parser.add_argument("--what-if-fit-seeds", default=None)
    parser.add_argument("--what-if-fit-levels", choices=FIT_LEVEL_CHOICES, default=None)
    parser.add_argument(after_sealed_run.FLAG, dest="after_sealed", action="store_true")
    args = parser.parse_args(argv)
    what_if = args.what_if_kappa is not None
    fit_option = (
        args.what_if_fit_seed is not None
        or args.what_if_fit_levels is not None
        or args.what_if_fit_seeds is not None
    )
    try:
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if args.splits < 1:
            raise MatrixError("--splits must be at least 1")
        if what_if and args.kappa is not None:
            raise MatrixError("give --kappa or --what-if-kappa, not both")
        matrix = load(args.matrix)
        seeds = check_fit_options(
            matrix,
            args.what_if_fit_seed,
            args.what_if_fit_levels,
            args.what_if_kappa,
            what_if,
            args.what_if_fit_seeds,
        )
        family = find_family(matrix)
        own, measured = family[0].kappa, matrix
        if args.kappa is not None:
            measured, family = family_at(matrix, family, args.kappa)
        if what_if:
            measured, family = family_at(matrix, family, args.what_if_kappa, what_if=True)
        fit_all = what_if if args.what_if_fit_levels is None else args.what_if_fit_levels == "all"
        built_from = fit_matrix(matrix, family[0]) if fit_all else matrix
        if seeds is not None:
            model = averaged_model(
                lambda: learn_run.load_model(args.model, built_from), args.model, seeds
            )
        else:
            model = learn_run.load_model(args.model, built_from)
        if fit_option and seeds is None:
            model = with_fit_seed(model, args.model, args.what_if_fit_seed)
    except MatrixError as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    fit_units = tuple(model.fit_units)
    try:
        learn_run.check_model(args.model, model, fit_units, matrix.gate_units)
    except learn_run.FitOnScored as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
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
    fit_frames = {name: loaded[name].cars for name in fit_units}
    try:
        factory, table = make_factory(
            args.model, model, matrix, units, fit_frames, family[0].kappa, what_if
        )
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: the model could not be scored: {problem}")
        return 1
    certificate_timeouts = None if args.kappa is None and not what_if and not fit_option else {}
    split_offers: dict = {}
    document, timing = run(
        measured,
        units,
        factory,
        splits=args.splits,
        workers=args.workers,
        family=family,
        certificate_timeouts=certificate_timeouts,
        split_offers=split_offers,
    )
    if what_if or fit_option:
        fit = None
        if fit_option:
            fit = fit_record(model, args.what_if_fit_seed, args.what_if_fit_levels, what_if, seeds)
        record = what_if_record(family, fit)
        document["label"] = f"dial choice: {record['label']}"
        document["what_if"] = record
        document["definitions"]["what_if"] = record["readings"]
    if not what_if and measured is not matrix:
        document["definitions"]["kappa_option"] = kappa_option(own, family)
    document["model"] = model.name
    document["fit_units"] = list(fit_units)
    rotation_whatif.mark(document, matrix, fit_units)
    after_sealed_run.mark(document, args.after_sealed)
    document["inputs"] = inputs
    document["matrix"] = {
        "file": args.matrix.name,
        "sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
    }
    names = WHAT_IF_NAMES if what_if or fit_option else PLAN_NAMES
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        write_table(table, folder / names["table"], what_if)
        document["forecast_table"] = {
            "file": names["table"],
            "rows": len(table),
            "sha256": hashlib.sha256((folder / names["table"]).read_bytes()).hexdigest(),
        }
    text = "\n".join(dial_summary.summary_lines(document)) + "\n"
    print(text, end="")
    if run_dir:
        results = _json(document)
        (folder / names["results"]).write_text(results)
        (folder / names["split_offers"]).write_text(
            _json(split_offers_document(document, split_offers, names["results"], results))
        )
        (folder / names["summary"]).write_text(text)
        (folder / names["timing"]).write_text(_json(timing))
        if certificate_timeouts is not None:
            counts = {"kappa": family[0].kappa, "certificate_timeouts": certificate_timeouts}
            (folder / names["certificate_timeouts"]).write_text(_json(counts))
        after_sealed_run.write_note(folder, args.after_sealed)
    return 1 if document["failed_configurations"] else 0


if __name__ == "__main__":
    sys.exit(main())
