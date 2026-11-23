"""The comparison methods through the offers, paths, calibration and loss of the re-solve rung
(plan section 10: "All get the same forecasts, requests, calibration and loss, plus oracle
departures"; plan 13 WP10: "shared offers, paths, output files").

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.compare_run --model <name> --dial <value> [--matrix <file>]
        [--splits N] [--workers N] [--tables <folder>] [--truth-run <folder>]
        [--methods <name> ...] [--no-pause-share 0|1/3] [--no-pause-seed N]
        [--held-out-tables <folder>] [--held-out-truth-run <folder>]

Inputs as `dial_choice` takes them, through its own functions: the model (`learn_run.load_model`,
`learn_run.check_model`, `dial_choice.make_factory` at the kappa of the dial's configuration,
which every configuration of the family shares, `dial_choice.find_family`), the truth
tables (`gate_run.check_inputs`), the units (the matrix's gate units), the splits
(`dial_choice.split_indices`, seeds 1 to N, default 200), the scales (`dial_splits.GRID`) and
the tiers (`dial_splits.TIERS`). `--dial`: one dial value of the matrix's dial family
(`dial_choice.find_family`); any other value is refused. The configuration measured is that dial's
configuration; the oracle-departure variant is its oracle partner (the configuration's field
`oracle_partner`, as `dial_choice.gate_reading` takes it). `--methods`: the names of
`compare_methods.METHODS` to measure, in the order given (default: all of them, in the list's
order); the results' list `methods`, the summary's rows and the pairs follow that order.

Readings of the plan's wording (each is the closest reading; the alternative is in brackets):

  C1. Shared offer. Every method runs on the day's certified box (`PreparedDay.certificate`, the
      same shape for all) at scale beta, with the requests `m + beta w zeta` on the day's audited
      paths (`PreparedDay.paths`), on the steered sub-day (`PreparedDay.day`) and with the
      configuration's `departure_seen`. A day that abstains for the certificate abstains for
      every method. [Alternative: each method certifies its own box.]
  C2. A method's day-level slot loss and driver loss at a scale are the largest over the audited
      paths, as `gate_day.evaluate` takes them for the rungs. [Alternative: the mean over the
      paths.]
  C3. Calibration and scoring are those of `dial_splits` (`UnitLosses`, `score_split`,
      `summarise`, through `dial_choice.score_splits`) on `dial_choice.split_indices`' splits, per
      method and tier: each method gets its own calibrated scale on the same calibration days and
      is scored on the same test days. A method's offer on a test day is its calibrated scale
      times the day's certified value. [Alternative: one scale, the re-solve rung's, for every
      method.]
  C4. "Plus oracle departures": every method is also run under the oracle partner of the dial's
      configuration: the box certified with oracle departures, and a controller's belief taken
      from that prepared day's forecasts (`controller.forecast_belief(prepared.day)`), as the
      re-solve rung is measured with oracle departures in `dial_choice.gate_reading`.
      [Alternative, not built: the forecast box with a belief that knows every true limit.]
  C5. A controller's hour-zero target is `controller.hour_zero_target(Plan.from_policy(policy,
      beta))`, the fleet total of the fixed plan in hour 0. [Alternative: no charging in hour 0.]

Per unit that can be split (`gate_run.plan_unit`): its ceilings (`gate_day.day_ceilings`), then
per variant ("forecast", "oracle-departure") every day prepared (`gate_day.prepare_day`). The
re-solve method's losses are `dial_choice.measure` of the prepared days (the one-way rung of
`gate_day.evaluate`, each (day, scale) evaluated once); the controller methods' losses are
`compare_day.evaluate` of each day that offers at each scale of the grid, one evaluation per
(day, scale) for all the controller methods together. Every day enters with its certified value
(0 for a day that abstains, `gate_day.prepare_day`); a day that abstains has no loss and is not
evaluated, as in `dial_choice.measure`. A unit that cannot be split is reported with its status
and no numbers. Per method: per variant and tier the summary of `dial_choice.score_splits`, and the
gate reading of `dial_splits.gate_split` and `dial_splits.summarise_gate` (tier A: the mean offer,
the share of each ceiling of the matrix, the oracle-departure variant's mean offer and its ratio
to the mean offer), read as `dial_choice` reads it for a configuration and its oracle partner.

Per split: the rows of `compare_splits` (one per unit, variant, tier, method and split, from the
same `dial_splits.score_split` calls the summaries are made of, with each ceiling's mean over the
split's test days), in the order unit, variant, tier, method, seed; a variant that failed has no
rows. From those rows, per unit that can be split, variant and tier, the paired differences of
`compare_paired` between methods on the same split (its module text says what the percentiles are
and are not, and defines the verdict words).

No-pause sensitivity (plan section 9: "a seeded 0 or 1/3 of cars charge as logged, without
flexibility (no-pause; Brinkel et al. 2024)"). `--no-pause-share`: "0" (the default) or "1/3"
(`no_pause.SHARES`; any other value is refused); `--no-pause-seed`: the seed of the draw (default
`no_pause.DEFAULT_SEED`; not read at share 0). At share 0 the run is the run without the option,
and every file it writes but the timing file is the same bytes. At share 1/3, per unit that can
be split, the cars of each counting day drawn by `no_pause.drawn` (readings N1 and N2 in
`no_pause`) are given to `gate_day.prepare_day` as `no_pause` under both variants, so every
method, split, scale, path and both variants see the same drawn cars left unsteered (reason
`gate_day.NO_PAUSE`); the splits, the scales and the audited paths are those of the run without
the option. Reading N3, the ceilings: the ceilings of the matrix keep their names and are those
of the whole fleet, the drawn cars included (so "of ceiling", `mean_share_of_best_causal_ceiling`
and the split table's `share_of_best_causal_ceiling` are shares of the whole fleet's best-causal
ceiling, as in the run without the option), and beside each of them the same ceiling of the
fleet without the drawn cars (`no_pause.without`; the uncertifiable cars kept) is written under
its name followed by `NO_PAUSE_CEILING_SUFFIX`, in the gate reading and the split table.
[Alternative: only one of the two, or the ceiling of the steered cars alone, which would also take
out the uncertifiable cars, which can pause.] At share 1/3 the run also writes, beside the usual
fields: `no_pause` (the share, the seed and the readings N1 to N5, N1, N2, N4 and N5 being those
of `no_pause`), per unit `no_pause` (`no_pause.counts` over the unit's counting days: cars, cars
drawn, days on which every car is drawn) and per variant `unsteered_cars` (the unsteered cars of
its prepared days counted by reason: a car on several counting days once per day; on a day that
abstains under "day-abstains", the drawn and the uncertifiable cars and not the day's other
cars), the timeouts of the second best-causal ceiling as
`best_causal_ceilings_no_pause_cars_removed`, a label that names the sensitivity, and every file
under its own name (`output_names`: `compare-no-pause-third` in place of `compare`), so that no
file of the sensitivity can be taken for one of the run without it; the summary's first line
(on the sealed route its third, after the route's two) names the sensitivity, and the timing
file holds its share and seed. Before the model is scored,
`check_draw` draws the cars of every counting day of every unit that can be split, and a site or
session identifier holding `|`, which `no_pause.is_drawn` refuses, stops the command with one
line naming the unit (exit 2).

Timeouts are recorded outcomes: a day whose certificate times out abstains (`gate_day`) for every
method, and the numbers of certificates, best-causal ceilings and re-solves that timed out are
counted. Of the controllers only `asa`, `asa-es` and `diagonal-split` solve programmes. The
solves of each are counted by outcome, timeouts included, with the hours it planned and the hours
it planned by its fall-back (`compare_day`, solver counts), per unit and variant under
`evaluations` as `controller_solver_outcomes`, and summed over units and variants in one line of
the summary; `diagonal-split` solves one programme per (day, scale) pair and also counts, per
kind of `diagonal_split.BOXES`, the (day, scale) pairs whose box is of that kind (`boxes`: the
pairs found to have no diagonal split are those of the kinds "stretched" and "no gains";
for "no programme" it was not found out). That field, a
definitions entry `controller_solver_outcomes` and a longer `timeouts` definition are written
only when a method reported solver counts, so a run without `asa`, `asa-es` and
`diagonal-split` writes none of them; the definitions entry `diagonal_split_boxes` is written
only when `diagonal-split` reported counts.

`max-entropy-feedback` solves no programme; it counts the hours it planned as in step 3 of
`mef_controller` (least-laxity-first's split of the target found to leave the believed future
unmet, the nearest total found to keep it met planned instead) and those in which it found no
such total (`compare_day`, deviation counts; `mef_controller.COUNTS`), per unit and
variant under `evaluations` as `controller_deviations`, summed over units and variants in one
line of the summary. That field and a definitions entry `controller_deviations` are written only
when a method reported deviation counts, so a run without `max-entropy-feedback` writes neither.

The sizing methods (`also-x` and `cvar`, kind "sizing" of `compare_methods`; readings S1 to S11 in
`chance_sizing`) run no controller: their losses are the re-solve method's, measured once per
variant whether or not `re-solve` is in the run, and only the rule that picks each split's scale
on its offering calibration days is theirs (`dial_splits.UnitLosses.rule`); every other field,
row and pair is made as for any method. Their rules solve programmes (`chance_programmes`, the
300-second limit of plan section 13 for other programmes on every solve), once per split's
calibration days; per unit and variant `evaluations.sizing_solver_outcomes` holds, per sizing
method, the calibration sets solved and the solves by outcome, and the summed timeouts are
counted as `sizing_programmes` and `oracle_sizing_programmes`. That field, those two counts, the
definitions entry `sizing` (what the methods are, the readings, the counts) and a longer
`timeouts` definition are written only when a sizing method is in the run, so a run without one
writes none of them.

The record of the days the methods share (plan 13 WP10: "shared offers, paths, output files").
Per unit that can be split and variant, `measured_on` (the function `measured_on`) records the
prepared days every method of the run is measured on under that variant: `measure_variant` gives
one list of prepared days to the re-solve, the sizing methods and every controller alike. Fields:
`days`, their number; `path_rows`, the number of audited-path rows over them (8 a day);
`boxes_sha256`, `paths_sha256` and `steered_sha256`, each the SHA-256 of the compact JSON text
(separators "," and ":", floats as Python's `repr`) of one entry per day, in day order: for the
boxes [date, site, whether the day offers, its certified value at scale 1, [the centre m, the
half-widths w] of its certified policy, or null on a day that abstains]; for the paths [date,
site, its audited paths as rows of -1 and 1]; for the steered cars [date, site, the positions in
the day's fleet of its steered cars, [position, reason] of each unsteered car]. Within a run the
record names the boxes, paths and steered cars the methods were measured on; between two runs
equal digests show the same ones (a run at `--no-pause-share 1/3` keeps the audited paths of the
run without it; its boxes and steered cars are its own). It is in the results only, never in
the summary or the split table, so they are the same bytes as before it was added; the
definitions entry `measured_on` says what it holds.

The per-day work runs in this process (`--workers 1`) or in a pool of worker processes; the work
in the pool is the work `gate_run`'s own pool does (ceilings, prepared days, evaluations), written
again here because those tasks are private to `gate_run`. The results, the summary and the split
table are the same bytes for 1, 2 and 3 workers on the made-up units of the tests; wall-clock
seconds are kept apart.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary also on
standard output:
- `compare.json`: inputs (model, dial, configurations, matrix hash, truth tables' hashes, splits),
  and per unit, variant, method and tier the summary fields, the gate reading per method, the
  counts of (day, scale) pairs evaluated, the solver counts of a controller that solves
  programmes, the deviation counts of a controller that counts them, per unit and variant the
  record `measured_on` and the readings C1 to C5 under `definitions`; the paired differences and
  their
  definitions under `paired_differences`; the split table's file name, number of rows and
  SHA-256 under `split_table`;
- `compare-summary.txt`: one table per unit and tier, then the tables of paired differences
  (`compare_summary`);
- `compare-forecasts.csv`: the model's forecast table (its hash is in the results);
- `compare-splits.csv`: the per-split rows (`compare_splits`; its hash is in the results);
- `compare-timing.json`: wall-clock seconds.

Exit status, as the tests of this command run it: 0 every unit and variant ran; 1 a unit and
variant failed (recorded; the other variants and units go on); 2 could not run: an unknown model,
a `--dial` not in the dial family, a matrix without a dial family, a missing truth build, a method
registry or method name refused by `compare_methods.select`, a number of workers or splits below
1, a `--no-pause-share` other than 0 and 1/3, at `--no-pause-share 1/3` a site or session
identifier holding `|` on a counting day of a unit that can be split (`check_draw`), and for
held-out units every refusal of the route below. Handled as `dial_choice.main` handles them, by
the same calls and exit codes, and not run by the tests of this command: exit 2 for a matrix file
`gate_config.read` refuses, a model fitted on a scored unit, a table missing, a truth build at
another commit and a table differing from it; exit 1 for a refused model, a model that fails and a
day that cannot be built from the tables.

Held-out units (plan section 10's baselines, among them ASA and LLF, which give outcome O1's A
and B, section 11), through the one shared route of `sealed_route` (`sealed_route.matrix_route`,
in its order), which starts only at the tag `frozen-plan` with a clean tree; its checks stop an
honest operator's ordinary mistakes and are not a defence against deliberate evasion
(`sealed_route`, module text, first paragraph, with the commands that have the route). Every
run first puts its path arguments through `gate_run.argument_paths` and its matrix file through
`gate_run.check_files`, and reads the matrix once (`gate_config.read`; reading H4). When the
matrix's gate units are held-out units (all of them; a mix with other units could not run),
before the model is loaded: every folder named lies outside the held-out places of the top
folder, the tag gate passes (the held-out loader's checks 2 to 8: at the tag `frozen-plan`,
clean, the running code the tag's) and `--held-out-truth-run` is given (`--held-out-tables`
defaults to the held-out build's main folder); then `--splits` must be 1 (`ONE_SPLIT`, reading
H1). After the model is loaded, a model of `forecast_model.OUTCOME_READERS` could not run
(`sealed_route.OUTCOME_READER`, as `learn_run` and the cold start refuse it) and the gate is
passed again (the model's module is then among the running code). Before any table is read the
matrix's split rule must be the held-out build's (`sealed_route.check_split_rule`). The model's
`fit_units` are then read from the development build (`--tables`, `--truth-run`) and the gate
units from the held-out build (`--held-out-tables`, `--held-out-truth-run`), each build checked
and read once and the two of one setting (`sealed_route.read_two_builds`; the bytes hashed are
the bytes parsed, `days_from_truth.parse_unit_days`), and each scored unit's calibration days as
the run draws them must be the build's where the build recorded them
(`sealed_route.check_split_days`). The run is then the run above on one split per unit, the
matrix's `split_seed`, which is the build's draw: every method is calibrated on that draw's
calibration days and scored on its test days. The results then hold `inputs` (the development
build's check, of the fit units), `sealed_route` (`sealed_route.mark`: the tag's commit, the
held-out build's check, the fit units, no kappa selection, the checks' text), the label
`ROUTE_LABEL` (with the sensitivity named after it at `--no-pause-share 1/3`), the definition of
the one split under `splits`, the definition `interval` of the paired differences replaced by
`ROUTE_INTERVAL`, and the readings H1 to H6 under `route_readings`; the summary's first two lines
are those of `sealed_route.summary_head`, then (at `--no-pause-share 1/3` only) the no-pause
sensitivity's line, then the line that names the one split and the label; its line on what the
columns are says what each is on the one split, and the paragraph on the paired differences is
followed by `ROUTE_INTERVAL` (`compare_summary`). With
other gate units the two held-out options could not run, and without them nothing above but the
path and matrix checks is made: the run, its outputs and its summary are as before.

What is fitted, chosen, tuned, sampled or calibrated, and from which days:
  1. The forecast model `--model`, which every method shares: fitted on its `fit_units`, training
     units only, from the development build (`learn_run.check_model`; a model fitted on a scored
     unit could not run), then forecasting each scored unit's counting days from their inputs at
     the levels up to the configuration's kappa (`dial_choice.make_factory`,
     `learn_run.forecast_frame`: no outcome or label column; a user's earlier sessions plugged out
     by the decision time are inputs, `sealed_route` reading S1). On the route a model that builds
     a scored unit's forecasts from that unit's own other days, test days included (`empirical`),
     could not run; a look-ahead model is refused on and off the route (`learn_run.check_model`).
  2. Each day's certified box and audited paths (`gate_day.prepare_day`, `audit_paths`): from that
     day's forecasts, the configuration and the matrix's seeded path rule, nothing carried from
     another day. The oracle-departure variant reads that day's true departures, plan section 10's
     "plus oracle departures"; it is a variant beside the forecast one, never in its place.
  3. The scale of every method in each tier (`re-solve`, the eight controllers, `also-x` and
     `cvar`): chosen per split on the split's offering calibration days only
     (`dial_splits.score_split`: the bound of `crc` on the grid, or for `also-x` and `cvar` the
     chance programme of `chance_sizing` on those days' losses, with the driver bound of `crc` in
     tier A), and the test days scored at that scale only. On the route the one split is the
     held-out build's draw (the three checks above; reading H1).
  4. The controllers `fixed-plan`, `edf`, `llf`, `smoothed-llf`, `asa`, `asa-es`,
     `diagonal-split` and `max-entropy-feedback`: nothing fitted or tuned on any day. Each plans a
     day from that day's box, fixed plan and forecasts as its hours pass (`asa`, `asa-es` and
     `diagonal-split` solve programmes on that day alone), with constants fixed in their modules.
  5. The no-pause draw (`--no-pause-share 1/3`): each car by itself from the share, the seed and
     its site and session identifier (`no_pause`, readings N1 and N2); no outcome is read.
  6. The dial and kappa: the one configuration of `--dial` for every unit and split; nothing
     chooses a dial or a kappa per unit or from results (reading H2).
  7. No other command's run folder is read: the inputs are the matrix, the development build and,
     on the route, the held-out build, each checked as above.

Readings of the route (each with its alternative in brackets):
  H1. `--splits` on the route. Plan section 6, Protocol, gives a held-out unit one draw: its
      seeded calibration days "and the rest test, one draw for all policies". The route scores
      that draw, the matrix's `split_seed` (which must be the held-out build's), so the
      comparison methods are calibrated and scored on the days `learn_run` uses, and `--splits`
      other than 1 could not run before any table is read. Off the route `--splits 1` draws seed
      1: the same draw when the matrix's `split_seed` is 1, as in the committed matrices.
      [Alternatives: N seeded splits of a held-out unit, as on development units: refused, since
      every seed but the build's would calibrate on days the build's draw tests; or `--splits`
      ignored on the route: not taken, since the run would then name more splits than it scores.]
  H2. The dial on the route is taken as given: `--dial`, one value for every scored unit of the
      run, fixed before the tag per site type (plan section 6); the command reads no result to
      choose it and has no option that chooses a dial or a kappa per unit, so there is no such
      choice to refuse. One run per site type, each at its type's dial, is the run list's to
      state. [Alternative: refuse a dial other than the one recorded for the scored units' site
      type; no committed record of the dials exists to compare with.]
  H3. `--methods`, `--no-pause-share`, `--no-pause-seed` and `--workers` are allowed on the
      route: none chooses anything from a scored unit's days. [Alternative: refuse a subset of
      the methods on the route, as `gate_run` refuses its trial option `--only`.]
  H4. Every run, on the route or off it, puts its path arguments through
      `gate_run.argument_paths` and its matrix file through `gate_run.check_files` before reading
      it, as `learn_run` does, since whether a run is on the route is known only from the matrix.
      Off the route this refuses only a path written as a URL and a matrix file in a refused place
      or with a second name on disk, and expands a leading `~`. [Alternative: those checks on the
      route only, which would need the matrix read before them.]
  H5. The calibration on the route is that of reading C3 (on the grid of `dial_splits.GRID`, up
      to 1), as off it; `learn_run` bisects (`gate_calibrate`; plan section 6, Protocol: "for
      rungs, bisection to 1e-3"), so the re-solve row here and `learn_run`'s re-solve offer are
      different figures. Not resolved here. [Alternative: bisection for every method.]
  H6. One `--dial` for every scored unit. The plan fixes the dial per site type (section 6,
      Protocol: workplaces from NREL from May 2019; Norway and Korea from the pooled Norwegian
      development sites), and a held-out matrix may hold units of both types (the a.s.r.
      workplaces with OSL_T, TRO_R or the Korean portfolios). Such a matrix runs today: every unit
      is scored at `--dial`, so the units of a type whose fixed dial is not `--dial` are scored at
      a dial the plan did not fix for them. Refusing a mix is not the literal reading: the plan
      fixes a dial per site type, not a run per site type, so two types whose fixed dials are
      equal may share a run, and a run of one type at the wrong dial is as wrong as a mix; the
      literal check compares each unit's `--dial` with its type's fixed value, of which no
      committed record exists (as in H2). One run per site type is the run list's to state.
      [Alternative: refuse a matrix whose held-out units are of two site types, which would also
      refuse two types with equal dials.]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from fleetreserve import (
    chance_sizing,
    compare_day,
    compare_methods,
    compare_paired,
    compare_splits,
    compare_summary,
    dial_choice,
    dial_splits,
    gate_day,
    gate_run,
    learn_run,
    no_pause,
    rotation_whatif,
    sealed_route,
)
from fleetreserve.compare_methods import MethodError
from fleetreserve.days_from_truth import load_unit_days, parse_unit_days
from fleetreserve.forecast_model import write_table
from fleetreserve.gate_calibrate import ceiling_values
from fleetreserve.gate_config import Matrix, MatrixError, read
from fleetreserve.no_pause import NoPauseError
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

RESULTS_NAME = "compare.json"
SUMMARY_NAME = "compare-summary.txt"
TABLE_NAME = "compare-forecasts.csv"
TIMING_NAME = "compare-timing.json"
SPLITS_NAME = "compare-splits.csv"
VARIANTS = ("forecast", "oracle-departure")
LABEL = "comparison methods on development units: development evidence, not a held-out result"
NO_PAUSE_STEM = "compare-no-pause-third"
NO_PAUSE_CEILING_SUFFIX = "-no-pause-cars-removed"
# The label of a run on the sealed route (module text), in place of `LABEL`.
ROUTE_LABEL = (
    "comparison methods on held-out units through the sealed-data route: a held-out result, each "
    "unit scored on the one split its held-out build drew; not computed here: the 95% bootstrap "
    "intervals over days of plan section 11"
)
# The refusal of `--splits` other than 1 on the route (module text, reading H1).
ONE_SPLIT = (
    "--splits {splits}: on the sealed route each held-out unit is scored on the one split its "
    "held-out build drew (plan section 6, Protocol: one draw per held-out unit; another seed "
    "would calibrate on days that draw scores), so --splits is 1; nothing read"
)
# The definition of the one split of a run on the route (module text).
ROUTE_SPLITS = (
    "1 split per unit on the sealed route: the one its held-out build drew its calibration days "
    "by (split_seed {seed}, n_calibration {n_calibration}, calibration_half_below {half_below}); "
    "the matrix's rule checked to be the build's before any table is read "
    "(sealed_route.check_split_rule) and each scored unit's drawn calibration days checked to be "
    "the build's where the build recorded them (sealed_route.check_split_days)"
)
# What the paired differences of a run on the route are (module text): the definition `interval`
# of the paired differences, and a line of the summary after their paragraph.
ROUTE_INTERVAL = (
    "On the sealed route each unit has one split: p5 and p95 are that split's one difference and "
    "the verdict follows its sign; neither is an interval, and plan section 11's 95% bootstrap "
    "intervals over days are not computed by this command."
)

READINGS = {
    "C1": (
        "shared offer: every method runs on the day's certified box (PreparedDay.certificate, "
        "the same shape for all) at scale beta, with the requests m + beta w zeta on the day's "
        "audited paths (PreparedDay.paths), on the steered sub-day (PreparedDay.day) and with "
        "the configuration's departure_seen; a day that abstains for the certificate abstains "
        "for every method"
    ),
    "C2": (
        "a method's day-level slot and driver losses at a scale are the largest over the "
        "audited paths, as gate_day.evaluate takes them for the rungs"
    ),
    "C3": (
        "calibration and scoring are dial_splits' (UnitLosses, score_split, summarise) on "
        "dial_choice.split_indices' splits, per method and tier: each method gets its own "
        "calibrated scale on the same calibration days and is scored on the same test days; a "
        "method's offer on a test day is its calibrated scale times the day's certified value"
    ),
    "C4": (
        "oracle departures: every method is also run under the oracle partner of the dial's "
        "configuration (the box certified with oracle departures, the controller's belief "
        "controller.forecast_belief of that prepared day), as the re-solve rung is measured with "
        "oracle departures in dial_choice.gate_reading; not built: the forecast box with a "
        "belief that knows every true limit"
    ),
    "C5": (
        "a controller's hour-zero target is controller.hour_zero_target(Plan.from_policy(policy, "
        "beta)), the fleet total of the fixed plan in hour 0"
    ),
}

PAIRED_DEFINITIONS = {
    "split_rows": (
        "compare-splits.csv (split_table: file, rows, sha256): one row per unit, variant, tier, "
        "method and split, from dial_splits.score_split, the calibration and scoring of the "
        "summary fields, with the failing test days and each ceiling's mean over the split's "
        "test days (compare_splits); an empty cell is a value that does not exist"
    ),
    "pairs": (
        "per unit that can be split, variant and tier: against_re_solve, every other method of "
        "the run against re-solve (none without re-solve in the run); other_pairs, every two "
        "methods other than re-solve, the one listed earlier in methods as reference"
    ),
    "difference": (
        "on each split where both methods have a finite offer: the method's offer minus the "
        "reference's offer on that same split, kWh per counting test day; a split without one "
        "is left out of the pair and counted in splits_left_out_value_missing, never read as 0"
    ),
    "figures": (
        "over the paired splits: the mean difference; its 5th and 95th percentiles "
        "(numpy.quantile, linear interpolation); the shares of paired splits with a difference "
        "above, exactly equal to and below 0; the two mean offers; the mean difference over the "
        "reference's mean offer (null when that mean offer is 0: no offer); all null with no "
        "paired split"
    ),
    "interval": (
        "the splits of a unit share the same days and differ only in which days calibrate and "
        "which are scored: the 5th to 95th percentile range is the spread of the difference over "
        "those draws at this unit, not a confidence interval for new days, other units or other "
        "sites"
    ),
    "verdicts": {
        "ahead": "the 5th percentile of the differences is above 0",
        "behind": "the 95th percentile of the differences is below 0",
        "not separated": "neither",
    },
}

NO_PAUSE_READINGS = {
    "N1": (
        "what is drawn: each car of a day's fleet by itself, with probability equal to the share, "
        "from its site and session identifier (no_pause.is_drawn); a session plugged in over "
        "several days is drawn alike on each; no count per day is fixed, and a day whose every "
        "car is drawn has no steered car and abstains; not built: exactly a third of each day's "
        "cars, or a third of drivers by user_id"
    ),
    "N2": (
        "the same draw everywhere: it reads only the share, the seed and the car's site and "
        "session identifier, so every method, split, scale, path and both variants see the same "
        "drawn cars, whatever the number of worker processes or the order of the days"
    ),
    "N3": (
        "ceilings: the ceilings named as in the matrix are the whole fleet's, the drawn cars "
        "included (of ceiling, mean_share_of_best_causal_ceiling and share_of_best_causal_ceiling "
        "read them); the same ceilings of the fleet without the drawn cars (no_pause.without, "
        "uncertifiable cars kept) are written beside them under the names ending "
        f"{NO_PAUSE_CEILING_SUFFIX}, in the gate reading and the split table; not built: the "
        "ceiling of the steered cars alone"
    ),
    "N4": (
        "where the draw is used: only compare_run takes it, and of the rungs it writes only the "
        "one-way re-solve rung, beside the controllers; plan section 9 names the sensitivity "
        "under two-way charging, and no command writes a two-way rung's figures under the draw; "
        "the two-way rung is still computed (gate_day.evaluate replays every rung on a day that "
        "offers, at a scale above 0) and compare_run reads only the one-way rung of it; not "
        "built: the two-way rungs of gate_run under the draw"
    ),
    "N5": (
        "the forecasts: the model, the units it is fitted on and the units' cars the forecaster "
        "reads are those of the run without the draw, so the drawn cars stay in the data the "
        "forecasts come from; a drawn car is only not forecast on a day it is drawn; not built: "
        "the drawn cars taken out of that data"
    ),
}

ROUTE_READINGS = {
    "H1": (
        "one split per held-out unit, the one its held-out build drew (plan section 6, Protocol: "
        "one draw for all policies), so every method is calibrated and scored on the days "
        "learn_run uses; --splits other than 1 could not run; not taken: N seeded splits of a "
        "held-out unit, which would calibrate on days the build's draw tests"
    ),
    "H2": (
        "the dial is --dial, taken as given for every scored unit of the run, fixed before the tag "
        "per site type; nothing chooses a dial or a kappa per unit or from results; not built: a "
        "check of the dial against a record of the dials fixed per site type"
    ),
    "H3": (
        "--methods, --no-pause-share, --no-pause-seed and --workers are allowed on the route: none "
        "chooses anything from a scored unit's days; not taken: refusing a subset of the methods"
    ),
    "H4": (
        "every run puts its path arguments through gate_run.argument_paths and its matrix file "
        "through gate_run.check_files before reading it, since the route is known only from the "
        "matrix; not taken: those checks on the route only"
    ),
    "H5": (
        "the calibration is that of reading C3 (on the grid, up to 1), as off the route, while "
        "learn_run bisects (plan section 6, Protocol), so the re-solve row here and learn_run's "
        "re-solve offer are different figures; not resolved here"
    ),
    "H6": (
        "one --dial for every scored unit: a matrix whose held-out units are of two site types "
        "(a.s.r. workplaces with Norwegian or Korean units) runs, every unit at --dial; refusing "
        "it is not the literal reading (the plan fixes a dial per site type, not a run per site "
        "type, and two types with equal fixed dials may share a run); not built: refusing a mix"
    ),
}


def output_names(share) -> dict:
    """The file names of a run at `share` (module text): kind -> name."""
    if share == 0:
        return {
            "results": RESULTS_NAME,
            "summary": SUMMARY_NAME,
            "table": TABLE_NAME,
            "timing": TIMING_NAME,
            "splits": SPLITS_NAME,
        }
    return {
        "results": f"{NO_PAUSE_STEM}.json",
        "summary": f"{NO_PAUSE_STEM}-summary.txt",
        "table": f"{NO_PAUSE_STEM}-forecasts.csv",
        "timing": f"{NO_PAUSE_STEM}-timing.json",
        "splits": f"{NO_PAUSE_STEM}-splits.csv",
    }


def no_pause_label(share, seed: int, label: str = LABEL) -> str:
    """The label of a run of the sensitivity (module text); `label`: that of the run without it
    (`ROUTE_LABEL` on the route)."""
    return (
        f"{label}; NO-PAUSE SENSITIVITY of plan section 9: a seeded share "
        f"{no_pause.share_text(share)} of the cars (seed {seed}) charge as logged, without "
        "flexibility, and are left unsteered"
    )


# --- the per-day work, in this process or in a pool ----------------------------------------------

_STATE: dict = {}


def _start(units: dict, matrix: Matrix, factory) -> None:
    _STATE.clear()
    _STATE.update(units=units, matrix=matrix, factory=factory, forecasters={})


def _forecaster(unit: str):
    found = _STATE["forecasters"].get(unit)
    if found is None:
        found = _STATE["factory"](_STATE["units"][unit], _STATE["matrix"])
        _STATE["forecasters"][unit] = found
    return found


def _ceiling_task(job):
    unit, index = job
    return gate_day.day_ceilings(_STATE["units"][unit].days[index].day)


def _reduced_ceiling_task(job):
    unit, index, drawn = job
    day = _STATE["units"][unit].days[index].day
    return gate_day.day_ceilings(no_pause.without(day, drawn))


def _prepare_task(job):
    unit, index, configuration, drawn = job
    truth = _STATE["units"][unit].days[index]
    return gate_day.prepare_day(
        truth,
        _forecaster(unit),
        configuration,
        gate_run.settings(_STATE["matrix"]),
        no_pause=drawn,
    )


def _evaluate_task(job):
    prepared, beta = job
    return gate_day.evaluate(prepared, beta).compact()


def _compare_task(job):
    prepared, beta, methods = job
    return compare_day.evaluate(prepared, beta, methods)


class Pool:
    """Maps the per-day tasks in this process (`workers` 1) or in a pool of processes."""

    def __init__(self, workers: int, units: dict, matrix: Matrix, factory):
        _start(units, matrix, factory)
        self.pool = None
        if workers > 1:
            self.pool = ProcessPoolExecutor(
                max_workers=workers, initializer=_start, initargs=(units, matrix, factory)
            )

    def map(self, task, jobs) -> list:
        jobs = list(jobs)
        if self.pool is None:
            return [task(job) for job in jobs]
        return list(self.pool.map(task, jobs))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()


# --- one unit under one variant ------------------------------------------------------------------


def measure_controllers(prepared, best_causal, methods, compare_batch, solved=None, deviated=None):
    """Name -> `dial_splits.UnitLosses` of each controller method, and the number of (day, scale)
    pairs evaluated: every day that offers at every scale of the grid, once. When `solved` (a
    dict) is given, it receives name -> the sum over those pairs of the method's solver counts
    (`compare_day.sum_counts`), for each method that reports them; `deviated` (a dict) likewise
    receives the sums of the deviation counts (`compare_day.sum_deviations`)."""
    days, scales = len(prepared), len(dial_splits.GRID)
    offering = [d for d, p in enumerate(prepared) if not p.abstains]
    jobs = [(prepared[d], scale, methods) for scale in dial_splits.GRID for d in offering]
    found = iter(compare_batch(jobs) if jobs else [])
    slot = {m.name: np.zeros((days, scales)) for m in methods}
    driver = {m.name: np.zeros((days, scales)) for m in methods}
    reported = {m.name: [] for m in methods}
    deviations = {m.name: [] for m in methods}
    for k in range(scales):
        for d in offering:
            day = next(found)
            for m in methods:
                slot[m.name][d, k] = day[m.name].slot_loss
                driver[m.name][d, k] = day[m.name].driver_loss
                if day[m.name].solver_counts is not None:
                    reported[m.name].append(day[m.name].solver_counts)
                if day[m.name].deviation_counts is not None:
                    deviations[m.name].append(day[m.name].deviation_counts)
    if solved is not None:
        solved.update({n: compare_day.sum_counts(r) for n, r in reported.items() if r})
    if deviated is not None:
        deviated.update({n: compare_day.sum_deviations(r) for n, r in deviations.items() if r})
    value = np.array([p.value for p in prepared], dtype=float)
    flags = np.array([not p.abstains for p in prepared], dtype=bool)
    ceiling = np.array([np.nan if c is None else c for c in best_causal], dtype=float)
    losses = {
        m.name: dial_splits.UnitLosses(value, flags, slot[m.name], driver[m.name], ceiling)
        for m in methods
    }
    return losses, len(jobs)


def measure_variant(prepared, best_causal, methods, evaluate_batch, compare_batch):
    """Name -> `UnitLosses` of every method of one variant, and the counts of evaluations. A
    sizing method's losses are the re-solve method's with its own rule (`Method.size` of the
    prepared days' numbers of service hours), measured once whether or not re-solve is run."""
    losses, counts = {}, {"re_solves_of_the_one_way_rung": {}}
    sizing = tuple(m for m in methods if m.kind == "sizing")
    if sizing or any(m.kind == "re-solve" for m in methods):
        found, cache = dial_choice.measure(prepared, best_causal, evaluate_batch)
        outcomes: Counter = Counter()
        for evaluation in cache.store.values():
            outcomes.update(evaluation.rungs[dial_choice.RUNG].resolves)
        counts["re_solve_day_scale_pairs"] = len(cache.calls)
        counts["re_solve_pairs_evaluated_more_than_once"] = sum(
            1 for n in cache.calls.values() if n > 1
        )
        counts["re_solves_of_the_one_way_rung"] = dict(sorted(outcomes.items()))
        losses[compare_methods.RE_SOLVE] = found
        hours = np.array([p.paths.shape[1] for p in prepared], dtype=int)
        for m in sizing:
            losses[m.name] = replace(found, rule=m.size(hours))
    controllers = tuple(m for m in methods if m.kind == "controller")
    if controllers:
        solved: dict = {}
        deviated: dict = {}
        found, pairs = measure_controllers(
            prepared, best_causal, controllers, compare_batch, solved, deviated
        )
        losses.update(found)
        counts["controller_day_scale_pairs"] = pairs
        if solved:
            counts["controller_solver_outcomes"] = solved
        if deviated:
            counts["controller_deviations"] = deviated
    return {m.name: losses[m.name] for m in methods}, counts


def score_methods(losses: dict, splits, tiers) -> dict:
    """Name -> tier -> `dial_splits.summarise` of each method (`dial_choice.score_splits`)."""
    return {name: dial_choice.score_splits(found, splits, tiers) for name, found in losses.items()}


def sizing_counts(losses: dict) -> dict:
    """Name -> `chance_sizing.ChanceRule.counts` of each sizing method of one variant."""
    return {name: found.rule.counts() for name, found in losses.items() if found.rule is not None}


def _digest(entries: list) -> str:
    text = json.dumps(entries, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def measured_on(prepared) -> dict:
    """The record of the prepared days every method of one unit and variant is measured on
    (module text)."""
    boxes, paths, steered = [], [], []
    for day in prepared:
        box = None
        if not day.abstains:
            policy = day.certificate.policy
            box = [[float(v) for v in policy.m], [float(v) for v in policy.w]]
        boxes.append([day.date, day.site, not day.abstains, float(day.value), box])
        paths.append([day.date, day.site, [[int(v) for v in row] for row in day.paths]])
        unsteered = [[int(car), str(why)] for car, why in day.unsteered]
        steered.append([day.date, day.site, [int(car) for car in day.steered], unsteered])
    return {
        "days": len(prepared),
        "path_rows": int(sum(len(day.paths) for day in prepared)),
        "boxes_sha256": _digest(boxes),
        "paths_sha256": _digest(paths),
        "steered_sha256": _digest(steered),
    }


def gate_methods(forecast: dict, oracle: dict, ceilings: dict, splits, names) -> dict:
    """Name -> the tier A gate reading of each method, its oracle-departure variant as partner."""
    return {
        name: dial_splits.summarise_gate(
            [
                dial_splits.gate_split(found, oracle[name], ceilings, seed, cal, test)
                for seed, cal, test in splits
            ],
            names,
        )
        for name, found in forecast.items()
    }


# --- one run -------------------------------------------------------------------------------------


def definitions(matrix: Matrix, splits: int, solving=(), deviating=(), sizing=()) -> dict:
    """The definitions of the results; `solving`: the names of the methods that reported solver
    counts in this run (none: the definitions of a run without them); `deviating`: likewise for
    deviation counts; `sizing`: the names of the sizing methods of the run."""
    found = {
        "readings": dict(READINGS),
        "variants": {
            "forecast": "the dial's configuration of the matrix",
            "oracle-departure": "that configuration's oracle partner (field oracle_partner)",
        },
        "scales": "dial_splits.GRID: crc.GRID up to and including 1",
        "splits": (
            f"{splits} seeded splits per unit (seeds 1 to {splits}), dial_choice.split_indices "
            f"with the calibration-days rule (n_calibration {matrix.n_calibration}, "
            f"calibration_half_below {matrix.calibration_half_below})"
        ),
        "tiers": "dial_splits.TIERS, scored as dial_choice scores them, never pooled",
        "measured_on": (
            "per unit that can be split and variant: the prepared days every method of the run is "
            "measured on (one list for all of them); days, their number; path_rows, the audited "
            "path rows over them; boxes_sha256, paths_sha256, steered_sha256: the SHA-256 of the "
            "compact JSON of one entry per day, in day order: [date, site, offers, certified "
            "value at scale 1, [m, w] of the certified policy or null on a day that abstains]; "
            "[date, site, audited paths as rows of -1 and 1]; [date, site, positions of the "
            "steered cars, [position, reason] of each unsteered car]"
        ),
        "summary_fields": "per method, variant and tier: dial_splits.summarise",
        "gate_reading": (
            "per method, tier A, over all the splits: dial_splits.summarise_gate of "
            "dial_splits.gate_split with the method's oracle-departure variant as partner"
        ),
        "day_scale_pairs": (
            "re_solve_day_scale_pairs: (day, scale) pairs evaluated by gate_day.evaluate for the "
            "re-solve method; controller_day_scale_pairs: (day, scale) pairs evaluated by "
            "compare_day.evaluate for all the controller methods together"
        ),
        "timeouts": (
            "a day whose certificate times out abstains for every method; a re-solve that times "
            "out leaves the rung on its predecessor's plan; the controllers solve no programme"
        ),
        "label": LABEL,
    }
    if solving:
        names = ", ".join(solving)
        found["timeouts"] = (
            "a day whose certificate times out abstains for every method; a re-solve that times "
            f"out leaves the rung on its predecessor's plan; a controller that solves programmes "
            f"({names}) counts every solve's outcome, timeouts included, and plans an hour whose "
            "programme is not optimal by its fall-back; the other controllers solve no programme"
        )
        found["controller_solver_outcomes"] = (
            "evaluations.controller_solver_outcomes, per unit, variant and method that solves "
            "programmes, summed over the (day, scale) pairs evaluated and the audited paths: "
            "solves, the number of solves by outcome; hours, the hours the controller planned; "
            "fallback_hours, the hours it planned by its fall-back"
        )
    if "diagonal-split" in solving:
        found["diagonal_split_boxes"] = (
            "controller_solver_outcomes.diagonal-split.boxes: the (day, scale) pairs evaluated, "
            "counted once per pair however many audited paths it has, by the kind of the day's "
            "box at that scale (diagonal_split, readings D1 to D4): split, the box has a "
            "diagonal split on the rows it was certified on; stretched, it has none and the "
            "split of the largest box of the same centre and shape that has one is scaled up to "
            "it; no gains, no positive multiple of the box's widths about that centre has one and "
            "the plan "
            "follows the centre; no programme, the programme did not end optimal (so whether the "
            "box has a split is not known) and every hour is planned by the least-laxity "
            "fall-back"
        )
    if deviating:
        found["controller_deviations"] = (
            "evaluations.controller_deviations, per unit, variant and method that counts "
            f"deviations ({', '.join(deviating)}), summed over the (day, scale) pairs evaluated "
            "and the audited paths: hours, the hours the controller planned (0 to n on each "
            "path); deviating_hours, those in which least-laxity-first's split of the hour's "
            "target was found to leave the believed future infeasible and the total nearest the "
            "target among those found to keep it feasible was planned instead (mef_controller; "
            "a total other than least-laxity-first's, except where a car's remaining need (its "
            "believed need less its drawn energy and its believed recovery supply) is above its "
            "remaining room by the tolerance to within rounding); deviating_service_hours, those "
            "of them in service hours 1 to n; no_feasible_hours, the hours in which "
            "least-laxity-first's split was "
            "found infeasible and no total was found to keep the believed future feasible, "
            "planned as least-laxity-first plans them"
        )
    if sizing:
        names = ", ".join(sizing)
        found["timeouts"] += (
            f"; a sizing method's programme ({names}) that does not end optimal is counted "
            "under evaluations.sizing_solver_outcomes and handled as chance_sizing reading S11 "
            "says, its timeouts summed under sizing_programmes and oracle_sizing_programmes"
        )
        found["sizing"] = {
            "methods": (
                f"{names}: the re-solve method's day-level losses, with each split's scale chosen "
                "on the same offering calibration days by the method's chance rule of Lunde et "
                "al. (2025) (chance_sizing, chance_programmes) in place of crc's slot bound, "
                "crc's driver bound still applying in tier A (the scale there is the smaller of "
                "the two), and scored on the same test days at that scale "
                "(dial_splits.score_split)"
            ),
            "readings": dict(chance_sizing.READINGS),
            "sizing_solver_outcomes": (
                "evaluations.sizing_solver_outcomes, per unit, variant and sizing method: "
                "calibration_sets, the distinct sets of offering calibration days whose slot "
                "scale was solved (once per set, read by both tiers, the split rows, the "
                "summary and the gate reading); solves, the outcome of every programme solved "
                "for them, by outcome (lp.Outcome, zeros included)"
            ),
        }
    return found


def _arrays(found, matrix: Matrix, suffix: str = "") -> dict:
    return {
        ceiling + suffix: np.array(
            [np.nan if v is None else v for v in (ceiling_values(c, ceiling) for c in found)],
            dtype=float,
        )
        for ceiling in matrix.ceilings
    }


def _ceilings(pool: Pool, name: str, n_days: int, matrix: Matrix, timeouts: Counter):
    found = pool.map(_ceiling_task, [(name, i) for i in range(n_days)])
    timeouts["best_causal_ceilings"] += sum(c.best_causal_outcome == "timeout" for c in found)
    return [c.best_causal for c in found], _arrays(found, matrix)


def _reduced_ceilings(pool: Pool, name: str, draws, matrix: Matrix, timeouts: Counter) -> dict:
    """The ceilings of the fleet without the drawn cars (module text, reading N3)."""
    found = pool.map(_reduced_ceiling_task, [(name, i, d) for i, d in enumerate(draws)])
    timeouts["best_causal_ceilings_no_pause_cars_removed"] += sum(
        c.best_causal_outcome == "timeout" for c in found
    )
    return _arrays(found, matrix, NO_PAUSE_CEILING_SUFFIX)


def check_draw(matrix: Matrix, units: dict, share, seed: int) -> None:
    """Draws the cars of every counting day of every unit that can be split, as `run` does at
    `share` (nothing at share 0); raises `NoPauseError` naming the unit when the draw refuses a
    car (module text)."""
    if share == 0:
        return
    for name in matrix.gate_units:
        days = units[name]
        plan = gate_run.plan_unit(name, days, matrix)
        if plan.status != "calibratable":
            continue
        for truth in days.days[: len(plan.keys)]:
            try:
                no_pause.drawn(truth, share, seed)
            except NoPauseError as problem:
                raise NoPauseError(
                    f"unit {name}: {problem}; --no-pause-share {no_pause.share_text(share)} draws "
                    "each car from the text no-pause|<seed>|<site>|<session id>, so no site or "
                    "session identifier may hold '|'"
                ) from None


def run(
    matrix: Matrix,
    units: dict,
    factory,
    configurations: dict,
    methods,
    *,
    splits,
    workers,
    no_pause_share=no_pause.SHARES["0"],
    no_pause_seed=no_pause.DEFAULT_SEED,
    seeds=None,
):
    """Everything but the input checks and the writing of files: (results document, timing,
    per-split rows of `compare_splits`). `configurations`: variant -> configuration; `units`: unit
    name -> `UnitDays`; `no_pause_share`, `no_pause_seed`: the no-pause sensitivity (module
    text); `seeds`: the seeds of the `splits` splits, 1 to `splits` when None (on the route, the
    matrix's `split_seed` alone)."""
    started = time.perf_counter()
    no_pause.share_text(no_pause_share)
    sensitive = no_pause_share > 0
    seeds = list(range(1, splits + 1)) if seeds is None else list(seeds)
    if len(seeds) != splits:
        raise ValueError(f"{splits} splits need {splits} seeds, got {seeds}")
    tiers = dial_splits.TIERS
    ceiling_names = list(matrix.ceilings)
    if sensitive:
        ceiling_names += [c + NO_PAUSE_CEILING_SUFFIX for c in matrix.ceilings]
    pool = Pool(workers, units, matrix, factory)
    entries, timeouts, failed, timing = {}, Counter(), {}, {"units": {}}
    rows: list[dict] = []
    solving: set = set()
    deviating: set = set()
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
                "variants": {},
            }
            entries[name] = entry
            if plan.status != "calibratable":
                continue
            begun = time.perf_counter()
            best_causal, ceiling_arrays = _ceilings(pool, name, len(plan.keys), matrix, timeouts)
            draws = [()] * len(plan.keys)
            if sensitive:
                truths = days.days[: len(plan.keys)]
                draws = [no_pause.drawn(t, no_pause_share, no_pause_seed) for t in truths]
                entry["no_pause"] = no_pause.counts(truths, draws).as_dict()
                ceiling_arrays |= _reduced_ceilings(pool, name, draws, matrix, timeouts)
            indices = dial_choice.split_indices(name, days, matrix, seeds)
            losses = {}
            for variant, configuration in configurations.items():
                try:
                    prepared = pool.map(
                        _prepare_task,
                        [(name, i, configuration, draws[i]) for i in range(len(plan.keys))],
                    )
                    found, counts = measure_variant(
                        prepared,
                        best_causal,
                        methods,
                        lambda pairs: pool.map(_evaluate_task, pairs),
                        lambda jobs: pool.map(_compare_task, jobs),
                    )
                except Exception as problem:  # recorded, the other variants go on
                    failed[(name, variant)] = f"{type(problem).__name__}: {problem}"
                    entry["variants"][variant] = {
                        "configuration": configuration.name,
                        "failed": failed[(name, variant)],
                    }
                    continue
                losses[variant] = found
                rows += compare_splits.split_rows(
                    name, variant, found, indices, tiers, ceiling_arrays
                )
                solving.update(counts.get("controller_solver_outcomes", {}))
                deviating.update(counts.get("controller_deviations", {}))
                key = "certificates" if variant == "forecast" else "oracle_certificates"
                timeouts[key] += sum(p.certificate_outcome == "timeout" for p in prepared)
                key = "re_solves" if variant == "forecast" else "oracle_re_solves"
                timeouts[key] += counts["re_solves_of_the_one_way_rung"].get("timeout", 0)
                entry["variants"][variant] = {
                    "configuration": configuration.name,
                    "offering_days": int(sum(not p.abstains for p in prepared)),
                    "evaluations": counts,
                    "methods": score_methods(found, indices, tiers),
                    "measured_on": measured_on(prepared),
                }
                sized = sizing_counts(found)
                if sized:
                    counts["sizing_solver_outcomes"] = sized
                    key = "oracle_sizing_programmes"
                    if variant == "forecast":
                        key = "sizing_programmes"
                    timeouts[key] += sum(c["solves"]["timeout"] for c in sized.values())
                if sensitive:
                    reasons = Counter(why for p in prepared for _, why in p.unsteered)
                    entry["variants"][variant]["unsteered_cars"] = dict(sorted(reasons.items()))
            if set(losses) == set(VARIANTS):
                entry["gate_reading"] = gate_methods(
                    losses["forecast"],
                    losses["oracle-departure"],
                    ceiling_arrays,
                    indices,
                    ceiling_names,
                )
            timing["units"][name] = time.perf_counter() - begun
    finally:
        pool.close()
    document = {
        "label": LABEL,
        "definitions": definitions(
            matrix,
            splits,
            [m.name for m in methods if m.name in solving],
            [m.name for m in methods if m.name in deviating],
            [m.name for m in methods if m.kind == "sizing"],
        ),
        "configurations": {variant: c.name for variant, c in configurations.items()},
        "dial": configurations["forecast"].driver_risk,
        "methods": [m.name for m in methods],
        "run": {
            "splits": splits,
            "seeds": [seeds[0], seeds[-1]] if seeds else [],
            "scales": list(dial_splits.GRID),
            "n_calibration": matrix.n_calibration,
            "calibration_half_below": matrix.calibration_half_below,
            "kappa": configurations["forecast"].kappa,
            "tiers": list(tiers),
            "ceilings": ceiling_names,
            "variants": list(configurations),
        },
        "units": entries,
        "timeouts": dict(sorted(timeouts.items())),
        "failed": sorted(f"{a} | {b}: {why}" for (a, b), why in failed.items()),
        "paired_differences": {
            "definitions": dict(PAIRED_DEFINITIONS),
            "units": compare_paired.paired(
                rows,
                [u for u, e in entries.items() if e["status"] == "calibratable"],
                list(configurations),
                tiers,
                [m.name for m in methods],
                seeds,
            ),
        },
    }
    if sensitive:
        share = no_pause.share_text(no_pause_share)
        document["label"] = document["definitions"]["label"] = no_pause_label(
            no_pause_share, no_pause_seed
        )
        document["no_pause"] = {
            "share": share,
            "seed": no_pause_seed,
            "readings": dict(NO_PAUSE_READINGS),
            "counts": (
                "units.<unit>.no_pause, over the unit's counting days: cars (a session on "
                "several days counted once per day), cars_drawn, days_every_car_drawn; "
                "units.<unit>.variants.<variant>.unsteered_cars: the unsteered cars of the "
                "variant's prepared days, counted by reason (gate_day): a car on several counting "
                "days once per day; on a day that abstains under day-abstains, the drawn and the "
                "uncertifiable cars and not the day's other cars"
            ),
        }
        timing["no_pause"] = {"share": share, "seed": no_pause_seed}
    timing["total_seconds"] = time.perf_counter() - started
    return document, timing, rows


def dial_configurations(matrix: Matrix, dial: float) -> dict:
    """Variant -> configuration for `dial` of the matrix's dial family; raises `MatrixError` when
    the matrix has no dial family or `dial` is not one of its dials."""
    _, dials = dial_choice.find_family(matrix)
    chosen = [c for c in dials if c.driver_risk == dial]
    if not chosen:
        values = ", ".join(f"{c.driver_risk:g}" for c in dials)
        raise MatrixError(f"--dial {dial:g} is not a dial of the matrix's dial family ({values})")
    (configuration,) = chosen
    return {
        "forecast": configuration,
        "oracle-departure": matrix.configuration(configuration.oracle_partner),
    }


def mark_route(document: dict, route, held_out_inputs, fit_units, matrix: Matrix, share, seed):
    """On the route (module text): the held-out label, the one split's definition, the paired
    differences' interval, the readings H1 to H6 and the route's record (`sealed_route.mark`);
    nothing off the route."""
    if not route.held_out:
        return
    label = ROUTE_LABEL if share == 0 else no_pause_label(share, seed, ROUTE_LABEL)
    found = document["definitions"]
    document["label"] = found["label"] = label
    found["splits"] = ROUTE_SPLITS.format(
        seed=matrix.split_seed,
        n_calibration=matrix.n_calibration,
        half_below=matrix.calibration_half_below,
    )
    found["route_readings"] = dict(ROUTE_READINGS)
    document["paired_differences"]["definitions"]["interval"] = ROUTE_INTERVAL
    sealed_route.mark(document, route, held_out_inputs, fit_units)


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None, root: Path = TOP) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dial", type=float, required=True)
    parser.add_argument("--matrix", type=Path, default=dial_choice.MATRIX)
    parser.add_argument("--splits", type=int, default=dial_choice.SPLITS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--no-pause-share", default="0")
    parser.add_argument("--no-pause-seed", type=int, default=no_pause.DEFAULT_SEED)
    parser.add_argument("--held-out-tables", type=Path, default=None)
    parser.add_argument("--held-out-truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        gate_run.argument_paths(args)  # reading H4
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if args.splits < 1:
            raise MatrixError("--splits must be at least 1")
        share = no_pause.read_share(args.no_pause_share)
        methods = compare_methods.select(args.methods)
        gate_run.check_files([args.matrix])
        matrix, matrix_sha256 = read(args.matrix)
        configurations = dial_configurations(matrix, args.dial)
    except (MatrixError, MethodError, NoPauseError, CouldNotRun) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
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
    if route.held_out and args.splits != 1:  # reading H1
        print(f"COULD NOT RUN: {ONE_SPLIT.format(splits=args.splits)}")
        return 2
    try:
        model = learn_run.load_model(args.model, matrix)
    except MatrixError as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    fit_units = tuple(model.fit_units)
    try:
        reader = learn_run.check_model(args.model, model, fit_units, matrix.gate_units)
    except learn_run.FitOnScored as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    if route.held_out and reader:
        print(f"COULD NOT RUN: {sealed_route.OUTCOME_READER.format(what=f'model {args.model}')}")
        return 2
    try:
        sealed_route.gate_again(route, root)
    except sealed_route.RouteRefused as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    needed = tuple(dict.fromkeys((*matrix.gate_units, *fit_units)))
    held_inputs = None
    try:
        if route.held_out:
            recorded = sealed_route.check_split_rule(matrix, route)
            inputs, held_inputs, table_bytes = sealed_route.read_two_builds(
                args.truth_run,
                args.tables,
                fit_units,
                route.truth_run,
                route.tables,
                matrix.gate_units,
            )
            matrix = sealed_route.scored_matrix(matrix, held_inputs)  # abstaining units out
            needed = tuple(dict.fromkeys((*matrix.gate_units, *fit_units)))
        else:
            inputs = gate_run.check_inputs(args.truth_run, args.tables, needed)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    readings = {"rho": matrix.rho, "no_fit": matrix.no_fit, "small_fleet": matrix.small_fleet}
    try:
        if route.held_out:
            loaded = {name: parse_unit_days(table_bytes, name, **readings) for name in needed}
        else:
            loaded = {name: load_unit_days(args.tables, name, **readings) for name in needed}
    except CouldNotRun as problem:  # a table changed after its check (on the route)
        print(f"COULD NOT RUN: {problem}")
        return 2
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    units = {name: loaded[name] for name in matrix.gate_units}
    if route.held_out:
        try:
            sealed_route.check_split_days(matrix, units, recorded, route.truth_run.name)
        except CouldNotRun as problem:
            print(f"COULD NOT RUN: {problem}")
            return 2
    try:
        check_draw(matrix, units, share, args.no_pause_seed)
    except NoPauseError as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    fit_frames = {name: loaded[name].cars for name in fit_units}
    kappa = configurations["forecast"].kappa
    try:
        factory, table = dial_choice.make_factory(
            args.model, model, matrix, units, fit_frames, kappa
        )
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: the model could not be scored: {problem}")
        return 1
    document, timing, rows = run(
        matrix,
        units,
        factory,
        configurations,
        methods,
        splits=args.splits,
        workers=args.workers,
        no_pause_share=share,
        no_pause_seed=args.no_pause_seed,
        seeds=[matrix.split_seed] if route.held_out else None,  # reading H1
    )
    names = output_names(share)
    document["model"] = model.name
    document["fit_units"] = list(fit_units)
    rotation_whatif.mark(document, matrix, fit_units)
    document["inputs"] = inputs
    document["matrix"] = {"file": args.matrix.name, "sha256": matrix_sha256}
    mark_route(document, route, held_inputs, fit_units, matrix, share, args.no_pause_seed)
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        write_table(table, folder / names["table"])
        document["forecast_table"] = {
            "file": names["table"],
            "rows": len(table),
            "sha256": hashlib.sha256((folder / names["table"]).read_bytes()).hexdigest(),
        }
        count = compare_splits.write(rows, folder / names["splits"], document["run"]["ceilings"])
        document["split_table"] = {
            "file": names["splits"],
            "rows": count,
            "sha256": hashlib.sha256((folder / names["splits"]).read_bytes()).hexdigest(),
        }
    text = "\n".join(compare_summary.summary_lines(document)) + "\n"
    print(text, end="")
    if run_dir:
        (folder / names["results"]).write_text(_json(document))
        (folder / names["summary"]).write_text(text)
        (folder / names["timing"]).write_text(_json(timing))
    return 1 if document["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
