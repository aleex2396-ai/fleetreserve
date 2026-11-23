"""The corner search of each method's calibrated box (plan section 11 O7; plan 13 WP11).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.corner_run --model <name> --dial <value> [--matrix <file>]
        [--days N] [--methods <name> ...] [--workers N] [--seed N] [--tables <folder>]
        [--truth-run <folder>] [--held-out-tables <folder>] [--held-out-truth-run <folder>]

Plan section 11, O7: "corner search of each method's calibrated box, 50 fixed random days per
unit, with and without oracle departures: Y% of days break ASA, LLF or maximum-entropy feedback,
Y0% the fixed plan." Plan 13, WP11: "Corner search; run matrix frozen. Pass: closed-form worst
case reproduced on 20 days." This command does the search and the pass mark's check; it does not
freeze a run matrix.

Inputs as `compare_run` takes them, by the same calls but the matrix file's (read here by
`gate_config.load`, below, and by `compare_run` once by `gate_config.read`): the model
(`learn_run.load_model`, `learn_run.check_model`, `dial_choice.make_factory` at the dial
configuration's kappa), the
truth tables (`gate_run.check_inputs`; on the sealed route `sealed_route.read_two_builds`, below),
the units (the matrix's gate units), `--dial` (one dial
of the matrix's dial family, `compare_run.dial_configurations`, which also gives the variants:
"forecast", the dial's configuration, and "oracle-departure", its oracle partner) and
`--methods` (`compare_methods.select`; default every method of `compare_methods.METHODS`, in its
order). The days are prepared by `compare_run`'s pool task (`gate_day.prepare_day`) and the
calibration losses come from `compare_run.measure_variant`; no loader or calibration is written
again here. The search is of the whole fleet: it has no no-pause option (plan section 9;
`compare_run --no-pause-share`), and its preparation task is given no drawn car
(`prepare_jobs`).

Path arguments and the matrix file, on and off the route (the route is known only once the
matrix's gate units are read): each path argument is first put through `gate_run.argument_path`
(a path written as a URL could not run; a leading `~` is expanded), and the matrix file passes
`gate_run.check_files` before it is read (`gate_config.load`; the SHA-256 recorded is that of
its bytes read again when the results are written, as before).

Held-out units (plan section 13: WP12 rehearses the sealed run on development units, WP14 runs it
once from the tag; the cut order lists "corner search beyond 50 days per unit" as an item to cut,
so the search on 50 days per held-out unit is part of that run), through the one shared route of
`sealed_route` (`sealed_route.matrix_route`, as `learn_run` has it), which starts only at the tag
`frozen-plan` with a clean tree; its checks stop an honest operator's ordinary mistakes and are
not a defence against deliberate evasion (`sealed_route`, module text, first paragraph). When the
matrix's gate units are held-out units (all of them; a mix with other units could not run), in
this order, each before any table is read: every folder named lies outside the held-out places of
the top folder, the tag gate passes (the held-out loader's checks 2 to 8: at the tag
`frozen-plan`, clean, the running code the tag's), `--held-out-truth-run` is given
(`--held-out-tables` defaults to the held-out build's main folder), and `--seed`, `--days` and
`--methods` are the route's (`route_refusal`, readings H1 to H3); the model is loaded and
checked, a model of `forecast_model.OUTCOME_READERS` could not run (`sealed_route.OUTCOME_READER`),
and the gate is passed again (the model's module is then among the running code); the matrix's
split rule must be
the held-out build's (`sealed_route.check_split_rule`). The fit units' tables are then read from
the development build (`--tables`, `--truth-run`) and the scored units' from the held-out build
(`--held-out-tables`, `--held-out-truth-run`), each build checked and read once and the two of one
setting (`sealed_route.read_two_builds`), and the bytes hashed are the bytes parsed
(`days_from_truth.parse_unit_days`); each scored unit's calibration days as the run draws them
must be the build's (`sealed_route.check_split_days`). Everything else runs as off the route.
`corner.json` then holds `inputs` (the development build's check, of the fit units' tables only)
and `sealed_route` (`sealed_route.mark`: the tag's commit, the held-out build's check of the
scored units' tables, the fit units, `kappa_selection`: none, and `checks`: what the route's checks
stop and what they do not); its label is `HELD_OUT_LABEL` in place of `LABEL`, its definitions
gain the readings H1 to H7 and the route's two fields, and the closed-form check's fields, their
definitions and reading K3 are named as H6 says (`mark_route`; the timing file's path-hours too,
`route_timing`). The check runs at the fixed plan's calibrated scale alone (H5; `run` and
`measure_variant` with `held_out`), and a unit and variant it does not check is recorded as H7
says. The summary's first two lines are the route's (`sealed_route.summary_head`), its first line
of the search names `HELD_OUT_LABEL`, and the check's three headings name it a closed-form
property check on held-out days (`corner_summary.summary_lines` with `held_out`).
With other gate units the two held-out options could not run, and without them nothing above is
checked: the run, its outputs and its summary are as before.

What a run fits, chooses, samples or calibrates, and from which days (on the route; off it the
same steps on development units):
  L1. The forecast model is fitted on its `fit_units` alone, training units read from the
      development build, never on a scored unit (`learn_run.check_model`); it forecasts the
      scored units' cars from inputs known at τ (`sealed_route` reading S1). A model that builds a
      scored unit's forecasts from that unit's own other days, test days included
      (`forecast_model.OUTCOME_READERS`: `empirical`), could not run on the route; a look-ahead
      model is refused (exit 1) on and off the route (`learn_run.check_model`, here and in
      `dial_choice.make_factory`).
  L2. Kappa is the dial configuration's, from the matrix; nothing selects kappa per unit.
  L3. The dial is `--dial`, taken as given (reading H4).
  L4. The split is the held-out build's one draw (reading H1): the calibration days are the days
      the build drew and searched the unit's window on, the test days the rest.
  L5. Each method's scale is calibrated on the split's calibration days alone (W1: the losses of
      `compare_run.measure_variant` on those days; a sizing method's chance rule on the same
      losses, W5); no test day enters a scale.
  L6. The searched days are the split's test days in an order drawn from the unit's name and the
      seed alone (K5), fixed before any day is prepared; in each variant the first `--days` of
      them that do not abstain (`gate_day.prepare_day`: under the forecast variant by the
      forecasts, information at τ; under the oracle-departure variant by the true departures,
      which is what that variant is).
  L7. The corner search is the measurement, and it sets nothing. Per method and searched day it
      reports whether some corner of the method's calibrated box breaks it: the worst case over
      the box for the fixed plan, a lower bound of it for every other method. Every scale is
      calibrated and every searched day chosen before the search runs (`measure_variant`), and
      nothing the search finds is read back into a setting of any method (a test replaces its
      results and finds the same scales, searched days and calibration counts).
  L8. The closed-form check (K3) compares, on the first `CHECK_DAYS` searched days, the fixed
      plan's closed form with the corners; it sets nothing. On the route it runs at one scale, the
      fixed plan's scale of L5, so no test day is scored at a scale not chosen (reading H5).
  L9. No other command's run folder is read: the only runs read are the records of the two truth
      builds (`--truth-run`, `--held-out-truth-run`), each verified at the commit checked out now.

Readings of the route (each with its alternative in brackets):
  H1. The split. A held-out unit has one split, the one its held-out build drew (plan section 6,
      Protocol: "one draw for all policies"). On the route `--seed` must be the matrix's
      `split_seed`, the matrix's split rule the build's (`sealed_route.check_split_rule`) and each
      scored unit's calibration days as drawn the build's (`sealed_route.check_split_days`), or the
      run could not run; the same seed orders the test days for the search (K5). [Alternatives:
      `--seed` ignored on the route and the build's seed taken; any seed accepted, a second draw
      of the unit's days, which the protocol does not have.]
  H2. The days. On the route `--days` below `DAYS` (50) could not run: O7 searches "50 fixed
      random days per unit", and only the search beyond 50 days per unit may be cut, so fewer are
      a trial; 50 or more run. A unit with fewer offering test days than asked gets all it has,
      as off the route (W2). [Alternatives: exactly 50; any number, the results saying how many.]
  H3. The methods. `--methods` is accepted on the route: one list for every unit, fixed before
      the run (the owner may cut maximum-entropy feedback, ALSO-X and CVaR: cut order items 4 and
      8), never chosen per unit by looking at results. It must hold the fixed plan, whose
      calibrated scale is the scale of the closed-form check on held-out test days (H5); without
      it the run could not run (`route_refusal`, exit 2): the matrix file has then been read (its
      gate units decide the route) and the route's folder and tag checks passed, but no table is
      read, no model loaded and nothing in the sealed folder opened. [Alternatives:
      `--methods` refused on the route, as `learn_run` refuses `--only`, every method of the
      registry searched; a list without the fixed plan accepted and the fixed plan calibrated
      apart for the check alone.]
  H4. The dial. `--dial` is one dial of the matrix's dial family, fixed before the tag per site
      type (plan section 6); the route takes it as given and checks nothing more about it.
      [Alternative: a check against a committed record of the dial of each site type; none exists
      yet.]
  H5. The closed-form check on held-out days. On the route it runs on the first `CHECK_DAYS`
      searched days, which are held-out test days, at one scale: the fixed plan's calibrated
      tier A scale of that unit and variant, the value `calibrate` chose on the calibration days
      and the search uses (`check_scales`), not every scale of `dial_splits.GRID`. Plan section 6
      (Protocol): "test days are scored only at the chosen β"; the owner's decision of 16 November
      2026 applies it to this check. A disagreement gives exit 1 there too. It is a property check
      (the closed form is the fixed plan's exact worst case), not WP11's pass mark, which
      `tools/corner_closed_form_check.py` reads only from a run on the committed
      `gate/matrix.json`, a matrix of development units. [Not taken: every scale of the grid on
      the test days, which scores the fixed plan on test days at scales not chosen; the check left
      out on the route; every scale on the split's calibration days in place of the test days.]
  H6. The names. So that no field or heading of a run on the route calls the check what H5 says
      it is not, on the route its fields are named `closed_form_check` in place of `pass_mark`
      (the document's and each variant's, their definitions, the timing file's path-hours), K3
      and the route's definitions of the check (`ROUTE_CHECK_DEFINITIONS`) name it a closed-form
      property check on held-out days at the fixed plan's calibrated scale, and so do the
      summary's three headings (its paragraph, each unit's table, the line over all units). The
      one sentence that says what it is not is the label's (`HELD_OUT_LABEL`, also in the
      summary's third line), with H5 and H6 among the definitions. [Alternative: the field names
      kept and only the headings and texts changed, so that one name is read on both kinds of
      run; the name would then still call a check on held-out days a pass mark.]
  H7. A unit and variant the route does not check. Where the fixed plan's calibrated scale is 0
      (it offers nothing; also where every calibration day abstains) there is no chosen scale,
      and where no day was searched (no test day offers under the variant) there is no day: the
      variant's check entry is then `status` "not checked" with its `reason` ("the fixed plan
      offers nothing (scale 0, <why>)", or "no day searched (no test day offers under the
      variant)", `unchecked_reason`), it is never counted as agreed and it does not decide the
      exit status. The run's check field sums the checked ones alone, lists the unchecked ones
      under `not_checked` ("<unit> | <variant>: <reason>"; a variant that failed is under `failed`
      and a unit that cannot be split has its status, neither there), and its `agreed` is null
      when none was checked (null as for every figure of nothing in these results), the summary's
      line over all units then saying "nothing checked on this run". On the route the exit
      status is 1 for a failed variant, for a disagreement of a checked pair and for no unit that
      could be split (a run in which no unit is calibrated or searched), and 0 otherwise
      (`exit_status`). Of the runs on which nothing was checked, those with no unit that could be
      split, or with a failed variant, exit 1; those whose units were split and whose every
      variant was left unchecked (every fixed plan at scale 0, no day searched, or some of each)
      exit 0. On the route exit 0 therefore means that every split unit was calibrated and
      searched (on no day where none offers), no variant failed and no checked pair disagreed; it
      does not mean that any pair agreed: whether anything was checked is said by `agreed` (null
      when nothing was) and by the summary's words ("nothing checked on this run"). Off the route
      the rule of the exit status below is unchanged. [Alternatives: the development rule on the
      route too, exit 1 where one unit's fixed plan offers nothing or one variant searched no
      day; every run on which nothing was checked exiting non-zero.]

Readings of the plan's wording (each is the closest reading; the alternative is in brackets):

  W1. "Each method's calibrated box": the day's certified box at the method's own calibrated tier
      A scale on one seeded split of the unit, the split `dial_choice.split_indices` gives for the
      seed `--seed` (default 1); each method is calibrated on that split's calibration days as
      `compare_run` calibrates it (its losses on the audited paths at every scale of
      `dial_splits.GRID`, `dial_splits.score_split` for tier A), per variant. [Alternative: the
      scale of every split, or tier B.]
  W2. "50 fixed random days per unit": the split's test days in an order drawn once per unit from
      the seed (`day_order`, before any day is prepared), and in each variant the first `--days`
      (default 50) of them that offer under that variant. A unit with fewer gets all it has, and
      the results and the summary say so. The two variants can search different days.
      [Alternative: the days that offer under both variants.]
  W3. The methods are those the registry holds; no method name is written into this command's
      logic. The pass mark's check reads the fixed plan itself (`corner_search.closed_form_check`),
      whichever methods are searched.
  W4. A method whose calibrated scale is 0 offers nothing: "no offer", with no break figures, never
      "does not break".
  W5. A sizing method's calibrated box (`also-x`, `cvar`: kind "sizing" of `compare_methods`,
      plan section 10): a sizing method operates exactly as the re-solve (the one-way re-solve
      rung on the day's certified box) and differs from it only in its scale, which its chance
      rule (`chance_sizing`) chooses on the same calibration losses as W1 says; so its box is the
      certified box at its own tier A scale and its corners are the re-solve's corners at that
      scale (`corner_search`): where the two scales are equal the two methods' days are equal
      but for the name; where they differ the sizing method's search tells what its own scale
      allows. Its sizing programmes on the calibration days are counted under the variant's
      `calibration_evaluations.sizing_solver_outcomes` as `compare_run` counts them, and their
      timeouts under `sizing_programmes_calibration` (`oracle_` in front for the oracle
      variant). This reading, those counts and keys are written only when a sizing method is
      searched, so a run without one is unchanged. [Alternative: sizing methods left out of the
      corner search, the reason written into the results; O7 then has no row for them, though
      plan section 10 scores them with the same calibration and loss as every other method.]
  K1. A corner breaks a method when an hour fails by plan section 1's rule (a delivery missing
      its request by more than `replay.FAILURE_KWH`, 1e-4 kWh: a slot loss above 0) or when it
      leaves more than the same `FAILURE_KWH` of protected energy undelivered at departure. Plan
      section 1 gives no tolerance for the driver loss; `FAILURE_KWH` is used so that a car one
      rounding step short of full is not a break (the rule reads the amount, not its cause: a
      shortfall that small is not a break whatever left it). The figures by kind are given
      beside (days with a slot break, days with a driver shortfall), and the days that hold but
      have a corner short by more than 0 and at most `FAILURE_KWH` are counted. [Alternatives: a
      driver loss above 0, which counts rounding as breaks; the slot loss alone, which is the
      "by slot loss" figure.]
  K2. A re-solve that times out leaves that corner without a result (`corner_search`); a day with
      a breaking corner without a timeout breaks, otherwise a day with a timed-out corner is
      "timed out", counted and left out of the shares. [Alternative: read the rung's fallback to
      its predecessor's plan as the result.]
  K3. The pass mark (on the sealed route a property check at the fixed plan's calibrated scale
      alone, H5 to H7): on the first `CHECK_DAYS` (20) searched days of each unit and variant, at
      every scale of `dial_splits.GRID` (the scales a calibration can choose), the fixed plan's
      closed form (`gate_calibrate.slot_steps` on the hour thresholds) against the worst case of
      replay over all corners (`corner_search.closed_form_check`). A pair agrees strictly when
      the first failing hours are equal and the slot losses differ by at most
      `corner_search.SLOT_TOLERANCE`; it agrees when it agrees strictly or when
      `fixed_plan_loss.compare_with_replay` labels the difference "within tolerance" (at the
      closed form's failing hour every corner's draws miss the plan by at most `FAILURE_KWH` in
      total, so by the failure rule of plan section 1 replay does not fail there; on the
      command's made-up early units it arises by rounding, an hour threshold of
      0.9999999999999998 where the certified box meets a car's room exactly). Every
      other difference is a disagreement and gives exit 1 (the closed form counts each car's
      decision-hour draw, so a car capped in hour 0 makes none, `fixed_plan_loss` module text;
      before the owner's decision of 16 November 2026 it made a "decision-hour cap" difference,
      a disagreement). Both counts are given,
      and the differences by cause. [Alternatives: only at the fixed plan's calibrated scale;
      strict agreement alone, under which rounding decides the exit status.]
  K4. The driver loss has a bound in closed form, not a worst case: the every-scale bound
      (`gate_calibrate.every_scale_loss`; `fixed_plan_loss` module text, reading R3, V1 to V8),
      which counts each car's decision-hour draw and bounds replay's driver loss at every scale
      up to the errors stated there. The pairs whose corner maximum is above it are counted and
      reported, and do not decide the exit status; beside them, for comparison only, the pairs
      above the earlier bound (reading R2), which is not a bound at a scale where an hour fails.
      Of the pairs above the every-scale bound, two kinds are counted, read on the corners
      whose driver loss is within `corner_search.DRIVER_TOLERANCE` of the largest (the corners
      of the largest loss): those on which every such corner has a car that drew less than
      planned, and those on which some such corner has no failing hour (`corner_search`). The
      counts do not depend on the order of the corners; when such corners differ in a flag (on
      a made-up day of the tests one of four fails an hour and three do not), the flag is read
      towards the case that questions the bound (a corner of the largest loss that delivered its
      plan) or the slot loss (that loss reached with no failing hour). [Alternative: the first
      corner of the exact largest loss in corner order, whose flags then depend on the order and
      on rounding.]
  K5. The day order: numpy's default generator seeded with the first 8 bytes of the SHA-256 of
      "<unit>|<seed>|corner-days", a permutation of the split's test days.

Per unit that can be split (`gate_run.plan_unit`): per variant, every counting day prepared; the
calibration on the split's calibration days; the searched days (W2); every method with a scale
above 0 on every corner of each searched day (`corner_search.search_day`); the pass-mark check.
A unit that cannot be split is reported with its status and no numbers. Timeouts are recorded
outcomes: a day whose certificate times out abstains (it offers nothing) and the re-solves that
time out are counted per variant, in the calibration and in the search. A controller that solves
programmes (`asa`, `asa-es`, `diagonal-split`: one with a method `solver_counts`) is counted as
`compare_run` counts it: in the calibration, the calibration
evaluations' `controller_solver_outcomes` of `compare_run.measure_variant`; in the search, the
method's `solver_counts` (solves by outcome, hours planned, hours planned by its fall-back, summed
over every corner of every searched day); its solves that time out are also added to the timeouts.
An hour whose programme is not optimal is planned by the controller's fall-back, as in
`compare_run`. A controller that counts its deviations (`max-entropy-feedback`, plan section 11
O7: one with a method `deviation_counts`) is searched as every other controller is; its counts
summed over every corner of every searched day are its entry's `deviation_counts`, with one line
of the summary per variant, and `definitions.fields` gains `method.deviation_counts` only when
some method wrote it, so a run without such a method is unchanged. A sizing method's programmes
run in the calibration only (W5); its re-solves on the searched corners are counted with the
re-solves.

The corners give the worst case over the box for the fixed plan only (reading P1 of
`corner_search`); for every other method they are a search, and an interior path can be worse
than every corner (`corner_search` gives an example), so a method's break share is a lower bound
of what its box allows.

Cost. Path-hours are paths times service hours, summed over days and methods, per unit and
variant (`count_path_hours`): of the calibration (the audited rows of each offering calibration
day at every scale of the grid, for every set of losses measured: one per method, but a sizing
method reads the re-solve's, measured once, `measured_losses`), of the search as an upper bound
(every corner of every searched day for every method) and of the pass-mark check (every corner
of each checked day at every check scale). They are counted once the days of a unit and variant
are prepared, before its calibration runs, and the search's count as run (methods with a scale
above 0) once its scales are known, before the search runs; with `FLEETRESERVE_RUN_DIR` set the
timing file is written at each of these points, and again at the end. On the sealed route the
check's count is first of one scale, the most it can be (H5), and once the scales are known the
count as run: one scale, or 0 for a unit and variant not checked (H7).

The per-day work runs in this process (`--workers 1`) or in `compare_run`'s pool of worker
processes; the results, the rows and the summary are the same bytes for 1, 2 and 3 workers on the
made-up units of the tests; wall-clock seconds are kept apart.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary also on
standard output:
- `corner.json`: inputs (model, dial, configurations, matrix hash, truth tables' hashes, split
  seed, the number of rows of the model's forecast table), per unit, variant and method the
  figures of `corner_summary.method_entry`, per unit and variant the pass-mark check
  (`corner_summary.pass_mark_entry`; on the route one not run, `corner_summary.unchecked_entry`,
  H7) and the searched days, and `definitions`: the readings and,
  under `fields`, an entry per field written at every level (the document and its `run`,
  `pass_mark` and `matrix`; each unit, variant, method, pass-mark check and disagreement; each
  CSV column), in `FIELD_DEFINITIONS` and `corner_summary.DEFINITIONS` (on the sealed route the
  check's fields and their definitions under the names of reading H6). A field that holds a
  record whose keys are not among those levels (for example `configurations`, `inputs`,
  `calibration_evaluations`, `timeouts`, `resolves`, `solver_counts`, `rotation_whatif`) has one
  entry for the whole record, not one per key inside it; the keys of `timeouts` are defined
  under `definitions.timeouts`;
- `corner-days.csv`: one row per unit, variant, method and searched day
  (`corner_summary.ROW_FIELDS`);
- `corner-summary.txt`: the text of `corner_summary.summary_lines`;
- `corner-timing.json`: wall-clock seconds and the path-hour counts.

Exit status (on the sealed route as reading H7 says instead, where 0 can come with nothing
checked; 0 and 1 below are off the route): 0 every unit and
variant ran and its pass-mark check agreed (at least one pair
checked, every pair agreeing); 1 a pass-mark pair disagreed, a unit and variant that ran had no
pair to check, no unit could be split, or a unit and variant failed (recorded; the others go on);
2 could not run: an unknown
model, a `--dial` not in the dial family, a missing truth build, a method list `compare_methods`
refuses, a number of workers, days or a seed below 1, a path written as a URL, a matrix file in a
refused place (`gate_run.check_files`), and for held-out units a refusal of the route (above;
among them `--methods` without the fixed plan, H3).
Handled as `compare_run.main` handles them, with the same exit codes and by the same calls but
the matrix file's (read here by `gate_config.load`, and by `compare_run.main` once by
`gate_config.read`, its reading H4), and not run by the tests of this command: exit 2 for a matrix
without a dial family or one `gate_config.load` refuses, a model fitted on a scored unit, a table
missing, a truth build at another commit and a table differing from it; exit 1 for a refused
model, a model that fails and a day that cannot be built from the tables. The refusals of the
route are run by `tests/wp2/test_corner_route.py`.

Untested: a real run (on held-out units only made-up ones, at a made-up tag); a rotation matrix;
a solve of `asa` that times out on a corner (no test
makes one time out, so the `controller_solves_*` timeout counts are seen only at 0); a sizing
programme that times out in this command's calibration (its sum is tested by hand counts, the
command's `sizing_programmes_calibration` seen only at 0).
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from fleetreserve import (
    compare_methods,
    compare_run,
    corner_search,
    corner_summary,
    dial_choice,
    dial_splits,
    gate_run,
    learn_run,
    rotation_whatif,
    sealed_route,
)
from fleetreserve.audit_paths import N_SEEDED
from fleetreserve.compare_methods import MethodError
from fleetreserve.days_from_truth import load_unit_days, parse_unit_days
from fleetreserve.gate_config import Matrix, MatrixError, load
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

RESULTS_NAME = "corner.json"
ROWS_NAME = "corner-days.csv"
SUMMARY_NAME = "corner-summary.txt"
TIMING_NAME = "corner-timing.json"
VARIANTS = compare_run.VARIANTS
DAYS = 50
CHECK_DAYS = 20
SEED = 1
TIER = "A"
AUDITED_ROWS = 3 + N_SEEDED
LABEL = "corner search on development units: development evidence, not a held-out result"
# The label of a run on the sealed route, in place of `LABEL` (module text).
HELD_OUT_LABEL = (
    f"corner search on held-out units through the sealed-data route {sealed_route.AT_THE_TAG}: a "
    "held-out result (plan section 11, O7); its closed-form check runs on these days too, at the "
    "fixed plan's "
    "calibrated scale, and is not WP11's pass mark, which is read on development units"
)
# Reading K3's rule of agreement, after the days and scales checked (`CHECK_READING` off the route,
# `ROUTE_K3` on it, readings H5 and H6).
CHECK_RULE = (
    "the closed form (gate_calibrate.slot_steps on the hour "
    "thresholds) agrees strictly with the worst case over all corners when the first "
    "failing hours are equal and the slot losses differ by at most "
    "corner_search.SLOT_TOLERANCE; it agrees when strictly or when "
    "fixed_plan_loss.compare_with_replay labels the difference 'within tolerance'; every "
    "other difference is a disagreement; the closed form counts each car's decision-hour "
    "draw"
)
# Reading K3's check, named in front by what it is: off the route WP11's pass mark.
CHECK_READING = (
    f"the first {CHECK_DAYS} searched days of each unit and variant at every "
    "scale of dial_splits.GRID; " + CHECK_RULE
)
# The fixed plan's name in the registry: its calibrated scale is the check's scale on the route
# (reading H5), and the route needs it searched (H3).
FIXED_PLAN = "fixed-plan"
# `run.check_scales` of a run on the route (reading H5), in place of the grid.
ROUTE_CHECK_SCALES = (
    "the fixed plan's calibrated scale of that unit and variant alone (none where it offers "
    "nothing or where no day was searched)"
)

READINGS = {
    "W1": (
        "each method's calibrated box: the certified box at the method's own tier A scale on the "
        "split of seed --seed (dial_choice.split_indices), calibrated on that split's "
        "calibration days as compare_run calibrates (dial_splits.score_split), per variant"
    ),
    "W2": (
        "the searched days: the split's test days in a seeded order fixed before any day is "
        "prepared, and per variant the first --days of them that offer under that variant; a "
        "unit with fewer gets all it has, and says so"
    ),
    "W3": "the methods are the registry's (compare_methods.METHODS) or those --methods names",
    "W4": "a method whose calibrated scale is 0 offers nothing: 'no offer', not 'does not break'",
    "K1": (
        "a corner breaks a method when an hour fails by plan section 1's rule (delivery missing "
        "its request by more than replay.FAILURE_KWH) or when more than replay.FAILURE_KWH of "
        "protected energy is undelivered at departure; days short by at most that are counted "
        "apart, not as breaks"
    ),
    "K2": (
        "a corner on which a re-solve timed out has no result; a day breaks when a corner "
        "without a timeout breaks, else it is 'timed out' if a corner timed out, else it holds"
    ),
    "K3": "pass mark: " + CHECK_READING,
    "K4": (
        "the driver loss has a closed-form bound, not a worst case: the every-scale bound "
        "(gate_calibrate.every_scale_loss; fixed_plan_loss, reading R3, V1 to V8), which counts "
        "each car's decision-hour draw and bounds replay's driver loss at every scale up to the "
        "errors stated there; corner maxima above it are counted and reported and do not decide "
        "the exit status, and beside them, for comparison only, those above the earlier bound "
        "(reading R2), which is not a bound at a scale where an hour fails; of the pairs above "
        "the every-scale bound are counted those "
        "on which every corner of the largest driver loss (within "
        "corner_search.DRIVER_TOLERANCE of it) has a car that drew less than planned, and "
        "those on which some such corner has no failing hour, whatever the order of the corners"
    ),
    "K5": (
        "day order: numpy's default generator seeded with the first 8 bytes of the SHA-256 of "
        "'<unit>|<seed>|corner-days', a permutation of the split's test days"
    ),
}
# Written only when a sizing method is searched (W5 in the module text).
SIZING_READING = (
    "a sizing method (kind 'sizing': also-x, cvar) operates as the re-solve and differs only in "
    "its scale, chosen by its chance rule (chance_sizing) on the same calibration losses: its "
    "corners are the re-solve's at its own tier A scale; its sizing programmes are counted "
    "under calibration_evaluations.sizing_solver_outcomes"
)

# The fields this command writes above the method and pass-mark entries (those and the CSV
# columns are in corner_summary.DEFINITIONS).
FIELD_DEFINITIONS = {
    "document.label": "what the results are (with a rotation what-if's label in front on one)",
    "document.definitions": "these definitions",
    "document.configurations": "per variant the name of its configuration of the matrix",
    "document.dial": "the dial: the driver-risk level of the forecast variant's configuration",
    "document.methods": "the searched methods' names, in order",
    "document.run": "the settings of the run (the document.run fields)",
    "document.units": "per gate unit its entry (the unit fields)",
    "document.pass_mark": (
        "the pass mark over every unit and variant (the document.pass_mark fields)"
    ),
    "document.timeouts": "timeouts by kind, summed over the units (definitions.timeouts)",
    "document.failed": "'<unit> | <variant>: <error>' for each variant that failed, sorted",
    "document.model": "the forecast model's name",
    "document.fit_units": "the units the model was fitted on",
    "document.forecast_table_rows": "the number of rows of the model's forecast table",
    "document.inputs": (
        "the verified truth build read: its setting, its folder's name and the SHA-256 of each "
        "table read (gate_run.check_inputs)"
    ),
    "document.matrix": "the matrix file (the document.matrix fields)",
    "document.matrix.file": "the matrix file's name",
    "document.matrix.sha256": "the SHA-256 of the matrix file's bytes",
    "document.rotation_whatif": (
        "on a rotation what-if matrix only: what the run is and which units it scores and left out "
        "of the fit (rotation_whatif.record)"
    ),
    "document.run.seed": "the split seed (--seed)",
    "document.run.days": "the searched days asked per unit and variant (--days)",
    "document.run.tier": "the tier of each method's calibrated scale",
    "document.run.check_days": "the searched days checked for the pass mark (CHECK_DAYS)",
    "document.run.check_scales": "the scales of each checked day (dial_splits.GRID)",
    "document.run.slot_tolerance": "corner_search.SLOT_TOLERANCE",
    "document.run.driver_tolerance": "corner_search.DRIVER_TOLERANCE",
    "document.run.kappa": "the forecast level kappa of the forecast variant's configuration",
    "document.run.variants": "the variants run, in order",
    "document.pass_mark.pairs": "pairs checked, summed over units and variants",
    "document.pass_mark.pairs_agreed_strict": "pairs that agree strictly, summed",
    "document.pass_mark.pairs_agreed": "pairs that agree (strictly or within tolerance), summed",
    "document.pass_mark.agreed": (
        "at least one unit and variant was checked and each check agreed (pass_mark.agreed)"
    ),
    "unit.counting_days": "the number of the unit's counting days (gate_run.plan_unit)",
    "unit.calibration_days": (
        "calibration days of a split of the unit (the same number for every split seed); 0 when "
        "the unit cannot be split"
    ),
    "unit.test_days": "test days of a split of the unit; 0 when the unit cannot be split",
    "unit.status": "'calibratable', or why the unit cannot be split (gate_run.plan_unit)",
    "unit.variants": "per variant its entry (the variant fields); empty when it cannot be split",
    "variant.configuration": "the name of the variant's configuration of the matrix",
    "variant.failed": (
        "only in a variant that failed: the error's type and message; such a variant has no "
        "other field but its configuration"
    ),
    "variant.offering_test_days": "test days of the split that offer under the variant",
    "variant.days_asked": "the searched days asked (--days)",
    "variant.days_searched": (
        "the number of days searched: the first --days offering test days, or all of them when "
        "there are fewer"
    ),
    "variant.offering_calibration_days": (
        "calibration days of the split that offer under the variant"
    ),
    "variant.calibration_evaluations": (
        "the counts of compare_run.measure_variant on the split's calibration days, as compare_run "
        "writes them (among them the one-way re-solves by outcome and, for a controller that "
        "solves programmes, its solver outcomes)"
    ),
    "variant.searched_days": (
        "the searched days' indices among the unit's counting days, in the searched order (W2)"
    ),
    "variant.methods": "per method its entry (the method fields)",
    "variant.pass_mark": "the pass-mark check of the unit and variant (the pass_mark fields)",
}

# Added to the definitions of a run on the sealed route only (`mark_route`, module text).
ROUTE_READINGS = {
    "H1": (
        "the split on the sealed route: the one draw the held-out build made; --seed must be the "
        "matrix's split_seed, the matrix's split rule the build's and each scored unit's "
        "calibration days as drawn the build's, or the run could not run"
    ),
    "H2": (
        f"the days on the sealed route: --days below {DAYS} could not run (O7's 50 fixed random "
        "days per unit; only the search beyond 50 days per unit may be cut)"
    ),
    "H3": (
        "the methods on the sealed route: --methods is accepted, one list for every unit fixed "
        f"before the run, never chosen per unit by looking at results; it must hold {FIXED_PLAN}, "
        "whose calibrated scale is the scale of the closed-form check on held-out test days (H5), "
        "or the run could not run, with the matrix file read but no table read, no model loaded "
        "and nothing in the sealed folder opened"
    ),
    "H4": "the dial on the sealed route: --dial taken as given, fixed before the tag per site type",
    "H5": (
        "the closed-form check on held-out days runs on the first searched days, which are test "
        "days, at one scale: the fixed plan's calibrated tier A scale of the unit and variant, "
        "the value its calibration chose and the search uses, not every scale of "
        "dial_splits.GRID (plan section 6: test days are scored only at the chosen scale; the "
        "owner's decision of 16 November 2026); a disagreement gives exit 1; it is a property "
        "check there, not WP11's pass mark, which is read on development units"
    ),
    "H6": (
        "the names on the sealed route: the check's fields, their definitions and its path-hours "
        f"are named {corner_summary.HELD_OUT_CHECK_KEY} and the summary's headings name it a "
        "closed-form property check on held-out days, at the fixed plan's calibrated scale, so "
        "that no field or heading of the run calls it what it is not (H5: not WP11's pass mark)"
    ),
    "H7": (
        "a unit and variant the sealed route does not check: where the fixed plan's calibrated "
        "scale is 0 (it offers nothing) or no day was searched, its check entry has status "
        "'not checked' and its reason, is never counted as agreed and does not decide the exit "
        "status; the run's check sums the checked ones alone, lists the others under "
        "not_checked, and its agreed is null when none was checked; the exit status is 1 for a "
        "failed variant, for no unit that could be split and for a disagreement, else 0, so exit "
        "0 with agreed null means the split units were calibrated and searched (perhaps on no "
        "day) and nothing was checked, not that anything agreed"
    ),
}
# Reading K3 on the sealed route (H5 to H7).
ROUTE_K3 = (
    f"closed-form property check on held-out days: the first {CHECK_DAYS} searched days of each "
    "unit and variant at the fixed plan's calibrated scale of that unit and variant alone (H5; "
    "a unit and variant not checked as H7 says); " + CHECK_RULE
)
# On the sealed route the definitions of the check's fields are read under their route names
# (`route_name`), and these in these words (H5 to H7).
ROUTE_CHECK_DEFINITIONS = {
    "document.closed_form_check": (
        "the closed-form property check over every unit and variant checked (the "
        "document.closed_form_check fields)"
    ),
    "document.closed_form_check.pairs": "pairs checked, summed over the units and variants checked",
    "document.closed_form_check.agreed": (
        "true when at least one unit and variant was checked and each check agreed, false when "
        "a checked pair disagreed, null when nothing was checked (H7)"
    ),
    "document.closed_form_check.not_checked": (
        "'<unit> | <variant>: <reason>' for each unit and variant whose check entry is 'not "
        "checked', sorted (H7); a variant that failed is not here (document.failed) and nor is a "
        "unit that cannot be split (its unit.status, with no variants)"
    ),
    "document.run.check_days": (
        "the searched days checked by the closed-form property check (CHECK_DAYS)"
    ),
    "document.run.check_scales": (
        "the scale of each checked day, in words: the fixed plan's calibrated scale of the unit "
        "and variant (H5)"
    ),
    "variant.closed_form_check": (
        "the closed-form property check of the unit and variant (the closed_form_check fields), "
        "or why it was not checked (closed_form_check.status, closed_form_check.reason)"
    ),
    "closed_form_check.scales": (
        "the scale of each checked day: the fixed plan's calibrated scale of the unit and "
        "variant (H5)"
    ),
    "closed_form_check.pairs": "(day, scale) pairs checked: one per checked day (H5)",
    "closed_form_check.status": (
        "only in a unit and variant not checked: 'not checked' (H7); such an entry has no other "
        "field but its reason"
    ),
    "closed_form_check.reason": (
        "why the unit and variant was not checked: the fixed plan offers nothing (scale 0 and "
        "why), or no day was searched (H7)"
    ),
}
ROUTE_FIELDS = {
    "document.sealed_route": (
        "on the sealed route only: the route's record (sealed_route.mark: its kind and label, the "
        "tag's commit, the held-out build's check of the scored units' tables, the fit units, "
        "kappa_selection and checks)"
    ),
    "document.inputs": (
        "on the sealed route: the development build's check (gate_run.check_inputs) of the fit "
        "units' tables only; the scored units' tables are checked against the held-out build, "
        "under sealed_route.held_out_inputs"
    ),
}


# --- the per-day work, in this process or in a pool ----------------------------------------------


def _search_task(job):
    prepared, method, beta = job
    return corner_search.search_day(prepared, method, beta)


def _check_task(job):
    prepared, betas = job
    return corner_search.closed_form_check(prepared, betas)


def prepare_jobs(unit: str, count: int, configuration) -> list[tuple]:
    """The jobs of `compare_run._prepare_task` for the first `count` days of `unit`: no car drawn
    (the search is of the whole fleet, module text)."""
    return [(unit, i, configuration, ()) for i in range(count)]


def day_order(unit: str, seed: int, days) -> list[int]:
    """The searched order of `days` (reading K5)."""
    digest = hashlib.sha256(f"{unit}|{seed}|corner-days".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    days = list(days)
    return [days[k] for k in rng.permutation(len(days))]


def calibrate(prepared, calibration, methods, pool: compare_run.Pool):
    """Per method its tier A scale and reason on the calibration days (W1), and the counts of
    `compare_run.measure_variant`."""
    days = [prepared[d] for d in calibration]
    losses, counts = compare_run.measure_variant(
        days,
        [None] * len(days),
        methods,
        lambda pairs: pool.map(compare_run._evaluate_task, pairs),
        lambda jobs: pool.map(compare_run._compare_task, jobs),
    )
    scales = {}
    for method in methods:
        score = dial_splits.score_split(losses[method.name], 0, list(range(len(days))), [], TIER)
        scales[method.name] = (score.scale, score.reason)
    sized = {m.name: losses[m.name].rule.counts() for m in methods if m.kind == "sizing"}
    if sized:  # W5, as compare_run.sizing_counts reads them
        counts["sizing_solver_outcomes"] = sized
    return scales, counts


def measured_losses(methods) -> int:
    """The sets of calibration losses `compare_run.measure_variant` measures for `methods`: one per
    method, but the sizing methods read the re-solve's, measured once (module text, "Cost")."""
    sizing = any(m.kind == "sizing" for m in methods)
    resolve = any(m.kind == "re-solve" for m in methods)
    return sum(m.kind != "sizing" for m in methods) + (sizing and not resolve)


def _path_hours(prepared, days, paths) -> int:
    return int(sum(paths(prepared[d].day.n_service) * prepared[d].day.n_service for d in days))


def _corners(n: int) -> int:
    return 2**n


def count_path_hours(
    prepared, calibration, searched, n_methods: int, check_days: int, *, measured=None, scales=None
) -> dict:
    """The path-hours of one unit and variant before anything is run (module text, "Cost").
    `measured`: the sets of calibration losses (`measured_losses`); None: `n_methods`. `scales`:
    the check's scales per day (1 on the sealed route, reading H5); None: `dial_splits.GRID`'s."""
    offering_calibration = [d for d in calibration if not prepared[d].abstains]
    audited = _path_hours(prepared, offering_calibration, lambda n: AUDITED_ROWS)
    measured = n_methods if measured is None else measured
    scales = len(dial_splits.GRID) if scales is None else scales
    return {
        "calibration": audited * len(dial_splits.GRID) * measured,
        "search_upper_bound": _path_hours(prepared, searched, _corners) * n_methods,
        "pass_mark": _path_hours(prepared, searched[:check_days], _corners) * scales,
    }


def check_scales(scales: dict, held_out: bool) -> tuple:
    """The scales of the closed-form check of one unit and variant: off the sealed route every
    scale of `dial_splits.GRID` (K3); on it the fixed plan's calibrated scale alone, the value
    `calibrate` chose (`scales`, its result), and none where that is 0 (readings H5, H7)."""
    if not held_out:
        return dial_splits.GRID
    scale = scales[FIXED_PLAN][0]
    return (scale,) if scale > 0 else ()


def unchecked_reason(scales: dict, checked) -> str | None:
    """Why the sealed route does not check a unit and variant (reading H7), or None when it
    does: the fixed plan's calibrated scale is 0, or no day was searched."""
    scale, why = scales[FIXED_PLAN]
    if scale <= 0:
        return f"the fixed plan offers nothing (scale 0, {why})"
    if not checked:
        return "no day searched (no test day offers under the variant)"
    return None


def search(prepared, searched, methods, scales, pool) -> dict:
    """Name -> the `CornerDay` of each searched day: searched at the method's scale above 0, "no
    offer" at a scale of 0 (W4)."""
    work = [
        (prepared[d], m, scales[m.name][0])
        for m in methods
        if scales[m.name][0] > 0
        for d in searched
    ]
    found = iter(pool.map(_search_task, work))
    return {
        m.name: (
            [next(found) for _ in searched]
            if scales[m.name][0] > 0
            else [corner_search.no_offer(m.name, scales[m.name][0]) for _ in searched]
        )
        for m in methods
    }


def measure_variant(
    prepared, calibration, searched, methods, pool, hours: dict, counted, held_out: bool = False
) -> tuple:
    """One unit under one variant: (its results entry, name -> `CornerDay`s, scales). `hours`
    (the path-hour counts) gains the search's count once the scales are known (on the sealed
    route, `held_out`, the check's count as run too, reading H7), and `counted()` is called then,
    before the search. The check runs at `check_scales`: on the sealed route at the fixed plan's
    calibrated scale alone (reading H5), and a unit and variant `unchecked_reason` names is not
    checked (H7)."""
    scales, counts = calibrate(prepared, calibration, methods, pool)
    hours["search"] = _path_hours(prepared, searched, _corners) * sum(
        scales[m.name][0] > 0 for m in methods
    )
    checked = searched[:CHECK_DAYS]
    betas = check_scales(scales, held_out)
    unchecked = unchecked_reason(scales, checked) if held_out else None
    if held_out:  # the check's count as run: one scale, or 0 where it is not checked
        hours[corner_summary.CHECK_KEY] = (
            0 if unchecked else _path_hours(prepared, checked, _corners) * len(betas)
        )
    counted()
    by_method = search(prepared, searched, methods, scales, pool)
    if unchecked is None:
        checks = pool.map(_check_task, [(prepared[d], betas) for d in checked])
        check = corner_summary.pass_mark_entry(checked, [prepared[d].date for d in checked], checks)
    else:
        check = corner_summary.unchecked_entry(unchecked)
    entry = {
        "offering_calibration_days": sum(not prepared[d].abstains for d in calibration),
        "calibration_evaluations": counts,
        "searched_days": list(searched),
        "methods": {
            m.name: corner_summary.method_entry(*scales[m.name], by_method[m.name]) for m in methods
        },
        "pass_mark": check,
    }
    return entry, by_method, scales


def definitions(matrix: Matrix, sizing=(), deviating=()) -> dict:
    """The definitions of a run; `sizing`: the names of its sizing methods (W5: the reading, the
    counts and keys are added only with one); `deviating`: the names of the methods that wrote
    deviation counts (`method.deviation_counts` is defined only with one)."""
    found: dict = {
        "readings": dict(READINGS),
        "fields": {**FIELD_DEFINITIONS, **corner_summary.DEFINITIONS},
        "variants": {
            "forecast": "the dial's configuration of the matrix",
            "oracle-departure": "that configuration's oracle partner (field oracle_partner)",
        },
        "calibration": (
            f"tier {TIER}: slot loss at alpha 0.10 and driver loss at alpha 0.05 on the split's "
            f"offering calibration days (n_calibration {matrix.n_calibration}, "
            f"calibration_half_below {matrix.calibration_half_below})"
        ),
        "timeouts": (
            "certificates: days whose certificate timed out (they offer nothing); "
            "re_solves_calibration: one-way re-solves timed out in the calibration; "
            "re_solves_search: re-solves timed out on searched corners; "
            "controller_solves_calibration, controller_solves_search: solves of a controller "
            "that solves programmes timed out, in the calibration and on searched corners (each "
            "hour so planned by its fall-back); the keys with oracle_ in front: the same under "
            "the oracle-departure variant"
        ),
        "label": LABEL,
    }
    if sizing:
        names = ", ".join(sizing)
        found["readings"]["W5"] = SIZING_READING
        found["fields"]["method.resolves"] = corner_summary.RESOLVES_WITH_SIZING
        found["fields"]["variant.calibration_evaluations"] += (
            f"; for a sizing method ({names}) its sizing programmes under sizing_solver_outcomes "
            "(compare_run's definitions.sizing.sizing_solver_outcomes)"
        )
        found["timeouts"] += (
            f"; sizing_programmes_calibration: programmes of a sizing method ({names}) timed out "
            "in the calibration, handled as chance_sizing reading S11 says"
        )
    if deviating:
        found["fields"]["method.deviation_counts"] = (
            f"{corner_summary.DEVIATION_DEFINITION} ({', '.join(deviating)})"
        )
    return found


def deviating_methods(methods, entries: dict) -> list[str]:
    """The names, in the run's order, of the methods whose entry of some unit and variant holds
    deviation counts."""
    written = {
        name
        for entry in entries.values()
        for variant in entry["variants"].values()
        for name, found in variant.get("methods", {}).items()
        if "deviation_counts" in found
    }
    return [m.name for m in methods if m.name in written]


def _count_timeouts(timeouts: Counter, variant: str, prepared, entry: dict) -> None:
    key = "" if variant == "forecast" else "oracle_"
    timeouts[f"{key}certificates"] += sum(p.certificate_outcome == "timeout" for p in prepared)
    calibration = entry["calibration_evaluations"]["re_solves_of_the_one_way_rung"]
    timeouts[f"{key}re_solves_calibration"] += calibration.get("timeout", 0)
    timeouts[f"{key}re_solves_search"] += sum(
        m.get("resolves", {}).get("timeout", 0) for m in entry["methods"].values()
    )
    solved = entry["calibration_evaluations"].get("controller_solver_outcomes", {})
    timeouts[f"{key}controller_solves_calibration"] += sum(
        c["solves"].get("timeout", 0) for c in solved.values() if c
    )
    timeouts[f"{key}controller_solves_search"] += sum(
        (m.get("solver_counts") or {}).get("solves", {}).get("timeout", 0)
        for m in entry["methods"].values()
    )
    sized = entry["calibration_evaluations"].get("sizing_solver_outcomes")
    if sized:  # W5: the key only with a sizing method
        timeouts[f"{key}sizing_programmes_calibration"] += sum(
            c["solves"]["timeout"] for c in sized.values()
        )


def _route_options(held_out: bool) -> dict:
    """The keyword `held_out` of `run` and `measure_variant`, given on the sealed route only, so
    that off it each is called with the arguments it had before the route (readings H5, H7)."""
    return {"held_out": True} if held_out else {}


def run(matrix: Matrix, units: dict, factory, configurations: dict, methods, **options):
    """Everything but the input checks and the writing of files: (results document, rows, timing).
    Options: `days`, `seed`, `workers`, `progress`, a function called with the timing record
    each time a path-hour count is added (None: not called), and `held_out`, a run on the sealed
    route (the check at one scale, readings H5 and H7; default False)."""
    days_asked, seed, workers = options["days"], options["seed"], options["workers"]
    progress = options.get("progress") or (lambda timing: None)
    held_out = options.get("held_out", False)
    started = time.perf_counter()
    pool = compare_run.Pool(workers, units, matrix, factory)
    entries, rows, timeouts, failed = {}, [], Counter(), {}
    timing = {"units": {}, "path_hours": {}}
    try:
        for name in matrix.gate_units:
            unit_days = units[name]
            plan = gate_run.plan_unit(name, unit_days, matrix)
            entry = {
                "counting_days": len(plan.keys),
                "calibration_days": plan.calibration_days,
                "test_days": plan.test_days,
                "status": plan.status,
                "variants": {},
            }
            entries[name] = entry
            if plan.status != "calibratable":
                continue
            begun = time.perf_counter()
            ((_, calibration, test),) = dial_choice.split_indices(name, unit_days, matrix, [seed])
            order = day_order(name, seed, test)
            timing["path_hours"][name] = {}
            for variant, configuration in configurations.items():
                try:
                    prepared = pool.map(
                        compare_run._prepare_task,
                        prepare_jobs(name, len(plan.keys), configuration),
                    )
                    offering = [d for d in order if not prepared[d].abstains]
                    searched = offering[:days_asked]
                    hours = count_path_hours(
                        prepared,
                        calibration,
                        searched,
                        len(methods),
                        CHECK_DAYS,
                        measured=measured_losses(methods),
                        scales=1 if held_out else None,
                    )
                    timing["path_hours"][name][variant] = hours
                    progress(timing)
                    found, by_method, scales = measure_variant(
                        prepared,
                        calibration,
                        searched,
                        methods,
                        pool,
                        hours,
                        lambda: progress(timing),
                        **_route_options(held_out),
                    )
                except Exception as problem:  # recorded, the other variants go on
                    failed[(name, variant)] = f"{type(problem).__name__}: {problem}"
                    entry["variants"][variant] = {
                        "configuration": configuration.name,
                        "failed": failed[(name, variant)],
                    }
                    continue
                _count_timeouts(timeouts, variant, prepared, found)
                entry["variants"][variant] = {
                    "configuration": configuration.name,
                    "offering_test_days": len(offering),
                    "days_asked": days_asked,
                    "days_searched": len(searched),
                    **found,
                }
                for method in methods:
                    rows += corner_summary.day_rows(
                        name,
                        variant,
                        method.name,
                        scales[method.name][0],
                        searched,
                        prepared,
                        by_method[method.name],
                    )
            timing["units"][name] = time.perf_counter() - begun
    finally:
        pool.close()
    entered = [
        (name, variant, v["pass_mark"])
        for name, e in entries.items()
        for variant, v in e["variants"].items()
        if "pass_mark" in v
    ]
    checks = [c for _, _, c in entered if corner_summary.is_checked(c)]
    total = {
        "pairs": sum(c["pairs"] for c in checks),
        "pairs_agreed_strict": sum(c["pairs_agreed_strict"] for c in checks),
        "pairs_agreed": sum(c["pairs_agreed"] for c in checks),
        "agreed": bool(checks) and all(c["agreed"] for c in checks),
    }
    if held_out:  # reading H7: never agreed when nothing was checked, the others listed
        total["agreed"] = all(c["agreed"] for c in checks) if checks else None
        total["not_checked"] = sorted(
            f"{name} | {variant}: {c['reason']}"
            for name, variant, c in entered
            if not corner_summary.is_checked(c)
        )
    document = {
        "label": LABEL,
        "definitions": definitions(
            matrix,
            [m.name for m in methods if m.kind == "sizing"],
            deviating_methods(methods, entries),
        ),
        "configurations": {variant: c.name for variant, c in configurations.items()},
        "dial": configurations["forecast"].driver_risk,
        "methods": [m.name for m in methods],
        "run": {
            "seed": seed,
            "days": days_asked,
            "tier": TIER,
            "check_days": CHECK_DAYS,
            "check_scales": ROUTE_CHECK_SCALES if held_out else list(dial_splits.GRID),
            "slot_tolerance": corner_search.SLOT_TOLERANCE,
            "driver_tolerance": corner_search.DRIVER_TOLERANCE,
            "kappa": configurations["forecast"].kappa,
            "variants": list(configurations),
        },
        "units": entries,
        "pass_mark": total,
        "timeouts": dict(sorted(timeouts.items())),
        "failed": sorted(f"{a} | {b}: {why}" for (a, b), why in failed.items()),
    }
    timing["total_seconds"] = time.perf_counter() - started
    return document, rows, timing


def route_refusal(
    route: sealed_route.Route, matrix: Matrix, seed: int, days: int, methods=(FIXED_PLAN,)
) -> str | None:
    """Why the split, sample and method arguments could not run on the sealed route (readings
    H1 to H3; `methods`: the searched methods' names), or None; always None off the route."""
    if not route.held_out:
        return None
    if seed != matrix.split_seed:
        return (
            f"--seed {seed} is not the matrix's split_seed {matrix.split_seed}; on the sealed "
            "route a held-out unit has one split, the one its held-out build drew, and --seed "
            "names it; nothing read"
        )
    if days < DAYS:
        return (
            f"--days {days} is below the {DAYS} fixed random days per unit of plan section 11 "
            "(O7); on the sealed route fewer days are a trial; nothing read"
        )
    if FIXED_PLAN not in methods:
        return (
            f"--methods without {FIXED_PLAN}: on the sealed route the closed-form check on "
            f"held-out test days runs at the fixed plan's calibrated scale, which only a run that "
            f"searches {FIXED_PLAN} calibrates; nothing read"
        )
    return None


def mark_route(document: dict, route: sealed_route.Route, held_out_inputs, fit_units) -> None:
    """Records the sealed route in a results document (module text); nothing off the route."""
    if not route.held_out:
        return
    sealed_route.mark(document, route, held_out_inputs, fit_units)
    document["label"] = HELD_OUT_LABEL
    held = corner_summary.HELD_OUT_CHECK_KEY
    document[held] = document.pop(corner_summary.CHECK_KEY)
    for entry in document["units"].values():
        for variant in entry["variants"].values():
            if corner_summary.CHECK_KEY in variant:
                variant[held] = variant.pop(corner_summary.CHECK_KEY)
    found = document["definitions"]
    found["label"] = HELD_OUT_LABEL
    found["readings"]["K3"] = ROUTE_K3
    found["readings"].update(ROUTE_READINGS)
    found["fields"] = {route_name(k): text for k, text in found["fields"].items()}
    found["fields"].update(ROUTE_CHECK_DEFINITIONS)
    found["fields"].update(ROUTE_FIELDS)


def route_name(name: str) -> str:
    """A field's or path-hour count's name on the sealed route (reading H6): each part named
    `corner_summary.CHECK_KEY` renamed `corner_summary.HELD_OUT_CHECK_KEY`."""
    parts = name.split(".")
    return ".".join(
        corner_summary.HELD_OUT_CHECK_KEY if part == corner_summary.CHECK_KEY else part
        for part in parts
    )


def route_timing(timing: dict, route: sealed_route.Route) -> dict:
    """The timing record as written: on the sealed route its path-hour counts under their route
    names (reading H6); off it `timing` itself."""
    if not route.held_out:
        return timing
    hours = {
        unit: {
            variant: {route_name(k): n for k, n in counts.items()} for variant, counts in by.items()
        }
        for unit, by in timing["path_hours"].items()
    }
    return {**timing, "path_hours": hours}


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _csv(rows) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(corner_summary.ROW_FIELDS)
    writer.writerows(rows)
    return out.getvalue()


def main(argv: list[str] | None = None, root: Path = TOP) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dial", type=float, required=True)
    parser.add_argument("--matrix", type=Path, default=dial_choice.MATRIX)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--held-out-tables", type=Path, default=None)
    parser.add_argument("--held-out-truth-run", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        gate_run.argument_paths(args)
        for flag in ("workers", "days", "seed"):
            if getattr(args, flag) < 1:
                raise MatrixError(f"--{flag} must be at least 1")
        methods = compare_methods.select(args.methods)
        gate_run.check_files([args.matrix])
        matrix = load(args.matrix)
        configurations = compare_run.dial_configurations(matrix, args.dial)
    except (MatrixError, MethodError, CouldNotRun) as problem:
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
    refusal = route_refusal(route, matrix, args.seed, args.days, [m.name for m in methods])
    if refusal is not None:
        print(f"COULD NOT RUN: {refusal}")
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
        loaded = {
            name: (
                parse_unit_days(table_bytes, name, **readings)
                if route.held_out
                else load_unit_days(args.tables, name, **readings)
            )
            for name in needed
        }
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
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    folder = Path(run_dir) if run_dir else None

    def progress(timing):
        if folder is not None:
            (folder / TIMING_NAME).write_text(_json(route_timing(timing, route)))

    document, rows, timing = run(
        matrix,
        units,
        factory,
        configurations,
        methods,
        days=args.days,
        seed=args.seed,
        workers=args.workers,
        progress=progress,
        **_route_options(route.held_out),
    )
    document["model"] = model.name
    document["fit_units"] = list(fit_units)
    document["forecast_table_rows"] = len(table)
    rotation_whatif.mark(document, matrix, fit_units)
    document["inputs"] = inputs
    mark_route(document, route, held_inputs, fit_units)
    sha256 = hashlib.sha256(args.matrix.read_bytes()).hexdigest()
    document["matrix"] = {"file": args.matrix.name, "sha256": sha256}
    lines = [
        *sealed_route.summary_head(document),
        *corner_summary.summary_lines(document, route.held_out),
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if folder is not None:
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / ROWS_NAME).write_text(_csv(rows))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / TIMING_NAME).write_text(_json(route_timing(timing, route)))
    return exit_status(document, route.held_out)


def exit_status(document: dict, held_out: bool) -> int:
    """The exit status of a run that wrote its results (module text, "Exit status"); on the
    sealed route as reading H7 says: a unit and variant not checked does not decide it."""
    agreed = document[corner_summary.check_key(held_out)]["agreed"]
    if not held_out:
        return 1 if document["failed"] or not agreed else 0
    split = any(e["status"] == "calibratable" for e in document["units"].values())
    return 1 if document["failed"] or not split or agreed is False else 0


if __name__ == "__main__":
    sys.exit(main())
