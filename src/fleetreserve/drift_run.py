"""The time-ordered run of the drift guard on units' counting days (plan section 6, T10): a
measurement, reported only.

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.drift_run --model <name> --dial <epsilon> [--matrix <file>]
        [--units <unit> ...] [--etas <eta> ...] [--calibration-days N] [--workers N]
        [--tables <folder>] [--truth-run <folder>]
        [--held-out-tables <folder>] [--held-out-truth-run <folder>]

Plan section 6, T10: "η set in WP9. ... Chronological run, reported only: first 50 days
calibrate, then T10, at OSL_T through COVID and a.s.r. across September 2022 and March 2023".
On training and development units this command measures that run over a grid of step sizes eta
and forms no choice: `drift_settings` applies a named rule to the results of a run on training
units. On held-out units it runs, through the sealed-data route below, at the one step size per
tier that the settings file written that way holds. The arithmetic of one unit is `drift_path`
(its readings E1 to E6: the two start readings, the calibration days, the days that do not
offer, the cap, the raw loss, the two guards of tier A).

Units (`--units`, default every unit of `truth_build.TRAIN_UNITS`), named once each: either every
one a training or a development unit (`truth_build.UNITS`), or every one a held-out unit
(`sealed_route.HELD_OUT_UNITS`), whose tables are read only through the sealed-data route below
(reading E8). Any other name, and held-out units named with other units, is refused before a
table is opened (exit 2, "COULD NOT RUN").

Model, per unit u, as `size_whatif` builds it: `fleetreserve.models.<name>` made from the matrix
with u as its only gate unit (`rotation_whatif` set when u is a training unit, so
`models.lgbm_tail_rotation` is fitted on the other training units), and `learn_run.check_model`
with the gate units (u,). Refused before a table is opened: `fit_units` not all training units,
exit 1 ("REFUSED"); `fit_units` holding u, exit 2; a model that reads the unit's other days
(`forecast_model.OUTCOME_READERS`, the empirical forecaster), exit 2 (reading E7). The dial:
the configuration of the matrix's dial family (`dial_choice.find_family`) whose `driver_risk` is
`--dial` (`compare_run.dial_configurations`; none: exit 2). Off the route the truth tables of the
units and of the models' fit units are checked as `gate_run.check_inputs` checks them (a verified
truth build at the commit checked out now, with the tables' hashes).

Per unit: the model is fitted and the forecast table made by `dial_choice.make_factory` at the
levels up to the configuration's kappa; every counting day is prepared by `gate_day.prepare_day`
(certificate with the box time limit; a timeout abstains the day); then `drift_path.run_unit`
with `--calibration-days` (default 50) and `--etas` (default `ETAS`), every (day, scale) it asks
for evaluated once by `gate_day.evaluate` in a pool of `--workers` processes, the one-way rung's
day-level slot and driver losses kept (`dial_choice.RUNG`, the rung the gate reads; every re-solve
has its time limit, and one that times out leaves the rung on its predecessor's plan, counted
under `evaluations`). A unit whose steps raise an error is recorded under `failed_units` and the
other units go on.

Held-out units, through the sealed-data route (`sealed_route.units_route`: the checks of
`learn_run`'s route, in its order, with texts naming the units named; it starts only at the tag
`frozen-plan` with a clean tree, and its checks stop an honest operator's ordinary mistakes and
are not a defence against deliberate evasion: `sealed_route`, module text, first paragraph).
When every unit of `--units` is a held-out unit, before any table is read: each path argument is
put through `gate_run.argument_path` (a path written as a URL could not run; a leading `~` is
expanded); every folder named lies outside the held-out places of the top folder
(`sealed_route.check_folders`); the tag gate passes (the held-out loader's checks 2 to 8: at the
tag `frozen-plan`, clean, the running code the tag's); and `--held-out-truth-run` is given
(`--held-out-tables` defaults to the held-out build's folder of the main setting,
`sealed_route.held_out_tables`). Away from the tag a held-out unit could not run (exit 2) before
any table is opened. Then, still before any table is read, each failure could not run (exit 2):
`--etas` or `--calibration-days` given (reading E19); the settings file `SETTINGS`
(`learn/drift-eta.json`, the file `drift_settings` writes) missing from the tag's commit, read
from git's objects as that commit holds it (`sealed_route.committed_text`), or of another rule,
start or grouping than the plan's, or written from units that are not training units (reading
E17); the matrix file in a refused place (`gate_run.check_files`), outside the top folder, not a
file of the tag's commit, or differing from it (reading E24). The model is built and checked
per unit as off the route, and the gate is passed again after it (its module is then among the
running code, `sealed_route.gate_again`). Then `sealed_route.read_two_builds`: the models' fit
units' tables from the development build (`--tables`, `--truth-run`) and the units' tables from
the held-out build, each build verified at the checked-out commit, the two of one setting, each
table read once, so the bytes hashed are the bytes parsed (`days_from_truth.parse_unit_days`, the
matrix's readings). After the tables are read, each failure could not run (exit 2): units of two
site types (reading E20); a site type the settings file holds no step size for; a step size there
that is not a finite number above 0 (a NaN, 0 or a value below 0) and not "offers nothing". Each
unit then runs as off the route, with the settings file's one step size per tier for its site
type (`drift_settings.step_size`) and the settings file's start alone (reading E18), its first
`drift_path.N_CALIBRATION` (50) counting days calibrating; a tier the file gives "offers
nothing" for that site type runs no path and offers nothing (reading E29). With other units the
two held-out options could not run, and without them nothing above is checked: the run, its
outputs and its summary are as before.

What the command fits, chooses, samples or calibrates on the route, and from which days:
  L1. The model: fitted once on its `fit_units`, all training units (`learn_run.check_model`:
      another unit refused, exit 1; the scored unit among them, exit 2), on those units' days of
      every date (reading E7). It forecasts each car of a scored day from inputs known at τ
      (`learn_features`; the user's earlier sessions plugged out by τ are known then:
      `sealed_route` reading S1). A model that reads the unit's other days could not run
      (reading E7) and a look-ahead model is refused (exit 1), on the route as off it.
  L2. The step size: one per tier, the settings file's as the tag's commit holds it, written by
      `drift_settings` from a recorded run on training units by the plan's rule (reading E17),
      or "offers nothing" for the tier, which then runs no path (reading E29). `--etas` is
      refused, so no grid of step sizes is measured on a scored unit and none is
      chosen there.
  L3. The calibration: the unit's first 50 counting days in date order (then site,
      `days_from_truth`), whether or not they offer (`drift_path` reading E2). Their on-grid
      scale per loss (`drift_path.calibrate`) is the start of the path "calibrated", which the
      route does not run (reading E18): under the start zero it is reported only. The guarded
      days are the unit's later counting days; each is offered at the guards' states before it,
      which only earlier guarded days' losses have moved (T10). The held-out build's own
      calibration days, a seeded draw over all the unit's counting days for the split protocol
      of plan section 6, are not this command's calibration days (reading E22).
  L4. The dial and kappa: `--dial` names one configuration of the matrix's dial family
      (`compare_run.dial_configurations`), at whose kappa the model predicts; the matrix is a
      file the tag's commit holds (reading E24) and the dial is given before the run (the run
      list), so nothing here picks a dial or a kappa from a result.
  L5. Per day: the certificate and the re-solves are `gate_day`'s, with their time limits;
      nothing passes from one day to another but the guards' states, and nothing from one unit
      to another.
  L6. Read: the matrix file, the texts of the settings file and the matrix file at the tag's
      commit (from git's objects), and the two builds' tables and run folders, each build
      verified at the checked-out commit. No other run folder is read: the training run the
      settings file was written from is named in that file and recorded, not opened.

Readings of this command, each with its alternative in brackets:

  E7. A forecast that reads the unit's other days (the empirical forecaster) is refused: its
      forecast of a day reads the unit's later days, which a run in time order must not see.
      [Alternative: allow it, labelled.] A model fitted on other units (the rotation model on a
      training unit, the training units' fit on a development unit) is fitted once on those
      units' days of every date, later dates included, as the gate fits it. [Alternative: refit
      before each guarded day on the fit units' days before it.]
  E8. Held-out units are read only through the sealed-data route above, when every unit named is
      a held-out unit; a mix with other units could not run, and away from the tag a held-out
      unit could not run (exit 2) before any table is opened. [Alternative: refuse held-out
      units outright, as before the route; or accept a mix, each unit's tables from its own
      build.]
  E9. The step-size grid `ETAS` (set here, a measurement grid, not a choice): 0.01, 0.02, 0.05,
      0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50. With alpha 0.10 a loss-free day raises a slot
      guard by 0.1 eta and a day lost in full lowers it by 0.9 eta. A step above 1 is valid: the
      scale offered is capped at 1 (`drift_path` reading E4), the state is not, and the bound
      alpha + (1 - alpha)/t from beta_1 = 0 holds at every step size, so the grid reaches 50.
      [Alternative: any other grid, by `--etas`.]
  E16. The units of the plan's sentence are OSL_T and the two a.s.r. units
      (`asr-before-march-2023`, `asr-from-march-2023`): September 2022 lies in the first, and
      March 2023 is where the second begins, so "across ... March 2023" is read as both units
      run, each from its own first counting days; no run passes from one unit into the other.
      The route takes every unit `sealed_route.check_units` takes as held-out (TRO_R and the
      Korean portfolios too); which units are run is the run list's. [Alternative: one
      time-ordered run over the a.s.r. site's days of both regimes, a unit the truth builds do
      not make; or the route limited to the units the sentence names.]
  E17. The step size on the route, per tier: the value for the units' site type
      (`drift_settings.step_size`) in the settings file as the tag's commit holds it, a file
      `drift_settings` wrote by the plan's rule (`PLAN_RULE`: rule
      `largest-below-first-failure`, its reading E25, start `zero`, grouping `site-type`; plan
      section 6, T10: "per tier and per site type ..., from the candidate step sizes of the
      chronological run on the training units from β_1=0, taken in increasing order: the
      largest one below the first that fails, or the largest one tried when none fails, a step
      size failing when its mean losses do not end within their levels on every unit of the
      type (slot loss; in Tier A also driver loss)", and "Where no candidate step size is valid
      (the smallest already fails), the drift guard's run offers nothing for that tier and site
      type") from training units only; any other could not run, the rule of 16 November 2026
      (`largest-offer-within-alpha`) included. [Alternative: any rule, start and grouping
      `drift_settings` writes, its start run; or a step size given on the command line.]
  E18. The start on the route is the settings file's (zero: β_1 = 0), alone; the start
      "calibrated" is not run, its step size not having been chosen for it. [Alternative: both
      starts at the settings' step size, as both are run off the route.]
  E19. On the route the first `drift_path.N_CALIBRATION` counting days (the plan's 50)
      calibrate, and `--calibration-days` and `--etas` are refused, even at the plan's values:
      the run list states neither. [Alternative: both taken as given, fixed in the run list, as
      `o3_run` takes its sample size (its reading R7).]
  E20. A run on the route names units of one site type (`learn_features.SITE_TYPES` of their
      days' sites; a unit with no counting day has none and is not counted): the dial is fixed
      per site type (plan section 6) and the command takes one `--dial`, so units of two types
      could not run, after their tables are read. [Alternative: accept them, one dial for both
      types.]
  E21. The settings file's model and dial (those of the training run it was written from) are
      recorded beside the run's and not compared with them: the run on training units fits the
      rotation model (`lgbm_tail_rotation`), where a run on held-out units fits `lgbm_tail` on
      every training unit, and plan T10 sets the step size per tier and site type only.
      [Alternative: refuse a dial other than the settings file's.]
  E22. The held-out build's drawn calibration days and its split rule are not used
      (`sealed_route.check_split_rule` and `check_split_days` are not called): T10's run
      calibrates on its first 50 days in time order, not on section 6's seeded draw. The build
      set the window of a Norway or Korea unit on those drawn days (the plugged-in counts of a
      seeded draw over all the unit's counting days, plan section 2), so some guarded days are
      among the days the window was set on: a look at plugged-in counts, not at a loss, an offer
      or a departure, made by the build before this command, in the tables every command reads.
      An a.s.r. unit's window is the workplace window, fixed on training units. [Alternative: a
      window searched on the first 50 days alone, which would need a build the plan does not
      state.]
  E23. On the route the results' `inputs` is the development build's check (the fit units'
      tables), and the held-out build's check is under `sealed_route`, as in `learn_run`.
      [Alternative: `inputs` the held-out build's check, as in `o3_run` (its reading R6).]
  E24. On the route the matrix (whose dial family gives the configuration and its kappa) is a
      file the tag's commit holds, inside the top folder, the working copy's file equal to it
      (`sealed_route.committed_text`); its git object is recorded. [Alternative: any matrix file
      outside the refused places, its SHA-256 recorded, as `learn_run` takes one.]
  E29. A tier whose value in the settings file is "offers nothing" for the units' site type
      (`drift_settings.OFFERS_NOTHING`, the plan's "offers nothing for that tier and site
      type"): the tier runs no path, so it offers 0 kWh and loses nothing on every guarded day;
      the step sizes record it as "offers nothing", the summary's step-size lines name it so and
      each unit's lines say that tier offers nothing; the run is not refused and the other tier
      runs at its step size. [Alternative: run the tier's path with every guarded day
      abstaining, which writes rows of zeros under a step size the file does not hold.]

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set, the summary also on standard output:
- `drift-run.json`: every number (`results`), per unit its role, fit units, the rotation
  what-if record of a training unit (`rotation_whatif.record`), the calibration, and per tier,
  start reading and step size the summary of `drift_path.Path.summary`; the day table's hash;
- `drift-run-days.csv`: one row per unit, tier, start, step size and guarded counting day
  (`DAY_COLUMNS`): the states before the day, the scale offered, the offer, the losses, and the
  running means and T10's bounds after the day;
- `drift-run-summary.txt`: the summary text (`summary_lines`);
- `drift-run-timing.json`: wall-clock seconds, kept apart from the other files.
On the route `drift-run.json` also holds `sealed_route` (`gate_summary.ROUTE_KEY`): the kind, the
label `ROUTE_LABEL` (a held-out result, reported only: no pass mark, nothing chosen), the tag's
commit, the held-out build's check, the fit units, `matrix` (the matrix file's path in the
top folder and its git object at the tag's commit), `settings` (the settings file's name, git
object id and SHA-256, its rule, start, grouping, units, model, dial, step sizes and the run it
was written from), `step_sizes` (per unit its site type and per tier the step size, or "offers
nothing", reading E29) and `checks`
(`sealed_route.CHECKS`: what the route's checks stop and what they do not); `run.starts` is the
settings file's start alone and `run.etas` the step sizes used. The summary's first three lines
are that label with the tag's commit and the two builds, the `checks` text, and the step sizes
with their file: its rule, start, grouping and units, checked as reading E17 says, then the run
it was written from (folder, commit, model, dial) as the file records it, which nothing here
checks (L6); its fourth line is `LABEL` (off the route the first line), its fifth, the run's,
gives each tier's step size by its tier, and its last line (`ROUTE_LAST`) says where the step
sizes came from. Off the route the outputs are as before the route.

Exit status: 0 every unit ran (a unit with no more counting days than calibrating days among
them: it runs no guarded day, and its status says too few counting days); 1 the model was refused
(fit units that are not all training units, a look-ahead model, or a declared outcome reader not
in `forecast_model.OUTCOME_READERS`) or failed, a day could not be built from the tables, or a
unit failed; 2 could not run (a bad argument, matrix or model name, a dial not in the matrix's
family, a unit that is not a training, development or held-out unit, a model fitted on a scored
unit or one that reads the unit's other days, a table missing, no verified truth build at this
commit, a table differing from it; for held-out units, a refusal of the route above, `--etas` or
`--calibration-days` given, no settings file at the tag's commit or one not of the plan's rule, a
matrix file that is not the tag's, builds of two settings, units of two site types, a site type
the settings file holds no step size for, a step size there that is not a finite number above 0
and not "offers nothing"). A tier that offers nothing (reading E29) is not a refusal.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from fleetreserve import (
    compare_run,
    dial_choice,
    drift_path,
    drift_settings,
    gate_run,
    gate_summary,
    learn_run,
    rotation_whatif,
    sealed_route,
)
from fleetreserve.days_from_truth import load_unit_days, parse_unit_days
from fleetreserve.gate_config import TIER_LABELS, MatrixError, load
from fleetreserve.learn_features import SITE_TYPES
from fleetreserve.sessions import TOP
from fleetreserve.t7_summary import TYPE_NAMES
from fleetreserve.truth_build import DEVELOPMENT_UNITS, OUT, TRAIN_UNITS
from fleetreserve.truth_verify import CouldNotRun

MATRIX = dial_choice.MATRIX
RESULTS_NAME = "drift-run.json"
DAYS_NAME = "drift-run-days.csv"
SUMMARY_NAME = "drift-run-summary.txt"
TIMING_NAME = "drift-run-timing.json"
RUNG = dial_choice.RUNG
ETAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 50.0)  # reading E9
ROLES = {
    **dict.fromkeys(TRAIN_UNITS, "training"),
    **dict.fromkeys(DEVELOPMENT_UNITS, "development"),
}
LABEL = (
    "TIME-ORDERED RUN OF THE DRIFT GUARD, REPORTED ONLY (plan section 6, T10): no step size is "
    "chosen here"
)
# `drift_settings` imports this module, so its names are read here only inside functions; the
# two values below are its `ETA` from the top folder and its names of the plan's rule (reading
# E17), which a test checks against it.
SETTINGS = Path("learn", "drift-eta.json")
PLAN_RULE = ("largest-below-first-failure", "zero", "site-type")  # (rule, start, grouping)
# The label of a run on the sealed route (module text: outputs; the leak list).
ROUTE_LABEL = (
    f"HELD-OUT RESULT, through the sealed-data gate {sealed_route.AT_THE_TAG}, with a clean tree: "
    "the time-ordered run of the drift guard on held-out units (plan section 6, T10), reported "
    "only, a measurement "
    "with no pass mark; nothing is chosen here: each tier's step size is the settings file's, "
    "written from a run on training units"
)
ROUTE_OPTIONS = (
    "{options} {verb} for runs on training and development units; on the sealed route each "
    "tier's step size is that of the settings file " + SETTINGS.as_posix() + " as the tag's "
    "commit holds it and the first {n} counting days calibrate (plan section 6, T10; readings E17 "
    "and E19); nothing read"
)
ROUTE_LAST = (
    "No step size is chosen here: each tier's is the settings file's, written by drift_settings "
    "from a run on training units; a held-out result, reported only."
)
DAY_COLUMNS = (
    "unit",
    "tier",
    "start",
    "eta",
    "day",
    "date",
    "site",
    "offering",
    "step",
    "state_slot",
    "state_driver",
    "scale",
    "offer_kwh",
    "slot_loss",
    "driver_loss",
    "mean_slot_loss",
    "mean_driver_loss",
    "bound_slot",
    "bound_driver",
)


class NotAllowed(Exception):
    """A unit this command does not run on (module text); the command could not run."""


class OutcomeReader(learn_run.FitOnScored):
    """A model that reads the unit's other days, later ones included (reading E7)."""


def check_units(units, held_out: bool = False) -> tuple[str, ...]:
    """`units` when each is a training or development unit, or (`held_out`, the route's finding
    that every unit is a held-out unit) a held-out unit, named once; raises `NotAllowed`."""
    units = tuple(units)
    outside = [] if held_out else [u for u in units if u not in ROLES]
    if outside:
        raise NotAllowed(
            f"{outside} are not training or development units ({', '.join(ROLES)}); held-out "
            f"units are read only through the sealed-data route {sealed_route.AT_THE_TAG}, when "
            "every unit named is a held-out unit (reading E8)"
        )
    if not units or len(set(units)) != len(units):
        raise NotAllowed(f"the units must be named once each, got {list(units)}")
    return units


def role(unit: str) -> str:
    """A unit's role: training, development or (on the route) held-out."""
    return ROLES.get(unit, sealed_route.HELD_OUT)


def take_route(root: Path, args) -> sealed_route.Route:
    """The sealed route when every unit named is a held-out unit, its path arguments first put
    through `gate_run.argument_path` (module text); else the route off, after refusing the
    held-out options if given. Raises `sealed_route.RouteRefused`, or `CouldNotRun` for a path
    written as a URL."""
    if all(unit in sealed_route.HELD_OUT_UNITS for unit in args.units):
        gate_run.argument_paths(args)
    return sealed_route.units_route(
        root, args.units, args.tables, args.truth_run, args.held_out_tables, args.held_out_truth_run
    )


def counts(args, held_out: bool) -> tuple[int, tuple[float, ...] | None]:
    """(the number of calibration days, the step sizes): off the route `--calibration-days` and
    `--etas` or their defaults; on the route the plan's number and no grid, both options
    refused (reading E19). Raises `NotAllowed` or `MatrixError`."""
    if held_out:
        given = [
            option
            for option, value in (
                ("--etas", args.etas),
                ("--calibration-days", args.calibration_days),
            )
            if value is not None
        ]
        if given:
            raise NotAllowed(
                ROUTE_OPTIONS.format(
                    options=" and ".join(given),
                    verb="is" if len(given) == 1 else "are",
                    n=drift_path.N_CALIBRATION,
                )
            )
        return drift_path.N_CALIBRATION, None
    n_calibration = (
        drift_path.N_CALIBRATION if args.calibration_days is None else args.calibration_days
    )
    if n_calibration < 1:
        raise MatrixError("--calibration-days must be at least 1")
    return n_calibration, check_etas([eta_key(e) for e in ETAS] if args.etas is None else args.etas)


def check_etas(etas) -> tuple[float, ...]:
    """The step sizes, each finite and above 0, distinct, in increasing order."""
    values = tuple(float(e) for e in etas)
    if not values or any(not (math.isfinite(e) and e > 0) for e in values):
        raise MatrixError(f"--etas must be finite numbers above 0, got {list(etas)}")
    if len(set(values)) != len(values) or list(values) != sorted(values):
        raise MatrixError(f"--etas must be distinct and in increasing order, got {list(etas)}")
    return values


def unit_matrix(matrix, unit: str):
    """The matrix with `unit` as its only gate unit; a rotation what-if for a training unit."""
    return replace(matrix, gate_units=(unit,), rotation_whatif=unit in TRAIN_UNITS)


def check_models(name: str, matrix, units) -> dict:
    """Per unit the model, built and checked (module text); raises `MatrixError`,
    `learn_run.FitOnScored` (could not run) or `learn_run.Refused`."""
    models = {}
    for unit in units:
        model = learn_run.load_model(name, unit_matrix(matrix, unit))
        fit_units = tuple(model.fit_units)
        if learn_run.check_model(name, model, fit_units, (unit,)):
            raise OutcomeReader(
                f"model {name} reads the other days of the unit {unit}, later days included; a "
                "run in time order must not see them (reading E7)"
            )
        models[unit] = (model, fit_units)
    return models


# --- the settings file and the step sizes on the route -------------------------------------------


def settings_record(text: str, blob: str) -> dict:
    """The settings file's record from its text (`text`, git object `blob`), when it is a file
    of the plan's rule written from training units (reading E17); raises `CouldNotRun`."""
    name = SETTINGS.as_posix()
    try:
        document = json.loads(text)
        if not isinstance(document, dict):
            raise TypeError(f"the file holds a {type(document).__name__}, not an object")
        grouping = document.get("grouping", drift_settings.ALL_UNITS)
        found = (document["rule"], document["start"], grouping)
        units = list(document["units"])
        written = dict(document["results"])
        absent = [key for key in ("run_folder", "commit") if key not in written]
        if absent:
            raise KeyError(f"results {absent}")
        eta = document["eta"]
    except (ValueError, KeyError, TypeError) as problem:
        raise CouldNotRun(
            f"{name} is not a settings file of drift_settings ({type(problem).__name__}: "
            f"{problem}); nothing read"
        ) from problem
    if found != PLAN_RULE:
        raise CouldNotRun(
            f"{name} holds step sizes of the rule {found[0]}, start {found[1]}, grouping "
            f"{found[2]}; plan section 6 (T10) sets them by the rule {PLAN_RULE[0]}, start "
            f"{PLAN_RULE[1]} (beta_1 = 0), grouping {PLAN_RULE[2]} (reading E17); nothing read"
        )
    outside = [u for u in units if u not in TRAIN_UNITS]
    if not units or outside:
        raise CouldNotRun(
            f"{name} was written from the units {units}; step sizes are chosen on training units "
            f"only ({', '.join(TRAIN_UNITS)}); nothing read"
        )
    return {
        "file": name,
        "git_object": blob,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "rule": found[0],
        "start": found[1],
        "grouping": found[2],
        "units": units,
        "model": document.get("model"),
        "dial": document.get("dial"),
        "eta": eta,
        "results": written,
    }


def route_settings(root: Path, route: sealed_route.Route) -> dict:
    """The record of the settings file as the tag's commit holds it (module text); raises
    `CouldNotRun`."""
    text, blob = sealed_route.committed_text(root, route.tag_commit, SETTINGS)
    return settings_record(text, blob)


def route_matrix(root: Path, route: sealed_route.Route, path: Path) -> dict:
    """The matrix file's record on the route, when it is a file the tag's commit holds and the
    working copy's file equals it (reading E24); raises `CouldNotRun`."""
    try:
        relative = Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError as problem:
        raise CouldNotRun(
            f"the matrix file {path} lies outside the top folder {root}; on the sealed route the "
            "dial's configuration and its kappa come from a matrix file the tag's commit holds "
            "(reading E24); nothing read"
        ) from problem
    text, blob = sealed_route.committed_text(root, route.tag_commit, relative)
    if Path(path).read_text() != text:
        raise CouldNotRun(
            f"the matrix file {relative.as_posix()} differs from the file the tag's commit holds "
            "(reading E24)"
        )
    return {"file": relative.as_posix(), "git_object": blob}


def site_type(days) -> int | None:
    """The site type (`learn_features.SITE_TYPES`) of the sites of a unit's counting days; None
    when it has none. Raises `NotAllowed` for a site with no type or sites of two types."""
    sites = {t.record.site for t in days.days}
    unknown = sorted(s for s in sites if s not in SITE_TYPES)
    if unknown:
        raise NotAllowed(f"the unit {days.unit} holds sites with no site type: {unknown}")
    found = {SITE_TYPES[s] for s in sites}
    if len(found) > 1:
        raise NotAllowed(f"the unit {days.unit} holds sites of two site types")
    return found.pop() if found else None


def route_steps(settings: dict, loaded: dict, units) -> dict:
    """Per unit its site type's name and the settings file's step size per tier for that type, or
    "offers nothing" (readings E17, E20 and E29); a unit with no counting day has neither. Raises
    `NotAllowed` for units of two site types and `CouldNotRun` for a step size the file does not
    hold."""
    types = {unit: site_type(loaded[unit]) for unit in units}
    named = sorted({TYPE_NAMES[t] for t in types.values() if t is not None})
    if len(named) > 1:
        raise NotAllowed(
            f"the units {list(units)} are of the site types {named}; the dial is fixed per site "
            "type (plan section 6) and this command takes one --dial, so a run on the sealed "
            "route names units of one site type (reading E20)"
        )
    steps = {}
    for unit, kind in types.items():
        eta = {}
        for tier in drift_path.TIERS if kind is not None else ():
            try:
                value = drift_settings.step_size(settings, tier, kind)
            except (KeyError, TypeError, ValueError) as problem:
                raise CouldNotRun(
                    f"{settings['file']} holds no step size of tier {tier} for the site type "
                    f"{TYPE_NAMES[kind]} ({type(problem).__name__}: {problem})"
                ) from problem
            if value != drift_settings.OFFERS_NOTHING and not (math.isfinite(value) and value > 0):
                raise CouldNotRun(
                    f"{settings['file']} gives tier {tier} of the site type {TYPE_NAMES[kind]} the "
                    f"step size {value}; a step size is finite and above 0"
                )
            eta[tier] = value
        steps[unit] = {"site_type": None if kind is None else TYPE_NAMES[kind], "eta": eta}
    return steps


def run_tiers(keys, values, offering, evaluate, *, n_calibration, steps, start):
    """`drift_path.run_unit` at one step size per tier (`steps`: tier -> step size) and the one
    start `start`, as on the route: one call per tier, each with the same first days'
    calibration; a tier that offers nothing runs no path (reading E29); with no step size (a unit
    with no counting day, or every tier offering nothing) the unit's status and calibration
    alone."""
    steps = {t: eta for t, eta in steps.items() if eta != drift_settings.OFFERS_NOTHING}
    found = [
        drift_path.run_unit(
            keys,
            values,
            offering,
            evaluate,
            n_calibration=n_calibration,
            etas=(eta,),
            tiers=(tier,),
            starts=(start,),
        )
        for tier, eta in steps.items()
    ]
    if not found:
        return drift_path.run_unit(
            keys, values, offering, evaluate, n_calibration=n_calibration, etas=(), tiers=()
        )
    paths = [path for one in found for path in one.paths]
    return drift_path.UnitRun(found[0].status, found[0].calibration, paths)


class Evaluator:
    """The one-way rung's (slot, driver) losses of one unit's days by (day, scale), each pair
    evaluated once (module text)."""

    def __init__(self, runner, prepared):
        self.runner = runner
        self.prepared = prepared
        self.losses: dict = {}
        self.outcomes: Counter = Counter()

    def __call__(self, pairs) -> dict:
        todo = [p for p in dict.fromkeys(pairs) if p not in self.losses]
        if todo:
            jobs = [(self.prepared[d], s) for d, s in todo]
            for key, found in zip(
                todo, self.runner.map(gate_run._evaluate_task, jobs), strict=True
            ):
                rung = found.rungs[RUNG]
                self.losses[key] = (float(rung.slot_loss), float(rung.driver_loss))
                self.outcomes.update(rung.resolves)
        return {p: self.losses[p] for p in pairs}

    def counts(self) -> dict:
        return {
            "day_scale_pairs": len(self.losses),
            "re_solves_of_the_one_way_rung": dict(sorted(self.outcomes.items())),
        }


def _reasons(prepared, days) -> dict:
    return dict(
        sorted(Counter(prepared[d].abstain_reason for d in days if prepared[d].abstains).items())
    )


def run_one(
    name,
    unit,
    model,
    fit_units,
    matrix,
    configuration,
    loaded,
    *,
    n_calibration,
    etas,
    workers,
    steps=None,
    start=None,
):
    """One unit (module text): (`drift_path.UnitRun`, the entry's further fields). With `steps`
    (tier -> step size, on the route) the unit runs at those step sizes and the one start
    `start` (`run_tiers`), else at every step size of `etas` and both starts."""
    days = loaded[unit]
    own = unit_matrix(matrix, unit)
    factory, _ = dial_choice.make_factory(
        name, model, own, {unit: days}, {u: loaded[u].cars for u in fit_units}, configuration.kappa
    )
    runner = gate_run.Runner(workers, {unit: days}, own, factory)
    try:
        prepared = runner.map(
            gate_run._prepare_task, [(unit, i, configuration) for i in range(len(days.days))]
        )
        evaluator = Evaluator(runner, prepared)
        keys = [f"{p.date} {p.site}" for p in prepared]
        values = [p.value for p in prepared]
        offering = [not p.abstains for p in prepared]
        if steps is None:
            found = drift_path.run_unit(
                keys, values, offering, evaluator, n_calibration=n_calibration, etas=etas
            )
        else:
            found = run_tiers(
                keys,
                values,
                offering,
                evaluator,
                n_calibration=n_calibration,
                steps=steps,
                start=start,
            )
    finally:
        runner.close()
    split = min(n_calibration, len(prepared))
    extra = {
        "first_day": keys[0] if keys else None,
        "last_day": keys[-1] if keys else None,
        "certificate_timeouts": sum(p.certificate_outcome == "timeout" for p in prepared),
        "days_not_offering": {
            "calibration": _reasons(prepared, range(split)),
            "guarded": _reasons(prepared, range(split, len(prepared))),
        },
        "evaluations": evaluator.counts(),
    }
    return found, extra


# --- the results ---------------------------------------------------------------------------------


def eta_key(eta: float) -> str:
    return f"{eta:g}"


def unit_entry(unit, found, counting_days, *, fit_units, rotation, extra=None) -> dict:
    """The results' entry of one unit (module text)."""
    calibration = found.calibration
    paths: dict = {}
    for path in found.paths:
        by_start = paths.setdefault(path.tier, {}).setdefault(path.start, {})
        by_start[eta_key(path.eta)] = path.summary()
    entry = {
        "role": role(unit),
        "fit_units": list(fit_units),
        "rotation_whatif": rotation,
        "counting_days": counting_days,
        "status": found.status,
        "calibration": None
        if calibration is None
        else {
            "days": calibration.days,
            "offering_days": calibration.offering_days,
            "first_day": calibration.first_key,
            "last_day": calibration.last_key,
            "tier_scale": calibration.tier_scale,
            "reason": calibration.reason,
            "loss_scale": calibration.loss_scale,
        },
        # with no path but a calibration (every tier offering nothing, reading E29), the days
        # after the calibrating ones
        "guarded_days": len(found.paths[0].rows)
        if found.paths
        else (0 if calibration is None else counting_days - calibration.days),
        "paths": paths,
    }
    entry.update(extra or {})
    return entry


def definitions() -> dict:
    return {
        "time_order": (
            "a unit's counting days in date order (then site); the first n_calibration "
            "calibrate (drift_path, reading E2) and the rest are guarded days"
        ),
        "calibration": (
            "dial_splits.on_grid_scale on the offering calibration days' one-way losses at every "
            "scale of dial_splits.GRID: per tier its scale, and per loss it holds down the scale "
            "of that loss alone (the other alpha 1)"
        ),
        "guards": (
            "tier A: a slot guard at alpha 0.10 and a driver guard at alpha 0.05, the scale "
            "offered the smaller state; tier B: a slot guard alone, the driver loss reported; "
            "each moves by drift_guard.step, unclipped; the tiers are never pooled"
        ),
        "starts": (
            "calibrated: each guard starts at its loss's calibrated scale; zero: each guard "
            "starts at 0, the plan's beta_1 = 0 (drift_path, reading E1)"
        ),
        "offering": (
            "a guarded day that does not offer is no step (reading E3); a step whose smallest "
            "state is 0 or below abstains with loss 0; the scale offered is capped at "
            f"{drift_path.CAP:g} (reading E4); the loss is the raw day-level loss (reading E5)"
        ),
        "bound": (
            "per guarded loss after t steps: alpha + (beta_1/eta + 1 - alpha)/t (plan T10); "
            "days_over_bound counts steps whose running mean is above it (by more than 1e-9)"
        ),
        "mean_offer_kwh_per_counting_day": (
            "the scale offered times the certified value at scale 1, averaged over the guarded "
            "counting days (0 on days that do not offer or abstain)"
        ),
        "label": LABEL,
    }


def results(
    model, configuration, units, etas, n_calibration, entries, failed, starts=drift_path.STARTS
) -> dict:
    """The results document without its inputs (module text)."""
    return {
        "label": LABEL,
        "definitions": definitions(),
        "run": {
            "model": model,
            "configuration": configuration.name,
            "dial": configuration.driver_risk,
            "kappa": configuration.kappa,
            "rung": RUNG,
            "n_calibration": n_calibration,
            "etas": list(etas),
            "cap": drift_path.CAP,
            "grid": list(drift_path.GRID),
            "tiers": list(drift_path.TIERS),
            "starts": list(starts),
            "units": list(units),
        },
        "units": entries,
        "failed_units": sorted(f"{unit}: {why}" for unit, why in failed.items()),
    }


def day_rows(unit: str, found) -> list[dict]:
    """The day table's rows of one unit (`DAY_COLUMNS`)."""
    rows = []
    for path in found.paths:
        for r in path.rows:
            date, site = r.key.split(" ", 1)
            mean = r.mean_loss or {}
            bound = r.bound or {}
            losses = r.losses or (None, None)
            rows.append(
                {
                    "unit": unit,
                    "tier": path.tier,
                    "start": path.start,
                    "eta": eta_key(path.eta),
                    "day": r.day,
                    "date": date,
                    "site": site,
                    "offering": r.offering,
                    "step": r.step,
                    "state_slot": r.state_before.get("slot"),
                    "state_driver": r.state_before.get("driver"),
                    "scale": r.scale,
                    "offer_kwh": r.offer_kwh,
                    "slot_loss": losses[0],
                    "driver_loss": losses[1],
                    "mean_slot_loss": mean.get("slot"),
                    "mean_driver_loss": mean.get("driver"),
                    "bound_slot": bound.get("slot"),
                    "bound_driver": bound.get("driver"),
                }
            )
    return rows


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | str):
        return str(value)
    return repr(float(value))


def days_csv(rows) -> str:
    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(DAY_COLUMNS)
    for row in rows:
        writer.writerow([_text(row[c]) for c in DAY_COLUMNS])
    return handle.getvalue()


# --- the summary ---------------------------------------------------------------------------------


def _number(value, form: str = ".3f") -> str:
    return "none" if value is None else format(value, form)


def _loss_text(entry: dict) -> str:
    if not entry["guarded"]:
        return f"{_number(entry['mean'])} (reported)"
    return f"{_number(entry['mean'])} ({_number(entry['final_bound'])})"


TABLE_HEAD = (
    f"    {'eta':<6}{'offering days':>14}{'guard abstains':>16}{'offer kWh/counting day':>24}"
    f"{'slot mean (bound)':>20}{'driver mean (bound)':>21}{'steps over bound':>18}"
)


def _etas_text(document: dict) -> str:
    """The step sizes of the summary's run line: off the route the grid; on it each tier's step
    size, the settings file's for the units' site type (module text: outputs)."""
    found = document.get(gate_summary.ROUTE_KEY)
    if not found:
        return ", ".join(eta_key(e) for e in document["run"]["etas"])
    tiers = []
    nothing = drift_settings.OFFERS_NOTHING
    for tier in drift_path.TIERS:
        values = {e["eta"][tier] for e in found["step_sizes"].values() if tier in e["eta"]}
        texts = [eta_key(e) for e in sorted(values - {nothing})]
        texts += [nothing] if nothing in values else []  # reading E29
        if texts:
            tiers.append(f"tier {tier} {', '.join(texts)}")
    return ", ".join(tiers) if tiers else "none (no unit has a counting day)"


def _nothing_lines(document: dict, unit: str) -> list[str]:
    """On the route, one line per tier the settings file gives "offers nothing" for the unit's
    site type (reading E29); none off the route."""
    found = document.get(gate_summary.ROUTE_KEY)
    if not found:
        return []
    step = found["step_sizes"].get(unit, {})
    return [
        f"  {TIER_LABELS[tier]}: offers nothing on every guarded day: the settings file gives the "
        f"site type {step['site_type']} no step size, the smallest step size of its run on "
        "training units already failing (plan section 6, T10; reading E29)."
        for tier in drift_path.TIERS
        if step.get("eta", {}).get(tier) == drift_settings.OFFERS_NOTHING
    ]


def summary_lines(document: dict) -> list[str]:
    run = document["run"]
    lines = [
        f"{LABEL}.",
        f"Model {run['model']}, dial {run['dial']:g} ({run['configuration']}), rung {run['rung']}; "
        f"the first {run['n_calibration']} counting days of each unit calibrate; step sizes "
        f"{_etas_text(document)}; the scale offered is capped at {run['cap']:g}.",
    ]
    for unit, entry in document["units"].items():
        if entry.get("rotation_whatif"):
            lines.append(f"{rotation_whatif.WHAT}: {entry['rotation_whatif']['evidence']}")
        lines.append("")
        head = f"{unit} ({entry['role']}): {entry['counting_days']} counting days"
        calibration = entry["calibration"]
        if calibration is None:
            lines.append(f"{head}; {entry['status']}.")
            continue
        scales = calibration["tier_scale"]
        lines.append(
            f"{head}, {entry.get('first_day')} to {entry.get('last_day')}; calibration on days "
            f"{calibration['first_day']} to {calibration['last_day']}: "
            f"{calibration['offering_days']} of {calibration['days']} offer; calibrated scale "
            f"tier A {scales['A']:g}, tier B {scales['B']:g}; {entry['guarded_days']} guarded "
            "counting days."
        )
        lines += _nothing_lines(document, unit)
        for tier, by_start in entry["paths"].items():
            for start, by_eta in by_start.items():
                lines.append(f"  {TIER_LABELS[tier]}, start {start}:")
                lines.append(TABLE_HEAD)
                for eta, s in by_eta.items():
                    over = sum(
                        e.get("days_over_bound", 0) for e in s["losses"].values() if e["guarded"]
                    )
                    lines.append(
                        f"    {eta:<6}{s['offering_days']:>14}{s['guard_abstaining_days']:>16}"
                        f"{_number(s['mean_offer_kwh_per_counting_day']):>24}"
                        f"{_loss_text(s['losses']['slot']):>20}"
                        f"{_loss_text(s['losses']['driver']):>21}{over:>18}"
                    )
    if document["failed_units"]:
        lines += ["", "FAILED UNITS:", *(f"  {f}" for f in document["failed_units"])]
    if document.get(gate_summary.ROUTE_KEY):
        return [*lines, "", ROUTE_LAST]
    lines += [
        "",
        "No step size is chosen here; drift_settings applies a named rule to a run on training "
        "units.",
    ]
    return lines


def mark_route(document, route, held_out_inputs, fit_units, settings, steps, matrix) -> None:
    """Records the route in `document` (module text: outputs); nothing when the route is off."""
    if not route.held_out:
        return
    document[gate_summary.ROUTE_KEY] = {
        "kind": sealed_route.HELD_OUT,
        "label": ROUTE_LABEL,
        "tag_commit": route.tag_commit,
        **sealed_route.fix_record(route),
        "held_out_inputs": held_out_inputs,
        "fit_units": list(fit_units),
        "matrix": matrix,
        "settings": settings,
        "step_sizes": steps,
        "checks": sealed_route.CHECKS,
    }


def _steps_text(steps: dict) -> str:
    parts = []
    for unit, entry in steps.items():
        if entry["site_type"] is None:
            parts.append(f"{unit} (no counting day) none")
            continue
        tiers = ", ".join(
            f"tier {tier} {drift_settings.eta_text(eta)}" for tier, eta in entry["eta"].items()
        )
        parts.append(f"{unit} ({entry['site_type']}) {tiers}")
    return "; ".join(parts)


def _not_made_lines(document: dict) -> list[str]:
    """For each unit the held-out build abstained, a line saying the drift measure could not be
    made for it, and when no unit was measured at all a line saying that nothing was (reading S4
    of `sealed_route`: the unit counts as not passing); none when no unit abstained."""
    not_made = document.get("measure_not_made") or {}
    lines = [
        f"The drift measure could not be made for {unit}: {why}; counted as "
        f"{sealed_route.COUNTS_AS}."
        for unit, why in not_made.items()
    ]
    if not_made and not document["units"]:
        lines.append("NOTHING WAS MEASURED: no unit of this run has tables.")
    return lines


def route_head(document: dict) -> list[str]:
    """The summary's first three lines on the route (module text: outputs); none off it."""
    found = document.get(gate_summary.ROUTE_KEY)
    if not found:
        return []
    fitted = found["fit_units"]
    fit = f"the fit units' ({', '.join(fitted)})" if fitted else "no fit unit's"
    s = found["settings"]
    fix = sealed_route.fix_text(found)
    return [
        f"{found['label']} (tag commit {found['tag_commit']}{fix}): the scored units' tables from "
        f"the held-out build {found['held_out_inputs']['truth_run']}, {fit} from the development "
        f"build {document['inputs']['truth_run']}.",
        found["checks"],
        f"Step sizes from {s['file']} as the tag's commit holds it (git object {s['git_object']}; "
        f"rule {s['rule']}, start {s['start']}, grouping {s['grouping']}, units "
        f"{', '.join(s['units'])}, checked to be the plan's rule, start and grouping and training "
        f"units; written, as the file records and nothing here checks, from the run "
        f"{s['results']['run_folder']} at commit {s['results']['commit']}, model {s['model']}, "
        f"dial {s['dial']}): {_steps_text(found['step_sizes'])}.",
        *sealed_route.abstention_lines(found["held_out_inputs"]),
        *_not_made_lines(document),
    ]


# --- the command ---------------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    found.add_argument("--model", required=True)
    found.add_argument("--dial", type=float, required=True)
    found.add_argument("--matrix", type=Path, default=MATRIX)
    found.add_argument("--units", nargs="+", default=list(TRAIN_UNITS))
    # Off the route: `ETAS` and `drift_path.N_CALIBRATION` when not given (`counts`).
    found.add_argument("--etas", nargs="+", default=None)
    found.add_argument("--calibration-days", type=int, default=None)
    found.add_argument("--workers", type=int, default=1)
    found.add_argument("--tables", type=Path, default=OUT)
    found.add_argument("--truth-run", type=Path, default=None)
    found.add_argument("--held-out-tables", type=Path, default=None)
    found.add_argument("--held-out-truth-run", type=Path, default=None)
    return found


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _load(args, matrix, route, needed, read) -> dict:
    """Each needed unit's counting days: on the route from the bytes the two builds' checks
    hashed (`read`), off it from the tables folder (module text). Raises `ValueError`."""
    readings = {"rho": matrix.rho, "no_fit": matrix.no_fit, "small_fleet": matrix.small_fleet}
    if route.held_out:
        return {name: parse_unit_days(read, name, **readings) for name in needed}
    return {name: load_unit_days(args.tables, name, **readings) for name in needed}


def main(argv: list[str] | None = None, root: Path = TOP) -> int:
    args = parser().parse_args(argv)
    settings = matrix_file = None
    try:
        route = take_route(root, args)
        units = check_units(args.units, route.held_out)
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        n_calibration, etas = counts(args, route.held_out)
        if route.held_out:
            settings = route_settings(root, route)
            gate_run.check_files([args.matrix])
            matrix_file = route_matrix(root, route, args.matrix)
        matrix = load(args.matrix)
        configuration = compare_run.dial_configurations(matrix, args.dial)["forecast"]
        models = check_models(args.model, matrix, units)
        sealed_route.gate_again(route, root)
    except (
        MatrixError,
        NotAllowed,
        learn_run.FitOnScored,
        ValueError,
        sealed_route.RouteRefused,
        CouldNotRun,
    ) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    fit = tuple(dict.fromkeys(u for unit in units for u in models[unit][1]))
    needed = tuple(dict.fromkeys(u for unit in units for u in (unit, *models[unit][1])))
    held_inputs, read = None, {}
    try:
        if route.held_out:
            inputs, held_inputs, read = sealed_route.read_two_builds(
                args.truth_run, args.tables, fit, route.truth_run, route.tables, units
            )
            units = sealed_route.scored_units(units, held_inputs)  # abstaining units out
            needed = tuple(dict.fromkeys(u for unit in units for u in (unit, *models[unit][1])))
        else:
            inputs = gate_run.check_inputs(args.truth_run, args.tables, needed)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        loaded = _load(args, matrix, route, needed, read)
    except CouldNotRun as problem:  # a table changed after its check (on the route)
        print(f"COULD NOT RUN: {problem}")
        return 2
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    steps, starts = {}, drift_path.STARTS
    if route.held_out:
        try:
            steps = route_steps(settings, loaded, units)
        except (NotAllowed, CouldNotRun) as problem:
            print(f"COULD NOT RUN: {problem}")
            return 2
        starts = (settings["start"],)
        etas = tuple(
            sorted(
                {
                    eta
                    for entry in steps.values()
                    for eta in entry["eta"].values()
                    if eta != drift_settings.OFFERS_NOTHING  # reading E29
                }
            )
        )
    started = time.perf_counter()
    entries, rows, failed, timing = {}, [], {}, {"units": {}}
    for unit in units:
        begun = time.perf_counter()
        model, fit_units = models[unit]
        try:
            found, extra = run_one(
                args.model,
                unit,
                model,
                fit_units,
                matrix,
                configuration,
                loaded,
                n_calibration=n_calibration,
                etas=etas,
                workers=args.workers,
                steps=steps[unit]["eta"] if route.held_out else None,
                start=starts[0] if route.held_out else None,
            )
        except Exception as problem:  # recorded, the other units go on
            failed[unit] = f"{type(problem).__name__}: {problem}"
            continue
        rotation = rotation_whatif.record(unit_matrix(matrix, unit), fit_units)
        entries[unit] = unit_entry(
            unit,
            found,
            len(loaded[unit].days),
            fit_units=fit_units,
            rotation=rotation,
            extra=extra,
        )
        rows.extend(day_rows(unit, found))
        timing["units"][unit] = time.perf_counter() - begun
    document = results(
        args.model, configuration, units, etas, n_calibration, entries, failed, starts
    )
    document["inputs"] = inputs
    if route.held_out and sealed_route.abstained_of(held_inputs):
        document["measure_not_made"] = {
            unit: sealed_route.abstained_text(reason)
            for unit, reason in sealed_route.abstained_of(held_inputs).items()
        }
    mark_route(document, route, held_inputs, fit, settings, steps, matrix_file)
    document["matrix"] = {
        "file": args.matrix.name,
        "sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
    }
    table = days_csv(rows)
    document["days_file"] = {
        "file": DAYS_NAME,
        "rows": len(rows),
        "sha256": hashlib.sha256(table.encode()).hexdigest(),
    }
    timing["total_seconds"] = time.perf_counter() - started
    text = "\n".join([*route_head(document), *summary_lines(document)]) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / DAYS_NAME).write_text(table)
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
        (folder / TIMING_NAME).write_text(_json(timing))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
