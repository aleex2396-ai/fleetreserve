"""The WP8 evaluation of one departure model on the development units (plan sections 7, 13 WP8).

Command line, from the project's top folder:
    python -m fleetreserve.learn_run --model <name> [--matrix <file>] [--tables <folder>]
        [--truth-run <folder>] [--workers N] [--interval-seed S [--interval-resamples N]]
        [--held-out-tables <folder>] [--held-out-truth-run <folder>]

`--interval-seed` and `--interval-resamples`: as `gate_run` takes them (the resamples' pairs for
`day_intervals`, written into `learn-day-losses.json`; every other output unchanged).

`--model <name>`: the module `fleetreserve.models.<name>`, loaded by name; it builds the model
with `make_model(matrix)` (`forecast_model.DepartureModel`). `--matrix`: a run matrix in the gate's
format (`gate_config`); the committed one is `learn/matrix.json`, the gate matrix's candidate family
at every kappa of plan section 2 with departures seen at the start of each hour (the reading the
gate matrix reports beside its main one; this matrix has not been moved to the main reading),
each with its oracle partner, every other value as in `gate/matrix.json`. Its gate units are the
development units scored. The matrix file passes `gate_run.check_files` before it is read, as
every file `gate_run` and this command are given does (`gate_run`, module text).

Steps:

0. The model is refused (exit 1) when its `fit_units` name a unit that is not a training unit
   (`truth_build.TRAIN_UNITS`), or when it declares `reads_other_day_outcomes` and its module is
   not named in `forecast_model.OUTCOME_READERS`. Development units are only forecast and
   scored, never fitted on; held-out units likewise, and only on the route below. A model whose
   `fit_units` share a unit with the matrix's gate units is refused as well (`FitOnScored`; exit
   2, could not run: the model and the matrix do not fit together), so no unit is scored by a
   model fitted on it. The committed
   matrices score development units, which no model is fitted on, so this changes none of their
   runs; it matters for a rotation what-if (`rotation_whatif`, `gate_config`), which scores a
   training unit with `models.lgbm_tail_rotation`. A model that reads a look-ahead column
   (`model_inputs`: values of times after τ) or whose module is in
   `forecast_model.LOOK_AHEAD_MODELS` is refused (exit 1) unless the caller allows look-ahead;
   only this command's `main`, the development evaluation, does, and a model that reads one
   must be in that list. Every other caller of `check_model` and `evaluate` (`dial_choice` and
   every command through `dial_choice.make_factory`) leaves it refused, and so does `main` when
   it scores held-out units (below; reading R3).
1. The truth tables of the gate units and of the model's `fit_units` are checked against the
   recorded truth build at the checked-out commit, as `gate_run` checks them, each read once
   (`gate_run.read_inputs`), and the bytes hashed are the bytes parsed
   (`days_from_truth.parse_unit_days`, the matrix's readings). Each path argument is first put
   through `gate_run.argument_path` (a path written as a URL could not run; a leading `~` is
   expanded), and the matrix file is read once (`gate_config.read`).
2. `fit` on the cars tables of `fit_units`; `predict` at every level of the matrix
   (`driver_risk.matrix_levels`: each configuration's kappa and, for a configuration with a
   driver dial, every level up to its kappa, which its driver-risk rows read) on the gate
   units' frame (`forecast_frame`): for a learned model `learn_features.feature_frame` of their
   counting days, with no outcome or label column, and for a model declaring `extra_inputs`
   those columns appended before the labels are dropped (`model_inputs`); for a model declaring
   `reads_other_day_outcomes`, their cars tables. An error raised inside `fit` or `predict` (for
   example a learned model reading a column it is not given) fails the model. The table is
   checked (`forecast_model.check_table`) and must hold every car of every counting day's fleet
   at each of those levels.
3. Losses and concordance index per unit and kappa of the matrix's configurations (`kappas_of`,
   `learn_eval.unit_scores`).
4. The calibrated offers: `gate_run.run_gate` with the table as the forecaster
   (`forecast_model.TableFactory`), so every certificate and re-solve keeps the gate's time limits
   and the split, calibration and scoring are the gate's.

Outputs, written into `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary
also on standard output:
- `learn.json`: per unit and kappa the losses and concordance index; per configuration the
  fixed and one-way rungs' calibrated scale and test-day offer (with shares of both ceilings, the
  numbers of offering and failing test days and the mean losses over the offering test days), and
  the same for its oracle partner. Each is given under tier A (driver-protected) and, in the key
  `tier_b` of the rung, under tier B (deadline contract) with its stay-to-bound check, when the
  matrix lists tier B (plan section 6); the tiers are shown side by side and never combined;
- `learn-forecasts.csv`: the model's forecast table (`forecast_model.write_table`);
- `learn-gate.json`: the gate-format results the offers are read from (every number of the gate
  run, as `gate_run` writes `gate.json`);
- `learn-gate-days.csv`: one row per unit, configuration and day, as `gate_run` writes
  `gate-days.csv` (the certified value, both ceilings and each rung's offer and losses per day;
  read by the ledger of plan T9, `ledger_run`);
- `learn-timing.json`: wall-clock and solver seconds, kept apart from the compared results;
- `learn-day-losses.json`: the per-day losses the calibration read and the test days' losses at the
  chosen scales (`day_losses`), for the intervals over days (`day_intervals`).

The share of days in E' (plan section 5, T2's event; the key `e_prime` of each unit of `learn.json`,
by configuration, its oracle partners included, and `e_prime_definition`, which words the
denominators; one summary line per unit and configuration). E'_i is `forecast_coverage.check_car`'s
`all_hold` (the one home of the test; `forecast_report` and `t2_day` read it too), applied to the
forecast limits the gate scored the configuration on (`gate_day._forecasts`, departure floored
where the configuration floors it), against the cars' true limits, with no replay:
  E1. A day is in E' when it was certified (its row of the gate's day rows has no abstain reason)
      and every steered car is in E'_i (as `t2_day`: a car with no forecast or an empty forecast
      set is not steered and is not judged). The counts are over the unit's counting test days:
      `test_days`, `certified_test_days`, `days_in_e_prime`, and the shares of the first two
      (`dev_day_verdicts` gives both denominators); the cars are the steered cars of the certified
      test days. Two differences from the other homes, both readings: a day is
      certified here only when its box has a value above `zero_value_kwh` (the gate's rule: no
      steered car, no optimal certificate, or a value at or below it makes a day abstain), while
      `t2_day` and `dev_day_verdicts` count any optimal certificate, a value of zero included, so
      the same forecasts can give more certified days and more days in E' there; and
      `forecast_report` judges every car with a forecast on every day, the cars of an empty
      forecast set too. [Alternatives: those two readings; offering days only.]
  E2. A unit with no counting test day (it cannot be split) has zero counts and a share of null;
      a share is null whenever its denominator is 0, never 0 and never 1. A configuration that
      failed has a `reason` and no counts.
  E3. The steered cars are found again here, not read from the gate; the gate's own count
      (`steered_cars` of its day rows) must equal them on every certified test day, or the run
      fails (exit 1, `EPrimeDisagrees`, message naming the unit, site, day and configuration);
      the results are written only after the scoring ends, so no results file is written then.
The same figures are written on the sealed route.

What the guard does not do: it prevents the accidental use of outcome columns, by what a model
is given and by the refusals of step 0. The decision on whether a model may be given the cars
tables is taken once, in `check_model`, before `fit`, from the module name and the declared
attribute; the attribute is not read again, so setting it later changes nothing. A model that runs
in the same process could still reach the true departures on purpose (for example by opening the
tables itself or by reading this module's memory). A learned model's code is therefore
reviewed before its result is used.

Exit status: 0 the model was scored and every configuration ran (on the held-out route below, also
when a configuration raised an error on a unit: that unit is written as `failed: <error>` under the
configuration, counts as not passing and is named in `failed_configurations`; the other units and
configurations are kept; the sealed-run check fails naming it); 1 the model was refused (step
0) or failed, its table was refused or incomplete, or (off that route) a configuration failed; 2
could not run (a bad matrix, argument or model name, a path written as a URL, a model fitted on a
gate unit, a table missing, a file in a refused place, no verified truth build at this commit, a
table differing from it; for held-out units, a refusal of the route below).

Held-out units, through the one shared route of `sealed_route` (`sealed_route.matrix_route`, the
cold start's checks, in its order), which starts only at the tag `frozen-plan` with a clean tree;
its checks stop an honest operator's ordinary mistakes and are not a defence against deliberate
evasion (`sealed_route`, module text, first paragraph, with the commands that have the route and
those that have none yet). When the matrix's gate units are
held-out units (all of them; a mix with other units could not run), before the model is loaded:
every folder named lies outside the held-out places of the top folder, the tag gate passes (the
held-out loader's checks 2 to 8: at the tag `frozen-plan`, clean, the running code the tag's), and
`--held-out-truth-run` is given (`--held-out-tables` defaults to the held-out build's main
folder); after the model is loaded, the gate is passed again (its module is then among the
running code). A model of
`forecast_model.OUTCOME_READERS` (`empirical`, which builds a scored unit's forecasts from that
unit's own other days, test days included) could not run there (exit 2,
`sealed_route.OUTCOME_READER`, as the cold start refuses it, its reading C12). Before any table is
read the matrix's split rule must be the held-out build's (`sealed_route.check_split_rule`). The
model's `fit_units` are then read from the development build (`--tables`, `--truth-run`) and the
gate units from the held-out build (`--held-out-tables`, `--held-out-truth-run`), each build
checked and read once as in step 1 and the two of one setting (`sealed_route.read_two_builds`),
and each scored unit's calibration days as the run draws them must be the build's
(`sealed_route.check_split_days`).
Everything else is step 0 to 4 unchanged, except that no kappa is selected per unit on the scored
units' test days (`gate_run.run_gate` with `selection` false). `learn.json` and `learn-gate.json`
then hold `inputs` (the development build's check, of the fit units) and `sealed_route` (the tag's
commit, the held-out build's check, the fit units, `kappa_selection`: none, and `checks`: what the
route's checks stop and what they do not, `sealed_route.CHECKS`); the summary's first line says
so, its second is that `checks` text, and its model line says held-out scores. With other gate
units the two held-out options could not run, and without them nothing above is checked: the
run, its outputs and its summary are as before.

Readings of the route, each with its alternative in brackets:
  R1. The kind of a run is that of the matrix's gate units, as in `gate_run`. [Alternative: a
      command-line choice of the kind, checked against the units.]
  R3. On the route a look-ahead model is refused (exit 1), as the cold start refuses it: a
      held-out score is never a look-ahead what-if. [Alternative: allowed, as on development
      units, with its look-ahead label.]

A rotation what-if says so in every output (`rotation_whatif`): the first lines of the summary and
the key `rotation_whatif` of `learn.json` and `learn-gate.json`. A model with extra input columns
adds the key `context_inputs` (its columns, the 9/80 parity, the SHA-256 of each context file
read) to both files, and a look-ahead model the key `look_ahead` and the label as the first lines
of the summary (`model_inputs.mark`, `model_inputs.summary_head`).

`--made-after-sealed-run` (`after_sealed_run`): a trial on development units made after the sealed
run's results were seen. Every output then says so: the first lines of the summary, the key
`made_after_sealed_run` of `learn.json` and `learn-gate.json`, and the file
`after_sealed_run.FILE_NAME` of the run folder. It is refused (exit 2) on the held-out route.
Without it nothing changes.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from fleetreserve import (
    after_sealed_run,
    day_losses,
    driver_risk,
    gate_day,
    gate_run,
    gate_summary,
    learn_eval,
    learn_features,
    model_inputs,
    rotation_whatif,
    sealed_route,
)
from fleetreserve.days_from_truth import parse_unit_days
from fleetreserve.forecast_coverage import check_car
from fleetreserve.forecast_model import (
    LOOK_AHEAD_MODELS,
    OUTCOME_READERS,
    TableFactory,
    TableForecaster,
    check_table,
    write_table,
)
from fleetreserve.gate_config import TIER_LABELS, MatrixError, read
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT, TRAIN_UNITS
from fleetreserve.truth_verify import CouldNotRun

MATRIX = TOP / "learn" / "matrix.json"
RESULTS_NAME = "learn.json"
TABLE_NAME = "learn-forecasts.csv"
GATE_NAME = "learn-gate.json"
DAYS_NAME = "learn-gate-days.csv"
TIMING_NAME = "learn-timing.json"
DAY_LOSSES_NAME = "learn-day-losses.json"


def load_model(name: str, matrix):
    """The model `fleetreserve.models.<name>` builds; raises `MatrixError` when there is none."""
    if not name.isidentifier():
        raise MatrixError(f"a model name is a Python name, got {name!r}")
    module_name = f"fleetreserve.models.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as problem:
        if problem.name != module_name:
            raise
        raise MatrixError(f"no model module {module_name}") from problem
    if not callable(getattr(module, "make_model", None)):
        raise MatrixError(f"{module_name} has no make_model(matrix)")
    return module.make_model(matrix)


class EPrimeDisagrees(Exception):
    """The steered cars found again for E' differ from the gate's count (module text, E3)."""


class Refused(Exception):
    """A model the shared path does not run (module text, step 0)."""


class FitOnScored(Refused):
    """A model fitted on a unit the run scores (module text, step 0): the run could not run."""


class ModelFailed(ValueError):
    """An error raised inside a model's `fit` or `predict`."""


def reads_outcomes(model) -> bool:
    return getattr(model, "reads_other_day_outcomes", False) is True


def check_model(
    name: str,
    model,
    fit_units: tuple[str, ...] | None = None,
    gate_units: tuple[str, ...] | None = None,
    look_ahead_allowed: bool = False,
) -> bool:
    """Step 0 of the module text; raises `Refused`. Returns the one decision on whether the model
    is an allowed outcome reader (declared, and listed in `OUTCOME_READERS`); it is taken here,
    before `fit`, and the attribute is never read again. `fit_units` is the one reading of the
    model's `fit_units` (read here when not given). `gate_units`: the units the run scores (the
    matrix's); a model whose `fit_units` share a unit with them raises `FitOnScored`. Every
    caller that has a matrix passes them; without them only the other checks run.
    `look_ahead_allowed`: only `main` passes true (module text, step 0)."""
    try:
        ahead = model_inputs.reads_look_ahead(model)
    except ValueError as problem:
        raise Refused(f"model {name}: {problem}") from problem
    if (ahead or name in LOOK_AHEAD_MODELS) and not look_ahead_allowed:
        raise Refused(
            f"model {name} reads values of times after the decision time (a look-ahead what-if); "
            "only the development evaluation of learn_run may run it"
        )
    if ahead and name not in LOOK_AHEAD_MODELS:
        raise Refused(
            f"model {name} reads a look-ahead column, which only the models "
            f"{list(LOOK_AHEAD_MODELS)} may (forecast_model.LOOK_AHEAD_MODELS)"
        )
    if fit_units is None:
        fit_units = tuple(model.fit_units)
    outside = [u for u in fit_units if u not in TRAIN_UNITS]
    if outside:
        raise Refused(
            f"model {name} declares fit_units {outside}, which are not training units; a model "
            f"is fitted on training units only ({', '.join(TRAIN_UNITS)})"
        )
    if gate_units is not None:
        overlap = [u for u in fit_units if u in gate_units]
        if overlap:
            raise FitOnScored(
                f"model {name} is fitted on {overlap}, which the matrix scores; a unit is never "
                f"scored by a model fitted on it (fit_units {list(fit_units)}, gate_units "
                f"{list(gate_units)})"
            )
    declared = reads_outcomes(model)
    if declared and name not in OUTCOME_READERS:
        raise Refused(
            f"model {name} declares reads_other_day_outcomes, which only the models "
            f"{list(OUTCOME_READERS)} may declare (forecast_model.OUTCOME_READERS)"
        )
    return declared


def kappas_of(matrix) -> list[float]:
    return sorted({c.kappa for c in matrix.configurations})


def forecast_frame(
    outcome_reader: bool, units: dict, names, extra: model_inputs.ExtraInputs | None = None
) -> pd.DataFrame:
    """The frame `predict` is given for the units `names` (module text, step 2); `outcome_reader`
    is the decision of `check_model`; `extra`, the model's extra input columns
    (`model_inputs.extra_of`), appended to each unit's table before its labels are dropped."""
    if outcome_reader:
        if extra is not None:
            raise ValueError("a model that reads outcomes takes no extra input columns")
        return pd.concat([units[u].cars for u in names], ignore_index=True)
    frames = []
    for u in names:
        table = learn_features.build_table(units[u].cars, rho=units[u].rho)
        if extra is not None:
            table = extra.add(table, units[u].cars)
        frames.append(learn_features.feature_frame(table))
    frame = pd.concat(frames, ignore_index=True)
    return frame[frame["counting"]].reset_index(drop=True)


def _call(what: str, method, *arguments):
    try:
        return method(*arguments)
    except Exception as problem:
        raise ModelFailed(
            f"the model's {what} raised {type(problem).__name__}: {problem}"
        ) from problem


E_PRIME_DEFINITION = (
    "E' (plan T2): a day is in E' when it was certified and every steered car is in E'_i, that "
    "is its forecast departure is at or before its true departure hour (d_hat <= d) and its "
    "forecast rate, stored-energy cap and needed energy are conservative; steered cars are the "
    "cars of the day with a forecast and a non-empty forecast set. Counted per configuration, "
    "as forecast at its kappa, departure rule and energy bounds, over the counting test days "
    "(the days left after the calibration days). certified test days: counting test days whose "
    "box has a certificate with value above zero_value_kwh (the gate's rule; t2_day and "
    "dev_day_verdicts count any optimal certificate, a value of zero included); a day with none "
    "is not in E'. "
    "share_of_test_days_in_e_prime: days_in_e_prime over counting test days; "
    "share_of_certified_test_days_in_e_prime: over certified test days; "
    "share_of_steered_cars_in_e_prime_i: steered cars in E'_i over the steered cars of the "
    "certified test days. A share is null when its denominator is 0."
)
COUNTS = (
    "test_days",
    "certified_test_days",
    "days_in_e_prime",
    "steered_cars_on_certified_test_days",
    "steered_cars_in_e_prime_i",
)


def _share(count: int, of: int) -> float | None:
    return None if of == 0 else count / of


def _steered_in_e_prime(truth, forecaster, configuration) -> tuple[int, int]:
    """(steered cars, steered cars in E'_i) of one day, as `configuration` forecasts it. The cars
    left out of the steered fleet are those `gate_day.prepare_day` leaves out for a missing
    forecast or an empty forecast set."""
    day = truth.day
    forecasts = gate_day._forecasts(truth, forecaster, replace(configuration, driver_risk=None))
    latest = max([0.0, *(f.departure for f in forecasts if f is not None)])
    n_recovery = max(day.n_recovery, math.ceil(latest) - 1 - day.n_service)
    steered = inside = 0
    for i, forecast in enumerate(forecasts):
        if forecast is None:
            continue
        true = day.cars[i]
        check = check_car(
            forecast,
            true.departure,
            truth.fleet[i].departure_exact,
            true.rate,
            true.room,
            day.rho,
            day.n_service,
            n_recovery,
        )
        if not check.empty:
            steered += 1
            inside += check.all_hold
    return steered, inside


def e_prime_figures(matrix, units: dict, table: pd.DataFrame, day_rows: list) -> dict:
    """Per gate unit and configuration of `matrix`, the counts and shares of E' (module text,
    readings E1 to E3) from the forecast `table`, the units' days and the gate's day rows."""
    rows = {(r["unit"], r["configuration"], r["site"], r["date"]): r for r in day_rows}
    out = {}
    for unit in matrix.gate_units:
        days = units[unit]
        plan = gate_run.plan_unit(unit, days, matrix)
        test = [t for t, role in zip(days.days, plan.roles, strict=True) if role == "test"]
        forecaster = TableForecaster(table[table["unit"] == unit]) if test else None
        out[unit] = {}
        for configuration in matrix.configurations:
            found = dict.fromkeys(COUNTS, 0)
            for truth in test:
                row = rows.get((unit, configuration.name, truth.record.site, truth.record.date))
                if row is None:
                    found = {"reason": "the configuration failed, no day was scored"}
                    break
                found["test_days"] += 1
                if row["abstain_reason"]:
                    continue
                steered, inside = _steered_in_e_prime(truth, forecaster, configuration)
                if steered != row["steered_cars"]:
                    raise EPrimeDisagrees(
                        f"the steered cars found again for E' ({steered}) disagree with the "
                        f"gate's count ({row['steered_cars']}) on day {truth.record.date} at site "
                        f"{truth.record.site} of unit {unit}, configuration "
                        f"{configuration.name}; the run is stopped and no result is written"
                    )
                found["certified_test_days"] += 1
                found["steered_cars_on_certified_test_days"] += steered
                found["steered_cars_in_e_prime_i"] += inside
                found["days_in_e_prime"] += inside == steered
            if "reason" not in found:
                found["share_of_test_days_in_e_prime"] = _share(
                    found["days_in_e_prime"], found["test_days"]
                )
                found["share_of_certified_test_days_in_e_prime"] = _share(
                    found["days_in_e_prime"], found["certified_test_days"]
                )
                found["share_of_steered_cars_in_e_prime_i"] = _share(
                    found["steered_cars_in_e_prime_i"],
                    found["steered_cars_on_certified_test_days"],
                )
            out[unit][configuration.name] = found
    return out


def evaluate(
    name: str,
    model,
    matrix,
    units: dict,
    fit_frames: dict,
    *,
    workers: int,
    fit_units: tuple[str, ...] | None = None,
    look_ahead_allowed: bool = False,
    selection: bool = True,
    loss_records: list | None = None,
    intervals: tuple[int, int] | None = None,
    counted_units: dict | None = None,
    held_out: bool = False,
):
    """Steps 0 and 2 to 4 of the module text: (results, forecast table, gate document, the gate's
    day rows, timing).
    `loss_records` and `intervals` are passed to `gate_run.run_gate` (the per-day losses of
    `day_losses`, with the resamples' pairs of `day_resample` when `intervals` is given).
    The step 0 checks run here, before anything is fitted, so no caller can skip them; `Refused`
    is raised when they fail. `name` is the model's module name. The model's `fit_units` is read
    once (here, unless the caller passes its own reading); every unit in `fit_frames` must be in
    that reading and be a training unit, or the model is refused. `look_ahead_allowed` is passed
    to `check_model` (only `main` passes true). `selection` is passed to `gate_run.run_gate`
    (`main` passes false on the sealed route: no kappa selection per unit). `held_out` (`main`
    passes true on the sealed route): a configuration that raised an error on a unit is written
    for that unit as `failed: <error>` (`gate_run.assemble`) and shown so in the E' figures."""
    if fit_units is None:
        fit_units = tuple(model.fit_units)
    outcome_reader = check_model(name, model, fit_units, matrix.gate_units, look_ahead_allowed)
    extra = model_inputs.extra_of(model, look_ahead_allowed)
    stray = [u for u in fit_frames if u not in fit_units or u not in TRAIN_UNITS]
    if stray:
        raise Refused(
            f"model {name} was given tables of {stray}, which are not training units among its "
            f"fit_units {list(fit_units)}; a model is fitted on those units only"
        )
    started = time.perf_counter()
    frames = list(fit_frames.values())
    _call("fit", model.fit, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
    fitted = time.perf_counter()
    frame = forecast_frame(outcome_reader, units, matrix.gate_units, extra)
    kappas = kappas_of(matrix)
    levels = list(driver_risk.matrix_levels(matrix.configurations))
    table = check_table(_call("predict", model.predict, frame, levels))
    predicted = time.perf_counter()
    scores = {name: learn_eval.unit_scores(units[name], table, kappas) for name in units}
    (document, day_rows, _), timing = gate_run.run_gate(
        matrix,
        units,
        workers=workers,
        only=None,
        trial=False,
        factory=TableFactory(table),
        selection=selection,
        loss_records=loss_records,
        intervals=intervals,
        counted_units=counted_units,
        held_out=held_out,
    )
    e_prime = e_prime_figures(matrix, units, table, day_rows)
    if held_out:  # a configuration that raised an error on a unit: `failed: <error>`
        for name, found in document["configurations"].items():
            for unit, entry in found["units"].items():
                if "failed" in entry:
                    e_prime[unit][name] = {"reason": entry["reason"]}
    results = {
        "model": model.name,
        "fit_units": list(fit_units),
        "units": {
            name: {
                "counting_days": document["units"][name]["counting_days"],
                "calibration_days_used": document["units"][name]["calibration_days_used"],
                "test_days": document["units"][name]["test_days"],
                "status": document["units"][name]["status"],
                "kappas": scores[name],
                "configurations": learn_eval.offers(document, matrix, name),
                "e_prime": e_prime[name],
            }
            for name in matrix.gate_units
        },
        "e_prime_definition": E_PRIME_DEFINITION,
        "calibration_rule": document["calibration_rule"],
        "t5": document["t5"],
        "timeouts": document["timeouts"],
        "failed_configurations": document["failed_configurations"],
    }
    timing = {
        "fit_seconds": fitted - started,
        "predict_seconds": predicted - fitted,
        "gate": timing,
        "total_seconds": time.perf_counter() - started,
    }
    return learn_eval.finite(results), table, document, day_rows, timing


def summary_lines(results: dict) -> list[str]:
    setting = results.get("inputs", {}).get("setting", "main")
    lines = [
        *sealed_route.summary_head(results),
        *model_inputs.summary_head(results),
        *rotation_whatif.summary_head(results),
        *after_sealed_run.summary_head(results),
        f"Model {results['model']}: {gate_summary.days_word(results)} scores and calibrated "
        f"offers (test days), {setting} truth setting.",
        gate_summary.calibration_rule_text(results["calibration_rule"]),
    ]
    if "e_prime_definition" in results:
        lines.append(results["e_prime_definition"])
    for unit, entry in results["units"].items():
        if entry["status"] == "calibratable":
            lines.append(
                f"{unit} ({entry['counting_days']} counting days, split into "
                f"{entry['calibration_days_used']} calibration days and {entry['test_days']} "
                "test days):"
            )
        else:
            lines.append(f"{unit} ({entry['status']}, {entry['counting_days']} counting days):")
        for kappa, score in entry["kappas"].items():
            loss = score["pinball_loss"]["departure_hours"]["mean"]
            index = score["concordance_index"]["value"]
            lines.append(
                f"  kappa {kappa}: departure pinball loss {_text(loss)} h, concordance index "
                f"{_text(index)}, {score['cars_with_departure_forecast']} of {score['cars']} cars "
                "with a departure forecast"
            )
        for name, offer in entry["configurations"].items():
            risk = (
                ""
                if offer.get("driver_risk") is None
                else f" [driver-risk rows at epsilon {offer['driver_risk']}]"
            )
            lines += _offer_lines(f"  {name}{risk}:", offer["rungs"])
            lines += _offer_lines("    oracle partner:", offer["oracle_partner"]["rungs"])
        lines += _e_prime_lines(entry.get("e_prime", {}))
    if results["failed_configurations"]:
        lines.append("FAILED CONFIGURATIONS: " + "; ".join(results["failed_configurations"]))
    return lines


def _text(value) -> str:
    return "none" if value is None else f"{value:.4f}"


def _percent(value) -> str:
    return "none" if value is None else f"{100 * value:.1f}%"


def _e_prime_lines(figures: dict) -> list[str]:
    """One line per configuration: the days in E' of the unit (module text, E1 to E3)."""
    lines = []
    for name, found in figures.items():
        if "reason" in found:
            lines.append(f"  {name}: no scored day, no share of days in E' ({found['reason']})")
        elif found["test_days"] == 0:
            lines.append(f"  {name}: no counting test day, no share of days in E'")
        else:
            lines.append(
                f"  {name}: {found['days_in_e_prime']} of {found['test_days']} counting test "
                f"days in E' ({_percent(found['share_of_test_days_in_e_prime'])}), "
                f"{found['days_in_e_prime']} of {found['certified_test_days']} certified test "
                f"days in E' ({_percent(found['share_of_certified_test_days_in_e_prime'])}); "
                f"{found['steered_cars_in_e_prime_i']} of "
                f"{found['steered_cars_on_certified_test_days']} steered cars in E'_i "
                f"({_percent(found['share_of_steered_cars_in_e_prime_i'])})"
            )
    return lines


def _tier_b(rungs: dict) -> dict | None:
    """The rungs as scored for tier B, in the shape `_offer` reads, or None when there are none."""
    if "reason" in rungs or any("tier_b" not in rungs[r] for r in learn_eval.OFFER_RUNGS):
        return None
    return {r: rungs[r]["tier_b"] for r in learn_eval.OFFER_RUNGS}


def _stay(rungs: dict) -> str:
    found = [rungs[r]["stay_to_bound"] for r in learn_eval.OFFER_RUNGS]
    return "; cars staying to their bound: " + ", ".join(
        f"{rung} {f['cars_checked']} checked, {f['cars_short']} short"
        for rung, f in zip(learn_eval.OFFER_RUNGS, found, strict=True)
    )


def _offer_lines(head: str, rungs: dict) -> list[str]:
    """The offer of a configuration (or its oracle partner) under each tier, one line per tier,
    each labelled and none combined with another."""
    lines = [f"{head} {TIER_LABELS['A']}: {_offer(rungs)}"]
    tier_b = _tier_b(rungs)
    if tier_b is not None:
        lines.append(f"{' ' * len(head)} {TIER_LABELS['B']}: {_offer(tier_b)}{_stay(tier_b)}")
    return lines


def _offer(rungs: dict) -> str:
    if "reason" in rungs:
        return f"no offer ({rungs['reason']})"
    parts = []
    for rung in learn_eval.OFFER_RUNGS:
        test, scale = rungs[rung]["test"], rungs[rung]["beta_hat"]
        offering = f"{test['offering_days']} of {test['days']} days have a box"
        if scale > 0:
            detail = f"scale {scale:.3g}, {offering}, {test['failing_days']} fail"
        else:
            detail = f"scale 0, {rungs[rung]['reason']}; {offering}, no offer made"
        parts.append(
            f"{rung} {_text(test['mean_offer_kwh_per_day'])} kWh/day over all test days ({detail})"
        )
    return ", ".join(parts)


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None, root: Path = TOP) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--held-out-tables", type=Path, default=None)
    parser.add_argument("--held-out-truth-run", type=Path, default=None)
    parser.add_argument("--interval-seed", type=int, default=None)
    parser.add_argument("--interval-resamples", type=int, default=None)
    parser.add_argument(after_sealed_run.FLAG, dest="after_sealed", action="store_true")
    args = parser.parse_args(argv)
    try:
        gate_run.argument_paths(args)
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        intervals = gate_run.interval_option(args.interval_seed, args.interval_resamples)
        gate_run.check_files([args.matrix])
        matrix, matrix_sha256 = read(args.matrix)
    except (MatrixError, CouldNotRun) as problem:
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
    refused = after_sealed_run.refuse_held_out(args.after_sealed, route.held_out)
    if refused is not None:
        print(f"COULD NOT RUN: {refused}")
        return 2
    look_ahead = not route.held_out  # reading R3
    try:
        model = load_model(args.model, matrix)
    except MatrixError as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    fit_units = tuple(model.fit_units)
    try:
        reader = check_model(
            args.model, model, fit_units, matrix.gate_units, look_ahead_allowed=look_ahead
        )
    except FitOnScored as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except Refused as problem:
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
            inputs, table_bytes = gate_run.read_inputs(args.truth_run, args.tables, needed)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        loaded = {
            name: parse_unit_days(
                table_bytes,
                name,
                rho=matrix.rho,
                no_fit=matrix.no_fit,
                small_fleet=matrix.small_fleet,
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
    loss_records: list = []
    try:
        results, table, document, day_rows, timing = evaluate(
            args.model,
            model,
            matrix,
            units,
            fit_frames,
            workers=args.workers,
            fit_units=fit_units,
            look_ahead_allowed=look_ahead,
            selection=not route.held_out,
            loss_records=loss_records,
            intervals=intervals,
            counted_units=sealed_route.abstained_gate_data(held_inputs),
            held_out=route.held_out,
        )
    except Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except EPrimeDisagrees as problem:
        print(f"FAILED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: the model could not be scored: {problem}")
        return 1
    rotation_whatif.mark(results, matrix, fit_units)
    after_sealed_run.mark(results, args.after_sealed)
    model_inputs.mark(results, model)
    results["inputs"] = inputs
    sealed_route.mark(results, route, held_inputs, fit_units)
    results["matrix"] = {"file": args.matrix.name, "sha256": matrix_sha256}
    text = "\n".join(summary_lines(results)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        write_table(table, folder / TABLE_NAME)
        results["forecast_table"] = {
            "file": TABLE_NAME,
            "rows": len(table),
            "sha256": hashlib.sha256((folder / TABLE_NAME).read_bytes()).hexdigest(),
        }
        rotation_whatif.mark(document, matrix, fit_units)
        after_sealed_run.mark(document, args.after_sealed)
        model_inputs.mark(document, model)
        document["inputs"] = inputs
        sealed_route.mark(document, route, held_inputs, fit_units)
        document["matrix"] = results["matrix"]
        (folder / GATE_NAME).write_text(_json(document))
        (folder / DAYS_NAME).write_text(gate_run.day_table(day_rows, document))
        (folder / RESULTS_NAME).write_text(_json(results))
        (folder / TIMING_NAME).write_text(_json(timing))
        (folder / DAY_LOSSES_NAME).write_text(day_losses.text(loss_records))
        after_sealed_run.write_note(folder, args.after_sealed)
    # On the sealed route a configuration that raised an error on a unit is written as `failed:
    # <error>` and counts as not passing, and the step still exits 0: the sealed-run check (WP14)
    # reads the failed configurations of every learn step and fails naming each.
    return 1 if results["failed_configurations"] and not route.held_out else 0


if __name__ == "__main__":
    sys.exit(main())
