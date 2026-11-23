"""The WP8 pass-mark item, read from the recorded learn runs (plan section 13, WP8: "development
pinball loss, concordance index and offer (WP6 calibration) logged per model").

Run as `python tools/learn_logged_check.py [--top <folder>]`.

The models are `MODELS`: the forecast models of plan section 7, in its order, each with its
registry name, the module `fleetreserve.models.<name>` that `fleetreserve.learn_run --model
<name>` loads. `HAZARD_MAIN` names the hazard model the plan calls the main model; which hazard
variant that is is not decided here, so both `models.hazard_tuned` and
`models.hazard_tuned_inputs` (that model with the plan's section 7 inputs as extra columns) are
listed, the first as `HAZARD_MAIN`. The four ablations are those of `models.hazard_tuned` (each
is that model with one part switched off) and do not follow `HAZARD_MAIN`: if the owner makes
another variant the main model, the four ablation entries must be revisited. Pooled LightGBM has
two entries: `models.lgbm`, and after it `models.lgbm_tail`, which plan section 13 (WP6) calls
"Section 7's pooled LightGBM with the lower tail of the user's past departures as further
inputs"; the WP6 gate is read with it, so the item must not be able to pass with the gate's own
model unlogged. Left out: Chronos-2 and TabPFN v2, cut (plan section 13, the owner's decision of
14 November 2026); `models.hazard_tuned_inputs_lookahead`, a what-if that reads the day's own
weather of the hours after the decision time, a labelled leak and never a model of the plan (plan
section 7, the owner's decision of 15 November 2026); and the day-size quantile net of section 7,
a sizing rule that makes no departure forecast, so it has no departure pinball loss or
concordance index to log (whether it is built is open). The earlier GBM and population models
are `models.earlier_gbm` and `models.earlier_population` (the constants `EARLIER_GBM` and
`EARLIER_POPULATION`). An entry with no registry name, with a name that is not a Python name
(`learn_run.load_model` refuses such a name too), or whose module does not exist is reported as
"no model".

For each model with a module, the run read is the newest folder in `runs.noindex/` named
`<UTC start>-wp8-<name>-gate-matrix` (`run_evidence.newest_run`; an older run is never used in its
place), recorded as
    python -m fleetreserve.run_record --name wp8-<name>-gate-matrix --long --max-seconds <n> --
        uv run --frozen --group learn python -m fleetreserve.learn_run --model <name>
        --matrix gate/matrix.json --workers <n>
That run decides nothing (could not run) unless:
- its results `learn.json` and its gate-format results `learn-gate.json` are verified
  (`run_evidence`: the run ended with exit 0, was neither stopped nor timed out, and the files
  are those it wrote) and it was recorded at the commit checked out now or is read from an
  earlier commit by the rule of `run_evidence` (module text there; else it is stale); every line
  this tool writes that names a run read from an earlier commit carries `Evidence.note` after the
  run's name, whatever the line then says (not written by this tool, so without it: Python's own
  text of an unexpected error, after "the check stopped on", which can name a file in a run's
  folder);
- its recorded command runs `python -m fleetreserve.learn_run` with `--model <name>`;
- it read the committed matrix: the SHA-256 `learn.json` records for its matrix equals that of
  `gate/matrix.json` at the checked-out commit;
- `learn-gate.json` says it is neither a trial nor a partial run, and `learn.json` says it read
  the main truth setting (the held setting is reported apart, plan section 3).

The three values, read from `learn.json` as `learn_run` writes it (`learn_eval`), are required on
every gate unit of the committed matrix, or on the units an entry names: declared departure on
the NREL units only (plan section 7: "declared departure (ACN, NREL)"; the Norwegian sites have
no declared departure, so it forecasts no car there).
- Development pinball loss: the departure pinball loss (hours) at every kappa of the matrix, a
  finite number >= 0 (reading L1 below). The rate and energy losses are not required.
- Concordance index: at every kappa, a finite number in [0, 1].
- Offer (WP6 calibration): for every configuration of the matrix that is not an oracle partner,
  the test-day offer (kWh per test day, tier A) of each rung of `learn_eval.OFFER_RUNGS`, a finite
  number >= 0. A configuration with no calibrated number for a unit (one not calibratable) has
  no offer logged there, so its offer is missing.
A value that is absent, not a number, not finite or outside its range is missing.

Reading (its alternative in brackets):

  L1. The plan's "development pinball loss" is read as the departure pinball loss alone. Every
      listed model forecasts the departure quantile, declared departure forecasts nothing else,
      and the offer takes rate and energy as known (`learn_eval`), so the rate and energy losses
      `learn_eval` logs beside it are not required. [Alternative: also require the rate and
      energy pinball losses (`r_hat` and `e_cap` at kappa, `e_need` at 1 - kappa); declared
      departure forecasts no rate or energy, so its entry could then never be logged.]

Output: a first line with the checked-out commit and the matrix; one line per model in the list's
order: `logged:` with the run folder and, for each of the three, the number of values logged out
of the number required and their range, and for the offer how many of the logged values are 0 (a
calibration that abstains gives the test-day offer 0.0, a finite number >= 0, so it counts as
logged); `MISSING:` with the same counts and the first places a value is missing; `NO RUN:`,
`COULD NOT RUN:` or `NO MODEL:` with the reason; a last line with the counts. On each of these
lines a run read from an earlier commit has `Evidence.note` after its name, as said above.

Exit status: 0 every listed model is logged; 1 a model's run was read and a value is missing (this
wins over 2, as in `check.sh`); 2 could not run (a model with no module or no run, a run that
cannot decide, no committed matrix or one that cannot be read, no model listed). An unexpected
error while checking is could not run too, never 1 (Python's own exit status for an uncaught error
is 1, which `passmark.sh` reads as failed): within one model's check, that model's line is
`COULD NOT RUN:` with the error, and the other models are still checked; anywhere else, the last
line is `COULD NOT RUN:` with the error.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from fleetreserve.gate_config import Matrix, parse
from fleetreserve.learn_eval import OFFER_RUNGS
from fleetreserve.learn_run import GATE_NAME, RESULTS_NAME, kappas_of
from fleetreserve.run_evidence import (
    CannotCheck,
    head_commit,
    newest_run,
    read_evidence,
    with_note,
)
from fleetreserve.truth import MAIN

MATRIX = "gate/matrix.json"
LEARN_RUN = "fleetreserve.learn_run"
SHOWN = 3  # the missing places named per value

# The hazard model plan section 7 calls the main model, by registry name. Another hazard variant
# becomes the main model by changing this one line.
HAZARD_MAIN = "hazard_tuned"
# The earlier GBM and population models of plan section 7 by registry name.
EARLIER_GBM = "earlier_gbm"
EARLIER_POPULATION = "earlier_population"
# The development units with declared departures (plan section 7: "declared departure (ACN, NREL)").
NREL_UNITS = ("nrel-before-may-2019", "nrel-from-may-2019")


@dataclass(frozen=True)
class Model:
    """A forecast model of plan section 7: its name there, its registry name in
    `fleetreserve.models` (None: no module implements it) and the gate units it is scored on
    (None: every gate unit of the matrix)."""

    plan_name: str
    registry: str | None
    units: tuple[str, ...] | None = None


MODELS = (
    Model("hazard model (main)", HAZARD_MAIN),
    Model("hazard model with the plan's section 7 inputs", "hazard_tuned_inputs"),
    Model("hazard model, no set encoder", "hazard_tuned_no_set"),
    Model("hazard model, no site embedding", "hazard_tuned_no_site"),
    Model("hazard model, no day factor", "hazard_tuned_no_day"),
    Model("hazard model, calendar-only inputs", "hazard_tuned_calendar"),
    Model("Kaplan-Meier (site, arrival hour)", "km"),
    Model("Cox", "cox"),
    Model("Weibull AFT", "weibull_aft"),
    Model("pooled LightGBM", "lgbm"),
    Model(
        "pooled LightGBM with the lower tail of the user's past departures (the WP6 gate's model)",
        "lgbm_tail",
    ),
    Model("CQR-GBM", "cqr"),
    Model("the earlier GBM model", EARLIER_GBM),
    Model("the earlier population model", EARLIER_POPULATION),
    Model("declared departure (ACN, NREL)", "declared", NREL_UNITS),
)

LOGGED, MISSING, COULD_NOT_RUN = 0, 1, 2


def label(registry: str) -> str:
    """The run label of a model's recorded learn run on the gate matrix."""
    return f"wp8-{registry}-gate-matrix"


def committed_matrix(top: Path) -> tuple[str, Matrix]:
    """The SHA-256 and the parsed matrix of `gate/matrix.json` as committed at the checked-out
    commit."""
    done = subprocess.run(
        ["git", "-C", str(top), "show", f"HEAD:{MATRIX}"], capture_output=True, check=False
    )
    if done.returncode != 0:
        raise CannotCheck(f"{MATRIX} is not committed at the checked-out commit")
    try:
        matrix = parse(json.loads(done.stdout))
    except Exception as problem:  # whatever the parser raises, the matrix cannot be read
        raise CannotCheck(
            f"the committed {MATRIX} cannot be read: {type(problem).__name__}: {problem}"
        ) from problem
    return hashlib.sha256(done.stdout).hexdigest(), matrix


def command_model(command) -> str | None:
    """The `--model` of a recorded command that runs `python -m fleetreserve.learn_run`, the last
    one given as the command line reads it; None for any other command."""
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        return None
    starts = [k for k in range(1, len(command)) if command[k - 1 : k + 1] == ["-m", LEARN_RUN]]
    if not starts:
        return None
    rest = command[starts[0] + 1 :]
    found = None
    for k, part in enumerate(rest):
        if part == "--model" and k + 1 < len(rest):
            found = rest[k + 1]
        elif part.startswith("--model="):
            found = part.removeprefix("--model=")
    return found


def _at(document, *keys):
    """`document[keys[0]][keys[1]]...`, or None where a key is absent or a level is no mapping."""
    for key in keys:
        if not isinstance(document, dict) or key not in document:
            return None
        document = document[key]
    return document


@dataclass
class Tally:
    """One of the three values over its required places."""

    what: str
    suffix: str
    low: float
    high: float = math.inf
    count_zeros: bool = False  # the text says how many logged values are 0
    found: list[float] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def add(self, place: str, value) -> None:
        number = isinstance(value, int | float) and not isinstance(value, bool)
        if number and math.isfinite(value) and self.low <= value <= self.high:
            self.found.append(float(value))
        else:
            self.missing.append(place)

    @property
    def required(self) -> int:
        return len(self.found) + len(self.missing)

    def text(self) -> str:
        counted = f"{self.what} {len(self.found)} of {self.required}"
        if not self.missing:
            spread = f"{min(self.found):.4g} to {max(self.found):.4g}{self.suffix}"
            if self.count_zeros:
                spread += f"; {self.found.count(0.0)} of {len(self.found)} are 0"
            return f"{counted} ({spread})"
        more = len(self.missing) - SHOWN
        named = ", ".join(self.missing[:SHOWN]) + (f" and {more} more" if more > 0 else "")
        return f"{counted}, missing at {named}"


def tallies(results: dict, matrix: Matrix, units) -> list[Tally]:
    """The three values of `learn.json` over the required places (module text)."""
    pinball = Tally("departure pinball loss", " h", 0.0)
    concordance = Tally("concordance index", "", 0.0, 1.0)
    offer = Tally("offer", " kWh per test day", 0.0, count_zeros=True)
    # Keys as `learn_eval.unit_scores` writes them; the configurations `learn_eval.offers` scores.
    kappas = [repr(float(kappa)) for kappa in kappas_of(matrix)]
    offered = [c.name for c in matrix.configurations if c.departure != "oracle"]
    for unit in units:
        entry = _at(results, "units", unit)
        for kappa in kappas:
            place = f"{unit} kappa {kappa}"
            score = _at(entry, "kappas", kappa)
            pinball.add(place, _at(score, "pinball_loss", "departure_hours", "mean"))
            concordance.add(place, _at(score, "concordance_index", "value"))
        for name in offered:
            for rung in OFFER_RUNGS:
                found = _at(entry, "configurations", name, "rungs", rung, "test")
                offer.add(f"{unit} {name} {rung}", _at(found, "mean_offer_kwh_per_day"))
    if any(t.required == 0 for t in (pinball, concordance, offer)):
        raise CannotCheck(f"the committed {MATRIX} leaves a value with no place to check")
    return [pinball, concordance, offer]


def read_run(run: Path, head: str, model: Model, committed: str) -> tuple[dict, str]:
    """`learn.json` of the model's run and the evidence's note (`Evidence.note`, empty for a run
    at the checked-out commit), once the run is shown to decide (module text); raises
    `CannotCheck` otherwise. Every message names the run, with the note once it is read."""
    evidence = read_evidence(run, RESULTS_NAME, head)
    name = f"{run.name}{evidence.note}"
    if not evidence.verified:
        raise CannotCheck(
            f"{name} ended with exit code {evidence.manifest.get('exit_code')}, so it is not "
            "verified and cannot decide the item"
        )
    if command_model(evidence.manifest.get("command")) != model.registry:
        raise CannotCheck(
            f"the recorded command of {name} does not run {LEARN_RUN} --model {model.registry}"
        )
    try:
        gate = read_evidence(run, GATE_NAME, head).data
    except CannotCheck as problem:  # the same run, its second results file
        raise with_note(problem, evidence) from problem
    if gate.get("trial") is not False:
        raise CannotCheck(f"{GATE_NAME} of {name} does not say it is not a trial run")
    if gate.get("partial") is not False:
        raise CannotCheck(f"{GATE_NAME} of {name} does not say it is not a partial run")
    results = evidence.data
    used = _at(results, "matrix", "sha256")
    if not isinstance(used, str):
        raise CannotCheck(
            f"{RESULTS_NAME} of {name} records no SHA-256 of its matrix, so {MATRIX} differs"
        )
    if used != committed:
        raise CannotCheck(
            f"{name} ran on another matrix: SHA-256 {used[:12]}, the committed {MATRIX} has "
            f"{committed[:12]}"
        )
    setting = _at(results, "inputs", "setting")
    if setting != MAIN:
        raise CannotCheck(f"{name} read the truth setting {setting!r}, not {MAIN!r}")
    return results, evidence.note


def check_model(model: Model, top: Path, head: str, committed: str, matrix: Matrix):
    """(outcome, line) of one listed model."""
    if model.registry is None:
        return COULD_NOT_RUN, (
            f"NO MODEL: {model.plan_name}: plan section 7 names it; no module in "
            "fleetreserve.models implements it"
        )
    name = f"{model.registry} ({model.plan_name})"
    if not isinstance(model.registry, str) or not model.registry.isidentifier():
        # as `learn_run.load_model` refuses it; a dotted name would make `find_spec` import a
        # parent module that is not a package and raise
        return COULD_NOT_RUN, (
            f"NO MODEL: {name}: {model.registry!r} is not a Python name, so no module of "
            "fleetreserve.models implements it"
        )
    if importlib.util.find_spec(f"fleetreserve.models.{model.registry}") is None:
        return COULD_NOT_RUN, f"NO MODEL: {name}: no module fleetreserve.models.{model.registry}"
    units = matrix.gate_units if model.units is None else model.units
    outside = [u for u in units if u not in matrix.gate_units]
    if outside:
        return COULD_NOT_RUN, (
            f"COULD NOT RUN: {name}: its listed units {', '.join(outside)} are not gate units of "
            f"the committed {MATRIX}"
        )
    try:
        run = newest_run(top, label(model.registry))
    except CannotCheck as problem:
        return COULD_NOT_RUN, f"NO RUN: {name}: {problem}"
    try:
        results, note = read_run(run, head, model, committed)
        found = tallies(results, matrix, units)
    except CannotCheck as problem:
        return COULD_NOT_RUN, f"COULD NOT RUN: {name}: {problem}"
    only = [] if model.units is None else [f"units {', '.join(units)} only"]
    parts = "; ".join([f"{run.name}{note}", *only, *(t.text() for t in found)])
    if any(t.missing for t in found):
        return MISSING, f"MISSING: {name}: {parts}"
    return LOGGED, f"logged: {name}: {parts}"


def check_all(top: Path, models) -> int:
    head = head_commit(top)
    committed, matrix = committed_matrix(top)
    print(
        f"WP8 learn runs checked at commit {head[:12]} (a run accepted from an earlier commit says "
        f"so on its line), on the committed {MATRIX} ({committed[:12]})"
    )
    if not models:
        raise CannotCheck("no model is listed")
    outcomes = []
    for model in models:
        try:
            outcome, line = check_model(model, top, head, committed, matrix)
        except Exception as problem:  # an unexpected error (module text)
            outcome = COULD_NOT_RUN
            line = (
                f"COULD NOT RUN: {model.registry} ({model.plan_name}): the check stopped on "
                f"{type(problem).__name__}: {problem}"
            )
        outcomes.append(outcome)
        print(line)
    print(
        f"{outcomes.count(LOGGED)} of {len(outcomes)} models logged, {outcomes.count(MISSING)} "
        f"with a value missing, {outcomes.count(COULD_NOT_RUN)} could not run"
    )
    if MISSING in outcomes:
        return 1
    return 2 if COULD_NOT_RUN in outcomes else 0


def main(argv: list[str] | None = None, models=MODELS) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/learn_logged_check.py",
        description="Decide the WP8 pass-mark item from the recorded learn runs.",
    )
    parser.add_argument(
        "--top",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the repository top folder, where runs.noindex/ is (default: this repository)",
    )
    options = parser.parse_args(argv)
    try:
        return check_all(options.top, models)
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except Exception as problem:  # an unexpected error (module text)
        print(f"COULD NOT RUN: the check stopped on {type(problem).__name__}: {problem}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
