"""The run matrix of the WP6 gate run (plan section 13, WP6): reading and checking its file.

The matrix file (JSON) lists the named configurations the gate run executes and the values every
configuration shares. Every field is required; an unknown field, a missing field, a value outside
its allowed set and a repeated configuration name are refused with `MatrixError`. No field has a
default in code: what the run does is written in the file.

Shared fields (the top level):

- `about`: free text, not read.
- `gate_units`: the units the gate is decided on (plan 13 WP6: NREL before and from May 2019, the
  pooled Norwegian development sites), and `required_units`: how many must pass (the plan's 2).
- `n_calibration`: calibration days per unit (plan section 6: 100), and `calibration_half_below`:
  the unit size under which half of the unit's counting days (rounded down) calibrate instead
  (plan section 6: 200), or null for no rule (`n_calibration` days for every unit). The count of
  one unit is `day_split.calibration_count`. A unit whose count leaves no calibration day or no
  test day (`day_split.is_split`) is not split and is reported as "not calibratable"; a unit that
  is split is reported with its counts, and the calibration may still abstain on it (plan T6).
- `split_seed`: the seed of the one split per unit (`day_split`), used for every configuration.
- `rho`: the unprotected share of the room (plan section 2; 0 is the main value).
- `two_way_budget_kwh`: the budget B of the two-way rung (plan 13 WP6: B = 5).
- `audited_paths`: the name of the audited-path rule; only `AUDITED_PATHS_RULE` (`audit_paths`) is
  built.
- `no_fit`, `small_fleet`: the readings of `days_from_truth`; `reference`, `min_cell`: those of
  `forecast_empirical`.
- `shape_tolerance`: the tolerance of the `largest-smallest-width` shape rule.
- `zero_value_kwh`: a certified value at or below this offers nothing (the day abstains).
- `t5_tolerance_kwh`: energies within this are equal when T5(b) is checked.
- `gate_rung`: the rung whose calibrated offer the gate uses (the "re-solve offer").
- `report`: the readings that the summary is written for: `ceilings`
  (a non-empty list of `CEILINGS`), `ratio_rules` (a non-empty list of `gate_stat.RATIO_RULES`)
  and `tiers` (a non-empty list of `TIERS`: "A", the driver-protected tier, whose scale is the
  smaller of the slot and driver scales, and "B", the deadline contract, whose scale comes from the
  slot loss alone; plan section 6). The tiers are reported side by side and never pooled; "A" is
  always scored, "B" is scored and reported only when listed.

Fields of one configuration (`configurations`, a non-empty list):

- `name`: unique; `family`: only used to group the summary: "main" (the candidate family, which sees
  departures from the rounded-down departure hour: the owner's decision of 12 November 2026),
  "beside" (the same candidate under the other reading of `departure_seen`, reported beside the
  main reading and never pooled with it) or "alternate" (a comparison that changes one field);
- `kappa`: one of plan section 2's levels; `departure`: "empirical" or "oracle";
- `energy_bounds`: "known" (rate and room known, only the departure forecast) or "empirical"
  (plan section 7's quantiles, `forecast_empirical` mode "empirical"). The project owner decided
  "known" on 16 Nov 2026 (plan section 7: in the main settings only the departure is forecast);
  "empirical" stays in the matrix as a comparison, the side not chosen. "cautious"
  (`forecast_empirical` mode `CAUTIOUS`) is the value of the forecast trial `forecast_energy`, a
  trial made after the sealed run: as "empirical", but the rate and room bounds at level 0.05
  whatever the configuration's level; no committed matrix of the plan uses it. "cautious-need" and
  "cautious-all" (`forecast_empirical` modes `CAUTIOUS_NEED`, `CAUTIOUS_ALL`) are the two further
  values of the second energy trial `forecast_energy_2`, made after the sealed run as well: the
  need bound alone at level 0.05, and all three bounds at level 0.05; no committed matrix of the
  plan uses them either;
- `uncertifiable_car`: what happens to a car with an empty forecast set or no reference sample:
  "leave-unsteered" (the car charges as logged, is outside the box and loses nothing) or
  "day-abstains" (the whole day offers nothing). The project owner decided "leave-unsteered" on
  18 Nov 2026 (a single car with no usable forecast is left unsteered and the offer is built from
  the rest; plan section 5); "day-abstains" stays as a comparison, the side not chosen;
- `shape_rule`: "largest-smallest-width" (`certificate.LargestSmallestWidth` at
  `shape_tolerance`) or "any-optimal" (the solver's own vertex; allowed only when named here);
- `departure_floor`: "as-planned" (the forecast departure as the plan writes it, fractional) or
  "floored" (rounded down to the whole hour, as the true sets round departures);
- `driver_loss_cars`: only "steered-only" is built (an unsteered car's protected energy does not
  enter the day's driver loss); the alternative, all cars, is not built;
- `departure_seen`: when a re-solving rung sees a car gone, passed to `day_replay.replay_day`:
  "hour-start" (the rung observes who is plugged in at the start of an hour, so a car leaving
  within hour t is seen gone only at the start of hour t + 1) or "rounded-hour" (the rung sees a
  car gone from the start of its rounded-down departure hour, as the hourly truth model places the
  departure);
- `oracle_partner`: for an empirical-departure configuration, the name of the configuration whose
  offer is the gate's "oracle-departure offer"; it must have departure "oracle" and agree in every
  other field but the name, family and kappa when its energy bounds are known (kappa then changes
  nothing). An oracle configuration has `oracle_partner` null;
- `driver_risk`: null (the forecast set without driver-risk rows) or the dial epsilon of plan
  section 2, one of `driver_risk.DIALS` (`driver_risk` module: a cap on the expected shortfall of
  needed energy over the lower kappa levels). An oracle partner carries the same value; with the
  oracle departure the rows hold on the whole set, so they change no value.

Roles of the gate units. The plan scores development and held-out units; a training unit is only
fitted on. A matrix naming a training unit (`truth_build.TRAIN_UNITS`) among its `gate_units` is
refused, unless it carries the one optional top-level field `rotation_whatif: true` (absent means
refused; the committed matrices do not carry it). That field makes the run a what-if, not a run of
the plan: the model must then be fitted on the training units that are not gate units
(`models.lgbm_tail_rotation`; `learn_run.check_model` refuses any model fitted on a gate unit), and
every output of the run says so (`rotation_whatif`). A matrix that sets the field without naming a
training unit among its gate units is refused too, so the label is never on a run it does not
describe.

A matrix of the run from the tag or of its rehearsal (`sealed_matrix`, which writes them) carries
a second optional top-level field, `sealed_choice`: the record of where its kappa and dials come
from (the model, the kappa-choice file and the reading of the joint rule). No run reads it; it is
checked here only to be an object whose `reading` is one of `SEALED_READINGS` (R2, one kappa per
model and the dial per site type: the owner's decision of 18 November 2026), and its whole
content by `sealed_matrix.check`.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from fleetreserve.driver_risk import DIALS
from fleetreserve.forecast_empirical import (
    CAUTIOUS,
    CAUTIOUS_ALL,
    CAUTIOUS_NEED,
    KAPPAS,
    REFERENCES,
)
from fleetreserve.gate_stat import RATIO_RULES
from fleetreserve.rungs import DEPARTURE_SEEN, RUNGS
from fleetreserve.truth_build import TRAIN_UNITS

AUDITED_PATHS_RULE = "three fixed and five seeded vertex paths per day (audit_paths)"
CEILINGS = ("hindsight-free-decision-hour", "best-causal")
TIERS = ("A", "B")
TIER_LABELS = {"A": "tier A (driver-protected)", "B": "tier B (deadline contract)"}
DEPARTURES = ("empirical", "oracle")
ENERGY_BOUNDS = ("known", "empirical", CAUTIOUS)
# The values a matrix file may name: `ENERGY_BOUNDS` (the first energy trial's list, unchanged) and
# the two values of the second energy trial `forecast_energy_2`.
MATRIX_ENERGY_BOUNDS = (*ENERGY_BOUNDS, CAUTIOUS_NEED, CAUTIOUS_ALL)
UNCERTIFIABLE = ("leave-unsteered", "day-abstains")
SHAPE_RULES = ("largest-smallest-width", "any-optimal")
DEPARTURE_FLOORS = ("as-planned", "floored")
DRIVER_LOSS_CARS = ("steered-only",)
FAMILIES = ("main", "beside", "alternate")
NO_FIT = ("cap", "leave-out")
SMALL_FLEET = ("keep", "drop")

TOP_FIELDS = (
    "about",
    "gate_units",
    "required_units",
    "n_calibration",
    "calibration_half_below",
    "split_seed",
    "rho",
    "two_way_budget_kwh",
    "audited_paths",
    "no_fit",
    "small_fleet",
    "reference",
    "min_cell",
    "shape_tolerance",
    "zero_value_kwh",
    "t5_tolerance_kwh",
    "gate_rung",
    "report",
    "configurations",
)
OPTIONAL_TOP_FIELDS = ("rotation_whatif", "sealed_choice")
SEALED_READINGS = (
    "R2",
)  # the joint rule's reading a sealed matrix records (decision of the owner)
REPORT_FIELDS = ("ceilings", "ratio_rules", "tiers")
CONFIGURATION_FIELDS = (
    "name",
    "family",
    "kappa",
    "departure",
    "energy_bounds",
    "uncertifiable_car",
    "shape_rule",
    "departure_floor",
    "driver_loss_cars",
    "departure_seen",
    "oracle_partner",
    "driver_risk",
)


class MatrixError(ValueError):
    """The matrix file cannot be used; the message says why."""


@dataclass(frozen=True)
class Configuration:
    name: str
    family: str
    kappa: float
    departure: str
    energy_bounds: str
    uncertifiable_car: str
    shape_rule: str
    departure_floor: str
    driver_loss_cars: str
    departure_seen: str
    oracle_partner: str | None
    driver_risk: float | None

    def work_key(self) -> tuple:
        """What decides the configuration's results. The forecast departure of an oracle
        configuration is a whole hour, so `departure_floor` changes nothing there, and with known
        energy bounds and the oracle departure `kappa` changes nothing either; configurations with
        equal keys are computed once."""
        oracle = self.departure == "oracle"
        kappa = None if oracle and self.energy_bounds == "known" else self.kappa
        floor = None if oracle else self.departure_floor
        return (
            kappa,
            self.departure,
            self.energy_bounds,
            self.uncertifiable_car,
            self.shape_rule,
            floor,
            self.driver_loss_cars,
            self.departure_seen,
            self.driver_risk,
        )


@dataclass(frozen=True)
class Matrix:
    gate_units: tuple[str, ...]
    required_units: int
    n_calibration: int
    calibration_half_below: int | None
    split_seed: int
    rho: float
    two_way_budget_kwh: float
    audited_paths: str
    no_fit: str
    small_fleet: str
    reference: str
    min_cell: int
    shape_tolerance: float
    zero_value_kwh: float
    t5_tolerance_kwh: float
    gate_rung: str
    ceilings: tuple[str, ...]
    ratio_rules: tuple[str, ...]
    tiers: tuple[str, ...]
    configurations: tuple[Configuration, ...]
    rotation_whatif: bool = False

    def configuration(self, name: str) -> Configuration:
        for c in self.configurations:
            if c.name == name:
                return c
        raise MatrixError(f"no configuration named {name!r}")


def _fields(entry, expected, where: str, optional=()) -> dict:
    if not isinstance(entry, dict):
        raise MatrixError(f"{where} must be a JSON object")
    unknown = sorted(set(entry) - set(expected) - set(optional))
    missing = [f for f in expected if f not in entry]
    if unknown:
        raise MatrixError(f"{where}: unknown fields {unknown}")
    if missing:
        raise MatrixError(f"{where}: missing fields {missing}")
    return entry


def _choice(value, allowed, where: str):
    if isinstance(value, bool) or value not in allowed:
        raise MatrixError(f"{where} must be one of {list(allowed)}, got {value!r}")
    return value


def _number(value, where: str, *, low=None, below=None, positive=False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MatrixError(f"{where} must be a number, got {value!r}")
    number = float(value)
    if (low is not None and number < low) or (below is not None and number >= below):
        raise MatrixError(f"{where} is out of range: {value!r}")
    if positive and not number > 0:
        raise MatrixError(f"{where} must be positive, got {value!r}")
    return number


def _whole(value, where: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise MatrixError(f"{where} must be a whole number of at least {low}, got {value!r}")
    return value


def _optional_whole(value, where: str, low: int) -> int | None:
    """null (no rule) or a whole number of at least `low`."""
    return None if value is None else _whole(value, where, low)


def _subset(value, allowed, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MatrixError(f"{where} must be a non-empty list")
    for item in value:
        _choice(item, allowed, where)
    if len(set(value)) != len(value):
        raise MatrixError(f"{where} lists a value twice")
    return tuple(value)


def _configuration(entry, k: int) -> Configuration:
    where = f"configuration {k + 1}"
    entry = _fields(entry, CONFIGURATION_FIELDS, where)
    name = entry["name"]
    if not isinstance(name, str) or not name or not all(c.isalnum() or c in "-_." for c in name):
        raise MatrixError(f"{where}: name must be letters, digits, '-', '_' or '.', got {name!r}")
    where = f"configuration {name!r}"
    kappa = entry["kappa"]
    if isinstance(kappa, bool) or kappa not in KAPPAS:
        raise MatrixError(f"{where}: kappa must be one of {list(KAPPAS)}, got {kappa!r}")
    partner = entry["oracle_partner"]
    if partner is not None and not isinstance(partner, str):
        raise MatrixError(f"{where}: oracle_partner must be a name or null")
    risk = entry["driver_risk"]
    if risk is not None and (isinstance(risk, bool) or risk not in DIALS):
        raise MatrixError(
            f"{where}: driver_risk must be null or one of {list(DIALS)}, got {risk!r}"
        )
    return Configuration(
        name=name,
        family=_choice(entry["family"], FAMILIES, f"{where}: family"),
        kappa=float(kappa),
        departure=_choice(entry["departure"], DEPARTURES, f"{where}: departure"),
        energy_bounds=_choice(
            entry["energy_bounds"], MATRIX_ENERGY_BOUNDS, f"{where}: energy_bounds"
        ),
        uncertifiable_car=_choice(
            entry["uncertifiable_car"], UNCERTIFIABLE, f"{where}: uncertifiable_car"
        ),
        shape_rule=_choice(entry["shape_rule"], SHAPE_RULES, f"{where}: shape_rule"),
        departure_floor=_choice(
            entry["departure_floor"], DEPARTURE_FLOORS, f"{where}: departure_floor"
        ),
        driver_loss_cars=_choice(
            entry["driver_loss_cars"], DRIVER_LOSS_CARS, f"{where}: driver_loss_cars"
        ),
        departure_seen=_choice(entry["departure_seen"], DEPARTURE_SEEN, f"{where}: departure_seen"),
        oracle_partner=partner,
        driver_risk=None if risk is None else float(risk),
    )


def _check_partners(configurations: tuple[Configuration, ...]) -> None:
    by_name = {c.name: c for c in configurations}
    for c in configurations:
        if c.departure == "oracle":
            if c.oracle_partner is not None:
                raise MatrixError(f"configuration {c.name!r}: an oracle has no oracle_partner")
            continue
        if c.oracle_partner is None:
            raise MatrixError(f"configuration {c.name!r}: oracle_partner is required")
        partner = by_name.get(c.oracle_partner)
        if partner is None:
            raise MatrixError(f"configuration {c.name!r}: no configuration {c.oracle_partner!r}")
        if partner.departure != "oracle":
            raise MatrixError(f"configuration {c.name!r}: its partner's departure is not oracle")
        same = ["energy_bounds", "uncertifiable_car", "shape_rule", "departure_floor"]
        same += ["driver_loss_cars", "departure_seen", "driver_risk"]
        if c.energy_bounds != "known":
            same.append("kappa")
        differ = [f for f in same if getattr(c, f) != getattr(partner, f)]
        if differ:
            raise MatrixError(f"configuration {c.name!r}: its oracle partner differs in {differ}")


def parse(document) -> Matrix:
    """The matrix from a parsed JSON document; raises `MatrixError`."""
    top = _fields(document, TOP_FIELDS, "the matrix", OPTIONAL_TOP_FIELDS)
    report = _fields(top["report"], REPORT_FIELDS, "report")
    units = top["gate_units"]
    if (
        not isinstance(units, list)
        or not units
        or not all(isinstance(u, str) and u for u in units)
        or len(set(units)) != len(units)
    ):
        raise MatrixError("gate_units must be a non-empty list of distinct unit names")
    required = _whole(top["required_units"], "required_units", 1)
    if required > len(units):
        raise MatrixError("required_units is above the number of gate units")
    rotation = top.get("rotation_whatif", False)
    if not isinstance(rotation, bool):
        raise MatrixError(f"rotation_whatif must be true or false, got {rotation!r}")
    scored_in_training = [u for u in units if u in TRAIN_UNITS]
    if scored_in_training and not rotation:
        raise MatrixError(
            f"gate_units {scored_in_training} are training units, which are fitted on and not "
            "scored in the plan; only a rotation what-if (rotation_whatif: true) scores them"
        )
    if rotation and not scored_in_training:
        raise MatrixError(
            "rotation_whatif is set but no gate unit is a training unit; the field is for "
            f"scoring a training unit ({', '.join(TRAIN_UNITS)}) and nothing else"
        )
    choice = top.get("sealed_choice")
    if "sealed_choice" in top and (
        not isinstance(choice, dict) or choice.get("reading") not in SEALED_READINGS
    ):
        raise MatrixError(
            "sealed_choice must be an object whose reading is one of "
            f"{', '.join(SEALED_READINGS)}, got {choice!r}"
        )
    if top["audited_paths"] != AUDITED_PATHS_RULE:
        raise MatrixError(f"audited_paths must be {AUDITED_PATHS_RULE!r}")
    seed = top["split_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MatrixError("split_seed must be a whole number")
    if not isinstance(top["about"], str):
        raise MatrixError("about must be text")
    entries = top["configurations"]
    if not isinstance(entries, list) or not entries:
        raise MatrixError("configurations must be a non-empty list")
    configurations = tuple(_configuration(entry, k) for k, entry in enumerate(entries))
    names = [c.name for c in configurations]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise MatrixError(f"configuration names appear twice: {repeated}")
    _check_partners(configurations)
    return Matrix(
        gate_units=tuple(units),
        required_units=required,
        n_calibration=_whole(top["n_calibration"], "n_calibration", 1),
        calibration_half_below=_optional_whole(
            top["calibration_half_below"], "calibration_half_below", 1
        ),
        split_seed=seed,
        rho=_number(top["rho"], "rho", low=0.0, below=1.0),
        two_way_budget_kwh=_number(top["two_way_budget_kwh"], "two_way_budget_kwh", low=0.0),
        audited_paths=top["audited_paths"],
        no_fit=_choice(top["no_fit"], NO_FIT, "no_fit"),
        small_fleet=_choice(top["small_fleet"], SMALL_FLEET, "small_fleet"),
        reference=_choice(top["reference"], REFERENCES, "reference"),
        min_cell=_whole(top["min_cell"], "min_cell", 1),
        shape_tolerance=_number(top["shape_tolerance"], "shape_tolerance", low=0.0, below=1.0),
        zero_value_kwh=_number(top["zero_value_kwh"], "zero_value_kwh", low=0.0),
        t5_tolerance_kwh=_number(top["t5_tolerance_kwh"], "t5_tolerance_kwh", low=0.0),
        gate_rung=_choice(top["gate_rung"], RUNGS[1:], "gate_rung"),
        ceilings=_subset(report["ceilings"], CEILINGS, "report.ceilings"),
        ratio_rules=_subset(report["ratio_rules"], RATIO_RULES, "report.ratio_rules"),
        tiers=_subset(report["tiers"], TIERS, "report.tiers"),
        configurations=configurations,
        rotation_whatif=rotation,
    )


def load(path: Path) -> Matrix:
    """The matrix in the file `path`; raises `MatrixError` (also for a missing or unreadable
    file)."""
    return read(path)[0]


def read(path: Path) -> tuple[Matrix, str]:
    """The matrix in the file `path` and the SHA-256 of the bytes it was parsed from: the file is
    read once, so the hash a run records is that of the matrix it ran. Raises `MatrixError` as
    `load` does."""
    try:
        data = Path(path).read_bytes()
        document = json.loads(data.decode())
    except (OSError, ValueError) as problem:
        raise MatrixError(f"the matrix file cannot be read: {problem}") from problem
    return parse(document), hashlib.sha256(data).hexdigest()
