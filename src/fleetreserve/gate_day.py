"""The per-day work of the WP6 gate run (plan sections 1, 2, 5, 13 WP6).

Everything here is a pure function of its inputs (no files, no clock in the results), so it can
run in a process pool and be cached.

`day_ceilings(day)`: the two ceilings of a day on its true cars (every car of the fleet):

- "hindsight-free-decision-hour": `exact.hourly_ceiling`, b(S) - p(S), where a hindsight plan may
  choose the power of the decision hour per request path;
- "best-causal": the best real-time value with known cars, `certificate.certify` on the true rows,
  hourly product, causal structure, where the power of the decision hour is chosen once at τ. The
  solver's own vertex (`ANY_OPTIMAL`) is used: only the value is read, and every optimal vertex
  has it. Any outcome but optimal leaves the ceiling missing (a timeout is never 0).

A day without cars has both ceilings 0.

`prepare_day(truth, forecaster, configuration, settings)`: the day as one configuration offers it.

1. Each car's forecast, `forecaster.forecast_car` at the configuration's kappa, energy bounds and
   departure. With `departure_floor` "floored" the forecast departure is rounded down to the whole
   hour (the true sets round departures down, so a fractional forecast departure above the whole
   hour is never conservative); "as-planned" keeps it. When `driver_risk` is a number, each
   forecast also carries its departure scenarios (`driver_risk.departure_scenarios`): the
   forecaster's departure at each of the plan's levels up to the configuration's kappa, floored
   the same way, a level the forecaster does not give read as the presence bound.
2. Uncertifiable cars: a car with no reference sample, or whose forecast set is empty
   (`forecast_coverage.forecast_set_empty`). Under "leave-unsteered" such a car charges as logged:
   it is outside the offered box, loses nothing, and is not in the steered fleet's programmes.
   Under "day-abstains" any such car makes the whole day offer nothing.
   No-pause cars (`no_pause`: positions in the day's cars, drawn by `no_pause.drawn` for the
   no-pause sensitivity of plan section 9; none by default): each is left unsteered with the
   reason `NO_PAUSE` under either value of `uncertifiable_car`, by the route above (it charges
   as logged, is outside the box, loses nothing and is in no programme), and is not forecast. It
   is not an uncertifiable car: the "day-abstains" rule looks only at the uncertifiable cars, so a
   drawn car does not by that rule make the day abstain; a day whose every car is drawn has no
   steered car and abstains as such (step 4). A car that is drawn has the reason `NO_PAUSE`
   whatever its forecast.
3. The steered sub-day: the steered cars' true limits and forecasts. Its recovery hours are the
   whole day's, extended so that every steered forecast departure lies inside the day (as
   `forecast_empirical.ForecastDay` does).
4. The certificate on the steered cars' forecast rows (with the driver-risk rows of
   `driver_risk` when the configuration's `driver_risk` is a number): hourly product, causal
   structure, the named shape rule, the box time limit. A day abstains (offers 0, loses 0 at every
   scale) when no car is steered, the certificate is not optimal (a timeout counted as such) or
   its value is at most `zero_value_kwh`. The rows the certificate was certified on are kept as
   `PreparedDay.rows` (the comparison methods of plan section 10 that plan on the same forecast
   sets read them there); `rows` is None on a day that abstains before rows exist (no car, no
   steered car, or an uncertifiable car under "day-abstains").
5. The departure bound the rungs re-solve with, per steered car and hour 0 to n
   (`departure_bounds`). Empirical departure: hour 0 holds the forecast departure; from hour 1 the
   survival-tightened bound `forecaster.survival_bound` (the kappa-quantile of departure among
   reference cars still plugged in at the start of the hour), rounded down too under "floored";
   when the forecaster has none for an hour (no reference car of the site left), the bound of the
   hour before is kept. Oracle departure: the car's true departure hour in every hour. An oracle
   forecast already knows the departure, so a survival bound drawn from other cars would only
   loosen or tighten it by information the oracle does not need; the true departure is the
   survival-tightened bound of a car whose departure is known. The bound only sets the forecast
   sets a re-solve plans on; what a rung decides on otherwise (which cars are still plugged in,
   what they drew) is what `rungs` lets it observe at the start of the hour, in both variants.
6. The audited paths of the day (`audit_paths`, seeded from the unit and date; two site-days of
   the pooled unit on the same date get the same paths).

`evaluate(prepared, beta)`: the day replayed at scale `beta` with every rung up to the two-way rung
(`replay`: `day_replay.replay_day`, budget `settings.budget`, re-solves with the time limit for
other programmes, `departure_seen` as the configuration names it: "hour-start" or "rounded-hour",
see `rungs`) on its audited paths. Per rung: the day's slot loss and driver loss, each the largest
over the paths; per path the first failing hour and each steered car's energy at departure; the
re-solves by outcome. The driver loss counts the steered cars only (`driver_loss_cars`
"steered-only"). T5 is checked on every path (`t5_breaks`): (a) no rung's first failure is before
its predecessor's; (b) each car ends with at least its predecessor's energy, within
`settings.t5_tolerance_kwh`. An abstaining day (or `beta <= 0`) has losses 0 and checks no path.

Per rung it also carries the stay-to-bound check (`stay_check`, plan section 6, tier B): the steered
cars whose true departure is at or after the presence bound the plan was built on (and whose other
bounds hold), counted short when their energy at departure is below their protected energy by
more than `settings.t5_tolerance_kwh`. It is computed for every evaluation, as T5 is; the
calibration reads it at the tier B scale. An abstaining day checks no car.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from fleetreserve import driver_risk, exact, products, stay_check
from fleetreserve.audit_paths import audited_paths
from fleetreserve.carsets import FleetRows, day_forecast_rows, true_rows
from fleetreserve.certificate import ANY_OPTIMAL, Certificate, LargestSmallestWidth, certify
from fleetreserve.day import Day
from fleetreserve.day_replay import replay_day
from fleetreserve.days_from_truth import TruthDay
from fleetreserve.forecast_coverage import forecast_set_empty
from fleetreserve.gate_config import Configuration
from fleetreserve.lp import BOX_SECONDS, OTHER_SECONDS
from fleetreserve.rungs import RUNGS

NO_REFERENCE = "no reference sample"
EMPTY_SET = "empty forecast set"
NO_CAR = "no car in the fleet"
NO_STEERED = "no steered car"
DAY_ABSTAINS = "an uncertifiable car (day-abstains)"
ZERO_VALUE = "certified value at most the zero value"
NO_PAUSE = "cannot pause (no-pause sensitivity)"
PREDECESSOR = dict(zip(RUNGS[1:], RUNGS, strict=False))


@dataclass(frozen=True)
class DaySettings:
    """The shared values of the matrix that the per-day work needs."""

    rho: float
    budget: float
    shape_tolerance: float
    zero_value_kwh: float
    t5_tolerance_kwh: float


@dataclass(frozen=True)
class Ceilings:
    hindsight: float
    best_causal: float | None  # None when the programme did not end optimal
    best_causal_outcome: str
    seconds: float  # timing only, never compared


def day_ceilings(day: Day) -> Ceilings:
    """Both ceilings of a day on all its true cars (see the module text)."""
    if not day.cars:
        return Ceilings(0.0, 0.0, "no car", 0.0)
    hindsight = exact.hourly_ceiling(exact.day_functions(day))
    found = certify(
        true_rows(day), products.hourly(), "causal", shape_rule=ANY_OPTIMAL, time_limit=BOX_SECONDS
    )
    return Ceilings(hindsight, found.value, found.outcome, found.seconds)


def shape_rule(configuration: Configuration, settings: DaySettings):
    if configuration.shape_rule == "largest-smallest-width":
        return LargestSmallestWidth(settings.shape_tolerance)
    if configuration.shape_rule == "any-optimal":
        return ANY_OPTIMAL
    raise ValueError(f"unknown shape rule {configuration.shape_rule!r}")


@dataclass(frozen=True)
class PreparedDay:
    """One day as one configuration offers it (see the module text)."""

    unit: str
    site: str
    date: str
    n_cars: int
    unsteered: tuple[tuple[int, str], ...]  # (car of the fleet, reason)
    steered: tuple[int, ...]  # cars of the fleet in the steered sub-day, in order
    abstain_reason: str | None
    certificate_outcome: str  # the certificate's outcome, or "not solved"
    value: float  # the certified value at scale 1 (0 when the day abstains)
    day: Day | None  # the steered sub-day with forecasts
    certificate: Certificate | None
    departure_bounds: np.ndarray | None  # (steered cars, n + 1)
    paths: np.ndarray  # (8, n)
    budget: float
    t5_tolerance_kwh: float
    certificate_seconds: float  # timing only, never compared
    departure_seen: str  # passed to the replay (`rungs.DEPARTURE_SEEN`)
    rows: FleetRows | None = None  # the rows certified on (module text, step 4)

    @property
    def abstains(self) -> bool:
        return self.abstain_reason is not None


def _forecast(truth: TruthDay, forecaster, configuration: Configuration, i: int, kappa: float):
    found = forecaster.forecast_car(
        truth, i, kappa, configuration.energy_bounds, configuration.departure
    )
    if found is not None and configuration.departure_floor == "floored":
        found = replace(found, departure=float(math.floor(found.departure)))
    return found


def _scenarios(truth: TruthDay, forecaster, configuration: Configuration, i: int, found):
    """The departure scenarios of car `i` (module text, step 1) for its forecast `found`."""
    levels = driver_risk.levels_up_to(configuration.kappa)
    departures = []
    for level in levels:
        at_level = _forecast(truth, forecaster, configuration, i, level)
        departures.append(None if at_level is None else at_level.departure)
    masses = driver_risk.level_masses(levels)
    return driver_risk.departure_scenarios(found.departure, departures, masses)


def _forecasts(truth: TruthDay, forecaster, configuration: Configuration, skip=frozenset()):
    out = []
    for i in range(len(truth.day.cars)):
        if i in skip:
            out.append(None)
            continue
        found = _forecast(truth, forecaster, configuration, i, configuration.kappa)
        if found is not None and configuration.driver_risk is not None:
            found = replace(found, scenarios=_scenarios(truth, forecaster, configuration, i, found))
        out.append(found)
    return out


def _bounds(truth, forecaster, configuration, steered, forecasts, n) -> np.ndarray:
    table = np.zeros((len(steered), n + 1))
    for p, i in enumerate(steered):
        if configuration.departure == "oracle":
            table[p, :] = float(truth.day.cars[i].departure)
            continue
        table[p, 0] = forecasts[i].departure
        for t in range(1, n + 1):
            bound = forecaster.survival_bound(truth, i, t, configuration.kappa)
            if bound is None:
                table[p, t] = table[p, t - 1]
            elif configuration.departure_floor == "floored":
                table[p, t] = float(math.floor(bound))
            else:
                table[p, t] = bound
    return table


def _no_pause_cars(day: Day, no_pause) -> frozenset:
    cars = tuple(no_pause)
    for i in cars:
        if isinstance(i, bool) or not isinstance(i, int | np.integer) or not 0 <= i < len(day.cars):
            raise ValueError(f"no_pause names {i!r}, not a car of the day")
    return frozenset(int(i) for i in cars)


def prepare_day(
    truth: TruthDay,
    forecaster,
    configuration: Configuration,
    settings: DaySettings,
    no_pause=(),
) -> PreparedDay:
    """The day as `configuration` offers it (see the module text). `forecaster` provides
    `forecast_car` and `survival_bound` as `forecast_empirical.EmpiricalForecaster` does;
    `no_pause`: the positions of the day's cars that cannot pause (module text, step 2)."""
    day = truth.day
    n = day.n_service
    paths = audited_paths(day.unit, day.date, n)
    skip = _no_pause_cars(day, no_pause)
    forecasts = _forecasts(truth, forecaster, configuration, skip)
    known = [f for f in forecasts if f is not None]
    latest = max([0.0, *(f.departure for f in known)])
    n_recovery = max(day.n_recovery, math.ceil(latest) - 1 - n)
    unsteered = []
    for i, found in enumerate(forecasts):
        if i in skip:
            unsteered.append((i, NO_PAUSE))
        elif found is None:
            unsteered.append((i, NO_REFERENCE))
        elif forecast_set_empty(found, n, n_recovery):
            unsteered.append((i, EMPTY_SET))
    out_of_box = {i for i, _ in unsteered}
    steered = tuple(i for i in range(len(day.cars)) if i not in out_of_box)
    uncertifiable = [i for i, why in unsteered if why != NO_PAUSE]
    reason = None
    if not day.cars:
        reason = NO_CAR
    elif uncertifiable and configuration.uncertifiable_car == "day-abstains":
        reason, steered = DAY_ABSTAINS, ()
    elif not steered:
        reason = NO_STEERED

    def abstaining(why, outcome="not solved", sub=None, certificate=None, seconds=0.0, rows=None):
        return PreparedDay(
            day.unit,
            truth.record.site,
            truth.record.date,
            len(day.cars),
            tuple(unsteered),
            steered,
            why,
            outcome,
            0.0,
            sub,
            certificate,
            None,
            paths,
            settings.budget,
            settings.t5_tolerance_kwh,
            seconds,
            configuration.departure_seen,
            rows,
        )

    if reason is not None:
        return abstaining(reason)
    sub = Day(
        day.unit,
        day.date,
        n,
        n_recovery,
        day.rho,
        tuple(day.cars[i] for i in steered),
        tuple(forecasts[i] for i in steered),
        configuration.kappa,
    )
    if configuration.driver_risk is None:
        rows = day_forecast_rows(sub)
    else:
        rows = driver_risk.day_rows(sub, configuration.driver_risk)
    certificate = certify(
        rows,
        products.hourly(),
        "causal",
        shape_rule=shape_rule(configuration, settings),
        time_limit=BOX_SECONDS,
    )
    if certificate.outcome != "optimal":
        return abstaining(
            f"no certificate ({certificate.outcome})",
            certificate.outcome,
            sub,
            certificate,
            certificate.seconds,
            rows,
        )
    if certificate.value <= settings.zero_value_kwh:
        return abstaining(ZERO_VALUE, "optimal", sub, certificate, certificate.seconds, rows)
    bounds = _bounds(truth, forecaster, configuration, steered, forecasts, n)
    return PreparedDay(
        day.unit,
        truth.record.site,
        truth.record.date,
        len(day.cars),
        tuple(unsteered),
        steered,
        None,
        "optimal",
        float(certificate.value),
        sub,
        certificate,
        bounds,
        paths,
        settings.budget,
        settings.t5_tolerance_kwh,
        certificate.seconds,
        configuration.departure_seen,
        rows,
    )


@dataclass(frozen=True)
class RungEvaluation:
    slot_loss: float  # the largest over the paths
    driver_loss: float
    first_failures: tuple[int | None, ...]  # per path
    finals: np.ndarray | None  # (paths, steered cars) energies at departure; None when abstaining
    resolves: dict  # outcome -> count, over the paths
    resolve_seconds: float  # timing only, never compared
    stay: stay_check.StayCheck = stay_check.NONE  # cars staying to their bound, all paths


@dataclass(frozen=True)
class T5Break:
    path: int  # index into the day's paths
    kind: str  # "a" (first failure earlier) or "b" (a car ends lower)
    rung: str
    predecessor: str
    car: int | None  # car of the fleet (kind "b")
    value: float | None  # the rung's energy (b) or first failing hour (a)
    predecessor_value: float | None


@dataclass(frozen=True)
class DayEvaluation:
    beta: float
    abstained: bool
    rungs: dict  # rung name -> RungEvaluation
    paths_checked: int
    t5_breaks: tuple[T5Break, ...]
    seconds: float  # timing only, never compared

    def compact(self) -> "DayEvaluation":
        """The same without the per-path energies (what the calibration keeps)."""
        rungs = {name: replace(r, finals=None) for name, r in self.rungs.items()}
        return replace(self, rungs=rungs)


def _earlier(first, before) -> bool:
    """True when `first` (a rung's first failure) is earlier than `before` (its predecessor's)."""
    return first is not None and (before is None or first < before)


def t5_breaks(rungs: dict, steered: tuple[int, ...], tolerance: float) -> tuple[T5Break, ...]:
    """T5 (a) and (b) on every path, from each rung's first failures and final energies."""
    found = []
    for name, previous in PREDECESSOR.items():
        now, before = rungs[name], rungs[previous]
        for k, (first, prior) in enumerate(
            zip(now.first_failures, before.first_failures, strict=True)
        ):
            if _earlier(first, prior):
                found.append(T5Break(k, "a", name, previous, None, float(first), _hour(prior)))
            lower = np.flatnonzero(now.finals[k] < before.finals[k] - tolerance)
            for p in lower:
                found.append(
                    T5Break(
                        k,
                        "b",
                        name,
                        previous,
                        steered[int(p)],
                        float(now.finals[k][p]),
                        float(before.finals[k][p]),
                    )
                )
    return tuple(sorted(found, key=lambda b: (b.path, b.kind, RUNGS.index(b.rung), b.car or 0)))


def _hour(value) -> float | None:
    return None if value is None else float(value)


def replay(prepared: PreparedDay, beta: float):
    """`day_replay.replay_day` of the prepared day at `beta` with every rung (module text)."""
    table = prepared.departure_bounds
    return replay_day(
        prepared.day,
        prepared.certificate,
        RUNGS[-1],
        beta,
        prepared.paths,
        departure_bound=lambda car, hour: float(table[car, hour]),
        budget=prepared.budget,
        time_limit=OTHER_SECONDS,
        departure_seen=prepared.departure_seen,
    )


def evaluate(prepared: PreparedDay, beta: float) -> DayEvaluation:
    """The day replayed at scale `beta` by every rung (see the module text)."""
    beta = float(beta)
    n_paths = prepared.paths.shape[0]
    if prepared.abstains or beta <= 0:
        zero = RungEvaluation(0.0, 0.0, (None,) * n_paths, None, {}, 0.0)
        return DayEvaluation(beta, True, dict.fromkeys(RUNGS, zero), 0, (), 0.0)
    start = time.perf_counter()
    result = replay(prepared, beta)
    tolerance = prepared.t5_tolerance_kwh
    rungs = {}
    for name in RUNGS:
        runs = result.by_rung[name]
        outcomes = Counter()
        seconds = 0.0
        for r in runs:
            outcomes.update(r.outcomes())
            seconds += sum(x.seconds for x in r.resolves)
        rungs[name] = RungEvaluation(
            max(r.slot_loss for r in runs),
            max(r.driver_loss for r in runs),
            tuple(r.first_failure for r in runs),
            np.array([r.final for r in runs]),
            dict(sorted(outcomes.items())),
            seconds,
            stay_check.check(prepared.day, np.array([r.final for r in runs]), tolerance),
        )
    breaks = t5_breaks(rungs, prepared.steered, tolerance)
    return DayEvaluation(beta, False, rungs, n_paths, breaks, time.perf_counter() - start)
