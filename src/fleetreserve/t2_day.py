"""T2's check on one development day with forecasts (plan section 5, T2: "closed-form worst case
equals replay of all 2^|S| vertex paths on 100 development days, no violation where E' holds").

Input (`T2Task`): the day's true cars, and per car its forecast at level kappa (`day.ForecastCar`,
or None when it has none) with its exact plug-out time. The forecasts are made by the caller
(`dev_day_checks`), which also applies the departure rule (`as-planned` or `floored`).

Steps:

1. A car is uncertifiable when it has no forecast or its forecast set is empty
   (`forecast_coverage.forecast_set_empty`). Rule `leave-unsteered`: such a car is left out of
   the certificate (it is not steered and carries no part of the box). Rule `day-abstains`: the
   day gets no certificate. A day with no steered car gets no certificate either.
2. The steered cars' forecast rows (`carsets.forecast_rows`, over the day's hours extended so
   that every forecast departure lies inside it, as `forecast_empirical.ForecastDay` does) are
   certified: hourly product, causal gains, shape rule `largest-smallest-width`
   (`certificate.LargestSmallestWidth`, tolerance `SHAPE_TOLERANCE`).
3. (a) For every forecast row, the closed-form worst case `AffinePolicy.worst_case` at beta = 1
   equals the largest value of the row over all 2^n corner paths of the policy's powers, within
   `REPLAY_TOLERANCE * max(1, |worst case|)`.
4. (b) On every corner path: every steered car whose forecast is conservative (E'_i, from
   `forecast_coverage.check_car`: d_hat <= d, r_hat <= r, e_cap <= e, e_need >= (1 - rho) e)
   meets every row of its TRUE one-way set (`carsets.true_one_way_rows`) within `ROW_TOLERANCE`
   (T2 (c1)); and the steered cars' powers sum to the request in every service hour within
   `REQUEST_TOLERANCE` (the plan's failure threshold, section 1). The day is in E' when it has a
   certificate and every steered car is in E'_i; on such a day both hold for every car (T2 (c2)).
   True rows broken by cars outside E'_i are counted, not judged.

Tolerances (set before any run on real tables): `REPLAY_TOLERANCE` 1e-7 relative,
`ROW_TOLERANCE` 1e-6 kW or kWh, `REQUEST_TOLERANCE` 1e-4 kWh.

A certificate that does not end optimal (a timeout included) is missing for the day, never a
pass. `t2_certificate` is the one call that solves; tests replace it to inject a violation.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from fleetreserve import products
from fleetreserve.carsets import forecast_rows, stack, true_one_way_rows
from fleetreserve.certificate import Certificate, LargestSmallestWidth, certify
from fleetreserve.day import ForecastCar, TrueCar
from fleetreserve.forecast_coverage import check_car, forecast_set_empty
from fleetreserve.products import corner_paths

LEAVE_UNSTEERED = "leave-unsteered"
DAY_ABSTAINS = "day-abstains"
UNCERTIFIABLE_RULES = (LEAVE_UNSTEERED, DAY_ABSTAINS)
SHAPE_RULE = "largest-smallest-width"
SHAPE_TOLERANCE = 1e-6
REPLAY_TOLERANCE = 1e-7
ROW_TOLERANCE = 1e-6
REQUEST_TOLERANCE = 1e-4

CERTIFIED = "certified"
NO_STEERED_CAR = "no certificate: no steered car"
ABSTAINED = "no certificate: the day abstains (an uncertifiable car)"
MAX_LISTED = 5


@dataclass(frozen=True)
class T2Task:
    unit: str
    site: str
    date: str
    n_service: int
    n_recovery: int
    rho: float
    cars: tuple[TrueCar, ...]
    forecasts: tuple[ForecastCar | None, ...]
    exact_departures: tuple[float, ...]
    uncertifiable: str


@dataclass(frozen=True)
class T2Result:
    """`status` is CERTIFIED, NO_STEERED_CAR, ABSTAINED or the certificate's outcome when it
    did not end optimal. The (a) and (b) counts are 0 unless certified."""

    unit: str
    site: str
    date: str
    cars: int
    status: str
    cars_without_forecast: int
    cars_with_empty_set: int
    steered: int
    steered_in_e_prime: int
    day_in_e_prime: bool
    value: float | None
    rows_checked: int
    replay_mismatches: int
    largest_replay_gap: float
    cars_checked_in_e_prime: int
    true_row_violations_in_e_prime: int
    cars_outside_e_prime_breaking_a_true_row: int
    paths_with_request_missed: int
    largest_request_miss: float
    problems: tuple[str, ...]
    seconds: float = field(default=0.0, compare=False)


def forecast_hours(task: T2Task) -> int:
    """Recovery hours covering the day and every forecast departure."""
    latest = max([0.0, *(f.departure for f in task.forecasts if f is not None)])
    return max(task.n_recovery, math.ceil(latest) - 1 - task.n_service)


def t2_certificate(rows) -> Certificate:
    return certify(
        rows,
        products.hourly(),
        "causal",
        shape_rule=LargestSmallestWidth(SHAPE_TOLERANCE),
    )


def _label(task: T2Task) -> str:
    return f"{task.unit} {task.site} {task.date}"


def check_day(task: T2Task) -> T2Result:
    n = task.n_service
    n_recovery = forecast_hours(task)
    no_forecast = [i for i, f in enumerate(task.forecasts) if f is None]
    empty = [
        i
        for i, f in enumerate(task.forecasts)
        if f is not None and forecast_set_empty(f, n, n_recovery)
    ]
    uncertifiable = set(no_forecast) | set(empty)
    steered = [i for i in range(len(task.cars)) if i not in uncertifiable]
    in_e_prime = {
        i: check_car(
            task.forecasts[i],
            task.cars[i].departure,
            task.exact_departures[i],
            task.cars[i].rate,
            task.cars[i].room,
            task.rho,
            n,
            n_recovery,
        ).all_hold
        for i in steered
    }
    counts = {
        "unit": task.unit,
        "site": task.site,
        "date": task.date,
        "cars": len(task.cars),
        "cars_without_forecast": len(no_forecast),
        "cars_with_empty_set": len(empty),
        "steered": len(steered),
        "steered_in_e_prime": sum(in_e_prime.values()),
    }
    zero = {
        "value": None,
        "rows_checked": 0,
        "replay_mismatches": 0,
        "largest_replay_gap": 0.0,
        "cars_checked_in_e_prime": 0,
        "true_row_violations_in_e_prime": 0,
        "cars_outside_e_prime_breaking_a_true_row": 0,
        "paths_with_request_missed": 0,
        "largest_request_miss": 0.0,
        "problems": (),
    }
    if task.uncertifiable == DAY_ABSTAINS and uncertifiable:
        return T2Result(**counts, status=ABSTAINED, day_in_e_prime=False, **zero)
    if not steered:
        return T2Result(**counts, status=NO_STEERED_CAR, day_in_e_prime=False, **zero)

    rows = stack([forecast_rows(task.forecasts[i], n, n_recovery) for i in steered])
    certificate = t2_certificate(rows)
    if certificate.outcome != "optimal":
        return T2Result(
            **counts,
            status=certificate.outcome,
            day_in_e_prime=False,
            **zero,
            seconds=certificate.seconds,
        )
    policy = certificate.policy
    paths = corner_paths(n)
    powers = policy.powers(paths)  # (paths, steered cars, n + 1)

    # (a) closed form against replay, row by row.
    replayed = np.einsum("kt,pkt->pk", rows.coefficients, powers[:, rows.car, :]).max(axis=0)
    closed = policy.worst_case(rows, 1.0)
    gap = np.abs(closed - replayed)
    mismatched = gap > REPLAY_TOLERANCE * np.maximum(1.0, np.abs(closed))
    problems = [
        f"{_label(task)}: forecast row {k} of steered car {steered[rows.car[k]]}: closed-form "
        f"worst case {closed[k]:.9g}, replay {replayed[k]:.9g}"
        for k in np.flatnonzero(mismatched)[:MAX_LISTED]
    ]

    # (b) true rows of the steered cars on every path, and the requests.
    truth = stack([true_one_way_rows(task.cars[i], n, task.rho) for i in steered])
    slack = truth.bounds[None, :] - np.einsum(
        "kt,pkt->pk", truth.coefficients, powers[:, truth.car, :]
    )
    broken_rows = (slack < -ROW_TOLERANCE).any(axis=0)
    broken_car = np.zeros(len(steered), bool)
    np.logical_or.at(broken_car, truth.car, broken_rows)
    conservative = np.array([in_e_prime[i] for i in steered])
    violating = np.flatnonzero(broken_car & conservative)
    problems += [
        f"{_label(task)}: steered car {steered[j]} is in E'_i but breaks a row of its true set "
        f"on a corner path (least slack {slack[:, truth.car == j].min():.9g})"
        for j in violating[:MAX_LISTED]
    ]
    miss = np.abs(powers[:, :, 1:].sum(axis=1) - policy.request(paths)).max(axis=1)
    missed = miss > REQUEST_TOLERANCE
    if missed.any():
        problems.append(
            f"{_label(task)}: the steered cars miss the request on {int(missed.sum())} corner "
            f"paths, by up to {miss.max():.9g} kWh"
        )
    return T2Result(
        **counts,
        status=CERTIFIED,
        day_in_e_prime=bool(conservative.all()),
        value=float(certificate.value),
        rows_checked=int(rows.n_rows),
        replay_mismatches=int(mismatched.sum()),
        largest_replay_gap=float(gap.max()),
        cars_checked_in_e_prime=int(conservative.sum()),
        true_row_violations_in_e_prime=int(violating.size),
        cars_outside_e_prime_breaking_a_true_row=int((broken_car & ~conservative).sum()),
        paths_with_request_missed=int(missed.sum()),
        largest_request_miss=float(miss.max()),
        problems=tuple(problems),
        seconds=certificate.seconds,
    )
