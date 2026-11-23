"""Running a plan hour by hour on a day's true cars, and the losses of one day-path (plan 1, 5, 9).

A plan (`plans.Plan`) says what each car should draw; the true car decides what it can take. Plan
section 5: "A failed policy runs on (departed cars at zero, cars capped at what they can take)".
In hour `t` (0 to `n`) car `i`, with energy `x` drawn since the decision time, draws:

- 0 if it has left (`t >= d`);
- for a planned charge `p >= 0`: `min(p, r, e - x)`, its true rate `r` and room `e`;
- for a planned discharge `p < 0`: nothing when the plan is one-way, or when the plan is two-way
  and its rung has zeroed discharges (after the rung's own first failure, plan section 5);
  otherwise `max(p, -r, -(B + x))`, so the stored energy never falls more than the budget `B`
  below its level at the decision time (plan section 9).

Failure (plan section 1, "set here"): in service hour `t` the fleet's energy misses the request
`y_t` by more than `FAILURE_KWH`. Hour 0 carries no request and cannot fail.

Recovery (plan section 1): after the service hours each car charges at its full true rate until it
is full or gone, so it ends with `min(e, x_n + r R)`, `R = max(d - 1 - n, 0)` its recovery hours.

Losses of one day-path:

- slot loss: the share of service hours from the first failure on, `(n - t_first + 1) / n`, or 0
  with no failure;
- driver loss: the share of the fleet's protected energy undelivered at departure, after
  recovery, `sum_i max(0, (1 - rho) e_i - final_i) / sum_i (1 - rho) e_i`, pooled over the cars
  of the day (not a mean of per-car shares); 0 when the day has no protected energy.
"""

from dataclasses import dataclass

import numpy as np

from fleetreserve.day import Day
from fleetreserve.plans import Plan

FAILURE_KWH = 1e-4  # plan section 1: a delivery missing its request by more than this fails


def take(planned, stored, present, rate, room, *, discharge: bool, budget: float | None):
    """What each car draws in one hour (see the module text); arrays of one entry per car.

    `discharge` is True when planned discharges are carried out (a two-way plan whose rung has
    not zeroed them); `budget` (kWh) is then required."""
    planned = np.asarray(planned, dtype=float)
    charge = np.maximum(np.minimum(planned, np.minimum(rate, room - stored)), 0.0)
    if discharge:
        if budget is None:
            raise ValueError("a discharge needs the budget")
        out = np.minimum(np.maximum(planned, np.maximum(-rate, -(budget + stored))), 0.0)
    else:
        out = np.zeros_like(planned)
    return np.where(present, np.where(planned >= 0, charge, out), 0.0)


def discharges_carried(plan: Plan, failed_before: bool) -> bool:
    """Plan section 5: a two-way plan's discharges are carried out until its rung's first
    failure ("a two-way rung zeroes planned discharges after its own first failure")."""
    return plan.two_way and not failed_before


def fails(delivered: float, request: float) -> bool:
    """Plan section 1: the delivery misses the request by more than `FAILURE_KWH`."""
    return abs(delivered - request) > FAILURE_KWH


def true_limits(day: Day) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(departure, rate, room)` of the day's true cars as arrays."""
    cars = day.cars
    return (
        np.array([c.departure for c in cars]),
        np.array([c.rate for c in cars], dtype=float),
        np.array([c.room for c in cars], dtype=float),
    )


def protected(day: Day) -> np.ndarray:
    """Each car's protected energy `(1 - rho) e`."""
    return np.array([c.need(day.rho) for c in day.cars], dtype=float)


def after_recovery(day: Day, stored_end) -> np.ndarray:
    """Each car's energy at departure, after the recovery hours, from its energy after hour `n`."""
    departure, rate, room = true_limits(day)
    hours = np.maximum(departure - 1 - day.n_service, 0)
    x = np.asarray(stored_end, dtype=float)
    return x + np.minimum(rate * hours, np.maximum(room - x, 0.0))


def slot_loss(first_failure: int | None, n_service: int) -> float:
    return 0.0 if first_failure is None else (n_service - first_failure + 1) / n_service


def driver_loss(day: Day, final) -> float:
    need = protected(day)
    total = need.sum()
    if total <= 0:
        return 0.0
    return float(np.maximum(need - np.asarray(final, dtype=float), 0.0).sum() / total)


@dataclass(frozen=True)
class Run:
    """One plan run on one path. Arrays per car and hour 0 to `n` (`planned`, `draws`, `stored`:
    the energy after each hour); `delivered` and `failed` per service hour (index `t - 1`);
    `final`: each car's energy at departure."""

    planned: np.ndarray
    draws: np.ndarray
    stored: np.ndarray
    delivered: np.ndarray
    failed: np.ndarray
    first_failure: int | None
    slot_loss: float
    final: np.ndarray
    driver_loss: float


def finish(day: Day, planned, draws, requests) -> Run:
    """The losses and final energies of a run from its planned powers and draws."""
    draws = np.asarray(draws, dtype=float)
    delivered = draws[:, 1:].sum(axis=0)
    failed = np.array([fails(d, y) for d, y in zip(delivered, requests, strict=True)])
    first = int(np.argmax(failed)) + 1 if failed.any() else None
    stored = np.cumsum(draws, axis=1)
    final = after_recovery(day, stored[:, -1])
    return Run(
        np.asarray(planned, dtype=float),
        draws,
        stored,
        delivered,
        failed,
        first,
        slot_loss(first, day.n_service),
        final,
        driver_loss(day, final),
    )


def run_plan(day: Day, plan: Plan, zeta, requests, *, budget: float | None = None) -> Run:
    """Run `plan` on the day's true cars along the path `zeta` with the requests `requests`.

    A two-way plan needs `budget`; its discharges are zeroed from the hour after its first failure
    (`discharges_carried`)."""
    n = day.n_service
    zeta = np.asarray(zeta, dtype=float)
    requests = np.asarray(requests, dtype=float)
    if plan.n_cars != len(day.cars) or plan.n_service != n:
        raise ValueError("the plan is for another fleet or window")
    if zeta.shape != (n,) or requests.shape != (n,):
        raise ValueError(f"zeta and requests must have {n} entries")
    departure, rate, room = true_limits(day)
    planned = np.zeros((plan.n_cars, n + 1))
    draws = np.zeros_like(planned)
    stored = np.zeros(plan.n_cars)
    failed_before = False
    for t in range(n + 1):
        planned[:, t] = plan.powers(t, zeta)
        discharge = discharges_carried(plan, failed_before)
        draws[:, t] = take(
            planned[:, t], stored, t < departure, rate, room, discharge=discharge, budget=budget
        )
        stored = stored + draws[:, t]
        if t >= 1 and fails(draws[:, t].sum(), requests[t - 1]):
            failed_before = True
    return finish(day, planned, draws, requests)
