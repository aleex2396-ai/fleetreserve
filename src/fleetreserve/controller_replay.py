"""One controller (`controller.py`) run on one day-path of true cars (plan sections 5 and 10).

`run_controller(day, controller, belief, zeta, requests, hour_zero_target=..., departure_seen=...)`
runs one-way through hours 0 to `n`. At the start of hour `t` it builds the controller's
`HourView`: the target (`hour_zero_target` in hour 0, `requests[t - 1]` in service hour `t`), the
request variables `zeta_1 .. zeta_t` and requests `y_1 .. y_t`, each car's energy drawn since the
decision time, the plugged-in flags of `rungs.present_for_decisions` under `departure_seen`
("hour-start" or "rounded-hour", required, no default) and the belief. The controller's planned
powers must be one per car, finite and at least 0; anything else is refused.

Reading R5: what a true car draws from a planned power, when an hour
fails, recovery, the slot loss and the driver loss are those of `replay.py`, called unchanged:
`replay.take` (one-way) with the true limits of `replay.true_limits` and true presence `t < d`,
then `replay.finish`, which uses `replay.fails`, `replay.after_recovery`, `replay.slot_loss` and
`replay.driver_loss`. The controller keeps running after a failure (plan section 5: "A failed
policy runs on (departed cars at zero, cars capped at what they can take)").

The result is a `replay.Run`: planned powers and draws per car and hour, stored energy after each
hour, delivery and failure per service hour, first failing hour, slot loss, each car's energy at
departure and driver loss.
"""

import math

import numpy as np

from fleetreserve import replay
from fleetreserve.controller import Belief, Controller, HourView
from fleetreserve.day import Day
from fleetreserve.rungs import check_departure_seen, present_for_decisions


def _checked_plan(planned, n_cars: int, hour: int) -> np.ndarray:
    power = np.asarray(planned, dtype=float)
    if power.shape != (n_cars,):
        raise ValueError(f"hour {hour}: a controller must return one planned power per car")
    if not np.all(np.isfinite(power)):
        raise ValueError(f"hour {hour}: planned powers must be finite")
    if np.any(power < 0):
        raise ValueError(f"hour {hour}: planned powers must be at least 0 (one-way)")
    return power


def run_controller(
    day: Day,
    controller: Controller,
    belief: Belief,
    zeta,
    requests,
    *,
    hour_zero_target: float,
    departure_seen: str,
) -> replay.Run:
    """Run `controller` on the day's true cars along `zeta` with `requests` (module text)."""
    check_departure_seen(departure_seen)
    n = day.n_service
    cars = len(day.cars)
    zeta = np.asarray(zeta, dtype=float)
    requests = np.asarray(requests, dtype=float)
    if zeta.shape != (n,):
        raise ValueError(f"zeta must have {n} entries")
    if requests.shape != (n,):
        raise ValueError(f"requests must have {n} entries")
    if belief.n_cars != cars:
        raise ValueError(f"the belief has {belief.n_cars} cars, the day {cars}")
    if not math.isfinite(hour_zero_target):
        raise ValueError("the hour-zero target must be finite")
    departure, rate, room = replay.true_limits(day)
    planned = np.zeros((cars, n + 1))
    draws = np.zeros_like(planned)
    stored = np.zeros(cars)
    for t in range(n + 1):
        view = HourView(
            hour=t,
            n_service=n,
            target=hour_zero_target if t == 0 else requests[t - 1],
            zeta_known=zeta[:t],
            requests_known=requests[:t],
            drawn=stored,
            plugged=present_for_decisions(t, departure, departure_seen),
            belief=belief,
        )
        planned[:, t] = _checked_plan(controller(view), cars, t)
        draws[:, t] = replay.take(
            planned[:, t], stored, t < departure, rate, room, discharge=False, budget=None
        )
        stored = stored + draws[:, t]
    return replay.finish(day, planned, draws, requests)
