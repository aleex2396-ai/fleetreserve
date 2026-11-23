"""The interface of an hourly tracking controller (plan section 10, the comparison methods).

Plan section 10: "All get the same forecasts, requests, calibration and loss, plus oracle
departures." This module says what such a controller sees at the start of an hour and what it
returns. It holds no rule; `priority_rules` and `plan_controller` hold rules, and
`controller_replay` runs a rule on one day-path.

A controller is any callable `controller(view) -> powers`: `view` is an `HourView`, `powers` an
array of one planned power (kW, so kWh in the hour) per car of the day, in car order.

Readings of the plan's wording (each is the closest reading; the alternative is in brackets):

  R1. A controller is a real-time rule. At the start of hour `t` (0 to `n`) it sees the target of
      the hour (the request `y_t` of service hour `t`; for hour 0, see R3), the requests and the
      request variables `zeta_1 .. zeta_t` known by then, the energy each car has drawn since the
      decision time, which cars it may count as plugged in, and its belief about each car
      (departure, rate, energy cap, energy need). It never sees a later request. It sees the true
      cars only through the energy drawn, the plugged-in flags and, under the oracle belief (R2),
      the belief itself. Which cars it counts as plugged in follows one of the two readings of
      `rungs.DEPARTURE_SEEN` ("hour-start", "rounded-hour"), chosen by the caller of
      `controller_replay.run_controller` with no default (`rungs.present_for_decisions`). It
      returns each car's planned power for hour `t`, one-way (at least 0). [Alternative: a rule
      that also sees later requests, which is not a real-time rule.]
  R2. A belief has two variants (plan section 10, "plus oracle departures"); the controller code is
      the same for both. "forecast" (`forecast_belief`): the forecast bounds of the day's cars at
      level kappa, read from `Day.forecasts` (`day.ForecastCar`: `departure`, `rate`, `cap`,
      `need`), the forecasts the certificate of a prepared day (`gate_day.PreparedDay.day`) is
      built on; nothing is recomputed. "oracle" (`oracle_belief`): the car's true departure, rate,
      room as cap, and protected energy `(1 - rho) e` as need (`replay.true_limits`,
      `replay.protected`). [Alternative: the oracle variant keeps the forecast rate and energy and
      replaces the departure only.]
  R3. The targets of service hours 1 to `n` are the shared requests `y_t = m_t + beta w_t zeta_t`
      of the certified box, which the caller passes to `controller_replay.run_controller` as it
      passes them to `replay.run_plan`. Hour 0 carries no request;
      its target is the fleet total of the certified fixed plan in hour 0 (`hour_zero_target`: the
      sum over cars of the plan's constant in hour 0), and each rule splits it by its own rule.
      [Alternative: charge nothing in hour 0, or every car at full rate.]

`HourView` copies its arrays and makes them read-only, and a `Belief` makes its arrays read-only,
so an ordinary write by a controller (`array[i] = x`) raises `ValueError` and does not change the
state the replay keeps. That holds for ordinary writes only: a controller that sets an array's
`flags.writeable` back to True can write to it, and the `Belief` in every hour's view is the
object given to `controller_replay.run_controller`, not a copy.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fleetreserve import replay
from fleetreserve.day import Day
from fleetreserve.plans import Plan


def _vector(value, name: str) -> np.ndarray:
    array = np.array(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one value per car, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class Belief:
    """A controller's belief about each car (R2): departure (hours from the decision hour, may be
    fractional), rate (kW, positive), energy cap and energy need (kWh, at least 0)."""

    departure: np.ndarray
    rate: np.ndarray
    cap: np.ndarray
    need: np.ndarray

    def __post_init__(self):
        fields = ("departure", "rate", "cap", "need")
        arrays = {k: _vector(getattr(self, k), f"belief {k}") for k in fields}
        sizes = {k: v.size for k, v in arrays.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"belief arrays must have one entry per car each, got sizes {sizes}")
        if np.any(arrays["rate"] <= 0):
            raise ValueError("a believed rate must be positive")
        for key in ("departure", "cap", "need"):
            if np.any(arrays[key] < 0):
                raise ValueError(f"a believed {key} must be at least 0")
        for key, value in arrays.items():
            object.__setattr__(self, key, value)

    @property
    def n_cars(self) -> int:
        return int(self.rate.size)


def forecast_belief(day: Day) -> Belief:
    """R2, "forecast": the day's forecast bounds at level kappa (`Day.forecasts`)."""
    if day.forecasts is None:
        raise ValueError("a forecast belief needs the day's forecasts")
    f = day.forecasts
    return Belief(
        [c.departure for c in f], [c.rate for c in f], [c.cap for c in f], [c.need for c in f]
    )


def oracle_belief(day: Day) -> Belief:
    """R2, "oracle": the true departure, rate, room (as cap) and protected energy (as need)."""
    departure, rate, room = replay.true_limits(day)
    return Belief(departure, rate, room, replay.protected(day))


def hour_zero_target(plan: Plan) -> float:
    """R3: the fleet total of `plan` in hour 0, the sum over cars of its constant in hour 0."""
    return float(plan.constant[:, 0].sum())


@dataclass(frozen=True)
class HourView:
    """What a controller sees at the start of `hour` (R1).

    `target`: the request of the hour (hour 0: the hour-zero target, R3). `zeta_known`,
    `requests_known`: `zeta_1 .. zeta_hour` and `y_1 .. y_hour` (empty in hour 0). `drawn`: each
    car's energy drawn since the decision time. `plugged`: whether the controller counts each car
    as plugged in. `n_service`: the number of service hours."""

    hour: int
    n_service: int
    target: float
    zeta_known: np.ndarray
    requests_known: np.ndarray
    drawn: np.ndarray
    plugged: np.ndarray
    belief: Belief

    def __post_init__(self):
        if not 0 <= self.hour <= self.n_service:
            raise ValueError(f"hour must be 0 to {self.n_service}, got {self.hour}")
        if not np.isfinite(self.target):
            raise ValueError("the target must be finite")
        object.__setattr__(self, "target", float(self.target))
        for key in ("zeta_known", "requests_known"):
            known = np.array(getattr(self, key), dtype=float).reshape(-1)
            if known.size != self.hour:
                raise ValueError(f"{key} must hold {self.hour} entries, got {known.size}")
            known.flags.writeable = False
            object.__setattr__(self, key, known)
        drawn = _vector(self.drawn, "drawn")
        plugged = np.array(self.plugged)
        if plugged.dtype != bool or plugged.shape != drawn.shape:
            raise ValueError("plugged must be one True or False per car")
        if drawn.size != self.belief.n_cars:
            raise ValueError(f"{drawn.size} cars drawn against a belief of {self.belief.n_cars}")
        plugged.flags.writeable = False
        object.__setattr__(self, "drawn", drawn)
        object.__setattr__(self, "plugged", plugged)

    @property
    def n_cars(self) -> int:
        return int(self.drawn.size)


Controller = Callable[[HourView], np.ndarray]
