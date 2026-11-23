"""Priority rules as hourly tracking controllers (plan section 10: "LLF, EDF, smoothed LLF").

Each rule is a controller of `controller.py`: from an `HourView` it returns each car's planned
power for the hour, using only the view (the hour, its target, the energy drawn, the plugged-in
flags and the belief). The same code runs with the forecast and the oracle belief.

Reading R4 of the plan's wording (the alternative is in brackets):

  R4. The most a car can take in hour `t`, as the controller believes it (`most`):
      `min(r_hat f_t, cap - drawn)`, at least 0, with `f_t = min(max(d_hat - t, 0), 1)` the share
      of hour `t` before the believed departure; 0 for a car not counted as plugged in. The rule
      plans `min(max(target, 0), sum_i most_i)` in total and never more than a car's most: when the
      believed total is short of the target, every car gets its most (the miss then shows as a
      failure in the replay, on the true cars); a target below 0 plans nothing. [Alternative: plan
      the whole target and let the true cars cap it.]

      - EDF (`earliest_deadline_first`): cars are filled up to their most in order of believed
        departure, earliest first; ties by car order.
      - LLF (`least_laxity_first`): cars are filled up to their most in order of laxity, least
        first; ties by car order. Laxity at the start of hour `t` (`laxity`):
        `(d_hat - t) - max(need - drawn, 0) / r_hat`.
      - Smoothed LLF (`smoothed_least_laxity`): the smoothed least-laxity rule as this module
        defines it. Each car with a believed most above 0 plans
        `clip(r_hat (L - laxity + 1), 0, most)`, with one level `L` for the hour such that the
        planned total is the total above. A car strictly inside its limits then ends the hour with
        laxity `laxity - 1 + power / r_hat = L` (counted this way, without taking the remaining
        need at 0 again), so all such cars end with equal laxity.

Relation of the smoothed rule to the smoothed least-laxity-first algorithm of Chen et al. (2022),
as read in its arXiv version 2102.08610v1 (17 February 2021); the journal version was not read,
and the equation numbers below are those of the arXiv version. In each time slot `t` the paper
charges car `i` at `rbar_i (L(t) - l_i(t) + 1)` projected on `[0, min(rbar_i, e_i(t))]` (its
eq. 16), with one level `L(t)` such that these rates add up to
`min(P(t), sum_i min(rbar_i, e_i(t)))` (its eq. 17). There `l_i(t) = [d_i - t]^+ - e_i(t) / rbar_i`
is the laxity (its Definition 4), `d_i` the departure, `e_i(t)` the remaining energy demand,
`rbar_i` the peak rate and `P(t)` the station's power limit; a car with `rbar_i <= e_i(t)` ends
the slot with laxity `L(t)` clipped to `[l_i(t) - 1, l_i(t)]` (the sentence after its eq. 17).
This module's smoothed rule has the form of eq. 16 and eq. 17 with `P(t)` read as
`max(target, 0)`, and differs from the paper in these points:

  (a) The upper bound of a car's power. The paper's is `min(rbar_i, e_i(t))`, the remaining
      demand at most. This module's is `most`: the believed rate times the share of the hour
      before the believed departure, and the believed cap minus the energy drawn, so a car may be
      planned more than its believed remaining need `need - drawn`. Following a request above
      what the fleet needs requires that.
  (b) The share of the hour before a believed departure inside the hour. The paper has none: its
      departures fall on slot boundaries (its eq. 1).
  (c) Forecast departures, rates and needs (or the oracle belief's) in place of the departure and
      energy demand that drivers enter on arrival (the paper's footnote 4).
  (d) The level `L` is solved exactly on the piecewise-linear total (below), in place of the
      paper's bisection to a tolerance (its Algorithm 1).
  (e) A target below 0 plans nothing. The paper's power limit `P(t)` is at least 0 (its eq. 2).
  (f) The laxity's time term is `d_hat - t`, without the paper's `[.]^+`. A car whose believed
      departure has passed has a most of 0 and is planned 0 either way.

How `L` is found. The planned total as a function of `L` is continuous, does not decrease and is
linear between the points where a car reaches 0 (`L = laxity - 1`) or its most
(`L = laxity - 1 + most / r_hat`). A bisection over the sorted points finds the two neighbouring
points between which the total reaches its aim, and `L` is then solved from the linear piece
between them; the planned total equals its aim up to floating-point rounding (the tests check
1e-9 kWh).
"""

import numpy as np

from fleetreserve.controller import HourView


def most(view: HourView) -> np.ndarray:
    """R4: the most each car can take in the hour as the controller believes it."""
    belief = view.belief
    share = np.clip(belief.departure - view.hour, 0.0, 1.0)
    limit = np.maximum(np.minimum(belief.rate * share, belief.cap - view.drawn), 0.0)
    return np.where(view.plugged, limit, 0.0)


def laxity(view: HourView) -> np.ndarray:
    """R4: `(d_hat - t) - max(need - drawn, 0) / r_hat` per car at the start of the hour."""
    belief = view.belief
    left = np.maximum(belief.need - view.drawn, 0.0)
    return (belief.departure - view.hour) - left / belief.rate


def _aim(view: HourView, limit: np.ndarray) -> float:
    """The planned total: `min(max(target, 0), sum of most)`."""
    return min(max(view.target, 0.0), float(limit.sum()))


def _fill(view: HourView, key: np.ndarray) -> np.ndarray:
    """Fill cars up to their most in the order of `key` (smallest first, ties by car order)."""
    limit = most(view)
    aim = _aim(view, limit)
    power = np.zeros(view.n_cars)
    if aim >= limit.sum():
        return limit
    left = aim
    for i in np.argsort(key, kind="stable"):
        power[i] = min(limit[i], left)
        left = max(left - power[i], 0.0)
    return power


def earliest_deadline_first(view: HourView) -> np.ndarray:
    """EDF (R4): earliest believed departure first."""
    return _fill(view, view.belief.departure)


def least_laxity_first(view: HourView) -> np.ndarray:
    """LLF (R4): least laxity first."""
    return _fill(view, laxity(view))


def smoothed_least_laxity(view: HourView) -> np.ndarray:
    """The smoothed least-laxity rule as this module defines it (R4 and the module text)."""
    limit = most(view)
    aim = _aim(view, limit)
    power = np.zeros(view.n_cars)
    if aim >= limit.sum():
        return limit
    live = limit > 0
    if aim <= 0 or not live.any():
        return power
    rate = view.belief.rate[live]
    start = laxity(view)[live] - 1.0  # the level at which a car starts to take power
    top = limit[live]

    def total(level: float) -> float:
        return float(np.clip(rate * (level - start), 0.0, top).sum())

    points = np.unique(np.concatenate([start, start + top / rate]))
    low, high = 0, points.size - 1  # total(points[0]) = 0 < aim < total(points[-1])
    while high - low > 1:
        middle = (low + high) // 2
        if total(points[middle]) < aim:
            low = middle
        else:
            high = middle
    a, b = points[low], points[high]
    below, above = total(a), total(b)
    level = a + (aim - below) * (b - a) / (above - below)
    power[live] = np.clip(rate * (level - start), 0.0, top)
    return power
