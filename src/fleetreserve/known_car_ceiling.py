"""The known-car ceiling of a day in closed form (plan section 2, workplace window).

Reading of plan section 2 ("the largest V_caus, hourly product, true sets, rho = 0"): V_caus is
taken as the known-car causal ceiling below, in which the power of the decision hour is fixed
once at the decision time. It is the gate's best-causal ceiling. It is not b(S) - p(S) with the
decision hour free per request path (V_hind, the hindsight ceiling), which can be larger.

The known-car ceiling is the best real-time value of a day on its true cars: the hourly product,
causal structure, one-way charging, ρ = 0, with the power of the decision hour (hour 0) fixed at
τ. It is what `gate_day.day_ceilings(day).best_causal` solves by a linear programme. For the
hourly product it is a sum over cars, so no programme is needed:

    V = sum_i  max over x in [x_low_i, x_high_i] of
            [ min(r_i s_i, e_i - x) - max(e_i - x - r_i c_i, 0) ]

For car i: `r` the rate (kW), `e` the room (kWh), `d` the departure hour (the car is present in
hours 0 to d - 1), `n` the number of service hours, and

- `s = min(d - 1, n)`: the service hours in which the car is present;
- `c = max(d - 1 - n, 0)`: the recovery hours in which the car is present;
- `x`: the energy drawn in hour 0, between `x_low = max(0, e - r (s + c))` (the rest must fit in
  the later hours) and `x_high = min(r, e)`.

Reading of the bracket: of the energy `y = e - x` still to draw after hour 0, the service hours
can take `min(r s, y)`, which is the flexibility the car gives; the part of `y` that the recovery
hours cannot take, `max(y - r c, 0)`, has to be drawn in the service hours and is subtracted.

The bracket is concave and piecewise linear in `y`, with its slope changing only at `y = r s`
and `y = r c`. Its maximum over the interval of `y` is therefore at an end point or at a break
point inside the interval, and the function tries exactly these.

The formula was found by a probe, not proved. `tests/wp2/test_known_car_ceiling.py` compares it
with the programme on seeded made-up fleets; a fleet on which they differ is a finding, and the
formula is not to be adjusted to fit.

Only ρ = 0 is implemented. A day with ρ above 0 is refused.
"""

from collections.abc import Sequence

import numpy as np

from fleetreserve.day import Day

TOLERANCE = 1e-12


def car_value(rate: float, room: float, departure: int, n_service: int) -> float:
    """One car's term of the sum (see the module text)."""
    if departure < 1:
        raise ValueError(f"departure must be at least 1 hour, got {departure}")
    if rate <= 0 or room < 0:
        raise ValueError(f"rate must be positive and room not negative, got {rate}, {room}")
    present_service = min(departure - 1, n_service)
    present_recovery = max(departure - 1 - n_service, 0)
    x_high = min(rate, room)
    x_low = max(0.0, room - rate * (present_service + present_recovery))
    if x_low > x_high + TOLERANCE:
        raise ValueError(
            f"the room {room} does not fit in a rate of {rate} over {departure} hours: "
            "the car's true set is empty"
        )
    x_low = min(x_low, x_high)
    y_low, y_high = room - x_high, room - x_low  # the energy left after hour 0
    candidates = {y_low, y_high}
    for breakpoint_ in (rate * present_service, rate * present_recovery):
        if y_low <= breakpoint_ <= y_high:
            candidates.add(breakpoint_)
    return max(
        min(rate * present_service, y) - max(y - rate * present_recovery, 0.0) for y in candidates
    )


def known_car_ceiling_arrays(
    rates: Sequence[float], rooms: Sequence[float], departures: Sequence[int], n_service: int
) -> float:
    """The ceiling of a fleet given as arrays: rate, room and departure hour of each car."""
    if not len(rates) == len(rooms) == len(departures):
        raise ValueError("rates, rooms and departures must have the same length")
    return float(
        sum(
            car_value(float(r), float(e), int(d), n_service)
            for r, e, d in zip(rates, rooms, departures, strict=True)
        )
    )


def known_car_ceiling(day: Day) -> float:
    """The known-car ceiling of a `Day` (ρ = 0 only); 0 for a day without cars."""
    if day.rho != 0:
        raise ValueError(f"only rho = 0 is implemented, got {day.rho}")
    return known_car_ceiling_arrays(
        np.array([car.rate for car in day.cars]),
        np.array([car.room for car in day.cars]),
        np.array([car.departure for car in day.cars], dtype=int),
        day.n_service,
    )
