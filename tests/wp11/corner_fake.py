"""Hand-made and seeded made-up prepared days for the tests of the corner search.

`compare_fake` (in `tests/wp10`, which the test configuration does not put on the import path)
is reached by adding its folder here, so its `prepared_of`, `offer_days` and `NAMES` are reused,
not copied (`NAMES` is passed on for the command's tests).

`hand(day, policy, seen)`: the prepared day of a hand-made day and policy, with a certificate
built by hand (outcome optimal, value `2 sum(w)`).

Hand days, each with two service hours, two recovery hours, rho 0 and forecasts equal to the
true limits:

- `SINGLE`: one car (leaves in hour 5, rate 1 kW, room 1.5 kWh); box m = (0.5, 0.5),
  w = (0.5, 0.5); the car takes the whole box: `0.5 + 0.5 beta zeta_t` in hour t. Its rows give
  the hour thresholds s_1 = 1 (power rows) and s_2 = 0.5 (its stored energy after hour 2,
  `1 + beta (zeta_1 + zeta_2) / 2 <= 1.5`). At scale 1 only the corner (+1, +1) breaks: the car
  is full after 1.5 kWh, draws 0.5 of 1 kWh in hour 2, slot loss 1/2; recovery fills it, so no
  driver loss on any corner.
- `HIDDEN`, dated 2020-01-04, a date whose audited paths (`audit_paths`, unit "hand") miss the
  corner (-1, +1): car A (leaves in hour 5, rate 1.5, room 3) plans `0.5 + 0.5 beta zeta_1` and
  `1 + 0.5 beta (zeta_2 - zeta_1)`; car B (leaves in hour 5, rate 2, room 3) plans 0.5 and
  `1 + 0.5 beta zeta_1`; box m = (1, 2), w = (0.5, 0.5). Thresholds s_1 = 1 (A's power at least
  0), s_2 = 0.5 (A's rate 1.5 against `1 + beta`). At scale 1 only the corner (-1, +1) breaks: A
  is asked for 2 kW in hour 2, draws 1.5, slot loss 1/2; no driver loss (recovery fills both).
- `SHORT`: `SINGLE` with the car leaving in hour 3 (no recovery for it): at scale 0.5 no hour
  fails on any corner, but the car ends with what it drew, `1 + 0.25 (zeta_1 + zeta_2)` kWh of
  its protected 1.5: 0.5 on (-1, -1) (driver loss 2/3), 1.0 on the two mixed corners (1/3), 1.5
  on (+1, +1) (0).

`early_units()`: made-up gate units for the command (its own text).

`solving_methods()`: the names of the registry's methods whose controllers solve programmes, by
the rule `corner_search` reads (a controller with a method `solver_counts`), each controller built
on `SINGLE` at scale 1 with `SINGLE`'s rows; no name is written here, so it finds a method
registered later whose controller, built so, has a method `solver_counts` (not one whose
controller gains it only on other days or scales, or one that cannot be built so).
"""

import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wp10"))

from compare_fake import NAMES, offer_days, prepared_of  # noqa: E402
from gate_fake import GATE_UNITS, gate_cars, gate_units  # noqa: E402

from fleetreserve import compare_methods, products  # noqa: E402
from fleetreserve.certificate import Certificate  # noqa: E402
from fleetreserve.day import Day, ForecastCar, TrueCar  # noqa: E402
from fleetreserve.plans import Plan  # noqa: E402
from fleetreserve.policy import AffinePolicy  # noqa: E402
from fleetreserve.sparse_programme import Sizes  # noqa: E402

__all__ = [
    "HIDDEN",
    "NAMES",
    "SHORT",
    "SINGLE",
    "early_units",
    "hand",
    "offer_days",
    "solving_methods",
]


def _forecast(car: TrueCar) -> ForecastCar:
    return ForecastCar(float(car.departure), car.rate, car.room, car.room)


def _day(cars, date=datetime.date(2020, 1, 1)) -> Day:
    return Day("hand", date, 2, 2, 0.0, tuple(cars), tuple(_forecast(c) for c in cars), 0.1)


def hand(day: Day, policy: AffinePolicy, seen: str = "rounded-hour"):
    value = float(2 * np.sum(policy.w))
    certificate = Certificate(
        "optimal",
        products.hourly(),
        "causal",
        "any",
        value,
        None,
        policy,
        value,
        0.0,
        Sizes(0, 0, 0),
    )
    return prepared_of(day, certificate, seen)


SINGLE_POLICY = AffinePolicy(
    [0.5, 0.5], [0.5, 0.5], [[0.0, 0.5, 0.5]], [[[0, 0], [0.5, 0], [0, 0.5]]]
)
SINGLE = hand(_day([TrueCar(5, 1.0, 1.5)]), SINGLE_POLICY)

HIDDEN_POLICY = AffinePolicy(
    [1.0, 2.0],
    [0.5, 0.5],
    [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
    [[[0, 0], [0.5, 0], [-0.5, 0.5]], [[0, 0], [0, 0], [0.5, 0]]],
)
HIDDEN = hand(
    _day([TrueCar(5, 1.5, 3.0), TrueCar(5, 2.0, 3.0)], datetime.date(2020, 1, 4)), HIDDEN_POLICY
)

SHORT = hand(_day([TrueCar(3, 1.0, 1.5)]), SINGLE_POLICY)


def early_units() -> dict:
    """The made-up gate units of `gate_fake` with more early departures: the second car leaves
    within the service hour on every seventh day of the NREL unit and every fifth day of the two
    pooled sites, so some methods have a scale above 0 and break on test days."""
    units = gate_units()
    units[GATE_UNITS[1]] = gate_cars(60, early_every=7)
    units[GATE_UNITS[2]] = gate_cars(30, "B", early_every=5) + gate_cars(30, "C", early_every=5)
    return units


def solving_methods() -> frozenset:
    """The registry's methods whose controllers have `solver_counts` (module text)."""
    policy = SINGLE.certificate.policy
    plan = Plan.from_policy(policy, 1.0)
    inputs = compare_methods.MethodInput(SINGLE.day, policy, 1.0, plan, SINGLE.rows)
    return frozenset(
        m.name
        for m in compare_methods.METHODS
        if m.kind == "controller" and hasattr(m.build(inputs), "solver_counts")
    )
