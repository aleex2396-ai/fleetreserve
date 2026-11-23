"""Made-up prepared days and the configuration names of the made-up gate matrix, for the tests of
the comparison command.

`prepared_of(day, certificate, seen)`: a `gate_day.PreparedDay` of a made-up day with forecasts
(`made_up_offer_days.offer_day`), every car steered, the made-up departure-bound table of
`made_up_offer_days.departure_rule`, the day's audited paths and, as its rows, the day's forecast
rows (`carsets.day_forecast_rows`), on which `offer_day` certifies the day's box.

`offer_days(count, seed)`: the prepared days of the first `count` seeded made-up offer days whose
certificate is optimal with a value above 1e-6 kWh, the reading of `departure_seen` alternating
with the draw (even draws "hour-start", odd draws "rounded-hour").

`solving_methods()`: the names, in the registry's order, of the controller methods of
`compare_methods.METHODS` whose controller has a method `solver_counts` (the rule by which
`compare_day` asks a controller for solver counts), each controller built from the fixed plan at
scale 0.5 of the first made-up offer day (`offer_days(1)`).

`NAMES`: the configurations of the made-up gate matrix (`gate_fake.matrix_document`): the dial
family at kappa 0.3 without rows and at the dials 0.03 and 0.1, each with its oracle partner.

`two_hour_cars(n_days, site)`: made-up cars for tables with `TWO_HOURS` (2) service hours, so that
the hindsight ceiling and the best-causal ceiling differ: five cars a day plugged in at 8:00 (τ at
10:00) and leaving 3.01, 3.01, 4.01, 5.01 and 6.01 hours after τ (true departure hours 3, 3, 4, 5
and 6), with rates 2, 2, 1, 2.5 and 3 kW and rooms 2, 1, 10, 1.5 and 2 kWh plus 0.1 kWh times the
day's number modulo 4 (the third car's room is cut to what it can draw at its rate before its
departure hour, 4 kWh). On the days of `two_hour_cars(28)`, `gate_day.day_ceilings` gives a
hindsight-free-decision-hour ceiling of 6.5, 6.8, 7.1 and 7.4 kWh and a best-causal ceiling of
3.5, 3.7, 3.9 and 4.1 kWh, in turn with the day's number modulo 4, so the two differ on every day
(`test_compare_splits_command` checks that the first is above the second by more than 0.5 kWh on
every day). Two cars of five leave at the earliest
departure, so the empirical departure quantile at kappa 0.3 over the other days' cars is that
departure, 3.01 hours after τ.
"""

import random

import numpy as np
from made_up_offer_days import departure_rule, offer_day
from truth_tables_fake import FakeCar, date_of

from fleetreserve import compare_methods
from fleetreserve.audit_paths import audited_paths
from fleetreserve.carsets import day_forecast_rows
from fleetreserve.compare_methods import MethodInput
from fleetreserve.gate_day import PreparedDay
from fleetreserve.plans import Plan
from fleetreserve.rungs import DEPARTURE_SEEN

PLAIN = ("main-rounded-hour-k0.3-empirical", "main-rounded-hour-k0.3-oracle")
LOW = ("main-rounded-hour-k0.3-eps0.03-empirical", "main-rounded-hour-k0.3-eps0.03-oracle")
HIGH = ("main-rounded-hour-k0.3-eps0.1-empirical", "main-rounded-hour-k0.3-eps0.1-oracle")
NAMES = PLAIN + LOW + HIGH


def prepared_of(day, certificate, seen) -> PreparedDay:
    rule = departure_rule(day, seen)
    n = day.n_service
    cars = len(day.cars)
    table = np.array([[rule(i, t) for t in range(n + 1)] for i in range(cars)])
    optimal = certificate.outcome == "optimal"
    return PreparedDay(
        day.unit,
        "made-up-site",
        day.date.isoformat(),
        cars,
        (),
        tuple(range(cars)),
        None if optimal else f"no certificate ({certificate.outcome})",
        certificate.outcome,
        float(certificate.value) if optimal else 0.0,
        day,
        certificate,
        table,
        audited_paths(day.unit, day.date, n),
        5.0,
        1e-6,
        0.0,
        seen,
        day_forecast_rows(day),
    )


def offer_days(count: int, seed: int = 20261114) -> list[PreparedDay]:
    rng = random.Random(seed)
    found = []
    k = 0
    while len(found) < count:
        day, certificate, _ = offer_day(rng, k)
        if certificate.outcome == "optimal" and certificate.value > 1e-6:
            found.append(prepared_of(day, certificate, DEPARTURE_SEEN[k % 2]))
        k += 1
    return found


def solving_methods() -> tuple[str, ...]:
    prepared = offer_days(1)[0]
    policy = prepared.certificate.policy
    inputs = MethodInput(prepared.day, policy, 0.5, Plan.from_policy(policy, 0.5), prepared.rows)
    return tuple(
        m.name
        for m in compare_methods.METHODS
        if m.kind == "controller" and hasattr(m.build(inputs), "solver_counts")
    )


TWO_HOURS = 2
TWO_HOUR_CARS = (
    (13.01, 2.0, 2.0),
    (13.01, 2.0, 1.0),
    (14.01, 1.0, 10.0),
    (15.01, 2.5, 1.5),
    (16.01, 3.0, 2.0),
)


def two_hour_cars(n_days: int, site: str = "A") -> list[FakeCar]:
    return [
        FakeCar(
            site,
            date_of(k),
            8.0,
            out,
            rate=rate,
            room=room + 0.1 * (k % 4),
            session=f"{site}-{k}-{j}",
        )
        for k in range(n_days)
        for j, (out, rate, room) in enumerate(TWO_HOUR_CARS)
    ]
