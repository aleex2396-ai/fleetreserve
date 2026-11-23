"""Hand-built re-solve inputs and made-up prepared days for the tests of the re-solve diagnosis.

`hand(cars, ...)`: a `resolve.Remaining` and a `resolve.Guard` for a re-solve at service hour
`hour`, written car by car (`HandCar`): whether the car is present, the energy it has drawn, its
known charge bound, its departure bound, its forecast and the powers the guarded plan gives it in
hours 0 to `n` (constants; `gains[car]` adds gains on the requests). The guard is anchored at the
re-solve's own hour with the cars' energies, so nothing is owed from earlier hours.

`prepared_of(day, certificate, seen)`: a `gate_day.PreparedDay` of a made-up day with forecasts
(`made_up_offer_days.offer_day`), every car steered, with the made-up departure-bound table of
`made_up_offer_days.departure_rule` and the day's audited paths (or `paths`).
"""

from dataclasses import dataclass

import numpy as np
from made_up_offer_days import departure_rule

from fleetreserve.audit_paths import audited_paths
from fleetreserve.day import ForecastCar
from fleetreserve.gate_day import PreparedDay
from fleetreserve.plans import Plan
from fleetreserve.resolve import Guard, Remaining

BUDGET = 5.0
T5_TOLERANCE = 1e-6


@dataclass(frozen=True)
class HandCar:
    planned: tuple  # the guarded plan's power in hours 0 to n
    rate: float = 2.0
    cap: float = 10.0
    need: float = 0.0
    stored: float = 0.0
    present: bool = True
    departure: float = 9.0  # the bound the re-solve is given; after every hour by default
    charge_bound: float = np.inf


def hand(cars, *, hour, m, w, known, beta=1.0, n_recovery=0, gains=None, clip=True):
    """(`Remaining`, `Guard`) of the module text."""
    n = len(m)
    constant = np.array([c.planned for c in cars], dtype=float)
    plan_gains = np.zeros((len(cars), n + 1, n))
    for car, value in (gains or {}).items():
        plan_gains[car] = value
    stored = np.array([c.stored for c in cars], dtype=float)
    state = Remaining(
        hour,
        np.asarray(known, dtype=float),
        np.asarray(m, dtype=float),
        np.asarray(w, dtype=float),
        beta,
        np.array([c.present for c in cars]),
        stored,
        np.array([c.charge_bound for c in cars], dtype=float),
        np.array([c.departure if c.present else np.nan for c in cars]),
        tuple(ForecastCar(9.0, c.rate, c.cap, c.need) for c in cars),
        n_recovery,
    )
    return state, Guard(Plan(constant, plan_gains, two_way=False), hour, stored.copy(), clip=clip)


def prepared_of(day, certificate, seen, paths=None) -> PreparedDay:
    rule = departure_rule(day, seen)
    n = day.n_service
    cars = len(day.cars)
    table = np.array([[rule(i, t) for t in range(n + 1)] for i in range(cars)])
    return PreparedDay(
        day.unit,
        "made-up-site",
        day.date.isoformat(),
        cars,
        (),
        tuple(range(cars)),
        None if certificate.outcome == "optimal" else f"no certificate ({certificate.outcome})",
        certificate.outcome,
        float(certificate.value) if certificate.outcome == "optimal" else 0.0,
        day,
        certificate,
        table,
        audited_paths(day.unit, day.date, n) if paths is None else np.asarray(paths, dtype=float),
        BUDGET,
        T5_TOLERANCE,
        0.0,
        seen,
    )
