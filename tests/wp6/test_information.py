"""The information model of `rungs.py`: a rung decides from what it is assumed to observe.

Twin days: on each certified made-up day (`made_up_offer_days`, every forecast kind), each car in
turn is changed in one fact a rung cannot observe at an hour's start, the rest kept:

- "departure": a car unplugged within service hour `d` is unplugged within hour `d + 1` instead.
  Under "hour-start" the two days look the same at the start of hours 0 to `d` (the car is plugged
  in at the start of hour `d` on both, and draws the same before it); under "rounded-hour" only at
  the start of hours 0 to `d - 1`, since the car is absent from hour `d` on the first day;
- "room": the car's true room raised half way to its rate times its hours;
- "rate": the car's true rate times 1.5.

On both days the rungs run the same certified policy on the same paths. For each rung, the hour
`T` up to which the two days look the same to it is the last hour before which the observed
plug-in states and the rung's own draws agree, and no later than its predecessor's `T`. Every
rung takes the same decisions on both days in hours 0 to `T` (`same_decisions`: plan in force,
broken or not, discharging or not, re-solves and their outcomes). Counted: `N_COMPARED`
(twin, path, rung) comparisons, `N_WITH_RESOLVE` of them with a re-solve by hour `T`, under each
reading. Under "rounded-hour" the decisions in hour `d` of a departure twin may differ, and do on
`N_DIFFER` comparisons: the two readings are not the same (under "hour-start" that count is 0 by
the check above).
"""

import dataclasses
import random

import numpy as np
import pytest
from made_up_offer_days import departure_rule, offer_day
from test_rungs_hand import same_decisions

from fleetreserve.audit_paths import audited_paths
from fleetreserve.rungs import DEPARTURE_SEEN, run_rungs

SEED = 3
N_DAYS = 12
N_PATHS = 4
BETA = 1.0
N_COMPARED = {"hour-start": 824, "rounded-hour": 824}
N_WITH_RESOLVE = {"hour-start": 198, "rounded-hour": 256}
N_DIFFER = {"hour-start": 0, "rounded-hour": 20}


def _twins(day, seen):
    n = day.n_service
    for i, car in enumerate(day.cars):
        changed = []
        if car.departure <= n:
            changed.append(("departure", dataclasses.replace(car, departure=car.departure + 1)))
        larger = (car.room + car.rate * car.departure) / 2  # its true set stays non-empty
        changed.append(("room", dataclasses.replace(car, room=larger)))
        changed.append(("rate", dataclasses.replace(car, rate=1.5 * car.rate)))
        for kind, twin in changed:
            cars = day.cars[:i] + (twin,) + day.cars[i + 1 :]
            seen_until = n
            if kind == "departure":
                seen_until = car.departure if seen == "hour-start" else car.departure - 1
            yield kind, dataclasses.replace(day, cars=cars), seen_until, car.departure


def _first_difference(a, b):
    differ = np.flatnonzero(np.any(a != b, axis=0))
    return int(differ[0]) if differ.size else a.shape[1]


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_decisions_do_not_depend_on_what_is_not_observable(seen):
    rng = random.Random(SEED)
    compared = with_resolve = differ = 0
    for k in range(N_DAYS):
        day, certificate, _ = offer_day(rng, k)
        if certificate.outcome != "optimal":
            continue
        kw = dict(departure_bound=departure_rule(day, seen), departure_seen=seen)
        for kind, twin, seen_until, departure in _twins(day, seen):
            for path in audited_paths(day.unit, day.date, day.n_service)[:N_PATHS]:
                first = run_rungs(day, certificate.policy, BETA, path, **kw)
                second = run_rungs(twin, certificate.policy, BETA, path, **kw)
                reach = _first_difference(first["fixed"].run.draws, second["fixed"].run.draws)
                upto = reach
                for name in ("one-way", "two-way"):
                    own = _first_difference(first[name].run.draws, second[name].run.draws)
                    reach = min(reach, own, day.n_service)
                    upto = min(upto, seen_until, own, day.n_service)
                    assert same_decisions(first[name], second[name], upto), (k, kind, name)
                    compared += 1
                    with_resolve += any(r.hour <= upto for r in first[name].resolves)
                    if kind == "departure":
                        differ += not same_decisions(
                            first[name], second[name], min(reach, departure)
                        )
    print("compared, with a re-solve by the hour, differ:", compared, with_resolve, differ)
    found = (compared, with_resolve, differ)
    assert found == (N_COMPARED[seen], N_WITH_RESOLVE[seen], N_DIFFER[seen])
