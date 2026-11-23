"""What the registry's method `asa` (`compare_methods.asa`) reads of a day, on the seeded made-up
offer days of `tests/wp6` with the day's certified box at the scale 0.5, replayed by
`controller_replay.run_controller` with the forecast belief (one seeded path per day).

1. No leak of the true limits. A second day equal to the first but for its true cars (every
   departure one hour later where the day's hours allow it, or, in the second case, one hour
   earlier where it is after hour 1; every rate 1.5 times; every room halved), with the same
   forecasts, has the same forecast belief. The controllers the registry
   builds for the two days plan bit-identical powers from the same views (those of the first
   day's run). Replayed each on its own day, the two runs plan bit-identical powers in every hour
   up to the first whose view differs; hour 0's views are equal on every day, and on some days a
   later view differs.
2. The centre. The registry builds the controller on the box's centre `m`: its centre is `m`, its
   plans equal those of a controller built on `m` directly, and on some of these days they differ
   from those of a controller built on the lower corner `m - beta w`.
"""

import dataclasses
import random

import numpy as np
import pytest
from made_up_offer_days import offer_day

from fleetreserve import compare_methods
from fleetreserve.asa_controller import AsaController
from fleetreserve.compare_methods import MethodInput
from fleetreserve.controller import forecast_belief, hour_zero_target
from fleetreserve.controller_replay import run_controller
from fleetreserve.plans import Plan

BETA = 0.5
SEEN = "hour-start"


def _cases(n_days):
    rng = random.Random(1114)
    paths = np.random.default_rng(1114)
    out = []
    for k in range(n_days):
        day, certificate, _ = offer_day(rng, k)
        if certificate.outcome != "optimal":
            continue
        out.append((day, certificate.policy, paths.uniform(-1, 1, size=day.n_service)))
    return out


CASES = _cases(16)


def _other_truth(day, shift):
    cars = tuple(
        dataclasses.replace(
            car,
            departure=min(max(car.departure + shift, 1), day.n_hours),
            rate=1.5 * car.rate,
            room=car.room / 2,
        )
        for car in day.cars
    )
    return dataclasses.replace(day, cars=cars)


def _inputs(day, policy):
    return MethodInput(day, policy, BETA, Plan.from_policy(policy, BETA))


class Recording:
    def __init__(self, controller):
        self.controller, self.views, self.powers = controller, [], []

    def __call__(self, view):
        power = np.array(self.controller(view))
        self.views.append(view)
        self.powers.append(power)
        return power


def _replay(day, policy, zeta, controller):
    plan = Plan.from_policy(policy, BETA)
    run_controller(
        day,
        controller,
        forecast_belief(day),
        zeta,
        policy.request(zeta, BETA),
        hour_zero_target=hour_zero_target(plan),
        departure_seen=SEEN,
    )
    return controller


def _same_view(a, b) -> bool:
    return (
        a.hour == b.hour
        and a.target == b.target
        and np.array_equal(a.zeta_known, b.zeta_known)
        and np.array_equal(a.requests_known, b.requests_known)
        and np.array_equal(a.drawn, b.drawn)
        and np.array_equal(a.plugged, b.plugged)
    )


@pytest.mark.parametrize("shift", [1, -1])
def test_asa_reads_no_true_limit(shift):
    later_views_differ = 0
    for day, policy, zeta in CASES:
        other = _other_truth(day, shift)
        assert other.cars != day.cars and other.forecasts == day.forecasts
        first, second = forecast_belief(day), forecast_belief(other)
        for field in ("departure", "rate", "cap", "need"):
            assert np.array_equal(getattr(first, field), getattr(second, field))
        base = _replay(day, policy, zeta, Recording(compare_methods.asa(_inputs(day, policy))))
        twin = compare_methods.asa(_inputs(other, policy))
        for view, power in zip(base.views, base.powers, strict=True):
            assert np.array_equal(twin(view), power)
        changed = _replay(
            other, policy, zeta, Recording(compare_methods.asa(_inputs(other, policy)))
        )
        assert _same_view(base.views[0], changed.views[0])
        for a, b, p, q in zip(base.views, changed.views, base.powers, changed.powers, strict=True):
            if not _same_view(a, b):
                later_views_differ += 1
                break
            assert np.array_equal(p, q)
    assert later_views_differ >= 3, "the changed truth must change some later view"


def test_asa_is_built_on_the_centre_of_the_box():
    differs = 0
    for day, policy, zeta in CASES:
        registry = compare_methods.asa(_inputs(day, policy))
        assert np.array_equal(registry.centre, policy.m)
        arguments = {"n_cars": len(day.cars), "n_recovery": day.n_recovery, "time_limit": 300.0}
        at_m = _replay(day, policy, zeta, Recording(AsaController(policy.m, **arguments)))
        lower = policy.m - BETA * policy.w
        at_lower = _replay(day, policy, zeta, Recording(AsaController(lower, **arguments)))
        run = _replay(day, policy, zeta, Recording(registry))
        assert all(np.array_equal(p, q) for p, q in zip(run.powers, at_m.powers, strict=True))
        differs += any(
            not np.array_equal(p, q) for p, q in zip(run.powers, at_lower.powers, strict=True)
        )
    assert differs >= 3, "the lower corner must change some plan"
