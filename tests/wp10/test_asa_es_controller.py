"""The controller of `asa-es` (`asa_controller.AsaEsController`) and its registry entry
(`compare_methods.asa_es`).

1. Fall-back. A solver stub that ends one stage of one hour in each outcome but "optimal" (each
   of the five stages in turn): the controller plans that hour by the least-laxity fill of the
   same view, solves no later stage of that hour, counts the outcome, the hour and the fall-back
   hour, and plans the other hours by its programme (`asa_equal_share.plan_hour`). The views are
   those of `test_asa_fallback`, where the least-laxity fill and the programme plan differently.
2. Every solve, five per hour, gets the time limit the controller was built with; the registry's
   `asa-es` builds an `AsaEsController` with `lp.OTHER_SECONDS`, whose programme is
   `asa_equal_share.plan_hour` (not `asa`'s), and `compare_day.evaluate` carries its counts, five
   optimal solves per hour, beside `asa`'s four, over the same hours.
3. No leak of the true limits and no look ahead, on the seeded made-up offer days of `tests/wp6`
   with the day's certified box at the scale 0.5, replayed with the forecast belief: a second day
   equal to the first but for its true cars (every departure one hour later where the day's hours
   allow it, every rate 1.5 times, every room halved), with the same forecasts, gets
   bit-identical plans from the registry's controller on the first day's views; and changing the
   request variables from hour k on leaves the plans and draws of hours 0 to k - 1 bit-identical.
4. The centre: the registry builds it on the box's centre `m` (its plans equal those of a
   controller built on `m`).
"""

import dataclasses
import random

import numpy as np
import pytest
from compare_fake import offer_days
from made_up_offer_days import offer_day

from fleetreserve import asa_equal_share, compare_day, compare_methods, lp
from fleetreserve.asa_controller import OUTCOMES, AsaController, AsaEsController
from fleetreserve.compare_methods import MethodInput
from fleetreserve.controller import Belief, HourView, forecast_belief, hour_zero_target
from fleetreserve.controller_replay import run_controller
from fleetreserve.lp import LpResult
from fleetreserve.plans import Plan
from fleetreserve.priority_rules import least_laxity_first

STAGES = asa_equal_share.STAGES
NOT_OPTIMAL = [o for o in OUTCOMES if o != "optimal"]
REAL = lp.solve_lp
BETA = 0.5
SEEN = "hour-start"


def _views():
    belief = Belief([1.5, 3, 3], [2, 1, 1], [10, 10, 10], [0.6, 1.9, 0])
    return [
        HourView(t, 2, 1.1, np.zeros(t), np.full(t, 1.1), np.zeros(3), np.ones(3, bool), belief)
        for t in range(3)
    ]


class Stub:
    """Ends stage `stage` of the `hour`-th programme in `outcome`; solves the rest for real.
    Records the time limit of every call."""

    def __init__(self, hour, stage, outcome):
        self.hour, self.stage, self.outcome = hour, stage, outcome
        self.calls, self.limits = 0, []

    def __call__(self, c, **arguments):
        self.limits.append(arguments["time_limit"])
        hour, stage = divmod(self.calls, len(STAGES))
        self.calls += 1
        if (hour, stage) == (self.hour, self.stage):
            limit = arguments["time_limit"]
            return LpResult(self.outcome, None, None, 0.0, limit, 99, "made-up outcome")
        return REAL(c, **arguments)


@pytest.mark.parametrize("stage", range(len(STAGES)))
@pytest.mark.parametrize("outcome", NOT_OPTIMAL)
def test_a_stage_that_is_not_optimal_falls_back_to_the_least_laxity_fill(outcome, stage):
    stub = Stub(1, stage, outcome)
    controller = AsaEsController([0.0, 0.0], n_cars=3, n_recovery=0, time_limit=7.5, solve=stub)
    reference = AsaEsController([0.0, 0.0], n_cars=3, n_recovery=0, time_limit=7.5)
    for t, view in enumerate(_views()):
        power = controller(view)
        if t == 1:
            assert np.array_equal(power, least_laxity_first(view))
            assert not np.allclose(power, reference(view))
        else:
            assert np.array_equal(power, reference(view))
    counts = controller.solver_counts()
    assert counts["hours"] == 3 and counts["fallback_hours"] == 1
    assert counts["solves"][outcome] == 1
    assert counts["solves"]["optimal"] == 2 * len(STAGES) + stage
    assert sum(counts["solves"].values()) == stub.calls == 2 * len(STAGES) + stage + 1


def test_every_solve_gets_the_controllers_time_limit():
    stub = Stub(-1, -1, "timeout")
    controller = AsaEsController([0.0, 0.0], n_cars=3, n_recovery=0, time_limit=7.5, solve=stub)
    for view in _views():
        controller(view)
    assert stub.limits == [7.5] * (3 * 5)
    assert controller.solver_counts()["solves"]["optimal"] == 3 * 5


def test_the_registry_builds_asa_es_with_its_own_programme(monkeypatch):
    limits = []

    def recording(c, **arguments):
        limits.append(arguments["time_limit"])
        return REAL(c, **arguments)

    monkeypatch.setattr(lp, "solve_lp", recording)
    (prepared,) = offer_days(1)
    policy = prepared.certificate.policy
    inputs = MethodInput(prepared.day, policy, 0.5, Plan.from_policy(policy, 0.5))
    built = compare_methods.asa_es(inputs)
    assert type(built) is AsaEsController and built.programme is asa_equal_share.plan_hour
    assert type(compare_methods.asa(inputs)) is AsaController
    found = compare_day.evaluate(prepared, 1.0, compare_methods.select(["asa", "asa-es"]))
    assert limits and set(limits) == {lp.OTHER_SECONDS}
    hours = len(prepared.paths) * (prepared.day.n_service + 1)
    for name, stages in (("asa", 4), ("asa-es", 5)):
        counts = found[name].solver_counts
        assert counts["hours"] == hours and counts["fallback_hours"] == 0
        assert counts["solves"] == {**dict.fromkeys(OUTCOMES, 0), "optimal": hours * stages}


# --- 3. and 4. on replayed made-up days --------------------------------------------------------


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


CASES = _cases(10)


class Recording:
    def __init__(self, controller):
        self.controller, self.views, self.powers = controller, [], []

    def __call__(self, view):
        power = np.array(self.controller(view))
        self.views.append(view)
        self.powers.append(power)
        return power


def _inputs(day, policy):
    return MethodInput(day, policy, BETA, Plan.from_policy(policy, BETA))


def _replay(day, policy, zeta, controller):
    plan = Plan.from_policy(policy, BETA)
    return run_controller(
        day,
        controller,
        forecast_belief(day),
        zeta,
        policy.request(zeta, BETA),
        hour_zero_target=hour_zero_target(plan),
        departure_seen=SEEN,
    )


def _other_truth(day):
    cars = tuple(
        dataclasses.replace(
            car,
            departure=min(car.departure + 1, day.n_hours),
            rate=1.5 * car.rate,
            room=car.room / 2,
        )
        for car in day.cars
    )
    return dataclasses.replace(day, cars=cars)


def test_asa_es_reads_no_true_limit():
    for day, policy, zeta in CASES:
        other = _other_truth(day)
        assert other.cars != day.cars and other.forecasts == day.forecasts
        base = Recording(compare_methods.asa_es(_inputs(day, policy)))
        _replay(day, policy, zeta, base)
        twin = compare_methods.asa_es(_inputs(other, policy))
        for view, power in zip(base.views, base.powers, strict=True):
            assert np.array_equal(twin(view), power)


def test_a_later_request_leaves_earlier_hours_bit_identical():
    rng = np.random.default_rng(7)
    changed_later = 0
    for day, policy, zeta in CASES:
        base = _replay(day, policy, zeta, compare_methods.asa_es(_inputs(day, policy)))
        for k in range(1, day.n_service + 1):
            other = zeta.copy()
            other[k - 1 :] = rng.uniform(-1, 1, size=day.n_service - k + 1)
            changed = _replay(day, policy, other, compare_methods.asa_es(_inputs(day, policy)))
            assert np.array_equal(base.planned[:, :k], changed.planned[:, :k])
            assert np.array_equal(base.draws[:, :k], changed.draws[:, :k])
            changed_later += not np.array_equal(base.planned[:, k], changed.planned[:, k])
    assert changed_later >= 5, "a changed request must change some later plan"


def test_asa_es_is_built_on_the_centre_of_the_box():
    for day, policy, zeta in CASES:
        registry = compare_methods.asa_es(_inputs(day, policy))
        assert np.array_equal(registry.centre, policy.m)
        arguments = {"n_cars": len(day.cars), "n_recovery": day.n_recovery, "time_limit": 300.0}
        at_m = Recording(AsaEsController(policy.m, **arguments))
        _replay(day, policy, zeta, at_m)
        run = Recording(registry)
        _replay(day, policy, zeta, run)
        assert all(np.array_equal(p, q) for p, q in zip(run.powers, at_m.powers, strict=True))
