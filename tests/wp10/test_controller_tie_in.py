"""The controller replay against `replay.run_plan`: one plan, two runners, the same day-path.

`plan_controller.PlanController` follows a `plans.Plan`. Run by `controller_replay.run_controller`
it must give exactly what `replay.run_plan` gives for the same plan, path and requests: the draws
of every car in every hour, the first failing hour, the slot loss, the driver loss and each car's
energy at departure. The planned powers agree after negative planned powers (which draw nothing
one-way) are read as 0. Exact equality: both runners compute the same numbers in the same order.

Days: the seeded made-up offer days of `tests/wp6` (2 to 4 service hours, 2 to 5 cars, forecasts
conservative, with an early departure, less room or a lower rate), certified on their forecast
sets; the fixed plan of the certified policy at scales 0.5, 1 and 1.5 on seeded paths, under both
readings of when a departure is seen. The test also counts that the sample holds cars that leave
before their forecast departure, cars that take less than planned and failing hours.
"""

import random

import numpy as np
import pytest
from made_up_offer_days import offer_day

from fleetreserve import replay
from fleetreserve.controller import forecast_belief, hour_zero_target
from fleetreserve.controller_replay import run_controller
from fleetreserve.plan_controller import PlanController
from fleetreserve.plans import Plan
from fleetreserve.rungs import DEPARTURE_SEEN

N_DAYS = 24
SCALES = (0.5, 1.0, 1.5)
PATHS_PER_DAY = 3


def _cases():
    rng = random.Random(20261114)
    paths = np.random.default_rng(20261114)
    for k in range(N_DAYS):
        day, certificate, kinds = offer_day(rng, k)
        if certificate.outcome != "optimal":
            continue
        for beta in SCALES:
            for _ in range(PATHS_PER_DAY):
                zeta = paths.choice([-1.0, 1.0], size=day.n_service) * paths.uniform(
                    0.5, 1.0, size=day.n_service
                )
                yield day, certificate.policy, beta, zeta, kinds


CASES = list(_cases())


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_the_plan_controller_replays_exactly_as_run_plan(seen):
    early = took_less = failing = 0
    for day, policy, beta, zeta, kinds in CASES:
        plan = Plan.from_policy(policy, beta)
        requests = policy.request(zeta, beta)
        expected = replay.run_plan(day, plan, zeta, requests)
        got = run_controller(
            day,
            PlanController(plan),
            forecast_belief(day),
            zeta,
            requests,
            hour_zero_target=hour_zero_target(plan),
            departure_seen=seen,
        )
        assert np.array_equal(got.draws, expected.draws)
        assert np.array_equal(got.planned, np.maximum(expected.planned, 0.0))
        assert np.array_equal(got.stored, expected.stored)
        assert np.array_equal(got.delivered, expected.delivered)
        assert np.array_equal(got.failed, expected.failed)
        assert got.first_failure == expected.first_failure
        assert got.slot_loss == expected.slot_loss
        assert got.driver_loss == expected.driver_loss
        assert np.array_equal(got.final, expected.final)
        early += "early departure" in kinds
        took_less += bool(np.any(expected.draws < np.maximum(expected.planned, 0) - 1e-9))
        failing += expected.first_failure is not None
    assert len(CASES) >= 150
    assert early >= 10, "the sample must hold cars that leave before their forecast"
    assert took_less >= 10, "the sample must hold cars that take less than planned"
    assert failing >= 10, "the sample must hold failing hours"
