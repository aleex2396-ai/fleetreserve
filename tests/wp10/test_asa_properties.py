"""Properties of the linear-programme form of ASA (`asa_controller`), replayed on made-up days.

On the seeded made-up offer days of `tests/wp6`, with the day's certified box at scale 1, replayed
by `controller_replay.run_controller` under both readings of `departure_seen` with the forecast
belief (one seeded path per day) and with the oracle belief (the day's audited paths):

- every planned power lies in [0, most_i] of the hour's view, and a car not counted as plugged in
  gets exactly 0;
- the planned total of the hour equals min(max(target, 0), believed total most) to 1e-6 kWh
  (stage 1 may exceed its optimum by `STAGE_TOLERANCE`, 1e-9, and HiGHS's rows hold to its primal
  feasibility tolerance, 1e-7);
- causality: changing the request variables from hour k on leaves the planned powers and draws of
  hours 0 to k - 1 bit-identical. Each hour's programme is built from that hour's view, which is
  the same up to hour k - 1, and HiGHS gives the same answer to the same programme, so the hours
  before k are not only within the solver's tolerance but equal;
- with the oracle belief, on the day-paths at scale 1 where the certified fixed plan meets every
  request: the run of this test's sample showed that the controller meets every request on 150
  of these 152 day-path-readings and fails on 2 (one audited path of one day, under both
  readings), while the least-laxity fill fails on 24. The test pins these counts; it is an
  observation on this sample, not a property of the controller;
- a box whose lower corner is below 0 (each day's centre moved to `w - 0.001` in every hour, so
  that the lower corner at scale 1 is -0.001 everywhere and the centre is below 0 in the hours
  where `w` is under 0.001) is not refused, and on the path `zeta = -1`, whose every request is
  -0.001, and on the seeded path, the controller plans every power in [0, most_i], solves every
  programme to optimality and plans the hour's total min(max(target, 0), believed total most)
  to 1e-6 kWh: 0 to 1e-6 kWh in every hour whose target is below 0 (the later goals may use the
  1e-9 of `STAGE_TOLERANCE` above goal 1's optimum 0, so the total is not always exactly 0).
"""

import random

import numpy as np
import pytest
from made_up_offer_days import offer_day

from fleetreserve.asa_controller import AsaController
from fleetreserve.audit_paths import audited_paths
from fleetreserve.controller import forecast_belief, hour_zero_target, oracle_belief
from fleetreserve.controller_replay import run_controller
from fleetreserve.plans import Plan
from fleetreserve.policy import AffinePolicy
from fleetreserve.priority_rules import least_laxity_first, most
from fleetreserve.replay import run_plan
from fleetreserve.rungs import DEPARTURE_SEEN

TOTAL_TOLERANCE = 1e-6


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


def _asa(day, policy, beta=1.0):
    plan = Plan.from_policy(policy, beta)
    return AsaController.for_box(
        policy, beta, plan, n_cars=len(day.cars), n_recovery=day.n_recovery, time_limit=300.0
    )


def _replay(day, policy, zeta, controller, belief, seen, beta=1.0):
    plan = Plan.from_policy(policy, beta)
    return run_controller(
        day,
        controller,
        belief,
        zeta,
        policy.request(zeta, beta),
        hour_zero_target=hour_zero_target(plan),
        departure_seen=seen,
    )


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
@pytest.mark.parametrize("belief_of", [forecast_belief, oracle_belief])
def test_plans_within_the_most_and_the_hours_total(seen, belief_of):
    checked = inside = 0
    for day, policy, zeta in CASES:
        asa = _asa(day, policy)
        seen_views = []

        def recording(view, asa=asa, seen_views=seen_views):
            power = asa(view)
            seen_views.append((view, np.array(power)))
            return power

        _replay(day, policy, zeta, recording, belief_of(day), seen)
        for view, power in seen_views:
            limit = most(view)
            assert np.all(power >= 0) and np.all(power <= limit)
            assert np.all(power[~view.plugged] == 0.0)
            expected = min(max(view.target, 0.0), float(limit.sum()))
            assert abs(power.sum() - expected) <= TOTAL_TOLERANCE
            inside += bool(0 < expected < limit.sum())
            checked += 1
        assert asa.fallback_hours == 0
    assert checked >= 40 and inside >= 10


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_causality_a_later_request_leaves_earlier_hours_bit_identical(seen):
    rng = np.random.default_rng(7)
    changed_later = 0
    for day, policy, zeta in CASES:
        belief = forecast_belief(day)
        base = _replay(day, policy, zeta, _asa(day, policy), belief, seen)
        for k in range(1, day.n_service + 1):
            other = zeta.copy()
            other[k - 1 :] = rng.uniform(-1, 1, size=day.n_service - k + 1)
            changed = _replay(day, policy, other, _asa(day, policy), belief, seen)
            assert np.array_equal(base.planned[:, :k], changed.planned[:, :k])
            assert np.array_equal(base.draws[:, :k], changed.draws[:, :k])
            changed_later += not np.array_equal(base.planned[:, k], changed.planned[:, k])
    assert changed_later >= 5, "a changed request must change some later plan"


def test_with_the_oracle_belief_where_the_fixed_plan_meets_every_request():
    rng = random.Random(1114)
    met = failed = llf_failed = 0
    for k in range(24):
        day, certificate, _ = offer_day(rng, k)
        if certificate.outcome != "optimal":
            continue
        policy = certificate.policy
        plan = Plan.from_policy(policy, 1.0)
        for zeta in audited_paths(day.unit, day.date, day.n_service):
            if run_plan(day, plan, zeta, policy.request(zeta, 1.0)).first_failure is not None:
                continue
            for seen in DEPARTURE_SEEN:
                belief = oracle_belief(day)
                run = _replay(day, policy, zeta, _asa(day, policy), belief, seen)
                met += run.first_failure is None
                failed += run.first_failure is not None
                llf = _replay(day, policy, zeta, least_laxity_first, belief, seen)
                llf_failed += llf.first_failure is not None
    assert (met, failed, llf_failed) == (150, 2, 24)


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_a_box_whose_lower_corner_is_below_0_plans_nothing_negative(seen):
    negative = checked = 0
    for day, policy, zeta in CASES:
        below = AffinePolicy(policy.w - 0.001, policy.w, policy.g, policy.G)
        assert np.all(below.m - below.w < 0)
        for path in (np.full(day.n_service, -1.0), zeta):
            asa = _asa(day, below)
            seen_views = []

            def recording(view, asa=asa, seen_views=seen_views):
                power = asa(view)
                seen_views.append((view, np.array(power)))
                return power

            _replay(day, below, path, recording, forecast_belief(day), seen)
            for view, power in seen_views:
                limit = most(view)
                assert np.all(power >= 0) and np.all(power <= limit)
                expected = min(max(view.target, 0.0), float(limit.sum()))
                assert abs(power.sum() - expected) <= TOTAL_TOLERANCE
                negative += view.target < 0
                checked += 1
            assert asa.fallback_hours == 0
    assert negative >= 20 and checked >= 80
