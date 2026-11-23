"""Properties of the three priority rules and of their replay, on seeded made-up data.

On seeded random hour views (1 to 6 cars, beliefs, energy drawn, plugged-in flags, targets from
below 0 to above the believed total; and as many views whose cars have laxities close together),
for EDF, LLF and smoothed LLF:

- the planned total equals min(max(target, 0), believed total most) to 1e-9;
- every planned power lies in [0, most_i]; a car not counted as plugged in gets exactly 0;
- the view is left unchanged;
- smoothed LLF: the cars strictly inside their limits end the hour with equal laxity (laxity at
  the start of the hour - 1 + power / believed rate) to 1e-9; a car at 0 would end with at least
  that laxity, a car at its most with at most that laxity.

On the seeded made-up offer days of `tests/wp6`, replayed by `controller_replay.run_controller`:

- each view handed to the rule holds the hour's target (hour 0: the hour-zero target; hour t: the
  request y_t), the energy drawn before the hour, the request variables and requests up to the
  hour, and the plugged-in flags of `rungs.present_for_decisions` under the chosen reading;
- causality: changing the request variables from hour k on leaves the planned powers and draws of
  hours 0 to k - 1 bit-identical;
- with the oracle belief every car draws exactly its planned power (the believed most of a car is
  then what the true car can take), so an hour fails only when the planned total misses the request.
"""

import random

import numpy as np
import pytest
from made_up_offer_days import offer_day

from fleetreserve.controller import (
    Belief,
    HourView,
    forecast_belief,
    hour_zero_target,
    oracle_belief,
)
from fleetreserve.controller_replay import run_controller
from fleetreserve.plans import Plan
from fleetreserve.priority_rules import (
    earliest_deadline_first,
    laxity,
    least_laxity_first,
    most,
    smoothed_least_laxity,
)
from fleetreserve.replay import FAILURE_KWH
from fleetreserve.rungs import DEPARTURE_SEEN, present_for_decisions

RULES = (earliest_deadline_first, least_laxity_first, smoothed_least_laxity)
TOTAL_TOLERANCE = 1e-9
N_VIEWS = 600


def _random_view(rng: np.random.Generator) -> HourView:
    cars = int(rng.integers(1, 7))
    n = int(rng.integers(1, 9))
    hour = int(rng.integers(0, n + 1))
    cap = rng.uniform(0, 30, cars)
    belief = Belief(
        departure=rng.uniform(0, 12, cars),
        rate=rng.uniform(0.5, 11, cars),
        cap=cap,
        need=cap * rng.uniform(0, 1, cars),
    )
    drawn = cap * rng.uniform(0, 1.1, cars)
    plugged = rng.uniform(size=cars) < 0.8
    view = HourView(hour, n, 0.0, np.zeros(hour), np.zeros(hour), drawn, plugged, belief)
    total = float(most(view).sum())
    kind = int(rng.integers(0, 4))
    target = (-1.0, 0.0, rng.uniform(0, total), total + rng.uniform(0, 5))[kind]
    return HourView(hour, n, target, np.zeros(hour), np.zeros(hour), drawn, plugged, belief)


def _close_laxity_view(rng: np.random.Generator) -> HourView:
    """A view whose plugged-in cars have laxities within about two hours of each other, so that
    the smoothed rule often leaves several cars strictly inside their limits."""
    cars = int(rng.integers(2, 7))
    n = int(rng.integers(1, 9))
    hour = int(rng.integers(0, n + 1))
    rate = rng.uniform(0.5, 11, cars)
    drawn = rng.uniform(0, 5, cars)
    belief = Belief(
        departure=hour + rng.uniform(1, 3, cars),
        rate=rate,
        cap=drawn + rate * rng.uniform(0.3, 3, cars),
        need=drawn + rate * rng.uniform(0, 1.5, cars),
    )
    plugged = rng.uniform(size=cars) < 0.9
    view = HourView(hour, n, 0.0, np.zeros(hour), np.zeros(hour), drawn, plugged, belief)
    target = rng.uniform(0, float(most(view).sum()))
    return HourView(hour, n, target, np.zeros(hour), np.zeros(hour), drawn, plugged, belief)


VIEWS = [_random_view(np.random.default_rng(seed)) for seed in range(N_VIEWS)]
VIEWS += [_close_laxity_view(np.random.default_rng(10_000 + seed)) for seed in range(N_VIEWS)]


@pytest.mark.parametrize("rule", RULES)
def test_the_planned_total_and_the_bounds_on_random_views(rule):
    inside = 0
    for view in VIEWS:
        limit = most(view)
        before = (view.drawn.copy(), view.plugged.copy(), view.belief.departure.copy())
        power = rule(view)
        assert power.shape == (view.n_cars,)
        expected = min(max(view.target, 0.0), float(limit.sum()))
        assert abs(power.sum() - expected) <= TOTAL_TOLERANCE
        assert np.all(power >= 0) and np.all(power <= limit)
        assert np.all(power[~view.plugged] == 0.0)
        assert np.all(limit[~view.plugged] == 0.0)
        assert np.array_equal(before[0], view.drawn)
        assert np.array_equal(before[1], view.plugged)
        assert np.array_equal(before[2], view.belief.departure)
        inside += bool(0 < expected < limit.sum())
    assert inside >= 100, "the sample must hold targets strictly between 0 and the believed total"


def test_smoothed_llf_ends_free_cars_with_equal_laxity():
    with_two_free = 0
    for view in VIEWS:
        limit = most(view)
        power = smoothed_least_laxity(view)
        after = laxity(view) - 1 + power / view.belief.rate
        live = view.plugged & (limit > 0)
        free = live & (power > TOTAL_TOLERANCE) & (power < limit - TOTAL_TOLERANCE)
        if not free.any():
            continue
        level = after[free].mean()
        assert np.all(np.abs(after[free] - level) <= TOTAL_TOLERANCE)
        at_zero = live & (power <= TOTAL_TOLERANCE) & ~free
        at_most = live & (power >= limit - TOTAL_TOLERANCE) & ~free
        assert np.all(after[at_zero] >= level - TOTAL_TOLERANCE)
        assert np.all(after[at_most] <= level + TOTAL_TOLERANCE)
        with_two_free += int(free.sum()) >= 2
    assert with_two_free >= 50, "the sample must hold hours with two or more free cars"


def _offer_cases(n_days=16):
    rng = random.Random(1114)
    paths = np.random.default_rng(1114)
    out = []
    for k in range(n_days):
        day, certificate, _ = offer_day(rng, k)
        if certificate.outcome != "optimal":
            continue
        zeta = paths.uniform(-1, 1, size=day.n_service)
        out.append((day, certificate.policy, zeta))
    return out


OFFER_CASES = _offer_cases()


def _replay(day, policy, zeta, rule, belief, seen, beta=1.0):
    plan = Plan.from_policy(policy, beta)
    return run_controller(
        day,
        rule,
        belief,
        zeta,
        policy.request(zeta, beta),
        hour_zero_target=hour_zero_target(plan),
        departure_seen=seen,
    )


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
@pytest.mark.parametrize("rule", RULES)
def test_the_replay_hands_each_rule_the_right_view(rule, seen):
    for day, policy, zeta in OFFER_CASES:
        views = []

        def recording(view, views=views):
            views.append(view)
            return rule(view)

        run = _replay(day, policy, zeta, recording, forecast_belief(day), seen)
        requests = policy.request(zeta, 1.0)
        departure = np.array([c.departure for c in day.cars])
        assert [v.hour for v in views] == list(range(day.n_service + 1))
        assert views[0].target == float(policy.g[:, 0].sum())
        for t, view in enumerate(views):
            if t >= 1:
                assert view.target == requests[t - 1]
            assert np.array_equal(view.zeta_known, zeta[:t])
            assert np.array_equal(view.requests_known, requests[:t])
            drawn = run.stored[:, t - 1] if t >= 1 else np.zeros(len(day.cars))
            assert np.array_equal(view.drawn, drawn)
            assert np.array_equal(view.plugged, present_for_decisions(t, departure, seen))
            assert view.n_service == day.n_service


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
@pytest.mark.parametrize("rule", RULES)
def test_causality_a_later_request_leaves_earlier_hours_bit_identical(rule, seen):
    rng = np.random.default_rng(7)
    checked = 0
    for day, policy, zeta in OFFER_CASES:
        base = _replay(day, policy, zeta, rule, forecast_belief(day), seen)
        for k in range(1, day.n_service + 1):
            other = zeta.copy()
            other[k - 1 :] = rng.uniform(-1, 1, size=day.n_service - k + 1)
            changed = _replay(day, policy, other, rule, forecast_belief(day), seen)
            assert np.array_equal(base.planned[:, :k], changed.planned[:, :k])
            assert np.array_equal(base.draws[:, :k], changed.draws[:, :k])
            checked += not np.array_equal(base.planned[:, k], changed.planned[:, k])
    assert checked >= 5, "a changed request must change some later plan, or the test sees nothing"


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
@pytest.mark.parametrize("rule", RULES)
def test_with_the_oracle_belief_every_car_draws_its_plan(rule, seen):
    failing = 0
    for day, policy, zeta in OFFER_CASES:
        for beta in (0.5, 1.0, 1.5):
            run = _replay(day, policy, zeta, rule, oracle_belief(day), seen, beta)
            assert np.array_equal(run.draws, run.planned)
            requests = policy.request(zeta, beta)
            missed = np.abs(run.planned[:, 1:].sum(axis=0) - requests) > FAILURE_KWH
            assert np.array_equal(run.failed, missed)
            failing += run.first_failure is not None
    assert failing >= 1, "the sample must hold a failing hour"
