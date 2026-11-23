"""The mixed family of one day (`centre_scale_day`): the front-loaded plan, the mixed policy and its
replay by the fixed plan and the one-way rung.

1. The hand day: one car (true departure hour 2, rate 2, room 4) whose forecast bound is hour 3
   (rate 2, cap 5, need 4), two service hours, no recovery hour, rho 0. The certified policy plans
   2, 1 and 1.5 kW in hours 0 to 2, with a gain of 0.5 on the request variable of hour 2. The
   front-loaded plan is 2, 2, 1 (full forecast power until the cap 5 is reached). The car is gone
   in hour 2, so it ends with what it drew in hours 0 and 1: unmixed 2 + 1 = 3 of 4, driver loss
   0.25 at every scale; mixed 2 + (2 - beta) = 4 - beta, driver loss beta / 4, so 0.0125 at 0.05
   and 0.25 at 1. Hour 2 still carries a request of at least 1 kW for a car that is gone, so both
   fail there: slot loss 0.5 for both, mixed and unmixed, at both scales.
2. Fifty seeded made-up days with departure scenarios: the front-loaded plan meets every forecast
   row and every driver-risk row at every dial of the plan (and without rows), within the
   tolerance; on the certified policy at three dials, the mixed policy's worst case meets every row
   at every grid scale up to 1, it shares out its box exactly, and at scale 1 its arrays are the
   certified policy's.
3. A day whose rows the front-loaded plan does not meet (the hand day with hour 1's power limit
   lowered to 1.5 kW) is reported outside the set and replayed unmixed.
4. At scale 1 the mixed replay equals the gate's replay (`gate_day.evaluate`) of the fixed plan and
   the one-way rung on 24 made-up days, path by path; at 0.5 it differs on some of them.
5. The mix is not defined above scale 1: the policy and the replay refuse it.

Room factors (the weight `min(1, c beta)` on the certified plan, widths `beta` x the certified):

6. The room hand case: two cars, one service hour, no recovery hour, rho 0, a departure seen at
   the rounded hour. Both cars are forecast to stay to hour 2 (rate 2, cap 2, need 0.5); car 0
   truly leaves at hour 1, before that bound; car 1 stays (true rate 2, room 2). The certified
   policy plans 0.5 kW in each hour for each car, with a gain of 0.5 on the request of hour 1, so
   m = 1 and w = 1. The front-loaded plan is (2, 0) for each car (the cap is reached in hour 0).
   At beta = 0.05:
   - factor 1, weight 0.05: centres 0.95 (2, 0) + 0.05 (0.5, 0.5) = (1.925, 0.025) per car, box
     centre 0.95 x 0 + 0.05 x 1 = 0.05; on the path zeta = +1 the ask is 0.05 + 0.05 = 0.1. After
     hour 0 each car holds 1.925, so car 1's remaining room is 2 - 1.925 = 0.075. Car 0 is gone,
     car 1 alone must take 0.1 > 0.075: the one-way re-solve is infeasible, the fixed plan stays,
     car 1 draws 0.025 + 0.025 = 0.05 and the hour fails (slot loss 1 on both rungs). On zeta = -1
     the ask is 0, nothing is planned in hour 1, nothing fails and nothing is re-solved.
   - factor 4, weight min(1, 0.2) = 0.2: centres 0.8 (2, 0) + 0.2 (0.5, 0.5) = (1.7, 0.1), box
     centre 0.2; the ask is 0.25 on zeta = +1 and 0.15 on zeta = -1. Car 1's room is 2 - 1.7 =
     0.3 >= 0.25: the re-solve is feasible (optimal) on every path, car 1 takes the whole ask and
     the one-way slot loss is 0; the fixed plan (car 1 draws 0.1 +- 0.025) fails on every path.
   - unmixed: centres (0.5, 0.5), room 1.5 >= 1 + 0.05: one-way slot loss 0.
7. Every row on 200 made-up days: at every dial of the plan and without driver-risk rows, every
   room factor 1, 2, 4 and 8, every grid scale up to 1 and every corner of the requests, each car's
   draw of the mixed policy meets every row the day was certified on, and the fleet's draws add up
   to the request.
8. A factor with c beta >= 1 replays the unmixed policy: on the 24 made-up days, factor 2 at scale
   0.5 equals the gate's replay path by path; factor 1.5 there (weight 0.75) differs on some days.
9. The weight `min(1, c beta)` by hand, and a factor below 1, infinite or not a number refused.
"""

import dataclasses
import datetime
import itertools
import math
import random

import numpy as np
import pytest
from made_up_offer_days import departure_rule, offer_day

from fleetreserve import centre_scale_day, driver_risk, gate_day, products
from fleetreserve.audit_paths import audited_paths
from fleetreserve.carsets import day_forecast_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.forecast_coverage import forecast_set_empty
from fleetreserve.policy import AffinePolicy
from fleetreserve.rungs import RUNGS, run_rungs

DATE = datetime.date(2020, 1, 1)
GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
MIXED_RUNGS = RUNGS[:2]


# --- 1. the hand day -----------------------------------------------------------------------------


def hand_prepared(rows=None) -> gate_day.PreparedDay:
    car = TrueCar(2, 2.0, 4.0)
    forecast = ForecastCar(3.0, 2.0, 5.0, 4.0)
    day = Day("hand", DATE, 2, 0, 0.0, (car,), (forecast,), 0.3)
    rows = day_forecast_rows(day) if rows is None else rows
    found = certify(rows, products.hourly(), "causal", shape_rule=ANY_OPTIMAL)
    assert found.outcome == "optimal"
    gains = np.zeros((1, 3, 2))
    gains[0, 2, 1] = 0.5
    policy = AffinePolicy([1.0, 1.5], [0.0, 0.5], [[2.0, 1.0, 1.5]], gains)
    certificate = dataclasses.replace(found, policy=policy, value=1.0)
    return gate_day.PreparedDay(
        "hand",
        "A",
        DATE.isoformat(),
        1,
        (),
        (0,),
        None,
        "optimal",
        1.0,
        day,
        certificate,
        np.full((1, 3), 3.0),
        audited_paths("hand", DATE, 2),
        5.0,
        1e-6,
        0.0,
        "rounded-hour",
        rows,
    )


def test_the_hand_days_front_loaded_plan():
    prepared = hand_prepared()
    np.testing.assert_array_equal(centre_scale_day.front_loaded_plan(prepared.day), [[2, 2, 1]])
    assert centre_scale_day.front_loaded_inside(prepared)


@pytest.mark.parametrize("rung", MIXED_RUNGS)
def test_the_hand_day_by_hand(rung):
    prepared = hand_prepared()
    for beta in (0.05, 1.0):
        unmixed = gate_day.evaluate(prepared, beta).rungs[rung]
        mixed = centre_scale_day.evaluate(prepared, beta).rungs[rung]
        assert unmixed.driver_loss == pytest.approx(0.25, abs=1e-12)
        assert mixed.driver_loss == pytest.approx(beta * 0.25, abs=1e-12)
        assert unmixed.slot_loss == mixed.slot_loss == 0.5
        assert set(unmixed.first_failures) == set(mixed.first_failures) == {2}
    low = centre_scale_day.evaluate(prepared, 0.05).rungs[rung].driver_loss
    assert low == pytest.approx(0.0125, abs=1e-12)


def test_the_hand_days_mixed_policy():
    prepared = hand_prepared()
    policy = prepared.certificate.policy
    plan = centre_scale_day.front_loaded_plan(prepared.day)
    mixed = centre_scale_day.mixed_policy(policy, plan, 0.25)
    np.testing.assert_allclose(mixed.g, [[2.0, 1.75, 1.125]], atol=1e-15)
    np.testing.assert_allclose(mixed.m, [1.75, 1.125], atol=1e-15)
    np.testing.assert_array_equal(mixed.w, policy.w)
    np.testing.assert_array_equal(mixed.G, policy.G)


# --- 2. fifty made-up days -----------------------------------------------------------------------

LEVELS = driver_risk.levels_up_to(0.3)
MASSES = driver_risk.level_masses(LEVELS)


def _forecast(rng: np.random.Generator, n_hours: int) -> ForecastCar:
    departure = float(rng.uniform(0.5, n_hours))
    rate = float(rng.choice([1.0, 2.0, 3.0, 6.6]))
    cap = float(rng.uniform(0.0, 1.2) * rate * departure)
    need = float(rng.uniform(0.0, 1.0) * cap)
    raw = [None if rng.uniform() < 0.2 else float(rng.uniform(0.0, departure)) for _ in LEVELS]
    raw = [x if x is None else min(x, departure) for x in raw]
    scenarios = driver_risk.departure_scenarios(departure, raw, MASSES)
    return ForecastCar(departure, rate, cap, need, scenarios)


def made_up_days(seed: int, count: int) -> list[Day]:
    """`count` days with 2 to 4 service hours, 1 to 4 cars, departure scenarios at the plan's levels
    up to 0.3, every forecast set not empty."""
    rng = np.random.default_rng(seed)
    days = []
    while len(days) < count:
        n = int(rng.integers(2, 5))
        n_recovery = int(rng.integers(0, 3))
        n_hours = 1 + n + n_recovery
        forecasts, cars = [], []
        for _ in range(int(rng.integers(1, 5))):
            forecasts.append(_forecast(rng, n_hours))
            departure = int(rng.integers(1, n_hours + 1))
            rate = float(rng.choice([1.0, 2.0, 3.0]))
            cars.append(TrueCar(departure, rate, float(rng.uniform(0.0, 1.0) * rate * departure)))
        if any(forecast_set_empty(f, n, n_recovery) for f in forecasts):
            continue
        days.append(Day("made-up", DATE, n, n_recovery, 0.0, tuple(cars), tuple(forecasts), 0.3))
    return days


def _rows(day: Day, dial):
    return day_forecast_rows(day) if dial is None else driver_risk.day_rows(day, dial)


def test_the_front_loaded_plan_meets_every_row_at_every_dial():
    days = made_up_days(11, 50)
    smallest = math.inf
    for day in days:
        plan = centre_scale_day.front_loaded_plan(day)
        for dial in (None, *driver_risk.DIALS):
            slack = _rows(day, dial).slack(plan)
            smallest = min(smallest, float(slack.min()))
    assert smallest >= -centre_scale_day.SET_TOLERANCE
    assert smallest <= 1e-9  # some row is tight: the check is not met by a wide margin alone


def test_the_mixed_policy_meets_every_row_at_every_scale_up_to_1():
    days = made_up_days(12, 50)
    certified = 0
    for k, day in enumerate(days):
        dial = (0.0, 0.02, 0.05)[k % 3]
        rows = _rows(day, dial)
        found = certify(rows, products.hourly(), "causal", shape_rule=ANY_OPTIMAL)
        assert found.outcome == "optimal"
        certified += found.value > 1e-6
        plan = centre_scale_day.front_loaded_plan(day)
        for beta in GRID:
            mixed = centre_scale_day.mixed_policy(found.policy, plan, beta)
            assert np.all(mixed.worst_case(rows, beta) <= rows.bounds + 1e-7), (k, beta)
            # the box is shared out exactly: sum_i g_i = m and sum_i G_i = W on the service hours
            np.testing.assert_allclose(mixed.g[:, 1:].sum(axis=0), mixed.m, atol=1e-9)
            np.testing.assert_allclose(mixed.G.sum(axis=0)[1:], np.diag(found.policy.w), atol=1e-9)
        at_one = centre_scale_day.mixed_policy(found.policy, plan, 1.0)
        for name in ("m", "w", "g", "G"):
            np.testing.assert_array_equal(getattr(at_one, name), getattr(found.policy, name))
    assert certified >= 25  # most days offer a box, so the mix is not checked on zero widths alone


# --- 3. a day outside the set --------------------------------------------------------------------


def _lowered(prepared, hour: int, limit: float):
    """The prepared day's rows with the power limit of `hour` set to `limit`."""
    rows = prepared.rows
    bounds = rows.bounds.copy()
    k = int(np.flatnonzero((rows.kinds == "power-high") & (rows.coefficients[:, hour] == 1))[0])
    bounds[k] = limit
    return dataclasses.replace(rows, bounds=bounds)


def test_a_day_outside_the_set_is_named_and_replayed_unmixed():
    inside = hand_prepared()
    rows = _lowered(inside, 1, 1.5)
    outside = dataclasses.replace(inside, rows=rows)
    assert not centre_scale_day.front_loaded_inside(outside)
    assert centre_scale_day.front_loaded_slack(outside) == pytest.approx(-0.5)
    assert centre_scale_day.front_loaded_slack(inside) >= 0
    for beta in (0.05, 0.5):
        gate = gate_day.evaluate(outside, beta)
        found = centre_scale_day.evaluate(outside, beta)
        for rung in MIXED_RUNGS:
            assert found.rungs[rung].driver_loss == gate.rungs[rung].driver_loss == 0.25
            np.testing.assert_array_equal(found.rungs[rung].finals, gate.rungs[rung].finals)
        assert centre_scale_day.evaluate(inside, beta).rungs["fixed"].driver_loss < 0.25


# --- 4. the identity at scale 1 ------------------------------------------------------------------


def offer_prepared(count: int) -> list[gate_day.PreparedDay]:
    rng = random.Random(5)
    found, k = [], 0
    while len(found) < count:
        day, certificate, _ = offer_day(rng, 2 * k)
        k += 1
        if certificate.outcome != "optimal" or certificate.value <= 1e-6:
            continue
        rule = departure_rule(day, "rounded-hour")
        n = day.n_service
        bounds = np.array([[rule(i, t) for t in range(n + 1)] for i in range(len(day.cars))])
        found.append(
            gate_day.PreparedDay(
                day.unit,
                "A",
                f"{DATE.isoformat()}-{k}",
                len(day.cars),
                (),
                tuple(range(len(day.cars))),
                None,
                "optimal",
                float(certificate.value),
                day,
                certificate,
                bounds,
                audited_paths(f"made-up-{k}", DATE, n),
                5.0,
                1e-6,
                0.0,
                "rounded-hour",
                day_forecast_rows(day),
            )
        )
    return found


@pytest.fixture(scope="module")
def offered():
    return offer_prepared(24)


def _same(a, b) -> bool:
    return (
        a.slot_loss == b.slot_loss
        and a.driver_loss == b.driver_loss
        and a.first_failures == b.first_failures
        and np.array_equal(a.finals, b.finals)
        and a.resolves == b.resolves
        and a.stay == b.stay
    )


def test_at_scale_1_the_mixed_replay_is_the_gates(offered):
    for prepared in offered:
        assert centre_scale_day.front_loaded_inside(prepared)
        gate = gate_day.evaluate(prepared, 1.0)
        found = centre_scale_day.evaluate(prepared, 1.0)
        assert not found.abstained
        for rung in MIXED_RUNGS:
            assert _same(found.rungs[rung], gate.rungs[rung]), (prepared.date, rung)


def test_below_scale_1_the_mixed_replay_differs_on_some_days(offered):
    differs = 0
    for prepared in offered:
        gate = gate_day.evaluate(prepared, 0.5)
        found = centre_scale_day.evaluate(prepared, 0.5)
        differs += not np.array_equal(found.rungs["fixed"].finals, gate.rungs["fixed"].finals)
    assert differs >= 5


def test_an_abstaining_day_and_scale_0_lose_nothing():
    prepared = hand_prepared()
    for found in (
        centre_scale_day.evaluate(prepared, 0.0),
        centre_scale_day.evaluate(dataclasses.replace(prepared, abstain_reason="x"), 0.5),
    ):
        assert found.abstained
        assert all(r.slot_loss == r.driver_loss == 0 for r in found.rungs.values())


# --- 5. above scale 1 ----------------------------------------------------------------------------


def test_the_mix_is_not_defined_above_scale_1():
    prepared = hand_prepared()
    plan = centre_scale_day.front_loaded_plan(prepared.day)
    with pytest.raises(ValueError, match="not defined"):
        centre_scale_day.mixed_policy(prepared.certificate.policy, plan, 1.5)
    with pytest.raises(ValueError, match="not defined"):
        centre_scale_day.evaluate(prepared, 1.5)


# --- 6. the room hand case -----------------------------------------------------------------------

BETA = 0.05


def room_prepared() -> gate_day.PreparedDay:
    """The day of item 6 of the module text."""
    cars = (TrueCar(1, 2.0, 2.0), TrueCar(2, 2.0, 2.0))
    forecasts = (ForecastCar(2.0, 2.0, 2.0, 0.5), ForecastCar(2.0, 2.0, 2.0, 0.5))
    day = Day("room", DATE, 1, 0, 0.0, cars, forecasts, 0.3)
    rows = day_forecast_rows(day)
    found = certify(rows, products.hourly(), "causal", shape_rule=ANY_OPTIMAL)
    assert found.outcome == "optimal"
    gains = np.zeros((2, 2, 1))
    gains[:, 1, 0] = 0.5
    policy = AffinePolicy([1.0], [1.0], [[0.5, 0.5], [0.5, 0.5]], gains)
    assert np.all(policy.worst_case(rows, 1.0) <= rows.bounds + 1e-12)  # in the forecast sets
    certificate = dataclasses.replace(found, policy=policy, value=1.0)
    return gate_day.PreparedDay(
        "room",
        "A",
        DATE.isoformat(),
        2,
        (),
        (0, 1),
        None,
        "optimal",
        1.0,
        day,
        certificate,
        np.full((2, 2), 2.0),
        audited_paths("room", DATE, 1),
        5.0,
        1e-6,
        0.0,
        "rounded-hour",
        rows,
    )


# factor: (weight, each car's centre, box centre, ask on +1 and -1, room after hour 0, re-solve
# feasible on +1, one-way slot loss), all by hand (module text, item 6)
ROOM_HAND = {
    1.0: (0.05, (1.925, 0.025), 0.05, (0.1, 0.0), 0.075, False, 1.0),
    4.0: (0.2, (1.7, 0.1), 0.2, (0.25, 0.15), 0.3, True, 0.0),
}


@pytest.mark.parametrize("factor", sorted(ROOM_HAND))
def test_the_room_hand_case(factor):
    weight, centre, box, asks, room, feasible, one_way = ROOM_HAND[factor]
    prepared = room_prepared()
    plan = centre_scale_day.front_loaded_plan(prepared.day)
    np.testing.assert_array_equal(plan, [[2.0, 0.0], [2.0, 0.0]])
    assert centre_scale_day.front_loaded_inside(prepared)
    assert centre_scale_day.mix_weight(BETA, factor) == pytest.approx(weight, abs=1e-15)
    mixed = centre_scale_day.mixed_policy(prepared.certificate.policy, plan, weight)
    np.testing.assert_allclose(mixed.g, [centre, centre], atol=1e-12)
    np.testing.assert_allclose(mixed.m, [box], atol=1e-12)
    for zeta, ask in zip((1.0, -1.0), asks, strict=True):
        assert mixed.request([zeta], BETA)[0] == pytest.approx(ask, abs=1e-12)
        runs = run_rungs(
            prepared.day,
            mixed,
            BETA,
            [zeta],
            departure_bound=lambda car, hour: 2.0,
            departure_seen="rounded-hour",
            last="one-way",
        )
        held = runs["fixed"].run.stored[:, 0]  # each car's energy after hour 0
        np.testing.assert_allclose(2.0 - held, [room, room], atol=1e-12)
        outcomes = [r.outcome for r in runs["one-way"].resolves]
        if zeta > 0 or factor > 1:
            assert outcomes == ["optimal" if feasible else "infeasible"], (zeta, outcomes)
        else:
            assert outcomes == []  # ask 0, nothing planned, nothing fails
    found = centre_scale_day.evaluate(prepared, BETA, factor)
    assert found.rungs["fixed"].slot_loss == 1.0
    assert found.rungs["one-way"].slot_loss == one_way
    ups = int((prepared.paths[:, 0] > 0).sum())
    expected = {"optimal": len(prepared.paths)} if feasible else {"infeasible": ups}
    assert found.rungs["one-way"].resolves == expected
    assert gate_day.evaluate(prepared, BETA).rungs["one-way"].slot_loss == 0.0  # unmixed


# --- 7. every row at every corner, scale, dial and factor ----------------------------------------

FACTORS = (1.0, 2.0, 4.0, 8.0)


def _corners(n: int) -> np.ndarray:
    return np.array(list(itertools.product((-1.0, 1.0), repeat=n)))


def test_every_row_holds_at_every_corner_scale_dial_and_factor():
    days = made_up_days(13, 200)
    smallest, checked, offered, inside = math.inf, 0, 0, 0
    for k, day in enumerate(days):
        plan = centre_scale_day.front_loaded_plan(day)
        corners = _corners(day.n_service)
        for dial in (None, *driver_risk.DIALS):
            rows = _rows(day, dial)
            if rows.slack(plan).min() < -centre_scale_day.SET_TOLERANCE:
                continue  # replayed unmixed (reading C2): not a mixed policy
            inside += 1
            found = certify(rows, products.hourly(), "causal", shape_rule=ANY_OPTIMAL)
            assert found.outcome == "optimal", (k, dial)
            offered += found.value > 1e-6
            for factor in FACTORS:
                for beta in GRID:
                    weight = centre_scale_day.mix_weight(beta, factor)
                    mixed = centre_scale_day.mixed_policy(found.policy, plan, weight)
                    draws = mixed.powers(corners, beta)  # (corners, cars, n + 1)
                    done = np.einsum("kt,pkt->pk", rows.coefficients, draws[:, rows.car, :])
                    slack = rows.bounds[None, :] - done
                    smallest = min(smallest, float(slack.min()))
                    checked += slack.size
                    np.testing.assert_allclose(
                        draws[:, :, 1:].sum(axis=1), mixed.request(corners, beta), atol=1e-9
                    )
    assert smallest >= -centre_scale_day.SET_TOLERANCE, smallest
    assert smallest <= 1e-9  # some row is tight: the check is not met by a wide margin alone
    assert inside >= 9 * 190 and offered >= 9 * 100  # most day-dials are mixed and offer a box
    assert checked > 1_000_000


# --- 8. a large factor replays the unmixed policy ------------------------------------------------


def test_a_factor_with_c_beta_at_least_1_replays_the_unmixed_policy(offered):
    differs = 0
    for prepared in offered:
        gate = gate_day.evaluate(prepared, 0.5)
        found = centre_scale_day.evaluate(prepared, 0.5, 2.0)
        for rung in MIXED_RUNGS:
            assert _same(found.rungs[rung], gate.rungs[rung]), (prepared.date, rung)
        below = centre_scale_day.evaluate(prepared, 0.5, 1.5)
        differs += not np.array_equal(below.rungs["fixed"].finals, gate.rungs["fixed"].finals)
    assert differs >= 3


# --- 9. the weight and the refusals --------------------------------------------------------------


def test_the_weight_by_hand_and_the_refused_factors():
    assert centre_scale_day.mix_weight(0.05) == 0.05
    assert centre_scale_day.mix_weight(0.05, 4.0) == pytest.approx(0.2, abs=1e-15)
    assert centre_scale_day.mix_weight(0.3, 4.0) == 1.0
    assert centre_scale_day.mix_weight(1.0, 8.0) == 1.0
    prepared = room_prepared()
    for factor in (0.5, 0.999, math.inf, math.nan):
        with pytest.raises(ValueError, match="room factor"):
            centre_scale_day.mix_weight(0.05, factor)
        with pytest.raises(ValueError, match="room factor"):
            centre_scale_day.evaluate(prepared, 0.05, factor)
    with pytest.raises(ValueError, match="not defined"):
        centre_scale_day.mix_weight(1.5, 1.0)
