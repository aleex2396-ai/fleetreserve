"""Properties of the diagonal split (`diagonal_split.split_box`) on made-up days.

The days: the first 40 seeded made-up offer days of `compare_fake.offer_days` (seed 5), each box
(the certified causal policy's centre and half-widths) at the scales 0.2, 0.6 and 1. On these
boxes the programme ends optimal every time; at scale 1 two boxes have no diagonal split
(asserted, so the checks below see both kinds).

1. The gains are diagonal: an hour's planned powers do not change when the request variable of
   another hour changes (random request variables, every hour, every other hour), and the planned
   powers of a service hour add up to that hour's request `m_t + beta w_t zeta_t` (to 1e-9) for
   the kinds "split" and "stretched".
2. Where the box has a diagonal split, every car's planned powers lie in its forecast set (plan
   section 2, checked here from the car's own fields, not from the programme's rows) at every
   corner of the box; where it has none (kind "stretched"), some car leaves its forecast set
   at some corner, so the corner check can fail.
3. The same with the driver-risk rows: on made-up days whose cars carry departure scenarios, the
   box certified on `driver_risk.day_rows` at the dial 0.02, split on those rows; at every corner
   of a box with a diagonal split every car is in its forecast set and
   `sum_k pi_k max{need_k - E_k(u), 0} <= 0.02 * need`, with `E_k` and `need_k` computed here from
   the plan's text (`need_k` the most `E_k` can be on the forecast set, by a linear programme
   built here from the car's fields).
4. Homothets (Zhao et al. (2017), their equation 14: car `i` takes the fixed share `s_i` of every
   hour's deviation, `G_i(t, t) = s_i w_t`, about its own centre split `g_i`): where some
   homothet split of the box holds every car's forecast set at every `zeta` (a linear programme in
   `g` and `s` built here from the cars' fields), the programme finds a diagonal split
   (`largest_scale >= 1 - SCALE_TOLERANCE`, kind "split"). On these boxes some homothet split
   exists on at least 30 of them.
"""

import dataclasses
import random

import numpy as np
import pytest
from compare_fake import offer_days
from scipy.optimize import linprog

from fleetreserve import diagonal_split, driver_risk, products
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.lp import BOX_SECONDS
from fleetreserve.products import corner_paths

TOLERANCE = 1e-7
SCALES = (0.2, 0.6, 1.0)
DIAL = 0.02


@pytest.fixture(scope="module")
def boxes():
    found = []
    for prepared in offer_days(40, seed=5):
        policy = prepared.certificate.policy
        for beta in SCALES:
            split = diagonal_split.split_box(
                prepared.rows, policy.m, beta * policy.w, time_limit=BOX_SECONDS
            )
            found.append((prepared, beta, split))
    return found


def test_every_programme_ends_optimal_and_both_kinds_occur(boxes):
    assert all(split.outcome == "optimal" for _, _, split in boxes)
    at_one = [split.kind for _, beta, split in boxes if beta == 1.0]
    assert at_one.count(diagonal_split.STRETCHED) == 2
    assert at_one.count(diagonal_split.SPLIT) == 38


def _powers(plan, zeta):
    return np.stack([plan.powers(t, zeta) for t in range(plan.n_service + 1)], axis=1)


def test_an_hour_does_not_depend_on_another_hours_request_and_adds_up_to_its_own(boxes):
    rng = np.random.default_rng(3)
    for prepared, beta, split in boxes:
        policy, plan, n = prepared.certificate.policy, split.plan, split.plan.n_service
        for _ in range(3):
            zeta = rng.uniform(-1, 1, n)
            powers = _powers(plan, zeta)
            for s in range(n):
                other = zeta.copy()
                other[s] = rng.uniform(-1, 1)
                moved = _powers(plan, other)
                for t in range(1, n + 1):
                    if t != s + 1:
                        assert np.array_equal(moved[:, t], powers[:, t])
            request = policy.request(zeta, beta)
            assert powers[:, 1:].sum(axis=0) == pytest.approx(request, abs=1e-9)


def in_forecast_set(car, u, n_service, n_recovery) -> bool:
    hours = range(n_service + 1)
    power = all(-TOLERANCE <= u[t] <= car.power_cap(t) + TOLERANCE for t in hours)
    stored = all(sum(u[: t + 1]) <= car.cap + TOLERANCE for t in hours)
    need = sum(u) + car.recovery_supply(n_service, n_recovery) >= car.need - TOLERANCE
    return power and stored and need


def _corners_in_sets(day, plan) -> bool:
    for zeta in corner_paths(day.n_service):
        powers = _powers(plan, zeta)
        for i, car in enumerate(day.forecasts):
            if not in_forecast_set(car, powers[i], day.n_service, day.n_recovery):
                return False
    return True


def test_a_box_with_a_diagonal_split_keeps_every_car_in_its_forecast_set_at_every_corner(boxes):
    checked = 0
    for prepared, _, split in boxes:
        if split.kind == diagonal_split.SPLIT:
            assert _corners_in_sets(prepared.day, split.plan)
            checked += 1
    assert checked >= 100


def test_a_stretched_plan_leaves_some_forecast_set_at_some_corner(boxes):
    stretched = [(p, s) for p, _, s in boxes if s.kind == diagonal_split.STRETCHED]
    assert stretched
    for prepared, split in stretched:
        assert not _corners_in_sets(prepared.day, split.plan)


# --- 3. with the driver-risk rows -----------------------------------------------------------------


def _with_scenarios(day, rng):
    cars = []
    for car in day.forecasts:
        low = rng.uniform(0.3, 1.0) * car.departure
        cars.append(
            dataclasses.replace(car, scenarios=((low, 0.05), ((low + car.departure) / 2, 0.05)))
        )
    return dataclasses.replace(day, forecasts=tuple(cars))


def _most_energy(car, weights, supply, n_service, n_recovery) -> float:
    """The most `E_k` on the car's forecast set, by a linear programme over hours 0 to n."""
    hours = n_service + 1
    prefix = np.tril(np.ones((hours, hours)))
    a_ub = np.vstack([prefix, -np.ones((1, hours))])
    b_ub = np.concatenate([np.full(hours, car.cap), [supply_need(car, n_service, n_recovery)]])
    bounds = [(0.0, car.power_cap(t)) for t in range(hours)]
    found = linprog(-weights, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    assert found.status == 0
    return -found.fun + supply


def supply_need(car, n_service, n_recovery) -> float:
    return car.recovery_supply(n_service, n_recovery) - car.need


def _shortfall(car, u, n_service, n_recovery) -> float:
    total = 0.0
    for bound, mass in car.scenarios:
        at = dataclasses.replace(car, departure=bound, scenarios=())
        weights = np.array(
            [
                1.0 if car.share(t) <= 0 else min(1.0, at.share(t) / car.share(t))
                for t in range(n_service + 1)
            ]
        )
        supply = at.recovery_supply(n_service, n_recovery)
        need_k = min(car.need, _most_energy(car, weights, supply, n_service, n_recovery))
        total += mass * max(need_k - (weights @ u + supply), 0.0)
    return total


def test_with_driver_risk_rows_a_diagonal_split_holds_them_at_every_corner():
    rng = random.Random(11)
    checked = 0
    for prepared in offer_days(30, seed=7):
        day = _with_scenarios(prepared.day, rng)
        rows = driver_risk.day_rows(day, DIAL)
        certificate = certify(rows, products.hourly(), "causal", shape_rule=ANY_OPTIMAL)
        if certificate.outcome != "optimal" or certificate.value <= 1e-6:
            continue
        policy = certificate.policy
        split = diagonal_split.split_box(rows, policy.m, policy.w, time_limit=BOX_SECONDS)
        if split.kind != diagonal_split.SPLIT:
            continue
        checked += 1
        for zeta in corner_paths(day.n_service):
            powers = _powers(split.plan, zeta)
            for i, car in enumerate(day.forecasts):
                assert in_forecast_set(car, powers[i], day.n_service, day.n_recovery)
                shortfall = _shortfall(car, powers[i], day.n_service, day.n_recovery)
                assert shortfall <= DIAL * car.need + 1e-6
    assert checked >= 15


# --- 4. homothets -------------------------------------------------------------------------------


def _homothet_exists(day, centre, widths) -> bool:
    """Some `(g, s)`, `s >= 0`, `sum s = 1`, `sum_i g_i(t) = m_t`, whose split
    `u_i = g_i + s_i W zeta` keeps every car in its forecast set at every `zeta`: with `s_i >= 0`
    the worst case of `u_i(t)` is `g_i(t) +- s_i w_t` and of a running sum
    `sum g_i +- s_i sum w`."""
    n, cars = day.n_service, len(day.forecasts)
    hours = n + 1
    w = np.concatenate([[0.0], widths])  # hour 0 has no request
    size = cars * hours + cars  # g (car, hour), then s (car)
    a_ub, b_ub = [], []
    for i, car in enumerate(day.forecasts):
        for t in range(hours):
            row = np.zeros(size)  # g + s w <= cap of the hour
            row[i * hours + t], row[cars * hours + i] = 1.0, w[t]
            a_ub.append(row)
            b_ub.append(car.power_cap(t))
            row = np.zeros(size)  # -(g - s w) <= 0
            row[i * hours + t], row[cars * hours + i] = -1.0, w[t]
            a_ub.append(row)
            b_ub.append(0.0)
            row = np.zeros(size)  # running sum, upper corner, <= cap
            row[i * hours : i * hours + t + 1] = 1.0
            row[cars * hours + i] = w[: t + 1].sum()
            a_ub.append(row)
            b_ub.append(car.cap)
        row = np.zeros(size)  # window energy, lower corner, + supply >= need
        row[i * hours : (i + 1) * hours] = -1.0
        row[cars * hours + i] = w.sum()
        a_ub.append(row)
        b_ub.append(supply_need(car, n, day.n_recovery))
    a_eq, b_eq = [], []
    for t in range(1, hours):
        row = np.zeros(size)
        row[[i * hours + t for i in range(cars)]] = 1.0
        a_eq.append(row)
        b_eq.append(centre[t - 1])
    row = np.zeros(size)
    row[cars * hours :] = 1.0
    a_eq.append(row)
    b_eq.append(1.0)
    bounds = [(None, None)] * (cars * hours) + [(0.0, None)] * cars
    found = linprog(
        np.zeros(size), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    return found.status == 0


def test_where_a_homothet_split_exists_the_programme_finds_a_diagonal_split(boxes):
    feasible = 0
    for prepared, beta, split in boxes:
        policy = prepared.certificate.policy
        if _homothet_exists(prepared.day, policy.m, beta * policy.w):
            feasible += 1
            assert split.largest_scale >= 1.0 - diagonal_split.SCALE_TOLERANCE
            assert split.kind == diagonal_split.SPLIT
    assert feasible >= 30
