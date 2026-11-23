"""The diagonal split (`diagonal_split`) on two made-up days built by hand, with every number worked
out here from the rows.

1. Day A, forecast cars: two service hours, no recovery hour, both cars plugged in for hours 0 to
   2. Car A: rate 1, cap 2, need 0. Car B: rate 2, cap 1, need 1, so it draws exactly 1 in hours
   0 to 2. Box: centre (1, 1), half-widths (1, 0). It has a causal split (written out below and
   checked at all four corners): each car takes half of hour 1's deviation and car B gives its half
   back in hour 2, where car A takes it. A diagonal split cannot give car B any gain on `zeta_1`
   (its total would move with `zeta_1`), so car A takes all of it, `g_A(1) +- lam` within
   `[0, 1]`: `lam* = 1/2`, `g_A(1) = 1/2`, `g_B(1) = 1 - 1/2`.
   - Scale 1: no diagonal split (kind "stretched", D2); the plan gives car A the gain 1 on
     `zeta_1` (1/2 scaled up to the half-width 1), car B none, no gain in hour 2 (half-width 0),
     and the centre split 1/2 and 1/2 in hour 1. On every corner path the replay fails in hour 1:
     at `zeta_1 = +1` car A is planned 1.5 and draws its rate 1, so 1.5 is delivered against 2; at
     `zeta_1 = -1` car A is planned -0.5, read as 0, so 0.5 is delivered against 0. The slot loss
     is 1 (both service hours from the first failure on). The causal fixed plan on the same box
     meets every request on the same paths.
   - Scale 0.4: a diagonal split (kind "split", D1), `lam* = 0.5 / 0.4 = 1.25`, car A's gain 0.4;
     no failure on any corner path.
   - Through the registry (`compare_methods.select(["diagonal-split"])`, a `MethodInput` with the
     day's forecast rows and the causal policy): the same kinds and gains at scales 1 and 0.4, and
     the controller counts one box of that kind.
2. Day C, rows by hand: two service hours. Car A: power 0 to 1 in hours 0 to 2, stored energy at
   most 1, need 1 (exactly 1 in hours 0 to 2). Car C: power 0 in hours 0 and 1, 0 to 2 in hour 2,
   stored energy at most 2, need 0. Box: centre (0.5, 1), half-widths (0.5, 0). Causal split: car
   A takes `0.5 + 0.5 zeta_1` in hour 1 and gives it back in hour 2, where car C takes
   `0.5 + 0.5 zeta_1`. Diagonally only car A can take `zeta_1` (car C has power 0 in hour 1) and
   its total may not move: `lam* = 0` (kind "no gains", D3); the plan has no gain, the centres of
   hour 1 are 0.5 (car A) and 0 (car C), and those of hour 2 add up to 1; each hour plans its
   centre whatever the request.
"""

import datetime

import numpy as np
import pytest

from fleetreserve import compare_methods, diagonal_split
from fleetreserve.carsets import CarRows, day_forecast_rows, stack
from fleetreserve.compare_methods import MethodInput
from fleetreserve.controller import forecast_belief
from fleetreserve.controller_replay import run_controller
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.lp import BOX_SECONDS
from fleetreserve.plan_controller import PlanController
from fleetreserve.plans import Plan
from fleetreserve.policy import AffinePolicy
from fleetreserve.products import corner_paths

TOLERANCE = 1e-7
DATE = datetime.date(2026, 1, 5)
CENTRE = np.array([1.0, 1.0])
HALF_WIDTHS = np.array([1.0, 0.0])


def day_a() -> Day:
    cars = (TrueCar(3, 1.0, 2.0), TrueCar(3, 2.0, 1.0))
    forecasts = (ForecastCar(3.0, 1.0, 2.0, 0.0), ForecastCar(3.0, 2.0, 1.0, 1.0))
    return Day("hand", DATE, 2, 0, 0.0, cars, forecasts, 0.1)


def causal_policy() -> AffinePolicy:
    g = np.array([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])
    gains = np.zeros((2, 3, 2))
    gains[0, 1, 0], gains[0, 2, 0] = 0.5, 0.5
    gains[1, 1, 0], gains[1, 2, 0] = 0.5, -0.5
    return AffinePolicy(CENTRE, HALF_WIDTHS, g, gains)


def in_forecast_set(car: ForecastCar, u, n_service: int, n_recovery: int) -> bool:
    """The forecast set of plan section 2, from the car's own fields."""
    hours = range(n_service + 1)
    power = all(-TOLERANCE <= u[t] <= car.power_cap(t) + TOLERANCE for t in hours)
    stored = all(sum(u[: t + 1]) <= car.cap + TOLERANCE for t in hours)
    need = sum(u) + car.recovery_supply(n_service, n_recovery) >= car.need - TOLERANCE
    return power and stored and need


def test_day_a_has_a_causal_split():
    day, policy = day_a(), causal_policy()
    for zeta in corner_paths(2):
        powers = policy.powers(zeta)
        assert np.allclose(powers[:, 1:].sum(axis=0), policy.request(zeta))
        assert all(in_forecast_set(car, powers[i], 2, 0) for i, car in enumerate(day.forecasts))


def test_day_a_at_scale_1_has_no_diagonal_split_and_stretches_the_split_of_half_the_box():
    found = diagonal_split.split_box(
        day_forecast_rows(day_a()), CENTRE, HALF_WIDTHS, time_limit=BOX_SECONDS
    )
    assert found.outcome == "optimal"
    assert found.kind == diagonal_split.STRETCHED
    assert found.largest_scale == pytest.approx(0.5, abs=TOLERANCE)
    plan = found.plan
    assert plan.gains[:, 1, 0] == pytest.approx([1.0, 0.0], abs=TOLERANCE)
    assert np.all(plan.gains[:, 2, :] == 0) and np.all(plan.gains[:, 1, 1] == 0)
    assert plan.constant[:, 1] == pytest.approx([0.5, 0.5], abs=TOLERANCE)
    assert plan.constant[:, 2].sum() == pytest.approx(1.0, abs=TOLERANCE)


def _replay(day, controller, zeta, beta):
    policy = causal_policy()
    return run_controller(
        day,
        controller,
        forecast_belief(day),
        zeta,
        policy.request(zeta, beta),
        hour_zero_target=0.0,
        departure_seen="hour-start",
    )


@pytest.mark.parametrize("zeta", [tuple(z) for z in corner_paths(2)])
def test_day_a_at_scale_1_the_stretched_plan_fails_in_hour_1_and_the_causal_plan_does_not(zeta):
    day = day_a()
    found = diagonal_split.split_box(
        day_forecast_rows(day), CENTRE, HALF_WIDTHS, time_limit=BOX_SECONDS
    )
    run = _replay(day, diagonal_split.DiagonalSplitController(found, counted=True), zeta, 1.0)
    assert run.first_failure == 1
    assert run.slot_loss == pytest.approx(1.0)
    delivered = 1.5 if zeta[0] > 0 else 0.5
    assert run.draws[:, 1].sum() == pytest.approx(delivered, abs=TOLERANCE)
    causal = _replay(day, PlanController(Plan.from_policy(causal_policy(), 1.0)), zeta, 1.0)
    assert causal.first_failure is None and causal.slot_loss == 0.0


@pytest.mark.parametrize("zeta", [tuple(z) for z in corner_paths(2)])
def test_day_a_at_scale_0_4_has_a_diagonal_split_that_meets_every_corner(zeta):
    day = day_a()
    found = diagonal_split.split_box(
        day_forecast_rows(day), CENTRE, 0.4 * HALF_WIDTHS, time_limit=BOX_SECONDS
    )
    assert found.kind == diagonal_split.SPLIT
    assert found.largest_scale == pytest.approx(1.25, abs=TOLERANCE)
    assert found.plan.gains[:, 1, 0] == pytest.approx([0.4, 0.0], abs=TOLERANCE)
    run = _replay(day, diagonal_split.DiagonalSplitController(found, counted=True), zeta, 0.4)
    assert run.first_failure is None


@pytest.mark.parametrize(("beta", "kind", "gain"), [(1.0, "stretched", 1.0), (0.4, "split", 0.4)])
def test_day_a_through_the_registry_splits_the_box_at_the_scale(beta, kind, gain):
    day, policy = day_a(), causal_policy()
    inputs = MethodInput(day, policy, beta, Plan.from_policy(policy, beta), day_forecast_rows(day))
    (method,) = compare_methods.select(["diagonal-split"])
    controller = method.build(inputs)
    assert controller.split.kind == kind
    assert controller.split.plan.gains[:, 1, 0] == pytest.approx([gain, 0.0], abs=TOLERANCE)
    counts = controller.solver_counts()
    assert counts["boxes"] == {k: int(k == kind) for k in diagonal_split.BOXES}


def rows_c():
    eye, prefix = np.eye(3), np.tril(np.ones((3, 3)))
    kinds = ("power-high",) * 3 + ("power-low",) * 3 + ("stored-high",) * 3 + ("need",)

    def car(high, cap, need):
        coefficients = np.vstack([eye, -eye, prefix, -np.ones((1, 3))])
        bounds = np.concatenate([high, np.zeros(3), np.full(3, cap), [-need]])
        return CarRows(2, coefficients, bounds, kinds)

    return stack([car(np.array([1.0, 1.0, 1.0]), 1.0, 1.0), car(np.array([0, 0, 2.0]), 2.0, 0)])


def test_day_c_has_a_causal_split():
    rows = rows_c()
    g = np.array([[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]])
    gains = np.zeros((2, 3, 2))
    gains[0, 1, 0], gains[0, 2, 0], gains[1, 2, 0] = 0.5, -0.5, 0.5
    policy = AffinePolicy(np.array([0.5, 1.0]), np.array([0.5, 0.0]), g, gains)
    for zeta in corner_paths(2):
        powers = policy.powers(zeta)
        assert np.allclose(powers[:, 1:].sum(axis=0), policy.request(zeta))
        assert np.all(rows.slack(powers) >= -TOLERANCE)


def test_day_c_has_no_diagonal_split_of_any_positive_scale_and_plans_its_centre():
    found = diagonal_split.split_box(rows_c(), [0.5, 1.0], [0.5, 0.0], time_limit=BOX_SECONDS)
    assert found.outcome == "optimal"
    assert found.kind == diagonal_split.NO_GAINS
    assert found.largest_scale == pytest.approx(0.0, abs=diagonal_split.ZERO_SCALE)
    plan = found.plan
    assert np.all(plan.gains == 0)
    assert plan.constant[0, 1] == pytest.approx(0.5, abs=TOLERANCE)
    assert plan.constant[1, 1] == pytest.approx(0.0, abs=TOLERANCE)
    assert plan.constant[:, 2].sum() == pytest.approx(1.0, abs=TOLERANCE)
    for zeta in corner_paths(2):
        assert plan.powers(1, zeta).sum() == pytest.approx(0.5, abs=TOLERANCE)
