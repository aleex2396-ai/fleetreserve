"""Tier A of the every-request route under the every-scale driver-loss bound, the driver bound
(`gate_calibrate`, module text; `fixed_plan_loss`, reading R3), with the earlier bound's figures
beside it under `gate_calibrate.EARLIER`, a comparison only.

1. By hand: 20 offering calibration days, 19 copies of `corner_fake.SINGLE` and one of the
   early-departure day `EARLY` (`test_rungs_hand.py`).
   - Slot loss: SINGLE's stored-energy row of hour 2 gives (1.5 - 1) / 1 = 0.5, EARLY's hour 2 is
     broken at every scale (`-inf`) and its hour 1 gives 1. Up to scale 0.5 only EARLY loses, 1/2
     of its hours: (0.5 + 1) / 21 <= 0.10; above 0.5 the 19 SINGLE days lose 1/2 each. The slot
     scale is 0.5.
   - The earlier driver-loss bound is 0 on every day up to scale 1.5 (SINGLE's recovery supply 2
     covers its need 1.5; EARLY's car A plans its need), so the earlier driver scale is above 1.5
     and the earlier tier A scale is the slot scale 0.5.
   - The every-scale bound of EARLY is 1/8 at every scale (car A leaves with 0.5 kW planned in hour
     2), so (1/8 + 1) / 21 > 0.05 just above 0: the driver scale is 0, and the unit's tier A
     abstains with "no scale is safe".
2. Every field of the route is the same with or without the earlier blocks; tier A's scale and the
   earlier block's are each the smaller of the slot scale and their own driver scale.
3. `every_scale_loss` on made-up days is the every-scale bound of `fixed_plan_loss` capped at 1, and
   `ramp_loss` bit for bit at every scale at which no car's power row is broken; it never exceeds
   its limit.
4. `beta_hat_every_scale` returns the largest scale at which the condition holds (to the
   bisection's tolerance) on seeded sets of made-up calibration days.
5. `run_job` writes the earlier block under `fixed_every_request`, on a made-up unit of 60 days.
"""

import json
import math
import random

import numpy as np
import pytest
from corner_fake import SINGLE, hand, offer_days
from gate_fake import gate_cars, matrix_document, write_tables
from test_rungs_hand import EARLY, EARLY_POLICY

from fleetreserve import crc, gate_calibrate, gate_config, gate_day
from fleetreserve.carsets import true_rows
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.fixed_plan_loss import every_scale_driver_bound, service_hours
from fleetreserve.forecast_empirical import EmpiricalForecaster
from fleetreserve.gate_calibrate import EARLIER
from fleetreserve.gate_run import plan_unit, settings

MADE_UP_DAYS = 60
GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0)


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


# --- 1. by hand ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hand_unit():
    prepared = [SINGLE] * 19 + [hand(EARLY, EARLY_POLICY)] + [SINGLE, SINGLE]
    return prepared, list(range(20)), [20, 21]


def test_the_early_departure_day_makes_tier_a_abstain(hand_unit):
    prepared, calibration, test = hand_unit
    found = gate_calibrate.every_request(prepared, calibration, test, earlier=True)
    assert found["slot_beta"] == pytest.approx(0.5)
    earlier = found[EARLIER]
    assert earlier["driver_beta"] > 1.5
    assert earlier["beta_used"] == pytest.approx(0.5) and not earlier["abstain"]
    assert found["driver_beta"] == 0.0 and found["beta"] == 0.0 and found["beta_used"] == 0.0
    assert found["abstain"] and not found["capped_at_grid_top"]
    assert found["reason"] == (
        "no scale is safe: 20 of 20 calibration days offer; "
        "the driver-loss bound fails at every beta above 0"
    )
    assert found["test_offering_days"] == 2 and found["test_failing_days"] is None
    assert found["test_mean_driver_loss_bound_offering_days"] is None


def test_the_hand_days_every_scale_losses(hand_unit):
    prepared, _, _ = hand_unit
    single, early = (gate_calibrate.every_scale_terms(prepared[d]) for d in (0, 19))
    # SINGLE: no power row broken up to scale 1, where today's bound is 0 (its recovery supply
    # covers its need); above 1 its rows break, and with its supply its every-scale term stays 0,
    # while today's grows to (2 - 1.5) / 1.5 at scale 2
    for beta in GRID:
        assert gate_calibrate.every_scale_loss(early, beta) == pytest.approx(1 / 8)
        assert gate_calibrate.ramp_loss(early[0], beta) == 0.0
        assert gate_calibrate.every_scale_loss(single, beta) == 0.0
        today = gate_calibrate.ramp_loss(single[0], beta)
        assert today == pytest.approx(max(0.0, beta - 1.5) / 1.5)


# --- 2. the fields unchanged by the earlier blocks -----------------------------------------------


@pytest.mark.parametrize("tier_b", [False, True])
def test_every_field_is_the_same_with_the_earlier_blocks(tier_b):
    prepared = offer_days(MADE_UP_DAYS)
    rng = random.Random(3)
    for _ in range(3):
        order = rng.sample(range(MADE_UP_DAYS), MADE_UP_DAYS)
        calibration, test = sorted(order[:40]), sorted(order[40:])
        plain = gate_calibrate.every_request(prepared, calibration, test, tier_b=tier_b)
        both = gate_calibrate.every_request(
            prepared, calibration, test, tier_b=tier_b, earlier=True
        )
        earlier = both.pop(EARLIER)
        if tier_b:
            assert both["tier_b"].pop(EARLIER)["about"] == gate_calibrate.EARLIER_ABOUT
        assert _json(both) == _json(plain)
        slot = float(plain["slot_beta"])
        assert float(plain["beta"]) == min(slot, float(plain["driver_beta"]))
        assert float(earlier["beta"]) == min(slot, float(earlier["driver_beta"]))
        used = float(earlier["beta_used"])
        assert used == (
            crc.GRID[-1] if math.isinf(float(earlier["beta"])) else float(earlier["beta"])
        )
        mean = np.mean([used * prepared[d].value for d in test])
        assert earlier["test_mean_offer_kwh_per_day"] == pytest.approx(mean, abs=1e-12)


# --- 3. the day loss -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def made_up():
    prepared = offer_days(MADE_UP_DAYS)
    return prepared, [gate_calibrate.every_scale_terms(p) for p in prepared]


def test_the_day_loss_is_the_every_scale_bound_and_todays_where_nothing_breaks(made_up):
    prepared, terms = made_up
    equal = 0
    for p, day_terms in zip(prepared, terms, strict=True):
        rows = true_rows(p.day)
        hours = service_hours(p.certificate.policy, rows)
        for beta in GRID:
            loss = gate_calibrate.every_scale_loss(day_terms, beta)
            bound = every_scale_driver_bound(p.certificate.policy, rows, p.day, beta)
            assert loss == pytest.approx(min(1.0, bound), abs=1e-12)
            if not (hours.thresholds < beta).any():
                assert loss == gate_calibrate.ramp_loss(day_terms[0], beta)
                equal += 1
            assert loss <= gate_calibrate._every_scale_limit(day_terms) + 1e-12
    assert equal > 100


# --- 4. the calibration --------------------------------------------------------------------------


def test_the_driver_scale_is_the_largest_at_which_the_condition_holds(made_up):
    _, terms = made_up
    rng = random.Random(11)
    alpha = crc.ALPHA_DRIVER
    found_finite = 0
    for _ in range(12):
        days = [terms[d] for d in rng.sample(range(MADE_UP_DAYS), 30)]

        def holds(beta, days=days):
            total = sum(gate_calibrate.every_scale_loss(t, beta) for t in days)
            return total + 1.0 <= alpha * (len(days) + 1) + crc.TOLERANCE

        beta = gate_calibrate.beta_hat_every_scale(days, alpha)
        if math.isinf(beta):
            assert holds(1e6)
            continue
        found_finite += 1
        assert beta == 0.0 or holds(beta)
        assert not holds(beta * (1 + 1e-10) + 1e-10)
    assert found_finite > 0


# --- 5. run_job ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unit(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tables")
    write_tables(folder, "u", gate_cars(60, early_every=4, early_out=12.5))
    days = load_unit_days(folder, "u", rho=0.0, no_fit="cap", small_fleet="keep")
    matrix = gate_config.parse(matrix_document())
    configuration = matrix.configuration("main-rounded-hour-k0.1-empirical")
    forecaster = EmpiricalForecaster(days, reference="other-days", min_cell=20)
    prepared = [
        gate_day.prepare_day(t, forecaster, configuration, settings(matrix)) for t in days.days
    ]
    ceilings = [gate_day.day_ceilings(t.day) for t in days.days]
    return prepared, ceilings, plan_unit("u", days, matrix)


def test_run_job_writes_the_earlier_block_beside_the_fields(unit):
    prepared, ceilings, plan = unit

    def evaluate_batch(pairs):
        return [gate_day.evaluate(p, b).compact() for p, b in pairs]

    results, *_ = gate_calibrate.run_job(prepared, ceilings, plan.roles, evaluate_batch)
    fixed = results["fixed_every_request"]
    calibration = [d for d, role in enumerate(plan.roles) if role == "calibration"]
    test = [d for d, role in enumerate(plan.roles) if role == "test"]
    expected = gate_calibrate.every_request(prepared, calibration, test, earlier=True)
    assert _json(fixed[EARLIER]) == _json(expected[EARLIER])
    assert set(fixed) - set(expected) == {
        "test_mean_offer_kwh_per_day",
        "test_mean_offer_kwh_per_offering_day",
    }
