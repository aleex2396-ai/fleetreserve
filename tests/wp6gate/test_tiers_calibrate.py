"""The two tiers of plan section 6 in the calibration of one unit (`gate_calibrate.run_job`).

Tier A (driver-protected) takes the smaller of the slot and driver scales, tier B (deadline
contract) the scale of the slot loss alone. On a made-up unit of 60 days where every fourth day has
a car leaving 2.5 hours after τ, after the one service hour but long before the forecast departure,
the slot loss never fails and the driver loss does: the driver scale is 0, so tier A offers
nothing, and tier B offers at the slot scale.

1. Tier A is unchanged by scoring tier B: every tier A number of the results, the per-day rows and
   the cache's tier A counts are equal with and without tier B, and a run without tier B has no
   tier B key.
2. Tier B's scale is the slot-only scale of the same calibration, and its test scoring (offers,
   shares of both ceilings, offering and failing days, mean losses over the offering test days)
   equals an independent re-computation from fresh replays at that scale.
3. No day is evaluated twice at one scale, and the test days are evaluated only at the two tiers'
   chosen scales.
4. The every-request route has the two tiers too: tier A unchanged by tier B, tier B at the
   slot-only scale of the closed form, scored by an independent re-computation of the closed
   forms and of the every-scale driver-loss bound, the earlier bound's mean beside it.
5. The stay-to-bound check of tier B is the sum of `stay_check` over fresh replays at the tier B
   scale of the offering calibration days and the test days; cars that leave before their bound
   are not in it, and no checked car is short.
"""

import json
import math

import numpy as np
import pytest
from gate_fake import gate_cars, matrix_document, write_tables

from fleetreserve import crc, gate_calibrate, gate_config, gate_day, gate_stat, stay_check
from fleetreserve.carsets import true_rows
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.fixed_plan_loss import closed_form_slot_loss, hour_thresholds
from fleetreserve.forecast_empirical import EmpiricalForecaster
from fleetreserve.gate_run import plan_unit, settings
from fleetreserve.rungs import RUNGS


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


def _job(unit, **options):
    prepared, ceilings, plan = unit
    batches = []

    def evaluate_batch(pairs):
        batches.append([(p.date, b) for p, b in pairs])
        return [gate_day.evaluate(p, b).compact() for p, b in pairs]

    found = gate_calibrate.run_job(prepared, ceilings, plan.roles, evaluate_batch, **options)
    return found, batches


@pytest.fixture(scope="module")
def plain(unit):
    return _job(unit)


@pytest.fixture(scope="module")
def tiered(unit):
    return _job(unit, tier_b=True)


def _json(value):
    return json.dumps(value, sort_keys=True)


def test_the_made_up_unit_separates_the_tiers(plain):
    (results, _, _, _), _ = plain
    for rung in RUNGS:
        found = results["rungs"][rung]
        assert found["slot_only"]["beta_hat"] > 0
        assert found["driver_only"]["beta_hat"] == 0
        assert found["beta_hat"] == 0 and found["reason"].startswith("no scale is safe")


def test_tier_a_is_unchanged_by_scoring_tier_b(plain, tiered):
    (without, rows_without, _, cache_without), _ = plain
    (with_b, rows_with, _, cache_with), _ = tiered
    for rung in RUNGS:
        assert "tier_b" not in without["rungs"][rung]
        assert "tier_b" in with_b["rungs"][rung]
        kept = {k: v for k, v in with_b["rungs"][rung].items() if k != "tier_b"}
        assert _json(kept) == _json(without["rungs"][rung])
    for key in without:
        if key not in ("rungs", "t5", "fixed_every_request"):  # T5 counts tier B day-paths too
            assert _json(with_b[key]) == _json(without[key]), key
    for before, after in zip(rows_without, rows_with, strict=True):
        stripped = {r: {k: v for k, v in x.items() if k != "tier_b"} for r, x in after.items()}
        assert _json(before) == _json(stripped)
    assert set(cache_without.store) <= set(cache_with.store)
    assert with_b["t5"]["day_paths_checked"] >= without["t5"]["day_paths_checked"]
    assert with_b["t5"]["breaks_a"] == with_b["t5"]["breaks_b"] == 0


def test_the_tier_b_scale_is_the_slot_only_scale_and_offers_where_tier_a_does_not(tiered):
    (results, _, _, _), _ = tiered
    for rung in RUNGS:
        found = results["rungs"][rung]
        tier_b = found["tier_b"]
        assert tier_b["beta_hat"] == found["slot_only"]["beta_hat"] > 0
        assert tier_b["reason"] == found["slot_only"]["reason"] == ""
        assert found["test"]["mean_offer_kwh_per_day"] == 0
        assert tier_b["test"]["mean_offer_kwh_per_day"] > 0
        assert tier_b["test"]["mean_driver_loss_offering_days"] > crc.ALPHA_DRIVER


def test_the_tier_b_test_scoring_equals_an_independent_recomputation(unit, tiered):
    prepared, ceilings, plan = unit
    (results, _, _, _), _ = tiered
    test = [d for d, role in enumerate(plan.roles) if role == "test"]
    values = [prepared[d].value for d in test]
    offering = [i for i, d in enumerate(test) if not prepared[d].abstains]
    assert len(offering) > 0
    for rung in RUNGS:
        beta = results["rungs"][rung]["slot_only"]["beta_hat"]
        found = results["rungs"][rung]["tier_b"]["test"]
        fresh = [gate_day.evaluate(prepared[d], beta).rungs[rung] for d in test]
        slot = [fresh[i].slot_loss for i in offering]
        driver = [fresh[i].driver_loss for i in offering]
        offers = [beta * v for v in values]
        assert found["days"] == len(test) and found["offering_days"] == len(offering)
        assert found["failing_days"] == sum(loss > 0 for loss in slot)
        assert found["mean_slot_loss_offering_days"] == pytest.approx(np.mean(slot))
        assert found["mean_driver_loss_offering_days"] == pytest.approx(np.mean(driver))
        assert found["mean_offer_kwh_per_day"] == pytest.approx(np.mean(offers))
        assert found["mean_offer_kwh_per_offering_day"] == pytest.approx(
            np.mean([offers[i] for i in offering])
        )
        cars = sum(prepared[d].n_cars for d in test)
        assert found["offer_kwh_per_car_per_day"] == pytest.approx(sum(offers) / cars)
        for name, pick in (
            (gate_config.CEILINGS[0], lambda c: c.hindsight),
            (gate_config.CEILINGS[1], lambda c: c.best_causal),
        ):
            expected = gate_stat.unit_offer(values, [pick(ceilings[d]) for d in test], beta)
            share = found["share_of_ceiling"][name]
            assert share["ratio_of_sums"] == pytest.approx(expected.share_of_sums)
            assert share["mean_of_daily_ratios"] == pytest.approx(expected.mean_of_daily_ratios)


def test_no_day_is_evaluated_twice_at_one_scale_and_test_days_only_at_chosen_scales(unit, tiered):
    _, _, plan = unit
    (results, _, _, cache), batches = tiered
    assert cache.calls and set(cache.calls.values()) == {1}
    flat = [pair for batch in batches for pair in batch]
    assert len(flat) == len(set(flat))
    test = {d for d, role in enumerate(plan.roles) if role == "test"}
    chosen = {
        float(results["rungs"][r][key]["beta_hat"] if key else results["rungs"][r]["beta_hat"])
        for r in RUNGS
        for key in (None, "tier_b")
    }
    seen = {beta for d, beta in cache.calls if d in test}
    assert seen <= {b for b in chosen if b > 0} and seen


def test_the_stay_to_bound_check_is_the_sum_over_fresh_replays_at_the_tier_b_scale(unit, tiered):
    prepared, _, plan = unit
    (results, _, _, _), _ = tiered
    days = [
        d
        for d, role in enumerate(plan.roles)
        if not prepared[d].abstains and role in ("calibration", "test")
    ]
    for rung in RUNGS:
        beta = results["rungs"][rung]["tier_b"]["beta_hat"]
        total = stay_check.NONE
        for d in days:
            total = stay_check.add(total, gate_day.evaluate(prepared[d], beta).rungs[rung].stay)
        found = results["rungs"][rung]["tier_b"]["stay_to_bound"]
        assert found["scale"] == beta
        assert found["days"] == len(days)
        assert found["tolerance_kwh"] == prepared[0].t5_tolerance_kwh
        assert found["cars_checked"] == total.cars > 0
        assert found["car_paths_checked"] == total.car_paths
        assert found["cars_short"] == total.short_cars == 0
        assert found["car_paths_short"] == total.short_car_paths == 0
        assert found["largest_shortfall_kwh"] == total.worst_kwh == 0.0
        every = sum(len(prepared[d].steered) for d in days)
        assert found["cars_checked"] < every  # the early cars leave before their bound


def test_the_every_request_route_has_both_tiers(unit, plain, tiered):
    prepared, _, plan = unit
    (without, _, _, _), _ = plain
    (with_b, _, _, _), _ = tiered
    assert "tier_b" not in without["fixed_every_request"]
    kept = {k: v for k, v in with_b["fixed_every_request"].items() if k != "tier_b"}
    assert _json(kept) == _json(without["fixed_every_request"])
    found = with_b["fixed_every_request"]
    calibration = [d for d, role in enumerate(plan.roles) if role == "calibration"]
    test = [d for d, role in enumerate(plan.roles) if role == "test"]
    offering = [d for d in calibration if not prepared[d].abstains]
    steps = []
    for d in offering:
        p = prepared[d]
        steps.append(
            gate_calibrate.slot_steps(hour_thresholds(p.certificate.policy, true_rows(p.day)))
        )
    slot_beta = crc.calibrate_exact(steps, [((), ())] * len(steps)).slot_beta
    tier_b = found["tier_b"]
    assert found["slot_beta"] == gate_calibrate.number(slot_beta)
    used = crc.GRID[-1] if math.isinf(slot_beta) else slot_beta
    assert tier_b["beta_used"] == used
    assert tier_b["capped_at_grid_top"] is math.isinf(slot_beta)
    assert found["beta_used"] <= tier_b["beta_used"]
    test_offering = [d for d in test if not prepared[d].abstains]
    slot, driver, earlier = [], [], []
    for d in test_offering:
        p = prepared[d]
        slot.append(
            closed_form_slot_loss(hour_thresholds(p.certificate.policy, true_rows(p.day)), used)
        )
        driver.append(gate_calibrate.every_scale_loss(gate_calibrate.every_scale_terms(p), used))
        earlier.append(gate_calibrate.ramp_loss(gate_calibrate.driver_terms(p), used))
    assert tier_b["test_offering_days"] == len(test_offering)
    assert tier_b["test_failing_days"] == sum(loss > 0 for loss in slot)
    assert tier_b["test_mean_slot_loss_offering_days"] == pytest.approx(np.mean(slot))
    assert tier_b["test_mean_driver_loss_bound_offering_days"] == pytest.approx(np.mean(driver))
    assert tier_b[gate_calibrate.EARLIER][
        "test_mean_driver_loss_bound_offering_days"
    ] == pytest.approx(np.mean(earlier))
    assert tier_b["test_mean_offer_kwh_per_day"] == pytest.approx(
        used * np.mean([prepared[d].value for d in test])
    )
