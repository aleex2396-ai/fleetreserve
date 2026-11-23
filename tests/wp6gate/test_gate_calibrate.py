"""Calibration and test scoring of one unit (`gate_calibrate`).

1. The fixed plan's closed-form slot loss as `crc` steps agrees with
   `fixed_plan_loss.closed_form_slot_loss` at every scale, on a hand case and on seeded thresholds.
2. The driver-loss bound's calibrated scale: a hand case (20 equal days, each losing
   `(beta - 1) / 10` above beta 1: the bound `(20 (beta - 1) / 10 + 1) / 21 <= 0.05` gives
   beta 1.025), and agreement with a dense scan on seeded days.
3. On a made-up unit of 60 days (main family, kappa 0.1, empirical departure, departures seen
   from the rounded-down hour): no day is evaluated twice at one scale; the test days are
   evaluated only at the rungs' chosen scales; the calibration uses 50 days; the offers and shares
   are those of `gate_stat.unit_offer` on the test days' certified values and ceilings.
"""

import random

import numpy as np
import pytest
from gate_fake import gate_cars, matrix_document, write_tables

from fleetreserve import crc, gate_calibrate, gate_config, gate_day, gate_stat
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.fixed_plan_loss import closed_form_slot_loss
from fleetreserve.forecast_empirical import EmpiricalForecaster
from fleetreserve.gate_run import plan_unit, settings


def test_slot_steps_by_hand():
    steps, levels = gate_calibrate.slot_steps([0.5, 0.3, np.inf])
    assert steps == [0.3, 0.3, 0.5]
    assert levels == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert gate_calibrate.slot_steps([np.inf, np.inf]) == ([], [])
    assert gate_calibrate.slot_steps([-0.2, 1.0]) == ([0.0, 0.0], [0.5, 1.0])


def test_slot_steps_agree_with_the_closed_form():
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        thresholds = rng.choice([np.inf, -0.1, *np.round(rng.uniform(0, 2, 6), 2)], size=n)
        steps, levels = gate_calibrate.slot_steps(thresholds)
        for beta in np.arange(0.005, 2.5, 0.01):
            assert crc.step_loss(steps, levels, beta) == pytest.approx(
                closed_form_slot_loss(thresholds, beta)
            )


def test_driver_bound_scale_by_hand():
    day = (np.array([-1.0]), np.array([1.0]), 10.0)
    assert gate_calibrate.beta_hat_ramps([day] * 20, 0.05) == pytest.approx(1.025, abs=1e-9)
    assert gate_calibrate.beta_hat_ramps([day] * 18, 0.05) == 0.0  # too few days
    never = (np.array([-1.0]), np.array([0.0]), 10.0)
    assert gate_calibrate.beta_hat_ramps([never] * 20, 0.05) == float("inf")
    at_once = (np.array([0.5]), np.array([0.0]), 1.0)
    assert gate_calibrate.beta_hat_ramps([at_once] * 20, 0.05) == 0.0


def test_driver_bound_scale_agrees_with_a_dense_scan():
    rng = random.Random(5)
    for _ in range(10):
        days = []
        for _ in range(25):
            k = rng.randint(1, 4)
            c = np.array([rng.uniform(-3, 0.02) for _ in range(k)])
            s = np.array([rng.choice([0.0, rng.uniform(0, 2)]) for _ in range(k)])
            days.append((c, s, rng.uniform(1, 10)))
        found = gate_calibrate.beta_hat_ramps(days, 0.05)
        grid = np.arange(1e-4, 4, 4e-3)
        holds = [
            sum(gate_calibrate.ramp_loss(d, b) for d in days) + 1 <= 0.05 * 26 + 1e-9 for b in grid
        ]
        if all(holds):
            assert found >= grid[-1]
            continue
        first = holds.index(False)
        assert all(not h for h in holds[first:])  # the bound fails from there on
        low = grid[first - 1] if first > 0 else 0.0
        assert low - 1e-9 <= found <= grid[first] + 1e-9


@pytest.fixture(scope="module")
def unit_job(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tables")
    write_tables(folder, "u", gate_cars(60))
    days = load_unit_days(folder, "u", rho=0.0, no_fit="cap", small_fleet="keep")
    matrix = gate_config.parse(matrix_document())
    configuration = matrix.configuration("main-rounded-hour-k0.1-empirical")
    forecaster = EmpiricalForecaster(days, reference="other-days", min_cell=20)
    prepared = [
        gate_day.prepare_day(t, forecaster, configuration, settings(matrix)) for t in days.days
    ]
    ceilings = [gate_day.day_ceilings(t.day) for t in days.days]
    plan = plan_unit("u", days, matrix)
    batches = []

    def evaluate_batch(pairs):
        batches.append([(p.date, b) for p, b in pairs])
        return [gate_day.evaluate(p, b).compact() for p, b in pairs]

    found = gate_calibrate.run_job(prepared, ceilings, plan.roles, evaluate_batch)
    return prepared, ceilings, plan, found, batches


def test_no_day_is_evaluated_twice_at_one_scale(unit_job):
    _, _, _, (results, _, _, cache), batches = unit_job
    assert cache.calls and set(cache.calls.values()) == {1}
    flat = [pair for batch in batches for pair in batch]
    assert len(flat) == len(set(flat))
    assert results["evaluations"]["calibration"] > 50


def test_test_days_are_evaluated_only_at_the_chosen_scales(unit_job):
    _, _, plan, (results, _, _, cache), _ = unit_job
    test = {d for d, role in enumerate(plan.roles) if role == "test"}
    chosen = {float(results["rungs"][r]["beta_hat"]) for r in results["rungs"]}
    seen = {beta for d, beta in cache.calls if d in test}
    assert seen <= {b for b in chosen if b > 0}
    assert results["calibration_days"] == 50 and results["test_days"] == 10


def test_the_offer_is_gate_stat_arithmetic(unit_job):
    prepared, ceilings, plan, (results, rows, _, _), _ = unit_job
    test = [d for d, role in enumerate(plan.roles) if role == "test"]
    values = [prepared[d].value for d in test]
    for rung, entry in results["rungs"].items():
        beta = float(entry["beta_hat"])
        offer = gate_stat.unit_offer(values, [ceilings[d].hindsight for d in test], beta)
        found = entry["test"]
        assert found["mean_offer_kwh_per_day"] == pytest.approx(offer.mean_offer_kwh)
        share = found["share_of_ceiling"]["hindsight-free-decision-hour"]
        assert share["ratio_of_sums"] == pytest.approx(offer.share_of_sums)
        assert share["mean_of_daily_ratios"] == pytest.approx(offer.mean_of_daily_ratios)
        assert [rows[d][rung]["offer_kwh"] for d in test] == pytest.approx(list(offer.offers_kwh))
        cars = sum(prepared[d].n_cars for d in test)
        assert found["offer_kwh_per_car_per_day"] == pytest.approx(sum(offer.offers_kwh) / cars)
