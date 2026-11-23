"""Per-day sizes and the calibrated fixed-plan offer (`size_scale`).

1. Composition: with size 1 on every offering day, the calibrated scalars, the offer per counting
   test day, the numbers of offering days and the mean test losses equal those of
   `gate_calibrate.every_request` exactly, in both tiers, on made-up prepared days (`size_fake`)
   and several splits; with every hour threshold tripled (`every_request` then gives a scale above
   1) the calibrated scalars still agree and the scalar used is 1; with every hour threshold
   raised 1e7 times, tier B's calibrated scalar is above `policy.VALIDITY_LIMIT`, where
   `every_request` writes the limit and keeps the calibrated value under its mark
   (`gate_calibrate.VALIDITY`), and the scalar used is still 1.
2. A day's loss at scalar g and size q is the fixed plan's at scale g q: the closed form at g q, the
   replay of every corner path at g q (`fixed_plan_loss.compare_with_replay`, where the two agree),
   and the driver-loss bound at g q (`fixed_plan_loss.every_scale_driver_bound`); doubling q is
   doubling g. No day's scale exceeds 1 (the cap).
3. Arms: "scale" clips the raw size to [FLOOR, 1]; "abstain" drops a day below FLOOR; a day that
   offers with no raw size is refused. A day without a size is in neither loss mean, offers 0 and
   is still a counting test day; it is not a calibration day.
4. Guarantee: with sizes drawn independently of the outcomes, the mean test loss over many draws of
   exchangeable made-up days is at most alpha (slot 0.10, driver 0.05) within a statistical
   margin. Power: sizes fitted to the calibration days' own thresholds and not to the scored day's
   (as a size predictor fitted on the calibration days would be) break it.
5. It does something: on planted days where 30% of days fail above scale 0.2 and the rest above
   5, sizes that follow the days' own thresholds raise the calibrated offer well above constant's;
   the same sizes shuffled over the days do not.
"""

import math

import numpy as np
import pytest
from size_fake import prepared_days, ramp_terms

from fleetreserve import crc, gate_calibrate, size_scale
from fleetreserve.carsets import true_rows
from fleetreserve.fixed_plan_loss import (
    closed_form_slot_loss,
    compare_with_replay,
    every_scale_driver_bound,
    hour_thresholds,
)
from fleetreserve.policy import VALIDITY_LIMIT


@pytest.fixture(scope="module")
def days():
    return prepared_days(21, 90, early=0.04)


def constant_q(terms):
    offering = np.array([t is not None for t in terms])
    return size_scale.sizes(np.where(offering, 1.0, math.nan), offering, "scale")


def splits(n, count=4, n_cal=55):
    rng = np.random.default_rng(3)
    out = []
    for _ in range(count):
        order = rng.permutation(n)
        out.append((sorted(order[:n_cal].tolist()), sorted(order[n_cal:].tolist())))
    return out


def compare(days, calibration, test):
    terms = [size_scale.day_terms(p) for p in days]
    q = constant_q(terms)
    found = size_scale.calibrate(terms, q, calibration)
    existing = gate_calibrate.every_request(days, calibration, test, tier_b=True)
    return terms, q, found, existing


# --- 1. composition ------------------------------------------------------------------------------


def test_size_one_gives_the_every_request_route_exactly(days):
    positive = 0
    for calibration, test in splits(len(days)):
        terms, q, found, existing = compare(days, calibration, test)
        a, b = found["A"], found["B"]
        assert a.calibrated == existing["beta"] and b.calibrated == existing["tier_b"]["beta"]
        assert existing["beta_used"] <= 1 and existing["tier_b"]["beta_used"] <= 1
        assert a.used == existing["beta_used"] and b.used == existing["tier_b"]["beta_used"]
        assert a.days == b.days == existing["offering_calibration_days"]
        for scalar, route in ((a, existing), (b, existing["tier_b"])):
            scored = size_scale.score(terms, q, scalar.used, test)
            assert scored.offering_days == route["test_offering_days"]
            slot = size_scale.mean_or_none(scored.slot_losses)
            driver = size_scale.mean_or_none(scored.driver_losses)
            assert slot == route["test_mean_slot_loss_offering_days"]
            assert driver == route["test_mean_driver_loss_bound_offering_days"]
            offer = gate_calibrate._mean([route["beta_used"] * days[d].value for d in test])
            assert scored.mean_offer_kwh == offer
            positive += scalar.used > 0
    assert positive >= 6


def test_above_scale_one_the_scalars_agree_and_the_one_used_is_1(days, monkeypatch):
    original = hour_thresholds

    def tripled(policy, rows):
        return 3.0 * original(policy, rows)

    monkeypatch.setattr(gate_calibrate, "hour_thresholds", tripled)
    monkeypatch.setattr(size_scale, "hour_thresholds", tripled)
    calibration, test = splits(len(days))[0]
    _, _, found, existing = compare(days, calibration, test)
    assert existing["tier_b"]["beta"] > 1
    assert found["B"].calibrated == existing["tier_b"]["beta"]
    assert found["A"].calibrated == existing["beta"]
    assert found["B"].used == 1.0
    assert found["A"].used == min(existing["beta"], 1.0)


def test_above_the_validity_limit_the_calibrated_scalar_is_the_value_under_the_routes_mark(
    days, monkeypatch
):
    original = hour_thresholds

    def raised(policy, rows):
        return 1e7 * original(policy, rows)

    monkeypatch.setattr(gate_calibrate, "hour_thresholds", raised)
    monkeypatch.setattr(size_scale, "hour_thresholds", raised)
    calibration, test = splits(len(days))[0]
    _, _, found, existing = compare(days, calibration, test)
    route = existing["tier_b"]
    assert found["B"].calibrated > VALIDITY_LIMIT
    assert route["beta"] == route["beta_used"] == VALIDITY_LIMIT
    assert route[gate_calibrate.VALIDITY]["formula_values"] == {"beta": found["B"].calibrated}
    assert found["B"].used == 1.0
    for scalar, block in ((found["A"], existing), (found["B"], route)):
        mark = block.get(gate_calibrate.VALIDITY, {}).get("formula_values", {})
        assert scalar.calibrated == mark.get("beta", block["beta"])
        assert scalar.used == min(block["beta_used"], 1.0)


# --- 2. one day's loss ---------------------------------------------------------------------------


def test_a_days_loss_at_scalar_g_and_size_q_is_the_fixed_plans_at_scale_g_q(days):
    rng = np.random.default_rng(4)
    agreed = 0
    for prepared in [p for p in days if not p.abstains][:12]:
        terms = [size_scale.day_terms(prepared)]
        rows = true_rows(prepared.day)
        policy = prepared.certificate.policy
        thresholds = hour_thresholds(policy, rows)
        for _ in range(3):
            q, g = float(rng.uniform(0.05, 1.0)), float(rng.uniform(0.05, 1.0))
            scored = size_scale.score(terms, np.array([q]), g, [0])
            (comparison,) = compare_with_replay(prepared.day, policy, rows, [g * q])
            assert scored.slot_losses[0] == closed_form_slot_loss(thresholds, g * q)
            assert scored.slot_losses[0] == comparison.closed_slot
            if comparison.reason is None:
                assert scored.slot_losses[0] == pytest.approx(comparison.replay_slot)
                agreed += 1
            bound = every_scale_driver_bound(policy, rows, prepared.day, g * q)
            assert scored.driver_losses[0] == pytest.approx(min(bound, 1.0), abs=1e-12)
            doubled = size_scale.score(terms, np.array([q / 2]), 2 * g, [0])
            assert doubled.slot_losses == scored.slot_losses
            assert doubled.driver_losses == pytest.approx(scored.driver_losses, abs=1e-12)
    assert agreed >= 20


def synthetic_days(rng, n):
    """Made-up day terms: two hours with thresholds in (0, 3), one driver ramp per day."""
    out = []
    for _ in range(n):
        thresholds = np.sort(rng.uniform(0.0, 3.0, 2))[::-1].copy()
        c = np.array([-float(rng.uniform(0.0, 3.0))])
        out.append(ramp_terms(float(rng.uniform(1, 5)), thresholds, (c, np.array([1.0]), 1.0)))
    return out


def test_no_days_scale_exceeds_1():
    rng = np.random.default_rng(6)
    capped = 0
    for _ in range(200):
        terms = synthetic_days(rng, 40)
        for t in terms:
            t.thresholds[:] = t.thresholds + 10.0  # no loss below scale 10
        raw = rng.uniform(0.0, 1.5, 40)
        for arm in size_scale.ARMS:
            q = size_scale.sizes(raw, np.ones(40, dtype=bool), arm)
            for tier, scalar in size_scale.calibrate(terms, q, range(25)).items():
                kept = q[~np.isnan(q)]
                assert scalar.used * kept.max() <= 1.0 + 1e-12, tier
                capped += scalar.used < scalar.calibrated
    assert capped > 100


# --- 3. arms and days without a size -------------------------------------------------------------


def test_the_two_arms():
    raw = np.array([0.01, 0.05, 0.5, 1.7, math.nan])
    offering = np.array([True, True, True, True, False])
    scale = size_scale.sizes(raw, offering, "scale")
    abstain = size_scale.sizes(raw, offering, "abstain")
    np.testing.assert_array_equal(scale, [0.05, 0.05, 0.5, 1.0, math.nan])
    np.testing.assert_array_equal(abstain, [math.nan, 0.05, 0.5, 1.0, math.nan])
    with pytest.raises(ValueError, match="no raw size"):
        size_scale.sizes(raw, np.ones(5, dtype=bool), "abstain")
    with pytest.raises(ValueError, match="arm"):
        size_scale.sizes(raw, offering, "other")


def test_a_day_without_a_size_is_in_no_mean_and_still_a_counting_test_day():
    terms = [
        ramp_terms(4.0, np.array([0.5, 0.5]), (np.array([-0.5]), np.array([1.0]), 1.0)),
        ramp_terms(2.0, np.array([0.1, 0.1]), (np.array([-0.1]), np.array([1.0]), 1.0)),
        None,
    ]
    q = np.array([1.0, math.nan, math.nan])
    scored = size_scale.score(terms, q, 0.8, [0, 1, 2])
    assert scored.offers_kwh == (0.8 * 4.0, 0.0, 0.0)
    assert scored.mean_offer_kwh == pytest.approx(0.8 * 4.0 / 3)
    assert scored.offering_days == 1
    assert scored.slot_losses == (1.0,) and scored.driver_losses == pytest.approx((0.3,))
    many = [terms[0]] * 20 + [terms[1]] * 5
    q = np.array([1.0] * 20 + [math.nan] * 5)
    assert size_scale.calibrate(many, q, range(25))["A"].days == 20
    # the days without a size would make the bound fail at every scalar if they were counted
    assert size_scale.calibrate(many, q, range(25))["B"].calibrated == 0.5
    assert size_scale.calibrate(many, np.ones(25), range(25))["B"].calibrated == 0.1


# --- 4. the guarantee ----------------------------------------------------------------------------


def margin(draws):
    return 4 * 0.5 / math.sqrt(draws)


def mean_test_losses(rng, draws, n, peek: bool):
    slot, driver = [], []
    for _ in range(draws):
        terms = synthetic_days(rng, n + 1)
        if peek:
            # sizes that follow the calibration days' own first-failure scale, not the scored day's
            q = np.array([min(1.0, t.thresholds.min() / 3.0) for t in terms[:n]] + [1.0])
            q = np.clip(q, size_scale.FLOOR, 1.0)
        else:
            q = rng.uniform(size_scale.FLOOR, 1.0, n + 1)
        scalars = size_scale.calibrate(terms, q, range(n))
        scored = size_scale.score(terms, q, scalars["A"].used, [n])
        slot.append(scored.slot_losses[0] if scored.slot_losses else 0.0)
        driver.append(scored.driver_losses[0] if scored.driver_losses else 0.0)
    return float(np.mean(slot)), float(np.mean(driver))


def test_sizes_independent_of_the_outcomes_keep_the_guarantee():
    draws = 3000
    slot, driver = mean_test_losses(np.random.default_rng(7), draws, 25, peek=False)
    assert slot <= crc.ALPHA_SLOT + margin(draws)
    assert driver <= crc.ALPHA_DRIVER + margin(draws)
    assert slot > 0.02 and driver > 0.005  # not trivially zero


def test_sizes_fitted_to_the_calibration_days_break_it():
    draws = 3000
    slot, _ = mean_test_losses(np.random.default_rng(7), draws, 25, peek=True)
    assert slot > crc.ALPHA_SLOT + margin(draws)


# --- 5. it does something ------------------------------------------------------------------------


def planted(rng, n):
    """30% of days fail above scale 0.2, the rest above 5; returns (terms, risky)."""
    risky = rng.random(n) < 0.3
    terms = []
    for r in risky:
        s = 0.2 if r else 5.0
        terms.append(ramp_terms(10.0, np.array([s, s]), (np.array([-s]), np.array([1.0]), 1.0)))
    return terms, risky


def test_sizes_that_follow_the_days_raise_the_offer_and_shuffled_ones_do_not():
    rng = np.random.default_rng(9)
    terms, risky = planted(rng, 300)
    offering = np.ones(300, dtype=bool)
    following = np.where(risky, 0.2, 1.0)
    sized = {
        "constant": size_scale.sizes(np.ones(300), offering, "scale"),
        "following": size_scale.sizes(following, offering, "scale"),
        "shuffled": size_scale.sizes(rng.permutation(following), offering, "scale"),
    }
    calibration, test = list(range(150)), list(range(150, 300))
    offers = {}
    for name, q in sized.items():
        scalar = size_scale.calibrate(terms, q, calibration)["A"]
        scored = size_scale.score(terms, q, scalar.used, test)
        offers[name] = scored.mean_offer_kwh
        assert size_scale.mean_or_none(scored.slot_losses) <= crc.ALPHA_SLOT + 0.05
    assert offers["constant"] == pytest.approx(10 * 0.2)
    assert offers["following"] > 3 * offers["constant"]
    assert offers["shuffled"] <= offers["constant"]
