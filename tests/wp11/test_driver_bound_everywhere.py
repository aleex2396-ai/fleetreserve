"""The every-scale driver-loss bound (`fixed_plan_loss.every_scale_driver_bound`, reading R3) is the
driver bound at every place that calibrates, sizes, tests or checks with one; the earlier bound
(`fixed_plan_loss.driver_loss_bound`, reading R2) is reported only beside it, under names that say
so, and never sets an offer, a scale or a pass.

The places: the comparison with replay (`fixed_plan_loss.compare_with_replay`, field
`driver_bound`); the every-request route's tier A and tier B (`gate_calibrate.every_request`, and
`run_job`, which writes it); the sizing what-if (`size_scale.calibrate` and `size_scale.score`);
T7's losses and constant scales (`t7_stat.day_losses`, `t7_unit.measure_unit`); the corner check
(`corner_search.closed_form_check`, `corner_summary.pass_mark_entry`).

Hand days, worked by hand (`test_every_scale_bound.py`):

- `EARLY` (`test_rungs_hand.py`): car A (rate 1, room 1) leaves in hour 2 with 0.5 kW planned in
  it, so hour 2 fails at every scale above 0 (its row `u_2 <= 0` is broken at the centre) and
  replay loses 1/8 of the protected energy 4 on every corner. The earlier bound counts A's plan
  as drawn: 0 up to scale 1.5. The every-scale bound counts hour 2 at what A can draw there, 0:
  1/8 at every scale.
- `SWAP` (`test_every_scale_bound.py`): one car (rate 1, room 1.6, no recovery) planning
  `0.8 + 0.5 x zeta_1` and `0.8 - 0.5 x zeta_1` at scale `x`; both hours' rate rows break above
  0.4, and on `zeta_1 = +1` the car draws 1 then `0.8 - 0.5 x`: replay and the every-scale bound
  `(0.5 x - 0.2) / 1.6` for `x` in (0.4, 1.6], 0.03125 at 0.5 and 0.1875 at 1; the earlier bound
  is 0 at every scale (the planned total 1.6 does not depend on the path).
- `SINGLE` (`corner_fake`): its recovery supply covers its need; both bounds are 0 up to scale 1.

1. The comparison with replay on `EARLY` at 0.5 and 1: `driver_bound` 1/8 (replay's), the earlier
   bound 0 beside it as `earlier_driver_bound`.
2. The every-request route on 19 days `SINGLE`, one `EARLY` (calibration) and the test days
   `EARLY`, `SINGLE`. Slot scale 0.5 (`test_every_scale_calibrate.py`). Driver scale: with one day
   at 1/8 at every scale, (1/8 + 1) / 21 > 0.05 just above 0, so 0, and tier A abstains ("no scale
   is safe"); under the earlier bound the driver scale is above 1.5 and tier A's scale 0.5, now
   only under the key `gate_calibrate.EARLIER`. Tier B (the slot scale 0.5): its driver-loss mean
   over the two test days is (1/8 + 0) / 2 = 1/16, the earlier bound's 0 beside it.
3. The sizing what-if: the same days with size 1 give tier A the scalar 0 (tier B 0.5), and
   `EARLY` scored at 0.5 has the driver loss 1/8. On `SWAP` the size scales the bound's scale:
   size 0.5 at the scalar 1 and at 2 gives the bound at 0.5 and 1, 0.03125 and 0.1875.
4. T7: `t7_stat.day_losses` of `EARLY` at 0.5 has the driver term 1/8; `t7_unit.measure_unit` of
   the unit of 2 gives tier A the scalar 0 on its one split.
5. The corner check on `EARLY` at 0.5 and 1: the bound 1/8 equals the corner maximum, so no pair
   is above the bound; the earlier bound's 0 is beside it, and the pass-mark entry counts the two
   pairs above it apart (`earlier_driver_pairs_above_bound`).
6. Replay agreement: on seeded made-up days (`corner_fake.offer_days`), at each place's own scales
   (the scales of the comparison and the corner check up to `policy.VALIDITY_LIMIT`, the sizing's
   day scales `gamma q_d`, T7's window ends) and, on the made-up days of `size_fake` with early
   departures, at the route's tier A and tier B scales on several splits (where the route's mean is
   the mean of the per-day bounds checked), the largest replayed driver loss over the corner paths
   is at most the place's driver bound plus `TOLERANCE`.
"""

import numpy as np
import pytest
from corner_fake import SINGLE, hand, offer_days
from size_fake import prepared_days
from test_every_scale_bound import SWAP_DAY, SWAP_POLICY
from test_rungs_hand import EARLY, EARLY_POLICY

from fleetreserve import (
    corner_search,
    corner_summary,
    dial_splits,
    gate_calibrate,
    size_scale,
    t7_stat,
    t7_unit,
)
from fleetreserve.carsets import true_rows
from fleetreserve.fixed_plan_loss import compare_with_replay
from fleetreserve.plans import Plan
from fleetreserve.policy import VALIDITY_LIMIT
from fleetreserve.products import corner_paths
from fleetreserve.replay import run_plan

TOLERANCE = 1e-9  # as tests/wp11/test_every_scale_bound.py
MADE_UP_DAYS = 60
FAR = (1.5, 2.0, 5.0, 100.0, VALIDITY_LIMIT)  # scales above the grid, up to the validity limit

E = hand(EARLY, EARLY_POLICY)
SWAP = hand(SWAP_DAY, SWAP_POLICY)
UNIT = [SINGLE] * 19 + [E] + [E, SINGLE]
CALIBRATION, TEST = list(range(20)), [20, 21]


def replayed_driver(prepared, beta: float) -> float:
    """The largest replayed driver loss of the fixed plan at `beta` over the corner paths."""
    if beta <= 0:
        return 0.0
    policy = prepared.certificate.policy
    plan = Plan.from_policy(policy, beta)
    return max(
        run_plan(prepared.day, plan, z, policy.request(z, beta)).driver_loss
        for z in corner_paths(prepared.day.n_service)
    )


# --- 1. the comparison with replay ---------------------------------------------------------------


def test_the_comparison_with_replay_reports_the_every_scale_bound():
    for found in compare_with_replay(EARLY, EARLY_POLICY, true_rows(EARLY), [0.5, 1.0]):
        assert found.closed_hour == 2  # an hour fails at the scale
        assert found.replay_driver == pytest.approx(1 / 8)
        assert found.driver_bound == pytest.approx(1 / 8)
        assert found.earlier_driver_bound == 0.0


# --- 2. the every-request route ------------------------------------------------------------------


def test_tier_a_of_the_every_request_route_is_calibrated_with_the_every_scale_bound():
    found = gate_calibrate.every_request(UNIT, CALIBRATION, TEST, tier_b=True)
    assert found["slot_beta"] == pytest.approx(0.5)
    assert found["driver_beta"] == 0.0
    assert found["beta"] == found["beta_used"] == 0.0
    assert found["abstain"] and not found["capped_at_grid_top"]
    assert found["reason"] == (
        "no scale is safe: 20 of 20 calibration days offer; "
        "the driver-loss bound fails at every beta above 0"
    )
    assert found["test_mean_driver_loss_bound_offering_days"] is None
    tier_b = found["tier_b"]
    assert tier_b["beta_used"] == pytest.approx(0.5)
    assert tier_b["test_mean_driver_loss_bound_offering_days"] == pytest.approx(1 / 16)
    assert gate_calibrate.EARLIER not in found


def test_the_earlier_bound_is_reported_beside_it_under_its_own_key():
    plain = gate_calibrate.every_request(UNIT, CALIBRATION, TEST, tier_b=True)
    found = gate_calibrate.every_request(UNIT, CALIBRATION, TEST, tier_b=True, earlier=True)
    earlier = found.pop(gate_calibrate.EARLIER)
    earlier_b = found["tier_b"].pop(gate_calibrate.EARLIER)
    assert found == plain  # the comparison changes no other field
    assert earlier["driver_beta"] > 1.5
    assert earlier["beta"] == earlier["beta_used"] == pytest.approx(0.5)
    assert not earlier["abstain"]
    assert earlier["test_mean_driver_loss_bound_offering_days"] == 0.0
    assert earlier["test_mean_offer_kwh_per_day"] == pytest.approx(0.5 * 2.0)
    assert earlier_b["test_mean_driver_loss_bound_offering_days"] == 0.0
    for block in (earlier, earlier_b):
        assert block["about"] == gate_calibrate.EARLIER_ABOUT
        assert "not a bound where an hour fails" in block["about"]


# --- 3. the sizing what-if -----------------------------------------------------------------------


def test_the_sizing_what_if_calibrates_and_scores_with_the_every_scale_bound():
    terms = [size_scale.day_terms(p) for p in UNIT]
    q = np.ones(len(UNIT))
    scalars = size_scale.calibrate(terms, q, CALIBRATION)
    assert scalars["A"].calibrated == scalars["A"].used == 0.0
    assert scalars["A"].reason == size_scale.NO_SAFE_SCALE
    assert scalars["B"].used == pytest.approx(0.5)
    scored = size_scale.score(terms, q, 0.5, TEST)
    assert scored.driver_losses == pytest.approx((1 / 8, 0.0))


def test_the_size_scales_the_every_scale_bounds_scale():
    terms = [size_scale.day_terms(SWAP)]
    for gamma, expected in ((1.0, 0.03125), (2.0, 0.1875)):
        half = size_scale.score(terms, np.array([0.5]), gamma, [0]).driver_losses[0]
        whole = size_scale.score(terms, np.array([1.0]), 0.5 * gamma, [0]).driver_losses[0]
        assert half == pytest.approx(expected) and whole == pytest.approx(expected)
        assert replayed_driver(SWAP, 0.5 * gamma) == pytest.approx(expected)


# --- 4. T7 ---------------------------------------------------------------------------------------


def test_t7s_driver_term_and_constant_scale_are_the_every_scale_bounds():
    terms = [size_scale.day_terms(p) for p in UNIT]
    losses = t7_stat.day_losses([terms[20], terms[21]], 0.5)
    assert losses["driver"].tolist() == pytest.approx([1 / 8, 0.0])
    found = t7_unit.measure_unit("hand", terms, ["Monday"] * len(UNIT), [(1, CALIBRATION, TEST)])
    (split,) = found["splits"]
    assert split["A"]["used"] == 0.0 and split["A"]["reason"] == size_scale.NO_SAFE_SCALE
    assert split["B"]["used"] == pytest.approx(0.5)


# --- 5. the corner check -------------------------------------------------------------------------


def test_the_corner_check_holds_the_corners_to_the_every_scale_bound():
    checks = corner_search.closed_form_check(E, (0.5, 1.0))
    for check in checks:
        assert check.corner_driver == pytest.approx(1 / 8)
        assert check.driver_bound == pytest.approx(1 / 8)
        assert check.driver_within_bound
        assert check.earlier_driver_bound == 0.0
    entry = corner_summary.pass_mark_entry([0], ["hand"], [checks])
    assert entry["driver_pairs_above_bound"] == 0
    assert entry["largest_driver_excess"] == 0.0
    assert entry["earlier_driver_pairs_above_bound"] == 2
    assert entry["earlier_largest_driver_excess"] == pytest.approx(1 / 8)


# --- 6. replay agreement at each place's own scales ----------------------------------------------


@pytest.fixture(scope="module")
def made_up():
    return offer_days(MADE_UP_DAYS)


def test_replay_is_within_the_comparisons_and_the_corner_checks_bound(made_up):
    scales = (*dial_splits.GRID, *FAR)
    pairs = 0
    for prepared in made_up:
        day, policy = prepared.day, prepared.certificate.policy
        for c in compare_with_replay(day, policy, true_rows(day), scales):
            assert c.replay_driver <= c.driver_bound + TOLERANCE, (day.date, c.beta)
            pairs += 1
    for prepared in made_up[:20]:
        for check in corner_search.closed_form_check(prepared, scales):
            assert check.corner_driver <= check.driver_bound + TOLERANCE
            assert check.driver_within_bound
    assert pairs == MADE_UP_DAYS * len(scales)


def _day_bound(prepared, beta: float) -> float:
    return gate_calibrate.every_scale_loss(gate_calibrate.every_scale_terms(prepared), beta)


@pytest.fixture(scope="module")
def sized():
    return prepared_days(21, 90, early=0.04)


def test_replay_is_within_the_routes_bound_at_both_tiers_scales(sized):
    rng = np.random.default_rng(17)
    scored = 0
    for _ in range(4):
        order = rng.permutation(len(sized))
        calibration, test = sorted(order[:55].tolist()), sorted(order[55:].tolist())
        offering = [d for d in test if not sized[d].abstains]
        found = gate_calibrate.every_request(sized, calibration, test, tier_b=True)
        for block in (found, found["tier_b"]):
            used = float(block["beta_used"])
            if used <= 0:
                continue
            bounds = [_day_bound(sized[d], used) for d in offering]
            assert block["test_mean_driver_loss_bound_offering_days"] == np.mean(bounds)
            for d, bound in zip(offering, bounds, strict=True):
                assert replayed_driver(sized[d], used) <= bound + TOLERANCE
            scored += 1
    assert scored >= 6


def test_replay_is_within_the_sizings_and_t7s_bound_at_their_day_scales(made_up):
    terms = [size_scale.day_terms(p) for p in made_up]
    rng = np.random.default_rng(19)
    checked = 0
    for _ in range(4):
        raw = rng.uniform(0.0, 1.2, MADE_UP_DAYS)
        q = size_scale.sizes(raw, np.ones(MADE_UP_DAYS, dtype=bool), "scale")
        order = rng.permutation(MADE_UP_DAYS)
        calibration, test = sorted(order[:40].tolist()), sorted(order[40:].tolist())
        scalars = size_scale.calibrate(terms, q, calibration)
        for gamma in (*(s.used for s in scalars.values()), 1.0):
            if gamma <= 0:
                continue
            scored = size_scale.score(terms, q, gamma, test)
            for d, loss in zip(test, scored.driver_losses, strict=True):
                assert replayed_driver(made_up[d], gamma * q[d]) <= loss + TOLERANCE
                checked += 1
    for scale in (0.3, 0.6, 1.0):
        for end in t7_stat.window(scale, t7_stat.MAIN_WINDOW):
            losses = t7_stat.day_losses(terms, end)["driver"]
            for prepared, loss in zip(made_up, losses, strict=True):
                assert replayed_driver(prepared, end) <= loss + TOLERANCE
                checked += 1
    assert checked > 300
