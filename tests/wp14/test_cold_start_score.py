"""The figures of one unit, k and tier (`cold_start_score`), on a loss table written by hand.

Thirteen days: the ten local days 0 to 9 all offer; days 10, 11 and 12 are scored, with certified
values 4, 0 and 6 kWh, day 11 not offering. Slot losses of the local days: 0 at every scale up to
0.5 and 0.5 on day 3 from 0.6 on, driver losses 0. So for the slot loss at alpha 0.10 the bound
(sum + 1) / (n + 1) <= 0.1 holds up to 0.5 (1 / 11) and fails at 0.6 (1.5 / 11): tier B's scale is
0.5. Tier A needs 19 offering days for the driver loss at alpha 0.05 and has 10: it abstains.

At scale 0.5 the scored days that offer have slot losses 0.2 and 0, driver losses 0.5 and 0. By
hand: offer per counting scored day 0.5 x (4 + 0 + 6) / 3 = 5/3; A_k = 1 - (0.2 + 0) / 2 = 0.9;
mean driver loss 0.25; one failing scored day. Hindsight ceilings 10, 5 and 15 (mean 10): Y_k =
(5/3) / 10 = 1/6. Best-causal ceilings 8, missing, 12: no share. Tier A: offer 0, A_k none, Y_k 0.

Every cell not read is NaN, so a figure that read one would not be a number.
"""

import math

import numpy as np
import pytest

from fleetreserve import cold_start_score, dial_splits

HINDSIGHT, BEST = "hindsight-free-decision-hour", "best-causal"
LOCAL, SCORED = list(range(10)), [10, 11, 12]


def _losses(offering_local: int = 10) -> dial_splits.UnitLosses:
    n, columns = 13, len(dial_splits.GRID)
    half, more = dial_splits.GRID.index(0.5), dial_splits.GRID.index(0.6)
    value = np.array([2.0] * 10 + [4.0, 0.0, 6.0])
    offering = np.array([d < offering_local for d in range(10)] + [True, False, True])
    slot = np.full((n, columns), np.nan)
    driver = np.full((n, columns), np.nan)
    slot[:10, :] = 0.0
    driver[:10, :] = 0.0
    slot[3, more:] = 0.5
    slot[10, half], slot[12, half] = 0.2, 0.0
    driver[10, half], driver[12, half] = 0.5, 0.0
    return dial_splits.UnitLosses(value, offering, slot, driver, np.full(n, np.nan))


def _ceilings() -> dict:
    hindsight = np.full(13, np.nan)
    best = np.full(13, np.nan)
    hindsight[SCORED] = [10.0, 5.0, 15.0]
    best[SCORED] = [8.0, np.nan, 12.0]
    return {HINDSIGHT: hindsight, BEST: best}


def test_tier_b_by_hand():
    found = cold_start_score.score_k(_losses(), 10, LOCAL, SCORED, "B", _ceilings())
    assert found["scale"] == 0.5
    assert found["scale_zero_reason"] == ""
    assert found["local_days"] == 10 and found["offering_local_days"] == 10
    assert found["scored_days"] == 3 and found["offering_scored_days"] == 2
    assert found["offer_kwh_per_counting_scored_day"] == pytest.approx(5 / 3)
    assert found["mean_slot_loss_offering_scored_days"] == pytest.approx(0.1)
    assert found["availability"] == pytest.approx(0.9)
    assert found["mean_driver_loss_offering_scored_days"] == pytest.approx(0.25)
    assert found["failing_scored_days"] == 1
    hindsight = found["offer_share"][HINDSIGHT]
    assert hindsight["mean_ceiling_kwh_per_scored_day"] == pytest.approx(10.0)
    assert hindsight["share_of_ceiling"] == pytest.approx(1 / 6)
    best = found["offer_share"][BEST]
    assert best["scored_days_ceiling_missing"] == 1
    assert best["share_of_ceiling"] is None and best["mean_ceiling_kwh_per_scored_day"] is None


def test_tier_a_abstains_below_nineteen_offering_days():
    found = cold_start_score.score_k(_losses(), 10, LOCAL, SCORED, "A", _ceilings())
    assert found["scale"] == 0.0
    assert found["scale_zero_reason"] == dial_splits.TOO_FEW
    assert found["offer_kwh_per_counting_scored_day"] == 0.0
    assert found["availability"] is None
    assert found["failing_scored_days"] is None
    assert found["offer_share"][HINDSIGHT]["share_of_ceiling"] == 0.0


def test_tier_b_abstains_below_nine_offering_days():
    found = cold_start_score.score_k(_losses(8), 10, LOCAL, SCORED, "B", _ceilings())
    assert found["scale"] == 0.0 and found["offering_local_days"] == 8
    assert found["scale_zero_reason"] == dial_splits.TOO_FEW
    assert found["availability"] is None
    assert found["offer_kwh_per_counting_scored_day"] == 0.0


def test_availability_and_share_by_hand():
    assert cold_start_score.availability(None) is None
    assert cold_start_score.availability(0.25) == 0.75
    assert cold_start_score.share(3.0, [4.0, 8.0])["share_of_ceiling"] == 0.5
    assert cold_start_score.share(3.0, [0.0, 0.0])["share_of_ceiling"] is None
    missing = cold_start_score.share(3.0, [4.0, math.nan])
    assert missing["share_of_ceiling"] is None and missing["scored_days_ceiling_missing"] == 1
