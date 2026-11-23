"""The comparison's calibration and scoring on a hand-made unit, worked out by hand.

The unit has 12 days; day d has certified value d + 1 kWh and best-causal ceiling 2 kWh, except
day 11, which abstains (value 0). Two controller methods, given by a made-up evaluation of each
(day, scale): "clean" never loses; "spoiled" loses on day 0 only, slot loss 0.5 and driver loss
0.2 from the scale 0.3 on (0 below it). Two splits: seed 1 calibrates on days 0 to 9 and tests
on days 10 and 11; seed 2 calibrates on days 2 to 11 and tests on days 0 and 1.

By hand (alpha 0.10 for the slot loss; the bound (n R + 1) / (n + 1) <= alpha):

- tier B, seed 1: n = 10 offering calibration days, so the bound holds while the summed slot loss
  is at most 0.1. "spoiled" sums 0 up to the scale 0.2 and 0.5 from 0.3: its scale is 0.2, its
  offer per counting test day 0.2 (11 + 0) / 2 = 1.1, its test losses those of day 10 (the only
  offering test day): 0 and 0; its ceiling share 2.2 / 4 = 0.55. "clean" gets the top of the grid,
  1.0, and offers 11 / 2 = 5.5;
- tier B, seed 2: n = 9 (day 11 abstains and is not counted), enough for alpha 0.10 (n >= 9);
  neither method loses on days 2 to 10, so both get 1.0 and offer (1 + 2) / 2 = 1.5; "spoiled" has
  test losses (0.5 + 0) / 2 = 0.25 and (0.2 + 0) / 2 = 0.1, share 3 / 4 = 0.75;
- tier B over the splits, "spoiled": mean offer 1.3, p10 1.1 + 0.1 (1.5 - 1.1) = 1.14, mean scale
  0.6, mean test slot loss 0.125, mean test driver loss 0.05, one split over the slot limit, mean
  share 0.65;
- tier A: 10 and 9 offering calibration days are fewer than the 19 the driver alpha 0.05 needs,
  so both splits abstain (scale 0, offer 0) for both methods.

The same unit shows that the calibration days are the split's for every method (the spoiled day
lowers "spoiled"'s scale only on the split that calibrates on it) and that the abstaining day
contributes no loss, value 0 and is never evaluated.
"""

from types import SimpleNamespace

import pytest

from fleetreserve import compare_methods, compare_run, dial_splits
from fleetreserve.compare_day import MethodDay

DAYS = 12
CLEAN = compare_methods.Method("clean", "controller", compare_methods.edf)
SPOILED = compare_methods.Method("spoiled", "controller", compare_methods.llf)
SPLITS = [(1, list(range(10)), [10, 11]), (2, list(range(2, 12)), [0, 1])]
PREPARED = [SimpleNamespace(abstains=d == 11, value=0.0 if d == 11 else d + 1.0) for d in range(12)]
CEILINGS = [2.0] * DAYS
SEEN = []


def _batch(jobs):
    found = []
    for prepared, scale, methods in jobs:
        day = PREPARED.index(prepared)
        SEEN.append((day, scale, tuple(m.name for m in methods)))
        hit = day == 0 and scale >= 0.3
        found.append(
            {
                "clean": MethodDay(0.0, 0.0, ()),
                "spoiled": MethodDay(0.5 if hit else 0.0, 0.2 if hit else 0.0, ()),
            }
        )
    return found


@pytest.fixture(scope="module")
def scored():
    SEEN.clear()
    losses, pairs = compare_run.measure_controllers(PREPARED, CEILINGS, (CLEAN, SPOILED), _batch)
    return losses, pairs, compare_run.score_methods(losses, SPLITS, ("A", "B"))


def test_every_offering_day_is_evaluated_once_at_every_scale(scored):
    _, pairs, _ = scored
    assert pairs == 11 * len(dial_splits.GRID)
    assert sorted((d, s) for d, s, _ in SEEN) == sorted(
        (d, s) for d in range(11) for s in dial_splits.GRID
    )
    assert all(names == ("clean", "spoiled") for _, _, names in SEEN)


def test_the_abstaining_day_has_value_zero_and_no_loss(scored):
    losses, _, _ = scored
    for found in losses.values():
        assert not found.offering[11] and found.value[11] == 0.0
        assert not found.slot[11].any() and not found.driver[11].any()
        assert found.offering[:11].all()


def test_both_tiers_by_hand(scored):
    _, _, tiers = scored
    spoiled, clean = tiers["spoiled"]["B"], tiers["clean"]["B"]
    assert spoiled["mean_offer_kwh_per_test_day"] == pytest.approx(1.3)
    assert spoiled["offer_p10"] == pytest.approx(1.14)
    assert spoiled["mean_scale"] == pytest.approx(0.6)
    assert spoiled["splits_that_offer"] == 2 and spoiled["splits_with_test_losses"] == 2
    assert spoiled["mean_test_slot_loss"] == pytest.approx(0.125)
    assert spoiled["mean_test_driver_loss"] == pytest.approx(0.05)
    assert spoiled["slot_loss_over_limit"]["splits"] == 1
    assert spoiled["mean_share_of_best_causal_ceiling"] == pytest.approx(0.65)
    assert clean["mean_offer_kwh_per_test_day"] == pytest.approx((5.5 + 1.5) / 2)
    assert clean["mean_scale"] == 1.0 and clean["mean_test_slot_loss"] == 0.0
    for name in ("clean", "spoiled"):  # tier A: too few offering calibration days
        found = tiers[name]["A"]
        assert found["mean_offer_kwh_per_test_day"] == 0.0
        assert found["splits_that_offer"] == 0
        assert found["splits_at_scale_zero_by_reason"] == {dial_splits.TOO_FEW: 2}
        assert found["mean_test_slot_loss"] is None


def test_each_split_calibrates_every_method_on_its_own_calibration_days(scored):
    losses, _, _ = scored
    by_method = {
        name: [dial_splits.score_split(found, *split, "B") for split in SPLITS]
        for name, found in losses.items()
    }
    assert [s.scale for s in by_method["spoiled"]] == [0.2, 1.0]
    assert [s.scale for s in by_method["clean"]] == [1.0, 1.0]
    for one, two in zip(by_method["spoiled"], by_method["clean"], strict=True):
        assert one.offering_calibration_days == two.offering_calibration_days
        assert one.offering_test_days == two.offering_test_days
    assert [s.offering_calibration_days for s in by_method["clean"]] == [10, 9]
    assert [s.offer_kwh for s in by_method["spoiled"]] == pytest.approx([1.1, 1.5])
