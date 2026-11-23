"""The arithmetic of the fleet-size what-if (`fleet_size_whatif_core`) on hand-made losses.

The unit has six counting days with 2, 3, 3, 5, 2 and 3 steered cars; days 2 and 4 do not offer.
Offering days have certified values 4, 6, 10 and 5 and, at the k-th grid scale, a slot loss of
(day + 1) * 0.01 * (k + 1) and twice that driver loss.

1. `restrict` with a minimum of 0 changes nothing; with a minimum N it turns every day of fewer
   than N cars into a day that does not offer (value 0, no loss, not offering) and touches nothing
   else; a day of exactly N cars stays.
2. `by_count` gives one row per exact count, with sums worked out by hand, and its rows add up to
   the unit's totals.
3. `below` counts the days under a minimum.
4. A day that stops offering also leaves the calibration: on a split where a few small fleets
   carry all the losses the scale rises from 0 to the top of the grid, and the test offer drops
   the small fleets' days.
5. Bad counts and bad minimums are refused.
6. `gate_reading` restricts the configuration by its counts and its oracle partner by the
   partner's own counts; `differing` counts the days on which the two counts differ.
"""

import numpy as np
import pytest

from fleetreserve import dial_splits
from fleetreserve import fleet_size_whatif_core as core
from fleetreserve.dial_splits import UnitLosses

GRID = len(dial_splits.GRID)
COUNTS = [2, 3, 3, 5, 2, 3]
OFFERING = np.array([True, True, False, True, False, True])
VALUE = np.array([4.0, 6.0, 0.0, 10.0, 0.0, 5.0])
CEILING = np.array([5.0, 7.0, 1.0, 11.0, 1.0, 6.0])


def unit() -> UnitLosses:
    slot = np.zeros((6, GRID))
    for d in range(6):
        if OFFERING[d]:
            slot[d] = (d + 1) * 0.01 * (np.arange(GRID) + 1)
    return UnitLosses(VALUE.copy(), OFFERING.copy(), slot, 2 * slot, CEILING.copy())


def same(a: UnitLosses, b: UnitLosses) -> bool:
    return all(
        np.array_equal(getattr(a, f), getattr(b, f))
        for f in ("value", "offering", "slot", "driver", "ceiling")
    )


# --- 1. restrict ---------------------------------------------------------------------------------


def test_a_minimum_of_zero_changes_nothing():
    assert same(core.restrict(unit(), COUNTS, 0), unit())


def test_a_minimum_of_the_smallest_count_changes_nothing():
    assert same(core.restrict(unit(), COUNTS, 2), unit())


def test_days_below_the_minimum_become_days_that_do_not_offer_and_nothing_else_changes():
    base = unit()
    found = core.restrict(base, COUNTS, 3)  # days 0 and 4 have 2 cars
    gone = [0, 4]
    assert found.offering.tolist() == [False, True, False, True, False, True]
    assert found.value.tolist() == [0.0, 6.0, 0.0, 10.0, 0.0, 5.0]
    assert (found.slot[gone] == 0).all() and (found.driver[gone] == 0).all()
    kept = [1, 2, 3, 5]  # a day of exactly 3 cars stays
    assert np.array_equal(found.slot[kept], base.slot[kept])
    assert np.array_equal(found.driver[kept], base.driver[kept])
    assert np.array_equal(found.ceiling, base.ceiling)  # the ceiling of a removed day is kept
    assert same(base, unit())  # the input is not changed


def test_a_larger_minimum_keeps_only_the_larger_fleets():
    found = core.restrict(unit(), COUNTS, 4)
    assert found.offering.tolist() == [False, False, False, True, False, False]
    assert found.value.tolist() == [0.0, 0.0, 0.0, 10.0, 0.0, 0.0]
    assert (found.slot[[0, 1, 5]] == 0).all()
    none = core.restrict(unit(), COUNTS, 6)
    assert not none.offering.any() and (none.value == 0).all()
    assert (none.slot == 0).all() and (none.driver == 0).all()


# --- 2. the table by count -----------------------------------------------------------------------


def test_the_table_by_count_is_worked_out_by_hand():
    rows = core.by_count(unit(), COUNTS)
    assert [r["steered_cars"] for r in rows] == [2, 3, 5]
    two, three, five = rows
    assert (two["counting_days"], two["offering_days"]) == (2, 1)
    assert two["offer_at_scale_1_kwh_sum"] == 4.0  # day 0; day 4 does not offer
    assert (three["counting_days"], three["offering_days"]) == (3, 2)
    assert three["offer_at_scale_1_kwh_sum"] == 11.0  # days 1 and 5
    assert (five["counting_days"], five["offering_days"]) == (1, 1)
    assert five["offer_at_scale_1_kwh_sum"] == 10.0
    last = three["scales"][-1]
    assert last["scale"] == 1.0
    assert last["slot_loss_sum"] == pytest.approx((2 + 6) * 0.01 * GRID)
    assert last["driver_loss_sum"] == pytest.approx(2 * (2 + 6) * 0.01 * GRID)
    assert (last["days_with_a_slot_loss"], last["days_with_a_driver_loss"]) == (2, 2)
    first = two["scales"][0]
    assert first["scale"] == dial_splits.GRID[0]
    assert first["slot_loss_sum"] == pytest.approx(0.01)
    assert (first["days_with_a_slot_loss"], first["days_with_a_driver_loss"]) == (1, 1)
    assert [s["scale"] for s in five["scales"]] == list(dial_splits.GRID)


def test_a_day_with_a_slot_loss_and_no_driver_loss_is_counted_in_its_own_column():
    base = unit()
    slot = base.slot.copy()
    driver = base.driver.copy()
    driver[1] = 0.0  # day 1: slot loss only
    found = core.by_count(UnitLosses(base.value, base.offering, slot, driver, base.ceiling), COUNTS)
    three = found[1]["scales"][-1]
    assert (three["days_with_a_slot_loss"], three["days_with_a_driver_loss"]) == (2, 1)


def test_the_table_adds_up_to_the_unit_totals():
    base = unit()
    rows = core.by_count(base, COUNTS)
    assert sum(r["counting_days"] for r in rows) == 6
    assert sum(r["offering_days"] for r in rows) == int(base.offering.sum())
    assert sum(r["offer_at_scale_1_kwh_sum"] for r in rows) == pytest.approx(base.value.sum())
    for k in range(GRID):
        assert sum(r["scales"][k]["slot_loss_sum"] for r in rows) == pytest.approx(
            base.slot[:, k].sum()
        )
        assert sum(r["scales"][k]["driver_loss_sum"] for r in rows) == pytest.approx(
            base.driver[:, k].sum()
        )
        assert sum(r["scales"][k]["days_with_a_slot_loss"] for r in rows) == int(
            (base.slot[:, k] > 0).sum()
        )


# --- 3. below ------------------------------------------------------------------------------------


def test_below_counts_the_days_under_a_minimum():
    assert core.below(unit(), COUNTS, 0) == {"counting_days": 0, "offering_days": 0}
    assert core.below(unit(), COUNTS, 2) == {"counting_days": 0, "offering_days": 0}
    assert core.below(unit(), COUNTS, 3) == {"counting_days": 2, "offering_days": 1}
    assert core.below(unit(), COUNTS, 4) == {"counting_days": 5, "offering_days": 3}
    assert core.below(unit(), COUNTS, 99) == {"counting_days": 6, "offering_days": 4}


# --- 4. calibration sees the removal -------------------------------------------------------------


def test_a_day_that_stops_offering_also_leaves_the_calibration():
    days = 100
    counts = np.array([2] * 10 + [4] * 90)  # the first ten days are the small fleets
    slot = np.zeros((days, GRID))
    slot[:10] = 1.0  # a loss at every scale on the small fleets only
    losses = UnitLosses(
        np.full(days, 8.0),
        np.ones(days, dtype=bool),
        slot,
        np.zeros((days, GRID)),
        np.full(days, 9.0),
    )
    calibration, test = list(range(60)), list(range(60, 100))  # all ten small days calibrate
    before = dial_splits.score_split(losses, 1, calibration, test, "A")
    after = dial_splits.score_split(core.restrict(losses, counts, 3), 1, calibration, test, "A")
    assert before.scale == 0.0 and before.offer_kwh == 0.0
    assert after.scale == 1.0 and after.offering_calibration_days == 50
    assert after.offer_kwh == pytest.approx(8.0)  # the test days are all large fleets here
    # when some test days are small fleets they offer nothing but still count in the mean
    mixed = np.array([2] * 10 + [4] * 70 + [2] * 20)
    again = dial_splits.score_split(core.restrict(losses, mixed, 3), 1, calibration, test, "A")
    assert again.scale == 1.0 and again.offer_kwh == pytest.approx(8.0 * 20 / 40)
    assert again.offering_test_days == 20


# --- 5. refusals ---------------------------------------------------------------------------------


def test_bad_counts_and_minimums_are_refused():
    with pytest.raises(ValueError, match="one entry per counting day"):
        core.restrict(unit(), [1, 2, 3], 1)
    with pytest.raises(ValueError, match="whole numbers"):
        core.by_count(unit(), [2.0, 3.0, 3.0, 5.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="whole numbers"):
        core.below(unit(), [2, 3, 3, 5, 2, -1], 1)
    with pytest.raises(ValueError, match="0 or more"):
        core.restrict(unit(), COUNTS, -1)


# --- 6. the gate reading -------------------------------------------------------------------------


def _gate_inputs():
    days = 100
    own = UnitLosses(
        np.full(days, 8.0),
        np.ones(days, dtype=bool),
        np.zeros((days, GRID)),
        np.zeros((days, GRID)),
        np.full(days, 9.0),
    )
    oracle = UnitLosses(
        np.full(days, 12.0),
        np.ones(days, dtype=bool),
        np.zeros((days, GRID)),
        np.zeros((days, GRID)),
        np.full(days, 9.0),
    )
    counts = np.array([4] * 60 + [2] * 20 + [4] * 20)  # twenty small fleets among the test days
    oracle_counts = np.full(days, 4)  # the partner's fleets are never small
    ceilings = {"hindsight": np.full(days, 10.0), "best": np.full(days, 9.0)}
    splits = [(1, list(range(60)), list(range(60, 100)))]
    return own, counts, oracle, oracle_counts, ceilings, splits


def test_the_gate_reading_with_no_minimum_is_the_gate_split_reading():
    own, counts, oracle, oracle_counts, ceilings, splits = _gate_inputs()
    found = core.gate_reading(own, counts, oracle, oracle_counts, 0, ceilings, splits, ceilings)
    assert found["mean_offer_kwh_per_test_day"] == pytest.approx(8.0)  # scale 1, no losses
    assert found["mean_oracle_offer_kwh_per_test_day"] == pytest.approx(12.0)
    assert found["oracle_to_offer_ratio"] == pytest.approx(1.5)
    assert found["ceilings"]["hindsight"]["share_of_ceiling"] == pytest.approx(0.8)
    assert found["ceilings"]["best"]["share_of_ceiling"] == pytest.approx(8.0 / 9.0)


def test_the_gate_reading_restricts_each_side_by_its_own_count():
    own, counts, oracle, oracle_counts, ceilings, splits = _gate_inputs()
    found = core.gate_reading(own, counts, oracle, oracle_counts, 3, ceilings, splits, ceilings)
    assert found["mean_offer_kwh_per_test_day"] == pytest.approx(8.0 * 20 / 40)  # 20 small days
    assert found["mean_oracle_offer_kwh_per_test_day"] == pytest.approx(12.0)  # no small day
    assert found["oracle_to_offer_ratio"] == pytest.approx(3.0)
    assert found["ceilings"]["hindsight"]["mean_ceiling_kwh_per_test_day"] == 10.0  # kept
    assert found["ceilings"]["hindsight"]["share_of_ceiling"] == pytest.approx(0.4)
    again = core.gate_reading(own, counts, oracle, counts, 3, ceilings, splits, ceilings)
    assert again["mean_oracle_offer_kwh_per_test_day"] == pytest.approx(6.0)  # the other count
    nothing = core.gate_reading(own, counts, oracle, oracle_counts, 5, ceilings, splits, ceilings)
    assert nothing["mean_offer_kwh_per_test_day"] == 0.0
    assert nothing["mean_oracle_offer_kwh_per_test_day"] == 0.0
    assert nothing["oracle_to_offer_ratio"] is None


def test_differing_counts_the_days_where_the_two_counts_differ():
    assert core.differing([2, 3, 4], [2, 4, 4]) == 1
    assert core.differing([2, 3, 4], [2, 3, 4]) == 0
    with pytest.raises(ValueError, match="one entry per counting day"):
        core.differing([1, 2], [1])
