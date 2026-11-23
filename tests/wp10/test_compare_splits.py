"""The per-split rows of a comparison run (`compare_splits`) on a hand-worked unit.

The unit has 14 days with certified values 1 to 14, except day 13, which abstains (value 0, no
loss). Day 2 has a slot loss of 1 from the scale 0.5 on; day 11 a slot loss of 0.25 and day 12 a
driver loss of 0.5 at every scale. Split seed 7 calibrates on days 0 to 10 and scores days 11 to
13; split seed 8 calibrates on days 3 to 10, 12 and 13 and scores days 0 to 2.

1. Seed 7, tier B: 11 offering calibration days; the bound `(sum + 1) / (n + 1) <= 0.1` holds up
   to the scale 0.4 and fails at 0.5 (day 2), so the scale is 0.4; the offer per counting test day
   is 0.4 (12 + 13 + 0) / 3 = 10 / 3; 2 offering test days, 1 failing (day 11); mean slot loss
   0.125, mean driver loss 0.25; share of the best-causal ceiling (ceilings 20, 30 and 10 on the
   test days) 10 / 60; the mean best-causal ceiling 20; the other ceiling, missing on day 12, is
   empty.
2. Seed 7, tier A: 11 offering calibration days are too few for the driver alpha 0.05, so the
   scale is 0 with its reason; the offer is 0 and the losses, the failing days and the share are
   empty.
3. Seed 8, tier B: day 2 is a test day; the calibration days hold 9 offering days with no slot
   loss, so the scale is 1 and day 2 fails at it.
4. The rows are the fields of `dial_splits.score_split` for every method, tier and split, in the
   order unit, variant, tier, method, seed.
5. The file: one header line of the columns, then one line per row, floats written so that they
   read back to the same number, a missing value as an empty cell.
"""

import csv
import io

import numpy as np
import pytest

from fleetreserve import compare_splits, dial_splits

GRID = dial_splits.GRID
DAYS = 14
CEILINGS = ("hindsight-free-decision-hour", "best-causal")


def _unit(spoil: float = 1.0) -> dial_splits.UnitLosses:
    value = np.arange(1, DAYS + 1, dtype=float)
    value[13] = 0.0
    offering = np.ones(DAYS, dtype=bool)
    offering[13] = False
    slot = np.zeros((DAYS, len(GRID)))
    driver = np.zeros((DAYS, len(GRID)))
    slot[2, GRID.index(0.5) :] = spoil
    slot[11, :] = 0.25
    driver[12, :] = 0.5
    best = np.full(DAYS, 10.0)
    best[11], best[12] = 20.0, 30.0
    return dial_splits.UnitLosses(value, offering, slot, driver, best)


SPLITS = [(7, list(range(11)), [11, 12, 13]), (8, [*range(3, 11), 12, 13], [0, 1, 2])]


def _ceilings():
    hindsight = np.full(DAYS, 12.0)
    hindsight[12] = np.nan
    best = _unit().ceiling
    return {CEILINGS[0]: hindsight, CEILINGS[1]: best}


def _rows(methods=("re-solve",), tiers=("A", "B")):
    losses = {m: _unit() for m in methods}
    return compare_splits.split_rows("u", "forecast", losses, SPLITS, tiers, _ceilings())


def _find(rows, tier, seed, method="re-solve"):
    (row,) = [r for r in rows if (r["tier"], r["seed"], r["method"]) == (tier, seed, method)]
    return row


def test_seed_7_tier_b_by_hand():
    row = _find(_rows(), "B", 7)
    assert row["unit"] == "u" and row["variant"] == "forecast"
    assert row["calibration_days"] == 11 and row["test_days"] == 3
    assert row["offering_calibration_days"] == 11
    assert row["scale"] == 0.4 and row["scale_zero_reason"] == ""
    assert row["offer_kwh_per_counting_test_day"] == pytest.approx(10 / 3)
    assert row["offering_test_days"] == 2 and row["failing_test_days"] == 1
    assert row["mean_test_slot_loss"] == pytest.approx(0.125)
    assert row["mean_test_driver_loss"] == pytest.approx(0.25)
    assert row["share_of_best_causal_ceiling"] == pytest.approx(10 / 60)
    assert row["mean_best-causal_ceiling_kwh_per_test_day"] == pytest.approx(20.0)
    assert row["mean_hindsight-free-decision-hour_ceiling_kwh_per_test_day"] is None


def test_seed_7_tier_a_abstains_with_its_reason():
    row = _find(_rows(), "A", 7)
    assert row["scale"] == 0.0 and row["scale_zero_reason"] == dial_splits.TOO_FEW
    assert row["offer_kwh_per_counting_test_day"] == 0.0
    assert row["offering_test_days"] == 2
    for key in (
        "failing_test_days",
        "mean_test_slot_loss",
        "mean_test_driver_loss",
        "share_of_best_causal_ceiling",
    ):
        assert row[key] is None, key


def test_seed_8_tier_b_by_hand():
    row = _find(_rows(), "B", 8)
    assert row["calibration_days"] == 10
    assert row["offering_calibration_days"] == 9
    assert row["scale"] == 1.0
    assert row["offer_kwh_per_counting_test_day"] == pytest.approx((1 + 2 + 3) / 3)
    assert row["failing_test_days"] == 1 and row["mean_test_slot_loss"] == pytest.approx(1 / 3)
    assert row["mean_hindsight-free-decision-hour_ceiling_kwh_per_test_day"] == 12.0


def test_the_rows_are_score_split_in_order():
    methods = ("re-solve", "llf")
    rows = _rows(methods)
    keys = [(r["tier"], r["method"], r["seed"]) for r in rows]
    assert keys == [(t, m, s) for t in ("A", "B") for m in methods for s, _, _ in SPLITS]
    for row in rows:
        seed, cal, test = next(s for s in SPLITS if s[0] == row["seed"])
        score = dial_splits.score_split(_unit(), seed, cal, test, row["tier"])
        assert row["scale"] == score.scale
        assert row["offer_kwh_per_counting_test_day"] == score.offer_kwh
        assert row["mean_test_slot_loss"] == score.slot_loss
        assert row["mean_test_driver_loss"] == score.driver_loss
        assert row["share_of_best_causal_ceiling"] == score.ceiling_share
        assert row["offering_test_days"] == score.offering_test_days


def test_the_file_reads_back_to_the_same_numbers(tmp_path):
    rows = _rows(("re-solve", "llf"))
    path = tmp_path / "splits.csv"
    count = compare_splits.write(rows, path, CEILINGS)
    assert count == len(rows)
    text = path.read_text()
    lines = text.splitlines()
    assert lines[0].split(",") == list(compare_splits.columns(CEILINGS))
    assert len(lines) == len(rows) + 1 and text.endswith("\n")
    read = list(csv.DictReader(io.StringIO(text)))
    for row, back in zip(rows, read, strict=True):
        for column in compare_splits.columns(CEILINGS):
            value = row[column]
            if value is None:
                assert back[column] == "", column
            elif isinstance(value, float):
                assert float(back[column]) == value, column
            else:
                assert back[column] == str(value), column
