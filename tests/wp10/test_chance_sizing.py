"""The sizing rules of `chance_sizing` on made-up loss tables of the grid `dial_splits.GRID`.

1. Thresholds (`pair_thresholds`): a two-hour day that loses its last hour at 0.2 (a dip to no
   loss at 0.3 left out) and both hours at 0.5 has pairs at 0.1 and 0.4; a one-hour day that never
   fails has 1, one that fails at 0.05 has 0; a slot loss that is not a whole number of hours of
   the day, a loss outside [0, 1] and a number of service hours per day that does not fit are
   refused.
2. The design note's hand case, ten one-hour days with thresholds 0.2, 0.3, 0.5, 0.5, 0.6, 0.7,
   0.8, 1, 1, 1: tier B gives ALSO-X 0.3, CVaR 0.2 and conformal risk control
   (`on_grid_scale`) 0.2; tier A abstains for all three (10 days, the driver alpha needs 19).
3. Tier A on 20 days: the smaller of the chance rule's slot scale and the driver scale of
   `crc`, whichever binds. Where the driver scale binds, tier A is that scale and not the
   re-solve's, which is lower when crc's slot scale is below the driver scale (reading S5: driver
   scale 0.4, ALSO-X's slot scale 1 and crc's 0.2 give ALSO-X 0.4 and the re-solve 0.2, directly
   and through `score_split`). 8 offering days are too few, 9 are enough.
4. A pair is a sample: one two-hour day lost from 0.05 among nine one-hour days that never fail
   gives ALSO-X 0 (2 of 11 pairs violated, 1 allowed), while the same day of one hour gives 1
   (1 of 10).
5. Scales of zero carry their reason: a programme below the smallest grid scale (`BELOW_GRID`),
   a programme not optimal (`NOT_OPTIMAL`, its outcome counted). Snapping: ALSO-X's value within
   1e-4 below a grid scale is raised to it, CVaR's only within 1e-6.
6. The slot scale is solved once per set of days and losses: tiers A and B read one solve; a
   second set is a second; `counts` holds every outcome of `lp.Outcome`, zeros included.
7. Through `dial_splits.score_split`: a `UnitLosses` with a rule gives the rule the offering
   calibration days only (an abstaining calibration day left out) and is scored at the rule's
   scale; without a rule the scale is `on_grid_scale`'s.
8. A kind other than "also-x" and "cvar" is refused; `also_x_rule` and `cvar_rule` make the two.
"""

import numpy as np
import pytest

from fleetreserve import chance_programmes, chance_sizing, crc, dial_splits
from fleetreserve.chance_programmes import ChanceScale
from fleetreserve.chance_sizing import (
    BELOW_GRID,
    NOT_OPTIMAL,
    ChanceRule,
    also_x_rule,
    cvar_rule,
    pair_thresholds,
)
from fleetreserve.dial_splits import GRID, TOO_FEW, UnitLosses, on_grid_scale, score_split

HAND = (0.2, 0.3, 0.5, 0.5, 0.6, 0.7, 0.8, 1.0, 1.0, 1.0)
A = (crc.ALPHA_SLOT, crc.ALPHA_DRIVER)
B = (crc.ALPHA_SLOT, 1.0)


def one_hour(theta: float) -> list[float]:
    """The slot losses of a one-hour day whose hour holds up to `theta` and is lost above it."""
    return [1.0 if g > theta + 1e-12 else 0.0 for g in GRID]


def table(thetas) -> np.ndarray:
    return np.array([one_hour(t) for t in thetas])


def rule_scale(kind, slot, driver, alphas, hours=None):
    days = len(slot)
    rule = ChanceRule(kind, np.ones(days, int) if hours is None else hours)
    return rule(slot, driver, np.arange(days), *alphas, GRID)


# --- 1. thresholds -------------------------------------------------------------------------------


def test_pair_thresholds_of_a_hand_table():
    two = [0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
    slot = np.array([two, one_hour(1.0), one_hour(0.0)])
    found = pair_thresholds(slot, [2, 1, 1], GRID)
    assert found.tolist() == [0.1, 0.4, 1.0, 0.0]


@pytest.mark.parametrize(
    ("row", "hours", "message"),
    [
        ([0.3] + [1.0] * 9, [2], "not a whole number of hours"),
        ([1.5] * 10, [1], r"in \[0, 1\]"),
        ([-0.5] * 10, [2], r"in \[0, 1\]"),
        ([0.0] * 10, [1, 1], "one entry per day"),
    ],
)
def test_pair_thresholds_refuses(row, hours, message):
    with pytest.raises(ValueError, match=message):
        pair_thresholds(np.array([row]), hours, GRID)


# --- 2. the design note's hand case --------------------------------------------------------------


def test_the_hand_case_of_ten_one_hour_days():
    slot, driver = table(HAND), np.zeros((10, len(GRID)))
    assert rule_scale("also-x", slot, driver, B) == (0.3, "")
    assert rule_scale("cvar", slot, driver, B) == (0.2, "")
    assert on_grid_scale(slot, driver, *B) == (0.2, "")
    assert rule_scale("also-x", slot, driver, A) == (0.0, TOO_FEW)
    assert rule_scale("cvar", slot, driver, A) == (0.0, TOO_FEW)
    assert on_grid_scale(slot, driver, *A) == (0.0, TOO_FEW)


# --- 3. tier A and too few days ------------------------------------------------------------------


def test_tier_a_takes_the_smaller_of_the_slot_and_driver_scales():
    slot = table(HAND * 2)  # 20 days: 2 may be violated, ALSO-X is the third smallest, 0.3
    quiet = np.zeros((20, len(GRID)))
    assert rule_scale("also-x", slot, quiet, B) == (0.3, "")
    assert rule_scale("also-x", slot, quiet, A) == (0.3, "")  # the slot scale binds
    driver = quiet.copy()
    driver[0] = [0.0, 0.0, 0.0] + [1.0] * 7  # one day's drivers lose everything from 0.3
    assert on_grid_scale(slot, driver, 1.0, crc.ALPHA_DRIVER) == (0.2, "")
    assert rule_scale("also-x", slot, driver, A) == (0.2, "")  # the driver scale binds
    assert rule_scale("also-x", slot, driver, B) == (0.3, "")


def test_where_the_driver_scale_binds_tier_a_is_the_driver_scale_not_the_re_solves():
    # Reading S5, by hand. 20 one-hour offering days: day 0 lost from 0.2 (threshold 0.1), day 1
    # from 0.3 (threshold 0.2), days 2 to 19 never (threshold 1); day 19's drivers lose 0.5 from
    # 0.5. crc's bound on n = 20 days: summed losses + 1 <= alpha * 21.
    #   slot, alpha 0.10 (<= 2.1): sums 0, 0, 1 at 0.05, 0.1, 0.2 hold; 2 at 0.3 fails -> 0.2.
    #   driver, alpha 0.05 (<= 1.05): sum 0 up to 0.4 holds; 0.5 at 0.5 fails -> 0.4.
    #   the re-solve's tier A (both bounds): the slot bound fails first, at 0.3 -> 0.2.
    #   ALSO-X: 20 pairs, floor(0.10 * 20) = 2 may be violated, so the third smallest
    #   threshold, 1 -> slot scale 1; tier A min(1, 0.4) = 0.4, above the re-solve's 0.2.
    days = 20
    slot = table([0.1, 0.2] + [1.0] * (days - 2))
    driver = np.zeros((days, len(GRID)))
    driver[days - 1] = [0.5 if g >= 0.5 else 0.0 for g in GRID]
    assert on_grid_scale(slot, driver, crc.ALPHA_SLOT, 1.0) == (0.2, "")  # crc's slot scale
    assert on_grid_scale(slot, driver, 1.0, crc.ALPHA_DRIVER) == (0.4, "")  # the driver scale
    assert rule_scale("also-x", slot, driver, B) == (1.0, "")  # ALSO-X's slot scale
    assert rule_scale("also-x", slot, driver, A) == (0.4, "")
    # The same through dial_splits.score_split, as compare_run scores both methods: the re-solve
    # (no rule) and ALSO-X (its rule) on the same losses, one test day after the 20.
    every = days + 1
    slot, driver = np.vstack([slot, one_hour(1.0)]), np.vstack([driver, np.zeros(len(GRID))])
    value, offering, ceiling = np.ones(every), np.ones(every, bool), np.full(every, np.nan)
    resolve = UnitLosses(value, offering, slot, driver, ceiling)
    also_x = UnitLosses(value, offering, slot, driver, ceiling, also_x_rule(np.ones(every, int)))
    calibration, test = list(range(days)), [days]
    assert score_split(resolve, 1, calibration, test, "A").scale == 0.2
    assert score_split(also_x, 1, calibration, test, "A").scale == 0.4


def test_eight_offering_days_are_too_few_and_nine_are_enough():
    driver = np.zeros((9, len(GRID)))
    slot = table([1.0] * 9)
    assert rule_scale("also-x", slot[:8], driver[:8], B) == (0.0, TOO_FEW)
    assert rule_scale("cvar", slot[:8], driver[:8], B) == (0.0, TOO_FEW)
    assert rule_scale("also-x", slot, driver, B) == (1.0, "")
    assert rule_scale("cvar", slot, driver, B) == (1.0, "")


# --- 4. a pair is a sample -----------------------------------------------------------------------


def test_a_longer_day_has_more_samples():
    driver = np.zeros((10, len(GRID)))
    slot = table([0.0] + [1.0] * 9)
    assert rule_scale("also-x", slot, driver, B, np.array([1] * 10)) == (1.0, "")
    assert rule_scale("also-x", slot, driver, B, np.array([2] + [1] * 9)) == (0.0, BELOW_GRID)


# --- 5. reasons and snapping ---------------------------------------------------------------------


def test_a_programme_below_the_smallest_grid_scale_abstains():
    slot, driver = table([0.0] * 10), np.zeros((10, len(GRID)))
    assert rule_scale("also-x", slot, driver, B) == (0.0, BELOW_GRID)
    assert rule_scale("cvar", slot, driver, B) == (0.0, BELOW_GRID)


@pytest.mark.parametrize("kind", ["also-x", "cvar"])
def test_a_programme_not_optimal_abstains_and_is_counted(monkeypatch, kind):
    missing = ChanceScale(None, ("timeout",))
    monkeypatch.setattr(chance_programmes, "also_x", lambda *args: missing)
    monkeypatch.setattr(chance_programmes, "cvar", lambda *args: missing)
    rule = ChanceRule(kind, np.ones(10, int))
    driver = np.zeros((10, len(GRID)))
    assert rule(table(HAND), driver, np.arange(10), *B, GRID) == (0.0, NOT_OPTIMAL)
    assert rule.counts()["solves"]["timeout"] == 1


@pytest.mark.parametrize(
    ("kind", "value", "scale"),
    [
        ("also-x", 0.3 - 0.5e-4, 0.3),
        ("also-x", 0.3 - 2e-4, 0.2),
        ("cvar", 0.3 - 0.5e-6, 0.3),
        ("cvar", 0.3 - 0.5e-5, 0.2),
    ],
)
def test_snapping_by_kind(monkeypatch, kind, value, scale):
    found = ChanceScale(value, ("optimal",))
    monkeypatch.setattr(chance_programmes, "also_x", lambda *args: found)
    monkeypatch.setattr(chance_programmes, "cvar", lambda *args: found)
    driver = np.zeros((10, len(GRID)))
    assert rule_scale(kind, table(HAND), driver, B) == (scale, "")


# --- 6. one solve per set of days ----------------------------------------------------------------


def test_the_slot_scale_is_solved_once_per_set_of_days():
    rule = ChanceRule("cvar", np.ones(12, int))
    slot, driver = table(HAND + (1.0, 1.0)), np.zeros((12, len(GRID)))
    rule(slot[:10], driver[:10], np.arange(10), *B, GRID)
    rule(slot[:10], driver[:10], np.arange(10), *A, GRID)
    counts = rule.counts()
    assert counts == {
        "calibration_sets": 1,
        "solves": dict.fromkeys(chance_sizing.OUTCOMES, 0) | {"optimal": 1},
    }
    assert list(counts["solves"]) == ["optimal", "infeasible", "unbounded", "timeout", "error"]
    rule(slot[2:], driver[2:], np.arange(2, 12), *B, GRID)
    assert rule.counts()["calibration_sets"] == 2
    assert rule.counts()["solves"]["optimal"] == 2


# --- 7. through score_split ----------------------------------------------------------------------


def test_score_split_gives_the_rule_the_offering_calibration_days_and_scores_at_its_scale():
    days = 14
    slot = table(HAND + (1.0, 0.5, 0.5, 0.5))
    driver = np.zeros((days, len(GRID)))
    value = np.arange(1.0, days + 1)
    offering = np.ones(days, bool)
    offering[3] = False
    seen = []

    def rule(s, d, used, alpha_slot, alpha_driver, grid):
        seen.append((s.copy(), used.tolist(), alpha_slot, alpha_driver))
        return 0.3, ""

    losses = UnitLosses(value, offering, slot, driver, np.full(days, np.nan), rule)
    calibration, test = list(range(10)), [10, 11, 12, 13]
    found = score_split(losses, 7, calibration, test, "B")
    used = [d for d in calibration if d != 3]
    assert seen[0][1] == used and seen[0][2:] == B
    assert np.array_equal(seen[0][0], slot[used])
    assert found.scale == 0.3
    assert found.offer_kwh == pytest.approx(0.3 * value[test].mean())
    assert found.slot_loss == pytest.approx(slot[test, GRID.index(0.3)].mean())
    plain = UnitLosses(value, offering, slot, driver, np.full(days, np.nan))
    assert plain.rule is None
    assert (
        score_split(plain, 7, calibration, test, "B").scale
        == on_grid_scale(slot[used], driver[used], *B)[0]
    )


# --- 8. kinds ------------------------------------------------------------------------------------


def test_kinds():
    with pytest.raises(ValueError, match="kind must be one of"):
        ChanceRule("var", [1])
    assert also_x_rule([1, 2]).kind == "also-x" and cvar_rule([1, 2]).kind == "cvar"
    assert also_x_rule([1, 2]).service_hours.tolist() == [1, 2]
    assert chance_sizing.KINDS == ("also-x", "cvar")
    assert dial_splits.TOO_FEW == "too few offering calibration days"
