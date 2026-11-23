"""The exact route of conformal risk control (plan section 6, T6) on hand-worked step losses.

Each day is (thresholds, levels): the loss is the level of the last threshold strictly below beta.
The bound is (sum of losses + 1) / (n + 1) <= alpha, so the allowance for the loss sum is
c = alpha (n + 1) - 1.
"""

import math

import pytest

from fleetreserve import crc


def binary(threshold):
    """A day that loses everything (loss 1) once beta is above `threshold`."""
    return ([threshold], [1.0])


NEVER = ([], [])


def test_four_days_second_smallest_threshold():
    # n = 4, alpha = 0.4: allowance c = 0.4 * 5 - 1 = 1. One day may lose; the second loss (at the
    # second smallest threshold, 0.5) breaks the bound. So beta_hat = 0.5, where the bound holds.
    days = [binary(0.3), binary(0.5), binary(0.7), binary(0.9)]
    assert crc.beta_hat_exact(days, 0.4) == 0.5
    assert crc.step_loss(*days[1], 0.5) == 0.0  # at the threshold the loss has not jumped
    assert crc.step_loss(*days[1], 0.5000001) == 1.0


def test_tie_at_threshold_convention():
    # Thresholds 0.3, 0.5, 0.5, 0.9, alpha 0.4 (c = 1): at beta = 0.5 the loss sum is 1 (only the
    # day at 0.3), so the bound holds at 0.5; just above 0.5 the sum is 3. Both days at 0.5 jump
    # together, so beta_hat is 0.5.
    days = [binary(0.3), binary(0.5), binary(0.5), binary(0.9)]
    assert crc.beta_hat_exact(days, 0.4) == 0.5
    at_threshold = sum(crc.step_loss(*d, 0.5) for d in days)
    just_above = sum(crc.step_loss(*d, 0.5 + 1e-9) for d in days)
    assert (at_threshold, just_above) == (1.0, 3.0)


def test_equality_at_the_bound_holds():
    # n = 4, alpha = 0.5: c = 1.5. Increments 0.5 (0.2), 0.5 (0.4), 0.5 (0.6), 1.0 (0.8).
    # Sum at beta = 0.8 is 1.5 = c: (1.5 + 1) / 5 = 0.5 <= 0.5 holds. Above 0.8 the sum is 2.5.
    days = [
        ([0.2, 0.6], [0.5, 1.0]),
        ([0.4], [0.5]),
        binary(0.8),
        NEVER,
    ]
    assert crc.beta_hat_exact(days, 0.5) == 0.8
    total = sum(crc.step_loss(t, lv, 0.8) for t, lv in days)
    assert total == 1.5
    assert (total + 1) / 5 <= 0.5


def test_binary_losses_give_the_floor_alpha_n_plus_1_th_smallest():
    # The plan's strict variant: the floor(alpha (n+1))-th smallest threshold. n = 19, alpha =
    # 0.10: floor(2.0) = 2, so the second smallest.
    thresholds = [0.05 * k + 0.013 for k in range(1, 20)]
    days = [binary(t) for t in reversed(thresholds)]  # order of days does not matter
    assert len(days) == 19
    assert crc.beta_hat_exact(days, 0.10) == thresholds[1]


def test_holds_at_every_beta_gives_infinity():
    assert crc.beta_hat_exact([NEVER] * 9, 0.10) == math.inf
    # a few losses within the allowance: n = 19, c = 1, one day loses
    assert crc.beta_hat_exact([binary(0.2)] + [NEVER] * 18, 0.10) == math.inf


def test_loss_from_just_above_zero_gives_zero():
    # n = 9, alpha = 0.1: c = 0, so any loss breaks the bound. A threshold of 0 loses at once.
    days = [binary(0.0)] + [NEVER] * 8
    assert crc.beta_hat_exact(days, 0.10) == 0.0


@pytest.mark.parametrize(
    ("n", "alpha", "expect_enough"),
    [
        (8, 0.10, False),
        (9, 0.10, True),
        (18, 0.05, False),
        (19, 0.05, True),
        (9, 1 - 0.9, True),  # 1 - 0.9 is 0.09999999999999998: tie must not flip
        (19, 1 - 0.95, True),
        (8, 1 - 0.9, False),
    ],
)
def test_abstain_edge_of_one_over_alpha_minus_one(n, alpha, expect_enough):
    assert crc.enough_days(n, alpha) is expect_enough
    zero_days = [NEVER] * n
    result = crc.beta_hat_exact(zero_days, alpha)
    assert (result == math.inf) is expect_enough
    assert (result == 0.0) is (not expect_enough)


def test_min_days():
    assert crc.min_days(0.10) == 9
    assert crc.min_days(0.05) == 19
    assert crc.min_days(1 - 0.9) == 9
    assert crc.min_days(0.5) == 1


def test_combination_is_the_smaller_of_slot_and_driver():
    # n = 19. Slot (alpha 0.10, c = 1): days lose at 0.2, 0.4, 0.6 -> beta 0.4.
    # Driver (alpha 0.05, c = 0): any driver loss breaks it; the first is at 0.7 -> beta 0.7.
    slot = [binary(0.2), binary(0.4), binary(0.6)] + [NEVER] * 16
    driver = [binary(0.7)] + [NEVER] * 18
    result = crc.calibrate_exact(slot, driver)
    assert (result.slot_beta, result.driver_beta, result.beta) == (0.4, 0.7, 0.4)
    assert not result.abstain
    assert result.reason == ""
    # swap which one binds
    result = crc.calibrate_exact(driver, slot)
    assert (result.slot_beta, result.driver_beta, result.beta) == (math.inf, 0.2, 0.2)
    result = crc.calibrate_exact([NEVER] * 19, [NEVER] * 19)
    assert result.beta == math.inf
    assert not result.abstain


@pytest.mark.parametrize(("n", "abstain"), [(8, True), (9, True), (18, True), (19, False)])
def test_combination_abstains_below_19_days(n, abstain):
    # Slot needs n >= 9, driver needs n >= 19 (reading R5): between 9 and 18 days the combination
    # abstains, and the reason names the driver loss only.
    result = crc.calibrate_exact([NEVER] * n, [NEVER] * n)
    assert result.abstain is abstain
    assert (result.beta == 0.0) is abstain
    assert (result.reason != "") is abstain
    if n == 8:
        assert "slot" in result.reason
    if n in (9, 18):
        assert "slot" not in result.reason
        assert "driver" in result.reason
        assert "n >= 19" in result.reason


def test_combination_abstains_when_the_bound_fails_above_zero():
    # n = 19, driver alpha 0.05: allowance 0, so one driver loss from just above 0 breaks it.
    result = crc.calibrate_exact([NEVER] * 19, [binary(0.0)] + [NEVER] * 18)
    assert result.abstain
    assert result.beta == 0.0
    assert "fails" in result.reason


def test_unequal_day_counts_refused():
    with pytest.raises(ValueError, match="same days"):
        crc.calibrate_exact([NEVER] * 19, [NEVER] * 18)


@pytest.mark.parametrize(
    "day",
    [
        ([0.5, 0.2], [0.1, 0.2]),  # thresholds out of order
        ([0.2, 0.5], [0.4, 0.2]),  # levels fall
        ([0.2], [1.5]),  # level above 1
        ([-0.1], [0.5]),  # negative threshold
        ([0.2], [0.5, 0.6]),  # lengths differ
        ([math.inf], [0.5]),
    ],
)
def test_malformed_days_refused(day):
    with pytest.raises(ValueError):
        crc.beta_hat_exact([day] + [NEVER] * 9, 0.10)


def test_alpha_out_of_range_refused():
    for alpha in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            crc.beta_hat_exact([NEVER] * 20, alpha)
