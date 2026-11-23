"""The guarantee of conformal risk control (plan section 6, T6), by simulation with fixed seeds.

Each draw is a set of n + 1 exchangeable made-up days (independent and identically distributed).
The first n calibrate, the last is scored at the calibrated scale. The average loss of the scored
day over many draws must be at most alpha, within a statistical margin.

Margin: the average of N draws of a loss in [0, 1] has standard deviation at most 0.5 / sqrt(N);
the tests allow 4 such deviations (`margin(N)`), about 0.0115 at N = 30000.
"""

import math

import numpy as np

from fleetreserve import crc

N_SETS = 30000
SEED = 20261112


def margin(n_sets):
    return 4 * 0.5 / math.sqrt(n_sets)


def binary(threshold):
    return ([float(threshold)], [1.0])


def mean_loss_of_new_day(thresholds, n, alpha):
    """Mean loss of the scored day over the sets; also the beta_hat of every set.

    `thresholds`: array (sets, n + 1); a day loses everything (loss 1) once beta is above its
    threshold. The scored day is the last column.
    """
    losses, betas = [], []
    for row in thresholds:
        beta = crc.beta_hat_exact([binary(s) for s in row[:n]], alpha)
        betas.append(beta)
        losses.append(crc.step_loss([row[n]], [1.0], beta))
    return float(np.mean(losses)), betas


def test_binary_losses_n14_expected_loss_is_one_fifteenth_and_within_alpha():
    # n = 14, alpha = 0.10: the allowance is 0.5, so beta_hat is the smallest threshold, and the
    # scored day loses with probability exactly 1/15 = floor(alpha (n + 1)) / (n + 1) < alpha.
    n, alpha = 14, 0.10
    thresholds = np.random.default_rng(SEED).random((N_SETS, n + 1))
    mean, betas = mean_loss_of_new_day(thresholds, n, alpha)
    expected = math.floor(alpha * (n + 1)) / (n + 1)
    assert math.isclose(expected, 1 / 15)
    assert all(b == row[:n].min() for b, row in zip(betas, thresholds, strict=True))
    assert mean <= alpha + margin(N_SETS)
    # not trivially zero: close to its theoretical value
    assert abs(mean - expected) <= margin(N_SETS)
    assert mean >= 0.05


def test_the_test_has_power_a_missing_plus_one_or_a_wrong_alpha_breaks_it():
    # Without the "+ 1" in (n R + 1)/(n + 1) the bound is n R / n <= alpha, which is the same as
    # the right bound at alpha' = (alpha n + 1)/(n + 1) = 0.16 for n = 14. A wrong alpha of 0.2
    # is the other mistake. Both give a mean loss above alpha plus the margin, so the guarantee
    # test above would fail on them.
    n, alpha = 14, 0.10
    thresholds = np.random.default_rng(SEED).random((N_SETS, n + 1))
    without_plus_one = (alpha * n + 1) / (n + 1)
    assert math.isclose(without_plus_one, 0.16)
    for wrong_alpha, expected in ((without_plus_one, 2 / 15), (0.20, 3 / 15)):
        mean, _ = mean_loss_of_new_day(thresholds, n, wrong_alpha)
        assert abs(mean - expected) <= margin(N_SETS)
        assert mean > alpha + margin(N_SETS)


def test_binary_losses_n19_is_the_second_smallest_threshold_and_loss_near_alpha():
    # n = 19, alpha = 0.10: floor(0.1 * 20) = 2, the second smallest; the loss probability is
    # 2/20 = alpha (the bound is tight when alpha (n + 1) is a whole number).
    n, alpha = 19, 0.10
    thresholds = np.random.default_rng(SEED + 1).random((N_SETS, n + 1))
    mean, betas = mean_loss_of_new_day(thresholds, n, alpha)
    assert all(b == np.sort(row[:n])[1] for b, row in zip(betas, thresholds, strict=True))
    assert mean <= alpha + margin(N_SETS)
    assert abs(mean - 0.10) <= margin(N_SETS)


def make_day(rng):
    """A multi-step day: 1 to 3 thresholds in (0, 1), increasing levels reaching at most 1."""
    k = int(rng.integers(1, 4))
    thresholds = np.sort(rng.random(k)).tolist()
    levels = np.minimum(np.cumsum(rng.uniform(0.1, 0.7, size=k)), 1.0).tolist()
    return thresholds, levels


def test_multi_step_losses_n19_average_loss_at_most_alpha():
    n, alpha = 19, 0.10
    rng = np.random.default_rng(SEED + 2)
    n_sets = 20000
    losses = []
    for _ in range(n_sets):
        days = [make_day(rng) for _ in range(n + 1)]
        beta = crc.beta_hat_exact(days[:n], alpha)
        losses.append(crc.step_loss(*days[n], beta))
    mean = float(np.mean(losses))
    assert mean <= alpha + margin(n_sets)
    assert mean >= 0.05  # not trivially zero


def test_combined_slot_and_driver_each_within_their_alpha():
    # n = 30: slot allowance floor(0.1 * 31) = 3 -> third smallest; driver floor(0.05 * 31) = 1.
    n = 30
    rng = np.random.default_rng(SEED + 3)
    slot = rng.random((N_SETS, n + 1))
    driver = rng.random((N_SETS, n + 1))
    slot_losses, driver_losses = [], []
    for s_row, d_row in zip(slot, driver, strict=True):
        result = crc.calibrate_exact([binary(s) for s in s_row[:n]], [binary(d) for d in d_row[:n]])
        assert result.beta == min(np.sort(s_row[:n])[2], np.sort(d_row[:n])[0])
        slot_losses.append(crc.step_loss([s_row[n]], [1.0], result.beta))
        driver_losses.append(crc.step_loss([d_row[n]], [1.0], result.beta))
    assert float(np.mean(slot_losses)) <= 0.10 + margin(N_SETS)
    assert float(np.mean(driver_losses)) <= 0.05 + margin(N_SETS)
    assert float(np.mean(driver_losses)) >= 0.01  # not trivially zero
    assert float(np.mean(slot_losses)) >= 0.02
