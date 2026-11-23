"""Conformal risk control for the scale beta (plan section 6, T6; WP6 "minimal CRC").

A day has a loss between 0 and 1 as a function of the scale beta >= 0. After the running-max
monotonisation the loss rises with beta and is 0 at beta <= 0. On n calibration days with mean
loss R_n(beta) the calibrated scale is

    beta_hat = sup{beta >= 0 : (n R_n(beta) + 1) / (n + 1) <= alpha}

(Angelopoulos et al., Theorem 1), for the slot loss at alpha = 0.10 and for the driver loss at
alpha_dr = 0.05. The smaller of the two is the scale to use. Nothing is calibrated, and the scale
is 0 (abstain), unless n >= 1/alpha - 1 for each of the two losses.

Two routes, both pure (numpy only, no randomness, no files):

  Exact route. For losses known in closed form as step functions. `beta_hat_exact` and
  `calibrate_exact`.
  Search route. For losses that can only be evaluated (the rungs). `calibrate_search` evaluates
  the grid, then bisects.

Readings of the plan's wording (each is the closest reading; the alternative is in brackets):

  R1. Where a step loss jumps. A day's loss jumps just above its threshold, not at it: the loss at
      beta equal to a threshold is still the lower level. This follows the plan's fixed plan ("t_min
      the first hour whose threshold s_dt is below beta"). [Alternative: jump at the threshold.]
  R2. The supremum is attained. The set of beta where the bound holds is then [0, beta_hat], and
      the exact route returns beta_hat itself, where the bound holds. The bound fails for every
      larger beta. If the bound holds for every beta the exact route returns infinity (the caller
      applies any cap). [Alternative: an open interval, returning a value just below.]
  R3. Equality at the bound holds ("<= alpha"). Sums of losses are compared with a tolerance of
      1e-9 (TOLERANCE) so that float rounding of alpha * (n + 1) cannot flip a tie, for example
      alpha = 1 - 0.9 at n = 9.
  R4. The grid "0.05, 0.1, 0.2, ..., 0.8, 1, 1.5, 2" is read as twelve values: 0.05, 0.1, then
      steps of 0.1 from 0.2 to 0.8, then 1, 1.5, 2 (no 0.9). [Alternative: 0.9 included.]
  R5. Abstain needs both conditions. Calibrating on both losses needs n >= 1/0.10 - 1 = 9 for the
      slot loss and n >= 1/0.05 - 1 = 19 for the driver loss, so the combination abstains below 19
      days; between 9 and 18 days the slot loss alone would be calibratable, but the combination
      is not. [Alternative: calibrate on the slot loss alone between 9 and 18 days.]
  R6. Bisection brackets only the first failing grid value. After the bound fails at a grid value,
      larger grid values are not evaluated: the monotonised mean loss cannot fall as beta rises,
      so the bound cannot hold again. The bisection runs between the last grid value where the
      bound held (0 if none) and that failing value, until the bracket is no wider than the
      tolerance. The returned scale is the lower end of the bracket, the largest evaluated beta
      where the bound holds, so it is never above the true crossing. [Alternative: report the
      middle of the bracket.]
  R7. Monotonisation of the search route uses evaluated points only: the loss of a day at beta is
      the largest raw loss over all evaluated beta' <= beta. A dip hidden between evaluated points
      is not seen. Raw losses are checked to lie in [0, 1].
  R8. Zero loss at beta <= 0 is taken as given: beta = 0 is never evaluated, and a bracket whose
      lower end is 0 stays at 0 when the bound fails at every evaluated beta.
"""

import math
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

ALPHA_SLOT = 0.10
ALPHA_DRIVER = 0.05
# Reading R4.
GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0)
BISECTION_TOLERANCE = 1e-3
# Reading R3.
TOLERANCE = 1e-9

Day = tuple[Sequence[float], Sequence[float]]  # (thresholds, levels) of one day's step loss


# --- the bound and the abstain rule -------------------------------------------------------------


def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")


def enough_days(n: int, alpha: float) -> bool:
    """True when n >= 1/alpha - 1, that is alpha * (n + 1) >= 1 (the bound can hold at beta = 0)."""
    _check_alpha(alpha)
    return alpha * (n + 1) >= 1.0 - TOLERANCE


def min_days(alpha: float) -> int:
    """The smallest n with n >= 1/alpha - 1 (9 for alpha = 0.10, 19 for alpha = 0.05)."""
    _check_alpha(alpha)
    n = max(0, math.ceil(1.0 / alpha - 1.0 - TOLERANCE))
    while not enough_days(n, alpha):  # guards the last float step
        n += 1
    return n


def _bound_holds(loss_sum: float, n: int, alpha: float) -> bool:
    """(n R_n + 1) / (n + 1) <= alpha, with n R_n = loss_sum."""
    return loss_sum + 1.0 <= alpha * (n + 1) + TOLERANCE


# The same test under a public name, for callers that apply the bound to losses of their own.
bound_holds = _bound_holds


# --- exact route --------------------------------------------------------------------------------


def _check_day(thresholds: Sequence[float], levels: Sequence[float]) -> None:
    if len(thresholds) != len(levels):
        raise ValueError("thresholds and levels must have the same length")
    previous_t, previous_level = 0.0, 0.0
    for t, level in zip(thresholds, levels, strict=True):
        if not (math.isfinite(t) and t >= previous_t):
            raise ValueError(f"thresholds must be finite, >= 0 and non-decreasing, got {t}")
        if not (previous_level <= level <= 1.0):
            raise ValueError(f"levels must be non-decreasing and at most 1, got {level}")
        previous_t, previous_level = t, level


def step_loss(thresholds: Sequence[float], levels: Sequence[float], beta: float) -> float:
    """One day's loss at beta: the level of the last threshold strictly below beta, else 0.

    Reading R1: at beta equal to a threshold the loss has not yet jumped.
    """
    _check_day(thresholds, levels)
    loss = 0.0
    for t, level in zip(thresholds, levels, strict=True):
        if t < beta:
            loss = level
    return loss


def beta_hat_exact(days: Sequence[Day], alpha: float) -> float:
    """The exact beta_hat for one loss: sup{beta >= 0 : (sum_d L_d(beta) + 1)/(n + 1) <= alpha}.

    `days`: one (thresholds, levels) pair per day, thresholds >= 0 non-decreasing, levels the loss
    reached once beta is above the threshold (non-decreasing, at most 1); an empty pair is a day
    that never loses. Returns 0.0 when n < 1/alpha - 1 (abstain), the attained supremum (R2) when
    the loss sum exceeds its allowance at some threshold (0.0 too if that threshold is 0), and
    infinity when the bound holds at every beta.
    """
    _check_alpha(alpha)
    n = len(days)
    if not enough_days(n, alpha):
        return 0.0
    events: list[tuple[float, float]] = []
    for thresholds, levels in days:
        _check_day(thresholds, levels)
        previous = 0.0
        for t, level in zip(thresholds, levels, strict=True):
            events.append((float(t), level - previous))
            previous = level
    events.sort()
    total = 0.0
    i = 0
    while i < len(events):
        t = events[i][0]
        while i < len(events) and events[i][0] == t:
            total += events[i][1]
            i += 1
        if not _bound_holds(total, n, alpha):
            return t  # holds at t (R1), fails for every beta above it
    return math.inf


@dataclass(frozen=True)
class ExactCalibration:
    """beta is min(slot_beta, driver_beta); infinity means no limit from either loss.

    abstain is True when beta <= 0; reason says why (empty when not abstaining).
    """

    beta: float
    slot_beta: float
    driver_beta: float
    abstain: bool
    reason: str
    n_days: int


def _abstain_reason(n: int, alpha_slot: float, alpha_driver: float) -> str:
    short = []
    for name, alpha in (("slot", alpha_slot), ("driver", alpha_driver)):
        if not enough_days(n, alpha):
            short.append(f"{name} loss at alpha {alpha} needs n >= {min_days(alpha)}")
    return f"too few days (n = {n}): " + "; ".join(short) if short else ""


def calibrate_exact(
    slot_days: Sequence[Day],
    driver_days: Sequence[Day],
    alpha_slot: float = ALPHA_SLOT,
    alpha_driver: float = ALPHA_DRIVER,
) -> ExactCalibration:
    """beta_hat = min over the slot loss at alpha_slot and the driver loss at alpha_driver (R5)."""
    if len(slot_days) != len(driver_days):
        raise ValueError("slot_days and driver_days must describe the same days")
    n = len(slot_days)
    slot = beta_hat_exact(slot_days, alpha_slot)
    driver = beta_hat_exact(driver_days, alpha_driver)
    beta = min(slot, driver)
    reason = _abstain_reason(n, alpha_slot, alpha_driver)
    if not reason and beta <= 0.0:
        reason = "the bound fails at every beta above 0"
    return ExactCalibration(beta, slot, driver, beta <= 0.0, reason, n)


# --- search route -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Evaluation:
    """One evaluated beta with the mean monotonised loss of each kind and whether the bound held."""

    beta: float
    mean_slot_loss: float
    mean_driver_loss: float
    holds: bool


@dataclass(frozen=True)
class SearchCalibration:
    """Result of `calibrate_search`.

    beta: the largest evaluated beta where the bound holds for both losses (0 if none, then
    abstain). bracket: (beta where the bound holds, first evaluated beta above it where it
    fails); the second is infinity when the grid's top was reached, and both are 0 when nothing
    was evaluated. evaluations: every evaluated beta in the order evaluated, with mean losses
    after monotonisation (end state). n_evaluations: how many calls to `loss_at` were made.
    """

    beta: float
    bracket: tuple[float, float]
    grid_top_reached: bool
    abstain: bool
    reason: str
    n_days: int
    evaluations: tuple[Evaluation, ...]

    @property
    def n_evaluations(self) -> int:
        return len(self.evaluations)


def calibrate_search(
    loss_at: Callable[[float], tuple[np.ndarray, np.ndarray]],
    n_days: int,
    alpha_slot: float = ALPHA_SLOT,
    alpha_driver: float = ALPHA_DRIVER,
    grid: Sequence[float] = GRID,
    tolerance: float = BISECTION_TOLERANCE,
) -> SearchCalibration:
    """Calibrate on losses that can only be evaluated.

    `loss_at(beta)` returns (slot_losses, driver_losses), two arrays of n_days values in [0, 1],
    one per calibration day. It is called once per beta, never twice for the same beta. The grid
    is evaluated in increasing order until the bound first fails (R6), then the bracket between the
    last holding value and the failing one is bisected to `tolerance`. A bisection point's loss is
    the running maximum over every evaluated beta' <= beta (R7). The bound is the one of the module
    text for the slot loss at alpha_slot and the driver loss at alpha_driver; it must hold for
    both. Too few days (R5): nothing is evaluated and the scale is 0.
    """
    _check_alpha(alpha_slot)
    _check_alpha(alpha_driver)
    if n_days < 0 or int(n_days) != n_days:
        raise ValueError(f"n_days must be a non-negative integer, got {n_days}")
    grid = tuple(float(g) for g in grid)
    if not grid or grid[0] <= 0.0 or any(b <= a for a, b in zip(grid, grid[1:], strict=False)):
        raise ValueError("grid must be non-empty, positive and strictly increasing")
    if not tolerance > 0.0:
        raise ValueError("tolerance must be positive")
    n = int(n_days)
    reason = _abstain_reason(n, alpha_slot, alpha_driver)
    if reason:
        return SearchCalibration(0.0, (0.0, 0.0), False, True, reason, n, ())

    # Evaluated points, sorted by beta, with their monotonised losses (R7).
    betas: list[float] = []
    mono: list[np.ndarray] = []  # each of shape (2, n): slot row, driver row
    order: list[float] = []  # betas in the order evaluated

    def holds_for(sums: np.ndarray) -> bool:
        return _bound_holds(float(sums[0]), n, alpha_slot) and _bound_holds(
            float(sums[1]), n, alpha_driver
        )

    def evaluate(beta: float) -> bool:
        i = bisect_left(betas, beta)
        if i < len(betas) and betas[i] == beta:
            raise RuntimeError(f"beta {beta} would be evaluated twice")
        slot, driver = loss_at(beta)
        raw = np.stack([np.asarray(slot, dtype=float), np.asarray(driver, dtype=float)])
        if raw.shape != (2, n):
            raise ValueError(f"loss_at must return two arrays of {n} values, got shape {raw.shape}")
        if not (np.all(np.isfinite(raw)) and raw.min() >= 0.0 and raw.max() <= 1.0):
            raise ValueError("losses must be finite and within [0, 1]")
        value = np.maximum(raw, mono[i - 1]) if i > 0 else raw
        betas.insert(i, beta)
        mono.insert(i, value)
        for j in range(i + 1, len(mono)):  # later points see this one too
            mono[j] = np.maximum(mono[j], value)
        order.append(beta)
        return holds_for(value.sum(axis=1))

    holding, failing = 0.0, math.inf
    for g in grid:
        if evaluate(g):
            holding = g
        else:
            failing = g
            break
    top_reached = math.isinf(failing)
    if not top_reached:
        while failing - holding > tolerance:
            mid = (holding + failing) / 2.0
            if not (holding < mid < failing):
                break
            if evaluate(mid):
                holding = mid
            else:
                failing = mid

    evaluations = []
    for b in order:
        value = mono[bisect_left(betas, b)]
        sums = value.sum(axis=1)
        evaluations.append(Evaluation(b, float(sums[0]) / n, float(sums[1]) / n, holds_for(sums)))
    reason = ""
    if holding <= 0.0:
        reason = "the bound fails at every evaluated beta down to the tolerance"
    return SearchCalibration(
        holding, (holding, failing), top_reached, holding <= 0.0, reason, n, tuple(evaluations)
    )
