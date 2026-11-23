"""The search route of conformal risk control (plan section 6, protocol) on made-up losses.

`loss_at(beta)` returns (slot, driver) arrays. Counts of `loss_at` calls, the early stop, the
running-maximum monotonisation and agreement with the exact route are checked.
"""

import math

import numpy as np
import pytest

from fleetreserve import crc

N_CASES = 200
SEED = 20261111
GRID = crc.GRID


def test_the_grid_is_the_twelve_values_of_the_plan():
    assert GRID == (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0)
    assert crc.BISECTION_TOLERANCE == 1e-3


class Recorder:
    """A made-up `loss_at` from per-day raw loss functions; records every beta asked."""

    def __init__(self, slot_fns, driver_fns):
        self.slot_fns, self.driver_fns = slot_fns, driver_fns
        self.betas = []

    def __call__(self, beta):
        self.betas.append(beta)
        return (
            np.array([f(beta) for f in self.slot_fns]),
            np.array([f(beta) for f in self.driver_fns]),
        )


def step_fn(threshold, level=1.0):
    return lambda beta: level if beta > threshold else 0.0


def zero_fn(beta):
    return 0.0


def run(slot_fns, driver_fns, **kwargs):
    recorder = Recorder(slot_fns, driver_fns)
    result = crc.calibrate_search(recorder, len(slot_fns), **kwargs)
    return result, recorder


# --- calls: no beta twice, early stop ------------------------------------------------------------


def test_all_zero_losses_reach_the_top_of_the_grid():
    result, recorder = run([zero_fn] * 19, [zero_fn] * 19)
    assert result.beta == 2.0
    assert result.grid_top_reached
    assert result.bracket == (2.0, math.inf)
    assert not result.abstain
    assert recorder.betas == list(GRID)  # the twelve grid values, once each, in order
    assert result.n_evaluations == 12
    assert all(e.holds for e in result.evaluations)


def test_early_stop_after_the_first_failing_grid_value_and_no_repeat():
    # n = 19, alpha 0.1: allowance 1. Two days lose from just above 0.4501, so the bound holds at
    # 0.4 and fails at 0.5. Larger grid values are never asked.
    slot = [step_fn(0.4501), step_fn(0.4501)] + [zero_fn] * 17
    result, recorder = run(slot, [zero_fn] * 19)
    betas = recorder.betas
    assert betas[:6] == [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert not any(b > 0.5 for b in betas)
    assert len(betas) == len(set(betas))
    # bisection of [0.4, 0.5] to 1e-3: widths 0.1 -> 0.05 -> ... -> 0.00078 is 7 halvings
    assert len(betas) == 6 + 7
    assert result.n_evaluations == 13
    assert not result.grid_top_reached
    assert 0.4501 - 1e-3 <= result.beta <= 0.4501
    lo, hi = result.bracket
    assert lo == result.beta
    assert hi - lo <= 1e-3
    assert hi > 0.4501


def test_failing_at_the_smallest_grid_value_bisects_towards_zero():
    # Two days lose from just above 0.01; the bound already fails at 0.05.
    slot = [step_fn(0.01), step_fn(0.01)] + [zero_fn] * 17
    result, recorder = run(slot, [zero_fn] * 19)
    assert recorder.betas[0] == 0.05
    assert all(b < 0.05 for b in recorder.betas[1:])  # nothing above 0.05 is asked
    assert len(recorder.betas) == 1 + 6  # [0, 0.05] halved to 0.00078
    assert 0.01 - 1e-3 <= result.beta <= 0.01
    assert not result.abstain


def test_loss_from_just_above_zero_gives_zero_and_abstains():
    slot = [step_fn(0.0), step_fn(0.0)] + [zero_fn] * 17
    result, recorder = run(slot, [zero_fn] * 19)
    assert result.beta == 0.0
    assert result.abstain
    assert "fails" in result.reason
    assert all(not e.holds for e in result.evaluations)
    assert min(recorder.betas) < 1e-3  # the bisection went down towards 0
    assert result.bracket[0] == 0.0 and result.bracket[1] < 1e-3


def test_driver_loss_binds_when_it_fails_first():
    # n = 19: the driver allowance is 0, so one driver loss from 0.25 breaks it; the slot loss
    # (allowance 1) is fine with that same one loss.
    driver = [step_fn(0.2501)] + [zero_fn] * 18
    slot = [step_fn(0.2501)] + [zero_fn] * 18
    result, _ = run(slot, driver)
    assert 0.2501 - 1e-3 <= result.beta <= 0.2501
    result2, _ = run([zero_fn] * 19, driver)
    assert result2.beta == result.beta


# --- abstain: not enough days, nothing evaluated -------------------------------------------------


@pytest.mark.parametrize(("n", "abstain"), [(8, True), (9, True), (18, True), (19, False)])
def test_abstain_when_too_few_days(n, abstain):
    result, recorder = run([zero_fn] * n, [zero_fn] * n)
    assert result.abstain is abstain
    if abstain:
        assert result.beta == 0.0
        assert recorder.betas == []
        assert result.n_evaluations == 0
        assert result.reason.startswith("too few days")
    else:
        assert result.beta == 2.0


def test_float_edge_at_nine_and_nineteen_days():
    # alpha written as 1 - 0.9 and 1 - 0.95 sit a rounding step off 0.1 and 0.05.
    result, _ = run([zero_fn] * 9, [zero_fn] * 9, alpha_slot=1 - 0.9, alpha_driver=0.5)
    assert not result.abstain
    result, _ = run([zero_fn] * 19, [zero_fn] * 19, alpha_slot=1 - 0.9, alpha_driver=1 - 0.95)
    assert not result.abstain


# --- monotonisation ------------------------------------------------------------------------------


def test_running_maximum_matters_hand_case():
    # n = 19, alpha 0.1: allowance 1. Day A loses everything for beta in (0.15, 0.25] and then
    # none (a dip); day B loses everything from just above 0.3. Raw losses at the grid values
    # 0.2, 0.3, 0.4: raw sums 1, 0, 1 would all hold, and the bound would hold up to 2. With the
    # running maximum day A keeps its loss, so the sum at 0.4 is 2 and the bound fails there;
    # beta_hat is 0.3 (the exact route on the monotonised days gives 0.3).
    def day_a(beta):
        return 1.0 if 0.15 < beta <= 0.25 else 0.0

    result, recorder = run([day_a, step_fn(0.3)] + [zero_fn] * 17, [zero_fn] * 19)
    assert result.beta == 0.3
    assert recorder.betas[:5] == [0.05, 0.1, 0.2, 0.3, 0.4]
    by_beta = {e.beta: e for e in result.evaluations}
    assert by_beta[0.2].mean_slot_loss == pytest.approx(1 / 19)
    assert by_beta[0.3].mean_slot_loss == pytest.approx(1 / 19)  # raw is 0, monotonised 1/19
    assert by_beta[0.4].mean_slot_loss == pytest.approx(2 / 19)
    exact = crc.beta_hat_exact([([0.15], [1.0]), ([0.3], [1.0])] + [([], [])] * 17, 0.10)
    assert exact == 0.3
    # Without the running maximum the raw losses would have held at 0.4: the sum of raw losses.
    raw_sum_at_04 = day_a(0.4) + step_fn(0.3)(0.4)
    assert raw_sum_at_04 == 1.0
    assert crc._bound_holds(raw_sum_at_04, 19, 0.10)
    assert not crc._bound_holds(2.0, 19, 0.10)


def test_a_bump_found_by_bisection_raises_the_reported_loss_at_larger_points():
    # n = 19, allowance 1. Days A and B lose 1 on (0.32, 0.37] only; days C and D lose 1 from just
    # above 0.38. The grid shows 0 up to 0.3 and a raw sum of 2 at 0.4, so 0.4 fails; the
    # bisection of [0.3, 0.4] finds the bump. The running maximum at 0.4 is then 4 days (A to D),
    # and the reported mean there is updated to 4/19, not left at the 2/19 seen when 0.4 was asked.
    def bump(beta):
        return 1.0 if 0.32 < beta <= 0.37 else 0.0

    slot = [bump, bump, step_fn(0.38), step_fn(0.38)] + [zero_fn] * 15
    result, _ = run(slot, [zero_fn] * 19)
    by_beta = {e.beta: e for e in result.evaluations}
    assert by_beta[0.4].mean_slot_loss == pytest.approx(4 / 19)
    assert not by_beta[0.4].holds
    assert 0.32 - 1e-3 <= result.beta <= 0.32


def test_a_bump_between_evaluated_points_is_missed_and_a_wider_one_is_seen():
    # Reading R7: monotonisation uses evaluated points only. A raw loss of 1 on (0.45, 0.46],
    # 0 elsewhere, on two of n = 19 days (allowance 1): no grid value falls in the bump, so it is
    # not seen. A bump that holds a grid value is seen, and the bisection then lands on its start.
    def bump(beta):
        return 1.0 if 0.45 < beta <= 0.46 else 0.0

    result, _ = run([bump, bump] + [zero_fn] * 17, [zero_fn] * 19)
    assert result.beta == 2.0 and result.grid_top_reached  # missed, as documented

    def wide(beta):
        return 1.0 if 0.35 < beta <= 0.55 else 0.0  # holds grid values 0.4 and 0.5

    result, _ = run([wide, wide] + [zero_fn] * 17, [zero_fn] * 19)
    # 0.4 gives raw sum 2 > 1: fails; bisect [0.3, 0.4]: monotonised sum 2 above 0.35.
    assert 0.35 - 1e-3 <= result.beta <= 0.35


# --- agreement with the exact route on a seeded sample --------------------------------------------


class MadeUpDay:
    """A monotone step function plus dips, built so that the search sees every step.

    Each step threshold lies just below a grid value (by 0.002 to 0.03), so that grid value shows
    the step's level before any dip. A dip lies in (that grid value, the next grid value], when
    the day has no step at the next grid value, and drops the raw loss back to the level before
    the step; it covers the next grid value and every bisection point in between. No other
    threshold of the day falls in a dip. So the exact route on the monotone function is what a
    search with the running maximum must reproduce.
    """

    def __init__(self, thresholds, levels, dips):
        self.thresholds, self.levels, self.dips = thresholds, levels, dips

    def mono(self, beta):
        return crc.step_loss(self.thresholds, self.levels, beta)

    def raw(self, beta):
        for lo, hi, level in self.dips:
            if lo < beta <= hi:
                return level
        return self.mono(beta)

    def as_exact(self):
        return (self.thresholds, self.levels)


def make_day(rng, lossy_probability):
    if rng.random() >= lossy_probability:
        return MadeUpDay([], [], [])
    n_steps = int(rng.integers(1, 4))
    cells = sorted(rng.choice(len(GRID) + 2, size=n_steps, replace=False))
    thresholds, dips = [], []
    for m in cells:
        if m < len(GRID):
            thresholds.append(GRID[m] - float(rng.uniform(0.002, 0.03)))
        else:
            # beyond the grid, still increasing with the cell
            thresholds.append(2.05 + 0.4 * (m - len(GRID)) + float(rng.uniform(0.0, 0.3)))
    levels = np.minimum(np.cumsum(rng.uniform(0.2, 1.0, size=n_steps)), 1.0).tolist()
    for k, m in enumerate(cells):
        if m + 1 < len(GRID) and (m + 1) not in cells and rng.random() < 0.6:
            before = levels[k - 1] if k > 0 else 0.0
            dips.append((GRID[m], GRID[m + 1], before))
    return MadeUpDay(thresholds, levels, dips)


def make_case(rng):
    n = int(rng.integers(19, 61))
    p = float(rng.uniform(0.03, 0.35))
    slot = [make_day(rng, p) for _ in range(n)]
    driver = [make_day(rng, p / 3) for _ in range(n)]
    return slot, driver


def test_search_agrees_with_exact_on_a_seeded_sample():
    rng = np.random.default_rng(SEED)
    cases = [make_case(rng) for _ in range(N_CASES)]
    assert len(cases) == 200
    top, interior, at_zero, dip_seen = 0, 0, 0, 0
    for slot, driver in cases:
        n = len(slot)
        seen_dip = []

        def loss_at(beta, slot=slot, driver=driver, seen_dip=seen_dip):
            raw = (np.array([d.raw(beta) for d in slot]), np.array([d.raw(beta) for d in driver]))
            mono = (
                np.array([d.mono(beta) for d in slot]),
                np.array([d.mono(beta) for d in driver]),
            )
            if any((r < m).any() for r, m in zip(raw, mono, strict=True)):
                seen_dip.append(beta)
            return raw

        found = crc.calibrate_search(loss_at, n)
        exact = crc.calibrate_exact([d.as_exact() for d in slot], [d.as_exact() for d in driver])
        dip_seen += bool(seen_dip)
        if exact.beta >= 2.0:
            assert found.beta == 2.0 and found.grid_top_reached
            top += 1
        else:
            assert not found.grid_top_reached
            # the search never goes above the exact value, and is within the tolerance of it
            assert exact.beta - 1e-3 < found.beta <= exact.beta
            if found.beta > 0.0:
                interior += 1
            else:
                at_zero += 1
        assert found.n_evaluations == len(set(e.beta for e in found.evaluations))
    # The sample is not trivial: it reaches the top, ends inside the grid, and meets dips.
    assert top >= 15
    assert interior >= 120
    assert dip_seen >= 100
    assert at_zero + top + interior == N_CASES


def test_non_monotone_dips_change_the_answer_when_the_running_maximum_is_dropped():
    # Same sample generator: with the running maximum switched off (take the raw sums) at least
    # some cases would hold at a larger beta than the exact value. That shows the sample can fail.
    rng = np.random.default_rng(SEED)
    differ = 0
    for _ in range(N_CASES):
        slot, driver = make_case(rng)
        n = len(slot)
        exact = crc.calibrate_exact([d.as_exact() for d in slot], [d.as_exact() for d in driver])
        raw_hold = [
            crc._bound_holds(sum(d.raw(g) for d in slot), n, 0.10)
            and crc._bound_holds(sum(d.raw(g) for d in driver), n, 0.05)
            for g in GRID
        ]
        # the largest grid value where the raw bound holds, at the first failure of the exact one
        first_fail = next((g for g in GRID if g > exact.beta), None)
        if first_fail is not None and raw_hold[GRID.index(first_fail)]:
            differ += 1
    assert differ >= 15


def test_loss_values_outside_zero_one_or_wrong_shape_are_refused():
    with pytest.raises(ValueError, match="within"):
        crc.calibrate_search(lambda b: (np.full(19, 1.5), np.zeros(19)), 19)
    with pytest.raises(ValueError, match="within"):
        crc.calibrate_search(lambda b: (np.full(19, np.nan), np.zeros(19)), 19)
    with pytest.raises(ValueError, match="two arrays of 19"):
        crc.calibrate_search(lambda b: (np.zeros(18), np.zeros(18)), 19)


def test_bad_grid_or_tolerance_refused():
    for grid in ([], [0.0, 1.0], [0.5, 0.5], [1.0, 0.5]):
        with pytest.raises(ValueError, match="grid"):
            crc.calibrate_search(lambda b: (np.zeros(19), np.zeros(19)), 19, grid=grid)
    with pytest.raises(ValueError, match="tolerance"):
        crc.calibrate_search(lambda b: (np.zeros(19), np.zeros(19)), 19, tolerance=0.0)
