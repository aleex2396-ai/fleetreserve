"""The time-ordered run of one unit under the drift guard (plan section 6, T10): the arithmetic.

Plan section 6, T10: "β_{t+1}=β_t+η(α-ℓ_t), unclipped, β<=0 abstaining: average loss after t
days <= α+(β_1/η+1-α)/t on any sequence (Lekeufack et al. 2024, Theorem 1). β_1=0 (set here), η
set in WP9. ... Chronological run, reported only: first 50 days calibrate, then T10". This
module is that run for one unit, given its counting days in time order and a way to evaluate a
day's losses at a scale; `drift_run` is the command that feeds it real days. Everything here is
pure (numpy and the standard library; no files, no solver, no randomness).

Input. Per counting day, in time order (`days_from_truth` sorts a unit's days by date, then
site): its key, its certified value at scale 1 and whether it offers (a day that does not offer
has value 0 and no loss, plan section 6). `evaluate(pairs)`: for a list of (day index, scale)
pairs, a mapping that holds (slot loss, driver loss) for each, both in [0, 1] (for the command:
the one-way rung's day-level losses of `gate_day.evaluate`, as `dial_choice` reads them).

Calibration (`calibrate`). The first `n_calibration` counting days (the plan's 50) calibrate.
Their offering days are evaluated at every scale of `dial_splits.GRID`, and the on-grid
calibration of `dial_splits.on_grid_scale` (the bound of `crc` on the running maximum over the
grid) gives, per tier, the tier's own scale (tier A: slot loss at 0.10 and driver loss at 0.05
together; tier B: the slot loss alone) and, per loss the tier holds down, the scale of that loss
alone (the other alpha set to 1, where the bound always holds). The scales where the bound holds
form a prefix of the grid, so the tier's scale is the smaller of its losses' scales; a test checks
it on a hand case.

Guards. Per tier, one drift guard per loss the tier holds down (`guards`): tier A a slot guard at
alpha 0.10 and a driver guard at alpha 0.05, tier B a slot guard alone (its driver loss is
reported, not held down, as in `dial_splits`). The tiers are run side by side and never pooled.
A path is one tier, one start reading and one step size eta. On each guarded day that offers,
the scale offered is the smallest guard state, capped at `CAP`, and the day abstains (offers
nothing, loses nothing) when that state is 0 or below. The day's losses at the offered scale are
evaluated, and each guard moves by `drift_guard.step(state, its loss, its alpha, eta,
clip=False)`, the plan's rule unchanged.

Why the bound holds for each guard: its state after T steps is beta_1 + eta(T alpha - sum of its
losses). A state of 0 or below makes the offered scale 0 or below, so that day loses nothing and
the state rises; a state above 0 falls by at most eta(1 - alpha) in a day. So the state never
goes below -eta(1 - alpha), and the average loss after T steps is at most alpha +
(beta_1/eta + 1 - alpha)/T. The cap and the minimum over two guards only lower the offered scale,
so a guard at or below 0 still abstains the day, and the argument is unchanged.

Outputs. Per path, one row per guarded counting day (`Row`): the states before the day, the
scale offered, the losses, the offer (scale times value), and after each step the running mean of
both losses and the bound of each guarded loss; and a summary (`Path.summary`).

Readings of the plan's wording, each with its alternative in brackets:

  E1. Start of the guards (`STARTS`, both run side by side, never pooled). "calibrated": each
      guard starts at the first days' on-grid scale of its own loss, so the first guarded day is
      offered at the tier's calibrated scale ("first 50 days calibrate, then T10"); its bound
      then carries beta_1/eta and depends on eta. "zero": each guard starts at 0 (the plan's
      "β_1=0 (set here)"); the first guarded day abstains and the first days' scale is reported
      only; the bound is alpha + (1 - alpha)/t for every eta. [Alternatives: one of the two
      alone; the guard as an offset added to the calibrated scale, which is the "calibrated"
      path under another name.]
  E2. The first `n_calibration` counting days in time order calibrate, whether or not they
      offer; the calibration's n counts the offering ones (plan section 6: "n counts a unit's
      offering calibration days"). They are not offered in this run. [Alternative: the first 50
      offering days.]
  E3. A guarded counting day that does not offer is no step of the guard: it has no loss, its
      offer is 0 and the states stay; t counts offering days (plan section 6: "A day that does
      not offer has no loss and is in neither mean"). A day the guard abstains is a step with
      loss 0. [Alternative: every counting day a step, the non-offering ones with loss 0.]
  E4. The scale offered is capped at `CAP` = 1, the top of `dial_splits.GRID` (the stay-to-bound
      property of tier B is claimed up to scale 1 only); the state itself is unclipped, so after
      many loss-free days it can stand above the cap and take several lossy days to come down
      (counted as `days_state_above_cap`). [Alternatives: no cap, scales above 1 offered; the
      state clipped above at the cap, which keeps the bound too.]
  E5. The loss of a guarded day is the raw day-level loss at the offered scale, not the running
      maximum over the grid (as `dial_splits.score_split` scores test days). [Alternative: the
      running maximum over the grid scales up to the offered scale.]
  E6. Tier A holds both losses down with one guard each and offers the smaller state; tier B
      guards the slot loss alone. [Alternative for tier A: one guard on a combined loss.]
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from fleetreserve import crc, dial_splits, drift_guard

TIERS = dial_splits.TIERS
STARTS = ("calibrated", "zero")
LOSSES = ("slot", "driver")
GRID = dial_splits.GRID
CAP = GRID[-1]  # reading E4
N_CALIBRATION = 50  # plan section 6, T10: "first 50 days calibrate"
TOLERANCE = crc.TOLERANCE


def guards(tier: str) -> dict[str, float]:
    """Loss -> alpha of the losses tier `tier` holds down (module text): an alpha of 1 in
    `dial_splits.tier_alphas` is a loss reported, not held down."""
    return {
        loss: alpha
        for loss, alpha in zip(LOSSES, dial_splits.tier_alphas(tier), strict=True)
        if alpha < 1.0
    }


def bound(alpha: float, beta_1: float, eta: float, steps: int) -> float:
    """T10's bound on the average loss after `steps` steps."""
    return alpha + (beta_1 / eta + 1.0 - alpha) / steps


# --- calibration ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """The first days' calibration (module text)."""

    days: int  # counting days that calibrate
    offering_days: int  # n: those that offer
    first_key: str | None
    last_key: str | None
    tier_scale: dict  # tier -> the tier's on-grid scale
    reason: dict  # tier -> why the tier's scale is 0, empty otherwise
    loss_scale: dict  # tier -> loss -> the on-grid scale of that loss alone (a guard's start)


def calibrate(slot, driver, *, days: int, keys=(None, None)) -> Calibration:
    """The calibration of the module text on the offering calibration days' losses at every grid
    scale (`slot`, `driver`: arrays of shape (offering days, len(GRID)))."""
    slot = np.asarray(slot, dtype=float).reshape(-1, len(GRID))
    driver = np.asarray(driver, dtype=float).reshape(-1, len(GRID))
    tier_scale, reason, loss_scale = {}, {}, {}
    for tier in TIERS:
        alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
        tier_scale[tier], reason[tier] = dial_splits.on_grid_scale(
            slot, driver, alpha_slot, alpha_driver
        )
        loss_scale[tier] = {
            loss: dial_splits.on_grid_scale(
                slot,
                driver,
                alpha if loss == "slot" else 1.0,
                alpha if loss == "driver" else 1.0,
            )[0]
            for loss, alpha in guards(tier).items()
        }
    return Calibration(days, slot.shape[0], *keys, tier_scale, reason, loss_scale)


# --- one path ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One guarded counting day of one path (module text)."""

    day: int
    key: str
    offering: bool
    step: int | None  # t after this day; None on a day that does not offer
    state_before: dict  # guarded loss -> state before the day
    scale: float  # offered; 0 when the day does not offer or the guard abstains
    losses: tuple | None  # (slot, driver) at the scale; None on a day that does not offer
    offer_kwh: float
    mean_loss: dict | None  # loss -> running mean over the steps so far (both losses)
    bound: dict | None  # guarded loss -> T10's bound after this step


@dataclass
class Path:
    """One tier, start reading and step size (module text)."""

    tier: str
    start: str
    eta: float
    beta_1: dict  # guarded loss -> start
    state: dict = field(init=False)
    rows: list = field(default_factory=list)
    steps: int = field(init=False, default=0)
    _sums: dict = field(init=False)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.eta) and self.eta > 0):
            raise ValueError(f"the step size must be finite and above 0, got {self.eta}")
        self.state = dict(self.beta_1)
        self._sums = dict.fromkeys(LOSSES, 0.0)

    def scale(self) -> float:
        """The scale offered on the next offering day: the smallest state, capped; 0 at or
        below 0 (the day abstains)."""
        low = min(self.state.values())
        return 0.0 if low <= 0 else min(low, CAP)

    def advance(self, day: int, key: str, value: float, offering: bool, losses) -> Row:
        """Record day `day`; `losses`: (slot, driver) at `self.scale()`, None when the day does not
        offer or the scale is 0 (then the day loses nothing)."""
        before = dict(self.state)
        if not offering:
            row = Row(day, key, False, None, before, 0.0, None, 0.0, None, None)
            self.rows.append(row)
            return row
        scale = self.scale()
        found = (0.0, 0.0) if scale <= 0 else tuple(float(v) for v in losses)
        if not all(0.0 <= v <= 1.0 for v in found):
            raise ValueError(f"losses must lie in [0, 1], got {found} on day {key}")
        self.steps += 1
        step = self.steps
        for loss, value_of in zip(LOSSES, found, strict=True):
            self._sums[loss] += value_of
        alphas = guards(self.tier)
        for k, loss in enumerate(LOSSES):
            if loss in alphas:
                self.state[loss] = drift_guard.step(
                    self.state[loss], found[k], alphas[loss], self.eta, clip=False
                )
        row = Row(
            day,
            key,
            True,
            step,
            before,
            scale,
            found,
            scale * float(value),
            {loss: self._sums[loss] / step for loss in LOSSES},
            {loss: bound(a, self.beta_1[loss], self.eta, step) for loss, a in alphas.items()},
        )
        self.rows.append(row)
        return row

    def summary(self) -> dict:
        """The path's figures (module text): offers, abstaining days and, per loss, the final
        mean, its alpha and bound, and how often the running mean was over each."""
        rows = self.rows
        stepped = [r for r in rows if r.step is not None]
        abstaining = [r for r in stepped if r.scale <= 0]
        alphas = guards(self.tier)
        losses = {}
        for k, loss in enumerate(LOSSES):
            means = [r.mean_loss[loss] for r in stepped]
            entry = {
                "guarded": loss in alphas,
                "alpha": alphas.get(loss),
                "mean": means[-1] if means else None,
                "failing_days": sum(r.losses[k] > 0 for r in stepped),
            }
            if loss in alphas:
                excess = [r.mean_loss[loss] - r.bound[loss] for r in stepped]
                entry.update(
                    beta_1=self.beta_1[loss],
                    final_state=self.state[loss],
                    final_bound=stepped[-1].bound[loss] if stepped else None,
                    within_alpha=None if not means else means[-1] <= alphas[loss] + TOLERANCE,
                    days_running_mean_over_alpha=sum(m > alphas[loss] + TOLERANCE for m in means),
                    days_over_bound=sum(e > TOLERANCE for e in excess),
                    largest_excess_over_bound=max(excess) if excess else None,
                )
            losses[loss] = entry
        offers = [r.offer_kwh for r in rows]
        offered = [r for r in stepped if r.scale > 0]
        return {
            "tier": self.tier,
            "start": self.start,
            "eta": self.eta,
            "guarded_counting_days": len(rows),
            "offering_days": len(stepped),
            "guard_abstaining_days": len(abstaining),
            "share_guard_abstaining_of_offering_days": (
                len(abstaining) / len(stepped) if stepped else None
            ),
            "days_offering_nothing": sum(1 for r in rows if r.scale <= 0),
            "share_offering_nothing_of_counting_days": (
                sum(1 for r in rows if r.scale <= 0) / len(rows) if rows else None
            ),
            "mean_offer_kwh_per_counting_day": _mean(offers),
            "mean_offer_kwh_per_offering_day": _mean([r.offer_kwh for r in stepped]),
            "mean_scale_offering_days": _mean([r.scale for r in stepped]),
            "days_state_above_cap": sum(1 for r in offered if min(r.state_before.values()) > CAP),
            "losses": losses,
        }


def _mean(values) -> float | None:
    return float(np.mean(values)) if len(values) else None


# --- one unit ------------------------------------------------------------------------------------


@dataclass
class UnitRun:
    """One unit's time-ordered run: `status` "run", or why it did not run."""

    status: str
    calibration: Calibration | None
    paths: list = field(default_factory=list)


def too_few(n: int, n_calibration: int) -> str:
    return (
        f"too few counting days: {n}, and the first {n_calibration} calibrate, so no day is left "
        "for the guard"
    )


def run_unit(
    keys: Sequence[str],
    values: Sequence[float],
    offering: Sequence[bool],
    evaluate: Callable,
    *,
    n_calibration: int = N_CALIBRATION,
    etas: Sequence[float],
    tiers: Sequence[str] = TIERS,
    starts: Sequence[str] = STARTS,
) -> UnitRun:
    """The time-ordered run of the module text on one unit's counting days, in time order."""
    n = len(keys)
    if not (len(values) == len(offering) == n):
        raise ValueError("keys, values and offering must have one entry per day")
    if n_calibration < 1:
        raise ValueError(f"n_calibration must be at least 1, got {n_calibration}")
    if n <= n_calibration:
        return UnitRun(too_few(n, n_calibration), None)
    if any(s not in STARTS for s in starts) or any(t not in TIERS for t in tiers):
        raise ValueError(f"tiers must be among {TIERS} and starts among {STARTS}")
    calibrating = [d for d in range(n_calibration) if offering[d]]
    found = evaluate([(d, g) for d in calibrating for g in GRID])
    slot = [[found[(d, g)][0] for g in GRID] for d in calibrating]
    driver = [[found[(d, g)][1] for g in GRID] for d in calibrating]
    calibration = calibrate(
        slot, driver, days=n_calibration, keys=(keys[0], keys[n_calibration - 1])
    )
    paths = [
        Path(
            tier,
            start,
            float(eta),
            {
                loss: (calibration.loss_scale[tier][loss] if start == "calibrated" else 0.0)
                for loss in guards(tier)
            },
        )
        for tier in tiers
        for start in starts
        for eta in etas
    ]
    for d in range(n_calibration, n):
        losses = {}
        if offering[d]:
            wanted = sorted({p.scale() for p in paths} - {0.0})
            losses = evaluate([(d, s) for s in wanted]) if wanted else {}
        for path in paths:
            scale = path.scale()
            day_losses = losses[(d, scale)] if offering[d] and scale > 0 else None
            path.advance(d, keys[d], values[d], bool(offering[d]), day_losses)
    return UnitRun("run", calibration, paths)
