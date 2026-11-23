"""ALSO-X and CVaR (Lunde et al. 2025) as sizing rules beside conformal risk control (plan section
10: "ALSO-X, CVaR (Lunde et al. 2025)"), for the methods `also-x` and `cvar` of `compare_methods`.

Lunde et al. (2025) size a reserve bid from past samples of a portfolio's flexibility so that it
fits in at least 90% of (minute, sample) pairs, by two approximations of that chance constraint
(`chance_programmes`). They have no per-car controller. Here the box, its shape and the
controller are those every method shares (`compare_run`, readings C1 to C5); only the rule that
picks a split's scale from its calibration days changes: in place of the slot bound of `crc`
(`dial_splits.on_grid_scale`), the largest scale the chance programme accepts on those days; in
tier A the driver bound of `crc` still applies (step 5). The test days are scored at that scale
as every method's are (`dial_splits.score_split`).

`ChanceRule(kind, service_hours, time_limit)` is that rule for one unit under one variant, in the
form `dial_splits.UnitLosses.rule` takes: called with the slot and driver losses of a split's
offering calibration days (shape `(days, len(grid))`), those days' indices, the tier's two alphas
and the grid, it returns `(scale, reason)`, the scale a grid scale or 0 and the reason empty
unless the scale is 0. `service_hours[d]` is the number of service hours of day `d` of the unit.
`also_x_rule` and `cvar_rule` make one for `compare_methods`; `counts()` gives the solves made.
`READINGS` holds the readings below in one line each, as `compare_run` writes them.

Steps, for one split and tier:

1. Too few days: with fewer offering calibration days than `crc` needs for the tier's slot alpha
   (`crc.enough_days`: 9 at 0.10), the scale is 0 for `dial_splits.TOO_FEW` (reading S6).
2. The samples (`pair_thresholds`): every (service hour, day) pair of those days. The slot loss
   at a grid scale times the day's service hours is the number of its last hours lost (the slot
   loss counts the hours from the first failure on, plan section 1), and must be a whole number
   to within `HOURS_TOLERANCE`. The pair of the `i`-th hour from the end is violated at a grid
   scale where at least `i` hours are lost, and its threshold is the largest grid scale below the
   first one where it is violated: 0 when that is the smallest grid scale, the largest grid scale
   when it is never violated (readings S2, S3). A pair is violated from its first violation on,
   whatever the losses at larger scales, as under the running maximum over the grid that
   `on_grid_scale` takes.
3. The programme (`chance_programmes.also_x` or `.cvar`) on those thresholds with `epsilon` the
   tier's slot alpha (0.10 in both tiers: the paper's 90%) and `top` the largest grid scale, with
   the rule's time limit (`lp.OTHER_SECONDS`, 300 s, by default). A result of None (a programme
   not optimal) gives the scale 0 for `NOT_OPTIMAL` (reading S11).
4. Snapping: the scale is the largest grid scale at most the programme's value plus the kind's
   snapping tolerance (`SNAP`), since test losses exist only on the grid (reading S10); none is,
   and the scale is 0 for `BELOW_GRID`.
5. The tier: tier B (`alpha_driver` 1) takes that scale. Tier A takes the smaller of it and the
   driver-loss scale of `crc` on the same days (`on_grid_scale` with the slot alpha 1, where the
   slot bound always holds, and the tier's driver alpha), whose 0 carries its own reason
   (reading S5).

The slot scale of steps 1 to 4 depends on the days and their slot losses only, so it is computed
once per split and kept (tier A, tier B, the summary, the split rows and the gate reading of a
split all read it). `counts()`: `calibration_sets`, the number of distinct (days, slot losses)
solved; `solves`, the outcome of every solve counted by outcome (`lp.Outcome` order, zeros
included).

Readings (each the closest reading of the plan and the paper; the design note
kept outside the repository, part A, names them A1 to A8; the
alternative in brackets, with its expected effect):

  S1. (A1) The samples are the split's offering calibration days, the same days every method is
      calibrated on (plan section 10: "the same ... calibration"). [Alternatives: the paper's
      literal form, one kilowatt bid per hour from past days whatever today's cars, which drops
      the day's forecasts and certified box; or today's forecast departure scenarios as samples,
      a sample-based certificate that is a new method, not the paper's. Large either way.]
  S2. (A2) A violation is a (service hour, day) pair whose hour is lost, the counterpart of their
      (minute, sample) pair; on a unit whose days share one number of service hours the share of
      violated pairs is the mean slot loss. On a pooled unit whose days differ in service hours a
      longer day has more pairs. [Alternative: the whole day, violated by any failure; moderate.]
  S3. (A3) A pair's threshold is the largest grid scale at which it holds; between grid scales
      nothing is known, so it is not counted as holding. [Alternative: a bisection per pair with
      new replays; at most one grid step.]
  S4. (A4) The CVaR excess is in scale units, `beta - theta` (the bid minus the flexibility, both
      proportional to the scale here). [Alternative: the hour's missed kWh; moderate, CVaR only.]
  S5. (A5) Tier B is the slot loss alone, the paper's 90% rule. Tier A has no counterpart in the
      paper (it has no driver term): the smaller of this slot scale and the driver scale of `crc`.
      Where the driver scale binds, tier A is crc's driver scale; it equals the re-solve's tier A
      (the smaller of crc's slot scale and the same driver scale) only where crc's slot scale is
      not below the driver scale: on 20 one-hour days with the driver scale 0.4, ALSO-X's slot
      scale 1 and crc's slot scale 0.2, ALSO-X's tier A is 0.4 and the re-solve's 0.2 (a test of
      `tests/wp10/test_chance_sizing.py`).
      [Alternative: a chance rule of the same kind on the driver loss, a rule no paper states;
      moderate.]
  S6. (A6, DECISIVE) The fewest samples are those of `crc` (9 offering calibration days at
      0.10). [Alternative: the paper's sample count, its equation 2, about 154 for one variable:
      with at most 100 calibration days every split abstains and both methods offer 0
      everywhere.]
  S7. (A7) The programmes are solved as written, through the project's sparse programme route
      with a time limit, in place of their closed forms; the tests compare them with closed forms
      and with an independent dense build of the same programmes on made-up samples. [Time
      only.]
  S8. (A8, departs from the design note) The losses are those of the re-solve rung, the method of
      the headline (outcome O1), so `also-x` and `cvar` are rows of every table beside it, paired
      with it on the same splits (`compare_paired`). [Alternative, the design note's: an option
      `--sizing` that applies each rule to every method's losses; a table per rule.]
  S9. The largest scale is the largest grid scale (1: `dial_splits.GRID`), so the programmes'
      `beta` is bounded by it and ALSO-X's `M` is it. [Alternative: no upper bound but the
      samples'; with a sample never violated on the grid the relaxed programme could then pass 1,
      where no test loss exists.]
  S10. Snapping: ALSO-X's value is the bisection's final lower end, below the crossing by at most
      the gap `DELTA` (1e-5) per sample below it, so its tolerance is 1e-4, as in the design note;
      CVaR's value is exact up to the solver's tolerance, so its tolerance is 1e-6, under the
      smallest nonzero distance between a grid scale and CVaR's value: with `epsilon` 0.10 that
      value is the mean of the smallest tenth of the `N` thresholds, the last counted in part,
      (10 S + r t) / N with S the sum of the floor(N / 10) smallest thresholds, t the next one and
      r the remainder of N divided by 10; every grid scale and threshold is a multiple of 0.05,
      so the value and its distance to a grid scale are multiples of 0.05 / N, and a nonzero
      distance is at least 0.05 / N: 6.25e-5 at 800 pairs, above 1e-6 for any N below 50,000.
      [Alternative: 1e-4 for both (the design note), which can raise a CVaR value up to 1e-4
      below a grid scale to it.]
  S11. A programme that does not end optimal is a recorded outcome: in ALSO-X's bisection it
      moves the upper end down; at the end, or for CVaR, the split's scale is 0 (abstain, the
      fall-back every rung has), with the reason `NOT_OPTIMAL`. [Alternative: the scale of `crc`
      as the fall-back.]
"""

from collections import Counter
from typing import get_args

import numpy as np

from fleetreserve import chance_programmes, crc, dial_splits
from fleetreserve.lp import OTHER_SECONDS, Outcome

KINDS = ("also-x", "cvar")
OUTCOMES: tuple[str, ...] = get_args(Outcome)
SNAP = {"also-x": 1e-4, "cvar": 1e-6}  # scale; reading S10
HOURS_TOLERANCE = 1e-6  # hours; a slot loss times the service hours is a whole number to this
READINGS = {
    "S1": (
        "samples: the split's offering calibration days, the days every method is calibrated on; "
        "not built: the paper's one kilowatt bid per hour from past days whatever today's cars, "
        "or today's forecast departure scenarios as samples"
    ),
    "S2": (
        "a violation is a (service hour, day) pair whose hour is lost under the slot loss (the "
        "hours from the first failure on, after the running maximum over the grid); not built: "
        "the whole day violated by any failure"
    ),
    "S3": (
        "a pair's threshold is the largest grid scale at which it holds (0 when it is lost at the "
        "smallest, the largest grid scale when never); not built: a bisection per pair"
    ),
    "S4": "the CVaR excess is beta - theta in scale units; not built: the hour's missed kWh",
    "S5": (
        "tier B: the chance rule on the slot loss alone, epsilon the slot alpha 0.10 (the "
        "paper's 90%); tier A: the smaller of that scale and the driver-loss scale of crc on the "
        "same days; not built: a chance rule on the driver loss"
    ),
    "S6": (
        "fewest samples: those of crc (9 offering calibration days at 0.10); not built: the "
        "paper's equation 2 (about 154), under which every split of at most 100 calibration days "
        "abstains"
    ),
    "S7": (
        "the programmes are solved as written (chance_programmes, sparse programme route, "
        "time limit lp.OTHER_SECONDS), not their closed forms"
    ),
    "S8": (
        "the losses are the re-solve rung's; not built: each rule applied to every method's losses"
    ),
    "S9": ("the largest scale is the largest grid scale (1), which bounds beta and is ALSO-X's M"),
    "S10": (
        "the scale is the largest grid scale at most the programme's value plus 1e-4 (ALSO-X, "
        "whose value is the bisection's lower end) or 1e-6 (CVaR), else 0"
    ),
    "S11": (
        "a programme not optimal: in ALSO-X's bisection it moves the upper end down; at the end, "
        "or for CVaR, the split's scale is 0 (abstain); not built: crc's scale as the fall-back"
    ),
}
NOT_OPTIMAL = "the sizing programme did not end optimal"
BELOW_GRID = "the sizing programme's scale is below the smallest grid scale"


def snap(value: float, grid, tolerance: float) -> float:
    """The largest scale of `grid` at most `value + tolerance`, or 0 when none is."""
    below = [float(g) for g in grid if g <= value + tolerance]
    return max(below) if below else 0.0


def pair_thresholds(slot, service_hours, grid) -> np.ndarray:
    """The threshold of every (service hour, day) pair of the days (step 2 of the module text).
    `slot`: shape (days, len(grid)); `service_hours`: one entry per day."""
    grid = [float(g) for g in grid]
    slot = np.asarray(slot, dtype=float).reshape(-1, len(grid))
    hours = np.asarray(service_hours).reshape(-1)
    if hours.size != slot.shape[0]:
        raise ValueError("service_hours must have one entry per day")
    if not np.all(np.isfinite(slot)) or slot.min(initial=0.0) < 0 or slot.max(initial=0.0) > 1:
        raise ValueError("every slot loss must be finite and in [0, 1]")
    lost = slot * hours[:, None]
    whole = np.round(lost)
    if np.any(np.abs(lost - whole) > HOURS_TOLERANCE):
        raise ValueError("a slot loss times the day's service hours is not a whole number of hours")
    found = []
    for d, n in enumerate(hours):
        for i in range(1, int(n) + 1):
            violated = np.nonzero(whole[d] >= i)[0]
            if not violated.size:
                found.append(grid[-1])
            elif violated[0] == 0:
                found.append(0.0)
            else:
                found.append(grid[violated[0] - 1])
    return np.array(found, dtype=float)


class ChanceRule:
    """The sizing rule of one kind for one unit under one variant (module text)."""

    def __init__(self, kind: str, service_hours, time_limit: float = OTHER_SECONDS):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.service_hours = np.array(service_hours, dtype=int).reshape(-1)
        self.time_limit = time_limit
        self._kept: dict = {}
        self._solves: Counter = Counter()

    def slot_scale(self, slot, used, alpha_slot: float, grid) -> tuple[float, str]:
        """Steps 1 to 4 of the module text, computed once per (days, slot losses)."""
        used = np.asarray(used, dtype=int).reshape(-1)
        slot = np.asarray(slot, dtype=float).reshape(-1, len(grid))
        if not crc.enough_days(len(used), alpha_slot):
            return 0.0, dial_splits.TOO_FEW
        key = (tuple(used.tolist()), slot.tobytes(), alpha_slot, tuple(grid))
        if key not in self._kept:
            theta = pair_thresholds(slot, self.service_hours[used], grid)
            top = float(max(grid))
            solve = chance_programmes.also_x if self.kind == "also-x" else chance_programmes.cvar
            found = solve(theta, alpha_slot, top, self.time_limit)
            self._solves.update(found.outcomes)
            if found.value is None:
                self._kept[key] = (0.0, NOT_OPTIMAL)
            else:
                scale = snap(found.value, grid, SNAP[self.kind])
                self._kept[key] = (scale, "" if scale > 0 else BELOW_GRID)
        return self._kept[key]

    def __call__(self, slot, driver, used, alpha_slot: float, alpha_driver: float, grid):
        """(scale, reason) of one split and tier (module text, steps 1 to 5)."""
        scale, reason = self.slot_scale(slot, used, alpha_slot, grid)
        if scale <= 0:
            return 0.0, reason
        if alpha_driver >= 1:
            return scale, ""
        driver_scale, why = dial_splits.on_grid_scale(slot, driver, 1.0, alpha_driver, grid)
        if driver_scale <= 0:
            return 0.0, why
        return min(scale, driver_scale), ""

    def counts(self) -> dict:
        """`calibration_sets` and `solves` by outcome (module text)."""
        solves = dict.fromkeys(OUTCOMES, 0)
        for outcome, number in self._solves.items():
            solves[outcome] = solves.get(outcome, 0) + number
        return {"calibration_sets": len(self._kept), "solves": solves}


def also_x_rule(service_hours) -> ChanceRule:
    """The rule of the method `also-x` (`compare_methods`)."""
    return ChanceRule("also-x", service_hours)


def cvar_rule(service_hours) -> ChanceRule:
    """The rule of the method `cvar` (`compare_methods`)."""
    return ChanceRule("cvar", service_hours)
