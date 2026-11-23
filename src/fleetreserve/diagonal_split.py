"""The diagonal split of a given box, run as a fixed plan (plan sections 1, 2 and 10: "Diagonal
split (includes homothets, Zhao et al. 2017)").

A diagonal split of the box `m + W zeta` (`W = diag(w)`) on per-car rows is a policy
`u_i = g_i + G_i zeta` with `sum_i g_i(t) = m_t`, `sum_i G_i(t, t) = w_t` on the service hours,
`G_i(t, s) = 0` when `s != t` (each hour's request is shared out within that hour only) and every
row of every car holding for every `zeta` in `[-1, 1]^n` (plan section 2, "diagonal"). A homothet
split of Zhao et al. (2017), their equation 14, gives car `i` the fixed share `beta_i / beta` of
every hour's deviation from the centre: it is the diagonal split with
`G_i(t, t) = (beta_i / beta) w_t`, so the diagonal splits include it.

`split_box(rows, centre, widths, time_limit=...)` finds a diagonal split of the box with centre
`centre` and half-widths `widths` on the rows `rows` by one linear programme:
`certificate.certify` with the product `products.fixed_shape(centre, widths)`, the structure
"diagonal", the solver's own optimal vertex (`certificate.ANY_OPTIMAL`) and the time limit given.
That programme holds the centre at `centre`, makes the half-widths `lam * widths` and maximises
`lam >= 0`; its value `lam*` (`DiagonalSplit.largest_scale`) is the largest multiple of the box's
half-widths, about the same centre, that has a diagonal split on the rows. `lam = 0` with no gains
is a point of the programme whenever some centre split holds every row, as the centre of a
certified box does.

Readings (each is the closest reading of the plan, then of Zhao et al. (2017); the alternative
is in brackets):

  D1. The box has a diagonal split when `lam* >= 1 - SCALE_TOLERANCE` (kind "split"). The plan
      (`DiagonalSplit.plan`) is then the programme's centre split `g` with, in each service hour
      of positive half-width, the programme's gains of that hour multiplied by one factor so that
      they add up to that hour's half-width (`widths_t / sum_i G_i(t, t)`, which is `1 / lam*` up
      to the solver's tolerance), and no gain in an hour of half-width 0 (or, which the
      programme's rows rule out up to rounding, an hour whose gains add up to 0). It is the
      split of the larger box `m +- lam* widths` scaled down to the box, so each row's worst case
      over the box is at most its worst case over that larger box. [Alternative: `lam` fixed at
      1, a programme that only finds some diagonal split of the box, the solver's vertex, with no
      margin chosen.]
  D2. When `ZERO_SCALE < lam* < 1 - SCALE_TOLERANCE` the box has no diagonal split (kind
      "stretched"). The plan is built as in D1: its gains, scaled up by about `1 / lam*`, share
      every request of the box out exactly, and the rows are only certified to hold on the
      requests with `|zeta_t| <= lam*` in every hour; on the paths beyond them the replay records
      what the true cars cannot follow. This is how the plan treats the fixed plan at a scale
      above its certificate (plan section 6: the policy `u_i = g_i + beta G_i zeta` still shares
      the request out, and its losses are those the replay finds), and how Zhao et al. (2017)
      share an aggregate request (their equation 14: fixed shares that always add up to the
      request, the cars' sets guaranteed only inside the sufficient set). [Alternative: run the
      split of the smaller box `m +- lam* w` unscaled: every car stays in its rows on every path,
      but the request is missed in every hour of positive half-width on every path whose
      `|zeta_t|` is above `lam*` (every audited path, whose `zeta_t` are -1 or +1).]
  D3. When `lam* <= ZERO_SCALE` no positive multiple of the box's half-widths about the same
      centre has a diagonal split (up to that tolerance), so the programme gives no gains to
      scale (kind "no gains"). The plan is the programme's centre split `g` with no gains: each
      service hour plans `m_t` whatever the request, and the replay counts a failure in the first
      hour whose request differs from `m_t` by more than its failure tolerance. [Alternative:
      share each hour's deviation by equal
      shares among the cars (a homothet split no programme chose), or by the causal certificate's
      gains (a causal rule, not a diagonal one).]
  D4. When the programme ends in any outcome but "optimal" (a timeout, "infeasible", "unbounded",
      "error") there is no plan (kind "no programme", `plan` None). `DiagonalSplitController`
      then plans every hour by the least-laxity fill (`priority_rules.least_laxity_first`) and
      counts it as planned by the fall-back, as the linear-programme form of ASA plans an hour
      whose programme is not optimal (`asa_controller`, A5). [Alternative: the day counted as
      failing from its first service hour.]

`DiagonalSplitController(split, counted=...)` is a controller of `controller.py`: in hour `t` it
returns `plan_controller.PlanController(split.plan)`'s powers (the plan's powers from the request
variables known by then, a negative one read as 0), or the fall-back of D4. It keeps no state but
its counts. `solver_counts()`: `solves`, the number of solves by outcome (every outcome of
`lp.Outcome`, zeros included); `hours`, the hours it planned; `fallback_hours`, the hours it
planned by the fall-back; `boxes`, the number of boxes of each kind of `BOXES`, in that
order. `solves` and `boxes` count the split's one programme and its kind when `counted` is true
and are all 0 when it is false (a controller that reuses a split another controller already
counted).

The plans are one-way (`plans.Plan` with `two_way` False), as the controller replay is.
"""

from dataclasses import dataclass
from typing import get_args

import numpy as np

from fleetreserve import products
from fleetreserve.carsets import FleetRows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.controller import HourView
from fleetreserve.lp import Outcome
from fleetreserve.plan_controller import PlanController
from fleetreserve.plans import Plan
from fleetreserve.priority_rules import least_laxity_first

SCALE_TOLERANCE = 1e-7  # D1: lam* at or above 1 - this counts as a split of the box
ZERO_SCALE = 1e-9  # D3: lam* at or below this gives no gains to scale
OUTCOMES: tuple[str, ...] = get_args(Outcome)
SPLIT, STRETCHED, NO_GAINS, NO_PROGRAMME = "split", "stretched", "no gains", "no programme"
BOXES = (SPLIT, STRETCHED, NO_GAINS, NO_PROGRAMME)


@dataclass(frozen=True)
class DiagonalSplit:
    """The outcome of `split_box` (module text). `largest_scale` is `lam*`, None unless the
    programme ended optimal; `plan` is None for the kind "no programme"; `seconds` is timing
    only."""

    outcome: str
    kind: str
    largest_scale: float | None
    plan: Plan | None
    seconds: float


def split_box(rows: FleetRows, centre, widths, *, time_limit: float) -> DiagonalSplit:
    """The diagonal split of the box `centre +- widths` on `rows` and its plan (D1 to D4)."""
    centre = np.asarray(centre, dtype=float)
    widths = np.asarray(widths, dtype=float)
    n = rows.n_service
    if centre.shape != (n,) or widths.shape != (n,):
        raise ValueError(f"the box must have one centre and one half-width per hour ({n})")
    found = certify(
        rows,
        products.fixed_shape(centre, widths),
        "diagonal",
        shape_rule=ANY_OPTIMAL,
        time_limit=time_limit,
    )
    if found.outcome != "optimal":
        return DiagonalSplit(found.outcome, NO_PROGRAMME, None, None, found.seconds)
    largest = float(found.scale)
    policy = found.policy
    hours = np.arange(1, n + 1)
    gains = np.zeros_like(policy.G)
    if largest <= ZERO_SCALE:
        kind = NO_GAINS
    else:
        kind = SPLIT if largest >= 1.0 - SCALE_TOLERANCE else STRETCHED
        own = policy.G[:, hours, hours - 1]  # (cars, n): G_i(t, t)
        total = own.sum(axis=0)
        wide = (widths > 0) & (total > 0)
        factor = np.zeros(n)
        factor[wide] = widths[wide] / total[wide]
        gains[:, hours, hours - 1] = own * factor
    return DiagonalSplit(
        "optimal", kind, largest, Plan(policy.g, gains, two_way=False), found.seconds
    )


class DiagonalSplitController:
    """Follows a `DiagonalSplit`'s plan, or the fall-back of D4 (module text)."""

    def __init__(self, split: DiagonalSplit, *, counted: bool):
        if split.kind not in BOXES:
            raise ValueError(f"unknown kind of box {split.kind!r}")
        self.split = split
        self.counted = bool(counted)
        self.follow = None if split.plan is None else PlanController(split.plan)
        self.hours = 0
        self.fallback_hours = 0

    def __call__(self, view: HourView) -> np.ndarray:
        self.hours += 1
        if self.follow is None:
            self.fallback_hours += 1
            return least_laxity_first(view)
        return self.follow(view)

    def solver_counts(self) -> dict:
        """The counts of the module text."""
        solves = dict.fromkeys(OUTCOMES, 0)
        boxes = dict.fromkeys(BOXES, 0)
        if self.counted:
            solves[self.split.outcome] += 1
            boxes[self.split.kind] += 1
        return {
            "solves": solves,
            "hours": self.hours,
            "fallback_hours": self.fallback_hours,
            "boxes": boxes,
        }
