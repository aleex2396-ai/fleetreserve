"""Maximum-entropy feedback as an hourly tracking controller, in a reduced form: least-laxity-first
guarded by the feasibility of the believed future (plan section 10: "Maximum-entropy feedback on
forecast departures (Li et al.)"; outcome O7 of plan section 11).

The method of Li et al. (arXiv 2012.11261), as read there. An operator chooses each hour's
aggregate energy `u_t`; an aggregator splits it among the cars by a fixed scheduler
(least-laxity-first in their experiments) under each car's limits, exact tracking and full
energy delivery (their eqs. 4a to 4e). The aggregator's feedback `p_t(u | u_<t)` is the
maximum-entropy feedback of their Definition III.1, which their Lemma 2 (eq. 8) gives as the
volume of the feasible future trajectories after `u` divided by the volume before. By their
Corollary III.1, an operator that only ever chooses an action with `p_t > 0` keeps the whole
trajectory feasible. The operator of their Algorithm 2 chooses
`u_t = arg inf (c_t(u) - beta_t log p_t(u | u_<t))` (eq. 14), with `c_t` a cost of the hour.
Their experiments learn `p_t` by reinforcement learning on a discrete action set, with the true
departures in the state, and measure tracking by the mean squared error of eq. 15.

This module's form. Here nobody chooses `u_t`: the request `y_t` of the certified box is given,
so the only freedom over least-laxity-first is to deliver something other than the request,
which the replay scores, in a service hour, as a failure from that hour on (hour 0 has no
request; B-R7). `MefController(lo, hi)` plans each hour
`t` (0 to `n`) from the controller's view (`controller.HourView`):

  1. it splits the hour's target (the request; in hour 0 the hour-zero target of
     `controller.hour_zero_target`) by `priority_rules.least_laxity_first`, which plans
     `min(max(target, 0), sum of most)` in total;
  2. if the believed future after that split is met (`mef_future.meets` of `future(view,
     powers)`), it returns that split: the same powers least-laxity-first plans;
  3. otherwise it looks over the totals `u` in `[0, sum of most]` (`priority_rules.most`): the
     totals whose least-laxity split leaves the believed future met form one interval
     `[low, high]` (in exact arithmetic; below), and when it is not empty the controller
     returns the least-laxity split of the total in it nearest the target,
     `min(max(target, low), high)` (a deviating hour);
  4. when no total leaves the believed future met, it returns the split of step 1 (an hour with
     no feasible total).

Why one interval, and how it is found. A larger total gives every car at least as much under the
least-laxity fill (its order does not depend on the total), so every car's remaining need and
remaining room fall; `mef_future.enough` can then only turn true and `mef_future.roomy` only turn
false. `mef_future.within_cars` compares each car's remaining need with its remaining room, which
fall by the same draw, so in exact arithmetic it does not change with the total; in floats its
two sides can round differently when a car's need is above its room by `TOLERANCE` to within that
rounding, and the controller reads it at the total 0. In such an hour step 2 can find the split
of the target unmet while step 3 reads the target's total as feasible and returns that same
split (an hour counted as deviating at least-laxity-first's own total; the tests show one). So
`low` is the smallest total for which `enough` holds and `high` the largest for which `roomy`
holds, each found by bisection to `BISECTION_KWH` (halving `[0, sum of most]`) and kept on the
side where its condition holds; the controller finds no total when `within_cars` fails at 0,
`enough` fails at `sum of most`, `roomy` fails at 0, or `low > high`. A bisection takes at most
`MAX_STEPS` (200) steps; halving a fleet's most of 1 kWh to 10 MWh down to `BISECTION_KWH` takes
30 to 44. One that has not narrowed by then raises `BisectionError` instead of running on
(`compare_run` records it as a failed variant). Below 2^23 kWh (about 8.4 GWh) neighbouring
floats lie less than `BISECTION_KWH` apart, so there every bisection narrows within the limit;
above it some intervals cannot narrow, and the error says so.

The ends of the interval. The split of a total `u + d` (`d >= 0`) gives the cars `d` more in
all than the split of `u`, so each sum `p(A)` and `b(A)` of `mef_future` falls by at most `d`.
Hence, in exact arithmetic, a total `u*` whose believed future is met without the tolerance has
`enough` from `u* - TOLERANCE` up (or from 0) and `roomy` up to `u* + TOLERANCE` (or up to the
fleet's most); a bisection lands within `BISECTION_KWH` of its edge on the holding side, and
`TOLERANCE` equals `BISECTION_KWH`, so inside `(0, sum of most)` the bisections keep `low <= u*
<= high`. At the two ends the tolerance cannot widen the interval (no total lies below 0 or
above the fleet's most), so a bisection there could lose a total that is met only at the end:
`low` is 0 itself when `enough` holds at 0, and `high` is the fleet's most itself when `roomy`
holds there, not a bisection's result. A total met only at the fleet's most (every car must
take its whole most in the hour for its believed need) is so kept: the controller plans every
car's most, where a `high` placed `BISECTION_KWH` below the most would leave `low > high` and an
hour with no feasible total, planned as least-laxity-first plans it. The tests show such hours,
and a day on which the controller, keeping the total at the most, has a driver loss below 1e-9
and least-laxity-first one of 0.5.

The believed future (`future`). The remaining hours are the service hours after `t` (none after
hour `n`). The remaining fleet is the cars the view counts as plugged in. For such a car `i`
with belief `(d_hat, r_hat, cap, need)` and energy `x_i` drawn before the hour, after the split's
power `s_i`: the most it can take in remaining hour `h` is `r_hat min(max(d_hat - h, 0), 1)`;
it may still draw `cap - x_i - s_i`; and it must still draw `need - x_i - s_i - r_hat
max(d_hat - n - 1, 0)`, the last term being the energy its recovery hours supply at its believed
rate before its believed departure (the sum of the shares `f_h` of the recovery hours, as
`day.ForecastCar.recovery_supply` counts it: `day.Day` refuses a departure, true or forecast,
after the day's `n_hours = n + R + 1`, so `d_hat - n - 1` is at most the `R` recovery hours). With
the forecast belief these are the car's forecast rows (`carsets.forecast_rows`) without the
driver-risk rows, with the powers of hours 0 to `t` fixed; with the oracle belief, its true rows.
The remaining box is `[m_h - beta w_h, m_h + beta w_h]` in each remaining hour `h`.

Readings (the owner has not ruled on them; each is the closest reading of the plan and of Li et
al. that this module found, then the alternative in brackets):

  B-R1. The operator's cost is the squared tracking error `(u - y_t)^2` (the paper's own
        measure of tracking, eq. 15) and the weight `beta_t` of eq. 14 tends to 0, so the
        controller delivers the request when its feedback is positive and the feasible total
        nearest the request otherwise. [Alternative: a finite weight, which trades tracking for
        volume of the future in every hour. In a service hour such a trade delivers something
        other than the request, a slot failure from that hour on. In hour 0 it costs no slot
        loss (B-R7) and can leave room that meets a later request this reading fails: one car
        (departure 2, rate 2, cap 2, need 0.5), one service hour, box `[0, 2]`, hour-zero
        target 2; this reading plans 2 in hour 0 and fails a request of 1 in hour 1, while the
        weight 2, with the feedback `p(u)` proportional to the length of the requests of hour 1
        still feasible after `u`, plans 1 and meets it (a hand case of the tests, with a target
        set by hand, not a certified day).]
  B-R2. The future requests the feedback counts are those of today's remaining box at the
        scale, `[m_h - beta w_h, m_h + beta w_h]`. [Alternative: the paper's action set
        instead of the box: in its experiments a discrete set of aggregate powers (0 to 150 kW
        in steps of 15 kW), the real line only in its Section V-D.]
  B-R3. A future request may be split among the cars in any way. [Alternative: least-laxity-
        first in the future hours too, as the paper defines its feasible set through the fixed
        scheduler.]
  B-R4. The feedback is exact (Definition III.1 through the closed form of `mef_future`), not
        learned. This favours the baseline over the paper's learned feedback.
  B-R5. The believed sets have no driver-risk rows. [Alternative: with them, which needs a
        programme per check.]
  B-R6. "Positive feedback" (`p_t > 0`, a positive volume of feasible futures) is read as a
        non-empty set of feasible futures: some request vector of the remaining box can still be
        followed with every believed need met (`mef_future.meets`). In the paper's own setting,
        where every car's total energy is fixed (its eq. 4d), the feasible future trajectories lie
        in a hyperplane and have volume 0, and here a box hour of zero width does the same; the
        non-empty reading keeps what Corollary III.1 uses the feedback for, that a chosen total
        leaves some feasible future. [Alternative: a positive volume within the box's own
        dimensions of positive width.]
  B-R7. The check also runs in hour 0, with the hour-zero target in place of a request; a
        deviation there costs no slot loss. [Alternative: hour 0 split by least-laxity-first
        without the check.]
  B-R8. The remaining fleet is the cars the view counts as plugged in: a car seen to have left
        takes no part in the check, since nothing chosen now can change what it got. [Alternative:
        it stays in with its believed need. With its power limits taken as 0 from the hour it is
        seen to have left, once a car leaves with its believed need unmet (by more than
        `mef_future.TOLERANCE`) no total is feasible for the rest of the day; with its believed
        limits kept (a car that left before its believed departure is still believed able to
        draw until then), the future can stay feasible.]
  B-R9. A deviating hour plans the least-laxity split of its total, as the paper's aggregator
        splits whatever total the operator sends by its fixed scheduler. [Alternative: any split
        of the total that keeps the believed future met.]

What this form cannot show. The paper's use of the feedback is to let an operator shift energy
to cheap hours while staying feasible; with the request given and every miss scored as a
failure, that use cannot appear here. This module is a feasibility-guarded least-laxity-first
built from the paper's feedback, not the published method: it has no price, no learned
feedback and no choice of the hour's total beyond keeping the believed future feasible.

`deviation_counts()`: `hours`, the hours planned; `deviating_hours`, the hours (0 to `n`) planned
as in step 3, at a total other than least-laxity-first's but in the rounding case above;
`deviating_service_hours`, those of them in service hours 1 to `n`; `no_feasible_hours`, the
hours planned as in step 4. The controller keeps no other state. Under the no-pause sensitivity
(`compare_run --no-pause-share 1/3`) a drawn car is not in the steered day the controller is
given, so it is in no view, no believed future, no box and no plan: the counts read as without
the sensitivity, over the steered cars alone, and the hours planned are those of the days that
still offer (the tests check this on a day built by hand and through the command on made-up
units).
"""

import dataclasses

import numpy as np

from fleetreserve.controller import HourView
from fleetreserve.mef_future import Future, enough, meets, roomy, within_cars
from fleetreserve.priority_rules import least_laxity_first, most

BISECTION_KWH = 1e-9  # the width at which the bisections for `low` and `high` stop
MAX_STEPS = 200  # the most steps of one bisection (module text)
COUNTS = ("hours", "deviating_hours", "deviating_service_hours", "no_feasible_hours")


class BisectionError(RuntimeError):
    """A bisection that did not narrow to `BISECTION_KWH` within `MAX_STEPS` steps."""


def _bisect(holds, start: float, end: float, *, rising: bool) -> float:
    """The edge of `holds` on `[start, end]`, kept on its holding side: the smallest total where
    it holds when `rising` (it fails at `start`, holds at `end`), else the largest (it holds at
    `start`, fails at `end`). Raises `BisectionError` when the interval is wider than
    `BISECTION_KWH` before each of `MAX_STEPS` halvings."""
    first = (start, end)
    for _ in range(MAX_STEPS):
        if end - start <= BISECTION_KWH:
            return end if rising else start
        middle = 0.5 * (start + end)
        if holds(middle) == rising:
            end = middle
        else:
            start = middle
    raise BisectionError(
        f"the bisection did not narrow [{first[0]!r}, {first[1]!r}] to {BISECTION_KWH} kWh in "
        f"{MAX_STEPS} steps"
    )


class MefController:
    """Least-laxity-first guarded by the believed future (module text). `lo`, `hi`: the box's
    lowest and highest request in each service hour 1 to `n` (index `t - 1`)."""

    def __init__(self, lo, hi):
        self.lo = np.array(lo, dtype=float).reshape(-1)
        self.hi = np.array(hi, dtype=float).reshape(-1)
        if self.hi.shape != self.lo.shape:
            raise ValueError("lo and hi must have one entry per service hour")
        self.counts = dict.fromkeys(COUNTS, 0)

    def future(self, view: HourView, powers) -> Future:
        """The believed future after `powers` in the view's hour (module text)."""
        if view.n_service != self.lo.size:
            raise ValueError(f"the box has {self.lo.size} service hours, the view {view.n_service}")
        belief = view.belief
        after = view.drawn + np.asarray(powers, dtype=float)
        hours = np.arange(view.hour + 1, view.n_service + 1)
        share = np.clip(belief.departure[:, None] - hours[None, :], 0.0, 1.0)
        supply = belief.rate * np.maximum(belief.departure - view.n_service - 1, 0.0)
        keep = view.plugged
        return Future(
            (belief.rate[:, None] * share)[keep],
            (belief.need - after - supply)[keep],
            (belief.cap - after)[keep],
            self.lo[view.hour :],
            self.hi[view.hour :],
        )

    def _split(self, view: HourView, total: float) -> np.ndarray:
        return least_laxity_first(dataclasses.replace(view, target=total))

    def nearest_total(self, view: HourView) -> float | None:
        """The total of step 3 (module text), or None when no total is feasible."""
        top = float(most(view).sum())

        def after(total: float) -> Future:
            return self.future(view, self._split(view, total))

        if not within_cars(after(0.0)) or not enough(after(top)) or not roomy(after(0.0)):
            return None
        if enough(after(0.0)):
            low = 0.0
        else:
            low = _bisect(lambda u: enough(after(u)), 0.0, top, rising=True)
        if roomy(after(top)):
            high = top
        else:
            high = _bisect(lambda u: roomy(after(u)), 0.0, top, rising=False)
        if low > high:
            return None
        return min(max(view.target, low), high)

    def __call__(self, view: HourView) -> np.ndarray:
        self.counts["hours"] += 1
        planned = least_laxity_first(view)
        if meets(self.future(view, planned)):
            return planned
        total = self.nearest_total(view)
        if total is None:
            self.counts["no_feasible_hours"] += 1
            return planned
        self.counts["deviating_hours"] += 1
        if view.hour >= 1:
            self.counts["deviating_service_hours"] += 1
        return self._split(view, total)

    def deviation_counts(self) -> dict:
        """The counts of the module text."""
        return dict(self.counts)
