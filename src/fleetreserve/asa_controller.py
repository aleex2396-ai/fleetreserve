"""This project's linear-programme form of ASA as an hourly tracking controller (plan section 10),
in two forms: `AsaController` (method `asa`) and `AsaEsController` (method `asa-es`).

A controller of `controller.py`: at the start of each hour it solves the hour's programme and
returns the planned powers of the programme's first hour, the hour of the view. `AsaController`
solves the programme of `asa_programme` (its module text holds the readings A1 to A5 and A7 and
what was read of Lee et al. (2021)); `AsaEsController` solves that of `asa_equal_share`, the same
four goals and then the paper's equal-sharing term in a linear form as a fifth (its readings E1
to E3). The two differ in that programme only (`programme`, the function each calls); everything
below holds for both. Reading A6: it is one-way only; its powers are at least 0 and a two-way plan
is refused. It keeps no state between hours but its counts and its record (below), which no
hour's programme reads: each hour's programme is built from that hour's view and the inputs the
controller was built with (the box centre `m`, the day's number of recovery hours, the time limit
and `solve`, which is passed to the programme's `plan_hour`; None there means `lp.solve_lp`).

Fall-back (A5): when a stage of the hour's programme ends in any outcome but "optimal" (a timeout,
"infeasible", "unbounded", "error"), the controller plans that hour by the least-laxity fill
(`priority_rules.least_laxity_first`) of the same view, and the hour is counted as planned by
the fall-back.

Counts (`solver_counts()`): the number of stage solves by outcome (every outcome of `lp.Outcome`,
zeros included; up to four solves per hour for `AsaController`, five for `AsaEsController`), the
number of hours planned and the number of hours planned by the fall-back, summed over every hour
the controller has planned since it was built.

Record (`hour_plans`): the list of the `asa_programme.HourPlan` that the programme returned for
each hour the controller has planned since it was built, in the order planned, fall-back hours
included (their `HourPlan` has no powers). It is read by `asa_diagnosis_day`; the controller
never reads it, so it changes no plan.

`for_box(policy, beta, plan, n_cars=..., n_recovery=..., time_limit=...)` builds either for one
day at one scale from the day's certified causal affine policy, of which it reads only the
centre `m`, and the fixed plan at that scale, of which it reads only whether it is two-way; it
reads neither the scale `beta` nor the half-widths `w` (each hour's request reaches it through
the view). Refused with `ValueError`: a two-way plan (A6). A box whose lower corner `m - beta w`
is below 0 is not refused: a request below 0 is read as 0 (A7 of `asa_programme`). When the
controller is built, a number of cars that is not a whole number at least 0, a centre that is not
finite, a number of recovery hours that is not a whole number at least 0 and a time limit that is
not a positive finite number are refused with `ValueError` (a missing time limit is a `TypeError`
of the call). When it plans an hour, a view whose number of cars or service hours is not that of
its day and box is refused with `ValueError`.
"""

from typing import get_args

import numpy as np

from fleetreserve import asa_equal_share, asa_programme
from fleetreserve.asa_programme import (
    checked_centre,
    checked_recovery_hours,
    checked_time_limit,
)
from fleetreserve.controller import HourView
from fleetreserve.lp import Outcome
from fleetreserve.plans import Plan
from fleetreserve.policy import AffinePolicy
from fleetreserve.priority_rules import least_laxity_first

OUTCOMES: tuple[str, ...] = get_args(Outcome)


class AsaController:
    """One day's linear-programme form of ASA, four goals (module text)."""

    programme = staticmethod(asa_programme.plan_hour)

    def __init__(self, centre, *, n_cars: int, n_recovery: int, time_limit, solve=None):
        if isinstance(n_cars, bool) or not isinstance(n_cars, int) or n_cars < 0:
            raise ValueError(f"the number of cars must be a whole number >= 0, got {n_cars!r}")
        self.centre = checked_centre(centre, np.asarray(centre).size)
        self.n_cars = n_cars
        self.n_recovery = checked_recovery_hours(n_recovery)
        self.time_limit = checked_time_limit(time_limit)
        self.solve = solve
        self.solves = dict.fromkeys(OUTCOMES, 0)
        self.hours = 0
        self.fallback_hours = 0
        self.hour_plans: list[asa_programme.HourPlan] = []

    @classmethod
    def for_box(
        cls,
        policy: AffinePolicy,
        beta: float,
        plan: Plan,
        *,
        n_cars: int,
        n_recovery: int,
        time_limit,
        solve=None,
    ) -> "AsaController":
        """For the day of `policy`, from its centre `m` (module text)."""
        if plan.two_way:
            raise ValueError(
                "the linear-programme form of ASA is one-way; a two-way plan is refused"
            )
        return cls(
            policy.m, n_cars=n_cars, n_recovery=n_recovery, time_limit=time_limit, solve=solve
        )

    def __call__(self, view: HourView) -> np.ndarray:
        if view.n_cars != self.n_cars:
            raise ValueError(f"the view has {view.n_cars} cars, the controller's day {self.n_cars}")
        if view.n_service != self.centre.size:
            raise ValueError(
                f"the view has {view.n_service} service hours, the box {self.centre.size}"
            )
        found = self.programme(
            view, self.centre, self.n_recovery, time_limit=self.time_limit, solve=self.solve
        )
        for outcome in found.outcomes:
            self.solves[outcome] += 1
        self.hours += 1
        self.hour_plans.append(found)
        if not found.optimal:
            self.fallback_hours += 1
            return least_laxity_first(view)
        return found.powers[:, 0]

    def solver_counts(self) -> dict:
        """The counts of the module text."""
        return {
            "solves": dict(self.solves),
            "hours": self.hours,
            "fallback_hours": self.fallback_hours,
        }


class AsaEsController(AsaController):
    """One day's linear-programme form of ASA with the fifth goal of `asa_equal_share`."""

    programme = staticmethod(asa_equal_share.plan_hour)
