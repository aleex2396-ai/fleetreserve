"""A controller that follows a plan (`plans.Plan`): meant for the plan section 10 baseline "fixed
plan without re-solving", and the tie between `controller_replay` and `replay.run_plan` in the
tests.

In hour `t` it returns `plan.powers(t, zeta_1 .. zeta_t)` from the view's known request
variables, with a negative planned power read as 0: the controller replay is one-way, and a
negative planned power of a one-way plan draws nothing in `replay.take` either. Besides the hour
and the known request variables it reads only the view's numbers of cars and service hours, which
it checks against the plan; not the target, the energy drawn, the plugged-in flags or the belief,
as `replay.run_plan` does not. A two-way plan is refused.
"""

import numpy as np

from fleetreserve.controller import HourView
from fleetreserve.plans import Plan


class PlanController:
    """Follows `plan` hour by hour (module text)."""

    def __init__(self, plan: Plan):
        if plan.two_way:
            raise ValueError("the controller replay is one-way; a two-way plan is refused")
        self.plan = plan

    def __call__(self, view: HourView) -> np.ndarray:
        if view.n_cars != self.plan.n_cars or view.n_service != self.plan.n_service:
            raise ValueError("the plan is for another fleet or window")
        return np.maximum(self.plan.powers(view.hour, view.zeta_known), 0.0)
