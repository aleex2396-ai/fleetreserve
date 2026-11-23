"""The fall-back and the solver counts of the linear-programme form of ASA (`asa_controller`).

1. A solver stub that ends one stage of one hour in each outcome but "optimal" (each stage in
   turn): the controller plans that hour by the least-laxity fill of the same view, solves no later
   stage of that hour, counts the outcome, the hour and the fall-back hour, and plans the other
   hours by the programme. In that hour's view the least-laxity fill, EDF and the programme give
   three different plans: B leaves after A but has the smaller laxity (0.1 against 0.2), so EDF
   fills A first, (1, 0.1, 0), the least-laxity fill B first, (0.1, 1, 0), and the programme
   gives A between 0.2 and 0.6 (goal 2's optimum, 0.4 of unmet need, is reached by every such
   plan).
2. Every solve is given the time limit the controller was built with (each call's argument is
   recorded); the method `asa` of the registry builds it with `lp.OTHER_SECONDS`.
3. The counts reach `compare_day.evaluate` for `asa` only (None for `llf` and `fixed-plan`), summed
   over the audited paths, and `compare_run.measure_variant` carries them, with a stubbed timeout,
   under `controller_solver_outcomes` for `asa` only.
"""

import numpy as np
import pytest
from compare_fake import offer_days

from fleetreserve import compare_day, compare_methods, compare_run, lp
from fleetreserve.asa_controller import OUTCOMES, AsaController
from fleetreserve.asa_programme import STAGES
from fleetreserve.controller import Belief, HourView
from fleetreserve.lp import LpResult
from fleetreserve.priority_rules import earliest_deadline_first, least_laxity_first

NOT_OPTIMAL = [o for o in OUTCOMES if o != "optimal"]
REAL = lp.solve_lp


def _views():
    belief = Belief([1.5, 3, 3], [2, 1, 1], [10, 10, 10], [0.6, 1.9, 0])
    return [
        HourView(t, 2, 1.1, np.zeros(t), np.full(t, 1.1), np.zeros(3), np.ones(3, bool), belief)
        for t in range(3)
    ]


class Stub:
    """Ends stage `stage` of the `hour`-th programme in `outcome`; solves the rest for real.
    Records the time limit of every call."""

    def __init__(self, hour, stage, outcome):
        self.hour, self.stage, self.outcome = hour, stage, outcome
        self.calls, self.limits = 0, []

    def __call__(self, c, **arguments):
        self.limits.append(arguments["time_limit"])
        hour, stage = divmod(self.calls, len(STAGES))
        self.calls += 1
        if (hour, stage) == (self.hour, self.stage):
            limit = arguments["time_limit"]
            return LpResult(self.outcome, None, None, 0.0, limit, 99, "made-up outcome")
        return REAL(c, **arguments)


@pytest.mark.parametrize("stage", range(len(STAGES)))
@pytest.mark.parametrize("outcome", NOT_OPTIMAL)
def test_a_stage_that_is_not_optimal_falls_back_to_the_least_laxity_fill(outcome, stage):
    stub = Stub(1, stage, outcome)
    controller = AsaController([0.0, 0.0], n_cars=3, n_recovery=0, time_limit=7.5, solve=stub)
    reference = AsaController([0.0, 0.0], n_cars=3, n_recovery=0, time_limit=7.5)
    views = _views()
    for t, view in enumerate(views):
        power = controller(view)
        if t == 1:
            assert np.array_equal(power, least_laxity_first(view))
            np.testing.assert_allclose(power, [0.1, 1.0, 0.0], atol=1e-12)
            assert not np.allclose(power, earliest_deadline_first(view))
            assert not np.allclose(power, reference(view))
        else:
            assert np.array_equal(power, reference(view))
    counts = controller.solver_counts()
    assert counts["hours"] == 3 and counts["fallback_hours"] == 1
    assert counts["solves"][outcome] == 1
    assert counts["solves"]["optimal"] == 2 * len(STAGES) + stage
    assert sum(counts["solves"].values()) == stub.calls == 2 * len(STAGES) + stage + 1
    assert set(counts["solves"]) == set(OUTCOMES)


def test_every_solve_gets_the_controllers_time_limit():
    stub = Stub(-1, -1, "timeout")
    controller = AsaController([0.0, 0.0], n_cars=3, n_recovery=0, time_limit=7.5, solve=stub)
    for view in _views():
        controller(view)
    assert stub.limits == [7.5] * (3 * len(STAGES))
    assert controller.solver_counts()["solves"]["optimal"] == 3 * len(STAGES)


def test_the_registry_builds_it_with_the_limit_for_other_programmes(monkeypatch):
    limits = []

    def recording(c, **arguments):
        limits.append(arguments["time_limit"])
        return REAL(c, **arguments)

    monkeypatch.setattr(lp, "solve_lp", recording)
    (prepared,) = offer_days(1)
    compare_day.evaluate(prepared, 0.5, compare_methods.select(["asa"]))
    assert limits and set(limits) == {lp.OTHER_SECONDS} and lp.OTHER_SECONDS == 300.0


def test_compare_day_carries_the_counts_for_asa_only():
    (prepared,) = offer_days(1)
    methods = compare_methods.select(["llf", "asa", "fixed-plan"])
    found = compare_day.evaluate(prepared, 1.0, methods)
    assert found["llf"].solver_counts is None and found["fixed-plan"].solver_counts is None
    counts = found["asa"].solver_counts
    hours = len(prepared.paths) * (prepared.day.n_service + 1)
    assert counts["hours"] == hours and counts["fallback_hours"] == 0
    assert counts["solves"] == {**dict.fromkeys(OUTCOMES, 0), "optimal": hours * len(STAGES)}


class TimeoutOnCentre:
    """Ends every hour's stage 3 ("centre") in a timeout; stages run in order and an hour's
    programme stops at its first outcome that is not optimal, so the stage is the call's place."""

    def __init__(self):
        self.stage = 0

    def __call__(self, c, **arguments):
        if STAGES[self.stage] == "centre":
            self.stage = 0
            limit = arguments["time_limit"]
            return LpResult("timeout", None, None, limit, limit, 1, "made-up timeout")
        self.stage = (self.stage + 1) % len(STAGES)
        return REAL(c, **arguments)


def test_measure_variant_carries_a_timeout_and_the_fallback_hours(monkeypatch):
    monkeypatch.setattr(lp, "solve_lp", TimeoutOnCentre())
    prepared = offer_days(2)
    losses, counts = compare_run.measure_variant(
        prepared,
        [None] * len(prepared),
        compare_methods.select(["asa", "llf"]),
        None,
        lambda jobs: [compare_day.evaluate(*job) for job in jobs],
    )
    assert set(losses) == {"asa", "llf"}
    assert set(counts["controller_solver_outcomes"]) == {"asa"}
    found = counts["controller_solver_outcomes"]["asa"]
    hours = sum(
        len(p.paths) * (p.day.n_service + 1) * len(compare_run.dial_splits.GRID) for p in prepared
    )
    assert found["hours"] == found["fallback_hours"] == hours
    assert found["solves"] == {
        **dict.fromkeys(OUTCOMES, 0),
        "optimal": 2 * hours,
        "timeout": hours,
    }
    # every hour planned by the least-laxity fill: the losses are llf's
    assert (losses["asa"].slot == losses["llf"].slot).all()
    assert (losses["asa"].driver == losses["llf"].driver).all()
