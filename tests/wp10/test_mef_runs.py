"""`mef_controller.MefController` on the made-up offer days of `compare_fake`, against
least-laxity-first and against linear programmes on the carsets rows.

1. On 12 made-up days, at the scales 0.2, 0.5 and 1, on every audited path and with both the
   forecast and the oracle belief: in every hour that does not deviate the planned powers are
   exactly least-laxity-first's on the same view; in every deviating hour they are
   least-laxity-first's split of another total, and that total leaves the believed future met
   while least-laxity-first's does not; on every path with no deviating hour the whole run
   (planned powers, draws and both losses) equals least-laxity-first's run. Both kinds of path
   occur under both beliefs, and some deviating hours are service hours after the first. The
   counts of every run are the hours recounted from what the run did (every hour; the deviating
   ones; those of them in service hours; those that did not deviate although least-laxity-first's
   split leaves no believed future); under both beliefs some hours deviate in hour 0 and some
   have no feasible total, and under the oracle belief some deviate in hour 1.
2. The believed future of every hour of those runs (with the least-laxity split of the hour's
   target) against a linear programme built from the rows of `carsets` instead of from the
   belief: the forecast rows of the day (`carsets.day_forecast_rows`) under the forecast belief
   and the true rows (`carsets.true_rows`) under the oracle belief, with the powers of the hours
   so far fixed at the run's draws and the split, a car not counted as plugged in held at 0 and
   without its need row, and each remaining hour's fleet total within the box (`lp.solve_lp`,
   30-second limit). `mef_future.meets` equals the programme's feasibility in every hour; both
   answers occur.
3. `compare_day.evaluate` sums each path's deviation counts into the method's `deviation_counts`
   (key by key, over the audited paths), gives `deviation_counts` None to every other method and
   to every method on a day that abstains or at scale 0, and leaves the other methods' losses as
   they are when the method runs beside them.
4. `compare_day.sum_deviations` sums key by key over the keys of the first counts, in their
   order (a key only in a later one is left out), and gives None for no counts.
"""

import dataclasses

import numpy as np
import pytest
from compare_fake import offer_days

from fleetreserve import compare_day, compare_methods
from fleetreserve.carsets import day_forecast_rows, true_rows
from fleetreserve.controller import forecast_belief, hour_zero_target, oracle_belief
from fleetreserve.controller_replay import run_controller
from fleetreserve.lp import BOX_SECONDS, solve_lp
from fleetreserve.mef_controller import MefController
from fleetreserve.mef_future import meets
from fleetreserve.plans import Plan
from fleetreserve.priority_rules import least_laxity_first

SCALES = (0.2, 0.5, 1.0)
BELIEFS = {"forecast": (forecast_belief, day_forecast_rows), "oracle": (oracle_belief, true_rows)}


class Spy(MefController):
    """Records, per hour, the view, least-laxity-first's split, the planned powers and whether
    the hour deviated."""

    def __init__(self, lo, hi):
        super().__init__(lo, hi)
        self.seen = []

    def __call__(self, view):
        before = self.counts["deviating_hours"]
        planned = super().__call__(view)
        plain = least_laxity_first(view)
        self.seen.append((view, plain, planned, self.counts["deviating_hours"] > before))
        return planned


def _runs():
    """(belief name, prepared day, scale, path index, spy, run, least-laxity run)."""
    found = []
    for prepared in offer_days(12):
        policy = prepared.certificate.policy
        for beta in SCALES:
            target = hour_zero_target(Plan.from_policy(policy, beta))
            for name, (make, _) in BELIEFS.items():
                belief = make(prepared.day)
                for k, zeta in enumerate(prepared.paths):
                    requests = policy.request(zeta, beta)
                    spy = Spy(policy.m - beta * policy.w, policy.m + beta * policy.w)
                    found.append(
                        (
                            name,
                            prepared,
                            beta,
                            k,
                            spy,
                            *(
                                run_controller(
                                    prepared.day,
                                    controller,
                                    belief,
                                    zeta,
                                    requests,
                                    hour_zero_target=target,
                                    departure_seen=prepared.departure_seen,
                                )
                                for controller in (spy, least_laxity_first)
                            ),
                        )
                    )
    return found


@pytest.fixture(scope="module")
def runs():
    return _runs()


def test_hours_that_do_not_deviate_plan_least_laxity_first_powers(runs):
    paths = {(name, quiet): 0 for name in BELIEFS for quiet in (True, False)}
    later = 0
    for name, _, _, _, spy, run, plain in runs:
        for view, llf, planned, deviated in spy.seen:
            if not deviated:
                assert np.array_equal(planned, llf)
                continue
            total = float(planned.sum())
            assert abs(total - float(llf.sum())) > 1e-6
            again = least_laxity_first(dataclasses.replace(view, target=total))
            assert np.allclose(planned, again, rtol=0, atol=1e-12)
            assert meets(spy.future(view, planned)) and not meets(spy.future(view, llf))
            later += view.hour >= 2
        quiet = spy.counts["deviating_hours"] == 0
        paths[(name, quiet)] += 1
        if quiet:
            assert np.array_equal(run.planned, plain.planned)
            assert np.array_equal(run.draws, plain.draws)
            assert (run.slot_loss, run.driver_loss) == (plain.slot_loss, plain.driver_loss)
    assert min(paths.values()) >= 10, paths
    assert later >= 5


def test_the_counts_are_the_hours_recounted(runs):
    seen = {name: {"zero": 0, "one": 0, "none": 0} for name in BELIEFS}
    for name, _, _, _, spy, _, _ in runs:
        recount = dict.fromkeys(spy.counts, 0)
        for view, llf, _, deviated in spy.seen:
            recount["hours"] += 1
            recount["deviating_hours"] += deviated
            recount["deviating_service_hours"] += deviated and view.hour >= 1
            none = not deviated and not meets(spy.future(view, llf))
            recount["no_feasible_hours"] += none
            seen[name]["zero"] += deviated and view.hour == 0
            seen[name]["one"] += deviated and view.hour == 1
            seen[name]["none"] += none
        assert spy.deviation_counts() == recount
        assert recount["hours"] == spy.lo.size + 1
    assert all(found["zero"] > 0 and found["none"] > 0 for found in seen.values()), seen
    assert seen["oracle"]["one"] > 0, seen


def _programme(rows, run, hour, split, plugged, lo, hi) -> bool:
    """Feasibility of the rows with hours 0 to `hour` fixed (module text, item 2)."""
    cars, hours = rows.n_cars, rows.n_service + 1
    fixed = np.array(run.draws[:, :hours], dtype=float)
    fixed[:, hour] = split
    bounds = []
    for i in range(cars):
        for t in range(hours):
            if t <= hour:
                bounds.append((fixed[i, t], fixed[i, t]))
            elif plugged[i]:
                bounds.append((0.0, None))
            else:
                bounds.append((0.0, 0.0))
    keep = np.array([plugged[c] or k != "need" for c, k in zip(rows.car, rows.kinds, strict=True)])
    a = np.zeros((int(keep.sum()), cars * hours))
    for r, k in enumerate(np.flatnonzero(keep)):
        c = rows.car[k]
        a[r, c * hours : (c + 1) * hours] = rows.coefficients[k]
    b = rows.bounds[keep]
    fleet = []
    rhs = []
    for t in range(hour + 1, hours):
        row = np.zeros(cars * hours)
        row[[i * hours + t for i in range(cars)]] = 1.0
        fleet += [row, -row]
        rhs += [hi[t - 1], -lo[t - 1]]
    if fleet:
        a = np.vstack([a, np.array(fleet)])
        b = np.concatenate([b, rhs])
    found = solve_lp(np.zeros(cars * hours), A_ub=a, b_ub=b, bounds=bounds, time_limit=BOX_SECONDS)
    assert found.outcome in ("optimal", "infeasible"), found.outcome
    return found.outcome == "optimal"


def test_the_believed_future_equals_a_programme_on_the_rows(runs):
    answers = {True: 0, False: 0}
    for name, prepared, _, _, spy, run, _ in runs[::3]:
        rows = BELIEFS[name][1](prepared.day)
        for view, llf, _, _ in spy.seen:
            # the hours before this one are fixed at the run's own draws, whose sum is the view's
            closed = meets(spy.future(view, llf))
            assert closed == _programme(rows, run, view.hour, llf, view.plugged, spy.lo, spy.hi)
            answers[closed] += 1
    assert min(answers.values()) >= 20, answers


def test_compare_day_sums_the_deviation_counts_over_the_paths():
    names = ["llf", "diagonal-split", "max-entropy-feedback"]
    methods = compare_methods.select(names)
    alone = compare_methods.select(["llf", "diagonal-split"])
    deviating = 0
    for prepared in offer_days(6):
        policy = prepared.certificate.policy
        for beta in SCALES:
            found = compare_day.evaluate(prepared, beta, methods)
            assert found["llf"].deviation_counts is None
            assert found["diagonal-split"].deviation_counts is None
            counts = found["max-entropy-feedback"].deviation_counts
            target = hour_zero_target(Plan.from_policy(policy, beta))
            belief = forecast_belief(prepared.day)
            summed = dict.fromkeys(counts, 0)
            for zeta in prepared.paths:
                spy = Spy(policy.m - beta * policy.w, policy.m + beta * policy.w)
                run_controller(
                    prepared.day,
                    spy,
                    belief,
                    zeta,
                    policy.request(zeta, beta),
                    hour_zero_target=target,
                    departure_seen=prepared.departure_seen,
                )
                for key, number in spy.deviation_counts().items():
                    summed[key] += number
            assert counts == summed
            assert counts["hours"] == len(prepared.paths) * (prepared.day.n_service + 1)
            deviating += counts["deviating_hours"]
            others = compare_day.evaluate(prepared, beta, alone)
            for name in ("llf", "diagonal-split"):
                assert found[name] == others[name]
        nothing = compare_day.evaluate(prepared, 0.0, methods)
        assert all(day.deviation_counts is None for day in nothing.values())
        abstains = dataclasses.replace(prepared, abstain_reason="made up")
        assert compare_day.evaluate(abstains, 0.5, methods)["max-entropy-feedback"] == (
            compare_day.MethodDay(0.0, 0.0, ())
        )
    assert deviating > 0


def test_sum_deviations_sums_key_by_key_over_the_keys_of_the_first():
    first, second = {"b": 1, "a": 2}, {"a": 30, "b": 40, "c": 500}
    summed = compare_day.sum_deviations([first, second])
    assert summed == {"b": 41, "a": 32} and list(summed) == ["b", "a"]
    assert compare_day.sum_deviations([first]) == first
    assert compare_day.sum_deviations([]) is None
