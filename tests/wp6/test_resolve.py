"""Plan section 5 ("Rungs"): the re-solve programme, checked independently of its construction.

1. Every optimal re-solve met on `N_DAYS` seeded made-up days (`made_up_offer_days`, every
   forecast kind), at scales 0.5 and 1.5, on each day's audited paths, is checked on every corner
   of the future requests against the rules written out again here: the present cars share each
   remaining request exactly; cars that have left plan nothing; each present car's power, stored
   energy and needed energy stay within its updated forecast set (survival-tightened departure,
   energy drawn subtracted, cap at least the energy drawn, power at most the known charge bound,
   need lowered to what the set can still reach; two-way: power down to `-r_hat f`, stored energy
   at least `-B`); and the guard holds: the car's planned stored energy is at least the
   predecessor's planned stored energy from the guard's hour, a negative planned power of a one-way
   plan counting 0. The sample gives `N_RESOLVES` re-solves, `N_OPTIMAL` of them optimal (each
   checked) and the rest infeasible.
2. A real infeasible re-solve: on the gone day of `test_rungs_hand.py`, a departure
   rule that says car B leaves at the end of hour 1 leaves nobody to take hour 2's request; the
   re-solve is "infeasible" and the fixed plan stays.
3. A real timeout: a made-up day of 30 cars and 6 service hours, all eight audited paths, with
   the real solver and a time limit of `TINY_LIMIT` seconds. The limit is a request, not a hard
   stop (`lp.py`): a small programme (the re-solves of the last hours, 15 to 141 variables) can
   finish before the solver first looks at the clock, and how many do varies from one process to
   the next; the answer then stays what the solver said, marked as late. So the test does not
   ask every re-solve to time out. It asks, for every re-solve: a "timeout" has no plan; any
   other answer is late and equals the answer of the same re-solve under the 300-second limit (a
   timeout is never recorded as infeasible or as an error). For every rung on every path, up to
   its first optimal re-solve: the plan in force at each hour is the predecessor's plan itself
   and the hourly draws equal the predecessor's; and a rung whose re-solves all ended "timeout",
   under predecessors that did the same, draws exactly what the fixed plan draws. It also asks
   that real timeouts were seen: at least `MIN_TIMEOUTS` in the checked stretch (a rung's first
   re-solve has 987 variables and ran past the limit in every run seen), so it cannot pass with no
   timeout in it. How many of the 16 rungs timed out throughout is printed, not asserted: it was
   seen anywhere from 0 to 16, so the strict case is made certain by the injected tests below.
   Two more tests inject a timeout into otherwise real re-solves, which a real timeout cannot do
   at chosen hours. At one chosen hour: the rung follows its predecessor's plan in that hour (not
   the plan of its own earlier optimal re-solve) and takes its own optimal plans at the other
   hours. At every hour: on every path both rungs record only "timeout", follow the fixed plan at
   every hour and draw exactly what it draws (the strict case, here certain).
4. Size and time: on that day, with the 300-second limit, every re-solve on the audited paths at
   scale 1 takes under `SECONDS` of solver time (plan section 13 asks only for a time limit; this
   guards against a programme that grows out of proportion).
"""

import functools
import itertools
import random

import numpy as np
import pytest
from made_up_offer_days import departure_rule, large_offer_day, offer_day
from test_rungs_hand import GONE, GONE_POLICY

from fleetreserve import resolve as resolve_module
from fleetreserve.audit_paths import audited_paths
from fleetreserve.resolve import ResolveResult
from fleetreserve.rungs import run_rungs
from fleetreserve.sparse_programme import Sizes

SEEN = "hour-start"  # when a rung learns of a departure (`rungs.py`); the programme is the same
SEED = 31
N_DAYS = 30
BETAS = (0.5, 1.5)
N_RESOLVES = 1076
N_OPTIMAL = 181
TOLERANCE = 1e-6
TINY_LIMIT = 1e-6
LONG_LIMIT = 300.0
MIN_TIMEOUTS = 12  # 8 paths, each with 4 re-solves, the first of 987 variables; 48 to 64 seen
SECONDS = 1.0


def _recording(log, time_limit=None):
    def resolver(state, guard, *, two_way, budget, time_limit=time_limit):
        result = resolve_module.resolve(
            state, guard, two_way=two_way, budget=budget, time_limit=time_limit
        )
        log.append((state, guard, two_way, budget, result))
        return result

    return resolver


@functools.cache
def _sample():
    rng = random.Random(SEED)
    log = []
    for k in range(N_DAYS):
        day, certificate, _ = offer_day(rng, k)
        if certificate.outcome != "optimal":
            continue
        for beta in BETAS:
            for path in audited_paths(day.unit, day.date, day.n_service):
                run_rungs(
                    day,
                    certificate.policy,
                    beta,
                    path,
                    departure_bound=departure_rule(day, SEEN),
                    departure_seen=SEEN,
                    resolver=_recording(log, 300.0),
                )
    return log


def _check(state, guard, two_way, budget, plan):
    t, n = state.hour, state.n_service
    hours = np.arange(t, n + 1)
    recovery = np.arange(n + 1, n + 1 + state.n_recovery)
    for corner in itertools.product((-1.0, 1.0), repeat=n - t):
        zeta = np.concatenate([state.known, corner])
        q = np.array([plan.powers(s, zeta) for s in hours]).T  # (cars, hours t..n)
        requests = state.m[t - 1 :] + state.beta * state.w[t - 1 :] * zeta[t - 1 :]
        np.testing.assert_allclose(q.sum(axis=0), requests, atol=TOLERANCE)
        assert np.all(q[~state.present] == 0)
        reference = np.array([guard.plan.powers(s, zeta) for s in range(guard.hour, n + 1)]).T
        if guard.clip:
            reference = np.maximum(reference, 0.0)
        for i in np.flatnonzero(state.present):
            f = state.forecasts[i]
            x = state.stored[i]
            share = np.clip(state.departure[i] - hours, 0, 1)
            high = np.minimum(f.rate * share, state.charge_bound[i])
            low = -f.rate * share if two_way else 0.0
            assert np.all(q[i] <= high + TOLERANCE) and np.all(q[i] >= low - TOLERANCE)
            level = x + np.cumsum(q[i])
            cap = max(f.cap, x)
            assert np.all(level <= cap + TOLERANCE)
            if two_way:
                assert np.all(level >= -budget - TOLERANCE)
            supply = f.rate * np.clip(state.departure[i] - recovery, 0, 1).sum()
            need = min(f.need, min(cap, x + high.sum()) + supply)
            assert level[-1] + supply >= need - TOLERANCE
            planned = guard.stored[i] + np.cumsum(reference[i])[t - guard.hour :]
            assert np.all(level >= planned - TOLERANCE), (level, planned)


def test_every_optimal_re_solve_meets_its_rows_on_every_future_corner():
    log = _sample()
    checked = 0
    for state, guard, two_way, budget, result in log:
        if result.outcome == "optimal":
            _check(state, guard, two_way, budget, result.plan)
            checked += 1
        else:
            assert result.plan is None
    outcomes = sorted({r.outcome for *_, r in log})
    print(f"{checked} optimal re-solves checked of {len(log)}; outcomes {outcomes}")
    assert (len(log), checked) == (N_RESOLVES, N_OPTIMAL)
    assert outcomes == ["infeasible", "optimal"]


def test_a_real_infeasible_re_solve_keeps_the_fixed_plan():
    def gone(car, hour):
        return 2.0 if car == 1 else 3.0

    runs = run_rungs(GONE, GONE_POLICY, 1.0, (1.0, 1.0), departure_bound=gone, departure_seen=SEEN)
    one = runs["one-way"]
    assert [(r.hour, r.outcome) for r in one.resolves] == [(2, "infeasible")]
    np.testing.assert_array_equal(one.run.draws, runs["fixed"].run.draws)
    assert one.run.first_failure == 2


@functools.cache
def _large():
    return large_offer_day(random.Random(5))


def _checked_against_the_long_limit(log):
    """The real re-solve; any answer that is not a timeout is also solved under the long limit."""

    def resolver(state, guard, *, two_way, budget, time_limit):
        result = resolve_module.resolve(
            state, guard, two_way=two_way, budget=budget, time_limit=time_limit
        )
        truth = None
        if result.outcome != "timeout":
            truth = resolve_module.resolve(
                state, guard, two_way=two_way, budget=budget, time_limit=LONG_LIMIT
            )
        log.append((result, truth))
        return result

    return resolver


def _before_the_first_optimal(rung):
    """The hours a rung spent on its predecessor's plan: up to its first optimal re-solve."""
    return min((r.hour for r in rung.resolves if r.outcome == "optimal"), default=None)


def test_a_real_timeout_is_recorded_as_a_timeout_and_keeps_the_predecessors_plan():
    day, certificate = _large()
    assert certificate.outcome == "optimal"
    log = []
    timeouts = strict = rungs = 0
    paths = audited_paths(day.unit, day.date, day.n_service)
    for path in paths:
        runs = run_rungs(
            day,
            certificate.policy,
            1.0,
            path,
            departure_bound=departure_rule(day, SEEN),
            departure_seen=SEEN,
            time_limit=TINY_LIMIT,
            resolver=_checked_against_the_long_limit(log),
        )
        fixed_draws = runs["fixed"].run.draws
        on_the_fixed_plan = True  # every rung so far has only timed out
        for previous, name in zip(("fixed", "one-way"), ("one-way", "two-way"), strict=True):
            rung, predecessor = runs[name], runs[previous]
            stop = _before_the_first_optimal(rung)
            hours = slice(0, len(rung.plans) if stop is None else stop)
            for plan, other in zip(rung.plans[hours], predecessor.plans[hours], strict=True):
                assert plan is other
            np.testing.assert_array_equal(rung.run.draws[:, hours], predecessor.run.draws[:, hours])
            timeouts += sum(
                r.outcome == "timeout" for r in rung.resolves if stop is None or r.hour <= stop
            )
            on_the_fixed_plan &= all(r.outcome == "timeout" for r in rung.resolves)
            rungs += 1
            if on_the_fixed_plan:
                np.testing.assert_array_equal(rung.run.draws, fixed_draws)
                strict += bool(rung.resolves)
    assert log
    for result, truth in log:
        if truth is None:
            assert result.outcome == "timeout" and result.plan is None
        else:
            assert result.outcome == truth.outcome  # a late answer stays what the solver said
            assert result.late_seconds > 0
    print(f"{timeouts} timeouts checked, {strict} of {rungs} rungs timed out throughout")
    assert timeouts >= MIN_TIMEOUTS
    assert rungs == 2 * len(paths)


def _timeout_at(hours, log):
    def resolver(state, guard, *, two_way, budget, time_limit):
        if state.hour in hours:
            result = ResolveResult("timeout", None, 0.0, 0.0, Sizes(0, 0, 0), "injected")
        else:
            result = resolve_module.resolve(
                state, guard, two_way=two_way, budget=budget, time_limit=LONG_LIMIT
            )
        log.append(result)
        return result

    return resolver


@pytest.mark.parametrize("hour", [3, 4, 5, 6])
def test_a_timeout_at_one_hour_puts_the_rung_back_on_its_predecessors_plan_for_that_hour(hour):
    day, certificate = _large()
    path = audited_paths(day.unit, day.date, day.n_service)[0]  # every re-solve optimal here
    kwargs = dict(departure_bound=departure_rule(day, SEEN), departure_seen=SEEN, last="one-way")
    real = run_rungs(
        day, certificate.policy, 1.0, path, resolver=_recording([], LONG_LIMIT), **kwargs
    )
    assert [(r.hour, r.outcome) for r in real["one-way"].resolves] == [
        (3, "optimal"),
        (4, "optimal"),
        (5, "optimal"),
        (6, "optimal"),
    ]
    log = []
    runs = run_rungs(
        day, certificate.policy, 1.0, path, resolver=_timeout_at({hour}, log), **kwargs
    )
    one, fixed = runs["one-way"], runs["fixed"]
    assert [(r.hour, r.outcome) for r in one.resolves] == [
        (t, "timeout" if t == hour else "optimal") for t in (3, 4, 5, 6)
    ]
    assert one.plans[hour] is fixed.plans[hour]  # not the plan of its earlier optimal re-solve
    for t in range(3, 7):
        assert (one.plans[t] is fixed.plans[t]) == (t == hour)
    # Until the timeout the two runs are the same run.
    np.testing.assert_array_equal(one.run.draws[:, :hour], real["one-way"].run.draws[:, :hour])
    assert one.own_from == (4 if hour == 3 else 3)
    assert [r.outcome for r in log].count("timeout") == 1


def test_a_timeout_at_every_hour_keeps_the_fixed_plan_throughout():
    day, certificate = _large()
    seen = []
    paths = audited_paths(day.unit, day.date, day.n_service)
    for path in paths:
        runs = run_rungs(
            day,
            certificate.policy,
            1.0,
            path,
            departure_bound=departure_rule(day, SEEN),
            departure_seen=SEEN,
            resolver=_timeout_at(set(range(1, day.n_service + 1)), seen),
        )
        fixed = runs["fixed"]
        for name in ("one-way", "two-way"):
            assert {r.outcome for r in runs[name].resolves} <= {"timeout"}
            assert runs[name].own_from is None
            assert all(a is b for a, b in zip(runs[name].plans, fixed.plans, strict=True))
            np.testing.assert_array_equal(runs[name].run.draws, fixed.run.draws)
    assert len(seen) >= 4 * len(paths)  # the one-way rung re-solves in hours 3 to 6 on each path


def test_re_solves_on_30_cars_and_6_hours_take_under_a_second():
    day, certificate = _large()
    log = []
    for path in audited_paths(day.unit, day.date, day.n_service):
        run_rungs(
            day,
            certificate.policy,
            1.0,
            path,
            departure_bound=departure_rule(day, SEEN),
            departure_seen=SEEN,
            last="one-way",
            resolver=_recording(log, 300.0),
        )
    seconds = [r.seconds for *_, r in log]
    optimal = sum(r.outcome == "optimal" for *_, r in log)
    print(f"{len(log)} re-solves ({optimal} optimal), median {np.median(seconds):.4f} s")
    assert optimal >= 8
    assert max(seconds) < SECONDS
