"""Solver errors in the relaxed programmes, and the retry (`resolve_diagnosis_lp`, "Solver errors").

On made-up runs the interior-point method ended in HiGHS status 4 on some relaxed programmes. The
diagnosis programmes (never the re-solve's own solve) are tried once more with the dual simplex
method after an "error"; a timeout is never retried and never read as infeasible. The errors are
injected through the `first` and `second` attempts of `relaxed.retrying`, because the real solver
does not fail on demand.

1. An error is retried once, with the same arguments and time limit; the second result replaces
   it and records the first outcome. An optimal, infeasible or timed-out first attempt is returned
   as it is and the second is not called. A second error stays an error; a second timeout stays a
   timeout.
2. `diagnose` counts the programmes whose first attempt ended in an error, and of those the
   retries that ended optimal or infeasible; the answers of the retried programmes are those of
   the real simplex solve.
3. A timeout injected as the first attempt is never turned into infeasible by a retry: the second
   attempt is not called, and the cause stays "not determined".
4. The two real methods agree on the box scale of a hand case; `maximise` with the interior-point
   method is `SparseProgramme.maximise`.
5. The re-solve's own solve is not retried: one solver call, the interior-point method, and an
   error stays the re-solve's outcome.
"""

import pytest
from diag_fake import BUDGET, HandCar, hand

from fleetreserve import resolve_diagnosis_lp as relaxed
from fleetreserve import sparse_programme
from fleetreserve.lp import LpResult
from fleetreserve.resolve import resolve

ONE_WAY = dict(two_way=False, budget=BUDGET)
N_SOLVES = 12  # the recheck, 7 single removals, the departure bound, 3 distances; no pair


def _tracking():
    """Hour 1 asks 0.5; hour 2 asks 0.8 + 0.5 zeta_2 against a rate of 1: infeasible, scale 0.4."""
    cars = [
        HandCar(planned=(0, 0, 0.5), present=False),
        HandCar(planned=(0, 0.5, 0.3), rate=1.0),
    ]
    return hand(cars, hour=1, m=[0.5, 0.8], w=[0.0, 0.5], known=[0.0])


def _answer(outcome, limit=5.0):
    return LpResult(outcome, None, None, 0.1, limit, 4 if outcome == "error" else 0, "made up")


def _simplex(programme, columns, values, time_limit):
    return relaxed.maximise(programme, columns, values, time_limit, "simplex")


def _small():
    programme, *_ = relaxed.assemble(*_tracking(), **ONE_WAY, drop=("power",))  # feasible
    return programme


def test_an_error_is_retried_once_with_the_same_arguments():
    calls = []

    def second(programme, columns, values, time_limit):
        calls.append((programme, list(columns), list(values), time_limit))
        return relaxed.maximise(programme, columns, values, time_limit, "simplex")

    solve = relaxed.retrying(lambda *args: _answer("error"), second)
    programme = _small()
    result = solve(programme, [0], [0.0], 7.0)
    assert [(c[1], c[2], c[3]) for c in calls] == [([0], [0.0], 7.0)] and calls[0][0] is programme
    assert isinstance(result, relaxed.Retried) and result.first_outcome == "error"
    assert result.outcome == "optimal" and result.time_limit == 7.0


@pytest.mark.parametrize("outcome", ["optimal", "infeasible", "timeout", "unbounded"])
def test_only_an_error_is_retried(outcome):
    calls = []

    def second(*args):
        calls.append(args)
        return _answer("optimal")

    first = _answer(outcome)
    result = relaxed.retrying(lambda *args: first, second)(_small(), [0], [0.0], 5.0)
    assert result is first and calls == []


def test_a_second_error_stays_an_error_and_a_second_timeout_stays_a_timeout():
    for second, expected in ((_answer("error"), "error"), (_answer("timeout"), "timeout")):
        solve = relaxed.retrying(
            lambda *args: _answer("error"), lambda *args, second=second: second
        )
        result = solve(_small(), [0], [0.0], 5.0)
        assert (result.outcome, result.first_outcome) == (expected, "error")


def _erroring(error_calls, second_errors=()):
    """A `solve` whose first attempt ends in an error on the calls numbered in `error_calls` and
    otherwise is the real interior-point solve; its retry is the real simplex solve except on the
    retries numbered in `second_errors`."""
    first_calls, second_calls = [], []

    def first(programme, columns, values, time_limit):
        first_calls.append(time_limit)
        if len(first_calls) - 1 in error_calls:
            return _answer("error", time_limit)
        return relaxed.maximise(programme, columns, values, time_limit, "ipm")

    def second(programme, columns, values, time_limit):
        second_calls.append(time_limit)
        if len(second_calls) - 1 in second_errors:
            return _answer("error", time_limit)
        return _simplex(programme, columns, values, time_limit)

    return relaxed.retrying(first, second), first_calls, second_calls


def test_diagnose_counts_the_first_errors_and_the_retries_that_answered():
    whole = relaxed.diagnose(*_tracking(), **ONE_WAY)
    assert (whole.first_errors, whole.retried_answered) == (0, 0)
    solve, first, second = _erroring({0, 3, 5, 11}, second_errors={1})
    found = relaxed.diagnose(*_tracking(), **ONE_WAY, time_limit=9.0, solve=solve)
    assert len(first) == N_SOLVES and second == [9.0] * 4
    assert (found.first_errors, found.retried_answered) == (4, 3)
    assert found.outcomes().count(relaxed.ERROR) == 1  # the retry that ended in an error
    # The retried programmes give the answers of the real solves; the programme that stays an
    # error (call 3) is the only difference.
    assert found.recheck == whole.recheck and found.single["tracking"] == whole.single["tracking"]
    assert found.request_miss_outcome == whole.request_miss_outcome  # call 11
    assert found.request_miss == whole.request_miss
    assert found.single["guard"] == relaxed.ERROR  # call 3 is the removal of "guard"
    assert found.box_scale == pytest.approx(whole.box_scale, abs=1e-7)


def test_every_first_attempt_in_error_is_answered_by_the_simplex_method():
    whole = relaxed.diagnose(*_tracking(), **ONE_WAY)
    solve, first, second = _erroring(set(range(N_SOLVES)))
    found = relaxed.diagnose(*_tracking(), **ONE_WAY, solve=solve)
    assert (found.first_errors, found.retried_answered) == (N_SOLVES, N_SOLVES)
    assert found.answers() == whole.answers() and relaxed.cause(found) == relaxed.cause(whole)
    assert found.box_scale == pytest.approx(0.4, abs=1e-7)
    assert found.need_slack_outcome == whole.need_slack_outcome == "infeasible"


def test_a_timeout_is_not_retried_into_an_infeasible():
    seconds = []

    def timing_out(programme, columns, values, time_limit):
        return _answer("timeout", time_limit)

    def second(programme, columns, values, time_limit):
        seconds.append(time_limit)
        return _simplex(programme, columns, values, time_limit)

    found = relaxed.diagnose(
        *_tracking(), **ONE_WAY, solve=relaxed.retrying(timing_out, second), time_limit=3.0
    )
    assert seconds == []
    assert set(found.answers()) == {relaxed.TIMEOUT}
    assert relaxed.INFEASIBLE not in found.outcomes()
    assert (found.first_errors, found.retried_answered) == (0, 0)
    assert relaxed.cause(found) == "not determined"


def test_the_default_solve_retries_with_the_simplex_method_and_keeps_its_time_limit(monkeypatch):
    methods = []
    real = relaxed.solve_lp

    def failing_first(*args, method, **kwargs):
        methods.append((method, kwargs["time_limit"]))
        if method == "ipm":
            return _answer("error", kwargs["time_limit"])
        return real(*args, method=method, **kwargs)

    monkeypatch.setattr(relaxed, "solve_lp", failing_first)
    result = relaxed.solve_programme(_small(), [0], [0.0], 11.0)
    assert methods == [("ipm", 11.0), ("simplex", 11.0)]
    assert result.outcome == "optimal" and result.first_outcome == "error"


def test_the_two_real_methods_agree_and_the_interior_point_method_is_the_programmes_own():
    state, guard = _tracking()
    programme, s, *_ = relaxed.assemble(state, guard, **ONE_WAY, scale=True)
    own, _ = programme.maximise([s], [1.0], 30.0)
    ipm = relaxed.maximise(programme, [s], [1.0], 30.0, "ipm")
    simplex = relaxed.maximise(programme, [s], [1.0], 30.0, "simplex")
    assert ipm.objective == own.objective
    assert simplex.objective == pytest.approx(ipm.objective, abs=1e-7)
    assert simplex.x[s] == pytest.approx(0.4, abs=1e-7)


def test_the_re_solves_own_solve_is_not_retried(monkeypatch):
    calls = []

    def failing(*args, method, **kwargs):
        calls.append(method)
        return _answer("error", kwargs["time_limit"])

    monkeypatch.setattr(sparse_programme, "solve_lp", failing)
    assert resolve(*_tracking(), **ONE_WAY).outcome == "error"
    assert calls == ["ipm"]
