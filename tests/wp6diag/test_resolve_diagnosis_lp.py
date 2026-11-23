"""The relaxed programmes of the re-solve diagnosis (`resolve_diagnosis_lp`) on hand-built
re-solves whose cause is known by construction.

One service hour unless said (so each re-solve plans one hour, `h_p` per present car, and every
row can be written out). The request is `y`; the guarded plan gives each car the power listed.
Every hand case is also solved by `resolve.resolve` itself and must be infeasible there.

- NEED: one car, rate 2, cap 10, needs 1, `y = 0.5`. Rows: `h = 0.5` (request), `h >= 1` (need),
  `h >= 0.5` (guard). Feasible without the request (`h = 1`) or without the need, and by no other
  single removal. Need slack 0.5 kWh; request miss -0.5 kWh (the car must draw 0.5 more).
- GUARD: two cars, the guarded plan A 1.5 and B -0.5 run one-way, `y = 1`. Rows:
  `h_A + h_B = 1`, `h_A >= 1.5` and `h_B >= 0` (guard; a negative planned power counts 0).
  Feasible without the request or without the guard; the floor does not help (B's guard row asks
  `h_B >= 0` too). No slack on the need helps.
- POWER: car A seen gone; B present with rate 1, `y = 1.5`, guarded plan B 1.0. Rows:
  `h_B = 1.5`, `h_B <= 1`. Feasible without the request or without the power limit.
- DEPARTURE: one car, rate 1, departure bound 1.5 (half of hour 1), `y = 0.8`, guarded plan 0.8.
  Rows: `h = 0.8`, `h <= 0.5`, `h >= 0.8` (guard). Only removing the power limit helps; with the
  car taken to stay the limit is 1 and the rows are feasible.
- CAP: one car that has drawn 0.8 of a cap of 1, `y = 0.5`, guarded plan 0.2. Rows: `h = 0.5`,
  `h <= 0.2` (cap). Feasible without the request or without the cap.
- TRACKING: two service hours, re-solve at hour 1 with `zeta_1 = 0`; box centre (0.5, 0.8),
  half-widths (0, 0.5). Car A is seen gone; B has rate 1. Hour 1 asks 0.5; hour 2 asks
  `0.8 + 0.5 zeta_2`, up to 1.3 against B's rate 1. Feasible without the tracking of hour 2 or
  without the power limit; removing the request of hour 1 does not help. Box scale: `0.8 + 0.5 s
  <= 1`, so `s = 0.4`. No slack on the need and no miss in hour 1 helps.
- PAIR: `y = 5`; car A rate 0.2, cap 0.2, guarded plan 1.0; car B rate 1, cap 10, guarded plan 0.
  A's guard (`h_A >= 1`) breaks both its power limit and its cap, and the request is above what
  the power limits allow, so no single removal helps. The pairs that do: request and guard; guard
  and power (`h_A <= 0.2`, `h_B = 4.8`); power and cap (`h_A >= 1`, `h_B` free).
- FEASIBLE: TRACKING with rate 2: box scale 1, need slack 0, request miss 0, recheck feasible.

Timeouts are injected through the `solve` argument (the solver's time limit is a request, so a
real tiny limit does not time out every small programme; `tests/wp6/test_resolve.py`): a relaxed
programme that times out is recorded as "timeout" and never as infeasible, the pairs are not
solved after a timed-out single removal, and the cause is then "not determined". A real tiny
limit is also run: every answer is then "timeout" or the answer of the 300-second limit.
"""

import pytest
from diag_fake import BUDGET, HandCar, hand

from fleetreserve import resolve_diagnosis_lp as relaxed
from fleetreserve.lp import LpResult
from fleetreserve.resolve import ROW_KINDS, resolve

TOLERANCE = 1e-7
ONE_WAY = dict(two_way=False, budget=BUDGET)
N_SOLVES = 12  # the recheck, 7 single removals, the departure bound, 3 distances; no pair


def _one_hour(cars, y, **kwargs):
    return hand(cars, hour=1, m=[y], w=[0.0], known=[0.0], **kwargs)


def _tracking(rate):
    cars = [
        HandCar(planned=(0, 0, 0.5), present=False),
        HandCar(planned=(0, 0.5, 0.3), rate=rate),
    ]
    gains = {0: [[0, 0], [0, 0], [0, 0.5]]}
    return hand(cars, hour=1, m=[0.5, 0.8], w=[0.0, 0.5], known=[0.0], gains=gains)


CASES = {
    "NEED": _one_hour([HandCar(planned=(0, 0.5), need=1.0)], 0.5),
    "GUARD": _one_hour([HandCar(planned=(0, 1.5)), HandCar(planned=(0, -0.5))], 1.0),
    "POWER": _one_hour(
        [HandCar(planned=(0, 0.5), present=False), HandCar(planned=(0, 1.0), rate=1.0)], 1.5
    ),
    "DEPARTURE": _one_hour([HandCar(planned=(0, 0.8), rate=1.0, departure=1.5)], 0.8),
    "CAP": _one_hour([HandCar(planned=(0, 0.2), cap=1.0, stored=0.8)], 0.5),
    "TRACKING": _tracking(1.0),
    "PAIR": _one_hour(
        [HandCar(planned=(0, 1.0), rate=0.2, cap=0.2), HandCar(planned=(0, 0.0), rate=1.0)], 5.0
    ),
}
SINGLE = {
    "NEED": {"request", "need"},
    "GUARD": {"request", "guard"},
    "POWER": {"request", "power"},
    "DEPARTURE": {"power"},
    "CAP": {"request", "cap"},
    "TRACKING": {"tracking", "power"},
    "PAIR": set(),
}
PAIR_FEASIBLE = {"request+guard", "guard+power", "power+cap"}


def _feasible(answers: dict) -> set:
    return {name for name, answer in answers.items() if answer == relaxed.FEASIBLE}


def test_the_groups_cover_every_requirement_of_the_re_solve_once():
    kinds = [kind for group in relaxed.GROUPS.values() for kind in group]
    assert sorted(kinds) == sorted(set(ROW_KINDS) - {"spread"})
    assert len(relaxed.PAIRS) == 21


@pytest.mark.parametrize("name", list(CASES))
def test_a_hand_case_is_infeasible_in_the_real_re_solve_and_its_cause_is_named(name):
    state, guard = CASES[name]
    assert resolve(state, guard, **ONE_WAY).outcome == "infeasible"
    found = relaxed.diagnose(state, guard, **ONE_WAY)
    assert found.recheck == relaxed.INFEASIBLE
    assert _feasible(found.single) == SINGLE[name]
    others = set(relaxed.GROUPS) - SINGLE[name]
    assert {found.single[g] for g in others} <= {relaxed.INFEASIBLE}
    if SINGLE[name]:
        assert found.pair is None
        assert set(relaxed.cause(found).split("|")) == SINGLE[name]
    else:
        assert _feasible(found.pair) == PAIR_FEASIBLE
        assert set(found.pair.values()) == {relaxed.FEASIBLE, relaxed.INFEASIBLE}
        assert set(relaxed.cause(found).split("|")) == PAIR_FEASIBLE


def test_only_the_relaxed_departure_bound_makes_the_departure_case_feasible():
    answers = {name: relaxed.diagnose(*CASES[name], **ONE_WAY).departure_relaxed for name in CASES}
    assert answers.pop("DEPARTURE") == relaxed.FEASIBLE
    assert set(answers.values()) == {relaxed.INFEASIBLE}


def test_the_distances_equal_the_hand_values():
    need = relaxed.diagnose(*CASES["NEED"], **ONE_WAY)
    assert (need.need_slack_outcome, need.request_miss_outcome) == ("optimal", "optimal")
    assert need.need_slack == pytest.approx(0.5, abs=TOLERANCE)
    assert need.request_miss == pytest.approx(-0.5, abs=TOLERANCE)
    assert (need.box_scale, need.box_scale_outcome) == (None, "infeasible")

    tracking = relaxed.diagnose(*CASES["TRACKING"], **ONE_WAY)
    assert tracking.box_scale_outcome == "optimal"
    assert tracking.box_scale == pytest.approx(0.4, abs=TOLERANCE)
    assert (tracking.need_slack, tracking.need_slack_outcome) == (None, "infeasible")
    assert (tracking.request_miss, tracking.request_miss_outcome) == (None, "infeasible")

    power = relaxed.diagnose(*CASES["POWER"], **ONE_WAY)
    assert power.request_miss == pytest.approx(0.5, abs=TOLERANCE)  # 1.5 asked, 1 deliverable
    guard = relaxed.diagnose(*CASES["GUARD"], **ONE_WAY)
    assert (guard.need_slack, guard.need_slack_outcome) == (None, "infeasible")
    assert guard.request_miss == pytest.approx(-0.5, abs=TOLERANCE)


def test_a_feasible_re_solve_has_scale_one_and_no_slack():
    state, guard = _tracking(2.0)
    assert resolve(state, guard, **ONE_WAY).outcome == "optimal"
    found = relaxed.diagnose(state, guard, **ONE_WAY)
    assert found.recheck == relaxed.FEASIBLE
    assert set(found.single.values()) == {relaxed.FEASIBLE}
    assert found.box_scale == pytest.approx(1.0, abs=TOLERANCE)
    assert found.need_slack == pytest.approx(0.0, abs=TOLERANCE)
    assert found.request_miss == pytest.approx(0.0, abs=TOLERANCE)
    assert relaxed.box_scale(state, guard, **ONE_WAY) == (found.box_scale, "optimal")
    assert relaxed.need_slack(state, guard, **ONE_WAY) == (found.need_slack, "optimal")


def test_an_unknown_group_is_refused():
    with pytest.raises(ValueError, match="unknown requirement groups"):
        relaxed.assemble(*CASES["NEED"], **ONE_WAY, drop=("needs",))


# --- timeouts ------------------------------------------------------------------------------------


def _timing_out(calls_that_time_out=None):
    """A `solve` that answers "timeout" on the calls numbered in `calls_that_time_out` (every call
    when None) and solves the others; `seen` holds the time limit of each call."""
    seen = []

    def solve(programme, columns, values, time_limit):
        seen.append(time_limit)
        if calls_that_time_out is None or len(seen) - 1 in calls_that_time_out:
            return LpResult("timeout", None, None, time_limit, time_limit, 1, "time limit reached")
        return relaxed.solve_programme(programme, columns, values, time_limit)

    return solve, seen


def test_a_relaxed_programme_that_times_out_is_recorded_as_a_timeout():
    solve, seen = _timing_out()
    found = relaxed.diagnose(*CASES["NEED"], **ONE_WAY, time_limit=7.0, solve=solve)
    assert seen == [7.0] * N_SOLVES
    assert found.recheck == relaxed.TIMEOUT
    assert set(found.single.values()) == {relaxed.TIMEOUT}
    assert found.pair is None
    assert found.departure_relaxed == relaxed.TIMEOUT
    assert (found.box_scale, found.box_scale_outcome) == (None, "timeout")
    assert (found.need_slack, found.need_slack_outcome) == (None, "timeout")
    assert (found.request_miss, found.request_miss_outcome) == (None, "timeout")
    assert relaxed.INFEASIBLE not in found.outcomes()
    assert relaxed.cause(found) == "not determined"


def test_one_timed_out_removal_is_not_read_as_infeasible():
    need = 1 + list(relaxed.GROUPS).index("need")  # call 0 is the recheck
    solve, _ = _timing_out({need})
    found = relaxed.diagnose(*CASES["NEED"], **ONE_WAY, solve=solve)
    assert found.single["need"] == relaxed.TIMEOUT
    assert _feasible(found.single) == {"request"}
    assert found.outcomes().count(relaxed.TIMEOUT) == 1


def test_pairs_are_not_solved_after_a_timed_out_single_removal():
    guard = 1 + list(relaxed.GROUPS).index("guard")
    solve, seen = _timing_out({guard})
    found = relaxed.diagnose(*CASES["PAIR"], **ONE_WAY, solve=solve)
    assert found.pair is None and len(seen) == N_SOLVES
    assert relaxed.cause(found) == "not determined"
    whole = relaxed.diagnose(*CASES["PAIR"], **ONE_WAY)
    assert len(whole.pair) == 21


def test_a_timed_out_pair_leaves_the_cause_not_determined_when_no_pair_is_feasible():
    solve, _ = _timing_out(set(range(8, 8 + 21)))  # every pair
    found = relaxed.diagnose(*CASES["PAIR"], **ONE_WAY, solve=solve)
    assert set(found.pair.values()) == {relaxed.TIMEOUT}
    assert relaxed.cause(found) == "not determined"


def test_a_solver_error_is_an_error_not_infeasible():
    def solve(programme, columns, values, time_limit):
        return LpResult("error", None, None, 0.0, time_limit, 4, "numerical difficulties")

    found = relaxed.diagnose(*CASES["NEED"], **ONE_WAY, solve=solve)
    assert set(found.answers()) == {relaxed.ERROR}
    assert found.box_scale_outcome == "error" and found.box_scale is None


def test_a_real_tiny_limit_gives_a_timeout_or_the_answer_of_the_long_limit():
    state, guard = CASES["TRACKING"]
    long = relaxed.diagnose(state, guard, **ONE_WAY)
    tiny = relaxed.diagnose(state, guard, **ONE_WAY, time_limit=1e-9)
    for group, answer in tiny.single.items():
        assert answer in (relaxed.TIMEOUT, long.single[group])
    assert tiny.box_scale_outcome in ("timeout", "optimal")
    if tiny.box_scale_outcome == "optimal":
        assert tiny.box_scale == pytest.approx(long.box_scale, abs=TOLERANCE)
