"""The corner search of one day on hand-made days whose breaking corners are known
(`corner_fake`: `SINGLE`, `HIDDEN`, `SHORT`), and the closed-form check on them.

1. Corner numbering: corner k has zeta_j = +1 where bit j of k is set, as
   `products.corner_paths` enumerates them; a path inside the box is refused with its message.
2. `SINGLE` at scale 1: only (+1, +1) breaks (slot loss 1/2, first failure in hour 2), and the
   audited paths, which hold (+1, +1), see it.
3. `HIDDEN` at scale 1: only (-1, +1) breaks; the audited paths of that day miss it. At scale 0.5
   (A's rate exactly met) no corner breaks: the scale is applied to the box.
4. `SHORT` at scale 0.5: no hour fails, three corners lose driver energy (2/3, 1/3, 1/3): the day
   breaks by the driver loss alone.
5. A scale of 0 is "no offer" at that scale with no path run; an abstaining day is refused by the
   search and by the closed-form check, its reason in the message; a method of an unknown kind is
   refused.
6. A re-solve that times out on a path makes that path missing: with every re-solve timed out the
   day is "timed out", never "holds" or "breaks"; `day_status` by hand.
7. The closed-form check on `SINGLE` and `SHORT` at scales where the day fails and where it does
   not: the first failing hour and slot loss of the closed form equal the corner worst case; the
   driver-loss bound of `gate_calibrate` (capped at 1) equals the corner maximum on `SHORT`. A
   closed form made to say less than the corners (its slot loss lowered by 0.25) differs by
   0.25 and does not agree ("slot losses differ"): the difference is its size, either way.
8. The decision-hour cap: one car, one service hour (`CAP`), its hour-0 plan 1.5 above its true
   rate 1. It draws 1 in hour 0, and the closed form counts that draw: its rows of hour 1 give
   (2 - 1.5) / 0.5 = 1 (stored energy `1.5 + 0.5 beta zeta_1` against a room of 2), (1 - 0.5) /
   0.5 = 1 (rate) and 0.5 / 0.5 = 1 (power at least 0), so s_1 = 1: no failing hour at scale 1
   and hour 1 above it, as on the corners (at 1.2, zeta_1 = -1 plans -0.1 and draws 0, zeta_1 =
   +1 plans 1.1 and draws its rate 1). With the planned 1.5 the stored-energy row read
   (2 - 2) / 0.5 = 0, broken at every scale above 0 (slot loss 1 against the corners' 0, the
   difference once named "decision-hour cap"). The driver-loss bound counts the draw too: the car
   protects its room 2 (rho 0) and ends, on the worst request, with 1 + 0.5 - 0.5 beta, so the
   earlier bound is (0.5 + 0.5 beta) / 2: 0.5 at scale 1, equal to the corners' largest driver
   loss (zeta_1 = -1: 1 kWh of 2 short), and 0.55 at 1.2 against the corners' 0.5 (the plan -0.1
   draws 0). The every-scale bound, the driver bound, is the same 0.5 at 1; at 1.2, where hour 1
   fails, it counts the least the car can draw in that hour, 0, so it is 0.5, the corners'
   largest. With the planned 1.5 the earlier bound read 0.25 and 0.3, below the corners.
   Second (`TIGHT`): one car, one service hour, room 0.99995 kWh, the plan `0.5 + 0.5 beta
   zeta_1`. At scale 1 its stored-energy row is broken on (+1) by 5e-5 kWh, so the closed form
   fails hour 1 (threshold 0.9999); replay draws 0.99995 of the requested 1 kWh, which plan section
   1 does not count as a failure (it needs a miss above 1e-4 kWh): "within tolerance", which
   agrees by reading K3 of `corner_run` but not strictly. At scale 0.9 both say no failure.
9. A path breaks on a failing hour or on more than `FAILURE_KWH` of protected energy undelivered;
   a shortfall above 0 and up to it (whatever its cause) is counted apart; the worst path is the
   first of equal losses among the breaking paths.
10. The undelivered protected energy of a corner is summed over the cars (`PAIR`): two cars each
    short by less than `FAILURE_KWH`, together by more, break the corner.
11. The driver flags by hand (`driver_flags`, reading K4): over the corners within
    `DRIVER_TOLERANCE` of the largest driver loss, a car short of its plan on every one, no
    failing hour on some one; a corner further below is not read, a corner at the tolerance is.
12. What the methods are given: a method that plans on the rows the box was certified on
    (`diagonal-split`) is searched with them (the audited reading equals `compare_day.evaluate`'s;
    one programme and one box counted for the day and scale) and refuses without them; each
    controller is run with the day's forecast belief, the plan's hour-zero target, the corner's
    requests and the day's departure reading; the re-solve with the one-way rung, the corners,
    the day's departure-bound table, budget and departure reading and `lp.OTHER_SECONDS`.
"""

import dataclasses
import datetime
import functools
from types import SimpleNamespace

import numpy as np
import pytest
from corner_fake import HIDDEN, SHORT, SINGLE, hand

from fleetreserve import compare_day, compare_methods, corner_search, day_replay, lp, rungs
from fleetreserve.controller import forecast_belief, hour_zero_target
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.plans import Plan
from fleetreserve.policy import AffinePolicy
from fleetreserve.products import corner_paths
from fleetreserve.resolve import ResolveResult
from fleetreserve.sparse_programme import Sizes

FIXED = compare_methods.select(["fixed-plan"])[0]
RE_SOLVE = compare_methods.select([compare_methods.RE_SOLVE])[0]


def _breaking(found):
    return [k for k, p in enumerate(found.paths) if p.slot_loss > 0 or p.driver_loss > 0]


# --- 1. numbering --------------------------------------------------------------------------------


def test_corner_numbering_follows_the_enumeration():
    assert corner_search.corner_index([-1, -1]) == 0
    assert corner_search.corner_index([1, -1]) == 1
    assert corner_search.corner_index([-1, 1]) == 2
    assert corner_search.corner_index([1, 1]) == 3
    for n in range(1, 6):
        paths = corner_paths(n)
        assert [corner_search.corner_index(z) for z in paths] == list(range(2**n))


def test_a_path_inside_the_box_is_refused():
    with pytest.raises(ValueError, match=r"^a corner has entries -1 or \+1 only$"):
        corner_search.corner_index([1.0, 0.5])


# --- 2. to 4. hand days --------------------------------------------------------------------------


def test_single_breaks_on_the_upper_corner_only():
    found = corner_search.search_day(SINGLE, FIXED, 1.0)
    assert found.status == "breaks" and found.n_paths == 4
    assert _breaking(found) == [3]
    assert found.breaking_paths == found.slot_breaking_paths == 1
    assert found.driver_short_paths == 0
    assert found.worst_slot_loss == pytest.approx(0.5)
    assert found.worst_driver_loss == pytest.approx(0.0)
    assert found.worst_path == (1, 1) and found.worst_first_failure == 2
    assert found.paths[3].first_failure == 2
    assert all(p.first_failure is None for p in found.paths[:3])
    assert found.audited_breaks is True
    assert found.audited_breaking_rows == 3  # rows 2, 4 and 5 of the day's audited set are (+1, +1)
    assert found.audited_worst_slot_loss == pytest.approx(0.5)


def test_hidden_breaks_on_a_corner_the_audited_paths_miss():
    found = corner_search.search_day(HIDDEN, FIXED, 1.0)
    assert found.status == "breaks"
    assert _breaking(found) == [2]
    assert found.worst_path == (-1, 1)
    assert found.worst_slot_loss == pytest.approx(0.5)
    assert found.audited_breaks is False and found.audited_breaking_rows == 0
    assert found.audited_worst_slot_loss == pytest.approx(0.0)


def test_hidden_holds_at_half_scale():
    found = corner_search.search_day(HIDDEN, FIXED, 0.5)
    assert found.status == "holds"
    assert found.breaking_paths == 0 and _breaking(found) == []
    assert found.worst_slot_loss == 0.0 and found.worst_driver_loss == 0.0
    assert found.worst_path is None and found.worst_first_failure is None


def test_short_breaks_by_driver_loss_alone():
    found = corner_search.search_day(SHORT, FIXED, 0.5)
    assert found.status == "breaks"
    assert found.slot_breaking_paths == 0
    assert found.driver_short_paths == found.breaking_paths == 3
    losses = [p.driver_loss for p in found.paths]
    assert losses == pytest.approx([2 / 3, 1 / 3, 1 / 3, 0.0])
    assert found.worst_driver_loss == pytest.approx(2 / 3)
    assert found.worst_path == (-1, -1) and found.worst_first_failure is None


# --- 5. no offer and refusals --------------------------------------------------------------------


def test_scale_zero_is_no_offer():
    found = corner_search.search_day(SINGLE, FIXED, 0.0)
    assert found.status == "no offer" and found.beta == 0.0 and found.method == FIXED.name
    assert found.n_paths == 0 and found.paths == ()
    assert found.worst_slot_loss is None and found.audited_breaks is None


def test_an_abstaining_day_is_refused():
    gone = dataclasses.replace(SINGLE, abstain_reason="no steered car")
    message = r"^the day abstains \(no steered car\): it has no box to search$"
    with pytest.raises(ValueError, match=message):
        corner_search.search_day(gone, FIXED, 1.0)
    message = r"^the day abstains \(no steered car\): it has no box to check$"
    with pytest.raises(ValueError, match=message):
        corner_search.closed_form_check(gone, (1.0,))


def test_a_method_of_an_unknown_kind_is_refused():
    odd = dataclasses.replace(FIXED, kind="oracle")
    with pytest.raises(ValueError, match=r"^method 'fixed-plan': unknown kind 'oracle'$"):
        corner_search.search_day(SINGLE, odd, 1.0)


# --- 6. timeouts ---------------------------------------------------------------------------------


def _timeout(*args, **kwargs):
    return ResolveResult("timeout", None, 0.0, 0.0, Sizes(0, 0, 0), "injected")


def test_every_re_solve_timed_out_is_timed_out_not_a_result(monkeypatch):
    monkeypatch.setattr(
        day_replay, "run_rungs", functools.partial(rungs.run_rungs, resolver=_timeout)
    )
    # at scale 1.2 hour 1 fails on every corner of SINGLE, so the rung re-solves at hour 2
    found = corner_search.search_day(SINGLE, RE_SOLVE, 1.2)
    assert found.timed_out_paths == 4
    assert found.status == "timed out"
    assert found.breaking_paths == 0
    assert found.worst_slot_loss is None
    # one re-solve at hour 2 on each corner, and one at hour 1 on the two corners with zeta_1 = -1,
    # where the planned -0.1 kW cannot meet the request of -0.1 kW (a certain failure)
    assert found.resolves == {"timeout": 6}
    assert found.audited_breaks is None


def test_re_solve_without_timeouts_has_a_result():
    found = corner_search.search_day(SINGLE, RE_SOLVE, 1.2)
    assert found.timed_out_paths == 0
    assert found.status in ("breaks", "holds")
    assert sum(found.resolves.values()) >= 4


def _path(slot, driver, timed_out=False, kwh=None):
    lost = driver if kwh is None else kwh  # a day with 1 kWh protected, unless given
    return corner_search.CornerPath(1 if slot else None, slot, driver, timed_out, lost)


def test_day_status_by_hand():
    status = corner_search.day_status
    assert status([_path(0.0, 0.0), _path(0.0, 0.0)]) == "holds"
    assert status([_path(0.0, 0.0), _path(0.5, 0.0)]) == "breaks"
    assert status([_path(0.0, 0.1), _path(0.0, 0.0)]) == "breaks"
    assert status([_path(0.0, 0.0), _path(0.0, 0.0, True)]) == "timed out"
    assert status([_path(0.5, 0.0, True), _path(0.0, 0.0)]) == "timed out"
    assert status([_path(0.5, 0.0), _path(0.0, 0.0, True)]) == "breaks"


def test_a_shortfall_up_to_the_failure_tolerance_is_not_a_break():
    status = corner_search.day_status
    tolerance = corner_search.FAILURE_KWH
    rounding = _path(0.0, 4e-17, kwh=4e-17)  # a car a rounding step short of its need
    assert not rounding.breaks and rounding.short_below_tolerance and not rounding.driver_short
    assert status([rounding, _path(0.0, 0.0)]) == "holds"
    at = _path(0.0, 1e-5, kwh=tolerance)
    assert not at.breaks and at.short_below_tolerance
    above = _path(0.0, 1e-5, kwh=1.5 * tolerance)
    assert above.breaks and above.driver_short and not above.short_below_tolerance
    assert status([above, _path(0.0, 0.0)]) == "breaks"
    slot = _path(0.5, 0.0)
    assert slot.breaks and not slot.short_below_tolerance
    reading = corner_search.read([rounding, above, slot, _path(0.0, 0.0)])
    assert (reading.breaking, reading.driver_short, reading.short_below_tolerance) == (2, 1, 1)
    assert reading.worst == 2  # the largest slot loss first, among the breaking paths


def test_the_worst_path_is_the_first_of_equal_losses():
    reading = corner_search.read([_path(0.0, 0.0), _path(0.5, 0.2), _path(0.5, 0.2)])
    assert reading.worst == 1
    reading = corner_search.read([_path(0.5, 0.1), _path(0.5, 0.2), _path(0.5, 0.2)])
    assert reading.worst == 1


# --- 7. the closed-form check by hand ------------------------------------------------------------


def test_closed_form_check_single():
    checks = corner_search.closed_form_check(SINGLE, (0.4, 0.75, 1.0, 1.2))
    assert [c.closed_hour for c in checks] == [None, 2, 2, 1]
    assert [c.corner_hour for c in checks] == [None, 2, 2, 1]
    assert [c.closed_slot for c in checks] == pytest.approx([0.0, 0.5, 0.5, 1.0])
    assert [c.corner_slot for c in checks] == pytest.approx([0.0, 0.5, 0.5, 1.0])
    assert all(c.agreed and c.reason is None and c.slot_difference == 0.0 for c in checks)
    assert all(c.agreed_strict for c in checks)
    assert all(c.driver_within_bound for c in checks)
    assert not checks[0].driver_limited and checks[0].driver_no_slot_failure


def test_a_corner_slot_loss_above_the_closed_form_does_not_agree(monkeypatch):
    # the closed form made to say a quarter less than the corners show on SINGLE at scale 1
    original = corner_search.crc.step_loss
    monkeypatch.setattr(corner_search.crc, "step_loss", lambda *a: original(*a) - 0.25)
    (check,) = corner_search.closed_form_check(SINGLE, (1.0,))
    assert check.closed_slot == pytest.approx(0.25) and check.corner_slot == pytest.approx(0.5)
    assert check.slot_difference == pytest.approx(0.25)
    assert not check.agreed_strict and not check.agreed
    assert check.reason == "slot losses differ"


def test_closed_form_check_short_driver_bound():
    checks = corner_search.closed_form_check(SHORT, (0.5, 1.0, 1.2))
    assert all(c.agreed for c in checks)
    assert [c.driver_bound for c in checks] == pytest.approx([2 / 3, 1.0, 1.0])
    assert [c.corner_driver for c in checks] == pytest.approx([2 / 3, 1.0, 1.0])
    assert all(c.driver_within_bound for c in checks)


# --- 8. the decision-hour cap and the failure tolerance ------------------------------------------

CAP_CAR = TrueCar(2, 1.0, 2.0)
CAP_DAY = Day(
    "hand",
    datetime.date(2020, 1, 1),
    1,
    0,
    0.0,
    (CAP_CAR,),
    (ForecastCar(2.0, 1.5, 2.5, 2.0),),
    0.1,
)
CAP = hand(CAP_DAY, AffinePolicy([0.5], [0.5], [[1.5, 0.5]], [[[0.0], [0.5]]]))


def test_the_decision_hour_cap_agrees_with_the_corners():
    one, above = corner_search.closed_form_check(CAP, (1.0, 1.2))
    assert one.closed_hour is None and one.closed_slot == 0.0
    assert one.corner_hour is None and one.corner_slot == 0.0
    assert one.agreed_strict and one.agreed and one.reason is None
    assert one.driver_limited  # the car drew 1 of its planned 1.5 kWh in hour 0
    assert (above.closed_hour, above.corner_hour) == (1, 1)
    assert above.closed_slot == above.corner_slot == pytest.approx(1.0)
    assert above.agreed_strict and above.reason is None
    runs = corner_search.fixed_plan_corners(CAP, 1.0)
    assert [float(r.draws[0, 0]) for r in runs] == [1.0, 1.0]  # rate 1 against the planned 1.5
    assert np.all([r.first_failure is None for r in runs])


def test_the_driver_bound_of_the_decision_hour_cap_counts_the_draw():
    one, above = corner_search.closed_form_check(CAP, (1.0, 1.2))
    assert one.driver_bound == one.earlier_driver_bound == pytest.approx(0.5)
    assert one.corner_driver == pytest.approx(0.5)
    assert above.driver_bound == pytest.approx(0.5)
    assert above.earlier_driver_bound == pytest.approx(0.55)
    assert above.corner_driver == pytest.approx(0.5)
    assert one.driver_within_bound and above.driver_within_bound


TIGHT = hand(
    Day(
        "hand",
        datetime.date(2020, 1, 1),
        1,
        0,
        0.0,
        (TrueCar(2, 1.0, 0.99995),),
        (ForecastCar(2.0, 1.0, 1.0, 0.99995),),
        0.1,
    ),
    AffinePolicy([0.5], [0.5], [[0.0, 0.5]], [[[0.0], [0.5]]]),
)


def test_second_disagreement_is_within_the_failure_tolerance():
    (check,) = corner_search.closed_form_check(TIGHT, (1.0,))
    assert check.closed_hour == 1 and check.corner_hour is None
    assert not check.agreed_strict and check.reason == "within tolerance"
    assert check.agreed  # reading K3: replay meets every request by plan section 1's rule
    (below,) = corner_search.closed_form_check(TIGHT, (0.9,))
    assert below.agreed_strict and below.agreed and below.closed_hour is None


# --- 10. the shortfall is summed over the cars ----------------------------------------------------

# Two equal cars, one service hour, no recovery: each leaves after hour 1 (rate 1 kW, room 1 kWh)
# with rho = 0.74994, so each protects 1 - 0.74994 = 0.25006 kWh. The plan gives each
# `0.5 + 0.25 beta zeta_1` in hour 1 (the request `1 + 0.5 beta zeta_1`). At scale 1 the corner
# (-1) asks 0.25 of each, which each draws: no hour fails, each car is 6e-5 kWh short, 1.2e-4 kWh
# in all, above FAILURE_KWH (1e-4) although each car alone is below it: the corner breaks. The
# corner (+1) gives each 0.75 kWh, above its protected energy: it holds.
PAIR_RHO = 0.74994
PAIR_CARS = (TrueCar(2, 1.0, 1.0), TrueCar(2, 1.0, 1.0))
PAIR = hand(
    Day(
        "hand",
        datetime.date(2020, 1, 1),
        1,
        0,
        PAIR_RHO,
        PAIR_CARS,
        tuple(ForecastCar(2.0, 1.0, 1.0, 1.0 - PAIR_RHO) for _ in PAIR_CARS),
        0.1,
    ),
    AffinePolicy([1.0], [0.5], [[0.0, 0.5], [0.0, 0.5]], [[[0.0], [0.25]], [[0.0], [0.25]]]),
)


def test_the_shortfall_of_a_corner_is_summed_over_the_cars():
    short = 2 * ((1.0 - PAIR_RHO) - 0.25)  # 1.2e-4 kWh: 6e-5 for each car
    assert corner_search.FAILURE_KWH < short < 2 * corner_search.FAILURE_KWH
    found = corner_search.search_day(PAIR, FIXED, 1.0)
    low, high = found.paths
    assert low.first_failure is None and high.first_failure is None
    assert low.shortfall_kwh == pytest.approx(short, abs=1e-12)
    assert high.shortfall_kwh == 0.0
    assert low.breaks and low.driver_short and not low.short_below_tolerance
    assert found.status == "breaks"
    assert (found.breaking_paths, found.driver_short_paths) == (1, 1)
    assert (found.slot_breaking_paths, found.short_below_tolerance_paths) == (0, 0)
    assert found.worst_path == (-1,)


# --- 11. the driver flags by hand -----------------------------------------------------------------


def _run(loss, short, fails):
    planned = np.array([[0.0, 1.0]])
    draws = np.array([[0.0, 0.5 if short else 1.0]])
    return SimpleNamespace(
        driver_loss=loss, planned=planned, draws=draws, first_failure=1 if fails else None
    )


def test_the_driver_flags_by_hand():
    flags = corner_search.driver_flags
    tolerance = corner_search.DRIVER_TOLERANCE
    assert corner_search.drew_less_than_planned(_run(0.1, True, False))
    assert not corner_search.drew_less_than_planned(_run(0.1, False, False))
    # one corner of the largest loss
    assert flags([_run(0.1, False, True), _run(0.3, True, True)]) == (True, False)
    assert flags([_run(0.3, False, False), _run(0.1, True, True)]) == (False, True)
    # two corners of the largest loss, half a tolerance apart, differing in both flags: not every
    # one is short of its plan; some one fails no hour; the order does not matter
    first, second = _run(0.3, True, True), _run(0.3 - 0.5 * tolerance, False, False)
    assert flags([first, second]) == flags([second, first]) == (False, True)
    # a corner twice the tolerance below the largest is not read
    below = _run(0.3 - 2 * tolerance, False, False)
    assert flags([first, below]) == flags([below, first]) == (True, False)
    # a corner exactly the tolerance below is read
    at = _run(0.25, False, False)
    assert flags([_run(0.25 + tolerance, True, True), at]) == (False, True)
    # every corner short and failing, and none of them
    assert flags([_run(0.2, True, True)] * 3) == (True, False)
    assert flags([_run(0.2, False, False)] * 3) == (False, True)


# --- 12. what the methods are given ---------------------------------------------------------------

DIAGONAL = compare_methods.select(["diagonal-split"])[0]


@pytest.mark.parametrize("prepared", [SINGLE, HIDDEN], ids=["single", "hidden"])
def test_a_method_that_plans_on_the_rows_is_searched_with_them(prepared):
    found = corner_search.search_day(prepared, DIAGONAL, 1.0)
    assert found.status in ("breaks", "holds") and found.n_paths == 4
    direct = compare_day.evaluate(prepared, 1.0, (DIAGONAL,))[DIAGONAL.name]
    assert found.audited_worst_slot_loss == pytest.approx(direct.slot_loss, abs=1e-12)
    assert found.audited_worst_driver_loss == pytest.approx(direct.driver_loss, abs=1e-12)
    counts = found.solver_counts
    # one programme and one box for the day and scale, the hours of every corner (hours 0 to 2)
    assert sum(counts["solves"].values()) == sum(counts["boxes"].values()) == 1
    assert counts["hours"] == 4 * 3 and counts["fallback_hours"] == 0


def test_without_the_rows_a_method_that_plans_on_them_refuses():
    gone = dataclasses.replace(SINGLE, rows=None)
    with pytest.raises(compare_methods.MethodError):
        corner_search.search_day(gone, DIAGONAL, 1.0)


def test_the_controllers_are_given_the_days_belief_target_requests_and_reading(monkeypatch):
    calls = []
    original = corner_search.run_controller

    def spy(day, controller, belief, zeta, requests, **options):
        calls.append((day, belief, np.array(zeta), np.array(requests), options))
        return original(day, controller, belief, zeta, requests, **options)

    monkeypatch.setattr(corner_search, "run_controller", spy)
    for seen in ("hour-start", "rounded-hour"):
        calls.clear()
        prepared = dataclasses.replace(HIDDEN, departure_seen=seen)
        corner_search.search_day(prepared, FIXED, 0.8)
        policy = prepared.certificate.policy
        target = hour_zero_target(Plan.from_policy(policy, 0.8))
        assert len(calls) == 4
        for (day, belief, zeta, requests, options), corner in zip(
            calls, corner_paths(2), strict=True
        ):
            assert day is prepared.day
            expected = forecast_belief(prepared.day)
            for field in ("departure", "rate", "cap", "need"):
                assert getattr(belief, field).tolist() == getattr(expected, field).tolist()
            assert zeta.tolist() == corner.tolist()
            assert requests.tolist() == np.asarray(policy.request(corner, 0.8)).tolist()
            assert options == {"hour_zero_target": target, "departure_seen": seen}


def test_the_re_solve_is_given_the_days_bounds_budget_limit_and_reading(monkeypatch):
    calls = []
    original = corner_search.replay_day

    def spy(day, certificate, rung, beta, paths, **options):
        calls.append((day, certificate, rung, beta, np.array(paths), options))
        return original(day, certificate, rung, beta, paths, **options)

    monkeypatch.setattr(corner_search, "replay_day", spy)
    prepared = dataclasses.replace(SINGLE, departure_seen="hour-start", budget=7.5)
    corner_search.search_day(prepared, RE_SOLVE, 1.2)
    ((day, certificate, rung, beta, paths, options),) = calls
    assert (day, certificate, rung, beta) == (prepared.day, prepared.certificate, "one-way", 1.2)
    assert paths.tolist() == corner_paths(2).tolist()
    assert set(options) == {"departure_bound", "budget", "time_limit", "departure_seen"}
    assert (options["budget"], options["time_limit"]) == (7.5, lp.OTHER_SECONDS)
    assert options["departure_seen"] == "hour-start"
    table = prepared.departure_bounds
    for car in range(table.shape[0]):
        for hour in range(table.shape[1]):
            assert options["departure_bound"](car, hour) == float(table[car, hour])
