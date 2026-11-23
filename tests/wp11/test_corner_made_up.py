"""The corner search and the closed-form check on seeded made-up prepared days (`compare_fake`).

1. The pass mark's substance: on `N_DAYS` made-up days, at every scale of `dial_splits.GRID`,
   the closed form's first failing hour and slot loss (from `gate_calibrate.slot_steps`) equal the
   corner worst case of replay on `N_STRICT` of `N_PAIRS` (day, scale) pairs (strictly); on every
   other pair the closed form says an earlier failure than any corner shows, for a reason of
   `REASONS` ("within tolerance": a row broken by less than the failure tolerance of 1e-4 kWh).
   With the tolerance (reading K3) `N_AGREED` agree. Two of the days have a car capped in hour 0;
   with its decision-hour draw in the closed form they agree (before the owner's decision of 16
   November 2026, with the planned hour-0 power: 229 strict, 230 agreed, 10 pairs "decision-hour
   cap"). The sample holds pairs that fail and pairs that do not. The
   calibration's step form gives the same slot loss as `fixed_plan_loss.closed_form_slot_loss` on
   every pair. The corner maximum of the driver loss is above the driver bound, the every-scale
   bound (`driver_bound`), on `N_EVERY_ABOVE` pairs; it is above the earlier, summed per-car
   bound (`earlier_driver_bound`) on `N_DRIVER_ABOVE` pairs, on each of which every corner of
   the largest driver loss has a car that drew less than planned; on `N_ABOVE_NO_FAILURE` of
   them some corner of that loss fails no hour (reading K4).
2. No path inside the box does worse than the worst corner for the fixed plan: on random interior
   paths the slot loss is never above the corner worst case. For the least-laxity rule an interior
   path can be worse than every corner (a seeded made-up day found by search, pinned).
3. The controllers' departure reading and the audited reading of every method on these days:
   `test_corner_properties` (slow); their arguments by hand: `test_corner_hand`.
4. The registry's `fixed-plan` (a controller, `plan_controller`) gives on every corner what
   `replay.run_plan` gives (`corner_search.fixed_plan_corners`), and the corner figures of the
   closed-form check (worst slot loss, earliest failing hour, largest driver loss) are that
   search's worst case.
5. The driver flags of a pair are read on the corners of the largest driver loss (within
   `DRIVER_TOLERANCE` of it; reading K4). On the made-up day `offer_days(40, seed=5)[11]` at
   scale 1 (2 service hours, 5 cars, rho 0) the corners differ in them: car 3 (index 2) leaves
   after hour 1, and the plan gives it power in hour 2 on corners 0, 2 and 3 (it draws nothing
   there: short of its plan for lack of presence). On corners 0 (-1, -1) and 2 (-1, +1) it is
   planned nothing before hour 2 and ends with nothing, so the driver loss is its room over the
   summed rooms, the largest, and hour 2 fails; on corner 1 (+1, -1) it is planned nothing in
   hour 2, ends with what it drew in hour 1 and no hour fails. The corners of the largest loss
   (0 and 2) both give (car short of its plan, no failing hour) = (True, False); corner 1, of the
   smallest loss, gives (False, True), and corner 3, of the same smallest loss, (True, False).
   On the made-up day `offer_days(40, seed=74)[20]` (2 service hours, 5 cars, rho 0) at the
   scales 0.4 to 1 all four corners have the largest driver loss, every one has a car short of
   its plan, and corners 0 to 2 fail no hour while corner 3 fails hour 2: the pair counts as
   having no failing hour on a corner of its largest loss (read on any one of them), which the
   last of the tied corners, or the corner of the largest slot loss, would not give. The car
   short of its plan on every corner is car 1 (index 0), planned 3.5659 kWh in hour 0 against its
   rate of 3 kW; with that decision-hour draw counted, the driver-loss bound equals the corners'
   largest loss (about 0.0694) at every scale, so no pair of the day is above it (before the
   owner's decision of 16 November 2026 the bound was 0 there, below every corner on all ten
   pairs). The summary's counts over the pairs above the bound are read on the day's checks
   marked above it. The expected values come from a replay of each corner in the test
   (`replay.run_plan`) and the day's rooms, not from the check. On the day of seed 5 the
   every-scale bound equals the largest loss (car 3 fails hour 2 on the corners of that loss and
   the bound counts the nothing it can draw there), so the pair is within it and above only the
   earlier bound; the summary's counts are read on the check marked above it.
"""

import dataclasses

import numpy as np
import pytest
from corner_fake import offer_days

from fleetreserve import (
    compare_methods,
    corner_search,
    corner_summary,
    dial_splits,
    replay,
)
from fleetreserve.carsets import true_rows
from fleetreserve.fixed_plan_loss import closed_form_slot_loss, hour_thresholds
from fleetreserve.plans import Plan
from fleetreserve.products import corner_paths

N_DAYS = 24
N_PAIRS = N_DAYS * len(dial_splits.GRID)
N_STRICT = 239
N_AGREED = 240
REASONS = {"within tolerance": 1}
N_EVERY_ABOVE = 0
N_DRIVER_ABOVE = 70
N_ABOVE_NO_FAILURE = 0
DAYS = offer_days(N_DAYS, seed=11)


def test_closed_form_equals_corner_worst_case_on_made_up_days():
    pairs = strict = agreed = failing = holding = 0
    reasons, above, every_above = {}, [], 0
    for prepared in DAYS:
        thresholds = hour_thresholds(prepared.certificate.policy, true_rows(prepared.day))
        for c in corner_search.closed_form_check(prepared, dial_splits.GRID):
            pairs += 1
            assert c.closed_slot == pytest.approx(closed_form_slot_loss(thresholds, c.beta))
            assert c.agreed_strict == (
                c.closed_hour == c.corner_hour and c.slot_difference <= corner_search.SLOT_TOLERANCE
            )
            assert c.agreed == (c.agreed_strict or c.reason == "within tolerance")
            strict += c.agreed_strict
            agreed += c.agreed
            if c.agreed_strict:
                assert c.reason is None
            else:
                reasons[c.reason] = reasons.get(c.reason, 0) + 1
                assert c.corner_slot < c.closed_slot
            failing += c.corner_hour is not None
            holding += c.corner_hour is None
            every_above += not c.driver_within_bound
            if c.corner_driver > c.earlier_driver_bound + corner_search.DRIVER_TOLERANCE:
                above.append(c)
            assert c.driver_within_bound == (
                c.corner_driver <= c.driver_bound + corner_search.DRIVER_TOLERANCE
            )
    limited = sum(c.driver_limited for c in above)
    no_failure = sum(c.driver_no_slot_failure for c in above)
    print(
        f"{strict} / {agreed} of {pairs}; {reasons}; above {every_above}; earlier above "
        f"{len(above)}, {limited}, {no_failure}"
    )
    assert failing > 0 and holding > 0
    assert (pairs, strict, agreed, reasons) == (N_PAIRS, N_STRICT, N_AGREED, REASONS)
    assert every_above == N_EVERY_ABOVE
    # every pair above the summed per-car bound has a car short of its plan (the bound assumes
    # the plan is delivered)
    assert (len(above), limited, no_failure) == (N_DRIVER_ABOVE, N_DRIVER_ABOVE, N_ABOVE_NO_FAILURE)


def test_no_interior_path_does_worse_than_the_worst_corner():
    rng = np.random.default_rng(5)
    for prepared in DAYS[:10]:
        policy = prepared.certificate.policy
        for beta in (0.3, 0.7, 1.0):
            worst = max(r.slot_loss for r in corner_search.fixed_plan_corners(prepared, beta))
            plan = Plan.from_policy(policy, beta)
            for _ in range(20):
                zeta = rng.uniform(-1, 1, size=prepared.day.n_service)
                run = replay.run_plan(prepared.day, plan, zeta, policy.request(zeta, beta))
                assert run.slot_loss <= worst + 1e-12


def test_fixed_plan_controller_matches_run_plan_on_every_corner():
    method = compare_methods.select(["fixed-plan"])[0]
    for prepared in DAYS:
        for beta in (0.2, 0.6, 1.0):
            found = corner_search.search_day(prepared, method, beta)
            runs = corner_search.fixed_plan_corners(prepared, beta)
            assert [p.first_failure for p in found.paths] == [r.first_failure for r in runs]
            assert [p.driver_loss for p in found.paths] == pytest.approx(
                [r.driver_loss for r in runs], abs=1e-12
            )


def test_closed_form_check_corner_figures_are_the_fixed_plan_search_worst_case():
    method = compare_methods.select(["fixed-plan"])[0]
    betas = (0.4, 1.0)
    for prepared in DAYS:
        for c in corner_search.closed_form_check(prepared, betas):
            found = corner_search.search_day(prepared, method, c.beta)
            assert c.corner_driver == pytest.approx(found.worst_driver_loss, abs=1e-12)
            assert c.corner_slot == pytest.approx(found.worst_slot_loss, abs=1e-12)
            hours = [p.first_failure for p in found.paths if p.first_failure is not None]
            assert c.corner_hour == (min(hours) if hours else None)


def test_for_the_least_laxity_rule_an_interior_path_is_worse_than_every_corner():
    llf = compare_methods.select(["llf"])[0]
    prepared = DAYS[8]
    assert (prepared.day.n_service, len(prepared.day.cars)) == (4, 5)
    corners = corner_search.search_day(prepared, llf, 0.6)
    (inside,), _, _ = corner_search.run_paths(prepared, llf, 0.6, [np.array([-1, -1, 0, -1.0])])
    assert corners.worst_driver_loss == pytest.approx(0.14664151087552885, abs=1e-12)
    assert inside.driver_loss == pytest.approx(0.19646538260407453, abs=1e-12)


def test_the_driver_flags_describe_the_first_corner_of_the_largest_driver_loss():
    prepared = offer_days(40, seed=5)[11]
    day = prepared.day
    assert (day.n_service, len(day.cars), day.rho) == (2, 5, 0.0)
    assert day.cars[2].departure == 2  # draws in hours 0 and 1 only
    policy = prepared.certificate.policy
    plan = Plan.from_policy(policy, 1.0)
    runs = [
        replay.run_plan(day, plan, z, policy.request(z, 1.0)) for z in corner_paths(day.n_service)
    ]
    rooms = np.array([c.room for c in day.cars])
    # by hand: car 2 ends with nothing on corners 0 and 2, with its hour-1 draw on 1 and 3; every
    # other car ends full on every corner
    assert [float(r.planned[2, 2]) > 0 for r in runs] == [True, False, True, True]
    ends = [float(r.final[2]) for r in runs]
    assert ends[0] == ends[2] == 0.0 and ends[1] == ends[3] == float(runs[1].draws[2, 1]) > 0
    for r in runs:
        assert np.delete(r.final, 2) == pytest.approx(np.delete(rooms, 2), abs=1e-12)
    largest = rooms[2] / rooms.sum()
    smallest = (rooms[2] - ends[1]) / rooms.sum()
    assert [r.driver_loss for r in runs] == pytest.approx(
        [largest, smallest, largest, smallest], abs=1e-12
    )
    short = [bool(np.any((r.planned > 0) & (r.draws < r.planned))) for r in runs]
    no_failure = [r.first_failure is None for r in runs]
    assert short == [True, False, True, True] and no_failure == [False, True, False, False]
    (check,) = corner_search.closed_form_check(prepared, (1.0,))
    assert check.corner_driver == pytest.approx(largest, abs=1e-12)
    assert check.driver_bound == pytest.approx(largest, abs=1e-12) and check.driver_within_bound
    assert check.corner_driver > check.earlier_driver_bound + corner_search.DRIVER_TOLERANCE
    assert check.earlier_driver_bound == pytest.approx(0.0986, abs=1e-4)  # below about 0.1604
    # corner 0, the first of the largest loss, is described; corner 1, the smallest, is not
    assert (check.driver_limited, check.driver_no_slot_failure) == (short[0], no_failure[0])
    assert (short[1], no_failure[1]) != (short[0], no_failure[0])
    entry = corner_summary.pass_mark_entry([11], ["made up"], [(check,)])
    assert (entry["driver_pairs_above_bound"], entry["earlier_driver_pairs_above_bound"]) == (0, 1)
    marked = dataclasses.replace(check, driver_within_bound=False)
    entry = corner_summary.pass_mark_entry([11], ["made up"], [(marked,)])
    assert entry["driver_pairs_above_bound"] == 1
    assert entry["driver_above_bound_limited"] == 1
    assert entry["driver_above_bound_no_slot_failure"] == 0


def test_the_driver_flags_read_every_corner_of_the_largest_driver_loss():
    prepared = offer_days(40, seed=74)[20]
    day = prepared.day
    assert (day.n_service, len(day.cars), day.rho) == (2, 5, 0.0)
    policy = prepared.certificate.policy
    checks = corner_search.closed_form_check(prepared, dial_splits.GRID)
    differ = 0
    for check in checks:
        beta = check.beta
        plan = Plan.from_policy(policy, beta)
        runs = [
            replay.run_plan(day, plan, z, policy.request(z, beta))
            for z in corner_paths(day.n_service)
        ]
        drivers = [r.driver_loss for r in runs]
        top = max(drivers)
        tied = [k for k, d in enumerate(drivers) if top - d <= corner_search.DRIVER_TOLERANCE]
        short = [bool(np.any((r.planned > 0) & (r.draws < r.planned))) for r in runs]
        no_failure = [r.first_failure is None for r in runs]
        # car index 0 draws its rate 3 of its planned 3.5659 in hour 0 on every corner, and the
        # bound counts that draw: equal to the largest loss at every scale of the grid
        assert [float(r.draws[0, 0]) for r in runs] == [3.0] * 4
        assert all(float(r.planned[0, 0]) == pytest.approx(3.5659, abs=1e-4) for r in runs)
        assert check.driver_bound == pytest.approx(top, abs=1e-12)
        assert check.driver_within_bound
        assert check.corner_driver == top
        assert check.driver_limited == all(short[k] for k in tied)
        assert check.driver_no_slot_failure == any(no_failure[k] for k in tied)
        if beta >= 0.4:
            assert tied == [0, 1, 2, 3] and no_failure == [True, True, True, False]
            assert all(short) and check.driver_no_slot_failure
            slots = [r.slot_loss for r in runs]
            # the last tied corner and the corner of the largest slot loss both fail hour 2
            assert no_failure[tied[-1]] is False and no_failure[int(np.argmax(slots))] is False
            differ += 1
    assert differ == 6
    assert (
        corner_summary.pass_mark_entry([20], ["made up"], [checks])["driver_pairs_above_bound"] == 0
    )
    marked = tuple(dataclasses.replace(c, driver_within_bound=False) for c in checks)
    entry = corner_summary.pass_mark_entry([20], ["made up"], [marked])
    assert entry["driver_pairs_above_bound"] == len(dial_splits.GRID)
    assert entry["driver_above_bound_no_slot_failure"] == sum(
        c.driver_no_slot_failure for c in checks
    )
    assert entry["driver_above_bound_no_slot_failure"] == len(dial_splits.GRID)
