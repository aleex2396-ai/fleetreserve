"""The diagnosis against the comparison it explains, on the seeded made-up offer days of
`compare_fake.offer_days`.

1. Plans unchanged: the planned powers of `asa` and `asa-es`, replayed by
   `controller_replay.run_controller` as `compare_day.evaluate` replays them, on the first 8 offer
   days at the scales 0.3 and 1 on every audited path, hash (SHA-256 of the float64 bytes, in that
   order) to the value computed at the commit before `asa_controller.AsaController.hour_plans` and
   `asa_equal_share.share_goal` were added (7dfb852): the record and the split of the fifth goal's
   construction changed no plan.
2. The record: after a path, `hour_plans` holds one `HourPlan` per hour planned, and each optimal
   one's first-hour powers are the powers the controller returned.
3. The replays of `diagnose_day` are those of `compare_day.evaluate` for the same day, scale and
   path: per method the largest slot and driver losses over the paths and each path's first
   failure are equal, and the planned powers equal those of a replay of the method built by
   `compare_methods` (first 6 offer days, scales 0.3 and 1, `asa-es` against `llf`). Every
   controller method of the registry can be the reference: the inputs carry the rows the day's
   box was certified on, as `compare_day.evaluate`'s do, and its largest slot loss and first
   failures equal that call's (first 3 offer days, scale 1, against `asa-es`).
4. The causes partition the method-only hours, and each row meets the exact test of its cause and
   fails those of the causes before it (first 15 offer days, scales 0.3, 0.6 and 1, `asa-es` and
   `asa` each against `llf`): rows = method-only hours, the four groups add up to paths times
   service hours; "state" rows have the request above `method_high` by more than the tolerance,
   no other row has; "belief" rows have `believed_on_true` below the request by more than it;
   "choice" rows do not; the deficits add up to `method_high - believed_on_true`; in a "choice"
   row the shortfall is `overplanned` plus the request minus the planned total, and that last
   term is within the tolerance (goal 1 kept). Rows of "state" and "choice" occur, so the test is
   not empty. The goal test of each "choice" row gets as each car's true take its upper end from
   the method's own state, recomputed here from the method's replay; on some rows that differs
   from the upper end from the reference's state. The day's goal-test counts are the sums of the
   outcomes the goal tests returned, over every path.
"""

import hashlib
from collections import Counter

import numpy as np
import pytest
from compare_fake import offer_days

from fleetreserve import asa_diagnosis_day, compare_day, compare_methods
from fleetreserve.asa_diagnosis_day import CAUSES, TOLERANCE, deliverable, diagnose_day
from fleetreserve.compare_methods import MethodInput
from fleetreserve.controller import forecast_belief, hour_zero_target
from fleetreserve.controller_replay import run_controller
from fleetreserve.lp import OTHER_SECONDS
from fleetreserve.plans import Plan

PLANS_BEFORE = "ac6b31f7d3ea705c0a193fc7050ad4b50512f850fb621332e821a913ee4caceb"


def _replay(prepared, beta, method, zeta):
    policy = prepared.certificate.policy
    plan = Plan.from_policy(policy, beta)
    controller = method.build(MethodInput(prepared.day, policy, beta, plan))
    run = run_controller(
        prepared.day,
        controller,
        forecast_belief(prepared.day),
        zeta,
        policy.request(zeta, beta),
        hour_zero_target=hour_zero_target(plan),
        departure_seen=prepared.departure_seen,
    )
    return run, controller


# --- 1. plans unchanged --------------------------------------------------------------------------


def test_the_plans_of_both_forms_are_those_before_the_record():
    methods = compare_methods.select(["asa", "asa-es"])
    digest = hashlib.sha256()
    for prepared in offer_days(8):
        for beta in (0.3, 1.0):
            for zeta in prepared.paths:
                for method in methods:
                    run, _ = _replay(prepared, beta, method, zeta)
                    digest.update(np.ascontiguousarray(run.planned, dtype=np.float64).tobytes())
    assert digest.hexdigest() == PLANS_BEFORE


# --- 2. the record -------------------------------------------------------------------------------


def test_the_record_holds_one_plan_per_hour_and_the_powers_returned():
    for method in compare_methods.select(["asa", "asa-es"]):
        prepared = offer_days(2)[1]
        run, controller = _replay(prepared, 1.0, method, prepared.paths[3])
        assert len(controller.hour_plans) == prepared.day.n_service + 1 == controller.hours
        for hour, plan in enumerate(controller.hour_plans):
            assert plan.optimal
            np.testing.assert_array_equal(plan.powers[:, 0], run.planned[:, hour])


# --- 3. the replays are compare_day's ------------------------------------------------------------


def test_the_replays_are_those_of_compare_day():
    method, reference = compare_methods.select(["asa-es", "llf"])
    for prepared in offer_days(6):
        for beta in (0.3, 1.0):
            found = diagnose_day(prepared, beta, method, reference, time_limit=OTHER_SECONDS)
            expected = compare_day.evaluate(prepared, beta, (method, reference))
            for k, m in enumerate((method, reference)):
                runs = [pair[k] for pair in found.runs]
                assert max(r.slot_loss for r in runs) == expected[m.name].slot_loss
                assert max(r.driver_loss for r in runs) == expected[m.name].driver_loss
                assert tuple(r.first_failure for r in runs) == expected[m.name].first_failures
                for zeta, run in zip(prepared.paths, runs, strict=True):
                    direct, _ = _replay(prepared, beta, m, zeta)
                    np.testing.assert_array_equal(run.planned, direct.planned)
            assert found.method_solver_counts == expected[method.name].solver_counts


@pytest.mark.parametrize(
    "reference",
    [m for m in compare_methods.METHODS if m.kind == "controller"],
    ids=lambda m: m.name,
)
def test_every_controller_method_of_the_registry_can_be_the_reference(reference):
    # The inputs carry the rows the day's box was certified on, as compare_day's do, so a method
    # that needs them builds; its losses are those compare_day.evaluate gives.
    (method,) = compare_methods.select(["asa-es"])
    for prepared in offer_days(3):
        found = diagnose_day(prepared, 1.0, method, reference, time_limit=OTHER_SECONDS)
        expected = compare_day.evaluate(prepared, 1.0, (reference,))[reference.name]
        runs = [pair[1] for pair in found.runs]
        assert max(r.slot_loss for r in runs) == expected.slot_loss
        assert tuple(r.first_failure for r in runs) == expected.first_failures


# --- 4. the causes partition the method-only hours -----------------------------------------------


def _rows_and_runs(found):
    for row in found.rows:
        yield row, found.runs[row.path][0]


@pytest.mark.parametrize("name", ["asa-es", "asa"])
def test_the_causes_partition_the_method_only_hours(name):
    method, reference = compare_methods.select([name, "llf"])
    seen = dict.fromkeys(CAUSES, 0)
    for prepared in offer_days(15):
        for beta in (0.3, 0.6, 1.0):
            found = diagnose_day(prepared, beta, method, reference, time_limit=OTHER_SECONDS)
            total = len(prepared.paths) * prepared.day.n_service
            assert sum(found.groups.values()) == total
            assert len(found.rows) == found.groups["method_only"]
            for row, run in _rows_and_runs(found):
                assert row.cause in CAUSES
                seen[row.cause] += 1
                above = row.request > row.method_high + TOLERANCE
                assert above == (row.cause == "state")
                short = row.method_delivered < row.request - TOLERANCE
                believed = row.believed_on_true < row.request - TOLERANCE
                if row.cause == "belief":
                    assert short and believed
                if row.cause == "choice":
                    assert short and not believed and row.goal
                    total = float(run.planned[:, row.hour].sum())
                    assert row.shortfall == pytest.approx(
                        row.request - total + row.overplanned, abs=1e-9
                    )
                    assert abs(row.request - total) <= TOLERANCE + 1e-7
                deficits = (
                    row.deficit_plugged + row.deficit_departure + row.deficit_rate + row.deficit_cap
                )
                assert deficits == pytest.approx(row.method_high - row.believed_on_true, abs=1e-9)
    assert seen["state"] > 0 and seen["choice"] > 0
    assert sum(seen.values()) > 0


def test_the_goal_test_takes_the_upper_ends_from_the_method_s_own_state(monkeypatch):
    calls = []
    outcomes = Counter()
    real = asa_diagnosis_day.forbidding_goal

    def spy(method, view, plan, take, request, **options):
        calls.append(np.array(take, dtype=float))
        found = real(method, view, plan, take, request, **options)
        outcomes.update(found[1])
        return found

    monkeypatch.setattr(asa_diagnosis_day, "forbidding_goal", spy)
    method, reference = compare_methods.select(["asa-es", "llf"])
    differ = paths = 0
    for prepared in offer_days(15):
        for beta in (0.3, 0.6, 1.0):
            calls.clear()
            outcomes.clear()
            found = diagnose_day(prepared, beta, method, reference, time_limit=OTHER_SECONDS)
            choice = [row for row in found.rows if row.cause == "choice"]
            assert len(calls) == len(choice)
            assert found.goal_tests == outcomes
            paths += len({row.path for row in choice}) > 1
            for row, take in zip(choice, calls, strict=True):
                ours, theirs = found.runs[row.path]
                _, high_m = deliverable(prepared.day, ours.stored[:, row.hour - 1], row.hour)
                _, high_r = deliverable(prepared.day, theirs.stored[:, row.hour - 1], row.hour)
                np.testing.assert_array_equal(take, high_m)
                differ += not np.array_equal(high_m, high_r)
    assert differ > 0 and paths > 0
