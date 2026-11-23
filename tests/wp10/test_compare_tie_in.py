"""The comparison's per-day work against the rungs of `gate_day.evaluate` (tie-in A).

On seeded made-up offer days (`compare_fake.offer_days`: 2 to 4 service hours, 2 to 5 cars, both
readings of `departure_seen`), at every scale of `dial_splits.GRID`, the `fixed-plan` method of
`compare_day.evaluate` has exactly the slot loss, driver loss and first failing hour per path of
the "fixed" rung of `gate_day.evaluate` on the same prepared day. Exact equality: both run the
same plan on the same requests and paths through the same replay arithmetic. The sample must hold
scales with a loss above 0, so the equality is not met by zeros alone.

Every controller method's row is the largest loss over the audited paths of its own
`controller_replay.run_controller` run with the shared inputs of readings C1 and C5 of
`compare_run` (the certified box's requests at the scale, the prepared day's forecast belief, the
fixed plan's hour-zero target, the prepared day's `departure_seen`) and the rows the box was
certified on (`PreparedDay.rows`, which `diagonal-split` splits the box on), on days where the
methods differ and where the other reading of `departure_seen` would give other rows.
"""

import dataclasses

from compare_fake import offer_days

from fleetreserve import compare_day, compare_methods, dial_splits, gate_day
from fleetreserve.compare_methods import MethodInput
from fleetreserve.controller import forecast_belief, hour_zero_target
from fleetreserve.controller_replay import run_controller
from fleetreserve.plans import Plan
from fleetreserve.rungs import DEPARTURE_SEEN

DAYS = offer_days(10)
FIXED = compare_methods.select(["fixed-plan"])
CONTROLLERS = tuple(m for m in compare_methods.select(None) if m.kind == "controller")


def test_the_fixed_plan_method_equals_the_fixed_rung_at_every_scale():
    slot_failures = driver_failures = 0
    for prepared in DAYS:
        for scale in dial_splits.GRID:
            rung = gate_day.evaluate(prepared, scale).rungs["fixed"]
            found = compare_day.evaluate(prepared, scale, FIXED)["fixed-plan"]
            assert found.slot_loss == rung.slot_loss, (prepared.date, scale)
            assert found.driver_loss == rung.driver_loss, (prepared.date, scale)
            assert found.first_failures == rung.first_failures, (prepared.date, scale)
            slot_failures += found.slot_loss > 0
            driver_failures += found.driver_loss > 0
    assert slot_failures >= 10, "the sample must hold failing day-scales"
    assert driver_failures >= 5, "the sample must hold day-scales with a driver loss"


def test_every_controller_runs_on_the_shared_box_paths_belief_and_target():
    differs = readings_differ = 0
    for prepared in DAYS:
        policy = prepared.certificate.policy
        other = dataclasses.replace(
            prepared, departure_seen=next(s for s in DEPARTURE_SEEN if s != prepared.departure_seen)
        )
        for scale in (0.3, 0.6, 1.0):
            plan = Plan.from_policy(policy, scale)
            inputs = MethodInput(prepared.day, policy, scale, plan, prepared.rows)
            found = compare_day.evaluate(prepared, scale, CONTROLLERS)
            for method in CONTROLLERS:
                runs = [
                    run_controller(
                        prepared.day,
                        method.build(inputs),
                        forecast_belief(prepared.day),
                        zeta,
                        policy.request(zeta, scale),
                        hour_zero_target=hour_zero_target(plan),
                        departure_seen=prepared.departure_seen,
                    )
                    for zeta in prepared.paths
                ]
                assert found[method.name].slot_loss == max(r.slot_loss for r in runs)
                assert found[method.name].driver_loss == max(r.driver_loss for r in runs)
                assert found[method.name].first_failures == tuple(r.first_failure for r in runs)
            differs += len({(d.slot_loss, d.driver_loss) for d in found.values()}) > 1
            readings_differ += compare_day.evaluate(other, scale, CONTROLLERS) != found
    assert differs >= 2, "the methods must differ on some day-scales"
    assert readings_differ >= 1, "the two readings of departure_seen must differ on some day-scale"
