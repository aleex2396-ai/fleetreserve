"""Maximum-entropy feedback (`max-entropy-feedback`, `mef_controller`; plan section 11 O7) in the
corner search: its corners on a hand day worked by hand, and its deviation counts kept on the
corners as `asa`'s solver counts are.

The hand day (`DAY`): one service hour, two recovery hours, rho 0; cars A (rate 1 kW, room 1 kWh)
and B (rate 1 kW, room 2 kWh) both truly leave in hour 4. Their forecasts equal the truth but for
B's need, 0.5 kWh (its true protected energy is 2). The box (`POLICY`): m = 1.6, w = 0.4, so at
scale beta the requests are 1.6 - 0.4 beta and 1.6 + 0.4 beta. The fixed plan gives B 1 kWh in
hour 0 and 1 in hour 1, and A 0 in hour 0 and `0.6 + 0.4 beta zeta_1` in hour 1: the hour-zero
target (`controller.hour_zero_target`) is 1.

By hand, at any scale 0 < beta <= 1 (checked at 0.5 and 1):
- Hour 0. Least-laxity-first's laxities are A: (4 - 0) - 1 / 1 = 3 and B: 4 - 0.5 / 1 = 3.5, so it
  gives the whole target to A, which is then full; in hour 1 only B could draw, at most 1 kWh,
  below the lowest request 1.6 - 0.4 beta (at least 1.2): the believed future is not met
  (`mef_future.roomy` fails). A total u <= 1 split to A alone leaves room (1 - u) + 1 in hour 1,
  so the lowest request stays feasible exactly when u <= 0.4 + 0.4 beta; a total above 1 leaves
  B alone, never enough. Both believed needs are met by the two recovery hours at 1 kW, so every
  total keeps the needs (`enough` holds at 0). The controller plans A 0.4 + 0.4 beta, B 0: a
  deviating hour, not a service hour (0.8 at scale 1; to within the 1e-9 kWh of its bisection).
- Hour 1. A can take 1 - (0.4 + 0.4 beta) = 0.6 - 0.4 beta more and B 1, together exactly the
  lowest request. On the corner (-1) both are filled and the request is met; on the corner (+1)
  the request 1.6 + 0.4 beta is 0.8 beta above what they can take: hour 1 fails, slot loss 1. No
  remaining hour follows, so the future is met and the hour does not deviate.
- Recovery fills both cars, so no corner loses driver energy.
So the method breaks on its upper corner only, with deviation counts per corner of hours 2
(hours 0 and 1), deviating hours 1, deviating service hours 0, hours with no feasible total 0:
summed over the two corners 4, 2, 0, 0. Least-laxity-first without the guard leaves A full and B
at most 1 kWh in hour 1 on both corners: it breaks on both. The fixed plan meets both.
"""

import dataclasses
import datetime

import pytest
from corner_fake import hand, offer_days

from fleetreserve import compare_day, compare_methods, corner_run, corner_search, corner_summary
from fleetreserve.compare_methods import MethodInput
from fleetreserve.controller import forecast_belief, hour_zero_target
from fleetreserve.controller_replay import run_controller
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.plans import Plan
from fleetreserve.policy import AffinePolicy
from fleetreserve.products import corner_paths

MEF = compare_methods.select(["max-entropy-feedback"])[0]
LLF = compare_methods.select(["llf"])[0]
FIXED = compare_methods.select(["fixed-plan"])[0]
DAY = Day(
    "hand",
    datetime.date(2020, 1, 1),
    1,
    2,
    0.0,
    (TrueCar(4, 1.0, 1.0), TrueCar(4, 1.0, 2.0)),
    (ForecastCar(4.0, 1.0, 1.0, 1.0), ForecastCar(4.0, 1.0, 2.0, 0.5)),
    0.1,
)
POLICY = AffinePolicy([1.6], [0.4], [[0.0, 0.6], [1.0, 1.0]], [[[0.0], [0.4]], [[0.0], [0.0]]])
PREPARED = hand(DAY, POLICY)
PER_CORNER = {
    "hours": 2,
    "deviating_hours": 1,
    "deviating_service_hours": 0,
    "no_feasible_hours": 0,
}
SUMMED = {"hours": 4, "deviating_hours": 2, "deviating_service_hours": 0, "no_feasible_hours": 0}


def _mef_runs(beta):
    """The controller run directly on each corner, with its counts (not through the search)."""
    plan = Plan.from_policy(POLICY, beta)
    inputs = MethodInput(DAY, POLICY, beta, plan, PREPARED.rows)
    found = []
    for zeta in corner_paths(1):
        controller = MEF.build(inputs)
        run = run_controller(
            DAY,
            controller,
            forecast_belief(DAY),
            zeta,
            POLICY.request(zeta, beta),
            hour_zero_target=hour_zero_target(plan),
            departure_seen=PREPARED.departure_seen,
        )
        found.append((run, controller.deviation_counts()))
    return found


@pytest.mark.parametrize("beta", (0.5, 1.0))
def test_mef_on_the_corners_of_the_hand_day(beta):
    assert hour_zero_target(Plan.from_policy(POLICY, beta)) == 1.0
    for run, counts in _mef_runs(beta):
        assert run.draws[:, 0] == pytest.approx([0.4 + 0.4 * beta, 0.0], abs=1e-8)
        assert counts == PER_CORNER
    found = corner_search.search_day(PREPARED, MEF, beta)
    assert found.status == "breaks" and found.n_paths == 2
    low, high = found.paths
    assert low.first_failure is None and low.slot_loss == 0.0
    assert high.first_failure == 1 and high.slot_loss == 1.0
    assert low.driver_loss == high.driver_loss == 0.0
    assert found.worst_path == (1,) and found.breaking_paths == 1
    llf = corner_search.search_day(PREPARED, LLF, beta)
    assert [p.first_failure for p in llf.paths] == [1, 1]
    fixed = corner_search.search_day(PREPARED, FIXED, beta)
    assert fixed.status == "holds"


@pytest.mark.parametrize("beta", (0.5, 1.0))
def test_the_corner_search_keeps_mefs_deviation_counts_summed_over_every_corner(beta):
    found = corner_search.search_day(PREPARED, MEF, beta)
    assert found.deviation_counts == SUMMED
    for method in (LLF, FIXED):  # their controllers keep no deviation counts
        assert corner_search.search_day(PREPARED, method, beta).deviation_counts is None
    assert corner_search.search_day(PREPARED, MEF, 0.0).deviation_counts is None  # no offer


def test_on_the_audited_paths_the_counts_are_compare_days():
    days = [PREPARED, *offer_days(6, seed=11)]
    for prepared in days:
        for beta in (0.4, 1.0):
            direct = compare_day.evaluate(prepared, beta, (MEF,))[MEF.name]
            _, _, _, counts = corner_search.path_results(prepared, MEF, beta, prepared.paths)
            assert counts == direct.deviation_counts
    # the hand day's eight audited rows hold the corner (-1) twice and (+1) six times
    assert compare_day.evaluate(PREPARED, 1.0, (MEF,))[MEF.name].deviation_counts == {
        "hours": 16,
        "deviating_hours": 8,
        "deviating_service_hours": 0,
        "no_feasible_hours": 0,
    }


def _day(counts):
    """A searched `CornerDay` that holds, with the deviation counts `counts`."""
    base = corner_search.search_day(PREPARED, FIXED, 1.0)
    return dataclasses.replace(base, deviation_counts=counts)


def test_the_method_entry_sums_the_deviation_counts_over_the_searched_days():
    second = {
        "hours": 6,
        "deviating_hours": 1,
        "deviating_service_hours": 1,
        "no_feasible_hours": 2,
    }
    entry = corner_summary.method_entry(1.0, "", [_day(SUMMED), _day(second)])
    assert entry["deviation_counts"] == {
        "hours": 10,
        "deviating_hours": 3,
        "deviating_service_hours": 1,
        "no_feasible_hours": 2,
    }
    # a method whose days keep no deviation counts writes no such field, nor does no offer
    assert "deviation_counts" not in corner_summary.method_entry(1.0, "", [_day(None)] * 2)
    assert "deviation_counts" not in corner_summary.method_entry(0.0, "scale 0", [_day(None)])


def test_the_definition_is_written_only_with_a_method_that_keeps_deviation_counts():
    matrix = type("Matrix", (), {"n_calibration": 100, "calibration_half_below": 200})()
    without = corner_run.definitions(matrix)
    assert "method.deviation_counts" not in without["fields"]
    found = corner_run.definitions(matrix, deviating=["max-entropy-feedback"])
    assert found["fields"]["method.deviation_counts"] == (
        corner_summary.DEVIATION_DEFINITION + " (max-entropy-feedback)"
    )
    assert set(found["fields"]) - set(without["fields"]) == {"method.deviation_counts"}


def test_the_summary_gives_one_line_per_method_that_keeps_deviation_counts():
    found = {
        "configuration": "made up",
        "days_searched": 2,
        "offering_test_days": 2,
        "days_asked": 2,
        "methods": {
            MEF.name: corner_summary.method_entry(1.0, "", [_day(SUMMED)]),
            LLF.name: corner_summary.method_entry(1.0, "", [_day(None)]),
        },
    }
    document = {"methods": [MEF.name, LLF.name]}
    lines = corner_summary._variant_lines(document, "forecast", found)
    assert lines[-1] == (
        "  max-entropy-feedback on the corners: hours planned 4, with least-laxity-first's split "
        "found infeasible and the total nearest the target found feasible planned 2 (in service "
        "hours 0), with no feasible total 0"
    )
    del found["methods"][MEF.name]["deviation_counts"]
    assert not any(
        "on the corners: hours planned" in line
        for line in corner_summary._variant_lines(document, "forecast", found)
    )
