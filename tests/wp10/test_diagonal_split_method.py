"""The method `diagonal-split` of `compare_methods`, its rows, its time limit and its counts.

1. The registry lists `diagonal-split` right after `asa-es`, a controller built by
   `compare_methods.diagonal_split`.
2. Without rows (`MethodInput.rows` None, the default) the builder refuses with a `MethodError`
   that names the method; `compare_day.evaluate` of a prepared day without rows refuses the same
   way. The other methods still build from a `MethodInput` without rows.
3. `gate_day.prepare_day` keeps the rows its certificate was certified on: the forecast rows of
   the steered sub-day without the dial, the driver-risk rows at the dial with it (certifying them
   again gives the same value); a day that abstains before any rows exist (no steered car) has
   `rows` None, and one that abstains after its certificate (a value at most the zero value)
   keeps them.
4. The time limit reaching the solver: the builder's one solve asks for `lp.BOX_SECONDS` (30 s,
   plan section 13 for a box), and `split_box`'s one solve asks for the time limit it is given.
5. A programme that times out: kind "no programme", no plan; the controller plans every hour by
   `priority_rules.least_laxity_first` of the same view and counts the hour as planned by the
   fall-back; its counts say one timeout and one box "no programme".
6. Counted once per day and scale: through `compare_day.evaluate` the method's summed counts hold
   one solve (optimal) and one box per (day, scale) pair, the hours of every audited path, none by
   the fall-back. Building twice from the same `MethodInput` object solves once (the second
   controller counts nothing); from a new object with the same fields it solves again.
7. `compare_day.sum_counts` sums the boxes over the paths when the controllers count them and
   writes no `boxes` field when none does (the counts of `asa`).
8. The summary's line of controller solves gives `diagonal-split`'s boxes after its hours, the
   outcomes in the order of `lp.Outcome` and the kinds in the order of `diagonal_split.BOXES`,
   so the line is the same from the counts as the controllers give them and from `compare.json`
   read back (keys sorted); the definitions entry `diagonal_split_boxes` is written only when
   `diagonal-split` reported counts.
"""

import dataclasses
import json
import types

import numpy as np
import pytest
from compare_fake import offer_days
from test_driver_risk import prepare, stub
from test_gate_day import CARS, DATE, SETTINGS, Stub, configuration, truth_day

from fleetreserve import (
    compare_day,
    compare_methods,
    compare_run,
    compare_summary,
    diagonal_split,
    driver_risk,
    gate_day,
    products,
    sparse_programme,
)
from fleetreserve.carsets import day_forecast_rows
from fleetreserve.certificate import certify
from fleetreserve.compare_methods import MethodError, MethodInput
from fleetreserve.controller import forecast_belief, hour_zero_target
from fleetreserve.controller_replay import run_controller
from fleetreserve.day import Day
from fleetreserve.lp import BOX_SECONDS, LpResult
from fleetreserve.plans import Plan
from fleetreserve.priority_rules import least_laxity_first

NAME = "diagonal-split"


def _inputs(prepared, beta=0.5, rows=True):
    policy = prepared.certificate.policy
    return MethodInput(
        prepared.day,
        policy,
        beta,
        Plan.from_policy(policy, beta),
        prepared.rows if rows else None,
    )


def test_the_registry_lists_the_method_after_asa_es():
    names = [m.name for m in compare_methods.METHODS]
    assert names.index(NAME) == names.index("asa-es") + 1
    (method,) = compare_methods.select([NAME])
    assert method.kind == "controller" and method.build is compare_methods.diagonal_split


def test_without_rows_the_method_refuses_and_names_itself():
    prepared = offer_days(1)[0]
    with pytest.raises(MethodError, match="'diagonal-split'.*MethodInput.rows"):
        compare_methods.diagonal_split(_inputs(prepared, rows=False))
    without = dataclasses.replace(prepared, rows=None)
    with pytest.raises(MethodError, match="'diagonal-split'"):
        compare_day.evaluate(without, 0.5, compare_methods.select([NAME]))
    for method in compare_methods.METHODS:
        if method.kind == "controller" and method.name != NAME:
            method.build(_inputs(prepared, rows=False))


def test_prepare_day_keeps_the_rows_it_certified_on():
    plain = prepare(stub(), driver_risk=None)
    np.testing.assert_array_equal(plain.rows.bounds, day_forecast_rows(plain.day).bounds)
    np.testing.assert_array_equal(
        plain.rows.coefficients, day_forecast_rows(plain.day).coefficients
    )
    dial = prepare(stub(), driver_risk=0.02)
    expected = driver_risk.day_rows(dial.day, 0.02)
    assert dial.rows.n_rows == expected.n_rows > plain.rows.n_rows
    np.testing.assert_array_equal(dial.rows.bounds, expected.bounds)
    np.testing.assert_array_equal(dial.rows.coefficients, expected.coefficients)
    again = certify(
        dial.rows,
        products.hourly(),
        "causal",
        shape_rule=gate_day.shape_rule(configuration(), SETTINGS),
    )
    assert again.value == dial.value


def test_a_day_that_abstains_before_rows_exist_has_none():
    truth = truth_day(Day("u", DATE, 2, 3, 0.0, CARS))
    abstains = gate_day.prepare_day(truth, Stub([None] * len(CARS)), configuration(), SETTINGS)
    assert abstains.abstain_reason == gate_day.NO_STEERED
    assert abstains.rows is None


def test_a_day_that_abstains_after_its_certificate_keeps_its_rows():
    high = dataclasses.replace(SETTINGS, zero_value_kwh=1e9)
    truth = truth_day(Day("u", DATE, 2, 3, 0.0, CARS))
    abstains = gate_day.prepare_day(truth, stub_of_two(), configuration(), high)
    assert abstains.abstain_reason == gate_day.ZERO_VALUE
    np.testing.assert_array_equal(abstains.rows.bounds, day_forecast_rows(abstains.day).bounds)


def stub_of_two():
    """Forecasts for the first and third car only (`test_gate_day.FORECASTS`)."""
    from test_gate_day import FORECASTS

    return Stub([FORECASTS[0], None, FORECASTS[2], None])


def test_the_box_time_limit_reaches_the_solver(monkeypatch):
    asked = []
    real = sparse_programme.solve_lp

    def recording(*args, **kwargs):
        asked.append(kwargs["time_limit"])
        return real(*args, **kwargs)

    inputs = _inputs(offer_days(1)[0])  # made before the solver is watched
    monkeypatch.setattr(sparse_programme, "solve_lp", recording)
    compare_methods.diagonal_split(inputs)
    assert asked == [BOX_SECONDS] and BOX_SECONDS == 30.0
    policy = inputs.policy
    diagonal_split.split_box(inputs.rows, policy.m, policy.w, time_limit=7.5)
    assert asked == [BOX_SECONDS, 7.5]


def _view_run(prepared, controller, beta=0.5):
    policy = prepared.certificate.policy
    zeta = prepared.paths[0]
    return run_controller(
        prepared.day,
        controller,
        forecast_belief(prepared.day),
        zeta,
        policy.request(zeta, beta),
        hour_zero_target=hour_zero_target(Plan.from_policy(policy, beta)),
        departure_seen=prepared.departure_seen,
    )


def test_a_timed_out_programme_plans_every_hour_by_the_least_laxity_fall_back(monkeypatch):
    def timeout(*args, **kwargs):
        return LpResult("timeout", None, None, 0.0, kwargs["time_limit"], 1, "made up")

    prepared = offer_days(1)[0]  # made before the solver is replaced
    monkeypatch.setattr(sparse_programme, "solve_lp", timeout)
    controller = compare_methods.diagonal_split(_inputs(prepared))
    assert controller.split.kind == diagonal_split.NO_PROGRAMME
    assert controller.split.plan is None and controller.split.largest_scale is None
    seen = []

    def watching(view):
        planned = controller(view)
        seen.append((planned, least_laxity_first(view)))
        return planned

    _view_run(prepared, watching)
    n = prepared.day.n_service
    assert len(seen) == n + 1
    assert all(np.array_equal(a, b) for a, b in seen)
    counts = controller.solver_counts()
    assert counts["hours"] == counts["fallback_hours"] == n + 1
    assert counts["solves"] == {o: int(o == "timeout") for o in diagonal_split.OUTCOMES}
    assert counts["boxes"] == {k: int(k == "no programme") for k in diagonal_split.BOXES}


def test_a_day_and_scale_is_solved_and_counted_once_over_its_paths():
    method = compare_methods.select([NAME])
    for prepared in offer_days(4):
        for beta in (0.3, 1.0):
            found = compare_day.evaluate(prepared, beta, method)[NAME].solver_counts
            paths = prepared.paths.shape[0]
            assert found["solves"] == {o: int(o == "optimal") for o in diagonal_split.OUTCOMES}
            assert sum(found["boxes"].values()) == 1
            assert found["hours"] == paths * (prepared.day.n_service + 1)
            assert found["fallback_hours"] == 0


def test_the_same_inputs_object_solves_once_and_a_new_one_solves_again(monkeypatch):
    solves = []
    real = sparse_programme.solve_lp

    def counting(*args, **kwargs):
        solves.append(1)
        return real(*args, **kwargs)

    inputs = _inputs(offer_days(1)[0])  # made before the solver is watched
    monkeypatch.setattr(sparse_programme, "solve_lp", counting)
    first = compare_methods.diagonal_split(inputs)
    second = compare_methods.diagonal_split(inputs)
    assert len(solves) == 1 and second.split is first.split
    assert first.counted and not second.counted
    assert sum(second.solver_counts()["solves"].values()) == 0
    assert sum(second.solver_counts()["boxes"].values()) == 0
    third = compare_methods.diagonal_split(dataclasses.replace(inputs))
    assert len(solves) == 2 and third.counted


def test_sum_counts_sums_boxes_only_when_they_are_counted():
    one = {"solves": {"optimal": 1}, "hours": 3, "fallback_hours": 0, "boxes": {"split": 1}}
    two = {"solves": {"optimal": 0}, "hours": 3, "fallback_hours": 1, "boxes": {"split": 0}}
    assert compare_day.sum_counts([one, two]) == {
        "solves": {"optimal": 1},
        "hours": 6,
        "fallback_hours": 1,
        "boxes": {"split": 1},
    }
    plain = [{k: v for k, v in c.items() if k != "boxes"} for c in (one, two)]
    assert "boxes" not in compare_day.sum_counts(plain)


def _document(counts_by_unit):
    units = {
        unit: {"variants": {"forecast": {"evaluations": {"controller_solver_outcomes": found}}}}
        for unit, found in counts_by_unit.items()
    }
    return {"units": units, "methods": ["asa", NAME]}


def test_the_summary_line_gives_the_boxes_of_the_method():
    boxes = {"split": 3, "stretched": 1, "no gains": 0, "no programme": 0}
    diagonal = {"solves": {"optimal": 4}, "hours": 12, "fallback_hours": 0, "boxes": boxes}
    asa = {"solves": {"optimal": 8}, "hours": 2, "fallback_hours": 0}
    document = _document({"u": {"asa": asa, NAME: diagonal}, "v": {NAME: diagonal}})
    (line,) = compare_summary._solver_line(document)
    assert line == (
        "Controller solves (all units and variants together): asa: solves optimal 8; hours "
        "planned 2, by the fall-back 0; diagonal-split: solves optimal 8; hours planned 24, by "
        "the fall-back 0; boxes split 6, stretched 2, no gains 0, no programme 0."
    )


def test_the_summary_line_is_the_same_from_the_results_read_back_with_sorted_keys():
    counts = [
        diagonal_split.DiagonalSplitController(
            diagonal_split.DiagonalSplit(outcome, kind, None, None, 0.0), counted=True
        ).solver_counts()
        for outcome, kind in (("optimal", "stretched"), ("timeout", "no programme"))
    ]
    document = _document({"u": {NAME: counts[0]}, "v": {NAME: counts[1]}})
    read_back = json.loads(json.dumps(document, sort_keys=True))
    assert list(
        read_back["units"]["u"]["variants"]["forecast"]["evaluations"][
            "controller_solver_outcomes"
        ][NAME]["boxes"]
    ) != list(diagonal_split.BOXES)
    (line,) = compare_summary._solver_line(document)
    assert compare_summary._solver_line(read_back) == [line]
    assert line.endswith(
        "diagonal-split: solves optimal 1, infeasible 0, unbounded 0, timeout 1, error 0; hours "
        "planned 0, by the fall-back 0; boxes split 0, stretched 1, no gains 0, no programme 1."
    )


def test_the_boxes_definition_is_written_only_with_the_method():
    matrix = types.SimpleNamespace(n_calibration=2, calibration_half_below=None)
    assert "diagonal_split_boxes" in compare_run.definitions(matrix, 4, [NAME])
    assert "diagonal_split_boxes" not in compare_run.definitions(matrix, 4, ["asa"])
    assert "diagonal_split_boxes" not in compare_run.definitions(matrix, 4)
