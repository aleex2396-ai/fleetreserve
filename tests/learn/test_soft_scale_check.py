"""`soft_scale_check` on made-up gate units (`exercising_units` below, written with three service
hours) and a made-up verified truth build, with the empirical model and the gate matrix's
configuration `main-rounded-hour-k0.1-empirical`.

The units. Three cars a day as `gate_fake.gate_cars` makes them (plugged in at 8:00, leaving about
4.4, 5.6 and 6.7 hours after the decision time, rates 2, 2.5 and 3 kW), with no early leaver and
with rooms that cycle over four patterns by day: one car small and two large, in each of the three
places, and two small and one large. A large room makes `p(A) > 0` (the departure and the need
move the lower terms); a small one makes the cap bind in `b(A)`. On `learn_fake.small_gate_units`
(rooms 1 to 2.4 kWh) every departure and need derivative of the checked days is exactly 0, so
those fields are not exercised there (test 3 shows the command fails on it); those units are not
changed.

1. Exit 0: the results and the summary are written; every unit has its checked days (at most the
   number asked), every checked day has its centre inside, a scale at least 1 (the certified box
   lies in the forecast set), no sandwich violation; no deciding comparison fails; the exact scale
   is written `tied` with nothing else exactly on the days whose smallest term is not unique, and
   the summary says how many those are; in each deciding field of each function some comparison
   is resolvable; the batch is timed on the CPU, and on MPS or marked "not available".
2. The scale the command reports for a prepared day equals the smallest largest step over the
   certified box's vertex directions, from programmes on the day's forecast rows
   (`soft_scale_fake.vertex_scale`), not from `soft_scale`.
3. Exit 1: with the gradient of the cap inside `b` halved (`soft_scale._smaller` given
   `(b + b.detach()) / 2`: the same values), the soft minimum's cap comparisons fail; with the
   gradient in one deciding field 1% too large everywhere (the field `x` read as
   `x + 0.01 (x - x.detach())`: the same values), that field's comparisons fail for both functions
   and no other field's do; with the gradient in the need NaN (a hook), its comparisons fail, are
   counted as not finite and written as null, and the results file is written; on
   `small_gate_units` the command says FIELD NOT EXERCISED for the departure and the need of both
   functions.
4. Exit 2: an unknown configuration, a unit that is not a gate unit, no verified truth build at
   the commit, and a sample in which no day offers.
5. The parts, on known inputs: `term_rounding` and `value_rounding` on a one-car day by hand;
   `compare` on a shape with no width (an infinite value: failed, written with nulls);
   `smallest_terms` at gaps of 1e-3 and 1e-12; `errors` counting outcomes and leaving tied
   comparisons out; the sandwich violation with `N` the number of finite terms; (the measuring
   rule's own parts: `test_soft_scale_measure.py`, `test_soft_scale_wrong_gradient.py`)
   `tracking` on known pairs and on a column constant up to rounding; the sample as the first
   offering days of the seeded order; every line of the summary of a made-up document, written
   out in full.
"""

import json
import math
import zlib
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pytest
import torch
from gate_fake import COMMIT, GATE_UNITS, matrix_document, write_build
from learn_fake import small_gate_units
from scipy import stats
from soft_scale_fake import car_rows, vertex_scale
from truth_tables_fake import FakeCar, date_of

from fleetreserve import gate_day, gate_run, run_verify, soft_scale, soft_scale_check
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import load

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net

NAME = "main-rounded-hour-k0.1-empirical"
NAMES = (NAME, "main-rounded-hour-k0.1-oracle")
DAYS = 6
SERVICE_HOURS = 3
ROOMS = ((1.0, 7.0, 9.0), (6.0, 1.5, 9.0), (6.0, 7.0, 2.0), (2.5, 3.0, 9.0))


def exercising_cars(n_days: int, site: str) -> list[FakeCar]:
    return [
        FakeCar(
            site,
            date_of(k),
            8.0,
            leave + 0.01 * (k % 7),
            rate=2.0 + 0.5 * j,
            room=ROOMS[k % len(ROOMS)][j] + 0.1 * (k % 4),
            session=f"{site}-{k}-{j}",
        )
        for k in range(n_days)
        for j, leave in enumerate((14.4, 15.6, 16.7))
    ]


def exercising_units() -> dict:
    return {
        GATE_UNITS[0]: exercising_cars(6, "A"),
        GATE_UNITS[1]: exercising_cars(24, "A"),
        GATE_UNITS[2]: exercising_cars(12, "B") + exercising_cars(12, "C"),
    }


def _build(tmp_path, monkeypatch, units) -> dict:
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    run, tables = write_build(tmp_path, units, SERVICE_HOURS)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    return {"arguments": arguments, "out": out, "tables": tables, "matrix": matrix}


@pytest.fixture
def made_up(tmp_path, monkeypatch):
    return _build(tmp_path, monkeypatch, exercising_units())


def _main(made_up, *extra, configuration=NAME):
    return soft_scale_check.main(
        ["--model", "empirical", "--configuration", configuration, "--days", str(DAYS)]
        + made_up["arguments"]
        + list(extra)
    )


def _document(made_up) -> dict:
    return json.loads((made_up["out"] / soft_scale_check.RESULTS_NAME).read_text())


def test_the_check_runs_end_to_end_and_passes(made_up, capsys):
    code = _main(made_up)
    printed = capsys.readouterr().out
    assert code == 0, printed
    document = _document(made_up)
    assert (made_up["out"] / soft_scale_check.SUMMARY_NAME).read_text() == printed
    assert document["passed"] and document["failed_comparisons"] == 0
    assert set(document["units"]) == set(GATE_UNITS)
    total = 0
    for unit in document["units"].values():
        assert 0 < unit["checked_days"] <= DAYS
        assert len(unit["days"]) == unit["checked_days"]
        assert unit["fewer_than_asked"] == (unit["checked_days"] < DAYS)
        for day in unit["days"]:
            total += 1
            assert day["n_service"] == 3
            assert day["centre_inside"]
            assert day["scale"] >= 1 - 1e-6
            assert day["sandwich_violation"] <= 1e-12
            scaled = day["scale"] * day["certified_value_kwh"]
            assert day["scaled_value_kwh"] == pytest.approx(scaled)
            for field in soft_scale_check.FIELDS:
                soft = day["comparisons"]["soft"][field]
                exact = day["comparisons"]["exact"][field]
                assert soft["outcome"] in ("passed", "unresolved", "kink"), soft
                if day["smallest_terms"] > 1:
                    assert exact == {"outcome": "tied"}
                else:
                    assert exact["outcome"] in ("passed", "unresolved", "kink"), exact
                    assert (exact["decided_step"] is not None) == (exact["outcome"] == "passed")
    assert total >= 2 * DAYS
    assert document["not_exercised"] == []
    tied = sum(d["smallest_terms"] > 1 for u in document["units"].values() for d in u["days"])
    assert document["tied_days"] == tied < total
    assert f"Tied days: {tied} of {total} checked days" in printed
    for kind in soft_scale_check.KINDS:
        for field in soft_scale_check.DECIDING:
            found = document["errors"][kind][field]
            assert found["resolvable"] > 0, (kind, field)
            assert found["failed"] == found["non_finite"] == 0
            assert found["tied"] == (tied if kind == "exact" else 0)
    assert document["tracking"]["below_one"] == 0
    cpu = document["batch_timing"]["cpu"]
    assert cpu["device"] == "cpu" and cpu["dtype"] == "float64" and cpu["seconds"] > 0
    mps = document["batch_timing"]["mps"]
    if torch.backends.mps.is_available():
        assert mps["device"] == "mps" and mps["dtype"] == "float32"
    else:
        assert mps == "not available"
    assert "PASSED" in printed
    settings = document["settings"]
    assert settings["steps"] == {"exact": 1e-4, "soft": 1e-5} and settings["shrink"] == 100
    assert (settings["limit"], settings["resolution"], settings["sigma"]) == (1e-3, 0.25, 0.01)
    assert (settings["tie"], settings["fields"]) == (1e-9, ["departure", "cap", "need", "rate"])
    assert settings["deciding_fields"] == ["departure", "cap", "need"]
    assert settings["not_applicable"] == []
    assert "NOT APPLICABLE" not in printed


ORACLE = NAMES[1]
DECIDING_ALL = ("departure", "cap", "need")


def wide_cars(n_days: int, site: str) -> list[FakeCar]:
    """Eight cars a day, departures spread over the day, some inside the service hours (the
    review of 2078c32's 'wide design')."""
    return [
        FakeCar(
            site,
            date_of(k),
            8.0,
            11.2 + 0.8 * j + 0.13 * (k % 5) + 0.0017 * k,
            rate=2.0 + 0.4 * (j % 4),
            room=1.0 + 1.3 * ((j + k) % 6),
            session=f"{site}-{k}-{j}",
        )
        for k in range(n_days)
        for j in range(8)
    ]


def test_the_departure_is_not_applicable_with_whole_hour_departures(made_up, capsys):
    """The owner's decision of 15 November 2026 (plan section 7, T8's check): with whole-hour
    forecast departures (the oracle partner) the departure direction is reported not applicable
    instead of failing the check. Before it this run also named the departure of both functions
    FIELD NOT EXERCISED. On these made-up units the exact scale's cap is not exercised with the
    oracle departure either (every such comparison unresolved, as before the change), so the run
    still exits 1, for that field alone."""
    matrix = load(made_up["matrix"])
    assert soft_scale_check.whole_hour_departures(matrix.configuration(ORACLE))
    assert not soft_scale_check.whole_hour_departures(matrix.configuration(NAME))
    floored = replace(matrix.configuration(NAME), departure_floor="floored")
    assert soft_scale_check.deciding_fields(floored) == ("cap", "need")
    assert soft_scale_check.deciding_fields(matrix.configuration(NAME)) == DECIDING_ALL
    code = _main(made_up, configuration=ORACLE)
    printed = capsys.readouterr().out
    assert code == 1, printed
    document = _document(made_up)
    assert document["failed_comparisons"] == 0 and document["not_exercised"] == ["exact cap"]
    assert document["settings"]["deciding_fields"] == ["cap", "need"]
    assert document["settings"]["not_applicable"] == ["departure"]
    assert "NOT APPLICABLE: departure" in printed
    assert "NOT EXERCISED: soft departure" not in printed
    assert "NOT EXERCISED: exact departure" not in printed
    for kind in soft_scale_check.KINDS:
        found = document["errors"][kind]["departure"]
        assert not found["deciding"] and found["compared"] > 0
        assert found["resolvable"] == 0  # every point is a kink in the departure
    # the forecast departures this configuration checks are whole hours
    days = load_unit_days(
        made_up["tables"], GATE_UNITS[1], rho=0.0, no_fit="cap", small_fleet="keep"
    )
    forecaster = gate_run.empirical_forecaster(days, matrix)
    checked, _, _ = soft_scale_check.sample_unit(
        days, forecaster, matrix.configuration(ORACLE), gate_run.settings(matrix), 3, 1
    )
    departures = [d for _, p in checked for d in soft_scale.forecast_bounds(p.day).departure]
    assert departures and all(float(d) == math.floor(d) for d in departures)


def test_a_wrong_departure_gradient_does_not_fail_where_the_departure_is_not_applicable(
    made_up, monkeypatch
):
    _with_gradient(monkeypatch, "departure", lambda x: x + 0.5 * (x - x.detach()))
    assert _main(made_up, configuration=ORACLE) == 1  # exact cap not exercised, as unchanged
    document = _document(made_up)
    assert document["failed_comparisons"] == 0 and document["not_exercised"] == ["exact cap"]


def test_comparisons_at_whole_hour_kinks_are_never_failed(tmp_path, monkeypatch, capsys):
    """Finding 1 of the review of 2078c32: on the wide design with whole-hour departures (the
    oracle partner) comparisons whose differences show a kink at both steps were failed. Every
    one is now recorded `kink` (or decided where a step shows none); none is failed, in any
    field."""
    units = {
        GATE_UNITS[0]: wide_cars(10, "A"),
        GATE_UNITS[1]: wide_cars(30, "A"),
        GATE_UNITS[2]: wide_cars(15, "B") + wide_cars(15, "C"),
    }
    made = _build(tmp_path, monkeypatch, units)
    code = soft_scale_check.main(
        ["--model", "empirical", "--configuration", ORACLE, "--days", "12"] + made["arguments"]
    )
    printed = capsys.readouterr().out
    document = _document(made)
    outcomes = Counter(
        day["comparisons"][kind][field]["outcome"]
        for unit in document["units"].values()
        for day in unit["days"]
        for kind in soft_scale_check.KINDS
        for field in soft_scale_check.FIELDS
    )
    assert outcomes["failed"] == 0, (outcomes, printed)
    assert outcomes["kink"] > 0 and document["failed_comparisons"] == 0
    # the need is all kinks or unresolved on this design (as before the change): exit 1 for it
    assert code == 1 and document["not_exercised"] == ["soft need", "exact need"], printed


def test_an_error_after_the_run_could_not_run(made_up, monkeypatch, capsys):
    """Finding 4 of the review of 2078c32: an error after the run (writing the results) exited
    1, 'failed'; any error raised while the check runs is 2, 'could not run'."""
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(made_up["out"] / "missing"))
    assert _main(made_up) == 2
    assert "COULD NOT RUN: FileNotFoundError" in capsys.readouterr().out

    def broken(*args, **kwargs):
        raise RuntimeError("made to fail")

    monkeypatch.setattr(soft_scale_check, "run", broken)
    assert _main(made_up) == 2
    assert "COULD NOT RUN: RuntimeError: made to fail" in capsys.readouterr().out


def test_the_reported_scale_equals_the_vertex_programmes(made_up):
    matrix = load(made_up["matrix"])
    configuration = matrix.configuration(NAME)
    unit = GATE_UNITS[1]
    days = load_unit_days(made_up["tables"], unit, rho=0.0, no_fit="cap", small_fleet="keep")
    forecaster = gate_run.empirical_forecaster(days, matrix)
    checked, _, _ = soft_scale_check.sample_unit(
        days, forecaster, configuration, gate_run.settings(matrix), 4, 1
    )
    assert len(checked) == 4
    for row, prepared in checked:
        centre, up, down = (t.tolist() for t in soft_scale_check.day_shape(prepared))
        truth = vertex_scale(car_rows(prepared.day), centre, up, down)
        assert row["scale"] == pytest.approx(truth, rel=1e-6, abs=1e-9)


def test_a_wrong_gradient_fails_the_check(made_up, monkeypatch, capsys):
    original = soft_scale._smaller
    monkeypatch.setattr(soft_scale, "_smaller", lambda a, b: original(a, (b + b.detach()) / 2))
    code = _main(made_up)
    printed = capsys.readouterr().out
    assert code == 1, printed
    document = _document(made_up)
    assert not document["passed"] and document["failed_comparisons"] > 0
    assert document["errors"]["soft"]["cap"]["failed"] > 0
    assert "FAILED" in printed


def _with_gradient(monkeypatch, field, change):
    """`soft_scale.set_functions` with `field` of the bounds read as `change(field)`."""
    original = soft_scale.set_functions

    def changed(bounds, n_service):
        return original(replace(bounds, **{field: change(getattr(bounds, field))}), n_service)

    monkeypatch.setattr(soft_scale, "set_functions", changed)


def test_a_wrong_gradient_in_the_rate_alone_is_reported_and_does_not_fail(made_up, monkeypatch):
    _with_gradient(monkeypatch, "rate", lambda x: x + 0.01 * (x - x.detach()))
    assert _main(made_up) == 0
    document = _document(made_up)
    assert document["passed"] and document["failed_comparisons"] == 0
    for kind in soft_scale_check.KINDS:
        assert document["errors"][kind]["rate"]["failed"] > 0
        assert not document["errors"][kind]["rate"]["deciding"]


@pytest.mark.parametrize("field", ("departure", "cap", "need"))
def test_a_gradient_one_percent_too_large_in_a_deciding_field_fails(made_up, monkeypatch, field):
    _with_gradient(monkeypatch, field, lambda x: x + 0.01 * (x - x.detach()))
    assert _main(made_up) == 1
    document = _document(made_up)
    assert not document["passed"] and document["not_exercised"] == []
    for kind in soft_scale_check.KINDS:
        for other in soft_scale_check.FIELDS:
            failed = document["errors"][kind][other]["failed"]
            assert (failed > 0) if other == field else (failed == 0), (kind, other)


def _nan_gradient(x):
    if not x.requires_grad:
        return x
    y = x.clone()
    y.register_hook(lambda g: torch.full_like(g, float("nan")))
    return y


def test_a_derivative_that_is_not_finite_fails_and_is_written(made_up, monkeypatch, capsys):
    _with_gradient(monkeypatch, "need", _nan_gradient)
    assert _main(made_up) == 1, capsys.readouterr().out
    document = _document(made_up)
    found = document["errors"]["soft"]["need"]
    assert found["non_finite"] == found["compared"] == found["failed"] > 0
    day = document["units"][GATE_UNITS[1]]["days"][0]["comparisons"]["soft"]["need"]
    assert day["non_finite"] and day["autograd"] is None and day["outcome"] == "failed"
    assert document["errors"]["soft"]["cap"]["non_finite"] == 0


def test_fields_whose_derivatives_are_all_zero_are_not_exercised(tmp_path, monkeypatch, capsys):
    made = _build(tmp_path, monkeypatch, small_gate_units())
    code = _main(made)
    printed = capsys.readouterr().out
    assert code == 1, printed
    document = _document(made)
    assert document["failed_comparisons"] == 0 and not document["passed"]
    for kind in soft_scale_check.KINDS:
        for field in ("departure", "need"):
            assert f"{kind} {field}" in document["not_exercised"]
            assert f"FIELD NOT EXERCISED: {kind} {field}" in printed
    assert "soft cap" not in document["not_exercised"]
    days = [d for u in document["units"].values() for d in u["days"]]
    tied = [d for d in days if d["smallest_terms"] > 1]
    assert tied and document["tied_days"] == len(tied)
    for unit in document["units"].values():
        assert unit["tied_days"] == sum(d["smallest_terms"] > 1 for d in unit["days"])
    assert any(unit["tied_days"] for unit in document["units"].values())
    assert f"Tied days: {len(tied)} of {len(days)} checked days" in printed
    for day in tied:
        for field in soft_scale_check.FIELDS:
            assert day["comparisons"]["exact"][field] == {"outcome": "tied"}
            assert day["comparisons"]["soft"][field]["outcome"] != "tied"
    assert document["errors"]["exact"]["need"]["tied"] == len(tied)


def test_could_not_run(made_up, monkeypatch, capsys):
    assert _main(made_up, configuration="no-such-configuration") == 2
    assert _main(made_up, "--units", "not-a-gate-unit") == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
    assert _main(made_up, "--days", "0") == 2
    assert "COULD NOT RUN: --days must be at least 1" in capsys.readouterr().out
    arguments = list(made_up["arguments"])
    arguments[arguments.index("--truth-run") + 1] = str(made_up["out"])  # not a truth build
    code = soft_scale_check.main(
        ["--model", "empirical", "--configuration", NAME, "--days", "2", *arguments]
    )
    assert code == 2
    original = gate_day.prepare_day

    def abstaining(*args, **kwargs):
        prepared = original(*args, **kwargs)
        return replace(prepared, abstain_reason="made to abstain")

    monkeypatch.setattr(gate_day, "prepare_day", abstaining)
    assert _main(made_up) == 2
    assert "no checked unit has an offering day" in capsys.readouterr().out


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def _bounds(values):
    t = _t(values)
    return soft_scale.Bounds(t, t.clone(), t.clone(), t.clone())


ONE_HOUR = (_t([0.5]), _t([1.0]), _t([1.0]))  # centre and widths of one service hour


def _eps(values) -> list[float]:
    """Values in units of machine epsilon, to be compared with `pytest.approx(..., abs=0)`:
    the default absolute tolerance of `pytest.approx` (1e-12) is above any bound of a few eps."""
    flat = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    return [float(v) / soft_scale_check.EPSILON for v in flat]


def test_the_rounding_bounds_on_a_known_day():
    # one car: departure 2, rate 2, cap 1, need 0; one service hour, centre 0.5, widths 1 and 1:
    # b = min(2 * 1, 1) = 1, p = max(0 - 2 * (2 - 1), 0) = 0, C = 0.5, P = 0 + 2 * 2 = 4, both
    # terms 0.5; moving the cap, Q = |cap| * 1 = 1. K = n = 1:
    # upper: eps ((K + n) b + n C + Q) / w + (n + 2) eps |T| = eps (2 + 0.5 + 1) + 1.5 eps = 5 eps
    # lower: eps ((K - 1) p + (n + 2) P + n C + Q) / w + 1.5 eps = eps (12 + 0.5 + 1) + 1.5 eps
    bounds = soft_scale.Bounds(_t([2.0]), _t([2.0]), _t([1.0]), _t([0.0]))
    eps = soft_scale_check.EPSILON
    assert eps == 2.0**-52
    found, bound = soft_scale_check.term_rounding(bounds, ONE_HOUR, "cap")
    assert found.tolist() == [math.inf, 0.5, math.inf, 0.5]
    assert _eps(bound) == pytest.approx([0, 5, 0, 15], rel=1e-12, abs=0)
    # departure: Q = |departure| * rate = 4, upper 2 + 0.5 + 4 + 1.5 = 8 eps, lower 12 + 0.5 + 4
    # + 1.5 = 18 eps; rate: Q = |rate| * (departure + n) = 6, upper 10 eps, lower 20 eps
    _, by_departure = soft_scale_check.term_rounding(bounds, ONE_HOUR, "departure")
    _, by_rate = soft_scale_check.term_rounding(bounds, ONE_HOUR, "rate")
    assert _eps(by_departure) == pytest.approx([0, 8, 0, 18], rel=1e-12, abs=0)
    assert _eps(by_rate) == pytest.approx([0, 10, 0, 20], rel=1e-12, abs=0)
    # the exact scale: both terms can be the smallest, so the larger bound; the soft minimum:
    # weights 1/2, 1/2: (5 + 1) eps / 2 + (15 + 1) eps / 2 + (N + 3) eps sigma + 2 eps |phi|
    assert soft_scale_check.value_rounding("exact", found, bound, 0.5) == 15 * eps
    sigma = soft_scale_check.SIGMA
    phi = 0.5 - sigma * math.log(2)
    soft = soft_scale_check.value_rounding("soft", found, bound, phi)
    assert _eps(soft) == pytest.approx([11 + 5 * sigma + 2 * phi], rel=1e-12, abs=0)
    # unequal terms 0.5 and 0.52: weights 1 / (1 + e^-2) and e^-2 / (1 + e^-2) on the smaller
    # and the larger term (each weight on its own bound plus 2 eps |T|)
    unequal = torch.tensor([math.inf, 0.5, math.inf, 0.52], dtype=torch.float64)
    low = 1 / (1 + math.exp(-2))
    by_hand = low * 6 + (1 - low) * (15 + 1.04) + 5 * sigma + 2 * 0.49
    soft = soft_scale_check.value_rounding("soft", unequal, bound, 0.49)
    assert _eps(soft) == pytest.approx([by_hand], rel=1e-12, abs=0)
    # need 3: p = max(3 - 2 * (2 - 1), 0) = 1, P = 3 + 4 = 7, lower term 0.5 - 1 = -0.5; with
    # K - 1 = 0 the sum adds nothing: eps (0 * 1 + 3 * 7 + 0.5 + 1) + 1.5 eps = 24 eps
    needing = soft_scale.Bounds(_t([2.0]), _t([2.0]), _t([1.0]), _t([3.0]))
    terms_, bound_ = soft_scale_check.term_rounding(needing, ONE_HOUR, "cap")
    assert terms_[3] == -0.5 and _eps(bound_[3]) == pytest.approx([24], rel=1e-12, abs=0)
    two = soft_scale.Bounds(_t([2.0, 2.0]), _t([2.0, 2.0]), _t([1.0, 1.0]), _t([3.0, 3.0]))
    _, bound_ = soft_scale_check.term_rounding(two, (_t([0.5]), _t([1.0]), _t([1.0])), "cap")
    # K = 2: upper eps ((2 + 1) 2 + 0.5 + 2) + 3 * 1.5 eps; lower eps (1 * 2 + 3 * 14 + 0.5 + 2)
    # + 3 * 1.5 eps (terms 1.5 and -1.5)
    assert _eps(bound_[[1, 3]]) == pytest.approx([13, 51], rel=1e-12, abs=0)
    # a term far above the smallest is not near it and its bound does not count
    far = torch.tensor([math.inf, 0.5, math.inf, 7.0], dtype=torch.float64)
    assert soft_scale_check.value_rounding("exact", far, bound, 0.5) == 5 * eps
    # a term above the smallest by less than both bounds together is near it: its bound counts
    near = torch.tensor([math.inf, 0.5, math.inf, 0.5 + 3 * eps], dtype=torch.float64)
    assert soft_scale_check.value_rounding("exact", near, bound, 0.5) == 15 * eps
    # widths up 1 and down 2: the upper term 0.5 keeps 5 eps; the lower term (0.5 - 0) / 2 = 0.25
    # takes eps (12 + 0.5 + 1) / 2 + 3 * 0.25 eps = 7.5 eps
    uneven = (_t([0.5]), _t([1.0]), _t([2.0]))
    found_, bound_ = soft_scale_check.term_rounding(bounds, uneven, "cap")
    assert found_.tolist() == [math.inf, 0.5, math.inf, 0.25]
    assert _eps(bound_) == pytest.approx([0, 5, 0, 7.5], rel=1e-12, abs=0)
    # a centre of -0.5 (outside the set): |C| = 0.5 enters; upper term 1.5, eps (2 + 0.5 + 1) +
    # 3 * 1.5 eps = 8 eps; lower term -0.5, eps (12 + 0.5 + 1) + 1.5 eps = 15 eps
    below = (_t([-0.5]), _t([1.0]), _t([1.0]))
    found_, bound_ = soft_scale_check.term_rounding(bounds, below, "cap")
    assert found_.tolist() == [math.inf, 1.5, math.inf, -0.5]
    assert _eps(bound_) == pytest.approx([0, 8, 0, 15], rel=1e-12, abs=0)


def test_an_infinite_value_fails_and_is_written_with_nulls():
    bounds = soft_scale.Bounds(_t([2.0]), _t([2.0]), _t([1.0]), _t([0.0]))
    no_width = (_t([0.5]), _t([0.0]), _t([0.0]))
    for kind in soft_scale_check.KINDS:
        found = soft_scale_check.compare(kind, bounds, "cap", _t([1.0]), no_width)
        assert found["outcome"] == "failed" and found["non_finite"]
        assert found["decided_step"] is None and found["relative_error"] is None
        json.dumps(found, allow_nan=False)  # writable to the results file


def test_smallest_terms_counts_the_terms_within_the_tie_tolerance(monkeypatch):
    def given(values):
        monkeypatch.setattr(soft_scale, "terms", lambda bounds, *shape: _t(values))
        return soft_scale_check.smallest_terms(None, ())

    inf = float("inf")
    assert given([1.0, 1.0 + 1e-3, inf, 5.0]) == 1
    assert given([1.0, 1.0 + 1e-12, inf, 5.0]) == 2
    assert given([-3.0, 1e-4, -3.0 + 2e-9]) == 2  # within 1e-9 * |scale|
    assert given([-3.0, 1e-4, -3.0 + 4e-9]) == 1
    assert given([1e-4, 1e-4 + 5e-10, inf]) == 2  # within 1e-9 * max(1, |scale|) = 1e-9
    assert given([1.0, 1.0 + 1e-9, inf]) == 2  # exactly at the tolerance counts


def _record(outcome: str, error: float = 1e-9, step: float = 1e-4, resolvable=None) -> dict:
    """A written comparison; one decided at a smaller step follows a first step of allowance
    1e-3 (not resolvable); `resolvable` False with `failed` is a step decided far outside its
    interval without being resolvable."""
    if outcome == "tied":
        return {"outcome": "tied"}
    decided = outcome in ("passed", "failed")
    resolvable = decided if resolvable is None else resolvable
    steps = [{"step": step, "difference": 1.0, "allowance": 1e-6, "resolvable": resolvable}]
    if step < 1e-4:
        steps.insert(0, {"step": 1e-4, "difference": 1.0, "allowance": 1e-3, "resolvable": False})
    return {
        "outcome": outcome,
        "non_finite": False,
        "autograd": 1.0 + error,
        "decided_step": step if decided else None,
        "resolvable": resolvable,
        "relative_error": error if decided else None,
        "steps": steps,
    }


def _row(*records) -> dict:
    """A day whose comparisons are `records` for the exact scale's need, all passed elsewhere."""
    fields = soft_scale_check.FIELDS
    comparisons = {kind: {f: _record("passed") for f in fields} for kind in soft_scale_check.KINDS}
    comparisons["exact"]["need"] = records[0]
    return {
        "comparisons": comparisons,
        "smallest_terms": 2 if records[0] == {"outcome": "tied"} else 1,
    }


def test_errors_count_outcomes_and_leave_the_tied_comparisons_out():
    rows = [
        _row(_record("passed", 1e-6)),
        _row(_record("tied")),
        _row(_record("passed", 2e-6, step=1e-6)),
        _row(_record("unresolved")),
        _row(_record("kink")),
    ]
    found = soft_scale_check.errors(rows)["exact"]["need"]
    assert found["compared"] == 4 and found["tied"] == 1
    assert (found["passed"], found["failed"], found["unresolved"], found["kink"]) == (2, 0, 1, 1)
    assert found["resolvable"] == 2 and found["at_smaller_step"] == 1
    assert found["largest_error"] == 2e-6
    assert found["largest_allowance_share"] == pytest.approx(1e-6 / soft_scale_check.LIMIT)
    assert soft_scale_check.failed_comparisons(rows) == 0
    assert soft_scale_check.tied_days(rows) == 1
    rows.append(_row(_record("failed", 0.5)))
    assert soft_scale_check.failed_comparisons(rows) == 1
    found = soft_scale_check.errors(rows)["exact"]["need"]
    assert found["failed"] == 1 and found["non_finite"] == 0  # failed, not for a non-finite value
    assert found["largest_error"] == 0.5
    # failed far outside an interval that is not resolvable: counted, not in the largest error
    rows[-1] = _row(_record("failed", 0.7, resolvable=False))
    found = soft_scale_check.errors(rows)["exact"]["need"]
    assert found["failed"] == 1 and found["resolvable"] == 2
    assert found["largest_error"] == 2e-6
    assert found["largest_allowance_share"] == pytest.approx(1e-6 / soft_scale_check.LIMIT)
    written = _record("failed", 0.5)
    written["non_finite"] = True
    rows.append(_row(written))
    assert soft_scale_check.errors(rows)["exact"]["need"]["non_finite"] == 1
    every = {"exact need"} | {f"soft {f}" for f in soft_scale_check.DECIDING}
    unresolved_only = [_row(_record("unresolved"))]
    for row in unresolved_only:
        for kind in soft_scale_check.KINDS:
            for field in soft_scale_check.FIELDS:
                if (kind, field) != ("exact", "need"):
                    row["comparisons"][kind][field] = _record("unresolved")
    missing = soft_scale_check.not_exercised(soft_scale_check.errors(unresolved_only))
    assert set(missing) == every | {"exact departure", "exact cap"}


def _one_prepared_day(made_up):
    matrix = load(made_up["matrix"])
    unit = GATE_UNITS[1]
    days = load_unit_days(made_up["tables"], unit, rho=0.0, no_fit="cap", small_fleet="keep")
    forecaster = gate_run.empirical_forecaster(days, matrix)
    ((_, prepared),), _, _ = soft_scale_check.sample_unit(
        days, forecaster, matrix.configuration(NAME), gate_run.settings(matrix), 1, 1
    )
    return prepared


def test_the_sandwich_violation_counts_the_finite_terms(made_up, monkeypatch):
    prepared = _one_prepared_day(made_up)
    exact = soft_scale.exact_scale

    def too_low(bounds, centre, up, down, sigma):  # below the sandwich by sigma ln N
        count = soft_scale.term_count(up, down)
        return exact(bounds, centre, up, down) - 2 * sigma * math.log(count)

    monkeypatch.setattr(soft_scale, "soft_scale", too_low)
    row = soft_scale_check.check_day(prepared, np.random.default_rng(0))
    assert row["terms"] != 2 ** (prepared.day.n_service + 1)
    expected = soft_scale_check.SIGMA * math.log(row["terms"])
    assert row["sandwich_violation"] == pytest.approx(expected, rel=1e-9)


def test_tracking_correlates_the_named_pairs():
    scale, value = [1.0, 2.0, 3.0, 4.5], [2.0, 1.0, 4.0, 3.0]
    scaled = [s * v for s, v in zip(scale, value, strict=True)]
    rows = [
        {"scale": s, "certified_value_kwh": v, "scaled_value_kwh": w}
        for s, v, w in zip(scale, value, scaled, strict=True)
    ]
    found = soft_scale_check.tracking(rows)
    expected = {
        "scale_with_certified_value": (scale, value),
        "scaled_value_with_certified_value": (scaled, value),
    }
    for name, (x, y) in expected.items():
        assert found[name]["pearson"] == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-12)
        ranks = (stats.rankdata(x), stats.rankdata(y))
        assert found[name]["spearman"] == pytest.approx(np.corrcoef(*ranks)[0, 1], rel=1e-12)
    columns = (scale, value, scaled)
    every_pair = {
        round(float(np.corrcoef(a, b)[0, 1]), 9)
        for i, a in enumerate(columns)
        for b in columns[i + 1 :]
    }
    assert len(every_pair) == 3  # the three pairs differ, so a wrong pair is seen
    flat = [dict(r, scale=1.0 + 1e-12 * k) for k, r in enumerate(rows)]
    assert soft_scale_check.tracking(flat)["scale_with_certified_value"]["pearson"] is None
    assert soft_scale_check.constant([5.0, 5.0 + 4e-9])
    assert not soft_scale_check.constant([5.0, 5.0 + 6e-9])
    assert soft_scale_check.tracking(rows[:2])["scale_with_certified_value"]["pearson"] is None
    below = [dict(r, scale=s) for r, s in zip(rows, (1 - 1e-5, 1 - 1e-7, 1.0, 2.0), strict=True)]
    assert soft_scale_check.tracking(below)["below_one"] == 1  # 1e-6 below 1 counts as 1


def test_the_sample_is_the_first_offering_days_of_the_seeded_order(made_up, monkeypatch):
    matrix = load(made_up["matrix"])
    configuration, settings = matrix.configuration(NAME), gate_run.settings(matrix)
    unit = GATE_UNITS[2]
    days = load_unit_days(made_up["tables"], unit, rho=0.0, no_fit="cap", small_fleet="keep")
    forecaster = gate_run.empirical_forecaster(days, matrix)
    seed = 3
    order = np.random.default_rng([seed, zlib.crc32(unit.encode())]).permutation(len(days.days))
    first = days.days[int(order[0])]
    original = gate_day.prepare_day

    def first_abstains(day, *args, **kwargs):
        prepared = original(day, *args, **kwargs)
        return replace(prepared, abstain_reason="made to abstain") if day is first else prepared

    monkeypatch.setattr(gate_day, "prepare_day", first_abstains)
    states, check_day = [], soft_scale_check.check_day

    def recording(prepared, rng):  # the directions' generator, as the first day finds it
        states.append(rng.bit_generator.state["state"])
        return check_day(prepared, rng)

    monkeypatch.setattr(soft_scale_check, "check_day", recording)
    checked, abstained, examined = soft_scale_check.sample_unit(
        days, forecaster, configuration, settings, 4, seed
    )
    records = [days.days[int(i)].record for i in order[1:5]]
    expected = [(record.site, record.date) for record in records]
    assert [(row["site"], row["date"]) for row, _ in checked] == expected
    assert abstained == {"made to abstain": 1} and examined == 5
    directions = np.random.default_rng([seed, zlib.crc32(unit.encode()), 1])
    assert states[0] == directions.bit_generator.state["state"]


def test_the_command_defaults_to_100_days_seed_1_and_the_soft_minimum_first(monkeypatch):
    seen = {}

    def capture(args):
        seen.update(days=args.days, seed=args.seed)
        raise soft_scale_check.NoCheck("captured")

    monkeypatch.setattr(soft_scale_check, "prepare_inputs", capture)
    assert soft_scale_check.main(["--model", "empirical", "--configuration", NAME]) == 2
    assert seen == {"days": 100, "seed": 1}
    assert soft_scale_check.KINDS == ("soft", "exact")


def test_a_certified_shape_with_no_width_is_not_checked(made_up, monkeypatch):
    prepared = _one_prepared_day(made_up)
    centre, up, _ = soft_scale_check.day_shape(prepared)
    monkeypatch.setattr(soft_scale_check, "day_shape", lambda p: (centre, 0 * up, 0 * up))
    with pytest.raises(soft_scale_check.NoCheck, match="has no width"):
        soft_scale_check.check_day(prepared, np.random.default_rng(0))


def test_each_field_gets_its_own_unit_direction_from_the_days_generator(made_up, monkeypatch):
    prepared = _one_prepared_day(made_up)
    seen = []

    def record(kind, bounds, field, direction, shape):
        seen.append((kind, field, direction.tolist()))
        return {"outcome": "passed"}

    monkeypatch.setattr(soft_scale_check, "compare", record)
    soft_scale_check.check_day(prepared, np.random.default_rng(11))
    rng, cars = np.random.default_rng(11), len(prepared.day.forecasts)
    got = [(field, direction) for kind, field, direction in seen if kind == "soft"]
    assert [field for field, _ in got] == list(soft_scale_check.FIELDS)
    for _, direction in got:
        raw = rng.standard_normal(cars)
        assert direction == pytest.approx(list(raw / np.linalg.norm(raw)), rel=1e-12)
    assert any(x < 0 for _, direction in got for x in direction)


def test_the_batch_is_the_most_common_day_size_padded_to_its_largest_fleet(made_up):
    def day(n):
        return SimpleNamespace(day=SimpleNamespace(n_service=n))

    days = [day(2), day(3), day(3), day(1), day(1)]
    assert [d.day.n_service for d in soft_scale_check.batch_of(days)] == [1, 1]  # tie: fewer
    days.append(day(3))
    assert [d.day.n_service for d in soft_scale_check.batch_of(days)] == [3, 3, 3]
    prepared = _one_prepared_day(made_up)
    timing = soft_scale_check.time_batch([prepared], "cpu", torch.float64)
    assert timing["padded_cars"] == len(prepared.day.forecasts) and timing["batch_days"] == 1


def _errors_entry(deciding: bool, first: int, largest, share) -> dict:
    names = ("compared", "tied", "resolvable", "passed", "failed", "unresolved", "kink")
    entry = {name: first + k for k, name in enumerate(names)}
    entry.update(at_smaller_step=first + 7, non_finite=first + 8)
    entry.update(deciding=deciding, largest_error=largest, largest_allowance_share=share)
    return entry


def test_every_line_of_the_summary_of_a_made_up_document():
    document = {
        "model": "m",
        "configuration": "c",
        "settings": {
            "steps": {"exact": 1e-4, "soft": 1e-5},
            "shrink": 100.0,
            "sigma": 0.01,
            "limit": 0.001,
            "resolution": 0.25,
            "deciding_fields": ["departure", "cap", "need"],
            "not_applicable": [],
        },
        "units": {
            "u1": {
                "checked_days": 3,
                "examined_days": 5,
                "counting_days": 9,
                "abstained": {"no offer": 2},
                "fewer_than_asked": True,
            },
            "u2": {
                "checked_days": 4,
                "examined_days": 4,
                "counting_days": 4,
                "abstained": {},
                "fewer_than_asked": False,
            },
        },
        "tied_days": 1,
        "checked_days": 7,
        "errors": {
            "soft": {"need": _errors_entry(True, 11, 0.000123, 0.25)},
            "exact": {"rate": _errors_entry(False, 0, None, None)},
        },
        "largest_sandwich_violation": 0.0,
        "tracking": {
            "scale_with_certified_value": "a",
            "scaled_value_with_certified_value": "b",
            "below_one": 3,
            "smallest_scale": 0.875,
        },
        "batch_timing": {
            "cpu": {
                "dtype": "float64",
                "batch_days": 2,
                "padded_cars": 6,
                "n_service": 3,
                "seconds": 0.0123,
                "warm_up_seconds": 0.5,
            },
            "mps": "not available",
        },
        "not_exercised": ["soft need"],
        "failed_comparisons": 15,
        "passed": False,
    }
    assert soft_scale_check.summary_lines(document) == [
        "T8 check: model m, configuration c, sigma 0.01, first steps exact 0.0001, soft 1e-05 "
        "(smaller: divided by 100), limit 0.001, resolvable at an allowance of 0.25 limit |D|.",
        "Deciding: departure, cap and need, for the soft minimum on every day and the exact scale "
        "on days whose smallest term is unique; rate is reported only.",
        "u1: 3 days checked of 5 examined (9 counting); abstained {'no offer': 2}; "
        "FEWER THAN ASKED",
        "u2: 4 days checked of 4 examined (4 counting); abstained none",
        "Tied days: 1 of 7 checked days have tied smallest terms; the exact scale is not compared "
        "on them (a kink at or very near the point: no derivative there, or one the steps used "
        "cannot measure).",
        "Comparisons over all checked days (compared | tied | resolvable | passed | failed | "
        "unresolved | kink | decided at the smaller step | not finite | largest relative error "
        "and largest allowance share over the resolvable ones):",
        "  soft need: 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 0.000123, 0.25",
        "  exact rate (not deciding): 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | none, none",
        "Largest sandwich violation: 0.",
        "Tracking (a reading): scale with certified value a; scale x value with certified value "
        "b; days with scale below 1: 3; smallest scale 0.875.",
        "Batch on cpu (float64): 2 days, 6 cars, 3 hours: 0.0123 s (warm-up 0.5000 s).",
        "Batch on mps: not available.",
        "FIELD NOT EXERCISED: soft need (no resolvable comparison).",
        "FAILED: 15 deciding comparisons failed; 1 deciding fields not exercised.",
    ]
    document.update(passed=True, failed_comparisons=0, not_exercised=[])
    assert soft_scale_check.summary_lines(document)[-1] == (
        "PASSED: 0 deciding comparisons failed; 0 deciding fields not exercised."
    )
    document["settings"].update(deciding_fields=["cap", "need"], not_applicable=["departure"])
    assert soft_scale_check.summary_lines(document)[1:3] == [
        "Deciding: cap and need, for the soft minimum on every day and the exact scale on days "
        "whose smallest term is unique; rate is reported only.",
        "NOT APPLICABLE: departure, on this configuration's whole-hour forecast departures (every "
        "point is a kink in that direction); its comparisons are reported only.",
    ]
