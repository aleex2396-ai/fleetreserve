"""The method `max-entropy-feedback` with the no-pause sensitivity (`compare_run --no-pause-share`),
on the made-up units of `test_no_pause_command` (two service hours; 2 days that cannot be split,
28 days of site A, 14 of site B and 14 of site C pooled), 20 calibration days, 6 splits, methods
`re-solve`, `fixed-plan`, `llf` and `max-entropy-feedback`; runs without the option, at share 0
and at share 1/3.

1. Without the option and at share 0 the results have, level by level, exactly the keys that
   `test_no_pause_command` fixes as literal sets for a run without the method, but for the method
   among the methods and gate readings; each variant's `evaluations` has exactly the literal keys
   `EVALUATIONS` (those of a run without the method and `controller_deviations`), and
   `controller_deviations` holds the method alone with the literal keys `test_mef_command.KEYS`;
   the definitions entry `controller_deviations` is `test_mef_command.definition` word for word.
   The file names, the split table's header and the timing file's keys are those fixed there.
2. At share 1/3 the same holds with the keys that `test_no_pause_command` adds at share 1/3; the
   file names, header and timing keys are those of the sensitivity, and leaving out the paths that
   name the sensitivity, the key paths of the whole document are those of the run without the
   option.
3. At share 0 every file but the timing file is the same bytes as the run without the option.
4. At share 1/3, with one worker: every day the controllers are evaluated on leaves exactly the
   recounted cars (`test_no_pause_command`'s own hash rule) unsteered with the reason
   `gate_day.NO_PAUSE`, none of them steered, and the method's controller sees the steered cars
   only (every view holds them, every believed future the view's plugged cars, every box is the
   day's `m -+ beta w` at the scale evaluated); the written deviation counts under each unit and
   variant are the sums of the counts the controllers gave there, and every hour planned is
   counted; some hours deviate under the forecast variant. The summary has no line of controller
   solves (no method here solves a programme) and, right after its line of timeouts, the line of
   deviations of the written counts summed over units and variants, then one empty line and the
   text of the paired differences.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from compare_fake import NAMES, TWO_HOURS, two_hour_cars
from gate_fake import COMMIT, GATE_UNITS, gate_cars, matrix_document, write_build
from test_mef_command import KEYS, definition
from test_no_pause_command import (
    DOCUMENT_KEYS,
    GATE_CEILING_KEYS,
    GATE_KEYS,
    PLAIN_FILES,
    RUN_KEYS,
    SPLIT_UNIT_KEYS,
    SUFFIX,
    TIER_A_KEYS,
    TIER_B_KEYS,
    TIMEOUT_KEYS,
    UNSPLIT_UNIT_KEYS,
    VARIANT_KEYS,
    _own_draw,
    _paths,
)

from fleetreserve import (
    compare_day,
    compare_methods,
    compare_run,
    compare_splits,
    compare_summary,
    gate_config,
    gate_day,
    gate_run,
    mef_controller,
    no_pause,
    run_verify,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.mef_controller import COUNTS

SPLITS = 6
NAME = "max-entropy-feedback"
METHODS = ["re-solve", "fixed-plan", "llf", NAME]
SPLIT_UNITS = GATE_UNITS[1:]
LINE = "Controller deviations (all units and variants together): "
EVALUATIONS = {
    "controller_day_scale_pairs",
    "controller_deviations",
    "re_solve_day_scale_pairs",
    "re_solve_pairs_evaluated_more_than_once",
    "re_solves_of_the_one_way_rung",
}


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        GATE_UNITS[0]: gate_cars(2),
        GATE_UNITS[1]: two_hour_cars(28),
        GATE_UNITS[2]: two_hour_cars(14, "B")
        + [
            dataclasses.replace(c, session=c.session.replace("C-", "B-", 1))
            for c in two_hour_cars(14, "C")
        ],
    }
    run, tables = write_build(folder, units, TWO_HOURS)
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix


def _run(build, folder: Path, *extra, record=None):
    """One run; `record` (a list) receives, per call of `measure_variant`, a list of [prepared
    day, the method's deviation counts, the set of what the method's controllers saw] per
    `compare_day.evaluate` call: per believed future, (the view's number of cars, whether the
    future holds exactly the view's plugged cars, whether the box is the day's `m -+ beta w`)."""
    run, tables, matrix = build
    folder.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        if record is not None:
            measure, evaluate = compare_run.measure_variant, compare_day.evaluate

            class Seeing(mef_controller.MefController):
                def future(self, view, powers):
                    found = super().future(view, powers)
                    prepared, beta = record[-1][-1][3]
                    policy = prepared.certificate.policy
                    box = np.array_equal(self.lo, policy.m - beta * policy.w) and np.array_equal(
                        self.hi, policy.m + beta * policy.w
                    )
                    plugged = found.high.shape[0] == view.plugged.sum()
                    record[-1][-1][2].add((view.n_cars, bool(plugged), bool(box)))
                    return found

            def measured(*args, **kwargs):
                record.append([])
                return measure(*args, **kwargs)

            def evaluated(prepared, beta, chosen):
                record[-1].append([prepared, None, set(), (prepared, float(beta))])
                found = evaluate(prepared, beta, chosen)
                record[-1][-1][1] = found[NAME].deviation_counts
                return found

            monkeypatch.setattr(compare_run, "measure_variant", measured)
            monkeypatch.setattr(compare_day, "evaluate", evaluated)
            monkeypatch.setattr(compare_methods, "MefController", Seeing)
        arguments = ["--model", "empirical", "--dial", "0.03", "--matrix", str(matrix)]
        arguments += ["--tables", str(tables), "--truth-run", str(run), "--splits", str(SPLITS)]
        arguments += ["--workers", "1", "--methods", *METHODS, *extra]
        return compare_run.main(arguments), folder


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    top = tmp_path_factory.mktemp("runs")
    record: list = []
    found = {
        "plain": _run(build, top / "plain"),
        "zero": _run(build, top / "zero", "--no-pause-share", "0"),
        "third": _run(build, top / "third", "--no-pause-share", "1/3", record=record),
    }
    return found, record


@pytest.fixture(scope="module")
def units(build):
    _, tables, matrix = build
    loaded = gate_config.load(matrix)
    return {
        u: load_unit_days(
            tables, u, rho=loaded.rho, no_fit=loaded.no_fit, small_fleet=loaded.small_fleet
        )
        for u in SPLIT_UNITS
    }


def _names(share: str) -> dict:
    return compare_run.output_names(no_pause.read_share(share))


def _document(folder: Path, share: str) -> dict:
    return json.loads((folder / _names(share)["results"]).read_text())


def _levels(document: dict, ceilings, extra=lambda level: set()):
    """The keys of `test_no_pause_command._levels`, with the method among the methods, and the
    evaluations' keys."""
    assert set(document) == DOCUMENT_KEYS | set(extra("document"))
    assert set(document["run"]) == RUN_KEYS
    assert document["run"]["ceilings"] == [*ceilings, *extra("ceilings")]
    assert set(document["timeouts"]) == TIMEOUT_KEYS | set(extra("timeouts"))
    assert set(document["units"]) == set(GATE_UNITS)
    assert set(document["units"][GATE_UNITS[0]]) == UNSPLIT_UNIT_KEYS
    assert document["units"][GATE_UNITS[0]]["variants"] == {}
    assert document["methods"] == METHODS
    for unit in SPLIT_UNITS:
        entry = document["units"][unit]
        assert set(entry) == SPLIT_UNIT_KEYS | set(extra("unit"))
        assert set(entry["variants"]) == {"forecast", "oracle-departure"}
        for variant in entry["variants"].values():
            assert set(variant) == VARIANT_KEYS | set(extra("variant"))
            assert set(variant["evaluations"]) == EVALUATIONS
            assert list(variant["evaluations"]["controller_deviations"]) == [NAME]
            assert list(variant["evaluations"]["controller_deviations"][NAME]) == KEYS
            assert set(variant["methods"]) == set(METHODS)
            for method in variant["methods"].values():
                assert set(method) == {"A", "B"}
                assert set(method["A"]) == TIER_A_KEYS and set(method["B"]) == TIER_B_KEYS
        assert set(entry["gate_reading"]) == set(METHODS)
        for method in entry["gate_reading"].values():
            assert set(method) == GATE_KEYS
            assert set(method["ceilings"]) == {*ceilings, *extra("ceilings")}
            assert all(set(c) == GATE_CEILING_KEYS for c in method["ceilings"].values())


def test_the_keys_and_files_without_the_option(runs, build):
    ceilings = list(gate_config.load(build[2]).ceilings)
    header = [*compare_splits.BASE_COLUMNS, *(compare_splits.ceiling_column(c) for c in ceilings)]
    for name in ("plain", "zero"):
        code, folder = runs[0][name]
        assert code == 0
        assert sorted(p.name for p in folder.iterdir()) == sorted(PLAIN_FILES)
        document = _document(folder, "0")
        _levels(document, ceilings)
        assert document["definitions"]["controller_deviations"] == definition(NAME)
        assert (folder / "compare-splits.csv").read_text().splitlines()[0] == ",".join(header)
        timing = json.loads((folder / "compare-timing.json").read_text())
        assert set(timing) == {"total_seconds", "units"}


def test_share_zero_is_the_run_without_the_option(runs):
    plain, zero = runs[0]["plain"][1], runs[0]["zero"][1]
    names = _names("0")
    for kind in ("results", "summary", "splits", "table"):
        assert (plain / names[kind]).read_bytes() == (zero / names[kind]).read_bytes(), kind


def test_the_keys_and_files_at_one_third(runs, build):
    ceilings = list(gate_config.load(build[2]).ceilings)
    reduced = [c + SUFFIX for c in ceilings]
    added = {
        "document": {"no_pause"},
        "ceilings": reduced,
        "timeouts": {"best_causal_ceilings_no_pause_cars_removed"},
        "unit": {"no_pause"},
        "variant": {"unsteered_cars"},
    }
    code, third = runs[0]["third"]
    assert code == 0
    names = _names("1/3")
    assert sorted(p.name for p in third.iterdir()) == sorted(names.values())
    document = _document(third, "1/3")
    _levels(document, ceilings, lambda level: added.get(level, ()))
    assert document["definitions"]["controller_deviations"] == definition(NAME)
    documented = [("no_pause",), ("timeouts", "best_causal_ceilings_no_pause_cars_removed")]
    for unit in SPLIT_UNITS:
        documented.append(("units", unit, "no_pause"))
        for variant in ("forecast", "oracle-departure"):
            documented.append(("units", unit, "variants", variant, "unsteered_cars"))
        for method in METHODS:
            for name in reduced:
                documented.append(("units", unit, "gate_reading", method, "ceilings", name))
    mine = set(_paths(document))
    assert all(p in mine for p in documented)
    left = {p for p in mine if not any(p[: len(d)] == d for d in documented)}
    assert left == set(_paths(_document(runs[0]["plain"][1], "0")))
    plain_header = (runs[0]["plain"][1] / "compare-splits.csv").read_text().splitlines()[0]
    header = (third / names["splits"]).read_text().splitlines()[0].split(",")
    assert header == plain_header.split(",") + [compare_splits.ceiling_column(c) for c in reduced]
    timing = json.loads((third / names["timing"]).read_text())
    assert set(timing) == {"total_seconds", "units", "no_pause"}


def test_at_one_third_the_counts_are_those_of_the_steered_cars(runs, units):
    found, record = runs
    third = found["third"][1]
    document = _document(third, "1/3")
    expected = {}
    for unit in SPLIT_UNITS:
        for truth in units[unit].days:
            drawn = tuple(
                i
                for i, car in enumerate(truth.fleet)
                if _own_draw(car.site, car.session_id, no_pause.DEFAULT_SEED)
            )
            expected[(truth.day.unit, truth.record.site, truth.record.date)] = drawn
    order = [(u, v) for u in SPLIT_UNITS for v in compare_run.VARIANTS]
    assert len(record) == len(order)
    totals = dict.fromkeys(COUNTS, 0)
    deviating = {}
    any_drawn = False
    for (unit, variant), calls in zip(order, record, strict=True):
        summed = dict.fromkeys(COUNTS, 0)
        for prepared, counts, seen, _ in calls:
            drawn = expected[(prepared.unit, prepared.site, prepared.date)]
            assert tuple(i for i, why in prepared.unsteered if why == gate_day.NO_PAUSE) == drawn
            assert not set(drawn) & set(prepared.steered)
            assert len(prepared.day.cars) == len(prepared.steered)
            assert seen == {(len(prepared.steered), True, True)}
            any_drawn = any_drawn or bool(drawn)
            for key in COUNTS:
                summed[key] += counts[key]
        found = document["units"][unit]["variants"][variant]
        assert found["evaluations"]["controller_deviations"][NAME] == summed
        assert summed["hours"] == sum(8 * (p.day.n_service + 1) for p, *_ in calls)
        assert len(calls) == found["offering_days"] * len(compare_run.dial_splits.GRID)
        deviating[(unit, variant)] = summed["deviating_hours"]
        for key in COUNTS:
            totals[key] += summed[key]
    assert any_drawn
    assert all(deviating[(u, "forecast")] > 0 for u in SPLIT_UNITS), deviating
    summary = (third / _names("1/3")["summary"]).read_text().splitlines()
    line = (
        f"{LINE}{NAME}: hours planned {totals['hours']}, with least-laxity-first's split found "
        "infeasible and the total nearest the target found feasible planned "
        f"{totals['deviating_hours']} (in service hours "
        f"{totals['deviating_service_hours']}), with no feasible total "
        f"{totals['no_feasible_hours']}."
    )
    at = next(k for k, x in enumerate(summary) if x.startswith("Timeouts"))
    assert not any(x.startswith("Controller solves") for x in summary)  # no method here solves
    assert summary[at + 1] == line
    assert summary[at + 2] == "" and summary[at + 3] == compare_summary.PAIRED_TEXT
    assert sum(x.startswith(LINE) for x in summary) == 1
