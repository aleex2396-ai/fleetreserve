"""The method `max-entropy-feedback` through the comparison command (`compare_run`), on made-up
gate units of two service hours (`compare_fake.two_hour_cars`: 28 days at one unit, 24 at
another; 2 days of `gate_fake.gate_cars` at the third, which cannot be split), 20 calibration
days, 4 splits, the dial 0.03.

1. A run of `re-solve`, `llf`, `diagonal-split` and `max-entropy-feedback` ends with exit 0.
   Under each unit that can be split and each variant (forecast and oracle departures),
   `evaluations` has exactly the keys of a run without the method plus `controller_deviations`,
   which holds the method alone with exactly the literal keys `KEYS` (in the written, sorted
   order); the definitions entry `controller_deviations` is `definition` word for word, naming
   the method, which the definitions of solver counts and timeouts do not name; the summary has,
   right after the line of timeouts, the line of controller solves (without the method), then one
   line of controller deviations, written out word for word, whose figures are the written counts
   summed over units and variants, then one empty line and the text of the paired differences.
2. The written counts are those the controllers gave: with one worker, every
   `compare_day.evaluate` call's deviation counts, summed per unit and variant in the order the
   command measures them, equal the counts written under that unit and variant; every planned
   hour is counted (8 audited paths times the hours 0 to n of each day, at each of the 10
   scales); some hours deviate under the forecast variant at both units (under the
   oracle-departure variant none does on these units; `test_mef_runs` has deviating hours under
   the oracle belief).
3. The results, the summary and the split table are the same bytes with 1 and 3 workers.
4. A run of the same methods but `max-entropy-feedback` writes no `controller_deviations`
   (results, definitions) and no line of deviations (the line of solves is followed by one empty
   line and the text of the paired differences); its methods' figures, evaluations, gate
   readings, split rows and paired differences against `re-solve` are those of the run with the
   method, whose own rows and pairs come on top.
5. Every table of the summary (tier, gate-reading and paired tables) starts its second column
   (and a paired table its third; a tier table's line of variant names its first) at the same
   place in the header and in every row: the method column is as wide as the longest method
   name of the run (20 with the method), and `compare_summary.METHOD_WIDTH` (14) without it.
6. The line of deviations on a results document built by hand, with a different figure for each
   count in each unit and variant: each count is summed over units and variants and written in
   its own place; a method without counts is left out, and the methods come in the run's order.
7. `compare_run.definitions` writes the entry `controller_deviations` word for word, naming the
   counting methods in the order given (two methods in an order that is not sorted keep it), and
   no entry when no method counts; `mef_controller.COUNTS` is the literal tuple of the four keys.
"""

import csv
import json
import re
from pathlib import Path

import pytest
from compare_fake import NAMES, TWO_HOURS, two_hour_cars
from gate_fake import COMMIT, GATE_UNITS, gate_cars, matrix_document, write_build

from fleetreserve import (
    compare_day,
    compare_run,
    compare_summary,
    gate_config,
    gate_run,
    run_verify,
)
from fleetreserve.mef_controller import COUNTS

SPLITS = 4
NAME = "max-entropy-feedback"
WITH = ["re-solve", "llf", "diagonal-split", NAME]
WITHOUT = ["re-solve", "llf", "diagonal-split"]
LINE = "Controller deviations (all units and variants together): "
KEYS = ["deviating_hours", "deviating_service_hours", "hours", "no_feasible_hours"]  # as written


def definition(names: str) -> str:
    """The definitions entry `controller_deviations` of a run whose counting methods are `names`
    (joined in the run's order), written out in full."""
    return (
        "evaluations.controller_deviations, per unit, variant and method that counts deviations ("
        + names
        + "), summed over the (day, scale) pairs evaluated and the audited paths: hours, the hours "
        "the controller planned (0 to n on each path); deviating_hours, those in which "
        "least-laxity-first's split of the hour's target was found to leave the believed future "
        "infeasible and the total nearest the target among those found to keep it feasible was "
        "planned instead (mef_controller; a total other than least-laxity-first's, except where a "
        "car's remaining need (its believed need less its drawn energy and its believed recovery "
        "supply) is above its remaining room by the tolerance to within rounding); "
        "deviating_service_hours, those of them in service hours 1 to n; no_feasible_hours, the "
        "hours in which least-laxity-first's split was found infeasible and no total was found to "
        "keep the believed future feasible, planned as least-laxity-first plans them"
    )


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        GATE_UNITS[0]: gate_cars(2),
        GATE_UNITS[1]: two_hour_cars(28),
        GATE_UNITS[2]: two_hour_cars(24, "B"),
    }
    run, tables = write_build(folder, units, TWO_HOURS)
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix


def _run(build, folder: Path, workers: int, methods, record=None):
    """One run; `record` (a list) receives, per call of `measure_variant`, a list of
    (unit, n_service, deviation counts of the method) per `compare_day.evaluate` call."""
    run, tables, matrix = build
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        if record is not None:
            measure, evaluate = compare_run.measure_variant, compare_day.evaluate

            def measured(*args, **kwargs):
                record.append([])
                return measure(*args, **kwargs)

            def evaluated(prepared, beta, chosen):
                found = evaluate(prepared, beta, chosen)
                counts = found[NAME].deviation_counts
                record[-1].append((prepared.unit, prepared.day.n_service, counts))
                return found

            monkeypatch.setattr(compare_run, "measure_variant", measured)
            monkeypatch.setattr(compare_day, "evaluate", evaluated)
        folder.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        arguments = ["--model", "empirical", "--dial", "0.03", "--matrix", str(matrix)]
        arguments += ["--tables", str(tables), "--truth-run", str(run), "--splits", str(SPLITS)]
        arguments += ["--workers", str(workers), "--methods", *methods]
        return compare_run.main(arguments), folder


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    top = tmp_path_factory.mktemp("runs")
    record: list = []
    found = {"with": _run(build, top / "with", 1, WITH, record)}
    found["with3"] = _run(build, top / "with3", 3, WITH)
    found["without"] = _run(build, top / "without", 1, WITHOUT)
    found["record"] = record
    return found


def _document(folder: Path) -> dict:
    return json.loads((folder / compare_run.RESULTS_NAME).read_text())


def _calibratable(document):
    return [u for u in GATE_UNITS if document["units"][u]["status"] == "calibratable"]


def test_a_run_with_the_method_writes_its_counts_under_every_unit_and_variant(runs):
    (code, out), (_, bare) = runs["with"], runs["without"]
    assert code == 0
    document, without = _document(out), _document(bare)
    assert document["methods"] == WITH
    units = _calibratable(document)
    assert len(units) == 2
    totals = dict.fromkeys(COUNTS, 0)
    for unit in units:
        assert set(document["units"][unit]["variants"]) == set(compare_run.VARIANTS)
        for variant, found in document["units"][unit]["variants"].items():
            plain = without["units"][unit]["variants"][variant]["evaluations"]
            assert set(found["evaluations"]) == {*plain, "controller_deviations"}
            assert "controller_deviations" not in plain
            assert list(found["evaluations"]["controller_deviations"]) == [NAME]
            counts = found["evaluations"]["controller_deviations"][NAME]
            assert list(counts) == KEYS  # the results are written with keys sorted
            for key in COUNTS:
                totals[key] += counts[key]
    assert document["definitions"]["controller_deviations"] == definition(NAME)
    solving = document["definitions"]["controller_solver_outcomes"]
    assert NAME not in solving and NAME not in document["definitions"]["timeouts"]
    summary = (out / compare_run.SUMMARY_NAME).read_text().splitlines()
    lines = [x for x in summary if x.startswith(LINE)]
    assert lines == [
        f"{LINE}{NAME}: hours planned {totals['hours']}, with least-laxity-first's split found "
        "infeasible and the total nearest the target found feasible planned "
        f"{totals['deviating_hours']} (in service hours "
        f"{totals['deviating_service_hours']}), with no feasible total "
        f"{totals['no_feasible_hours']}."
    ]
    at = next(k for k, x in enumerate(summary) if x.startswith("Timeouts"))
    assert summary[at + 1].startswith("Controller solves") and NAME not in summary[at + 1]
    assert summary[at + 2] == lines[0] and summary[at + 3] == ""
    assert summary[at + 4] == compare_summary.PAIRED_TEXT


def test_the_written_counts_are_the_controllers_counts_in_their_unit_and_variant(runs):
    document = _document(runs["with"][1])
    record = runs["record"]
    order = [(u, v) for u in _calibratable(document) for v in compare_run.VARIANTS]
    assert len(record) == len(order)
    deviating = {}
    for (unit, variant), calls in zip(order, record, strict=True):
        found = document["units"][unit]["variants"][variant]
        assert {u for u, _, _ in calls} == {unit}
        summed = dict.fromkeys(COUNTS, 0)
        for _, _, counts in calls:
            for key in COUNTS:
                summed[key] += counts[key]
        assert found["evaluations"]["controller_deviations"][NAME] == summed
        assert found["evaluations"]["controller_day_scale_pairs"] == len(calls)
        assert summed["hours"] == sum(8 * (n + 1) for _, n, _ in calls)
        assert len(calls) == found["offering_days"] * 10
        deviating[(unit, variant)] = summed["deviating_hours"]
    assert all(deviating[(u, "forecast")] > 0 for u in _calibratable(document)), deviating
    sums = [found for found in deviating.values()]
    assert len(set(sums)) > 1, deviating  # a swap of two places would show


def test_one_and_three_workers_write_the_same_bytes(runs):
    one, three = runs["with"][1], runs["with3"][1]
    assert runs["with3"][0] == 0
    for name in (compare_run.RESULTS_NAME, compare_run.SUMMARY_NAME, compare_run.SPLITS_NAME):
        assert (one / name).read_bytes() == (three / name).read_bytes(), name


def _rows(folder: Path, methods) -> list[dict]:
    with open(folder / compare_run.SPLITS_NAME, newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["method"] in methods]


def test_a_run_without_the_method_writes_none_of_its_fields_and_the_same_figures(runs):
    (code, bare), out = runs["without"], runs["with"][1]
    assert code == 0
    text = (bare / compare_run.RESULTS_NAME).read_text()
    assert "controller_deviations" not in text
    summary = (bare / compare_run.SUMMARY_NAME).read_text().splitlines()
    assert not any(LINE in x for x in summary)
    at = next(k for k, x in enumerate(summary) if x.startswith("Timeouts"))
    assert summary[at + 1].startswith("Controller solves") and summary[at + 2] == ""
    assert summary[at + 3] == compare_summary.PAIRED_TEXT
    without, document = _document(bare), _document(out)
    assert "controller_deviations" not in without["definitions"]
    assert {k: v for k, v in document["definitions"].items() if k != "controller_deviations"} == (
        without["definitions"]
    )
    for unit in _calibratable(document):
        entry, plain = document["units"][unit], without["units"][unit]
        for variant, found in plain["variants"].items():
            other = entry["variants"][variant]
            for method in WITHOUT:
                assert found["methods"][method] == other["methods"][method]
            evaluations = dict(other["evaluations"])
            del evaluations["controller_deviations"]
            assert found["evaluations"] == evaluations
        assert plain["gate_reading"] == {m: entry["gate_reading"][m] for m in WITHOUT}
        for variant in compare_run.VARIANTS:
            for tier in ("A", "B"):
                pairs = document["paired_differences"]["units"][unit][variant][tier]
                plain_pairs = without["paired_differences"]["units"][unit][variant][tier]
                kept = [row for row in pairs["against_re_solve"] if row["method"] != NAME]
                assert plain_pairs["against_re_solve"] == kept
                assert len(pairs["against_re_solve"]) == len(kept) + 1
    assert _rows(bare, WITHOUT) == _rows(out, WITHOUT)
    assert len(_rows(out, [NAME])) == len(_rows(out, ["llf"])) > 0


def test_the_definition_of_the_deviation_counts_word_for_word(build):
    """The definitions entry names the counting methods in the order given (the run's order), not
    sorted, and is written only when some method counts deviations."""
    matrix = gate_config.load(build[2])
    found = compare_run.definitions(matrix, SPLITS, (), [NAME])
    assert found["controller_deviations"] == definition(NAME)
    two = compare_run.definitions(matrix, SPLITS, (), ["zeta", NAME])
    assert two["controller_deviations"] == definition(f"zeta, {NAME}")
    assert "controller_deviations" not in compare_run.definitions(matrix, SPLITS, ["asa"])
    assert sorted(COUNTS) == KEYS and COUNTS == (
        "hours",
        "deviating_hours",
        "deviating_service_hours",
        "no_feasible_hours",
    )


def test_the_line_of_deviations_sums_each_count_into_its_own_place():
    def counts(base):
        return dict(zip(COUNTS, (base, base // 10, base // 100, base // 1000), strict=True))

    def variant(*named):
        return {"evaluations": {"controller_deviations": dict(named)}}

    document = {
        "methods": ["llf", "second", "first"],
        "units": {
            "u1": {"variants": {"forecast": variant(("first", counts(1000))), "oracle": {}}},
            "u2": {
                "variants": {
                    "forecast": variant(("first", counts(2000)), ("second", counts(5000))),
                    "oracle": variant(("first", counts(4000))),
                }
            },
        },
    }
    assert compare_summary._deviation_line(document) == [
        f"{LINE}second: hours planned 5000, with least-laxity-first's split found infeasible and "
        "the total nearest the target found feasible planned 500 (in service hours 50), with no "
        "feasible total 5; first: hours planned 7000, with least-laxity-first's split found "
        "infeasible and the total nearest the target found feasible planned 700 (in service "
        "hours 70), with no feasible total 7."
    ]
    bare = {"methods": ["llf"], "units": {"u1": {"variants": {"forecast": {"evaluations": {}}}}}}
    assert compare_summary._deviation_line(bare) == []


def _starts(line: str) -> list[int]:
    return [match.start() for match in re.finditer(r"\S+", line)]


def _tables(lines: list[str]):
    """Each table of the summary: its header (a line starting "  method ") and its rows (the
    lines after it that start with two spaces and a method name of the run)."""
    for k, line in enumerate(lines):
        if not line.startswith("  method "):
            continue
        rows = []
        for row in lines[k + 1 :]:
            if not row.startswith("  ") or row.split()[0] not in WITH:
                break
            rows.append(row)
        yield k, line, rows


@pytest.mark.parametrize("which", ["with", "without"])
def test_the_method_column_is_as_wide_as_the_longest_method_name(runs, which):
    lines = (runs[which][1] / compare_run.SUMMARY_NAME).read_text().splitlines()
    width = max(len(m) for m in (WITH if which == "with" else WITHOUT))
    column = 2 + max(width, compare_summary.METHOD_WIDTH) + 1
    tables = list(_tables(lines))
    assert len(tables) >= 2 * 3 + 2 * 2 * 2  # per unit: 3 tier and gate tables, 4 paired
    tiers = 0
    for k, header, rows in tables:
        assert rows
        paired = header.split()[1] == "reference"
        for line in (header, *rows):
            assert _starts(line)[1] == column, line
            if paired:
                assert _starts(line)[2] == 2 * column - 2, line
        if "slot loss" in header:
            assert lines[k - 1].split() == list(compare_run.VARIANTS)
            assert _starts(lines[k - 1])[0] == column
            tiers += 1
    assert tiers == 2 * 2
