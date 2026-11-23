"""The corner-search command (`corner_run`) on the made-up gate units of `gate_fake` (one service
hour) and on `corner_fake.early_units` (more early departures).

1. A run on the gate units at the dial 0.03 ends with exit 0 and writes the four files; both
   variants and every method are given for each unit that can be split; the unit that cannot be
   split has no numbers; a unit with fewer offering test days than asked says so; one CSV row per
   unit, variant, method and searched day; the path-hour counts are in the timing file. The
   default list searches exactly the methods that solve programmes (`solving_methods`: `asa`,
   `asa-es` and `diagonal-split` among them) with their solver counts kept per method (search;
   `diagonal-split` one programme and one box per day) and per calibration
   (`controller_solver_outcomes`), with their timeouts in the timeouts, and no other method keeps
   counts; the sizing methods (reading W5) are searched with no failure recorded, their sizing
   programmes counted per calibration (`sizing_solver_outcomes`) and their timeout keys present.
   Every field written at every level (document, unit, variant, method, pass-mark check,
   disagreement, CSV column) has an entry under `definitions["fields"]`, here, on `early_units`,
   with a disagreement and with a failed variant; and every entry there is of a field these runs
   write, the fields of a disagreement, of a failed variant and of a rotation what-if each in its
   own case (`SITUATIONAL`).
2. The results, the rows and the summary are byte-identical for 1, 2 and 3 workers.
3. Each method's scale is the tier A scale `compare_run` calibrates on the same split (its run
   with one split, seed 1), in both variants.
4. The searched days are the first offering days of the seeded order (`day_order`), which is a
   permutation of the split's test days fixed by unit and seed; another seed gives another order.
   When some test days abstain, the searched days are the first offering days of that order.
5. The pass mark is checked on the first searched days, in order, and its path-hours count
   those days only.
6. On `early_units` the closed form and the corner worst case differ at scale 1 by rounding
   ("within tolerance"): they agree with the tolerance and not strictly (reading K3), exit 0,
   both counts recorded and printed. An "unexplained" difference (made by replacing one
   check per day) is a disagreement: exit 1. Nothing checked, or no unit that can be split,
   exits 1. Every row on `early_units` equals `corner_search.search_day` of the day as the
   row's variant prepares it, at the row's scale.
7. Exit 2 for an unknown model, a `--dial` not in the dial family, a missing truth build, an
   unknown method, and a number of workers, days or a seed below 1; exit 1 when one variant
   fails (recorded; the other variant still runs).
"""

import csv
import dataclasses
import json
from pathlib import Path

import pytest
from corner_fake import NAMES, early_units, solving_methods
from gate_fake import COMMIT, GATE_UNITS, gate_units, matrix_document, write_build

from fleetreserve import (
    compare_methods,
    compare_run,
    corner_run,
    corner_search,
    dial_choice,
    gate_day,
    gate_run,
    learn_run,
    run_verify,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import load

METHODS = [m.name for m in compare_methods.METHODS]
SOLVING = solving_methods()  # the methods whose controllers solve programmes
SIZING = [m.name for m in compare_methods.METHODS if m.kind == "sizing"]
SPLIT_UNITS = GATE_UNITS[1:]
FILES = (corner_run.RESULTS_NAME, corner_run.ROWS_NAME, corner_run.SUMMARY_NAME)


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


def _build(folder: Path, units: dict):
    folder.mkdir()
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return run, tables, matrix


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("corner") / "plain", gate_units())


@pytest.fixture(scope="module")
def early(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("corner") / "early", early_units())


def _arguments(build):
    run, tables, matrix = build
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]


def _run(build, folder: Path, *extra, main=corner_run.main):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        folder.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        arguments = ["--model", "empirical", "--dial", "0.03", *_arguments(build), *extra]
        return main(arguments), folder


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    top = tmp_path_factory.mktemp("runs")
    return {w: _run(build, top / f"w{w}", "--workers", str(w)) for w in (1, 2, 3)}


@pytest.fixture(scope="module")
def early_run(early, tmp_path_factory):
    return _run(early, tmp_path_factory.mktemp("early") / "out", "--workers", "2")


def _document(folder: Path) -> dict:
    return json.loads((folder / corner_run.RESULTS_NAME).read_text())


def _rows(folder: Path) -> list[dict]:
    with (folder / corner_run.ROWS_NAME).open() as handle:
        return list(csv.DictReader(handle))


# --- 1. a full run -------------------------------------------------------------------------------


def test_a_run_writes_the_four_files_with_both_variants_and_every_method(runs):
    code, out = runs[1]
    assert code == 0
    for name in (*FILES, corner_run.TIMING_NAME):
        assert (out / name).is_file()
    document = _document(out)
    assert document["failed"] == [] and document["methods"] == METHODS
    assert document["pass_mark"]["agreed"] is True
    assert document["pass_mark"]["pairs"] == document["pass_mark"]["pairs_agreed"] > 0
    assert set(document["definitions"]["readings"]) == {
        "W1",
        "W2",
        "W3",
        "W4",
        "K1",
        "K2",
        "K3",
        "K4",
        "K5",
        "W5",  # the registry holds sizing methods
    }
    assert document["run"]["seed"] == 1 and document["run"]["days"] == corner_run.DAYS
    searched = set()
    for unit in SPLIT_UNITS:
        entry = document["units"][unit]
        assert set(entry["variants"]) == set(corner_run.VARIANTS)
        for variant in entry["variants"].values():
            assert set(variant["methods"]) == set(METHODS)
            assert variant["days_searched"] == variant["offering_test_days"] == 10
            assert variant["days_searched"] < variant["days_asked"]
            for name, method in variant["methods"].items():
                if method["status"] == "no offer":
                    assert method["scale"] == 0.0 and "share_of_days_breaking" not in method
                    continue
                assert method["days_with_result"] == 10
                counts = method["solver_counts"]
                if name in SOLVING:  # counted on every corner of every day
                    searched.add(name)
                    assert counts["hours"] == 2 * 10 * 2  # 2 corners, 10 days, hours 0 and 1
                    assert sum(counts["solves"].values()) > 0
                    if "boxes" in counts:  # one programme and one box per day and scale
                        assert sum(counts["boxes"].values()) == sum(counts["solves"].values())
                        assert sum(counts["solves"].values()) == 10
                else:
                    assert counts is None
            solved = variant["calibration_evaluations"].get("controller_solver_outcomes", {})
            assert set(solved) <= SOLVING
            sized = variant["calibration_evaluations"]["sizing_solver_outcomes"]
            assert list(sized) == SIZING  # reading W5: each sizing method's programmes
            for counts in sized.values():
                assert counts["calibration_sets"] in (0, 1)  # one calibration set, if solved
                assert (sum(counts["solves"].values()) > 0) == (counts["calibration_sets"] == 1)
    # the default list searches every method that solves programmes (by the rule corner_search
    # reads) and keeps its counts; every method that does not keep solver counts has none
    # (asserted above)
    assert searched == SOLVING
    assert {"asa", "asa-es", "diagonal-split"} <= SOLVING
    controllers = {m.name for m in compare_methods.METHODS if m.kind == "controller"}
    assert controllers > SOLVING  # some controller keeps no solver counts
    assert {"controller_solves_calibration", "controller_solves_search"} <= set(
        document["timeouts"]
    )
    assert document["timeouts"]["sizing_programmes_calibration"] == 0
    assert document["timeouts"]["oracle_sizing_programmes_calibration"] == 0
    rows = _rows(out)
    assert len(rows) == len(SPLIT_UNITS) * len(corner_run.VARIANTS) * len(METHODS) * 10
    assert list(rows[0]) == list(corner_search_fields())
    text = (out / corner_run.SUMMARY_NAME).read_text()
    assert "fewer than the 50 asked: the unit has no more" in text
    timing = json.loads((out / corner_run.TIMING_NAME).read_text())
    for unit in SPLIT_UNITS:
        for variant in corner_run.VARIANTS:
            hours = timing["path_hours"][unit][variant]
            assert set(hours) == {"calibration", "search_upper_bound", "search", "pass_mark"}
            # one service hour: 2 corners a day, 10 searched days, every method of the registry,
            # 10 check scales
            assert hours["search_upper_bound"] == 2 * 10 * len(METHODS)
            assert hours["pass_mark"] == 2 * 10 * 10


def corner_search_fields():
    from fleetreserve.corner_summary import ROW_FIELDS

    return ROW_FIELDS


def _undefined(document: dict, rows) -> set:
    """The written fields without an entry in `definitions["fields"]`, by level: the document,
    its `run`, `pass_mark` and `matrix`, each unit, variant, method, pass-mark check and
    disagreement, and each CSV column."""
    return _written(document, rows) - set(document["definitions"]["fields"])


# The defined fields that a run writes only in their own case: a disagreement's fields (written in
# test_a_disagreement_exits_1_and_is_recorded), a failed variant's error (written in
# test_a_failed_variant_is_recorded_and_the_other_still_runs) and the label of a rotation what-if
# (no test of this command makes one).
SITUATIONAL = {
    *(
        f"pass_mark.disagreements.{k}"
        for k in (
            "day",
            "date",
            "scale",
            "closed_form_first_failure",
            "corner_first_failure",
            "closed_form_slot_loss",
            "corner_worst_slot_loss",
            "reason",
        )
    ),
    "variant.failed",
    "document.rotation_whatif",
}


def _written(document: dict, rows) -> set:
    """The fields written, by level (`_undefined`)."""
    written = {f"document.{k}" for k in document}
    written |= {f"document.run.{k}" for k in document["run"]}
    written |= {f"document.pass_mark.{k}" for k in document["pass_mark"]}
    written |= {f"document.matrix.{k}" for k in document.get("matrix", {})}
    for unit in document["units"].values():
        written |= {f"unit.{k}" for k in unit}
        for variant in unit["variants"].values():
            written |= {f"variant.{k}" for k in variant}
            for method in variant.get("methods", {}).values():
                written |= {f"method.{k}" for k in method}
            check = variant.get("pass_mark", {})
            written |= {f"pass_mark.{k}" for k in check}
            for wrong in check.get("disagreements", []):
                written |= {f"pass_mark.disagreements.{k}" for k in wrong}
    written |= {f"row.{k}" for row in rows for k in row}
    return written


def test_every_written_field_has_a_definition(runs, early_run):
    written = set()
    for _, out in (runs[1], early_run):
        document = _document(out)
        assert _undefined(document, _rows(out)) == set()
        written |= _written(document, _rows(out))
    # and every definition is of a field these runs write, but for the fields of their own case
    assert set(document["definitions"]["fields"]) - written == SITUATIONAL
    seen = {
        f"variant.{k}"
        for k in _document(runs[1][1])["units"][SPLIT_UNITS[0]]["variants"]["forecast"]
    }
    assert "variant.searched_days" in seen and "variant.calibration_evaluations" in seen


def test_the_unit_that_cannot_be_split_has_no_numbers(runs):
    _, out = runs[1]
    entry = _document(out)["units"][GATE_UNITS[0]]
    assert entry["status"] == "not calibratable: 10 days" and entry["variants"] == {}
    block = (out / corner_run.SUMMARY_NAME).read_text().split(f"{GATE_UNITS[0]} (")[1]
    assert block.split("\n\n")[0].endswith("no numbers: the unit cannot be split")
    assert not any(r["unit"] == GATE_UNITS[0] for r in _rows(out))


# --- 2. workers ----------------------------------------------------------------------------------


def test_the_files_are_byte_identical_for_one_two_and_three_workers(runs):
    assert {w: code for w, (code, _) in runs.items()} == {1: 0, 2: 0, 3: 0}
    for name in FILES:
        first = (runs[1][1] / name).read_bytes()
        assert (runs[2][1] / name).read_bytes() == first
        assert (runs[3][1] / name).read_bytes() == first


# --- 3. the scales are compare_run's -------------------------------------------------------------


def test_each_methods_scale_is_compare_runs_tier_a_scale_on_the_split(build, runs, tmp_path):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        out = tmp_path / "compare"
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        arguments = ["--model", "empirical", "--dial", "0.03", *_arguments(build), "--splits", "1"]
        assert compare_run.main(arguments) == 0
    theirs = json.loads((out / compare_run.RESULTS_NAME).read_text())
    ours = _document(runs[1][1])
    seen = set()
    for unit in SPLIT_UNITS:
        for variant in corner_run.VARIANTS:
            for method in METHODS:
                scale = ours["units"][unit]["variants"][variant]["methods"][method]["scale"]
                found = theirs["units"][unit]["variants"][variant]["methods"][method]["A"]
                assert scale == found["mean_scale"]
                seen.add(scale)
    assert len(seen) > 1  # the variants differ here: 0 under forecasts, above 0 with oracle


# --- 4. the seeded order -------------------------------------------------------------------------


def test_the_day_order_is_a_seeded_permutation():
    days = list(range(3, 40, 3))
    first = corner_run.day_order("unit", 1, days)
    assert sorted(first) == days and first != days
    assert corner_run.day_order("unit", 1, days) == first
    assert corner_run.day_order("unit", 2, days) != first
    assert corner_run.day_order("other", 1, days) != first


def _split(build, unit, seed):
    _, tables, matrix_file = build
    matrix = load(matrix_file)
    days = load_unit_days(
        tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )
    ((_, _, test),) = dial_choice.split_indices(unit, days, matrix, [seed])
    return test


@pytest.mark.parametrize("seed", (1, 2))
def test_the_searched_days_are_the_first_of_the_seeded_order(build, tmp_path, seed):
    code, out = _run(
        build, tmp_path / "few", "--days", "4", "--seed", str(seed), "--methods", "edf"
    )
    assert code == 0
    document = _document(out)
    for unit in SPLIT_UNITS:
        order = corner_run.day_order(unit, seed, _split(build, unit, seed))
        for variant in document["units"][unit]["variants"].values():
            assert variant["offering_test_days"] == 10  # every test day offers here
            assert variant["searched_days"] == order[:4]
            assert variant["days_searched"] == 4


# --- 5. the pass mark's days ---------------------------------------------------------------------


def test_the_pass_mark_reads_the_first_searched_days(build, tmp_path, monkeypatch):
    monkeypatch.setattr(corner_run, "CHECK_DAYS", 3)
    code, out = _run(build, tmp_path / "check", "--methods", "edf")
    assert code == 0
    document = _document(out)
    assert document["run"]["check_days"] == 3
    for unit in SPLIT_UNITS:
        for variant in document["units"][unit]["variants"].values():
            assert variant["pass_mark"]["days"] == variant["searched_days"][:3]
            assert variant["pass_mark"]["pairs"] == 3 * 10
    timing = json.loads((out / corner_run.TIMING_NAME).read_text())
    for unit in SPLIT_UNITS:
        for hours in timing["path_hours"][unit].values():
            assert hours["pass_mark"] == 2 * 3 * 10  # 2 corners, 3 checked days, 10 scales


# --- 6. a disagreement, and every row recomputed ------------------------------------------------


def test_a_rounding_difference_agrees_within_tolerance_and_exits_0(early_run):
    code, out = early_run
    assert code == 0
    document = _document(out)
    pass_mark = document["pass_mark"]
    assert document["failed"] == [] and pass_mark["agreed"] is True
    assert pass_mark["pairs_agreed_strict"] < pass_mark["pairs_agreed"] == pass_mark["pairs"]
    reasons = set()
    for unit in SPLIT_UNITS:
        found = document["units"][unit]["variants"]["forecast"]["pass_mark"]
        assert found["agreed"] is True and found["disagreements"] == []
        assert found["pairs_agreed_strict"] < found["pairs_agreed"] == found["pairs"]
        reasons |= set(found["differences_by_reason"])
    assert reasons == {"within tolerance"}
    text = (out / corner_run.SUMMARY_NAME).read_text()
    assert "pair(s) differ, agree within tolerance: within tolerance" in text
    assert (
        f"{pass_mark['pairs_agreed_strict']} of {pass_mark['pairs']} pairs agree strictly, "
        f"{pass_mark['pairs_agreed']} with the tolerance: agreed." in text
    )


def test_a_disagreement_exits_1_and_is_recorded(build, tmp_path, monkeypatch):
    original = corner_search.closed_form_check

    def disagreeing(prepared, betas):
        found = original(prepared, betas)
        wrong = dataclasses.replace(
            found[-1], agreed_strict=False, agreed=False, reason="unexplained"
        )
        return (*found[:-1], wrong)

    monkeypatch.setattr(corner_search, "closed_form_check", disagreeing)
    code, out = _run(build, tmp_path / "disagreeing", "--methods", "edf")
    assert code == 1
    document = _document(out)
    assert document["failed"] == [] and document["pass_mark"]["agreed"] is False
    for unit in SPLIT_UNITS:
        found = document["units"][unit]["variants"]["forecast"]["pass_mark"]
        assert found["disagreements_by_reason"] == {"unexplained": found["days_checked"]}
        assert found["pairs_agreed"] == found["pairs"] - found["days_checked"]
    assert _undefined(document, _rows(out)) == set()  # the disagreements' fields too
    disagreement = {k for k in SITUATIONAL if k.startswith("pass_mark.disagreements.")}
    assert disagreement <= _written(document, _rows(out))
    text = (out / corner_run.SUMMARY_NAME).read_text()
    assert "DISAGREED" in text and "pair(s) differ, disagree: unexplained" in text


def test_nothing_checked_exits_1(build, tmp_path, monkeypatch):
    monkeypatch.setattr(corner_run, "CHECK_DAYS", 0)
    code, out = _run(build, tmp_path / "none", "--methods", "edf")
    assert code == 1
    document = _document(out)
    assert document["failed"] == [] and document["pass_mark"]["pairs"] == 0
    assert document["pass_mark"]["agreed"] is False
    assert "nothing checked" in (out / corner_run.SUMMARY_NAME).read_text()


def test_no_unit_that_can_be_split_exits_1(build, tmp_path, monkeypatch):
    original = gate_run.plan_unit

    def unsplittable(name, days, matrix):
        return dataclasses.replace(original(name, days, matrix), status="not calibratable: made up")

    monkeypatch.setattr(gate_run, "plan_unit", unsplittable)
    code, out = _run(build, tmp_path / "unsplit", "--methods", "edf")
    assert code == 1
    document = _document(out)
    assert document["failed"] == [] and document["pass_mark"]["pairs"] == 0
    assert all(e["variants"] == {} for e in document["units"].values())


def test_the_days_searched_skip_the_days_that_abstain(build, tmp_path, monkeypatch):
    original = gate_day.prepare_day

    def prepare(truth, forecaster, configuration, settings, no_pause=()):
        assert no_pause == ()  # the whole fleet: no car drawn
        found = original(truth, forecaster, configuration, settings, no_pause=no_pause)
        if found.date.endswith(("1", "3", "5", "7", "9")):
            return dataclasses.replace(found, abstain_reason="made-up abstention")
        return found

    monkeypatch.setattr(gate_day, "prepare_day", prepare)
    code, out = _run(build, tmp_path / "abstain", "--days", "4", "--methods", "edf")
    document = _document(out)
    assert document["failed"] == []
    prepared = _prepared_days(build)
    skipped = 0
    for unit in SPLIT_UNITS:
        order = corner_run.day_order(unit, 1, _split(build, unit, 1))
        for name, variant in document["units"][unit]["variants"].items():
            dates = [prepared[unit][name][d].date for d in order]
            offering = [d for d, date in zip(order, dates, strict=True) if date[-1] not in "13579"]
            assert variant["offering_test_days"] == len(offering) < len(order)
            assert variant["searched_days"] == offering[:4]
            skipped += offering[:4] != order[:4]
    assert skipped > 0
    assert code in (0, 1)


def _prepared_days(build):
    """unit -> variant -> every counting day prepared as the command prepares it."""
    _, tables, matrix_file = build
    matrix = load(matrix_file)
    configurations = compare_run.dial_configurations(matrix, 0.03)
    model = learn_run.load_model("empirical", matrix)
    units = {
        n: load_unit_days(
            tables, n, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        for n in matrix.gate_units
    }
    factory, _ = dial_choice.make_factory("empirical", model, matrix, units, {}, 0.3)
    pool = compare_run.Pool(1, units, matrix, factory)
    return {
        unit: {
            variant: pool.map(
                compare_run._prepare_task, corner_run.prepare_jobs(unit, len(units[unit].days), c)
            )
            for variant, c in configurations.items()
        }
        for unit in SPLIT_UNITS
    }


def _text(value) -> str:
    return "" if value is None else repr(value) if isinstance(value, float) else str(value)


def test_every_row_is_the_search_of_its_variants_day(early, early_run):
    _, out = early_run
    prepared = _prepared_days(early)
    methods = {m.name: m for m in compare_methods.METHODS}
    statuses = set()
    for row in _rows(out):
        day = prepared[row["unit"]][row["variant"]][int(row["day"])]
        found = corner_search.search_day(day, methods[row["method"]], float(row["scale"]))
        statuses.add(found.status)
        assert row["status"] == found.status
        assert row["breaking_paths"] == str(found.breaking_paths)
        assert row["worst_slot_loss"] == _text(found.worst_slot_loss)
        assert row["worst_driver_loss"] == _text(found.worst_driver_loss)
        assert row["audited_status"] == _text(found.audited_status)
    assert {"breaks", "holds", "no offer"} <= statuses


# --- 7. refusals and a failed variant ------------------------------------------------------------


def _code(build, *extra, model="empirical", dial="0.03"):
    return corner_run.main(["--model", model, "--dial", dial, *_arguments(build), *extra])


def test_refusals_could_not_run(build, tmp_path, capsys):
    assert _code(build, model="no_such_model") == 2
    assert _code(build, dial="0.05") == 2
    assert _code(build, "--methods", "edf", "no-such-method") == 2
    for flag in ("--workers", "--days", "--seed"):
        assert _code(build, flag, "0") == 2
    arguments = [*_arguments(build)]
    arguments[arguments.index("--truth-run") + 1] = str(tmp_path / "20260101T000000Z-wp2-truth")
    assert corner_run.main(["--model", "empirical", "--dial", "0.03", *arguments]) == 2
    printed = capsys.readouterr().out
    assert printed.count("COULD NOT RUN") == 7
    assert "no model module" in printed and "is not a dial of the matrix's dial family" in printed
    assert "unknown method 'no-such-method'" in printed and "--seed must be at least 1" in printed
    assert "truth-summary.json: the file is missing" in printed


def test_a_failed_variant_is_recorded_and_the_other_still_runs(build, monkeypatch, tmp_path):
    original = gate_day.prepare_day

    def prepare(truth, forecaster, configuration, settings, no_pause=()):
        if configuration.departure == "oracle":
            raise RuntimeError("made-up failure")
        return original(truth, forecaster, configuration, settings, no_pause=no_pause)

    monkeypatch.setattr(gate_day, "prepare_day", prepare)
    code, out = _run(build, tmp_path / "failed", "--methods", "fixed-plan")
    assert code == 1
    document = _document(out)
    assert document["failed"] == sorted(
        f"{unit} | oracle-departure: RuntimeError: made-up failure" for unit in SPLIT_UNITS
    )
    for unit in SPLIT_UNITS:
        variants = document["units"][unit]["variants"]
        assert variants["oracle-departure"]["failed"].endswith("made-up failure")
        assert variants["forecast"]["pass_mark"]["agreed"] is True
    assert _undefined(document, _rows(out)) == set()  # a failed variant's fields too
    assert "variant.failed" in _written(document, _rows(out))
    text = (out / corner_run.SUMMARY_NAME).read_text()
    assert "oracle-departure: failed (RuntimeError: made-up failure)" in text
    assert "FAILED: " in text
