"""The sizing methods `also-x` and `cvar` through the comparison command (`compare_run`), on
made-up gate units of one service hour (`gate_fake.gate_cars` without its early departures, then
the second car of every fourth day at one unit (28 days) and of every fifth day at another (24
days) leaving early, at 10.5, 10.9, 11.3 or 11.7 and at 10.7, 11.1 or 11.5 in turn, so that the
days fail at different scales; 2 days at the third unit, which cannot be split), 20 calibration
days, 4 splits, the dial 0.03. (Pairs of days with several service hours are in
`test_chance_sizing`; `measure_variant` with days of 2 to 4 service hours is in 7.)

1. A run of `re-solve`, `llf`, `also-x` and `cvar` ends with exit 0. Under each unit that can be
   split and each variant, `evaluations` has exactly the keys of a run without the two plus
   `sizing_solver_outcomes`, holding both methods with 4 calibration sets each (one per split),
   CVaR one optimal solve per set and ALSO-X more, none other than optimal; the timeouts hold
   `sizing_programmes` and `oracle_sizing_programmes` at 0; the definitions entry `sizing` names
   both and holds `chance_sizing.READINGS`; the summary has the line on sizing methods after the
   calibration rule and, right after the line of timeouts, the line of sizing solves whose
   figures are the written counts summed over units and variants.
2. The written scales are recomputed without the command's rule: from the re-solve method's
   losses of each unit and variant as the command measured them, on each split of
   `dial_choice.split_indices`, the thresholds of every (service hour, offering calibration day)
   pair, the closed forms of ALSO-X (the `(floor(0.1 N) + 1)`-th smallest) and CVaR (the mean of
   the smallest tenth), snapped to the grid, and for tier A the smaller of that and the driver
   scale of `crc`, give every split row's scale, and the row's offer is that scale times the
   mean certified value over the split's test days. Both methods offer on some split, and ALSO-X
   offers more than re-solve on some split.
3. The results, the summary and the split table are the same bytes with 1 and 3 workers.
4. A run of `re-solve` and `llf` alone writes no field, count, definition or line of the sizing
   methods (the word "sizing" is in none of its files), and its figures, evaluations, gate
   readings, split rows, definitions and pairs against `re-solve` are those of the run with them.
5. A run of `cvar` and `also-x` alone (no `re-solve`) writes their figures, split rows and counts
   as the run with `re-solve` does, the re-solve's losses measured once per variant.
6. A run whose every sizing programme times out ends with exit 0: both methods abstain on every
   split for `chance_sizing.NOT_OPTIMAL`, every solve is counted as a timeout under
   `sizing_solver_outcomes` and in `sizing_programmes` and `oracle_sizing_programmes`, and the
   other methods' figures are those of the run without the timeouts.
7. `measure_variant` on made-up prepared days of 2 to 4 service hours (`compare_fake.offer_days`)
   gives each sizing method the re-solve method's loss arrays and a rule of its own kind holding
   each day's number of service hours, and a run of the re-solve method alone gets no rule.
"""

import csv
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from compare_fake import NAMES, offer_days
from gate_fake import COMMIT, GATE_UNITS, SERVICE_HOURS, gate_cars, matrix_document, write_build

from fleetreserve import (
    chance_programmes,
    chance_sizing,
    compare_methods,
    compare_run,
    crc,
    dial_choice,
    dial_splits,
    gate_run,
    run_verify,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import load
from fleetreserve.lp import LpResult

SPLITS = 4
SIZING = ["also-x", "cvar"]
WITH = ["re-solve", "llf", *SIZING]
WITHOUT = ["re-solve", "llf"]
LINE = "Sizing solves (all units and variants together): "


def _early(cars, every: int, leaves) -> list:
    """`cars` with the second car of every `every`-th day leaving at `leaves` in turn."""
    found = list(cars)
    for k, car in enumerate(found):
        if k % every == 1:
            found[k] = dataclasses.replace(car, plug_out=leaves[(k // every) % len(leaves)])
    return found


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        GATE_UNITS[0]: gate_cars(2),
        GATE_UNITS[1]: _early(gate_cars(28, early_every=0), 4, (10.5, 10.9, 11.3, 11.7)),
        GATE_UNITS[2]: _early(gate_cars(24, "B", early_every=0), 5, (10.7, 11.1, 11.5)),
    }
    run, tables = write_build(folder, units)
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix


def _run(build, folder: Path, workers: int, methods, record=None, timeout=False):
    """One run; `record` (a list) receives the losses of every `measure_variant` call."""
    run, tables, matrix = build
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        if record is not None:
            measure = compare_run.measure_variant

            def measured(*args, **kwargs):
                found = measure(*args, **kwargs)
                record.append(found[0])
                return found

            monkeypatch.setattr(compare_run, "measure_variant", measured)
        if timeout:

            class Late(chance_programmes.SparseProgramme):
                def maximise(self, column, value, time_limit):
                    late = LpResult("timeout", None, None, 0.0, time_limit, 1, "Time limit")
                    return late, None

            monkeypatch.setattr(chance_programmes, "SparseProgramme", Late)
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
    found["record"] = record
    found["with3"] = _run(build, top / "with3", 3, WITH)
    found["without"] = _run(build, top / "without", 1, WITHOUT)
    found["only"] = _run(build, top / "only", 1, ["cvar", "also-x"])
    found["late"] = _run(build, top / "late", 1, WITH, timeout=True)
    return found


def _document(folder: Path) -> dict:
    return json.loads((folder / compare_run.RESULTS_NAME).read_text())


def _calibratable(document):
    return [u for u in GATE_UNITS if document["units"][u]["status"] == "calibratable"]


def _rows(folder: Path, methods) -> list[dict]:
    with open(folder / compare_run.SPLITS_NAME, newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["method"] in methods]


# --- 1. the fields of a run with the sizing methods ----------------------------------------------


def test_a_run_with_the_sizing_methods_writes_their_counts_and_lines(runs):
    (code, out), (_, bare) = runs["with"], runs["without"]
    assert code == 0
    document, without = _document(out), _document(bare)
    assert document["methods"] == WITH
    units = _calibratable(document)
    assert len(units) == 2
    totals = {m: {"calibration_sets": 0, "optimal": 0} for m in SIZING}
    for unit in units:
        for variant, found in document["units"][unit]["variants"].items():
            plain = without["units"][unit]["variants"][variant]["evaluations"]
            assert set(found["evaluations"]) == {*plain, "sizing_solver_outcomes"}
            counts = found["evaluations"]["sizing_solver_outcomes"]
            assert list(counts) == SIZING
            for method in SIZING:
                assert counts[method]["calibration_sets"] == SPLITS
                solves = counts[method]["solves"]
                assert set(solves) == set(chance_sizing.OUTCOMES)
                assert sum(solves.values()) == solves["optimal"]
                totals[method]["calibration_sets"] += SPLITS
                totals[method]["optimal"] += solves["optimal"]
            assert counts["cvar"]["solves"]["optimal"] == SPLITS
            assert counts["also-x"]["solves"]["optimal"] > 2 * SPLITS
    assert document["timeouts"]["sizing_programmes"] == 0
    assert document["timeouts"]["oracle_sizing_programmes"] == 0
    sizing = document["definitions"]["sizing"]
    assert sizing["readings"] == chance_sizing.READINGS
    assert sizing["methods"].startswith("also-x, cvar: the re-solve method's day-level losses")
    assert "also-x, cvar" in document["definitions"]["timeouts"]
    summary = (out / compare_run.SUMMARY_NAME).read_text().splitlines()
    rule = next(k for k, x in enumerate(summary) if x.startswith("Calibration days per unit"))
    assert summary[rule + 1] == (
        f"Sizing methods, {sizing['methods']}. Readings S1 to S11 in the results "
        "(definitions.sizing.readings)."
    )
    at = next(k for k, x in enumerate(summary) if x.startswith("Timeouts"))
    assert summary[at + 1] == (
        f"{LINE}also-x: calibration sets {totals['also-x']['calibration_sets']}; solves optimal "
        f"{totals['also-x']['optimal']}, infeasible 0, unbounded 0, timeout 0, error 0; cvar: "
        f"calibration sets {totals['cvar']['calibration_sets']}; solves optimal "
        f"{totals['cvar']['optimal']}, infeasible 0, unbounded 0, timeout 0, error 0."
    )
    assert summary[at + 2] == ""


# --- 2. the scales recomputed --------------------------------------------------------------------


def _thresholds(slot, hours):
    """Every (service hour, day) pair's threshold, written apart from `chance_sizing`."""
    grid = list(dial_splits.GRID)
    found = []
    for row in slot:
        for i in range(1, hours + 1):
            lost = [k for k, loss in enumerate(row) if round(loss * hours) >= i]
            found.append(grid[-1] if not lost else (0.0 if lost[0] == 0 else grid[lost[0] - 1]))
    return np.sort(found)


def _closed(kind, slot, hours):
    theta = _thresholds(slot, hours)
    share = 0.1 * theta.size
    if kind == "also-x":
        value, tolerance = theta[math.floor(share + 1e-9)], 1e-4
    else:
        whole = math.floor(share)
        part = (share - whole) * theta[whole] if whole < theta.size else 0.0
        value, tolerance = (theta[:whole].sum() + part) / share, 1e-6
    below = [g for g in dial_splits.GRID if g <= value + tolerance]
    return max(below) if below else 0.0


def _loaded(build):
    _, tables, matrix_file = build
    matrix = load(matrix_file)
    return matrix, {
        name: load_unit_days(
            tables, name, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        for name in matrix.gate_units
    }


def test_the_written_scales_are_the_closed_forms_on_the_re_solve_losses(runs, build):
    document = _document(runs["with"][1])
    matrix, loaded = _loaded(build)
    order = [(u, v) for u in _calibratable(document) for v in compare_run.VARIANTS]
    assert len(runs["record"]) == len(order)
    rows = {
        (r["unit"], r["variant"], r["tier"], r["method"], int(r["seed"])): r
        for r in _rows(runs["with"][1], WITH)
    }
    offered, above = set(), set()
    for (unit, variant), losses in zip(order, runs["record"], strict=True):
        resolve = losses["re-solve"]
        splits = dial_choice.split_indices(unit, loaded[unit], matrix, range(1, SPLITS + 1))
        for seed, calibration, test in splits:
            used = [d for d in calibration if resolve.offering[d]]
            for tier in dial_splits.TIERS:
                for kind in SIZING:
                    row = rows[(unit, variant, tier, kind, seed)]
                    expected = 0.0
                    if len(used) >= crc.min_days(crc.ALPHA_SLOT):
                        expected = _closed(kind, resolve.slot[used], SERVICE_HOURS)
                    if tier == "A" and expected > 0:
                        driver, _ = dial_splits.on_grid_scale(
                            resolve.slot[used], resolve.driver[used], 1.0, crc.ALPHA_DRIVER
                        )
                        expected = min(expected, driver)
                    assert float(row["scale"]) == expected, (unit, variant, tier, kind, seed)
                    offer = expected * resolve.value[test].mean()
                    assert float(row["offer_kwh_per_counting_test_day"]) == pytest.approx(offer)
                    if expected > 0:
                        offered.add(kind)
                    if expected > float(rows[(unit, variant, tier, "re-solve", seed)]["scale"]):
                        above.add(kind)
    assert offered == set(SIZING) and "also-x" in above, (offered, above)


# --- 3. workers ----------------------------------------------------------------------------------


def test_one_and_three_workers_write_the_same_bytes(runs):
    one, three = runs["with"][1], runs["with3"][1]
    assert runs["with3"][0] == 0
    for name in (compare_run.RESULTS_NAME, compare_run.SUMMARY_NAME, compare_run.SPLITS_NAME):
        assert (one / name).read_bytes() == (three / name).read_bytes(), name


# --- 4. a run without the sizing methods ---------------------------------------------------------


def test_a_run_without_the_sizing_methods_writes_none_of_their_fields_and_the_same_figures(runs):
    (code, bare), out = runs["without"], runs["with"][1]
    assert code == 0
    for name in (compare_run.RESULTS_NAME, compare_run.SUMMARY_NAME, compare_run.SPLITS_NAME):
        assert "sizing" not in (bare / name).read_text().lower(), name
    without, document = _document(bare), _document(out)
    definitions = dict(document["definitions"])
    del definitions["sizing"]
    assert definitions["timeouts"] != without["definitions"]["timeouts"]
    del definitions["timeouts"]
    assert definitions == {k: v for k, v in without["definitions"].items() if k != "timeouts"}
    assert without["timeouts"] == {
        k: v for k, v in document["timeouts"].items() if "sizing" not in k
    }
    for unit in _calibratable(document):
        entry, plain = document["units"][unit], without["units"][unit]
        for variant, found in plain["variants"].items():
            other = entry["variants"][variant]
            for method in WITHOUT:
                assert found["methods"][method] == other["methods"][method]
            evaluations = dict(other["evaluations"])
            del evaluations["sizing_solver_outcomes"]
            assert found["evaluations"] == evaluations
        assert plain["gate_reading"] == {m: entry["gate_reading"][m] for m in WITHOUT}
        for variant in compare_run.VARIANTS:
            for tier in dial_splits.TIERS:
                pairs = document["paired_differences"]["units"][unit][variant][tier]
                plain_pairs = without["paired_differences"]["units"][unit][variant][tier]
                kept = [r for r in pairs["against_re_solve"] if r["method"] not in SIZING]
                assert plain_pairs["against_re_solve"] == kept
                assert len(pairs["against_re_solve"]) == len(kept) + 2
    assert _rows(bare, WITHOUT) == _rows(out, WITHOUT)
    assert len(_rows(out, ["also-x"])) == len(_rows(out, ["cvar"])) == len(_rows(out, ["llf"]))


# --- 5. the sizing methods without re-solve ------------------------------------------------------


def test_the_sizing_methods_alone_measure_the_re_solve_losses_themselves(runs):
    (code, only), out = runs["only"], runs["with"][1]
    assert code == 0
    alone, document = _document(only), _document(out)
    assert alone["methods"] == ["cvar", "also-x"]
    for unit in _calibratable(document):
        for variant, found in alone["units"][unit]["variants"].items():
            other = document["units"][unit]["variants"][variant]
            for method in SIZING:
                assert found["methods"][method] == other["methods"][method]
            sized = found["evaluations"]["sizing_solver_outcomes"]
            assert sized == other["evaluations"]["sizing_solver_outcomes"]
            pairs = found["evaluations"]["re_solve_day_scale_pairs"]
            assert pairs == other["evaluations"]["re_solve_day_scale_pairs"] > 0

    def key(row):
        return row["unit"], row["variant"], row["tier"], row["method"], int(row["seed"])

    assert sorted(_rows(only, SIZING), key=key) == sorted(_rows(out, SIZING), key=key)


# --- 6. every sizing programme timing out --------------------------------------------------------


def test_sizing_programmes_that_time_out_abstain_and_are_counted(runs):
    (code, late), out = runs["late"], runs["with"][1]
    assert code == 0
    document, normal = _document(late), _document(out)
    total = 0
    for unit in _calibratable(document):
        for variant, found in document["units"][unit]["variants"].items():
            other = normal["units"][unit]["variants"][variant]
            for method in SIZING:
                for tier in dial_splits.TIERS:
                    summary = found["methods"][method][tier]
                    assert summary["splits_that_offer"] == 0
                    reasons = summary["splits_at_scale_zero_by_reason"]
                    assert reasons == {chance_sizing.NOT_OPTIMAL: SPLITS}
            for method in WITHOUT:
                assert found["methods"][method] == other["methods"][method]
            counts = found["evaluations"]["sizing_solver_outcomes"]
            for method in SIZING:
                solves = counts[method]["solves"]
                assert solves["timeout"] == sum(solves.values()) > 0
                total += solves["timeout"]
    timeouts = document["timeouts"]
    assert timeouts["sizing_programmes"] + timeouts["oracle_sizing_programmes"] == total
    assert timeouts["sizing_programmes"] > 0 and timeouts["oracle_sizing_programmes"] > 0
    assert {r["scale"] for r in _rows(late, SIZING)} == {"0.0"}


# --- 7. measure_variant --------------------------------------------------------------------------


def test_measure_variant_gives_the_sizing_methods_the_re_solve_losses_and_the_days_hours():
    prepared = offer_days(4)
    hours = [p.day.n_service for p in prepared]
    assert len(set(hours)) > 1
    methods = compare_methods.select(["also-x", "cvar"])

    def evaluate(pairs):
        return [compare_run._evaluate_task(p) for p in pairs]

    found, counts = compare_run.measure_variant(prepared, [None] * 4, methods, evaluate, None)
    assert list(found) == ["also-x", "cvar"]
    assert counts["re_solve_day_scale_pairs"] > 0
    alone, _ = compare_run.measure_variant(
        prepared, [None] * 4, compare_methods.select(["re-solve"]), evaluate, None
    )
    resolve = alone["re-solve"]
    assert resolve.rule is None
    for name in ("also-x", "cvar"):
        losses = found[name]
        assert losses.rule.kind == name
        assert losses.rule.service_hours.tolist() == hours
        for field in ("value", "offering", "slot", "driver", "ceiling"):
            assert np.array_equal(getattr(losses, field), getattr(resolve, field), equal_nan=True)
