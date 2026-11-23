"""The comparison command (`compare_run`) with its default method list, which holds the two
linear-programme forms of ASA (`asa` and `asa-es`) and `diagonal-split`, on small made-up gate
units built with `gate_fake`: 2, 3 and 3 days (the pooled unit from two sites) with 2 calibration
days, so that the first unit cannot be split. The units are this small because `asa` and `asa-es`
solve four and five programmes for each hour of each path and the command evaluates every
offering day at 10 scales on 8 paths under two variants; `test_compare_command` runs the command
without them on the larger units of `gate_fake`. The methods that solve programmes (`SOLVING`)
are those whose controller has a method `solver_counts` (`compare_fake.solving_methods`): `asa`,
`asa-es` and `diagonal-split`.

1. A run without `--methods` at the dial 0.03 ends with exit 0 and runs every method of the
   registry, in the registry's order (`methods` of the results is `compare_methods.METHODS` by
   name, `asa-es` right after `asa`); each unit that can be split has the rows of every method,
   both tiers, in both variants, and a gate reading for every method; the unit that cannot be
   split has its status and no numbers; the paired differences hold, per unit that can be split,
   variant and tier, every method but `re-solve` against `re-solve` and every two of those
   methods as other pairs, `asa-es` with `asa` as reference among them.
2. Such a run writes the results, the summary, the forecast table and the split table
   byte-identical for 1, 2 and 3 workers.
3. Every unit and variant that ran carries `controller_solver_outcomes` for the methods of
   `SOLVING` only, in the registry's order (every solve optimal on these units, no hour planned by
   the fall-back; four solves per hour for `asa`, five for `asa-es`, one per (day, scale) pair for
   `diagonal-split` with one box per pair, and the same hours for all three), the definitions name
   them, and the summary has one line of controller solves whose numbers are the sums over units
   and variants.
4. A run of every method that solves no programme (`--methods` with the methods not in
   `SOLVING` and not of the kind "sizing", whose rules solve programmes of their own and are
   tested in `test_sizing_command`) writes none of the fields added for the solver counts: no
   `controller_solver_outcomes` under any `evaluations`, no definitions entry of that name, the
   `timeouts` definition without `asa` word for word, and no summary line of controller solves;
   its rows for those methods are those of the default run.
5. A day whose certificate times out under the oracle-departure variant (the first day of the
   second unit that offers under it) offers nothing for every method of that variant, `asa` and
   `asa-es` included: the variant has fewer offering days and every controller of `SOLVING` is
   evaluated on the offering days only (each plans fewer hours than in the run without the
   timeout); the forecast variant is that of the run without the timeout, and the timeout is
   counted under the oracle-departure variant only.
6. A box whose lower corner is below 0 (on each offering day of the second unit, in both
   variants, the centre of the first service hour moved to `w - 0.001`, so that its lower corner
   at scale 1 is -0.001) does not fail a unit: the run ends with exit 0, no unit or variant
   failed, and every method has its rows in both variants of every unit that can be split, and a
   gate reading.
7. A run of the registry without one method (`asa-es`, and in a second run `diagonal-split`;
   `--methods` in the registry's order): every other method's rows, gate reading and split-table
   rows, the solver counts of the other methods of `SOLVING` and the paired differences of every
   pair without the method left out are those of the default run, and the `timeouts` definition
   names the other methods of `SOLVING`. (That such a run writes the bytes the registry without
   that method wrote before it was added is a comparison between two commits, not run here.)
8. The default run's summary has a row of `diagonal-split` in every tier table and every
   gate-reading table of a unit that can be split, and a paired row of it against `re-solve` in
   every table of paired differences against `re-solve`.
"""

import dataclasses
import json
from pathlib import Path

import pytest
from compare_fake import NAMES, solving_methods
from gate_fake import COMMIT, GATE_UNITS, gate_cars, matrix_document, write_build

from fleetreserve import (
    compare_methods,
    compare_run,
    diagonal_split,
    gate_day,
    gate_run,
    run_verify,
)
from fleetreserve.asa_controller import OUTCOMES
from fleetreserve.gate_config import TIER_LABELS
from fleetreserve.policy import AffinePolicy

SPLITS = 4
REGISTRY = [m.name for m in compare_methods.METHODS]
SOLVING = solving_methods()
ASA = ("asa", "asa-es")
SIZING = [m.name for m in compare_methods.METHODS if m.kind == "sizing"]
WITHOUT = [name for name in REGISTRY if name not in SOLVING and name not in SIZING]
LEFT_OUT = ("asa-es", "diagonal-split")
BASE_TIMEOUTS = (
    "a day whose certificate times out abstains for every method; a re-solve that times out "
    "leaves the rung on its predecessor's plan; the controllers solve no programme"
)
SOLVES = "Controller solves (all units and variants together): "


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


def _units():
    return {
        GATE_UNITS[0]: gate_cars(2),
        GATE_UNITS[1]: gate_cars(3),
        GATE_UNITS[2]: gate_cars(2, "B") + gate_cars(1, "C"),
    }


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, _units())
    document = matrix_document(NAMES)
    document["n_calibration"] = 2
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix


def _run(build, folder: Path, workers, methods=None, prepare=None):
    """One run; `methods` None runs the default list (no `--methods`); `prepare` replaces
    `gate_day.prepare_day` for the run."""
    run, tables, matrix = build
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        if prepare is not None:
            monkeypatch.setattr(gate_day, "prepare_day", prepare(gate_day.prepare_day))
        folder.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        arguments = [
            "--model",
            "empirical",
            "--dial",
            "0.03",
            "--matrix",
            str(matrix),
            "--tables",
            str(tables),
            "--truth-run",
            str(run),
            "--splits",
            str(SPLITS),
            "--workers",
            str(workers),
        ]
        if methods is not None:
            arguments += ["--methods", *methods]
        code = compare_run.main(arguments)
    return code, folder


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    top = tmp_path_factory.mktemp("runs")
    found = {w: _run(build, top / f"default{w}", w) for w in (1, 2, 3)}
    found["without"] = _run(build, top / "without", 1, WITHOUT)
    for name in LEFT_OUT:
        found[name] = _run(build, top / name, 1, [m for m in REGISTRY if m != name])
    return found


def _document(folder: Path) -> dict:
    return json.loads((folder / compare_run.RESULTS_NAME).read_text())


def _calibratable(document):
    return [u for u in GATE_UNITS if document["units"][u]["status"] == "calibratable"]


# --- 1. the default list -------------------------------------------------------------------------


def test_a_run_without_methods_runs_every_method_of_the_registry_in_order(runs):
    code, out = runs[1]
    assert code == 0
    document = _document(out)
    assert document["methods"] == REGISTRY
    assert REGISTRY.index("asa-es") == REGISTRY.index("asa") + 1
    assert document["failed"] == []
    assert document["units"][GATE_UNITS[0]]["status"] == "not calibratable: 2 days"
    assert document["units"][GATE_UNITS[0]]["variants"] == {}
    assert _calibratable(document) == list(GATE_UNITS[1:])
    for unit in GATE_UNITS[1:]:
        entry = document["units"][unit]
        assert set(entry["variants"]) == {"forecast", "oracle-departure"}
        for variant in entry["variants"].values():
            assert set(variant["methods"]) == set(REGISTRY)
            assert variant["offering_days"] > 0
            for tiers in variant["methods"].values():
                assert set(tiers) == {"A", "B"}
                assert tiers["A"]["splits"] == tiers["B"]["splits"] == SPLITS
        assert set(entry["gate_reading"]) == set(REGISTRY)
        for tiers in document["paired_differences"]["units"][unit].values():
            for tables in tiers.values():
                against = [(p["method"], p["reference"]) for p in tables["against_re_solve"]]
                other = [(p["method"], p["reference"]) for p in tables["other_pairs"]]
                assert against == [(m, "re-solve") for m in REGISTRY[1:]]
                others = len(REGISTRY) - 1
                assert len(other) == others * (others - 1) // 2 and ("asa-es", "asa") in other


# --- 2. workers ----------------------------------------------------------------------------------


def test_a_default_run_is_byte_identical_for_one_two_and_three_workers(runs):
    assert {w: runs[w][0] for w in (1, 2, 3)} == {1: 0, 2: 0, 3: 0}
    names = (
        compare_run.RESULTS_NAME,
        compare_run.SUMMARY_NAME,
        compare_run.TABLE_NAME,
        compare_run.SPLITS_NAME,
    )
    for name in names:
        first = (runs[1][1] / name).read_bytes()
        assert (runs[2][1] / name).read_bytes() == first
        assert (runs[3][1] / name).read_bytes() == first


# --- 3. the solver counts ------------------------------------------------------------------------


def test_the_solver_counts_are_carried_for_the_solving_methods_only(runs):
    assert (*ASA, "diagonal-split") == SOLVING
    document = _document(runs[1][1])
    stages = {"asa": 4, "asa-es": 5}
    total = {
        name: {"solves": dict.fromkeys(OUTCOMES, 0), "hours": 0, "fallback_hours": 0}
        for name in SOLVING
    }
    boxes = dict.fromkeys(diagonal_split.BOXES, 0)
    variants_seen = 0
    for unit in GATE_UNITS[1:]:
        for variant in document["units"][unit]["variants"].values():
            found = variant["evaluations"]["controller_solver_outcomes"]
            pairs = variant["evaluations"]["controller_day_scale_pairs"]
            assert list(found) == list(SOLVING)
            assert len({found[name]["hours"] for name in SOLVING}) == 1
            for name in SOLVING:
                counts = found[name]
                assert counts["hours"] > 0 and counts["fallback_hours"] == 0
                optimal = stages[name] * counts["hours"] if name in stages else pairs
                assert counts["solves"] == {**dict.fromkeys(OUTCOMES, 0), "optimal": optimal}
                for outcome, number in counts["solves"].items():
                    total[name]["solves"][outcome] += number
                total[name]["hours"] += counts["hours"]
            assert sum(found["diagonal-split"]["boxes"].values()) == pairs
            for kind, number in found["diagonal-split"]["boxes"].items():
                boxes[kind] += number
            variants_seen += 1
    assert variants_seen == 2 * len(GATE_UNITS[1:])
    definitions = document["definitions"]
    assert f"({', '.join(SOLVING)})" in definitions["timeouts"]
    assert definitions["controller_solver_outcomes"].startswith(
        "evaluations.controller_solver_outcomes"
    )
    assert "diagonal_split_boxes" in definitions
    lines = (runs[1][1] / compare_run.SUMMARY_NAME).read_text().splitlines()
    (line,) = [x for x in lines if x.startswith(SOLVES)]
    parts = []
    for name in SOLVING:
        solves = ", ".join(f"{o} {n}" for o, n in total[name]["solves"].items())
        part = f"{name}: solves {solves}; hours planned {total[name]['hours']}, by the fall-back 0"
        if name == "diagonal-split":
            part += "; boxes " + ", ".join(f"{k} {n}" for k, n in boxes.items())
        parts.append(part)
    assert line == SOLVES + "; ".join(parts) + "."


# --- 4. a run without the methods that solve programmes -----------------------------------------


def test_a_run_without_the_solving_methods_writes_none_of_the_added_fields(runs):
    code, out = runs["without"]
    assert code == 0
    document = _document(out)
    assert document["methods"] == WITHOUT
    text = (out / compare_run.RESULTS_NAME).read_text()
    assert "controller_solver_outcomes" not in text
    assert document["definitions"]["timeouts"] == BASE_TIMEOUTS
    summary = (out / compare_run.SUMMARY_NAME).read_text()
    assert SOLVES not in summary
    default = _document(runs[1][1])
    for unit in GATE_UNITS[1:]:
        for variant, found in document["units"][unit]["variants"].items():
            other = default["units"][unit]["variants"][variant]
            for method in WITHOUT:
                assert found["methods"][method] == other["methods"][method]
            added = ("controller_solver_outcomes", "sizing_solver_outcomes")
            assert found["evaluations"] == {
                k: v for k, v in other["evaluations"].items() if k not in added
            }


# --- 5. a certificate timeout --------------------------------------------------------------------


def test_a_certificate_timeout_abstains_for_every_method_asa_included(build, runs, tmp_path):
    timed = []

    def prepare(original):
        def replaced(truth, forecaster, configuration, settings, no_pause=()):
            found = original(truth, forecaster, configuration, settings, no_pause=no_pause)
            first = configuration.departure == "oracle" and found.unit == GATE_UNITS[1]
            if first and not found.abstains and not timed:
                timed.append(found.date)
                return dataclasses.replace(
                    found,
                    abstain_reason="no certificate (timeout)",
                    certificate_outcome="timeout",
                    value=0.0,
                )
            return found

        return replaced

    code, out = _run(build, tmp_path / "timeout", 1, prepare=prepare)
    assert code == 0 and len(timed) == 1
    document, base = _document(out), _document(runs[1][1])
    assert document["methods"] == REGISTRY
    assert document["timeouts"]["oracle_certificates"] == 1
    assert document["timeouts"]["certificates"] == 0
    before = base["units"][GATE_UNITS[1]]["variants"]
    after = document["units"][GATE_UNITS[1]]["variants"]
    assert after["forecast"] == before["forecast"]
    oracle, was = after["oracle-departure"], before["oracle-departure"]
    assert oracle["offering_days"] == was["offering_days"] - 1
    pairs = oracle["evaluations"]
    assert pairs["controller_day_scale_pairs"] == oracle["offering_days"] * 10
    assert pairs["re_solve_day_scale_pairs"] == pairs["controller_day_scale_pairs"]
    for name in SOLVING:
        hours, hours_was = (
            v["evaluations"]["controller_solver_outcomes"][name]["hours"] for v in (oracle, was)
        )
        assert 0 < hours < hours_was
    assert set(oracle["methods"]) == set(REGISTRY)


# --- 6. a box whose lower corner is below 0 ------------------------------------------------------


def test_a_box_whose_lower_corner_is_below_0_fails_no_unit(build, tmp_path):
    moved = []

    def prepare(original):
        def replaced(truth, forecaster, configuration, settings, no_pause=()):
            found = original(truth, forecaster, configuration, settings, no_pause=no_pause)
            if found.unit != GATE_UNITS[1] or found.abstains:
                return found
            policy = found.certificate.policy
            m = policy.m.copy()
            m[0] = policy.w[0] - 0.001
            below = AffinePolicy(m, policy.w, policy.g, policy.G)
            moved.append((configuration.departure, found.date))
            certificate = dataclasses.replace(found.certificate, policy=below)
            return dataclasses.replace(found, certificate=certificate)

        return replaced

    code, out = _run(build, tmp_path / "below", 1, prepare=prepare)
    kinds = {departure for departure, _ in moved}
    assert len(kinds) == 2 and "oracle" in kinds  # both variants
    document = _document(out)
    assert code == 0 and document["failed"] == []
    assert document["methods"] == REGISTRY
    for unit in GATE_UNITS[1:]:
        entry = document["units"][unit]
        assert set(entry["variants"]) == {"forecast", "oracle-departure"}
        for variant in entry["variants"].values():
            assert "failed" not in variant
            assert set(variant["methods"]) == set(REGISTRY)
        assert set(entry["gate_reading"]) == set(REGISTRY)


# --- 7. a run without one method -----------------------------------------------------------------


def _rows(folder: Path, methods) -> list[str]:
    lines = (folder / compare_run.SPLITS_NAME).read_text().splitlines()
    column = lines[0].split(",").index("method")
    return [lines[0]] + [x for x in lines[1:] if x.split(",")[column] in methods]


@pytest.mark.parametrize("left_out", LEFT_OUT)
def test_a_run_without_one_method_gives_every_other_methods_figures_of_the_default_run(
    runs, left_out
):
    code, out = runs[left_out]
    assert code == 0
    kept = [m for m in REGISTRY if m != left_out]
    solving = [m for m in SOLVING if m != left_out]
    document, default = _document(out), _document(runs[1][1])
    assert document["methods"] == kept
    assert f"({', '.join(solving)})" in document["definitions"]["timeouts"]
    for unit in GATE_UNITS[1:]:
        entry, other = document["units"][unit], default["units"][unit]
        for variant, found in entry["variants"].items():
            was = other["variants"][variant]
            assert found["methods"] == {m: was["methods"][m] for m in kept}
            assert found["evaluations"]["controller_solver_outcomes"] == {
                m: was["evaluations"]["controller_solver_outcomes"][m] for m in solving
            }
        assert entry["gate_reading"] == {m: other["gate_reading"][m] for m in kept}
        for variant, tiers in document["paired_differences"]["units"][unit].items():
            for tier, tables in tiers.items():
                was = default["paired_differences"]["units"][unit][variant][tier]
                for table, pairs in tables.items():
                    expected = [
                        p for p in was[table] if left_out not in (p["method"], p["reference"])
                    ]
                    assert pairs == expected
    assert _rows(out, kept) == _rows(runs[1][1], kept)
    assert len(_rows(out, kept)) < len(_rows(runs[1][1], REGISTRY))


# --- 8. the summary rows of diagonal-split -------------------------------------------------------


def _tables(lines):
    """Heading -> the lines under it, for every heading of a table (one space, then a letter)."""
    found, heading = [], None
    for line in lines:
        if line.startswith(" ") and not line.startswith("  "):
            heading = line
            found.append((heading, []))
        elif heading is not None and line.startswith("  "):
            found[-1][1].append(line)
        else:
            heading = None
    return found


def test_the_summary_has_rows_of_diagonal_split(runs):
    lines = (runs[1][1] / compare_run.SUMMARY_NAME).read_text().splitlines()
    tables = _tables(lines)
    tiers = [(h, rows) for h, rows in tables if h in (" " + t for t in TIER_LABELS.values())]
    gates = [(h, rows) for h, rows in tables if h.startswith(" Gate reading, tier A")]
    against = [(h, rows) for h, rows in tables if h.endswith(", against re-solve")]
    units = len(GATE_UNITS[1:])
    assert (len(tiers), len(gates), len(against)) == (2 * units, units, 2 * 2 * units)
    for _, rows in tiers + gates:
        assert sum(1 for x in rows if x.startswith("  diagonal-split ")) == 1
    for _, rows in against:
        mine = [x for x in rows if x.startswith("  diagonal-split ")]
        assert len(mine) == 1 and mine[0].split()[1] == "re-solve"
