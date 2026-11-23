"""The comparison command (`compare_run`) on the made-up gate units of `gate_fake`.

Every run of `compare_run.main` in this file that writes results names its methods: every method
of the registry but `asa` and `asa-es`, unless a test names fewer (the refusals of item 4 end
before any method runs). `test_asa_command` runs the command with its default list (`asa` and
`asa-es` included) on smaller units: every method in the registry's order, the worker counts, the
certificate timeout and a box whose lower corner is below 0.

1. A run at the dial 0.03 of every method of the registry but `asa` and `asa-es` ends with exit
   0 and writes both files; both variants and every method run are given for each unit that can
   be split, with both tiers; the unit that cannot be split is reported with its status and no
   numbers, never as an offer of 0, in the results and the summary.
2. Tie-in B: the `re-solve` rows (both tiers), its gate reading and its counts of evaluations are
   those `dial_choice` gives for the same configuration, tables and splits, field by field
   (`dial_choice` run on a matrix holding that configuration and the configuration without rows of
   the family, each with its oracle partner, and the same shared fields).
3. The results, the summary, the forecast table and the split table of the runs of item 1 are
   byte-identical for 1, 2 and 3 workers, and a second run with one worker gives the same results,
   summary and split table again.
4. Exit status 2 for an unknown model, a `--dial` not in the dial family, a matrix without a dial
   family, a missing truth build, a method name registered twice with different controllers, an
   unknown method and a bad number of workers or splits.
5. Exit status 1 when one variant fails: it is recorded and the other variant still runs; the
   split table holds rows of the other variant only, and every pair of the failed variant has no
   paired split, every split counted as left out, and no figure (a dash in the summary).
6. A day whose certificate times out under the oracle-departure variant offers nothing for every
   method run (every method but `asa` and `asa-es`) of that variant, and the timeout is counted
   under that variant only.
"""

import dataclasses
import json
from pathlib import Path

import pytest
from compare_fake import LOW, NAMES, PLAIN
from gate_fake import COMMIT, GATE_UNITS, gate_units, matrix_document, write_build

from fleetreserve import compare_methods, compare_run, dial_choice, gate_day, gate_run, run_verify

SPLITS = 12
# Every method but the two linear-programme forms of ASA, which `test_asa_command` runs on smaller
# units: they solve four or five programmes for each hour of each path, too slow for the units of
# this file.
METHODS = [m.name for m in compare_methods.METHODS if m.name not in ("asa", "asa-es")]


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, gate_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return run, tables, matrix


def _arguments(build, matrix=None):
    run, tables, committed = build
    return [
        "--matrix",
        str(matrix or committed),
        "--tables",
        str(tables),
        "--truth-run",
        str(run),
        "--splits",
        str(SPLITS),
    ]


def _run(build, folder: Path, *extra, main=compare_run.main, dial=True, matrix=None):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        folder.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        dials = ["--dial", "0.03"] if dial else []
        arguments = _arguments(build, matrix)
        if main is compare_run.main:  # a later --methods in `extra` replaces this one
            arguments += ["--methods", *METHODS]
        return main(["--model", "empirical", *dials, *arguments, *extra]), folder


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    top = tmp_path_factory.mktemp("runs")
    return {
        workers: _run(build, top / f"w{workers}", "--workers", str(workers))
        for workers in (1, 2, 3)
    }


def _document(folder: Path) -> dict:
    return json.loads((folder / compare_run.RESULTS_NAME).read_text())


# --- 1. a full run -------------------------------------------------------------------------------


def test_a_run_writes_both_files_with_both_variants_and_every_method(runs):
    code, out = runs[1]
    assert code == 0
    assert (out / compare_run.RESULTS_NAME).is_file()
    assert (out / compare_run.SUMMARY_NAME).is_file()
    document = _document(out)
    assert document["failed"] == [] and set(document["timeouts"].values()) == {0}
    assert document["dial"] == 0.03 and document["methods"] == METHODS
    assert document["configurations"] == {"forecast": LOW[0], "oracle-departure": LOW[1]}
    assert document["inputs"]["setting"] == "main" and len(document["matrix"]["sha256"]) == 64
    assert set(document["inputs"]["table_sha256"]) == {
        f"{u}-{kind}.csv" for u in GATE_UNITS for kind in ("cars", "days")
    }
    assert document["run"]["splits"] == SPLITS
    assert set(document["definitions"]["readings"]) == {"C1", "C2", "C3", "C4", "C5"}
    for unit in GATE_UNITS[1:]:
        entry = document["units"][unit]
        assert entry["status"] == "calibratable"
        assert set(entry["variants"]) == {"forecast", "oracle-departure"}
        for variant in entry["variants"].values():
            assert set(variant["methods"]) == set(METHODS)
            assert variant["offering_days"] > 0
            pairs = variant["evaluations"]
            assert pairs["controller_day_scale_pairs"] == pairs["re_solve_day_scale_pairs"]
            assert pairs["re_solve_day_scale_pairs"] == variant["offering_days"] * 10
            for tiers in variant["methods"].values():
                assert set(tiers) == {"A", "B"}
                assert tiers["A"]["splits"] == tiers["B"]["splits"] == SPLITS
        assert set(entry["gate_reading"]) == set(METHODS)


def test_the_unit_that_cannot_be_split_has_no_numbers(runs):
    _, out = runs[1]
    entry = _document(out)["units"][GATE_UNITS[0]]
    assert entry["status"] == "not calibratable: 10 days"
    assert entry["variants"] == {} and "gate_reading" not in entry
    text = (out / compare_run.SUMMARY_NAME).read_text()
    block = text.split(f"{GATE_UNITS[0]} (")[1].split("\n\n")[0]
    assert "no numbers: the unit cannot be split" in block
    assert "0.00" not in block


def test_the_summary_shows_each_methods_numbers_of_both_variants(runs):
    _, out = runs[1]
    document = _document(out)
    lines = (out / compare_run.SUMMARY_NAME).read_text().splitlines()
    unit, tier, checked = None, None, 0
    for line in lines:
        if line.startswith(GATE_UNITS):
            unit = line.split(" ")[0]
        elif line.startswith(" tier "):
            tier = line.split()[1]
        elif line.startswith(" Gate reading"):
            tier = None
        elif tier and unit in GATE_UNITS[1:] and line.split()[:1] in ([m] for m in METHODS):
            cells = line.split()
            variants = document["units"][unit]["variants"]
            for k, variant in enumerate(("forecast", "oracle-departure")):
                found = variants[variant]["methods"][cells[0]][tier]
                assert cells[1 + 6 * k] == f"{found['mean_offer_kwh_per_test_day']:.2f}"
                p10 = found["offer_p10"]
                assert cells[6 + 6 * k] == f"{p10:.2f}"
            checked += 1
    assert checked == 2 * 2 * len(METHODS)


# --- 2. tie-in B: the re-solve rows are dial_choice's ---------------------------------------------


@pytest.fixture(scope="module")
def dial_document(build, tmp_path_factory):
    folder = tmp_path_factory.mktemp("dial")
    matrix = folder / "matrix.json"  # the dial family with the one dial 0.03 only
    matrix.write_text(json.dumps(matrix_document(PLAIN + LOW)))
    code, out = _run(build, folder / "out", main=dial_choice.main, dial=False, matrix=matrix)
    assert code == 0
    return json.loads((out / dial_choice.RESULTS_NAME).read_text())


def test_the_re_solve_rows_are_those_of_dial_choice(runs, dial_document):
    document = _document(runs[1][1])
    for unit in GATE_UNITS[1:]:
        ours = document["units"][unit]
        theirs = dial_document["units"][unit]["configurations"][LOW[0]]
        forecast = ours["variants"]["forecast"]
        assert forecast["methods"]["re-solve"] == theirs["tiers"]
        assert forecast["offering_days"] == theirs["offering_days"]
        assert (
            forecast["evaluations"]["re_solve_day_scale_pairs"]
            == (theirs["evaluations"]["day_scale_pairs"])
        )
        assert (
            forecast["evaluations"]["re_solves_of_the_one_way_rung"]
            == (theirs["evaluations"]["re_solves_of_the_one_way_rung"])
        )
        reading = dict(theirs["gate_reading"])
        assert reading.pop("oracle_partner") == LOW[1]
        assert ours["gate_reading"]["re-solve"] == reading
        assert dial_document["units"][unit]["counting_days"] == ours["counting_days"]
    assert PLAIN[0] in dial_document["units"][GATE_UNITS[1]]["configurations"]


# --- 3. the same bytes for any number of workers -------------------------------------------------


def test_the_files_are_byte_identical_for_one_two_and_three_workers(runs):
    codes = {w: code for w, (code, _) in runs.items()}
    assert codes == {1: 0, 2: 0, 3: 0}
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


def test_a_second_run_gives_the_same_bytes(build, runs, tmp_path):
    code, out = _run(build, tmp_path / "again", "--workers", "1")
    assert code == 0
    for name in (compare_run.RESULTS_NAME, compare_run.SUMMARY_NAME, compare_run.SPLITS_NAME):
        assert (out / name).read_bytes() == (runs[1][1] / name).read_bytes()
    assert "total_seconds" in json.loads((out / compare_run.TIMING_NAME).read_text())


# --- 4. refusals ---------------------------------------------------------------------------------


def test_an_unknown_model_could_not_run(build, capsys):
    code = compare_run.main(["--model", "no_such_model", "--dial", "0.03", *_arguments(build)])
    assert code == 2
    assert "COULD NOT RUN: no model module" in capsys.readouterr().out


def test_a_dial_not_in_the_family_could_not_run(build, capsys):
    code = compare_run.main(["--model", "empirical", "--dial", "0.05", *_arguments(build)])
    assert code == 2
    assert "is not a dial of the matrix's dial family (0.03, 0.1)" in capsys.readouterr().out


def test_a_matrix_without_a_dial_family_could_not_run(build, tmp_path, capsys):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(matrix_document(PLAIN)))
    arguments = ["--model", "empirical", "--dial", "0.03", *_arguments(build, bare)]
    assert compare_run.main(arguments) == 2
    assert "no dial family" in capsys.readouterr().out


def test_a_missing_truth_build_could_not_run(build, tmp_path, capsys):
    arguments = ["--model", "empirical", "--dial", "0.03", *_arguments(build)]
    arguments[arguments.index("--truth-run") + 1] = str(tmp_path / "20260101T000000Z-wp2-truth")
    assert compare_run.main(arguments) == 2
    assert "truth-summary.json: the file is missing" in capsys.readouterr().out


def test_a_method_registered_twice_with_different_controllers_could_not_run(
    build, monkeypatch, capsys
):
    twice = (*compare_methods.METHODS, compare_methods.Method("edf", "controller", _other))
    monkeypatch.setattr(compare_methods, "METHODS", twice)
    assert compare_run.main(["--model", "empirical", "--dial", "0.03", *_arguments(build)]) == 2
    assert "'edf' is registered twice with different controllers" in capsys.readouterr().out


def _other(inputs):
    return compare_methods.llf(inputs)


def test_an_unknown_method_could_not_run(build, capsys):
    arguments = ["--model", "empirical", "--dial", "0.03", *_arguments(build)]
    assert compare_run.main([*arguments, "--methods", "edf", "no-such-method"]) == 2
    assert "COULD NOT RUN: unknown method 'no-such-method'" in capsys.readouterr().out


def test_a_bad_number_of_workers_or_splits_could_not_run(build, capsys):
    arguments = ["--model", "empirical", "--dial", "0.03", *_arguments(build)]
    assert compare_run.main([*arguments, "--workers", "0"]) == 2
    assert compare_run.main([*arguments, "--splits", "0"]) == 2
    assert capsys.readouterr().out.count("COULD NOT RUN") == 2


# --- 5. a failed variant -------------------------------------------------------------------------


def test_a_failed_variant_is_recorded_and_the_other_still_runs(build, monkeypatch, tmp_path):
    original = gate_day.prepare_day

    def prepare(truth, forecaster, configuration, settings, no_pause=()):
        if configuration.departure == "oracle":
            raise RuntimeError("made-up failure")
        return original(truth, forecaster, configuration, settings, no_pause=no_pause)

    monkeypatch.setattr(gate_day, "prepare_day", prepare)
    code, out = _run(build, tmp_path / "failed", "--methods", "fixed-plan", "edf")
    assert code == 1
    document = _document(out)
    assert document["failed"] == sorted(
        f"{unit} | oracle-departure: RuntimeError: made-up failure" for unit in GATE_UNITS[1:]
    )
    for unit in GATE_UNITS[1:]:
        entry = document["units"][unit]
        assert entry["variants"]["oracle-departure"]["failed"].endswith("made-up failure")
        assert set(entry["variants"]["forecast"]["methods"]) == {"fixed-plan", "edf"}
        assert "gate_reading" not in entry
    text = (out / compare_run.SUMMARY_NAME).read_text()
    assert "oracle-departure: failed (RuntimeError: made-up failure)" in text
    assert "not computed: a variant did not run" in text and "FAILED: " in text
    splits = (out / compare_run.SPLITS_NAME).read_text().splitlines()
    assert len(splits) == 1 + len(GATE_UNITS[1:]) * 2 * 2 * SPLITS  # header; 2 tiers, 2 methods
    assert all(",forecast," in line for line in splits[1:])
    assert document["split_table"]["rows"] == len(splits) - 1
    for unit in GATE_UNITS[1:]:
        for tier in ("A", "B"):
            found = document["paired_differences"]["units"][unit]["oracle-departure"][tier]
            assert found["against_re_solve"] == []  # no re-solve in this run
            (row,) = found["other_pairs"]
            assert (row["method"], row["reference"]) == ("edf", "fixed-plan")
            assert row["splits_paired"] == 0 and row["splits_left_out_value_missing"] == SPLITS
            assert row["verdict"] is None and row["mean_difference_kwh_per_test_day"] is None
            (row,) = document["paired_differences"]["units"][unit]["forecast"][tier]["other_pairs"]
            assert row["splits_paired"] == SPLITS
    assert "  none: re-solve is not in the run" in text
    failed_rows = [
        line.split()
        for line in text.splitlines()
        if line.startswith("  edf ") and line.split()[-1] == str(SPLITS)
    ]
    assert len(failed_rows) == 2 * len(GATE_UNITS[1:])  # per unit, both tiers
    for cells in failed_rows:
        assert cells[2:10] == ["-"] * 8 and cells[10] == "0"


# --- 6. a certificate timeout --------------------------------------------------------------------


def test_a_certificate_timeout_abstains_for_every_method_of_its_variant(
    build, runs, monkeypatch, tmp_path
):
    original = gate_day.prepare_day
    first_date = sorted({c.date for c in gate_units()[GATE_UNITS[1]]})[0]

    def prepare(truth, forecaster, configuration, settings, no_pause=()):
        found = original(truth, forecaster, configuration, settings, no_pause=no_pause)
        timed_out = configuration.departure == "oracle" and truth.record.date == first_date
        if timed_out and not found.abstains:
            return dataclasses.replace(
                found,
                abstain_reason="no certificate (timeout)",
                certificate_outcome="timeout",
                value=0.0,
            )
        return found

    monkeypatch.setattr(gate_day, "prepare_day", prepare)
    code, out = _run(build, tmp_path / "timeout")
    assert code == 0
    document, base = _document(out), _document(runs[1][1])
    assert document["timeouts"]["oracle_certificates"] >= 1
    assert document["timeouts"]["certificates"] == 0
    unit = GATE_UNITS[1]
    before = base["units"][unit]["variants"]
    after = document["units"][unit]["variants"]
    assert after["oracle-departure"]["offering_days"] < before["oracle-departure"]["offering_days"]
    assert after["forecast"] == before["forecast"]
    pairs = after["oracle-departure"]["evaluations"]
    assert pairs["controller_day_scale_pairs"] == after["oracle-departure"]["offering_days"] * 10
    assert pairs["re_solve_day_scale_pairs"] == pairs["controller_day_scale_pairs"]
