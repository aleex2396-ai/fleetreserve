"""The re-solve diagnosis command (`resolve_diagnosis`) on the made-up gate units of `gate_fake`.

The made-up units have one service hour and, on some days, a car that leaves within it while its
forecast says it stays; under the reading "rounded-hour" the one-way rung sees it gone at the
start of hour 1 and re-solves. The scales are 0.5 and 1.

1. A full run ends with exit 0 and writes the rows, the results, the summary and the forecast
   table. The rows of the file are, per unit and scale, the re-solves the gate's own evaluation
   reports for the same prepared days (`gate_day.evaluate`), in count and outcome; the made-up
   units give infeasible and optimal re-solves. The results' counts and shares are recounted
   from the rows; every infeasible row carries a cause and its distances.
2. One worker and two workers write byte-identical files.
3. With the oracle departure no re-solve is infeasible: the summary says so and the exit is 0.
4. `--unit` keeps the named units only.
5. A unit whose measurement raises is recorded as failed and the exit is 1; the other unit's
   numbers are still written.
6. Exit status 2 for a bad scale, an unknown configuration, unit or model, a stale truth build
   and a changed table.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_units, matrix_document, write_build

from fleetreserve import (
    gate_config,
    gate_day,
    gate_run,
    resolve_diagnosis,
    resolve_diagnosis_day,
    resolve_diagnosis_summary,
    run_verify,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_empirical import EmpiricalForecaster

EMPIRICAL = "main-rounded-hour-k0.3-eps0.01-empirical"
ORACLE = "main-rounded-hour-k0.3-eps0.01-oracle"
SCALES = ("0.5", "1.0")
OUTPUTS = (
    resolve_diagnosis.ROWS_NAME,
    resolve_diagnosis.RESULTS_NAME,
    resolve_diagnosis.SUMMARY_NAME,
    resolve_diagnosis.TABLE_NAME,
)


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
    matrix.write_text(json.dumps(matrix_document((EMPIRICAL, ORACLE))))
    return run, tables, matrix


def command(build, out: Path, monkeypatch, *extra, configuration=EMPIRICAL, model="empirical"):
    run, tables, matrix = build
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    scales = [] if "--scale" in extra else ["--scale", *SCALES]
    chosen = ["--model", model, "--configuration", configuration]
    return resolve_diagnosis.main([*chosen, *arguments, *scales, *extra])


def _module_run(build, folder: Path, *extra, patch=None, **kwargs):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        if patch is not None:
            patch(monkeypatch)
        out = folder / "out"
        return command(build, out, monkeypatch, *extra, **kwargs), out


@pytest.fixture(scope="module")
def full(build, tmp_path_factory):
    return _module_run(build, tmp_path_factory.mktemp("full"), "--workers", "1")


def _rows(out: Path) -> list[dict]:
    with (out / resolve_diagnosis.ROWS_NAME).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _document(out: Path) -> dict:
    return json.loads((out / resolve_diagnosis.RESULTS_NAME).read_text())


# --- 1. a full run -------------------------------------------------------------------------------


def test_a_full_run_writes_every_output(full):
    status, out = full
    assert status == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(OUTPUTS)
    document = _document(out)
    assert document["label"] == resolve_diagnosis_summary.LABEL
    assert document["run"]["configuration"] == EMPIRICAL
    assert document["run"]["scales"] == list(SCALES)
    assert document["run"]["rung"] == "one-way"
    assert document["model"] == "empirical" and document["failed_units"] == []
    assert sorted(document["units"]) == sorted(GATE_UNITS)
    assert set(document["timeouts"].values()) == {0}
    assert (out / resolve_diagnosis.SUMMARY_NAME).read_text().startswith(document["label"])


def test_the_rows_are_the_re_solves_the_gate_s_evaluation_reports(build, full):
    """The prepared days are made again here with the gate's own functions and the empirical
    forecaster (the model `empirical` gives its forecasts as a table), and evaluated by the gate."""
    _, tables, matrix_file = build
    matrix = gate_config.load(matrix_file)
    configuration = matrix.configuration(EMPIRICAL)
    rows = _rows(full[1])
    seen = Counter()
    for unit in GATE_UNITS:
        days = load_unit_days(
            tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        forecaster = EmpiricalForecaster(days, reference=matrix.reference, min_cell=matrix.min_cell)
        prepared = [
            gate_day.prepare_day(truth, forecaster, configuration, gate_run.settings(matrix))
            for truth in days.days
        ]
        for scale in SCALES:
            expected = Counter()
            for p in prepared:
                expected.update(gate_day.evaluate(p, float(scale)).rungs["one-way"].resolves)
            mine = Counter(r["outcome"] for r in rows if (r["unit"], r["scale"]) == (unit, scale))
            assert mine == expected
            seen.update(mine)
    assert seen["infeasible"] >= 20 and seen["optimal"] >= 20, seen
    assert sum(seen.values()) == len(rows)


def test_the_results_are_recounted_from_the_rows(full):
    document, rows = _document(full[1]), _rows(full[1])
    assert tuple(rows[0]) == resolve_diagnosis.COLUMNS
    for unit, entry in document["units"].items():
        for scale in SCALES:
            mine = [r for r in rows if (r["unit"], r["scale"]) == (unit, scale)]
            found = entry["scales"][scale]
            assert found["re_solves"] == len(mine)
            assert found["by_outcome"] == dict(sorted(Counter(r["outcome"] for r in mine).items()))
            bad = [r for r in mine if r["outcome"] == "infeasible"]
            assert found["infeasible"]["count"] == len(bad)
            causes = Counter(r["cause"] for r in bad)
            assert {k: v["count"] for k, v in found["infeasible"]["by_cause"].items()} == causes
            for group, counted in found["infeasible"]["single_group"].items():
                assert counted["count"] == sum(group in r["single"].split("|") for r in bad)
                assert counted["share"] == (counted["count"] / len(bad) if bad else None)
            early = sum(int(r["early_gone_cars"]) > 0 for r in bad)
            assert found["infeasible"]["after_early_leaver"]["count"] == early
            miss = [abs(float(r["request_miss_kwh"])) for r in bad if r["request_miss_kwh"]]
            assert found["infeasible"]["request_miss_kwh"]["count"] == len(miss)
            if miss:
                mean = found["infeasible"]["request_miss_kwh"]["mean"]
                assert mean == pytest.approx(sum(miss) / len(miss))
        total = entry["all_scales"]["re_solves"]
        assert total == sum(entry["scales"][s]["re_solves"] for s in SCALES)


def test_every_infeasible_row_is_diagnosed(full):
    rows = _rows(full[1])
    bad = [r for r in rows if r["outcome"] == "infeasible"]
    for row in bad:
        assert row["recheck"] == "infeasible" and row["cause"] not in ("", "not determined")
        assert row["box_scale_outcome"] and row["need_slack_outcome"]
        assert row["request_miss_outcome"] and row["departure_relaxed"]
        assert int(row["early_gone_cars"]) >= 1  # made up so: the re-solve follows an early leaver
    for row in rows:
        if row["outcome"] != "infeasible":
            assert all(row[c] == "" for c in resolve_diagnosis_day.INFEASIBLE_ONLY)


# --- 2. workers ----------------------------------------------------------------------------------


def test_one_and_two_workers_write_the_same_bytes(build, full, tmp_path):
    status, out = _module_run(build, tmp_path, "--workers", "2")
    assert status == 0
    for name in OUTPUTS:
        assert (out / name).read_bytes() == (full[1] / name).read_bytes(), name


# --- 3. no infeasible re-solve -------------------------------------------------------------------


def test_with_no_infeasible_re_solve_the_summary_says_so(build, tmp_path, capsys):
    status, out = _module_run(build, tmp_path, configuration=ORACLE)
    assert status == 0
    document = _document(out)
    for entry in document["units"].values():
        assert entry["all_scales"]["infeasible"]["count"] == 0
        assert entry["all_scales"]["infeasible"]["box_scale"]["quantiles"] is None
    text = (out / resolve_diagnosis.SUMMARY_NAME).read_text()
    assert text.count(resolve_diagnosis_summary.NONE_INFEASIBLE) == len(GATE_UNITS) * 3
    assert text in capsys.readouterr().out
    assert all(r["outcome"] != "infeasible" for r in _rows(out))
    assert sorted(p.name for p in out.iterdir()) == sorted(OUTPUTS)


# --- 4. units ------------------------------------------------------------------------------------


def test_only_the_named_units_are_measured(build, full, tmp_path):
    status, out = _module_run(build, tmp_path, "--unit", GATE_UNITS[2], "--scale", "1")
    assert status == 0
    assert list(_document(out)["units"]) == [GATE_UNITS[2]]
    kept = [r for r in _rows(full[1]) if (r["unit"], r["scale"]) == (GATE_UNITS[2], "1.0")]
    assert _rows(out) == kept and kept


# --- 5. a failed unit ----------------------------------------------------------------------------


def test_a_unit_that_fails_is_recorded_and_the_exit_is_one(build, tmp_path):
    real = resolve_diagnosis.measure_unit

    def failing(runner, name, *rest):
        if name == GATE_UNITS[1]:
            raise RuntimeError("made to fail")
        return real(runner, name, *rest)

    status, out = _module_run(
        build,
        tmp_path,
        patch=lambda m: m.setattr(resolve_diagnosis, "measure_unit", failing),
    )
    assert status == 1
    document = _document(out)
    assert document["failed_units"] == [f"{GATE_UNITS[1]}: RuntimeError: made to fail"]
    assert "all_scales" in document["units"][GATE_UNITS[2]]
    assert "FAILED: " in (out / resolve_diagnosis.SUMMARY_NAME).read_text()


# --- 6. could not run ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra, kwargs",
    [
        (("--scale", "0"), {}),
        (("--scale", "0.5", "0.5"), {}),
        (("--scale", "nan"), {}),
        (("--workers", "0"), {}),
        (("--unit", "no-such-unit"), {}),
        ((), {"configuration": "no-such-configuration"}),
        ((), {"model": "no_such_model"}),
    ],
)
def test_a_bad_argument_could_not_run(build, tmp_path, monkeypatch, capsys, extra, kwargs):
    assert command(build, tmp_path / "out", monkeypatch, *extra, **kwargs) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
    assert list((tmp_path / "out").iterdir()) == []


def test_a_stale_truth_build_could_not_run(build, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate_run, "_head_commit", lambda: "b" * 40)
    assert command(build, tmp_path / "out", monkeypatch) == 2
    assert "not at the checked-out commit" in capsys.readouterr().out


def test_a_changed_table_could_not_run(tmp_path, monkeypatch, capsys):
    run, tables = write_build(tmp_path, gate_units(20))
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(matrix_document((EMPIRICAL, ORACLE))))
    table = tables / f"{GATE_UNITS[1]}-cars.csv"
    table.write_text(table.read_text() + "\n")
    assert command((run, tables, matrix), tmp_path / "out", monkeypatch) == 2
    assert "differs from the table" in capsys.readouterr().out
