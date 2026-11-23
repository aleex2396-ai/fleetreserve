"""The report command on made-up tables: it runs on tables that match a verified truth build at
the checked-out commit, and reports "could not run" (exit 2) otherwise."""

import hashlib
import json

import pytest
from truth_tables_fake import regular_days, write

from fleetreserve import forecast_report, run_verify

COMMIT = "a" * 40


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tables = tmp_path / "tables"
    tables.mkdir()
    units = {}
    for unit in forecast_report.UNITS:
        cars = regular_days(25, site="A")
        if unit == "norway-development-pooled":
            cars += regular_days(25, site="B", per_day=3)
        write(tables, unit, cars)
        units[unit] = {
            "sha256": {
                kind: hashlib.sha256((tables / f"{unit}-{kind}.csv").read_bytes()).hexdigest()
                for kind in ("cars", "days")
            }
        }
    run = tmp_path / "20260101T000000Z-wp2-truth"
    run.mkdir()
    summary = run / "truth-summary.json"
    summary.write_text(json.dumps({"units": units}))
    manifest = {
        "commit": COMMIT,
        "command": ["build"],
        "start": "s",
        "end": "e",
        "exit_code": 0,
        "outputs": {"truth-summary.json": hashlib.sha256(summary.read_bytes()).hexdigest()},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(forecast_report, "_head_commit", lambda: COMMIT)
    return run, tables


def _main(run, tables):
    return forecast_report.main(["--run", str(run), "--tables", str(tables)])


def test_report_runs_and_writes_its_results(setup, tmp_path, monkeypatch, capsys):
    run, tables = setup
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    assert _main(run, tables) == 0
    document = json.loads((out / forecast_report.RESULTS_NAME).read_text())
    assert set(document["units"]) == set(forecast_report.UNITS)
    entry = document["units"]["norway-development-pooled"]
    assert entry["counting_days"] == 50
    rows = entry["forecasts"]
    assert len(rows) == 6 * 3 * 2  # the six levels of plan section 2, three modes, two departures
    assert sorted({r["kappa"] for r in rows}) == [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    oracle_known = [r for r in rows if r["mode"] == "known" and r["departure"] == "oracle"]
    assert all(r["share_days_in_e_prime"] == 1.0 for r in oracle_known)
    assert all(r["share_cars_empty_forecast_set"] == 0.0 for r in oracle_known)
    assert (out / forecast_report.TEXT_NAME).read_text() in capsys.readouterr().out


def test_another_commit_could_not_run(setup, monkeypatch, capsys):
    run, tables = setup
    monkeypatch.setattr(forecast_report, "_head_commit", lambda: "b" * 40)
    assert _main(run, tables) == 2
    assert "not at the checked-out commit" in capsys.readouterr().out


def test_a_missing_table_could_not_run(setup, capsys):
    run, tables = setup
    (tables / "jpl-days.csv").unlink()
    assert _main(run, tables) == 2
    assert "missing" in capsys.readouterr().out


def test_a_changed_table_could_not_run(setup, capsys):
    run, tables = setup
    path = tables / "caltech-cars.csv"
    path.write_text(path.read_text().replace("5.000000", "5.500000", 1) + "\n")
    assert _main(run, tables) == 2
    assert "differs" in capsys.readouterr().out


def test_an_unverified_build_could_not_run(setup, capsys):
    run, tables = setup
    (run / "truth-summary.json").write_text("{}")
    assert _main(run, tables) == 2
    assert "not verified" in capsys.readouterr().out


def test_no_recorded_build_could_not_run(tmp_path, capsys):
    assert _main(tmp_path / "none", tmp_path) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
