"""The ceiling report command on made-up tables (see `ceiling_fakes`): the sample, the files, the
refusals, the T4 check's failure and the worker count."""

import csv
import json

import pytest
from ceiling_fakes import setup_build

from fleetreserve import ceiling_report, ceiling_values

FILES = (
    ceiling_report.RESULTS_NAME,
    ceiling_report.TEXT_NAME,
    ceiling_report.DAYS_NAME,
    ceiling_report.SAMPLE_NAME,
)


def _run(run, tables, out, workers, patch):
    out.mkdir(exist_ok=True)
    patch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return ceiling_report.main(
        ["--run", str(run), "--tables", str(tables), "--workers", str(workers)]
    )


def _table(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The command on the made-up tables with 1 and with 3 workers."""
    root = tmp_path_factory.mktemp("ceiling")
    with pytest.MonkeyPatch.context() as patch:
        run, tables = setup_build(root, patch)
        one = _run(run, tables, root / "out1", 1, patch)
        three = _run(run, tables, root / "out3", 3, patch)
    return {"codes": (one, three), "one": root / "out1", "three": root / "out3"}


def test_the_run_ends_with_exit_0(runs):
    assert runs["codes"] == (0, 0)


def test_the_sample_is_50_days_proportional_with_each_unit_present(runs):
    document = json.loads((runs["one"] / ceiling_report.RESULTS_NAME).read_text())
    plan = document["sample"]
    assert plan["counting_days"] == {
        "nrel-before-may-2019": 20,
        "nrel-from-may-2019": 25,
        "norway-development-pooled": 30,
    }
    assert plan["sampled_days"] == {
        "nrel-before-may-2019": 13,
        "nrel-from-may-2019": 17,
        "norway-development-pooled": 20,
    }
    rows = _table(runs["one"] / ceiling_report.SAMPLE_NAME)
    assert len(rows) == 50
    assert len({(r["unit"], r["site"], r["date"]) for r in rows}) == 50
    order = [(r["unit"], r["date"], r["site"]) for r in rows]
    assert order == sorted(order)
    days = _table(runs["one"] / ceiling_report.DAYS_NAME)
    assert len(days) == 75
    order = [(r["unit"], r["date"], r["site"]) for r in days]
    assert order == sorted(order)
    assert sum(r["in_block_sample"] == "True" for r in days) == 50


def test_hourly_values_cover_every_day_and_the_tree_matches_the_causal_value(runs):
    rows = _table(runs["one"] / ceiling_report.DAYS_NAME)
    outcomes = {r["hourly_best_causal_outcome"] for r in rows}
    assert outcomes == {"optimal", "empty fleet"}
    sample = _table(runs["one"] / ceiling_report.SAMPLE_NAME)
    for r in sample:
        assert float(r["hourly_tree_kwh"]) == pytest.approx(float(r["hourly_best_causal_kwh"]))


def test_t4_holds_on_the_hypotheses_days_and_a_real_check_ran(runs):
    document = json.loads((runs["one"] / ceiling_report.RESULTS_NAME).read_text())
    check = document["block_sample"]["t4_check"]
    assert check["held"] is True
    assert check["days_checked"] >= 5
    assert check["violations"] == []
    text = (runs["one"] / ceiling_report.TEXT_NAME).read_text()
    assert "all checks held" in text
    assert "mean V_caus/V_hind = " in text
    assert "the owner's rule threshold is 0.99" in text


def test_the_day_kinds_are_told_apart(runs):
    rows = _table(runs["one"] / ceiling_report.SAMPLE_NAME)
    hypotheses = [r for r in rows if r["meets_t4_hypotheses"] == "True"]
    empty = [r for r in rows if r["cars"] == "0"]
    assert hypotheses and empty
    assert all(r["in_window_departure"] == "False" for r in hypotheses)
    assert all(float(r["block_v_hind_kwh"]) == 0.0 for r in empty)
    assert all(r["meets_t4_hypotheses"] == "False" for r in empty)


def test_results_files_do_not_depend_on_the_worker_count(runs):
    for name in FILES:
        assert (runs["one"] / name).read_bytes() == (runs["three"] / name).read_bytes(), name
    for name in FILES:
        assert "seconds" not in (runs["one"] / name).read_text()
    timing = json.loads((runs["three"] / ceiling_report.SECONDS_NAME).read_text())
    assert timing["workers"] == 3
    assert json.loads((runs["one"] / ceiling_report.SECONDS_NAME).read_text())["workers"] == 1


def test_a_broken_equality_on_a_hypotheses_day_is_a_failed_line_and_exit_1(
    tmp_path, monkeypatch, capsys
):
    run, tables = setup_build(tmp_path, monkeypatch)
    real = ceiling_values.block_values

    def broken(functions, rows):
        found = real(functions, rows)
        value, outcome, seconds = found["block_caus"]
        found["block_caus"] = (None if value is None else 0.5 * value, outcome, seconds)
        return found

    monkeypatch.setattr(ceiling_values, "block_values", broken)
    assert _run(run, tables, tmp_path / "out", 1, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "FAILED: " in out
    assert "differs from V_caus" in out
    assert "on a day meeting T4's hypotheses" in out
    assert "CHECKS FAILED" in out
    document = json.loads((tmp_path / "out" / ceiling_report.RESULTS_NAME).read_text())
    assert document["block_sample"]["t4_check"]["held"] is False


def test_fewer_than_50_counting_days_could_not_run(tmp_path, monkeypatch, capsys):
    run, tables = setup_build(tmp_path, monkeypatch, scale=0.5)
    assert _run(run, tables, tmp_path / "out", 1, monkeypatch) == 2
    out = capsys.readouterr().out
    assert "COULD NOT RUN" in out
    assert "fewer than the sample size 50" in out


def test_another_commit_could_not_run(tmp_path, monkeypatch, capsys):
    run, tables = setup_build(tmp_path, monkeypatch)
    monkeypatch.setattr(ceiling_report, "_head_commit", lambda: "b" * 40)
    assert _run(run, tables, tmp_path / "out", 1, monkeypatch) == 2
    assert "not at the checked-out commit" in capsys.readouterr().out


def test_a_missing_table_could_not_run(tmp_path, monkeypatch, capsys):
    run, tables = setup_build(tmp_path, monkeypatch)
    (tables / "nrel-from-may-2019-days.csv").unlink()
    assert _run(run, tables, tmp_path / "out", 1, monkeypatch) == 2
    assert "missing" in capsys.readouterr().out


def test_a_changed_table_could_not_run(tmp_path, monkeypatch, capsys):
    run, tables = setup_build(tmp_path, monkeypatch)
    path = tables / "norway-development-pooled-cars.csv"
    path.write_text(path.read_text().replace("3.000000", "3.500000", 1))
    assert _run(run, tables, tmp_path / "out", 1, monkeypatch) == 2
    assert "differs" in capsys.readouterr().out


def test_an_unverified_build_could_not_run(tmp_path, monkeypatch, capsys):
    run, tables = setup_build(tmp_path, monkeypatch)
    (run / "truth-summary.json").write_text("{}")
    assert _run(run, tables, tmp_path / "out", 1, monkeypatch) == 2
    assert "not verified" in capsys.readouterr().out


def test_no_recorded_build_could_not_run(tmp_path, capsys):
    assert ceiling_report.main(["--run", str(tmp_path / "none"), "--tables", str(tmp_path)]) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out


def test_without_a_run_folder_variable_the_report_is_only_printed(tmp_path, monkeypatch, capsys):
    run, tables = setup_build(tmp_path, monkeypatch)
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    code = ceiling_report.main(["--run", str(run), "--tables", str(tables), "--workers", "1"])
    assert code == 0
    assert "(a) BLOCK SAMPLE: 50 counting days" in capsys.readouterr().out
    assert not list(tmp_path.glob("ceiling-*"))
