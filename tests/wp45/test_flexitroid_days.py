"""The flexitroid comparison command on made-up tables (`ceiling_fakes`): the sample, the files,
the exit status, the worker count and the refusals."""

import importlib.metadata
import json

import numpy as np
import pytest
from ceiling_fakes import setup_build

from fleetreserve import exact, flexitroid_days
from fleetreserve import flexitroid_match as fm

FILES = (flexitroid_days.RESULTS_NAME, flexitroid_days.TEXT_NAME)


def _run(run, tables, out, workers, patch):
    out.mkdir(exist_ok=True)
    patch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return flexitroid_days.main(
        ["--run", str(run), "--tables", str(tables), "--workers", str(workers)]
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The command on the made-up tables with 1 and with 2 workers."""
    root = tmp_path_factory.mktemp("flexitroid")
    with pytest.MonkeyPatch.context() as patch:
        run, tables = setup_build(root, patch)
        one = _run(run, tables, root / "out1", 1, patch)
        two = _run(run, tables, root / "out2", 2, patch)
    return {"codes": (one, two), "one": root / "out1", "two": root / "out2"}


def _doc(folder):
    return json.loads((folder / flexitroid_days.RESULTS_NAME).read_text())


def test_the_sample_is_the_block_sample_of_50_days(runs):
    plan = _doc(runs["one"])["sample"]
    assert plan["size"] == 50 and plan["seed"] == 2026
    assert plan["sampled_days"] == {
        "nrel-before-may-2019": 13,
        "nrel-from-may-2019": 17,
        "norway-development-pooled": 20,
    }


def test_empty_fleet_days_could_not_run_and_the_rest_match(runs):
    # Every fifth made-up day has only cars unplugged within the decision hour: its fleet is
    # empty, and flexitroid has no empty aggregate, so the day could not run and the run ends 2.
    doc = _doc(runs["one"])
    assert runs["codes"] == (2, 2)
    empty = [d for d in doc["days"] if d["cars"] == 0]
    assert empty and all(d["verdict"] == fm.COULD_NOT_RUN for d in empty)
    assert doc["counts"]["days"] == 50
    assert doc["counts"]["strict"] == {
        fm.MATCH: 50 - len(empty),
        fm.DIFFERS: 0,
        fm.COULD_NOT_RUN: len(empty),
    }
    text = (runs["one"] / flexitroid_days.TEXT_NAME).read_text()
    assert text.startswith("COULD NOT RUN: ")
    assert doc["flexitroid"]["commit"] == "098963567e6d8ca39b7b6dffebc6259b731f5dc3"


def test_results_files_do_not_depend_on_the_worker_count(runs):
    for name in FILES:
        assert (runs["one"] / name).read_bytes() == (runs["two"] / name).read_bytes()
    seconds = json.loads((runs["two"] / flexitroid_days.SECONDS_NAME).read_text())
    assert seconds["workers"] == 2 and len(seconds["days"]) == 50


def test_a_wrong_flexitroid_function_is_a_failed_line_and_exit_1(tmp_path, monkeypatch):
    real = fm.flexitroid_functions

    def scaled(aggregate, n_service):
        found = real(aggregate, n_service)
        return exact.SetFunctions(n_service, found.lower, np.asarray(found.upper) * 1.01)

    monkeypatch.setattr(fm, "flexitroid_functions", scaled)
    run, tables = setup_build(tmp_path, monkeypatch)
    assert _run(run, tables, tmp_path / "out", 1, monkeypatch) == 1
    text = (tmp_path / "out" / flexitroid_days.TEXT_NAME).read_text()
    assert text.startswith("FAILED: ")
    assert _doc(tmp_path / "out")["counts"]["strict"][fm.DIFFERS] > 0


def test_all_days_matching_is_exit_0():
    doc = {"counts": {"strict": {fm.MATCH: 50, fm.DIFFERS: 0, fm.COULD_NOT_RUN: 0}}}
    assert flexitroid_days.exit_status(doc) == 0
    doc["counts"]["strict"][fm.MATCH] = 49
    assert flexitroid_days.exit_status(doc) == 2


def test_fewer_than_50_counting_days_could_not_run(tmp_path, monkeypatch, capsys):
    run, tables = setup_build(tmp_path, monkeypatch, scale=0.5)
    assert _run(run, tables, tmp_path / "out", 1, monkeypatch) == 2
    assert "fewer than the sample size 50" in capsys.readouterr().out


def test_a_missing_table_could_not_run(tmp_path, monkeypatch, capsys):
    run, tables = setup_build(tmp_path, monkeypatch)
    (tables / "nrel-from-may-2019-cars.csv").unlink()
    assert _run(run, tables, tmp_path / "out", 1, monkeypatch) == 2
    assert "is missing" in capsys.readouterr().out


def test_without_flexitroid_it_could_not_run(tmp_path, monkeypatch, capsys):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    run, tables = setup_build(tmp_path, monkeypatch)
    assert _run(run, tables, tmp_path / "out", 1, monkeypatch) == 2
    assert capsys.readouterr().out == "COULD NOT RUN: flexitroid is not installed\n"
