"""The truth setting (main, or held: plan 3) in the evaluation command (`learn_run`), on the
made-up units of `learn_fake` with a made-up verified truth build of each setting.

1. A run on held tables with the held truth build works, writes the setting into both results
   documents (`inputs`) and says it on the first line of its summary; a run on main tables says
   `main`.
2. Held tables with a main truth build, and main tables with a held truth build, could not run
   (exit 2).
"""

import json

import pytest
from gate_fake import COMMIT, write_build
from learn_fake import small_gate_units, small_matrix
from setting_fake import held_copy

from fleetreserve import gate_run, learn_run, run_verify


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    folder = tmp_path_factory.mktemp("builds")
    main_run, main_tables = write_build(folder, small_gate_units())
    held_run, held_tables = held_copy(main_run, main_tables, folder)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    return {"main": (main_run, main_tables), "held": (held_run, held_tables), "matrix": matrix}


def _run(builds, tables, run, out, monkeypatch) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return learn_run.main(
        [
            "--matrix",
            str(builds["matrix"]),
            "--tables",
            str(builds[tables][1]),
            "--truth-run",
            str(builds[run][0]),
            "--model",
            "empirical",
        ]
    )


@pytest.mark.parametrize("setting", ["main", "held"])
def test_a_learn_run_records_and_states_the_setting_of_its_tables(
    builds, tmp_path, monkeypatch, capsys, setting
):
    out = tmp_path / "out"
    assert _run(builds, setting, setting, out, monkeypatch) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("Model empirical:") and first.endswith(f"{setting} truth setting.")
    for name in (learn_run.RESULTS_NAME, learn_run.GATE_NAME):
        assert json.loads((out / name).read_text())["inputs"]["setting"] == setting


@pytest.mark.parametrize(("tables", "run"), [("held", "main"), ("main", "held")])
def test_tables_and_truth_build_of_different_settings_could_not_run(
    builds, tmp_path, monkeypatch, capsys, tables, run
):
    out = tmp_path / "out"
    assert _run(builds, tables, run, out, monkeypatch) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN") and f"of the {tables} setting" in printed
    assert not list(out.iterdir())
