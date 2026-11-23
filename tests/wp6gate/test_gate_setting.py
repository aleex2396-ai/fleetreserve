"""The truth setting (main, or held: plan 3) in the gate run, on made-up gate units with a made-up
verified truth build of each setting (`setting_fake.held_copy`).

1. A gate run on held tables with the held truth build works, writes the setting `held` into its
   results document (`inputs`) and states it on the first line of its summary; a run on main
   tables writes `main`.
2. Held tables with a main truth build, and main tables with a held truth build, could not run
   (exit 2), although the table bytes are the ones the build recorded.
3. Without `--truth-run` the latest run of the tables' own setting is used; a held run that is
   newer than the latest main run is never taken for the main tables, nor the reverse.
"""

import json
import shutil

import pytest
from gate_fake import COMMIT, GATE_UNITS, ROUNDED, gate_units, matrix_document, write_build
from setting_fake import held_copy

from fleetreserve import ceiling_report, forecast_report, gate_run, run_verify


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    for module in (gate_run, ceiling_report, forecast_report):
        monkeypatch.setattr(module, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    folder = tmp_path_factory.mktemp("builds")
    units = gate_units()
    units.update(caltech=units[GATE_UNITS[1]], jpl=units[GATE_UNITS[1]])  # for forecast_report
    main_run, main_tables = write_build(folder, units)
    held_run, held_tables = held_copy(main_run, main_tables, folder)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(ROUNDED)))
    return {
        "main": (main_run, main_tables),
        "held": (held_run, held_tables),
        "matrix": matrix,
        "folder": folder,
    }


def _arguments(builds, tables, run) -> list[str]:
    return [
        "--matrix",
        str(builds["matrix"]),
        "--tables",
        str(builds[tables][1]),
        "--truth-run",
        str(builds[run][0]),
    ]


def _run(builds, tables, run, out, monkeypatch, *extra) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return gate_run.main(
        [*_arguments(builds, tables, run), "--limit-days", "55", "--only", ROUNDED[0], *extra]
    )


@pytest.mark.parametrize("setting", ["main", "held"])
def test_a_gate_run_records_the_setting_of_its_tables(builds, tmp_path, monkeypatch, setting):
    out = tmp_path / "out"
    assert _run(builds, setting, setting, out, monkeypatch) == 0
    document = json.loads((out / gate_run.RESULTS_NAME).read_text())
    assert document["inputs"]["setting"] == setting
    assert document["inputs"]["truth_run"] == builds[setting][0].name
    summary = (out / gate_run.SUMMARY_NAME).read_text().splitlines()
    assert summary[0] == f"Truth tables: {setting} setting."
    assert gate_run.TRIAL_TEXT in summary[1]


@pytest.mark.parametrize(("tables", "run"), [("held", "main"), ("main", "held")])
def test_tables_and_truth_build_of_different_settings_could_not_run(
    builds, tmp_path, monkeypatch, capsys, tables, run
):
    out = tmp_path / "out"
    assert _run(builds, tables, run, out, monkeypatch) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN") and f"of the {tables} setting" in printed
    assert not list(out.iterdir())  # nothing was written


def test_the_latest_run_of_the_tables_own_setting_is_used(builds, tmp_path, monkeypatch):
    runs = tmp_path / "runs.noindex"
    runs.mkdir()
    for run in ("main", "held"):
        shutil.copytree(builds[run][0], runs / builds[run][0].name)
    # The held run is the newer one by name.
    assert builds["held"][0].name > builds["main"][0].name
    monkeypatch.setattr(gate_run, "TOP", tmp_path)
    for setting in ("main", "held"):
        found = gate_run.check_inputs(None, builds[setting][1], GATE_UNITS)
        assert found["truth_run"] == builds[setting][0].name
        assert found["setting"] == setting
    (runs / builds["held"][0].name).rename(runs / "20260103T000000Z-wp2-truth-held")
    shutil.rmtree(runs / builds["main"][0].name)
    with pytest.raises(gate_run.CouldNotRun, match=r"runs.noindex/\*-wp2-truth\)"):
        gate_run.check_inputs(None, builds["main"][1], GATE_UNITS)


@pytest.mark.parametrize("module", [ceiling_report, forecast_report])
def test_the_reports_check_the_setting_too(builds, module):
    for setting in ("main", "held"):
        run, tables = builds[setting]
        assert module.check_inputs(run, tables)["setting"] == setting
    with pytest.raises(module.CouldNotRun, match="of the main setting"):
        module.check_inputs(builds["main"][0], builds["held"][1])
    with pytest.raises(module.CouldNotRun, match="of the held setting"):
        module.check_inputs(builds["held"][0], builds["main"][1])
