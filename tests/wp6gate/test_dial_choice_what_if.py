"""`dial_choice --what-if-kappa <level>`: the dial family measured at a level off plan section 2's
grid (0.6 and 0.7; the grid is 0.05, 0.1, 0.2, 0.3, 0.4, 0.5), as a labelled what-if, on the
made-up gate units of `test_dial_choice_kappa` (the empirical model), and `kappa_choice` reading
such runs apart from the six runs of the grid. The matrix of the tests lists the six levels.

1. Without the option a level off the grid is refused as before (exit 2), and so are a level off
   the what-if list, and `--kappa` given with `--what-if-kappa`.
2. At 0.6 the run ends with exit 0 and writes only the what-if file names (no `dial-choice.json`):
   the results' label, the key `what_if` and the summary's first line say "what-if, level not on
   the plan's grid"; the configurations are the family's with `-k0.6-` (the configuration
   without rows and its oracle partner made too, reading W2); the forecast table holds the levels
   of the grid and 0.6; the driver-risk rows run over those seven levels (reading W1).
3. At 0.7, the top of the what-if list, the run ends with exit 0.
4. At a grid level, 0.3 or 0.5 (0.4 and 0.5 were what-if levels before the grid had six), the
   option gives the figures of the run with `--kappa` at that level (the empirical model fits
   nothing, so its forecasts do not depend on the levels fitted), under the what-if label.
5. `kappa_choice` with the six grid runs and two what-if runs (0.6, 0.7): exit 0; the plan
   readings and every grid pair are those of the six grid runs alone; the what-if pairs are
   those of the what-if runs; a second reading over all levels given is labelled what-if; the
   summary prints the what-if block after the plan block, which is unchanged.
6. `kappa_choice` refuses (exit 2) two what-if runs at one level, and a what-if results file
   offered as a run of the grid is not counted as one.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from gate_fake import (
    COMMIT,
    GATE_UNITS,
    PLAN_LEVELS,
    gate_cars,
    gate_units,
    matrix_document_six,
    write_build,
)

from fleetreserve import dial_choice, driver_risk, gate_run, kappa_choice, run_verify
from fleetreserve.forecast_model import KEY_COLUMNS, TABLE_COLUMNS, check_table, read_table

PLAINS = tuple(
    f"main-rounded-hour-k{k:g}-{d}" for k in PLAN_LEVELS for d in ("empirical", "oracle")
)
DIALS = tuple(
    f"main-rounded-hour-k0.3-eps{e:g}-{d}" for e in (0.03, 0.1) for d in ("empirical", "oracle")
)
NAMES = PLAINS + DIALS
SPLITS = 6
LEVELS = PLAN_LEVELS
CALIBRATABLE = GATE_UNITS[1:]
WHAT_IF = "what-if, level not on the plan's grid"


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


def units() -> dict:
    """The made-up gate units of `test_dial_choice_kappa`."""
    found = gate_units()
    found[GATE_UNITS[1]] = gate_cars(60, early_every=3, early_out=12.0)
    found[GATE_UNITS[2]] = gate_cars(30, "B", early_every=3, early_out=12.0) + gate_cars(
        30, "C", early_every=4, early_out=12.0
    )
    return found


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document_six(NAMES)))
    return run, tables, matrix


def _argv(build, *extra) -> list[str]:
    run, tables, matrix = build
    arguments = ["--model", "empirical", "--matrix", str(matrix)]
    arguments += ["--tables", str(tables), "--truth-run", str(run), "--splits", str(SPLITS)]
    return [*arguments, *extra]


def _run(build, out: Path, *extra) -> int:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        out.mkdir(parents=True)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return dial_choice.main(_argv(build, *extra))


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    """(option, level) -> (exit status, run folder), for the grid runs and the what-if runs."""
    folder = tmp_path_factory.mktemp("runs")
    found = {}
    for kappa in LEVELS:
        out = folder / f"k{kappa:g}"
        found[("--kappa", kappa)] = (_run(build, out, "--kappa", str(kappa)), out)
    for kappa in (0.3, 0.5, 0.6, 0.7):
        out = folder / f"what-if-k{kappa:g}"
        found[("what-if", kappa)] = (_run(build, out, "--what-if-kappa", str(kappa)), out)
    return found


def _what_if_document(out: Path) -> dict:
    return json.loads((out / dial_choice.WHAT_IF_NAMES["results"]).read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- 1. refusals ---------------------------------------------------------------------------------


def test_without_the_option_a_level_off_the_grid_is_refused(build, capsys):
    assert dial_choice.main(_argv(build, "--kappa", "0.6")) == 2
    assert "--kappa must be one of [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]" in capsys.readouterr().out


def test_a_level_off_the_what_if_list_is_refused(build, capsys):
    assert dial_choice.main(_argv(build, "--what-if-kappa", "0.45")) == 2
    assert "--what-if-kappa must be one of" in capsys.readouterr().out


def test_both_options_together_are_refused(build, capsys):
    assert dial_choice.main(_argv(build, "--kappa", "0.3", "--what-if-kappa", "0.6")) == 2
    assert "not both" in capsys.readouterr().out


def test_the_levels_off_the_grid_need_the_what_if_list():
    assert driver_risk.levels_up_to(0.5) == LEVELS
    assert driver_risk.levels_up_to(0.6) == (*LEVELS, 0.6)
    assert driver_risk.levels_up_to(0.7) == (*LEVELS, 0.6, 0.7)
    with pytest.raises(ValueError, match="kappa"):
        driver_risk.levels_up_to(0.45)


def test_a_forecast_table_at_a_level_off_the_grid_needs_the_what_if_flag():
    assert set(check_table(_table_at(0.4))["kappa"]) == {0.4}  # a level of the grid
    table = _table_at(0.6)
    with pytest.raises(ValueError, match="kappa must be one of"):
        check_table(table)
    assert set(check_table(table, what_if=True)["kappa"]) == {0.6}
    with pytest.raises(ValueError, match="kappa must be one of"):
        check_table(_table_at(0.45), what_if=True)


def _table_at(kappa: float) -> pd.DataFrame:
    """A one-row forecast table at level `kappa`."""
    row = {c: "x" for c in KEY_COLUMNS} | {c: 1.0 for c in TABLE_COLUMNS if c not in KEY_COLUMNS}
    row |= {"kappa": kappa, "rho": 0.0}
    return pd.DataFrame([row])[list(TABLE_COLUMNS)]


# --- 2. a run at 0.6 -----------------------------------------------------------------------------


def test_a_run_at_0_6_is_a_labelled_what_if(runs):
    code, out = runs[("what-if", 0.6)]
    assert code == 0
    written = sorted(p.name for p in out.iterdir())
    assert written == sorted(dial_choice.WHAT_IF_NAMES.values())
    assert all(name.startswith("dial-choice-what-if") for name in written)
    assert not (out / dial_choice.RESULTS_NAME).exists()
    document = _what_if_document(out)
    assert WHAT_IF in document["label"]
    assert document["label"] != f"dial choice: {dial_choice.LABEL}"
    what_if = document["what_if"]
    assert what_if["kappa"] == 0.6 and what_if["on_the_plans_grid"] is False
    assert what_if["plan_grid"] == list(LEVELS)
    assert what_if["levels_fitted"] == [*LEVELS, 0.6, 0.7]
    assert what_if["driver_risk_levels"] == [*LEVELS, 0.6]
    assert document["run"]["kappa"] == 0.6
    names = [
        "main-rounded-hour-k0.6-empirical",
        "main-rounded-hour-k0.6-eps0.03-empirical",
        "main-rounded-hour-k0.6-eps0.1-empirical",
    ]
    for unit in CALIBRATABLE:
        configurations = document["units"][unit]["configurations"]
        assert list(configurations) == names
        partners = [c["gate_reading"]["oracle_partner"] for c in configurations.values()]
        assert partners == [n.replace("-empirical", "-oracle") for n in names]
        assert all("tiers" in c for c in configurations.values())
    summary = (out / dial_choice.WHAT_IF_NAMES["summary"]).read_text()
    assert WHAT_IF.upper() in summary.splitlines()[0]
    table = read_table(out / dial_choice.WHAT_IF_NAMES["table"], what_if=True)
    assert set(table["kappa"]) == {*LEVELS, 0.6}
    counts = json.loads((out / dial_choice.WHAT_IF_NAMES["certificate_timeouts"]).read_text())
    assert counts["kappa"] == 0.6


# --- 3. the top of the what-if list --------------------------------------------------------------


def test_a_run_at_0_7_ends_with_exit_0(runs):
    code, out = runs[("what-if", 0.7)]
    assert code == 0
    document = _what_if_document(out)
    assert document["failed_configurations"] == []
    assert document["what_if"]["driver_risk_levels"] == [*LEVELS, 0.6, 0.7]


# --- 4. a grid level under the option ------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.3, 0.5])
def test_at_a_grid_level_the_option_gives_the_grid_runs_figures_under_the_what_if_label(
    runs, kappa
):
    (code, plan), (code_w, what_if) = runs[("--kappa", kappa)], runs[("what-if", kappa)]
    assert code == code_w == 0
    grid = json.loads((plan / dial_choice.RESULTS_NAME).read_text())
    found = _what_if_document(what_if)
    assert found["what_if"]["on_the_plans_grid"] is True
    assert found["what_if"]["kappa"] == kappa and found["what_if"]["plan_grid"] == list(LEVELS)
    assert "not a run of the plan" in found["label"]
    assert found["units"] == grid["units"]
    assert found["dial_choice"] == grid["dial_choice"]
    assert _sha(plan / dial_choice.TABLE_NAME) == _sha(what_if / dial_choice.WHAT_IF_NAMES["table"])


# --- 5. kappa_choice with what-if runs -----------------------------------------------------------


def _recorded(source: Path, folder: Path) -> None:
    """A copy of the run folder `source` with a made-up manifest, as `run_record` writes it."""
    shutil.copytree(source, folder)
    outputs = {p.name: _sha(p) for p in sorted(folder.iterdir())}
    manifest = {"commit": COMMIT, "command": ["python", "-m", "fleetreserve.dial_choice"]}
    manifest |= {"start": "s", "end": "e", "exit_code": 0, "outputs": outputs}
    (folder / "manifest.json").write_text(json.dumps(manifest))


def _files(runs, folder: Path, what_if_levels) -> list[str]:
    files = []
    for kappa in LEVELS:
        _recorded(runs[("--kappa", kappa)][1], folder / f"k{kappa:g}")
        files.append(str(folder / f"k{kappa:g}" / dial_choice.RESULTS_NAME))
    for kappa in what_if_levels:
        _recorded(runs[("what-if", kappa)][1], folder / f"what-if-k{kappa:g}")
        files.append(str(folder / f"what-if-k{kappa:g}" / dial_choice.WHAT_IF_NAMES["results"]))
    return files


def _joint(files, out: Path, monkeypatch) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return kappa_choice.main(files)


def test_kappa_choice_reads_the_what_if_runs_apart(runs, tmp_path, monkeypatch):
    grid_only = tmp_path / "grid-only"
    assert _joint(_files(runs, tmp_path / "a", ()), grid_only, monkeypatch) == 0
    both = tmp_path / "both"
    assert _joint(_files(runs, tmp_path / "b", (0.7, 0.6)), both, monkeypatch) == 0
    alone = json.loads((grid_only / kappa_choice.RESULTS_NAME).read_text())
    joint = json.loads((both / kappa_choice.RESULTS_NAME).read_text())
    assert "what_if" not in alone
    assert joint["readings"] == alone["readings"]
    assert joint["run"]["kappas"] == list(LEVELS) == alone["run"]["kappas"]
    for unit in GATE_UNITS:
        assert joint["units"][unit] == alone["units"][unit]
    what_if = joint["what_if"]
    assert WHAT_IF in what_if["label"]
    assert [r["kappa"] for r in what_if["runs"]] == [0.6, 0.7]
    for kappa in (0.6, 0.7):
        document = _what_if_document(runs[("what-if", kappa)][1])
        for unit in CALIBRATABLE:
            for found in document["units"][unit]["configurations"].values():
                key = kappa_choice.key(found["driver_risk"])
                cell = what_if["units"][unit]["pairs"]["A"][f"{kappa:g}"][key]
                for field in kappa_choice.CELL_FIELDS:
                    assert cell[field] == found["tiers"]["A"][field], (kappa, unit, key)
    over_all = what_if["readings_over_all_levels"]
    assert over_all["levels"] == [*LEVELS, 0.6, 0.7]
    assert WHAT_IF in over_all["label"]
    assert set(over_all["R2"]["sums_by_kappa"]) == {
        "0.05",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
    }
    plan_text = (grid_only / kappa_choice.SUMMARY_NAME).read_text()
    text = (both / kappa_choice.SUMMARY_NAME).read_text()
    assert text.startswith(plan_text)
    rest = text[len(plan_text) :]
    assert WHAT_IF.upper() in rest and "0.6" in rest and "0.7" in rest


def test_a_what_if_run_at_a_grid_level_is_printed_and_in_no_reading(runs, tmp_path, monkeypatch):
    out = tmp_path / "joint"
    assert _joint(_files(runs, tmp_path / "a", (0.5, 0.6)), out, monkeypatch) == 0
    what_if = json.loads((out / kappa_choice.RESULTS_NAME).read_text())["what_if"]
    assert [r["kappa"] for r in what_if["runs"]] == [0.5, 0.6]
    assert "0.5" in what_if["units"][CALIBRATABLE[0]]["pairs"]["A"]
    assert what_if["readings_over_all_levels"]["levels"] == [*LEVELS, 0.6]


# --- 6. refusals of kappa_choice -----------------------------------------------------------------


def test_two_what_if_runs_at_one_level_could_not_run(runs, tmp_path, monkeypatch, capsys):
    files = _files(runs, tmp_path / "a", (0.6,))
    _recorded(runs[("what-if", 0.6)][1], tmp_path / "again")
    files.append(str(tmp_path / "again" / dial_choice.WHAT_IF_NAMES["results"]))
    assert _joint(files, tmp_path / "joint", monkeypatch) == 2
    assert "two what-if runs" in capsys.readouterr().out


def test_a_what_if_run_does_not_count_as_a_run_of_the_grid(runs, tmp_path, monkeypatch, capsys):
    files = _files(runs, tmp_path / "a", (0.3,))
    files = [f for f in files if "/k0.3/" not in f]
    assert _joint(files, tmp_path / "joint", monkeypatch) == 2
    assert "one run at each level of plan section 2" in capsys.readouterr().out
