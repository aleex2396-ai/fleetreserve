"""`dial_choice --what-if-fit-seed N` and `--what-if-fit-levels all|grid` with the neural hazard
model on the made-up units of `test_dial_choice_what_if_models` (readings W5 and W6 of
`dial_choice`).

1. The same seed repeats byte for byte: two runs at seed 1 give the same forecast table and the
   same figures; seed 2 gives another network (its energy columns differ) and each run records
   its seed in `what_if.fit`.
2. `grid` fits the plan's six levels and `all` the eight, whichever level is read: the model
   built for each has 18 and 24 energy outputs (3 quantities, 6 and 8 levels each), both hold the
   outputs of the level read, 0.3 (and 0.7 for the protected energy), and the runs' tables hold
   the same keys; `all` is what `--what-if-kappa` fits without the option (same table byte for
   byte), and `--kappa` fits the grid without the option (same table as `grid`).
3. The runs are what-if runs: only the what-if file names, the label says so, and the summary's
   head prints the seed and the levels fitted; a run without a fit option has no key `fit`.
4. A learned model outside the neural hazard family is refused (exit 2, nothing written); the
   other refusals are in `tests/wp6gate/test_dial_choice_fit_options.py`.
"""

import hashlib
import json

import pytest
from gate_fake import (
    COMMIT,
    GATE_UNITS,
    PLAN_LEVELS,
    gate_cars,
    matrix_document_six,
    write_build,
)

from fleetreserve import dial_choice, gate_run, run_verify
from fleetreserve.forecast_model import read_table
from fleetreserve.gate_config import load
from fleetreserve.models import hazard
from fleetreserve.truth_build import TRAIN_UNITS

GRID = PLAN_LEVELS
ALL = (*GRID, 0.6, 0.7)
NAMES = (
    *(f"main-rounded-hour-k{k:g}-{d}" for k in PLAN_LEVELS for d in ("empirical", "oracle")),
    *(f"main-rounded-hour-k0.3-eps{e:g}-{d}" for e in (0.03, 0.1) for d in ("empirical", "oracle")),
)
ENERGY = ["r_hat", "e_cap", "e_need"]
KEYS = ["unit", "site", "date", "session_id", "kappa"]
WHAT_IF = dial_choice.WHAT_IF_NAMES


def _units() -> dict:
    """The units of `test_dial_choice_what_if_models`: sites the committed site-type mapping
    knows."""
    units = {
        GATE_UNITS[0]: gate_cars(5, "nrel"),
        GATE_UNITS[1]: gate_cars(24, "nrel", early_every=4),
        GATE_UNITS[2]: gate_cars(12, "OSL_1", early_every=3) + gate_cars(12, "BER", early_every=5),
    }
    for j, unit in enumerate(TRAIN_UNITS):
        units[unit] = gate_cars(24, unit, early_every=3 + j)
    return units


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, _units())
    document = matrix_document_six(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix, folder


def _argv(build, model, *extra) -> list[str]:
    run, tables, matrix, _ = build
    arguments = ["--model", model, "--matrix", str(matrix), "--tables", str(tables)]
    return [*arguments, "--truth-run", str(run), "--splits", "4", *extra]


def _run(build, name: str, *extra, model: str = "hazard") -> tuple[int, object]:
    """(exit status, the run's folder) of the command with the options `extra`."""
    out = build[3] / name
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return dial_choice.main(_argv(build, model, *extra)), out


@pytest.fixture(scope="module")
def runs(build):
    """name -> run folder; each ends with exit 0."""
    wanted = {
        "seed1": ("--what-if-kappa", "0.3", "--what-if-fit-seed", "1"),
        "seed1-again": ("--what-if-kappa", "0.3", "--what-if-fit-seed", "1"),
        "seed2": ("--what-if-kappa", "0.3", "--what-if-fit-seed", "2"),
        "grid": ("--what-if-kappa", "0.3", "--what-if-fit-levels", "grid"),
        "all": ("--what-if-kappa", "0.3", "--what-if-fit-levels", "all"),
        "plain": ("--what-if-kappa", "0.3"),
        "kappa-seed1": ("--kappa", "0.2", "--what-if-fit-seed", "1"),
        "kappa-grid-seed1": (
            "--kappa",
            "0.2",
            "--what-if-fit-levels",
            "grid",
            "--what-if-fit-seed",
            "1",
        ),
    }
    found = {}
    for name, extra in wanted.items():
        code, out = _run(build, name, *extra)
        assert code == 0, name
        found[name] = out
    return found


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _document(folder) -> dict:
    return json.loads((folder / WHAT_IF["results"]).read_text())


def _table(folder):
    return read_table(folder / WHAT_IF["table"], what_if=True)


# --- 1. the seed ---------------------------------------------------------------------------------


def test_the_same_seed_repeats_byte_for_byte(runs):
    assert _sha(runs["seed1"] / WHAT_IF["table"]) == _sha(runs["seed1-again"] / WHAT_IF["table"])
    first, again = _document(runs["seed1"]), _document(runs["seed1-again"])
    assert first["units"] == again["units"]
    assert first["dial_choice"] == again["dial_choice"]


def test_another_seed_fits_another_network_and_each_run_records_its_seed(runs):
    one, two = _table(runs["seed1"]), _table(runs["seed2"])
    assert one[KEYS].equals(two[KEYS])
    assert not one[ENERGY].equals(two[ENERGY])
    fits = {name: _document(runs[name])["what_if"]["fit"] for name in ("seed1", "seed2")}
    assert fits["seed1"]["seed"] == 1 and fits["seed1"]["seed_from"] == "--what-if-fit-seed"
    assert fits["seed2"]["seed"] == 2


def test_without_the_seed_option_the_settings_seed_is_used_and_said(runs):
    fit = _document(runs["grid"])["what_if"]["fit"]
    assert fit["seed"] == 0 and fit["seed_from"] == "the settings' seed"


# --- 2. the levels fitted ------------------------------------------------------------------------


def test_grid_and_all_fit_6_and_8_levels_and_the_level_read_is_in_both(build):
    matrix = load(build[2])
    plain = dial_choice.find_family(matrix)[0]
    grid = hazard.make_model(matrix)
    every = hazard.make_model(dial_choice.fit_matrix(matrix, plain))
    assert len(grid.outputs) == 3 * 6 and len(every.outputs) == 3 * 8
    for model in (grid, every):
        for quantity, level in (("rate", 0.3), ("room", 0.3), ("need", 0.7)):
            assert (quantity, level) in model.outputs
    assert grid.outputs.index(("room", 0.3)) != every.outputs.index(("room", 0.3))


def test_the_runs_record_the_levels_they_fitted_and_where_that_came_from(runs):
    fits = {name: _document(folder)["what_if"] for name, folder in runs.items()}
    assert fits["grid"]["levels_fitted"] == list(GRID)
    assert fits["grid"]["fit"]["levels_fitted"] == list(GRID)
    assert fits["grid"]["fit"]["energy_levels_per_quantity"] == 6
    assert fits["grid"]["fit"]["levels_from"] == "--what-if-fit-levels grid"
    assert fits["all"]["levels_fitted"] == list(ALL)
    assert fits["all"]["fit"]["energy_levels_per_quantity"] == 8
    assert fits["seed1"]["fit"]["levels_from"] == "default of --what-if-kappa: all eight"
    assert fits["kappa-seed1"]["fit"]["levels_from"] == "default: the matrix's levels"
    assert fits["kappa-seed1"]["levels_fitted"] == list(GRID)


def test_the_networks_of_grid_and_all_differ_but_the_table_has_the_same_keys(runs):
    grid, every = _table(runs["grid"]), _table(runs["all"])
    assert grid[KEYS].equals(every[KEYS])
    assert set(grid["kappa"]) == {0.05, 0.1, 0.2, 0.3}  # the levels up to the level read, 0.3
    assert not grid[ENERGY].equals(every[ENERGY])


def test_all_is_what_what_if_kappa_fits_without_the_option(runs):
    assert _sha(runs["all"] / WHAT_IF["table"]) == _sha(runs["plain"] / WHAT_IF["table"])
    assert _document(runs["all"])["units"] == _document(runs["plain"])["units"]
    assert "fit" not in _document(runs["plain"])["what_if"]


def test_grid_is_what_kappa_fits_without_the_option(runs):
    one, other = runs["kappa-seed1"], runs["kappa-grid-seed1"]
    assert _sha(one / WHAT_IF["table"]) == _sha(other / WHAT_IF["table"])
    assert _document(one)["units"] == _document(other)["units"]


# --- 3. what-if names, label and summary ---------------------------------------------------------


def test_a_fit_option_run_writes_only_what_if_files_and_labels_itself(runs):
    for name in ("seed1", "grid", "kappa-seed1"):
        folder = runs[name]
        written = sorted(p.name for p in folder.iterdir())
        assert written == sorted(WHAT_IF.values()), name
        assert not (folder / dial_choice.RESULTS_NAME).exists()
        document = _document(folder)
        assert "not a run of the plan" in document["label"]
        assert document["definitions"]["what_if"] == document["what_if"]["readings"]
        counts = json.loads((folder / WHAT_IF["certificate_timeouts"]).read_text())
        assert counts["kappa"] == document["run"]["kappa"]


def test_the_label_and_the_summary_head_name_the_seed_and_the_levels(runs):
    document = _document(runs["seed1"])
    assert "seed 1" in document["what_if"]["what"]
    assert "seed 1" in document["label"]
    head = (runs["seed1"] / WHAT_IF["summary"]).read_text().splitlines()[:4]
    text = "\n".join(head)
    assert "Fit of the network: seed 1 (--what-if-fit-seed); levels fitted" in text
    assert "0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7" in text
    grid_head = (runs["grid"] / WHAT_IF["summary"]).read_text().splitlines()[:4]
    assert any(
        "levels fitted 0.05, 0.1, 0.2, 0.3, 0.4, 0.5 (--what-if-fit-levels grid)" in x
        for x in grid_head
    )


def test_a_kappa_run_with_a_fit_option_keeps_its_kappa_option_text(runs):
    document = _document(runs["kappa-seed1"])
    assert document["run"]["kappa"] == 0.2
    assert "kappa_option" in document["definitions"]
    assert document["what_if"]["on_the_plans_grid"] is True


# --- 4. refusals ---------------------------------------------------------------------------------


def test_a_learned_model_outside_the_hazard_family_is_refused(build, capsys):
    code, out = _run(build, "lgbm-seed", "--what-if-fit-seed", "1", model="lgbm")
    assert code == 2
    text = capsys.readouterr().out
    assert "COULD NOT RUN" in text and "neural hazard family" in text and "lgbm is not one" in text
    assert list(out.iterdir()) == []
