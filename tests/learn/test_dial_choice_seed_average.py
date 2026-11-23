"""`dial_choice --what-if-fit-seeds N1,N2` with the neural hazard model on the made-up units of
`test_dial_choice_what_if_fit` (`seed_average`, reading A1; `dial_choice` reading W5).

Four runs at `--what-if-kappa 0.3`: seeds 1,2 averaged, seeds 2,1 averaged, seed 1 alone and seed 2
alone; the model each run fits is kept by a spy on `dial_choice.make_factory` (it passes every
argument on and returns what the command returns).

1. Each seed is used and each network once: the averaged model has two members, at seeds 1 and
   2, whose networks are not one object and differ in their weights; each member's weights equal,
   tensor for tensor, those of the single-seed run at its seed (fitted exactly as
   `--what-if-fit-seed N` fits it).
2. The average follows reading A1, recomputed apart from `seed_average`: on the forecast frame,
   the mean of the two single-seed models' survival curves, the departure columns read from it by
   `HazardModel.departure_columns`, and the energy columns the mean of the two models' values,
   give the averaged run's table (to 1e-12).
3. The average over [1, 2] differs from each single fit (in the energy columns), and the order of
   the list changes nothing: the tables of 1,2 and 2,1 are the same byte for byte, and so are
   their figures.
4. The run is a labelled what-if: only the what-if files, `what_if.fit` holds the sorted seed list
   (and no `seed`), the levels fitted and what is averaged; label and summary head say
   "seeds 1, 2 averaged", the head names A1 as a reading of `seed_average` (not of
   `dial_choice`) and says that the energy columns are means of the members' quantiles.
"""

import hashlib
import json

import numpy as np
import pytest
from gate_fake import COMMIT, matrix_document_six, write_build
from test_dial_choice_what_if_fit import NAMES, _units

from fleetreserve import dial_choice, dial_summary, gate_run, learn_run, run_verify
from fleetreserve.forecast_model import TABLE_COLUMNS, read_table
from fleetreserve.models.hazard import SetCurves, forecast_table

ENERGY = ["r_hat", "e_cap", "e_need"]
WHAT_IF = dial_choice.WHAT_IF_NAMES
ALL = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, _units())
    document = matrix_document_six(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix, folder


def _run(build, name: str, *extra) -> tuple[int, object, dict]:
    """(exit status, the run's folder, what the spy kept: model, units, levels, table)."""
    run, tables, matrix, folder = build
    out = folder / name
    out.mkdir()
    argv = ["--model", "hazard", "--matrix", str(matrix), "--tables", str(tables)]
    argv += ["--truth-run", str(run), "--splits", "4", "--what-if-kappa", "0.3", *extra]
    kept = {}
    original = dial_choice.make_factory

    def spy(name, model, matrix, units, fit_frames, kappa, what_if=False):
        found = original(name, model, matrix, units, fit_frames, kappa, what_if)
        kept.update(model=model, units=units, gate_units=matrix.gate_units, kappa=kappa)
        kept["table"] = found[1]
        return found

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setattr(dial_choice, "make_factory", spy)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return dial_choice.main(argv), out, kept


@pytest.fixture(scope="module")
def runs(build):
    wanted = {
        "avg12": ("--what-if-fit-seeds", "1,2"),
        "avg21": ("--what-if-fit-seeds", "2,1"),
        "seed1": ("--what-if-fit-seed", "1"),
        "seed2": ("--what-if-fit-seed", "2"),
    }
    found = {}
    for name, extra in wanted.items():
        code, out, kept = _run(build, name, *extra)
        assert code == 0, name
        found[name] = (out, kept)
    return found


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _document(folder) -> dict:
    return json.loads((folder / WHAT_IF["results"]).read_text())


def _weights(net) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}


# --- 1. every seed used, each network once -------------------------------------------------------


def test_each_seed_is_one_member_fitted_as_its_single_seed_run(runs):
    average = runs["avg12"][1]["model"]
    members = average.members
    assert [m.settings.seed for m in members] == [1, 2]
    assert members[0] is not members[1] and members[0].net is not members[1].net
    one, two = _weights(members[0].net), _weights(members[1].net)
    assert any(not np.array_equal(one[k], two[k]) for k in one)
    for member, single in zip(members, ("seed1", "seed2"), strict=True):
        alone = _weights(runs[single][1]["model"].net)
        assert alone.keys() == _weights(member.net).keys()
        assert all(np.array_equal(alone[k], _weights(member.net)[k]) for k in alone), single


# --- 2. the average of reading A1, recomputed ----------------------------------------------------


def test_the_table_is_the_mean_curve_and_the_mean_energy_of_the_single_fits(runs):
    kept = runs["avg12"][1]
    singles = [runs[name][1]["model"] for name in ("seed1", "seed2")]
    frame = learn_run.forecast_frame(False, kept["units"], kept["gate_units"], None)
    levels = [k for k in ALL if k <= kept["kappa"]]
    asked = {float(k) for k in levels}
    curves = [model.set_curves(frame) for model in singles]
    mean = SetCurves(
        curves[0].n,
        curves[0].rows,
        curves[0].kinds,
        [(a + b) / 2 for a, b in zip(curves[0].survival, curves[1].survival, strict=True)],
        curves[0].energy,
    )
    departure = singles[0].departure_columns(mean, asked)
    each = [m.energy_columns(c, asked) for m, c in zip(singles, curves, strict=True)]
    energy = {k: {n: (each[0][k][n] + each[1][k][n]) / 2 for n in ENERGY} for k in asked}
    expected = forecast_table(frame, levels, singles[0].rho, departure, energy)
    table = kept["table"]
    assert list(table.columns) == list(TABLE_COLUMNS) and len(table) == len(expected) > 0
    numbers = [c for c in TABLE_COLUMNS if c not in ("unit", "site", "date", "session_id")]
    for column in ("unit", "site", "date", "session_id"):
        assert table[column].tolist() == expected[column].tolist()
    np.testing.assert_allclose(
        table[numbers].to_numpy(float), expected[numbers].to_numpy(float), rtol=0, atol=1e-12
    )


# --- 3. differs from each single fit; order of the list ------------------------------------------


def test_the_average_differs_from_each_single_fit(runs):
    average = read_table(runs["avg12"][0] / WHAT_IF["table"], what_if=True)
    for single in ("seed1", "seed2"):
        alone = read_table(runs[single][0] / WHAT_IF["table"], what_if=True)
        assert not average[ENERGY].equals(alone[ENERGY]), single


def test_the_order_of_the_list_changes_nothing(runs):
    one, other = runs["avg12"][0], runs["avg21"][0]
    assert _sha(one / WHAT_IF["table"]) == _sha(other / WHAT_IF["table"])
    assert _document(one)["units"] == _document(other)["units"]
    assert _document(one)["what_if"] == _document(other)["what_if"]


# --- 4. labelled ---------------------------------------------------------------------------------


def test_the_run_is_a_labelled_what_if_that_records_the_seed_list(runs):
    folder = runs["avg21"][0]
    assert sorted(p.name for p in folder.iterdir()) == sorted(WHAT_IF.values())
    document = _document(folder)
    fit = document["what_if"]["fit"]
    assert fit["seeds"] == [1, 2] and "seed" not in fit
    assert fit["seed_from"] == "--what-if-fit-seeds"
    assert fit["averaged"] == dial_choice.AVERAGED
    assert fit["levels_fitted"] == list(ALL)
    assert "seeds 1, 2 averaged" in document["label"]
    assert "not a run of the plan" in document["label"]
    lines = (folder / WHAT_IF["summary"]).read_text().splitlines()
    head = "\n".join(lines[:6])
    assert "Fit of the network: seeds 1, 2 averaged (--what-if-fit-seeds); levels fitted" in head
    assert "Averaged: the forecast is the average over the seeds' networks" in head
    assert "(dial_choice reading w3, w5, w6 and seed_average reading a1)" in lines[0].lower()
    assert "dial_choice reading w3, w5, w6, a1" not in head.lower()
    assert dial_summary.ENERGY_MEANS in head
    assert "means of the members' quantiles, not quantiles of the averaged distribution" in head
