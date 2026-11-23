"""`kappa_choice` reading what-if runs of an average over seeds (`dial_choice --what-if-fit-seeds`:
the key `seeds` of `what_if.fit`, reading K10), on the made-up runs of `test_kappa_choice_fits`
(the empirical model, so the fit record is written into a copy of a run's results, as
`dial_choice` writes it for an averaged network).

1. At kappa 0.3, the average over seeds 1, 2, 3, the average over 0, 1, 2, 3 and seed 1 alone are
   three fits read apart: exit 0, three blocks under `separate_fits` (the single seed first, then
   the seed lists in order), each holding only its own run's figures (a figure of the 1, 2, 3
   average set to 99 appears in its block and in no other), the summary naming each list.
2. Two runs of one seed list at one level could not be read (exit 2); two lists at one level can.
3. One seed list at two levels is one block that records the list.
"""

import json

import pytest
from gate_fake import COMMIT, matrix_document_six, write_build
from test_dial_choice_what_if import NAMES, units
from test_kappa_choice_fits import (
    ALL,
    CALIBRATABLE,
    LEVELS,
    _copy,
    _dial_choice,
    _grid_files,
    _joint,
    _joint_document,
)

from fleetreserve import dial_choice, gate_run, kappa_choice, run_verify


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    """The made-up runs carry a made-up commit, which the manifest check would not find."""
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """Run folders: the six of the grid and what-if runs at 0.3 and 0.6, as in
    `test_kappa_choice_fits`."""
    folder = tmp_path_factory.mktemp("made")
    run, tables = write_build(folder, units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document_six(NAMES)))
    build = (run, tables, matrix)
    found = {}
    for kappa in LEVELS:
        found[("grid", kappa)] = folder / f"k{kappa:g}"
        assert _dial_choice(build, found[("grid", kappa)], "--kappa", str(kappa)) == 0
    for kappa in (0.3, 0.6):
        found[("what-if", kappa)] = folder / f"w{kappa:g}"
        assert _dial_choice(build, found[("what-if", kappa)], "--what-if-kappa", str(kappa)) == 0
    return found


def _average(seeds, levels=ALL) -> dict:
    return {
        "seeds": list(seeds),
        "seed_from": "--what-if-fit-seeds",
        "averaged": dial_choice.AVERAGED,
        "levels_fitted": list(levels),
        "levels_from": "--what-if-fit-levels all",
        "energy_levels_per_quantity": len(levels),
    }


def _single(seed: int) -> dict:
    return {
        "seed": seed,
        "seed_from": "--what-if-fit-seed",
        "levels_fitted": list(ALL),
        "levels_from": "--what-if-fit-levels all",
        "energy_levels_per_quantity": len(ALL),
    }


LEVELS_TEXT = "levels fitted 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7"


def _cells(block) -> list:
    return [
        cell["mean_offer_kwh_per_test_day"]
        for unit in CALIBRATABLE
        for cell in block["units"][unit]["pairs"]["A"]["0.3"].values()
        if cell["defined"]
    ]


def test_an_average_is_read_apart_from_a_single_seed_and_from_another_list(
    made, tmp_path, monkeypatch
):
    files = _grid_files(made, tmp_path / "grid")
    source = made[("what-if", 0.3)]
    files.append(_copy(source, tmp_path / "avg0123", _average((0, 1, 2, 3))))
    files.append(_copy(source, tmp_path / "avg123", _average((1, 2, 3)), 99.0))
    files.append(_copy(source, tmp_path / "seed1", _single(1)))
    out = tmp_path / "joint"
    assert _joint(files, out, monkeypatch) == 0
    blocks = _joint_document(out)["what_if"]["separate_fits"]
    assert len(blocks) == 3
    assert blocks[0]["fit"]["seed"] == 1 and "seeds" not in blocks[0]["fit"]
    assert [b["fit"].get("seeds") for b in blocks[1:]] == [[0, 1, 2, 3], [1, 2, 3]]
    for block in blocks:
        assert block["runs"][0]["fit"] == block["fit"]
        cells = _cells(block)
        assert cells
        marked = block["fit"].get("seeds") == [1, 2, 3]
        assert all(c == 99.0 for c in cells) == marked
        assert any(c == 99.0 for c in cells) == marked
    text = (out / kappa_choice.SUMMARY_NAME).read_text()
    assert f"FIT: seeds 1, 2, 3 averaged, {LEVELS_TEXT}" in text
    assert f"FIT: seeds 0, 1, 2, 3 averaged, {LEVELS_TEXT}" in text
    assert f"FIT: seed 1, {LEVELS_TEXT}" in text
    assert "3 sets of what-if runs of different fits follow, each read apart" in text


def test_the_definition_of_fits_names_the_seed_list_option():
    assert "--what-if-fit-seeds" in kappa_choice.FIT_DEFINITIONS["fits"]


def test_two_runs_of_one_list_at_one_level_could_not_run(made, tmp_path, monkeypatch, capsys):
    files = _grid_files(made, tmp_path / "grid")
    source = made[("what-if", 0.3)]
    files.append(_copy(source, tmp_path / "a", _average((1, 2, 3))))
    files.append(_copy(source, tmp_path / "b", _average((1, 2, 3))))
    assert _joint(files, tmp_path / "joint", monkeypatch) == 2
    assert "two what-if runs of one fit measure kappa 0.3" in capsys.readouterr().out


def test_one_list_at_two_levels_is_one_block_that_records_it(made, tmp_path, monkeypatch):
    files = _grid_files(made, tmp_path / "grid")
    files.append(_copy(made[("what-if", 0.3)], tmp_path / "a", _average((1, 2, 3))))
    files.append(_copy(made[("what-if", 0.6)], tmp_path / "b", _average((1, 2, 3))))
    out = tmp_path / "joint"
    assert _joint(files, out, monkeypatch) == 0
    found = _joint_document(out)["what_if"]
    assert "separate_fits" not in found
    assert found["fit"]["seeds"] == [1, 2, 3] and found["kappas"] == [0.3, 0.6]
    assert "seeds 1, 2, 3 averaged" in found["readings_over_all_levels"]["mixes"]
    text = (out / kappa_choice.SUMMARY_NAME).read_text()
    assert f"Fit of the network: seeds 1, 2, 3 averaged, {LEVELS_TEXT}." in text
    assert text.count(f"[seeds 1, 2, 3 averaged, {LEVELS_TEXT}]") == 2
