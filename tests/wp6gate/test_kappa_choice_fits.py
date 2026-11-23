"""`kappa_choice` reading what-if runs that carry a fit (`dial_choice --what-if-fit-seed`,
`--what-if-fit-levels`: the key `what_if.fit` of their results; reading K10), on the made-up gate
units of `test_dial_choice_what_if` (the empirical model, which fits nothing, so the fit record
of a run is written into a copy of its results here, as `dial_choice` writes it for a network).

1. Three what-if runs at kappa 0.3 of three seeds are three fits, not three runs at one level:
   exit 0, one block per fit under `separate_fits`, in the order of seed, each block holding only
   its own run's figures (a figure of the second run set to 99 appears in its block and in no
   other), every block heading and the summary naming the seed and the levels fitted, and the
   summary saying that the fits are read apart.
2. Two what-if runs of one fit at one level could not be read (exit 2).
3. What-if runs of one fit at two levels are one block (no `separate_fits`), the block records the
   fit, every run line prints it and the over-all-levels reading says where its grid levels come
   from.
4. A what-if run without a fit option is a fit of its own: read apart from one with a fit, and
   named as one whose seed is not stated.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from gate_fake import COMMIT, GATE_UNITS, PLAN_LEVELS, matrix_document_six, write_build
from test_dial_choice_what_if import NAMES, SPLITS, units

from fleetreserve import dial_choice, gate_run, kappa_choice, run_verify

LEVELS = PLAN_LEVELS
ALL = (*LEVELS, 0.6, 0.7)
WHAT_IF = dial_choice.WHAT_IF_NAMES
CALIBRATABLE = GATE_UNITS[1:]


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    """The made-up runs carry a made-up commit, which the manifest check would not find."""
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


def _dial_choice(build, out: Path, *extra) -> int:
    run, tables, matrix = build
    arguments = ["--model", "empirical", "--matrix", str(matrix), "--tables", str(tables)]
    arguments += ["--truth-run", str(run), "--splits", str(SPLITS), *extra]
    out.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return dial_choice.main(arguments)


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """Run folders: the six of the grid and what-if runs at 0.3 and 0.6."""
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


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded(folder: Path) -> None:
    """A made-up manifest in `folder`, as `run_record` writes it."""
    outputs = {p.name: _sha(p) for p in sorted(folder.iterdir()) if p.name != "manifest.json"}
    manifest = {"commit": COMMIT, "command": ["python", "-m", "fleetreserve.dial_choice"]}
    manifest |= {"start": "s", "end": "e", "exit_code": 0, "outputs": outputs}
    (folder / "manifest.json").write_text(json.dumps(manifest))


def _copy(source: Path, folder: Path, fit: dict | None = None, offer: float | None = None) -> str:
    """A recorded copy of the what-if run `source` in `folder`, with the fit record `fit` written
    into its results and, with `offer`, every tier A mean offer set to it; returns the path of its
    results file."""
    shutil.copytree(source, folder)
    path = folder / WHAT_IF["results"]
    document = json.loads(path.read_text())
    if fit is not None:
        document["what_if"]["fit"] = fit
    if offer is not None:
        for entry in document["units"].values():
            for found in entry.get("configurations", {}).values():
                if "tiers" in found:
                    found["tiers"]["A"]["mean_offer_kwh_per_test_day"] = offer
    path.write_text(json.dumps(document, indent=1, sort_keys=True))
    _recorded(folder)
    return str(path)


def _fit(seed: int, levels=ALL) -> dict:
    return {
        "seed": seed,
        "seed_from": "--what-if-fit-seed",
        "levels_fitted": list(levels),
        "levels_from": "--what-if-fit-levels all",
        "energy_levels_per_quantity": len(levels),
    }


def _grid_files(made, folder: Path) -> list[str]:
    files = []
    for kappa in LEVELS:
        shutil.copytree(made[("grid", kappa)], folder / f"k{kappa:g}")
        _recorded(folder / f"k{kappa:g}")
        files.append(str(folder / f"k{kappa:g}" / dial_choice.RESULTS_NAME))
    return files


def _joint(files, out: Path, monkeypatch) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return kappa_choice.main(files)


def _joint_document(out: Path) -> dict:
    return json.loads((out / kappa_choice.RESULTS_NAME).read_text())


# --- 1. three seeds ------------------------------------------------------------------------------


def test_three_seeds_at_one_level_are_three_fits_read_apart(made, tmp_path, monkeypatch):
    files = _grid_files(made, tmp_path / "grid")
    source = made[("what-if", 0.3)]
    for seed, offer in ((3, None), (1, None), (2, 99.0)):
        files.append(_copy(source, tmp_path / f"seed{seed}", _fit(seed), offer))
    out = tmp_path / "joint"
    assert _joint(files, out, monkeypatch) == 0
    found = _joint_document(out)["what_if"]
    blocks = found["separate_fits"]
    assert [b["fit"]["seed"] for b in blocks] == [1, 2, 3]
    assert all(b["kappas"] == [0.3] for b in blocks)
    for block in blocks:
        assert block["runs"][0]["fit"] == block["fit"]
        cells = [
            cell["mean_offer_kwh_per_test_day"]
            for unit in CALIBRATABLE
            for cell in block["units"][unit]["pairs"]["A"]["0.3"].values()
            if cell["defined"]
        ]
        assert cells
        assert all(c == 99.0 for c in cells) == (block["fit"]["seed"] == 2)
        assert any(c == 99.0 for c in cells) == (block["fit"]["seed"] == 2)
    text = (out / kappa_choice.SUMMARY_NAME).read_text()
    for seed in (1, 2, 3):
        assert f"FIT: seed {seed}, levels fitted 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7" in text
    assert "3 sets of what-if runs of different fits follow, each read apart" in text
    assert text.index("FIT: seed 1") < text.index("FIT: seed 2") < text.index("FIT: seed 3")


# --- 2. one fit twice ----------------------------------------------------------------------------


def test_two_runs_of_one_fit_at_one_level_could_not_run(made, tmp_path, monkeypatch, capsys):
    files = _grid_files(made, tmp_path / "grid")
    source = made[("what-if", 0.3)]
    files.append(_copy(source, tmp_path / "a", _fit(1)))
    files.append(_copy(source, tmp_path / "b", _fit(1)))
    assert _joint(files, tmp_path / "joint", monkeypatch) == 2
    assert "two what-if runs of one fit measure kappa 0.3" in capsys.readouterr().out


# --- 3. one fit at two levels --------------------------------------------------------------------


def test_one_fit_at_two_levels_is_one_block_that_records_the_fit(made, tmp_path, monkeypatch):
    files = _grid_files(made, tmp_path / "grid")
    files.append(_copy(made[("what-if", 0.3)], tmp_path / "a", _fit(5)))
    files.append(_copy(made[("what-if", 0.6)], tmp_path / "b", _fit(5)))
    out = tmp_path / "joint"
    assert _joint(files, out, monkeypatch) == 0
    found = _joint_document(out)["what_if"]
    assert "separate_fits" not in found
    assert found["fit"]["seed"] == 5 and found["kappas"] == [0.3, 0.6]
    assert all(r["fit"]["seed"] == 5 for r in found["runs"])
    mixes = found["readings_over_all_levels"]["mixes"]
    assert "plan's six runs" in mixes and "seed 5" in mixes
    text = (out / kappa_choice.SUMMARY_NAME).read_text()
    assert (
        "Fit of the network: seed 5, levels fitted 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7." in text
    )
    assert text.count("[seed 5, levels fitted 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]") == 2
    assert "each read apart" not in text


# --- 4. a run without a fit option ---------------------------------------------------------------


def test_a_run_without_a_fit_option_is_a_fit_of_its_own(made, tmp_path, monkeypatch):
    files = _grid_files(made, tmp_path / "grid")
    files.append(_copy(made[("what-if", 0.3)], tmp_path / "plain"))
    files.append(_copy(made[("what-if", 0.3)], tmp_path / "seeded", _fit(1)))
    out = tmp_path / "joint"
    assert _joint(files, out, monkeypatch) == 0
    blocks = _joint_document(out)["what_if"]["separate_fits"]
    assert len(blocks) == 2
    assert "fit" not in blocks[0] and blocks[1]["fit"]["seed"] == 1
    text = (out / kappa_choice.SUMMARY_NAME).read_text()
    assert "FIT: none given, the seed is not stated." in text
    assert "FIT: seed 1, levels fitted" in text
