"""The tuned settings file (`learn_tune_settings`): written only from a verified tuning results
file made with the committed grid, and read only when it fits that grid.

Made-up results: every candidate of the committed grid with a made-up score, in a made-up run
folder with a manifest (the commit check of `run_verify` is replaced, as no such commit exists).

1. The writer writes the chosen settings, the score, the grid's hash, the results file's hash,
   the run folder's name and the manifest's commit, and `load_tuned` reads the settings back.
2. The writer writes nothing when the results record another grid hash, when the run is not
   verified (changed file, failed or timed-out run), when the results were fitted on a part of
   the training units, when they do not score the grid's candidates, or when the candidate they
   name is not the one the choice rule gives.
3. `load_tuned` refuses an absent file, a file recording another grid hash, a file whose
   settings are not a candidate and a file with other keys.
"""

import hashlib
import json

import pytest

from fleetreserve import learn_tune_settings as lts
from fleetreserve import run_verify
from fleetreserve.gate_config import MatrixError
from fleetreserve.learn_tune_grid import GRID, Settings, load_grid
from fleetreserve.truth_build import TRAIN_UNITS

COMMIT = "b" * 40
BEST = Settings(31, 100, 0.05, 400)


def _results(**changes) -> dict:
    grid = load_grid()
    candidates = [
        {"settings": c.as_dict(), "score": 1.0 if c == BEST else 2.0 + 0.01 * k}
        for k, c in enumerate(grid.candidates)
    ]
    document = {
        "grid": {"file": grid.file, "sha256": grid.sha256},
        "fit_units": list(TRAIN_UNITS),
        "candidates": candidates,
        "chosen": {"settings": BEST.as_dict(), "score": 1.0},
    }
    document.update(changes)
    return document


def _run(tmp_path, results: dict, **manifest_changes):
    folder = tmp_path / "20260101T000000Z-made-up-tune"
    folder.mkdir()
    file = folder / "tune.json"
    file.write_text(json.dumps(results))
    manifest = {
        "commit": COMMIT,
        "command": ["tune"],
        "start": "s",
        "end": "e",
        "exit_code": 0,
        "outputs": {"tune.json": hashlib.sha256(file.read_bytes()).hexdigest()},
        **manifest_changes,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return file


@pytest.fixture(autouse=True)
def _any_commit(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)


def test_the_writer_writes_the_chosen_settings_and_where_they_came_from(tmp_path, capsys):
    file = _run(tmp_path, _results())
    out = tmp_path / "tuned.json"
    assert lts.main(["--results", str(file), "--out", str(out)]) == 0
    document = json.loads(out.read_text())
    grid = load_grid()
    assert document["settings"] == BEST.as_dict()
    assert document["score"] == 1.0
    assert document["grid"] == {"file": "tune-grid.json", "sha256": grid.sha256}
    assert document["results"] == {
        "file": "tune.json",
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "run_folder": "20260101T000000Z-made-up-tune",
        "commit": COMMIT,
    }
    assert sorted(document) == sorted(lts.TUNED_KEYS)
    assert lts.load_tuned(out) == BEST
    assert "Wrote tuned.json" in capsys.readouterr().out


def test_the_default_path_is_read_when_the_call_is_made(tmp_path, monkeypatch):
    file = _run(tmp_path, _results())
    out = tmp_path / "elsewhere.json"
    monkeypatch.setattr(lts, "TUNED", out)
    assert lts.main(["--results", str(file)]) == 0
    assert lts.load_tuned() == BEST


def _refused(tmp_path, capsys, file, code=1, grid=None) -> str:
    out = tmp_path / "tuned.json"
    arguments = ["--results", str(file), "--out", str(out)]
    if grid is not None:
        arguments += ["--grid", str(grid)]
    assert lts.main(arguments) == code
    assert not out.exists()
    return capsys.readouterr().out


def test_the_writer_refuses_results_made_with_another_grid(tmp_path, capsys):
    other = _results()
    other["grid"]["sha256"] = "0" * 64
    printed = _refused(tmp_path, capsys, _run(tmp_path, other))
    assert "another grid" in printed and "nothing written" in printed


def test_the_writer_refuses_when_the_committed_grid_has_changed_since_the_run(tmp_path, capsys):
    file = _run(tmp_path, _results())
    changed = tmp_path / "tune-grid.json"
    changed.write_text(GRID.read_text() + "\n")
    assert "another grid" in _refused(tmp_path, capsys, file, grid=changed)


@pytest.mark.parametrize(
    "manifest_changes, message",
    [
        ({"exit_code": 1}, "exit code 1"),
        ({"timed_out": True}, "time limit"),
        ({"interrupted": "SIGTERM"}, "interrupted"),
        ({"outputs": {"tune.json": "0" * 64}}, "changed since the run"),
        ({"outputs": {}}, "not listed"),
    ],
)
def test_the_writer_refuses_an_unverified_run(tmp_path, capsys, manifest_changes, message):
    file = _run(tmp_path, _results(), **manifest_changes)
    assert message in _refused(tmp_path, capsys, file)


def test_the_writer_cannot_run_without_a_results_file_or_a_manifest(tmp_path, capsys):
    assert "COULD NOT RUN" in _refused(tmp_path, capsys, tmp_path / "none.json", code=2)
    file = _run(tmp_path, _results())
    (file.parent / "manifest.json").unlink()
    assert "COULD NOT RUN" in _refused(tmp_path, capsys, file, code=2)


def test_the_writer_refuses_results_fitted_on_a_part_of_the_training_units(tmp_path, capsys):
    file = _run(tmp_path, _results(fit_units=list(TRAIN_UNITS[:3])))
    assert "not on every training unit" in _refused(tmp_path, capsys, file)


def test_the_writer_refuses_results_that_do_not_score_the_grids_candidates(tmp_path, capsys):
    short = _results()
    short["candidates"] = short["candidates"][:-1]
    assert "exactly the candidates" in _refused(tmp_path, capsys, _run(tmp_path, short))


def test_the_writer_applies_the_choice_rule_itself(tmp_path, capsys):
    other = _results(chosen={"settings": Settings(7, 20, 0.05, 200).as_dict(), "score": 0.0})
    assert "the choice rule gives" in _refused(tmp_path, capsys, _run(tmp_path, other))


def test_the_writer_refuses_a_file_that_is_not_a_results_file(tmp_path, capsys):
    assert "not a tuning results file" in _refused(tmp_path, capsys, _run(tmp_path, {"a": 1}))


def _tuned(tmp_path, **changes):
    grid = load_grid()
    document = {
        "about": "made up",
        "settings": BEST.as_dict(),
        "score": 1.0,
        "grid": {"file": grid.file, "sha256": grid.sha256},
        "results": {"file": "tune.json", "sha256": "1" * 64, "run_folder": "r", "commit": COMMIT},
        **changes,
    }
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps(document))
    return path


def test_load_tuned_reads_a_file_that_fits_the_committed_grid(tmp_path):
    assert lts.load_tuned(_tuned(tmp_path)) == BEST


def test_load_tuned_refuses_an_absent_file(tmp_path, monkeypatch):
    with pytest.raises(lts.SettingsError, match="absent"):
        lts.load_tuned(tmp_path / "tuned.json")
    monkeypatch.setattr(lts, "TUNED", tmp_path / "tuned.json")
    with pytest.raises(MatrixError, match="absent"):
        lts.load_tuned()


def test_load_tuned_refuses_another_grid_hash(tmp_path):
    path = _tuned(tmp_path, grid={"file": "tune-grid.json", "sha256": "0" * 64})
    with pytest.raises(lts.SettingsError, match="another grid"):
        lts.load_tuned(path)
    # the same file against a changed grid file
    changed = tmp_path / "tune-grid.json"
    changed.write_text(GRID.read_text() + "\n")
    with pytest.raises(lts.SettingsError, match="another grid"):
        lts.load_tuned(_tuned(tmp_path), changed)


def test_load_tuned_refuses_settings_that_are_not_a_candidate(tmp_path):
    path = _tuned(tmp_path, settings=Settings(8, 20, 0.05, 200).as_dict())
    with pytest.raises(lts.SettingsError, match="not a candidate"):
        lts.load_tuned(path)


@pytest.mark.parametrize("changes", [{"extra": 1}, {"settings": {"num_leaves": 15}}, {"grid": 3}])
def test_load_tuned_refuses_a_file_of_another_shape(tmp_path, changes):
    with pytest.raises(lts.SettingsError, match="cannot be used"):
        lts.load_tuned(_tuned(tmp_path, **changes))
    (tmp_path / "tuned.json").write_text("not json")
    with pytest.raises(lts.SettingsError, match="cannot be used"):
        lts.load_tuned(tmp_path / "tuned.json")
