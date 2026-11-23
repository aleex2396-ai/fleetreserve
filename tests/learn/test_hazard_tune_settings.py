"""The grid, the choice rule, the tuned settings file and the tuned model of the neural hazard
model (`hazard_tune_grid`, `hazard_tune_settings`, `models.hazard_tuned`), on made-up data.

1. The committed grid file has 12 fitted candidates and 60 candidates, three seeds, five epoch
   values, holds the present settings, and the present settings are the defaults of
   `HazardSettings`. A malformed grid is refused.
2. The choice rule: the lowest score; each tie rule in its order (smaller width, fewer epochs,
   larger divergence weight, larger learning rate); a tie is a difference within the tolerance.
3. The writer writes the chosen settings and where they came from from a made-up verified run, and
   writes nothing for an unverified run, another grid, a part of the training units, another set
   of candidates or a chosen candidate that is not the rule's. `load_tuned` refuses a missing
   file, a changed grid, settings that are not a candidate and a file with other keys.
4. `models.hazard_tuned` is not built without a settings file (and `learn_run` then ends with exit
   2); with a valid file it fits and forecasts on made-up units with those settings, and its
   forecast differs from `models.hazard`'s when the settings differ.
"""

import hashlib
import json
from types import SimpleNamespace

import hazard_fake as hf
import pytest

from fleetreserve import hazard_tune_grid as grids
from fleetreserve import hazard_tune_settings as hts
from fleetreserve import learn_run, run_verify
from fleetreserve.gate_config import MatrixError
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.hazard_tune_grid import GRID, PRESENT, GridError, Setting, choose, load_grid, rank
from fleetreserve.models import hazard, hazard_tuned
from fleetreserve.truth_build import TRAIN_UNITS

COMMIT = "c" * 40
BEST = Setting(64, 0.001, 0.3, 50)
FILE = {
    "about": "made up",
    "seeds": [0, 1],
    "epochs": [4, 30],
    "width": [16, 32],
    "learning_rate": [0.003],
    "kl_weight": [1.0, 0.3],
}


def _grid(tmp_path, **changes):
    document = {**FILE, **changes}
    for key in [k for k, v in document.items() if v is ...]:
        del document[key]
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(document))
    return path


# --- 1. the grid ---------------------------------------------------------------------------------


def test_the_committed_grid_has_sixty_candidates_of_twelve_fits_and_holds_the_present_settings():
    grid = load_grid()
    assert grid.file == "hazard-grid.json" == GRID.name
    assert grid.sha256 == hashlib.sha256(GRID.read_bytes()).hexdigest()
    assert grid.seeds == (0, 1, 2) and grid.epochs == (10, 20, 30, 50, 80)
    assert len(grid.fits) == 12 and len(grid.candidates) == 60 == len(set(grid.candidates))
    assert {c.width for c in grid.candidates} == {32, 64}
    assert {c.learning_rate for c in grid.candidates} == {0.003, 0.001}
    assert {c.kl_weight for c in grid.candidates} == {1.0, 0.3, 0.1}
    # the order of the file: width, learning rate, divergence weight, then the epoch values
    assert grid.candidates[0] == Setting(32, 0.003, 1.0, 10)
    assert grid.candidates[1] == Setting(32, 0.003, 1.0, 20)
    assert grid.candidates[5] == Setting(32, 0.003, 0.3, 10)
    assert grid.fits[:3] == ((32, 0.003, 1.0), (32, 0.003, 0.3), (32, 0.003, 0.1))
    assert PRESENT in grid.candidates
    defaults = HazardSettings()
    same = Setting(defaults.width, defaults.learning_rate, defaults.kl_weight, defaults.epochs)
    assert same == PRESENT
    assert PRESENT.hazard() == defaults
    assert "default" in json.loads(GRID.read_text())["about"]


def test_a_setting_becomes_hazard_settings_with_a_seed_and_changes():
    found = Setting(16, 0.001, 0.3, 7).hazard(2, set_encoder=False)
    assert (found.width, found.learning_rate, found.kl_weight, found.epochs) == (16, 0.001, 0.3, 7)
    assert found.seed == 2 and found.set_encoder is False and found.day_factor is True


def test_a_grid_file_is_read_with_its_fits_and_sorted_epochs(tmp_path):
    grid = load_grid(_grid(tmp_path, epochs=[30, 4]))
    assert grid.epochs == (4, 30) and grid.seeds == (0, 1)
    assert grid.fits == ((16, 0.003, 1.0), (16, 0.003, 0.3), (32, 0.003, 1.0), (32, 0.003, 0.3))
    assert len(grid.candidates) == 8 and PRESENT in grid.candidates


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"extra": 1}, "exactly the keys"),
        ({"seeds": ...}, "exactly the keys"),
        ({"seeds": []}, "non-empty list"),
        ({"seeds": [0, 0]}, "twice"),
        ({"seeds": [-1]}, "seed"),
        ({"seeds": [1.5]}, "seed"),
        ({"epochs": [True, 30]}, "epoch"),
        ({"epochs": [0, 30]}, "epoch"),
        ({"epochs": ["30"]}, "epoch"),
        ({"width": [15, 32]}, "cannot be used"),
        ({"width": [0, 32]}, "width"),
        ({"learning_rate": [0.0, 0.003]}, "range"),
        ({"learning_rate": [1.5, 0.003]}, "range"),
        ({"learning_rate": ["x"]}, "number"),
        ({"kl_weight": [-0.1, 1.0]}, "range"),
        ({"kl_weight": [1.0, 1.0]}, "twice"),
        ({"kl_weight": [1, 1.0]}, "twice"),  # the same value written two ways
        ({"kl_weight": [1.0, 0.3, 1]}, "twice"),
        ({"width": [32, 32.0]}, "twice"),
        ({"epochs": [30, 4, 30.0]}, "twice"),
        ({"seeds": [1, 1.0]}, "twice"),
        ({"width": [16]}, "present settings"),
        ({"epochs": [4, 10]}, "present settings"),
        ({"learning_rate": [0.001]}, "present settings"),
        ({"kl_weight": [0.3]}, "present settings"),
    ],
)
def test_a_malformed_grid_is_refused(tmp_path, changes, message):
    with pytest.raises(GridError, match=message):
        load_grid(_grid(tmp_path, **changes))


def test_an_unreadable_grid_is_refused(tmp_path):
    with pytest.raises(GridError, match="cannot be read"):
        load_grid(tmp_path / "none.json")
    path = tmp_path / "bad.json"
    path.write_text("[1, 2")
    with pytest.raises(GridError, match="cannot be read"):
        load_grid(path)
    path.write_text("[1, 2]")
    with pytest.raises(GridError, match="exactly the keys"):
        load_grid(path)


# --- 2. the choice rule --------------------------------------------------------------------------


def test_the_lowest_score_is_chosen():
    scored = [(Setting(16, 0.003, 1.0, 4), 3.0), (Setting(64, 0.001, 0.1, 80), 1.0)]
    assert choose(scored) == Setting(64, 0.001, 0.1, 80)
    assert choose(list(reversed(scored))) == Setting(64, 0.001, 0.1, 80)


def test_a_tie_goes_to_the_smaller_width_first():
    tied = [(Setting(64, 0.003, 1.0, 10), 2.0), (Setting(32, 0.001, 0.1, 80), 2.0)]
    assert choose(tied) == Setting(32, 0.001, 0.1, 80)
    assert choose(list(reversed(tied))) == Setting(32, 0.001, 0.1, 80)


def test_a_tie_with_the_same_width_goes_to_the_fewer_epochs():
    tied = [(Setting(32, 0.003, 1.0, 80), 2.0), (Setting(32, 0.001, 0.1, 20), 2.0)]
    assert choose(tied) == Setting(32, 0.001, 0.1, 20)


def test_a_tie_with_the_same_width_and_epochs_goes_to_the_larger_divergence_weight():
    tied = [(Setting(32, 0.003, 0.1, 30), 2.0), (Setting(32, 0.001, 1.0, 30), 2.0)]
    assert choose(tied) == Setting(32, 0.001, 1.0, 30)
    assert choose(list(reversed(tied))) == Setting(32, 0.001, 1.0, 30)


def test_a_tie_in_all_else_goes_to_the_larger_learning_rate():
    tied = [(Setting(32, 0.001, 0.3, 30), 2.0), (Setting(32, 0.003, 0.3, 30), 2.0)]
    assert choose(tied) == Setting(32, 0.003, 0.3, 30)
    assert choose(list(reversed(tied))) == Setting(32, 0.003, 0.3, 30)


def test_a_tie_is_a_difference_within_the_tolerance_and_not_more():
    low, other = Setting(64, 0.003, 1.0, 10), Setting(32, 0.003, 1.0, 10)
    assert choose([(low, 2.0), (other, 2.0 + grids.TIE / 2)]) == other
    assert choose([(low, 2.0), (other, 2.0 + 1e-6)]) == low


def test_the_rule_refuses_nothing_to_choose_from_and_a_score_that_is_not_a_number():
    with pytest.raises(ValueError, match="no candidate"):
        choose([])
    with pytest.raises(ValueError, match="not a finite"):
        choose([(PRESENT, float("nan"))])


def test_the_rank_is_one_more_than_the_candidates_with_a_lower_score():
    scored = [(Setting(32, 0.003, 1.0, 10), 3.0), (PRESENT, 1.0), (BEST, 1.0)]
    assert rank(scored, PRESENT) == 1 == rank(scored, BEST)
    assert rank(scored, Setting(32, 0.003, 1.0, 10)) == 3


# --- 3. the settings file ------------------------------------------------------------------------


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
    folder = tmp_path / "20260101T000000Z-made-up-hazard-tune"
    folder.mkdir()
    file = folder / "hazard-tune.json"
    file.write_text(json.dumps(results))
    manifest = {
        "commit": COMMIT,
        "command": ["tune"],
        "start": "s",
        "end": "e",
        "exit_code": 0,
        "outputs": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()},
        **manifest_changes,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return file


@pytest.fixture(autouse=True)
def _any_commit(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)


def test_the_writer_writes_the_chosen_settings_and_where_they_came_from(tmp_path, capsys):
    file = _run(tmp_path, _results())
    out = tmp_path / "hazard-tuned.json"
    assert hts.main([str(file), "--out", str(out)]) == 0
    document = json.loads(out.read_text())
    assert document["settings"] == BEST.as_dict() and document["score"] == 1.0
    assert document["grid"] == {"file": "hazard-grid.json", "sha256": load_grid().sha256}
    assert document["results"] == {
        "file": "hazard-tune.json",
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "run_folder": "20260101T000000Z-made-up-hazard-tune",
        "commit": COMMIT,
    }
    assert sorted(document) == sorted(hts.TUNED_KEYS) and "Not edited by hand" in document["about"]
    assert hts.load_tuned(out) == BEST
    assert "Wrote hazard-tuned.json" in capsys.readouterr().out


def test_the_committed_tuned_settings_file_is_one_the_loader_accepts_and_names_its_run():
    assert hts.TUNED.is_file() and hts.TUNED.name == "hazard-tuned.json"
    grid = load_grid()
    setting = hts.load_tuned()
    assert isinstance(setting, Setting) and setting in grid.candidates
    document = json.loads(hts.TUNED.read_text())
    assert sorted(document) == sorted(hts.TUNED_KEYS)
    assert document["grid"] == {
        "file": GRID.name,
        "sha256": hashlib.sha256(GRID.read_bytes()).hexdigest(),
    }
    results = document["results"]
    assert sorted(results) == ["commit", "file", "run_folder", "sha256"]
    assert len(results["commit"]) == 40 and set(results["commit"]) <= set("0123456789abcdef")
    assert results["file"] == "hazard-tune.json"
    assert results["run_folder"].endswith("-wp8-hazard-tune")
    assert isinstance(document["score"], float) and 0 < document["score"] < float("inf")


def _refused(tmp_path, capsys, file, code=1, grid=None) -> str:
    out = tmp_path / "hazard-tuned.json"
    arguments = [str(file), "--out", str(out)]
    if grid is not None:
        arguments += ["--grid", str(grid)]
    assert hts.main(arguments) == code
    assert not out.exists()
    return capsys.readouterr().out


def test_the_writer_refuses_results_made_with_another_grid(tmp_path, capsys):
    other = _results()
    other["grid"]["sha256"] = "0" * 64
    printed = _refused(tmp_path, capsys, _run(tmp_path, other))
    assert "another grid" in printed and "nothing written" in printed


def test_the_writer_refuses_when_the_committed_grid_has_changed_since_the_run(tmp_path, capsys):
    file = _run(tmp_path, _results())
    changed = tmp_path / "hazard-grid.json"
    changed.write_text(GRID.read_text() + "\n")
    assert "another grid" in _refused(tmp_path, capsys, file, grid=changed)


@pytest.mark.parametrize(
    "manifest_changes, message",
    [
        ({"exit_code": 1}, "exit code 1"),
        ({"timed_out": True}, "time limit"),
        ({"interrupted": "SIGTERM"}, "interrupted"),
        ({"outputs": {"hazard-tune.json": "0" * 64}}, "changed since the run"),
        ({"outputs": {}}, "not listed"),
    ],
)
def test_the_writer_refuses_an_unverified_run(tmp_path, capsys, manifest_changes, message):
    file = _run(tmp_path, _results(), **manifest_changes)
    assert message in _refused(tmp_path, capsys, file)


def test_the_writer_refuses_a_part_of_the_training_units_other_candidates_and_a_wrong_choice(
    tmp_path, capsys
):
    part = _run(tmp_path, _results(fit_units=list(TRAIN_UNITS)[:3]))
    assert "not on every training unit" in _refused(tmp_path, capsys, part)
    fewer = _results()
    fewer["candidates"] = fewer["candidates"][:-1]
    folder = tmp_path / "two"
    folder.mkdir()
    assert "exactly the candidates" in _refused(folder, capsys, _run(folder, fewer))
    wrong = _results(chosen={"settings": PRESENT.as_dict(), "score": 2.0})
    folder = tmp_path / "three"
    folder.mkdir()
    printed = _refused(folder, capsys, _run(folder, wrong))
    assert "the choice rule gives" in printed


def test_the_writer_cannot_run_without_a_results_file_a_manifest_or_a_grid(tmp_path, capsys):
    assert "COULD NOT RUN" in _refused(tmp_path, capsys, tmp_path / "none.json", code=2)
    file = _run(tmp_path, _results())
    (file.parent / "manifest.json").unlink()
    assert "COULD NOT RUN" in _refused(tmp_path, capsys, file, code=2)
    again = tmp_path / "again"
    again.mkdir()
    file = _run(again, _results())
    printed = _refused(tmp_path, capsys, file, code=2, grid=tmp_path / "no-grid.json")
    assert "COULD NOT RUN" in printed and "grid file" in printed


def _settings_file(folder, name="hazard-tuned.json", **changes):
    grid_path = changes.pop("grid_path", None)
    grid = load_grid(grid_path)
    document = {
        "about": "made up",
        "settings": BEST.as_dict(),
        "score": 1.0,
        "grid": {"file": grid.file, "sha256": grid.sha256},
        "results": {"file": "x", "sha256": "1" * 64, "run_folder": "r", "commit": COMMIT},
        **changes,
    }
    path = folder / name
    path.write_text(json.dumps(document))
    return path


def test_load_tuned_refuses_an_absent_file_another_grid_other_settings_and_other_keys(tmp_path):
    with pytest.raises(hts.SettingsError, match="absent"):
        hts.load_tuned(tmp_path / "none.json")
    path = _settings_file(tmp_path)
    assert hts.load_tuned(path) == BEST
    _settings_file(tmp_path, grid={"file": "hazard-grid.json", "sha256": "0" * 64})
    with pytest.raises(hts.SettingsError, match="another grid"):
        hts.load_tuned(path)
    _settings_file(tmp_path, settings=Setting(48, 0.001, 0.3, 50).as_dict())
    with pytest.raises(hts.SettingsError, match="not a candidate"):
        hts.load_tuned(path)
    _settings_file(tmp_path, settings=Setting(64, 0.001, 0.3, 51).as_dict())
    with pytest.raises(hts.SettingsError, match="not a candidate"):
        hts.load_tuned(path)
    _settings_file(tmp_path, settings={"width": 64})
    with pytest.raises(hts.SettingsError, match="cannot be used"):
        hts.load_tuned(path)
    document = json.loads(_settings_file(tmp_path).read_text())
    document["extra"] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(hts.SettingsError, match="exactly the keys"):
        hts.load_tuned(path)
    path.write_text("{")
    with pytest.raises(hts.SettingsError, match="cannot be used"):
        hts.load_tuned(path)


def test_a_settings_error_is_a_matrix_error_so_the_runs_end_with_could_not_run():
    assert issubclass(hts.SettingsError, MatrixError)


def test_the_default_path_is_read_when_the_call_is_made(tmp_path, monkeypatch):
    monkeypatch.setattr(hts, "TUNED", tmp_path / "elsewhere.json")
    with pytest.raises(hts.SettingsError, match="absent"):
        hts.load_tuned()
    _settings_file(tmp_path, "elsewhere.json")
    assert hts.load_tuned() == BEST


# --- 4. the tuned model --------------------------------------------------------------------------

MATRIX = SimpleNamespace(rho=0.0, configurations=[SimpleNamespace(kappa=k) for k in hf.KAPPAS])
TUNED = Setting(16, 0.003, 1.0, 4)


@pytest.fixture
def small_grid(tmp_path, monkeypatch):
    """The committed grid replaced by a made-up one that holds `TUNED`, and a settings file."""
    monkeypatch.setattr(grids, "GRID", _grid(tmp_path))
    monkeypatch.setattr(hts, "TUNED", tmp_path / "hazard-tuned.json")
    return tmp_path


def test_a_tuned_model_is_not_built_without_settings(small_grid, capsys):
    with pytest.raises(MatrixError, match="absent"):
        hazard_tuned.make_model(MATRIX)
    _settings_file(
        small_grid, grid={"file": "grid.json", "sha256": "0" * 64}, settings=TUNED.as_dict()
    )
    with pytest.raises(MatrixError, match="another grid"):
        hazard_tuned.make_model(MATRIX)
    _settings_file(small_grid, settings=Setting(8, 0.003, 1.0, 4).as_dict())
    with pytest.raises(MatrixError, match="not a candidate"):
        hazard_tuned.make_model(MATRIX)
    with pytest.raises(ValueError, match="settings of the tuned settings file"):
        hazard_tuned.HazardTunedModel(rho=0.0, kappas=hf.KAPPAS, tuned=None)
    with pytest.raises(ValueError, match="settings of the tuned settings file"):
        hazard_tuned.HazardTunedModel(rho=0.0, kappas=hf.KAPPAS, tuned=HazardSettings())


def test_learn_run_cannot_run_the_tuned_model_without_its_settings(small_grid, capsys):
    assert learn_run.main(["--model", "hazard_tuned"]) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN:") and "absent" in printed
    assert learn_run.load_model("hazard", MATRIX).name == "hazard"


def test_the_tuned_model_fits_and_forecasts_with_the_settings_of_the_file(small_grid):
    _settings_file(small_grid, settings=TUNED.as_dict())
    model = hazard_tuned.make_model(MATRIX)
    assert model.name == "hazard_tuned" and model.fit_units == TRAIN_UNITS
    assert model.tuned == TUNED
    assert model.settings == HazardSettings(width=16, epochs=4)
    train, fresh, _ = hf.known_data(60, 10)
    model.fit(train)
    assert len(model.history) == 4 and model.net is not None
    table = model.predict(fresh, hf.KAPPAS)
    assert len(table) == len(fresh) * len(hf.KAPPAS) and table["d_hat"].notna().all()


def test_the_tuned_forecast_differs_from_the_hazard_models_when_the_settings_differ(small_grid):
    _settings_file(small_grid, settings=TUNED.as_dict())
    tuned = hazard_tuned.make_model(MATRIX)
    train, fresh, _ = hf.known_data(60, 10)
    tuned.fit(train)
    same = hazard.HazardModel(rho=0.0, kappas=hf.KAPPAS, settings=TUNED.hazard())
    same.fit(train)
    plain = hazard.make_model(MATRIX)
    plain.fit(train)
    assert tuned.predict(fresh, hf.KAPPAS).equals(same.predict(fresh, hf.KAPPAS))
    assert not tuned.predict(fresh, hf.KAPPAS)["d_hat"].equals(
        plain.predict(fresh, hf.KAPPAS)["d_hat"]
    )
    assert plain.settings == HazardSettings() and plain.name == "hazard"
