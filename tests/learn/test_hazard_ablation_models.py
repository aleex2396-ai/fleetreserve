"""The four ablation models of the tuned neural hazard model (`models.hazard_tuned_no_set`,
`hazard_tuned_no_site`, `hazard_tuned_no_day`, `hazard_tuned_calendar`), on made-up data, with a
made-up tuned settings file (`hazard_fake.use_small_tuned`).

1. Each model's settings are `hazard_tuned`'s with exactly the fields of its part of
   `hazard_parts.PARTS` changed; the tuning command reads the same parts.
2. Each part acts, and the dependence it removes exists in `hazard_tuned` on the same data:
   - no set encoder: a car's energy columns do not change when the inputs of the other cars of its
     site-day change (in `hazard_tuned` they do);
   - no site embedding: a set's curves do not change with the site code given to the network (in
     `hazard_tuned` they do), and the network has no site vectors;
   - no day factor: the network has no prior, posterior or label network (`hazard_tuned`'s has);
   - calendar-only inputs: changing inputs other than the calendar ones changes no forecast cell
     (in `hazard_tuned` it does).
3. Each is refused, as `hazard_tuned` is, without a usable settings file, and `learn_run` then
   ends with exit 2.
4. `learn_run --model <name>` runs each end to end on made-up units (exit 0).
"""

import importlib
import json
from dataclasses import replace
from types import SimpleNamespace

import hazard_fake as hf
import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import pytest
import torch
from gate_fake import COMMIT, GATE_UNITS, write_build
from learn_fake import small_matrix
from test_hazard_learn_run import _units

from fleetreserve import gate_run, hazard_parts, hazard_tune, learn_run, run_verify
from fleetreserve import hazard_tune_settings as hts
from fleetreserve.forecast_model import KEY_COLUMNS, read_table
from fleetreserve.gate_config import MatrixError
from fleetreserve.hazard_tune_grid import Setting
from fleetreserve.models import hazard_tuned
from fleetreserve.truth_build import TRAIN_UNITS

MODELS = {
    "hazard_tuned_no_set": "no_set_encoder",
    "hazard_tuned_no_site": "no_site_embedding",
    "hazard_tuned_no_day": "no_day_factor",
    "hazard_tuned_calendar": "calendar_only_inputs",
}
MATRIX = SimpleNamespace(rho=0.0, configurations=[SimpleNamespace(kappa=k) for k in hf.KAPPAS])
ENERGY = ("r_hat", "e_cap", "e_need")


def module(name: str):
    return importlib.import_module(f"fleetreserve.models.{name}")


def bytes_of(table: pd.DataFrame) -> bytes:
    return b"".join(table[c].to_numpy().tobytes() for c in table if table[c].dtype.kind in "fi")


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Every ablation model and `hazard_tuned`, fitted on the same made-up cars; the later days'
    feature frame. The tests only read them."""
    folder = tmp_path_factory.mktemp("tuned")
    train, fresh, _ = hf.known_data(60, 10)
    models = {}
    with pytest.MonkeyPatch.context() as patch:
        hf.use_small_tuned(patch, folder)
        for name in ("hazard_tuned", *MODELS):
            model = module(name).make_model(MATRIX)
            model.fit(train)
            models[name] = model
    return models, fresh


# --- 1. settings ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_the_settings_are_hazard_tuneds_with_the_one_part_changed(fitted, name):
    models, _ = fitted
    model, tuned = models[name], models["hazard_tuned"]
    changes = hazard_parts.PARTS[MODELS[name]]
    assert model.name == name and model.fit_units == TRAIN_UNITS
    assert model.tuned == hf.SMALL_TUNED == tuned.tuned
    assert model.settings == replace(tuned.settings, **changes)
    assert model.settings != tuned.settings and model.settings.seed == 0
    differing = {
        field
        for field in model.settings.__dataclass_fields__
        if getattr(model.settings, field) != getattr(tuned.settings, field)
    }
    assert differing == set(changes)
    assert model.levels == tuned.levels and model.rho == tuned.rho


def test_the_tuning_command_reads_the_same_parts():
    assert hazard_tune.PARTS is hazard_parts.PARTS
    assert hazard_tune.CALENDAR_INPUTS is hazard_parts.CALENDAR_INPUTS
    assert set(MODELS.values()) == set(hazard_parts.PARTS) - {"no_set_encoder_no_day_factor"}


# --- 2. each part acts ---------------------------------------------------------------------------


def _first_set(fresh: pd.DataFrame) -> pd.Index:
    """The rows of the first site-day of `fresh` that has at least three cars."""
    for _, rows in fresh.groupby(["unit", "site", "date"], sort=True).groups.items():
        if len(rows) >= 3:
            return rows
    raise AssertionError("no site-day with three cars")


def _rows_of(table: pd.DataFrame, frame: pd.DataFrame, row) -> pd.DataFrame:
    key = frame.loc[row, list(KEY_COLUMNS)]
    found = table[(table[list(KEY_COLUMNS)] == key.to_numpy()).all(axis=1)]
    assert len(found) == len(hf.KAPPAS)
    return found.reset_index(drop=True)


def test_without_the_set_encoder_a_cars_energy_columns_ignore_the_other_cars(fitted):
    models, fresh = fitted
    rows = _first_set(fresh)
    target, others = rows[0], rows[1:]
    moved = fresh.copy()
    moved.loc[others, "arrival_hour"] = (moved.loc[others, "arrival_hour"] + 5) % 24
    moved.loc[others, "hours_since_plug_in"] = moved.loc[others, "hours_since_plug_in"] * 2 + 1
    moved.loc[others, "history_mean_stay_h"] = (
        moved.loc[others, "history_mean_stay_h"].fillna(1.0) + 2.0
    )
    same = {}
    for name in ("hazard_tuned_no_set", "hazard_tuned"):
        model = models[name]
        before = _rows_of(model.predict(fresh, hf.KAPPAS), fresh, target)
        after = _rows_of(model.predict(moved, hf.KAPPAS), moved, target)
        same[name] = bytes_of(before[list(ENERGY)]) == bytes_of(after[list(ENERGY)])
    assert same == {"hazard_tuned_no_set": True, "hazard_tuned": False}
    assert not hasattr(models["hazard_tuned_no_set"].net, "blocks")
    assert hasattr(models["hazard_tuned"].net, "blocks")


def test_without_the_site_embedding_the_curves_ignore_the_site_code(fitted):
    models, fresh = fitted
    torch.set_num_threads(1)
    same = {}
    for name in ("hazard_tuned_no_site", "hazard_tuned"):
        model = models[name]
        x, sets = model.inputs(fresh)
        rows = sets.members(0)
        found = [model.curves(x[rows], code) for code in range(len(model.fit_sites) + 1)]
        same[name] = all(
            np.array_equal(curve, found[0][0]) and np.array_equal(energy, found[0][1])
            for curve, energy in found
        )
    assert same == {"hazard_tuned_no_site": True, "hazard_tuned": False}
    assert not hasattr(models["hazard_tuned_no_site"].net, "site")
    assert hasattr(models["hazard_tuned"].net, "site")


def test_without_the_day_factor_the_network_has_no_day_factor_path(fitted):
    models, _ = fitted
    for part in ("prior", "posterior", "label_token"):
        assert not hasattr(models["hazard_tuned_no_day"].net, part)
        assert hasattr(models["hazard_tuned"].net, part)
    width = models["hazard_tuned"].settings.width
    sizes = {
        name: models[name].net.hazard[0].in_features
        for name in ("hazard_tuned_no_day", "hazard_tuned")
    }
    day = models["hazard_tuned"].settings.day_factor_dim
    assert sizes["hazard_tuned"] - sizes["hazard_tuned_no_day"] == day
    assert sizes["hazard_tuned_no_day"] > width  # the site vector is still given


def test_with_calendar_only_inputs_other_inputs_change_nothing(fitted):
    models, fresh = fitted
    moved = fresh.copy()
    moved["arrival_hour"] = (moved["arrival_hour"] + 5) % 24
    moved["hours_since_plug_in"] = moved["hours_since_plug_in"] * 2.0 + 1.0
    moved["fleet_at_tau"] = moved["fleet_at_tau"] + 3
    moved["user_known"] = ~moved["user_known"].astype(bool)
    moved["history_mean_stay_h"] = moved["history_mean_stay_h"].fillna(1.0) + 2.0
    same = {
        name: bytes_of(models[name].predict(fresh, hf.KAPPAS))
        == bytes_of(models[name].predict(moved, hf.KAPPAS))
        for name in ("hazard_tuned_calendar", "hazard_tuned")
    }
    assert same == {"hazard_tuned_calendar": True, "hazard_tuned": False}
    assert models["hazard_tuned_calendar"].scaler.keep is not None
    assert models["hazard_tuned"].scaler.keep is None


# --- 3. refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODELS)
def test_each_model_is_refused_without_usable_settings_as_hazard_tuned_is(
    name, tmp_path, monkeypatch, capsys
):
    hf.use_small_tuned(monkeypatch, tmp_path)
    monkeypatch.setattr(hts, "TUNED", tmp_path / "absent.json")
    for refused in (name, "hazard_tuned"):
        with pytest.raises(MatrixError, match="absent"):
            module(refused).make_model(MATRIX)
    assert learn_run.main(["--model", name, "--tables", str(tmp_path)]) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN:") and "absent" in printed
    hf.use_small_tuned(monkeypatch, tmp_path)
    for usable in (name, "hazard_tuned"):
        assert module(usable).make_model(MATRIX).tuned == hf.SMALL_TUNED
    document = json.loads(hts.TUNED.read_text())
    document["settings"] = Setting(8, 0.003, 1.0, 4).as_dict()  # not a candidate of the grid
    hts.TUNED.write_text(json.dumps(document))
    for refused in (name, "hazard_tuned"):
        with pytest.raises(MatrixError, match="not a candidate"):
            module(refused).make_model(MATRIX)
    document["grid"]["sha256"] = "0" * 64
    document["settings"] = hf.SMALL_TUNED.as_dict()
    hts.TUNED.write_text(json.dumps(document))
    for refused in (name, "hazard_tuned"):
        with pytest.raises(MatrixError, match="another grid"):
            module(refused).make_model(MATRIX)
    model_class = next(
        value
        for value in vars(module(name)).values()
        if isinstance(value, type)
        and issubclass(value, hazard_tuned.HazardTunedModel)
        and value is not hazard_tuned.HazardTunedModel
    )
    with pytest.raises(ValueError, match="settings of the tuned settings file"):
        model_class(rho=0.0, kappas=hf.KAPPAS, tuned=None)


# --- 4. end to end -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = _units()
    run, tables = write_build(folder, units)
    rows = sum(len(units[u]) for u in GATE_UNITS)
    return ["--tables", str(tables), "--truth-run", str(run)], rows


@pytest.mark.parametrize("name", MODELS)
def test_learn_run_scores_each_ablation_model_end_to_end(
    name, build, tmp_path, monkeypatch, capsys
):
    arguments, rows = build
    hf.use_small_tuned(monkeypatch, tmp_path)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    code = learn_run.main([*arguments, "--matrix", str(matrix), "--model", name])
    printed = capsys.readouterr().out
    assert code == 0, printed
    results = json.loads((out / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == name and results["fit_units"] == list(TRAIN_UNITS)
    assert not results["failed_configurations"]
    table = read_table(out / learn_run.TABLE_NAME)
    assert len(table) == rows and table["d_hat"].notna().all()
