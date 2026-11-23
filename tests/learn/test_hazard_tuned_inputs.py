"""`models.hazard_tuned_inputs` and its look-ahead what-if on made-up cars (`hazard_fake`) with
made-up context sources and a made-up tuned settings file: the encoding by hand, the wider
network, the extra columns read, the refusals, and `make_model` without its context files."""

from types import SimpleNamespace

import hazard_fake as hf
import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import pytest
import survival_fake as sf
import torch
from truth_tables_fake import date_of

from fleetreserve import context_places, hazard_inputs, model_inputs, school_inputs
from fleetreserve.forecast_model import check_table
from fleetreserve.gate_config import MatrixError
from fleetreserve.hazard_data import N_INPUTS, Scaler
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.learn_features import build_table, feature_frame, with_context
from fleetreserve.models import hazard_tuned_inputs, hazard_tuned_inputs_lookahead

MATRIX = SimpleNamespace(rho=0.0, configurations=[SimpleNamespace(kappa=k) for k in hf.KAPPAS])
DAYS, FRESH = 40, 8


def weather(period=23):
    hours = pd.date_range("2019-12-01", periods=24 * 90, freq="h", tz="UTC")
    k = np.arange(len(hours), dtype=float)
    return pd.DataFrame(
        {"temperature": k % period, "precipitation": k % 3, "snowfall": (k % 7 == 0) * 1.0},
        index=pd.DatetimeIndex(hours, name="time"),
    )


def context(period=23) -> model_inputs.Context:
    places = {p for site in hf.SITES for p in context_places.SITE_PLACES[site]}
    days = pd.Index(pd.date_range("2020-01-01", periods=120).date, name="date")
    jurisdictions = {context_places.SITE_OXCGRT[site] for site in hf.SITES}
    sources = hazard_inputs.ContextSources(
        weather=dict.fromkeys(places, weather(period)),
        oxcgrt={j: pd.Series(np.arange(120.0), index=days) for j in jurisdictions},
        nine_eighty_parity=0,
        school=school_inputs.load_terms(),
    )
    return model_inputs.Context(sources, {"made/up.csv": "0" * 64})


@pytest.fixture(scope="module")
def cars():
    torch.set_num_threads(1)
    fakes, users = hf.known_cars(DAYS + FRESH, seed=5)
    return sf.cars_table(fakes, users)


def fresh_frame(cars, model):
    extra = model_inputs.ExtraInputs(model.extra_inputs, model.context_sources, model.look_ahead)
    table = extra.add(build_table(cars, rho=0.0), cars)
    later = table[table["date"] >= date_of(DAYS)].reset_index(drop=True)
    return feature_frame(later)


def fitted(monkeypatch, folder, cars, make=hazard_tuned_inputs.HazardTunedInputsModel, period=23):
    hf.use_small_tuned(monkeypatch, folder)
    from fleetreserve.hazard_tune_settings import load_tuned

    model = make(rho=0.0, kappas=hf.KAPPAS, tuned=load_tuned(), context=context(period))
    model.fit(hf.train_cars(cars, DAYS))
    return model


def test_the_encoding_by_hand_keeps_the_old_inputs_first(cars):
    table = with_context(build_table(cars, rho=0.0)).head(3).copy()
    table["school_holiday"] = [1.0, 3.0, np.nan]
    settings = HazardSettings()
    scaler = model_inputs.InputsScaler.fit(table, settings, ("school_holiday",))
    out = scaler.transform(table)
    assert out.shape == (3, N_INPUTS + 2) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:, :N_INPUTS], Scaler.fit(table, settings).transform(table))
    # mean 2 and standard deviation 1 of the finite values: -1, 1, and 0 with its missing flag
    assert out[:, N_INPUTS].tolist() == [-1.0, 1.0, 0.0]
    assert out[:, N_INPUTS + 1].tolist() == [0.0, 0.0, 1.0]
    table.loc[table.index[0], "school_holiday"] = np.inf
    with pytest.raises(ValueError):
        scaler.transform(table)


def test_the_variant_fits_a_wider_network_and_reads_its_extra_columns(monkeypatch, tmp_path, cars):
    model = fitted(monkeypatch, tmp_path, cars)
    width = N_INPUTS + 2 * len(hazard_tuned_inputs.EXTRA_INPUTS)
    assert model.scaler.n_inputs == width
    assert model.net.token[0].in_features == width
    frame = fresh_frame(cars, model)
    table = check_table(model.predict(frame, hf.KAPPAS))
    assert len(table) == len(frame) * len(hf.KAPPAS) and table["d_hat"].notna().any()
    other = fitted(monkeypatch, tmp_path, cars, period=13)
    moved = other.predict(fresh_frame(cars, other), hf.KAPPAS)
    assert not moved["d_hat"].equals(table["d_hat"]) or not moved["r_hat"].equals(table["r_hat"])
    with pytest.raises(KeyError):
        model.predict(frame.drop(columns=["school_holiday"]), hf.KAPPAS)


def test_the_look_ahead_model_forecasts_development_units_only(monkeypatch, tmp_path, cars):
    model = fitted(
        monkeypatch,
        tmp_path,
        cars,
        make=hazard_tuned_inputs_lookahead.HazardTunedInputsLookaheadModel,
    )
    assert model.look_ahead and model_inputs.reads_look_ahead(model)
    frame = fresh_frame(cars, model)
    with pytest.raises(ValueError, match="development units only"):
        model.predict(frame, hf.KAPPAS)  # the fake's unit is not a development unit
    renamed = frame.assign(unit="nrel-from-may-2019")
    assert len(check_table(model.predict(renamed, hf.KAPPAS))) == len(frame) * len(hf.KAPPAS)


def test_make_model_without_the_9_80_settings_file_could_not_run(monkeypatch, tmp_path):
    hf.use_small_tuned(monkeypatch, tmp_path)
    monkeypatch.setattr(model_inputs, "PHASE", tmp_path / "absent.json")
    for module in (hazard_tuned_inputs, hazard_tuned_inputs_lookahead):
        with pytest.raises(MatrixError):
            module.make_model(MATRIX)
    monkeypatch.setattr(model_inputs, "load_context", lambda: context())
    model = hazard_tuned_inputs.make_model(MATRIX)
    assert model.name == "hazard_tuned_inputs" and model.context_files == {"made/up.csv": "0" * 64}
