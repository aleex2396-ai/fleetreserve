"""Extra input columns through the shared path (`model_inputs`, `learn_run`, `dial_choice`), on
made-up gate units at sites of the committed site tables, with a made-up verified truth build and
made-up context sources; and the look-ahead fence.

1. A model declaring `extra_inputs` is given them in its forecast frame, appended in order and
   equal to `ExtraInputs.add` on each unit's table; `learn.json` records them; a model without
   them gets the frame it got before.
2. A look-ahead model runs only in `learn_run`'s development evaluation, under its listed module
   name, with the label in the summary's first lines and in both results files; under another
   name, in `dial_choice`, or through `check_model` without the allowance it is refused before
   anything is fitted.
3. `load_context`: a missing 9/80 settings file or weather file is "could not run".
"""

import hashlib
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_cars, matrix_document, write_build
from learn_fake import small_matrix

from fleetreserve import (
    context_places,
    dial_choice,
    gate_run,
    hazard_inputs,
    learn_features,
    learn_run,
    model_inputs,
    nine_eighty_settings,
    run_verify,
    school_inputs,
    weather_lookahead,
)
from fleetreserve.forecast_model import KEY_COLUMNS, TABLE_COLUMNS
from fleetreserve.gate_config import MatrixError

TRAINING = "caltech"
PLAIN = ("regime_covid", "school_holiday", "energy_drawn_kwh")
AHEAD = (*PLAIN, *weather_lookahead.COLUMNS)
LISTED = "hazard_tuned_inputs_lookahead"


def _weather():
    hours = pd.date_range("2019-12-01", periods=24 * 70, freq="h", tz="UTC")
    k = np.arange(len(hours), dtype=float)
    return pd.DataFrame(
        {"temperature": k % 17, "precipitation": k % 3, "snowfall": (k % 5 == 0) * 1.0},
        index=pd.DatetimeIndex(hours, name="time"),
    )


def _sources():
    weather = _weather()
    places = {p for site in ("nrel", "OSL_1", "BER") for p in context_places.SITE_PLACES[site]}
    return hazard_inputs.ContextSources(
        weather=dict.fromkeys(places, weather), school=school_inputs.load_terms()
    )


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        GATE_UNITS[0]: gate_cars(5, "nrel"),
        GATE_UNITS[1]: gate_cars(8, "nrel", early_every=4),
        GATE_UNITS[2]: gate_cars(4, "OSL_1", early_every=3) + gate_cars(4, "BER", early_every=2),
        TRAINING: gate_cars(8, "caltech"),
    }
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)], tables


def _rows(frame, kappas):
    """A forecast table with one row per car of `frame` and level, departure 1 h."""
    rows = []
    for key in frame[list(KEY_COLUMNS)].itertuples(index=False):
        for kappa in kappas:
            row = dict.fromkeys(TABLE_COLUMNS, float("nan"))
            row.update(zip(KEY_COLUMNS, key, strict=True))
            row.update(kappa=float(kappa), rho=0.0, d_hat=1.0)
            rows.append(row)
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))


class _Extra:
    """A learned model with extra inputs: a constant departure of one hour."""

    name = "extra"
    fit_units = (TRAINING,)
    look_ahead = False

    def __init__(self, names=PLAIN, look_ahead=False):
        self.extra_inputs = names
        self.look_ahead = look_ahead
        self.context_sources = _sources()
        self.context_files = {"data/raw/x.csv": "0" * 64}
        self.fitted = 0
        self.given = None

    def fit(self, frame):
        self.fitted += 1

    def predict(self, frame, kappas):
        self.given = frame
        return _rows(frame, kappas)


def _module(monkeypatch, name, model):
    module = types.ModuleType(f"fleetreserve.models.{name}")
    module.make_model = lambda matrix: model
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _expected(tables, names, look_ahead):
    from fleetreserve.days_from_truth import load_unit_days

    extra = model_inputs.ExtraInputs(names, _sources(), look_ahead)
    frames = []
    for unit in GATE_UNITS:
        days = load_unit_days(tables, unit, rho=0.0, no_fit="cap", small_fleet="keep")
        table = extra.add(learn_features.build_table(days.cars, rho=0.0), days.cars)
        frames.append(learn_features.feature_frame(table))
    frame = pd.concat(frames, ignore_index=True)
    return frame[frame["counting"]].reset_index(drop=True)


def test_a_model_with_extra_inputs_is_given_them(build, monkeypatch, capsys, tmp_path):
    arguments, tables = build
    model = _Extra()
    _module(monkeypatch, "extra", model)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    assert learn_run.main([*arguments, "--model", "extra"]) == 0, capsys.readouterr().out
    given = model.given
    assert list(given.columns[-len(PLAIN) :]) == list(PLAIN)
    assert model.fitted == 1
    pd.testing.assert_frame_equal(given, _expected(tables, PLAIN, False))
    results = json.loads((tmp_path / learn_run.RESULTS_NAME).read_text())
    assert results["context_inputs"]["columns"] == list(PLAIN)
    assert results["context_inputs"]["look_ahead_columns"] == []
    assert results["context_inputs"]["files_sha256"] == {"data/raw/x.csv": "0" * 64}
    assert "look_ahead" not in results


def test_the_look_ahead_model_runs_only_as_the_labelled_development_what_if(
    build, monkeypatch, capsys, tmp_path
):
    arguments, tables = build
    model = _Extra(AHEAD, look_ahead=True)
    _module(monkeypatch, LISTED, model)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    assert learn_run.main([*arguments, "--model", LISTED]) == 0
    out = capsys.readouterr().out
    assert out.startswith(model_inputs.LOOK_AHEAD_WHAT + "\n")
    pd.testing.assert_frame_equal(model.given, _expected(tables, AHEAD, True))
    for name in (learn_run.RESULTS_NAME, learn_run.GATE_NAME):
        document = json.loads((tmp_path / name).read_text())
        assert document["look_ahead"]["what"] == model_inputs.LOOK_AHEAD_WHAT
        assert document["look_ahead"]["columns"] == list(weather_lookahead.COLUMNS)


def test_the_look_ahead_model_under_another_name_is_refused(build, monkeypatch, capsys):
    arguments, _ = build
    model = _Extra(AHEAD, look_ahead=True)
    _module(monkeypatch, "renamed", model)
    assert learn_run.main([*arguments, "--model", "renamed"]) == 1
    assert capsys.readouterr().out.startswith("REFUSED")
    assert model.fitted == 0 and model.given is None


def test_dial_choice_gives_a_model_its_extra_inputs(build):
    from fleetreserve.days_from_truth import load_unit_days
    from fleetreserve.gate_config import load

    arguments, tables = build
    matrix = load(arguments[1])
    units = {
        u: load_unit_days(tables, u, rho=0.0, no_fit="cap", small_fleet="keep")
        for u in (*GATE_UNITS, TRAINING)
    }
    model = _Extra()
    gate = {u: units[u] for u in GATE_UNITS}
    dial_choice.make_factory("extra", model, matrix, gate, {TRAINING: units[TRAINING].cars}, 0.1)
    pd.testing.assert_frame_equal(model.given, _expected(tables, PLAIN, False))
    with pytest.raises(learn_run.Refused):
        dial_choice.make_factory(LISTED, _Extra(AHEAD, True), matrix, gate, {}, 0.1)


def test_dial_choice_refuses_the_look_ahead_model(build, monkeypatch, capsys, tmp_path):
    arguments, _ = build
    names = [
        "main-rounded-hour-k0.3-empirical",
        "main-rounded-hour-k0.3-oracle",
        "main-rounded-hour-k0.3-eps0.03-empirical",
        "main-rounded-hour-k0.3-eps0.03-oracle",
    ]
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(names)))
    model = _Extra(AHEAD, look_ahead=True)
    _module(monkeypatch, LISTED, model)
    code = dial_choice.main(["--model", LISTED, "--matrix", str(matrix), *arguments[2:]])
    out = capsys.readouterr().out
    assert code == 1 and out.startswith("REFUSED") and "look-ahead" in out
    assert model.fitted == 0


def test_check_model_and_extra_of_refuse_without_the_allowance():
    listed_plain = _Extra(PLAIN)
    with pytest.raises(learn_run.Refused):
        learn_run.check_model(LISTED, listed_plain)
    ahead = _Extra(AHEAD, look_ahead=True)
    with pytest.raises(learn_run.Refused):
        learn_run.check_model(LISTED, ahead)
    assert learn_run.check_model(LISTED, ahead, look_ahead_allowed=True) is False
    with pytest.raises(model_inputs.LookAheadRefused):
        model_inputs.extra_of(ahead)
    # declaring a look-ahead column alone makes a model a look-ahead model
    quiet = _Extra(AHEAD, look_ahead=False)
    with pytest.raises(learn_run.Refused):
        learn_run.check_model(LISTED, quiet)
    assert model_inputs.reads_look_ahead(quiet)


def test_extra_of_and_the_labels_of_a_model_without_extra_inputs():
    class Plain:
        name = "plain"

    assert model_inputs.extra_of(Plain()) is None
    document = {}
    model_inputs.mark(document, Plain())
    assert document == {} and model_inputs.summary_head(document) == []
    unknown = _Extra(("energy_before_kwh",))
    with pytest.raises(ValueError):
        model_inputs.extra_of(unknown)
    twice = _Extra(("regime_covid", "regime_covid"))
    with pytest.raises(ValueError):
        model_inputs.extra_of(twice)
    bare = _Extra()
    bare.context_sources = None
    with pytest.raises(ValueError):
        model_inputs.extra_of(bare)


def _phase(path, parity=0):
    document = {"about": "made up", "estimate": {}, "parity": parity, "results": {}}
    path.write_text(json.dumps(document))
    return path


def test_load_context_reads_every_place_and_says_could_not_run_when_a_file_is_missing(tmp_path):
    root = tmp_path / "root"
    phase = _phase(tmp_path / "phase.json", parity=1)
    with pytest.raises(nine_eighty_settings.PhaseError):
        model_inputs.load_context(root, tmp_path / "absent.json")
    with pytest.raises(model_inputs.ContextError):
        model_inputs.load_context(root, phase)
    assert issubclass(model_inputs.ContextError, MatrixError)
    folder = root / "data" / "raw" / context_places.WEATHER_SET
    folder.mkdir(parents=True)
    times = pd.date_range("2019-01-01", periods=30, freq="h")
    head = "latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation"
    body = "\n".join(f"{t:%Y-%m-%dT%H:%M},1.0,0.0,0.0" for t in times)
    text = f"{head}\n0,0,0,0,GMT,GMT\n\ntime,temperature_2m (°C),precipitation (mm),"
    text += f"snowfall (cm)\n{body}\n"
    for place in context_places.PLACES:
        (folder / context_places.weather_file(place)).write_text(text, encoding="utf-8")
    with pytest.raises(model_inputs.ContextError):
        model_inputs.load_context(root, phase)  # no OxCGRT file
    oxcgrt = root / hazard_inputs.OXCGRT_FILE
    oxcgrt.parent.mkdir(parents=True)
    oxcgrt.write_text(
        "CountryCode,CountryName,RegionCode,RegionName,CityCode,CityName,Jurisdiction,"
        "01Jan2020,02Jan2020\nNOR,Norway,,,,,NAT_TOTAL,1,2\n"
    )
    found = model_inputs.load_context(root, phase)
    assert found.sources.nine_eighty_parity == 1
    assert set(found.sources.weather) == set(context_places.PLACES)
    assert found.files[hazard_inputs.OXCGRT_FILE.as_posix()] == (
        hashlib.sha256(oxcgrt.read_bytes()).hexdigest()
    )
    assert len(found.files) == len(context_places.PLACES) + 3
