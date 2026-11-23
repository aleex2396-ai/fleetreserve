"""Every committed run matrix under `gate/` gives `learn_run` and `cold_start` a model with a
forecast at every level the gate reads, for every model the run list names.

The cases (`CASES`): `lgbm_tail` on each of the 17 committed matrices (the gate matrix, the two T8
training matrices and the rehearsal and sealed matrices of the run list), for `learn_run` and for
the cold start; and each model a `learn_run` or `cold_start` step of `run-lists/run-list.json`
names (`--model`, and the cold start's `--reference`, by default `lgbm`) on each matrix that step
reads (its rehearsal and sealed `--matrix`). Made-up units (`gate_fake.gate_cars`, the matrix's
rho) at real site names, so the context inputs of `hazard_tuned_inputs` can be made up for them
(`_context`); that model's tuned settings are the small made-up ones (`hazard_fake`).

1. `learn_run.evaluate` fits the model and predicts its table as on a run; the gate's step that
   reads the table (`gate_run.run_gate`, replaced here by its forecasting step alone) then gives
   every configuration of the matrix its forecasts on one day of each unit
   (`gate_day._forecasts`: the kappa of the configuration and, under a driver dial, every level up
   to it). The gate units of every matrix are replaced by the three made-up development units
   (the T8 matrices score training units, the sealed matrices units with no made-up tables here;
   the levels do not depend on the units).
2. For each dial of the matrix's dial family, `cold_start.check_models`, `fit` and `forecast_at`
   (k = 0, the last day of each unit scored) make the cold start's table, and the gate's
   forecasting step reads it on that day for that dial's configuration and its oracle partner.
3. Each model the run list names, on the one-kappa rehearsal matrix its first step reads: its
   forecasts at the matrix's own kappa are the same as under the rule before
   `driver_risk.matrix_levels` (the matrix's own kappas, as at c72df31, whose model modules are
   those of this commit), fitted on the same rows. Not so for `hazard_tuned_inputs`
   (`JOINT_FIT`), whose network fits every level together: there only the departure columns
   (`d_hat`, `d_median`) are the same; its rate, room, need and survival-bound columns at the
   own kappa differ, since the network is now fitted at the six levels.
4. `hazard_tuned_inputs`, kappa 0.5, every column: its forecast rows are the same from (a) its
   one-kappa rehearsal matrix (driver dials, no other level), (b) that matrix with undialled
   configurations at the five lower levels added, and (c) `gate/matrix.json`: each matrix gives
   the network the same six levels (`driver_risk.matrix_levels`), so the same fit.
"""

import json
from dataclasses import replace
from pathlib import Path

import hazard_fake as hf
import numpy as np
import pandas as pd
import pytest
from gate_fake import gate_cars, write_build

from fleetreserve import (
    cold_start,
    compare_run,
    context_places,
    dial_choice,
    driver_risk,
    gate_config,
    gate_day,
    gate_run,
    hazard_inputs,
    learn_run,
    model_inputs,
    school_inputs,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import KEY_COLUMNS
from fleetreserve.truth_build import TRAIN_UNITS

ROOT = Path(__file__).resolve().parents[2]
MATRICES = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "gate").rglob("*.json"))
RUN_LIST = json.loads((ROOT / "run-lists" / "run-list.json").read_text())
LEARN, COLD = "fleetreserve.learn_run", "fleetreserve.cold_start"
DEVELOPMENT = {
    "nrel-before-may-2019": (("nrel", 5, 12),),
    "nrel-from-may-2019": (("nrel", 24, 4),),
    "norway-development-pooled": (("OSL_1", 12, 3), ("OSL_2", 12, 5)),
}


def _steps() -> dict:
    """(module, model) -> the matrices the run list's steps of that module give that model."""
    found = {}
    for entry in RUN_LIST["entries"]:
        command = entry["command"]
        module = next((m for m in (LEARN, COLD) if m in command), None)
        if module is None:
            continue
        models = [command[command.index("--model") + 1]]
        if module == COLD:
            given = "--reference" in command
            models.append(command[command.index("--reference") + 1] if given else "lgbm")
        matrices = [
            extra[extra.index("--matrix") + 1]
            for extra in (entry.get("extra") or {}).values()
            if "--matrix" in extra
        ]
        for model in models:
            found.setdefault((module, model), [])
            found[(module, model)] += [m for m in matrices if m not in found[(module, model)]]
    return found


STEPS = _steps()
CASES = sorted(
    {(module, "lgbm_tail", path) for module in (LEARN, COLD) for path in MATRICES}
    | {(module, model, path) for (module, model), paths in STEPS.items() for path in paths}
)
MODELS = sorted({model for _, model in STEPS})
JOINT_FIT = ("hazard_tuned_inputs",)  # one network over all levels (module text, 3)
DEPARTURE = ["d_hat", "d_median"]


class _Read(Exception):
    """Raised by the stand-in gate once every configuration has read the table."""


def test_the_committed_matrices_and_the_run_lists_models():
    assert len(MATRICES) == 17
    assert sum(m.startswith("gate/rehearsal-matrix/") for m in MATRICES) == 7
    assert sum(m.startswith("gate/sealed-matrix/") for m in MATRICES) == 7
    named = ("cox", "earlier_gbm", "earlier_population", "hazard_tuned_inputs", "km", "lgbm")
    assert sorted([*named, "lgbm_tail"]) == MODELS
    assert all(path in MATRICES for _, _, path in CASES)


def _cars() -> dict:
    units = {
        unit: [c for site, days, early in parts for c in gate_cars(days, site, early_every=early)]
        for unit, parts in DEVELOPMENT.items()
    }
    for j, unit in enumerate(TRAIN_UNITS):
        units[unit] = gate_cars(24, unit, early_every=3 + j)  # the site of each is its unit
    return units


def _context() -> model_inputs.Context:
    """Made-up context sources for every site of `_cars` (as `test_hazard_tuned_inputs` makes
    them for its sites)."""
    sites = {site for parts in DEVELOPMENT.values() for site, _, _ in parts}
    sites |= set(TRAIN_UNITS)
    places = {p for site in sites for p in context_places.SITE_PLACES[site]}
    hours = pd.date_range("2019-12-01", periods=24 * 90, freq="h", tz="UTC")
    k = np.arange(len(hours), dtype=float)
    weather = pd.DataFrame(
        {"temperature": k % 23, "precipitation": k % 3, "snowfall": (k % 7 == 0) * 1.0},
        index=pd.DatetimeIndex(hours, name="time"),
    )
    days = pd.Index(pd.date_range("2020-01-01", periods=120).date, name="date")
    jurisdictions = {context_places.SITE_OXCGRT[site] for site in sites}
    sources = hazard_inputs.ContextSources(
        weather=dict.fromkeys(places, weather),
        oxcgrt={j: pd.Series(np.arange(120.0), index=days) for j in jurisdictions},
        nine_eighty_parity=0,
        school=school_inputs.load_terms(),
    )
    return model_inputs.Context(sources, {"made/up.csv": "0" * 64})


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    _, tables = write_build(tmp_path_factory.mktemp("build"), _cars())
    return tables, {}


@pytest.fixture(autouse=True)
def made_up_inputs(monkeypatch, tmp_path):
    hf.use_small_tuned(monkeypatch, tmp_path)
    monkeypatch.setattr(model_inputs, "load_context", _context)


def _matrix(build, path: str):
    """The matrix at `path` with the made-up development units as gate units (module text, 1),
    the gate units' days and the training units' cars tables."""
    return _prepared(build, gate_config.read(ROOT / path)[0])


def _prepared(build, matrix):
    matrix = replace(matrix, gate_units=tuple(DEVELOPMENT))
    tables, loaded = build
    key = (matrix.rho, matrix.no_fit, matrix.small_fleet)
    if key not in loaded:
        loaded[key] = {
            name: load_unit_days(
                tables, name, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
            )
            for name in (*DEVELOPMENT, *TRAIN_UNITS)
        }
    every = loaded[key]
    units = {name: every[name] for name in matrix.gate_units}
    return matrix, units, {name: every[name].cars for name in TRAIN_UNITS}


def _read_all(factory, units, matrix, configurations, day: int = 0) -> int:
    """The gate's forecasting step on day `day` of each unit for each configuration."""
    read = 0
    for name in units:
        forecaster = factory(units[name], matrix)
        for configuration in configurations:
            found = gate_day._forecasts(units[name].days[day], forecaster, configuration)
            read += sum(f is not None for f in found)
    return read


@pytest.mark.parametrize(("model", "path"), [(m, p) for k, m, p in CASES if k == LEARN])
def test_learn_run_forecasts_at_every_level_the_gate_reads(build, model, path, monkeypatch):
    matrix, units, fit_frames = _matrix(build, path)

    def gate(matrix, units, *, factory, **_):
        assert _read_all(factory, units, matrix, matrix.configurations) > 0
        raise _Read

    monkeypatch.setattr(gate_run, "run_gate", gate)
    built = learn_run.load_model(model, matrix)
    with pytest.raises(_Read):
        learn_run.evaluate(model, built, matrix, units, fit_frames, workers=1)


@pytest.mark.parametrize(("model", "path"), [(m, p) for k, m, p in CASES if k == COLD])
def test_the_cold_start_forecasts_at_every_level_the_gate_reads(build, model, path):
    matrix, units, fit_frames = _matrix(build, path)
    (checked,) = cold_start.check_models((model,), matrix, tuple(units))
    cold_start.fit(checked, fit_frames)
    scored = {name: [len(units[name].days) - 1] for name in units}
    _, dials = dial_choice.find_family(matrix)
    assert dials
    for dial in dials:
        found = compare_run.dial_configurations(matrix, dial.driver_risk)
        table = cold_start.forecast_at(checked, units, 0, scored, found["forecast"].kappa)
        factory = cold_start.TableFactory(table)
        assert _read_all(factory, units, matrix, list(found.values()), day=-1) > 0


def _forecasts_at_own_kappas(build, model: str, path: str, levels) -> pd.DataFrame:
    matrix, units, fit_frames = _matrix(build, path)
    built = learn_run.load_model(model, matrix)
    built.fit(pd.concat(list(fit_frames.values()), ignore_index=True))
    frame = learn_run.forecast_frame(False, units, matrix.gate_units, model_inputs.extra_of(built))
    table = built.predict(frame, list(levels(matrix.configurations)))
    own = sorted({c.kappa for c in matrix.configurations})
    found = table[table["kappa"].isin(own)]
    return found.sort_values([*KEY_COLUMNS, "kappa"]).reset_index(drop=True)


@pytest.mark.parametrize("model", MODELS)
def test_the_forecasts_at_a_one_kappa_matrixs_own_kappa_are_unchanged(build, model, monkeypatch):
    path = next(p for (_, m), paths in STEPS.items() if m == model for p in paths)
    matrix, _ = gate_config.read(ROOT / path)
    assert len({c.kappa for c in matrix.configurations}) == 1
    now = _forecasts_at_own_kappas(build, model, path, driver_risk.matrix_levels)

    def before(configurations):
        return tuple(sorted({c.kappa for c in configurations}))

    monkeypatch.setattr(driver_risk, "matrix_levels", before)
    then = _forecasts_at_own_kappas(build, model, path, before)
    assert len(now) > 0
    if model in JOINT_FIT:
        now, then = (
            now[[*KEY_COLUMNS, "kappa", *DEPARTURE]],
            then[[*KEY_COLUMNS, "kappa", *DEPARTURE]],
        )
    pd.testing.assert_frame_equal(now, then, check_exact=True)


def _kappa_half_rows(build, matrix) -> tuple[list, pd.DataFrame]:
    """The levels `learn_run` predicts at for `matrix`, and `hazard_tuned_inputs`' forecast rows
    at kappa 0.5 from the model fitted and predicted as `learn_run` does, in every column."""
    matrix, units, fit_frames = _prepared(build, matrix)
    built = learn_run.load_model("hazard_tuned_inputs", matrix)
    built.fit(pd.concat(list(fit_frames.values()), ignore_index=True))
    frame = learn_run.forecast_frame(False, units, matrix.gate_units, model_inputs.extra_of(built))
    levels = list(driver_risk.matrix_levels(matrix.configurations))
    table = built.predict(frame, levels)
    rows = table[table["kappa"] == 0.5]
    return levels, rows.sort_values([*KEY_COLUMNS]).reset_index(drop=True)


def test_hazard_tuned_inputs_kappa_half_rows_are_the_same_from_any_matrix_giving_the_six_levels(
    build,
):
    path = "gate/rehearsal-matrix/hazard_tuned_inputs.json"
    rehearsal, _ = gate_config.read(ROOT / path)
    assert {c.kappa for c in rehearsal.configurations} == {0.5}
    assert any(c.driver_risk is not None for c in rehearsal.configurations)
    plain = next(
        c for c in rehearsal.configurations if c.driver_risk is None and c.departure == "empirical"
    )
    added = tuple(
        replace(plain, kappa=k, name=f"added-level-{k:g}", oracle_partner=None)
        for k in driver_risk.LEVELS
        if k != 0.5
    )
    with_five = replace(rehearsal, configurations=(*rehearsal.configurations, *added))
    gate_matrix, _ = gate_config.read(ROOT / "gate" / "matrix.json")
    levels, base = _kappa_half_rows(build, rehearsal)
    assert levels == sorted(driver_risk.LEVELS)
    assert len(base) > 0
    assert set(base.columns) > {"d_hat", "d_median", "r_hat", "e_cap", "e_need"}
    for other in (with_five, gate_matrix):
        other_levels, rows = _kappa_half_rows(build, other)
        assert other_levels == levels
        assert list(rows.columns) == list(base.columns)
        for column in base.columns:  # every column, named, so a failure says which
            pd.testing.assert_series_equal(rows[column], base[column], check_exact=True)
