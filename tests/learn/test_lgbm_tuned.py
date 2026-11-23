"""`models.lgbm_ctx`, `models.lgbm_tuned` and `models.lgbm_tuned_surv`: the pooled LightGBM with
the tail-history and context columns, at the present settings or at those of a settings file.

Made-up cars: ten users a day at each of two sites over 190 days, `caltech` (a workplace, four
cars in ten leaving early) and `OSL_S` (the other type, eight in ten leaving early); the first
150 days are fitted on, the later 40 forecast.

1. The five older learned models' forecast tables keep the bytes they had at the commit before
   these models were added, and the settings arguments of `QuantileSet` and `ConditionalBounds`
   default to the present settings (`learn_tune_grid.PRESENT` is `models.lgbm`'s).
2. Each new model is fitted on the training units only, passes the learn guard, and has boosters
   on `FEATURE_COLUMNS`, `TAIL_HISTORY_COLUMNS` and `CONTEXT_COLUMNS`; `lgbm_ctx` at the present
   settings, the tuned ones at the settings given, the bound's boosters too.
3. `lgbm_tuned_surv` fills every other column as `lgbm_tuned` does; `lgbm_tuned` at the present
   settings is `lgbm_ctx`.
4. The context columns are read: the models compute them from `site` and `date` (a frame without
   them, or with wrong ones, gives the same forecast), a moved site changes the forecast, an
   unknown site fails, and the departure boosters split on the made-up site-type signal.
5. No leak: label columns handed to `predict` change nothing, and altering every outcome of the
   cars being forecast changes no forecast cell.
6. A tuned model is not built without settings: `make_model` refuses an absent settings file and
   one recording another grid hash, and `learn_run` then ends with exit 2 and writes nothing.
7. With a made-up settings file `learn_run --model lgbm_tuned` and `--model lgbm_tuned_surv` run
   end to end on made-up units (exit 0), and `lgbm_ctx` does without one.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import survival_fake as sf
from gate_fake import COMMIT, GATE_UNITS, gate_cars, write_build
from learn_fake import PAIR, small_matrix
from test_lgbm_tail import TRAIN_DAYS as TAIL_TRAIN_DAYS
from test_lgbm_tail import _made_up
from truth_tables_fake import date_of

from fleetreserve import gate_run, learn_run, run_verify
from fleetreserve import learn_tune_settings as lts
from fleetreserve.forecast_model import SURVIVAL_COLUMNS, check_table, read_table, write_table
from fleetreserve.gate_config import MatrixError
from fleetreserve.learn_features import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    TAIL_HISTORY_COLUMNS,
    build_table,
    feature_frame,
    with_context,
)
from fleetreserve.learn_tune_grid import PRESENT, Settings, load_grid
from fleetreserve.models import (
    conditional_fit,
    cqr,
    lgbm,
    lgbm_ctx,
    lgbm_surv,
    lgbm_tail,
    lgbm_tail_surv,
    lgbm_tuned,
    lgbm_tuned_surv,
)
from fleetreserve.truth_build import TRAIN_UNITS

KAPPAS = (0.1, 0.3)
ALL_KAPPAS = (0.05, 0.1, 0.2, 0.3)
TRAIN_DAYS = 150
DAYS = 190
SURVIVAL = list(SURVIVAL_COLUMNS)
OTHER = ["d_hat", "r_hat", "e_cap", "e_need", "d_median"]
TUNED = Settings(num_leaves=7, min_data_in_leaf=50, learning_rate=0.02, rounds=60)
NAMES = ("lgbm_ctx", "lgbm_tuned", "lgbm_tuned_surv")
# The SHA-256 of each older model's forecast table (`forecast_model.write_table`) at the commit
# before these models were added: fitted on the first 150 days of `test_lgbm_tail._made_up()` and
# written for the later 40 days at the four levels.
OLDER_SHA256 = {
    "lgbm": "5284432857b33e8d2a25dec702c9526cff136c1500794495d1c80de9a16a7186",
    "cqr": "58b6047c2d7fa7c6998c3c3b6fac0d434bc0dc55a37ac2cafa8b0dc52ce29573",
    "lgbm_tail": "26f5c1173e1adfbbdfb6090f2325d23775e49d58c8fe4338830a2361c1311e4a",
    "lgbm_surv": "bbcc0458ebb0d3c0cfaa1a75973f8cd903538a50c130f218d53d9e37f54d37c8",
    "lgbm_tail_surv": "1f4ca34b59f8bad15e3456995da7aef70b73a7b47e544d937d8e115d863220f8",
}
OLDER = {
    "lgbm": lgbm,
    "cqr": cqr,
    "lgbm_tail": lgbm_tail,
    "lgbm_surv": lgbm_surv,
    "lgbm_tail_surv": lgbm_tail_surv,
}
STAYS = {
    "caltech": ((0.4, 0.2, 3.8), (1.0, 5.0, 7.5)),
    "OSL_S": ((0.8, 0.2, 3.8), (1.0, 5.0, 7.5)),
}


# --- 1. the older models -------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(OLDER))
def test_the_older_models_forecast_tables_keep_their_bytes(name, tmp_path):
    cars = _made_up()
    train = cars[cars["date"] < date_of(TAIL_TRAIN_DAYS)].reset_index(drop=True)
    table = build_table(cars, rho=0.0)
    fresh = feature_frame(table[table["date"] >= date_of(TAIL_TRAIN_DAYS)].reset_index(drop=True))
    matrix = SimpleNamespace(rho=0.0, configurations=[SimpleNamespace(kappa=k) for k in ALL_KAPPAS])
    model = OLDER[name].make_model(matrix)
    model.fit(train)
    path = tmp_path / "table.csv"
    write_table(model.predict(fresh, ALL_KAPPAS), path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == OLDER_SHA256[name]


def test_the_present_settings_are_those_of_models_lgbm():
    assert PRESENT.rounds == lgbm.ROUNDS
    for key, value in PRESENT.overrides().items():
        assert lgbm.PARAMETERS[key] == value
    assert lgbm_ctx.lightgbm_settings(None) == (None, None)
    assert lgbm_ctx.lightgbm_settings(PRESENT) == (lgbm.PARAMETERS, lgbm.ROUNDS)
    parameters, rounds = lgbm_ctx.lightgbm_settings(TUNED)
    assert rounds == 60
    assert parameters == {
        **lgbm.PARAMETERS,
        "num_leaves": 7,
        "min_data_in_leaf": 50,
        "learning_rate": 0.02,
    }


# --- the made-up cars ----------------------------------------------------------------------------


def _made_up_sites(alter=None) -> pd.DataFrame:
    fakes, users = [], []
    for seed, (site, groups) in enumerate(STAYS.items()):
        found, names = sf.stay_cars(DAYS, 0, groups, seed=20 + seed, site=site)
        fakes += found
        users += names
    if alter is not None:
        alter(fakes)
    return sf.cars_table(fakes, users)


def _fresh(cars: pd.DataFrame, first_day: int = TRAIN_DAYS) -> pd.DataFrame:
    table = build_table(cars, rho=0.0)
    return feature_frame(table[table["date"] >= date_of(first_day)].reset_index(drop=True))


@pytest.fixture(scope="module")
def data():
    cars = _made_up_sites()
    train = cars[cars["date"] < date_of(TRAIN_DAYS)].reset_index(drop=True)
    return train, _fresh(cars)


@pytest.fixture(scope="module")
def fitted(data):
    train, _ = data
    out = {
        "lgbm_ctx": lgbm_ctx.LgbmCtxModel(rho=0.0, kappas=KAPPAS),
        "lgbm_tuned": lgbm_tuned.LgbmTunedModel(rho=0.0, kappas=KAPPAS, settings=TUNED),
        "lgbm_tuned_surv": lgbm_tuned_surv.LgbmTunedSurvModel(
            rho=0.0, kappas=KAPPAS, settings=TUNED
        ),
    }
    for model in out.values():
        model.fit(train)
    return out


@pytest.fixture(scope="module")
def forecasts(data, fitted):
    _, fresh = data
    return {name: check_table(model.predict(fresh, KAPPAS)) for name, model in fitted.items()}


def test_the_settings_arguments_default_to_the_present_settings(data):
    train, fresh = data
    table = build_table(train, rho=0.0)
    levels = lgbm.levels_of(KAPPAS)
    x = lgbm.features(fresh)
    default = lgbm.QuantileSet(levels).fit(table)
    explicit = lgbm.QuantileSet(levels, FEATURE_COLUMNS, dict(lgbm.PARAMETERS), lgbm.ROUNDS)
    explicit.fit(table)
    other = lgbm.QuantileSet(levels, FEATURE_COLUMNS, *lgbm_ctx.lightgbm_settings(TUNED)).fit(table)
    for key, booster in default.boosters.items():
        assert booster.num_trees() == lgbm.ROUNDS == explicit.boosters[key].num_trees()
        assert booster.params["num_leaves"] == lgbm.PARAMETERS["num_leaves"]
        assert other.boosters[key].num_trees() == 60
        assert other.boosters[key].params["num_leaves"] == 7
        assert np.array_equal(default.predict(x, *key), explicit.predict(x, *key))
    assert not np.array_equal(
        default.predict(x, "departure", 0.1), other.predict(x, "departure", 0.1)
    )
    bounds = conditional_fit.ConditionalBounds(KAPPAS).fit(table)
    same = conditional_fit.ConditionalBounds(
        KAPPAS, FEATURE_COLUMNS, dict(lgbm.PARAMETERS), lgbm.ROUNDS
    ).fit(table)
    for key, booster in bounds.boosters.items():
        assert booster.num_trees() == lgbm.ROUNDS
        assert np.array_equal(bounds.predict(x, *key), same.predict(x, *key))


# --- 2. the models -------------------------------------------------------------------------------


def test_each_model_is_fitted_on_the_training_units_with_the_three_column_lists(fitted):
    assert (*FEATURE_COLUMNS, *TAIL_HISTORY_COLUMNS, *CONTEXT_COLUMNS) == lgbm_ctx.COLUMNS
    assert lgbm_ctx.COLUMNS[: len(lgbm_tail.COLUMNS)] == lgbm_tail.COLUMNS
    plain = lgbm.LgbmModel(rho=0.0, kappas=KAPPAS)
    for name in NAMES:
        model = fitted[name]
        assert model.name == name
        assert model.fit_units == TRAIN_UNITS
        assert learn_run.check_model(name, model) is False
        assert model.levels == plain.levels
        assert set(model.quantiles.boosters) == {
            (quantity, level) for quantity, levels in plain.levels.items() for level in levels
        }
        for booster in model.quantiles.boosters.values():
            assert booster.feature_name() == list(lgbm_ctx.COLUMNS)


def _settings_of(booster) -> Settings:
    return Settings(
        booster.params["num_leaves"],
        booster.params["min_data_in_leaf"],
        booster.params["learning_rate"],
        booster.num_trees(),
    )


def test_the_boosters_have_the_present_or_the_given_settings(fitted):
    for key, booster in fitted["lgbm_ctx"].quantiles.boosters.items():
        assert _settings_of(booster) == PRESENT
        assert booster.params["alpha"] == key[1]
    for name in ("lgbm_tuned", "lgbm_tuned_surv"):
        for key, booster in fitted[name].quantiles.boosters.items():
            assert _settings_of(booster) == TUNED
            assert booster.params["alpha"] == key[1]
            for parameter in ("seed", "deterministic", "objective", "bagging_fraction", "max_bin"):
                assert booster.params[parameter] == lgbm.PARAMETERS[parameter]
    bounds = fitted["lgbm_tuned_surv"].bounds
    assert bounds.columns == lgbm_ctx.COLUMNS
    assert bounds.boosters
    for (_, kappa), booster in bounds.boosters.items():
        assert _settings_of(booster) == TUNED
        assert booster.params["alpha"] == kappa
        assert booster.feature_name() == list(lgbm_ctx.COLUMNS)
    assert lgbm_tuned_surv.SURVIVAL_BOUND == conditional_fit.CONDITIONAL_FIT


def test_a_tuned_model_is_not_built_without_settings():
    for make in (lgbm_tuned.LgbmTunedModel, lgbm_tuned_surv.LgbmTunedSurvModel):
        with pytest.raises(ValueError, match="settings"):
            make(rho=0.0, kappas=KAPPAS, settings=None)
        with pytest.raises(ValueError, match="settings"):
            make(rho=0.0, kappas=KAPPAS, settings=PRESENT.as_dict())
        with pytest.raises(TypeError):
            make(rho=0.0, kappas=KAPPAS)


# --- 3. the tables -------------------------------------------------------------------------------


def test_the_tables_have_the_format_and_differ_as_the_settings_do(data, forecasts):
    _, fresh = data
    for name in NAMES:
        table = forecasts[name]
        assert len(table) == len(fresh) * len(KAPPAS)
        assert (table[OTHER].to_numpy() >= 0).all()
    assert forecasts["lgbm_ctx"][SURVIVAL].isna().all().all()
    assert forecasts["lgbm_tuned"][SURVIVAL].isna().all().all()
    pd.testing.assert_frame_equal(
        forecasts["lgbm_tuned"].drop(columns=SURVIVAL),
        forecasts["lgbm_tuned_surv"].drop(columns=SURVIVAL),
        check_exact=True,
    )
    assert forecasts["lgbm_tuned_surv"][SURVIVAL[:5]].notna().all().all()
    assert (forecasts["lgbm_tuned_surv"]["survival_h3"] >= 3.0).all()
    assert (forecasts["lgbm_ctx"]["d_hat"] != forecasts["lgbm_tuned"]["d_hat"]).any()


def test_the_tuned_model_at_the_present_settings_is_lgbm_ctx(data, forecasts):
    train, fresh = data
    model = lgbm_tuned.LgbmTunedModel(rho=0.0, kappas=KAPPAS, settings=PRESENT)
    model.fit(train)
    pd.testing.assert_frame_equal(
        forecasts["lgbm_ctx"], check_table(model.predict(fresh, KAPPAS)), check_exact=True
    )


def test_two_fits_agree(data, forecasts):
    train, fresh = data
    again = lgbm_tuned_surv.LgbmTunedSurvModel(rho=0.0, kappas=KAPPAS, settings=TUNED)
    again.fit(train)
    pd.testing.assert_frame_equal(
        forecasts["lgbm_tuned_surv"], check_table(again.predict(fresh, KAPPAS)), check_exact=True
    )


# --- 4. the context columns ----------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_the_model_sets_the_context_columns_itself_from_site_and_date(
    data, fitted, forecasts, name
):
    _, fresh = data
    assert not set(CONTEXT_COLUMNS) & set(fresh.columns)
    model = fitted[name]
    wrong = fresh.assign(public_holiday=True, site_type=5)
    pd.testing.assert_frame_equal(
        forecasts[name], check_table(model.predict(wrong, KAPPAS)), check_exact=True
    )
    right = with_context(fresh)
    pd.testing.assert_frame_equal(
        forecasts[name], check_table(model.predict(right, KAPPAS)), check_exact=True
    )
    # the site name is not an input, the site type is: jpl is a workplace as caltech is
    renamed = fresh.assign(site=fresh["site"].replace({"caltech": "jpl"}))
    assert np.array_equal(
        forecasts[name][OTHER].to_numpy(), model.predict(renamed, KAPPAS)[OTHER].to_numpy()
    )
    moved = fresh.assign(site=fresh["site"].replace({"caltech": "BOD"}))
    assert (model.predict(moved, KAPPAS)["d_hat"].to_numpy() != forecasts[name]["d_hat"]).any()
    with pytest.raises(ValueError, match="no site type"):
        model.predict(fresh.assign(site="elsewhere"), KAPPAS)
    with pytest.raises(KeyError):
        model.predict(fresh.drop(columns=list(TAIL_HISTORY_COLUMNS)), KAPPAS)


def test_the_departure_boosters_split_on_the_made_up_site_type_signal(fitted):
    place = lgbm_ctx.COLUMNS.index("site_type")
    for name in NAMES:
        used = sum(
            booster.feature_importance("split")[place]
            for key, booster in fitted[name].quantiles.boosters.items()
            if key[0] == "departure"
        )
        assert used > 0


# --- 5. no leak ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_label_columns_handed_to_predict_change_nothing(data, fitted, forecasts, name):
    _, fresh = data
    labelled = fresh.assign(**{c: -7.0 * np.arange(len(fresh)) + 3.0 for c in LABEL_COLUMNS})
    pd.testing.assert_frame_equal(
        forecasts[name], fitted[name].predict(labelled, KAPPAS), check_exact=True
    )
    part = fresh.iloc[5:27].reset_index(drop=True)  # a car's forecast does not depend on the others
    whole = forecasts[name]
    keys = set(zip(part["site"], part["session_id"], strict=True))
    kept = [key in keys for key in zip(whole["site"], whole["session_id"], strict=True)]
    pd.testing.assert_frame_equal(
        whole[kept].reset_index(drop=True), fitted[name].predict(part, KAPPAS), check_exact=True
    )


@pytest.mark.parametrize("name", NAMES)
def test_altering_the_outcomes_of_the_cars_forecast_does_not_change_their_forecast(fitted, name):
    last = DAYS - 1
    base = fitted[name].predict(_fresh(_made_up_sites(), last), KAPPAS)
    assert len(base) == 2 * sf.PER_DAY * len(KAPPAS)
    changed = []

    def alter(fakes):
        for k, fake in enumerate(fakes):
            if fake.date == date_of(last):
                fake.plug_out = 10.0 + (0.3 if fake.plug_out > 14.0 else 9.0) + 0.01 * k
                fake.energy = 17.0 + k
                fake.room = 0.5 + 0.001 * k
                fake.rate = 1.5
                changed.append(k)

    altered = _made_up_sites(alter)
    assert len(changed) == 2 * sf.PER_DAY
    after = fitted[name].predict(_fresh(altered, last), KAPPAS)
    pd.testing.assert_frame_equal(base, after, check_exact=True)
    assert after[OTHER].notna().all().all()

    # the alteration is a real one: the same change on an earlier day does change the forecast
    def earlier(fakes):
        for k, fake in enumerate(fakes):
            if fake.date < date_of(last):
                fake.plug_out = 10.0 + (0.3 if fake.plug_out > 14.0 else 9.0) + 0.01 * (k % 7)

    moved = fitted[name].predict(_fresh(_made_up_sites(earlier), last), KAPPAS)
    assert (moved["d_hat"].to_numpy() != base["d_hat"].to_numpy()).any()


# --- 6. and 7. the settings file and `learn_run` -------------------------------------------------


def _settings_file(folder, **changes):
    grid = load_grid()
    document = {
        "about": "made up",
        "settings": Settings(7, 100, 0.05, 200).as_dict(),
        "score": 1.0,
        "grid": {"file": grid.file, "sha256": grid.sha256},
        "results": {"file": "tune.json", "sha256": "1" * 64, "run_folder": "r", "commit": COMMIT},
        **changes,
    }
    path = folder / "tuned.json"
    path.write_text(json.dumps(document))
    return path


MATRIX = SimpleNamespace(rho=0.0, configurations=[SimpleNamespace(kappa=k) for k in KAPPAS])


@pytest.mark.parametrize("module", [lgbm_tuned, lgbm_tuned_surv])
def test_make_model_reads_the_settings_file_and_refuses_a_missing_or_mismatched_one(
    module, tmp_path, monkeypatch
):
    monkeypatch.setattr(lts, "TUNED", tmp_path / "tuned.json")
    with pytest.raises(MatrixError, match="absent"):
        module.make_model(MATRIX)
    _settings_file(tmp_path, grid={"file": "tune-grid.json", "sha256": "0" * 64})
    with pytest.raises(MatrixError, match="another grid"):
        module.make_model(MATRIX)
    _settings_file(tmp_path, settings=Settings(9, 100, 0.05, 200).as_dict())
    with pytest.raises(MatrixError, match="not a candidate"):
        module.make_model(MATRIX)
    _settings_file(tmp_path)
    model = module.make_model(MATRIX)
    assert model.settings == Settings(7, 100, 0.05, 200)
    assert model.fit_units == TRAIN_UNITS
    assert lgbm_ctx.make_model(MATRIX).settings is None


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """Made-up units whose sites are in the committed mappings: the gate units at `nrel` and two
    Norwegian development sites, each training unit at the site of its own name."""
    folder = tmp_path_factory.mktemp("build")
    units = {
        GATE_UNITS[0]: gate_cars(5, "nrel"),
        GATE_UNITS[1]: gate_cars(24, "nrel", early_every=4),
        GATE_UNITS[2]: gate_cars(12, "OSL_1", early_every=3) + gate_cars(12, "BER", early_every=5),
    }
    rows = sum(len(cars) for cars in units.values())
    for j, unit in enumerate(TRAIN_UNITS):
        units[unit] = gate_cars(24, unit, early_every=3 + j)
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)], rows


def _learn_run(build, name, folder, monkeypatch) -> int:
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
    return learn_run.main([*build[0], "--model", name])


@pytest.mark.parametrize("name", ["lgbm_tuned", "lgbm_tuned_surv"])
@pytest.mark.parametrize("fault", ["absent", "another grid"])
def test_learn_run_cannot_run_a_tuned_model_without_its_settings(
    build, tmp_path, monkeypatch, capsys, name, fault
):
    settings = tmp_path / "settings"
    settings.mkdir()
    monkeypatch.setattr(lts, "TUNED", settings / "tuned.json")
    if fault == "another grid":
        _settings_file(settings, grid={"file": "tune-grid.json", "sha256": "0" * 64})
    out = tmp_path / "run"
    out.mkdir()
    assert _learn_run(build, name, out, monkeypatch) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN:") and fault in printed
    assert not list(out.iterdir())


@pytest.mark.parametrize("name", NAMES)
def test_learn_run_scores_the_model_end_to_end(build, tmp_path, monkeypatch, capsys, name):
    settings = tmp_path / "settings"
    settings.mkdir()
    monkeypatch.setattr(lts, "TUNED", settings / "tuned.json")
    if name != "lgbm_ctx":  # lgbm_ctx needs no settings file: none exists for it here
        _settings_file(settings)
    out = tmp_path / "run"
    out.mkdir()
    code = _learn_run(build, name, out, monkeypatch)
    printed = capsys.readouterr().out
    assert code == 0, printed
    results = json.loads((out / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == name
    assert results["fit_units"] == list(TRAIN_UNITS)
    assert not results["failed_configurations"]
    table = read_table(out / learn_run.TABLE_NAME)
    assert len(table) == build[1]  # one level, every day counts
    assert sorted(set(table["unit"])) == sorted(GATE_UNITS)
    assert table["d_hat"].notna().all()
    if name == "lgbm_tuned_surv":
        assert table["survival_h1"].notna().all() and (table["survival_h1"] >= 1.0).all()
    else:
        assert table[SURVIVAL].isna().all().all()
    for unit in GATE_UNITS:
        score = results["units"][unit]["kappas"]["0.1"]
        assert score["cars_with_departure_forecast"] == score["cars"]
        offer = results["units"][unit]["configurations"][PAIR[0]]
        assert offer["oracle_partner"]["name"] == PAIR[1]
