"""`models.lgbm_tail`: the pooled LightGBM of `models.lgbm` plus the ten tail-history columns.

Made-up cars tables: ten users plugged in every day, the even ones leaving within an hour on
three days in ten, the odd ones never; 150 training days and 40 later days.

1. The model is `lgbm_tail`, fitted on the training units only, passes the learn guard, and has the
   fit units, levels, settings and rounds of `lgbm`; every booster is named for the columns
   `FEATURE_COLUMNS` then `TAIL_HISTORY_COLUMNS`.
2. It reads the tail columns by name (a frame without them fails; a changed tail changes the
   forecast) and no label column.
3. `lgbm` and `cqr` do not read them: their forecasts are byte-identical when the tail columns are
   removed or changed, and `lgbm`'s boosters name `FEATURE_COLUMNS` only.
4. The table has the format of `lgbm`'s (checked by `forecast_model.check_table`), no negative
   prediction, no survival value, and two fits give the same forecasts.
5. `learn_run --model lgbm_tail` runs end to end on made-up units (exit 0).
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_cars, write_build
from learn_fake import PAIR, small_gate_units, small_matrix
from test_learn_features import _cars
from truth_tables_fake import FakeCar, date_of

from fleetreserve import gate_run, learn_run, run_verify
from fleetreserve.forecast_model import SURVIVAL_COLUMNS, check_table, read_table
from fleetreserve.learn_features import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    TAIL_HISTORY_COLUMNS,
    build_table,
    feature_frame,
)
from fleetreserve.models import cqr, lgbm, lgbm_tail
from fleetreserve.truth_build import TRAIN_UNITS

KAPPAS = (0.1, 0.3)
TRAIN_DAYS = 150
DAYS = 190
TAIL = list(TAIL_HISTORY_COLUMNS)


def _made_up() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    fakes, users = [], []
    for k in range(DAYS):
        for j in range(10):
            quick = j % 2 == 0 and rng.random() < 0.3
            stay = 0.5 + 0.5 * rng.random() if quick else 6.0 + 2.0 * rng.random()
            fakes.append(
                FakeCar(
                    "S",
                    date_of(k),
                    7.0 + 0.2 * j,
                    10.0 + stay,
                    rate=3.0 + rng.random(),
                    room=1.0 + 2.0 * rng.random(),
                    session=f"S-{k}-{j}",
                    energy=4.0,
                )
            )
            users.append(f"user{j}")
    return _cars(fakes, users)


@pytest.fixture(scope="module")
def data():
    cars = _made_up()
    train = cars[cars["date"] < date_of(TRAIN_DAYS)].reset_index(drop=True)
    table = build_table(cars, rho=0.0)
    fresh = feature_frame(table[table["date"] >= date_of(TRAIN_DAYS)].reset_index(drop=True))
    return train, fresh


@pytest.fixture(scope="module")
def fitted(data):
    train, _ = data
    out = {}
    for module, model in (
        ("lgbm", lgbm.LgbmModel(rho=0.0, kappas=KAPPAS)),
        ("lgbm_tail", lgbm_tail.LgbmTailModel(rho=0.0, kappas=KAPPAS)),
        ("cqr", cqr.CqrModel(rho=0.0, kappas=KAPPAS)),
    ):
        model.fit(train)
        out[module] = model
    return out


def test_the_model_is_pooled_lgbm_on_the_training_units_with_the_tail_columns(fitted):
    model, plain = fitted["lgbm_tail"], fitted["lgbm"]
    assert model.name == "lgbm_tail"
    assert model.fit_units == TRAIN_UNITS == plain.fit_units
    assert learn_run.check_model(model.name, model) is False
    assert model.levels == plain.levels
    assert (*FEATURE_COLUMNS, *TAIL_HISTORY_COLUMNS) == lgbm_tail.COLUMNS
    assert set(model.quantiles.boosters) == set(plain.quantiles.boosters)
    for key, booster in model.quantiles.boosters.items():
        assert booster.feature_name() == list(lgbm_tail.COLUMNS)
        assert plain.quantiles.boosters[key].feature_name() == list(FEATURE_COLUMNS)
        assert booster.params["num_leaves"] == lgbm.PARAMETERS["num_leaves"]
        assert booster.params["alpha"] == key[1]
        assert booster.num_trees() == plain.quantiles.boosters[key].num_trees()


def test_a_frame_without_the_tail_columns_is_refused_and_a_changed_tail_changes_the_forecast(
    data, fitted
):
    _, fresh = data
    model = fitted["lgbm_tail"]
    with pytest.raises(KeyError):
        model.predict(fresh.drop(columns=TAIL), KAPPAS)
    base = model.predict(fresh, KAPPAS)
    changed = fresh.copy()
    changed["history_min_departure_h"] = 0.0
    changed["history_q10_departure_h"] = 0.0
    assert not changed[TAIL].equals(fresh[TAIL])
    after = model.predict(changed, KAPPAS)
    assert (after["d_hat"].to_numpy() != base["d_hat"].to_numpy()).any()
    # no label column is read
    labelled = fresh.assign(**{c: -7.0 * np.arange(len(fresh)) for c in LABEL_COLUMNS})
    pd.testing.assert_frame_equal(base, model.predict(labelled, KAPPAS), check_exact=True)


@pytest.mark.parametrize("name", ["lgbm", "cqr"])
def test_the_older_models_do_not_read_the_tail_columns(data, fitted, name):
    _, fresh = data
    model = fitted[name]
    base = model.predict(fresh, KAPPAS)
    pd.testing.assert_frame_equal(
        base, model.predict(fresh.drop(columns=TAIL), KAPPAS), check_exact=True
    )
    changed = fresh.copy()
    changed[TAIL] = -123.0
    pd.testing.assert_frame_equal(base, model.predict(changed, KAPPAS), check_exact=True)
    for booster in model.quantiles.boosters.values():
        assert booster.feature_name() == list(FEATURE_COLUMNS)


def test_the_departure_boosters_split_on_the_tail_columns_of_a_made_up_tail_signal(fitted):
    used = sum(
        booster.feature_importance("split")[len(FEATURE_COLUMNS) :].sum()
        for key, booster in fitted["lgbm_tail"].quantiles.boosters.items()
        if key[0] == "departure"
    )
    assert used > 0


def test_the_table_has_the_format_of_lgbm_and_two_fits_agree(data, fitted):
    train, fresh = data
    model = fitted["lgbm_tail"]
    table = check_table(model.predict(fresh, KAPPAS))
    assert len(table) == len(fresh) * len(KAPPAS)
    assert list(table.columns) == list(fitted["lgbm"].predict(fresh, KAPPAS).columns)
    numbers = ["d_hat", "r_hat", "e_cap", "e_need", "d_median"]
    assert (table[numbers].to_numpy() >= 0).all()
    assert table[list(SURVIVAL_COLUMNS)].isna().all().all()
    again = lgbm_tail.make_model(_matrix())
    again.fit(train)
    pd.testing.assert_frame_equal(table, again.predict(fresh, KAPPAS), check_exact=True)


def _matrix() -> SimpleNamespace:
    """The two fields of a run matrix `make_model` reads."""
    return SimpleNamespace(rho=0.0, configurations=[SimpleNamespace(kappa=k) for k in KAPPAS])


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    training = {
        unit: gate_cars(24, f"T{j}", early_every=3 + j) for j, unit in enumerate(TRAIN_UNITS)
    }
    run, tables = write_build(folder, {**small_gate_units(), **training})
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]


def test_learn_run_scores_the_model_end_to_end(build, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = learn_run.main([*build, "--model", "lgbm_tail"])
    out = capsys.readouterr().out
    assert code == 0, out
    results = json.loads((tmp_path / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == "lgbm_tail"
    assert results["fit_units"] == list(TRAIN_UNITS)
    assert not results["failed_configurations"]
    table = read_table(tmp_path / learn_run.TABLE_NAME)
    cars = small_gate_units()
    assert len(table) == sum(len(cars[u]) for u in GATE_UNITS)
    assert sorted(set(table["unit"])) == sorted(GATE_UNITS)
    for unit in GATE_UNITS:
        score = results["units"][unit]["kappas"]["0.1"]
        assert score["cars_with_departure_forecast"] == score["cars"]
        offer = results["units"][unit]["configurations"][PAIR[0]]
        assert offer["oracle_partner"]["name"] == PAIR[1]
