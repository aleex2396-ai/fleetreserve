"""The seed ensemble of the tuned neural hazard model (`models.hazard_ens`), on made-up data, with a
made-up tuned settings file (`hazard_fake.use_small_tuned`).

1. Five members, seeds 0 to 4, each with `hazard_tuned`'s settings at its seed.
2. The mean curve equals a recomputation from members fitted separately, and the whole table
   equals the one read from that recomputed curve (departure columns by
   `hazard_curves.departure_table` with the presence rules, energy columns as the plain mean of
   the separately fitted members' columns).
3. On two made-up members with given curves: the bound is the quantile of the mean curve, not the
   mean of the members' quantiles; a survival cell that one member alone leaves NaN (its presence
   probability is under the floor) is a number when the mean curve's presence probability is
   not; the energy columns are the mean of the members' values. On the fitted ensemble, the mean
   of the members' `d_hat` differs from the ensemble's `d_hat` on some cars.
4. With one member (seed 0) the table is `hazard_tuned`'s bit for bit.
5. Two fits give identical tables; the table passes `check_table`.
6. A member whose fit fails raises, and the ensemble then refuses to forecast.
7. Refused without a usable settings file; `learn_run --model hazard_ens` runs end to end on
   made-up units (exit 0).
"""

import hashlib
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

from fleetreserve import gate_run, hazard_curves, hazard_fit, learn_run, run_verify
from fleetreserve import hazard_tune_settings as hts
from fleetreserve.forecast_model import (
    KEY_COLUMNS,
    SURVIVAL_COLUMNS,
    check_table,
    read_table,
    write_table,
)
from fleetreserve.gate_config import MatrixError
from fleetreserve.hazard_data import site_codes
from fleetreserve.learn_features import build_table, with_context
from fleetreserve.models import hazard_ens, hazard_tuned
from fleetreserve.models.hazard import (
    HazardModel,
    SetCurves,
    presence_support_of,
    site_types_of,
)
from fleetreserve.truth_build import TRAIN_UNITS

MATRIX = SimpleNamespace(rho=0.0, configurations=[SimpleNamespace(kappa=k) for k in hf.KAPPAS])
ENERGY = ("r_hat", "e_cap", "e_need")


def written(table: pd.DataFrame, folder) -> bytes:
    path = folder / "table.csv"
    write_table(table, path)
    return path.read_bytes()


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """The ensemble built by `make_model` and `hazard_tuned`, both fitted on the same made-up cars,
    and five `HazardModel`s at the tuned settings and seeds 0 to 4 fitted separately; the later
    days' feature frame. The tests only read them."""
    folder = tmp_path_factory.mktemp("tuned")
    train, fresh, _ = hf.known_data(60, 10)
    with pytest.MonkeyPatch.context() as patch:
        hf.use_small_tuned(patch, folder)
        ensemble = hazard_ens.make_model(MATRIX)
        ensemble.fit(train)
        single = hazard_tuned.make_model(MATRIX)
        single.fit(train)
    separate = []
    for seed in range(5):
        model = HazardModel(rho=0.0, kappas=hf.KAPPAS, settings=hf.SMALL_TUNED.hazard(seed))
        model.fit(train)
        separate.append(model)
    return SimpleNamespace(
        ensemble=ensemble, single=single, separate=separate, train=train, fresh=fresh
    )


# --- 1. members ----------------------------------------------------------------------------------


def test_five_members_with_seeds_zero_to_four_at_hazard_tuneds_settings(fitted):
    ensemble, single = fitted.ensemble, fitted.single
    assert hazard_ens.SEEDS == (0, 1, 2, 3, 4)
    assert ensemble.seeds == hazard_ens.SEEDS and len(ensemble.members) == 5
    assert ensemble.name == "hazard_ens" and ensemble.fit_units == TRAIN_UNITS
    assert ensemble.tuned == hf.SMALL_TUNED
    for seed, member in zip(range(5), ensemble.members, strict=True):
        assert member.settings == replace(single.settings, seed=seed)
        assert type(member) is HazardModel and member.net is not None


# --- 2. the mean curve and the table read from it ------------------------------------------------


def test_the_table_is_read_from_the_mean_of_separately_fitted_members_curves(fitted, tmp_path):
    torch.set_num_threads(1)
    ensemble, separate, fresh = fitted.ensemble, fitted.separate, fitted.fresh
    mean, _ = ensemble.mean_curves(fresh)
    first = separate[0]
    x, sets = first.inputs(fresh)
    codes = site_codes(sets, first.fit_sites)
    kinds = site_types_of(with_context(fresh), sets)
    support = presence_support_of(with_context(build_table(fitted.train, rho=0.0)))
    settings = first.settings
    times = hazard_curves.bin_starts(settings)
    n = len(fresh)
    expected = {
        k: {c: np.full(n, np.nan) for c in ("d_hat", "d_median", *SURVIVAL_COLUMNS)}
        for k in hf.KAPPAS
    }
    assert len(mean.rows) == sets.n_sets
    for k in range(sets.n_sets):
        rows = sets.members(k)
        total = 0.0
        for model in separate:
            total = total + model.curves(x[rows], int(codes[k]))[0]
        curve = total / len(separate)
        np.testing.assert_array_equal(mean.rows[k], rows)
        np.testing.assert_array_equal(mean.survival[k], curve)
        columns = hazard_curves.departure_table(
            times,
            curve,
            hf.KAPPAS,
            support[int(kinds[k])] >= settings.min_presence_support,
            settings.min_presence_probability,
        )
        for kappa, part in columns.items():
            for name, values in part.items():
                expected[kappa][name][rows] = values
    tables = [model.predict(fresh, hf.KAPPAS) for model in separate]
    table = ensemble.predict(fresh, hf.KAPPAS)
    for kappa in hf.KAPPAS:
        at = (table["kappa"] == kappa).to_numpy()
        for name, values in expected[kappa].items():
            np.testing.assert_array_equal(table.loc[at, name].to_numpy(), values)
        for name in ENERGY:
            total = 0.0
            for other in tables:
                total = total + other.loc[(other["kappa"] == kappa).to_numpy(), name].to_numpy()
            np.testing.assert_array_equal(table.loc[at, name].to_numpy(), total / len(tables))
    assert table["survival_h1"].notna().any() and table[list(SURVIVAL_COLUMNS)].isna().any().any()


# --- 3. quantile of the mean, presence from the mean, mean energy --------------------------------


def _given(curves: list[np.ndarray], energy_means: list[float]):
    """A two-member ensemble whose members return the given curves for a frame of one car."""
    ensemble = hazard_ens.HazardEnsembleModel(
        rho=0.0, kappas=hf.KAPPAS, tuned=hf.SMALL_TUNED, seeds=(0, 1)
    )
    for member, curve, level in zip(ensemble.members, curves, energy_means, strict=True):
        member.net = object()
        member.presence_support = {kind: np.full(8, 10_000) for kind in (0, 1)}
        member.energy_mean = np.full(3, level)
        member.energy_std = np.ones(3)
        found = SetCurves(
            1, [np.array([0])], np.array([1]), [curve[None]], np.zeros((1, len(member.outputs)))
        )
        member.set_curves = lambda frame, found=found: found
    ensemble.fitted = True
    frame = pd.DataFrame({c: ["x"] for c in KEY_COLUMNS})
    return ensemble, frame


def test_the_bound_is_the_quantile_of_the_mean_curve_and_presence_is_read_from_it():
    times = hazard_curves.bin_starts(hf.SMALL_TUNED.hazard())
    early = np.zeros(len(times))  # every car of this member leaves in the bin that starts at τ
    late = np.where(times < 10.0, 1.0, 0.0)  # leaves in the bin that starts 10 h after τ
    ensemble, frame = _given([early, late], [1.0, 3.0])
    table = ensemble.predict(frame, [0.1])
    alone = [
        hazard_curves.departure_table(times, curve[None], [0.1], None, 1e-3)[0.1]
        for curve in (early, late)
    ]
    assert [a["d_hat"][0] for a in alone] == [0.0, 10.0]  # mean of the quantiles: 5
    assert table["d_hat"].tolist() == [0.0]  # the quantile of the mean curve
    assert np.isnan(alone[0]["survival_h1"][0]) and alone[1]["survival_h1"][0] == 10.0
    assert table["survival_h1"].tolist() == [10.0]  # present at hour 1 with probability 0.5
    assert table["r_hat"].tolist() == [2.0] and table["e_cap"].tolist() == [2.0]
    assert table["e_need"].tolist() == [2.0]
    floor = replace(hf.SMALL_TUNED.hazard(), min_presence_probability=0.6)
    for member in ensemble.members:
        member.settings = floor
    assert np.isnan(ensemble.predict(frame, [0.1])["survival_h1"][0])  # 0.5 is under 0.6


def test_on_the_fitted_ensemble_the_mean_of_member_bounds_is_not_the_bound(fitted):
    ensemble, fresh = fitted.ensemble, fitted.fresh
    table = ensemble.predict(fresh, hf.KAPPAS)
    member_bounds = np.mean(
        [m.predict(fresh, hf.KAPPAS)["d_hat"].to_numpy() for m in ensemble.members], axis=0
    )
    assert (member_bounds != table["d_hat"].to_numpy()).any()
    times = hazard_curves.bin_starts(ensemble.members[0].settings)
    assert np.isin(table["d_hat"].dropna().to_numpy(), times).all()
    assert not np.isin(member_bounds[~np.isnan(member_bounds)], times).all()


# --- 4. one member is hazard_tuned ---------------------------------------------------------------


def test_one_member_with_seed_zero_is_hazard_tuned_bit_for_bit(fitted, tmp_path):
    one = hazard_ens.HazardEnsembleModel(
        rho=0.0, kappas=hf.KAPPAS, tuned=hf.SMALL_TUNED, seeds=(0,)
    )
    one.fit(fitted.train)
    table = one.predict(fitted.fresh, hf.KAPPAS)
    single = fitted.single.predict(fitted.fresh, hf.KAPPAS)
    pd.testing.assert_frame_equal(table, single, check_exact=True)
    assert written(table, tmp_path) == written(single, tmp_path)


# --- 5. determinism and the table ----------------------------------------------------------------


def test_two_fits_give_identical_tables_and_the_table_passes_check_table(fitted, tmp_path):
    again = hazard_ens.HazardEnsembleModel(rho=0.0, kappas=hf.KAPPAS, tuned=hf.SMALL_TUNED)
    again.fit(fitted.train)
    table = fitted.ensemble.predict(fitted.fresh, hf.KAPPAS)
    other = again.predict(fitted.fresh, hf.KAPPAS)
    assert (
        hashlib.sha256(written(table, tmp_path)).digest()
        == hashlib.sha256(written(other, tmp_path)).digest()
    )
    checked = check_table(table)
    assert len(checked) == len(fitted.fresh) * len(hf.KAPPAS)
    assert checked["d_hat"].notna().all()
    assert fitted.ensemble.predict(fitted.fresh.iloc[:0], hf.KAPPAS).empty


# --- 6. a failing member -------------------------------------------------------------------------


def test_a_member_whose_fit_fails_raises_and_the_ensemble_then_refuses_to_forecast(
    fitted, monkeypatch
):
    original = hazard_fit.fit
    seen = []

    def failing(data, settings, *args, **kwargs):
        seen.append(settings.seed)
        if settings.seed == 2:
            raise RuntimeError("made-up failure of the member with seed 2")
        return original(data, settings, *args, **kwargs)

    monkeypatch.setattr(hazard_fit, "fit", failing)
    ensemble = hazard_ens.HazardEnsembleModel(rho=0.0, kappas=hf.KAPPAS, tuned=hf.SMALL_TUNED)
    with pytest.raises(RuntimeError, match="seed 2"):
        ensemble.fit(fitted.train)
    assert seen == [0, 1, 2]
    with pytest.raises(ValueError, match="predict before fit"):
        ensemble.predict(fitted.fresh, hf.KAPPAS)


# --- 7. refusals and end to end ------------------------------------------------------------------


def test_the_ensemble_is_refused_without_usable_settings(tmp_path, monkeypatch, capsys):
    hf.use_small_tuned(monkeypatch, tmp_path)
    assert hazard_ens.make_model(MATRIX).tuned == hf.SMALL_TUNED
    monkeypatch.setattr(hts, "TUNED", tmp_path / "absent.json")
    with pytest.raises(MatrixError, match="absent"):
        hazard_ens.make_model(MATRIX)
    assert learn_run.main(["--model", "hazard_ens", "--tables", str(tmp_path)]) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN:") and "absent" in printed
    with pytest.raises(ValueError, match="settings of the tuned settings file"):
        hazard_ens.HazardEnsembleModel(rho=0.0, kappas=hf.KAPPAS, tuned=None)
    for seeds in ((), (0, 0)):
        with pytest.raises(ValueError, match="distinct"):
            hazard_ens.HazardEnsembleModel(
                rho=0.0, kappas=hf.KAPPAS, tuned=hf.SMALL_TUNED, seeds=seeds
            )


def test_learn_run_scores_the_ensemble_end_to_end(tmp_path_factory, tmp_path, monkeypatch, capsys):
    units = _units()
    run, tables = write_build(tmp_path_factory.mktemp("build"), units)
    rows = sum(len(units[u]) for u in GATE_UNITS)
    hf.use_small_tuned(monkeypatch, tmp_path)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = ["--tables", str(tables), "--truth-run", str(run), "--matrix", str(matrix)]
    code = learn_run.main([*arguments, "--model", "hazard_ens"])
    printed = capsys.readouterr().out
    assert code == 0, printed
    results = json.loads((out / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == "hazard_ens" and results["fit_units"] == list(TRAIN_UNITS)
    assert not results["failed_configurations"]
    table = read_table(out / learn_run.TABLE_NAME)
    assert len(table) == rows and table["d_hat"].notna().all()
