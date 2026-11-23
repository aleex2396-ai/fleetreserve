"""The tuning command of the neural hazard model (`hazard_tune`), on made-up data.

Tiny hand-made tables (three sites, a few rows, the fit-and-forecast step replaced by a constant
bound of 4 hours or by a seed-dependent one) and made-up training units (`hazard_tune_fake`) with
real fits on a tiny grid.

1. Units: a unit that is not a training unit is refused before anything is read (exit 2); a site
   in two units ends the command with exit 1.
2. Folds: the network is fitted on the rows of the other sites only; the held-out site is
   forecast from a frame with no label column and gets the vector of a site not seen in fitting;
   changing the held-out site's labels changes neither the fitted weights nor the forecast.
3. Score: on the tiny table the score of a candidate equals the pinball losses computed by hand,
   as a mean over the sites and over the seeds, with the spread over seeds, the per-type means and
   the miss rates by hand; the same score comes out of an independent route (`HazardModel.fit` on
   the other units' cars tables, the frame of `learn_run`); the reference entry equals
   `learn_tune.fold_scores` for the same feature set and rows.
4. The command on made-up training units: exit 0, the estimate line first, the three files, a
   result for every candidate, every part and the reference, the present settings named; the
   results do not depend on `--workers`.
"""

import contextlib
import io
import json
import os
import sys

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import pytest
import torch
from gate_fake import COMMIT, write_build
from hazard_tune_fake import DAYS, TINY, frames, tiny_grid
from learn_fake import small_matrix
from quantile_fake import cars

from fleetreserve import gate_run, hazard_data, hazard_tune, learn_tune, run_verify
from fleetreserve import hazard_tune_grid as grids
from fleetreserve.forecast_empirical import TUNING_KAPPAS
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.hazard_tune_grid import PRESENT, Setting, choose, load_grid
from fleetreserve.learn_eval import pinball_loss
from fleetreserve.learn_features import LABEL_COLUMNS, build_table, feature_frame
from fleetreserve.learn_tune_grid import PRESENT as LGBM_PRESENT
from fleetreserve.models.hazard import HazardModel, presence_support_of

SITES = {"caltech": 1, "jpl": 1, "OSL_S": 0}  # site -> site type
CODES = {"caltech": 1.0, "jpl": 2.0, "OSL_S": 3.0}
ROWS = [  # site, counting, departure label
    ("caltech", True, 2.0),
    ("caltech", True, 6.0),
    ("caltech", False, 100.0),
    ("jpl", True, 4.0),
    ("jpl", True, 5.0),
    ("jpl", True, 9.0),
    ("OSL_S", True, 1.0),
    ("OSL_S", False, 50.0),
]
BY_HAND = {"caltech": 4.0, "jpl": 1.3, "OSL_S": 10.05}  # as in the test of `learn_tune`


def _bytes(frame: pd.DataFrame) -> bytes:
    """The bytes of the number columns of a forecast table."""
    return b"".join(frame[c].to_numpy().tobytes() for c in frame if frame[c].dtype.kind in "fi")


def _tiny() -> pd.DataFrame:
    """The hand-made table: every feature 0 except `weekday`, which carries the site's code."""
    table = pd.DataFrame(
        {c: 0.0 for c in learn_tune.FEATURE_SETS[learn_tune.TUNED_SET]}, index=range(len(ROWS))
    )
    table["site"] = [site for site, _, _ in ROWS]
    table["unit"] = table["site"]
    table["counting"] = [counting for _, counting, _ in ROWS]
    table[learn_tune.LABEL] = [label for _, _, label in ROWS]
    table["weekday"] = [CODES[site] for site, _, _ in ROWS]
    table["site_type"] = [SITES[site] for site, _, _ in ROWS]
    return table


@pytest.fixture
def constant_bound(monkeypatch):
    """The fit and forecast replaced by a bound of 4 hours for every row, the booster of the
    reference too; the calls are kept."""
    calls = []

    def fake(fold, settings, epochs):
        calls.append({"site": fold.site, "settings": settings, "epochs": tuple(epochs)})
        return {
            e: {float(k): np.full(fold.scored_rows, 4.0) for k in TUNING_KAPPAS} for e in epochs
        }

    monkeypatch.setattr(hazard_tune, "forecast_bounds", fake)
    monkeypatch.setattr(
        learn_tune,
        "_quantile",
        lambda x, y, x_new, columns, parameters, rounds: np.full(len(x_new), 4.0),
    )
    return calls


# --- 1. units ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "units", [("nrel-from-may-2019",), ("caltech", "OSL_1"), ("OSL_T",), ("no-such-unit",)]
)
def test_a_unit_that_is_not_a_training_unit_is_refused_before_anything_is_read(
    units, tmp_path, capsys
):
    missing = tmp_path / "no-such-folder"
    code = hazard_tune.main(
        ["--units", *units, "--tables", str(missing), "--truth-run", str(missing)]
    )
    assert code == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN:") and "training units" in printed
    with pytest.raises(learn_tune.NotTraining):
        learn_tune.tune_table({u: None for u in units}, rho=0.0)


def test_a_unit_named_twice_and_a_bad_worker_count_cannot_run(tmp_path, capsys):
    assert hazard_tune.main(["--units", "caltech", "caltech"]) == 2
    assert "once" in capsys.readouterr().out
    assert hazard_tune.main(["--workers", "0"]) == 2
    assert "--workers" in capsys.readouterr().out
    assert hazard_tune.main(["--grid", str(tmp_path / "none.json")]) == 2
    assert "grid file" in capsys.readouterr().out


def _command(tmp_path, units: dict, grid=None) -> list[str]:
    run, tables = write_build(tmp_path, units)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    grid = grid if grid is not None else tiny_grid(tmp_path)
    return [
        *("--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)),
        *("--grid", str(grid), "--units", *units),
    ]


def _patched_run(arguments, folder, workers: int) -> tuple[int, str]:
    """`hazard_tune.main` with the commit checks of the made-up build replaced; (exit, output)."""
    folder.mkdir(parents=True, exist_ok=True)
    out = io.StringIO()
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(out):
        patch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        patch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        patch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        code = hazard_tune.main([*arguments, "--workers", str(workers)])
    return code, out.getvalue()


def test_a_site_in_two_units_ends_the_command_with_exit_1(tmp_path):
    units = {"caltech": cars(6, 0, site="caltech"), "jpl": cars(6, 1, site="caltech")}
    code, printed = _patched_run(_command(tmp_path, units), tmp_path / "run", 1)
    assert code == 1 and "two units" in printed
    assert not list((tmp_path / "run").iterdir())


def test_an_unknown_site_ends_the_command_with_exit_1(tmp_path):
    units = {"caltech": cars(6, 0, site="caltech"), "jpl": cars(6, 1, site="elsewhere")}
    code, printed = _patched_run(_command(tmp_path, units), tmp_path / "run", 1)
    assert code == 1 and "no site type given" in printed
    assert not list((tmp_path / "run").iterdir())


def test_the_command_cannot_run_without_a_verified_truth_build(tmp_path, capsys):
    units = {"caltech": cars(6, 0, site="caltech"), "jpl": cars(6, 1, site="jpl")}
    arguments = _command(tmp_path, units)
    arguments[arguments.index("--truth-run") + 1] = str(tmp_path / "none")
    assert hazard_tune.main(arguments) == 2
    assert capsys.readouterr().out.startswith("COULD NOT RUN:")


# --- 2. folds ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    return learn_tune.tune_table(frames(tmp_path_factory.mktemp("units")), rho=0.0)


def test_a_fold_is_the_other_sites_rows_and_a_frame_of_the_held_out_sites_counting_rows(table):
    fold = hazard_tune.prepare_fold(table, "jpl")
    assert set(fold.fit_table["site"]) == {"caltech", "OSL_S"}
    assert fold.fit_rows == len(fold.fit_table) == 12 * (30 + 22)
    assert fold.scored_rows == len(fold.frame) == len(fold.truth) == 12 * 14
    assert set(fold.frame["site"]) == {"jpl"} and fold.frame["counting"].all()
    assert not set(LABEL_COLUMNS) & set(fold.frame.columns)
    held = table[(table["site"] == "jpl") & table["counting"]]
    assert fold.truth.tolist() == held[learn_tune.LABEL].tolist()


def test_a_site_with_no_counting_row_cannot_be_scored(table):
    none = table.assign(counting=table["counting"] & (table["site"] != "jpl"))
    with pytest.raises(ValueError, match="no row of a counting day"):
        hazard_tune.prepare_fold(none, "jpl")


def test_the_network_is_fitted_on_the_other_sites_and_the_held_out_site_is_an_unseen_site(
    table, monkeypatch
):
    fitted, forecast = [], []
    original_fit, original_predict = hazard_tune.TuneModel.fit_prepared, HazardModel.predict

    def fit_spy(self, prepared, checkpoints=None):
        fitted.append(prepared)
        return original_fit(self, prepared, checkpoints)

    def predict_spy(self, frame, kappas):
        forecast.append((frame, self.fit_sites, self.inputs(frame)[1]))
        return original_predict(self, frame, kappas)

    monkeypatch.setattr(hazard_tune.TuneModel, "fit_prepared", fit_spy)
    monkeypatch.setattr(HazardModel, "predict", predict_spy)
    fold = hazard_tune.prepare_fold(table, "jpl")
    hazard_tune.forecast_bounds(fold, Setting(16, 0.003, 1.0, 2).hazard(0), (1, 2))
    (prepared,) = fitted
    assert prepared.fit_sites == ("OSL_S", "caltech")
    sets = hazard_data.make_sets(fold.fit_table)
    assert {key[1] for key in sets.keys} == {"OSL_S", "caltech"}
    assert prepared.data.x.shape[0] == fold.fit_rows
    assert len(forecast) == 2  # one forecast at each checkpoint
    for frame, fit_sites, forecast_sets in forecast:
        assert frame is fold.frame and not set(LABEL_COLUMNS) & set(frame.columns)
        codes = hazard_data.site_codes(forecast_sets, fit_sites)
        assert len(codes) > 0 and (codes == hazard_data.UNKNOWN_SITE).all()


def test_the_held_out_sites_labels_cannot_reach_its_forecast_or_the_fitted_weights(table):
    altered = table.copy()
    held = (altered["site"] == "jpl").to_numpy()
    for column in LABEL_COLUMNS:
        altered.loc[held, column] = altered.loc[held, column] * 3.0 + 1.0
    settings = Setting(16, 0.003, 1.0, 3).hazard(0)
    results = []
    for source in (table, altered):
        fold = hazard_tune.prepare_fold(source, "jpl")
        prepared = hazard_tune.prepare(fold, settings)
        model = hazard_tune.TuneModel(kappas=TUNING_KAPPAS, settings=settings)
        model.attach(fold)
        model.fit_prepared(prepared)
        results.append((fold, model, model.predict(fold.frame, TUNING_KAPPAS)))
    (first, model_a, table_a), (second, model_b, table_b) = results
    assert not np.array_equal(first.truth, second.truth)  # the labels did change
    assert table_a.equals(table_b) and _bytes(table_a) == _bytes(table_b)
    for (name, a), (_, b) in zip(
        model_a.net.state_dict().items(), model_b.net.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
    assert model_a.history == model_b.history


def test_a_fold_forecast_is_the_forecast_of_the_models_own_fit_and_predict(table, tmp_path):
    """`TuneModel` (shared inputs, fit from the fold's table) and `HazardModel.fit` on the other
    units' cars tables with the frame `learn_run` builds give the same table, byte for byte, the
    survival columns and the support counts included."""
    cars_tables = frames(tmp_path)
    settings = Setting(16, 0.003, 1.0, 3).hazard(1)
    fold = hazard_tune.prepare_fold(table, "jpl")
    found = hazard_tune.forecast_bounds(fold, settings, (3,))[3]
    prepared = hazard_tune.prepare(fold, settings)
    tuned = hazard_tune.TuneModel(kappas=TUNING_KAPPAS, settings=settings)
    tuned.attach(fold)
    tuned.fit_prepared(prepared)
    plain = HazardModel(rho=0.0, kappas=TUNING_KAPPAS, settings=settings)
    plain.fit(pd.concat([cars_tables["caltech"], cars_tables["OSL_S"]], ignore_index=True))
    jpl = feature_frame(build_table(cars_tables["jpl"], rho=0.0))
    jpl = jpl[jpl["counting"]].reset_index(drop=True)
    expected = plain.predict(jpl, TUNING_KAPPAS)
    assert sorted(tuned.presence_support) == sorted(plain.presence_support) == [0, 1]
    for kind in (0, 1):  # both site types have support, and the counts are the same
        assert tuned.presence_support[kind].tolist() == plain.presence_support[kind].tolist()
        assert tuned.presence_support[kind].min() > 0
    from_tuned = tuned.predict(fold.frame, TUNING_KAPPAS)
    assert from_tuned.equals(expected) and _bytes(from_tuned) == _bytes(expected)
    assert expected["survival_h1"].notna().any()
    for kappa in TUNING_KAPPAS:
        bound = expected.loc[expected["kappa"] == kappa, "d_hat"].to_numpy()
        assert found[float(kappa)].tobytes() == np.maximum(bound, 0.0).tobytes()
    assert len(jpl) == fold.scored_rows


def test_the_fit_support_counts_are_those_of_the_models_own_function_by_site_type(table):
    fold = hazard_tune.prepare_fold(table, "jpl")  # fit rows: caltech (a workplace), OSL_S (not)
    labels = fold.fit_table[learn_tune.LABEL].to_numpy()
    kinds = fold.fit_table["site_type"].to_numpy()
    expected = {
        kind: [int(((labels >= hour) & (kinds == kind)).sum()) for hour in range(1, 9)]
        for kind in (0, 1)
    }
    assert expected[0] != expected[1] and min(expected[0] + expected[1]) > 0
    found = presence_support_of(fold.fit_table)
    assert {kind: counts.tolist() for kind, counts in found.items()} == expected
    prepared = hazard_tune.prepare(fold, Setting(16, 0.003, 1.0, 1).hazard(0))
    assert {k: v.tolist() for k, v in prepared.presence_support.items()} == expected


# --- 3. the score --------------------------------------------------------------------------------


def test_the_score_is_the_by_hand_pinball_loss_and_a_mean_over_sites_and_seeds(
    tmp_path, constant_bound
):
    """Bound 4 h for every car; levels 0.05, 0.1, 0.2, 0.3 (their sum 0.65). Per level k:
    caltech, departures 2 and 6: (2(1 - k) + 2k) / 2 = 1, summed over the levels 4.0 (its
    non-counting row at 100 is not scored); jpl, departures 4, 5, 9: (0 + k + 5k) / 3 = 2k,
    summed 1.3; OSL_S, departure 1: 3(1 - k), summed 3 (4 - 0.65) = 10.05. The score is the mean
    of the three, 15.35 / 3, whatever the seed: the mean over the six cars would be
    (2 x 4.0 + 3 x 1.3 + 10.05) / 6."""
    results = hazard_tune.tune(_tiny(), load_grid(tiny_grid(tmp_path)))
    assert len(results["candidates"]) == 4
    for candidate in [*results["candidates"], results["present"], *results["parts"].values()]:
        for site, expected in BY_HAND.items():
            assert candidate["sites"][site]["score"] == pytest.approx(expected, abs=1e-12)
            assert candidate["sites"][site]["score_by_seed"] == pytest.approx(
                {"0": expected, "1": expected}
            )
        assert candidate["sites"]["caltech"]["pinball_loss"] == pytest.approx(
            {repr(k): 1.0 for k in TUNING_KAPPAS}
        )
        assert candidate["score"] == pytest.approx(15.35 / 3, abs=1e-12)
        assert abs(candidate["score"] - (2 * 4.0 + 3 * 1.3 + 10.05) / 6) > 1.0
        assert candidate["score_by_seed"] == pytest.approx({"0": 15.35 / 3, "1": 15.35 / 3})
        assert candidate["seed_spread"] == pytest.approx(
            {"min": 15.35 / 3, "max": 15.35 / 3, "std": 0.0}
        )
        assert candidate["score_by_site_type"] == pytest.approx({"workplace": 2.65, "other": 10.05})
        assert candidate["miss_rate"] == pytest.approx({repr(k): 0.5 for k in TUNING_KAPPAS})
        assert candidate["sites"]["caltech"]["miss_rate"] == {repr(k): 0.5 for k in TUNING_KAPPAS}
        assert candidate["sites"]["caltech"]["scored_rows"] == 2
        assert candidate["sites"]["caltech"]["fit_rows"] == 5
        assert candidate["sites"]["OSL_S"]["fit_rows"] == 6
    assert results["kappas"] == [0.05, 0.1, 0.2, 0.3] and results["seeds"] == [0, 1]
    assert results["sites"]["caltech"] == {
        "unit": "caltech",
        "site_type": "workplace",
        "rows": 3,
        "scored_rows": 2,
    }
    # equal scores: the tie goes to the fewer epochs, then the larger divergence weight
    assert results["chosen"]["settings"] == Setting(32, 0.003, 1.0, 3).as_dict()
    assert results["present"]["settings"] == PRESENT.as_dict()
    assert results["present"]["rank"] == 1 == results["chosen"]["rank"]
    assert all(
        part["score_minus_present"] == pytest.approx(0.0) for part in results["parts"].values()
    )
    # the reference is the booster's constant bound on the same folds: the same by-hand scores
    assert results["reference"]["score"] == pytest.approx(15.35 / 3, abs=1e-12)


def test_each_task_asks_for_one_fit_with_the_epochs_of_the_grid_or_the_present_settings(
    tmp_path, constant_bound
):
    grid = load_grid(tiny_grid(tmp_path))
    hazard_tune.tune(_tiny(), grid)
    tasks = hazard_tune.tasks_of(grid, list(SITES))
    neural = [t for t in tasks if t.kind != "reference"]
    assert len(neural) == len(constant_bound) == (2 + 5) * 2 * 3
    assert [t.kind for t in tasks].count("reference") == 3
    for task, call in zip(neural, constant_bound, strict=True):
        assert call["site"] == task.site and call["settings"].seed == task.seed
        if task.kind == "candidate":
            assert call["epochs"] == (3, 30) and call["settings"].epochs == 30
            assert (call["settings"].width, call["settings"].kl_weight) == (32, task.fit[2])
        else:
            assert call["epochs"] == (30,) == (PRESENT.epochs,)
            assert call["settings"] == PRESENT.hazard(task.seed, **hazard_tune.PARTS[task.name])


def test_the_score_by_seed_and_its_spread_are_by_hand(tmp_path, monkeypatch):
    """The bound is 4 + seed hours (seeds 0 and 1) at epochs 30 and 5 + seed at epochs 3. At
    epochs 30, caltech (departures 2 and 6) at bound b and level k loses
    ((1 - k)(b - 2) + max(k (6 - b), (1 - k)(b - 6))) / 2: written out below for b = 4, 5."""

    def fake(fold, settings, epochs):
        return {
            e: {
                float(k): np.full(fold.scored_rows, (4.0 if e == 30 else 5.0) + settings.seed)
                for k in TUNING_KAPPAS
            }
            for e in epochs
        }

    monkeypatch.setattr(hazard_tune, "forecast_bounds", fake)
    monkeypatch.setattr(
        learn_tune,
        "_quantile",
        lambda x, y, x_new, columns, parameters, rounds: np.full(len(x_new), 4.0),
    )

    def caltech(bound: float) -> float:
        total = 0.0
        for k in TUNING_KAPPAS:
            first = (1 - k) * (bound - 2.0)  # the car leaving at 2 is below the bound
            second = k * (6.0 - bound) if bound < 6.0 else (1 - k) * (bound - 6.0)
            total += (first + second) / 2
        return total

    grid = load_grid(tiny_grid(tmp_path))
    results = hazard_tune.tune(_tiny(), grid)
    late = next(c for c in results["candidates"] if c["settings"] == PRESENT.as_dict())
    early = next(c for c in results["candidates"] if c["settings"]["epochs"] == 3)
    per_seed = {"0": caltech(4.0), "1": caltech(5.0)}
    assert late["sites"]["caltech"]["score_by_seed"] == pytest.approx(per_seed)
    assert late["sites"]["caltech"]["score"] == pytest.approx((caltech(4.0) + caltech(5.0)) / 2)
    assert early["sites"]["caltech"]["score_by_seed"] == pytest.approx(
        {"0": caltech(5.0), "1": caltech(6.0)}
    )
    # jpl (departures 4, 5, 9): no car leaves before a bound of 4, one before a bound of 5
    assert late["sites"]["jpl"]["miss_rate"]["0.05"] == pytest.approx(1 / 6)
    for entry in (late, early):
        scores = list(entry["score_by_seed"].values())
        spread = entry["seed_spread"]
        assert spread["min"] == min(scores) and spread["max"] == max(scores)
        assert spread["std"] == pytest.approx(abs(scores[0] - scores[1]) / 2)
        assert spread["std"] > 0
        assert entry["score"] == pytest.approx(np.mean(scores))
    assert late["score"] != early["score"]


def test_the_chosen_candidate_is_the_one_the_choice_rule_picks(tmp_path, monkeypatch):
    """A bound of 4 hours with the divergence weight 0.1 and of 8 with the weight 1, at every
    checkpoint: the weight 0.1 scores lower, and its two epoch values tie, so the fewer epochs
    are chosen; the present settings have the weight 1 and rank behind both."""

    def fake(fold, settings, epochs):
        bound = 4.0 if settings.kl_weight == 0.1 else 8.0
        return {
            e: {float(k): np.full(fold.scored_rows, bound) for k in TUNING_KAPPAS} for e in epochs
        }

    monkeypatch.setattr(hazard_tune, "forecast_bounds", fake)
    monkeypatch.setattr(
        learn_tune,
        "_quantile",
        lambda x, y, x_new, columns, parameters, rounds: np.full(len(x_new), 4.0),
    )
    grid = load_grid(tiny_grid(tmp_path))
    results = hazard_tune.tune(_tiny(), grid)
    scored = [(Setting(**c["settings"]), c["score"]) for c in results["candidates"]]
    assert [s for s, _ in scored] == list(grid.candidates)
    assert results["chosen"]["settings"] == choose(scored).as_dict()
    assert results["chosen"]["settings"] == Setting(32, 0.003, 0.1, 3).as_dict()
    ranks = {Setting(**c["settings"]): c["rank"] for c in results["candidates"]}
    assert ranks == {s: grids.rank(scored, s) for s, _ in scored}
    assert ranks[Setting(32, 0.003, 0.1, 3)] == ranks[Setting(32, 0.003, 0.1, 30)] == 1
    assert ranks[PRESENT] == 3 == results["present"]["rank"]
    assert results["chosen"]["rank"] == 1


# --- 4. the command ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {unit: cars(n, seed, site=unit) for seed, (unit, n) in enumerate(DAYS.items())}
    return _command(folder, units)


@pytest.fixture(scope="module")
def one_worker(build, tmp_path_factory):
    folder = tmp_path_factory.mktemp("one")
    code, printed = _patched_run(build, folder / "run", 1)
    return folder / "run", code, printed


@pytest.fixture(scope="module")
def two_workers(build, one_worker, tmp_path_factory):
    folder = tmp_path_factory.mktemp("two")
    code, printed = _patched_run(build, folder / "run", 2)
    return folder / "run", code, printed


def test_the_command_scores_every_candidate_part_and_the_reference(one_worker, build, table):
    folder, code, printed = one_worker
    assert code == 0, printed
    assert printed.startswith("Estimate: 45 tasks, 42 network fits (12 of candidates, 30 of ")
    assert "and 12 boosting fits of the reference; about " in printed.splitlines()[0]
    results = json.loads((folder / hazard_tune.RESULTS_NAME).read_text())
    summary = (folder / hazard_tune.SUMMARY_NAME).read_text()
    assert printed.endswith(summary) and summary.startswith("Tuning of the neural hazard")
    timing = json.loads((folder / hazard_tune.TIMING_NAME).read_text())
    assert timing["workers"] == 1 and timing["total_seconds"] > 0
    grid = load_grid(build[build.index("--grid") + 1])
    assert results["grid"] == {
        "file": "grid.json",
        "sha256": grid.sha256,
        "candidates": 4,
        "fitted_candidates": 2,
        "epochs": [3, 30],
    }
    assert results["fit_units"] == ["caltech", "jpl", "OSL_S"]
    assert results["rows"] == len(table) == 12 * (30 + 14 + 22)
    assert sorted(results["sites"]) == ["OSL_S", "caltech", "jpl"]
    assert results["inputs"]["truth_run"] and results["matrix"]["rho"] == 0.0
    assert results["seeds"] == [0, 1] and results["choice_rule"] == grids.CHOICE_RULE
    assert [Setting(**c["settings"]) for c in results["candidates"]] == list(grid.candidates)
    scored = []
    for candidate in results["candidates"]:
        sites = candidate["sites"]
        assert sorted(sites) == ["OSL_S", "caltech", "jpl"]
        assert candidate["score"] == pytest.approx(np.mean([s["score"] for s in sites.values()]))
        for site in sites.values():
            assert len(site["score_by_seed"]) == 2 and site["score"] > 0
            assert site["score"] == pytest.approx(np.mean(list(site["score_by_seed"].values())))
        spread = candidate["seed_spread"]
        assert spread["min"] <= candidate["score"] <= spread["max"]
        assert sites["jpl"]["fit_rows"] == 12 * (30 + 22)
        assert sites["jpl"]["scored_rows"] == 12 * 14
        scored.append((Setting(**candidate["settings"]), candidate["score"]))
    assert results["chosen"]["settings"] == choose(scored).as_dict()
    assert results["present"]["settings"] == PRESENT.as_dict()
    assert results["present"]["score"] == dict(scored)[PRESENT]
    assert results["present"]["rank"] == grids.rank(scored, PRESENT)
    assert sorted(results["parts"]) == sorted(hazard_tune.PARTS)
    for name, part in results["parts"].items():
        assert part["score"] == pytest.approx(np.mean([s["score"] for s in part["sites"].values()]))
        assert part["score_minus_present"] == pytest.approx(
            part["score"] - results["present"]["score"]
        )
        assert part["score_minus_present"] != 0.0, name


TABLES_READ = (
    "Tables read: caltech-cars.csv, caltech-days.csv, jpl-cars.csv, jpl-days.csv, "
    "OSL_S-cars.csv, OSL_S-days.csv, besides the recorded truth build's summary and manifest; "
    "a table of a unit that is not a training unit was not opened."
)


def test_the_summary_says_the_parts_are_not_candidates_and_which_tables_were_read(
    one_worker,
):
    folder, code, printed = one_worker
    assert code == 0
    lines = (folder / hazard_tune.SUMMARY_NAME).read_text().splitlines()
    assert TABLES_READ in lines
    assert any(
        line.startswith("The parts and the reference below are not candidates")
        and line.endswith("take no part in the choice.")
        for line in lines
    )
    assert printed.count("not candidates") == 1  # the same lines are on standard output
    # the lines come before the results they qualify
    assert lines.index(TABLES_READ) < next(
        k for k, line in enumerate(lines) if line.startswith("Parts at the present settings")
    )


OPENED: list | None = None  # (path, mode) of every `open` while a test collects them


def _record_opens(event, arguments):
    if event == "open" and OPENED is not None and isinstance(arguments[0], str | os.PathLike):
        OPENED.append((os.fspath(arguments[0]), arguments[1]))


sys.addaudithook(_record_opens)


def test_the_files_the_command_opens_are_those_its_module_text_lists(tmp_path):
    """The made-up tables folder also holds the tables of a unit that is not named, and the truth
    summary lists that unit; its tables are not opened. One worker, so every open is in this
    process."""
    global OPENED
    units = {unit: cars(n, seed, site=unit) for seed, (unit, n) in enumerate(DAYS.items())}
    other = {"nrel-from-may-2019": cars(6, 9, site="nrel"), **units}
    command = _command(tmp_path, other)
    command[command.index("--units") + 1 :] = list(units)
    assert (tmp_path / "tables" / "nrel-from-may-2019-cars.csv").is_file()
    OPENED = []
    try:
        code, printed = _patched_run(command, tmp_path / "run", 1)
    finally:
        opened, OPENED = OPENED, None
    assert code == 0, printed
    root = str(tmp_path.resolve())
    read = {
        os.path.relpath(os.path.realpath(path), root)
        for path, mode in opened
        if os.path.realpath(path).startswith(root) and not set(str(mode or "r")) & set("wax+")
    }
    run = "20260101T000000Z-wp2-truth"
    grid = command[command.index("--grid") + 1]
    assert read == {
        *(f"tables/{unit}-{kind}.csv" for unit in units for kind in ("cars", "days")),
        f"{run}/truth-summary.json",
        f"{run}/manifest.json",
        "matrix.json",
        os.path.relpath(os.path.realpath(grid), root),
    }


def test_with_one_worker_the_first_fold_is_made_twice_and_every_other_fold_once(
    tmp_path, constant_bound, monkeypatch
):
    """Once for the probe fits of the time estimate, once for the tasks (module text)."""
    made = []
    real = hazard_tune.prepare_fold

    def counted(table, site):
        made.append(site)
        return real(table, site)

    monkeypatch.setattr(hazard_tune, "prepare_fold", counted)
    units = {unit: cars(n, seed, site=unit) for seed, (unit, n) in enumerate(DAYS.items())}
    code, printed = _patched_run(_command(tmp_path, units), tmp_path / "run", 1)
    assert code == 0, printed
    first = made[0]  # the fold of the time estimate
    assert made.count(first) == 2 and sorted(made) == sorted([first, *DAYS])


def test_the_commands_numbers_are_those_of_independent_routes(one_worker, table, tmp_path):
    folder, code, _ = one_worker
    assert code == 0
    results = json.loads((folder / hazard_tune.RESULTS_NAME).read_text())
    cars_tables = frames(tmp_path)
    # a candidate's fold result at the epoch of its checkpoint, from HazardModel's own fit
    for epochs in (3, 30):
        settings = Setting(32, 0.003, 1.0, epochs).hazard(0)
        plain = HazardModel(rho=0.0, kappas=TUNING_KAPPAS, settings=settings)
        plain.fit(pd.concat([cars_tables["caltech"], cars_tables["jpl"]], ignore_index=True))
        osl = feature_frame(build_table(cars_tables["OSL_S"], rho=0.0))
        osl = osl[osl["counting"]].reset_index(drop=True)
        forecast = plain.predict(osl, TUNING_KAPPAS)
        truth = table.loc[(table["site"] == "OSL_S") & table["counting"], learn_tune.LABEL]
        losses = []
        for kappa in TUNING_KAPPAS:
            bound = forecast.loc[forecast["kappa"] == kappa, "d_hat"].to_numpy()
            losses.append(pinball_loss(truth.to_numpy(), np.maximum(bound, 0.0), kappa))
        total = float(sum(losses))
        candidate = next(
            c
            for c in results["candidates"]
            if c["settings"] == Setting(32, 0.003, 1.0, epochs).as_dict()
        )
        assert candidate["sites"]["OSL_S"]["score_by_seed"]["0"] == total
    # the reference is `learn_tune.fold_scores` of the features+tail set at the settings of lgbm
    columns = learn_tune.FEATURE_SETS["features+tail"]
    reference = results["reference"]
    assert reference["feature_set"] == "features+tail" and reference["columns"] == list(columns)
    assert (
        reference["settings"] == LGBM_PRESENT.as_dict() and "not a candidate" in reference["about"]
    )
    for site in ("caltech", "jpl", "OSL_S"):
        assert reference["sites"][site] == learn_tune.fold_scores(
            table, columns, LGBM_PRESENT, site
        )
    assert reference["score"] == pytest.approx(
        np.mean([reference["sites"][s]["score"] for s in reference["sites"]])
    )


def test_one_worker_and_two_workers_write_the_same_bytes(one_worker, two_workers):
    (one, code_one, _), (two, code_two, _) = one_worker, two_workers
    assert code_one == code_two == 0
    for name in (hazard_tune.RESULTS_NAME, hazard_tune.SUMMARY_NAME):
        first = (one / name).read_bytes()
        assert first and first == (two / name).read_bytes()
    assert json.loads((two / hazard_tune.TIMING_NAME).read_text())["workers"] == 2


def test_the_estimate_counts_the_fits_by_kind(table, tmp_path):
    grid = load_grid(tiny_grid(tmp_path))
    tasks = hazard_tune.tasks_of(grid, ["caltech", "jpl", "OSL_S"])
    assert hazard_tune.work_of(tasks) == {"candidate": 12, "part": 30, "reference": 12}
    line = hazard_tune.estimate_line(hazard_tune.prepare_fold(table, "jpl"), grid, tasks, 2)
    assert line.startswith("Estimate: 45 tasks, 42 network fits (12 of candidates, 30 of parts)")
    assert "with 2 worker(s)" in line


def test_the_committed_grid_on_eight_sites_asks_for_the_fits_the_report_counts():
    grid = load_grid()
    tasks = hazard_tune.tasks_of(grid, [f"s{k}" for k in range(8)])
    work = hazard_tune.work_of(tasks)
    assert work == {"candidate": 12 * 3 * 8, "part": 5 * 3 * 8, "reference": 8 * 4}
    assert len(tasks) == 288 + 120 + 8
    assert HazardSettings().epochs == PRESENT.epochs == 30 and TINY["epochs"] == [3, 30]
