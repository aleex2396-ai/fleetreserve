"""The tuning command of the pooled LightGBM departure model (`learn_tune`), on made-up data.

Tiny hand-made tables (three sites, a few rows, the booster replaced by a constant bound of 4
hours or by a spy) and made-up training units (`quantile_fake`) with real fits on a small grid.

1. Folds: the fit rows of a fold are every row of the other sites and never a row of the held-out
   site; the scored rows are the held-out site's rows of counting days. The booster of every task
   is handed the other sites' rows only.
2. Units: a unit that is not a training unit is refused before anything is read (`check_units`,
   `tune_table`, and the command, exit 2); a site in two units and a single site are refused.
3. Score: on the tiny table the score equals the pinball losses computed by hand; it is the mean
   over sites, not over cars; rows of non-counting days are not scored; the per-type means and the
   miss rates are the by-hand ones. With real fits a fold's losses equal those of
   `models.lgbm.QuantileSet` fitted on the other sites at the candidate's settings.
4. The results name the candidate `learn_tune_grid.choose` picks, rank the present settings, and
   score the present settings on the three feature sets.
5. One worker and three workers write byte-identical results and summaries.
6. The command on made-up training units: exit 0, the estimate line first, the three files
   written; an unknown site ends with exit 1.
"""

import json

import numpy as np
import pandas as pd
import pytest
from gate_fake import COMMIT, write_build
from learn_fake import small_matrix
from quantile_fake import cars, unit_cars

from fleetreserve import gate_run, learn_tune, run_verify
from fleetreserve.forecast_empirical import TUNING_KAPPAS
from fleetreserve.learn_eval import pinball_loss
from fleetreserve.learn_tune_grid import PRESENT, Settings, choose, load_grid
from fleetreserve.models.lgbm import PARAMETERS, QuantileSet, features

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
BY_HAND = {"caltech": 4.0, "jpl": 1.3, "OSL_S": 10.05}  # module text of the test below


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


def _small_grid(tmp_path):
    path = tmp_path / "grid.json"
    path.write_text(
        json.dumps(
            {
                "about": "made up",
                "num_leaves": [7, 15],
                "min_data_in_leaf": [20],
                "learning_rate_and_rounds": [[0.05, 200]],
            }
        )
    )
    return path


@pytest.fixture
def constant_bound(monkeypatch):
    """The booster replaced by a bound of 4 hours for every row; the calls are kept."""
    calls = []

    def fake(x, y, x_new, columns, parameters, rounds):
        calls.append({"x": x, "y": y, "x_new": x_new, "parameters": parameters, "rounds": rounds})
        return np.full(x_new.shape[0], 4.0)

    monkeypatch.setattr(learn_tune, "_quantile", fake)
    return calls


# --- folds ---------------------------------------------------------------------------------------


def test_fold_rows_fit_on_the_other_sites_and_score_the_counting_rows_of_the_site():
    table = _tiny()
    fit, scored = learn_tune.fold_rows(table, "caltech")
    assert fit.tolist() == [False, False, False, True, True, True, True, True]
    assert scored.tolist() == [True, True, False, False, False, False, False, False]
    fit, scored = learn_tune.fold_rows(table, "OSL_S")
    assert fit.tolist() == [True, True, True, True, True, True, False, False]
    assert scored.tolist() == [False, False, False, False, False, False, True, False]


def test_a_held_out_sites_rows_never_enter_its_own_folds_fit(tmp_path, constant_bound):
    table = _tiny()
    grid = load_grid(_small_grid(tmp_path))
    learn_tune.tune(table, grid)
    tasks = learn_tune.tasks_of(grid, list(SITES))
    assert len(constant_bound) == len(tasks) * len(TUNING_KAPPAS)
    labels = {site: sorted(label for s, _, label in ROWS if s == site) for site in SITES}
    seen = set()
    for call in constant_bound:
        held = set(call["x_new"][:, 0])
        assert len(held) == 1
        (code,) = held
        site = next(s for s, c in CODES.items() if c == code)
        seen.add(site)
        # the booster is fitted on every row of the other two sites and on nothing else
        assert code not in set(call["x"][:, 0])
        others = sorted(label for s in SITES if s != site for label in labels[s])
        assert sorted(call["y"]) == others
        assert not set(call["y"]) & set(labels[site])
        # and read at the held-out site's rows of counting days
        assert call["x_new"].shape[0] == sum(1 for s, counting, _ in ROWS if s == site and counting)
    assert seen == set(SITES)


def test_every_task_is_a_feature_set_a_candidate_and_a_site():
    grid = load_grid()
    sites = [f"s{k}" for k in range(8)]
    tasks = learn_tune.tasks_of(grid, sites)
    assert len(tasks) == 48 * 8 + 2 * 8 == len(set(tasks))
    assert {name for name, _, _ in tasks} == set(learn_tune.FEATURE_SETS)
    assert all(s == PRESENT for name, s, _ in tasks if name != learn_tune.TUNED_SET)
    assert learn_tune.work_of(tasks) == (1600, 4 * 8 * (16 * 1600 + 2 * 200))


# --- units ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "units", [("nrel-from-may-2019",), ("caltech", "OSL_1"), ("OSL_T",), ("no-such-unit",), ()]
)
def test_a_unit_that_is_not_a_training_unit_is_refused_before_anything_is_read(
    units, tmp_path, capsys
):
    with pytest.raises(learn_tune.NotTraining):
        learn_tune.check_units(units)
    if units:
        with pytest.raises(learn_tune.NotTraining):
            # the frames are not cars tables: reading one would raise something else
            learn_tune.tune_table({u: None for u in units}, rho=0.0)
        missing = tmp_path / "no-such-folder"
        code = learn_tune.main(
            ["--units", *units, "--tables", str(missing), "--truth-run", str(missing)]
        )
        assert code == 2
        printed = capsys.readouterr().out
        assert printed.startswith("COULD NOT RUN:") and "training units" in printed


def test_a_training_unit_named_twice_is_refused():
    with pytest.raises(learn_tune.NotTraining, match="once"):
        learn_tune.check_units(("caltech", "caltech"))
    assert learn_tune.check_units(("caltech", "BOD")) == ("caltech", "BOD")


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    folder = tmp_path_factory.mktemp("units")
    days = {"caltech": 30, "jpl": 14, "OSL_S": 22}
    return {
        unit: unit_cars(folder, unit, cars(n, seed, site=unit))
        for seed, (unit, n) in enumerate(days.items())
    }


@pytest.fixture(scope="module")
def table(frames):
    return learn_tune.tune_table(frames, rho=0.0)


def test_the_tune_table_holds_the_features_the_context_and_the_label(frames, table):
    assert len(table) == sum(len(f) for f in frames.values())
    for column in (*learn_tune.FEATURE_SETS[learn_tune.TUNED_SET], learn_tune.LABEL, "counting"):
        assert column in table.columns
    assert learn_tune.sites_of(table) == ["caltech", "jpl", "OSL_S"]
    assert table.groupby("site")["site_type"].first().to_dict() == SITES
    assert table["public_holiday"].any()  # 1 January 2020 is among the made-up days


def test_a_site_in_two_units_and_a_single_site_are_refused(frames):
    twice = {"caltech": frames["caltech"], "jpl": frames["jpl"].assign(site="caltech")}
    with pytest.raises(ValueError, match="two units"):
        learn_tune.tune_table(twice, rho=0.0)
    alone = learn_tune.tune_table({"caltech": frames["caltech"]}, rho=0.0)
    with pytest.raises(ValueError, match="at least two sites"):
        learn_tune.sites_of(alone)
    with pytest.raises(ValueError, match="no site type"):
        learn_tune.tune_table({"caltech": frames["caltech"].assign(site="elsewhere")}, rho=0.0)


# --- the score -----------------------------------------------------------------------------------


def test_the_score_is_the_by_hand_pinball_loss_and_a_mean_over_sites(tmp_path, constant_bound):
    """Bound 4 h for every car; levels 0.05, 0.1, 0.2, 0.3 (their sum 0.65). Per level k:
    caltech, departures 2 and 6: (2(1 - k) + 2k) / 2 = 1, summed over the levels 4.0 (its
    non-counting row at 100 is not scored); jpl, departures 4, 5, 9: (0 + k + 5k) / 3 = 2k,
    summed 1.3; OSL_S, departure 1: 3(1 - k), summed 3 (4 - 0.65) = 10.05. The score is the mean
    of the three, 15.35 / 3; the mean over the six cars would be (2 x 4.0 + 3 x 1.3 + 10.05) / 6."""
    results = learn_tune.tune(_tiny(), load_grid(_small_grid(tmp_path)))
    for candidate in results["candidates"]:
        for site, expected in BY_HAND.items():
            assert candidate["sites"][site]["score"] == pytest.approx(expected, abs=1e-12)
        assert candidate["sites"]["caltech"]["pinball_loss"] == pytest.approx(
            {repr(k): 1.0 for k in TUNING_KAPPAS}
        )
        assert candidate["sites"]["jpl"]["pinball_loss"] == pytest.approx(
            {repr(k): 2 * k for k in TUNING_KAPPAS}
        )
        assert candidate["score"] == pytest.approx(15.35 / 3, abs=1e-12)
        assert abs(candidate["score"] - (2 * 4.0 + 3 * 1.3 + 10.05) / 6) > 1.0
        assert candidate["score_by_site_type"] == pytest.approx({"workplace": 2.65, "other": 10.05})
        # a car leaves before the bound: one of two at caltech, none at jpl, the one at OSL_S
        assert candidate["sites"]["caltech"]["miss_rate"] == {repr(k): 0.5 for k in TUNING_KAPPAS}
        assert candidate["miss_rate"] == pytest.approx({repr(k): 0.5 for k in TUNING_KAPPAS})
        assert candidate["sites"]["caltech"]["scored_rows"] == 2
        assert candidate["sites"]["caltech"]["fit_rows"] == 5
        assert candidate["sites"]["OSL_S"]["fit_rows"] == 6
    assert results["kappas"] == [0.05, 0.1, 0.2, 0.3]
    assert results["sites"]["caltech"] == {
        "unit": "caltech",
        "site_type": "workplace",
        "rows": 3,
        "scored_rows": 2,
    }
    assert results["sites"]["OSL_S"]["site_type"] == "other"
    # equal scores: the tie goes to the fewer leaves
    assert results["chosen"]["settings"] == Settings(7, 20, 0.05, 200).as_dict()
    assert results["present"]["rank"] == 1 == results["chosen"]["rank"]


def test_each_candidate_is_fitted_at_its_own_settings_and_levels(tmp_path, constant_bound):
    grid = load_grid(_small_grid(tmp_path))
    learn_tune.tune(_tiny(), grid)
    tasks = learn_tune.tasks_of(grid, list(SITES))
    for k, (_, settings, _) in enumerate(tasks):
        calls = constant_bound[4 * k : 4 * k + 4]
        assert [c["parameters"]["alpha"] for c in calls] == list(TUNING_KAPPAS)
        for call in calls:
            assert call["rounds"] == settings.rounds
            assert call["parameters"] == {
                **PARAMETERS,
                **settings.overrides(),
                "alpha": call["parameters"]["alpha"],
            }


def test_a_site_with_no_counting_row_cannot_be_scored(tmp_path, constant_bound):
    table = _tiny()
    table.loc[table["site"] == "OSL_S", "counting"] = False
    with pytest.raises(ValueError, match="no row of a counting day"):
        learn_tune.tune(table, load_grid(_small_grid(tmp_path)))


def test_the_bound_is_raised_to_zero():
    x = np.zeros((40, 1))
    bound = learn_tune._quantile(
        x, np.full(40, -5.0), x[:3], ["f"], {**PARAMETERS, "alpha": 0.1}, 5
    )
    assert bound.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("settings", [PRESENT, Settings(7, 50, 0.02, 60)])
def test_a_folds_losses_are_those_of_the_models_boosters_on_the_other_sites(table, settings):
    columns = learn_tune.FEATURE_SETS[learn_tune.TUNED_SET]
    found = learn_tune.fold_scores(table, columns, settings, "jpl")
    others = table[table["site"] != "jpl"]
    held = table[(table["site"] == "jpl") & table["counting"]]
    assert found["fit_rows"] == len(others) and found["scored_rows"] == len(held) > 0
    quantiles = QuantileSet(
        {"departure": TUNING_KAPPAS},
        columns,
        {**PARAMETERS, **settings.overrides()},
        settings.rounds,
    ).fit(others)
    truth = held[learn_tune.LABEL].to_numpy(float)
    total = 0.0
    for kappa in TUNING_KAPPAS:
        bound = quantiles.predict(features(held, columns), "departure", kappa)
        assert found["pinball_loss"][repr(kappa)] == pinball_loss(truth, bound, kappa)
        assert found["miss_rate"][repr(kappa)] == float(np.mean(truth < bound))
        total += pinball_loss(truth, bound, kappa)
    assert found["score"] == pytest.approx(total, rel=1e-12) and total > 0


# --- results, workers and the command ------------------------------------------------------------


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    days = {"caltech": 30, "jpl": 14, "OSL_S": 22}
    units = {unit: cars(n, seed, site=unit) for seed, (unit, n) in enumerate(days.items())}
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    grid = _small_grid(folder)
    return [
        *("--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)),
        *("--grid", str(grid), "--units", *units),
    ]


def _run(build, folder, monkeypatch, workers: int) -> int:
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    folder.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
    return learn_tune.main([*build, "--workers", str(workers)])


def test_the_command_scores_every_candidate_on_made_up_training_units(
    build, table, tmp_path, monkeypatch, capsys
):
    assert _run(build, tmp_path / "run", monkeypatch, 1) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("Estimate: 12 tasks, 48 fits, 9600 boosting rounds in all; about ")
    results = json.loads((tmp_path / "run" / learn_tune.RESULTS_NAME).read_text())
    summary = (tmp_path / "run" / learn_tune.SUMMARY_NAME).read_text()
    assert printed.endswith(summary) and summary.startswith("Tuning of the pooled LightGBM")
    assert "total_seconds" in json.loads((tmp_path / "run" / learn_tune.TIMING_NAME).read_text())
    grid = load_grid(build[build.index("--grid") + 1])
    assert results["grid"] == {"file": "grid.json", "sha256": grid.sha256, "candidates": 2}
    assert results["fit_units"] == ["caltech", "jpl", "OSL_S"]
    assert results["rows"] == len(table) == 12 * (30 + 14 + 22)
    assert sorted(results["sites"]) == ["OSL_S", "caltech", "jpl"]
    assert results["inputs"]["truth_run"] and results["matrix"]["rho"] == 0.0
    assert [Settings(**c["settings"]) for c in results["candidates"]] == list(grid.candidates)
    scored = []
    for candidate in results["candidates"]:
        sites = candidate["sites"]
        assert sorted(sites) == ["OSL_S", "caltech", "jpl"]
        assert candidate["score"] == pytest.approx(np.mean([s["score"] for s in sites.values()]))
        assert candidate["score_by_site_type"]["other"] == sites["OSL_S"]["score"]
        assert sites["jpl"]["fit_rows"] == 12 * (30 + 22)
        assert sites["jpl"]["scored_rows"] == 12 * 14
        scored.append((Settings(**candidate["settings"]), candidate["score"]))
    # the command's numbers are those of the functions on the same made-up tables
    columns = learn_tune.FEATURE_SETS[learn_tune.TUNED_SET]
    again = learn_tune.fold_scores(table, columns, PRESENT, "OSL_S")
    assert results["candidates"][1]["sites"]["OSL_S"] == again
    assert results["chosen"]["settings"] == choose(scored).as_dict()
    assert results["chosen"]["score"] == min(score for _, score in scored)
    assert results["present"]["settings"] == PRESENT.as_dict()
    assert results["present"]["score"] == dict(scored)[PRESENT]
    by_set = results["present_settings_by_feature_set"]
    assert set(by_set) == set(learn_tune.FEATURE_SETS) == set(results["feature_sets"])
    assert by_set[learn_tune.TUNED_SET]["score"] == results["present"]["score"]
    plain = learn_tune.fold_scores(table, learn_tune.FEATURE_SETS["features"], PRESENT, "caltech")
    assert by_set["features"]["sites"]["caltech"] == plain
    assert len(results["feature_sets"]["features"]) == 14
    assert len(results["feature_sets"]["features+tail"]) == 24
    assert results["feature_sets"][learn_tune.TUNED_SET][-2:] == ["public_holiday", "site_type"]


def test_one_worker_and_three_workers_write_the_same_bytes(build, tmp_path, monkeypatch, capsys):
    assert _run(build, tmp_path / "one", monkeypatch, 1) == 0
    assert _run(build, tmp_path / "three", monkeypatch, 3) == 0
    capsys.readouterr()
    for name in (learn_tune.RESULTS_NAME, learn_tune.SUMMARY_NAME):
        one = (tmp_path / "one" / name).read_bytes()
        assert one and one == (tmp_path / "three" / name).read_bytes()


def test_run_tasks_keeps_the_order_of_the_tasks_with_several_workers(table):
    columns = learn_tune.FEATURE_SETS["features"]
    tasks = [("features", Settings(7, 20, 0.05, 20), site) for site in ("OSL_S", "caltech", "jpl")]
    found = learn_tune.run_tasks(table, tasks, workers=2)
    assert found == [learn_tune.fold_scores(table, columns, s, site) for _, s, site in tasks]
    assert [f["scored_rows"] for f in found] == [12 * 22, 12 * 30, 12 * 14]


def test_an_unknown_site_ends_the_command_with_exit_1(tmp_path, monkeypatch, capsys):
    run, tables = write_build(
        tmp_path, {"caltech": cars(6, 0, site="caltech"), "jpl": cars(6, 1, site="elsewhere")}
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    arguments += ["--grid", str(_small_grid(tmp_path)), "--units", "caltech", "jpl"]
    assert _run(arguments, tmp_path / "run", monkeypatch, 1) == 1
    assert "no site type given" in capsys.readouterr().out
    assert not list((tmp_path / "run").iterdir())


def test_the_command_cannot_run_without_a_verified_truth_build(build, tmp_path, capsys):
    arguments = list(build)
    arguments[arguments.index("--truth-run") + 1] = str(tmp_path / "none")
    assert learn_tune.main(arguments) == 2
    assert capsys.readouterr().out.startswith("COULD NOT RUN:")
