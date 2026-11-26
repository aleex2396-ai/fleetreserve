"""The files of the trials made after the sealed run: the folder `trials-after-sealed-run/`,
its matrix and its README, and the levels the no-dial trial asks of `dial_choice`. Written, not run.

Each test says which edit turns it red.

1. Every gate matrix of the folder parses (`gate_config.parse`), scores development units only,
   has no `sealed_choice` and says in `about` that it was written on 24 November 2026 after
   held-out results were seen. Red if a matrix with a held-out unit, or with another `about`, is
   added. A file is a gate matrix when it has the top-level key `configurations`, the key the gate
   loader requires of every matrix (`gate_config.TOP_FIELDS`); no list of names decides it, so a
   new matrix is checked without an edit here, and a gate matrix with a wrong field is refused by
   `parse`, not skipped.
1b. The five forecast settings files of the folder are not gate matrices. Each has its own check
   of the same labelling rule through the reader its own trial uses: `forecast-local.json` through
   `forecast_local.read_settings` (its units are checked there against the development units),
   `forecast-mix.json`, `forecast-noset.json`, `forecast-energy.json` and `forecast-energy-2.json`
   through `load_settings` of their modules (they score the units of the committed gate matrix;
   that those units are development units is already tested in
   `tests/forecast_trials_mix/test_mix_command.py` and
   `tests/forecast_trials_noset/test_noset_command.py`,
   `test_the_committed_matrix_passes_the_check_with_its_three_development_units`,
   and is not repeated here; this file adds that the settings name those units and no held-out
   unit). Red if a settings file loses the label or names a held-out unit.
1c. No JSON file of the folder escapes both checks: a file that is neither a gate matrix nor one
   of the five known settings files fails `test_no_json_file_of_the_folder_escapes_both_checks`.
   The function behind it is run once on a made-up file in a temporary folder to show it red.
2. The matrix of trials a and c holds exactly the configurations the README names: forecast
   energy (energy bounds empirical) at level 0.2 with dial 0.02, and the main reading (energy
   bounds known) at level 0.1 with dial 0.02, each with its oracle partner. Red if a level, a dial
   or an energy reading in the file is changed.
3. The model built from the matrix is fitted at the levels 0.05, 0.1 and 0.2 (`matrix_levels`).
4. Every command of the README is named with its flag and its output files: red if a command or
   an output file is dropped from the README.
6. The folder is outside `gate/`: other tests read every JSON file under `gate/` as a committed
   run matrix of the plan and count them (`tests/learn/test_committed_matrix_levels.py`: 17;
   `tests/wp6gate/test_gate_summary.py`). Red if the matrix is put back under `gate/`.
5. The no-dial trial: `dial_choice.family_at` makes the family at 0.05 and 0.1 from the committed
   gate matrix, with the configuration without rows and the dial 0.02 of the trial.
"""

import json

import pytest
from gate_fake import TOP

from fleetreserve import (
    dial_choice,
    driver_risk,
    forecast_energy,
    forecast_energy_2,
    forecast_local,
    forecast_mix,
    forecast_noset,
    gate_config,
    sealed_route,
)
from fleetreserve.truth_build import DEVELOPMENT_UNITS

FOLDER = TOP / "trials-after-sealed-run"
README = FOLDER / "README.md"
TRIAL_MATRIX = FOLDER / "lgbm_tail-trials-a-c.json"
# The settings files of the five forecast trials, by name; each is read by its own trial's reader.
FORECAST_SETTINGS = (
    "forecast-local.json",
    "forecast-mix.json",
    "forecast-noset.json",
    "forecast-energy.json",
    "forecast-energy-2.json",
)
LABEL_PHRASES = (
    "24 November 2026",
    "after held-out results were seen",
    "fills no pre-registered slot",
)


def is_gate_matrix(path) -> bool:
    """Whether the JSON file has `configurations`, the top-level key `gate_config` requires of every
    matrix. A file that has it is parsed as a matrix by the tests below; one that fails to parse
    is a red test, never a skipped file."""
    document = json.loads(path.read_text())
    return isinstance(document, dict) and "configurations" in document


def files_of_no_check(folder) -> list[str]:
    """The JSON files of `folder` that are neither a gate matrix nor a known settings file."""
    return [
        p.name
        for p in sorted(folder.glob("*.json"))
        if not is_gate_matrix(p) and p.name not in FORECAST_SETTINGS
    ]


ALL_JSON = sorted(FOLDER.glob("*.json"))
MATRICES = [p for p in ALL_JSON if is_gate_matrix(p)]


def test_the_folder_holds_the_matrix_of_trials_a_and_c():
    assert TRIAL_MATRIX in MATRICES and README.is_file()


@pytest.mark.parametrize("path", MATRICES, ids=lambda p: p.name)
def test_every_matrix_of_the_folder_scores_development_units_only_and_is_labelled(path):
    document = json.loads(path.read_text())
    matrix = gate_config.parse(document)
    assert set(matrix.gate_units) <= set(DEVELOPMENT_UNITS)
    assert not set(matrix.gate_units) & set(sealed_route.HELD_OUT_UNITS)
    assert "sealed_choice" not in document and "rotation_whatif" not in document
    for phrase in LABEL_PHRASES:
        assert phrase in document["about"]


def test_no_json_file_of_the_folder_escapes_both_checks():
    assert files_of_no_check(FOLDER) == []  # red if a JSON file of another kind is added
    assert TRIAL_MATRIX in MATRICES and not set(FORECAST_SETTINGS) & {p.name for p in MATRICES}
    assert {p.name for p in ALL_JSON} >= set(FORECAST_SETTINGS)  # a known file cannot go missing


def test_the_check_for_files_of_no_check_is_red_on_a_made_up_file(tmp_path):
    for p in FOLDER.glob("*.json"):
        (tmp_path / p.name).write_text(p.read_text())
    assert files_of_no_check(tmp_path) == []  # a copy of the folder passes
    (tmp_path / "some-other-settings.json").write_text(json.dumps({"about": "made up", "level": 1}))
    assert files_of_no_check(tmp_path) == ["some-other-settings.json"]  # the made-up file is red
    (tmp_path / "some-other-settings.json").write_text(json.dumps({"configurations": []}))
    assert files_of_no_check(tmp_path) == []  # a file with `configurations` is a matrix: parse
    with pytest.raises(gate_config.MatrixError):  # refuses it, so it is red there
        gate_config.parse(json.loads((tmp_path / "some-other-settings.json").read_text()))


def _names_of_units_in(text: str) -> set[str]:
    return {u for u in (*DEVELOPMENT_UNITS, *sealed_route.HELD_OUT_UNITS) if u in text}


def test_forecast_local_settings_score_development_units_only_and_are_labelled():
    path = FOLDER / "forecast-local.json"
    settings = forecast_local.read_settings(path)  # the trial's own reader; checks its units too
    assert settings.units and set(settings.units) <= set(DEVELOPMENT_UNITS)
    assert not set(settings.units) & set(sealed_route.HELD_OUT_UNITS)
    assert not _names_of_units_in(path.read_text()) & set(sealed_route.HELD_OUT_UNITS)
    document = json.loads(path.read_text())
    for phrase in LABEL_PHRASES:
        assert phrase in document["about"]
    assert "sealed_choice" not in document


@pytest.mark.parametrize(
    "module",
    [forecast_mix, forecast_noset, forecast_energy, forecast_energy_2],
    ids=[
        "forecast-mix.json",
        "forecast-noset.json",
        "forecast-energy.json",
        "forecast-energy-2.json",
    ],
)
def test_forecast_mix_and_noset_settings_name_the_development_units_only_and_are_labelled(module):
    """Also `forecast-energy.json` and `forecast-energy-2.json` (the third and fourth parameters):
    the name keeps the first two."""
    path = module.SETTINGS_FILE
    assert path.parent == FOLDER and path.name in FORECAST_SETTINGS
    settings = module.load_settings(path)  # the trial's own reader
    for phrase in LABEL_PHRASES:
        assert phrase in settings["about"]
    # The units these trials score are the matrix's; the settings name them in prose.
    matrix = gate_config.load(module.MATRIX)
    named = _names_of_units_in(json.dumps(settings["units"]))
    assert named == set(matrix.gate_units) and named <= set(DEVELOPMENT_UNITS)
    assert not _names_of_units_in(path.read_text()) & set(sealed_route.HELD_OUT_UNITS)
    assert "sealed_choice" not in settings


def test_the_matrix_of_trials_a_and_c_holds_the_two_trials_and_their_oracle_partners():
    matrix = gate_config.load(TRIAL_MATRIX)
    found = {
        c.name: (c.kappa, c.driver_risk, c.energy_bounds, c.departure, c.departure_seen)
        for c in matrix.configurations
    }
    seen = "rounded-hour"
    assert found == {
        "forecast-energy-k0.2-d0.02-empirical": (0.2, 0.02, "empirical", "empirical", seen),
        "forecast-energy-k0.2-d0.02-oracle": (0.2, 0.02, "empirical", "oracle", seen),
        "main-k0.1-d0.02-empirical": (0.1, 0.02, "known", "empirical", seen),
        "main-k0.1-d0.02-oracle": (0.1, 0.02, "known", "oracle", seen),
        "main-k0.2-d0.02-empirical": (0.2, 0.02, "known", "empirical", seen),
        "main-k0.2-d0.02-oracle": (0.2, 0.02, "known", "oracle", seen),
    }
    committed = gate_config.load(TOP / "gate" / "matrix.json")
    assert matrix.gate_units == committed.gate_units  # the development units of the gate matrix
    shared = ("rho", "split_seed", "n_calibration", "calibration_half_below", "tiers")
    assert all(getattr(matrix, field) == getattr(committed, field) for field in shared)


def test_no_file_of_the_trials_is_under_gate_where_other_tests_read_every_json_as_a_matrix():
    under_gate = {p.name for p in (TOP / "gate").rglob("*.json")}
    assert not under_gate & {p.name for p in ALL_JSON}  # red if a trial file is put under gate/
    other = (TOP / "tests" / "learn" / "test_committed_matrix_levels.py").read_text()
    assert '(ROOT / "gate").rglob("*.json")' in other  # the glob this test stands guard against


def test_trial_a_has_a_known_energy_baseline_at_its_own_level_dial_and_departure():
    matrix = gate_config.load(TRIAL_MATRIX)
    forecast = [c for c in matrix.configurations if c.energy_bounds == "empirical"]
    assert forecast  # trial a is in the file
    for c in forecast:
        like = [
            m
            for m in matrix.configurations
            if m.energy_bounds == "known"
            and (m.kappa, m.driver_risk, m.departure) == (c.kappa, c.driver_risk, c.departure)
        ]
        assert len(like) == 1, c.name  # red if the main pair at level 0.2 is removed


def test_the_fixed_plan_of_trial_c_is_read_from_the_pair_at_level_0_1():
    matrix = gate_config.load(TRIAL_MATRIX)
    pair = [c for c in matrix.configurations if c.kappa == 0.1]
    assert sorted(c.name for c in pair) == ["main-k0.1-d0.02-empirical", "main-k0.1-d0.02-oracle"]
    assert {c.energy_bounds for c in pair} == {"known"} and {c.driver_risk for c in pair} == {0.02}


def test_the_model_of_the_matrix_is_fitted_at_the_levels_the_trials_read():
    matrix = gate_config.load(TRIAL_MATRIX)
    assert driver_risk.matrix_levels(matrix.configurations) == (0.05, 0.1, 0.2)


def test_the_readme_names_every_command_with_its_flag_and_its_output_files():
    text = README.read_text()
    assert "24 November 2026" in text and "after held-out results were seen" in text
    for command in (
        "fleetreserve.learn_run --model lgbm_tail --matrix trials-after-sealed-run/"
        "lgbm_tail-trials-a-c.json --made-after-sealed-run",
        "fleetreserve.dial_choice --model lgbm_tail --kappa 0.05 --splits 200 "
        "--made-after-sealed-run",
        "fleetreserve.dial_choice --model lgbm_tail --kappa 0.1 --splits 200 "
        "--made-after-sealed-run",
        "fleetreserve.pooled_calibration --model lgbm_tail --splits 200",
    ):
        assert command in text
    for output in (
        "learn.json",
        "dial-choice.json",
        "dial-choice-summary.txt",
        "pooled-calibration.json",
        "pooled-calibration-summary.txt",
        "made-after-sealed-run.txt",
    ):
        assert output in text
    assert "/Users" not in text and "/private" not in text


@pytest.mark.parametrize("kappa", [0.05, 0.1])
def test_dial_choice_makes_the_family_the_no_dial_trial_reads(kappa):
    matrix = gate_config.load(TOP / "gate" / "matrix.json")
    family = dial_choice.find_family(matrix)
    _, (plain, dials) = dial_choice.family_at(matrix, family, kappa)
    assert plain.kappa == kappa and plain.driver_risk is None
    assert plain.name == f"main-rounded-hour-k{kappa:g}-empirical"
    assert 0.02 in [d.driver_risk for d in dials]
    assert all(d.kappa == kappa for d in dials)


def test_the_levels_of_the_no_dial_trial_are_levels_of_the_plan_grid():
    assert {0.05, 0.1} <= set(driver_risk.LEVELS)
