"""The command of the forecast trial `mix` (`forecast_mix`) on the made-up gate units. Written, not
run. Each test says which edit turns it red.

1. The refusal. A matrix with a gate unit that is not a development unit ends with exit 2 before a
   truth build, a table or a model is touched; nothing is written. Red if the `check_units` call
   of `main` is removed (the run would go on to read the inputs) or moved after the model is
   loaded. The committed matrix passes the check with its three development gate units.
2. The settings file is required: a missing or incomplete one is exit 2 before the matrix is read.
   Red if `load_settings` is not called first in `main`.
3. The label. A whole run writes the key and the label in the comparison, in both models' results
   and in both models' per-split files, the first two lines of the summary (printed and in the
   file) and the label file, whatever the options. Red if `after_sealed_run.mark` is removed for
   the comparison or for a model, if `summary_head` is removed from `summary_lines`, or if
   `write_note` is removed.
4. A whole run with the same model on both sides: every pair is a pair of equal offers (difference
   0, equal in every split), both models' forecast tables are written and their hashes are in the
   results. Red if the trial side is measured with another factory than its own model's.
5. A whole run with a trial model that forecasts earlier departures: the paired figures of every
   cell equal the figures worked out again from the two per-split files of the run (same seeds,
   same order); the headline cells exist for the units that can be split; the unit that cannot be
   split has its status only. Red if the pairing is by position in another order, or if the trial's
   results are read from the main model's document.
6a. What was run is what is reported: the level, splits, models and members written to the output
   are those of the run, and `settings_check` says whether every value equals the committed
   settings file's. Red if `run.kappa` or `run.members` is read from the settings file again, or if
   `settings_check` ignores a key.
6. The committed settings file: its values are the ones the code and the plan give (level 0.5, 200
   splits, the main model, the trial model and its two members); the neural seed is the default
   seed of the settings the neural member is built with; the command and the model have no weight
   option. Red if a value in the file changes without the code, or the code without the file.
6b. The extra line is run at its own level, labelled, judged by the same rule and never read by
   the one-line reading (a test forces its words to gains and sees the line unchanged); each
   override is refused without `--not-the-trial` before the matrix is read, and the committed
   values given out loud are no override; with the flag the first summary line and every label say
   so; the check's reference is the committed file even when `--settings` names another. Red if
   `overrides` skips an option, if the extra line's units reach `one_line`, or if `settings_check`
   is given the file passed by `--settings` as its reference.
7. The new modules name no loader of the sealed data and import none.
"""

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from gate_fake import COMMIT, gate_units, matrix_document, write_build

from fleetreserve import (
    after_sealed_run,
    dial_choice,
    driver_risk,
    forecast_average,
    forecast_mix,
    forecast_mix_pairs,
    forecast_mix_reading,
    gate_config,
    gate_run,
    learn_run,
    run_verify,
    sealed_route,
)
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.hazard_tune_grid import Setting
from fleetreserve.models import mix_tail_inputs
from fleetreserve.truth_build import DEVELOPMENT_UNITS, TRAIN_UNITS
from fleetreserve.truth_verify import CouldNotRun

NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.03-empirical",
    "main-rounded-hour-k0.3-eps0.03-oracle",
    "main-rounded-hour-k0.3-eps0.1-empirical",
    "main-rounded-hour-k0.3-eps0.1-oracle",
    "main-rounded-hour-k0.2-empirical",
    "main-rounded-hour-k0.2-oracle",
)
SPLITS = 4
EXTRA_LEVEL = 0.2  # the extra level of these runs: the made-up matrix lists 0.3 and 0.2
SETTINGS = json.loads(Path(forecast_mix.SETTINGS_FILE).read_text())
# The words that open the two entries of docs/prior-looks.md that the settings cite: the entry
# with the bars, and the entry that motivates the extra level.
SEED_ENTRY = "The neural model's seed test"
EXTRA_ENTRY = "The neural model at kappa 0.4 and 0.5, seeds 1 to 3"


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, gate_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return run, tables, matrix


def arguments(build, *extra) -> list[str]:
    run, tables, matrix = build
    return [
        "--matrix",
        str(matrix),
        "--tables",
        str(tables),
        "--truth-run",
        str(run),
        "--not-the-trial",
        "--kappa",
        "0.3",
        "--extra-level",
        str(EXTRA_LEVEL),
        "--splits",
        str(SPLITS),
        "--main-model",
        "empirical",
        "--trial-model",
        "empirical",
        *extra,
    ]


def run_command(build, out: Path, monkeypatch, *extra) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return forecast_mix.main(arguments(build, *extra))


def files(folder: Path) -> dict:
    return {p.name: p.read_text() for p in sorted(folder.iterdir()) if p.is_file()}


# --- 1. the refusal -----------------------------------------------------------------------------


def forbid(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a truth build, a table or a model was touched")

    monkeypatch.setattr(gate_run, "check_inputs", forbidden)
    monkeypatch.setattr(forecast_mix, "load_unit_days", forbidden)
    monkeypatch.setattr(learn_run, "load_model", forbidden)


HELD_OUT_NAMES = tuple(sealed_route.HELD_OUT_UNITS)  # the project's own list of the names


@pytest.mark.parametrize("name", ["caltech", "no-such-unit", *HELD_OUT_NAMES])
def test_a_matrix_with_a_unit_that_is_not_a_development_unit_is_refused_first(
    build, tmp_path, monkeypatch, capsys, name
):
    document = matrix_document(NAMES)
    document["gate_units"] = ["nrel-from-may-2019", name]
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(document))
    forbid(monkeypatch)
    run, tables, _ = build
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = forecast_mix.main(
        [
            "--matrix",
            str(path),
            "--tables",
            str(tables),
            "--truth-run",
            str(run),
            "--not-the-trial",
        ]
    )
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("COULD NOT RUN") and name in out
    assert [p.name for p in tmp_path.iterdir()] == ["matrix.json"]


def test_every_held_out_unit_name_is_refused_by_the_check():
    assert HELD_OUT_NAMES and "OSL_T" in HELD_OUT_NAMES
    assert not set(HELD_OUT_NAMES) & set(DEVELOPMENT_UNITS)
    for name in HELD_OUT_NAMES:
        with pytest.raises(CouldNotRun, match=name):
            forecast_mix.check_units(["nrel-from-may-2019", name])


def test_the_committed_matrix_passes_the_check_with_its_three_development_units():
    matrix = gate_config.load(forecast_mix.MATRIX)
    assert forecast_mix.check_units(matrix.gate_units) == (
        "nrel-before-may-2019",
        "nrel-from-may-2019",
        "norway-development-pooled",
    )
    assert set(matrix.gate_units) <= set(DEVELOPMENT_UNITS)
    for name in TRAIN_UNITS:
        with pytest.raises(CouldNotRun, match="not a development unit"):
            forecast_mix.check_units([name])
    with pytest.raises(CouldNotRun, match="no gate unit"):
        forecast_mix.check_units([])


# --- 2. the settings file -----------------------------------------------------------------------


def test_a_missing_or_incomplete_settings_file_could_not_run_before_the_matrix_is_read(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("the matrix was read")

    monkeypatch.setattr(forecast_mix, "load", forbidden)
    code = forecast_mix.main(["--settings", str(tmp_path / "none.json")])
    assert code == 2 and "settings file" in capsys.readouterr().out
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"level": 0.5}))
    code = forecast_mix.main(["--settings", str(partial)])
    out = capsys.readouterr().out
    assert code == 2 and "lacks" in out and "splits" in out


# --- 2b. what was run is what is reported -------------------------------------------------------

COMMITTED_HASHES = {"matrix_sha256": "m" * 64, "settings_sha256": "s" * 64}


def committed_run() -> dict:
    return {
        "level": SETTINGS["level"],
        "extra_level": SETTINGS["extra_level"],
        "splits": SETTINGS["splits"],
        "main_model": SETTINGS["main_model"],
        "trial_model": SETTINGS["trial_model"],
        "members": list(SETTINGS["members"]),
        **COMMITTED_HASHES,
    }


def test_a_run_with_every_committed_value_is_reported_as_the_committed_one():
    check = forecast_mix.settings_check(SETTINGS, committed_run(), COMMITTED_HASHES)
    assert check == {
        "every_value_equals_the_committed_settings": True,
        "differences": {},
        "inputs_that_differ_from_the_defaults": {},
    }
    other = forecast_mix.settings_check(
        SETTINGS, committed_run(), COMMITTED_HASHES, {"tables": "elsewhere"}
    )
    assert other["every_value_equals_the_committed_settings"] is True
    assert other["inputs_that_differ_from_the_defaults"] == {"tables": "elsewhere"}


@pytest.mark.parametrize(
    "key,value",
    [
        ("level", 0.3),
        ("extra_level", 0.2),
        ("splits", 4),
        ("main_model", "empirical"),
        ("trial_model", "empirical"),
        ("members", ["empirical"]),
        ("matrix_sha256", "x" * 64),
        ("settings_sha256", "x" * 64),
    ],
)
def test_a_changed_value_is_reported_as_changed_with_both_values(key, value):
    ran = committed_run()
    ran[key] = value
    check = forecast_mix.settings_check(SETTINGS, ran, COMMITTED_HASHES)
    assert check["every_value_equals_the_committed_settings"] is False
    assert list(check["differences"]) == [key]
    assert check["differences"][key]["run"] == value


def test_members_are_read_from_the_model_that_ran():
    class Named:
        """What `MixModel.__init__` and `members_of` read: a name and the fit units."""

        fit_units = TRAIN_UNITS

        def __init__(self, name):
            self.name = name

    mix = mix_tail_inputs.MixModel(tail=Named("a"), neural=Named("b"))
    assert forecast_mix.members_of(mix) == ["a", "b"]
    assert forecast_mix.members_of(Named("empirical")) == ["empirical"]


# --- 3 and 4. a whole run, the same model on both sides -----------------------------------------


@pytest.fixture(scope="module")
def same(build, tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        out = tmp_path_factory.mktemp("same") / "out"
        code = run_command(build, out, monkeypatch)
    return code, out


def test_a_whole_run_writes_every_output_with_the_label(same):
    code, out = same
    assert code == 0
    found = files(out)
    expected = {
        forecast_mix.RESULTS_NAME,
        forecast_mix.SUMMARY_NAME,
        forecast_mix.TIMING_NAME,
        after_sealed_run.FILE_NAME,
        forecast_mix.NOT_THE_TRIAL_FILE,
        *(
            forecast_mix.side_name(side, kind)
            for side in forecast_mix.RUNS
            for kind in ("dial-choice", "split-offers", "forecasts")
        ),
    }
    assert set(found) == expected
    for name in expected:
        if name.endswith(".json") and name != forecast_mix.TIMING_NAME:
            document = json.loads(found[name])
            assert document[after_sealed_run.KEY] == after_sealed_run.record(), name
            assert document["label"].startswith(forecast_mix.NOT_THE_TRIAL), name
            assert after_sealed_run.WHAT in document["label"], name
    lines = found[forecast_mix.SUMMARY_NAME].splitlines()
    assert lines[0].startswith(forecast_mix.NOT_THE_TRIAL)  # the first line says so
    assert lines[1:3] == [after_sealed_run.WHAT, after_sealed_run.SENTENCE]
    label_lines = found[after_sealed_run.FILE_NAME].splitlines()
    assert label_lines[:2] == lines[1:3]
    # a run that is not the trial says so in this file too
    assert label_lines[2:] == [forecast_mix.NOT_THE_TRIAL]
    assert found[forecast_mix.NOT_THE_TRIAL_FILE].startswith(forecast_mix.NOT_THE_TRIAL)
    assert forecast_mix.UNCERTAINTY in lines and forecast_mix.ONE_TRAINING in lines
    definitions = json.loads(found[forecast_mix.RESULTS_NAME])["definitions"]
    assert definitions["uncertainty"] == [forecast_mix.UNCERTAINTY, forecast_mix.ONE_TRAINING]


def test_a_run_with_other_options_reports_what_was_run_not_the_settings_file(same):
    """The fixture ran `--kappa 0.3 --splits 4` and the model `empirical` on both sides."""
    _, out = same
    document = json.loads((out / forecast_mix.RESULTS_NAME).read_text())
    run = document["run"]
    assert run["kappa"] == 0.3 and run["splits"] == SPLITS
    assert run["main_model"] == run["trial_model"] == "empirical"
    assert run["members"] == ["empirical"] != SETTINGS["members"]
    check = document["settings_check"]
    assert check["every_value_equals_the_committed_settings"] is False
    for key in ("level", "splits", "main_model", "trial_model", "members", "matrix_sha256"):
        assert key in check["differences"], key
    assert check["differences"]["level"] == {"run": 0.3, "committed": SETTINGS["level"]}
    assert check["inputs_that_differ_from_the_defaults"].keys() == {"tables", "truth_run"}
    summary = (out / forecast_mix.SUMMARY_NAME).read_text()
    assert "differs from the committed settings file's" in summary
    assert "Data options differing from the defaults" in summary
    assert document["definitions"]["committed_settings"] == SETTINGS


def test_the_same_model_on_both_sides_gives_equal_offers_in_every_split(same):
    _, out = same
    document = json.loads((out / forecast_mix.RESULTS_NAME).read_text())
    seen = 0
    for _unit, entry in document["comparison"].items():
        if entry["status"] != "calibratable":
            assert set(entry) == {"status", "counting_days"}
            continue
        for _tier, block in entry["tiers"].items():
            cells = [*block["dials"].values(), block["same_dial"], block["own_dial"]]
            for cell in filter(None, cells):
                seen += 1
                assert cell["splits"] == SPLITS and cell["splits_equal"] == SPLITS
                assert cell["mean_difference_kwh_per_test_day"] == 0.0
    assert seen >= 2 * 4
    for side in forecast_mix.RUNS:
        entry = document["models"][side]
        table = out / forecast_mix.side_name(side, "forecasts")
        assert entry["forecast_table"]["sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
        assert entry["forecast_table"]["file"] == table.name


def test_the_printed_summary_is_the_summary_file(build, tmp_path, monkeypatch, capsys):
    code = run_command(build, tmp_path / "out", monkeypatch)
    text = capsys.readouterr().out
    assert code == 0, text
    assert (tmp_path / "out" / forecast_mix.SUMMARY_NAME).read_text() == text
    assert "own dial" in text and "same dial" in text and "kappa 0.3" in text


# --- the extra line, the reading and the overrides, on the same run -------------------------------


def test_the_extra_line_is_run_labelled_and_reported_beside_the_main_line(same):
    _, out = same
    document = json.loads((out / forecast_mix.RESULTS_NAME).read_text())
    assert document["run"]["kappa"] == 0.3 and document["run"]["extra_level"] == EXTRA_LEVEL
    assert document["models"]["main"]["level"] == document["models"]["trial"]["level"] == 0.3
    assert document["models"]["extra"]["level"] == EXTRA_LEVEL
    extra = document["reading"]["extra_line"]
    assert extra["label"] == f"extra line: the mix at level {EXTRA_LEVEL:g}"
    assert extra["in_the_one_line_reading"] is False and extra["main_arm_level"] == 0.3
    assert set(extra["units"]) == set(SETTINGS["reading"]["bars"])
    for unit, entry in document["extra_comparison"].items():
        if entry["status"] == "calibratable":
            assert entry["tiers"]["A"]["dials"], unit  # paired by dial across the two levels
    summary = (out / forecast_mix.SUMMARY_NAME).read_text()
    assert extra["label"] in summary and "never part of the one-line reading" in summary
    assert forecast_mix_reading.RULE in summary and forecast_mix_reading.NO_OTHER_SITES in summary
    assert "Reading (tier A, own dial)" in summary
    assert "One-line reading (level 0.3 only)" in summary
    assert "information only, not read by the rule" in summary  # tier B


def test_the_one_line_reading_never_takes_the_extra_line(build, tmp_path, monkeypatch):
    real = forecast_mix_reading.read_units
    calls = []

    def read_units(comparison, bars):
        found = real(comparison, bars)
        calls.append(1)
        if len(calls) == 2:  # the second call is the extra line's
            found = {u: {**v, "word": forecast_mix_reading.GAIN} for u, v in found.items()}
        return found

    monkeypatch.setattr(forecast_mix_reading, "read_units", read_units)
    code = run_command(build, tmp_path / "out", monkeypatch)
    assert code == 0
    document = json.loads((tmp_path / "out" / forecast_mix.RESULTS_NAME).read_text())
    reading = document["reading"]
    words = [u["word"] for u in reading["extra_line"]["units"].values()]
    assert words and set(words) == {forecast_mix_reading.GAIN}
    assert reading["main_level"]["one_line"] == forecast_mix_reading.MIXED  # equal offers: no gain
    assert reading["main_level"]["one_line"] == forecast_mix_reading.one_line(
        reading["main_level"]["units"]
    )


# --- the overrides ------------------------------------------------------------------------------


class Reached(Exception):
    pass


def reaching_the_matrix(monkeypatch):
    def reached(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(forecast_mix, "load", reached)


@pytest.fixture
def other_files(tmp_path):
    settings = tmp_path / "other-settings.json"
    settings.write_text(json.dumps({**SETTINGS, "splits": 4}))
    matrix = tmp_path / "other-matrix.json"
    matrix.write_text(Path(forecast_mix.MATRIX).read_text() + "\n")
    return {"--settings": str(settings), "--matrix": str(matrix)}


@pytest.mark.parametrize(
    "option,value",
    [
        ("--kappa", "0.4"),
        ("--extra-level", "0.3"),
        ("--splits", "4"),
        ("--main-model", "empirical"),
        ("--trial-model", "empirical"),
        ("--settings", None),
        ("--matrix", None),
    ],
)
def test_each_override_is_refused_without_the_flag_before_the_matrix_is_read(
    option, value, other_files, tmp_path, monkeypatch, capsys
):
    reaching_the_matrix(monkeypatch)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = forecast_mix.main([option, other_files.get(option, value)])
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("COULD NOT RUN") and option in out
    assert forecast_mix.NOT_THE_TRIAL_FLAG in out
    written = ("forecast-mix", after_sealed_run.FILE_NAME, forecast_mix.NOT_THE_TRIAL_FILE)
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(written)]


def test_with_the_flag_an_override_goes_on_and_the_committed_values_need_no_flag(
    other_files, monkeypatch
):
    reaching_the_matrix(monkeypatch)
    with pytest.raises(Reached):  # the flag: the refusal is passed
        forecast_mix.main(["--kappa", "0.4", forecast_mix.NOT_THE_TRIAL_FLAG])
    with pytest.raises(Reached):  # the committed values given out loud are no override
        forecast_mix.main(
            ["--kappa", "0.5", "--extra-level", "0.4", "--splits", "200", "--workers", "1"]
        )
    committed = [
        "--main-model",
        "lgbm_tail",
        "--trial-model",
        "mix_tail_inputs",
        "--settings",
        str(forecast_mix.SETTINGS_FILE),
        "--matrix",
        str(forecast_mix.MATRIX),
    ]
    with pytest.raises(Reached):
        forecast_mix.main(committed)


def test_a_matrix_that_is_not_the_committed_one_is_refused_without_the_flag(
    build, tmp_path, monkeypatch, capsys
):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix_document(NAMES)))
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = forecast_mix.main(["--matrix", str(path)])
    assert code == 2 and "--matrix" in capsys.readouterr().out


def test_the_reference_of_the_check_is_the_committed_file_not_the_file_given(
    build, tmp_path, monkeypatch, capsys
):
    other = tmp_path / "other.json"
    other.write_text(
        json.dumps(
            {
                **SETTINGS,
                "level": 0.3,
                "extra_level": EXTRA_LEVEL,
                "splits": SPLITS,
                "main_model": "empirical",
                "trial_model": "empirical",
            }
        )
    )
    run, tables, matrix = build
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    code = forecast_mix.main(
        ["--settings", str(other), "--matrix", str(matrix), "--tables", str(tables)]
        + ["--truth-run", str(run), forecast_mix.NOT_THE_TRIAL_FLAG]
    )
    assert code == 0, capsys.readouterr().out
    document = json.loads((out / forecast_mix.RESULTS_NAME).read_text())
    assert document["run"]["kappa"] == 0.3 and document["run"]["splits"] == SPLITS
    assert document["definitions"]["committed_settings"] == SETTINGS
    differences = document["settings_check"]["differences"]
    assert differences["level"] == {"run": 0.3, "committed": SETTINGS["level"]}
    assert differences["splits"] == {"run": SPLITS, "committed": SETTINGS["splits"]}
    assert differences["settings_sha256"]["run"] != differences["settings_sha256"]["committed"]
    assert document["settings_check"]["every_value_equals_the_committed_settings"] is False
    assert "--settings" in " ".join(document["not_the_trial"]["overrides"])


# --- 5. a whole run, a trial that forecasts earlier departures ----------------------------------


class Earlier:
    """The empirical model with every departure bound halved: a made-up trial model."""

    name = "empirical"
    fit_units = ()
    reads_other_day_outcomes = True

    def __init__(self, inner):
        self.inner = inner

    def fit(self, frame):
        self.inner.fit(frame)

    def predict(self, frame, kappas):
        table = self.inner.predict(frame, kappas)
        table["d_hat"] = table["d_hat"] * 0.5
        return table


@pytest.fixture(scope="module")
def earlier(build, tmp_path_factory):
    real = learn_run.load_model
    calls = []

    def load_model(name, matrix):
        calls.append(name)
        model = real(name, matrix)
        return model if len(calls) % 2 == 1 else Earlier(model)  # main first, then the trial

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setattr(learn_run, "load_model", load_model)
        out = tmp_path_factory.mktemp("earlier") / "out"
        code = run_command(build, out, monkeypatch)
    return code, out


def pair_differences(main, trial) -> np.ndarray:
    """Per split, the trial's offer minus the main model's, from two lists of per-split entries."""
    return np.array([y["offer_kwh"] - x["offer_kwh"] for x, y in zip(main, trial, strict=True)])


def test_the_paired_figures_are_those_of_the_two_per_split_files(earlier):
    code, out = earlier
    assert code == 0
    document = json.loads((out / forecast_mix.RESULTS_NAME).read_text())
    main = json.loads((out / forecast_mix.side_name("main", "split-offers")).read_text())
    trial = json.loads((out / forecast_mix.side_name("trial", "split-offers")).read_text())
    checked = 0
    for unit, entry in document["comparison"].items():
        if entry["status"] != "calibratable":
            continue
        for tier, block in entry["tiers"].items():
            for name, cell in block["dials"].items():
                a = main["units"][unit]["configurations"][name]["tiers"][tier]
                b = trial["units"][unit]["configurations"][name]["tiers"][tier]
                assert [s["seed"] for s in a] == [s["seed"] for s in b] == [1, 2, 3, 4]
                difference = pair_differences(a, b)
                assert cell["mean_difference_kwh_per_test_day"] == pytest.approx(difference.mean())
                assert cell["splits_trial_offers_more"] == int((difference > 0).sum())
                assert cell["splits_trial_offers_less"] == int((difference < 0).sum())
                checked += 1
            own = block["own_dial"]
            if own is not None:
                picks = block["picked_dial"]
                names = {
                    c["driver_risk"]: n
                    for n, c in main["units"][unit]["configurations"].items()
                    if c["driver_risk"] is not None
                }
                a = main["units"][unit]["configurations"][names[picks["main"]]]["tiers"][tier]
                b = trial["units"][unit]["configurations"][names[picks["trial"]]]["tiers"][tier]
                mean = pair_differences(a, b).mean()
                assert own["mean_difference_kwh_per_test_day"] == pytest.approx(mean)
    assert checked >= 2 * 3  # at least two tiers over three configurations of one unit


def test_each_models_picked_dial_is_that_models_own(earlier):
    _, out = earlier
    document = json.loads((out / forecast_mix.RESULTS_NAME).read_text())
    for side in forecast_mix.SIDES:
        full = json.loads((out / forecast_mix.side_name(side, "dial-choice")).read_text())
        assert document["models"][side]["dial_choice"] == full["dial_choice"]
    for unit, entry in document["comparison"].items():
        if entry["status"] == "calibratable":
            for tier, block in entry["tiers"].items():
                assert (
                    block["picked_dial"]["main"]
                    == (document["models"]["main"]["dial_choice"][unit][tier]["dial"])
                )
                assert (
                    block["picked_dial"]["trial"]
                    == (document["models"]["trial"]["dial_choice"][unit][tier]["dial"])
                )


# --- 6. the committed settings ------------------------------------------------------------------


def prior_looks_entry(opening):
    """The one entry (a bullet line) of `docs/prior-looks.md` that opens with `opening`; it is
    found by its words, never by its line number."""
    lines = (forecast_mix.TOP / "docs" / "prior-looks.md").read_text().splitlines()
    found = [line for line in lines if line.strip().startswith(f"- {opening}")]
    assert len(found) == 1, opening
    return found[0]


def test_the_bars_and_the_extra_level_rest_on_the_prior_looks_entries_found_by_their_words():
    seed_entry = prior_looks_entry(SEED_ENTRY)
    assert "kappa 0.3" in seed_entry
    assert "about 1.5 kWh (8%) at workplaces and 0.5 kWh (30%) at Norway" in seed_entry
    extra_entry = prior_looks_entry(EXTRA_ENTRY)
    assert "workplaces 19.29 (level 0.4" in extra_entry and "Norway 2.38 (0.5" in extra_entry
    assert forecast_mix.bars_of(SETTINGS) == {
        "nrel-from-may-2019": 0.08,
        "norway-development-pooled": 0.30,
    }


def test_the_committed_settings_are_the_ones_the_code_and_the_plan_give():
    assert SETTINGS["level"] == 0.5 and SETTINGS["level"] in driver_risk.LEVELS
    assert SETTINGS["extra_level"] == 0.4 and SETTINGS["extra_level"] in driver_risk.LEVELS
    assert forecast_mix.bars_of(SETTINGS) == {
        "nrel-from-may-2019": 0.08,
        "norway-development-pooled": 0.30,
    }
    for found in SETTINGS["reading"]["bars"].values():
        assert "docs/prior-looks.md" in found["source"]
        assert SEED_ENTRY in found["source"] and "analogy" in found["source"]
        assert "line 34" not in found["source"] and "decisions record" not in found["source"]
    note = SETTINGS["reading"]["bars_note"]
    assert "analogy" in note and "kappa 0.3" in note and "not this trial's measured noise" in note
    assert EXTRA_ENTRY in SETTINGS["sources"]["extra_level"]
    assert "line 34" not in json.dumps(SETTINGS) and "audit" not in json.dumps(SETTINGS["sources"])
    assert SETTINGS["reading"]["no_other_sites"] == forecast_mix_reading.NO_OTHER_SITES
    assert SETTINGS["splits"] == 200
    assert SETTINGS["main_model"] == "lgbm_tail"
    assert SETTINGS["trial_model"] == mix_tail_inputs.MixModel.name
    assert tuple(SETTINGS["members"]) == mix_tail_inputs.MEMBERS
    seed = inspect.signature(Setting.hazard).parameters["seed"].default
    assert SETTINGS["neural_member_seed"] == seed == HazardSettings().seed == 0
    matrix = gate_config.load(forecast_mix.MATRIX)
    for unit in matrix.gate_units:
        assert unit in SETTINGS["units"]
    _, (plain, dials) = dial_choice.family_at(
        matrix, dial_choice.find_family(matrix), SETTINGS["level"]
    )
    assert plain.kappa == SETTINGS["level"] and dials
    _, (extra_plain, extra_dials) = dial_choice.family_at(
        matrix, dial_choice.find_family(matrix), SETTINGS["extra_level"]
    )
    assert extra_plain.kappa == SETTINGS["extra_level"] and extra_dials
    assert set(forecast_mix.bars_of(SETTINGS)) <= set(matrix.gate_units)
    assert forecast_mix.REQUIRED_SETTINGS == (
        "level",
        "extra_level",
        "splits",
        "main_model",
        "trial_model",
        "reading",
    )
    for key in ("sources", "weights", "nan_rule", "headline", "fit_units"):
        assert SETTINGS[key]


def test_there_is_no_weight_to_choose():
    for function in (forecast_average.average_tables, mix_tail_inputs.make_model):
        assert "weight" not in " ".join(inspect.signature(function).parameters)
    options = [a for a in inspect.getsource(forecast_mix.main).split('"') if a.startswith("--")]
    assert options and not any("weight" in option for option in options)
    assert "weights" not in vars(mix_tail_inputs.MixModel)


# --- 7. no sealed data ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", [forecast_mix, forecast_mix_pairs, forecast_mix_reading, forecast_average]
)
def test_the_new_modules_name_no_loader_of_the_sealed_data(module):
    text = Path(module.__file__).read_text()
    assert "heldout" not in text.lower().replace("held-out", "")
    modules = []
    tree = ast.parse(text)
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules += [node.module or "", *(a.name for a in node.names)]
        elif isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
    # A module may rightly import nothing; the file was read and parsed if its tree holds
    # top-level functions or classes that the loaded module also has.
    assert defined and all(hasattr(module, name) for name in defined)
    assert not [m for m in modules if "held" in m.lower() or "sealed_route" in m]
