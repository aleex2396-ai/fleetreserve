"""The command of the forecast trial `energy` (`forecast_energy`) on the made-up gate units. Written
to be run through the test slot. Each test says which edit turns it red.

1. The held-out refusal. A matrix with a gate unit that is not a development unit (every held-out
   name of the project's own list, a training unit, an unknown name) ends with exit 2 before a
   truth build, a table or a model is touched; nothing is written. Red if the `check_units` call of
   `main` is removed or moved after the model is loaded.
2. The settings file is required: a missing, incomplete or wrongly armed one is exit 2 before the
   matrix is read. Red if `load_settings` is not called first.
3. Approved to run (the owner's decision of 24 November 2026): the committed file says true and the
   plain command gets past the guard; a copy of the committed file that says false (the false copy)
   ends the plain command with exit 2 and reads nothing; a settings copy that says true cannot
   grant it, with or without `--not-the-trial`; a copy that does not say true refuses even when the
   committed value is true. Red if `main` reads the approval from the settings file it was given,
   or if the `committed_approval` test is removed or moved after the inputs are read.
4. What was run is what is reported: `settings_check` says whether every value equals the committed
   file's and lists each difference with both values. Red if a key is ignored.
5. The overrides: each is refused without `--not-the-trial` before the matrix is read; with the
   flag the first summary line and every label say so; the committed values given out loud are no
   override; the reference of the check is the committed file even when `--settings` names another.
6. The arms. `with_energy_bounds` changes the energy bounds of every configuration and nothing
   else; the three arms' families have the same names, dials and oracle partners; in a whole run
   the forecaster is asked for the mode `known` in arm 1, `empirical` in arm 2 and `cautious` in
   arm 3 and for no other. Red if an arm is measured with another arm's matrix.
7. The label. A whole run writes the key and the label in every results and per-split file, the
   first lines of the summary (printed and in the file) and the label file, with and without the
   option; a run that is not the trial says so in every one of them. Red if `after_sealed_run.mark`
   is removed for an arm or for the comparison, if `summary_head` is removed, or if `write_note` is
   removed.
8. A whole run. The forecast table is written once, its hash is in every arm, and it holds the
   level-0.05 rows; the paired figures equal the figures worked out again from the per-split
   files of the arms at each arm's own dial; one fit serves the three arms. Red if the cell is
   taken from another dial or if an arm is paired with itself.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gate_fake import COMMIT, gate_units, matrix_document, write_build

from fleetreserve import (
    after_sealed_run,
    dial_choice,
    forecast_energy,
    forecast_energy_reading,
    forecast_model,
    gate_config,
    gate_run,
    learn_run,
    run_verify,
    sealed_route,
)
from fleetreserve.forecast_empirical import CAUTIOUS
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
SETTINGS = json.loads(Path(forecast_energy.SETTINGS_FILE).read_text())
ARMS = forecast_energy.ARMS
HELD_OUT_NAMES = tuple(sealed_route.HELD_OUT_UNITS)  # the project's own list of the names
GATE_UNITS = ("nrel-before-may-2019", "nrel-from-may-2019", "norway-development-pooled")


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
        forecast_energy.NOT_THE_TRIAL_FLAG,
        "--kappa",
        "0.2",
        "--splits",
        str(SPLITS),
        "--model",
        "empirical",
        *extra,
    ]


def approve(monkeypatch, value: bool = True) -> None:
    """The way a test approves: the function that reads the committed value gives `value`."""
    monkeypatch.setattr(forecast_energy, "committed_approval", lambda: value)


def forbid(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a truth build, a table or a model was touched")

    monkeypatch.setattr(gate_run, "check_inputs", forbidden)
    monkeypatch.setattr(forecast_energy, "load_unit_days", forbidden)
    monkeypatch.setattr(learn_run, "load_model", forbidden)


class Reached(Exception):
    pass


def reaching_the_matrix(monkeypatch):
    def reached(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(forecast_energy, "load", reached)


def files(folder: Path) -> dict:
    return {p.name: p.read_text() for p in sorted(folder.iterdir()) if p.is_file()}


# --- 1. the held-out refusal ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["caltech", "no-such-unit", *HELD_OUT_NAMES, *TRAIN_UNITS])
def test_a_matrix_with_a_unit_that_is_not_a_development_unit_is_refused_first(
    build, tmp_path, monkeypatch, capsys, name
):
    document = matrix_document(NAMES)
    document["gate_units"] = ["nrel-from-may-2019", name]
    document["rotation_whatif"] = name in TRAIN_UNITS  # a training unit parses only so
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(document))
    approve(monkeypatch)
    forbid(monkeypatch)
    run, tables, _ = build
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = forecast_energy.main(
        [
            "--matrix",
            str(path),
            "--tables",
            str(tables),
            "--truth-run",
            str(run),
            forecast_energy.NOT_THE_TRIAL_FLAG,
        ]
    )
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("COULD NOT RUN") and name in out
    assert [p.name for p in tmp_path.iterdir()] == ["matrix.json"]


def test_every_held_out_unit_name_is_refused_and_none_is_a_development_unit():
    assert HELD_OUT_NAMES and "OSL_T" in HELD_OUT_NAMES
    assert not set(HELD_OUT_NAMES) & set(DEVELOPMENT_UNITS)
    for name in HELD_OUT_NAMES:
        with pytest.raises(CouldNotRun, match=name):
            forecast_energy.forecast_mix.check_units(["nrel-from-may-2019", name])


def test_the_committed_matrix_passes_the_check_with_its_three_development_units():
    matrix = gate_config.load(forecast_energy.MATRIX)
    assert forecast_energy.forecast_mix.check_units(matrix.gate_units) == GATE_UNITS
    assert set(matrix.gate_units) <= set(DEVELOPMENT_UNITS)


# --- 2. the settings file -------------------------------------------------------------------------


def test_a_missing_or_incomplete_settings_file_could_not_run_before_the_matrix_is_read(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("the matrix was read")

    monkeypatch.setattr(forecast_energy, "load", forbidden)
    code = forecast_energy.main(["--settings", str(tmp_path / "none.json")])
    assert code == 2 and "settings file" in capsys.readouterr().out
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"level": 0.2}))
    code = forecast_energy.main(["--settings", str(partial)])
    out = capsys.readouterr().out
    assert code == 2 and "lacks" in out and "splits" in out and "approved_to_run" in out
    for name, changes in (
        ("arms.json", {"arms": ["known", "cautious", "empirical"]}),
        ("reading.json", {"reading": {"bars": SETTINGS["reading"]["bars"]}}),
        ("bars.json", {"reading": {"loss_limits": SETTINGS["reading"]["loss_limits"]}}),
    ):
        path = tmp_path / name
        path.write_text(json.dumps({**SETTINGS, **changes}))
        code = forecast_energy.main(["--settings", str(path)])
        out = capsys.readouterr().out
        assert code == 2 and out.startswith("COULD NOT RUN") and name in out, name


# --- 3. approved to run ---------------------------------------------------------------------------


@pytest.fixture
def unapproved_committed_file(tmp_path, monkeypatch):
    """The committed file replaced by a copy that says `approved_to_run: false` (the false copy),
    so the refusal stays tested while the real file says true."""
    path = tmp_path / "committed-unapproved.json"
    path.write_text(json.dumps({**SETTINGS, "approved_to_run": False}))
    monkeypatch.setattr(forecast_energy, "SETTINGS_FILE", path)
    return path


@pytest.fixture(scope="module")
def approved(tmp_path_factory):
    """A copy of the committed settings that says true. It approves nothing by itself."""
    path = tmp_path_factory.mktemp("approved") / "settings.json"
    path.write_text(json.dumps({**SETTINGS, "approved_to_run": True}))
    return path


def test_the_plain_command_with_the_committed_file_gets_past_the_approval_guard(
    tmp_path, monkeypatch
):
    assert SETTINGS["approved_to_run"] is True
    assert forecast_energy.committed_approval() is True
    forbid(monkeypatch)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    with pytest.raises(AssertionError, match="a truth build, a table or a model was touched"):
        forecast_energy.main([])
    assert list(tmp_path.iterdir()) == []


def test_the_plain_command_with_an_unapproved_committed_file_ends_with_exit_2_and_reads_nothing(
    unapproved_committed_file, tmp_path, monkeypatch, capsys
):
    forbid(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out_dir))
    code = forecast_energy.main([])
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("COULD NOT RUN") and "not approved to run" in out
    assert "committed settings file" in out
    assert list(out_dir.iterdir()) == []


def test_a_settings_copy_that_says_true_does_not_get_past_an_unapproved_committed_file(
    build, approved, unapproved_committed_file, tmp_path, monkeypatch, capsys
):
    forbid(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out_dir))
    code = forecast_energy.main(arguments(build, "--settings", str(approved)))
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("COULD NOT RUN") and "not approved to run" in out
    assert "cannot grant" in out
    assert list(out_dir.iterdir()) == []


def test_no_option_turns_the_committed_value_true(build, approved, tmp_path, monkeypatch, capsys):
    approve(monkeypatch, False)
    forbid(monkeypatch)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    flag = forecast_energy.NOT_THE_TRIAL_FLAG
    routes = [
        [],
        arguments(build, "--settings", str(approved)),
        ["--settings", str(approved), flag],
    ]
    for route in routes:
        code = forecast_energy.main(route)
        out = capsys.readouterr().out
        assert code == 2 and "not approved to run" in out, route
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("flag", [False, "yes", 1, None])
def test_a_copy_that_does_not_say_true_refuses_even_when_the_committed_value_is_true(
    flag, tmp_path, monkeypatch, capsys
):
    path = tmp_path / "settings-copy.json"
    path.write_text(json.dumps({**SETTINGS, "approved_to_run": flag}))
    approve(monkeypatch)
    forbid(monkeypatch)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = forecast_energy.main(["--settings", str(path), forecast_energy.NOT_THE_TRIAL_FLAG])
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("COULD NOT RUN") and "not approved to run" in out
    assert [p.name for p in tmp_path.iterdir()] == ["settings-copy.json"]


# --- 4. what was run is what is reported ----------------------------------------------------------

HASHES = {"matrix_sha256": "m" * 64, "settings_sha256": "s" * 64}


def committed_run() -> dict:
    return {
        "level": SETTINGS["level"],
        "rate_cap_level": SETTINGS["rate_cap_level"],
        "need_level": SETTINGS["need_level"],
        "splits": SETTINGS["splits"],
        "model": SETTINGS["model"],
        "arms": list(SETTINGS["arms"]),
        **HASHES,
    }


def test_a_run_with_every_committed_value_is_reported_as_the_committed_one():
    check = forecast_energy.settings_check(SETTINGS, committed_run(), HASHES)
    assert check == {
        "every_value_equals_the_committed_settings": True,
        "differences": {},
        "inputs_that_differ_from_the_defaults": {},
    }
    other = forecast_energy.settings_check(SETTINGS, committed_run(), HASHES, {"tables": "x"})
    assert other["every_value_equals_the_committed_settings"] is True
    assert other["inputs_that_differ_from_the_defaults"] == {"tables": "x"}


@pytest.mark.parametrize(
    "key,value",
    [
        ("level", 0.3),
        ("rate_cap_level", 0.1),
        ("need_level", 0.3),
        ("splits", 4),
        ("model", "empirical"),
        ("arms", ["known", "empirical"]),
        ("matrix_sha256", "x" * 64),
        ("settings_sha256", "x" * 64),
    ],
)
def test_a_changed_value_is_reported_as_changed_with_both_values(key, value):
    ran = committed_run()
    ran[key] = value
    check = forecast_energy.settings_check(SETTINGS, ran, HASHES)
    assert check["every_value_equals_the_committed_settings"] is False
    assert list(check["differences"]) == [key]
    assert check["differences"][key]["run"] == value


# --- 5. the overrides -----------------------------------------------------------------------------


@pytest.fixture
def other_files(tmp_path):
    settings = tmp_path / "other-settings.json"
    settings.write_text(json.dumps({**SETTINGS, "splits": 4}))
    matrix = tmp_path / "other-matrix.json"
    matrix.write_text(Path(forecast_energy.MATRIX).read_text() + "\n")
    return {"--settings": str(settings), "--matrix": str(matrix)}


@pytest.mark.parametrize(
    "option,value",
    [
        ("--kappa", "0.3"),
        ("--splits", "4"),
        ("--model", "empirical"),
        ("--settings", None),
        ("--matrix", None),
    ],
)
def test_each_override_is_refused_without_the_flag_before_the_matrix_is_read(
    option, value, other_files, tmp_path, monkeypatch, capsys
):
    reaching_the_matrix(monkeypatch)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = forecast_energy.main([option, other_files.get(option, value)])
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("COULD NOT RUN") and option in out
    assert forecast_energy.NOT_THE_TRIAL_FLAG in out
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("forecast-energy")]


def test_with_the_flag_an_override_goes_on_and_the_committed_values_need_no_flag(monkeypatch):
    reaching_the_matrix(monkeypatch)
    with pytest.raises(Reached):  # the flag: the refusal is passed
        forecast_energy.main(["--kappa", "0.3", forecast_energy.NOT_THE_TRIAL_FLAG])
    committed = [
        "--kappa",
        "0.2",
        "--splits",
        "200",
        "--model",
        "lgbm_tail",
        "--workers",
        "1",
        "--settings",
        str(forecast_energy.SETTINGS_FILE),
        "--matrix",
        str(forecast_energy.MATRIX),
    ]
    with pytest.raises(Reached):  # the committed values given out loud are no override
        forecast_energy.main(committed)


# --- 6. the arms ----------------------------------------------------------------------------------


def test_with_energy_bounds_changes_the_energy_bounds_of_every_configuration_and_nothing_else():
    matrix = gate_config.load(forecast_energy.MATRIX)
    run_matrix, (plain, dials) = dial_choice.family_at(
        matrix, dial_choice.find_family(matrix), SETTINGS["level"]
    )
    members = [plain, *dials, *(run_matrix.configuration(d.oracle_partner) for d in dials)]
    assert {c.energy_bounds for c in members} == {"known"}  # arm 1 is the matrix as committed
    # (a configuration outside the family, such as the committed forecast-energy comparison, is
    # changed too and is never measured)
    for arm in ARMS:
        changed = forecast_energy.with_energy_bounds(matrix, arm)
        assert {c.energy_bounds for c in changed.configurations} == {arm}
        back = tuple(
            replace(c, energy_bounds=o.energy_bounds)
            for c, o in zip(changed.configurations, matrix.configurations, strict=True)
        )
        assert replace(changed, configurations=back) == matrix
    with pytest.raises(gate_config.MatrixError, match="energy_bounds"):
        forecast_energy.with_energy_bounds(matrix, "cautious-ish")


def test_the_three_arms_have_the_same_family_at_the_trials_level():
    matrix = gate_config.load(forecast_energy.MATRIX)
    seen = {}
    for arm in ARMS:
        arm_matrix = forecast_energy.with_energy_bounds(matrix, arm)
        run_matrix, (plain, dials) = dial_choice.family_at(
            arm_matrix, dial_choice.find_family(arm_matrix), SETTINGS["level"]
        )
        assert plain.kappa == SETTINGS["level"] and dials
        for c in (plain, *dials, *(run_matrix.configuration(d.oracle_partner) for d in dials)):
            assert c.energy_bounds == arm
        seen[arm] = (
            plain.name,
            [(d.name, d.driver_risk, d.oracle_partner) for d in dials],
        )
    assert seen["known"] == seen["empirical"] == seen["cautious"]


@pytest.fixture(scope="module")
def whole(build, tmp_path_factory):
    """A whole run (not the trial: the model `empirical`, level 0.2, 4 splits) with the forecaster's
    modes recorded per arm."""
    modes: dict[str, set] = {}
    current: list[str] = []
    real_measure = forecast_energy.measure
    real_forecast = forecast_model.TableForecaster.forecast_car

    def measure(arm, *args, **kwargs):
        current.append(arm)
        return real_measure(arm, *args, **kwargs)

    def forecast_car(self, truth, i, kappa, mode, departure="empirical"):
        modes.setdefault(current[-1], set()).add(mode)
        return real_forecast(self, truth, i, kappa, mode, departure)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        approve(monkeypatch)
        monkeypatch.setattr(forecast_energy, "measure", measure)
        monkeypatch.setattr(forecast_model.TableForecaster, "forecast_car", forecast_car)
        out = tmp_path_factory.mktemp("whole") / "out"
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        code = forecast_energy.main(arguments(build))
    return code, out, modes


def test_each_arm_asks_the_forecaster_for_its_own_energy_bounds_and_no_other(whole):
    code, _, modes = whole
    assert code == 0
    assert modes == {"known": {"known"}, "empirical": {"empirical"}, "cautious": {CAUTIOUS}}
    assert list(modes) == list(ARMS)  # measured in the order of the settings


# --- 7. the label ---------------------------------------------------------------------------------


def test_a_whole_run_writes_every_output_with_the_label(whole):
    code, out, _ = whole
    assert code == 0
    found = files(out)
    expected = {
        forecast_energy.RESULTS_NAME,
        forecast_energy.SUMMARY_NAME,
        forecast_energy.TIMING_NAME,
        forecast_energy.TABLE_NAME,
        after_sealed_run.FILE_NAME,
        forecast_energy.NOT_THE_TRIAL_FILE,
        *(
            forecast_energy.side_name(arm, kind)
            for arm in ARMS
            for kind in ("dial-choice", "split-offers")
        ),
    }
    assert set(found) == expected
    labelled = 0
    for name in expected:
        if name.endswith(".json") and name != forecast_energy.TIMING_NAME:
            document = json.loads(found[name])
            assert document[after_sealed_run.KEY] == after_sealed_run.record(), name
            assert document["label"].startswith(forecast_energy.NOT_THE_TRIAL), name
            assert after_sealed_run.WHAT in document["label"], name
            labelled += 1
    assert labelled == 1 + 2 * len(ARMS)
    lines = found[forecast_energy.SUMMARY_NAME].splitlines()
    assert lines[0].startswith(forecast_energy.NOT_THE_TRIAL)
    assert lines[1:3] == [after_sealed_run.WHAT, after_sealed_run.SENTENCE]
    label_lines = found[after_sealed_run.FILE_NAME].splitlines()
    assert label_lines[:2] == lines[1:3]
    assert label_lines[2:] == [forecast_energy.NOT_THE_TRIAL]
    assert found[forecast_energy.NOT_THE_TRIAL_FILE].startswith(forecast_energy.NOT_THE_TRIAL)
    assert forecast_energy.UNCERTAINTY in lines and forecast_energy.ONE_FIT in lines
    assert forecast_energy_reading.RULE in lines
    assert forecast_energy_reading.NO_OTHER_SITES in lines


def test_a_run_without_the_option_writes_the_label_file_as_the_other_trials_do(
    build, tmp_path, monkeypatch
):
    """Without the not-the-trial option the label file holds its two lines and nothing else, and no
    not-the-trial file is written. The made-up matrix and options are not the committed ones, so
    `overrides` is made to find none; nothing else differs."""
    monkeypatch.setattr(forecast_energy, "overrides", lambda args, committed: [])
    approve(monkeypatch)
    out = tmp_path / "plain"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    given = arguments(build)
    given.remove(forecast_energy.NOT_THE_TRIAL_FLAG)
    assert forecast_energy.main(given) == 0
    label_text = (out / after_sealed_run.FILE_NAME).read_text()
    assert label_text == f"{after_sealed_run.WHAT}\n{after_sealed_run.SENTENCE}\n"
    assert not (out / forecast_energy.NOT_THE_TRIAL_FILE).exists()
    summary = (out / forecast_energy.SUMMARY_NAME).read_text().splitlines()
    assert summary[:2] == [after_sealed_run.WHAT, after_sealed_run.SENTENCE]
    for arm in ARMS:
        document = json.loads((out / forecast_energy.side_name(arm, "dial-choice")).read_text())
        assert document[after_sealed_run.KEY] == after_sealed_run.record()
        assert not document["label"].startswith(forecast_energy.NOT_THE_TRIAL)


# --- 8. a whole run -------------------------------------------------------------------------------


def test_the_printed_summary_is_the_summary_file(build, tmp_path, monkeypatch, capsys):
    approve(monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    assert forecast_energy.main(arguments(build)) == 0
    text = capsys.readouterr().out
    assert (out / forecast_energy.SUMMARY_NAME).read_text() == text
    assert "cautious minus empirical" in text and "One-line reading:" in text
    assert "Reading (tier A)" in text and "information only, not read by the rule" in text
    assert "nrel-before-may-2019 (not calibratable" in text and "the unit cannot be split" in text


def test_the_verdict_line_says_the_word_the_note_and_the_losses_it_read():
    losses = {"slot": 0.05, "driver": 0.02}
    gain = {"share": 0.2, "bar": 0.08, "word": forecast_energy_reading.GAIN, "note": None}
    line = forecast_energy._verdict_line(gain, losses)
    assert "20.0%" in line and "bar 8%" in line and "slot 0.050, driver 0.020" in line
    assert line.endswith(forecast_energy_reading.GAIN)
    stopped = {**gain, "word": forecast_energy_reading.NONE, "note": "above the bar, but over"}
    assert forecast_energy._verdict_line(stopped, losses).endswith("(above the bar, but over)")
    unjudged = {"share": 0.9, "bar": None, "word": None, "note": forecast_energy_reading.NOT_JUDGED}
    line = forecast_energy._verdict_line(unjudged, losses)
    assert "not judged" in line and "figures printed only" in line
    assert "none" in forecast_energy._verdict_line(gain, None)


def test_one_table_is_written_for_the_three_arms_and_holds_the_level_0_05_rows(whole):
    _, out, _ = whole
    document = json.loads((out / forecast_energy.RESULTS_NAME).read_text())
    table = out / forecast_energy.TABLE_NAME
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    for arm in ARMS:
        record = document["arms"][arm]["forecast_table"]
        assert record == {"file": table.name, "rows": record["rows"], "sha256": digest}
        full = json.loads((out / forecast_energy.side_name(arm, "dial-choice")).read_text())
        assert full["forecast_table"] == record and full["energy_bounds"] == arm
        assert full["model"] == "empirical"
    read = forecast_model.read_table(table)
    assert {0.05, 0.1, 0.2} == set(read["kappa"])


def test_the_run_record_says_what_was_run(whole):
    _, out, _ = whole
    document = json.loads((out / forecast_energy.RESULTS_NAME).read_text())
    run = document["run"]
    assert run["kappa"] == 0.2 and run["splits"] == SPLITS and run["model"] == "empirical"
    assert run["rate_cap_level"] == 0.05 and run["need_level"] == 0.2
    assert run["arms"] == list(ARMS) and run["loss_limits"] == {"slot": 0.1, "driver": 0.05}
    check = document["settings_check"]
    assert check["every_value_equals_the_committed_settings"] is False
    for key in ("splits", "model", "matrix_sha256"):
        assert key in check["differences"], key
    assert "level" not in check["differences"]  # the committed level 0.2 was given
    assert check["inputs_that_differ_from_the_defaults"].keys() == {"tables", "truth_run"}
    assert document["definitions"]["committed_settings"] == SETTINGS
    assert set(document["reading"]["units"]) == set(GATE_UNITS)
    assert document["reading"]["units"]["nrel-before-may-2019"]["word"] is None
    assert document["reading"]["one_line"] == forecast_energy_reading.one_line(
        document["reading"]["units"]
    )


def names_by_dial(split_offers: dict, unit: str) -> dict:
    return {
        c["driver_risk"]: n
        for n, c in split_offers["units"][unit]["configurations"].items()
        if c["driver_risk"] is not None
    }


def test_the_paired_figures_are_those_of_the_per_split_files_at_each_arms_own_dial(whole):
    code, out, _ = whole
    assert code == 0
    document = json.loads((out / forecast_energy.RESULTS_NAME).read_text())
    offers = {
        arm: json.loads((out / forecast_energy.side_name(arm, "split-offers")).read_text())
        for arm in ARMS
    }
    checked = 0
    for unit, entry in document["comparison"].items():
        if entry["status"] != "calibratable":
            assert set(entry) == {"status", "counting_days"}
            continue
        for tier, block in entry["tiers"].items():
            if block["cell"] is None:
                continue
            series = {}
            for arm in ARMS:
                dial = block["picked_dial"][arm]
                name = names_by_dial(offers[arm], unit)[dial]
                found = offers[arm]["units"][unit]["configurations"][name]["tiers"][tier]
                assert [s["seed"] for s in found] == list(range(1, SPLITS + 1))
                series[arm] = np.array([s["offer_kwh"] for s in found])
            difference = series["cautious"] - series["empirical"]
            cell = block["cell"]
            assert cell["mean_difference_kwh_per_test_day"] == pytest.approx(difference.mean())
            assert cell["splits_trial_offers_more"] == int((difference > 0).sum())
            assert cell["splits_trial_offers_less"] == int((difference < 0).sum())
            means = block["mean_offer_kwh_per_test_day"]
            for arm in ARMS:
                assert means[arm] == pytest.approx(series[arm].mean())
            checked += 1
    assert checked >= 2  # a unit that can be split, in both tiers


def test_each_arms_picked_dial_is_that_arms_own(whole):
    _, out, _ = whole
    document = json.loads((out / forecast_energy.RESULTS_NAME).read_text())
    for arm in ARMS:
        full = json.loads((out / forecast_energy.side_name(arm, "dial-choice")).read_text())
        assert document["arms"][arm]["dial_choice"] == full["dial_choice"]
    for unit, entry in document["comparison"].items():
        if entry["status"] == "calibratable":
            for tier, block in entry["tiers"].items():
                for arm in ARMS:
                    assert (
                        block["picked_dial"][arm]
                        == (document["arms"][arm]["dial_choice"][unit][tier]["dial"])
                    )
