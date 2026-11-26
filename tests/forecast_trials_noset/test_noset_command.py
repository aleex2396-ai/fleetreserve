"""The command of the forecast trial `noset` (`forecast_noset`) on the made-up gate units. Written,
not run. Each test says which edit turns it red.

1. The refusal. A matrix with a gate unit that is not a development unit (a training unit, an
   unknown name, every held-out name) ends with exit 2 before a truth build, a table or a model is
   touched; nothing is written. Red if the `check_units` call of `main` is removed or moved after
   the model is loaded. The committed matrix passes with its three development units.
1b. Approved to run (the owner's decision of 24 November 2026). The approval is the COMMITTED
   settings file's alone (`committed_approval`): the committed file says true and the plain command
   gets past the guard; with the committed file replaced by a copy that says false, the plain
   command ends with exit 2 and reads nothing, and a settings copy that says true, given with
   `--settings` and `--not-the-trial`, does not get past the guard either; a copy that says
   false refuses even when the committed value is true. Tests that need a run approve by replacing
   `committed_approval`, never by a settings copy. Red if `main` reads the approval from the
   settings file it was given, or if the `committed_approval` test is removed or moved after the
   first read of a table, a truth build or a model.
2. The settings file is required and checked (missing, incomplete, no model for an arm, seeds
   listed twice, no bars): exit 2 before the matrix is read. Red if `load_settings` is not called
   first in `main` or loses one of its checks.
3. The overrides. Each option that changes a value is refused without `--not-the-trial` before the
   matrix is read; the committed values given out loud are no override; with the flag the first
   summary line and every label say so; the check's reference is the committed file even when
   `--settings` names another; `--tables` and `--truth-run` are reported, not refused. Red if
   `overrides` skips an option.
4. What was run is what is reported: `settings_check` names every difference with both values.
   Red if `settings_check` ignores a key or `run` is read from the settings file again.
5. A whole run (a made-up model standing for each arm; the no-encoder arms forecast earlier
   departures): every file is written with the label; the seeds are applied to the neural arms
   only and in order; the paired figures of every cell on both lines equal the figures worked out
   again here from the per-run files, the plan's rule included. Red if a seed is not applied, if
   the pairing is by position in another order, if an arm is read from another arm's document, if
   the judged line reads the arm S1, or if `mark` / `summary_head` / `write_note` is removed.
6. A refused arm: its runs are recorded as failed, the others stay, the reading says NOT_READ,
   exit 1. Red if the `except` of the run loop is removed or if a failed arm is read as zero.
7. The dial of the check line is chosen from calibration days alone: changing the losses of the
   test days changes no choice (`site_dial_whatif.split_arms`, the arm S1). Red if the choosing
   offers of S1 read the test days (`pick` replaced by all days in `split_arms`).
8. The committed settings are the ones the code and the plan give; every entry of
   `docs/prior-looks.md` is cited by a phrase of its own words that occurs exactly once there, and
   by no line number; the README carries the one command and says "approved to run" at the top
   of the trial's section.
9. The new modules name no loader of the sealed data and import none.
10. The differences between the neural models are those the README states, read from the models'
   source text (the part list, the extra input columns), not from the README.
"""

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_units, matrix_document, write_build

from fleetreserve import (
    after_sealed_run,
    dial_choice,
    dial_splits,
    driver_risk,
    forecast_noset,
    forecast_noset_pairs,
    forecast_noset_reading,
    gate_config,
    gate_run,
    hazard_parts,
    learn_run,
    run_verify,
    sealed_route,
    site_dial_whatif,
)
from fleetreserve.truth_build import DEVELOPMENT_UNITS, TRAIN_UNITS
from fleetreserve.truth_verify import CouldNotRun

PLAIN = ("main-rounded-hour-k0.3-empirical", "main-rounded-hour-k0.3-oracle")
LOW = ("main-rounded-hour-k0.3-eps0.03-empirical", "main-rounded-hour-k0.3-eps0.03-oracle")
HIGH = ("main-rounded-hour-k0.3-eps0.1-empirical", "main-rounded-hour-k0.3-eps0.1-oracle")
NAMES = PLAIN + LOW + HIGH  # kappa 0.3 and the dials 0.03 and 0.1: the family of these runs
SPLITS = 6
SEEDS = (1, 2)
RUN_NAMES = [
    "tail",
    "tuned-seed1",
    "tuned-seed2",
    "inputs-seed1",
    "inputs-seed2",
    "noset-seed1",
    "noset-seed2",
]
SETTINGS = json.loads(Path(forecast_noset.SETTINGS_FILE).read_text())
JUDGED = ("nrel-from-may-2019", "norway-development-pooled")


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


@pytest.fixture(scope="module")
def approved(tmp_path_factory):
    """A copy of the committed settings that says `approved_to_run: true`. It does not approve
    anything by itself: only `committed_approval` does (see `approve`)."""
    path = tmp_path_factory.mktemp("approved") / "settings.json"
    path.write_text(json.dumps({**SETTINGS, "approved_to_run": True}))
    return path


def approve(monkeypatch, value: bool = True) -> None:
    """The way a test approves: the function that reads the committed value gives `value`."""
    monkeypatch.setattr(forecast_noset, "committed_approval", lambda: value)


def settings_copy(folder: Path, **changes) -> Path:
    path = folder / "settings-copy.json"
    path.write_text(json.dumps({**SETTINGS, **changes}))
    return path


def arguments(build, settings, *extra) -> list[str]:
    run, tables, matrix = build
    return [
        "--matrix",
        str(matrix),
        "--tables",
        str(tables),
        "--truth-run",
        str(run),
        "--settings",
        str(settings),
        forecast_noset.NOT_THE_TRIAL_FLAG,
        "--kappa",
        "0.3",
        "--splits",
        str(SPLITS),
        "--seeds",
        ",".join(map(str, SEEDS)),
        "--model-tail",
        "empirical",
        "--model-tuned",
        "empirical",
        "--model-inputs",
        "empirical",
        "--model-noset",
        "empirical",
        *extra,
    ]


def forbid(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a truth build, a table or a model was touched")

    monkeypatch.setattr(gate_run, "check_inputs", forbidden)
    monkeypatch.setattr(forecast_noset, "load_unit_days", forbidden)
    monkeypatch.setattr(learn_run, "load_model", forbidden)


class Reached(Exception):
    pass


def reaching_the_matrix(monkeypatch):
    def reached(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(forecast_noset, "load", reached)


# --- 1. the refusal -----------------------------------------------------------------------------

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
    code = forecast_noset.main(
        [
            "--matrix",
            str(path),
            "--tables",
            str(tables),
            "--truth-run",
            str(run),
            forecast_noset.NOT_THE_TRIAL_FLAG,
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
            forecast_noset.forecast_mix.check_units(["nrel-from-may-2019", name])


def test_the_committed_matrix_passes_the_check_with_its_three_development_units():
    matrix = gate_config.load(forecast_noset.MATRIX)
    assert forecast_noset.forecast_mix.check_units(matrix.gate_units) == GATE_UNITS
    assert set(matrix.gate_units) <= set(DEVELOPMENT_UNITS)
    for name in TRAIN_UNITS:
        with pytest.raises(CouldNotRun, match="not a development unit"):
            forecast_noset.forecast_mix.check_units([name])


# --- 1b. not approved to run --------------------------------------------------------------------


def test_the_plain_command_with_the_committed_file_gets_past_the_approval_guard(
    tmp_path, monkeypatch
):
    """The committed settings file says `approved_to_run: true` (the owner's decision of 24 November
    2026): the plain command is not refused by the guard and goes on to the first model load, where
    `forbid` raises. Red if the committed file says anything but true, or if `main` refuses an
    approved committed file."""
    assert SETTINGS["approved_to_run"] is True
    assert forecast_noset.committed_approval() is True
    forbid(monkeypatch)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    with pytest.raises(AssertionError, match="a truth build, a table or a model was touched"):
        forecast_noset.main([])
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def unapproved_committed_file(tmp_path, monkeypatch):
    """The committed file replaced by a copy that says `approved_to_run: false`, so the refusal
    stays tested now that the real file says true."""
    path = tmp_path / "committed-unapproved.json"
    path.write_text(json.dumps({**SETTINGS, "approved_to_run": False}))
    monkeypatch.setattr(forecast_noset, "SETTINGS_FILE", path)
    return path


def test_the_plain_command_with_an_unapproved_committed_file_ends_with_exit_2_and_reads_nothing(
    unapproved_committed_file, tmp_path, monkeypatch, capsys
):
    """The plain command (no option) reads the committed settings file; with a file that says
    `approved_to_run: false`: exit 2, and no truth build, table or model is touched (`forbid`
    raises if one is) and nothing is written. Red if the `committed_approval` test is removed from
    `main` (the command then goes on to the truth build and `forbid` raises), or if it is placed
    after `gate_run.check_inputs`."""
    forbid(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out_dir))
    code = forecast_noset.main([])
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("COULD NOT RUN") and "not approved to run" in out
    assert "committed settings file" in out
    assert list(out_dir.iterdir()) == []


def test_a_settings_copy_that_says_true_does_not_get_past_an_unapproved_committed_file(
    build, approved, unapproved_committed_file, tmp_path, monkeypatch, capsys
):
    """The bypass: `--settings` names a copy with `approved_to_run: true` and `--not-the-trial` is
    given, so the overrides are allowed. The committed file says false: exit 2, and nothing is
    touched. Red if `main` tests `settings["approved_to_run"]` (the copy's) in place of
    `committed_approval()`: the run then goes on and `forbid` raises."""
    assert json.loads(approved.read_text())["approved_to_run"] is True
    forbid(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out_dir))
    code = forecast_noset.main(arguments(build, approved))
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("COULD NOT RUN") and "not approved to run" in out
    assert "cannot grant" in out
    assert list(out_dir.iterdir()) == []


def test_the_guard_reads_the_committed_value_and_no_option_changes_it(
    build, approved, tmp_path, monkeypatch, capsys
):
    """With the committed value replaced by False, every route ends with exit 2 before anything is
    read: the plain command, the copy that says true with every override flag, and the copy with
    the flag. Red if any option or file given to `main` can turn the committed value true."""
    approve(monkeypatch, False)
    forbid(monkeypatch)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    flag = forecast_noset.NOT_THE_TRIAL_FLAG
    routes = [[], arguments(build, approved), ["--settings", str(approved), flag]]
    for route in routes:
        code = forecast_noset.main(route)
        out = capsys.readouterr().out
        assert code == 2 and "not approved to run" in out, route
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("flag", [False, "yes", 1, None])
def test_a_copy_that_does_not_say_true_refuses_even_when_the_committed_value_is_true(
    flag, tmp_path, monkeypatch, capsys
):
    """The committed value is replaced by True; the settings file given says anything but true:
    exit 2 before a model is loaded. Red if the `settings["approved_to_run"]` test of `main` is
    removed."""
    settings = settings_copy(tmp_path, approved_to_run=flag)
    approve(monkeypatch)
    forbid(monkeypatch)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = forecast_noset.main(["--settings", str(settings), forecast_noset.NOT_THE_TRIAL_FLAG])
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("COULD NOT RUN") and "not approved to run" in out
    assert [p.name for p in tmp_path.iterdir()] == ["settings-copy.json"]


def test_the_committed_approval_is_read_from_the_committed_file_only(tmp_path, monkeypatch):
    """`committed_approval` reads `SETTINGS_FILE`: a true there gives True, anything else False;
    no argument can change the file. Red if it reads another path or compares with `bool`."""
    assert forecast_noset.committed_approval() is True  # the committed file says true
    for value, expected in ((True, True), (False, False), ("yes", False), (1, False)):
        path = tmp_path / f"approval-{expected}-{value!r}.json"
        path.write_text(json.dumps({**SETTINGS, "approved_to_run": value}))
        monkeypatch.setattr(forecast_noset, "SETTINGS_FILE", path)
        assert forecast_noset.committed_approval() is expected


def test_a_settings_file_without_the_key_is_refused_as_incomplete(tmp_path, monkeypatch, capsys):
    document = {k: v for k, v in SETTINGS.items() if k != "approved_to_run"}
    path = tmp_path / "no-flag.json"
    path.write_text(json.dumps(document))
    reaching_the_matrix(monkeypatch)
    code = forecast_noset.main(["--settings", str(path), forecast_noset.NOT_THE_TRIAL_FLAG])
    out = capsys.readouterr().out
    assert code == 2 and "lacks approved_to_run" in out


# --- 2. the settings file -----------------------------------------------------------------------


def test_a_missing_or_incomplete_settings_file_could_not_run_before_the_matrix_is_read(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("the matrix was read")

    monkeypatch.setattr(forecast_noset, "load", forbidden)
    code = forecast_noset.main(["--settings", str(tmp_path / "none.json")])
    assert code == 2 and "settings file" in capsys.readouterr().out
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"level": 0.5}))
    code = forecast_noset.main(["--settings", str(partial)])
    out = capsys.readouterr().out
    assert code == 2 and "lacks" in out and "splits" in out


@pytest.mark.parametrize(
    "changes,text",
    [
        ({"neural_seeds": [1, 1, 2]}, "seeds that are not distinct"),
        ({"neural_seeds": []}, "seeds that are not distinct"),
        ({"neural_seeds": [1, "2"]}, "seeds that are not distinct"),
        ({"neural_seeds": [-1, 2]}, "seeds that are not distinct"),
        ({"arms": {"tail": "lgbm_tail"}}, "must name a model for each"),
        ({"arms": ["lgbm_tail"]}, "must name a model for each"),
        ({"reading": {"bars": {}}}, "no bars"),
        ({"reading": []}, "no bars"),
    ],
)
def test_an_unusable_settings_file_is_refused_before_the_matrix_is_read(
    changes, text, tmp_path, monkeypatch, capsys
):
    path = settings_copy(tmp_path, **changes)
    reaching_the_matrix(monkeypatch)
    code = forecast_noset.main(["--settings", str(path), forecast_noset.NOT_THE_TRIAL_FLAG])
    out = capsys.readouterr().out
    assert code == 2 and text in out


# --- 3. the overrides ---------------------------------------------------------------------------


@pytest.fixture
def other_files(tmp_path):
    settings = tmp_path / "other-settings.json"
    settings.write_text(json.dumps({**SETTINGS, "splits": 4}))
    matrix = tmp_path / "other-matrix.json"
    matrix.write_text(Path(forecast_noset.MATRIX).read_text() + "\n")
    return {"--settings": str(settings), "--matrix": str(matrix)}


@pytest.mark.parametrize(
    "option,value",
    [
        ("--kappa", "0.4"),
        ("--splits", "4"),
        ("--seeds", "1,2"),
        ("--seeds", "1,2,4"),
        ("--model-tail", "empirical"),
        ("--model-tuned", "empirical"),
        ("--model-inputs", "empirical"),
        ("--model-noset", "empirical"),
        ("--settings", None),
        ("--matrix", None),
    ],
)
def test_each_override_is_refused_without_the_flag_before_the_matrix_is_read(
    option, value, other_files, tmp_path, monkeypatch, capsys
):
    reaching_the_matrix(monkeypatch)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = forecast_noset.main([option, other_files.get(option, value)])
    out = capsys.readouterr().out
    assert code == 2 and out.startswith("COULD NOT RUN") and option in out
    assert forecast_noset.NOT_THE_TRIAL_FLAG in out
    written = ("forecast-noset", after_sealed_run.FILE_NAME, forecast_noset.NOT_THE_TRIAL_FILE)
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(written)]


def test_with_the_flag_an_override_goes_on_and_the_committed_values_need_no_flag(
    other_files, monkeypatch
):
    reaching_the_matrix(monkeypatch)
    with pytest.raises(Reached):  # the flag: the refusal is passed
        forecast_noset.main(["--kappa", "0.4", forecast_noset.NOT_THE_TRIAL_FLAG])
    with pytest.raises(Reached):  # the committed values given out loud are no override
        forecast_noset.main(
            ["--kappa", "0.5", "--splits", "200", "--seeds", "3,2,1", "--workers", "1"]
        )
    committed = [
        "--model-tail",
        "lgbm_tail",
        "--model-tuned",
        "hazard_tuned",
        "--model-inputs",
        "hazard_tuned_inputs",
        "--model-noset",
        "hazard_tuned_no_set",
        "--settings",
        str(forecast_noset.SETTINGS_FILE),
        "--matrix",
        str(forecast_noset.MATRIX),
    ]
    with pytest.raises(Reached):
        forecast_noset.main(committed)


def test_the_data_options_are_reported_and_not_refused(tmp_path, monkeypatch):
    """`--tables` and `--truth-run` change the data read: no override, no refusal (the run goes on
    to the matrix), and `data_options` reports them for `settings_check`. Red if `overrides` lists
    them."""
    reaching_the_matrix(monkeypatch)
    with pytest.raises(Reached):
        forecast_noset.main(["--tables", str(tmp_path), "--truth-run", str(tmp_path)])
    args = SimpleNamespace(tables=tmp_path, truth_run=tmp_path)
    assert forecast_noset.forecast_mix.data_options(args) == {
        "tables": str(tmp_path),
        "truth_run": str(tmp_path),
    }
    assert (
        forecast_noset.overrides(
            SimpleNamespace(
                kappa=None,
                splits=None,
                seeds=None,
                model_tail=None,
                model_tuned=None,
                model_inputs=None,
                model_noset=None,
                settings=forecast_noset.SETTINGS_FILE,
                matrix=forecast_noset.MATRIX,
                tables=tmp_path,
                truth_run=tmp_path,
            ),
            SETTINGS,
        )
        == []
    )


def test_seeds_that_are_not_whole_numbers_are_refused(monkeypatch, capsys):
    reaching_the_matrix(monkeypatch)
    for text in ("1,x", "1,1", "-1,2", ""):
        code = forecast_noset.main([f"--seeds={text}", forecast_noset.NOT_THE_TRIAL_FLAG])
        assert code == 2 and "--seeds" in capsys.readouterr().out
    assert forecast_noset.parse_seeds("3, 1,2") == (1, 2, 3)


# --- 4. what was run is what is reported --------------------------------------------------------

COMMITTED_HASHES = {"matrix_sha256": "m" * 64, "settings_sha256": "s" * 64}


def committed_run() -> dict:
    return {
        "level": SETTINGS["level"],
        "splits": SETTINGS["splits"],
        "neural_seeds": list(SETTINGS["neural_seeds"]),
        "models": dict(SETTINGS["arms"]),
        **COMMITTED_HASHES,
    }


def test_a_run_with_every_committed_value_is_reported_as_the_committed_one():
    check = forecast_noset.settings_check(SETTINGS, committed_run(), COMMITTED_HASHES)
    assert check == {
        "every_value_equals_the_committed_settings": True,
        "differences": {},
        "inputs_that_differ_from_the_defaults": {},
    }
    other = forecast_noset.settings_check(
        SETTINGS, committed_run(), COMMITTED_HASHES, {"tables": "elsewhere"}
    )
    assert other["every_value_equals_the_committed_settings"] is True
    assert other["inputs_that_differ_from_the_defaults"] == {"tables": "elsewhere"}


@pytest.mark.parametrize(
    "key,value",
    [
        ("level", 0.3),
        ("splits", 4),
        ("neural_seeds", [1, 2]),
        ("models", {"tail": "a", "tuned": "t", "inputs": "b", "noset": "c"}),
        ("matrix_sha256", "x" * 64),
        ("settings_sha256", "x" * 64),
    ],
)
def test_a_changed_value_is_reported_as_changed_with_both_values(key, value):
    ran = committed_run()
    ran[key] = value
    check = forecast_noset.settings_check(SETTINGS, ran, COMMITTED_HASHES)
    assert check["every_value_equals_the_committed_settings"] is False
    assert list(check["differences"]) == [key]
    assert check["differences"][key]["run"] == value


# --- 5. a whole run -----------------------------------------------------------------------------


class Earlier:
    """The empirical model with every departure bound halved: a made-up no-encoder model."""

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


def fake_seeding(monkeypatch, calls: list, refuse_after: int | None = None):
    """Replace `seeded_model`: it records (model name, seed), hands the fifth and later seeded
    models (the no-encoder arms; the first four seeded calls are the tuned arm's and the inputs
    arm's) back as `Earlier`, and refuses them all when `refuse_after` is 4."""

    def seeded(model, name, seed):
        calls.append((name, seed))
        if refuse_after is not None and len(calls) > refuse_after:
            raise learn_run.Refused("made up: this arm is refused")
        return Earlier(model) if len(calls) > 4 else model

    monkeypatch.setattr(forecast_noset, "seeded_model", seeded)


def run_whole(build, approved, folder: Path, refuse_after=None, plain=False):
    """`plain`: the command without the not-the-trial option. The made-up matrix and options are
    not the committed ones, so `overrides` is made to find none; nothing else differs."""
    calls: list = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        if plain:
            monkeypatch.setattr(forecast_noset, "overrides", lambda args, committed: [])
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        approve(monkeypatch)  # the committed value; the copy `approved` is not what approves
        fake_seeding(monkeypatch, calls, refuse_after)
        folder.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        given = arguments(build, approved)
        if plain:
            given.remove(forecast_noset.NOT_THE_TRIAL_FLAG)
        code = forecast_noset.main(given)
    return code, calls


@pytest.fixture(scope="module")
def whole(build, approved, tmp_path_factory):
    out = tmp_path_factory.mktemp("whole") / "out"
    code, calls = run_whole(build, approved, out)
    return code, out, calls


def results(out: Path) -> dict:
    return json.loads((out / forecast_noset.RESULTS_NAME).read_text())


def test_a_whole_run_writes_every_output_with_the_label(whole):
    code, out, _ = whole
    assert code == 0, (out / forecast_noset.SUMMARY_NAME).read_text()
    found = {p.name: p.read_text() for p in sorted(out.iterdir()) if p.is_file()}
    expected = {
        forecast_noset.RESULTS_NAME,
        forecast_noset.SUMMARY_NAME,
        forecast_noset.TIMING_NAME,
        after_sealed_run.FILE_NAME,
        forecast_noset.NOT_THE_TRIAL_FILE,
        *(forecast_noset.side_name(r, k) for r in RUN_NAMES for k in ("dial-whatif", "forecasts")),
    }
    assert set(found) == expected
    for name in expected:
        if name.endswith(".json") and name != forecast_noset.TIMING_NAME:
            document = json.loads(found[name])
            assert document[after_sealed_run.KEY] == after_sealed_run.record(), name
            assert document["label"].startswith(forecast_noset.NOT_THE_TRIAL), name
            assert after_sealed_run.WHAT in document["label"], name
    lines = found[forecast_noset.SUMMARY_NAME].splitlines()
    assert lines[0].startswith(forecast_noset.NOT_THE_TRIAL)
    assert lines[1:3] == [after_sealed_run.WHAT, after_sealed_run.SENTENCE]
    label_lines = found[after_sealed_run.FILE_NAME].splitlines()
    assert label_lines[:2] == lines[1:3]
    # a run that is not the trial says so in this file too, as the mix command's does
    assert label_lines[2:] == [forecast_noset.NOT_THE_TRIAL]
    assert found[forecast_noset.NOT_THE_TRIAL_FILE].startswith(forecast_noset.NOT_THE_TRIAL)
    for sentence in (
        forecast_noset.UNCERTAINTY,
        forecast_noset.FEW_SEEDS,
        forecast_noset_reading.RULE,
        forecast_noset_reading.NO_OTHER_SITES,
    ):
        assert sentence in lines
    text = found[forecast_noset.SUMMARY_NAME]
    assert "information only, not read by the rule" in text  # tier B
    assert "Overall reading:" in text and "One-line reading (the judged line)" in text
    assert "information: hazard_tuned_inputs against lgbm_tail (not read)" in text
    for sentence in (forecast_noset.PLAN_RULE_NOTE, forecast_noset.CHECK_NOTE):
        assert sentence in lines
    assert "Judged line (the plan's development rule" in text
    assert "Check line (the dial chosen from calibration days alone" in text
    assert "no-encoder model against hazard_tuned (the same model with its set encoder" in text
    assert "(least " in text and ", greatest " in text and "dial picked " in text


def test_a_run_without_the_option_writes_the_label_file_as_it_always_did(build, approved, tmp_path):
    """Without the not-the-trial option the label file holds its two lines and nothing else, and
    no not-the-trial file is written. Together with the whole run above (the option given: a
    third line), red if the third line is written whatever the option."""
    out = tmp_path / "plain"
    code, _ = run_whole(build, approved, out, plain=True)
    assert code == 0, (out / forecast_noset.SUMMARY_NAME).read_text()
    label_text = (out / after_sealed_run.FILE_NAME).read_text()
    assert label_text == f"{after_sealed_run.WHAT}\n{after_sealed_run.SENTENCE}\n"
    assert forecast_noset.NOT_THE_TRIAL not in label_text
    assert not (out / forecast_noset.NOT_THE_TRIAL_FILE).exists()


def test_the_seeds_are_applied_to_the_neural_arms_only_and_in_order(whole):
    _, out, calls = whole
    assert calls == [("empirical", 1), ("empirical", 2)] * 3  # tuned, inputs, noset
    document = results(out)
    assert document["run"]["runs"] == RUN_NAMES
    for name in RUN_NAMES:
        one = json.loads((out / forecast_noset.side_name(name, "dial-whatif")).read_text())
        assert one["run_name"] == name
        expected = None if name == "tail" else int(name[-1])
        assert one["fit_seed"] == expected
        assert document["runs"][name]["fit_seed"] == expected
        table = out / forecast_noset.side_name(name, "forecasts")
        recorded = document["runs"][name]["forecast_table"]
        assert recorded["sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
        assert one["forecast_table"] == recorded


def test_a_run_with_other_options_reports_what_was_run_not_the_settings_file(whole):
    _, out, _ = whole
    document = results(out)
    run = document["run"]
    assert run["kappa"] == 0.3 and run["splits"] == SPLITS and run["seeds"] == list(SEEDS)
    assert run["models"] == {
        "tail": "empirical",
        "tuned": "empirical",
        "inputs": "empirical",
        "noset": "empirical",
    }
    assert run["models_as_run"] == {name: "empirical" for name in RUN_NAMES}
    assert run["fallback_dials"] == {u: 0.03 for u in GATE_UNITS}  # the smallest dial
    check = document["settings_check"]
    assert check["every_value_equals_the_committed_settings"] is False
    for key in ("level", "splits", "neural_seeds", "models", "matrix_sha256", "settings_sha256"):
        assert key in check["differences"], key
    assert check["differences"]["level"] == {"run": 0.3, "committed": SETTINGS["level"]}
    assert check["differences"]["neural_seeds"]["committed"] == SETTINGS["neural_seeds"]
    assert check["inputs_that_differ_from_the_defaults"].keys() == {"tables", "truth_run"}
    # the check's reference is the committed file, not the file `--settings` named
    assert document["definitions"]["committed_settings"] == SETTINGS
    differences = check["differences"]["settings_sha256"]
    assert differences["run"] != differences["committed"]
    assert "--settings" in " ".join(document["not_the_trial"]["overrides"])
    assert (
        "differs from the committed settings file's"
        in (out / forecast_noset.SUMMARY_NAME).read_text()
    )


def run_file(out: Path, name: str) -> dict:
    return json.loads((out / forecast_noset.side_name(name, "dial-whatif")).read_text())


def per_split(out: Path, name: str, unit: str, tier: str, line: str) -> np.ndarray:
    """One run's offers per split on `line`, worked out here from its own file: the arm S1 for the
    check line; for the judged line the offers of the dial with the largest mean over the splits (a
    tie, within `dial_splits.TIE`, to the smaller dial), picked here and not by the code under
    test."""
    document = run_file(out, name)
    splits = document["units"][unit]["splits"]
    if line == forecast_noset_pairs.CHECK:
        return np.array([s["tiers"][tier]["S1"]["offer_kwh_per_test_day"] for s in splits])
    offers = {
        float(dial): np.array([s["tiers"][tier]["by_dial"][dial] for s in splits])
        for dial in splits[0]["tiers"][tier]["by_dial"]
    }
    best = None
    for dial in sorted(offers):
        if best is None or offers[dial].mean() > offers[best].mean() + dial_splits.TIE:
            best = dial
    return offers[best]


def test_the_paired_figures_are_those_worked_out_again_from_the_per_run_files(whole):
    _, out, _ = whole
    document = results(out)
    checked = 0
    for unit, entry in document["comparison"].items():
        if entry["status"] != "calibratable":
            assert set(entry) == {"status", "counting_days"}
            continue
        for tier, tier_block in entry["tiers"].items():
            for line, block in tier_block.items():
                assert line in forecast_noset_pairs.LINES
                one = {
                    "tail": [per_split(out, "tail", unit, tier, line)],
                    **{
                        arm: [per_split(out, f"{arm}-seed{s}", unit, tier, line) for s in SEEDS]
                        for arm in ("tuned", "inputs", "noset")
                    },
                }
                figures = block["seed_figures"]
                for arm, found in one.items():
                    assert figures[arm] == pytest.approx([x.mean() for x in found])
                means = {arm: np.mean(found, axis=0) for arm, found in one.items()}
                for which, (first, second) in forecast_noset_pairs.PAIRS.items():
                    difference = means[first] - means[second]
                    cell = block[which]
                    assert cell["splits"] == SPLITS
                    assert cell["mean_difference_kwh_per_test_day"] == pytest.approx(
                        difference.mean()
                    )
                    assert cell["mean_main_offer_kwh_per_test_day"] == pytest.approx(
                        means[second].mean()
                    )
                    assert cell["splits_trial_offers_more"] == int((difference > 0).sum())
                    assert cell["splits_trial_offers_less"] == int((difference < 0).sum())
                    checked += 1
    assert checked == 2 * 2 * 2 * 4  # two split units, two tiers, two lines, four cells


def test_the_judged_line_is_not_the_check_line_and_each_seed_has_its_dial(whole):
    _, out, _ = whole
    document = results(out)
    for unit in JUDGED:
        plan = document["comparison"][unit]["tiers"]["A"][forecast_noset_pairs.PLAN_RULE]
        check = document["comparison"][unit]["tiers"]["A"][forecast_noset_pairs.CHECK]
        for arm, seeds in (("tail", 1), ("tuned", 2), ("inputs", 2), ("noset", 2)):
            found = plan["arms"][arm]
            assert len(found["dials_picked"]) == seeds
            assert set(found["dials_picked"]) <= {0.03, 0.1}
            assert found["seed_min"] <= found["mean_offer"] <= found["seed_max"]
            assert "dials_picked" not in check["arms"][arm]


def test_the_reading_is_the_rule_applied_to_the_comparison(whole):
    """Also: each comparison carries `silent_units` and `silent_count`, and the summary says per
    comparison at how many judged units the check line was silent. Red if `main` stops storing
    them or if `summary_lines` stops printing the count."""
    _, out, _ = whole
    document = results(out)
    reading = document["reading"]
    assert reading["rule"] == forecast_noset_reading.RULE
    lines = {}
    for which in forecast_noset_reading.COMPARISONS:
        entry = reading["comparisons"][which]
        units = entry["units"]
        assert set(units) == set(JUDGED)
        bars = forecast_noset.bars_of(SETTINGS)
        again = forecast_noset_reading.read_units(document["comparison"], bars, which)
        check = forecast_noset_reading.check_units(document["comparison"], bars, which)
        assert units == again and entry["check"] == check
        assert entry["disagreeing_units"] == forecast_noset_reading.disagreements(units, check)
        silent = forecast_noset_reading.silent_units(units, check)
        assert entry["silent_units"] == silent and entry["silent_count"] == len(silent)
        assert set(silent) <= set(JUDGED)
        count = f"The check line was silent at {len(silent)} of {len(units)} judged units"
        assert count in (out / forecast_noset.SUMMARY_NAME).read_text()
        lines[which] = entry["one_line"]
        assert lines[which] == forecast_noset_reading.one_line(units, check)
    assert reading["overall"] == forecast_noset_reading.overall(lines)
    assert document["comparison"]["nrel-before-may-2019"]["status"] != "calibratable"


# --- 6. a refused arm ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def refused(build, approved, tmp_path_factory):
    out = tmp_path_factory.mktemp("refused") / "out"
    code, calls = run_whole(build, approved, out, refuse_after=4)
    return code, out, calls


def test_a_refused_arm_is_recorded_the_others_stay_and_the_reading_is_not_read(refused):
    code, out, calls = refused
    assert code == 1
    assert len(calls) == 6  # both no-encoder seeds were tried
    document = results(out)
    assert set(document["failed_runs"]) == {"noset-seed1", "noset-seed2"}
    assert all("Refused: made up" in text for text in document["failed_runs"].values())
    names = {p.name for p in out.iterdir()}
    for name in ("tail", "tuned-seed1", "tuned-seed2", "inputs-seed1", "inputs-seed2"):
        assert forecast_noset.side_name(name, "dial-whatif") in names
    for name in ("noset-seed1", "noset-seed2"):
        assert forecast_noset.side_name(name, "dial-whatif") not in names
        assert name not in document["runs"]
    for which in forecast_noset_reading.COMPARISONS:
        entry = document["reading"]["comparisons"][which]
        assert {u["word"] for u in entry["units"].values()} == {forecast_noset_reading.NOT_READ}
        assert entry["one_line"] == forecast_noset_reading.NOT_READ
    assert document["reading"]["overall"] == forecast_noset_reading.NOT_READ
    for unit in JUDGED:
        for line in forecast_noset_pairs.LINES:
            block = document["comparison"][unit]["tiers"]["A"][line]
            assert block["against_tuned"] is None and block["against_inputs"] is None
            assert block["against_tail"] is None
            assert block["inputs_against_tail"] is not None  # the arms that ran are still compared
    text = (out / forecast_noset.SUMMARY_NAME).read_text()
    assert "FAILED: noset-seed1: Refused: made up" in text
    assert any(f.startswith("noset-seed2: ") for f in document["failed_configurations"])


# --- 7. the dial is chosen from calibration days alone ------------------------------------------

GRID = dial_splits.GRID
CAL, TEST, PICK = list(range(40)), list(range(40, 60)), list(range(20))
FIXED, OTHER = 0.02, 0.1


def unit(value, loss=None) -> dial_splits.UnitLosses:
    """Every one of 60 days offers with certified value `value`; `loss`: the per-day slot and
    driver loss at every grid scale (none by default)."""
    shape = (60, len(GRID))
    zero = np.zeros(shape)
    return dial_splits.UnitLosses(
        value=np.full(60, float(value)),
        offering=np.ones(60, dtype=bool),
        slot=zero if loss is None else loss,
        driver=zero if loss is None else loss,
        ceiling=np.full(60, np.nan),
    )


def test_the_dial_s1_takes_does_not_depend_on_the_losses_of_the_test_days():
    quiet = {FIXED: unit(10), OTHER: unit(12)}
    loud = np.zeros((60, len(GRID)))
    loud[TEST, :] = 1.0  # the richer dial loses on every test day, and on no other day
    noisy = {FIXED: unit(10), OTHER: unit(12, loud)}
    one = site_dial_whatif.split_arms(quiet, FIXED, 1, CAL, TEST, PICK, "A")["S1"]
    two = site_dial_whatif.split_arms(noisy, FIXED, 1, CAL, TEST, PICK, "A")["S1"]
    assert one["dial"] == two["dial"] == OTHER  # 12 against 10 on the choosing days
    assert one["choosing_offers"] == two["choosing_offers"] == {"0.02": 10.0, "0.1": 12.0}
    assert one["scale"] == two["scale"] == 1.0
    assert one["slot_loss"] == 0.0 and two["slot_loss"] == 1.0  # the test days are scored, only
    assert one["offering_calibration_days"] == two["offering_calibration_days"] == 20


def test_the_fallback_dial_is_the_smallest_dial_of_the_family():
    def configuration(dial):
        return SimpleNamespace(driver_risk=dial)

    matrix = SimpleNamespace(gate_units=("a", "b"))
    family = (configuration(None), tuple(configuration(d) for d in (0.1, 0.0, 0.03)))
    assert forecast_noset.fallback_dials(matrix, family) == {"a": 0.0, "b": 0.0}


# --- 8. the committed settings and the README ---------------------------------------------------

PRIOR_LOOKS = Path(forecast_noset.TOP) / "docs" / "prior-looks.md"
ENTRY = re.compile(r'entry beginning "([^"]+)"')


def strings(value):
    """Every string of a loaded JSON document."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for found in value.values():
            yield from strings(found)
    elif isinstance(value, list):
        for found in value:
            yield from strings(found)


def test_the_committed_settings_are_the_ones_the_code_and_the_plan_give():
    assert SETTINGS["level"] == 0.5 and SETTINGS["level"] in driver_risk.LEVELS
    assert SETTINGS["splits"] == 200
    assert SETTINGS["neural_seeds"] == [1, 2, 3]
    assert SETTINGS["arms"] == {
        "tail": "lgbm_tail",
        "tuned": "hazard_tuned",
        "inputs": "hazard_tuned_inputs",
        "noset": "hazard_tuned_no_set",
    }
    assert forecast_noset.ARMS == ("tail", "tuned", "inputs", "noset")
    assert forecast_noset.NEURAL_ARMS == ("tuned", "inputs", "noset")
    for name in SETTINGS["arms"].values():
        assert importlib.util.find_spec(f"fleetreserve.models.{name}") is not None
    assert forecast_noset.bars_of(SETTINGS) == {
        "nrel-from-may-2019": 0.08,
        "norway-development-pooled": 0.30,
    }
    for found in SETTINGS["reading"]["bars"].values():
        assert 'entry beginning "The neural model\'s seed test"' in found["source"]
        assert "analogy" in found["source"] and "judged line only" in found["source"]
    assert SETTINGS["reading"]["no_other_sites"] == forecast_noset_reading.NO_OTHER_SITES
    assert SETTINGS["reading"]["judged_comparison"].startswith(forecast_noset_reading.JUDGED)
    assert set(SETTINGS["reading"]["comparisons"]) == set(forecast_noset_reading.COMPARISONS)
    matrix = gate_config.load(forecast_noset.MATRIX)
    assert set(forecast_noset.bars_of(SETTINGS)) <= set(matrix.gate_units)
    _, (plain, dials) = dial_choice.family_at(
        matrix, dial_choice.find_family(matrix), SETTINGS["level"]
    )
    assert plain.kappa == SETTINGS["level"] and dials
    assert set(forecast_noset.REQUIRED_SETTINGS) <= set(SETTINGS)
    assert isinstance(SETTINGS["approved_to_run"], bool)
    for key in ("question", "fallback_dial", "dial_rules", "model_differences", "seed_spread"):
        assert SETTINGS[key]
    assert set(SETTINGS["dial_rules"]) >= set(forecast_noset_pairs.LINES)
    plan_rule = SETTINGS["dial_rules"][forecast_noset_pairs.PLAN_RULE]
    assert "dial_splits.pick_dial" in plan_rule and "THE JUDGED LINE" in plan_rule
    assert "reads the test-day offers of the development units" in plan_rule
    assert "every arm alike" in plan_rule and "every seed" in plan_rule
    assert "bars of the reading apply to this line" in plan_rule
    assert callable(dial_splits.pick_dial)
    check = SETTINGS["dial_rules"][forecast_noset_pairs.CHECK]
    assert "THE CHECK LINE" in check and "No bar applies" in check and "no verdict" in check
    for found in ("pick_part", "choose", "calibrated_offer"):
        assert found in check
        assert callable(getattr(site_dial_whatif, found))
    rule = SETTINGS["dial_rules"]["disagreement"]
    assert "OPPOSITE sign" in rule and "SILENT" in rule
    assert "check line silent (difference 0)" in rule
    assert "cannot change it" in rule and "silent_count" in rule
    assert "differs from its sign on the check line" not in rule
    assert "smallest dial" in SETTINGS["fallback_dial"]
    assert forecast_noset.plan_runs(SETTINGS["arms"], SETTINGS["neural_seeds"]) == [
        forecast_noset.Run("tail", "tail", "lgbm_tail", None),
        *(
            forecast_noset.Run(f"{arm}-seed{seed}", arm, SETTINGS["arms"][arm], seed)
            for arm in ("tuned", "inputs", "noset")
            for seed in (1, 2, 3)
        ),
    ]
    assert len(forecast_noset.plan_runs(SETTINGS["arms"], SETTINGS["neural_seeds"])) == 10


def test_every_citation_of_the_prior_looks_is_a_phrase_that_occurs_once_and_no_line_number():
    """Each `entry beginning "<phrase>"` of the settings file and of the folder's README section
    names a phrase that occurs exactly once in `docs/prior-looks.md`; no citation is a line number.
    Red if a phrase is changed, if the entry is edited or duplicated, or if a line number is
    written back."""
    text = PRIOR_LOOKS.read_text()
    readme = (Path(forecast_noset.SETTINGS_FILE).parent / "README.md").read_text()
    section = " ".join(readme.split("## Trial forecast-noset", 1)[1].split())
    found = set()
    for source in (*strings(SETTINGS), section):
        found |= set(ENTRY.findall(source))
    assert len(found) >= 5
    for phrase in sorted(found):
        assert text.count(phrase) == 1, phrase
    assert not [s for s in strings(SETTINGS) if re.search(r"\blines? \d+", s)]
    assert not re.search(r"prior-looks\.md`? lines? \d+", section)
    assert not re.search(r"prior-looks\.md`? lines? \d+", Path(__file__).read_text())


def test_the_readme_gives_the_one_command_and_says_approved_at_the_top_of_the_section():
    text = (Path(forecast_noset.SETTINGS_FILE).parent / "README.md").read_text()
    section = text.split("## Trial forecast-noset", 1)[1]
    first = [line for line in section.splitlines()[1:] if line.strip()][0]
    assert first.startswith("APPROVED TO RUN by the owner's decision of 24 November 2026")
    section = " ".join(section.split())  # one line: a phrase may wrap in the file
    command = (
        "python -m fleetreserve.run_record --name forecast-noset --long --max-seconds 172800 -- "
        "uv run --frozen "
        "--group learn python -m fleetreserve.forecast_noset"
    )
    assert command in section
    for stated in (
        "judged line",
        "check line",
        "reads the test-day offers of the development units",
        "an analogy",
        "least and the greatest",
        "`--tables` and `--truth-run`",
        "reported, not refused",
        "one difference",
        "two differences",
    ):
        assert stated in section, stated


# --- 10. the differences between the neural models ----------------------------------------------

MODELS = Path(forecast_noset.TOP) / "src" / "fleetreserve"


def assigned(path: Path, name: str):
    """The literal value assigned to the module-level `name` in the source file `path`."""
    for node in ast.parse(path.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        targets = [*targets, node.target] if isinstance(node, ast.AnnAssign) else targets
        if any(getattr(target, "id", None) == name for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in {path}")


def test_the_neural_models_differ_as_the_readme_and_the_settings_state():
    """Read from the source text: `hazard_tuned_no_set` is `hazard_tuned` with the part
    `no_set_encoder` applied and nothing else; `hazard_tuned_inputs` is `hazard_tuned` with the
    columns of `EXTRA_INPUTS` and nothing else. Red if a model is changed so that another
    difference arises (an extra part, another settings file, extra columns in the no-encoder
    model)."""
    assert hazard_parts.PARTS["no_set_encoder"] == {"set_encoder": False}  # light: no learn group
    no_set = (MODELS / "models" / "hazard_tuned_no_set.py").read_text()
    assert assigned(MODELS / "models" / "hazard_tuned_no_set.py", "PART") == "no_set_encoder"
    assert "tuned.hazard(**PARTS[PART])" in no_set and "load_tuned()" in no_set
    assert "EXTRA_INPUTS" not in no_set and "model_inputs" not in no_set
    tuned = (MODELS / "models" / "hazard_tuned.py").read_text()
    assert "PARTS" not in tuned and "EXTRA_INPUTS" not in tuned
    assert "settings=required(tuned).hazard()" in tuned
    extra = assigned(MODELS / "models" / "hazard_tuned_inputs.py", "EXTRA_INPUTS")
    assert len(extra) == 14 and len(set(extra)) == 14
    inputs = (MODELS / "models" / "hazard_tuned_inputs.py").read_text()
    assert "class HazardTunedInputsModel(HazardTunedModel)" in inputs
    assert "PARTS" not in inputs and "set_encoder" not in inputs
    assert "tuned=tuned" in inputs and "load_tuned()" in inputs
    differences = SETTINGS["model_differences"]
    assert differences["noset_against_tuned"].startswith("one difference")
    assert differences["inputs_against_tuned"].startswith("one difference")
    assert differences["noset_against_inputs"].startswith("two differences")
    assert "fourteen" in differences["inputs_against_tuned"]


# --- 9. no sealed data --------------------------------------------------------------------------


@pytest.mark.parametrize("module", [forecast_noset, forecast_noset_pairs, forecast_noset_reading])
def test_the_new_modules_name_no_loader_of_the_sealed_data(module):
    text = Path(module.__file__).read_text()
    assert "heldout" not in text.lower().replace("held-out", "")
    modules = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom):
            modules += [node.module or "", *(a.name for a in node.names)]
        elif isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
    assert modules and not [m for m in modules if "held" in m.lower() or "sealed_route" in m]
