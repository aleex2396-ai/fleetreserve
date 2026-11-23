"""The held setting of the truth build (plan 3): no charging before τ, so room is the session
energy; everything else is the main build.

1. On the made-up raw tree every held car has energy before τ 0, the held rule text and room equal
   to its session energy; every other column equals the main build's (`room_fits` follows the new
   room); the days tables are equal.
2. The readings about energy before τ are reported as not applicable (`None`) in the held setting,
   the rate readings are those of the main setting.
3. The main setting's written files are the bytes they were before the held setting existed.
4. The two settings write to different default folders; neither overwrites the other, and a build
   refuses a folder that holds the other setting's tables.
5. The held report and summary state the setting on their first lines; the main ones have no
   setting line.
6. The held tables pass the WP2 table checks of their setting and fail them when energy before τ
   is not 0; the setting a summary names must be the setting the tables folder holds; the
   latest-run lookup of one setting never finds a run of the other.
7. `truth_verify` as a command with `--setting held`: exit 0 on held tables with the held run;
   non-zero, naming the mismatch, on main tables or a main run.

The made-up tree is cut for the window `TEST_WINDOW` (τ 10:00, six service hours). Every build here
passes it explicitly, so the tests do not depend on the value of `windows.WORKPLACE`. The command
line takes no window, so the tests of `truth_build.main` run it with `build_all` wrapped to pass
`TEST_WINDOW`; the rest of the command is unchanged.
"""

import hashlib
import json

import pandas as pd
import pytest
from fake_raw import full_tree

from fleetreserve import run_verify, truth, truth_build, truth_readings, truth_verify
from fleetreserve.windows import Window

TEST_WINDOW = Window(tau_hour=10, service_hours=6, rule="workplace: τ 10:00, S 11:00 to 17:00")
_BUILD_ALL = truth_build.build_all

# SHA-256 of the lines "<folder> <file> <SHA-256 of the file>", folders out (tables) then run
# (summary and report), files sorted, as the main build wrote them on this made-up tree at the
# commit before the held setting existed. A change to the main truth rules changes it.
MAIN_FILES_BEFORE_HELD = "3107d8c988d7f2e7bc524ec4d9cf80d5309affa05df089649d781fb8708f9276"
CHANGED_BY_HELD = {"energy_before_kwh", "energy_before_rule", "room_kwh", "room_fits"}


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    return full_tree(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="module")
def main_build(raw):
    return truth_build.build_all(raw, TEST_WINDOW)


@pytest.fixture(scope="module")
def held_build(raw):
    return truth_build.build_all(raw, TEST_WINDOW, setting=truth.HELD)


@pytest.fixture(autouse=True)
def command_uses_the_test_window(monkeypatch):
    def build_all(raw_root, workplace=TEST_WINDOW, setting=truth.MAIN):
        return _BUILD_ALL(raw_root, workplace, setting)

    monkeypatch.setattr(truth_build, "build_all", build_all)


def _command(raw, tmp_path, monkeypatch, *extra, out="out", run="run"):
    run_dir = tmp_path / run
    run_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(run_dir))
    status = truth_build.main(["--raw-root", str(raw), "--out", str(tmp_path / out), *extra])
    return status, tmp_path / out, run_dir


def _listing(out, run_dir) -> str:
    lines = [
        f"{folder.name} {path.name} {hashlib.sha256(path.read_bytes()).hexdigest()}"
        for folder in (out, run_dir)
        for path in sorted(folder.iterdir())
    ]
    return "\n".join(lines) + "\n"


def _files(folder) -> dict:
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


# --- 1. the cars --------------------------------------------------------------------------------


def test_held_cars_draw_nothing_before_tau_and_room_is_the_session_energy(held_build):
    assert set(held_build.tables) == set(truth_build.TRAIN_UNITS) | set(
        truth_build.DEVELOPMENT_UNITS
    )
    for unit, (cars, _) in held_build.tables.items():
        assert len(cars) > 0
        assert (cars["energy_before_kwh"] == 0.0).all(), unit
        assert (cars["energy_before_rule"] == truth.BEFORE_RULE_HELD).all(), unit
        assert (cars["room_kwh"] == cars["energy_kwh"]).all(), unit


def test_every_other_column_and_the_days_equal_the_main_build(main_build, held_build):
    drawn_before = 0
    for unit, (main_cars, main_days) in main_build.tables.items():
        held_cars, held_days = held_build.tables[unit]
        drawn_before += int((main_cars["energy_before_kwh"] > 0).sum())
        keep = [name for name in truth.CAR_COLUMNS if name not in CHANGED_BY_HELD]
        pd.testing.assert_frame_equal(held_cars[keep], main_cars[keep])
        pd.testing.assert_frame_equal(held_days, main_days)
        fits = held_cars["room_kwh"] <= held_cars["rate_kw"] * held_cars["depart_hour"] + 1e-9
        assert (held_cars["room_fits"] == fits).all(), unit
    assert drawn_before > 0  # the main build draws energy before τ here, so the test can fail


def test_the_summary_differs_only_where_the_energy_before_tau_does(main_build, held_build):
    for unit, entry in held_build.summary.items():
        main_entry = main_build.summary[unit]
        assert entry["counting_days"] == main_entry["counting_days"]
        assert entry["cars_on_counting_days"] == main_entry["cars_on_counting_days"]
        assert entry["share_zero_room"] <= main_entry["share_zero_room"]
    assert held_build.windows == main_build.windows


# --- 2. the readings ----------------------------------------------------------------------------


def test_readings_about_energy_before_tau_are_not_applicable_in_the_held_setting(
    main_build, held_build
):
    for unit, held in held_build.summary.items():
        changes, main_changes = held["reading_changes"], main_build.summary[unit]["reading_changes"]
        for key in truth_readings.NOT_APPLICABLE_HELD:
            assert changes[key] is None, (unit, key)
            assert isinstance(main_changes[key], int)
        for key in ("cars_on_counting_days", "rate_history_ended_before_tau"):
            assert changes[key] == main_changes[key]
        assert (
            changes["rate_average_over_plug_in_to_plug_out"]
            == (main_changes["rate_average_over_plug_in_to_plug_out"])
        )


# --- 3. the main files --------------------------------------------------------------------------


def test_the_main_files_are_the_bytes_they_were_before_the_held_setting(raw, tmp_path, monkeypatch):
    status, out, run_dir = _command(raw, tmp_path, monkeypatch)
    assert status == 0
    digest = hashlib.sha256(_listing(out, run_dir).encode()).hexdigest()
    assert digest == MAIN_FILES_BEFORE_HELD
    assert not (out / truth_build.SETTING_FILE).exists()
    assert "setting" not in json.loads((run_dir / truth_build.SUMMARY_NAME).read_text())


def test_naming_the_main_setting_changes_no_byte(raw, tmp_path, monkeypatch):
    _, out, run_dir = _command(raw, tmp_path, monkeypatch, out="a", run="ra")
    _, named, named_run = _command(
        raw, tmp_path, monkeypatch, "--setting", "main", out="b", run="rb"
    )
    assert _files(out) == _files(named) and _files(run_dir) == _files(named_run)


# --- 4. the folders -----------------------------------------------------------------------------


def test_the_settings_write_to_different_default_folders(raw, tmp_path, monkeypatch):
    assert truth_build.OUT != truth_build.OUT_HELD
    assert truth_build.DEFAULT_OUT == {
        truth.MAIN: truth_build.OUT,
        truth.HELD: truth_build.OUT_HELD,
    }
    main_folder, held_folder = tmp_path / "truth", tmp_path / "truth-held"
    monkeypatch.setattr(
        truth_build, "DEFAULT_OUT", {truth.MAIN: main_folder, truth.HELD: held_folder}
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(run_dir))
    assert truth_build.main(["--raw-root", str(raw)]) == 0
    before = _files(main_folder)
    assert truth_build.main(["--raw-root", str(raw), "--setting", "held"]) == 0
    assert _files(main_folder) == before  # the held build did not touch the main tables
    assert truth_build.read_setting(main_folder) == truth.MAIN
    assert truth_build.read_setting(held_folder) == truth.HELD
    held_before = _files(held_folder)
    assert truth_build.main(["--raw-root", str(raw)]) == 0
    assert _files(held_folder) == held_before  # nor the main build the held tables
    assert before != {k: v for k, v in held_before.items() if k in before}
    assert held_before[truth_build.SETTING_FILE] == b"held\n"


def test_a_build_refuses_a_folder_that_holds_the_other_setting(raw, tmp_path, monkeypatch, capsys):
    _, main_folder, _ = _command(raw, tmp_path, monkeypatch, out="main-tables", run="r1")
    _, held_folder, _ = _command(
        raw, tmp_path, monkeypatch, "--setting", "held", out="held-tables", run="r2"
    )
    main_before, held_before = _files(main_folder), _files(held_folder)
    status, _, _ = _command(raw, tmp_path, monkeypatch, "--setting", "held", out="main-tables")
    assert status == 2 and "main setting" in capsys.readouterr().out
    status, _, _ = _command(raw, tmp_path, monkeypatch, out="held-tables")
    assert status == 2 and "held setting" in capsys.readouterr().out
    assert _files(main_folder) == main_before and _files(held_folder) == held_before


# --- 5. the first lines -------------------------------------------------------------------------


def test_the_held_report_and_summary_state_the_setting_first(raw, tmp_path, monkeypatch, capsys):
    status, out, run_dir = _command(raw, tmp_path, monkeypatch, "--setting", "held")
    assert status == 0
    printed = capsys.readouterr().out
    report = (run_dir / truth_build.REPORT_NAME).read_text()
    assert printed == report
    assert report.splitlines()[0].startswith("SETTING: held. No charging before τ")
    assert report.splitlines()[2].startswith("FINDINGS")
    assert "not applicable in the held setting" in report
    text = (run_dir / truth_build.SUMMARY_NAME).read_text()
    assert text.splitlines()[:2] == ["{", ' "setting": "held",']
    document = json.loads(text)
    assert document["setting"] == "held" and set(document["units"])
    assert truth_build.read_setting(out) == truth.HELD


def test_the_main_report_has_no_setting_line(raw, tmp_path, monkeypatch):
    _, _, run_dir = _command(raw, tmp_path, monkeypatch)
    assert (run_dir / truth_build.REPORT_NAME).read_text().startswith("FINDINGS")
    assert "SETTING" not in (run_dir / truth_build.REPORT_NAME).read_text()


def test_an_unknown_setting_is_refused(raw):
    with pytest.raises(ValueError):
        truth_build.Build("both")
    with pytest.raises(SystemExit):
        truth_build.main(["--raw-root", str(raw), "--setting", "both"])


# --- 6. the checks on the written tables --------------------------------------------------------


@pytest.fixture(scope="module")
def written(held_build, main_build, tmp_path_factory):
    folder = tmp_path_factory.mktemp("written")
    held_out, main_out = folder / "held", folder / "main"
    held_files = truth_build.write(held_build, held_out)
    main_files = truth_build.write(main_build, main_out)
    held = json.loads(json.dumps(truth_build.summary_document(held_build, held_files), default=str))
    main = json.loads(json.dumps(truth_build.summary_document(main_build, main_files), default=str))
    return held, held_out, main, main_out


@pytest.mark.parametrize("role", ["train", "development"])
def test_held_tables_pass_the_table_checks_of_their_setting(written, raw, role):
    held, held_out, _, _ = written
    status, lines = truth_verify.verify_role(role, held, held_out, raw)
    assert status == 0, lines
    assert len(lines) == len(truth_build.UNITS[role])


def test_held_tables_with_energy_drawn_before_tau_fail(written):
    held, held_out, _, _ = written
    cars = pd.read_csv(held_out / "caltech-cars.csv", dtype=str, keep_default_na=False)
    days = pd.read_csv(held_out / "caltech-days.csv", dtype=str, keep_default_na=False)
    tau_hours = {"caltech": TEST_WINDOW.tau_hour}
    assert truth_verify.check_tables(cars, days, tau_hours, None, truth.HELD) == []
    drawn = cars.copy()
    drawn.loc[0, "energy_before_kwh"] = "1.000000"
    drawn.loc[0, "room_kwh"] = f"{float(drawn.loc[0, 'energy_kwh']) - 1:.6f}"
    assert truth_verify.check_tables(drawn, days, tau_hours, None, truth.HELD) == [
        "energy before τ is not 0 in the held setting"
    ]
    ruled = cars.copy()
    ruled.loc[0, "energy_before_rule"] = truth.BEFORE_RULE
    assert truth_verify.check_tables(ruled, days, tau_hours, None, truth.HELD) == [
        "energy_before_rule is not the held rule everywhere"
    ]


def test_the_setting_a_summary_names_must_be_the_setting_the_tables_hold(written):
    held, held_out, main, main_out = written
    assert truth_verify.summary_setting(held) == truth.HELD
    assert truth_verify.summary_setting(main) == truth.MAIN  # a main summary has no key
    assert truth_verify.require_matching_setting(held, held_out, "r") == truth.HELD
    assert truth_verify.require_matching_setting(main, main_out, "r") == truth.MAIN
    for summary, tables, others in ((held, main_out, "main"), (main, held_out, "held")):
        with pytest.raises(truth_verify.CouldNotRun, match=f"tables .* are of the {others}"):
            truth_verify.require_matching_setting(summary, tables, "r")
    with pytest.raises(truth_verify.CouldNotRun, match="unknown setting"):
        truth_verify.summary_setting({"setting": "both"})


def test_the_latest_run_lookup_of_one_setting_never_finds_the_other(tmp_path):
    for name in (
        "20260101T000000Z-wp2-truth",
        "20260103T000000Z-wp2-truth-held",
        "20260102T000000Z-wp2-truth",
        "20260104T000000Z-wp6-gate",
    ):
        (tmp_path / name).mkdir()
    assert truth_verify.latest_run(tmp_path).name == "20260102T000000Z-wp2-truth"
    assert truth_verify.latest_run(tmp_path, truth.MAIN).name == "20260102T000000Z-wp2-truth"
    assert truth_verify.latest_run(tmp_path, truth.HELD).name == "20260103T000000Z-wp2-truth-held"
    (tmp_path / "20260103T000000Z-wp2-truth-held").rmdir()
    with pytest.raises(truth_verify.CouldNotRun, match="wp2-truth-held"):
        truth_verify.latest_run(tmp_path, truth.HELD)


def test_the_held_report_says_its_windows_are_the_main_setting_s(raw, tmp_path, monkeypatch):
    _, _, run_dir = _command(raw, tmp_path, monkeypatch, "--setting", "held", out="h", run="rh")
    held = (run_dir / truth_build.REPORT_NAME).read_text().splitlines()
    assert held[held.index("WINDOWS") + 1] == truth_build.WINDOWS_AS_MAIN
    _, _, run_dir = _command(raw, tmp_path, monkeypatch, out="m", run="rm")
    assert truth_build.WINDOWS_AS_MAIN not in (run_dir / truth_build.REPORT_NAME).read_text()


# --- 7. `truth_verify` as a command with --setting ----------------------------------------------

COMMIT = "c" * 40


def _record(run_dir) -> None:
    """A manifest that makes the run folder a verified recorded run."""
    outputs = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in run_dir.iterdir()
    }
    manifest = {
        "commit": COMMIT,
        "command": ["build"],
        "start": "s",
        "end": "e",
        "exit_code": 0,
        "outputs": outputs,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture(scope="module")
def recorded(raw, tmp_path_factory):
    folder = tmp_path_factory.mktemp("recorded")
    found = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            truth_build,
            "build_all",
            lambda raw_root, setting=truth.MAIN: _BUILD_ALL(raw_root, TEST_WINDOW, setting),
        )
        for setting, run_name in ((truth.MAIN, "wp2-truth"), (truth.HELD, "wp2-truth-held")):
            run = folder / f"20260101T000000Z-{run_name}"
            run.mkdir()
            monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(run))
            tables = folder / f"tables-{setting}"
            arguments = ["--raw-root", str(raw), "--setting", setting, "--out", str(tables)]
            assert truth_build.main(arguments) == 0
            _record(run)
            found[setting] = (run, tables)
    return found


@pytest.fixture
def verifying(monkeypatch, raw):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(truth_verify, "_head_commit", lambda: COMMIT)
    monkeypatch.setattr(truth_verify, "RAW_ROOT", raw)


def _verify(what, setting, run, tables) -> int:
    arguments = [what, "--run", str(run), "--tables", str(tables)]
    return truth_verify.main([*arguments, "--setting", setting] if setting else arguments)


@pytest.mark.parametrize("what", ["train", "development", "fallbacks"])
def test_the_command_checks_held_tables_with_the_held_run(recorded, verifying, what, capsys):
    run, tables = recorded[truth.HELD]
    assert _verify(what, "held", run, tables) == 0
    assert "passed:" in capsys.readouterr().out


def test_the_command_still_checks_main_tables_with_the_main_run(recorded, verifying):
    run, tables = recorded[truth.MAIN]
    assert _verify("train", None, run, tables) == 0
    assert _verify("train", "main", run, tables) == 0


def test_the_command_refuses_held_asked_of_main_tables(recorded, verifying, capsys):
    held_run, _ = recorded[truth.HELD]
    main_run, main_tables = recorded[truth.MAIN]
    assert _verify("train", "held", held_run, main_tables) == 2
    assert "of the main setting" in capsys.readouterr().out
    assert _verify("train", "held", main_run, main_tables) == 2
    assert "is of the main setting, not the held setting asked for" in capsys.readouterr().out
    assert _verify("train", "main", held_run, main_tables) == 2
    assert "is of the held setting, not the main setting asked for" in capsys.readouterr().out
