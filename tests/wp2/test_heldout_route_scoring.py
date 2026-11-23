"""`gate_run` and `learn_run` scoring held-out units through the sealed route (`sealed_route`).

One made-up repository (`fake_heldout.sealed_repo`, at its made-up tag `frozen-plan`) holds the
made-up sealed files of `heldout_truth_fake`. The held-out truth build (`heldout_truth_build`)
runs once at the tag, through the opener, and writes the held-out tables (`data/derived/
truth-heldout`); a made-up development build (`gate_fake.write_build`: the training unit caltech,
site T) lies in `data/derived/development`. Both runs are recorded at the tag's commit (a manifest
listing the summary's SHA-256), and the commit check of `run_verify` is git's own, in the made-up
repository; only the checked-out commit of the package's own top folder is replaced by the tag's.
The scored unit is OSL_T (30 counting days, 15 calibration days). The tag gate is the loader's own
(checks 2 to 8 on the made-up repository).

1. `gate_run` on a matrix whose gate unit is OSL_T is refused (exit 2, one line, no table read):
   its only forecaster, the empirical forecaster, builds a unit's forecasts from the unit's own
   other days, test days included.
2. `learn_run` fits the model on the development build's caltech tables only and scores OSL_T from
   the held-out build; the route is recorded in `learn.json` and `learn-gate.json` (the tag's
   commit, the held-out build's check with the build's own hashes, the label naming the tag or a
   logged parsing fix of it, with a clean tree, and what the route's checks stop and do not,
   `checks`, both written out here by hand), the first summary line
   says so and the second is that `checks` text, the calibration days are the build's (its
   `calibration_dates`), and no sealed file is opened again.
3. Refused, each with exit 2, one plain line and no table read, for both commands: off the tag (a
   made-up repository without it), units of two kinds in the matrix, the two tables folders
   swapped, the held-out arguments with a development matrix, no held-out truth run; and for
   `learn_run` builds of two settings (the held setting's held-out build with the main
   development build).
4. On the route, `learn_run` refuses a look-ahead model (exit 1), as the cold start does.
5. Spellings: with copies of the scored unit's held-out tables and of the held-out run placed in
   the made-up sealed folder, a path that reaches the sealed folder spelled `DATA/heldout` or
   `Data/heldout` (as the held-out tables folder, a plain folder of links to its files, a link to
   the folder, a held-out truth run inside it) is refused by both commands with exit 2 and one
   line, and no file or folder of the sealed folder is opened, listed or scanned, counted by
   identity on disk (`sealed_watch`); the held-out build's folder spelled `DATA/derived/...` is
   refused as development tables; and the folder check refuses a top folder spelled with its
   accent decomposed.
6. A plain folder of hard links to the sealed copies, given as the held-out tables, is refused by
   both commands with exit 2 and one line, no sealed file opened.
7. A matrix whose split seed is not the one the held-out build drew its calibration days by is
   refused by `learn_run` (exit 2, a line naming both rules, no table read); the matrix of item 2,
   of the build's rule, scores.
8. No look at a scored unit's test days beyond scoring them: `learn_run --model empirical` is
   refused (exit 2, no table read); a run that scores holds no kappa selection per unit in its
   gate document, tier A or B, and its summary lines hold none; `gate_run --limit-days` and
   `--only` are refused (exit 2, no table read).
9. At a logged parsing fix after the tag (the opener's reading H6), in a second made-up
   repository whose made-up plan is frozen at the tag: `learn_run` scores at the fix's entry
   commit, after a refusal of the a.s.r. layout logged at the tag (written by hand), and its
   first summary line and its route record in `learn.json` and `learn-gate.json` name the tag's
   commit and the fix commit; `sealed_route.units_route` there returns both. Refused with exit 2,
   the opener's reason, no table read and no model loaded: at the correction without its entry,
   and at a fix of the Korean loader whose layout no line of the refusal log names. The gate
   again (`sealed_route.gate_again`) refuses a run whose route passed at the first fix once HEAD
   has moved to a later fix on the same line (a second a.s.r. correction and its entry), though
   the gate itself passes there.
"""

import builtins
import hashlib
import io
import json
import os
import shutil
import sys
import types
import unicodedata
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fake_heldout import ASR_FILE, correction, entry, freeze, git, log_refusal, sealed_repo
from gate_fake import COMMITTED, gate_cars, matrix_document, write_build
from heldout_truth_fake import sealed_files
from sealed_watch import watch
from swap_after_read import file_refusal, grown, swapped_after_first_read, table_refusal

import fleetreserve
from fleetreserve import gate_run, gate_summary, heldout_loader, learn_run, sealed_route
from fleetreserve import heldout_truth_build as h
from fleetreserve.forecast_model import KEY_COLUMNS, SURVIVAL_COLUMNS, TABLE_COLUMNS
from fleetreserve.gate_config import load

MATRIX = Path(fleetreserve.__file__).resolve().parents[2] / "gate" / "matrix.json"
NAMES = ("main-rounded-hour-k0.1-empirical", "main-rounded-hour-k0.1-oracle")
SCORED = "OSL_T"
FIT_UNIT = "caltech"
MODEL = "route_model"
LOOK_AHEAD = "hazard_tuned_inputs_lookahead"  # a name of forecast_model.LOOK_AHEAD_MODELS
COMMANDS = ("gate_run", "learn_run")
# What a run on the route says its checks are, as the summary's second line and under `checks`.
CHECKS_TEXT = (
    "The route's checks stop ordinary mistakes by an honest operator; they are not a defence "
    "against deliberate evasion (links or copies made on purpose, a second working copy). The "
    "proof that the seal held is the plan's own: the committed hashes, the held-out loader, "
    "which opens a sealed file only at the tag frozen-plan, or at a logged parsing fix of it, "
    "with a clean tree, and its openings log."
)
# The route's label: one text, true at the tag and at a logged parsing fix of it.
LABEL_TEXT = (
    "Held-out units, through the sealed-data gate at the tag frozen-plan, or at a logged parsing "
    "fix of it, with a clean tree"
)


def _record(run: Path, commit: str) -> None:
    """The manifest of a run that wrote the truth summary in `run`, at `commit`."""
    summary = run / "truth-summary.json"
    manifest = {
        "commit": commit,
        "command": ["build"],
        "start": "s",
        "end": "e",
        "exit_code": 0,
        "outputs": {summary.name: hashlib.sha256(summary.read_bytes()).hexdigest()},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))


def _held_out_build(root: Path, run: Path, *options: str) -> None:
    run.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as patched:
        patched.setenv("FLEETRESERVE_RUN_DIR", str(run))
        assert h.main(["--matrix", str(MATRIX), *options], root=root) == 0


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    folder = tmp_path_factory.mktemp("route")
    return sealed_repo(folder, sealed_files(folder / "made"))


@pytest.fixture(scope="module")
def builds(repo):
    """The two builds, recorded at the tag's commit (module text)."""
    tag = git(repo, "rev-parse", "HEAD")
    derived = repo / "data" / "derived"
    held_run = derived / "runs" / "20261116T000000Z-wp14-heldout-truth"
    _held_out_build(repo, held_run)
    _record(held_run, tag)
    (derived / "development").mkdir()
    run, tables = write_build(derived / "development", {FIT_UNIT: gate_cars(40, "T")})
    _record(run, tag)
    return {
        "tag": tag,
        "run": run,
        "tables": tables,
        "held_run": held_run,
        "held_tables": sealed_route.held_out_tables(repo),
        "summary": json.loads((held_run / "truth-summary.json").read_text()),
    }


class RouteModel:
    """A stand-in departure model fitted on caltech: every car leaves 3 hours after τ."""

    def __init__(self, name: str, rho: float) -> None:
        self.name = name
        self.rho = rho
        self.fit_units = (FIT_UNIT,)
        self.fitted: list[pd.DataFrame] = []
        self.predicted: list[pd.DataFrame] = []

    def fit(self, frame: pd.DataFrame) -> None:
        self.fitted.append(frame.copy())

    def predict(self, frame: pd.DataFrame, kappas) -> pd.DataFrame:
        self.predicted.append(frame.copy())
        parts = []
        for kappa in kappas:
            part = frame[list(KEY_COLUMNS)].reset_index(drop=True)
            part["kappa"] = float(kappa)
            part["rho"] = float(self.rho)
            for column in ("d_hat", "d_median"):
                part[column] = 3.0
            part["r_hat"] = 2.0
            part["e_cap"] = 1.0
            part["e_need"] = 1.0
            for column in SURVIVAL_COLUMNS:
                part[column] = np.nan
            parts.append(part[list(TABLE_COLUMNS)])
        return pd.concat(parts, ignore_index=True)


@pytest.fixture
def setup(builds, tmp_path, monkeypatch):
    """The checked-out commit is the tag's; the stand-in models registered; a run folder."""
    monkeypatch.setattr(gate_run, "_head_commit", lambda: builds["tag"])
    built = []
    for name in (MODEL, LOOK_AHEAD):

        def make_model(matrix, name=name):
            model = RouteModel(name, matrix.rho)
            built.append(model)
            return model

        module = types.ModuleType(f"fleetreserve.models.{name}")
        module.make_model = make_model
        monkeypatch.setitem(sys.modules, module.__name__, module)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return {**builds, "out": out, "built": built, "tmp": tmp_path}


def _matrix(folder: Path, units, **changes) -> Path:
    document = matrix_document(NAMES)
    document["gate_units"] = list(units)
    document["required_units"] = 1
    document["n_calibration"] = COMMITTED["n_calibration"]  # the build's split rule
    document["calibration_half_below"] = COMMITTED["calibration_half_below"]
    document.update(changes)
    path = folder / f"matrix-{len(list(folder.iterdir()))}.json"
    path.write_text(json.dumps(document))
    return path


def _arguments(command: str, matrix: Path, tables, run, held_tables, held_run, model=MODEL):
    found = [] if command == "gate_run" else ["--model", model]
    found += ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    if held_tables is not None:
        found += ["--held-out-tables", str(held_tables)]
    if held_run is not None:
        found += ["--held-out-truth-run", str(held_run)]
    return found


def _main(command: str, arguments, root: Path) -> int:
    return {"gate_run": gate_run.main, "learn_run": learn_run.main}[command](arguments, root=root)


@contextmanager
def _opens():
    """Every path opened through `open`, `io.open` or `os.open` while inside."""
    seen = []
    real_open, real_os_open = builtins.open, os.open

    def watched_open(file, *args, **kwargs):
        seen.append(str(file))
        return real_open(file, *args, **kwargs)

    def watched_os_open(path, *args, **kwargs):
        seen.append(str(path))
        return real_os_open(path, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(builtins, "open", watched_open)
        patched.setattr(io, "open", watched_open)
        patched.setattr(os, "open", watched_os_open)
        yield seen


def _tables_read(seen) -> list[str]:
    return [path for path in seen if path.endswith(("-cars.csv", "-days.csv"))]


def _held_out_hashes(summary: dict, unit: str) -> dict:
    found = summary["units"][unit]["sha256"]
    return {f"{unit}-{kind}.csv": found[kind] for kind in ("cars", "days")}


# --- 1 and 2. scoring through the route ----------------------------------------------------------


def test_gate_runs_forecaster_is_refused_on_the_route(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(
        "gate_run", matrix, s["tables"], s["run"], s["held_tables"], s["held_run"]
    )
    with _opens() as seen:
        status = gate_run.main(arguments, root=repo)
    assert status == 2
    assert capsys.readouterr().out == (
        "COULD NOT RUN: the empirical forecaster of gate_run (reference other-days) builds a "
        "scored unit's forecasts from that unit's own other days, test days included; on the "
        "sealed route a scored unit's days reach a model only through its inputs, as in the cold "
        "start (its reading C12); nothing read\n"
    )
    assert _tables_read(seen) == []
    assert not (s["out"] / "gate.json").exists()


def test_learn_run_fits_on_the_development_build_and_scores_the_sealed_tables(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(
        "learn_run", matrix, s["tables"], s["run"], s["held_tables"], s["held_run"]
    )
    openings = heldout_loader.logged_openings(repo)
    assert learn_run.main(arguments, root=repo) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[0] == (
        f"{LABEL_TEXT} (tag commit {s['tag']}): the scored units' tables from the "
        f"held-out build {s['held_run'].name}, the fit units' ({FIT_UNIT}) from the development "
        f"build {s['run'].name}."
    )
    assert shown[1] == CHECKS_TEXT
    assert shown[2].startswith(f"Model {MODEL}: held-out scores and calibrated offers")
    (model,) = s["built"]
    assert [sorted(set(frame["site"])) for frame in model.fitted] == [["T"]]
    assert [sorted(set(frame["unit"])) for frame in model.predicted] == [[SCORED]]
    results = json.loads((s["out"] / "learn.json").read_text())
    gate = json.loads((s["out"] / "learn-gate.json").read_text())
    for document in (results, gate):
        route = document["sealed_route"]
        assert route["tag_commit"] == s["tag"] and route["fit_units"] == [FIT_UNIT]
        assert route["checks"] == CHECKS_TEXT
        assert route["label"] == LABEL_TEXT
        assert route["held_out_inputs"]["table_sha256"] == _held_out_hashes(s["summary"], SCORED)
        assert document["inputs"]["truth_run"] == s["run"].name
        assert set(document["inputs"]["table_sha256"]) == {
            f"{FIT_UNIT}-cars.csv",
            f"{FIT_UNIT}-days.csv",
        }
    assert results["units"][SCORED]["calibration_days_used"] == 15
    figures = results["units"][SCORED]["e_prime"]  # the share of days in E' is written here too
    assert results["e_prime_definition"].startswith("E' (plan T2)")
    assert figures and all(
        found["test_days"] == results["units"][SCORED]["test_days"] > 0
        and found["days_in_e_prime"] <= found["certified_test_days"] <= found["test_days"]
        for found in figures.values()
    )
    window = s["summary"]["windows"][SCORED]
    assert gate["units"][SCORED]["split"]["calibration"] == [
        f"{d} {SCORED}" for d in window["calibration_dates"]
    ]
    assert heldout_loader.logged_openings(repo) == openings


# --- 3. refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize("command", COMMANDS)
def test_off_the_tag_each_command_is_refused_before_any_table(setup, command, capsys):
    s = setup
    untagged = sealed_repo(s["tmp"] / "untagged", {}, tag=False)
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(command, matrix, s["tables"], s["run"], s["held_tables"], s["held_run"])
    with _opens() as seen:
        status = _main(command, arguments, untagged)
    shown = capsys.readouterr().out
    assert status == 2
    assert shown.startswith("COULD NOT RUN: REFUSED BY THE GATE: ")
    assert shown.endswith("; nothing read\n") and shown.count("\n") == 1
    assert _tables_read(seen) == []
    assert s["built"] == []  # refused before the model is loaded


@pytest.mark.parametrize("command", COMMANDS)
def test_units_of_two_kinds_are_refused_before_anything_is_read(repo, setup, command, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED, "nrel-from-may-2019"])
    arguments = _arguments(command, matrix, s["tables"], s["run"], s["held_tables"], s["held_run"])
    with _opens() as seen:
        status = _main(command, arguments, repo)
    shown = capsys.readouterr().out
    assert status == 2
    assert shown.startswith("COULD NOT RUN: the units are refused: the matrix's gate units ")
    assert "never both" in shown and shown.endswith("; nothing read\n")
    assert [path for path in seen if str(repo / "data") in path] == []
    assert s["built"] == []


@pytest.mark.parametrize("command", COMMANDS)
def test_the_two_tables_folders_swapped_are_refused(repo, setup, command, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(command, matrix, s["held_tables"], s["held_run"], s["tables"], s["run"])
    with _opens() as seen:
        status = _main(command, arguments, repo)
    shown = capsys.readouterr().out
    assert status == 2
    assert shown == (
        f"COULD NOT RUN: a folder is refused: the tables folder {s['held_tables']} lies in the "
        f"held-out build's folder {s['held_tables']}; nothing read\n"
    )
    assert [path for path in seen if str(repo / "data") in path] == []


@pytest.mark.parametrize("command", COMMANDS)
def test_the_held_out_arguments_with_development_units_are_refused(repo, setup, command, capsys):
    s = setup
    matrix = _matrix(s["tmp"], ["nrel-from-may-2019"])
    arguments = _arguments(command, matrix, s["tables"], s["run"], None, s["held_run"])
    with _opens() as seen:
        status = _main(command, arguments, repo)
    shown = capsys.readouterr().out
    assert status == 2
    assert shown == (
        "COULD NOT RUN: --held-out-truth-run is for held-out units, and none of the matrix's "
        "gate units ['nrel-from-may-2019'] is a held-out unit; nothing read\n"
    )
    assert [path for path in seen if str(repo / "data") in path] == []


@pytest.mark.parametrize("command", COMMANDS)
def test_held_out_units_need_the_held_out_truth_run(repo, setup, command, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(command, matrix, s["tables"], s["run"], s["held_tables"], None)
    with _opens() as seen:
        status = _main(command, arguments, repo)
    assert status == 2
    assert capsys.readouterr().out == "COULD NOT RUN: held-out units need --held-out-truth-run\n"
    assert _tables_read(seen) == []


@pytest.fixture(scope="module")
def held_setting(repo, builds):
    """The held-out build of the held setting, recorded at the tag's commit."""
    run = repo / "data" / "derived" / "runs" / "20261116T000001Z-wp14-heldout-truth-held"
    _held_out_build(repo, run, "--setting", "held")
    _record(run, builds["tag"])
    return run, sealed_route.held_out_tables(repo, "held")


@pytest.mark.parametrize("command", ["learn_run"])  # gate_run: refused before, by its forecaster
def test_builds_of_two_settings_are_refused_before_any_table(
    repo, setup, held_setting, command, capsys
):
    s = setup
    run, tables = held_setting
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(command, matrix, s["tables"], s["run"], tables, run)
    with _opens() as seen:
        status = _main(command, arguments, repo)
    assert status == 2
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: the held-out tables in {tables} are of the held setting but the "
        f"development build {s['run'].name} is of the main setting; a run reads the two builds "
        "of one setting\n"
    )
    assert [path for path in _tables_read(seen) if str(tables) in path] == []


# --- 4. look-ahead -------------------------------------------------------------------------------


def test_a_look_ahead_model_is_refused_on_the_route(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(
        "learn_run", matrix, s["tables"], s["run"], s["held_tables"], s["held_run"], LOOK_AHEAD
    )
    with _opens() as seen:
        status = learn_run.main(arguments, root=repo)
    shown = capsys.readouterr().out
    assert status == 1
    assert shown.startswith(f"REFUSED: model {LOOK_AHEAD} reads values of times after the decision")
    assert _tables_read(seen) == []


# --- 5. spellings of the sealed folder -----------------------------------------------------------


@pytest.fixture
def sealed_copies(repo, builds):
    """Copies of the scored unit's held-out tables and of the held-out run in the made-up sealed
    folder, only so that a path reaching the sealed folder would find files of the names a run
    opens; removed after the test, so no build of this file sees them."""
    sealed = repo / "data" / "heldout"
    placed = [sealed / "norway" / f"{SCORED}-{kind}.csv" for kind in ("cars", "days")]
    for path in placed:
        shutil.copyfile(builds["held_tables"] / path.name, path)
    run = sealed / builds["held_run"].name
    shutil.copytree(builds["held_run"], run)
    yield sealed
    for path in placed:
        path.unlink()
    shutil.rmtree(run)


SPELLED_ROUTES = ("sealed-folder", "file-links", "folder-link", "truth-run-inside")


def _spelled_route_in(route: str, spelling: str, repo: Path, s: dict):
    """(held-out tables, held-out truth run) of one route into the sealed folder spelled so."""
    written = repo / spelling / "heldout"
    if route == "sealed-folder":
        return written / "norway", s["held_run"]
    if route == "file-links":
        plain = s["tmp"] / "plain"
        plain.mkdir()
        for kind in ("cars", "days"):
            name = f"{SCORED}-{kind}.csv"
            (plain / name).symlink_to(written / "norway" / name)
        return plain, s["held_run"]
    if route == "folder-link":
        (s["tmp"] / "folder-link").symlink_to(written / "norway", target_is_directory=True)
        return s["tmp"] / "folder-link", s["held_run"]
    return s["held_tables"], written / s["held_run"].name


@pytest.mark.parametrize("spelling", ["DATA", "Data"])
@pytest.mark.parametrize("route", SPELLED_ROUTES)
@pytest.mark.parametrize("command", COMMANDS)
def test_the_sealed_folder_spelled_otherwise_is_refused_unopened(
    repo, setup, sealed_copies, command, route, spelling, capsys
):
    s = setup
    held_tables, held_run = _spelled_route_in(route, spelling, repo, s)
    assert (repo / spelling / "heldout" / "norway" / f"{SCORED}-cars.csv").is_file()
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(command, matrix, s["tables"], s["run"], held_tables, held_run)
    with watch(sealed_copies) as seen:
        status = _main(command, arguments, repo)
    shown = capsys.readouterr().out
    assert (status, seen) == (2, [])
    assert shown.startswith("COULD NOT RUN: ") and shown.count("\n") == 1


@pytest.mark.parametrize("spelling", ["DATA", "Data"])
@pytest.mark.parametrize("command", COMMANDS)
def test_the_held_out_build_spelled_otherwise_is_refused_as_development_tables(
    repo, setup, command, spelling, capsys
):
    s = setup
    written = repo / spelling / "derived" / s["held_tables"].name
    assert os.path.samefile(written, s["held_tables"])
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(command, matrix, written, s["held_run"], s["tables"], s["run"])
    with _opens() as seen:
        status = _main(command, arguments, repo)
    shown = capsys.readouterr().out
    assert status == 2
    assert shown == (
        f"COULD NOT RUN: a folder is refused: the tables folder {written} lies in the held-out "
        f"build's folder {s['held_tables']}; nothing read\n"
    )
    assert _tables_read(seen) == []


def test_the_top_folder_spelled_with_its_accent_decomposed_is_refused_by_the_folder_check(
    tmp_path,
):
    composed = "caf\u00e9"
    root = tmp_path / composed
    sealed = root / "data" / "heldout" / "norway"
    sealed.mkdir(parents=True)
    written = tmp_path / unicodedata.normalize("NFD", composed) / "data" / "heldout" / "norway"
    assert str(written) != str(sealed) and os.path.samefile(written, sealed)
    with watch(root / "data" / "heldout") as seen, pytest.raises(sealed_route.FoldersRefused):
        sealed_route.check_folders(root, tmp_path / "plain", (written,))
    assert seen == []


# --- 6. hard links -------------------------------------------------------------------------------


@pytest.mark.parametrize("command", COMMANDS)
def test_a_plain_folder_of_hard_links_to_sealed_files_is_refused_unopened(
    repo, setup, sealed_copies, command, capsys
):
    s = setup
    plain = s["tmp"] / "hard"
    plain.mkdir()
    for kind in ("cars", "days"):
        name = f"{SCORED}-{kind}.csv"
        os.link(sealed_copies / "norway" / name, plain / name)
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(command, matrix, s["tables"], s["run"], plain, s["held_run"])
    with watch(sealed_copies) as seen:
        status = _main(command, arguments, repo)
    shown = capsys.readouterr().out
    assert (status, seen) == (2, [])
    assert shown.startswith("COULD NOT RUN: ") and shown.count("\n") == 1


# --- 7. the split rule ---------------------------------------------------------------------------


def test_a_matrix_of_another_split_rule_is_refused_on_the_route(repo, setup, capsys):
    s = setup
    rule = s["summary"]["calibration_rule"]
    assert rule["seed"] == 1  # the build's, from the committed matrix
    matrix = _matrix(s["tmp"], [SCORED], split_seed=2)
    arguments = _arguments(
        "learn_run", matrix, s["tables"], s["run"], s["held_tables"], s["held_run"]
    )
    with _opens() as seen:
        status = learn_run.main(arguments, root=repo)
    assert status == 2
    assert capsys.readouterr().out == (
        "COULD NOT RUN: the matrix's split rule (split_seed 2, n_calibration "
        f"{rule['n_calibration']}, calibration_half_below {rule['half_below']}) is not the rule "
        f"the held-out build {s['held_run'].name} drew its calibration days by (split_seed 1, "
        f"n_calibration {rule['n_calibration']}, calibration_half_below {rule['half_below']}); "
        "on the sealed route every scored unit is split by its build's rule\n"
    )
    assert _tables_read(seen) == []
    assert not (s["out"] / "learn.json").exists()


# --- 8. no look at a scored unit's test days beyond scoring them ---------------------------------


def test_a_model_that_reads_the_scored_units_other_days_is_refused_on_the_route(
    repo, setup, capsys
):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(
        "learn_run", matrix, s["tables"], s["run"], s["held_tables"], s["held_run"], "empirical"
    )
    with _opens() as seen:
        status = learn_run.main(arguments, root=repo)
    assert status == 2
    assert capsys.readouterr().out == (
        "COULD NOT RUN: model empirical builds a scored unit's forecasts from that unit's own "
        "other days, test days included; on the sealed route a scored unit's days reach a model "
        "only through its inputs, as in the cold start (its reading C12); nothing read\n"
    )
    assert _tables_read(seen) == []


def test_kappa_is_not_selected_on_the_scored_units_test_days_on_the_route(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(
        "learn_run", matrix, s["tables"], s["run"], s["held_tables"], s["held_run"]
    )
    assert learn_run.main(arguments, root=repo) == 0
    shown = capsys.readouterr().out
    gate = json.loads((s["out"] / "learn-gate.json").read_text())
    for key in ("gate", "gate_tier_b"):
        assert "kappa_selected_per_unit" not in gate.get(key, {})
    assert "gate_tier_b" in gate  # tier B is scored, without a selection either
    assert gate["sealed_route"]["kappa_selection"] == sealed_route.NO_SELECTION
    lines = gate_summary.summary_lines(gate, load(matrix))
    assert not [line for line in lines if "kappa selected" in line or "largest offer" in line]
    assert "kappa selected" not in shown


@pytest.mark.parametrize("option", [["--limit-days", "10"], ["--only", NAMES[0]]])
def test_the_trial_options_are_refused_on_the_route(repo, setup, option, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(
        "gate_run", matrix, s["tables"], s["run"], s["held_tables"], s["held_run"]
    )
    with _opens() as seen:
        status = gate_run.main([*arguments, *option], root=repo)
    assert status == 2
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: {option[0]} is for trials on development units; a run on the sealed "
        "route scores every counting day of every scored unit under every configuration of its "
        "matrix; nothing read\n"
    )
    assert _tables_read(seen) == []


# --- 9. at a logged parsing fix after the tag ----------------------------------------------------


@pytest.fixture(scope="module")
def fixed(tmp_path_factory):
    """A made-up repository with a made-up plan frozen at the tag, the two builds made there, the
    a.s.r. file's layout logged as refused at the tag (by hand: no parser refuses these made-up
    files), a parsing fix of the a.s.r. loader after the tag (a correction of a test file of it,
    then its entry in section 14) and, on a branch from the tag, one of the Korean loader whose
    layout no line of the refusal log names."""
    folder = tmp_path_factory.mktemp("fix")
    files = sealed_files(folder / "made")
    root = sealed_repo(folder, files, tag=False)
    tag = freeze(root)
    derived = root / "data" / "derived"
    held_run = derived / "runs" / "20261116T000000Z-wp14-heldout-truth"
    _held_out_build(root, held_run)
    (derived / "development").mkdir()
    run, tables = write_build(derived / "development", {FIT_UNIT: gate_cars(40, "T")})
    log_refusal(root, ASR_FILE, files[ASR_FILE], tag)
    corrected = correction(root, "asr")
    entered = entry(root, corrected)
    # a later fix on the same line: the a.s.r. layout logged as refused again at the first fix
    # (by hand), a second correction of its loader and its entry (the opener's reading H6(e))
    log_refusal(root, ASR_FILE, files[ASR_FILE], entered, tag)
    second = correction(root, "asr", name="second")
    second_entered = entry(root, corrected, second)
    git(root, "checkout", "-q", "-b", "korea-fix", tag)
    korea_corrected = correction(root, "korea")
    return {
        "root": root,
        "tag": tag,
        "corrected": corrected,
        "entered": entered,
        "second_entered": second_entered,
        "korea_corrected": korea_corrected,
        "korea_entered": entry(root, korea_corrected),
        "run": run,
        "tables": tables,
        "held_run": held_run,
        "held_tables": sealed_route.held_out_tables(root),
    }


def _checked_out(fixed: dict, commit: str, monkeypatch) -> Path:
    """`commit` checked out and the two builds recorded at it; the made-up repository."""
    root = fixed["root"]
    git(root, "checkout", "-q", "--detach", commit)
    for run in (fixed["held_run"], fixed["run"]):
        _record(run, commit)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: commit)
    return root


def _fix_arguments(fixed: dict, setup: dict) -> list[str]:
    matrix = _matrix(setup["tmp"], [SCORED])
    f = fixed
    return _arguments("learn_run", matrix, f["tables"], f["run"], f["held_tables"], f["held_run"])


def test_learn_run_scores_at_a_logged_parsing_fix_and_states_the_fix(
    fixed, setup, monkeypatch, capsys
):
    tag, fix = fixed["tag"], fixed["entered"]
    root = _checked_out(fixed, fix, monkeypatch)
    assert learn_run.main(_fix_arguments(fixed, setup), root=root) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[0] == (
        f"{LABEL_TEXT} (tag commit {tag}; at the logged parsing fix {fix}): the scored "
        f"units' tables from the held-out build {fixed['held_run'].name}, the fit units' "
        f"({FIT_UNIT}) from the development build {fixed['run'].name}."
    )
    for name in ("learn.json", "learn-gate.json"):
        route = json.loads((setup["out"] / name).read_text())["sealed_route"]
        assert (route["tag_commit"], route["fix_commit"]) == (tag, fix)
    # the route of the commands whose units are named on their command lines (o3_run, drift_run)
    found = sealed_route.units_route(
        root, [SCORED], fixed["tables"], fixed["run"], None, fixed["held_run"]
    )
    assert (found.tag_commit, found.fix_commit) == (tag, fix)


def test_the_gate_again_refuses_a_run_whose_head_moved_to_a_later_fix(fixed, setup, monkeypatch):
    tag, first, later = fixed["tag"], fixed["entered"], fixed["second_entered"]
    root = _checked_out(fixed, first, monkeypatch)
    route = sealed_route.units_route(
        root, [SCORED], fixed["tables"], fixed["run"], None, fixed["held_run"]
    )
    assert (route.tag_commit, route.fix_commit) == (tag, first)
    sealed_route.gate_again(route, root)  # HEAD unchanged: passes
    git(root, "checkout", "-q", "--detach", later)
    found = sealed_route.require_gate(root)  # the later commit is a logged parsing fix itself
    assert (found, found.fix_commit) == (tag, later)
    with pytest.raises(sealed_route.RouteRefused) as refused:
        sealed_route.gate_again(route, root)
    assert str(refused.value) == (
        f"REFUSED BY THE GATE: the gate passes now at the logged parsing fix {later} of the tag "
        f"frozen-plan ({tag}), not at the logged parsing fix {first} of the tag frozen-plan "
        f"({tag}), where it passed when this run started; nothing read"
    )


def _refused_at(fixed, setup, commit, reason, monkeypatch, capsys) -> None:
    root = _checked_out(fixed, commit, monkeypatch)
    with _opens() as seen:
        status = learn_run.main(_fix_arguments(fixed, setup), root=root)
    assert status == 2
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: REFUSED BY THE GATE: HEAD {commit} is a commit after the tag "
        f"frozen-plan ({fixed['tag']}) and not a logged parsing fix of it: {reason} (reading "
        "H6); nothing read\n"
    )
    assert _tables_read(seen) == []
    assert setup["built"] == []  # refused before the model is loaded


def test_at_a_correction_without_its_entry_the_route_is_refused(fixed, setup, monkeypatch, capsys):
    reason = "HEAD is a loader correction (asr) without its entry in plan section 14 after it"
    _refused_at(fixed, setup, fixed["corrected"], reason, monkeypatch, capsys)


def test_at_a_fix_of_a_set_with_no_logged_refusal_the_route_is_refused(
    fixed, setup, monkeypatch, capsys
):
    reason = (
        "no layout refusal of the korea file is logged in data/heldout-refusals.log at the tag's "
        f"commit or at a commit of the fix before the correction {fixed['korea_corrected']}"
    )
    _refused_at(fixed, setup, fixed["korea_entered"], reason, monkeypatch, capsys)


# --- a file changed between its hash check and its parse -----------------------------------------


@pytest.mark.parametrize(
    "where", [("held_tables", f"{SCORED}-days.csv"), ("held_run", "truth-summary.json")]
)
def test_learn_run_refuses_a_held_out_file_changed_after_its_check(repo, setup, where, capsys):
    """The scored unit's table, or the held-out build's summary (first read by the split rule's
    check), holds other bytes (`swap_after_read.grown`) from just after its first read, the read
    its SHA-256 is checked on: the run could not run and writes nothing."""
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(
        "learn_run", matrix, s["tables"], s["run"], s["held_tables"], s["held_run"]
    )
    path = s[where[0]] / where[1]
    with swapped_after_first_read(path, grown(path)) as swaps:
        status = learn_run.main(arguments, root=repo)
    expected = table_refusal(path) if path.suffix == ".csv" else file_refusal(path)
    assert (status, capsys.readouterr().out) == (2, expected)
    assert swaps == [str(path.resolve())]
    assert not any(s["out"].iterdir())
