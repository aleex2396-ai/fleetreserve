"""`drift_run` (plan section 6, T10) on held-out units through the sealed route
(`sealed_route.units_route`).

The made-up repository is built by `fake_heldout.sealed_repo` from the made-up sealed files of
`heldout_truth_fake`, with two more files committed before its made-up tag `frozen-plan`: a
made-up step-size settings file `learn/drift-eta.json` in the layout `drift_settings` writes
(rule largest-below-first-failure, start zero, grouping site-type; tier A 0.5 for workplaces and
0.2 for the type other, tier B 1 and 0.1), and a made-up matrix `gate/matrix.json` (four
configurations at kappa 0.3, without and with the dial 0.02, each taking the forecast table's
energies). The
helpers are those of the other route tests
(`test_heldout_route_scoring`: the held-out truth build run once at the tag, a run recorded at
the tag's commit, the stand-in model fitted on caltech, here with the forecasts of
`test_heldout_route_ledger`'s, which give boxes, the watch of `open`). The held-out build
writes the tables of OSL_T (30 counting days from 6 January 2020, 15 of them the calibration days
its window was searched on) and TRO_R (6 days) to `data/derived/truth-heldout`; a made-up
development build (`gate_fake.write_build`: the training unit caltech, site T) lies in
`data/derived/development`. Both runs are recorded at the tag's commit; only the checked-out
commit of the package's own top folder is replaced by the tag's.

1. End to end, at the made-up tag, with the plan's 50 calibration days replaced by 5 (the made-up
   units have 30 and 6 counting days): exit 0; the summary's first line is the held-out label
   (reported only, no pass mark, nothing chosen) with the tag's commit and the two builds, its
   second the route's `checks` text, its third the step sizes with their file, its last line
   where they came from; the model is fitted on caltech's tables only and forecasts the scored
   units only; `drift-run.json` holds the route (the tag's commit, the held-out build's check
   with that build's own hashes, the fit units, the matrix's path and git object at the tag, the
   settings file's record with its git object,
   the step sizes of the type other) and `inputs` the development build's check; each unit's
   role is held-out and its paths are tier A at 0.2 and tier B at 0.1, start zero alone; the
   calibration days are the first 5 counting days in date order (6 to 10 January 2020 at OSL_T),
   not the build's 15 drawn calibration days. Hand-worked: the guards start at 0, so the first
   guarded day that offers abstains with loss 0 and the states before the second are tier A
   slot 0.2 x 0.10 = 0.02 and driver 0.2 x 0.05 = 0.01, tier B slot 0.1 x 0.10 = 0.01 (the
   workplace step sizes would give 0.05, 0.025 and 0.1). No sealed file is opened (no new
   opening logged, nothing under the sealed folder opened, by identity on disk). With the gate
   returning a fix commit as at a logged parsing fix (`fake_heldout.gate_claiming_a_fix`), the
   summary's first line and the route in `drift-run.json` name the tag's commit and the fix's.
2. At the plan's 50 calibration days the two made-up units have too few counting days (30 and
   6): exit 0, each unit's status says so and no day is guarded.
3. Refused, each with exit 2, one line, no output file and no table opened: away from the tag (a
   made-up repository without it; nothing under either sealed folder opened, no model built);
   held-out units with a development or a training unit; the two held-out options with
   development units; held-out units without `--held-out-truth-run`; `--etas` (a grid of step
   sizes) and `--calibration-days` on the route; a tagged repository whose tag's commit holds no
   settings file (no model built); a matrix file outside the top folder, and one inside it that
   the tag's commit does not hold (no model built; reading E24); the two tables folders swapped;
   the sealed folder named as the held-out tables (nothing under it opened). A matrix file that
   differs from the tag's is refused by `drift_run.route_matrix` (through the command the gate's
   clean-tree check refuses it first).
4. On the route the model is checked as off it: a model fitted on a development unit is refused
   (exit 1, no table opened); the empirical forecaster, which reads the unit's other days, could
   not run (exit 2, no table opened).
5. Tables of another build or setting, refused with exit 2 and no output file: the development
   build named as the held-out build (OSL_T is not in it); a held-out table changed in a copy of
   the build's folder; the held setting's held-out build beside the main development build (two
   settings; no held-out table opened); units of two site types (OSL_T and an a.s.r. unit),
   refused after the tables are read.
6. The settings file's record (`drift_run.settings_record`): a file of another rule, start or
   grouping than the plan's, one written from a development unit, and one that is not a settings
   file could not run; a site type the file holds no step size for could not run
   (`drift_run.route_steps`); `sealed_route.committed_text` refuses a path the tag's commit holds
   as no file (absent, or a folder) and, while the working copy's file holds other step sizes,
   returns the tag's text and git object; the names `drift_run` keeps for the file and the rule
   are those of `drift_settings`. On the route, with the gate's clean-tree check switched off and
   the working copy's settings file holding other step sizes, the run records and prints the
   tag's step sizes, SHA-256 and git object. The rule of 16 November 2026
   (`largest-offer-within-alpha`) is among the rules refused.
7. "Offers nothing" (reading E29): `drift_run.route_steps` takes the value "offers nothing" for
   a tier and site type and still refuses any other text or a step size of 0. On the route, a
   settings file giving the type other "offers nothing" in tier A, or in both tiers, runs with
   exit 0: that tier runs no path, the step sizes and each unit's summary lines say it offers
   nothing, and the other tier runs at its step size.
"""

import copy
import csv
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import pytest
from fake_heldout import CLAIMED_FIX, TAG, commit, gate_claiming_a_fix, git, sealed_repo
from gate_fake import gate_cars, matrix_document, write_build
from heldout_truth_fake import sealed_files
from sealed_watch import watch
from swap_after_read import grown, swapped_after_first_read, table_refusal
from test_heldout_route_scoring import (
    RouteModel,
    _held_out_build,
    _opens,
    _record,
    _tables_read,
)

from fleetreserve import (
    drift_path,
    drift_run,
    drift_settings,
    gate_run,
    heldout_loader,
    sealed_route,
)
from fleetreserve.days_from_truth import parse_unit_days
from fleetreserve.gate_config import TIER_LABELS, load
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import TRAIN_UNITS
from fleetreserve.truth_verify import CouldNotRun

UNITS = ("OSL_T", "TRO_R")
FIT_UNIT = "caltech"
DEVELOPMENT_UNIT = "nrel-from-may-2019"
MODEL = "route_model"
OFF_TRAINING = "route_model_off_training"  # a stand-in fitted on a development unit
NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.02-empirical",
    "main-rounded-hour-k0.3-eps0.02-oracle",
)
ETA = {"A": {"workplace": 0.5, "other": 0.2}, "B": {"workplace": 1.0, "other": 0.1}}
SETTINGS = {
    "about": drift_settings.ABOUT_BY_TYPE,
    "rule": "largest-below-first-failure",
    "rule_text": drift_settings.RULE_TEXTS["largest-below-first-failure"],
    "start": "zero",
    "grouping": "site-type",
    "grouping_text": drift_settings.GROUPING_TEXT,
    "units": list(TRAIN_UNITS),
    "model": "lgbm_tail_rotation",
    "dial": 0.02,
    "results": {
        "file": "drift-run.json",
        "sha256": "0" * 64,
        "run_folder": "20261116T000000Z-wp9-drift-run-lgbm_tail_rotation",
        "commit": "b" * 40,
    },
    "eta": ETA,
    "figures": {},
    "site_types": {},
}
# What a run on the route prints first: the label of a held-out drift-guard result.
LABEL = (
    "HELD-OUT RESULT, through the sealed-data gate at the tag frozen-plan, or at a logged parsing "
    "fix of it, with a clean tree: the time-ordered run of the drift guard on held-out units "
    "(plan section 6, T10), reported only, a measurement with no pass mark; nothing is chosen "
    "here: each tier's step size is the settings file's, written from a run on training units"
)
LAST = (
    "No step size is chosen here: each tier's is the settings file's, written by drift_settings "
    "from a run on training units; a held-out result, reported only."
)
CHECKS_TEXT = (
    "The route's checks stop ordinary mistakes by an honest operator; they are not a defence "
    "against deliberate evasion (links or copies made on purpose, a second working copy). The "
    "proof that the seal held is the plan's own: the committed hashes, the held-out loader, "
    "which opens a sealed file only at the tag frozen-plan, or at a logged parsing fix of it, "
    "with a clean tree, and its openings log."
)


def _settings_text(document=SETTINGS) -> str:
    return json.dumps(document, indent=1, sort_keys=True) + "\n"


def _matrix_text() -> str:
    """The made-up matrix: the configurations `NAMES`, each taking the forecast table's energies
    (`energy_bounds` empirical; the dial family needs the configurations to agree in every field
    but the dial) so that the stand-in's boxes are offered."""
    document = matrix_document(NAMES)
    for configuration in document["configurations"]:
        configuration["energy_bounds"] = "empirical"
    return json.dumps(document)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """The made-up repository, the settings file and the matrix committed before the tag (module
    text)."""
    folder = tmp_path_factory.mktemp("drift-route")
    root = sealed_repo(folder, sealed_files(folder / "made"), tag=False)
    (root / "learn").mkdir()
    (root / "learn" / "drift-eta.json").write_text(_settings_text())
    (root / "gate").mkdir(exist_ok=True)
    (root / "gate" / "matrix.json").write_text(_matrix_text())
    commit(root, "made-up step-size settings and matrix")
    git(root, "tag", TAG)
    return root


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
    uncommitted = derived / "matrix.json"  # ignored by git: the tree stays clean
    uncommitted.write_text(_matrix_text())
    return {
        "tag": tag,
        "run": run,
        "tables": tables,
        "matrix": repo / "gate" / "matrix.json",
        "matrix_blob": git(repo, "rev-parse", "HEAD:gate/matrix.json"),
        "uncommitted_matrix": uncommitted,
        "held_run": held_run,
        "held_tables": sealed_route.held_out_tables(repo),
        "summary": json.loads((held_run / "truth-summary.json").read_text()),
        "blob": git(repo, "rev-parse", "HEAD:learn/drift-eta.json"),
    }


class BoxModel(RouteModel):
    """The stand-in model fitted on caltech, with forecasts that give boxes (as in
    `test_heldout_route_ledger`): every car stays to hour 7 after τ (past the service hours) at
    2 kW, room 12 kWh, needing 6 kWh. `RouteModel`'s cars, which leave 3 hours after τ with room
    1 kWh and a need of 1 kWh, have empty forecast sets on the made-up days, so no day offers."""

    def predict(self, frame, kappas):
        table = super().predict(frame, kappas)
        table["d_hat"] = table["d_median"] = 7.0
        table["r_hat"], table["e_cap"], table["e_need"] = 2.0, 12.0, 6.0
        return table


class OffTrainingModel(RouteModel):
    """The stand-in model, declaring a development unit among its fit units."""

    def __init__(self, name: str, rho: float) -> None:
        super().__init__(name, rho)
        self.fit_units = (DEVELOPMENT_UNIT,)


@pytest.fixture
def setup(builds, tmp_path, monkeypatch):
    """The checked-out commit is the tag's; the stand-in models registered; a run folder."""
    monkeypatch.setattr(gate_run, "_head_commit", lambda: builds["tag"])
    built = []
    for name, kind in ((MODEL, BoxModel), (OFF_TRAINING, OffTrainingModel)):

        def make_model(matrix, name=name, kind=kind):
            model = kind(name, matrix.rho)
            built.append(model)
            return model

        module = types.ModuleType(f"fleetreserve.models.{name}")
        module.make_model = make_model
        monkeypatch.setitem(sys.modules, module.__name__, module)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return {**builds, "out": out, "built": built, "tmp": tmp_path}


def _arguments(s, units=UNITS, tables=None, run=None, held=(), model=MODEL, extra=()):
    return [
        "--model",
        model,
        "--dial",
        "0.02",
        "--matrix",
        str(s["matrix"]),
        "--units",
        *units,
        "--tables",
        str(s["tables"] if tables is None else tables),
        "--truth-run",
        str(s["run"] if run is None else run),
        *held,
        *extra,
    ]


def _held(tables=None, run=None) -> list[str]:
    found = [] if tables is None else ["--held-out-tables", str(tables)]
    return found + ([] if run is None else ["--held-out-truth-run", str(run)])


def _no_outputs(out: Path) -> bool:
    return not any(out.iterdir())


def _build_hashes(summary: dict, units) -> dict:
    return {
        f"{unit}-{kind}.csv": summary["units"][unit]["sha256"][kind]
        for unit in units
        for kind in ("cars", "days")
    }


# --- 1. end to end -------------------------------------------------------------------------------


def _rows(out: Path) -> list[dict]:
    with (out / drift_run.DAYS_NAME).open() as handle:
        return list(csv.DictReader(handle))


def test_drift_run_runs_held_out_units_through_the_route(repo, setup, monkeypatch, capsys):
    s = setup
    monkeypatch.setattr(drift_path, "N_CALIBRATION", 5)
    openings = heldout_loader.logged_openings(repo)
    with watch(repo / "data" / "heldout") as seen:
        status = drift_run.main(_arguments(s, held=_held(run=s["held_run"])), root=repo)
    shown = capsys.readouterr().out.splitlines()
    assert (status, seen) == (0, [])
    assert shown[0] == (
        f"{LABEL} (tag commit {s['tag']}): the scored units' tables from the held-out build "
        f"{s['held_run'].name}, the fit units' ({FIT_UNIT}) from the development build "
        f"{s['run'].name}."
    )
    assert shown[1] == CHECKS_TEXT
    assert shown[2] == (
        f"Step sizes from learn/drift-eta.json as the tag's commit holds it (git object "
        f"{s['blob']}; rule largest-below-first-failure, start zero, grouping site-type, units "
        f"{', '.join(TRAIN_UNITS)}, checked to be the plan's rule, start and grouping and "
        "training units; written, as the file records and nothing here checks, from the run "
        f"{SETTINGS['results']['run_folder']} at commit {'b' * 40}, model lgbm_tail_rotation, "
        "dial 0.02): OSL_T (other) tier A 0.2, tier B 0.1; TRO_R (other) tier A 0.2, tier B 0.1."
    )
    assert shown[3] == f"{drift_run.LABEL}."
    assert (
        "the first 5 counting days of each unit calibrate; step sizes tier A 0.2, tier B 0.1; "
        in shown[4]
    )
    assert shown[-1] == LAST
    model = s["built"]
    assert [sorted({site for f in m.fitted for site in f["site"]}) for m in model] == [["T"]] * 2
    assert [sorted({u for f in m.predicted for u in f["unit"]}) for m in model] == [
        ["OSL_T"],
        ["TRO_R"],
    ]
    document = json.loads((s["out"] / drift_run.RESULTS_NAME).read_text())
    assert document["failed_units"] == []
    assert document["inputs"]["truth_run"] == s["run"].name
    assert set(document["inputs"]["table_sha256"]) == {
        f"{FIT_UNIT}-cars.csv",
        f"{FIT_UNIT}-days.csv",
    }
    route = document["sealed_route"]
    assert route["kind"] == "held-out" and route["label"] == LABEL
    assert route["tag_commit"] == s["tag"] and route["checks"] == CHECKS_TEXT
    assert route["fit_units"] == [FIT_UNIT]
    assert route["matrix"] == {"file": "gate/matrix.json", "git_object": s["matrix_blob"]}
    assert route["held_out_inputs"] == {
        "setting": "main",
        "truth_run": s["held_run"].name,
        "table_sha256": _build_hashes(s["summary"], UNITS),
    }
    settings = route["settings"]
    assert (settings["file"], settings["git_object"]) == ("learn/drift-eta.json", s["blob"])
    assert (settings["rule"], settings["start"], settings["grouping"]) == drift_run.PLAN_RULE
    assert settings["eta"] == ETA and settings["units"] == list(TRAIN_UNITS)
    expected = {"site_type": "other", "eta": {"A": 0.2, "B": 0.1}}
    assert route["step_sizes"] == {"OSL_T": expected, "TRO_R": expected}
    assert document["run"]["starts"] == ["zero"] and document["run"]["etas"] == [0.1, 0.2]
    assert document["run"]["n_calibration"] == 5
    entries = document["units"]
    assert [entries[u]["role"] for u in UNITS] == ["held-out", "held-out"]
    for unit in UNITS:
        assert entries[unit]["paths"].keys() == {"A", "B"}
        assert {
            t: {st: list(e) for st, e in p.items()} for t, p in entries[unit]["paths"].items()
        } == {
            "A": {"zero": ["0.2"]},
            "B": {"zero": ["0.1"]},
        }
    osl = entries["OSL_T"]
    assert osl["counting_days"] == 30 and osl["guarded_days"] == 25
    assert (osl["calibration"]["first_day"], osl["calibration"]["last_day"]) == (
        "2020-01-06 OSL_T",
        "2020-01-10 OSL_T",
    )
    drawn = sorted(s["summary"]["windows"]["OSL_T"]["calibration_dates"])
    assert len(drawn) == 15 and drawn[:5] != [f"2020-01-{d:02d}" for d in range(6, 11)]
    assert entries["TRO_R"]["guarded_days"] == 1
    rows = [r for r in _rows(s["out"]) if r["unit"] == "OSL_T" and r["step"]]
    second = {}
    for tier in ("A", "B"):
        stepped = [r for r in rows if r["tier"] == tier]
        assert len(stepped) >= 2, (osl["days_not_offering"], osl["paths"][tier])
        assert stepped[0]["scale"] == "0.0" and stepped[0]["slot_loss"] == "0.0"
        second[tier] = (stepped[1]["state_slot"], stepped[1]["state_driver"])
    assert second == {"A": (repr(0.2 * 0.1), repr(0.2 * 0.05)), "B": (repr(0.1 * 0.1), "")}
    assert heldout_loader.logged_openings(repo) == openings


def test_at_a_parsing_fix_the_record_and_the_first_line_name_the_fix_commit(
    repo, setup, monkeypatch, capsys
):
    s = setup
    monkeypatch.setattr(drift_path, "N_CALIBRATION", 5)
    gate_claiming_a_fix(monkeypatch)
    assert drift_run.main(_arguments(s, held=_held(run=s["held_run"])), root=repo) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[0].startswith(
        f"{LABEL} (tag commit {s['tag']}; at the logged parsing fix {CLAIMED_FIX}): the scored "
    )
    route = json.loads((s["out"] / drift_run.RESULTS_NAME).read_text())["sealed_route"]
    assert (route["tag_commit"], route["fix_commit"]) == (s["tag"], CLAIMED_FIX)


def test_at_the_plans_50_calibration_days_the_made_up_units_have_too_few(repo, setup, capsys):
    s = setup
    assert drift_run.main(_arguments(s, held=_held(run=s["held_run"])), root=repo) == 0
    document = json.loads((s["out"] / drift_run.RESULTS_NAME).read_text())
    assert document["run"]["n_calibration"] == 50
    assert [document["units"][u]["status"] for u in UNITS] == [
        drift_path.too_few(30, 50),
        drift_path.too_few(6, 50),
    ]
    assert [document["units"][u]["guarded_days"] for u in UNITS] == [0, 0]
    assert capsys.readouterr().out.splitlines()[-1] == LAST


# --- 3. refusals ---------------------------------------------------------------------------------


def _refused(repo, s, arguments, capsys, root=None):
    """(the one line printed, the tables opened) of a run that must be refused (exit 2)."""
    with _opens() as seen:
        status = drift_run.main(arguments, root=repo if root is None else root)
    shown = capsys.readouterr().out
    assert status == 2
    assert shown.count("\n") == 1 and shown.startswith("COULD NOT RUN: ")
    assert _no_outputs(s["out"])
    return shown, _tables_read(seen)


def test_away_from_the_tag_held_out_units_could_not_run_and_open_nothing(repo, setup, capsys):
    s = setup
    untagged = sealed_repo(s["tmp"] / "untagged", sealed_files(s["tmp"] / "made"), tag=False)
    arguments = _arguments(s, held=_held(s["held_tables"], s["held_run"]))
    with watch(repo / "data" / "heldout", untagged / "data" / "heldout") as sealed:
        shown, tables = _refused(repo, s, arguments, capsys, root=untagged)
    assert shown.startswith("COULD NOT RUN: REFUSED BY THE GATE: ")
    assert shown.endswith("; nothing read\n")
    assert (tables, sealed, s["built"]) == ([], [], [])


@pytest.mark.parametrize("other", [DEVELOPMENT_UNIT, "caltech"])
def test_held_out_units_with_other_units_are_refused(repo, setup, other, capsys):
    s = setup
    arguments = _arguments(s, units=("OSL_T", other), held=_held(run=s["held_run"]))
    shown, tables = _refused(repo, s, arguments, capsys)
    assert shown == (
        f"COULD NOT RUN: the units are refused: the units named ['OSL_T', '{other}'] mix held-out "
        "units (['OSL_T']) with others; a run scores held-out units or others, never both: their "
        "tables come from two different truth builds; nothing read\n"
    )
    assert tables == []


def test_the_held_out_options_with_development_units_are_refused(repo, setup, capsys):
    s = setup
    held = _held(s["held_tables"], s["held_run"])
    shown, tables = _refused(repo, s, _arguments(s, units=(DEVELOPMENT_UNIT,), held=held), capsys)
    assert shown == (
        "COULD NOT RUN: --held-out-tables and --held-out-truth-run are for held-out units, and "
        f"none of the units named ['{DEVELOPMENT_UNIT}'] is a held-out unit; nothing read\n"
    )
    assert tables == []


def test_held_out_units_need_the_held_out_truth_run(repo, setup, capsys):
    s = setup
    shown, tables = _refused(repo, s, _arguments(s, held=_held(s["held_tables"])), capsys)
    assert shown == "COULD NOT RUN: held-out units need --held-out-truth-run\n"
    assert tables == []


@pytest.mark.parametrize(
    ("extra", "named", "verb"),
    [
        (["--etas", "0.1", "0.2"], "--etas", "is"),
        (["--etas", "0.2"], "--etas", "is"),
        (["--calibration-days", "50"], "--calibration-days", "is"),
        (["--etas", "0.2", "--calibration-days", "10"], "--etas and --calibration-days", "are"),
    ],
)
def test_a_grid_of_step_sizes_or_a_calibration_count_is_refused_on_the_route(
    repo, setup, extra, named, verb, capsys
):
    s = setup
    arguments = _arguments(s, held=_held(run=s["held_run"]), extra=extra)
    shown, tables = _refused(repo, s, arguments, capsys)
    assert shown == (
        f"COULD NOT RUN: {named} {verb} for runs on training and development units; on the sealed "
        "route each tier's step size is that of the settings file learn/drift-eta.json as the "
        "tag's commit holds it and the first 50 counting days calibrate (plan section 6, T10; "
        "readings E17 and E19); nothing read\n"
    )
    assert (tables, s["built"]) == ([], [])


@pytest.fixture(scope="module")
def bare(tmp_path_factory):
    """A made-up repository at its made-up tag whose commit holds no settings file."""
    folder = tmp_path_factory.mktemp("drift-route-bare")
    return sealed_repo(folder, sealed_files(folder / "made"))


def test_no_settings_file_at_the_tag_could_not_run_before_any_table(bare, setup, capsys):
    s = setup
    assert not (bare / "learn" / "drift-eta.json").exists()
    arguments = _arguments(s, held=_held(s["held_tables"], s["held_run"]))
    with watch(bare / "data" / "heldout") as sealed:
        shown, tables = _refused(bare, s, arguments, capsys)
    tag = git(bare, "rev-parse", "HEAD")
    assert shown == (
        f"COULD NOT RUN: learn/drift-eta.json is not a file of the tag's commit {tag}; on the "
        "sealed route a setting is read only from the file committed at the tag; nothing read\n"
    )
    assert (tables, sealed, s["built"]) == ([], [], [])


def test_a_matrix_outside_the_top_folder_or_not_committed_could_not_run(repo, setup, capsys):
    s = setup
    outside = s["tmp"] / "matrix.json"
    outside.write_text(_matrix_text())
    for matrix, expected in (
        (
            outside,
            f"COULD NOT RUN: the matrix file {outside} lies outside the top folder {repo}; on the "
            "sealed route the dial's configuration and its kappa come from a matrix file the "
            "tag's commit holds (reading E24); nothing read\n",
        ),
        (
            s["uncommitted_matrix"],
            "COULD NOT RUN: data/derived/matrix.json is not a file of the tag's commit "
            f"{s['tag']}; on the sealed route a setting is read only from the file committed at "
            "the tag; nothing read\n",
        ),
    ):
        arguments = _arguments(s, held=_held(run=s["held_run"])) + ["--matrix", str(matrix)]
        shown, tables = _refused(repo, s, arguments, capsys)
        assert shown == expected
        assert (tables, s["built"]) == ([], [])


def test_a_matrix_differing_from_the_tags_could_not_run(repo, builds):
    path = builds["matrix"]
    route = types.SimpleNamespace(tag_commit=builds["tag"])
    assert drift_run.route_matrix(repo, route, path) == {
        "file": "gate/matrix.json",
        "git_object": builds["matrix_blob"],
    }
    kept = path.read_bytes()
    try:
        path.write_text(_matrix_text().replace('"driver_risk": 0.02', '"driver_risk": 0.05'))
        with pytest.raises(CouldNotRun) as refused:
            drift_run.route_matrix(repo, route, path)
    finally:
        path.write_bytes(kept)
    assert str(refused.value) == (
        "the matrix file gate/matrix.json differs from the file the tag's commit holds (reading "
        "E24)"
    )


def test_the_two_tables_folders_swapped_are_refused(repo, setup, capsys):
    s = setup
    arguments = _arguments(
        s, tables=s["held_tables"], run=s["held_run"], held=_held(s["tables"], s["run"])
    )
    shown, tables = _refused(repo, s, arguments, capsys)
    assert shown == (
        f"COULD NOT RUN: a folder is refused: the tables folder {s['held_tables']} lies in the "
        f"held-out build's folder {s['held_tables']}; nothing read\n"
    )
    assert tables == []


def test_the_sealed_folder_named_as_the_held_out_tables_is_refused_unopened(repo, setup, capsys):
    s = setup
    sealed = repo / "data" / "heldout"
    arguments = _arguments(s, held=_held(sealed / "norway", s["held_run"]))
    with watch(sealed) as seen:
        shown, tables = _refused(repo, s, arguments, capsys)
    assert shown.startswith(f"COULD NOT RUN: a folder is refused: the folder {sealed / 'norway'}")
    assert (tables, seen) == ([], [])


# --- 4. the model on the route -------------------------------------------------------------------


def test_a_model_fitted_on_a_development_unit_is_refused_on_the_route(repo, setup, capsys):
    s = setup
    arguments = _arguments(s, held=_held(run=s["held_run"]), model=OFF_TRAINING)
    with _opens() as seen:
        status = drift_run.main(arguments, root=repo)
    shown = capsys.readouterr().out
    assert status == 1
    assert shown == (
        f"REFUSED: model {OFF_TRAINING} declares fit_units ['{DEVELOPMENT_UNIT}'], which are not "
        f"training units; a model is fitted on training units only ({', '.join(TRAIN_UNITS)})\n"
    )
    assert _tables_read(seen) == [] and _no_outputs(s["out"])


def test_a_model_that_reads_the_units_other_days_could_not_run_on_the_route(repo, setup, capsys):
    s = setup
    arguments = _arguments(s, held=_held(run=s["held_run"]), model="empirical")
    shown, tables = _refused(repo, s, arguments, capsys)
    assert shown == (
        "COULD NOT RUN: model empirical reads the other days of the unit OSL_T, later days "
        "included; a run in time order must not see them (reading E7)\n"
    )
    assert tables == []


# --- 5. tables of another build or setting -------------------------------------------------------


def test_the_development_build_named_as_the_held_out_build_is_refused(repo, setup, capsys):
    s = setup
    shown, _ = _refused(repo, s, _arguments(s, held=_held(s["tables"], s["run"])), capsys)
    assert shown == f"COULD NOT RUN: OSL_T is not in the recorded truth build {s['run'].name}\n"


def test_a_held_out_table_that_differs_from_its_build_is_refused(repo, setup, capsys):
    s = setup
    copy_folder = s["tmp"] / "copied-tables"
    shutil.copytree(s["held_tables"], copy_folder)
    changed = copy_folder / "OSL_T-cars.csv"
    changed.write_bytes(changed.read_bytes() + b"\n")
    shown, _ = _refused(repo, s, _arguments(s, held=_held(copy_folder, s["held_run"])), capsys)
    assert shown == (
        "COULD NOT RUN: OSL_T-cars.csv differs from the table the recorded build "
        f"{s['held_run'].name} wrote\n"
    )


@pytest.fixture(scope="module")
def held_setting(repo, builds):
    """The held-out build of the held setting, recorded at the tag's commit."""
    run = repo / "data" / "derived" / "runs" / "20261116T000001Z-wp14-heldout-truth-held"
    _held_out_build(repo, run, "--setting", "held")
    _record(run, builds["tag"])
    return run, sealed_route.held_out_tables(repo, "held")


def test_builds_of_two_settings_are_refused_before_a_held_out_table(
    repo, setup, held_setting, capsys
):
    s = setup
    run, tables = held_setting
    shown, read = _refused(repo, s, _arguments(s, held=_held(tables, run)), capsys)
    assert shown == (
        f"COULD NOT RUN: the held-out tables in {tables} are of the held setting but the "
        f"development build {s['run'].name} is of the main setting; a run reads the two builds "
        "of one setting\n"
    )
    assert [path for path in read if str(tables) in path] == []


def test_units_of_two_site_types_could_not_run(repo, setup, capsys):
    s = setup
    units = ("OSL_T", "asr-before-march-2023")
    shown, _ = _refused(repo, s, _arguments(s, units=units, held=_held(run=s["held_run"])), capsys)
    assert shown == (
        f"COULD NOT RUN: the units {list(units)} are of the site types ['other', 'workplace']; "
        "the dial is fixed per site type (plan section 6) and this command takes one --dial, so a "
        "run on the sealed route names units of one site type (reading E20)\n"
    )


# --- 6. the settings file ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rule", "largest-offer"),
        ("rule", "largest-offer-within-alpha"),  # the rule of 16 November 2026
        ("start", "calibrated"),
        ("grouping", "all-units"),
    ],
)
def test_a_settings_file_not_of_the_plans_rule_could_not_run(field, value):
    document = copy.deepcopy(SETTINGS)
    document[field] = value
    if field == "grouping":
        document["eta"] = {"A": 0.5, "B": 1.0}
    found = dict(zip(("rule", "start", "grouping"), drift_run.PLAN_RULE, strict=True))
    found[field] = value
    with pytest.raises(CouldNotRun) as refused:
        drift_run.settings_record(_settings_text(document), "c" * 40)
    assert str(refused.value) == (
        f"learn/drift-eta.json holds step sizes of the rule {found['rule']}, start "
        f"{found['start']}, grouping {found['grouping']}; plan section 6 (T10) sets them by the "
        "rule largest-below-first-failure, start zero (beta_1 = 0), grouping site-type (reading "
        "E17); nothing read"
    )


def test_a_settings_file_written_from_another_unit_could_not_run():
    document = copy.deepcopy(SETTINGS)
    document["units"] = ["caltech", DEVELOPMENT_UNIT]
    with pytest.raises(CouldNotRun, match="step sizes are chosen on training units only"):
        drift_run.settings_record(_settings_text(document), "c" * 40)


@pytest.mark.parametrize("text", ["not json", "[]", json.dumps({"rule": "x"})])
def test_a_file_that_is_no_settings_file_could_not_run(text):
    with pytest.raises(CouldNotRun, match="is not a settings file of drift_settings"):
        drift_run.settings_record(text, "c" * 40)


def test_a_site_type_the_settings_file_lacks_could_not_run(builds):
    document = copy.deepcopy(SETTINGS)
    del document["eta"]["B"]["other"]
    record = drift_run.settings_record(_settings_text(document), "c" * 40)
    matrix = load(builds["matrix"])
    read = {
        name: (builds["held_tables"] / name).read_bytes()
        for name in ("TRO_R-cars.csv", "TRO_R-days.csv")
    }
    days = parse_unit_days(
        read, "TRO_R", rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )
    assert len(days.days) == 6
    with pytest.raises(CouldNotRun, match="holds no step size of tier B for the site type other"):
        drift_run.route_steps(record, {"TRO_R": days}, ("TRO_R",))


def test_committed_text_refuses_a_file_the_tags_commit_does_not_hold(bare):
    tag = git(bare, "rev-parse", "HEAD")
    with pytest.raises(CouldNotRun, match="is not a file of the tag's commit"):
        sealed_route.committed_text(bare, tag, Path("learn", "drift-eta.json"))
    with pytest.raises(CouldNotRun, match="is not a file of the tag's commit"):
        sealed_route.committed_text(bare, tag, Path("src"))  # a folder, not a file


def test_committed_text_reads_the_tags_file_not_the_working_copy(repo, builds):
    path = repo / "learn" / "drift-eta.json"
    kept = path.read_bytes()
    try:
        path.write_text(_settings_text({**SETTINGS, "eta": {"A": 9.0, "B": 9.0}}))
        text, blob = sealed_route.committed_text(repo, builds["tag"], Path(path.relative_to(repo)))
    finally:
        path.write_bytes(kept)
    assert (text, blob) == (_settings_text(), builds["blob"])
    assert path.read_text() == text


def test_on_the_route_the_step_sizes_are_the_tags_text_not_the_working_copys(
    repo, setup, monkeypatch, capsys
):
    """The gate's clean-tree check switched off (`heldout_loader._check_clean`), the working
    copy's settings file holds other step sizes (tier A 0.7 and tier B 0.9 for the type other)
    than the tag's commit (0.2 and 0.1): the run on the route takes the text from git's objects
    at the tag, so its record (step sizes, SHA-256 and git object) and its summary are the
    committed file's. A run whose text came from the working copy while its git object came
    from the commit would record 0.7 and 0.9 under the tag's git object."""
    s = setup
    monkeypatch.setattr(heldout_loader, "_check_clean", lambda root: None)
    path = repo / "learn" / "drift-eta.json"
    kept = path.read_bytes()
    other = copy.deepcopy(SETTINGS)
    other["eta"] = {"A": {"workplace": 0.5, "other": 0.7}, "B": {"workplace": 1.0, "other": 0.9}}
    try:
        path.write_text(_settings_text(other))
        status = drift_run.main(_arguments(s, held=_held(run=s["held_run"])), root=repo)
    finally:
        path.write_bytes(kept)
    shown = capsys.readouterr().out.splitlines()
    assert status == 0, shown
    route = json.loads((s["out"] / drift_run.RESULTS_NAME).read_text())["sealed_route"]
    settings = route["settings"]
    assert (settings["eta"], settings["git_object"]) == (ETA, s["blob"])
    assert settings["sha256"] == hashlib.sha256(_settings_text().encode()).hexdigest()
    expected = {"site_type": "other", "eta": {"A": 0.2, "B": 0.1}}
    assert route["step_sizes"] == {"OSL_T": expected, "TRO_R": expected}
    assert shown[2].endswith(
        "OSL_T (other) tier A 0.2, tier B 0.1; TRO_R (other) tier A 0.2, tier B 0.1."
    )


def test_route_steps_takes_offers_nothing_and_no_other_text(builds):
    matrix = load(builds["matrix"])
    read = {
        name: (builds["held_tables"] / name).read_bytes()
        for name in ("TRO_R-cars.csv", "TRO_R-days.csv")
    }
    days = parse_unit_days(
        read, "TRO_R", rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )
    document = copy.deepcopy(SETTINGS)
    document["eta"]["A"]["other"] = drift_settings.OFFERS_NOTHING
    record = drift_run.settings_record(_settings_text(document), "c" * 40)
    assert drift_run.route_steps(record, {"TRO_R": days}, ("TRO_R",)) == {
        "TRO_R": {"site_type": "other", "eta": {"A": "offers nothing", "B": 0.1}}
    }
    for value, message in (("nothing", "holds no step size of tier A"), (0, "finite and above 0")):
        document["eta"]["A"]["other"] = value
        record = drift_run.settings_record(_settings_text(document), "c" * 40)
        with pytest.raises(CouldNotRun, match=message):
            drift_run.route_steps(record, {"TRO_R": days}, ("TRO_R",))


@pytest.mark.parametrize("nothing", [("A",), ("A", "B")])
def test_a_tier_that_offers_nothing_runs_no_path_and_is_reported_as_such(
    repo, setup, monkeypatch, capsys, nothing
):
    """Reading E29: the settings file gives the type other "offers nothing" in tier A (or in both
    tiers), as the plan's rule writes it where the smallest step size already fails. The route
    reads that file's record (`drift_run.route_settings` replaced by `settings_record` on this
    text; the rest of the route as in item 1): exit 0, not "could not run"; the step sizes name
    "offers nothing"; the tier runs no path (no paths, no day rows, not among `run.etas`) and each
    unit's summary lines say it offers nothing; the other tier, if any, runs at 0.1. OSL_T keeps
    its 25 guarded counting days."""
    s = setup
    monkeypatch.setattr(drift_path, "N_CALIBRATION", 5)
    document = copy.deepcopy(SETTINGS)
    for tier in nothing:
        document["eta"][tier]["other"] = drift_settings.OFFERS_NOTHING
    text = _settings_text(document)
    monkeypatch.setattr(
        drift_run, "route_settings", lambda root, route: drift_run.settings_record(text, "c" * 40)
    )
    status = drift_run.main(_arguments(s, held=_held(run=s["held_run"])), root=repo)
    shown = capsys.readouterr().out.splitlines()
    assert status == 0, shown
    eta = {t: "offers nothing" if t in nothing else 0.1 for t in drift_path.TIERS}  # B: 0.1
    steps = ", ".join(f"tier {t} {'offers nothing' if t in nothing else '0.1'}" for t in eta)
    assert shown[2].endswith(f"): OSL_T (other) {steps}; TRO_R (other) {steps}.")
    assert f"; step sizes {steps}; " in shown[4]
    said = [
        f"  {TIER_LABELS[t]}: offers nothing on every guarded day: the settings file gives the "
        "site type other no step size, the smallest step size of its run on training units "
        "already failing (plan section 6, T10; reading E29)."
        for t in nothing
    ]
    assert [line for line in shown if "offers nothing on every guarded day" in line] == said * 2
    result = json.loads((s["out"] / drift_run.RESULTS_NAME).read_text())
    assert result["failed_units"] == []
    route = result["sealed_route"]
    assert route["settings"]["eta"]["A"]["other"] == "offers nothing"
    expected = {"site_type": "other", "eta": eta}
    assert route["step_sizes"] == {"OSL_T": expected, "TRO_R": expected}
    ran = [t for t in drift_path.TIERS if t not in nothing]
    assert result["run"]["etas"] == ([0.1] if ran else [])
    osl = result["units"]["OSL_T"]
    assert list(osl["paths"]) == ran and osl["guarded_days"] == 25
    assert osl["calibration"]["last_day"] == "2020-01-10 OSL_T"
    assert {r["tier"] for r in _rows(s["out"])} == set(ran)


def test_the_names_kept_for_the_settings_file_and_the_rule_are_drift_settings_own():
    assert TOP / drift_run.SETTINGS == drift_settings.ETA
    rule, start, grouping = drift_run.PLAN_RULE
    assert rule in drift_settings.RULES and start in drift_path.STARTS
    assert grouping == drift_settings.SITE_TYPE


# --- a table changed between its hash check and its parse ----------------------------------------


def test_a_held_out_table_changed_after_its_check_is_refused(repo, setup, monkeypatch, capsys):
    """The table holds other bytes (`swap_after_read.grown`) from just after its first read, the
    read its SHA-256 is checked on: the run could not run and writes nothing."""
    s = setup
    monkeypatch.setattr(drift_path, "N_CALIBRATION", 5)
    path = s["held_tables"] / "TRO_R-days.csv"
    with swapped_after_first_read(path, grown(path)) as swaps:
        shown, _ = _refused(repo, s, _arguments(s, held=_held(run=s["held_run"])), capsys)
    assert (swaps, shown) == ([str(path.resolve())], table_refusal(path))
