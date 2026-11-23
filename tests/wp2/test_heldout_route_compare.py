"""`compare_run` scoring held-out units through the sealed route (`sealed_route`).

The made-up repository at its made-up tag `frozen-plan`, its made-up sealed files, the held-out
truth build recorded at the tag and the made-up development build (the training unit caltech,
site T) are those of `test_heldout_route_scoring` (its repository, built again here by the same
two calls, its fixtures `builds` and `held_setting`, its stand-in model, fitted on caltech, every
car leaving 3 hours after τ, and its helpers); the checked-out commit of the package's own top
folder is replaced by the tag's. The matrix is the committed one with the dial family at kappa
0.3 without rows and at the dial 0.03, each with its oracle partner, and the build's split rule
(seed 1, 100 calibration days or half the days under 200). The held-out units scored are OSL_T
(30 counting days, 15 calibration days) and TRO_R (6 counting days, 3 calibration days); the
methods are `re-solve`, `fixed-plan`, `llf` and `also-x` (the linear-programme forms of ASA solve
four or five programmes per hour of each path, too slow for these six-hour days; item 2 runs
`asa` on TRO_R alone).

1. A run on the route ends with exit 0: the model is fitted on the development build's caltech
   tables only and forecasts the scored units only; the scored units' tables are the held-out
   build's (their SHA-256 those the build recorded); each unit is scored on the one split the build
   drew (its calibration days the build's `calibration_dates`, seed 1, one split, every split-table
   row of seed 1); the label says a held-out result and names what is not computed; the route is
   recorded under `sealed_route` (the tag's commit, the held-out build's check, the fit units, the
   checks' text) with the readings H1 to H6; the summary's first line names the tag and the two
   builds, its second is the checks' text, its third the one split and the held-out label, and the
   paired differences say that one split gives no interval; no sealed file is opened and the
   held-out loader logs no opening.
2. `asa` and `llf` (outcome O1's A and B) score TRO_R on the route, in both tiers and variants.
3. Off the tag (a made-up repository without it) the run could not run (exit 2, one line), before
   the model is loaded and before any table or sealed file is opened.
4. Units of two kinds in the matrix could not run (exit 2), nothing read.
5. The leak rules, each exit 2 with one line and no table read unless said: a model that builds a
   scored unit's forecasts from its own other days (`empirical`); `--splits` other than 1 (and the
   default 200); a matrix of another split rule than the build's; and, after the tables are read,
   a matrix whose readings make other counting days than the build's (`small_fleet` "drop" on a
   unit with a day whose fleet holds 2 cars, the made-up sealed files of
   `test_heldout_calibration_days`), so its drawn calibration days are not the build's.
6. Tables of another build or setting could not run (exit 2, one line, no table read): the two
   tables folders swapped, the held-out arguments with development units, no held-out truth run,
   and the held setting's held-out build with the main development build.
7. The no-pause sensitivity on the route keeps the held-out label, with the sensitivity named after
   it, and writes its files under the sensitivity's names.
8. A held-out build whose calibration days were drawn with split seed 2 (the committed matrix with
   that seed, built at the tag into a tables folder of its own): the route run with a matrix of
   seed 2 scores seed 2, the build's draw, whose calibration days differ from seed 1's; a run that
   scored seeds 1 to N would score seed 1 here.

Every made-up held-out day abstains (the stand-in's forecast sets are empty on these days, so no
car is steered): these tests show which tables, split, days and labels a run on the route uses
and what it refuses, and no offer above 0 is computed on the route here.
"""

import csv
import json
import sys
import types
from pathlib import Path

import pytest
import test_heldout_route_scoring as scoring
from fake_heldout import git, sealed_repo
from gate_fake import COMMITTED, gate_cars, matrix_document, write_build
from heldout_truth_fake import sealed_files
from sealed_watch import watch
from swap_after_read import grown, swapped_after_first_read, table_refusal
from test_heldout_calibration_days import sealed_files as short_day_files

from fleetreserve import compare_run, gate_run, heldout_loader, sealed_route


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """The made-up repository of `test_heldout_route_scoring`, built again here so that its
    root is seen to be made up (`test_heldout_routes`, the opener's made-up roots)."""
    folder = tmp_path_factory.mktemp("route")
    return sealed_repo(folder, sealed_files(folder / "made"))


# The other fixtures of `test_heldout_route_scoring`, used here by their names.
builds = scoring.builds
held_setting = scoring.held_setting

NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.03-empirical",
    "main-rounded-hour-k0.3-eps0.03-oracle",
)
SCORED = ("OSL_T", "TRO_R")
METHODS = ["re-solve", "fixed-plan", "llf", "also-x"]
DIAL = "0.03"
MODEL = scoring.MODEL
FIT_UNIT = scoring.FIT_UNIT


@pytest.fixture
def setup(builds, tmp_path, monkeypatch):
    """The checked-out commit is the tag's; the stand-in model registered; a run folder."""
    monkeypatch.setattr(gate_run, "_head_commit", lambda: builds["tag"])
    built = []

    def make_model(matrix):
        model = scoring.RouteModel(MODEL, matrix.rho)
        built.append(model)
        return model

    module = types.ModuleType(f"fleetreserve.models.{MODEL}")
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


def _arguments(
    matrix, tables, run, held_tables, held_run, *, model=MODEL, splits="1", methods=METHODS
):
    found = ["--model", model, "--dial", DIAL, "--matrix", str(matrix)]
    found += ["--tables", str(tables), "--truth-run", str(run)]
    if splits is not None:
        found += ["--splits", splits]
    if methods is not None:
        found += ["--methods", *methods]
    if held_tables is not None:
        found += ["--held-out-tables", str(held_tables)]
    if held_run is not None:
        found += ["--held-out-truth-run", str(held_run)]
    return found


def _route_arguments(s: dict, units=SCORED, **options):
    matrix = _matrix(s["tmp"], units)
    return _arguments(matrix, s["tables"], s["run"], s["held_tables"], s["held_run"], **options)


def _sealed(root: Path) -> Path:
    return root / "data" / "heldout"


# --- 1 and 2. scoring through the route ----------------------------------------------------------


def test_compare_run_scores_the_sealed_tables_on_the_builds_own_split(repo, setup, capsys):
    s = setup
    openings = heldout_loader.logged_openings(repo)
    with watch(_sealed(repo)) as sealed_seen:
        status = compare_run.main(_route_arguments(s), root=repo)
    assert (status, sealed_seen) == (0, [])
    assert heldout_loader.logged_openings(repo) == openings
    shown = capsys.readouterr().out.splitlines()
    assert shown[0] == (
        f"{sealed_route.LABEL} (tag commit {s['tag']}): the scored units' tables from the "
        f"held-out build {s['held_run'].name}, the fit units' ({FIT_UNIT}) from the development "
        f"build {s['run'].name}."
    )
    assert shown[1] == scoring.CHECKS_TEXT
    assert shown[2] == (
        f"Comparison methods for model {MODEL}, dial 0.03: the one split per unit its held-out "
        f"build drew (seed 1), main truth setting, kappa 0.3. {compare_run.ROUTE_LABEL}."
    )
    assert compare_run.ROUTE_LABEL.startswith(
        "comparison methods on held-out units through the sealed-data route: a held-out result"
    )
    assert "not computed here: the 95% bootstrap intervals over days" in compare_run.ROUTE_LABEL
    assert compare_run.ROUTE_INTERVAL in shown  # one split gives no interval
    assert not [line for line in shown if "development evidence" in line]
    (model,) = s["built"]
    assert [sorted(set(frame["site"])) for frame in model.fitted] == [["T"]]
    assert [sorted(set(frame["unit"])) for frame in model.predicted] == [sorted(SCORED)]
    document = json.loads((s["out"] / compare_run.RESULTS_NAME).read_text())
    assert document["label"] == document["definitions"]["label"] == compare_run.ROUTE_LABEL
    readings = document["definitions"]["route_readings"]
    assert set(readings) == {"H1", "H2", "H3", "H4", "H5", "H6"}
    assert readings["H6"].startswith("one --dial for every scored unit")
    assert document["definitions"]["splits"].startswith("1 split per unit on the sealed route")
    route = document["sealed_route"]
    assert route["tag_commit"] == s["tag"] and route["fit_units"] == [FIT_UNIT]
    assert route["checks"] == scoring.CHECKS_TEXT
    assert route["kappa_selection"] == sealed_route.NO_SELECTION
    assert route["held_out_inputs"]["truth_run"] == s["held_run"].name
    assert route["held_out_inputs"]["table_sha256"] == {
        name: digest
        for unit in SCORED
        for name, digest in scoring._held_out_hashes(s["summary"], unit).items()
    }
    assert document["inputs"]["truth_run"] == s["run"].name
    assert set(document["inputs"]["table_sha256"]) == {
        f"{FIT_UNIT}-cars.csv",
        f"{FIT_UNIT}-days.csv",
    }
    assert document["run"]["splits"] == 1 and document["run"]["seeds"] == [1, 1]
    assert document["methods"] == METHODS
    windows = s["summary"]["windows"]
    for unit, calibration in (("OSL_T", 15), ("TRO_R", 3)):
        entry = document["units"][unit]
        assert entry["status"] == "calibratable"
        assert entry["calibration_days_used"] == calibration
        assert entry["split_of_the_gate_run"]["calibration"] == [
            f"{d} {unit}" for d in windows[unit]["calibration_dates"]
        ]
        for variant in ("forecast", "oracle-departure"):
            for method in METHODS:
                tiers = entry["variants"][variant]["methods"][method]
                assert tiers["A"]["splits"] == tiers["B"]["splits"] == 1
            assert entry["variants"][variant]["measured_on"]["days"] == entry["counting_days"]
    with (s["out"] / compare_run.SPLITS_NAME).open() as table:
        seeds = {row["seed"] for row in csv.DictReader(table)}
    assert seeds == {"1"}


@pytest.fixture(scope="module")
def seed_two(repo, builds, tmp_path_factory):
    """A second held-out build recorded at the tag, its calibration days drawn with split seed 2
    (the committed matrix with that seed), its tables in a folder of their own (item 8)."""
    folder = tmp_path_factory.mktemp("seed-two")
    document = json.loads(scoring.MATRIX.read_text())
    document["split_seed"] = 2
    matrix = folder / "matrix-seed-2.json"
    matrix.write_text(json.dumps(document))
    derived = repo / "data" / "derived"
    run = derived / "runs" / "20261116T000002Z-wp14-heldout-truth-seed-2"
    tables = derived / "truth-heldout-seed-2"
    scoring._held_out_build(repo, run, "--matrix", str(matrix), "--out", str(tables))
    scoring._record(run, builds["tag"])
    return run, tables


def test_the_builds_own_seed_is_scored_when_it_is_not_1(repo, setup, seed_two, capsys):
    s = setup
    run, tables = seed_two
    windows = json.loads((run / "truth-summary.json").read_text())["windows"]
    first = s["summary"]["windows"]["OSL_T"]["calibration_dates"]
    assert windows["OSL_T"]["calibration_dates"] != first  # seed 2 draws other days than seed 1
    matrix = _matrix(s["tmp"], SCORED, split_seed=2)
    arguments = _arguments(matrix, s["tables"], s["run"], tables, run, methods=["re-solve"])
    assert compare_run.main(arguments, root=repo) == 0
    document = json.loads((s["out"] / compare_run.RESULTS_NAME).read_text())
    assert document["run"]["splits"] == 1 and document["run"]["seeds"] == [2, 2]
    assert document["sealed_route"]["held_out_inputs"]["truth_run"] == run.name
    for unit in SCORED:
        entry = document["units"][unit]
        assert entry["split_of_the_gate_run"]["seed"] == 2
        assert entry["split_of_the_gate_run"]["calibration"] == [
            f"{d} {unit}" for d in windows[unit]["calibration_dates"]
        ]
    with (s["out"] / compare_run.SPLITS_NAME).open() as table:
        seeds = {row["seed"] for row in csv.DictReader(table)}
    assert seeds == {"2"}


def test_asa_and_llf_score_a_held_out_unit_on_the_route(repo, setup, capsys):
    s = setup
    arguments = _route_arguments(s, ("TRO_R",), methods=["re-solve", "asa", "llf"])
    assert compare_run.main(arguments, root=repo) == 0
    document = json.loads((s["out"] / compare_run.RESULTS_NAME).read_text())
    assert document["label"] == compare_run.ROUTE_LABEL
    entry = document["units"]["TRO_R"]
    for variant in ("forecast", "oracle-departure"):
        found = entry["variants"][variant]
        assert set(found["methods"]) == {"re-solve", "asa", "llf"}
        for method in ("asa", "llf"):
            tiers = found["methods"][method]
            assert tiers["A"]["splits"] == tiers["B"]["splits"] == 1
        assert "controller_day_scale_pairs" in found["evaluations"]
    assert set(entry["gate_reading"]) == {"re-solve", "asa", "llf"}


# --- 3 and 4. the tag and the kind of the units --------------------------------------------------


def test_off_the_tag_nothing_is_read(repo, setup, capsys):
    s = setup
    untagged = sealed_repo(s["tmp"] / "untagged", {}, tag=False)
    with scoring._opens() as seen, watch(_sealed(repo)) as sealed_seen:
        status = compare_run.main(_route_arguments(s), root=untagged)
    shown = capsys.readouterr().out
    assert (status, sealed_seen) == (2, [])
    assert shown.startswith("COULD NOT RUN: REFUSED BY THE GATE: ")
    assert shown.endswith("; nothing read\n") and shown.count("\n") == 1
    assert scoring._tables_read(seen) == []
    assert s["built"] == []  # refused before the model is loaded
    assert list(s["out"].iterdir()) == []


def test_units_of_two_kinds_are_refused(repo, setup, capsys):
    s = setup
    arguments = _route_arguments(s, ("OSL_T", "nrel-from-may-2019"))
    with scoring._opens() as seen:
        status = compare_run.main(arguments, root=repo)
    shown = capsys.readouterr().out
    assert status == 2
    assert shown.startswith("COULD NOT RUN: the units are refused: the matrix's gate units ")
    assert "never both" in shown and shown.endswith("; nothing read\n")
    assert [path for path in seen if str(repo / "data") in path] == []
    assert s["built"] == []


# --- 5. the leak rules ---------------------------------------------------------------------------


def test_a_model_that_reads_the_scored_units_other_days_is_refused(repo, setup, capsys):
    s = setup
    with scoring._opens() as seen:
        status = compare_run.main(_route_arguments(s, model="empirical"), root=repo)
    assert status == 2
    assert capsys.readouterr().out == (
        "COULD NOT RUN: model empirical builds a scored unit's forecasts from that unit's own "
        "other days, test days included; on the sealed route a scored unit's days reach a model "
        "only through its inputs, as in the cold start (its reading C12); nothing read\n"
    )
    assert scoring._tables_read(seen) == []


@pytest.mark.parametrize("splits", ["200", "2", None])  # None: the default, 200
def test_any_split_but_the_builds_one_is_refused(repo, setup, splits, capsys):
    s = setup
    with scoring._opens() as seen:
        status = compare_run.main(_route_arguments(s, splits=splits), root=repo)
    assert status == 2
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: --splits {splits or 200}: on the sealed route each held-out unit is "
        "scored on the one split its held-out build drew (plan section 6, Protocol: one draw per "
        "held-out unit; another seed would calibrate on days that draw scores), so --splits is 1; "
        "nothing read\n"
    )
    assert scoring._tables_read(seen) == []
    assert s["built"] == []


def test_a_matrix_of_another_split_rule_is_refused(repo, setup, capsys):
    s = setup
    rule = s["summary"]["calibration_rule"]
    matrix = _matrix(s["tmp"], SCORED, split_seed=2)
    arguments = _arguments(matrix, s["tables"], s["run"], s["held_tables"], s["held_run"])
    with scoring._opens() as seen:
        status = compare_run.main(arguments, root=repo)
    assert status == 2
    assert capsys.readouterr().out == (
        "COULD NOT RUN: the matrix's split rule (split_seed 2, n_calibration "
        f"{rule['n_calibration']}, calibration_half_below {rule['half_below']}) is not the rule "
        f"the held-out build {s['held_run'].name} drew its calibration days by (split_seed 1, "
        f"n_calibration {rule['n_calibration']}, calibration_half_below {rule['half_below']}); "
        "on the sealed route every scored unit is split by its build's rule\n"
    )
    assert scoring._tables_read(seen) == []
    assert list(s["out"].iterdir()) == []


@pytest.fixture(scope="module")
def short_day(tmp_path_factory):
    """A made-up repository whose OSL_T has a day whose fleet holds 2 cars
    (`test_heldout_calibration_days`), its held-out build and a development build, both recorded
    at its tag's commit."""
    folder = tmp_path_factory.mktemp("short-day")
    root = sealed_repo(folder, short_day_files(folder / "made"))
    tag = git(root, "rev-parse", "HEAD")
    derived = root / "data" / "derived"
    held_run = derived / "runs" / "20261116T000000Z-wp14-heldout-truth"
    scoring._held_out_build(root, held_run)
    scoring._record(held_run, tag)
    (derived / "development").mkdir()
    run, tables = write_build(derived / "development", {FIT_UNIT: gate_cars(40, "T")})
    scoring._record(run, tag)
    held_tables = sealed_route.held_out_tables(root)
    return {
        "root": root,
        "tag": tag,
        "run": run,
        "tables": tables,
        "held_run": held_run,
        "held_tables": held_tables,
    }


def test_a_matrix_whose_readings_draw_other_calibration_days_is_refused(
    short_day, setup, monkeypatch, capsys
):
    s, d = setup, short_day
    monkeypatch.setattr(gate_run, "_head_commit", lambda: d["tag"])
    matrix = _matrix(s["tmp"], ["OSL_T"], small_fleet="drop")
    arguments = _arguments(matrix, d["tables"], d["run"], d["held_tables"], d["held_run"])
    with watch(_sealed(d["root"])) as sealed_seen:
        status = compare_run.main(arguments, root=d["root"])
    shown = capsys.readouterr().out
    assert (status, sealed_seen) == (2, [])
    assert shown.startswith(
        "COULD NOT RUN: OSL_T: the 14 calibration days this run draws are not the 15 the "
        f"held-out build {d['held_run'].name} searched its window on ("
    )
    assert shown.endswith(
        "); the matrix's readings (no_fit cap, small_fleet drop) make other counting days than "
        "the build's, so on the sealed route the unit is not scored\n"
    )
    assert s["built"] != [] and s["built"][0].fitted == []  # refused before the model is fitted
    assert list(s["out"].iterdir()) == []


# --- 6. tables of another build or setting -------------------------------------------------------


def test_the_two_tables_folders_swapped_are_refused(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], SCORED)
    arguments = _arguments(matrix, s["held_tables"], s["held_run"], s["tables"], s["run"])
    with scoring._opens() as seen:
        status = compare_run.main(arguments, root=repo)
    assert status == 2
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: a folder is refused: the tables folder {s['held_tables']} lies in the "
        f"held-out build's folder {s['held_tables']}; nothing read\n"
    )
    assert [path for path in seen if str(repo / "data") in path] == []


def test_the_held_out_arguments_with_development_units_are_refused(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], ["nrel-from-may-2019"])
    arguments = _arguments(matrix, s["tables"], s["run"], None, s["held_run"])
    with scoring._opens() as seen:
        status = compare_run.main(arguments, root=repo)
    assert status == 2
    assert capsys.readouterr().out == (
        "COULD NOT RUN: --held-out-truth-run is for held-out units, and none of the matrix's "
        "gate units ['nrel-from-may-2019'] is a held-out unit; nothing read\n"
    )
    assert [path for path in seen if str(repo / "data") in path] == []


def test_held_out_units_need_the_held_out_truth_run(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], SCORED)
    arguments = _arguments(matrix, s["tables"], s["run"], s["held_tables"], None)
    with scoring._opens() as seen:
        status = compare_run.main(arguments, root=repo)
    assert status == 2
    assert capsys.readouterr().out == "COULD NOT RUN: held-out units need --held-out-truth-run\n"
    assert scoring._tables_read(seen) == []


def test_builds_of_two_settings_are_refused(repo, setup, held_setting, capsys):
    s = setup
    run, tables = held_setting
    matrix = _matrix(s["tmp"], SCORED)
    arguments = _arguments(matrix, s["tables"], s["run"], tables, run)
    with scoring._opens() as seen:
        status = compare_run.main(arguments, root=repo)
    assert status == 2
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: the held-out tables in {tables} are of the held setting but the "
        f"development build {s['run'].name} is of the main setting; a run reads the two builds "
        "of one setting\n"
    )
    assert [path for path in scoring._tables_read(seen) if str(tables) in path] == []


# --- 7. the no-pause sensitivity on the route ----------------------------------------------------


def test_the_no_pause_sensitivity_on_the_route_keeps_the_held_out_label(repo, setup, capsys):
    s = setup
    arguments = _route_arguments(s, ("TRO_R",), methods=["re-solve", "llf"])
    arguments += ["--no-pause-share", "1/3", "--no-pause-seed", "7"]
    assert compare_run.main(arguments, root=repo) == 0
    names = compare_run.output_names(compare_run.no_pause.SHARES["1/3"])
    document = json.loads((s["out"] / names["results"]).read_text())
    expected = (
        f"{compare_run.ROUTE_LABEL}; NO-PAUSE SENSITIVITY of plan section 9: a seeded share "
        "1/3 of the cars (seed 7) charge as logged, without flexibility, and are left unsteered"
    )
    assert document["label"] == document["definitions"]["label"] == expected
    assert document["sealed_route"]["tag_commit"] == s["tag"]
    assert not (s["out"] / compare_run.RESULTS_NAME).exists()


# --- a table changed between its hash check and its parse ----------------------------------------


def test_a_held_out_table_changed_after_its_check_is_refused(repo, setup, capsys):
    """The scored unit's table holds other bytes (`swap_after_read.grown`) from just after its
    first read, the read its SHA-256 is checked on: the run could not run and writes nothing."""
    s = setup
    path = s["held_tables"] / f"{SCORED[0]}-cars.csv"
    with swapped_after_first_read(path, grown(path)) as swaps:
        status = compare_run.main(_route_arguments(s), root=repo)
    assert (status, capsys.readouterr().out) == (2, table_refusal(path))
    assert swaps == [str(path.resolve())]
    assert list(s["out"].iterdir()) == []
