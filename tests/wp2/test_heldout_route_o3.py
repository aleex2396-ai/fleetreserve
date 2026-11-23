"""`o3_run` (outcome O3) on held-out units through the sealed route (`sealed_route.units_route`).

The made-up repository and its made-up tag `frozen-plan` are those of the other route tests
(`fake_heldout.sealed_repo` holding the made-up sealed files of `heldout_truth_fake`), and so are
their helpers (`test_heldout_route_scoring`: the held-out truth build run once at the tag, a run
recorded at the tag's commit, the watch of `open`). The held-out build writes the tables of OSL_T
(30 counting days, 15 of them the calibration days its window was searched on) and TRO_R (6
days) to `data/derived/truth-heldout`; a made-up development build (`gate_fake.write_build`: the
development unit nrel-from-may-2019) lies in `data/derived/development`. Both runs are recorded
at the tag's commit; only the checked-out commit of the package's own top folder is replaced by
the tag's.

1. End to end, at the made-up tag: O3 on OSL_T and TRO_R, one day each, computed by `o3_day`
   (exit 0). The summary's first line is the held-out label (a held-out result, every value a
   hindsight quantity of the sampled days' true sets, nothing fitted, calibrated or chosen from
   a result) with the tag's commit, the held-out build and the development build (checked for its
   setting, none of its tables read); its second the route's `checks` text; then the report as
   off the route, its truth build the held-out one. `o3.json` holds `inputs` (the held-out build's
   check, with that build's own hashes of the two units' tables) and `sealed_route` (the kind, the
   label, the tag's commit, the development build's check with no table, `checks`). The held-out
   tables folder is the build's default one, not named on the command line. No sealed file is
   opened (no new opening logged, nothing under the sealed folder opened, by identity on disk).
   With the gate returning a fix commit as at a logged parsing fix
   (`fake_heldout.gate_claiming_a_fix`), the first line and `sealed_route` name the tag's commit
   and the fix's.
2. The sample on the route (the day's values made up: the route, not the values, is tested): it
   is drawn from every counting day of the held-out table, the build's calibration days and the
   others alike (`o3_run` reading R5): `--days 100` gives OSL_T's 30 counting days, 15 of them the
   build's calibration days; `--days 4 --seed 7` gives the 4 days with the smallest SHA-256 of
   [unit, seed, "site date"] over those 30, the rule written out here.
3. Refused, each with exit 2, one line, no output file and no table opened: away from the tag (a
   made-up repository without it; nothing under either sealed folder opened); held-out units with
   a development or a training unit; the two held-out options with development units; held-out
   units without `--held-out-truth-run`; the two tables folders swapped; the sealed folder named
   as the held-out tables (nothing under it opened).
4. Tables of another build or setting, refused with exit 2 and no output file: the development
   build named as the held-out build (OSL_T is not in it); a held-out table changed in a copy of
   the build's folder (it differs from the table the build wrote); the held setting's held-out
   build beside the main development build (two settings; no held-out table opened).
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from fake_heldout import CLAIMED_FIX, gate_claiming_a_fix, git, sealed_repo
from gate_fake import gate_cars, write_build
from heldout_truth_fake import sealed_files
from sealed_watch import watch
from swap_after_read import grown, swapped_after_first_read, table_refusal
from test_heldout_route_scoring import _held_out_build, _opens, _record, _tables_read

from fleetreserve import gate_run, heldout_loader, o3_run, sealed_route
from fleetreserve.o3_day import BOUND, KEYS, PRODUCTS, DayResult

UNITS = ("OSL_T", "TRO_R")
DEVELOPMENT_UNIT = "nrel-from-may-2019"
# What a run on the route prints first: the label of a held-out O3 result.
LABEL = (
    "HELD-OUT RESULT, through the sealed-data gate at the tag frozen-plan, or at a logged parsing "
    "fix of it, with a clean tree: outcome O3 on held-out units, every value a hindsight quantity "
    "of the sampled days' true sets; nothing is fitted, calibrated or chosen from a result"
)
CHECKS_TEXT = (
    "The route's checks stop ordinary mistakes by an honest operator; they are not a defence "
    "against deliberate evasion (links or copies made on purpose, a second working copy). The "
    "proof that the seal held is the plan's own: the committed hashes, the held-out loader, "
    "which opens a sealed file only at the tag frozen-plan, or at a logged parsing fix of it, "
    "with a clean tree, and its openings log."
)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    folder = tmp_path_factory.mktemp("o3-route")
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
    run, tables = write_build(derived / "development", {DEVELOPMENT_UNIT: gate_cars(12, "T")})
    _record(run, tag)
    return {
        "tag": tag,
        "run": run,
        "tables": tables,
        "held_run": held_run,
        "held_tables": sealed_route.held_out_tables(repo),
        "summary": json.loads((held_run / "truth-summary.json").read_text()),
    }


@pytest.fixture
def setup(builds, tmp_path, monkeypatch):
    """The checked-out commit is the tag's; a run folder."""
    monkeypatch.setattr(gate_run, "_head_commit", lambda: builds["tag"])
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return {**builds, "out": out, "tmp": tmp_path}


def _arguments(s, units=UNITS, days="1", seed="7", tables=None, run=None, held=(), extra=()):
    return [
        "--units",
        *units,
        "--days",
        days,
        "--seed",
        seed,
        "--base-set",
        "day-average",
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


def test_o3_scores_held_out_units_through_the_route(repo, setup, capsys):
    s = setup
    openings = heldout_loader.logged_openings(repo)
    arguments = _arguments(s, held=_held(run=s["held_run"]))
    with watch(repo / "data" / "heldout") as seen:
        status = o3_run.main(arguments, root=repo)
    shown = capsys.readouterr().out.splitlines()
    assert (status, seen) == (0, [])
    assert shown[0] == (
        f"{LABEL} (tag commit {s['tag']}). The units' tables from the held-out build "
        f"{s['held_run'].name}; the development build {s['run'].name} checked for its setting "
        "only, none of its tables read."
    )
    assert shown[1] == CHECKS_TEXT
    assert shown[2].startswith("O3, HEADLINE MEASUREMENT ON TRUE SETS (plan section 11)")
    truth = f"truth build {s['held_run'].name} (main setting); units OSL_T, TRO_R; 1 days per unit"
    assert shown[3].startswith(truth)
    assert shown[-1] == "all checks held"
    document = json.loads((s["out"] / o3_run.RESULTS_NAME).read_text())
    assert document["inputs"] == {
        "setting": "main",
        "truth_run": s["held_run"].name,
        "table_sha256": _build_hashes(s["summary"], UNITS),
    }
    assert document["sealed_route"] == {
        "kind": "held-out",
        "label": LABEL,
        "tag_commit": s["tag"],
        "development_inputs": {"setting": "main", "truth_run": s["run"].name, "table_sha256": {}},
        "checks": CHECKS_TEXT,
    }
    assert [document["sample"][unit]["sampled_days"] for unit in UNITS] == [1, 1]
    assert document["failures"] == []
    assert heldout_loader.logged_openings(repo) == openings


def test_at_a_parsing_fix_the_record_and_the_first_line_name_the_fix_commit(
    repo, setup, monkeypatch, capsys
):
    s = setup
    gate_claiming_a_fix(monkeypatch)
    assert o3_run.main(_arguments(s, held=_held(run=s["held_run"])), root=repo) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[0].startswith(
        f"{LABEL} (tag commit {s['tag']}; at the logged parsing fix {CLAIMED_FIX}). The units' "
    )
    route = json.loads((s["out"] / o3_run.RESULTS_NAME).read_text())["sealed_route"]
    assert (route["tag_commit"], route["fix_commit"]) == (s["tag"], CLAIMED_FIX)


# --- 2. the sample on the route ------------------------------------------------------------------


def _made_up_day(task) -> DayResult:
    """Every method 1 kWh and the ceiling 2 kWh, with no solve (the route is tested here)."""
    values = {p: {k: 2.0 if k == "ceiling" else 1.0 for k in KEYS} for p in PRODUCTS}
    outcomes = {p: {k: "closed" if k == BOUND else "optimal" for k in KEYS} for p in PRODUCTS}
    return DayResult(
        task.unit, task.site, task.date, len(task.day.cars), task.day.n_service, None, values,
        outcomes, 1.0, 1.0,
    )  # fmt: skip


def _smallest_ranks(unit, seed, texts, count):
    """The sample rule written out: the `count` texts with the smallest SHA-256 of
    [unit, seed, text] as JSON, in text order."""
    ranked = sorted(
        texts, key=lambda t: hashlib.sha256(json.dumps([unit, seed, t]).encode()).digest()
    )
    return sorted(ranked[:count])


def test_the_sample_is_drawn_from_every_counting_day_of_the_held_out_table(
    repo, setup, monkeypatch
):
    s = setup
    monkeypatch.setattr(o3_run, "compute_day", _made_up_day)
    held = _held(run=s["held_run"])
    assert o3_run.main(_arguments(s, units=("OSL_T",), days="100", held=held), root=repo) == 0
    sample = json.loads((s["out"] / o3_run.RESULTS_NAME).read_text())["sample"]["OSL_T"]
    calibration = s["summary"]["windows"]["OSL_T"]["calibration_dates"]
    assert len(calibration) == 15
    every = sample["days"]
    assert (sample["counting_days"], sample["sampled_days"], len(every)) == (30, 30, 30)
    assert sample["fewer_than_asked"]
    assert {f"OSL_T {date}" for date in calibration} < set(every)
    assert o3_run.main(_arguments(s, units=("OSL_T",), days="4", held=held), root=repo) == 0
    drawn = json.loads((s["out"] / o3_run.RESULTS_NAME).read_text())["sample"]["OSL_T"]["days"]
    assert drawn == _smallest_ranks("OSL_T", 7, every, 4)


# --- 3. refusals ---------------------------------------------------------------------------------


def _refused(repo, s, arguments, capsys, root=None):
    """(the one line printed, the tables opened) of a run that must be refused (exit 2)."""
    with _opens() as seen:
        status = o3_run.main(arguments, root=repo if root is None else root)
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
    assert (tables, sealed) == ([], [])


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


# --- 4. tables of another build or setting -------------------------------------------------------


def test_the_development_build_named_as_the_held_out_build_is_refused(repo, setup, capsys):
    s = setup
    arguments = _arguments(s, held=_held(s["tables"], s["run"]))
    shown, _ = _refused(repo, s, arguments, capsys)
    assert shown == f"COULD NOT RUN: OSL_T is not in the recorded truth build {s['run'].name}\n"


def test_a_held_out_table_that_differs_from_its_build_is_refused(repo, setup, capsys):
    s = setup
    copy = s["tmp"] / "copied-tables"
    shutil.copytree(s["held_tables"], copy)
    changed = copy / "OSL_T-cars.csv"
    changed.write_bytes(changed.read_bytes() + b"\n")
    shown, _ = _refused(repo, s, _arguments(s, held=_held(copy, s["held_run"])), capsys)
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


# --- 5. a table changed between its hash check and its parse -------------------------------------


def test_a_held_out_table_changed_after_its_check_is_refused(repo, setup, monkeypatch, capsys):
    """The table holds other bytes (`swap_after_read.grown`) from just after its first read, the
    read its SHA-256 is checked on: the run could not run."""
    s = setup
    monkeypatch.setattr(o3_run, "compute_day", _made_up_day)
    path = s["held_tables"] / "OSL_T-days.csv"
    with swapped_after_first_read(path, grown(path)) as swaps:
        shown, _ = _refused(repo, s, _arguments(s, held=_held(run=s["held_run"])), capsys)
    assert (swaps, shown) == ([str(path.resolve())], table_refusal(path))
