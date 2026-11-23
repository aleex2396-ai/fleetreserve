"""`ceiling_report` (outcome O2) on held-out units through the sealed route
(`sealed_route.units_route`), with its every-day option.

The made-up repository and its made-up tag `frozen-plan` are those of the other route tests
(`fake_heldout.sealed_repo` holding the made-up sealed files of `heldout_truth_fake`), and so are
their helpers (`test_heldout_route_scoring`: the held-out truth build run once at the tag, a run
recorded at the tag's commit, the watch of `open`). The held-out build writes the tables of OSL_T
(30 counting days, 15 of them the calibration days its window was searched on) and TRO_R (6
days) to `data/derived/truth-heldout`; a made-up development build (`gate_fake.write_build`: the
development unit nrel-from-may-2019) lies in `data/derived/development`. Both runs are recorded
at the tag's commit; only the checked-out commit of the package's own top folder is replaced by
the tag's.

1. End to end, at the made-up tag: O2 on OSL_T and TRO_R with `--every-day`, every value
   computed by `ceiling_values`. The summary's first line is the held-out label with the tag's
   commit, the held-out build and the development build (checked for its setting, none of its
   tables read); its second the route's `checks` text; then the report as off the route, its
   truth build the held-out one. `ceiling-report.json` holds `inputs` (the held-out build's
   check, with that build's own hashes of the two units' tables) and `sealed_route` (the kind,
   the label, the tag's commit, the development build's check with no table, `checks`). No
   sealed file is opened (no new opening logged, nothing under the sealed folder opened, by
   identity on disk). At a logged parsing fix the first line and `sealed_route` name the fix.
2. The days on the route (the day's values made up: the route, not the values, is tested): M is
   every counting day of the held-out table with at most 8 service hours, the build's
   calibration days and the others alike (`ceiling_report` reading O2-1); the days of M do not
   change when every value and every departure flag of the days changes (no day's outcome
   chooses a day).
3. Refused, each with exit 2, one line, no output file and no table opened: away from the tag (a
   made-up repository without it; nothing under either sealed folder opened); held-out units with
   a development or a training unit; the two held-out options with development units; held-out
   units without `--held-out-truth-run`; the two tables folders swapped; the sealed folder named
   as the held-out tables (nothing under it opened); held-out units without `--every-day` (the
   block sample is a development rule); a held-out unit named twice.
4. Tables of another build or setting, refused with exit 2 and no output file: the development
   build named as the held-out build (OSL_T is not in it); a held-out table changed in a copy of
   the build's folder; the held setting's held-out build beside the main development build.
5. The exit status on the route (reading O2-4, the owner's decisions of 19 and 20 November
   2026): O2's pass failed on a unit gives exit 0, its verdict and the violation in the results;
   O2's pass could not run gives exit 0 (as in 1 and 2); the command's own faults on a day give
   exit 0 too, each written in the results and counted in its unit's `faults`, and in its
   verdict only when it is on an M day (reading O2-6, decision 85 (1) of 20 November 2026): an M
   day that could not be built (could not run), a block order broken alone (failed), a block
   order broken beside a failed O2 (failed), an hourly order broken on every M day (passed,
   "CHECKS FAILED" shown). A day not parsed from the held-out tables read could not run (exit
   2, one line, no output file).
"""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from fake_heldout import CLAIMED_FIX, gate_claiming_a_fix, git, sealed_repo
from gate_fake import gate_cars, write_build
from heldout_truth_fake import sealed_files
from sealed_watch import watch
from swap_after_read import file_refusal, grown, swapped_after_first_read, table_refusal
from test_heldout_route_scoring import _held_out_build, _opens, _record, _tables_read

from fleetreserve import ceiling_report, gate_run, heldout_loader, sealed_route
from fleetreserve.ceiling_values import VALUE_KEYS, DayResult

UNITS = ("OSL_T", "TRO_R")
DEVELOPMENT_UNIT = "nrel-from-may-2019"
LABEL = (
    "HELD-OUT RESULT, through the sealed-data gate at the tag frozen-plan, or at a logged parsing "
    "fix of it, with a clean tree: outcome O2 on held-out units, on every counting day with at "
    "most 8 service hours, every value a hindsight or best real-time value of the day's true "
    "sets; nothing is fitted, calibrated, sampled or chosen from a result"
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
    folder = tmp_path_factory.mktemp("ceiling-route")
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


def _arguments(s, units=UNITS, tables=None, run=None, held=(), every_day=True, extra=()):
    return [
        "--units",
        *units,
        "--tables",
        str(s["tables"] if tables is None else tables),
        "--truth-run",
        str(s["run"] if run is None else run),
        "--workers",
        "1",
        *(["--every-day"] if every_day else []),
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


def test_o2_scores_held_out_units_through_the_route(repo, setup, capsys):
    s = setup
    openings = heldout_loader.logged_openings(repo)
    arguments = _arguments(s, held=_held(run=s["held_run"]))
    with watch(repo / "data" / "heldout") as seen:
        status = ceiling_report.main(arguments, root=repo)
    shown = capsys.readouterr().out.splitlines()
    assert seen == []
    assert heldout_loader.logged_openings(repo) == openings
    scored = {"status": status, "out": s["out"]}
    assert shown[0] == (
        f"{LABEL} (tag commit {s['tag']}). The units' tables from the held-out build "
        f"{s['held_run'].name}; the development build {s['run'].name} checked for its setting "
        "only, none of its tables read."
    )
    assert shown[1] == CHECKS_TEXT
    assert shown[2].startswith(f"truth build {s['held_run'].name}; rho 0.0. True one-way sets.")
    assert "(a) O2 ON EVERY COUNTING DAY WITH AT MOST 8 SERVICE HOURS" in "\n".join(shown)
    document = json.loads((scored["out"] / ceiling_report.RESULTS_NAME).read_text())
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
    units = document["every_day"]["units"]
    assert [units[unit]["counting_days"] for unit in UNITS] == [30, 6]
    assert document["failures"] == []
    assert [units[unit]["days_in_m"] for unit in UNITS] == [30, 6]
    # no made-up held-out day meets T4's hypotheses: each unit has no day to check, so with no
    # unit judged O2's pass over the units is no day to check (never a pass on nothing checked,
    # never could not run; the owner's decision of 19 November 2026), in the results alone: exit
    # 0 (reading O2-4)
    assert [units[unit]["t4_check"]["days_meeting_hypotheses"] for unit in UNITS] == [0, 0]
    assert [units[unit]["o2_pass"] for unit in UNITS] == ["no day to check"] * 2
    assert document["every_day"]["pass_over_units"] == "no day to check"
    assert document["every_day"]["no_day_to_check"] == list(UNITS)
    assert scored["status"] == 0
    assert shown[-1] == (
        "O2's pass over the units: no day to check (no day to check on OSL_T, TRO_R)"
    )


def test_at_a_parsing_fix_the_record_and_the_first_line_name_the_fix_commit(
    repo, setup, monkeypatch, capsys
):
    s = setup
    gate_claiming_a_fix(monkeypatch)
    monkeypatch.setattr(ceiling_report, "compute_day", _made_up_day)
    assert ceiling_report.main(_arguments(s, held=_held(run=s["held_run"])), root=repo) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[0].startswith(
        f"{LABEL} (tag commit {s['tag']}; at the logged parsing fix {CLAIMED_FIX}). The units' "
    )
    route = json.loads((s["out"] / ceiling_report.RESULTS_NAME).read_text())["sealed_route"]
    assert (route["tag_commit"], route["fix_commit"]) == (s["tag"], CLAIMED_FIX)


# --- 2. the days on the route --------------------------------------------------------------------


def _made_up_day(task, value: float = 1.0, departs: bool = False) -> DayResult:
    """Every value `value` kWh with no solve, every day meeting T4's hypotheses unless `departs`
    (the route is tested here)."""
    values = dict.fromkeys(VALUE_KEYS, value if task.sampled else None)
    values["hourly_hind"] = values["hourly_causal"] = value
    outcomes = {k: "optimal" if values[k] is not None else "not solved" for k in VALUE_KEYS}
    return DayResult(
        task.unit, task.site, task.date, len(task.day.cars), task.day.n_service,
        task.day.n_recovery, departs, not departs, task.sampled, None, values, outcomes,
    )  # fmt: skip


def _m_days(out: Path) -> dict:
    document = json.loads((out / ceiling_report.RESULTS_NAME).read_text())
    rows = (out / ceiling_report.EVERY_DAY_NAME).read_text().splitlines()[1:]
    return document["every_day"], sorted(tuple(row.split(",")[:3]) for row in rows)


def test_m_is_every_counting_day_of_the_held_out_table_whatever_its_values(
    repo, setup, monkeypatch
):
    s = setup
    held = _held(run=s["held_run"])
    monkeypatch.setattr(ceiling_report, "compute_day", _made_up_day)
    assert ceiling_report.main(_arguments(s, units=("OSL_T",), held=held), root=repo) == 0
    first, days = _m_days(s["out"])
    calibration = s["summary"]["windows"]["OSL_T"]["calibration_dates"]
    assert len(calibration) == 15
    hours = {int(row) for row in _column(s["out"], "service_hours")}
    assert max(hours) <= 8  # every made-up held-out day has at most 8 service hours
    assert first["units"]["OSL_T"]["counting_days"] == 30
    assert first["units"]["OSL_T"]["days_in_m"] == len(days) == 30
    assert {date for date in calibration} < {date for _, _, date in days}

    def other(task):
        return _made_up_day(replace(task), value=7.5, departs=True)

    monkeypatch.setattr(ceiling_report, "compute_day", other)
    assert ceiling_report.main(_arguments(s, units=("OSL_T",), held=held), root=repo) == 0
    second, again = _m_days(s["out"])
    assert again == days
    assert second["units"]["OSL_T"]["days_in_m"] == 30
    assert second["units"]["OSL_T"]["o2_pass"] == "no day to check"  # no day meets them now


def _column(out: Path, name: str) -> list[str]:
    lines = (out / ceiling_report.EVERY_DAY_NAME).read_text().splitlines()
    at = lines[0].split(",").index(name)
    return [line.split(",")[at] for line in lines[1:]]


# --- 3. refusals ---------------------------------------------------------------------------------


def _refused(repo, s, arguments, capsys, root=None):
    """(the one line printed, the tables opened) of a run that must be refused (exit 2)."""
    with _opens() as seen:
        status = ceiling_report.main(arguments, root=repo if root is None else root)
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


def test_held_out_units_without_every_day_are_refused(repo, setup, capsys):
    s = setup
    arguments = _arguments(s, held=_held(run=s["held_run"]), every_day=False)
    shown, tables = _refused(repo, s, arguments, capsys)
    assert shown == f"COULD NOT RUN: {ceiling_report.ROUTE_NEEDS_EVERY_DAY}\n"
    assert tables == []


def test_a_held_out_unit_named_twice_is_refused(repo, setup, capsys):
    s = setup
    arguments = _arguments(s, units=("OSL_T", "OSL_T"), held=_held(run=s["held_run"]))
    shown, tables = _refused(repo, s, arguments, capsys)
    assert shown == (
        "COULD NOT RUN: the units are refused: a unit is named twice: ['OSL_T', 'OSL_T']\n"
    )
    assert tables == []


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


# --- 5. the exit status (reading O2-4) -----------------------------------------------------------


def _day_with(**values):
    """`_made_up_day` with the block values of the M days in `values` (every M day meets T4's
    hypotheses), or with `error` set: an error makes the day one that could not be built."""
    error = values.pop("error", None)

    def day(task):
        found = _made_up_day(task)
        if task.sampled:
            found.values.update(values)
        return replace(found, error=error)

    return day


def _scored(repo, s, monkeypatch, capsys, day) -> tuple[int, dict, list[str]]:
    monkeypatch.setattr(ceiling_report, "compute_day", day)
    status = ceiling_report.main(_arguments(s, held=_held(run=s["held_run"])), root=repo)
    shown = capsys.readouterr().out.splitlines()
    return status, json.loads((s["out"] / ceiling_report.RESULTS_NAME).read_text()), shown


def test_o2s_pass_failed_on_a_unit_exits_0_with_the_verdict_in_the_results(
    repo, setup, monkeypatch, capsys
):
    """V_caus 0.5 below V_hind 1 on every M day meeting T4's hypotheses (V_aff and V_diag 0.5, so
    every order holds): O2 failed on both units, in the results; exit 0. Exit 1 before decision
    74."""
    s = setup
    day = _day_with(block_caus=0.5, block_aff=0.5, block_diag=0.5)
    status, document, shown = _scored(repo, s, monkeypatch, capsys, day)
    every = document["every_day"]
    assert [every["units"][unit]["o2_pass"] for unit in UNITS] == ["failed"] * 2
    assert every["pass_over_units"] == "failed"
    violations = every["units"]["OSL_T"]["t4_check"]["violations"]
    assert len(violations) == every["units"]["OSL_T"]["days_in_m"] == 30
    assert violations[0]["block_hind"] == 1.0 and violations[0]["block_caus"] == 0.5
    assert len(document["failures"]) == 36  # the T4 lines, written as before
    assert all("differs from V_caus" in line for line in document["failures"])
    assert shown[-1] == "O2's pass failed on OSL_T, TRO_R (reading O2-4: in the results alone)"
    assert status == 0


def test_a_day_that_could_not_be_built_exits_0_in_the_results_and_the_verdict(
    repo, setup, monkeypatch, capsys
):
    """Every M day could not be built (each day otherwise one where V_hind = V_caus on T4's
    hypotheses): exit 0, the 36 days in `failures`, each unit could not run with 30 and 6 days
    not built in its `faults`. Exit 1 and both units passed at b1e8061."""
    s = setup
    status, document, shown = _scored(
        repo, s, monkeypatch, capsys, _day_with(error="made-up failure")
    )
    every = document["every_day"]
    assert len(document["failures"]) == 36
    assert document["failures"][0].endswith("the day could not be built: made-up failure")
    assert [every["units"][unit]["faults"] for unit in UNITS] == [
        {"order_breaks": 0, "days_not_built": 30, "block_order_breaks_on_m_days": 0},
        {"order_breaks": 0, "days_not_built": 6, "block_order_breaks_on_m_days": 0},
    ]
    assert [every["units"][unit]["o2_pass"] for unit in UNITS] == ["could not run"] * 2
    assert every["pass_over_units"] == "could not run"
    assert shown[-1] == "CHECKS FAILED"
    assert status == 0


def test_a_broken_order_alone_exits_0_with_o2_failed_and_the_break_in_the_results(
    repo, setup, monkeypatch, capsys
):
    """V_aff 1.5 above V_caus 1 on every M day, V_hind = V_caus there (the T4 equality holds):
    exit 0, the 36 breaks in `failures`, each unit failed by its breaks alone (reading O2-6).
    Exit 1 and both units passed at b1e8061."""
    s = setup
    status, document, shown = _scored(repo, s, monkeypatch, capsys, _day_with(block_aff=1.5))
    every = document["every_day"]
    assert [every["units"][unit]["t4_check"]["violations"] for unit in UNITS] == [[], []]
    assert len(document["failures"]) == 36
    assert all("block_aff 1.5 above block_caus 1" in line for line in document["failures"])
    assert [every["units"][unit]["faults"] for unit in UNITS] == [
        {"order_breaks": 30, "days_not_built": 0, "block_order_breaks_on_m_days": 30},
        {"order_breaks": 6, "days_not_built": 0, "block_order_breaks_on_m_days": 6},
    ]
    assert [every["units"][unit]["o2_pass"] for unit in UNITS] == ["failed"] * 2
    assert every["pass_over_units"] == "failed"
    text = (s["out"] / ceiling_report.TEXT_NAME).read_text()
    assert (
        "own faults on the unit's days (reading O2-6): 30 broken orders, 0 days not built" in text
    )
    assert shown[-1] == "CHECKS FAILED"
    assert status == 0


def test_a_broken_order_exits_0_with_o2_failed_beside_it(repo, setup, monkeypatch, capsys):
    """V_aff 0.9 above V_caus 0.5: an order broken on every M day, and O2 failed too."""
    s = setup
    day = _day_with(block_caus=0.5, block_aff=0.9, block_diag=0.5)
    status, document, shown = _scored(repo, s, monkeypatch, capsys, day)
    assert document["every_day"]["pass_over_units"] == "failed"
    assert sum("block_aff 0.9 above block_caus 0.5" in line for line in document["failures"]) == 36
    assert shown[-1] == "CHECKS FAILED"
    assert status == 0


def test_an_hourly_break_on_every_m_day_exits_0_with_o2_passed_and_the_breaks_counted(
    repo, setup, monkeypatch, capsys
):
    """The best causal hourly value 2 above the hourly ceiling 1 (and the hourly tree 1) on every
    M day, the block values as made up (V_hind = V_caus): exit 0, the 72 breaks in `failures`
    and each unit's `faults`, O2 passed on both units (an hourly break is not in O2's pass,
    decision 85 (1)), "CHECKS FAILED" the last line. Both units failed at c22e5c1."""
    s = setup
    status, document, shown = _scored(repo, s, monkeypatch, capsys, _day_with(hourly_causal=2.0))
    every = document["every_day"]
    assert [every["units"][unit]["o2_pass"] for unit in UNITS] == ["passed"] * 2
    assert every["pass_over_units"] == "passed"
    assert len(document["failures"]) == 72
    assert all(": hourly " in line for line in document["failures"])
    assert [every["units"][unit]["faults"] for unit in UNITS] == [
        {"order_breaks": 60, "days_not_built": 0, "block_order_breaks_on_m_days": 0},
        {"order_breaks": 12, "days_not_built": 0, "block_order_breaks_on_m_days": 0},
    ]
    assert shown[-1] == "CHECKS FAILED"
    assert status == 0


def test_a_day_not_parsed_from_the_held_out_tables_could_not_run(repo, setup, monkeypatch, capsys):
    """The parse of a held-out unit's tables read raises (`parse_unit_days`): no day is judged,
    so the command could not run (exit 2), one line, no output file. Exit 1 at b1e8061."""
    s = setup

    def no_parse(*args, **kwargs):
        raise ValueError("made-up layout")

    monkeypatch.setattr(ceiling_report, "parse_unit_days", no_parse)
    status = ceiling_report.main(_arguments(s, held=_held(run=s["held_run"])), root=repo)
    shown = capsys.readouterr().out
    assert shown == "COULD NOT RUN: a day could not be built from the tables: made-up layout\n"
    assert _no_outputs(s["out"])
    assert status == 2


def test_the_reading_o2_4_states_the_owners_decision():
    text = " ".join(ceiling_report.__doc__.split())
    assert (
        "O2-4. Exit status with `--every-day` (below), by the owner's decisions of 19 and 20 "
        "November 2026 (\"O2's held-out command always exits 0 when it ran, its verdict only in "
        'its results file, so a scientific fail does not stop the sealed run"'
    ) in text
    assert (
        "On held-out units the exit status is 0 whenever the command ran, a broken order or a day "
        "that could not be built included"
    ) in text
    assert "O2's pass, whatever it is, is in the results alone" in text


# --- 6. a file changed between its hash check and its parse --------------------------------------


@pytest.mark.parametrize("name", ["OSL_T-days.csv", "OSL_T-cars.csv"])
def test_a_held_out_table_changed_after_its_check_is_refused(
    repo, setup, monkeypatch, capsys, name
):
    """The table holds other bytes (`swap_after_read.grown`) from just after its first read, the
    read its SHA-256 is checked on: the run could not run, never scoring a table other than the
    one checked."""
    s = setup
    monkeypatch.setattr(ceiling_report, "compute_day", _made_up_day)
    path = s["held_tables"] / name
    with swapped_after_first_read(path, grown(path)) as swaps:
        shown, _ = _refused(repo, s, _arguments(s, held=_held(run=s["held_run"])), capsys)
    assert (swaps, shown) == ([str(path.resolve())], table_refusal(path))


def test_the_held_out_build_summary_changed_after_its_check_is_refused(
    repo, setup, monkeypatch, capsys
):
    """The summary, which holds the tables' SHA-256, holds other bytes from just after its first
    read, the read its manifest's hash is checked on: the run could not run."""
    s = setup
    monkeypatch.setattr(ceiling_report, "compute_day", _made_up_day)
    path = s["held_run"] / "truth-summary.json"
    with swapped_after_first_read(path, grown(path)) as swaps:
        shown, _ = _refused(repo, s, _arguments(s, held=_held(run=s["held_run"])), capsys)
    assert (swaps, shown) == ([str(path.resolve())], file_refusal(path))
