"""`--units` of the four report commands (`unit_list`): `ceiling_report`, `dev_day_checks`,
`flexitroid_days` and `forecast_report`.

1. Each command reads the units named, and only them, on a made-up build of two units outside
   every default list (`LAYOUT`, made-up days of `ceiling_fakes`): the development unit `OSL_1`
   with 25 days at one site and the training unit `caltech` with 15 days at each of two sites, so
   25 and 30 counting days (every made-up day lists 3 cars). Named as `caltech OSL_1`, not in
   sorted order, so a report's lines per unit follow the order given (reading U1).
   By hand, the 50-day sample of `ceiling_report` and `flexitroid_days` (`block_sample.allocate`):
   quotas 50 x 30/55 = 27.27 and 50 x 25/55 = 22.73, rounded down 27 + 22 = 49, the last day to
   the larger remainder (0.73, OSL_1): caltech 27, OSL_1 23. At the scale 0.2 of
   `dev_day_fakes`, 3 + 3 = 6 and 5 counting days; its 5-day sample: quotas 2.73 and 2.27,
   2 + 2 = 4, the last to caltech (remainder 0.73): caltech 3, OSL_1 2.
2. A name the truth build does not write (each held-out unit's name, and a made-up one) could not
   run, exit 2, before any run folder or table is read: the run folder and tables folder given do
   not exist, so a later check would print another line. `ceiling_report`, which has the sealed
   route, refuses a held-out unit named beside a development unit with the route's refusal of a
   mix (`sealed_route.units_route`), in the same way.
3. A training or development unit the recorded truth build and its tables do not hold could not
   run, exit 2, with one plain line, and nothing is written.
4. The known units are the truth build's training and development units, none held-out;
   `unit_list.check_units` refuses a unit named twice or no unit.
5. A list naming the pooled unit `norway-development-pooled` beside one of the five development
   sites it pools could not run, exit 2, before any run folder or table is read (reading U2); a
   list naming it beside other units only, each command's default list among them, is read.
"""

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest
from ceiling_fakes import setup_build

from fleetreserve import (
    ceiling_report,
    dev_day_checks,
    flexitroid_days,
    forecast_report,
    heldout_truth_build,
    load_norway,
    truth_build,
    unit_list,
)
from fleetreserve import flexitroid_match as fm
from fleetreserve.truth_verify import CouldNotRun

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wp45"))
dev_day_fakes = importlib.import_module("dev_day_fakes")
CONFIG, SCALE = dev_day_fakes.CONFIG, dev_day_fakes.SCALE

LAYOUT = {"OSL_1": (("A", 25),), "caltech": (("A", 15), ("B", 15))}
NAMED = ["caltech", "OSL_1"]
TABLES = {f"{unit}-{kind}.csv" for unit in NAMED for kind in ("cars", "days")}


def _table(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _out(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return out


def _argv(command, run, tables) -> list[str]:
    """The command's arguments other than `--units` (workers 1)."""
    found = ["--run", str(run), "--tables", str(tables)]
    if command is dev_day_checks:
        return [*found, "--t2-sample-size", "5", *CONFIG, "--workers", "1"]
    if command is forecast_report:
        return found
    return [*found, "--workers", "1"]


@pytest.fixture
def named_build(tmp_path, monkeypatch):
    """The made-up build of `LAYOUT` at full scale."""
    monkeypatch.setattr(forecast_report, "_head_commit", lambda: "a" * 40)
    return setup_build(tmp_path, monkeypatch, layout=LAYOUT)


# --- 1. the units named are read ------------------------------------------------------------------


def test_ceiling_report_reads_the_units_named(named_build, tmp_path, monkeypatch, capsys):
    run, tables = named_build
    out = _out(tmp_path, monkeypatch)
    argv = [*_argv(ceiling_report, run, tables), "--units", *NAMED]
    assert ceiling_report.main(argv) == 0
    document = json.loads((out / ceiling_report.RESULTS_NAME).read_text())
    assert document["sample"]["counting_days"] == {"caltech": 30, "OSL_1": 25}
    assert document["sample"]["sampled_days"] == {"caltech": 27, "OSL_1": 23}
    assert set(document["hourly"]) == set(NAMED)
    assert set(document["inputs"]["table_sha256"]) == TABLES
    days = _table(out / ceiling_report.DAYS_NAME)
    assert len(days) == 55 and {r["unit"] for r in days} == set(NAMED)
    sample = _table(out / ceiling_report.SAMPLE_NAME)
    assert len(sample) == 50
    assert sum(r["unit"] == "OSL_1" for r in sample) == 23
    lines = capsys.readouterr().out.splitlines()
    first = lines.index("  caltech: 30 counting days, 27 sampled")
    assert lines[first + 1] == "  OSL_1: 25 counting days, 23 sampled"
    assert lines.index("  UNIT caltech") < lines.index("  UNIT OSL_1")


def test_dev_day_checks_reads_the_units_named(tmp_path, monkeypatch, capsys):
    run, tables = setup_build(tmp_path, monkeypatch, scale=SCALE, layout=LAYOUT)
    out = _out(tmp_path, monkeypatch)
    code = dev_day_checks.main([*_argv(dev_day_checks, run, tables), "--units", *NAMED])
    assert code == 0
    document = json.loads((out / dev_day_checks.RESULTS_NAME).read_text())
    assert document["counting_days"] == 11
    assert document["sample"]["counting_days"] == {"caltech": 6, "OSL_1": 5}
    assert document["sample"]["sampled_days"] == {"caltech": 3, "OSL_1": 2}
    assert set(document["inputs"]["table_sha256"]) == TABLES
    days = _table(out / dev_day_checks.DAYS_NAME)
    assert len(days) == 11 and {r["unit"] for r in days} == set(NAMED)
    assert {r["unit"] for r in _table(out / dev_day_checks.SAMPLE_NAME)} == set(NAMED)
    assert "T2 sample: 5 days, seed 2026, label t2-check; OSL_1 2, caltech 3" in (
        capsys.readouterr().out
    )


def test_flexitroid_days_reads_the_units_named_and_says_it_is_not_the_block_sample(
    named_build, tmp_path, monkeypatch
):
    run, tables = named_build
    out = _out(tmp_path, monkeypatch)
    code = flexitroid_days.main([*_argv(flexitroid_days, run, tables), "--units", *NAMED])
    document = json.loads((out / flexitroid_days.RESULTS_NAME).read_text())
    plan = document["sample"]
    assert plan["counting_days"] == {"caltech": 30, "OSL_1": 25}
    assert plan["sampled_days"] == {"caltech": 27, "OSL_1": 23}
    assert set(document["inputs"]["table_sha256"]) == TABLES
    assert len(document["days"]) == 50
    assert {d["unit"] for d in document["days"]} == set(NAMED)
    assert plan["reading"].startswith(
        "the draw of the block sample of plan 13 WP2 on the units named by --units, not that "
        "sample: "
    )
    assert plan["reading"].endswith(flexitroid_days.READING)
    # Every fifth made-up day has an empty fleet, which flexitroid cannot compare (as on the
    # default units): such a day could not run, so the run ends 2.
    empty = [d for d in document["days"] if d["cars"] == 0]
    assert empty and all(d["verdict"] == fm.COULD_NOT_RUN for d in empty)
    assert code == 2


def test_forecast_report_reads_the_units_named(named_build, tmp_path, monkeypatch, capsys):
    run, tables = named_build
    out = _out(tmp_path, monkeypatch)
    code = forecast_report.main([*_argv(forecast_report, run, tables), "--units", *NAMED])
    assert code == 0
    document = json.loads((out / forecast_report.RESULTS_NAME).read_text())
    assert set(document["units"]) == set(NAMED)
    assert document["units"]["caltech"]["counting_days"] == 30
    assert document["units"]["OSL_1"]["counting_days"] == 25
    assert set(document["inputs"]["table_sha256"]) == TABLES
    lines = capsys.readouterr().out.splitlines()
    assert [line for line in lines if line.startswith("UNIT ")] == ["UNIT caltech", "UNIT OSL_1"]


# --- 2. a name the truth build does not write ---------------------------------------------------

COMMANDS = [ceiling_report, dev_day_checks, flexitroid_days, forecast_report]
NOT_WRITTEN = [*heldout_truth_build.UNITS, "no-such-unit"]
ROUTED = (ceiling_report,)  # the report commands with the sealed route


@pytest.mark.parametrize("name", NOT_WRITTEN)
@pytest.mark.parametrize("command", COMMANDS, ids=lambda module: module.__name__.split(".")[-1])
def test_a_name_the_truth_build_does_not_write_could_not_run_before_anything_is_read(
    command, name, tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    argv = _argv(command, tmp_path / "no-run", tmp_path / "no-tables")
    assert command.main([*argv, "--units", "nrel-from-may-2019", name]) == 2
    printed = capsys.readouterr().out
    if command in ROUTED and name in heldout_truth_build.UNITS:
        assert printed == (
            "COULD NOT RUN: the units are refused: the units named ['nrel-from-may-2019', "
            f"'{name}'] mix held-out units (['{name}']) with others; a run scores held-out units "
            "or others, never both: their tables come from two different truth builds; nothing "
            "read\n"
        )
        assert not list(tmp_path.iterdir())
        return
    assert printed.startswith(
        f"COULD NOT RUN: --units names {name}, not a training or development unit of the truth "
        "build ("
    )
    assert (
        "held-out units are read only through the sealed-data path at the tag frozen-plan, or at a "
        "logged parsing fix of it, and this command has no route to them"
    ) in printed
    assert len(printed.splitlines()) == 1
    assert not list(tmp_path.iterdir())


# --- 3. a known unit the build does not hold ------------------------------------------------------


@pytest.mark.parametrize("command", COMMANDS, ids=lambda module: module.__name__.split(".")[-1])
def test_a_known_unit_the_build_does_not_hold_could_not_run(command, tmp_path, monkeypatch, capsys):
    run, tables = setup_build(tmp_path, monkeypatch)  # the three default development units
    monkeypatch.setattr(forecast_report, "_head_commit", lambda: "a" * 40)
    out = _out(tmp_path, monkeypatch)
    assert command.main([*_argv(command, run, tables), "--units", "nrel-from-may-2019", "jpl"]) == 2
    printed = capsys.readouterr().out
    assert printed == f"COULD NOT RUN: jpl is not in the recorded truth build {run.name}\n"
    assert not list(out.iterdir())


# --- 4. the list itself ---------------------------------------------------------------------------


def test_the_known_units_are_the_training_and_development_units_of_the_truth_build():
    known = unit_list.KNOWN_UNITS
    assert known == (*truth_build.TRAIN_UNITS, *truth_build.DEVELOPMENT_UNITS)
    assert len(known) == 16
    assert not set(known) & set(heldout_truth_build.UNITS)


@pytest.mark.parametrize(
    ("units", "message"),
    [
        (["caltech", "OSL_1", "caltech"], "--units names caltech more than once"),
        ([], "--units names no unit"),
    ],
)
def test_a_unit_named_twice_or_no_unit_could_not_run(units, message):
    with pytest.raises(CouldNotRun) as refused:
        unit_list.check_units(units)
    assert str(refused.value) == message


def test_a_unit_named_twice_on_the_command_line_could_not_run(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    argv = _argv(ceiling_report, tmp_path / "no-run", tmp_path / "no-tables")
    assert ceiling_report.main([*argv, "--units", "jpl", "jpl"]) == 2
    assert capsys.readouterr().out == "COULD NOT RUN: --units names jpl more than once\n"


# --- 5. a pooled unit beside a site it pools (reading U2) ---------------------------------------

POOLED_SITES = load_norway.SITES["development"]


def _beside(sites) -> str:
    return (
        f"--units names {truth_build.POOLED} beside {', '.join(sites)}, which it pools, so their "
        "site-days would be read twice and counted twice in a figure pooled over the units named"
    )


@pytest.mark.parametrize("pooled_first", [True, False], ids=["pooled first", "site first"])
@pytest.mark.parametrize("site", POOLED_SITES)
def test_a_pooled_unit_beside_a_site_it_pools_could_not_run(site, pooled_first):
    units = [truth_build.POOLED, site] if pooled_first else [site, "caltech", truth_build.POOLED]
    with pytest.raises(CouldNotRun) as refused:
        unit_list.check_units(units)
    assert str(refused.value) == _beside([site])


def test_each_site_named_beside_its_pooled_unit_is_said_in_the_order_given():
    with pytest.raises(CouldNotRun) as refused:
        unit_list.check_units(["KRO", "nrel-from-may-2019", truth_build.POOLED, "OSL_1"])
    assert str(refused.value) == _beside(["KRO", "OSL_1"])


@pytest.mark.parametrize("command", COMMANDS, ids=lambda module: module.__name__.split(".")[-1])
def test_a_pooled_unit_beside_a_site_it_pools_could_not_run_before_anything_is_read(
    command, tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    argv = _argv(command, tmp_path / "no-run", tmp_path / "no-tables")
    assert command.main([*argv, "--units", "OSL_1", truth_build.POOLED]) == 2
    assert capsys.readouterr().out == f"COULD NOT RUN: {_beside(['OSL_1'])}\n"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "units",
    [
        *(command.UNITS for command in COMMANDS),
        POOLED_SITES,
        (truth_build.POOLED, "nrel-from-may-2019", "caltech", "OSL_S", "BAR"),
    ],
    ids=[
        *(f"{command.__name__.split('.')[-1]} default" for command in COMMANDS),
        "the five sites",
        "pooled beside other units",
    ],
)
def test_a_list_without_a_pooled_unit_beside_its_own_site_is_read(units):
    # The training sites OSL_S and BAR are not among the pooled development sites.
    assert unit_list.check_units(units) == tuple(units)
