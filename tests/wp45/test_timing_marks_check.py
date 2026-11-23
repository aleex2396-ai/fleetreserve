"""`tools/timing_marks_check.py` on made-up recorded timing runs (`recorded_fakes`): each item
passes when the solve times are logged with the machine, whatever the times and the machine's
load; it could not run when the run is missing, unverified, stale, at other sizes or malformed."""

import importlib
import sys
from pathlib import Path

import pytest
from recorded_fakes import (
    clean_git_environment,
    docs_commit,
    earlier_note,
    make_repo,
    make_run,
    new_commit,
)

from fleetreserve.timing_marks import PROGRAMMES, RESULTS_NAME

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
check = importlib.import_module("timing_marks_check")

LABEL = "wp4-timing"
MACHINE = {
    "platform": "macOS-15.0-arm64-arm-64bit",
    "processor": "made-up processor",
    "cpu_count": 10,
    "python": "3.12.0",
}


def programme(seconds=(1.0, 0.5, 0.7), outcomes=("optimal",) * 3) -> dict:
    solves = [
        {"outcome": o, "seconds": s, "solver_seconds": s / 2}
        for s, o in zip(seconds, outcomes, strict=True)
    ]
    optimal = all(o == "optimal" for o in outcomes)
    return {
        "solves": solves,
        "best_seconds": min(seconds) if optimal else None,
        "sizes": {"variables": 1, "rows": 1, "nonzeros": 1},
    }


def document(**programmes) -> dict:
    found = {name: programme() for name in PROGRAMMES}
    found.update({name.replace("_", "-"): value for name, value in programmes.items()})
    return {
        "machine": dict(MACHINE),
        "machine_at_start": {"load_average_1_5_15_min": [1.0, 1.5, 2.0], "cpu_count": 10},
        "machine_at_end": {"load_average_1_5_15_min": [1.2, 1.5, 2.0], "cpu_count": 10},
        "box_day": {"cars": 100, "service_hours": 8, "recovery_hours": 5},
        "lp24_fleet": {"cars": 50, "hours": 14},
        "repeats": 3,
        "programmes": found,
    }


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    clean_git_environment(monkeypatch)


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "repo")


def run_check(repo: Path, item: str, *extra: str) -> int:
    return check.main([f"--{item}", "--top", str(repo), *extra])


def recorded(repo, doc, **options):
    return make_run(repo, LABEL, RESULTS_NAME, doc, **options)


# --- run time is reported, never judged ----------------------------------------------------------


@pytest.mark.parametrize("seconds", [0.001, 3.0, 30.0, 60.0, 5000.0])
def test_any_logged_solve_time_passes_every_item(repo, capsys, seconds):
    doc = document()
    doc["programmes"] = {name: programme((seconds,) * 3) for name in PROGRAMMES}
    recorded(repo, doc)
    for item in check.ITEMS:
        capsys.readouterr()
        assert run_check(repo, item) == 0
        out = capsys.readouterr().out
        assert "logged: " in out and "FAILED" not in out
        assert f"{seconds:.3f} s" in out


def test_a_busy_machine_does_not_decide_an_item(repo, capsys):
    doc = document()
    doc["machine_at_start"]["load_average_1_5_15_min"][0] = 12.5
    doc["machine_at_end"]["load_average_1_5_15_min"][0] = 11.0
    recorded(repo, doc)
    for item in check.ITEMS:
        capsys.readouterr()
        assert run_check(repo, item) == 0
        out = capsys.readouterr().out
        assert "12.50" in out and "11.00" in out  # reported beside the times


def test_each_item_prints_the_machine_and_says_time_is_not_judged(repo, capsys):
    recorded(repo, document())
    capsys.readouterr()
    assert run_check(repo, "box") == 0
    out = capsys.readouterr().out
    assert "made-up processor" in out and "10 processors" in out
    assert "macOS-15.0-arm64-arm-64bit" in out and "Python 3.12.0" in out
    assert "run time is reported, never judged" in out


def test_the_box_item_logs_the_certificate_and_prints_the_others_beside(repo, capsys):
    recorded(repo, document())
    capsys.readouterr()
    assert run_check(repo, "box") == 0
    out = capsys.readouterr().out
    assert "logged: hourly-certificate:" in out
    assert "beside it: hourly-certificate-first-stage" in out
    assert "beside it: block-certificate" in out
    assert "general split, which is slower, is not the box programme" in out


def test_the_tree_item_logs_both_trees(repo, capsys):
    recorded(repo, document())
    capsys.readouterr()
    assert run_check(repo, "tree") == 0
    out = capsys.readouterr().out
    assert "logged: hourly-tree" in out and "logged: block-tree" in out


def test_the_lp24_item_logs_the_steered_reading_and_prints_the_other_beside(repo, capsys):
    recorded(repo, document())
    capsys.readouterr()
    assert run_check(repo, "lp24") == 0
    out = capsys.readouterr().out
    assert "logged: lp24-steered" in out and "beside it: lp24-eliminated" in out


# --- the machine description ---------------------------------------------------------------------


def test_a_run_without_the_machine_description_could_not_run(repo, capsys):
    doc = document()
    del doc["machine"]
    recorded(repo, doc)
    for item in check.ITEMS:
        capsys.readouterr()
        assert run_check(repo, item) == 2
        assert "the machine it ran on is not recorded" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("platform", ""),
        ("processor", None),
        ("python", 3),
        ("cpu_count", 0),
        ("cpu_count", True),
        ("cpu_count", "10"),
    ],
)
def test_a_malformed_machine_description_could_not_run(repo, field, value):
    doc = document()
    doc["machine"][field] = value
    recorded(repo, doc)
    assert run_check(repo, "box") == 2


def test_a_missing_field_of_the_machine_description_could_not_run(repo):
    doc = document()
    del doc["machine"]["processor"]
    recorded(repo, doc)
    assert run_check(repo, "tree") == 2


# --- sizes, outcomes and malformed times ---------------------------------------------------------


@pytest.mark.parametrize(
    ("item", "key", "field", "value"),
    [
        ("box", "box_day", "cars", 99),
        ("box", "box_day", "service_hours", 7),
        ("tree", "box_day", "cars", 10),
        ("lp24", "lp24_fleet", "cars", 49),
        ("lp24", "lp24_fleet", "hours", 9),
    ],
)
def test_a_run_at_other_sizes_could_not_run(repo, capsys, item, key, field, value):
    doc = document()
    doc[key][field] = value
    recorded(repo, doc)
    capsys.readouterr()
    assert run_check(repo, item) == 2
    assert "not at the plan's sizes" in capsys.readouterr().out


def test_another_number_of_repeats_could_not_run(repo):
    doc = document()
    doc["repeats"] = 2
    recorded(repo, doc)
    assert run_check(repo, "box") == 2


@pytest.mark.parametrize("seconds", [None, "1.0", True])
def test_a_solve_without_a_number_for_its_time_could_not_run(repo, capsys, seconds):
    doc = document()
    doc["programmes"]["hourly-tree"]["solves"][1]["seconds"] = seconds
    recorded(repo, doc)
    capsys.readouterr()
    assert run_check(repo, "tree") == 2
    assert "a solve without a time" in capsys.readouterr().out
    assert run_check(repo, "box") == 0


def test_a_run_where_every_solve_timed_out_could_not_run_and_never_fails(repo, capsys):
    doc = document(hourly_certificate=programme((30.2, 30.1, 30.4), ("timeout",) * 3))
    recorded(repo, doc, exit_code=1)
    for item in check.ITEMS:
        capsys.readouterr()
        assert run_check(repo, item) == 2
        assert "not verified" in capsys.readouterr().out


def test_a_single_solve_not_optimal_makes_the_run_unverified(repo, capsys):
    doc = document(hourly_tree=programme((1.0, 1.0, 1.0), ("optimal", "error", "optimal")))
    recorded(repo, doc, exit_code=1)
    capsys.readouterr()
    assert run_check(repo, "tree") == 2
    assert "not verified" in capsys.readouterr().out


# --- which run is read ---------------------------------------------------------------------------


def test_no_run_could_not_run(repo, capsys):
    for item in check.ITEMS:
        assert run_check(repo, item) == 2
    assert "no run named" in capsys.readouterr().out


def test_the_newest_run_is_read_not_an_older_good_one(repo):
    recorded(repo, document(), stamp="20260101T000000Z")
    doc = document()
    del doc["machine"]
    recorded(repo, doc, stamp="20260102T000000Z")
    assert run_check(repo, "box") == 2


def test_a_newer_broken_run_is_not_skipped_for_an_older_good_one(repo):
    recorded(repo, document(), stamp="20260101T000000Z")
    recorded(repo, document(), time_out=True, stamp="20260102T000000Z")
    assert run_check(repo, "box") == 2


def test_a_run_at_an_older_commit_is_stale(repo, capsys):
    recorded(repo, document())
    new_commit(repo)
    capsys.readouterr()
    assert run_check(repo, "box") == 2
    assert "is stale" in capsys.readouterr().out


def test_a_run_from_an_earlier_commit_with_only_docs_changed_passes_and_says_so(repo, capsys):
    run = recorded(repo, document())
    earlier = docs_commit(repo)
    capsys.readouterr()
    assert run_check(repo, "box") == 0
    last = capsys.readouterr().out.splitlines()[-1]
    assert last.startswith("logged: every solve time of ")
    assert f"in {run.name}{earlier_note(earlier)}" in last


def test_a_run_from_an_earlier_commit_that_cannot_decide_is_named_with_the_note(repo, capsys):
    doc = document(hourly_tree=programme((1.0, 1.0, 1.0), ("optimal", "error", "optimal")))
    run = recorded(repo, doc, exit_code=1)
    earlier = docs_commit(repo)
    capsys.readouterr()
    assert run_check(repo, "tree") == 2
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"COULD NOT RUN: {run.name}{earlier_note(earlier)}")
    assert lines[0].endswith(
        "ended with exit code 1, so it is not verified and cannot pass an item"
    )


def test_a_changed_results_file_could_not_run(repo, capsys):
    folder = recorded(repo, document())
    text = (folder / RESULTS_NAME).read_text()
    (folder / RESULTS_NAME).write_text(text + " ")
    capsys.readouterr()
    assert run_check(repo, "box") == 2
    assert "is not verified" in capsys.readouterr().out


def test_a_programme_missing_from_the_run_could_not_run(repo):
    doc = document()
    del doc["programmes"]["block-tree"]
    recorded(repo, doc)
    assert run_check(repo, "tree") == 2
    assert run_check(repo, "box") == 0


def test_exactly_one_item_is_required(repo):
    for arguments in ([], ["--box", "--tree"]):
        with pytest.raises(SystemExit) as raised:
            check.main([*arguments, "--top", str(repo)])
        assert raised.value.code == 2
