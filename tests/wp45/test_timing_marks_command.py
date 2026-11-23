"""`python -m fleetreserve.timing_marks` on small made-up fleets (the plan's sizes are for the
recorded run only): what it writes, its exit statuses, and that it gives no verdict."""

import json
from types import SimpleNamespace

import pytest

from fleetreserve import timing_marks
from fleetreserve.sparse_programme import Sizes

SMALL = ["--cars", "6", "--service-hours", "3", "--lp24-cars", "4", "--lp24-hours", "10"]


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    folder = tmp_path_factory.mktemp("timing")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        code = timing_marks.main([*SMALL, "--repeats", "2"])
    return code, json.loads((folder / timing_marks.RESULTS_NAME).read_text())


def test_every_programme_is_timed_with_every_solve_listed(written):
    code, document = written
    assert code == 0
    assert list(document["programmes"]) == list(timing_marks.PROGRAMMES)
    for found in document["programmes"].values():
        assert [s["outcome"] for s in found["solves"]] == ["optimal", "optimal"]
        assert found["best_seconds"] == min(s["seconds"] for s in found["solves"])
        assert all(s["seconds"] >= s["solver_seconds"] for s in found["solves"])
        assert found["sizes"]["variables"] > 0


def test_sizes_and_the_machine_state_are_recorded(written):
    _, document = written
    assert document["box_day"]["cars"] == 6
    assert document["box_day"]["service_hours"] == 3
    assert document["box_day"]["recovery_hours"] == 5
    assert document["box_day"]["shape_rule"] == "largest-smallest-width"
    assert document["lp24_fleet"] == {
        "generator": "lp24_timing.made_up_cars",
        "cars": 4,
        "hours": 10,
        "rho": 0.1,
        "seed": 0,
        "base_set": "day average",
    }
    for when in ("start", "end"):
        assert len(document[f"machine_at_{when}"]["load_average_1_5_15_min"]) == 3
    machine = document["machine"]
    for field in ("platform", "processor", "python"):
        assert isinstance(machine[field], str) and machine[field]
    assert isinstance(machine["cpu_count"], int) and machine["cpu_count"] >= 1
    assert document["programmes"]["lp24-steered"]["programme_hours"] == 10
    assert document["programmes"]["lp24-eliminated"]["programme_hours"] == 9
    text = json.dumps(document)
    assert "passed" not in text and "verdict" not in text


def test_the_two_stages_are_both_in_the_hourly_certificate(written):
    _, document = written
    both = document["programmes"]["hourly-certificate"]
    first = document["programmes"]["hourly-certificate-first-stage"]
    assert both["sizes"]["rows"] > first["sizes"]["rows"]  # the second stage's extra rows


def test_a_solve_that_is_not_optimal_gives_exit_1(tmp_path, monkeypatch, capsys):
    def timed_out(rows, product, time_limit=None):
        return SimpleNamespace(outcome="timeout", seconds=0.01, sizes=Sizes(1, 1, 1))

    monkeypatch.setattr(timing_marks, "corner_tree", timed_out)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    assert timing_marks.main([*SMALL, "--repeats", "1"]) == 1
    assert "FAILED: a solve did not end optimal" in capsys.readouterr().out
    document = json.loads((tmp_path / timing_marks.RESULTS_NAME).read_text())
    assert document["programmes"]["hourly-tree"]["best_seconds"] is None


@pytest.mark.parametrize(
    "bad",
    [["--cars", "0"], ["--service-hours", "9"], ["--lp24-hours", "8"], ["--repeats", "0"]],
)
def test_bad_arguments_could_not_run(bad, capsys):
    assert timing_marks.main(bad) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out


def test_a_missing_run_folder_could_not_run_before_any_solve(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path / "none"))
    assert timing_marks.main([*SMALL, "--repeats", "1"]) == 2
    out = capsys.readouterr().out
    assert "does not exist" in out and "variables" not in out
