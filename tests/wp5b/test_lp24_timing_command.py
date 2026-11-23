"""The timing command `python -m fleetreserve.lp24_timing` on small sizes, and its three results.

1. `SMALL` (a few cars and hours, both readings, two repeats) with `FLEETRESERVE_RUN_DIR` set:
   exit 0, and the results file holds, per reading, the programme's hours, its sizes and every
   solve, each optimal. The steered reading's rows are checked against a second build: each
   car's set is the same as its `carsets` true one-way set when there are at most 8 hours after
   hour 0 and no recovery hour (every car leaves within the hours written out).
2. A solve that ends otherwise (the solver door replaced by a made-up timeout): exit 1.
3. Arguments that cannot run (no cars; too few hours for the eliminated reading; an unknown
   reading): exit 2.
"""

import json

import numpy as np
import pytest

from fleetreserve import lp24_timing, sparse_programme
from fleetreserve.carsets import stack, true_one_way_rows
from fleetreserve.lp import LpResult

SMALL = ["--cars", "4", "--hours", "10", "--repeats", "2", "--reading", "both", "--seed", "3"]


def test_small_run_writes_its_results(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    assert lp24_timing.main(SMALL) == 0
    assert capsys.readouterr().out.rstrip().endswith("solved")
    document = json.loads((tmp_path / lp24_timing.RESULTS_NAME).read_text())
    assert document["cars"] == 4 and document["hours"] == 10 and document["rho"] == 0.1
    readings = document["readings"]
    assert sorted(readings) == ["eliminated", "steered"]
    assert readings["steered"]["programme_hours"] == 10
    assert readings["eliminated"]["programme_hours"] == 9
    for reading in readings.values():
        assert reading["cars"] == 4
        assert len(reading["solves"]) == 2
        assert all(s["outcome"] == "optimal" for s in reading["solves"])
        assert reading["variables"] > 0 and reading["rows"] > 0 and reading["nonzeros"] > 0


def test_steered_rows_are_the_true_sets_written_out():
    hours = 9
    cars = lp24_timing.made_up_cars(6, hours, 11)
    steered = lp24_timing.steered_rows(cars, hours, 0.1)
    reference = stack([true_one_way_rows(car, hours - 1, 0.1) for car in cars])
    np.testing.assert_array_equal(steered.car, reference.car)
    np.testing.assert_allclose(steered.coefficients, reference.coefficients)
    np.testing.assert_allclose(steered.bounds, reference.bounds)
    np.testing.assert_array_equal(steered.kinds, reference.kinds)


def test_a_failed_solve_exits_1(monkeypatch, capsys):
    def timeout(c, **kwargs):
        return LpResult("timeout", None, None, 0.5, kwargs["time_limit"], 1, "made up timeout")

    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    monkeypatch.setattr(sparse_programme, "solve_lp", timeout)
    assert lp24_timing.main(["--cars", "3", "--hours", "4", "--repeats", "1"]) == 1
    assert "FAILED" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["--cars", "0"],
        ["--hours", "5", "--reading", "eliminated"],
    ],
)
def test_arguments_that_cannot_run_exit_2(monkeypatch, argv, capsys):
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    assert lp24_timing.main(argv) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out


def test_an_unknown_reading_exits_2(monkeypatch):
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    with pytest.raises(SystemExit) as stop:
        lp24_timing.main(["--reading", "other"])
    assert stop.value.code == 2
