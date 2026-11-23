"""The 9/80 phase: the recorded command `nine_eighty_run` on a made-up JPL day table, the
settings file `nine_eighty_settings` writes from its results, and `load_parity`. No project data
is read."""

import json

import pandas as pd
import pytest

from fleetreserve import gate_run, nine_eighty_run, nine_eighty_settings
from fleetreserve.run_record import MANIFEST_NAME
from fleetreserve.truth import CAR_COLUMNS, DAY_COLUMNS
from fleetreserve.truth_verify import CouldNotRun


def write_days(folder, rows):
    """`jpl-days.csv` from (date, cars, counts) rows and an empty `jpl-cars.csv`."""
    days = pd.DataFrame(
        [("jpl", "jpl", d, f"{d}T08:00:00-08:00", "3", str(c), str(k)) for d, c, k in rows],
        columns=list(DAY_COLUMNS),
    )
    days.to_csv(folder / "jpl-days.csv", index=False)
    pd.DataFrame(columns=list(CAR_COLUMNS)).to_csv(folder / "jpl-cars.csv", index=False)


# Fridays 4, 11, 18 and 25 Jan 2019: parity 1, 0, 1, 0 (`calendar_inputs.fortnight_parity`)
ROWS = [
    ("2019-01-04", 3, True),
    ("2019-01-11", 9, True),
    ("2019-01-18", 5, True),
    ("2019-01-25", 11, True),
    ("2019-01-10", 1, True),  # a Thursday, not read
    ("2019-02-01", 0, False),  # parity 1, not a counting day, not read
]


@pytest.fixture
def inputs(monkeypatch):
    calls = []

    def check(run, tables, units):
        calls.append(tuple(units))
        return {"setting": "main", "truth_run": "r", "table_sha256": {}}

    monkeypatch.setattr(gate_run, "check_inputs", check)
    return calls


def test_the_run_estimates_the_parity_by_hand(tmp_path, monkeypatch, capsys, inputs):
    write_days(tmp_path, ROWS)
    out = tmp_path / "run"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    assert nine_eighty_run.main(["--tables", str(tmp_path)]) == 0
    assert inputs == [("jpl",)]
    # parity 0: 9 and 11, mean 10; parity 1: 3 and 5, mean 4; fewer cars on parity 1
    found = json.loads((out / nine_eighty_settings.RESULTS_NAME).read_text())
    assert found["unit"] == "jpl"
    assert found["estimate"] == {
        "parity": 1,
        "fridays_0": 2,
        "fridays_1": 2,
        "mean_cars_0": 10.0,
        "mean_cars_1": 4.0,
    }
    assert "estimated parity of Fridays off 1" in capsys.readouterr().out


def test_the_run_without_a_decision_and_without_a_build(tmp_path, monkeypatch, capsys, inputs):
    write_days(tmp_path, [("2019-01-04", 3, True), ("2019-01-11", 3, True)])
    out = tmp_path / "run"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    assert nine_eighty_run.main(["--tables", str(tmp_path)]) == 1
    assert (
        json.loads((out / nine_eighty_settings.RESULTS_NAME).read_text())["estimate"]["parity"]
        is None
    )

    def refuse(run, tables, units):
        raise CouldNotRun("no build")

    monkeypatch.setattr(gate_run, "check_inputs", refuse)
    assert nine_eighty_run.main(["--tables", str(tmp_path)]) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out


def results_file(folder, estimate, unit="jpl"):
    folder.mkdir()
    (folder / MANIFEST_NAME).write_text("{}")
    path = folder / nine_eighty_settings.RESULTS_NAME
    path.write_text(json.dumps({"unit": unit, "estimate": estimate}))
    return path


GOOD = {"parity": 1, "fridays_0": 2, "fridays_1": 2, "mean_cars_0": 10.0, "mean_cars_1": 4.0}


@pytest.fixture
def verified(monkeypatch):
    monkeypatch.setattr(nine_eighty_settings, "require_verified", lambda path: {"commit": "c" * 40})


def test_the_settings_file_is_written_from_verified_results_and_read_back(tmp_path, verified):
    path = results_file(tmp_path / "run", GOOD)
    out = tmp_path / "phase.json"
    assert nine_eighty_settings.main([str(path), "--out", str(out)]) == 0
    document = json.loads(out.read_text())
    assert document["parity"] == 1 and document["estimate"] == GOOD
    assert document["results"]["run_folder"] == "run"
    assert nine_eighty_settings.load_parity(out) == 1


@pytest.mark.parametrize(
    "estimate,unit",
    [
        ({**GOOD, "parity": 0}, "jpl"),
        ({**GOOD, "mean_cars_1": 10.0}, "jpl"),
        ({**GOOD, "fridays_1": 0}, "jpl"),
        ({**GOOD, "mean_cars_0": None}, "jpl"),
        (GOOD, "caltech"),
    ],
)
def test_the_settings_file_is_refused_for_results_that_do_not_fit(
    tmp_path, verified, estimate, unit
):
    path = results_file(tmp_path / "run", estimate, unit)
    out = tmp_path / "phase.json"
    assert nine_eighty_settings.main([str(path), "--out", str(out)]) == 1
    assert not out.exists()


def test_load_parity_refuses_an_absent_or_bad_file(tmp_path):
    with pytest.raises(nine_eighty_settings.PhaseError):
        nine_eighty_settings.load_parity(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    for parity in (2, "1", None, True):
        bad.write_text(json.dumps({"about": "", "estimate": {}, "parity": parity, "results": {}}))
        with pytest.raises(nine_eighty_settings.PhaseError):
            nine_eighty_settings.load_parity(bad)
    bad.write_text(json.dumps({"parity": 1}))
    with pytest.raises(nine_eighty_settings.PhaseError):
        nine_eighty_settings.load_parity(bad)
