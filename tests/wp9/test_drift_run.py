"""The time-ordered drift-guard command (`drift_run`; plan section 6, T10) on made-up units.

1. A name that is no unit is refused before any table is opened: exit 2, "COULD NOT RUN", the
   unit named, no file written. A held-out unit (the two held-out Norwegian sites, both a.s.r.
   units, every Korean portfolio) away from the tag is refused by the sealed route's gate before
   any table is opened: exit 2, one line, no file written (the route itself:
   `tests/wp2/test_heldout_route_drift.py`). Training and development units pass the unit check,
   and the unit check refuses a held-out unit named with a training unit.
2. A model that reads the unit's other days (`empirical`, which reads days after the one it
   forecasts) is refused (exit 2), and so is a model fitted on the scored unit (`km` on a
   training unit, exit 2).
3. End to end with `km` (fitted on eight made-up training units) on a made-up development unit
   of 40 counting days, calibrated on the first 20, two step sizes: exit 0; the four files are
   written; the day table has one row per guarded day and path; each path's mean offer per
   counting day and final mean losses in the results are those of its rows in the day table;
   the summary says no step size is chosen; the results name the unit's role, the model's fit
   units and the matrix configuration of the dial.
"""

import csv
import json

import pytest
from gate_fake import COMMIT, gate_cars, matrix_document, write_build
from test_model_km import _training_cars

from fleetreserve import (
    drift_run,
    gate_run,
    load_asr,
    load_heldout_norway,
    load_korea,
    run_verify,
)
from fleetreserve.truth_build import DEVELOPMENT_UNITS, TRAIN_UNITS

HELD_OUT = (*load_heldout_norway.SITES, *load_asr.UNITS, *load_korea.UNITS)
NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.02-empirical",
    "main-rounded-hour-k0.3-eps0.02-oracle",
)
UNIT = "nrel-from-may-2019"
DAYS = 40
N_CAL = 20
ETAS = ("0.5", "1")


def _arguments(folder, *units, model="km"):
    return ["--model", model, "--dial", "0.02", "--tables", str(folder / "absent")] + [
        "--units",
        *units,
    ]


def test_an_unknown_unit_is_refused_before_any_table(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    assert drift_run.main(_arguments(tmp_path, "nowhere")) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN") and "nowhere" in printed
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("unit", HELD_OUT)
def test_a_held_out_unit_away_from_the_tag_could_not_run(unit, tmp_path, monkeypatch, capsys):
    """Refused by the sealed route's gate (the top folder given is no repository, so no tag),
    before any table is opened; the route itself is tested in
    `tests/wp2/test_heldout_route_drift.py`."""
    out = tmp_path / "out"
    out.mkdir()
    nowhere = tmp_path / "no-repository"
    nowhere.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    assert drift_run.main(_arguments(tmp_path, unit), root=nowhere) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN: REFUSED BY THE GATE: ")
    assert printed.endswith("; nothing read\n") and printed.count("\n") == 1
    assert list(out.iterdir()) == []


def test_training_and_development_units_pass_the_unit_check():
    units = (*TRAIN_UNITS, *DEVELOPMENT_UNITS)
    assert drift_run.check_units(units) == units
    for unit in HELD_OUT:
        with pytest.raises(drift_run.NotAllowed, match=unit):
            drift_run.check_units(("caltech", unit))


# --- 2 and 3: made-up tables ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("drift")
    units = {UNIT: gate_cars(DAYS, early_every=5), **{u: _training_cars(u) for u in TRAIN_UNITS}}
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return folder, run, tables, matrix


def _command(build, out, *units, model="km"):
    _, run, tables, matrix = build
    arguments = ["--model", model, "--dial", "0.02", "--matrix", str(matrix)]
    arguments += ["--tables", str(tables), "--truth-run", str(run), "--units", *units]
    arguments += ["--calibration-days", str(N_CAL), "--etas", *ETAS]
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return drift_run.main(arguments)


@pytest.mark.parametrize(("model", "unit"), [("empirical", UNIT), ("km", "jpl")])
def test_a_model_reading_the_unit_or_fitted_on_it_could_not_run(build, tmp_path, model, unit):
    out = tmp_path / "out"
    assert _command(build, out, unit, model=model) == 2
    assert list(out.iterdir()) == []


@pytest.fixture(scope="module")
def ran(build, tmp_path_factory):
    out = tmp_path_factory.mktemp("ran") / "out"
    return _command(build, out, UNIT), out


def test_the_command_runs_end_to_end_and_writes_its_files(ran):
    code, out = ran
    assert code == 0
    names = {p.name for p in out.iterdir()}
    assert names == {
        drift_run.RESULTS_NAME,
        drift_run.DAYS_NAME,
        drift_run.SUMMARY_NAME,
        drift_run.TIMING_NAME,
    }
    document = json.loads((out / drift_run.RESULTS_NAME).read_text())
    assert document["failed_units"] == []
    assert document["run"]["units"] == [UNIT]
    assert document["run"]["etas"] == [0.5, 1.0]
    assert document["run"]["n_calibration"] == N_CAL
    assert document["run"]["configuration"] == NAMES[2]
    entry = document["units"][UNIT]
    assert entry["role"] == "development"
    assert entry["fit_units"] == list(TRAIN_UNITS)
    assert entry["rotation_whatif"] is None
    assert entry["counting_days"] == DAYS
    assert entry["calibration"]["days"] == N_CAL
    assert entry["guarded_days"] == DAYS - N_CAL
    summary = (out / drift_run.SUMMARY_NAME).read_text()
    assert "no step size is chosen" in summary.lower()


def test_the_results_are_the_day_tables_rows(ran):
    _, out = ran
    document = json.loads((out / drift_run.RESULTS_NAME).read_text())
    with (out / drift_run.DAYS_NAME).open() as handle:
        rows = list(csv.DictReader(handle))
    paths = document["units"][UNIT]["paths"]
    assert sorted(paths) == ["A", "B"]
    count = 0
    for tier, by_start in paths.items():
        assert sorted(by_start) == ["calibrated", "zero"]
        for start, by_eta in by_start.items():
            assert sorted(by_eta) == sorted(ETAS)
            for eta, summary in by_eta.items():
                mine = [
                    r
                    for r in rows
                    if (r["unit"], r["tier"], r["start"], r["eta"]) == (UNIT, tier, start, eta)
                ]
                count += len(mine)
                assert len(mine) == DAYS - N_CAL == summary["guarded_counting_days"]
                offers = [float(r["offer_kwh"]) for r in mine]
                assert summary["mean_offer_kwh_per_counting_day"] == pytest.approx(
                    sum(offers) / len(offers)
                )
                stepped = [r for r in mine if r["step"]]
                assert len(stepped) == summary["offering_days"]
                if stepped:
                    for loss in ("slot", "driver"):
                        values = [float(r[f"{loss}_loss"]) for r in stepped]
                        assert summary["losses"][loss]["mean"] == pytest.approx(
                            sum(values) / len(values)
                        )
                        assert float(stepped[-1][f"mean_{loss}_loss"]) == pytest.approx(
                            summary["losses"][loss]["mean"]
                        )
    assert count == len(rows) == (DAYS - N_CAL) * 2 * 2 * len(ETAS)
