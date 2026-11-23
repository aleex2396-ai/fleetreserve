"""A gate run in which some days abstain and an offering test day fails, end to end.

On the usual made-up units every test day offers and the calibrated scale is 0, so a mean over the
offering days and a mean over all days cannot be told apart. Here the unit has 41 days, 30 of them
calibrating (the plan's 50 is replaced to keep the run short). `zero_value_kwh` is raised to 4.7 so
that every fourth day (certified value 4.5 kWh) abstains, and one test day has a car that leaves
in the service hour, which the fixed plan cannot cover. The unit's seeded split puts that day in the
test set. The test checks, from the per-day file, that

1. some test days abstain, and a rung with a calibrated scale above 0 has an offering test day
   that fails;
2. that rung's written mean slot and driver losses equal the recount over the offering test days,
   and the mean slot loss over all test days is a different number;
3. the written counts of offering and failing days equal the recount.
"""

import csv
import dataclasses
import json

import pytest
from gate_fake import (
    COMMIT,
    GATE_UNITS,
    ROUNDED,
    gate_cars,
    matrix_document,
    write_build,
)
from truth_tables_fake import date_of

from fleetreserve import gate_run, run_verify

DAYS = 41
EARLY_DAY = 13  # a test day (module text) with a certified value above the zero value


def _units() -> dict:
    early = date_of(EARLY_DAY)
    cars = [
        dataclasses.replace(c, plug_out=11.3) if c.date == early and c.session.endswith("-1") else c
        for c in gate_cars(DAYS, early_every=0)
    ]
    return {GATE_UNITS[0]: gate_cars(5), GATE_UNITS[1]: cars, GATE_UNITS[2]: gate_cars(5, "C")}


@pytest.fixture(scope="module")
def run_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("offering")
    run, tables = write_build(folder, _units())
    document = matrix_document(ROUNDED)
    document["n_calibration"] = 30
    document["zero_value_kwh"] = 4.7
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    out = folder / "out"
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
        code = gate_run.main([*arguments, "--only", ROUNDED[0]])
    assert code == 0
    return out


def test_means_over_offering_days_differ_from_means_over_all_days(run_folder):
    document = json.loads((run_folder / gate_run.RESULTS_NAME).read_text())
    lines = [
        line
        for line in (run_folder / gate_run.DAYS_NAME).read_text().splitlines()
        if not line.startswith("#")
    ]
    rows = [
        r
        for r in csv.DictReader(lines)
        if r["unit"] == GATE_UNITS[1] and r["configuration"] == ROUNDED[0] and r["role"] == "test"
    ]
    offering = [r for r in rows if r["abstain_reason"] == ""]
    assert any(r["date"] == date_of(EARLY_DAY) for r in rows)  # the split puts it in the test set
    assert 0 < len(offering) < len(rows)
    entry = document["configurations"][ROUNDED[0]]["units"][GATE_UNITS[1]]
    assert entry["offering_calibration_days"] >= 19
    checked = 0
    for rung, found in entry["rungs"].items():
        test = found["test"]
        assert test["days"] == len(rows) and test["offering_days"] == len(offering)
        if found["beta_hat"] <= 0:
            continue
        slot = [float(r[f"{rung}_slot_loss"]) for r in offering]
        driver = [float(r[f"{rung}_driver_loss"]) for r in offering]
        every_slot = [float(r[f"{rung}_slot_loss"]) for r in rows]
        assert test["failing_days"] == sum(loss > 0 for loss in slot)
        assert test["mean_slot_loss_offering_days"] == pytest.approx(sum(slot) / len(slot))
        assert test["mean_driver_loss_offering_days"] == pytest.approx(sum(driver) / len(driver))
        if test["failing_days"] > 0:
            checked += 1
            assert sum(every_slot) / len(every_slot) != pytest.approx(
                test["mean_slot_loss_offering_days"]
            )
    assert checked >= 1  # a scale above 0 and an offering test day that fails
