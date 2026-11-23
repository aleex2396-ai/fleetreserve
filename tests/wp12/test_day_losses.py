"""The per-day losses file of the scoring commands (`day_losses`, written by `gate_run` and
`learn_run`), on the small made-up gate units of `learn_fake` with two service hours and a made-up
verified truth build. With two service hours the rungs' raw losses differ (checked: on some job
the one-way or two-way losses differ from the fixed plan's), so a file that writes one rung's
losses under another fails the per-search means below.

1. `gate_run` writes `gate-day-losses.json`, and it holds exactly what the calibration read: per
   job, its scales are the union of the scales every search of every rung evaluated (from
   `gate.json`), and the raw losses of the offering calibration days, made a running maximum over
   each search's own evaluated scales and averaged, give back every mean of that search in
   `gate.json`; the days' roles, cars, certified values and ceilings, and the test days' losses at
   each tier's scale, are those of `gate-days.csv`.
2. `learn_run` with the empirical model writes `learn-day-losses.json`, byte-identical to the
   gate run's file on the same build and matrix (its gate-format results equal the gate run's,
   `tests/wp8/test_learn_run.py`).
"""

import csv
import json

import numpy as np
import pytest
from gate_fake import COMMIT, write_build
from learn_fake import small_gate_units, small_matrix

from fleetreserve import gate_run, learn_run, run_verify
from fleetreserve.gate_config import CEILINGS
from fleetreserve.rungs import RUNGS

SEARCHES = ("evaluations", "slot_only", "driver_only")


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, small_gate_units(), service_hours=2)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    outs = {}
    for name, command, extra in (
        ("gate", gate_run.main, []),
        ("learn", learn_run.main, ["--model", "empirical"]),
    ):
        out = folder / name
        out.mkdir()
        with pytest.MonkeyPatch.context() as monkeypatch:
            _verified(monkeypatch)
            monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
            assert command([*arguments, *extra]) == 0
        outs[name] = out
    return outs


def _search_evaluations(entry: dict, search: str) -> list:
    return entry["evaluations"] if search == "evaluations" else entry[search]["evaluations"]


def _days_rows(out) -> list[dict]:
    lines = (out / gate_run.DAYS_NAME).read_text().splitlines()
    return list(csv.DictReader(line for line in lines if not line.startswith("#")))


def test_the_gate_run_file_holds_what_the_calibration_read(runs):
    out = runs["gate"]
    document = json.loads((out / gate_run.RESULTS_NAME).read_text())
    losses = json.loads((out / gate_run.DAY_LOSSES_NAME).read_text())
    assert losses["rungs"] == list(RUNGS)
    calibrated = {
        (name, unit)
        for name, c in document["configurations"].items()
        for unit, entry in c["units"].items()
        if "rungs" in entry
    }
    assert {(n, r["unit"]) for r in losses["jobs"] for n in r["configurations"]} == calibrated
    rows = _days_rows(out)
    rungs_differ = False
    for record in losses["jobs"]:
        names = record["configurations"]
        first = document["configurations"][names[0]]
        assert sorted(names) == sorted([names[0], *first["work_shared_with"]])
        entry = first["units"][record["unit"]]
        calibration = record["calibration"]
        scales = calibration["scales"]
        union = {
            e[0] for r in RUNGS for s in SEARCHES for e in _search_evaluations(entry["rungs"][r], s)
        }
        assert scales == sorted(union) and len(scales) > 0
        n = len(calibration["days"])
        assert n == entry["offering_calibration_days"] > 0
        for rung in RUNGS:
            slot = np.array(calibration["slot_loss"][rung])
            driver = np.array(calibration["driver_loss"][rung])
            assert slot.shape == driver.shape == (n, len(scales))
            rungs_differ = rungs_differ or any(
                calibration[key][rung] != calibration[key][RUNGS[0]]
                for key in ("slot_loss", "driver_loss")
            )
            for search in SEARCHES:
                found = _search_evaluations(entry["rungs"][rung], search)
                own = sorted(e[0] for e in found)
                columns = [scales.index(b) for b in own]
                mono_slot = np.maximum.accumulate(slot[:, columns], axis=1).sum(axis=0) / n
                mono_driver = np.maximum.accumulate(driver[:, columns], axis=1).sum(axis=0) / n
                means = {e[0]: (e[1], e[2]) for e in found}
                for k, beta in enumerate(own):
                    assert mono_slot[k] == pytest.approx(means[beta][0], abs=1e-12)
                    assert mono_driver[k] == pytest.approx(means[beta][1], abs=1e-12)
        mine = [r for r in rows if r["unit"] == record["unit"] and r["configuration"] == names[0]]
        days = record["days"]
        assert [r["role"] for r in mine] == days["role"]
        assert [int(r["cars"]) for r in mine] == days["cars"]
        assert [r["abstain_reason"] == "" for r in mine] == days["offers"]
        assert [float(r["certified_value_kwh"]) for r in mine] == days["certified_value_kwh"]
        assert [float(r["ceiling_hindsight_kwh"]) for r in mine] == days["ceiling_kwh"][CEILINGS[0]]
        test = [d for d, role in enumerate(days["role"]) if role == "test"]
        assert record["test"]["days"] == test
        for rung in RUNGS:
            by_tier = record["test"]["rungs"][rung]
            assert by_tier["A"]["scale"] == entry["rungs"][rung]["beta_hat"]
            assert by_tier["B"]["scale"] == entry["rungs"][rung]["tier_b"]["beta_hat"]
            for tier, column in (("A", rung), ("B", f"{rung}_tier_b")):
                if by_tier[tier]["scale"] <= 0:
                    assert by_tier[tier]["slot_loss"] is None
                    continue
                for key in ("slot_loss", "driver_loss"):
                    expected = [float(mine[d][f"{column}_{key}"]) for d in test]
                    assert by_tier[tier][key] == expected
    assert rungs_differ


def test_the_learn_run_writes_the_same_file(runs):
    gate = (runs["gate"] / gate_run.DAY_LOSSES_NAME).read_bytes()
    learn = (runs["learn"] / learn_run.DAY_LOSSES_NAME).read_bytes()
    assert learn == gate
