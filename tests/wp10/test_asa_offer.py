"""The comparison command (`compare_run`) with `asa` calibrated to an offer above 0, end to end.

The units of `test_asa_command` give `asa` no offer anywhere: 2 calibration days are too few for
either tier. Here the second gate unit holds 11 made-up days of `gate_fake.gate_cars` without the
early departure (`early_every=0`), 9 of which calibrate in each split: enough for tier B (the slot
alpha 0.1 needs at least 9 offering calibration days), not for tier A (the driver alpha 0.05 needs
19). The first and third units hold 2 days and cannot be split. To keep the test short, the
audited paths of every prepared day are cut to their first 2 (`gate_day.audited_paths`
replaced); everything else runs as the command runs it, `--methods asa`, 3 splits, one worker.

1. The run ends with exit 0, `asa`'s tier B mean offer in the forecast variant is above 0, and
   every split of it offers.
2. Its offer is recomputed in this file: the prepared days of the forecast variant built here
   through the same public pieces (`dial_choice.make_factory`, `gate_day.prepare_day`), `asa`'s
   day-level slot losses at every grid scale from `compare_day.evaluate`, the calibration of
   plan section 6 for tier B written out here (running maximum over the scales, the largest scale
   at which `(sum + 1) / (n + 1) <= 0.1` holds, stopping at the first failure), and the offer
   the scale times the mean certified value of the split's test days. The recomputed scale and
   offer of each split equal the rows of `compare-splits.csv`, and their mean the results' mean
   offer.
"""

import csv
import io
import json

import numpy as np
import pytest
from compare_fake import NAMES
from gate_fake import COMMIT, GATE_UNITS, gate_cars, matrix_document, write_build

from fleetreserve import (
    compare_day,
    compare_methods,
    compare_run,
    dial_choice,
    dial_splits,
    gate_config,
    gate_day,
    gate_run,
    learn_run,
    run_verify,
)
from fleetreserve.days_from_truth import load_unit_days

SPLITS = 3
UNIT = GATE_UNITS[1]
PATHS = 2


def _cut(original):
    def cut(unit, date, n_service):
        return original(unit, date, n_service)[:PATHS]

    return cut


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        GATE_UNITS[0]: gate_cars(2),
        UNIT: gate_cars(11, early_every=0),
        GATE_UNITS[2]: gate_cars(2, "B"),
    }
    run, tables = write_build(folder, units)
    document = matrix_document(NAMES)
    document["n_calibration"] = 9
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix


@pytest.fixture(scope="module")
def out(build, tmp_path_factory):
    run, tables, matrix = build
    folder = tmp_path_factory.mktemp("run")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setattr(gate_day, "audited_paths", _cut(gate_day.audited_paths))
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        arguments = ["--model", "empirical", "--dial", "0.03", "--matrix", str(matrix)]
        arguments += ["--tables", str(tables), "--truth-run", str(run), "--splits", str(SPLITS)]
        code = compare_run.main([*arguments, "--methods", "asa"])
    return code, folder


def _rows(folder):
    text = (folder / compare_run.SPLITS_NAME).read_text()
    found = [
        r
        for r in csv.DictReader(io.StringIO(text))
        if (r["unit"], r["variant"], r["tier"], r["method"]) == (UNIT, "forecast", "B", "asa")
    ]
    return sorted(found, key=lambda r: int(r["seed"]))


def test_asa_offers_in_tier_b(out):
    code, folder = out
    assert code == 0
    document = json.loads((folder / compare_run.RESULTS_NAME).read_text())
    assert document["methods"] == ["asa"]
    tier = document["units"][UNIT]["variants"]["forecast"]["methods"]["asa"]["B"]
    assert tier["mean_offer_kwh_per_test_day"] > 0
    assert tier["splits_that_offer"] == SPLITS
    rows = _rows(folder)
    assert len(rows) == SPLITS and all(float(r["scale"]) > 0 for r in rows)


def _calibrated(slot: np.ndarray, offering_calibration: list[int]) -> float:
    """Tier B: the largest grid scale at which (sum + 1) / (n + 1) <= 0.1 for the running maximum
    of the slot losses over the scales; 0 below 9 offering calibration days."""
    n = len(offering_calibration)
    if 0.1 * (n + 1) < 1 - 1e-9:
        return 0.0
    running = np.maximum.accumulate(slot[offering_calibration], axis=1)
    scale = 0.0
    for k, g in enumerate(dial_splits.GRID):
        if running[:, k].sum() + 1 > 0.1 * (n + 1) + 1e-9:
            break
        scale = g
    return scale


def test_asas_offer_is_recomputed(build, out):
    run, tables, path = build
    _, folder = out
    matrix = gate_config.load(path)
    configuration = compare_run.dial_configurations(matrix, 0.03)["forecast"]
    loaded = {
        name: load_unit_days(
            tables, name, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        for name in matrix.gate_units
    }
    model = learn_run.load_model("empirical", matrix)
    factory, _ = dial_choice.make_factory(
        "empirical", model, matrix, loaded, {}, configuration.kappa
    )
    forecaster = factory(loaded[UNIT], matrix)
    settings = gate_run.settings(matrix)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(gate_day, "audited_paths", _cut(gate_day.audited_paths))
        prepared = [
            gate_day.prepare_day(t, forecaster, configuration, settings) for t in loaded[UNIT].days
        ]
    assert all(len(p.paths) == PATHS for p in prepared)
    (asa,) = compare_methods.select(["asa"])
    slot = np.zeros((len(prepared), len(dial_splits.GRID)))
    for d, day in enumerate(prepared):
        for k, g in enumerate(dial_splits.GRID):
            slot[d, k] = compare_day.evaluate(day, g, (asa,))["asa"].slot_loss
    value = np.array([p.value for p in prepared])
    offers = np.array([not p.abstains for p in prepared])
    rows = _rows(folder)
    splits = dial_choice.split_indices(UNIT, loaded[UNIT], matrix, range(1, SPLITS + 1))
    found = []
    for (seed, calibration, test), row in zip(splits, rows, strict=True):
        assert int(row["seed"]) == seed
        scale = _calibrated(slot, [d for d in calibration if offers[d]])
        offer = scale * value[test].mean()
        assert float(row["scale"]) == scale
        assert float(row["offer_kwh_per_counting_test_day"]) == pytest.approx(offer, abs=1e-12)
        found.append(offer)
    document = json.loads((folder / compare_run.RESULTS_NAME).read_text())
    tier = document["units"][UNIT]["variants"]["forecast"]["methods"]["asa"]["B"]
    assert tier["mean_offer_kwh_per_test_day"] == pytest.approx(np.mean(found), abs=1e-12)
    assert np.mean(found) > 0
