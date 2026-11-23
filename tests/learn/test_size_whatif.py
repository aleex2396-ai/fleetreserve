"""The sizing what-if command (`size_whatif`) on eight made-up training units
(`gate_fake.gate_cars`, 36 days of three cars each at a site named after the unit; every other
unit has an early leaver on one day in 10 to 16, the others none) and a made-up verified truth
build, with the dial family of the committed gate matrix at kappa 0.3.

1. Refusals before any table is opened (the tables folder given does not exist, so opening one
   would end otherwise): a unit that is not a training unit, a model fitted on every training unit
   (`lgbm_tail`), a model that reads other days' outcomes (`empirical`), a dial the matrix does not
   have; each ends with exit 2, "COULD NOT RUN" and its own reason (the missing tables would end
   the command with exit 2 as well). Without a verified truth build: exit 2.
2. `--model lgbm_tail_rotation` runs end to end with exit 0 and writes the day table, the results
   and the summary: one row per labelled day, each unit's fit units without the unit, labels in
   the table equal to `size_label` of the prepared days, counts that add up, both tiers of every
   sizing, oracle lines that say what they are, and the summary's first line saying it is a what-if.
3. With `constant` the calibrated offer of each split equals the every-request route of
   `gate_calibrate` on that split (capped at 1), for every unit and tier.
4. Two runs (one worker and two workers) write byte-identical day tables, results and summaries.
"""

import csv
import io
import json

import numpy as np
import pytest
from gate_fake import COMMIT, gate_cars, matrix_document, write_build
from test_dial_choice_lgbm import NAMES

from fleetreserve import (
    dial_choice,
    gate_calibrate,
    gate_run,
    run_verify,
    size_gbm,
    size_label,
    size_scale,
    size_whatif,
    size_whatif_summary,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.truth_build import TRAIN_UNITS

SPLITS = 3
DAYS = 36


def document() -> dict:
    found = matrix_document(NAMES)
    found["n_calibration"] = 25
    return found


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        unit: gate_cars(DAYS, unit, early_every=0 if j % 2 == 0 else 9 + j)
        for j, unit in enumerate(TRAIN_UNITS)
    }
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document()))
    return {"run": run, "tables": tables, "matrix": matrix}


def arguments(build, *extra, model="lgbm_tail_rotation", dial="0.03"):
    return [
        "--model",
        model,
        "--dial",
        dial,
        "--matrix",
        str(build["matrix"]),
        "--tables",
        str(build["tables"]),
        "--truth-run",
        str(build["run"]),
        "--splits",
        str(SPLITS),
        *extra,
    ]


@pytest.fixture
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


def run_command(argv, folder, monkeypatch, capsys):
    folder.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
    code = size_whatif.main(argv)
    return code, capsys.readouterr().out


# --- 1. refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"extra": ("--units", "nrel-from-may-2019")}, "are not training units"),
        ({"extra": ("--units", "caltech", "OSL_T")}, "are not training units"),
        ({"model": "lgbm_tail"}, "which the matrix scores"),
        ({"model": "empirical"}, "reads other days' outcomes"),
        ({"dial": "0.02"}, "no dial configuration at 0.02"),
    ],
)
def test_refusals_come_before_any_table_is_opened(
    build, tmp_path, monkeypatch, capsys, change, reason
):
    missing = tmp_path / "no-such-folder"
    argv = arguments(
        build, *change.get("extra", ()), **{k: v for k, v in change.items() if k != "extra"}
    )
    argv[argv.index("--tables") + 1] = str(missing)
    argv[argv.index("--truth-run") + 1] = str(missing)
    code, text = run_command(argv, tmp_path / "out", monkeypatch, capsys)
    assert code == 2 and text.startswith("COULD NOT RUN") and reason in text, text
    assert list((tmp_path / "out").iterdir()) == []


def test_without_a_verified_truth_build_it_could_not_run(build, tmp_path, monkeypatch, capsys):
    code, text = run_command(arguments(build), tmp_path / "out", monkeypatch, capsys)
    assert code == 2 and text.startswith("COULD NOT RUN"), text


# --- 2. to 4. end to end -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    out = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        patch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        for workers in (1, 2):
            folder = tmp_path_factory.mktemp(f"workers{workers}")
            patch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
            code = size_whatif.main(arguments(build, "--workers", str(workers)))
            out[workers] = (code, folder)
    return out


def files(folder) -> dict:
    names = (size_whatif.DAYS_NAME, size_whatif.RESULTS_NAME, size_whatif.SUMMARY_NAME)
    return {name: (folder / name).read_bytes() for name in names}


def test_the_command_runs_and_writes_its_files(runs, build):
    code, folder = runs[1]
    assert code == 0
    written = files(folder)
    results = json.loads(written[size_whatif.RESULTS_NAME])
    rows = list(csv.DictReader(io.StringIO(written[size_whatif.DAYS_NAME].decode())))
    assert list(rows[0]) == list(size_whatif.DAY_COLUMNS)
    assert results["failed_units"] == [] and set(results["units"]) == set(TRAIN_UNITS)
    total = 0
    for unit, entry in results["units"].items():
        assert entry["fit_units"] == [u for u in TRAIN_UNITS if u != unit]
        labels = entry["labels"]
        assert labels["offering_days"] + labels["days_not_offering"] == labels["counting_days"]
        assert labels["counting_days"] == DAYS
        total += labels["offering_days"]
        assert sum(r["unit"] == unit for r in rows) == labels["offering_days"]
        for tier in size_scale.TIERS:
            names = set(entry["tiers"][tier])
            assert {"constant", "oracle-scale", "oracle-abstain"} <= names
            assert {
                f"gbm-{level:g}-{arm}" for level in (0.05, 0.5) for arm in size_scale.ARMS
            } <= names
            assert entry["tiers"][tier]["constant"]["splits"] == SPLITS
    assert len(rows) == total
    summary = written[size_whatif.SUMMARY_NAME].decode().splitlines()
    assert summary[0] == size_whatif_summary.WHAT
    oracle = [line for line in summary if line.strip().startswith("oracle-")]
    assert oracle and all(size_whatif_summary.ORACLE_NOTE in line for line in oracle)
    assert set(results["pooled_by_site_type"]) == {"workplace", "other"}


def test_the_labels_in_the_table_are_those_of_the_prepared_days(runs, build):
    _, folder = runs[1]
    rows = list(csv.DictReader(io.StringIO((folder / size_whatif.DAYS_NAME).read_text())))
    matrix = gate_config_load(build["matrix"])
    configuration = size_whatif.find_configuration(matrix, 0.03)
    unit = "jpl"
    models = size_whatif.check_models("lgbm_tail_rotation", matrix, (unit,))
    loaded = {u: load_days(build, u, matrix) for u in TRAIN_UNITS}
    model, fit_units = models[unit]
    prepared = size_whatif.prepare_unit(
        unit, model, fit_units, matrix, configuration, {unit: loaded[unit]}, loaded, 1
    )
    labels = [(p.date, size_label.day_label(p)) for p in prepared if not p.abstains]
    expected = [(d, repr(x.scale), repr(x.clipped), repr(x.fixed_plan_scale)) for d, x in labels]
    found = [
        (r["date"], r["label_scale"], r[size_gbm.LABEL], r["fixed_plan_scale"])
        for r in rows
        if r["unit"] == unit
    ]
    assert found == expected and len(found) > 10


def gate_config_load(path):
    from fleetreserve.gate_config import load

    return load(path)


def load_days(build, unit, matrix):
    return load_unit_days(
        build["tables"], unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )


def test_constant_is_the_every_request_route_on_every_split(runs, build):
    _, folder = runs[1]
    results = json.loads((folder / size_whatif.RESULTS_NAME).read_text())
    matrix = gate_config_load(build["matrix"])
    configuration = size_whatif.find_configuration(matrix, 0.03)
    unit = "caltech"
    models = size_whatif.check_models("lgbm_tail_rotation", matrix, (unit,))
    loaded = {u: load_days(build, u, matrix) for u in TRAIN_UNITS}
    model, fit_units = models[unit]
    prepared = size_whatif.prepare_unit(
        unit, model, fit_units, matrix, configuration, {unit: loaded[unit]}, loaded, 1
    )
    splits = dial_choice.split_indices(unit, loaded[unit], matrix, range(1, SPLITS + 1))
    offers = {"A": [], "B": []}
    for _, calibration, test in splits:
        route = gate_calibrate.every_request(prepared, calibration, test, tier_b=True)
        for tier, scale in (("A", route["beta_used"]), ("B", route["tier_b"]["beta_used"])):
            used = min(scale, 1.0)
            offers[tier].append(float(np.mean([used * prepared[d].value for d in test])))
    for tier in size_scale.TIERS:
        found = results["units"][unit]["tiers"][tier]["constant"]
        assert found["mean_offer_kwh_per_counting_test_day"] == pytest.approx(
            np.mean(offers[tier]), abs=1e-12
        )
    assert any(o > 0 for o in offers["B"])


def test_one_and_two_workers_write_the_same_bytes(runs):
    assert runs[1][0] == runs[2][0] == 0
    assert files(runs[1][1]) == files(runs[2][1])
