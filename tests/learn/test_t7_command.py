"""T7's measurement command (`t7_test`) on made-up training units and a made-up verified truth
build (`gate_fake`), with the dial family of the committed gate matrix at kappa 0.3, as the tests
of `size_whatif` use them. The two units scored, `caltech` and `OSL_S` (one of each site type),
have 140 days of three cars at a site named after the unit, `caltech` with an early leaver every
7th day (always a Wednesday), `OSL_S` every 9th; the other six training units, which only fit the
models, have 36 days.

1. Refusals before any table is opened, as `size_whatif`'s: a unit that is not a training unit, a
   model fitted on every training unit (`lgbm_tail`), a model that reads other days' outcomes
   (`empirical`), a dial the matrix does not have; each exit 2 and "COULD NOT RUN", nothing
   written. Without a verified truth build: exit 2.
2. `--model lgbm_tail_rotation` runs end to end with exit 0 and writes `t7.json`, `t7-summary.txt`
   and `t7-timing.json`; the summary's first line says what it is; every unit and tier has a
   verdict of the module's list.
3. The constant scale of each split equals `size_scale.calibrate`'s on the same prepared days and
   splits (`dial_choice.split_indices`), and its mean over the splits and the number of splits
   that offer equal those of `size_whatif`'s `constant` sizing run on the same build.
4. One, two and three workers write byte-identical `t7.json` and `t7-summary.txt`.
5. A failed internal check (here `size_scale.score` made to give other losses) ends with exit 1,
   "FAILED: internal check" and no results file.
6. A unit that fails (here `size_whatif.prepare_unit` made to raise for `caltech`) is recorded
   under `failed_units` and in the summary, the other unit is measured, the files are written, and
   the command exits 1.
"""

import json

import numpy as np
import pandas as pd
import pytest
from gate_fake import COMMIT, gate_cars, matrix_document, write_build
from test_dial_choice_lgbm import NAMES
from truth_tables_fake import date_of

from fleetreserve import (
    dial_choice,
    gate_run,
    learn_features,
    run_verify,
    size_scale,
    size_whatif,
    t7_stat,
    t7_summary,
    t7_test,
    t7_unit,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import load
from fleetreserve.truth_build import TRAIN_UNITS

SPLITS = 3
SCORED = ("caltech", "OSL_S")
EARLY = {"caltech": 7, "OSL_S": 9}


def document() -> dict:
    found = matrix_document(NAMES)
    found["n_calibration"] = 60
    return found


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        unit: gate_cars(140, unit, early_every=EARLY[unit])
        if unit in SCORED
        else gate_cars(36, unit, early_every=0 if j % 2 == 0 else 9 + j)
        for j, unit in enumerate(TRAIN_UNITS)
    }
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document()))
    return {"run": run, "tables": tables, "matrix": matrix}


def arguments(build, *extra, model="lgbm_tail_rotation", dial="0.03", units=SCORED):
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
        "--units",
        *units,
        *extra,
    ]


def run_command(argv, folder, monkeypatch, capsys):
    folder.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
    code = t7_test.main(argv)
    return code, capsys.readouterr().out


# --- 1. refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"units": ("nrel-from-may-2019",)}, "are not training units"),
        ({"units": ("caltech", "OSL_T")}, "are not training units"),
        ({"model": "lgbm_tail"}, "which the matrix scores"),
        ({"model": "empirical"}, "reads other days' outcomes"),
        ({"dial": "0.02"}, "no dial configuration at 0.02"),
    ],
)
def test_refusals_come_before_any_table_is_opened(
    build, tmp_path, monkeypatch, capsys, change, reason
):
    missing = tmp_path / "no-such-folder"
    argv = arguments(build, **change)
    argv[argv.index("--tables") + 1] = str(missing)
    argv[argv.index("--truth-run") + 1] = str(missing)
    code, text = run_command(argv, tmp_path / "out", monkeypatch, capsys)
    assert code == 2 and text.startswith("COULD NOT RUN") and reason in text, text
    assert list((tmp_path / "out").iterdir()) == []


def test_without_a_verified_truth_build_it_could_not_run(build, tmp_path, monkeypatch, capsys):
    code, text = run_command(arguments(build), tmp_path / "out", monkeypatch, capsys)
    assert code == 2 and text.startswith("COULD NOT RUN"), text
    assert list((tmp_path / "out").iterdir()) == []


# --- 2. to 4. end to end -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    out = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        patch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        for workers in (1, 2, 3):
            folder = tmp_path_factory.mktemp(f"workers{workers}")
            patch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
            code = t7_test.main(arguments(build, "--workers", str(workers)))
            out[workers] = (code, folder)
        folder = tmp_path_factory.mktemp("size-whatif")
        patch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        argv = arguments(build)
        out["size_whatif"] = (size_whatif.main(argv), folder)
    return out


def files(folder) -> dict:
    return {
        name: (folder / name).read_bytes() for name in (t7_test.RESULTS_NAME, t7_test.SUMMARY_NAME)
    }


def results(runs, key=1) -> dict:
    return json.loads((runs[key][1] / t7_test.RESULTS_NAME).read_text())


def test_the_command_runs_and_writes_its_files(runs):
    code, folder = runs[1]
    assert code == 0
    found = results(runs)
    assert (folder / t7_test.TIMING_NAME).is_file()
    assert found["failed_units"] == [] and sorted(found["units"]) == sorted(SCORED)
    summary = (folder / t7_test.SUMMARY_NAME).read_text().splitlines()
    assert summary[0] == t7_summary.WHAT
    # the summary keeps the order of --units: four heading lines, the units, one closing line
    heads = [line.split(" (")[0] for line in summary if not line.startswith(" ")]
    assert heads[4:-1] == list(SCORED) and summary[-1].startswith("UNDER THE VERDICTS: ")
    verdicts = {
        t7_unit.UNEQUAL,
        t7_unit.NOT_SHOWN,
        t7_unit.NOT_SPLIT,
        t7_unit.NO_OFFER,
        t7_unit.ONE_TYPE,
        t7_unit.FLAT,
    }
    for unit, entry in found["units"].items():
        assert entry["fit_units"] == [u for u in TRAIN_UNITS if u != unit]
        assert entry["counting_days"] == 140
        assert entry["offering_days"] + sum(entry["days_not_offering_by_reason"].values()) == 140
        assert len(entry["splits"]) == SPLITS
        for tier in size_scale.TIERS:
            verdict = entry["tiers"][tier]["verdict"]
            assert verdict in verdicts
            assert any(line.strip().startswith(f"tier {tier}: {verdict}") for line in summary)
    assert found["run"]["rounds"] == 10_000 and found["run"]["units"] == list(SCORED)


def test_day_types_read_the_dates_and_public_holidays(runs):
    found = results(runs)
    for unit in SCORED:
        dates = [date_of(k) for k in range(140)]
        frame = pd.DataFrame({"site": [unit] * 140, "date": dates})
        flags = learn_features.context_columns(frame)["public_holiday"].to_numpy(bool)
        raw = [t7_stat.raw_type(d, bool(f)) for d, f in zip(dates, flags, strict=True)]
        entry = found["units"][unit]
        assert sum(entry["raw_day_types"].values()) == entry["offering_days"]
        assert entry["raw_day_types"][t7_stat.HOLIDAY] >= 3
        if unit == "caltech":  # every caltech day offers, so the counts are those of all days
            assert entry["offering_days"] == 140
            assert entry["raw_day_types"] == t7_stat.type_counts(raw)
            assert entry["day_types"] == t7_stat.type_counts(t7_stat.pool_types(raw))
        else:  # the offering days are some of the days: no type has more days than all days
            counts = t7_stat.type_counts(raw)
            assert all(n <= counts[t] for t, n in entry["raw_day_types"].items())


def test_the_planted_weekday_of_caltech_has_the_largest_marginal_loss(runs):
    # every 7th day (a Wednesday, or 1 January) has an early leaver; about 1 in 7 calibration days
    # breaking at every scale above 0 is more than the slot limit allows, so no split offers
    entry = results(runs)["units"]["caltech"]
    for tier in size_scale.TIERS:
        reading = entry["tiers"][tier]
        assert reading["verdict"] == t7_unit.NO_OFFER and reading["splits_scalar_zero"] == SPLITS
        types = reading["pooled"]["slot"]["0.1"]["types"]
        assert max(types, key=lambda t: types[t]["marginal_per_kwh"]) == "Wednesday"


def load_days(build, unit, matrix):
    return load_unit_days(
        build["tables"], unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )


def test_the_constant_scale_is_that_of_size_whatif(runs, build):
    found = results(runs)
    assert runs["size_whatif"][0] == 0
    theirs = json.loads((runs["size_whatif"][1] / size_whatif.RESULTS_NAME).read_text())
    matrix = load(build["matrix"])
    configuration = size_whatif.find_configuration(matrix, 0.03)
    loaded = {u: load_days(build, u, matrix) for u in TRAIN_UNITS}
    for unit in SCORED:
        entry = found["units"][unit]
        for tier in size_scale.TIERS:
            used = [row[tier]["used"] for row in entry["splits"]]
            constant = theirs["units"][unit]["tiers"][tier]["constant"]
            assert float(np.mean(used)) == constant["mean_scalar_used"]
            assert sum(u > 0 for u in used) == constant["splits_that_offer"]
        model, fit_units = size_whatif.check_models("lgbm_tail_rotation", matrix, (unit,))[unit]
        prepared = size_whatif.prepare_unit(
            unit, model, fit_units, matrix, configuration, {unit: loaded[unit]}, loaded, 1
        )
        terms = [size_scale.day_terms(p) for p in prepared]
        q = t7_unit.constant_sizes(terms)
        splits = dial_choice.split_indices(unit, loaded[unit], matrix, range(1, SPLITS + 1))
        for row, (seed, calibration, _) in zip(entry["splits"], splits, strict=True):
            scalars = size_scale.calibrate(terms, q, calibration)
            assert row["seed"] == seed
            assert all(row[t]["used"] == scalars[t].used for t in size_scale.TIERS)
    assert any(row["B"]["used"] > 0 for e in found["units"].values() for row in e["splits"])


def test_one_two_and_three_workers_write_the_same_bytes(runs):
    assert runs[1][0] == runs[2][0] == runs[3][0] == 0
    assert files(runs[1][1]) == files(runs[2][1]) == files(runs[3][1])


# --- 5. the internal check -----------------------------------------------------------------------


def test_a_failed_internal_check_exits_1(build, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    real = size_scale.score

    def shifted(terms, q, gamma, test):
        found = real(terms, q, gamma, test)
        driver = tuple(x + 0.25 for x in found.driver_losses)
        return size_scale.TestScore(
            found.offers_kwh, found.offering_days, found.slot_losses, driver
        )

    monkeypatch.setattr(size_scale, "score", shifted)
    code, text = run_command(
        arguments(build, units=("OSL_S",)), tmp_path / "out", monkeypatch, capsys
    )
    assert code == 1 and "FAILED: internal check" in text and "driver terms" in text, text
    assert list((tmp_path / "out").iterdir()) == []


# --- 6. a failed unit ----------------------------------------------------------------------------


def test_a_failed_unit_is_recorded_and_exits_1(build, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    real = size_whatif.prepare_unit

    def failing(unit, *rest):
        if unit == "caltech":
            raise ValueError("made to fail")
        return real(unit, *rest)

    monkeypatch.setattr(size_whatif, "prepare_unit", failing)
    folder = tmp_path / "out"
    code, text = run_command(arguments(build), folder, monkeypatch, capsys)
    assert code == 1, text
    found = json.loads((folder / t7_test.RESULTS_NAME).read_text())
    assert found["failed_units"] == ["caltech: ValueError: made to fail"]
    assert list(found["units"]) == ["OSL_S"]
    summary = (folder / t7_test.SUMMARY_NAME).read_text().splitlines()
    assert "FAILED UNITS: caltech: ValueError: made to fail" in summary
    assert (folder / t7_test.TIMING_NAME).is_file()
