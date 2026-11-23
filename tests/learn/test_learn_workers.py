"""The learn steps of the run list at 6 workers (`run-lists/run-list.json`, `learn_run --workers`).

1. Every `learn_run` step of the run list gives `--workers 6`, once, in the command both purposes
   run, and no purpose's extra arguments give `--workers`. (Steps of other commands, such as
   outcome O2's `ceiling_report --workers 4`, are not this test's.)
2. `learn_run` writes the same bytes at 1 and at 6 workers: on made-up units (the three gate
   units at sites of the committed site mappings, with no early departures so that the forecast
   configurations, not only their oracle partners, offer on test days; eight made-up training
   units with early departures), with each committed rehearsal matrix those steps read
   (`n_calibration` 20 and no half-the-days rule, as `learn_fake.small_matrix` shortens the
   learn matrix) and the run list's own interval seed and resamples (`--interval-seed
   {seed:intervals} --interval-resamples {count:resamples}`), every output file but the timing
   file, and the summary printed, are byte-identical; the timing file names 1 and 6 workers.
   `hazard_tuned_inputs` reads made-up context sources (as `test_hazard_tuned_inputs` makes
   them) and the small tuned settings of `hazard_fake`. The held setting's step differs from the
   main one only in the truth tables it is given, and rho 0.2 from rho 0.1 only in its matrix's
   rho, neither of which reaches the pool's split of the work.

Why the result cannot depend on the count (`gate_run`, module text: "The results files do not
depend on N"): the model is fitted and its forecast table predicted once, in the calling process,
before any pool starts; the pool (`gate_run.Runner`) gets that table (`TableFactory`) and runs
only the per-day ceilings, preparations and evaluations, each a function of its (unit, day,
configuration) or (prepared day, scale) alone; `ProcessPoolExecutor.map` and
`ThreadPoolExecutor.map` return results in the order of their inputs; the split seed is the
matrix's, per unit; the resample draws are `day_bootstrap.draws(seed, unit, ...)`, per unit and
drawn in the calling process; the audited paths are seeded from the unit and the date.
"""

import json
from pathlib import Path

import hazard_fake as hf
import numpy as np
import pandas as pd
import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_cars, write_build

from fleetreserve import (
    context_places,
    gate_run,
    hazard_inputs,
    learn_run,
    model_inputs,
    run_verify,
    school_inputs,
)
from fleetreserve.truth_build import TRAIN_UNITS

TOP = Path(__file__).resolve().parents[2]
LIST = json.loads((TOP / "run-lists" / "run-list.json").read_text())
WORKERS = 6
SEED = LIST["fixed"]["seeds"]["intervals"]
RESAMPLES = LIST["fixed"]["counts"]["resamples"]
RUNS = (  # (model, committed rehearsal matrix)
    ("lgbm_tail", "lgbm_tail.json"),
    ("lgbm_tail", "lgbm_tail-forecast-energy.json"),
    ("lgbm_tail", "lgbm_tail-rho0.1.json"),
    ("hazard_tuned_inputs", "hazard_tuned_inputs.json"),
    ("earlier_population", "earlier_population.json"),
    ("earlier_gbm", "earlier_gbm.json"),
    ("km", "lgbm_tail.json"),  # the fixed-plan steps read the main matrix (run list)
    ("lgbm", "lgbm_tail.json"),
    ("cox", "lgbm_tail.json"),
)


def _learn_steps() -> dict:
    found = {}
    for entry in LIST["entries"]:
        command = entry["command"]
        if command[command.index("-m") + 1] == "fleetreserve.learn_run":
            found[entry["label"]] = (command[command.index("--model") + 1], entry)
    return found


def test_every_learn_step_gives_six_workers():
    steps = _learn_steps()
    assert steps
    for label, (_, entry) in steps.items():
        given = [a for a in entry["command"] if a.startswith("--work")]
        for extra in entry["extra"].values():
            assert not [a for a in extra if a.startswith("--work")], label
        at = entry["command"].index("--workers")
        assert given == ["--workers"] and entry["command"][at + 1] == str(WORKERS), label


def _units() -> dict:
    units = {
        GATE_UNITS[0]: gate_cars(5, "nrel"),
        GATE_UNITS[1]: gate_cars(24, "nrel", early_every=0),
        GATE_UNITS[2]: gate_cars(12, "OSL_1", early_every=0) + gate_cars(12, "BER", early_every=0),
    }
    for j, unit in enumerate(TRAIN_UNITS):
        units[unit] = gate_cars(24, unit, early_every=3 + j)
    return units


def _context() -> model_inputs.Context:
    """Made-up context sources of every mapped site, as `test_hazard_tuned_inputs` makes them."""
    sites = sorted(context_places.SITE_PLACES)
    places = {p for site in sites for p in context_places.SITE_PLACES[site]}
    hours = pd.date_range("2019-12-01", periods=24 * 90, freq="h", tz="UTC")
    k = np.arange(len(hours), dtype=float)
    weather = pd.DataFrame(
        {"temperature": k % 23, "precipitation": k % 3, "snowfall": (k % 7 == 0) * 1.0},
        index=pd.DatetimeIndex(hours, name="time"),
    )
    days = pd.Index(pd.date_range("2020-01-01", periods=120).date, name="date")
    sources = hazard_inputs.ContextSources(
        weather=dict.fromkeys(places, weather),
        oxcgrt={
            j: pd.Series(np.arange(120.0), index=days)
            for j in {context_places.SITE_OXCGRT[s] for s in sites}
        },
        nine_eighty_parity=0,
        school=school_inputs.load_terms(),
    )
    return model_inputs.Context(sources, {"made/up.csv": "0" * 64})


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, _units())
    return ["--tables", str(tables), "--truth-run", str(run)]


def _matrix(folder: Path, name: str) -> Path:
    document = json.loads((TOP / "gate" / "rehearsal-matrix" / name).read_text())
    document["n_calibration"] = 20
    document["calibration_half_below"] = None
    path = folder / name
    path.write_text(json.dumps(document))
    return path


def _run(arguments, folder: Path, workers: int, capsys) -> tuple[int, str]:
    folder.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setattr(model_inputs, "load_context", _context)
        hf.use_small_tuned(monkeypatch, folder.parent)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        code = learn_run.main([*arguments, "--workers", str(workers)])
    return code, capsys.readouterr().out


@pytest.mark.parametrize(("model", "matrix"), RUNS)
def test_one_and_six_workers_write_the_same_bytes(build, tmp_path, capsys, model, matrix):
    arguments = [
        *build,
        "--matrix",
        str(_matrix(tmp_path, matrix)),
        "--model",
        model,
        "--interval-seed",
        str(SEED),
        "--interval-resamples",
        str(RESAMPLES),
    ]
    one, six = tmp_path / "one", tmp_path / "six"
    code_one, printed_one = _run(arguments, one, 1, capsys)
    code_six, printed_six = _run(arguments, six, WORKERS, capsys)
    assert code_one == 0, printed_one
    assert code_six == 0, printed_six
    assert printed_six == printed_one
    names = sorted(p.name for p in one.iterdir())
    assert names == sorted(p.name for p in six.iterdir())
    assert learn_run.TIMING_NAME in names and learn_run.DAY_LOSSES_NAME in names
    for name in names:
        if name != learn_run.TIMING_NAME:
            assert (six / name).read_bytes() == (one / name).read_bytes(), name
    for folder, workers in ((one, 1), (six, WORKERS)):
        timing = json.loads((folder / learn_run.TIMING_NAME).read_text())
        assert timing["gate"]["workers"] == workers
        assert timing["gate"]["interval_pairs"], "no resample pairs were evaluated"
    results = json.loads((one / learn_run.RESULTS_NAME).read_text())
    assert results["model"] == model and not results["failed_configurations"]
    offers = [
        rung["test"]["mean_offer_kwh_per_day"]
        for unit in results["units"].values()
        for found in unit["configurations"].values()
        for rung in found["rungs"].values()  # a configuration's own rungs, not its partner's
        if isinstance(rung, dict) and (rung.get("test") or {}).get("mean_offer_kwh_per_day")
    ]
    assert offers, "no forecast configuration offered on a test day: the comparison is too weak"
