"""The WP8 evaluation command (`learn_run`) and the forecaster factory of the gate run, on the
made-up gate units of `learn_fake` with a made-up verified truth build.

1. `learn_run --model empirical` ends with exit 0 and writes every output; its gate-format results
   equal, configuration by configuration, those `gate_run` writes for the same matrix, and the
   offers in `learn.json` are those numbers, for the configuration and its oracle partner.
2. With 2 workers the results are the same (the table factory reaches the worker processes).
3. The factory hook is used: a factory whose forecaster forecasts nothing makes every day abstain.
4. A model name with no module, a module without `make_model`, and a name that is not a Python
   name could not run (2). The failures and refusals of other models are in `test_learn_guard`.
"""

import json
import sys
import types

import pytest
from gate_fake import COMMIT, GATE_UNITS, write_build
from learn_fake import PAIR, small_gate_units, small_matrix

from fleetreserve import forecast_model, gate_run, learn_eval, learn_run, run_verify
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import parse


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, small_gate_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    return run, tables, matrix


def _arguments(build) -> list[str]:
    run, tables, matrix = build
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]


def _run(command, build, out, *extra) -> int:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return command([*_arguments(build), *extra])


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    folder = tmp_path_factory.mktemp("runs")
    gate = _run(gate_run.main, build, folder / "gate")
    learn = _run(learn_run.main, build, folder / "learn", "--model", "empirical")
    return (gate, folder / "gate"), (learn, folder / "learn")


def _read(path):
    return json.loads(path.read_text())


def test_the_empirical_model_reproduces_the_gate(runs):
    (gate_code, gate_out), (learn_code, learn_out) = runs
    assert gate_code == 0 and learn_code == 0
    for name in (
        learn_run.RESULTS_NAME,
        learn_run.TABLE_NAME,
        learn_run.GATE_NAME,
        learn_run.TIMING_NAME,
    ):
        assert (learn_out / name).is_file()
    gate = _read(gate_out / gate_run.RESULTS_NAME)
    learned = _read(learn_out / learn_run.GATE_NAME)
    assert learned["configurations"] == gate["configurations"]
    assert learned["gate"] == gate["gate"] and learned["t5"] == gate["t5"]
    results = _read(learn_out / learn_run.RESULTS_NAME)
    assert results["model"] == "empirical" and results["failed_configurations"] == []
    offered = 0
    for unit in GATE_UNITS:
        entry = results["units"][unit]
        assert set(entry["kappas"]) == {"0.1"}
        assert entry["kappas"]["0.1"]["pinball_loss"]["departure_hours"]["mean"] is not None
        assert set(entry["configurations"]) == {PAIR[0]}
        found = entry["configurations"][PAIR[0]]
        assert found["oracle_partner"]["name"] == PAIR[1]
        for name, rungs in ((PAIR[0], found["rungs"]), (PAIR[1], found["oracle_partner"]["rungs"])):
            source = gate["configurations"][name]["units"][unit]
            if "rungs" not in source:
                assert rungs == {"reason": source["reason"]}
                continue
            for rung in learn_eval.OFFER_RUNGS:
                assert rungs[rung]["test"] == source["rungs"][rung]["test"]
                assert rungs[rung]["beta_hat"] == source["rungs"][rung]["beta_hat"]
                offered += rungs[rung]["test"]["mean_offer_kwh_per_day"] > 0
    assert offered > 0  # the comparison is not between two empty offers
    assert results["units"][GATE_UNITS[0]]["status"] == "not calibratable: 5 days"
    table = forecast_model.read_table(learn_out / learn_run.TABLE_NAME)
    assert results["forecast_table"]["rows"] == len(table) > 0


def test_learn_json_and_its_summary_carry_the_offering_and_failing_days(runs):
    _, (_, learn_out) = runs
    results = _read(learn_out / learn_run.RESULTS_NAME)
    found = results["units"][GATE_UNITS[1]]["configurations"][PAIR[0]]
    for rungs in (found["rungs"], found["oracle_partner"]["rungs"]):
        for rung in learn_eval.OFFER_RUNGS:
            test, scale = rungs[rung]["test"], rungs[rung]["beta_hat"]
            assert 0 <= test["offering_days"] <= test["days"]
            for field in ("mean_slot_loss_offering_days", "mean_driver_loss_offering_days"):
                assert field in test
            assert "mean_slot_loss" not in test and "mean_driver_loss" not in test
            if scale > 0:
                assert 0 <= test["failing_days"] <= test["offering_days"]
            else:  # nothing is offered: no loss to average, and the reason is carried
                assert test["failing_days"] is None
                assert test["mean_slot_loss_offering_days"] is None
                assert rungs[rung]["reason"].startswith(("abstains: ", "no scale is safe: "))
    text = "\n".join(learn_run.summary_lines(results))
    for rung in learn_eval.OFFER_RUNGS:
        test, scale = found["rungs"][rung]["test"], found["rungs"][rung]["beta_hat"]
        boxes = f"{test['offering_days']} of {test['days']} days have a box"
        if scale > 0:
            assert f"{boxes}, {test['failing_days']} fail" in text
        else:
            assert f"scale 0, {found['rungs'][rung]['reason']}; {boxes}, no offer made" in text


def test_the_day_rows_are_written_as_the_gate_run_writes_them(runs):
    # The per-day numbers (certified value, both ceilings, each rung's offer and losses per
    # test day) that `gate_run` writes to gate-days.csv; the ledger of plan T9 reads them.
    (_, gate_out), (_, learn_out) = runs
    written = learn_out / "learn-gate-days.csv"
    assert written.is_file()
    assert written.read_bytes() == (gate_out / gate_run.DAYS_NAME).read_bytes()
    lines = written.read_text().splitlines()
    assert lines[0].split(",") == list(gate_run.DAY_COLUMNS) and len(lines) > 1


def test_the_learn_summary_tells_a_scale_of_zero_from_a_calibrated_scale():
    test = {
        "days": 192,
        "offering_days": 27,
        "failing_days": 1,
        "mean_offer_kwh_per_day": 0.81,
    }
    shown = learn_run._offer(
        {
            "fixed": {"beta_hat": 0.3, "reason": "", "test": test},
            "one-way": {
                "beta_hat": 0.0,
                "reason": "abstains: 11 of 50 calibration days offer; needs 19",
                "test": dict(test, failing_days=None, mean_offer_kwh_per_day=0.0),
            },
        }
    )
    assert "scale 0.3, 27 of 192 days have a box, 1 fail" in shown
    assert (
        "scale 0, abstains: 11 of 50 calibration days offer; needs 19; 27 of 192 days have a box, "
        "no offer made" in shown
    )


def test_the_results_do_not_depend_on_the_number_of_workers(build, runs, tmp_path):
    _, (_, one) = runs
    out = tmp_path / "two"
    assert _run(learn_run.main, build, out, "--model", "empirical", "--workers", "2") == 0
    for name in (learn_run.RESULTS_NAME, learn_run.TABLE_NAME, learn_run.GATE_NAME):
        assert (out / name).read_bytes() == (one / name).read_bytes()


class _Nothing:
    def forecast_car(self, *args):
        return None

    def survival_bound(self, *args):
        return None


def _nothing(unit_days, matrix):
    return _Nothing()


def test_the_gate_uses_the_factory_it_is_given(build):
    _, tables, matrix_file = build
    matrix = parse(json.loads(matrix_file.read_text()))
    units = {
        name: load_unit_days(tables, name, rho=0.0, no_fit="cap", small_fleet="keep")
        for name in matrix.gate_units
    }
    (_, rows, _), _ = gate_run.run_gate(
        matrix, units, workers=1, only=PAIR[0], trial=False, factory=_nothing
    )
    assert {r["abstain_reason"] for r in rows} == {"no steered car"}
    (_, plain_rows, _), _ = gate_run.run_gate(matrix, units, workers=1, only=PAIR[0], trial=False)
    assert {r["abstain_reason"] for r in plain_rows} != {"no steered car"}


def _module(monkeypatch, name, **attributes):
    module = types.ModuleType(f"fleetreserve.models.{name}")
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_a_bad_model_could_not_run(build, tmp_path, monkeypatch, capsys):
    base = _arguments(build)
    assert learn_run.main([*base, "--model", "no_such_model"]) == 2
    _module(monkeypatch, "without_maker")
    assert learn_run.main([*base, "--model", "without_maker"]) == 2
    assert learn_run.main([*base, "--model", "../empirical"]) == 2
    assert learn_run.main([*base, "--model", "empirical", "--workers", "0"]) == 2
    assert capsys.readouterr().out.count("COULD NOT RUN") == 4
