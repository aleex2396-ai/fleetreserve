"""The learn run states per unit the days, the calibration days used and the test days, and the
matrix values of the calibration-days rule (plan section 6), on the made-up units of `learn_fake`
(5, 24 and 24 days) with `calibration_half_below` 30: every unit is under 30 days, so each is
split in halves (2 and 3, 12 and 12, 12 and 12), and the 5-day unit is split and abstains.
"""

import contextlib
import io
import json

import pytest
from gate_fake import COMMIT, GATE_UNITS, write_build
from learn_fake import PAIR, small_gate_units, small_matrix

from fleetreserve import gate_run, learn_eval, learn_run, run_verify

EXPECTED = {GATE_UNITS[0]: (5, 2, 3), GATE_UNITS[1]: (24, 12, 12), GATE_UNITS[2]: (24, 12, 12)}


@pytest.fixture(scope="module")
def learn(tmp_path_factory):
    folder = tmp_path_factory.mktemp("learn")
    run, tables = write_build(folder, small_gate_units())
    document = small_matrix()
    document["calibration_half_below"] = 30
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    out = folder / "out"
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = learn_run.main([*arguments, "--model", "empirical"])
    return code, out, printed.getvalue()


def test_the_learn_results_state_the_counts_and_the_rule(learn):
    code, out, text = learn
    assert code == 0
    results = json.loads((out / learn_run.RESULTS_NAME).read_text())
    assert results["calibration_rule"]["n_calibration"] == 20
    assert results["calibration_rule"]["calibration_half_below"] == 30
    for unit, (days, calibration, test) in EXPECTED.items():
        entry = results["units"][unit]
        assert (entry["counting_days"], entry["calibration_days_used"], entry["test_days"]) == (
            days,
            calibration,
            test,
        )
        assert entry["status"] == "calibratable"
    assert "n_calibration 20" in text and "calibration_half_below 30" in text
    for unit, (days, calibration, test) in EXPECTED.items():
        assert (
            f"{unit} ({days} counting days, split into {calibration} calibration days and "
            f"{test} test days):"
        ) in text
    assert "calibratable" not in text.replace("not calibratable", "")


def test_the_five_day_unit_is_split_and_offers_nothing(learn):
    _, out, _ = learn
    results = json.loads((out / learn_run.RESULTS_NAME).read_text())
    rungs = results["units"][GATE_UNITS[0]]["configurations"][PAIR[0]]["rungs"]
    for rung in learn_eval.OFFER_RUNGS:
        assert rungs[rung]["beta_hat"] == 0.0
        assert rungs[rung]["test"]["mean_offer_kwh_per_day"] == 0.0
