"""The number of calibration days of a unit is a rule of its number of days (plan section 6).

"Per held-out unit, 100 seeded random days calibrate, or half of the unit's days (rounded down)
when it has fewer than 200, and the rest test, one draw for all policies."

1. The rule (`day_split.calibration_count`, `day_split.is_split`) at 1, 2, 3, 199, 200, 201 and 334
   days with the committed values; the null threshold means `n_calibration` for every unit.
2. The matrix field `calibration_half_below`: required, null or a whole number of at least 1; the
   committed matrices hold 100 and 200.
3. `gate_run.plan_unit` on made-up units: 80 days split 40 and 40, 242 days split 100 and 142, a
   9-day unit 4 and 5, a 1-day unit is not split and still reported "not calibratable".
4. The gate run, the dial choice (and, in `tests/wp8`, the learn run) state per unit the number of
   days, the calibration days used and the test days, and the matrix values of both fields; a
   9-day unit is split and its calibration abstains, and no output calls it calibratable.
"""

import copy
import json
from pathlib import Path

import pytest
from gate_fake import (
    COMMIT,
    COMMITTED,
    GATE_UNITS,
    TOP,
    gate_cars,
    matrix_document,
    write_build,
    write_tables,
)

from fleetreserve import day_split, dial_choice, gate_config, gate_run, run_verify
from fleetreserve.day_split import calibration_count, is_split
from fleetreserve.days_from_truth import load_unit_days

PAIR = ("main-rounded-hour-k0.3-empirical", "main-rounded-hour-k0.3-oracle")
LOW = ("main-rounded-hour-k0.3-eps0.03-empirical", "main-rounded-hour-k0.3-eps0.03-oracle")
N_CAL, BELOW = COMMITTED["n_calibration"], COMMITTED["calibration_half_below"]


def _committed_rule(names=PAIR) -> dict:
    """The made-up matrix of `gate_fake` with the plan's rule of the committed matrix."""
    document = matrix_document(names)
    document["n_calibration"] = N_CAL
    document["calibration_half_below"] = BELOW
    return document


# --- 1. the rule ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_days", "count", "split"),
    [
        (1, 0, False),
        (2, 1, True),
        (3, 1, True),
        (9, 4, True),
        (199, 99, True),
        (200, 100, True),
        (201, 100, True),
        (334, 100, True),
    ],
)
def test_the_rule_at_the_committed_values(n_days, count, split):
    assert (N_CAL, BELOW) == (100, 200)
    assert calibration_count(n_days, N_CAL, BELOW) == count
    assert is_split(n_days, count) is split


@pytest.mark.parametrize(("n_days", "count"), [(149, 74), (150, 100), (151, 100), (1, 0)])
def test_the_threshold_itself_gets_n_calibration_days_boundary(n_days, count):
    # At 200 days half the days is 100 as well, so the committed values cannot tell "at least the
    # threshold" from "more than the threshold"; another threshold can.
    assert calibration_count(n_days, 100, 150) == count


def test_the_rule_leaves_a_test_day_whenever_the_unit_is_split():
    for n_days in range(1, 400):
        count = calibration_count(n_days, 100, 200)
        if is_split(n_days, count):
            assert count >= 1 and n_days - count >= 1


@pytest.mark.parametrize(("n_days", "split"), [(1, False), (3, False), (100, False), (101, True)])
def test_a_null_threshold_gives_n_calibration_to_every_unit(n_days, split):
    count = calibration_count(n_days, 100, None)
    assert count == 100
    assert is_split(n_days, count) is split


def test_the_defaults_of_the_split_are_the_plans():
    assert (day_split.N_CALIBRATION, day_split.HALF_BELOW, day_split.MIN_DAYS) == (100, 200, 101)


# --- 2. the matrix field --------------------------------------------------------------------------


def _refused(document, text):
    with pytest.raises(gate_config.MatrixError, match=text):
        gate_config.parse(document)


def test_the_field_is_required():
    document = copy.deepcopy(COMMITTED)
    del document["calibration_half_below"]
    _refused(document, "missing fields.*calibration_half_below")


@pytest.mark.parametrize("value", [-1, 0, -200, 200.0, 199.5, "200", True, [200]])
def test_a_threshold_that_is_not_a_whole_number_of_at_least_one_is_refused(value):
    document = copy.deepcopy(COMMITTED)
    document["calibration_half_below"] = value
    _refused(document, "calibration_half_below must be a whole number")


def test_a_null_threshold_is_accepted_and_kept_as_none():
    document = copy.deepcopy(COMMITTED)
    document["calibration_half_below"] = None
    assert gate_config.parse(document).calibration_half_below is None
    document["calibration_half_below"] = 150
    assert gate_config.parse(document).calibration_half_below == 150


@pytest.mark.parametrize("folder", ["gate", "learn"])
def test_the_committed_matrices_hold_100_and_200(folder):
    document = json.loads((TOP / folder / "matrix.json").read_text())
    assert (document["n_calibration"], document["calibration_half_below"]) == (100, 200)
    matrix = gate_config.load(TOP / folder / "matrix.json")
    assert (matrix.n_calibration, matrix.calibration_half_below) == (100, 200)


# --- 3. the split of made-up units ----------------------------------------------------------------


@pytest.fixture(scope="module")
def units(tmp_path_factory):
    folder = tmp_path_factory.mktemp("units")
    found = {}
    for n_days in (1, 9, 80, 242):
        write_tables(folder, f"u{n_days}", gate_cars(n_days))
        found[n_days] = load_unit_days(
            folder, f"u{n_days}", rho=0.0, no_fit="cap", small_fleet="keep"
        )
    return found


@pytest.mark.parametrize(
    ("n_days", "calibration", "test"), [(80, 40, 40), (242, 100, 142), (9, 4, 5)]
)
def test_a_made_up_unit_is_split_by_the_rule(units, n_days, calibration, test):
    matrix = gate_config.parse(_committed_rule())
    plan = gate_run.plan_unit(f"u{n_days}", units[n_days], matrix)
    assert plan.status == "calibratable"
    assert (plan.calibration_days, plan.test_days) == (calibration, test)
    assert plan.roles.count("calibration") == calibration and plan.roles.count("test") == test
    assert len(plan.split["calibration"]) == calibration


def test_a_one_day_unit_is_not_split_and_says_not_calibratable(units):
    matrix = gate_config.parse(_committed_rule())
    plan = gate_run.plan_unit("u1", units[1], matrix)
    assert plan.status == "not calibratable: 1 days"
    assert (plan.calibration_days, plan.test_days) == (0, 0)
    assert set(plan.roles) == {"not calibratable"} and plan.split == {}


def test_the_old_rule_is_still_available_through_the_matrix(units):
    document = _committed_rule()
    document["n_calibration"], document["calibration_half_below"] = 50, None
    matrix = gate_config.parse(document)
    assert gate_run.plan_unit("u9", units[9], matrix).status == "not calibratable: 9 days"
    plan = gate_run.plan_unit("u80", units[80], matrix)
    assert (plan.calibration_days, plan.test_days) == (50, 30)


# --- 4. what the commands state -------------------------------------------------------------------

SIZES = (9, 80, 31)  # the first is split 4 and 5 and abstains; the others 40 and 40, 15 and 16


def _units() -> dict:
    return {
        GATE_UNITS[0]: gate_cars(SIZES[0]),
        GATE_UNITS[1]: gate_cars(SIZES[1]),
        GATE_UNITS[2]: gate_cars(15, "B") + gate_cars(16, "C", early_every=9),
    }


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, _units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(_committed_rule(PAIR + LOW)))
    return run, tables, matrix


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


def _run(command, build, out: Path, *extra) -> int:
    run, tables, matrix = build
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return command([*arguments, *extra])


EXPECTED = {
    GATE_UNITS[0]: (9, 4, 5),
    GATE_UNITS[1]: (80, 40, 40),
    GATE_UNITS[2]: (31, 15, 16),
}


@pytest.fixture(scope="module")
def gate(build, tmp_path_factory):
    out = tmp_path_factory.mktemp("gate") / "out"
    code = _run(gate_run.main, build, out, "--only", PAIR[0])
    return code, out


def test_the_gate_run_states_the_counts_and_the_rule(gate):
    code, out = gate
    assert code == 0
    document = json.loads((out / gate_run.RESULTS_NAME).read_text())
    assert document["calibration_rule"]["n_calibration"] == 100
    assert document["calibration_rule"]["calibration_half_below"] == 200
    for unit, (days, calibration, test) in EXPECTED.items():
        entry = document["units"][unit]
        assert (entry["counting_days"], entry["calibration_days_used"], entry["test_days"]) == (
            days,
            calibration,
            test,
        )
        assert entry["calibratable"] and entry["status"] == "calibratable"
        found = document["configurations"][PAIR[0]]["units"][unit]
        assert (found["calibration_days"], found["test_days"]) == (calibration, test)
    text = (out / gate_run.SUMMARY_NAME).read_text()
    assert "n_calibration 100" in text and "calibration_half_below 200" in text
    for unit, (days, calibration, test) in EXPECTED.items():
        assert (
            f"{unit}: {days} days, at least 100: no; split into {calibration} calibration days "
            f"and {test} test days"
        ) in text


def test_a_nine_day_unit_is_split_and_abstains_and_is_not_called_calibratable(gate):
    _, out = gate
    document = json.loads((out / gate_run.RESULTS_NAME).read_text())
    found = document["configurations"][PAIR[0]]["units"][GATE_UNITS[0]]
    for rung in ("fixed", "one-way", "two-way"):
        assert found["rungs"][rung]["beta_hat"] == 0.0
        assert found["rungs"][rung]["reason"].startswith("abstains")
        assert "too few days" in found["rungs"][rung]["reason"]
        assert found["rungs"][rung]["test"]["mean_offer_kwh_per_day"] == 0.0
    text = (out / gate_run.SUMMARY_NAME).read_text()
    (line,) = [x for x in text.splitlines() if x.startswith(f"  {GATE_UNITS[0]}: 9 days")]
    assert "calibratable" not in line
    assert "split into 4 calibration days and 5 test days" in line
    assert "scale 0" in text  # the offer lines say the offer is an abstention


@pytest.fixture(scope="module")
def dial(build, tmp_path_factory):
    out = tmp_path_factory.mktemp("dial") / "out"
    code = _run(dial_choice.main, build, out, "--model", "empirical", "--splits", "3")
    return code, out


def test_the_dial_choice_states_the_counts_and_the_rule(dial):
    code, out = dial
    assert code == 0
    document = json.loads((out / dial_choice.RESULTS_NAME).read_text())
    assert document["run"]["n_calibration"] == 100
    assert document["run"]["calibration_half_below"] == 200
    for unit, (days, calibration, test) in EXPECTED.items():
        entry = document["units"][unit]
        assert (entry["counting_days"], entry["calibration_days_used"], entry["test_days"]) == (
            days,
            calibration,
            test,
        )
        assert entry["status"] == "calibratable"
    text = (out / dial_choice.SUMMARY_NAME).read_text()
    assert "n_calibration 100" in text and "calibration_half_below 200" in text
    for unit, (days, calibration, test) in EXPECTED.items():
        assert (
            f"{unit} ({days} counting days, each split: {calibration} calibration days and "
            f"{test} test days):"
        ) in text
    assert "calibratable" not in text.replace("not calibratable", "")


def test_in_the_dial_choice_the_nine_day_unit_never_offers(dial):
    _, out = dial
    document = json.loads((out / dial_choice.RESULTS_NAME).read_text())
    entry = document["units"][GATE_UNITS[0]]
    for found in entry["configurations"].values():
        for tier in ("A", "B"):
            assert found["tiers"][tier]["mean_offer_kwh_per_test_day"] == 0.0
            assert found["tiers"][tier]["share_of_splits_that_offer"] == 0.0
