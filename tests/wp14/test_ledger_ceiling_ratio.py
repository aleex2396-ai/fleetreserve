"""Outcome O8's ceiling ratio V in the ledger command (`ledger_run`, `ceiling_ratio`) on the
hand-worked run of `ledger_fake`, with the budget read from the matrix, and with a test day whose
one-way ceiling is 0.

The hand run (`ledger_fake`'s module text): one service hour (hour 1), rho 0 (every car needs its
whole room), two units with one test day each, cars given as departure hour d (the car is
plugged in in hours 0 to d - 1), rate r (kW) and room e (kWh). Its one-way ceiling is 3 on both
units. Two-way at budget B (`day.TrueCar.two_way`: power in [-r, r] before the departure hour,
the energy stored since the decision time in [-B, e] after every hour, at least e at the end):

- `nrel-from-may-2019`. Car 1 (d 2, r 1, e 1) must hold 1 kWh after hour 1, so it holds 0 to 1
  after hour 0 and draws 0 to 1 in hour 1: 1. Car 2 (d 3, r 1, e 1) may give back 1 in hour 0 and
  take 1 in hours 1 and 2 (b(S) = 1), or take 1 in hour 0, give back 1 in hour 1 and take it back
  in hour 2 (p(S) = -1): 2. Car 3 (d 2, r 2, e 1) may give back 1 in hour 0 and take 2 in hour 1
  (b(S) = 2; giving back more leaves it short), or take 1 in hour 0 and nothing in hour 1
  (p(S) = 0): 2. At B = 5 the two-way ceiling is 1 + 2 + 2 = 5 and V = 5 / 3.
- `norway-development-pooled`: three cars like car 2, 2 each: the two-way ceiling 6 and V = 2.

The run's two-way offer as a share of the two-way ceiling (its offers: `ledger_fake.HAND_SCALES`
times `HAND_CERTIFIED`, one test day): NREL empirical, tier A 1.25 x 0.8 = 1 of 5 = 0.2, tier B
2 x 0.8 = 1.6 of 5 = 0.32; NREL oracle 1 of 5 = 0.2 in both tiers; Norway empirical, tier A
1.25 x 2.5 = 3.125 of 6, tier B 2 x 2.5 = 5 of 6; Norway oracle 3 of 6 = 0.5 in both tiers.

The budget read from the matrix: the same run on a matrix whose `two_way_budget_kwh` is 0.5.
Car 3 gives back only 0.5 in hour 0 and then takes 1.5 in hour 1 (b(S) = 1.5, p(S) = 0); cars 1
and 2 are unchanged (car 2's hour-1 power is limited by its rate, and its p(S) uses its room
first). NREL's two-way ceiling is 4.5 and V = 1.5; Norway's stays 6 and V = 2. The label is
"B=0.5" and every O8 line says that it is not outcome O8's B=5; a ledger that kept 5 kWh would
give 5 and 5 / 3 at NREL.

A test day whose one-way ceiling is 0: the Norway unit's three cars with room 0 (d 3, r 1). One-way
none can draw anything (ceiling 0, V_aff 0); two-way each may give back 1 in hour 0 and take it
back in hour 1 (b(S) = 1), or give back 1 in hour 1 and take it back in hour 2 (p(S) = -1): 2
each, 6 together. Norway's one-way sum is 0, so V is not defined there; the day is counted and
named with its two-way 6 kWh.
"""

import csv
import json

import ledger_fake
import pytest
from gate_fake import COMMIT, ROUNDED
from ledger_fake import HAND_UNITS, write_hand_run

from fleetreserve import gate_run, ledger_run, run_verify

EMPIRICAL, ORACLE = ROUNDED
NREL, NORWAY = HAND_UNITS
D_TEXT = "Discharge D: not run (the owner's decision of 16 November 2026)."
# unit -> (one-way sum, two-way sum) at 5 kWh, by hand (module text).
AT_FIVE = {NREL: (3.0, 5.0), NORWAY: (3.0, 6.0)}
# (configuration, unit) -> {tier: the two-way offer's share of the two-way ceiling}, by hand.
SHARES = {
    (EMPIRICAL, NREL): {"A": 0.2, "B": 0.32},
    (ORACLE, NREL): {"A": 0.2, "B": 0.2},
    (EMPIRICAL, NORWAY): {"A": 3.125 / 6, "B": 5.0 / 6},
    (ORACLE, NORWAY): {"A": 0.5, "B": 0.5},
}


def _hand_ledger(folder, monkeypatch, budget=None, empty_norway=False):
    """The ledger on the hand run (module text), its matrix's budget set to `budget` when given,
    the Norway unit's rooms set to 0 when `empty_norway`."""
    if budget is not None:
        document = ledger_fake.hand_matrix()
        document["two_way_budget_kwh"] = budget
        monkeypatch.setattr(ledger_fake, "hand_matrix", lambda: document)
    if empty_norway:
        site, fleet = ledger_fake.HAND_FLEETS[NORWAY]
        empty = tuple((d, r, 0.0) for d, r, _ in fleet)
        monkeypatch.setitem(ledger_fake.HAND_FLEETS, NORWAY, (site, empty))
        monkeypatch.setitem(ledger_fake.HAND_TRUE, NORWAY, (0.0, 0.0, 0.0))
    paths = write_hand_run(folder)
    out = folder / "ledger"
    out.mkdir()
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setattr(ledger_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = ["--run", str(paths["run"]), "--matrix", str(paths["matrix"])]
    arguments += ["--tables", str(paths["tables"]), "--truth-run", str(paths["truth"])]
    code = ledger_run.main([*arguments, "--tree-days", "1", "--seed", "3"])
    document = json.loads((out / ledger_run.RESULTS_NAME).read_text())
    return code, document, out


@pytest.fixture(scope="module")
def at_five(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        return _hand_ledger(tmp_path_factory.mktemp("o8-hand"), monkeypatch)


def _o8_lines(out) -> list[str]:
    lines = (out / ledger_run.SUMMARY_NAME).read_text().splitlines()
    return [line for line in lines if "O8" in line and "O8 offer ratio" not in line]


def test_the_hand_runs_ceiling_ratio_is_the_hand_values(at_five):
    code, document, _ = at_five
    assert code == 0 and document["checks"]["failed"] is False
    assert document["checks"]["ceiling_ratio_one_way_reproduced"]["breaks"] == []
    run = document["ceiling_ratio"]
    assert (run["budget_kwh"], run["label"], run["note"]) == (5.0, "B=5", "")
    assert run["discharge_d"] == "not run (the owner's decision of 16 November 2026)"
    assert run["readings"] == [f"C{k}" for k in range(1, 10)]
    for unit, (one, two) in AT_FIVE.items():
        found = document["units"][unit]["ceiling_ratio"]
        assert found["test_days"] == 1
        assert found["one_way_sum_kwh"] == pytest.approx(one, abs=1e-9)
        assert found["two_way_sum_kwh"] == pytest.approx(two, abs=1e-9)
        assert found["v"] == pytest.approx(two / one, abs=1e-9)
        assert found["one_way_zero_days"] == []
        for (name, owner), tiers in SHARES.items():
            if owner == unit:
                shares = found["two_way_offer_share_of_two_way_ceiling"][name]
                assert shares == pytest.approx(tiers, abs=1e-12), (name, unit)


def test_the_summary_prints_v_at_b5_and_the_discharge_as_not_run(at_five):
    _, _, out = at_five
    lines = _o8_lines(out)
    assert lines[0].startswith("Outcome O8 at B=5 kWh (the matrix's two_way_budget_kwh): ")
    assert D_TEXT in lines[0]
    assert (
        "Offer ratio X: X = the sum over a unit's test days of the run's two-way offer" in lines[0]
    )
    assert lines[1:] == [
        "  O8 ceiling ratio V at B=5 kWh, true sets, 1 test days: one-way ceiling sum 3.000 kWh, "
        "two-way 5.000 kWh, V = 1.667; no test day with a one-way ceiling of 0",
        "  O8 ceiling ratio V at B=5 kWh, true sets, 1 test days: one-way ceiling sum 3.000 kWh, "
        "two-way 6.000 kWh, V = 2.000; no test day with a one-way ceiling of 0",
    ]
    text = (out / ledger_run.SUMMARY_NAME).read_text()
    assert (
        f"    {EMPIRICAL}: tier A (driver-protected) 20.0%, tier B (deadline contract) 32.0%"
    ) in text


def test_the_ceiling_ratio_rows_hold_each_test_days_ceilings(at_five):
    _, document, out = at_five
    rows = list(csv.DictReader((out / ledger_run.RATIO_NAME).read_text().splitlines()))
    assert [r["unit"] for r in rows] == [NREL, NORWAY]
    for row in rows:
        one, two = AT_FIVE[row["unit"]]
        assert float(row["one_way_ceiling_kwh"]) == pytest.approx(one, abs=1e-9)
        assert float(row["run_ceiling_hindsight_kwh"]) == one
        assert float(row["two_way_ceiling_kwh"]) == pytest.approx(two, abs=1e-9)
        assert float(row["budget_kwh"]) == 5.0
        assert f"{row['site']} {row['date']}" in document["tree_sample"]["days"][row["unit"]]


def test_the_budget_is_the_matrixs(tmp_path, monkeypatch):
    code, document, out = _hand_ledger(tmp_path, monkeypatch, budget=0.5)
    assert code == 0
    run = document["ceiling_ratio"]
    assert (run["budget_kwh"], run["label"]) == (0.5, "B=0.5")
    assert run["note"] == ", not outcome O8's B=5"
    nrel, norway = (document["units"][u]["ceiling_ratio"] for u in (NREL, NORWAY))
    assert nrel["two_way_sum_kwh"] == pytest.approx(4.5, abs=1e-9)
    assert nrel["v"] == pytest.approx(1.5, abs=1e-9)
    assert norway["two_way_sum_kwh"] == pytest.approx(6.0, abs=1e-9)
    assert norway["v"] == pytest.approx(2.0, abs=1e-9)
    lines = _o8_lines(out)
    assert len(lines) == 3
    assert lines[0].startswith("Outcome O8 at B=0.5 kWh (the matrix's two_way_budget_kwh, not ")
    for line in lines[1:]:
        assert line.startswith("  O8 ceiling ratio V at B=0.5 kWh, not outcome O8's B=5, ")
    assert all("at B=5" not in line for line in lines)


def test_a_test_day_whose_one_way_ceiling_is_0_is_counted_and_shown(tmp_path, monkeypatch):
    code, document, out = _hand_ledger(tmp_path, monkeypatch, empty_norway=True)
    assert code == 0, document["checks"]
    found = document["units"][NORWAY]["ceiling_ratio"]
    (day,) = document["tree_sample"]["days"][NORWAY]  # the unit's one test day
    assert found["test_days"] == 1 and found["one_way_sum_kwh"] == 0.0
    assert found["v"] is None
    assert found["one_way_zero_days"] == [day]
    assert found["two_way_sum_kwh"] == pytest.approx(6.0, abs=1e-9)
    assert found["two_way_sum_on_one_way_zero_days_kwh"] == pytest.approx(6.0, abs=1e-9)
    assert _o8_lines(out)[2] == (
        "  O8 ceiling ratio V at B=5 kWh, true sets, 1 test days: one-way ceiling sum 0.000 kWh, "
        "two-way 6.000 kWh, V = not defined (one-way sum 0); 1 test days with a one-way ceiling "
        f"of 0, in both sums (two-way 6.000 kWh on them): {day}"
    )
