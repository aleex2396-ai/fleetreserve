"""Outcome O8's offer ratio X and outcome O5's offer order (`ceiling_ratio.offer_ratio`), formed by
the ledger command (`ledger_run`) from the per-day offers of a recorded run's three rungs.

The function, on hand-worked days given as (day, fixed, one-way, two-way offer in kWh):

- ratio above 1, ordered: (1, 2, 4), (1, 1, 1) and (0, 0, 2). The sums are 2, 3 and 7, so
  X = 7 / 3 (reading C7, a ratio of sums). The third day has a one-way offer of 0: it stays in
  both sums, is counted and named, with its two-way 2 kWh (reading C9); the mean of the daily
  ratios would be undefined there (2 / 0). All three days obey fixed <= one-way <= two-way: the
  share is 3 / 3.
- equal: (1, 1, 1) and (2, 2, 2): X = 1, ordered on both days (the order allows equality).
- a violating day: (1, 2, 4) and (1, 3, 2): X = 6 / 5 over the sums 6 and 5, ordered on 1 of 2
  days; the second day has the two-way offer below the one-way one and is named with its three
  offers (reading C8). A day with the one-way offer below the fixed one, (2, 1, 3), is broken too.
- no one-way offer at all: (0, 0, 2) alone: X is None and the day is a zero day, ordered.
- no day: X and the share are None.

The command, on the hand-worked run of `ledger_fake` (one test day per unit; the day's offer of a
rung is its scale times V_cert, so the offers per tier are:

- `nrel-from-may-2019`, empirical, V_cert 0.8: tier A scales 0.5, 0.75, 1.25 give 0.4, 0.6, 1.0
  and X = 1.0 / 0.6 = 5 / 3; tier B scales 1, 1.5, 2 give 0.8, 1.2, 1.6 and X = 4 / 3.
- `norway-development-pooled`, empirical, V_cert 2.5: tier A 1.25, 1.875, 3.125, X = 5 / 3; tier B
  2.5, 3.75, 5, X = 4 / 3.
- the oracle configuration, every scale 1: X = 1 on both units and both tiers (equal offers).

Changed scales: the empirical tier A scales of the one-way and the two-way rung set to 1.5 and
1.25 (the two-way offer 1.0 below the one-way 1.2 at NREL: X = 5 / 6, ordered on 0 of 1 days); the
empirical tier A scales of the fixed and the one-way rung set to 0 (offers 0, 0, 1.0 at NREL: the
day has a one-way offer of 0, X is None, the day is ordered and named with its two-way 1.0 kWh).
"""

import dataclasses

import pytest
from gate_fake import ROUNDED
from ledger_fake import HAND_SCALES, HAND_UNITS
from test_ledger_ceiling_ratio import _hand_ledger

from fleetreserve import ceiling_ratio, ledger_run, ledger_summary

EMPIRICAL, ORACLE = ROUNDED
NREL, NORWAY = HAND_UNITS
TIERS = ("A", "B")


def _days(*offers):
    return [(f"site 2020-01-0{k}", *three) for k, three in enumerate(offers, start=1)]


def test_a_ratio_above_1_with_a_one_way_zero_day_is_a_ratio_of_sums():
    found = ceiling_ratio.offer_ratio(_days((1, 2, 4), (1, 1, 1), (0, 0, 2)))
    assert found.test_days == 3
    assert (found.fixed_sum_kwh, found.one_way_sum_kwh, found.two_way_sum_kwh) == (2, 3, 7)
    assert found.x == pytest.approx(7 / 3, abs=1e-12)
    assert found.one_way_zero_days == ("site 2020-01-03",)
    assert found.two_way_sum_on_one_way_zero_days_kwh == 2
    assert (found.fixed_one_way_days, found.one_way_two_way_days) == (3, 3)
    assert (found.fixed_one_way_share, found.one_way_two_way_share) == (1.0, 1.0)
    assert found.fixed_one_way_broken_days == found.one_way_two_way_broken_days == ()


def test_equal_offers_are_ordered_and_give_x_1():
    found = ceiling_ratio.offer_ratio(_days((1, 1, 1), (2, 2, 2)))
    assert found.x == 1.0
    assert (found.fixed_one_way_days, found.one_way_two_way_days) == (2, 2)
    assert found.fixed_one_way_broken_days == found.one_way_two_way_broken_days == ()
    assert found.one_way_zero_days == ()


def test_a_day_with_the_two_way_offer_below_the_one_way_one_is_named():
    found = ceiling_ratio.offer_ratio(_days((1, 2, 4), (1, 3, 2)))
    assert found.x == pytest.approx(6 / 5, abs=1e-12)
    assert (found.fixed_one_way_days, found.fixed_one_way_share) == (2, 1.0)
    assert (found.one_way_two_way_days, found.one_way_two_way_share) == (1, 0.5)
    assert found.fixed_one_way_broken_days == ()
    assert found.one_way_two_way_broken_days == (
        "site 2020-01-02 (fixed 1, one-way 3, two-way 2 kWh)",
    )


def test_a_day_with_the_one_way_offer_below_the_fixed_one_is_named_and_the_steps_are_apart():
    # only fixed > one-way breaks: the second step holds on the same day
    found = ceiling_ratio.offer_ratio(_days((2, 1, 3)))
    assert (found.fixed_one_way_days, found.fixed_one_way_share) == (0, 0.0)
    assert (found.one_way_two_way_days, found.one_way_two_way_share) == (1, 1.0)
    assert found.fixed_one_way_broken_days == (
        "site 2020-01-01 (fixed 2, one-way 1, two-way 3 kWh)",
    )
    assert found.one_way_two_way_broken_days == ()
    # both steps break on one day: it is named under each
    found = ceiling_ratio.offer_ratio(_days((3, 2, 1)))
    assert (found.fixed_one_way_days, found.one_way_two_way_days) == (0, 0)
    assert found.fixed_one_way_broken_days == found.one_way_two_way_broken_days


def test_no_one_way_offer_at_all_leaves_x_undefined_and_the_zero_day_ordered():
    found = ceiling_ratio.offer_ratio(_days((0, 0, 2)))
    assert found.x is None
    assert found.one_way_zero_days == ("site 2020-01-01",)
    assert (found.fixed_one_way_days, found.one_way_two_way_days) == (1, 1)


def test_no_day_gives_no_ratio_and_no_share():
    found = ceiling_ratio.offer_ratio([])
    assert (found.test_days, found.x) == (0, None)
    assert (found.fixed_one_way_share, found.one_way_two_way_share) == (None, None)
    assert (found.fixed_one_way_days, found.one_way_two_way_days) == (0, 0)


def test_a_difference_within_the_tolerance_is_not_a_break():
    found = ceiling_ratio.offer_ratio(_days((1.0, 1.0 + 1e-9, 1.0 + 1e-9 - 1e-8)))
    assert (found.fixed_one_way_share, found.one_way_two_way_share) == (1.0, 1.0)
    found = ceiling_ratio.offer_ratio(_days((1.0, 1.0 + 1e-3, 1.0)))
    assert found.one_way_two_way_share == 0.0


# --- the command, on the hand-worked run ---------------------------------------------------------


STEPS_OK = (
    "fixed <= one-way on 1 of 1 test days (100.0%), one-way <= two-way on 1 of 1 test days (100.0%)"
)


def _x_lines(out) -> list[str]:
    lines = (out / ledger_run.SUMMARY_NAME).read_text().splitlines()
    return [line for line in lines if "offer ratio X" in line and line.startswith("  O8")]


def test_the_hand_runs_offer_ratio_is_the_hand_values(tmp_path, monkeypatch):
    code, document, out = _hand_ledger(tmp_path, monkeypatch)
    assert code == 0 and document["checks"]["failed"] is False
    by_hand = {
        (EMPIRICAL, NREL): {"A": 5 / 3, "B": 4 / 3},
        (EMPIRICAL, NORWAY): {"A": 5 / 3, "B": 4 / 3},
        (ORACLE, NREL): {"A": 1.0, "B": 1.0},
        (ORACLE, NORWAY): {"A": 1.0, "B": 1.0},
    }
    for (name, unit), tiers in by_hand.items():
        found = document["units"][unit]["offer_ratio"][name]
        assert set(found) == set(TIERS)
        for tier, x in tiers.items():
            assert found[tier]["x"] == pytest.approx(x, abs=1e-12), (name, unit, tier)
            assert found[tier]["test_days"] == 1
            for step in ("fixed_one_way", "one_way_two_way"):
                assert (found[tier][f"{step}_days"], found[tier][f"{step}_share"]) == (1, 1.0)
                assert found[tier][f"{step}_broken_days"] == []
            assert found[tier]["one_way_zero_days"] == []
    nrel = document["units"][NREL]["offer_ratio"][EMPIRICAL]["A"]
    assert nrel["fixed_sum_kwh"] == pytest.approx(0.4, abs=1e-12)
    assert nrel["one_way_sum_kwh"] == pytest.approx(0.6, abs=1e-12)
    assert nrel["two_way_sum_kwh"] == pytest.approx(1.0, abs=1e-12)
    run = document["ceiling_ratio"]
    assert "not computed" not in run["offer_ratio_x"]
    assert run["readings"] == [f"C{k}" for k in range(1, 10)]
    lines = (out / ledger_run.SUMMARY_NAME).read_text().splitlines()
    head = [line for line in lines if "Offer ratio" in line]
    assert len(head) == 1 and "not computed" not in head[0]


def test_the_summary_prints_x_and_the_share_per_configuration_and_tier(tmp_path, monkeypatch):
    _, _, out = _hand_ledger(tmp_path, monkeypatch)
    lines = _x_lines(out)
    assert len(lines) == 2 and all("at B=5" in line for line in lines)
    text = (out / ledger_run.SUMMARY_NAME).read_text()
    assert (
        f"    {EMPIRICAL}: tier A (driver-protected) X = 1.667, {STEPS_OK}; "
        f"tier B (deadline contract) X = 1.333, {STEPS_OK}"
    ) in text
    assert (
        f"    {ORACLE}: tier A (driver-protected) X = 1.000, {STEPS_OK}; "
        f"tier B (deadline contract) X = 1.000, {STEPS_OK}"
    ) in text
    assert "ORDER BROKEN" not in text


def _check_failed_lines(out) -> list[str]:
    lines = (out / ledger_run.SUMMARY_NAME).read_text().splitlines()
    return [line for line in lines if line.startswith("CHECK FAILED")]


def test_a_two_way_offer_below_the_one_way_one_fails_the_ledger(tmp_path, monkeypatch):
    scales = {**HAND_SCALES[(EMPIRICAL, "A")], "one-way": 1.5, "two-way": 1.25}
    monkeypatch.setitem(HAND_SCALES, (EMPIRICAL, "A"), scales)
    code, document, out = _hand_ledger(tmp_path, monkeypatch)
    assert code == 1  # plan O5: a violation is a bug (T5: the order always holds)
    assert document["checks"]["failed"] is True
    assert (out / ledger_run.SUMMARY_NAME).exists() and (out / ledger_run.RESULTS_NAME).exists()
    found = document["units"][NREL]["offer_ratio"][EMPIRICAL]
    assert found["A"]["x"] == pytest.approx(1.0 / 1.2, abs=1e-12)
    assert (found["A"]["fixed_one_way_days"], found["A"]["fixed_one_way_share"]) == (1, 1.0)
    assert (found["A"]["one_way_two_way_days"], found["A"]["one_way_two_way_share"]) == (0, 0.0)
    (day,) = document["tree_sample"]["days"][NREL]
    assert found["A"]["fixed_one_way_broken_days"] == []
    (broken,) = found["A"]["one_way_two_way_broken_days"]
    assert broken == f"{day} (fixed 0.4, one-way 1.2, two-way 1 kWh)"
    assert found["B"]["one_way_two_way_share"] == 1.0  # tier B is untouched
    # the scales are the configuration's, so the same step breaks on both units
    nrel_entry, norway_entry = document["checks"]["offer_order"]["breaks"]
    assert nrel_entry == (
        f"{NREL}, {EMPIRICAL}, tier A, one-way <= two-way: 1 of 1 test days: {broken}"
    )
    assert norway_entry.startswith(f"{NORWAY}, {EMPIRICAL}, tier A, one-way <= two-way: 1 of 1 ")
    (line,) = _check_failed_lines(out)
    assert line == (
        "CHECK FAILED: the offer order fixed <= one-way <= two-way (plan T5, outcome O5), "
        f"2 breaks: {nrel_entry}; {norway_entry}"
    )
    text = (out / ledger_run.SUMMARY_NAME).read_text()
    assert text.splitlines()[0] == line  # the opening lines lead with the failed check
    assert (
        "      ORDER BROKEN, one-way <= two-way, tier A (driver-protected), 1 of 1 test days: "
        f"{broken}"
    ) in text
    assert "ORDER BROKEN, fixed <= one-way" not in text
    assert (
        "X = 0.833, fixed <= one-way on 1 of 1 test days (100.0%), one-way <= two-way on 0 of 1 "
        "test days (0.0%)"
    ) in text


def test_only_fixed_above_one_way_breaks_the_first_step_and_fails_the_ledger(tmp_path, monkeypatch):
    scales = {**HAND_SCALES[(EMPIRICAL, "A")], "fixed": 1.0}
    monkeypatch.setitem(HAND_SCALES, (EMPIRICAL, "A"), scales)
    code, document, out = _hand_ledger(tmp_path, monkeypatch)
    assert code == 1
    found = document["units"][NREL]["offer_ratio"][EMPIRICAL]["A"]
    (day,) = document["tree_sample"]["days"][NREL]
    assert (found["fixed_one_way_days"], found["fixed_one_way_share"]) == (0, 0.0)
    assert (found["one_way_two_way_days"], found["one_way_two_way_share"]) == (1, 1.0)
    assert found["fixed_one_way_broken_days"] == [f"{day} (fixed 0.8, one-way 0.6, two-way 1 kWh)"]
    assert found["one_way_two_way_broken_days"] == []
    (line,) = _check_failed_lines(out)
    assert f"{NREL}, {EMPIRICAL}, tier A, fixed <= one-way: 1 of 1 test days: {day}" in line
    assert f"{NORWAY}, {EMPIRICAL}, tier A, fixed <= one-way: 1 of 1 test days" in line
    text = (out / ledger_run.SUMMARY_NAME).read_text()
    assert (
        "X = 1.667, fixed <= one-way on 0 of 1 test days (0.0%), one-way <= two-way on 1 of 1 "
        "test days (100.0%)"
    ) in text
    assert "ORDER BROKEN, one-way <= two-way" not in text


def test_a_run_with_the_order_holding_has_no_failed_check_line(tmp_path, monkeypatch):
    code, document, out = _hand_ledger(tmp_path, monkeypatch)
    assert code == 0 and document["checks"]["offer_order"]["breaks"] == []
    assert _check_failed_lines(out) == []


def test_a_one_way_zero_day_is_counted_named_and_leaves_x_undefined(tmp_path, monkeypatch):
    scales = {**HAND_SCALES[(EMPIRICAL, "A")], "fixed": 0.0, "one-way": 0.0}
    monkeypatch.setitem(HAND_SCALES, (EMPIRICAL, "A"), scales)
    code, document, out = _hand_ledger(tmp_path, monkeypatch)
    assert code == 0, document["checks"]
    found = document["units"][NREL]["offer_ratio"][EMPIRICAL]["A"]
    (day,) = document["tree_sample"]["days"][NREL]
    assert found["x"] is None
    assert found["one_way_zero_days"] == [day]
    assert found["two_way_sum_on_one_way_zero_days_kwh"] == pytest.approx(1.0, abs=1e-12)
    assert (found["fixed_one_way_days"], found["one_way_two_way_days"]) == (1, 1)
    text = (out / ledger_run.SUMMARY_NAME).read_text()
    assert (
        "      one-way offer 0, tier A (driver-protected), in both sums (two-way 1.000 kWh on "
        f"them), 1 test days: {day}"
    ) in text
    assert (
        "tier A (driver-protected) X = not defined (one-way sum 0), fixed <= one-way on 1 of 1 "
        "test days (100.0%)"
    ) in text


def test_a_unit_with_no_test_days_gives_that_reason_not_a_zero_sum():
    empty = dataclasses.asdict(ceiling_ratio.offer_ratio([]))
    assert "no test days" in ledger_summary._x_text(empty)
    assert "one-way sum 0" not in ledger_summary._x_text(empty)
    assert "X = not defined (no test days), fixed <= one-way on 0 of 0 test days (-)" in (
        ledger_summary._x_text(empty)
    )
    zero = dataclasses.asdict(ceiling_ratio.offer_ratio(_days((0, 0, 2))))
    assert "X = not defined (one-way sum 0)" in ledger_summary._x_text(zero)
