"""`dev_day_verdicts` on made-up per-day values: the three results, the coverage rule, the T4 log's
check and findings, and T2's "looked at nothing" reading."""

from dataclasses import replace

import pytest

from fleetreserve import dev_day_verdicts as v
from fleetreserve.dev_day_values import ORDER_KEYS, OrderValues
from fleetreserve.t2_day import CERTIFIED, NO_STEERED_CAR, T2Result

GOOD = {
    "hourly_diag": 1.0,
    "hourly_causal": 2.0,
    "hourly_tree": 2.0,
    "hourly_hind": 3.0,
    "block_diag": 0.5,
    "block_aff": 1.0,
    "block_caus": 1.0,
    "block_hind": 1.0,
}


def day(k=0, hypotheses=True, in_window=False, **values) -> OrderValues:
    found = GOOD | values
    outcomes = {key: "optimal" if found[key] is not None else "error" for key in ORDER_KEYS}
    return OrderValues(
        "u", "A", f"2020-01-{k + 1:02d}", 3, 4, in_window, hypotheses, None, found, outcomes
    )


def test_ordered_days_pass():
    assert v.order_check([day(0), day(1)])["verdict"] == v.PASSED


@pytest.mark.parametrize(
    "values",
    [
        {"hourly_diag": 2.5},  # above affine
        {"hourly_tree": 1.5},  # tree below affine
        {"hourly_hind": 1.9},  # hindsight below tree
        {"block_aff": 0.4},  # affine below diagonal
        {"block_aff": None, "block_diag": 1.2},  # a pair across a missing value
    ],
)
def test_a_broken_order_fails(values):
    found = v.order_check([day(0), day(1, **values)])
    assert found["verdict"] == v.FAILED
    assert found["failure_lines"][0].startswith("u A 2020-01-02:")


def test_a_difference_within_the_tolerance_is_not_a_violation():
    assert v.order_check([day(0, hourly_tree=2.0 - 1e-7)])["verdict"] == v.PASSED


def test_a_timeout_is_missing_and_the_check_could_not_run():
    timed = day(1, hourly_tree=None)
    timed.outcomes["hourly_tree"] = "timeout"
    found = v.order_check([day(0), timed, day(2, block_hind=None)])
    assert found["verdict"] == v.COULD_NOT_RUN
    assert (found["covered_days"], found["timeout_days"], found["other_missing_days"]) == (1, 1, 1)


def test_a_day_that_could_not_be_built_fails_the_order_and_is_missing_from_the_log():
    broken = replace(day(1), error="empty set")
    assert v.order_check([day(0), broken])["verdict"] == v.FAILED
    log = v.t4_log([day(0), broken])
    assert log["verdict"] == v.COULD_NOT_RUN and log["other_missing_days"] == 1


def test_the_t4_log_checks_hypothesis_days_and_records_findings():
    days = [
        day(0),
        day(1, hypotheses=False, block_hind=2.0, block_caus=1.5, block_aff=1.5),  # gap only
        day(2, hypotheses=False, in_window=True, block_aff=0.8),  # finding, not a failure
    ]
    log = v.t4_log(days)
    assert log["verdict"] == v.PASSED
    assert log["hypothesis_days"] == log["hypothesis_days_checked"] == 1
    assert [g["days"] for g in log["groups"].values()] == [1, 1, 1]
    assert log["groups"]["other days, no in-window departure"]["days_tree_below_hind"] == 1
    assert len(log["findings_tree_above_affine"]) == 1
    broken = v.t4_log([day(0, block_hind=2.0, block_caus=1.5)])
    assert broken["verdict"] == v.FAILED
    assert "on a day meeting T4's hypotheses" in broken["failure_lines"][0]


def t2(status=CERTIFIED, in_e=True, problems=()) -> T2Result:
    certified = status == CERTIFIED
    return T2Result(
        "u", "A", "2020-01-01", 3, status, 0, 0, 3 if certified else 0, 3 if in_e else 2,
        in_e and certified, 1.0 if certified else None, 30, 0, 0.0, 3, 0, 0, 0, 0.0,
        tuple(problems),
    )  # fmt: skip


def test_t2_passes_with_a_certified_day_in_e_prime_and_counts_rule_days_as_covered():
    found = v.t2_check([t2(), t2(NO_STEERED_CAR)], 2, 0)
    assert found["verdict"] == v.PASSED and found["covered_days"] == 2
    assert found["share_of_sampled_days_in_e_prime"] == 0.5


def test_t2_with_no_day_in_e_prime_or_no_certificate_could_not_run():
    assert v.t2_check([t2(in_e=False)], 1, 0)["verdict"] == v.COULD_NOT_RUN
    assert v.t2_check([t2(NO_STEERED_CAR)], 1, 0)["verdict"] == v.COULD_NOT_RUN


def test_t2_with_a_problem_fails_and_a_timeout_could_not_run():
    assert v.t2_check([t2(problems=["x"])], 1, 0)["verdict"] == v.FAILED
    timed = v.t2_check([t2(), t2("timeout")], 2, 0)
    assert timed["verdict"] == v.COULD_NOT_RUN and timed["timeout_days"] == 1
