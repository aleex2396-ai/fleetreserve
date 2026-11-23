"""Plan section 13: a timeout or any outcome but optimal is a missing result, never a number.

1. A real timeout: a made-up day of `N_CARS` cars and `N_SERVICE` service hours solved with a
   time limit of `TINY_LIMIT` seconds; the certificate and the tree return "timeout" with no
   value, scale or policy.
2. Made-up solver answers (the solver door replaced in `sparse_programme`): each outcome but
   optimal is passed through unchanged with no numbers; for the certificate's second stage, a
   failed second solve after an optimal first one is missing too, with the first value kept.
"""

import datetime
import random

import pytest
from made_up_days import made_up_car

from fleetreserve import products, sparse_programme
from fleetreserve.carsets import true_rows
from fleetreserve.certificate import ANY_OPTIMAL, LargestSmallestWidth, certify
from fleetreserve.day import Day, covering_recovery_hours
from fleetreserve.lp import LpResult, solve_lp
from fleetreserve.tree import corner_tree

N_CARS = 30
N_SERVICE = 6
TINY_LIMIT = 1e-6
NOT_OPTIMAL = ["timeout", "infeasible", "unbounded", "error"]


def _day():
    rng = random.Random(5)
    cars = [made_up_car(rng, N_SERVICE, 0.1)[0] for _ in range(N_CARS)]
    n_recovery = covering_recovery_hours(cars, N_SERVICE)
    return Day("made-up", datetime.date(2020, 1, 1), N_SERVICE, n_recovery, 0.1, tuple(cars))


def _assert_missing(result, outcome):
    assert result.outcome == outcome
    assert result.value is None
    assert result.scale is None
    assert result.policy is None


def test_a_real_timeout_is_missing():
    rows = true_rows(_day())
    for product in (products.hourly(), products.block()):
        _assert_missing(corner_tree(rows, product, time_limit=TINY_LIMIT), "timeout")
        certificate = certify(
            rows, product, "causal", shape_rule=ANY_OPTIMAL, time_limit=TINY_LIMIT
        )
        _assert_missing(certificate, "timeout")
        assert certificate.first_value is None


def _made_up(outcome):
    def fake(c, **kwargs):
        return LpResult(outcome, None, None, 0.5, kwargs["time_limit"], 1, f"made up {outcome}")

    return fake


@pytest.mark.parametrize("outcome", NOT_OPTIMAL)
def test_every_outcome_but_optimal_is_missing(monkeypatch, outcome):
    rows = true_rows(_day())
    monkeypatch.setattr(sparse_programme, "solve_lp", _made_up(outcome))
    _assert_missing(corner_tree(rows, products.block()), outcome)
    _assert_missing(certify(rows, products.block(), "causal", shape_rule=ANY_OPTIMAL), outcome)


@pytest.mark.parametrize("outcome", NOT_OPTIMAL)
def test_a_failed_second_stage_is_missing(monkeypatch, outcome):
    rows = true_rows(_day())
    calls = []

    def first_real_then_fail(c, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return solve_lp(c, **kwargs)
        return _made_up(outcome)(c, **kwargs)

    monkeypatch.setattr(sparse_programme, "solve_lp", first_real_then_fail)
    result = certify(rows, products.hourly(), "causal", shape_rule=LargestSmallestWidth(1e-6))
    _assert_missing(result, outcome)
    assert result.first_value is not None and result.first_value > 0
    assert len(calls) == 2
