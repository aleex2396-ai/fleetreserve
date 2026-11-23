"""Plan section 13: a timeout or any outcome but optimal from LP (24), LP (27) or the box in their
sets is a missing result, never a number.

1. A real timeout: a made-up day of `N_CARS` cars and `N_SERVICE` service hours, LP (24) and
   LP (27) solved with a time limit of `TINY_LIMIT` seconds, and the box programme over corners
   likewise: each returns "timeout" with no numbers.
2. Made-up solver answers (the solver door replaced in `sparse_programme`): each outcome but
   optimal is passed through unchanged with no numbers, and the box of a missing inner set is
   missing with the same outcome.
3. The default time limits: `lp.LP24_SECONDS` for LP (24), `lp.OTHER_SECONDS` for LP (27) (plan
   section 13: 600 s LP (24), 300 s others).
"""

import datetime
import random

import numpy as np
import pytest
from made_up_days import made_up_car

from fleetreserve import products, sparse_programme
from fleetreserve.base_sets import day_average
from fleetreserve.carsets import true_rows
from fleetreserve.day import Day, covering_recovery_hours
from fleetreserve.inner_box import largest_box, vertex_box
from fleetreserve.lp import LP24_SECONDS, OTHER_SECONDS, LpResult
from fleetreserve.lp24 import inner_set

N_CARS = 30
N_SERVICE = 6
TINY_LIMIT = 1e-6
NOT_OPTIMAL = ["timeout", "infeasible", "unbounded", "error"]


def _rows():
    rng = random.Random(5)
    cars = [made_up_car(rng, N_SERVICE, 0.1)[0] for _ in range(N_CARS)]
    n_recovery = covering_recovery_hours(cars, N_SERVICE)
    day = Day("made-up", datetime.date(2020, 1, 1), N_SERVICE, n_recovery, 0.1, tuple(cars))
    return true_rows(day)


def _assert_missing_set(result, outcome):
    assert result.outcome == outcome
    for field in ("value", "alpha", "p", "P", "gamma", "Gamma"):
        assert getattr(result, field) is None
    with pytest.raises(ValueError):
        result.profile(np.zeros(N_SERVICE + 1))


def _assert_missing_box(box, outcome):
    assert box.outcome == outcome
    assert box.value is None and box.m is None and box.w is None


def test_a_real_timeout_is_missing():
    rows = _rows()
    base = day_average(rows)
    for programme in ("24", "27"):
        result = inner_set(rows, base, programme, keep_maps=True, time_limit=TINY_LIMIT)
        _assert_missing_set(result, "timeout")
    solved = inner_set(rows, base, "27")
    assert solved.outcome == "optimal"
    box = vertex_box(solved.p, solved.P, base, products.hourly(), time_limit=TINY_LIMIT)
    _assert_missing_box(box, "timeout")


def _made_up(outcome):
    def fake(c, **kwargs):
        return LpResult(outcome, None, None, 0.5, kwargs["time_limit"], 1, f"made up {outcome}")

    return fake


@pytest.mark.parametrize("outcome", NOT_OPTIMAL)
def test_every_outcome_but_optimal_is_missing(monkeypatch, outcome):
    rows = _rows()
    base = day_average(rows)
    solved = inner_set(rows, base, "27")
    monkeypatch.setattr(sparse_programme, "solve_lp", _made_up(outcome))
    for programme in ("24", "27"):
        result = inner_set(rows, base, programme, keep_maps=True)
        _assert_missing_set(result, outcome)
        for product in (products.hourly(), products.block()):
            _assert_missing_box(largest_box(result, product), outcome)
    for product in (products.hourly(), products.block()):
        _assert_missing_box(vertex_box(solved.p, solved.P, base, product), outcome)


@pytest.mark.parametrize(("programme", "limit"), [("24", LP24_SECONDS), ("27", OTHER_SECONDS)])
def test_default_time_limits(monkeypatch, programme, limit):
    seen = []

    def record(c, **kwargs):
        seen.append(kwargs["time_limit"])
        return _made_up("timeout")(c, **kwargs)

    rows = _rows()
    monkeypatch.setattr(sparse_programme, "solve_lp", record)
    assert inner_set(rows, day_average(rows), programme).outcome == "timeout"
    assert seen == [limit]
