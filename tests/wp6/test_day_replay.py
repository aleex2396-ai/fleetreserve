"""The day-level replay the calibration calls (`day_replay.replay_day`).

1. The gone day of `test_rungs_hand.py` (a car seen gone at the start of hour 2), wrapped in an
   optimal certificate, on its audited paths: the one-way rung's results per path are those of
   `run_rungs` (first failing
   hour, slot and driver loss, energies, re-solves with their outcomes and solver seconds), the
   fixed plan's results come with them, and the two-way rung's request adds its own.
2. Abstain (plan section 1): a day whose certificate is missing (infeasible or timeout) and any
   scale `beta <= 0` offer nothing: losses 0, no energies, no re-solves, and the reason is given so
   that the caller can count such days. On a made-up sample of `N_DAYS` days with every kind of
   forecast, `N_MISSING` certificates are missing and are counted as abstaining.
"""

import random

import numpy as np
import pytest
from made_up_offer_days import departure_rule, offer_day
from test_rungs_hand import EARLY, EARLY_POLICY, GONE, GONE_POLICY, _stay

from fleetreserve import products
from fleetreserve.audit_paths import audited_paths
from fleetreserve.certificate import Certificate
from fleetreserve.day_replay import replay_day
from fleetreserve.rungs import run_rungs
from fleetreserve.sparse_programme import Sizes

SEED = 2
SEEN = "hour-start"  # the abstain decision does not depend on the reading
N_DAYS = 40
N_MISSING = 1


def _certificate(outcome, policy=None):
    value = None if policy is None else float(2 * np.sum(policy.w))
    return Certificate(
        outcome, products.hourly(), "causal", "any", value, None, policy, value, 0.0, Sizes(0, 0, 0)
    )


def test_replay_day_returns_each_rung_per_path():
    paths = audited_paths(GONE.unit, GONE.date, GONE.n_service)
    result = replay_day(
        GONE,
        _certificate("optimal", GONE_POLICY),
        "one-way",
        1.0,
        paths,
        departure_bound=_stay(GONE, SEEN),
        departure_seen=SEEN,
    )
    assert not result.abstained and result.abstain_reason is None
    assert set(result.by_rung) == {"fixed", "one-way"}
    assert len(result.paths) == 8
    for path, replayed in zip(paths, result.paths, strict=True):
        runs = run_rungs(
            GONE, GONE_POLICY, 1.0, path, departure_bound=_stay(GONE, SEEN), departure_seen=SEEN
        )
        expected = runs["one-way"]
        np.testing.assert_array_equal(replayed.path, path)
        assert replayed.first_failure is None and expected.run.first_failure is None
        assert replayed.slot_loss == 0.0
        assert replayed.driver_loss == pytest.approx(expected.run.driver_loss)
        np.testing.assert_allclose(replayed.final, expected.run.final)
        assert [(r.hour, r.outcome) for r in replayed.resolves] == [(2, "optimal")]
        assert replayed.outcomes() == {"optimal": 1}
        assert replayed.resolves[0].seconds >= 0
    assert all(p.first_failure == 2 for p in result.by_rung["fixed"])
    two = replay_day(
        GONE,
        _certificate("optimal", GONE_POLICY),
        "two-way",
        1.0,
        paths[:1],
        departure_bound=_stay(GONE, SEEN),
        departure_seen=SEEN,
    )
    assert set(two.by_rung) == {"fixed", "one-way", "two-way"}


@pytest.mark.parametrize(
    ("outcome", "beta", "reason"),
    [
        ("infeasible", 1.0, "no certificate (infeasible)"),
        ("timeout", 1.0, "no certificate (timeout)"),
        ("optimal", 0.0, "beta <= 0"),
        ("optimal", -0.5, "beta <= 0"),
    ],
)
def test_a_day_that_abstains_offers_nothing(outcome, beta, reason):
    policy = EARLY_POLICY if outcome == "optimal" else None
    paths = audited_paths(EARLY.unit, EARLY.date, EARLY.n_service)
    result = replay_day(
        EARLY,
        _certificate(outcome, policy),
        "two-way",
        beta,
        paths,
        departure_bound=_stay(EARLY, SEEN),
        departure_seen=SEEN,
    )
    assert result.abstained and result.abstain_reason == reason
    for name in ("fixed", "one-way", "two-way"):
        for p in result.by_rung[name]:
            assert (p.first_failure, p.slot_loss, p.driver_loss, p.final, p.resolves) == (
                None,
                0.0,
                0.0,
                None,
                (),
            )


def test_missing_certificates_are_counted_as_abstaining():
    rng = random.Random(SEED)
    abstained = 0
    for k in range(N_DAYS):
        day, certificate, _ = offer_day(rng, k)
        paths = audited_paths(day.unit, day.date, day.n_service)[:1]
        result = replay_day(
            day,
            certificate,
            "fixed",
            1.0,
            paths,
            departure_bound=departure_rule(day, SEEN),
            departure_seen=SEEN,
        )
        abstained += result.abstained
        assert result.abstained == (certificate.outcome != "optimal")
    assert abstained == N_MISSING


@pytest.mark.parametrize("outcome", ["optimal", "infeasible"])
def test_the_reading_of_when_a_departure_is_seen_must_be_given(outcome):
    certificate = _certificate(outcome, EARLY_POLICY if outcome == "optimal" else None)
    args = (EARLY, certificate, "two-way", 1.0, [[1.0, 1.0]])
    with pytest.raises(TypeError):
        replay_day(*args, departure_bound=_stay(EARLY, SEEN))
    with pytest.raises(ValueError, match="departure_seen"):
        replay_day(*args, departure_bound=_stay(EARLY, SEEN), departure_seen="hour-end")
