"""Plan sections 1 and 6 (T6): the affine policy at a scale beta, and each row's largest beta.

1. Hand case: one car, rate 2 kW, departure 3, room 5 kWh, rho 0, one service hour; policy
   `g = (0.5, 1)`, gain 0.5 in hour 1. Thresholds: `u_1 <= 2` gives (2 - 1) / 0.5 = 2; `-u_1 <= 0`
   gives 1 / 0.5 = 2; the hour-0 rows do not depend on the request (+inf); the stored-energy row
   after hour 1 gives (5 - 1.5) / 0.5 = 7; the need row (5 kWh, 2 of them from the recovery hour)
   gives (-3 + 1.5) / 0.5 = -3, a row that fails at the centre.
2. On the conservative certified days of `test_certificate_check.py`, on the true rows: every
   threshold is at least 1 (the certified box at scale 1 is safe); at each finite threshold
   `beta` the worst case equals the bound, and at `beta * (1 + STEP)` some corner path breaks it;
   at every scale the fleet's powers meet the scaled request `m + beta w zeta`.
3. A policy whose decision hour depends on the request is refused.
"""

import numpy as np
import pytest
from test_certificate_check import _cases, _corner_values

from fleetreserve.carsets import stack, true_one_way_rows
from fleetreserve.day import TrueCar
from fleetreserve.policy import AffinePolicy
from fleetreserve.products import corner_paths

STEP = 1e-3
TOLERANCE = 1e-7
MIN_FINITE = 100  # finite thresholds checked


def test_hand_case_thresholds():
    rows = stack([true_one_way_rows(TrueCar(3, 2.0, 5.0), 1, 0.0)])
    policy = AffinePolicy([1.0], [0.5], [[0.5, 1.0]], [[[0.0], [0.5]]])
    by_row = dict(
        zip(
            [(k, tuple(a)) for k, a in zip(rows.kinds, rows.coefficients, strict=True)],
            policy.row_thresholds(rows),
            strict=True,
        )
    )
    assert by_row["power-high", (0.0, 1.0)] == pytest.approx(2.0)
    assert by_row["power-low", (0.0, -1.0)] == pytest.approx(2.0)
    assert by_row["power-high", (1.0, 0.0)] == np.inf
    assert by_row["power-low", (-1.0, 0.0)] == np.inf
    assert by_row["stored-high", (1.0, 1.0)] == pytest.approx(7.0)
    assert by_row["need", (-1.0, -1.0)] == pytest.approx(-3.0)
    assert list(rows.last_hours) == [0, 1, 0, 1, 0, 1, 1]
    assert policy.worst_case(rows, beta=2.0)[1] == pytest.approx(2.0)


def test_thresholds_on_certified_policies():
    cases, _ = _cases()
    finite = 0
    for _, _, truth, result in cases["conservative"]:
        policy = result.policy
        thresholds = policy.row_thresholds(truth)
        assert thresholds.min() >= 1.0 - 1e-6
        paths = corner_paths(policy.n_service)
        for k in np.flatnonzero(np.isfinite(thresholds) & (thresholds > 0)):
            beta = thresholds[k]
            assert policy.worst_case(truth, beta)[k] == pytest.approx(truth.bounds[k], abs=1e-6)
            above = policy.powers(paths, beta * (1 + STEP))[:, truth.car[k], :]
            assert (above @ truth.coefficients[k]).max() > truth.bounds[k] + TOLERANCE
            finite += 1
        for beta in (0.5, 1.0):
            delivered = policy.powers(paths, beta).sum(axis=1)[:, 1:]
            np.testing.assert_allclose(delivered, policy.request(paths, beta), atol=TOLERANCE)
        assert (truth.bounds - _corner_values(policy, truth)).min() >= -TOLERANCE
    print(f"{finite} finite thresholds checked")
    assert finite >= MIN_FINITE


def test_the_decision_hour_cannot_depend_on_the_request():
    with pytest.raises(ValueError):
        AffinePolicy([1.0], [0.5], [[0.5, 1.0]], [[[0.1], [0.5]]])
